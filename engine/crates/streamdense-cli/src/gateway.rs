//! streamdense gateway --model DIR --port 8898 --max-ctx 8192 [--mtp | --no-mtp] [--spec-depth N|auto] [--spec-cost ..] [--draft-table DIR] [--gemv mma|old] [--draft-vocab FILE] [--no-mtp-decode] [--think] [--log FILE]
//!                     [--tokenizer DIR] [--cap 17] [--reserve GB] [--kv bf16|q8 ([gateway] kv)] [--seqs 1..4] [--abort-gb 16] [--swap-gb 7]
//!                     [--prefill-k N] [--prefill-t N] [--prefill-exact] [--threads N] [--host 127.0.0.1] [--name qwen38-27b]
//!                     [--no-snapshots] [--snapshot-dir DIR] [--snapshot-max-files N] [--snapshot-max-gb GB] ([snapshots], default 32 / 8) [--no-activity]
//!                     [--watcher-dir DIR (or $SD_WATCHER_DIR)] [--forheader off|12gb] [--no-fast] [--no-fused] [--ctx-draft | --no-ctx-draft | --ctx-draft-dry] [--plan (print the budget / 12 GB K and exit; loads nothing)]
//!                     [--verify-fused | --no-verify-fused] [--history | --no-history] [--history-dir DIR] [--history-project DIR]
//!                     [--drafter DIR | --no-drafter] [--dflash2 DIR | --no-dflash2] [--fold-ids (tests only)]
//! streamdense gateway --print-env [--port N]   (shell exports for Claude Code; loads nothing)
//! Anthropic Messages API on 127.0.0.1 served by the in-process engine (streamdense-gateway, docs/GATEWAY.md).
//! Defaults for --model / --port / --max-ctx / --mtp / --spec-depth / --gemv / --draft-vocab / --draft-table ([gateway]) and --abort-gb (= cap_gb) / --swap-gb ([ram]) come from
//! streamdense.toml (flag > config > built-in). Launch: `streamdense up` (= sd-watchdog + nice + gateway from the config).
//! stdout: one {"ready":..} JSON line once listening; request log: JSON lines on stderr or --log FILE.
//! 12 GB mode (research; [forheader] mode = "12gb" or --forheader 12gb): compact backend (streamdense-gateway/src/compact.rs), resident
//! set sized from [forheader] cap_gb by formula, global score file [forheader] t1; recorder and answer library off; snapshots on (format 2). Fast prompt read + fused decode on by default ([forheader] fast / fused,
//! --no-fast / --no-fused). Context drafts ([forheader] ctx_draft, default off; --ctx-draft / --no-ctx-draft): drafts from the context +
//! the [gateway] draft_table, checked in one pass (engine ctx_draft.rs); --ctx-draft-dry keeps its RAM (same K) but decodes plainly (A/B).
//! Fused verify matmuls ([gateway] / [forheader] verify_fused, default off; --verify-fused / --no-verify-fused): bit-identical,
//! faster verify passes of 3..8 rows. History index ([history], default off; --history / --no-history, --history-dir,
//! --history-project): a third draft source of the 12 GB context drafts (streamdense-core history_index.rs); the
//! updater thread (background QoS) re-scans the project dir at start, gets every finished reply, pauses during feeds.
//! DFlash drafter ([gateway] drafter, default "" = off; --drafter DIR / --no-drafter; full mode only): block drafts from
//! the trained drafter (engine dflash.rs), one exact verify pass per block; replaces MTP (not loaded) and turns snapshots off.
//! DFlash2 ([gateway] dflash2, default "" = off; --dflash2 DIR / --no-dflash2; experimental): refused with --drafter or the
//! 12 GB mode, turns MTP and snapshots off, refused unless the whole budget fits --abort-gb and the weights match
//! [dflash2] expected_sha256; a drafting failure ends the request with an error (no automatic fallback).
use crate::{conditions, config_root, load_config, watcher, Args};
use serde_json::json;
use std::sync::Arc;
use streamdense_model::activity::script_class;
use streamdense_core::{alloc, footprint, gb};
use streamdense_gateway::{GatewayConfig, WorkerConfig};
use streamdense_model::qwen_engine::{KvFormat, QwenEngine, QwenOptions};
use streamdense_model::qwen_layers::GemvImpl;
use streamdense_model::ffn_set::{SetRule, T1};
use streamdense_model::snapshot::SnapshotCache;
use streamdense_tokenizer::Tokenizer;

fn die(e: String) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }

const DFLASH2_NOTE: &str = "DFlash2 is an experimental option: not yet proven identical to plain decoding on the real 27B in this engine";

/// sha256 of a file through the system tool (a small separate process; no new dependency, the weights are never loaded here).
fn sha256_file(p: &std::path::Path) -> Result<String, String> {
    let o = std::process::Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(p).output().map_err(|e| format!("/usr/bin/shasum: {}", e))?;
    if !o.status.success() { return Err(format!("/usr/bin/shasum {}: {}", p.display(), String::from_utf8_lossy(&o.stderr).trim())); }
    let s = String::from_utf8_lossy(&o.stdout);
    s.split_whitespace().next().map(str::to_ascii_lowercase).ok_or_else(|| format!("/usr/bin/shasum {}: no output", p.display()))
}

/// --dflash2: the watcher's routes, plus the live DFlash2 stats in /health and the request log. `Extension` has one
/// log field, so while the option is on the watcher summary moves under `dflash2.watcher`.
struct Df2Ext { inner: Arc<dyn streamdense_gateway::Extension> }
impl streamdense_gateway::Extension for Df2Ext {
    fn route(&self, method: &str, path: &str, body: &[u8]) -> Option<(u16, serde_json::Value)> { self.inner.route(method, path, body) }
    fn log_field(&self) -> Option<(&'static str, serde_json::Value)> {
        let mut v = json!({"state": "on", "note": DFLASH2_NOTE, "stats": streamdense_gateway::worker::dflash2_stats_json()});
        if let Some((k, f)) = self.inner.log_field() { v[k] = f; }
        Some(("dflash2", v))
    }
}

/// Env var read by macOS libmalloc at process start: freed memory goes back to the OS (no large-block cache, freed
/// small-region pages madvised). Without it, every block freed after setup (tokenizer JSON parse ~42 MB, migration
/// and snapshot copies) stays dirty in the process and counts in phys_footprint, the number sd-watchdog kills on:
/// ~119 MB of freed-but-charged heap measured on the 27B gateway (vmmap).
const SPACE_EFFICIENT: &str = "MallocSpaceEfficient";

/// Re-exec this process (same pid, args, stdio) with `SPACE_EFFICIENT` set, once: libmalloc reads it only at start.
/// Set it yourself (any value) to skip this. Returns only if the exec failed (the gateway then runs as before).
fn space_efficient_malloc() {
    use std::os::unix::process::CommandExt;
    if std::env::var_os(SPACE_EFFICIENT).is_some() { return; }
    let Ok(exe) = std::env::current_exe() else { return };
    let e = std::process::Command::new(exe).args(std::env::args_os().skip(1)).env(SPACE_EFFICIENT, "1").exec();
    eprintln!("WARNING: re-exec with {}=1 failed ({}): freed heap stays in the footprint", SPACE_EFFICIENT, e);
}

pub fn main(a: &Args) {
    let host = a.sd("--host", "127.0.0.1");
    let conf = load_config();
    let port: u16 = a.s("--port").map(|p| p.parse().unwrap_or_else(|_| die(format!("--port {}", p)))).unwrap_or(conf.cfg.gateway.port);
    let name = a.sd("--name", streamdense_gateway::MODEL_NAME);
    if a.has("--print-env") {
        print!("{}", streamdense_gateway::print_env(&host, port, &name));
        return;
    }
    space_efficient_malloc();
    eprintln!("malloc: {}={} (freed heap returned to the OS)", SPACE_EFFICIENT, std::env::var(SPACE_EFFICIENT).unwrap_or_default());
    let root = crate::ROOT;
    let model = a.s("--model").unwrap_or_else(|| config_root(&conf).join(&conf.cfg.gateway.model).display().to_string());
    eprintln!("config: {}", conf.source_str());
    let mut opt = QwenOptions::default();
    opt.cap = (a.f("--cap", 17.0) * 1e9) as u64;
    if let Some(r) = a.s("--reserve") { opt.reserve = (r.parse::<f64>().unwrap() * 1e9) as u64; }
    opt.max_ctx = a.f("--max-ctx", conf.cfg.gateway.max_ctx as f64) as usize;
    opt.kv = match a.sd("--kv", &conf.cfg.gateway.kv).as_str() { "bf16" => KvFormat::Bf16, "q8" => KvFormat::Q8, k => die(format!("--kv {}: want bf16|q8", k)) };
    opt.mtp = if a.has("--no-mtp") { false } else { a.has("--mtp") || conf.cfg.gateway.mtp };
    // 12 GB mode (research): flag --forheader off|12gb > [forheader] mode; off by default
    let mut fhc = conf.cfg.forheader.clone();
    // 12 GB mode speed-ups (default on, 2026-11-03): --no-fast / --no-fused > [forheader] fast / fused
    if a.has("--no-fast") { fhc.fast = false; }
    if a.has("--no-fused") { fhc.fused = false; }
    // context drafts: --ctx-draft / --no-ctx-draft > [forheader] ctx_draft; --ctx-draft-dry = its RAM + plain decode
    if a.has("--ctx-draft") { fhc.ctx_draft = true; }
    if a.has("--no-ctx-draft") { fhc.ctx_draft = false; }
    let ctx_dry = a.has("--ctx-draft-dry");
    if ctx_dry { fhc.ctx_draft = true; }
    let fh_on = match a.s("--forheader").unwrap_or_else(|| fhc.mode.clone()).as_str() { "off" => false, "12gb" => true, m => die(format!("--forheader {}: want off|12gb", m)) };
    if fh_on {
        // MTP comes from [forheader] mtp (it costs resident neurons); [gateway] mtp / --mtp do not apply, --no-mtp still does
        let m = fhc.mtp && !a.has("--no-mtp");
        if opt.mtp != m { eprintln!("12 GB mode: MTP {} ([forheader] mtp = {}; [gateway] mtp / --mtp do not apply here)", if m { "on" } else { "off" }, fhc.mtp); }
        opt.mtp = m;
    }
    // DFlash drafter: --drafter DIR / --no-drafter > [gateway] drafter ("" = off); full mode only, replaces MTP
    opt.dflash = if a.has("--no-drafter") { None } else {
        a.s("--drafter").filter(|s| !s.is_empty()).map(std::path::PathBuf::from)
            .or_else(|| (!conf.cfg.gateway.drafter.is_empty()).then(|| config_root(&conf).join(&conf.cfg.gateway.drafter)))
    };
    if fh_on && opt.dflash.is_some() { eprintln!("drafter ignored: the 12 GB mode has its own decode loop (full mode only)"); opt.dflash = None; }
    if let Some(p) = &opt.dflash {
        if opt.mtp { eprintln!("drafter {}: MTP off (the DFlash drafter replaces the MTP drafts)", p.display()); }
        opt.mtp = false;
    }
    // DFlash2 (experimental): --dflash2 DIR / --no-dflash2 > [gateway] dflash2 ("" = off)
    opt.dflash2 = if a.has("--no-dflash2") { None } else {
        a.s("--dflash2").filter(|s| !s.is_empty()).map(std::path::PathBuf::from)
            .or_else(|| (!conf.cfg.gateway.dflash2.is_empty()).then(|| config_root(&conf).join(&conf.cfg.gateway.dflash2)))
    };
    if let Some(p) = &opt.dflash2 {
        if let Some(d) = &opt.dflash { die(format!("DFlash2 refused: --dflash2 and --drafter cannot be combined (drafter {})", d.display())); }
        if fh_on { die("DFlash2 refused: --dflash2 cannot be combined with the 12 GB mode (--forheader 12gb)".into()); }
        eprintln!("{}", DFLASH2_NOTE);
        if opt.mtp { eprintln!("dflash2 {}: MTP off for this run (DFlash2 replaces the MTP drafts)", p.display()); }
        opt.mtp = false;
    }
    opt.activity = !a.has("--no-activity") && !fh_on; // activity recorder, on by default (the compact FFN refuses it)
    if fh_on { eprintln!("12 GB mode: activity recorder off (the compact FFN refuses it): the watcher runs on text features only; answer library off"); }
    opt.activity_batch = opt.activity; // same-label jobs summed in model memory before one disk fold
    // read mask (dry / on): the engine's read-mask region (2.9 MB model memory on the 27B) so it records prompt_active; no
    // profile is ever set here (set_read_mask stays None = bit-identical) until mask = "on" is approved and wired
    opt.read_mask = opt.activity && conf.cfg.watcher.mask != "off";
    // chained-draft depth: flag > [gateway] spec_depth; only with MTP (depth 1 = draft 1 / verify 2)
    // + "auto" (adaptive) and the draft table: flag > [gateway] draft_table ("" = off)
    crate::qwen::spec_opts(a, &mut opt, Some(conf.cfg.gateway.spec_depth.arg()), Some(conf.cfg.gateway.draft_table.clone()));
    if !opt.mtp && (opt.spec_depth > 1 || opt.spec_auto) {
        eprintln!("spec_depth {} ignored: MTP is off", if opt.spec_auto { "auto".to_string() } else { opt.spec_depth.to_string() });
        opt.spec_depth = 1;
        opt.spec_auto = false;
    }
    // the draft table also feeds the 12 GB mode's context drafts
    let fh_ctx = fh_on && fhc.ctx_draft && !opt.mtp;
    // fused verify matmuls: flag > [forheader] verify_fused (12 GB mode) / [gateway] verify_fused
    opt.verify_fused = if fh_on { fhc.verify_fused } else { conf.cfg.gateway.verify_fused };
    if a.has("--verify-fused") { opt.verify_fused = true; }
    if a.has("--no-verify-fused") { opt.verify_fused = false; }
    // history index: flags > [history]; used by the 12 GB context drafts only (full-mode MTP does not take it yet)
    let mut hc = conf.cfg.history.clone();
    if a.has("--history") { hc.enabled = true; }
    if a.has("--no-history") { hc.enabled = false; }
    if let Some(d) = a.s("--history-dir") { hc.dir = d; }
    if let Some(d) = a.s("--history-project") { hc.project = d; }
    // --ctx-draft-dry keeps its RAM in the plan (same K, A/B) but opens nothing
    let hist_plan = hc.enabled && fh_ctx;
    let hist_on = hist_plan && !ctx_dry;
    if hc.enabled && !hist_plan { eprintln!("history index: off (used only by the 12 GB mode's context drafts: --forheader 12gb --ctx-draft)"); }
    if hist_plan && ctx_dry { eprintln!("history index: dry (its RAM is in the plan, nothing opened)"); }
    let hist_cfg = hist_plan.then(|| {
        let mut c = streamdense_core::history_index::Config::with_dir(&watcher::expand_home(&hc.dir));
        c.budget_bytes = (hc.budget_mb * 1048576.0) as usize;
        c
    });
    // its RAM joins the 12 GB plan (keep formula) like the draft table's
    if let Some(c) = &hist_cfg { opt.history_bytes = c.budget_bytes; }
    if !opt.spec_auto && !fh_ctx && opt.draft_table.is_some() { eprintln!("draft_table ignored: only used with spec_depth = auto or 12 GB context drafts"); opt.draft_table = None; }
    // batched decode (model batch.rs): up to N requests decode together, each in its own engine slot (KV + MTP KV +
    // DeltaNet state per slot); needs MTP with spec depth >= 2 or auto; refused with the drafter / 12 GB mode / dumps
    opt.seqs = a.s("--seqs").map(|s| s.parse().unwrap_or_else(|_| die(format!("--seqs {}: want 1..=4", s)))).unwrap_or(1);
    if opt.seqs > 1 && (!opt.mtp || (opt.spec_depth < 2 && !opt.spec_auto)) {
        eprintln!("--seqs {} ignored: batching needs MTP with --spec-depth >= 2 or auto", opt.seqs);
        opt.seqs = 1;
    }
    // matmul family of the exact passes: flag > [gateway] gemv (mma | old)
    opt.gemv = GemvImpl::parse(&a.s("--gemv").unwrap_or_else(|| conf.cfg.gateway.gemv.clone())).unwrap_or_else(|e| die(e));
    if let Some(k) = a.s("--prefill-k") { opt.prefill_k = k.parse().unwrap(); }
    if let Some(t) = a.s("--prefill-t") { opt.prefill_t = t.parse().unwrap(); }
    opt.prefill_exact = a.has("--prefill-exact") || fh_on; // the compact FFN needs the exact prefill
    if let Some(t) = a.s("--threads") { opt.load_threads = t.parse().unwrap(); }
    // pruned MTP draft vocab: flag > [gateway] draft_vocab ("" = off; relative to the repo root); only with MTP
    opt.draft_vocab = a.s("--draft-vocab").map(std::path::PathBuf::from).or_else(|| {
        (!conf.cfg.gateway.draft_vocab.is_empty()).then(|| config_root(&conf).join(&conf.cfg.gateway.draft_vocab))
    });
    if !opt.mtp && opt.draft_vocab.is_some() { eprintln!("draft_vocab ignored: MTP is off"); opt.draft_vocab = None; }
    // a keep-list made for another model (or unreadable) is a warning + off, not a crash: drafts use the full vocab, output unchanged
    if let Some(p) = opt.draft_vocab.clone() {
        if let Err(e) = QwenEngine::check_draft_vocab(&model, &p) {
            eprintln!("WARNING: draft_vocab off: {} (drafts use the full vocab; output unchanged)", e);
            opt.draft_vocab = None;
        }
    }
    // the cap shown in the budget = the one enforced live (sd-watchdog + --abort-gb); --cap stays the startup estimate limit
    opt.live_cap = (a.f("--abort-gb", conf.cfg.ram.cap_gb) * 1e9) as u64;

    // 12 GB mode: streamed prompt read + the resident set size from the cap by formula, the global score file, the
    // initial set (global top K per layer)
    let fh = fh_on.then(|| forheader_setup(&model, &mut opt, &fhc));

    // DFlash2 file check: the weights' sha256 before anything else reads them (config + tensor shapes: Df2Plan::read in the budget)
    let df2_sha = opt.dflash2.as_ref().map(|d| {
        let f = d.join("model.safetensors");
        let want = conf.cfg.dflash2.expected_sha256.to_ascii_lowercase();
        eprintln!("dflash2: checking the sha256 of {}", f.display());
        let got = sha256_file(&f).unwrap_or_else(|e| die(format!("DFlash2 refused: cannot check the weights: {}", e)));
        if got != want { die(format!("DFlash2 refused: {} sha256 mismatch: expected {} ([dflash2] expected_sha256), file has {}", f.display(), want, got)); }
        got
    });
    let bud = QwenEngine::budget(&model, &opt).unwrap_or_else(|e| die(e));
    eprintln!("budget ({}):\n{}", model, bud.report());
    // DFlash2: the whole budget must fit the set cap; the cap is never raised and there is no automatic fallback
    if opt.dflash2.is_some() {
        let need = bud.arena_bytes() + bud.reserve;
        if need > opt.live_cap {
            die(format!("DFlash2 refused: the budget needs {:.3} GB (arena {:.3} + reserve {:.3}) but the cap is {:.3} GB (--abort-gb / [ram] cap_gb). The cap is not raised and nothing falls back; start without --dflash2 or set a higher cap yourself",
                        gb(need), gb(bud.arena_bytes()), gb(bud.reserve), gb(opt.live_cap)));
        }
    }
    if let Err(e) = bud.check() { die(e); }
    if a.has("--plan") { eprintln!("--plan: budget only, nothing loaded"); return; }

    // tokenizer before the engine (keeps the heap limit meaningful): --tokenizer DIR, else the model dir, else the 27B dir
    let tdir = a.s("--tokenizer").unwrap_or_else(|| {
        if std::path::Path::new(&model).join("tokenizer.json").exists() { model.clone() } else { format!("{}/{}", root, crate::qwen::DEFAULT_MODEL) }
    });
    let t0 = std::time::Instant::now();
    let tk = Tokenizer::from_path(&tdir).unwrap_or_else(|e| die(format!("tokenizer {}: {}", tdir, e)));
    eprintln!("tokenizer: {} ({} ids) in {:.0} ms", tdir, tk.vocab_size(), t0.elapsed().as_secs_f64() * 1e3);
    // history index: opened (table allocated once) before the engine, so the heap limit set after the open covers it
    let hist_idx = hist_cfg.filter(|_| hist_on).map(|c| {
        let d = c.dir.clone();
        let (idx, note) = streamdense_core::history_index::HistoryIndex::open(c).unwrap_or_else(|e| die(format!("history index {}: {}", d.display(), e)));
        let st = idx.stats();
        eprintln!("history index: {} ({:?}) | {:.1} MB RAM (budget {:.1} MB) | {} tokens in {} docs, {} of {} slots used",
                  d.display(), note, st.ram_bytes as f64 / 1048576.0, st.budget_bytes as f64 / 1048576.0, st.tokens, st.docs, st.used_slots, st.slots);
        Arc::new(idx)
    });

    let abort = (a.f("--abort-gb", conf.cfg.ram.cap_gb) * 1e9) as u64;
    let mon = footprint::Monitor::start(abort, (a.f("--swap-gb", conf.cfg.ram.swap_gb) * 1e9) as u64, 250);
    eprintln!("conditions:\n{}", conditions());
    let t_open = std::time::Instant::now();
    let eng = QwenEngine::open(&model, opt.clone()).unwrap_or_else(|e| die(e));
    let open_s = t_open.elapsed().as_secs_f64();
    eprintln!("opened in {:.2} s | footprint {:.3} GB | heap {:.1} MB", open_s, gb(footprint::self_footprint()), alloc::heap_now() as f64 / 1e6);
    let fold = a.has("--fold-ids");
    if tk.vocab_size() > eng.cfg.vocab && !fold {
        die(format!("tokenizer has {} ids but the model vocab is {} (wrong --tokenizer?)", tk.vocab_size(), eng.cfg.vocab));
    }
    alloc::set_limit_extra(256 << 20); // request bodies + prompts fit easily; the decode loop may not grow the heap past this

    let wc = WorkerConfig { think: a.has("--think"), mtp: !a.has("--no-mtp-decode"), fold: if fold { Some(eng.cfg.vocab as u32) } else { None } };
    // 12 GB mode: the compact backend (status: K, planned / resident GB, stream settings)
    let fh_info = match &fh {
        Some(f) => json!({"mode": "12gb", "k": f.rule.k, "kres": f.rule.kres, "inter": eng.cfg.inter, "share": f.rule.k as f64 / eng.cfg.inter as f64,
            "cap_gb": fhc.cap_gb, "planned_gb": gb(f.plan_total), "resident_gb": gb(footprint::self_footprint()), "mtp": opt.mtp, "slots": opt.ffn_stream_slots,
            "host": opt.ffn_stream_host, "stream_tokens": opt.ffn_stream_tokens, "fast": opt.ffn_stream_fast, "fused": opt.decode_fused, "ctx_draft": opt.ctx_draft, "ctx_draft_dry": ctx_dry, "t1": f.t1_path.display().to_string()}),
        None => json!({"mode": "off"}),
    };
    let snapshots = !a.has("--no-snapshots") && opt.dflash.is_none() && opt.dflash2.is_none();
    if opt.dflash.is_some() && !a.has("--no-snapshots") { eprintln!("drafter on: snapshots off (snapshot files hold no drafter context)"); }
    if opt.dflash2.is_some() && !a.has("--no-snapshots") { eprintln!("dflash2 on: snapshots off (streamed prefill and snapshot restore leave no context rows)"); }
    let info = json!({"pid": std::process::id(), "model_dir": model, "max_ctx": opt.max_ctx, "kv": if opt.kv == KvFormat::Q8 { "q8" } else { "bf16" }, "mtp_available": opt.mtp,
        "mtp_decode": wc.mtp && opt.mtp, "spec_depth": opt.spec_depth, "spec_auto": opt.spec_auto, "seqs": opt.seqs, "gemv": opt.gemv.name(), "verify_fused": opt.verify_fused, "history": hist_on,
        "draft_table": opt.draft_table.as_ref().map(|p| p.display().to_string()),
        "drafter": opt.dflash.as_ref().map(|p| p.display().to_string()),
        "dflash2": match &opt.dflash2 {
            Some(p) => json!({"state": "on", "dir": p.display().to_string(), "sha256": df2_sha, "drafting": wc.mtp, "experimental": true, "note": DFLASH2_NOTE}),
            None => json!({"state": "off"}),
        },
        "draft_vocab": opt.draft_vocab.as_ref().map(|p| p.display().to_string()), "think": wc.think, "open_s": open_s, "abort_gb": gb(abort), "snapshots": snapshots,
        "forheader": fh_info});
    let mut cb = fh.map(|f| {
        let mut c = streamdense_gateway::compact::CompactBackend::new(&eng, f.rule, fh_info.clone()).unwrap_or_else(|e| die(e));
        c.peak = Some(mon.peak.clone());
        c.ctx_dry = ctx_dry;
        c
    });

    if let Some(c) = &cb {
        eprintln!("12 GB mode (research): K = {} of {} per layer ({:.4}), reserve {} | resident {:.3} GB after open | {} stream slot(s){}, <= {} tokens per streamed pass{}",
                  c.rule.k, eng.cfg.inter, fh_info["share"].as_f64().unwrap_or(0.0), c.rule.kres, gb(footprint::self_footprint()), opt.ffn_stream_slots,
                  if opt.ffn_stream_host { " inside the idle lm_head" } else { "" }, c.piece, if opt.mtp { " | MTP on" } else { "" });
    }
    let cfg = GatewayConfig { host: host.clone(), port, model_name: name.clone(), worker: wc, log: a.s("--log"), info: info.clone() };
    // snapshot cache (both modes; 12 GB mode: format-2 files with the prompt sums, under their own keys)
    let hook: Option<streamdense_gateway::CacheHook> = if !snapshots { None } else {
        let mid = streamdense_gateway::CacheHook::model_id(&model);
        let mut cache = match a.s("--snapshot-dir") {
            Some(d) => SnapshotCache::open(d.into(), &mid),
            None => SnapshotCache::open_default(&mid),
        }.unwrap_or_else(|e| die(format!("snapshot cache: {}", e)));
        if let Some(c) = &cb { cache.mode = c.snap_mode(); }
        // tiers (snap_tier.rs): [snapshots] (flags win for the caps); hot RAM copies only under the live cap, none in 12 GB mode
        let sc = &conf.cfg.snapshots;
        let mut tc = streamdense_gateway::snap_tier::TierConfig::from_config(sc, abort);
        tc.max_files = (a.f("--snapshot-max-files", sc.max_files as f64) as usize).max(1);
        tc.max_bytes = (a.f("--snapshot-max-gb", sc.max_disk_gb) * 1e9) as u64;
        if cb.is_some() { tc.hot_bytes = 0; }
        let h = streamdense_gateway::CacheHook::with_tiers(cache, tc);
        let t = &h.tiers.cfg;
        eprintln!("snapshot tiers: hot <= {:.2} GB (headroom {:.2} under cap {:.2}) | warm cap {} files / {:.2} GB, free disk >= {:.1} GB | idle work after {:.1} s: save {}, pre-read {} ({} files / {:.2} GB) | hot shared prefix {} | switch save {} (>= {} tokens)",
                  gb(t.hot_bytes), gb(t.hot_headroom), gb(t.ram_cap), t.max_files, gb(t.max_bytes), gb(t.min_free), t.idle.map_or(0.0, |d| d.as_secs_f64()),
                  t.idle_save, t.preread, t.preread_files, gb(t.preread_bytes), t.hot_stable, t.switch_save, t.switch_save_min);
        let (n, b) = h.usage();
        eprintln!("snapshots: {} ({} files, {:.1} MB; cap {} files / {:.2} GB){}", h.cache.dir.display(), n, b as f64 / 1e6, h.max_files, gb(h.max_bytes),
                  if cb.is_some() { format!(" | 12 GB mode files ({}), + {:.1} MB prompt sums each", h.cache.mode, (eng.cfg.layers * 2 * eng.cfg.inter * 4) as f64 / 1e6) } else { String::new() });
        Some(h)
    };
    // domain watcher: always on ([watcher], no off switch; --watcher-dir DIR > $SD_WATCHER_DIR > [watcher] dir)
    let wcf = &conf.cfg.watcher;
    let wdir = a.s("--watcher-dir").map(std::path::PathBuf::from).unwrap_or_else(|| watcher::config_dir(&wcf.dir));
    // dims + q4 layout + resident bytes from the activity recorder (no recorder: config dims, footprint-based estimates)
    // (the template: no snapshot, so no down norms kept in RAM and no copy of the sums; the norms live in the store)
    let mut dims = watcher::Dims::from_stats(&eng.activity_template(), wcf.granularity);
    if eng.activity_tap().is_some() { dims.vocab = eng.cfg.vocab; }
    // the neuron store / predictor files exist only with the recorder (nothing else writes them): none in 12 GB mode
    // or with --no-activity (the text watcher runs without them)
    let w = watcher::Watcher::open_opts(wdir, watcher::WCfg::from_config(wcf), Some(dims.clone()), eng.activity_tap().is_some()).unwrap_or_else(|e| die(e));
    eprintln!("watcher: {} | mode {} | {} | activity: {}", w.dir.display(), wcf.mode, w.summary(),
              if eng.activity_tap().is_some() { "activity recorder" } else if fh_on { "off (12 GB mode): text features only" } else { "off (--no-activity): text features only" });
    if wcf.mode == "unload" { eprintln!("watcher: mode \"unload\" is not implemented yet: observing"); }
    if wcf.mask != "off" {
        eprintln!("watcher: mask {} (CLI side): profile + predictor score + exploration schedule; {}", wcf.mask,
                  if wcf.mask == "on" { "\"on\" acts as dry until the engine mask is wired and approved" } else { "nothing masked" });
    }
    let w_ready = json!({"dir": w.dir.display().to_string(), "mode": wcf.mode, "state": w.summary(),
        "activity": if eng.activity_tap().is_some() { "activity recorder" } else { "text features only" }});
    let ws = Arc::new(watcher::Shared::new(w, footprint::self_footprint));
    // heat on disk: the down-column norms into the store (once; layer 0 re-checked at every start), the job source
    // on the recorder (in place), a read-only store for the sd-heat thread (summary + audits at background QoS) and the
    // script class per token id from the tokenizer on the fly (no table).
    let tap = eng.activity_tap();
    let tk = Arc::new(tk);
    // history index updater (background QoS): project / sessions re-scan (changed files only), finished replies
    if let (Some(idx), Some(c)) = (hist_idx, cb.as_mut()) {
        let tk3 = tk.clone();
        let enc: streamdense_core::history_index::Encoder = Arc::new(move |s: &str| tk3.encode(s));
        let up = streamdense_core::history_index::Updater::spawn(idx.clone(), enc, Default::default()).unwrap_or_else(|e| die(format!("history updater: {}", e)));
        for (what, p) in [("project", &hc.project), ("sessions", &hc.sessions)] {
            if p.is_empty() { continue; }
            let pb = watcher::expand_home(p);
            if !pb.exists() { eprintln!("WARNING: history {} {} does not exist (no scan)", what, pb.display()); continue; }
            if what == "project" { up.scan_dir(pb.clone()); } else { up.scan_sessions(pb.clone()); }
            eprintln!("history index: {} scan of {} queued (background)", what, pb.display());
        }
        c.set_history(streamdense_gateway::compact::HistoryHook { index: idx, updater: up });
    }
    let (wdir2, src, ro) = {
        let mut w = ws.w.lock().unwrap();
        let wdir2 = w.dir.clone();
        let mut src = None;
        let mut ro = None;
        if let (Some(tp), Some(st)) = (&tap, w.store.as_mut()) {
            let t0 = std::time::Instant::now();
            match st.ensure_down(tp, 4) {
                Ok(new) => eprintln!("store: down norms {} in {:.2} s", if new { "computed" } else { "checked" }, t0.elapsed().as_secs_f64()),
                Err(e) => eprintln!("WARNING: store: down norms: {} (heat FFN = |a| only)", e),
            }
            let s = crate::store::TapSrc::new(tp.clone(), st);
            src = Some(s);
            ro = crate::store::Store::open(&wdir2, st.d.clone(), watcher::label_names(), watcher::MIXED, true).ok().map(|mut r| { r.prog = st.prog.clone(); r });
            eprintln!("store: {} ({:.2} MB on disk, cap {:.0} MB)", st.path().display(), st.disk_bytes() as f64 / 1e6, st.cap as f64 / 1e6);
        }
        if let (Some(s), Some(tp)) = (&src, &tap) { if tp.has_batch() { w.batcher = Some(Arc::new(crate::store::TapBatch(s.batch_view()))); } }
        (wdir2, src, ro)
    };
    // the heat files of the in-RAM era (SDWHEAT2 maps): derived data, replaced by the store
    if let Ok(rd) = std::fs::read_dir(crate::heat::heat_dir(&wdir2)) {
        for p in rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "bin")) {
            let mut m = [0u8; 7]; // the magic only (the map itself is MBs)
            if std::fs::File::open(&p).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut m)).is_ok() && &m == b"SDWHEAT" { let _ = std::fs::remove_file(&p); eprintln!("heat: removed old in-RAM-era map {}", p.display()); }
        }
    }
    let tk2 = tk.clone();
    let script: Option<Box<dyn Fn(u32) -> u8 + Send>> = Some(Box::new(move |id| script_class(tk2.id_to_bytes(id), tk2.is_special(id))));
    let hc = crate::heat::HeatCfg { sample_ms: wcf.sample_ms, keep: wcf.keep_frac, dir: wdir2 };
    crate::heat::spawn(tap.clone(), ro, script, ws.clone(), hc).unwrap_or_else(|e| die(format!("heat thread: {}", e)));
    eprintln!("heat: sd-heat thread, every {} ms at background QoS; heat on disk (host heap 0)", wcf.sample_ms);
    let has_src = src.is_some();
    // answer library (`[watcher] library = true`, off by default): ring file under <watcher dir>/library, prompt mark on
    // the recorder (worker: one memcpy at the first token), capture + encode + write on sd-heat
    let lib_text = wcf.library_text;
    if wcf.library && fh_on { eprintln!("library: off in 12 GB mode (needs the activity recorder, which the compact FFN refuses)"); }
    if wcf.library && !fh_on {
        match (&tap, &src) {
            (Some(tp), Some(s)) => {
                let (dfile, base, inter) = (s.down.as_ref().and_then(|f| f.try_clone().ok()), s.down_base, dims.inter);
                let down: Box<dyn FnMut(usize, &mut [f32]) + Send> = Box::new(move |l, out| crate::store::Store::read_down(dfile.as_ref(), base, l * inter, out));
                let cap = (wcf.library_cap_gb * 1e9) as u64;
                match crate::library::Collector::open(&wdir2_lib(&ws), cap, lib_text, tp.clone(), down) {
                    Ok(c) => {
                        eprintln!("library: on: {} | {} | {:.2} MB per turn, {} turns fit; text {} | heap {:.1} MB", c.ring.path.display(), c.line(),
                                  c.ring.d.slot_bytes() as f64 / 1e6, c.ring.slots, if lib_text { "STORED (library_text = true)" } else { "not stored" }, c.heap_bytes() as f64 / 1e6);
                        let _ = ws.lib.set(std::sync::Mutex::new(c));
                    }
                    Err(e) => eprintln!("WARNING: library off: {}", e),
                }
            }
            _ => eprintln!("WARNING: library off: needs the activity recorder (--no-activity) and the heat store"),
        }
    }
    let lib_on = ws.lib.get().is_some();
    if let Some(s) = src { let _ = ws.src.set(s); }
    crate::heat::install_shutdown();
    let ws2 = ws.clone();
    let mask_rec = opt.read_mask;
    let job = move |eng: &QwenEngine, req: &streamdense_gateway::ChatReq, o: &streamdense_gateway::Outcome| {
        if o.error.is_some() || o.mode == "skipped" { return; }
        let key = req.msgs.first().filter(|m| m.role == "system").map(|m| watcher::short_hash(&m.content)).unwrap_or_default();
        let f = watcher::job_feat(&req.msgs, req.tools.len(), &o.raw);
        let tokens = (o.n_feed + o.n_new) as u64;
        // recorder on: hand the job to sd-heat (score, label, batch / fold there, then the recorder is zeroed and the
        // gate opened); the worker goes on at once, the next job's first recorder write waits only if that is not done
        if has_src {
            // truth for the predictor (engine idle here): this sequence's prompt-active set (a few KB clone per job)
            let prompt = if mask_rec { eng.prompt_active() } else { None };
            // answer library: token counts (no text unless library_text)
            let lib = lib_on.then(|| crate::library::LibJob { n_prompt: o.n_prompt as u32, n_reused: o.n_reused as u32, n_feed: o.n_feed as u32,
                n_new: o.n_new as u32, mtp: o.mtp, spec: o.spec,
                text: if lib_text { req.msgs.iter().rev().find(|m| m.role == "user").map(|m| watcher::trunc(&m.content, streamdense_core::library::TEXT_MAX).to_string()) } else { None } });
            ws2.submit(watcher::JobMsg { f, tokens, key, msg_id: o.msg_id.clone(), t_end: watcher::now_s(), prompt, lib });
        }
        else { ws2.on_job(&f, tokens, None, &key, &o.msg_id); }
    };
    let ext_w = Some(ws as Arc<dyn streamdense_gateway::Extension>);
    let ext_w = if opt.dflash2.is_some() { ext_w.map(|inner| Arc::new(Df2Ext { inner }) as Arc<dyn streamdense_gateway::Extension>) } else { ext_w };
    let ready_fn = |p: u16| {
        let mut ready = info.clone();
        ready["ready"] = json!(true); ready["port"] = json!(p); ready["host"] = json!(host);
        ready["watcher"] = w_ready;
        ready["footprint_gb"] = json!(gb(footprint::self_footprint())); ready["peak_footprint_gb"] = json!(gb(mon.peak()));
        ready["heap_mb"] = json!(alloc::heap_now() as f64 / 1e6);
        println!("{}", ready);
        use std::io::Write;
        let _ = std::io::stdout().flush();
        eprintln!("gateway: listening on http://{}:{} (model name {}); Claude Code env: streamdense gateway --print-env --port {}", host, p, name, p);
    };
    let r = match &cb {
        // 12 GB mode: the compact backend; the job hook sees the engine inside it
        Some(c) => {
            let on_job: streamdense_gateway::JobHook<streamdense_gateway::compact::CompactBackend> = Box::new(move |b, req, o| job(b.eng, req, o));
            let hook = hook.map(|h| Box::new(h) as Box<dyn streamdense_gateway::SnapshotHook<streamdense_gateway::compact::CompactBackend>>);
            streamdense_gateway::serve_ext(c, tk, cfg, hook, streamdense_gateway::Ext { ext: ext_w, on_job: Some(on_job) }, ready_fn)
        }
        None => {
            let on_job: streamdense_gateway::JobHook<QwenEngine> = Box::new(move |e, req, o| job(e, req, o));
            let hook = hook.map(|h| Box::new(h) as Box<dyn streamdense_gateway::SnapshotHook<QwenEngine>>);
            streamdense_gateway::serve_ext(&eng, tk, cfg, hook, streamdense_gateway::Ext { ext: ext_w, on_job: Some(on_job) }, ready_fn)
        }
    };
    if let Err(e) = r { die(e); }
}

/// 12 GB mode setup (before the budget / open): what the formula and the set rule need.
struct FhSetup { rule: SetRule, plan_total: u64, t1_path: std::path::PathBuf }

/// Streamed prompt read options, K from `fhc.cap_gb` by the keep formula (with every other option of this run), the
/// global score file (clear error when missing / for another model), the initial set = global top K per layer.
fn forheader_setup(model: &str, opt: &mut QwenOptions, fhc: &streamdense_core::config::Forheader) -> FhSetup {
    if opt.activity || opt.read_mask { die("12 GB mode: the activity recorder / read mask must be off".into()); }
    let st = fhc.stream_tokens.min(opt.max_ctx);
    if st < fhc.stream_tokens { eprintln!("12 GB mode: stream_tokens {} capped at max_ctx {}", fhc.stream_tokens, opt.max_ctx); }
    opt.ffn_stream_tokens = st;
    opt.ffn_stream_slots = fhc.slots;
    opt.ffn_stream_host = fhc.host;
    // fast prompt read: needs bf16 KV (the fast prefill attention); under --kv q8 it falls back to the exact read
    // and stream pieces of at least prefill_t rows (the compact backend aligns pieces to its chunks)
    opt.ffn_stream_fast = fhc.fast && opt.kv == KvFormat::Bf16 && st >= opt.prefill_t;
    if fhc.fast && !opt.ffn_stream_fast {
        eprintln!("12 GB mode: fast prompt read off (needs bf16 KV and stream_tokens {} >= prefill_t {}): exact read", st, opt.prefill_t);
    }
    // fused decode (MmaD GEMV twins with gemv = mma, + encode-ahead): bit-identical output; this mode only
    opt.decode_fused = fhc.fused;
    // context drafts (MTP off only): max drafts per pass; the hold save, lookup index and draft table join the budget
    // before the keep formula below, so K leaves room for them
    opt.ctx_draft = if fhc.ctx_draft && !opt.mtp { fhc.ctx_draft_max } else { 0 };
    if fhc.ctx_draft && opt.mtp { eprintln!("12 GB mode: ctx_draft ignored (MTP is on: MTP drafts are used)"); }
    let t1_path = watcher::expand_home(&fhc.t1);
    if !t1_path.is_file() {
        die(format!("12 GB mode: global score file {} not found ([forheader] t1). Make it with `streamdense forheader export --data <T1 dir>` \
                     or copy one in with `streamdense forheader install FILE`", t1_path.display()));
    }
    let t1 = T1::read(&t1_path).unwrap_or_else(|e| die(format!("12 GB mode: {}", e)));
    let cap = (fhc.cap_gb * 1e9) as u64;
    let p = QwenEngine::compact_keep_for_cap(model, opt, cap).unwrap_or_else(|e| die(format!("12 GB mode: keep formula: {}", e)));
    t1.check_dims(p.layers, p.inter).unwrap_or_else(|e| die(format!("12 GB mode: {}: {}", t1_path.display(), e)));
    eprintln!("12 GB mode: keep formula | cap {:.3} GB, MTP {}, ctx {}, ctx_draft {}: rest {:.3} GB (non-FFN rows + reserve) + FFN fixed {:.3} GB -> room {:.3} GB / ({} layers x {:.1} B/neuron) \
               -> K = {} of {} ({:.4}); planned total {:.3} GB",
              gb(cap), opt.mtp, opt.max_ctx, opt.ctx_draft, gb(p.rest), p.ffn_fixed / 1e9, (p.cap as f64 - p.rest as f64 - p.ffn_fixed) / 1e9, p.layers, p.per_neuron,
              p.k, p.inter, p.share, gb(p.total));
    let rule = SetRule::new(t1, p.k).unwrap_or_else(|e| die(format!("12 GB mode: {}", e)));
    opt.ffn_compact = Some(Arc::new(rule.initial()));
    FhSetup { rule, plan_total: p.total, t1_path }
}

/// The answer library's directory: <watcher dir>/library.
fn wdir2_lib(ws: &watcher::Shared) -> std::path::PathBuf { ws.w.lock().map(|w| w.dir.join("library")).unwrap_or_default() }
