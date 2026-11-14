//! `streamdense mask-test [--keep 0.95,0.9,0.8] --prompts FILE [--max-tokens 64] [--model DIR] [--dir ~/.streamdense/watcher]
//! [--window all] [--only CFG,..] [--spec-depth N|auto] [--draft-table DIR] [--no-mtp] [--max-ctx 4096] [--out FILE.json]`
//! Read mask on the real model: speed and output agreement. Loads the model ONCE with `read_mask` on, builds the
//! profile ReadMask per keep from the watcher store (read only, exactly as `watcher mask` builds it: `build_profile` +
//! the joined units of predict.bin), then per prompt runs greedy decode through the gateway's path
//! (`feed_stream_mtp`: MTP + spec depth from streamdense.toml, after `reset`) in these configs:
//!   base       set_read_mask(None)                         (today's code path)
//!   ones       all-ones mask, default policy               (must equal base bit for bit)
//!   kK         profile at keep K, default policy           (unsure pass reads everything)
//!   kK-pure    profile at keep K, gap_thr 0 + oop_min 0    (never unsure; reads profile ∪ prompt-active)
//!   kK-prof    kK-pure + recorder thresholds 1e30          (diagnostic: prompt-active empty, decode reads the profile only)
//! `--only base,k0.8-prof` runs a subset (base always).
//! Reports per config: decode tok/s, tokens identical to base (prefix and % of positions), mask passes (masked / full).
//! The draft table is OFF unless --draft-table is given: it learns each generation, so a repeated prompt would draft
//! from its own earlier output and the later configs would look faster than they are (output is unchanged either way).
//! Prompts: {"prompts": [{"text": "..."}, ...]}, each rendered as one user turn with the chat template (thinking off).
use serde_json::{json, Value};
use streamdense_core::{footprint, gb};
use streamdense_model::activity::ActivityStats;
use streamdense_model::qwen_layers::GemvImpl;
use streamdense_model::readmask::{MaskPolicy, MaskStats, ReadMask};
use streamdense_model::{QwenConfig, QwenEngine, QwenOptions};
use streamdense_tokenizer::{render_chat, ChatMessage, ChatOptions, Tokenizer, ENDOFTEXT, IM_END};

use crate::watcher::{self, Dims};
use crate::{config_root, load_config, Args};

fn die(e: String) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }

/// Engine-format profile masks per keep, built read only from the store. (keep, mask, masked blocks, masked heads)
fn profiles(a: &Args, cfg: &QwenConfig, keeps: &[f64]) -> Vec<(f64, ReadMask, usize, usize)> {
    use crate::watcher::predict::{self, Predictor, ReadMask as PMask, Window};
    let conf = load_config();
    let wc = &conf.cfg.watcher;
    let dir = a.s("--dir").map(|d| watcher::expand_home(&d)).unwrap_or_else(|| watcher::config_dir(&wc.dir));
    let mut d = Dims::from_stats(&ActivityStats::dims(cfg), wc.granularity);
    d.vocab = cfg.vocab;
    let mut st = crate::store::Store::open(&dir, d, crate::watcher_curve::names(), watcher::MIXED, true).unwrap_or_else(|e| die(format!("store {}: {}", dir.display(), e)));
    st.refresh();
    let win = Window::parse(&a.s("--window").unwrap_or_else(|| "all".into()));
    let mut joined = PMask::boxed(&st);
    if let Some(p) = Predictor::open_ro(&st.dir) { let _ = p.load_joined(&mut joined); }
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|x| x.as_secs_f64()).unwrap_or(0.0);
    let full = ReadMask::full(cfg);
    let mut pm = PMask::boxed(&st);
    keeps.iter().map(|&k| {
        let tok = predict::build_profile(&mut st, win, k, t, &mut pm);
        pm.or_from(&joined);
        let mut e = ReadMask::empty(cfg);
        if !pm.to_engine(&mut e) { die("store shape != model shape".into()); }
        let (mb, mh) = (full.count_ffn() - e.count_ffn(), full.count_heads() - e.count_heads());
        eprintln!("profile keep {} (window {}, {} tokens all-time, {} joined): masked {} / {} FFN blocks ({:.2}%), {} / {} heads",
                  k, win.name(), tok.2, joined.count().0 + joined.count().1, mb, full.count_ffn(), 100.0 * mb as f64 / full.count_ffn() as f64, mh, full.count_heads());
        (k, e, mb, mh)
    }).collect()
}

struct Cfg { name: String, mask: Option<ReadMask>, policy: MaskPolicy, masked_blocks: usize, masked_heads: usize }
#[derive(Default, Clone)]
struct Run { tokens: Vec<u32>, decode_s: f64, prefill_s: f64, gen: usize, stats: MaskStats, spec: (usize, usize), read: (usize, usize), prompt: (usize, usize) }

fn run(eng: &QwenEngine, c: &Cfg, ids: &[u32], n: usize, mtp: bool) -> Run {
    eng.set_mask_policy(c.policy);
    eng.set_read_mask(c.mask.as_ref()).unwrap_or_else(|e| die(e));
    eng.reset();
    eng.reset_mask_stats();
    let mut toks = Vec::with_capacity(n);
    let mut cb = |t: u32| { toks.push(t); true };
    let stop = [IM_END, ENDOFTEXT];
    let (s, spec) = if mtp {
        let (s, m) = eng.feed_stream_mtp(ids, n, &stop, &mut cb).unwrap_or_else(|e| die(e));
        (s, (m.passes, m.accepted))
    } else {
        (eng.feed_stream(ids, n, &stop, &mut cb).unwrap_or_else(|e| die(e)), (0, 0))
    };
    // the decode read set at the end (profile ∪ prompt-active ∪ learned; full when no mask) and the prompt-active set
    let cnt = |m: Option<ReadMask>| m.map_or((0, 0), |m| (m.count_ffn(), m.count_heads()));
    Run { tokens: toks, decode_s: s.decode_s, prefill_s: s.prefill_s, gen: s.gen, stats: eng.mask_stats(), spec, read: cnt(eng.decode_mask()), prompt: cnt(eng.prompt_active()) }
}

/// (identical prefix length, identical positions, compared positions = max of the two lengths)
fn agree(b: &[u32], x: &[u32]) -> (usize, usize, usize) {
    let pre = b.iter().zip(x).take_while(|(p, q)| p == q).count();
    let same = b.iter().zip(x).filter(|(p, q)| p == q).count();
    (pre, same, b.len().max(x.len()))
}

pub fn main(a: &Args) {
    let conf = load_config();
    let model = a.s("--model").unwrap_or_else(|| config_root(&conf).join(&conf.cfg.gateway.model).display().to_string());
    let keeps: Vec<f64> = a.sd("--keep", "0.95,0.9,0.8").split(',').map(|k| k.trim().parse().unwrap_or_else(|_| die(format!("--keep {}", k)))).collect();
    let n = a.f("--max-tokens", 64.0) as usize;
    let pfile = a.s("--prompts").unwrap_or_else(|| die("--prompts FILE".into()));
    let pj: Value = serde_json::from_str(&std::fs::read_to_string(&pfile).unwrap_or_else(|e| die(format!("{}: {}", pfile, e)))).unwrap_or_else(|e| die(format!("{}: {}", pfile, e)));
    let texts: Vec<String> = pj["prompts"].as_array().unwrap_or_else(|| die("prompts file: want {\"prompts\": [{\"text\": ..}]}".into()))
        .iter().filter_map(|p| p["text"].as_str().map(String::from)).collect();
    let cfg = QwenConfig::from_model_dir(&model).unwrap_or_else(|e| die(e));

    // profiles + prompt ids before the engine (CPU, small)
    let profs = profiles(a, &cfg, &keeps);
    let tk = Tokenizer::from_path(&model).unwrap_or_else(|e| die(format!("tokenizer {}: {}", model, e)));
    let prompts: Vec<Vec<u32>> = texts.iter().map(|t| {
        let s = render_chat(&[ChatMessage::new("user", t)], &[], &ChatOptions::default()).unwrap_or_else(|e| die(e));
        tk.encode(&s)
    }).collect();
    eprintln!("{} prompts, {} .. {} tokens", prompts.len(), prompts.iter().map(Vec::len).min().unwrap_or(0), prompts.iter().map(Vec::len).max().unwrap_or(0));

    let mut opt = QwenOptions::default();
    opt.cap = (a.f("--cap", 17.0) * 1e9) as u64;
    opt.max_ctx = a.f("--max-ctx", 4096.0) as usize;
    opt.mtp = !a.has("--no-mtp") && conf.cfg.gateway.mtp;
    opt.read_mask = true;
    crate::qwen::spec_opts(a, &mut opt, Some(conf.cfg.gateway.spec_depth.arg()), None);
    if !opt.mtp { opt.spec_depth = 1; opt.spec_auto = false; }
    opt.gemv = GemvImpl::Mma;
    opt.draft_vocab = (opt.mtp && !conf.cfg.gateway.draft_vocab.is_empty()).then(|| config_root(&conf).join(&conf.cfg.gateway.draft_vocab))
        .filter(|p| QwenEngine::check_draft_vocab(&model, p).is_ok());
    opt.live_cap = (conf.cfg.ram.cap_gb * 1e9) as u64;
    let bud = QwenEngine::budget(&model, &opt).unwrap_or_else(|e| die(e));
    if let Err(e) = bud.check() { die(e); }
    let mon = footprint::Monitor::start((conf.cfg.ram.cap_gb * 1e9) as u64, (conf.cfg.ram.swap_gb * 1e9) as u64, 250);
    let t0 = std::time::Instant::now();
    let eng = QwenEngine::open(&model, opt.clone()).unwrap_or_else(|e| die(e));
    let path = if opt.mtp { format!("feed_stream_mtp, MTP spec depth {}, draft table {}, draft vocab {}", if opt.spec_auto { "auto".into() } else { opt.spec_depth.to_string() },
        opt.draft_table.as_ref().map_or("off".into(), |p| p.display().to_string()), if opt.draft_vocab.is_some() { "on" } else { "off" }) } else { "feed_stream (plain greedy, no MTP)".into() };
    eprintln!("opened in {:.1} s | footprint {:.3} GB | decode path: {}", t0.elapsed().as_secs_f64(), gb(footprint::self_footprint()), path);

    let def = MaskPolicy::default();
    let pure = MaskPolicy { gap_thr: 0.0, oop_min: 0, ..def };
    // diagnostic: recorder thresholds out of reach -> prompt-active empty -> decode reads exactly the profile (never unsure)
    let prof = MaskPolicy { ffn_thr: 1e30, head_thr: 1e30, ..pure };
    let mut cfgs = vec![Cfg { name: "base".into(), mask: None, policy: def, masked_blocks: 0, masked_heads: 0 },
                        Cfg { name: "ones".into(), mask: Some(ReadMask::full(&eng.cfg)), policy: def, masked_blocks: 0, masked_heads: 0 }];
    for (k, m, mb, mh) in &profs {
        cfgs.push(Cfg { name: format!("k{}", k), mask: Some(m.clone()), policy: def, masked_blocks: *mb, masked_heads: *mh });
        cfgs.push(Cfg { name: format!("k{}-pure", k), mask: Some(m.clone()), policy: pure, masked_blocks: *mb, masked_heads: *mh });
        cfgs.push(Cfg { name: format!("k{}-prof", k), mask: Some(m.clone()), policy: prof, masked_blocks: *mb, masked_heads: *mh });
    }
    // --only base,k0.8-prof,..: a subset (base always first: agreement is measured against it)
    if let Some(o) = a.s("--only") {
        let keep: Vec<&str> = o.split(',').map(str::trim).collect();
        cfgs.retain(|c| c.name == "base" || keep.contains(&c.name.as_str()));
    }

    // warm-up (GPU clocks, pipelines): one short unmasked + one masked run, discarded
    let _ = run(&eng, &cfgs[0], &prompts[0], 8, opt.mtp);
    if let Some(c) = cfgs.get(1) { let _ = run(&eng, c, &prompts[0], 8, opt.mtp); }

    let mut res: Vec<Vec<Run>> = vec![vec![]; cfgs.len()];
    let mut rows = vec![];
    for (pi, ids) in prompts.iter().enumerate() {
        for (ci, c) in cfgs.iter().enumerate() {
            let r = run(&eng, c, ids, n, opt.mtp);
            let (pre, same, tot) = agree(&res[0].get(pi).map_or(r.tokens.clone(), |b| b.tokens.clone()), &r.tokens);
            let tps = (r.gen.saturating_sub(1)) as f64 / r.decode_s.max(1e-9);
            eprintln!("prompt {} ({} tok) {:>10}: {:3} tok, {:6.2} tok/s decode, prefix {:3}, identical {:3}/{:3}, passes masked {} full {} (gap {}, oop {}), learned {}+{}, read set {} blocks + {} heads (prompt-active {} + {}) | footprint {:.2} GB",
                      pi, ids.len(), c.name, r.tokens.len(), tps, pre, same, tot, r.stats.masked_passes, r.stats.full_passes, r.stats.full_gap, r.stats.full_oop,
                      r.stats.learned_blocks, r.stats.learned_heads, r.read.0, r.read.1, r.prompt.0, r.prompt.1, gb(footprint::self_footprint()));
            rows.push(json!({"prompt": pi, "prompt_tokens": ids.len(), "config": c.name, "tokens": r.tokens, "decode_s": r.decode_s, "prefill_s": r.prefill_s,
                "gen": r.gen, "decode_tok_s": tps, "prefix": pre, "identical": same, "compared": tot, "spec_passes": r.spec.0, "spec_accepted": r.spec.1,
                "masked_passes": r.stats.masked_passes, "full_passes": r.stats.full_passes, "full_gap": r.stats.full_gap, "full_oop": r.stats.full_oop,
                "learned_blocks": r.stats.learned_blocks, "learned_heads": r.stats.learned_heads,
                "read_blocks": r.read.0, "read_heads": r.read.1, "prompt_active_blocks": r.prompt.0, "prompt_active_heads": r.prompt.1}));
            res[ci].push(r);
        }
    }
    eng.set_read_mask(None).unwrap_or_else(|e| die(e));

    println!("read mask on {} | {} prompts x {} max tokens | decode path: {}", model, prompts.len(), n, path);
    println!("| config | masked 64-blocks | masked heads | decode tok/s (sum tokens / sum time) | vs base | prompts identical | identical tokens | mean identical prefix | masked / full passes | full share | blocks actually masked in decode |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let base_tps = { let r = &res[0]; r.iter().map(|x| x.gen.saturating_sub(1)).sum::<usize>() as f64 / r.iter().map(|x| x.decode_s).sum::<f64>().max(1e-9) };
    let full_ffn = ReadMask::full(&eng.cfg).count_ffn();
    let mut summary = vec![];
    for (ci, c) in cfgs.iter().enumerate() {
        let r = &res[ci];
        let tps = r.iter().map(|x| x.gen.saturating_sub(1)).sum::<usize>() as f64 / r.iter().map(|x| x.decode_s).sum::<f64>().max(1e-9);
        let (mut pre, mut same, mut tot, mut exact) = (0usize, 0usize, 0usize, 0usize);
        for (pi, x) in r.iter().enumerate() {
            let (p, s, t) = agree(&res[0][pi].tokens, &x.tokens);
            pre += p; same += s; tot += t;
            if s == t { exact += 1; }
        }
        let (mp, fp) = (r.iter().map(|x| x.stats.masked_passes).sum::<u64>(), r.iter().map(|x| x.stats.full_passes).sum::<u64>());
        let rb = r.iter().map(|x| x.read.0).sum::<usize>() as f64 / r.len().max(1) as f64;
        let read_pct = if c.mask.is_some() { format!("{:.1}%", 100.0 * (1.0 - rb / full_ffn as f64)) } else { "-".into() };
        let share = if mp + fp > 0 { format!("{:.1}%", 100.0 * fp as f64 / (mp + fp) as f64) } else { "-".into() };
        println!("| {} | {} ({:.1}%) | {} | {:.2} | {:+.1}% | {} / {} | {:.1}% | {:.1} | {} / {} | {} | {} |", c.name, c.masked_blocks, 100.0 * c.masked_blocks as f64 / full_ffn as f64,
                 c.masked_heads, tps, 100.0 * (tps / base_tps - 1.0), exact, r.len(), 100.0 * same as f64 / tot.max(1) as f64, pre as f64 / r.len().max(1) as f64, mp, fp, share, read_pct);
        summary.push(json!({"config": c.name, "masked_blocks": c.masked_blocks, "masked_heads": c.masked_heads, "decode_tok_s": tps, "prompts_identical": exact,
            "identical_tokens_pct": 100.0 * same as f64 / tot.max(1) as f64, "mean_prefix": pre as f64 / r.len().max(1) as f64, "masked_passes": mp, "full_passes": fp, "mean_read_blocks": rb}));
    }
    let ones_ran = cfgs.get(1).is_some_and(|c| c.name == "ones");
    let ones_ok = ones_ran && res[1].iter().zip(&res[0]).all(|(x, b)| x.tokens == b.tokens);
    println!("all-ones == base on every prompt: {}", if !ones_ran { "not run" } else if ones_ok { "YES" } else { "NO" });
    println!("peak phys_footprint {:.3} GB", gb(mon.peak()));
    if let Some(o) = a.s("--out") {
        let v = json!({"model": model, "decode_path": path, "max_tokens": n, "prompts": texts, "summary": summary, "rows": rows,
                       "ones_identical": if ones_ran { json!(ones_ok) } else { Value::Null }, "peak_footprint_gb": gb(mon.peak())});
        std::fs::write(&o, serde_json::to_string_pretty(&v).unwrap()).unwrap_or_else(|e| die(format!("{}: {}", o, e)));
        eprintln!("wrote {}", o);
    }
}
