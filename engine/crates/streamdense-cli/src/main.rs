//! streamdense check | bench | run   (Phase 1: in-RAM decode under a hard cap)
//!   common: --export DIR --cap GB --max-ctx N --ffn sparse|dense --compact ordered|atomic --head q8|fp16 --depth N
//!   check:  --ref engine/refs/....json --prompts engine/refs/prompts7b.json [--nnew 128] [--out FILE] [--nosync]
//!   bench:  --prompts engine/refs/prompts7b.json [--ctxs 96,992,2016] [--ngen 64] [--out FILE]
//!   run:    --tokens 1,2,3 | --prompts FILE --id N   [--nnew 64]
//! streamdense bench gemv-k [--ks 1,2,3,4,5,8] [--ds 2,4,8] [--reps 5] [--no-head] [--impl old|mma|both]   (synthetic K-row GEMV microbench,
//!   no model load; see streamdense-model/src/gemv_bench.rs)
//! streamdense qwen budget|check|teacher|dump|bench|run ...   (Qwen3.8-27B engine, see src/qwen.rs)
//! streamdense gateway ...   (Anthropic Messages API for Claude Code on the 27B engine, see src/gateway.rs)
//! streamdense vocab keep --model DIR [--out FILE] [--drop cjk,ja,ko,tl,id,ms] [--drop-list FILE]   (pruned MTP draft vocab, src/vocab.rs)
//! streamdense prompts spec [--out engine/refs/qwen38_spec_prompts.json]   (spec-bench default prompt set, src/prompts.rs)
//! streamdense config [--json] [--profile NAME]   (resolved streamdense.toml + the file it came from, see src/up.rs)
//! streamdense up [--profile NAME] [gateway args...]   (sd-watchdog + nice + gateway, all from streamdense.toml, see src/up.rs)
//! streamdense table stats DIR [--budget-mb 32]   (speculative draft table stats, see src/table.rs)
//! streamdense watcher curve [--dir ~/.streamdense/watcher] [--out FILE.json]   (unload curve of the saved watcher profiles, src/watcher_curve.rs)
//! streamdense mask-test --prompts FILE [--keep 0.95,0.9,0.8] [--max-tokens 64] [--out FILE.json]   (read mask on the real 27B: speed + agreement, src/mask_test.rs)
//! streamdense help | --help | -h   (usage, exit 0; an unknown or missing command prints usage, exit 2, no model load)
//! streamdense watcher heat [--dir ~/.streamdense/watcher] [--layers]   (heat map summary of the running / last gateway, src/heat.rs)
//! streamdense forheader export --data DIR | install FILE | info [FILE]   (12 GB mode input file, src/forheader.rs)
//! streamdense index build|stats|query ...   (history index of text the user already has → drafts, src/history.rs)
mod forheader;
mod gateway;
mod heat;
mod history;
mod library;
mod mask_test;
mod store;
mod prompts;
mod up;
mod qwen;
mod vocab;
mod table;
mod watcher;
mod watcher_curve;
use serde_json::{json, Value};
use streamdense_core::{alloc, config, footprint, gb};
use streamdense_model::{Engine, Options};

/// Repo root baked in at build time (engine/crates/streamdense-cli/../../..).
pub(crate) const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

/// streamdense.toml (see streamdense_core::config); a bad file is fatal so a typo never silently falls back to defaults.
pub(crate) fn load_config() -> config::Loaded {
    config::load().unwrap_or_else(|e| { eprintln!("ERROR: {}", e); std::process::exit(2) })
}

/// `load_config` with `--profile NAME` from the command line (it replaces the file's `[profile] name`; "" = none).
pub(crate) fn load_config_profile(a: &Args) -> config::Loaded {
    config::load_profile(a.s("--profile").as_deref()).unwrap_or_else(|e| { eprintln!("ERROR: {}", e); std::process::exit(2) })
}

/// The profile line of a summary: `profile coding: key=value, ...` or `profile: none`.
pub(crate) fn profile_line(c: &config::Loaded) -> String {
    c.profile.as_ref().map(|p| p.line()).unwrap_or_else(|| "profile: none".into())
}

/// Base dir for relative paths in the config: the dir of a repo-style streamdense.toml, else the built-in repo root.
pub(crate) fn config_root(c: &config::Loaded) -> std::path::PathBuf {
    match &c.source {
        Some(p) if p.file_name().is_some_and(|n| n == config::FILE_NAME) => p.parent().map(|d| d.to_path_buf()).unwrap_or_else(|| ROOT.into()),
        _ => std::path::PathBuf::from(ROOT),
    }
}

#[global_allocator]
static A: alloc::Counting = alloc::Counting;

struct Args(Vec<String>);
impl Args {
    fn s(&self, k: &str) -> Option<String> { self.0.iter().position(|x| x == k).map(|i| self.0[i + 1].clone()) }
    fn sd(&self, k: &str, d: &str) -> String { self.s(k).unwrap_or(d.to_string()) }
    fn f(&self, k: &str, d: f64) -> f64 { self.s(k).map(|v| v.parse().unwrap()).unwrap_or(d) }
    fn has(&self, k: &str) -> bool { self.0.iter().any(|x| x == k) }
}
fn ids(v: &Value) -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() }
fn conditions() -> String {
    let o = std::process::Command::new("/bin/sh").arg("-c").arg("df -h / | tail -1; sysctl vm.swapusage; memory_pressure | tail -1").output().unwrap();
    String::from_utf8_lossy(&o.stdout).to_string()
}

/// Top-level usage (printed by `help` / `--help` / `-h`, and on any unknown or missing subcommand).
const USAGE: &str = "usage: streamdense <command> [options]

commands:
  qwen budget|check|teacher|dump|bench|run|serve ...   Qwen3.8-27B engine (src/qwen.rs)
  gateway ...                Anthropic Messages API on the 27B engine (src/gateway.rs)
  up [--profile NAME] [args]  sd-watchdog + nice + gateway from streamdense.toml (src/up.rs; profile: coding)
  config [--json] [--profile NAME]  resolved streamdense.toml, its source and profile
  mask-test --prompts FILE   read mask on the real 27B (src/mask_test.rs)
  vocab keep --model DIR     pruned MTP draft vocab (src/vocab.rs)
  prompts spec               spec-bench default prompt set (src/prompts.rs)
  table stats DIR            speculative draft table stats (src/table.rs)
  watcher curve|heat ...     watcher unload curve / heat map (src/watcher_curve.rs, src/heat.rs)
  library stats|export ...   answer library of the watcher (records, disk, metadata JSONL; src/library.rs)
  forheader export|install|info   input file of the 12 GB mode (research; src/forheader.rs)
  index build|stats|query     history index: drafts from project files / sessions / replies (src/history.rs)
  bench gemv-k|lut-k ...     synthetic kernel microbenchmarks (no model load)
  check|bench|run ...        7B ProSparse in-RAM decode (loads the 7B export; see the top of src/main.rs)
  help, --help, -h           this text";

/// What `main` does with argv, decided before anything is loaded.
#[derive(Debug, PartialEq, Eq)]
enum Route {
    /// `help` / `--help` / `-h`: usage on stdout, exit 0.
    Help,
    /// Missing or unknown subcommand: usage on stderr, exit 2, no model load.
    Unknown(String),
    /// A known subcommand: dispatched exactly as before.
    Run,
}

/// The only subcommands that reach the 7B model load at the end of `main` (plus `bench` microbench forms handled earlier).
const MODEL_CMDS: [&str; 3] = ["check", "bench", "run"];
/// Subcommands dispatched to their own module before the 7B path.
const MODULE_CMDS: [&str; 12] = ["qwen", "index", "mask-test", "gateway", "config", "up", "vocab", "prompts", "table", "watcher", "library", "forheader"];

fn route(args: &[String]) -> Route {
    match args.get(1).map(String::as_str) {
        Some("help" | "--help" | "-h") => Route::Help,
        Some(c) if MODEL_CMDS.contains(&c) || MODULE_CMDS.contains(&c) => Route::Run,
        Some(c) => Route::Unknown(c.to_string()),
        None => Route::Unknown(String::new()),
    }
}

fn main() {
    let a = Args(std::env::args().collect());
    match route(&a.0) {
        Route::Help => { println!("{}", USAGE); return; }
        Route::Unknown(c) => {
            if c.is_empty() { eprintln!("ERROR: no command given\n{}", USAGE); } else { eprintln!("ERROR: unknown command '{}'\n{}", c, USAGE); }
            std::process::exit(2);
        }
        Route::Run => {}
    }
    let cmd = a.0.get(1).cloned().unwrap_or_default();
    if cmd == "qwen" { qwen::main(&a); return; }
    if cmd == "mask-test" { mask_test::main(&a); return; }
    if cmd == "gateway" { gateway::main(&a); return; }
    if cmd == "config" { up::config_main(&a); return; }
    if cmd == "up" { up::up_main(&a); return; }
    if cmd == "vocab" { vocab::main(&a); return; }
    if cmd == "prompts" { prompts::main(&a); return; }
    if cmd == "table" { table::main(&a); return; }
    if cmd == "watcher" && a.0.get(2).is_some_and(|s| s == "heat") {
        let dir = a.s("--dir").map(|d| watcher::expand_home(&d)).unwrap_or_else(|| watcher::config_dir(&load_config().cfg.watcher.dir));
        if let Err(e) = heat::print(&dir, a.has("--layers")) { eprintln!("ERROR: {}", e); std::process::exit(2); }
        return;
    }
    if cmd == "watcher" { watcher_curve::main(&a); return; }
    if cmd == "library" { library::main(&a); return; }
    if cmd == "forheader" { forheader::main(&a); return; }
    if cmd == "index" { history::main(&a); return; }
    if cmd == "bench" && a.0.get(2).is_some_and(|s| s == "lut-k") { if let Err(e) = streamdense_model::lut_bench::run_cli(&a.0) { eprintln!("ERROR: {}", e); std::process::exit(2); } return; }
    if cmd == "bench" && a.0.get(2).is_some_and(|s| s == "gemv-k") {
        if let Err(e) = streamdense_model::gemv_bench::run_cli(&a.0) { eprintln!("ERROR: {}", e); std::process::exit(2); }
        return;
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let export = a.sd("--export", &format!("{}/exports/q4_prosparse_llama2_7b", root));
    let mut opt = Options::default();
    opt.cap = (a.f("--cap", 5.5) * 1e9) as u64;
    opt.max_ctx = a.f("--max-ctx", 2176.0) as usize;
    opt.ffn = a.sd("--ffn", "sparse");
    opt.ordered = a.sd("--compact", "ordered") == "ordered";
    opt.head = a.sd("--head", "q8");
    opt.depth = a.f("--depth", 2.0) as usize;
    let mon = footprint::Monitor::start(opt.cap + 500_000_000, 8_000_000_000, 250);
    eprintln!("conditions:\n{}", conditions());
    let eng = match Engine::open(&export, opt.clone()) { Ok(e) => e, Err(e) => { eprintln!("ERROR: {}", e); std::process::exit(2); } };
    eprintln!("{} | loaded {:.3} GB in {:.2} s ({:.2} GB/s, F_NOCACHE pread) | {} | footprint {:.3} GB | heap {:.1} MB",
        eng.gpu.name(), gb(eng.loaded_bytes), eng.load_s, gb(eng.loaded_bytes) / eng.load_s, eng.arena_report(), gb(footprint::self_footprint()), eng.heap_mb());
    alloc::set_limit_extra(256 << 20); // the decode loop may not grow the CPU heap past this
    let settings = json!({"ffn": opt.ffn, "compaction": if opt.ordered {"ordered"} else {"atomic"}, "kv": "8:32", "attn_kernel": "v2",
        "lm_head": opt.head, "pipelined_depth": opt.depth, "max_ctx": opt.max_ctx, "cap_gb": gb(opt.cap), "threshold": opt.thr});
    let out_path = a.s("--out");
    let mut out = json!({"settings": settings, "conditions": conditions(), "budget": eng.budget_report, "arena": eng.arena_report()});
    match cmd.as_str() {
        "check" => {
            let refj: Value = serde_json::from_str(&std::fs::read_to_string(a.s("--ref").expect("--ref")).unwrap()).unwrap();
            let pj: Value = serde_json::from_str(&std::fs::read_to_string(a.sd("--prompts", &format!("{}/engine/refs/prompts7b.json", root))).unwrap()).unwrap();
            let nnew = a.f("--nnew", 128.0) as usize;
            let mut rows = vec![];
            let (mut n_ok, mut n_all) = (0, 0);
            for rr in refj["results"].as_array().unwrap() {
                let id = rr["id"].as_u64().unwrap() as usize;
                let rt = ids(&rr["tokens"]);
                let rm: Vec<f64> = rr["margins"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                let prompt = ids(&pj["prompts"][id]["ids"]);
                let g = eng.generate(&prompt, nnew, true).unwrap();
                let gs = if a.has("--nosync") { g.clone() } else { eng.generate(&prompt, nnew, false).unwrap() };
                let first = (0..g.tokens.len().min(rt.len())).find(|&i| g.tokens[i] != rt[i]);
                let same = first.is_none() && g.tokens.len() == rt.len();
                let ms = g.gen_wall_s * 1e3 / g.tokens.len().max(1) as f64;
                n_all += 1; if same { n_ok += 1; }
                let act_gen = g.act_gen.iter().sum::<f64>() / (g.n_gen.max(1) * eng.cfg.layers * eng.cfg.inter) as f64;
                println!("prompt {:2} (P={:3}): {:3} tok, identical to ref: {}{}, pipelined == sync: {}, {:.2} ms/token pipelined, {:.2} sync, active gen {:.2}%",
                    id, prompt.len(), g.tokens.len(), if same {"YES"} else {"NO"},
                    first.map(|f| format!(" (first diff @{}: ref margin {:.4}, ours {:.4})", f, rm[f], g.margins[f])).unwrap_or_default(),
                    g.tokens == gs.tokens, ms, gs.gen_wall_s * 1e3 / gs.tokens.len().max(1) as f64, 100.0 * act_gen);
                rows.push(json!({"id": id, "tokens": g.tokens, "tokens_sync": gs.tokens, "identical_to_ref": same, "first_diff": first,
                    "ref_margin_at_diff": first.map(|f| rm[f]), "margin_at_diff": first.map(|f| g.margins[f]), "margins": g.margins,
                    "pipelined_equals_sync": g.tokens == gs.tokens, "ms_per_token_wall": ms,
                    "ms_per_token_gpu": g.gpu_ms.iter().sum::<f64>() / g.gpu_ms.len().max(1) as f64,
                    "sync_ms_per_token_wall": gs.gen_wall_s * 1e3 / gs.tokens.len().max(1) as f64, "active_frac_gen": act_gen}));
            }
            println!("identical to reference on {} / {} prompts", n_ok, n_all);
            out["results"] = json!(rows); out["identical_prompts"] = json!(n_ok); out["prompts"] = json!(n_all);
        }
        "bench" => {
            let pj: Value = serde_json::from_str(&std::fs::read_to_string(a.sd("--prompts", &format!("{}/engine/refs/prompts7b.json", root))).unwrap()).unwrap();
            let ngen = a.f("--ngen", 64.0) as usize;
            let mut rows = vec![];
            println!("{:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>7}", "prompt", "sync ms", "GPU ms", "pipe ms", "pipe t/s", "active%", "same");
            for c in a.sd("--ctxs", "96,992,2016").split(',') {
                let prompt = ids(&pj["long"][c]);
                let (w, g, p, same, af) = eng.bench_ctx(&prompt, ngen).unwrap();
                println!("{:>6} {:>9.2} {:>9.2} {:>9.2} {:>9.1} {:>8.2} {:>7}", c, w, g, p, 1e3 / p, 100.0 * af, same);
                rows.push(json!({"prompt_len": prompt.len(), "ctx_key": c, "n_gen": ngen, "sync_wall_ms": w, "sync_gpu_ms": g,
                    "pipelined_ms_per_token": p, "tok_per_s_pipelined": 1e3 / p, "pipelined_tokens_identical": same, "active_frac_gen": af}));
            }
            out["rows"] = json!(rows);
        }
        "run" => {
            let prompt: Vec<u32> = match a.s("--tokens") {
                Some(t) => t.split(',').map(|x| x.trim().parse().unwrap()).collect(),
                None => {
                    let pj: Value = serde_json::from_str(&std::fs::read_to_string(a.sd("--prompts", &format!("{}/engine/refs/prompts7b.json", root))).unwrap()).unwrap();
                    ids(&pj["prompts"][a.f("--id", 32.0) as usize]["ids"])
                }
            };
            let g = eng.generate(&prompt, a.f("--nnew", 64.0) as usize, true).unwrap();
            println!("{}", g.tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
            println!("{} tokens, {:.2} ms/token ({:.1} tok/s)", g.tokens.len(), g.gen_wall_s * 1e3 / g.tokens.len() as f64, g.tokens.len() as f64 / g.gen_wall_s);
            out["tokens"] = json!(g.tokens);
        }
        _ => unreachable!("route() only lets check|bench|run reach the model path"),
    }
    let peak = mon.peak();
    println!("peak phys_footprint {:.3} GB (cap {:.3} GB), CPU heap peak {:.1} MB", gb(peak), gb(opt.cap), alloc::heap_peak() as f64 / 1e6);
    out["peak_phys_footprint_gb"] = json!(gb(peak)); out["cpu_heap_peak_mb"] = json!(alloc::heap_peak() as f64 / 1e6);
    if let Some(p) = out_path { alloc::clear_limit(); std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap(); println!("wrote {}", p); }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(v: &[&str]) -> Route { route(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()) }
    #[test]
    fn help_and_unknown_never_reach_a_model() {
        for h in ["help", "--help", "-h"] { assert_eq!(r(&["streamdense", h]), Route::Help); }
        assert_eq!(r(&["streamdense", "down"]), Route::Unknown("down".into()));
        assert_eq!(r(&["streamdense", "--export"]), Route::Unknown("--export".into()));
        assert_eq!(r(&["streamdense"]), Route::Unknown(String::new()));
        for c in MODEL_CMDS.iter().chain(MODULE_CMDS.iter()) { assert_eq!(r(&["streamdense", c]), Route::Run, "{}", c); }
    }
}
