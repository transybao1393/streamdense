//! Static planner cut (worker.rs `static_prefix_len`, snap_tier `StableTag::plan_static`) on the tiny model
//! (models/tiny_qwen35, ids folded) + the 27B tokenizer, over real HTTP: two planner-style system turns share the
//! static part (instructions) and differ in the `PROJECT FACTS` part.
//! 1. facts A, cold cache: saves the static cut, the full system-turn cut and the rolling state;
//! 2. facts B: restores the static cut and prefills only from the marker on (n_reused >= static length); its reply
//!    equals a cold run of the same request on a fresh gateway.
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::client::request;
use streamdense_gateway::{CacheHook, GatewayConfig, SnapshotHook, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_model::snapshot::SnapshotCache;
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("sd_gwstatic_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

fn start(dir: std::path::PathBuf, log: String) -> Option<u16> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", tok)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer missing");
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(&tok).unwrap());
        let eng = QwenEngine::open(&tiny, QwenOptions { max_ctx: 4096, mtp: true, ..Default::default() }).unwrap();
        let vocab = eng.cfg.vocab as u32;
        let cache = SnapshotCache::open(dir, &CacheHook::model_id(&tiny)).unwrap();
        let hook: Box<dyn SnapshotHook<QwenEngine>> = Box::new(CacheHook::new(cache, 8, 1 << 30));
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab), ..Default::default() }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, Some(hook), |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(120)).unwrap())
}

const STATIC: &str = "You are the planner of a small coding agent. Read the user's request and the project facts, then \
write a numbered plan of tool calls that completes the request. Use only the tools listed below. Each step names one \
tool and its arguments. Prefer reading a file before editing it. Never invent file paths: use the paths in the facts. \
Keep the plan short: at most eight steps. When the request is already done, answer DONE.\n\nSTART\nThe first line of \
your answer is the first step.\n\nTOOLS\n- read_file(path): the file's text\n- edit_file(path, old, new): replace \
old with new\n- run(cmd): run a shell command in the project root\n- list(dir): the entries of a directory";

fn body(facts: &str) -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut b = v["body"].clone();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(12);
    b["system"] = json!(format!("{}\n\nPROJECT FACTS (tree-sitter, exact):\n{}", STATIC, facts));
    b["messages"] = json!([{"role": "user", "content": "Add a --verbose flag to the CLI."}]);
    b
}

fn post(port: u16, b: &Value) -> String {
    let r = request(port, "POST", "/v1/messages", Some(b)).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()["content"].to_string()
}

fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}
fn kinds(l: &Value) -> Vec<String> { l["snap"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect() }

#[test]
fn static_planner_cut_tiny() {
    let facts_a = "files: src/main.rs src/cli.rs\nrecords: Args { input: String }\noutline: main() -> parse_args()";
    let facts_b = "files: src/main.rs src/cli.rs src/log.rs\nrecords: Args { input: String, quiet: bool }\noutline: main() -> parse_args() -> run()";
    let dir = tmp("cache");
    let log = tmp("log.jsonl").display().to_string();
    let Some(port) = start(dir.clone(), log.clone()) else { return };

    let _ = post(port, &body(facts_a));
    let l = log_lines(&log);
    assert_eq!(l[0]["mode"], "full", "{}", l[0]);
    assert_eq!(kinds(&l[0]), ["save_stable", "save_stable", "save_roll"], "{}", l[0]);
    let n_static = l[0]["snap"][0]["n"].as_u64().unwrap();
    let n_sys = l[0]["snap"][1]["n"].as_u64().unwrap();
    assert!(n_static >= 64 && n_static < n_sys, "static {} sys {}", n_static, n_sys);

    // facts B: same static part -> restore at the static cut, prefill from the marker on; new facts cut saved
    let r2 = post(port, &body(facts_b));
    let l = log_lines(&log);
    assert_eq!(l[1]["mode"], "restore", "{}", l[1]);
    assert_eq!(kinds(&l[1])[0], "restore");
    let reused = l[1]["n_reused"].as_u64().unwrap();
    assert!(reused >= n_static, "reused {} < static {}", reused, n_static);
    assert_eq!(reused, n_static, "the facts differ right after the marker: the static cut is the longest prefix");
    assert_eq!(l[1]["n_feed"].as_u64().unwrap(), l[1]["n_prompt"].as_u64().unwrap() - n_static);
    let k = kinds(&l[1]);
    assert_eq!(k.iter().filter(|x| *x == "save_stable").count(), 1, "only the new facts cut is saved: {:?}", k);

    // cold run of the facts-B request on a fresh gateway: the same greedy reply
    let dir2 = tmp("cold");
    let log2 = tmp("cold.jsonl").display().to_string();
    let Some(port2) = start(dir2.clone(), log2.clone()) else { return };
    let cold = post(port2, &body(facts_b));
    assert_eq!(log_lines(&log2)[0]["mode"], "full");
    assert_eq!(r2, cold, "restore at the static cut must give the cold reply");
    println!("  tiny: static {} tokens, system turn {} tokens, second request reused {} fed {}", n_static, n_sys, reused, l[1]["n_feed"]);
    for d in [&dir, &dir2] { let _ = std::fs::remove_dir_all(d); }
    for f in [&log, &log2] { let _ = std::fs::remove_file(f); }
}
