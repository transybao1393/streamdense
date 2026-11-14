//! Snapshot hook (snap::CacheHook) in the gateway on the tiny model (models/tiny_qwen35, ids folded) + the 27B
//! tokenizer, over real HTTP:
//! 1. a coding-agent request (testdata/anthropic_agent_req.json), cold cache -> full prefill, saves the stable prefix (system turn) + the rolling state;
//! 2. the identical re-send -> restore of the rolling state (all but the last prompt token), same reply;
//! 3. a new conversation with the same system + tools -> restore of the stable prefix;
//! 4. cache cap: a hook capped at 1 file keeps only the newest.
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
    let d = std::env::temp_dir().join(format!("sd_gwsnap_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

fn start(dir: std::path::PathBuf, log: String, max_files: usize) -> Option<u16> {
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
        let hook: Box<dyn SnapshotHook<QwenEngine>> = Box::new(CacheHook::new(cache, max_files, 1 << 30));
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab) }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, Some(hook), |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(120)).unwrap())
}

fn body(user: &str) -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut b = v["body"].clone();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(12);
    if !user.is_empty() { b["messages"] = json!([{"role": "user", "content": user}]); }
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
fn sdsnaps(d: &std::path::Path) -> usize {
    std::fs::read_dir(d).map(|r| r.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".sdsnap")).count()).unwrap_or(0)
}

#[test]
fn gateway_snapshots_tiny() {
    let dir = tmp("cache");
    let log = tmp("log.jsonl").display().to_string();
    let Some(port) = start(dir.clone(), log.clone(), 4) else { return };

    let b1 = body("");
    let r1 = post(port, &b1);
    let l = log_lines(&log);
    assert_eq!(l[0]["mode"], "full", "{}", l[0]);
    assert_eq!(kinds(&l[0]), ["save_stable", "save_roll"], "{}", l[0]);
    let n_prompt = l[0]["n_prompt"].as_u64().unwrap();
    let n_stable = l[0]["snap"][0]["n"].as_u64().unwrap();
    assert_eq!(l[0]["snap"][1]["n"].as_u64().unwrap(), n_prompt - 1);
    assert!(n_stable > 64 && n_stable < n_prompt - 1);
    assert_eq!(sdsnaps(&dir), 2);

    // identical re-send: the planner says full (the prompt is a prefix of the engine state), the rolling file restores
    let r2 = post(port, &b1);
    let l = log_lines(&log);
    assert_eq!(l[1]["mode"], "restore", "{}", l[1]);
    assert_eq!(l[1]["n_reused"].as_u64().unwrap(), n_prompt - 1);
    assert_eq!(l[1]["n_feed"].as_u64().unwrap(), 1);
    assert_eq!(kinds(&l[1]), ["restore"]);
    assert_eq!(r1, r2, "restored state must give the same greedy reply");

    // new conversation, same system + tools: restore of the stable prefix, no new stable save
    let _ = post(port, &body("Something else entirely."));
    let l = log_lines(&log);
    assert_eq!(l[2]["mode"], "restore", "{}", l[2]);
    assert_eq!(l[2]["n_reused"].as_u64().unwrap(), n_stable);
    assert_eq!(kinds(&l[2])[0], "restore");
    assert!(!kinds(&l[2]).contains(&"save_stable".to_string()));
    assert!(sdsnaps(&dir) <= 2, "one stable + one rolling");
    println!("  tiny: prompt {} tokens, stable {} tokens; restore {} ms; files {}", n_prompt, n_stable, l[1]["snap"][0]["ms"], sdsnaps(&dir));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&log);

    // cap: 1 file -> after the stable + rolling saves only the newest (rolling) is left
    let dir = tmp("cap");
    let log = tmp("cap.jsonl").display().to_string();
    let Some(port) = start(dir.clone(), log.clone(), 1) else { return };
    let _ = post(port, &b1);
    assert_eq!(sdsnaps(&dir), 1);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&log);
}
