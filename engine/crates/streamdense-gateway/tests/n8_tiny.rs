//! N8 on the tiny model (models/tiny_qwen35, ids folded) + the 27B tokenizer, over HTTP, with the coding-agent fixture
//! (system prompt + tools), snapshot tiers over a fake free-disk figure (`Tiers::fake_free`) held at the
//! `min_free_disk_gb` floor, and "junk" files (other conversations' old states, sparse) that eviction first may delete.
//! Each scenario runs with the N8 keys on (`evict_for_save`, `stable_ensure`, `ckpt_system`, as the coding profile
//! sets them) and off (built-in); the off run reproduces the logged failure and is printed for comparison.
//! 1. side request then return: a warm-up of the main chat (its rolling file removed: the save the floor skipped in
//!    the logged session), the disk at the floor; a short side request (other system turn, no tools) replaces the live
//!    state. On: the switch save evicts junk first and saves the main chat, which comes back with a parked restore of
//!    its own file (mode "restore", n_reused >= the warm-up's end), its reply == a from-zero gateway's (exact prefill).
//!    Off: the switch save is skipped by the floor; the return re-reads from the shared prefix.
//! 2. tools change + compaction: R0 with 2 tools (warm-up), disk at the floor; R1 with 3 tools (a new system + tools
//!    prefix): on, its stable prefix file is saved (eviction first) and the system checkpoint is taken; R2 compacts the
//!    history (same tools, a new first message): reused >= the system + tools boundary (rewind to the system checkpoint
//!    or the stable file), reply == from zero; R3 a side request; R4 compacts again: on, restored from the stable file
//!    (>= the boundary), off: no file of that prefix (full from 0, the logged "snap best 0").
//! Run: cargo test --release -p streamdense-gateway --test n8_tiny -- --nocapture --test-threads 1
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::client::request;
use streamdense_gateway::{CacheHook, GatewayConfig, SnapshotHook, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_model::snapshot::SnapshotCache;
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");
/// the fake disk floor (min_free) and the free figure before the floor is applied
const FLOOR: u64 = 100_000_000_000;
const PLENTY: u64 = 1_000_000_000_000_000;

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("sd_n8_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn fixture() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    v["body"].clone()
}

/// The fixture's system + `tools` (None = all), these messages.
fn body(msgs: &Value, tools: Option<usize>, max_tokens: u64) -> Value {
    let mut b = fixture();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(max_tokens);
    b["messages"] = msgs.clone();
    if let Some(n) = tools { let t: Vec<Value> = b["tools"].as_array().unwrap()[..n].to_vec(); b["tools"] = json!(t); }
    b
}

/// The side request: another system turn, no tools, one user message.
fn side() -> Value {
    let text: String = (0..40).map(|i| format!("Line {}: the build failed with error E{} in module m{}.\n", i, 300 + i, i % 7)).collect();
    json!({"model": "tiny", "max_tokens": 8, "stream": false, "system": "Summarize the text in one line.", "messages": [{"role": "user", "content": text}]})
}

/// A gateway: 4 rewind checkpoints, exact prefill, snapshot tiers in `dir` over the fake disk `free` (N8 keys `on`);
/// `snap` false = no snapshots (the from-zero reference).
fn start(dir: Option<std::path::PathBuf>, free: Arc<AtomicU64>, on: bool, log: String) -> Option<u16> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer missing");
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let cfg = streamdense_model::QwenConfig::from_model_dir(&tiny).unwrap();
        let rb = 4 * streamdense_model::rewind::ckpt_bytes(&cfg) as u64;
        let eng = QwenEngine::open(&tiny, QwenOptions { max_ctx: 4096, mtp: true, prefill_exact: true, rewind_bytes: rb, ..Default::default() }).unwrap();
        assert_eq!(eng.ckpt_slots(), 4);
        let vocab = eng.cfg.vocab as u32;
        let hook: Option<Box<dyn SnapshotHook<QwenEngine>>> = dir.map(|dir| {
            let cache = SnapshotCache::open(dir, &CacheHook::model_id(&tiny)).unwrap();
            let mut tc = streamdense_gateway::snap_tier::TierConfig::legacy(64, 1 << 40);
            tc.per_conv = true; tc.index = true; tc.switch_save = true; tc.switch_save_min = 8; tc.min_free = FLOOR;
            tc.evict_for_save = on; tc.evict_for_save_min = 256; tc.stable_ensure = on;
            let mut h = CacheHook::with_tiers(cache, tc);
            h.tiers.fake_free = Some(free);
            Box::new(h) as Box<dyn SnapshotHook<QwenEngine>>
        });
        let gc = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab), ckpt_system: on, ..Default::default() }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, gc, hook, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(120)).unwrap())
}

fn post(port: u16, b: &Value) -> Value {
    let r = request(port, "POST", "/v1/messages", Some(b)).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()
}

/// .sdsnap files in `dir`: (path, token count from the name `...-{n}.sdsnap`, bytes).
fn files(dir: &std::path::Path) -> Vec<(std::path::PathBuf, usize, u64)> {
    std::fs::read_dir(dir).unwrap().flatten().filter_map(|e| {
        let n = e.file_name().to_string_lossy().to_string();
        let k = n.strip_suffix(".sdsnap")?.rsplit('-').next()?.parse().ok()?;
        Some((e.path(), k, e.metadata().ok()?.len()))
    }).collect()
}

/// `k` sparse junk files of `bytes` each (other conversations' old states: no header, kind unknown).
fn junk(dir: &std::path::Path, k: usize, bytes: u64) {
    for i in 0..k { std::fs::File::create(dir.join(format!("junk{:02}.sdsnap", i))).unwrap().set_len(bytes).unwrap(); }
}

fn junk_left(dir: &std::path::Path) -> usize { std::fs::read_dir(dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("junk")).count() }

fn snap_ev(x: &Value, kind: &str) -> Option<Value> { x["snap"].as_array().unwrap().iter().find(|e| e["kind"] == kind).cloned() }

fn show(what: &str, x: &Value) {
    println!("  {:34} mode {:8} reused {:5} fed {:5} of {:5} | {} | snap {:?}", what, x["mode"].as_str().unwrap_or(""), x["n_reused"], x["n_feed"], x["n_prompt"],
             x["reuse_why"].as_str().unwrap_or(""), x["snap"].as_array().unwrap().iter().map(|e| format!("{}:{}{}", e["kind"].as_str().unwrap(), e["n"], if e["error"].is_null() { "" } else { "!" })).collect::<Vec<_>>());
}

fn main_hist() -> Value {
    let pad: String = (0..30).map(|i| format!("fn h{}(v: &[u8]) -> usize {{ v.len() * {} }}\n", i, i % 5 + 1)).collect();
    json!([
        {"role": "user", "content": format!("We are building a small command line tool in Rust. Review this file first:\n{}", pad)},
        {"role": "assistant", "content": "The helpers multiply the length; rename them and add tests for empty input."},
        {"role": "user", "content": "Rename h0..h29 to scale_0..scale_29 and add one test per function."},
    ])
}

#[test]
fn n8_side_request_then_return_tiny() {
    let main = main_hist();
    let mut back = main.as_array().unwrap().clone();
    back.push(json!({"role": "assistant", "content": "Renamed; the tests are in tests/scale.rs."}));
    back.push(json!({"role": "user", "content": "Now run the tests and fix what fails."}));
    let back = Value::Array(back);
    let mut reused = [0u64; 2];
    for on in [true, false] {
        let tag = if on { "on" } else { "off" };
        let dir = tmp(&format!("side_cache_{}", tag));
        let log = tmp(&format!("side_log_{}.jsonl", tag)).display().to_string();
        let free = Arc::new(AtomicU64::new(PLENTY));
        let Some(port) = start(Some(dir.clone()), free.clone(), on, log.clone()) else { return };
        // 1. warm-up of the main chat: stable prefix + its end
        let w = post(port, &body(&main, None, 0));
        assert_eq!(w["streamdense"]["warm"], "done", "{}", w);
        let n_w = w["streamdense"]["n_prompt"].as_u64().unwrap() as usize;
        let l = log_lines(&log);
        show(&format!("{} warm-up main", tag), &l[0]);
        let s = snap_ev(&l[0], "save_stable").expect("stable prefix saved")["n"].as_u64().unwrap() as usize;
        // the main chat's own file is not on disk (the logged roll save skipped by the floor); the disk is at the floor
        let fs = files(&dir);
        let big = fs.iter().map(|f| f.2).max().unwrap();
        for (p, n, _) in &fs { if *n != s { std::fs::remove_file(p).unwrap(); } }
        junk(&dir, 16, big);
        free.store(FLOOR, Ordering::SeqCst);
        // 2. the side request replaces the live state
        post(port, &side());
        let x = log_lines(&log)[1].clone();
        show(&format!("{} side request", tag), &x);
        let sw = snap_ev(&x, "save_switch");
        if on {
            assert!(sw.as_ref().is_some_and(|e| e["error"].is_null()), "eviction first: the switch save is not skipped: {}", x);
            assert!(junk_left(&dir) < 16, "junk evicted");
            assert!(files(&dir).iter().any(|f| f.1 == s), "the main chat's stable prefix is kept");
        } else {
            // built-in: skipped by the floor, or not tried (the index still lists the removed file)
            assert!(sw.is_none_or(|e| e["error"].as_str().unwrap_or("").contains("disk low")), "built-in: no switch file: {}", x);
        }
        // 3. the main chat comes back
        let r = post(port, &body(&back, None, 12));
        let x = log_lines(&log)[2].clone();
        show(&format!("{} main returns", tag), &x);
        assert!(x["error"].is_null(), "{}", x);
        reused[on as usize] = x["n_reused"].as_u64().unwrap();
        if on {
            assert_eq!(x["mode"], "restore", "{}", x);
            assert!(x["n_reused"].as_u64().unwrap() as usize >= n_w - 1, "restored from its own file (warm-up end {}): {}", n_w, x);
            assert!(x["reuse_why"].as_str().unwrap().contains("own"), "{}", x);
            // == a from-zero gateway (exact prefill)
            let log_z = tmp("side_log_zero.jsonl").display().to_string();
            let Some(pz) = start(None, Arc::new(AtomicU64::new(PLENTY)), false, log_z.clone()) else { return };
            let z = post(pz, &body(&back, None, 12));
            assert_eq!(log_lines(&log_z)[0]["mode"], "full");
            assert_eq!(r["content"], z["content"], "the restored main chat's reply != from zero");
            let _ = std::fs::remove_file(&log_z);
        } else {
            assert!(x["n_reused"].as_u64().unwrap() as usize <= s, "built-in: only the shared prefix: {}", x);
        }
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&log);
    }
    println!("  main chat return reused: on {} / off {}", reused[1], reused[0]);
    assert!(reused[1] > reused[0]);
}

#[test]
fn n8_tools_change_and_compaction_tiny() {
    let main = main_hist();
    let compact = |k: usize| json!([{"role": "user", "content": format!("Summary of the work so far (compacted, version {}): the helpers were renamed to scale_*, tests were added, two failed on empty input. Continue: fix the failing tests.", k)}]);
    let mut at_r4 = [0u64; 2];
    for on in [true, false] {
        let tag = if on { "on" } else { "off" };
        let dir = tmp(&format!("tools_cache_{}", tag));
        let log = tmp(&format!("tools_log_{}.jsonl", tag)).display().to_string();
        let free = Arc::new(AtomicU64::new(PLENTY));
        let Some(port) = start(Some(dir.clone()), free.clone(), on, log.clone()) else { return };
        // R0: 2 tools (warm-up); then the disk sits at the floor with junk to evict
        let w = post(port, &body(&main, Some(2), 0));
        assert_eq!(w["streamdense"]["warm"], "done", "{}", w);
        show(&format!("{} R0 warm-up, 2 tools", tag), &log_lines(&log)[0]);
        let big = files(&dir).iter().map(|f| f.2).max().unwrap();
        junk(&dir, 16, big);
        free.store(FLOOR, Ordering::SeqCst);
        // R1: 3 tools: a new system + tools prefix
        post(port, &body(&main, None, 8));
        let x = log_lines(&log)[1].clone();
        show(&format!("{} R1 3 tools", tag), &x);
        let st = snap_ev(&x, "save_stable").expect("a stable save of the new prefix was tried");
        let s = st["n"].as_u64().unwrap();
        if on {
            assert!(st["error"].is_null(), "stable_ensure: the new prefix is saved (eviction first): {}", x);
            assert!(x["ckpts"].as_array().unwrap().iter().any(|c| c["p"] == s && c["sys"] == true), "the system checkpoint: {}", x);
        } else {
            assert!(st["error"].as_str().unwrap_or("").contains("disk low"), "built-in: skipped by the floor: {}", x);
        }
        // R2: compacted history, same tools: rewind / restore at the system + tools boundary
        let r2 = post(port, &body(&compact(1), None, 8));
        let x = log_lines(&log)[2].clone();
        show(&format!("{} R2 compaction", tag), &x);
        assert!(x["error"].is_null(), "{}", x);
        if on {
            assert!(x["n_reused"].as_u64().unwrap() >= s, "reuse >= the system + tools boundary {}: {}", s, x);
            assert!(x["mode"] == "rewind" || x["mode"] == "restore", "{}", x);
        }
        // R3: a side request (another lineage); R4: compaction again
        post(port, &side());
        show(&format!("{} R3 side request", tag), &log_lines(&log)[3]);
        let r4 = post(port, &body(&compact(2), None, 8));
        let x = log_lines(&log)[4].clone();
        show(&format!("{} R4 compaction after side", tag), &x);
        assert!(x["error"].is_null(), "{}", x);
        at_r4[on as usize] = x["n_reused"].as_u64().unwrap();
        if on {
            assert!(x["n_reused"].as_u64().unwrap() >= s, "the stable prefix file (>= boundary {}): {}", s, x);
            // == from zero (exact prefill): a fresh gateway per request
            for (k, r) in [(1, &r2), (2, &r4)] {
                let log_z = tmp(&format!("tools_log_zero{}.jsonl", k)).display().to_string();
                let Some(pz) = start(None, Arc::new(AtomicU64::new(PLENTY)), false, log_z.clone()) else { return };
                let z = post(pz, &body(&compact(k), None, 8));
                assert_eq!(log_lines(&log_z)[0]["mode"], "full");
                assert_eq!(r["content"], z["content"], "compaction {} != from zero", k);
                let _ = std::fs::remove_file(&log_z);
            }
        } else {
            assert_eq!(x["n_reused"].as_u64(), Some(0), "built-in: no file of the new prefix (snap best 0): {}", x);
        }
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&log);
    }
    println!("  R4 (compaction after a side request) reused: on {} / off {}", at_r4[1], at_r4[0]);
}
