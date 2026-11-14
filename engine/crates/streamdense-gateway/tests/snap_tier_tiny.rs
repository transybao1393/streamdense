//! Snapshot tiers (snap_tier.rs + snap::CacheHook::with_tiers) on the tiny model (models/tiny_qwen35). Temp dirs only.
//! 1. hook level: idle save + hot copy of the live state; another prompt in between; restore from hot == restore from
//!    warm == the in-memory continuation (token for token); a restart (new hook, same dir) restores warm; a queued
//!    request stops idle work; no hot copy when the footprint leaves no room under the cap; disk byte budget.
//! 2. gateway over HTTP (+ 27B tokenizer, ids folded): /health "snapshots", request-log "tiers" (idle work since the
//!    last job) and "snap[].tier"; per-conversation rolling files: after conversation B, conversation A's rolling state
//!    still restores (the legacy one-per-gateway rule deleted it).
//! 3. multi-user: switch save of the live conversation before another one replaces it (hook level: restore of it is
//!    "own" and == in memory; a new conversation restores the shared prefix, hot copy from idle); over HTTP: 3 users
//!    with the same system + tools interleaved A1 B1 C1 A2 B2 C2: cold, shared, shared, own, own, own ("prefix" field).
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use streamdense_gateway::client::request;
use streamdense_gateway::snap_tier::TierConfig;
use streamdense_gateway::{Backend, CacheHook, GatewayConfig, SnapshotHook, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_model::snapshot::SnapshotCache;
use streamdense_tokenizer::{Tokenizer, IM_END};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn tiny() -> Option<String> {
    let t = format!("{}/models/tiny_qwen35", ROOT);
    if !Path::new(&format!("{}/config.json", t)).exists() { eprintln!("SKIP: tiny model missing"); return None; }
    Some(t)
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd_snaptier_tiny_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

fn cfg() -> TierConfig {
    TierConfig { max_files: 8, max_bytes: 1 << 30, min_free: 0, idle: Some(Duration::from_millis(40)), idle_save: true, idle_save_min: 1, preread: true,
                 preread_files: 4, preread_bytes: 1 << 30, hot_bytes: 1 << 30, hot_headroom: 0, ram_cap: 1 << 40, per_conv: true, index: true,
                 hot_stable: true, switch_save: true, switch_save_min: 8 }
}

fn hook(dir: &Path, model: &str, c: TierConfig) -> CacheHook {
    CacheHook::with_tiers(SnapshotCache::open(dir.to_path_buf(), &CacheHook::model_id(model)).unwrap(), c)
}

/// Synthetic prompt with session markers: `sys` (70 ids) <|im_end|> `user` (10 ids) <|im_end|> filler up to n.
fn prompt(sys: u32, user: u32, n: usize) -> Vec<u32> {
    let mut t: Vec<u32> = (0..70).map(|i| (sys * 31 + i * 7) % 1500 + 3).collect();
    t.push(IM_END);
    t.extend((0..10).map(|i| (user * 17 + i * 11) % 1500 + 3));
    t.push(IM_END);
    let mut i = 0;
    while t.len() < n { t.push((i * 7919 + sys * 13 + user) % 1500 + 3); i += 1; }
    t
}

/// Feed (ids folded into the tiny vocab), decode n; the generated ids.
fn feed(e: &QwenEngine, ids: &[u32], n: usize) -> Vec<u32> {
    let v = e.cfg.vocab as u32;
    let f: Vec<u32> = ids.iter().map(|x| x % v).collect();
    let mut t = vec![];
    let s = Backend::feed(e, &f, n, &[], false, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(s.gen, t.len());
    t
}

/// Run the hook's idle steps as the worker would (wait, step) until it has none left.
fn run_idle(h: &mut CacheHook, e: &QwenEngine, fed: &[u32]) {
    let t0 = Instant::now();
    while let Some(d) = SnapshotHook::<QwenEngine>::idle_wait(h) {
        std::thread::sleep(d.max(Duration::from_millis(1)));
        SnapshotHook::<QwenEngine>::idle(h, e, Some(fed));
        assert!(t0.elapsed() < Duration::from_secs(30));
    }
    let t0 = Instant::now();
    while h.tiers.preread_running() && t0.elapsed() < Duration::from_secs(10) { std::thread::sleep(Duration::from_millis(5)); }
}

fn health(h: &mut CacheHook) -> Value { h.tiers.publish(); h.tiers.shared.health() }

#[test]
fn tiers_hook_tiny() {
    let Some(model) = tiny() else { return };
    let e = QwenEngine::open(&model, QwenOptions { max_ctx: 1024, ..Default::default() }).unwrap();
    let dir = tmp("hook");
    let mut h = hook(&dir, &model, cfg());

    // job A: prompt + 8 generated (last pending) -> live state = prompt + 7
    let pa = prompt(1, 2, 150);
    let _ = SnapshotHook::<QwenEngine>::job_begin(&mut h, Instant::now());
    e.reset();
    let ga = feed(&e, &pa, 8);
    let mut fed = pa.clone();
    fed.extend_from_slice(&ga[..ga.len() - 1]);
    assert_eq!(Backend::pos(&e), fed.len());
    SnapshotHook::<QwenEngine>::job_end(&mut h, &e, Some(&fed));

    // a request queued before the idle period: nothing happens
    h.tiers.shared.arrived();
    std::thread::sleep(Duration::from_millis(60));
    SnapshotHook::<QwenEngine>::idle(&mut h, &e, Some(&fed));
    assert_eq!(h.tiers.usage().0, 0, "idle work ran with a request queued");
    let v = SnapshotHook::<QwenEngine>::job_begin(&mut h, Instant::now()).unwrap();
    assert_eq!(v["idle_saves"], 0, "{}", v);
    SnapshotHook::<QwenEngine>::job_end(&mut h, &e, Some(&fed));

    // idle: save + hot copy of the live state
    run_idle(&mut h, &e, &fed);
    let s = health(&mut h);
    assert_eq!((s["idle"]["saves"].as_u64(), s["hot"]["copies"].as_u64(), s["warm"]["files"].as_u64()), (Some(1), Some(1), Some(1)), "{}", s);
    assert_eq!(s["live"]["n"].as_u64(), Some(fed.len() as u64), "{}", s);
    let v = SnapshotHook::<QwenEngine>::job_begin(&mut h, Instant::now()).unwrap();
    assert_eq!((v["idle_saves"].as_u64(), v["hot_admitted"].as_u64()), (Some(1), Some(1)), "{}", v);

    // reference: continue in memory with the pending token + 9 new ids
    let mut ext = vec![*ga.last().unwrap()];
    ext.extend((0..9).map(|i| 40 + i * 3));
    let want = feed(&e, &ext, 8);
    let mut all = fed.clone();
    all.extend_from_slice(&ext);

    // another prompt in between (other conversation)
    e.reset();
    let _ = feed(&e, &prompt(3, 4, 120), 4);
    SnapshotHook::<QwenEngine>::job_end(&mut h, &e, None);

    // restore: hot copy first
    e.reset();
    let _ = SnapshotHook::<QwenEngine>::job_begin(&mut h, Instant::now());
    let ev = SnapshotHook::<QwenEngine>::restore(&mut h, &e, &all[..all.len() - 1]).unwrap();
    assert!(ev.error.is_none(), "{:?}", ev.error);
    assert_eq!((ev.tier, ev.n, Backend::pos(&e)), ("hot", fed.len(), fed.len()));
    assert_eq!(feed(&e, &ext, 8), want, "hot restore + extension != in memory");
    SnapshotHook::<QwenEngine>::job_end(&mut h, &e, None);

    // without the hot copy: the warm file, same tokens
    h.tiers.drop_hot(0);
    e.reset();
    let ev = SnapshotHook::<QwenEngine>::restore(&mut h, &e, &all[..all.len() - 1]).unwrap();
    assert_eq!((ev.tier, ev.n, ev.error.is_none()), ("warm", fed.len(), true), "{:?}", ev.error);
    assert_eq!(feed(&e, &ext, 8), want, "warm restore + extension != in memory");
    let s = health(&mut h);
    assert_eq!((s["hits"]["hot"].as_u64(), s["hits"]["warm"].as_u64()), (Some(1), Some(1)), "{}", s);
    SnapshotHook::<QwenEngine>::job_end(&mut h, &e, None);
    drop(h);

    // restart: a new hook over the same dir restores warm; the index kept the hit count and kind
    let mut c2 = cfg();
    c2.ram_cap = 1; // footprint + copy > cap: no hot copy admitted
    let mut h2 = hook(&dir, &model, c2);
    let ent = h2.tiers.entries.values().find(|x| x.n == fed.len()).cloned().expect("idle file in the index");
    assert_eq!((ent.kind.as_str(), ent.hits), ("idle", 1));
    e.reset();
    let ev = SnapshotHook::<QwenEngine>::restore(&mut h2, &e, &all[..all.len() - 1]).unwrap();
    assert_eq!((ev.tier, ev.n), ("warm", fed.len()));
    assert_eq!(feed(&e, &ext, 8), want, "restart restore + extension != in memory");
    // live state now = fed ++ ext ++ 7 generated: the idle save writes it; the hot copy is refused (cap)
    let mut fed2 = all.clone();
    e.reset();
    let _ = SnapshotHook::<QwenEngine>::restore(&mut h2, &e, &all[..all.len() - 1]);
    let g2 = feed(&e, &ext, 8);
    fed2.extend_from_slice(&g2[..g2.len() - 1]);
    SnapshotHook::<QwenEngine>::job_end(&mut h2, &e, Some(&fed2));
    run_idle(&mut h2, &e, &fed2);
    let s = health(&mut h2);
    assert_eq!((s["idle"]["saves"].as_u64(), s["hot"]["copies"].as_u64(), s["hot_refused"].as_u64()), (Some(1), Some(0), Some(1)), "{}", s);
    // one conversation, two non-stable files kept (CONV_KEEP)
    assert_eq!(h2.tiers.usage().0, 2, "{}", s);

    // disk budget: room for one file only -> the next save keeps just the newest
    let one = h2.tiers.entries.values().map(|x| x.bytes).max().unwrap();
    h2.tiers.cfg.max_bytes = one + one / 2;
    e.reset();
    let pc = prompt(5, 6, 130);
    let _ = feed(&e, &pc, 1);
    let ev = SnapshotHook::<QwenEngine>::save(&mut h2, &e, &pc, false, None).unwrap();
    assert!(ev.error.is_none(), "{:?}", ev.error);
    assert_eq!(h2.tiers.usage().0, 1, "byte cap");
    assert!(h2.cache.path_for(&e, &pc).exists(), "the file just written is kept");
    println!("  tiny tiers: live {} tokens, file {:.2} MB; hot/warm/restart restores == in memory", fed.len(), one as f64 / 1e6);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- 2. over HTTP

fn start(dir: PathBuf, log: String) -> Option<u16> { start_cfg(dir, log, Duration::from_millis(300)) }

fn start_cfg(dir: PathBuf, log: String, idle: Duration) -> Option<u16> {
    let model = tiny()?;
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !Path::new(&format!("{}/tokenizer.json", tok)).exists() { eprintln!("SKIP: 27B tokenizer missing"); return None; }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(&tok).unwrap());
        let eng = QwenEngine::open(&model, QwenOptions { max_ctx: 4096, ..Default::default() }).unwrap();
        let vocab = eng.cfg.vocab as u32;
        let mut c = cfg();
        c.idle = Some(idle);
        let h: Box<dyn SnapshotHook<QwenEngine>> = Box::new(hook(&dir, &model, c));
        let gc = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: false, fold: Some(vocab) }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, gc, Some(h), |p| { tx.send(p).unwrap(); }).unwrap();
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

fn log_lines(p: &str) -> Vec<Value> { std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect() }

fn snaps_health(port: u16) -> Value { request(port, "GET", "/health", None).unwrap().json()["snapshots"].clone() }

/// Wait until the gateway's idle work is done (no pending steps, no pre-read running is not visible: steps only).
fn wait_idle(port: u16) -> Value {
    let t0 = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(100));
        let s = snaps_health(port);
        if s["idle"]["pending_steps"] == 0 || t0.elapsed() > Duration::from_secs(20) { return s; }
    }
}

#[test]
fn tiers_gateway_tiny() {
    let dir = tmp("http");
    let log = tmp("http.jsonl").display().to_string();
    let Some(port) = start(dir.clone(), log.clone()) else { return };
    let s = snaps_health(port);
    assert_eq!(s["warm"]["files"], 0, "{}", s);

    // A: full prefill, stable + rolling saves; then idle: the live state (prompt + reply) saved + hot copy
    let ba = body("");
    let ra = post(port, &ba);
    let s = wait_idle(port);
    // hot: the live state + the shared prefix (hot_stable)
    assert_eq!((s["idle"]["saves"].as_u64(), s["hot"]["copies"].as_u64()), (Some(1), Some(2)), "{}", s);
    assert_eq!(s["warm"]["files"], 3, "stable + roll + idle: {}", s);

    // B: other conversation, same system: restores the stable prefix (warm); B prefills > 256 tokens -> its own roll
    let long: String = (0..120).map(|i| format!("item {} ", i)).collect();
    let _ = post(port, &body(&format!("Please summarise this list: {}", long)));
    let l = log_lines(&log);
    assert_eq!(l[1]["mode"], "restore", "{}", l[1]);
    // the shared prefix: hot copy (hot_stable) from the idle period after A
    assert_eq!((l[1]["snap"][0]["tier"].as_str(), l[1]["prefix"].as_str()), (Some("hot"), Some("shared")), "{}", l[1]);
    assert_eq!(l[1]["tiers"]["idle_saves"], 1, "idle work since the last job: {}", l[1]);
    assert!(l[1]["snap"].as_array().unwrap().iter().any(|e| e["kind"] == "save_roll"), "B saved its rolling state: {}", l[1]);

    // A again right away (no idle gap): A's rolling file survived B's (per conversation) -> restore of prompt - 1
    let ra2 = post(port, &ba);
    let l = log_lines(&log);
    let n_prompt = l[0]["n_prompt"].as_u64().unwrap();
    assert_eq!(l[2]["mode"], "restore", "{}", l[2]);
    assert_eq!(l[2]["n_reused"].as_u64(), Some(n_prompt - 1), "{}", l[2]);
    assert_eq!(l[2]["tiers"]["idle_saves"], 0, "no idle work between back-to-back requests: {}", l[2]);
    assert!(l[2]["tiers"]["blocked_ms"].is_number(), "{}", l[2]);
    assert_eq!(ra, ra2, "restored state must give the same greedy reply");

    let s = wait_idle(port);
    assert!(s["hits"]["warm"].as_u64().unwrap() + s["hits"]["hot"].as_u64().unwrap() >= 2, "{}", s);
    assert!(s["warm"]["files"].as_u64().unwrap() <= 8, "{}", s);
    assert!(dir.join("tiers.json").exists(), "index written");
    println!("  tiny gateway tiers: {}", s);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&log);
}

// ---- 3. multi-user

#[test]
fn switch_save_hook_tiny() {
    let Some(model) = tiny() else { return };
    let e = QwenEngine::open(&model, QwenOptions { max_ctx: 1024, ..Default::default() }).unwrap();
    let dir = tmp("switch");
    let mut h = hook(&dir, &model, cfg());
    // the shared prefix (system turn) of project 1
    let pa = prompt(1, 2, 150);
    e.reset();
    let _ = feed(&e, &pa[..71], 1);
    let ev = SnapshotHook::<QwenEngine>::save(&mut h, &e, &pa[..71], true, None).unwrap();
    assert!(ev.error.is_none(), "{:?}", ev.error);
    // user A: prompt + 8 generated -> live state
    e.reset();
    let ga = feed(&e, &pa, 8);
    let mut fed = pa.clone();
    fed.extend_from_slice(&ga[..ga.len() - 1]);
    // the same conversation next, or too little new: no switch save
    assert!(SnapshotHook::<QwenEngine>::switch(&mut h, &e, &fed, &prompt(1, 2, 200)).is_none(), "same conversation");
    h.tiers.cfg.switch_save_min = 10_000;
    assert!(SnapshotHook::<QwenEngine>::switch(&mut h, &e, &fed, &prompt(1, 3, 140)).is_none(), "below switch_save_min");
    h.tiers.cfg.switch_save_min = 8;
    // user B arrives: A's live state is saved first
    let ev = SnapshotHook::<QwenEngine>::switch(&mut h, &e, &fed, &prompt(1, 3, 140)).expect("switch save");
    assert_eq!((ev.kind, ev.n, ev.error.is_none()), ("save_switch", fed.len(), true), "{:?}", ev.error);
    assert!(SnapshotHook::<QwenEngine>::switch(&mut h, &e, &fed, &prompt(1, 3, 140)).is_none(), "already saved");
    assert!(h.tiers.entries.values().any(|x| x.kind.as_str() == "switch" && x.n == fed.len()));
    // reference: A continues in memory
    let mut ext = vec![*ga.last().unwrap()];
    ext.extend((0..9).map(|i| 50 + i * 5));
    let want = feed(&e, &ext, 8);
    let mut all = fed.clone();
    all.extend_from_slice(&ext);
    // B runs (its own state), then A again: restores its own switch file, same continuation
    e.reset();
    let _ = feed(&e, &prompt(1, 3, 140), 4);
    e.reset();
    let ev = SnapshotHook::<QwenEngine>::restore(&mut h, &e, &all[..all.len() - 1]).unwrap();
    assert_eq!((ev.tier, ev.src, ev.n, ev.error.is_none()), ("warm", "own", fed.len(), true), "{:?}", ev.error);
    assert_eq!(feed(&e, &ext, 8), want, "switch restore + extension != in memory");
    // a new user C of the same project: the shared prefix
    e.reset();
    let pc = prompt(1, 5, 120);
    let ev = SnapshotHook::<QwenEngine>::restore(&mut h, &e, &pc[..pc.len() - 1]).unwrap();
    assert_eq!((ev.tier, ev.src, ev.n), ("warm", "shared", 71));
    // idle with A live: hot copies of A's state and of the shared prefix; C then restores the prefix from RAM
    SnapshotHook::<QwenEngine>::job_end(&mut h, &e, Some(&fed));
    e.reset();
    let _ = feed(&e, &pa, 8);
    run_idle(&mut h, &e, &fed);
    let s = health(&mut h);
    assert_eq!(s["hot"]["copies"].as_u64(), Some(2), "{}", s);
    assert_eq!(s["switch_saves"].as_u64(), Some(1), "{}", s);
    e.reset();
    let ev = SnapshotHook::<QwenEngine>::restore(&mut h, &e, &pc[..pc.len() - 1]).unwrap();
    assert_eq!((ev.tier, ev.src, ev.n, ev.error.is_none()), ("hot", "shared", 71, true), "{:?}", ev.error);
    // no room next to the other copies: the shared prefix is not copied (never drops another copy for it)
    let mut c2 = cfg();
    c2.hot_bytes = h.tiers.hot.iter().map(|x| x.bytes).max().unwrap();
    drop(h);
    let mut h2 = hook(&dir, &model, c2);
    SnapshotHook::<QwenEngine>::job_end(&mut h2, &e, Some(&fed));
    e.reset();
    let _ = feed(&e, &pa, 8);
    run_idle(&mut h2, &e, &fed);
    let s = health(&mut h2);
    assert_eq!(s["hot"]["copies"].as_u64(), Some(1), "only the live state: {}", s);
    let _ = std::fs::remove_dir_all(&dir);
}

fn turn(port: u16, b: &Value) -> Value {
    let r = request(port, "POST", "/v1/messages", Some(b)).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()["content"].clone()
}

#[test]
fn multi_user_gateway_tiny() {
    let dir = tmp("multi");
    let log = tmp("multi.jsonl").display().to_string();
    // idle work only after 30 s: back-to-back interleaved users, no idle saves in between
    let Some(port) = start_cfg(dir.clone(), log.clone(), Duration::from_secs(30)) else { return };
    let users = ["Explain what a hash map is.", "Write a haiku about rust.", "List three prime numbers."];
    let mut convs: Vec<Value> = users.iter().map(|u| { let mut b = body(""); b["messages"] = json!([{"role": "user", "content": u}]); b }).collect();
    for round in 0..2 {
        for (i, b) in convs.iter_mut().enumerate() {
            let content = turn(port, b);
            let m = b["messages"].as_array_mut().unwrap();
            m.push(json!({"role": "assistant", "content": content}));
            m.push(json!({"role": "user", "content": format!("Round {} for user {}: go on, more detail please.", round + 2, i)}));
        }
    }
    let l = log_lines(&log);
    let pre: Vec<&str> = l.iter().map(|x| x["prefix"].as_str().unwrap_or("?")).collect();
    for x in &l { println!("  {} n_prompt {} reused {} fed {} snap {}", x["prefix"], x["n_prompt"], x["n_reused"], x["n_feed"], x["snap"]); }
    assert_eq!(&pre[..3], &["cold", "shared", "shared"], "{:?}", pre);
    assert_eq!(&pre[3..], &["own", "own", "own"], "{:?}", pre);
    // B1 and C1 each saved the previous user's live state before replacing it; round 2 reuses more than the prefix
    for k in [1, 2, 3, 4, 5] {
        assert!(l[k]["snap"].as_array().unwrap().iter().any(|e| e["kind"] == "save_switch" && e["error"].is_null()), "switch save in {}: {}", k, l[k]);
    }
    let stable = l[1]["n_reused"].as_u64().unwrap();
    for k in [3, 4, 5] { assert!(l[k]["n_reused"].as_u64().unwrap() > stable, "{}", l[k]); }
    let s = snaps_health(port);
    assert!(s["switch_saves"].as_u64().unwrap() >= 5, "{}", s);

    // lossless: user 1 alone on a fresh gateway (cold, then followup on the live state) gives the same replies as
    // when interleaved with the others (shared prefix restore, then its parked state restored)
    let dir2 = tmp("multi_alone");
    let log2 = tmp("multi_alone.jsonl").display().to_string();
    let port2 = start_cfg(dir2.clone(), log2.clone(), Duration::from_secs(30)).unwrap();
    let b = &convs[1];
    let m = b["messages"].as_array().unwrap();
    let mut solo = body("");
    solo["messages"] = json!([m[0].clone()]);
    let r1 = turn(port2, &solo);
    assert_eq!(r1, m[1]["content"], "turn 1 alone != interleaved");
    solo["messages"] = json!([m[0].clone(), m[1].clone(), m[2].clone()]);
    let r2 = turn(port2, &solo);
    assert_eq!(r2, m[3]["content"], "turn 2 alone != interleaved");
    let l2 = log_lines(&log2);
    assert_eq!((l2[0]["prefix"].as_str(), l2[1]["prefix"].as_str()), (Some("cold"), Some("kv")), "{:?}", l2);
    for d in [&dir, &dir2] { let _ = std::fs::remove_dir_all(d); }
    for f in [&log, &log2] { let _ = std::fs::remove_file(f); }
}
