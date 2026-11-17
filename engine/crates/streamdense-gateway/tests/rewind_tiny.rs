//! Rewind checkpoints in the gateway (worker.rs `try_rewind`, engine rewind.rs) on the tiny model (models/tiny_qwen35,
//! ids folded) + the 27B tokenizer, over HTTP, with the coding-agent fixture (system prompt + tools):
//! 1. a warm-up of a chat (W) takes a checkpoint at its end (snap kind "ckpt");
//! 2. the next turn (W + a user message U3) reuses the live state ("prefix");
//! 3. a later request rewrites that user message (U3 -> U3', then a reply and one more turn): the planner says full,
//!    the worker rewinds the live state to the checkpoint (mode "rewind", n_reused = W's end, snap "rewind", prefix
//!    "kv") instead of a reset + snapshot restore, and re-reads only the rest;
//! 4. its reply == a gateway without checkpoints (same snapshot hook, so the warm-up reads in the same pieces) that
//!    runs W and then that request (the same continuation, no detour); with the exact prefill also == a cold gateway's
//!    one-shot prefill of the whole prompt (the fast prefill's result depends on where pieces start: printed only).
//! Run: cargo test --release -p streamdense-gateway --test rewind_tiny -- --nocapture --test-threads 1
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::client::request;
use streamdense_gateway::{CacheHook, GatewayConfig, SnapshotHook, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_model::snapshot::SnapshotCache;
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("sd_gwrewind_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn body(msgs: &Value, max_tokens: u64) -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut b = v["body"].clone();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(max_tokens);
    b["messages"] = msgs.clone();
    b
}

/// A gateway on the tiny model: `rewind` = 4 checkpoints, `snap` = a snapshot cache dir (CacheHook), `exact` prefill.
fn start(rewind: bool, snap: Option<std::path::PathBuf>, exact: bool, log: String) -> Option<u16> { start_rs(rewind, snap, exact, log, false) }

/// `start` with `reply_start` = `WorkerConfig::ckpt_reply_start` (`[gateway] ckpt_reply_start`).
fn start_rs(rewind: bool, snap: Option<std::path::PathBuf>, exact: bool, log: String, reply_start: bool) -> Option<u16> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer missing");
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let cfg = streamdense_model::QwenConfig::from_model_dir(&tiny).unwrap();
        let rb = if rewind { 4 * streamdense_model::rewind::ckpt_bytes(&cfg) as u64 } else { 0 };
        let eng = QwenEngine::open(&tiny, QwenOptions { max_ctx: 4096, mtp: true, prefill_exact: exact, rewind_bytes: rb, ..Default::default() }).unwrap();
        assert_eq!(eng.ckpt_slots(), if rewind { 4 } else { 0 });
        let vocab = eng.cfg.vocab as u32;
        let hook: Option<Box<dyn SnapshotHook<QwenEngine>>> = snap.map(|dir| {
            let cache = SnapshotCache::open(dir, &CacheHook::model_id(&tiny)).unwrap();
            // legacy tiers + the switch save (a chat switched away from is saved and parked, as in the gateway config)
            let mut tc = streamdense_gateway::snap_tier::TierConfig::legacy(16, 1 << 30);
            tc.switch_save = true;
            Box::new(CacheHook::with_tiers(cache, tc)) as Box<dyn SnapshotHook<QwenEngine>>
        });
        let gc = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab), ckpt_reply_start: reply_start, ..Default::default() }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, gc, hook, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(120)).unwrap())
}

fn post(port: u16, b: &Value) -> Value {
    let r = request(port, "POST", "/v1/messages", Some(b)).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()
}

fn kinds(l: &Value) -> Vec<String> { l["snap"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect() }

#[test]
fn rewind_rewritten_message_tiny() {
    let u3 = "Now read src/main.rs and tell me which function parses the command line arguments, and whether it handles --help.";
    let u3b = "Now read src/lib.rs and tell me which function parses the command line arguments, and whether it handles --help.";
    let hist = json!([
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "Hi! What should we work on today?"},
        {"role": "user", "content": "We are building a small command line tool in Rust that downloads files over HTTP with resume support. First, list what modules we need."},
        {"role": "assistant", "content": "We need: a CLI argument parser, an HTTP client wrapper with range requests, a resume state file, a progress reporter, and an error type."},
    ]);
    let with = |extra: Value| { let mut v = hist.as_array().unwrap().clone(); v.extend(extra.as_array().unwrap().iter().cloned()); Value::Array(v) };
    let r1 = with(json!([{"role": "user", "content": u3}]));
    // the client rewrote the earlier user message u3 (and the chat went on)
    let r2 = with(json!([{"role": "user", "content": u3b}, {"role": "assistant", "content": "It is parse_args."}, {"role": "user", "content": "Add a --quiet flag."}]));

    for exact in [false, true] {
        let tag = if exact { "exact" } else { "fast" };
        let dir = tmp(&format!("cache_{}", tag));
        let log = tmp(&format!("log_{}.jsonl", tag)).display().to_string();
        let Some(port) = start(true, Some(dir.clone()), exact, log.clone()) else { return };
        // 1. warm-up of the chat so far: a checkpoint at its end
        let w = post(port, &body(&hist, 0));
        assert_eq!(w["streamdense"]["warm"], "done", "{}", w);
        let n_w = w["streamdense"]["n_prompt"].as_u64().unwrap() as usize;
        let l = log_lines(&log);
        assert!(kinds(&l[0]).contains(&"ckpt".to_string()), "warm-up took no checkpoint: {}", l[0]);
        // 2. the next turn: prefix from the live state
        post(port, &body(&r1, 12));
        let l = log_lines(&log);
        assert_eq!(l[1]["mode"], "prefix", "{}", l[1]);
        // 3. the rewrite: rewind to the warm-up's end
        let rw = post(port, &body(&r2, 16));
        let l = log_lines(&log);
        assert_eq!(l[2]["mode"], "rewind", "{}", l[2]);
        assert_eq!(l[2]["prefix"], "kv", "{}", l[2]);
        // the warm-up's end (n_w - 1) or, later, R1's last-user-message start (n_w: the warm-up's pending "\n" fed)
        let n_rw = l[2]["n_reused"].as_u64().unwrap() as usize;
        assert!(n_rw >= n_w - 1, "{}", l[2]);
        let ev = l[2]["snap"].as_array().unwrap().iter().find(|e| e["kind"] == "rewind").cloned().unwrap_or(Value::Null);
        assert!(ev["error"].is_null() && ev["n"].as_u64() == Some(n_rw as u64), "{}", l[2]);
        assert!(!kinds(&l[2]).contains(&"restore".to_string()), "no snapshot restore: {}", l[2]);
        // diagnostics (2026-11-17): the reason names the checkpoint, the divergence and the checkpoint list are logged
        assert_eq!(l[1]["reuse_why"], "append", "{}", l[1]);
        assert!(l[2]["reuse_why"].as_str().unwrap().starts_with(&format!("rewind ckpt@{}", n_rw)), "{}", l[2]);
        assert_eq!(l[2]["div"]["ckpt"].as_u64().map(|_| true), Some(true), "{}", l[2]);
        assert!(l[2]["div"]["live"].as_u64().unwrap() >= n_rw as u64 && l[2]["div"]["msg"].is_number(), "{}", l[2]);
        let cks = l[2]["ckpts"].as_array().expect("checkpoint list");
        assert!(!cks.is_empty() && cks.iter().all(|c| c["p"].is_u64() && c["pin"].is_boolean() && c["live"].is_boolean()), "{}", l[2]);
        assert!(l[2]["timing"]["rewind_ms"].as_f64().unwrap() > 0.0 && l[2]["ram"]["footprint_gb"].as_f64().unwrap() > 0.0, "{}", l[2]);
        let n_prompt = l[2]["n_prompt"].as_u64().unwrap() as usize;

        // 4a. the same continuation without the detour: warm-up, then the request (prefix)
        let log_c = tmp(&format!("logc_{}.jsonl", tag)).display().to_string();
        let dir_c = tmp(&format!("cache_c_{}", tag));
        let Some(pc) = start(false, Some(dir_c.clone()), exact, log_c.clone()) else { return };
        post(pc, &body(&hist, 0));
        let cont = post(pc, &body(&r2, 16));
        assert_eq!(log_lines(&log_c)[1]["mode"], "prefix");
        // 4b. cold: one prefill of the whole prompt
        let log_f = tmp(&format!("logf_{}.jsonl", tag)).display().to_string();
        let Some(pf) = start(false, None, exact, log_f.clone()) else { return };
        let cold = post(pf, &body(&r2, 16));
        assert_eq!(log_lines(&log_f)[0]["mode"], "full");
        let (eq_c, eq_f) = (rw["content"] == cont["content"], rw["content"] == cold["content"]);
        println!("  {}: prompt {} tokens, warm-up {}, rewound to {} (fed {}), rewind {:.1} ms | reply == continuation {} | == cold one-shot {}",
                 tag, n_prompt, n_w, n_rw, l[2]["n_feed"], ev["ms"].as_f64().unwrap_or(0.0), eq_c, eq_f);
        assert!(eq_c, "{}: rewind reply != the same continuation: {} vs {}", tag, rw["content"], cont["content"]);
        if exact { assert!(eq_f, "{}: exact prefill: rewind reply != cold: {} vs {}", tag, rw["content"], cold["content"]); }
        for d in [&dir, &dir_c] { let _ = std::fs::remove_dir_all(d); }
        for f in [&log, &log_c, &log_f] { let _ = std::fs::remove_file(f); }
    }
}

/// A real client pattern: each request appends a user message ending with a "\n[state]\n..." block and
/// strips that block from the previous user message, so every request diverges inside the previous user message, near
/// its end (planner: full). The worker cuts each request's prefill at the start of its last user message (render-aware)
/// and keeps that checkpoint pinned, so from the third request on (k >= 2; the first user message starts right after
/// the short system turn, below CKPT_MIN) every request rewinds there (mode "rewind", n_reused >= the
/// previous request's last-user start) and re-reads only that message and the newer ones. Replies are compared with
/// a gateway without checkpoints or snapshots that prefills every request from zero: equal with the exact prefill
/// (asserted), printed with the fast one.
#[test]
fn rewind_client_state_block_tiny() {
    let src: String = (0..400).map(|i| format!("fn f{}(x: u32) -> u32 {{ x * {} + {} }}\n", i, i % 7 + 1, i % 13)).collect();
    let pieces: Vec<&str> = (0..7).map(|k| &src[k * 600..(k + 1) * 600]).collect();
    for exact in [false, true] {
        let tag = if exact { "exact" } else { "fast" };
        let dir = tmp(&format!("client_cache_{}", tag));
        let log = tmp(&format!("client_log_{}.jsonl", tag)).display().to_string();
        let log_c = tmp(&format!("client_logc_{}.jsonl", tag)).display().to_string();
        let Some(port) = start(true, Some(dir.clone()), exact, log.clone()) else { return };
        let Some(pc) = start(false, None, exact, log_c.clone()) else { return };
        let mut hist: Vec<Value> = vec![];
        let (mut same, mut prev_user) = (0usize, 0usize);
        for (k, piece) in pieces.iter().enumerate() {
            for m in hist.iter_mut() {
                if m["role"] == "user" {
                    let c = m["content"].as_str().unwrap().to_string();
                    if let Some(i) = c.find("\n[state]\n") { m["content"] = json!(c[..i].to_string()); }
                }
            }
            let state = format!("\n[state]\nstep: {}\ngoal: review the file piece by piece\ndone: pieces 0..{}\n", k, k as i64 - 1);
            hist.push(json!({"role": "user", "content": format!("Piece {}:\n```\n{}\n```\nSummarize this piece in one line.{}", k, piece, state)}));
            let b = json!({"model": "tiny", "max_tokens": 8, "stream": false, "system": "You are a coding assistant. Answer briefly.", "messages": hist.clone()});
            let r = post(port, &b);
            let c = post(pc, &b);
            let l = log_lines(&log);
            let x = &l[k];
            let ck: Vec<u64> = x["snap"].as_array().unwrap().iter().filter(|e| e["kind"] == "ckpt" && e["error"].is_null()).map(|e| e["n"].as_u64().unwrap()).collect();
            let n_prompt = x["n_prompt"].as_u64().unwrap();
            println!("  {} turn {}: mode {} reused {} fed {} of {} | ckpt {:?} | reply == from-zero {}", tag, k, x["mode"], x["n_reused"], x["n_feed"], n_prompt, ck, r["content"] == c["content"]);
            // turn 0's last user message starts after the short system turn (< CKPT_MIN): no checkpoint there, so
            // turn 1 re-reads from zero; from turn 2 on every request rewinds
            if k >= 2 {
                assert_eq!(x["mode"], "rewind", "turn {}: {}", k, x);
                assert!(x["n_reused"].as_u64().unwrap() as usize >= prev_user, "turn {}: reused {} < previous user start {}", k, x["n_reused"], prev_user);
            }
            // this request's last-user-message checkpoint (below the end-of-prompt one)
            let u = ck.iter().filter(|&&n| n + 1 < n_prompt).max().copied();
            if k >= 1 { assert!(u.is_some(), "turn {}: no user-start checkpoint: {}", k, x); }
            prev_user = u.unwrap_or(0) as usize;
            same += (r["content"] == c["content"]) as usize;
            if exact { assert_eq!(r["content"], c["content"], "turn {}: exact prefill: rewind reply != from-zero reply", k); }
            hist.push(json!({"role": "assistant", "content": format!("Piece {} reviewed.", k)}));
        }
        println!("  {}: {}/{} replies == from-zero", tag, same, pieces.len());
        let _ = std::fs::remove_dir_all(&dir);
        for f in [&log, &log_c] { let _ = std::fs::remove_file(f); }
    }
}

/// A client with tools: each turn is an instruction (a user text message ending with a "[state]" block) followed by
/// tool rounds (assistant tool_use + user tool_result); a new instruction strips the block from the previous
/// instruction. (a) Every new instruction rewinds to the start of the previous instruction (the pinned checkpoint,
/// taken while the tool-round requests ran as followup: the engine holds the model's own reply tokens there, so the
/// match is by messages, `CkDesc`). (b) The client swaps the tool list (another system turn: switch save + park of
/// the chat), comes back with the first list (parked restore of the chat's own state: its checkpoints become valid
/// again), then rewrites the last instruction: rewind again.
#[test]
fn rewind_client_tools_and_switch_tiny() {
    let dir = tmp("tools_cache");
    let log = tmp("tools_log.jsonl").display().to_string();
    let Some(port) = start(true, Some(dir.clone()), false, log.clone()) else { return };
    let bash = json!({"name": "bash", "description": "Run a shell command in the project folder.",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}});
    let read = json!({"name": "read_file", "description": "Read a file of the project.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}});
    let tools1 = json!([bash]);
    let tools2 = json!([bash, read]);
    let body = |msgs: &[Value], tools: &Value| json!({"model": "tiny", "max_tokens": 8, "stream": false,
        "system": "You are a coding agent. Use the tools.", "tools": tools, "messages": msgs});
    let strip = |h: &mut Vec<Value>| for m in h.iter_mut() {
        if let Some(c) = m["content"].as_str().map(|c| c.to_string()) { if let Some(i) = c.find("\n[state]\n") { m["content"] = json!(c[..i].to_string()); } }
    };
    let instr = |k: usize| json!({"role": "user", "content": format!("Step {}: add a function parse_{}() to src/lib.rs that reads the config file, validates every field and returns a typed struct; then run the tests.\n[state]\nstep: {}\ngoal: config loader\ndone: steps 0..{}\n", k, k, k, k as i64 - 1)});
    let round = |k: usize, j: usize| -> [Value; 2] {
        let id = format!("toolu_{}_{}", k, j);
        [json!({"role": "assistant", "content": [{"type": "tool_use", "id": id, "name": "bash", "input": {"command": format!("cargo test parse_{} -- --nocapture", k)}}]}),
         json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": format!("running 3 tests\ntest parse_{k}_ok ... ok\ntest parse_{k}_missing_field ... ok\ntest parse_{k}_bad_type ... FAILED (round {j})\nthread panicked at src/lib.rs:{}:5\n[exit 101]", 40 + j)}]})]
    };
    let mut hist: Vec<Value> = vec![];
    let mut n_log = 0;
    let mut last = || -> Value { let l = log_lines(&log); n_log = l.len(); l[n_log - 1].clone() };
    let user_ckpt = |x: &Value| -> usize {
        let np = x["n_prompt"].as_u64().unwrap();
        x["snap"].as_array().unwrap().iter().filter(|e| e["kind"] == "ckpt" && e["error"].is_null()).map(|e| e["n"].as_u64().unwrap()).filter(|&n| n + 1 < np).max().unwrap_or(0) as usize
    };
    let show = |what: &str, x: &Value| println!("  {:28} mode {:9} reused {:5} (engine {:5}) fed {:4} of {:5} | snap {:?}", what, x["mode"].as_str().unwrap_or(""), x["n_reused"], x["n_engine_reused"], x["n_feed"], x["n_prompt"],
        x["snap"].as_array().unwrap().iter().map(|e| format!("{}:{}", e["kind"].as_str().unwrap(), e["n"])).collect::<Vec<_>>());
    let mut pinned = 0usize;
    for k in 0..3 {
        strip(&mut hist);
        hist.push(instr(k));
        post(port, &body(&hist, &tools1));
        let x = last();
        show(&format!("instruction {}", k), &x);
        assert!(x["error"].is_null(), "{}", x);
        if k >= 1 {
            assert_eq!(x["mode"], "rewind", "instruction {}: {}", k, x);
            assert!(x["n_reused"].as_u64().unwrap() as usize >= pinned, "instruction {}: reused {} < previous instruction start {}", k, x["n_reused"], pinned);
        }
        pinned = user_ckpt(&x);
        assert!(pinned > 0, "instruction {}: no instruction-start checkpoint: {}", k, x);
        for j in 0..2 {
            hist.extend(round(k, j));
            post(port, &body(&hist, &tools1));
            let x = last();
            show(&format!("  tool round {}.{}", k, j), &x);
            assert!(x["mode"] == "followup" || x["mode"] == "prefix", "tool round {}.{}: {}", k, j, x);
        }
    }
    // (c) the client trims the tail of the last tool result (the engine diverges from the render earlier: its ids hold
    // the model's own replies, the history re-renders the client's tool_use): rewind to the last turn start, matched by
    // messages, re-reading only that tool result
    let prev = { let l = log_lines(&log); l[l.len() - 1].clone() };
    if let Some(c) = hist.last_mut().and_then(|m| m["content"][0]["content"].as_str().map(|c| c.to_string())) {
        let cut = c.rfind('\n').unwrap();
        hist.last_mut().unwrap()["content"][0]["content"] = json!(c[..cut].to_string());
    }
    post(port, &body(&hist, &tools1));
    let x = last();
    show("last tool result trimmed", &x);
    assert_eq!(x["mode"], "rewind", "{}", x);
    assert!(x["n_reused"].as_u64().unwrap() >= prev["n_reused"].as_u64().unwrap(), "reused {} < previous request start {}", x["n_reused"], prev["n_reused"]);
    assert!(x["n_reused"].as_u64().unwrap() <= x["n_prompt"].as_u64().unwrap(), "{}", x);
    // (b) the same chat with another tool list (another system turn), then back
    let mut away = hist.clone();
    away.extend(round(9, 0));
    post(port, &body(&away, &tools2));
    let x = last();
    show("tools switched away", &x);
    assert!(x["snap"].as_array().unwrap().iter().any(|e| e["kind"] == "save_switch"), "switch save: {}", x);
    hist.extend(round(2, 2));
    post(port, &body(&hist, &tools1));
    let x = last();
    show("tools switched back", &x);
    assert_eq!(x["mode"], "restore", "the chat's own state restored: {}", x);
    strip(&mut hist);
    hist.push(instr(3));
    post(port, &body(&hist, &tools1));
    let x = last();
    show("instruction 3 (after switch)", &x);
    assert_eq!(x["mode"], "rewind", "after the switch back: {}", x);
    assert!(x["n_reused"].as_u64().unwrap() as usize >= pinned, "reused {} < instruction 2 start {}", x["n_reused"], pinned);
    let _ = n_log;
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&log);
}

/// N7 (`[gateway] ckpt_reply_start`): the client edits the previous assistant reply (here: appends text to it).
/// R1 (cold, client history) takes a reply-start checkpoint (after its last message, before the generation prompt);
/// R2 = R1 + reply A1 + a user message runs as followup; R3 = R1 + A1 with text appended + .. diverges inside A1
/// (planner: full) and rewinds to R1's reply start (n_reused = that checkpoint) instead of R1's last user message
/// start; with the exact prefill its reply == a from-zero gateway's. Then the message-equal path: R4 = R3 + A3 + user
/// (followup: the engine holds the model's own reply tokens for A3), R5 = R4 + A4 + user (followup), R6 edits A4 (the
/// reply R4 generated): it rewinds to R4's reply start, matched by messages ("msg-equal": the engine ids there hold the
/// model's own A3 tokens, not the render), and that checkpoint survived R5's checkpoints.
/// The same chain with the key off rewinds R3 only to R1's user-message start (less reused).
#[test]
fn rewind_edited_reply_tiny() {
    let pad: String = (0..60).map(|i| format!("fn g{}(v: &[u8]) -> usize {{ v.len() * {} }}\n", i, i % 5 + 1)).collect();
    let u = |k: usize| json!({"role": "user", "content": format!("Step {}: review this and say what to change next.\n{}", k, &pad[k * 40..k * 40 + 300])});
    let a = |k: usize, extra: &str| json!({"role": "assistant", "content": format!("Step {} reviewed: rename g{} and add a test for empty input.{}", k, k, extra)});
    let b = |msgs: &[Value]| json!({"model": "tiny", "max_tokens": 8, "stream": false, "system": format!("You are a coding assistant. Answer briefly.\n{}", &pad[..900]), "messages": msgs});
    let edit = "\n\n(Edited by the client: also check the error type.)";
    let r1: Vec<Value> = vec![u(0), a(0, ""), u(1)];
    let r2: Vec<Value> = [r1.clone(), vec![a(1, ""), u(2)]].concat();
    let r3: Vec<Value> = [r1.clone(), vec![a(1, edit), u(2), a(2, ""), u(3)]].concat();
    let r4: Vec<Value> = [r3.clone(), vec![a(3, ""), u(4)]].concat();
    let r5: Vec<Value> = [r4.clone(), vec![a(4, ""), u(5)]].concat();
    let r6: Vec<Value> = [r4.clone(), vec![a(4, edit), u(5), a(5, ""), u(6)]].concat();
    let ckpts = |x: &Value| -> Vec<u64> { x["snap"].as_array().unwrap().iter().filter(|e| e["kind"] == "ckpt" && e["error"].is_null()).map(|e| e["n"].as_u64().unwrap()).collect() };
    let mut reused_r3 = [0u64; 2];
    for key in [true, false] {
        let log = tmp(&format!("reply_log_{}", key));
        let log = log.display().to_string();
        let Some(port) = start_rs(true, None, true, log.clone(), key) else { return };
        let mut x = vec![];
        let mut rep3 = Value::Null;
        for (k, m) in [&r1, &r2, &r3, &r4, &r5, &r6].iter().enumerate() {
            let r = post(port, &b(m));
            if k == 2 { rep3 = r["content"].clone(); }
            let l = log_lines(&log);
            let y = l[k].clone();
            println!("  key {} R{}: mode {:9} reused {:5} fed {:4} of {:5} | ckpt {:?} | {} | ckpts {}", key, k + 1, y["mode"].as_str().unwrap_or(""), y["n_reused"], y["n_feed"], y["n_prompt"], ckpts(&y), y["reuse_why"], y["ckpts"]);
            assert!(y["error"].is_null(), "{}", y);
            x.push(y);
        }
        reused_r3[key as usize] = x[2]["n_reused"].as_u64().unwrap();
        assert_eq!((x[0]["mode"].as_str(), x[1]["mode"].as_str(), x[3]["mode"].as_str(), x[4]["mode"].as_str()), (Some("full"), Some("followup"), Some("followup"), Some("followup")));
        assert_eq!(x[2]["mode"], "rewind", "R3: {}", x[2]);
        if !key { let _ = std::fs::remove_file(&log); continue; }
        // R1's reply start: its last checkpoint, just before the generation prompt
        let rs1 = *ckpts(&x[0]).iter().max().expect("R1 checkpoints");
        let np1 = x[0]["n_prompt"].as_u64().unwrap();
        assert!(rs1 < np1 && rs1 + 12 > np1, "R1 reply start {} vs prompt {}", rs1, np1);
        assert_eq!(x[2]["n_reused"].as_u64(), Some(rs1), "R3 rewinds to R1's reply start: {}", x[2]);
        assert!(x[2]["reuse_why"].as_str().unwrap().starts_with(&format!("rewind ckpt@{}", rs1)), "{}", x[2]);
        // == a from-zero gateway (exact prefill)
        let log_z = tmp("reply_log_zero").display().to_string();
        let Some(pz) = start(false, None, true, log_z.clone()) else { return };
        let z = post(pz, &b(&r3));
        assert_eq!(log_lines(&log_z)[0]["mode"], "full");
        assert_eq!(rep3, z["content"], "R3 after the rewind != from zero");
        // R6: R4's reply start, matched by messages (the engine holds R3's own reply tokens before it)
        let rs4 = *ckpts(&x[3]).iter().max().expect("R4 checkpoints");
        assert_eq!(x[5]["mode"], "rewind", "R6: {}", x[5]);
        let why = x[5]["reuse_why"].as_str().unwrap();
        assert!(why.starts_with(&format!("rewind ckpt@{} (msg-equal", rs4)), "R6 rewinds to R4's reply start by messages: {}", x[5]);
        assert!(x[5]["ckpts"].as_array().unwrap().iter().any(|c| c["reply"] == true), "{}", x[5]);
        for f in [&log, &log_z] { let _ = std::fs::remove_file(f); }
    }
    println!("  R3 reused: key on {} / off {}", reused_r3[1], reused_r3[0]);
    assert!(reused_r3[1] > reused_r3[0], "the reply-start checkpoint reuses more: on {} off {}", reused_r3[1], reused_r3[0]);
}
