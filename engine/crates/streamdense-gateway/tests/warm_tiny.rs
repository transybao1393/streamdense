//! Warm-up requests (prefill only: `max_tokens: 0` or `x-streamdense-warm: 1`, docs/GATEWAY.md "Warm-up"):
//! 1. tiny model (models/tiny_qwen35, ids folded) + the 27B tokenizer + snapshot files, over HTTP: a warm-up of a
//!    conversation returns no content and saves the rolling state; the next real request (same messages + a new user
//!    turn) reuses the prompt from the live KV ("prefix", reused = warm prompt - 1); a fresh gateway on the same
//!    snapshot dir restores it from the file ("own"); both replies == a cold gateway's reply.
//! 2. priority, scripted fake engine (slow prefill, warm pieces of 64 tokens): a real request of another
//!    conversation cancels a warm-up in progress (and waits at most a piece); a real request queued after a warm-up
//!    runs first (the warm-up still completes after it); a real request of the same conversation cancels the
//!    warm-up too and reuses its fed pieces (prefix).
//! 3. tiny model, no snapshot files: a running warm-up of a long prompt is cancelled by a real request (another
//!    conversation), which starts within about one chunk (queue_s << the full warm-up prefill); cancelled by its own
//!    conversation, the fed whole chunks are reused (prefix) and the reply == a cold gateway's.
use serde_json::{json, Value};
use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};
use streamdense_gateway::client::{request, request_with};
use streamdense_gateway::{Backend, CacheHook, FeedStats, GatewayConfig, SnapshotHook, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_model::snapshot::SnapshotCache;
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("sd_gwwarm_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// The agent fixture with these messages (non-streaming).
fn body(msgs: Value, max_tokens: u64) -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut b = v["body"].clone();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(max_tokens);
    b["messages"] = msgs;
    b
}

fn start_tiny(dir: std::path::PathBuf, log: String) -> Option<u16> { start_tiny_opt(Some(dir), log) }

fn start_tiny_opt(dir: Option<std::path::PathBuf>, log: String) -> Option<u16> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer missing");
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = QwenEngine::open(&tiny, QwenOptions { max_ctx: 4096, mtp: true, ..Default::default() }).unwrap();
        let vocab = eng.cfg.vocab as u32;
        let hook: Option<Box<dyn SnapshotHook<QwenEngine>>> = dir.map(|dir| {
            let cache = SnapshotCache::open(dir, &CacheHook::model_id(&tiny)).unwrap();
            Box::new(CacheHook::new(cache, 8, 1 << 30)) as Box<dyn SnapshotHook<QwenEngine>>
        });
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab) }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, hook, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(120)).unwrap())
}

fn kinds(l: &Value) -> Vec<String> { l["snap"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect() }

#[test]
fn warm_then_real_tiny() {
    let hist = json!([{"role": "user", "content": "say hi"}, {"role": "assistant", "content": "Hi! What should we work on today?"}]);
    let mut next = hist.as_array().unwrap().clone();
    next.push(json!({"role": "user", "content": "List the files in this folder."}));
    let next = Value::Array(next);

    let dir = tmp("cache");
    let log = tmp("log.jsonl").display().to_string();
    let Some(port) = start_tiny(dir.clone(), log.clone()) else { return };

    // warm-up: no content, usage = the whole prompt prefilled, the rolling state saved (all but the last token)
    let r = request(port, "POST", "/v1/messages", Some(&body(hist.clone(), 0))).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let w = r.json();
    assert_eq!(w["content"], json!([]), "{}", w);
    assert_eq!(w["streamdense"]["warm"], "done", "{}", w);
    let n_w = w["streamdense"]["n_prompt"].as_u64().unwrap();
    assert_eq!(w["usage"]["input_tokens"].as_u64().unwrap(), n_w);
    assert_eq!(w["usage"]["output_tokens"], 0);
    let l = log_lines(&log);
    assert_eq!(l[0]["warm"], true, "{}", l[0]);
    assert_eq!(l[0]["n_new"], 0);
    assert_eq!(kinds(&l[0]), ["save_stable", "save_roll"], "{}", l[0]);
    assert_eq!(l[0]["snap"][1]["n"].as_u64().unwrap(), n_w - 1);

    // the real request (history + a new turn): the live KV covers the warm prompt -> prefix, reused = n_w - 1
    let r = request(port, "POST", "/v1/messages", Some(&body(next.clone(), 12))).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let real = r.json();
    let l = log_lines(&log);
    assert_eq!(l[1]["mode"], "prefix", "{}", l[1]);
    assert_eq!(l[1]["n_reused"].as_u64().unwrap(), n_w - 1);
    let n_r = l[1]["n_prompt"].as_u64().unwrap();
    assert_eq!(real["usage"]["cache_read_input_tokens"].as_u64().unwrap(), n_w - 1);
    assert!(n_r - (n_w - 1) < 40, "only the new turn is fed: {} of {}", n_r - (n_w - 1), n_r);

    // a fresh gateway on the same snapshot dir: the warm-up's rolling file restores ("own")
    let log2 = tmp("log2.jsonl").display().to_string();
    let Some(port2) = start_tiny(dir.clone(), log2.clone()) else { return };
    let r2 = request(port2, "POST", "/v1/messages", Some(&body(next.clone(), 12))).unwrap().json();
    let l2 = log_lines(&log2);
    assert_eq!(l2[0]["mode"], "restore", "{}", l2[0]);
    assert_eq!(l2[0]["prefix"], "own", "{}", l2[0]);
    assert_eq!(l2[0]["n_reused"].as_u64().unwrap(), n_w - 1);

    // a cold gateway (empty dir): the same greedy reply
    let dir3 = tmp("cold");
    let log3 = tmp("log3.jsonl").display().to_string();
    let Some(port3) = start_tiny(dir3.clone(), log3.clone()) else { return };
    let r3 = request(port3, "POST", "/v1/messages", Some(&body(next.clone(), 12))).unwrap().json();
    assert_eq!(log_lines(&log3)[0]["mode"], "full");
    assert_eq!(real["content"], r3["content"], "warm + prefix == cold");
    assert_eq!(r2["content"], r3["content"], "warm file restore == cold");
    println!("  tiny warm: warm prompt {} tokens, real prompt {} tokens, reused {}", n_w, n_r, n_w - 1);
    for d in [&dir, &dir3] { let _ = std::fs::remove_dir_all(d); }
    for f in [&log, &log2, &log3] { let _ = std::fs::remove_file(f); }
}

// ---------------------------------------------------------------- priority (fake engine)

/// Prefill costs PER_TOKEN per fed token, decode 2 ms per token; position rules as the engine (last token pending).
struct Slow { pos: Cell<usize> }
const PER_TOKEN: Duration = Duration::from_micros(400);

impl Backend for Slow {
    fn reset(&self) { self.pos.set(0); }
    fn pos(&self) -> usize { self.pos.get() }
    fn max_ctx(&self) -> usize { 32768 }
    fn vocab(&self) -> usize { 248320 }
    fn mtp_available(&self) -> bool { false }
    fn warm_chunk(&self) -> usize { 64 }
    fn feed(&self, ids: &[u32], n_new: usize, stop: &[u32], _mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> {
        std::thread::sleep(PER_TOKEN * ids.len() as u32);
        self.pos.set(self.pos.get() + ids.len());
        let mut st = FeedStats { stop_reason: "length".into(), prefill_s: (PER_TOKEN * ids.len() as u32).as_secs_f64(), ..Default::default() };
        loop {
            let t = 13; // "."
            st.gen += 1;
            st.last = Some(t);
            if n_new > 1 { std::thread::sleep(Duration::from_millis(2)); }
            if !on_tok(t) { st.stop_reason = "cancelled".into(); break; }
            if stop.contains(&t) { st.stop_reason = "eos".into(); break; }
            if st.gen >= n_new { break; }
            self.pos.set(self.pos.get() + 1);
        }
        Ok(st)
    }
}

fn start_slow(log: &str) -> Option<u16> {
    if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer.json missing"); return None; }
    let (tx, rx) = std::sync::mpsc::channel();
    let log = log.to_string();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = Slow { pos: Cell::new(0) };
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: false, fold: None }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, None, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(60)).unwrap())
}

/// A conversation `tag` with a long first message (~3000 tokens: ~1.2 s of fake prefill).
fn conv(tag: &str) -> Value {
    let long: String = (0..600).map(|i| format!("{} line {} of the notes. ", tag, i)).collect();
    json!([{"role": "user", "content": long}, {"role": "assistant", "content": format!("Noted ({}).", tag)}])
}
fn with_turn(h: &Value, text: &str) -> Value { let mut v = h.as_array().unwrap().clone(); v.push(json!({"role": "user", "content": text})); Value::Array(v) }

fn post_bg(port: u16, b: Value, headers: Vec<(&'static str, &'static str)>) -> std::thread::JoinHandle<(Value, f64)> {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = request_with(port, "POST", "/v1/messages", Some(&b), &headers).unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        (r.json(), t0.elapsed().as_secs_f64())
    })
}

#[test]
fn warm_priority_fake() {
    let log = tmp("prio.jsonl").display().to_string();
    let Some(port) = start_slow(&log) else { return };
    assert_eq!(request(port, "GET", "/health", None).unwrap().json()["info"]["warm"], true);

    // 1. a warm-up in progress is cancelled by a real request of another conversation
    let w = post_bg(port, body(conv("alpha"), 0), vec![]);
    std::thread::sleep(Duration::from_millis(300));
    let (r, r_s) = post_bg(port, body(json!([{"role": "user", "content": "hello"}]), 4), vec![]).join().unwrap();
    let (w, w_s) = w.join().unwrap();
    assert_eq!(w["streamdense"]["warm"], "cancelled", "{}", w);
    assert_eq!(r["content"][0]["type"], "text", "{}", r);
    let l = log_lines(&log);
    assert_eq!(l.len(), 2);
    assert_eq!((l[0]["warm"].as_bool(), l[0]["stop_reason"].as_str()), (Some(true), Some("cancelled")), "{}", l[0]);
    assert!(w_s < 1.0 && r_s < 1.5, "the warm-up stopped at a piece: warm {:.2} s, real {:.2} s", w_s, r_s);
    assert!(l[1]["queue_s"].as_f64().unwrap() < 0.2, "{}", l[1]);

    // 2. a real request queued behind a warm-up runs first; the warm-up (another conversation, queued, not running)
    //    still runs after it
    let busy = post_bg(port, body(json!([{"role": "user", "content": "busy one"}]), 150), vec![]); // ~0.3 s decode
    std::thread::sleep(Duration::from_millis(60));
    let w = post_bg(port, body(conv("beta"), 0), vec![]);
    std::thread::sleep(Duration::from_millis(60));
    let r = post_bg(port, body(json!([{"role": "user", "content": "quick question"}]), 4), vec![]);
    let _ = busy.join().unwrap();
    let (r, _) = r.join().unwrap();
    let (w, _) = w.join().unwrap();
    assert_eq!(w["streamdense"]["warm"], "done", "{}", w);
    assert!(r["content"].as_array().is_some_and(|c| !c.is_empty()));
    let l = log_lines(&log);
    let order: Vec<(bool, u64)> = l[2..].iter().map(|x| (x["warm"].as_bool().unwrap(), x["n_msgs"].as_u64().unwrap())).collect();
    assert_eq!(order.iter().map(|o| o.0).collect::<Vec<_>>(), [false, false, true], "busy, real, then the warm-up: {:?}", order);

    // 3. a real request of the same conversation cancels the warm-up (header form) too and reuses its fed pieces
    let h = conv("gamma");
    let w = post_bg(port, body(h.clone(), 64), vec![("x-streamdense-warm", "1")]);
    std::thread::sleep(Duration::from_millis(200));
    let (_r, _) = post_bg(port, body(with_turn(&h, "and now?"), 4), vec![]).join().unwrap();
    let (w, _) = w.join().unwrap();
    assert_eq!(w["streamdense"]["warm"], "cancelled", "{}", w);
    let n_w = w["streamdense"]["n_prompt"].as_u64().unwrap();
    let l = log_lines(&log);
    let last = l.last().unwrap();
    assert_eq!(last["warm"], false);
    assert!(last["queue_s"].as_f64().unwrap() < 0.2, "{}", last);
    assert_eq!(last["mode"], "prefix", "{}", last);
    let reused = last["n_reused"].as_u64().unwrap();
    assert!(reused > 0 && reused < n_w - 1 && reused % 64 == 0, "whole pieces reused: {} of {}", reused, n_w);
    println!("  fake warm: cancel after {:.2} s, real {:.2} s; same-conversation partial reuse {} of {} tokens", w_s, r_s, reused, n_w);
    let _ = std::fs::remove_file(&log);
}

/// Queued warm-ups are dropped before they start: the client closed the connection ("client_gone"), or a newer
/// warm-up of the same project + lane arrived ("superseded"); the newest one runs. (Age cap: worker.rs unit tests.)
#[test]
fn warm_queue_drop_fake() {
    use std::io::Write;
    let log = tmp("drop.jsonl").display().to_string();
    let Some(port) = start_slow(&log) else { return };
    // a real request decodes ~1 s; everything below queues behind it
    let busy = post_bg(port, body(json!([{"role": "user", "content": "busy one"}]), 500), vec![]);
    std::thread::sleep(Duration::from_millis(60));
    // 1. a warm-up whose client exits while it is queued (raw socket, closed without reading the reply)
    {
        let b = body(conv("delta"), 0).to_string();
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(s, "POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", b.len(), b).unwrap();
        std::thread::sleep(Duration::from_millis(100));
    }
    // 2. two warm-ups of one project (agent lane), different conversations: the newer replaces the older
    const PROJ: (&str, &str) = ("x-streamdense-project", "00000000000000ab");
    let w1 = post_bg(port, body(conv("eps1"), 0), vec![PROJ]);
    std::thread::sleep(Duration::from_millis(60));
    let w2 = post_bg(port, body(conv("eps2"), 0), vec![PROJ]);
    let (_, busy_s) = busy.join().unwrap();
    let (w1, _) = w1.join().unwrap();
    let (w2, _) = w2.join().unwrap();
    assert_eq!((w1["streamdense"]["warm"].as_str(), w1["streamdense"]["drop"].as_str()), (Some("cancelled"), Some("superseded")), "{}", w1);
    assert_eq!(w2["streamdense"]["warm"], "done", "{}", w2);
    std::thread::sleep(Duration::from_millis(200));
    let l = log_lines(&log);
    let warm: Vec<&Value> = l.iter().filter(|x| x["warm"] == true).collect();
    assert_eq!(warm.len(), 3, "{:?}", l);
    let gone = warm.iter().find(|x| x["drop"] == "client_gone").unwrap_or_else(|| panic!("no client_gone drop: {:?}", warm));
    assert_eq!((gone["mode"].as_str(), gone["n_feed"].as_u64()), (Some("dropped"), Some(0)), "{}", gone);
    // the dropped warm-ups never ran: the last line is the newest warm-up, nothing fed but its prompt
    let last = l.last().unwrap();
    assert_eq!((last["warm"].as_bool(), last["stop_reason"].as_str()), (Some(true), Some("done")), "{}", last);
    println!("  fake drop: busy {:.2} s; client_gone + superseded dropped, newest done", busy_s);
    let _ = std::fs::remove_file(&log);
}

/// A long conversation `tag` (`lines` lines of notes) + a short assistant reply.
fn long_conv(tag: &str, lines: usize) -> Value {
    let long: String = (0..lines).map(|i| format!("{} line {} of the notes. ", tag, i)).collect();
    json!([{"role": "user", "content": long}, {"role": "assistant", "content": format!("Noted ({}).", tag)}])
}

#[test]
fn warm_cancel_tiny() {
    let log = tmp("cancel.jsonl").display().to_string();
    let Some(port) = start_tiny_opt(None, log.clone()) else { return };
    const LINES: usize = 270;

    // the full prefill time of a long warm-up (run to completion)
    let (w, _) = post_bg(port, body(long_conv("alpha", LINES), 0), vec![]).join().unwrap();
    assert_eq!(w["streamdense"]["warm"], "done", "{}", w);
    let n_full = w["streamdense"]["n_prompt"].as_u64().unwrap();
    let p_full = w["streamdense"]["prefill_s"].as_f64().unwrap();
    let chunk_s = p_full * 128.0 / n_full as f64;
    assert!(p_full > 0.2, "the long warm-up must take measurable time: {:.3} s", p_full);

    // 1. a warm-up of another long prompt is running; a real request of a third conversation arrives
    let w = post_bg(port, body(long_conv("beta", LINES), 0), vec![]);
    std::thread::sleep(Duration::from_secs_f64(p_full * 0.35));
    let (r, _) = post_bg(port, body(json!([{"role": "user", "content": "hello"}]), 4), vec![]).join().unwrap();
    let (w, _) = w.join().unwrap();
    assert_eq!(w["streamdense"]["warm"], "cancelled", "{}", w);
    assert!(r["content"].as_array().is_some_and(|c| !c.is_empty()), "{}", r);
    let l = log_lines(&log);
    let (lw, lr) = (&l[1], &l[2]);
    assert_eq!((lw["warm"].as_bool(), lw["stop_reason"].as_str()), (Some(true), Some("cancelled")), "{}", lw);
    assert_eq!(lr["warm"], false, "{}", lr);
    let q1 = lr["queue_s"].as_f64().unwrap();
    assert!(lw["prefill_s"].as_f64().unwrap() < p_full * 0.8, "the warm-up stopped early: {}", lw);
    assert!(q1 < p_full * 0.5 && q1 < chunk_s * 2.0 + 0.1, "real waited {:.3} s (full warm prefill {:.3} s, chunk {:.3} s)", q1, p_full, chunk_s);

    // 2. a warm-up cancelled by its own conversation: the fed chunks are reused (prefix), reply == cold
    let h = long_conv("gamma", LINES);
    let next = with_turn(&h, "Summarize the notes in one line.");
    let w = post_bg(port, body(h.clone(), 0), vec![]);
    std::thread::sleep(Duration::from_secs_f64(p_full * 0.35));
    let (real, _) = post_bg(port, body(next.clone(), 12), vec![]).join().unwrap();
    let (w, _) = w.join().unwrap();
    assert_eq!(w["streamdense"]["warm"], "cancelled", "{}", w);
    let l = log_lines(&log);
    let lr = l.last().unwrap();
    let q2 = lr["queue_s"].as_f64().unwrap();
    let n_w = w["streamdense"]["n_prompt"].as_u64().unwrap();
    let reused = lr["n_reused"].as_u64().unwrap();
    assert_eq!(lr["mode"], "prefix", "{}", lr);
    assert!(reused > 0 && reused < n_w - 1 && reused % 128 == 0, "whole chunks reused: {} of {}", reused, n_w);
    assert!(q2 < p_full * 0.5, "real waited {:.3} s", q2);

    let log2 = tmp("cancel_cold.jsonl").display().to_string();
    let Some(port2) = start_tiny_opt(None, log2.clone()) else { return };
    let (cold, _) = post_bg(port2, body(next, 12), vec![]).join().unwrap();
    assert_eq!(log_lines(&log2)[0]["mode"], "full");
    assert_eq!(real["content"], cold["content"], "partial warm + prefix == cold");
    println!("  tiny cancel: full warm prefill {:.3} s ({} tokens, chunk ~{:.3} s); real queue_s {:.3} s (other conv), {:.3} s (same conv, reused {} of {})",
             p_full, n_full, chunk_s, q1, q2, reused, n_w);
    for f in [&log, &log2] { let _ = std::fs::remove_file(f); }
}
