//! The whole gateway over real HTTP with a scripted fake engine (no model, no GPU) and the 27B tokenizer:
//! SSE order + text streaming, a tool round trip (tool_use with coerced args -> tool_result -> answer), KV mode
//! prefix on turn 2 (checked in the request log), non-stream JSON, count_tokens, models, health, errors, queueing.
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::client::{request, Response};
use streamdense_gateway::{Backend, FeedStats, GatewayConfig, WorkerConfig};
use streamdense_tokenizer::{Tokenizer, IM_END};

const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

/// Replays scripted replies token by token with the engine's position rules (prompt fed, last token pending).
struct Fake { pos: Cell<usize>, replies: RefCell<VecDeque<Vec<u32>>>, max_ctx: usize, feeds: Arc<std::sync::Mutex<Vec<usize>>> }

impl Backend for Fake {
    fn reset(&self) { self.pos.set(0); }
    fn pos(&self) -> usize { self.pos.get() }
    fn max_ctx(&self) -> usize { self.max_ctx }
    fn vocab(&self) -> usize { 248320 }
    fn mtp_available(&self) -> bool { false }
    fn feed(&self, ids: &[u32], n_new: usize, stop: &[u32], _mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> {
        self.feeds.lock().unwrap().push(ids.len());
        let mut st = FeedStats { stop_reason: "length".into(), ..Default::default() };
        if self.pos.get() + ids.len() > self.max_ctx { st.stop_reason = "ctx_full".into(); return Ok(st); }
        self.pos.set(self.pos.get() + ids.len());
        let reply = self.replies.borrow_mut().pop_front().unwrap_or_default();
        let mut i = 0;
        loop {
            let t = reply.get(i).copied().unwrap_or(13); // "." filler once the script runs out
            i += 1;
            st.gen += 1;
            st.last = Some(t);
            std::thread::sleep(Duration::from_millis(2));
            if !on_tok(t) { st.stop_reason = "cancelled".into(); break; }
            if stop.contains(&t) { st.stop_reason = "eos".into(); break; }
            if st.gen >= n_new { st.stop_reason = "length".into(); break; }
            if self.pos.get() + 1 > self.max_ctx { st.stop_reason = "ctx_full".into(); break; }
            self.pos.set(self.pos.get() + 1);
        }
        st.decode_s = 0.01;
        Ok(st)
    }
}

fn start(replies: Vec<Vec<u32>>, log: &str) -> Option<(u16, Arc<std::sync::Mutex<Vec<usize>>>)> {
    start_wc(replies, log, WorkerConfig { think: false, mtp: true, fold: None, ..Default::default() })
}

fn start_wc(replies: Vec<Vec<u32>>, log: &str, wc: WorkerConfig) -> Option<(u16, Arc<std::sync::Mutex<Vec<usize>>>)> {
    if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer.json missing"); return None; }
    let feeds = Arc::new(std::sync::Mutex::new(vec![]));
    let f2 = feeds.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let log = log.to_string();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = Fake { pos: Cell::new(0), replies: RefCell::new(replies.into()), max_ctx: 4096, feeds: f2 };
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: wc, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, None, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some((rx.recv_timeout(Duration::from_secs(60)).unwrap(), feeds))
}

fn tk() -> Tokenizer { Tokenizer::from_path(TOK_DIR).unwrap() }
fn reply(tk: &Tokenizer, s: &str) -> Vec<u32> { let mut v = tk.encode(s); v.push(IM_END); v }
fn body() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    v["body"].clone()
}
fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}
fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("sd_gw_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_file(&d);
    d.display().to_string()
}

/// Check the Anthropic event order; returns (streamed text, tool_use blocks (name, input), stop_reason, usage).
fn check_stream(r: &Response) -> (String, Vec<(String, Value)>, String, Value) {
    assert_eq!(r.status, 200, "{}", r.text());
    assert!(r.header("content-type").unwrap().starts_with("text/event-stream"));
    let ev: Vec<(String, Value)> = r.events().into_iter().filter(|(n, _)| n != "ping").collect();
    assert_eq!(ev.first().unwrap().0, "message_start");
    assert_eq!(ev.last().unwrap().0, "message_stop");
    assert_eq!(ev[ev.len() - 2].0, "message_delta");
    let (mut text, mut tools, mut open): (String, Vec<(String, Value)>, Option<(usize, String)>) = (String::new(), vec![], None);
    let mut next_index = 0;
    let mut partial = String::new();
    for (n, d) in &ev[1..ev.len() - 2] {
        assert_eq!(d["type"].as_str().unwrap(), n);
        match n.as_str() {
            "content_block_start" => {
                assert!(open.is_none());
                assert_eq!(d["index"].as_u64().unwrap() as usize, next_index);
                let ty = d["content_block"]["type"].as_str().unwrap().to_string();
                if ty == "tool_use" { tools.push((d["content_block"]["name"].as_str().unwrap().to_string(), Value::Null)); partial.clear(); }
                else { assert_eq!(ty, "text"); assert!(tools.is_empty(), "text block after a tool_use block"); }
                open = Some((next_index, ty));
            }
            "content_block_delta" => {
                let (i, ty) = open.as_ref().expect("delta outside a block");
                assert_eq!(d["index"].as_u64().unwrap() as usize, *i);
                if ty == "text" { text.push_str(d["delta"]["text"].as_str().unwrap()); }
                else { partial.push_str(d["delta"]["partial_json"].as_str().unwrap()); }
            }
            "content_block_stop" => {
                let (i, ty) = open.take().expect("stop outside a block");
                assert_eq!(d["index"].as_u64().unwrap() as usize, i);
                if ty == "tool_use" { tools.last_mut().unwrap().1 = serde_json::from_str(&partial).unwrap(); }
                next_index += 1;
            }
            e => panic!("unexpected event {}", e),
        }
    }
    assert!(open.is_none());
    let md = &ev[ev.len() - 2].1;
    (text, tools, md["delta"]["stop_reason"].as_str().unwrap().to_string(), md["usage"].clone())
}

#[test]
fn gateway_fake_engine_end_to_end() {
    let t = if std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { tk() } else { eprintln!("SKIP: needs models/qwen38_27b_mlx4/tokenizer.json (missing)"); return };
    let r1 = "Hello there.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n5000\n</parameter>\n</function>\n</tool_call>";
    let replies = vec![reply(&t, r1), reply(&t, "Two files: a.txt and b.txt."), reply(&t, "plain json"), reply(&t, "x")];
    let log = tmp("fake_log");
    let Some((port, feeds)) = start(replies, &log) else { return };

    // health, models
    let h = request(port, "GET", "/health", None).unwrap();
    assert_eq!((h.status, h.json()["status"].clone()), (200, json!("ok")));
    assert_eq!(request(port, "HEAD", "/", None).unwrap().status, 200);
    assert!(h.json()["ram"]["footprint_gb"].as_f64().unwrap() > 0.0, "{}", h.text());
    let hr = request(port, "GET", "/health/ram", None).unwrap().json();
    assert!(hr["ram"]["footprint_gb"].as_f64().unwrap() > 0.0 && hr["ram"]["swap_used_gb"].is_number() && hr["pid"].is_u64(), "{}", hr);
    let m = request(port, "GET", "/v1/models", None).unwrap().json();
    assert_eq!(m["data"][0]["id"], "qwen38-27b");
    assert_eq!(request(port, "GET", "/v1/nope", None).unwrap().status, 404);

    // count_tokens == input tokens of the real request
    let b1 = body();
    let ct = request(port, "POST", "/v1/messages/count_tokens?beta=true", Some(&b1)).unwrap().json();
    let n_in = ct["input_tokens"].as_u64().unwrap();
    assert!(n_in > 500, "{}", ct);

    // turn 1: stream text, then a tool_use with coerced args (timeout: number)
    let r = request(port, "POST", "/v1/messages?beta=true", Some(&b1)).unwrap();
    let (text, tools, stop, usage) = check_stream(&r);
    assert_eq!(text, "Hello there.");
    assert_eq!(tools, vec![("Bash".to_string(), json!({"command": "ls -la", "timeout": 5000}))]);
    assert_eq!(stop, "tool_use");
    assert_eq!(usage["input_tokens"].as_u64().unwrap(), n_in);
    let deltas = r.events().iter().filter(|(n, d)| n == "content_block_delta" && d["delta"]["type"] == "text_delta").count();
    assert!(deltas >= 2, "text streamed per token, got {} deltas", deltas);
    // the tool call streams while it is written (toolctl.rs ToolStream): several input pieces whose concatenation is
    // exactly the input JSON of the non-stream reply
    let pieces: Vec<String> = r.events().iter().filter(|(n, d)| n == "content_block_delta" && d["delta"]["type"] == "input_json_delta")
        .map(|(_, d)| d["delta"]["partial_json"].as_str().unwrap().to_string()).collect();
    assert!(pieces.len() >= 3, "tool input in pieces, got {:?}", pieces);
    assert_eq!(pieces.concat(), json!({"command": "ls -la", "timeout": 5000}).to_string());

    // turn 2: Claude Code appends our reply + the tool_result -> prefix reuse
    let mut b2 = b1.clone();
    let msgs = b2["messages"].as_array_mut().unwrap();
    msgs.push(json!({"role": "assistant", "content": [{"type": "text", "text": "Hello there."},
        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls -la", "timeout": 5000}}]}));
    msgs.push(json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt\nb.txt"}]}));
    let r = request(port, "POST", "/v1/messages?beta=true", Some(&b2)).unwrap();
    let (text, tools, stop, usage) = check_stream(&r);
    assert_eq!((text.as_str(), tools.len(), stop.as_str()), ("Two files: a.txt and b.txt.", 0, "end_turn"));
    assert!(usage["cache_read_input_tokens"].as_u64().unwrap() > n_in);
    let l = log_lines(&log);
    assert_eq!(l.len(), 2, "{:?}", l);
    // diagnostics: correlation id, reuse reason + divergence, RAM, stop, timing (2026-11-17)
    for x in &l {
        for k in ["rid", "reuse_why", "div", "ram", "stop", "timing", "prefill_tok_s", "ckpts", "tool_ctl"] { assert!(x.get(k).is_some(), "{} missing: {}", k, x); }
        assert!(x["ram"]["footprint_gb"].as_f64().unwrap() > 0.0 && x["ram"]["swap_used_gb"].is_number() && x["ram"]["peak_footprint_gb"].is_number(), "{}", x);
        assert!(x["timing"]["total_s"].is_number() && x["timing"]["queue_s"].is_number() && x["timing"]["decode_s"].is_number(), "{}", x);
        assert_eq!(x["stop"]["n_out"], x["n_new"]);
        assert!(x["rid"].as_str().unwrap().starts_with("g-"), "{}", x);
    }
    assert!(l[0]["reuse_why"].as_str().unwrap().starts_with("full: no live state"), "{}", l[0]);
    assert_eq!(l[1]["reuse_why"], "append", "{}", l[1]);
    assert!(l[1]["div"]["live"].as_u64().unwrap() == l[1]["div"]["live_len"].as_u64().unwrap(), "{}", l[1]);
    assert_eq!(l[1]["div"]["msg"], json!(2), "turn 2 diverges at the new assistant message: {}", l[1]);
    // the stream's last ping carries the reuse summary; the SSE head the id
    assert!(r.header("x-request-id").unwrap().starts_with("g-"));
    let meta = r.events().into_iter().find(|(n, d)| n == "ping" && d.get("streamdense").is_some()).expect("summary ping").1;
    assert_eq!((meta["streamdense"]["mode"].clone(), meta["streamdense"]["rid"].clone()), (json!("prefix"), l[1]["rid"].clone()));
    assert_eq!(l[0]["mode"], "full");
    assert_eq!(l[0]["tools"], json!(["Bash"]));
    assert_eq!(l[1]["mode"], "prefix", "{}", l[1]);
    assert!(l[1]["n_feed"].as_u64().unwrap() < 40, "{}", l[1]);
    assert_eq!(feeds.lock().unwrap()[1], l[1]["n_feed"].as_u64().unwrap() as usize);

    // non-stream JSON; a changed system prompt -> full prefill
    let mut b3 = body();
    b3["stream"] = json!(false);
    b3["system"] = json!("other system");
    let r = streamdense_gateway::client::request_with(port, "POST", "/v1/messages", Some(&b3), &[("X-Request-Id", "r-123-4")]).unwrap();
    assert_eq!(r.status, 200);
    // the client's id is echoed and logged; reuse headers on a JSON reply
    assert_eq!(r.header("x-request-id"), Some("r-123-4"));
    assert_eq!(r.header("x-sd-mode"), Some("full"));
    assert!(r.header("x-sd-prompt").unwrap().parse::<usize>().unwrap() > 500 && r.header("x-sd-reused").is_some() && r.header("x-sd-ttft").is_some());
    let l3 = &log_lines(&log)[2];
    assert_eq!(l3["rid"], "r-123-4");
    assert!(l3["reuse_why"].as_str().unwrap().starts_with("full: no ckpt <= div"), "a changed system prompt: {}", l3);
    assert_eq!(l3["div"]["msg"], json!(0), "the system message: {}", l3);
    let j = r.json();
    assert_eq!((j["type"].clone(), j["role"].clone(), j["stop_reason"].clone()), (json!("message"), json!("assistant"), json!("end_turn")));
    assert_eq!(j["content"], json!([{"type": "text", "text": "plain json"}]));
    assert_eq!(log_lines(&log)[2]["mode"], "full");

    // two requests at once: both complete (one generation at a time, the other waits)
    let mut b4 = body();
    b4["max_tokens"] = json!(3);
    let hs: Vec<_> = (0..2).map(|_| { let b = b4.clone(); std::thread::spawn(move || request(port, "POST", "/v1/messages", Some(&b)).unwrap()) }).collect();
    for h in hs { let (_, _, stop, _) = check_stream(&h.join().unwrap()); assert!(stop == "end_turn" || stop == "max_tokens"); }

    // errors: bad JSON, empty messages -> Anthropic error JSON, 400
    let bad = request(port, "POST", "/v1/messages", Some(&json!({"messages": []}))).unwrap();
    assert_eq!(bad.status, 400);
    assert_eq!(bad.json()["type"], "error");
    assert_eq!(bad.json()["error"]["type"], "invalid_request_error");

    // context full: friendly text, end_turn, no generation
    let mut b5 = body();
    b5["messages"] = json!([{"role": "user", "content": "word ".repeat(5000)}]);
    let r = request(port, "POST", "/v1/messages", Some(&b5)).unwrap();
    let (text, _, stop, _) = check_stream(&r);
    assert!(text.starts_with("[streamdense] Context full") && text.contains("/clear"), "{}", text);
    assert_eq!(stop, "end_turn");
    let _ = std::fs::remove_file(&log);
}

/// /health/ram and /health answer at once while a long generation runs (they never wait on the worker / GPU), and the
/// streamed reply is unaffected.
#[test]
fn health_answers_during_a_long_request() {
    let log = tmp("health_busy_log");
    let Some((port, _)) = start(vec![], &log) else { return };
    let mut b = body();
    b["max_tokens"] = json!(1500); // the fake writes filler at 2 ms / token: ~3 s busy
    let long = std::thread::spawn(move || { let r = request(port, "POST", "/v1/messages", Some(&b)).unwrap(); check_stream(&r) });
    std::thread::sleep(Duration::from_millis(400));
    let (mut worst_ram, mut worst_health) = (0f64, 0f64);
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let r = request(port, "GET", "/health/ram", None).unwrap();
        worst_ram = worst_ram.max(t.elapsed().as_secs_f64() * 1e3);
        let v = r.json();
        assert!(v["ram"]["footprint_gb"].as_f64().unwrap() > 0.0 && v["ram"]["age_ms"].as_u64().unwrap() < 2000, "{}", v);
        let t = std::time::Instant::now();
        assert_eq!(request(port, "GET", "/health", None).unwrap().json()["status"], "ok");
        worst_health = worst_health.max(t.elapsed().as_secs_f64() * 1e3);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!long.is_finished(), "the long request must still run while health is polled");
    eprintln!("worst /health/ram {:.1} ms, /health {:.1} ms", worst_ram, worst_health);
    assert!(worst_ram < 50.0 && worst_health < 50.0, "/health/ram {:.1} ms, /health {:.1} ms", worst_ram, worst_health);
    let (_, _, stop, _) = long.join().unwrap();
    assert_eq!(stop, "max_tokens");
    let _ = std::fs::remove_file(&log);
}

/// N6 (`[gateway] cut_tool_call`): a reply cut by max_tokens inside a tool call. Former behaviour (false): the call is
/// closed and returned as a complete tool_use with its truncated argument. "length" (true): stop_reason "max_tokens";
/// JSON: the partial call is dropped (the text before it stays); SSE: the tool_use block already streamed is closed as
/// it is (partial input) and the message ends with "max_tokens"; the request log has `tool_cut: {name, chars}`. A call
/// that completes is still "tool_use".
#[test]
fn cut_tool_call_reports_max_tokens() {
    let t = if std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { tk() } else { eprintln!("SKIP: 27B tokenizer.json missing"); return };
    let cut = "I'll count the lines.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nfor f in src/*.js; do wc -l \"$f\" &&";
    let ids = t.encode(cut);
    let n = ids.len() as u64;
    let done = "Done.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    let req = |stream: bool, max: u64, q: &str| { let mut b = body(); b["stream"] = json!(stream); b["max_tokens"] = json!(max);
        b["messages"] = json!([{"role": "user", "content": q}]); b };
    // former behaviour
    let log0 = tmp("cut_log0");
    let Some((p0, _)) = start(vec![ids.clone()], &log0) else { return };
    let r = request(p0, "POST", "/v1/messages", Some(&req(false, n, "count lines a"))).unwrap().json();
    assert_eq!(r["stop_reason"], "tool_use", "{}", r);
    assert_eq!(r["content"][1]["input"]["command"], "for f in src/*.js; do wc -l \"$f\" &&", "{}", r);
    assert!(log_lines(&log0)[0]["tool_cut"].is_null());
    // "length"
    let log = tmp("cut_log1");
    let wc = WorkerConfig { cut_tool_call: true, ..Default::default() };
    let Some((port, _)) = start_wc(vec![ids.clone(), ids.clone(), reply(&t, done)], &log, wc) else { return };
    let r = request(port, "POST", "/v1/messages", Some(&req(false, n, "count lines b"))).unwrap().json();
    assert_eq!(r["stop_reason"], "max_tokens", "{}", r);
    assert_eq!(r["content"], json!([{"type": "text", "text": "I'll count the lines."}]), "{}", r);
    // stream: the call streamed from its name on, closed partial, stop max_tokens
    let s = request(port, "POST", "/v1/messages", Some(&req(true, n, "count lines c"))).unwrap();
    assert_eq!(s.status, 200);
    let ev: Vec<(String, Value)> = s.events().into_iter().filter(|(n, _)| n != "ping").collect();
    let names: Vec<&str> = ev.iter().map(|e| e.0.as_str()).collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(&names[names.len() - 2..], &["message_delta", "message_stop"]);
    assert_eq!(ev[ev.len() - 2].1["delta"]["stop_reason"], "max_tokens", "{:?}", ev);
    let tool_start = ev.iter().position(|e| e.0 == "content_block_start" && e.1["content_block"]["type"] == "tool_use").expect("tool_use block streamed");
    assert_eq!(ev[tool_start].1["content_block"]["name"], "Bash");
    let idx = ev[tool_start].1["index"].clone();
    let partial: String = ev.iter().filter(|e| e.0 == "content_block_delta" && e.1["index"] == idx).map(|e| e.1["delta"]["partial_json"].as_str().unwrap().to_string()).collect();
    assert!(partial.starts_with("{\"command\":\"for f in src/*.js") && serde_json::from_str::<Value>(&partial).is_err(), "partial input: {}", partial);
    assert_eq!(ev.iter().filter(|e| e.0 == "content_block_stop" && e.1["index"] == idx).count(), 1, "the tool block is closed once");
    assert_eq!(ev.iter().filter(|e| e.0 == "content_block_start").count(), ev.iter().filter(|e| e.0 == "content_block_stop").count());
    // a complete call is still tool_use
    let r = request(port, "POST", "/v1/messages", Some(&req(false, 200, "list"))).unwrap().json();
    assert_eq!((r["stop_reason"].clone(), r["content"][1]["input"]["command"].clone()), (json!("tool_use"), json!("ls")), "{}", r);
    let l = log_lines(&log);
    for x in &l[..2] {
        assert_eq!((x["stop"]["engine"].clone(), x["stop"]["finish"].clone(), x["tool_cut"]["name"].clone()), (json!("length"), json!("max_tokens"), json!("Bash")), "{}", x);
        assert!(x["tool_cut"]["chars"].as_u64().unwrap() > 20 && x["tools"] == json!([]), "{}", x);
    }
    assert_eq!(l[1]["tool_stream"]["mismatch"], false, "{}", l[1]);
    assert_eq!(l[1]["tool_stream"]["streamed"], 1, "{}", l[1]);
    assert!(l[2]["tool_cut"].is_null() && l[2]["stop_reason"] == "tool_use", "{}", l[2]);
    for f in [&log0, &log] { let _ = std::fs::remove_file(f); }
}
