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
    if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer.json missing"); return None; }
    let feeds = Arc::new(std::sync::Mutex::new(vec![]));
    let f2 = feeds.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let log = log.to_string();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = Fake { pos: Cell::new(0), replies: RefCell::new(replies.into()), max_ctx: 4096, feeds: f2 };
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: None }, ..Default::default() };
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
    assert_eq!(l[0]["mode"], "full");
    assert_eq!(l[0]["tools"], json!(["Bash"]));
    assert_eq!(l[1]["mode"], "prefix", "{}", l[1]);
    assert!(l[1]["n_feed"].as_u64().unwrap() < 40, "{}", l[1]);
    assert_eq!(feeds.lock().unwrap()[1], l[1]["n_feed"].as_u64().unwrap() as usize);

    // non-stream JSON; a changed system prompt -> full prefill
    let mut b3 = body();
    b3["stream"] = json!(false);
    b3["system"] = json!("other system");
    let r = request(port, "POST", "/v1/messages", Some(&b3)).unwrap();
    assert_eq!(r.status, 200);
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
