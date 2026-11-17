//! Tool-call control + MTP context-draft stats on the tiny model (models/tiny_qwen35):
//! 1. engine allow-list (`set_allow`, toolctl.rs name constraint): every emitted id is allowed, the pick is the greedy
//!    one among the allowed ids (plain decode and MTP verify), cleared by `set_ban(&[])`.
//! 2. MTP + context drafts (`mtp_ctx_draft` > 0): `FeedStats::ctx_draft` is filled ("mode": "mtp_ctx", passes split
//!    by draft source) on a copy-heavy prompt, and the output equals the engine without context drafts.
//! 3. gateway over HTTP (27B tokenizer, ids folded): tool_choice tool -> a tool_use of that tool (the forced opener
//!    never leaks as text); any -> only allowed tool names; auto == absent; allowed_tools [] -> never a tool_use.
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::client::request;
use streamdense_gateway::{Backend, GatewayConfig, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

fn tiny() -> Option<String> {
    let t = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", t)).exists() { eprintln!("SKIP: tiny model missing"); return None; }
    Some(t)
}

fn run(eng: &QwenEngine, prompt: &[u32], allow: Option<&[u32]>, ban: &[u32], mtp: bool, n: usize) -> Vec<u32> {
    eng.reset();
    match allow { Some(a) => assert!(Backend::set_allow(eng, a), "set_allow"), None => assert!(Backend::set_ban(eng, ban), "set_ban") }
    let mut out = vec![];
    Backend::feed(eng, prompt, n, &[], mtp, &mut |t| { out.push(t); true }).unwrap();
    assert!(Backend::set_ban(eng, &[]));
    out
}

#[test]
fn allow_list_is_greedy_among_allowed_tiny() {
    let Some(t) = tiny() else { return };
    let eng = QwenEngine::open(&t, QwenOptions { max_ctx: 1024, mtp: true, ..Default::default() }).unwrap();
    let v = eng.cfg.vocab as u32;
    let prompt: Vec<u32> = (0..48u32).map(|i| (i * 7919 + 13) % v).collect();
    let n = 32;
    for mtp in [false, true] {
        let plain = run(&eng, &prompt, None, &[], mtp, n);
        let g = plain[0];
        // second-best first token: the greedy pick with g banned
        let h = run(&eng, &prompt, None, &[g], mtp, 1)[0];
        assert_ne!(g, h);
        // allowed set holding the argmax: same first token
        let others: Vec<u32> = (0..v).filter(|x| *x != g && *x != h).step_by(997).take(20).collect();
        let mut a = others.clone();
        a.push(g);
        let out = run(&eng, &prompt, Some(&a), &[], mtp, n);
        assert_eq!(out[0], g, "mtp {}: allowed argmax not picked", mtp);
        assert!(out.iter().all(|x| a.contains(x)), "mtp {}: id outside the allow-list: {:?}", mtp, out);
        // without the argmax: the runner-up h beats every other allowed id
        let mut b = others.clone();
        b.push(h);
        let out = run(&eng, &prompt, Some(&b), &[], mtp, n);
        assert_eq!(out[0], h, "mtp {}: greedy among the allowed ids", mtp);
        assert!(out.iter().all(|x| b.contains(x)), "mtp {}: id outside the allow-list: {:?}", mtp, out);
        // cleared: plain again
        assert_eq!(run(&eng, &prompt, None, &[], mtp, n), plain, "mtp {}: allow-list not cleared", mtp);
    }
}

#[test]
fn mtp_ctx_draft_stats_tiny() {
    let Some(t) = tiny() else { return };
    let opt = |d: usize| QwenOptions { max_ctx: 1024, mtp: true, spec_auto: true, spec_depth: 7, mtp_ctx_draft: d, ..Default::default() };
    let v;
    // copy-heavy: a block repeated three times
    let mk = |v: u32| -> Vec<u32> { let b: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 13) % v).collect(); [b.clone(), b.clone(), b].concat() };
    let base = {
        let eng = QwenEngine::open(&t, opt(0)).unwrap();
        v = eng.cfg.vocab as u32;
        let mut out = vec![];
        let st = Backend::feed(&eng, &mk(v), 96, &[], true, &mut |x| { out.push(x); true }).unwrap();
        assert!(st.ctx_draft.is_none(), "no stats without context drafts");
        out
    };
    let eng = QwenEngine::open(&t, opt(7)).unwrap();
    let mut out = vec![];
    let st = Backend::feed(&eng, &mk(v), 96, &[], true, &mut |x| { out.push(x); true }).unwrap();
    assert_eq!(out, base, "context drafts changed the output");
    let c = st.ctx_draft.expect("ctx_draft stats with mtp_ctx_draft > 0");
    eprintln!("mtp_ctx stats: {}", c);
    assert_eq!(c["mode"], "mtp_ctx");
    let n = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("{} missing: {}", k, c));
    assert!(n("passes") > 0);
    assert_eq!(n("passes_ctx") + n("passes_mixed") + n("passes_mtp") + n("passes_nodraft"), n("passes"));
    assert!(n("accepted") <= n("drafted"));
    assert_eq!(c["lookup"][0], c["drafted"]);
}

fn body(extra: Value) -> Value {
    let mut b = json!({"model": "tiny", "max_tokens": 32, "stream": false,
        "messages": [{"role": "user", "content": "List the files in the current directory."}],
        "tools": [{"name": "Bash", "description": "Run a shell command.",
                   "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
                  {"name": "Read", "description": "Read a file.",
                   "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]});
    for (k, x) in extra.as_object().unwrap() { b[k] = x.clone(); }
    b
}

fn text_of(r: &Value) -> String {
    r["content"].as_array().unwrap().iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join("")
}
fn tool_names(r: &Value) -> Vec<String> {
    r["content"].as_array().unwrap().iter().filter(|c| c["type"] == "tool_use").map(|c| c["name"].as_str().unwrap().to_string()).collect()
}

#[test]
fn tool_choice_http_tiny() {
    let Some(t) = tiny() else { return };
    if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer missing"); return; }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = QwenEngine::open(&t, QwenOptions { max_ctx: 2048, mtp: true, ..Default::default() }).unwrap();
        let vocab = eng.cfg.vocab as u32;
        let cfg = GatewayConfig { port: 0, worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab), ..Default::default() }, ..Default::default() };
        streamdense_gateway::serve::<QwenEngine>(&eng, tk, cfg, None, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    let port = rx.recv_timeout(Duration::from_secs(120)).unwrap();
    let post = |b: &Value, ok: u16| { let r = request(port, "POST", "/v1/messages", Some(b)).unwrap(); assert_eq!(r.status, ok, "{}", r.text()); r };

    let plain = post(&body(json!({})), 200).json();
    let auto = post(&body(json!({"tool_choice": {"type": "auto"}})), 200).json();
    assert_eq!(auto["content"], plain["content"], "tool_choice auto != absent");
    for tc in [json!({"type": "tool", "name": "Read"}), json!({"type": "function", "function": {"name": "Read"}})] {
        let r = post(&body(json!({"tool_choice": tc})), 200).json();
        eprintln!("forced Read: {}", r);
        assert_eq!(tool_names(&r), vec!["Read".to_string()], "{}", r);
        assert_eq!(r["stop_reason"], "tool_use");
        assert!(!text_of(&r).contains("<tool_call>") && !text_of(&r).contains("<function"), "opener leaked: {}", r);
    }
    // streaming: the forced opener never shows as text
    let ev = request(port, "POST", "/v1/messages", Some(&body(json!({"stream": true, "tool_choice": {"type": "tool", "name": "Bash"}})))).unwrap().events();
    let streamed: String = ev.iter().filter(|(n, d)| n == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string()).collect();
    assert!(!streamed.contains("<tool_call>") && !streamed.contains("<function"), "{:?}", streamed);
    assert!(ev.iter().any(|(n, d)| n == "content_block_start" && d["content_block"]["name"] == "Bash"), "no Bash tool_use block");
    let r = post(&body(json!({"tool_choice": {"type": "any"}})), 200).json();
    eprintln!("forced any: {}", r);
    assert!(tool_names(&r).iter().all(|n| n == "Bash" || n == "Read"), "{}", r);
    assert!(!text_of(&r).contains("<tool_call>"));
    let r = post(&body(json!({"streamdense": {"allowed_tools": []}})), 200).json();
    assert!(tool_names(&r).is_empty(), "{}", r);
    // errors: unknown forced tool, forced tool not allowed
    post(&body(json!({"tool_choice": {"type": "tool", "name": "Nope"}})), 400);
    post(&body(json!({"tool_choice": {"type": "tool", "name": "Bash"}, "streamdense": {"allowed_tools": ["Read"]}})), 400);
}
