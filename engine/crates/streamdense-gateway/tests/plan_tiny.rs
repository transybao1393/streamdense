//! `"streamdense": {"format": "plan"}` / `ban_ids` / stop sequences on the tiny model (models/tiny_qwen35):
//! 1. engine: a banned id is never emitted (plain decode and MTP verify), the output is bit-identical to plain greedy
//!    up to the first place the ban binds, fully identical when it never binds, and the ban is gone once cleared.
//! 2. gateway over HTTP (27B tokenizer, ids folded): a stop sequence ends the reply there, trimmed, stop_reason
//!    "stop_sequence" (JSON and SSE); a plan request whose ban never binds returns the plain reply.
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

fn run(eng: &QwenEngine, prompt: &[u32], ban: &[u32], mtp: bool, n: usize) -> Vec<u32> {
    eng.reset();
    assert!(Backend::set_ban(eng, ban), "set_ban");
    let mut out = vec![];
    Backend::feed(eng, prompt, n, &[], mtp, &mut |t| { out.push(t); true }).unwrap();
    assert!(Backend::set_ban(eng, &[]));
    out
}

#[test]
fn ban_never_emitted_and_lossless_tiny() {
    let Some(t) = tiny() else { return };
    let eng = QwenEngine::open(&t, QwenOptions { max_ctx: 1024, mtp: true, ..Default::default() }).unwrap();
    let v = eng.cfg.vocab as u32;
    let prompt: Vec<u32> = (0..48u32).map(|i| (i * 7919 + 13) % v).collect();
    let n = 64;
    for mtp in [false, true] {
        let plain = run(&eng, &prompt, &[], mtp, n);
        assert!(plain.len() > 8, "mtp {}: {} tokens", mtp, plain.len());
        // a ban that never binds: identical output
        let unused: Vec<u32> = (0..v).filter(|x| !plain.contains(x)).take(3).collect();
        assert_eq!(run(&eng, &prompt, &unused, mtp, n), plain, "mtp {}: non-binding ban changed the output", mtp);
        // ban the most frequent id, then it + the first id: never emitted; identical up to where the ban binds
        let mut cnt = std::collections::HashMap::new();
        for &x in &plain { *cnt.entry(x).or_insert(0) += 1; }
        let top = *cnt.iter().max_by_key(|(&x, &c)| (c, std::cmp::Reverse(x))).unwrap().0;
        for ban in [vec![top], vec![top, plain[0]]] {
            let out = run(&eng, &prompt, &ban, mtp, n);
            assert!(out.iter().all(|x| !ban.contains(x)), "mtp {}: banned {:?} emitted: {:?}", mtp, ban, out);
            let k = plain.iter().position(|x| ban.contains(x)).unwrap();
            assert_eq!(out[..k], plain[..k], "mtp {}: output before the ban binds differs", mtp);
            assert!(out.len() > k, "mtp {}: stopped at the ban", mtp);
        }
        // cleared: plain again
        assert_eq!(run(&eng, &prompt, &[], mtp, n), plain, "mtp {}: ban not cleared", mtp);
    }
}

fn body(extra: Value, stream: bool) -> Value {
    let mut b = json!({"model": "tiny", "max_tokens": 48, "stream": stream,
        "messages": [{"role": "user", "content": "Write a short plan for adding a login page."}]});
    for (k, x) in extra.as_object().unwrap() { b[k] = x.clone(); }
    b
}

fn text_of(r: &Value) -> String {
    r["content"].as_array().unwrap().iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join("")
}

#[test]
fn stop_sequences_and_plan_http_tiny() {
    let Some(t) = tiny() else { return };
    if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer missing"); return; }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = QwenEngine::open(&t, QwenOptions { max_ctx: 2048, mtp: true, ..Default::default() }).unwrap();
        let vocab = eng.cfg.vocab as u32;
        let cfg = GatewayConfig { port: 0, worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab) }, ..Default::default() };
        streamdense_gateway::serve::<QwenEngine>(&eng, tk, cfg, None, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    let port = rx.recv_timeout(Duration::from_secs(120)).unwrap();
    let post = |b: &Value| { let r = request(port, "POST", "/v1/messages", Some(b)).unwrap(); assert_eq!(r.status, 200, "{}", r.text()); r };

    let plain = post(&body(json!({}), false)).json();
    let full = text_of(&plain);
    eprintln!("plain reply ({} chars, {}): {:?}", full.chars().count(), plain["stop_reason"], full);
    // plan format: the 27B `<tool_call>` id is outside the tiny vocab, so the ban never binds -> the plain reply
    let p = post(&body(json!({"streamdense": {"format": "plan"}}), false)).json();
    assert_eq!(text_of(&p), full, "plan reply != plain reply");

    let chars: Vec<(usize, char)> = full.char_indices().collect();
    if chars.len() < 12 { eprintln!("SKIP stop part: reply too short"); return; }
    let (a, b) = (chars[chars.len() / 2].0, chars.get(chars.len() / 2 + 3).map_or(full.len(), |c| c.0));
    let stop = full[a..b].to_string();
    let want = full[..full.find(&stop).unwrap()].trim_end().to_string();
    let r = post(&body(json!({"stop_sequences": [stop.clone(), "\u{1}never\u{1}"]}), false)).json();
    assert_eq!(r["stop_reason"], "stop_sequence", "{}", r);
    assert_eq!(r["stop_sequence"], stop.as_str());
    let got = text_of(&r);
    assert_eq!(got, if want.is_empty() { "(no output)".to_string() } else { want.clone() });

    // streaming: the deltas never carry the stop text, and end at the same place
    let ev = post(&body(json!({"stop": stop.clone()}), true)).events();
    let streamed: String = ev.iter().filter(|(n, d)| n == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string()).collect();
    assert_eq!(streamed, got);
    let md = &ev.iter().find(|(n, _)| n == "message_delta").unwrap().1;
    assert_eq!(md["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(md["delta"]["stop_sequence"], stop.as_str());
}
