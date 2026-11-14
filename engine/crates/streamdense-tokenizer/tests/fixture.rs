//! Fixture check against HF transformers (engine/testdata/tokenizer_fixture.json, made by
//! a separate Python script with HF transformers): encode, decode, streaming decode and chat template must match 100%.
use serde_json::Value;
use std::sync::OnceLock;
use streamdense_tokenizer::*;

fn tk() -> Option<&'static Tokenizer> {
    static T: OnceLock<Option<Tokenizer>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let p = format!("{}/tokenizer.json", DEFAULT_TOKENIZER_DIR);
        if !std::path::Path::new(&p).exists() { return None; }
        let t0 = std::time::Instant::now();
        let t = Tokenizer::from_path(DEFAULT_TOKENIZER_DIR).expect("load tokenizer");
        eprintln!("tokenizer loaded in {:.1} ms: {} ids, {} model tokens, {} merges", t0.elapsed().as_secs_f64() * 1e3,
                  t.vocab_size(), t.model_vocab_len(), t.merges_len());
        Some(t)
    }).as_ref();
    // printed per test (not once in the OnceLock) so every skipped test says why
    if t.is_none() { eprintln!("SKIP: needs models/qwen38_27b_mlx4/tokenizer.json (missing)"); }
    t
}

fn fixture() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tokenizer_fixture.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn ids(v: &Value) -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() }

fn stream_decode(t: &Tokenizer, ids: &[u32]) -> String {
    let mut d = StreamDecoder::new();
    let mut s = String::new();
    for &i in ids {
        let piece = d.push(t, i);
        assert!(!piece.contains('\u{FFFD}') || t.decode(ids).contains('\u{FFFD}'), "stream piece has U+FFFD");
        s.push_str(&piece);
    }
    s.push_str(&d.finish());
    s
}

#[test]
fn fixture_strings_encode_decode() {
    let Some(t) = tk() else { return };
    let f = fixture();
    assert_eq!(t.vocab_size() as u64, f["vocab_len"].as_u64().unwrap());
    let rows = f["strings"].as_array().unwrap();
    let mut bad = vec![];
    for r in rows {
        let text = r["text"].as_str().unwrap();
        let want = ids(&r["ids"]);
        let got = t.encode(text);
        if got != want { bad.push(format!("encode {:?}\n  want {:?}\n  got  {:?}", text, want, got)); continue; }
        let dec = r["decoded"].as_str().unwrap();
        if t.decode(&want) != dec { bad.push(format!("decode {:?}: got {:?}", dec, t.decode(&want))); }
        if stream_decode(t, &want) != dec { bad.push(format!("stream decode {:?}", dec)); }
    }
    eprintln!("strings: {}/{} match", rows.len() - bad.len(), rows.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    for r in f["decode"].as_array().unwrap() {
        assert_eq!(t.decode(&ids(&r["ids"])), r["decoded"].as_str().unwrap());
    }
}

#[test]
fn fixture_chats_template_and_ids() {
    let Some(t) = tk() else { return };
    let f = fixture();
    let rows = f["chats"].as_array().unwrap();
    for c in rows {
        let msgs: Vec<Message> = c["messages"].as_array().unwrap().iter().map(|m| Message {
            role: m["role"].as_str().unwrap().into(), content: m["content"].as_str().unwrap().into(),
            reasoning_content: m["reasoning_content"].as_str().map(|s| s.to_string()) }).collect();
        let opt = ChatOptions { add_generation_prompt: c["add_generation_prompt"].as_bool().unwrap(),
            enable_thinking: c["think"].as_bool().unwrap(), reasoning_effort: c["reasoning_effort"].as_str().map(|s| s.into()) };
        let text = render(&msgs, &opt).unwrap();
        assert_eq!(text, c["text"].as_str().unwrap(), "template text of {}", c["name"]);
        assert_eq!(t.encode(&text), ids(&c["ids"]), "ids of {}", c["name"]);
        assert_eq!(t.decode(&ids(&c["ids"])), text, "decode of {}", c["name"]);
    }
    eprintln!("chats: {}/{} match", rows.len(), rows.len());
}

/// Tool conversations (tools in the system turn, assistant tool_calls, grouped `tool` results, think on/off) from
/// the fixture's "tool_chats" (built from the synthetic agent request in testdata/agent_request.json): text and ids must match 100%.
fn chat_message(m: &Value) -> ChatMessage {
    let calls = m["tool_calls"].as_array().map(|a| a.iter().map(|c| {
        let f = if c["function"].is_object() { &c["function"] } else { c };
        ToolCall { name: f["name"].as_str().unwrap().into(), args: f["arguments"].as_object().cloned().unwrap_or_default() }
    }).collect()).unwrap_or_default();
    ChatMessage { role: m["role"].as_str().unwrap().into(), content: m["content"].as_str().unwrap_or("").into(),
                  reasoning_content: m["reasoning_content"].as_str().map(|s| s.to_string()), tool_calls: calls }
}

#[test]
fn fixture_tool_chats_template_and_ids() {
    let Some(t) = tk() else { return };
    let f = fixture();
    let rows = f["tool_chats"].as_array().expect("fixture has no tool_chats: regenerate it");
    assert!(!rows.is_empty());
    let mut bad = vec![];
    for c in rows {
        let name = c["name"].as_str().unwrap();
        let msgs: Vec<ChatMessage> = c["messages"].as_array().unwrap().iter().map(chat_message).collect();
        let tools: Vec<ToolDef> = c["tools"].as_array().unwrap().iter().map(|x| ToolDef::from_openai(x).unwrap()).collect();
        for (x, d) in c["tools"].as_array().unwrap().iter().zip(&tools) { assert_eq!(&d.to_template_value(), x, "tool def roundtrip {}", name); }
        let opt = ChatOptions { add_generation_prompt: c["add_generation_prompt"].as_bool().unwrap(),
            enable_thinking: c["think"].as_bool().unwrap(), reasoning_effort: c["reasoning_effort"].as_str().map(|s| s.into()) };
        let text = render_chat(&msgs, &tools, &opt).unwrap();
        let want = c["text"].as_str().unwrap();
        if text != want {
            let i = text.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(text.len().min(want.len()));
            let lo = i.saturating_sub(80);
            bad.push(format!("{}: text differs at byte {}\n  want ..{:?}\n  got  ..{:?}", name, i,
                             want.get(lo..(i + 80).min(want.len())).unwrap_or(""), text.get(lo..(i + 80).min(text.len())).unwrap_or("")));
            continue;
        }
        if t.encode(&text) != ids(&c["ids"]) { bad.push(format!("{}: ids differ", name)); continue; }
        if t.decode(&ids(&c["ids"])) != text { bad.push(format!("{}: decode differs", name)); }
    }
    eprintln!("tool chats: {}/{} match", rows.len() - bad.len(), rows.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The assistant tool-call turns of the fixture parse back to the same calls (render -> parse roundtrip).
#[test]
fn fixture_tool_calls_roundtrip_through_parser() {
    let f = fixture();
    let Some(rows) = f["tool_chats"].as_array() else { return };
    let mut n = 0;
    for c in rows {
        for m in c["messages"].as_array().unwrap().iter().map(chat_message).filter(|m| !m.tool_calls.is_empty()) {
            let single = ChatMessage { reasoning_content: None, ..m.clone() };
            let text = render_chat(&[single], &[], &ChatOptions { add_generation_prompt: false, ..Default::default() }).unwrap();
            let body = text.strip_prefix("<|im_start|>assistant\n").unwrap().strip_suffix("<|im_end|>\n").unwrap();
            let p = parse_reply(body);
            assert_eq!(p.visible, m.content.trim(), "{}", c["name"]);
            assert_eq!(p.calls.len(), m.tool_calls.len());
            for (got, want) in p.calls.iter().zip(&m.tool_calls) {
                assert_eq!(got.name, want.name);
                let keys: Vec<&String> = got.args.keys().collect();
                assert_eq!(keys, want.args.keys().collect::<Vec<_>>(), "arg order {}", c["name"]);
                for (k, v) in &want.args {
                    let as_text = match v { Value::String(s) => s.clone(), v => py_json(v) };
                    assert_eq!(got.args[k], Value::String(as_text), "{} arg {}", c["name"], k);
                }
            }
            n += 1;
        }
    }
    eprintln!("parser roundtrip: {} assistant tool-call turns", n);
}

#[test]
fn stream_decoder_holds_partial_utf8() {
    let Some(t) = tk() else { return };
    // byte-level tokens of a 4-byte emoji, fed one byte token at a time
    let s = "😀";
    let byte_ids: Vec<u32> = s.as_bytes().iter().map(|&b| t.encode(std::str::from_utf8(&[b]).unwrap_or("")).first().copied()
        .unwrap_or_else(|| { (0..t.vocab_size() as u32).find(|&i| t.id_to_bytes(i) == [b]).unwrap() })).collect();
    let mut d = StreamDecoder::new();
    for (k, &i) in byte_ids.iter().enumerate() {
        let p = d.push(t, i);
        if k + 1 < byte_ids.len() { assert_eq!(p, "", "partial char must be held"); } else { assert_eq!(p, s); }
    }
    assert_eq!(d.pending(), 0);
    // an invalid byte becomes U+FFFD like from_utf8_lossy, and an unfinished tail is flushed by finish()
    let ff = (0..t.vocab_size() as u32).find(|&i| t.id_to_bytes(i) == [0xFF]).unwrap();
    let f0 = (0..t.vocab_size() as u32).find(|&i| t.id_to_bytes(i) == [0xF0]).unwrap();
    assert_eq!(d.push(t, ff), "\u{FFFD}");
    assert_eq!(d.push(t, f0), "");
    assert_eq!(d.finish(), "\u{FFFD}");
}

#[test]
fn special_ids() {
    let Some(t) = tk() else { return };
    assert_eq!(t.token_id("<|im_end|>"), Some(IM_END));
    assert_eq!(t.token_id("<|endoftext|>"), Some(ENDOFTEXT));
    assert_eq!(t.token_id("<think>"), Some(248068));
    assert_eq!(t.token_id("</think>"), Some(248069));
    assert!(t.is_special(IM_END) && !t.is_special(248068));
}

#[test]
fn encode_speed_2k_tokens() {
    let Some(t) = tk() else { return };
    let f = fixture();
    let mut text = String::new();
    let parts: Vec<String> = f["strings"].as_array().unwrap().iter().map(|r| r["text"].as_str().unwrap().to_string()).collect();
    while t.encode(&text).len() < 2000 { for p in &parts { text.push_str(p); text.push('\n'); } }
    let n = t.encode(&text).len();
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { std::hint::black_box(t.encode(std::hint::black_box(&text))); }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    eprintln!("encode: {} chars -> {} tokens in {:.3} ms ({:.2} M tok/s)", text.chars().count(), n, ms, n as f64 / ms / 1e3);
    let t1 = std::time::Instant::now();
    let ids = t.encode(&text);
    for _ in 0..reps { std::hint::black_box(t.decode(&ids)); }
    eprintln!("decode: {} tokens in {:.3} ms", n, t1.elapsed().as_secs_f64() * 1e3 / reps as f64);
    assert!(ms < 50.0, "encode of ~2K tokens took {:.1} ms", ms);
}

/// Optional bigger cross-check: SD_TOK_XCHECK=file.json ([{"text","ids"}] from transformers) cargo test --release.
#[test]
fn xcheck_env_corpus() {
    let Ok(p) = std::env::var("SD_TOK_XCHECK") else { return };
    let Some(t) = tk() else { return };
    let rows: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let rows = rows.as_array().unwrap();
    let (mut ok, mut ntok) = (0, 0);
    for r in rows {
        let (text, want) = (r["text"].as_str().unwrap(), ids(&r["ids"]));
        ntok += want.len();
        let got = t.encode(text);
        if got == want { ok += 1; } else {
            let i = (0..got.len().min(want.len())).find(|&i| got[i] != want[i]).unwrap_or(got.len().min(want.len()));
            eprintln!("MISMATCH at {}: {:?}\n want {:?}\n got  {:?}", i, text.chars().take(80).collect::<String>(), &want[i..(i + 6).min(want.len())], &got[i..(i + 6).min(got.len())]);
        }
    }
    eprintln!("xcheck: {}/{} texts match ({} tokens)", ok, rows.len(), ntok);
    assert_eq!(ok, rows.len());
}
