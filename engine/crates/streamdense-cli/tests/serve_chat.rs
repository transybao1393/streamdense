//! `streamdense qwen serve` chat protocol (docs/SERVE_PROTOCOL.md) end to end on the tiny model
//! (models/tiny_qwen35, vocab 2048, no tokenizer of its own): the 27B tokenizer drives it with --fold-ids
//! (prompt ids fed modulo 2048: plumbing only, the text is meaningless). Never loads the 27B weights.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

struct Srv { child: Child, stdin: Option<ChildStdin>, rx: Receiver<Value> }

impl Srv {
    fn start(extra: &[&str]) -> Option<Srv> {
        let tiny = format!("{}/models/tiny_qwen35", ROOT);
        let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
        if !std::path::Path::new(&format!("{}/config.json", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", tok)).exists() {
            eprintln!("SKIP: tiny model or 27B tokenizer.json missing");
            return None;
        }
        let mut args = vec!["qwen", "serve", "--model", &tiny, "--tokenizer", &tok, "--max-ctx", "512"];
        args.extend_from_slice(extra);
        // its own config (built-in defaults): never the repo streamdense.toml
        let conf = std::env::temp_dir().join(format!("sd_serve_chat_{}.toml", std::process::id()));
        std::fs::write(&conf, "# serve_chat test: built-in defaults\n").unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(&args)
            .env("SD_CONFIG", &conf).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
        let out = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for l in BufReader::new(out).lines() {
                let Ok(l) = l else { break };
                let v: Value = serde_json::from_str(&l).unwrap_or_else(|e| panic!("non-JSON stdout line {:?}: {}", l, e));
                if tx.send(v).is_err() { break; }
            }
        });
        let stdin = child.stdin.take();
        Some(Srv { child, stdin, rx })
    }
    fn send(&mut self, v: Value) {
        let s = self.stdin.as_mut().unwrap();
        writeln!(s, "{}", v).unwrap();
        s.flush().unwrap();
    }
    fn recv(&self) -> Value { self.rx.recv_timeout(Duration::from_secs(120)).expect("serve: no output line within 120 s") }
    /// Lines of one chat request up to and including `done`: (start line, text pieces, done line).
    fn chat(&mut self, req: Value) -> (Option<Value>, Vec<String>, Value) {
        self.send(req);
        let (mut start, mut texts) = (None, vec![]);
        loop {
            let v = self.recv();
            if v.get("start").is_some() { assert!(start.is_none() && texts.is_empty(), "start must come first once"); start = Some(v); }
            else if let Some(t) = v.get("text") { texts.push(t.as_str().unwrap().to_string()); }
            else if v.get("done").is_some() { return (start, texts, v); }
            else { panic!("unexpected line in a chat reply: {}", v); }
        }
    }
    fn close(mut self) {
        drop(self.stdin.take());
        let st = self.child.wait().unwrap();
        assert!(st.success(), "serve exit status {:?}", st);
    }
}

fn msgs(v: &[(&str, &str)]) -> Value { json!(v.iter().map(|(r, c)| json!({"role": r, "content": c})).collect::<Vec<_>>()) }

#[test]
fn serve_chat_tiny_folded() {
    let Some(mut s) = Srv::start(&["--fold-ids", "--mtp"]) else { return };
    let ready = s.recv();
    assert_eq!(ready["ready"], json!(true), "{}", ready);
    assert_eq!(ready["chat_available"], json!(true), "{}", ready);
    assert_eq!(ready["stop_ids"], json!([248046, 248044]));
    assert!(ready["tokenizer_load_ms"].as_f64().is_some());

    // turn 1: full prefill
    let mut m = vec![("system", "Be brief."), ("user", "Hello there, 你好 😀")];
    let (st, texts, d) = s.chat(json!({"op": "chat", "messages": msgs(&m), "think": false, "nnew": 8, "mtp": false, "continue": true}));
    let st = st.expect("start line");
    assert_eq!(st["kv_mode"], json!("full"));
    assert_eq!(st["kv_reused"], json!(0));
    assert_eq!(d["stop_reason"], json!("length"), "{}", d);
    assert_eq!(d["n_new"], json!(8));
    assert_eq!(d["n_prompt"], st["n_prompt"]);
    let reply = texts.concat();
    assert_eq!(d["reply"].as_str().unwrap(), reply, "done.reply == concatenated text pieces");
    for k in ["prefill_ms", "ttft_ms", "ms_per_tok", "footprint_gb", "kv_reused", "gen_tokens", "pos", "ctx_used"] {
        assert!(d.get(k).is_some(), "done misses {}", k);
    }
    let n1 = d["n_prompt"].as_u64().unwrap();

    // turn 2: history + reply + new user turn -> KV reused (prefix or followup), MTP decode
    m.push(("assistant", &reply));
    m.push(("user", "And again?"));
    let (st, texts, d) = s.chat(json!({"op": "chat", "messages": msgs(&m), "think": false, "nnew": 6, "mtp": true, "continue": true}));
    let st = st.unwrap();
    assert_ne!(st["kv_mode"], json!("full"), "{}", st);
    assert_eq!(st["kv_reused"].as_u64().unwrap(), n1 + 8 - 1, "KV holds turn 1 prompt + 7 fed reply tokens (+1 pending)");
    assert_eq!(st["n_prompt"].as_u64().unwrap(), st["kv_reused"].as_u64().unwrap() + st["n_feed"].as_u64().unwrap());
    assert_eq!(d["mtp"], json!(true));
    assert!(d["spec"].is_object(), "{}", d);
    assert_eq!(d["reply"].as_str().unwrap(), texts.concat());

    // stats
    s.send(json!({"op": "stats"}));
    let v = s.recv();
    assert_eq!(v["kv_tokens"].as_u64().unwrap(), v["pos"].as_u64().unwrap() + v["pending"].is_number() as u64);

    // continue=false -> full; thinking on
    let (st, _, d) = s.chat(json!({"op": "chat", "messages": msgs(&[("user", "x")]), "think": true, "nnew": 3, "continue": false}));
    assert_eq!(st.unwrap()["kv_mode"], json!("full"));
    assert_eq!(d["think"], json!(true));

    // stop cancels a long reply
    s.send(json!({"op": "chat", "messages": msgs(&[("user", "long one")]), "nnew": 400, "continue": false}));
    s.send(json!({"op": "stop"}));
    let d = loop { let v = s.recv(); if v.get("done").is_some() { break v; } };
    assert_eq!(d["stop_reason"], json!("cancelled"), "{}", d);
    assert!(d["n_new"].as_u64().unwrap() < 400);

    // after cancel, the KV is still reusable
    let (st, _, _) = s.chat(json!({"op": "chat", "messages": msgs(&[("user", "long one"), ("assistant", "zzz"), ("user", "next")]), "nnew": 2}));
    assert_ne!(st.unwrap()["kv_mode"], json!("full"));

    // request errors: done line with stop_reason error, no start
    for bad in [json!({"op": "chat", "messages": []}), json!({"op": "chat"}),
                json!({"op": "chat", "messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}]})] {
        let (st, texts, d) = s.chat(bad);
        assert!(st.is_none() && texts.is_empty());
        assert_eq!(d["stop_reason"], json!("error"));
        assert!(d["error"].is_string());
    }

    // context too long for max_ctx 512 -> ctx_full with n_new 0
    let long = "word ".repeat(800);
    let (_, _, d) = s.chat(json!({"op": "chat", "messages": msgs(&[("user", &long)]), "nnew": 4}));
    assert_eq!(d["stop_reason"], json!("ctx_full"));
    assert_eq!(d["n_new"], json!(0));

    // token-level gen still works; then the next chat is a full prefill
    s.send(json!({"op": "gen", "tokens": [1, 2, 3], "nnew": 4}));
    let mut toks = 0;
    let d = loop { let v = s.recv(); if v.get("tok").is_some() { toks += 1; } else { break v; } };
    assert_eq!((toks, d["done"].clone(), d["gen_tokens"].clone()), (4, json!(true), json!(4)));
    let (st, _, _) = s.chat(json!({"op": "chat", "messages": msgs(&[("user", "long one"), ("assistant", "zzz"), ("user", "next")]), "nnew": 2}));
    assert_eq!(st.unwrap()["kv_mode"], json!("full"));

    // reset, unknown op, bad JSON
    s.send(json!({"op": "reset"}));
    assert_eq!(s.recv(), json!({"reset": true, "pos": 0}));
    s.send(json!({"op": "nope"}));
    assert!(s.recv()["error"].is_string());
    writeln!(s.stdin.as_mut().unwrap(), "{{not json").unwrap();
    s.stdin.as_mut().unwrap().flush().unwrap();
    assert!(s.recv()["error"].as_str().unwrap().starts_with("bad JSON line"));
    s.close();
}

#[test]
fn serve_chat_unavailable_without_fold() {
    // the 27B tokenizer has more ids than the tiny model: chat is refused, gen still works
    let Some(mut s) = Srv::start(&[]) else { return };
    let ready = s.recv();
    assert_eq!(ready["chat_available"], json!(false));
    assert!(ready["tokenizer_error"].is_string());
    let (st, _, d) = s.chat(json!({"op": "chat", "messages": msgs(&[("user", "hi")])}));
    assert!(st.is_none());
    assert_eq!(d["stop_reason"], json!("error"));
    s.close();
}
