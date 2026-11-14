//! `streamdense gateway` end to end over real HTTP on the tiny model (models/tiny_qwen35, vocab 2048) with the 27B
//! tokenizer and --fold-ids (prompt ids fed modulo 2048: engine-core plumbing only, the text is meaningless; the
//! tiny model never emits <|im_end|> or <tool_call>, so replies end on max_tokens). Never loads the 27B weights.
//! Uses --port 0 (a free port, read from the ready line): never touches 8898.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use streamdense_gateway::client::{request, Response};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

struct Gw { child: Child, port: u16, log: String, snaps: String }
impl Drop for Gw {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); let _ = std::fs::remove_file(&self.log); let _ = std::fs::remove_dir_all(&self.snaps); }
}

fn start() -> Option<Gw> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", tok)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer.json missing");
        return None;
    }
    let log = std::env::temp_dir().join(format!("sd_gateway_tiny_{}.jsonl", std::process::id())).display().to_string();
    let _ = std::fs::remove_file(&log);
    let snaps = std::env::temp_dir().join(format!("sd_gateway_tiny_snaps_{}", std::process::id())).display().to_string();
    let _ = std::fs::remove_dir_all(&snaps);
    // its own config (built-in defaults): never the repo streamdense.toml (its 27B draft_vocab etc. do not fit the tiny model)
    let conf = std::env::temp_dir().join(format!("sd_gateway_tiny_{}.toml", std::process::id())).display().to_string();
    std::fs::write(&conf, "# gateway_http test: built-in defaults\n").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_streamdense"))
        .args(["gateway", "--model", &tiny, "--tokenizer", &tok, "--max-ctx", "1024", "--port", "0", "--fold-ids", "--mtp", "--log", &log, "--snapshot-dir", &snaps, "--watcher-dir", &format!("{}/watcher", snaps)])
        .env("SD_CONFIG", &conf).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let out = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for l in BufReader::new(out).lines() { let Ok(l) = l else { break }; let _ = tx.send(l); }
    });
    let line = rx.recv_timeout(Duration::from_secs(120)).expect("gateway: no ready line within 120 s");
    let ready: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("ready line {:?}: {}", line, e));
    assert_eq!(ready["ready"], json!(true), "{}", ready);
    assert_eq!(ready["mtp_decode"], json!(true), "{}", ready);
    // 12 GB mode off by default: full model, snapshots on, no RAM-mode fields
    assert_eq!(ready["forheader"], json!({"mode": "off"}), "{}", ready);
    assert_eq!(ready["snapshots"], json!(true), "{}", ready);
    Some(Gw { child, port: ready["port"].as_u64().unwrap() as u16, log, snaps })
}

fn log_lines(g: &Gw) -> Vec<Value> {
    std::fs::read_to_string(&g.log).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// Event names in order (pings dropped) + streamed text + stop_reason.
fn stream_summary(r: &Response) -> (Vec<String>, String, String) {
    assert_eq!(r.status, 200, "{}", r.text());
    let ev: Vec<(String, Value)> = r.events().into_iter().filter(|(n, _)| n != "ping").collect();
    let names: Vec<String> = ev.iter().map(|(n, _)| n.clone()).collect();
    let text: String = ev.iter().filter(|(n, d)| n == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string()).collect();
    let stop = ev.iter().find(|(n, _)| n == "message_delta").map(|(_, d)| d["delta"]["stop_reason"].as_str().unwrap().to_string()).unwrap_or_default();
    (names, text, stop)
}

fn tools() -> Value {
    json!([{"name": "Bash", "description": "execute shell commands", "input_schema": {"type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"]}}])
}

#[test]
fn gateway_tiny_folded_http() {
    let Some(g) = start() else { return };
    let p = g.port;
    let h = request(p, "GET", "/health", None).unwrap();
    assert_eq!(h.status, 200);
    assert!(h.json().get("ram_mode").is_none(), "default off: no ram_mode in /health");
    assert_eq!(request(p, "GET", "/v1/models", None).unwrap().json()["data"][0]["id"], "qwen38-27b");

    // turn 1: streamed reply, Anthropic event order, ends on max_tokens
    let sys = json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1.0.0;"}, {"type": "text", "text": "Be brief."}]);
    let mut msgs = vec![json!({"role": "user", "content": [{"type": "text", "text": "Hello there"}]})];
    let b1 = json!({"model": "qwen38-27b", "system": sys, "tools": tools(), "messages": msgs, "max_tokens": 8, "stream": true});
    let r = request(p, "POST", "/v1/messages?beta=true", Some(&b1)).unwrap();
    let (names, text1, stop) = stream_summary(&r);
    assert_eq!(names.first().unwrap(), "message_start");
    assert_eq!(&names[names.len() - 2..], &["message_delta".to_string(), "message_stop".to_string()]);
    assert_eq!(names[1], "content_block_start");
    assert_eq!(stop, "max_tokens");

    // turn 2: a tool round trip (our reply + a fake tool_use + its tool_result) -> KV reused (followup: the fake
    // tool_use was never generated)
    msgs.push(json!({"role": "assistant", "content": [{"type": "text", "text": text1},
        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls", "timeout": 5}}]}));
    msgs.push(json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt"}]}]}));
    let b2 = json!({"model": "qwen38-27b", "system": sys, "tools": tools(), "messages": msgs, "max_tokens": 6, "stream": true});
    let (names, _, stop) = stream_summary(&request(p, "POST", "/v1/messages?beta=true", Some(&b2)).unwrap());
    assert_eq!(names.last().unwrap(), "message_stop");
    assert_eq!(stop, "max_tokens");

    let (_, text2, _) = stream_summary(&request(p, "POST", "/v1/messages?beta=true", Some(&b2)).unwrap());
    let mut b3 = b2.clone();
    b3["messages"].as_array_mut().unwrap().push(json!({"role": "assistant", "content": text2}));
    b3["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": "and?"}));
    b3["max_tokens"] = json!(2);
    let (names, _, _) = stream_summary(&request(p, "POST", "/v1/messages", Some(&b3)).unwrap());
    assert_eq!(names.last().unwrap(), "message_stop");
    // KV reuse: prefix, or followup when the tiny model's random text does not re-tokenize to the same ids
    // (prefix is pinned by the fake-engine test in streamdense-gateway/tests/http_fake.rs)
    // 4 requests: b1, b2, b2 again (the state already ran past that prompt and DeltaNet cannot rewind -> full prefill,
    // but b1's rolling snapshot (b1's prompt minus its last token, a prefix of b2) restores), b3
    let l = log_lines(&g);
    assert_eq!(l.len(), 4, "{:?}", l);
    assert!(l.iter().all(|x| x.get("ram_mode").is_none()), "default off: no ram_mode in the log");
    assert_eq!(l[0]["mode"], "full");
    assert_eq!(l[2]["mode"], "restore", "identical repeat: {}", l[2]);
    assert_eq!(l[2]["snap"][0]["kind"], "restore", "{}", l[2]);
    assert!(l[2]["n_reused"].as_u64().unwrap() > 200, "{}", l[2]);
    assert!(l[0]["n_prompt"].as_u64().unwrap() > 200, "tools in the prompt: {}", l[0]);
    for x in [&l[1], &l[3]] {
        assert!(x["mode"] == "prefix" || x["mode"] == "followup", "KV reused: {}", x);
        assert!(x["n_reused"].as_u64().unwrap() > 0);
        assert!(x["n_feed"].as_u64().unwrap() < x["n_prompt"].as_u64().unwrap());
        assert_eq!(x["mtp"], json!(true));
    }

    // non-stream JSON
    let mut b4 = b1.clone();
    b4["stream"] = json!(false);
    b4["max_tokens"] = json!(3);
    let j = request(p, "POST", "/v1/messages", Some(&b4)).unwrap().json();
    assert_eq!(j["type"], "message");
    assert_eq!(j["stop_reason"], "max_tokens");
    assert_eq!(j["usage"]["output_tokens"], 3);

    // count_tokens
    let ct = request(p, "POST", "/v1/messages/count_tokens", Some(&b1)).unwrap().json();
    assert_eq!(ct["input_tokens"], l[0]["n_prompt"]);

    // context full (max_ctx 1024): friendly text, end_turn
    let b5 = json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": "word ".repeat(2000)}], "max_tokens": 4, "stream": true});
    let (_, text, stop) = stream_summary(&request(p, "POST", "/v1/messages", Some(&b5)).unwrap());
    assert!(text.starts_with("[streamdense] Context full") && text.contains("/clear"), "{}", text);
    assert_eq!(stop, "end_turn");

    // the gateway still works after ctx_full (b1 again: its rolling snapshot restores)
    let (_, _, stop) = stream_summary(&request(p, "POST", "/v1/messages", Some(&b1)).unwrap());
    assert_eq!(stop, "max_tokens");
    assert_eq!(log_lines(&g).last().unwrap()["mode"], "restore");
}

#[test]
fn gateway_print_env() {
    let conf = std::env::temp_dir().join(format!("sd_print_env_{}.toml", std::process::id()));
    std::fs::write(&conf, "# print-env test: built-in defaults\n").unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(["gateway", "--print-env", "--port", "8898"]).env("SD_CONFIG", &conf).output().unwrap();
    let _ = std::fs::remove_file(&conf);
    assert!(o.status.success());
    let s = String::from_utf8(o.stdout).unwrap();
    for k in ["export ANTHROPIC_BASE_URL=http://127.0.0.1:8898", "export ANTHROPIC_AUTH_TOKEN=dummy", "export ANTHROPIC_API_KEY=\n",
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "ANTHROPIC_MODEL=qwen38-27b", "ANTHROPIC_SMALL_FAST_MODEL=qwen38-27b",
              "ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen38-27b", "ANTHROPIC_DEFAULT_SONNET_MODEL=qwen38-27b", "ANTHROPIC_DEFAULT_OPUS_MODEL=qwen38-27b"] {
        assert!(s.contains(k), "missing {:?} in\n{}", k, s);
    }
}
