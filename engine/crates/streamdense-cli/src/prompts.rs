//! streamdense prompts spec [--model DIR] [--base engine/refs/qwen38_prompts.json] [--out engine/refs/qwen38_spec_prompts.json]
//!                          [--system FILE]
//! Builds the spec-bench default prompt set (`qwen spec-bench` reads it): short, code, math, ctx1k, readme2k carried
//! over from --base (their ids re-encoded with the Rust tokenizer and checked), plus `vietnamese` and `agent` (a coding-
//! agent turn: a system prompt (built-in `AGENT_SYSTEM`, or --system FILE) + six file/shell tools, the user's edit
//! request, the assistant's `read` tool call and the line-numbered file as the tool result), rendered with the chat template the gateway uses
//! (streamdense-tokenizer render_chat, thinking off, generation prompt) and encoded with the Rust tokenizer.
//! Same schema as engine/refs/qwen38_prompts.json: {"prompts":[{"id","user","text","ids","n"}]}.
use crate::Args;
use serde_json::{json, Map, Value};
use streamdense_tokenizer::{render_chat, ChatMessage, ChatOptions, ToolCall, ToolDef, Tokenizer};

fn die(e: String) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }

const CARRY: [&str; 5] = ["short", "code", "math", "ctx1k", "readme2k"];
const VIETNAMESE: &str = "Giải thích trong ba câu tại sao bầu trời có màu xanh.";
const AGENT_FILE: &str = "src/stats.rs";
const AGENT_RS: &str = "/// Running statistics over f64 samples.
pub struct Stats {
    n: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl Stats {
    pub fn new() -> Self {
        Stats { n: 0, sum: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY }
    }

    pub fn push(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    pub fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }

    pub fn range(&self) -> (f64, f64) {
        (self.min, self.max)
    }
}
";
const AGENT_REQUEST: &str = "In src/stats.rs, mean() returns NaN and range() returns (inf, -inf) when no sample was pushed. \
Make both return an Option (None when empty), add a variance() method (Welford's online algorithm, population variance, \
also None when empty) and keep the rest of the file as it is.";

/// Built-in system prompt of the `agent` prompt (a generic coding agent; --system FILE replaces it).
const AGENT_SYSTEM: &str = "You are a coding agent working in the user's project through the tools above.

Rules:
- Read the relevant code before changing it; do not guess file contents.
- Use `write` to create files and `edit` to change them; use bash to build, test and run.
- After a change, run the build or tests and fix any failures.
- When done, reply briefly: what changed and how it was verified.
";

/// The six tools of the `agent` prompt (names, descriptions, schemas, order).
fn agent_tools() -> Vec<ToolDef> {
    let s = || json!({"type": "string"});
    let sd = |d: &str| json!({"type": "string", "description": d});
    let def = |name: &str, description: &str, props: Value, required: &[&str]| ToolDef {
        name: name.into(), description: description.into(),
        parameters: json!({"type": "object", "properties": props, "required": required}),
    };
    vec![
        def("read", "Read a file, line-numbered.",
            json!({"path": s(), "offset": {"type": "integer", "description": "1-based line"}, "limit": {"type": "integer", "description": "default 400"}}), &["path"]),
        def("write", "Create or overwrite a whole file.", json!({"path": s(), "content": s()}), &["path", "content"]),
        def("edit", "Replace exact old text by new; unique unless all=true.",
            json!({"path": s(), "old": s(), "new": s(), "all": {"type": "boolean"}}), &["path", "old", "new"]),
        def("bash", "Run a zsh command in cwd. Never use it to write files.", json!({"command": s()}), &["command"]),
        def("glob", "Find files by glob, e.g. src/**/*.rs.", json!({"pattern": s(), "path": sd("default cwd")}), &["pattern"]),
        def("grep", "Search file contents by regex; returns file:line: text.", json!({"pattern": s(), "path": s(), "glob": sd("e.g. *.rs")}), &["pattern"]),
    ]
}

pub fn main(a: &Args) {
    if a.0.get(2).map(|s| s.as_str()) != Some("spec") { die("usage: streamdense prompts spec [--model DIR] [--base FILE] [--out FILE] [--system FILE]".into()); }
    let root = crate::ROOT;
    let model = a.sd("--model", &format!("{}/{}", root, crate::qwen::DEFAULT_MODEL));
    let base = a.sd("--base", &format!("{}/engine/refs/qwen38_prompts.json", root));
    let out = a.sd("--out", &format!("{}/engine/refs/qwen38_spec_prompts.json", root));
    let tk = Tokenizer::from_path(&model).unwrap_or_else(|e| die(format!("tokenizer {}: {}", model, e)));
    let bj: Value = serde_json::from_str(&std::fs::read_to_string(&base).unwrap_or_else(|e| die(format!("{}: {}", base, e))))
        .unwrap_or_else(|e| die(format!("{}: {}", base, e)));
    let opt = ChatOptions { add_generation_prompt: true, enable_thinking: false, reasoning_effort: None };
    let entry = |id: &str, user: &str, text: String| -> Value {
        let ids = tk.encode(&text);
        json!({"id": id, "user": user, "text": text, "ids": ids, "n": ids.len()})
    };
    let mut prompts: Vec<Value> = Vec::new();
    let mut carried: Vec<Value> = Vec::new();
    for id in CARRY {
        let p = bj["prompts"].as_array().and_then(|v| v.iter().find(|p| p["id"] == id)).unwrap_or_else(|| die(format!("{}: no prompt {:?}", base, id)));
        let user = p["user"].as_str().unwrap_or_default();
        let text = render_chat(&[ChatMessage::new("user", user)], &[], &opt).unwrap_or_else(|e| die(e));
        if text != p["text"].as_str().unwrap_or_default() { die(format!("{}: template render differs from the stored text", id)); }
        let ids: Vec<u64> = tk.encode(&text).into_iter().map(|x| x as u64).collect();
        let same = p["ids"].as_array().is_some_and(|v| v.iter().map(|x| x.as_u64().unwrap_or(u64::MAX)).eq(ids.iter().copied()));
        if !same { die(format!("{}: Rust tokenizer ids differ from {}", id, base)); }
        println!("{}: carried over ({} ids, Rust render + encode identical)", id, ids.len());
        carried.push(p.clone());
    }
    let viet = render_chat(&[ChatMessage::new("user", VIETNAMESE)], &[], &opt).unwrap_or_else(|e| die(e));
    let system = match a.s("--system") {
        Some(p) => std::fs::read_to_string(&p).unwrap_or_else(|e| die(format!("{}: {}", p, e))),
        None => AGENT_SYSTEM.to_string(),
    };
    let system = format!("{}\n", system.trim_end());
    let first = format!("Working directory: /work/stats\nOS: macos\nDate: 2026-11-01\n\n{}", AGENT_REQUEST);
    let mut args = Map::new();
    args.insert("path".into(), json!(AGENT_FILE));
    let numbered: String = AGENT_RS.lines().enumerate().map(|(i, l)| format!("{}\t{}\n", i + 1, l)).collect();
    let msgs = [
        ChatMessage::new("system", &system),
        ChatMessage::new("user", &first),
        ChatMessage::assistant("", None, vec![ToolCall::new("read", args)]),
        ChatMessage::tool(&numbered),
    ];
    let agent = render_chat(&msgs, &agent_tools(), &opt).unwrap_or_else(|e| die(e));
    // order: the short prompts, then the new ones, then the long contexts
    for p in &carried[..3] { prompts.push(p.clone()); }
    prompts.push(entry("vietnamese", VIETNAMESE, viet));
    prompts.push(entry("agent", AGENT_REQUEST, agent));
    for p in &carried[3..] { prompts.push(p.clone()); }
    for p in &prompts[3..5] { println!("{}: {} ids", p["id"].as_str().unwrap(), p["n"]); }
    let mut o = bj.clone();
    o["prompts"] = Value::Array(prompts);
    o["template"] = json!("streamdense-tokenizer render_chat(add_generation_prompt, enable_thinking=false; agent: coding-agent system + tools), Tokenizer::encode (streamdense prompts spec)");
    std::fs::write(&out, serde_json::to_string_pretty(&o).unwrap() + "\n").unwrap_or_else(|e| die(format!("{}: {}", out, e)));
    println!("wrote {}", out);
}
