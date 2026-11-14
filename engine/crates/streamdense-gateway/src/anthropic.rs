//! Anthropic Messages API <-> chat messages: request conversion, SSE framing, response / error JSON.
use crate::chatmap::{ChatMsg, ToolCall, ToolDef};
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};

/// The block Claude Code puts first in `system`; it changes with every Claude Code version (and per entrypoint),
/// so keeping it would break prefix reuse across sessions. Dropped.
pub const BILLING_HEADER: &str = "x-anthropic-billing-header";

#[derive(Clone, Debug)]
pub struct ChatReq {
    pub model: String,
    pub msgs: Vec<ChatMsg>,
    pub tools: Vec<ToolDef>,
    /// Thinking requested by the client (`thinking.type` enabled / adaptive); used only with `--think`.
    pub thinking_requested: bool,
    pub max_tokens: usize,
    pub stream: bool,
    /// Warm-up (`max_tokens: 0`, or the `x-streamdense-warm: 1` header set by the server): prefill only, no reply.
    pub warm: bool,
    /// `"streamdense": {"format": "plan"}`: the tool-call open token can never be emitted (banned on the GPU before
    /// the argmax: lossless greedy over the other ids; see worker.rs `Prepared::ban`).
    pub plan: bool,
    /// `"streamdense": {"ban_ids": [..]}`: more token ids that can never be emitted (any format).
    pub ban_ids: Vec<u32>,
    /// Stop sequences (`stop_sequences`, or OpenAI-style `stop`: a string or an array): the reply ends as soon as its
    /// text contains one; the stop text is trimmed; stop_reason "stop_sequence".
    pub stop: Vec<String>,
    /// `x-streamdense-project: <16 hex>` (set by the server; a client may send it on every request, warm-ups too): the
    /// project id its stable prefixes are stored under (snap_tier.rs `StableTag`). None = no header.
    pub project: Option<u64>,
    /// `x-streamdense-lane: plan` (set by the server): a planner call (with `plan`, the planner lane of `stable_tag`).
    pub lane_plan: bool,
}

impl ChatReq {
    pub fn tool_schema(&self, name: &str) -> Option<&Value> { self.tools.iter().find(|t| t.name == name).map(|t| &t.parameters) }
    pub fn tool_names(&self) -> Vec<String> { self.tools.iter().map(|t| t.name.clone()).collect() }
    /// Project + lane of this request's stable prefix (None without a project header: the token-overlap fallback).
    pub fn stable_tag(&self) -> Option<crate::snap_tier::StableTag> {
        self.project.map(|project| crate::snap_tier::StableTag { project, plan: self.plan || self.lane_plan, stat: false })
    }
    /// Conversation key (warm-up priority): the system prompt, the tool names and the first non-system message, the
    /// message-level twin of `snap_tier::session_keys`' conv key (system turn + first turn). Same process = same key.
    pub fn conv_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let sys = self.msgs.iter().take_while(|m| m.role == "system").map(|m| m.content.as_str()).collect::<Vec<_>>();
        sys.hash(&mut h);
        for t in &self.tools { t.name.hash(&mut h); }
        if let Some(m) = self.msgs.iter().find(|m| m.role != "system") { m.role.hash(&mut h); m.content.hash(&mut h); }
        h.finish()
    }
    /// Lane key: system turn(s) + tool names only. A warm-up and the real request it prepares share it even when
    /// their first user message differs (the plan lane's warm-up sends a placeholder user turn).
    pub fn lane_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let sys = self.msgs.iter().take_while(|m| m.role == "system").map(|m| m.content.as_str()).collect::<Vec<_>>();
        sys.hash(&mut h);
        for t in &self.tools { t.name.hash(&mut h); }
        h.finish()
    }
}

fn text_of_blocks(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|b| match b["type"].as_str() {
            Some("text") => b["text"].as_str().map(|s| s.to_string()),
            Some("image") => Some("[image omitted]".into()),
            Some("document") => Some("[document omitted]".into()),
            _ => None,
        }).collect::<Vec<_>>().join("\n"),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

/// Anthropic request body -> chat messages + tool defs (port of `to_qwen` in the project's earlier Python gateway, gw27b.py, not kept).
pub fn convert(body: &Value) -> Result<ChatReq, String> {
    let arr = body["messages"].as_array().ok_or("messages: must be an array")?;
    if arr.is_empty() { return Err("messages: at least one message is required".into()); }
    let mut msgs = vec![];
    let sys = match &body["system"] {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|b| b["text"].as_str())
            .filter(|t| !t.starts_with(BILLING_HEADER)).collect::<Vec<_>>().join("\n"),
        Value::Null => String::new(),
        _ => return Err("system: must be a string or an array of text blocks".into()),
    };
    if !sys.is_empty() { msgs.push(ChatMsg::new("system", &sys)); }
    for (i, m) in arr.iter().enumerate() {
        let role = m["role"].as_str().ok_or_else(|| format!("messages.{}.role: must be a string", i))?;
        if role != "user" && role != "assistant" { return Err(format!("messages.{}.role: unexpected role {:?}", i, role)); }
        let c = &m["content"];
        if let Some(s) = c.as_str() { msgs.push(ChatMsg::new(role, s)); continue; }
        let blocks = c.as_array().ok_or_else(|| format!("messages.{}.content: must be a string or an array", i))?;
        if role == "user" {
            let mut text = vec![];
            for b in blocks {
                match b["type"].as_str().unwrap_or("") {
                    "text" => text.push(b["text"].as_str().unwrap_or("").to_string()),
                    "tool_result" => {
                        let mut r = text_of_blocks(&b["content"]);
                        if b["is_error"].as_bool() == Some(true) { r = format!("[error] {}", r); }
                        msgs.push(ChatMsg::tool(&r));
                    }
                    "image" => text.push("[image omitted]".into()),
                    "document" => text.push("[document omitted]".into()),
                    _ => {}
                }
            }
            if !text.is_empty() { msgs.push(ChatMsg::new("user", &text.join("\n"))); }
        } else {
            let mut text = String::new();
            let mut reasoning: Option<String> = None;
            let mut calls = vec![];
            for b in blocks {
                match b["type"].as_str().unwrap_or("") {
                    "text" => text.push_str(b["text"].as_str().unwrap_or("")),
                    "thinking" => { reasoning.get_or_insert_with(String::new).push_str(b["thinking"].as_str().unwrap_or("")); }
                    "tool_use" => calls.push(ToolCall::new(b["name"].as_str().unwrap_or(""),
                        b["input"].as_object().cloned().unwrap_or_else(Map::new))),
                    _ => {}
                }
            }
            msgs.push(ChatMsg::assistant(&text, reasoning.as_deref(), calls));
        }
    }
    let mut tools = vec![];
    if let Some(ts) = body["tools"].as_array() {
        for t in ts {
            // server tools (web_search_..., etc.) have no input_schema: the local model cannot run them
            if t.get("input_schema").is_none() { continue; }
            if let Ok(d) = ToolDef::from_anthropic(t) { tools.push(d); }
        }
    }
    let thinking_requested = matches!(body["thinking"]["type"].as_str(), Some("enabled") | Some("adaptive"));
    let sd = &body["streamdense"];
    let plan = match &sd["format"] {
        Value::Null => false,
        Value::String(f) if f == "plan" => true,
        f => return Err(format!("streamdense.format: unknown format {}", f)),
    };
    let ban_ids = match &sd["ban_ids"] {
        Value::Null => vec![],
        Value::Array(a) => a.iter().map(|x| x.as_u64().filter(|&v| v <= u32::MAX as u64).map(|v| v as u32))
            .collect::<Option<Vec<u32>>>().ok_or("streamdense.ban_ids: must be an array of token ids")?,
        _ => return Err("streamdense.ban_ids: must be an array of token ids".into()),
    };
    let mut stop: Vec<String> = vec![];
    for key in ["stop_sequences", "stop"] {
        match &body[key] {
            Value::Null => {}
            Value::String(x) => stop.push(x.clone()),
            Value::Array(a) => for x in a { stop.push(x.as_str().ok_or_else(|| format!("{}: must be strings", key))?.to_string()); },
            _ => return Err(format!("{}: must be a string or an array of strings", key)),
        }
    }
    stop.retain(|x| !x.is_empty());
    stop.dedup();
    Ok(ChatReq {
        model: body["model"].as_str().unwrap_or("qwen38-27b").to_string(),
        msgs, tools, thinking_requested,
        max_tokens: body["max_tokens"].as_u64().unwrap_or(4096).max(1) as usize,
        stream: body["stream"].as_bool().unwrap_or(false),
        warm: body["max_tokens"].as_u64() == Some(0),
        plan, ban_ids, stop,
        project: None, lane_plan: false,
    })
}

/// `x-streamdense-project` header value: exactly 16 hex digits -> the id (anything else: None).
pub fn parse_project(v: &str) -> Option<u64> {
    let v = v.trim();
    if v.len() == 16 && v.bytes().all(|b| b.is_ascii_hexdigit()) { u64::from_str_radix(v, 16).ok() } else { None }
}

// ---------------------------------------------------------------- ids, SSE framing, payloads
static SEQ: AtomicU64 = AtomicU64::new(1);

/// `prefix` + 24 hex chars (time ^ counter mix; unique per process, no RNG crate needed).
pub fn new_id(prefix: &str) -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut x = t ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((std::process::id() as u64) << 40);
    x ^= x >> 33; x = x.wrapping_mul(0xff51_afd7_ed55_8ccd); x ^= x >> 33;
    format!("{}{:016x}{:08x}", prefix, x, n as u32)
}

/// One server-sent event: `event: NAME\ndata: JSON\n\n` (the JSON on one line).
pub fn sse(name: &str, data: &Value) -> String { format!("event: {}\ndata: {}\n\n", name, data) }

pub fn usage(input: usize, cached: usize, output: usize) -> Value {
    json!({"input_tokens": input, "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached, "output_tokens": output})
}

pub fn message_start(id: &str, model: &str, usage_v: Value) -> String {
    sse("message_start", &json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant",
        "model": model, "content": [], "stop_reason": null, "stop_sequence": null, "usage": usage_v}}))
}
pub fn text_block_start(index: usize) -> String {
    sse("content_block_start", &json!({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}))
}
pub fn text_delta(index: usize, text: &str) -> String {
    sse("content_block_delta", &json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}))
}
pub fn block_stop(index: usize) -> String { sse("content_block_stop", &json!({"type": "content_block_stop", "index": index})) }
/// A whole tool_use block: start (empty input), one input_json_delta with the coerced arguments, stop.
pub fn tool_use_block(index: usize, id: &str, name: &str, input: &Value) -> String {
    let mut s = sse("content_block_start", &json!({"type": "content_block_start", "index": index,
        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}));
    s += &sse("content_block_delta", &json!({"type": "content_block_delta", "index": index,
        "delta": {"type": "input_json_delta", "partial_json": input.to_string()}}));
    s += &block_stop(index);
    s
}
pub fn message_delta(stop_reason: &str, stop_sequence: Option<&str>, usage_v: Value) -> String {
    sse("message_delta", &json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence}, "usage": usage_v}))
}
pub fn message_stop() -> String { sse("message_stop", &json!({"type": "message_stop"})) }
pub fn ping() -> String { sse("ping", &json!({"type": "ping"})) }
/// a ping with a phase hint (`prefill`: the prompt is being read; `decode`: the first token came), for clients that
/// show what the model is doing; other clients ignore pings
pub fn ping_phase(phase: &str) -> String { sse("ping", &json!({"type": "ping", "phase": phase})) }
/// a ping naming the tool call being written (`phase: tool`, its name and `path` argument, empty until complete)
pub fn ping_tool(name: &str, path: &str) -> String { sse("ping", &json!({"type": "ping", "phase": "tool", "name": name, "path": path})) }
pub fn sse_error(kind: &str, msg: &str) -> String { sse("error", &error_json(kind, msg)) }

/// Anthropic error body. kind: invalid_request_error | not_found_error | api_error | overloaded_error.
pub fn error_json(kind: &str, msg: &str) -> Value { json!({"type": "error", "error": {"type": kind, "message": msg}}) }

/// Non-streaming response body.
pub fn message_json(id: &str, model: &str, content: Vec<Value>, stop_reason: &str, stop_sequence: Option<&str>, usage_v: Value) -> Value {
    json!({"id": id, "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": stop_reason, "stop_sequence": stop_sequence, "usage": usage_v})
}

pub fn ctx_full_text(n_prompt: usize, max_ctx: usize) -> String {
    format!("[streamdense] Context full: this conversation is {} tokens and the engine limit is {} (--max-ctx). Run /clear to start fresh.",
        n_prompt, max_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    #[test]
    fn convert_agent_body() {
        let f = fixture();
        assert_eq!(f["path"], "/v1/messages?beta=true");
        let r = convert(&f["body"]).unwrap();
        assert!(r.stream && r.thinking_requested);
        assert_eq!(r.max_tokens, 32000);
        assert_eq!(r.model, "qwen38-27b");
        assert_eq!(r.msgs.len(), 2);
        assert_eq!(r.msgs[0].role, "system");
        assert!(!r.msgs[0].content.contains(BILLING_HEADER), "billing header block dropped");
        assert_eq!(r.msgs[0].content, "You are a test assistant for the streamdense gateway.\nYou are a coding agent on the user's Mac. Use the tools to read, edit and run code. Be concise.");
        assert_eq!(r.msgs[1], ChatMsg::new("user", "say hi"));
        assert_eq!(r.tool_names(), vec!["Bash", "Edit", "Read"]);
        assert_eq!(r.tool_schema("Read").unwrap()["properties"]["limit"]["type"], "integer");
        // the rendered prompt carries the tools, never the billing header
        let s = crate::chatmap::render(&r.msgs, &r.tools, false, true).unwrap();
        assert!(s.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"Bash\", \"description\": \"execute shell commands\", \"parameters\": {\"$schema\""));
        assert!(!s.contains(BILLING_HEADER));
        // a different Claude Code version changes only the dropped block -> identical render
        let mut g = f["body"].clone();
        g["system"][0]["text"] = json!("x-anthropic-billing-header: cc_version=9.9.9; cc_entrypoint=cli;");
        assert_eq!(crate::chatmap::render(&convert(&g).unwrap().msgs, &r.tools, false, true).unwrap(), s);
    }

    #[test]
    fn convert_tool_round_trip() {
        let mut b = fixture()["body"].clone();
        b["messages"] = json!([
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure."},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls", "timeout": 5}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt"}, {"type": "text", "text": "b.txt"}]},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "boom", "is_error": true},
                {"type": "text", "text": "thanks"}]}]);
        let r = convert(&b).unwrap();
        let roles: Vec<&str> = r.msgs.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "tool", "user"]);
        assert_eq!(r.msgs[2].content, "Sure.");
        assert_eq!(r.msgs[2].tool_calls, vec![ToolCall::new("Bash", json!({"command": "ls", "timeout": 5}).as_object().unwrap().clone())]);
        assert_eq!(r.msgs[3].content, "a.txt\nb.txt");
        assert_eq!(r.msgs[4].content, "[error] boom");
        assert_eq!(r.msgs[5].content, "thanks");
        assert!(convert(&json!({"messages": []})).is_err());
        assert!(!r.plan && r.ban_ids.is_empty() && r.stop.is_empty());
        let p = convert(&json!({"messages": [{"role": "user", "content": "x"}], "streamdense": {"format": "plan", "ban_ids": [5, 7]},
            "stop_sequences": ["\n\n", ""], "stop": "END"})).unwrap();
        assert!(p.plan);
        assert_eq!(p.ban_ids, vec![5, 7]);
        assert_eq!(p.stop, vec!["\n\n".to_string(), "END".to_string()]);
        assert!(convert(&json!({"messages": [{"role": "user", "content": "x"}], "streamdense": {"format": "json"}})).is_err());
        assert!(convert(&json!({"messages": [{"role": "user", "content": "x"}], "streamdense": {"ban_ids": ["a"]}})).is_err());
        assert!(convert(&json!({"messages": [{"role": "system", "content": "x"}]})).is_err());
    }

    #[test]
    fn sse_framing() {
        let s = text_delta(0, "a\"b\nc");
        assert!(s.starts_with("event: content_block_delta\ndata: {") && s.ends_with("}\n\n") && s.trim_end().lines().count() == 2, "{:?}", s);
        let d: Value = serde_json::from_str(s.lines().nth(1).unwrap().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(d, json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a\"b\nc"}}));
        let t = tool_use_block(1, "toolu_x", "Bash", &json!({"command": "ls"}));
        let ev: Vec<&str> = t.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert_eq!(ev.len(), 3);
        for e in &ev {
            let mut l = e.lines();
            assert!(l.next().unwrap().starts_with("event: "));
            let d: Value = serde_json::from_str(l.next().unwrap().strip_prefix("data: ").unwrap()).unwrap();
            assert_eq!(d["index"], 1);
            assert!(l.next().is_none(), "one data line per event");
        }
        let d: Value = serde_json::from_str(ev[1].lines().nth(1).unwrap().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(serde_json::from_str::<Value>(d["delta"]["partial_json"].as_str().unwrap()).unwrap(), json!({"command": "ls"}));
        let m = message_start("msg_1", "qwen38-27b", usage(10, 5, 0));
        let d: Value = serde_json::from_str(m.lines().nth(1).unwrap().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(d["message"]["usage"]["cache_read_input_tokens"], 5);
        assert_eq!(d["message"]["stop_reason"], Value::Null);
        assert!(new_id("msg_") != new_id("msg_"));
        assert_eq!(new_id("toolu_").len(), 6 + 24);
    }
}
