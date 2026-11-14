//! Qwen3.8 chat template (models/qwen38_27b_mlx4/chat_template.jinja), written out by hand for the cases the
//! chat UI and the gateway use: system / user / assistant / tool messages with string content, tool definitions,
//! assistant tool calls, `add_generation_prompt`, `enable_thinking` (+ `reasoning_effort`). No images / videos.
//!
//! Origin: the template's fixed strings are reproduced word for word from `chat_template.jinja` of Qwen/Qwen3.8-27B,
//! licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Changes: rewritten from
//! Jinja into Rust; images / videos left out. See THIRD_PARTY_NOTICES.md.
//!
//! What the template does (and this file reproduces):
//! * leading system (or developer) messages are merged: each `|trim`med, the non-empty ones joined by "\n";
//! * enable_thinking true -> a reasoning-effort instruction (xhigh default, "high" == xhigh, medium -> none, low)
//!   goes first in the system turn ("<instr>\n\n<system>", or alone when there is no system message);
//! * user turns: `<|im_start|>user\n{content|trim}<|im_end|>\n`;
//! * assistant turns: `preserve_thinking` is undefined in apply_chat_template -> treated as true, so EVERY
//!   assistant turn renders `<|im_start|>assistant\n<think>\n{reasoning_content|trim}\n</think>\n\n{content|trim}<|im_end|>\n`;
//! * tools (non-empty list): the system turn becomes `<|im_start|>system\n` + [reasoning instr + "\n\n"] +
//!   "# Tools ..." block with one `tojson` line per tool + format instructions + ["\n\n" + merged system] +
//!   `<|im_end|>\n`;
//! * assistant tool calls follow the content: first call prefixed "\n\n" when content is non-empty, later calls
//!   "\n"; each `<tool_call>\n<function=NAME>\n` + per argument `<parameter=K>\n{V}\n</parameter>\n` (V: strings
//!   as-is, everything else `tojson`) + `</function>\n</tool_call>`;
//! * `tool` messages: a run of consecutive tool messages is one user turn: `<|im_start|>user` (only when there is a
//!   previous message that is not a tool message) + per message `\n<tool_response>\n{content|trim}\n</tool_response>`
//!   + `<|im_end|>\n` after the last one of the run;
//! * generation prompt: `<|im_start|>assistant\n` + (`<think>\n\n</think>\n\n` if thinking is off else `<think>\n`).

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message { Message { role: role.into(), content: content.into(), reasoning_content: None } }

    /// An assistant reply exactly as generated (after the `<think>\n` of the prompt when thinking was on):
    /// split at the last `</think>` into reasoning (leading `<think>` dropped) and content. Without `</think>`
    /// the text is all content.
    pub fn assistant_from_raw(raw: &str) -> Message {
        match raw.rfind("</think>") {
            Some(k) => {
                let r = raw[..k].trim_start();
                let r = r.strip_prefix("<think>").unwrap_or(r);
                Message { role: "assistant".into(), content: raw[k + "</think>".len()..].to_string(), reasoning_content: Some(r.to_string()) }
            }
            None => Message::new("assistant", raw),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    /// None = template default ("xhigh"). Used only when thinking is on.
    pub reasoning_effort: Option<String>,
}

impl Default for ChatOptions {
    fn default() -> Self { ChatOptions { add_generation_prompt: true, enable_thinking: false, reasoning_effort: None } }
}

/// Jinja `|trim` = Python `str.strip()` (Unicode whitespace incl. U+001C..U+001F).
fn py_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn is_system(role: &str) -> bool { role == "system" || role == "developer" }


use crate::tools::{parse_reply, py_json, ToolCall};
use serde_json::Value;

/// A chat message with tool support (see crate docs). `tool_calls` is used for assistant messages only.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.into(), content: content.into(), reasoning_content: None, tool_calls: vec![] }
    }
    /// A tool result (role "tool").
    pub fn tool(content: &str) -> ChatMessage { ChatMessage::new("tool", content) }
    pub fn assistant(content: &str, reasoning: Option<&str>, calls: Vec<ToolCall>) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), reasoning_content: reasoning.map(|r| r.into()), tool_calls: calls }
    }
    /// Generated assistant text -> message: reasoning, visible content and tool calls via [`parse_reply`]
    /// (prefix `raw` with "<think>\n" when thinking was on).
    pub fn assistant_from_reply(raw: &str) -> ChatMessage {
        let p = parse_reply(raw);
        ChatMessage { role: "assistant".into(), content: p.visible, reasoning_content: p.reasoning, tool_calls: p.calls }
    }
}

impl From<Message> for ChatMessage {
    fn from(m: Message) -> ChatMessage { ChatMessage { role: m.role, content: m.content, reasoning_content: m.reasoning_content, tool_calls: vec![] } }
}
impl From<&Message> for ChatMessage {
    fn from(m: &Message) -> ChatMessage { ChatMessage::from(m.clone()) }
}

/// A tool (function) definition, rendered as `{"type": "function", "function": {"name", "description", "parameters"}}`.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON schema of the arguments (Anthropic `input_schema`).
    pub parameters: Value,
}

impl ToolDef {
    /// Anthropic Messages API tool: `{"name", "description"?, "input_schema"?}`.
    pub fn from_anthropic(t: &Value) -> Result<ToolDef, String> {
        let name = t["name"].as_str().ok_or_else(|| format!("tool without a name: {}", t))?;
        Ok(ToolDef { name: name.into(), description: t["description"].as_str().unwrap_or("").into(),
                     parameters: if t["input_schema"].is_null() { Value::Object(Default::default()) } else { t["input_schema"].clone() } })
    }
    /// OpenAI-style tool: `{"type": "function", "function": {"name", "description"?, "parameters"?}}` or the inner object.
    pub fn from_openai(t: &Value) -> Result<ToolDef, String> {
        let f = if t["function"].is_object() { &t["function"] } else { t };
        let name = f["name"].as_str().ok_or_else(|| format!("tool without a name: {}", t))?;
        Ok(ToolDef { name: name.into(), description: f["description"].as_str().unwrap_or("").into(),
                     parameters: if f["parameters"].is_null() { Value::Object(Default::default()) } else { f["parameters"].clone() } })
    }
    /// The object the template's `tool | tojson` sees.
    pub fn to_template_value(&self) -> Value {
        let mut f = serde_json::Map::new();
        f.insert("name".into(), Value::String(self.name.clone()));
        f.insert("description".into(), Value::String(self.description.clone()));
        f.insert("parameters".into(), self.parameters.clone());
        let mut o = serde_json::Map::new();
        o.insert("type".into(), Value::String("function".into()));
        o.insert("function".into(), Value::Object(f));
        Value::Object(o)
    }
}

const TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Legacy entry point (no tools): same as [`render_chat`] with no tool definitions.
pub fn render(messages: &[Message], opt: &ChatOptions) -> Result<String, String> {
    let m: Vec<ChatMessage> = messages.iter().map(ChatMessage::from).collect();
    render_chat(&m, &[], opt)
}

/// `apply_chat_template(messages, tools=tools or None, add_generation_prompt, enable_thinking, reasoning_effort)`.
pub fn render_chat(messages: &[ChatMessage], tools: &[ToolDef], opt: &ChatOptions) -> Result<String, String> {
    if messages.is_empty() { return Err("No messages provided.".into()); }
    let mut num_sys = 0;
    let mut merged = String::new();
    for m in messages {
        if !is_system(&m.role) { break; }
        let c = py_trim(&m.content);
        if !c.is_empty() {
            if !merged.is_empty() { merged.push('\n'); }
            merged.push_str(c);
        }
        num_sys += 1;
    }
    let mut instr = "";
    if opt.enable_thinking {
        let eff = opt.reasoning_effort.as_deref().unwrap_or("xhigh");
        let eff = if eff == "high" { "xhigh" } else { eff };
        instr = match eff {
            "xhigh" => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            "low" => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            "medium" => "",
            _ => return Err(format!("Unexpected reasoning effort {}. Supported types are xhigh (default), medium, and low.",
                                    opt.reasoning_effort.as_deref().unwrap_or(""))),
        };
    }
    let mut out = String::with_capacity(messages.iter().map(|m| m.content.len() + 64).sum::<usize>() + 512 + tools.len() * 1024);
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !instr.is_empty() { out.push_str(instr); out.push_str("\n\n"); }
        out.push_str(TOOLS_HEAD);
        for t in tools { out.push('\n'); out.push_str(&py_json(&t.to_template_value())); }
        out.push_str(TOOLS_TAIL);
        if !merged.is_empty() { out.push_str("\n\n"); out.push_str(&merged); }
        out.push_str("<|im_end|>\n");
    } else if !merged.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !instr.is_empty() { out.push_str(instr); out.push_str("\n\n"); }
        out.push_str(&merged);
        out.push_str("<|im_end|>\n");
    } else if !instr.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(instr);
        out.push_str("<|im_end|>\n");
    }
    for i in num_sys..messages.len() {
        let m = &messages[i];
        let content = py_trim(&m.content);
        match m.role.as_str() {
            r if is_system(r) => return Err("System message must be at the beginning.".into()),
            "user" => { out.push_str("<|im_start|>user\n"); out.push_str(content); out.push_str("<|im_end|>\n"); }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(py_trim(m.reasoning_content.as_deref().unwrap_or("")));
                out.push_str("\n</think>\n\n");
                out.push_str(content);
                for (k, tc) in m.tool_calls.iter().enumerate() {
                    if k > 0 { out.push('\n'); } else if !content.is_empty() { out.push_str("\n\n"); }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&tc.name);
                    out.push_str(">\n");
                    for (an, av) in &tc.args {
                        out.push_str("<parameter=");
                        out.push_str(an);
                        out.push_str(">\n");
                        match av { Value::String(s) => out.push_str(s), v => out.push_str(&py_json(v)) }
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i > 0 && messages[i - 1].role != "tool" { out.push_str("<|im_start|>user"); }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if i + 1 == messages.len() || messages[i + 1].role != "tool" { out.push_str("<|im_end|>\n"); }
            }
            r => return Err(format!("Unexpected message role {:?} (supported: system, developer, user, assistant, tool).", r)),
        }
    }
    if opt.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        out.push_str(if opt.enable_thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" });
    }
    Ok(out)
}
