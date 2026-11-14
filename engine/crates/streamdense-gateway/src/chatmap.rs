//! The ONE module that touches tool-aware chat text. Everything comes from G-A's `streamdense-tokenizer` API
//! (documented at the top of its lib.rs): `render_chat` (Qwen3.8 template with tools / `tool` role / assistant
//! `tool_calls`), `parse_tool_calls`, `stream_visible`, `coerce_args`. Other gateway modules use only this file,
//! so an API change in the tokenizer crate is fixed here.
use serde_json::{Map, Value};
use streamdense_tokenizer::{render_chat, stream_visible, ChatOptions};

pub use streamdense_tokenizer::{ChatMessage as ChatMsg, ToolCall, ToolDef};

/// Render `msgs` with the chat template. `think` = enable_thinking (template default effort, xhigh).
pub fn render(msgs: &[ChatMsg], tools: &[ToolDef], think: bool, add_generation_prompt: bool) -> Result<String, String> {
    render_chat(msgs, tools, &ChatOptions { add_generation_prompt, enable_thinking: think, reasoning_effort: None })
}

/// The text the parsers expect for a generated reply: with thinking on the generation prompt opened the think block,
/// so the raw reply gets the "<think>\n" prefix (tokenizer crate contract).
pub fn reply_text(raw: &str, think: bool) -> String { if think { format!("<think>\n{}", raw) } else { raw.to_string() } }

/// The tool-call open tokens a `format: plan` request bans: `<tool_call>` when the tokenizer has it as one token.
pub fn plan_ban_ids(tk: &streamdense_tokenizer::Tokenizer) -> Vec<u32> {
    ["<tool_call>"].iter().filter_map(|s| tk.token_id(s)).collect()
}

/// (visible text, trimmed; tool calls) of a finished reply (`text` from `reply_text`).
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) { streamdense_tokenizer::parse_tool_calls(text) }

/// Type string arguments by the tool's JSON schema; without a schema the arguments pass through.
pub fn coerce_args(args: &Map<String, Value>, schema: Option<&Value>) -> Value {
    match schema { Some(s) => Value::Object(streamdense_tokenizer::coerce_args(args, s)), None => Value::Object(args.clone()) }
}

/// Streaming visible-text filter over `stream_visible`: raw text pieces in (as decoded per token), new visible
/// text out. Hides think blocks, stops at a tool call, holds partial tags / trailing whitespace; the concatenated
/// output equals the `parse_tool_calls` visible text of the whole reply.
pub struct Visible { text: String, sent: usize, stopped: bool }

impl Visible {
    pub fn new(think: bool) -> Visible { Visible { text: reply_text("", think), sent: 0, stopped: false } }
    pub fn stopped(&self) -> bool { self.stopped }

    pub fn push(&mut self, piece: &str) -> String {
        self.text.push_str(piece);
        // after the first tool call nothing more is visible; the text still grows for `call_hint` and `finish`
        if self.stopped { return String::new(); }
        let v = stream_visible(&self.text);
        if v.tool_call { self.stopped = true; }
        self.take(&v.text)
    }

    /// The tool call being written now, once the reply turned into tool calls: (its number from 1, the function
    /// name, its `path` / `file_path` argument once complete). A hint for the client's companion, never the result.
    pub fn call_hint(&self) -> Option<(usize, String, Option<String>)> {
        if !self.stopped { return None; }
        const F: &str = "<function=";
        let n = self.text.matches(F).count();
        let at = self.text.rfind(F)? + F.len();
        let rest = &self.text[at..];
        let name = rest[..rest.find('>')?].trim().to_string();
        let path = ["<parameter=path>", "<parameter=file_path>"].iter().find_map(|k| {
            let v = &rest[rest.find(k)? + k.len()..];
            let end = v.find("</parameter>")?;
            Some(v[..end].trim().to_string())
        });
        Some((n, name, path))
    }

    /// End of the reply: the rest of the final visible text.
    pub fn finish(&mut self) -> String {
        let (vis, _) = parse_tool_calls(&self.text);
        self.take(&vis)
    }

    fn take(&mut self, vis: &str) -> String {
        if vis.len() <= self.sent || !vis.is_char_boundary(self.sent) { return String::new(); }
        let out = vis[self.sent..].to_string();
        self.sent = vis.len();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> { v.as_object().unwrap().clone() }

    #[test]
    fn call_hint_names_the_tool_and_its_path() {
        let mut v = Visible::new(false);
        assert!(v.push("Let me fix it.\n\n").starts_with("Let me fix it."));
        assert_eq!(v.call_hint(), None);
        v.push("<tool_call>\n<function=edit>\n<parameter=pa");
        assert_eq!(v.call_hint(), Some((1, "edit".into(), None)));
        v.push("th>\nsrc/main.py\n</parameter>\n<parameter=old_string>\nx");
        assert_eq!(v.call_hint(), Some((1, "edit".into(), Some("src/main.py".into()))));
        v.push("\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n");
        assert_eq!(v.call_hint(), Some((2, "bash".into(), None)));
    }

    #[test]
    fn render_tools_and_tool_role() {
        let tools = vec![ToolDef { name: "Bash".into(), description: "run".into(), parameters: json!({"type": "object"}) }];
        let msgs = vec![
            ChatMsg::new("system", "sys"), ChatMsg::new("user", "hi"),
            ChatMsg::assistant("ok", None, vec![ToolCall::new("Bash", args(json!({"command": "ls", "timeout": 5})))]),
            ChatMsg::tool("a.txt"), ChatMsg::tool("b"),
        ];
        let s = render(&msgs, &tools, false, true).unwrap();
        assert!(s.starts_with("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"Bash\""), "{}", s);
        assert!(s.contains("</IMPORTANT>\n\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"));
        assert!(s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nok\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>\n"));
        assert!(s.ends_with("<|im_start|>user\n<tool_response>\na.txt\n</tool_response>\n<tool_response>\nb\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn parse_and_coerce() {
        let raw = "Let me look.\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/a b.txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>";
        let (vis, calls) = parse_tool_calls(&reply_text(raw, false));
        assert_eq!(vis, "Let me look.");
        assert_eq!(calls.len(), 1);
        assert_eq!(Value::Object(calls[0].args.clone()), json!({"file_path": "/a b.txt", "limit": "20"}));
        let schema = json!({"properties": {"limit": {"type": "integer"}, "file_path": {"type": "string"}}});
        assert_eq!(coerce_args(&calls[0].args, Some(&schema)), json!({"file_path": "/a b.txt", "limit": 20}));
        assert_eq!(coerce_args(&calls[0].args, None), json!({"file_path": "/a b.txt", "limit": "20"}));
        // thinking on: the reasoning before </think> is never visible
        let (vis, calls) = parse_tool_calls(&reply_text("hmm <tool_call> no\n</think>\n\nAnswer", true));
        assert_eq!((vis.as_str(), calls.len()), ("Answer", 0));
    }

    #[test]
    fn visible_stream_holds_tags() {
        let raw = "\n Hello <b> wor\u{e9}ld.\n\n<tool_call>\n<function=X>";
        for step in [1usize, 2, 3, 7] {
            let mut v = Visible::new(false);
            let chars: Vec<char> = raw.chars().collect();
            let mut out = String::new();
            for c in chars.chunks(step) { let p: String = c.iter().collect(); out.push_str(&v.push(&p)); }
            out.push_str(&v.finish());
            assert_eq!(out, "Hello <b> wor\u{e9}ld.", "step {}", step);
            assert!(v.stopped());
        }
        let mut v = Visible::new(true);
        let mut out = v.push("reasoning </thi");
        out += &v.push("nk>\n\nAnswer  ");
        out += &v.finish();
        assert_eq!(out, "Answer");
    }
}
