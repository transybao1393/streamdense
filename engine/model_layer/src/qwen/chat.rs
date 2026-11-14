//! Qwen family chat prompt for `--chat` (qwen2, qwen3 dense on dense-engine; qwen3.8-hybrid on the main engine). No new
//! template text: the kind is read from the model folder's own template (`chat_template.jinja` / tokenizer_config.json
//! `chat_template`, via `llama::template::template_text`) by text markers, and rendered by renderers that already exist:
//!
//! - `Qwen38`: Qwen3.5 / Qwen3.8 template (text contains "Reasoning effort is set to"). Rendered by
//!   `streamdense_tokenizer::render`, the renderer the gateway uses. `enable_thinking` true, `reasoning_effort` unset:
//!   the template's defaults (`enable_thinking is undefined or enable_thinking is true` -> xhigh instruction in a system
//!   turn; generation prompt `<|im_start|>assistant\n<think>\n`).
//! - `ChatMl`: Qwen2 / Qwen2.5 (and Qwen3 templates whose generation prompt is plain): per message
//!   `<|im_start|>{role}\n{content}<|im_end|>\n` (not trimmed) + `<|im_start|>assistant\n`; the template's default
//!   system text (Qwen2.5: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") is read from the
//!   template and written when the first message is not a system message. Rendered by `llama::TemplateKind::ChatMl`.
//! - `Qwen3`: Qwen3 hybrid-thinking dense template: generation prompt `<|im_start|>assistant\n`, plus an empty think
//!   block only when `enable_thinking is defined and enable_thinking is false`. HF default leaves enable_thinking
//!   undefined, so thinking is on and nothing follows the assistant header: the same text as `ChatMl` without a default
//!   system message.
//!
//! - `Qwen35`: small Qwen3.5 template (generation prompt `<|im_start|>assistant\n`, then `<think>\n` only when
//!   `enable_thinking is defined and enable_thinking is true`, else the empty block `<think>\n\n</think>\n\n`). HF
//!   default leaves enable_thinking undefined, so the empty think block is written. Message content goes through
//!   `render_content(..)|trim` in this template, so system / user text is trimmed (Jinja `trim` = Python `str.strip`);
//!   a system message is written only as the first message (`<|im_start|>system\n{content}<|im_end|>\n`, no default
//!   system text); a message list without a user message is refused, as the template raises. Rendered here, the
//!   template's text literals only.
//!
//! Assistant history is refused for templates that rewrite assistant turns (`</think>` in the template text: Qwen3
//! strips reasoning from earlier turns), and tool messages are refused (the ChatML renderer takes system / user /
//! assistant only); `--chat` sends one user message. A template whose generation prompt is anything else (e.g. one
//! that always opens `<think>\n`) is refused by name.
use std::path::Path;

use streamdense_tokenizer::{render, ChatOptions, Message};

use crate::llama::TemplateKind as LlamaKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateKind {
    Qwen38,
    ChatMl { default_system: Option<String>, assistant_history: bool },
    Qwen3,
    Qwen35,
}

impl TemplateKind {
    pub fn name(&self) -> &'static str {
        match self {
            TemplateKind::Qwen38 => "qwen3.8 (thinking on, effort xhigh: template default)",
            TemplateKind::ChatMl { .. } => "qwen chatml",
            TemplateKind::Qwen3 => "qwen3 (thinking on: template default)",
            TemplateKind::Qwen35 => "qwen3.5 (thinking off, empty think block: template default)",
        }
    }

    /// Render `messages` ((role, content), in order) as `apply_chat_template(messages, add_generation_prompt)` does
    /// with the template's defaults.
    pub fn render(&self, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
        match self {
            TemplateKind::Qwen38 => {
                let m: Vec<Message> = messages.iter().map(|(r, c)| Message::new(r, c)).collect();
                render(&m, &ChatOptions { add_generation_prompt, enable_thinking: true, reasoning_effort: None })
            }
            TemplateKind::ChatMl { default_system, assistant_history } => {
                if !assistant_history && messages.iter().any(|(r, _)| r == "assistant") {
                    return Err("chat template qwen: assistant history is rewritten by this template (think stripping): not ported".into());
                }
                LlamaKind::ChatMl { default_system: default_system.clone() }.render(messages, add_generation_prompt)
            }
            TemplateKind::Qwen3 => {
                if messages.iter().any(|(r, _)| r == "assistant") {
                    return Err("chat template qwen3: assistant history is rewritten by this template (think stripping): not ported".into());
                }
                LlamaKind::ChatMl { default_system: None }.render(messages, add_generation_prompt)
            }
            TemplateKind::Qwen35 => render_qwen35(messages, add_generation_prompt),
        }
    }
}

/// Jinja `trim` (Python `str.strip()`: `str.isspace` characters, which add U+001C..U+001F to Unicode White_Space).
fn py_strip(s: &str) -> &str { s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)) }

/// Qwen3.5 template, no tools: system (first only, trimmed), user (trimmed), generation prompt with the empty think
/// block. Assistant turns are rewritten by the template (think split, last-query index): refused.
fn render_qwen35(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    let is_query = |(r, c): &(String, String)| {
        let c = py_strip(c);
        r == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
    };
    if messages.is_empty() { return Err("chat template qwen3.5: no messages".into()); }
    if !messages.iter().any(is_query) { return Err("chat template qwen3.5: no user query in messages".into()); }
    let mut out = String::new();
    for (i, (r, c)) in messages.iter().enumerate() {
        match r.as_str() {
            "system" if i == 0 => out += &format!("<|im_start|>system\n{}<|im_end|>\n", py_strip(c)),
            "system" => return Err("chat template qwen3.5: system message must be at the beginning".into()),
            "user" => out += &format!("<|im_start|>user\n{}<|im_end|>\n", py_strip(c)),
            "assistant" => return Err("chat template qwen3.5: assistant history is rewritten by this template (think split): not ported".into()),
            other => return Err(format!("chat template qwen3.5: role {} not ported", other)),
        }
    }
    if add_generation_prompt { out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"; }
    Ok(out)
}

/// Literal default system text: an `<|im_start|>system` + newline (real or `\n` escape) followed by plain text up to
/// `<|im_end|>` (no quote / Jinja syntax in between, so `'<|im_start|>system\n' + messages[0]...` does not count).
fn default_system(t: &str) -> Option<String> {
    let key = "<|im_start|>system";
    let mut from = 0;
    while let Some(i) = t[from..].find(key) {
        let rest = &t[from + i + key.len()..];
        from += i + key.len();
        let Some(body) = rest.strip_prefix("\\n").or(rest.strip_prefix('\n')) else { continue };
        let Some(end) = body.find("<|im_end|>") else { continue };
        let s = &body[..end];
        if !s.is_empty() && !s.contains(['\'', '"', '{', '}', '+', '\n', '\\']) { return Some(s.to_string()); }
    }
    None
}

/// Template kind from the template text.
pub fn kind_from_text(tpl: Option<&str>) -> Result<TemplateKind, String> {
    let t = tpl.ok_or("chat template: none in this model folder (base model?): use --prompt")?;
    if !(t.contains("<|im_start|>") && t.contains("<|im_end|>")) {
        return Err("chat template: Qwen folder without a ChatML (<|im_start|>) template: not ported".into());
    }
    if t.contains("Reasoning effort is set to") {
        if !t.contains("enable_thinking is undefined or enable_thinking is true") {
            return Err("chat template qwen3.8: thinking default not `enable_thinking is undefined or ... true`: not ported".into());
        }
        return Ok(TemplateKind::Qwen38);
    }
    let gen = t.rfind("add_generation_prompt").map(|i| &t[i..])
        .ok_or("chat template qwen: no add_generation_prompt block")?;
    if !(gen.contains("<|im_start|>assistant\\n'") || gen.contains("<|im_start|>assistant\n'")) {
        return Err("chat template qwen: generation prompt is not a plain `<|im_start|>assistant\\n`: not ported".into());
    }
    if gen.contains("<think>") {
        // Qwen3.5: `<think>\n` only when enable_thinking is explicitly true, else the empty block (HF default)
        if gen.contains("enable_thinking is defined and enable_thinking is true") && gen.contains("'<think>\\n\\n</think>\\n\\n'")
            && t.contains("|trim") && !t.contains("enable_thinking is undefined") {
            return Ok(TemplateKind::Qwen35);
        }
        // Qwen3: the empty think block is written only when enable_thinking is explicitly false
        if gen.contains("enable_thinking is defined and enable_thinking is false") && !t.contains("enable_thinking is undefined") {
            return Ok(TemplateKind::Qwen3);
        }
        return Err("chat template qwen: generation prompt with a think block in another form: not ported".into());
    }
    Ok(TemplateKind::ChatMl { default_system: default_system(t), assistant_history: !t.contains("</think>") })
}

/// The Qwen chat template kind `model_dir` declares.
pub fn template_kind(model_dir: &Path) -> Result<TemplateKind, String> {
    kind_from_text(crate::llama::template::template_text(model_dir)?.as_deref())
}
