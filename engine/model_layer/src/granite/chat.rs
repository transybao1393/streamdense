//! Chat prompt of the local Granite 4.2 template (models/granite-4.2-3b-MLX-4bit/chat_template.jinja, the same text as
//! tokenizer_config.json `chat_template`), ported for plain (role, content) messages with the template's defaults:
//! no tools, `enable_thinking` true, `low_effort` false, `truncate_history_thinking` true.
//!
//! The template is ChatML (`<|im_start|>role\n content <|im_end|>\n`), not the `<|start_of_role|>` format of older
//! Granite releases. What it does that a plain ChatML renderer would not:
//! - a system turn is ALWAYS written: with no leading system message it is empty (`<|im_start|>system\n<|im_end|>\n`).
//! - assistant turns: content without `<think>` / `</think>` gets `<think></think>` in front; the content is trimmed.
//!   An assistant turn before the last user turn keeps only the text after its last `</think>` when it has both tags.
//! - tool results: consecutive `tool` messages share one user turn, each as `<tool_response>\n..\n</tool_response>\n`.
//! - generation prompt: `<|im_start|>assistant\n<think>\n` (thinking on, the template default).
//!
//! Exactness: checked against jinja2 renders of the template (transformers' settings: trim_blocks, lstrip_blocks) for
//! the cases in the tests. Jinja `trim` is Python `str.strip()`, mirrored by `py_trim`.

/// Python `str.isspace` (Rust `char::is_whitespace` plus U+001C..U+001F).
fn py_space(c: char) -> bool { c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c) }
fn py_trim(s: &str) -> &str { s.trim_matches(py_space) }

/// Render `messages` ((role, content), in order) as the template does. `add_generation_prompt` appends the assistant
/// header with the open think block. Err for an empty message list (the template reads `messages[0]`).
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    let first = messages.first().ok_or("granite chat: no messages")?;
    let (system, rest) = if first.0 == "system" { (first.1.as_str(), &messages[1..]) } else { ("", messages) };
    let last_user = rest.iter().rposition(|(r, _)| r == "user");
    let mut out = format!("<|im_start|>system\n{}<|im_end|>\n", system);
    for (i, (role, content)) in rest.iter().enumerate() {
        match role.as_str() {
            "assistant" => {
                let c = if content.contains("<think>") || content.contains("</think>") {
                    content.clone()
                } else {
                    format!("<think></think>{}", content)
                };
                let history = last_user.is_some_and(|u| i < u);
                let c = if history && c.contains("<think>") && c.contains("</think>") {
                    format!("<think></think>{}", c.rsplit("</think>").next().unwrap_or(""))
                } else {
                    c
                };
                // both template branches write the same text for non-empty trimmed content; an empty one only occurs
                // in the history branch, which writes the same empty turn
                out += &format!("<|im_start|>assistant\n{}<|im_end|>\n", py_trim(&c));
            }
            "user" | "system" => out += &format!("<|im_start|>{}\n{}<|im_end|>\n", role, content),
            "tool" => {
                if i > 0 && rest[i - 1].0 != "tool" { out += "<|im_start|>user\n"; }
                out += &format!("<tool_response>\n{}\n</tool_response>\n", content);
                let next_tool = rest.get(i + 1).is_some_and(|(r, _)| r == "tool");
                if !next_tool { out += "<|im_end|>\n"; }
            }
            other => out += &format!("<|im_start|>{}\n{}<|im_end|>\n", other, content),
        }
    }
    if add_generation_prompt { out += "<|im_start|>assistant\n<think>\n"; }
    Ok(out)
}
