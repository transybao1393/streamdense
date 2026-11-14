//! Gemma chat template, ported from the Jinja template shipped with the instruction-tuned Gemma 3 models (Gemma 1 / 2
//! use the same turn markers):
//!
//! ```text
//! {{ bos_token }}
//! system first: first_user_prefix = system content + "\n\n" (not trimmed), loop over the rest
//! per message: roles must alternate user / assistant starting with user, else
//!              raise "Conversation roles must alternate user/assistant/user/assistant/..."
//!              role = "model" for assistant
//!              '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first) + content | trim + '<end_of_turn>\n'
//! add_generation_prompt: '<start_of_turn>model\n'
//! ```
//!
//! BOS: the template text starts with the literal `<bos>` (as `{{ bos_token }}` renders it). Encode the result with
//! `Tokenizer::encode` (no special tokens: `<bos>` is an added token and is split out as its id), NOT with
//! `encode_with_special`, whose post-processor would add a second BOS.
//!
//! Differences from the Jinja template, both cases where the template's own output is silently wrong:
//! - Gemma 1 / 2 templates raise on a system message; this merges it as Gemma 3 does (one function for all three).
//! - A system message with no following user turn: the template drops the system text; this returns an error.
//!
//! Roles accepted: "system" (first message only), "user", "assistant" (or "model"). Anything else is an error.

/// Jinja `trim` = Python `str.strip()`: Rust whitespace plus the separators U+001C..U+001F that Python counts as space.
fn py_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    let mut out = String::from("<bos>");
    let (prefix, rest) = match messages.first() {
        Some((r, c)) if r == "system" => {
            if messages.len() == 1 {
                return Err("chat: a system message needs a user turn after it (Gemma merges system into the first user turn)".into());
            }
            (format!("{}\n\n", c), &messages[1..])
        }
        _ => (String::new(), messages),
    };
    for (i, (role, content)) in rest.iter().enumerate() {
        let r = match role.as_str() {
            "user" => "user",
            "assistant" | "model" => "model",
            "system" => return Err("chat: Gemma takes a system message only as the first message".into()),
            o => return Err(format!("chat: role {:?} not supported by the Gemma template (system, user, assistant)", o)),
        };
        if (r == "user") != (i % 2 == 0) {
            return Err("chat: Conversation roles must alternate user/assistant/user/assistant/...".into());
        }
        out.push_str("<start_of_turn>");
        out.push_str(r);
        out.push('\n');
        if i == 0 { out.push_str(&prefix); }
        out.push_str(py_trim(content));
        out.push_str("<end_of_turn>\n");
    }
    if add_generation_prompt { out.push_str("<start_of_turn>model\n"); }
    Ok(out)
}
