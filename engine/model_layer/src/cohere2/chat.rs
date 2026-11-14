//! Command R7B chat prompt, ported from the "default" entry of the `chat_template` list in the model's
//! tokenizer_config.json, with HF `apply_chat_template` defaults: no `documents`,
//! no `tools`. Without documents the template takes its short branch (the long safety / tool-use preamble is only
//! written when documents are passed):
//!
//! ```text
//! system first:  <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system, as is}<|END_OF_TURN_TOKEN|>
//! else:          <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>
//! then, roles alternating user / assistant starting with user (else raise "Conversation roles must alternate ..."):
//!   user       <|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content.strip()}<|END_OF_TURN_TOKEN|>
//!   assistant  <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{content.strip()}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
//! add_generation_prompt: <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>
//! ```
//!
//! Differences from the template (cases where it silently drops input): a system message after the first message and
//! roles other than system / user / assistant are errors; an empty message list is an error.
//!
//! BOS: the template writes no `<BOS_TOKEN>`. HF `apply_chat_template(tokenize=True)` therefore feeds no BOS, while
//! mlx_lm `generate` encodes the rendered string with `add_special_tokens=True` (it does not start with the BOS text),
//! so its prompt starts with `<BOS_TOKEN>` (id 5, the tokenizer's TemplateProcessing post-processor). To match mlx_lm,
//! encode with `Tokenizer::encode_with_special`.
use std::path::Path;

use serde_json::Value;

/// Python `str.isspace` (Rust whitespace plus U+001C..U+001F); Python `strip()` trims these.
fn py_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

const TOKENS: [&str; 7] = ["<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|SYSTEM_TOKEN|>", "<|USER_TOKEN|>",
                           "<|CHATBOT_TOKEN|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>"];

/// Render `messages` ((role, content), in order) as the R7B template does without documents / tools.
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    if messages.is_empty() { return Err("cohere2 chat: no messages".into()); }
    let (system, rest) = match messages.split_first() {
        Some(((r, c), rest)) if r == "system" => (c.as_str(), rest),
        _ => ("", messages),
    };
    let mut out = format!("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>", system);
    for (i, (role, content)) in rest.iter().enumerate() {
        match role.as_str() {
            "system" => return Err("cohere2 chat: a system message is taken only as the first message (the template drops later ones)".into()),
            "user" | "assistant" => {}
            o => return Err(format!("cohere2 chat: role {:?} not supported (system, user, assistant)", o)),
        }
        if (role == "user") != (i % 2 == 0) {
            return Err("cohere2 chat: Conversation roles must alternate user/assistant/user/assistant/...".into());
        }
        if role == "user" {
            out += &format!("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>", py_trim(content));
        } else {
            out += &format!("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>", py_trim(content));
        }
    }
    if add_generation_prompt { out += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"; }
    Ok(out)
}

/// Whether a template text is the R7B one this port reproduces (`has_token`: the tokenizer defines that special token).
pub fn check_template(tpl: Option<&str>, has_token: &dyn Fn(&str) -> bool) -> Result<(), String> {
    let t = tpl.ok_or("cohere2 chat template: none in this model folder (base model?): use plain-text prompts")?;
    let markers = ["{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>' }}",
                   "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}",
                   "'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>'  + content.strip() + '<|END_RESPONSE|><|END_OF_TURN_TOKEN|>'",
                   "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>' }}",
                   "Conversation roles must alternate"];
    if let Some(m) = markers.iter().find(|m| !t.contains(**m)) {
        return Err(format!("cohere2 chat template: not the Command R7B template (missing {:?}): not ported", m));
    }
    let missing: Vec<&str> = TOKENS.iter().copied().filter(|k| !has_token(k)).collect();
    if !missing.is_empty() { return Err(format!("cohere2 chat template: tokenizer lacks {}", missing.join(", "))); }
    Ok(())
}

/// The chat template text of a model folder: `chat_template.jinja`, else `tokenizer_config.json` `chat_template`
/// (string, or the list form's "default" entry).
pub fn template_text(model_dir: &Path) -> Result<Option<String>, String> {
    if let Ok(t) = std::fs::read_to_string(model_dir.join("chat_template.jinja")) { return Ok(Some(t)); }
    let p = model_dir.join("tokenizer_config.json");
    let Ok(txt) = std::fs::read_to_string(&p) else { return Ok(None) };
    let v: Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", p.display(), e))?;
    Ok(match &v["chat_template"] {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => a.iter().find(|e| e["name"].as_str() == Some("default"))
            .and_then(|e| e["template"].as_str()).map(|s| s.to_string()),
        _ => None,
    })
}

/// Check that `model_dir` declares the R7B template (special tokens looked up as quoted JSON strings in
/// `tokenizer.json` / `tokenizer_config.json`).
pub fn template_check_dir(model_dir: &Path) -> Result<(), String> {
    let tpl = template_text(model_dir)?;
    let mut tok_text = String::new();
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        if let Ok(t) = std::fs::read_to_string(model_dir.join(f)) { tok_text.push_str(&t); }
    }
    if tok_text.is_empty() { return Err(format!("{}: no tokenizer.json / tokenizer_config.json", model_dir.display())); }
    check_template(tpl.as_deref(), &|k| tok_text.contains(&format!("\"{}\"", k)))
}
