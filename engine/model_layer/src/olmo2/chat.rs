//! OLMo 2 Instruct chat template (OLMo-2-1124-7B-Instruct, OLMo-2-0425-1B-Instruct: the same text), as a fixed string:
//!
//! ```text
//! {{ bos_token }}
//! per message: system    '<|system|>\n' + content + '\n'
//!              user      '<|user|>\n' + content + '\n'
//!              assistant '<|assistant|>\n' + content + eos_token (+ '\n' unless it is the last message)
//!              (other roles: nothing)
//!              after the LAST message, with add_generation_prompt: '<|assistant|>\n'
//! ```
//!
//! `bos_token` and `eos_token` are both `<|endoftext|>` in the OLMo 2 tokenizer_config.json (checked by
//! `template_kind`). Content is not trimmed. Difference from the template: roles other than system / user / assistant
//! (silently dropped by it) are an error here.
//!
//! Prompt encoding: the text starts with the literal `<|endoftext|>`; encode with `Tokenizer::encode` (special tokens
//! split out). The OLMo 2 tokenizer adds no BOS of its own.
use std::path::Path;

use serde_json::Value;

const TOK: &str = "<|endoftext|>";
/// Statements of the template text that must all be present (newlines inside the quotes are real newlines).
const MARKERS: [&str; 5] = [
    "{{ bos_token }}{% for message in messages %}",
    "{{ '<|system|>\n' + message['content'] + '\n' }}",
    "{{ '<|user|>\n' + message['content'] + '\n' }}",
    "{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}",
    "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}",
];

/// Render `messages` ((role, content) pairs) as the OLMo 2 template does.
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    if messages.is_empty() { return Err("chat template olmo2: no messages".into()); }
    let mut s = String::from(TOK);
    for (i, (r, c)) in messages.iter().enumerate() {
        let last = i + 1 == messages.len();
        match r.as_str() {
            "system" => s += &format!("<|system|>\n{}\n", c),
            "user" => s += &format!("<|user|>\n{}\n", c),
            "assistant" => s += &format!("<|assistant|>\n{}{}{}", c, TOK, if last { "" } else { "\n" }),
            o => return Err(format!("chat template olmo2: role {:?} not supported (system, user, assistant only)", o)),
        }
    }
    if add_generation_prompt { s += "<|assistant|>\n"; }
    Ok(s)
}

/// Check that a template text (and tokenizer_config.json, when given) is the ported OLMo 2 template.
pub fn check_template(tpl: Option<&str>, tok_cfg: Option<&Value>) -> Result<(), String> {
    let t = tpl.ok_or("chat template: none in this model folder (base model?): use plain-text prompts")?;
    if let Some(m) = MARKERS.iter().find(|m| !t.contains(**m)) {
        return Err(format!("chat template olmo2: not the OLMo 2 Instruct template (missing {:?}): not ported", m));
    }
    if let Some(tc) = tok_cfg {
        for k in ["bos_token", "eos_token"] {
            let v = match &tc[k] { Value::String(s) => Some(s.as_str()), v => v["content"].as_str() };
            if v != Some(TOK) { return Err(format!("chat template olmo2: tokenizer_config {} is {:?}, want {:?}", k, v, TOK)); }
        }
    }
    Ok(())
}

/// Name of the template `model_dir` declares, or why it is refused.
pub fn template_kind(model_dir: &Path) -> Result<&'static str, String> {
    let tpl = crate::llama::template::template_text(model_dir)?;
    let tc: Option<Value> = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    check_template(tpl.as_deref(), tc.as_ref())?;
    Ok("olmo2 <|user|>")
}
