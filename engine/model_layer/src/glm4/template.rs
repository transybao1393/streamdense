//! GLM-4 0414 chat templates as fixed strings (no Jinja engine), ported from the templates shipped in
//! `tokenizer_config.json` of GLM-4-9B-0414 / GLM-4-32B-0414 (same output, different whitespace in the source) and
//! GLM-Z1-9B-0414 / GLM-Z1-32B-0414. Defaults of HF `apply_chat_template`: no tools, no message metadata.
//!
//! ```text
//! [gMASK]<sop>
//! every system message, in order, first:            <|system|>\n{content}
//! then every other message, in order:  user        <|user|>\n{c}
//!                                      assistant   <|assistant|>\n{c}
//!                                      observation <|observation|>\n{c}
//! add_generation_prompt:                            <|assistant|>
//! ```
//! - `Glm4` (GLM-4-0414): c = content as is (not trimmed).
//! - `Glm4Z1` (GLM-Z1-0414, template text contains `split('</think>')`): c = content.split('</think>')[-1].strip()
//!   (Python strip) for user / assistant / observation; system content as is.
//!
//! No separator between turns and no end-of-turn token: the model ends a reply by generating `<|user|>` /
//! `<|observation|>` / `<|endoftext|>` (the config's eos list). The text starts with the special tokens `[gMASK]<sop>`:
//! encode with special-token recognition and without a post-processor BOS (`Tokenizer::encode`; GLM-4's
//! post-processor adds none anyway).
//!
//! Differences from the Jinja templates (cases where the template's own output silently loses input): a role other
//! than system / user / assistant / observation (the template drops it) and an empty message list are errors. Tools
//! and message metadata are not ported (the plain (role, content) interface has neither).
use std::path::Path;

use serde_json::Value;

const TOKENS: [&str; 6] = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateKind {
    /// GLM-4-0414: content as is
    Glm4,
    /// GLM-Z1-0414: content after the last `</think>`, trimmed
    Glm4Z1,
}

/// Python `str.isspace` (Rust whitespace plus U+001C..U+001F); Jinja / Python `strip()` trims these.
fn py_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

impl TemplateKind {
    pub fn name(&self) -> &'static str {
        match self {
            TemplateKind::Glm4 => "glm4 (GLM-4-0414)",
            TemplateKind::Glm4Z1 => "glm4-z1 (GLM-Z1-0414)",
        }
    }

    /// Render `messages` ((role, content) pairs) as the template does.
    pub fn render(&self, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
        if messages.is_empty() { return Err("glm4 chat: no messages".into()); }
        for (r, _) in messages {
            if !matches!(r.as_str(), "system" | "user" | "assistant" | "observation") {
                return Err(format!("glm4 chat template {}: role {:?} not supported (system, user, assistant, observation)",
                                   self.name(), r));
            }
        }
        let mut s = String::from("[gMASK]<sop>");
        for (_, c) in messages.iter().filter(|(r, _)| r == "system") {
            s.push_str("<|system|>\n");
            s.push_str(c);
        }
        for (r, c) in messages.iter().filter(|(r, _)| r != "system") {
            let c = match self {
                TemplateKind::Glm4 => c.as_str(),
                TemplateKind::Glm4Z1 => py_trim(c.rsplit("</think>").next().unwrap_or("")),
            };
            s.push_str(&format!("<|{}|>\n{}", r, c));
        }
        if add_generation_prompt { s.push_str("<|assistant|>"); }
        Ok(s)
    }
}

/// Template kind from the template text; `has_token(t)` says whether the tokenizer defines special token `t`.
pub fn kind_from_text(tpl: Option<&str>, has_token: &dyn Fn(&str) -> bool) -> Result<TemplateKind, String> {
    let t = tpl.ok_or("glm4 chat template: none in this model folder (base model?): use plain-text prompts")?;
    let glm = t.contains("[gMASK]<sop>") && t.contains("<|user|>") && t.contains("<|assistant|>");
    if !glm { return Err("glm4 chat template: not the GLM-4 [gMASK]<sop> <|user|> template: not ported".into()); }
    // the 0414 generation prompt is a bare <|assistant|>; a template that opens a think block or adds text is another
    if !t.contains("add_generation_prompt %}<|assistant|>{%") {
        return Err("glm4 chat template: generation prompt is not the bare <|assistant|> of the 0414 templates: not ported".into());
    }
    let missing: Vec<&str> = TOKENS.iter().copied().filter(|k| !has_token(k)).collect();
    if !missing.is_empty() { return Err(format!("glm4 chat template: tokenizer lacks {}", missing.join(", "))); }
    Ok(if t.contains("split('</think>')") || t.contains("split(\"</think>\")") { TemplateKind::Glm4Z1 } else { TemplateKind::Glm4 })
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

/// The chat template kind `model_dir` declares. Special tokens are looked up (as quoted JSON strings) in
/// `tokenizer.json` and `tokenizer_config.json`.
pub fn template_kind(model_dir: &Path) -> Result<TemplateKind, String> {
    let tpl = template_text(model_dir)?;
    let mut tok_text = String::new();
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        if let Ok(t) = std::fs::read_to_string(model_dir.join(f)) { tok_text.push_str(&t); }
    }
    if tok_text.is_empty() { return Err(format!("{}: no tokenizer.json / tokenizer_config.json", model_dir.display())); }
    kind_from_text(tpl.as_deref(), &|k| tok_text.contains(&format!("\"{}\"", k)))
}
