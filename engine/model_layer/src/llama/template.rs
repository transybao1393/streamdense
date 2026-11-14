//! Llama-family chat templates as fixed strings (no Jinja engine). Which one a model folder uses is read from its
//! `chat_template.jinja` (newer transformers) or `tokenizer_config.json` `chat_template` (string, or the list form's
//! "default" entry), by text markers; the special tokens a template writes must exist in the tokenizer files.
//! Anything else is refused by name.
//!
//! Ported templates (messages: system / user / assistant text only; tools, ipython and tool roles are refused):
//! - `Llama3`: Meta Llama 3 Instruct. `<|begin_of_text|>` + per message
//!   `<|start_header_id|>{role}<|end_header_id|>\n\n{content|trim}<|eot_id|>`, + `<|start_header_id|>assistant<|end_header_id|>\n\n`.
//! - `Llama3Dated`: Llama 3.1 / 3.2 Instruct (template text contains "Cutting Knowledge Date"). Always opens with a
//!   system block `<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: {knowledge}\nToday Date: {date}\n\n{system|trim}<|eot_id|>`
//!   (system = the first message when its role is system, else empty), then the other messages as `Llama3`.
//!   `knowledge` and `date` are read from the template text (`set date_string = "..."`, the template's fallback when
//!   no `strftime_now` is defined). transformers defines `strftime_now`, so for 3.2 templates HF writes TODAY's date
//!   there instead; set `date` on the kind to match a specific HF rendering.
//! - `ChatMl`: SmolLM / SmolLM2 Instruct. Per message `<|im_start|>{role}\n{content}<|im_end|>\n` (content NOT
//!   trimmed), + `<|im_start|>assistant\n`. When the template inserts a default system message for a conversation
//!   that does not start with one (SmolLM2), its text is read from the template and inserted the same way.
//!
//! Refused with a named error: Llama 2 / Mistral `[INST]`, Zephyr `<|user|>` (TinyLlama-Chat), no template (base
//! model), ChatML templates with tool handling, any other template.
use std::path::Path;

use serde_json::Value;

const BOS: &str = "<|begin_of_text|>";
const LLAMA3_TOKENS: [&str; 4] = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"];
const CHATML_TOKENS: [&str; 2] = ["<|im_start|>", "<|im_end|>"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateKind {
    Llama3,
    Llama3Dated { knowledge: String, date: String },
    ChatMl { default_system: Option<String> },
}

impl TemplateKind {
    pub fn name(&self) -> &'static str {
        match self {
            TemplateKind::Llama3 => "llama3",
            TemplateKind::Llama3Dated { .. } => "llama3-dated (3.1 / 3.2)",
            TemplateKind::ChatMl { .. } => "chatml (SmolLM / SmolLM2)",
        }
    }

    /// Render `messages` ((role, content) pairs) as the template does.
    pub fn render(&self, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
        if messages.is_empty() { return Err("chat template: no messages".into()); }
        for (r, _) in messages {
            if !matches!(r.as_str(), "system" | "user" | "assistant") {
                return Err(format!("chat template {}: role {:?} not supported (system, user, assistant only)", self.name(), r));
            }
        }
        let header = |s: &mut String, role: &str, content: &str| {
            s.push_str("<|start_header_id|>");
            s.push_str(role);
            s.push_str("<|end_header_id|>\n\n");
            s.push_str(content.trim());
            s.push_str("<|eot_id|>");
        };
        let mut s = String::new();
        match self {
            TemplateKind::Llama3 => {
                s.push_str(BOS);
                for (r, c) in messages { header(&mut s, r, c); }
                if add_generation_prompt { s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n"); }
            }
            TemplateKind::Llama3Dated { knowledge, date } => {
                let (system, rest) = match messages.split_first() {
                    Some(((r, c), rest)) if r == "system" => (c.trim(), rest),
                    _ => ("", messages),
                };
                s.push_str(BOS);
                s.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                s.push_str(&format!("Cutting Knowledge Date: {}\nToday Date: {}\n\n", knowledge, date));
                s.push_str(system);
                s.push_str("<|eot_id|>");
                for (r, c) in rest { header(&mut s, r, c); }
                if add_generation_prompt { s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n"); }
            }
            TemplateKind::ChatMl { default_system } => {
                if let Some(d) = default_system {
                    if messages[0].0 != "system" { s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", d)); }
                }
                for (r, c) in messages { s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", r, c)); }
                if add_generation_prompt { s.push_str("<|im_start|>assistant\n"); }
            }
        }
        Ok(s)
    }
}

/// Text from `start` up to the first real newline, `\n` escape (backslash n) or quote.
fn until_line_end(s: &str) -> &str {
    let end = [s.find('\n'), s.find("\\n"), s.find('"'), s.find('\'')].into_iter().flatten().min().unwrap_or(s.len());
    &s[..end]
}

/// Template kind from the template text; `has_token(t)` says whether the tokenizer defines special token `t`.
pub fn kind_from_text(tpl: Option<&str>, has_token: &dyn Fn(&str) -> bool) -> Result<TemplateKind, String> {
    let t = tpl.ok_or("chat template: none in this model folder (base model?): use plain-text prompts")?;
    let need = |toks: &[&str], name: &str| -> Result<(), String> {
        let missing: Vec<&str> = toks.iter().copied().filter(|k| !has_token(k)).collect();
        if missing.is_empty() { Ok(()) } else { Err(format!("chat template {}: tokenizer lacks {}", name, missing.join(", "))) }
    };
    if t.contains("<|start_header_id|>") && t.contains("<|eot_id|>") {
        need(&LLAMA3_TOKENS, "llama3")?;
        let Some(i) = t.find("Cutting Knowledge Date: ") else { return Ok(TemplateKind::Llama3) };
        let knowledge = until_line_end(&t[i + "Cutting Knowledge Date: ".len()..]).to_string();
        let key = "set date_string = \"";
        let date = t.find(key).map(|j| t[j + key.len()..].split('"').next().unwrap_or("").to_string())
            .filter(|d| !d.is_empty())
            .ok_or("chat template llama3-dated: no `set date_string = \"...\"` fallback date in the template")?;
        if knowledge.is_empty() { return Err("chat template llama3-dated: empty Cutting Knowledge Date".into()); }
        return Ok(TemplateKind::Llama3Dated { knowledge, date });
    }
    if t.contains("<|im_start|>") && t.contains("<|im_end|>") {
        if t.contains("tool") {
            return Err("chat template: ChatML template with tool handling (Qwen-style): not ported for the llama family".into());
        }
        need(&CHATML_TOKENS, "chatml")?;
        let conditional = t.contains("!= 'system'") || t.contains("!= \"system\"");
        let default_system = match t.find("<|im_start|>system") {
            Some(i) if conditional => {
                let rest = &t[i + "<|im_start|>system".len()..];
                let body = rest.strip_prefix('\n').or(rest.strip_prefix("\\n"))
                    .ok_or("chat template chatml: default system message not followed by a newline")?;
                let end = body.find("<|im_end|>").ok_or("chat template chatml: default system message without <|im_end|>")?;
                Some(body[..end].to_string())
            }
            _ => None,
        };
        return Ok(TemplateKind::ChatMl { default_system });
    }
    if t.contains("[INST]") {
        return Err("chat template: Llama 2 / Mistral [INST] template: not ported".into());
    }
    if t.contains("<|user|>") {
        return Err("chat template: Zephyr <|user|> template (e.g. TinyLlama-Chat): not ported".into());
    }
    Err("chat template: unknown template (not Llama 3 header, ChatML, [INST] or Zephyr): not ported".into())
}

/// The chat template text of a model folder: `chat_template.jinja`, else `tokenizer_config.json` `chat_template`.
pub fn template_text(model_dir: &Path) -> Result<Option<String>, String> {
    if let Ok(t) = std::fs::read_to_string(model_dir.join("chat_template.jinja")) { return Ok(Some(t)); }
    let p = model_dir.join("tokenizer_config.json");
    let txt = match std::fs::read_to_string(&p) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
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
