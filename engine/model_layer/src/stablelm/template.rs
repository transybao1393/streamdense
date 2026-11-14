//! StableLM-2 chat templates as fixed strings (no Jinja engine), ported from the templates in the
//! `tokenizer_config.json` of stablelm-2-1_6b-chat / stablelm-2-12b-chat (ChatML) and stablelm-2-zephyr-1_6b (Zephyr).
//! Messages: system / user / assistant text only.
//!
//! - `ChatMl { default_system }` (stablelm-2-chat): when the first message is a system message it is the system text,
//!   else `default_system` (read from the template: "You are a helpful assistant."). Output:
//!   `<|im_start|>system\n{system}<|im_end|>\n` then per remaining message `<|im_start|>{role}\n{content}<|im_end|>\n`
//!   (content NOT trimmed), + `<|im_start|>assistant\n` when `add_generation_prompt`. A later system message is a
//!   normal ChatML turn.
//! - `Zephyr { eos }` (stablelm-2-zephyr): per message `<|{role}|>\n{content}{eos}\n` (eos = the tokenizer's eos_token
//!   text, `<|endoftext|>`), + `<|assistant|>\n`. `<|user|>` etc. are plain text in this tokenizer, not special tokens.
//!
//! Differences from the Jinja templates (cases where the template's output silently loses input): an empty message
//! list, a system message with no other message (ChatML: the template writes nothing for it) and roles other than
//! system / user / assistant (Zephyr drops them) are errors.
//!
//! Stop tokens: the config's eos is `<|endoftext|>` (100257) only; the ChatML turn end `<|im_end|>` (100278) is not in
//! config.json and there is no generation_config.json in the MLX conversions, so (as in mlx_lm) generation stops at
//! `<|endoftext|>`.
use std::path::Path;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateKind {
    ChatMl { default_system: Option<String> },
    Zephyr { eos: String },
}

impl TemplateKind {
    pub fn name(&self) -> &'static str {
        match self {
            TemplateKind::ChatMl { .. } => "chatml (stablelm-2-chat)",
            TemplateKind::Zephyr { .. } => "zephyr (stablelm-2-zephyr)",
        }
    }

    /// Render `messages` ((role, content) pairs) as the template does.
    pub fn render(&self, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
        if messages.is_empty() { return Err("stablelm chat: no messages".into()); }
        for (r, _) in messages {
            if !matches!(r.as_str(), "system" | "user" | "assistant") {
                return Err(format!("stablelm chat template {}: role {:?} not supported (system, user, assistant)", self.name(), r));
            }
        }
        let mut s = String::new();
        match self {
            TemplateKind::ChatMl { default_system } => {
                let (system, rest) = match messages.split_first() {
                    Some(((r, c), rest)) if r == "system" => (Some(c.as_str()), rest),
                    _ => (default_system.as_deref(), messages),
                };
                if rest.is_empty() {
                    return Err("stablelm chat: a system message needs another message after it (the template writes nothing for a system message alone)".into());
                }
                if let Some(sys) = system { s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys)); }
                for (r, c) in rest { s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", r, c)); }
                if add_generation_prompt { s.push_str("<|im_start|>assistant\n"); }
            }
            TemplateKind::Zephyr { eos } => {
                for (r, c) in messages { s.push_str(&format!("<|{}|>\n{}{}\n", r, c, eos)); }
                if add_generation_prompt { s.push_str("<|assistant|>\n"); }
            }
        }
        Ok(s)
    }
}

/// Template kind from the template text. `has_token(t)`: the tokenizer defines special token `t`; `eos`: the
/// tokenizer_config.json eos_token text (Zephyr writes it).
pub fn kind_from_text(tpl: Option<&str>, has_token: &dyn Fn(&str) -> bool, eos: Option<&str>) -> Result<TemplateKind, String> {
    let t = tpl.ok_or("stablelm chat template: none in this model folder (base model?): use plain-text prompts")?;
    if t.contains("<|im_start|>") && t.contains("<|im_end|>") {
        if t.contains("tool") { return Err("stablelm chat template: ChatML template with tool handling: not ported".into()); }
        // the stablelm-2-chat form: system turn written at the first loop iteration, from messages[0] or a literal default
        if !(t.contains("set system_message") && t.contains("loop.index0 == 0")) {
            return Err("stablelm chat template: ChatML variant other than the stablelm-2-chat one: not ported".into());
        }
        for k in ["<|im_start|>", "<|im_end|>"] {
            if !has_token(k) { return Err(format!("stablelm chat template chatml: tokenizer lacks {}", k)); }
        }
        let key = "set system_message = '";
        let default_system = t.find(key).map(|i| t[i + key.len()..].split('\'').next().unwrap_or("").to_string());
        return Ok(TemplateKind::ChatMl { default_system });
    }
    if t.contains("<|user|>") && t.contains("<|assistant|>") && t.contains("eos_token") {
        let eos = eos.ok_or("stablelm chat template zephyr: tokenizer_config.json has no eos_token")?;
        if !has_token(eos) { return Err(format!("stablelm chat template zephyr: tokenizer lacks {}", eos)); }
        return Ok(TemplateKind::Zephyr { eos: eos.to_string() });
    }
    Err("stablelm chat template: unknown template (not the stablelm-2 ChatML or Zephyr one): not ported".into())
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
/// `tokenizer.json` and `tokenizer_config.json`; the eos text is tokenizer_config.json `eos_token`.
pub fn template_kind(model_dir: &Path) -> Result<TemplateKind, String> {
    let tpl = template_text(model_dir)?;
    let mut tok_text = String::new();
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        if let Ok(t) = std::fs::read_to_string(model_dir.join(f)) { tok_text.push_str(&t); }
    }
    if tok_text.is_empty() { return Err(format!("{}: no tokenizer.json / tokenizer_config.json", model_dir.display())); }
    let tc: Option<Value> = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let eos = tc.as_ref().and_then(|v| v["eos_token"].as_str().or(v["eos_token"]["content"].as_str()).map(|s| s.to_string()));
    kind_from_text(tpl.as_deref(), &|k| tok_text.contains(&format!("\"{}\"", k)), eos.as_deref())
}
