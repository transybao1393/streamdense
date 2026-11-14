//! Phi-3 / Phi-4-mini chat templates as fixed strings (no Jinja engine), picked from the folder's template text by
//! the literal statements below (all must be present); anything else is refused by name.
//!
//! - `Phi3` (Phi-3-mini-4k-instruct, Phi-3.5-mini-instruct): per message `<|{role}|>\n{content}<|end|>\n` for system,
//!   user, assistant; then `<|assistant|>\n` with `add_generation_prompt`, else the tokenizer's `eos_token` text
//!   (`<|end|>` in both). Phi-3.5 skips a system message with empty content (`message['role'] == 'system' and
//!   message['content']`): `skip_empty_system`.
//! - `Phi4` (Phi-4-mini-instruct): per message `<|{role}|>{content}<|end|>` (no newlines), then `<|assistant|>`, else
//!   `eos_token` (`<|endoftext|>`). A system message with `tools` is not representable here (text messages only).
//!
//! No BOS in either template. Content is not trimmed. Differences from the templates: roles other than system / user /
//! assistant (which `Phi3` silently drops and `Phi4` writes as `<|{role}|>`) are an error here.
//!
//! Prompt encoding: `Tokenizer::encode` (no BOS; HF `apply_chat_template` tokenizes with `add_special_tokens=False`).
use std::path::Path;

use serde_json::Value;

const P3_SYS: &str = "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}";
const P3_USER: &str = "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}";
const P3_ASSISTANT: &str = "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}";
const P3_GEN: &str = "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}";
const P3_SKIP_EMPTY_SYSTEM: &str = "message['role'] == 'system' and message['content']";
const P4_MSG: &str = "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}";
const P4_GEN: &str = "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateKind {
    Phi3 { skip_empty_system: bool, eos: String },
    Phi4 { eos: String },
}

impl TemplateKind {
    pub fn name(&self) -> &'static str {
        match self {
            TemplateKind::Phi3 { .. } => "phi-3 <|user|>\\n",
            TemplateKind::Phi4 { .. } => "phi-4-mini <|user|>",
        }
    }

    /// Render `messages` ((role, content) pairs) as the template does.
    pub fn render(&self, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
        if messages.is_empty() { return Err(format!("chat template {}: no messages", self.name())); }
        let mut s = String::new();
        for (r, c) in messages {
            if !matches!(r.as_str(), "system" | "user" | "assistant") {
                return Err(format!("chat template {}: role {:?} not supported (system, user, assistant only)", self.name(), r));
            }
            match self {
                TemplateKind::Phi3 { skip_empty_system, .. } => {
                    if r == "system" && c.is_empty() && *skip_empty_system { continue; }
                    s += &format!("<|{}|>\n{}<|end|>\n", r, c);
                }
                TemplateKind::Phi4 { .. } => s += &format!("<|{}|>{}<|end|>", r, c),
            }
        }
        match (self, add_generation_prompt) {
            (TemplateKind::Phi3 { .. }, true) => s += "<|assistant|>\n",
            (TemplateKind::Phi4 { .. }, true) => s += "<|assistant|>",
            (TemplateKind::Phi3 { eos, .. } | TemplateKind::Phi4 { eos }, false) => s += eos,
        }
        Ok(s)
    }
}

/// Template kind from the template text; `eos` = tokenizer_config.json `eos_token` text (written by the templates when
/// `add_generation_prompt` is false).
pub fn kind_from_text(tpl: Option<&str>, eos: Option<&str>) -> Result<TemplateKind, String> {
    let t = tpl.ok_or("chat template: none in this model folder (base model?): use plain-text prompts")?;
    let eos = || eos.map(|e| e.to_string()).ok_or("chat template phi: tokenizer_config.json has no eos_token".to_string());
    if t.contains("bos_token") {
        return Err("chat template: Phi template that writes bos_token (older Phi-3 release): not ported".into());
    }
    if t.contains(P4_MSG) && t.contains(P4_GEN) {
        return Ok(TemplateKind::Phi4 { eos: eos()? });
    }
    if [P3_SYS, P3_USER, P3_ASSISTANT, P3_GEN].iter().all(|p| t.contains(p)) {
        return Ok(TemplateKind::Phi3 { skip_empty_system: t.contains(P3_SKIP_EMPTY_SYSTEM), eos: eos()? });
    }
    if t.contains("<|user|>") {
        return Err("chat template: Phi <|user|> template of another release (not Phi-3 / 3.5 mini, Phi-4-mini): not ported".into());
    }
    Err("chat template: not a Phi-3 / Phi-4-mini template: not ported for the phi3 family".into())
}

/// tokenizer_config.json `eos_token` text (string or `{content}`).
fn eos_text(model_dir: &Path) -> Option<String> {
    let tc: Value = serde_json::from_str(&std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()?).ok()?;
    match &tc["eos_token"] {
        Value::String(s) => Some(s.clone()),
        v => v["content"].as_str().map(|s| s.to_string()),
    }
}

/// The chat template kind `model_dir` declares.
pub fn template_kind(model_dir: &Path) -> Result<TemplateKind, String> {
    let tpl = crate::llama::template::template_text(model_dir)?;
    kind_from_text(tpl.as_deref(), eos_text(model_dir).as_deref())
}

/// Chat prompt in the family default template (Phi-3.5-mini-instruct: `Phi3`, empty system skipped, eos `<|end|>`).
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    TemplateKind::Phi3 { skip_empty_system: true, eos: "<|end|>".into() }.render(messages, add_generation_prompt)
}

/// Chat prompt in the template `model_dir` declares.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    template_kind(model_dir)?.render(messages, add_generation_prompt)
}
