//! Mistral `[INST]` chat templates as fixed strings (no Jinja engine). Which one a model folder uses is read from its
//! template text (`chat_template.jinja`, else `tokenizer_config.json` `chat_template`: string, or the list form's
//! "default" entry, as `llama::template::template_text`) by literal markers. Anything else is refused by name.
//!
//! Ported templates (system / user / assistant text only; tools, tool calls and tool results are refused). `<s>` and
//! `</s>` are the `bos_token` / `eos_token` of every fetched Mistral tokenizer_config.json (checked by `template_kind`).
//! None of them writes anything for `add_generation_prompt` (the prompt ends with `[/INST]`).
//!
//! - `V3` (Mistral-7B-Instruct-v0.3, "default" template): `<s>`, then per message, roles alternating user / assistant
//!   starting with user: user ` [INST] {content} [/INST]`, assistant ` {content} </s>`. A system message is not
//!   allowed (the template raises "Conversation roles must alternate ...").
//!   Marker: `' [INST] ' + message['content'] + ' [/INST]'`.
//! - `Nemo` (Mistral-Nemo-Instruct-2407): optional first system message, the rest alternating from user. `<s>`, then
//!   user `[INST] {content}[/INST]` (the LAST user message, when there is a system message:
//!   `[INST] {system}\n\n{content}[/INST]`), assistant ` {content}</s>`.
//!   Marker: `'[INST] ' + message['content'] + '[/INST]'`.
//! - `Tekken` (Ministral-8B-Instruct-2410): as `Nemo` without the space after `[INST]` (`[INST]{content}[/INST]`), and
//!   assistant ` {content|trim}</s>`. Marker: `"[INST]" + message["content"] + "[/INST]"`.
//!
//! Content is not trimmed except where the template trims it (Tekken assistant). Difference from the templates, where
//! they silently drop text: a system message with no user message as the LAST message (`Nemo` / `Tekken` put the system
//! text into the last message only when it is a user message) is an error here.
//!
//! Prompt encoding: the text starts with the literal `<s>`; encode it with `Tokenizer::encode` (special tokens split
//! out), not `encode_with_special` (the Mistral-7B v0.3 post-processor would add a second BOS).
use std::path::Path;

use serde_json::Value;

const BOS: &str = "<s>";
const EOS: &str = "</s>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateKind {
    V3,
    Nemo,
    Tekken,
}

/// Python `str.isspace` (Rust whitespace plus U+001C..U+001F); Jinja `trim` = `str.strip()`.
fn py_trim(s: &str) -> &str { s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)) }

impl TemplateKind {
    pub fn name(&self) -> &'static str {
        match self {
            TemplateKind::V3 => "mistral v0.3 [INST]",
            TemplateKind::Nemo => "mistral nemo [INST]",
            TemplateKind::Tekken => "ministral tekken [INST]",
        }
    }

    /// Render `messages` ((role, content) pairs) as the template does. `add_generation_prompt` changes nothing.
    pub fn render(&self, messages: &[(String, String)], _add_generation_prompt: bool) -> Result<String, String> {
        if messages.is_empty() { return Err(format!("chat template {}: no messages", self.name())); }
        let (system, rest) = match (self, messages.split_first()) {
            (TemplateKind::Nemo | TemplateKind::Tekken, Some(((r, c), rest))) if r == "system" => (Some(c.as_str()), rest),
            _ => (None, messages),
        };
        if system.is_some() && rest.last().map(|(r, _)| r.as_str()) != Some("user") {
            return Err(format!("chat template {}: a system message needs a user message as the last message (the \
                                template writes it into the last user turn only)", self.name()));
        }
        let mut s = String::from(BOS);
        for (i, (r, c)) in rest.iter().enumerate() {
            if !matches!(r.as_str(), "user" | "assistant" | "system") {
                return Err(format!("chat template {}: role {:?} not supported (system, user, assistant only)", self.name(), r));
            }
            if (r == "user") != (i % 2 == 0) {
                return Err(match self {
                    TemplateKind::V3 => "chat template mistral v0.3: Conversation roles must alternate user/assistant/user/assistant/... (no system message)".into(),
                    _ => format!("chat template {}: After the optional system message, conversation roles must alternate user/assistant/user/assistant/...", self.name()),
                });
            }
            if r == "system" {
                return Err(format!("chat template {}: Only user and assistant roles are supported (system: first message \
                                    only, and not in the v0.3 template)", self.name()));
            }
            let last = i + 1 == rest.len();
            match (self, r.as_str()) {
                (TemplateKind::V3, "user") => s += &format!(" [INST] {} [/INST]", c),
                (TemplateKind::V3, _) => s += &format!(" {} {}", c, EOS),
                (TemplateKind::Nemo, "user") => match system {
                    Some(sys) if last => s += &format!("[INST] {}\n\n{}[/INST]", sys, c),
                    _ => s += &format!("[INST] {}[/INST]", c),
                },
                (TemplateKind::Nemo, _) => s += &format!(" {}{}", c, EOS),
                (TemplateKind::Tekken, "user") => match system {
                    Some(sys) if last => s += &format!("[INST]{}\n\n{}[/INST]", sys, c),
                    _ => s += &format!("[INST]{}[/INST]", c),
                },
                (TemplateKind::Tekken, _) => s += &format!(" {}{}", py_trim(c), EOS),
            }
        }
        Ok(s)
    }
}

/// Template kind from the template text; `tok_cfg` = tokenizer_config.json (bos / eos token text must be `<s>` / `</s>`).
pub fn kind_from_text(tpl: Option<&str>, tok_cfg: Option<&Value>) -> Result<TemplateKind, String> {
    let t = tpl.ok_or("chat template: none in this model folder (base model?): use plain-text prompts")?;
    if let Some(tc) = tok_cfg {
        let tok = |k: &str| match &tc[k] {
            Value::String(s) => Some(s.clone()),
            v => v["content"].as_str().map(|s| s.to_string()),
        };
        for (k, want) in [("bos_token", BOS), ("eos_token", EOS)] {
            if let Some(got) = tok(k) {
                if got != want { return Err(format!("chat template mistral: tokenizer_config {} is {:?}, want {:?}", k, got, want)); }
            }
        }
    }
    if t.contains("' [INST] ' + message['content'] + ' [/INST]'") { return Ok(TemplateKind::V3); }
    if t.contains("'[INST] ' + message['content'] + '[/INST]'") && t.contains("'[INST] ' + system_message + '\\n\\n'") {
        return Ok(TemplateKind::Nemo);
    }
    if t.contains("\"[INST]\" + message[\"content\"] + \"[/INST]\"") && t.contains("\"[INST]\" + system_message + \"\\n\\n\"") {
        return Ok(TemplateKind::Tekken);
    }
    if t.contains("[SYSTEM_PROMPT]") {
        return Err("chat template: Mistral [SYSTEM_PROMPT] template (Mistral Small 3 / Ministral 3): not ported".into());
    }
    if t.contains("[INST]") {
        return Err("chat template: Mistral [INST] template of another release (not v0.3, Nemo or Ministral-8B): not ported".into());
    }
    Err("chat template: not a Mistral [INST] template: not ported for the mistral family".into())
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// The chat template kind `model_dir` declares.
pub fn template_kind(model_dir: &Path) -> Result<TemplateKind, String> {
    let tpl = crate::llama::template::template_text(model_dir)?;
    kind_from_text(tpl.as_deref(), read_json(&model_dir.join("tokenizer_config.json")).as_ref())
}

/// Chat prompt in the family default template (`V3`, Mistral-7B-Instruct-v0.3).
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    TemplateKind::V3.render(messages, add_generation_prompt)
}

/// Chat prompt in the template `model_dir` declares.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    template_kind(model_dir)?.render(messages, add_generation_prompt)
}
