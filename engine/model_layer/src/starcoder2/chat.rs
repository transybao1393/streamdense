//! StarCoder2 chat: none. The StarCoder2 3B / 7B / 15B releases are base code models without a chat template
//! (tokenizer_config.json of mlx-community/starcoder2-3b-4bit has no `chat_template`); prompts are plain text (code,
//! or the FIM tokens `<fim_prefix>` / `<fim_suffix>` / `<fim_middle>`). Plain-text encoding: the tokenizer has no
//! post-processor, so no BOS is added (as mlx_lm).
//!
//! Instruction tunes (e.g. starcoder2-15b-instruct-v0.1) ship their own template, which is not ported: refused by
//! name.
use std::path::Path;

use serde_json::Value;

/// Always an error: StarCoder2 base models have no chat template.
pub fn chat_prompt(_messages: &[(String, String)], _add_generation_prompt: bool) -> Result<String, String> {
    Err("starcoder2: base code model without a chat template: use plain-text (or FIM) prompts".into())
}

/// Folder form: names a template the folder declares (an instruction tune) as not ported; else the base-model error.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    let has_jinja = model_dir.join("chat_template.jinja").exists();
    let tc: Option<Value> = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    if has_jinja || tc.is_some_and(|v| !v["chat_template"].is_null()) {
        return Err("starcoder2: this folder's chat template (instruction tune) is not ported".into());
    }
    chat_prompt(messages, add_generation_prompt)
}
