//! GLM-4 family (`model_type` "glm4": GLM-4-9B-0414, GLM-4-32B-0414, GLM-Z1-0414): config.json ->
//! `dense_engine::config::DenseConfig`, the chat templates as fixed strings, and the list of what blocks a config. No
//! inference code: the math runs in dense-engine (`Family::Glm4`; four norms with GLM names, fused gate_up, partial
//! interleaved RoPE).
//!
//! - `config`: field mapping (table in its doc), refusals of the non-dense / multimodal relatives.
//! - `template`: `TemplateKind` (GLM-4-0414, GLM-Z1-0414), picked from the folder's template text.
//!
//! Checked against real files (mlx-community/GLM-4-9B-0414-4bit, GLM-Z1-9B-0414-4bit: config.json,
//! tokenizer_config.json, weight index): chat strings equal to HF `apply_chat_template` renders (tests). Not run on the
//! engine yet.
//!
//! Prompt encoding: the rendered text starts with `[gMASK]<sop>` (special tokens): encode with special-token
//! recognition and no post-processor BOS (`Tokenizer::encode`).
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `chat_prompt_dir`, `template_kind`, `blockers`.
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod config;
pub mod template;
#[cfg(test)]
mod tests;

pub use template::{template_kind, TemplateKind};

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["glm4"];

/// The engine config of a GLM-4 model: `cfg` = parsed config.json, `model_dir` for `generation_config.json` (extra eos
/// ids, optional). Errors: refused relatives, missing required keys, a missing eos, every blocker (config-level and
/// `DenseConfig::unsupported()`).
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gp = model_dir.join("generation_config.json");
    let gen: Option<Value> = match std::fs::read_to_string(&gp) {
        Ok(t) => Some(serde_json::from_str(&t).map_err(|e| format!("{}: {}", gp.display(), e))?),
        Err(_) => None,
    };
    let (d, extra) = config::map_config(cfg, gen.as_ref())?;
    if d.eos.is_empty() { return Err("glm4: no eos_token_id in config.json or generation_config.json".into()); }
    let mut errs = d.unsupported();
    errs.extend(extra);
    if !errs.is_empty() { return Err(format!("glm4 config not supported: {}", errs.join("; "))); }
    Ok(d)
}

/// Chat prompt in the GLM-4-0414 template (the family default). For a model folder use `chat_prompt_dir`, which
/// picks GLM-4 or GLM-Z1 from the folder's template.
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    TemplateKind::Glm4.render(messages, add_generation_prompt)
}

/// Chat prompt in the template `model_dir` declares (`template_kind`); unknown templates are refused by name.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    template_kind(model_dir)?.render(messages, add_generation_prompt)
}

/// Everything that stops this config from running, by name (config.json only). Empty = runs.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
