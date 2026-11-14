//! Cohere2 family (`model_type` "cohere2": Command R7B 12-2024): config.json -> `dense_engine::config::DenseConfig`,
//! the R7B chat template as fixed strings, and the list of what blocks a config. No inference code: the math runs in
//! dense-engine (`Family::Cohere2`; LayerNorm, parallel attention + FFN branches, 3 sliding RoPE layers to 1 global
//! no-RoPE layer, interleaved RoPE, tied head with a logit scale).
//!
//! - `config`: field mapping (table in its doc), refusals.
//! - `chat`: `chat_prompt` (template without documents / tools), `check_template` / `template_check_dir`.
//!
//! Checked against real files (mlx-community/c4ai-command-r7b-12-2024-4bit: config.json, tokenizer_config.json,
//! weight index): chat strings equal to HF `apply_chat_template` renders (tests). Not run on the engine yet.
//!
//! Prompt encoding: see `chat` (mlx_lm prepends `<BOS_TOKEN>`: `Tokenizer::encode_with_special`).
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `chat_prompt_dir`, `blockers`.
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::chat_prompt;

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["cohere2"];

/// The engine config of a Cohere2 model: `cfg` = parsed config.json, `model_dir` for `generation_config.json` (extra
/// eos ids, optional). Errors: refusals, missing required keys, a missing eos, every blocker.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gp = model_dir.join("generation_config.json");
    let gen: Option<Value> = match std::fs::read_to_string(&gp) {
        Ok(t) => Some(serde_json::from_str(&t).map_err(|e| format!("{}: {}", gp.display(), e))?),
        Err(_) => None,
    };
    let (d, extra) = config::map_config(cfg, gen.as_ref())?;
    if d.eos.is_empty() { return Err("cohere2: no eos_token_id in config.json or generation_config.json".into()); }
    let mut errs = d.unsupported();
    errs.extend(extra);
    if !errs.is_empty() { return Err(format!("cohere2 config not supported: {}", errs.join("; "))); }
    Ok(d)
}

/// Chat prompt for a model folder: checks that the folder declares the R7B template, then renders it.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    chat::template_check_dir(model_dir)?;
    chat_prompt(messages, add_generation_prompt)
}

/// Everything that stops this config from running, by name (config.json only). Empty = runs.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
