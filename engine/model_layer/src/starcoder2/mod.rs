//! StarCoder2 family (`model_type` "starcoder2": StarCoder2 3B / 7B / 15B): config.json ->
//! `dense_engine::config::DenseConfig` and the list of what blocks a config. No chat template (base code models):
//! `chat_prompt` refuses by name. No inference code: the math runs in dense-engine (`Family::Starcoder2`; LayerNorm
//! with bias, plain GELU (erf) MLP c_fc / c_proj, biases everywhere, tied head).
//!
//! - `config`: field mapping (table in its doc), refusals.
//! - `chat`: `chat_prompt` / `chat_prompt_dir` (refusals).
//!
//! Checked against real files (mlx-community/starcoder2-3b-4bit: config.json, tokenizer_config.json, weight index).
//! Not run on the engine yet. The MLX conversion keeps `embed_tokens` (also the tied head) and every norm / bias in
//! F16 (mlx_lm then computes in float16, the engine in bf16).
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `chat_prompt_dir`, `blockers`.
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::{chat_prompt, chat_prompt_dir};

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["starcoder2"];

/// The engine config of a StarCoder2 model: `cfg` = parsed config.json, `model_dir` for `generation_config.json`
/// (extra eos ids, optional). Errors: refusals, missing required keys, a missing eos, every blocker.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gp = model_dir.join("generation_config.json");
    let gen: Option<Value> = match std::fs::read_to_string(&gp) {
        Ok(t) => Some(serde_json::from_str(&t).map_err(|e| format!("{}: {}", gp.display(), e))?),
        Err(_) => None,
    };
    let (d, extra) = config::map_config(cfg, gen.as_ref())?;
    if d.eos.is_empty() { return Err("starcoder2: no eos_token_id in config.json or generation_config.json".into()); }
    let mut errs = d.unsupported();
    errs.extend(extra);
    if !errs.is_empty() { return Err(format!("starcoder2 config not supported: {}", errs.join("; "))); }
    Ok(d)
}

/// Everything that stops this config from running, by name (config.json only). Empty = runs.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
