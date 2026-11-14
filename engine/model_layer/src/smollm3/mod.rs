//! SmolLM3 family of the model layer (`model_type` "smollm3": SmolLM3-3B). Reads the config into
//! `dense_engine::config::DenseConfig` (`Family::SmolLM3`: Llama layers with RoPE switched off on every 4th layer) and
//! renders the SmolLM3 chat template (reasoning-mode system header) as a fixed string. No inference code.
//!
//! - `config`: config.json -> `DenseConfig` (mapping table in its doc), refusals, `blockers`.
//! - `chat`: `chat_prompt` / `chat_prompt_on` (explicit date), `template_kind` (checks a folder's template).
//!
//! Prompt encoding: no BOS in the template (`bos_token` null in tokenizer_config.json): `Tokenizer::encode`.
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `blockers` (+ `chat_prompt_on`, `template_kind`).
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::{chat_prompt, chat_prompt_on, template_kind};

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["smollm3"];

/// The engine config of a SmolLM3 folder: `cfg` = parsed config.json; `model_dir` for generation_config.json and
/// tokenizer_config.json (eos, as mlx_lm). Errors: non-SmolLM3 / non-dense configs (named), no eos, `validate()`'s message.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gen: Option<Value> = std::fs::read_to_string(model_dir.join("generation_config.json"))
        .ok().and_then(|t| serde_json::from_str(&t).ok());
    config::dense_config_with(cfg, gen.as_ref(), crate::mistral::tokenizer_eos_id(model_dir))
}

/// Everything the engine still refuses for this config, readable. Empty = no config blocker.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
