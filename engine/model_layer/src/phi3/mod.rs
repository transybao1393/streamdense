//! Phi-3 family of the model layer (`model_type` "phi3": Phi-3-mini-4k / 128k, Phi-3.5-mini, Phi-4-mini). Reads the
//! config into `dense_engine::config::DenseConfig` (`Family::Phi3`: fused qkv / gate_up tensors, partial rotary,
//! LongRoPE as mlx_lm) and renders the Phi chat templates as fixed strings. No inference code.
//!
//! - `config`: config.json -> `DenseConfig` (mapping table in its doc), refusals, `blockers`.
//! - `chat`: template kinds `Phi3` (Phi-3 / 3.5) and `Phi4` (Phi-4-mini), picked from the folder's template text.
//!
//! The real configs (Phi-3 / 3.5-mini head_dim 96, Phi-4-mini partial rotary; fused tensors, LongRoPE) have no engine
//! blockers; Phi-3.5-mini and Phi-4-mini were checked against mlx_lm (2026-11-15).
//!
//! Prompt encoding: no BOS in the templates; encode with `Tokenizer::encode`.
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `blockers` (+ `chat_prompt_dir`, `template_kind`).
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::{chat_prompt, chat_prompt_dir, template_kind, TemplateKind};

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["phi3"];

/// The engine config of a Phi-3 folder: `cfg` = parsed config.json; `model_dir` for generation_config.json and
/// tokenizer_config.json (eos, as mlx_lm). Errors: non-Phi-3 / non-dense configs (named), no eos, `validate()`'s message.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gen: Option<Value> = std::fs::read_to_string(model_dir.join("generation_config.json"))
        .ok().and_then(|t| serde_json::from_str(&t).ok());
    config::dense_config_with(cfg, gen.as_ref(), crate::mistral::tokenizer_eos_id(model_dir))
}

/// Everything the engine still refuses for this config, readable. Empty = no config blocker.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
