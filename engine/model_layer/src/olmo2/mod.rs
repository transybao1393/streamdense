//! OLMo 2 family of the model layer (`model_type` "olmo2": OLMo-2-1124-7B / 13B, OLMo-2-0425-1B, OLMo-2-0325-32B).
//! Reads the config into `dense_engine::config::DenseConfig` (`Family::Olmo2`: post-norm layers, RMS QK-norm over the
//! whole q / k projection) and renders the OLMo 2 chat template as a fixed string. No inference code.
//!
//! - `config`: config.json -> `DenseConfig` (mapping table in its doc), refusals, `blockers`.
//! - `chat`: `chat_prompt` (the one OLMo 2 Instruct template), `template_kind` (checks a folder's template).
//!
//! Prompt encoding: the text starts with the literal `<|endoftext|>` (the template's bos_token): `Tokenizer::encode`.
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `blockers` (+ `template_kind`).
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::{chat_prompt, template_kind};

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["olmo2"];

/// The engine config of an OLMo 2 folder: `cfg` = parsed config.json; `model_dir` for generation_config.json and
/// tokenizer_config.json (eos, as mlx_lm). Errors: non-OLMo-2 / non-dense configs (named), no eos, `validate()`'s message.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gen: Option<Value> = std::fs::read_to_string(model_dir.join("generation_config.json"))
        .ok().and_then(|t| serde_json::from_str(&t).ok());
    config::dense_config_with(cfg, gen.as_ref(), crate::mistral::tokenizer_eos_id(model_dir))
}

/// Everything the engine still refuses for this config, readable. Empty = no config blocker.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
