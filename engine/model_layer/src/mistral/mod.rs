//! Mistral family of the model layer (`model_type` "mistral": Mistral-7B-Instruct v0.3, Mistral-Nemo-Instruct-2407,
//! Ministral-8B-Instruct-2410). Reads the config into `dense_engine::config::DenseConfig` (`Family::Mistral`) and
//! renders the `[INST]` chat templates as fixed strings. No inference code: the model runs on `engine/dense-engine`
//! with the Llama layer math (mlx_lm runs Mistral through `models/llama.py`).
//!
//! - `config`: config.json -> `DenseConfig` (mapping table in its doc), refusals, `blockers`.
//! - `chat`: template kinds `V3`, `Nemo`, `Tekken`, picked from the folder's template text.
//!
//! Not this family: Ministral 3 / Mistral Small 3.x (`mistral3` root with a `ministral3` text_config, vision tower,
//! YaRN RoPE), Mixtral (MoE): refused by name.
//!
//! Prompt encoding: the rendered text starts with the literal `<s>`: encode with `Tokenizer::encode` (special tokens
//! split out), not `encode_with_special`.
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
pub const MODEL_TYPES: &[&str] = &["mistral"];

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// The id of tokenizer_config.json `eos_token` (string or `{content}`), looked up in its `added_tokens_decoder` (a
/// small file; tokenizer.json is not read). mlx_lm adds the tokenizer's eos to the config's eos ids.
pub(crate) fn tokenizer_eos_id(dir: &Path) -> Option<u32> {
    let tc = read_json(&dir.join("tokenizer_config.json"))?;
    let eos = match &tc["eos_token"] {
        Value::String(s) => s.clone(),
        v => v["content"].as_str()?.to_string(),
    };
    tc["added_tokens_decoder"].as_object()?.iter()
        .find(|(_, v)| v["content"].as_str() == Some(eos.as_str()))
        .and_then(|(k, _)| k.parse().ok())
}

/// The engine config of a Mistral folder: `cfg` = parsed config.json; `model_dir` for generation_config.json and
/// tokenizer_config.json (eos). Errors: non-Mistral / non-dense configs (named), no eos, `validate()`'s message.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gen = read_json(&model_dir.join("generation_config.json"));
    config::dense_config_with(cfg, gen.as_ref(), tokenizer_eos_id(model_dir))
}

/// Everything the engine still refuses for this config, readable. Empty = no config blocker.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
