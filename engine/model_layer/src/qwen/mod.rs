//! Qwen family of the model layer: qwen2 / qwen3 dense (engine/dense-engine) and the qwen3.8 hybrid (main engine,
//! engine/crates/streamdense-model). No inference code.
//!
//! - `config`: the Qwen model_type lists (`QWEN2_TYPES`, `QWEN3_TYPES`, `HYBRID_TYPES`, used by `family::FAMILIES`),
//!   `dense_config` (`DenseConfig::from_model_dir` for the dense members), `blockers` (`DenseConfig::unsupported`).
//! - `chat`: `--chat` prompt. The template kind is read from the folder's own template text; rendering reuses the
//!   gateway's Qwen3.8 renderer (`streamdense_tokenizer::render`) and the layer's ChatML renderer
//!   (`llama::TemplateKind::ChatMl`). No template text of its own.
//!
//! Prompt encoding: ChatML has no BOS; encode the rendered text with `Tokenizer::encode`, as before for Qwen.
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `blockers` (+ `template_kind`, `TemplateKind`).
use std::path::Path;

pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::{kind_from_text, template_kind, TemplateKind};
pub use config::{blockers, dense_config, HYBRID_TYPES, QWEN2_TYPES, QWEN3_TYPES};

/// Every `model_type` string this family takes.
pub const MODEL_TYPES: &[&str] = &["qwen2", "qwen3", "qwen3_5", "qwen3_5_text"];

/// Chat prompt in the template `model_dir` declares (`template_kind`); unknown templates are refused by name.
pub fn chat_prompt(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    template_kind(model_dir)?.render(messages, add_generation_prompt)
}
