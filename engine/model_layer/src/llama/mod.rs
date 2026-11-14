//! Llama family of the model layer (`model_type` "llama": Llama 2, Llama 3 / 3.1 / 3.2, TinyLlama, SmolLM / SmolLM2).
//! Reads a Llama `config.json` into the `dense_engine::config::DenseConfig` the engine takes, and renders the family's
//! chat templates as fixed strings (no Jinja engine). No inference code: the model runs on `engine/dense-engine`.
//!
//! Run on real models through the model layer (SmolLM-135M-Instruct, Llama-3.2-1B-Instruct, MLX 4-bit): greedy output
//! equal to mlx_lm, chat strings equal to `apply_chat_template` (see engine/model_layer/README.md).
//!
//! - `config`: config.json -> `DenseConfig` (`parse`), Llama-only refusals, `blockers`.
//! - `template`: chat template kinds (Llama 3 header, Llama 3.1 / 3.2 dated header, ChatML for SmolLM2), picked from
//!   the model folder's `chat_template.jinja` / `tokenizer_config.json` text markers.
//!
//! "mistral" is NOT in `MODEL_TYPES`: its `sliding_window` (v0.1: 4096 on every layer) is handled differently by HF
//! (masks keys outside the window) and mlx_lm (Mistral runs through the Llama module, which ignores it), and its chat
//! template is the `[INST]` one, which is not ported. A Mistral folder therefore gets the model layer's
//! "not supported yet" error, not a silently different model.
//!
//! Prompt encoding: the rendered chat strings already contain the BOS text where the template puts one
//! (`<|begin_of_text|>`), so encode them with special-token recognition and WITHOUT the tokenizer's post-processor BOS
//! (HF `apply_chat_template` tokenizes with `add_special_tokens=False`). Plain-text prompts (base models) do take the
//! post-processor BOS (`Tokenizer::encode_with_special`).
use std::path::Path;

use dense_engine::config::DenseConfig;
use serde_json::Value;

pub mod config;
pub mod template;
#[cfg(test)]
mod tests;

pub use template::{template_kind, TemplateKind};

/// `model_type` strings this family takes.
pub const MODEL_TYPES: &[&str] = &["llama"];

/// The engine config of a Llama model: `cfg` = parsed config.json, `model_dir` for `generation_config.json`
/// (eos / bos). Errors: non-dense / non-Llama configs, Llama features the engine would get wrong (named), and
/// `DenseConfig::validate()`'s message. A config without an MLX quantization block passes here (the weight loader
/// refuses it; `blockers` names it).
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gen: Option<Value> = std::fs::read_to_string(model_dir.join("generation_config.json"))
        .ok().and_then(|t| serde_json::from_str(&t).ok());
    config::dense_config_with_gen(cfg, gen.as_ref())
}

/// Chat prompt in the Llama 3 header template (the family default):
/// `<|begin_of_text|>` + per message `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` (content
/// trimmed, as the template's `| trim`), + `<|start_header_id|>assistant<|end_header_id|>\n\n` when
/// `add_generation_prompt`. Roles: system, user, assistant. For a model folder use `chat_prompt_dir`, which picks the
/// template the folder declares (Llama 3.1 / 3.2 dated system block, ChatML for SmolLM2).
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    TemplateKind::Llama3.render(messages, add_generation_prompt)
}

/// Chat prompt in the template `model_dir` declares (`template_kind`); unknown templates are refused by name.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    template_kind(model_dir)?.render(messages, add_generation_prompt)
}

/// Everything the engine still refuses for this config, readable. Empty = no config blocker. Config-only: eos from
/// generation_config.json, tokenizer and chat template are checked by `dense_config` / `template_kind`.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
