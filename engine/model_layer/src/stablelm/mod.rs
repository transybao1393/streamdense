//! StableLM family (`model_type` "stablelm": StableLM-2 1.6B / 12B base, chat and zephyr tunes): config.json ->
//! `dense_engine::config::DenseConfig`, the chat templates as fixed strings, and the list of what blocks a config. No
//! inference code: the math runs in dense-engine (`Family::StableLm`; LayerNorm with bias, partial RoPE, optional
//! parallel residual and per-head QK LayerNorm).
//!
//! - `config`: field mapping (table in its doc), refusals.
//! - `template`: `TemplateKind` (ChatML of stablelm-2-chat, Zephyr of stablelm-2-zephyr), picked from the folder.
//!
//! Checked against real files (mlx-community/stablelm-2-1_6b-chat-4bit, stablelm-2-zephyr-1_6b-4bit: config.json,
//! tokenizer_config.json, weight index): chat strings equal to HF `apply_chat_template` renders (tests). Not run on the
//! engine yet. The MLX conversions keep `embed_tokens` and every norm in F16 (mlx_lm then computes in float16, the
//! engine in bf16).
//!
//! Tokenizer: StableLM-2 ships a `tokenizer.json` (GPT-2 style BPE converted from the Arcade100k tiktoken file); the
//! repo's custom `tokenization_arcade100k.py` is not needed. Prompt encoding: no post-processor, no BOS
//! (`Tokenizer::encode` with special tokens recognized).
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
pub const MODEL_TYPES: &[&str] = &["stablelm"];

/// The default system text of the stablelm-2-chat template.
pub const CHAT_DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// The engine config of a StableLM model: `cfg` = parsed config.json, `model_dir` for `generation_config.json` (extra
/// eos ids, optional). Errors: refusals, missing required keys, a missing eos, every blocker.
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let gp = model_dir.join("generation_config.json");
    let gen: Option<Value> = match std::fs::read_to_string(&gp) {
        Ok(t) => Some(serde_json::from_str(&t).map_err(|e| format!("{}: {}", gp.display(), e))?),
        Err(_) => None,
    };
    let (d, extra) = config::map_config(cfg, gen.as_ref())?;
    if d.eos.is_empty() { return Err("stablelm: no eos_token_id in config.json or generation_config.json".into()); }
    let mut errs = d.unsupported();
    errs.extend(extra);
    if !errs.is_empty() { return Err(format!("stablelm config not supported: {}", errs.join("; "))); }
    Ok(d)
}

/// Chat prompt in the stablelm-2-chat ChatML template (the family default, default system "You are a helpful
/// assistant."). For a model folder use `chat_prompt_dir` (picks ChatML or Zephyr from the folder).
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    TemplateKind::ChatMl { default_system: Some(CHAT_DEFAULT_SYSTEM.into()) }.render(messages, add_generation_prompt)
}

/// Chat prompt in the template `model_dir` declares (`template_kind`); unknown templates are refused by name.
pub fn chat_prompt_dir(model_dir: &Path, messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    template_kind(model_dir)?.render(messages, add_generation_prompt)
}

/// Everything that stops this config from running, by name (config.json only). Empty = runs.
pub fn blockers(cfg: &Value) -> Vec<String> {
    config::blockers(cfg)
}
