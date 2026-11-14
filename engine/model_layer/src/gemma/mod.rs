//! Gemma family: config.json -> `dense_engine::config::DenseConfig` (Gemma 1, Gemma 2, Gemma 3 text) and the Gemma chat
//! template. No inference code: dense-engine runs the math from the `DenseConfig` fields (GeGLU, (1 + w) norms, four
//! norms, QK-norm, embedding scale, soft caps, sliding windows, two RoPE tables, linear RoPE scaling).
//!
//! - `config`: per-generation field mapping (table in its doc), refusals of non-dense / multimodal relatives.
//! - `chat`: `chat_prompt`, the Gemma turn template. Its text starts with `<bos>`: encode it with `Tokenizer::encode`,
//!   not `encode_with_special` (that would add a second BOS).
//!
//! Multimodal `gemma3` (root config with `text_config` + `vision_config`) is taken only when the folder's weights are
//! the language model alone under `model.*` (what the dense loader reads); a checkpoint with `language_model.*` or
//! vision tensors is refused by name. Run on a real Gemma 3 checkpoint (gemma-3-270m-it, MLX 4-bit): greedy output equal to
//! mlx_lm. Gemma 1 (gemma-1.1-2b-it) and Gemma 2 (gemma-2-2b-it): greedy output equal to mlx_lm (2026-11-14). Gemma 1
//! activation: tanh GELU per the model config, as transformers; mlx_lm `models/gemma.py` uses exact-erf `nn.gelu`, yet
//! the greedy tokens matched.
//!
//! Interface: `MODEL_TYPES`, `dense_config`, `chat_prompt`, `blockers`.
pub mod chat;
pub mod config;
#[cfg(test)]
mod tests;

pub use chat::chat_prompt;

use dense_engine::config::DenseConfig;
use serde_json::Value;
use std::path::Path;

/// The model_type strings this family takes (plus a text-only usable multimodal `gemma3`, see the module doc).
pub const MODEL_TYPES: &[&str] = &["gemma", "gemma2", "gemma3_text"];

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// The `<end_of_turn>` id from `tokenizer_config.json` `added_tokens_decoder` (a small file; tokenizer.json is not
/// read). Instruction-tuned Gemma ends each reply with it; not every generation_config.json lists it.
fn end_of_turn_id(dir: &Path) -> Option<u32> {
    let tc = read_json(&dir.join("tokenizer_config.json"))?;
    tc["added_tokens_decoder"].as_object()?.iter()
        .find(|(_, v)| v["content"].as_str() == Some("<end_of_turn>"))
        .and_then(|(k, _)| k.parse().ok())
}

/// Tensor names of the folder: index.json weight_map keys, else the model.safetensors header.
fn weight_names(dir: &Path) -> Result<Vec<String>, String> {
    let idx = dir.join("model.safetensors.index.json");
    if idx.exists() {
        let v = read_json(&idx).ok_or(format!("{}: not readable JSON", idx.display()))?;
        let wm = v["weight_map"].as_object().ok_or(format!("{}: no weight_map", idx.display()))?;
        return Ok(wm.keys().cloned().collect());
    }
    let f = dir.join("model.safetensors");
    let si = streamdense_core::SafeIndex::open_file(&f.to_string_lossy())?;
    Ok(si.names().map(str::to_string).collect())
}

/// Multimodal `gemma3`: the weights must be the language model only, under the names the dense loader reads.
fn check_text_only_weights(dir: &Path) -> Result<(), String> {
    let names = weight_names(dir).map_err(|e| format!("multimodal gemma3: weights not readable, cannot confirm they are \
                                                       the language model only: {}", e))?;
    if let Some(n) = names.iter().find(|n| n.starts_with("language_model.")) {
        return Err(format!("multimodal gemma3 refused: weights are a multimodal checkpoint ({} ...); the dense loader \
                            reads model.* (convert the text model alone, as gemma3_text)", n));
    }
    if let Some(n) = names.iter().find(|n| n.contains("vision") || n.contains("multi_modal_projector")) {
        return Err(format!("multimodal gemma3 refused: vision tensors present ({}); text-only weights wanted", n));
    }
    if let Some(n) = names.iter().find(|n| !(n.starts_with("model.") || n.starts_with("lm_head."))) {
        return Err(format!("multimodal gemma3 refused: tensor {} is not part of a text model (model.* / lm_head.*)", n));
    }
    if !names.iter().any(|n| n.starts_with("model.embed_tokens.")) {
        return Err("multimodal gemma3 refused: no model.embed_tokens tensor in the weights".into());
    }
    Ok(())
}

/// config.json (root object) of the folder `model_dir` -> validated `DenseConfig`. Reads from `model_dir`:
/// generation_config.json (eos, optional), tokenizer_config.json (`<end_of_turn>` id, optional) and, for a multimodal
/// `gemma3` root, the weight names (index.json or the safetensors header; never weight data).
pub fn dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String> {
    let tv = config::text_view(cfg)?;
    if tv.multimodal_wrapper { check_text_only_weights(model_dir)?; }
    let gen = read_json(&model_dir.join("generation_config.json"));
    let extra: Vec<u32> = end_of_turn_id(model_dir).into_iter().collect();
    let d = config::map_config(cfg, gen.as_ref(), &extra)?;
    if d.eos.is_empty() {
        return Err("config: no eos_token_id in config.json or generation_config.json, and no <end_of_turn> token".into());
    }
    d.validate()?;
    Ok(d)
}

/// What stops this config.json from running, from the config alone (empty = nothing found): the refusal / parse error,
/// else every `DenseConfig::unsupported()` item. Not covered here: eos from generation_config.json and, for a
/// multimodal `gemma3` root, the weight check (both need the folder: `dense_config`).
pub fn blockers(cfg: &Value) -> Vec<String> {
    match config::map_config(cfg, None, &[]) {
        Err(e) => vec![e],
        Ok(d) => d.unsupported(),
    }
}
