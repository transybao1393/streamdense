//! Qwen model_type lists and config reading (moved here from family.rs / engine.rs; behaviour unchanged).
//!
//! The dense members (qwen2, qwen3) are read by `dense_engine::config::DenseConfig::from_model_dir`, which validates
//! nothing here (`blockers` lists what the engine still refuses). The hybrid (qwen3_5 / qwen3_5_text, MLX Qwen3.8)
//! runs on the main engine and has no `DenseConfig`. The hybrid model_type strings are taken from the MLX Qwen3.8
//! config (`QwenConfig` itself never reads model_type).
use dense_engine::config::DenseConfig;

use crate::family::{Detected, EngineKind};

/// Qwen2 / Qwen2.5 dense.
pub const QWEN2_TYPES: &[&str] = &["qwen2"];
/// Qwen3 dense.
pub const QWEN3_TYPES: &[&str] = &["qwen3"];
/// Qwen3.5 / Qwen3.8 hybrid (root "qwen3_5", text_config "qwen3_5_text").
pub const HYBRID_TYPES: &[&str] = &["qwen3_5", "qwen3_5_text"];

/// The dense-engine config of a Qwen folder (`DenseConfig::from_model_dir`, not validated here). Err for the hybrid,
/// which does not run on dense-engine.
pub fn dense_config(dir: &str, d: &Detected) -> Result<DenseConfig, String> {
    if d.entry.engine == EngineKind::DenseEngine {
        DenseConfig::from_model_dir(dir)
    } else {
        Err(format!("{}: runs on {}, not dense-engine", d.entry.name, d.entry.engine.name()))
    }
}

/// What the engine still refuses for a Qwen dense config (`DenseConfig::unsupported`). Empty = no blocker.
pub fn blockers(cfg: &DenseConfig) -> Vec<String> {
    cfg.unsupported()
}
