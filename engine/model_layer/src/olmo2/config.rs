//! OLMo 2 `config.json` -> `DenseConfig`. Field meanings and defaults: mlx_lm `models/olmo2.py` `ModelArgs`, the same
//! fields and defaults as mlx_lm llama (read by `mistral::config::llama_like`, table there) without `layer_types` /
//! `sliding_window` (mlx olmo2 has neither: every layer full attention). OLMo 2 specifics:
//!
//! | mlx_lm olmo2 | DenseConfig |
//! |---|---|
//! | no input norms; `h += post_attention_layernorm(attn(h)); h += post_feedforward_layernorm(mlp(h))` | `block = Block::PostNorm` (default `names` for both norms) |
//! | `q_norm` RMSNorm over n_heads * head_dim, `k_norm` over n_kv_heads * head_dim, before the head split and RoPE | `qk_norm = true`, `qk_norm_kind = QkNorm::FullRms` |
//! | `attention_bias` on q, k, v and o | `attention_bias` AND `o_bias` |
//! | `mlp_bias` on gate, up, down | `mlp_bias` |
//! | `tie_word_embeddings` default TRUE (mlx) | `tie_word_embeddings` (both fetched configs: false) |
//! | rms_norm_eps (all norms incl. q / k norm) | `eps` |
//!
//! Refused by name: OLMo 1 (`olmo`: non-parametric LayerNorm, clip_qkv), OLMoE (`olmoe`: MoE), OLMo 3 (`olmo3`: its own
//! model_type, not read by this family), multimodal wrappers.
use dense_engine::config::{Block, DenseConfig, Family, QkNorm};
use serde_json::Value;

use crate::mistral::config::llama_like;

/// OLMo relatives refused with their reason.
pub const REFUSED_TYPES: [(&str, &str); 4] = [
    ("olmo", "OLMo 1: non-parametric LayerNorm (no weights), a different layer"),
    ("olmoe", "mixture-of-experts: dense models only"),
    ("olmo3", "OLMo 3: its own model_type, not ported for this family"),
    ("molmo", "multimodal"),
];

const MOE_KEYS: [&str; 4] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts"];

fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("olmo2: config.json has no model_type")?;
    if let Some((_, why)) = REFUSED_TYPES.iter().find(|(t, _)| *t == mt) {
        return Err(format!("olmo2: model_type {:?} refused: {}", mt, why));
    }
    if mt != "olmo2" { return Err(format!("olmo2: model_type {:?} is not \"olmo2\"", mt)); }
    if c["text_config"].is_object() || !c["vision_config"].is_null() {
        return Err("olmo2: config has a text_config / vision_config (multimodal wrapper): dense text models only".into());
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("olmo2: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(a) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "Olmo2ForCausalLM") {
            return Err(format!("olmo2: architecture {:?}: want Olmo2ForCausalLM", a));
        }
    }
    Ok(())
}

/// Parse an OLMo 2 config.json (`gen` = generation_config.json, `tok_eos` = tokenizer eos id).
pub fn parse(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    check_kind(c)?;
    let mut d = llama_like(c, "olmo2", Family::Olmo2, false, gen, tok_eos)?;
    d.block = Block::PostNorm;
    d.qk_norm = true;
    d.qk_norm_kind = QkNorm::FullRms;
    Ok(d)
}

/// `parse` + eos present + `validate()`.
pub fn dense_config_with(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    let d = parse(c, gen, tok_eos)?;
    if d.eos.is_empty() { return Err("olmo2: no eos_token_id in config.json / generation_config.json / tokenizer_config.json".into()); }
    d.validate().map_err(|e| format!("olmo2: {}", e))?;
    Ok(d)
}

/// Everything the engine refuses for this config (see `super::blockers`).
pub fn blockers(c: &Value) -> Vec<String> {
    let d = match parse(c, None, None) {
        Ok(d) => d,
        Err(e) => return vec![e],
    };
    let mut b = d.unsupported();
    if d.quant.is_none() {
        b.push("no MLX quantization block: the weight loader takes MLX 4-bit group-64 safetensors only".into());
    }
    b
}
