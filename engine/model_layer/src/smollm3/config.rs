//! SmolLM3 `config.json` -> `DenseConfig`. mlx_lm `models/smollm3.py`: `ModelArgs` subclasses mlx_lm llama `ModelArgs`
//! (all llama fields and defaults, read by `mistral::config::llama_like`, table there, incl. `layer_types` /
//! `sliding_window`) and the model is `llama.LlamaModel` with RoPE switched off on some layers:
//!
//! | config.json key (mlx default) | DenseConfig | note |
//! |---|---|---|
//! | no_rope_layers (per layer, 1 = use RoPE, 0 = no RoPE) | `no_rope_layers[i] = (cfg[i] == 0)` | inverse sense; length must be num_hidden_layers |
//! | no_rope_layer_interval (4), when no_rope_layers is absent | `no_rope_layers[i] = ((i + 1) % interval == 0)` | |
//! | tie_word_embeddings (TRUE in mlx llama) | `tie_word_embeddings` | SmolLM3-3B has no key: tied (its weights have no lm_head) |
//! | use_sliding_window, max_window_layers | not read | mlx llama reads `layer_types` only (SmolLM3-3B: all full_attention) |
//!
//! Refused: other model_types, MoE keys, multimodal wrappers (SmolVLM is model_type "smolvlm" / "idefics3").
use dense_engine::config::{DenseConfig, Family};
use serde_json::Value;

use crate::mistral::config::llama_like;

const MOE_KEYS: [&str; 4] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts"];

fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("smollm3: config.json has no model_type")?;
    if mt != "smollm3" { return Err(format!("smollm3: model_type {:?} is not \"smollm3\"", mt)); }
    if c["text_config"].is_object() || !c["vision_config"].is_null() {
        return Err("smollm3: config has a text_config / vision_config (multimodal wrapper): dense text models only".into());
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("smollm3: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(a) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "SmolLM3ForCausalLM") {
            return Err(format!("smollm3: architecture {:?}: want SmolLM3ForCausalLM", a));
        }
    }
    Ok(())
}

/// Per layer, true = no RoPE (the engine's sense), from `no_rope_layers` / `no_rope_layer_interval` as mlx_lm.
pub fn no_rope(c: &Value, layers: usize) -> Result<Vec<bool>, String> {
    match &c["no_rope_layers"] {
        Value::Array(a) => {
            if a.len() != layers {
                return Err(format!("smollm3: no_rope_layers has {} entries, num_hidden_layers = {} (mlx_lm: length mismatch)", a.len(), layers));
            }
            a.iter().enumerate().map(|(i, x)| match x.as_i64() {
                Some(v) => Ok(v == 0),
                None => Err(format!("smollm3: no_rope_layers[{}] = {} is not an integer", i, x)),
            }).collect()
        }
        Value::Null => {
            let iv = c["no_rope_layer_interval"].as_u64().unwrap_or(4) as usize;
            if iv == 0 { return Err("smollm3: no_rope_layer_interval = 0".into()); }
            Ok((0..layers).map(|i| (i + 1) % iv == 0).collect())
        }
        v => Err(format!("smollm3: no_rope_layers {} is not a list", v)),
    }
}

/// Parse a SmolLM3 config.json (`gen` = generation_config.json, `tok_eos` = tokenizer eos id).
pub fn parse(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    check_kind(c)?;
    let mut d = llama_like(c, "smollm3", Family::SmolLM3, true, gen, tok_eos)?;
    d.no_rope_layers = no_rope(c, d.layers)?;
    Ok(d)
}

/// `parse` + eos present + `validate()`.
pub fn dense_config_with(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    let d = parse(c, gen, tok_eos)?;
    if d.eos.is_empty() { return Err("smollm3: no eos_token_id in config.json / generation_config.json / tokenizer_config.json".into()); }
    d.validate().map_err(|e| format!("smollm3: {}", e))?;
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
