//! Granite family: config.json -> `dense_engine::config::DenseConfig`, the chat prompt of the model's own template, and
//! the list of what blocks a config. No inference code: the math runs in dense-engine (`Family::Granite`).
//!
//! Field meaning comes from transformers `models/granite/modeling_granite.py` and mlx_lm `models/granite.py`:
//!
//! | Granite config.json | DenseConfig | note |
//! |---|---|---|
//! | `attention_multiplier` | `attn_scale = Some(m)` | q . k scale, replaces head_dim^-0.5 |
//! | `embedding_multiplier` | `embed_scale` | embeddings * m |
//! | `residual_multiplier` | `residual_multiplier` | h = h + r * m |
//! | `logits_scaling` | `logits_scaling` | logits / s (same direction in HF, mlx_lm and the engine) |
//! | hidden_size / num_attention_heads | `head_dim`, `rope_dims` | Granite computes it; a `head_dim` key that disagrees is refused |
//! | `rope_theta` / `rope_parameters.rope_theta` | `rope_theta` | both present and different: refused |
//! | `rope_scaling` / `rope_parameters` | `rope_scale` (linear, llama3) or `rope_scaling` (other: named by `unsupported()`) | |
//! | `attention_bias` | `attention_bias` AND `o_bias` | Granite puts the bias on q, k, v and o |
//! | `mlp_bias` | `mlp_bias` | gate, up, down |
//! | `tie_word_embeddings` | `tie_word_embeddings` | |
//! | `hidden_act` | `activation` (`Activation::from_hf`) | |
//! | `eos_token_id` (+ generation_config.json) | `eos` | merged, in order, no duplicates |
//! | `quantization` / `quantization_config` | `quant` (group, bits) | mode other than affine or per-layer entries: blocker |
//!
//! The math fields (`attention_multiplier`, `embedding_multiplier`, `residual_multiplier`, `logits_scaling`,
//! `tie_word_embeddings`, `attention_bias`, `mlp_bias`, `rms_norm_eps`, `hidden_act`, rope theta) must be present:
//! the sources disagree on some defaults (tie_word_embeddings: transformers false, mlx_lm true) and a wrong multiplier
//! gives wrong output without any error, so nothing is guessed.
//!
//! Refused by name: the MoE members (`granitemoe`, `granitemoeshared`, `granitemoehybrid` with experts): "dense models
//! only"; non-attention layers (`layer_types` with "mamba"): "not supported yet"; `granitemoehybrid` without experts and
//! attention only (FFN under `shared_mlp`, which the weight loader does not read): "not supported yet"; configs that
//! wrap a `text_config` (vision / speech): "not supported yet".
mod chat;

pub use chat::chat_prompt;

use dense_engine::config::{rope_scaling_kind, Activation, DenseConfig, Family, LayerKind, RopeScaling};
use serde_json::Value;

/// model_type values this family reads.
pub const MODEL_TYPES: &[&str] = &["granite"];

/// The other model_type values of the Granite family: refused with their reason, not as "unknown".
const OTHER_MEMBERS: [&str; 3] = ["granitemoe", "granitemoeshared", "granitemoehybrid"];

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

/// Refuse what is not a plain dense Granite: wrappers, MoE, non-attention layers, other family members.
fn check_kind(root: &Value) -> Result<(), String> {
    if root["text_config"].is_object() {
        return Err(format!("granite: model_type {:?} wraps a text_config (vision / speech model): not supported yet",
                           root["model_type"].as_str().unwrap_or("?")));
    }
    let model_type = root["model_type"].as_str().ok_or("granite: config.json has no model_type")?;
    if !MODEL_TYPES.contains(&model_type) && !OTHER_MEMBERS.contains(&model_type) {
        return Err(format!("granite: model_type {:?} is not a Granite model (this family reads {:?})", model_type, MODEL_TYPES));
    }
    // expert count: transformers defaults to 8 for the MoE model types when the key is absent
    let experts = ["num_local_experts", "num_experts", "n_routed_experts"].iter().find_map(|k| root[*k].as_u64())
        .unwrap_or(if model_type == "granite" { 0 } else { 8 });
    if experts > 0 {
        return Err(format!("granite: model_type {:?} has {} experts per layer: mixture-of-experts model rejected, dense models only",
                           model_type, experts));
    }
    if let Some(a) = root["layer_types"].as_array() {
        let mut other: Vec<&str> = a.iter().map(|t| t.as_str().unwrap_or("?")).filter(|t| *t != "attention").collect();
        other.sort_unstable();
        other.dedup();
        if !other.is_empty() {
            return Err(format!("granite: layer type {} not supported yet (model_type {:?}, hybrid model): the engine has attention layers only",
                               other.join(", "), model_type));
        }
    }
    if model_type != "granite" {
        return Err(format!("granite: model_type {:?} without experts not supported yet: its FFN is under shared_mlp, \
                            the weight loader reads mlp.gate_proj / up_proj / down_proj", model_type));
    }
    Ok(())
}

/// Non-default RoPE scaling as text (`None` for absent / "default").
fn scaling_text(s: &Value) -> Option<String> {
    if !s.is_object() { return None; }
    let ty = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
    if ty == "default" { return None; }
    let mut d = ty.to_string();
    if let Some(f) = s["factor"].as_f64() { d += &format!(" factor {}", f); }
    Some(d)
}

/// Parse into a `DenseConfig` plus the blockers `DenseConfig` has no field for (quantization mode, per-layer
/// quantization). `Err` = the config is refused outright. An empty `eos` is left to the caller.
fn build(root: &Value, gen: Option<&Value>) -> Result<(DenseConfig, Vec<String>), String> {
    check_kind(root)?;
    let model_type = root["model_type"].as_str().unwrap_or("granite");
    let u = |k: &str| root[k].as_u64().map(|v| v as usize).ok_or(format!("granite: config.json has no {}", k));
    let not_guessed = |k: &str| format!("granite: config.json has no {} (it changes the math, so it is not guessed)", k);
    let f = |k: &str| root[k].as_f64().ok_or_else(|| not_guessed(k));
    let b = |k: &str| root[k].as_bool().ok_or_else(|| not_guessed(k));

    let (hidden, heads) = (u("hidden_size")?, u("num_attention_heads")?);
    if heads == 0 || hidden % heads != 0 {
        return Err(format!("granite: hidden_size {} is not a multiple of num_attention_heads {}", hidden, heads));
    }
    let head_dim = hidden / heads;
    if let Some(hd) = root["head_dim"].as_u64() {
        if hd as usize != head_dim {
            return Err(format!("granite: head_dim {} in config.json, but Granite (mlx_lm) computes hidden_size / num_attention_heads = {}",
                               hd, head_dim));
        }
    }
    let layers = u("num_hidden_layers")?;
    let kv_heads = root["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads);
    let mut d = DenseConfig::base(model_type, Family::Granite, hidden, u("intermediate_size")?, layers, heads, kv_heads,
                                  head_dim, u("vocab_size")?);
    d.head_dim_explicit = !root["head_dim"].is_null();
    d.kinds = vec![LayerKind::Full; layers];

    let rp = &root["rope_parameters"];
    d.rope_theta = match (root["rope_theta"].as_f64(), rp["rope_theta"].as_f64()) {
        (Some(a), Some(c)) if a != c => return Err(format!("granite: rope_theta {} and rope_parameters.rope_theta {} differ", a, c)),
        (Some(a), _) | (None, Some(a)) => a,
        (None, None) => return Err(not_guessed("rope_theta")),
    };
    let prf = root["partial_rotary_factor"].as_f64().or(rp["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
    d.rope_dims = (prf * head_dim as f64).round() as usize;
    match rope_scaling_kind(&root["rope_scaling"]).or(rope_scaling_kind(rp)) {
        Some(k) => (d.rope_scale, d.rope_scaling) = (k, None),
        None => (d.rope_scale, d.rope_scaling) = (RopeScaling::None, scaling_text(&root["rope_scaling"]).or(scaling_text(rp))),
    }

    d.attn_scale = Some(f("attention_multiplier")?);
    d.embed_scale = f("embedding_multiplier")?;
    d.residual_multiplier = f("residual_multiplier")?;
    d.logits_scaling = f("logits_scaling")?;
    let attention_bias = b("attention_bias")?;
    (d.attention_bias, d.o_bias) = (attention_bias, attention_bias);
    d.mlp_bias = b("mlp_bias")?;
    d.tie_word_embeddings = b("tie_word_embeddings")?;
    d.qk_norm = false;
    d.eps = f("rms_norm_eps")? as f32;
    d.max_pos = root["max_position_embeddings"].as_u64().unwrap_or(0) as usize;
    let act = root["hidden_act"].as_str().ok_or_else(|| not_guessed("hidden_act"))?;
    (d.hidden_act, d.activation) = (act.to_string(), Activation::from_hf(act));
    d.bos = root["bos_token_id"].as_u64().map(|v| v as u32);
    d.eos = vec![];
    for v in [Some(&root["eos_token_id"]), gen.map(|g| &g["eos_token_id"])].into_iter().flatten() {
        for id in ids(v) { if !d.eos.contains(&id) { d.eos.push(id); } }
    }

    let mut extra = vec![];
    let qb = if root["quantization"].is_object() { &root["quantization"] } else { &root["quantization_config"] };
    d.quant = match qb.as_object() {
        None => {
            extra.push("no quantization block: the weight loader reads MLX 4-bit models only".to_string());
            None
        }
        Some(o) => {
            let g = qb["group_size"].as_u64().ok_or("granite: quantization without group_size")? as usize;
            let bits = qb["bits"].as_u64().ok_or("granite: quantization without bits")? as usize;
            if let Some(m) = qb["mode"].as_str() {
                if m != "affine" { extra.push(format!("quantization mode {:?} (the kernels read MLX affine only)", m)); }
            }
            let per: Vec<&String> = o.keys().filter(|k| !["bits", "group_size", "mode"].contains(&k.as_str())).collect();
            if !per.is_empty() {
                extra.push(format!("per-layer quantization ({} entries, first {}): the loader wants every linear at one bit width",
                                   per.len(), per[0]));
            }
            Some((g, bits))
        }
    };
    Ok((d, extra))
}

/// The `DenseConfig` dense-engine runs for a Granite config.json (`model_dir/generation_config.json` adds eos ids when
/// present). `Err` names every reason it cannot run: refusals, then `DenseConfig::validate()` failures and the
/// quantization blockers.
pub fn dense_config(cfg: &Value, model_dir: &std::path::Path) -> Result<DenseConfig, String> {
    let gp = model_dir.join("generation_config.json");
    let gen: Option<Value> = match std::fs::read_to_string(&gp) {
        Ok(t) => Some(serde_json::from_str(&t).map_err(|e| format!("{}: {}", gp.display(), e))?),
        Err(_) => None,
    };
    let (d, extra) = build(cfg, gen.as_ref())?;
    if d.eos.is_empty() { return Err("granite: no eos_token_id in config.json or generation_config.json".into()); }
    let mut errs = match d.validate() { Ok(()) => vec![], Err(_) => d.unsupported() };
    errs.extend(extra);
    if !errs.is_empty() { return Err(format!("granite config not supported: {}", errs.join("; "))); }
    Ok(d)
}

/// Everything that stops this config from running, by name (config.json only; eos may come from
/// generation_config.json, so a missing eos is not listed here). Empty = runs.
pub fn blockers(cfg: &Value) -> Vec<String> {
    match build(cfg, None) {
        Err(e) => vec![e],
        Ok((d, extra)) => {
            let mut e = d.unsupported();
            e.extend(extra);
            e
        }
    }
}

#[cfg(test)]
mod tests;
