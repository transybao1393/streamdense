//! StableLM (`model_type` "stablelm", `StableLmForCausalLM`: StableLM-2 1.6B / 12B and their chat / zephyr tunes,
//! StableLM-3B-4E1T) config.json -> `DenseConfig`. Semantics follow mlx_lm `models/stablelm.py` (the engine's
//! reference); keys it requires are required here, its defaults are used for the rest.
//!
//! | config.json | DenseConfig | note |
//! |---|---|---|
//! | `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `vocab_size` | sizes | required |
//! | hidden_size / num_attention_heads | `head_dim` | computed (mlx_lm); a `head_dim` key that disagrees is refused |
//! | `partial_rotary_factor` | `rope_dims = int(f * head_dim)` | required; 0.25 in StableLM-2 (16 of 64 dims) |
//! | `rope_theta` | `rope_theta` | required; rotate-half RoPE (`rope_traditional = false`) |
//! | `rope_scaling` | `rope_scaling` (named by `unsupported()`) | mlx_lm ignores it, HF applies it: refused, not silently dropped |
//! | `use_qkv_bias` | `attention_bias` (q / k / v) | required; o_proj and the MLP have no bias |
//! | `layer_norm_eps` | `eps`, `norm = LayerNorm { bias: true }` | required; every hidden norm is nn.LayerNorm with weight + bias |
//! | `use_parallel_residual` | `block = Parallel` (true) / `Sequential` (false) | mlx_lm default false; parallel: h + attn(n(h)) + mlp(n(h)) |
//! | `qk_layernorm` | `qk_norm`, `qk_norm_kind = HeadLayerNorm`, `names.q_norm = "self_attn.q_layernorm"`, `names.k_norm = "self_attn.k_layernorm"` | mlx_lm default false; per-head LayerNorm without bias (StableLM-2 12B) |
//! | `hidden_act` | `hidden_act` (text), `activation = Silu` | mlx_lm always runs SwiGLU; a config that says otherwise is refused |
//! | `tie_word_embeddings` | `tie_word_embeddings = false` | mlx_lm stablelm always reads lm_head; `true` is refused |
//! | `max_position_embeddings` | `max_pos` | informational |
//! | `bos_token_id`, `eos_token_id` (+ generation_config.json) | `bos`, `eos` | merged in order, no duplicates |
//! | `quantization` / `quantization_config` | `quant` | MLX (group_size, bits); other methods refused |
//!
//! Every layer is full attention; attention scale head_dim^-0.5 (mlx_lm computes it in f32 over f32 q / k).
use dense_engine::config::{Activation, Block, DenseConfig, Family, LayerKind, NormKind, QkNorm, RopeScaling, TensorNames};
use serde_json::Value;

const MOE_KEYS: [&str; 4] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts"];

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

/// Non-default `rope_scaling` / `rope_parameters` as text (`None` for absent / null / "default").
fn scaling_desc(s: &Value) -> Option<String> {
    if !s.is_object() { return None; }
    let ty = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
    if ty == "default" { return None; }
    Some(match s["factor"].as_f64() { Some(f) => format!("{} factor {}", ty, f), None => ty.to_string() })
}

/// Refuse what is not a dense StableLM text model.
fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("stablelm: config.json has no model_type")?;
    if mt == "stablelm_epoch" {
        return Err("stablelm: model_type \"stablelm_epoch\" (remote-code checkpoint, older key names) not supported: \
                    use a transformers-native \"stablelm\" conversion".into());
    }
    if mt != "stablelm" { return Err(format!("stablelm: model_type {:?} is not StableLM (want \"stablelm\")", mt)); }
    if c["text_config"].is_object() { return Err("stablelm: config wraps a text_config: text models only".into()); }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("stablelm: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(x) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "StableLmForCausalLM") {
            return Err(format!("stablelm: architecture {:?} refused: want StableLmForCausalLM", x));
        }
    }
    Ok(())
}

/// MLX quantization block -> (group_size, bits); other quantization formats refused by name. Extra blockers
/// (non-affine mode, per-layer entries, no block) are pushed to `extra`.
fn quant(c: &Value, extra: &mut Vec<String>) -> Result<Option<(usize, usize)>, String> {
    let qb = if c["quantization"].is_object() { &c["quantization"] } else { &c["quantization_config"] };
    let Some(o) = qb.as_object() else {
        extra.push("no MLX quantization block: the weight loader reads MLX 4-bit group-64 models only".into());
        return Ok(None);
    };
    if let Some(m) = qb["quant_method"].as_str() {
        return Err(format!("stablelm: quantization quant_method {:?} is not MLX quantization (MLX 4-bit group 64 only)", m));
    }
    let g = qb["group_size"].as_u64().ok_or("stablelm: quantization block without group_size")? as usize;
    let b = qb["bits"].as_u64().ok_or("stablelm: quantization block without bits")? as usize;
    if let Some(m) = qb["mode"].as_str() {
        if m != "affine" { extra.push(format!("quantization mode {:?} (the kernels read MLX affine only)", m)); }
    }
    let per: Vec<&String> = o.keys().filter(|k| !["bits", "group_size", "mode"].contains(&k.as_str())).collect();
    if !per.is_empty() {
        extra.push(format!("per-layer quantization ({} entries, first {}): the loader wants every linear at one bit width",
                           per.len(), per[0]));
    }
    Ok(Some((g, b)))
}

/// Parse a StableLM config.json (`gen` = generation_config.json for extra eos ids) into a `DenseConfig` plus the
/// blockers that have no `DenseConfig` field. Does not call `validate()`; `eos` may come back empty.
pub fn map_config(c: &Value, gen: Option<&Value>) -> Result<(DenseConfig, Vec<String>), String> {
    check_kind(c)?;
    let req = |k: &str| format!("stablelm: config.json has no {} (required by mlx_lm stablelm; not guessed)", k);
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| req(k));
    let (hidden, inter, layers) = (u("hidden_size")?, u("intermediate_size")?, u("num_hidden_layers")?);
    let (heads, kv_heads, vocab) = (u("num_attention_heads")?, u("num_key_value_heads")?, u("vocab_size")?);
    if heads == 0 || kv_heads == 0 || layers == 0 || hidden % heads != 0 {
        return Err(format!("stablelm: hidden_size {} must be a non-zero multiple of num_attention_heads {} (mlx_lm \
                            raises otherwise); layers and key-value heads must not be 0", hidden, heads));
    }
    let head_dim = hidden / heads;
    if let Some(hd) = c["head_dim"].as_u64() {
        if hd as usize != head_dim {
            return Err(format!("stablelm: head_dim {} in config.json, but mlx_lm computes hidden_size / num_attention_heads = {}", hd, head_dim));
        }
    }
    let mut d = DenseConfig::base("stablelm", Family::StableLm, hidden, inter, layers, heads, kv_heads, head_dim, vocab);
    d.head_dim_explicit = !c["head_dim"].is_null();
    d.kinds = vec![LayerKind::Full; layers];
    let mut extra = vec![];

    // RoPE: partial, rotate-half
    d.rope_theta = c["rope_theta"].as_f64().ok_or_else(|| req("rope_theta"))?;
    let prf = c["partial_rotary_factor"].as_f64().ok_or_else(|| req("partial_rotary_factor"))?;
    d.rope_dims = (prf * head_dim as f64) as usize; // mlx: int(partial_rotary_factor * head_dim)
    d.rope_traditional = false;
    d.rope_scale = RopeScaling::None;
    d.rope_scaling = scaling_desc(&c["rope_scaling"]).or(scaling_desc(&c["rope_parameters"]))
        .map(|s| format!("{} (mlx_lm stablelm ignores rope scaling, HF applies it)", s));

    // attention, norms, block
    d.attention_bias = c["use_qkv_bias"].as_bool().ok_or_else(|| req("use_qkv_bias"))?;
    d.o_bias = false;
    d.mlp_bias = false;
    d.eps = c["layer_norm_eps"].as_f64().ok_or_else(|| req("layer_norm_eps"))? as f32;
    d.norm = NormKind::LayerNorm { bias: true };
    d.block = if c["use_parallel_residual"].as_bool().unwrap_or(false) { Block::Parallel } else { Block::Sequential };
    let mut names = TensorNames::default();
    if c["qk_layernorm"].as_bool().unwrap_or(false) {
        d.qk_norm = true;
        d.qk_norm_kind = QkNorm::HeadLayerNorm;
        names.q_norm = "self_attn.q_layernorm".into();
        names.k_norm = "self_attn.k_layernorm".into();
    }
    d.names = names;
    d.activation = Activation::Silu;
    d.hidden_act = c["hidden_act"].as_str().unwrap_or("silu").to_string();
    if d.hidden_act != "silu" && d.hidden_act != "swish" {
        extra.push(format!("hidden_act {:?}: mlx_lm stablelm always runs SwiGLU (silu), HF would use {:?}", d.hidden_act, d.hidden_act));
    }
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(0) as usize;

    d.tie_word_embeddings = false;
    if c["tie_word_embeddings"].as_bool() == Some(true) {
        extra.push("tie_word_embeddings = true: mlx_lm stablelm always reads its own lm_head".into());
    }

    d.bos = c["bos_token_id"].as_u64().map(|v| v as u32);
    for v in [Some(&c["eos_token_id"]), gen.map(|g| &g["eos_token_id"])].into_iter().flatten() {
        for id in ids(v) { if !d.eos.contains(&id) { d.eos.push(id); } }
    }
    d.quant = quant(c, &mut extra)?;
    Ok((d, extra))
}

/// Everything that stops this config from running (config.json alone). Empty = runs.
pub fn blockers(c: &Value) -> Vec<String> {
    match map_config(c, None) {
        Err(e) => vec![e],
        Ok((d, extra)) => {
            let mut b = d.unsupported();
            b.extend(extra);
            b
        }
    }
}
