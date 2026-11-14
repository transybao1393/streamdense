//! Llama `config.json` -> `DenseConfig`. Field meanings and defaults follow HF transformers `LlamaConfig`
//! (`models/llama/configuration_llama.py`): `num_key_value_heads` defaults to the head count, `head_dim` to
//! hidden / heads, `rope_theta` to 10000, `rms_norm_eps` to 1e-6, `attention_bias`, `mlp_bias` and
//! `tie_word_embeddings` to false, `hidden_act` to silu, `max_position_embeddings` to 2048. Token ids get no default:
//! a config without eos (config.json or generation_config.json) is refused.
//!
//! HF Llama semantics mapped onto the engine fields:
//! - `attention_bias` puts a bias on q / k / v AND o_proj (`LlamaAttention`): `attention_bias` + `o_bias`.
//! - `mlp_bias` = gate / up / down bias: `mlp_bias`.
//! - RoPE rotate-half over the full head dim; `rope_scaling` llama3 / linear -> `rope_scale`; anything else (yarn,
//!   dynamic, longrope) is kept as a description in `rope_scaling` and refused by `validate()`.
//! - no q / k norm, (w) RMSNorm, scale head_dim^-0.5, no sliding window (HF Llama never reads `sliding_window`).
//!
//! Parsing is separate from checking: `parse` fails only for configs that are not dense Llama at all;
//! `unsupported()` / `blockers` name what the engine cannot run.
use dense_engine::config::{is_hybrid, rope_scaling_kind, Activation, DenseConfig, Family, LayerKind, RopeScaling};
use serde_json::Value;

/// Keys that mark a mixture-of-experts config (the engine's list + Mixtral / Llama 4 `num_local_experts`).
const MOE_KEYS: [&str; 4] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts"];

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

/// Readable description of a non-default `rope_scaling` / `rope_parameters` object, `None` for absent / default.
fn scaling_desc(s: &Value) -> Option<String> {
    if !s.is_object() { return None; }
    let ty = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
    if ty == "default" { return None; }
    let mut d = ty.to_string();
    for k in ["factor", "low_freq_factor", "high_freq_factor", "beta_fast", "beta_slow"] {
        if let Some(f) = s[k].as_f64() { d += &format!(" {} {}", k, f); }
    }
    if let Some(o) = s["original_max_position_embeddings"].as_u64() { d += &format!(" original_max_position_embeddings {}", o); }
    Some(d)
}

/// Parse a Llama config.json (`gen` = generation_config.json, for eos / bos). Errors only for configs that are not a
/// dense Llama; `eos` may come back empty (checked by `dense_config_with_gen`).
pub fn parse(c: &Value, gen: Option<&Value>) -> Result<DenseConfig, String> {
    let model_type = c["model_type"].as_str().unwrap_or("").to_string();
    let inner = &c["text_config"];
    let moe = [c, inner].iter().any(|v| MOE_KEYS.iter().any(|k| !v[*k].is_null()));
    if moe || model_type.contains("moe") || model_type.starts_with("llama4") {
        return Err(format!("llama: model_type {:?} is a mixture-of-experts config: dense models only", model_type));
    }
    if is_hybrid(c) {
        return Err(format!("llama: model_type {:?} has linear-attention (hybrid) fields: dense models only", model_type));
    }
    if model_type != "llama" {
        return Err(format!("llama: model_type {:?} is not a Llama dense config (want \"llama\")", model_type));
    }
    if inner.is_object() {
        return Err("llama: config has a text_config (multimodal wrapper), not a plain LlamaForCausalLM: dense models only (text)".into());
    }
    if let Some(a) = c["architectures"].as_array() {
        if !a.is_empty() && !a.iter().any(|x| x.as_str() == Some("LlamaForCausalLM")) {
            return Err(format!("llama: architectures {:?}: want LlamaForCausalLM", a));
        }
    }
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("llama: config missing {}", k));
    let hidden = u("hidden_size")?;
    let heads = u("num_attention_heads")?;
    let layers = u("num_hidden_layers")?;
    if heads == 0 { return Err("llama: num_attention_heads = 0".into()); }
    let head_dim_explicit = c["head_dim"].as_u64().is_some();
    let head_dim = match c["head_dim"].as_u64() {
        Some(v) => v as usize,
        None if hidden % heads == 0 => hidden / heads,
        None => return Err(format!("llama: no head_dim and hidden_size {} is not a multiple of num_attention_heads {}", hidden, heads)),
    };
    if let Some(a) = c["layer_types"].as_array() {
        if let Some((l, t)) = a.iter().enumerate().find(|(_, t)| t.as_str() != Some("full_attention")) {
            return Err(format!("llama: layer_types[{}] = {}: Llama layers are all full attention", l, t));
        }
    }
    let kv_heads = c["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads);
    let mut d = DenseConfig::base(&model_type, Family::Llama, hidden, u("intermediate_size")?, layers, heads, kv_heads,
                                  head_dim, u("vocab_size")?);
    d.kinds = vec![LayerKind::Full; layers];
    d.head_dim_explicit = head_dim_explicit;

    // RoPE: transformers >= 4.56 may write rope_parameters {rope_theta, rope_type, ...} instead of rope_theta /
    // rope_scaling.
    let rp = &c["rope_parameters"];
    d.rope_theta = rp["rope_theta"].as_f64().or(c["rope_theta"].as_f64()).unwrap_or(10000.0);
    let prf = c["partial_rotary_factor"].as_f64().or(rp["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
    d.rope_dims = (prf * head_dim as f64).round() as usize;
    match rope_scaling_kind(&c["rope_scaling"]).or(rope_scaling_kind(rp)) {
        Some(k) => { d.rope_scale = k; d.rope_scaling = None; }
        None => { d.rope_scale = RopeScaling::None; d.rope_scaling = scaling_desc(&c["rope_scaling"]).or(scaling_desc(rp)); }
    }

    let attention_bias = c["attention_bias"].as_bool().unwrap_or(false);
    d.attention_bias = attention_bias;
    d.o_bias = attention_bias;
    d.mlp_bias = c["mlp_bias"].as_bool().unwrap_or(false);
    d.tie_word_embeddings = c["tie_word_embeddings"].as_bool().unwrap_or(false);
    d.qk_norm = false;
    d.eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(2048) as usize;
    d.sliding_window = None;
    d.hidden_act = c["hidden_act"].as_str().unwrap_or("silu").to_string();
    d.activation = Activation::from_hf(&d.hidden_act);

    let mut eos: Vec<u32> = vec![];
    for v in [Some(&c["eos_token_id"]), gen.map(|g| &g["eos_token_id"])].into_iter().flatten() {
        for id in ids(v) { if !eos.contains(&id) { eos.push(id); } }
    }
    d.eos = eos;
    d.bos = c["bos_token_id"].as_u64().or(gen.and_then(|g| g["bos_token_id"].as_u64())).map(|v| v as u32);
    d.quant = quant(c)?;
    Ok(d)
}

/// MLX `quantization` (or `quantization_config`) block: (group_size, bits). Other quantization formats (GPTQ, AWQ,
/// bitsandbytes, ...: a `quant_method` key) are refused by name.
fn quant(c: &Value) -> Result<Option<(usize, usize)>, String> {
    let qb = if c["quantization"].is_object() { &c["quantization"] } else { &c["quantization_config"] };
    if !qb.is_object() { return Ok(None); }
    if let Some(m) = qb["quant_method"].as_str() {
        return Err(format!("llama: quantization_config quant_method {:?} is not MLX quantization (MLX 4-bit group 64 only)", m));
    }
    let g = qb["group_size"].as_u64().ok_or("llama: quantization block without group_size")? as usize;
    let b = qb["bits"].as_u64().ok_or("llama: quantization block without bits")? as usize;
    Ok(Some((g, b)))
}

/// Llama-only features the engine would get silently wrong (not in `DenseConfig`, so `validate()` cannot see them).
fn llama_only(c: &Value) -> Vec<String> {
    let mut b = vec![];
    if c["rope_traditional"].as_bool() == Some(true) {
        b.push("rope_traditional = true: the engine implements rotate-half RoPE only".into());
    }
    b
}

/// `parse` + eos present + Llama-only refusals + `validate()`.
pub fn dense_config_with_gen(c: &Value, gen: Option<&Value>) -> Result<DenseConfig, String> {
    let d = parse(c, gen)?;
    if d.eos.is_empty() { return Err("llama: no eos_token_id in config.json or generation_config.json".into()); }
    let b = llama_only(c);
    if !b.is_empty() { return Err(format!("llama config not supported: {}", b.join("; "))); }
    d.validate().map_err(|e| format!("llama: {}", e))?;
    Ok(d)
}

/// Everything the engine refuses for this config (see `super::blockers`).
pub fn blockers(c: &Value) -> Vec<String> {
    let d = match parse(c, None) {
        Ok(d) => d,
        Err(e) => return vec![e],
    };
    let mut b = llama_only(c);
    b.extend(d.unsupported());
    if d.quant.is_none() {
        b.push("no MLX quantization block: the weight loader takes MLX 4-bit group-64 safetensors only".into());
    }
    b
}
