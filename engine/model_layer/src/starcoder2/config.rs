//! StarCoder2 (`model_type` "starcoder2", `Starcoder2ForCausalLM`: StarCoder2 3B / 7B / 15B) config.json ->
//! `DenseConfig`. Semantics follow mlx_lm `models/starcoder2.py` (the engine's reference).
//!
//! | config.json | DenseConfig | note |
//! |---|---|---|
//! | `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` | sizes | required (mlx_lm has no default) |
//! | `vocab_size` | `vocab` | default 49152 (mlx_lm and HF agree) |
//! | hidden_size / num_attention_heads | `head_dim`, `rope_dims` | computed; must divide; a `head_dim` key that disagrees is refused |
//! | `rope_theta` | `rope_theta` | required: mlx_lm default 100000, HF default 10000, so not guessed; rotate-half RoPE over the full head |
//! | `rope_scaling` | `rope_scaling` (named by `unsupported()`) | mlx_lm ignores it: refused, not silently dropped |
//! | `norm_epsilon` | `eps`, `norm = LayerNorm { bias: true }` | default 1e-5; `norm_type` other than "layer_norm" refused |
//! | `use_bias` | `attention_bias`, `o_bias`, `mlp_bias` = true | mlx_lm always has biases on q / k / v / o / c_fc / c_proj; `use_bias = false` refused |
//! | (architecture) | `mlp = Plain`, `names.up = "mlp.c_fc"`, `names.down = "mlp.c_proj"` | down(gelu(up x)), no gate; `mlp_type` other than "default" refused |
//! | `hidden_act` | `activation = GeluErf`, `hidden_act` (text) | mlx_lm calls `nn.gelu` (exact erf) whatever the config says; the released configs say gelu_pytorch_tanh (HF runs the tanh form) |
//! | `sliding_window` | none: every layer Full | mlx_lm ignores it (HF masks keys older than the window: outputs differ only beyond 4096 tokens) |
//! | `tie_word_embeddings` | `tie_word_embeddings` | default true (mlx_lm and HF) |
//! | `max_position_embeddings` | `max_pos` | informational |
//! | `bos_token_id`, `eos_token_id` (+ generation_config.json) | `bos`, `eos` | merged in order, no duplicates |
//! | `quantization` / `quantization_config` | `quant` | MLX (group_size, bits); other methods refused |
use dense_engine::config::{Activation, Block, DenseConfig, Family, LayerKind, Mlp, NormKind, RopeScaling, TensorNames};
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

fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("starcoder2: config.json has no model_type")?;
    if mt == "gpt_bigcode" || mt == "starcoder" {
        return Err(format!("starcoder2: model_type {:?} (StarCoder 1 / GPT-BigCode: multi-query attention with learned \
                            positions) is not StarCoder2: not supported", mt));
    }
    if mt != "starcoder2" { return Err(format!("starcoder2: model_type {:?} is not StarCoder2 (want \"starcoder2\")", mt)); }
    if c["text_config"].is_object() { return Err("starcoder2: config wraps a text_config: text models only".into()); }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("starcoder2: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(x) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "Starcoder2ForCausalLM") {
            return Err(format!("starcoder2: architecture {:?} refused: want Starcoder2ForCausalLM", x));
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
        return Err(format!("starcoder2: quantization quant_method {:?} is not MLX quantization (MLX 4-bit group 64 only)", m));
    }
    let g = qb["group_size"].as_u64().ok_or("starcoder2: quantization block without group_size")? as usize;
    let b = qb["bits"].as_u64().ok_or("starcoder2: quantization block without bits")? as usize;
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

/// Parse a StarCoder2 config.json (`gen` = generation_config.json for extra eos ids) into a `DenseConfig` plus the
/// blockers that have no `DenseConfig` field. Does not call `validate()`; `eos` may come back empty.
pub fn map_config(c: &Value, gen: Option<&Value>) -> Result<(DenseConfig, Vec<String>), String> {
    check_kind(c)?;
    let req = |k: &str| format!("starcoder2: config.json has no {} (not guessed)", k);
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| req(k));
    let (hidden, inter, layers) = (u("hidden_size")?, u("intermediate_size")?, u("num_hidden_layers")?);
    let (heads, kv_heads) = (u("num_attention_heads")?, u("num_key_value_heads")?);
    let vocab = c["vocab_size"].as_u64().unwrap_or(49152) as usize;
    if heads == 0 || kv_heads == 0 || layers == 0 || hidden % heads != 0 {
        return Err(format!("starcoder2: hidden_size {} must be a non-zero multiple of num_attention_heads {}; layers and \
                            key-value heads must not be 0", hidden, heads));
    }
    let head_dim = hidden / heads;
    if let Some(hd) = c["head_dim"].as_u64() {
        if hd as usize != head_dim {
            return Err(format!("starcoder2: head_dim {} in config.json, but mlx_lm computes hidden_size / num_attention_heads = {}", hd, head_dim));
        }
    }
    let mut d = DenseConfig::base("starcoder2", Family::Starcoder2, hidden, inter, layers, heads, kv_heads, head_dim, vocab);
    d.head_dim_explicit = !c["head_dim"].is_null();
    d.kinds = vec![LayerKind::Full; layers];
    d.sliding_window = None;
    let mut extra = vec![];

    d.rope_theta = c["rope_theta"].as_f64().ok_or("starcoder2: config.json has no rope_theta (mlx_lm default 100000, HF default 10000: not guessed)")?;
    d.rope_dims = head_dim;
    d.rope_traditional = false;
    d.rope_scale = RopeScaling::None;
    d.rope_scaling = scaling_desc(&c["rope_scaling"]).or(scaling_desc(&c["rope_parameters"]))
        .map(|s| format!("{} (mlx_lm starcoder2 ignores rope scaling, HF applies it)", s));

    d.norm = NormKind::LayerNorm { bias: true };
    d.eps = c["norm_epsilon"].as_f64().unwrap_or(1e-5) as f32;
    if let Some(n) = c["norm_type"].as_str() {
        if n != "layer_norm" { extra.push(format!("norm_type {:?}: mlx_lm starcoder2 runs LayerNorm", n)); }
    }
    d.block = Block::Sequential;
    (d.attention_bias, d.o_bias, d.mlp_bias) = (true, true, true);
    if c["use_bias"].as_bool() == Some(false) {
        extra.push("use_bias = false: mlx_lm starcoder2 always reads q / k / v / o / c_fc / c_proj biases".into());
    }
    d.mlp = Mlp::Plain;
    d.names = TensorNames { up: "mlp.c_fc".into(), down: "mlp.c_proj".into(), ..TensorNames::default() };
    if let Some(m) = c["mlp_type"].as_str() {
        if m != "default" { extra.push(format!("mlp_type {:?}: mlx_lm starcoder2 runs the plain c_fc / c_proj MLP", m)); }
    }
    d.activation = Activation::GeluErf;
    d.hidden_act = match c["hidden_act"].as_str() {
        Some("gelu") | None => "gelu (exact erf, mlx_lm nn.gelu)".to_string(),
        Some(h) => format!("gelu (exact erf, mlx_lm nn.gelu; config hidden_act {:?})", h),
    };
    d.tie_word_embeddings = c["tie_word_embeddings"].as_bool().unwrap_or(true);
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(0) as usize;

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
