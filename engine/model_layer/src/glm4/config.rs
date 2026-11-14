//! GLM-4 (0414 generation, `model_type` "glm4", `Glm4ForCausalLM`) config.json -> `DenseConfig`. Semantics follow
//! mlx_lm `models/glm4.py` (the engine's reference); keys it requires are required here, its defaults are used for the
//! rest.
//!
//! | config.json | DenseConfig | note |
//! |---|---|---|
//! | `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size` | sizes | required (mlx_lm ModelArgs has no default) |
//! | `partial_rotary_factor` | `rope_dims = int(head_dim * f)` | required; 0.5 in the released models (64 of 128 dims) |
//! | `rope_theta` (or `rope_parameters.rope_theta`) | `rope_theta` | required |
//! | `rope_traditional` | `rope_traditional` | mlx_lm default TRUE (interleaved pairs, as HF GLM's rotate_half on x[..., 0::2] / 1::2) |
//! | `rope_scaling` / `rope_parameters` type | `rope_scaling` (named by `unsupported()`) | mlx_lm glm4 ignores it, HF applies it: refused, not silently dropped |
//! | `attention_bias` | `attention_bias` (q / k / v only) | required; o_proj never has a bias (`o_bias = false`) |
//! | `rms_norm_eps` | `eps` | required; RMSNorm (`norm = Rms`) |
//! | (architecture) | `four_norms = true`, `names.post_attn_norm = "post_self_attn_layernorm"`, `names.pre_ffn_norm = "post_attention_layernorm"`, `names.post_ffn_norm = "post_mlp_layernorm"` | h += post_self_attn_norm(attn(input_norm(h))); h += post_mlp_norm(mlp(post_attention_norm(h))) |
//! | (architecture) | `names.gate_up = Some("mlp.gate_up_proj")`, `mlp = Gated`, `activation = Silu` | fused gate / up rows, first `intermediate_size` rows = gate (mlx `mx.split(x, 2)`) |
//! | `hidden_act` | `hidden_act` (text) | mlx_lm always runs SwiGLU; a config that says otherwise is refused |
//! | `tie_word_embeddings` | `tie_word_embeddings = false` | mlx_lm glm4 always has its own lm_head; `true` is refused |
//! | `max_position_embeddings` | `max_pos` | mlx_lm default 32768 |
//! | `eos_token_id` (+ generation_config.json) | `eos` | merged in order, no duplicates (0414: <\|endoftext\|>, <\|user\|>, <\|observation\|>) |
//! | `quantization` / `quantization_config` | `quant` | MLX (group_size, bits); other methods refused |
//!
//! Every layer is full attention (no sliding window in GLM-4); attention scale head_dim^-0.5.
use dense_engine::config::{Activation, DenseConfig, Family, LayerKind, RopeScaling, TensorNames};
use serde_json::Value;

/// Relatives of GLM-4 that are not this dense architecture, with the reason.
pub const REFUSED_TYPES: [(&str, &str); 5] = [
    ("glm4_moe", "mixture-of-experts (GLM-4.5 / 4.6): dense models only"),
    ("glm4v", "multimodal (vision encoder + GLM-4)"),
    ("glm4v_moe", "multimodal mixture-of-experts"),
    ("glm", "the earlier HF GLM port (glm-4-9b-chat-hf): a different layer layout (no post-attention / post-mlp norms)"),
    ("chatglm", "ChatGLM remote-code checkpoints (glm-4-9b-chat): different tensor names and a custom tokenizer"),
];

const MOE_KEYS: [&str; 5] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts", "moe_intermediate_size"];

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

/// Refuse what is not a dense GLM-4 text model.
fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("glm4: config.json has no model_type")?;
    if let Some((_, why)) = REFUSED_TYPES.iter().find(|(t, _)| *t == mt) {
        return Err(format!("glm4: model_type {:?} refused: {}", mt, why));
    }
    if mt != "glm4" { return Err(format!("glm4: model_type {:?} is not GLM-4 (want \"glm4\")", mt)); }
    if c["text_config"].is_object() || c["vision_config"].is_object() {
        return Err("glm4: config wraps a text_config / vision_config (multimodal): text models only".into());
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("glm4: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(x) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "Glm4ForCausalLM") {
            return Err(format!("glm4: architecture {:?} refused: want Glm4ForCausalLM", x));
        }
    }
    Ok(())
}

/// MLX quantization block -> (group_size, bits); other quantization formats refused by name. Extra blockers
/// (non-affine mode, per-layer entries) are pushed to `extra`.
fn quant(c: &Value, tag: &str, extra: &mut Vec<String>) -> Result<Option<(usize, usize)>, String> {
    let qb = if c["quantization"].is_object() { &c["quantization"] } else { &c["quantization_config"] };
    let Some(o) = qb.as_object() else {
        extra.push("no MLX quantization block: the weight loader reads MLX 4-bit group-64 models only".into());
        return Ok(None);
    };
    if let Some(m) = qb["quant_method"].as_str() {
        return Err(format!("{}: quantization quant_method {:?} is not MLX quantization (MLX 4-bit group 64 only)", tag, m));
    }
    let g = qb["group_size"].as_u64().ok_or(format!("{}: quantization block without group_size", tag))? as usize;
    let b = qb["bits"].as_u64().ok_or(format!("{}: quantization block without bits", tag))? as usize;
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

/// Parse a GLM-4 config.json (`gen` = generation_config.json for extra eos ids) into a `DenseConfig` plus the blockers
/// that have no `DenseConfig` field. Does not call `validate()`; `eos` may come back empty (checked by the caller).
pub fn map_config(c: &Value, gen: Option<&Value>) -> Result<(DenseConfig, Vec<String>), String> {
    check_kind(c)?;
    let req = |k: &str| format!("glm4: config.json has no {} (required by mlx_lm glm4; not guessed)", k);
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| req(k));
    let (hidden, inter, layers) = (u("hidden_size")?, u("intermediate_size")?, u("num_hidden_layers")?);
    let (heads, kv_heads, head_dim, vocab) = (u("num_attention_heads")?, u("num_key_value_heads")?, u("head_dim")?, u("vocab_size")?);
    if hidden == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 || layers == 0 {
        return Err("glm4: hidden_size, num_hidden_layers, heads, key-value heads and head_dim must not be 0".into());
    }
    let mut d = DenseConfig::base("glm4", Family::Glm4, hidden, inter, layers, heads, kv_heads, head_dim, vocab);
    d.head_dim_explicit = true;
    d.kinds = vec![LayerKind::Full; layers];
    let mut extra = vec![];

    // RoPE: partial, interleaved by default
    let rp = &c["rope_parameters"];
    let pick = |k: &str| -> Result<f64, String> {
        match (c[k].as_f64(), rp[k].as_f64()) {
            (Some(a), Some(b)) if a != b => Err(format!("glm4: {} {} and rope_parameters.{} {} differ", k, a, k, b)),
            (Some(a), _) | (None, Some(a)) => Ok(a),
            (None, None) => Err(req(k)),
        }
    };
    d.rope_theta = pick("rope_theta")?;
    let prf = pick("partial_rotary_factor")?;
    d.rope_dims = (head_dim as f64 * prf) as usize; // mlx: int(head_dim * partial_rotary_factor)
    d.rope_traditional = c["rope_traditional"].as_bool().unwrap_or(true);
    d.rope_scale = RopeScaling::None;
    d.rope_scaling = scaling_desc(&c["rope_scaling"]).or(scaling_desc(rp))
        .map(|s| format!("{} (mlx_lm glm4 ignores rope scaling, HF applies it)", s));

    // attention, norms, FFN
    d.attention_bias = c["attention_bias"].as_bool().ok_or_else(|| req("attention_bias"))?;
    d.o_bias = false;
    d.mlp_bias = false;
    d.eps = c["rms_norm_eps"].as_f64().ok_or_else(|| req("rms_norm_eps"))? as f32;
    d.four_norms = true;
    d.names = TensorNames {
        post_attn_norm: "post_self_attn_layernorm".into(),
        pre_ffn_norm: "post_attention_layernorm".into(),
        post_ffn_norm: "post_mlp_layernorm".into(),
        gate_up: Some("mlp.gate_up_proj".into()),
        ..TensorNames::default()
    };
    d.activation = Activation::Silu;
    d.hidden_act = c["hidden_act"].as_str().unwrap_or("silu").to_string();
    if d.hidden_act != "silu" && d.hidden_act != "swish" {
        extra.push(format!("hidden_act {:?}: mlx_lm glm4 always runs SwiGLU (silu), HF would use {:?}", d.hidden_act, d.hidden_act));
    }
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(32768) as usize;

    // embeddings
    d.tie_word_embeddings = false;
    if c["tie_word_embeddings"].as_bool() == Some(true) {
        extra.push("tie_word_embeddings = true: mlx_lm glm4 always reads its own lm_head".into());
    }

    // tokens
    d.bos = c["bos_token_id"].as_u64().map(|v| v as u32);
    for v in [Some(&c["eos_token_id"]), gen.map(|g| &g["eos_token_id"])].into_iter().flatten() {
        for id in ids(v) { if !d.eos.contains(&id) { d.eos.push(id); } }
    }
    d.quant = quant(c, "glm4", &mut extra)?;
    Ok((d, extra))
}

/// Everything that stops this config from running (config.json alone; eos from generation_config.json is checked by
/// `dense_config`). Empty = runs.
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
