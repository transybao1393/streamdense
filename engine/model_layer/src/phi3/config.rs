//! Phi-3 `config.json` -> `DenseConfig`. Field meanings and defaults: mlx_lm `models/phi3.py` `ModelArgs`.
//!
//! | config.json key (mlx default) | DenseConfig | note |
//! |---|---|---|
//! | hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, vocab_size, rms_norm_eps (required) | `hidden`, `inter`, `layers`, `heads`, `vocab`, `eps` | |
//! | num_key_value_heads (= heads) | `kv_heads` | |
//! | (none) | `head_dim` = hidden / heads | mlx phi3 never reads `head_dim`; a `head_dim` key that disagrees is refused |
//! | partial_rotary_factor (1.0) | `rope_dims` = int(head_dim * factor) | Phi-4-mini: 0.75 x 128 = 96 |
//! | rope_theta (10000) | `rope_theta` | |
//! | rope_traditional (false) | `rope_traditional` | |
//! | rope_scaling type "longrope" / "su" | `rope_su` = `SuRope::new(long_factor, max_position_embeddings, original_max_position_embeddings, None)` | mlx_lm SuScaledRoPE: ALWAYS the long factors, q / k scaled by mscale (HF uses short_factor up to the original length) |
//! | rope_scaling type "linear" | `rope_scale = Linear { factor }` | |
//! | max_position_embeddings (131072), original_max_position_embeddings (4096) | `max_pos`, SuRope mscale | |
//! | tie_word_embeddings (false) | `tie_word_embeddings` | Phi-4-mini: true |
//! | sliding_window | ignored: all layers full | mlx phi3 never reads it (HF masks: Phi-3-mini-4k 2047) |
//! | (none) | `names.qkv = self_attn.qkv_proj`, `names.gate_up = mlp.gate_up_proj` | fused tensors, no biases (mlx: bias=False everywhere) |
//! | eos_token_id / generation_config.json / tokenizer eos | `eos` | as mlx_lm (`mistral::config::eos_ids`) |
//! | bos_token_id | `bos` | |
//! | quantization / quantization_config | `quant` | |
//!
//! Refused (mlx_lm would fail or silently differ): `attention_bias` / `mlp_bias` / `lm_head_bias` true (mlx phi3 has no
//! bias tensors), rope_scaling without `long_factor` / `type` (mlx raises), rope_scaling types other than
//! longrope / su / linear (mlx drops them with a warning; refused here instead of running a different RoPE), a
//! non-silu `hidden_act` (mlx runs SwiGLU), MoE (`phimoe`) and multimodal (`phi3_v`, `phi4mm`, `text_config`).
use dense_engine::config::{Activation, DenseConfig, Family, LayerKind, RopeScaling, SuRope, TensorNames};
use serde_json::Value;

use crate::mistral::config::{eos_ids, quant};

/// Phi relatives refused with their reason.
pub const REFUSED_TYPES: [(&str, &str); 5] = [
    ("phimoe", "mixture-of-experts: dense models only"),
    ("phi3_v", "multimodal (vision)"),
    ("phi4mm", "multimodal (vision + audio)"),
    ("phi4_multimodal", "multimodal (vision + audio)"),
    ("phi3small", "Phi-3-small: block-sparse attention, a different architecture"),
];

const MOE_KEYS: [&str; 4] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts"];

fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("phi3: config.json has no model_type")?;
    if let Some((_, why)) = REFUSED_TYPES.iter().find(|(t, _)| *t == mt) {
        return Err(format!("phi3: model_type {:?} refused: {}", mt, why));
    }
    if mt != "phi3" { return Err(format!("phi3: model_type {:?} is not \"phi3\"", mt)); }
    if c["text_config"].is_object() || !c["vision_config"].is_null() || !c["embd_layer"].is_null() {
        return Err("phi3: config has a text_config / vision_config / embd_layer (multimodal): dense text models only".into());
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("phi3: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(a) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "Phi3ForCausalLM") {
            return Err(format!("phi3: architecture {:?}: want Phi3ForCausalLM", a));
        }
    }
    for k in ["attention_bias", "mlp_bias", "lm_head_bias"] {
        if c[k].as_bool() == Some(true) {
            return Err(format!("phi3: {} = true: mlx_lm phi3 has no bias tensors", k));
        }
    }
    Ok(())
}

fn floats(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(|x| x.as_f64()).collect()
}

/// Parse a Phi-3 config.json (`gen` = generation_config.json, `tok_eos` = tokenizer eos id).
pub fn parse(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    check_kind(c)?;
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("phi3: config missing {}", k));
    let (hidden, heads, layers) = (u("hidden_size")?, u("num_attention_heads")?, u("num_hidden_layers")?);
    if heads == 0 || hidden % heads != 0 {
        return Err(format!("phi3: hidden_size {} is not a multiple of num_attention_heads {}", hidden, heads));
    }
    let head_dim = hidden / heads;
    if let Some(h) = c["head_dim"].as_u64() {
        if h as usize != head_dim { return Err(format!("phi3: head_dim {} != hidden_size / num_attention_heads {} (mlx phi3 uses the latter)", h, head_dim)); }
    }
    let kv_heads = c["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads);
    let mut d = DenseConfig::base("phi3", Family::Phi3, hidden, u("intermediate_size")?, layers, heads, kv_heads, head_dim,
                                  u("vocab_size")?);
    d.head_dim_explicit = false;
    d.kinds = vec![LayerKind::Full; layers];
    d.sliding_window = None;
    d.eps = c["rms_norm_eps"].as_f64().ok_or("phi3: config missing rms_norm_eps")? as f32;
    d.tie_word_embeddings = c["tie_word_embeddings"].as_bool().unwrap_or(false);
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(131072) as usize;
    let orig = c["original_max_position_embeddings"].as_u64().unwrap_or(4096) as usize;

    let prf = c["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    d.rope_dims = (head_dim as f64 * prf).floor() as usize;
    d.rope_theta = c["rope_theta"].as_f64().unwrap_or(10000.0);
    d.rope_traditional = c["rope_traditional"].as_bool().unwrap_or(false);
    d.rope_scale = RopeScaling::None;
    d.rope_scaling = None;
    let rs = &c["rope_scaling"];
    if rs.as_object().is_some_and(|o| !o.is_empty()) {
        let ty = rs["type"].as_str().ok_or("phi3: rope_scaling without \"type\" (mlx_lm phi3 requires long_factor and type)")?;
        if rs["long_factor"].is_null() {
            return Err("phi3: rope_scaling without \"long_factor\" (mlx_lm phi3 requires long_factor and type)".into());
        }
        match ty {
            "longrope" | "su" => {
                let lf = floats(&rs["long_factor"]).ok_or("phi3: rope_scaling long_factor is not a list of numbers")?;
                if lf.len() * 2 != d.rope_dims {
                    return Err(format!("phi3: rope_scaling long_factor has {} entries, want rope dims / 2 = {}", lf.len(), d.rope_dims / 2));
                }
                d.rope_su = Some(SuRope::new(lf, d.max_pos, orig, None));
            }
            "linear" => {
                let f = rs["factor"].as_f64().ok_or("phi3: rope_scaling linear without a numeric factor")?;
                d.rope_scale = RopeScaling::Linear { factor: f };
            }
            o => return Err(format!("phi3: rope_scaling type {:?}: mlx_lm phi3 takes longrope / su / linear (and drops \
                                     others with a warning): refused", o)),
        }
    } else if !rs.is_null() && !rs.is_object() {
        return Err(format!("phi3: rope_scaling {} is not an object", rs));
    }

    d.attention_bias = false;
    d.o_bias = false;
    d.mlp_bias = false;
    d.qk_norm = false;
    d.hidden_act = c["hidden_act"].as_str().unwrap_or("silu").to_string();
    d.activation = match Activation::from_hf(&d.hidden_act) {
        Activation::Silu => Activation::Silu,
        _ => Activation::Other(format!("{} (mlx_lm phi3 runs SwiGLU whatever hidden_act says)", d.hidden_act)),
    };
    d.names = TensorNames {
        qkv: Some("self_attn.qkv_proj".into()),
        gate_up: Some("mlp.gate_up_proj".into()),
        ..TensorNames::default()
    };
    d.eos = eos_ids(c, gen, tok_eos);
    d.bos = c["bos_token_id"].as_u64().or(gen.and_then(|g| g["bos_token_id"].as_u64())).map(|v| v as u32);
    d.quant = quant(c, "phi3")?;
    Ok(d)
}

/// `parse` + eos present + `validate()`.
pub fn dense_config_with(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    let d = parse(c, gen, tok_eos)?;
    if d.eos.is_empty() { return Err("phi3: no eos_token_id in config.json / generation_config.json / tokenizer_config.json".into()); }
    d.validate().map_err(|e| format!("phi3: {}", e))?;
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
