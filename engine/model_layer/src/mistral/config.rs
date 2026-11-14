//! Mistral `config.json` -> `DenseConfig`. mlx_lm runs `model_type` "mistral" through `models/llama.py`
//! (`mlx_lm/utils.py` MODEL_REMAPPING "mistral" -> "llama"), so field meanings and defaults are mlx_lm llama
//! `ModelArgs`:
//!
//! | config.json key (mlx default) | DenseConfig | note |
//! |---|---|---|
//! | hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, vocab_size, rms_norm_eps (required) | `hidden`, `inter`, `layers`, `heads`, `vocab`, `eps` | |
//! | num_key_value_heads (= heads) | `kv_heads` | |
//! | head_dim (hidden / heads) | `head_dim`, `rope_dims` (full rotary), `head_dim_explicit` | |
//! | rope_theta (10000) | `rope_theta` | a `rope_parameters` object without `rope_theta` is refused: mlx_lm llama does not read it |
//! | rope_traditional (false) | `rope_traditional` | |
//! | rope_scaling (none) | `rope_scale` linear / llama3; yarn, dynamic, longrope -> `rope_scaling` (named by `unsupported()`) | |
//! | attention_bias (false) | `attention_bias` AND `o_bias` | mlx llama: bias on q, k, v and o |
//! | mlp_bias (false) | `mlp_bias` | |
//! | tie_word_embeddings (TRUE in mlx llama) | `tie_word_embeddings` | every fetched Mistral config says false |
//! | layer_types (all full) | `kinds` | "sliding_attention" -> `LayerKind::Sliding` with `sliding_window` (required then) |
//! | sliding_window | `sliding_window` only when `layer_types` has sliding layers | see below |
//! | max_position_embeddings | `max_pos` | (32768 when absent, informational) |
//! | hidden_act | `activation` | mlx llama always runs SwiGLU; a non-silu value is refused |
//! | eos_token_id / generation_config.json eos_token_id / tokenizer_config.json eos_token | `eos` | mlx_lm: generation_config eos replaces config eos; the tokenizer's eos is added |
//! | bos_token_id | `bos` | |
//! | quantization / quantization_config | `quant` | other `quant_method` refused |
//!
//! `sliding_window` without `layer_types` (Mistral-7B v0.1: 4096 on every layer): mlx_lm ignores it (all layers full
//! attention), HF masks keys older than the window. This follows mlx_lm (the engine's reference); the two agree for
//! contexts up to the window. v0.3, Nemo and Ministral-8B configs have `sliding_window: null`.
//!
//! Refused by name: MoE (`num_local_experts`, Mixtral is model_type "mixtral" anyway), multimodal wrappers
//! (`text_config`; model_type "mistral3" / "ministral3" / "pixtral" / "llava" are refused before any size is read).
use dense_engine::config::{rope_scaling_kind, Activation, DenseConfig, Family, LayerKind, RopeScaling};
use serde_json::Value;

/// Mistral relatives refused with their reason.
pub const REFUSED_TYPES: [(&str, &str); 5] = [
    ("mistral3", "multimodal (Pixtral vision encoder + Mistral / Ministral 3 text, YaRN RoPE)"),
    ("ministral3", "Ministral 3 text model (YaRN RoPE, llama_4_scaling): not supported by the dense engine"),
    ("pixtral", "vision encoder"),
    ("llava", "multimodal"),
    ("mixtral", "mixture-of-experts: dense models only"),
];

const MOE_KEYS: [&str; 4] = ["num_experts", "n_routed_experts", "num_experts_per_tok", "num_local_experts"];

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

/// Readable description of a `rope_scaling` object the engine does not implement; `None` for absent / default.
fn scaling_desc(s: &Value) -> Option<String> {
    if !s.is_object() { return None; }
    let ty = s["type"].as_str().or(s["rope_type"].as_str()).unwrap_or("default");
    if ty == "default" { return None; }
    let mut d = ty.to_string();
    for k in ["factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"] {
        if let Some(f) = s[k].as_f64() { d += &format!(" {} {}", k, f); }
    }
    Some(d)
}

/// MLX `quantization` (or `quantization_config`) block: (group_size, bits).
pub(crate) fn quant(c: &Value, who: &str) -> Result<Option<(usize, usize)>, String> {
    let qb = if c["quantization"].is_object() { &c["quantization"] } else { &c["quantization_config"] };
    if !qb.is_object() { return Ok(None); }
    if let Some(m) = qb["quant_method"].as_str() {
        return Err(format!("{}: quantization_config quant_method {:?} is not MLX quantization (MLX 4-bit group 64 only)", who, m));
    }
    if let Some(m) = qb["mode"].as_str() {
        if m != "affine" { return Err(format!("{}: MLX quantization mode {:?} (affine only)", who, m)); }
    }
    let g = qb["group_size"].as_u64().ok_or(format!("{}: quantization block without group_size", who))? as usize;
    let b = qb["bits"].as_u64().ok_or(format!("{}: quantization block without bits", who))? as usize;
    Ok(Some((g, b)))
}

/// eos ids as mlx_lm builds them: generation_config.json `eos_token_id` when present, else config.json's; plus the
/// tokenizer's eos id (`tok_eos`). In order, no duplicates.
pub(crate) fn eos_ids(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Vec<u32> {
    let g = gen.map(|g| ids(&g["eos_token_id"])).unwrap_or_default();
    let mut eos = if g.is_empty() { ids(&c["eos_token_id"]) } else { g };
    if let Some(t) = tok_eos { if !eos.contains(&t) { eos.push(t); } }
    eos
}

/// Classify the root config: refused relatives, MoE, wrappers. Nothing else read.
fn check_kind(c: &Value) -> Result<(), String> {
    let mt = c["model_type"].as_str().ok_or("mistral: config.json has no model_type")?;
    if let Some((_, why)) = REFUSED_TYPES.iter().find(|(t, _)| *t == mt) {
        return Err(format!("mistral: model_type {:?} refused: {}", mt, why));
    }
    if mt != "mistral" { return Err(format!("mistral: model_type {:?} is not \"mistral\"", mt)); }
    if c["text_config"].is_object() || !c["vision_config"].is_null() {
        return Err("mistral: config has a text_config / vision_config (multimodal wrapper): dense text models only".into());
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("mistral: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(a) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "MistralForCausalLM") {
            return Err(format!("mistral: architecture {:?}: want MistralForCausalLM", a));
        }
    }
    Ok(())
}

/// Parse a Mistral config.json (`gen` = generation_config.json, `tok_eos` = tokenizer eos id). Errors for configs that
/// are not a dense Mistral or miss a required field; everything the engine cannot run is left to `unsupported()`.
pub fn parse(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    check_kind(c)?;
    llama_like(c, "mistral", Family::Mistral, true, gen, tok_eos)
}

/// The mlx_lm llama `ModelArgs` reading (table in the module doc), shared with the families whose mlx_lm module builds
/// on it (SmolLM3: `smollm3.py` subclasses llama `ModelArgs`; OLMo 2: the same fields in `olmo2.py`). `who` prefixes
/// errors; `layer_types`: read `layer_types` / `sliding_window` (mlx llama) or keep every layer full (mlx olmo2 has
/// neither). Model kind checks are the caller's.
pub(crate) fn llama_like(c: &Value, who: &str, family: Family, layer_types: bool, gen: Option<&Value>,
                         tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("{}: config missing {}", who, k));
    let (hidden, heads, layers) = (u("hidden_size")?, u("num_attention_heads")?, u("num_hidden_layers")?);
    if heads == 0 { return Err(format!("{}: num_attention_heads = 0", who)); }
    let head_dim_explicit = c["head_dim"].as_u64().is_some();
    let head_dim = match c["head_dim"].as_u64() {
        Some(v) => v as usize,
        None if hidden % heads == 0 => hidden / heads,
        None => return Err(format!("{}: no head_dim and hidden_size {} is not a multiple of num_attention_heads {}", who, hidden, heads)),
    };
    let kv_heads = c["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads);
    let mut d = DenseConfig::base(who, family, hidden, u("intermediate_size")?, layers, heads, kv_heads,
                                  head_dim, u("vocab_size")?);
    d.head_dim_explicit = head_dim_explicit;
    d.eps = c["rms_norm_eps"].as_f64().ok_or(format!("{}: config missing rms_norm_eps", who))? as f32;

    // layers: mlx llama reads layer_types (default all full); sliding_window only applies to sliding layers
    d.kinds = vec![LayerKind::Full; layers];
    if let Some(a) = c["layer_types"].as_array().filter(|_| layer_types) {
        if a.len() != layers { return Err(format!("{}: layer_types has {} entries, num_hidden_layers = {}", who, a.len(), layers)); }
        for (i, t) in a.iter().enumerate() {
            d.kinds[i] = match t.as_str() {
                Some("full_attention") => LayerKind::Full,
                Some("sliding_attention") => LayerKind::Sliding,
                _ => return Err(format!("{}: layer_types[{}] = {}: want full_attention / sliding_attention", who, i, t)),
            };
        }
    }
    d.sliding_window = if d.n_sliding() > 0 { c["sliding_window"].as_u64().map(|v| v as usize) } else { None };

    // RoPE
    if c["rope_theta"].is_null() && c["rope_parameters"].is_object() {
        return Err(format!("{}: rope_parameters without rope_theta: mlx_lm reads rope_theta / rope_scaling only", who));
    }
    d.rope_theta = c["rope_theta"].as_f64().unwrap_or(10000.0);
    d.rope_dims = head_dim;
    d.rope_traditional = c["rope_traditional"].as_bool().unwrap_or(false);
    match rope_scaling_kind(&c["rope_scaling"]) {
        Some(k) => { d.rope_scale = k; d.rope_scaling = None; }
        None => { d.rope_scale = RopeScaling::None; d.rope_scaling = scaling_desc(&c["rope_scaling"]); }
    }

    let attention_bias = c["attention_bias"].as_bool().unwrap_or(false);
    d.attention_bias = attention_bias;
    d.o_bias = attention_bias;
    d.mlp_bias = c["mlp_bias"].as_bool().unwrap_or(false);
    d.tie_word_embeddings = c["tie_word_embeddings"].as_bool().unwrap_or(true);
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(32768) as usize;
    d.hidden_act = c["hidden_act"].as_str().unwrap_or("silu").to_string();
    // mlx llama always computes swiglu: a different hidden_act would run as silu there
    d.activation = match Activation::from_hf(&d.hidden_act) {
        Activation::Silu => Activation::Silu,
        _ => Activation::Other(format!("{} (mlx_lm runs SwiGLU whatever hidden_act says)", d.hidden_act)),
    };
    d.eos = eos_ids(c, gen, tok_eos);
    d.bos = c["bos_token_id"].as_u64().or(gen.and_then(|g| g["bos_token_id"].as_u64())).map(|v| v as u32);
    d.quant = quant(c, who)?;
    Ok(d)
}

/// `parse` + eos present + `validate()`.
pub fn dense_config_with(c: &Value, gen: Option<&Value>, tok_eos: Option<u32>) -> Result<DenseConfig, String> {
    let d = parse(c, gen, tok_eos)?;
    if d.eos.is_empty() { return Err("mistral: no eos_token_id in config.json / generation_config.json / tokenizer_config.json".into()); }
    d.validate().map_err(|e| format!("mistral: {}", e))?;
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
