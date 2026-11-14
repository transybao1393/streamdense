//! Cohere2 (`model_type` "cohere2", `Cohere2ForCausalLM`: Command R7B 12-2024) config.json -> `DenseConfig`.
//! Semantics follow mlx_lm `models/cohere2.py` (the engine's reference).
//!
//! | config.json | DenseConfig | note |
//! |---|---|---|
//! | `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size` | sizes | required here (mlx_lm has Command-R7B-sized defaults; sizes are not guessed); head_dim * heads must equal hidden_size (mlx_lm raises) |
//! | `rope_theta` | `rope_theta` | required: mlx_lm default 50000, HF Cohere2Config default 10000, so not guessed |
//! | (architecture) | `rope_traditional = true`, `rope_dims = head_dim` | interleaved RoPE over the full head (`position_embedding_type` "rope_gptj"); `rotary_pct` / `partial_rotary_factor` other than 1 refused |
//! | `sliding_window_pattern` (mlx default 4), `sliding_window` (default 4096) | `kinds`, `sliding_window`, `no_rope_layers` | layer i is Full iff i % pattern == pattern - 1, else Sliding; Full layers apply NO RoPE (mlx_lm: rope only when sliding). A `layer_types` list that disagrees is refused |
//! | `layer_norm_eps` (default 1e-5), `layer_norm_bias` (default false) | `eps`, `norm = LayerNorm { bias }` | input norm and final norm |
//! | (architecture) | `block = ParallelBranchesFirst` | x = norm(h); h = attn(x) + mlp(x) + h (`use_parallel_block`: false refused) |
//! | `attention_bias` (default false) | `attention_bias` AND `o_bias` | q / k / v / o |
//! | `logit_scale` (default 0.0625) | `logits_scaling = 1 / logit_scale` | logits * logit_scale (exact for the powers of two in use, 0.25) |
//! | `tie_word_embeddings` | `tie_word_embeddings = true` | mlx_lm always uses embed_tokens as the head; `false` refused |
//! | `hidden_act` / `use_gated_activation` | `activation = Silu`, `mlp = Gated` | mlx_lm always SwiGLU; other values refused |
//! | `max_position_embeddings` | `max_pos` | informational |
//! | `bos_token_id`, `eos_token_id` (+ generation_config.json) | `bos`, `eos` | merged in order, no duplicates |
//! | `quantization` / `quantization_config` | `quant` | MLX (group_size, bits); other methods refused |
//!
//! Attention scale head_dim^-0.5. Sliding layers see the last `sliding_window` keys (mlx_lm RotatingKVCache + window
//! mask).
use dense_engine::config::{Activation, Block, DenseConfig, Family, LayerKind, Mlp, NormKind, RopeScaling};
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
    let mt = c["model_type"].as_str().ok_or("cohere2: config.json has no model_type")?;
    if mt == "cohere" {
        return Err("cohere2: model_type \"cohere\" (Command-R v1, all-global attention, different template) is not cohere2: not supported yet".into());
    }
    if mt != "cohere2" { return Err(format!("cohere2: model_type {:?} is not Cohere2 (want \"cohere2\")", mt)); }
    if c["text_config"].is_object() || c["vision_config"].is_object() {
        return Err("cohere2: config wraps a text_config / vision_config (Aya Vision etc.): text models only".into());
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null()) {
        return Err(format!("cohere2: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = c["architectures"].as_array() {
        if let Some(x) = a.iter().filter_map(|x| x.as_str()).find(|x| *x != "Cohere2ForCausalLM") {
            return Err(format!("cohere2: architecture {:?} refused: want Cohere2ForCausalLM", x));
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
        return Err(format!("cohere2: quantization quant_method {:?} is not MLX quantization (MLX 4-bit group 64 only)", m));
    }
    let g = qb["group_size"].as_u64().ok_or("cohere2: quantization block without group_size")? as usize;
    let b = qb["bits"].as_u64().ok_or("cohere2: quantization block without bits")? as usize;
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

/// Parse a Cohere2 config.json (`gen` = generation_config.json for extra eos ids) into a `DenseConfig` plus the
/// blockers that have no `DenseConfig` field. Does not call `validate()`; `eos` may come back empty.
pub fn map_config(c: &Value, gen: Option<&Value>) -> Result<(DenseConfig, Vec<String>), String> {
    check_kind(c)?;
    let req = |k: &str| format!("cohere2: config.json has no {} (size / math field; not guessed)", k);
    let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| req(k));
    let (hidden, inter, layers) = (u("hidden_size")?, u("intermediate_size")?, u("num_hidden_layers")?);
    let (heads, kv_heads, head_dim, vocab) = (u("num_attention_heads")?, u("num_key_value_heads")?, u("head_dim")?, u("vocab_size")?);
    if heads == 0 || kv_heads == 0 || layers == 0 || head_dim * heads != hidden {
        return Err(format!("cohere2: head_dim {} x num_attention_heads {} must equal hidden_size {} (mlx_lm raises \
                            otherwise); layers and key-value heads must not be 0", head_dim, heads, hidden));
    }
    let mut d = DenseConfig::base("cohere2", Family::Cohere2, hidden, inter, layers, heads, kv_heads, head_dim, vocab);
    d.head_dim_explicit = true;
    let mut extra = vec![];

    // layer kinds: global (full attention, no RoPE) every `pattern`-th layer, the others sliding with RoPE
    let pattern = c["sliding_window_pattern"].as_u64().unwrap_or(4) as usize;
    if pattern == 0 { return Err("cohere2: sliding_window_pattern = 0".into()); }
    d.kinds = (0..layers).map(|i| if i % pattern == pattern - 1 { LayerKind::Full } else { LayerKind::Sliding }).collect();
    if let Some(lt) = c["layer_types"].as_array() {
        let want: Vec<&str> = d.kinds.iter().map(|k| if *k == LayerKind::Full { "full_attention" } else { "sliding_attention" }).collect();
        let got: Vec<&str> = lt.iter().map(|t| t.as_str().unwrap_or("?")).collect();
        if got != want {
            return Err(format!("cohere2: layer_types disagrees with sliding_window_pattern {} (mlx_lm uses the pattern): {:?}", pattern, got));
        }
    }
    d.sliding_window = if d.kinds.contains(&LayerKind::Sliding) {
        Some(c["sliding_window"].as_u64().unwrap_or(4096) as usize)
    } else {
        None
    };
    d.no_rope_layers = d.kinds.iter().map(|k| *k == LayerKind::Full).collect();

    // RoPE: interleaved, full head
    let rp = &c["rope_parameters"];
    d.rope_theta = match (c["rope_theta"].as_f64(), rp["rope_theta"].as_f64()) {
        (Some(a), Some(b)) if a != b => return Err(format!("cohere2: rope_theta {} and rope_parameters.rope_theta {} differ", a, b)),
        (Some(a), _) | (None, Some(a)) => a,
        (None, None) => return Err("cohere2: config.json has no rope_theta (mlx_lm default 50000, HF default 10000: not guessed)".into()),
    };
    d.rope_traditional = true;
    d.rope_dims = head_dim;
    for k in ["rotary_pct", "partial_rotary_factor"] {
        if let Some(f) = c[k].as_f64().or(rp[k].as_f64()) {
            if f != 1.0 { extra.push(format!("{} {}: mlx_lm cohere2 rotates the full head dim", k, f)); }
        }
    }
    if let Some(p) = c["position_embedding_type"].as_str() {
        if p != "rope_gptj" { extra.push(format!("position_embedding_type {:?}: mlx_lm cohere2 runs interleaved (gptj) RoPE", p)); }
    }
    d.rope_scale = RopeScaling::None;
    d.rope_scaling = scaling_desc(&c["rope_scaling"]).or(scaling_desc(rp))
        .map(|s| format!("{} (mlx_lm cohere2 ignores rope scaling, HF applies it)", s));

    // block, norms, FFN, head
    d.block = Block::ParallelBranchesFirst;
    d.norm = NormKind::LayerNorm { bias: c["layer_norm_bias"].as_bool().unwrap_or(false) };
    d.eps = c["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
    let bias = c["attention_bias"].as_bool().unwrap_or(false);
    (d.attention_bias, d.o_bias, d.mlp_bias) = (bias, bias, false);
    d.mlp = Mlp::Gated;
    d.activation = Activation::Silu;
    d.hidden_act = c["hidden_act"].as_str().unwrap_or("silu").to_string();
    if d.hidden_act != "silu" && d.hidden_act != "swish" {
        extra.push(format!("hidden_act {:?}: mlx_lm cohere2 always runs SwiGLU (silu)", d.hidden_act));
    }
    for k in ["use_gated_activation", "use_parallel_block"] {
        if c[k].as_bool() == Some(false) { extra.push(format!("{} = false: mlx_lm cohere2 always runs it", k)); }
    }
    let logit_scale = c["logit_scale"].as_f64().unwrap_or(0.0625);
    if !(logit_scale.is_finite() && logit_scale != 0.0) { return Err(format!("cohere2: logit_scale {} must be finite and != 0", logit_scale)); }
    d.logits_scaling = 1.0 / logit_scale;
    d.tie_word_embeddings = true;
    if c["tie_word_embeddings"].as_bool() == Some(false) {
        extra.push("tie_word_embeddings = false: mlx_lm cohere2 always uses embed_tokens as the output head".into());
    }
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
