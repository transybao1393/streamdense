//! config.json -> `dense_engine::config::DenseConfig` for the dense text Gemma generations. Field names follow
//! transformers (`configuration_gemma.py`, `configuration_gemma2.py`, `configuration_gemma3.py`) and mlx_lm
//! (`models/gemma.py`, `gemma2.py`, `gemma3_text.py`).
//!
//! | DenseConfig field | gemma (1) | gemma2 | gemma3_text |
//! |---|---|---|---|
//! | `family` | `Gemma` | `Gemma2` | `Gemma3` |
//! | `activation` | `hidden_activation`, else tanh GELU (see below) | `hidden_activation` (default gelu_pytorch_tanh) | same as gemma2 |
//! | `attn_scale` | `None` (head_dim^-0.5) | `query_pre_attn_scalar`^-0.5 (required) | same as gemma2 |
//! | `attn_softcap` / `final_softcap` | none | `attn_logit_softcapping` / `final_logit_softcapping`: absent = 50 / 30, null = off | absent or null = off |
//! | `norm_plus_one` | true | true | true (incl. q_norm / k_norm) |
//! | `four_norms` | false | true | true |
//! | `qk_norm` | false | false | true |
//! | `embed_scale` | sqrt(hidden_size) | sqrt(hidden_size) | sqrt(hidden_size) |
//! | `kinds` | all full | `layer_types`, else even layers (0, 2, ...) sliding | `layer_types`, else sliding unless (i + 1) % `sliding_window_pattern` == 0 |
//! | `sliding_window` | none | `sliding_window` (required if any sliding layer) | same |
//! | `rope_theta` | `rope_theta` (required) | same | full layers: `rope_parameters.full_attention.rope_theta` or `rope_theta` |
//! | `rope_local_theta` | none | none | `rope_parameters.sliding_attention.rope_theta` or `rope_local_base_freq` (required) |
//! | `rope_scale` | `rope_scaling` (linear / llama3) | same | `rope_scaling` or `rope_parameters.full_attention` (global layers only; Gemma 3 4B+ use linear 8) |
//! | `attention_bias` / `o_bias` | `attention_bias` on q/k/v AND o_proj (modeling_gemma) | same | same |
//! | `tie_word_embeddings` | default true | default true | default true |
//!
//! Embedding scale: HF multiplies by `torch.tensor(hidden_size ** 0.5, dtype=hidden.dtype)` (rounded to bf16), MLX by a
//! Python float against a bf16 array (also bf16). The engine rounds `embed_scale` to bf16 itself (config.rs doc), so the
//! exact sqrt is passed here: e.g. 640 -> 25.298.. (bf16 25.25), 2304 -> 48 (exact).
//!
//! Gemma 1 activation: the original configs say `hidden_act: "gelu"`. transformers PR 29402 (4.39) runs Gemma with
//! `gelu_pytorch_tanh` unless `hidden_activation` is set, because the reference JAX code uses the tanh approximation.
//! This reader follows that. Unverified: whether current mlx_lm `gemma.py` uses exact or approximate GELU.
//!
//! Size fields (and the math scalars whose defaults differ between transformers and mlx_lm, e.g. gemma2
//! `query_pre_attn_scalar` 256 vs 144) must be in the file: a missing one is an error, never a guessed default.
use dense_engine::config::{rope_scaling_kind, Activation, DenseConfig, Family, LayerKind, RopeScaling};
use serde_json::Value;

/// Gemma relatives this layer refuses, with the reason.
pub const REFUSED_TYPES: [(&str, &str); 9] = [
    ("gemma3n", "multimodal, and a different text architecture (per-layer embeddings, AltUp)"),
    ("gemma3n_text", "a different text architecture (per-layer embeddings, AltUp)"),
    ("gemma4", "not supported by the dense engine"),
    ("gemma4_text", "not supported by the dense engine"),
    ("paligemma", "multimodal (vision encoder + Gemma)"),
    ("recurrent_gemma", "recurrent (Griffin) blocks, not a dense attention stack"),
    ("t5gemma", "encoder-decoder"),
    ("vaultgemma", "not supported by the dense engine"),
    ("shieldgemma2", "multimodal classifier"),
];

const MOE_KEYS: [&str; 3] = ["num_experts", "n_routed_experts", "num_experts_per_tok"];
/// Keys that mark a multimodal text config (a gemma3_text config must not carry them).
const MULTIMODAL_KEYS: [&str; 5] = ["vision_config", "audio_config", "mm_tokens_per_image", "image_token_index", "boi_token_index"];

/// Which config object the text model lives in.
pub struct TextView<'a> {
    /// the text config (root, or root.text_config for multimodal `gemma3`)
    pub c: &'a Value,
    pub family: Family,
    pub model_type: &'static str,
    /// root model_type was the multimodal `gemma3` wrapper
    pub multimodal_wrapper: bool,
}

/// Classify a root config: refused relatives, MoE, multimodal shapes. No size fields read yet.
pub fn text_view(root: &Value) -> Result<TextView<'_>, String> {
    let mt = root["model_type"].as_str().ok_or("config: no model_type")?;
    if let Some((_, why)) = REFUSED_TYPES.iter().find(|(t, _)| *t == mt) {
        return Err(format!("config: model_type {:?} refused: {} (the Gemma layer takes gemma, gemma2, gemma3_text, \
                            and text-only usable gemma3)", mt, why));
    }
    let (c, multimodal_wrapper) = match mt {
        "gemma3" => {
            let tc = &root["text_config"];
            if !tc.is_object() {
                return Err("config: multimodal gemma3 without a text_config object refused".into());
            }
            match tc["model_type"].as_str() {
                None | Some("gemma3_text") => {}
                Some(o) => return Err(format!("config: gemma3 text_config model_type {:?} refused (want gemma3_text)", o)),
            }
            (tc, true)
        }
        "gemma" | "gemma2" | "gemma3_text" => (root, false),
        o => return Err(format!("config: model_type {:?} is not gemma, gemma2, gemma3_text or gemma3", o)),
    };
    let (family, model_type) = match mt {
        "gemma" => (Family::Gemma, "gemma"),
        "gemma2" => (Family::Gemma2, "gemma2"),
        _ => (Family::Gemma3, "gemma3_text"),
    };
    if !multimodal_wrapper {
        if let Some(k) = MULTIMODAL_KEYS.iter().chain(["text_config"].iter()).find(|k| !c[**k].is_null()) {
            return Err(format!("config: model_type {:?} with {}: multimodal config refused, text models only", mt, k));
        }
    }
    if let Some(k) = MOE_KEYS.iter().find(|k| !c[**k].is_null() || !root[**k].is_null()) {
        return Err(format!("config: {} present: mixture-of-experts refused, dense models only", k));
    }
    if let Some(a) = root["architectures"].as_array() {
        let ok = |x: &str| x.ends_with("ForCausalLM") || (multimodal_wrapper && x == "Gemma3ForConditionalGeneration");
        if let Some(a) = a.iter().filter_map(|x| x.as_str()).find(|x| !ok(x)) {
            return Err(format!("config: architecture {:?} refused: causal LM text models only", a));
        }
    }
    if c["use_bidirectional_attention"].as_bool() == Some(true) {
        return Err("config: use_bidirectional_attention = true refused: not a causal model (embedding model)".into());
    }
    Ok(TextView { c, family, model_type, multimodal_wrapper })
}

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

/// Key absent: `default`. Explicit null: off. Number: that value.
fn softcap(c: &Value, key: &str, default: Option<f64>) -> Result<Option<f64>, String> {
    match c.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or(format!("config: {} is not a number: {}", key, v)),
    }
}

/// A RoPE block's type other than "default", as text (for `DenseConfig::rope_scaling`, which `unsupported()` names).
fn rope_desc(s: &Value) -> Option<String> {
    if !s.is_object() { return None; }
    let ty = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
    if ty == "default" { return None; }
    Some(match s["factor"].as_f64() { Some(f) => format!("{} factor {}", ty, f), None => ty.to_string() })
}

/// Implemented scaling -> (`rope_scale`, None); other non-default -> (None, Some(description)); none -> (None, None).
fn rope_of(blocks: &[&Value]) -> (RopeScaling, Option<String>) {
    for b in blocks {
        if let Some(k) = rope_scaling_kind(b) { return (k, None); }
        if let Some(d) = rope_desc(b) { return (RopeScaling::None, Some(d)); }
    }
    (RopeScaling::None, None)
}

fn layer_types(c: &Value, layers: usize) -> Result<Option<Vec<LayerKind>>, String> {
    let Some(a) = c["layer_types"].as_array() else { return Ok(None) };
    let mut k = Vec::with_capacity(a.len());
    for (l, t) in a.iter().enumerate() {
        k.push(match t.as_str() {
            Some("full_attention") => LayerKind::Full,
            Some("sliding_attention") => LayerKind::Sliding,
            other => return Err(format!("config: layer_types[{}] unknown: {:?}", l, other)),
        });
    }
    if k.len() != layers {
        return Err(format!("config: layer_types has {} entries, num_hidden_layers = {}", k.len(), layers));
    }
    Ok(Some(k))
}

/// Map a root config.json (+ generation_config.json, + extra stop ids) to a `DenseConfig`. Does not call `validate()`
/// and reads no file. `extra_eos` are appended after the config / generation_config ids (deduplicated).
pub fn map_config(root: &Value, gen: Option<&Value>, extra_eos: &[u32]) -> Result<DenseConfig, String> {
    let tv = text_view(root)?;
    let c = tv.c;
    let u = |k: &str| c[k].as_u64().map(|v| v as usize)
        .ok_or(format!("config: missing {} (size fields are not guessed)", k));
    let (hidden, inter, layers, heads, kv_heads, head_dim, vocab) = (u("hidden_size")?, u("intermediate_size")?,
        u("num_hidden_layers")?, u("num_attention_heads")?, u("num_key_value_heads")?, u("head_dim")?, u("vocab_size")?);
    if hidden == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 || layers == 0 {
        return Err("config: hidden_size, num_hidden_layers, heads, key-value heads and head_dim must not be 0".into());
    }
    let mut d = DenseConfig::base(tv.model_type, tv.family, hidden, inter, layers, heads, kv_heads, head_dim, vocab);

    // activation
    // every generation reads hidden_activation (default gelu_pytorch_tanh). Gemma 1: transformers PR 29402 ignores
    // hidden_act ("gelu" in the original files) unless hidden_activation is set.
    let act = c["hidden_activation"].as_str().unwrap_or("gelu_pytorch_tanh");
    d.hidden_act = match (tv.family, c["hidden_activation"].as_str(), c["hidden_act"].as_str()) {
        (Family::Gemma, None, Some(h)) if h != act => format!("{} (hidden_act {:?}; transformers runs gelu_pytorch_tanh)", act, h),
        _ => act.to_string(),
    };
    d.activation = Activation::from_hf(act);

    // attention scale + soft caps
    if tv.family != Family::Gemma {
        let s = c["query_pre_attn_scalar"].as_f64().ok_or(format!(
            "config: {} without query_pre_attn_scalar (class defaults differ between transformers and mlx_lm; not guessed)",
            tv.model_type))?;
        if !(s.is_finite() && s > 0.0) { return Err(format!("config: query_pre_attn_scalar {} must be > 0", s)); }
        d.attn_scale = Some(s.powf(-0.5));
    }
    (d.attn_softcap, d.final_softcap) = match tv.family {
        Family::Gemma => (None, None),
        Family::Gemma2 => (softcap(c, "attn_logit_softcapping", Some(50.0))?, softcap(c, "final_logit_softcapping", Some(30.0))?),
        _ => (softcap(c, "attn_logit_softcapping", None)?, softcap(c, "final_logit_softcapping", None)?),
    };

    // norms, embedding
    d.norm_plus_one = true;
    d.four_norms = tv.family != Family::Gemma;
    d.qk_norm = tv.family == Family::Gemma3;
    d.embed_scale = (hidden as f64).sqrt();
    d.eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    d.max_pos = c["max_position_embeddings"].as_u64().unwrap_or(0) as usize;

    // layer kinds
    d.kinds = match tv.family {
        Family::Gemma => vec![LayerKind::Full; layers],
        Family::Gemma2 => match layer_types(c, layers)? {
            Some(k) => k,
            // transformers Gemma2: is_sliding = not bool(layer_idx % 2) -> layer 0 sliding
            None => (0..layers).map(|i| if i % 2 == 0 { LayerKind::Sliding } else { LayerKind::Full }).collect(),
        },
        _ => match layer_types(c, layers)? {
            Some(k) => k,
            None => {
                let p = c["sliding_window_pattern"].as_u64().or(c["_sliding_window_pattern"].as_u64())
                    .ok_or("config: gemma3_text without layer_types or sliding_window_pattern (not guessed)")? as usize;
                if p == 0 { return Err("config: sliding_window_pattern = 0".into()); }
                (0..layers).map(|i| if (i + 1) % p != 0 { LayerKind::Sliding } else { LayerKind::Full }).collect()
            }
        },
    };
    if d.kinds.contains(&LayerKind::Sliding) {
        let w = c["sliding_window"].as_u64().ok_or("config: sliding layers but no sliding_window (not guessed)")?;
        d.sliding_window = Some(w as usize);
    }

    // RoPE
    let rp = &c["rope_parameters"];
    if tv.family == Family::Gemma3 {
        let (full, slide) = (&rp["full_attention"], &rp["sliding_attention"]);
        d.rope_theta = full["rope_theta"].as_f64().or(c["rope_theta"].as_f64())
            .ok_or("config: gemma3_text without rope_theta (not guessed)")?;
        let local = slide["rope_theta"].as_f64().or(c["rope_local_base_freq"].as_f64())
            .ok_or("config: gemma3_text without rope_local_base_freq (not guessed)")?;
        d.rope_local_theta = Some(local);
        let (scale, other) = rope_of(&[&c["rope_scaling"], full]);
        d.rope_scale = scale;
        // the local table is plain RoPE in the engine: any scaling there is named as unsupported
        let local_other = rope_desc(slide).map(|s| format!("{} on sliding layers", s));
        d.rope_scaling = match (other, local_other) {
            (None, None) => None,
            (a, b) => Some([a, b].into_iter().flatten().collect::<Vec<_>>().join(", ")),
        };
    } else {
        d.rope_theta = rp["rope_theta"].as_f64().or(c["rope_theta"].as_f64())
            .ok_or(format!("config: {} without rope_theta (not guessed)", tv.model_type))?;
        let (scale, other) = rope_of(&[&c["rope_scaling"], rp]);
        d.rope_scale = scale;
        d.rope_scaling = other;
    }

    // bias, tie
    let bias = c["attention_bias"].as_bool().unwrap_or(false);
    d.attention_bias = bias;
    d.o_bias = bias;
    d.mlp_bias = false;
    d.tie_word_embeddings = c["tie_word_embeddings"].as_bool().or(root["tie_word_embeddings"].as_bool()).unwrap_or(true);

    // bos / eos
    d.bos = c["bos_token_id"].as_u64().or(root["bos_token_id"].as_u64()).map(|v| v as u32);
    let mut eos: Vec<u32> = vec![];
    let srcs = [Some(&root["eos_token_id"]), Some(&c["eos_token_id"]), gen.map(|g| &g["eos_token_id"])];
    for id in srcs.into_iter().flatten().flat_map(ids).chain(extra_eos.iter().copied()) {
        if !eos.contains(&id) { eos.push(id); }
    }
    // empty eos is refused by `dense_config` (generation_config.json may hold it), not here
    d.eos = eos;

    // MLX quantization (root for the multimodal wrapper too)
    let qb = [&root["quantization"], &root["quantization_config"], &c["quantization"], &c["quantization_config"]]
        .into_iter().find(|v| v.is_object());
    d.quant = match qb {
        None => None,
        Some(q) => match (q["group_size"].as_u64(), q["bits"].as_u64()) {
            (Some(g), Some(b)) => Some((g as usize, b as usize)),
            _ => return Err(format!("config: quantization block is not MLX (group_size + bits): {}", q)),
        },
    };
    Ok(d)
}
