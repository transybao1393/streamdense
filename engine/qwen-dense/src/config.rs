//! Config reader + validator for ordinary Qwen dense models (Qwen2 / Qwen2.5 / Qwen3 dense): every layer is full
//! attention, GQA, no linear-attention (DeltaNet) fields. Parsing is separate from `validate()`, as in
//! streamdense-model's `QwenConfig`: a config parses whenever its fields are readable, and `validate()` names every
//! feature the dense path does not support yet (head dim, partial rotary, RoPE scaling such as YaRN, sliding-window
//! layers, quantization other than MLX 4-bit / group 64, ...). Attention bias (q/k/v) and tied embeddings
//! (lm_head = embed_tokens) are supported by the engine (`engine.rs`).
//!
//! Facts that are architectural, not in config.json (from the HF `modeling_qwen2.py` / `modeling_qwen3.py` sources):
//! - Qwen2 / Qwen2.5: q/k/v projections always carry a bias (o_proj none); Qwen2 configs have no `attention_bias` key.
//!   Qwen3 configs carry `attention_bias` (false for all released dense Qwen3 models).
//! - Qwen3: per-head RMSNorm on q and k (`q_norm`, `k_norm`, weight [head_dim]) before RoPE; Qwen2 has none.
//! - Both: non-traditional (rotate-half) RoPE over the full head dim unless `partial_rotary_factor` says otherwise.
//!
//! Hybrid configs (Qwen3.5 / 3.8: `linear_attention` layers, `linear_*` fields, `full_attention_interval`) are refused
//! at parse time with "hybrid ... use the main engine": streamdense-model's `QwenConfig` handles those.
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family { Qwen2, Qwen3 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Full, Sliding }

#[derive(Clone, Debug)]
pub struct DenseConfig {
    pub model_type: String,
    pub family: Family,
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub kinds: Vec<LayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `head_dim` was given in config.json (else hidden / heads). Qwen3-32B needs it: 64 x 128 != 5120.
    pub head_dim_explicit: bool,
    pub rope_dims: usize,
    pub rope_theta: f64,
    /// `None` = plain RoPE; `Some(description)` for any scaling (yarn, linear, dynamic, ...).
    pub rope_scaling: Option<String>,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub qk_norm: bool,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    pub sliding_window: Option<usize>,
    pub hidden_act: String,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    /// MLX `quantization` (or `quantization_config`) block: (group_size, bits). None = not an MLX-quantized config
    /// (the HF originals); the weight loader then refuses the model, `validate()` does not.
    pub quant: Option<(usize, usize)>,
}

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

/// True for a hybrid (linear-attention) Qwen config: any `linear_attention` layer, any `linear_*` field, or
/// `full_attention_interval`. Looks inside `text_config` when present.
pub fn is_hybrid(root: &Value) -> bool {
    let c = if root["text_config"].is_object() { &root["text_config"] } else { root };
    let lin_layer = c["layer_types"].as_array().is_some_and(|a| a.iter().any(|t| t.as_str() == Some("linear_attention")));
    let lin_field = c.as_object().is_some_and(|o| o.keys().any(|k| k.starts_with("linear_")));
    lin_layer || lin_field || !c["full_attention_interval"].is_null()
}

impl DenseConfig {
    /// Parse `dir/config.json` (+ `dir/generation_config.json` for eos when present). Does not call `validate()`.
    pub fn from_model_dir(dir: &str) -> Result<DenseConfig, String> {
        let path = format!("{}/config.json", dir);
        let txt = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path, e))?;
        let root: Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path, e))?;
        let gen: Option<Value> = std::fs::read_to_string(format!("{}/generation_config.json", dir))
            .ok().and_then(|t| serde_json::from_str(&t).ok());
        Self::from_json(&root, gen.as_ref())
    }

    /// Parse from an already-loaded config (root object; `text_config` is used when present).
    pub fn from_json(root: &Value, gen: Option<&Value>) -> Result<DenseConfig, String> {
        let c = if root["text_config"].is_object() { &root["text_config"] } else { root };
        let model_type = c["model_type"].as_str().or(root["model_type"].as_str()).unwrap_or("").to_string();
        if is_hybrid(root) {
            return Err(format!("config: hybrid linear-attention model (model_type {:?}): use the main engine \
                                (streamdense-model QwenConfig), not qwen-dense", model_type));
        }
        let moe = ["num_experts", "n_routed_experts", "num_experts_per_tok"].iter().any(|k| !c[*k].is_null());
        if moe || model_type.contains("moe") {
            return Err(format!("config: mixture-of-experts model (model_type {:?}): qwen-dense is for dense models only", model_type));
        }
        let family = match model_type.as_str() {
            "qwen2" => Family::Qwen2,
            "qwen3" => Family::Qwen3,
            other => return Err(format!("config: model_type {:?} is not a Qwen dense model (want qwen2 or qwen3)", other)),
        };
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config: missing {}", k));
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let layers = u("num_hidden_layers")?;
        if heads == 0 { return Err("config: num_attention_heads = 0".into()); }
        let head_dim_explicit = c["head_dim"].as_u64().is_some();
        let head_dim = match c["head_dim"].as_u64() {
            Some(v) => v as usize,
            None if hidden % heads == 0 => hidden / heads,
            None => return Err(format!("config: no head_dim and hidden_size {} is not a multiple of num_attention_heads {}", hidden, heads)),
        };

        // Sliding window applies only with use_sliding_window = true (HF Qwen2: layers >= max_window_layers).
        let use_sw = c["use_sliding_window"].as_bool().unwrap_or(false);
        let sliding_window = if use_sw { c["sliding_window"].as_u64().map(|v| v as usize) } else { None };
        let kinds: Vec<LayerKind> = match c["layer_types"].as_array() {
            Some(a) => {
                let mut k = Vec::with_capacity(a.len());
                for (l, t) in a.iter().enumerate() {
                    k.push(match t.as_str() {
                        Some("full_attention") => LayerKind::Full,
                        Some("sliding_attention") => LayerKind::Sliding,
                        other => return Err(format!("config: layer_types[{}] unknown: {:?}", l, other)),
                    });
                }
                k
            }
            None => {
                let mwl = c["max_window_layers"].as_u64().map(|v| v as usize).unwrap_or(layers);
                (0..layers).map(|l| if sliding_window.is_some() && l >= mwl { LayerKind::Sliding } else { LayerKind::Full }).collect()
            }
        };
        if kinds.len() != layers {
            return Err(format!("config: layer_types has {} entries but num_hidden_layers = {}", kinds.len(), layers));
        }

        // RoPE: transformers >= 4.56 may write rope_parameters {rope_theta, rope_type, ...} instead of rope_theta /
        // rope_scaling. rope_type / type "default" (or absent) = plain RoPE.
        let rp = &c["rope_parameters"];
        let rope_theta = rp["rope_theta"].as_f64().or(c["rope_theta"].as_f64()).unwrap_or(10000.0);
        let prf = c["partial_rotary_factor"].as_f64().or(rp["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let rope_dims = (prf * head_dim as f64).round() as usize;
        let scaling = |s: &Value| -> Option<String> {
            if !s.is_object() { return None; }
            let ty = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
            if ty == "default" { return None; }
            let mut d = ty.to_string();
            if let Some(f) = s["factor"].as_f64() { d += &format!(" factor {}", f); }
            if let Some(o) = s["original_max_position_embeddings"].as_u64() { d += &format!(" original_max_position_embeddings {}", o); }
            Some(d)
        };
        let rope_scaling = scaling(&c["rope_scaling"]).or(scaling(rp));

        let attention_bias = c["attention_bias"].as_bool().unwrap_or(family == Family::Qwen2);
        let tie_word_embeddings = c["tie_word_embeddings"].as_bool().or(root["tie_word_embeddings"].as_bool()).unwrap_or(false);

        let mut eos: Vec<u32> = vec![];
        for v in [Some(&root["eos_token_id"]), Some(&c["eos_token_id"]), gen.map(|g| &g["eos_token_id"])].into_iter().flatten() {
            for id in ids(v) { if !eos.contains(&id) { eos.push(id); } }
        }
        if eos.is_empty() { return Err("config: no eos_token_id in config.json or generation_config.json".into()); }

        let qb = if root["quantization"].is_object() { &root["quantization"] } else { &root["quantization_config"] };
        let quant = if qb.is_object() {
            let g = qb["group_size"].as_u64().ok_or("config: quantization without group_size")? as usize;
            let b = qb["bits"].as_u64().ok_or("config: quantization without bits")? as usize;
            Some((g, b))
        } else { None };

        Ok(DenseConfig {
            model_type,
            family,
            hidden,
            inter: u("intermediate_size")?,
            layers,
            kinds,
            heads,
            kv_heads: c["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads),
            head_dim,
            head_dim_explicit,
            rope_dims,
            rope_theta,
            rope_scaling,
            attention_bias,
            tie_word_embeddings,
            qk_norm: family == Family::Qwen3,
            vocab: u("vocab_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            max_pos: c["max_position_embeddings"].as_u64().unwrap_or(0) as usize,
            sliding_window,
            hidden_act: c["hidden_act"].as_str().unwrap_or("silu").to_string(),
            bos: c["bos_token_id"].as_u64().map(|v| v as u32),
            eos,
            quant,
        })
    }

    /// Every feature the dense path (attn_hd128 kernels + the engine's 4-bit FFN / head) does not support yet, by
    /// name. Empty = supported. Attention bias and tied embeddings are supported.
    pub fn unsupported(&self) -> Vec<String> {
        let mut errs: Vec<String> = vec![];
        if self.head_dim != 128 { errs.push(format!("head_dim must be 128 (got {})", self.head_dim)); }
        if self.rope_dims != self.head_dim {
            errs.push(format!("partial rotary: rope dims {} != head_dim {} (attn_hd128 is tested with full rotary)", self.rope_dims, self.head_dim));
        }
        if let Some(s) = &self.rope_scaling { errs.push(format!("rope_scaling {} (only plain RoPE is implemented; YaRN etc. not yet)", s)); }
        if self.kv_heads == 0 || self.heads % self.kv_heads != 0 {
            errs.push(format!("num_attention_heads ({}) must be a multiple of num_key_value_heads ({})", self.heads, self.kv_heads));
        }
        if let Some((g, b)) = self.quant {
            if (g, b) != (64, 4) { errs.push(format!("quantization group_size {} bits {} (only MLX 4-bit, group 64)", g, b)); }
        }
        let n_sw = self.kinds.iter().filter(|k| **k == LayerKind::Sliding).count();
        if n_sw > 0 { errs.push(format!("sliding-window attention on {} layers (window {:?}) not implemented", n_sw, self.sliding_window)); }
        if self.hidden_act != "silu" { errs.push(format!("hidden_act must be silu (got {})", self.hidden_act)); }
        if self.kinds.len() != self.layers { errs.push(format!("kinds has {} entries, layers = {}", self.kinds.len(), self.layers)); }
        for (name, d) in [("hidden_size", self.hidden), ("intermediate_size", self.inter), ("q dim (heads x head_dim)", self.q_dim()),
                          ("kv dim (kv_heads x head_dim)", self.kv_dim())] {
            if d % 64 != 0 { errs.push(format!("{} = {} must be a multiple of 64 (4-bit group size)", name, d)); }
        }
        errs
    }

    /// `Ok` if `unsupported()` is empty, else every unsupported feature in one message.
    pub fn validate(&self) -> Result<(), String> {
        let errs = self.unsupported();
        if errs.is_empty() { Ok(()) } else { Err(format!("qwen dense config not supported: {}", errs.join("; "))) }
    }

    pub fn q_dim(&self) -> usize { self.heads * self.head_dim }
    pub fn kv_dim(&self) -> usize { self.kv_heads * self.head_dim }
    /// q heads per kv head (GQA group).
    pub fn grp(&self) -> usize { self.heads.checked_div(self.kv_heads).unwrap_or(0) }
    /// Fused attention input rows: q | k | v (no output gate in Qwen dense).
    pub fn attn_in_rows(&self) -> usize { self.q_dim() + 2 * self.kv_dim() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Public HF config.json shapes (Instruct repos). Field values are from memory of the public files; values marked
    // "unverified" below are ones I am not sure of. None of the unverified fields affect parsing results asserted here
    // except where stated. All dims (hidden / inter / layers / heads / kv / head_dim / vocab / tie) I am confident of.

    // Qwen/Qwen2.5-0.5B-Instruct. unverified: max_window_layers (21 vs 24), sliding_window (32768 vs 131072),
    // transformers_version. Base repo differs in eos (151643).
    const QWEN25_0_5B: &str = r#"{
      "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
      "hidden_act": "silu", "hidden_size": 896, "initializer_range": 0.02, "intermediate_size": 4864,
      "max_position_embeddings": 32768, "max_window_layers": 21, "model_type": "qwen2", "num_attention_heads": 14,
      "num_hidden_layers": 24, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
      "sliding_window": 32768, "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.43.1",
      "use_cache": true, "use_sliding_window": false, "vocab_size": 151936 }"#;

    // Qwen/Qwen2.5-7B-Instruct. unverified: sliding_window 131072, max_window_layers 28. No rope_scaling key in the
    // shipped file (the model card says to ADD yarn factor 4 for > 32K context; see yarn_reported).
    const QWEN25_7B: &str = r#"{
      "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
      "hidden_act": "silu", "hidden_size": 3584, "initializer_range": 0.02, "intermediate_size": 18944,
      "max_position_embeddings": 32768, "max_window_layers": 28, "model_type": "qwen2", "num_attention_heads": 28,
      "num_hidden_layers": 28, "num_key_value_heads": 4, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
      "sliding_window": 131072, "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.43.1",
      "use_cache": true, "use_sliding_window": false, "vocab_size": 152064 }"#;

    // Qwen/Qwen2.5-14B-Instruct. unverified: rms_norm_eps (1e-06 vs 1e-05), max_window_layers (70 vs 48),
    // sliding_window 131072.
    const QWEN25_14B: &str = r#"{
      "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
      "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02, "intermediate_size": 13824,
      "max_position_embeddings": 32768, "max_window_layers": 70, "model_type": "qwen2", "num_attention_heads": 40,
      "num_hidden_layers": 48, "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
      "sliding_window": 131072, "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.43.1",
      "use_cache": true, "use_sliding_window": false, "vocab_size": 152064 }"#;

    // Qwen/Qwen3-8B. unverified: transformers_version only.
    const QWEN3_8B: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643,
      "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02,
      "intermediate_size": 12288, "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
      "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
      "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null, "tie_word_embeddings": false,
      "torch_dtype": "bfloat16", "transformers_version": "4.51.0", "use_cache": true, "use_sliding_window": false,
      "vocab_size": 151936 }"#;

    // Qwen/Qwen3-14B. unverified: max_window_layers 40, transformers_version.
    const QWEN3_14B: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643,
      "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02,
      "intermediate_size": 17408, "max_position_embeddings": 40960, "max_window_layers": 40, "model_type": "qwen3",
      "num_attention_heads": 40, "num_hidden_layers": 40, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
      "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null, "tie_word_embeddings": false,
      "torch_dtype": "bfloat16", "transformers_version": "4.51.0", "use_cache": true, "use_sliding_window": false,
      "vocab_size": 151936 }"#;

    // Qwen/Qwen3-32B. Note q dim 64 x 128 = 8192 != hidden 5120: head_dim must come from the field.
    // unverified: max_window_layers 64, transformers_version.
    const QWEN3_32B: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643,
      "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02,
      "intermediate_size": 25600, "max_position_embeddings": 40960, "max_window_layers": 64, "model_type": "qwen3",
      "num_attention_heads": 64, "num_hidden_layers": 64, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
      "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null, "tie_word_embeddings": false,
      "torch_dtype": "bfloat16", "transformers_version": "4.51.0", "use_cache": true, "use_sliding_window": false,
      "vocab_size": 151936 }"#;

    fn parse(s: &str) -> DenseConfig { DenseConfig::from_json(&serde_json::from_str(s).unwrap(), None).unwrap() }
    fn dims(c: &DenseConfig) -> (usize, usize, usize, usize, usize, usize, usize) {
        (c.hidden, c.inter, c.layers, c.heads, c.kv_heads, c.head_dim, c.vocab)
    }
    fn report(name: &str, c: &DenseConfig) {
        println!("  {:<14} {:?} -> {}", name, dims(c), match c.validate() { Ok(()) => "supported".into(), Err(e) => e });
    }

    #[test]
    fn qwen25_0_5b() {
        let c = parse(QWEN25_0_5B);
        report("Qwen2.5-0.5B", &c);
        assert_eq!(c.family, Family::Qwen2);
        assert_eq!(dims(&c), (896, 4864, 24, 14, 2, 64, 151936));
        assert!(!c.head_dim_explicit);
        assert!(c.attention_bias && c.tie_word_embeddings && !c.qk_norm);
        assert_eq!((c.rope_theta, c.rope_dims, c.rope_scaling.clone()), (1e6, 64, None));
        assert!(c.kinds.iter().all(|k| *k == LayerKind::Full));
        assert_eq!(c.eos, vec![151645]);
        // head dim 64 (896 / 14): not the 128 kernel (bias and tied embeddings are supported)
        let u = c.unsupported();
        assert_eq!(u.len(), 1, "{:?}", u);
        assert!(u[0].contains("head_dim must be 128 (got 64)"));
    }

    #[test]
    fn qwen25_7b() {
        let c = parse(QWEN25_7B);
        report("Qwen2.5-7B", &c);
        assert_eq!(dims(&c), (3584, 18944, 28, 28, 4, 128, 152064));
        assert_eq!((c.grp(), c.q_dim(), c.kv_dim(), c.attn_in_rows()), (7, 3584, 512, 4608));
        assert!(c.attention_bias && !c.tie_word_embeddings && !c.qk_norm);
        assert_eq!(c.sliding_window, None); // use_sliding_window = false
        c.validate().unwrap(); // attention bias supported
    }

    #[test]
    fn qwen25_14b() {
        let c = parse(QWEN25_14B);
        report("Qwen2.5-14B", &c);
        assert_eq!(dims(&c), (5120, 13824, 48, 40, 8, 128, 152064));
        assert_eq!(c.grp(), 5);
        // max_window_layers (70) > layers is harmless while use_sliding_window = false
        assert!(c.kinds.iter().all(|k| *k == LayerKind::Full));
        assert_eq!(c.unsupported().len(), 0);
    }

    #[test]
    fn qwen3_dense_supported() {
        for (name, s, want, inter) in [("Qwen3-8B", QWEN3_8B, (4096, 36, 32, 8), 12288),
                                        ("Qwen3-14B", QWEN3_14B, (5120, 40, 40, 8), 17408),
                                        ("Qwen3-32B", QWEN3_32B, (5120, 64, 64, 8), 25600)] {
            let c = parse(s);
            report(name, &c);
            assert_eq!(c.family, Family::Qwen3);
            assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads), want, "{}", name);
            assert_eq!((c.inter, c.head_dim, c.rope_dims, c.vocab), (inter, 128, 128, 151936), "{}", name);
            assert!(c.head_dim_explicit && c.qk_norm && !c.attention_bias && !c.tie_word_embeddings, "{}", name);
            assert_eq!((c.rope_theta, c.eps, c.max_pos), (1e6, 1e-6, 40960));
            assert_eq!((c.bos, c.eos.clone()), (Some(151643), vec![151645]));
            c.validate().unwrap_or_else(|e| panic!("{}: {}", name, e));
        }
        let c = parse(QWEN3_32B);
        assert_eq!((c.q_dim(), c.kv_dim(), c.attn_in_rows()), (8192, 1024, 10240));
    }

    #[test]
    fn yarn_reported() {
        // Qwen2.5 / Qwen3 model cards: add this block for long context
        let mut v: Value = serde_json::from_str(QWEN3_8B).unwrap();
        v["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768});
        let c = DenseConfig::from_json(&v, None).unwrap();
        let e = c.validate().unwrap_err();
        println!("  yarn: {}", e);
        assert!(e.contains("rope_scaling yarn factor 4 original_max_position_embeddings 32768"), "{}", e);
        // older spelling "type", and the rope_parameters form
        let mut v: Value = serde_json::from_str(QWEN25_7B).unwrap();
        v["rope_scaling"] = serde_json::json!({"type": "yarn", "factor": 4.0});
        assert!(DenseConfig::from_json(&v, None).unwrap().rope_scaling.unwrap().starts_with("yarn"));
        let mut v: Value = serde_json::from_str(QWEN3_8B).unwrap();
        v.as_object_mut().unwrap().remove("rope_theta");
        v["rope_parameters"] = serde_json::json!({"rope_type": "default", "rope_theta": 5e5});
        let c = DenseConfig::from_json(&v, None).unwrap();
        assert_eq!((c.rope_theta, c.rope_scaling), (5e5, None));
    }

    #[test]
    fn other_unsupported_named() {
        let edit = |s: &str, f: &dyn Fn(&mut Value)| { let mut v: Value = serde_json::from_str(s).unwrap(); f(&mut v); DenseConfig::from_json(&v, None) };
        // sliding window on: layers >= max_window_layers become sliding
        let c = edit(QWEN25_7B, &|v| { v["use_sliding_window"] = true.into(); v["max_window_layers"] = 20.into(); }).unwrap();
        assert_eq!(c.kinds.iter().filter(|k| **k == LayerKind::Sliding).count(), 8);
        assert!(c.validate().unwrap_err().contains("sliding-window attention on 8 layers"));
        let c = edit(QWEN3_8B, &|v| v["partial_rotary_factor"] = 0.5.into()).unwrap();
        assert!(c.validate().unwrap_err().contains("partial rotary"));
        let c = edit(QWEN3_8B, &|v| v["num_key_value_heads"] = 5.into()).unwrap();
        assert!(c.validate().unwrap_err().contains("multiple of num_key_value_heads"));
        let c = edit(QWEN3_8B, &|v| v["hidden_act"] = "gelu".into()).unwrap();
        assert!(c.validate().unwrap_err().contains("hidden_act"));
        // parse errors
        assert!(edit(QWEN3_8B, &|v| { v.as_object_mut().unwrap().remove("vocab_size"); }).unwrap_err().contains("vocab_size"));
        assert!(edit(QWEN3_8B, &|v| v["model_type"] = "falcon".into()).unwrap_err().contains("not a Qwen dense model"));
        assert!(edit(QWEN3_8B, &|v| { v["model_type"] = "qwen3_moe".into(); v["num_experts"] = 128.into(); }).unwrap_err().contains("mixture-of-experts"));
        assert!(edit(QWEN3_8B, &|v| { v.as_object_mut().unwrap().remove("eos_token_id"); }).unwrap_err().contains("eos"));
        // eos from generation_config is merged
        let v: Value = serde_json::from_str(QWEN3_8B).unwrap();
        let g = serde_json::json!({"eos_token_id": [151645, 151643]});
        assert_eq!(DenseConfig::from_json(&v, Some(&g)).unwrap().eos, vec![151645, 151643]);
    }

    #[test]
    fn quantization_parsed() {
        let mut v: Value = serde_json::from_str(QWEN3_8B).unwrap();
        assert_eq!(DenseConfig::from_json(&v, None).unwrap().quant, None);
        v["quantization"] = serde_json::json!({"group_size": 64, "bits": 4});
        let c = DenseConfig::from_json(&v, None).unwrap();
        assert_eq!(c.quant, Some((64, 4)));
        c.validate().unwrap();
        v["quantization"] = serde_json::json!({"group_size": 32, "bits": 3});
        assert!(DenseConfig::from_json(&v, None).unwrap().validate().unwrap_err().contains("group_size 32 bits 3"));
    }

    /// The two local MLX 4-bit models (config.json only; no weights read). Skipped when the folder is missing.
    #[test]
    fn local_models_validate() {
        for (m, fam, bias) in [("qwen3-1.7b-mlx4", Family::Qwen3, false), ("qwen2.5-1.5b-instruct-mlx4", Family::Qwen2, true)] {
            let dir = format!("{}/models/{}", env!("CARGO_MANIFEST_DIR"), m);
            let c = match DenseConfig::from_model_dir(&dir) {
                Err(e) if e.contains("No such file") => { println!("  SKIP {}: missing", m); continue; }
                r => r.unwrap(),
            };
            report(m, &c);
            c.validate().unwrap_or_else(|e| panic!("{}: {}", m, e));
            assert_eq!((c.family, c.attention_bias, c.tie_word_embeddings, c.qk_norm), (fam, bias, true, fam == Family::Qwen3));
            assert_eq!((c.head_dim, c.quant), (128, Some((64, 4))));
        }
    }

    /// Qwen3.8-27B (qwen3_5) shape, fields as in models/qwen38_27b_mlx4/config.json (layer_types generated: every 4th
    /// layer full attention).
    fn qwen38_hybrid() -> Value {
        let lt: Vec<&str> = (0..64).map(|l| if (l + 1) % 4 == 0 { "full_attention" } else { "linear_attention" }).collect();
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "eos_token_id": [248046, 248044],
            "text_config": {
                "attention_bias": false, "attn_output_gate": true, "full_attention_interval": 4, "head_dim": 256,
                "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 17408, "layer_types": lt,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
                "model_type": "qwen3_5_text", "num_attention_heads": 24, "num_hidden_layers": 64, "num_key_value_heads": 4,
                "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06, "vocab_size": 248320
            }
        })
    }

    #[test]
    fn hybrid_refused() {
        let v = qwen38_hybrid();
        assert!(is_hybrid(&v));
        let e = DenseConfig::from_json(&v, None).unwrap_err();
        println!("  Qwen3.8 hybrid: {}", e);
        assert!(e.contains("hybrid") && e.contains("use the main engine"), "{}", e);
        // any single hybrid marker is enough
        let mut v = qwen38_hybrid();
        v["text_config"].as_object_mut().unwrap().retain(|k, _| !k.starts_with("linear_") && k != "full_attention_interval");
        assert!(DenseConfig::from_json(&v, None).unwrap_err().contains("hybrid"));
        // the real local file when present (config.json only; no weights are read)
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/qwen38_27b_mlx4");
        match DenseConfig::from_model_dir(dir) {
            Err(e) if e.contains("No such file") => println!("  SKIP real Qwen3.8 config: {} missing", dir),
            r => assert!(r.unwrap_err().contains("use the main engine")),
        }
        for s in [QWEN25_0_5B, QWEN25_7B, QWEN25_14B, QWEN3_8B, QWEN3_14B, QWEN3_32B] {
            assert!(!is_hybrid(&serde_json::from_str(s).unwrap()));
        }
    }
}
