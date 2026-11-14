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
//!
//! Other dense families (Llama incl. SmolLM2 / Llama 3, Granite, Gemma 1 / 2 / 3-text) are not parsed here: their
//! config readers build a `DenseConfig` directly (start from `DenseConfig::base`, or any parsed config, and set the
//! fields below). Every math switch is an explicit field whose default is the Qwen behaviour:
//!
//! | field | meaning (default = Qwen) |
//! |---|---|
//! | `head_dim` | 64, 128 or 256 |
//! | `attn_scale` | `None` = head_dim^-0.5; Granite `attention_multiplier`; Gemma `query_pre_attn_scalar`^-0.5 |
//! | `attn_softcap` / `final_softcap` | `None`; Gemma 2: scores / logits = cap * tanh(x / cap) |
//! | `activation` | `Silu` (SwiGLU) or `GeluTanh` (GeGLU, gelu_pytorch_tanh) |
//! | `norm_plus_one` | false: RMSNorm weight w; true: (1 + w) (Gemma; also q_norm / k_norm) |
//! | `four_norms` | false: input + post_attention (pre-FFN) norm; true: Gemma 2 / 3 input, post_attention (on the attention output), pre_feedforward, post_feedforward |
//! | `qk_norm` | per-head RMSNorm on q / k before RoPE (Qwen3, Gemma 3) |
//! | `embed_scale` | 1.0; Gemma sqrt(hidden), Granite `embedding_multiplier` (applied in bf16) |
//! | `residual_multiplier` | 1.0; Granite: h += r * m (bf16) |
//! | `logits_scaling` | 1.0; Granite: logits / logits_scaling |
//! | `kinds` + `sliding_window` | per-layer full / sliding attention (key j visible to query i iff i - j < window) |
//! | `rope_local_theta` | `None`; Gemma 3: sliding layers use plain RoPE with this theta (full layers: `rope_theta` + `rope_scale`) |
//! | `rope_scale` | `RopeScaling::None`, `Linear`, `Llama3`; anything else goes to `rope_scaling` (reported unsupported) |
//! | `attention_bias` / `o_bias` / `mlp_bias` | q/k/v bias (Qwen2 true) / o_proj bias / gate, up, down bias (Llama `attention_bias` sets both of the first two) |
//! | `tie_word_embeddings` | lm_head = embed_tokens |
use serde_json::Value;

/// Head dims the attention kernels are built for.
pub const HEAD_DIMS: [usize; 3] = [64, 128, 256];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family { Qwen2, Qwen3, Llama, Granite, Gemma, Gemma2, Gemma3 }

impl Family {
    pub fn is_qwen(self) -> bool { matches!(self, Family::Qwen2 | Family::Qwen3) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Full, Sliding }

/// FFN gate activation: out = act(gate x) * (up x).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    /// silu (SwiGLU), MLX bf16 rounding chain
    Silu,
    /// tanh-approximated GELU (GeGLU): HF `gelu_pytorch_tanh` / `gelu_new`, mlx `gelu_approx`
    GeluTanh,
    /// anything else (named by `unsupported()`), e.g. exact-erf "gelu"
    Other(String),
}

impl Activation {
    /// From a HF `hidden_act` / `hidden_activation` string.
    pub fn from_hf(s: &str) -> Activation {
        match s {
            "silu" | "swish" => Activation::Silu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_tanh" => Activation::GeluTanh,
            o => Activation::Other(o.to_string()),
        }
    }
}

/// RoPE frequency scaling the engine implements (folded into the inverse-frequency table at load).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    /// positions / factor (HF "linear", mlx nn.RoPE scale = 1 / factor)
    Linear { factor: f64 },
    /// HF / mlx_lm "llama3" frequency smoothing
    Llama3 { factor: f64, low_freq_factor: f64, high_freq_factor: f64, original_max_pos: usize },
}

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
    /// Scaling the engine does NOT implement (yarn, dynamic, longrope, ...): `Some(description)`, reported by
    /// `unsupported()`. `None` = plain RoPE or one of `rope_scale`.
    pub rope_scaling: Option<String>,
    /// Scaling the engine implements (full-attention layers; sliding layers too unless `rope_local_theta` is set).
    pub rope_scale: RopeScaling,
    /// Theta of the sliding layers' plain RoPE (Gemma 3 `rope_local_base_freq`); `None` = same table as full layers.
    pub rope_local_theta: Option<f64>,
    /// q / k / v projection bias.
    pub attention_bias: bool,
    /// o_proj bias (Llama `attention_bias`).
    pub o_bias: bool,
    /// gate / up / down projection bias (Llama `mlp_bias`).
    pub mlp_bias: bool,
    pub tie_word_embeddings: bool,
    pub qk_norm: bool,
    pub vocab: usize,
    pub eps: f32,
    pub max_pos: usize,
    pub sliding_window: Option<usize>,
    /// The config's activation string (informational; the engine reads `activation`).
    pub hidden_act: String,
    pub activation: Activation,
    /// `None` = head_dim^-0.5.
    pub attn_scale: Option<f64>,
    /// Attention-score soft cap (cap * tanh(s / cap), after scaling, before the mask / softmax).
    pub attn_softcap: Option<f64>,
    /// Final-logit soft cap (after `logits_scaling`).
    pub final_softcap: Option<f64>,
    /// RMSNorm weight applied as (1 + w), every norm incl. QK-norm (folded at load, rounded to bf16 as MLX).
    pub norm_plus_one: bool,
    /// Gemma 2 / 3 layer: h += post_attention_layernorm(attn); h += post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h))).
    pub four_norms: bool,
    /// Embedding output multiplier (rounded to bf16, as MLX multiplies a bf16 array by it).
    pub embed_scale: f64,
    /// Residual branch multiplier (rounded to bf16): h = h + r * m.
    pub residual_multiplier: f64,
    /// Logits are divided by this (f32).
    pub logits_scaling: f64,
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

/// The implemented RoPE scalings from a `rope_scaling` / `rope_parameters` object ("linear", "llama3"); `None` for
/// absent / default / any other type. llama3 defaults (when a field is missing) are mlx_lm's Llama3RoPE ones:
/// low_freq_factor 1, high_freq_factor 4, original_max_position_embeddings 8192.
pub fn rope_scaling_kind(s: &Value) -> Option<RopeScaling> {
    if !s.is_object() { return None; }
    let ty = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
    let factor = s["factor"].as_f64();
    match ty {
        "linear" => factor.map(|factor| RopeScaling::Linear { factor }),
        "llama3" => factor.map(|factor| RopeScaling::Llama3 {
            factor,
            low_freq_factor: s["low_freq_factor"].as_f64().unwrap_or(1.0),
            high_freq_factor: s["high_freq_factor"].as_f64().unwrap_or(4.0),
            original_max_pos: s["original_max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
        }),
        _ => None,
    }
}

/// RoPE inverse frequencies (f32, rdims / 2 entries) for `theta` with `scale`, computed in f64. Plain RoPE matches
/// `attn128_rope_invf` up to f32 rounding (the engine uses that kernel for plain tables). llama3: mlx_lm Llama3RoPE
/// (wavelen > low-freq wavelen: freq / factor; between: smoothed; else unchanged). Linear: invf / factor (mlx
/// nn.RoPE(scale = 1 / factor) multiplies the position instead: same angle up to rounding).
pub fn rope_inv_freq(rdims: usize, theta: f64, scale: RopeScaling) -> Vec<f32> {
    let n = rdims / 2;
    (0..n).map(|i| {
        let period = theta.powf((2 * i) as f64 / rdims as f64); // mlx "freqs" = base ** (2i / dims)
        let p = match scale {
            RopeScaling::None => period,
            RopeScaling::Linear { factor } => period * factor,
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_pos } => {
                let old = original_max_pos as f64;
                let (low_wl, high_wl) = (old / low_freq_factor, old / high_freq_factor);
                let wl = 2.0 * std::f64::consts::PI * period;
                if wl > low_wl {
                    period * factor
                } else if wl > high_wl && wl < low_wl {
                    let smooth = (old / wl - low_freq_factor) / (high_freq_factor - low_freq_factor);
                    period / ((1.0 - smooth) / factor + smooth)
                } else {
                    period
                }
            }
        };
        (1.0 / p) as f32
    }).collect()
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
                                (streamdense-model QwenConfig), not dense-engine", model_type));
        }
        let moe = ["num_experts", "n_routed_experts", "num_experts_per_tok"].iter().any(|k| !c[*k].is_null());
        if moe || model_type.contains("moe") {
            return Err(format!("config: mixture-of-experts model (model_type {:?}): dense-engine is for dense models only", model_type));
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
        let (rope_scale, rope_scaling) = match rope_scaling_kind(&c["rope_scaling"]).or(rope_scaling_kind(rp)) {
            Some(k) => (k, None),
            None => (RopeScaling::None, rope_scaling),
        };

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
            rope_scale,
            rope_local_theta: None,
            attention_bias,
            o_bias: false,
            mlp_bias: false,
            tie_word_embeddings,
            qk_norm: family == Family::Qwen3,
            vocab: u("vocab_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            max_pos: c["max_position_embeddings"].as_u64().unwrap_or(0) as usize,
            sliding_window,
            hidden_act: c["hidden_act"].as_str().unwrap_or("silu").to_string(),
            activation: Activation::from_hf(c["hidden_act"].as_str().unwrap_or("silu")),
            attn_scale: None,
            attn_softcap: None,
            final_softcap: None,
            norm_plus_one: false,
            four_norms: false,
            embed_scale: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            bos: c["bos_token_id"].as_u64().map(|v| v as u32),
            eos,
            quant,
        })
    }

    /// A plain all-full-attention config with every math switch at its default (Qwen2-like without bias: SiLU,
    /// w norms, head_dim^-0.5, plain RoPE theta 10000, untied, eps 1e-6, MLX 4-bit group 64). Family config readers
    /// start here (or from any parsed config) and set the fields of their family.
    pub fn base(model_type: &str, family: Family, hidden: usize, inter: usize, layers: usize, heads: usize,
                kv_heads: usize, head_dim: usize, vocab: usize) -> DenseConfig {
        DenseConfig {
            model_type: model_type.to_string(), family, hidden, inter, layers, kinds: vec![LayerKind::Full; layers],
            heads, kv_heads, head_dim, head_dim_explicit: true, rope_dims: head_dim, rope_theta: 10000.0,
            rope_scaling: None, rope_scale: RopeScaling::None, rope_local_theta: None, attention_bias: false,
            o_bias: false, mlp_bias: false, tie_word_embeddings: false, qk_norm: false, vocab, eps: 1e-6, max_pos: 0,
            sliding_window: None, hidden_act: "silu".into(), activation: Activation::Silu, attn_scale: None,
            attn_softcap: None, final_softcap: None, norm_plus_one: false, four_norms: false, embed_scale: 1.0,
            residual_multiplier: 1.0, logits_scaling: 1.0, bos: None, eos: vec![], quant: Some((64, 4)),
        }
    }

    /// Attention scale the kernels use (f32).
    pub fn attn_scale_f32(&self) -> f32 {
        self.attn_scale.unwrap_or((self.head_dim as f64).powf(-0.5)) as f32
    }
    pub fn n_sliding(&self) -> usize { self.kinds.iter().filter(|k| **k == LayerKind::Sliding).count() }

    /// Every feature the dense path (attention kernels at head dim 64 / 128 / 256 + the engine's 4-bit FFN / head)
    /// does not support, by name. Empty = supported.
    pub fn unsupported(&self) -> Vec<String> {
        let mut errs: Vec<String> = vec![];
        if !HEAD_DIMS.contains(&self.head_dim) { errs.push(format!("head_dim must be 64, 128 or 256 (got {})", self.head_dim)); }
        if self.rope_dims != self.head_dim {
            errs.push(format!("partial rotary: rope dims {} != head_dim {} (the prep kernel takes rdims but only full rotary is tested)", self.rope_dims, self.head_dim));
        }
        if let Some(s) = &self.rope_scaling {
            errs.push(format!("rope_scaling {} (implemented: plain RoPE, linear, llama3; YaRN, dynamic etc. not yet)", s));
        }
        let pos_f = |x: f64| x.is_finite() && x > 0.0;
        match self.rope_scale {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } => if !pos_f(factor) { errs.push(format!("rope_scaling linear factor {}", factor)); },
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_pos } => {
                if !(pos_f(factor) && pos_f(low_freq_factor) && high_freq_factor > low_freq_factor && original_max_pos > 0) {
                    errs.push(format!("rope_scaling llama3 factor {} low {} high {} original {}", factor, low_freq_factor,
                                      high_freq_factor, original_max_pos));
                }
            }
        }
        if !pos_f(self.rope_theta) { errs.push(format!("rope_theta {}", self.rope_theta)); }
        if let Some(t) = self.rope_local_theta { if !pos_f(t) { errs.push(format!("rope_local_theta {}", t)); } }
        if self.kv_heads == 0 || self.heads % self.kv_heads != 0 {
            errs.push(format!("num_attention_heads ({}) must be a multiple of num_key_value_heads ({})", self.heads, self.kv_heads));
        } else if self.heads / self.kv_heads > 32 {
            errs.push(format!("GQA group {} > 32 (flash attention kernel limit)", self.heads / self.kv_heads));
        }
        if let Some((g, b)) = self.quant {
            if (g, b) != (64, 4) { errs.push(format!("quantization group_size {} bits {} (only MLX 4-bit, group 64)", g, b)); }
        }
        let n_sw = self.n_sliding();
        if n_sw > 0 {
            if self.family.is_qwen() {
                // HF Qwen2 (use_sliding_window / max_window_layers) and mlx_lm (ignores it) disagree; not mirrored
                errs.push(format!("sliding-window attention on {} layers (window {:?}) not implemented for Qwen (use_sliding_window)",
                                  n_sw, self.sliding_window));
            } else if !matches!(self.sliding_window, Some(w) if w > 0) {
                errs.push(format!("{} sliding layers but sliding_window {:?}", n_sw, self.sliding_window));
            }
        }
        if let Activation::Other(a) = &self.activation {
            errs.push(format!("hidden_act must be silu or gelu_pytorch_tanh (got {})", a));
        }
        for (name, v) in [("attn_scale", self.attn_scale), ("attn_softcap", self.attn_softcap), ("final_softcap", self.final_softcap)] {
            if let Some(x) = v { if !pos_f(x) { errs.push(format!("{} {} must be finite and > 0", name, x)); } }
        }
        for (name, x) in [("embed_scale", self.embed_scale), ("residual_multiplier", self.residual_multiplier),
                          ("logits_scaling", self.logits_scaling)] {
            if !x.is_finite() || x == 0.0 { errs.push(format!("{} {} must be finite and != 0", name, x)); }
        }
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
        // head dim 64 (896 / 14): supported since the head-dim-64 kernels (bias and tied embeddings too)
        c.validate().unwrap();
        let mut c2 = c.clone();
        c2.head_dim = 96;
        c2.rope_dims = 96;
        assert!(c2.validate().unwrap_err().contains("head_dim must be 64, 128 or 256 (got 96)"));
    }

    #[test]
    fn rope_scalings() {
        // llama3 (Llama 3.1 / 3.2 values) and linear parse into rope_scale; yarn stays unsupported
        let mut v: Value = serde_json::from_str(QWEN3_8B).unwrap();
        v["rope_scaling"] = serde_json::json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                               "high_freq_factor": 4.0, "original_max_position_embeddings": 8192});
        let c = DenseConfig::from_json(&v, None).unwrap();
        assert_eq!(c.rope_scale, RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_pos: 8192 });
        assert_eq!(c.rope_scaling, None);
        c.validate().unwrap();
        v["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 8.0});
        assert_eq!(DenseConfig::from_json(&v, None).unwrap().rope_scale, RopeScaling::Linear { factor: 8.0 });
        v["rope_scaling"] = serde_json::json!({"type": "dynamic", "factor": 2.0});
        assert!(DenseConfig::from_json(&v, None).unwrap().validate().unwrap_err().contains("rope_scaling dynamic factor 2"));
        // tables: plain = theta^(-2i/d); llama3 leaves the high frequencies, divides the lowest by factor
        let p = rope_inv_freq(128, 500000.0, RopeScaling::None);
        assert_eq!(p[0], 1.0);
        assert!((p[1] as f64 - 500000f64.powf(-2.0 / 128.0)).abs() < 1e-7);
        let l = rope_inv_freq(128, 500000.0, RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_pos: 8192 });
        assert_eq!(l[0], p[0]);
        assert!((l[63] / p[63] - 0.125).abs() < 1e-6);
        let lin = rope_inv_freq(64, 10000.0, RopeScaling::Linear { factor: 4.0 });
        assert!((lin[5] / rope_inv_freq(64, 10000.0, RopeScaling::None)[5] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn family_fields_validate() {
        // a Gemma 2-like config built directly, as a family reader does
        let mut c = DenseConfig::base("gemma2", Family::Gemma2, 2304, 9216, 26, 8, 4, 256, 256000);
        c.activation = Activation::GeluTanh;
        c.norm_plus_one = true;
        c.four_norms = true;
        c.attn_scale = Some(256f64.powf(-0.5));
        c.attn_softcap = Some(50.0);
        c.final_softcap = Some(30.0);
        c.embed_scale = (2304f64).sqrt();
        c.kinds = (0..26).map(|l| if l % 2 == 0 { LayerKind::Sliding } else { LayerKind::Full }).collect();
        c.tie_word_embeddings = true;
        assert!(c.validate().unwrap_err().contains("13 sliding layers but sliding_window None"));
        c.sliding_window = Some(4096);
        c.validate().unwrap();
        assert_eq!(c.attn_scale_f32(), 0.0625);
        c.activation = Activation::from_hf("gelu");
        c.attn_softcap = Some(0.0);
        c.residual_multiplier = f64::NAN;
        let e = c.validate().unwrap_err();
        assert!(e.contains("hidden_act must be silu or gelu_pytorch_tanh (got gelu)") && e.contains("attn_softcap 0")
                && e.contains("residual_multiplier NaN"), "{}", e);
        let mut g = DenseConfig::base("llama", Family::Llama, 2048, 8192, 16, 64, 1, 32, 128256);
        assert!(g.validate().unwrap_err().contains("head_dim must be 64, 128 or 256 (got 32)"));
        g.head_dim = 64;
        g.rope_dims = 64;
        assert!(g.validate().unwrap_err().contains("GQA group 64 > 32"));
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
