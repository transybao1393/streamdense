//! Qwen3.5/3.8 hybrid config (Gated DeltaNet + full attention) from an MLX model dir. Text settings may be nested
//! under `text_config`. Every derived size comes from the fields by formula; `validate()` checks the kernels'
//! fixed assumptions separately, so a config can be parsed and inspected even when the kernels can't run it.
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { DeltaNet, Attention }

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub kinds: Vec<LayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_theta: f64,
    pub dn_kh: usize,
    pub dn_vh: usize,
    pub dn_dk: usize,
    pub dn_dv: usize,
    pub conv_k: usize,
    pub vocab: usize,
    pub eps: f32,
    pub eos: Vec<u32>,
    pub mtp_layers: usize,
    /// `tie_word_embeddings` (text_config first, then root; absent = false): the output head is the embedding matrix
    /// (`embed_tokens`), as mlx_lm's `embed_tokens.as_linear`; no `lm_head` tensor is read.
    pub tie_word_embeddings: bool,
}

/// Token ids from a scalar or an array field.
fn ids(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
        _ => vec![],
    }
}

impl QwenConfig {
    /// Parse `dir/config.json` (+ `dir/generation_config.json` for eos when present). Does not call `validate()`.
    pub fn from_model_dir(dir: &str) -> Result<QwenConfig, String> {
        let path = format!("{}/config.json", dir);
        let txt = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path, e))?;
        let root: Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path, e))?;
        let gen: Option<Value> = std::fs::read_to_string(format!("{}/generation_config.json", dir))
            .ok().and_then(|t| serde_json::from_str(&t).ok());
        Self::from_json(&root, gen.as_ref())
    }

    /// Parse from an already-loaded config (root object; `text_config` is used when present).
    pub fn from_json(root: &Value, gen: Option<&Value>) -> Result<QwenConfig, String> {
        let c = if root["text_config"].is_object() { &root["text_config"] } else { root };
        let rp = &c["rope_parameters"];
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config: missing {}", k));

        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let layers = u("num_hidden_layers")?;
        let head_dim = c["head_dim"].as_u64().map(|v| v as usize).unwrap_or(hidden / heads);

        let kinds: Vec<LayerKind> = match c["layer_types"].as_array() {
            Some(a) => {
                let mut k = Vec::with_capacity(a.len());
                for (l, t) in a.iter().enumerate() {
                    k.push(match t.as_str() {
                        Some("linear_attention") => LayerKind::DeltaNet,
                        Some("full_attention") => LayerKind::Attention,
                        other => return Err(format!("config: layer_types[{}] unknown: {:?}", l, other)),
                    });
                }
                k
            }
            None => {
                let every = c["full_attention_interval"].as_u64().map(|v| v as usize).unwrap_or(4).max(1);
                (0..layers).map(|l| if (l + 1) % every == 0 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect()
            }
        };
        if kinds.len() != layers {
            return Err(format!("config: layer_types has {} entries but num_hidden_layers = {}", kinds.len(), layers));
        }

        let prf = c["partial_rotary_factor"].as_f64().or(rp["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let rope_dims = (prf * head_dim as f64).round() as usize;
        let rope_theta = rp["rope_theta"].as_f64().or(c["rope_theta"].as_f64()).unwrap_or(10000.0);

        // eos: union of root, text_config and generation_config entries, first-seen order.
        let mut eos: Vec<u32> = vec![];
        for v in [Some(&root["eos_token_id"]), Some(&c["eos_token_id"]), gen.map(|g| &g["eos_token_id"])].into_iter().flatten() {
            for id in ids(v) { if !eos.contains(&id) { eos.push(id); } }
        }
        if eos.is_empty() { return Err("config: no eos_token_id in config.json or generation_config.json".into()); }

        Ok(QwenConfig {
            hidden,
            inter: u("intermediate_size")?,
            layers,
            kinds,
            heads,
            kv_heads: c["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads),
            head_dim,
            rope_dims,
            rope_theta,
            dn_kh: u("linear_num_key_heads")?,
            dn_vh: u("linear_num_value_heads")?,
            dn_dk: u("linear_key_head_dim")?,
            dn_dv: u("linear_value_head_dim")?,
            conv_k: u("linear_conv_kernel_dim")?,
            vocab: u("vocab_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            eos,
            mtp_layers: c["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize,
            tie_word_embeddings: c["tie_word_embeddings"].as_bool().or(root["tie_word_embeddings"].as_bool()).unwrap_or(false),
        })
    }

    /// Kernel assumptions. Each failure names the field and the expected value.
    pub fn validate(&self) -> Result<(), String> {
        let mut errs: Vec<String> = vec![];
        if self.head_dim != 256 { errs.push(format!("head_dim must be 256 (got {})", self.head_dim)); }
        if self.rope_dims != 64 { errs.push(format!("rope dims (partial_rotary_factor x head_dim) must be 64 (got {})", self.rope_dims)); }
        if self.dn_dk != 128 { errs.push(format!("linear_key_head_dim must be 128 (got {})", self.dn_dk)); }
        if self.dn_dv != 128 { errs.push(format!("linear_value_head_dim must be 128 (got {})", self.dn_dv)); }
        if self.dn_kh == 0 || self.dn_vh % self.dn_kh != 0 {
            errs.push(format!("linear_num_value_heads ({}) must be a multiple of linear_num_key_heads ({})", self.dn_vh, self.dn_kh));
        }
        if self.kv_heads == 0 || self.heads % self.kv_heads != 0 {
            errs.push(format!("num_attention_heads ({}) must be a multiple of num_key_value_heads ({})", self.heads, self.kv_heads));
        }
        if self.conv_k != 4 { errs.push(format!("linear_conv_kernel_dim must be 4 (got {})", self.conv_k)); }
        if self.kinds.len() != self.layers { errs.push(format!("kinds has {} entries, layers = {}", self.kinds.len(), self.layers)); }
        for (name, d) in [("hidden_size", self.hidden), ("intermediate_size", self.inter), ("q dim (heads x head_dim)", self.q_dim()),
                          ("kv dim (kv_heads x head_dim)", self.kv_dim()), ("DeltaNet conv dim", self.conv_dim()),
                          ("DeltaNet value dim (vh x dv)", self.dn_v_dim())] {
            if d % 64 != 0 { errs.push(format!("{} = {} must be a multiple of 64 (4-bit group size)", name, d)); }
        }
        if errs.is_empty() { Ok(()) } else { Err(format!("qwen config not supported by the kernels: {}", errs.join("; "))) }
    }

    pub fn q_dim(&self) -> usize { self.heads * self.head_dim }
    pub fn kv_dim(&self) -> usize { self.kv_heads * self.head_dim }
    /// DeltaNet q (and k) width: kh x dk.
    pub fn dn_k_dim(&self) -> usize { self.dn_kh * self.dn_dk }
    /// DeltaNet v (and z) width: vh x dv.
    pub fn dn_v_dim(&self) -> usize { self.dn_vh * self.dn_dv }
    /// conv1d channels = q | k | v of in_proj_qkv.
    pub fn conv_dim(&self) -> usize { 2 * self.dn_k_dim() + self.dn_v_dim() }
    /// Fused DeltaNet input rows: qkv | z | b | a.
    pub fn dn_in_rows(&self) -> usize { self.conv_dim() + self.dn_v_dim() + 2 * self.dn_vh }
    /// Fused attention input rows: q_proj (q | gate per head) | k | v.
    pub fn attn_in_rows(&self) -> usize { 2 * self.q_dim() + 2 * self.kv_dim() }
    pub fn n_attention(&self) -> usize { self.kinds.iter().filter(|k| **k == LayerKind::Attention).count() }
    pub fn n_deltanet(&self) -> usize { self.layers - self.n_attention() }
}
