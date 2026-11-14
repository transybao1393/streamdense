//! Model config + tensor index from the earlier 7B research q4 export (index.json). Shapes come from the config, never
//! hard-coded; the kernels' own assumption (head_dim 128) is checked here.
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act { Relu, Silu }

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub act: Act,
    pub bos: u32,
    pub eos: u32,
    pub q_scale_folded: Option<f64>,
}
impl Config {
    pub fn qdim(&self) -> usize { self.heads * self.head_dim }
    pub fn kvdim(&self) -> usize { self.kv_heads * self.head_dim }
    pub fn from_json(c: &Value) -> Result<Config, String> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config: missing {}", k));
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let cfg = Config {
            hidden,
            inter: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            heads,
            kv_heads: c["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(heads),
            head_dim: c["head_dim"].as_u64().map(|v| v as usize).unwrap_or(hidden / heads),
            vocab: u("vocab_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: c["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            act: match c["hidden_act"].as_str().unwrap_or("relu") { "silu" | "swish" => Act::Silu, _ => Act::Relu },
            bos: c["bos_token_id"].as_u64().unwrap_or(1) as u32,
            eos: c["eos_token_id"].as_u64().unwrap_or(2) as u32,
            q_scale_folded: c["streamdense_q_scale_folded"].as_f64(),
        };
        if cfg.head_dim != 128 { return Err(format!("kernels assume head_dim 128 (got {})", cfg.head_dim)); }
        if cfg.heads % cfg.kv_heads != 0 { return Err("num_attention_heads must be a multiple of num_key_value_heads".into()); }
        if cfg.hidden % 64 != 0 || cfg.inter % 64 != 0 || cfg.qdim() % 64 != 0 { return Err("dims must be multiples of 64 (4-bit g64)".into()); }
        Ok(cfg)
    }
}

#[derive(Clone, Debug)]
pub struct Tensor { pub off: u64, pub nbytes: usize, pub shape: Vec<usize>, pub dtype: String }

pub struct Index { pub cfg: Config, pub tensors: HashMap<String, Tensor>, pub bin: String, pub raw: Value }
impl Index {
    pub fn open(dir: &str) -> Result<Index, String> {
        let txt = std::fs::read_to_string(format!("{}/index.json", dir)).map_err(|e| format!("{}/index.json: {}", dir, e))?;
        let raw: Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        let cfg = Config::from_json(&raw["config"])?;
        let mut tensors = HashMap::new();
        for (k, t) in raw["tensors"].as_object().ok_or("index: no tensors")? {
            tensors.insert(k.clone(), Tensor {
                off: t["offset"].as_u64().unwrap(),
                nbytes: t["nbytes"].as_u64().unwrap() as usize,
                shape: t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect(),
                dtype: t["dtype"].as_str().unwrap_or("").to_string(),
            });
        }
        Ok(Index { cfg, tensors, bin: format!("{}/model_q4.bin", dir), raw })
    }
    pub fn t(&self, name: &str) -> Result<&Tensor, String> { self.tensors.get(name).ok_or(format!("missing tensor {}", name)) }
    pub fn q4_bytes(&self, name: &str) -> Result<usize, String> {
        Ok(self.t(&format!("{}.w", name))?.nbytes + self.t(&format!("{}.s", name))?.nbytes + self.t(&format!("{}.b", name))?.nbytes)
    }
}
