//! MLX 4-bit weight loader for Qwen dense models (`model.safetensors[.index.json]`, every linear as
//! `weight` u32 [R][C/8] + `scales` / `biases` [R][C/group], group from config `quantization`).
//!
//! Two steps: `layout()` decides where every tensor goes (offsets relative to the start of the weight region) and
//! returns the copy jobs; `load_into()` runs them into the caller's memory. Both go through `TensorSource`, so the
//! same code loads a real model dir (`SafeSource`: safetensors header index from streamdense-core) and the synthetic
//! in-memory weights of the tests (`MemSource`).
//!
//! What the layout does besides copying:
//! - q / k / v are FUSED row-wise into one matrix `qkv` (weight, scales and biases each as q rows | k rows | v rows),
//!   so one K-row gemv writes p = [T][q | k | v], the layout attn128_prep reads. Their biases (Qwen2 / 2.5) are fused
//!   the same way into one bf16 vector.
//! - F16 (and F32) tensors (Qwen2.5 MLX conversions keep the HF f16 dtype for scales / biases / norms) are converted
//!   to bf16, which every kernel reads (round to nearest even).
//! - Tied embeddings: lm_head = the 4-bit embed_tokens matrix (no copy).
//! - Unquantized embedding (MLX conversions that keep `embed_tokens.weight` F16 / BF16 [vocab][hidden], no scales /
//!   biases, e.g. gemma-1.1-2b-it-4bit): stored as bf16 [vocab][hidden] (`Weights::embed_bf16`); the lookup copies
//!   rows and a tied lm_head is a dense bf16 gemv over it. Every other linear stays 4-bit.
//! - `norm_plus_one` (Gemma): every RMSNorm weight (input / post-attention / pre- and post-feedforward / final / q_norm /
//!   k_norm) is stored as bf(1 + w), as MLX computes `1.0 + weight` in bf16 before `mx.fast.rms_norm`.
//! - Optional tensors by config: o_proj.bias (`o_bias`), mlp.{gate,up,down}_proj.bias (`mlp_bias`), the Gemma 2 / 3
//!   `pre_feedforward_layernorm` / `post_feedforward_layernorm` (`four_norms`; then `post_attention_layernorm` is the
//!   norm on the attention output and `pre_feedforward_layernorm` the FFN input norm).
use std::collections::HashMap;
use std::os::unix::fs::FileExt;

use streamdense_core::f16::f16_to_f32;
use streamdense_core::{bf16_to_f32, f32_to_bf16, DType, SafeIndex};

use crate::config::DenseConfig;

/// Tensor provider: dtype + shape by name, and the raw little-endian bytes.
pub trait TensorSource {
    fn info(&self, name: &str) -> Option<(DType, Vec<usize>)>;
    /// Read the whole tensor into `dst` (`dst.len()` = its byte size).
    fn read(&self, name: &str, dst: &mut [u8]) -> Result<(), String>;
}

/// Safetensors files of a model dir (header index from streamdense-core; data read with pread).
pub struct SafeSource {
    pub idx: SafeIndex,
    files: Vec<std::fs::File>,
}

impl SafeSource {
    /// `dir/model.safetensors.index.json` when present (multi-shard), else `dir/model.safetensors`.
    pub fn open(dir: &str) -> Result<SafeSource, String> {
        let p = std::path::Path::new(dir);
        let idx = if p.join("model.safetensors.index.json").exists() {
            SafeIndex::open_mlx_dir(dir)?
        } else {
            SafeIndex::open_file(&p.join("model.safetensors").to_string_lossy())?
        };
        let files = idx.files().iter().map(|f| std::fs::File::open(f).map_err(|e| format!("{}: {}", f, e)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SafeSource { idx, files })
    }
}

impl TensorSource for SafeSource {
    fn info(&self, name: &str) -> Option<(DType, Vec<usize>)> { self.idx.get(name).map(|t| (t.dtype, t.shape.clone())) }
    fn read(&self, name: &str, dst: &mut [u8]) -> Result<(), String> {
        let t = self.idx.get(name).ok_or_else(|| format!("tensor {} missing", name))?;
        if dst.len() != t.nbytes { return Err(format!("tensor {}: {} B wanted, {} B in file", name, dst.len(), t.nbytes)); }
        self.files[t.file].read_exact_at(dst, t.off).map_err(|e| format!("read {}: {}", name, e))
    }
}

/// In-memory tensors (tests, synthetic models).
#[derive(Default)]
pub struct MemSource {
    pub map: HashMap<String, (DType, Vec<usize>, Vec<u8>)>,
}

impl MemSource {
    pub fn insert(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: Vec<u8>) {
        assert_eq!(bytes.len(), shape.iter().product::<usize>() * dtype.size(), "MemSource {}: size", name);
        self.map.insert(name.to_string(), (dtype, shape.to_vec(), bytes));
    }
}

impl TensorSource for MemSource {
    fn info(&self, name: &str) -> Option<(DType, Vec<usize>)> { self.map.get(name).map(|t| (t.0, t.1.clone())) }
    fn read(&self, name: &str, dst: &mut [u8]) -> Result<(), String> {
        let t = self.map.get(name).ok_or_else(|| format!("tensor {} missing", name))?;
        if dst.len() != t.2.len() { return Err(format!("tensor {}: {} B wanted, {} B held", name, dst.len(), t.2.len())); }
        dst.copy_from_slice(&t.2);
        Ok(())
    }
}

/// One 4-bit matrix (offsets into the weight region / arena): W u32 [rows][cols/8], S / B bf16 [rows][cols/64].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Q4 {
    pub w: usize,
    pub s: usize,
    pub b: usize,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct LayerW {
    /// bf16 [hidden]
    pub in_norm: usize,
    /// FFN input norm: post_attention_layernorm (2-norm layout) or pre_feedforward_layernorm (four_norms)
    pub post_norm: usize,
    /// four_norms: post_attention_layernorm (on the attention output) / post_feedforward_layernorm (on the FFN output)
    pub post_attn_norm: Option<usize>,
    pub post_ffw_norm: Option<usize>,
    /// fused q | k | v: rows q_dim + 2 kv_dim, cols hidden
    pub qkv: Q4,
    /// fused q | k | v bias, bf16 [q_dim + 2 kv_dim] (Qwen2 / 2.5)
    pub qkv_bias: Option<usize>,
    /// bf16 [head_dim] (Qwen3 QK-norm)
    pub q_norm: Option<usize>,
    pub k_norm: Option<usize>,
    pub o: Q4,
    pub gate: Q4,
    pub up: Q4,
    pub down: Q4,
    /// bf16 bias vectors: o_proj [hidden], gate / up [inter], down [hidden]
    pub o_bias: Option<usize>,
    pub gate_bias: Option<usize>,
    pub up_bias: Option<usize>,
    pub down_bias: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Weights {
    /// 4-bit, or with `embed_bf16` bf16 [rows][cols] at `embed.w` (`s` / `b` unused)
    pub embed: Q4,
    /// = `embed` when tied
    pub lm_head: Q4,
    /// embed_tokens was F16 / BF16 in the file: dense bf16 lookup, and (tied) a dense bf16 lm_head
    pub embed_bf16: bool,
    pub tied: bool,
    pub final_norm: usize,
    pub layers: Vec<LayerW>,
    /// bytes of the weight region
    pub bytes: usize,
}

impl Weights {
    /// Shift every offset by `base` (the weight region's start in the arena).
    pub fn rebase(&mut self, base: usize) {
        let q = |m: &mut Q4| { m.w += base; m.s += base; m.b += base; };
        q(&mut self.embed);
        q(&mut self.lm_head);
        self.final_norm += base;
        for l in &mut self.layers {
            l.in_norm += base;
            l.post_norm += base;
            for m in [&mut l.qkv, &mut l.o, &mut l.gate, &mut l.up, &mut l.down] { q(m); }
            for o in [&mut l.qkv_bias, &mut l.q_norm, &mut l.k_norm, &mut l.post_attn_norm, &mut l.post_ffw_norm, &mut l.o_bias,
                      &mut l.gate_bias, &mut l.up_bias, &mut l.down_bias].into_iter().flatten() { *o += base; }
        }
    }
}

/// How a source tensor lands in the region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Conv {
    /// raw bytes
    Raw,
    /// F16 / F32 / BF16 -> bf16
    ToBf16,
    /// F16 / F32 / BF16 -> bf16(1 + w) (Gemma RMSNorm weights)
    ToBf16PlusOne,
}

/// Copy job: tensor `name` -> region offset `dst` (bytes after conversion: `len`).
#[derive(Clone, Debug)]
pub struct Job {
    pub name: String,
    pub dst: usize,
    pub len: usize,
    pub conv: Conv,
}

const ALIGN: usize = 256;

struct Lay<'a> {
    src: &'a dyn TensorSource,
    used: usize,
    jobs: Vec<Job>,
}

impl Lay<'_> {
    fn info(&self, name: &str) -> Result<(DType, Vec<usize>), String> {
        self.src.info(name).ok_or_else(|| format!("weights: tensor {} missing", name))
    }
    fn reserve(&mut self, len: usize) -> usize {
        let off = (self.used + ALIGN - 1) / ALIGN * ALIGN;
        self.used = off + len;
        off
    }
    /// Several tensors back to back (concatenated along dim 0) as one region; returns its offset.
    fn cat(&mut self, names: &[String], conv: Conv) -> Result<usize, String> {
        let mut lens = vec![];
        for n in names {
            let (dt, shape) = self.info(n)?;
            let elems: usize = shape.iter().product();
            let len = match conv {
                Conv::Raw => elems * dt.size(),
                Conv::ToBf16 | Conv::ToBf16PlusOne => {
                    if !matches!(dt, DType::BF16 | DType::F16 | DType::F32) {
                        return Err(format!("weights: {} is {:?}, want a float tensor", n, dt));
                    }
                    elems * 2
                }
            };
            lens.push(len);
        }
        let off = self.reserve(lens.iter().sum());
        let mut at = off;
        for (n, len) in names.iter().zip(lens) {
            self.jobs.push(Job { name: n.clone(), dst: at, len, conv: conv.clone() });
            at += len;
        }
        Ok(off)
    }
    fn vec(&mut self, name: &str, n: usize) -> Result<usize, String> { self.vec_c(name, n, Conv::ToBf16) }
    fn vec_c(&mut self, name: &str, n: usize, conv: Conv) -> Result<usize, String> {
        let (_, shape) = self.info(name)?;
        if shape != [n] { return Err(format!("weights: {} shape {:?}, want [{}]", name, shape, n)); }
        self.cat(&[name.to_string()], conv)
    }
    /// 4-bit linear(s) `prefixes` (each `{p}.weight/.scales/.biases`) fused row-wise; every part must have `cols`
    /// input columns. Returns the fused matrix.
    fn q4(&mut self, prefixes: &[String], cols: usize, group: usize) -> Result<Q4, String> {
        if group != 64 { return Err(format!("weights: group size {} (kernels need 64)", group)); }
        let mut rows = 0;
        for p in prefixes {
            let (wd, ws) = self.info(&format!("{}.weight", p))?;
            if wd != DType::U32 || ws.len() != 2 || ws[1] * 8 != cols {
                return Err(format!("weights: {}.weight is {:?} {:?}, want U32 [R, {}] (4-bit, {} cols)", p, wd, ws, cols / 8, cols));
            }
            for sfx in ["scales", "biases"] {
                let (_, ss) = self.info(&format!("{}.{}", p, sfx))?;
                if ss != [ws[0], cols / group] {
                    return Err(format!("weights: {}.{} shape {:?}, want [{}, {}]", p, sfx, ss, ws[0], cols / group));
                }
            }
            rows += ws[0];
        }
        let names = |s: &str| prefixes.iter().map(|p| format!("{}.{}", p, s)).collect::<Vec<_>>();
        let w = self.cat(&names("weight"), Conv::Raw)?;
        let s = self.cat(&names("scales"), Conv::ToBf16)?;
        let b = self.cat(&names("biases"), Conv::ToBf16)?;
        Ok(Q4 { w, s, b, rows, cols })
    }
}

/// Weight layout of `cfg` from `src` (offsets relative to the weight region) + the copy jobs.
pub fn layout(cfg: &DenseConfig, src: &dyn TensorSource) -> Result<(Weights, Vec<Job>), String> {
    let (group, bits) = cfg.quant.ok_or("weights: config has no `quantization` block (want an MLX 4-bit model)")?;
    if bits != 4 { return Err(format!("weights: {}-bit model (only 4-bit)", bits)); }
    let mut l = Lay { src, used: 0, jobs: vec![] };
    let (h, qd, kvd, hd) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.head_dim);
    let en = "model.embed_tokens.weight";
    let embed_bf16 = matches!(l.info(en)?, (DType::F16 | DType::BF16, _));
    let embed = if embed_bf16 {
        let (_, es) = l.info(en)?;
        if es.len() != 2 || es[1] != h { return Err(format!("weights: {} shape {:?}, want [{}, {}]", en, es, cfg.vocab, h)); }
        Q4 { w: l.cat(&[en.to_string()], Conv::ToBf16)?, s: 0, b: 0, rows: es[0], cols: h }
    } else {
        l.q4(&["model.embed_tokens".to_string()], h, group)?
    };
    if embed.rows != cfg.vocab { return Err(format!("weights: embed_tokens has {} rows, vocab_size {}", embed.rows, cfg.vocab)); }
    let nc = if cfg.norm_plus_one { Conv::ToBf16PlusOne } else { Conv::ToBf16 };
    let final_norm = l.vec_c("model.norm.weight", h, nc.clone())?;
    let mut layers = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let p = format!("model.layers.{}", i);
        let a = format!("{}.self_attn", p);
        let mut norm = |n: &str| l.vec_c(&format!("{}.{}.weight", p, n), h, nc.clone());
        let mut lw = LayerW { in_norm: norm("input_layernorm")?, ..Default::default() };
        if cfg.four_norms {
            lw.post_attn_norm = Some(norm("post_attention_layernorm")?);
            lw.post_norm = norm("pre_feedforward_layernorm")?;
            lw.post_ffw_norm = Some(norm("post_feedforward_layernorm")?);
        } else {
            lw.post_norm = norm("post_attention_layernorm")?;
        }
        lw.qkv = l.q4(&[format!("{}.q_proj", a), format!("{}.k_proj", a), format!("{}.v_proj", a)], h, group)?;
        if lw.qkv.rows != qd + 2 * kvd {
            return Err(format!("weights: layer {} q|k|v rows {}, want {} + 2 x {}", i, lw.qkv.rows, qd, kvd));
        }
        if cfg.attention_bias {
            let names = [format!("{}.q_proj.bias", a), format!("{}.k_proj.bias", a), format!("{}.v_proj.bias", a)];
            for (n, want) in names.iter().zip([qd, kvd, kvd]) {
                let (_, s) = l.info(n)?;
                if s != [want] { return Err(format!("weights: {} shape {:?}, want [{}]", n, s, want)); }
            }
            lw.qkv_bias = Some(l.cat(&names, Conv::ToBf16)?);
        }
        if cfg.qk_norm {
            lw.q_norm = Some(l.vec_c(&format!("{}.q_norm.weight", a), hd, nc.clone())?);
            lw.k_norm = Some(l.vec_c(&format!("{}.k_norm.weight", a), hd, nc.clone())?);
        }
        lw.o = l.q4(&[format!("{}.o_proj", a)], qd, group)?;
        lw.gate = l.q4(&[format!("{}.mlp.gate_proj", p)], h, group)?;
        lw.up = l.q4(&[format!("{}.mlp.up_proj", p)], h, group)?;
        lw.down = l.q4(&[format!("{}.mlp.down_proj", p)], cfg.inter, group)?;
        if lw.o.rows != h || lw.gate.rows != cfg.inter || lw.up.rows != cfg.inter || lw.down.rows != h {
            return Err(format!("weights: layer {} o/gate/up/down rows {}/{}/{}/{} do not match the config", i,
                               lw.o.rows, lw.gate.rows, lw.up.rows, lw.down.rows));
        }
        if cfg.o_bias { lw.o_bias = Some(l.vec(&format!("{}.o_proj.bias", a), h)?); }
        if cfg.mlp_bias {
            lw.gate_bias = Some(l.vec(&format!("{}.mlp.gate_proj.bias", p), cfg.inter)?);
            lw.up_bias = Some(l.vec(&format!("{}.mlp.up_proj.bias", p), cfg.inter)?);
            lw.down_bias = Some(l.vec(&format!("{}.mlp.down_proj.bias", p), h)?);
        }
        layers.push(lw);
    }
    let (lm_head, tied) = if cfg.tie_word_embeddings || src.info("lm_head.weight").is_none() {
        if !cfg.tie_word_embeddings { return Err("weights: no lm_head and tie_word_embeddings = false".into()); }
        (embed, true)
    } else {
        let m = l.q4(&["lm_head".to_string()], h, group)?;
        if m.rows != cfg.vocab { return Err(format!("weights: lm_head has {} rows, vocab_size {}", m.rows, cfg.vocab)); }
        (m, false)
    };
    let bytes = (l.used + ALIGN - 1) / ALIGN * ALIGN;
    Ok((Weights { embed, lm_head, tied, embed_bf16, final_norm, layers, bytes }, l.jobs))
}

/// Run the jobs into `region` (the weight region: `Weights::bytes` long, offsets as `layout` returned them).
/// F16 / F32 sources are read into a temporary buffer (one tensor at a time) and converted to bf16.
pub fn load_into(src: &dyn TensorSource, jobs: &[Job], region: &mut [u8]) -> Result<(), String> {
    let mut tmp: Vec<u8> = Vec::new();
    for j in jobs {
        let dst = region.get_mut(j.dst..j.dst + j.len).ok_or_else(|| format!("weights: job {} out of region", j.name))?;
        let (dt, _) = src.info(&j.name).ok_or_else(|| format!("weights: tensor {} missing", j.name))?;
        let add = if j.conv == Conv::ToBf16PlusOne { 1.0f32 } else { 0.0 };
        match (&j.conv, dt) {
            (Conv::Raw, _) | (Conv::ToBf16, DType::BF16) => src.read(&j.name, dst)?,
            (Conv::ToBf16PlusOne, DType::BF16) => {
                src.read(&j.name, dst)?;
                for o in dst.chunks_exact_mut(2) {
                    let v = f32_to_bf16(1.0 + bf16_to_f32(u16::from_le_bytes([o[0], o[1]])));
                    o.copy_from_slice(&v.to_le_bytes());
                }
            }
            (Conv::ToBf16 | Conv::ToBf16PlusOne, DType::F16) => {
                tmp.resize(j.len, 0);
                src.read(&j.name, &mut tmp)?;
                for (o, i) in dst.chunks_exact_mut(2).zip(tmp.chunks_exact(2)) {
                    let v = f32_to_bf16(add + f16_to_f32(u16::from_le_bytes([i[0], i[1]])));
                    o.copy_from_slice(&v.to_le_bytes());
                }
            }
            (Conv::ToBf16 | Conv::ToBf16PlusOne, DType::F32) => {
                tmp.resize(j.len * 2, 0);
                src.read(&j.name, &mut tmp)?;
                for (o, i) in dst.chunks_exact_mut(2).zip(tmp.chunks_exact(4)) {
                    let v = f32_to_bf16(add + f32::from_le_bytes([i[0], i[1], i[2], i[3]]));
                    o.copy_from_slice(&v.to_le_bytes());
                }
            }
            (Conv::ToBf16 | Conv::ToBf16PlusOne, d) => return Err(format!("weights: {} is {:?}, cannot convert to bf16", j.name, d)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_converted_and_fused() {
        let mut s = MemSource::default();
        let f16 = |v: &[f32]| v.iter().flat_map(|x| streamdense_core::f16::f32_to_f16(*x).to_le_bytes()).collect::<Vec<u8>>();
        s.insert("a", DType::F16, &[2], f16(&[1.5, -0.25]));
        s.insert("b", DType::BF16, &[1], f32_to_bf16(3.0).to_le_bytes().to_vec());
        let mut l = Lay { src: &s, used: 0, jobs: vec![] };
        let off = l.cat(&["a".into(), "b".into()], Conv::ToBf16).unwrap();
        let mut region = vec![0u8; l.used];
        load_into(&s, &l.jobs, &mut region).unwrap();
        let v: Vec<f32> = region[off..off + 6].chunks_exact(2)
            .map(|c| streamdense_core::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        assert_eq!(v, vec![1.5, -0.25, 3.0]);
    }
}
