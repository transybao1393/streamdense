//! Qwen3.5/3.8 weights: budget rows, arena placement and parallel F_NOCACHE load straight from the MLX safetensors
//! (no conversion). One walker serves both `plan()` (header bytes only) and `load()` (allocate + pread), so the
//! budget and the arena always agree. Layout rules (plan section 3):
//! - row-concatenated groups: DeltaNet `dn_in` = in_proj_qkv | in_proj_z | in_proj_b | in_proj_a, attention
//!   `attn_in` = q_proj | k_proj | v_proj; each group is 3 regions (W, S, B), each part pread at its row offset;
//! - MTP norms (all 7) get +1 at load, main-file norms are used as stored; `linear_attn.norm` never gets +1;
//! - sanity check: main input_layernorm mean > 0.5, MTP pre_fc_norm_embedding mean (raw) < 0.5;
//! - embedding rows are not loaded: `EmbedSrc` keeps the file and the 3 offsets.
use streamdense_core::arena::{Arena, Region};
use streamdense_core::budget::Budget;
use streamdense_core::io::NoCacheFile;
use streamdense_core::{bf16_to_f32, f32_to_bf16, DType, SafeIndex, TensorInfo};

use crate::qwen_config::{LayerKind, QwenConfig};

/// MLX affine 4-bit g64 matrix [rows, cols]: W u32 [rows][cols/8], S and B bf16 [rows][cols/64].
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4b {
    pub rows: usize,
    pub cols: usize,
    pub w: Region,
    pub s: Region,
    pub b: Region,
}

/// Gated DeltaNet mixer. `in_proj` rows = qkv | z | b | a (output offsets 0, conv_dim, conv_dim + v_dim, + vh).
#[derive(Clone, Copy, Debug, Default)]
pub struct DnW {
    pub in_proj: Q4b,
    pub conv_w: Region,
    pub a_log: Region,
    pub dt_bias: Region,
    pub norm_w: Region,
    pub out_proj: Q4b,
}

/// Gated full attention. `in_proj` rows = q_proj (q | gate interleaved per head) | k_proj | v_proj.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnW {
    pub in_proj: Q4b,
    pub q_norm: Region,
    pub k_norm: Region,
    pub o_proj: Q4b,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FfnW {
    pub gate: Q4b,
    pub up: Q4b,
    pub down: Q4b,
    /// forheader compact FFN (ffn_compact.rs; None = today's full layer): gate / up hold only the kept rows, `down`
    /// the kept columns repacked (cols = kc) with the original S / B; this holds the group map.
    pub cd: Option<crate::ffn_compact::CDown>,
}

#[derive(Clone, Copy, Debug)]
pub enum Mixer {
    Dn(DnW),
    Attn(AttnW),
}

#[derive(Clone, Copy, Debug)]
pub struct LayerW {
    pub in_norm: Region,
    pub post_norm: Region,
    pub mixer: Mixer,
    pub ffn: FfnW,
}

/// MTP head: fc bf16 [hidden][2 * hidden] (input = emb half first), the 2 pre-fc norms, the final norm and one
/// full-attention layer. All norms here already have +1 applied.
#[derive(Clone, Copy, Debug)]
pub struct MtpW {
    pub fc: Region,
    pub pre_e: Region,
    pub pre_h: Region,
    pub norm: Region,
    pub layer: LayerW,
}

/// Where the 4-bit embedding rows live on SSD (absolute file offsets of the weight / scales / biases tensors).
#[derive(Clone, Debug, Default)]
pub struct EmbedSrc {
    pub file: String,
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
}

#[derive(Clone, Debug)]
pub struct Weights {
    pub layers: Vec<LayerW>,
    pub final_norm: Region,
    pub lm_head: Q4b,
    pub mtp: Option<MtpW>,
    pub embed: EmbedSrc,
}

// Budget rows (plan section 2), in table order.
const C_FFN: &str = "FFN gate/up/down (4-bit)";
const C_DNP: &str = "DeltaNet proj (4-bit)";
const C_DNS: &str = "DeltaNet small (conv, A, dt, norm)";
const C_ATT: &str = "attention q/k/v/o (4-bit)";
const C_NORM: &str = "norms";
const C_HEAD: &str = "lm_head (4-bit)";
const C_MTP: &str = "MTP layer (fc bf16 + 4-bit)";
const CATS: [&str; 7] = [C_FFN, C_DNP, C_DNS, C_ATT, C_NORM, C_HEAD, C_MTP];

const MAIN: &str = "language_model.model.";

/// Walks the tensor list. Without an arena it only checks headers and counts bytes per budget row; with one it
/// also allocates each region and queues the pread jobs (per file).
struct Walk<'a> {
    idx: &'a SafeIndex,
    arena: Option<&'a mut Arena>,
    jobs: Vec<Vec<(usize, usize, u64)>>,
    bytes: [u64; 7],
    cat: usize,
    /// MTP: every norm region placed while true gets +1 after the read
    mtp_norms: bool,
    plus_one: Vec<Region>,
    /// forheader compact FFN: kept neurons per main layer (None = full FFN) and the gather jobs it produced
    compact: Option<&'a crate::ffn_compact::CompactSet>,
    cjobs: Vec<crate::ffn_compact::CJob>,
}

impl<'a> Walk<'a> {
    fn new(idx: &'a SafeIndex, arena: Option<&'a mut Arena>) -> Self {
        Walk { idx, arena, jobs: vec![Vec::new(); idx.files().len()], bytes: [0; 7], cat: 0, mtp_norms: false, plus_one: Vec::new(),
               compact: None, cjobs: Vec::new() }
    }
    fn set_cat(&mut self, c: &str) { self.cat = CATS.iter().position(|x| *x == c).unwrap(); }

    fn info(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<&'a TensorInfo, String> {
        let t = self.idx.get(name).ok_or_else(|| format!("weights: tensor {} not found", name))?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!("weights: {} is {:?} {:?}, expected {:?} {:?}", name, t.dtype, t.shape, dtype, shape));
        }
        Ok(t)
    }

    fn alloc(&mut self, label: &str, len: usize) -> Result<Region, String> {
        self.bytes[self.cat] += len as u64;
        match self.arena.as_deref_mut() {
            Some(a) => a.alloc(&format!("w:{}", label), len),
            None => Ok(Region::default()),
        }
    }
    fn job(&mut self, dst: Region, sub: usize, t: &TensorInfo) {
        if let Some(a) = self.arena.as_deref() {
            self.jobs[t.file].push((a.base() as usize + dst.off + sub, t.nbytes, t.off));
        }
    }

    /// One tensor into its own region.
    fn raw(&mut self, name: &str, dtype: DType, shape: &[usize]) -> Result<Region, String> {
        let t = self.info(name, dtype, shape)?;
        let r = self.alloc(name, t.nbytes)?;
        self.job(r, 0, t);
        Ok(r)
    }

    /// A bf16 norm weight [n]; under `mtp_norms` it is marked for +1.
    fn norm(&mut self, name: &str, n: usize) -> Result<Region, String> {
        let r = self.raw(name, DType::BF16, &[n])?;
        if self.mtp_norms { self.plus_one.push(r); }
        Ok(r)
    }

    /// Row-concatenated 4-bit group: parts `(prefix, rows)` (tensor names `prefix.{weight,scales,biases}`), all with
    /// `cols` input columns, placed back to back in one W, one S and one B region.
    fn q4(&mut self, label: &str, parts: &[(String, usize)], cols: usize) -> Result<Q4b, String> {
        if cols % 64 != 0 { return Err(format!("weights: {} cols {} not a multiple of 64", label, cols)); }
        let (wc, gc) = (cols / 8, cols / 64);
        let mut infos = Vec::with_capacity(parts.len());
        for (p, r) in parts {
            infos.push((
                self.info(&format!("{}.weight", p), DType::U32, &[*r, wc])?,
                self.info(&format!("{}.scales", p), DType::BF16, &[*r, gc])?,
                self.info(&format!("{}.biases", p), DType::BF16, &[*r, gc])?,
            ));
        }
        let rows: usize = parts.iter().map(|p| p.1).sum();
        let w = self.alloc(&format!("{}.w", label), rows * wc * 4)?;
        let s = self.alloc(&format!("{}.s", label), rows * gc * 2)?;
        let b = self.alloc(&format!("{}.b", label), rows * gc * 2)?;
        let mut r0 = 0;
        for (&(tw, ts, tb), (_, r)) in infos.iter().zip(parts) {
            self.job(w, r0 * wc * 4, tw);
            self.job(s, r0 * gc * 2, ts);
            self.job(b, r0 * gc * 2, tb);
            r0 += r;
        }
        Ok(Q4b { rows, cols, w, s, b })
    }

    /// One 4-bit matrix [rows, cols] in its own group.
    fn q4one(&mut self, prefix: &str, rows: usize, cols: usize) -> Result<Q4b, String> {
        self.q4(prefix, &[(prefix.to_string(), rows)], cols)
    }

    /// One decoder layer under `pre` (e.g. `language_model.model.layers.7.` or `mtp.layers.0.`).
    /// `cat_override`: put every byte of this layer into one budget row (the MTP layer).
    fn layer(&mut self, cfg: &QwenConfig, pre: &str, kind: LayerKind, cat_override: Option<&str>) -> Result<LayerW, String> {
        self.layer_ex(cfg, pre, kind, cat_override, None)
    }

    /// `layer` with an optional compact FFN (`main`: the main layer index whose kept set to use).
    fn layer_ex(&mut self, cfg: &QwenConfig, pre: &str, kind: LayerKind, cat_override: Option<&str>, main: Option<usize>) -> Result<LayerW, String> {
        let h = cfg.hidden;
        let cat = |w: &mut Self, c: &str| w.set_cat(cat_override.unwrap_or(c));
        cat(self, C_NORM);
        let in_norm = self.norm(&format!("{}input_layernorm.weight", pre), h)?;
        let post_norm = self.norm(&format!("{}post_attention_layernorm.weight", pre), h)?;
        let mixer = match kind {
            LayerKind::DeltaNet => {
                let p = format!("{}linear_attn.", pre);
                cat(self, C_DNP);
                let in_proj = self.q4(&format!("{}dn_in", p), &[
                    (format!("{}in_proj_qkv", p), cfg.conv_dim()),
                    (format!("{}in_proj_z", p), cfg.dn_v_dim()),
                    (format!("{}in_proj_b", p), cfg.dn_vh),
                    (format!("{}in_proj_a", p), cfg.dn_vh),
                ], h)?;
                let out_proj = self.q4one(&format!("{}out_proj", p), h, cfg.dn_v_dim())?;
                cat(self, C_DNS);
                let conv_w = self.raw(&format!("{}conv1d.weight", p), DType::BF16, &[cfg.conv_dim(), cfg.conv_k, 1])?;
                let a_log = self.raw(&format!("{}A_log", p), DType::BF16, &[cfg.dn_vh])?;
                let dt_bias = self.raw(&format!("{}dt_bias", p), DType::BF16, &[cfg.dn_vh])?;
                // gated RMSNorm(dv): stored raw, used without +1 (never in plus_one, also for MTP)
                let norm_w = self.raw(&format!("{}norm.weight", p), DType::BF16, &[cfg.dn_dv])?;
                Mixer::Dn(DnW { in_proj, conv_w, a_log, dt_bias, norm_w, out_proj })
            }
            LayerKind::Attention => {
                let p = format!("{}self_attn.", pre);
                cat(self, C_ATT);
                let in_proj = self.q4(&format!("{}attn_in", p), &[
                    (format!("{}q_proj", p), 2 * cfg.q_dim()),
                    (format!("{}k_proj", p), cfg.kv_dim()),
                    (format!("{}v_proj", p), cfg.kv_dim()),
                ], h)?;
                let o_proj = self.q4one(&format!("{}o_proj", p), h, cfg.q_dim())?;
                cat(self, C_NORM);
                let q_norm = self.norm(&format!("{}q_norm.weight", p), cfg.head_dim)?;
                let k_norm = self.norm(&format!("{}k_norm.weight", p), cfg.head_dim)?;
                Mixer::Attn(AttnW { in_proj, q_norm, k_norm, o_proj })
            }
        };
        cat(self, C_FFN);
        let m = format!("{}mlp.", pre);
        let keep = match (self.compact, main) { (Some(c), Some(l)) => Some(c.keep[l].len()), _ => None };
        let ffn = match keep {
            Some(kc) => self.ffn_compact(cfg, &m, kc, main.unwrap())?,
            None => FfnW {
                gate: self.q4one(&format!("{}gate_proj", m), cfg.inter, h)?,
                up: self.q4one(&format!("{}up_proj", m), cfg.inter, h)?,
                down: self.q4one(&format!("{}down_proj", m), h, cfg.inter)?,
                cd: None,
            },
        };
        Ok(LayerW { in_norm, post_norm, mixer, ffn })
    }

    /// Compact FFN of main layer `l` with `kc` kept neurons (ffn_compact.rs): regions sized for kc, filled later by
    /// `ffn_compact::gather` (gate / up rows, down columns); the down S / B are read whole by plain jobs.
    fn ffn_compact(&mut self, cfg: &QwenConfig, m: &str, kc: usize, l: usize) -> Result<FfnW, String> {
        use crate::ffn_compact::{CDown, CJob, CSrc};
        let (h, n) = (cfg.hidden, cfg.inter);
        if n % 64 != 0 || h % 64 != 0 { return Err("ffn_compact: hidden / inter not multiples of 64".into()); }
        let src = |w: &Self, p: &str, rows: usize, cols: usize| -> Result<[CSrc; 3], String> {
            let t = [w.info(&format!("{}.weight", p), DType::U32, &[rows, cols / 8])?,
                     w.info(&format!("{}.scales", p), DType::BF16, &[rows, cols / 64])?,
                     w.info(&format!("{}.biases", p), DType::BF16, &[rows, cols / 64])?];
            Ok(t.map(|x| CSrc { file: x.file, off: x.off }))
        };
        let (gp, upn, dp) = (format!("{}gate_proj", m), format!("{}up_proj", m), format!("{}down_proj", m));
        let gate_s = src(self, &gp, n, h)?;
        let up_s = src(self, &upn, n, h)?;
        let down_s = src(self, &dp, h, n)?;
        let rows_q4 = |w: &mut Self, label: &str| -> Result<Q4b, String> {
            Ok(Q4b { rows: kc, cols: h, w: w.alloc(&format!("{}.cw", label), kc * h / 2)?, s: w.alloc(&format!("{}.cs", label), kc * h / 64 * 2)?,
                     b: w.alloc(&format!("{}.cb", label), kc * h / 64 * 2)? })
        };
        let gate = rows_q4(self, &gp)?;
        let up = rows_q4(self, &upn)?;
        let dw = self.alloc(&format!("{}.cw", dp), h * kc / 2)?;
        let ts = self.info(&format!("{}.scales", dp), DType::BF16, &[h, n / 64])?;
        let tb = self.info(&format!("{}.biases", dp), DType::BF16, &[h, n / 64])?;
        let ds = self.alloc(&format!("{}.s", dp), ts.nbytes)?;
        let db = self.alloc(&format!("{}.b", dp), tb.nbytes)?;
        self.job(ds, 0, ts);
        self.job(db, 0, tb);
        let gid = self.alloc(&format!("{}.gid", dp), kc * 2)?;
        let gstart = self.alloc(&format!("{}.gstart", dp), (n / 64 + 1) * 4)?;
        let wmeta = self.alloc(&format!("{}.wmeta", dp), kc / 8 * 4)?;
        let ffn = FfnW { gate, up, down: Q4b { rows: h, cols: kc, w: dw, s: ds, b: db }, cd: Some(CDown { kc, groups: n / 64, gid, gstart, wmeta, mma: false }) };
        self.cjobs.push(CJob { layer: l, gate: gate_s, up: up_s, down_w: down_s[0], ffn });
        Ok(ffn)
    }

    /// Everything resident: 64 layers, final norm, lm_head, MTP (optional). Embedding: offsets only.
    fn all(&mut self, cfg: &QwenConfig, mtp: bool) -> Result<Weights, String> {
        let h = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            layers.push(self.layer_ex(cfg, &format!("{}layers.{}.", MAIN, l), cfg.kinds[l], None, Some(l))?);
        }
        self.set_cat(C_NORM);
        let final_norm = self.norm(&format!("{}norm.weight", MAIN), h)?;
        self.set_cat(C_HEAD);
        let lm_head = self.q4one("language_model.lm_head", cfg.vocab, h)?;

        let e = |s: &str| format!("{}embed_tokens.{}", MAIN, s);
        let ew = self.info(&e("weight"), DType::U32, &[cfg.vocab, h / 8])?;
        let es = self.info(&e("scales"), DType::BF16, &[cfg.vocab, h / 64])?;
        let eb = self.info(&e("biases"), DType::BF16, &[cfg.vocab, h / 64])?;
        if ew.file != es.file || ew.file != eb.file { return Err("weights: embed_tokens weight/scales/biases in different files".into()); }
        let embed = EmbedSrc { file: self.idx.files()[ew.file].clone(), w_off: ew.off, s_off: es.off, b_off: eb.off };

        let mtp_w = if mtp {
            if cfg.mtp_layers != 1 { return Err(format!("weights: MTP needs mtp_num_hidden_layers = 1 (got {})", cfg.mtp_layers)); }
            self.set_cat(C_MTP);
            self.mtp_norms = true;
            let fc = self.raw("mtp.fc.weight", DType::BF16, &[h, 2 * h])?;
            let pre_e = self.norm("mtp.pre_fc_norm_embedding.weight", h)?;
            let pre_h = self.norm("mtp.pre_fc_norm_hidden.weight", h)?;
            let norm = self.norm("mtp.norm.weight", h)?;
            let layer = self.layer(cfg, "mtp.layers.0.", LayerKind::Attention, Some(C_MTP))?;
            self.mtp_norms = false;
            Some(MtpW { fc, pre_e, pre_h, norm, layer })
        } else { None };
        Ok(Weights { layers, final_norm, lm_head, mtp: mtp_w, embed })
    }
}

/// Add the section 2 weight rows to the budget, from the real headers (no tensor data read). Returns the total
/// weight bytes (before arena alignment).
pub fn plan(idx: &SafeIndex, cfg: &QwenConfig, mtp: bool, bud: &mut Budget) -> Result<u64, String> {
    plan_ex(idx, cfg, mtp, None, bud)
}

/// `plan` with an optional compact FFN (ffn_compact.rs): the FFN row then counts only the kept neurons.
pub fn plan_ex(idx: &SafeIndex, cfg: &QwenConfig, mtp: bool, compact: Option<&crate::ffn_compact::CompactSet>, bud: &mut Budget) -> Result<u64, String> {
    let mut w = Walk::new(idx, None);
    w.compact = compact;
    w.all(cfg, mtp)?;
    for (i, c) in CATS.iter().enumerate() {
        if w.bytes[i] > 0 { bud.add(c, w.bytes[i]); }
    }
    Ok(w.bytes.iter().sum())
}

fn bf16_mean(arena: &Arena, r: Region) -> f64 {
    let v: &mut [u16] = arena.slice_mut(r);
    v.iter().map(|&x| bf16_to_f32(x) as f64).sum::<f64>() / v.len().max(1) as f64
}

/// Allocate every weight region in `arena` and pread it (F_NOCACHE, `threads` threads per file). Applies +1 to the
/// MTP norms and checks the norm convention. Returns the placed weights and the bytes read.
pub fn load(idx: &SafeIndex, cfg: &QwenConfig, arena: &mut Arena, mtp: bool, threads: usize) -> Result<(Weights, u64), String> {
    let (w, n, _) = load_ex(idx, cfg, arena, mtp, threads, None)?;
    Ok((w, n))
}

/// `load` with an optional compact FFN (ffn_compact.rs): the kept rows / columns are gathered after the plain reads.
/// Also returns the gather jobs (empty without compact) for in-place reloads.
pub fn load_ex(idx: &SafeIndex, cfg: &QwenConfig, arena: &mut Arena, mtp: bool, threads: usize, compact: Option<&crate::ffn_compact::CompactSet>)
               -> Result<(Weights, u64, Vec<crate::ffn_compact::CJob>), String> {
    let (mut weights, jobs, plus_one, cjobs) = {
        let mut w = Walk::new(idx, Some(arena));
        w.compact = compact;
        let weights = w.all(cfg, mtp)?;
        (weights, w.jobs, w.plus_one, w.cjobs)
    };
    let mut loaded = 0u64;
    for (fi, js) in jobs.iter().enumerate() {
        if js.is_empty() { continue; }
        let f = NoCacheFile::open(&idx.files()[fi])?;
        loaded += f.read_many(js, threads, 8 << 20)?;
    }
    // compact + MTP: the bit-identical MMA / twin down family (ffn_compact.rs CDown.mma)
    for l in weights.layers.iter_mut() { if let Some(cd) = l.ffn.cd.as_mut() { cd.mma = mtp; } }
    if let Some(c) = compact {
        loaded += crate::ffn_compact::gather(idx.files(), &cjobs, c, cfg.hidden, cfg.inter, arena.base(), threads)?;
    }

    // norm convention: main norms stored with +1 (mean well above 0.5), MTP norms raw (mean below 0.5)
    let main_mean = weights.layers.first().map(|l| bf16_mean(arena, l.in_norm)).unwrap_or(1.0);
    if main_mean <= 0.5 {
        return Err(format!("norm convention mismatch: main input_layernorm mean {:.4} <= 0.5 (expected +1 already folded)", main_mean));
    }
    if let Some(m) = &weights.mtp {
        let pre_mean = bf16_mean(arena, m.pre_e);
        if pre_mean >= 0.5 {
            return Err(format!("norm convention mismatch: MTP pre_fc_norm_embedding mean {:.4} >= 0.5 (expected raw, +1 at load)", pre_mean));
        }
        for r in &plus_one {
            for x in arena.slice_mut::<u16>(*r).iter_mut() { *x = f32_to_bf16(bf16_to_f32(*x) + 1.0); }
        }
    }
    Ok((weights, loaded, cjobs))
}
