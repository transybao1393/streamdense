//! DFlash block drafter (2026-11-03): a small drafter trained separately (
//! `DFlashDraft`) proposes a whole block of B - 1 tokens in ONE forward; the target checks [pending, D1..D(B-1)] in one
//! exact verify pass of K = B rows (<= 8). Every emitted token is the verify pass's argmax, so the output is plain
//! greedy's whatever the drafts (lossless). Off unless `QwenOptions.dflash` names a drafter directory.
//!
//! Drafter (kit `dflash_common.DFlashDraft`, exported by `export_engine.py` to `dflash.json` + `dflash.safetensors`):
//!   taps     the target's residual stream after layers layer_ids[i] - 1 (HF hidden_states[layer_ids[i]]), per position;
//!            optional per-tap RMSNorm (tap_norm, the learned per-tap scale folded into its weight by the exporter);
//!   context  feature = ctx_norm(fc(concat taps)); every draft layer turns it into its own K / V (its k_proj / v_proj,
//!            k_norm, RoPE at the context position) = the drafter's context KV, one row per target position;
//!   block    rows = [embed(pending), mask_emb x (B - 1)] at positions a .. a + B - 1 (a = the pending token's position);
//!            n_layers gated full-attention layers (target full_attention semantics: q | gate per head, q / k norm,
//!            partial RoPE, GQA, sigmoid gate) attending to context keys 0 .. a - 1 and to the whole block
//!            (bidirectional); then the drafter's final norm and the TARGET's lm_head (shared, 0 extra bytes) on rows
//!            1 .. B - 1 -> drafts D1 .. D(B-1) (argmax).
//! Engine side:
//!   capture  only with a drafter loaded: every target pass that goes through `encode_pass_full` (prefill chunks, plain
//!            decode) or this module's verify copies the tap rows (`DFlash::tap`, after each tap layer) and, at the end
//!            of the same command buffer, turns them into context KV rows at the pass positions (`DFlash::absorb`).
//!            No CPU work and no extra command buffer; nothing at all without a drafter. Rows past the accepted ones
//!            leave stale context rows at positions >= the next anchor, which the drafter never reads (keys < a) and
//!            the next pass overwrites (as the target's own KV).
//!   loop     `feed_stream_dflash` (the `feed_stream` contract): round at position p with pending token t:
//!            draft CB (embed(t) from staging slot 1, block forward, argmax) -> drafts; verify = the context-draft HOLD
//!            pass of K = d + 1 rows (state only read, rows saved; ctx_draft.rs) + taps + absorb; decide j (longest
//!            prefix D_{i+1} == a_i, a_i not eos); emit a_0 .. a_je; replay rows 0 .. je into the DeltaNet state
//!            (spec_deep `encode_replay_from`, bit-identical to feeding them one by one); pending = a_je.
//!   memory   drafter weights (4-bit g64 like the target) + context KV (n_layers x 2 x kv_heads x max_ctx x head_dim
//!            bf16) + scratch (tap rows for tmax rows, block rows, B - 1 logits rows), all arena budget rows; the hold
//!            save is sized for B rows.
//! Not covered (drafts still lossless, only weaker): passes that bypass `encode_pass_full` (streamed prefill, snapshot
//! restore, the MTP / context-draft loops) leave no context rows; the drafter then sees only the contiguous prefix it
//! has (`DfStats::short_ctx` counts such rounds).
use std::cell::Cell;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use streamdense_core::arena::{Arena, Region};
use streamdense_core::{DType, SafeIndex, TensorInfo};
use streamdense_metal::{cb_signal, cb_wait, commit, gpu_ms, wait, CmdBuf, Enc, Gpu};

use crate::qwen_config::QwenConfig;
use crate::qwen_engine::{QwenEngine, StreamStats};
use crate::qwen_layers::{encode_head, encode_layer, gemv_m, swiglu_m, PassIo};
use crate::weights::Q4b;

const BF: usize = 2;
/// Engine export format of the separately trained drafter.
pub const FORMAT: &str = "streamdense-dflash-q4";
pub const CONFIG_FILE: &str = "dflash.json";
pub const WEIGHTS_FILE: &str = "dflash.safetensors";
/// Largest block: the verify pass has at most 8 rows (EXACT_KS, logits scratch).
pub const BLOCK_MAX: usize = 8;
/// Rows per absorb matmul (the K-row GEMVs take 1..=8 rows).
const ABSORB_ROWS: usize = 8;

/// The drafter's config (`dflash.json`: the kit's dflash_config.json + the export fields).
#[derive(Clone, Debug, PartialEq)]
pub struct DfConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot_dims: usize,
    pub inter: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub vocab: usize,
    pub n_layers: usize,
    pub block: usize,
    /// HF hidden_states indices (output of target layer i - 1)
    pub layer_ids: Vec<usize>,
    pub tap_norm: bool,
}
impl DfConfig {
    pub fn q_dim(&self) -> usize { self.heads * self.head_dim }
    pub fn kv_dim(&self) -> usize { self.kv_heads * self.head_dim }
    /// in_proj rows: q | gate per head, then k, then v
    pub fn prow(&self) -> usize { 2 * self.q_dim() + 2 * self.kv_dim() }
    pub fn n_taps(&self) -> usize { self.layer_ids.len() }
    /// fc input width
    pub fn fw(&self) -> usize { self.n_taps() * self.hidden }
    /// target layer after which tap i is taken
    pub fn tap_layer(&self, i: usize) -> usize { self.layer_ids[i] - 1 }

    pub fn from_json(v: &serde_json::Value) -> Result<DfConfig, String> {
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("dflash.json: no integer {:?}", k));
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("dflash.json: no number {:?}", k));
        if let Some(fm) = v.get("format").and_then(|x| x.as_str()) {
            if fm != FORMAT { return Err(format!("dflash.json: format {:?}, want {:?} (export_engine.py)", fm, FORMAT)); }
        } else {
            return Err(format!("dflash.json: no \"format\" (a kit training dir? export it with export_engine.py to {:?})", FORMAT));
        }
        let layer_ids: Vec<usize> = v.get("layer_ids").and_then(|x| x.as_array()).ok_or("dflash.json: no layer_ids")?
            .iter().map(|x| x.as_u64().map(|y| y as usize).ok_or("dflash.json: layer_ids not integers")).collect::<Result<_, _>>()?;
        Ok(DfConfig { hidden: u("hidden")?, heads: u("heads")?, kv_heads: u("kv_heads")?, head_dim: u("head_dim")?, rot_dims: u("rot_dims")?,
                      inter: u("intermediate")?, eps: f("eps")? as f32, rope_theta: f("rope_theta")?, vocab: u("vocab")?,
                      n_layers: u("n_layers")?, block: u("block")?, layer_ids,
                      tap_norm: v.get("tap_norm").and_then(|x| x.as_bool()).unwrap_or(false) })
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"format": FORMAT, "bits": 4, "group": 64, "hidden": self.hidden, "heads": self.heads, "kv_heads": self.kv_heads,
            "head_dim": self.head_dim, "rot_dims": self.rot_dims, "intermediate": self.inter, "eps": self.eps, "rope_theta": self.rope_theta,
            "vocab": self.vocab, "n_layers": self.n_layers, "block": self.block, "layer_ids": self.layer_ids, "tap_norm": self.tap_norm,
            "layer_kind": "gated_full_attention"})
    }
    /// Shapes the engine kernels and the target need.
    pub fn check(&self, t: &QwenConfig) -> Result<(), String> {
        let e = |m: String| Err(format!("drafter: {}", m));
        if self.hidden != t.hidden { return e(format!("hidden {} != target {}", self.hidden, t.hidden)); }
        if self.vocab != t.vocab { return e(format!("vocab {} != target {} (the target's lm_head is shared)", self.vocab, t.vocab)); }
        if self.rot_dims != t.rope_dims { return e(format!("rot_dims {} != target {} (shared RoPE frequencies)", self.rot_dims, t.rope_dims)); }
        if (self.rope_theta - t.rope_theta).abs() > 1e-6 * t.rope_theta.abs() { return e(format!("rope_theta {} != target {}", self.rope_theta, t.rope_theta)); }
        if !(2..=BLOCK_MAX).contains(&self.block) { return e(format!("block {} not in 2..={}", self.block, BLOCK_MAX)); }
        if self.head_dim % 32 != 0 || self.head_dim > 256 { return e(format!("head_dim {}: want a multiple of 32, <= 256", self.head_dim)); }
        if self.rot_dims > self.head_dim || self.rot_dims % 2 != 0 { return e(format!("rot_dims {} vs head_dim {}", self.rot_dims, self.head_dim)); }
        if self.kv_heads == 0 || self.heads % self.kv_heads != 0 { return e(format!("heads {} / kv_heads {}", self.heads, self.kv_heads)); }
        if self.n_layers == 0 || self.layer_ids.is_empty() { return e("no layers / no taps".into()); }
        if let Some(&i) = self.layer_ids.iter().find(|&&i| i == 0 || i > t.layers) {
            return e(format!("layer_id {} not in 1..={} (hidden_states index; 0 = the embedding is not tapped)", i, t.layers));
        }
        for (n, x) in [("hidden", self.hidden), ("intermediate", self.inter), ("q_dim", self.q_dim())] {
            if x % 64 != 0 { return e(format!("{} {} not a multiple of 64 (4-bit g64)", n, x)); }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DfLayer {
    pub in_norm: Region,
    pub post_norm: Region,
    /// q_proj (q | gate per head) | k_proj | v_proj
    pub in_proj: Q4b,
    /// rows k | v of `in_proj` (a view, no copy): the context K / V projection
    pub kv: Q4b,
    pub q_norm: Region,
    pub k_norm: Region,
    pub o_proj: Q4b,
    pub gate: Q4b,
    pub up: Q4b,
    pub down: Q4b,
    /// context KV of this layer: bf16 [kv_heads][max_ctx][head_dim]
    pub kc: Region,
    pub vc: Region,
}

/// Header-only plan of a drafter directory (budget rows before anything is allocated).
pub struct DfPlan {
    pub dir: PathBuf,
    pub cfg: DfConfig,
    idx: SafeIndex,
    weight_bytes: usize,
    n_tensors: usize,
}

fn tensor<'a>(idx: &'a SafeIndex, name: &str, dt: DType, shape: &[usize]) -> Result<&'a TensorInfo, String> {
    let t = idx.get(name).ok_or_else(|| format!("drafter: no tensor {}", name))?;
    if t.dtype != dt || t.shape != shape { return Err(format!("drafter: {} is {:?} {:?}, want {:?} {:?}", name, t.dtype, t.shape, dt, shape)); }
    Ok(t)
}

impl DfPlan {
    pub fn read(dir: &Path, t: &QwenConfig) -> Result<DfPlan, String> {
        let cf = dir.join(CONFIG_FILE);
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cf).map_err(|e| format!("{}: {}", cf.display(), e))?)
            .map_err(|e| format!("{}: {}", cf.display(), e))?;
        let cfg = DfConfig::from_json(&v)?;
        cfg.check(t)?;
        let wf = dir.join(WEIGHTS_FILE);
        let idx = SafeIndex::open_file(&wf.display().to_string())?;
        let mut p = DfPlan { dir: dir.to_path_buf(), cfg, idx, weight_bytes: 0, n_tensors: 0 };
        for (name, dt, shape) in p.tensors() {
            let ti = tensor(&p.idx, &name, dt, &shape)?;
            p.weight_bytes += ti.nbytes;
            p.n_tensors += 1;
        }
        Ok(p)
    }

    /// Every tensor the engine reads: (name, dtype, shape).
    fn tensors(&self) -> Vec<(String, DType, Vec<usize>)> {
        let c = &self.cfg;
        let h = c.hidden;
        let mut v: Vec<(String, DType, Vec<usize>)> = Vec::new();
        let q4 = |v: &mut Vec<(String, DType, Vec<usize>)>, n: &str, rows: usize, cols: usize| {
            v.push((format!("{}.weight", n), DType::U32, vec![rows, cols / 8]));
            v.push((format!("{}.scales", n), DType::BF16, vec![rows, cols / 64]));
            v.push((format!("{}.biases", n), DType::BF16, vec![rows, cols / 64]));
        };
        let vecn = |v: &mut Vec<(String, DType, Vec<usize>)>, n: String, len: usize| v.push((n, DType::BF16, vec![len]));
        q4(&mut v, "fc", h, c.fw());
        vecn(&mut v, "ctx_norm.weight".into(), h);
        vecn(&mut v, "norm.weight".into(), h);
        vecn(&mut v, "mask_emb".into(), h);
        if c.tap_norm { for i in 0..c.n_taps() { vecn(&mut v, format!("tap_norms.{}.weight", i), h); } }
        for l in 0..c.n_layers {
            let p = format!("layers.{}", l);
            vecn(&mut v, format!("{}.input_layernorm.weight", p), h);
            vecn(&mut v, format!("{}.post_attention_layernorm.weight", p), h);
            q4(&mut v, &format!("{}.self_attn.in_proj", p), c.prow(), h);
            vecn(&mut v, format!("{}.self_attn.q_norm.weight", p), c.head_dim);
            vecn(&mut v, format!("{}.self_attn.k_norm.weight", p), c.head_dim);
            q4(&mut v, &format!("{}.self_attn.o_proj", p), h, c.q_dim());
            q4(&mut v, &format!("{}.mlp.gate_proj", p), c.inter, h);
            q4(&mut v, &format!("{}.mlp.up_proj", p), c.inter, h);
            q4(&mut v, &format!("{}.mlp.down_proj", p), h, c.inter);
        }
        v
    }

    fn kv_bytes(&self, max_ctx: usize) -> usize { self.cfg.n_layers * 2 * self.cfg.kv_heads * max_ctx * self.cfg.head_dim * BF }

    /// Scratch regions (name, bytes) for passes of up to `tmax` rows.
    fn scratch(&self, tmax: usize, max_ctx: usize, log: bool) -> Vec<(&'static str, usize)> {
        let c = &self.cfg;
        let (b, h) = (c.block, c.hidden);
        let mut v = vec![
            ("taps", tmax * c.fw() * BF), ("feat", ABSORB_ROWS * h * BF), ("ctx", ABSORB_ROWS * h * BF),
            ("kvp", ABSORB_ROWS * 2 * c.kv_dim() * BF),
            ("h", b * h * BF), ("x", b * h * BF), ("hf", b * h * BF), ("p", b * c.prow() * BF), ("qr", b * c.q_dim() * BF),
            ("ao", b * c.q_dim() * BF), ("f", b * c.inter * BF), ("kb", b * c.kv_dim() * BF), ("vb", b * c.kv_dim() * BF),
            ("logits", (b - 1) * c.vocab * 4), ("out", b * 8),
        ];
        if log { v.push(("taplog", max_ctx * c.fw() * BF)); }
        v
    }

    /// Budget rows of a drafter for passes of up to `tmax` rows.
    pub fn budget_rows(&self, tmax: usize, max_ctx: usize, log: bool) -> Vec<(String, u64)> {
        let sc: usize = self.scratch(tmax, max_ctx, log).iter().map(|x| x.1).sum();
        vec![
            (format!("DFlash drafter weights ({} layers, 4-bit, {})", self.cfg.n_layers, self.dir.display()), self.weight_bytes as u64),
            (format!("DFlash drafter context KV bf16 @ {} ctx", max_ctx), self.kv_bytes(max_ctx) as u64),
            ("DFlash drafter scratch (tap rows, block rows, logits)".to_string(), sc as u64),
        ]
    }
    /// Arena regions it allocates (for the alignment slack row).
    pub fn regions(&self, log: bool) -> usize { self.n_tensors + self.scratch(1, 1, log).len() + 2 * self.cfg.n_layers }
}

/// Statistics of one `feed_stream_dflash` (or summed: `add`).
#[derive(Clone, Debug, Default)]
pub struct DfStats {
    /// verify passes (rounds), drafts verified / accepted, passes with a rejected draft
    pub passes: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub rejects: usize,
    /// tokens emitted by the rounds (not the prefill's first token)
    pub tokens: usize,
    /// accept_hist[j] = passes with j drafts accepted
    pub accept_hist: [usize; BLOCK_MAX],
    /// GPU ms: drafter forwards, verify passes, replays
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub replay_ms: f64,
    /// rounds whose drafter context was shorter than the anchor (positions fed by passes without capture)
    pub short_ctx: usize,
}
impl DfStats {
    pub fn add(&mut self, o: &DfStats) {
        self.passes += o.passes; self.drafted += o.drafted; self.accepted += o.accepted; self.rejects += o.rejects; self.tokens += o.tokens;
        for i in 0..BLOCK_MAX { self.accept_hist[i] += o.accept_hist[i]; }
        self.draft_ms += o.draft_ms; self.verify_ms += o.verify_ms; self.replay_ms += o.replay_ms; self.short_ctx += o.short_ctx;
    }
    pub fn acceptance(&self) -> f64 { if self.drafted > 0 { self.accepted as f64 / self.drafted as f64 } else { 0.0 } }
    pub fn tokens_per_pass(&self) -> f64 { if self.passes > 0 { self.tokens as f64 / self.passes as f64 } else { 0.0 } }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"passes": self.passes, "drafted": self.drafted, "accepted": self.accepted, "rejects": self.rejects,
            "tokens": self.tokens, "acceptance": self.acceptance(), "tokens_per_pass": self.tokens_per_pass(), "accept_hist": self.accept_hist,
            "draft_ms": self.draft_ms, "verify_ms": self.verify_ms, "replay_ms": self.replay_ms, "short_ctx": self.short_ctx})
    }
}

/// A loaded drafter: weights, context KV and scratch in the engine arena.
pub struct DFlash {
    pub cfg: DfConfig,
    fc: Q4b,
    ctx_norm: Region,
    norm: Region,
    mask_emb: Region,
    tap_norms: Vec<Region>,
    pub layers: Vec<DfLayer>,
    taps: Region,
    feat: Region,
    ctx: Region,
    kvp: Region,
    h: Region,
    x: Region,
    hf: Region,
    p: Region,
    qr: Region,
    ao: Region,
    f: Region,
    kb: Region,
    vb: Region,
    logits: Region,
    out: Region,
    taplog: Option<Region>,
    max_ctx: usize,
    tmax: usize,
    /// contiguous prefix of positions with context rows (host copy, updated at encode time = execution order)
    ctx_len: Cell<usize>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DfPrep { p: u32, k_off: u32, v_off: u32, nq: u32, nkv: u32, hd: u32, rdims: u32, pos0: u32, kv_rows: u32, row0: u32, eps: f32 }
#[repr(C)]
#[derive(Clone, Copy)]
struct DfAttn { nh: u32, nkv: u32, hd: u32, n_ctx: u32, nb: u32, max_ctx: u32, p: u32, scale: f32 }

pub const PIPELINES: [&str; 4] = ["df_copy_rows", "df_prep", "df_attn", "df_argmax"];
pub const SHADER: &str = "dflash";

impl DFlash {
    /// Allocate + read the drafter (weights, context KV, scratch) into the engine arena; build its pipelines.
    pub fn load(plan: &DfPlan, arena: &mut Arena, gpu: &mut Gpu, tmax: usize, max_ctx: usize, log: bool) -> Result<DFlash, String> {
        let c = plan.cfg.clone();
        let files: Vec<std::fs::File> = plan.idx.files().iter().map(|f| std::fs::File::open(f).map_err(|e| format!("{}: {}", f, e)))
            .collect::<Result<_, _>>()?;
        let rd = |arena: &mut Arena, name: &str| -> Result<Region, String> {
            let t = plan.idx.get(name).ok_or_else(|| format!("drafter: no tensor {}", name))?;
            let r = arena.alloc(&format!("w:dflash.{}", name), t.nbytes)?;
            let dst = arena.slice_mut::<u8>(r);
            files[t.file].read_exact_at(&mut dst[..t.nbytes], t.off).map_err(|e| format!("drafter: read {}: {}", name, e))?;
            Ok(r)
        };
        let q4 = |arena: &mut Arena, n: &str, rows: usize, cols: usize| -> Result<Q4b, String> {
            Ok(Q4b { rows, cols, w: rd(arena, &format!("{}.weight", n))?, s: rd(arena, &format!("{}.scales", n))?, b: rd(arena, &format!("{}.biases", n))? })
        };
        let h = c.hidden;
        let fc = q4(arena, "fc", h, c.fw())?;
        let ctx_norm = rd(arena, "ctx_norm.weight")?;
        let norm = rd(arena, "norm.weight")?;
        let mask_emb = rd(arena, "mask_emb")?;
        let mut tap_norms = vec![];
        if c.tap_norm { for i in 0..c.n_taps() { tap_norms.push(rd(arena, &format!("tap_norms.{}.weight", i))?); } }
        let kvb = c.kv_heads * max_ctx * c.head_dim * BF;
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            let p = format!("layers.{}", l);
            let in_norm = rd(arena, &format!("{}.input_layernorm.weight", p))?;
            let post_norm = rd(arena, &format!("{}.post_attention_layernorm.weight", p))?;
            let in_proj = q4(arena, &format!("{}.self_attn.in_proj", p), c.prow(), h)?;
            let q_norm = rd(arena, &format!("{}.self_attn.q_norm.weight", p))?;
            let k_norm = rd(arena, &format!("{}.self_attn.k_norm.weight", p))?;
            let o_proj = q4(arena, &format!("{}.self_attn.o_proj", p), h, c.q_dim())?;
            let gate = q4(arena, &format!("{}.mlp.gate_proj", p), c.inter, h)?;
            let up = q4(arena, &format!("{}.mlp.up_proj", p), c.inter, h)?;
            let down = q4(arena, &format!("{}.mlp.down_proj", p), h, c.inter)?;
            // rows k | v of in_proj: W u32 [rows][cols / 8], S / B bf16 [rows][cols / 64]
            let r0 = 2 * c.q_dim();
            let n = 2 * c.kv_dim();
            let sub = |r: Region, row_bytes: usize| Region { off: r.off + r0 * row_bytes, len: n * row_bytes };
            let kv = Q4b { rows: n, cols: h, w: sub(in_proj.w, h / 8 * 4), s: sub(in_proj.s, h / 64 * BF), b: sub(in_proj.b, h / 64 * BF) };
            let kc = arena.alloc(&format!("kv:dflash.{}.k", l), kvb)?;
            let vc = arena.alloc(&format!("kv:dflash.{}.v", l), kvb)?;
            layers.push(DfLayer { in_norm, post_norm, in_proj, kv, q_norm, k_norm, o_proj, gate, up, down, kc, vc });
        }
        let mut s = std::collections::HashMap::new();
        for (n, b) in plan.scratch(tmax, max_ctx, log) {
            let r = arena.alloc(&format!("s:dflash.{}", n), b)?;
            arena.slice_mut::<u8>(r).fill(0);
            s.insert(n, r);
        }
        for n in PIPELINES { gpu.pipeline(n)?; }
        Ok(DFlash { cfg: c, fc, ctx_norm, norm, mask_emb, tap_norms, layers, taps: s["taps"], feat: s["feat"], ctx: s["ctx"], kvp: s["kvp"],
                    h: s["h"], x: s["x"], hf: s["hf"], p: s["p"], qr: s["qr"], ao: s["ao"], f: s["f"], kb: s["kb"], vb: s["vb"],
                    logits: s["logits"], out: s["out"], taplog: s.get("taplog").copied(), max_ctx, tmax, ctx_len: Cell::new(0) })
    }

    /// Positions 0 .. ctx_len() have context rows (a contiguous prefix).
    pub fn ctx_len(&self) -> usize { self.ctx_len.get() }

    fn rmsnorm(&self, e: &Enc, rows: usize, x: usize, w: usize, out: usize, out_stride: usize) {
        e.pso("rmsnorm_b").b(0, x).b(1, w).b(2, out).u(3, self.cfg.hidden as u32).f(4, self.cfg.eps).u(5, out_stride as u32)
            .groups((rows, 1), 1024);
    }
    fn copy_rows(e: &Enc, src: usize, dst: usize, n: usize, rows: usize, ss: usize, ds: usize) {
        e.pso("df_copy_rows").b(0, src).b(1, dst).u(2, n as u32).u(3, ss as u32).u(4, ds as u32).threads((n, rows), 256);
    }

    /// After target layer `l` of a pass (`io.k` rows at host position `io.pos`): copy the rows of every tap taken
    /// after `l` into the tap buffer [row][tap][hidden] (RMS-normed with the tap's weight when tap_norm).
    pub(crate) fn tap(&self, e: &Enc, io: &PassIo, l: usize) {
        let c = &self.cfg;
        let (h, fw, k) = (c.hidden, c.fw(), io.k);
        for i in 0..c.n_taps() {
            if c.tap_layer(i) != l { continue; }
            if let Some(log) = self.taplog {
                if io.pos + k <= self.max_ctx { Self::copy_rows(e, io.h, log.off + (io.pos * fw + i * h) * BF, h, k, h, fw); }
            }
            let dst = self.taps.off + i * h * BF;
            if c.tap_norm { self.rmsnorm(e, k, io.h, self.tap_norms[i].off, dst, fw); } else { Self::copy_rows(e, io.h, dst, h, k, h, fw); }
        }
    }

    /// End of a pass of `k` rows at position `pos`: context rows for positions pos .. pos + k (fc + ctx_norm, then per
    /// draft layer k / v projection, k_norm + RoPE, into its context KV), in blocks of <= 8 rows.
    pub(crate) fn absorb(&self, e: &Enc, io: &PassIo, pos: usize, k: usize) -> Result<(), String> {
        let c = &self.cfg;
        if k > self.tmax { return Err(format!("dflash: pass of {} rows > tap rows {}", k, self.tmax)); }
        if pos + k > self.max_ctx { return Err(format!("dflash: positions {}..{} > max_ctx {}", pos, pos + k, self.max_ctx)); }
        let (h, fw, kvd) = (c.hidden, c.fw(), c.kv_dim());
        let mut r0 = 0;
        while r0 < k {
            let kb = (k - r0).min(ABSORB_ROWS);
            gemv_m(e, io.gemv, &self.fc, kb, 0, self.taps.off + r0 * fw * BF, self.feat.off, None);
            self.rmsnorm(e, kb, self.feat.off, self.ctx_norm.off, self.ctx.off, h);
            for lw in &self.layers {
                gemv_m(e, io.gemv, &lw.kv, kb, 0, self.ctx.off, self.kvp.off, None);
                let a = DfPrep { p: (2 * kvd) as u32, k_off: 0, v_off: kvd as u32, nq: 0, nkv: c.kv_heads as u32, hd: c.head_dim as u32,
                                 rdims: c.rot_dims as u32, pos0: (pos + r0) as u32, kv_rows: self.max_ctx as u32, row0: (pos + r0) as u32, eps: c.eps };
                e.pso("df_prep").b(0, self.kvp.off).b(1, lw.k_norm.off).b(2, lw.k_norm.off).b(3, self.kvp.off).b(4, lw.kc.off).b(5, lw.vc.off)
                    .b(6, io.invf).bytes(7, &a).groups((c.kv_heads, kb), c.head_dim);
            }
            r0 += kb;
        }
        // contiguous prefix: a pass at or before its end extends / truncates it; a pass past it leaves a gap
        if pos <= self.ctx_len.get() { self.ctx_len.set(pos + k); }
        Ok(())
    }

    /// Block forward at anchor position `a` (pending token's embedding row staged in `stage` row 0): drafts D1 ..
    /// D(B-1) into `out` (u32 tokens, then f32 max logits), logits f32 [B - 1][vocab].
    fn encode_draft(&self, e: &Enc, io: &PassIo, lm_head: &Q4b, stage: usize, a: usize) -> Result<(), String> {
        let c = &self.cfg;
        let (b, h) = (c.block, c.hidden);
        let n_ctx = a.min(self.ctx_len.get());
        e.pso("embed_q4").b(0, stage).b(1, self.h.off).u(2, 1);
        e.threads((h / 8, 1), 64);
        Self::copy_rows(e, self.mask_emb.off, self.h.off + h * BF, h, b - 1, 0, h);
        let g = io.gemv;
        for lw in &self.layers {
            self.rmsnorm(e, b, self.h.off, lw.in_norm.off, self.x.off, h);
            gemv_m(e, g, &lw.in_proj, b, 0, self.x.off, self.p.off, None);
            let pp = DfPrep { p: c.prow() as u32, k_off: (2 * c.q_dim()) as u32, v_off: (2 * c.q_dim() + c.kv_dim()) as u32, nq: c.heads as u32,
                              nkv: c.kv_heads as u32, hd: c.head_dim as u32, rdims: c.rot_dims as u32, pos0: a as u32, kv_rows: b as u32, row0: 0, eps: c.eps };
            e.pso("df_prep").b(0, self.p.off).b(1, lw.q_norm.off).b(2, lw.k_norm.off).b(3, self.qr.off).b(4, self.kb.off).b(5, self.vb.off)
                .b(6, io.invf).bytes(7, &pp).groups((c.heads + c.kv_heads, b), c.head_dim);
            let at = DfAttn { nh: c.heads as u32, nkv: c.kv_heads as u32, hd: c.head_dim as u32, n_ctx: n_ctx as u32, nb: b as u32,
                              max_ctx: self.max_ctx as u32, p: c.prow() as u32, scale: (c.head_dim as f32).powf(-0.5) };
            e.pso("df_attn").b(0, self.qr.off).b(1, lw.kc.off).b(2, lw.vc.off).b(3, self.kb.off).b(4, self.vb.off).b(5, self.p.off)
                .b(6, self.ao.off).bytes(7, &at).groups((c.heads, b), 256);
            gemv_m(e, g, &lw.o_proj, b, 2, self.ao.off, self.h.off, None);
            self.rmsnorm(e, b, self.h.off, lw.post_norm.off, self.x.off, h);
            swiglu_m(e, g, &lw.gate, &lw.up, b, self.x.off, self.f.off, None);
            gemv_m(e, g, &lw.down, b, 2, self.f.off, self.h.off, None);
        }
        self.rmsnorm(e, b, self.h.off, self.norm.off, self.hf.off, h);
        gemv_m(e, g, lm_head, b - 1, 3, self.hf.off + h * BF, self.logits.off, None);
        e.pso("df_argmax").b(0, self.logits.off).b(1, self.out.off).b(2, self.out.off + b * 4).u(3, c.vocab as u32).groups((b - 1, 1), 1024);
        Ok(())
    }
}

/// Write a random drafter in the engine format for the target `t` (tests; deterministic in `seed`). Weights are random
/// 4-bit codes with per-group scale / bias of a ~1/sqrt(cols) spread; norms ~1.
pub fn write_random(dir: &Path, t: &QwenConfig, n_layers: usize, block: usize, tap_norm: bool, seed: u64) -> Result<DfConfig, String> {
    let n = 5.min(t.layers.max(1));
    let (lo, hi) = (2usize.min(t.layers), t.layers.saturating_sub(2).max(1));
    let layer_ids: Vec<usize> = (0..n).map(|i| (lo as f64 + i as f64 * (hi as f64 - lo as f64) / (n.max(2) - 1) as f64).round() as usize)
        .map(|x| x.clamp(1, t.layers)).collect();
    let c = DfConfig { hidden: t.hidden, heads: t.heads, kv_heads: t.kv_heads, head_dim: t.head_dim, rot_dims: t.rope_dims, inter: t.inter,
                       eps: t.eps, rope_theta: t.rope_theta, vocab: t.vocab, n_layers, block, layer_ids, tap_norm };
    c.check(t)?;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let bf = streamdense_core::f32_to_bf16;
    let mut owned: Vec<(String, DType, Vec<usize>, Vec<u8>)> = vec![];
    let q4 = |o: &mut Vec<(String, DType, Vec<usize>, Vec<u8>)>, rnd: &mut dyn FnMut() -> u64, n: &str, rows: usize, cols: usize| {
        let w: Vec<u8> = (0..rows * cols / 2).map(|_| rnd() as u8).collect();
        let a = 1.7 / (cols as f32).sqrt();
        let g = rows * cols / 64;
        let sc: Vec<u8> = (0..g).flat_map(|_| bf(a * (0.8 + 0.4 * (rnd() % 1000) as f32 / 1000.0) * 2.0 / 15.0).to_le_bytes()).collect();
        let bi: Vec<u8> = (0..g).flat_map(|_| bf(-a * (0.8 + 0.4 * (rnd() % 1000) as f32 / 1000.0)).to_le_bytes()).collect();
        o.push((format!("{}.weight", n), DType::U32, vec![rows, cols / 8], w));
        o.push((format!("{}.scales", n), DType::BF16, vec![rows, cols / 64], sc));
        o.push((format!("{}.biases", n), DType::BF16, vec![rows, cols / 64], bi));
    };
    let vecn = |o: &mut Vec<(String, DType, Vec<usize>, Vec<u8>)>, rnd: &mut dyn FnMut() -> u64, n: String, len: usize, base: f32, spread: f32| {
        let d: Vec<u8> = (0..len).flat_map(|_| bf(base + spread * ((rnd() % 2001) as f32 / 1000.0 - 1.0)).to_le_bytes()).collect();
        o.push((n, DType::BF16, vec![len], d));
    };
    let h = c.hidden;
    q4(&mut owned, &mut rnd, "fc", h, c.fw());
    vecn(&mut owned, &mut rnd, "ctx_norm.weight".into(), h, 1.0, 0.2);
    vecn(&mut owned, &mut rnd, "norm.weight".into(), h, 1.0, 0.2);
    vecn(&mut owned, &mut rnd, "mask_emb".into(), h, 0.0, 0.05);
    if tap_norm { for i in 0..c.n_taps() { vecn(&mut owned, &mut rnd, format!("tap_norms.{}.weight", i), h, 1.0, 0.3); } }
    for l in 0..n_layers {
        let p = format!("layers.{}", l);
        vecn(&mut owned, &mut rnd, format!("{}.input_layernorm.weight", p), h, 1.0, 0.2);
        vecn(&mut owned, &mut rnd, format!("{}.post_attention_layernorm.weight", p), h, 1.0, 0.2);
        q4(&mut owned, &mut rnd, &format!("{}.self_attn.in_proj", p), c.prow(), h);
        vecn(&mut owned, &mut rnd, format!("{}.self_attn.q_norm.weight", p), c.head_dim, 1.0, 0.2);
        vecn(&mut owned, &mut rnd, format!("{}.self_attn.k_norm.weight", p), c.head_dim, 1.0, 0.2);
        q4(&mut owned, &mut rnd, &format!("{}.self_attn.o_proj", p), h, c.q_dim());
        q4(&mut owned, &mut rnd, &format!("{}.mlp.gate_proj", p), c.inter, h);
        q4(&mut owned, &mut rnd, &format!("{}.mlp.up_proj", p), c.inter, h);
        q4(&mut owned, &mut rnd, &format!("{}.mlp.down_proj", p), h, c.inter);
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let views: Vec<(&str, DType, &[usize], &[u8])> = owned.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice())).collect();
    streamdense_core::write_safetensors(&dir.join(WEIGHTS_FILE).display().to_string(), &views)?;
    std::fs::write(dir.join(CONFIG_FILE), serde_json::to_string_pretty(&c.to_json()).unwrap()).map_err(|e| format!("{}: {}", dir.display(), e))?;
    Ok(c)
}

/// Test hook of the spec loop: (anchor position, the drafter's drafts) -> the drafts to verify instead.
pub type DraftHook<'a> = &'a dyn Fn(usize, &[u32]) -> Vec<u32>;

impl QwenEngine {
    /// A DFlash drafter is loaded (`QwenOptions.dflash`).
    pub fn dflash_on(&self) -> bool { self.dflash.is_some() }
    pub fn dflash_config(&self) -> Option<&DfConfig> { self.dflash.as_ref().map(|d| &d.cfg) }
    /// Positions with drafter context rows (0 without a drafter).
    pub fn dflash_ctx_len(&self) -> usize { self.dflash.as_ref().map_or(0, |d| d.ctx_len()) }

    fn df(&self) -> Result<&DFlash, String> { self.dflash.as_ref().ok_or_else(|| "dflash: no drafter loaded (QwenOptions.dflash)".to_string()) }

    /// One drafter forward at anchor position `a` with pending token `t` (synchronous; staging slot 1): the B - 1
    /// drafts. Reads only the drafter's own state (the context rows < a); the target's KV / state are untouched.
    pub fn dflash_draft(&self, a: usize, t: u32) -> Result<(Vec<u32>, f64), String> {
        let d = self.df()?;
        self.reader.read_rows(&[t], self.stage_ptr(1))?;
        let cb = self.enc_cb(&|e| d.encode_draft(e, &self.io(), &self.w.lm_head, self.sl().stage.at(self.slot_bytes), a))?;
        commit(&cb);
        wait(&cb)?;
        let b = d.cfg.block;
        let out = self.arena.slice_mut::<u32>(d.out);
        Ok((out[..b - 1].to_vec(), gpu_ms(&cb)))
    }
    /// Logits f32 [B - 1][vocab] of the last `dflash_draft` (tests).
    pub fn dflash_logits(&self) -> Vec<f32> {
        match &self.dflash { Some(d) => self.arena.slice_mut::<f32>(d.logits)[..(d.cfg.block - 1) * d.cfg.vocab].to_vec(), None => vec![] }
    }
    /// Raw tap rows (bf16 bits [n][n_taps * hidden]) of positions 0 .. n, logged with `QwenOptions.dflash_log_taps` (tests).
    pub fn dflash_tap_log(&self, n: usize) -> Option<Vec<u16>> {
        let d = self.dflash.as_ref()?;
        let r = d.taplog?;
        Some(self.arena.slice_mut::<u16>(r)[..n.min(d.max_ctx) * d.cfg.fw()].to_vec())
    }

    /// Verify pass of k rows staged in slot 0 at position p (ctx_draft.rs `encode_ctx_verify` + the drafter capture):
    /// k = 1 the plain decode pass, k >= 2 a HOLD pass. Head + argmax on all rows: rec[p + i]. ctrl.pos not advanced.
    fn encode_df_verify(&self, e: &Enc, d: &DFlash, k: usize, p: usize) -> Result<(), String> {
        let c = &self.cfg;
        let hold = if k > 1 { Some(self.ctx_hold.ok_or("dflash: no hold save")?.off) } else { None };
        let io = PassIo { k, pos: p, nch_max: self.nch_for(p, k), dn_save: None, dn_hold: hold, rm: self.rm_io(true), ..self.io() };
        e.pso("embed_q4").b(0, self.sl().stage.at(0)).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        for l in 0..c.layers {
            encode_layer(e, c, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, false, None, l)?;
            d.tap(e, &io, l);
        }
        encode_head(e, c, &self.w, &io, k, None)?;
        d.absorb(e, &io, p, k)
    }

    /// Streaming greedy decode with DFlash block drafts (`feed_stream` contract: feed `ids` from the current position,
    /// then decode, `on_tok` per token, same stop rules; the last emitted token is not fed). Output == `feed_stream`
    /// token for token. `ids.len() == 1`: that token is row 0 of the first verify pass (its context rows must exist:
    /// the earlier passes ran on this engine with the drafter loaded).
    pub fn feed_stream_dflash(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool)
                              -> Result<(StreamStats, DfStats), String> {
        self.feed_stream_dflash_ov(ids, n_new, stop, on_tok, None)
    }

    /// `feed_stream_dflash` with a test hook that replaces the drafts of each round (the drafter still runs).
    pub fn feed_stream_dflash_ov(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                                 hook: Option<DraftHook>) -> Result<(StreamStats, DfStats), String> {
        let d = self.df()?;
        let hold = self.ctx_hold.ok_or("feed_stream_dflash: no hold save")?.off;
        let mut st = StreamStats { stop_reason: "", ..Default::default() };
        let mut ds = DfStats::default();
        if ids.is_empty() { return Err("feed_stream_dflash: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok((st, ds)); }
        let tp = Instant::now();
        let mut pending = ids[0];
        if ids.len() > 1 {
            let t = self.prefill(ids)?;
            st.gen = 1;
            st.last = Some(t);
            pending = t;
            if !on_tok(t) { st.stop_reason = "cancelled"; }
            else if self.is_eos(t) || stop.contains(&t) { st.stop_reason = "eos"; }
            else if st.gen >= n_new { st.stop_reason = "length"; }
        }
        st.prefill_s = tp.elapsed().as_secs_f64();
        if !st.stop_reason.is_empty() { return Ok((st, ds)); }
        let td = Instant::now();
        let r = self.df_loop(d, pending, hold, n_new, stop, on_tok, hook, &mut st, &mut ds);
        st.decode_s = td.elapsed().as_secs_f64();
        st.gpu_ms = ds.draft_ms + ds.verify_ms + ds.replay_ms;
        r?;
        if st.stop_reason.is_empty() { st.stop_reason = "length"; }
        Ok((st, ds))
    }

    #[allow(clippy::too_many_arguments)]
    fn df_loop(&self, d: &DFlash, mut pending: u32, hold: usize, n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
               hook: Option<DraftHook>, st: &mut StreamStats, ds: &mut DfStats) -> Result<(), String> {
        let max_ctx = self.opt.max_ctx;
        let b = d.cfg.block;
        let ev = self.gpu.shared_event();
        let mut nsig = 0u64;
        let mut wait_v: Option<u64> = None;
        let mut last_replay: Option<CmdBuf> = None;
        let fin = |lr: &mut Option<CmdBuf>, ds: &mut DfStats| -> Result<(), String> {
            if let Some(cb) = lr.take() { wait(&cb)?; ds.replay_ms += gpu_ms(&cb); }
            Ok(())
        };
        let r = (|| -> Result<(), String> {
            loop {
                let p = self.pos();
                if p + 1 > max_ctx { st.stop_reason = "ctx_full"; break; }
                // ---- drafts: one drafter forward (it reads only its own context rows, so it may overlap the replay)
                let left = n_new.saturating_sub(st.gen).max(1);
                let cap = (b - 1).min(max_ctx - p - 1).min(left - 1);
                let mut drafts: Vec<u32> = vec![];
                if cap > 0 {
                    if d.ctx_len() < p { ds.short_ctx += 1; }
                    let (dr, ms) = self.dflash_draft(p, pending)?;
                    ds.draft_ms += ms;
                    drafts = match hook { Some(f) => f(p, &dr), None => dr };
                    drafts.truncate(cap);
                }
                let nd = drafts.len();
                let k = nd + 1;
                let mut rows = Vec::with_capacity(k);
                rows.push(pending);
                rows.extend_from_slice(&drafts);
                self.reader.read_rows(&rows, self.stage_ptr(0))?;
                // ---- verify (waits on the GPU for the previous replay)
                let vcb = objc2::rc::autoreleasepool(|_| -> Result<CmdBuf, String> {
                    let cb = self.gpu.command_buffer();
                    if let Some(v) = wait_v { cb_wait(&cb, &ev, v); }
                    let e = Enc::new(&self.gpu, &cb, &self.buf);
                    let r = self.encode_df_verify(&e, d, k, p);
                    e.end();
                    r.map(|_| cb)
                })?;
                self.rm_before_pass();
                commit(&vcb);
                wait(&vcb)?;
                ds.verify_ms += gpu_ms(&vcb);
                fin(&mut last_replay, ds)?;
                // ---- decide
                let a: Vec<u32> = (0..k).map(|i| self.rec_token(p + i).0).collect();
                let mut j = 0;
                while j < nd && a[j] == drafts[j] && !self.is_eos(a[j]) { j += 1; }
                ds.passes += 1;
                ds.accept_hist[j] += 1;
                ds.drafted += nd;
                ds.accepted += j;
                if j < nd { ds.rejects += 1; }
                let mut why: Option<&'static str> = None;
                let mut used = 0;
                for &t in &a[..=j] {
                    used += 1;
                    st.gen += 1;
                    ds.tokens += 1;
                    st.last = Some(t);
                    if !on_tok(t) { why = Some("cancelled"); break; }
                    if self.is_eos(t) || stop.contains(&t) { why = Some("eos"); break; }
                    if st.gen >= n_new { why = Some("length"); break; }
                }
                let je = used - 1;
                let gaps: Vec<f32> = (0..=je).map(|i| self.rec_gap(p + i)).collect();
                self.rm_after_pass(&gaps);
                self.ctrl().pos = (p + je + 1) as u32;
                self.arena.slice_mut::<f32>(self.sl().rec)[4 * (p + je + 1)..4 * (p + k)].fill(0.0);
                self.set_pos(p + je + 1);
                // ---- replay of the fed rows (HOLD pass); the next verify waits for its signal on the GPU
                wait_v = None;
                if k > 1 {
                    let cb = self.enc_cb(&|e| self.encode_replay_from(e, hold, je + 1))?;
                    nsig += 1;
                    cb_signal(&cb, &ev, nsig);
                    commit(&cb);
                    wait_v = Some(nsig);
                    last_replay = Some(cb);
                }
                if why.is_none() && self.pos() + 1 > max_ctx { why = Some("ctx_full"); }
                if let Some(w) = why { st.stop_reason = w; break; }
                pending = a[je];
            }
            Ok(())
        })();
        let f = fin(&mut last_replay, ds);
        r.and(f)
    }
}
