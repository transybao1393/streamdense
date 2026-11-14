//! READ MASK, stage A: all weights stay in RAM; a
//! mask makes the exact passes skip weight rows the user's traffic never activates, so fewer bytes are read per pass.
//! OFF by default everywhere: `QwenOptions.read_mask` (false = no arena region, no kernels, today's code) and, with
//! it on, `set_read_mask(None)` (the default) = today's code path, bit-identical.
//!
//! Units (`ReadMask`, bit set = READ):
//! - FFN: per layer one bit per 64-neuron block b (inter / 64 = 272 blocks on the 27B). A masked block is treated as
//!   inactive: its gate / up rows are not read (its activations a are written as 0) and its down input columns, one
//!   whole q4 group of 64, are not read either (`qm_gemvx` walks a compact list of the read groups).
//! - Heads (where mathematically clean; `policy.heads`): attention q head h = its q + gate rows of the fused in_proj
//!   and its head_dim o_proj input columns; the head's output ao is zeroed (`rm_zero_heads`), the shared k / v rows
//!   and kv heads are always read. DeltaNet value head h = its z rows of in_proj (z only gates the head's output) and
//!   its dv out_proj input columns. Its v / b / a rows, the conv and the recurrent state are ALWAYS computed, so the
//!   state stays exact for later tokens (skipping them would corrupt every later token's state: not clean, not
//!   done). Head masking needs head_dim / dv % 64 == 0 (whole q4 column groups), else that kind's heads stay read.
//! - lm_head, embeddings, norms, the MTP draft layer: never masked.
//! Kernels (gemv_mma_q4.metal `qm_*x`, the Mma family only; `--gemv old` refuses a mask): rows by 8-row tiles (a
//! threadgroup whose tiles are all off reads nothing and writes zeros), columns by a per-layer list of 64-groups
//! (count + ascending indices, the split-K slices over list positions). The "read everything" flag (fl) makes a pass
//! run exactly the unmasked arithmetic. Mask = all-ones, or fl set, is BIT-IDENTICAL to no mask (same groups, same
//! slices, same order); tests/readmask.rs checks plain + deep spec + adaptive.
//!
//! Prompt first: prefill (exact and fast path) ALWAYS runs unmasked; `rm_seen` records which blocks / heads the
//! prompt activated (per-neuron |a|·‖W_down col‖ > `ffn_thr` on any row; heads: Σ_i |o_i|·‖W_out col i‖ > `head_thr`)
//! into `prompt_active` (accumulates over the feeds of one sequence; `reset` clears it). Decode then reads
//! profile ∪ prompt-active ∪ learned (the CLI computes the profile and passes it with `set_read_mask`).
//!
//! Unsure-token full read: after every masked-capable pass the CPU decides the NEXT pass:
//! - gap: an emitted token's bf16 top-1 − top-2 logit < `gap_thr` (spec verify: ANY emitted row; the whole next K-row
//!   pass is one pass), or
//! - out-of-profile: >= `oop_min` blocks + heads active this pass among the rows still read that are NOT in the
//!   profile (read only because the prompt or a full pass needed them);
//! → the next pass runs unmasked (fl = 1) and the blocks / heads it finds active outside the current read set join
//!   `learned` (fetch with `learned` / `take_learned`) and the read set of the rest of this request. Counted in
//!   `MaskStats`. Decisions are made in the sync loops and the pipelined deep / adaptive loop (`feed_stream`,
//!   `generate(.., false)`, `feed_stream_mtp`, deep + adaptive spec); the hand-off-thread loops (`generate(.., true)`,
//!   `generate_mtp(.., pipe = true)`) run masked with the flag as the prefill left it and count nothing.
//!
//! Memory (model memory, arena `s:readmask`, only with `QwenOptions.read_mask`): per layer the mask words (row-tile
//! bits, group lists, head bits, seen bits; ~1.5 KB on the 27B) + bf16 column norms of down and of the output
//! projection (2·(inter + max(q_dim, v_dim)) B per layer = 2.9 MB on the 27B, computed once on the CPU at open).
//! Host: a few `ReadMask` sets of layers·(272 + 48) bits each (< 4 KB on the 27B).
//! Every arena write happens between passes (nothing masked in flight); `set_read_mask` must be called with the
//! engine idle.
use crate::qwen_config::{LayerKind, QwenConfig};
use crate::qwen_engine::QwenEngine;
use crate::weights::{Mixer, Q4b};

/// Neurons per FFN mask block = down q4 group.
pub const BLOCK: usize = 64;
/// Rows per row-mask tile (the twin / MMA threadgroup row granularity).
pub const TILE: usize = 8;

/// A read set: bit set = read. `ffn_blocks[l]` has inter / 64 bits, `heads[l]` the layer's mixer heads (DeltaNet
/// value heads or attention q heads), u64 words, bits past the count always 0.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ReadMask {
    pub ffn_blocks: Vec<Vec<u64>>,
    pub heads: Vec<Vec<u64>>,
}

/// Mixer heads of layer `l` (DeltaNet value heads / attention q heads).
pub fn layer_heads(cfg: &QwenConfig, l: usize) -> usize {
    match cfg.kinds[l] { LayerKind::DeltaNet => cfg.dn_vh, LayerKind::Attention => cfg.heads }
}
/// FFN mask blocks per layer.
pub fn ffn_blocks(cfg: &QwenConfig) -> usize { cfg.inter / BLOCK }

fn words(n: usize) -> usize { n.div_ceil(64) }
fn full_words(n: usize) -> Vec<u64> {
    let mut v = vec![u64::MAX; words(n)];
    if n % 64 != 0 { if let Some(w) = v.last_mut() { *w = (1u64 << (n % 64)) - 1; } }
    v
}

impl ReadMask {
    /// Nothing read.
    pub fn empty(cfg: &QwenConfig) -> ReadMask {
        ReadMask { ffn_blocks: vec![vec![0; words(ffn_blocks(cfg))]; cfg.layers],
                   heads: (0..cfg.layers).map(|l| vec![0; words(layer_heads(cfg, l))]).collect() }
    }
    /// Everything read (bit-identical to no mask).
    pub fn full(cfg: &QwenConfig) -> ReadMask {
        ReadMask { ffn_blocks: vec![full_words(ffn_blocks(cfg)); cfg.layers],
                   heads: (0..cfg.layers).map(|l| full_words(layer_heads(cfg, l))).collect() }
    }
    /// Each FFN block and head read with probability `keep` (tests, benches); deterministic per seed.
    pub fn random(cfg: &QwenConfig, keep: f64, seed: u64) -> ReadMask {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
        let mut m = ReadMask::empty(cfg);
        for l in 0..cfg.layers {
            for b in 0..ffn_blocks(cfg) { if rnd() < keep { m.set_ffn(l, b, true); } }
            for h in 0..layer_heads(cfg, l) { if rnd() < keep { m.set_head(l, h, true); } }
        }
        m
    }
    /// Shape matches `cfg` (layers, words per layer, no bits past the counts).
    pub fn check(&self, cfg: &QwenConfig) -> Result<(), String> {
        if self.ffn_blocks.len() != cfg.layers || self.heads.len() != cfg.layers {
            return Err(format!("read mask: {} / {} layers, model has {}", self.ffn_blocks.len(), self.heads.len(), cfg.layers));
        }
        for l in 0..cfg.layers {
            let (nb, nh) = (ffn_blocks(cfg), layer_heads(cfg, l));
            let ok = |v: &[u64], n: usize| v.len() == words(n) && v.iter().zip(full_words(n)).all(|(a, f)| a & !f == 0);
            if !ok(&self.ffn_blocks[l], nb) { return Err(format!("read mask: layer {} ffn_blocks want {} bits ({} u64)", l, nb, words(nb))); }
            if !ok(&self.heads[l], nh) { return Err(format!("read mask: layer {} heads want {} bits ({} u64)", l, nh, words(nh))); }
        }
        Ok(())
    }
    pub fn ffn(&self, l: usize, b: usize) -> bool { (self.ffn_blocks[l][b / 64] >> (b % 64)) & 1 == 1 }
    pub fn head(&self, l: usize, h: usize) -> bool { (self.heads[l][h / 64] >> (h % 64)) & 1 == 1 }
    pub fn set_ffn(&mut self, l: usize, b: usize, on: bool) {
        let w = &mut self.ffn_blocks[l][b / 64];
        if on { *w |= 1 << (b % 64) } else { *w &= !(1 << (b % 64)) }
    }
    pub fn set_head(&mut self, l: usize, h: usize, on: bool) {
        let w = &mut self.heads[l][h / 64];
        if on { *w |= 1 << (h % 64) } else { *w &= !(1 << (h % 64)) }
    }
    fn zip(&self, o: &ReadMask, f: impl Fn(u64, u64) -> u64) -> ReadMask {
        let z = |a: &Vec<Vec<u64>>, b: &Vec<Vec<u64>>| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).map(|(p, q)| f(*p, *q)).collect()).collect();
        ReadMask { ffn_blocks: z(&self.ffn_blocks, &o.ffn_blocks), heads: z(&self.heads, &o.heads) }
    }
    /// self ∪ o
    pub fn union(&self, o: &ReadMask) -> ReadMask { self.zip(o, |a, b| a | b) }
    /// self ∖ o
    pub fn minus(&self, o: &ReadMask) -> ReadMask { self.zip(o, |a, b| a & !b) }
    /// self ∩ o
    pub fn and(&self, o: &ReadMask) -> ReadMask { self.zip(o, |a, b| a & b) }
    pub fn count_ffn(&self) -> usize { self.ffn_blocks.iter().flatten().map(|w| w.count_ones() as usize).sum() }
    pub fn count_heads(&self) -> usize { self.heads.iter().flatten().map(|w| w.count_ones() as usize).sum() }
    pub fn is_empty(&self) -> bool { self.count_ffn() == 0 && self.count_heads() == 0 }
}

/// Thresholds of the recorder and of the unsure-token rule (no 27B numbers yet: tune on real traffic).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskPolicy {
    /// FFN block active: some neuron's |a|·‖W_down col‖ > this on some row of the pass
    pub ffn_thr: f32,
    /// head active: Σ_i |o_i|·‖W_out col i‖ over the head's columns > this on some row
    pub head_thr: f32,
    /// unsure token: emitted bf16 top-1 − top-2 logit < this → next pass reads everything (NEG_INFINITY = off)
    pub gap_thr: f32,
    /// out-of-profile: >= this many blocks + heads active among the read rows outside the profile → next pass reads
    /// everything (0 = off)
    pub oop_min: usize,
    /// mask heads too (where clean); false = heads always read
    pub heads: bool,
}
impl Default for MaskPolicy {
    fn default() -> Self { MaskPolicy { ffn_thr: 0.01, head_thr: 0.01, gap_thr: 1.0, oop_min: 1, heads: true } }
}

/// Counters since the last `reset_mask_stats` (masked-capable decode / verify passes only; prefill never counts).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaskStats {
    /// passes run with the mask
    pub masked_passes: u64,
    /// passes run unmasked by the unsure rule ("read everything")
    pub full_passes: u64,
    /// reasons (a pass may have both): small top-1 / top-2 gap, out-of-profile activity
    pub full_gap: u64,
    pub full_oop: u64,
    /// blocks / heads added to `learned` by full passes
    pub learned_blocks: u64,
    pub learned_heads: u64,
}

/// Arena offsets of the mask region, carried in `PassIo.rm` (Copy). Per layer (stride bytes): u32 words
/// rbf (gate / up row tiles) | clf (down group list) | rbm (mixer in_proj row tiles) | clm (output-projection group
/// list) | hb (head bits) | sf (seen FFN blocks) | sh (seen heads); norms per layer (nstride bytes): bf16 down column
/// norms [inter] | output-projection column norms [out_max].
#[derive(Clone, Copy, Debug, Default)]
pub struct RmIo {
    /// u32 "read everything" flag of the pass
    pub fl: usize,
    pub base: usize,
    pub stride: usize,
    pub o_rbf: usize,
    pub o_clf: usize,
    pub o_rbm: usize,
    pub o_clm: usize,
    pub o_hb: usize,
    pub o_sf: usize,
    pub o_sh: usize,
    pub norms: usize,
    pub nstride: usize,
    pub o_on: usize,
    /// masked kernels on (decode / verify with a mask); false = only the `rm_seen` recorder
    pub apply: bool,
    pub ffn_thr: f32,
    pub head_thr: f32,
}
impl RmIo {
    pub fn at(&self, l: usize, o: usize) -> usize { self.base + l * self.stride + o }
    pub fn norm(&self, l: usize) -> usize { self.norms + l * self.nstride }
}

/// (in_proj rows, output-projection input columns, heads, columns per head) of a mixer kind.
fn mix_dims(cfg: &QwenConfig, k: LayerKind) -> (usize, usize, usize, usize) {
    match k {
        LayerKind::DeltaNet => (cfg.dn_in_rows(), cfg.dn_v_dim(), cfg.dn_vh, cfg.dn_dv),
        LayerKind::Attention => (cfg.attn_in_rows(), cfg.q_dim(), cfg.heads, cfg.head_dim),
    }
}

/// Layout of the region for `cfg` (offsets relative to its start) and its bytes.
pub fn layout(cfg: &QwenConfig) -> (RmIo, usize) {
    let kinds = [LayerKind::DeltaNet, LayerKind::Attention];
    let present: Vec<LayerKind> = kinds.into_iter().filter(|k| cfg.kinds.contains(k)).collect();
    let (mut in_max, mut out_max, mut h_max) = (0, 0, 0);
    for k in &present { let (i, o, h, _) = mix_dims(cfg, *k); in_max = in_max.max(i); out_max = out_max.max(o); h_max = h_max.max(h); }
    let w32 = |n: usize| n.div_ceil(32);
    let nb = ffn_blocks(cfg);
    let sizes = [w32(cfg.inter.div_ceil(TILE)), 1 + nb, w32(in_max.div_ceil(TILE)), 1 + out_max.div_ceil(64), w32(h_max), w32(nb), w32(h_max)];
    let mut o = [0usize; 7];
    let mut acc = 0;
    for (i, s) in sizes.iter().enumerate() { o[i] = acc * 4; acc += s; }
    let stride = (acc * 4).div_ceil(16) * 16;
    let nstride = (2 * (cfg.inter + out_max)).div_ceil(16) * 16;
    let io = RmIo { fl: 0, base: 16, stride, o_rbf: o[0], o_clf: o[1], o_rbm: o[2], o_clm: o[3], o_hb: o[4], o_sf: o[5], o_sh: o[6],
                    norms: 16 + cfg.layers * stride, nstride, o_on: 2 * cfg.inter, apply: false, ffn_thr: 0.0, head_thr: 0.0 };
    (io, 16 + cfg.layers * (stride + nstride))
}
/// `layout` moved to the region's arena offset.
pub(crate) fn at_offset(cfg: &QwenConfig, off: usize) -> RmIo {
    let (io, _) = layout(cfg);
    RmIo { fl: off, base: off + io.base, norms: off + io.norms, ..io }
}

/// Kernels the read mask needs (built with `QwenOptions.read_mask`).
pub fn pipelines() -> Vec<String> {
    let mut v: Vec<String> = ["rm_zero_heads", "rm_seen"].iter().map(|s| s.to_string()).collect();
    v.extend(crate::qwen_layers::mma_mask_pipelines().iter().map(|s| s.to_string()));
    v
}

/// Head masking is clean for a kind when its heads cover whole q4 column groups of the output projection.
fn heads_clean(cfg: &QwenConfig, k: LayerKind) -> bool { mix_dims(cfg, k).3 % 64 == 0 }

/// Engine-side state (RefCell in `QwenEngine.rm`).
pub(crate) struct RmState {
    pub io: RmIo,
    pub profile: Option<ReadMask>,
    pub prompt: ReadMask,
    pub learned: ReadMask,
    /// the read set the arena holds (profile ∪ prompt ∪ learned while live)
    pub read: ReadMask,
    /// decode phase of a request with a mask: masked kernels on
    pub live: bool,
    pub next_full: bool,
    pub cur_full: bool,
    pub policy: MaskPolicy,
    pub stats: MaskStats,
}
impl RmState {
    pub(crate) fn new(cfg: &QwenConfig, io: RmIo) -> RmState {
        RmState { io, profile: None, prompt: ReadMask::empty(cfg), learned: ReadMask::empty(cfg), read: ReadMask::full(cfg),
                  live: false, next_full: false, cur_full: false, policy: MaskPolicy::default(), stats: MaskStats::default() }
    }
}

/// bf16 (round to nearest even) of a finite f32.
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }

impl QwenEngine {
    fn rm_cell(&self) -> Result<&std::cell::RefCell<RmState>, String> {
        self.rm.as_ref().ok_or_else(|| "read mask: engine opened without QwenOptions.read_mask".to_string())
    }
    #[allow(clippy::mut_from_ref)] // GPU-shared arena words; callers touch disjoint mask regions between passes
    unsafe fn rm_words(&self, off: usize, n: usize) -> &mut [u32] { std::slice::from_raw_parts_mut(self.arena.base().add(off) as *mut u32, n) }

    /// Column norms into the region (once at open): down [inter] and the output projection's input columns.
    pub(crate) fn rm_init_norms(&self) {
        let Some(c) = &self.rm else { return };
        let io = c.borrow().io;
        let cfg = &self.cfg;
        let base = self.arena.base() as usize;
        let nt = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8);
        let per = cfg.layers.div_ceil(nt).max(1);
        let jobs: Vec<(usize, Q4b, Q4b)> = (0..cfg.layers).map(|l| {
            let lw = &self.w.layers[l];
            let out = match &lw.mixer { Mixer::Dn(d) => d.out_proj, Mixer::Attn(a) => a.o_proj };
            (l, lw.ffn.down, out)
        }).collect();
        std::thread::scope(|sc| {
            for ch in jobs.chunks(per) {
                sc.spawn(move || {
                    let mut tmp: Vec<f32> = Vec::new();
                    for &(l, down, out) in ch {
                        let p = (base + io.norm(l)) as *mut u16;
                        for (m, o) in [(down, 0usize), (out, io.o_on / 2)] {
                            tmp.resize(m.cols, 0.0);
                            crate::activity::col_norms(base, &m, &mut tmp);
                            for (j, v) in tmp.iter().enumerate() { unsafe { *p.add(o + j) = to_bf(*v); } }
                        }
                    }
                });
            }
        });
    }

    /// Write `m` as the read set into the arena (row tiles, group lists, head bits). Heads of a kind whose masking is
    /// not clean, or all heads with `policy.heads` off, are written as read.
    fn rm_write(&self, st: &RmState, m: &ReadMask) {
        let cfg = &self.cfg;
        let io = st.io;
        let nb = ffn_blocks(cfg);
        for l in 0..cfg.layers {
            unsafe {
                // FFN: 8 tiles per block; list = read blocks (block b = down group b)
                let rbf = self.rm_words(io.at(l, io.o_rbf), (io.o_clf - io.o_rbf) / 4);
                rbf.fill(0);
                let clf = self.rm_words(io.at(l, io.o_clf), 1 + nb);
                let mut n = 0;
                for b in 0..nb {
                    if !m.ffn(l, b) { continue; }
                    for t in b * BLOCK / TILE..(b + 1) * BLOCK / TILE { rbf[t / 32] |= 1 << (t % 32); }
                    clf[1 + n] = b as u32;
                    n += 1;
                }
                clf[0] = n as u32;
                // mixer
                let kind = cfg.kinds[l];
                let (in_rows, out_cols, nh, hw) = mix_dims(cfg, kind);
                let on_head = |h: usize| !(st.policy.heads && heads_clean(cfg, kind)) || m.head(l, h);
                let hb = self.rm_words(io.at(l, io.o_hb), (io.o_sf - io.o_hb) / 4);
                hb.fill(0);
                for h in 0..nh { if on_head(h) { hb[h / 32] |= 1 << (h % 32); } }
                // rows of a head: DeltaNet z rows conv + h dv ..; attention q|gate rows h 2 hd .. (2 hd rows)
                let row_head = |r: usize| -> Option<usize> {
                    match kind {
                        LayerKind::DeltaNet => { let z0 = cfg.conv_dim(); if r >= z0 && r < z0 + cfg.dn_v_dim() { Some((r - z0) / cfg.dn_dv) } else { None } }
                        LayerKind::Attention => if r < 2 * cfg.q_dim() { Some(r / (2 * cfg.head_dim)) } else { None },
                    }
                };
                let rbm = self.rm_words(io.at(l, io.o_rbm), (io.o_clm - io.o_rbm) / 4);
                rbm.fill(0);
                for t in 0..in_rows.div_ceil(TILE) {
                    let on = (t * TILE..((t + 1) * TILE).min(in_rows)).any(|r| row_head(r).is_none_or(on_head));
                    if on { rbm[t / 32] |= 1 << (t % 32); }
                }
                let ng = out_cols / 64;
                let clm = self.rm_words(io.at(l, io.o_clm), 1 + ng);
                let mut n = 0;
                for g in 0..ng {
                    if (g * 64..(g + 1) * 64).any(|c| on_head(c / hw)) { clm[1 + n] = g as u32; n += 1; }
                }
                clm[0] = n as u32;
            }
        }
    }

    /// Seen bits of the passes since the last take, as a ReadMask, then cleared.
    fn rm_take_seen(&self, st: &RmState) -> ReadMask {
        let cfg = &self.cfg;
        let io = st.io;
        let mut m = ReadMask::empty(cfg);
        for l in 0..cfg.layers {
            unsafe {
                let sf = self.rm_words(io.at(l, io.o_sf), (io.o_sh - io.o_sf) / 4);
                for b in 0..ffn_blocks(cfg) { if (sf[b / 32] >> (b % 32)) & 1 == 1 { m.set_ffn(l, b, true); } }
                sf.fill(0);
                let nh = layer_heads(cfg, l);
                let sh = self.rm_words(io.at(l, io.o_sh), nh.div_ceil(32));
                for h in 0..nh { if (sh[h / 32] >> (h % 32)) & 1 == 1 { m.set_head(l, h, true); } }
                sh.fill(0);
            }
        }
        m
    }
    fn rm_set_flag(&self, st: &RmState, full: bool) { unsafe { self.rm_words(st.io.fl, 1)[0] = full as u32; } }

    // ---- public API -------------------------------------------------------------------------------------------

    /// True when the engine was opened with `QwenOptions.read_mask` (the region and kernels exist).
    pub fn read_mask_enabled(&self) -> bool { self.rm.is_some() }

    /// Set the decode read mask's PROFILE (the CLI's activity profile); None = mask off = today's code path,
    /// bit-identical. Decode passes read profile ∪ prompt-active ∪ learned; prefill never uses it. Takes effect from the
    /// next prefill (and at once for the rest of the current request when one is decoding). Engine idle only.
    /// Errors: engine opened without `read_mask`, `--gemv old` (masked kernels exist for the Mma family only), shape.
    pub fn set_read_mask(&self, profile: Option<&ReadMask>) -> Result<(), String> {
        let c = self.rm_cell()?;
        let mut st = c.borrow_mut();
        match profile {
            None => {
                st.profile = None;
                st.live = false;
                st.next_full = false;
                self.rm_set_flag(&st, false);
            }
            Some(p) => {
                p.check(&self.cfg)?;
                if self.opt.gemv != crate::qwen_layers::GemvImpl::Mma {
                    return Err("read mask: the masked kernels exist for --gemv mma only".into());
                }
                st.profile = Some(p.clone());
                if st.live {
                    let r = p.union(&st.prompt).union(&st.learned);
                    self.rm_write(&st, &r);
                    st.read = r;
                }
            }
        }
        Ok(())
    }
    /// The profile set with `set_read_mask` (None = off).
    pub fn read_mask(&self) -> Option<ReadMask> { self.rm.as_ref().and_then(|c| c.borrow().profile.clone()) }
    /// The read set decode passes use now (profile ∪ prompt-active ∪ learned), None when no masked decode is active.
    pub fn decode_mask(&self) -> Option<ReadMask> {
        self.rm.as_ref().and_then(|c| { let s = c.borrow(); if s.live { Some(s.read.clone()) } else { None } })
    }
    /// Blocks / heads the prompt of the current sequence activated (all prefills since `reset`), None without the
    /// region. Recorded whether or not a mask is set.
    pub fn prompt_active(&self) -> Option<ReadMask> { self.rm.as_ref().map(|c| c.borrow().prompt.clone()) }
    /// Blocks / heads full ("read everything") passes found active outside the read set, since the last take.
    pub fn learned(&self) -> Option<ReadMask> { self.rm.as_ref().map(|c| c.borrow().learned.clone()) }
    /// `learned`, then cleared (the read set of a request in progress keeps them until its next prefill).
    pub fn take_learned(&self) -> Option<ReadMask> {
        self.rm.as_ref().map(|c| { let mut s = c.borrow_mut(); let e = ReadMask::empty(&self.cfg); std::mem::replace(&mut s.learned, e) })
    }
    pub fn mask_stats(&self) -> MaskStats { self.rm.as_ref().map_or_else(MaskStats::default, |c| c.borrow().stats) }
    pub fn reset_mask_stats(&self) { if let Some(c) = &self.rm { c.borrow_mut().stats = MaskStats::default(); } }
    pub fn mask_policy(&self) -> MaskPolicy { self.rm.as_ref().map_or_else(MaskPolicy::default, |c| c.borrow().policy) }
    /// Thresholds / rule switches; with a decode in progress the head switch applies from the next prefill.
    pub fn set_mask_policy(&self, p: MaskPolicy) { if let Some(c) = &self.rm { c.borrow_mut().policy = p; } }

    // ---- engine hooks -----------------------------------------------------------------------------------------

    /// Mask offsets for a pass: Some whenever the region exists (the seen recorder runs on every pass); `apply` only
    /// for exact decode / verify passes of a request with a mask.
    pub(crate) fn rm_io(&self, apply: bool) -> Option<RmIo> {
        self.rm.as_ref().map(|c| {
            let s = c.borrow();
            RmIo { apply: apply && s.live, ffn_thr: s.policy.ffn_thr, head_thr: s.policy.head_thr, ..s.io }
        })
    }
    /// New sequence (`QwenEngine::reset`): prompt-active cleared, mask idle until the next prefill ends.
    pub(crate) fn rm_reset(&self) {
        let Some(c) = &self.rm else { return };
        let mut s = c.borrow_mut();
        s.prompt = ReadMask::empty(&self.cfg);
        s.live = false;
        s.next_full = false;
        self.rm_set_flag(&s, false);
        let _ = self.rm_take_seen(&s);
    }
    /// Prefill start: unmasked from here (the prompt is always read in full); seen bits cleared.
    pub(crate) fn rm_prefill_begin(&self) {
        let Some(c) = &self.rm else { return };
        let mut s = c.borrow_mut();
        s.live = false;
        s.next_full = false;
        self.rm_set_flag(&s, false);
        let _ = self.rm_take_seen(&s);
    }
    /// Prefill end (nothing in flight): prompt-active |= what the prefill activated; with a profile, decode reads
    /// profile ∪ prompt-active ∪ learned; the first decode pass is full when the prefill's token was unsure.
    pub(crate) fn rm_prefill_end(&self) {
        let Some(c) = &self.rm else { return };
        let mut s = c.borrow_mut();
        let seen = self.rm_take_seen(&s);
        s.prompt = s.prompt.union(&seen);
        let Some(p) = s.profile.clone() else { return };
        let r = p.union(&s.prompt).union(&s.learned);
        self.rm_write(&s, &r);
        s.read = r;
        s.live = true;
        let p0 = self.pos();
        let g = if p0 > 0 { let (_, a, b) = self.rec_token(p0 - 1); a - b } else { f32::INFINITY };
        s.next_full = g < s.policy.gap_thr;
        if s.next_full { s.stats.full_gap += 1; }
    }
    /// Before committing a masked-capable pass: sets its flag from the previous decision and counts it.
    pub(crate) fn rm_before_pass(&self) {
        let Some(c) = &self.rm else { return };
        let mut s = c.borrow_mut();
        if !s.live { return; }
        let f = s.next_full;
        s.cur_full = f;
        self.rm_set_flag(&s, f);
        if f { s.stats.full_passes += 1 } else { s.stats.masked_passes += 1 }
    }
    /// After the pass finished: `gaps` = bf16 top-1 − top-2 of the tokens it emitted. Full pass: what it found active
    /// outside the read set is learned (and read from now on). Masked pass: out-of-profile activity in the read rows.
    /// Decides whether the next pass reads everything.
    pub(crate) fn rm_after_pass(&self, gaps: &[f32]) {
        let Some(c) = &self.rm else { return };
        let mut s = c.borrow_mut();
        if !s.live { return; }
        let seen = self.rm_take_seen(&s);
        let mut oop = false;
        if s.cur_full {
            let new = seen.minus(&s.read);
            let (nb, nh) = (new.count_ffn(), new.count_heads());
            if nb + nh > 0 {
                s.stats.learned_blocks += nb as u64;
                s.stats.learned_heads += nh as u64;
                s.learned = s.learned.union(&new);
                let r = s.read.union(&new);
                self.rm_write(&s, &r);
                s.read = r;
            }
        } else if s.policy.oop_min > 0 {
            let prof = s.profile.clone().unwrap_or_else(|| ReadMask::empty(&self.cfg));
            let o = seen.and(&s.read).minus(&prof);
            let mut n = o.count_ffn();
            if s.policy.heads { n += o.count_heads(); }
            oop = n >= s.policy.oop_min;
        }
        let gap = gaps.iter().any(|g| !(*g >= s.policy.gap_thr));
        if gap { s.stats.full_gap += 1; }
        if oop { s.stats.full_oop += 1; }
        s.next_full = gap || oop;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 86 blocks (two u64 words), 70 DeltaNet heads (two u64 words, three u32 words), 4 attention heads.
    fn test_cfg() -> QwenConfig {
        QwenConfig { hidden: 512, inter: 1024 + 64 * 70, layers: 4, kinds: vec![LayerKind::DeltaNet, LayerKind::DeltaNet, LayerKind::DeltaNet, LayerKind::Attention],
                     heads: 4, kv_heads: 2, head_dim: 256, rope_dims: 64, rope_theta: 1e7, dn_kh: 2, dn_vh: 70, dn_dk: 128, dn_dv: 128, conv_k: 4,
                     vocab: 2048, eps: 1e-6, eos: vec![1], mtp_layers: 0, tie_word_embeddings: false }
    }

    #[test]
    fn set_ops_and_shapes() {
        let c = test_cfg();
        let f = ReadMask::full(&c);
        f.check(&c).unwrap();
        assert_eq!(f.count_ffn(), 4 * ffn_blocks(&c));
        assert_eq!(f.count_heads(), 3 * 70 + 4);
        let e = ReadMask::empty(&c);
        assert!(e.is_empty());
        let r = ReadMask::random(&c, 0.5, 3);
        r.check(&c).unwrap();
        assert_eq!(r.union(&f), f);
        assert_eq!(r.and(&e), e);
        assert_eq!(r.union(&f.minus(&r)), f);
        assert_eq!(r.minus(&r), e);
        let mut m = e.clone();
        m.set_ffn(1, 85, true);
        m.set_head(0, 69, true);
        assert!(m.ffn(1, 85) && m.head(0, 69) && !m.ffn(1, 84));
        assert_eq!((m.count_ffn(), m.count_heads()), (1, 1));
        let mut bad = f.clone();
        bad.ffn_blocks[0][1] = u64::MAX; // bits past 86 blocks
        assert!(bad.check(&c).is_err());
    }

    #[test]
    fn layout_fits() {
        let c = test_cfg();
        let (io, bytes) = layout(&c);
        assert!(io.o_rbf < io.o_clf && io.o_clf < io.o_rbm && io.o_rbm < io.o_clm && io.o_clm < io.o_hb && io.o_hb < io.o_sf && io.o_sf < io.o_sh);
        assert!(io.o_sh + 4 * 70usize.div_ceil(32) <= io.stride);
        assert_eq!(bytes, 16 + c.layers * (io.stride + io.nstride));
        assert!(io.nstride >= 2 * (c.inter + c.dn_v_dim().max(c.q_dim())));
    }
}
