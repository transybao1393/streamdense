//! Read mask, prompt predictor and exploration, CLI side (activity-only, full read + learn). Three parts, all
//! streamed from disk through fixed stack buffers (no heap in steady state; the four `ReadMask` boxes of `MaskCtl`
//! are allocated once at open, ~37 KB each):
//! 1. PROFILE -> READ MASK. ONE profile pooled over all labels (domains are not an input): for every part of the
//!    window (`[watcher] profile_window`: now / session = the pooled decayed sets, all = every label slot + the held
//!    slot), per layer, the 64-neuron FFN blocks and the heads holding `keep_frac` of that part's mass are kept
//!    (`store::cold_units`, the `Store::plan` rule: coldest first while the removed mass <= (1 - keep) x total). Keep =
//!    the union over the parts, plus the blocks exploration found active (`joined`). A part with no mass in a layer
//!    adds nothing; a layer with no mass in any part keeps everything. Masked = never active in the window.
//! 2. CPU PROMPT PREDICTOR (user: "from the prompt directly"). A table on disk (`predict.bin`, next to `store.bin`,
//!    fixed size <= `PREDICT_CAP` = 16 MB): token ids hash into `nb` buckets; per bucket the number of requests it was
//!    in (u32) and per unit (64-block or head) an u8 count of those requests where the unit was active (halved with
//!    the bucket count at 255, so ratios stay); per unit an u32 count over all requests. Prediction for a request's
//!    new prompt tokens: unit u is predicted active when max(P(u), max over the request's `KB` rarest known buckets of
//!    (c_bu + ALPHA P(u)) / (n_b + ALPHA)) >= THETA. Active (truth) = the job's own keep set at `keep_frac` (same rule
//!    as the profile, on the job's recorder window). Stage A: scored only (precision / recall overall and on the
//!    masked units, bytes per request), then learned. Stage B: the same prediction drives SSD prefetch.
//! 3. EXPLORATION. In idle time, every `EXPLORE_EVERY_S`, a rotating share (`explore_frac`, <=
//!    `EXPLORE_MAX`) of the masked units, ranked by the predictor's P(u) (most likely active first), is scheduled:
//!    a unit explored in round r is not eligible again before r + ceil(masked / share), so the whole masked set
//!    rotates. Bookkeeping per unit on disk (last round, found active, joined round). Units the read finds active
//!    join the profile (kept) for `JOIN_TTL_ROUNDS` rounds (by then the recorder has measured them in the store). The
//!    GPU read is the `Explorer` HOOK (engine API not ready): never called with mask = off or dry (in dry every block
//!    is read anyway, so there is no blind spot); with "on" and no hook the rounds are scheduled, not read.
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};
use streamdense_model::unload::{head_bytes, q4_row_bytes, MASK_BLOCK, READ_BW};
use streamdense_model::LayerKind;

use crate::store::{check, cold_units, JobSrc, Rec, Sel, Store, CH, HELD, MAXH, MAXL, MAXU, NOW, SESS, TAUS_S};
use crate::watcher::{Dims, NP, PLAN_MIN_TOKENS};

/// u64 words per layer: FFN 64-blocks (<= MAXU blocks), heads (<= MAXH)
pub const BW: usize = MAXU / 64;
pub const HW: usize = MAXH / 64;

fn r3(x: f64) -> f64 { if x.is_finite() { (x * 1000.0).round() / 1000.0 } else { 0.0 } }
fn ratio(a: u64, b: u64) -> Value { if b > 0 { json!(r3(a as f64 / b as f64)) } else { Value::Null } }

// ---- config ----------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Off, Dry, On }
impl Mode {
    pub fn parse(s: &str) -> Mode { match s { "dry" => Mode::Dry, "on" => Mode::On, _ => Mode::Off } }
    pub fn name(&self) -> &'static str { match self { Mode::Off => "off", Mode::Dry => "dry", Mode::On => "on" } }
}

/// The profile window: which pooled parts the keep set is the union of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window { pub now: bool, pub session: bool, pub all: bool }
impl Window {
    pub fn parse(s: &str) -> Window {
        let has = |p: &str| s.split('+').any(|x| x.trim() == p);
        let w = Window { now: has("now"), session: has("session"), all: has("all") };
        if w.now || w.session || w.all { w } else { Window { now: false, session: true, all: true } }
    }
    pub fn name(&self) -> &'static str {
        match (self.now, self.session, self.all) {
            (true, false, false) => "now", (false, true, false) => "session", (false, false, true) => "all", (true, false, true) => "now+all",
            (false, true, true) => "session+all", (true, true, true) => "now+session+all", (true, true, false) => "now+session", _ => "session+all",
        }
    }
    fn sels(&self, out: &mut [Sel; 3]) -> usize {
        let mut n = 0;
        if self.now { out[n] = Sel::Set(NOW); n += 1; }
        if self.session { out[n] = Sel::Set(SESS); n += 1; }
        if self.all { out[n] = Sel::Labels; n += 1; }
        n
    }
}

// ---- the read mask ---------------------------------------------------------------------------------------------------

/// A set of units (FFN 64-blocks + heads), fixed size (~37 KB; boxed once). As the read mask: bit set = KEEP (read),
/// clear = masked (skipped). Engine format (`unload.rs` module doc): per layer the sorted hot 64-block list
/// (`hot_blocks`) and the hot-head bits (`head_bits`, u64 words). Unit index: FFN l x blocks + b, then heads layer by
/// layer.
#[derive(Clone)]
pub struct ReadMask {
    pub layers: usize,
    pub inter: usize,
    /// 64-blocks per layer
    pub blocks: usize,
    /// heads per layer, their offset in the unit index (cumulative) and their start in the store's unified layout
    pub nh: [u16; MAXL],
    hoff: [u32; MAXL + 1],
    pub hstart: [u32; MAXL],
    pub kinds_attn: [bool; MAXL],
    pub ffn: [[u64; BW]; MAXL],
    pub head: [[u64; HW]; MAXL],
}

impl ReadMask {
    /// Empty set shaped like the store's model (setup; callers box it once).
    pub fn shape(st: &Store) -> ReadMask {
        let d = &st.d;
        let mut m = ReadMask { layers: d.layers.min(MAXL), inter: d.inter, blocks: d.inter.div_ceil(MASK_BLOCK).min(MAXU), nh: [0; MAXL], hoff: [0; MAXL + 1],
                               hstart: [0; MAXL], kinds_attn: [false; MAXL], ffn: [[0; BW]; MAXL], head: [[0; HW]; MAXL] };
        for l in 0..m.layers {
            let r = st.heads_of(l);
            m.nh[l] = r.len().min(MAXH) as u16;
            m.hstart[l] = r.start as u32;
            m.hoff[l + 1] = m.hoff[l] + m.nh[l] as u32;
            m.kinds_attn[l] = d.tmpl.kinds[l] == LayerKind::Attention;
        }
        m
    }
    pub fn boxed(st: &Store) -> Box<ReadMask> { Box::new(ReadMask::shape(st)) }
    pub fn n_ffn(&self) -> usize { self.layers * self.blocks }
    pub fn n_heads(&self) -> usize { self.hoff[self.layers] as usize }
    pub fn units(&self) -> usize { self.n_ffn() + self.n_heads() }
    pub fn clear(&mut self) { for l in 0..MAXL { self.ffn[l] = [0; BW]; self.head[l] = [0; HW]; } }
    /// every unit set
    #[cfg(test)]
    pub fn fill(&mut self) {
        self.clear();
        for l in 0..self.layers { for b in 0..self.blocks { self.set_ffn(l, b, true); } for h in 0..self.nh[l] as usize { self.set_head(l, h, true); } }
    }
    #[inline] pub fn ffn(&self, l: usize, b: usize) -> bool { self.ffn[l][b >> 6] >> (b & 63) & 1 != 0 }
    #[inline] pub fn set_ffn(&mut self, l: usize, b: usize, v: bool) { let w = &mut self.ffn[l][b >> 6]; if v { *w |= 1 << (b & 63) } else { *w &= !(1 << (b & 63)) } }
    #[inline] pub fn head(&self, l: usize, h: usize) -> bool { self.head[l][h >> 6] >> (h & 63) & 1 != 0 }
    #[inline] pub fn set_head(&mut self, l: usize, h: usize, v: bool) { let w = &mut self.head[l][h >> 6]; if v { *w |= 1 << (h & 63) } else { *w &= !(1 << (h & 63)) } }
    /// unit -> (is head, layer, block / head index)
    pub fn locate(&self, u: usize) -> (bool, usize, usize) {
        let nf = self.n_ffn();
        if u < nf { return (false, u / self.blocks, u % self.blocks); }
        let x = (u - nf) as u32;
        // the last layer whose offset <= x (layers with no heads are skipped by the strict upper bound)
        let (mut lo, mut hi) = (0usize, self.layers);
        while hi - lo > 1 { let m = (lo + hi) / 2; if self.hoff[m] <= x { lo = m } else { hi = m } }
        while lo + 1 < self.layers && self.hoff[lo + 1] <= x { lo += 1; }
        (true, lo, (x - self.hoff[lo]) as usize)
    }
    pub fn unit(&self, u: usize) -> bool { let (h, l, i) = self.locate(u); if h { self.head(l, i) } else { self.ffn(l, i) } }
    pub fn set_unit(&mut self, u: usize, v: bool) { let (h, l, i) = self.locate(u); if h { self.set_head(l, i, v) } else { self.set_ffn(l, i, v) } }
    pub fn or_from(&mut self, o: &ReadMask) {
        for l in 0..MAXL { for w in 0..BW { self.ffn[l][w] |= o.ffn[l][w]; } for w in 0..HW { self.head[l][w] |= o.head[l][w]; } }
    }
    /// (FFN blocks set, heads set)
    pub fn count(&self) -> (usize, usize) {
        let mut f = 0;
        let mut h = 0;
        for l in 0..self.layers { f += self.ffn[l].iter().map(|w| w.count_ones() as usize).sum::<usize>(); h += self.head[l].iter().map(|w| w.count_ones() as usize).sum::<usize>(); }
        (f, h)
    }
    /// Engine format: the sorted kept 64-block indices of layer l into `out`; returns the count.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn hot_blocks(&self, l: usize, out: &mut [u32]) -> usize {
        let mut n = 0;
        for b in 0..self.blocks { if self.ffn(l, b) && n < out.len() { out[n] = b as u32; n += 1; } }
        n
    }
    /// Engine format (`streamdense_model::readmask::ReadMask`, bit set = read): the words copied in place (no allocation
    /// when `e` is already shaped, e.g. `ReadMask::empty(cfg)`). The engine masks whole 64-blocks only (inter / 64): a
    /// partial last block stays read. False when the shapes differ. HOOK (mask = "on", once the engine
    /// hook is wired): the gateway passes this to `QwenEngine::set_read_mask` between jobs.
    /// From the engine format (the engine's `prompt_active()`): words copied in place; units the engine does not cover
    /// (a partial last block) are clear. False when the shapes differ (then the caller uses the job keep set).
    pub fn load_engine(&mut self, e: &streamdense_model::readmask::ReadMask) -> bool {
        if e.ffn_blocks.len() != self.layers || e.heads.len() != self.layers { return false; }
        self.clear();
        for l in 0..self.layers {
            if e.ffn_blocks[l].len() > BW || e.heads[l].len() > HW { self.clear(); return false; }
            for (w, x) in e.ffn_blocks[l].iter().enumerate() { self.ffn[l][w] = *x; }
            for (w, x) in e.heads[l].iter().enumerate() { self.head[l][w] = *x; }
        }
        true
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn to_engine(&self, e: &mut streamdense_model::readmask::ReadMask) -> bool {
        let full = self.inter / MASK_BLOCK;
        if e.ffn_blocks.len() != self.layers || e.heads.len() != self.layers { return false; }
        for l in 0..self.layers {
            let (f, h) = (&mut e.ffn_blocks[l], &mut e.heads[l]);
            if f.len() != full.div_ceil(64) || h.len() != (self.nh[l] as usize).div_ceil(64) { return false; }
            for (w, x) in f.iter_mut().enumerate() { let n = full.saturating_sub(w * 64).min(64); *x = self.ffn[l][w] & if n == 64 { u64::MAX } else { (1u64 << n) - 1 }; }
            for (w, x) in h.iter_mut().enumerate() { *x = self.head[l][w]; }
        }
        true
    }
}

/// Bytes a masked unit saves per pass (stage A), the `Store::plan` byte rule: a 64-block = 64 x (gate + up rows + its
/// down columns) + the down group's scale / bias column; a head = its in / out projection bytes (`head_bytes().0`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost { pub block: u64, pub last_block: u64, pub dn_head: u64, pub attn_head: u64, pub blocks: usize }
impl Cost {
    pub fn of(d: &Dims) -> Cost {
        let t = &d.tmpl;
        let row = q4_row_bytes(t.hidden, t.group, t.bits);
        let per_neuron = 2 * row + (t.hidden * t.bits / 8) as u64;
        let per_group = (t.hidden * 4) as u64;
        let bl = |len: usize| len as u64 * per_neuron + if len == t.group.max(1) { per_group } else { 0 };
        let blocks = d.inter.div_ceil(MASK_BLOCK);
        let last = d.inter - (blocks.max(1) - 1) * MASK_BLOCK;
        Cost { block: bl(MASK_BLOCK.min(d.inter)), last_block: bl(last), dn_head: head_bytes(t, LayerKind::DeltaNet).0, attn_head: head_bytes(t, LayerKind::Attention).0, blocks }
    }
    pub fn unit(&self, m: &ReadMask, head: bool, l: usize, i: usize) -> u64 {
        if head { if m.kinds_attn[l] { self.attn_head } else { self.dn_head } } else if i + 1 == self.blocks { self.last_block } else { self.block }
    }
}

/// What a mask skips (dry run numbers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Report {
    pub ts: f64,
    pub build_ms: f64,
    pub window: &'static str,
    pub keep: f64,
    /// pooled tokens: now / session decayed to `ts`, all-time
    pub tokens_now: f64,
    pub tokens_session: f64,
    pub tokens_all: u64,
    pub ready: bool,
    pub layers: usize,
    pub blocks: usize,
    pub masked_blocks: usize,
    pub heads: usize,
    pub masked_heads: usize,
    /// units kept only because exploration found them active
    pub joined: usize,
    pub ffn_skip: u64,
    pub head_skip: u64,
    pub per_layer_masked: [u16; MAXL],
    pub per_layer_masked_heads: [u16; MAXL],
}
impl Default for Report {
    fn default() -> Report {
        Report { ts: 0.0, build_ms: 0.0, window: "", keep: 0.0, tokens_now: 0.0, tokens_session: 0.0, tokens_all: 0, ready: false, layers: 0, blocks: 0, masked_blocks: 0,
                 heads: 0, masked_heads: 0, joined: 0, ffn_skip: 0, head_skip: 0, per_layer_masked: [0; MAXL], per_layer_masked_heads: [0; MAXL] }
    }
}
impl Report {
    pub fn skip_bytes(&self) -> u64 { self.ffn_skip + self.head_skip }
    pub fn ms(&self) -> f64 { self.skip_bytes() as f64 / READ_BW * 1e3 }
    pub fn json(&self) -> Value {
        json!({"ts": self.ts, "build_ms": r3(self.build_ms), "window": self.window, "keep_frac": self.keep,
            "tokens": {"now": self.tokens_now.round(), "session": self.tokens_session.round(), "all": self.tokens_all}, "ready": self.ready,
            "blocks": self.blocks, "masked_blocks": self.masked_blocks, "masked_block_frac": ratio(self.masked_blocks as u64, self.blocks as u64),
            "heads": self.heads, "masked_heads": self.masked_heads, "joined": self.joined,
            "skip_bytes_per_pass": self.skip_bytes(), "skip_ffn_bytes": self.ffn_skip, "skip_head_bytes": self.head_skip,
            "skip_gb_per_pass": r3(self.skip_bytes() as f64 / 1e9), "ms_saved_per_pass": r3(self.ms()),
            "per_layer_masked": &self.per_layer_masked[..self.layers], "per_layer_masked_heads": &self.per_layer_masked_heads[..self.layers]})
    }
}

/// Masked counts + bytes of `m` (stack only).
pub fn report(m: &ReadMask, c: &Cost, out: &mut Report) {
    (out.layers, out.blocks, out.heads, out.masked_blocks, out.masked_heads, out.ffn_skip, out.head_skip) = (m.layers, m.n_ffn(), m.n_heads(), 0, 0, 0, 0);
    for l in 0..m.layers {
        let (mut f, mut h) = (0u16, 0u16);
        for b in 0..m.blocks { if !m.ffn(l, b) { f += 1; out.ffn_skip += c.unit(m, false, l, b); } }
        for i in 0..m.nh[l] as usize { if !m.head(l, i) { h += 1; out.head_skip += c.unit(m, true, l, i); } }
        out.per_layer_masked[l] = f;
        out.per_layer_masked_heads[l] = h;
        out.masked_blocks += f as usize;
        out.masked_heads += h as usize;
    }
}

/// Mark the units of one layer's masses holding `keep` of the mass (`cold_units` complement). False: no mass.
fn keep_units(mass: &[f64], keep: f64, cold: &mut [bool], set: &mut dyn FnMut(usize)) -> bool {
    if !(mass.iter().sum::<f64>() > 0.0) { return false; }
    cold_units(mass, keep, cold);
    for (i, c) in cold[..mass.len()].iter().enumerate() { if !c { set(i); } }
    true
}

/// The profile (keep set) of the window at clock `t` into `out` (cleared first). Returns (now, session tokens decayed,
/// all-time tokens). Streams the store through stack buffers; no heap.
pub fn build_profile(st: &mut Store, win: Window, keep: f64, t: f64, out: &mut ReadMask) -> (f64, f64, u64) {
    st.t = t;
    out.clear();
    let (layers, inter, nb) = (out.layers, out.inter, out.blocks);
    let mut sels = [Sel::Labels; 3];
    let ns = win.sels(&mut sels);
    let mut buf = [0f32; CH];
    let mut rec: Rec = [0u8; CH * 4 + 16];
    let mut mass = [0f64; MAXU];
    let mut cold = [false; MAXU];
    let mut hm = [0f32; MAXH];
    for l in 0..layers {
        let nh = out.nh[l] as usize;
        let hs = out.hstart[l] as usize;
        let (mut anyf, mut anyh) = (false, false);
        for &sel in &sels[..ns] {
            mass[..nb].fill(0.0);
            let mut o = 0;
            while o < inter {
                let len = CH.min(inter - o);
                st.read_sel(sel, l * inter + o, &mut buf[..len], &mut rec);
                for j in 0..len { let x = buf[j]; if x > 0.0 && x.is_finite() { mass[(o + j) / MASK_BLOCK] += x as f64; } }
                o += len;
            }
            anyf |= keep_units(&mass[..nb], keep, &mut cold, &mut |b| out.set_ffn(l, b, true));
            st.read_sel(sel, hs, &mut hm[..nh], &mut rec);
            for h in 0..nh { mass[h] = if hm[h] > 0.0 && hm[h].is_finite() { hm[h] as f64 } else { 0.0 }; }
            anyh |= keep_units(&mass[..nh], keep, &mut cold, &mut |h| out.set_head(l, h, true));
        }
        if !anyf { for b in 0..nb { out.set_ffn(l, b, true); } }
        if !anyh { for h in 0..nh { out.set_head(l, h, true); } }
    }
    let dec = |s: usize, k: usize| { let h = st.hdr(s); h.rows_d * (-(t - h.t0).max(0.0) / TAUS_S[k]).exp() };
    let all = (0..NP).chain([HELD]).map(|s| st.hdr(s).tokens).sum();
    (dec(NOW, 0), dec(SESS, 1), all)
}

/// One job's active units (truth for the predictor): per layer, the 64-blocks / heads holding `keep` of the JOB's
/// own mass (the profile rule on the job's recorder window). Stack only.
pub fn active_units(src: &dyn JobSrc, keep: f64, out: &mut ReadMask) {
    out.clear();
    let (inter, nb) = (out.inter, out.blocks);
    let mut buf = [0f32; CH];
    let mut mass = [0f64; MAXU];
    let mut cold = [false; MAXU];
    let mut hm = [0f32; MAXH];
    for l in 0..out.layers {
        mass[..nb].fill(0.0);
        let mut o = 0;
        while o < inter {
            let len = CH.min(inter - o);
            src.read(l * inter + o, &mut buf[..len]);
            for j in 0..len { let x = buf[j]; if x > 0.0 && x.is_finite() { mass[(o + j) / MASK_BLOCK] += x as f64; } }
            o += len;
        }
        keep_units(&mass[..nb], keep, &mut cold, &mut |b| out.set_ffn(l, b, true));
        let nh = out.nh[l] as usize;
        src.read(out.hstart[l] as usize, &mut hm[..nh]);
        for h in 0..nh { mass[h] = if hm[h] > 0.0 && hm[h].is_finite() { hm[h] as f64 } else { 0.0 }; }
        keep_units(&mass[..nh], keep, &mut cold, &mut |h| out.set_head(l, h, true));
    }
}

// ---- the CPU prompt predictor ------------------------------------------------------------------------------------------

/// Disk cap of `predict.bin` (proposed 2026-11-01: 16 MB next to the 64 MB store; 27B: 787 buckets).
pub const PREDICT_CAP: u64 = 16_000_000;
pub const NBMAX: usize = 4096;
/// buckets used per request (the rarest known ones; learning: the rarest ones incl. new)
pub const KB: usize = 64;
/// a bucket needs this many requests before it predicts
pub const N_MIN: u32 = 3;
pub const THETA: f32 = 0.5;
pub const ALPHA: f32 = 2.0;
/// aggregated predictor scores go to notes.jsonl every this many scored requests
pub const NOTE_EVERY: u64 = 8;
const CU: usize = 4096;
const PH: u64 = 4096;
const PSLOT: u64 = 2048;
const PMAGIC: &[u8; 8] = b"SDPRED01";
const NF: usize = 17;
const PHB: usize = 8 + 8 * (3 + NF) + 8;

/// The request's new prompt tokens: (id, count) pairs.
pub trait Tokens { fn each(&self, f: &mut dyn FnMut(u32, u32)); }
/// Explicit list (tests).
#[cfg(test)]
pub struct TokList<'a>(pub &'a [u32]);
#[cfg(test)]
impl Tokens for TokList<'_> { fn each(&self, f: &mut dyn FnMut(u32, u32)) { for &t in self.0 { f(t, 1); } } }
/// The recorder in place: embedding rows read in this job minus the tokens it emitted (= the fed prompt tokens, i.e.
/// the non-reused part; plus draft rows the verifier rejected). HOOK: replace by the worker's exact new-token ids when
/// the engine / gateway hands them over.
impl Tokens for crate::store::TapSrc {
    fn each(&self, f: &mut dyn FnMut(u32, u32)) {
        use std::sync::atomic::Ordering::Relaxed;
        self.tap.embed_counts.with_counts(|e| self.tap.out_counts.with_counts(|o| {
            for (i, x) in e.iter().enumerate() {
                let a = x.load(Relaxed);
                if a == 0 { continue; }
                let b = o.get(i).map_or(0, |y| y.load(Relaxed));
                if a > b { f(i as u32, a - b); }
            }
        }));
    }
}

/// Persistent counters (header A / B copies, seq + checksum).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PHdr {
    pub seq: u64,
    pub jobs: u64,
    pub toks: u64,
    pub scored: u64,
    pub tp: u64,
    pub fp: u64,
    pub fneg: u64,
    pub o_act: u64,
    pub o_tp: u64,
    pub o_fp: u64,
    pub bytes_pred: u64,
    pub bytes_miss: u64,
    pub rounds: u64,
    pub sched: u64,
    pub read: u64,
    pub found: u64,
    pub expired: u64,
}
impl PHdr {
    fn arr(&self) -> [u64; NF] {
        [self.jobs, self.toks, self.scored, self.tp, self.fp, self.fneg, self.o_act, self.o_tp, self.o_fp, self.bytes_pred, self.bytes_miss, self.rounds, self.sched, self.read, self.found, self.expired, 0]
    }
    fn from(seq: u64, a: &[u64; NF]) -> PHdr {
        PHdr { seq, jobs: a[0], toks: a[1], scored: a[2], tp: a[3], fp: a[4], fneg: a[5], o_act: a[6], o_tp: a[7], o_fp: a[8], bytes_pred: a[9], bytes_miss: a[10],
               rounds: a[11], sched: a[12], read: a[13], found: a[14], expired: a[15] }
    }
}

/// One request's predictor score.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Score {
    pub tokens: u64,
    pub buckets: usize,
    /// buckets used for the prediction (known, rarest first)
    pub used: usize,
    pub units: usize,
    pub active: usize,
    pub predicted: usize,
    pub tp: usize,
    pub fp: usize,
    pub fneg: usize,
    /// masked units (no mask yet: 0) and, among them: active ("out of profile"), predicted + active, predicted + idle
    pub masked: usize,
    pub o_act: usize,
    pub o_tp: usize,
    pub o_fp: usize,
    /// bytes of the predicted masked units (what the CPU would add back / prefetch) and of the active masked units it
    /// missed (left to the fallback)
    pub bytes_pred: u64,
    pub bytes_miss: u64,
    pub ms: f64,
}
impl Score {
    pub fn add(&mut self, o: &Score) {
        self.tokens += o.tokens; self.buckets += o.buckets; self.used += o.used; self.units += o.units; self.active += o.active; self.predicted += o.predicted;
        self.tp += o.tp; self.fp += o.fp; self.fneg += o.fneg; self.masked += o.masked; self.o_act += o.o_act; self.o_tp += o.o_tp; self.o_fp += o.o_fp;
        self.bytes_pred += o.bytes_pred; self.bytes_miss += o.bytes_miss; self.ms += o.ms;
    }
    #[cfg(test)]
    pub fn precision(&self) -> f64 { if self.tp + self.fp > 0 { self.tp as f64 / (self.tp + self.fp) as f64 } else { f64::NAN } }
    #[cfg(test)]
    pub fn recall(&self) -> f64 { if self.tp + self.fneg > 0 { self.tp as f64 / (self.tp + self.fneg) as f64 } else { f64::NAN } }
    pub fn json(&self, n: u64) -> Value {
        let n = n.max(1) as f64;
        json!({"requests": n as u64, "tokens_mean": r3(self.tokens as f64 / n), "buckets_used_mean": r3(self.used as f64 / n),
            "active_mean": r3(self.active as f64 / n), "predicted_mean": r3(self.predicted as f64 / n),
            "precision": ratio(self.tp as u64, (self.tp + self.fp) as u64), "recall": ratio(self.tp as u64, (self.tp + self.fneg) as u64),
            "masked_mean": r3(self.masked as f64 / n), "out_of_profile_active_mean": r3(self.o_act as f64 / n),
            "masked_precision": ratio(self.o_tp as u64, (self.o_tp + self.o_fp) as u64), "masked_recall": ratio(self.o_tp as u64, self.o_act as u64),
            "bytes_predicted_per_request": (self.bytes_pred as f64 / n).round(), "bytes_missed_per_request": (self.bytes_miss as f64 / n).round(),
            "ms_mean": r3(self.ms / n)})
    }
}

fn rd_u32s(f: &File, off: u64, out: &mut [u32]) -> std::io::Result<()> {
    let b = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    f.read_exact_at(b, off)?;
    #[cfg(target_endian = "big")]
    for x in out.iter_mut() { *x = u32::from_le(*x); }
    Ok(())
}
fn wr_u32s(f: &File, off: u64, v: &mut [u32]) -> std::io::Result<()> {
    #[cfg(target_endian = "big")]
    for x in v.iter_mut() { *x = x.to_le(); }
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let r = f.write_all_at(b, off);
    #[cfg(target_endian = "big")]
    for x in v.iter_mut() { *x = u32::from_le(*x); }
    r
}
fn u64at(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

/// Bucket of a token id (multiplicative hash).
#[inline] pub fn bucket(id: u32, nb: usize) -> usize { ((id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize % nb.max(1) }

/// `predict.bin`: [header A / B (4096 B) | P(u) counts u32 [units] | bucket counts u32 [nb] | exploration u32 x 2
/// [units] (last round + found bit 31 | joined round) | bucket rows u8 [nb][units]]. Fixed size, written once at
/// creation. Derived statistics: no journal (a torn row is noise in counts that are re-learned).
pub struct Predictor {
    f: File,
    pub path: PathBuf,
    pub units: usize,
    pub nb: usize,
    pub h: PHdr,
}

impl Predictor {
    pub fn size(units: usize, nb: usize) -> u64 { PH + 8 * units as u64 + 4 * nb as u64 + 4 * units as u64 + (nb * units) as u64 }
    /// buckets that fit `cap` (<= NBMAX)
    pub fn nb_for(units: usize, cap: u64) -> usize { (cap.saturating_sub(PH + 12 * units as u64) / (units as u64 + 4)).min(NBMAX as u64) as usize }
    fn o_glob(&self) -> u64 { PH }
    fn o_bn(&self) -> u64 { PH + 4 * self.units as u64 }
    fn o_ex(&self) -> u64 { self.o_bn() + 4 * self.nb as u64 }
    fn o_rows(&self) -> u64 { self.o_ex() + 8 * self.units as u64 }
    pub fn disk_bytes(&self) -> u64 { self.f.metadata().map_or(0, |m| m.len()) }

    fn read_hdr(f: &File) -> Option<(usize, usize, PHdr)> {
        let mut best: Option<(usize, usize, PHdr)> = None;
        let mut b = [0u8; PHB];
        for c in 0..2u64 {
            if f.read_exact_at(&mut b, c * PSLOT).is_err() || &b[..8] != PMAGIC || check(&b[..PHB - 8]) != u64at(&b, PHB - 8) { continue; }
            let (units, nb, seq) = (u64at(&b, 8) as usize, u64at(&b, 16) as usize, u64at(&b, 24));
            let mut a = [0u64; NF];
            for (i, x) in a.iter_mut().enumerate() { *x = u64at(&b, 32 + i * 8); }
            if best.is_none_or(|x| seq >= x.2.seq) { best = Some((units, nb, PHdr::from(seq, &a))); }
        }
        best
    }
    fn write_hdr(&mut self) -> std::io::Result<()> {
        self.h.seq += 1;
        let mut b = [0u8; PHB];
        b[..8].copy_from_slice(PMAGIC);
        for (i, x) in [self.units as u64, self.nb as u64, self.h.seq].iter().chain(self.h.arr().iter()).enumerate() { b[8 + i * 8..16 + i * 8].copy_from_slice(&x.to_le_bytes()); }
        let c = check(&b[..PHB - 8]);
        b[PHB - 8..].copy_from_slice(&c.to_le_bytes());
        self.f.write_all_at(&b, (self.h.seq % 2) * PSLOT)
    }

    /// Open (create when missing or shaped for other units / cap) `dir/predict.bin`.
    pub fn open(dir: &Path, units: usize, cap: u64) -> std::io::Result<Predictor> {
        let nb = Predictor::nb_for(units, cap);
        if nb == 0 || units == 0 { return Err(std::io::Error::other(format!("predictor: cap {} B holds no bucket of {} units", cap, units))); }
        std::fs::create_dir_all(dir)?;
        let path = dir.join("predict.bin");
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
        let size = Predictor::size(units, nb);
        let ok = Predictor::read_hdr(&f).filter(|x| x.0 == units && x.1 == nb && f.metadata().is_ok_and(|m| m.len() == size));
        let mut p = Predictor { f, path, units, nb, h: PHdr::default() };
        match ok {
            Some((_, _, h)) => p.h = h,
            None => {
                p.f.set_len(0)?;
                let z = [0u8; 8192];
                let mut o = 0u64;
                while o < size { let l = (size - o).min(z.len() as u64) as usize; p.f.write_all_at(&z[..l], o)?; o += l as u64; }
                p.write_hdr()?;
                p.f.sync_all()?;
                eprintln!("predict: created {} ({} B: {} units x {} buckets, cap {} B)", p.path.display(), size, units, nb, cap);
            }
        }
        Ok(p)
    }
    /// Read only (CLI): None when missing / invalid.
    pub fn open_ro(dir: &Path) -> Option<Predictor> {
        let path = dir.join("predict.bin");
        let f = File::open(&path).ok()?;
        let (units, nb, h) = Predictor::read_hdr(&f)?;
        Some(Predictor { f, path, units, nb, h })
    }

    /// Distinct buckets of the request (bitset) -> token count.
    fn buckets(&self, toks: &dyn Tokens, set: &mut [u64; NBMAX / 64]) -> u64 {
        set.fill(0);
        let nb = self.nb;
        let mut n = 0u64;
        toks.each(&mut |id, c| { let b = bucket(id, nb); set[b >> 6] |= 1 << (b & 63); n += c as u64; });
        n
    }
    /// The `KB` rarest buckets of the set with count >= min_n (ties: lower index), into `out`; returns how many.
    fn select(&self, set: &[u64; NBMAX / 64], bn: &[u32], min_n: u32, out: &mut [u16; KB]) -> usize {
        let mut k = 0usize;
        for b in 0..self.nb {
            if set[b >> 6] >> (b & 63) & 1 == 0 || bn[b] < min_n { continue; }
            // insertion into the sorted (n asc, index asc) list of at most KB
            let mut i = k;
            while i > 0 && bn[out[i - 1] as usize] > bn[b] { i -= 1; }
            if i >= KB { continue; }
            let end = (k + 1).min(KB);
            for j in (i + 1..end).rev() { out[j] = out[j - 1]; }
            out[i] = b as u16;
            k = end;
        }
        k
    }
    /// Predicted units of the buckets `sel` into `pred` (cleared first). Streams the counts in CU chunks.
    fn predict(&self, sel: &[u16], bn: &[u32], pred: &mut ReadMask) -> std::io::Result<()> {
        pred.clear();
        let jobs = self.h.jobs;
        if jobs == 0 { return Ok(()); }
        let (mut g, mut sc, mut row) = ([0u32; CU], [0f32; CU], [0u8; CU]);
        let mut u0 = 0;
        while u0 < self.units {
            let len = CU.min(self.units - u0);
            rd_u32s(&self.f, self.o_glob() + 4 * u0 as u64, &mut g[..len])?;
            for j in 0..len { sc[j] = g[j] as f32 / jobs as f32; }
            for &b in sel {
                let n = bn[b as usize] as f32;
                self.f.read_exact_at(&mut row[..len], self.o_rows() + (b as usize * self.units + u0) as u64)?;
                for j in 0..len { let p = (row[j] as f32 + ALPHA * g[j] as f32 / jobs as f32) / (n + ALPHA); if p > sc[j] { sc[j] = p; } }
            }
            for j in 0..len { if sc[j] >= THETA { pred.set_unit(u0 + j, true); } }
            u0 += len;
        }
        Ok(())
    }
    /// Learn one request: P(u) counts + the rows of the buckets `sel` (+ their request counts, halved at 255).
    fn learn(&mut self, sel: &[u16], bn: &mut [u32], active: &ReadMask) -> std::io::Result<()> {
        let (mut g, mut row) = ([0u32; CU], [0u8; CU]);
        let mut u0 = 0;
        while u0 < self.units {
            let len = CU.min(self.units - u0);
            let o = self.o_glob() + 4 * u0 as u64;
            rd_u32s(&self.f, o, &mut g[..len])?;
            for j in 0..len { if active.unit(u0 + j) { g[j] = g[j].saturating_add(1); } }
            wr_u32s(&self.f, o, &mut g[..len])?;
            u0 += len;
        }
        for &b in sel {
            let b = b as usize;
            let halve = bn[b] + 1 >= 255;
            let mut u0 = 0;
            while u0 < self.units {
                let len = CU.min(self.units - u0);
                let o = self.o_rows() + (b * self.units + u0) as u64;
                self.f.read_exact_at(&mut row[..len], o)?;
                for j in 0..len {
                    let mut c = row[j] as u32 + active.unit(u0 + j) as u32;
                    if halve { c /= 2; }
                    row[j] = c.min(255) as u8;
                }
                self.f.write_all_at(&row[..len], o)?;
                u0 += len;
            }
            bn[b] = if halve { (bn[b] + 1) / 2 } else { bn[b] + 1 };
        }
        wr_u32s(&self.f, self.o_bn(), &mut bn[..self.nb])?;
        self.h.jobs += 1;
        Ok(())
    }

    /// One request (stage A): predict from its new tokens (state before it), score against `active` (and the masked
    /// units of `mask`), then learn it. `pred` = scratch. Stack only.
    pub fn job(&mut self, toks: &dyn Tokens, active: &ReadMask, mask: Option<&ReadMask>, cost: &Cost, pred: &mut ReadMask) -> std::io::Result<Score> {
        let t0 = std::time::Instant::now();
        let mut set = [0u64; NBMAX / 64];
        let ntok = self.buckets(toks, &mut set);
        let mut bn = [0u32; NBMAX];
        rd_u32s(&self.f, self.o_bn(), &mut bn[..self.nb])?;
        let mut sel = [0u16; KB];
        let k = self.select(&set, &bn, N_MIN, &mut sel);
        self.predict(&sel[..k], &bn, pred)?;
        let mut s = Score { tokens: ntok, buckets: set.iter().map(|w| w.count_ones() as usize).sum(), used: k, units: self.units, ..Default::default() };
        for u in 0..self.units.min(active.units()) {
            let (hd, l, i) = active.locate(u);
            let a = if hd { active.head(l, i) } else { active.ffn(l, i) };
            let p = if hd { pred.head(l, i) } else { pred.ffn(l, i) };
            s.active += a as usize;
            s.predicted += p as usize;
            match (a, p) { (true, true) => s.tp += 1, (false, true) => s.fp += 1, (true, false) => s.fneg += 1, _ => {} }
            if let Some(m) = mask {
                if !(if hd { m.head(l, i) } else { m.ffn(l, i) }) {
                    s.masked += 1;
                    let b = cost.unit(m, hd, l, i);
                    if a { s.o_act += 1; }
                    match (a, p) { (true, true) => s.o_tp += 1, (false, true) => s.o_fp += 1, (true, false) => s.bytes_miss += b, _ => {} }
                    if p { s.bytes_pred += b; }
                }
            }
        }
        let mut sl = [0u16; KB];
        let kl = self.select(&set, &bn, 0, &mut sl);
        self.learn(&sl[..kl], &mut bn, active)?;
        let h = &mut self.h;
        h.toks += ntok; h.scored += 1; h.tp += s.tp as u64; h.fp += s.fp as u64; h.fneg += s.fneg as u64; h.o_act += s.o_act as u64; h.o_tp += s.o_tp as u64;
        h.o_fp += s.o_fp as u64; h.bytes_pred += s.bytes_pred; h.bytes_miss += s.bytes_miss;
        self.write_hdr()?;
        s.ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(s)
    }

    /// P(u) of every unit in [u0, u0 + out.len()) (0 before any request).
    pub fn probs(&self, u0: usize, out: &mut [f32]) -> std::io::Result<()> {
        let mut g = [0u32; CU];
        let mut o = 0;
        while o < out.len() {
            let len = CU.min(out.len() - o);
            rd_u32s(&self.f, self.o_glob() + 4 * (u0 + o) as u64, &mut g[..len])?;
            for j in 0..len { out[o + j] = if self.h.jobs > 0 { g[j] as f32 / self.h.jobs as f32 } else { 0.0 }; }
            o += len;
        }
        Ok(())
    }

    /// Joined units from the exploration bookkeeping into `joined` (open / restart).
    pub fn load_joined(&self, joined: &mut ReadMask) -> std::io::Result<usize> {
        joined.clear();
        let mut e = [0u32; 2 * CU];
        let (mut u0, mut n) = (0, 0);
        while u0 < self.units {
            let len = CU.min(self.units - u0);
            rd_u32s(&self.f, self.o_ex() + 8 * u0 as u64, &mut e[..2 * len])?;
            for j in 0..len { if e[2 * j + 1] > 0 { joined.set_unit(u0 + j, true); n += 1; } }
            u0 += len;
        }
        Ok(n)
    }

    /// One exploration round (see the module doc). `mask` = the installed mask (kept = profile + joined); `joined` is
    /// updated (found active -> joined; expired after JOIN_TTL_ROUNDS). `reader` = the engine HOOK (None: scheduled,
    /// not read). Stack only.
    pub fn explore(&mut self, mask: &ReadMask, joined: &mut ReadMask, frac: f64, reader: Option<&dyn Explorer>, t: f64) -> std::io::Result<Explore> {
        let units = self.units.min(mask.units());
        let masked = (0..units).filter(|&u| !mask.unit(u)).count();
        let round = self.h.rounds + 1;
        let mut x = Explore { round, ts: t, masked, ..Default::default() };
        let n = if masked == 0 { 0 } else { ((frac * masked as f64).ceil() as usize).clamp(1, EXPLORE_MAX).min(masked) };
        x.cooldown = if n == 0 { 0 } else { masked.div_ceil(n) as u64 };
        let (mut pu, mut pp) = ([0u32; EXPLORE_MAX], [0f32; EXPLORE_MAX]);
        let mut k = 0usize;
        let mut e = [0u32; 2 * CU];
        let mut g = [0f32; CU];
        let mut u0 = 0;
        while u0 < units {
            let len = CU.min(units - u0);
            let o = self.o_ex() + 8 * u0 as u64;
            rd_u32s(&self.f, o, &mut e[..2 * len])?;
            self.probs(u0, &mut g[..len])?;
            let mut dirty = false;
            for j in 0..len {
                let u = u0 + j;
                let (last, jr) = ((e[2 * j] & 0x7fff_ffff) as u64, e[2 * j + 1] as u64);
                if jr > 0 && round - jr >= JOIN_TTL_ROUNDS as u64 { e[2 * j + 1] = 0; joined.set_unit(u, false); x.expired += 1; dirty = true; }
                if n == 0 || mask.unit(u) || (last > 0 && round - last < x.cooldown) { continue; }
                // top-n by P(u) desc, ties: lower unit first (scan order)
                let p = g[j];
                let mut i = k;
                while i > 0 && pp[i - 1] < p { i -= 1; }
                if i >= n { continue; }
                let end = (k + 1).min(n);
                for q in (i + 1..end).rev() { pu[q] = pu[q - 1]; pp[q] = pp[q - 1]; }
                (pu[i], pp[i]) = (u as u32, p);
                k = end;
            }
            if dirty { wr_u32s(&self.f, o, &mut e[..2 * len])?; }
            u0 += len;
        }
        x.n = k;
        let mut act = [false; EXPLORE_MAX];
        x.read = k > 0 && reader.is_some_and(|r| r.read(mask, &pu[..k], &mut act[..k]));
        for i in 0..k {
            let u = pu[i] as usize;
            if i < x.top.len() { x.top[i] = (pu[i], pp[i]); }
            let mut w = [0u32; 2];
            let o = self.o_ex() + 8 * u as u64;
            rd_u32s(&self.f, o, &mut w)?;
            w[0] = round as u32 | if x.read && act[i] { 1 << 31 } else { 0 };
            if x.read {
                if act[i] { w[1] = round as u32; joined.set_unit(u, true); x.found += 1; } else { w[1] = 0; joined.set_unit(u, false); }
            }
            wr_u32s(&self.f, o, &mut w)?;
        }
        x.joined = joined.count().0 + joined.count().1;
        let h = &mut self.h;
        h.rounds = round; h.sched += k as u64; h.expired += x.expired as u64;
        if x.read { h.read += k as u64; h.found += x.found as u64; }
        self.write_hdr()?;
        Ok(x)
    }

    pub fn json(&self) -> Value {
        let h = &self.h;
        json!({"file": self.path.display().to_string(), "disk_bytes": self.disk_bytes(), "cap_bytes": PREDICT_CAP, "units": self.units, "buckets": self.nb,
            "requests_learned": h.jobs, "scored": h.scored, "tokens": h.toks,
            "precision": ratio(h.tp, h.tp + h.fp), "recall": ratio(h.tp, h.tp + h.fneg),
            "masked_precision": ratio(h.o_tp, h.o_tp + h.o_fp), "masked_recall": ratio(h.o_tp, h.o_act),
            "out_of_profile_active_per_request": if h.scored > 0 { json!(r3(h.o_act as f64 / h.scored as f64)) } else { Value::Null },
            "bytes_predicted_per_request": if h.scored > 0 { json!((h.bytes_pred as f64 / h.scored as f64).round()) } else { Value::Null },
            "bytes_missed_per_request": if h.scored > 0 { json!((h.bytes_miss as f64 / h.scored as f64).round()) } else { Value::Null },
            "rule": format!("u predicted when max(P(u), max over the request's {} rarest known token buckets (n >= {}) of (c + {} P(u)) / (n + {})) >= {}; truth = the engine's prompt_active when recorded, else the job's own keep set at keep_frac", KB, N_MIN, ALPHA, ALPHA, THETA)})
    }
}

// ---- exploration -----------------------------------------------------------------------------------------------------

pub const EXPLORE_MAX: usize = 256;
pub const EXPLORE_EVERY_S: f64 = 60.0;
/// a unit exploration found active stays joined (kept) this many rounds (~4 h at one round a minute)
pub const JOIN_TTL_ROUNDS: u32 = 240;

/// HOOK (engine side not built yet): read the given masked units once with the full model in idle GPU
/// time (all weights are in RAM: only that pass's speed is spent) and report which were active (the job keep rule).
/// Returns false when it could not run (busy / not available). `ReadMask::locate` maps a unit to (head, layer, index).
pub trait Explorer: Send + Sync { fn read(&self, mask: &ReadMask, units: &[u32], active: &mut [bool]) -> bool; }

/// One exploration round.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Explore {
    pub round: u64,
    pub ts: f64,
    pub masked: usize,
    /// scheduled this round, rotation period (rounds)
    pub n: usize,
    pub cooldown: u64,
    /// the engine read them (hook present and mode on)
    pub read: bool,
    pub found: usize,
    pub joined: usize,
    pub expired: usize,
    /// the first scheduled units (unit, P(u)), most likely active first
    pub top: [(u32, f32); 8],
}
impl Explore {
    pub fn json(&self, m: &ReadMask) -> Value {
        let top: Vec<Value> = self.top[..self.n.min(8)].iter().map(|&(u, p)| { let (h, l, i) = m.locate(u as usize); json!({"unit": u, "layer": l, "kind": if h { "head" } else { "ffn_block" }, "index": i, "p": r3(p as f64)}) }).collect();
        json!({"round": self.round, "ts": self.ts, "masked": self.masked, "scheduled": self.n, "rotation_rounds": self.cooldown, "read": self.read,
               "found_active": self.found, "joined": self.joined, "expired": self.expired, "first": top})
    }
}

// ---- the controller (one per watcher) ----------------------------------------------------------------------------------

pub struct MaskCtl {
    pub mode: Mode,
    pub win: Window,
    pub keep: f64,
    pub explore_frac: f64,
    /// the installed mask (profile | joined), the joined units, scratch sets (job activity, prediction)
    pub mask: Box<ReadMask>,
    pub joined: Box<ReadMask>,
    act: Box<ReadMask>,
    pred: Box<ReadMask>,
    pub cost: Cost,
    pub built: bool,
    pub rep: Report,
    pub pf: Option<Predictor>,
    pub err: Option<String>,
    pub last: Option<Score>,
    agg: Score,
    agg_n: u64,
    pub last_explore: Option<Explore>,
    last_explore_t: f64,
    /// HOOK: the engine's exploration reader (None until the engine API exists)
    /// the last score's truth came from the engine's prompt_active (else the job keep set)
    pub truth_engine: bool,
    pub explorer: Option<Arc<dyn Explorer>>,
}

impl MaskCtl {
    /// None when mask = off (nothing built, nothing written). Setup: four mask boxes + the predictor file.
    pub fn open(st: &Store, mode: &str, window: &str, keep: f64, explore_frac: f64) -> Option<Box<MaskCtl>> {
        let mode = Mode::parse(mode);
        if mode == Mode::Off { return None; }
        let mask = ReadMask::boxed(st);
        let units = mask.units();
        let (pf, err) = match Predictor::open(&st.dir, units, PREDICT_CAP) { Ok(p) => (Some(p), None), Err(e) => (None, Some(format!("predictor: {}", e))) };
        let mut joined = ReadMask::boxed(st);
        if let Some(p) = &pf { let _ = p.load_joined(&mut joined); }
        Some(Box::new(MaskCtl { mode, win: Window::parse(window), keep, explore_frac, mask, joined, act: ReadMask::boxed(st), pred: ReadMask::boxed(st), cost: Cost::of(&st.d),
            built: false, rep: Report::default(), pf, err, last: None, agg: Score::default(), agg_n: 0, last_explore: None, last_explore_t: f64::NEG_INFINITY, explorer: None, truth_engine: false }))
    }
    /// The mask is applied by the engine. HOOK (engine API): false until the engine takes a `ReadMask`; with
    /// mode "on" it becomes true once that hook is wired.
    pub fn applied(&self) -> bool { false }
    /// Install a freshly built profile: mask = profile | joined, report. No heap.
    pub fn install(&mut self, profile: &ReadMask, tok: (f64, f64, u64), build_ms: f64, t: f64) {
        (*self.mask).clone_from(profile);
        self.installed(tok, build_ms, t);
    }
    /// `mask` holds the new profile: add the joined units, report.
    fn installed(&mut self, tok: (f64, f64, u64), build_ms: f64, t: f64) {
        let (jf, jh) = self.joined.count();
        self.mask.or_from(&self.joined);
        report(&self.mask, &self.cost, &mut self.rep);
        (self.rep.ts, self.rep.build_ms, self.rep.window, self.rep.keep, self.rep.joined) = (t, build_ms, self.win.name(), self.keep, jf + jh);
        (self.rep.tokens_now, self.rep.tokens_session, self.rep.tokens_all) = tok;
        self.rep.ready = tok.2 >= PLAN_MIN_TOKENS;
        self.built = true;
    }
    /// Build from `st` (scratch = the job box) and install (tests, CLI; the gateway builds on sd-heat's read-only store).
    #[cfg(test)]
    pub fn rebuild(&mut self, st: &mut Store, t: f64) {
        let t0 = std::time::Instant::now();
        let tok = build_profile(st, self.win, self.keep, t, &mut self.mask);
        self.installed(tok, t0.elapsed().as_secs_f64() * 1e3, t);
    }
    /// Score + learn one finished job (recorder in place). Some(score) every request; `note` = Some(aggregate) every
    /// NOTE_EVERY requests. No heap.
    pub fn score_job(&mut self, src: &dyn JobSrc, toks: &dyn Tokens, prompt: Option<&streamdense_model::readmask::ReadMask>) -> (Option<Score>, Option<(Score, u64)>) {
        let Some(pf) = self.pf.as_mut() else { return (None, None) };
        // truth: the engine's prompt-active set when it recorded one (read_mask region on), else the job's own keep set
        let from_engine = prompt.is_some_and(|p| self.act.load_engine(p));
        self.truth_engine = from_engine;
        if !from_engine { active_units(src, self.keep, &mut self.act); }
        let mask = if self.built { Some(&*self.mask) } else { None };
        match pf.job(toks, &self.act, mask, &self.cost, &mut self.pred) {
            Ok(s) => {
                self.last = Some(s);
                self.agg.add(&s);
                self.agg_n += 1;
                let note = (self.agg_n >= NOTE_EVERY).then(|| { let a = (self.agg, self.agg_n); (self.agg, self.agg_n) = (Score::default(), 0); a });
                (Some(s), note)
            }
            Err(e) => { self.err = Some(format!("predictor: {}", e)); (None, None) }
        }
    }
    /// An exploration round is due (mode dry / on, a mask built, EXPLORE_EVERY_S since the last).
    pub fn explore_due(&self, t: f64) -> bool { self.mode != Mode::Off && self.built && self.pf.is_some() && t - self.last_explore_t >= EXPLORE_EVERY_S }
    /// One round (dry: schedule only, never the GPU; on: the engine hook when present). Joined changes re-install the
    /// mask. No heap.
    pub fn explore(&mut self, t: f64) -> Option<Explore> {
        if self.mode == Mode::Off || !self.built { return None; }
        self.last_explore_t = t;
        let reader = if self.mode == Mode::On && self.applied() { self.explorer.as_deref() } else { None };
        let pf = self.pf.as_mut()?;
        match pf.explore(&self.mask, &mut self.joined, self.explore_frac, reader, t) {
            Ok(x) => {
                // found active -> kept now (expired ones leave at the next build)
                if x.found > 0 { self.mask.or_from(&self.joined); report(&self.mask, &self.cost, &mut self.rep); }
                self.rep.joined = x.joined;
                self.last_explore = Some(x);
                Some(x)
            }
            Err(e) => { self.err = Some(format!("explore: {}", e)); None }
        }
    }
    pub fn json(&self) -> Value {
        let note = match self.mode {
            Mode::Off => "off",
            Mode::Dry => "dry: profile + predictor score + exploration schedule; nothing masked",
            Mode::On => "on requested: the engine mask hook is not wired yet, acting as dry",
        };
        let ex = self.pf.as_ref().map(|p| json!({"rounds": p.h.rounds, "scheduled": p.h.sched, "read": p.h.read, "found_active": p.h.found, "expired": p.h.expired,
            "joined_now": self.joined.count().0 + self.joined.count().1, "every_s": EXPLORE_EVERY_S, "share": self.explore_frac, "max_per_round": EXPLORE_MAX,
            "join_ttl_rounds": JOIN_TTL_ROUNDS, "gpu": if self.mode == Mode::On && self.applied() && self.explorer.is_some() { "engine hook" } else { "not called (dry run or no engine hook): schedule + bookkeeping only" },
            "last": self.last_explore.map(|x| x.json(&self.mask))}));
        json!({"mode": self.mode.name(), "applied": self.applied(), "effective": if self.applied() { "on" } else { "dry" }, "note": note,
            "window": self.win.name(), "keep_frac": self.keep, "explore_frac": self.explore_frac,
            "profile": if self.built { self.rep.json() } else { json!({"built": false, "state": "waiting for the first build (sd-heat, idle after a summary)"}) },
            "predictor": self.pf.as_ref().map(|p| { let mut v = p.json(); v["last"] = self.last.map_or(Value::Null, |s| s.json(1)); v }),
            "truth": if self.truth_engine { "engine prompt_active (this sequence's prefills)" } else { "job keep set at keep_frac (recorder window, prefill + decode)" },
            "explore": ex, "error": self.err})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::JobSrc;
    use crate::watcher::{label_names, MIXED};
    use streamdense_model::activity::ActivityStats;

    fn tmpdir(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_predict_{}_{}", n, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn dims(layers: usize, inter: usize) -> Dims {
        let kinds = (0..layers).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let t = ActivityStats { enabled: true, layers, hidden: 128, inter, kinds, dn_kh: 1, dn_dk: 128, dn_vh: 2, dn_dv: 128, heads: 3, kv_heads: 1,
                                head_dim: 128, conv_k: 4, group: 64, bits: 4, ..Default::default() };
        let mut d = Dims::from_stats(&t, 64);
        d.vocab = 300;
        d
    }
    struct V(Vec<f32>, u64);
    impl JobSrc for V {
        fn rows(&self) -> (u64, u64) { (self.1, 0) }
        fn read(&self, off: usize, out: &mut [f32]) { for (j, o) in out.iter_mut().enumerate() { *o = self.0.get(off + j).copied().unwrap_or(0.0); } }
    }
    fn rng(x: &mut u64) -> u64 { *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17; *x }

    /// Brute force (heap, independent formulation): per part, the blocks / heads holding `keep` of the mass = the
    /// fewest largest (sort descending, take until the running sum reaches keep x total); union over the parts.
    fn brute(st: &mut Store, win: Window, keep: f64, t: f64) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        st.t = t;
        let d = st.d.clone();
        let nb = d.inter.div_ceil(64);
        let mut f = vec![vec![false; nb]; d.layers];
        let mut h: Vec<Vec<bool>> = (0..d.layers).map(|l| vec![false; st.heads_of(l).len()]).collect();
        let mut sels = [Sel::Labels; 3];
        let ns = win.sels(&mut sels);
        let mut rec: Rec = [0u8; CH * 4 + 16];
        let pick = |m: &[f64], out: &mut [bool]| {
            let tot: f64 = m.iter().sum();
            if !(tot > 0.0) { return false; }
            let mut idx: Vec<usize> = (0..m.len()).collect();
            idx.sort_by(|&a, &b| m[b].partial_cmp(&m[a]).unwrap());
            let mut acc = 0.0;
            for i in idx { if acc >= keep * tot * (1.0 - 1e-12) { break; } acc += m[i]; out[i] = true; }
            true
        };
        for l in 0..d.layers {
            let (mut af, mut ah) = (false, false);
            for &sel in &sels[..ns] {
                let mut v = vec![0f32; d.inter];
                st.read_sel(sel, l * d.inter, &mut v, &mut rec);
                let mut m = vec![0f64; nb];
                for (j, x) in v.iter().enumerate() { m[j / 64] += *x as f64; }
                af |= pick(&m, &mut f[l]);
                let r = st.heads_of(l);
                let mut hv = vec![0f32; r.len()];
                st.read_sel(sel, r.start, &mut hv, &mut rec);
                ah |= pick(&hv.iter().map(|x| *x as f64).collect::<Vec<_>>(), &mut h[l]);
            }
            if !af { f[l].fill(true); }
            if !ah { h[l].fill(true); }
        }
        (f, h)
    }

    /// The profile from a fixed synthetic store == the brute-force keep set, for every window and a few keep levels;
    /// and the dry report counts / bytes match the set.
    #[test]
    fn profile_equals_brute_force() {
        let d = dims(4, 1000); // 16 blocks per layer, the last one partial (40 neurons)
        let dir = tmpdir("profile");
        let mut st = Store::open(&dir, d.clone(), label_names(), MIXED, false).unwrap();
        let n = st.n();
        let mut x = 0x1234_5678u64;
        for (i, t) in [0.0, 100.0, 400.0, 2000.0, 2100.0].iter().enumerate() {
            // skewed activity: a few hot blocks per job (different per job), the rest small / zero
            let hot = (rng(&mut x) % 16) as usize;
            let v: Vec<f32> = (0..n).map(|g| { let r = rng(&mut x); if g < st.n_ffn() && (g % 1000) / 64 == hot { 50.0 + (r % 100) as f32 } else if r % 3 == 0 { 0.0 } else { (r % 1000) as f32 / 997.0 } }).collect();
            st.fold_job(i % 3, &V(v, 100), i as u64, *t).unwrap();
        }
        let mut m = ReadMask::boxed(&st);
        for win in ["now", "session", "all", "now+all", "session+all", "now+session+all"].map(Window::parse) {
            for keep in [0.995, 0.95, 0.8, 0.5] {
                let tok = build_profile(&mut st, win, keep, 2200.0, &mut m);
                let (bf, bh) = brute(&mut st, win, keep, 2200.0);
                for l in 0..d.layers {
                    for b in 0..m.blocks { assert_eq!(m.ffn(l, b), bf[l][b], "{} keep {} layer {} block {}", win.name(), keep, l, b); }
                    for h in 0..m.nh[l] as usize { assert_eq!(m.head(l, h), bh[l][h], "{} keep {} layer {} head {}", win.name(), keep, l, h); }
                }
                assert_eq!(tok.2, 500);
                let mut r = Report::default();
                let c = Cost::of(&d);
                report(&m, &c, &mut r);
                let mf: usize = bf.iter().flatten().filter(|x| !**x).count();
                let mh: usize = bh.iter().flatten().filter(|x| !**x).count();
                assert_eq!((r.masked_blocks, r.masked_heads), (mf, mh));
                let want_ffn: u64 = (0..d.layers).map(|l| (0..m.blocks).filter(|&b| !bf[l][b]).map(|b| if b == 15 { c.last_block } else { c.block }).sum::<u64>()).sum();
                assert_eq!(r.ffn_skip, want_ffn);
            }
        }
        // a lower keep masks more (monotone), and keep 1.0 masks only never-active blocks
        build_profile(&mut st, Window::parse("all"), 0.8, 2200.0, &mut m);
        let lo = m.count();
        build_profile(&mut st, Window::parse("all"), 0.995, 2200.0, &mut m);
        assert!(m.count().0 >= lo.0);
        // engine format round trip
        let mut hb = [0u32; MAXU];
        let k = m.hot_blocks(0, &mut hb);
        assert_eq!(k, (0..m.blocks).filter(|&b| m.ffn(0, b)).count());
        assert!(hb[..k].windows(2).all(|w| w[0] < w[1]));
        // the engine's ReadMask (bit set = read): same words; the partial last block (1000 = 15 x 64 + 40) stays read there
        let mut e = streamdense_model::readmask::ReadMask { ffn_blocks: vec![vec![0; 1]; 4], heads: (0..4).map(|l| vec![0; (m.nh[l] as usize).div_ceil(64)]).collect() };
        assert!(m.to_engine(&mut e));
        for l in 0..4 {
            for b in 0..15 { assert_eq!(e.ffn_blocks[l][0] >> b & 1 == 1, m.ffn(l, b)); }
            assert_eq!(e.ffn_blocks[l][0] >> 15, 0);
            for h in 0..m.nh[l] as usize { assert_eq!(e.heads[l][0] >> h & 1 == 1, m.head(l, h)); }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A planted token -> block pattern: precision / recall > 0.9 on held-out requests (overall and on the masked
    /// units), and the table stays at its fixed size whatever the traffic.
    #[test]
    fn predictor_learns_planted_pattern() {
        let d = dims(4, 1024); // 16 blocks per layer
        let dir = tmpdir("pred");
        let st = Store::open(&dir, d.clone(), label_names(), MIXED, false).unwrap();
        let shape = ReadMask::boxed(&st);
        let units = shape.units();
        let cap = 1_000_000;
        let mut p = Predictor::open(&st.dir, units, cap).unwrap();
        let size = Predictor::size(units, p.nb);
        assert!(size <= cap && p.disk_bytes() == size && p.nb == NBMAX.min(Predictor::nb_for(units, cap)), "{} {} {}", size, p.disk_bytes(), p.nb);
        // 16 planted tokens (1000 + k): token k makes block 8 + k % 8 of layer k / 4 % 4 active; blocks 0..8 + every
        // head always active; filler tokens 0..500
        let mut x = 99u64;
        let n = st.n();
        let mut mask = ReadMask::boxed(&st);
        mask.fill();
        for l in 0..4 { for b in 8..16 { mask.set_ffn(l, b, false); } } // the profile would mask the rare blocks
        let cost = Cost::of(&d);
        let (mut act, mut pred) = (ReadMask::boxed(&st), ReadMask::boxed(&st));
        let mut test = Score::default();
        for r in 0..600 {
            let mut toks = vec![];
            let mut v = vec![0f32; n];
            for g in 0..st.n_ffn() { v[g] = if (g % 1024) / 64 < 8 { 10.0 } else { 0.0001 }; }
            for l in 0..4 { for h in st.heads_of(l) { v[h] = 1.0; } }
            for _ in 0..1 + rng(&mut x) % 3 {
                let k = (rng(&mut x) % 16) as usize;
                toks.push(1000 + k as u32);
                let (l, b) = ((k / 4) % 4, 8 + k % 8);
                for j in 0..64 { v[l * 1024 + b * 64 + j] = 10.0; }
            }
            for _ in 0..20 { toks.push((rng(&mut x) % 500) as u32); }
            active_units(&V(v, 50), 0.995, &mut act);
            let s = p.job(&TokList(&toks), &act, Some(&mask), &cost, &mut pred).unwrap();
            if r >= 500 { test.add(&s); }
            assert_eq!(p.disk_bytes(), size);
        }
        let (pr, rc) = (test.precision(), test.recall());
        let (opr, orc) = (test.o_tp as f64 / (test.o_tp + test.o_fp).max(1) as f64, test.o_tp as f64 / test.o_act.max(1) as f64);
        eprintln!("planted: precision {:.3} recall {:.3}; masked precision {:.3} recall {:.3}; {} B predicted / request; table {} B ({} buckets x {} units)",
                  pr, rc, opr, orc, test.bytes_pred / 100, size, p.nb, units);
        assert!(pr > 0.9 && rc > 0.9 && opr > 0.9 && orc > 0.9, "{:?}", test);
        assert_eq!(p.h.jobs, 600);
        // reopen keeps the counts
        drop(p);
        let p = Predictor::open(&st.dir, units, cap).unwrap();
        assert_eq!((p.h.jobs, p.h.scored), (600, 600));
        // 27B shape: the proposed 16 MB cap
        let u27 = 64 * 272 + 48 * 48 + 16 * 24;
        let nb27 = Predictor::nb_for(u27, PREDICT_CAP);
        eprintln!("27B: {} units, {} buckets, {} B", u27, nb27, Predictor::size(u27, nb27));
        assert!(Predictor::size(u27, nb27) <= PREDICT_CAP && nb27 > 700);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Exploration rotates through every masked unit (most likely first) and units the read finds active join the
    /// mask; dry (no reader) only schedules.
    #[test]
    fn exploration_rotates_and_joins() {
        struct R(Vec<u32>);
        impl Explorer for R { fn read(&self, _m: &ReadMask, units: &[u32], active: &mut [bool]) -> bool { for (i, u) in units.iter().enumerate() { active[i] = self.0.contains(u); } true } }
        let d = dims(4, 1024);
        let dir = tmpdir("explore");
        let st = Store::open(&dir, d.clone(), label_names(), MIXED, false).unwrap();
        let mut mask = ReadMask::boxed(&st);
        mask.fill();
        for l in 0..4 { for b in 10..16 { mask.set_ffn(l, b, false); } } // 24 masked blocks
        let units = mask.units();
        let mut p = Predictor::open(&st.dir, units, 1_000_000).unwrap();
        // P(u): one learned request where block (0, 15) was active -> ranked first
        let mut act = ReadMask::boxed(&st);
        act.set_ffn(0, 15, true);
        let (mut pred, cost) = (ReadMask::boxed(&st), Cost::of(&d));
        p.job(&TokList(&[1, 2]), &act, Some(&mask), &cost, &mut pred).unwrap();
        let mut joined = ReadMask::boxed(&st);
        let mut seen = vec![0u32; units];
        // dry: no reader: 24 masked x 0.25 = 6 per round, rotation 4 rounds
        for r in 0..8 {
            let x = p.explore(&mask, &mut joined, 0.25, None, r as f64).unwrap();
            assert_eq!((x.n, x.cooldown, x.read, x.found), (6, 4, false, 0));
            if r == 0 { assert_eq!(x.top[0].0 as usize, 15, "the most likely active masked block first"); }
            for &(u, _) in &x.top[..6] { seen[u as usize] += 1; }
        }
        // every masked unit exactly twice in 2 rotations, kept units never
        for u in 0..units { assert_eq!(seen[u], if mask.unit(u) { 0 } else { 2 }, "unit {}", u); }
        assert_eq!(joined.count(), (0, 0));
        // with the engine reader: units 14 and 3 x 16 + 12 are active -> joined; the mask keeps them
        let rd = R(vec![14, 3 * 16 + 12]);
        let mut found = 0;
        for r in 0..4 {
            let x = p.explore(&mask, &mut joined, 0.25, Some(&rd), 10.0 + r as f64).unwrap();
            assert!(x.read);
            found += x.found;
        }
        assert_eq!(found, 2);
        assert!(joined.ffn(0, 14) && joined.ffn(3, 12) && joined.count() == (2, 0));
        mask.or_from(&joined);
        assert!(mask.ffn(0, 14));
        // restart: the joined set comes back from disk; it expires after JOIN_TTL_ROUNDS
        let mut j2 = ReadMask::boxed(&st);
        assert_eq!(p.load_joined(&mut j2).unwrap(), 2);
        p.h.rounds += JOIN_TTL_ROUNDS as u64;
        let x = p.explore(&mask, &mut j2, 0.25, None, 99.0).unwrap();
        assert_eq!((x.expired, j2.count()), (2, (0, 0)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Steady state: profile build + report + job activity + predict / score / learn + exploration allocate nothing.
    #[test]
    fn zero_heap_steady_state() {
        let d = dims(8, 1024);
        let dir = tmpdir("zero");
        let mut st = Store::open(&dir, d.clone(), label_names(), MIXED, false).unwrap();
        let n = st.n();
        let job = V((0..n).map(|i| (i % 13) as f32).collect(), 5);
        st.fold_job(0, &job, 1, 0.0).unwrap();
        let mut mk = MaskCtl::open(&st, "dry", "session+all", 0.95, 0.1).unwrap();
        let toks = [5u32, 9, 77, 5];
        // first calls = setup (files exist, boxes allocated)
        mk.rebuild(&mut st, 1.0);
        mk.score_job(&job, &TokList(&toks), None);
        let a0 = streamdense_core::alloc::thread_allocs();
        mk.rebuild(&mut st, 2.0);
        for _ in 0..3 { mk.score_job(&job, &TokList(&toks), None); }
        let x = mk.explore(100.0);
        let a1 = streamdense_core::alloc::thread_allocs();
        assert_eq!(a1 - a0, 0, "heap allocations in profile + predictor + exploration");
        assert!(mk.built && x.is_some() && mk.last.is_some_and(|s| s.units == mk.mask.units()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
