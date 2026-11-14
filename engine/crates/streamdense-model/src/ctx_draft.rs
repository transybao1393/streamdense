//! Context drafting (queue item 0b "D", 2026-11-03): lossless speculative decode WITHOUT the MTP head, for the 12 GB
//! mode (compact FFN, MTP off). Drafts come from text already in the context (prompt, files, tool results, earlier
//! turns, the reply so far): an n-gram lookup over every token in the KV (`CtxDrafter`, CPU), plus the learning draft
//! table (draft_table.rs) when the engine has one open. Each round verifies [pending, D1..Dd] in ONE exact pass of
//! K = d + 1 rows; every emitted token is the pass's argmax, so the output is plain greedy's whatever the drafts.
//!
//! Round at position p (pending token t not yet fed, ctrl.pos == p):
//!   drafts   lookup: the most recent earlier occurrence of the longest suffix n-gram (n = 4..1) of context + t, its
//!            continuation (self-overlapping, LZ77 style) up to d tokens; table: `DraftSource::propose`. The source and
//!            d come from argmax E(d) / T(d): E = 1 + sum of prefix acceptance products (per-draft conditional rates,
//!            EWMA-calibrated per match-length bin and depth / per table confidence bin), T = the cost model's pass ms
//!            for K = d + 1 (`QwenOptions.spec_cost`) + the replay. Deterministic (no timings), only speed depends on it.
//!   verify   d = 0: the plain K = 1 decode pass (state updated in place). d >= 1: a HOLD pass of K rows: the DeltaNet
//!            state is only read (dn_conv ck = DN_HOLD, dn_recur_hold) and the raw in_proj columns [qkv | b | a] of all
//!            K rows are saved (dn_save_all, `QwenOptions.ctx_draft + 1` rows: 0.99 MB per row on the 27B).
//!   decide   j = longest prefix with D_{i+1} == a_i (a_i not eos); emit a_0..a_j (stop / cancel / length may cut at
//!            je <= j); rows 0..je are fed: ctrl.pos = p + je + 1, rec[p + je + 1 .. p + K) zeroed.
//!   replay   (d >= 1) dn_conv / dn_prep / dn_recur_ck0 over the saved rows 0..je, in place (spec_deep.rs replay, same
//!            kernels and per-token code: bit-identical to feeding those rows one by one by K-invariance).
//! Attention needs no rollback (rows past the accepted ones left K/V at positions >= the new pos; every later pass
//! writes its rows before reading them). No second DeltaNet state copy (151 MB on the 27B): the price is the replay of
//! the accepted rows every drafted round (one more state read + write) instead of only on rejections.
//! Overlap: while a verify runs, the CPU encodes the replay buffer of every outcome je and the next round's verify for
//! K = 1 and the K just used over the position range [p + 1, p + K] (`path_pos`); the next verify waits on the GPU
//! for the replay's event, so the CPU commits replay + verify back to back.
//! The lookup index is CPU only: tokens + an n-gram -> last-end-position map (`lookup_bytes`, ~100 B per context token).
//! Third draft source (2026-11-03 connector): the history index (streamdense-core history_index.rs; text the user
//! already has: project files, earlier replies), `CtxDrafter::hindex`, queried with the context tail + pending token;
//! its per-token confidence enters the same E/T choice (blended with an EWMA per confidence bin, like the table), and
//! the verify result goes back as `feedback(order, accepted)`. Picked only when its E/T beats the other two.
use std::collections::HashMap;
use std::sync::Arc;

use streamdense_core::history_index::{Drafts, HistoryIndex};
use std::time::Instant;

use streamdense_metal::{cb_signal, cb_wait, commit, gpu_ms, wait, CmdBuf, Enc, Event};

use crate::mtp::path_pos;
use crate::qwen_config::QwenConfig;
use crate::qwen_engine::{KvFormat, QwenEngine, StreamStats};
use crate::qwen_layers::{encode_head, encode_layer, PassIo};
use crate::spec_deep::{replay_row_bytes, SpecCost, SPEC_DEPTH_MAX};

/// Longest n-gram the lookup keys on (then the match is extended backwards up to `MAXM` to bin it).
const NG: usize = 4;
const MAXM: usize = 32;
/// match-length bins: 1, 2, 3-4, 5-8, 9-16, 17+
const NB: usize = 6;
/// prior conditional acceptance of the first draft per bin, and of later drafts (EWMA-updated per bin and depth)
const PRIOR0: [f64; NB] = [0.10, 0.25, 0.40, 0.55, 0.70, 0.80];
const PRIOR_D: f64 = 0.75;
const NBT: usize = 5;
const EW: f64 = 0.1;
/// table drafts: tokens of context tail handed to `DraftSource::propose` (as spec_deep's TAIL)
const TAIL: usize = 16;
/// replay of the accepted rows per drafted round (GPU ms estimate, 27B: one DeltaNet state read + write, ~0.3 GB at
/// ~0.27 TB/s, + 3 dispatches per DeltaNet layer). Only steers d; never affects the output.
const REPLAY_MS: f64 = 1.5;
/// A context-draft run stops drafting below this prefix probability.
const MIN_PREFIX: f64 = 0.03;

/// Bytes of the hold save for `d` drafts per pass (d + 1 rows), 0 when off.
pub fn hold_bytes(cfg: &QwenConfig, d: usize) -> usize { if d >= 1 { (d + 1) * replay_row_bytes(cfg) } else { 0 } }

/// CPU bytes of the lookup index for a context of `max_ctx` tokens (tokens + up to NG map entries per token at
/// ~24 B each incl. hash map overhead). For the RAM plan (the 12 GB keep formula counts it).
pub fn lookup_bytes(max_ctx: usize) -> usize { max_ctx * (4 + NG * 24) }

fn mbin(m: usize) -> usize { match m { 0..=1 => 0, 2 => 1, 3..=4 => 2, 5..=8 => 3, 9..=16 => 4, _ => 5 } }
fn tbin(c: f64) -> usize { ((c.clamp(0.0, 1.0) * NBT as f64) as usize).min(NBT - 1) }
fn mix(h: u64, t: u32) -> u64 { (h ^ t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) }
fn key(n: usize, toks: &[u32]) -> u64 { toks.iter().fold(0x243F_6A88_85A3_08D3u64 ^ n as u64, |h, &t| mix(h, t)) }

/// Statistics of one `feed_stream_ctx` (or summed over several: `add`).
#[derive(Clone, Debug, Default)]
pub struct CtxStats {
    /// verify passes (rounds), passes with >= 1 draft
    pub passes: usize,
    pub drafted_passes: usize,
    /// drafts verified / accepted / passes with a rejected draft
    pub drafted: usize,
    pub accepted: usize,
    pub rejects: usize,
    /// tokens emitted by the rounds (not the prefill's first token)
    pub tokens: usize,
    pub lookup_drafts: usize,
    pub lookup_acc: usize,
    pub table_drafts: usize,
    pub table_acc: usize,
    pub history_drafts: usize,
    pub history_acc: usize,
    /// accept_hist[j] = passes with j drafts accepted; k_hist[k - 1] = passes of k rows
    pub accept_hist: [usize; SPEC_DEPTH_MAX + 1],
    pub k_hist: [usize; SPEC_DEPTH_MAX + 1],
    /// GPU ms: verify passes, replays; passes that used a buffer encoded ahead
    pub verify_ms: f64,
    pub replay_ms: f64,
    pub preencoded: usize,
    /// CPU ms spent choosing drafts (lookup + table)
    pub draft_cpu_ms: f64,
}
impl CtxStats {
    pub fn add(&mut self, o: &CtxStats) {
        self.passes += o.passes; self.drafted_passes += o.drafted_passes; self.drafted += o.drafted; self.accepted += o.accepted;
        self.rejects += o.rejects; self.tokens += o.tokens; self.lookup_drafts += o.lookup_drafts; self.lookup_acc += o.lookup_acc;
        self.table_drafts += o.table_drafts; self.table_acc += o.table_acc;
        self.history_drafts += o.history_drafts; self.history_acc += o.history_acc;
        for i in 0..=SPEC_DEPTH_MAX { self.accept_hist[i] += o.accept_hist[i]; self.k_hist[i] += o.k_hist[i]; }
        self.verify_ms += o.verify_ms; self.replay_ms += o.replay_ms; self.preencoded += o.preencoded; self.draft_cpu_ms += o.draft_cpu_ms;
    }
    /// accepted / drafted (0 when nothing was drafted)
    pub fn acceptance(&self) -> f64 { if self.drafted > 0 { self.accepted as f64 / self.drafted as f64 } else { 0.0 } }
    /// emitted tokens per verify pass
    pub fn tokens_per_pass(&self) -> f64 { if self.passes > 0 { self.tokens as f64 / self.passes as f64 } else { 0.0 } }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"passes": self.passes, "drafted_passes": self.drafted_passes, "drafted": self.drafted, "accepted": self.accepted,
            "rejects": self.rejects, "tokens": self.tokens, "acceptance": self.acceptance(), "tokens_per_pass": self.tokens_per_pass(),
            "lookup": [self.lookup_drafts, self.lookup_acc], "table": [self.table_drafts, self.table_acc], "history": [self.history_drafts, self.history_acc],
            "accept_hist": self.accept_hist, "k_hist": self.k_hist, "verify_ms": self.verify_ms, "replay_ms": self.replay_ms,
            "preencoded": self.preencoded, "draft_cpu_ms": self.draft_cpu_ms})
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src { Lookup, Table, History }

/// The context lookup + its acceptance calibration (lives across feeds: the caller keeps `history` == the tokens in
/// the engine's KV; a mismatch only costs drafts, never correctness).
pub struct CtxDrafter {
    hist: Vec<u32>,
    /// n-gram (n = 1..=NG) key -> end (exclusive) of its most recent occurrence in `hist`
    idx: HashMap<u64, u32>,
    cal: [[f64; SPEC_DEPTH_MAX]; NB],
    cal_t: [f64; NBT],
    cal_h: [f64; NBT],
    /// history index (third draft source; None = off). Shared with the gateway's background updater.
    pub hindex: Option<Arc<HistoryIndex>>,
    hq: Drafts,
    /// cumulative stats over every feed (gateway /health)
    pub total: CtxStats,
}
impl Default for CtxDrafter { fn default() -> Self { CtxDrafter::new() } }
impl CtxDrafter {
    pub fn new() -> CtxDrafter {
        let mut cal = [[PRIOR_D; SPEC_DEPTH_MAX]; NB];
        for (b, row) in cal.iter_mut().enumerate() { row[0] = PRIOR0[b]; }
        CtxDrafter { hist: vec![], idx: HashMap::new(), cal, cal_t: [0.1, 0.3, 0.5, 0.7, 0.9], cal_h: [0.1, 0.3, 0.5, 0.7, 0.9], hindex: None,
                     hq: Drafts::default(), total: CtxStats::default() }
    }
    /// New sequence (engine reset): history + index cleared, calibration kept.
    pub fn reset(&mut self) { self.hist.clear(); self.idx.clear(); }
    /// Replace the history (snapshot restore).
    pub fn set_history(&mut self, toks: &[u32]) { self.reset(); self.push(toks); }
    pub fn history(&self) -> &[u32] { &self.hist }
    /// Tokens fed into the KV, in order.
    pub fn push(&mut self, toks: &[u32]) {
        for &t in toks {
            self.hist.push(t);
            let e = self.hist.len();
            for n in 1..=NG.min(e) { self.idx.insert(key(n, &self.hist[e - n..e]), e as u32); }
        }
    }
    /// Bytes of the history + index now (CPU heap).
    pub fn ram_bytes(&self) -> usize { self.hist.capacity() * 4 + self.idx.capacity() * 24 }
    /// Attach (or drop) the history index as a draft source.
    pub fn with_history(mut self, h: Option<Arc<HistoryIndex>>) -> CtxDrafter { self.hindex = h; self }

    /// Longest suffix n-gram (n = NG..1) of history + [pending]: (match length, up to `max` continuation tokens).
    pub fn lookup(&self, pending: u32, max: usize) -> (usize, Vec<u32>) {
        let h = &self.hist;
        let l = h.len() + 1;
        let at = |i: usize| if i < h.len() { h[i] } else { pending };
        let mut tail = [0u32; NG];
        let nt = NG.min(l);
        for (q, x) in tail[NG - nt..].iter_mut().enumerate() { *x = at(l - nt + q); }
        for n in (1..=nt).rev() {
            let Some(&e) = self.idx.get(&key(n, &tail[NG - n..])) else { continue };
            let e = e as usize;
            if e == 0 || e >= l { continue; }
            // extend backwards (also checks the hash): match length m
            let mut m = 0;
            while m < MAXM && m < e && at(e - 1 - m) == at(l - 1 - m) { m += 1; }
            if m < n { continue; } // hash collision
            let mut out: Vec<u32> = Vec::with_capacity(max);
            for i in 0..max {
                let s = e + i;
                out.push(if s < l { at(s) } else { out[s - l] });
            }
            return (m, out);
        }
        (0, vec![])
    }
}

/// argmax over d in 0..=q.len() of E(d) / T(d): (d, ratio).
fn best(q: &[f64], cost: &SpecCost) -> (usize, f64) {
    let t = |d: usize| cost.pass_ms[d.min(7)] + if d > 0 { REPLAY_MS } else { 0.0 };
    let (mut e, mut pr) = (1.0, 1.0);
    let mut b = (0, e / t(0));
    for d in 1..=q.len() {
        pr *= q[d - 1];
        if pr < MIN_PREFIX { break; }
        e += pr;
        let r = e / t(d);
        if r > b.1 + 1e-12 { b = (d, r); }
    }
    b
}

/// Command buffer encoded inside its own autorelease pool (spec_deep `enc_cb`), optionally waiting on the GPU for
/// `ev` >= value before its first encoder.
fn enc_after(eng: &QwenEngine, w: Option<(&Event, u64)>, f: &dyn Fn(&Enc) -> Result<(), String>) -> Result<CmdBuf, String> {
    objc2::rc::autoreleasepool(|_| {
        let cb = eng.gpu.command_buffer();
        if let Some((ev, v)) = w { cb_wait(&cb, ev, v); }
        let e = Enc::new(&eng.gpu, &cb, &eng.buf);
        let r = f(&e);
        e.end();
        r.map(|_| cb)
    })
}

impl QwenEngine {
    /// Verify pass of k rows staged in slot 0, ctrl.pos somewhere in [lo, hi] (exact when lo == hi): k = 1 is the
    /// plain decode pass (state in place), k >= 2 a HOLD pass (state only read, all rows saved for the replay). Head +
    /// argmax on all k rows: rec[ctrl.pos + i]. ctrl.pos is NOT advanced (the CPU sets it after the decision).
    fn encode_ctx_verify(&self, e: &Enc, k: usize, lo: usize, hi: usize) -> Result<(), String> {
        let c = &self.cfg;
        let hold = if k > 1 { Some(self.ctx_hold.ok_or("ctx_draft: no hold save (QwenOptions.ctx_draft = 0)")?.off) } else { None };
        let pos = path_pos(lo, hi, k, self.opt.kv == KvFormat::Q8)?;
        let io = PassIo { k, pos, nch_max: self.nch_for(hi, k), dn_save: None, dn_hold: hold, rm: self.rm_io(true), ..self.io() };
        e.pso("embed_q4").b(0, self.sl().stage.at(0)).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        for l in 0..c.layers {
            encode_layer(e, c, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, false, None, l)?;
        }
        encode_head(e, c, &self.w, &io, k, None)
    }

    /// Streaming greedy decode with context drafts (`feed_stream` contract: feed `ids` from the current position, then
    /// decode, `on_tok` per token, same stop rules; the last emitted token is not fed). Needs `opt.ctx_draft >= 1`.
    /// `ids.len() == 1` (a continuation's pending token, the 12 GB mode's first token): that token is row 0 of the
    /// first verify pass (no separate pass). Output == `feed_stream` token for token; `cx` keeps the lookup history
    /// (the caller resets it with the engine).
    pub fn feed_stream_ctx(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool, cx: &mut CtxDrafter)
                           -> Result<(StreamStats, CtxStats), String> {
        let dmax = self.opt.ctx_draft;
        let hold = self.ctx_hold.ok_or("feed_stream_ctx: QwenOptions.ctx_draft = 0 (no hold save)")?.off;
        let mut st = StreamStats { stop_reason: "", ..Default::default() };
        let mut cs = CtxStats::default();
        if ids.is_empty() { return Err("feed_stream_ctx: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok((st, cs)); }
        let tp = Instant::now();
        let mut pending = ids[0];
        if ids.len() > 1 {
            let t = self.prefill(ids)?;
            cx.push(ids);
            st.gen = 1;
            st.last = Some(t);
            pending = t;
            if !on_tok(t) { st.stop_reason = "cancelled"; }
            else if self.is_eos(t) || stop.contains(&t) { st.stop_reason = "eos"; }
            else if st.gen >= n_new { st.stop_reason = "length"; }
        }
        st.prefill_s = tp.elapsed().as_secs_f64();
        if !st.stop_reason.is_empty() { return Ok((st, cs)); }
        let td = Instant::now();
        let r = self.ctx_loop(pending, dmax, hold, n_new, stop, on_tok, cx, &mut st, &mut cs);
        st.decode_s = td.elapsed().as_secs_f64();
        st.gpu_ms = cs.verify_ms + cs.replay_ms;
        cx.total.add(&cs);
        r?;
        if st.stop_reason.is_empty() { st.stop_reason = "length"; }
        Ok((st, cs))
    }

    #[allow(clippy::too_many_arguments)]
    fn ctx_loop(&self, mut pending: u32, dmax: usize, hold: usize, n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                cx: &mut CtxDrafter, st: &mut StreamStats, cs: &mut CtxStats) -> Result<(), String> {
        let max_ctx = self.opt.max_ctx;
        let cost = self.opt.spec_cost.clone();
        let ahead = self.rm.is_none() && self.dumper.is_none();
        let mut src_b = self.draft_src.borrow_mut();
        if let Some(s) = src_b.as_mut() {
            let mut seq = cx.history().to_vec();
            seq.push(pending);
            s.begin_sequence(&seq);
        }
        let ev = self.gpu.shared_event();
        let mut nsig = 0u64;            // signal value of the last committed replay
        let mut wait_v: Option<u64> = None; // the next verify must wait for this replay signal
        let mut pre: Vec<(CmdBuf, usize, usize, usize, Option<u64>)> = vec![]; // (cb, k, lo, hi, waits on)
        let mut last_replay: Option<CmdBuf> = None;
        let fin = |lr: &mut Option<CmdBuf>, cs: &mut CtxStats| -> Result<(), String> {
            if let Some(cb) = lr.take() { wait(&cb)?; cs.replay_ms += gpu_ms(&cb); }
            Ok(())
        };
        loop {
            let p = self.pos();
            if p + 1 > max_ctx { st.stop_reason = "ctx_full"; break; }
            // ---- drafts (CPU)
            let tc = Instant::now();
            let left = n_new.saturating_sub(st.gen).max(1);
            let cap = dmax.min(max_ctx - p - 1).min(left - 1);
            let h = cx.history();
            let mut tail: Vec<u32> = h[h.len().saturating_sub(TAIL - 1)..].to_vec();
            tail.push(pending);
            let mut pick: (usize, f64, Src, Vec<u32>, Vec<f64>, Vec<u8>) = (0, 1.0 / cost.pass_ms[0], Src::Lookup, vec![], vec![], vec![]);
            if cap > 0 {
                let (m, toks) = cx.lookup(pending, cap);
                if !toks.is_empty() {
                    let b = mbin(m);
                    let q: Vec<f64> = (0..toks.len()).map(|i| cx.cal[b][i]).collect();
                    let (d, r) = best(&q, &cost);
                    if d > 0 && r > pick.1 { pick = (d, r, Src::Lookup, toks, vec![b as f64; d], vec![]); }
                }
                if let Some(s) = src_b.as_mut() {
                    let prop = s.propose(&tail, cap);
                    if !prop.is_empty() {
                        let cond: Vec<f64> = prop.iter().enumerate()
                            .map(|(i, &(_, c))| (c as f64 / if i == 0 { 1.0 } else { (prop[i - 1].1 as f64).max(1e-6) }).min(1.0)).collect();
                        let q: Vec<f64> = cond.iter().map(|&c| 0.5 * cx.cal_t[tbin(c)] + 0.5 * c).collect();
                        let (d, r) = best(&q, &cost);
                        if d > 0 && r > pick.1 { pick = (d, r, Src::Table, prop.iter().map(|x| x.0).collect(), cond, vec![]); }
                    }
                }
                if let Some(hx) = cx.hindex.clone() {
                    let mut hq = std::mem::take(&mut cx.hq);
                    hx.query_into(&tail, cap, 1, &mut hq);
                    if let Some(row) = hq.rows.first().filter(|r| !r.tokens.is_empty()) {
                        let cond: Vec<f64> = row.conf.iter().enumerate()
                            .map(|(i, &c)| (c as f64 / if i == 0 { 1.0 } else { (row.conf[i - 1] as f64).max(1e-6) }).min(1.0)).collect();
                        let q: Vec<f64> = cond.iter().map(|&c| 0.5 * cx.cal_h[tbin(c)] + 0.5 * c).collect();
                        let (d, r) = best(&q, &cost);
                        if d > 0 && r > pick.1 { pick = (d, r, Src::History, row.tokens.clone(), cond, row.order.clone()); }
                    }
                    cx.hq = hq;
                }
            }
            let (nd, src) = (pick.0, pick.2);
            let drafts: Vec<u32> = pick.3[..nd].to_vec();
            cs.draft_cpu_ms += tc.elapsed().as_secs_f64() * 1e3;
            let k = nd + 1;
            let mut rows = Vec::with_capacity(k);
            rows.push(pending);
            rows.extend_from_slice(&drafts);
            self.reader.read_rows(&rows, self.stage_ptr(0))?;
            // ---- verify: encoded ahead (same K, range holds p, same wait) or now
            let w = wait_v.map(|v| (&ev, v));
            let vcb = match pre.iter().position(|x| x.1 == k && x.2 <= p && p <= x.3 && x.4 == wait_v) {
                Some(i) => { cs.preencoded += 1; pre.swap_remove(i).0 }
                None => enc_after(self, w, &|e| self.encode_ctx_verify(e, k, p, p))?,
            };
            pre.clear();
            self.rm_before_pass();
            commit(&vcb);
            // overlap: replay buffers of every outcome + the next round's verify (K = 1 and K = k)
            let mut after: Vec<CmdBuf> = vec![];
            if k > 1 {
                for je in 0..k {
                    let cb = enc_after(self, None, &|e| self.encode_replay_from(e, hold, je + 1))?;
                    after.push(cb);
                }
            }
            if ahead {
                let nw = if k > 1 { Some(nsig + 1) } else { None };
                let (lo, hi) = (p + 1, p + k);
                let mut kc = vec![1, k];
                kc.dedup();
                for kn in kc {
                    if hi + kn > max_ctx { continue; }
                    if let Ok(cb) = enc_after(self, nw.map(|v| (&ev, v)), &|e| self.encode_ctx_verify(e, kn, lo, hi)) { pre.push((cb, kn, lo, hi, nw)); }
                }
            }
            wait(&vcb)?;
            cs.verify_ms += gpu_ms(&vcb);
            fin(&mut last_replay, cs)?; // finished before this verify ran (it waited for its signal)
            // ---- decide
            let a: Vec<u32> = (0..k).map(|i| self.rec_token(p + i).0).collect();
            let mut j = 0;
            while j < nd && a[j] == drafts[j] && !self.is_eos(a[j]) { j += 1; }
            cs.passes += 1;
            cs.k_hist[k - 1] += 1;
            cs.accept_hist[j] += 1;
            if nd > 0 {
                cs.drafted_passes += 1;
                cs.drafted += nd;
                cs.accepted += j;
                if j < nd { cs.rejects += 1; }
                match src {
                    Src::Lookup => { cs.lookup_drafts += nd; cs.lookup_acc += j; }
                    Src::Table => { cs.table_drafts += nd; cs.table_acc += j; }
                    Src::History => { cs.history_drafts += nd; cs.history_acc += j; }
                }
                // calibration: conditional acceptance up to the first miss
                for i in 0..=j.min(nd - 1) {
                    let ok = if i < j { 1.0 } else { 0.0 };
                    match src {
                        Src::Lookup => { let b = pick.4[0] as usize; let c = &mut cx.cal[b][i]; *c += EW * (ok - *c); }
                        Src::Table => { let b = tbin(pick.4[i]); let c = &mut cx.cal_t[b]; *c += EW * (ok - *c); }
                        Src::History => {
                            let b = tbin(pick.4[i]); let c = &mut cx.cal_h[b]; *c += EW * (ok - *c);
                            if let Some(hx) = &cx.hindex { hx.feedback(pick.5[i], i < j); }
                        }
                    }
                }
            }
            let mut why: Option<&'static str> = None;
            let mut used = 0;
            for &t in &a[..=j] {
                used += 1;
                st.gen += 1;
                cs.tokens += 1;
                st.last = Some(t);
                if !on_tok(t) { why = Some("cancelled"); break; }
                if self.is_eos(t) || stop.contains(&t) { why = Some("eos"); break; }
                if st.gen >= n_new { why = Some("length"); break; }
            }
            let je = used - 1;
            let gaps: Vec<f32> = (0..=je).map(|i| self.rec_gap(p + i)).collect();
            self.rm_after_pass(&gaps);
            if let Some(s) = src_b.as_mut() {
                let top3: Vec<[u32; 3]> = (0..=je).map(|i| [a[i], u32::MAX, u32::MAX]).collect();
                s.observe(&tail, &a[..je], &top3);
            }
            self.ctrl().pos = (p + je + 1) as u32;
            self.arena.slice_mut::<f32>(self.sl().rec)[4 * (p + je + 1)..4 * (p + k)].fill(0.0);
            self.set_pos(p + je + 1);
            cx.push(&rows[..=je]);
            // ---- replay of the fed rows (HOLD pass): committed now, the next verify waits for its signal on the GPU
            wait_v = None;
            if k > 1 {
                let cb = after.swap_remove(je);
                nsig += 1;
                cb_signal(&cb, &ev, nsig);
                commit(&cb);
                wait_v = Some(nsig);
                last_replay = Some(cb);
            }
            drop(after);
            if why.is_none() && self.pos() + 1 > max_ctx { why = Some("ctx_full"); }
            if let Some(w) = why { st.stop_reason = w; break; }
            pending = a[je];
        }
        drop(pre);
        fin(&mut last_replay, cs)
    }
}
