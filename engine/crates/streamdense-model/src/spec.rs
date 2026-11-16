//! MTP speculative decode, draft 1 / verify 2, greedy and lossless (plan 7.3, 7.5; U20).
//! Step n (staging slot n % 2 holds [t1, d]):
//!   enc A: embed_q4 [t1, d] → copy emb(d) out of h → 64 layers K = 2 with the DeltaNet checkpoint (ck) → final
//!          norm (2 rows → hf) → lm_head K = 2 → argmax_b → spec_decide (emit, pos, role flip, mtp_rows, stop)
//!   CPU:   read ctrl.emit → tokens; eos / max → stop; pread emb(new t1) → slot (n+1) % 2 row 0
//!   enc B: embed_q4 emb(new t1) → mtp_prep (spec mode) → fc → MTP layer → select → norm → lm_head → mtp_draft
//!   CPU:   read ctrl.draft (optional oracle override) → pread emb(d) → slot (n+1) % 2 row 1
//! Pipelined (`pipe = true`): one command buffer per step with GPU/CPU events (E_dec, F_mtp, E_draft, F_emb) and the
//! two `Handoff` threads (MtpDec / MtpDraft); the main thread encodes CB(n+1) ahead, before step n decides, so the
//! host position is only a range [lo, hi] (`mtp::path_pos`). Sync: the same encoders in two command buffers per
//! step with the CPU in between. Both run identical GPU work, so their tokens are bit-identical.
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use streamdense_metal::{cb_signal, cb_wait, commit, ev_set, gpu_ms, wait, CmdBuf, Enc, SdCtrl};

use crate::embed::{CommitFn, DraftHook, Handoff, HandoffKind, HandoffLink, HandoffPolicy, LatStats};
use crate::mtp::{encode_mtp, path_pos, MtpRows};
use crate::qwen_engine::{QwenEngine, StreamStats};
use crate::qwen_layers::{encode_head, encode_layer, PassIo};

const BF: usize = 2;

/// Test-only draft override (plan 7.5). `Oracle`: when the MTP draft for absolute position q is read by the CPU,
/// replace it with `seq[q]` (the true greedy token, `seq` = prompt ++ plain tokens) with probability `p`.
#[derive(Clone, Debug, Default)]
pub enum DraftOverride {
    #[default]
    None,
    Oracle { seq: Vec<u32>, p: f64, seed: u64 },
    /// Bench hook (oracle-drafter bench, off unless passed): every draft for absolute position q is `seq[q]` (q <
    /// seq.len(); else the MTP draft), as `Oracle` with p = 1, AND the pipelined deep loop (`spec_loop_pipe`) skips the
    /// chained MTP steps 2..d (their command buffers are encoded but never committed), so a round costs what it would
    /// with an external zero-cost drafter: verify + replay + the MTP rows of the accepted tokens. The sync loop and the
    /// d = 1 loop ignore the skip (they behave as `Oracle` with p = 1).
    External { seq: Vec<u32> },
}

pub(crate) struct Oracle { seq: Vec<u32>, p: f64, s: u64, pub(crate) used: usize, pub(crate) skip_chain: bool }
impl Oracle {
    pub(crate) fn new(ov: &DraftOverride) -> Option<Oracle> {
        match ov {
            DraftOverride::None => None,
            DraftOverride::Oracle { seq, p, seed } => Some(Oracle { seq: seq.clone(), p: *p, s: seed ^ 0x9E37_79B9_7F4A_7C15, used: 0, skip_chain: false }),
            DraftOverride::External { seq } => Some(Oracle { seq: seq.clone(), p: 1.0, s: 0x9E37_79B9_7F4A_7C15, used: 0, skip_chain: true }),
        }
    }
    fn next_f64(&mut self) -> f64 { // splitmix64
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Draft for absolute position `q` (one random draw per call, so sync and pipelined runs see the same drafts).
    pub(crate) fn apply(&mut self, q: usize, d: u32) -> u32 {
        let r = self.next_f64();
        if q < self.seq.len() && r < self.p {
            if self.seq[q] != d { self.used += 1; }
            return self.seq[q];
        }
        d
    }
}

#[derive(Default, Debug, Clone)]
pub struct SpecResult {
    /// emitted tokens; tokens[0] comes from the prefill
    pub tokens: Vec<u32>,
    /// verify passes (steps)
    pub passes: usize,
    pub accepted: usize,
    pub rejects: usize,
    /// accepted / passes
    pub acceptance: f64,
    /// (tokens - 1) / passes
    pub tokens_per_pass: f64,
    /// decode wall time (s): first step commit → last token
    pub wall: f64,
    /// GPU ms per step (sync: enc A + enc B command buffers)
    pub gpu_ms: Vec<f64>,
    pub prefill_s: f64,
    /// bf16 top-1 − top-2 of the target logit that produced each token (from the rec table)
    pub margins: Vec<f32>,
    /// drafts replaced by the oracle (different from the MTP draft)
    pub overridden: usize,
    /// the MTP head's own drafts (before any override), in order: after the prefill, then after each step that
    /// continued. drafts[i] predicts the token after the one at position ctrl.pos at that time.
    pub drafts: Vec<u32>,
    /// ctrl.pos (position of the token the draft follows) per entry of `drafts`
    pub draft_pos: Vec<u32>,
    /// pipelined only: E_dec → next CB committed, and E_draft → F_emb signalled
    pub handoff_dec: Option<LatStats>,
    pub handoff_draft: Option<LatStats>,
    /// drafts verified per pass (1 = this file's draft 1 / verify 2; >= 2 = spec_deep.rs). With depth d, `accepted`
    /// counts accepted drafts (0..=d per pass), so `acceptance` = mean accepted drafts per pass.
    pub depth: usize,
    /// spec_deep only: passes that accepted j drafts, j = 0..=depth
    pub accept_hist: Vec<usize>,
    /// spec_deep only: per DeltaNet replay (rows replayed, GPU ms of its command buffer)
    pub replay: Vec<(usize, f64)>,
    /// spec_deep only: GPU ms per verify pass, and per step of the MTP work (rows + chained drafts, summed)
    pub verify_ms: Vec<f64>,
    pub draft_ms: Vec<f64>,
    /// spec_deep pipelined loop: depth policy ("3", "auto7"); passes by real draft rows (pads excluded) 0..=max
    pub mode: String,
    pub depth_hist: Vec<usize>,
    /// spec_deep / batch: verify passes straddling an attention path boundary (stats only)
    pub straddle: crate::qwen_layers::AttnStraddle,
    /// draft rows verified / accepted by source: MTP head, draft table, padding (spare rows of a pass encoded ahead)
    pub drafts_mtp: usize,
    pub drafts_table: usize,
    pub drafts_pad: usize,
    pub acc_mtp: usize,
    pub acc_table: usize,
    pub acc_pad: usize,
    /// per round: GPU idle ms between its command buffers (incl. the gap before its first) and wall ms
    pub idle_ms: Vec<f64>,
    pub round_ms: Vec<f64>,
    /// CPU ms: verify passes encoded on the critical path (no usable pass encoded ahead), and the encode work done
    /// while a verify runs (outcome buffers + next verify); verify passes taken from the encoded-ahead one
    pub reencode_ms: Vec<f64>,
    pub overlap_encode_ms: Vec<f64>,
    pub preencoded: usize,
}

impl SpecResult {
    pub(crate) fn finish(&mut self) {
        self.depth = self.depth.max(1);
        self.acceptance = if self.passes > 0 { self.accepted as f64 / self.passes as f64 } else { 0.0 };
        self.tokens_per_pass = if self.passes > 0 { (self.tokens.len().saturating_sub(1)) as f64 / self.passes as f64 } else { 0.0 };
    }
}

struct SendCb(CmdBuf);
unsafe impl Send for SendCb {}
#[derive(Default)]
struct PendState { cbs: std::collections::HashMap<u64, SendCb>, committed: u64, closed: bool }
#[derive(Default)]
struct Pending { m: Mutex<PendState>, cv: Condvar }

/// Position range of step j (j >= 1) when step 1 starts at p1: each step advances 1 or 2.
fn bounds(eng: &QwenEngine, p1: usize, j: usize) -> (usize, usize) {
    let lo = p1 + (j - 1);
    let hi = (p1 + 2 * (j - 1)).min(eng.opt.max_ctx.saturating_sub(2)).max(lo);
    (lo, hi)
}

impl QwenEngine {
    /// enc A of step with staging slot `slot`, ctrl.pos in [lo, hi].
    fn encode_verify(&self, e: &Enc, slot: usize, lo: usize, hi: usize) -> Result<(), String> {
        let (_, _, ms) = self.mtp_parts()?;
        let c = &self.cfg;
        let q8 = self.opt.kv == crate::qwen_engine::KvFormat::Q8;
        let io = PassIo { k: 2, pos: path_pos(lo, hi, 2, q8)?, nch_max: self.nch_for(hi, 2), rm: self.rm_io(true), ..self.io() };
        e.pso("embed_q4").b(0, self.sl().stage.at(slot * self.slot_bytes)).b(1, io.h).u(2, 2);
        e.threads((c.hidden / 8, 2), 64);
        // emb(d) = row 1 of the verify input, before the layers update h in place
        e.pso("copy_bytes").b(0, io.h + c.hidden * BF).b(1, ms.emb).u(2, (c.hidden * BF) as u32).threads((c.hidden * BF, 1), 256);
        for l in 0..c.layers {
            encode_layer(e, c, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, true, None, l)?;
        }
        encode_head(e, c, &self.w, &io, 2, None)?;
        e.pso("spec_decide").b(0, io.ctrl).b(1, io.rec).b(2, ms.eos.off).u(3, ms.n_eos as u32).threads((1, 1), 1);
        Ok(())
    }

    /// enc B: emb(new t1) from row 0 of staging slot `next_slot`, then the MTP draft; mctrl.pos (= ctrl.pos before
    /// the step) in [lo, hi].
    fn encode_draft(&self, e: &Enc, next_slot: usize, lo: usize, hi: usize) -> Result<(), String> {
        let (w, kv, ms) = self.mtp_parts()?;
        let c = &self.cfg;
        let q8 = self.opt.kv == crate::qwen_engine::KvFormat::Q8;
        e.pso("embed_q4").b(0, self.sl().stage.at(next_slot * self.slot_bytes)).b(1, ms.emb + c.hidden * BF).u(2, 1);
        e.threads((c.hidden / 8, 1), 64);
        encode_mtp(e, c, w, &self.w.lm_head, kv, &self.io(), ms, MtpRows::Spec, path_pos(lo, hi, 2, q8)?, self.nch_for(hi, 2), true)
    }

    /// Prefill + MTP warm-up + first draft (override applied); leaves [t1, d] staged in slot 1. Returns t1.
    fn spec_start(&self, ids: &[u32], oracle: &mut Option<Oracle>, r: &mut SpecResult) -> Result<u32, String> {
        let (t1, hf) = self.prefill_mtp(ids)?;
        let mut d = self.mtp_first_draft(t1, hf, 1)?;
        let c = self.ctrl();
        r.drafts.push(d);
        r.draft_pos.push(c.pos);
        if let Some(o) = oracle.as_mut() {
            let nd = o.apply(c.pos as usize + 1, d);
            if nd != d { c.draft = nd; d = nd; }
        }
        self.reader.read_rows(&[d], unsafe { self.stage_ptr(1).add(self.reader.row_bytes()) })?;
        Ok(t1)
    }

    pub(crate) fn margins_check(&self, p_len: usize, r: &mut SpecResult) -> Result<(), String> {
        let rec = self.arena.slice_mut::<f32>(self.sl().rec);
        for (i, &t) in r.tokens.iter().enumerate() {
            let p = p_len - 1 + i;
            let rt = rec[4 * p].to_bits();
            if rt != t { return Err(format!("mtp token {}: record {} != emitted {}", i, rt, t)); }
            r.margins.push(rec[4 * p + 1] - rec[4 * p + 2]);
        }
        Ok(())
    }

    /// Synchronous step loop from the current state ([t1, d] staged in slot 1, t1 already emitted and counted in
    /// `st.gen`). Emits through `on_tok`; stops on eos, a token in `stop`, `n_new` total tokens, cancel, or when a
    /// verify pass would pass max_ctx. On exit the last emitted token is never fed (it is ctrl.pos's token): a stop
    /// right after an accepted draft d rolls d back (role flip, pos − 1), and the MTP row of the last fed position
    /// is always fed with emb(last token), so a later `feed_stream_mtp` can continue.
    fn spec_loop_sync(&self, n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool, oracle: &mut Option<Oracle>,
                      st: &mut StreamStats, r: &mut SpecResult) -> Result<(), String> {
        let rb = self.reader.row_bytes();
        let mut j = 1usize;
        loop {
            let p = self.pos();
            if p + 2 > self.opt.max_ctx { st.stop_reason = "ctx_full"; break; }
            let slot = j % 2;
            let next = (j + 1) % 2;
            let cb = self.gpu.command_buffer();
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            let res = self.encode_verify(&e, slot, p, p);
            e.end();
            res?;
            self.rm_before_pass();
            commit(&cb);
            wait(&cb)?;
            let mut g = gpu_ms(&cb);
            let c = self.ctrl_read();
            let n_emit = (c.n_emit as usize).min(2);
            r.passes += 1;
            if n_emit == 2 { r.accepted += 1; } else { r.rejects += 1; }
            let mut why: Option<&'static str> = None;
            let mut used = 0;
            for &t in &c.emit[..n_emit] {
                used += 1;
                st.gen += 1;
                st.last = Some(t);
                r.tokens.push(t);
                if !on_tok(t) { why = Some("cancelled"); break; }
                if self.is_eos(t) || stop.contains(&t) { why = Some("eos"); break; }
                if st.gen >= n_new { why = Some("length"); break; }
            }
            let held = c.emit[used - 1];
            let gaps: Vec<f32> = (0..used).map(|i| self.rec_gap(p + i)).collect();
            self.rm_after_pass(&gaps);
            if used < n_emit {
                // stopped at the accepted draft d: back to the post-t1 state, d held back
                let cw = self.ctrl();
                cw.role ^= 1;
                cw.pos -= 1;
                cw.mtp_rows = 1;
                cw.n_emit = 1;
            }
            self.set_pos(self.ctrl_read().pos as usize);
            self.reader.read_rows(&[held], self.stage_ptr(next))?;
            let cb = self.gpu.command_buffer();
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            let res = self.encode_draft(&e, next, p, p);
            e.end();
            res?;
            commit(&cb);
            wait(&cb)?;
            g += gpu_ms(&cb);
            r.gpu_ms.push(g);
            if let Some(w) = why { st.stop_reason = w; break; }
            let cw = self.ctrl();
            let mut d = cw.draft;
            r.drafts.push(d);
            r.draft_pos.push(cw.pos);
            if let Some(o) = oracle.as_mut() {
                let nd = o.apply(cw.pos as usize + 1, d);
                if nd != d { cw.draft = nd; d = nd; }
            }
            self.reader.read_rows(&[d], unsafe { self.stage_ptr(next).add(rb) })?;
            j += 1;
        }
        st.stop_reason = if st.stop_reason.is_empty() { "length" } else { st.stop_reason };
        Ok(())
    }

    /// Pipelined steps (plan 7.3): `max_emit` tokens after t1 at most. Tokens, stats into `r`.
    fn spec_loop_pipelined(&self, max_emit: usize, oracle: Option<Oracle>, r: &mut SpecResult) -> Result<(), String> {
        self.mtp_parts()?;
        let (e_dec, e_draft, f_mtp, f_emb) = (self.gpu.shared_event(), self.gpu.shared_event(), self.gpu.shared_event(), self.gpu.shared_event());
        ev_set(&f_emb, 1); // [t1, d] of step 1 are staged
        let p1 = self.pos();
        let enc = |j: u64| -> Result<CmdBuf, String> {
            let (lo, hi) = bounds(self, p1, j as usize);
            let slot = (j % 2) as usize;
            let cb = self.gpu.command_buffer();
            cb_wait(&cb, &f_emb, j);
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            let res = self.encode_verify(&e, slot, lo, hi);
            e.end();
            res?;
            cb_signal(&cb, &e_dec, j);
            cb_wait(&cb, &f_mtp, j);
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            let res = self.encode_draft(&e, 1 - slot, lo, hi);
            e.end();
            res?;
            cb_signal(&cb, &e_draft, j);
            Ok(cb)
        };
        let pend = Arc::new(Pending::default());
        let pc = pend.clone();
        let commit_fn: CommitFn = Box::new(move |n| {
            let mut g = pc.m.lock().unwrap();
            loop {
                if let Some(cb) = g.cbs.remove(&n) { commit(&cb.0); g.committed = n; pc.cv.notify_all(); return; }
                if g.closed { return; }
                g = pc.cv.wait(g).unwrap();
            }
        });
        // per-step n_emit (read in the draft hand-off, after spec_decide) + oracle override
        let log: Arc<Mutex<Vec<(u32, u32, u32)>>> = Arc::new(Mutex::new(Vec::new()));
        let orc = Arc::new(Mutex::new(oracle));
        let ctrl_addr = self.arena.ptr(self.sl().ctrl) as usize;
        let (lg, oc) = (log.clone(), orc.clone());
        let hook: DraftHook = Box::new(move |_n, d| {
            let c: SdCtrl = unsafe { std::ptr::read_volatile(ctrl_addr as *const SdCtrl) };
            lg.lock().unwrap().push((c.n_emit, d, c.pos));
            match oc.lock().unwrap().as_mut() { Some(o) => o.apply(c.pos as usize + 1, d), None => d }
        });
        let link = HandoffLink::new();
        let mut pdec = HandoffPolicy::new(HandoffKind::MtpDec, 1, self.tmax, self.cfg.eos.clone(), max_emit);
        pdec.signal_ev = Some(f_mtp.clone());
        pdec.link = Some(link.clone());
        let mut pdr = HandoffPolicy::new(HandoffKind::MtpDraft, 1, self.tmax, self.cfg.eos.clone(), max_emit);
        pdr.signal_ev = Some(f_emb.clone());
        pdr.link = Some(link.clone());
        pdr.draft_hook = Some(hook);

        let cb1 = enc(1)?;
        let mut all: Vec<CmdBuf> = vec![cb1.clone()];
        let ctrl_ptr = self.arena.ptr(self.sl().ctrl) as *mut SdCtrl;
        let t0 = Instant::now();
        let hdr = Handoff::spawn(self.reader.clone(), e_draft.clone(), ctrl_ptr, self.stage_ptr(0), pdr, None);
        let hdec = Handoff::spawn(self.reader.clone(), e_dec.clone(), ctrl_ptr, self.stage_ptr(0), pdec, Some(commit_fn));
        commit(&cb1);
        pend.m.lock().unwrap().committed = 1;
        let close = |p: &Pending| { let mut g = p.m.lock().unwrap(); g.closed = true; g.cbs.clear(); p.cv.notify_all(); };
        let depth = self.opt.depth.max(1) as u64;
        let mut err: Option<String> = None;
        for j in 2..=max_emit as u64 {
            {
                let mut g = pend.m.lock().unwrap();
                while g.committed + depth < j && !hdec.is_finished() {
                    g = pend.cv.wait_timeout(g, Duration::from_millis(1)).unwrap().0;
                }
            }
            if hdec.is_finished() { break; }
            match enc(j) {
                Ok(cb) => { all.push(cb.clone()); let mut g = pend.m.lock().unwrap(); g.cbs.insert(j, SendCb(cb)); pend.cv.notify_all(); }
                Err(e) => { hdec.abort(); err = Some(e); break; }
            }
        }
        let out_dec = hdec.join();
        close(&pend);
        let out_dr = hdr.join();
        let committed = pend.m.lock().unwrap().committed;
        for (i, cb) in all.iter().enumerate() {
            if (i + 1) as u64 > committed { break; }
            wait(cb)?;
            r.gpu_ms.push(gpu_ms(cb));
        }
        r.wall = t0.elapsed().as_secs_f64();
        if let Some(e) = err { return Err(e); }
        let out_dec = out_dec?;
        let out_dr = out_dr?;
        r.tokens.extend_from_slice(&out_dec.tokens);
        r.passes = out_dec.steps as usize;
        // steps 1..passes-1 were logged by the draft hand-off; the last step's n_emit is still in ctrl
        let lgv = log.lock().unwrap().clone();
        r.drafts.extend(lgv.iter().map(|x| x.1));
        r.draft_pos.extend(lgv.iter().map(|x| x.2));
        let mut ne: Vec<u32> = lgv.iter().map(|x| x.0).collect();
        ne.truncate(r.passes.saturating_sub(1));
        ne.push(self.ctrl_read().n_emit);
        r.accepted = ne.iter().filter(|&&n| n == 2).count();
        r.rejects = r.passes - r.accepted;
        r.overridden += orc.lock().unwrap().as_ref().map_or(0, |o| o.used);
        r.handoff_dec = Some(out_dec.handoff);
        r.handoff_draft = Some(out_dr.handoff);
        self.set_pos(self.ctrl_read().pos as usize);
        Ok(())
    }

    /// Streaming MTP decode for `qwen serve` (sync loop): same contract as `feed_stream` (feeds `ids` from the
    /// current position, the last emitted token is held back), plus the spec stats. Verify passes need pos + 2 <=
    /// max_ctx, so "ctx_full" comes one token earlier than plain. Verifies `opt.spec_depth` chained drafts per pass
    /// (1 = this file's loop; >= 2 = `feed_stream_spec`, spec_deep.rs).
    pub fn feed_stream_mtp(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool)
                           -> Result<(StreamStats, SpecResult), String> {
        if self.opt.spec_depth > 1 || self.opt.spec_auto {
            let r = self.feed_stream_spec_mode(ids, n_new, stop, on_tok, self.spec_mode(), &DraftOverride::None);
            self.flush_draft_source();
            return r;
        }
        self.feed_stream_mtp_ov(ids, n_new, stop, on_tok, &DraftOverride::None)
    }

    /// `feed_stream_mtp` with a draft override (test hook: the oracle's `seq` is indexed by absolute position, so it
    /// is the whole session's token sequence, fed and generated). `DraftOverride::None` = `feed_stream_mtp`.
    pub fn feed_stream_mtp_ov(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                              ov: &DraftOverride) -> Result<(StreamStats, SpecResult), String> {
        let mut st = StreamStats { stop_reason: "", ..Default::default() };
        let mut r = SpecResult::default();
        if ids.is_empty() { return Err("feed_stream_mtp: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok((st, r)); }
        let tp = Instant::now();
        let mut none = Oracle::new(ov);
        let t1 = self.spec_start(ids, &mut none, &mut r)?;
        st.prefill_s = tp.elapsed().as_secs_f64();
        let td = Instant::now();
        st.gen = 1;
        st.last = Some(t1);
        r.tokens.push(t1);
        if !on_tok(t1) { st.stop_reason = "cancelled"; }
        else if self.is_eos(t1) || stop.contains(&t1) { st.stop_reason = "eos"; }
        else if n_new <= 1 { st.stop_reason = "length"; }
        else { self.spec_loop_sync(n_new, stop, on_tok, &mut none, &mut st, &mut r)?; }
        st.decode_s = td.elapsed().as_secs_f64();
        st.gpu_ms = r.gpu_ms.iter().sum();
        r.wall = st.decode_s;
        r.overridden += none.as_ref().map_or(0, |o| o.used);
        r.finish();
        Ok((st, r))
    }
}

/// Greedy MTP generation from a fresh state (plan U20): prefill + warm-up + first draft, then draft-1 / verify-2
/// steps until eos or `n_new` tokens. `pipe`: one command buffer per step with the hand-off threads, else the sync
/// loop (same GPU work, same tokens). Tokens must equal `QwenEngine::generate` bit-exactly.
pub fn generate_mtp(eng: &QwenEngine, prompt: &[u32], n_new: usize, ov: DraftOverride, pipe: bool) -> Result<SpecResult, String> {
    eng.mtp_parts()?;
    let p_len = prompt.len();
    if p_len == 0 { return Err("generate_mtp: empty prompt".into()); }
    if p_len + n_new + 1 > eng.opt.max_ctx {
        return Err(format!("prompt {} + {} new (+1 verify row) > max_ctx {}", p_len, n_new, eng.opt.max_ctx));
    }
    let mut r = SpecResult::default();
    eng.reset();
    if n_new == 0 { eng.prefill(prompt)?; return Ok(r); }
    let mut oracle = Oracle::new(&ov);
    let tp = Instant::now();
    let t1 = eng.spec_start(prompt, &mut oracle, &mut r)?;
    r.prefill_s = tp.elapsed().as_secs_f64();
    r.tokens.push(t1);
    if n_new > 1 && !eng.is_eos(t1) {
        if pipe {
            let used0 = oracle.as_ref().map_or(0, |o| o.used);
            eng.spec_loop_pipelined(n_new - 1, oracle, &mut r)?;
            r.overridden += used0;
        } else {
            let mut st = StreamStats { gen: 1, ..Default::default() };
            let t0 = Instant::now();
            eng.spec_loop_sync(n_new, &[], &mut |_| true, &mut oracle, &mut st, &mut r)?;
            r.wall = t0.elapsed().as_secs_f64();
            r.overridden += oracle.as_ref().map_or(0, |o| o.used);
        }
    }
    r.tokens.truncate(n_new);
    eng.margins_check(p_len, &mut r)?;
    r.finish();
    Ok(r)
}
