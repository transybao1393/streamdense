//! 12 GB mode (research; `[forheader] mode = "12gb"`): a `Backend` over a compact-FFN engine
//! (`QwenOptions.ffn_compact` + streamed prefill). OFF by default: the gateway serves the plain `QwenEngine` then.
//!
//! Every fed prompt is read with the FULL FFN, streamed from the model file layer by layer
//! (`QwenEngine::stream_prefill_seeded`), and each layer's kept neuron set is rebuilt from the prompt sums in the same
//! pass (`ffn_set::SetRule`); decode then runs on the compact set (`feed_stream` / `feed_stream_mtp`).
//!
//! KV reuse: the backend keeps the per-layer sums of every token it stream-read since the last `reset` (`seed`,
//! f32 [layers][2 x inter]; 8.9 MB on the 27B, + one copy for the pass). A feed that continues the sequence (planner
//! prefix / followup) stream-reads only the new tokens, seeded with those sums, so the set is rebuilt from the whole
//! prompt: for a held sequence that was entirely stream-read, KV / state / sums / set / tokens equal one stream read
//! of the whole sequence bit for bit when the held length is a multiple of `prefill_k` (`prefill_t` with the fast read,
//! `[forheader] fast`; the an sums are added per chunk; other lengths change only their fp32 order; with the exact
//! read the KV is exact either way). Tokens the engine DECODED (a
//! previous reply) sit in the KV from the compact passes, as in any KV reuse, and are not in the sums.
//! Long feeds go in pieces of `piece` tokens (a multiple of that alignment; the set is built once, after the last piece).
//!
//! Snapshots (gateway snapshot cache, `snap::CacheHook`): a 12 GB-mode snapshot is the engine state + the sums + the
//! held length + the piece alignment (`snapshot::CompactMeta`, file format 2; `save_snapshot` / `restore_snapshot`
//! here). Restoring it gives the backend exactly the state it had in memory, so the next feed continues as in-memory
//! reuse does (same rebuild rule: the set is built from the whole stream-read sequence, after the last piece of
//! the feed that decodes). Snapshot cut feeds (`feed_cut`) read without rebuilding the set, and the worker rounds
//! cuts to `align` tokens after the feed's start (`cut_align`), so a request with cuts reads in the same chunks as
//! one without: snapshots change no output bit. Full-model snapshots are refused here and these are refused by the
//! full model (clear error; separate cache keys keep them apart).
//! Context drafts (`QwenOptions.ctx_draft` > 0, `[forheader] ctx_draft`; engine ctx_draft.rs): with MTP off the decode
//! after the first token is `feed_stream_ctx`: drafts from the lookup over every token in the KV (`CtxDrafter`, kept
//! here: every stream-read / cut / decoded token, replaced on a snapshot restore, cleared on reset) and the draft table
//! when the engine has one; one exact verify pass per block, output identical to `feed_stream`. `ctx_dry`: the engine
//! keeps the ctx_draft RAM (same K) but decodes plainly (A/B tests at the same resident set).
//! History index (`[history]`, `set_history`): a third draft source of the context drafts (streamdense-core
//! history_index.rs). Its background updater is paused while a feed runs (GPU + SSD work) and gets every finished
//! reply (8 tokens of context + the reply) afterwards.
//! Not supported here: the activity recorder / answer library (compact refuses the recorder).
use crate::worker::{Backend, FeedStats, StreamFeed};
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::Instant;
use std::sync::Arc;
use streamdense_core::history_index::{HistoryIndex, Updater};
use streamdense_model::ctx_draft::{CtxDrafter, CtxStats};
use streamdense_model::ffn_set::SetRule;
use streamdense_model::snapshot::CompactMeta;
use streamdense_model::QwenEngine;

pub struct CompactBackend<'a> {
    pub eng: &'a QwenEngine,
    pub rule: SetRule,
    /// sums of every stream-read token since the last reset: [layer][an | ag]
    seed: RefCell<Vec<Vec<f32>>>,
    next: RefCell<Vec<Vec<f32>>>,
    /// stream-read tokens in `seed` (decoded tokens are in the KV only)
    held: Cell<usize>,
    /// max tokens per streamed pass (a multiple of `align`, <= the engine's ffn_stream_tokens)
    pub piece: usize,
    /// chunk alignment of a streamed read: prefill_k (exact read) or lcm(prefill_t, prefill_k) (fast read)
    pub align: usize,
    /// /health + request log (`Backend::mode_info`)
    pub info: Value,
    /// process peak footprint (the CLI's monitor; `Backend::peak_footprint`)
    pub peak: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    /// context drafts: lookup history + calibration (Some when the engine has ctx_draft > 0)
    ctx: Option<RefCell<CtxDrafter>>,
    /// keep the ctx_draft RAM but decode plainly (A/B at the same K)
    pub ctx_dry: bool,
    /// history index + its background updater (`set_history`); None = off
    history: Option<HistoryHook>,
}

/// The history index the context drafts query + the updater that appends finished replies.
pub struct HistoryHook {
    pub index: Arc<HistoryIndex>,
    pub updater: Updater,
}

/// Resumes the updater when the feed ends (also on errors).
struct Resume<'a>(Option<&'a Updater>);
impl Drop for Resume<'_> { fn drop(&mut self) { if let Some(u) = self.0 { u.resume(); } } }

impl<'a> CompactBackend<'a> {
    /// `eng` must have the compact FFN with this rule's K in every layer and the streamed prefill on.
    pub fn new(eng: &'a QwenEngine, rule: SetRule, info: Value) -> Result<CompactBackend<'a>, String> {
        if !eng.stream_prefill_on() { return Err("12 GB mode: the engine has no streamed prefill (QwenOptions.ffn_stream_tokens)".into()); }
        let set = eng.ffn_compact_set().ok_or("12 GB mode: the engine has no compact FFN (QwenOptions.ffn_compact)")?;
        let (l, n) = (eng.cfg.layers, eng.cfg.inter);
        rule.t1.check_dims(l, n)?;
        if set.counts().iter().any(|&c| c != rule.k) { return Err(format!("12 GB mode: engine set sizes {:?} != K {}", &set.counts()[..1], rule.k)); }
        // pieces line up with one read's chunks: prefill_k rows (exact read) or prefill_t rows (fast read, a multiple of 8)
        let k = eng.opt.prefill_k.max(1);
        let al = if eng.opt.ffn_stream_fast { (eng.opt.prefill_t.max(1) * k) / gcd(eng.opt.prefill_t.max(1), k) } else { k };
        let piece = eng.opt.ffn_stream_tokens / al * al;
        if piece == 0 { return Err(format!("12 GB mode: stream tokens {} < piece alignment {} (prefill_k, or lcm with prefill_t on the fast read)", eng.opt.ffn_stream_tokens, al)); }
        Ok(CompactBackend { eng, rule, seed: RefCell::new(vec![vec![0f32; 2 * n]; l]), next: RefCell::new(vec![vec![0f32; 2 * n]; l]), held: Cell::new(0),
                            piece, align: al, info, peak: None, ctx: (eng.opt.ctx_draft > 0).then(|| RefCell::new(CtxDrafter::new())), ctx_dry: false, history: None })
    }

    /// Attach the history index as a draft source of the context drafts (no-op for the drafts when ctx_draft is off;
    /// the updater still gets the replies).
    pub fn set_history(&mut self, h: HistoryHook) {
        if let Some(c) = &self.ctx { c.borrow_mut().hindex = Some(h.index.clone()); }
        self.history = Some(h);
    }
    pub fn history(&self) -> Option<&HistoryHook> { self.history.as_ref() }
    /// The prompt sums held now (tests).
    pub fn sums(&self) -> Vec<Vec<f32>> { self.seed.borrow().clone() }
    /// Context-draft stats summed over every decode since the backend was made (None = ctx_draft off).
    pub fn ctx_totals(&self) -> Option<CtxStats> { self.ctx.as_ref().map(|c| c.borrow().total.clone()) }
    /// Context drafts are used by `feed` (ctx_draft > 0, not dry, MTP not requested).
    pub fn ctx_live(&self) -> bool { self.ctx.is_some() && !self.ctx_dry }
    /// Stream-read tokens in the sums since the last reset.
    pub fn held(&self) -> usize { self.held.get() }

    /// The snapshot section of the current state.
    pub fn meta(&self) -> CompactMeta {
        CompactMeta { layers: self.eng.cfg.layers, width: 2 * self.eng.cfg.inter, align: self.align, fast: self.eng.opt.ffn_stream_fast,
                      piece: self.piece, held: self.held.get() }
    }
    /// Snapshot cache tag of this setup (`SnapshotCache.mode`): files of another alignment / read never match.
    pub fn snap_mode(&self) -> String { format!("c{}{}", self.align, if self.eng.opt.ffn_stream_fast { "f" } else { "x" }) }

    /// Save the engine state + sums + held length (format 2). `tokens` = every fed id (len == pos()).
    pub fn save_snapshot(&self, path: &Path, tokens: &[u32]) -> Result<(), String> {
        self.eng.save_snapshot_compact(path, tokens, &self.meta(), &self.seed.borrow())
    }
    /// Restore a 12 GB-mode snapshot: engine state, sums and held length exactly as when saved. A full-model
    /// snapshot (or one of another alignment / read) is refused with a clear error. On any error the backend is
    /// reset (engine + sums; the worker resets after a failed restore anyway).
    pub fn restore_snapshot(&self, path: &Path) -> Result<Vec<u32>, String> {
        let r = self.eng.restore_snapshot_compact(path, &self.meta(), &mut self.seed.borrow_mut());
        match r {
            Ok((t, m)) => { self.held.set(m.held); if let Some(c) = &self.ctx { c.borrow_mut().set_history(&t); } Ok(t) }
            Err(e) => {
                // a refusal touches nothing, a short read leaves half-read sums: start clean either way
                self.reset();
                Err(e)
            }
        }
    }

    /// Stream-read `ids` (pieces); `build`: rebuild the set after the last one (else a read-only pass that keeps the
    /// current set: snapshot cuts). Returns (first generated token, stats).
    fn read(&self, ids: &[u32], build: bool) -> Result<(u32, StreamFeed), String> {
        let n = self.eng.cfg.inter;
        let mut sf = StreamFeed::default();
        let t0 = Instant::now();
        let (mut b0, mut tok) = (0usize, 0u32);
        while b0 < ids.len() {
            let b1 = if ids.len() - b0 <= self.piece { ids.len() } else { b0 + self.piece };
            let fin = build && b1 == ids.len();
            let st = {
                let seed = self.seed.borrow();
                let mut next = self.next.borrow_mut();
                let rule = &self.rule;
                self.eng.stream_prefill_seeded(&ids[b0..b1], Some(&seed[..]), &mut |li, an, ag| {
                    next[li][..n].copy_from_slice(an);
                    next[li][n..].copy_from_slice(ag);
                    Ok(if fin { rule.layer(li, an, ag) } else { vec![] })
                })?
            };
            std::mem::swap(&mut *self.seed.borrow_mut(), &mut *self.next.borrow_mut());
            self.held.set(self.held.get() + (b1 - b0));
            if let Some(c) = &self.ctx { c.borrow_mut().push(&ids[b0..b1]); }
            sf.add(&StreamFeed { pieces: 1, tokens: st.tokens, ssd_bytes: st.ssd_bytes, ssd_busy_s: st.ssd_busy_s, wall_s: 0.0, gpu_s: st.gpu_s,
                                 load_wait_s: st.load_wait_s, policy_s: st.policy_s, fill_s: st.fill_s, changed: st.changed, host_reload_bytes: st.host_reload_bytes });
            tok = st.tok;
            b0 = b1;
        }
        sf.wall_s = t0.elapsed().as_secs_f64();
        Ok((tok, sf))
    }
}

fn gcd(a: usize, b: usize) -> usize { if b == 0 { a } else { gcd(b, a % b) } }

impl Backend for CompactBackend<'_> {
    fn reset(&self) {
        self.eng.reset();
        for v in self.seed.borrow_mut().iter_mut() { v.fill(0.0); }
        self.held.set(0);
        if let Some(c) = &self.ctx { c.borrow_mut().reset(); }
    }
    fn pos(&self) -> usize { self.eng.pos() }
    fn max_ctx(&self) -> usize { self.eng.opt.max_ctx }
    fn vocab(&self) -> usize { self.eng.cfg.vocab }
    fn mtp_available(&self) -> bool { self.eng.opt.mtp }
    fn mode_info(&self) -> Option<Value> { Some(self.info.clone()) }
    fn peak_footprint(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicU64>> { self.peak.clone() }
    fn cut_align(&self) -> usize { self.align }
    /// whole streamed pieces: a warm-up reads in as many passes as one read of the prompt
    fn warm_chunk(&self) -> usize { let p = self.piece.max(1); (crate::worker::WARM_CHUNK / p).max(1) * p }
    /// A snapshot cut: read `ids` with the full FFN into the KV + sums, keep the set (it is rebuilt by the feed that
    /// decodes, from the whole stream-read sequence). The generated token is discarded by the worker.
    fn feed_cut(&self, ids: &[u32], _mtp: bool) -> Result<FeedStats, String> {
        let mut f = FeedStats { stop_reason: "length".into(), ..Default::default() };
        if ids.is_empty() { return Err("feed: no tokens".into()); }
        if self.eng.pos() + ids.len() > self.eng.opt.max_ctx { f.stop_reason = "ctx_full".into(); return Ok(f); }
        let (t1, sf) = self.read(ids, false)?;
        f.prefill_s = sf.wall_s;
        f.stream = Some(sf);
        f.gen = 1;
        f.last = Some(t1);
        Ok(f)
    }
    fn feed(&self, ids: &[u32], n_new: usize, stop: &[u32], mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> {
        let Some(h) = &self.history else { return self.feed_inner(ids, n_new, stop, mtp, on_tok) };
        // history index: no background indexing while the GPU / SSD work; the reply goes into the index afterwards
        h.updater.pause();
        let _resume = Resume(Some(&h.updater));
        let mut out: Vec<u32> = ids[ids.len().saturating_sub(8)..].to_vec();
        let n0 = out.len();
        let mut tap = |t: u32| { out.push(t); on_tok(t) };
        let r = self.feed_inner(ids, n_new, stop, mtp, &mut tap);
        if out.len() > n0 + 1 { h.updater.push_tokens(out); }
        r
    }
}

impl CompactBackend<'_> {
    fn feed_inner(&self, ids: &[u32], n_new: usize, stop: &[u32], mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> {
        let mut f = FeedStats { stop_reason: "length".into(), ..Default::default() };
        if ids.is_empty() { return Err("feed: no tokens".into()); }
        if self.eng.pos() + ids.len() > self.eng.opt.max_ctx { f.stop_reason = "ctx_full".into(); return Ok(f); }
        let (t1, sf) = self.read(ids, true)?;
        f.prefill_s = sf.wall_s;
        f.stream = Some(sf);
        // the first token comes from the streamed read; the rest from the compact decode, which feeds t1 first
        f.gen = 1;
        f.last = Some(t1);
        if !on_tok(t1) { f.stop_reason = "cancelled".into(); return Ok(f); }
        if self.eng.cfg.eos.contains(&t1) || stop.contains(&t1) { f.stop_reason = "eos".into(); return Ok(f); }
        if n_new <= 1 { return Ok(f); }
        let td = Instant::now();
        let (st, spec) = if mtp && self.eng.opt.mtp {
            let (s, m) = self.eng.feed_stream_mtp(&[t1], n_new - 1, stop, on_tok)?;
            (s, Some((m.passes, m.accepted, m.rejects, m.acceptance)))
        } else if let (Some(c), false) = (&self.ctx, self.ctx_dry) {
            // context drafts: t1 is row 0 of the first verify pass
            let (s, cs) = self.eng.feed_stream_ctx(&[t1], n_new - 1, stop, on_tok, &mut c.borrow_mut())?;
            f.ctx_draft = Some(cs.json());
            (s, Some((cs.passes, cs.accepted, cs.rejects, cs.acceptance())))
        } else {
            (self.eng.feed_stream(&[t1], n_new - 1, stop, on_tok)?, None)
        };
        f.decode_s = td.elapsed().as_secs_f64();
        f.gen += st.gen;
        if st.last.is_some() { f.last = st.last; }
        f.stop_reason = st.stop_reason.to_string();
        f.spec = spec;
        Ok(f)
    }
}
