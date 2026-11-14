//! The generation worker: owns the engine (on the thread that opened it; QwenEngine is not Sync), takes one job
//! at a time from a channel (concurrent HTTP requests queue here), plans KV reuse, feeds, streams raw text pieces
//! back to the request's handler thread, and keeps the KV bookkeeping.
use crate::anthropic::ChatReq;
use crate::chatmap::{plan_ban_ids, render, ChatMsg, ToolDef};
use crate::planner::{KvState, Mode, Plan};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use streamdense_model::QwenEngine;
use streamdense_tokenizer::{StreamDecoder, Tokenizer, ENDOFTEXT, IM_END};

pub const STOP_IDS: [u32; 2] = [IM_END, ENDOFTEXT];

/// What one feed + decode returned (engine-independent copy of StreamStats + MTP counters).
#[derive(Clone, Debug, Default)]
pub struct FeedStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub gen: usize,
    pub last: Option<u32>,
    /// eos | length | cancelled | ctx_full
    pub stop_reason: String,
    /// (passes, accepted, rejects, acceptance) of an MTP run.
    pub spec: Option<(usize, usize, usize, f64)>,
    /// 12 GB mode (compact.rs): the streamed prompt read of this feed.
    pub stream: Option<StreamFeed>,
    /// 12 GB mode context drafts (engine ctx_draft.rs `CtxStats::json`): drafted / accepted / tokens per verify pass.
    pub ctx_draft: Option<serde_json::Value>,
}

/// 12 GB mode: one feed's streamed prompt read (sum over its pieces).
#[derive(Clone, Debug, Default)]
pub struct StreamFeed {
    /// streamed passes (one per piece of <= stream_tokens tokens)
    pub pieces: usize,
    pub tokens: usize,
    /// bytes read from the model file and the loader's busy time
    pub ssd_bytes: u64,
    pub ssd_busy_s: f64,
    /// wall time of the prompt read (= the prefill)
    pub wall_s: f64,
    pub gpu_s: f64,
    pub load_wait_s: f64,
    pub policy_s: f64,
    pub fill_s: f64,
    /// neurons (all layers) that changed against the set loaded before
    pub changed: usize,
    pub host_reload_bytes: u64,
}
impl StreamFeed {
    pub fn add(&mut self, o: &StreamFeed) {
        self.pieces += o.pieces; self.tokens += o.tokens; self.ssd_bytes += o.ssd_bytes; self.ssd_busy_s += o.ssd_busy_s; self.wall_s += o.wall_s;
        self.gpu_s += o.gpu_s; self.load_wait_s += o.load_wait_s; self.policy_s += o.policy_s; self.fill_s += o.fill_s; self.changed += o.changed;
        self.host_reload_bytes += o.host_reload_bytes;
    }
}

/// Totals of the DFlash2 option since the gateway started (`dflash2_stats_json`).
#[derive(Default)]
struct Df2Totals {
    requests: usize,
    errors: usize,
    last_error: Option<String>,
    last: Option<serde_json::Value>,
    total: streamdense_model::dflash2::Df2Stats,
}
static DFLASH2_TOTALS: Mutex<Option<Df2Totals>> = Mutex::new(None);

/// Live DFlash2 stats for /health and the request log: requests, errors, last error, last request's and summed stats
/// (`Df2Stats::json`, the same JSON as the drafter's).
pub fn dflash2_stats_json() -> serde_json::Value {
    let g = DFLASH2_TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_ref() {
        Some(t) => serde_json::json!({"requests": t.requests, "errors": t.errors, "last_error": t.last_error, "last": t.last, "total": t.total.json()}),
        None => serde_json::json!({"requests": 0, "errors": 0}),
    }
}

/// The engine as the gateway drives it. QwenEngine implements it; tests can plug in a fake.
pub trait Backend {
    fn reset(&self);
    /// Position of the next fed token (tokens in the KV, without the pending one).
    fn pos(&self) -> usize;
    fn max_ctx(&self) -> usize;
    fn vocab(&self) -> usize;
    fn mtp_available(&self) -> bool;
    /// Feed `ids` from the current position, then decode up to `n_new` tokens (stop on eos / `stop`), calling
    /// `on_tok` per token; `on_tok` returns false to cancel. The last token stays pending (not fed).
    fn feed(&self, ids: &[u32], n_new: usize, stop: &[u32], mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String>;
    /// Activity recorder handle (thread-safe; GET /v1/streamdense/activity). None = no recorder.
    fn activity(&self) -> Option<std::sync::Arc<streamdense_model::activity::ActivityTap>> { None }
    /// Answer library: the prompt / answer boundary of the current job (called once, at its first generated token).
    fn mark_prompt(&self) {}
    /// RAM mode of this backend for /health and the request log (None = the normal full model; 12 GB mode: compact.rs).
    fn mode_info(&self) -> Option<serde_json::Value> { None }
    /// Peak footprint tracker of the process (the CLI's footprint monitor), shown with the RAM mode. None = not shown.
    fn peak_footprint(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicU64>> { None }
    /// Snapshot cut points are rounded down to a multiple of this many tokens after the feed's start (12 GB mode: the
    /// streamed read's chunk alignment, so a request with cuts reads in the same chunks as one without). 1 = exact.
    fn cut_align(&self) -> usize { 1 }
    /// Feed `ids` up to a snapshot cut point; the one generated token is discarded (the next segment starts with the
    /// token at the cut). Default: a 1-token `feed`. 12 GB mode: a read that keeps the set (compact.rs).
    fn feed_cut(&self, ids: &[u32], mtp: bool) -> Result<FeedStats, String> { self.feed(ids, 1, &STOP_IDS, mtp, &mut |_| true) }
    /// A warm-up (prefill-only request) feeds in pieces of this many tokens and checks its cancel flag between them,
    /// so a real request (any conversation) waits at most one piece. A multiple of `cut_align` (and of the
    /// engine's prefill chunk rows, so the pieces split no chunk).
    fn warm_chunk(&self) -> usize { WARM_CHUNK }
    /// Decode slots for batched decoding (engine batch.rs, `--seqs N`): > 1 = the worker runs the batching scheduler
    /// (`Worker::run_batched`): up to this many requests decode together. 1 = one job at a time (today).
    fn seqs(&self) -> usize { 1 }
    /// Make slot `i` the state every following call works on (reset, pos, feed, snapshots, ...).
    fn select(&self, _i: usize) {}
    /// Batched mode: start a decode in slot `slot` (selected): feed `ids` from the slot's position (MTP warm-up), emit
    /// the first token (`on_tok`). The sequence may be done at once (eos, length 1, cancel).
    fn dec_begin(&self, _slot: usize, _ids: &[u32], _n_new: usize, _stop: &[u32], _on_tok: &mut dyn FnMut(u32) -> bool) -> Result<DecSeq, String> {
        Err("this backend has no batched decode".into())
    }
    /// Batched mode: decode `s` alone until it is done or `yield_now()` is true at a pass boundary.
    fn dec_solo(&self, _s: &mut DecSeq, _on_tok: &mut dyn FnMut(u32) -> bool, _yield_now: &dyn Fn() -> bool) -> Result<(), String> {
        Err("this backend has no batched decode".into())
    }
    /// Batched mode: one shared pass over the active sequences of `s`; `on_tok(i, t)` per token of `s[i]`.
    fn dec_round(&self, _s: &mut [&mut DecSeq], _on_tok: &mut dyn FnMut(usize, u32) -> bool) -> Result<(), String> {
        Err("this backend has no batched decode".into())
    }
    /// Ban `ids` on the selected slot before the argmax of every following pass (empty = none). true = done on the
    /// engine (lossless: a banned id is never the argmax); false = not supported, the worker stops at the token instead.
    fn set_ban(&self, _ids: &[u32]) -> bool { false }
}

/// One sequence of the batched scheduler (`Backend::dec_begin`): its slot, and once done its stop reason + stats.
pub struct DecSeq {
    pub slot: usize,
    /// Some once the sequence is finished (`stats` then final)
    pub done: Option<String>,
    pub stats: FeedStats,
    /// the backend's own sequence state (engine: batch.rs `BatchSeq`)
    pub inner: Box<dyn std::any::Any>,
}

/// Target tokens per warm-up piece (`Backend::warm_chunk`; rounded to the engine's chunk alignment, at least one
/// aligned chunk). 48 = ~0.5 s at the ~90-110 tok/s 27B prefill; with prefill_t 128 / prefill_k 8 the piece is one
/// 128-row chunk (~1.2-1.4 s), the floor that keeps the warm-up's chunks == a one-shot prefill's (lossless).
pub const WARM_CHUNK: usize = 48;

/// A queued warm-up older than this is dropped before it starts (`prune_warm`; `Worker::warm_max_age`).
pub const WARM_MAX_AGE: Duration = Duration::from_secs(30);

/// Same warm-up slot: the same conversation, or the same project + lane (`ChatReq::stable_tag`).
fn same_warm_slot(a: &Job, b: &Job) -> bool {
    a.conv == b.conv || matches!((a.req.stable_tag(), b.req.stable_tag()), (Some(x), Some(y)) if x.project == y.project && x.plan == y.plan)
}

/// Queued warm-ups to drop before the next pick (queue order = arrival order), with the reason: "cancelled" (its
/// cancel flag is set: the client closed the connection, or a real request of its conversation was picked), "stale"
/// (queued longer than `max_age`), "superseded" (a newer queued warm-up of the same conversation, or the same project
/// + lane). Real requests are never dropped.
pub(crate) fn prune_warm(q: &mut VecDeque<Job>, max_age: Duration, now: Instant) -> Vec<(Job, &'static str)> {
    let mut out = vec![];
    let mut i = 0;
    while i < q.len() {
        let j = &q[i];
        let why = if !j.warm { None }
            else if j.cancel.load(Ordering::SeqCst) { Some("cancelled") }
            else if now.saturating_duration_since(j.t_recv) > max_age { Some("stale") }
            else if q.iter().skip(i + 1).any(|n| n.warm && same_warm_slot(j, n)) { Some("superseded") }
            else { None };
        match why {
            Some(w) => { if let Some(j) = q.remove(i) { out.push((j, w)); } }
            None => i += 1,
        }
    }
    out
}

impl Backend for QwenEngine {
    fn reset(&self) { QwenEngine::reset(self) }
    fn pos(&self) -> usize { QwenEngine::pos(self) }
    fn max_ctx(&self) -> usize { self.opt.max_ctx }
    fn vocab(&self) -> usize { self.cfg.vocab }
    /// speculative decode available: MTP, or a DFlash / DFlash2 drafter (dflash.rs, dflash2.rs)
    fn mtp_available(&self) -> bool { self.opt.mtp || self.dflash_on() || self.dflash2_on() }
    fn activity(&self) -> Option<std::sync::Arc<streamdense_model::activity::ActivityTap>> { self.activity_tap() }
    fn mark_prompt(&self) { if let Some(t) = self.activity_tap() { t.mark_prompt(); } }
    /// whole fast-prefill (prefill_t) and exact (prefill_k) chunks: the pieces split no chunk
    fn warm_chunk(&self) -> usize {
        let (t, k) = (self.opt.prefill_t.max(1), self.opt.prefill_k.max(1));
        let (mut a, mut b) = (t, k);
        while b != 0 { (a, b) = (b, a % b); }
        let al = t / a * k;
        (WARM_CHUNK / al).max(1) * al
    }
    fn feed(&self, ids: &[u32], n_new: usize, stop: &[u32], mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> {
        // emitted tokens go to the recorder's output-vocab log (CPU only; no effect on decoding)
        let tap = self.activity_tap();
        let mut logged = |t: u32| { if let Some(a) = &tap { a.log_out(&[t]); } on_tok(t) };
        let on_tok: &mut dyn FnMut(u32) -> bool = &mut logged;
        let conv = |s: &streamdense_model::qwen_engine::StreamStats| FeedStats { prefill_s: s.prefill_s, decode_s: s.decode_s, gen: s.gen,
            last: s.last, stop_reason: s.stop_reason.to_string(), spec: None, stream: None, ctx_draft: None };
        if mtp && self.dflash2_on() {
            // DFlash2 block drafts (engine dflash2.rs). Fail loud: an error ends this request (`finish`), there is no
            // fallback to another decoding mode
            let r = self.feed_stream_dflash2(ids, n_new, stop, on_tok);
            let mut g = DFLASH2_TOTALS.lock().unwrap_or_else(|e| e.into_inner());
            let t = g.get_or_insert_with(Df2Totals::default);
            t.requests += 1;
            let (s, m) = match r {
                Ok(x) => x,
                Err(e) => {
                    t.errors += 1;
                    t.last_error = Some(e.clone());
                    return Err(format!("DFlash2 decode failed, request ended (no fallback to another decoding mode): {}", e));
                }
            };
            t.last = Some(m.json());
            t.total.add(&m);
            drop(g);
            if std::env::var("SD_DFLASH_LOG").map(|v| v == "1").unwrap_or(false) { eprintln!("dflash2_stats {}", m.json()); }
            let mut f = conv(&s);
            f.spec = Some((m.passes, m.accepted, m.rejects, m.acceptance()));
            Ok(f)
        } else if mtp && self.dflash_on() {
            // DFlash block drafts (engine dflash.rs): one exact verify pass per block, output == plain greedy
            let (s, m) = self.feed_stream_dflash(ids, n_new, stop, on_tok)?;
            // per-request drafter stats (draft / verify ms, accept histogram, short_ctx) on stderr, opt-in
            if std::env::var("SD_DFLASH_LOG").map(|v| v == "1").unwrap_or(false) { eprintln!("dflash_stats {}", m.json()); }
            let mut f = conv(&s);
            f.spec = Some((m.passes, m.accepted, m.rejects, m.acceptance()));
            Ok(f)
        } else if mtp && self.opt.mtp {
            let (s, m) = self.feed_stream_mtp(ids, n_new, stop, on_tok)?;
            let mut f = conv(&s);
            f.spec = Some((m.passes, m.accepted, m.rejects, m.acceptance));
            Ok(f)
        } else {
            Ok(conv(&self.feed_stream(ids, n_new, stop, on_tok)?))
        }
    }
    /// batching needs the MTP deep spec loop (spec_depth >= 2 or auto) and no DFlash drafter
    fn seqs(&self) -> usize {
        if self.opt.mtp && !self.dflash_on() && self.spec_mode().max_depth() >= 2 { self.n_seqs() } else { 1 }
    }
    fn select(&self, i: usize) { self.select_seq(i) }
    fn set_ban(&self, ids: &[u32]) -> bool { QwenEngine::set_ban(self, ids).is_ok() }
    fn dec_begin(&self, slot: usize, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool) -> Result<DecSeq, String> {
        let tap = self.activity_tap();
        let mut logged = |t: u32| { if let Some(a) = &tap { a.log_out(&[t]); } on_tok(t) };
        let s = self.batch_begin(slot, ids, n_new, stop, &streamdense_model::spec::DraftOverride::None, &mut logged)?;
        let mut d = DecSeq { slot, done: None, stats: FeedStats::default(), inner: Box::new(s) };
        self.dec_sync(&mut d);
        Ok(d)
    }
    fn dec_solo(&self, s: &mut DecSeq, on_tok: &mut dyn FnMut(u32) -> bool, yield_now: &dyn Fn() -> bool) -> Result<(), String> {
        let tap = self.activity_tap();
        let mut logged = |t: u32| { if let Some(a) = &tap { a.log_out(&[t]); } on_tok(t) };
        let b = s.inner.downcast_mut::<streamdense_model::batch::BatchSeq>().ok_or("dec_solo: not an engine sequence")?;
        let r = self.solo_run(b, &mut logged, yield_now);
        self.dec_sync(s);
        r
    }
    fn dec_round(&self, s: &mut [&mut DecSeq], on_tok: &mut dyn FnMut(usize, u32) -> bool) -> Result<(), String> {
        let tap = self.activity_tap();
        let mut logged = |i: usize, t: u32| { if let Some(a) = &tap { a.log_out(&[t]); } on_tok(i, t) };
        let mut bs: Vec<&mut streamdense_model::batch::BatchSeq> = Vec::with_capacity(s.len());
        for d in s.iter_mut() { bs.push(d.inner.downcast_mut().ok_or("dec_round: not an engine sequence")?); }
        let r = self.batch_round(&mut bs, &mut logged);
        drop(bs);
        for d in s.iter_mut() { self.dec_sync(d); }
        r
    }
}

/// Engine side of `DecSeq`: copy the stop reason and stats out of the engine's `BatchSeq` once it is done.
trait DecSync { fn dec_sync(&self, d: &mut DecSeq); }
impl DecSync for QwenEngine {
    fn dec_sync(&self, d: &mut DecSeq) {
        if d.done.is_some() { return; }
        let Some(b) = d.inner.downcast_ref::<streamdense_model::batch::BatchSeq>() else { return };
        let Some(w) = b.done else { return };
        let (st, r) = (&b.st, &b.r);
        d.stats = FeedStats { prefill_s: st.prefill_s, decode_s: st.decode_s, gen: st.gen, last: st.last, stop_reason: w.to_string(),
                              spec: Some((r.passes, r.accepted, r.rejects, r.acceptance)), stream: None, ctx_draft: None };
        d.done = Some(w.to_string());
        self.flush_draft_source();
    }
}

/// One snapshot operation of a request (logged in the request log under "snap").
#[derive(Clone, Debug, Default)]
pub struct SnapEvent {
    /// restore | save_stable | save_roll | save_switch | hot_stable (a warm-up's stable prefix copied to RAM)
    pub kind: &'static str,
    /// Tokens covered (engine position of the state).
    pub n: usize,
    pub ms: f64,
    pub bytes: u64,
    pub error: Option<String>,
    /// restore: the tier it came from ("hot" RAM copy | "warm" file; snap_tier.rs); "" otherwise
    pub tier: &'static str,
    /// restore: "own" (the request's conversation) | "shared" (the project prefix: system prompt + tools); "" otherwise
    pub src: &'static str,
}

/// Engine state snapshots (streamdense-model snapshot.rs; `snap::CacheHook` is the real one). The worker drives the
/// policy: before a full prefill it asks `restore` for the longest cached prefix; while prefilling it stops at the
/// stable prefix (end of the system turn: system prompt + tool defs) and at the end of the prompt (the rolling
/// "last turn" state) and calls `save`. `NoSnapshots` (tests, `--no-snapshots`) does nothing.
pub trait SnapshotHook<B: ?Sized> {
    fn enabled(&self) -> bool { false }
    /// Restore the state covering the longest cached prefix of `tokens` (callers pass the prompt without its last
    /// token). Some(ev) with ev.error None => eng.pos() == ev.n; an error leaves the engine undefined (reset it).
    fn restore(&mut self, _eng: &B, _tokens: &[u32]) -> Option<SnapEvent> { None }
    /// A snapshot of exactly these tokens is cached already.
    fn has(&self, _eng: &B, _tokens: &[u32]) -> bool { false }
    /// Save the current state; `tokens` = every fed id (len == eng.pos()). `stable`: the shared prefix (kept);
    /// otherwise the rolling last-turn snapshot (replaces the previous one). `tag`: the request's project + lane
    /// (`ChatReq::stable_tag`; a stable prefix supersedes the older one of the same tag).
    fn save(&mut self, _eng: &B, _tokens: &[u32], _stable: bool, _tag: Option<crate::snap_tier::StableTag>) -> Option<SnapEvent> { None }
    /// A warm-up just saved the stable prefix `tokens`: put it in the hot (RAM) tier now, so the next request of that
    /// system turn (e.g. a plan call) restores it from RAM. None = nothing done.
    fn stable_hot(&mut self, _eng: &B, _tokens: &[u32]) -> Option<SnapEvent> { None }
    /// Multi-user: a request for `next` (another conversation) is about to replace the live state `live` (len ==
    /// eng.pos()); save it first if that is worth it (snap.rs `switch`). Called before the reset of a full prefill.
    fn switch(&mut self, _eng: &B, _live: &[u32], _next: &[u32]) -> Option<SnapEvent> { None }
    /// Snapshot tiers (snap_tier.rs): pending-request counter + status shared with the HTTP threads (a queued request
    /// stops idle work at once; /health shows the status). None = no tier manager.
    fn tiers(&self) -> Option<Arc<crate::snap_tier::TierShared>> { None }
    /// A job left the queue (`t_recv`: when it arrived). Returns the request-log "tiers" fields (idle work since the last job).
    fn job_begin(&mut self, _t_recv: Instant) -> Option<serde_json::Value> { None }
    /// The job is done; `fed` = the ids in the engine (len == pos) when the KV bookkeeping is exact, else None.
    fn job_end(&mut self, _eng: &B, _fed: Option<&[u32]>) {}
    /// How long the worker may wait for a job before calling `idle` (None = block: no idle work left).
    fn idle_wait(&self) -> Option<Duration> { None }
    /// One step of idle work (worker thread, nothing queued or running); `fed` as in `job_end`.
    fn idle(&mut self, _eng: &B, _fed: Option<&[u32]>) {}
}
pub struct NoSnapshots;
impl<B: ?Sized> SnapshotHook<B> for NoSnapshots {}

/// The rolling snapshot is taken only when a request prefills at least this many tokens (short feeds are cheap to
/// redo; each save writes the whole state, ~150 MB + 69 KB/token on the 27B).
pub const ROLL_MIN_FEED: usize = 256;
/// The stable prefix must be at least this long to be worth a file.
pub const STABLE_MIN: usize = 64;

/// End of the stable prefix: the tokens through the first `<|im_end|>` (the system turn: system prompt + tools).
pub fn stable_prefix_len(prompt: &[u32]) -> Option<usize> {
    let i = prompt.iter().position(|&t| t == IM_END)?;
    if i + 1 >= STABLE_MIN && i + 2 < prompt.len() { Some(i + 1) } else { None }
}

/// Start of the per-project part of a client's planner system turn. Everything before it (instructions + START + TOOLS,
/// ~603 tokens) is byte-identical for every project; the facts after it (files, record shapes, outline) change after
/// edits. The worker cuts a second stable snapshot right before it (`static_prefix_len`, lane `plan-static`), so a
/// facts change re-prefills only from here on. Searched as tokens: `PROJECT_FACTS_WORDS` (the newline run in front is
/// its own pre-token and may merge with newlines before it, so it is matched by decoding instead).
pub const PROJECT_FACTS_MARKER: &str = "\n\nPROJECT FACTS";
/// The words of `PROJECT_FACTS_MARKER` (tokenized alone they give the same ids as in context: the newline run before
/// and the " (" after are separate pre-tokens).
pub const PROJECT_FACTS_WORDS: &str = "PROJECT FACTS";

/// End of the static part of a planner system turn: the tokens before the `PROJECT_FACTS_MARKER` inside the system
/// turn (before the first `<|im_end|>`). `words` = the ids of `PROJECT_FACTS_WORDS`; `nl(t)` = token t decodes to
/// whitespace only. The cut goes before the whole whitespace run in front of the words, which must hold >= 2
/// newlines (`n_nl`). None when there is no marker or the static part is shorter than `STABLE_MIN`.
pub fn static_prefix_len(prompt: &[u32], words: &[u32], ws: impl Fn(u32) -> Option<usize>) -> Option<usize> {
    if words.is_empty() { return None; }
    let end = prompt.iter().position(|&t| t == IM_END)?;
    let sys = &prompt[..end];
    let p = sys.windows(words.len()).position(|w| w == words)?;
    // walk back over the whitespace tokens before the words, counting their newlines
    let (mut c, mut nls) = (p, 0);
    while c > 0 { match ws(sys[c - 1]) { Some(n) => { nls += n; c -= 1; } None => break } }
    if nls >= 2 && c >= STABLE_MIN { Some(c) } else { None }
}

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// Honor the client's thinking request (`--think`); otherwise thinking is always off (faster).
    pub think: bool,
    /// Decode with the MTP head when the engine has it (default on).
    pub mtp: bool,
    /// Tests only: feed ids modulo this (tiny model + the 27B tokenizer).
    pub fold: Option<u32>,
}

pub struct Job {
    pub req: ChatReq,
    /// Warm-up: prefill the prompt (rendered without the generation prompt), save its snapshots, decode nothing.
    /// Lowest priority: queued real requests run first (`Worker::run`).
    pub warm: bool,
    /// Conversation key (`ChatReq::conv_key`): a real request cancels a running warm-up; if every canceller is the
    /// warm-up's own conversation, the prefilled part stays in the live KV (`WarmGate::keep_partial`).
    pub conv: u64,
    /// the reply's message id (`msg_…`, sent to the client in message_start / the JSON reply)
    pub id: String,
    pub events: Sender<Ev>,
    pub cancel: Arc<AtomicBool>,
    pub t_recv: Instant,
}

#[derive(Clone, Debug)]
pub enum Ev {
    /// Planned: the prompt is n_prompt tokens, n_reused of them already in the KV.
    Start { n_prompt: usize, n_reused: usize, n_feed: usize, mode: &'static str, think: bool },
    /// Raw decoded text (complete UTF-8; tags included; stop tokens excluded).
    Text(String),
    Done(Outcome),
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// the message id of this reply (= Job.id; the watcher maps votes to the domain of the job)
    pub msg_id: String,
    /// Raw reply text without the stop token (starts inside `<think>` when thinking was on).
    pub raw: String,
    pub n_prompt: usize,
    pub n_reused: usize,
    pub n_feed: usize,
    pub mode: String,
    pub think: bool,
    /// Generated tokens incl. a final stop token.
    pub n_new: usize,
    /// eos | length | cancelled | ctx_full | error
    /// the stop sequence that ended the reply (stop_reason "stop_sequence")
    pub stop_sequence: Option<String>,
    pub stop_reason: String,
    pub error: Option<String>,
    pub queue_s: f64,
    pub tokenize_s: f64,
    pub ttft_s: Option<f64>,
    pub prefill_s: f64,
    pub decode_tok_s: Option<f64>,
    pub mtp: bool,
    pub spec: Option<(usize, usize, usize, f64)>,
    pub max_ctx: usize,
    /// Snapshot restore / saves of this request.
    pub snaps: Vec<SnapEvent>,
    /// 12 GB mode: the streamed prompt read of this request (sum over its feeds).
    pub stream: Option<StreamFeed>,
    /// 12 GB mode context drafts of this request's decode (`FeedStats::ctx_draft`).
    pub ctx_draft: Option<serde_json::Value>,
    /// Snapshot tiers: idle work done since the previous job (`SnapshotHook::job_begin`).
    pub tiers: Option<serde_json::Value>,
    /// A warm-up (prefill only): stop_reason "warm" when the prompt is in the KV, "cancelled" otherwise.
    pub warm: bool,
    /// A queued warm-up dropped before it started (`prune_warm`): cancelled | stale | superseded (mode "dropped").
    pub drop_reason: Option<&'static str>,
}

/// The running warm-up (conversation key + cancel flag + keep), shared by the worker and the HTTP threads. The worker
/// sets it under the lock while it drains the queue and picks the next job; an HTTP thread calls `real_arrived` after
/// queueing a real request, so the request is either seen by the worker's pick or cancels the warm-up it started.
#[derive(Default)]
pub struct WarmGate { running: Mutex<Option<RunningWarm>> }
struct RunningWarm { conv: u64, lane: u64, cancel: Arc<AtomicBool>, keep: bool }
impl WarmGate {
    /// A real request for conversation `conv` was queued: cancel the running warm-up (any conversation; it stops at
    /// its next piece). Returns whether one was cancelled. `keep` = every canceller so far is the warm-up's own
    /// conversation (its prefilled part is then the prefix that request reuses).
    /// `lane` (`ChatReq::lane_key`): same system turn + tools also keeps the partial KV (the plan lane's warm-up has a
    /// placeholder user turn, so its conv key never equals the real plan call's; its fed part is that call's prefix).
    pub fn real_arrived(&self, conv: u64, lane: u64) -> bool {
        let Ok(mut g) = self.running.lock() else { return false };
        match g.as_mut() {
            Some(r) => {
                let same = r.conv == conv || r.lane == lane;
                if !r.cancel.swap(true, Ordering::SeqCst) { r.keep = same; } else if !same { r.keep = false; }
                true
            }
            None => false,
        }
    }
    /// The cancelled running warm-up's prefilled part should stay in the live KV (see `real_arrived`).
    pub fn keep_partial(&self) -> bool {
        self.running.lock().map(|g| g.as_ref().is_some_and(|r| r.keep)).unwrap_or(false)
    }
    fn set(g: &mut Option<RunningWarm>, j: &Job) { *g = Some(RunningWarm { conv: j.conv, lane: j.req.lane_key(), cancel: j.cancel.clone(), keep: false }); }
}

pub struct Worker<'a, B: Backend + ?Sized> {
    pub eng: &'a B,
    pub tk: Arc<Tokenizer>,
    pub cfg: WorkerConfig,
    pub kv: KvState,
    /// Multi-user: the KV bookkeeping of conversations switched away from whose state is in a snapshot file (newest
    /// last, at most `PARKED_MAX`). A request continuing one restores that file and plans prefix / followup against it,
    /// exactly as if that user had been alone (the model's own reply tokens stay, as in `followup`).
    pub parked: Vec<KvState>,
    pub hook: Box<dyn SnapshotHook<B> + 'a>,
    /// Per-job hook (the watcher), run after the Done event went out.
    pub on_job: Option<crate::JobHook<'a, B>>,
    /// The running warm-up (shared with the HTTP threads).
    pub gate: Arc<WarmGate>,
    /// Queued warm-ups older than this are dropped (default `WARM_MAX_AGE`).
    pub warm_max_age: Duration,
}

impl<'a, B: Backend + ?Sized> Worker<'a, B> {
    pub fn new(eng: &'a B, tk: Arc<Tokenizer>, cfg: WorkerConfig) -> Self {
        Worker { eng, tk, cfg, kv: KvState::default(), parked: vec![], hook: Box::new(NoSnapshots), on_job: None, gate: Arc::new(WarmGate::default()), warm_max_age: WARM_MAX_AGE }
    }

    /// Serve jobs until every sender is gone. Between jobs the snapshot hook gets idle steps (`idle_wait` / `idle`).
    /// Priority: every queued job is taken first; real requests run in arrival order, warm-ups only when no real
    /// request waits. Picking a real request skips the queued warm-ups of its own conversation (it prefills that
    /// prefix itself).
    pub fn run(&mut self, rx: Receiver<Job>) {
        if self.eng.seqs() > 1 && self.cfg.mtp && self.eng.mtp_available() { return self.run_batched(rx); }
        let mut q: VecDeque<Job> = VecDeque::new();
        loop {
            let gate = self.gate.clone();
            let (job, dropped) = {
                let mut g = gate.running.lock().unwrap_or_else(|e| e.into_inner());
                while let Ok(j) = rx.try_recv() { q.push_back(j); }
                let dropped = prune_warm(&mut q, self.warm_max_age, Instant::now());
                let pick = q.iter().position(|j| !j.warm).or(if q.is_empty() { None } else { Some(0) });
                let job = pick.and_then(|i| q.remove(i));
                if let Some(j) = &job {
                    if j.warm { WarmGate::set(&mut g, j); }
                    else { for w in q.iter().filter(|w| w.warm && w.conv == j.conv) { w.cancel.store(true, Ordering::SeqCst); } }
                }
                (job, dropped)
            };
            for (d, why) in dropped { self.drop_warm(d, why); }
            let Some(job) = job else {
                let j = match self.hook.idle_wait() {
                    None => match rx.recv() { Ok(j) => j, Err(_) => return },
                    Some(d) => match rx.recv_timeout(d) {
                        Ok(j) => j,
                        Err(RecvTimeoutError::Timeout) => { let fed = fed_ids(&self.kv, self.eng.pos()); self.hook.idle(self.eng, fed); continue; }
                        Err(RecvTimeoutError::Disconnected) => return,
                    },
                };
                q.push_back(j);
                continue;
            };
            let warm = job.warm;
            self.one(job);
            if warm { if let Ok(mut g) = gate.running.lock() { *g = None; } }
        }
    }

    /// Batched serving (`Backend::seqs() > 1`, engine batch.rs): up to `seqs` real requests decode together, each in its
    /// own engine slot (KV, DeltaNet, MTP state) with its own KV bookkeeping, so every reply is token-identical to
    /// serving it alone. Between passes: every queued job is taken, one queued real request is admitted into a free slot
    /// (its prefill runs alone: the others pause for it), then one shared pass decodes all active sequences
    /// (`dec_round`; a lone sequence runs `dec_solo`, the 1x8 loop of today, yielding at the next pass boundary when a
    /// real request arrives). Warm-ups run only when no sequence is active and no real request waits (as in `run`).
    /// Slot choice: the slot that last served the same conversation (its KV is reused), else an empty one, else the
    /// least recently used (the snapshot hook's switch save + parked list cover the state it replaces).
    pub fn run_batched(&mut self, rx: Receiver<Job>) {
        let n = self.eng.seqs();
        let q: std::cell::RefCell<VecDeque<Job>> = std::cell::RefCell::new(VecDeque::new());
        // per-slot bookkeeping; the selected slot's lives in self.kv (slot_kv[cur] is a placeholder then)
        let mut slot_kv: Vec<KvState> = vec![KvState::default(); n];
        let mut slot_conv: Vec<Option<u64>> = vec![None; n];
        let mut last_use: Vec<u64> = vec![0; n];
        let mut cur = 0usize;
        let mut tick = 0u64;
        self.eng.select(0);
        let mut act: Vec<Active> = Vec::with_capacity(n);
        let mut gone = false;
        loop {
            {
                let mut qq = q.borrow_mut();
                loop { match rx.try_recv() { Ok(j) => qq.push_back(j), Err(std::sync::mpsc::TryRecvError::Disconnected) => { gone = true; break; } Err(_) => break } }
            }
            let dropped = prune_warm(&mut q.borrow_mut(), self.warm_max_age, Instant::now());
            for (d, why) in dropped { self.drop_warm(d, why); }
            // admit one queued real request into a free slot
            let real = if act.len() < n { let mut qq = q.borrow_mut(); qq.iter().position(|j| !j.warm).and_then(|i| qq.remove(i)) } else { None };
            if let Some(job) = real {
                for w in q.borrow().iter().filter(|w| w.warm && w.conv == job.conv) { w.cancel.store(true, Ordering::SeqCst); }
                let busy: Vec<usize> = act.iter().map(|a| a.d.slot).collect();
                let free = (0..n).filter(|i| !busy.contains(i));
                let slot = free.clone().find(|&i| slot_conv[i] == Some(job.conv))
                    .or_else(|| free.clone().find(|&i| slot_conv[i].is_none()))
                    .or_else(|| free.min_by_key(|&i| last_use[i])).expect("a free slot");
                self.use_slot(slot, &mut cur, &mut slot_kv);
                tick += 1; last_use[slot] = tick; slot_conv[slot] = Some(job.conv);
                let tiers = self.hook.job_begin(job.t_recv);
                match self.prepare(&job) {
                    Err(o) => self.done(&job, o, tiers),
                    Ok(mut p) => {
                        let fed = std::mem::take(&mut p.fed);
                        let (eng, tk) = (self.eng, self.tk.clone());
                        p.arm_ban(eng);
                        let r = eng.dec_begin(slot, &fed, p.nnew, &STOP_IDS, &mut |t| p.on_tok(&job, eng, &tk, t));
                        match r {
                            Ok(d) => act.push(Active { job, p, d, tiers }),
                            Err(e) => { let o = self.finish(&job, p, Err(e)); self.done(&job, o, tiers); }
                        }
                    }
                }
                self.reap(&mut act, &mut cur, &mut slot_kv);
                continue; // take the next queued request before the next pass
            }
            if act.is_empty() {
                // nothing decoding, no real request waiting: a warm-up, else idle work / wait
                let warm = q.borrow_mut().pop_front();
                if let Some(job) = warm {
                    let gate = self.gate.clone();
                    { let mut g = gate.running.lock().unwrap_or_else(|e| e.into_inner()); WarmGate::set(&mut g, &job); }
                    let slot = slot_conv.iter().position(|c| *c == Some(job.conv))
                        .or_else(|| slot_conv.iter().position(|c| c.is_none()))
                        .unwrap_or_else(|| (0..n).min_by_key(|&i| last_use[i]).unwrap_or(0));
                    self.use_slot(slot, &mut cur, &mut slot_kv);
                    tick += 1; last_use[slot] = tick; slot_conv[slot] = Some(job.conv);
                    self.one(job);
                    if let Ok(mut g) = gate.running.lock() { *g = None; }
                    continue;
                }
                if gone { return; }
                let j = match self.hook.idle_wait() {
                    None => match rx.recv() { Ok(j) => j, Err(_) => return },
                    Some(d) => match rx.recv_timeout(d) {
                        Ok(j) => j,
                        Err(RecvTimeoutError::Timeout) => { let fed = fed_ids(&self.kv, self.eng.pos()); self.hook.idle(self.eng, fed); continue; }
                        Err(RecvTimeoutError::Disconnected) => return,
                    },
                };
                q.borrow_mut().push_back(j);
                continue;
            }
            let (eng, tk) = (self.eng, self.tk.clone());
            let r = if act.len() == 1 {
                // alone: the plain loop; hand back at the next pass boundary once a real request waits
                let a = &mut act[0];
                let yield_now = || {
                    let mut qq = q.borrow_mut();
                    while let Ok(j) = rx.try_recv() { qq.push_back(j); }
                    qq.iter().any(|j| !j.warm)
                };
                let (job, p) = (&a.job, &mut a.p);
                eng.dec_solo(&mut a.d, &mut |t| p.on_tok(job, eng, &tk, t), &yield_now)
            } else {
                let mut ds: Vec<&mut DecSeq> = Vec::with_capacity(act.len());
                let mut ps: Vec<(&Job, &mut Prepared)> = Vec::with_capacity(act.len());
                for a in act.iter_mut() { ds.push(&mut a.d); ps.push((&a.job, &mut a.p)); }
                eng.dec_round(&mut ds, &mut |i, t| { let (job, p) = &mut ps[i]; p.on_tok(job, eng, &tk, t) })
            };
            for a in act.iter() { tick += 1; last_use[a.d.slot] = tick; }
            if let Err(e) = r {
                // the shared pass failed: every active sequence ends with the error (its slot reset by finish)
                for a in std::mem::take(&mut act) {
                    self.use_slot(a.d.slot, &mut cur, &mut slot_kv);
                    slot_conv[a.d.slot] = None;
                    let o = self.finish(&a.job, a.p, Err(e.clone()));
                    self.done(&a.job, o, a.tiers);
                }
                continue;
            }
            self.reap(&mut act, &mut cur, &mut slot_kv);
        }
    }

    /// Make engine slot `i` current: its KV bookkeeping into `self.kv` (the previous slot's back into `slot_kv`).
    fn use_slot(&mut self, i: usize, cur: &mut usize, slot_kv: &mut [KvState]) {
        if i != *cur { std::mem::swap(&mut self.kv, &mut slot_kv[*cur]); std::mem::swap(&mut self.kv, &mut slot_kv[i]); *cur = i; }
        self.eng.select(i);
    }

    /// Finish every sequence that is done: bookkeeping on its slot, Done event, per-job hooks.
    fn reap(&mut self, act: &mut Vec<Active>, cur: &mut usize, slot_kv: &mut [KvState]) {
        let mut i = 0;
        while i < act.len() {
            if act[i].d.done.is_none() { i += 1; continue; }
            let a = act.remove(i);
            self.use_slot(a.d.slot, cur, slot_kv);
            let st = a.d.stats;
            let o = self.finish(&a.job, a.p, Ok(st));
            self.done(&a.job, o, a.tiers);
        }
    }

    /// Send a job's Done (+ the per-job hook for real requests), then the snapshot hook's job_end (current slot).
    fn done(&mut self, job: &Job, mut out: Outcome, tiers: Option<serde_json::Value>) {
        out.tiers = tiers;
        out.msg_id = job.id.clone();
        match self.on_job.as_mut() {
            Some(h) if !job.warm => { let o = out.clone(); let _ = job.events.send(Ev::Done(out)); h(self.eng, &job.req, &o); }
            _ => { let _ = job.events.send(Ev::Done(out)); }
        }
        self.hook.job_end(self.eng, fed_ids(&self.kv, self.eng.pos()));
    }

    /// A queued warm-up dropped before it started (`prune_warm`): its client gets "cancelled" (mode "dropped",
    /// `drop_reason`); the snapshot hook sees an empty job (the pending-request count stays right).
    fn drop_warm(&mut self, job: Job, why: &'static str) {
        let tiers = self.hook.job_begin(job.t_recv);
        let o = Outcome { stop_reason: "cancelled".into(), mode: "dropped".into(), queue_s: job.t_recv.elapsed().as_secs_f64(), warm: true,
                          drop_reason: Some(why), tiers, msg_id: job.id.clone(), ..Default::default() };
        let _ = job.events.send(Ev::Done(o));
        self.hook.job_end(self.eng, fed_ids(&self.kv, self.eng.pos()));
    }

    pub fn one(&mut self, job: Job) {
        let tiers = self.hook.job_begin(job.t_recv);
        let mut out = self.generate(&job);
        out.tiers = tiers;
        out.msg_id = job.id.clone();
        match self.on_job.as_mut() {
            // a warm-up has no reply: the per-job hook (watcher, answer library) does not see it
            Some(h) if !job.warm => { let o = out.clone(); let _ = job.events.send(Ev::Done(out)); h(self.eng, &job.req, &o); }
            _ => { let _ = job.events.send(Ev::Done(out)); }
        }
        self.hook.job_end(self.eng, fed_ids(&self.kv, self.eng.pos()));
    }

    /// Plan + prefill a job up to its last feed (snapshot restore / cuts / saves, warm-ups whole). Err = the job is over
    /// already (skipped, error, ctx_full, a finished warm-up); Ok = feed `fed` + decode next, then `finish`.
    #[allow(clippy::result_large_err)] // Err is the finished reply (sent once per job), not an error path to propagate
    pub(crate) fn prepare(&mut self, job: &Job) -> Result<Prepared, Outcome> {
        let t0 = Instant::now();
        let req = &job.req;
        let warm = job.warm;
        if job.cancel.load(Ordering::SeqCst) {
            return Err(Outcome { stop_reason: "cancelled".into(), mode: "skipped".into(), queue_s: t0.duration_since(job.t_recv).as_secs_f64(), warm, ..Default::default() });
        }
        let think = self.cfg.think && req.thinking_requested;
        let max_ctx = self.eng.max_ctx();
        let mut o = Outcome { think, queue_s: t0.duration_since(job.t_recv).as_secs_f64(), max_ctx, warm, ..Default::default() };
        let fail = |mut o: Outcome, e: String| { o.stop_reason = "error".into(); o.error = Some(e); o };
        let tk = self.tk.clone();
        let (msgs, tools): (&[ChatMsg], &[ToolDef]) = (&req.msgs, &req.tools);
        // a warm-up renders the history without the generation prompt: it ends with "<|im_end|>\n", a token prefix
        // of the next request (the same messages + a new turn + the generation prompt)
        let text = match render(msgs, tools, think, !warm) { Ok(t) => t, Err(e) => return Err(fail(o, e)) };
        let full = tk.encode(&text);
        // the bookkeeping must describe the engine state exactly, else nothing is reused
        if self.kv.is_valid() && self.kv.seq.len() != self.eng.pos() + 1 {
            eprintln!("gateway: KV bookkeeping ({} tokens) != engine ({} + pending): full prefill", self.kv.seq.len(), self.eng.pos());
            self.kv.invalidate();
        }
        let rend = |m: &[ChatMsg], g: bool| render(m, tools, think, g);
        let enc = |s: &str| tk.encode(s);
        // followup assumes our reply is the next message and adds a generation prompt: never for a warm-up
        let downgrade = |p: Plan| if warm && p.mode == Mode::Followup { Plan { mode: Mode::Full, tokens: p.full.clone(), full: p.full } } else { p };
        let mut plan = downgrade(self.kv.plan(msgs, tools, think, full, &rend, &enc));
        let mut parked_hit = false;
        if plan.mode == Mode::Full && self.hook.enabled() {
            // multi-user: another conversation's live state is about to go -> file (switch save) + park its bookkeeping
            if let Some(live) = fed_ids(&self.kv, self.eng.pos()) {
                if let Some(ev) = self.hook.switch(self.eng, live, &plan.full) { o.snaps.push(ev); }
                if self.hook.has(self.eng, live) { park(&mut self.parked, self.kv.clone()); }
            }
            // a parked conversation this request continues: restore its file, then plan against its state
            let hit = self.parked.iter().rposition(|p| p.seq.len() >= 2 && self.hook.has(self.eng, &p.seq[..p.seq.len() - 1])
                && { let m = p.plan(msgs, tools, think, plan.full.clone(), &rend, &enc).mode; if warm { m == Mode::Prefix } else { m != Mode::Full } });
            if let Some(i) = hit {
                let p = self.parked.remove(i);
                let n = p.seq.len() - 1;
                self.eng.reset();
                self.kv.invalidate();
                if let Some(ev) = self.hook.restore(self.eng, &p.seq[..n]) {
                    let ok = ev.error.is_none() && ev.n == n && self.eng.pos() == n;
                    o.snaps.push(ev);
                    if ok {
                        self.kv = p;
                        plan = downgrade(self.kv.plan(msgs, tools, think, plan.full, &rend, &enc));
                        parked_hit = plan.mode != Mode::Full;
                    }
                }
                if !parked_hit { self.eng.reset(); self.kv.invalidate(); }
            }
        }
        o.tokenize_s = t0.elapsed().as_secs_f64();
        let (mut mode, mut toks) = (plan.mode, plan.tokens);
        let mut feed: Vec<u32> = vec![];
        if mode != Mode::Full {
            feed.push(*self.kv.seq.last().unwrap()); // the pending token goes in first
            feed.extend_from_slice(&toks);
            if self.eng.pos() + feed.len() + 1 > max_ctx { mode = Mode::Full; toks = plan.full.clone(); }
        }
        let n_prompt = plan.full.len();
        let mut reused = 0;
        // ids already in the engine (len == pos) before `feed`
        let mut base: Vec<u32> = vec![];
        let mut label = mode.as_str();
        if mode == Mode::Full {
            feed = toks.clone();
            if self.hook.enabled() && n_prompt >= 2 && n_prompt + 1 <= max_ctx {
                self.eng.reset();
                self.kv.invalidate(); // the engine state changes from here on
                if let Some(ev) = self.hook.restore(self.eng, &plan.full[..n_prompt - 1]) {
                    if ev.error.is_none() && ev.n > 0 && ev.n < n_prompt && self.eng.pos() == ev.n {
                        base = plan.full[..ev.n].to_vec();
                        feed = plan.full[ev.n..].to_vec();
                        reused = ev.n;
                        label = "restore";
                    } else {
                        self.eng.reset();
                    }
                    o.snaps.push(ev);
                }
            }
        } else {
            reused = self.eng.pos();
            base = self.kv.seq[..self.kv.seq.len() - 1].to_vec(); // feed[0] is the pending last token
            if parked_hit { label = "restore"; }
        }
        o.n_prompt = n_prompt; o.n_reused = reused; o.n_feed = feed.len(); o.mode = label.into();
        if n_prompt + 1 > max_ctx || reused + feed.len() + 1 > max_ctx {
            self.kv.invalidate();
            o.stop_reason = "ctx_full".into();
            return Err(o);
        }
        let _ = job.events.send(Ev::Start { n_prompt, n_reused: reused, n_feed: feed.len(), mode: label, think });
        if mode == Mode::Full && reused == 0 { self.eng.reset(); }
        let pos0 = base.len();
        // every id in the engine after this prefill (prefix / full / restore: == plan.full; followup: differs)
        let mut all = base;
        all.extend_from_slice(&feed);
        // snapshot cut points (absolute fed-token counts): the stable prefix (first full prefill of a system prompt),
        // then the end of the prompt minus its last token (rolling last-turn state) when this request prefills a lot
        // (cut, stable, save)
        let mut cuts: Vec<(usize, bool, bool)> = vec![];
        // the static planner cut (saved under the shared `plan-static` tag, no project)
        let mut static_cut: Option<usize> = None;
        if self.hook.enabled() && all == plan.full {
            // cut_align 1 (full model): unchanged; 12 GB mode: whole chunks after pos0
            let al = self.eng.cut_align().max(1);
            let align = |c: usize| if c > pos0 { pos0 + (c - pos0) / al * al } else { c };
            if mode == Mode::Full {
                if let Some(s) = stable_prefix_len(&plan.full).map(align) {
                    if s > pos0 && !self.hook.has(self.eng, &plan.full[..s]) { cuts.push((s, true, true)); }
                    // the static part of a planner turn (before the project facts): shared by every project
                    let words = tk.encode(PROJECT_FACTS_WORDS);
                    let ws = |t: u32| { let d = tk.decode(&[t]); if !d.is_empty() && d.chars().all(char::is_whitespace) { Some(d.matches('\n').count()) } else { None } };
                    if let Some(f) = static_prefix_len(&plan.full, &words, ws).map(align) {
                        if f >= STABLE_MIN && f > pos0 && f < s && !self.hook.has(self.eng, &plan.full[..f]) { cuts.insert(0, (f, true, true)); static_cut = Some(f); }
                    }
                }
            }
            let l = align(all.len() - 1);
            if l >= pos0 + ROLL_MIN_FEED && cuts.last().map_or(true, |c| l > c.0) { cuts.push((l, false, true)); }
        }
        // a warm-up feeds the whole prompt but its last token (left pending, as after a reply), saving nothing more
        let end = all.len() - 1;
        if warm && end > pos0 && cuts.last().map_or(true, |c| c.0 < end) { cuts.push((end, false, false)); }
        let fold = self.cfg.fold;
        let fid = |v: &[u32]| -> Vec<u32> { let mut x = v.to_vec(); if let Some(m) = fold { for t in x.iter_mut() { *t %= m; } } x };
        let use_mtp = self.cfg.mtp && self.eng.mtp_available();
        let mut at = pos0;
        let mut seg_prefill = 0.0;
        // a warm-up feeds in pieces and checks its cancel flag between them; a real request feeds each segment at once
        let piece = if warm { self.eng.warm_chunk().max(1) } else { usize::MAX };
        for (cut, stable, save) in cuts {
            while at < cut {
                if job.cancel.load(Ordering::SeqCst) {
                    // cancelled by its own conversation's real request: the fed part (whole aligned pieces, all[at]
                    // pending) stays in the KV for free and that request plans prefix against it; else dropped
                    if at > 0 && self.eng.pos() == at && self.gate.keep_partial() {
                        self.kv = KvState { seq: all[..=at].to_vec(), msgs: vec![], tools: tools.to_vec(), think };
                    } else { self.eng.reset(); self.kv.invalidate(); }
                    o.stop_reason = "cancelled".into(); o.prefill_s = seg_prefill;
                    return Err(o);
                }
                let to = if cut - at > piece { at + piece } else { cut };
                // feed up to the cut; the one generated token is discarded (the next segment starts with all[cut])
                let r = self.eng.feed_cut(&fid(&all[at..to]), use_mtp);
                match r {
                    Ok(st) if self.eng.pos() == to => { seg_prefill += st.prefill_s; add_stream(&mut o.stream, &st.stream); }
                    Ok(_) => { self.eng.reset(); self.kv.invalidate(); return Err(fail(o, format!("snapshot segment: engine at {} != {}", self.eng.pos(), to))); }
                    Err(e) => { self.eng.reset(); self.kv.invalidate(); return Err(fail(o, e)); }
                }
                at = to;
            }
            if save {
                let tag = if static_cut == Some(cut) { Some(crate::snap_tier::StableTag::plan_static()) } else { job.req.stable_tag() };
                if let Some(ev) = self.hook.save(self.eng, &all[..cut], stable, tag) {
                    let ok = ev.error.is_none();
                    o.snaps.push(ev);
                    // a warm-up's stable prefix goes to the hot tier at once (the next plan call restores it from RAM)
                    if ok && warm && stable { if let Some(ev) = self.hook.stable_hot(self.eng, &all[..cut]) { o.snaps.push(ev); } }
                }
            }
        }
        if warm {
            o.prefill_s = seg_prefill;
            if self.eng.pos() != end { self.eng.reset(); self.kv.invalidate(); return Err(fail(o, format!("warm-up: engine at {} != {}", self.eng.pos(), end))); }
            // the engine holds the prompt minus its last token (pending): the state of a reply, without one. No
            // messages: followup (which assumes our reply comes next) never plans against it; prefix does.
            self.kv = KvState { seq: all, msgs: vec![], tools: tools.to_vec(), think };
            o.stop_reason = "warm".into();
            return Err(o);
        }
        let room = max_ctx - (reused + feed.len());
        let nnew = req.max_tokens.min(room).max(1);
        let fed = fid(&all[at..]);
        let mut ban = if req.plan { plan_ban_ids(&tk) } else { vec![] };
        for &t in &req.ban_ids { if !ban.contains(&t) { ban.push(t); } }
        Ok(Prepared { o, all, fed, nnew, use_mtp, seg_prefill, think, ttft: None, gen: Vec::with_capacity(nnew.min(8192)),
                      dec: StreamDecoder::new(), ban, ban_armed: false, stops: req.stop.clone(), text: String::new(), sent: 0,
                      stop_hit: None, format_stop: false })
    }

    /// One job start to end (the one-at-a-time worker): prepare, feed + decode, finish.
    fn generate(&mut self, job: &Job) -> Outcome {
        let mut p = match self.prepare(job) { Ok(p) => p, Err(o) => return o };
        let fed = std::mem::take(&mut p.fed);
        let (eng, tk) = (self.eng, self.tk.clone());
        p.arm_ban(eng);
        let r = self.eng.feed(&fed, p.nnew, &STOP_IDS, p.use_mtp, &mut |t| p.on_tok(job, eng, &tk, t));
        self.finish(job, p, r)
    }

    /// After the decode of a prepared job (the engine on its slot): bookkeeping, reply text, stats.
    fn finish(&mut self, job: &Job, mut p: Prepared, r: Result<FeedStats, String>) -> Outcome {
        // the ban is per request: cleared on this (selected) slot before anything else runs on it
        if p.ban_armed { self.eng.set_ban(&[]); }
        let tail = p.dec.finish();
        p.text_out(job, &tail, true);
        let Prepared { mut o, all, use_mtp, seg_prefill, think, ttft, gen, text, stop_hit, format_stop, .. } = p;
        let tk = self.tk.clone();
        o.ttft_s = ttft;
        o.mtp = use_mtp;
        let (msgs, tools): (&[ChatMsg], &[ToolDef]) = (&job.req.msgs, &job.req.tools);
        match r {
            Ok(st) => {
                // the engine holds all ++ gen (last generated token pending)
                self.kv.commit(Mode::Full, &all, &gen, msgs, tools, think, st.gen > 0);
                if self.kv.is_valid() && (st.last.is_none() || self.kv.seq.len() != self.eng.pos() + 1) {
                    eprintln!("gateway: KV bookkeeping ({} tokens) != engine ({} + pending) after the reply: next request is a full prefill",
                              self.kv.seq.len(), self.eng.pos());
                    self.kv.invalidate();
                }
                let body = match gen.last() { Some(l) if STOP_IDS.contains(l) || format_stop => &gen[..gen.len() - 1], _ => &gen[..] };
                o.n_new = st.gen;
                o.stop_reason = st.stop_reason.clone();
                if let Some((cut, i)) = stop_hit {
                    // the reply up to the stop sequence (the engine stopped there; its tokens stay in the KV as usual)
                    o.raw = text[..cut].to_string();
                    o.stop_reason = "stop_sequence".into();
                    o.stop_sequence = Some(job.req.stop[i].clone());
                } else {
                    o.raw = tk.decode(body);
                    if format_stop { o.stop_reason = "format".into(); }
                }
                o.prefill_s = st.prefill_s + seg_prefill;
                add_stream(&mut o.stream, &st.stream);
                if st.gen > 1 && st.decode_s > 0.0 { o.decode_tok_s = Some((st.gen - 1) as f64 / st.decode_s); }
                o.spec = st.spec;
                o.ctx_draft = st.ctx_draft;
                o
            }
            Err(e) => {
                self.eng.reset();
                self.kv.invalidate();
                o.n_new = gen.len();
                o.stop_reason = "error".into();
                o.error = Some(e);
                o
            }
        }
    }
}

/// A request decoding in the batched scheduler (`Worker::run_batched`).
struct Active {
    job: Job,
    p: Prepared,
    d: DecSeq,
    tiers: Option<serde_json::Value>,
}

/// A job planned and prefilled up to its last feed (`Worker::prepare`): what the decode feeds and what `finish` needs.
pub(crate) struct Prepared {
    o: Outcome,
    /// every id in the engine after the last feed's prompt part (the bookkeeping base)
    all: Vec<u32>,
    /// the last feed (the end of the prompt), decoded right after
    fed: Vec<u32>,
    nnew: usize,
    use_mtp: bool,
    seg_prefill: f64,
    think: bool,
    ttft: Option<f64>,
    gen: Vec<u32>,
    dec: StreamDecoder,
    /// banned ids of this request (`format: plan` tool-call open token + `ban_ids`); `ban_armed`: set on the engine
    ban: Vec<u32>,
    ban_armed: bool,
    /// stop sequences; with any, `text` is the raw reply so far, `sent` the bytes of it sent to the client (the
    /// rest may be the start of a stop sequence), `stop_hit` (byte cut, stop index) once one matched
    stops: Vec<String>,
    text: String,
    sent: usize,
    stop_hit: Option<(usize, usize)>,
    /// a banned id came out (a backend without `set_ban`): the reply ends there, stop_reason "format"
    format_stop: bool,
}
impl Prepared {
    /// Per generated token: first-token time + answer-library boundary, text piece to the client, cancel check.
    fn on_tok<B: Backend + ?Sized>(&mut self, job: &Job, eng: &B, tk: &Tokenizer, t: u32) -> bool {
        // first token: the prompt's passes are done, no decode pass is encoded yet (answer library boundary; a
        // no-op unless the library is on)
        if self.ttft.is_none() { self.ttft = Some(job.t_recv.elapsed().as_secs_f64()); eng.mark_prompt(); }
        self.gen.push(t);
        // stop-level guard: never fires when the engine applied the ban before its argmax (`Backend::set_ban`)
        if self.ban.contains(&t) { self.format_stop = true; return false; }
        if !STOP_IDS.contains(&t) {
            let piece = self.dec.push(tk, t);
            if self.text_out(job, &piece, false) { return false; }
        }
        !job.cancel.load(Ordering::SeqCst)
    }

    /// Ban this request's ids on the engine (selected slot) before its decode.
    fn arm_ban<B: Backend + ?Sized>(&mut self, eng: &B) {
        if !self.ban.is_empty() { self.ban_armed = eng.set_ban(&self.ban); }
    }

    /// Send decoded text to the client, holding back what may be the start of a stop sequence (`last`: flush it).
    /// true = a stop sequence matched (nothing from it on is sent).
    fn text_out(&mut self, job: &Job, piece: &str, last: bool) -> bool {
        if self.stop_hit.is_some() { return true; }
        let send = |x: &str| if !x.is_empty() && job.events.send(Ev::Text(x.to_string())).is_err() { job.cancel.store(true, Ordering::SeqCst); };
        if self.stops.is_empty() { send(piece); return false; }
        self.text.push_str(piece);
        let hit = self.stops.iter().enumerate().filter_map(|(i, x)| self.text[self.sent..].find(x.as_str()).map(|at| (self.sent + at, i))).min();
        let upto = match hit {
            Some((cut, _)) => cut,
            None if last => self.text.len(),
            None => self.text.len() - stop_prefix_tail(&self.text, &self.stops),
        };
        send(&self.text[self.sent..upto]);
        self.sent = upto;
        self.stop_hit = hit;
        hit.is_some()
    }
}

/// Bytes of the longest suffix of `text` that is a proper prefix of one of `stops` (held back while streaming).
fn stop_prefix_tail(text: &str, stops: &[String]) -> usize {
    stops.iter().filter_map(|x| (1..x.len()).rev().find(|&l| x.is_char_boundary(l) && text.ends_with(&x[..l]))).max().unwrap_or(0)
}

/// Parked conversations kept (bookkeeping only: token list + messages + tools, ~100 KB each for a Claude Code session).
pub const PARKED_MAX: usize = 16;

/// Park `kv` (newest last): replaces an older entry of the same conversation; the oldest goes beyond `PARKED_MAX`.
fn park(parked: &mut Vec<KvState>, kv: KvState) {
    let conv = crate::snap_tier::session_keys(&kv.seq).1;
    parked.retain(|p| crate::snap_tier::session_keys(&p.seq).1 != conv);
    parked.push(kv);
    if parked.len() > PARKED_MAX { parked.remove(0); }
}

/// The ids in the engine (len == pos: the bookkeeping minus the pending token) when the bookkeeping is exact.
fn fed_ids(kv: &KvState, pos: usize) -> Option<&[u32]> {
    (kv.is_valid() && kv.seq.len() == pos + 1).then(|| &kv.seq[..pos])
}

fn add_stream(acc: &mut Option<StreamFeed>, s: &Option<StreamFeed>) {
    if let Some(s) = s { acc.get_or_insert_with(StreamFeed::default).add(s); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A queued job: conversation `first` (its first user message), warm-up or not, project + lane, age in seconds.
    fn job(first: &str, warm: bool, project: Option<u64>, plan: bool, age_s: u64) -> Job {
        let mut req = crate::anthropic::convert(&json!({"model": "m", "max_tokens": if warm { 0 } else { 8 },
            "messages": [{"role": "user", "content": first}]})).unwrap();
        req.project = project;
        req.lane_plan = plan;
        let (tx, _rx) = std::sync::mpsc::channel();
        let conv = req.conv_key();
        Job { req, warm, conv, id: first.into(), events: tx, cancel: Arc::new(AtomicBool::new(false)),
              t_recv: Instant::now() - Duration::from_secs(age_s) }
    }
    fn ids(q: &VecDeque<Job>) -> Vec<&str> { q.iter().map(|j| j.id.as_str()).collect() }

    #[test]
    fn prune_warm_rules() {
        let mut q: VecDeque<Job> = VecDeque::new();
        q.push_back(job("old", true, None, false, 31));          // stale
        q.push_back(job("a", true, None, false, 1));             // superseded by the newer "a" (same conversation)
        q.push_back(job("real", false, None, false, 60));        // real requests are never dropped (any age)
        q.push_back(job("p1", true, Some(7), false, 1));         // superseded by p2 (same project, agent lane)
        q.push_back(job("plan1", true, Some(7), true, 1));       // other lane: kept
        q.push_back(job("p2", true, Some(7), false, 0));
        q.push_back(job("a", true, None, false, 0));
        q.push_back(job("other", true, Some(8), false, 0));      // other project: kept
        q.push_back(job("gone", true, None, false, 0));          // client closed: cancel flag set
        q.back().unwrap().cancel.store(true, Ordering::SeqCst);
        let d = prune_warm(&mut q, WARM_MAX_AGE, Instant::now());
        let d: Vec<(&str, &str)> = d.iter().map(|(j, w)| (j.id.as_str(), *w)).collect();
        assert_eq!(d, [("old", "stale"), ("a", "superseded"), ("p1", "superseded"), ("gone", "cancelled")]);
        assert_eq!(ids(&q), ["real", "plan1", "p2", "a", "other"]);
        // a second pass drops nothing more
        assert!(prune_warm(&mut q, WARM_MAX_AGE, Instant::now()).is_empty());
    }

    #[test]
    fn drop_reaches_client() {
        struct Nop;
        impl Backend for Nop {
            fn reset(&self) {}
            fn pos(&self) -> usize { 0 }
            fn max_ctx(&self) -> usize { 4096 }
            fn vocab(&self) -> usize { 1000 }
            fn mtp_available(&self) -> bool { false }
            fn feed(&self, _: &[u32], _: usize, _: &[u32], _: bool, _: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> { panic!("a dropped warm-up must not feed") }
        }
        let Ok(tk) = Tokenizer::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4")) else { eprintln!("SKIP: 27B tokenizer missing"); return };
        let eng = Nop;
        let mut w = Worker::new(&eng, Arc::new(tk), WorkerConfig { think: false, mtp: false, fold: None });
        let (tx, rx) = std::sync::mpsc::channel();
        let mut j = job("x", true, None, false, 40);
        let (etx, erx) = std::sync::mpsc::channel();
        j.events = etx;
        tx.send(j).unwrap();
        drop(tx);
        w.run(rx);
        match erx.recv().unwrap() {
            Ev::Done(o) => { assert_eq!((o.stop_reason.as_str(), o.mode.as_str(), o.drop_reason), ("cancelled", "dropped", Some("stale"))); assert!(o.warm); }
            e => panic!("{:?}", e),
        }
    }
}
