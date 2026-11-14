//! SSD embedding reader + CPU hand-off thread (plan sections 2, 3, 6.2, 7.3).
//! - Embedding rows stay on SSD: `EmbedSrc` keeps the shard file and the absolute offsets of
//!   `embed_tokens.{weight, scales, biases}`. One token = 3 F_NOCACHE preads.
//! - Staging row layout (what `embed_q4` reads): [u32 w[hidden/8] | bf16 s[hidden/64] | bf16 b[hidden/64]]
//!   (2560 + 160 + 160 = 2880 B at hidden 5120).
//! - Staging = 2 slots of `slot_rows` rows; the step n hand-off fills slot (n+1) % 2.
//! - `Handoff` = one CPU thread per GPU event: wait for the GPU signal, read `SdCtrl`, pread the next row(s),
//!   signal the CPU-side event (MTP) and commit the next command buffer through the engine's callback.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use streamdense_core::io::NoCacheFile;
use streamdense_metal::{Event, SdCtrl};

use crate::qwen_config::QwenConfig;
use crate::weights::EmbedSrc;

/// Packed u32 weight bytes of one row (8 nibbles per u32).
pub const fn w_bytes(hidden: usize) -> usize { hidden / 2 }
/// bf16 scale (or bias) bytes of one row (one per 64 weights).
pub const fn sb_bytes(hidden: usize) -> usize { hidden / 64 * 2 }
/// Bytes of one staged row: W | S | B.
pub const fn row_bytes(hidden: usize) -> usize { w_bytes(hidden) + 2 * sb_bytes(hidden) }
/// Staged row size of Qwen3.8-27B (hidden 5120). Convenience only: the reader uses `row_bytes(cfg.hidden)`.
pub const ROW_BYTES: usize = 2880;
const _: () = assert!(row_bytes(5120) == ROW_BYTES);

// ---------------------------------------------------------------------------------------------------------
// Reader

pub struct EmbedReader {
    f: NoCacheFile,
    src: EmbedSrc,
    hidden: usize,
    vocab: usize,
    threads: usize,
    /// per id count of rows read (after a successful read; ActivityStats `embed_rows`)
    pub counts: Option<crate::activity::SharedCounts>,
}

impl EmbedReader {
    /// Open the shard holding the embedding. Sizes come from `cfg.hidden` / `cfg.vocab`; the three tensors
    /// must fit in the file.
    pub fn open(src: &EmbedSrc, cfg: &QwenConfig, threads: usize) -> Result<EmbedReader, String> {
        let hidden = cfg.hidden;
        if hidden == 0 || hidden % 64 != 0 { return Err(format!("embed: hidden {} not a multiple of 64", hidden)); }
        let f = NoCacheFile::open(&src.file)?;
        let v = cfg.vocab as u64;
        for (name, off, per) in [("weight", src.w_off, w_bytes(hidden)), ("scales", src.s_off, sb_bytes(hidden)), ("biases", src.b_off, sb_bytes(hidden))] {
            let end = off + v * per as u64;
            if end > f.len { return Err(format!("embed: {} ends at {} past file length {} ({})", name, end, f.len, src.file)); }
        }
        Ok(EmbedReader { f, src: src.clone(), hidden, vocab: cfg.vocab, threads: threads.max(1), counts: None })
    }
    pub fn hidden(&self) -> usize { self.hidden }
    pub fn vocab(&self) -> usize { self.vocab }
    pub fn row_bytes(&self) -> usize { row_bytes(self.hidden) }

    /// The 3 (dst, len, file_off) jobs for token `tok` staged at `dst`.
    fn jobs(&self, tok: u32, dst: usize, out: &mut Vec<(usize, usize, u64)>) {
        let (wb, sb) = (w_bytes(self.hidden), sb_bytes(self.hidden));
        let t = tok as u64;
        out.push((dst, wb, self.src.w_off + t * wb as u64));
        out.push((dst + wb, sb, self.src.s_off + t * sb as u64));
        out.push((dst + wb + sb, sb, self.src.b_off + t * sb as u64));
    }

    /// pread the rows of `toks` to `dst` (row i at dst + i * row_bytes). Small batches run on the calling
    /// thread (decode: 1–2 rows); larger ones (prefill chunks) use `threads` threads.
    pub fn read_rows(&self, toks: &[u32], dst: *mut u8) -> Result<(), String> {
        let rb = self.row_bytes();
        let mut jobs = Vec::with_capacity(toks.len() * 3);
        for (i, &t) in toks.iter().enumerate() {
            if t as usize >= self.vocab { return Err(format!("embed: token {} >= vocab {}", t, self.vocab)); }
            self.jobs(t, dst as usize + i * rb, &mut jobs);
        }
        if self.threads > 1 && toks.len() > 4 {
            self.f.read_many(&jobs, self.threads.min(toks.len()), usize::MAX)?;
        } else {
            for &(d, n, off) in &jobs { self.f.read_at(d as *mut u8, n, off)?; }
        }
        if let Some(c) = &self.counts { c.add(toks); }
        Ok(())
    }
}

// Shared-event access lives in `streamdense-metal` (re-exported here for existing callers).
pub use streamdense_metal::{ev_set, ev_value};

/// Value a hand-off sets on its CPU event when it exits abnormally, so no GPU wait stays blocked.
/// The engine must make fresh events after such an exit.
pub const RELEASE_ALL: u64 = 1 << 62;

// ---------------------------------------------------------------------------------------------------------
// Latency stats

#[derive(Clone, Copy, Debug, Default)]
pub struct LatStats { pub n: usize, pub p50_us: f64, pub p99_us: f64, pub max_us: f64, pub mean_us: f64 }

impl LatStats {
    pub fn from_us(s: &[f64]) -> LatStats {
        if s.is_empty() { return LatStats::default(); }
        let mut v = s.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| v[(((v.len() - 1) as f64) * p).round() as usize];
        LatStats { n: v.len(), p50_us: q(0.50), p99_us: q(0.99), max_us: v[v.len() - 1], mean_us: v.iter().sum::<f64>() / v.len() as f64 }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Hand-off

/// Plain: one CB per token (6.2). MtpDec / MtpDraft: the two hand-offs inside one MTP step (7.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffKind { Plain, MtpDec, MtpDraft }

/// Commits the already-encoded CB(n) (U15 owns command buffers).
pub type CommitFn = Box<dyn FnMut(u64) + Send>;
/// Test-only draft override (7.5): (step n, GPU draft) -> draft to use; written back to `ctrl.draft`.
pub type DraftHook = Box<dyn FnMut(u64, u32) -> u32 + Send>;

/// State shared by the hand-off(s) of one generate call and the engine.
pub struct HandoffLink {
    /// Last step whose CB is the final one (u64::MAX while running). Set by Plain / MtpDec.
    pub stop_at: AtomicU64,
    /// Engine or a failing thread asks every thread to exit.
    pub abort: AtomicBool,
    /// Tokens emitted so far (Plain / MtpDec), in order; eos is included when hit.
    pub emitted: Mutex<Vec<u32>>,
}
impl HandoffLink {
    pub fn new() -> Arc<HandoffLink> {
        Arc::new(HandoffLink { stop_at: AtomicU64::new(u64::MAX), abort: AtomicBool::new(false), emitted: Mutex::new(Vec::new()) })
    }
}

pub struct HandoffPolicy {
    pub kind: HandoffKind,
    /// First step the thread waits for (the engine committed CB(first) itself).
    pub first: u64,
    /// Rows per staging slot (slot stride = slot_rows * row_bytes).
    pub slot_rows: usize,
    pub eos: Vec<u32>,
    /// Stop after this many emitted tokens (Plain / MtpDec).
    pub max_emit: usize,
    /// CPU event this thread signals: F_mtp (MtpDec, value n) or F_emb (MtpDraft, value n + 1). None for Plain.
    pub signal_ev: Option<Event>,
    /// Shared with the partner thread in MTP mode; None makes a fresh one.
    pub link: Option<Arc<HandoffLink>>,
    pub draft_hook: Option<DraftHook>,
    /// Max wait for one GPU signal before giving up.
    pub timeout: Duration,
}
impl HandoffPolicy {
    pub fn new(kind: HandoffKind, first: u64, slot_rows: usize, eos: Vec<u32>, max_emit: usize) -> HandoffPolicy {
        HandoffPolicy { kind, first, slot_rows, eos, max_emit, signal_ev: None, link: None, draft_hook: None, timeout: Duration::from_secs(30) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason { Eos, CtrlStop, MaxTokens, Draft /* MtpDraft: partner stopped */, Abort }

#[derive(Debug)]
pub struct HandoffOut {
    pub tokens: Vec<u32>,
    pub steps: u64,
    pub last_step: u64,
    pub stop: StopReason,
    /// GPU signal seen -> next CB committed (or CPU event signaled for MtpDraft).
    pub handoff: LatStats,
    /// The pread part of it.
    pub pread: LatStats,
}

pub struct Handoff {
    th: Option<std::thread::JoinHandle<Result<HandoffOut, String>>>,
    link: Arc<HandoffLink>,
}

struct SendPtr(usize);

impl Handoff {
    /// Spawn the hand-off thread.
    /// - `gpu_ev`: E (Plain), E_dec (MtpDec) or E_draft (MtpDraft); CB(n) signals it to n.
    /// - `ctrl_ptr`: the `SdCtrl` in shared GPU memory. `stage_ptr`: staging slot 0 (slot 1 follows).
    /// - Rows written: Plain → slot (n+1)%2 row 0 = ctrl.tok[0]; MtpDec → slot (n+1)%2 row 0 = new t1
    ///   (last emitted); MtpDraft → slot (n+1)%2 row 1 = ctrl.draft.
    /// - `commit(n + 1)` is called by Plain / MtpDec unless stopping; MtpDraft takes None.
    /// Safety contract (caller): ctrl_ptr and both staging slots stay valid until `join` returns.
    pub fn spawn(reader: Arc<EmbedReader>, gpu_ev: Event, ctrl_ptr: *mut SdCtrl, stage_ptr: *mut u8,
                 mut policy: HandoffPolicy, mut commit: Option<CommitFn>) -> Handoff {
        let link = policy.link.take().unwrap_or_else(HandoffLink::new);
        let l2 = link.clone();
        let (ctrl, stage) = (SendPtr(ctrl_ptr as usize), SendPtr(stage_ptr as usize));
        let th = std::thread::Builder::new().name(format!("sd-handoff-{:?}", policy.kind)).spawn(move || {
            let (ctrl, stage) = (ctrl, stage);
            let r = run(&reader, &gpu_ev, ctrl.0 as *mut SdCtrl, stage.0 as *mut u8, &mut policy, commit.as_mut(), &l2);
            if r.is_err() || l2.abort.load(Ordering::SeqCst) {
                l2.abort.store(true, Ordering::SeqCst);
                if let Some(ev) = &policy.signal_ev { ev_set(ev, RELEASE_ALL); }
            }
            r
        }).expect("spawn hand-off thread");
        Handoff { th: Some(th), link }
    }
    pub fn link(&self) -> &Arc<HandoffLink> { &self.link }
    pub fn abort(&self) { self.link.abort.store(true, Ordering::SeqCst); }
    pub fn is_finished(&self) -> bool { self.th.as_ref().map_or(true, |t| t.is_finished()) }
    pub fn join(mut self) -> Result<HandoffOut, String> {
        self.th.take().unwrap().join().map_err(|_| "hand-off thread panicked".to_string())?
    }
}
impl Drop for Handoff {
    fn drop(&mut self) {
        if let Some(t) = self.th.take() { self.link.abort.store(true, Ordering::SeqCst); let _ = t.join(); }
    }
}

/// Wait until `ev` >= v: spin, then yield, then short sleeps. Err on abort or timeout.
fn wait_ev(ev: &Event, v: u64, link: &HandoffLink, timeout: Duration) -> Result<(), String> {
    let t0 = Instant::now();
    let mut i = 0u32;
    loop {
        if ev_value(ev) >= v { std::sync::atomic::fence(Ordering::Acquire); return Ok(()); }
        if link.abort.load(Ordering::Relaxed) { return Err(format!("aborted while waiting for step {}", v)); }
        i += 1;
        if i < 2000 { std::hint::spin_loop(); }
        else if i < 4000 { std::thread::yield_now(); }
        else {
            if t0.elapsed() > timeout { return Err(format!("timeout waiting for GPU signal {} (at {})", v, ev_value(ev))); }
            std::thread::sleep(Duration::from_micros(20));
        }
    }
}

fn run(reader: &EmbedReader, ev: &Event, ctrl: *mut SdCtrl, stage: *mut u8, p: &mut HandoffPolicy,
       mut commit: Option<&mut CommitFn>, link: &HandoffLink) -> Result<HandoffOut, String> {
    let slot = p.slot_rows.max(1) * reader.row_bytes();
    let rb = reader.row_bytes();
    let (mut hand, mut pr) = (Vec::new(), Vec::new());
    let mut n = p.first;
    let mut tokens: Vec<u32> = Vec::new();
    let mut steps = 0u64;
    let stop;
    loop {
        if p.kind == HandoffKind::MtpDraft && n >= link.stop_at.load(Ordering::SeqCst) { stop = StopReason::Draft; break; }
        if let Err(e) = wait_ev(ev, n, link, p.timeout) {
            if link.abort.load(Ordering::SeqCst) { stop = StopReason::Abort; break; }
            return Err(e);
        }
        let t0 = Instant::now();
        let c: SdCtrl = unsafe { std::ptr::read_volatile(ctrl) };
        let dst = unsafe { stage.add(((n + 1) % 2) as usize * slot) };
        match p.kind {
            HandoffKind::MtpDraft => {
                // Partner may have stopped at this very step (it signalled F_mtp(n) after setting stop_at).
                if n >= link.stop_at.load(Ordering::SeqCst) { stop = StopReason::Draft; break; }
                let mut d = c.draft;
                if let Some(h) = p.draft_hook.as_mut() {
                    let o = h(n, d);
                    if o != d { unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*ctrl).draft), o) }; d = o; }
                }
                let tp = Instant::now();
                reader.read_rows(&[d], unsafe { dst.add(rb) })?;
                pr.push(tp.elapsed().as_secs_f64() * 1e6);
                if let Some(e) = &p.signal_ev { ev_set(e, n + 1); }
            }
            HandoffKind::Plain | HandoffKind::MtpDec => {
                let new: Vec<u32> = if p.kind == HandoffKind::Plain { vec![c.tok[0]] } else { c.emit[..(c.n_emit as usize).min(2)].to_vec() };
                let mut why = None;
                let start = tokens.len();
                for &t in &new {
                    tokens.push(t);
                    if p.eos.contains(&t) { why = Some(StopReason::Eos); break; }
                    if tokens.len() >= p.max_emit { why = Some(StopReason::MaxTokens); break; }
                }
                if why.is_none() && c.stop != 0 { why = Some(StopReason::CtrlStop); }
                if why.is_none() && new.is_empty() { return Err(format!("step {}: n_emit = 0", n)); }
                link.emitted.lock().unwrap().extend_from_slice(&tokens[start..]);
                if let Some(w) = why {
                    link.stop_at.store(n, Ordering::SeqCst);
                    // Let CB(n) finish its MTP half; no CB(n+1).
                    if let Some(e) = &p.signal_ev { ev_set(e, n); }
                    hand.push(t0.elapsed().as_secs_f64() * 1e6);
                    steps += 1;
                    stop = w;
                    break;
                }
                let tp = Instant::now();
                reader.read_rows(&[*new.last().unwrap()], dst)?;
                pr.push(tp.elapsed().as_secs_f64() * 1e6);
                if let Some(e) = &p.signal_ev { ev_set(e, n); }
                if let Some(cm) = commit.as_mut() { cm(n + 1); }
            }
        }
        hand.push(t0.elapsed().as_secs_f64() * 1e6);
        steps += 1;
        n += 1;
    }
    Ok(HandoffOut { steps, last_step: n, tokens, stop, handoff: LatStats::from_us(&hand), pread: LatStats::from_us(&pr) })
}
