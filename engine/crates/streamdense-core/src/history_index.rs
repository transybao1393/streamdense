//! history_index: a bounded, persistent index of token text the user already has (project files, past sessions,
//! finished replies) that proposes speculative-decoding drafts for a context tail. Lossless by design: it only
//! proposes; the full model verifies every drafted token, so a wrong or stale entry costs speed, never output.
//!
//! Index: n-gram hash (orders 2,3,4,6,8 by default) → up to 4 next tokens with counts, the most recently seen one
//! marked. One fixed table, allocated once from the RAM budget: 64 shards (one Mutex each) × buckets of 4 slots of
//! 32 B (`check u32, meta u32 = epoch<<2|last, tok [u32;4], cnt [u16;4]`). A miss in a full bucket evicts the slot
//! with the lowest decayed frequency `Σcnt·2^16 / (1 + age)`, age in epochs of 4096 ingested tokens, so frequent
//! and recent n-grams stay. Removed or changed files keep their old counts (lossy, verified anyway).
//!
//! Query: longest indexed order that matches the tail (≤ 8 tokens + drafted ones) → next token; repeated up to
//! `max_len`. Confidence per token = cumulative product of `trust(n) · cnt / Σcnt`; `trust(n)` is a prior
//! per order blended with accept/reject feedback (`feedback`). With `max_branches > 1` the best alternatives
//! (highest cumulative confidence) become extra rows of a small tree (≤ 8 rows), each extended greedily.
//!
//! RAM (by formula from `Config::budget_bytes`, default 32 MB): table = 3/4 budget; file manifest ≤ 1/16 budget
//! (`max_files` = budget/16/32 B); build threads × 3 × `max_file_bytes` + session parse ≤ the rest; updater queue
//! ≤ 1/16 budget (pushes past it are dropped and counted). The tokenizer used by the encoder is not counted (the
//! gateway already holds it).
//!
//! Files in `Config::dir` (written as `.tmp` + rename): `index.sdh` = 4 KB header + slots (crc32 checked on open; a
//! bad crc or a different geometry starts empty), `files.bin` = manifest (path hash → size/mtime hash), so a re-scan
//! only tokenizes new or changed files.
//!
//! API:
//! ```ignore
//! let idx = Arc::new(HistoryIndex::open(Config::default())?.0);      // load or start empty
//! let d: Drafts = idx.query(&tail, 8, 4);                             // rows[0] = best chain; ≤ 0.2 ms
//! idx.feedback(d.rows[0].order[i], accepted);                         // optional calibration
//! let up = Updater::spawn(idx.clone(), encoder, UpdaterOptions::default());   // background QoS thread
//! up.push_tokens(prompt_tail_plus_reply); up.scan_dir(project); up.scan_sessions(path); up.pause(); up.resume();
//! ```
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

pub const MAX_ORDER: usize = 8;
pub const MAX_ROWS: usize = 8;
pub const MAX_DRAFT: usize = 32;
pub const DEFAULT_BUDGET: usize = 32 << 20;
pub const MIN_BUDGET: usize = 256 << 10;
pub const DEFAULT_ORDERS: [usize; 5] = [2, 3, 4, 6, 8];
const WAYS: usize = 4;
const CONT: usize = 4;
const SHARDS: usize = 64;
const SLOT_BYTES: usize = 32;
const EPOCH_SHIFT: u32 = 12;
const BATCH: usize = 4096;
const MANIFEST_ENTRY: usize = 32;
const MAGIC: u64 = 0x3158_4449_4854_5344; // "SDTHIDX1"
const FILES_MAGIC: u64 = 0x3153_454c_4946_4453; // "SDFILES1"
const VERSION: u32 = 1;
const HDR: usize = 4096;
const INDEX_FILE: &str = "index.sdh";
const FILES_FILE: &str = "files.bin";
/// prior P(accept) of the top continuation at order n (index = n); blended with feedback
const PRIOR: [f64; MAX_ORDER + 1] = [0.0, 0.3, 0.45, 0.6, 0.72, 0.8, 0.85, 0.88, 0.9];
const PRIOR_W: f64 = 16.0;
const SEED: u64 = 0x243F_6A88_85A3_08D3;
/// extensions that are never text worth indexing (checked before reading)
const BIN_EXT: [&str; 34] = ["png", "jpg", "jpeg", "gif", "webp", "ico", "icns", "pdf", "zip", "gz", "tgz", "xz", "zst", "bz2", "7z",
    "bin", "safetensors", "gguf", "npy", "npz", "pt", "pth", "onnx", "so", "dylib", "a", "o", "rlib", "wasm", "mp3", "mp4", "mov", "woff2", "sqlite"];
/// JSON keys whose string values are metadata, not text (session logs)
const SKIP_KEYS: [&str; 10] = ["id", "type", "role", "tool_use_id", "signature", "model", "stop_reason", "media_type", "cache_control", "name"];

/// Text → token ids (the gateway's tokenizer; tests use a toy one).
pub type Encoder = Arc<dyn Fn(&str) -> Vec<u32> + Send + Sync>;

// ---------- config ----------
#[derive(Clone, Debug)]
pub struct Config {
    /// hard RAM bound of the index (table + manifest + build buffers), bytes
    pub budget_bytes: usize,
    /// persistence dir (`index.sdh`, `files.bin`)
    pub dir: PathBuf,
    /// n-gram orders indexed, each in 1..=8
    pub orders: Vec<usize>,
    /// files larger than this are skipped (also sizes the build buffers)
    pub max_file_bytes: u64,
    /// build threads; 0 = auto (cores − 2, bounded by the budget)
    pub threads: usize,
    /// rows stop when the cumulative confidence falls below this
    pub min_conf: f32,
}
impl Default for Config {
    fn default() -> Self {
        Config { budget_bytes: DEFAULT_BUDGET, dir: default_dir(), orders: DEFAULT_ORDERS.to_vec(), max_file_bytes: 256 << 10, threads: 0, min_conf: 0.1 }
    }
}
impl Config {
    pub fn with_dir(dir: &Path) -> Config { Config { dir: dir.to_path_buf(), ..Config::default() } }
    /// table bytes = 3/4 of the budget, whole slots
    pub fn table_bytes(&self) -> usize { buckets_per_shard(self.budget_bytes) * SHARDS * WAYS * SLOT_BYTES }
    /// manifest entries ≤ budget/16
    pub fn max_files(&self) -> usize { self.budget_bytes.max(MIN_BUDGET) / 16 / MANIFEST_ENTRY }
    /// session files larger than this are skipped (parsed whole: ≤ budget/8)
    pub fn max_session_bytes(&self) -> u64 { (self.budget_bytes.max(MIN_BUDGET) / 8) as u64 }
    /// build threads: requested (or cores − 2), bounded so threads × 3 × max_file_bytes ≤ 3/16 budget
    pub fn build_threads(&self) -> usize {
        let auto = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1);
        let want = if self.threads == 0 { auto } else { self.threads };
        let fit = (self.budget_bytes.max(MIN_BUDGET) * 3 / 16) / (3 * self.max_file_bytes.max(1) as usize);
        want.min(fit).max(1)
    }
}
fn buckets_per_shard(budget: usize) -> usize { (budget.max(MIN_BUDGET) * 3 / 4 / (SHARDS * WAYS * SLOT_BYTES)).max(1) }

/// `~/.streamdense/history_index` (or `./.streamdense/history_index` without $HOME).
pub fn default_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")).join(".streamdense/history_index")
}

// ---------- hashing ----------
const CRC: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 { let mut c = i as u32; let mut k = 0; while k < 8 { c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }; k += 1 } t[i] = c; i += 1 }
    t
};
fn crc_up(c: u32, b: &[u8]) -> u32 { !b.iter().fold(!c, |c, &x| CRC[((c ^ x as u32) & 0xff) as usize] ^ (c >> 8)) }
fn mix(h: u64, t: u32) -> u64 { (h ^ t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31) }
fn fin(h: u64, n: usize) -> u64 {
    let mut x = h ^ ((n as u64) << 56);
    x ^= x >> 33; x = x.wrapping_mul(0xff51_afd7_ed55_8ccd); x ^= x >> 33; x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53); x ^= x >> 33;
    x | 1
}
/// k[n] = key of the n-gram ending at the end of `last` (n = 1..=8), 0 when `last` is shorter.
fn keys(last: &[u32]) -> [u64; MAX_ORDER + 1] {
    let mut k = [0u64; MAX_ORDER + 1];
    let mut h = SEED;
    for n in 1..=MAX_ORDER.min(last.len()) { h = mix(h, last[last.len() - n]); k[n] = fin(h, n); }
    k
}
fn hash_bytes(b: &[u8]) -> u64 { fin(b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3)), 0) }

// ---------- slots ----------
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Slot { check: u32, meta: u32, tok: [u32; CONT], cnt: [u16; CONT] }
impl Slot {
    fn stamp(&self) -> u32 { self.meta >> 2 }
    fn last(&self) -> usize { (self.meta & 3) as usize }
    fn sum(&self) -> u32 { self.cnt.iter().map(|&c| c as u32).sum() }
    fn prio(&self, now: u32) -> u64 { ((self.sum() as u64) << 16) / (1 + now.saturating_sub(self.stamp()) as u64) }
    /// one more observation of `next` after this context
    fn bump(&mut self, next: u32, now: u32) {
        let mut j = match (0..CONT).find(|&i| self.cnt[i] > 0 && self.tok[i] == next) {
            Some(j) => {
                if self.cnt[j] == u16::MAX { for c in self.cnt.iter_mut() { *c = (*c + 1) / 2 } }
                self.cnt[j] += 1; j
            }
            None => match (0..CONT).find(|&i| self.cnt[i] == 0) {
                Some(k) => { self.tok[k] = next; self.cnt[k] = 1; k }
                // full: the weakest entry is worn down first, then replaced (a new continuation needs repeats to stay)
                None if self.cnt[CONT - 1] <= 1 => { self.tok[CONT - 1] = next; self.cnt[CONT - 1] = 1; CONT - 1 }
                None => { self.cnt[CONT - 1] -= 1; self.meta = (now << 2) | self.meta & 3; return }
            },
        };
        while j > 0 && self.cnt[j] > self.cnt[j - 1] { self.cnt.swap(j, j - 1); self.tok.swap(j, j - 1); j -= 1 }
        self.meta = (now << 2) | j as u32;
    }
    fn to_bytes(self) -> [u8; SLOT_BYTES] {
        let mut b = [0u8; SLOT_BYTES];
        b[0..4].copy_from_slice(&self.check.to_le_bytes()); b[4..8].copy_from_slice(&self.meta.to_le_bytes());
        for i in 0..CONT { b[8 + 4 * i..12 + 4 * i].copy_from_slice(&self.tok[i].to_le_bytes()); b[24 + 2 * i..26 + 2 * i].copy_from_slice(&self.cnt[i].to_le_bytes()) }
        b
    }
    fn from_bytes(b: &[u8]) -> Slot {
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let mut s = Slot { check: u(0), meta: u(4), ..Slot::default() };
        for i in 0..CONT { s.tok[i] = u(8 + 4 * i); s.cnt[i] = u16::from_le_bytes(b[24 + 2 * i..26 + 2 * i].try_into().unwrap()) }
        s
    }
}
/// insert one (context, next) observation into bucket `b`; true = a live slot was evicted
fn upsert(tab: &mut [Slot], b: usize, chk: u32, next: u32, now: u32) -> bool {
    let bk = &mut tab[b * WAYS..(b + 1) * WAYS];
    let (mut empty, mut victim, mut best) = (None, 0, u64::MAX);
    for (i, s) in bk.iter_mut().enumerate() {
        if s.check == chk { s.bump(next, now); return false }
        if s.check == 0 { if empty.is_none() { empty = Some(i) } } else { let p = s.prio(now); if p < best { best = p; victim = i } }
    }
    let i = empty.unwrap_or(victim);
    let evicted = bk[i].check != 0;
    bk[i] = Slot { check: chk, meta: now << 2, tok: [next, 0, 0, 0], cnt: [1, 0, 0, 0] };
    evicted
}

// ---------- pause / stop / progress ----------
/// Pause + stop + progress shared by the updater and the build workers (checked between batches and files).
#[derive(Default)]
pub struct Control {
    paused: Mutex<bool>,
    cv: Condvar,
    stop: AtomicBool,
    pub files_found: AtomicU64,
    pub files_done: AtomicU64,
    pub bytes_done: AtomicU64,
    pub tokens_done: AtomicU64,
}
impl Control {
    pub fn pause(&self) { *lock(&self.paused) = true }
    pub fn resume(&self) { *lock(&self.paused) = false; self.cv.notify_all() }
    pub fn is_paused(&self) -> bool { *lock(&self.paused) }
    pub fn stop(&self) { self.stop.store(true, Relaxed); self.resume() }
    pub fn stopped(&self) -> bool { self.stop.load(Relaxed) }
    /// block while paused; false = stop requested
    pub fn wait(&self) -> bool {
        let mut p = lock(&self.paused);
        while *p && !self.stopped() { p = self.cv.wait(p).unwrap_or_else(|e| e.into_inner()) }
        !self.stopped()
    }
}
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> { m.lock().unwrap_or_else(|e| e.into_inner()) }

// ---------- results ----------
/// One drafted chain: tokens in order after the tail. Row r > 0 shares `tokens[..shared]` with row `parent`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DraftRow {
    pub tokens: Vec<u32>,
    /// cumulative confidence ≈ P(this and all earlier tokens of the row accepted); non-increasing
    pub conf: Vec<f32>,
    /// n-gram order that proposed each token (pass to `feedback`)
    pub order: Vec<u8>,
    pub parent: usize,
    pub shared: usize,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drafts { pub rows: Vec<DraftRow>, pub lookups: u32 }
impl Drafts {
    /// best chain (row 0), empty when nothing matched
    pub fn best(&self) -> &[u32] { self.rows.first().map(|r| r.tokens.as_slice()).unwrap_or(&[]) }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub budget_bytes: usize,
    pub table_bytes: usize,
    /// table + manifest (the long-lived part; build buffers are transient and bounded separately)
    pub ram_bytes: usize,
    pub slots: usize,
    pub used_slots: usize,
    pub tokens: u64,
    pub docs: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub files_tracked: usize,
    pub orders: Vec<usize>,
    /// (accepted, rejected) feedback per order 0..=8
    pub feedback: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScanReport {
    pub files_seen: u64,
    pub files_indexed: u64,
    pub files_unchanged: u64,
    pub skipped_ignored: u64,
    pub skipped_binary: u64,
    pub skipped_large: u64,
    pub errors: u64,
    pub bytes: u64,
    pub tokens: u64,
}
impl ScanReport {
    fn add(&mut self, o: &ScanReport) {
        self.files_seen += o.files_seen; self.files_indexed += o.files_indexed; self.files_unchanged += o.files_unchanged;
        self.skipped_ignored += o.skipped_ignored; self.skipped_binary += o.skipped_binary; self.skipped_large += o.skipped_large;
        self.errors += o.errors; self.bytes += o.bytes; self.tokens += o.tokens;
    }
}

/// What `open` found on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenNote { New, Loaded, GeometryChanged, Corrupt(String) }

// ---------- the index ----------
pub struct HistoryIndex {
    cfg: Config,
    orders: Vec<usize>,
    nb: usize,
    shards: Box<[Mutex<Box<[Slot]>>]>,
    tokens: AtomicU64,
    docs: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
    fb: [[AtomicU32; 2]; MAX_ORDER + 1],
    manifest: Mutex<HashMap<u64, u64>>,
}

impl HistoryIndex {
    /// Empty index sized from `cfg.budget_bytes` (everything allocated here, once).
    pub fn new(cfg: Config) -> HistoryIndex {
        let mut orders: Vec<usize> = cfg.orders.iter().copied().filter(|&n| (1..=MAX_ORDER).contains(&n)).collect();
        orders.sort_unstable_by(|a, b| b.cmp(a)); orders.dedup();
        if orders.is_empty() { orders = DEFAULT_ORDERS.iter().rev().copied().collect() }
        let nb = buckets_per_shard(cfg.budget_bytes);
        let shards = (0..SHARDS).map(|_| Mutex::new(vec![Slot::default(); nb * WAYS].into_boxed_slice())).collect();
        HistoryIndex { cfg, orders, nb, shards, tokens: AtomicU64::new(0), docs: AtomicU64::new(0), inserts: AtomicU64::new(0),
            evictions: AtomicU64::new(0), fb: Default::default(), manifest: Mutex::new(HashMap::new()) }
    }

    /// Load `cfg.dir` if it holds an index of the same geometry, else start empty (the note says which).
    pub fn open(cfg: Config) -> io::Result<(HistoryIndex, OpenNote)> {
        let idx = HistoryIndex::new(cfg);
        let p = idx.cfg.dir.join(INDEX_FILE);
        if !p.exists() { return Ok((idx, OpenNote::New)) }
        let note = match idx.load(&p) { Ok(n) => n, Err(e) if e.kind() == io::ErrorKind::InvalidData => OpenNote::Corrupt(e.to_string()), Err(e) => return Err(e) };
        if note != OpenNote::Loaded { idx.clear(); return Ok((idx, note)) }
        if let Ok(m) = read_manifest(&idx.cfg.dir.join(FILES_FILE)) { *lock(&idx.manifest) = m.into_iter().take(idx.cfg.max_files()).collect() }
        Ok((idx, note))
    }

    pub fn config(&self) -> &Config { &self.cfg }
    fn epoch(&self) -> u32 { ((self.tokens.load(Relaxed) >> EPOCH_SHIFT) as u32) & 0x3FFF_FFFF }
    fn loc(&self, k: u64) -> (usize, usize, u32) { ((k >> 58) as usize & (SHARDS - 1), (((k & 0xffff_ffff) * self.nb as u64) >> 32) as usize, ((k >> 26) as u32) | 1) }
    fn clear(&self) {
        for s in self.shards.iter() { lock(s).fill(Slot::default()) }
        for c in [&self.tokens, &self.docs, &self.inserts, &self.evictions] { c.store(0, Relaxed) }
        for f in self.fb.iter() { f[0].store(0, Relaxed); f[1].store(0, Relaxed) }
        lock(&self.manifest).clear();
    }

    // ----- ingest -----
    /// Add one token sequence (a document, or prompt tail + finished reply). n-grams never cross calls.
    pub fn ingest(&self, toks: &[u32]) { self.ingest_with(toks, &|| true); }
    /// `ingest` calling `between()` before each batch of ~4096 observations; false from it stops early (returns false).
    pub fn ingest_with(&self, toks: &[u32], between: &dyn Fn() -> bool) -> bool {
        if toks.len() < 2 { return true }
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(BATCH + MAX_ORDER);
        let mut flushed = 0;
        for i in 1..toks.len() {
            let k = keys(&toks[i.saturating_sub(MAX_ORDER)..i]);
            for &n in &self.orders { if k[n] != 0 { pairs.push((k[n], toks[i])) } }
            if pairs.len() >= BATCH {
                if !between() { return false }
                self.flush(&mut pairs, (i - flushed) as u64); flushed = i;
            }
        }
        if !between() { return false }
        self.flush(&mut pairs, (toks.len() - flushed) as u64);
        self.docs.fetch_add(1, Relaxed);
        true
    }
    fn flush(&self, pairs: &mut Vec<(u64, u32)>, ntok: u64) {
        pairs.sort_by_key(|p| p.0 >> 58); // stable: same order as the text within a shard
        let now = self.epoch();
        let (mut ev, mut i) = (0u64, 0);
        while i < pairs.len() {
            let sh = (pairs[i].0 >> 58) as usize;
            let j = i + pairs[i..].iter().take_while(|p| (p.0 >> 58) as usize == sh).count();
            let mut tab = lock(&self.shards[sh]);
            for &(k, nx) in &pairs[i..j] { let (_, b, c) = self.loc(k); ev += upsert(&mut tab, b, c, nx, now) as u64 }
            drop(tab);
            i = j;
        }
        self.inserts.fetch_add(pairs.len() as u64, Relaxed);
        self.evictions.fetch_add(ev, Relaxed);
        self.tokens.fetch_add(ntok, Relaxed);
        pairs.clear();
    }

    // ----- query -----
    /// Longest indexed order matching the end of `ctx`: (order, slot copy).
    fn lookup(&self, ctx: &[u32]) -> Option<(usize, Slot)> {
        let k = keys(&ctx[ctx.len().saturating_sub(MAX_ORDER)..]);
        for &n in &self.orders {
            if k[n] == 0 { continue }
            let (sh, b, c) = self.loc(k[n]);
            let tab = lock(&self.shards[sh]);
            if let Some(s) = tab[b * WAYS..(b + 1) * WAYS].iter().find(|s| s.check == c && s.cnt[0] > 0) { return Some((n, *s)) }
        }
        None
    }
    /// P(top continuation accepted | match at order n): prior blended with feedback
    pub fn trust(&self, n: usize) -> f64 {
        let n = n.min(MAX_ORDER);
        let (a, r) = (self.fb[n][0].load(Relaxed) as f64, self.fb[n][1].load(Relaxed) as f64);
        (a + PRIOR_W * PRIOR[n]) / (a + r + PRIOR_W)
    }
    /// Verifier result for one drafted token proposed at `order` (DraftRow::order[i]).
    pub fn feedback(&self, order: u8, accepted: bool) {
        if let Some(f) = self.fb.get(order as usize) { f[if accepted { 0 } else { 1 }].fetch_add(1, Relaxed); }
    }
    /// continuations of a slot, best first (count, then most recent): (token, p)
    fn cands(&self, n: usize, s: &Slot) -> ([(u32, f32); CONT], usize) {
        let mut idx = [0usize, 1, 2, 3];
        let last = s.last();
        idx.sort_by(|&a, &b| (s.cnt[b], b == last).cmp(&(s.cnt[a], a == last)));
        let (sum, t) = (s.sum() as f64, self.trust(n));
        let mut out = [(0u32, 0f32); CONT];
        let mut m = 0;
        for &i in &idx { if s.cnt[i] > 0 { out[m] = (s.tok[i], (t * s.cnt[i] as f64 / sum) as f32); m += 1 } }
        (out, m)
    }

    /// Drafts for the context `tail` (the last ≤ 8 tokens are used): ≤ `max_len` tokens per row, ≤ `max_branches` rows.
    pub fn query(&self, tail: &[u32], max_len: usize, max_branches: usize) -> Drafts {
        let mut d = Drafts::default();
        self.query_into(tail, max_len, max_branches, &mut d);
        d
    }
    /// `query` into a reused result (no allocation once the rows have grown).
    pub fn query_into(&self, tail: &[u32], max_len: usize, max_branches: usize, out: &mut Drafts) {
        let (max_len, cap) = (max_len.min(MAX_DRAFT), max_branches.clamp(1, MAX_ROWS));
        let rows = &mut out.rows;
        out.lookups = 0;
        let base = &tail[tail.len().saturating_sub(MAX_ORDER)..];
        let mut alts: BinaryHeap<Alt> = BinaryHeap::new();
        let min = self.cfg.min_conf;
        // rows[used] := rows[parent][..shared] (row objects are reused across calls)
        let start = |rows: &mut Vec<DraftRow>, used: usize, parent: usize, shared: usize| {
            if rows.len() == used { rows.push(DraftRow::default()) }
            let (done, rest) = rows.split_at_mut(used);
            let r = &mut rest[0];
            r.tokens.clear(); r.conf.clear(); r.order.clear();
            if parent < used {
                let p = &done[parent];
                r.tokens.extend_from_slice(&p.tokens[..shared]); r.conf.extend_from_slice(&p.conf[..shared]); r.order.extend_from_slice(&p.order[..shared]);
            }
            r.parent = parent; r.shared = shared;
        };
        let mut used = 0;
        start(rows, 0, 0, 0);
        loop {
            // extend rows[used] greedily, queueing its alternatives
            let row = &mut rows[used];
            let mut buf = [0u32; MAX_ORDER + MAX_DRAFT];
            let nb = base.len();
            buf[..nb].copy_from_slice(base);
            buf[nb..nb + row.tokens.len()].copy_from_slice(&row.tokens);
            let mut len = nb + row.tokens.len();
            let mut cum = row.conf.last().copied().unwrap_or(1.0);
            while row.tokens.len() < max_len && len > 0 {
                out.lookups += 1;
                let Some((n, s)) = self.lookup(&buf[..len]) else { break };
                let (c, m) = self.cands(n, &s);
                let p = cum * c[0].1;
                if p < min { break }
                if cap > 1 {
                    for &(t, q) in &c[1..m] { let sc = cum * q; if sc >= min { alts.push(Alt { score: sc, parent: used, depth: row.tokens.len(), tok: t, order: n as u8 }) } }
                }
                row.tokens.push(c[0].0); row.conf.push(p); row.order.push(n as u8);
                buf[len] = c[0].0; len += 1; cum = p;
            }
            if row.tokens.is_empty() { break }
            used += 1;
            if used >= cap { break }
            let Some(a) = alts.pop() else { break };
            start(rows, used, a.parent, a.depth);
            let r = &mut rows[used];
            r.tokens.push(a.tok); r.conf.push(a.score); r.order.push(a.order);
        }
        rows.truncate(used);
    }

    // ----- text sources -----
    /// Tokenize and ingest one text document.
    pub fn ingest_text(&self, text: &str, enc: &Encoder) -> u64 { let t = enc(text); self.ingest(&t); t.len() as u64 }

    /// Index the text files under `root` (respects .gitignore files and `.git/info/exclude`, skips `.git`, symlinks,
    /// binaries and files over `max_file_bytes`; unchanged files per the manifest are skipped). Multi-threaded; each
    /// worker runs at background QoS. `ctl` pauses / stops it between files and batches.
    pub fn index_dir(&self, root: &Path, enc: &Encoder, ctl: &Control) -> ScanReport {
        let threads = self.cfg.build_threads();
        let (tx, rx) = mpsc::sync_channel::<PathBuf>(256);
        let rx = Mutex::new(rx);
        let total = Mutex::new(ScanReport::default());
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    set_background_qos();
                    let mut rep = ScanReport::default();
                    loop {
                        let next = lock(&rx).recv();
                        let Ok(p) = next else { break };
                        if !ctl.wait() { continue } // stopped: drain so the walker never blocks on a full channel
                        self.index_file(&p, enc, ctl, &mut rep);
                        ctl.files_done.fetch_add(1, Relaxed);
                    }
                    lock(&total).add(&rep);
                });
            }
            let mut rep = ScanReport::default();
            walk(root, self.cfg.max_files().max(1) as u64 * 4, ctl, &mut rep, &mut |p| { ctl.files_found.fetch_add(1, Relaxed); tx.send(p).is_ok() });
            drop(tx);
            lock(&total).add(&rep);
        });
        total.into_inner().unwrap_or_else(|e| e.into_inner())
    }
    fn index_file(&self, p: &Path, enc: &Encoder, ctl: &Control, rep: &mut ScanReport) {
        rep.files_seen += 1;
        let Ok(md) = fs::symlink_metadata(p) else { rep.errors += 1; return };
        if md.len() > self.cfg.max_file_bytes { rep.skipped_large += 1; return }
        let key = hash_bytes(p.as_os_str().as_encoded_bytes());
        let val = file_stamp(&md);
        if lock(&self.manifest).get(&key) == Some(&val) { rep.files_unchanged += 1; return }
        let Ok(bytes) = fs::read(p) else { rep.errors += 1; return };
        if bytes[..bytes.len().min(8000)].contains(&0) { rep.skipped_binary += 1; return }
        let Ok(text) = std::str::from_utf8(&bytes) else { rep.skipped_binary += 1; return };
        let toks = enc(text);
        drop(bytes);
        if !self.ingest_with(&toks, &|| ctl.wait()) { return }
        rep.files_indexed += 1; rep.bytes += md.len(); rep.tokens += toks.len() as u64;
        ctl.bytes_done.fetch_add(md.len(), Relaxed); ctl.tokens_done.fetch_add(toks.len() as u64, Relaxed);
        let mut m = lock(&self.manifest);
        if m.len() < self.cfg.max_files() || m.contains_key(&key) { m.insert(key, val); }
    }

    /// Index session logs: one `.json`/`.jsonl` file, or every such file under a dir (a client's saved
    /// sessions, gateway logs). Every text string value (not ids/types/roles) is one document. Sequential, background QoS
    /// is the caller's thread's.
    pub fn index_sessions(&self, path: &Path, enc: &Encoder, ctl: &Control) -> ScanReport {
        let mut rep = ScanReport::default();
        let mut files = Vec::new();
        if path.is_dir() {
            walk(path, self.cfg.max_files() as u64, ctl, &mut rep, &mut |p| {
                if p.extension().is_some_and(|e| e == "json" || e == "jsonl") { files.push(p) } true
            });
        } else { files.push(path.to_path_buf()) }
        for p in files {
            if !ctl.wait() { break }
            rep.files_seen += 1;
            let Ok(md) = fs::metadata(&p) else { rep.errors += 1; continue };
            if md.len() > self.cfg.max_session_bytes() { rep.skipped_large += 1; continue }
            let key = hash_bytes(p.as_os_str().as_encoded_bytes());
            let val = file_stamp(&md);
            if lock(&self.manifest).get(&key) == Some(&val) { rep.files_unchanged += 1; continue }
            let Ok(text) = fs::read_to_string(&p) else { rep.skipped_binary += 1; continue };
            let mut docs: Vec<String> = Vec::new();
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) => collect_strings(&v, &mut docs),
                Err(_) => for line in text.lines() { if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) { collect_strings(&v, &mut docs) } },
            }
            drop(text);
            let mut ntok = 0u64;
            for d in &docs { let t = enc(d); ntok += t.len() as u64; if !self.ingest_with(&t, &|| ctl.wait()) { return rep } }
            rep.files_indexed += 1; rep.bytes += md.len(); rep.tokens += ntok;
            ctl.files_done.fetch_add(1, Relaxed); ctl.tokens_done.fetch_add(ntok, Relaxed);
            let mut m = lock(&self.manifest);
            if m.len() < self.cfg.max_files() || m.contains_key(&key) { m.insert(key, val); }
        }
        rep
    }

    // ----- stats / persistence -----
    pub fn stats(&self) -> Stats {
        let used = self.shards.iter().map(|s| lock(s).iter().filter(|x| x.check != 0).count()).sum();
        let files = lock(&self.manifest).len();
        let table = self.nb * SHARDS * WAYS * SLOT_BYTES;
        Stats {
            budget_bytes: self.cfg.budget_bytes, table_bytes: table, ram_bytes: table + files * MANIFEST_ENTRY, slots: self.nb * SHARDS * WAYS,
            used_slots: used, tokens: self.tokens.load(Relaxed), docs: self.docs.load(Relaxed), inserts: self.inserts.load(Relaxed),
            evictions: self.evictions.load(Relaxed), files_tracked: files, orders: self.orders.clone(),
            feedback: self.fb.iter().map(|f| (f[0].load(Relaxed), f[1].load(Relaxed))).collect(),
        }
    }

    /// Write `index.sdh` + `files.bin` into `cfg.dir` (tmp + rename; shards locked one at a time). Returns bytes written.
    pub fn save(&self) -> io::Result<u64> {
        fs::create_dir_all(&self.cfg.dir)?;
        let p = self.cfg.dir.join(INDEX_FILE);
        let tmp = p.with_extension("sdh.tmp");
        let mut w = BufWriter::with_capacity(64 << 10, File::create(&tmp)?);
        w.write_all(&[0u8; HDR])?;
        let mut crc = 0u32;
        for s in self.shards.iter() {
            let tab = lock(s);
            for x in tab.iter() { let b = x.to_bytes(); crc = crc_up(crc, &b); w.write_all(&b)? }
        }
        let f = w.into_inner().map_err(|e| e.into_error())?;
        let hdr = self.header(crc);
        use std::os::unix::fs::FileExt;
        f.write_all_at(&hdr, 0)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, &p)?;
        let m: Vec<(u64, u64)> = lock(&self.manifest).iter().map(|(&k, &v)| (k, v)).collect();
        write_manifest(&self.cfg.dir.join(FILES_FILE), &m)?;
        Ok((HDR + self.nb * SHARDS * WAYS * SLOT_BYTES + 20 + m.len() * 16) as u64)
    }
    fn header(&self, crc: u32) -> Vec<u8> {
        let mut h = Vec::with_capacity(HDR);
        h.extend_from_slice(&MAGIC.to_le_bytes());
        for v in [VERSION, SHARDS as u32, self.nb as u32, WAYS as u32, CONT as u32, self.orders_mask(), crc] { h.extend_from_slice(&v.to_le_bytes()) }
        for c in [&self.tokens, &self.docs, &self.inserts, &self.evictions] { h.extend_from_slice(&c.load(Relaxed).to_le_bytes()) }
        for f in self.fb.iter() { h.extend_from_slice(&f[0].load(Relaxed).to_le_bytes()); h.extend_from_slice(&f[1].load(Relaxed).to_le_bytes()) }
        let hc = crc_up(0, &h);
        h.extend_from_slice(&hc.to_le_bytes());
        h.resize(HDR, 0);
        h
    }
    fn orders_mask(&self) -> u32 { self.orders.iter().fold(0, |m, &n| m | 1 << n) }
    fn load(&self, p: &Path) -> io::Result<OpenNote> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {}", p.display(), m));
        let mut r = BufReader::with_capacity(64 << 10, File::open(p)?);
        let mut h = vec![0u8; HDR];
        r.read_exact(&mut h).map_err(|_| bad("short header"))?;
        let u32a = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let u64a = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        if u64a(0) != MAGIC { return Err(bad("bad magic")) }
        let hlen = 8 + 7 * 4 + 4 * 8 + (MAX_ORDER + 1) * 8;
        if crc_up(0, &h[..hlen]) != u32a(hlen) { return Err(bad("header crc")) }
        if u32a(8) != VERSION { return Err(bad("version")) }
        if u32a(12) != SHARDS as u32 || u32a(16) != self.nb as u32 || u32a(20) != WAYS as u32 || u32a(24) != CONT as u32 || u32a(28) != self.orders_mask() {
            return Ok(OpenNote::GeometryChanged)
        }
        let want = u32a(32);
        let mut crc = 0u32;
        let mut b = [0u8; SLOT_BYTES];
        for s in self.shards.iter() {
            let mut tab = lock(s);
            for x in tab.iter_mut() { r.read_exact(&mut b).map_err(|_| bad("short body"))?; crc = crc_up(crc, &b); *x = Slot::from_bytes(&b) }
        }
        if crc != want { return Err(bad("body crc")) }
        for (i, c) in [&self.tokens, &self.docs, &self.inserts, &self.evictions].iter().enumerate() { c.store(u64a(36 + 8 * i), Relaxed) }
        for (i, f) in self.fb.iter().enumerate() { f[0].store(u32a(68 + 8 * i), Relaxed); f[1].store(u32a(72 + 8 * i), Relaxed) }
        Ok(OpenNote::Loaded)
    }
}

#[derive(Clone, Copy, Debug)]
struct Alt { score: f32, parent: usize, depth: usize, tok: u32, order: u8 }
impl PartialEq for Alt { fn eq(&self, o: &Self) -> bool { self.cmp(o) == std::cmp::Ordering::Equal } }
impl Eq for Alt {}
impl PartialOrd for Alt { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Alt {
    // highest score first; ties: earlier row, shallower, smaller token (deterministic)
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&o.score).then(o.parent.cmp(&self.parent)).then(o.depth.cmp(&self.depth)).then(o.tok.cmp(&self.tok))
    }
}

fn file_stamp(md: &fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    fin(mix(mix(SEED, md.len() as u32 ^ (md.len() >> 32) as u32), md.mtime() as u32) ^ md.mtime_nsec() as u64, 0)
}
fn write_manifest(p: &Path, m: &[(u64, u64)]) -> io::Result<()> {
    let tmp = p.with_extension("bin.tmp");
    let mut w = BufWriter::new(File::create(&tmp)?);
    let mut body = Vec::with_capacity(m.len() * 16);
    for &(k, v) in m { body.extend_from_slice(&k.to_le_bytes()); body.extend_from_slice(&v.to_le_bytes()) }
    w.write_all(&FILES_MAGIC.to_le_bytes())?; w.write_all(&(m.len() as u64).to_le_bytes())?; w.write_all(&crc_up(0, &body).to_le_bytes())?; w.write_all(&body)?;
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    fs::rename(&tmp, p)
}
fn read_manifest(p: &Path) -> io::Result<Vec<(u64, u64)>> {
    let b = fs::read(p)?;
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "manifest");
    if b.len() < 20 || u64::from_le_bytes(b[0..8].try_into().unwrap()) != FILES_MAGIC { return Err(bad()) }
    let n = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    let body = &b[20..];
    if body.len() != n * 16 || crc_up(0, body) != u32::from_le_bytes(b[16..20].try_into().unwrap()) { return Err(bad()) }
    Ok(body.chunks_exact(16).map(|c| (u64::from_le_bytes(c[0..8].try_into().unwrap()), u64::from_le_bytes(c[8..16].try_into().unwrap()))).collect())
}
fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) if s.len() >= 2 => out.push(s.clone()),
        serde_json::Value::Array(a) => for x in a { collect_strings(x, out) },
        serde_json::Value::Object(o) => for (k, x) in o { if !SKIP_KEYS.contains(&k.as_str()) { collect_strings(x, out) } },
        _ => {}
    }
}

/// This thread → background QoS (efficiency cores on Apple silicon). Ok = the call succeeded.
pub fn set_background_qos() -> bool {
    #[cfg(target_os = "macos")]
    { unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0) == 0 } }
    #[cfg(not(target_os = "macos"))]
    { false }
}

// ---------- .gitignore walk ----------
#[derive(Clone, Debug)]
struct Rule { pat: Vec<u8>, neg: bool, dir_only: bool, anchored: bool }
fn parse_ignore(text: &str) -> Vec<Rule> {
    let mut v = Vec::new();
    for l in text.lines() {
        let l = l.trim_end();
        if l.is_empty() || l.starts_with('#') { continue }
        let (neg, l) = match l.strip_prefix('!') { Some(r) => (true, r), None => (false, l.strip_prefix('\\').unwrap_or(l)) };
        let (dir_only, l) = match l.strip_suffix('/') { Some(r) => (true, r), None => (false, l) };
        if l.is_empty() { continue }
        let anchored = l.contains('/');
        v.push(Rule { pat: l.trim_start_matches('/').as_bytes().to_vec(), neg, dir_only, anchored });
    }
    v
}
/// gitignore glob: `*` `?` `[..]` within a component, `**` across components (`a/**/b` also matches `a/b`)
fn glob(p: &[u8], s: &[u8]) -> bool {
    match p.first() {
        None => s.is_empty(),
        Some(b'*') if p.get(1) == Some(&b'*') => {
            let rest = &p[2..];
            if let Some(r) = rest.strip_prefix(b"/") {
                if glob(r, s) { return true }
                (0..s.len()).any(|i| s[i] == b'/' && glob(r, &s[i + 1..]))
            } else { (0..=s.len()).any(|i| glob(rest, &s[i..])) }
        }
        Some(b'*') => (0..=s.len()).take_while(|&i| i == 0 || s[i - 1] != b'/').any(|i| glob(&p[1..], &s[i..])),
        Some(b'?') => !s.is_empty() && s[0] != b'/' && glob(&p[1..], &s[1..]),
        Some(b'[') => {
            let Some(end) = p[1..].iter().position(|&c| c == b']').map(|e| e + 1) else { return !s.is_empty() && s[0] == b'[' && glob(&p[1..], &s[1..]) };
            if s.is_empty() || s[0] == b'/' { return false }
            let set = &p[1..end];
            let (neg, set) = match set.first() { Some(b'!') | Some(b'^') => (true, &set[1..]), _ => (false, set) };
            let mut hit = false;
            let mut i = 0;
            while i < set.len() {
                if i + 2 < set.len() && set[i + 1] == b'-' { hit |= set[i] <= s[0] && s[0] <= set[i + 2]; i += 3 } else { hit |= set[i] == s[0]; i += 1 }
            }
            hit != neg && glob(&p[end + 1..], &s[1..])
        }
        Some(&c) => !s.is_empty() && s[0] == c && glob(&p[1..], &s[1..]),
    }
}
/// stack of (dir relative to root with trailing '/', rules); last matching rule wins
fn ignored(stack: &[(Vec<u8>, Vec<Rule>)], rel: &[u8], is_dir: bool) -> bool {
    let name = rel.rsplit(|&c| c == b'/').next().unwrap_or(rel);
    let mut out = false;
    for (base, rules) in stack {
        let Some(sub) = rel.strip_prefix(base.as_slice()) else { continue };
        for r in rules {
            if r.dir_only && !is_dir { continue }
            if glob(&r.pat, if r.anchored { sub } else { name }) { out = !r.neg }
        }
    }
    out
}
/// Depth-first walk of `root` in name order: calls `f` for every non-ignored regular file (stops when `f` returns
/// false, after `limit` files, or on `ctl.stop`). Skips `.git`, symlinks, BIN_EXT extensions.
fn walk(root: &Path, limit: u64, ctl: &Control, rep: &mut ScanReport, f: &mut dyn FnMut(PathBuf) -> bool) {
    let mut root_rules = fs::read_to_string(root.join(".git/info/exclude")).map(|t| parse_ignore(&t)).unwrap_or_default();
    root_rules.extend(fs::read_to_string(root.join(".gitignore")).map(|t| parse_ignore(&t)).unwrap_or_default());
    let mut stack: Vec<(Vec<u8>, Vec<Rule>)> = vec![(Vec::new(), root_rules)];
    let mut n = 0u64;
    fn rec(dir: &Path, rel: &[u8], stack: &mut Vec<(Vec<u8>, Vec<Rule>)>, n: &mut u64, limit: u64, ctl: &Control, rep: &mut ScanReport, f: &mut dyn FnMut(PathBuf) -> bool) -> bool {
        let Ok(rd) = fs::read_dir(dir) else { rep.errors += 1; return true };
        let mut ents: Vec<(Vec<u8>, PathBuf, fs::FileType)> = rd.filter_map(|e| e.ok()).filter_map(|e| Some((e.file_name().as_encoded_bytes().to_vec(), e.path(), e.file_type().ok()?))).collect();
        ents.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, path, ft) in ents {
            if ctl.stopped() || *n >= limit { return false }
            if ft.is_symlink() || name == b".git" { continue }
            let mut r = rel.to_vec(); r.extend_from_slice(&name);
            if ignored(stack, &r, ft.is_dir()) { rep.skipped_ignored += 1; continue }
            if ft.is_dir() {
                r.push(b'/');
                let rules = fs::read_to_string(path.join(".gitignore")).map(|t| parse_ignore(&t)).unwrap_or_default();
                let pushed = !rules.is_empty();
                if pushed { stack.push((r.clone(), rules)) }
                let go = rec(&path, &r, stack, n, limit, ctl, rep, f);
                if pushed { stack.pop(); }
                if !go { return false }
            } else if ft.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
                if ext.is_some_and(|e| BIN_EXT.contains(&e.as_str())) { rep.skipped_binary += 1; continue }
                *n += 1;
                if !f(path) { return false }
            }
        }
        true
    }
    rec(root, b"", &mut stack, &mut n, limit, ctl, rep, f);
}

// ---------- background updater ----------
#[derive(Clone, Debug)]
pub struct UpdaterOptions {
    /// save to `cfg.dir` after a scan job and once this many tokens were appended since the last save (0 = never)
    pub autosave_tokens: u64,
    /// set the updater thread to background QoS (macOS) / nice 19
    pub background: bool,
}
impl Default for UpdaterOptions { fn default() -> Self { UpdaterOptions { autosave_tokens: 1 << 16, background: true } } }

enum Job { Tokens(Vec<u32>), Text(String), Dir(PathBuf), Sessions(PathBuf), Save }

/// One thread that applies index updates in order, at background QoS. Dropping it stops the thread (the current
/// file / batch finishes; queued jobs are discarded) — call `save` + `wait_idle` first to keep them.
pub struct Updater {
    tx: Option<mpsc::Sender<Job>>,
    ctl: Arc<Control>,
    queued_bytes: Arc<AtomicUsize>,
    submitted: AtomicU64,
    done: Arc<(Mutex<u64>, Condvar)>,
    dropped: AtomicU64,
    last: Arc<Mutex<ScanReport>>,
    last_error: Arc<Mutex<Option<String>>>,
    queue_cap: usize,
    handle: Option<std::thread::JoinHandle<()>>,
}
impl Updater {
    pub fn spawn(index: Arc<HistoryIndex>, enc: Encoder, opt: UpdaterOptions) -> io::Result<Updater> {
        let (tx, rx) = mpsc::channel::<Job>();
        let ctl = Arc::new(Control::default());
        let queued_bytes = Arc::new(AtomicUsize::new(0));
        let done = Arc::new((Mutex::new(0u64), Condvar::new()));
        let last = Arc::new(Mutex::new(ScanReport::default()));
        let last_error = Arc::new(Mutex::new(None));
        let queue_cap = index.cfg.budget_bytes.max(MIN_BUDGET) / 16;
        let (c, q, d, l, le) = (ctl.clone(), queued_bytes.clone(), done.clone(), last.clone(), last_error.clone());
        let handle = std::thread::Builder::new().name("sd-history-index".into()).spawn(move || {
            if opt.background { set_background_qos(); }
            let mut since_save = 0u64;
            let save = |since: &mut u64| { if let Err(e) = index.save() { *lock(&le) = Some(e.to_string()) } *since = 0 };
            while let Ok(job) = rx.recv() {
                if c.stopped() || !c.wait() { break }
                match job {
                    Job::Tokens(t) => {
                        q.fetch_sub(t.len() * 4, Relaxed);
                        index.ingest_with(&t, &|| c.wait());
                        since_save += t.len() as u64;
                        if opt.autosave_tokens > 0 && since_save >= opt.autosave_tokens { save(&mut since_save) }
                    }
                    Job::Text(s) => {
                        q.fetch_sub(s.len(), Relaxed);
                        let t = enc(&s);
                        index.ingest_with(&t, &|| c.wait());
                        since_save += t.len() as u64;
                        if opt.autosave_tokens > 0 && since_save >= opt.autosave_tokens { save(&mut since_save) }
                    }
                    Job::Dir(p) => { let r = index.index_dir(&p, &enc, &c); *lock(&l) = r; if opt.autosave_tokens > 0 { save(&mut since_save) } }
                    Job::Sessions(p) => { let r = index.index_sessions(&p, &enc, &c); *lock(&l) = r; if opt.autosave_tokens > 0 { save(&mut since_save) } }
                    Job::Save => save(&mut since_save),
                }
                let (m, cv) = &*d;
                *lock(m) += 1; cv.notify_all();
            }
        })?;
        Ok(Updater { tx: Some(tx), ctl, queued_bytes, submitted: AtomicU64::new(0), done, dropped: AtomicU64::new(0), last, last_error, queue_cap, handle: Some(handle) })
    }
    fn send(&self, j: Job, bytes: usize) -> bool {
        if bytes > 0 && self.queued_bytes.load(Relaxed) + bytes > self.queue_cap { self.dropped.fetch_add(1, Relaxed); return false }
        self.queued_bytes.fetch_add(bytes, Relaxed);
        let ok = self.tx.as_ref().is_some_and(|t| t.send(j).is_ok());
        if ok { self.submitted.fetch_add(1, Relaxed); } else { self.queued_bytes.fetch_sub(bytes, Relaxed); }
        ok
    }
    /// Append a finished reply (pass the prompt tail + reply as one sequence so its first n-grams have context).
    /// False = dropped (queue over budget/16 or updater gone).
    pub fn push_tokens(&self, toks: Vec<u32>) -> bool { let b = toks.len() * 4; self.send(Job::Tokens(toks), b) }
    pub fn push_text(&self, s: String) -> bool { let b = s.len(); self.send(Job::Text(s), b) }
    /// Index (incrementally) the project dir / session logs in the background.
    pub fn scan_dir(&self, p: PathBuf) -> bool { self.send(Job::Dir(p), 0) }
    pub fn scan_sessions(&self, p: PathBuf) -> bool { self.send(Job::Sessions(p), 0) }
    pub fn save(&self) -> bool { self.send(Job::Save, 0) }
    /// Pause at the next batch / file boundary (e.g. while the GPU decodes); `resume` continues.
    pub fn pause(&self) { self.ctl.pause() }
    pub fn resume(&self) { self.ctl.resume() }
    pub fn is_paused(&self) -> bool { self.ctl.is_paused() }
    pub fn control(&self) -> &Control { &self.ctl }
    /// Jobs submitted and not yet finished.
    pub fn pending(&self) -> u64 { self.submitted.load(Relaxed) - *lock(&self.done.0) }
    pub fn dropped(&self) -> u64 { self.dropped.load(Relaxed) }
    pub fn last_scan(&self) -> ScanReport { lock(&self.last).clone() }
    pub fn last_error(&self) -> Option<String> { lock(&self.last_error).clone() }
    /// Wait until every submitted job is done (false on timeout).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let want = self.submitted.load(Relaxed);
        let (m, cv) = &*self.done;
        let g = lock(m);
        let (g, _) = cv.wait_timeout_while(g, timeout, |d| *d < want).unwrap_or_else(|e| e.into_inner());
        *g >= want
    }
}
impl Drop for Updater {
    fn drop(&mut self) {
        self.ctl.stop();
        self.tx.take();
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn tmp(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_history_{}_{}", n, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }
    fn cfg(dir: &Path, mb: f64) -> Config { Config { budget_bytes: (mb * 1048576.0) as usize, threads: 2, ..Config::with_dir(dir) } }
    /// toy encoder: one id per whitespace-separated word (stable hash)
    fn toy() -> Encoder { Arc::new(|s: &str| s.split_whitespace().map(|w| (hash_bytes(w.as_bytes()) % 50_000) as u32 + 10).collect()) }
    /// xorshift tokens in [lo, lo + span)
    fn noise(seed: u64, n: usize, lo: u32, span: u32) -> Vec<u32> {
        let mut x = seed | 1;
        (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; lo + (x % span as u64) as u32 }).collect()
    }

    #[test]
    fn deterministic_chain() {
        let d = tmp("chain");
        let idx = HistoryIndex::new(cfg(&d, 1.0));
        let seq: Vec<u32> = (100..140).collect();
        idx.ingest(&seq);
        let q = idx.query(&[103, 104, 105], 8, 1);
        assert_eq!(q.rows.len(), 1);
        assert_eq!(q.best(), &[106, 107, 108, 109, 110, 111, 112, 113]);
        let r = &q.rows[0];
        assert!(r.conf.windows(2).all(|w| w[1] <= w[0]) && r.conf[0] > 0.0 && r.conf[0] <= 1.0, "{:?}", r.conf);
        assert_eq!(r.order[0], 3); // tail of 3 → order 3; then grows to 4, 6, 8
        assert_eq!(r.order[7], 8);
        assert_eq!(q, idx.query(&[103, 104, 105], 8, 1));
        // max_len and end of the text
        assert_eq!(idx.query(&[136, 137], 8, 1).best(), &[138, 139]);
        assert_eq!(idx.query(&[103, 104, 105], 3, 1).best(), &[106, 107, 108]);
        // unseen context
        assert!(idx.query(&[7, 8, 9], 8, 4).rows.is_empty());
        // n-grams don't cross documents
        idx.ingest(&[1, 2]); idx.ingest(&[3, 4]);
        assert!(idx.query(&[1, 2], 4, 1).rows.is_empty());
    }

    #[test]
    fn branches_tree() {
        let d = tmp("tree");
        let idx = HistoryIndex::new(cfg(&d, 1.0));
        for _ in 0..3 { idx.ingest(&[1, 2, 3, 4, 5, 6, 7]) }
        idx.ingest(&[1, 2, 3, 4, 9, 10, 11]);
        let q = idx.query(&[2, 3, 4], 3, 4);
        assert_eq!(q.rows[0].tokens, vec![5, 6, 7]);
        assert_eq!(q.rows.len(), 2);
        let r = &q.rows[1];
        assert_eq!((r.parent, r.shared), (0, 0));
        assert_eq!(r.tokens[..2], [9, 10]);
        assert!(r.conf[0] < q.rows[0].conf[0]);
        // one row only when asked
        assert_eq!(idx.query(&[2, 3, 4], 3, 1).rows.len(), 1);
        // recency breaks a count tie
        let idx2 = HistoryIndex::new(cfg(&d, 1.0));
        idx2.ingest(&[1, 2, 3, 50]); idx2.ingest(&[1, 2, 3, 60]);
        assert_eq!(idx2.query(&[1, 2, 3], 1, 1).best(), &[60]);
        idx2.ingest(&[1, 2, 3, 50]);
        assert_eq!(idx2.query(&[1, 2, 3], 1, 1).best(), &[50]);
    }

    #[test]
    fn feedback_moves_trust() {
        let idx = HistoryIndex::new(cfg(&tmp("fb"), 0.5));
        let t0 = idx.trust(4);
        for _ in 0..100 { idx.feedback(4, false) }
        assert!(idx.trust(4) < t0 * 0.3);
        for _ in 0..1000 { idx.feedback(4, true) }
        assert!(idx.trust(4) > 0.8);
    }

    #[test]
    fn memory_bound_and_eviction() {
        let d = tmp("mem");
        let c = cfg(&d, 1.0);
        let idx = HistoryIndex::new(c.clone());
        let st0 = idx.stats();
        assert!(st0.table_bytes <= c.budget_bytes * 3 / 4 && st0.table_bytes > c.budget_bytes / 2);
        // frequent pattern early, then a flood of unique n-grams far over capacity
        for _ in 0..200 { idx.ingest(&[7, 8, 9, 10, 11, 12]) }
        for s in 0..40 { idx.ingest(&noise(s + 1, 10_000, 1000, 1_000_000)) }
        let recent = noise(999, 64, 1000, 1_000_000);
        idx.ingest(&recent);
        let st = idx.stats();
        assert_eq!(st.table_bytes, st0.table_bytes);
        assert!(st.ram_bytes <= c.budget_bytes, "{} > {}", st.ram_bytes, c.budget_bytes);
        assert!(st.evictions > 0 && st.inserts > st.slots as u64 * 2, "{:?}", st);
        assert_eq!(idx.query(&[8, 9, 10], 2, 1).best(), &[11, 12], "frequent survives");
        assert_eq!(idx.query(&recent[20..28], 4, 1).best(), &recent[28..32], "recent survives");
    }

    #[test]
    fn persistence_round_trip() {
        let d = tmp("persist");
        let c = cfg(&d, 0.5);
        let idx = HistoryIndex::new(c.clone());
        idx.ingest(&(0..500).map(|i| i * 7 % 101).collect::<Vec<u32>>());
        idx.ingest(&noise(5, 5000, 200, 300)); // = corpus below
        idx.feedback(6, true); idx.feedback(2, false);
        lock(&idx.manifest).insert(42, 43);
        let corpus = noise(5, 5000, 200, 300);
        let mut tails: Vec<Vec<u32>> = (0..25).map(|i| noise(i + 77, 6, 200, 300)).collect();
        tails.extend((0..25).map(|i| corpus[i * 150..i * 150 + 6].to_vec()));
        let before: Vec<Drafts> = tails.iter().map(|t| idx.query(t, 8, 4)).collect();
        assert!(idx.save().unwrap() > 0);
        let (b, note) = HistoryIndex::open(c.clone()).unwrap();
        assert_eq!(note, OpenNote::Loaded);
        assert_eq!(b.stats(), idx.stats());
        for (t, q) in tails.iter().zip(&before) { assert_eq!(&b.query(t, 8, 4), q) }
        assert!(before.iter().any(|q| !q.rows.is_empty()));
        // another budget = other geometry → empty
        let (g, n) = HistoryIndex::open(cfg(&d, 1.0)).unwrap();
        assert_eq!((n, g.stats().used_slots), (OpenNote::GeometryChanged, 0));
        // flipped body byte → corrupt → empty
        let p = d.join(INDEX_FILE);
        let mut bytes = fs::read(&p).unwrap();
        let k = HDR + 1000; bytes[k] ^= 0x55; fs::write(&p, &bytes).unwrap();
        let (e, n) = HistoryIndex::open(c).unwrap();
        assert!(matches!(n, OpenNote::Corrupt(_)) && e.stats().used_slots == 0 && e.stats().tokens == 0, "{:?}", n);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn globs() {
        assert!(glob(b"*.log", b"a.log") && !glob(b"*.log", b"a.logx") && !glob(b"*.log", b"d/a.log"));
        assert!(glob(b"**/foo", b"foo") && glob(b"**/foo", b"a/b/foo") && glob(b"a/**/b", b"a/b") && glob(b"a/**/b", b"a/x/y/b"));
        assert!(glob(b"a/**", b"a/x/y") && glob(b"file?.[ch]", b"file1.c") && !glob(b"file?.[ch]", b"file1.o") && glob(b"[!a]x", b"bx"));
        let st = vec![(Vec::new(), parse_ignore("target/\n*.log\n!keep.log\n/root_only.txt\n# c\n"))];
        assert!(ignored(&st, b"target", true) && !ignored(&st, b"target", false) && ignored(&st, b"x/target", true));
        assert!(ignored(&st, b"a.log", false) && !ignored(&st, b"keep.log", false) && ignored(&st, b"root_only.txt", false) && !ignored(&st, b"s/root_only.txt", false));
    }

    #[test]
    fn index_dir_gitignore_incremental() {
        let d = tmp("dir");
        let proj = d.join("proj");
        let w = |p: &str, s: &[u8]| { let f = proj.join(p); fs::create_dir_all(f.parent().unwrap()).unwrap(); fs::write(f, s).unwrap() };
        w(".gitignore", b"target/\n*.log\n!keep.log\n");
        w("src/main.rs", b"fn main ( ) { println ! ( hello world ) ; }");
        w("src/lib.rs", b"pub fn alpha beta gamma delta epsilon zeta");
        w("keep.log", b"kept log text one two three");
        w("drop.log", b"dropped log text");
        w("target/out.txt", b"build output");
        w("sub/.gitignore", b"secret.txt\n");
        w("sub/secret.txt", b"secret words here");
        w("sub/ok.md", b"ok words here and there");
        w("img.png", b"\x89PNG");
        w("blob.dat", b"abc\x00def");
        w("big.txt", &vec![b'a'; 300 << 10]);
        w(".git/config", b"[core]");
        let mut c = cfg(&d.join("idx"), 1.0);
        c.max_file_bytes = 64 << 10;
        let idx = HistoryIndex::new(c.clone());
        let ctl = Control::default();
        let r = idx.index_dir(&proj, &toy(), &ctl);
        // indexed: .gitignore, src/main.rs, src/lib.rs, keep.log, sub/.gitignore, sub/ok.md
        assert_eq!(r.files_indexed, 6, "{:?}", r);
        assert_eq!((r.skipped_binary, r.skipped_large), (2, 1), "{:?}", r);
        assert_eq!(r.skipped_ignored, 3, "{:?}", r); // drop.log, target/, sub/secret.txt
        let enc = toy();
        let q = idx.query(&enc("alpha beta gamma"), 3, 1);
        assert_eq!(q.best(), enc("delta epsilon zeta").as_slice());
        assert!(idx.query(&enc("secret words"), 1, 1).rows.is_empty());
        // incremental: second scan skips unchanged; a changed file is re-indexed
        let r2 = idx.index_dir(&proj, &toy(), &ctl);
        assert_eq!((r2.files_indexed, r2.files_unchanged), (0, 6), "{:?}", r2);
        std::thread::sleep(Duration::from_millis(20));
        w("src/lib.rs", b"pub fn alpha beta gamma omega psi");
        let r3 = idx.index_dir(&proj, &toy(), &ctl);
        assert_eq!((r3.files_indexed, r3.files_unchanged), (1, 5), "{:?}", r3);
        // manifest persists
        idx.save().unwrap();
        let (b, _) = HistoryIndex::open(c).unwrap();
        assert_eq!(b.index_dir(&proj, &toy(), &ctl).files_unchanged, 6);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn sessions_json_and_jsonl() {
        let d = tmp("sess");
        let s = d.join("sessions");
        fs::create_dir_all(&s).unwrap();
        fs::write(s.join("a.json"), r#"[{"role":"user","content":[{"type":"text","text":"please refactor the parser module now"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"edit","input":{"path":"src/p.rs","new":"fn parse tokens into tree"}}]}]"#).unwrap();
        fs::write(s.join("b.jsonl"), "{\"role\":\"assistant\",\"text\":\"the cache key is the prompt hash\"}\nnot json\n").unwrap();
        fs::write(s.join("c.txt"), "ignored entirely here").unwrap();
        let idx = HistoryIndex::new(cfg(&d.join("idx"), 1.0));
        let r = idx.index_sessions(&s, &toy(), &Control::default());
        assert_eq!(r.files_indexed, 2, "{:?}", r);
        let enc = toy();
        assert_eq!(idx.query(&enc("refactor the parser"), 2, 1).best(), enc("module now").as_slice());
        assert_eq!(idx.query(&enc("parse tokens"), 2, 1).best(), enc("into tree").as_slice());
        assert_eq!(idx.query(&enc("cache key is"), 3, 1).best(), enc("the prompt hash").as_slice());
        assert!(idx.query(&enc("ignored entirely"), 1, 1).rows.is_empty());
        assert_eq!(idx.index_sessions(&s, &toy(), &Control::default()).files_unchanged, 2);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn updater_incremental_pause_save() {
        let d = tmp("upd");
        let c = cfg(&d, 1.0);
        let idx = Arc::new(HistoryIndex::new(c.clone()));
        let up = Updater::spawn(idx.clone(), toy(), UpdaterOptions { autosave_tokens: 0, background: true }).unwrap();
        assert!(up.push_tokens(vec![5, 6, 7, 8, 9]));
        assert!(up.wait_idle(Duration::from_secs(10)));
        assert_eq!(idx.query(&[5, 6], 3, 1).best(), &[7, 8, 9]);
        up.pause();
        assert!(up.push_tokens(vec![20, 21, 22, 23]));
        assert!(!up.wait_idle(Duration::from_millis(100)));
        assert!(idx.query(&[20, 21], 2, 1).rows.is_empty(), "paused: not applied");
        assert_eq!(up.pending(), 1);
        up.resume();
        assert!(up.wait_idle(Duration::from_secs(10)));
        assert_eq!(idx.query(&[20, 21], 2, 1).best(), &[22, 23]);
        // earlier data still there; text + save
        assert_eq!(idx.query(&[5, 6], 3, 1).best(), &[7, 8, 9]);
        assert!(up.push_text("one two three four".into()));
        assert!(up.save());
        assert!(up.wait_idle(Duration::from_secs(10)));
        assert_eq!(up.last_error(), None);
        let (b, n) = HistoryIndex::open(c.clone()).unwrap();
        assert_eq!(n, OpenNote::Loaded);
        let enc = toy();
        assert_eq!(b.query(&enc("one two"), 2, 1).best(), enc("three four").as_slice());
        // queue bound: a push larger than budget/16 is dropped
        assert!(!up.push_tokens(vec![1; c.budget_bytes / 16 / 4 + 1]));
        assert_eq!(up.dropped(), 1);
        drop(up);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn parallel_build_is_complete() {
        // same result set (queries) regardless of thread count when nothing is evicted
        let d = tmp("par");
        let proj = d.join("p");
        fs::create_dir_all(&proj).unwrap();
        for i in 0..40 { fs::write(proj.join(format!("f{:02}.txt", i)), format!("file {} says w{} x{} y{} z{}", i, i, i * 2, i * 3, i * 5)).unwrap() }
        let enc = toy();
        let mut res = Vec::new();
        for th in [1, 4] {
            let mut c = cfg(&d.join(format!("i{}", th)), 2.0); c.threads = th;
            let idx = HistoryIndex::new(c);
            let r = idx.index_dir(&proj, &enc, &Control::default());
            assert_eq!(r.files_indexed, 40);
            res.push((0..40).map(|i| idx.query(&enc(&format!("file {} says", i)), 4, 1).best().to_vec()).collect::<Vec<_>>());
        }
        assert_eq!(res[0], res[1]);
        assert_eq!(res[0][3], enc("w3 x6 y9 z15"));
        let _ = fs::remove_dir_all(&d);
    }

    /// cargo test --release -p streamdense-core history_index::tests::latency -- --ignored --nocapture
    #[test]
    #[ignore]
    fn latency() {
        let d = tmp("lat");
        let idx = HistoryIndex::new(Config { budget_bytes: DEFAULT_BUDGET, ..Config::with_dir(&d) });
        // code-like synthetic corpus: 4M tokens, 2k "lines" templates with random tokens mixed in
        let lines: Vec<Vec<u32>> = (0..2000).map(|i| noise(i + 1, 12, 100, 20_000)).collect();
        let pick = noise(42, 400_000, 0, 2000);
        let mut corpus = Vec::with_capacity(4_000_000);
        for (j, &p) in pick.iter().enumerate() { corpus.extend_from_slice(&lines[p as usize][..10]); if j % 3 == 0 { corpus.push(30_000 + (j % 5000) as u32) } }
        let t = Instant::now();
        for ch in corpus.chunks(8192) { idx.ingest(ch) }
        let ing = t.elapsed().as_secs_f64();
        let tails: Vec<&[u32]> = (0..20_000).map(|i| { let s = (i * 977) % (corpus.len() - 16); &corpus[s..s + 8] }).collect();
        for (ml, br) in [(8, 1), (8, 8), (16, 8)] {
            let mut out = Drafts::default();
            let mut us: Vec<f64> = Vec::with_capacity(tails.len());
            let mut toks = 0usize;
            for tl in &tails { let t = Instant::now(); idx.query_into(tl, ml, br, &mut out); us.push(t.elapsed().as_secs_f64() * 1e6); toks += out.rows.iter().map(|r| r.tokens.len()).sum::<usize>() }
            us.sort_by(|a, b| a.total_cmp(b));
            let (p50, p99, max) = (us[us.len() / 2], us[us.len() * 99 / 100], us[us.len() - 1]);
            println!("query max_len {} branches {}: p50 {:.1} us, p99 {:.1} us, max {:.1} us, avg tokens/query {:.1}", ml, br, p50, p99, max, toks as f64 / tails.len() as f64);
            assert!(p50 < 200.0);
        }
        let s = idx.stats();
        println!("ingest {} tokens in {:.2} s ({:.2} M tok/s); table {:.1} MB, used {}/{} slots, evictions {}", s.tokens, ing, s.tokens as f64 / ing / 1e6,
            s.table_bytes as f64 / 1048576.0, s.used_slots, s.slots, s.evictions);
        let _ = fs::remove_dir_all(&d);
    }
}
