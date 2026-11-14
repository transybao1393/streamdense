//! Heat-on-disk store: the watcher's and the heat maps' ONLY store is disk, full
//! f32, ONE fixed-size file (<= `CAP_BYTES`, pre-allocated at creation, never grows), updated in place; host RAM for
//! the numbers = 0 (fixed stack arrays: a `CH` f32 chunk + one record). The GPU recorder region (model arena) is the
//! only live accumulator: each job is folded from it (in place, no copy) into its label's slot and the pooled now /
//! session regions, then the recorder is reset (zeroed).
//!
//! File: <watcher dir>/store/<Dims tag>_V<vocab>/store.bin, regions in order:
//! - slot table (`HDR` = 4096 B: two copies A / B at 0 / 2048, each [magic 8 | n | n_cent | nslots | seq | total bytes
//!   | slot -> label map (NP x u64, FREE = u64::MAX) | held owner | move target slot | move pre-header (6 x u64) |
//!   check u64]; the newer valid copy wins);
//! - `nslots` label slots of `lsz()` bytes (all-time sums, no decay; nslots = min(NP, (cap - fixed) / lsz));
//! - the held slot (`HELD`, `lsz()` bytes): the jobs of ONE label held out of its slot by the watcher's similarity gate
//!   (owner in the table). On confirm it is added into the label's slot (`move_held`), else dropped (`drop_held`).
//!   Mixed is never a label any more, so 6 labels + held fit where 7 label slots were (same file size);
//! - now / session (`nsz()` each; all labels pooled, decayed: tau 5 min / 4 h, lazily per chunk);
//! - down column norms (`dsz()`: 4096 B header + f32 [layers x inter]; computed once at gateway start);
//! - journal (`jsz()`: [region base | chunk | check] + the pre-image of the label chunk being rewritten).
//! A set region = HDR (4096 B: header copies A / B at 0 / 2048, each [magic 8 | n | n_cent | label | seq | tokens |
//! turns | mtp_rows | out_tok | emb_rows (u64) | last_ts | t0 | rows_d | mtp_d | out_d | emb_d (f64) | check u64]) +
//! chunks of `CH` f32 each followed by a 16 B trailer [t0 f64 | check u64] + (label slots) the centroid f32
//! [layers x blocks]. Unified index layout: [FFN contribution |a|·‖W_down col‖ layers x inter | heads + resid (recorder
//! order: dn heads, attn heads, resid) | MTP layer |a| inter + heads | emitted tokens vocab | embedding rows read vocab].
//! Crash safety: table and set headers are A / B with seq + checksum; chunk records carry their own clock + checksum (a
//! fold cut in the middle leaves every chunk consistent: folded or not; the rest of that job was in the recorder and
//! is lost with the process). Writes are pwrite in the order journal -> chunk; a torn chunk (checksum mismatch) is
//! restored from the journal when it matches, else zeroed and counted (`repaired` / `lost`). Slot replace = (1) table
//! unmaps the old label + journal invalidated, (2) region zeroed, (3) table maps the new label; a crash between the
//! steps leaves a free slot (zeroed again when claimed), never a mixed one.
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::Arc;

use streamdense_model::unload::{head_bytes, q4_row_bytes, MASK_BLOCK, READ_BW};
use streamdense_model::LayerKind;

use crate::watcher::{Dims, NP};

/// f32 per chunk (8 KB): the size of the one stack buffer every streaming path uses
pub const CH: usize = 2048;
const CHB: usize = CH * 4 + 16;
pub const HDR: usize = 4096;
const SLOT: u64 = 2048;
const HB: usize = 144;
const MAGIC: &[u8; 8] = b"SDSTOR02";
const TMAGIC: &[u8; 8] = b"SDSLOT02";
const DMAGIC: &[u8; 8] = b"SDDOWN01";
/// table copy bytes: magic + 5 u64 + map + held owner + move target + move pre-header (6) + check
const TB: usize = 48 + NP * 8 + 64 + 8;
/// free slot in the table map
pub const FREE: u64 = u64::MAX;
const JH: usize = 24;
pub const CAP_BYTES: u64 = 64_000_000;
/// stack-array limits: layers, units (blocks) per layer, heads per layer
pub const MAXL: usize = 256;
pub const MAXU: usize = 1024;
pub const MAXH: usize = 128;
pub const NOW: usize = NP;
pub const SESS: usize = NP + 1;
/// the held slot's set id
pub const HELD: usize = NP + 2;
/// decay time constants of now / session
pub const TAUS_S: [f64; 2] = [300.0, 4.0 * 3600.0];

pub type Rec = [u8; CHB];
pub type Buf = [f32; CH];

#[inline] fn rd(b: &[u8], i: usize) -> f32 { f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]) }
#[inline] fn wr(b: &mut [u8], i: usize, x: f32) { b[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes()); }
/// The f32s as their little-endian bytes, in place (no copy; big-endian hosts swap them first).
fn le_bytes(v: &mut [f32]) -> &[u8] {
    #[cfg(target_endian = "big")]
    for x in v.iter_mut() { *x = f32::from_bits(x.to_bits().swap_bytes()); }
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn u64at(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }
fn f64at(b: &[u8], o: usize) -> f64 { f64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

/// Checksum over 8-byte words (all zeros -> 0, so a never-written, zeroed record is valid).
pub fn check(b: &[u8]) -> u64 {
    let mut h = 0u64;
    for c in b.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    h
}

/// What a job adds (unified layout, see the module doc): the recorder in place (`TapSrc`) or an owned `Act` (tests,
/// migration).
pub trait JobSrc {
    /// (main rows, MTP rows) of the job
    fn rows(&self) -> (u64, u64);
    /// values [off, off + out.len()) of the unified layout
    fn read(&self, off: usize, out: &mut [f32]);
}

/// Live progress (atomics: shared with the GET view, no allocation).
#[derive(Default, Debug)]
pub struct Progress {
    /// index into `PHASES`
    pub phase: AtomicU8,
    pub job: AtomicU64,
    pub chunk: AtomicU64,
    pub chunks: AtomicU64,
    pub folded: AtomicU64,
    pub last_fold_us: AtomicU64,
    pub last_fold_bytes: AtomicU64,
    pub bytes_written: AtomicU64,
    pub repaired: AtomicU64,
    pub lost: AtomicU64,
}
pub const PHASES: [&str; 8] = ["idle", "folding", "summary", "plan", "migrating", "down norms", "replacing slot", "moving held"];

/// One header copy's values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hdr {
    pub seq: u64,
    pub tokens: u64,
    pub turns: u64,
    pub mtp_rows: u64,
    pub out_tok: u64,
    pub emb_rows: u64,
    pub last_ts: f64,
    pub t0: f64,
    pub rows_d: f64,
    pub mtp_d: f64,
    pub out_d: f64,
    pub emb_d: f64,
}

/// Selection read by the streaming paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sel {
    /// one set (decayed sets are decayed to `t` per chunk)
    Set(usize),
    /// every label slot summed (all-time pooled)
    Labels,
}

/// Similarity of a job to a label slot: cosine of the block fingerprints (job 0/1 "hotter than the layer mean" vs the
/// slot centroid, layers x blocks) and cosine of the head masses (job vs the slot's all-time sums). `min()` gates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sim { pub blocks: f32, pub heads: f32 }
impl Sim { pub fn min(&self) -> f32 { self.blocks.min(self.heads) } }

pub struct Store {
    pub dir: PathBuf,
    pub d: Dims,
    pub cap: u64,
    pub ro: bool,
    path: PathBuf,
    file: Option<File>,
    nslots: usize,
    /// slot -> label (FREE)
    map: [u64; NP],
    /// owner label of the held slot (FREE)
    held: u64,
    /// an unfinished `move_held`: (target slot, the target's header before the move)
    mv: Option<(usize, Hdr)>,
    tseq: u64,
    down_ok: bool,
    names: [&'static str; NP],
    pub prog: Arc<Progress>,
    /// clock the decayed sets are read at (Sel::Set(NOW / SESS))
    pub t: f64,
    /// test hook: 1 = stop a replace after the unmap, 2 = in the middle of zeroing the region (simulated crash)
    #[cfg(test)]
    pub crash: u8,
}

/// Per-plan result (fixed arrays: no heap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanOut {
    pub layers: usize,
    pub tokens: u64,
    pub neurons: usize,
    pub cold_neurons: usize,
    pub heads: usize,
    pub cold_heads: usize,
    pub ffn_bytes: u64,
    pub head_bytes: u64,
    pub skip_bytes: u64,
    pub per_layer_cold: [u32; MAXL],
    pub per_layer_cold_heads: [u16; MAXL],
}
impl Default for PlanOut {
    fn default() -> PlanOut {
        PlanOut { layers: 0, tokens: 0, neurons: 0, cold_neurons: 0, heads: 0, cold_heads: 0, ffn_bytes: 0, head_bytes: 0, skip_bytes: 0,
                  per_layer_cold: [0; MAXL], per_layer_cold_heads: [0; MAXL] }
    }
}

/// Radix-select scratch (12 KB: one per streaming caller, on its stack).
pub struct Hist { cnt: [u32; 1024], sum: [f64; 1024] }
impl Default for Hist { fn default() -> Hist { Hist { cnt: [0; 1024], sum: [0.0; 1024] } } }

/// Result of a hot / cold selection: `n` entries; boundary value `t` (exact) and `k` of its ties (lowest indices first).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pick { pub n: usize, pub t: f32, pub k: usize, pub total: f64, pub seen: usize }
impl Pick {
    /// membership pass helper: `hot` = x > t or a tie within the first k (call in index order with a tie counter)
    #[inline] pub fn hot(&self, x: f32, ties: &mut usize) -> bool { if x > self.t { true } else if x == self.t && x > 0.0 && *ties < self.k { *ties += 1; true } else { false } }
    /// cold membership (greedy ascending): x < t (x <= 0 always) or a tie within the first k
    #[inline] pub fn cold(&self, x: f32, ties: &mut usize) -> bool { if x <= 0.0 || x < self.t { true } else if x == self.t && *ties < self.k { *ties += 1; true } else { false } }
}

const LEVELS: [(u32, u32); 4] = [(21, 10), (11, 10), (1, 10), (0, 1)];

/// Exact "hot" selection over a stream (`read(off, buf)` fills buf with entries off..): the fewest largest positive
/// entries holding `keep` of the positive mass (the old `Hist::hot` / sort semantics, ties by index). 5 passes.
pub fn hot(len: usize, keep: f64, read: &mut dyn FnMut(usize, &mut [f32]), buf: &mut Buf, h: &mut Hist) -> Pick {
    let mut p = Pick::default();
    each(len, read, buf, &mut |x| if x > 0.0 && x.is_finite() { p.total += x as f64; p.seen += 1; });
    let target = keep * p.total;
    if !(p.total > 0.0) || len == 0 { p.t = f32::INFINITY; return p; }
    let (mut prefix, mut acc, mut n) = (0u32, 0f64, 0usize);
    for &(s, w) in &LEVELS {
        h.cnt[..1 << w].fill(0);
        h.sum[..1 << w].fill(0.0);
        let (mut a_s, mut a_n) = (0f64, 0usize);
        each(len, read, buf, &mut |x| {
            if !(x > 0.0) || !x.is_finite() { return; }
            let b = x.to_bits();
            let pr = b >> (s + w);
            if pr == prefix { let i = ((b >> s) & ((1 << w) - 1)) as usize; h.cnt[i] += 1; h.sum[i] += x as f64; } else if pr > prefix { a_s += x as f64; a_n += 1; }
        });
        // above the prefix: counted by this level only (earlier levels' "above" are exactly these)
        acc = a_s;
        n = a_n;
        let mut chosen = None;
        for i in (0..1usize << w).rev() {
            if h.cnt[i] == 0 { continue; }
            if acc + h.sum[i] >= target { chosen = Some(i); break; }
            acc += h.sum[i];
            n += h.cnt[i] as usize;
        }
        match chosen { Some(i) => prefix = (prefix << w) | i as u32, None => { p.n = n; p.t = 0.0; return p; } }
    }
    let t = f32::from_bits(prefix);
    let cnt = h.cnt[(prefix & 1) as usize] as usize;
    let k = (((target - acc) / t as f64 - 1e-9).ceil().max(1.0) as usize).min(cnt);
    p.n = n + k;
    p.t = t;
    p.k = k;
    p
}

/// Exact greedy-ascending "cold" selection (`unload::cold_mask` with g = 1): smallest entries first while the removed
/// mass <= (1 - keep) x total (x (1 + 1e-9)); entries <= 0 are always cold. 5 passes.
pub fn cold(len: usize, keep: f64, read: &mut dyn FnMut(usize, &mut [f32]), buf: &mut Buf, h: &mut Hist) -> Pick {
    let mut p = Pick::default();
    let mut nonpos = 0usize;
    each(len, read, buf, &mut |x| if x > 0.0 { p.total += x as f64; p.seen += 1; } else { nonpos += 1; });
    if keep >= 1.0 || len == 0 || !(p.total > 0.0) { p.t = 0.0; p.n = 0; p.k = 0; if !(p.total > 0.0) || keep >= 1.0 { p.t = f32::MIN_POSITIVE; p.n = if keep >= 1.0 { 0 } else { len }; } return p; }
    let budget = (1.0 - keep.max(0.0)) * p.total * (1.0 + 1e-9);
    let (mut prefix, mut acc, mut n) = (0u32, 0f64, nonpos);
    for &(s, w) in &LEVELS {
        h.cnt[..1 << w].fill(0);
        h.sum[..1 << w].fill(0.0);
        let (mut b_s, mut b_n) = (0f64, 0usize);
        each(len, read, buf, &mut |x| {
            if !(x > 0.0) { return; }
            let b = x.to_bits();
            let pr = b >> (s + w);
            if pr == prefix { let i = ((b >> s) & ((1 << w) - 1)) as usize; h.cnt[i] += 1; h.sum[i] += x as f64; } else if pr < prefix { b_s += x as f64; b_n += 1; }
        });
        acc = b_s;
        n = nonpos + b_n;
        let mut chosen = None;
        for i in 0..1usize << w {
            if h.cnt[i] == 0 { continue; }
            if acc + h.sum[i] > budget { chosen = Some(i); break; }
            acc += h.sum[i];
            n += h.cnt[i] as usize;
        }
        match chosen { Some(i) => prefix = (prefix << w) | i as u32, None => { p.n = n; p.t = f32::INFINITY; return p; } }
    }
    let t = f32::from_bits(prefix);
    let cnt = h.cnt[(prefix & 1) as usize] as usize;
    let k = (((budget - acc) / t as f64).floor().max(0.0) as usize).min(cnt);
    p.n = n + k;
    p.t = t;
    p.k = k;
    p
}

/// Stream `len` entries through `buf`, calling `f` on each.
pub fn each(len: usize, read: &mut dyn FnMut(usize, &mut [f32]), buf: &mut Buf, f: &mut dyn FnMut(f32)) {
    let mut o = 0;
    while o < len {
        let l = CH.min(len - o);
        read(o, &mut buf[..l]);
        for &x in &buf[..l] { f(x); }
        o += l;
    }
}

/// Greedy cold mask over at most MAXU unit masses (`unload::cold_mask` on units): coldest first (ties by index)
/// while the removed mass <= (1 - keep) x total; `cold[u]` set. Stack only.
pub fn cold_units(mass: &[f64], keep: f64, cold: &mut [bool]) -> usize {
    let n = mass.len().min(MAXU);
    cold[..n].fill(false);
    let total: f64 = mass[..n].iter().sum();
    if keep >= 1.0 || n == 0 || !(total > 0.0) { return 0; }
    let budget = (1.0 - keep.max(0.0)) * total * (1.0 + 1e-9);
    let mut idx = [0u16; MAXU];
    for (i, x) in idx[..n].iter_mut().enumerate() { *x = i as u16; }
    idx[..n].sort_unstable_by(|&a, &b| mass[a as usize].total_cmp(&mass[b as usize]).then(a.cmp(&b)));
    let (mut c, mut k) = (0f64, 0);
    for &u in &idx[..n] {
        let m = mass[u as usize];
        if c + m > budget { break; }
        c += m;
        cold[u as usize] = true;
        k += 1;
    }
    k
}

fn oe(m: &str) -> std::io::Error { std::io::Error::other(m.to_string()) }

impl Store {
    pub fn n_ffn(&self) -> usize { self.d.layers * self.d.inter }
    pub fn n_hr(&self) -> usize { let t = &self.d.tmpl; t.n_deltanet() * t.dn_vh + t.n_attention() * t.heads + t.layers }
    pub fn n_mtp(&self) -> usize { self.d.inter + self.d.tmpl.heads }
    pub fn o_mtp(&self) -> usize { self.n_ffn() + self.n_hr() }
    pub fn o_out(&self) -> usize { self.o_mtp() + self.n_mtp() }
    pub fn o_emb(&self) -> usize { self.o_out() + self.d.vocab }
    pub fn n(&self) -> usize { self.o_emb() + self.d.vocab }
    pub fn n_cent(&self) -> usize { self.d.layers * self.d.blocks }
    pub fn nch(&self) -> usize { self.n().div_ceil(CH) }
    /// head range of layer l in the unified layout
    pub fn heads_of(&self, l: usize) -> std::ops::Range<usize> {
        let t = &self.d.tmpl;
        let i = t.kind_index(l);
        let base = self.n_ffn();
        match t.kinds[l] {
            LayerKind::DeltaNet => base + i * t.dn_vh..base + (i + 1) * t.dn_vh,
            LayerKind::Attention => { let b = base + t.n_deltanet() * t.dn_vh; b + i * t.heads..b + (i + 1) * t.heads }
        }
    }

    // ---- layout (bytes) ----
    /// one label slot: header + chunks + centroid
    pub fn lsz(&self) -> u64 { (HDR + self.nch() * CHB + self.n_cent() * 4) as u64 }
    /// now / session region
    pub fn nsz(&self) -> u64 { (HDR + self.nch() * CHB) as u64 }
    /// down norms region
    pub fn dsz(&self) -> u64 { (HDR + self.n_ffn() * 4) as u64 }
    /// journal region
    pub fn jsz(&self) -> u64 { (JH + CHB) as u64 }
    /// everything but the label slots (table, held slot, now, session, down norms, journal)
    pub fn fixed_bytes(&self) -> u64 { HDR as u64 + self.lsz() + 2 * self.nsz() + self.dsz() + self.jsz() }
    /// label slots that fit `cap` (at most NP)
    pub fn slots_for(&self, cap: u64) -> usize { (cap.saturating_sub(self.fixed_bytes()) / self.lsz()).min(NP as u64) as usize }
    pub fn nslots(&self) -> usize { self.nslots }
    /// the file's fixed size
    pub fn total_bytes(&self) -> u64 { self.fixed_bytes() + self.nslots as u64 * self.lsz() }
    pub fn slot_base(&self, i: usize) -> u64 { HDR as u64 + i as u64 * self.lsz() }
    fn held_base(&self) -> u64 { self.slot_base(self.nslots) }
    fn now_base(&self) -> u64 { self.held_base() + self.lsz() }
    pub fn down_base(&self) -> u64 { self.now_base() + 2 * self.nsz() }
    fn jnl_base(&self) -> u64 { self.down_base() + self.dsz() }
    /// the slot label `s` lives in
    pub fn slot_of(&self, s: usize) -> Option<usize> { (0..self.nslots).find(|&i| self.map[i] == s as u64) }
    /// label in slot i (None = free)
    pub fn slot_label(&self, i: usize) -> Option<usize> { (i < self.nslots && self.map[i] != FREE).then_some(self.map[i] as usize) }
    pub fn free_slot(&self) -> Option<usize> { (0..self.nslots).find(|&i| self.map[i] == FREE) }
    /// byte offset of set s's region (None: a label without a slot)
    fn base(&self, s: usize) -> Option<u64> {
        if s < NP { self.slot_of(s).map(|i| self.slot_base(i)) } else if s == NOW { Some(self.now_base()) } else if s == SESS { Some(self.now_base() + self.nsz()) }
        else if s == HELD { (self.held != FREE).then(|| self.held_base()) } else { None }
    }
    pub fn name(&self, s: usize) -> &'static str { if s < NP { self.names[s] } else if s == NOW { "now" } else if s == SESS { "session" } else { "held" } }
    /// label whose held jobs fill the held slot
    pub fn held_owner(&self) -> Option<usize> { (self.held != FREE).then_some(self.held as usize) }
    /// set s has a region (a label: holds a slot; now / session: the file exists)
    pub fn has(&self, s: usize) -> bool { self.file.is_some() && self.base(s).is_some() }
    pub fn dir_of(watcher_dir: &Path, d: &Dims) -> PathBuf { watcher_dir.join("store").join(format!("{}_V{}", d.tag(), d.vocab)) }
    pub fn path(&self) -> &Path { &self.path }

    /// Open (create unless `ro`) the store of these dims at the default cap. `names` = label names by index.
    pub fn open(watcher_dir: &Path, d: Dims, names: [&'static str; NP], mixed: usize, ro: bool) -> std::io::Result<Store> {
        Store::open_cap(watcher_dir, d, names, mixed, ro, CAP_BYTES)
    }
    /// Same with an explicit cap (the file size is fixed from it at creation; an existing file of the same dims keeps
    /// its slot count). Creation writes the whole file once (pre-allocated, one-time setup).
    pub fn open_cap(watcher_dir: &Path, d: Dims, names: [&'static str; NP], _mixed: usize, ro: bool, cap: u64) -> std::io::Result<Store> {
        if d.layers > MAXL || d.blocks > MAXU || d.hmax > MAXH { return Err(oe(&format!("store: dims over the stack limits ({} layers, {} blocks, {} heads)", d.layers, d.blocks, d.hmax))); }
        let dir = Store::dir_of(watcher_dir, &d);
        let path = dir.join("store.bin");
        let mut s = Store { dir, d, cap, ro, path, file: None, nslots: 0, map: [FREE; NP], held: FREE, mv: None, tseq: 0, down_ok: false, names, prog: Arc::new(Progress::default()), t: 0.0,
                            #[cfg(test)] crash: 0 };
        s.nslots = s.slots_for(cap);
        if s.nslots == 0 { return Err(oe(&format!("store: cap {} B holds no label slot (fixed {} B + slot {} B)", cap, s.fixed_bytes(), s.lsz()))); }
        if ro { s.refresh(); return Ok(s); }
        std::fs::create_dir_all(&s.dir)?;
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&s.path)?;
        s.file = Some(f);
        if !s.read_table() || s.file.as_ref().unwrap().metadata()?.len() != s.total_bytes() {
            s.nslots = s.slots_for(cap);
            s.create()?;
        }
        s.check_down();
        if s.mv.is_some() { eprintln!("store: resuming an interrupted held -> slot move"); s.resume_move()?; }
        Ok(s)
    }
    /// Write the whole file once (zeros: every record / header valid-empty) and a fresh table.
    fn create(&mut self) -> std::io::Result<()> {
        let f = self.file.as_ref().ok_or_else(|| oe("store: no file"))?;
        f.set_len(0)?;
        let total = self.total_bytes();
        let z = [0u8; CHB];
        let mut o = 0u64;
        while o < total { let l = (total - o).min(CHB as u64) as usize; f.write_all_at(&z[..l], o)?; o += l as u64; }
        f.sync_all()?;
        self.map = [FREE; NP];
        (self.held, self.mv, self.tseq) = (FREE, None, 0);
        self.write_table()?;
        eprintln!("store: created {} ({} B: {} label slots + 1 held slot x {} B, now / session {} B each, down {} B, journal {} B)",
                  self.path.display(), total, self.nslots, self.lsz(), self.nsz(), self.dsz(), self.jsz());
        Ok(())
    }
    /// (Re)read the table (read-only stores call this before each summary; opens the file when it appeared).
    pub fn refresh(&mut self) {
        if self.file.is_none() && self.ro { self.file = File::open(&self.path).ok(); }
        if self.file.is_some() && !self.read_table() { (self.map, self.held, self.mv) = ([FREE; NP], FREE, None); if self.ro { self.file = None; } }
        self.check_down();
    }
    fn check_down(&mut self) {
        self.down_ok = self.file.as_ref().is_some_and(|f| {
            let mut h = [0u8; 32];
            f.read_exact_at(&mut h, self.down_base()).is_ok() && &h[..8] == DMAGIC && u64at(&h, 8) == self.n_ffn() as u64 && u64at(&h, 16) == 1
        });
    }
    /// Bytes of the store file (fixed).
    pub fn disk_bytes(&self) -> u64 { self.file.as_ref().and_then(|f| f.metadata().ok()).map_or(0, |m| m.len()) }

    // ---- slot table ----
    fn read_table(&mut self) -> bool {
        let Some(f) = &self.file else { return false };
        let mut b = [0u8; TB];
        let mut best: Option<(u64, usize, [u64; NP], u64, Option<(usize, Hdr)>)> = None;
        for c in 0..2u64 {
            if f.read_exact_at(&mut b, c * SLOT).is_err() || &b[..8] != TMAGIC || check(&b[..TB - 8]) != u64at(&b, TB - 8) { continue; }
            if u64at(&b, 8) != self.n() as u64 || u64at(&b, 16) != self.n_cent() as u64 { continue; }
            let (ns, seq) = (u64at(&b, 24) as usize, u64at(&b, 32));
            if ns == 0 || ns > NP { continue; }
            let mut m = [FREE; NP];
            for (i, x) in m.iter_mut().enumerate() { *x = u64at(&b, 48 + i * 8); }
            let o = 48 + NP * 8;
            let mv = (u64at(&b, o + 8) != FREE).then(|| (u64at(&b, o + 8) as usize, Hdr { tokens: u64at(&b, o + 16), turns: u64at(&b, o + 24), mtp_rows: u64at(&b, o + 32),
                out_tok: u64at(&b, o + 40), emb_rows: u64at(&b, o + 48), last_ts: f64at(&b, o + 56), ..Default::default() }));
            if best.is_none_or(|x| seq >= x.0) { best = Some((seq, ns, m, u64at(&b, o), mv)); }
        }
        let Some((seq, ns, m, held, mv)) = best else { return false };
        self.tseq = seq;
        self.nslots = ns;
        self.map = m;
        (self.held, self.mv) = (held, mv);
        true
    }
    fn write_table(&mut self) -> std::io::Result<()> {
        let f = self.file.as_ref().ok_or_else(|| oe("store: no file"))?;
        self.tseq += 1;
        let mut b = [0u8; TB];
        b[..8].copy_from_slice(TMAGIC);
        let total = self.total_bytes();
        for (i, x) in [self.n() as u64, self.n_cent() as u64, self.nslots as u64, self.tseq, total].iter().enumerate() { b[8 + i * 8..16 + i * 8].copy_from_slice(&x.to_le_bytes()); }
        for (i, x) in self.map.iter().enumerate() { b[48 + i * 8..56 + i * 8].copy_from_slice(&x.to_le_bytes()); }
        let o = 48 + NP * 8;
        let (t, h) = self.mv.map_or((FREE, Hdr::default()), |(i, h)| (i as u64, h));
        for (i, x) in [self.held, t, h.tokens, h.turns, h.mtp_rows, h.out_tok, h.emb_rows, h.last_ts.to_bits()].iter().enumerate() { b[o + i * 8..o + 8 + i * 8].copy_from_slice(&x.to_le_bytes()); }
        let c = check(&b[..TB - 8]);
        b[TB - 8..].copy_from_slice(&c.to_le_bytes());
        f.write_all_at(&b, (self.tseq % 2) * SLOT)?;
        f.sync_data()?;
        self.prog.bytes_written.fetch_add(TB as u64, Relaxed);
        Ok(())
    }
    /// Put label `s` into slot `i`, evicting its current label (if any) first: (1) unmap + journal invalidated, (2)
    /// region zeroed, (3) map `s`. Returns the evicted label.
    pub fn claim(&mut self, i: usize, s: usize) -> std::io::Result<Option<usize>> {
        if self.ro || i >= self.nslots || s >= NP { return Err(oe("store: bad claim")); }
        if self.slot_of(s).is_some() { return Err(oe("store: label already has a slot")); }
        let old = self.slot_label(i);
        let ph = self.prog.phase.swap(6, Relaxed);
        self.map[i] = FREE;
        self.write_table()?;
        self.invalidate_journal()?;
        #[cfg(test)]
        if self.crash == 1 { return Err(oe("simulated crash after unmap")); }
        self.zero_region(self.slot_base(i))?;
        self.map[i] = s as u64;
        self.write_table()?;
        self.prog.phase.store(ph, Relaxed);
        Ok(old)
    }
    /// Zero one label-sized region (a slot or the held slot).
    fn zero_region(&self, base: u64) -> std::io::Result<()> {
        let len = self.lsz();
        let f = self.file.as_ref().ok_or_else(|| oe("store: no file"))?;
        let z = [0u8; CHB];
        let mut o = 0u64;
        while o < len {
            #[cfg(test)]
            if self.crash == 2 && o >= len / 2 { return Err(oe("simulated crash while zeroing")); }
            let l = (len - o).min(CHB as u64) as usize;
            f.write_all_at(&z[..l], base + o)?;
            o += l as u64;
        }
        f.sync_data()?;
        self.prog.bytes_written.fetch_add(len, Relaxed);
        Ok(())
    }
    /// Give the held slot to label `s` (it must be free): zeroed, then owned.
    pub fn set_held_owner(&mut self, s: usize) -> std::io::Result<()> {
        if self.ro || s >= NP || self.held != FREE { return Err(oe("store: held slot busy")); }
        self.zero_region(self.held_base())?;
        self.held = s as u64;
        self.write_table()
    }
    /// Drop the held jobs (they stay in now / session): unowned first, then zeroed.
    pub fn drop_held(&mut self) -> std::io::Result<()> {
        if self.held == FREE { return Ok(()); }
        self.invalidate_journal()?;
        self.held = FREE;
        self.write_table()?;
        self.zero_region(self.held_base())
    }
    /// Add the held jobs into label `to`'s slot (it must hold one), then free the held slot. Crash-safe and
    /// idempotent: the target's pre-move header is in the table; each held chunk is zeroed right after it is added
    /// (journaled: an add whose held chunk was not zeroed yet is undone from the journal and redone).
    pub fn move_held(&mut self, to: usize) -> std::io::Result<()> {
        let i = self.slot_of(to).ok_or_else(|| oe("store: move_held: no target slot"))?;
        if self.held == FREE { return Err(oe("store: move_held: nothing held")); }
        let (hs, hh) = (self.hdr(to), self.hdr(HELD));
        // merged centroid (weighted by turns) into the held centroid first (copied over at the end)
        let (a, w) = (hs.turns as f32, hh.turns as f32);
        if a + w > 0.0 {
            let nb = self.d.blocks;
            let (mut ct, mut ch) = ([0f32; MAXU], [0f32; MAXU]);
            for l in 0..self.d.layers {
                if !self.read_cent(to, l, &mut ct[..nb]) { ct[..nb].fill(0.0); }
                if !self.read_cent(HELD, l, &mut ch[..nb]) { ch[..nb].fill(0.0); }
                for j in 0..nb { ch[j] = (ct[j] * a + ch[j] * w) / (a + w); }
                self.write_cent(HELD, l, &ch[..nb])?;
            }
        }
        self.invalidate_journal()?;
        self.mv = Some((i, hs));
        self.write_table()?;
        self.resume_move()
    }
    fn resume_move(&mut self) -> std::io::Result<()> {
        let Some((i, pre)) = self.mv else { return Ok(()) };
        let Some(to) = self.slot_label(i) else { self.mv = None; return self.write_table() };
        let ph = self.prog.phase.swap(7, Relaxed);
        let (sb, jb) = (self.slot_base(i), self.jnl_base());
        let mut hrec = [0u8; CHB];
        let mut srec = [0u8; CHB];
        {
            let f = self.file.as_ref().ok_or_else(|| oe("store: no file"))?;
            // an add interrupted before its held chunk was zeroed: restore the slot chunk's pre-image
            let mut jh = [0u8; JH];
            if f.read_exact_at(&mut jh, jb).is_ok() && u64at(&jh, 0) == sb {
                let k = u64at(&jh, 8) as usize;
                if k < self.nch() && self.read_rec(HELD, k, &mut hrec).is_ok() && hrec[..CH * 4].iter().any(|x| *x != 0) {
                    f.read_exact_at(&mut srec, jb + JH as u64)?;
                    if check(&srec[..CHB - 8]) == u64at(&srec, CHB - 8) && check(&srec[..]) == u64at(&jh, 16) { f.write_all_at(&srec, Store::rec_off(sb, k))?; }
                }
            }
        }
        for k in 0..self.nch() {
            self.prog.chunk.store(k as u64, Relaxed);
            self.read_rec(HELD, k, &mut hrec)?;
            if hrec[..CH * 4].iter().all(|x| *x == 0) { continue; }
            self.read_rec(to, k, &mut srec)?;
            self.journal(to, k, &srec)?;
            for j in 0..CH { let y = rd(&srec, j) + rd(&hrec, j); wr(&mut srec, j, y); }
            self.write_rec(to, k, &mut srec)?;
            #[cfg(test)]
            if self.crash == 4 { return Err(oe("simulated crash between an add and its held chunk's zeroing")); }
            hrec.fill(0);
            self.write_rec(HELD, k, &mut hrec)?;
            #[cfg(test)]
            if self.crash == 3 { return Err(oe("simulated crash while moving")); }
        }
        let hh = self.hdr(HELD);
        let h = Hdr { tokens: pre.tokens + hh.tokens, turns: pre.turns + hh.turns, mtp_rows: pre.mtp_rows + hh.mtp_rows, out_tok: pre.out_tok + hh.out_tok,
                      emb_rows: pre.emb_rows + hh.emb_rows, last_ts: pre.last_ts.max(hh.last_ts), ..Default::default() };
        self.write_hdr(to, &h)?;
        let nb = self.d.blocks;
        let mut c = [0f32; MAXU];
        for l in 0..self.d.layers { if self.read_cent(HELD, l, &mut c[..nb]) { self.write_cent(to, l, &c[..nb])?; } }
        self.file.as_ref().unwrap().sync_data()?;
        let base = self.held_base();
        (self.held, self.mv) = (FREE, None);
        self.write_table()?;
        self.zero_region(base)?;
        self.prog.phase.store(ph, Relaxed);
        Ok(())
    }
    fn invalidate_journal(&self) -> std::io::Result<()> {
        let Some(f) = &self.file else { return Ok(()) };
        f.write_all_at(&[0xffu8; JH], self.jnl_base())?;
        f.sync_data()
    }

    // ---- set headers ----
    pub fn hdr(&self, s: usize) -> Hdr {
        let (Some(f), Some(base)) = (&self.file, self.base(s)) else { return Hdr::default() };
        let mut best = Hdr::default();
        let mut b = [0u8; HB];
        for c in 0..2u64 {
            if f.read_exact_at(&mut b, base + c * SLOT).is_err() || &b[..8] != MAGIC || check(&b[..HB - 8]) != u64at(&b, HB - 8) || u64at(&b, 24) != s as u64 { continue; }
            let h = Hdr { seq: u64at(&b, 32), tokens: u64at(&b, 40), turns: u64at(&b, 48), mtp_rows: u64at(&b, 56), out_tok: u64at(&b, 64),
                emb_rows: u64at(&b, 72), last_ts: f64at(&b, 80), t0: f64at(&b, 88), rows_d: f64at(&b, 96), mtp_d: f64at(&b, 104),
                out_d: f64at(&b, 112), emb_d: f64at(&b, 120) };
            if h.seq >= best.seq { best = h; }
        }
        best
    }
    fn write_hdr(&self, s: usize, h: &Hdr) -> std::io::Result<()> {
        let (Some(f), Some(base)) = (&self.file, self.base(s)) else { return Err(std::io::ErrorKind::NotFound.into()) };
        let seq = self.hdr(s).seq + 1;
        let mut b = [0u8; HB];
        b[..8].copy_from_slice(MAGIC);
        let u = [self.n() as u64, self.n_cent() as u64, s as u64, seq, h.tokens, h.turns, h.mtp_rows, h.out_tok, h.emb_rows];
        for (i, x) in u.iter().enumerate() { b[8 + i * 8..16 + i * 8].copy_from_slice(&x.to_le_bytes()); }
        for (i, x) in [h.last_ts, h.t0, h.rows_d, h.mtp_d, h.out_d, h.emb_d].iter().enumerate() { b[80 + i * 8..88 + i * 8].copy_from_slice(&x.to_le_bytes()); }
        let c = check(&b[..HB - 8]);
        b[HB - 8..].copy_from_slice(&c.to_le_bytes());
        f.write_all_at(&b, base + (seq % 2) * SLOT)?;
        self.prog.bytes_written.fetch_add(HB as u64, Relaxed);
        Ok(())
    }

    // ---- chunk records ----
    fn rec_off(base: u64, k: usize) -> u64 { base + (HDR + k * CHB) as u64 }
    /// Read chunk k of set s into `b`, verified. A torn record is restored from the journal (writable store) or retried
    /// / reported as zero (read-only store). A label without a slot = zeros.
    pub fn read_rec(&self, s: usize, k: usize, b: &mut Rec) -> std::io::Result<()> {
        let (Some(f), Some(base)) = (&self.file, self.base(s)) else { b.fill(0); return Ok(()) };
        let off = Store::rec_off(base, k);
        for attempt in 0..20 {
            f.read_exact_at(b, off)?;
            if check(&b[..CHB - 8]) == u64at(b, CHB - 8) { return Ok(()); }
            if self.ro { if attempt < 19 { std::thread::sleep(std::time::Duration::from_millis(2)); continue; } b.fill(0); return Ok(()); }
            // torn: the journal's pre-image if it is this region's chunk, else zero (counted)
            let jb = self.jnl_base();
            let mut jh = [0u8; JH];
            if f.read_exact_at(&mut jh, jb).is_ok() && u64at(&jh, 0) == base && u64at(&jh, 8) == k as u64 {
                f.read_exact_at(b, jb + JH as u64)?;
                if check(&b[..CHB - 8]) == u64at(b, CHB - 8) && check(&b[..]) == u64at(&jh, 16) {
                    f.write_all_at(b, off)?;
                    self.prog.repaired.fetch_add(1, Relaxed);
                    return Ok(());
                }
            }
            b.fill(0);
            f.write_all_at(b, off)?;
            self.prog.lost.fetch_add(1, Relaxed);
            return Ok(());
        }
        Ok(())
    }
    fn journal(&self, s: usize, k: usize, b: &Rec) -> std::io::Result<()> {
        let (Some(f), Some(base)) = (&self.file, self.base(s)) else { return Ok(()) };
        let jb = self.jnl_base();
        let mut jh = [0xffu8; JH]; // invalid while the record is written
        f.write_all_at(&jh, jb)?;
        f.write_all_at(b, jb + JH as u64)?;
        jh[..8].copy_from_slice(&base.to_le_bytes());
        jh[8..16].copy_from_slice(&(k as u64).to_le_bytes());
        jh[16..24].copy_from_slice(&check(&b[..]).to_le_bytes());
        f.write_all_at(&jh, jb)?;
        self.prog.bytes_written.fetch_add((CHB + 2 * JH) as u64, Relaxed);
        Ok(())
    }
    fn write_rec(&self, s: usize, k: usize, b: &mut Rec) -> std::io::Result<()> {
        let (Some(f), Some(base)) = (&self.file, self.base(s)) else { return Err(std::io::ErrorKind::NotFound.into()) };
        let c = check(&b[..CHB - 8]);
        b[CHB - 8..].copy_from_slice(&c.to_le_bytes());
        f.write_all_at(b, Store::rec_off(base, k))?;
        self.prog.bytes_written.fetch_add(CHB as u64, Relaxed);
        Ok(())
    }
    fn tau(s: usize) -> Option<f64> { match s { NOW => Some(TAUS_S[0]), SESS => Some(TAUS_S[1]), _ => None } }
    fn decay(s: usize, t0: f64, t: f64) -> f64 { Store::tau(s).map_or(1.0, |tau| (-(t - t0).max(0.0) / tau).exp()) }

    /// Values [off, off + out.len()) of a selection (decayed sets at `self.t`), through record `b`.
    pub fn read_sel(&self, sel: Sel, off: usize, out: &mut [f32], b: &mut Rec) {
        out.fill(0.0);
        let n = self.n();
        let mut i = 0;
        while i < out.len() && off + i < n {
            let g = off + i;
            let (k, c0) = (g / CH, g % CH);
            let l = (CH - c0).min(out.len() - i);
            let mut add = |s: usize, b: &mut Rec| {
                if !self.has(s) || self.read_rec(s, k, b).is_err() { return; }
                let f = Store::decay(s, f64at(b, CH * 4), self.t) as f32;
                for j in 0..l { out[i + j] += rd(b, c0 + j) * f; }
            };
            match sel { Sel::Set(s) => add(s, b), Sel::Labels => for s in (0..NP).chain([HELD]) { add(s, b) } }
            i += l;
        }
    }
    /// Centroid values of layer l of label s into `out` (len blocks).
    pub fn read_cent(&self, s: usize, l: usize, out: &mut [f32]) -> bool {
        let (Some(f), Some(base)) = (&self.file, if s < NP || s == HELD { self.base(s) } else { None }) else { return false };
        let mut b = [0u8; 4 * MAXU];
        let nb = self.d.blocks;
        if f.read_exact_at(&mut b[..nb * 4], base + (HDR + self.nch() * CHB + l * nb * 4) as u64).is_err() { return false; }
        for j in 0..nb { out[j] = rd(&b, j); }
        true
    }
    fn write_cent(&self, s: usize, l: usize, v: &[f32]) -> std::io::Result<()> {
        let (Some(f), Some(base)) = (&self.file, self.base(s)) else { return Err(std::io::ErrorKind::NotFound.into()) };
        let mut b = [0u8; 4 * MAXU];
        for (j, x) in v.iter().enumerate() { wr(&mut b, j, *x); }
        f.write_all_at(&b[..v.len() * 4], base + (HDR + self.nch() * CHB + l * v.len() * 4) as u64)?;
        self.prog.bytes_written.fetch_add(v.len() as u64 * 4, Relaxed);
        Ok(())
    }

    /// Per-block FFN contribution of layer l from a source (`blk` len blocks).
    pub fn layer_blocks(&self, l: usize, read: &mut dyn FnMut(usize, &mut [f32]), buf: &mut Buf, blk: &mut [f64]) {
        let (inter, g) = (self.d.inter, self.d.gran);
        blk[..self.d.blocks].fill(0.0);
        let mut o = 0;
        while o < inter {
            let len = CH.min(inter - o);
            read(l * inter + o, &mut buf[..len]);
            for (j, &x) in buf[..len].iter().enumerate() { blk[(o + j) / g] += x as f64; }
            o += len;
        }
    }
    /// 1 bit per block: hotter than the layer's mean (the watcher's activity fingerprint).
    fn fp(blk: &[f64], out: &mut [f32]) {
        let mean = blk.iter().sum::<f64>() / blk.len().max(1) as f64;
        for (o, &x) in out.iter_mut().zip(blk) { *o = if x > mean { 1.0 } else { 0.0 }; }
    }

    /// Cosine of the job's fingerprint with the centroid of each label in `use_label` (others 0). Stack only.
    pub fn score(&self, src: &dyn JobSrc, use_label: &[bool; NP]) -> [f32; NP] {
        let mut out = [0f32; NP];
        let (mut ab, mut bb, mut aa) = ([0f64; NP], [0f64; NP], 0f64);
        let mut buf = [0f32; CH];
        let mut blk = [0f64; MAXU];
        let (mut fp, mut c) = ([0f32; MAXU], [0f32; MAXU]);
        let nb = self.d.blocks;
        for l in 0..self.d.layers {
            self.layer_blocks(l, &mut |o, b| src.read(o, b), &mut buf, &mut blk);
            Store::fp(&blk[..nb], &mut fp[..nb]);
            aa += fp[..nb].iter().map(|x| (*x * *x) as f64).sum::<f64>();
            for s in 0..NP {
                if !use_label[s] || !self.read_cent(s, l, &mut c[..nb]) { continue; }
                for j in 0..nb { ab[s] += (fp[j] * c[j]) as f64; bb[s] += (c[j] * c[j]) as f64; }
            }
        }
        for s in 0..NP { if use_label[s] && aa > 0.0 && bb[s] > 0.0 { out[s] = (ab[s] / (aa.sqrt() * bb[s].sqrt())) as f32; } }
        out
    }

    /// Head-mass range of the unified layout (dn + attention heads; the residual entries after them excluded).
    fn heads_range(&self) -> std::ops::Range<usize> { self.n_ffn()..self.o_mtp() - self.d.layers }

    /// Similarity of a job to label s's slot (None: no slot or nothing learned yet). Stack only.
    pub fn sim(&self, s: usize, src: &dyn JobSrc) -> Option<Sim> {
        if s >= NP || !self.has(s) || self.hdr(s).turns == 0 { return None; }
        let mut use_l = [false; NP];
        use_l[s] = true;
        let blocks = self.score(src, &use_l)[s];
        let r = self.heads_range();
        let (mut a, mut b) = ([0f32; CH], [0f32; CH]);
        let mut rec = [0u8; CHB];
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        let mut o = r.start;
        while o < r.end {
            let l = CH.min(r.end - o);
            src.read(o, &mut a[..l]);
            self.read_sel(Sel::Set(s), o, &mut b[..l], &mut rec);
            for j in 0..l { let (x, y) = (a[j] as f64, b[j] as f64); ab += x * y; aa += x * x; bb += y * y; }
            o += l;
        }
        let heads = if aa > 0.0 && bb > 0.0 { (ab / (aa.sqrt() * bb.sqrt())) as f32 } else { 1.0 };
        Some(Sim { blocks, heads })
    }
    /// Cosine of the centroids of labels a and b (0 when either has none). Stack only.
    pub fn cent_cos(&self, a: usize, b: usize) -> f32 {
        let nb = self.d.blocks;
        let (mut x, mut y) = ([0f32; MAXU], [0f32; MAXU]);
        let (mut xy, mut xx, mut yy) = (0f64, 0f64, 0f64);
        for l in 0..self.d.layers {
            if !self.read_cent(a, l, &mut x[..nb]) || !self.read_cent(b, l, &mut y[..nb]) { return 0.0; }
            for j in 0..nb { xy += (x[j] * y[j]) as f64; xx += (x[j] * x[j]) as f64; yy += (y[j] * y[j]) as f64; }
        }
        if xx > 0.0 && yy > 0.0 { (xy / (xx.sqrt() * yy.sqrt())) as f32 } else { 0.0 }
    }

    /// Fold one job into label s's slot (all-time, journaled; a label without a slot takes a free one, else error)
    /// and the now / session regions, then its header counters and the centroid (EMA, alpha = max(1 / turns, 0.05)).
    /// The source stays untouched (the caller resets the recorder after). Stack only: one chunk + one record. Returns
    /// bytes written.
    #[cfg(test)]
    pub fn fold_job(&mut self, s: usize, src: &dyn JobSrc, job: u64, t: f64) -> std::io::Result<u64> {
        if s >= NP { return Err(oe("store: not a label")); }
        if self.slot_of(s).is_none() {
            let i = self.free_slot().ok_or_else(|| oe("store: no free slot (the watcher replaces one first)"))?;
            self.claim(i, s)?;
        }
        let w = self.fold_to(Some(s), src, 1, job, t)?;
        self.update_cent(s, src, 0)?;
        Ok(w)
    }
    /// Fold a job into the pooled now / session regions only (a held, pending job: no label slot touched).
    #[cfg(test)]
    pub fn fold_pooled(&mut self, src: &dyn JobSrc, job: u64, t: f64) -> std::io::Result<u64> { self.fold_to(None, src, 1, job, t) }

    /// Fold `turns` jobs summed in `src` (a batch: one label, or the held slot, or pooled only = None) into that set
    /// and now / session, with header counters; no centroid (the caller updates it per job, `update_cent`). The
    /// decayed sets take the whole batch at clock `t`. Stack only. Returns bytes written.
    pub fn fold_to(&mut self, s: Option<usize>, src: &dyn JobSrc, turns: u64, job: u64, t: f64) -> std::io::Result<u64> {
        if let Some(s) = s { if !(s < NP || s == HELD) || !self.has(s) { return Err(oe("store: fold target has no region")); } }
        if self.file.is_none() || self.ro { return Err(oe("store: not writable")); }
        let t_start = std::time::Instant::now();
        let w0 = self.prog.bytes_written.load(Relaxed);
        let (n, nch) = (self.n(), self.nch());
        let (o_out, o_emb) = (self.o_out(), self.o_emb());
        self.prog.phase.store(1, Relaxed);
        self.prog.job.store(job, Relaxed);
        self.prog.chunks.store(nch as u64, Relaxed);
        let mut d = [0f32; CH];
        let mut b = [0u8; CHB];
        let (mut out_tok, mut emb_rows) = (0f64, 0f64);
        let sets = [s.unwrap_or(NOW), NOW, SESS];
        let sets = if s.is_some() { &sets[..] } else { &sets[1..] };
        for k in 0..nch {
            self.prog.chunk.store(k as u64, Relaxed);
            let off = k * CH;
            let len = CH.min(n - off);
            src.read(off, &mut d[..len]);
            for (j, &x) in d[..len].iter().enumerate() {
                let g = off + j;
                if g >= o_emb { emb_rows += x as f64 } else if g >= o_out { out_tok += x as f64 }
            }
            if d[..len].iter().all(|x| *x == 0.0) { continue; }
            for &set in sets {
                self.read_rec(set, k, &mut b)?;
                if set != NOW && set != SESS { self.journal(set, k, &b)?; }
                let f = Store::decay(set, f64at(&b, CH * 4), t) as f32;
                for j in 0..len { let y = rd(&b, j) * f + d[j]; wr(&mut b, j, y); }
                b[CH * 4..CH * 4 + 8].copy_from_slice(&t.to_le_bytes());
                self.write_rec(set, k, &mut b)?;
            }
        }
        let (rows, mtp_rows) = src.rows();
        for set in [NOW, SESS] {
            let mut h = self.hdr(set);
            let f = Store::decay(set, h.t0, t);
            (h.rows_d, h.mtp_d, h.out_d, h.emb_d) = (h.rows_d * f + rows as f64, h.mtp_d * f + mtp_rows as f64, h.out_d * f + out_tok, h.emb_d * f + emb_rows);
            h.t0 = t;
            h.turns += turns;
            h.tokens += rows;
            h.last_ts = t;
            self.write_hdr(set, &h)?;
        }
        if let Some(s) = s {
            let mut h = self.hdr(s);
            h.tokens += rows;
            h.turns += turns;
            h.mtp_rows += mtp_rows;
            h.out_tok += out_tok.round() as u64;
            h.emb_rows += emb_rows.round() as u64;
            h.last_ts = t;
            self.write_hdr(s, &h)?;
        }
        let wb = self.prog.bytes_written.load(Relaxed) - w0;
        self.prog.last_fold_bytes.store(wb, Relaxed);
        self.prog.last_fold_us.store(t_start.elapsed().as_micros() as u64, Relaxed);
        self.prog.folded.fetch_add(1, Relaxed);
        self.prog.phase.store(0, Relaxed);
        Ok(wb)
    }

    /// One job's fingerprint into the centroid of set `s` (a label slot or the held slot): EMA with alpha =
    /// max(1 / turns, 0.05), turns = the header's + `pending` (jobs batched, not folded yet) + this one. Stack only.
    pub fn update_cent(&mut self, s: usize, src: &dyn JobSrc, pending: u64) -> std::io::Result<()> {
        if !self.has(s) { return Ok(()); }
        let turns = self.hdr(s).turns + pending + 1;
        let alpha = (1.0 / turns as f32).max(0.05);
        let nb = self.d.blocks;
        let mut d = [0f32; CH];
        let mut blk = [0f64; MAXU];
        let (mut fp, mut c) = ([0f32; MAXU], [0f32; MAXU]);
        for l in 0..self.d.layers {
            self.layer_blocks(l, &mut |o, x| src.read(o, x), &mut d, &mut blk);
            Store::fp(&blk[..nb], &mut fp[..nb]);
            if !self.read_cent(s, l, &mut c[..nb]) { c[..nb].fill(0.0); }
            for j in 0..nb { c[j] += alpha * (fp[j] - c[j]); }
            self.write_cent(s, l, &c[..nb])?;
        }
        Ok(())
    }

    /// Add a source into label s's slot as all-time sums (migration): header counters + `tokens` / `turns`, centroid
    /// weighted by turns. No decayed sets touched. A label without a slot takes a free one (else error).
    pub fn add_raw(&mut self, s: usize, src: &dyn JobSrc, tokens: u64, turns: u64, cent: Option<&[f32]>) -> std::io::Result<()> {
        if s >= NP { return Err(oe("store: not a label")); }
        if self.slot_of(s).is_none() {
            let i = self.free_slot().ok_or_else(|| oe("store: no free slot"))?;
            self.claim(i, s)?;
        }
        self.prog.phase.store(4, Relaxed);
        let (n, o_out, o_emb) = (self.n(), self.o_out(), self.o_emb());
        let mut d = [0f32; CH];
        let mut b = [0u8; CHB];
        let (mut out_tok, mut emb_rows) = (0f64, 0f64);
        for k in 0..self.nch() {
            let off = k * CH;
            let len = CH.min(n - off);
            src.read(off, &mut d[..len]);
            for (j, &x) in d[..len].iter().enumerate() { let g = off + j; if g >= o_emb { emb_rows += x as f64 } else if g >= o_out { out_tok += x as f64 } }
            if d[..len].iter().all(|x| *x == 0.0) { continue; }
            self.read_rec(s, k, &mut b)?;
            self.journal(s, k, &b)?;
            for j in 0..len { let y = rd(&b, j) + d[j]; wr(&mut b, j, y); }
            self.write_rec(s, k, &mut b)?;
        }
        let mut h = self.hdr(s);
        let (a, w) = (h.turns as f32, turns as f32);
        if let Some(c) = cent.filter(|c| c.len() == self.n_cent() && a + w > 0.0) {
            let nb = self.d.blocks;
            let mut ct = [0f32; MAXU];
            for l in 0..self.d.layers {
                if !self.read_cent(s, l, &mut ct[..nb]) { ct[..nb].fill(0.0); }
                for j in 0..nb { ct[j] = (ct[j] * a + c[l * nb + j] * w) / (a + w); }
                self.write_cent(s, l, &ct[..nb])?;
            }
        }
        h.tokens += tokens; h.turns += turns; h.mtp_rows += src.rows().1; h.out_tok += out_tok.round() as u64; h.emb_rows += emb_rows.round() as u64;
        self.write_hdr(s, &h)?;
        self.prog.phase.store(0, Relaxed);
        Ok(())
    }

    /// Down-column norms region: present + matching (layer 0 recomputed equal) -> kept; else computed with `threads`
    /// threads (one-time setup, progress on stderr).
    pub fn ensure_down(&mut self, tap: &streamdense_model::activity::ActivityTap, threads: usize) -> std::io::Result<bool> {
        let (layers, inter, base) = (self.d.layers, self.d.inter, self.down_base());
        // host RAM: one inter-sized f32 row per thread (70 KB on the 27B), freed on return; the check reads the file in
        // stack chunks (2026-11-01 RAM fix: no byte copies of a row)
        let mut l0 = vec![0f32; inter];
        tap.down_norm_layer(0, &mut l0);
        let f = self.file.as_ref().ok_or_else(|| oe("store: no file"))?;
        if self.down_ok {
            let mut b = [0u8; CH * 4];
            let same = l0.chunks(CH).enumerate().all(|(k, c)| {
                f.read_exact_at(&mut b[..c.len() * 4], base + (HDR + k * CH * 4) as u64).is_ok() && c.iter().enumerate().all(|(j, x)| rd(&b, j) == *x)
            });
            if same { return Ok(false); }
        }
        drop(l0);
        self.prog.phase.store(5, Relaxed);
        f.write_all_at(&[0u8; 32], base)?;
        let done = AtomicU64::new(0);
        let nt = threads.clamp(1, 8);
        std::thread::scope(|sc| {
            for ti in 0..nt {
                let done = &done;
                sc.spawn(move || {
                    let mut o = vec![0f32; inter];
                    for l in (ti..layers).step_by(nt) {
                        tap.down_norm_layer(l, &mut o);
                        let _ = f.write_all_at(le_bytes(&mut o), base + (HDR + l * inter * 4) as u64);
                        let n = done.fetch_add(1, Relaxed) + 1;
                        if n % 16 == 0 || n as usize == layers { eprintln!("store: down norms {}/{} layers", n, layers); }
                    }
                });
            }
        });
        let mut h = [0u8; 32];
        h[..8].copy_from_slice(DMAGIC);
        h[8..16].copy_from_slice(&(self.n_ffn() as u64).to_le_bytes());
        h[16..24].copy_from_slice(&1u64.to_le_bytes());
        f.sync_data()?;
        f.write_all_at(&h, base)?;
        f.sync_data()?;
        self.prog.phase.store(0, Relaxed);
        self.check_down();
        Ok(true)
    }
    /// A second handle on the file for the down norms (for `TapSrc`; None until they are computed).
    pub fn down_file(&self) -> Option<File> { if self.down_ok { self.file.as_ref().and_then(|f| f.try_clone().ok()) } else { None } }
    /// Down norms [off, off + out.len()) into `out` from the region at `base` (1.0 when missing).
    pub fn read_down(f: Option<&File>, base: u64, off: usize, out: &mut [f32]) {
        let Some(f) = f else { out.fill(1.0); return };
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        if f.read_exact_at(bytes, base + (HDR + off * 4) as u64).is_err() { out.fill(1.0); return; }
        #[cfg(target_endian = "big")]
        for x in out.iter_mut() { *x = f32::from_le_bytes(x.to_bits().to_be_bytes()); }
    }

    /// Plan of a selection at `keep` with units of `g` neurons (g >= inter / MAXU: unit sort on the stack; smaller g:
    /// exact radix selection per layer). Stack only.
    pub fn plan(&self, sel: Sel, keep: f64, g: usize) -> PlanOut {
        let t = &self.d.tmpl;
        let (layers, inter) = (self.d.layers, self.d.inter);
        let g = g.max(1);
        let row = q4_row_bytes(t.hidden, t.group, t.bits);
        let per_neuron = 2 * row + (t.hidden * t.bits / 8) as u64;
        let per_group = (t.hidden * 4) as u64;
        let group = t.group.max(1);
        let mut p = PlanOut { layers, neurons: layers * inter, ..Default::default() };
        p.tokens = match sel { Sel::Set(s) => self.hdr(s).tokens, Sel::Labels => (0..NP).chain([HELD]).map(|s| self.hdr(s).tokens).sum() };
        let mut buf = [0f32; CH];
        let mut b = [0u8; CHB];
        let mut h = Hist::default();
        let mut blk = [0f64; MAXU];
        let mut cu = [false; MAXU];
        let units = inter.div_ceil(g);
        for l in 0..layers {
            let mut read = |o: usize, x: &mut [f32]| self.read_sel(sel, l * inter + o, x, &mut b);
            let pick = if units <= MAXU {
                blk[..units].fill(0.0);
                let mut o = 0;
                while o < inter { let len = CH.min(inter - o); read(o, &mut buf[..len]); for j in 0..len { blk[(o + j) / g] += buf[j] as f64; } o += len; }
                cold_units(&blk[..units], keep, &mut cu);
                None
            } else { Some(cold(inter, keep, &mut read, &mut buf, &mut h)) };
            // membership pass in index order: cold neurons, whole cold q4 groups, whole cold 64-blocks (stage A)
            let (mut cn, mut ties) = (0usize, 0usize);
            let (mut gcold, mut gall, mut bcold, mut ball, mut blen) = (0u64, true, 0u64, true, 0usize);
            let (mut grp_n, mut skip) = (0usize, 0u64);
            let mut o = 0;
            while o < inter {
                let len = CH.min(inter - o);
                read(o, &mut buf[..len]);
                for j in 0..len {
                    let i = o + j;
                    let c = match &pick { None => cu[i / g], Some(pk) => pk.cold(buf[j], &mut ties) };
                    if c { cn += 1; }
                    gall &= c; grp_n += 1;
                    if grp_n == group || i + 1 == inter { if gall && grp_n == group { gcold += 1; } gall = true; grp_n = 0; }
                    ball &= c; blen += 1;
                    if blen == MASK_BLOCK || i + 1 == inter {
                        if ball { skip += blen as u64 * per_neuron + if blen == group { per_group } else { 0 }; }
                        if ball { bcold += 1; }
                        ball = true; blen = 0;
                    }
                }
                o += len;
            }
            let _ = bcold;
            // heads (stack)
            let hr = self.heads_of(l);
            let nh = hr.len();
            let mut hm = [0f32; MAXH];
            self.read_sel(sel, hr.start, &mut hm[..nh], &mut b);
            let mut hmass = [0f64; MAXU];
            for i in 0..nh { hmass[i] = hm[i] as f64; }
            let ch = cold_units(&hmass[..nh], keep, &mut cu);
            let (ha, hb) = head_bytes(t, t.kinds[l]);
            p.cold_neurons += cn;
            p.per_layer_cold[l] = cn as u32;
            p.per_layer_cold_heads[l] = ch as u16;
            p.heads += nh;
            p.cold_heads += ch;
            p.ffn_bytes += cn as u64 * per_neuron + gcold * per_group;
            p.head_bytes += ch as u64 * hb;
            p.skip_bytes += skip + ch as u64 * ha;
        }
        p
    }
}

impl PlanOut {
    pub fn unload_bytes(&self) -> u64 { self.ffn_bytes + self.head_bytes }
    pub fn ms_a(&self) -> f64 { self.skip_bytes as f64 / READ_BW * 1e3 }
    pub fn ms_b(&self) -> f64 { self.unload_bytes() as f64 / READ_BW * 1e3 }
}

/// The recorder as a job source: FFN = live |a| sums x the down norms (disk), heads / resid / MTP live, token counters
/// live (all read in place: model memory, no host copy). `batch` = read the batch region instead (same layout).
pub struct TapSrc {
    pub tap: Arc<streamdense_model::activity::ActivityTap>,
    pub down: Option<File>,
    pub down_base: u64,
    pub n_ffn: usize,
    pub n_main: usize,
    pub o_out: usize,
    pub vocab: usize,
    pub batch: bool,
}

impl TapSrc {
    pub fn new(tap: Arc<streamdense_model::activity::ActivityTap>, st: &Store) -> TapSrc {
        TapSrc { tap, down: st.down_file(), down_base: st.down_base(), n_ffn: st.n_ffn(), n_main: st.o_mtp(), o_out: st.o_out(), vocab: st.d.vocab, batch: false }
    }
    /// The same view on the batch region (setup: clones the file handle).
    pub fn batch_view(&self) -> TapSrc {
        TapSrc { tap: self.tap.clone(), down: self.down.as_ref().and_then(|f| f.try_clone().ok()), batch: true, ..*self }
    }
    fn main<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R { if self.batch { self.tap.with_batch(f) } else { self.tap.with_live(f) } }
    fn mtp<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R { if self.batch { self.tap.with_mtp_batch(f) } else { self.tap.with_mtp_live(f) } }
}

impl JobSrc for TapSrc {
    fn rows(&self) -> (u64, u64) { (self.tap.counters().0 as u64, self.tap.mtp_counters().0 as u64) }
    fn read(&self, off: usize, out: &mut [f32]) {
        let mut i = 0;
        while i < out.len() {
            let g = off + i;
            let rest = out.len() - i;
            let o = &mut out[i..];
            let l = if g < self.n_ffn {
                let l = rest.min(self.n_ffn - g);
                Store::read_down(self.down.as_ref(), self.down_base, g, &mut o[..l]);
                self.main(|v| for j in 0..l { o[j] *= v.get(g + j).copied().unwrap_or(0.0); });
                l
            } else if g < self.n_main {
                let l = rest.min(self.n_main - g);
                self.main(|v| for j in 0..l { o[j] = v.get(g + j).copied().unwrap_or(0.0); });
                l
            } else if g < self.o_out {
                let l = rest.min(self.o_out - g);
                self.mtp(|v| for j in 0..l { o[j] = v.get(g - self.n_main + j).copied().unwrap_or(0.0); });
                l
            } else if g < self.o_out + 2 * self.vocab {
                let emb = g >= self.o_out + self.vocab;
                let base = if emb { self.o_out + self.vocab } else { self.o_out };
                let l = rest.min(base + self.vocab - g);
                if self.batch {
                    self.tap.with_batch_counts(|a, b| { let c = if emb { b } else { a }; for j in 0..l { o[j] = c.get(g - base + j).map_or(0.0, |x| *x as f32); } });
                } else {
                    let c = if emb { &self.tap.embed_counts } else { &self.tap.out_counts };
                    c.with_counts(|v| for j in 0..l { o[j] = v.get(g - base + j).map_or(0.0, |x| x.load(Relaxed) as f32); });
                }
                l
            } else { o.fill(0.0); rest };
            i += l;
        }
    }
}

/// Where same-label jobs are summed before one disk fold (batching): the recorder's batch region in model memory
/// (`TapBatch`), or a test double.
pub trait Batcher: Send + Sync {
    /// add one job (the recorder / `job`) into the batch
    fn add(&self, job: &dyn JobSrc);
    fn clear(&self);
    /// batch values [off, off + out.len()) of the unified layout
    fn read(&self, off: usize, out: &mut [f32]);
}
/// The recorder's batch region.
pub struct TapBatch(pub TapSrc);
impl Batcher for TapBatch {
    fn add(&self, _job: &dyn JobSrc) { self.0.tap.add_to_batch(); }
    fn clear(&self) { self.0.tap.reset_batch(); }
    fn read(&self, off: usize, out: &mut [f32]) { JobSrc::read(&self.0, off, out) }
}
/// A batch as a job source (rows counted by the caller).
pub struct BatchView<'a> { pub b: &'a dyn Batcher, pub rows: (u64, u64) }
impl JobSrc for BatchView<'_> {
    fn rows(&self) -> (u64, u64) { self.rows }
    fn read(&self, off: usize, out: &mut [f32]) { self.b.read(off, out) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::watcher::{label_names, MIXED};
    use streamdense_model::activity::ActivityStats;

    fn tmpdir(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_store_{}_{}", n, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn dims() -> Dims {
        let kinds = (0..4).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let t = ActivityStats { enabled: true, layers: 4, hidden: 128, inter: 256, kinds, dn_kh: 1, dn_dk: 128, dn_vh: 2, dn_dv: 128, heads: 3, kv_heads: 1,
                                head_dim: 128, conv_k: 4, group: 64, bits: 4, ..Default::default() };
        let mut d = Dims::from_stats(&t, 64);
        d.vocab = 100;
        d
    }
    struct V(Vec<f32>);
    impl JobSrc for V {
        fn rows(&self) -> (u64, u64) { (10, 0) }
        fn read(&self, off: usize, out: &mut [f32]) { for (j, o) in out.iter_mut().enumerate() { *o = self.0.get(off + j).copied().unwrap_or(0.0); } }
    }
    fn job(st: &Store, seed: f32) -> V { V((0..st.n()).map(|i| ((i % 7) as f32 + 1.0) * seed).collect()) }
    fn all(st: &Store, s: usize) -> Vec<f32> {
        let mut v = vec![0f32; st.n()];
        let mut rec = [0u8; CHB];
        st.read_sel(Sel::Set(s), 0, &mut v, &mut rec);
        v
    }
    /// cap holding exactly `k` label slots (+ the held slot and the fixed regions)
    fn cap_for(dir: &Path, k: u64) -> u64 {
        let st = Store::open_cap(dir, dims(), label_names(), MIXED, true, CAP_BYTES).unwrap();
        st.fixed_bytes() + k * st.lsz()
    }

    /// One file, pre-allocated to its fixed size at creation; folds, slot claims, held moves and reopen never change it.
    #[test]
    fn file_size_constant() {
        let dir = tmpdir("size");
        let cap = cap_for(&dir, 2) + 1000;
        let mut st = Store::open_cap(&dir, dims(), label_names(), MIXED, false, cap).unwrap();
        let total = st.total_bytes();
        assert_eq!((st.nslots(), st.disk_bytes()), (2, total));
        assert!(total <= cap);
        let len = || std::fs::metadata(Store::dir_of(&dir, &dims()).join("store.bin")).unwrap().len();
        for i in 0..6 { st.fold_job(i % 2, &job(&st, 1.0), i as u64, i as f64).unwrap(); assert_eq!(len(), total); }
        st.fold_pooled(&job(&st, 2.0), 7, 7.0).unwrap();
        st.set_held_owner(3).unwrap();
        st.fold_to(Some(HELD), &job(&st, 1.0), 2, 8, 8.0).unwrap();
        assert_eq!(st.claim(1, 3).unwrap(), Some(1));
        st.move_held(3).unwrap();
        assert_eq!((st.held_owner(), st.hdr(3).turns), (None, 2));
        st.fold_job(3, &job(&st, 1.0), 8, 8.0).unwrap();
        assert!(st.fold_job(4, &job(&st, 1.0), 9, 9.0).is_err(), "no free slot: the watcher replaces one first");
        assert_eq!(len(), total);
        drop(st);
        let st = Store::open_cap(&dir, dims(), label_names(), MIXED, false, CAP_BYTES).unwrap();
        assert_eq!((st.nslots(), len(), st.slot_label(0), st.slot_label(1)), (2, total, Some(0), Some(3)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A crash in the middle of a slot replace (after the unmap, or half-way through zeroing) recovers: the slot is
    /// free, the other slots are untouched, and the next claim starts from zeros (nothing of the old label leaks).
    #[test]
    fn crash_mid_replace_recovers() {
        for crash in [1u8, 2] {
            let dir = tmpdir(&format!("replace{}", crash));
            let cap = cap_for(&dir, 2);
            let mut st = Store::open_cap(&dir, dims(), label_names(), MIXED, false, cap).unwrap();
            let total = st.total_bytes();
            let (a, c, b) = (job(&st, 1.0), job(&st, 3.0), job(&st, 5.0));
            st.fold_job(0, &a, 1, 0.0).unwrap();
            st.fold_job(1, &c, 2, 0.0).unwrap();
            let a_before = all(&st, 0);
            st.crash = crash;
            assert!(st.claim(1, 2).is_err());
            drop(st);
            let mut st = Store::open_cap(&dir, dims(), label_names(), MIXED, false, cap).unwrap();
            assert_eq!((st.disk_bytes(), st.slot_label(0), st.slot_label(1)), (total, Some(0), None), "crash {}", crash);
            assert_eq!((all(&st, 0), st.hdr(0).turns), (a_before.clone(), 1));
            assert_eq!((st.has(1), st.hdr(1)), (false, Hdr::default()));
            st.fold_job(2, &b, 3, 1.0).unwrap();
            assert_eq!(st.slot_of(2), Some(1));
            assert_eq!(all(&st, 2), b.0, "the new label's slot holds only its own job");
            assert_eq!((st.hdr(2).turns, all(&st, 0)), (1, a_before));
            assert_eq!(st.disk_bytes(), total);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Held -> slot move: exact sums + counters; a crash in the middle of the move resumes on reopen without double
    /// counting (every chunk exactly slot + held once).
    #[test]
    fn move_held_exact_and_crash_resume() {
        for crash in [0u8, 3, 4] {
            let dir = tmpdir(&format!("move{}", crash));
            let mut st = Store::open_cap(&dir, dims(), label_names(), MIXED, false, cap_for(&dir, 2)).unwrap();
            let (a, b) = (job(&st, 1.0), job(&st, 7.0));
            st.fold_job(0, &a, 1, 0.0).unwrap();
            st.set_held_owner(0).unwrap();
            st.fold_to(Some(HELD), &b, 3, 2, 0.0).unwrap();
            let want: Vec<f32> = a.0.iter().zip(&b.0).map(|(x, y)| x + y).collect();
            st.crash = crash;
            let r = st.move_held(0);
            if crash > 0 {
                assert!(r.is_err());
                drop(st);
                st = Store::open_cap(&dir, dims(), label_names(), MIXED, false, CAP_BYTES).unwrap();
            }
            assert_eq!(all(&st, 0), want, "crash {}", crash);
            assert_eq!((st.hdr(0).turns, st.hdr(0).tokens, st.held_owner()), (4, 20, None));
            assert!(all(&st, HELD).iter().all(|x| *x == 0.0));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
