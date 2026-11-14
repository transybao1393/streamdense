//! draft_table: CPU-side, learning token lookup table that proposes speculative-decoding drafts.
//! The GPU verifies every draft, so nothing here affects correctness, only speed.
//!
//! Sources, tried longest n-gram (n = 4..1) first:
//! - `seq`: the current sequence (prompt + committed tokens) in a RAM ring + n-gram hash (prompt lookup, n=1..4).
//! - `hist`: the most recent generations across sessions in a RAM ring + n-gram hash (n=2..4), absolute
//!   positions = token index in `tokens.bin`.
//! - `disk`: everything ever generated, via an on-disk trigram hash (`index.sdt`) → `tokens.bin` offsets;
//!   consulted only when RAM has no n≥3 match, at most once per propose (2 preads), behind a 64-entry FIFO cache.
//! - `alt`: verifier top-3 per (last 2 tokens), last-resort single-token guess.
//!
//! Confidence: per-token conditional P(accept | earlier drafts accepted) = per-entry (acc,rej) blended with a
//! global calibrator per (source, n, depth); the returned value is the cumulative product ≈ P(this token accepted).
//!
//! Files in `dir` (append-only data + journal; the index is a lossy derived cache, every hit is re-verified):
//! - `tokens.bin`: raw u32 LE tokens; a generation is `SEP, last 4 prompt tokens, generated...`.
//! - `commit.log`: records `[len u32][crc32 u32][end u64, start u64, chunk_crc u32, calibrators]`. On open the
//!   last record whose chunk crc verifies wins; torn log tails and uncommitted token tails are truncated.
//! - `index.sdt`: 4 KB header + 2^21 × `[chk u32][pos u32]` slots in 64 B groups (16 MB, sparse on disk).
//!
//! RAM: every structure is allocated once in `open` and sized from `ram_budget_bytes` (see `ram_bytes`).
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::mem::size_of;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};

pub const SEP: u32 = u32::MAX;
const NMAX: usize = 4;
const PROBE: usize = 4;
const DISK_N: usize = 3;
const AUTO_FLUSH: u32 = 4096;
const DISK_SLOTS: u64 = 1 << 21;
const GROUP: u64 = 8;
const HDR: u64 = 4096;
const IDX_MAGIC: u64 = 0x3158_4449_5444_5344; // "SDTDIDX1"
const CONT: usize = 16;
const LRU: usize = 64;
const MAXP: usize = 64;
const MIN_CONF: f32 = 0.02;
const PRIOR_W: f32 = 4.0;
const SRCS: usize = 4;
const DEPTHS: usize = 8;
const NCAL: usize = SRCS * NMAX * DEPTHS;
const S_SEQ: u8 = 0;
const S_HIST: u8 = 1;
const S_DISK: u8 = 2;
const S_ALT: u8 = 3;
const TAG_RAM: u64 = 0;
const TAG_DISK: u64 = 0x5bd1_e995_0000_0000;
const TAG_ALT: u64 = 0x2545_f491_0000_0000;
const PAYLOAD: usize = 20 + NCAL * 8;
const CHUNK: usize = 16 << 10; // tokens per transient I/O buffer (64 KB)
pub const MIN_BUDGET: usize = 256 << 10;

// ---------- helpers ----------
const CRC: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 { let mut c = i as u32; let mut k = 0; while k < 8 { c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }; k += 1 } t[i] = c; i += 1 }
    t
};
fn crc_up(c: u32, b: &[u8]) -> u32 { !b.iter().fold(!c, |c, &x| CRC[((c ^ x as u32) & 0xff) as usize] ^ (c >> 8)) }
pub fn crc32(b: &[u8]) -> u32 { crc_up(0, b) }
fn pow2_floor(x: usize) -> usize { if x == 0 { 0 } else { 1 << (usize::BITS - 1 - x.leading_zeros()) } }
fn mix(h: u64, t: u32) -> u64 { (h ^ t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31) }
fn fin(h: u64, n: usize, tag: u64) -> u64 {
    let mut x = h ^ ((n as u64) << 56) ^ tag;
    x ^= x >> 33; x = x.wrapping_mul(0xff51_afd7_ed55_8ccd); x ^= x >> 33;
    x | 1
}
/// keys[n-1] = hash of the n-gram ending at the end of `last`; 0 = none (too short or contains SEP).
fn keys(last: &[u32], tag: u64) -> [u64; NMAX] {
    let mut k = [0u64; NMAX];
    let mut h = 0x243F_6A88_85A3_08D3u64;
    for n in 1..=NMAX.min(last.len()) {
        let t = last[last.len() - n];
        if t == SEP { break }
        h = mix(h, t); k[n - 1] = fin(h, n, tag);
    }
    k
}
fn chk(key: u64) -> u32 { (key >> 32) as u32 | 1 }
fn sig(tail: &[u32]) -> u64 { tail[tail.len().saturating_sub(8)..].iter().fold(tail.len().min(8) as u64, |h, &t| mix(h, t)) }
fn p0(n: usize, d: usize) -> f32 { if d > 0 { 0.8 } else { [0.15, 0.3, 0.45, 0.6][n - 1] } }
fn r32(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn r64(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }
fn es(e: io::Error) -> String { e.to_string() }

/// Read tokens [from, to) of `f` (u32 LE) into `out`; short file → fewer tokens.
fn read_toks(f: &File, from: u64, to: u64, bytes: &mut Vec<u8>, out: &mut Vec<u32>) -> io::Result<()> {
    out.clear();
    if to <= from { return Ok(()) }
    bytes.resize(((to - from) * 4) as usize, 0);
    let mut got = 0;
    while got < bytes.len() {
        let k = f.read_at(&mut bytes[got..], from * 4 + got as u64)?;
        if k == 0 { break }
        got += k;
    }
    out.extend(bytes[..got / 4 * 4].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
    Ok(())
}

// ---------- RAM ring + n-gram hash ----------
#[derive(Clone, Copy, Default)]
struct Slot { chk: u32, pos: u32, acc: u16, rej: u16 }
fn bump(acc: &mut u16, rej: &mut u16, ok: bool) {
    if *acc == u16::MAX || *rej == u16::MAX { *acc /= 2; *rej /= 2 }
    if ok { *acc += 1 } else { *rej += 1 }
}

/// Token ring (absolute positions) + open-addressing hash: n-gram ending at p-1 → continuation p (most recent wins).
struct Ring { toks: Vec<u32>, mask: u32, lo: u32, total: u32, sep_after: u32, slots: Vec<Slot>, nmin: usize }
impl Ring {
    fn new(bytes: usize, nmin: usize) -> Ring {
        let nn = NMAX - nmin + 1;
        let cap = pow2_floor(bytes / (4 + size_of::<Slot>() * nn)).max(64);
        let ns = (bytes.saturating_sub(cap * 4) / size_of::<Slot>()).max(PROBE);
        Ring { toks: vec![0; cap], mask: cap as u32 - 1, lo: 0, total: 0, sep_after: 0, slots: vec![Slot::default(); ns], nmin }
    }
    /// probe i of `key`: fastrange of the low 32 bits (chk uses the high 32)
    fn idx(&self, key: u64, i: usize) -> usize { let n = self.slots.len(); (((key as u32 as u64 * n as u64) >> 32) as usize + i) % n }
    fn bytes(&self) -> usize { self.toks.capacity() * 4 + self.slots.capacity() * size_of::<Slot>() }
    fn cap(&self) -> u32 { self.mask + 1 }
    fn reset_at(&mut self, p: u32) { self.slots.fill(Slot::default()); self.lo = p; self.total = p; self.sep_after = p }
    fn low(&self) -> u32 { self.lo.max(self.total.saturating_sub(self.cap())) }
    fn len(&self) -> u32 { self.total - self.low() }
    fn at(&self, p: u32) -> u32 { self.toks[(p & self.mask) as usize] }
    fn push(&mut self, t: u32) {
        let p = self.total;
        let avail = (p - self.low().max(self.sep_after)) as usize;
        if avail >= self.nmin {
            let m = avail.min(NMAX);
            let mut last = [0u32; NMAX];
            for i in 0..m { last[i] = self.at(p - m as u32 + i as u32) }
            let k = keys(&last[..m], TAG_RAM);
            for n in self.nmin..=m { self.insert(k[n - 1], p) }
        }
        self.toks[(p & self.mask) as usize] = t;
        self.total = p + 1;
        if t == SEP { self.sep_after = p + 1 }
    }
    fn insert(&mut self, key: u64, p: u32) {
        let c = chk(key);
        let (mut v, mut vp) = (0, u32::MAX);
        for i in 0..PROBE {
            let j = self.idx(key, i);
            let s = &mut self.slots[j];
            if s.chk == c { s.pos = p; return }
            if s.chk == 0 { v = j; break }
            if s.pos < vp { vp = s.pos; v = j }
        }
        self.slots[v] = Slot { chk: c, pos: p, acc: 0, rej: 0 };
    }
    /// (slot, continuation pos) of the most recent verified occurrence of the last n tokens of `last`.
    fn find(&self, last: &[u32], k: &[u64; NMAX], n: usize) -> Option<(usize, u32)> {
        let key = k[n - 1];
        if key == 0 || n < self.nmin { return None }
        let (c, lo) = (chk(key), self.low());
        for i in 0..PROBE {
            let j = self.idx(key, i);
            let s = self.slots[j];
            if s.chk == 0 { return None }
            if s.chk == c {
                let p = s.pos;
                let ok = p >= lo + n as u32 && p < self.total && (0..n).all(|i| self.at(p - n as u32 + i as u32) == last[last.len() - n + i]);
                return if ok { Some((j, p)) } else { None };
            }
        }
        None
    }
    fn used(&self) -> usize { self.slots.iter().filter(|s| s.chk != 0).count() }
}

#[derive(Clone, Copy, Default)]
struct Alt { chk: u32, t: [u32; 3], acc: u16, rej: u16 }
#[derive(Clone, Copy)]
struct Lru { key: u64, len: u8, toks: [u32; CONT] }
#[derive(Clone, Copy)]
struct Prop { tok: u32, src: u8, n: u8, d: u8, slot: u32 }
#[derive(Clone, Copy)]
struct Cand { src: u8, n: u8, slot: u32, pos: u32, tok: u32 }

#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// committed history tokens on disk / in the RAM history ring / in the current sequence ring
    pub disk_tokens: u64, pub ram_hist_tokens: u64, pub seq_tokens: u64,
    /// occupied hash slots in RAM
    pub seq_entries: usize, pub hist_entries: usize, pub alt_entries: usize,
    pub ram_bytes: usize, pub budget: usize, pub disk_bytes: u64,
    /// cap (tokens) of the RAM rings
    pub seq_cap: u32, pub hist_cap: u32,
    /// (accepted, scored) draft tokens per source seq/hist/disk/alt (calibrator counts, decayed)
    pub accept: [(u64, u64); SRCS],
    pub disk_reads: u64, pub torn_tail: bool,
}

pub struct DraftTable {
    dir: PathBuf, budget: usize,
    seq: Ring, hist: Ring, alt: Vec<Alt>, amask: usize,
    lru: Vec<Lru>, lru_next: usize,
    cal: [[u32; 2]; NCAL], cal_dirty: bool,
    last: Vec<Prop>, last_sig: u64, com: Vec<u32>, dbuf: [u32; CONT], dlen: usize,
    tokf: File, logf: File, idxf: File, flushed: u32, indexed: u32, log_len: u64,
    disk_reads: u64, torn: bool,
}

impl DraftTable {
    pub fn open(dir: &Path, ram_budget_bytes: usize) -> Result<Self, String> {
        if ram_budget_bytes < MIN_BUDGET { return Err(format!("draft_table: budget {} < {} bytes", ram_budget_bytes, MIN_BUDGET)) }
        fs::create_dir_all(dir).map_err(es)?;
        let fixed = size_of::<Self>() + LRU * size_of::<Lru>() + MAXP * size_of::<Prop>() + 4096;
        let na = pow2_floor(ram_budget_bytes / 32 / size_of::<Alt>()).max(256);
        let rest = ram_budget_bytes - fixed - na * size_of::<Alt>();
        let seq = Ring::new(rest / 4, 1);
        let hist = Ring::new(rest - rest / 4, 2);
        let op = |n: &str| OpenOptions::new().read(true).write(true).create(true).truncate(false).open(dir.join(n)).map_err(es);
        let mut t = DraftTable {
            dir: dir.to_path_buf(), budget: ram_budget_bytes, seq, hist, alt: vec![Alt::default(); na], amask: na - 1,
            lru: vec![Lru { key: 0, len: 0, toks: [0; CONT] }; LRU], lru_next: 0,
            cal: [[0; 2]; NCAL], cal_dirty: false,
            last: Vec::with_capacity(MAXP), last_sig: 0, com: Vec::with_capacity(MAXP), dbuf: [0; CONT], dlen: 0,
            tokf: op("tokens.bin")?, logf: op("commit.log")?, idxf: op("index.sdt")?, flushed: 0, indexed: 0, log_len: 0,
            disk_reads: 0, torn: false,
        };
        t.recover().map_err(es)?;
        Ok(t)
    }

    /// Replay commit.log, truncate torn tails, reload the RAM history ring, catch the disk index up.
    fn recover(&mut self) -> io::Result<()> {
        let len = self.logf.metadata()?.len();
        let mut recs: VecDeque<(u64, u64, u64, u32, Vec<u8>)> = VecDeque::new(); // (offset, end, start, crc, payload)
        let (mut off, mut hdr, mut pl) = (0u64, [0u8; 8], vec![0u8; PAYLOAD]);
        while off + 8 + PAYLOAD as u64 <= len {
            self.logf.read_exact_at(&mut hdr, off)?;
            if r32(&hdr, 0) as usize != PAYLOAD { break }
            self.logf.read_exact_at(&mut pl, off + 8)?;
            if crc32(&pl) != r32(&hdr, 4) { break }
            if recs.len() == 16 { recs.pop_front(); }
            recs.push_back((off, r64(&pl, 0), r64(&pl, 8), r32(&pl, 16), pl.clone()));
            off += 8 + PAYLOAD as u64;
        }
        self.torn = off != len;
        let tlen = self.tokf.metadata()?.len() / 4;
        let (mut bytes, mut toks) = (Vec::new(), Vec::new());
        let mut chosen = None;
        while let Some((o, end, start, c, p)) = recs.pop_back() {
            let mut cc = 0;
            let mut s = start;
            if end <= tlen {
                while s < end { let e = end.min(s + CHUNK as u64); read_toks(&self.tokf, s, e, &mut bytes, &mut toks)?; cc = crc_up(cc, &bytes); s = e }
                if cc == c { chosen = Some((o, end, p)); break }
            }
            self.torn = true;
        }
        let (log_end, end) = match chosen {
            Some((o, end, p)) => {
                for i in 0..NCAL { self.cal[i] = [r32(&p, 20 + i * 8), r32(&p, 24 + i * 8)] }
                (o + 8 + PAYLOAD as u64, end)
            }
            None => { if len > 0 { self.torn = true } (0, 0) }
        };
        if log_end != len { self.logf.set_len(log_end)? }
        let flen = self.tokf.metadata()?.len();
        if flen != end * 4 { self.torn |= flen > end * 4; self.tokf.set_len(end * 4)? }
        self.log_len = log_end;
        self.flushed = end as u32;
        // RAM history ring = last cap tokens
        let start = end.saturating_sub(self.hist.cap() as u64);
        self.hist.reset_at(start as u32);
        let mut s = start;
        while s < end { let e = end.min(s + CHUNK as u64); read_toks(&self.tokf, s, e, &mut bytes, &mut toks)?; for &x in &toks { self.hist.push(x) } s = e }
        // disk index
        let want = HDR + DISK_SLOTS * 8;
        let mut h = [0u8; 24];
        let ok = self.idxf.metadata()?.len() == want && self.idxf.read_exact_at(&mut h, 0).is_ok() && r64(&h, 0) == IDX_MAGIC && r64(&h, 8) == DISK_SLOTS;
        let mut indexed = if ok { r64(&h, 16).min(end) } else { self.idxf.set_len(0)?; self.idxf.set_len(want)?; 0 };
        if indexed < end { self.index_range(indexed, end)?; indexed = end }
        self.indexed = indexed as u32;
        self.write_idx_hdr(indexed)
    }

    fn write_idx_hdr(&self, indexed: u64) -> io::Result<()> {
        let mut h = [0u8; 24];
        h[0..8].copy_from_slice(&IDX_MAGIC.to_le_bytes()); h[8..16].copy_from_slice(&DISK_SLOTS.to_le_bytes()); h[16..24].copy_from_slice(&indexed.to_le_bytes());
        self.idxf.write_all_at(&h, 0)
    }

    /// Insert disk trigram keys for continuation positions [a, b) (tokens already in tokens.bin).
    fn index_range(&mut self, a: u64, b: u64) -> io::Result<()> {
        let (mut bytes, mut toks) = (Vec::new(), Vec::new());
        let mut s = a.max(DISK_N as u64);
        while s < b {
            let e = b.min(s + CHUNK as u64);
            read_toks(&self.tokf, s - DISK_N as u64, e, &mut bytes, &mut toks)?;
            for (i, w) in toks.windows(DISK_N).enumerate() {
                let p = s + i as u64;
                if p >= e { break }
                let k = keys(w, TAG_DISK)[DISK_N - 1];
                if k != 0 { self.disk_insert(k, p as u32)? }
            }
            s = e;
        }
        Ok(())
    }
    fn group_off(key: u64) -> u64 { HDR + (key & (DISK_SLOTS / GROUP - 1)) * GROUP * 8 }
    fn disk_insert(&self, key: u64, p: u32) -> io::Result<()> {
        let (o, c) = (Self::group_off(key), chk(key));
        let mut g = [0u8; (GROUP * 8) as usize];
        self.idxf.read_exact_at(&mut g, o)?;
        let (mut v, mut vp) = (0usize, u32::MAX);
        for i in 0..GROUP as usize {
            let (sc, sp) = (r32(&g, i * 8), r32(&g, i * 8 + 4));
            if sc == c { v = i; break }
            if sc == 0 { v = i; break }
            if sp < vp { vp = sp; v = i }
        }
        let mut s = [0u8; 8];
        s[..4].copy_from_slice(&c.to_le_bytes()); s[4..].copy_from_slice(&p.to_le_bytes());
        self.idxf.write_all_at(&s, o + v as u64 * 8)
    }

    /// Disk trigram lookup → continuation in self.dbuf (≤ CONT tokens). One cache probe, else ≤ 2 preads.
    fn disk_find(&mut self, last: &[u32]) -> bool {
        let key = keys(&last[last.len() - DISK_N..], TAG_DISK)[DISK_N - 1];
        if key == 0 { return false }
        if let Some(e) = self.lru.iter().find(|e| e.key == key) {
            self.dlen = e.len as usize; self.dbuf = e.toks; return self.dlen > 0;
        }
        self.disk_reads += 1;
        self.dlen = 0;
        let (o, c) = (Self::group_off(key), chk(key));
        let mut g = [0u8; (GROUP * 8) as usize];
        if self.idxf.read_exact_at(&mut g, o).is_ok() {
            let pos = (0..GROUP as usize).filter(|&i| r32(&g, i * 8) == c).map(|i| r32(&g, i * 8 + 4)).max();
            if let Some(p) = pos.filter(|&p| p as usize >= DISK_N && p < self.flushed) {
                let mut b = [0u8; (DISK_N + CONT) * 4];
                let from = (p as u64 - DISK_N as u64) * 4;
                let n = (self.flushed as u64 * 4 - from).min(b.len() as u64) as usize;
                if self.tokf.read_exact_at(&mut b[..n], from).is_ok() && (0..DISK_N).all(|i| r32(&b, i * 4) == last[last.len() - DISK_N + i]) {
                    for i in DISK_N..n / 4 { let t = r32(&b, i * 4); if t == SEP { break } self.dbuf[self.dlen] = t; self.dlen += 1 }
                }
            }
        }
        self.lru[self.lru_next] = Lru { key, len: self.dlen as u8, toks: self.dbuf };
        self.lru_next = (self.lru_next + 1) % LRU;
        self.dlen > 0
    }

    fn ci(src: u8, n: usize, d: usize) -> usize { (src as usize * NMAX + n - 1) * DEPTHS + d.min(DEPTHS - 1) }
    fn conf(&self, src: u8, n: usize, d: usize, e: Option<(u16, u16)>) -> f32 {
        let [a, t] = self.cal[Self::ci(src, n, d)];
        let g = (a as f32 + PRIOR_W * p0(n, d)) / (t as f32 + PRIOR_W);
        match e { Some((a, r)) => (a as f32 + PRIOR_W * g) / ((a + r) as f32 + PRIOR_W), None => g }
    }
    fn alt_slot(last: &[u32]) -> Option<u64> { if last.len() < 2 { None } else { Some(keys(&last[last.len() - 2..], TAG_ALT)[1]).filter(|&k| k != 0) } }

    fn best(&mut self, last: &[u32], disk_used: &mut bool) -> Option<Cand> {
        let k = keys(last, TAG_RAM);
        for n in (1..=last.len().min(NMAX)).rev() {
            if n == DISK_N - 1 && !*disk_used && last.len() >= DISK_N {
                *disk_used = true;
                if self.disk_find(last) { return Some(Cand { src: S_DISK, n: DISK_N as u8, slot: u32::MAX, pos: 0, tok: 0 }) }
            }
            let a = self.seq.find(last, &k, n).map(|(j, p)| { let s = self.seq.slots[j]; (self.conf(S_SEQ, n, 0, Some((s.acc, s.rej))), S_SEQ, j, p) });
            let b = self.hist.find(last, &k, n).map(|(j, p)| { let s = self.hist.slots[j]; (self.conf(S_HIST, n, 0, Some((s.acc, s.rej))), S_HIST, j, p) });
            let pick = match (a, b) { (Some(a), Some(b)) => Some(if b.0 > a.0 { b } else { a }), (a, b) => a.or(b) };
            if let Some((_, src, j, p)) = pick { return Some(Cand { src, n: n as u8, slot: j as u32, pos: p, tok: 0 }) }
        }
        let key = Self::alt_slot(last)?;
        let j = key as usize & self.amask;
        let e = self.alt[j];
        (e.chk == chk(key)).then_some(Cand { src: S_ALT, n: 2, slot: j as u32, pos: 0, tok: e.t[0] })
    }

    /// (token, confidence 0..1) continuation of context_tail, longest/most trusted match first; empty = no idea.
    /// Confidence is cumulative: ≈ P(this and all earlier drafted tokens are accepted).
    pub fn propose(&mut self, context_tail: &[u32], max: usize) -> Vec<(u32, f32)> {
        self.last.clear();
        self.last_sig = sig(context_tail);
        let max = max.min(MAXP);
        let mut out = Vec::with_capacity(max);
        let mut ext = [0u32; NMAX];
        let mut el = context_tail.len().min(NMAX);
        ext[NMAX - el..].copy_from_slice(&context_tail[context_tail.len() - el..]);
        let (mut cum, mut disk_used) = (1f32, false);
        'outer: for _ in 0..4 {
            if out.len() >= max || el == 0 { break }
            let Some(c) = self.best(&ext[NMAX - el..], &mut disk_used) else { break };
            let n = c.n as usize;
            for d in 0.. {
                let t = match c.src {
                    S_SEQ => { let p = c.pos + d as u32; if p < self.seq.total { self.seq.at(p) } else { break } }
                    S_HIST => { let p = c.pos + d as u32; if p < self.hist.total { self.hist.at(p) } else { break } }
                    S_DISK => { if d < self.dlen { self.dbuf[d] } else { break } }
                    _ => { if d == 0 { c.tok } else { break } }
                };
                if t == SEP { break 'outer }
                let e = match c.src {
                    S_SEQ => { let s = self.seq.slots[c.slot as usize]; Some((s.acc, s.rej)) }
                    S_HIST => { let s = self.hist.slots[c.slot as usize]; Some((s.acc, s.rej)) }
                    S_ALT => { let s = self.alt[c.slot as usize]; Some((s.acc, s.rej)) }
                    _ => None,
                };
                cum *= self.conf(c.src, n, d, e);
                if cum < MIN_CONF { break 'outer }
                out.push((t, cum));
                self.last.push(Prop { tok: t, src: c.src, n: c.n, d: d.min(255) as u8, slot: c.slot });
                ext.copy_within(1.., 0); ext[NMAX - 1] = t; el = (el + 1).min(NMAX);
                if out.len() >= max { break 'outer }
            }
        }
        out
    }

    /// After each verify. Committed tokens = `accepted` + the verifier's top-1 of row `accepted.len()` (the
    /// bonus/correction token) when that row is present. Row i of `verifier_top3` is the verifier's top-3 for the
    /// token after context_tail + committed[..i]; u32::MAX = unknown.
    pub fn observe(&mut self, context_tail: &[u32], accepted: &[u32], verifier_top3: &[[u32; 3]]) {
        self.com.clear();
        self.com.extend_from_slice(accepted);
        if let Some(r) = verifier_top3.get(accepted.len()) { if r[0] != SEP { self.com.push(r[0]) } }
        // score the last proposal (conditional: stop at the first miss; unverified positions are not scored)
        if !self.last.is_empty() && sig(context_tail) == self.last_sig {
            for j in 0..self.last.len().min(self.com.len()) {
                let p = self.last[j];
                let ok = p.tok == self.com[j];
                let c = &mut self.cal[Self::ci(p.src, p.n as usize, p.d as usize)];
                if c[1] >= 1 << 16 { c[0] /= 2; c[1] /= 2 }
                c[0] += ok as u32; c[1] += 1;
                let s = p.slot as usize;
                match p.src {
                    S_SEQ => { let x = &mut self.seq.slots[s]; bump(&mut x.acc, &mut x.rej, ok) }
                    S_HIST => { let x = &mut self.hist.slots[s]; bump(&mut x.acc, &mut x.rej, ok) }
                    S_ALT => { let x = &mut self.alt[s]; bump(&mut x.acc, &mut x.rej, ok) }
                    _ => {}
                }
                if !ok { break }
            }
            self.cal_dirty = true;
        }
        self.last.clear();
        // verifier top-3 per row → alt table keyed by the row's last 2 context tokens
        for (i, r) in verifier_top3.iter().enumerate().take(self.com.len()) {
            if r[0] == SEP { continue }
            let mut l = [0u32; 2];
            let m = (context_tail.len() + i).min(2);
            for q in 0..m { let x = context_tail.len() + i - m + q; l[q] = if x < context_tail.len() { context_tail[x] } else { self.com[x - context_tail.len()] } }
            let Some(key) = Self::alt_slot(&l[..m]) else { continue };
            let j = key as usize & self.amask;
            let e = &mut self.alt[j];
            if e.chk == chk(key) && e.t[0] == r[0] { e.t = *r } else { *e = Alt { chk: chk(key), t: *r, acc: 0, rej: 0 } }
        }
        for i in 0..self.com.len() { let t = self.com[i]; self.seq.push(t); self.hist.push(t) }
        if self.hist.total - self.flushed >= AUTO_FLUSH.min(self.hist.cap() / 2) { self.save(false) }
    }

    /// New generation: index the prompt for prompt lookup; history gets `SEP + last 4 prompt tokens`.
    pub fn begin_sequence(&mut self, prompt: &[u32]) {
        self.last.clear();
        self.seq.reset_at(0);
        for &t in &prompt[prompt.len().saturating_sub(self.seq.cap() as usize)..] { self.seq.push(t) }
        self.hist.push(SEP);
        for &t in &prompt[prompt.len().saturating_sub(NMAX)..] { self.hist.push(t) }
        if self.hist.total - self.flushed >= AUTO_FLUSH.min(self.hist.cap() / 2) { self.save(false) }
    }

    /// Persist new history tokens + calibrators (append-only, no fsync); cheap when nothing changed.
    /// Hot-path auto-saves (every AUTO_FLUSH history tokens) only append tokens + commit; the disk index catches up here.
    pub fn flush(&mut self) { self.save(true) }
    fn save(&mut self, index: bool) { if let Err(e) = self.flush_io(index) { eprintln!("draft_table: flush {}: {}", self.dir.display(), e) } }
    fn flush_io(&mut self, index: bool) -> io::Result<()> {
        let (a, b) = (self.flushed, self.hist.total);
        if a == b && !self.cal_dirty && (!index || self.indexed == b) { return Ok(()) }
        if a != b || self.cal_dirty { self.commit(a, b)? }
        if !index { return Ok(()) }
        self.index_range(self.indexed as u64, b as u64)?;
        self.indexed = b;
        self.write_idx_hdr(b as u64)
    }
    /// Append history tokens [a, b) to tokens.bin, then a commit record (end, start, chunk crc, calibrators).
    fn commit(&mut self, a: u32, b: u32) -> io::Result<()> {
        let mut crc = 0;
        let mut buf = Vec::with_capacity(CHUNK.min((b - a) as usize) * 4);
        let mut s = a;
        while s < b {
            let e = b.min(s + CHUNK as u32);
            buf.clear();
            for p in s..e { buf.extend_from_slice(&self.hist.at(p).to_le_bytes()) }
            self.tokf.write_all_at(&buf, s as u64 * 4)?;
            crc = crc_up(crc, &buf); s = e;
        }
        let mut pl = Vec::with_capacity(8 + PAYLOAD);
        pl.extend_from_slice(&(PAYLOAD as u32).to_le_bytes()); pl.extend_from_slice(&[0; 4]);
        pl.extend_from_slice(&(b as u64).to_le_bytes()); pl.extend_from_slice(&(a as u64).to_le_bytes()); pl.extend_from_slice(&crc.to_le_bytes());
        for c in &self.cal { pl.extend_from_slice(&c[0].to_le_bytes()); pl.extend_from_slice(&c[1].to_le_bytes()) }
        let c = crc32(&pl[8..]);
        pl[4..8].copy_from_slice(&c.to_le_bytes());
        self.logf.write_all_at(&pl, self.log_len)?;
        self.log_len += pl.len() as u64;
        self.flushed = b; self.cal_dirty = false;
        Ok(())
    }

    /// Bytes of every RAM structure (fixed after open).
    pub fn ram_bytes(&self) -> usize {
        size_of::<Self>() + self.seq.bytes() + self.hist.bytes() + self.alt.capacity() * size_of::<Alt>()
            + self.lru.capacity() * size_of::<Lru>() + self.last.capacity() * size_of::<Prop>() + self.com.capacity() * 4
    }

    pub fn stats(&self) -> Stats {
        let mut acc = [(0u64, 0u64); SRCS];
        for (i, c) in self.cal.iter().enumerate() { let s = i / (NMAX * DEPTHS); acc[s].0 += c[0] as u64; acc[s].1 += c[1] as u64 }
        let fl = |n: &str| fs::metadata(self.dir.join(n)).map(|m| m.len()).unwrap_or(0);
        Stats {
            disk_tokens: self.flushed as u64, ram_hist_tokens: self.hist.len() as u64, seq_tokens: self.seq.len() as u64,
            seq_entries: self.seq.used(), hist_entries: self.hist.used(), alt_entries: self.alt.iter().filter(|e| e.chk != 0).count(),
            ram_bytes: self.ram_bytes(), budget: self.budget,
            disk_bytes: fl("tokens.bin") + fl("commit.log") + fs::metadata(self.dir.join("index.sdt")).map(|m| m.blocks() * 512).unwrap_or(0),
            seq_cap: self.seq.cap(), hist_cap: self.hist.cap(), accept: acc, disk_reads: self.disk_reads, torn_tail: self.torn,
        }
    }

    /// Occupied slots of the on-disk index (full 16 MB scan in 64 KB reads; for stats only).
    pub fn disk_index_entries(&self) -> u64 {
        let mut b = vec![0u8; 64 << 10];
        let (mut o, end, mut n) = (HDR, HDR + DISK_SLOTS * 8, 0u64);
        while o < end {
            if self.idxf.read_exact_at(&mut b, o).is_err() { break }
            n += b.chunks_exact(8).filter(|s| r32(s, 0) != 0).count() as u64;
            o += b.len() as u64;
        }
        n
    }
}

impl Drop for DraftTable { fn drop(&mut self) { self.flush() } }

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;
    static N: AtomicU32 = AtomicU32::new(0);
    struct Tmp(PathBuf);
    impl Tmp { fn new() -> Tmp { let p = std::env::temp_dir().join(format!("sd_draft_{}_{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst))); let _ = fs::remove_dir_all(&p); Tmp(p) } }
    impl Drop for Tmp { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    fn toks(p: &[(u32, f32)]) -> Vec<u32> { p.iter().map(|x| x.0).collect() }
    /// run a "generation": greedy decode of `gen` with the table drafting; returns accepted draft tokens
    fn run(t: &mut DraftTable, prompt: &[u32], gen: &[u32], k: usize) -> usize {
        t.begin_sequence(prompt);
        let mut ctx = prompt.to_vec();
        let (mut i, mut hit) = (0, 0);
        while i < gen.len() {
            let p = toks(&t.propose(&ctx, k));
            let mut a = 0;
            while a < p.len() && i + a < gen.len() && p[a] == gen[i + a] { a += 1 }
            hit += a;
            let acc = &gen[i..i + a];
            let mut top3: Vec<[u32; 3]> = acc.iter().map(|&x| [x, SEP, SEP]).collect();
            if i + a < gen.len() { top3.push([gen[i + a], SEP, SEP]) }
            t.observe(&ctx, acc, &top3);
            let n = top3.len();
            ctx.extend_from_slice(&gen[i..i + n]);
            i += n;
        }
        hit
    }
    fn lcg(seed: &mut u64, m: u32) -> u32 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*seed >> 33) % m as u64) as u32 }

    #[test]
    fn prompt_lookup_copies_spans() {
        let d = Tmp::new();
        let mut t = DraftTable::open(&d.0, 4 << 20).unwrap();
        let prompt: Vec<u32> = (100..200).chain([7, 8, 9]).collect();
        t.begin_sequence(&prompt);
        let p = t.propose(&[1, 2, 140, 141, 142], 8);
        assert_eq!(toks(&p), (143..151).collect::<Vec<_>>());
        assert!(p.windows(2).all(|w| w[1].1 <= w[0].1) && p[0].1 > 0.0 && p[0].1 <= 1.0);
        assert!(t.propose(&[5000, 5001], 8).is_empty());
    }

    #[test]
    fn confidence_learns() {
        let d = Tmp::new();
        let mut t = DraftTable::open(&d.0, 4 << 20).unwrap();
        let prompt: Vec<u32> = (100..200).collect();
        t.begin_sequence(&prompt);
        let tail = [140u32, 141, 142];
        let c0 = t.propose(&tail, 4)[0].1;
        for _ in 0..50 { t.propose(&tail, 4); t.observe(&tail, &[143, 144, 145, 146], &[[143, 1, 2]; 4]) }
        let c1 = t.propose(&tail, 4)[0].1;
        for _ in 0..1000 { t.propose(&tail, 4); t.observe(&tail, &[], &[[9, 1, 2]]) }
        let c2 = t.propose(&tail, 4)[0].1;
        eprintln!("conf: fresh {c0:.3} after-accepts {c1:.3} after-rejects {c2:.3}");
        assert!(c1 > 0.9 && c1 > c0 && c2 < c1 / 2.0);
    }

    #[test]
    fn cross_session_learning() {
        let d = Tmp::new();
        let tool: Vec<u32> = (1000..1040).collect(); // a tool-call format generated earlier
        {
            let mut t = DraftTable::open(&d.0, 4 << 20).unwrap();
            run(&mut t, &[1, 2, 3, 4, 5], &tool, 8);
            t.flush();
        }
        let mut t = DraftTable::open(&d.0, 4 << 20).unwrap();
        assert_eq!(t.stats().disk_tokens, 5 + 40);
        t.begin_sequence(&[9, 9, 9]);
        assert_eq!(toks(&t.propose(&[9, 1000, 1001, 1002], 8)), (1003..1011).collect::<Vec<_>>());
        let hit = run(&mut t, &[1, 2, 3, 4, 5], &tool, 8);
        assert!(hit >= 30, "hit {hit}");
    }

    #[test]
    fn disk_index_serves_evicted_history() {
        let d = Tmp::new();
        let old: Vec<u32> = (50_000..50_030).collect();
        {
            let mut t = DraftTable::open(&d.0, MIN_BUDGET).unwrap();
            run(&mut t, &[1, 2, 3], &old, 8);
            let mut s = 7u64;
            for _ in 0..20 { let g: Vec<u32> = (0..1000).map(|_| lcg(&mut s, 30_000)).collect(); run(&mut t, &[4, 5, 6], &g, 4); }
            assert!(t.hist.low() > 30, "old sequence must be evicted from RAM");
        }
        let mut t = DraftTable::open(&d.0, MIN_BUDGET).unwrap();
        t.begin_sequence(&[8]);
        let p = toks(&t.propose(&[50_010, 50_011, 50_012], 32));
        assert!(p.len() >= 8 && p.len() <= CONT, "{p:?}");
        assert_eq!(p, (50_013..50_013 + p.len() as u32).collect::<Vec<_>>());
        assert_eq!(t.disk_reads, 1);
        t.propose(&[50_010, 50_011, 50_012], 32);
        assert_eq!(t.disk_reads, 1, "second lookup served from cache");
        assert!(t.disk_index_entries() > 20_000);
    }

    #[test]
    fn ram_budget_respected() {
        for b in [MIN_BUDGET, 1 << 20, 4 << 20, 32 << 20] {
            let d = Tmp::new();
            let mut t = DraftTable::open(&d.0, b).unwrap();
            let mut s = 1u64;
            for _ in 0..5 { let pr: Vec<u32> = (0..20_000).map(|_| lcg(&mut s, 500)).collect(); let g: Vec<u32> = (0..2000).map(|_| lcg(&mut s, 500)).collect(); run(&mut t, &pr, &g, 8); }
            let st = t.stats();
            eprintln!("budget {} → ram {} (seq cap {}, hist cap {})", b, st.ram_bytes, st.seq_cap, st.hist_cap);
            assert!(st.ram_bytes <= b && st.ram_bytes > b * 3 / 4, "{} vs {}", st.ram_bytes, b);
        }
    }

    #[test]
    fn torn_tail_recovery() {
        let d = Tmp::new();
        let g: Vec<u32> = (2000..2050).collect();
        {
            let mut t = DraftTable::open(&d.0, 1 << 20).unwrap();
            run(&mut t, &[1, 2, 3], &g, 8);
            t.flush();
        }
        let n0 = DraftTable::open(&d.0, 1 << 20).unwrap().stats().disk_tokens;
        // torn commit record + uncommitted token garbage
        let ap = |f: &str, b: &[u8]| { use std::io::Write; OpenOptions::new().append(true).open(d.0.join(f)).unwrap().write_all(b).unwrap() };
        ap("commit.log", &[(PAYLOAD as u8), 4, 0, 0, 1, 2, 3]);
        ap("tokens.bin", &[9; 13]);
        {
            let mut t = DraftTable::open(&d.0, 1 << 20).unwrap();
            let s = t.stats();
            assert!(s.torn_tail);
            assert_eq!(s.disk_tokens, n0);
            assert_eq!(toks(&t.propose(&[2010, 2011, 2012], 4)), vec![2013, 2014, 2015, 2016]);
            run(&mut t, &[7], &[1, 2, 3, 4], 4);
        }
        // tokens.bin lost its last chunk (commit reached disk, data did not) → fall back to the previous record
        let n1 = DraftTable::open(&d.0, 1 << 20).unwrap().stats().disk_tokens;
        assert!(n1 > n0);
        let f = OpenOptions::new().write(true).open(d.0.join("tokens.bin")).unwrap();
        f.set_len(n1 * 4 - 8).unwrap();
        let t = DraftTable::open(&d.0, 1 << 20).unwrap();
        assert!(t.stats().torn_tail);
        assert_eq!(t.stats().disk_tokens, n0);
        assert_eq!(fs::metadata(d.0.join("tokens.bin")).unwrap().len(), n0 * 4);
    }

    #[test]
    fn hot_path_timing() {
        let d = Tmp::new();
        let mut t = DraftTable::open(&d.0, 32 << 20).unwrap();
        let mut s = 3u64;
        // 300K tokens of history built from 200 repeated "phrases" + noise; 16K-token prompt
        let phr: Vec<Vec<u32>> = (0..200).map(|_| (0..24).map(|_| lcg(&mut s, 150_000)).collect()).collect();
        let mut gen = Vec::new();
        while gen.len() < 300_000 { if lcg(&mut s, 3) == 0 { gen.push(lcg(&mut s, 150_000)) } else { gen.extend_from_slice(&phr[lcg(&mut s, 200) as usize]) } }
        let prompt: Vec<u32> = gen[..16_384].to_vec();
        let t0 = Instant::now();
        t.begin_sequence(&prompt);
        let tb = t0.elapsed().as_secs_f64() * 1e3;
        let (mut mp, mut mo) = (0f64, 0f64);
        let (mut tp, mut to, mut rounds, mut ctx, mut i, mut hit) = (0f64, 0f64, 0, prompt.clone(), 16_384, 0);
        while i < gen.len() - 64 {
            let a0 = Instant::now();
            let p = toks(&t.propose(&ctx[ctx.len() - 8..], 16));
            let e = a0.elapsed().as_secs_f64(); tp += e; mp = mp.max(e);
            let mut a = 0;
            while a < p.len() && p[a] == gen[i + a] { a += 1 }
            hit += a;
            let top3: Vec<[u32; 3]> = gen[i..=i + a].iter().map(|&x| [x, x + 1, x + 2]).collect();
            let a1 = Instant::now();
            t.observe(&ctx[ctx.len() - 8..], &gen[i..i + a], &top3);
            let e = a1.elapsed().as_secs_f64(); to += e; mo = mo.max(e);
            ctx.extend_from_slice(&gen[i..=i + a]);
            i += a + 1; rounds += 1;
        }
        let f0 = Instant::now(); t.flush(); let tf = f0.elapsed().as_secs_f64() * 1e3;
        eprintln!("final flush (index catch-up) {tf:.1} ms");
        let (up, uo) = (tp / rounds as f64 * 1e6, to / rounds as f64 * 1e6);
        eprintln!("rounds {rounds} begin_sequence(16K) {tb:.2} ms | propose {up:.2} µs | observe {uo:.2} µs (incl. auto-flush) | max propose {:.0} µs observe {:.0} µs | accepted/round {:.2} | {:?}",
            mp * 1e6, mo * 1e6, hit as f64 / rounds as f64, t.stats());
        let lim = if cfg!(debug_assertions) { 2000.0 } else { 500.0 };
        assert!(up < lim && uo < lim, "propose {up} µs observe {uo} µs");
    }
}
