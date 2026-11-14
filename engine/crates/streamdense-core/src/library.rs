//! Answer library (watcher `library = true`, OFF by default): one compact record per gateway turn, (prompt FFN
//! activity -> answer FFN activity), in one fixed-size ring file. Read offline (`streamdense library`, forheader
//! `g16 --mode lib`) to test "predict the answer's neurons from similar past prompts".
//!
//! Values: per layer l and neuron j, v = Σ_rows |a_lj| · ‖W_down[:, j]‖ (the watcher's contribution measure).
//! - prompt = the rows fed before the first generated token (prefill, snapshot segments included);
//! - answer = every main-model row after it (decode, MTP verify rows incl. rejected drafts: the recorder sums all rows
//!   of a pass; `answer_rows` vs `n_new` shows the share).
//! Quantisation (measured on 12 real 27B prompts): u8 log codes per layer relative to
//! the layer max, 16 steps per octave, 255 codes = 15.9 octaves (below = 0). Answer mass kept by own-answer top-K
//! 0.7728 vs 0.7730 exact, kNN mass identical to 4 decimals; u4 / u5 / u6 lose 0.002..0.04 and were rejected.
//! Prompt sketch (for kNN without reading the full vectors): per layer a count sketch of the layer-normalised prompt
//! vector (sum 1, times inter) into `SKETCH_D` f16 buckets (hash + sign per neuron); Σ_l <sketch_l, sketch'_l>
//! approximates the exact "pa" Gram (Spearman 0.998 vs exact cosines, max error 0.002, same top-3 neighbours).
//!
//! File `library_<tag>.sdl` (little-endian):
//! - header (4096 B): magic "SDLIB001", version u32, layers u32, inter u32, sketch u32, slot_bytes u64, slots u64,
//!   created f64, seed u64.
//! - `slots` fixed-size slots of `slot_bytes` (4096-aligned) at 4096 + i · slot_bytes; written in order of free slot,
//!   then oldest record (min seq): a ring, oldest dropped first. The file grows to the cap, never past it.
//! - slot: record header (`REC_HDR` = 4096 B: magic "SDLREC01", seq u64 (0 = empty), ts f64, checksum u64 of the
//!   payload, meta, optional text) | f32 [4][layers] (prompt max, answer max, prompt sum, answer sum) | u8 prompt
//!   codes [layers][inter] | u8 answer codes [layers][inter] | f16 sketch [layers][sketch].
//! A slot is written with one pwrite; a torn / crashed write fails its checksum and is skipped by readers.
//! 27B (64 x 17,408, sketch 1024): slot 2,367,488 B (2.26 MiB) per turn: 453 turns per GiB, 2.37 GB per 1,000 turns.
use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Result};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::f16::{f16_to_f32, f32_to_f16};

pub const MAGIC: &[u8; 8] = b"SDLIB001";
pub const REC_MAGIC: &[u8; 8] = b"SDLREC01";
pub const VERSION: u32 = 1;
pub const FILE_HDR: u64 = 4096;
pub const REC_HDR: usize = 4096;
/// optional prompt text (`library_text = true` only) inside the record header
pub const TEXT_OFF: usize = 256;
pub const TEXT_MAX: usize = REC_HDR - TEXT_OFF;
/// count-sketch buckets per layer
pub const SKETCH_D: usize = 1024;
pub const SEED: u64 = 0x5d1b_2026_1103_0001;
/// log2 quantisation: codes per octave
pub const STEPS: f32 = 16.0;
const NAME_MAX: usize = 23;

fn ie(m: impl Into<String>) -> Error { Error::new(ErrorKind::InvalidData, m.into()) }

/// Record geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims { pub layers: usize, pub inter: usize, pub sketch: usize }
impl Dims {
    pub fn new(layers: usize, inter: usize) -> Dims { Dims { layers, inter, sketch: SKETCH_D } }
    pub fn n(&self) -> usize { self.layers * self.inter }
    pub fn tag(&self) -> String { format!("L{}_I{}_S{}", self.layers, self.inter, self.sketch) }
    pub fn o_scale(&self) -> usize { REC_HDR }
    pub fn o_prompt(&self) -> usize { REC_HDR + 16 * self.layers }
    pub fn o_answer(&self) -> usize { self.o_prompt() + self.n() }
    pub fn o_sketch(&self) -> usize { self.o_answer() + self.n() }
    pub fn slot_bytes(&self) -> usize { (self.o_sketch() + 2 * self.layers * self.sketch).div_ceil(4096) * 4096 }
    /// slots that fit in `cap` bytes of file (header included)
    pub fn slots_for(&self, cap: u64) -> usize { (cap.saturating_sub(FILE_HDR) / self.slot_bytes() as u64) as usize }
}

/// Which vector of a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part { Prompt, Answer }

/// Record metadata (the record header minus the checksum). No prompt text unless `text` is Some (config switch).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Meta {
    /// 1-based write order (0 = empty slot)
    pub seq: u64,
    pub ts: f64,
    /// the watcher's label of this turn and the domain active when it ran (coarse names, <= 23 bytes)
    pub label: String,
    pub active: String,
    /// recorder rows: prompt = before the first generated token, answer = after it (verify rows incl. rejected drafts)
    pub prompt_rows: u32,
    pub answer_rows: u32,
    /// tokens: whole prompt, reused from the KV / a snapshot, fed (prefilled) this turn, generated (incl. a stop token)
    pub n_prompt: u32,
    pub n_reused: u32,
    pub n_feed: u32,
    pub n_new: u32,
    /// MTP decode and its counters (passes, accepted, rejects)
    pub mtp: bool,
    pub spec_passes: u32,
    pub spec_accepted: u32,
    pub spec_rejects: u32,
    pub text: Option<String>,
}

fn put_str(b: &mut [u8], s: &str) {
    let mut n = s.len().min(b.len());
    while !s.is_char_boundary(n) { n -= 1; }
    b[..n].copy_from_slice(&s.as_bytes()[..n]);
    b[n..].fill(0);
}
fn get_str(b: &[u8]) -> String { let n = b.iter().position(|&x| x == 0).unwrap_or(b.len()); String::from_utf8_lossy(&b[..n]).into_owned() }
fn u32_at(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn u64_at(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

impl Meta {
    /// Fill the record header (`h` = REC_HDR bytes) with the magic, this meta and `checksum`.
    pub fn write(&self, h: &mut [u8], checksum: u64) {
        h[..REC_HDR].fill(0);
        h[0..8].copy_from_slice(REC_MAGIC);
        h[8..16].copy_from_slice(&self.seq.to_le_bytes());
        h[16..24].copy_from_slice(&self.ts.to_le_bytes());
        h[24..32].copy_from_slice(&checksum.to_le_bytes());
        h[32] = self.mtp as u8;
        let mut o = 36;
        for x in [self.prompt_rows, self.answer_rows, self.n_prompt, self.n_reused, self.n_feed, self.n_new, self.spec_passes, self.spec_accepted, self.spec_rejects] {
            h[o..o + 4].copy_from_slice(&x.to_le_bytes());
            o += 4;
        }
        put_str(&mut h[80..80 + NAME_MAX], &self.label);
        put_str(&mut h[104..104 + NAME_MAX], &self.active);
        if let Some(t) = &self.text {
            put_str(&mut h[TEXT_OFF..REC_HDR], t);
            let n = h[TEXT_OFF..REC_HDR].iter().position(|&x| x == 0).unwrap_or(TEXT_MAX);
            h[128..130].copy_from_slice(&(n as u16).to_le_bytes());
            h[33] = 1;
        }
    }
    /// (meta, checksum) of a record header; None = no record (empty slot / bad magic).
    pub fn read(h: &[u8]) -> Option<(Meta, u64)> {
        if h.len() < REC_HDR || &h[0..8] != REC_MAGIC { return None; }
        let seq = u64_at(h, 8);
        if seq == 0 { return None; }
        let r = |i: usize| u32_at(h, 36 + 4 * i);
        let text = (h[33] == 1).then(|| {
            let n = (u16::from_le_bytes([h[128], h[129]]) as usize).min(TEXT_MAX);
            String::from_utf8_lossy(&h[TEXT_OFF..TEXT_OFF + n]).into_owned()
        });
        Some((Meta { seq, ts: f64::from_le_bytes(h[16..24].try_into().unwrap()), mtp: h[32] != 0, prompt_rows: r(0), answer_rows: r(1), n_prompt: r(2),
                     n_reused: r(3), n_feed: r(4), n_new: r(5), spec_passes: r(6), spec_accepted: r(7), spec_rejects: r(8),
                     label: get_str(&h[80..80 + NAME_MAX]), active: get_str(&h[104..104 + NAME_MAX]), text }, u64_at(h, 24)))
    }
}

/// FNV-1a over u64 words (payload checksum; `b.len()` % 8 == 0).
pub fn checksum(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in b.chunks_exact(8) { h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x0000_0100_0000_01b3); }
    h
}

/// Bucket and sign of neuron j of layer l in the count sketch (splitmix64 of (l, j, seed)).
#[inline]
pub fn sketch_hash(l: usize, j: usize, d: usize) -> (usize, f32) {
    let mut z = ((l as u64) << 32 | j as u64) ^ SEED;
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z % d as u64) as usize, if z >> 63 == 0 { 1.0 } else { -1.0 })
}

/// u8 log-code encoder / decoder (no allocation per call).
pub struct Codec { /// log2 of the mantissa midpoint per top-8 mantissa bits
    lg: [f32; 256], /// code -> value / layer max
    pub deq: [f32; 256] }
impl Default for Codec { fn default() -> Self { Codec::new() } }
impl Codec {
    pub fn new() -> Codec {
        let mut lg = [0f32; 256];
        for (i, x) in lg.iter_mut().enumerate() { *x = (1.0 + (i as f64 + 0.5) / 256.0).log2() as f32; }
        let mut deq = [0f32; 256];
        for (c, x) in deq.iter_mut().enumerate().skip(1) { *x = (-((255 - c) as f64 + 0.5) / STEPS as f64).exp2() as f32; }
        Codec { lg, deq }
    }
    /// fast log2 (error < 0.003) of a positive normal f32
    #[inline]
    fn log2(&self, v: f32) -> f32 { let b = v.to_bits(); ((b >> 23) & 0xff) as f32 - 127.0 + self.lg[((b >> 15) & 0xff) as usize] }
    /// One layer: codes of x (>= 0) relative to its max; returns (max, sum).
    pub fn encode(&self, x: &[f32], out: &mut [u8]) -> (f32, f32) {
        let (mut mx, mut sum) = (0f32, 0f64);
        for &v in x { if v > mx { mx = v; } if v > 0.0 { sum += v as f64; } }
        if !(mx > 0.0) || !mx.is_finite() { out.fill(0); return (0.0, 0.0); }
        let inv = 1.0 / mx;
        for (o, &v) in out.iter_mut().zip(x) {
            let r = v * inv;
            *o = if r > 0.0 && r.is_normal() {
                let e = (-self.log2(r) * STEPS).max(0.0);
                if e >= 255.0 { 0 } else { 255 - e as u8 }
            } else { 0 };
        }
        (mx, sum as f32)
    }
    pub fn decode(&self, codes: &[u8], mx: f32, out: &mut [f32]) { for (o, &c) in out.iter_mut().zip(codes) { *o = self.deq[c as usize] * mx; } }
}

/// Build one slot image from a turn's raw sums. `prompt` / `answer`: Σ|a| [layers][inter] (no down norms); `down`:
/// the down column norms of layer l into its buffer. Scratch: one inter row + one sketch row (owned, reused).
pub struct Encoder { pub d: Dims, codec: Codec, row: Vec<f32>, dn: Vec<f32>, sk: Vec<f32> }
impl Encoder {
    pub fn new(d: Dims) -> Encoder { Encoder { d, codec: Codec::new(), row: vec![0.0; d.inter], dn: vec![0.0; d.inter], sk: vec![0.0; d.sketch] } }
    /// Heap bytes held (scratch).
    pub fn heap_bytes(&self) -> usize { 4 * (2 * self.d.inter + self.d.sketch) }
    /// Fill `slot` (slot_bytes) payload + header from the sums; meta.seq must be set by the caller (the ring).
    pub fn encode(&mut self, meta: &Meta, prompt: &[f32], answer: &[f32], down: &mut dyn FnMut(usize, &mut [f32]), slot: &mut [u8]) {
        let d = self.d;
        let (li, ni) = (d.layers, d.inter);
        slot[REC_HDR..].fill(0);
        for l in 0..li {
            down(l, &mut self.dn);
            for part in 0..2 {
                let src = if part == 0 { prompt } else { answer };
                for (o, (&x, &w)) in self.row.iter_mut().zip(src[l * ni..(l + 1) * ni].iter().zip(&self.dn)) { *o = (x * w).max(0.0); }
                let off = if part == 0 { d.o_prompt() } else { d.o_answer() } + l * ni;
                let (mx, sum) = self.codec.encode(&self.row, &mut slot[off..off + ni]);
                let so = d.o_scale();
                slot[so + 4 * (part * li + l)..][..4].copy_from_slice(&mx.to_le_bytes());
                slot[so + 4 * ((2 + part) * li + l)..][..4].copy_from_slice(&sum.to_le_bytes());
                if part == 0 {
                    // count sketch of the layer-normalised prompt (sum 1, times inter: values ~1, f16-safe)
                    self.sk.fill(0.0);
                    if sum > 0.0 {
                        let s = ni as f32 / sum;
                        for (j, &v) in self.row.iter().enumerate() { if v > 0.0 { let (b, sg) = sketch_hash(l, j, d.sketch); self.sk[b] += sg * v * s; } }
                    }
                    let o = d.o_sketch() + 2 * l * d.sketch;
                    for (k, &v) in self.sk.iter().enumerate() { slot[o + 2 * k..o + 2 * k + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes()); }
                }
            }
        }
        let c = checksum(&slot[REC_HDR..]);
        meta.write(&mut slot[..REC_HDR], c);
    }
}

fn header(d: &Dims, slots: usize, created: f64) -> [u8; FILE_HDR as usize] {
    let mut h = [0u8; FILE_HDR as usize];
    h[0..8].copy_from_slice(MAGIC);
    h[8..12].copy_from_slice(&VERSION.to_le_bytes());
    h[12..16].copy_from_slice(&(d.layers as u32).to_le_bytes());
    h[16..20].copy_from_slice(&(d.inter as u32).to_le_bytes());
    h[20..24].copy_from_slice(&(d.sketch as u32).to_le_bytes());
    h[24..32].copy_from_slice(&(d.slot_bytes() as u64).to_le_bytes());
    h[32..40].copy_from_slice(&(slots as u64).to_le_bytes());
    h[40..48].copy_from_slice(&created.to_le_bytes());
    h[48..56].copy_from_slice(&SEED.to_le_bytes());
    h
}
/// (dims, slots, created) of a file header.
fn parse_header(h: &[u8]) -> Result<(Dims, usize, f64)> {
    if &h[0..8] != MAGIC { return Err(ie("not a streamdense library file (magic)")); }
    if u32_at(h, 8) != VERSION { return Err(ie(format!("library version {} (this build reads {})", u32_at(h, 8), VERSION))); }
    let d = Dims { layers: u32_at(h, 12) as usize, inter: u32_at(h, 16) as usize, sketch: u32_at(h, 20) as usize };
    if u64_at(h, 24) != d.slot_bytes() as u64 { return Err(ie("library slot size does not match its dims")); }
    if u64_at(h, 48) != SEED { return Err(ie("library sketch seed differs")); }
    Ok((d, u64_at(h, 32) as usize, f64::from_le_bytes(h[40..48].try_into().unwrap())))
}

/// seq of slot i (0 = empty / not in the file).
fn slot_seq(f: &File, d: &Dims, i: usize) -> u64 {
    let mut b = [0u8; 16];
    let off = FILE_HDR + (i * d.slot_bytes()) as u64;
    if f.read_exact_at(&mut b, off).is_err() || &b[0..8] != REC_MAGIC { return 0; }
    u64_at(&b, 8)
}

/// The writer: a fixed-capacity ring of slots (oldest record replaced first).
pub struct Ring { pub path: PathBuf, f: File, pub d: Dims, pub slots: usize, seqs: Vec<u64>, next_seq: u64 }

impl Ring {
    pub fn path_for(dir: &Path, d: &Dims) -> PathBuf { dir.join(format!("library_{}.sdl", d.tag())) }
    /// Open or create the ring for `d` in `dir` with room for `cap` bytes. A file with another slot count (the cap
    /// changed) is resized in place: growing adds empty slots; shrinking keeps the newest records (moved down one slot
    /// at a time through `slot_bytes` of scratch) and truncates.
    pub fn open(dir: &Path, d: Dims, cap: u64, now: f64) -> Result<Ring> {
        let slots = d.slots_for(cap);
        if slots == 0 { return Err(Error::new(ErrorKind::InvalidInput, format!("library cap {} B is below one record ({} B + {} B header)", cap, d.slot_bytes(), FILE_HDR))); }
        std::fs::create_dir_all(dir)?;
        let path = Ring::path_for(dir, &d);
        let f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
        let len = f.metadata()?.len();
        let (old, created) = if len >= FILE_HDR {
            let mut h = [0u8; FILE_HDR as usize];
            f.read_exact_at(&mut h, 0)?;
            let (fd, s, c) = parse_header(&h)?;
            if fd != d { return Err(ie(format!("{}: dims {:?}, want {:?}", path.display(), fd, d))); }
            (s, c)
        } else { (0, now) };
        let mut r = Ring { path, f, d, slots, seqs: vec![], next_seq: 1 };
        r.seqs = (0..old.max(slots)).map(|i| slot_seq(&r.f, &d, i)).collect();
        if old > slots { r.shrink()?; }
        r.seqs.truncate(slots);
        r.seqs.resize(slots, 0);
        r.f.write_all_at(&header(&d, slots, created), 0)?;
        r.next_seq = r.seqs.iter().max().copied().unwrap_or(0) + 1;
        Ok(r)
    }
    fn shrink(&mut self) -> Result<()> {
        let (n, sb) = (self.slots, self.d.slot_bytes());
        let mut by: Vec<(u64, usize)> = self.seqs.iter().enumerate().filter(|x| *x.1 > 0).map(|(i, &s)| (s, i)).collect();
        by.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        by.truncate(n);
        let keep: std::collections::HashSet<usize> = by.iter().map(|x| x.1).collect();
        let mut free: Vec<usize> = (0..n).filter(|i| !keep.contains(i)).collect();
        let mut buf = vec![0u8; sb];
        for &(s, i) in &by {
            if i < n { continue; }
            let dst = free.pop().ok_or_else(|| ie("library shrink: no free slot"))?;
            self.f.read_exact_at(&mut buf, FILE_HDR + (i * sb) as u64)?;
            self.f.write_all_at(&buf, FILE_HDR + (dst * sb) as u64)?;
            self.seqs[dst] = s;
        }
        for &i in &free { if self.seqs[i] > 0 { self.f.write_all_at(&[0u8; 16], FILE_HDR + (i * sb) as u64)?; self.seqs[i] = 0; } }
        let used = (0..n).rev().find(|&i| self.seqs[i] > 0).map_or(0, |i| i + 1);
        self.f.set_len(FILE_HDR + (used * sb) as u64)?;
        Ok(())
    }
    /// Records stored.
    pub fn count(&self) -> usize { self.seqs.iter().filter(|&&s| s > 0).count() }
    pub fn disk_bytes(&self) -> u64 { self.f.metadata().map(|m| m.len()).unwrap_or(0) }
    pub fn cap_bytes(&self) -> u64 { FILE_HDR + (self.slots * self.d.slot_bytes()) as u64 }
    /// The slot the next record goes to and whether it replaces a record (false = an empty slot, the file grows).
    pub fn next_slot(&self) -> (usize, bool) {
        match self.seqs.iter().position(|&s| s == 0) {
            Some(i) => (i, false),
            None => (self.seqs.iter().enumerate().min_by_key(|x| *x.1).map_or(0, |x| x.0), true),
        }
    }
    /// The seq the next record gets (the caller writes it into the meta before encoding).
    pub fn next_seq(&self) -> u64 { self.next_seq }
    /// Write an encoded slot image (meta.seq == next_seq). Returns (slot, evicted a record).
    pub fn append(&mut self, slot: &[u8]) -> Result<(usize, bool)> {
        let (m, _) = Meta::read(slot).ok_or_else(|| ie("append: slot image has no record header"))?;
        if m.seq != self.next_seq { return Err(ie(format!("append: seq {} != next {}", m.seq, self.next_seq))); }
        if slot.len() != self.d.slot_bytes() { return Err(ie("append: slot size")); }
        let (i, ev) = self.next_slot();
        self.f.write_all_at(slot, FILE_HDR + (i * self.d.slot_bytes()) as u64)?;
        self.seqs[i] = m.seq;
        self.next_seq += 1;
        Ok((i, ev))
    }
}

/// The reader (offline tools): fixed-size reads at computed offsets, no full-file load.
pub struct Lib { pub path: PathBuf, f: File, pub d: Dims, pub slots: usize, pub created: f64, codec: Codec }

impl Lib {
    /// `p`: a library file, or a directory holding exactly one `library_*.sdl`.
    pub fn open(p: &Path) -> Result<Lib> {
        let path = if p.is_dir() {
            let v: Vec<PathBuf> = std::fs::read_dir(p)?.filter_map(|e| e.ok()).map(|e| e.path())
                .filter(|x| x.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("library_") && n.ends_with(".sdl"))).collect();
            match v.len() {
                1 => v[0].clone(),
                0 => return Err(Error::new(ErrorKind::NotFound, format!("{}: no library_*.sdl", p.display()))),
                _ => return Err(ie(format!("{}: {} library files; name one", p.display(), v.len()))),
            }
        } else { p.to_path_buf() };
        let f = File::open(&path)?;
        let mut h = [0u8; FILE_HDR as usize];
        f.read_exact_at(&mut h, 0)?;
        let (d, slots, created) = parse_header(&h)?;
        Ok(Lib { path, f, d, slots, created, codec: Codec::new() })
    }
    pub fn disk_bytes(&self) -> u64 { self.f.metadata().map(|m| m.len()).unwrap_or(0) }
    fn off(&self, i: usize) -> u64 { FILE_HDR + (i * self.d.slot_bytes()) as u64 }
    /// (seq, slot) of every record with a header, oldest first (headers only; `verify` checks the payload).
    pub fn records(&self) -> Vec<(u64, usize)> {
        let mut v: Vec<(u64, usize)> = (0..self.slots).map(|i| (slot_seq(&self.f, &self.d, i), i)).filter(|x| x.0 > 0).collect();
        v.sort_unstable();
        v
    }
    /// Meta + stored checksum of slot i.
    pub fn meta(&self, i: usize) -> Result<(Meta, u64)> {
        let mut h = vec![0u8; REC_HDR];
        self.f.read_exact_at(&mut h, self.off(i))?;
        Meta::read(&h).ok_or_else(|| ie(format!("slot {}: no record", i)))
    }
    /// Payload checksum ok (streams the payload through `buf`, 64 KB chunks).
    pub fn verify(&self, i: usize) -> bool {
        let Ok((_, want)) = self.meta(i) else { return false };
        let mut buf = vec![0u8; 64 << 10];
        let (mut o, end) = (REC_HDR, self.d.slot_bytes());
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        while o < end {
            let n = buf.len().min(end - o);
            if self.f.read_exact_at(&mut buf[..n], self.off(i) + o as u64).is_err() { return false; }
            for c in buf[..n].chunks_exact(8) { h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x0000_0100_0000_01b3); }
            o += n;
        }
        h == want
    }
    /// f32 [4][layers]: prompt max, answer max, prompt sum, answer sum (|a|·dn units).
    pub fn scales(&self, i: usize) -> Result<Vec<f32>> {
        let mut b = vec![0u8; 16 * self.d.layers];
        self.f.read_exact_at(&mut b, self.off(i) + self.d.o_scale() as u64)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    /// Layer l of a record's prompt / answer, dequantised (|a|·dn units; `mx` = that layer's max from `scales`).
    pub fn layer(&self, i: usize, part: Part, l: usize, mx: f32, codes: &mut [u8], out: &mut [f32]) -> Result<()> {
        let o = match part { Part::Prompt => self.d.o_prompt(), Part::Answer => self.d.o_answer() } + l * self.d.inter;
        self.f.read_exact_at(&mut codes[..self.d.inter], self.off(i) + o as u64)?;
        self.codec.decode(&codes[..self.d.inter], mx, out);
        Ok(())
    }
    /// The prompt sketch [layers][sketch] as f32.
    pub fn sketch(&self, i: usize, out: &mut [f32]) -> Result<()> {
        let n = self.d.layers * self.d.sketch;
        let mut b = vec![0u8; 2 * n];
        self.f.read_exact_at(&mut b, self.off(i) + self.d.o_sketch() as u64)?;
        for (o, c) in out.iter_mut().zip(b.chunks_exact(2)) { *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]])); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tmp(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_library_{}_{}", n, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }
    fn synth(d: &Dims, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32).powi(6) * 50.0 };
        let p: Vec<f32> = (0..d.n()).map(|_| r()).collect();
        let a: Vec<f32> = (0..d.n()).map(|_| r()).collect();
        (p, a)
    }
    fn rec(e: &mut Encoder, ring: &Ring, seed: u64, label: &str) -> Vec<u8> {
        let d = e.d;
        let (p, a) = synth(&d, seed);
        let m = Meta { seq: ring.next_seq(), ts: 1.0 + seed as f64, label: label.into(), active: "chat-general".into(), prompt_rows: 30, answer_rows: 12,
                       n_prompt: 40, n_reused: 10, n_feed: 30, n_new: 9, mtp: true, spec_passes: 5, spec_accepted: 4, spec_rejects: 3, ..Default::default() };
        let mut slot = vec![0u8; d.slot_bytes()];
        e.encode(&m, &p, &a, &mut |l, o| o.iter_mut().enumerate().for_each(|(j, x)| *x = 1.0 + ((l + j) % 3) as f32), &mut slot);
        slot
    }

    #[test]
    fn codec_accuracy() {
        let c = Codec::new();
        let x: Vec<f32> = (0..4000).map(|i| (1.0005f32).powi(i) * 1e-3).collect();
        let mut q = vec![0u8; x.len()];
        let (mx, _) = c.encode(&x, &mut q);
        let mut y = vec![0f32; x.len()];
        c.decode(&q, mx, &mut y);
        for (a, b) in x.iter().zip(&y) {
            if *a > mx * (-15.5f32).exp2() { assert!((b / a).log2().abs() <= 0.5 / STEPS + 0.01, "{} {}", a, b); }
        }
        assert_eq!(q.iter().max(), Some(&255));
        let mut z = [7u8; 3];
        assert_eq!(c.encode(&[0.0, 0.0, 0.0], &mut z), (0.0, 0.0));
        assert_eq!(z, [0, 0, 0]);
    }

    #[test]
    fn round_trip_eviction_cap() {
        let dir = tmp("rt");
        let d = Dims { layers: 3, inter: 96, sketch: 16 };
        let sb = d.slot_bytes() as u64;
        let mut e = Encoder::new(d);
        // cap for 3 records
        let mut ring = Ring::open(&dir, d, FILE_HDR + 3 * sb + 100, 5.0).unwrap();
        assert_eq!((ring.slots, ring.count()), (3, 0));
        for k in 0..5u64 {
            let s = rec(&mut e, &ring, k, if k % 2 == 0 { "code-rust" } else { "chat-general" });
            let (i, ev) = ring.append(&s).unwrap();
            assert_eq!((i, ev), ((k % 3) as usize, k >= 3));
            assert!(ring.disk_bytes() <= ring.cap_bytes());
        }
        assert_eq!(ring.count(), 3);
        assert_eq!(ring.disk_bytes(), FILE_HDR + 3 * sb);
        drop(ring);
        let lib = Lib::open(&dir).unwrap();
        let recs = lib.records();
        assert_eq!(recs.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 4, 5], "oldest two dropped");
        let (m, _) = lib.meta(recs[2].1).unwrap();
        assert_eq!((m.seq, m.label.as_str(), m.n_new, m.answer_rows, m.text.clone()), (5, "code-rust", 9, 12, None));
        assert!(recs.iter().all(|r| lib.verify(r.1)));
        // dequantised layer == |a|·dn within one code step; scales hold the exact max / sum
        let (p, a) = synth(&d, 4);
        let sc = lib.scales(recs[2].1).unwrap();
        let (mut codes, mut out) = (vec![0u8; d.inter], vec![0f32; d.inter]);
        for l in 0..d.layers {
            for (part, src, k) in [(Part::Prompt, &p, 0), (Part::Answer, &a, 1)] {
                lib.layer(recs[2].1, part, l, sc[k * d.layers + l], &mut codes, &mut out).unwrap();
                let ex: Vec<f32> = (0..d.inter).map(|j| src[l * d.inter + j] * (1.0 + ((l + j) % 3) as f32)).collect();
                let mx = ex.iter().cloned().fold(0.0, f32::max);
                assert_eq!(sc[k * d.layers + l], mx);
                assert!((sc[(2 + k) * d.layers + l] - ex.iter().sum::<f32>()).abs() < 1e-3 * mx * d.inter as f32);
                for (x, y) in ex.iter().zip(&out) { if *x > mx * 1e-4 { assert!((y / x).log2().abs() < 0.05, "{} {}", x, y); } }
            }
        }
        let mut sk = vec![0f32; d.layers * d.sketch];
        lib.sketch(recs[2].1, &mut sk).unwrap();
        assert!(sk.iter().any(|&x| x != 0.0));
        // torn payload -> checksum fails
        let f = OpenOptions::new().write(true).open(&lib.path).unwrap();
        f.write_all_at(&[1, 2, 3], lib.off(recs[0].1) + REC_HDR as u64 + 50).unwrap();
        assert!(!lib.verify(recs[0].1));
        // reopen with a bigger cap: grows, new records go to the empty slots first; smaller cap keeps the newest
        let mut ring = Ring::open(&dir, d, FILE_HDR + 5 * sb, 6.0).unwrap();
        assert_eq!((ring.slots, ring.count(), ring.next_seq()), (5, 3, 6));
        let s = rec(&mut e, &ring, 9, "x");
        assert_eq!(ring.append(&s).unwrap(), (3, false));
        drop(ring);
        let ring = Ring::open(&dir, d, FILE_HDR + 2 * sb, 7.0).unwrap();
        assert_eq!((ring.slots, ring.count()), (2, 2));
        assert!(ring.disk_bytes() <= FILE_HDR + 2 * sb);
        drop(ring);
        let lib = Lib::open(&dir).unwrap();
        assert_eq!(lib.records().iter().map(|x| x.0).collect::<Vec<_>>(), vec![5, 6]);
        assert!(lib.records().iter().all(|r| lib.verify(r.1)), "moved records stay intact");
        // too small a cap is an error; other dims are refused
        assert!(Ring::open(&tmp("small"), d, sb, 1.0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_only_when_given() {
        let mut h = vec![0u8; REC_HDR];
        let m = Meta { seq: 1, label: "code-rust".into(), ..Default::default() };
        m.write(&mut h, 7);
        assert_eq!(Meta::read(&h).unwrap(), (m.clone(), 7));
        assert!(h[TEXT_OFF..].iter().all(|&b| b == 0));
        let t = Meta { text: Some("héllo ".repeat(2000)), ..m };
        t.write(&mut h, 7);
        let back = Meta::read(&h).unwrap().0.text.unwrap();
        assert!(back.len() <= TEXT_MAX && back.starts_with("héllo"));
    }

    #[test]
    fn slot_size_27b() {
        let d = Dims::new(64, 17408);
        assert_eq!(d.slot_bytes(), 2_367_488); // 2.26 MiB: 2 x 1,114,112 codes + 131,072 sketch + 5,120 header / scales
        assert_eq!(d.slots_for(1 << 30), 453);
    }
}
