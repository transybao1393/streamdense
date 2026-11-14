//! Forheader step 3 (2026-11-02): the REAL 12 GB loader for the per-prompt neuron set "A". OFF by default
//! (`QwenOptions.ffn_compact = None`: no region, no shader, no dispatch, today's code path).
//!
//! With a `CompactSet` (per main layer: the kept FFN neuron indices, sorted, a multiple of 8 each) the engine holds
//! ONLY those neurons of every main layer's FFN, read from the unchanged MLX model file:
//! - gate / up: the selected rows (W u32 [kc][hidden/8], S / B bf16 [kc][hidden/64]); the normal swiglu kernels run on
//!   them unchanged and write f [k][kc] (per row bit-identical to the full matrix's row: rows are independent);
//! - down: the selected COLUMNS repacked per neuron, W u32 [hidden][kc/8] (nibble i of row r = original column
//!   keep[i]), with the ORIGINAL scales / biases kept whole (bf16 [hidden][inter/64]) and `gid` u16 [kc] (each kept
//!   neuron's original 64-group) + `gstart` u32 [inter/64 + 1] (first compact column of each group). Exact, no
//!   requantizing: y_r = sum_g s_rg * (sum_{j in g} f_j q_rj) + b_rg * (sum_{j in g} f_j) over the kept j only.
//!   `ffn_compact.metal`: `cd_fsum` (per-group f sums) then `cd_down` (the residual add h += y, MODE 2 rounding).
//!   Versus the full kernels with the dropped neurons zeroed (the ffn_select hook, mode Static) only the fp32
//!   summation order differs.
//! Everything else (attention / DeltaNet, norms, lm_head, KV, embeddings from SSD) is as today. The MTP layer is not
//! compacted; MTP speculative decode works since 2026-11-02 (verify passes: gate / up through the K-row Mma swiglu on
//! the kept rows, down through `cd_downk_r4_k{K}`, which reads W once for all K rows and is per row bit-identical to
//! K = 1, so compact + MTP == plain compact token for token). Refused with compact on: the recorder, the read mask,
//! the ffn_select hook and fast prefill (they index the FFN by the full `inter`). `QwenEngine::ffn_compact_load` re-reads another set of the same
//! per-layer sizes in place (the per-prompt load at Enter). Load reads every FFN tensor in 8 MB chunks per thread
//! (bounded: threads x 8 MB) and gathers the kept rows / columns.
use std::fmt;
use std::sync::Arc;

use streamdense_core::arena::Region;
use streamdense_core::io::NoCacheFile;
use streamdense_metal::Enc;

use crate::qwen_engine::QwenEngine;
use crate::qwen_layers::PassIo;
use crate::weights::FfnW;

/// Per main layer: the kept neuron indices (sorted, unique, < inter, a multiple of 8 and >= 8 per layer).
#[derive(Clone, PartialEq, Eq)]
pub struct CompactSet {
    pub inter: usize,
    pub keep: Vec<Vec<u32>>,
}
impl fmt::Debug for CompactSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n: usize = self.keep.iter().map(|k| k.len()).sum();
        write!(f, "CompactSet {{ layers: {}, inter: {}, kept: {} ({:.4}) }}", self.keep.len(), self.inter, n,
               n as f64 / (self.keep.len().max(1) * self.inter) as f64)
    }
}

const MAGIC: &[u8; 8] = b"SDFCMP1\0";

impl CompactSet {
    /// All neurons (keep share 1.0).
    pub fn all(layers: usize, inter: usize) -> CompactSet { CompactSet { inter, keep: vec![(0..inter as u32).collect(); layers] } }
    /// From a [layers][inter] mask (non-zero = keep).
    pub fn from_mask(mask: &[u8], layers: usize, inter: usize) -> Result<CompactSet, String> {
        if mask.len() != layers * inter { return Err(format!("ffn_compact: mask has {} entries, want {}", mask.len(), layers * inter)); }
        let keep = (0..layers).map(|l| (0..inter as u32).filter(|&j| mask[l * inter + j as usize] != 0).collect()).collect();
        let s = CompactSet { inter, keep };
        s.validate()?;
        Ok(s)
    }
    pub fn validate(&self) -> Result<(), String> {
        for (l, k) in self.keep.iter().enumerate() {
            if k.len() < 8 || k.len() % 8 != 0 { return Err(format!("ffn_compact: layer {} keeps {} neurons (want a multiple of 8, >= 8)", l, k.len())); }
            if k.windows(2).any(|w| w[0] >= w[1]) || *k.last().unwrap() as usize >= self.inter {
                return Err(format!("ffn_compact: layer {} indices not sorted / unique / < {}", l, self.inter));
            }
        }
        Ok(())
    }
    pub fn counts(&self) -> Vec<usize> { self.keep.iter().map(|k| k.len()).collect() }
    /// Binary file: magic, layers u32, inter u32, per layer count u32 + indices u32 (little endian).
    pub fn write(&self, path: &std::path::Path) -> Result<(), String> {
        let mut b = MAGIC.to_vec();
        b.extend_from_slice(&(self.keep.len() as u32).to_le_bytes());
        b.extend_from_slice(&(self.inter as u32).to_le_bytes());
        for k in &self.keep {
            b.extend_from_slice(&(k.len() as u32).to_le_bytes());
            for x in k { b.extend_from_slice(&x.to_le_bytes()); }
        }
        std::fs::write(path, b).map_err(|e| format!("{}: {}", path.display(), e))
    }
    pub fn read(path: &std::path::Path) -> Result<CompactSet, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        if b.len() < 16 || &b[..8] != MAGIC { return Err(format!("{}: not a compact set file", path.display())); }
        let u = |i: usize| -> Result<u32, String> {
            b.get(i..i + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).ok_or_else(|| format!("{}: truncated", path.display()))
        };
        let (layers, inter) = (u(8)? as usize, u(12)? as usize);
        let mut off = 16;
        let mut keep = Vec::with_capacity(layers);
        for _ in 0..layers {
            let n = u(off)? as usize;
            off += 4;
            let mut k = Vec::with_capacity(n);
            for i in 0..n { k.push(u(off + 4 * i)?); }
            off += 4 * n;
            keep.push(k);
        }
        let s = CompactSet { inter, keep };
        s.validate()?;
        Ok(s)
    }
}

/// Compact down descriptor of one layer (FfnW.cd): `kc` kept neurons, `groups` = inter / 64, `gid` u16 [kc],
/// `gstart` u32 [groups + 1], `wmeta` u32 [kc / 8] per packed word: g0 | g7 << 12 | p << 24 | complex << 28 (groups of
/// its first / last column, columns in g0, 3+ groups). The down matrix itself is `FfnW.down` with rows = hidden, cols = kc (W row stride
/// kc / 8 words) and S / B of the ORIGINAL shape [hidden][groups].
#[derive(Clone, Copy, Debug, Default)]
pub struct CDown {
    pub kc: usize,
    pub groups: usize,
    pub gid: Region,
    pub gstart: Region,
    pub wmeta: Region,
    /// down family: true = MMA `cd_dm` (k >= 2) + its bit-identical twin `cd_dt` (k = 1), set when the engine has MTP
    /// (verify passes must equal K = 1 decode bit for bit); false = `cd_downk_r4_k{k}` (k = 1 is cd_down_r4's
    /// arithmetic, the fastest decode; compute-bound at k >= 2). `SD_CD_DOWN` overrides (`down_family`).
    pub mma: bool,
}

/// Source of one tensor in the model files.
#[derive(Clone, Copy, Debug)]
pub struct CSrc { pub file: usize, pub off: u64 }

/// One main layer's gather job (sources in the model file, destinations in the arena).
#[derive(Clone, Debug)]
pub struct CJob {
    pub layer: usize,
    /// gate, up: [W, S, B] each
    pub gate: [CSrc; 3],
    pub up: [CSrc; 3],
    pub down_w: CSrc,
    pub ffn: FfnW,
}

/// Engine-side state of the compact FFN: the gather jobs (for reloads) and the set now loaded.
pub struct CompactState {
    pub files: Vec<String>,
    pub jobs: Vec<CJob>,
    pub set: std::cell::RefCell<Arc<CompactSet>>,
    pub hidden: usize,
    pub inter: usize,
    pub threads: usize,
}

/// Fill the arena destinations of `jobs` for `set` (gid / gstart computed, gate / up rows and down columns gathered
/// from the model files). `base` = arena base pointer. Returns the bytes read.
pub fn gather(files: &[String], jobs: &[CJob], set: &CompactSet, hidden: usize, inter: usize, base: *mut u8, threads: usize) -> Result<u64, String> {
    const BOUNCE: usize = 8 << 20;
    let fds: Vec<NoCacheFile> = files.iter().map(|f| NoCacheFile::open(f)).collect::<Result<_, _>>()?;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let err = std::sync::Mutex::new(None::<String>);
    let total = std::sync::atomic::AtomicU64::new(0);
    let base = base as usize;
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let mut bounce = vec![0u8; BOUNCE];
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= jobs.len() { break; }
                    let j = &jobs[i];
                    match gather_one(&fds, j, &set.keep[j.layer], hidden, inter, base as *mut u8, &mut bounce) {
                        Ok(n) => { total.fetch_add(n, std::sync::atomic::Ordering::Relaxed); }
                        Err(e) => { *err.lock().unwrap() = Some(e); break; }
                    }
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() { return Err(e); }
    Ok(total.into_inner())
}

/// Rows `keep` of a row-major tensor [inter][rb bytes] at `src` -> dst rows 0..keep.len().
fn gather_rows(fd: &NoCacheFile, src: u64, rb: usize, nrows: usize, keep: &[u32], dst: *mut u8, bounce: &mut [u8]) -> Result<u64, String> {
    let per = (bounce.len() / rb).max(1);
    let (mut r0, mut ki, mut read) = (0usize, 0usize, 0u64);
    while r0 < nrows && ki < keep.len() {
        let r1 = (r0 + per).min(nrows);
        fd.read_at(bounce.as_mut_ptr(), (r1 - r0) * rb, src + (r0 * rb) as u64)?;
        read += ((r1 - r0) * rb) as u64;
        while ki < keep.len() && (keep[ki] as usize) < r1 {
            let r = keep[ki] as usize - r0;
            unsafe { std::ptr::copy_nonoverlapping(bounce.as_ptr().add(r * rb), dst.add(ki * rb), rb); }
            ki += 1;
        }
        r0 = r1;
    }
    Ok(read)
}

fn gather_one(fds: &[NoCacheFile], j: &CJob, keep: &[u32], hidden: usize, inter: usize, base: *mut u8, bounce: &mut [u8]) -> Result<u64, String> {
    let cd = j.ffn.cd.ok_or("ffn_compact: job without a compact down")?;
    if keep.len() != cd.kc { return Err(format!("ffn_compact: layer {} set has {} neurons, arena holds {}", j.layer, keep.len(), cd.kc)); }
    let at = |r: Region| unsafe { base.add(r.off) };
    let mut read = 0u64;
    // gate / up rows: W [inter][hidden/8] u32, S / B [inter][hidden/64] bf16
    for (src, q) in [(&j.gate, &j.ffn.gate), (&j.up, &j.ffn.up)] {
        let rbs = [hidden / 2, hidden / 64 * 2, hidden / 64 * 2];
        let dsts = [q.w, q.s, q.b];
        for t in 0..3 {
            read += gather_rows(&fds[src[t].file], src[t].off, rbs[t], inter, keep, at(dsts[t]), bounce)?;
        }
    }
    // down: W [hidden][inter/8] u32 -> [hidden][kc/8], nibble i = column keep[i]
    let (srow, drow) = (inter / 2, cd.kc / 2);
    let per = (bounce.len() / srow).max(1);
    let dw = at(j.ffn.down.w);
    let f = &fds[j.down_w.file];
    let mut r0 = 0;
    while r0 < hidden {
        let r1 = (r0 + per).min(hidden);
        f.read_at(bounce.as_mut_ptr(), (r1 - r0) * srow, j.down_w.off + (r0 * srow) as u64)?;
        read += ((r1 - r0) * srow) as u64;
        for r in r0..r1 {
            let s = &bounce[(r - r0) * srow..(r - r0 + 1) * srow];
            let d = unsafe { std::slice::from_raw_parts_mut(dw.add(r * drow), drow) };
            for (i, pair) in keep.chunks_exact(2).enumerate() {
                let nib = |c: u32| (s[c as usize / 2] >> (4 * (c % 2))) & 0xf;
                d[i] = nib(pair[0]) | (nib(pair[1]) << 4);
            }
        }
        r0 = r1;
    }
    fill_meta(&cd, keep, inter, base);
    Ok(read)
}

/// gid / gstart / wmeta of a compact layer for `keep` (arena destinations of `cd`).
pub(crate) fn fill_meta(cd: &CDown, keep: &[u32], inter: usize, base: *mut u8) {
    let at = |r: Region| unsafe { base.add(r.off) };
    let g = inter / 64;
    let gid = unsafe { std::slice::from_raw_parts_mut(at(cd.gid) as *mut u16, cd.kc) };
    let gs = unsafe { std::slice::from_raw_parts_mut(at(cd.gstart) as *mut u32, g + 1) };
    let mut c = 0usize;
    for gi in 0..g {
        gs[gi] = c as u32;
        while c < keep.len() && (keep[c] as usize) / 64 == gi { gid[c] = gi as u16; c += 1; }
    }
    gs[g] = c as u32;
    let wm = unsafe { std::slice::from_raw_parts_mut(at(cd.wmeta) as *mut u32, cd.kc / 8) };
    for (w, m) in wm.iter_mut().enumerate() {
        let gw = &gid[w * 8..w * 8 + 8];
        let (g0, g7) = (gw[0] as u32, gw[7] as u32);
        let p = gw.iter().filter(|&&x| x as u32 == g0).count() as u32;
        let complex = gw.iter().any(|&x| x as u32 != g0 && x as u32 != g7);
        if g < 4096 { *m = g0 | (g7 << 12) | (p << 24) | ((complex as u32) << 28); } else { *m = 1 << 28; }
    }
}

/// Fill compact layer `j` for `keep` from a FULL copy of the layer already in RAM (ffn_stream.rs: the streamed-prefill
/// slot): `gate` / `up` = [W, S, B] arena offsets of the full [inter][..] tensors, `down_w` = the full down W
/// [hidden][inter/8]. Byte for byte what `gather` writes from the file (same rows / nibbles, same meta). The down
/// repack runs on `threads` threads (row blocks).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_from_full(j: &CJob, keep: &[u32], hidden: usize, inter: usize, base: *mut u8, gate: [usize; 3], up: [usize; 3],
                             down_w: usize, threads: usize) -> Result<(), String> {
    let cd = j.ffn.cd.ok_or("ffn_compact: job without a compact down")?;
    if keep.len() != cd.kc { return Err(format!("ffn_compact: layer {} set has {} neurons, arena holds {}", j.layer, keep.len(), cd.kc)); }
    let at = |off: usize| unsafe { base.add(off) };
    for (src, q) in [(gate, &j.ffn.gate), (up, &j.ffn.up)] {
        let rbs = [hidden / 2, hidden / 64 * 2, hidden / 64 * 2];
        let dsts = [q.w, q.s, q.b];
        for t in 0..3 {
            let (s, d, rb) = (at(src[t]), at(dsts[t].off), rbs[t]);
            for (i, &r) in keep.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(s.add(r as usize * rb), d.add(i * rb), rb); }
            }
        }
    }
    let (srow, drow) = (inter / 2, cd.kc / 2);
    let (sw, dw) = (at(down_w) as usize, at(j.ffn.down.w.off) as usize);
    let per = hidden.div_ceil(threads.max(1));
    std::thread::scope(|sc| {
        for r0 in (0..hidden).step_by(per.max(1)) {
            let r1 = (r0 + per).min(hidden);
            sc.spawn(move || {
                for r in r0..r1 {
                    let s = unsafe { std::slice::from_raw_parts((sw as *const u8).add(r * srow), srow) };
                    let d = unsafe { std::slice::from_raw_parts_mut((dw as *mut u8).add(r * drow), drow) };
                    for (i, pair) in keep.chunks_exact(2).enumerate() {
                        let nib = |c: u32| (s[c as usize / 2] >> (4 * (c % 2))) & 0xf;
                        d[i] = nib(pair[0]) | (nib(pair[1]) << 4);
                    }
                }
            });
        }
    });
    fill_meta(&cd, keep, inter, base);
    Ok(())
}

/// FFN down of a compact layer: h[t] += down_kept(f[t]) for the pass rows (MODE 2 residual add, as `gemv_m`).
pub(crate) fn encode_down(e: &Enc, ff: &FfnW, cd: &CDown, io: &PassIo) {
    e.pso("cd_fsum").b(0, io.f).b(1, cd.gstart.off).b(2, io.cfs).u(3, cd.kc as u32).u(4, cd.groups as u32)
        .groups((cd.groups, io.k), 32);
    let rows = ff.down.rows;
    let fam = down_family().unwrap_or(if cd.mma { DownFam::Mma } else { DownFam::K });
    if fam == DownFam::Mma {
        let (nb, sk) = dm_shape();
        let enc = if io.k == 1 { e.pso(dt_name(sk)) } else { e.pso(dm_name(nb, sk)) };
        enc.b(0, ff.down.w.off).b(1, ff.down.s.off).b(2, ff.down.b.off).b(3, io.f).b(4, cd.gid.off).b(5, io.cfs)
            .b(6, io.h).u(7, rows as u32).u(8, cd.kc as u32).u(9, cd.groups as u32).u(10, io.k as u32).b(11, cd.wmeta.off)
            .b(12, cd.gstart.off);
        if io.k == 1 { enc.groups((rows.div_ceil(32), 1), 32 * sk); } else { enc.groups((rows.div_ceil(8 * nb), 1), 32 * sk); }
        return;
    }
    let (name, nr) = if fam == DownFam::Old { down_kernel() } else { (downk_kernel(io.k), 4) };
    e.pso(name).b(0, ff.down.w.off).b(1, ff.down.s.off).b(2, ff.down.b.off).b(3, io.f).b(4, cd.gid.off).b(5, io.cfs)
        .b(6, io.h).u(7, rows as u32).u(8, cd.kc as u32).u(9, cd.groups as u32).u(10, io.k as u32).b(11, cd.wmeta.off)
        .groups((rows.div_ceil(2 * nr), 1), 64);
}

pub fn pipelines() -> Vec<&'static str> {
    let mut v = vec!["cd_fsum", "cd_down_r4", "cd_down_r8", "cd_downk_r4_k1", "cd_downk_r4_k2", "cd_downk_r4_k3", "cd_downk_r4_k4",
                     "cd_downk_r4_k5", "cd_downk_r4_k6", "cd_downk_r4_k7", "cd_downk_r4_k8"];
    v.extend_from_slice(&DM_NAMES.map(|x| x.2));
    v.extend_from_slice(&DT_NAMES.map(|x| x.1));
    v
}
/// Compact down family (`SD_CD_DOWN` overrides `CDown.mma`): `mma` (`cd_dm_n{NB}_s{SK}` for k >= 2, its bit-identical scalar twin
/// `cd_dt_s{SK}` for k = 1), `k` (`cd_downk_r4_k{k}`: scalar, W once per pass but compute-bound at k >= 2), `old`
/// (`down_kernel()`: per-row loop). Every family is K-invariant per row; they differ from each other in fp32 order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DownFam { Mma, K, Old }
pub fn down_family() -> Option<DownFam> {
    static F: std::sync::OnceLock<Option<DownFam>> = std::sync::OnceLock::new();
    *F.get_or_init(|| match std::env::var("SD_CD_DOWN").ok().as_deref() {
        Some("k") => Some(DownFam::K),
        Some("old") => Some(DownFam::Old),
        Some("mma") => Some(DownFam::Mma),
        _ => None,
    })
}
pub const DM_NAMES: [(usize, usize, &str); 11] = [(4, 1, "cd_dm_n4_s1"), (8, 1, "cd_dm_n8_s1"), (8, 2, "cd_dm_n8_s2"), (1, 1, "cd_dm_n1_s1"), (1, 2, "cd_dm_n1_s2"), (1, 4, "cd_dm_n1_s4"),
    (2, 1, "cd_dm_n2_s1"), (2, 2, "cd_dm_n2_s2"), (2, 4, "cd_dm_n2_s4"), (4, 2, "cd_dm_n4_s2"), (4, 4, "cd_dm_n4_s4")];
pub const DT_NAMES: [(usize, &str); 3] = [(1, "cd_dt_s1"), (2, "cd_dt_s2"), (4, "cd_dt_s4")];
pub fn dm_name(nb: usize, sk: usize) -> &'static str { DM_NAMES.iter().find(|x| x.0 == nb && x.1 == sk).map(|x| x.2).unwrap_or("cd_dm_n2_s4") }
/// The MMA family's split-K SK (any NB) must equal the twin's: SK is part of the arithmetic.
pub fn dt_name(sk: usize) -> &'static str { DT_NAMES.iter().find(|x| x.0 == sk).map(|x| x.1).unwrap_or("cd_dt_s4") }
/// (NB, SK) of the MMA family: `SD_CD_DM=NB,SK`, default DM_DEFAULT (cpa-bench, 27B down shape).
pub const DM_DEFAULT: (usize, usize) = (4, 2);
pub fn dm_shape() -> (usize, usize) {
    static S: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let v: Option<(usize, usize)> = std::env::var("SD_CD_DM").ok().and_then(|s| {
            let mut it = s.split(',').filter_map(|x| x.trim().parse().ok());
            Some((it.next()?, it.next()?))
        });
        match v { Some(x) if DM_NAMES.iter().any(|y| y.0 == x.0 && y.1 == x.1) => x, _ => DM_DEFAULT }
    })
}
/// The multi-row down (`cd_downk_r4_k{k}`, k = 1..=8; 2026-11-02, compact + MTP): W read once per pass for all k rows,
/// per row bit-identical to cd_down_r4 (so K-row verify == K = 1 decode). Used for every compact pass unless
/// `SD_CD_OLD=1` (the per-row-looping `down_kernel()` family, kept for benches).
pub fn downk_kernel(k: usize) -> &'static str {
    const N: [&str; 8] = ["cd_downk_r4_k1", "cd_downk_r4_k2", "cd_downk_r4_k3", "cd_downk_r4_k4", "cd_downk_r4_k5", "cd_downk_r4_k6",
                          "cd_downk_r4_k7", "cd_downk_r4_k8"];
    N[k.clamp(1, 8) - 1]
}

/// The down kernel and its rows per simdgroup (2 simdgroups per threadgroup): `cd_down_r{CD_NR}`, CD_NR from
/// `SD_CD_NR` (4 or 8), default 4 (cpa-bench, 27B down shape, k = 1: r4 0.163 ms/layer = 207 GB/s, r8 0.193 ms; the
/// full Mma down 0.300 ms).
pub fn down_kernel() -> (&'static str, usize) {
    static NR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let nr = *NR.get_or_init(|| match std::env::var("SD_CD_NR").ok().as_deref() { Some("8") => 8, _ => 4 });
    (if nr == 4 { "cd_down_r4" } else { "cd_down_r8" }, nr)
}

/// Bytes of the f-sum scratch for `tmax` rows.
pub fn scratch_bytes(inter: usize, tmax: usize) -> usize { tmax * (inter / 64) * 4 }

/// Arena bytes of one compact layer's FFN (gate + up rows, packed down + full down S / B, gid, gstart).
pub fn layer_bytes(hidden: usize, inter: usize, kc: usize) -> usize {
    let a16 = |x: usize| x.div_ceil(16) * 16;
    2 * kc * (hidden / 2 + 2 * (hidden / 64 * 2)) + hidden * kc / 2 + 2 * hidden * (inter / 64) * 2 + a16(kc * 2) + a16((inter / 64 + 1) * 4)
        + a16(kc / 8 * 4)
}

impl QwenEngine {
    /// The compact set now loaded (None with ffn_compact off).
    pub fn ffn_compact_set(&self) -> Option<Arc<CompactSet>> { self.cfc.as_ref().map(|c| c.set.borrow().clone()) }
    /// Load another compact set of the same per-layer sizes in place (re-reads the FFN tensors and gathers). Engine
    /// idle only (no pass in flight). Returns (bytes read, seconds).
    pub fn ffn_compact_load(&self, set: Arc<CompactSet>) -> Result<(u64, f64), String> {
        let c = self.cfc.as_ref().ok_or("ffn_compact: off (QwenOptions.ffn_compact)")?;
        set.validate()?;
        if set.inter != c.inter || set.keep.len() != c.jobs.len() { return Err("ffn_compact: set shape differs".into()); }
        for j in &c.jobs {
            let kc = j.ffn.cd.map_or(0, |d| d.kc);
            if set.keep[j.layer].len() != kc { return Err(format!("ffn_compact: layer {} has {} neurons, the arena holds {}", j.layer, set.keep[j.layer].len(), kc)); }
        }
        let t0 = std::time::Instant::now();
        let n = gather(&c.files, &c.jobs, &set, c.hidden, c.inter, self.arena.base(), c.threads)?;
        *c.set.borrow_mut() = set;
        Ok((n, t0.elapsed().as_secs_f64()))
    }
}

/// Kept FFN neurons per main layer that fit a RAM cap (2026-11-02; the rule: sizes from a formula, never hardcoded).
/// Everything comes from the engine's own budget plan for the run's options `opt` (MTP on / off, spec depth, max_ctx,
/// KV format):
///   K = floor8((cap − rest − ffn_fixed) / (layers × per_neuron))
/// - rest = every non-FFN budget row (attention / DeltaNet weights, lm_head, norms, KV at max_ctx, scratch, alignment
///   slack; with MTP also the MTP layer, MTP KV, the 2nd DeltaNet state copy and the replay rows) + the process reserve;
/// - ffn_fixed + layers × per_neuron × K = the compact FFN row, linear in K (two plans solve it): per_neuron = gate row
///   + up row + down column + gid + wmeta of one neuron; ffn_fixed = the down scales / biases kept whole + gstart.
/// Then checked against the plan at K itself (down by 8 while page alignment pushes it over the cap).
#[derive(Clone, Debug)]
pub struct KeepPlan {
    pub k: usize,
    pub share: f64,
    pub cap: u64,
    /// non-FFN budget rows + process reserve
    pub rest: u64,
    pub ffn_fixed: f64,
    /// bytes one kept neuron adds to one layer
    pub per_neuron: f64,
    pub layers: usize,
    pub inter: usize,
    /// planned total (arena + reserve) at K
    pub total: u64,
}

fn uniform(layers: usize, inter: usize, k: usize) -> CompactSet { CompactSet { inter, keep: vec![(0..k as u32).collect(); layers] } }

/// (total = arena + reserve, FFN row, rest) of the plan with a uniform compact set of `k` per layer.
fn plan_at(dir: &str, opt: &crate::qwen_engine::QwenOptions, layers: usize, inter: usize, k: usize) -> Result<(u64, u64, u64), String> {
    let mut o = opt.clone();
    o.ffn_compact = Some(Arc::new(uniform(layers, inter, k)));
    let b = QwenEngine::budget(dir, &o)?;
    let al = |x: u64| x.div_ceil(streamdense_core::arena::ALIGN as u64) * streamdense_core::arena::ALIGN as u64;
    let ffn: u64 = b.items.iter().filter(|(n, _)| n.starts_with("FFN")).map(|x| al(x.1)).sum();
    let total = b.arena_bytes() + b.reserve;
    Ok((total, ffn, total - ffn))
}

/// Pure part of the formula: K (multiple of 8, in [8, inter]) from the cap and the three plan terms.
pub fn keep_formula(cap: u64, rest: u64, ffn_fixed: f64, per_neuron: f64, layers: usize, inter: usize) -> usize {
    let room = cap as f64 - rest as f64 - ffn_fixed;
    let k = if room <= 0.0 { 0 } else { (room / (layers as f64 * per_neuron)).floor() as usize };
    (k / 8 * 8).clamp(8, inter / 8 * 8)
}

impl QwenEngine {
    /// `KeepPlan` for `cap` bytes with the run's options (CPU only: safetensors headers, nothing loaded).
    pub fn compact_keep_for_cap(dir: &str, opt: &crate::qwen_engine::QwenOptions, cap: u64) -> Result<KeepPlan, String> {
        let cfg = crate::qwen_config::QwenConfig::from_model_dir(dir)?;
        let (l, n) = (cfg.layers, cfg.inter);
        let (k1, k2) = (n / 8 * 8, (n / 16) * 8);
        let (_, f1, rest) = plan_at(dir, opt, l, n, k1)?;
        let (_, f2, _) = plan_at(dir, opt, l, n, k2)?;
        let per = (f1 - f2) as f64 / (l * (k1 - k2)) as f64;
        let fixed = f1 as f64 - per * (l * k1) as f64;
        let mut k = keep_formula(cap, rest, fixed, per, l, n);
        let mut total = plan_at(dir, opt, l, n, k)?.0;
        while total > cap && k > 8 { k -= 8; total = plan_at(dir, opt, l, n, k)?.0; }
        Ok(KeepPlan { k, share: k as f64 / n as f64, cap, rest, ffn_fixed: fixed, per_neuron: per, layers: l, inter: n, total })
    }
}

#[cfg(test)]
mod keep_tests {
    use super::keep_formula;
    #[test]
    fn keep_formula_floors_to_8_and_clamps() {
        // room 1000 B over 2 layers x 10 B = 50 -> 48
        assert_eq!(keep_formula(2000, 900, 100.0, 10.0, 2, 1024), 48);
        assert_eq!(keep_formula(100, 900, 0.0, 10.0, 2, 1024), 8);
        assert_eq!(keep_formula(u64::MAX / 2, 0, 0.0, 10.0, 2, 1020), 1016);
    }
}
