//! per-layer RESIDENT FFN sets (real unload to SSD, reload, prefetch).
//!
//! Each layer's FFN lives in its OWN Metal buffer (`ResidentLayer`) holding only the hot 64-neuron blocks, packed as
//! a regular MLX q4 g64 FFN with inter' = 64 · hot: gate / up keep the hot rows (+ their scale / bias rows), down keeps
//! the hot 64-column groups (+ their scale / bias column) of every row, in ascending block order. The existing q4
//! kernels run on it unchanged (they only see a smaller `inter`); kept values are bit-identical, nothing is
//! requantized. With every block hot the bytes are exactly the safetensors tensors (lossless).
//!
//! Changing the set = `Resident::build`: a NEW buffer for the new set is filled from the old buffer (blocks kept, RAM
//! copy) and from SSD (blocks added: gate / up rows pread from the original safetensors by offset; down groups from
//! the down sidecar, else from full safetensors rows through a small staging buffer); blocks leaving are written to the
//! sidecar first. The old buffer stays valid (the GPU may still read it) until the caller swaps and drops it, so the
//! peak is the old footprint + one new layer buffer. Dropping the old buffer returns its pages to phys_footprint;
//! madvise on part of a Metal buffer does not (measured), hence one buffer per layer.
//!
//! The down sidecar: one sparse file, a fixed slot per (layer, block) = the block's down groups in block-major order
//! (w [hidden][32 B] | s [hidden] bf16 | b [hidden] bf16). Only written slots use disk (= the unloaded down bytes);
//! it is created per process, filled from RAM (bytes identical to the safetensors) and deleted on drop.
//!
//! `BlockSet` = one layer's FFN row of the engine's `ReadMask` (`readmask.rs`), same u64 word layout:
//! `BlockSet::from_words(inter / 64, &mask.ffn_blocks[l])` (no dependency on readmask.rs, so it builds alone).
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use streamdense_core::arena::Region;
use streamdense_core::io::NoCacheFile;
use streamdense_core::{DType, SafeIndex};
use streamdense_metal::{contents, Buffer, Gpu};

use crate::weights::{FfnW, Q4b};

/// Neurons per block = MLX q4 group of the down projection = mask block.
pub const BLOCK: usize = 64;
const ALIGN: usize = 256;
const PAGE: usize = 16384;
/// Rows of the down tensor per staging read when down blocks come from the safetensors (full rows).
const ST_ROWS: usize = 32;

// ------------------------------------------------------------------------------------------------ block sets

/// Hot 64-neuron blocks of one layer (bit b = block b, i.e. neurons [64·b, 64·b + 64)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockSet {
    n: usize,
    bits: Vec<u64>,
}
impl BlockSet {
    pub fn empty(n: usize) -> Self { BlockSet { n, bits: vec![0; n.div_ceil(64)] } }
    pub fn full(n: usize) -> Self {
        let mut s = Self::empty(n);
        for b in 0..n { s.insert(b as u32); }
        s
    }
    pub fn from_list(n: usize, list: &[u32]) -> Self {
        let mut s = Self::empty(n);
        for &b in list { s.insert(b); }
        s
    }
    /// From one layer row of the engine's `ReadMask` (`mask.ffn_blocks[l]`: same layout, u64 words, bit b % 64 of
    /// word b / 64 = block b, bit set = read / resident). Bits past `n` are ignored.
    pub fn from_words(n: usize, words: &[u64]) -> Self {
        let mut s = Self::empty(n);
        for (d, w) in s.bits.iter_mut().zip(words) { *d = *w; }
        if n % 64 != 0 { if let Some(last) = s.bits.last_mut() { *last &= (1u64 << (n % 64)) - 1; } }
        s
    }
    /// The words, in `ReadMask.ffn_blocks[l]` layout.
    pub fn words(&self) -> &[u64] { &self.bits }
    /// blocks in the layer (inter / 64)
    pub fn blocks(&self) -> usize { self.n }
    pub fn insert(&mut self, b: u32) {
        assert!((b as usize) < self.n, "block {} out of {}", b, self.n);
        self.bits[b as usize / 64] |= 1 << (b % 64);
    }
    pub fn remove(&mut self, b: u32) { if (b as usize) < self.n { self.bits[b as usize / 64] &= !(1 << (b % 64)); } }
    pub fn contains(&self, b: u32) -> bool { (b as usize) < self.n && self.bits[b as usize / 64] >> (b % 64) & 1 == 1 }
    pub fn count(&self) -> usize { self.bits.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn is_full(&self) -> bool { self.count() == self.n }
    pub fn union(&self, o: &BlockSet) -> BlockSet {
        assert_eq!(self.n, o.n);
        BlockSet { n: self.n, bits: self.bits.iter().zip(&o.bits).map(|(a, b)| a | b).collect() }
    }
    /// sorted block indices
    pub fn list(&self) -> Vec<u32> { (0..self.n as u32).filter(|&b| self.contains(b)).collect() }
}

// ------------------------------------------------------------------------------------------------ shapes / layout

/// One layer's FFN: gate / up [inter, hidden], down [hidden, inter], MLX affine q4 g64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfnShape {
    pub hidden: usize,
    pub inter: usize,
}

/// A resident layer's placement inside its own buffer (Regions are offsets from the buffer start).
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub ffn: FfnW,
    /// buffer size (page multiple, ≥ one page)
    pub bytes: usize,
    pub hot: usize,
}

impl FfnShape {
    pub fn new(hidden: usize, inter: usize) -> Result<Self, String> {
        if hidden % 64 != 0 || inter % 64 != 0 || hidden == 0 || inter == 0 {
            return Err(format!("resident: hidden {} / inter {} must be non-zero multiples of 64", hidden, inter));
        }
        Ok(FfnShape { hidden, inter })
    }
    pub fn blocks(&self) -> usize { self.inter / BLOCK }
    /// bytes of one packed gate / up row and of its scale (or bias) row
    fn gu_w(&self) -> usize { self.hidden / 2 }
    fn gu_sb(&self) -> usize { self.hidden / 64 * 2 }
    fn down_row_w(&self) -> usize { self.inter / 2 }
    fn down_row_sb(&self) -> usize { self.inter / 64 * 2 }
    /// bytes one block holds across gate + up + down (w, s, b)
    pub fn block_bytes(&self) -> usize { 2 * BLOCK * (self.gu_w() + 2 * self.gu_sb()) + self.hidden * (32 + 2 * 2) }
    /// the safetensors bytes of the whole layer FFN (= block_bytes · blocks)
    pub fn tensor_bytes(&self) -> usize { self.block_bytes() * self.blocks() }
    /// sidecar slot: down groups of one block, block-major, padded to a page
    fn slot_bytes(&self) -> usize { (self.hidden * 36).div_ceil(PAGE) * PAGE }

    /// Compact layout for `h` hot blocks.
    pub fn layout(&self, h: usize) -> Layout {
        let n = h * BLOCK;
        let mut off = 0usize;
        let mut take = |len: usize| {
            let r = Region { off, len };
            off = (off + len + ALIGN - 1) / ALIGN * ALIGN;
            r
        };
        let mut gu = |rows: usize| Q4b { rows, cols: self.hidden, w: take(rows * self.gu_w()), s: take(rows * self.gu_sb()), b: take(rows * self.gu_sb()) };
        let gate = gu(n);
        let up = gu(n);
        let down = Q4b { rows: self.hidden, cols: n, w: take(self.hidden * h * 32), s: take(self.hidden * h * 2), b: take(self.hidden * h * 2) };
        Layout { ffn: FfnW { gate, up, down, cd: None }, bytes: off.div_ceil(PAGE).max(1) * PAGE, hot: h }
    }
    pub fn full_bytes(&self) -> usize { self.layout(self.blocks()).bytes }
}

// ------------------------------------------------------------------------------------------------ SSD sources

/// Absolute byte offset of a tensor in file `file` of the model's file list.
#[derive(Clone, Copy, Debug, Default)]
pub struct At {
    pub file: usize,
    pub off: u64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4At {
    pub w: At,
    pub s: At,
    pub b: At,
}
/// Where one layer's FFN tensors are on SSD (the original safetensors; nothing is converted).
#[derive(Clone, Copy, Debug, Default)]
pub struct FfnSrc {
    pub gate: Q4At,
    pub up: Q4At,
    pub down: Q4At,
}
impl FfnSrc {
    /// From a model index; `mlp` = tensor prefix up to and including `mlp.` (e.g. `language_model.model.layers.7.mlp.`).
    pub fn from_index(idx: &SafeIndex, mlp: &str, shape: FfnShape) -> Result<Self, String> {
        let (h, i) = (shape.hidden, shape.inter);
        let q = |name: &str, rows: usize, cols: usize| -> Result<Q4At, String> {
            let get = |part: &str, dt: DType, c: usize| -> Result<At, String> {
                let n = format!("{}{}.{}", mlp, name, part);
                let t = idx.get(&n).ok_or_else(|| format!("resident: tensor {} not found", n))?;
                if t.dtype != dt || t.shape != [rows, c] {
                    return Err(format!("resident: {} is {:?} {:?}, expected {:?} [{}, {}]", n, t.dtype, t.shape, dt, rows, c));
                }
                Ok(At { file: t.file, off: t.off })
            };
            Ok(Q4At { w: get("weight", DType::U32, cols / 8)?, s: get("scales", DType::BF16, cols / 64)?, b: get("biases", DType::BF16, cols / 64)? })
        };
        Ok(FfnSrc { gate: q("gate_proj", i, h)?, up: q("up_proj", i, h)?, down: q("down_proj", h, i)? })
    }
    /// Synthetic layout: gate w, s, b | up w, s, b | down w, s, b back to back from `base` in file `file`.
    pub fn packed(file: usize, base: u64, shape: FfnShape) -> FfnSrc {
        let mut o = base;
        let mut at = |len: usize| { let a = At { file, off: o }; o += len as u64; a };
        let (gw, gs) = (shape.inter * shape.gu_w(), shape.inter * shape.gu_sb());
        let (dw, ds) = (shape.hidden * shape.down_row_w(), shape.hidden * shape.down_row_sb());
        let gate = Q4At { w: at(gw), s: at(gs), b: at(gs) };
        let up = Q4At { w: at(gw), s: at(gs), b: at(gs) };
        let down = Q4At { w: at(dw), s: at(ds), b: at(ds) };
        FfnSrc { gate, up, down }
    }
}

// ------------------------------------------------------------------------------------------------ sidecar

/// Sparse per-process file holding the down groups of unloaded blocks (see module doc). Deleted on drop.
pub struct Sidecar {
    fd: i32,
    path: String,
    slot: usize,
    blocks: usize,
    /// per layer: blocks whose slot holds valid bytes
    valid: Vec<BlockSet>,
}
impl Sidecar {
    pub fn create(path: &str, layers: usize, shape: FfnShape) -> Result<Self, String> {
        let c = std::ffi::CString::new(path).map_err(|e| e.to_string())?;
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC, 0o600 as libc::c_uint) };
        if fd < 0 { return Err(format!("resident: cannot create sidecar {}: {}", path, std::io::Error::last_os_error())); }
        if unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) } != 0 {
            unsafe { libc::close(fd) };
            return Err(format!("resident: F_NOCACHE failed on {}", path));
        }
        let blocks = shape.blocks();
        Ok(Sidecar { fd, path: path.to_string(), slot: shape.slot_bytes(), blocks, valid: vec![BlockSet::empty(blocks); layers] })
    }
    fn off(&self, layer: usize, b: u32) -> u64 { ((layer * self.blocks + b as usize) * self.slot) as u64 }
    pub fn valid(&self, layer: usize) -> &BlockSet { &self.valid[layer] }
    /// bytes the file really occupies on disk (sparse: only written slots)
    pub fn disk_bytes(&self) -> u64 {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        unsafe { libc::fstat(self.fd, &mut st) };
        st.st_blocks as u64 * 512
    }
    pub fn path(&self) -> &str { &self.path }
}
impl Drop for Sidecar {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
        let _ = std::fs::remove_file(&self.path);
    }
}

fn pwrite_all(fd: i32, src: &[u8], off: u64) -> Result<(), String> {
    let mut done = 0usize;
    while done < src.len() {
        let r = unsafe { libc::pwrite(fd, src[done..].as_ptr() as *const libc::c_void, src.len() - done, (off + done as u64) as libc::off_t) };
        if r <= 0 { return Err(format!("resident: sidecar pwrite at {}: {}", off, std::io::Error::last_os_error())); }
        done += r as usize;
    }
    Ok(())
}
fn pread_all(fd: i32, dst: &mut [u8], off: u64) -> Result<(), String> {
    let mut done = 0usize;
    while done < dst.len() {
        let r = unsafe { libc::pread(fd, dst[done..].as_mut_ptr() as *mut libc::c_void, dst.len() - done, (off + done as u64) as libc::off_t) };
        if r <= 0 { return Err(format!("resident: sidecar pread at {}: {}", off, std::io::Error::last_os_error())); }
        done += r as usize;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ resident layer

/// One layer's resident FFN: its own Metal buffer with the compact q4 matrices of the hot blocks.
pub struct ResidentLayer {
    pub layer: usize,
    pub set: BlockSet,
    hot: Vec<u32>,
    pub lay: Layout,
    buf: Buffer,
}
// The buffer handle moves between the engine thread and a background I/O thread (prefetch / reload). MTLBuffer is
// thread-safe per Apple's Metal docs (objc2 just does not mark protocol objects Send); the CPU writes all happen in
// `build` before the layer is handed over, and the weights are read-only afterwards.
unsafe impl Send for ResidentLayer {}

impl ResidentLayer {
    /// The buffer the FFN kernels bind (offsets from `ffn_w`).
    pub fn buffer(&self) -> &Buffer { &self.buf }
    /// gate / up / down placement inside `buffer()`; `gate.rows` = `down.cols` = inter' = 64 · hot.
    pub fn ffn_w(&self) -> FfnW { self.lay.ffn }
    /// sorted original block index of each compact block (compact block i = original block hot()[i])
    pub fn hot(&self) -> &[u32] { &self.hot }
    /// compact index of original block `b`, if resident
    pub fn compact(&self, b: u32) -> Option<usize> { self.hot.binary_search(&b).ok() }
    /// an original-index block list (e.g. a ReadMask row) in compact indices; non-resident blocks are dropped
    pub fn to_compact(&self, list: &[u32]) -> Vec<u32> { list.iter().filter_map(|&b| self.compact(b).map(|i| i as u32)).collect() }
    pub fn bytes(&self) -> usize { self.lay.bytes }
    pub fn base(&self) -> *const u8 { contents(&self.buf) }
    /// the buffer's bytes (CPU view; the GPU must not be writing it, it never does: weights are read-only)
    pub fn as_bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(contents(&self.buf), self.lay.bytes) } }
}

/// What one `build` did.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildStats {
    /// bytes copied from the old buffer
    pub ram_bytes: u64,
    /// bytes read from the safetensors (gate / up rows + full down rows when no sidecar slot)
    pub st_bytes: u64,
    /// of `st_bytes`: full down rows read for blocks without a sidecar slot
    pub st_down_bytes: u64,
    pub side_read: u64,
    pub side_written: u64,
    /// new buffer size
    pub new_bytes: u64,
    pub secs: f64,
}
impl BuildStats {
    pub fn ssd_read(&self) -> u64 { self.st_bytes + self.side_read }
}

enum Job {
    /// RAM → RAM (gate / up row runs)
    Copy { dst: usize, src: usize, len: usize },
    /// down rows [r0, r1): copy the kept column runs from the old buffer
    DownCopy { r0: usize, r1: usize },
    /// safetensors → new buffer, contiguous
    Read { dst: usize, len: usize, file: usize, off: u64 },
    /// sidecar slot of original block `b` → compact down column `i`
    SideRead { i: usize, b: u32 },
    /// safetensors down rows [r0, r1) → the compact columns of `st_cols`
    StDown { r0: usize, r1: usize },
    /// old compact column `j` (original block `b`) → sidecar slot
    SideWrite { j: usize, b: u32 },
}

/// The per-model resident manager: SSD sources of every layer, the sidecar, the I/O thread count.
pub struct Resident {
    pub shape: FfnShape,
    files: Vec<NoCacheFile>,
    src: Vec<FfnSrc>,
    side: Option<Sidecar>,
    pub threads: usize,
    /// max bytes per contiguous read / copy piece
    pub chunk: usize,
}

impl Resident {
    /// From the model index: `mlp_prefixes[l]` = tensor prefix of layer l's `mlp.`; `sidecar` = path of the down
    /// sidecar (None: down blocks are always re-read as full safetensors rows).
    pub fn new(idx: &SafeIndex, shape: FfnShape, mlp_prefixes: &[String], sidecar: Option<&str>, threads: usize) -> Result<Self, String> {
        let src = mlp_prefixes.iter().map(|p| FfnSrc::from_index(idx, p, shape)).collect::<Result<Vec<_>, _>>()?;
        Self::with_sources(idx.files(), src, shape, sidecar, threads)
    }
    pub fn with_sources(files: &[String], src: Vec<FfnSrc>, shape: FfnShape, sidecar: Option<&str>, threads: usize) -> Result<Self, String> {
        let files = files.iter().map(|f| NoCacheFile::open(f)).collect::<Result<Vec<_>, _>>()?;
        let side = match sidecar { Some(p) => Some(Sidecar::create(p, src.len(), shape)?), None => None };
        Ok(Resident { shape, files, src, side, threads: threads.max(1), chunk: 1 << 20 })
    }
    pub fn layers(&self) -> usize { self.src.len() }
    pub fn sidecar(&self) -> Option<&Sidecar> { self.side.as_ref() }
    /// true if building a layer of `want` now keeps phys_footprint ≤ `cap` (old buffer still alive during the build)
    pub fn fits(&self, want: &BlockSet, cap: u64) -> bool {
        streamdense_core::footprint::self_footprint() + self.shape.layout(want.count()).bytes as u64 <= cap
    }

    /// First load of a layer for `want`. Full set (or no sidecar): one read from the safetensors. Partial set with a
    /// sidecar: full layer first, then repacked (fills the sidecar slots of the cold blocks; peak +1 layer).
    pub fn load_layer(&mut self, gpu: &Gpu, layer: usize, want: &BlockSet) -> Result<(ResidentLayer, BuildStats), String> {
        if want.is_full() || self.side.is_none() { return self.build(gpu, layer, None, want); }
        let (full, s1) = self.build(gpu, layer, None, &BlockSet::full(self.shape.blocks()))?;
        let (r, s2) = self.build(gpu, layer, Some(&full), want)?;
        drop(full);
        Ok((r, BuildStats {
            ram_bytes: s2.ram_bytes, st_bytes: s1.st_bytes, st_down_bytes: s1.st_down_bytes, side_read: 0,
            side_written: s2.side_written, new_bytes: s2.new_bytes, secs: s1.secs + s2.secs,
        }))
    }

    /// Build layer `layer` for `want` into a NEW buffer: kept blocks copied from `old` (left untouched and usable),
    /// added blocks read from SSD, blocks leaving `old` written to the sidecar (once). Unload, reload, prefetch and
    /// the fallback are all this call with a different `want`.
    pub fn build(&mut self, gpu: &Gpu, layer: usize, old: Option<&ResidentLayer>, want: &BlockSet) -> Result<(ResidentLayer, BuildStats), String> {
        let t0 = Instant::now();
        let sh = self.shape;
        if want.blocks() != sh.blocks() { return Err(format!("resident: set of {} blocks for a layer of {}", want.blocks(), sh.blocks())); }
        if let Some(o) = old {
            if o.layer != layer { return Err(format!("resident: old buffer is layer {}, building {}", o.layer, layer)); }
        }
        let hot = want.list();
        let h = hot.len();
        let lay = sh.layout(h);
        let buf = gpu.buffer(lay.bytes)?;
        let base = contents(&buf) as usize;
        let src = self.src[layer];
        let side_valid = self.side.as_ref().map(|s| s.valid[layer].clone());

        // where each new block comes from
        let from_old: Vec<Option<usize>> = hot.iter().map(|&b| old.and_then(|o| o.compact(b))).collect();
        let (gw, gsb) = (sh.gu_w(), sh.gu_sb());
        let mut jobs: Vec<Job> = Vec::new();
        let mut st = BuildStats { new_bytes: lay.bytes as u64, ..Default::default() };
        let split = |jobs: &mut Vec<Job>, mk: &dyn Fn(usize, usize) -> Job, len: usize, chunk: usize| {
            let mut d = 0;
            while d < len { let n = chunk.min(len - d); jobs.push(mk(d, n)); d += n; }
        };

        // gate / up: runs of consecutive new blocks with one source each
        let mut i = 0;
        while i < h {
            let mut k = i + 1;
            match from_old[i] {
                Some(j) => while k < h && from_old[k] == Some(j + (k - i)) { k += 1; },
                None => while k < h && from_old[k].is_none() && hot[k] == hot[i] + (k - i) as u32 { k += 1; },
            }
            let rows = (k - i) * BLOCK;
            let r_new = i * BLOCK;
            for (mi, (nm, at)) in [(lay.ffn.gate, src.gate), (lay.ffn.up, src.up)].into_iter().enumerate() {
                for (reg, rb, a, oreg) in [(nm.w, gw, at.w, 0usize), (nm.s, gsb, at.s, 1), (nm.b, gsb, at.b, 2)] {
                    let dst = base + reg.off + r_new * rb;
                    let len = rows * rb;
                    match from_old[i] {
                        Some(j) => {
                            let o = old.unwrap();
                            let om = if mi == 0 { o.lay.ffn.gate } else { o.lay.ffn.up };
                            let oreg = [om.w, om.s, om.b][oreg];
                            let srcp = o.base() as usize + oreg.off + j * BLOCK * rb;
                            split(&mut jobs, &|d, n| Job::Copy { dst: dst + d, src: srcp + d, len: n }, len, self.chunk);
                            st.ram_bytes += len as u64;
                        }
                        None => {
                            let off = a.off + (hot[i] as usize * BLOCK * rb) as u64;
                            split(&mut jobs, &|d, n| Job::Read { dst: dst + d, len: n, file: a.file, off: off + d as u64 }, len, self.chunk);
                            st.st_bytes += len as u64;
                        }
                    }
                }
            }
            i = k;
        }

        // down: kept column runs (new i0, old j0, len) copied row by row; added blocks from the sidecar or full rows
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for (i, f) in from_old.iter().enumerate() {
            if let Some(j) = *f {
                match runs.last_mut() {
                    Some(r) if r.0 + r.2 == i && r.1 + r.2 == j => r.2 += 1,
                    _ => runs.push((i, j, 1)),
                }
            }
        }
        if !runs.is_empty() {
            let mut r0 = 0;
            while r0 < sh.hidden { let r1 = (r0 + 64).min(sh.hidden); jobs.push(Job::DownCopy { r0, r1 }); r0 = r1; }
            st.ram_bytes += (runs.iter().map(|r| r.2).sum::<usize>() * sh.hidden * 36) as u64;
        }
        let mut st_cols: Vec<(usize, u32)> = Vec::new();
        for (i, f) in from_old.iter().enumerate() {
            if f.is_some() { continue; }
            match &side_valid {
                Some(v) if v.contains(hot[i]) => { jobs.push(Job::SideRead { i, b: hot[i] }); st.side_read += (sh.hidden * 36) as u64; }
                _ => st_cols.push((i, hot[i])),
            }
        }
        if !st_cols.is_empty() {
            let mut r0 = 0;
            while r0 < sh.hidden { let r1 = (r0 + ST_ROWS).min(sh.hidden); jobs.push(Job::StDown { r0, r1 }); r0 = r1; }
            let b = (sh.hidden * (sh.down_row_w() + 2 * sh.down_row_sb())) as u64;
            st.st_bytes += b;
            st.st_down_bytes += b;
        }
        // leaving blocks → sidecar (once per block)
        let mut leaving: Vec<u32> = Vec::new();
        if let (Some(o), Some(v)) = (old, &side_valid) {
            for (j, &b) in o.hot.iter().enumerate() {
                if !want.contains(b) && !v.contains(b) { jobs.push(Job::SideWrite { j, b }); leaving.push(b); st.side_written += (sh.hidden * 36) as u64; }
            }
        }

        // run
        let side_fd = self.side.as_ref().map(|s| s.fd).unwrap_or(-1);
        let side_slot = |b: u32| self.side.as_ref().map(|s| s.off(layer, b)).unwrap_or(0);
        let (ohot, obase, olay) = match old { Some(o) => (o.hot.len(), o.base() as usize, o.lay), None => (0, 0, lay) };
        let next = AtomicUsize::new(0);
        let err: Mutex<Option<String>> = Mutex::new(None);
        let files = &self.files;
        let (dw_new, dsb_new) = (h * 32, h * 2);
        let (dw_old, dsb_old) = (ohot * 32, ohot * 2);
        let d_new = lay.ffn.down;
        let d_old = olay.ffn.down;
        let hid = sh.hidden;
        let (drw, drsb) = (sh.down_row_w(), sh.down_row_sb());
        let run_job = |job: &Job, stage: &mut Vec<u8>| -> Result<(), String> {
            match *job {
                Job::Copy { dst, src, len } => unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) },
                Job::Read { dst, len, file, off } => files[file].read_at(dst as *mut u8, len, off)?,
                Job::DownCopy { r0, r1 } => unsafe {
                    for r in r0..r1 {
                        for &(i0, j0, n) in &runs {
                            let (nw, ow) = (base + d_new.w.off + r * dw_new + i0 * 32, obase + d_old.w.off + r * dw_old + j0 * 32);
                            std::ptr::copy_nonoverlapping(ow as *const u8, nw as *mut u8, n * 32);
                            for (nr, or) in [(d_new.s, d_old.s), (d_new.b, d_old.b)] {
                                let (np, op) = (base + nr.off + r * dsb_new + i0 * 2, obase + or.off + r * dsb_old + j0 * 2);
                                std::ptr::copy_nonoverlapping(op as *const u8, np as *mut u8, n * 2);
                            }
                        }
                    }
                },
                Job::SideRead { i, b } => {
                    stage.resize(hid * 36, 0);
                    pread_all(side_fd, &mut stage[..hid * 36], side_slot(b))?;
                    unsafe {
                        for r in 0..hid {
                            std::ptr::copy_nonoverlapping(stage.as_ptr().add(r * 32), (base + d_new.w.off + r * dw_new + i * 32) as *mut u8, 32);
                            std::ptr::copy_nonoverlapping(stage.as_ptr().add(hid * 32 + r * 2), (base + d_new.s.off + r * dsb_new + i * 2) as *mut u8, 2);
                            std::ptr::copy_nonoverlapping(stage.as_ptr().add(hid * 34 + r * 2), (base + d_new.b.off + r * dsb_new + i * 2) as *mut u8, 2);
                        }
                    }
                }
                Job::StDown { r0, r1 } => {
                    let n = r1 - r0;
                    let (wl, sl) = (n * drw, n * drsb);
                    stage.resize(wl + 2 * sl, 0);
                    let (w, rest) = stage.split_at_mut(wl);
                    let (s, b) = rest.split_at_mut(sl);
                    files[src.down.w.file].read_at(w.as_mut_ptr(), wl, src.down.w.off + (r0 * drw) as u64)?;
                    files[src.down.s.file].read_at(s.as_mut_ptr(), sl, src.down.s.off + (r0 * drsb) as u64)?;
                    files[src.down.b.file].read_at(b.as_mut_ptr(), sl, src.down.b.off + (r0 * drsb) as u64)?;
                    unsafe {
                        for r in 0..n {
                            let rr = r0 + r;
                            for &(i, ob) in &st_cols {
                                let ob = ob as usize;
                                std::ptr::copy_nonoverlapping(w.as_ptr().add(r * drw + ob * 32), (base + d_new.w.off + rr * dw_new + i * 32) as *mut u8, 32);
                                std::ptr::copy_nonoverlapping(s.as_ptr().add(r * drsb + ob * 2), (base + d_new.s.off + rr * dsb_new + i * 2) as *mut u8, 2);
                                std::ptr::copy_nonoverlapping(b.as_ptr().add(r * drsb + ob * 2), (base + d_new.b.off + rr * dsb_new + i * 2) as *mut u8, 2);
                            }
                        }
                    }
                }
                Job::SideWrite { j, b } => {
                    stage.resize(hid * 36, 0);
                    unsafe {
                        for r in 0..hid {
                            std::ptr::copy_nonoverlapping((obase + d_old.w.off + r * dw_old + j * 32) as *const u8, stage.as_mut_ptr().add(r * 32), 32);
                            std::ptr::copy_nonoverlapping((obase + d_old.s.off + r * dsb_old + j * 2) as *const u8, stage.as_mut_ptr().add(hid * 32 + r * 2), 2);
                            std::ptr::copy_nonoverlapping((obase + d_old.b.off + r * dsb_old + j * 2) as *const u8, stage.as_mut_ptr().add(hid * 34 + r * 2), 2);
                        }
                    }
                    pwrite_all(side_fd, &stage[..hid * 36], side_slot(b))?;
                }
            }
            Ok(())
        };
        std::thread::scope(|s| {
            for _ in 0..self.threads.min(jobs.len()).max(1) {
                s.spawn(|| {
                    let mut stage: Vec<u8> = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= jobs.len() || err.lock().unwrap().is_some() { break; }
                        if let Err(e) = run_job(&jobs[k], &mut stage) { *err.lock().unwrap() = Some(e); break; }
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap() { return Err(e); }
        if let Some(sc) = self.side.as_mut() { for b in leaving { sc.valid[layer].insert(b); } }
        st.secs = t0.elapsed().as_secs_f64();
        Ok((ResidentLayer { layer, set: want.clone(), hot, lay, buf }, st))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes_27b() {
        let sh = FfnShape::new(5120, 17408).unwrap();
        assert_eq!(sh.blocks(), 272);
        assert_eq!(sh.block_bytes(), 552_960);
        assert_eq!(sh.tensor_bytes(), 150_405_120);
        // 64 layers = the 9.63 GB FFN row of the budget
        assert_eq!(sh.tensor_bytes() as u64 * 64, 9_625_927_680);
        let full = sh.layout(272);
        assert!(full.bytes >= sh.tensor_bytes() && full.bytes < sh.tensor_bytes() + 10 * ALIGN + PAGE);
        assert_eq!(full.ffn.gate.rows, 17408);
        assert_eq!(full.ffn.down.cols, 17408);
        let half = sh.layout(136);
        assert_eq!(half.ffn.down.w.len, 5120 * 136 * 32);
        assert!(sh.layout(0).bytes == PAGE);
    }

    #[test]
    fn block_set_ops() {
        let mut s = BlockSet::from_list(130, &[0, 5, 64, 129]);
        assert_eq!(s.count(), 4);
        assert!(s.contains(129) && !s.contains(128) && !s.contains(500));
        s.remove(5);
        assert_eq!(s.list(), vec![0, 64, 129]);
        let u = s.union(&BlockSet::from_list(130, &[1]));
        assert_eq!(u.list(), vec![0, 1, 64, 129]);
        assert!(BlockSet::full(130).is_full());
        let w = BlockSet::from_words(130, &[u64::MAX, 1, u64::MAX]);
        assert_eq!(w.count(), 64 + 1 + 2);
        assert_eq!(BlockSet::from_words(130, u.words()), u);
    }
}
