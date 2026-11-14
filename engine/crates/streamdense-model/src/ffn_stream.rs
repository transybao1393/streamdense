//! Single-process streamed prefill (forheader, 2026-11-02). OFF by default (`QwenOptions.ffn_stream_tokens = 0`: no
//! region, no kernel, no dispatch). Needs the compact FFN (`QwenOptions.ffn_compact`, ffn_compact.rs).
//!
//! Goal: ONE process at the compact cap both reads the prompt with the FULL model and answers with the compact set
//! (before: a full-model process read the prompt and handed a snapshot to a compact process).
//!
//! `QwenEngine::stream_prefill(ids, policy)` reads the prompt LAYER-MAJOR: the embeddings of all prompt rows go into
//! `hrows` (bf16 [max_tokens][hidden]); then for each main layer l every prompt chunk (the same `chunks()` as the
//! exact prefill: prefill_k rows, then 4 / 2 / 1) runs `encode_layer` with layer l's FULL FFN: gate / up W, S, B and
//! the down W streamed from the ORIGINAL model file into a slot (F_NOCACHE pread, a loader thread, `slots` slots so
//! the read of layer l+1 overlaps the GPU on layer l), the down S / B taken from the compact layer (kept whole there).
//! Same kernels, same rows, same positions as the full engine's exact prefill, so KV / DeltaNet state / the first token
//! are bit-identical to the two-process path (full prefill + snapshot restore). `cs_setpos` puts ctrl.pos back to the
//! prompt start at each layer; the head runs after the last layer's last chunk exactly as `encode_pass_full`.
//! In the same pass, per chunk: `cs_acc` adds the layer's prompt activity (an = Σ|a| per pass as the activity recorder;
//! ag = Σ|silu(gate x)| per row as the ffn-select hook summed token by token; the gate x comes from one extra gate GEMV).
//! When layer l's command buffer is done the CPU calls `policy(l, an, ag)` -> the layer's new kept set (exactly the
//! arena's kc neurons, sorted), fills the compact layer from the slot (`ffn_compact::fill_from_full`: no second read)
//! and frees the slot for layer l + slots. After the prompt, decode runs on the compact set as today.
//! RAM (budget rows, in the compact keep formula): slots x full-layer FFN (2 x (W + S + B) + down W; 144.8 MB on the
//! 27B), hrows, the gate scratch bf16 [tmax][inter] and per slot 2 x inter f32 accumulators. With
//! `QwenOptions.ffn_stream_host` they live inside the idle lm_head instead (re-read before the head; see `place`) and
//! leave the budget: 27B 12 GB MTP off K 10,416 / 10,144 (1 / 2 slots) -> 10,728 = the no-stream K.
//! SSD: the whole FFN of every layer once per prompt (64 x 144.8 MB = 9.27 GB on the 27B), nothing written.
//!
//! FAST MODE (`QwenOptions.ffn_stream_fast`, 2026-11-03, OFF by default): the same layer-major loop, slots, loader,
//! host mode, pieces and policy, but each layer runs the full engine's FAST prefill on the fast chunks (`chunks_ex`:
//! up to prefill_t rows per GEMM pass, a tail of <= 8 rows in K <= 2 exact passes): `encode_layer_prefill_g` (GEMM
//! projections / FFN, prefill mixers, flash attention) whose swiglu GEMM also writes the gate output bf(gate x) into
//! gscr, so ag needs no extra gate pass on those chunks (the <= 8-row tail keeps the exact pass + gate GEMV). Same
//! kernels, rows and positions as a full engine's fast prefill, so it equals that path (KV / state / first token), not
//! the exact one (GEMM / flash summation order). an / ag use the same `cs_acc` (an: one add per pass; ag: per row).
//! Pieces: bit-identical to one shot when every piece but the last is a multiple of prefill_t (the GEMM chunks line
//! up); other cuts change the chunking, so only close. Scratch grows to prefill_t rows (budget, via tmax).
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use streamdense_core::arena::{Arena, Region};
use streamdense_core::io::NoCacheFile;
use streamdense_metal::{cb_signal, cb_wait, commit, ext_off, gpu_ms, page_size, wait, Buffer, CmdBuf, Enc};

use crate::ffn_compact::{CJob, CompactSet};
use crate::qwen_engine::{QwenEngine, ATTN_CHUNK};
use crate::qwen_layers::{encode_head, encode_layer, encode_layer_prefill_g, gemv_m, PassIo, EXACT_KS};
use crate::weights::{FfnW, LayerW, Q4b};

/// One full-layer FFN slot (arena regions) + its accumulators f32 [2][inter] (an | ag).
#[derive(Clone, Copy, Debug)]
pub struct Slot { pub gate: [Region; 3], pub up: [Region; 3], pub down_w: Region, pub acc: Region }

/// Arena regions of the streamed prefill. `host`: the lm_head ranges the hosted buffers overwrite (empty unless
/// `QwenOptions.ffn_stream_host`), re-read from the model file before the head.
#[derive(Clone, Debug)]
pub struct StreamRegions { pub slots: Vec<Slot>, pub hrows: Region, pub gscr: Region, pub max_tokens: usize, pub host: Vec<HostReload> }

/// One lm_head tensor range (arena `off`, `len`) that hosted stream buffers overwrite, and its source in the model files.
#[derive(Clone, Copy, Debug)]
pub struct HostReload { pub off: usize, pub len: usize, pub file: usize, pub src: u64 }

// ---- idle-memory hosting (2026-11-02, `QwenOptions.ffn_stream_host`, OFF by default). The lm_head (W + S + B, 715 MB
// on the 27B) is read ONLY by the head, which runs once after the last layer of the streamed prefill; every other
// stream buffer is dead by then. So the slots, hrows, gscr and accumulators are placed INSIDE the lm_head tensors
// (first fit, largest first, 16 KB aligned); only the pieces that do not fit get their own arena region and budget row.
// After the last layer (and on every error path) the overwritten lm_head prefix of each tensor is re-read from the
// model file (F_NOCACHE), then the head runs in its own command buffer on a copy of the last chunk's rows: same
// kernels, same bytes, so the first token / state / set are unchanged. Cost: SSD bytes = the hosted bytes per prompt
// (2 slots on the 27B: ~311 MB), no RAM.

/// Bytes of every stream buffer in alloc order: per slot gate W, S, B, up W, S, B, down W, acc; then hrows, gscr.
pub fn pieces(hidden: usize, inter: usize, tmax: usize, max_tokens: usize, slots: usize) -> Vec<usize> {
    let (wb, sb) = (inter * hidden / 2, inter * hidden / 64 * 2);
    let mut v = Vec::with_capacity(slots * 8 + 2);
    for _ in 0..slots { v.extend_from_slice(&[wb, sb, sb, wb, sb, sb, hidden * inter / 2, 2 * inter * 4]); }
    v.extend_from_slice(&[max_tokens * hidden * 2, tmax * inter * 2]);
    v
}
/// The host pool: lm_head W / S / B bytes (4-bit, group 64).
pub fn host_pool(vocab: usize, hidden: usize) -> [usize; 3] { [vocab * hidden / 2, vocab * hidden / 64 * 2, vocab * hidden / 64 * 2] }
/// First fit, largest piece first: piece i -> Some((pool index, byte offset, 16 KB aligned)) or None (own region).
pub fn place(pieces: &[usize], pool: &[usize]) -> Vec<Option<(usize, usize)>> {
    let al = streamdense_core::arena::ALIGN;
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by(|&a, &b| pieces[b].cmp(&pieces[a]).then(a.cmp(&b)));
    let mut used = vec![0usize; pool.len()];
    let mut out = vec![None; pieces.len()];
    for i in order {
        for (p, &cap) in pool.iter().enumerate() {
            let o = used[p].div_ceil(al) * al;
            if o + pieces[i] <= cap { out[i] = Some((p, o)); used[p] = o + pieces[i]; break; }
        }
    }
    out
}
/// `budget_rows` with an optional host pool: only the pieces left outside it are counted (the keep formula then gives
/// the hosted bytes back to K).
pub fn budget_rows_ex(hidden: usize, inter: usize, tmax: usize, max_tokens: usize, slots: usize, pool: Option<[usize; 3]>) -> Vec<(String, u64)> {
    let Some(pool) = pool else { return budget_rows(hidden, inter, tmax, max_tokens, slots) };
    let pc = pieces(hidden, inter, tmax, max_tokens, slots);
    let pl = place(&pc, &pool);
    let (mut w, mut r, mut hosted) = (0u64, 0u64, 0u64);
    for (i, (&b, p)) in pc.iter().zip(&pl).enumerate() {
        if p.is_some() { hosted += b as u64; } else if i < slots * 8 && i % 8 < 7 { w += b as u64; } else { r += b as u64; }
    }
    vec![
        (format!("streamed prefill: {} full-layer FFN slot(s), part outside the idle lm_head ({:.1} MB of all buffers hosted there)",
                 slots, hosted as f64 / 1e6), w),
        (format!("streamed prefill: hidden rows ({} tok) + gate scratch + accumulators, part outside the lm_head", max_tokens), r),
    ]
}
/// `regions` with an optional host pool (hosted pieces take no arena region).
pub fn regions_ex(hidden: usize, inter: usize, tmax: usize, max_tokens: usize, slots: usize, pool: Option<[usize; 3]>) -> usize {
    match pool {
        None => regions(slots),
        Some(p) => place(&pieces(hidden, inter, tmax, max_tokens, slots), &p).iter().filter(|x| x.is_none()).count(),
    }
}
/// lm_head regions + sources for hosting (`alloc_ex`).
pub(crate) struct HostPool { pub regions: [Region; 3], pub src: [(usize, u64); 3] }

/// Bytes of one slot's weights: gate + up (W + S + B, all inter rows) + down W (the down S / B stay in the compact layer).
pub fn slot_bytes(hidden: usize, inter: usize) -> usize {
    2 * (inter * hidden / 2 + 2 * (inter * hidden / 64 * 2)) + hidden * inter / 2
}
/// Budget rows (name, bytes). Names must not start with "FFN" (the keep formula takes those as the compact FFN row).
pub fn budget_rows(hidden: usize, inter: usize, tmax: usize, max_tokens: usize, slots: usize) -> Vec<(String, u64)> {
    vec![
        (format!("streamed prefill: {} full-layer FFN slot(s) ({:.1} MB each)", slots, slot_bytes(hidden, inter) as f64 / 1e6),
         (slots * slot_bytes(hidden, inter)) as u64),
        (format!("streamed prefill: hidden rows ({} tok) + gate scratch + accumulators", max_tokens),
         (max_tokens * hidden * 2 + tmax * inter * 2 + slots * 2 * inter * 4) as u64),
    ]
}
/// Arena regions it adds (for the alignment slack row).
pub fn regions(slots: usize) -> usize { slots * 8 + 2 }

/// `alloc` with an optional host pool (the lm_head): pieces placed by `place` live inside it, the rest get arena regions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn alloc_ex(arena: &mut Arena, hidden: usize, inter: usize, tmax: usize, max_tokens: usize, slots: usize, vocab: usize,
                       host: Option<HostPool>) -> Result<StreamRegions, String> {
    let Some(hp) = host else { return alloc(arena, hidden, inter, tmax, max_tokens, slots) };
    let pool = host_pool(vocab, hidden);
    for (r, &n) in hp.regions.iter().zip(&pool) {
        if r.len != n { return Err(format!("ffn_stream_host: lm_head region {} B, expected {} B", r.len, n)); }
    }
    let pc = pieces(hidden, inter, tmax, max_tokens, slots);
    let pl = place(&pc, &pool);
    let mut end = [0usize; 3];
    let mut rs = Vec::with_capacity(pc.len());
    for (i, (&b, p)) in pc.iter().zip(&pl).enumerate() {
        rs.push(match p {
            Some((pi, o)) => { end[*pi] = end[*pi].max(o + b); Region { off: hp.regions[*pi].off + o, len: b } }
            None => arena.alloc(&format!("s:stream.piece{}", i), b)?,
        });
    }
    let v = (0..slots).map(|s| {
        let r = &rs[s * 8..s * 8 + 8];
        Slot { gate: [r[0], r[1], r[2]], up: [r[3], r[4], r[5]], down_w: r[6], acc: r[7] }
    }).collect();
    let host = (0..3).filter(|&p| end[p] > 0)
        .map(|p| HostReload { off: hp.regions[p].off, len: end[p], file: hp.src[p].0, src: hp.src[p].1 }).collect();
    Ok(StreamRegions { slots: v, hrows: rs[slots * 8], gscr: rs[slots * 8 + 1], max_tokens, host })
}

pub(crate) fn alloc(arena: &mut Arena, hidden: usize, inter: usize, tmax: usize, max_tokens: usize, slots: usize) -> Result<StreamRegions, String> {
    let (wb, sb) = (inter * hidden / 2, inter * hidden / 64 * 2);
    let mut v = Vec::with_capacity(slots);
    for i in 0..slots {
        let mut q = |n: &str| -> Result<[Region; 3], String> {
            Ok([arena.alloc(&format!("s:stream{}.{}.w", i, n), wb)?, arena.alloc(&format!("s:stream{}.{}.s", i, n), sb)?,
                arena.alloc(&format!("s:stream{}.{}.b", i, n), sb)?])
        };
        let gate = q("gate")?;
        let up = q("up")?;
        let down_w = arena.alloc(&format!("s:stream{}.down.w", i), hidden * inter / 2)?;
        let acc = arena.alloc(&format!("s:stream{}.acc", i), 2 * inter * 4)?;
        v.push(Slot { gate, up, down_w, acc });
    }
    let hrows = arena.alloc("s:stream.hrows", max_tokens * hidden * 2)?;
    let gscr = arena.alloc("s:stream.gscr", tmax * inter * 2)?;
    Ok(StreamRegions { slots: v, hrows, gscr, max_tokens, host: Vec::new() })
}

/// Re-read the hosted lm_head ranges from the model files (bytes read).
fn host_reload(files: &[String], host: &[HostReload], base: usize, maps: Option<&[(usize, usize)]>) -> Result<u64, String> {
    if let Some(m) = maps {
        // SD_RECHECK_MMAP: copy from the mappings
        let mut n = 0;
        for h in host {
            if h.src as usize + h.len > m[h.file].1 { return Err("ffn_stream: lm_head range past the end of the model file".into()); }
            let (s, d) = (m[h.file].0 + h.src as usize, base + h.off);
            let per = h.len.div_ceil(threads()).max(1);
            std::thread::scope(|sc| {
                for o in (0..h.len).step_by(per) {
                    let k = per.min(h.len - o);
                    sc.spawn(move || unsafe { std::ptr::copy_nonoverlapping((s + o) as *const u8, (d + o) as *mut u8, k) });
                }
            });
            n += h.len as u64;
        }
        return Ok(n);
    }
    let mut n = 0;
    for (fi, f) in files.iter().enumerate() {
        let js: Vec<(usize, usize, u64)> = host.iter().filter(|h| h.file == fi).map(|h| (base + h.off, h.len, h.src)).collect();
        if !js.is_empty() { n += NoCacheFile::open(f)?.read_many(&js, threads(), 4 << 20)?; }
    }
    Ok(n)
}

/// Restores the lm_head on every exit of a hosted `stream_prefill` (error paths included); `finish` on success.
struct HostGuard<'a> { files: &'a [String], host: &'a [HostReload], base: usize, done: bool, maps: Option<&'a [(usize, usize)]> }
impl HostGuard<'_> {
    fn finish(&mut self) -> Result<u64, String> { self.done = true; host_reload(self.files, self.host, self.base, self.maps) }
}
impl Drop for HostGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            if let Err(e) = host_reload(self.files, self.host, self.base, self.maps) { eprintln!("ffn_stream: lm_head restore FAILED ({}): reopen the engine", e); }
        }
    }
}

pub fn pipelines() -> Vec<&'static str> { vec!["cs_setpos", "cs_acc", "cs_rowscatter", "cs_bytecopy"] }

/// Per-prompt numbers of `stream_prefill`.
#[derive(Clone, Debug, Default)]
pub struct StreamStats {
    /// greedy next token (ctrl.tok[0], as `prefill`)
    pub tok: u32,
    pub tokens: usize,
    /// wall time of the whole call (= time to first token)
    pub wall_s: f64,
    pub embed_s: f64,
    /// bytes the loader read from the model file, and its busy time (sum over layers)
    pub ssd_bytes: u64,
    pub ssd_busy_s: f64,
    /// main thread blocked on the loader / on a layer's command buffer
    pub load_wait_s: f64,
    pub gpu_wait_s: f64,
    /// sum of the layer command buffers' GPU time
    pub gpu_s: f64,
    pub policy_s: f64,
    pub fill_s: f64,
    /// neurons per layer that changed against the set loaded before
    pub changed: usize,
    /// `ffn_stream_host`: lm_head bytes re-read after the last layer, and that time
    pub host_reload_bytes: u64,
    pub host_reload_s: f64,
}

fn load_layer(fds: &[NoCacheFile], j: &CJob, sl: &Slot, base: usize, threads: usize) -> Result<u64, String> {
    let mut per: Vec<Vec<(usize, usize, u64)>> = vec![Vec::new(); fds.len()];
    for (src, dst) in j.gate.iter().zip(&sl.gate).chain(j.up.iter().zip(&sl.up)).chain(std::iter::once((&j.down_w, &sl.down_w))) {
        per[src.file].push((base + dst.off, dst.len, src.off));
    }
    let mut n = 0;
    for (fi, js) in per.iter().enumerate() {
        if !js.is_empty() { n += fds[fi].read_many(js, threads, 4 << 20)?; }
    }
    Ok(n)
}

/// Missing-only load (gate_recheck.rs `missing` mode): the slot gets the layer's FULL FFN bytes exactly as
/// `load_layer` writes them, but the gate / up W rows of the resident set `keep` are copied from the compact layer in RAM
/// (the same bytes: `gather` / `fill_from_full` copy file rows unchanged) and only the other (missing) rows are read
/// from the file, as runs of missing rows merged across kept gaps of <= `gap` rows (`SD_RECHECK_GAP`, fewer preads).
/// gate / up S, B (2.8 MB each on the 27B) and the down W are read whole: a missing neuron's down column is one nibble in
/// every row. Slot bytes identical to `load_layer`, so the pass is bit-identical to the full-read recheck. Returns the
/// bytes read from the file.
/// `gpu_idx`: Some(arena offset) = the resident rows are NOT copied here; `keep` is written there as u32 [kc] and
/// the layer's command buffer copies the rows itself (`cs_rowscatter`, encode_stream_layer) before its FFN: same slot
/// bytes, but the copy moves from the CPU loader (the critical path) to the GPU stream (idle half the time).
#[allow(clippy::too_many_arguments)]
fn load_layer_missing(fds: &[NoCacheFile], maps: Option<&[(usize, usize)]>, j: &CJob, sl: &Slot, base: usize, threads: usize, keep: &[u32],
                      hidden: usize, inter: usize, gap: usize, gpu_idx: Option<usize>) -> Result<u64, String> {
    let rb = hidden / 2;
    let mut per: Vec<Vec<(usize, usize, u64)>> = vec![Vec::new(); fds.len()];
    for (src, dst) in j.gate[1..].iter().zip(&sl.gate[1..]).chain(j.up[1..].iter().zip(&sl.up[1..])).chain(std::iter::once((&j.down_w, &sl.down_w))) {
        per[src.file].push((base + dst.off, dst.len, src.off));
    }
    // runs [r0, r1) of missing rows, merged over kept gaps <= gap
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let (mut r, mut ki) = (0usize, 0usize);
    while r < inter {
        while ki < keep.len() && (keep[ki] as usize) < r { ki += 1; }
        if ki < keep.len() && keep[ki] as usize == r { r += 1; continue; }
        let r1 = if ki < keep.len() { keep[ki] as usize } else { inter };
        match runs.last_mut() {
            Some(last) if r - last.1 <= gap => last.1 = r1,
            _ => runs.push((r, r1)),
        }
        r = r1;
    }
    for (src, dst) in [(&j.gate[0], &sl.gate[0]), (&j.up[0], &sl.up[0])] {
        for &(r0, r1) in &runs { per[src.file].push((base + dst.off + r0 * rb, (r1 - r0) * rb, src.off + (r0 * rb) as u64)); }
    }
    let mut n = 0;
    match maps {
        None => for (fi, js) in per.iter().enumerate() {
            if !js.is_empty() { n += fds[fi].read_many(js, threads, 4 << 20)?; }
        },
        Some(m) => {
            // memcpy from the read-only mappings, 1 MB pieces over `threads` threads (no syscalls: the scattered row
            // runs cost no more than their bytes)
            let mut pc: Vec<(usize, usize, usize)> = Vec::new();
            for (fi, js) in per.iter().enumerate() {
                for &(d, len, off) in js {
                    if off as usize + len > m[fi].1 { return Err(format!("ffn_stream: read past the end of model file {}", fi)); }
                    let mut x = 0;
                    while x < len { let k = (1 << 20).min(len - x); pc.push((d + x, m[fi].0 + off as usize + x, k)); x += k; }
                    n += len as u64;
                }
            }
            let per_t = pc.len().div_ceil(threads.max(1)).max(1);
            std::thread::scope(|sc| {
                for part in pc.chunks(per_t) {
                    sc.spawn(move || for &(d, s, k) in part { unsafe { std::ptr::copy_nonoverlapping(s as *const u8, d as *mut u8, k); } });
                }
            });
        }
    }
    if let Some(io) = gpu_idx {
        // the GPU copies every kept row (rows a merged run already wrote get the same bytes again)
        unsafe { std::ptr::copy_nonoverlapping(keep.as_ptr() as *const u8, (base + io) as *mut u8, keep.len() * 4); }
        return Ok(n);
    }
    // kept rows not covered by a merged run: compact row i -> slot row keep[i]
    let mut todo: Vec<(usize, usize)> = Vec::with_capacity(keep.len());
    let mut ri = 0;
    for (i, &x) in keep.iter().enumerate() {
        let x = x as usize;
        while ri < runs.len() && runs[ri].1 <= x { ri += 1; }
        if ri < runs.len() && runs[ri].0 <= x { continue; }
        todo.push((i, x));
    }
    let per_t = todo.len().div_ceil(threads.max(1)).max(1);
    let pairs = [(j.ffn.gate.w.off, sl.gate[0].off), (j.ffn.up.w.off, sl.up[0].off)];
    std::thread::scope(|sc| {
        for part in todo.chunks(per_t) {
            sc.spawn(move || {
                for &(s, d) in &pairs {
                    for &(i, x) in part {
                        unsafe { std::ptr::copy_nonoverlapping((base + s + i * rb) as *const u8, (base + d + x * rb) as *mut u8, rb); }
                    }
                }
            });
        }
    });
    Ok(n)
}

/// `SD_RECHECK_GAP`: kept rows a missing-only load may read over to merge two missing runs (default 16 with pread,
/// 0 with the mmap copy).
fn recheck_gap(mmap: bool) -> usize {
    std::env::var("SD_RECHECK_GAP").ok().and_then(|s| s.parse().ok()).unwrap_or(if mmap { 0 } else { 16 })
}

/// The missing-only recheck's copy source (DEFAULT ON, `SD_RECHECK_MMAP=0` turns it off): read-only
/// MAP_SHARED mappings of the model files, copied with memcpy instead of F_NOCACHE pread (only this path maps the
/// model). Mapped once per process and kept (address, length); clean file pages are not in the process footprint, but
/// pages faulted in from SSD stay in the OS file cache (unlike F_NOCACHE). If a file cannot be mapped the recheck falls
/// back to pread (None, said once on stderr).
fn recheck_maps(files: &[String]) -> Option<Vec<(usize, usize)>> {
    if std::env::var("SD_RECHECK_MMAP").ok().as_deref() == Some("0") { return None; }
    map_files(files)
}

/// The process-wide read-only mappings of `files` (mapped on first use, never unmapped), or None (said once on stderr).
fn map_files(files: &[String]) -> Option<Vec<(usize, usize)>> {
    static MAPS: std::sync::Mutex<Vec<(String, usize, usize)>> = std::sync::Mutex::new(Vec::new());
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let mut g = MAPS.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        if let Some(x) = g.iter().find(|x| &x.0 == f) { out.push((x.1, x.2)); continue; }
        let r = (|| -> Result<(usize, usize), String> {
            let fd = std::fs::File::open(f).map_err(|e| format!("{}: {}", f, e))?;
            let len = fd.metadata().map_err(|e| format!("{}: {}", f, e))?.len() as usize;
            let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, std::os::fd::AsRawFd::as_raw_fd(&fd), 0) };
            if p == libc::MAP_FAILED { return Err(format!("{}: mmap failed ({})", f, std::io::Error::last_os_error())); }
            Ok((p as usize, len))
        })();
        match r {
            Ok((p, len)) => { g.push((f.clone(), p, len)); out.push((p, len)); }
            Err(e) => {
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) { eprintln!("ffn_stream: recheck mmap off, using pread ({})", e); }
                return None;
            }
        }
    }
    Some(out)
}

/// Arena offsets (one per slot) for the keep indices of the GPU resident-row copy of a missing-only load, placed at
/// the END of hrows (a recheck uses only its first `used` bytes: 1 row): u32 [max kc], 16-byte aligned. None (the CPU
/// copies the rows) when `SD_RECHECK_GPU_ROWS=0`, a row / region is not 16-byte aligned or the indices do not fit.
/// No new memory.
fn gpu_rows_idx(jobs: &[CJob], keep: &[Vec<u32>], slots: &[Slot], hrows: Region, used: usize, hidden: usize) -> Option<Vec<usize>> {
    if std::env::var("SD_RECHECK_GPU_ROWS").ok().as_deref() == Some("0") { return None; }
    let a16 = |x: usize| x % 16 == 0;
    if !a16(hidden / 2) || slots.iter().any(|s| !a16(s.gate[0].off) || !a16(s.up[0].off))
        || jobs.iter().any(|j| !a16(j.ffn.gate.w.off) || !a16(j.ffn.up.w.off)) { return None; }
    let ib = keep.iter().map(|k| k.len()).max().unwrap_or(0).div_ceil(4) * 16;
    let end = (hrows.off + hrows.len) / 16 * 16;
    if ib == 0 || hrows.off + used + slots.len() * ib > end { return None; }
    Some((0..slots.len()).map(|s| end - (s + 1) * ib).collect())
}

/// Zero-copy recheck (DEFAULT ON for the missing-only recheck, `SD_RECHECK_ZEROCOPY=0` = the copy path
/// above): one no-copy Metal buffer (`Gpu::buffer_no_copy`) per FFN tensor of every main layer (gate W / S / B, up W / S / B,
/// down W: 7 x layers), each wrapping the page-aligned span of the tensor in the process-wide read-only mapping of its
/// model file (`map_files`). The CPU loader does nothing. In the recheck's full-FFN pass a tensor at a 16-byte aligned
/// file offset is bound directly (`Enc::with_ext`, region at an `ext_off` offset); a misaligned one (the kernels need
/// >= 4-byte aligned weights; the 27B's files 2 / 3 put tensors at odd / 2 mod 4 offsets) is first copied by the GPU into
/// the layer's slot (`cs_bytecopy`, in the layer's command buffer). Same bytes, same kernels, so bit-identical to the
/// copy path. The down S / B still come from the compact layer. File pages stay clean, file-backed (OS file cache).
///
/// Residency set (`SD_RECHECK_RESIDENT=1`, default OFF): the buffers are also put in one MTLResidencySet (`Gpu::residency_set`,
/// requestResidency, attached to the queue) when built, so the per-commit residency work (~110-190 ms between commit
/// and GPU start per recheck) is done once (27B: recheck 295 -> 164 ms). Cost: the set WIRES the 9.3 GB of mapped file
/// pages for the process lifetime (vm_stat wired +9.2 GB; phys_footprint unchanged), so it stays opt-in. Declared first:
/// ends before the buffers drop.
pub struct ZeroCopy { _res: Option<streamdense_metal::ResidencySet>, bufs: Vec<Buffer>, layers: Vec<[ZcTensor; 7]> }

/// One FFN tensor in the no-copy buffers: `at` = its first byte (an `ext_off` offset), `shift` = its byte offset inside
/// buffer `buf` (bound at offset 0 for `cs_bytecopy`), `direct` = 16-byte aligned (bound as is).
#[derive(Clone, Copy, Debug, Default)]
struct ZcTensor { at: Region, buf: usize, shift: usize, direct: bool }

/// Build the no-copy buffers for `jobs` (tensor lengths = a slot's region lengths). Err = any file / buffer failure (the
/// caller falls back to the copy path). `SD_RECHECK_ZC_COPYALL=1` (test only) copies every tensor with `cs_bytecopy`.
fn zero_copy_build(gpu: &streamdense_metal::Gpu, files: &[String], jobs: &[CJob], sl: &Slot) -> Result<ZeroCopy, String> {
    let maps = map_files(files).ok_or("model files could not be mapped")?;
    let pg = page_size();
    let copy_all = std::env::var("SD_RECHECK_ZC_COPYALL").ok().as_deref() == Some("1");
    let lens = [sl.gate[0].len, sl.gate[1].len, sl.gate[2].len, sl.up[0].len, sl.up[1].len, sl.up[2].len, sl.down_w.len];
    if lens.iter().any(|&n| n % 16 != 0) { return Err(format!("tensor lengths {:?} not multiples of 16 B", lens)); }
    let (mut bufs, mut layers) = (Vec::new(), Vec::with_capacity(jobs.len()));
    for j in jobs {
        let srcs = [j.gate[0], j.gate[1], j.gate[2], j.up[0], j.up[1], j.up[2], j.down_w];
        let mut ts = [ZcTensor::default(); 7];
        for (i, (src, &len)) in srcs.iter().zip(&lens).enumerate() {
            let (base, flen) = maps[src.file];
            let off = src.off as usize;
            if off + len > flen { return Err(format!("layer {} tensor {} past the end of model file {}", j.layer, i, src.file)); }
            let (a, b) = (off / pg * pg, (off + len).div_ceil(pg) * pg); // b <= the mapping's page-rounded end
            let buf = unsafe { gpu.buffer_no_copy((base + a) as *mut u8, b - a)? };
            ts[i] = ZcTensor { at: Region { off: ext_off(bufs.len(), off - a), len }, buf: bufs.len(), shift: off - a,
                               direct: off % 16 == 0 && !copy_all };
            bufs.push(buf);
        }
        layers.push(ts);
    }
    let res = if std::env::var("SD_RECHECK_RESIDENT").ok().as_deref() != Some("1") { None } else {
        let t = std::time::Instant::now();
        let r = gpu.residency_set(&bufs)?;
        eprintln!("ffn_stream: recheck residency set: {} buffers, {:.2} GB, {:.1} ms", bufs.len(), r.allocated_size() as f64 / 1e9,
                  t.elapsed().as_secs_f64() * 1e3);
        Some(r)
    };
    Ok(ZeroCopy { _res: res, bufs, layers })
}

/// Loader threads (`SD_STREAM_THREADS`, default 8).
fn threads() -> usize { std::env::var("SD_STREAM_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8) }

impl QwenEngine {
    /// The zero-copy buffers for the missing-only recheck, built once (None: `SD_RECHECK_ZEROCOPY=0`, or building failed:
    /// said once on stderr, the copy path is used for the rest of the process).
    fn recheck_zero_copy(&self, c: &crate::ffn_compact::CompactState, st: &StreamRegions) -> Option<&ZeroCopy> {
        if std::env::var("SD_RECHECK_ZEROCOPY").ok().as_deref() == Some("0") { return None; }
        self.zc.get_or_init(|| match zero_copy_build(&self.gpu, &c.files, &c.jobs, &st.slots[0]) {
            Ok(z) => Some(z),
            Err(e) => { eprintln!("ffn_stream: recheck zero-copy off, using the copy path ({})", e); None }
        }).as_ref()
    }

    /// Streamed prefill is available (`QwenOptions.ffn_stream_tokens` > 0).
    pub fn stream_prefill_on(&self) -> bool { self.cst.is_some() }

    /// Read `ids` from the current position with the FULL model, layer by layer (module doc), and rebuild every compact
    /// layer from `policy(layer, an, ag)` (an / ag: f32 [inter], the layer's prompt activity, see `cs_acc`), which
    /// must return exactly that layer's kept count of sorted unique neuron ids. On return the engine is where `prefill`
    /// leaves it (pos += ids.len(), the next token in ctrl.tok[0] and in the stats) with the new set loaded
    /// (`ffn_compact_set`). On error the compact contents are partly rebuilt (`ffn_compact_set` says which set each
    /// layer holds) and the sequence state is undefined: `reset()`. Engine idle only (nothing in flight).
    pub fn stream_prefill(&self, ids: &[u32], policy: &mut dyn FnMut(usize, &[f32], &[f32]) -> Result<Vec<u32>, String>)
                          -> Result<StreamStats, String> {
        self.stream_prefill_seeded(ids, None, policy)
    }

    /// `stream_prefill` for a prompt that arrives in pieces ("read while the user types"): call it once per piece,
    /// each continuing the sequence (KV / DeltaNet state carry over as for `prefill`). `seed[l]` (f32 [2 * inter], an | ag)
    /// starts layer l's accumulators instead of 0, so passing the an / ag the previous piece's policy saw makes the sums
    /// the policy sees after the last piece bit-identical to one call on the whole prompt (same per-chunk adds in the same
    /// order, when every piece but the last is a multiple of `prefill_k` tokens so the chunks line up). A policy that
    /// returns an EMPTY set keeps that layer's current set and skips its fill (read-only pass: build the set once at the end).
    pub fn stream_prefill_seeded(&self, ids: &[u32], seed: Option<&[Vec<f32>]>,
                                 policy: &mut dyn FnMut(usize, &[f32], &[f32]) -> Result<Vec<u32>, String>) -> Result<StreamStats, String> {
        self.stream_pass(ids, seed, policy, 0, false)
    }

    /// `stream_prefill_seeded` with layers `0..full_from` run on the RESIDENT compact set (no read, no slot, no policy
    /// call) and only layers `full_from..` streamed with their full FFN (gate_recheck.rs "last" mode). `full_from = 0`
    /// is `stream_prefill_seeded` exactly. `missing`: each streamed layer's slot is filled by `load_layer_missing`
    /// (resident rows copied from RAM, only the missing rows + S / B + down W read): same slot bytes, fewer read.
    pub(crate) fn stream_pass(&self, ids: &[u32], seed: Option<&[Vec<f32>]>,
                              policy: &mut dyn FnMut(usize, &[f32], &[f32]) -> Result<Vec<u32>, String>, full_from: usize,
                              missing: bool) -> Result<StreamStats, String> {
        let st = self.cst.as_ref().ok_or("ffn_stream: off (QwenOptions.ffn_stream_tokens)")?;
        let c = self.cfc.as_ref().ok_or("ffn_stream: needs ffn_compact")?;
        let n = ids.len();
        if n == 0 || n > st.max_tokens { return Err(format!("stream_prefill: {} tokens (1..={})", n, st.max_tokens)); }
        let p0 = self.pos();
        if let Some(sd) = seed {
            if sd.len() != self.cfg.layers || sd.iter().any(|v| v.len() != 2 * self.cfg.inter) {
                return Err(format!("stream_prefill: seed must be {} x [2 x {}]", self.cfg.layers, self.cfg.inter));
            }
        }
        if p0 + n > self.opt.max_ctx { return Err(format!("stream_prefill: {} + {} tokens > max_ctx {}", p0, n, self.opt.max_ctx)); }
        let cfg = &self.cfg;
        let (hidden, inter, nl, ns) = (cfg.hidden, cfg.inter, cfg.layers, st.slots.len());
        let ff = full_from;
        if ff >= nl { return Err(format!("stream_pass: full_from {} >= {} layers", ff, nl)); }
        let hb = hidden * 2;
        let ch = self.chunks_ex(n, self.opt.ffn_stream_fast);
        let mut s = StreamStats { tokens: n, ..Default::default() };
        let t0 = Instant::now();
        // hosted buffers (ffn_stream_host): from here on the lm_head may be overwritten; the guard restores it on any exit
        let zc = if missing { self.recheck_zero_copy(c, st) } else { None };
        let maps = if missing { recheck_maps(&c.files) } else { None };
        let mut hg = (!st.host.is_empty()).then(|| HostGuard { files: &c.files, host: &st.host, base: self.arena.base() as usize, done: false,
                                                               maps: maps.as_deref() });

        // ---- embeddings of every prompt row -> hrows (2 staging slots, as prefill)
        let mut infl: VecDeque<CmdBuf> = VecDeque::new();
        for (ci, &(i, k)) in ch.iter().enumerate() {
            let slot = ci % 2;
            if infl.len() >= 2 { wait(&infl.pop_front().unwrap())?; }
            self.reader.read_rows(&ids[i..i + k], self.stage_ptr(slot))?;
            let cb = self.gpu.command_buffer();
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            e.pso("embed_q4").b(0, self.sl().stage.at(slot * self.slot_bytes)).b(1, st.hrows.off + i * hb).u(2, k as u32);
            e.threads((hidden / 8, k), 64);
            e.end();
            commit(&cb);
            infl.push_back(cb);
        }
        for cb in &infl { wait(cb)?; }
        drop(infl);
        s.embed_s = t0.elapsed().as_secs_f64();

        // ---- layers
        let fds: Vec<NoCacheFile> = c.files.iter().map(|f| NoCacheFile::open(f)).collect::<Result<_, _>>()?;
        let base = self.arena.base() as usize;
        let old = c.set.borrow().clone();
        let mut keeps: Vec<Vec<u32>> = old.keep.clone();
        let (free_tx, free_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<Result<(usize, u64, f64), String>>();
        let nthr = threads();
        let gap = recheck_gap(maps.is_some());
        let zc_on = zc.is_some();
        let gidx = if missing && !zc_on { gpu_rows_idx(&c.jobs[ff..], &old.keep[ff..], &st.slots, st.hrows, n * hb, hidden) } else { None };
        let res = std::thread::scope(|sc| -> Result<(), String> {
            let free_tx = free_tx; // dropped when this closure returns, so the loader never waits forever
            let (jobs, slots, fds, okeep, maps, gidx) = (&c.jobs, &st.slots, &fds, &old.keep, maps.as_deref(), gidx.as_deref());
            sc.spawn(move || {
                for l in ff..nl {
                    if l - ff >= ns && free_rx.recv().is_err() { return; }
                    let t = Instant::now();
                    let sl = &slots[(l - ff) % ns];
                    let r = if zc_on { Ok(0) } // zero-copy: the GPU reads the mapped file, nothing to load
                            else if missing { load_layer_missing(fds, maps, &jobs[l], sl, base, nthr, &okeep[l], hidden, inter, gap,
                                                               gidx.map(|g| g[(l - ff) % ns])) }
                            else { load_layer(fds, &jobs[l], sl, base, nthr) };
                    let r = r.map(|b| (l, b, t.elapsed().as_secs_f64()));
                    let fail = r.is_err();
                    if done_tx.send(r).is_err() || fail { return; }
                }
            });
            let order = self.gpu.shared_event();
            let mut inflight: VecDeque<(usize, CmdBuf)> = VecDeque::new();
            let mut consume = |l: usize, cb: CmdBuf, s: &mut StreamStats, keeps: &mut Vec<Vec<u32>>| -> Result<(), String> {
                let tw = Instant::now();
                wait(&cb)?;
                s.gpu_wait_s += tw.elapsed().as_secs_f64();
                s.gpu_s += gpu_ms(&cb) / 1e3;
                let sl = &slots[(l - ff) % ns];
                let acc = self.arena.slice_mut::<f32>(sl.acc);
                let tp = Instant::now();
                let keep = policy(l, &acc[..inter], &acc[inter..2 * inter])?;
                s.policy_s += tp.elapsed().as_secs_f64();
                if keep.is_empty() { let _ = free_tx.send(()); return Ok(()); } // read-only pass: keep the set, no fill
                let kc = jobs[l].ffn.cd.map_or(0, |d| d.kc);
                if keep.len() != kc || keep.windows(2).any(|w| w[0] >= w[1]) || keep.last().is_some_and(|&x| x as usize >= inter) {
                    return Err(format!("stream_prefill: policy gave layer {} {} neurons (want {} sorted unique < {})", l, keep.len(), kc, inter));
                }
                let tf = Instant::now();
                let at3 = |r: &[Region; 3]| [r[0].off, r[1].off, r[2].off];
                crate::ffn_compact::fill_from_full(&jobs[l], &keep, hidden, inter, self.arena.base(), at3(&sl.gate), at3(&sl.up), sl.down_w.off, nthr)?;
                s.fill_s += tf.elapsed().as_secs_f64();
                let prev = &old.keep[l];
                let (mut a, mut b, mut same) = (0, 0, 0);
                while a < prev.len() && b < keep.len() {
                    if prev[a] == keep[b] { same += 1; a += 1; b += 1; } else if prev[a] < keep[b] { a += 1; } else { b += 1; }
                }
                s.changed += kc - same;
                keeps[l] = keep;
                let _ = free_tx.send(());
                Ok(())
            };
            let mut head_cb: Option<CmdBuf> = None;
            let r = (|| -> Result<(), String> {
                if ff > 0 {
                    // layers 0..ff on the resident compact set, one command buffer (signals ff: layer ff waits on it)
                    let cb = self.gpu.command_buffer();
                    self.encode_compact_layers(&cb, ff, st, &ch, p0)?;
                    cb_signal(&cb, &order, ff as u64);
                    commit(&cb);
                    head_cb = Some(cb);
                }
                for l in ff..nl {
                    let tl = Instant::now();
                    let (l2, bytes, secs) = done_rx.recv().map_err(|_| "stream_prefill: loader stopped".to_string())??;
                    if l2 != l { return Err(format!("stream_prefill: loader sent layer {} for {}", l2, l)); }
                    s.load_wait_s += tl.elapsed().as_secs_f64();
                    s.ssd_bytes += bytes;
                    s.ssd_busy_s += secs;
                    let sl = &slots[(l - ff) % ns];
                    match seed {
                        Some(sd) => self.arena.slice_mut::<f32>(sl.acc)[..2 * inter].copy_from_slice(&sd[l]),
                        None => self.arena.slice_mut::<f32>(sl.acc).fill(0.0),
                    }
                    let cb = self.gpu.command_buffer();
                    if l > ff || ff > 0 { cb_wait(&cb, &order, l as u64); }
                    self.encode_stream_layer(&cb, l, sl, st, &ch, p0, gidx.map(|g| (g[(l - ff) % ns], okeep[l].len())), zc)?;
                    cb_signal(&cb, &order, l as u64 + 1);
                    commit(&cb);
                    inflight.push_back((l, cb));
                    while inflight.len() >= ns {
                        let (j, cb) = inflight.pop_front().unwrap();
                        consume(j, cb, &mut s, &mut keeps)?;
                    }
                }
                while let Some((j, cb)) = inflight.pop_front() { consume(j, cb, &mut s, &mut keeps)?; }
                Ok(())
            })();
            for (_, cb) in &inflight { let _ = wait(cb); }
            if let Some(cb) = &head_cb { let w = wait(cb); if r.is_ok() { w?; } }
            r
        });
        *c.set.borrow_mut() = std::sync::Arc::new(CompactSet { inter, keep: keeps });
        res?;
        if let Some(g) = hg.as_mut() {
            // the head was left out of the last layer's command buffer: copy the last chunk's rows out of hrows (it may be
            // hosted), restore the lm_head, then run the head on the copy (same rows, same kernels as encode_stream_layer)
            let &(i, k) = ch.last().unwrap();
            let io = PassIo { k, pos: p0 + i, nch_max: (p0 + n).div_ceil(ATTN_CHUNK).min(self.nch_max), ..self.io() };
            unsafe { std::ptr::copy_nonoverlapping(self.arena.base().add(st.hrows.off + i * hb), self.arena.base().add(io.h), k * hb); }
            let tr = Instant::now();
            s.host_reload_bytes = g.finish()?;
            s.host_reload_s = tr.elapsed().as_secs_f64();
            let cb = self.gpu.command_buffer();
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            let r = encode_head(&e, &self.cfg, &self.w, &io, 1, None);
            if r.is_ok() { e.pso("advance1").b(0, io.ctrl); e.threads((1, 1), 1); }
            e.end();
            r?;
            commit(&cb);
            wait(&cb)?;
        }
        self.set_pos(p0 + n);
        s.tok = self.ctrl_read().tok[0];
        s.wall_s = t0.elapsed().as_secs_f64();
        Ok(s)
    }

    /// Layers `0..nl` over every chunk on the resident compact set (the normal exact pass layers; rows in hrows).
    fn encode_compact_layers(&self, cb: &CmdBuf, nl: usize, st: &StreamRegions, ch: &[(usize, usize)], p0: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let e = Enc::new(&self.gpu, cb, &self.buf);
        let r = (|| -> Result<(), String> {
            let ctrl = self.io().ctrl;
            for l in 0..nl {
                e.pso("cs_setpos").b(0, ctrl).u(1, p0 as u32);
                e.threads((1, 1), 1);
                for &(i, k) in ch {
                    if !EXACT_KS.contains(&k) { return Err(format!("stream_pass: compact layers need exact chunks (k {})", k)); }
                    let pos = p0 + i;
                    let io = PassIo { k, pos, nch_max: (pos + k).div_ceil(ATTN_CHUNK).min(self.nch_max), h: st.hrows.off + i * cfg.hidden * 2, rm: None,
                                      ..self.io() };
                    encode_layer(&e, cfg, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, false, None, l)?;
                    for _ in 0..k { e.pso("advance1").b(0, ctrl); e.threads((1, 1), 1); }
                }
            }
            Ok(())
        })();
        e.end();
        r
    }

    /// One layer over every prompt chunk with the slot's full FFN, + the activity accumulators; the head after the
    /// last layer's last chunk. Chunks of > 8 rows (fast mode only) run the fast prefill layer with the gate output
    /// written to gscr by the swiglu GEMM itself.
    /// `scatter` = Some((keep index offset, kc)) (missing-only recheck): first copy the resident gate / up W rows of the
    /// compact layer into the slot rows keep[i] (`cs_rowscatter`; the loader left them out).
    /// `zc` (zero-copy recheck): the FFN gate / up / down W come from the no-copy file buffers, not the slot.
    #[allow(clippy::too_many_arguments)]
    fn encode_stream_layer(&self, cb: &CmdBuf, l: usize, sl: &Slot, st: &StreamRegions, ch: &[(usize, usize)], p0: usize,
                           scatter: Option<(usize, usize)>, zc: Option<&ZeroCopy>) -> Result<(), String> {
        let cfg = &self.cfg;
        let (hidden, inter) = (cfg.hidden, cfg.inter);
        let lw0 = &self.w.layers[l];
        if lw0.ffn.cd.is_none() { return Err(format!("stream_prefill: layer {} is not compact", l)); }
        let q = |r: &[Region]| Q4b { rows: inter, cols: hidden, w: r[0], s: r[1], b: r[2] };
        let slot = [sl.gate[0], sl.gate[1], sl.gate[2], sl.up[0], sl.up[1], sl.up[2], sl.down_w];
        // zero-copy: aligned tensors straight from the file buffers, the rest re-aligned into the slot (cs_bytecopy below)
        let src: [Region; 7] = match zc {
            Some(z) => std::array::from_fn(|i| { let t = &z.layers[l][i]; if t.direct { t.at } else { slot[i] } }),
            None => slot,
        };
        let ffn = FfnW { gate: q(&src[0..3]), up: q(&src[3..6]), down: Q4b { rows: hidden, cols: inter, w: src[6], s: lw0.ffn.down.s, b: lw0.ffn.down.b }, cd: None };
        let lw = LayerW { ffn, ..*lw0 };
        let e = Enc::with_ext(&self.gpu, cb, &self.buf, zc.map_or(&[], |z| &z.bufs[..]));
        let r = (|| -> Result<(), String> {
            if let Some(z) = zc {
                for (t, d) in z.layers[l].iter().zip(&slot) {
                    if t.direct { continue; }
                    let n16 = t.at.len / 16;
                    e.pso("cs_bytecopy").b(0, ext_off(t.buf, 0)).b(1, d.off).u(2, t.shift as u32).u(3, n16 as u32);
                    e.threads((n16, 1), 256);
                }
            }
            if let Some((ix, kc)) = scatter {
                let rw = hidden / 32; // uint4 per 4-bit row (hidden / 2 bytes)
                for (src, dst) in [(lw0.ffn.gate.w.off, sl.gate[0].off), (lw0.ffn.up.w.off, sl.up[0].off)] {
                    if kc > 0 {
                        e.pso("cs_rowscatter").b(0, src).b(1, dst).b(2, ix).u(3, rw as u32).u(4, kc as u32);
                        e.threads((rw, kc), 32);
                    }
                }
            }
            let ctrl = self.io().ctrl;
            e.pso("cs_setpos").b(0, ctrl).u(1, p0 as u32);
            e.threads((1, 1), 1);
            let adv = |n: usize| for _ in 0..n { e.pso("advance1").b(0, ctrl); e.threads((1, 1), 1); };
            for (ci, &(i, k)) in ch.iter().enumerate() {
                let pos = p0 + i;
                let io = PassIo { k, pos, nch_max: (pos + k).div_ceil(ATTN_CHUNK).min(self.nch_max), h: st.hrows.off + i * hidden * 2, rm: None,
                                  ..self.io() };
                if EXACT_KS.contains(&k) {
                    encode_layer(&e, cfg, &lw, Some(&self.dn()), self.kv()[l].as_ref(), &io, false, None, l)?;
                    // prompt activity: io.x still holds the FFN input, io.f the activations (the down only reads f)
                    gemv_m(&e, io.gemv, &lw.ffn.gate, k, 0, io.x, st.gscr.off, None);
                } else {
                    // fast chunk (ffn_stream_fast; chunks_ex gives > 8 rows only then): gate output from the swiglu GEMM
                    encode_layer_prefill_g(&e, cfg, &lw, Some(&self.dn()), self.kv()[l].as_ref(), &io, l, Some(st.gscr.off))?;
                }
                e.pso("cs_acc").b(0, io.f).b(1, st.gscr.off).b(2, sl.acc.off).b(3, sl.acc.off + inter * 4).u(4, inter as u32).u(5, k as u32)
                    .groups((inter.div_ceil(256), 1), 256);
                if l + 1 == cfg.layers && ci + 1 == ch.len() {
                    adv(k - 1);
                    if st.host.is_empty() { // hosted: the lm_head is restored first, the head runs in its own command buffer
                        encode_head(&e, cfg, &self.w, &io, 1, None)?;
                        adv(1);
                    }
                } else { adv(k); }
            }
            Ok(())
        })();
        e.end();
        r
    }
}
