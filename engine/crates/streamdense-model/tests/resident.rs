//! resident sets (src/resident.rs).
//! - `tiny_*`: the tiny model's real safetensors: full set = the tensors byte for byte, a partial set = exactly the
//!   gathered hot rows / column groups, reload (sidecar and safetensors-only) = bit-identical again.
//! - `synthetic_27b_layer` (ignored; run alone, release): synthetic files shaped like 27B FFN layers on the internal
//!   SSD: (a) phys_footprint drop = unloaded bytes, (b) reload bit-identical, (c) reload GB/s, (d) peak during a
//!   repack ≤ baseline + one layer. `cargo test --release -p streamdense-model --test resident -- --ignored
//!   --nocapture --test-threads=1` (RAM ≤ ~0.5 GB, disk 1.2 GB temp, deleted; dir: SD_RESIDENT_DIR or the temp dir).
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use streamdense_core::footprint::self_footprint;
use streamdense_core::io::NoCacheFile;
use streamdense_core::SafeIndex;
use streamdense_metal::Gpu;
use streamdense_model::resident::{BlockSet, FfnShape, FfnSrc, Q4At, Resident, ResidentLayer, BLOCK};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models");

fn gpu() -> Gpu { Gpu::from_shader_files(&[]).expect("Metal device") }

fn tmp(name: &str) -> String {
    let dir = std::env::var("SD_RESIDENT_DIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().to_string());
    format!("{}/sd_resident_{}_{}", dir, std::process::id(), name)
}

/// deterministic pseudo-random block subset: keeps about `keep` of `n`
fn pick(n: usize, keep: f64, seed: u64) -> BlockSet {
    let mut x = seed | 1;
    let mut s = BlockSet::empty(n);
    for b in 0..n {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        if ((x >> 11) as f64 / (1u64 << 53) as f64) < keep { s.insert(b as u32); }
    }
    s
}

/// Original tensors of one layer (w, s, b of gate, up, down), read straight from the files.
struct Orig { gate: [Vec<u8>; 3], up: [Vec<u8>; 3], down: [Vec<u8>; 3] }
fn read_orig(files: &[NoCacheFile], src: &FfnSrc, sh: FfnShape) -> Orig {
    let rd = |q: &Q4At, rows: usize, cols: usize| -> [Vec<u8>; 3] {
        let mut out = [vec![0u8; rows * cols / 2], vec![0u8; rows * cols / 64 * 2], vec![0u8; rows * cols / 64 * 2]];
        for (v, a) in out.iter_mut().zip([q.w, q.s, q.b]) { files[a.file].read_at(v.as_mut_ptr(), v.len(), a.off).unwrap(); }
        out
    };
    Orig { gate: rd(&src.gate, sh.inter, sh.hidden), up: rd(&src.up, sh.inter, sh.hidden), down: rd(&src.down, sh.hidden, sh.inter) }
}

/// The resident layer must hold exactly the hot blocks of the original tensors, in compact order.
fn check_layer(r: &ResidentLayer, o: &Orig, sh: FfnShape) {
    let hot = r.hot().to_vec();
    assert_eq!(hot, r.set.list());
    let h = hot.len();
    let f = r.ffn_w();
    assert_eq!((f.gate.rows, f.up.rows, f.down.cols, f.down.rows), (h * BLOCK, h * BLOCK, h * BLOCK, sh.hidden));
    let mem = r.as_bytes();
    let reg = |x: streamdense_core::arena::Region| &mem[x.off..x.off + x.len];
    for (m, ov) in [(f.gate, &o.gate), (f.up, &o.up)] {
        for (k, rg) in [m.w, m.s, m.b].into_iter().enumerate() {
            let rb = ov[k].len() / sh.inter; // bytes per original row
            let got = reg(rg);
            assert_eq!(got.len(), h * BLOCK * rb);
            for (i, &b) in hot.iter().enumerate() {
                let (a, z) = (b as usize * BLOCK * rb, (b as usize + 1) * BLOCK * rb);
                assert!(got[i * BLOCK * rb..(i + 1) * BLOCK * rb] == ov[k][a..z], "gate/up part {} block {}", k, b);
            }
        }
    }
    for (k, rg) in [f.down.w, f.down.s, f.down.b].into_iter().enumerate() {
        let gb = if k == 0 { 32 } else { 2 }; // bytes per 64-column group per row
        let (orow, nrow) = (sh.inter / 64 * gb, h * gb);
        let got = reg(rg);
        assert_eq!(got.len(), sh.hidden * nrow);
        for r in 0..sh.hidden {
            for (i, &b) in hot.iter().enumerate() {
                let o0 = r * orow + b as usize * gb;
                assert!(got[r * nrow + i * gb..r * nrow + (i + 1) * gb] == o.down[k][o0..o0 + gb], "down part {} row {} block {}", k, r, b);
            }
        }
    }
}

#[test]
fn tiny_unload_reload_bit_identical() {
    let dir = format!("{}/tiny_qwen35", std::env::var("SD_MODELS").unwrap_or_else(|_| MODELS.to_string()));
    if !std::path::Path::new(&dir).exists() { eprintln!("skip: no {}", dir); return; }
    let idx = SafeIndex::open_mlx_dir(&dir).unwrap();
    let cfg = streamdense_model::QwenConfig::from_model_dir(&dir).unwrap();
    let sh = FfnShape::new(cfg.hidden, cfg.inter).unwrap();
    let pre: Vec<String> = (0..cfg.layers).map(|l| format!("language_model.model.layers.{}.mlp.", l)).collect();
    let side = tmp("tiny.side");
    let g = gpu();
    let mut res = Resident::new(&idx, sh, &pre, Some(&side), 4).unwrap();
    let mut res_st = Resident::new(&idx, sh, &pre, None, 3).unwrap();
    let files: Vec<NoCacheFile> = idx.files().iter().map(|f| NoCacheFile::open(f).unwrap()).collect();
    let n = sh.blocks();
    for l in 0..cfg.layers {
        let src = FfnSrc::from_index(&idx, &pre[l], sh).unwrap();
        let o = read_orig(&files, &src, sh);
        // full load = the safetensors bytes
        let (full, _) = res.load_layer(&g, l, &BlockSet::full(n)).unwrap();
        check_layer(&full, &o, sh);
        // unload to a partial set (blocks leave into the sidecar), then a different partial set (mixed sources)
        let a = pick(n, 0.6, 11 + l as u64);
        let (pa, sa) = res.build(&g, l, Some(&full), &a).unwrap();
        check_layer(&pa, &o, sh);
        assert_eq!(sa.ssd_read(), 0);
        assert_eq!(sa.side_written as usize, (n - a.count()) * sh.hidden * 36);
        drop(full);
        let b = pick(n, 0.5, 97 + l as u64);
        let (pb, _) = res.build(&g, l, Some(&pa), &b).unwrap();
        check_layer(&pb, &o, sh);
        // reload to full: down from the sidecar / from full safetensors rows; both bit-identical
        let (back, s1) = res.build(&g, l, Some(&pb), &BlockSet::full(n)).unwrap();
        check_layer(&back, &o, sh);
        assert!(s1.side_read > 0 && s1.st_down_bytes == 0);
        let (back2, s2) = res_st.build(&g, l, Some(&pb), &BlockSet::full(n)).unwrap();
        check_layer(&back2, &o, sh);
        assert!(s2.side_read == 0 && s2.st_down_bytes > 0);
        assert!(back.as_bytes() == back2.as_bytes());
        // partial first load (load_layer path with the sidecar) and the empty set
        let (pl, _) = res.load_layer(&g, l, &b).unwrap();
        check_layer(&pl, &o, sh);
        let (e, _) = res.build(&g, l, Some(&pl), &BlockSet::empty(n)).unwrap();
        assert_eq!(e.ffn_w().gate.rows, 0);
        let (again, _) = res.build(&g, l, Some(&e), &BlockSet::full(n)).unwrap();
        check_layer(&again, &o, sh);
    }
    drop(res);
    assert!(!std::path::Path::new(&side).exists(), "sidecar not deleted");
}

// ------------------------------------------------------------------------------------------------ synthetic 27B

struct TempFile(String);
impl Drop for TempFile { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }

/// Synthetic layers in one file, written with F_NOCACHE (so the reads below come from the SSD, not the file cache).
fn write_synth(path: &str, bytes: usize) {
    use std::io::Write;
    use std::os::unix::io::AsRawFd;
    let mut f = std::fs::File::create(path).unwrap();
    unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
    let mut x: u64 = 0x9e3779b97f4a7c15;
    let mut buf = vec![0u8; 8 << 20];
    let mut done = 0;
    while done < bytes {
        for c in buf.chunks_exact_mut(8) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; c.copy_from_slice(&x.to_le_bytes()); }
        let n = buf.len().min(bytes - done);
        f.write_all(&buf[..n]).unwrap();
        done += n;
    }
    f.sync_all().unwrap();
}

/// Peak phys_footprint sampler (every ~200 µs).
struct Sampler { peak: Arc<AtomicU64>, stop: Arc<AtomicBool>, h: Option<std::thread::JoinHandle<()>> }
impl Sampler {
    fn start() -> Sampler {
        let (peak, stop) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicBool::new(false)));
        let (p, s) = (peak.clone(), stop.clone());
        let h = std::thread::spawn(move || while !s.load(Ordering::Relaxed) {
            p.fetch_max(self_footprint(), Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_micros(200));
        });
        Sampler { peak, stop, h: Some(h) }
    }
    /// peak since the last reset, then reset to the current footprint
    fn take(&self) -> u64 { let f = self_footprint(); self.peak.swap(f, Ordering::Relaxed).max(f) }
}
impl Drop for Sampler { fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); if let Some(h) = self.h.take() { let _ = h.join(); } } }

fn settle() -> u64 { std::thread::sleep(std::time::Duration::from_millis(150)); self_footprint() }
fn mb(b: f64) -> f64 { b / 1e6 }

#[test]
#[ignore]
fn synthetic_27b_layer() {
    let sh = FfnShape::new(5120, 17408).unwrap();
    let n = sh.blocks();
    let layers = 8;
    let lb = sh.tensor_bytes();
    let data = TempFile(tmp("synth27b.bin"));
    let t = Instant::now();
    write_synth(&data.0, lb * layers);
    eprintln!("synthetic file: {} layers x {} B = {:.2} GB written in {:.2} s", layers, lb, (lb * layers) as f64 / 1e9, t.elapsed().as_secs_f64());
    let src: Vec<FfnSrc> = (0..layers).map(|l| FfnSrc::packed(0, (l * lb) as u64, sh)).collect();
    let files = vec![data.0.clone()];
    let side = tmp("synth27b.side");
    let g = gpu();
    let full_set = BlockSet::full(n);
    let full_bytes = sh.full_bytes() as f64;
    eprintln!("27B layer FFN: tensors {} B, full resident buffer {} B, block {} B", lb, sh.full_bytes(), sh.block_bytes());

    // (c) cold reads: whole layers from the safetensors-like file, thread sweep (each layer read once per setting)
    let rd = [NoCacheFile::open(&data.0).unwrap()];
    for (k, th) in [4usize, 8, 16].into_iter().enumerate() {
        let mut r = Resident::with_sources(&files, src.clone(), sh, None, th).unwrap();
        let ls: Vec<usize> = (0..layers).filter(|l| l % 3 == k).collect();
        let t = Instant::now();
        let mut bytes = 0u64;
        for &l in &ls {
            let (x, s) = r.build(&g, l, None, &full_set).unwrap();
            bytes += s.ssd_read();
            if l == ls[0] && k == 0 { check_layer(&x, &read_orig(&rd, &src[l], sh), sh); }
            objc2::rc::autoreleasepool(|_| drop(x));
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!("(c) full-layer load, {:>2} threads: {} layers, {:.3} GB in {:.3} s = {:.2} GB/s", th, ls.len(), bytes as f64 / 1e9, s, bytes as f64 / 1e9 / s);
    }

    let mut res = Resident::with_sources(&files, src.clone(), sh, Some(&side), 8).unwrap();
    let mut res_st = Resident::with_sources(&files, src.clone(), sh, None, 8).unwrap();
    let samp = Sampler::start();
    let l = 3;
    let o = read_orig(&rd, &src[l], sh);
    let (mut cur, s0) = res.load_layer(&g, l, &full_set).unwrap();
    check_layer(&cur, &o, sh);
    eprintln!("(b) full layer {} loaded in {:.3} s ({:.2} GB/s), bit-identical to the file", l, s0.secs, s0.ssd_read() as f64 / 1e9 / s0.secs);
    let mut worst_over = 0f64;
    for (keep, seed) in [(0.78, 5u64), (0.50, 6), (0.05, 7)] {
        // (a) + (d): repack full → keep, drop the old buffer
        let want = pick(n, keep, seed);
        let f0 = settle();
        samp.take();
        let (next, s) = res.build(&g, l, Some(&cur), &want).unwrap();
        let peak = samp.take();
        objc2::rc::autoreleasepool(|_| drop(std::mem::replace(&mut cur, next)));
        let f1 = settle();
        let expect = full_bytes - cur.bytes() as f64;
        let over = peak as f64 - f0 as f64;
        worst_over = worst_over.max(over);
        eprintln!("(a) unload to {:>3} of {} blocks ({:.0}% kept): footprint {:+.1} MB (expected -{:.1} MB = freed bytes), repack {:.3} s, sidecar +{:.1} MB",
            want.count(), n, 100.0 * want.count() as f64 / n as f64, mb(f1 as f64 - f0 as f64), mb(expect), s.secs, mb(s.side_written as f64));
        eprintln!("(d)   peak during repack = baseline {:+.1} MB (one full layer = {:.1} MB, new buffer {:.1} MB)", mb(over), mb(full_bytes), mb(cur.bytes() as f64));
        check_layer(&cur, &o, sh);
        // (c) prefetch-like: add 8 cold blocks (sidecar vs full safetensors rows)
        let mut add = cur.set.clone();
        let mut k = 0;
        for b in 0..n as u32 { if k < 8 && !add.contains(b) { add.insert(b); k += 1; } }
        let (p1, ps1) = res.build(&g, l, Some(&cur), &add).unwrap();
        let (p2, ps2) = res_st.build(&g, l, Some(&cur), &add).unwrap();
        check_layer(&p1, &o, sh);
        assert!(p1.as_bytes() == p2.as_bytes());
        eprintln!("(c)   +8 blocks: sidecar {:.1} ms (ssd {:.2} MB, ram copy {:.1} MB) | safetensors-only {:.1} ms (ssd {:.2} MB)",
            ps1.secs * 1e3, mb(ps1.ssd_read() as f64), mb(ps1.ram_bytes as f64), ps2.secs * 1e3, mb(ps2.ssd_read() as f64));
        objc2::rc::autoreleasepool(|_| { drop(p1); drop(p2); });
        // (b) + (c) reload everything (fallback), both down sources
        let f2 = settle();
        samp.take();
        let (back, rs) = res.build(&g, l, Some(&cur), &full_set).unwrap();
        let peak2 = samp.take();
        let (back2, rs2) = res_st.build(&g, l, Some(&cur), &full_set).unwrap();
        check_layer(&back, &o, sh);
        assert!(back.as_bytes() == back2.as_bytes());
        worst_over = worst_over.max(peak2 as f64 - f2 as f64);
        eprintln!("(c)   reload all {} cold blocks: sidecar {:.3} s ({:.1} MB ssd = {:.2} GB/s, ram copy {:.1} MB) | safetensors-only {:.3} s ({:.1} MB ssd = {:.2} GB/s); bit-identical",
            n - want.count(), rs.secs, mb(rs.ssd_read() as f64), rs.ssd_read() as f64 / 1e9 / rs.secs, mb(rs.ram_bytes as f64),
            rs2.secs, mb(rs2.ssd_read() as f64), rs2.ssd_read() as f64 / 1e9 / rs2.secs);
        eprintln!("(d)   peak during reload = baseline {:+.1} MB", mb(peak2 as f64 - f2 as f64));
        objc2::rc::autoreleasepool(|_| { drop(back2); drop(std::mem::replace(&mut cur, back)); });
    }
    let sc = res.sidecar().unwrap();
    eprintln!("sidecar on disk: {:.1} MB (sparse; slots written once per block)", mb(sc.disk_bytes() as f64));
    eprintln!("worst peak over baseline: {:.1} MB vs one full layer {:.1} MB", mb(worst_over), mb(full_bytes));
    assert!(worst_over <= full_bytes + 16e6, "peak above baseline + one layer");
}
