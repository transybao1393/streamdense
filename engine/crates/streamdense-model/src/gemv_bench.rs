//! `streamdense bench gemv-k`: synthetic microbench of the exact K-row 4-bit GEMV (q4b_gemv / q4b_swiglu,
//! gemv_q4_bf16.metal) at the real 27B projection shapes from `config.json` (random weights, no model load).
//! For each distinct shape, K and D (row-dots per simdgroup): GPU µs per dispatch (min over `--reps` command
//! buffers), weight GB/s, t / t(K=1 default), and a bit-identity check of every row of the K-row output against K
//! separate K = 1 (D = 2, the decode kernel) dispatches on the same x rows (m3 f32 bits and m0 bf16; swiglu bf16).
//! Then the full-pass GEMV estimate per (K, D) = sum over shapes of (layers using it) x t, its ratio to K = 1, and the
//! `SD_GEMV_D` string of the fastest D per K. Small matrices are rotated over copies (>= `--rot-mb` MB of weights)
//! so they stream from DRAM like the real pass instead of sitting in the SLC.
//! `--impl mma|both` (default old) also times gemv_mma_q4.metal: the MMA kernels qm_gemv / qm_swiglu per (K, NB, SK)
//! and the scalar twins qm_gemvt / qm_swiglut per (K, SK); checks (per SK) every K = 1..8 row of both families, every
//! NB, and K = 8 with neighbour rows replaced, against the per-row MMA K = 1 output (bit-identical required); prints
//! MMA-only and twin-only pass estimates, the engine plan (one SK per shape, faster family per K) and closeness of
//! the new arithmetic vs today's K = 1 q4b_gemv (bit-equal fraction, max ulp, max |diff| / rms).
//! Run: streamdense bench gemv-k [--model DIR] [--ks 1,2,3,4,5,8] [--ds 2,4,8] [--reps 5] [--rot-mb 256] [--no-head]
//!      [--impl old|mma|both] [--nbs 1,2,4] [--sks 1,2,4,8] [--tmax 512] [--compile-only]
//!      [--head-keep FILE]: instead, the MTP draft head full vs pruned (keep-list of `streamdense vocab keep`): lm_head
//!      K = 1 + draft argmax per family (old / mma), GPU ms, kept-row logits / argmax bit-identical to the full head.
//! RAM: one shape at a time; peak arena ~ lm_head (vocab x hidden x 0.5625 B, ~0.72 GB) + 17 MB, or ~0.3 GB with
//! --no-head.
use streamdense_core::arena::Region;
use streamdense_metal::{commit, contents, gpu_ms, wait, Enc, Gpu};

use crate::qwen_config::QwenConfig;
use crate::qwen_layers::{
    gemv_d, gemv_ds, gemv_enc, gemv_kernel, mma_gemv_enc, mma_gemv_kernel, mma_swiglu_enc, mma_swiglu_kernel, swiglu_enc,
    swiglu_kernel, GEMV_KS, MMA_K_MAX, MMA_NBS, mma_sks, mma_twin_gemv_enc, mma_twin_gemv_kernel,
    mma_twin_swiglu_enc, mma_twin_swiglu_kernel, mma_twin_nt, MMA_TWIN_KS, MMA_TWIN_SKS,
};
use crate::weights::Q4b;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 } // [0, 1)
    fn n(&mut self) -> f32 { (0..4).map(|_| self.f()).sum::<f32>() * 1.732 - 3.464 } // ~N(0, 1)
}
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }

struct Arena { buf: streamdense_metal::Buffer, ptr: *mut u8, top: usize, len: usize }
impl Arena {
    fn alloc(&mut self, n: usize) -> Result<usize, String> {
        let o = self.top;
        self.top = (self.top + n + 255) & !255;
        if self.top > self.len { return Err(format!("bench arena too small ({} > {})", self.top, self.len)); }
        Ok(o)
    }
    #[allow(clippy::mut_from_ref)] // bench arena: disjoint offsets into one shared GPU buffer
    fn sl<T>(&self, off: usize, n: usize) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut T, n) } }
}

fn reg(off: usize, len: usize) -> Region { Region { off, len } }
fn q4_bytes(r: usize, c: usize) -> usize { r * c / 2 + 2 * (r * c / 64 * 2) }

fn put_mat(a: &mut Arena, rng: &mut Rng, r: usize, c: usize) -> Result<Q4b, String> {
    let (wl, sl) = (r * c / 2, r * c / 64 * 2);
    let (w, s, b) = (a.alloc(wl)?, a.alloc(sl)?, a.alloc(sl)?);
    for v in a.sl::<u32>(w, r * c / 8) { *v = rng.next() as u32; }
    let (ss, bb) = (a.sl::<u16>(s, r * c / 64), a.sl::<u16>(b, r * c / 64));
    for i in 0..r * c / 64 {
        let sc = 0.002 + 0.018 * rng.f();
        ss[i] = to_bf(sc);
        bb[i] = to_bf(-8.0 * sc + 0.01 * rng.n());
    }
    Ok(Q4b { rows: r, cols: c, w: reg(w, wl), s: reg(s, sl), b: reg(b, sl) })
}

fn run(g: &Gpu, a: &Arena, f: &dyn Fn(&Enc)) -> Result<(), String> {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &a.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb)
}
/// min over `reps` command buffers of (gpu ms / n) for n encodes of f(enc, i).
fn time(g: &Gpu, a: &Arena, reps: usize, n: usize, f: &dyn Fn(&Enc, usize)) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let cb = g.command_buffer();
        let e = Enc::new(g, &cb, &a.buf);
        for i in 0..n { f(&e, i); }
        e.end();
        commit(&cb);
        wait(&cb)?;
        best = best.min(gpu_ms(&cb) / n as f64);
    }
    Ok(best)
}

/// One distinct projection shape: rows x cols, how many times a pass runs it, the engine's mode (None = swiglu pair).
struct Shape { name: String, r: usize, c: usize, count: usize, mode: Option<u32> }

/// MLX-style ordered int of an f32 / bf16 bit pattern (ulp distance = |a - b| of these).
fn ord32(u: u32) -> i64 { if u & 0x8000_0000 != 0 { -((u & 0x7fff_ffff) as i64) } else { u as i64 } }
fn bf16f(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }

/// Closeness of `tst` vs `refv` (n elements of ob = 4 (f32) or 2 (bf16) bytes): bit-equal fraction, max ulp
/// (in the element type), max |diff| / rms(ref), max |diff|.
fn close(refv: &[u8], tst: &[u8], ob: usize) -> String {
    let val = |b: &[u8]| -> (f32, i64) {
        if ob == 4 { let u = u32::from_le_bytes([b[0], b[1], b[2], b[3]]); (f32::from_bits(u), ord32(u)) }
        else { let h = u16::from_le_bytes([b[0], b[1]]); (bf16f(h), ord32((h as u32) << 16) >> 16) }
    };
    let (mut neq, mut n, mut mulp, mut mabs, mut ss) = (0usize, 0usize, 0i64, 0f64, 0f64);
    for (p, q) in refv.chunks_exact(ob).zip(tst.chunks_exact(ob)) {
        let ((a, ia), (b, ib)) = (val(p), val(q));
        n += 1;
        ss += (a as f64) * (a as f64);
        if p == q { neq += 1; continue; }
        mulp = mulp.max((ia - ib).abs());
        mabs = mabs.max((a as f64 - b as f64).abs());
    }
    let rms = (ss / n.max(1) as f64).sqrt();
    format!("{:.3}% bit-equal, max {} ulp ({}), max |d| {:.3e} = {:.2e} x rms", 100.0 * neq as f64 / n.max(1) as f64, mulp,
            if ob == 4 { "f32" } else { "bf16" }, mabs, mabs / rms.max(1e-30))
}

/// Scalar twin dispatch, k rows (bit-identical to the MMA kernel with the same SK).
#[allow(clippy::too_many_arguments)]
fn tenc(e: &Enc, k: usize, sk: usize, tmax: usize, mode: Option<u32>, m0: &Q4b, m1: &Q4b, x: usize, out: usize) {
    let nt = mma_twin_nt(m0.cols, mode.is_none(), k, sk, tmax);
    match mode {
        Some(m) => mma_twin_gemv_enc(e, &mma_twin_gemv_kernel(k, sk, m), nt, m0, x, out),
        None => mma_twin_swiglu_enc(e, &mma_twin_swiglu_kernel(k, sk), nt, m0, m1, x, out),
    }
}

/// MMA dispatch of one shape: Some(mode) = qm_gemv, None = qm_swiglu (m0 gate, m1 up).
#[allow(clippy::too_many_arguments)]
fn menc(e: &Enc, nb: usize, sk: usize, mode: Option<u32>, m0: &Q4b, m1: &Q4b, k: usize, x: usize, out: usize) {
    match mode {
        Some(m) => mma_gemv_enc(e, &mma_gemv_kernel(nb, sk, m), nb, sk, m0, k, x, out),
        None => mma_swiglu_enc(e, &mma_swiglu_kernel(nb, sk), nb, sk, m0, m1, k, x, out),
    }
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let list = |k: &str, d: &[usize]| -> Result<Vec<usize>, String> {
        match arg(k) {
            Some(v) => v.split(',').map(|x| x.trim().parse().map_err(|_| format!("{} {}: want a,b,c", k, v))).collect(),
            None => Ok(d.to_vec()),
        }
    };
    let model = arg("--model").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4").into());
    let imp = arg("--impl").unwrap_or_else(|| "old".into());
    let (do_old, do_mma) = match imp.as_str() {
        "old" => (true, false), "mma" => (false, true), "both" => (true, true),
        _ => return Err(format!("--impl {}: want old|mma|both", imp)),
    };
    let ks = list("--ks", &GEMV_KS)?;
    let ds_filter = list("--ds", &[2, 4, 8])?;
    let nbs = list("--nbs", &MMA_NBS)?;
    let sks = list("--sks", &[1, 2, 4, 8])?;
    let tmax: usize = arg("--tmax").map_or(Ok(512), |v| v.parse().map_err(|_| "--tmax".to_string()))?;
    let reps: usize = arg("--reps").map_or(Ok(5), |v| v.parse().map_err(|_| "--reps".to_string()))?;
    let rot = arg("--rot-mb").map_or(Ok(256.0), |v| v.parse::<f64>().map_err(|_| "--rot-mb".to_string()))? * 1e6;
    let head = !args.iter().any(|a| a == "--no-head");
    let compile_only = args.iter().any(|a| a == "--compile-only");
    if let Some(p) = arg("--head-keep") { return head_keep_bench(&model, &p, reps.max(10)); }
    if let Some(f) = arg("--mask-frac") {
        let fr: Vec<f64> = f.split(',').map(|x| x.trim().parse::<f64>().map_err(|_| format!("--mask-frac {}: want e.g. 0,10,25,50 (percent)", f))).collect::<Result<_, _>>()?;
        let mks = if arg("--ks").is_some() { ks.clone() } else { vec![1, 8] };
        return mask_bench(&model, &mks, &fr, reps, rot);
    }
    if do_old {
        if let Some(k) = ks.iter().find(|k| !GEMV_KS.contains(k)) { return Err(format!("--ks: no kernels for k {} (have {:?})", k, GEMV_KS)); }
    }
    if let Some(k) = ks.iter().find(|k| **k == 0 || **k > MMA_K_MAX) { return Err(format!("--ks: k {} not in 1..=8", k)); }
    if let Some(nb) = nbs.iter().find(|n| !MMA_NBS.contains(n)) { return Err(format!("--nbs: no qm kernels for nb {} (have {:?})", nb, MMA_NBS)); }
    let cfgs: Vec<(usize, usize)> = nbs.iter().flat_map(|&nb| sks.iter().filter(move |s| mma_sks(nb).contains(s)).map(move |&s| (nb, s))).collect();
    if cfgs.is_empty() { return Err("--nbs / --sks: no compiled (nb, sk) pair".into()); }

    let t0 = std::time::Instant::now();
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16", "gemv_mma_q4"])?;
    let mut nk = 0;
    // old K = 1 (D = 2) is the reference of both modes
    let old_ks: Vec<usize> = if compile_only { GEMV_KS.to_vec() } else if do_old { ks.clone() } else { vec![1] };
    for &k in &old_ks {
        for &d in gemv_ds(k) {
            for m in [0, 2, 3] { g.pipeline(&gemv_kernel(k, d, m))?; }
            g.pipeline(&swiglu_kernel(k, d))?;
            nk += 4;
        }
    }
    if do_mma || compile_only {
        for nb in MMA_NBS {
            for &sk in mma_sks(nb) {
                for m in [0, 2, 3] { g.pipeline(&mma_gemv_kernel(nb, sk, m))?; }
                g.pipeline(&mma_swiglu_kernel(nb, sk))?;
                nk += 4;
            }
        }
        for k in MMA_TWIN_KS {
            for sk in [1, 2, 4, 8] {
                for m in [0, 2, 3] { g.pipeline(&mma_twin_gemv_kernel(k, sk, m))?; }
                g.pipeline(&mma_twin_swiglu_kernel(k, sk))?;
                nk += 4;
            }
        }
    }
    eprintln!("compiled gemv_q4_bf16 + gemv_mma_q4, {} pipelines in {:.2} s on {}", nk, t0.elapsed().as_secs_f64(), g.name());
    if compile_only { return Ok(()); }

    let cfg = QwenConfig::from_model_dir(&model)?;
    let (h, nd, na, nl) = (cfg.hidden, cfg.n_deltanet(), cfg.n_attention(), cfg.layers);
    let mut shapes: Vec<Shape> = Vec::new();
    let mut add = |name: &str, r: usize, c: usize, count: usize, mode: Option<u32>| {
        if count == 0 { return; }
        // same matrix shape + kernel = same timing: merge (DeltaNet out_proj and attention o_proj are both h x 6144)
        match shapes.iter_mut().find(|s| s.r == r && s.c == c && s.mode == mode) {
            Some(s) => { s.name = format!("{}|{}", s.name, name); s.count += count; }
            None => shapes.push(Shape { name: name.into(), r, c, count, mode }),
        }
    };
    add("dn_in", cfg.dn_in_rows(), h, nd, Some(0));
    add("dn_out", h, cfg.dn_v_dim(), nd, Some(2));
    add("attn_in", cfg.attn_in_rows(), h, na, Some(0));
    add("o_proj", h, cfg.q_dim(), na, Some(2));
    add("gate+up", cfg.inter, h, nl, None);
    add("down", h, cfg.inter, nl, Some(2));
    if head { add("lm_head", cfg.vocab, h, 1, Some(3)); }

    // arena: the largest shape's rotation set + 3 x tiles (8 rows) + 4 outputs (8 rows f32)
    let kmax = MMA_K_MAX;
    let copies = |s: &Shape| -> usize {
        let b = q4_bytes(s.r, s.c) * if s.mode.is_none() { 2 } else { 1 };
        ((rot / b as f64).ceil() as usize).max(1)
    };
    let wmax = shapes.iter().map(|s| copies(s) * (q4_bytes(s.r, s.c) + 1024) * if s.mode.is_none() { 2 } else { 1 }).max().unwrap();
    let (maxr, maxc) = (shapes.iter().map(|s| s.r).max().unwrap(), shapes.iter().map(|s| s.c).max().unwrap());
    let len = wmax + 3 * (kmax * maxc * 2 + 256) + 5 * (kmax * maxr * 4 + 256) + (4 << 20);
    eprintln!("bench arena {:.3} GB (largest shape x rotation copies + x + outputs)", len as f64 / 1e9);
    let buf = g.buffer(len)?;
    let mut a = Arena { ptr: contents(&buf), buf, top: 0, len };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let x = a.alloc(kmax * maxc * 2)?;
    for v in a.sl::<u16>(x, kmax * maxc) { *v = to_bf(rng.n()); }
    // neighbour-content variants (row tiles [8][c] per shape are built from these): x2 = odd rows replaced by large
    // values, x3 = even rows replaced; the kept rows must give the same bits
    let (x2, x3) = (a.alloc(kmax * maxc * 2)?, a.alloc(kmax * maxc * 2)?);
    let (oref, otst) = (a.alloc(kmax * maxr * 4)?, a.alloc(kmax * maxr * 4)?);
    let (oldref, mref, mtst) = (a.alloc(kmax * maxr * 4)?, a.alloc(kmax * maxr * 4)?, a.alloc(kmax * maxr * 4)?);
    let wbase = a.top;

    // results[(shape, k, d)] = ms (old); mres[(shape, k, nb, sk)] = ms (mma)
    let mut res: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut mres: Vec<(usize, usize, usize, usize, f64)> = Vec::new();
    let mut all_ok = true;
    let mut mma_ok = true;
    let mut close_lines: Vec<String> = Vec::new();
    let mut mrows: Vec<String> = Vec::new();
    if do_old {
        println!("| shape | R x C | per pass | K | D | µs | GB/s | t/t1 | bit-identical to K=1 |");
        println!("|---|---|---|---|---|---|---|---|---|");
    }
    for (si, s) in shapes.iter().enumerate() {
        a.top = wbase;
        let nc = copies(s);
        eprintln!("[{}/{}] {} {} x {}: filling {} cop{} ({:.0} MB)...", si + 1, shapes.len(), s.name, s.r, s.c, nc,
                  if nc == 1 { "y" } else { "ies" }, (nc * q4_bytes(s.r, s.c) * if s.mode.is_none() { 2 } else { 1 }) as f64 / 1e6);
        let mut mats: Vec<(Q4b, Q4b)> = Vec::with_capacity(nc);
        for _ in 0..nc {
            let m0 = put_mat(&mut a, &mut rng, s.r, s.c)?;
            let m1 = if s.mode.is_none() { put_mat(&mut a, &mut rng, s.r, s.c)? } else { m0 };
            mats.push((m0, m1));
        }
        let wb = q4_bytes(s.r, s.c) * if s.mode.is_none() { 2 } else { 1 };
        let (r, c) = (s.r, s.c);
        let n = if nc == 1 { 3 } else { 2 * nc };
        // reference: K = 1, D = 2 per row (the decode kernels), checked modes 3 then 0 (swiglu: bf16 out)
        let check_modes: &[Option<u32>] = if s.mode.is_none() { &[None] } else { &[Some(3), Some(0)] };
        if do_old {
            let mut ident: std::collections::HashMap<(usize, usize), String> = Default::default();
            for &cm in check_modes {
                let ob = if cm == Some(3) { 4 } else { 2 };
                let (m0, m1) = mats[0];
                let kref = *ks.iter().max().unwrap();
                run(&g, &a, &|e| for t in 0..kref {
                    let (xo, oo) = (x + t * c * 2, oref + t * r * ob);
                    match cm { Some(m) => gemv_enc(e, &gemv_kernel(1, 2, m), 2, &m0, xo, oo),
                               None => swiglu_enc(e, &swiglu_kernel(1, 2), 2, &m0, &m1, xo, oo) }
                })?;
                for &k in &ks {
                    for &d in gemv_ds(k).iter().filter(|d| ds_filter.contains(d)) {
                        a.sl::<u8>(otst, k * r * ob).fill(0xa5);
                        run(&g, &a, &|e| match cm { Some(m) => gemv_enc(e, &gemv_kernel(k, d, m), d, &m0, x, otst),
                                                    None => swiglu_enc(e, &swiglu_kernel(k, d), d, &m0, &m1, x, otst) })?;
                        let (p, q) = (a.sl::<u8>(oref, k * r * ob), a.sl::<u8>(otst, k * r * ob));
                        let nbad = p.chunks_exact(ob).zip(q.chunks_exact(ob)).filter(|(u, v)| u != v).count();
                        let e = ident.entry((k, d)).or_insert_with(|| "yes".into());
                        if nbad > 0 {
                            all_ok = false;
                            let tag = match cm { Some(m) => format!("m{}", m), None => "swiglu".into() };
                            *e = if e == "yes" { format!("NO: {} {} / {} differ", tag, nbad, k * r) } else { format!("{}; {} {} differ", e, tag, nbad) };
                        }
                    }
                }
            }
            // timing in the engine's mode (m2 adds into otst in place: fine for timing)
            let mut t1 = None;
            for &k in &ks {
                for &d in gemv_ds(k).iter().filter(|d| ds_filter.contains(d)) {
                    let ms = time(&g, &a, reps, n, &|e, i| {
                        let (m0, m1) = mats[i % nc];
                        match s.mode { Some(m) => gemv_enc(e, &gemv_kernel(k, d, m), d, &m0, x, otst),
                                       None => swiglu_enc(e, &swiglu_kernel(k, d), d, &m0, &m1, x, otst) }
                    })?;
                    if k == 1 && d == gemv_d(1) { t1 = Some(ms); }
                    let rel = t1.map_or("-".to_string(), |b| format!("{:.2}", ms / b));
                    println!("| {} | {} x {} | {} | {} | {} | {:.0} | {:.0} | {} | {} |", s.name, r, c, s.count, k, d, ms * 1e3,
                             wb as f64 / ms / 1e6, rel, ident.get(&(k, d)).map_or("-", |v| v.as_str()));
                    res.push((si, k, d, ms));
                }
            }
        }
        if do_mma {
            // x tiles for this shape: x2 / x3 = x with odd / even rows replaced (x is [8][maxc], rows use c columns)
            {
                let (xs, x2s, x3s) = (a.sl::<u16>(x, kmax * maxc), a.sl::<u16>(x2, kmax * maxc), a.sl::<u16>(x3, kmax * maxc));
                x2s.copy_from_slice(xs);
                x3s.copy_from_slice(xs);
                for t in 0..kmax {
                    let dst = if t % 2 == 1 { &mut *x2s } else { &mut *x3s };
                    for v in &mut dst[t * c..(t + 1) * c] { *v = to_bf(300.0 * rng.n()); }
                }
            }
            let (m0, m1) = mats[0];
            let mut bad: Vec<String> = Vec::new();
            for &cm in check_modes {
                let ob = if cm == Some(3) { 4 } else { 2 };
                let tag = match cm { Some(m) => format!("m{}", m), None => "swiglu".into() };
                // today's decode output (old K = 1, D = 2) per row
                run(&g, &a, &|e| for t in 0..kmax {
                    let (xo, oo) = (x + t * c * 2, oldref + t * r * ob);
                    match cm { Some(m) => gemv_enc(e, &gemv_kernel(1, 2, m), 2, &m0, xo, oo),
                               None => swiglu_enc(e, &swiglu_kernel(1, 2), 2, &m0, &m1, xo, oo) }
                })?;
                // per SK (part of the result): reference = first NB with that SK, K = 1 per row (x row t alone)
                for &sk in &sks {
                    let snbs: Vec<usize> = cfgs.iter().filter(|q| q.1 == sk).map(|q| q.0).collect();
                    let Some(&nbr) = snbs.first() else { continue };
                    run(&g, &a, &|e| for t in 0..kmax { menc(e, nbr, sk, cm, &m0, &m1, 1, x + t * c * 2, mref + t * r * ob) })?;
                    close_lines.push(format!("| {} | {} | {}/{} | {} |", s.name, tag, nbr, sk,
                                             close(a.sl::<u8>(oldref, kmax * r * ob), a.sl::<u8>(mref, kmax * r * ob), ob)));
                    let mut cmp = |label: String, rows: &[usize]| {
                        let (p, q) = (a.sl::<u8>(mref, kmax * r * ob), a.sl::<u8>(mtst, kmax * r * ob));
                        let nbad: usize = rows.iter().map(|&t| p[t * r * ob..(t + 1) * r * ob].chunks_exact(ob)
                            .zip(q[t * r * ob..(t + 1) * r * ob].chunks_exact(ob)).filter(|(u, v)| u != v).count()).sum();
                        if nbad > 0 { bad.push(format!("{} {}: {} / {} differ", tag, label, nbad, rows.len() * r)); }
                    };
                    for &nb in &snbs {
                        // (i) every K: K-row output == per-row K = 1 output
                        for k in 1..=kmax {
                            a.sl::<u8>(mtst, kmax * r * ob).fill(0xa5);
                            run(&g, &a, &|e| menc(e, nb, sk, cm, &m0, &m1, k, x, mtst))?;
                            cmp(format!("K={} nb={} sk={}", k, nb, sk), &(0..k).collect::<Vec<_>>());
                        }
                        // (ii) other rows' content: K = 8 with odd (x2) / even (x3) rows replaced, kept rows must match
                        for (xn, keep) in [(x2, 0usize), (x3, 1usize)] {
                            a.sl::<u8>(mtst, kmax * r * ob).fill(0xa5);
                            run(&g, &a, &|e| menc(e, nb, sk, cm, &m0, &m1, kmax, xn, mtst))?;
                            cmp(format!("K=8 nb={} sk={} neighbours replaced (kept {} rows)", nb, sk, if keep == 0 { "even" } else { "odd" }),
                                &(0..kmax).filter(|t| t % 2 == keep).collect::<Vec<_>>());
                        }
                    }
                    // (iii) the scalar twins (no padding), every K: rows == per-row MMA K = 1; K = 8 with neighbours replaced
                    for k in (1..=kmax).filter(|_| MMA_TWIN_SKS.contains(&sk)) {
                        a.sl::<u8>(mtst, kmax * r * ob).fill(0xa5);
                        run(&g, &a, &|e| tenc(e, k, sk, tmax, cm, &m0, &m1, x, mtst))?;
                        cmp(format!("twin K={} sk={}", k, sk), &(0..k).collect::<Vec<_>>());
                    }
                    for (xn, keep) in [(x2, 0usize), (x3, 1usize)].into_iter().filter(|_| MMA_TWIN_SKS.contains(&sk)) {
                        a.sl::<u8>(mtst, kmax * r * ob).fill(0xa5);
                        run(&g, &a, &|e| tenc(e, kmax, sk, tmax, cm, &m0, &m1, xn, mtst))?;
                        cmp(format!("twin K=8 sk={} neighbours replaced", sk), &(0..kmax).filter(|t| t % 2 == keep).collect::<Vec<_>>());
                    }
                }
            }
            let ident = if bad.is_empty() { "yes".to_string() } else { mma_ok = false; format!("NO: {}", bad.join("; ")) };
            // timing in the engine's mode
            let mut t1 = None;
            for &k in &ks {
                for &(nb, sk) in &cfgs {
                        let ms = time(&g, &a, reps, n, &|e, i| {
                            let (m0, m1) = mats[i % nc];
                            menc(e, nb, sk, s.mode, &m0, &m1, k, x, mtst)
                        })?;
                        if k == 1 && (nb, sk) == cfgs[0] { t1 = Some(ms); }
                        let rel = t1.map_or("-".to_string(), |b| format!("{:.2}", ms / b));
                        mrows.push(format!("| {} | {} x {} | {} | {} | {} | {} | {:.0} | {:.0} | {} | {} |", s.name, r, c, s.count, k, nb,
                                           sk, ms * 1e3, wb as f64 / ms / 1e6, rel, ident));
                        mres.push((si, k, nb, sk, ms));
                }
            }
            for &k in &ks {
                for &sk in sks.iter().filter(|s| MMA_TWIN_SKS.contains(s)) {
                    let ms = time(&g, &a, reps, n, &|e, i| { let (m0, m1) = mats[i % nc]; tenc(e, k, sk, tmax, s.mode, &m0, &m1, x, mtst) })?;
                    mrows.push(format!("| {} | {} x {} | {} | {} | twin | {} | {:.0} | {:.0} | {} | {} |", s.name, r, c, s.count, k, sk,
                                       ms * 1e3, wb as f64 / ms / 1e6, t1.map_or("-".to_string(), |b| format!("{:.2}", ms / b)), ident));
                    mres.push((si, k, 0, sk, ms));
                }
            }
        }
    }

    // full-pass GEMV estimate per (K, D): sum over shapes count x t
    let base = |ks1: &[(usize, f64)]| ks1.iter().find(|p| p.0 == 1).map(|p| p.1);
    let mut old_pass: Vec<(usize, f64)> = Vec::new();
    if do_old {
        println!("\nFull-pass GEMV estimate (sum of shape x layers using it{}):", if head { ", lm_head once" } else { ", no lm_head" });
        println!("| K | D | pass ms | t/t1 | engine default D |");
        println!("|---|---|---|---|---|");
        let pass = |k: usize, d: usize| -> Option<f64> {
            let mut tot = 0.0;
            for (si, s) in shapes.iter().enumerate() {
                tot += s.count as f64 * res.iter().find(|r| r.0 == si && r.1 == k && r.2 == d)?.3;
            }
            Some(tot)
        };
        let b1 = pass(1, gemv_d(1));
        let mut best: Vec<(usize, usize)> = Vec::new();
        for &k in &ks {
            let mut bk: Option<(usize, f64)> = None;
            for &d in gemv_ds(k).iter().filter(|d| ds_filter.contains(d)) {
                let Some(p) = pass(k, d) else { continue };
                if bk.is_none_or(|b| p < b.1) { bk = Some((d, p)); }
                println!("| {} | {} | {:.2} | {} | {} |", k, d, p, b1.map_or("-".into(), |b| format!("{:.3}", p / b)),
                         if d == gemv_d(k) { "yes" } else { "" });
            }
            if let Some((d, p)) = bk { best.push((k, d)); old_pass.push((k, p)); }
        }
        let sdd: Vec<String> = best.iter().map(|(k, d)| format!("{}:{}", k, d)).collect();
        println!("\nfastest D per K: SD_GEMV_D={}", sdd.join(","));
        println!("bit-identity (every K-row output row == K=1 output on that row): {}", if all_ok { "ALL OK" } else { "FAILED (see table)" });
    }
    if do_mma {
        println!("\nMMA (qm_gemv / qm_swiglu, gemv_mma_q4.metal), per shape:");
        println!("| shape | R x C | per pass | K | NB | SK | µs | GB/s | t/t1 | K-invariant |");
        println!("|---|---|---|---|---|---|---|---|---|---|");
        for l in &mrows { println!("{}", l); }
        println!("\nMMA-only full-pass estimate (best NB / SK per shape{}):", if head { ", lm_head once" } else { ", no lm_head" });
        println!("| K | pass ms | t/t1 (mma) | t / old K=1 | old best ms (this run) | config per shape |");
        println!("|---|---|---|---|---|---|");
        let mut mp: Vec<(usize, f64, String)> = Vec::new();
        for &k in &ks {
            let (mut tot, mut cfgs) = (0.0, Vec::new());
            for (si, s) in shapes.iter().enumerate() {
                let b = mres.iter().filter(|r| r.0 == si && r.1 == k && r.2 != 0).min_by(|p, q| p.4.total_cmp(&q.4)).unwrap();
                tot += s.count as f64 * b.4;
                cfgs.push(format!("{}:{}/{}", s.name.split('|').next().unwrap(), b.2, b.3));
            }
            mp.push((k, tot, cfgs.join(" ")));
        }
        let mb1 = base(&mp.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>());
        let ob1 = base(&old_pass);
        for (k, p, cf) in &mp {
            println!("| {} | {:.2} | {} | {} | {} | {} |", k, p, mb1.map_or("-".into(), |b| format!("{:.3}", p / b)),
                     ob1.map_or("-".into(), |b| format!("{:.3}", p / b)),
                     old_pass.iter().find(|q| q.0 == *k).map_or("-".into(), |q| format!("{:.2}", q.1)), cf);
        }
        // engine plan: one SK per shape (it is part of the result); per K the faster of the twin (NB "twin") and the
        // MMA kernel (best NB) at that SK; SK per shape minimising the sum over the K measured
        let best_at = |si: usize, k: usize, sk: usize| -> Option<(usize, f64)> {
            mres.iter().filter(|r| r.0 == si && r.1 == k && r.3 == sk).map(|r| (r.2, r.4)).min_by(|p, q| p.1.total_cmp(&q.1))
        };
        let mut plan: Vec<(usize, usize)> = Vec::new();
        for si in 0..shapes.len() {
            let cost = |sk: usize| -> Option<f64> { ks.iter().map(|&k| best_at(si, k, sk).map(|b| b.1)).sum() };
            if let Some((sk, _)) = sks.iter().filter_map(|&sk| cost(sk).map(|c| (sk, c))).min_by(|p, q| p.1.total_cmp(&q.1)) {
                plan.push((si, sk));
            }
        }
        let label = |nb: usize| if nb == 0 { "twin".to_string() } else { format!("mma{}", nb) };
        println!("\nEngine plan (one SK per shape; per K the faster of twin / MMA at that SK; every K bit-identical per row):");
        println!("| K | pass ms | t/t1 | t / old K=1 | per shape (kernel/SK) |");
        println!("|---|---|---|---|---|");
        let mut p1 = None;
        for &k in &ks {
            let (mut tot, mut cf) = (0.0, Vec::new());
            for &(si, sk) in &plan {
                let s = &shapes[si];
                let (nb, t) = best_at(si, k, sk).unwrap();
                tot += s.count as f64 * t;
                cf.push(format!("{}:{}/{}", s.name.split('|').next().unwrap(), label(nb), sk));
            }
            if k == ks[0] { p1 = Some(tot); }
            println!("| {} | {:.2} | {} | {} | {} |", k, tot, p1.map_or("-".into(), |b| format!("{:.3}", tot / b)),
                     base(&old_pass).map_or("-".into(), |b| format!("{:.3}", tot / b)), cf.join(" "));
        }
        // twin only (best SK per shape and K) for reference
        println!("\nTwin only (qm_gemvt / qm_swiglut, best SK per shape and K):");
        println!("| K | pass ms |");
        println!("|---|---|");
        for &k in &ks {
            let tot: f64 = shapes.iter().enumerate().map(|(si, s)| s.count as f64 * mres.iter().filter(|r| r.0 == si && r.1 == k && r.2 == 0)
                .map(|r| r.4).fold(f64::MAX, f64::min)).sum();
            println!("| {} | {:.2} |", k, tot);
        }
        println!("\nMMA K=1 output vs today's decode (old q4b K=1, D=2), 8 x rows per shape:");
        println!("| shape | mode | NB/SK | closeness |");
        println!("|---|---|---|---|");
        for l in &close_lines { println!("{}", l); }
        println!("\nMMA K-invariance (rows of K=1..8, every NB per SK, K=8 with neighbour rows replaced, and the twins at every K == per-row MMA K=1): {}",
                 if mma_ok { "ALL OK" } else { "FAILED (see table)" });
    }
    if !all_ok { return Err("K-row GEMV not bit-identical to K = 1".into()); }
    if !mma_ok { return Err("MMA K-row output not K-invariant".into()); }
    Ok(())
}

/// `--head-keep FILE`: MTP draft head at the config's lm_head shape (random weights), full (lm_head K = 1 m3 +
/// mtp_draft) vs pruned (head_pruned + mtp_draft_idx over the keep-list) per family; checks kept logits bit-identical
/// to the full head's rows and the pruned draft == the full argmax restricted to kept rows (bf16-rounded, lowest id).
fn head_keep_bench(model: &str, keep: &str, reps: usize) -> Result<(), String> {
    use crate::mtp::{head_pruned, pruned_pipelines};
    use crate::qwen_layers::{gemv, GemvImpl};
    let cfg = QwenConfig::from_model_dir(model)?;
    let kl = crate::draft_vocab::KeepList::read_for(std::path::Path::new(keep), cfg.vocab)?;
    let (v, h, nk) = (cfg.vocab, cfg.hidden, kl.ids.len());
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16", "gemv_mma_q4", "spec", "head_pruned"])?;
    let fams = [GemvImpl::Old, GemvImpl::Mma];
    let full_name = |f: GemvImpl| match f { GemvImpl::Old => gemv_kernel(1, gemv_d(1), 3), GemvImpl::Mma | GemvImpl::MmaD => mma_twin_gemv_kernel(1, crate::qwen_layers::MMA_SK, 3) };
    for f in fams {
        g.pipeline(&full_name(f))?;
        for p in pruned_pipelines(f) { g.pipeline(p)?; }
    }
    g.pipeline("mtp_draft")?;
    let cs = std::mem::size_of::<streamdense_metal::SdCtrl>();
    let len = q4_bytes(v, h) + h * 2 + 2 * v * 4 + nk * 4 + 2 * cs + (8 << 10) + 6 * 256;
    eprintln!("head-keep bench: lm_head {} x {} ({:.3} GB), keep {} / {} rows ({:.1}%), arena {:.3} GB", v, h,
              q4_bytes(v, h) as f64 / 1e9, nk, v, 100.0 * nk as f64 / v as f64, len as f64 / 1e9);
    let buf = g.buffer(len)?;
    let mut a = Arena { ptr: contents(&buf), buf, top: 0, len };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let m = put_mat(&mut a, &mut rng, v, h)?;
    let x = a.alloc(h * 2)?;
    for e in a.sl::<u16>(x, h) { *e = to_bf(rng.n()); }
    let (of, ok_, idx, ctrl, mctrl) = (a.alloc(v * 4)?, a.alloc(v * 4)?, a.alloc(nk * 4)?, a.alloc(cs)?, a.alloc(cs)?);
    a.sl::<u32>(idx, nk).copy_from_slice(&kl.ids);
    a.sl::<u8>(ctrl, cs).fill(0);
    a.sl::<u8>(mctrl, cs).fill(0);
    let draft = |a: &Arena| a.sl::<streamdense_metal::SdCtrl>(ctrl, 1)[0].draft;
    let full = |e: &Enc, f: GemvImpl| {
        gemv(e, f, &m, 1, 3, x, of);
        e.pso("mtp_draft").b(0, of).b(1, ctrl).b(2, mctrl).u(3, v as u32).groups((1, 1), 1024);
    };
    let pruned = |e: &Enc, f: GemvImpl| {
        head_pruned(e, f, &m, idx, nk, x, ok_);
        e.pso("mtp_draft_idx").b(0, ok_).b(1, ctrl).b(2, mctrl).u(3, nk as u32).b(4, idx).groups((1, 1), 1024);
    };
    let (bytes_full, bytes_kept) = (q4_bytes(v, h) as f64, q4_bytes(v, h) as f64 * nk as f64 / v as f64);
    println!("| family | head | rows | GPU ms (head + argmax) | head GB/s | kept logits bit-identical | draft == restricted argmax |");
    println!("|---|---|---|---|---|---|---|");
    let mut all_ok = true;
    let mut saved: Vec<(GemvImpl, f64, f64)> = Vec::new();
    for f in fams {
        run(&g, &a, &|e| full(e, f))?;
        let df = draft(&a);
        a.sl::<u8>(ok_, nk * 4).fill(0xa5);
        run(&g, &a, &|e| pruned(e, f))?;
        let dk = draft(&a);
        let (lf, lk) = (a.sl::<u32>(of, v), a.sl::<u32>(ok_, nk));
        let nbad = kl.ids.iter().zip(lk.iter()).filter(|(&t, &b)| lf[t as usize] != b).count();
        // restricted argmax on the CPU: bf16-rounded value, lowest id on ties
        let bfr = |u: u32| f32::from_bits((to_bf(f32::from_bits(u)) as u32) << 16);
        let mut best = (f32::NEG_INFINITY, u32::MAX);
        for &t in &kl.ids { let val = bfr(lf[t as usize]); if best.1 == u32::MAX || val > best.0 { best = (val, t); } }
        let (ok1, ok2) = (nbad == 0, dk == best.1);
        all_ok &= ok1 && ok2;
        let tf = time(&g, &a, reps, 3, &|e, _| full(e, f))?;
        let tk = time(&g, &a, reps, 3, &|e, _| pruned(e, f))?;
        let nm = f.name();
        println!("| {} | full | {} | {:.3} | {:.0} | - | - (full draft {}) |", nm, v, tf, bytes_full / tf / 1e6, df);
        println!("| {} | pruned | {} | {:.3} | {:.0} | {} | {} |", nm, nk, tk, bytes_kept / tk / 1e6,
                 if ok1 { "yes".to_string() } else { format!("NO: {} / {} differ", nbad, nk) },
                 if ok2 { format!("yes ({})", dk) } else { format!("NO: gpu {} cpu {}", dk, best.1) });
        saved.push((f, tf, tk));
    }
    println!();
    for (f, tf, tk) in saved {
        println!("{}: full {:.3} ms, pruned {:.3} ms -> {:.3} ms saved per MTP draft step ({:.1}%)", f.name(), tf, tk, tf - tk, 100.0 * (tf - tk) / tf);
    }
    if !all_ok { return Err("pruned head not bit-identical to the full head on kept rows".into()); }
    Ok(())
}

/// `--mask-frac 0,10,25,50` (read mask, readmask.rs): the engine's exact-pass matmuls (Mma family, the engine's
/// twin / MMA choice per K) at the config's shapes, with the given percent of FFN 64-neuron blocks masked (gate / up
/// row tiles skipped, down groups not listed); mixer projections and lm_head unmasked (heads: not in this bench).
/// Checks per shape and K = 1..8: all-ones mask == plain kernel and fl = 1 ("read everything") with a random mask ==
/// plain kernel, bit for bit; masked swiglu rows exactly 0 and kept rows bit-identical. Prints GPU µs per shape and
/// the full-pass estimate (Σ layers x t) per K and fraction, plain kernels as the 0% reference.
fn mask_bench(model: &str, ks: &[usize], fracs: &[f64], reps: usize, rot: f64) -> Result<(), String> {
    use crate::qwen_layers::{gemv_m, mma_mask_pipelines, mma_pipelines, swiglu_m, GemvImpl, MatMask};
    let cfg = QwenConfig::from_model_dir(model)?;
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_mma_q4"])?;
    for n in mma_pipelines().into_iter().chain(mma_mask_pipelines()) { g.pipeline(n)?; }
    let (h, nd, na, nl) = (cfg.hidden, cfg.n_deltanet(), cfg.n_attention(), cfg.layers);
    // (name, rows, cols, count, mode; None = swiglu), masked: 1 = rows (gate+up), 2 = columns (down), 0 = none
    let shapes: Vec<(&str, usize, usize, usize, Option<u32>, u32)> = vec![
        ("dn_in", cfg.dn_in_rows(), h, nd, Some(0), 0), ("dn_out", h, cfg.dn_v_dim(), nd, Some(2), 0),
        ("attn_in", cfg.attn_in_rows(), h, na, Some(0), 0), ("o_proj", h, cfg.q_dim(), na, Some(2), 0),
        ("gate+up", cfg.inter, h, nl, None, 1), ("down", h, cfg.inter, nl, Some(2), 2), ("lm_head", cfg.vocab, h, 1, Some(3), 0)];
    let kmax = MMA_K_MAX;
    let mb = |s: &(&str, usize, usize, usize, Option<u32>, u32)| q4_bytes(s.1, s.2) * if s.4.is_none() { 2 } else { 1 };
    let copies = |b: usize| ((rot / b as f64).ceil() as usize).max(1);
    let wmax = shapes.iter().map(|s| copies(mb(s)) * (mb(s) + 2048)).max().unwrap();
    let (maxr, maxc) = (shapes.iter().map(|s| s.1).max().unwrap(), shapes.iter().map(|s| s.2).max().unwrap());
    let nblk = cfg.inter / 64;
    let len = wmax + kmax * maxc * 2 + 4 * (kmax * maxr * 4 + 256) + 64 * (nblk + 64) + (4 << 20);
    eprintln!("mask bench: arena {:.3} GB, FFN {} blocks / layer, fractions {:?} %, K {:?}", len as f64 / 1e9, nblk, fracs, ks);
    let buf = g.buffer(len)?;
    let mut a = Arena { ptr: contents(&buf), buf, top: 0, len };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let x = a.alloc(kmax * maxc * 2)?;
    for v in a.sl::<u16>(x, kmax * maxc) { *v = to_bf(rng.n()); }
    let (o_ref, o_tst) = (a.alloc(kmax * maxr * 4)?, a.alloc(kmax * maxr * 4)?);
    let fl = a.alloc(16)?;
    // per fraction: row-tile bits (inter / 8 tiles) + group list; index 0 of `masks` = all-ones (0% masked)
    let nt = cfg.inter / 8;
    let mut all_fr = vec![0.0];
    all_fr.extend(fracs.iter().copied().filter(|f| *f > 0.0));
    let mut masks: Vec<(f64, usize, usize, Vec<bool>)> = Vec::new(); // (frac, rb, cl, hot per block)
    for &f in &all_fr {
        let off = ((f / 100.0) * nblk as f64).round() as usize;
        // deterministic: masked blocks = a shuffled prefix
        let mut ord: Vec<usize> = (0..nblk).collect();
        for i in (1..nblk).rev() { let j = (rng.next() % (i as u64 + 1)) as usize; ord.swap(i, j); }
        let mut hot = vec![true; nblk];
        for &b in &ord[..off.min(nblk)] { hot[b] = false; }
        let (rb, cl) = (a.alloc(nt.div_ceil(32) * 4)?, a.alloc((1 + nblk) * 4)?);
        let rbs = a.sl::<u32>(rb, nt.div_ceil(32));
        rbs.fill(0);
        for t in 0..nt { if hot[t / 8] { rbs[t / 32] |= 1 << (t % 32); } }
        let cls = a.sl::<u32>(cl, 1 + nblk);
        let mut n = 0;
        for b in 0..nblk { if hot[b] { cls[1 + n] = b as u32; n += 1; } }
        cls[0] = n as u32;
        masks.push((f, rb, cl, hot));
    }
    let wbase = a.top;
    let set_fl = |a: &Arena, v: u32| a.sl::<u32>(fl, 1)[0] = v;
    let mk = |m: &(f64, usize, usize, Vec<bool>), kind: u32| MatMask { rb: m.1, cl: m.2, fl, flags: kind };
    let mut ok = true;
    // t[(shape, k, frac index)] = ms; frac index usize::MAX = plain kernel
    let mut tm: Vec<(usize, usize, usize, f64)> = Vec::new();
    println!("| shape | R x C | per pass | K | plain µs | {} |", masks.iter().map(|m| format!("{}% µs", m.0)).collect::<Vec<_>>().join(" | "));
    println!("|---|---|---|---|---|{}", "---|".repeat(masks.len()));
    for (si, s) in shapes.iter().enumerate() {
        let (name, r, c, count, mode, kind) = *s;
        a.top = wbase;
        let nc = copies(mb(s));
        let mut mats = Vec::with_capacity(nc);
        for _ in 0..nc {
            let m0 = put_mat(&mut a, &mut rng, r, c)?;
            let m1 = if mode.is_none() { put_mat(&mut a, &mut rng, r, c)? } else { m0 };
            mats.push((m0, m1));
        }
        let enc = |e: &Enc, i: usize, k: usize, mm: Option<MatMask>, md: Option<u32>, out: usize| {
            let (m0, m1) = mats[i % nc];
            match md { Some(md) => gemv_m(e, GemvImpl::Mma, &m0, k, md, x, out, mm), None => swiglu_m(e, GemvImpl::Mma, &m0, &m1, k, x, out, mm) }
        };
        // bit-identity (checked mode: m3 f32 for gemv shapes, bf16 for swiglu)
        if kind != 0 {
            let (cm, ob) = (if mode.is_none() { None } else { Some(3) }, if mode.is_none() { 2 } else { 4 });
            for k in 1..=kmax {
                run(&g, &a, &|e| enc(e, 0, k, None, cm, o_ref))?;
                let p: Vec<u8> = a.sl::<u8>(o_ref, k * r * ob).to_vec();
                let mut cmp = |label: &str, mm: MatMask, fv: u32, rows_only: Option<&Vec<bool>>| -> Result<(), String> {
                    set_fl(&a, fv);
                    a.sl::<u8>(o_tst, k * r * ob).fill(0xa5);
                    run(&g, &a, &|e| enc(e, 0, k, Some(mm), cm, o_tst))?;
                    let q = a.sl::<u8>(o_tst, k * r * ob);
                    let bad = match rows_only {
                        None => p.iter().zip(q.iter()).filter(|(u, v)| u != v).count(),
                        Some(hot) => (0..k * r).filter(|&i| { let row = i % r; let e = &q[i * ob..(i + 1) * ob];
                            if hot[row / 64] { e != &p[i * ob..(i + 1) * ob] } else { e.iter().any(|b| *b != 0) } }).count(),
                    };
                    if bad > 0 { ok = false; println!("  {} K={} {}: {} bytes / elements differ", name, k, label, bad); }
                    Ok(())
                };
                cmp("all-ones mask == plain", mk(&masks[0], kind), 0, None)?;
                if let Some(mr) = masks.iter().find(|m| m.0 > 0.0) {
                    cmp("random mask + read-everything flag == plain", mk(mr, kind), 1, None)?;
                    if kind == 1 { cmp("masked rows 0, kept rows == plain", mk(mr, kind), 0, Some(&mr.3))?; }
                }
            }
            set_fl(&a, 0);
        }
        let n = if nc == 1 { 3 } else { 2 * nc };
        for &k in ks {
            let tp = time(&g, &a, reps, n, &|e, i| enc(e, i, k, None, mode, o_tst))?;
            tm.push((si, k, usize::MAX, tp));
            let mut cells = Vec::new();
            for (fi, m) in masks.iter().enumerate() {
                let t = if kind == 0 { tp } else { time(&g, &a, reps, n, &|e, i| enc(e, i, k, Some(mk(m, kind)), mode, o_tst))? };
                tm.push((si, k, fi, t));
                cells.push(format!("{:.0}", t * 1e3));
            }
            println!("| {} | {} x {} | {} | {} | {:.0} | {} |", name, r, c, count, k, tp * 1e3, cells.join(" | "));
        }
    }
    println!("\nFull-pass estimate (Σ layers x µs; FFN blocks masked, everything else plain; lm_head once):");
    println!("| K | plain ms | {} |", masks.iter().map(|m| format!("{}% ms (x plain)", m.0)).collect::<Vec<_>>().join(" | "));
    println!("|---|---|{}", "---|".repeat(masks.len()));
    for &k in ks {
        let pass = |fi: usize| -> f64 { shapes.iter().enumerate().map(|(si, s)| s.3 as f64 * tm.iter().find(|t| t.0 == si && t.1 == k && t.2 == fi).map_or(0.0, |t| t.3)).sum() };
        let p0 = pass(usize::MAX);
        let cells: Vec<String> = (0..masks.len()).map(|fi| { let p = pass(fi); format!("{:.2} ({:.3})", p, p / p0) }).collect();
        println!("| {} | {:.2} | {} |", k, p0, cells.join(" | "));
    }
    println!("\nbit-identity (all-ones == plain, read-everything == plain, masked rows 0 / kept rows equal; K = 1..8): {}", if ok { "ALL OK" } else { "FAILED" });
    if !ok { return Err("masked kernels not bit-identical where required".into()); }
    Ok(())
}
