//! U25 microbench + quick check of the prefill GEMM (`gemm_q4.metal`) vs the K=8 GEMV (`gemv_q4_bf16.metal`) at the
//! real Qwen shapes from `config.json` (random weights; no model load). For each shape and T: GEMM µs (min of
//! `--reps` command buffers), effective GB/s (weights + x + out bytes) and TFLOPS (2·R·C·T), and the K=8 GEMV path
//! (ceil(T/8) dispatches, as the exact prefill runs it). Check: GEMM vs GEMV outputs on the same data (max abs, max
//! bf16 ulp, share of outputs != ), modes 0 / 2 / 3 and SwiGLU, including a tail T.
//! Run: cargo run --release --offline -p streamdense-model --example qwen_gemm_bench -- [--model DIR] [--lm-head]
//!      [--reps 3] [--ts 8,32,64,128] [--tile 64x128]
use streamdense_metal::{commit, contents, gpu_ms, wait, Enc, Gpu};
use streamdense_model::QwenConfig;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 } // [0, 1)
    fn n(&mut self) -> f32 { (0..4).map(|_| self.f()).sum::<f32>() * 1.732 - 3.464 } // ~N(0, 1)
}
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }
fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
fn ulp(a: u16, b: u16) -> u32 {
    let k = |h: u16| if h & 0x8000 != 0 { -((h & 0x7fff) as i32) } else { h as i32 };
    (k(a) - k(b)).unsigned_abs()
}

struct Arena { buf: streamdense_metal::Buffer, ptr: *mut u8, top: usize, len: usize }
impl Arena {
    fn alloc(&mut self, n: usize) -> usize {
        let o = self.top;
        self.top = (self.top + n + 255) & !255;
        assert!(self.top <= self.len, "bench arena too small");
        o
    }
    #[allow(clippy::mut_from_ref)] // bench arena: disjoint regions of one shared GPU buffer
    fn sl<T>(&self, off: usize, n: usize) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut T, n) } }
}

#[derive(Clone, Copy)]
struct M { r: usize, c: usize, w: usize, s: usize, b: usize }

fn put_mat(a: &mut Arena, rng: &mut Rng, r: usize, c: usize) -> M {
    let (w, s, b) = (a.alloc(r * c / 2), a.alloc(r * c / 64 * 2), a.alloc(r * c / 64 * 2));
    for v in a.sl::<u32>(w, r * c / 8) { *v = rng.next() as u32; }
    let (ss, bb) = (a.sl::<u16>(s, r * c / 64), a.sl::<u16>(b, r * c / 64));
    for i in 0..r * c / 64 {
        let sc = 0.002 + 0.018 * rng.f();
        ss[i] = to_bf(sc);
        bb[i] = to_bf(-8.0 * sc + 0.01 * rng.n());
    }
    M { r, c, w, s, b }
}

const TILES: [(usize, usize); 2] = [(32, 64), (64, 64)];
/// Tile (BM, BN) for T rows: `force` if given, else the bench default.
fn pick_b(t: usize, force: Option<(usize, usize)>) -> (usize, usize) { force.unwrap_or(if t <= 32 { (32, 64) } else { (64, 64) }) }

#[allow(clippy::too_many_arguments)]
fn enc_gemm(e: &Enc, m: M, t: usize, mode: u32, x: usize, out: usize, b: (usize, usize)) {
    e.pso(&format!("q4b_gemm_{}x{}_m{}", b.0, b.1, mode)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, x).b(4, out)
        .u(5, m.r as u32).u(6, m.c as u32).b(7, out).b(8, out).u(9, t as u32)
        .groups((m.r.div_ceil(b.1), t.div_ceil(b.0)), 128);
}
fn enc_swiglu_gemm(e: &Enc, g: M, u: M, t: usize, x: usize, out: usize, b: (usize, usize)) {
    e.pso(&format!("q4b_gemm_swiglu_{}x{}", b.0, b.1)).b(0, g.w).b(1, g.s).b(2, g.b).b(3, u.w).b(4, u.s).b(5, u.b)
        .b(6, x).b(7, out).u(8, g.r as u32).u(9, g.c as u32).u(10, t as u32)
        .groups((g.r.div_ceil(b.1 / 2), t.div_ceil(b.0)), 128);
}
/// GEMV path for t rows: K=8 chunks then 4/2/1 (as the exact prefill).
fn enc_gemv(e: &Enc, m: M, t: usize, mode: u32, x: usize, out: usize) {
    let mut i = 0;
    while i < t {
        let k = [8, 4, 2, 1].into_iter().find(|&k| k <= t - i).unwrap();
        let (xo, oo) = (x + i * m.c * 2, out + i * m.r * if mode == 3 { 4 } else { 2 });
        e.pso(&format!("q4b_gemv_k{}_m{}", k, mode)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, xo).b(4, oo)
            .u(5, m.r as u32).u(6, m.c as u32).b(7, oo).b(8, oo).groups((m.r.div_ceil(4), 1), 64);
        i += k;
    }
}
fn enc_swiglu_gemv(e: &Enc, g: M, u: M, t: usize, x: usize, out: usize) {
    let mut i = 0;
    while i < t {
        let k = [8, 4, 2, 1].into_iter().find(|&k| k <= t - i).unwrap();
        e.pso(&format!("q4b_swiglu_k{}", k)).b(0, g.w).b(1, g.s).b(2, g.b).b(3, u.w).b(4, u.s).b(5, u.b)
            .b(6, x + i * g.c * 2).b(7, out + i * g.r * 2).u(8, g.r as u32).u(9, g.c as u32)
            .groups((g.r.div_ceil(2), 1), 64);
        i += k;
    }
}

/// min over `reps` command buffers of (gpu ms / n) for n encodes of f.
fn time(g: &Gpu, a: &Arena, reps: usize, n: usize, f: &dyn Fn(&Enc)) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let cb = g.command_buffer();
        let e = Enc::new(g, &cb, &a.buf);
        for _ in 0..n { f(&e); }
        e.end();
        commit(&cb);
        wait(&cb).unwrap();
        best = best.min(gpu_ms(&cb) / n as f64);
    }
    best
}
fn run(g: &Gpu, a: &Arena, f: &dyn Fn(&Enc)) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &a.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

/// (max abs, max ulp, share differing) of two bf16 arrays.
fn cmp16(x: &[u16], y: &[u16]) -> (f32, u32, f64) {
    let (mut ma, mut mu, mut nd) = (0f32, 0u32, 0usize);
    for (&p, &q) in x.iter().zip(y) {
        ma = ma.max((bf(p) - bf(q)).abs());
        mu = mu.max(ulp(p, q));
        nd += (p != q) as usize;
    }
    (ma, mu, nd as f64 / x.len() as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let model = arg("--model").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4").into());
    let reps: usize = arg("--reps").map_or(3, |v| v.parse().unwrap());
    let force_b: Option<(usize, usize)> = arg("--tile").map(|v| { let p: Vec<usize> = v.split('x').map(|x| x.parse().unwrap()).collect(); (p[0], p[1]) });
    let ts: Vec<usize> = arg("--ts").map_or(vec![8, 32, 64, 128], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    let lm = args.iter().any(|a| a == "--lm-head");
    let cfg = QwenConfig::from_model_dir(&model).expect("config");
    let tmax = *ts.iter().max().unwrap().max(&131);

    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16", "gemm_q4"]).expect("compile");
    for (bm, bn) in TILES {
        for m in [0, 2, 3] { g.pipeline(&format!("q4b_gemm_{}x{}_m{}", bm, bn, m)).unwrap(); }
        g.pipeline(&format!("q4b_gemm_swiglu_{}x{}", bm, bn)).unwrap();
    }
    for k in [1, 2, 4, 8] {
        for m in [0, 2, 3] { g.pipeline(&format!("q4b_gemv_k{}_m{}", k, m)).unwrap(); }
        g.pipeline(&format!("q4b_swiglu_k{}", k)).unwrap();
    }
    let (h, inter) = (cfg.hidden, cfg.inter);
    let mut shapes: Vec<(&str, usize, usize)> = vec![
        ("dn_in", cfg.dn_in_rows(), h), ("attn_in", cfg.attn_in_rows(), h), ("out_proj", h, cfg.dn_v_dim()),
        ("o_proj", h, cfg.q_dim()), ("gate|up", inter, h), ("down", h, inter),
    ];
    if lm { shapes.push(("lm_head", cfg.vocab, h)); }
    let maxr = shapes.iter().map(|s| s.1).max().unwrap();
    let maxc = shapes.iter().map(|s| s.2).max().unwrap();
    let wbytes: usize = shapes.iter().map(|s| (s.1 * s.2 * 5 / 8 + 1024) * if s.0 == "gate|up" { 2 } else { 1 }).max().unwrap() + (1 << 20);
    let len = wbytes + tmax * maxc * 2 + 4 * tmax * maxr * 4 + (16 << 20);
    eprintln!("bench arena {:.2} GB on {}", len as f64 / 1e9, g.name());
    let buf = g.buffer(len).expect("buffer");
    let mut a = Arena { ptr: contents(&buf), buf, top: 0, len };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let x = a.alloc(tmax * maxc * 2);
    for v in a.sl::<u16>(x, tmax * maxc) { *v = to_bf(rng.n()); }
    let (o1, o2) = (a.alloc(tmax * maxr * 4), a.alloc(tmax * maxr * 4));
    let (o3, o4) = (a.alloc(tmax * maxr * 4), a.alloc(tmax * maxr * 4));

    println!("| shape | R x C | T | tile | GEMM µs | GB/s | TFLOPS | GEMV-k8 µs | GEMV GB/s | speedup |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut checks: Vec<String> = Vec::new();
    for &(name, r, c) in &shapes {
        let top0 = a.top;
        let sw = name == "gate|up";
        let m0 = put_mat(&mut a, &mut rng, r, c);
        let m1 = if sw { put_mat(&mut a, &mut rng, r, c) } else { m0 };
        let wb = (r * c / 2 + r * c / 64 * 4) * if sw { 2 } else { 1 };
        // ---- correctness on T = 131 (two full 64-tiles + a tail for every b) or the largest T
        for &t in &[tmax.min(131), 5] {
            for b in TILES {
                if force_b.is_some_and(|f| f != b) { continue; }
                if sw {
                    run(&g, &a, &|e| enc_swiglu_gemm(e, m0, m1, t, x, o1, b));
                    run(&g, &a, &|e| enc_swiglu_gemv(e, m0, m1, t, x, o2));
                    let (ma, mu, nd) = cmp16(a.sl::<u16>(o1, t * r), a.sl::<u16>(o2, t * r));
                    checks.push(format!("{} swiglu T={} {:?}: max abs {:.3e}, max ulp {}, differ {:.2}%", name, t, b, ma, mu, nd * 100.0));
                    continue;
                }
                // m0
                run(&g, &a, &|e| enc_gemm(e, m0, t, 0, x, o1, b));
                run(&g, &a, &|e| enc_gemv(e, m0, t, 0, x, o2));
                let (ma, mu, nd) = cmp16(a.sl::<u16>(o1, t * r), a.sl::<u16>(o2, t * r));
                checks.push(format!("{} m0 T={} {:?}: max abs {:.3e}, max ulp {}, differ {:.2}%", name, t, b, ma, mu, nd * 100.0));
                // m3 (f32): max rel to the row scale
                run(&g, &a, &|e| enc_gemm(e, m0, t, 3, x, o3, b));
                run(&g, &a, &|e| enc_gemv(e, m0, t, 3, x, o4));
                let (p, q) = (a.sl::<f32>(o3, t * r), a.sl::<f32>(o4, t * r));
                let rms = (q.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / q.len() as f64).sqrt();
                let md = p.iter().zip(q.iter()).map(|(u, v)| (u - v).abs() as f64).fold(0.0, f64::max);
                checks.push(format!("{} m3 T={} {:?}: max |diff| {:.3e} (rms out {:.3e}, rel {:.2e})", name, t, b, md, rms, md / rms));
                // m2: res = residual seeded from o1's bf16 out (same start for both)
                let seed: Vec<u16> = (0..t * r).map(|i| to_bf(((i * 7919) % 1000) as f32 * 0.01 - 5.0)).collect();
                a.sl::<u16>(o1, t * r).copy_from_slice(&seed);
                a.sl::<u16>(o2, t * r).copy_from_slice(&seed);
                run(&g, &a, &|e| enc_gemm(e, m0, t, 2, x, o1, b));
                run(&g, &a, &|e| enc_gemv(e, m0, t, 2, x, o2));
                let (ma, mu, nd) = cmp16(a.sl::<u16>(o1, t * r), a.sl::<u16>(o2, t * r));
                checks.push(format!("{} m2 T={} {:?}: max abs {:.3e}, max ulp {}, differ {:.2}%", name, t, b, ma, mu, nd * 100.0));
            }
        }
        // ---- timing
        for &t in &ts {
            let b = pick_b(t, force_b);
            let n = if r > 100_000 { 2 } else { 8 };
            let (tg, tv) = if sw {
                (time(&g, &a, reps, n, &|e| enc_swiglu_gemm(e, m0, m1, t, x, o1, b)),
                 time(&g, &a, reps, n, &|e| enc_swiglu_gemv(e, m0, m1, t, x, o2)))
            } else {
                (time(&g, &a, reps, n, &|e| enc_gemm(e, m0, t, 0, x, o1, b)),
                 time(&g, &a, reps, n, &|e| enc_gemv(e, m0, t, 0, x, o2)))
            };
            let bytes = (wb + t * c * 2 + t * r * 2 * if sw { 1 } else { 1 }) as f64;
            let flops = 2.0 * (r * c * t) as f64 * if sw { 2.0 } else { 1.0 };
            println!("| {} | {} x {} | {} | {}x{} | {:.0} | {:.0} | {:.2} | {:.0} | {:.0} | {:.1}x |", name, r, c, t, b.0, b.1, tg * 1e3,
                     bytes / tg / 1e6, flops / tg / 1e9, tv * 1e3, bytes / tv / 1e6, tv / tg);
        }
        a.top = top0;
    }
    println!();
    for c in checks { println!("{}", c); }
}
