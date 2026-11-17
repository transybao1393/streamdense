//! S2 self-test (P2 U17): bf16_gemv_k1 / bf16_gemv_k2 (engine/shaders/gemv_bf16.metal), the MTP fc.
//! - CPU mirror of the MLX GEMVKernel order (lane l: cols 128 i + 4 l .. + 3, fp32 acc in order, load_safe tail,
//!   shuffle-down 16/8/4/2/1 tree, one bf16 rounding): bit-exact on aligned shapes, C tails (C % 128 != 0,
//!   C % 4 != 0 scalar path), R % 32 != 0 and R % 4 != 0 (inward-shifted last simdgroup), and the real fc shape.
//! - K-invariance: each row of a K = 2 dispatch == a K = 1 dispatch on that row, bitwise.
//! - MLX vectors (testdata/vectors/gemv_bf16.safetensors, made with MLX): y_k1 (M = 1 per row).
//! Run: cargo test --release --offline -p streamdense-metal --test gemv_bf16 -- --nocapture --test-threads 1
mod a_util;
use a_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu};

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_bf16"]).expect("compile");
    for k in ["bf16_gemv_k1", "bf16_gemv_k2", "bf16_gemv_k4", "bf16_gemv_k8", "bf16_gemv_g4"] { g.pipeline(k).unwrap(); }
    g
}

fn gemv(g: &Gpu, ar: &Ar, w: usize, x: usize, out: usize, r: usize, c: usize, k: usize) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    e.pso(&format!("bf16_gemv_k{}", k)).b(0, w).b(1, x).b(2, out).u(3, r as u32).u(4, c as u32);
    e.groups((r.div_ceil(32), 1), 256);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

/// CPU mirror of one output (MLX gemv order, see the shader header).
fn mirror(w: &[u16], x: &[u16], c: usize, row: usize) -> u16 {
    let m = &w[row * c..(row + 1) * c];
    let mut acc = [0f32; 32];
    let n_iter = c / 128;
    for it in 0..n_iter {
        for (l, a) in acc.iter_mut().enumerate() {
            let bn = it * 128 + 4 * l;
            for j in 0..4 { *a += bf(m[bn + j]) * bf(x[bn + j]); }
        }
    }
    if c > n_iter * 128 {
        for (l, a) in acc.iter_mut().enumerate() {
            let bn = n_iter * 128 + 4 * l;
            for j in 0..4 {
                let (wv, xv) = if bn + j < c { (bf(m[bn + j]), bf(x[bn + j])) } else { (0.0, 0.0) };
                *a += wv * xv;
            }
        }
    }
    for s in [16usize, 8, 4, 2, 1] {
        let prev = acc;
        for l in 0..32 { if l + s < 32 { acc[l] = prev[l] + prev[l + s]; } }
    }
    to_bf(acc[0])
}

/// Runs K = 1 per row and K = 2 on (R, C) random data; returns (K = 2 out, K = 1 rows concatenated).
fn run_both(g: &Gpu, w: &[u16], x: &[u16], r: usize, c: usize) -> (Vec<u16>, Vec<u16>) {
    let mut ar = Ar::new(g, (w.len() + x.len() + 4 * r) * 2 + (1 << 16));
    let (wo, xo) = (ar.alloc_put(w), ar.alloc_put(x));
    let out = ar.alloc(2 * r * 2);
    ar.fill(out, 2 * r * 2, 0xff);
    gemv(g, &ar, wo, xo, out, r, c, 2);
    let k2 = ar.get::<u16>(out, 2 * r);
    let mut k1 = Vec::new();
    for t in 0..2 {
        ar.fill(out, 2 * r * 2, 0xff);
        gemv(g, &ar, wo, xo + t * c * 2, out, r, c, 1);
        k1.extend(ar.get::<u16>(out, r));
    }
    (k2, k1)
}

#[test]
fn bf16_gemv_mirror_kinv_tails() {
    let g = gpu();
    let mut rng = Rng::new(11);
    let mut ok = true;
    // (R, C): aligned; tails in C (128 / 4 alignment); R % 32 != 0; R % 4 != 0; tiny; the real fc (5120 x 10240)
    for &(r, c) in &[(512, 1024), (36, 1000), (6, 1003), (4, 4), (100, 130), (5, 129), (64, 2048), (5120, 10240)] {
        let w = rng.bf_n(r * c, 0.05);
        let x = rng.bf_n(2 * c, 1.0);
        let (k2, k1) = run_both(&g, &w, &x, r, c);
        let cpu: Vec<u16> = (0..2).flat_map(|t| { let xt = &x[t * c..(t + 1) * c]; (0..r).map(|row| mirror(&w, xt, c, row)).collect::<Vec<_>>() }).collect();
        let kinv = k2 == k1;
        let (u, n) = cmp_bf(&k2, &cpu);
        println!("  R {:5} C {:5}: K=2 == K=1 {}, vs CPU mirror max ulp {} ({} / {} differ)", r, c, kinv, u, n, cpu.len());
        ok &= kinv && u == 0;
    }
    assert!(ok, "bf16_gemv: K-invariance or CPU mirror mismatch");
}

#[test]
fn bf16_gemv_vs_mlx() {
    let g = gpu();
    let Some(v) = vectors("gemv_bf16") else { return; };
    let mut ok = true;
    for n in ["a", "b", "c", "d"] {
        if !v.has(&format!("{}.w", n)) { continue; }
        let sh = v.shape(&format!("{}.w", n));
        let (r, c) = (sh[0], sh[1]);
        let (w, x) = (v.u16s(&format!("{}.w", n)), v.u16s(&format!("{}.x", n)));
        let (y1, y) = (v.u16s(&format!("{}.y_k1", n)), v.u16s(&format!("{}.y", n)));
        let (k2, k1) = run_both(&g, &w, &x, r, c);
        let (u1, n1) = cmp_bf(&k2, &y1);
        let (um, nm) = cmp_bf(&k2, &y);
        println!("  MLX case {} (R {} C {}): vs y_k1 (M=1) max ulp {} ({} / {}), vs y (M=2, info) {} ({}), K=2 == K=1 {}",
                 n, r, c, u1, n1, y1.len(), um, nm, k2 == k1);
        ok &= u1 == 0 && k2 == k1;
    }
    assert!(ok, "bf16_gemv vs MLX M = 1 mismatch");
}

/// MTP warm-up v2 (2026-11-17): K = 4 / 8 / 16 rows are bit-equal to K = 1 per row (the real fc shape and tails),
/// the one-dispatch grid `bf16_gemv_g4` likewise, and the GPU time of 128 fc rows: 64 x K = 2 (today) vs 32 x K = 4 vs
/// 16 x K = 8 vs one g4 grid (the MTP warm-up v2 fc).
#[test]
fn bf16_gemv_wide_k_kinv_and_time() {
    let g = gpu();
    let mut rng = Rng::new(23);
    let mut ok = true;
    for &(r, c) in &[(36, 1000), (100, 130), (5120, 10240)] {
        let w = rng.bf_n(r * c, 0.05);
        let x = rng.bf_n(16 * c, 1.0);
        let mut ar = Ar::new(&g, (w.len() + x.len() + 32 * r) * 2 + (1 << 16));
        let (wo, xo) = (ar.alloc_put(&w), ar.alloc_put(&x));
        let out = ar.alloc(16 * r * 2);
        let mut k1 = Vec::new();
        for t in 0..16 {
            ar.fill(out, 16 * r * 2, 0xff);
            gemv(&g, &ar, wo, xo + t * c * 2, out, r, c, 1);
            k1.extend(ar.get::<u16>(out, r));
        }
        for k in [4usize, 8] {
            let mut got = Vec::new();
            for t0 in (0..16).step_by(k) {
                ar.fill(out, 16 * r * 2, 0xff);
                gemv(&g, &ar, wo, xo + t0 * c * 2, out, r, c, k);
                got.extend(ar.get::<u16>(out, k * r));
            }
            let eq = got == k1;
            println!("  R {:5} C {:5}: K={:2} rows == K=1 rows {}", r, c, k, eq);
            ok &= eq;
        }
        for k in [4usize] {
            ar.fill(out, 16 * r * 2, 0xff);
            let cb = g.command_buffer();
            let e = Enc::new(&g, &cb, &ar.buf);
            e.pso(&format!("bf16_gemv_g{}", k)).b(0, wo).b(1, xo).b(2, out).u(3, r as u32).u(4, c as u32);
            e.groups((r.div_ceil(32), 16 / k), 256);
            e.end();
            commit(&cb);
            wait(&cb).unwrap();
            let eq = ar.get::<u16>(out, 16 * r) == k1;
            println!("  R {:5} C {:5}: grid G={} 16 rows == K=1 rows {}", r, c, k, eq);
            ok &= eq;
        }
    }
    assert!(ok, "bf16_gemv wide K: not bit-equal to K = 1");
    // timing: 128 rows of the real fc shape
    let (r, c, n) = (5120usize, 10240usize, 128usize);
    let w = rng.bf_n(r * c, 0.05);
    let x = rng.bf_n(n * c, 1.0);
    let mut ar = Ar::new(&g, (w.len() + x.len() + n * r) * 2 + (1 << 16));
    let (wo, xo) = (ar.alloc_put(&w), ar.alloc_put(&x));
    let out = ar.alloc(n * r * 2);
    for k in [2usize, 4, 8] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let cb = g.command_buffer();
            let e = Enc::new(&g, &cb, &ar.buf);
            for t0 in (0..n).step_by(k) {
                e.pso(&format!("bf16_gemv_k{}", k)).b(0, wo).b(1, xo + t0 * c * 2).b(2, out + t0 * r * 2).u(3, r as u32).u(4, c as u32);
                e.groups((r.div_ceil(32), 1), 256);
            }
            e.end();
            commit(&cb);
            wait(&cb).unwrap();
            best = best.min(streamdense_metal::gpu_ms(&cb));
        }
        println!("  fc 5120 x 10240, {} rows as {} x K={:2}: {:.2} ms GPU (min of 5)", n, n / k, k, best);
    }
    for k in [4usize] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let cb = g.command_buffer();
            let e = Enc::new(&g, &cb, &ar.buf);
            e.pso(&format!("bf16_gemv_g{}", k)).b(0, wo).b(1, xo).b(2, out).u(3, r as u32).u(4, c as u32);
            e.groups((r.div_ceil(32), n / k), 256);
            e.end();
            commit(&cb);
            wait(&cb).unwrap();
            best = best.min(streamdense_metal::gpu_ms(&cb));
        }
        println!("  fc 5120 x 10240, {} rows as 1 grid dispatch G={}: {:.2} ms GPU (min of 5)", n, k, best);
    }
}
