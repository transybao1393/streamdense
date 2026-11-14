//! U5 self-test: rmsnorm_b and rmsnorm_gated (engine/shaders/norm_bf16.metal).
//! CPU mirror (the kernel formula, f64 sum of squares) + MLX vectors (fast.rms_norm, Qwen3NextRMSNormGated),
//! T = 1 vs T = 8 bit-equality, out_stride writes into concat halves.
//! Run: cargo test --release --offline -p streamdense-metal --test norm_bf16 -- --nocapture --test-threads 1
mod a_util;
use a_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu};

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "norm_bf16"]).expect("compile");
    g.pipeline("rmsnorm_b").unwrap();
    g.pipeline("rmsnorm_gated").unwrap();
    g
}

fn rms_b(g: &Gpu, ar: &Ar, x: usize, w: usize, out: usize, n: usize, t: usize, eps: f32, stride: usize) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    e.pso("rmsnorm_b").b(0, x).b(1, w).b(2, out).u(3, n as u32).f(4, eps).u(5, stride as u32);
    e.groups((t, 1), 1024);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

fn rms_gated(g: &Gpu, ar: &Ar, y: usize, z: usize, zs: usize, w: usize, out: usize, vh: usize, t: usize, eps: f32) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    e.pso("rmsnorm_gated").b(0, y).b(1, z).u(2, zs as u32).b(3, w).b(4, out).f(5, eps);
    e.groups((vh, t), 128);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

/// CPU: out = bf(bf(w) * bf(x * r)), r = 1/sqrt(mean(x^2) + eps) (sum in f64).
fn cpu_rms(x: &[u16], w: &[u16], eps: f32) -> Vec<u16> {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| (bf(v) as f64).powi(2)).sum();
    let r = (1.0 / ((ss / n as f64) as f32 + eps).sqrt()) as f32;
    x.iter().zip(w).map(|(&v, &wv)| to_bf(bf(wv) * rbf(bf(v) * r))).collect()
}
fn cpu_gated(y: &[u16], z: &[u16], w: &[u16], eps: f32) -> Vec<u16> {
    let n = cpu_rms(y, w, eps);
    n.iter().zip(z).map(|(&nv, &zv)| {
        let zf = bf(zv);
        let e = 1.0 / (1.0 + zf.abs().exp());
        let sig = if zf < 0.0 { e } else { 1.0 - e };
        to_bf(zf * sig * bf(nv))
    }).collect()
}

#[test]
fn rmsnorm_b_cpu_mlx_kinv_stride() {
    let g = gpu();
    let eps = 1e-6f32;
    let mut ok = true;
    if let Some(v) = vectors("norm") {
        // hidden: [8][5120]
        let (x, w, y, y1) = (v.u16s("hidden.x"), v.u16s("hidden.w"), v.u16s("hidden.y"), v.u16s("hidden.y_k1"));
        let n = w.len();
        let t = x.len() / n;
        let mut ar = Ar::new(&g, 4 << 20);
        let (xo, wo) = (ar.alloc_put(&x), ar.alloc_put(&w));
        let out = ar.alloc(t * n * 2);
        rms_b(&g, &ar, xo, wo, out, n, t, eps, n);
        let gk = ar.get::<u16>(out, t * n);
        let mut g1 = Vec::new();
        for r in 0..t { rms_b(&g, &ar, xo + r * n * 2, wo, out, n, 1, eps, n); g1.extend(ar.get::<u16>(out, n)); }
        let cpu: Vec<u16> = (0..t).flat_map(|r| cpu_rms(&x[r * n..(r + 1) * n], &w, eps)).collect();
        let (u1, n1) = cmp_bf(&gk, &y1);
        let (u8_, n8) = cmp_bf(&gk, &y);
        let (uc, nc) = cmp_bf(&gk, &cpu);
        println!("  hidden 8x5120: vs MLX y_k1 max ulp {} ({} / {}), vs MLX y {} ({}), vs CPU {} ({}), T=8 == T=1 {}",
                 u1, n1, y1.len(), u8_, n8, uc, nc, gk == g1);
        ok &= u1 == 0 && gk == g1;
        // q/k norm: [8][24][256] -> 192 rows of 256
        let (x, w, y) = (v.u16s("qk.x"), v.u16s("qk.w"), v.u16s("qk.y"));
        let n = w.len();
        let rows = x.len() / n;
        let mut ar = Ar::new(&g, 4 << 20);
        let (xo, wo) = (ar.alloc_put(&x), ar.alloc_put(&w));
        let out = ar.alloc(rows * n * 2);
        rms_b(&g, &ar, xo, wo, out, n, rows, eps, n);
        let gk = ar.get::<u16>(out, rows * n);
        let cpu: Vec<u16> = (0..rows).flat_map(|r| cpu_rms(&x[r * n..(r + 1) * n], &w, eps)).collect();
        let (u, nn) = cmp_bf(&gk, &y);
        let (uc, _) = cmp_bf(&gk, &cpu);
        println!("  qk 8x24x256: vs MLX max ulp {} ({} / {}), vs CPU {}", u, nn, y.len(), uc);
        ok &= u == 0;
    }
    // out_stride: two rows of 5120 normed into the halves of a [T][10240] concat (MTP cat), random data
    let mut rng = Rng::new(3);
    let (n, t) = (5120, 3);
    let x = rng.bf_n(t * n, 2.0);
    let w = rng.bf_n(n, 1.0);
    let mut ar = Ar::new(&g, 4 << 20);
    let (xo, wo) = (ar.alloc_put(&x), ar.alloc_put(&w));
    let cat = ar.alloc(t * 2 * n * 2);
    ar.fill(cat, t * 2 * n * 2, 0x7f);
    rms_b(&g, &ar, xo, wo, cat + n * 2, n, t, eps, 2 * n);
    let c = ar.get::<u16>(cat, t * 2 * n);
    let mut sok = true;
    for r in 0..t {
        sok &= c[r * 2 * n..r * 2 * n + n].iter().all(|&v| v == 0x7f7f);
        sok &= c[r * 2 * n + n..(r + 1) * 2 * n] == cpu_rms(&x[r * n..(r + 1) * n], &w, eps)[..];
    }
    println!("  out_stride 2N into right half: untouched left + CPU-exact right {}", sok);
    assert!(ok && sok);
}

#[test]
fn rmsnorm_gated_cpu_mlx_kinv() {
    let g = gpu();
    let eps = 1e-6f32;
    let Some(v) = vectors("norm") else { return; };
    let (y, z, w, out_ref, xn) = (v.u16s("gated.y"), v.u16s("gated.z"), v.u16s("gated.w"), v.u16s("gated.out"), v.u16s("gated.xn"));
    let sh = v.shape("gated.y");
    let (t, vh, d) = (sh[0], sh[1], sh[2]);
    // z inside a wider projection row to exercise z_stride: [T][stride] with z at column 64
    let zs = vh * d + 192;
    let mut zw = vec![0u16; t * zs];
    for r in 0..t { zw[r * zs + 64..r * zs + 64 + vh * d].copy_from_slice(&z[r * vh * d..(r + 1) * vh * d]); }
    let mut ar = Ar::new(&g, 4 << 20);
    let (yo, zo, wo) = (ar.alloc_put(&y), ar.alloc_put(&zw), ar.alloc_put(&w));
    let out = ar.alloc(t * vh * d * 2);
    rms_gated(&g, &ar, yo, zo + 64 * 2, zs, wo, out, vh, t, eps);
    let gk = ar.get::<u16>(out, t * vh * d);
    let mut g1 = Vec::new();
    for r in 0..t {
        rms_gated(&g, &ar, yo + r * vh * d * 2, zo + (r * zs + 64) * 2, zs, wo, out, vh, 1, eps);
        g1.extend(ar.get::<u16>(out, vh * d));
    }
    let rows = t * vh;
    let cpu_xn: Vec<u16> = (0..rows).flat_map(|r| cpu_rms(&y[r * d..(r + 1) * d], &w, eps)).collect();
    let cpu: Vec<u16> = (0..rows).flat_map(|r| cpu_gated(&y[r * d..(r + 1) * d], &z[r * d..(r + 1) * d], &w, eps)).collect();
    let (u, n) = cmp_bf(&gk, &out_ref);
    let (ux, _) = cmp_bf(&cpu_xn, &xn);
    let (uc, nc) = cmp_bf(&gk, &cpu);
    println!("  gated 8x48x128: vs MLX max ulp {} ({} / {}), CPU xn vs MLX xn {}, vs CPU {} ({}), T=8 == T=1 {}",
             u, n, out_ref.len(), ux, uc, nc, gk == g1);
    assert!(u == 0 && gk == g1, "rmsnorm_gated not bit-exact vs MLX or not K-invariant");
}
