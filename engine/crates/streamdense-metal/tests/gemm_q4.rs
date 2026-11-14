//! S1 self-test (PF-A): q4b_gemm_{32x64,64x64}_m{0,2,3} and q4b_gemm_swiglu_{32x64,64x64} (engine/shaders/gemm_q4.metal)
//! vs the exact GEMV path (q4b_gemv_k{8,1}_m* / q4b_swiglu_k{8,1}, K-invariant, so K = 8 chunks + K = 1 tails).
//! The GEMM is NOT bit-exact (MMA summation order). Bars:
//! - epilogues bit-exact: m0 == bf(m3), m2 == bf(bf(res) + bf(m3)), swiglu == MLX chain(m3 gate, m3 up);
//! - m0 (and the swiglu gate / up pre-activations): |gemm - gemv| <= 1 bf16 ulp at max(|gemm|, |gemv|), except
//!   near-cancelled outputs (< rms / 64), judged by absolute error <= ulp(rms) / 16;
//! - m3 (f32): |diff| <= 1e-4 * rms.
//! - m2 and the swiglu output vs the GEMV are printed only: a 1-ulp flip of v (m2) or of g / u (swiglu) legitimately
//!   becomes 2 (m2: v across a binade edge) or up to 5 (swiglu, measured at 17408 x 5120) output ulps.
//! Prints the fraction of outputs that differ. Shapes: odd R (37, 517, 1001), R < BN, C % 512 != 0 (320, 768), and
//! two real 27B shapes; T in {1, 5, 8, 9, 131} (all tails of both tiles, BM 32 / 64).
//! Run: cargo test --release --offline -p streamdense-metal --test gemm_q4 -- --nocapture --test-threads 1
mod a_util;
use a_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu};

const TILES: [(usize, usize); 2] = [(32, 64), (64, 64)];
const TS: [usize; 5] = [1, 5, 8, 9, 131];

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16", "gemm_q4"]).expect("compile");
    for k in [1, 8] {
        for m in [0, 2, 3] { g.pipeline(&format!("q4b_gemv_k{}_m{}", k, m)).unwrap(); }
        g.pipeline(&format!("q4b_swiglu_k{}", k)).unwrap();
    }
    for (bm, bn) in TILES {
        for m in [0, 2, 3] { g.pipeline(&format!("q4b_gemm_{}x{}_m{}", bm, bn, m)).unwrap(); }
        g.pipeline(&format!("q4b_gemm_swiglu_{}x{}", bm, bn)).unwrap();
    }
    g
}

#[derive(Clone, Copy)]
struct M { r: usize, c: usize, w: usize, s: usize, b: usize }

fn put_mat(ar: &mut Ar, rng: &mut Rng, r: usize, c: usize) -> M {
    let (w, s, b) = rand_q4(rng, r, c);
    M { r, c, w: ar.alloc_put(&w), s: ar.alloc_put(&s), b: ar.alloc_put(&b) }
}

fn run(g: &Gpu, ar: &Ar, f: impl Fn(&Enc)) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

/// Exact path: K = 8 chunks, then K = 1 rows (K-invariant).
fn chunks(t: usize) -> Vec<(usize, usize)> {
    let mut v = vec![]; let mut i = 0;
    while i < t { let k = if t - i >= 8 { 8 } else { 1 }; v.push((i, k)); i += k; }
    v
}

fn gemv(g: &Gpu, ar: &Ar, m: M, t: usize, mode: u32, x: usize, out: usize) {
    let ob = if mode == 3 { 4 } else { 2 };
    run(g, ar, |e| for (i, k) in chunks(t) {
        let o = out + i * m.r * ob;
        e.pso(&format!("q4b_gemv_k{}_m{}", k, mode)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, x + i * m.c * 2)
            .b(4, o).u(5, m.r as u32).u(6, m.c as u32).b(7, o).b(8, o).groups((m.r.div_ceil(4), 1), 64);
    });
}
fn gemm(g: &Gpu, ar: &Ar, m: M, t: usize, mode: u32, x: usize, out: usize, tile: (usize, usize)) {
    let (bm, bn) = tile;
    run(g, ar, |e| {
        e.pso(&format!("q4b_gemm_{}x{}_m{}", bm, bn, mode)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, x)
            .b(4, out).u(5, m.r as u32).u(6, m.c as u32).b(7, out).b(8, out).u(9, t as u32)
            .groups((m.r.div_ceil(bn), t.div_ceil(bm)), 128);
    });
}
fn swiglu_gemv(g: &Gpu, ar: &Ar, gm: M, um: M, t: usize, x: usize, out: usize) {
    run(g, ar, |e| for (i, k) in chunks(t) {
        e.pso(&format!("q4b_swiglu_k{}", k)).b(0, gm.w).b(1, gm.s).b(2, gm.b).b(3, um.w).b(4, um.s).b(5, um.b)
            .b(6, x + i * gm.c * 2).b(7, out + i * gm.r * 2).u(8, gm.r as u32).u(9, gm.c as u32)
            .groups((gm.r.div_ceil(2), 1), 64);
    });
}
fn swiglu_gemm(g: &Gpu, ar: &Ar, gm: M, um: M, t: usize, x: usize, out: usize, tile: (usize, usize)) {
    let (bm, bn) = tile;
    run(g, ar, |e| {
        e.pso(&format!("q4b_gemm_swiglu_{}x{}", bm, bn)).b(0, gm.w).b(1, gm.s).b(2, gm.b).b(3, um.w).b(4, um.s).b(5, um.b)
            .b(6, x).b(7, out).u(8, gm.r as u32).u(9, gm.c as u32).u(10, t as u32)
            .groups((gm.r.div_ceil(bn / 2), t.div_ceil(bm)), 128);
    });
}

/// bf16 ulp at magnitude |x| (spacing of the bf16 grid there; subnormal floor ignored).
fn ulp_at(x: f32) -> f32 {
    let a = x.abs().max(f32::MIN_POSITIVE);
    f32::from_bits(((a.to_bits() >> 23) << 23).saturating_sub(7 << 23)).max(f32::MIN_POSITIVE)
}
fn rms(v: &[f32]) -> f32 { (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt() as f32 }

/// bf16 comparison at the output magnitude. `scale[i]`: the magnitude whose ulp bounds element i.
/// Returns (worst |diff| / ulp(scale), n differing, n violating).
fn judge(gemm: &[u16], gemv: &[u16], scale: &[f32], rmsv: f32, tol: f32) -> (f32, usize, usize) {
    let (mut worst, mut nd, mut bad) = (0f32, 0usize, 0usize);
    for i in 0..gemv.len() {
        let (a, b) = (bf(gemm[i]), bf(gemv[i]));
        if !a.is_finite() { bad += 1; continue; }
        let d = (a - b).abs();
        if d == 0.0 { continue; }
        nd += 1;
        let r = if scale[i].abs() < rmsv / 64.0 { d / (ulp_at(rmsv) / 16.0) } else { d / ulp_at(scale[i]) } / tol;
        if r > worst && std::env::var("GQ_DEBUG").is_ok() {
            println!("    worst@{}: gemm {:e} gemv {:e} scale {:e} rms {:e}", i, a, b, scale[i], rmsv);
        }
        worst = worst.max(r);
        if r > 1.0 + 1e-6 { bad += 1; }
    }
    (worst, nd, bad)
}

struct Tot { cases: usize, n: usize, nd: usize, bad: usize, dec_bad: usize, info_m2: f32, info_sw: f32 }

/// CPU mirror of q4b_silu_mul / gq_silu_mul (MLX bf16 swiglu rounding chain).
fn silu_mul(gv: f32, uv: f32) -> u16 {
    let (g, u) = (rbf(gv), rbf(uv));
    let e = rbf(g.abs().exp());
    let d = rbf(1.0 + e);
    let y = rbf(1.0 / d);
    let s = if g < 0.0 { y } else { rbf(1.0 - y) };
    to_bf(rbf(g * s) * u)
}

/// Plain GEMM: (1) m0 == bf(m3) and m2 == bf(bf(res) + bf(m3)) bit-exact (one accumulation, three epilogues);
/// (2) m0 vs GEMV <= 1 ulp at max(|gemm|, |gemv|); m3 vs GEMV |d| <= 1e-4 rms; m2 vs GEMV printed (info).
fn check_plain(g: &Gpu, ar: &mut Ar, rng: &mut Rng, r: usize, c: usize, ts: &[usize], tot: &mut Tot) {
    let m = put_mat(ar, rng, r, c);
    let tmax = *ts.iter().max().unwrap();
    let x = ar.alloc_put(&rng.bf_n(tmax * c, 1.0));
    let res0 = rng.bf_n(tmax * r, 0.5);
    let (o_ref, o_g) = (ar.alloc(tmax * r * 4), ar.alloc(tmax * r * 4));
    for &t in ts {
        let n = t * r;
        ar.fill(o_ref, n * 4, 0xff); gemv(g, ar, m, t, 0, x, o_ref); let v0: Vec<u16> = ar.get(o_ref, n);
        ar.fill(o_ref, n * 4, 0xff); gemv(g, ar, m, t, 3, x, o_ref); let v3: Vec<f32> = ar.get(o_ref, n);
        ar.put(o_ref, &res0[..n]); gemv(g, ar, m, t, 2, x, o_ref); let v2: Vec<u16> = ar.get(o_ref, n);
        let rv = rms(&v3);
        for tile in TILES {
            ar.fill(o_g, n * 4, 0xff); gemm(g, ar, m, t, 0, x, o_g, tile); let a0: Vec<u16> = ar.get(o_g, n);
            ar.fill(o_g, n * 4, 0xff); gemm(g, ar, m, t, 3, x, o_g, tile); let a3: Vec<f32> = ar.get(o_g, n);
            ar.put(o_g, &res0[..n]); gemm(g, ar, m, t, 2, x, o_g, tile); let a2: Vec<u16> = ar.get(o_g, n);
            let dec = (0..n).filter(|&i| a0[i] != to_bf(a3[i]) || a2[i] != to_bf(rbf(bf(res0[i]) + rbf(a3[i])))).count();
            let s0: Vec<f32> = (0..n).map(|i| bf(a0[i]).abs().max(bf(v0[i]).abs())).collect();
            let (w0, d0, b0) = judge(&a0, &v0, &s0, rv, 1.0);
            let s2: Vec<f32> = (0..n).map(|i| bf(res0[i]).abs().max(bf(a2[i]).abs()).max(bf(v2[i]).abs())).collect();
            let (w2, d2, _) = judge(&a2, &v2, &s2, rv, 1.0);
            let mut f3 = 0f32;
            for i in 0..n { f3 = f3.max((a3[i] - v3[i]).abs() / rv); }
            let b3 = (f3 > 1e-4) as usize;
            println!("  {:5}x{:5} T {:3} tile {}x{}: m0 worst {:.2} ulp, {:.3}% differ | m3 max |d|/rms {:.1e} | m2 (info) worst {:.2} ulp, {:.3}% | m0/m2 == m3 epilogue: {}{}",
                r, c, t, tile.0, tile.1, w0, 100.0 * d0 as f64 / n as f64, f3, w2, 100.0 * d2 as f64 / n as f64,
                if dec == 0 { "exact".to_string() } else { format!("{} differ", dec) }, if b0 + b3 + dec > 0 { "  <-- VIOLATION" } else { "" });
            tot.cases += 3; tot.n += n; tot.nd += d0; tot.bad += b0 + b3; tot.dec_bad += dec; tot.info_m2 = tot.info_m2.max(w2);
        }
    }
}

/// SwiGLU GEMM: (1) out == silu_mul(gemm_m3(gate), gemm_m3(up)) bit-exact (same accumulation as the plain GEMM);
/// (2) the gate / up pre-activations bf(m3) vs the GEMV's within 1 ulp (that is the GEMM bar; the swiglu chain then
/// maps a 1-ulp flip of g and/or u to up to ~4 output ulp at a binade edge); out vs the GEMV swiglu printed (info).
fn check_swiglu(g: &Gpu, ar: &mut Ar, rng: &mut Rng, r: usize, c: usize, ts: &[usize], tot: &mut Tot) {
    let gm = put_mat(ar, rng, r, c);
    let um = put_mat(ar, rng, r, c);
    let tmax = *ts.iter().max().unwrap();
    let x = ar.alloc_put(&rng.bf_n(tmax * c, 1.0));
    let (o_ref, o_g, fg, fu) = (ar.alloc(tmax * r * 2), ar.alloc(tmax * r * 2), ar.alloc(tmax * r * 4), ar.alloc(tmax * r * 4));
    for &t in ts {
        let n = t * r;
        ar.fill(o_ref, n * 2, 0xff); swiglu_gemv(g, ar, gm, um, t, x, o_ref); let v: Vec<u16> = ar.get(o_ref, n);
        gemv(g, ar, gm, t, 3, x, fg); let vg: Vec<f32> = ar.get(fg, n);
        gemv(g, ar, um, t, 3, x, fu); let vu: Vec<f32> = ar.get(fu, n);
        let gemv_dec = (0..n).filter(|&i| v[i] != silu_mul(vg[i], vu[i])).count();
        let s: Vec<f32> = v.iter().map(|&h| bf(h)).collect();
        let rv = rms(&s);
        let (vgb, vub): (Vec<u16>, Vec<u16>) = ((0..n).map(|i| to_bf(vg[i])).collect(), (0..n).map(|i| to_bf(vu[i])).collect());
        let (rg, ru) = (rms(&vg[..n]), rms(&vu[..n]));
        for tile in TILES {
            ar.fill(o_g, n * 2, 0xff); swiglu_gemm(g, ar, gm, um, t, x, o_g, tile); let a: Vec<u16> = ar.get(o_g, n);
            gemm(g, ar, gm, t, 3, x, fg, tile); let ag: Vec<f32> = ar.get(fg, n);
            gemm(g, ar, um, t, 3, x, fu, tile); let au: Vec<f32> = ar.get(fu, n);
            let dec = (0..n).filter(|&i| a[i] != silu_mul(ag[i], au[i])).count();
            let (agb, aub): (Vec<u16>, Vec<u16>) = ((0..n).map(|i| to_bf(ag[i])).collect(), (0..n).map(|i| to_bf(au[i])).collect());
            let sg: Vec<f32> = (0..n).map(|i| bf(agb[i]).abs().max(bf(vgb[i]).abs())).collect();
            let su: Vec<f32> = (0..n).map(|i| bf(aub[i]).abs().max(bf(vub[i]).abs())).collect();
            let (wg, dg, bg) = judge(&agb, &vgb, &sg, rg, 1.0);
            let (wu, du, bu) = judge(&aub, &vub, &su, ru, 1.0);
            let so: Vec<f32> = (0..n).map(|i| bf(a[i]).abs().max(s[i].abs())).collect();
            let (w, d, _) = judge(&a, &v, &so, rv, 1.0);
            println!("  swiglu {:5}x{:5} T {:3} tile {}x{}: gate worst {:.2} ulp ({:.3}%), up {:.2} ({:.3}%) | out (info) worst {:.2} ulp, {:.3}% differ | out == chain(m3): {} (GEMV: {}){}",
                r, c, t, tile.0, tile.1, wg, 100.0 * dg as f64 / n as f64, wu, 100.0 * du as f64 / n as f64, w,
                100.0 * d as f64 / n as f64, if dec == 0 { "exact".to_string() } else { format!("{} differ", dec) },
                if gemv_dec == 0 { "exact".to_string() } else { format!("{} differ", gemv_dec) },
                if bg + bu + dec > 0 { "  <-- VIOLATION" } else { "" });
            tot.cases += 1; tot.n += n; tot.nd += d; tot.bad += bg + bu; tot.dec_bad += dec; tot.info_sw = tot.info_sw.max(w);
        }
    }
}

/// No write outside the T x R output (rows >= T, cols >= R masked).
fn check_bounds(g: &Gpu, ar: &mut Ar, rng: &mut Rng) {
    let (r, c) = (37, 320);
    let m = put_mat(ar, rng, r, c);
    let gm = put_mat(ar, rng, r, c);
    let x = ar.alloc_put(&rng.bf_n(9 * c, 1.0));
    let pad = 64 * r * 4;
    let o = ar.alloc(9 * r * 4 + pad);
    for tile in TILES {
        for t in [1usize, 5, 9] {
            for mode in [0u32, 3] {
                let ob = if mode == 3 { 4 } else { 2 };
                ar.fill(o, 9 * r * 4 + pad, 0xab);
                gemm(g, ar, m, t, mode, x, o, tile);
                let tail: Vec<u8> = ar.get(o + t * r * ob, 9 * r * 4 + pad - t * r * ob);
                assert!(tail.iter().all(|&b| b == 0xab), "gemm m{} {:?} T {} wrote past T x R", mode, tile, t);
            }
            ar.fill(o, 9 * r * 4 + pad, 0xab);
            swiglu_gemm(g, ar, gm, m, t, x, o, tile);
            let tail: Vec<u8> = ar.get(o + t * r * 2, 9 * r * 4 + pad - t * r * 2);
            assert!(tail.iter().all(|&b| b == 0xab), "swiglu {:?} T {} wrote past T x R", tile, t);
        }
    }
    println!("  bounds: no writes past T x R (R 37, T 1/5/9, both tiles, m0/m3/swiglu)");
}

#[test]
fn gemm_q4_vs_gemv() {
    let g = gpu();
    let mut ar = Ar::new(&g, 384 << 20);
    let mut rng = Rng::new(25);
    let mut tot = Tot { cases: 0, n: 0, nd: 0, bad: 0, dec_bad: 0, info_m2: 0.0, info_sw: 0.0 };
    check_bounds(&g, &mut ar, &mut rng);
    for (r, c) in [(37, 320), (517, 1024), (1001, 768)] { check_plain(&g, &mut ar, &mut rng, r, c, &TS, &mut tot); }
    for (r, c) in [(37, 320), (517, 1024), (1001, 768)] { check_swiglu(&g, &mut ar, &mut rng, r, c, &TS, &mut tot); }
    // real 27B shapes (o_proj / out_proj 5120 x 6144; FFN gate|up 17408 x 5120), the longest tail only
    check_plain(&g, &mut ar, &mut rng, 5120, 6144, &[131], &mut tot);
    check_swiglu(&g, &mut ar, &mut rng, 17408, 5120, &[9, 131], &mut tot);
    println!("gemm_q4: {} comparisons, {} m0 / swiglu outputs, {:.3}% differ from the GEMV; {} beyond the 1-ulp bar \
              (m0, gate/up pre-activations); epilogue mismatches {}; info: m2 worst {:.2} ulp, swiglu out worst {:.2} ulp",
             tot.cases, tot.n, 100.0 * tot.nd as f64 / tot.n as f64, tot.bad, tot.dec_bad, tot.info_m2, tot.info_sw);
    assert_eq!(tot.dec_bad, 0, "GEMM m0 / m2 / swiglu epilogue != the m3 accumulation + the GEMV rounding chain");
    assert_eq!(tot.bad, 0, "GEMM vs GEMV beyond 1 bf16 ulp at the output magnitude");
}
