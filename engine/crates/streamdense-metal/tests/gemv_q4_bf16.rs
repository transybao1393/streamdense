//! U4 self-test: q4b_gemv_k{1..8}_m{0,2,3} and q4b_swiglu_k{1..8} (engine/shaders/gemv_q4_bf16.metal).
//! - CPU reference (f64 exact affine dot) at the real shapes, sampled rows: max bf16 ulp.
//! - MLX vectors (testdata/vectors/gemv + swiglu): y_k1 (decode path) is the target.
//! - K-invariance: rows of K = 2..8 dispatches (8 rows, last dispatch shorter if K does not divide 8) bit-equal to K = 1, every mode.
//! - Mode consistency: m3 rounded == m0, m2 == bf(bf(res) + m0).
//! - GB/s at real shapes (random weights), printed.
//! Run: cargo test --release --offline -p streamdense-metal --test gemv_q4_bf16 -- --nocapture --test-threads 1
mod a_util;
use a_util::*;
use streamdense_metal::{commit, gpu_ms, wait, Enc, Gpu};

const KS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16"]).expect("compile");
    for k in KS {
        for m in [0, 2, 3] { g.pipeline(&format!("q4b_gemv_k{}_m{}", k, m)).unwrap(); }
        g.pipeline(&format!("q4b_swiglu_k{}", k)).unwrap();
    }
    g
}

#[derive(Clone, Copy)]
struct M { r: usize, c: usize, w: usize, s: usize, b: usize }

fn put_mat(ar: &mut Ar, r: usize, c: usize, w: &[u32], s: &[u16], b: &[u16]) -> M {
    M { r, c, w: ar.alloc_put(w), s: ar.alloc_put(s), b: ar.alloc_put(b) }
}

/// One q4b_gemv dispatch over `k` rows of x at x_off.
fn gemv(g: &Gpu, ar: &Ar, m: M, k: usize, mode: u32, x: usize, out: usize, res: usize, outf: usize) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    e.pso(&format!("q4b_gemv_k{}_m{}", k, mode)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, x).b(4, out)
        .u(5, m.r as u32).u(6, m.c as u32).b(7, res).b(8, outf);
    e.groups(((m.r + 3) / 4, 1), 64);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

fn swiglu(g: &Gpu, ar: &Ar, gm: M, um: M, k: usize, x: usize, out: usize) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    e.pso(&format!("q4b_swiglu_k{}", k)).b(0, gm.w).b(1, gm.s).b(2, gm.b).b(3, um.w).b(4, um.s).b(5, um.b)
        .b(6, x).b(7, out).u(8, gm.r as u32).u(9, gm.c as u32);
    e.groups(((gm.r + 1) / 2, 1), 64);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

/// CPU mirror of q4b_silu_mul (MLX bf16 swiglu rounding chain).
fn silu_mul(gv: f32, uv: f32) -> u16 {
    let (g, u) = (rbf(gv), rbf(uv));
    let e = rbf(g.abs().exp());
    let d = rbf(1.0 + e);
    let y = rbf(1.0 / d);
    let s = if g < 0.0 { y } else { rbf(1.0 - y) };
    to_bf(rbf(g * s) * u)
}

struct KRes { out: Vec<Vec<u16>>, res: Vec<Vec<u16>>, outf: Vec<Vec<f32>> } // per K index, [k*R]

/// Run all modes for K in KS; K = 1 is run row by row. Returns the outputs for each K (rows 0..8 of x).
fn run_all_k(g: &Gpu, ar: &mut Ar, m: M, x: &[u16], res0: &[u16]) -> KRes {
    let kmax = 8;
    let x_off = ar.alloc_put(x);
    let out = ar.alloc(kmax * m.r * 2);
    let res = ar.alloc(kmax * m.r * 2);
    let outf = ar.alloc(kmax * m.r * 4);
    let mut kr = KRes { out: vec![], res: vec![], outf: vec![] };
    for k in KS {
        let (mut o, mut rs, mut of) = (Vec::new(), Vec::new(), Vec::new());
        // 8 rows in K-row dispatches (the last one shorter when K does not divide 8)
        let calls: Vec<(usize, usize)> = (0..kmax).step_by(k).map(|t0| (t0, k.min(kmax - t0))).collect();
        ar.put(res, res0);
        ar.fill(out, kmax * m.r * 2, 0xff);
        ar.fill(outf, kmax * m.r * 4, 0xff);
        for &(t0, kk) in &calls {
            let xo = x_off + t0 * m.c * 2;
            gemv(g, ar, m, kk, 0, xo, out + t0 * m.r * 2, 0, 0);
            gemv(g, ar, m, kk, 2, xo, 0, res + t0 * m.r * 2, 0);
            gemv(g, ar, m, kk, 3, xo, 0, 0, outf + t0 * m.r * 4);
        }
        o.extend(ar.get::<u16>(out, kmax * m.r));
        rs.extend(ar.get::<u16>(res, kmax * m.r));
        of.extend(ar.get::<f32>(outf, kmax * m.r));
        kr.out.push(o); kr.res.push(rs); kr.outf.push(of);
    }
    kr
}

fn bits_eq_f32(a: &[f32], b: &[f32]) -> bool { a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }

fn check_k_invariance(name: &str, kr: &KRes) -> bool {
    let mut ok = true;
    for i in 1..KS.len() {
        let eo = kr.out[i] == kr.out[0];
        let er = kr.res[i] == kr.res[0];
        let ef = bits_eq_f32(&kr.outf[i], &kr.outf[0]);
        if !(eo && er && ef) {
            ok = false;
            println!("  {} K={} vs K=1: m0 {} m2 {} m3 {}  max ulp m0 {:?}", name, KS[i], eo, er, ef, cmp_bf(&kr.out[i], &kr.out[0]));
        }
    }
    ok
}

fn check_modes(name: &str, kr: &KRes, res0: &[u16]) -> bool {
    let o = &kr.out[0];
    let m3r: Vec<u16> = kr.outf[0].iter().map(|&v| to_bf(v)).collect();
    let m2e: Vec<u16> = res0.iter().zip(o).map(|(&r, &v)| to_bf(bf(r) + bf(v))).collect();
    let a = m3r == *o;
    let b = m2e == kr.res[0];
    if !(a && b) { println!("  {} mode consistency: bf(m3)==m0 {} m2==bf(res+m0) {}", name, a, b); }
    a && b
}

/// Real-shape test: random MLX-affine weights; CPU f64 ref on sampled rows; K-invariance; modes.
fn real_shape(g: &Gpu, r: usize, c: usize, seed: u64) -> (u32, bool, bool) {
    let mut rng = Rng::new(seed);
    let (w, s, b) = rand_q4(&mut rng, r, c);
    let x = rng.bf_n(8 * c, 1.0);
    let res0 = rng.bf_n(8 * r, 4.0);
    let bytes = w.len() * 4 + s.len() * 4 + 8 * c * 2 + 8 * r * 16 + (1 << 20);
    let mut ar = Ar::new(g, bytes + (4 << 20));
    let m = put_mat(&mut ar, r, c, &w, &s, &b);
    let kr = run_all_k(g, &mut ar, m, &x, &res0);
    let name = format!("{}x{}", r, c);
    let kinv = check_k_invariance(&name, &kr);
    let modes = check_modes(&name, &kr, &res0);
    // CPU ref: ~600 sampled rows (first, last, stride), all 8 tokens
    let step = (r / 600).max(1);
    let mut rows: Vec<usize> = (0..r).step_by(step).collect();
    rows.push(r - 1);
    let mut maxu = 0u32;
    let mut n1 = 0usize;
    for &row in &rows {
        for t in 0..8 {
            let v = dot_affine_f64(&w, &s, &b, c, row, &x[t * c..(t + 1) * c]);
            let u = ulp_bf(kr.out[0][t * r + row], f64_to_bf(v));
            if u > 0 { n1 += 1; }
            maxu = maxu.max(u);
        }
    }
    println!("  {:>12}: CPU-ref max ulp {} ({} of {} sampled outputs differ), K-invariant {}, modes {}",
             name, maxu, n1, rows.len() * 8, kinv, modes);
    (maxu, kinv, modes)
}

#[test]
fn gemv_real_shapes_cpu_ref_and_k_invariance() {
    let g = gpu();
    let shapes = [(16480, 5120), (5120, 6144), (5120, 17408), (17408, 5120), (248320, 5120)];
    let mut all_ok = true;
    for (i, &(r, c)) in shapes.iter().enumerate() {
        let (u, k, m) = real_shape(&g, r, c, 100 + i as u64);
        all_ok &= u <= 1 && k && m;
    }
    assert!(all_ok, "gemv: CPU ref > 1 ulp, K-invariance or mode consistency failed (see output)");
}

#[test]
fn gemv_mlx_vectors() {
    let Some(v) = vectors("gemv") else { return; };
    let g = gpu();
    // dequant convention check of the CPU helper vs MLX dequantize
    {
        let (w, s, b) = (v.u32s("deq.w"), v.u16s("deq.s"), v.u16s("deq.b"));
        let sh = v.shape("deq.y");
        let y = v.u16s("deq.y");
        let mut mine = Vec::new();
        for r in 0..sh[0] { mine.extend(deq_row(&w, &s, &b, sh[1], r).into_iter().map(to_bf)); }
        assert_eq!(mine, y, "CPU dequant != MLX dequantize");
        println!("  deq: CPU dequant == MLX bit-exact");
    }
    let mut ok = true;
    for case in ["q4_k5120", "q4_k6144", "q4_k17408"] {
        let n = |s: &str| format!("{}.{}", case, s);
        let (w, s, b, x) = (v.u32s(&n("w")), v.u16s(&n("s")), v.u16s(&n("b")), v.u16s(&n("x")));
        let (r, c) = (v.shape(&n("w"))[0], v.shape(&n("x"))[1]);
        let y1 = v.u16s(&n("y_k1"));
        let yk = v.u16s(&n("y"));
        let mut ar = Ar::new(&g, w.len() * 4 + s.len() * 4 + x.len() * 2 + 8 * r * 16 + (1 << 20));
        let m = put_mat(&mut ar, r, c, &w, &s, &b);
        let res0 = vec![0u16; 8 * r];
        let kr = run_all_k(&g, &mut ar, m, &x, &res0);
        let kinv = check_k_invariance(case, &kr);
        let (u1, n1) = cmp_bf(&kr.out[0], &y1);
        let (uk, nk) = cmp_bf(&kr.out[0], &yk);
        // also: CPU f64 ref vs MLX (how far MLX itself is from exact)
        let mut mu = 0u32;
        for t in 0..8 { for row in 0..r { mu = mu.max(ulp_bf(f64_to_bf(dot_affine_f64(&w, &s, &b, c, row, &x[t * c..(t + 1) * c])), y1[t * r + row])); } }
        println!("  {:>10}: engine vs MLX y_k1 max ulp {} ({} / {} differ); vs MLX y(qmm) {} ({}); MLX y_k1 vs exact {} ; K-inv {}",
                 case, u1, n1, y1.len(), uk, nk, mu, kinv);
        ok &= u1 <= 1 && kinv;
    }
    assert!(ok, "gemv vs MLX > 1 ulp or not K-invariant");
}

#[test]
fn swiglu_cpu_and_mlx() {
    let g = gpu();
    // ffn case: CPU mirror of the kernel's silu_mul chain vs MLX swiglu on given bf16 gate/up
    if let Some(v) = vectors("swiglu") {
        let (gt, up, y) = (v.u16s("ffn.gate"), v.u16s("ffn.up"), v.u16s("ffn.y"));
        let mine: Vec<u16> = gt.iter().zip(&up).map(|(&a, &b)| silu_mul(bf(a), bf(b))).collect();
        let (u, n) = cmp_bf(&mine, &y);
        println!("  ffn: CPU mirror of q4b_silu_mul vs MLX swiglu: max ulp {} ({} / {} differ)", u, n, y.len());
        assert!(u <= 1, "silu_mul chain != MLX swiglu");

        // fused case on the GPU
        let x = v.u16s("fused.x");
        let (r, c) = (v.shape("fused.g_w")[0], v.shape("fused.x")[1]);
        let kx = v.shape("fused.x")[0];
        let mut ar = Ar::new(&g, 8 << 20);
        let gm = put_mat(&mut ar, r, c, &v.u32s("fused.g_w"), &v.u16s("fused.g_s"), &v.u16s("fused.g_b"));
        let um = put_mat(&mut ar, r, c, &v.u32s("fused.u_w"), &v.u16s("fused.u_s"), &v.u16s("fused.u_b"));
        let xo = ar.alloc_put(&x);
        let out = ar.alloc(kx * r * 2);
        let gp = ar.alloc(kx * r * 2);
        let upo = ar.alloc(kx * r * 2);
        let mut per_k: Vec<Vec<u16>> = Vec::new();
        for k in [1usize, 2, 4] {
            ar.fill(out, kx * r * 2, 0xff);
            for t0 in (0..kx).step_by(k) { swiglu(&g, &ar, gm, um, k, xo + t0 * c * 2, out + t0 * r * 2); }
            per_k.push(ar.get(out, kx * r));
        }
        for t in 0..kx {
            gemv(&g, &ar, gm, 1, 0, xo + t * c * 2, gp + t * r * 2, 0, 0);
            gemv(&g, &ar, um, 1, 0, xo + t * c * 2, upo + t * r * 2, 0, 0);
        }
        let (gpv, upv) = (ar.get::<u16>(gp, kx * r), ar.get::<u16>(upo, kx * r));
        let (ug, _) = cmp_bf(&gpv, &v.u16s("fused.gp"));
        let (uu, _) = cmp_bf(&upv, &v.u16s("fused.up"));
        let chained: Vec<u16> = gpv.iter().zip(&upv).map(|(&a, &b)| silu_mul(bf(a), bf(b))).collect();
        let fused_eq_chain = chained == per_k[0];
        let (uy1, ny1) = cmp_bf(&per_k[0], &v.u16s("fused.y_k1"));
        let (uy, _) = cmp_bf(&per_k[0], &v.u16s("fused.y"));
        let kinv = per_k[1] == per_k[0] && per_k[2] == per_k[0];
        println!("  fused: gate GEMV vs MLX gp {} ulp, up {} ulp; swiglu == silu_mul(m0 gate, m0 up) {}; vs MLX y_k1 max ulp {} ({} differ), vs y {}; K-inv(1,2,4) {}",
                 ug, uu, fused_eq_chain, uy1, ny1, uy, kinv);
        assert!(fused_eq_chain && kinv, "fused swiglu not consistent / not K-invariant");
    }
    // K = 8 invariance at the real FFN shape (17408 x 5120), random weights
    let (r, c) = (17408, 5120);
    let mut rng = Rng::new(7);
    let (w1, s1, b1) = rand_q4(&mut rng, r, c);
    let (w2, s2, b2) = rand_q4(&mut rng, r, c);
    let x = rng.bf_n(8 * c, 1.0);
    let mut ar = Ar::new(&g, 2 * (w1.len() * 4 + s1.len() * 4) + 8 * r * 2 * 2 + 8 * c * 2 + (4 << 20));
    let gm = put_mat(&mut ar, r, c, &w1, &s1, &b1);
    let um = put_mat(&mut ar, r, c, &w2, &s2, &b2);
    let xo = ar.alloc_put(&x);
    let out = ar.alloc(8 * r * 2);
    let mut per_k = Vec::new();
    for k in KS {
        ar.fill(out, 8 * r * 2, 0xff);
        for t0 in (0..8).step_by(k) { swiglu(&g, &ar, gm, um, k.min(8 - t0), xo + t0 * c * 2, out + t0 * r * 2); }
        per_k.push(ar.get::<u16>(out, 8 * r));
    }
    let kinv = per_k.iter().all(|p| *p == per_k[0]);
    // vs the m0 GEMV chain
    let gp = ar.alloc(8 * r * 2);
    let up = ar.alloc(8 * r * 2);
    gemv(&g, &ar, gm, 8, 0, xo, gp, 0, 0);
    gemv(&g, &ar, um, 8, 0, xo, up, 0, 0);
    let chained: Vec<u16> = ar.get::<u16>(gp, 8 * r).iter().zip(ar.get::<u16>(up, 8 * r).iter()).map(|(&a, &b)| silu_mul(bf(a), bf(b))).collect();
    println!("  swiglu 17408x5120: K-invariant(1,2,4,8) {}; == silu_mul(m0 gate, m0 up) {}", kinv, chained == per_k[0]);
    assert!(kinv && chained == per_k[0]);
}

/// GB/s at the real shapes, K = 1 (decode) and K = 2 (MTP verify), MODE 0, random weights.
/// Enough weight copies are cycled to exceed ~256 MB per batch so the SLC can't hold them.
#[test]
fn gemv_bandwidth() {
    let g = gpu();
    let shapes = [("dn_in 16480x5120", 16480, 5120), ("attn o / dn out 5120x6144", 5120, 6144), ("down 5120x17408", 5120, 17408),
                  ("gate|up 17408x5120", 17408, 5120), ("lm_head 248320x5120", 248320, 5120)];
    for (label, r, c) in shapes {
        let mut rng = Rng::new(9);
        let (w, s, b) = rand_q4(&mut rng, r, c);
        let mbytes = w.len() * 4 + s.len() * 4;
        let copies = ((256usize << 20) + mbytes - 1) / mbytes;
        let mut ar = Ar::new(&g, copies * (mbytes + 1024) + 8 * c * 2 + 8 * r * 4 + (1 << 20));
        let ms: Vec<M> = (0..copies).map(|_| put_mat(&mut ar, r, c, &w, &s, &b)).collect();
        let x = ar.alloc_put(&rng.bf_n(8 * c, 1.0));
        let out = ar.alloc(8 * r * 2);
        let mut line = format!("  {:>28}: {:7.1} MB x{:<3}", label, mbytes as f64 / 1e6, copies);
        for k in [1usize, 2] {
            let reps = (copies * 4).max(8);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let cb = g.command_buffer();
                let e = Enc::new(&g, &cb, &ar.buf);
                for i in 0..reps {
                    let m = ms[i % copies];
                    e.pso(&format!("q4b_gemv_k{}_m0", k)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, x).b(4, out)
                        .u(5, r as u32).u(6, c as u32).b(7, out).b(8, out);
                    e.groups(((r + 3) / 4, 1), 64);
                }
                e.end();
                commit(&cb);
                wait(&cb).unwrap();
                best = best.min(gpu_ms(&cb) / reps as f64);
            }
            line += &format!("  K={}: {:7.3} ms {:6.1} GB/s", k, best, mbytes as f64 / best / 1e6);
        }
        println!("{}", line);
    }
}
