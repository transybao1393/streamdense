//! U6 self-test: engine/shaders/deltanet.metal (dn_conv, dn_prep, dn_recur_ck0/ck1).
//! - CPU reference of conv + silu, normalize_qk + gates, and the gated delta rule (same reduction order as the
//!   kernel) at the tiny dims (kh 2, vh 6) and the real dims (kh 16, vh 48), T in {1, 2, 8, 64}.
//! - T-token dispatch == T one-token dispatches, bit-exact (all outputs and both states).
//! - Checkpoint: the CK copy == the state after a 1-token run; role swap continues from the copy.
//! - MLX vectors (dn.small.*, dn.real.*) if engine/testdata/vectors exists.
//! Run: cargo test --offline -p streamdense-metal --test deltanet -- --nocapture --test-threads=1
mod b_util;
use b_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu, SdCtrl};

const D: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DnDims { kh: u32, vh: u32, conv: u32, prow: u32, b_off: u32, a_off: u32, layers: u32, pad: u32 }
fn dims(kh: usize, vh: usize, layers: usize) -> DnDims {
    let conv = 2 * kh * D + vh * D;
    let b_off = conv + vh * D;
    DnDims { kh: kh as u32, vh: vh as u32, conv: conv as u32, prow: (b_off + 2 * vh) as u32, b_off: b_off as u32,
             a_off: (b_off + vh) as u32, layers: layers as u32, pad: 0 }
}

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "deltanet"]).unwrap();
    for k in ["dn_conv", "dn_prep", "dn_recur_ck0", "dn_recur_ck1"] { g.pipeline(k).unwrap(); }
    g
}

// ------------------------------------------------------------------------------------------- CPU reference
fn sig_bf(x: f32) -> f32 {
    let e = rbf(x.abs().exp());
    let d = rbf(1.0 + e);
    let y = rbf(1.0 / d);
    if x < 0.0 { y } else { rbf(1.0 - y) }
}

/// conv + silu over T tokens; returns out [T][conv], updates st3 = [3][conv] in place.
fn cpu_conv(dd: &DnDims, p: &[u16], w: &[u16], st3: &mut [u16], t_n: usize) -> Vec<u16> {
    let c_n = dd.conv as usize; let prow = dd.prow as usize;
    let mut out = vec![0u16; t_n * c_n];
    for c in 0..c_n {
        let w4: Vec<f32> = (0..4).map(|j| bf(w[c * 4 + j])).collect();
        let (mut r0, mut r1, mut r2) = (st3[c], st3[c_n + c], st3[2 * c_n + c]);
        for t in 0..t_n {
            let x = p[t * prow + c];
            let mut acc = 0f32;
            acc += bf(r0) * w4[0]; acc += bf(r1) * w4[1]; acc += bf(r2) * w4[2]; acc += bf(x) * w4[3];
            let cv = rbf(acc);
            out[t * c_n + c] = to_bf(cv * sig_bf(cv));
            r0 = r1; r1 = r2; r2 = x;
        }
        st3[c] = r0; st3[c_n + c] = r1; st3[2 * c_n + c] = r2;
    }
    out
}

/// normalize_qk in place on cv [T][conv]; returns (g [T][vh], beta [T][vh]).
fn cpu_prep(dd: &DnDims, cv: &mut [u16], p: &[u16], a_log: &[u16], dt: &[u16], t_n: usize) -> (Vec<f32>, Vec<u16>) {
    let (kh, vh, c_n, prow) = (dd.kh as usize, dd.vh as usize, dd.conv as usize, dd.prow as usize);
    let rms_eps = f32::from_bits(0x320637bd);
    for t in 0..t_n {
        for h in 0..2 * kh {
            let col = h * D; // q heads then k heads are contiguous
            let x: Vec<f32> = (0..D).map(|i| bf(cv[t * c_n + col + i])).collect();
            // lane l: 4 sequential squares; simd_sum as a xor butterfly
            let mut lanes: Vec<f32> = (0..32).map(|l| { let mut a = 0f32; for i in 0..4 { a += x[4 * l + i] * x[4 * l + i]; } a }).collect();
            for s in [16usize, 8, 4, 2, 1] { let prev = lanes.clone(); for l in 0..32 { lanes[l] = prev[l] + prev[l ^ s]; } }
            let acc = lanes[0];
            let inv = (1.0f64 / ((acc / 128.0 + rms_eps) as f64).sqrt()) as f32;
            let sc = if h < kh { bf(0x3c00) } else { bf(0x3db5) };
            for i in 0..D { cv[t * c_n + col + i] = to_bf(rbf(x[i] * inv) * sc); }
        }
    }
    let mut g = vec![0f32; t_n * vh]; let mut beta = vec![0u16; t_n * vh];
    for t in 0..t_n {
        for h in 0..vh {
            let pr = &p[t * prow..];
            beta[t * vh + h] = to_bf(sig_bf(bf(pr[dd.b_off as usize + h])));
            let x = rbf(bf(pr[dd.a_off as usize + h]) + bf(dt[h]));
            let sp = if x.is_nan() { x } else {
                let (mx, mn) = (x.max(0.0), x.min(0.0));
                if mn == f32::NEG_INFINITY || mx == f32::INFINITY { mx } else {
                    let e = rbf(rbf(mn - mx).exp());
                    let xp1 = 1.0f32 + e;
                    let l = if xp1 == 1.0 { e } else { rbf(e * (xp1.ln() / (xp1 - 1.0))) };
                    rbf(mx + l)
                }
            };
            let ea = bf(a_log[h]).exp();
            g[t * vh + h] = (-ea * sp).exp();
        }
    }
    (g, beta)
}

/// gated delta rule over T tokens in the kernel's order; S = [vh][128][128] updated in place; returns y [T][vh][128].
fn cpu_recur(dd: &DnDims, cv: &[u16], g: &[f32], beta: &[u16], s: &mut [f32], t_n: usize) -> Vec<u16> {
    let (kh, vh, c_n) = (dd.kh as usize, dd.vh as usize, dd.conv as usize);
    let kd = kh * D;
    let mut y = vec![0u16; t_n * vh * D];
    for t in 0..t_n {
        for hv in 0..vh {
            let hk = hv / (vh / kh);
            let row = &cv[t * c_n..];
            let q: Vec<f32> = (0..D).map(|i| bf(row[hk * D + i])).collect();
            let k: Vec<f32> = (0..D).map(|i| bf(row[kd + hk * D + i])).collect();
            let gt = g[t * vh + hv]; let bt = bf(beta[t * vh + hv]);
            for dv in 0..D {
                let st = &mut s[(hv * D + dv) * D..(hv * D + dv + 1) * D];
                let dot = |st: &mut [f32], f: &dyn Fn(&mut [f32], usize, f32) -> f32| -> f32 {
                    // per lane (32 values): 8 partials of 4 sequential, local tree, then xor 1, 2 over 4 lanes
                    let mut lanes = [0f32; 4];
                    for (li, lane) in lanes.iter_mut().enumerate() {
                        let mut part = [0f32; 8];
                        for pb in 0..8 { let mut a = 0f32; for i in 0..4 { a = f(st, li * 32 + pb * 4 + i, a); } part[pb] = a; }
                        *lane = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
                    }
                    let l1 = [lanes[0] + lanes[1], lanes[1] + lanes[0], lanes[2] + lanes[3], lanes[3] + lanes[2]];
                    l1[0] + l1[2]
                };
                let kv = dot(st, &|st, e, a| { st[e] *= gt; st[e].mul_add(k[e], a) });
                let delta = (bf(row[2 * kd + hv * D + dv]) - kv) * bt;
                let o = dot(st, &|st, e, a| { st[e] = k[e].mul_add(delta, st[e]); st[e].mul_add(q[e], a) });
                y[(t * vh + hv) * D + dv] = to_bf(o);
            }
        }
    }
    y
}

// ------------------------------------------------------------------------------------------- GPU harness
struct Bufs {
    dd: DnDims,
    tmax: usize,
    layer: usize,
    p: usize, w: usize, st: usize, cv: usize, ctrl: usize, alog: usize, dt: usize, g: usize, beta: usize, s: usize, y: usize,
}
impl Bufs {
    fn new(ar: &mut Ar, dd: DnDims, tmax: usize, layer: usize) -> Bufs {
        let (c, vh) = (dd.conv as usize, dd.vh as usize);
        Bufs { dd, tmax, layer,
               p: ar.alloc(tmax * dd.prow as usize * 2), w: ar.alloc(c * 8), st: ar.alloc(2 * dd.layers as usize * 3 * c * 2),
               cv: ar.alloc(tmax * c * 2), ctrl: ar.alloc(72), alog: ar.alloc(vh * 2), dt: ar.alloc(vh * 2),
               g: ar.alloc(tmax * vh * 4), beta: ar.alloc(tmax * vh * 2), s: ar.alloc(2 * vh * D * D * 4),
               y: ar.alloc(tmax * vh * D * 2) }
    }
    fn conv_st(&self, role: usize) -> usize { self.st + ((role * self.dd.layers as usize + self.layer) * 3) * self.dd.conv as usize * 2 }
    fn s_copy(&self, role: usize) -> usize { self.s + role * self.dd.vh as usize * D * D * 4 }
    fn set_role(&self, ar: &Ar, role: u32) { ar.put(self.ctrl, &[SdCtrl { role, ..Default::default() }]); }
    fn conv(&self, e: &Enc, t0: usize, t_n: usize, ck: u32) {
        let dd = &self.dd;
        e.pso("dn_conv").b(0, self.p + t0 * dd.prow as usize * 2).b(1, self.w).b(2, self.st).b(3, self.cv + t0 * dd.conv as usize * 2)
            .b(4, self.ctrl).u(5, self.layer as u32).u(6, t_n as u32).u(7, ck).bytes(8, dd).threads((dd.conv as usize, 1), 256);
    }
    fn prep(&self, e: &Enc, t0: usize, t_n: usize) {
        let dd = &self.dd; let vh = dd.vh as usize;
        e.pso("dn_prep").b(0, self.cv + t0 * dd.conv as usize * 2).b(1, self.p + t0 * dd.prow as usize * 2).b(2, self.alog)
            .b(3, self.dt).b(4, self.g + t0 * vh * 4).b(5, self.beta + t0 * vh * 2).u(6, t_n as u32).bytes(7, dd)
            .groups((2 * dd.kh as usize + 1, t_n), 128);
    }
    fn recur(&self, e: &Enc, t0: usize, t_n: usize, ck: bool) {
        let dd = &self.dd; let vh = dd.vh as usize;
        e.pso(if ck { "dn_recur_ck1" } else { "dn_recur_ck0" }).b(0, self.cv + t0 * dd.conv as usize * 2).b(1, self.g + t0 * vh * 4)
            .b(2, self.beta + t0 * vh * 2).b(3, self.s).b(4, self.y + t0 * vh * D * 2).b(5, self.ctrl).u(6, t_n as u32).bytes(7, dd)
            .groups((4 * vh, 1), 128);
    }
}

fn run(gpu: &Gpu, ar: &Ar, f: impl Fn(&Enc)) {
    let cb = gpu.command_buffer();
    let e = Enc::new(gpu, &cb, &ar.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

/// Everything a chain run produces.
#[derive(PartialEq)]
struct Out { conv_out: Vec<u16>, cv: Vec<u16>, g: Vec<f32>, beta: Vec<u16>, y: Vec<u16>, st: Vec<u16>, s: Vec<f32> }

struct Inp { p: Vec<u16>, w: Vec<u16>, st3: Vec<u16>, alog: Vec<u16>, dt: Vec<u16>, s0: Vec<f32> }
fn rand_inputs(dd: &DnDims, t_n: usize, seed: u64) -> Inp {
    let mut r = Rng::new(seed);
    let (c, vh, prow) = (dd.conv as usize, dd.vh as usize, dd.prow as usize);
    let mut p = r.bf_n(t_n * prow, 1.0);
    for t in 0..t_n { for h in 0..vh { p[t * prow + dd.b_off as usize + h] = to_bf(1.5 * r.normal()); } }
    Inp { p, w: r.bf_n(c * 4, 0.5), st3: r.bf_n(3 * c, 1.0),
          alog: (0..vh).map(|_| to_bf((0.02 + 0.98 * r.unif()).ln())).collect(), dt: r.bf_n(vh, 0.5),
          s0: (0..vh * D * D).map(|_| 0.05 * r.normal()).collect() }
}

/// Load inputs into both state copies' slot `role` (other copy filled with a marker), run, read back.
/// split = T one-token dispatches (decode order conv, prep, recur per token) in one command buffer;
/// else one T-token dispatch per kernel, each in its own command buffer (conv_out snapshot before prep).
fn load(ar: &Ar, b: &Bufs, inp: &Inp, role: usize) {
    let (c, vh) = (b.dd.conv as usize, b.dd.vh as usize);
    ar.put(b.p, &inp.p); ar.put(b.w, &inp.w); ar.put(b.alog, &inp.alog); ar.put(b.dt, &inp.dt);
    ar.fill(b.st, 2 * b.dd.layers as usize * 3 * c * 2, 0x5a);
    ar.put(b.conv_st(role), &inp.st3);
    ar.fill(b.s, 2 * vh * D * D * 4, 0x7f);
    ar.put(b.s_copy(role), &inp.s0);
    ar.fill(b.cv, b.tmax * c * 2, 0); ar.fill(b.y, b.tmax * vh * D * 2, 0);
    b.set_role(ar, role as u32);
}
fn chain(gpu: &Gpu, ar: &Ar, b: &Bufs, inp: &Inp, t_n: usize, split: bool, role: usize) -> Out {
    let (c, vh) = (b.dd.conv as usize, b.dd.vh as usize);
    assert!(t_n <= b.tmax);
    load(ar, b, inp, role);
    let conv_out;
    if split {
        run(gpu, ar, |e| for t in 0..t_n { b.conv(e, t, 1, !0); b.prep(e, t, 1); b.recur(e, t, 1, false); });
        conv_out = Vec::new();
    } else {
        run(gpu, ar, |e| b.conv(e, 0, t_n, !0));
        conv_out = ar.get::<u16>(b.cv, t_n * c);
        run(gpu, ar, |e| b.prep(e, 0, t_n));
        run(gpu, ar, |e| b.recur(e, 0, t_n, false));
    }
    Out { conv_out, cv: ar.get(b.cv, t_n * c), g: ar.get(b.g, t_n * vh), beta: ar.get(b.beta, t_n * vh),
          y: ar.get(b.y, t_n * vh * D), st: ar.get(b.conv_st(role), 3 * c), s: ar.get(b.s_copy(role), vh * D * D) }
}

fn arena_for(gpu: &Gpu, dd: &DnDims, tmax: usize) -> Ar {
    let c = dd.conv as usize; let vh = dd.vh as usize;
    let sz = tmax * (dd.prow as usize * 2 + c * 2 + vh * 6 + vh * D * 2) + c * 8 + 2 * dd.layers as usize * 3 * c * 2
        + 2 * vh * D * D * 4;
    Ar::new(gpu, sz + (1 << 20))
}

fn check_cpu_and_split(kh: usize, vh: usize) {
    let gpu = gpu();
    let dd = dims(kh, vh, 3);
    let tmax = 64;
    let mut ar = arena_for(&gpu, &dd, tmax);
    let b = Bufs::new(&mut ar, dd, tmax, 1);
    for &t_n in &[1usize, 2, 8, 64] {
        let inp = rand_inputs(&dd, t_n, 1000 + t_n as u64 + kh as u64);
        let o = chain(&gpu, &ar, &b, &inp, t_n, false, 0);
        // CPU reference, stage-isolated (each stage fed the GPU output of the previous one)
        let mut st3 = inp.st3.clone();
        let co = cpu_conv(&dd, &inp.p, &inp.w, &mut st3, t_n);
        let (cu, cn) = cmp_bf(&co, &o.conv_out);
        let (su, _) = cmp_bf(&st3, &o.st);
        let mut cv = o.conv_out.clone();
        let (g, beta) = cpu_prep(&dd, &mut cv, &inp.p, &inp.alog, &inp.dt, t_n);
        let (pu, pn) = cmp_bf(&cv, &o.cv);
        let (gu, gn, _) = cmp_f32(&g, &o.g);
        let (bu, _) = cmp_bf(&beta, &o.beta);
        let mut s = inp.s0.clone();
        let y = cpu_recur(&dd, &o.cv, &o.g, &o.beta, &mut s, t_n);
        let (yu, yn) = cmp_bf(&y, &o.y);
        let (sul, sn, sad) = cmp_f32(&s, &o.s);
        println!("kh{} vh{} T{:2}: conv {}ulp/{} diff, state {}; qk {}ulp/{}; g {}ulp/{}; beta {}ulp; y {}ulp/{} of {}; S {}ulp/{} (abs {:.2e})",
                 kh, vh, t_n, cu, cn, su, pu, pn, gu, gn, bu, yu, yn, y.len(), sul, sn, sad);
        assert_eq!(su, 0, "conv state");
        assert!(cu <= 1 && pu <= 1 && bu <= 1 && gu <= 4, "conv/prep vs CPU");
        // CPU recur uses fma where Metal contracts (safe math still contracts a*b+c): bit-exact expected
        assert_eq!((yu, sul), (0, 0), "y / S vs CPU");
        // T one-token dispatches == one T-token dispatch, bit-exact
        let os = chain(&gpu, &ar, &b, &inp, t_n, true, 0);
        let same = o.cv == os.cv && o.g == os.g && o.beta == os.beta && o.y == os.y && o.st == os.st
            && o.s.iter().zip(&os.s).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("kh{} vh{} T{:2}: T-dispatch == {}x T=1 bit-exact: {}", kh, vh, t_n, t_n, same);
        assert!(same, "split mismatch at T={}", t_n);
        // role 1 gives the same result as role 0 (state copy selected by ctrl.role)
        if t_n == 8 {
            let o1 = chain(&gpu, &ar, &b, &inp, t_n, false, 1);
            assert!(o1.y == o.y && o1.st == o.st && o1.s.iter().zip(&o.s).all(|(a, b)| a.to_bits() == b.to_bits()), "role 1");
            println!("kh{} vh{} T 8: role 1 == role 0: true", kh, vh);
        }
    }
}

#[test]
fn dn_cpu_ref_and_split_tiny() { check_cpu_and_split(2, 6); }
#[test]
fn dn_cpu_ref_and_split_real() { check_cpu_and_split(16, 48); }

/// Verify pass with checkpoint (T=2, ck=0 / dn_recur_ck1) from role r: copy r^1 must equal the state after a
/// T=1 run; reject = set role r^1 and run token 1 alone -> must equal the T=2 result (accept path).
fn check_ck(kh: usize, vh: usize) {
    let gpu = gpu();
    let dd = dims(kh, vh, 2);
    let mut ar = arena_for(&gpu, &dd, 2);
    let b = Bufs::new(&mut ar, dd, 2, 1);
    let (c, n_s) = (dd.conv as usize, vh * D * D);
    for role in [0usize, 1] {
        let inp = rand_inputs(&dd, 2, 77 + role as u64);
        let one = chain(&gpu, &ar, &b, &inp, 1, false, role); // reference state after token 0
        let two = chain(&gpu, &ar, &b, &inp, 2, false, role); // accept path reference (no CK)
        // verify pass with CK
        load(&ar, &b, &inp, role);
        run(&gpu, &ar, |e| { b.conv(e, 0, 2, 0); b.prep(e, 0, 2); b.recur(e, 0, 2, true); });
        let y_ck: Vec<u16> = ar.get(b.y, 2 * vh * D);
        let st_cur: Vec<u16> = ar.get(b.conv_st(role), 3 * c);
        let s_cur: Vec<f32> = ar.get(b.s_copy(role), n_s);
        let st_ck: Vec<u16> = ar.get(b.conv_st(role ^ 1), 3 * c);
        let s_ck: Vec<f32> = ar.get(b.s_copy(role ^ 1), n_s);
        let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let ck_ok = st_ck == one.st && bits(&s_ck, &one.s);
        let acc_ok = st_cur == two.st && bits(&s_cur, &two.s) && y_ck == two.y;
        // reject: role -> role ^ 1, run token 1 alone, its state must equal the accept state
        b.set_role(&ar, (role ^ 1) as u32);
        run(&gpu, &ar, |e| { b.conv(e, 1, 1, !0); b.prep(e, 1, 1); b.recur(e, 1, 1, false); });
        let st_sw: Vec<u16> = ar.get(b.conv_st(role ^ 1), 3 * c);
        let s_sw: Vec<f32> = ar.get(b.s_copy(role ^ 1), n_s);
        let y1: Vec<u16> = ar.get(b.y + vh * D * 2, vh * D);
        let sw_ok = st_sw == two.st && bits(&s_sw, &two.s) && y1 == two.y[vh * D..];
        let untouched = ar.get::<u16>(b.conv_st(role), 3 * c) == st_cur && bits(&ar.get::<f32>(b.s_copy(role), n_s), &s_cur);
        println!("kh{} vh{} role {}: CK copy == T=1 state {}; CK-run current == T=2 {}; role swap + token 1 == T=2 {}; old copy untouched {}",
                 kh, vh, role, ck_ok, acc_ok, sw_ok, untouched);
        assert!(ck_ok && acc_ok && sw_ok && untouched);
    }
}

#[test]
fn dn_checkpoint_tiny() { check_ck(2, 6); }
#[test]
fn dn_checkpoint_real() { check_ck(16, 48); }

// ------------------------------------------------------------------------------------------- MLX vectors
fn mlx_case(v: &St, tag: &str) {
    let qs = v.shape(&format!("{}.q", tag)); // [1, T, kh, 128]
    let (t_n, kh) = (qs[1], qs[2]);
    let vh = v.shape(&format!("{}.v", tag))[2];
    let dd = dims(kh, vh, 1);
    let (c, prow, kd) = (dd.conv as usize, dd.prow as usize, kh * D);
    let gpu = gpu();
    let mut ar = arena_for(&gpu, &dd, t_n);
    let b = Bufs::new(&mut ar, dd, t_n, 0);
    let ten = |n: &str| v.u16s(&format!("{}.{}", tag, n));
    // conv: p rows = qkv | z (0) | b | a
    let qkv = ten("qkv");
    let (bb, aa) = (ten("b"), ten("a"));
    let mut p = vec![0u16; t_n * prow];
    for t in 0..t_n {
        p[t * prow..t * prow + c].copy_from_slice(&qkv[t * c..(t + 1) * c]);
        for h in 0..vh { p[t * prow + dd.b_off as usize + h] = bb[t * vh + h]; p[t * prow + dd.a_off as usize + h] = aa[t * vh + h]; }
    }
    ar.put(b.p, &p); ar.put(b.w, &ten("conv_w")); ar.put(b.conv_st(0), &ten("conv_state"));
    ar.put(b.alog, &ten("A_log")); ar.put(b.dt, &ten("dt_bias"));
    b.set_role(&ar, 0);
    run(&gpu, &ar, |e| b.conv(e, 0, t_n, !0));
    let (cu, cn) = cmp_bf(&ar.get::<u16>(b.cv, t_n * c), &ten("conv_out"));
    let (su, _) = cmp_bf(&ar.get::<u16>(b.conv_st(0), 3 * c), &ten("conv_state_out"));
    println!("MLX dn.{} conv T{}: max {} ulp ({} of {} differ); state_out {} ulp", tag, t_n, cu, cn, t_n * c, su);
    // prep from MLX conv_out
    ar.put(b.cv, &ten("conv_out"));
    run(&gpu, &ar, |e| b.prep(e, 0, t_n));
    let cvg: Vec<u16> = ar.get(b.cv, t_n * c);
    let (mq, mk, mv) = (ten("q"), ten("k"), ten("v"));
    let mut qd = (0u32, 0usize); let mut kdf = (0u32, 0usize); let mut vd = 0u32;
    for t in 0..t_n {
        let r = &cvg[t * c..];
        let a = cmp_bf(&r[..kd], &mq[t * kd..(t + 1) * kd]); qd = (qd.0.max(a.0), qd.1 + a.1);
        let a = cmp_bf(&r[kd..2 * kd], &mk[t * kd..(t + 1) * kd]); kdf = (kdf.0.max(a.0), kdf.1 + a.1);
        vd = vd.max(cmp_bf(&r[2 * kd..c], &mv[t * vh * D..(t + 1) * vh * D]).0);
    }
    let (gu, gn, _) = cmp_f32(&ar.get::<f32>(b.g, t_n * vh), &v.f32s(&format!("{}.g", tag)));
    let (bu, bn) = cmp_bf(&ar.get::<u16>(b.beta, t_n * vh), &ten("beta"));
    assert_eq!((cu, su), (0, 0), "conv vs MLX");
    println!("MLX dn.{} prep: q {} ulp ({} diff), k {} ulp ({} diff), v {} ulp, g {} f32-ulp ({} diff), beta {} ulp ({} diff)",
             tag, qd.0, qd.1, kdf.0, kdf.1, vd, gu, gn, bu, bn);
    assert_eq!((qd.0, kdf.0, vd, gu, bu), (0, 0, 0, 0, 0), "prep vs MLX");
    if !v.has(&format!("{}.S_in", tag)) {
        println!("MLX dn.{} recur: no vectors (generated without --big)", tag);
        return;
    }
    // recur from MLX q, k, v, g, beta
    let mut cvm = vec![0u16; t_n * c];
    for t in 0..t_n {
        cvm[t * c..t * c + kd].copy_from_slice(&mq[t * kd..(t + 1) * kd]);
        cvm[t * c + kd..t * c + 2 * kd].copy_from_slice(&mk[t * kd..(t + 1) * kd]);
        cvm[t * c + 2 * kd..(t + 1) * c].copy_from_slice(&mv[t * vh * D..(t + 1) * vh * D]);
    }
    let s_in = v.f32s(&format!("{}.S_in", tag));
    let mut res = Vec::new();
    for split in [false, true] {
        ar.put(b.cv, &cvm); ar.put(b.g, &v.f32s(&format!("{}.g", tag))); ar.put(b.beta, &ten("beta"));
        ar.put(b.s_copy(0), &s_in); b.set_role(&ar, 0);
        if split { run(&gpu, &ar, |e| for t in 0..t_n { b.recur(e, t, 1, t == 0); }); }
        else { run(&gpu, &ar, |e| b.recur(e, 0, t_n, true)); }
        let y: Vec<u16> = ar.get(b.y, t_n * vh * D);
        let s: Vec<f32> = ar.get(b.s_copy(0), vh * D * D);
        let s1: Vec<f32> = ar.get(b.s_copy(1), vh * D * D);
        let (yu, yn) = cmp_bf(&y, &ten(if split { "y_step" } else { "y" }));
        let (su, sn, sa) = cmp_f32(&s, &v.f32s(&format!("{}.S", tag)));
        let (s1u, s1n, s1a) = cmp_f32(&s1, &v.f32s(&format!("{}.S1", tag)));
        println!("MLX dn.{} recur {}: y {} ulp ({} of {} differ); S {} f32-ulp ({} diff, abs {:.2e}); S1 (CK copy) {} f32-ulp ({} diff, abs {:.2e})",
                 tag, if split { "T=1 x T (vs y_step)" } else { "T-dispatch (vs y)" }, yu, yn, y.len(), su, sn, sa, s1u, s1n, s1a);
        assert_eq!((yu, su, s1u), (0, 0, 0), "recur vs MLX");
        res.push((y, s));
    }
    println!("MLX dn.{} recur: GPU T-dispatch == GPU T=1 x T: {}", tag, res[0] == res[1]);
    assert!(res[0] == res[1]);
}

#[test]
fn dn_mlx_vectors() {
    let Some(v) = vectors("dn") else { return };
    mlx_case(&v, "small");
    mlx_case(&v, "real");
}
