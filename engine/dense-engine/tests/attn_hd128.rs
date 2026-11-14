//! Self-test of shaders/attn_hd128.metal (attn128_rope_invf, attn128_prep_{b16,q8}, attn128_partial_k{1..8}_{b16,q8},
//! attn128_reduce, attn128_reduce_gate) against a CPU f32 reference, at Qwen dense shapes (head dim 128, full rotary,
//! rope_theta 1e6, GQA).
//! - prep: invf vs CPU exp2 (<= 2 f32 ulp); qr and the appended K/V vs a CPU RMSNorm (kernel summation order) + RoPE
//!   (f64 cos/sin): <= 1 bf16 ulp (near-cancelled |ref| < 1/64: abs <= 1e-6); q8: codes vs CPU quantization of the
//!   CPU K (|diff| <= 1), V codes + scales exact.
//! - attention: CPU f32 softmax attention (sequential sums, libm exp) over the GPU's qr and the GPU's (dequantized)
//!   cache, so the q8 cases test the kernel arithmetic, not the quantization error.
//!   Tolerance: <= 2 bf16 ulp; outputs with |ref| < 1/64 (near-cancelled sums) are judged by abs err <= 2e-5 instead.
//! - K-invariance: a K-token pass == K one-token passes, bit-exact (also with one spare, always-empty chunk).
//!
//! Run: cargo test --release --test attn_hd128 -- --nocapture --test-threads=1   (inside engine/dense-engine)
use dense_engine::gpu::{run, Arena, Enc, Gpu, SdCtrl};
use dense_engine::SHADER_SOURCE;

const HD: usize = 128;
const RD: usize = 128; // full rotary
const CH: usize = 256;
const EPS: f32 = 1e-6;
const THETA: f32 = 1e6;
const SCALE: f32 = 0.08838834764831845; // 128^-0.5 in f32, as the kernel
const CTRL_STRIDE: usize = 256;

fn gpu() -> Gpu {
    let t = std::time::Instant::now();
    let mut g = Gpu::from_sources(&SHADER_SOURCE).unwrap();
    for n in ["attn128_rope_invf", "attn128_reduce", "attn128_reduce_gate"] { g.pipeline(n).unwrap(); }
    for q in ["b16", "q8"] {
        g.pipeline(&format!("attn128_prep_{}", q)).unwrap();
        for k in 1..=8 { g.pipeline(&format!("attn128_partial_k{}_{}", k, q)).unwrap(); }
    }
    println!("library + 21 pipelines: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    g
}
fn log2base() -> f32 { THETA.log2() }

// ------------------------------------------------------------------------------------------- bf16 / f16 / rng
fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
fn to_bf(f: f32) -> u16 {
    let mut u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 { return ((u >> 16) | 0x0040) as u16; }
    u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (u >> 16) as u16
}
fn rbf(f: f32) -> f32 { bf(to_bf(f)) }
fn key16(u: u16) -> i32 { if u >= 0x8000 { 0x8000 - u as i32 } else { u as i32 } }
fn ulp_bf(a: u16, b: u16) -> u32 { (key16(a) - key16(b)).unsigned_abs() }
fn ulp_f32(a: f32, b: f32) -> u64 {
    let k = |f: f32| { let u = f.to_bits(); if u >= 0x8000_0000 { 0x8000_0000i64 - u as i64 } else { u as i64 } };
    (k(a) - k(b)).unsigned_abs()
}
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    if e == 0 { let v = m as f32 * (2f32).powi(-24); return if s != 0 { -v } else { v }; }
    if e == 31 { return f32::from_bits(s | 0x7f80_0000 | (m << 13)); }
    f32::from_bits(s | (((e - 15 + 127) as u32) << 23) | (m << 13))
}
/// Round to nearest even f16 by search over the representable neighbours (values here are small and normal).
fn f32_to_f16(f: f32) -> u16 {
    let s = ((f.to_bits() >> 16) & 0x8000) as u16;
    let a = f.abs();
    if a.is_nan() { return s | 0x7e00; }
    if a >= 65520.0 { return s | 0x7c00; }
    let (mut lo, mut hi): (u16, u16) = (0, 0x7bff);
    while lo < hi { let mid = (lo + hi + 1) / 2; if f16_to_f32(mid) <= a { lo = mid; } else { hi = mid - 1; } }
    if lo == 0x7bff { return s | lo; }
    let (d0, d1) = (a as f64 - f16_to_f32(lo) as f64, f16_to_f32(lo + 1) as f64 - a as f64);
    s | if d0 < d1 || (d0 == d1 && lo & 1 == 0) { lo } else { lo + 1 }
}
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng { Rng(s.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1) }
    fn u64(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unif(&mut self) -> f32 { (self.u64() >> 40) as f32 / (1u64 << 24) as f32 }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.unif().max(1e-7), self.unif());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn bf_n(&mut self, n: usize, std: f32) -> Vec<u16> { (0..n).map(|_| to_bf(std * self.normal())).collect() }
}

// ------------------------------------------------------------------------------------------- CPU reference
fn sig_bf(x: f32) -> f32 {
    let e = rbf(x.abs().exp());
    let y = rbf(1.0 / rbf(1.0 + e));
    if x < 0.0 { y } else { rbf(1.0 - y) }
}
fn cpu_invf() -> Vec<f32> { (0..RD / 2).map(|i| (-(i as f32 / (RD / 2) as f32) * log2base()).exp2()).collect() }

/// [RMSNorm (kernel order: lanes 0..31 sum 4 dims each, a 32-lane butterfly, then the 4 simdgroup partials in
/// order)] + RoPE at pos.
fn cpu_norm_rope(x: &[u16], w: Option<&[u16]>, pos: u32, invf: &[f32]) -> Vec<u16> {
    let xf: Vec<f32> = x.iter().map(|&h| bf(h)).collect();
    let n: Vec<f32> = match w {
        None => xf.clone(),
        Some(w) => {
            let mut l: Vec<f32> = (0..32).map(|d| { let mut a = 0f32; for i in 0..4 { a += xf[4 * d + i] * xf[4 * d + i]; } a }).collect();
            for s in [16usize, 8, 4, 2, 1] { let p = l.clone(); for i in 0..32 { l[i] = p[i] + p[i ^ s]; } }
            let tot = ((l[0] + 0.0) + 0.0) + 0.0;
            let r = (1.0f64 / ((tot / HD as f32 + EPS) as f64).sqrt()) as f32;
            (0..HD).map(|d| rbf(bf(w[d]) * rbf(xf[d] * r))).collect()
        }
    };
    let h = RD / 2;
    (0..HD).map(|d| {
        if d >= RD { return to_bf(n[d]); }
        let i = if d < h { d } else { d - h };
        let th = pos as f32 * invf[i];
        let (c, s) = ((th as f64).cos() as f32, (th as f64).sin() as f32);
        let (x1, x2) = (n[i], n[i + h]);
        to_bf(if d < h { x1 * c - x2 * s } else { x1 * s + x2 * c })
    }).collect()
}
/// q8 group quantization as attn128_prep: (codes [32], scale f16, min f16).
fn cpu_q8(x: &[u16]) -> (Vec<u8>, u16, u16) {
    let xf: Vec<f32> = x.iter().map(|&h| bf(h)).collect();
    let lo = xf.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = xf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let (sc16, lo16) = (f32_to_f16((hi - lo).max(1e-8) / 255.0), f32_to_f16(lo));
    let scf = if f16_to_f32(sc16) > 0.0 { f16_to_f32(sc16) } else { 1.0 };
    (xf.iter().map(|&v| ((v - f16_to_f32(lo16)) / scf).round_ties_even().clamp(0.0, 255.0) as u8).collect(), sc16, lo16)
}
/// CPU f32 attention for the query at position pos over keys 0..=pos. Returns (unrounded o / l, bf16 output).
fn cpu_attn(kf: &[f32], vf: &[f32], q: &[u16], pos: usize, gate: Option<&[u16]>) -> (Vec<f32>, Vec<u16>) {
    let n = pos + 1;
    let qf: Vec<f32> = q.iter().map(|&h| SCALE * bf(h)).collect();
    let s: Vec<f32> = (0..n).map(|j| { let mut a = 0f32; for d in 0..HD { a += qf[d] * kf[j * HD + d]; } a }).collect();
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
    let l: f32 = p.iter().sum();
    let mut o = vec![0f32; HD];
    for j in 0..n { for d in 0..HD { o[d] += p[j] * vf[j * HD + d]; } }
    let r: Vec<f32> = o.iter().map(|x| x / l).collect();
    let out = (0..HD).map(|d| match gate { None => to_bf(r[d]), Some(g) => to_bf(rbf(r[d]) * sig_bf(bf(g[d]))) }).collect();
    (r, out)
}

// ------------------------------------------------------------------------------------------- GPU harness
#[derive(Clone, Copy, Debug)]
struct Shape { nh: usize, nkv: usize, qkn: bool, gated: bool, q8: bool }
impl Shape {
    fn qs(&self) -> usize { if self.gated { 2 * HD } else { HD } }
    fn p(&self) -> usize { self.nh * self.qs() + 2 * self.nkv * HD }
    fn sfx(&self) -> &'static str { if self.q8 { "q8" } else { "b16" } }
    fn flags(&self) -> u32 { self.qkn as u32 | ((self.gated as u32) << 1) }
}
struct Bufs {
    s: Shape, maxctx: usize, nch: usize,
    p: usize, qn: usize, kn: usize, qr: usize, kc: usize, vc: usize, ksb: usize, vsb: usize, invf: usize, ctrl: usize,
    po: usize, pml: usize, ao: usize,
}
impl Bufs {
    fn arena_size(s: Shape, tmax: usize, maxctx: usize, nch: usize, nctrl: usize) -> usize {
        let es = if s.q8 { 1 } else { 2 };
        tmax * s.p() * 2 + 2 * HD * 2 + tmax * s.nh * HD * 4 + 2 * s.nkv * maxctx * (HD * es + 16) + RD * 2
            + nctrl * CTRL_STRIDE + tmax * s.nh * nch * (HD * 4 + 8) + (1 << 16)
    }
    fn new(ar: &mut Arena, s: Shape, tmax: usize, maxctx: usize, nch: usize, nctrl: usize) -> Bufs {
        let es = if s.q8 { 1 } else { 2 };
        let ng = HD / 32;
        Bufs { s, maxctx, nch,
               p: ar.alloc(tmax * s.p() * 2), qn: ar.alloc(HD * 2), kn: ar.alloc(HD * 2), qr: ar.alloc(tmax * s.nh * HD * 2),
               kc: ar.alloc(s.nkv * maxctx * HD * es), vc: ar.alloc(s.nkv * maxctx * HD * es),
               ksb: ar.alloc(s.nkv * maxctx * ng * 4), vsb: ar.alloc(s.nkv * maxctx * ng * 4), invf: ar.alloc(RD / 2 * 4),
               ctrl: ar.alloc(nctrl * CTRL_STRIDE), po: ar.alloc(tmax * s.nh * nch * HD * 4), pml: ar.alloc(tmax * s.nh * nch * 8),
               ao: ar.alloc(tmax * s.nh * HD * 2) }
    }
    fn set_pos(&self, ar: &Arena, slot: usize, pos: u32) { ar.put(self.ctrl + slot * CTRL_STRIDE, &[SdCtrl { pos, ..Default::default() }]); }
    fn invf(&self, e: &Enc) { e.pso("attn128_rope_invf").b(0, self.invf).f(1, log2base()).threads((RD / 2, 1), RD / 2); }
    /// prep + partial + reduce for k tokens starting at row t0 (ctrl slot `slot` holds pos), nch chunks.
    fn pass(&self, e: &Enc, slot: usize, t0: usize, k: usize, nch: usize) {
        let (s, nh) = (self.s, self.s.nh);
        assert!(nch <= self.nch);
        let ctrl = self.ctrl + slot * CTRL_STRIDE;
        let (p, qr) = (self.p + t0 * s.p() * 2, self.qr + t0 * nh * HD * 2);
        e.pso(&format!("attn128_prep_{}", s.sfx())).b(0, p).b(1, self.qn).b(2, self.kn).b(3, qr)
            .b(4, self.kc).b(5, self.vc).b(6, self.ksb).b(7, self.vsb).b(8, self.invf).b(9, ctrl)
            .u(10, self.maxctx as u32).f(11, EPS).u(12, s.nkv as u32).u(13, RD as u32).u(14, s.flags())
            .groups((nh + s.nkv, k), HD);
        let rs = nh * nch; // po / pml rows per token for this nch
        let (po, pml) = (self.po + t0 * rs * HD * 4, self.pml + t0 * rs * 8);
        e.pso(&format!("attn128_partial_k{}_{}", k, s.sfx())).b(0, qr).b(1, self.kc).b(2, self.vc)
            .b(3, self.ksb).b(4, self.vsb).b(5, po).b(6, pml).b(7, ctrl).u(8, self.maxctx as u32).u(9, (nh / s.nkv) as u32)
            .groups((nch, nh), 128);
        e.pso(if s.gated { "attn128_reduce_gate" } else { "attn128_reduce" }).b(0, po).b(1, pml).b(2, p)
            .b(3, self.ao + t0 * nh * HD * 2).u(4, nch as u32).u(5, s.p() as u32).groups((nh, k), HD / 4);
    }
    /// Random cache rows 0..n for every kv head.
    fn fill_cache(&self, ar: &Arena, n: usize, r: &mut Rng) {
        let ng = HD / 32;
        for kvh in 0..self.s.nkv {
            if self.s.q8 {
                let qk: Vec<u8> = (0..n * HD).map(|_| (r.u64() & 0xff) as u8).collect();
                let qv: Vec<u8> = (0..n * HD).map(|_| (r.u64() & 0xff) as u8).collect();
                ar.put(self.kc + kvh * self.maxctx * HD, &qk);
                ar.put(self.vc + kvh * self.maxctx * HD, &qv);
                let sb = |r: &mut Rng| -> Vec<u16> {
                    (0..n * ng).flat_map(|_| [f32_to_f16((4.0 + 2.0 * r.unif()) / 255.0), f32_to_f16(-2.0 - r.unif())]).collect()
                };
                let (ks, vs) = (sb(r), sb(r));
                ar.put(self.ksb + kvh * self.maxctx * ng * 4, &ks);
                ar.put(self.vsb + kvh * self.maxctx * ng * 4, &vs);
            } else {
                ar.put(self.kc + kvh * self.maxctx * HD * 2, &r.bf_n(n * HD, 1.0));
                ar.put(self.vc + kvh * self.maxctx * HD * 2, &r.bf_n(n * HD, 1.0));
            }
        }
    }
    /// Dequantized (f32) K and V of kv head kvh, keys 0..n, from the GPU cache.
    fn load_kv(&self, ar: &Arena, kvh: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let ng = HD / 32;
        let (mut kf, mut vf) = (vec![0f32; n * HD], vec![0f32; n * HD]);
        if self.s.q8 {
            for (base, sb, out) in [(self.kc, self.ksb, &mut kf), (self.vc, self.vsb, &mut vf)] {
                let q: Vec<u8> = ar.get(base + kvh * self.maxctx * HD, n * HD);
                let s: Vec<u16> = ar.get(sb + kvh * self.maxctx * ng * 4, n * ng * 2);
                for j in 0..n { for d in 0..HD {
                    let g = j * ng + d / 32;
                    out[j * HD + d] = f16_to_f32(s[2 * g]) * q[j * HD + d] as f32 + f16_to_f32(s[2 * g + 1]);
                } }
            }
        } else {
            let kk: Vec<u16> = ar.get(self.kc + kvh * self.maxctx * HD * 2, n * HD);
            let vv: Vec<u16> = ar.get(self.vc + kvh * self.maxctx * HD * 2, n * HD);
            for i in 0..n * HD { kf[i] = bf(kk[i]); vf[i] = bf(vv[i]); }
        }
        (kf, vf)
    }
}

#[derive(Default, Debug)]
struct Stats { prep_bad: usize, ao_rel: f32, qr_ulp: u32, k_ulp: u32, v_ulp: u32, kq8_code: u32, vq8_bad: u32, ao_ulp: u32, ao_abs: f32, ao_diff: usize, n: usize, fail: usize }

/// One case: ctx keys after the pass (pos = ctx - k), k-token pass. Returns the stats; asserts the tolerances.
fn case(gpu: &Gpu, s: Shape, ctx: usize, k: usize, seed: u64) -> Stats {
    let maxctx = ctx + 8;
    let nch = (ctx + CH - 1) / CH;
    let nch_max = nch + 1;
    let mut ar = Arena::new(gpu, Bufs::arena_size(s, k, maxctx, nch_max, k + 1)).unwrap();
    let b = Bufs::new(&mut ar, s, k, maxctx, nch_max, k + 1);
    let mut r = Rng::new(seed);
    let qn: Vec<u16> = (0..HD).map(|_| to_bf(0.8 + 0.8 * r.unif())).collect();
    let kn: Vec<u16> = (0..HD).map(|_| to_bf(0.8 + 0.8 * r.unif())).collect();
    ar.put(b.qn, &qn);
    ar.put(b.kn, &kn);
    let pos = ctx - k;
    let pr = r.bf_n(k * s.p(), 1.5);
    let cseed = r.u64();
    ar.put(b.p, &pr);
    b.fill_cache(&ar, pos, &mut Rng::new(cseed));
    b.set_pos(&ar, 0, pos as u32);
    run(gpu, &ar, |e| { b.invf(e); b.pass(e, 0, 0, k, nch); }).unwrap();
    let invf: Vec<f32> = ar.get(b.invf, RD / 2);
    let iu = invf.iter().zip(cpu_invf()).map(|(a, c)| ulp_f32(*a, c)).max().unwrap();
    assert!(iu <= 2, "invf {} f32 ulp", iu);
    let qr: Vec<u16> = ar.get(b.qr, k * s.nh * HD);
    let ao: Vec<u16> = ar.get(b.ao, k * s.nh * HD);
    let es = if s.q8 { 1 } else { 2 };
    let kc: Vec<u8> = ar.get(b.kc, s.nkv * maxctx * HD * es);
    let vc: Vec<u8> = ar.get(b.vc, s.nkv * maxctx * HD * es);
    let vsb: Vec<u16> = ar.get(b.vsb, s.nkv * maxctx * (HD / 32) * 2);
    let kvs: Vec<(Vec<f32>, Vec<f32>)> = (0..s.nkv).map(|kv| b.load_kv(&ar, kv, ctx)).collect();
    let (qnw, knw) = if s.qkn { (Some(&qn[..]), Some(&kn[..])) } else { (None, None) };
    let u16at = |v: &[u8], i: usize| u16::from_le_bytes([v[2 * i], v[2 * i + 1]]);
    let mut st = Stats::default();
    // prep tolerance: <= 1 bf16 ulp, or a near-cancelled RoPE output (|ref| < 1/64) within 1e-6 abs (fast::cos/sin vs
    // f64 trig, ~5e-8 abs observed)
    let pchk = |r: u16, g: u16, bad: &mut usize| -> u32 {
        let u = ulp_bf(r, g);
        if u > 1 && !(bf(r).abs() < 1.0 / 64.0 && (bf(r) - bf(g)).abs() <= 1e-6) { *bad += 1; }
        u
    };
    for t in 0..k {
        let row = &pr[t * s.p()..(t + 1) * s.p()];
        let p_t = (pos + t) as u32;
        for h in 0..s.nh {
            let qref = cpu_norm_rope(&row[h * s.qs()..h * s.qs() + HD], qnw, p_t, &invf);
            for d in 0..HD { st.qr_ulp = st.qr_ulp.max(pchk(qref[d], qr[(t * s.nh + h) * HD + d], &mut st.prep_bad)); }
        }
        let kb = s.nh * s.qs();
        for kv in 0..s.nkv {
            let kref = cpu_norm_rope(&row[kb + kv * HD..kb + (kv + 1) * HD], knw, p_t, &invf);
            let vref = &row[kb + s.nkv * HD + kv * HD..kb + s.nkv * HD + (kv + 1) * HD];
            let slot = kv * maxctx + pos + t;
            if s.q8 {
                for g in 0..HD / 32 {
                    let (qk, _, _) = cpu_q8(&kref[g * 32..g * 32 + 32]);
                    let (qv, sv, lv) = cpu_q8(&vref[g * 32..g * 32 + 32]);
                    for i in 0..32 {
                        st.kq8_code = st.kq8_code.max((qk[i] as i32 - kc[slot * HD + g * 32 + i] as i32).unsigned_abs());
                        if qv[i] != vc[slot * HD + g * 32 + i] { st.vq8_bad += 1; }
                    }
                    let gi = (slot * (HD / 32) + g) * 2;
                    if vsb[gi] != sv || vsb[gi + 1] != lv { st.vq8_bad += 1; }
                }
            } else {
                for d in 0..HD {
                    st.k_ulp = st.k_ulp.max(pchk(kref[d], u16at(&kc, slot * HD + d), &mut st.prep_bad));
                    st.v_ulp = st.v_ulp.max(ulp_bf(vref[d], u16at(&vc, slot * HD + d)));
                }
            }
        }
        for h in 0..s.nh {
            let gate = if s.gated { Some(&row[h * 2 * HD + HD..(h + 1) * 2 * HD]) } else { None };
            let (kf, vf) = &kvs[h / (s.nh / s.nkv)];
            let (rf, oref) = cpu_attn(kf, vf, &qr[(t * s.nh + h) * HD..(t * s.nh + h + 1) * HD], pos + t, gate);
            for d in 0..HD {
                let g = ao[(t * s.nh + h) * HD + d];
                let u = ulp_bf(oref[d], g);
                let want = if s.gated { bf(oref[d]) } else { rf[d] };
                let abs = (bf(g) - want).abs();
                st.n += 1;
                if u > 0 { st.ao_diff += 1; }
                st.ao_ulp = st.ao_ulp.max(u);
                st.ao_abs = st.ao_abs.max(abs);
                if want.abs() >= 1.0 / 64.0 { st.ao_rel = st.ao_rel.max(abs / want.abs()); }
                let ok = u <= 2 || (bf(oref[d]).abs() < 1.0 / 64.0 && (bf(oref[d]) - bf(g)).abs() <= 2e-5);
                if !ok {
                    st.fail += 1;
                    if st.fail <= 5 { println!("    FAIL t {} h {} d {}: ref {:e} gpu {:e} ({} ulp)", t, h, d, bf(oref[d]), bf(g), u); }
                }
            }
        }
    }
    println!("  {:?} ctx {:>4} K {}: qr {} ulp, K {} ulp / q8 code {}, V {} ulp / q8 mismatches {}; ao max {} bf16-ulp, \
              max abs err {:.2e}, max rel err {:.2e} (|ref| >= 1/64), {} of {} differ from rounded ref",
             s, ctx, k, st.qr_ulp, st.k_ulp, st.kq8_code, st.v_ulp, st.vq8_bad, st.ao_ulp, st.ao_abs, st.ao_rel, st.ao_diff, st.n);
    assert!(st.prep_bad == 0 && st.v_ulp == 0, "prep mismatch {:?}", st);
    assert!(st.kq8_code <= 1 && st.vq8_bad == 0, "q8 quantization mismatch {:?}", st);
    assert_eq!(st.fail, 0, "attention outside tolerance {:?}", st);

    // K-invariance: K one-token passes (each with its own ctrl slot, minimal nch) and the K-token pass with one
    // spare chunk must give the same ao and cache bit for bit.
    if k > 1 {
        ar.put(b.p, &pr);
        b.fill_cache(&ar, pos, &mut Rng::new(cseed));
        for t in 0..k { b.set_pos(&ar, 1 + t, (pos + t) as u32); }
        run(gpu, &ar, |e| for t in 0..k { b.pass(e, 1 + t, t, 1, (pos + t + 1 + CH - 1) / CH); }).unwrap();
        assert_eq!(ar.get::<u16>(b.ao, k * s.nh * HD), ao, "K-invariance: ao (K one-token passes)");
        assert_eq!(ar.get::<u8>(b.kc, s.nkv * maxctx * HD * es), kc, "K-invariance: K cache");
        ar.put(b.p, &pr);
        b.fill_cache(&ar, pos, &mut Rng::new(cseed));
        run(gpu, &ar, |e| b.pass(e, 0, 0, k, nch_max)).unwrap();
        assert_eq!(ar.get::<u16>(b.ao, k * s.nh * HD), ao, "K-invariance: ao (spare chunk)");
    }
    st
}

fn sweep(name: &str, s: Shape, cases: &[(usize, usize)]) {
    let t0 = std::time::Instant::now();
    let gpu = gpu();
    let (mut worst_ulp, mut worst_abs, mut worst_rel) = (0, 0f32, 0f32);
    for (i, &(ctx, k)) in cases.iter().enumerate() {
        let st = case(&gpu, s, ctx, k, 1000 * (s.nh + s.nkv) as u64 + i as u64);
        worst_ulp = worst_ulp.max(st.ao_ulp);
        worst_abs = worst_abs.max(st.ao_abs);
        worst_rel = worst_rel.max(st.ao_rel);
    }
    println!("{}: {} cases OK, worst ao {} bf16-ulp, worst abs err {:.2e}, worst rel err {:.2e}, {:.2} s", name, cases.len(), worst_ulp, worst_abs, worst_rel,
             t0.elapsed().as_secs_f64());
}

// Qwen2.5-7B attention shape: 28 q heads, 4 kv heads (grp 7), no QK-norm, plain reduce.
#[test]
fn qwen25_7b_b16() {
    sweep("qwen25_7b_b16", Shape { nh: 28, nkv: 4, qkn: false, gated: false, q8: false },
          &[(1, 1), (2, 2), (300, 1), (300, 3), (600, 8), (513, 5)]);
}
#[test]
fn qwen25_7b_q8() {
    sweep("qwen25_7b_q8", Shape { nh: 28, nkv: 4, qkn: false, gated: false, q8: true }, &[(1, 1), (300, 1), (600, 4), (257, 7)]);
}
// Qwen3-8B: 32 / 8 (grp 4), QK-norm.
#[test]
fn qwen3_8b_b16() {
    sweep("qwen3_8b_b16", Shape { nh: 32, nkv: 8, qkn: true, gated: false, q8: false }, &[(1, 1), (256, 1), (257, 2), (513, 5), (40, 8)]);
}
#[test]
fn qwen3_8b_q8() {
    sweep("qwen3_8b_q8", Shape { nh: 32, nkv: 8, qkn: true, gated: false, q8: true }, &[(1, 1), (513, 1), (513, 7)]);
}
// Qwen3-32B: 64 / 8 (grp 8), QK-norm; > 4 chunks.
#[test]
fn qwen3_32b_b16() {
    sweep("qwen3_32b_b16", Shape { nh: 64, nkv: 8, qkn: true, gated: false, q8: false }, &[(1100, 1), (1100, 4)]);
}
// Qwen2.5-0.5B head counts (14 / 2, grp 7) and plain MHA (8 / 8) - GQA edge cases.
#[test]
fn gqa_edges() {
    sweep("gqa_14_2_b16", Shape { nh: 14, nkv: 2, qkn: false, gated: false, q8: false }, &[(40, 1), (40, 2), (700, 3)]);
    sweep("mha_8_8_q8", Shape { nh: 8, nkv: 8, qkn: true, gated: false, q8: true }, &[(256, 1), (257, 1), (300, 6)]);
}
// Gated variant at HD 128 (q | gate interleaved per head, attn128_reduce_gate): matches the CPU reference with gate.
#[test]
fn gated_hd128() {
    sweep("gated_16_4_b16", Shape { nh: 16, nkv: 4, qkn: true, gated: true, q8: false }, &[(1, 1), (300, 1), (300, 3)]);
    sweep("gated_16_4_q8", Shape { nh: 16, nkv: 4, qkn: true, gated: true, q8: true }, &[(300, 2), (520, 1)]);
}
