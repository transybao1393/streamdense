//! U7 self-test: engine/shaders/attn_gated.metal (attn_rope_invf, attn_prep_{b16,q8}, attn_partial_k{1..8}_{b16,q8},
//! attn_reduce_gate) at the real dims (24 q heads, 4 kv heads, head_dim 256, RoPE 64, theta 1e7).
//! - CPU reference (prep: RMSNorm + RoPE + KV append / q8 quantization; attention in f64) at ctx {1, 17, 300, 4100},
//!   K {1..8} (odd K: the last V sweep has one query), both the bf16 and the q8 KV paths.
//! - K-invariance: a K-token pass == K one-token passes, bit-exact (ao, qr, KV cache), also with an oversized nch.
//! - MLX vectors (attn.prep / sdpa_* / gate, rope.off*) if engine/testdata/vectors exists.
//! Run: cargo test --offline --release -p streamdense-metal --test attn_gated -- --nocapture --test-threads=1
mod b_util;
use b_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu, SdCtrl};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const GRP: usize = NH / NKV;
const RD: usize = 64;
const EPS: f32 = 1e-6;
const CH: usize = 256;
const CTRL_STRIDE: usize = 256;

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "attn_gated"]).unwrap();
    g.pipeline("attn_rope_invf").unwrap();
    g.pipeline("attn_reduce_gate").unwrap();
    for q in ["b16", "q8"] {
        g.pipeline(&format!("attn_prep_{}", q)).unwrap();
        for k in 1..=8 { g.pipeline(&format!("attn_partial_k{}_{}", k, q)).unwrap(); }
    }
    g
}
fn log2base() -> f32 { (1e7f32).log2() }

// ------------------------------------------------------------------------------------------- f16 helpers
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    if e == 0 { let v = m as f32 * (2f32).powi(-24); return if s != 0 { -v } else { v }; }
    if e == 31 { return f32::from_bits(s | 0x7f80_0000 | (m << 13)); }
    f32::from_bits(s | (((e - 15 + 127) as u32) << 23) | (m << 13))
}
fn f32_to_f16(f: f32) -> u16 {
    // round to nearest even via f64 search over the representable neighbours (values here are small and normal)
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = f.abs();
    if a.is_nan() { return s | 0x7e00; }
    if a >= 65520.0 { return s | 0x7c00; }
    let mut lo: u16 = 0; let mut hi: u16 = 0x7bff;
    while lo < hi { let mid = (lo + hi + 1) / 2; if f16_to_f32(mid) <= a { lo = mid; } else { hi = mid - 1; } }
    let (v0, v1) = (f16_to_f32(lo) as f64, f16_to_f32(lo + 1) as f64);
    let d0 = a as f64 - v0; let d1 = v1 - a as f64;
    let r = if lo == 0x7bff { lo } else if d0 < d1 { lo } else if d1 < d0 { lo + 1 } else if lo & 1 == 0 { lo } else { lo + 1 };
    s | r
}

// ------------------------------------------------------------------------------------------- CPU reference
fn sig_bf(x: f32) -> f32 {
    let e = rbf(x.abs().exp());
    let y = rbf(1.0 / rbf(1.0 + e));
    if x < 0.0 { y } else { rbf(1.0 - y) }
}
fn cpu_invf() -> Vec<f32> { (0..RD / 2).map(|i| (-(i as f32 / (RD / 2) as f32) * log2base()).exp2()).collect() }

/// RMSNorm (kernel order: butterfly simd_sum per 32 dims, then the 8 partials in order) + RoPE at pos.
fn cpu_norm_rope(x: &[u16], w: &[u16], pos: u32, invf: &[f32]) -> Vec<u16> {
    let xf: Vec<f32> = x.iter().map(|&h| bf(h)).collect();
    let mut tot = 0f32;
    for sg in 0..HD / 32 {
        let mut l: Vec<f32> = (0..32).map(|i| xf[sg * 32 + i] * xf[sg * 32 + i]).collect();
        for s in [16usize, 8, 4, 2, 1] { let p = l.clone(); for i in 0..32 { l[i] = p[i] + p[i ^ s]; } }
        tot += l[0];
    }
    let r = (1.0f64 / ((tot / HD as f32 + EPS) as f64).sqrt()) as f32;
    let n: Vec<f32> = (0..HD).map(|d| rbf(bf(w[d]) * rbf(xf[d] * r))).collect();
    rope(&n, pos, invf)
}
fn rope(n: &[f32], pos: u32, invf: &[f32]) -> Vec<u16> {
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
/// q8 group quantization as attn_prep: returns (q bytes [32], scale half, min half)
fn cpu_q8(x: &[u16]) -> (Vec<u8>, u16, u16) {
    let xf: Vec<f32> = x.iter().map(|&h| bf(h)).collect();
    let lo = xf.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = xf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sc16 = f32_to_f16((hi - lo).max(1e-8) / 255.0); let lo16 = f32_to_f16(lo);
    let scf = if f16_to_f32(sc16) > 0.0 { f16_to_f32(sc16) } else { 1.0 };
    (xf.iter().map(|&v| ((v - f16_to_f32(lo16)) / scf).round_ties_even().clamp(0.0, 255.0) as u8).collect(), sc16, lo16)
}

// ------------------------------------------------------------------------------------------- GPU harness
const P: usize = NH * 2 * HD + 2 * NKV * HD;
struct Bufs {
    q8: bool, maxctx: usize, nch: usize,
    p: usize, qn: usize, kn: usize, qr: usize, kc: usize, vc: usize, ksb: usize, vsb: usize, invf: usize, ctrl: usize,
    po: usize, pml: usize, ao: usize,
}
impl Bufs {
    fn new(ar: &mut Ar, q8: bool, tmax: usize, maxctx: usize, nch: usize, nctrl: usize) -> Bufs {
        let es = if q8 { 1 } else { 2 };
        Bufs { q8, maxctx, nch,
               p: ar.alloc(tmax * P * 2), qn: ar.alloc(HD * 2), kn: ar.alloc(HD * 2), qr: ar.alloc(tmax * NH * HD * 2),
               kc: ar.alloc(NKV * maxctx * HD * es), vc: ar.alloc(NKV * maxctx * HD * es),
               ksb: ar.alloc(NKV * maxctx * 8 * 4), vsb: ar.alloc(NKV * maxctx * 8 * 4), invf: ar.alloc(32 * 4),
               ctrl: ar.alloc(nctrl * CTRL_STRIDE), po: ar.alloc(tmax * NH * nch * HD * 4), pml: ar.alloc(tmax * NH * nch * 8),
               ao: ar.alloc(tmax * NH * HD * 2) }
    }
    fn arena_size(q8: bool, tmax: usize, maxctx: usize, nch: usize, nctrl: usize) -> usize {
        let es = if q8 { 1 } else { 2 };
        tmax * P * 2 + tmax * NH * HD * 4 + 2 * NKV * maxctx * (HD * es + 32) + nctrl * CTRL_STRIDE
            + tmax * NH * nch * (HD * 4 + 8) + (1 << 16)
    }
    fn sfx(&self) -> &'static str { if self.q8 { "q8" } else { "b16" } }
    fn set_pos(&self, ar: &Ar, slot: usize, pos: u32) { ar.put(self.ctrl + slot * CTRL_STRIDE, &[SdCtrl { pos, ..Default::default() }]); }
    fn invf(&self, e: &Enc) { e.pso("attn_rope_invf").b(0, self.invf).f(1, log2base()).threads((32, 1), 32); }
    fn prep(&self, e: &Enc, slot: usize, t0: usize, k: usize, nkv: u32) {
        e.pso(&format!("attn_prep_{}", self.sfx())).b(0, self.p + t0 * P * 2).b(1, self.qn).b(2, self.kn)
            .b(3, self.qr + t0 * NH * HD * 2).b(4, self.kc).b(5, self.vc).b(6, self.ksb).b(7, self.vsb).b(8, self.invf)
            .b(9, self.ctrl + slot * CTRL_STRIDE).u(10, self.maxctx as u32).f(11, EPS).u(12, nkv).u(13, RD as u32)
            .groups((NH + nkv as usize, k), 256);
    }
    fn partial(&self, e: &Enc, slot: usize, t0: usize, k: usize, nch: usize) {
        assert!(nch <= self.nch);
        let rs = NH * nch; // rows of po / pml per token for this nch
        e.pso(&format!("attn_partial_k{}_{}", k, self.sfx())).b(0, self.qr + t0 * NH * HD * 2).b(1, self.kc).b(2, self.vc)
            .b(3, self.ksb).b(4, self.vsb).b(5, self.po + t0 * rs * HD * 4).b(6, self.pml + t0 * rs * 8)
            .b(7, self.ctrl + slot * CTRL_STRIDE).u(8, self.maxctx as u32).u(9, GRP as u32).groups((nch, NH), 128);
    }
    fn reduce(&self, e: &Enc, t0: usize, k: usize, nch: usize) {
        let rs = NH * nch;
        e.pso("attn_reduce_gate").b(0, self.po + t0 * rs * HD * 4).b(1, self.pml + t0 * rs * 8).b(2, self.p + t0 * P * 2)
            .b(3, self.ao + t0 * NH * HD * 2).u(4, nch as u32).u(5, P as u32).groups((NH, k), HD);
    }
    /// One pass of k tokens at ctrl slot `slot` (ctrl.pos set by the caller): prep, partial, reduce.
    fn pass(&self, e: &Enc, slot: usize, t0: usize, k: usize, nch: usize) {
        self.prep(e, slot, t0, k, NKV as u32);
        self.partial(e, slot, t0, k, nch);
        self.reduce(e, t0, k, nch);
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

/// Dequantized (f64) K and V of kv head kvh, keys 0..n, from the GPU cache.
fn load_kv(b: &Bufs, ar: &Ar, kvh: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut kf = vec![0f64; n * HD]; let mut vf = vec![0f64; n * HD];
    if b.q8 {
        for (base, sb, out) in [(b.kc, b.ksb, &mut kf), (b.vc, b.vsb, &mut vf)] {
            let q: Vec<u8> = ar.get(base + kvh * b.maxctx * HD, n * HD);
            let s: Vec<u16> = ar.get(sb + kvh * b.maxctx * 32, n * 16);
            for j in 0..n { for d in 0..HD {
                let g = j * 8 + d / 32;
                out[j * HD + d] = f16_to_f32(s[2 * g]) as f64 * q[j * HD + d] as f64 + f16_to_f32(s[2 * g + 1]) as f64;
            } }
        }
    } else {
        let kk: Vec<u16> = ar.get(b.kc + kvh * b.maxctx * HD * 2, n * HD);
        let vv: Vec<u16> = ar.get(b.vc + kvh * b.maxctx * HD * 2, n * HD);
        for i in 0..n * HD { kf[i] = bf(kk[i]) as f64; vf[i] = bf(vv[i]) as f64; }
    }
    (kf, vf)
}

/// CPU attention (f64) for a query at position pos over keys 0..=pos; returns ao bf16 [256].
fn cpu_attn(kf: &[f64], vf: &[f64], q: &[u16], pos: usize, gate: &[u16]) -> Vec<u16> {
    let n = pos + 1;
    let qf: Vec<f64> = q.iter().map(|&h| bf(h) as f64 / 16.0).collect();
    let s: Vec<f64> = (0..n).map(|j| (0..HD).map(|d| qf[d] * kf[j * HD + d]).sum()).collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let l: f64 = p.iter().sum();
    let mut o = vec![0f64; HD];
    for j in 0..n { for d in 0..HD { o[d] += p[j] * vf[j * HD + d]; } }
    (0..HD).map(|d| to_bf(rbf((o[d] / l) as f32) * sig_bf(bf(gate[d])))).collect()
}

fn fill_cache(b: &Bufs, ar: &Ar, n: usize, r: &mut Rng) {
    for kvh in 0..NKV {
        if b.q8 {
            let qk: Vec<u8> = (0..n * HD).map(|_| (r.u64() & 0xff) as u8).collect();
            let qv: Vec<u8> = (0..n * HD).map(|_| (r.u64() & 0xff) as u8).collect();
            ar.put(b.kc + kvh * b.maxctx * HD, &qk); ar.put(b.vc + kvh * b.maxctx * HD, &qv);
            let sb = |r: &mut Rng| -> Vec<u16> {
                (0..n * 8).flat_map(|_| [f32_to_f16((4.0 + 2.0 * r.unif()) / 255.0), f32_to_f16(-2.0 - r.unif())]).collect()
            };
            let ks = sb(r); let vs = sb(r);
            ar.put(b.ksb + kvh * b.maxctx * 32, &ks); ar.put(b.vsb + kvh * b.maxctx * 32, &vs);
        } else {
            ar.put(b.kc + kvh * b.maxctx * HD * 2, &r.bf_n(n * HD, 1.0));
            ar.put(b.vc + kvh * b.maxctx * HD * 2, &r.bf_n(n * HD, 1.0));
        }
    }
}

struct Snap { ao: Vec<u16>, qr: Vec<u16>, kc: Vec<u8>, vc: Vec<u8>, ksb: Vec<u8>, vsb: Vec<u8> }
fn snap(b: &Bufs, ar: &Ar, k: usize) -> Snap {
    let es = if b.q8 { 1 } else { 2 };
    Snap { ao: ar.get(b.ao, k * NH * HD), qr: ar.get(b.qr, k * NH * HD), kc: ar.get(b.kc, NKV * b.maxctx * HD * es),
           vc: ar.get(b.vc, NKV * b.maxctx * HD * es), ksb: ar.get(b.ksb, NKV * b.maxctx * 32), vsb: ar.get(b.vsb, NKV * b.maxctx * 32) }
}

fn check(q8: bool) {
    let gpu = gpu();
    let maxctx = 4104;
    let nch_max = (maxctx + CH - 1) / CH + 1; // one spare chunk: always-empty chunk must not change anything
    let kmax = 8;
    let mut ar = Ar::new(&gpu, Bufs::arena_size(q8, kmax, maxctx, nch_max, kmax + 1));
    let b = Bufs::new(&mut ar, q8, kmax, maxctx, nch_max, kmax + 1);
    let mut r = Rng::new(if q8 { 8 } else { 16 });
    ar.put(b.qn, &(0..HD).map(|_| to_bf(0.8 + 0.8 * r.unif())).collect::<Vec<_>>());
    ar.put(b.kn, &(0..HD).map(|_| to_bf(0.8 + 0.8 * r.unif())).collect::<Vec<_>>());
    run(&gpu, &ar, |e| b.invf(e));
    let invf: Vec<f32> = ar.get(b.invf, 32);
    let (iu, _, _) = cmp_f32(&invf, &cpu_invf());
    println!("[{}] invf vs CPU exp2: max {} f32-ulp", b.sfx(), iu);
    let (qn, kn): (Vec<u16>, Vec<u16>) = (ar.get(b.qn, HD), ar.get(b.kn, HD));
    for &ctx in &[1usize, 17, 300, 4100] {
        for &k in &[1usize, 2, 3, 4, 5, 6, 7, 8] {
            if k > ctx { continue; }
            let pos = ctx - k;
            let nch = (ctx + CH - 1) / CH;
            let mut rr = Rng::new(ctx as u64 * 31 + k as u64);
            let pr = rr.bf_n(k * P, 1.5);
            let cache = |rr: &mut Rng| { let mut c = Rng::new(rr.u64()); fill_cache(&b, &ar, pos, &mut c); };
            let seed = rr.u64();
            // (a) one K-token pass
            ar.put(b.p, &pr); cache(&mut Rng::new(seed)); b.set_pos(&ar, 0, pos as u32);
            run(&gpu, &ar, |e| b.pass(e, 0, 0, k, nch));
            let a = snap(&b, &ar, k);
            // CPU: prep (qr, appended K/V) and attention
            let kvs: Vec<(Vec<f64>, Vec<f64>)> = (0..NKV).map(|kv| load_kv(&b, &ar, kv, ctx)).collect();
            let (mut qu, mut ku, mut vu, mut kqd, mut ou, mut on, mut worst) = (0u32, 0u32, 0u32, 0u32, 0u32, 0usize, 0u32);
            for t in 0..k {
                let row = &pr[t * P..(t + 1) * P];
                for h in 0..NH {
                    let qref = cpu_norm_rope(&row[h * 2 * HD..h * 2 * HD + HD], &qn, (pos + t) as u32, &invf);
                    qu = qu.max(cmp_bf(&qref, &a.qr[(t * NH + h) * HD..(t * NH + h + 1) * HD]).0);
                }
                for kv in 0..NKV {
                    let kref = cpu_norm_rope(&row[NH * 2 * HD + kv * HD..NH * 2 * HD + (kv + 1) * HD], &kn, (pos + t) as u32, &invf);
                    let vref = &row[NH * 2 * HD + NKV * HD + kv * HD..NH * 2 * HD + NKV * HD + (kv + 1) * HD];
                    let slot = kv * maxctx + pos + t;
                    if q8 {
                        // quantization of the GPU's own bf16 K (from qr-equivalent path) is checked against CPU q8 of kref
                        for gi in 0..8 {
                            let (qk, _, _) = cpu_q8(&kref[gi * 32..gi * 32 + 32]);
                            let (qv, sv, lv) = cpu_q8(&vref[gi * 32..gi * 32 + 32]);
                            for i in 0..32 {
                                kqd = kqd.max((qk[i] as i32 - a.kc[slot * HD + gi * 32 + i] as i32).unsigned_abs());
                                vu = vu.max((qv[i] as i32 - a.vc[slot * HD + gi * 32 + i] as i32).unsigned_abs());
                            }
                            let gsb = &a.vsb[(slot * 8 + gi) * 4..(slot * 8 + gi) * 4 + 4];
                            if u16::from_le_bytes([gsb[0], gsb[1]]) != sv || u16::from_le_bytes([gsb[2], gsb[3]]) != lv { vu = vu.max(1000); }
                        }
                    } else {
                        let kg: Vec<u16> = a.kc[slot * HD * 2..(slot + 1) * HD * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                        let vg: Vec<u16> = a.vc[slot * HD * 2..(slot + 1) * HD * 2].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                        ku = ku.max(cmp_bf(&kref, &kg).0);
                        vu = vu.max(cmp_bf(vref, &vg).0);
                    }
                }
                for h in 0..NH {
                    let gate = &row[h * 2 * HD + HD..(h + 1) * 2 * HD];
                    let (kf, vf) = &kvs[h / GRP];
                    let oref = cpu_attn(kf, vf, &a.qr[(t * NH + h) * HD..(t * NH + h + 1) * HD], pos + t, gate);
                    let ag = &a.ao[(t * NH + h) * HD..(t * NH + h + 1) * HD];
                    for d in 0..HD {
                        let u = ulp_bf(oref[d], ag[d]);
                        if u > 0 { on += 1; }
                        // above 2 ulp only accepted for near-cancelled outputs (|ao| < 1/64, abs err <= 1e-6): print them
                        if u > 2 { println!("    t {} h {} d {}: ref {:e} gpu {:e} ({} ulp, abs {:.1e})", t, h, d, bf(oref[d]), bf(ag[d]), u, (bf(oref[d]) - bf(ag[d])).abs()); }
                        // near-cancelled outputs (fp32 accumulation of O(1) terms): judge by absolute error instead
                        let small = bf(oref[d]).abs() < 1.0 / 64.0 && (bf(oref[d]) - bf(ag[d])).abs() <= 1e-6;
                        ou = ou.max(if small { u.min(2) } else { u });
                        worst = worst.max(u);
                    }
                }
            }
            // (b) K one-token passes (positions pos..pos+K-1), each with its own ctrl slot and minimal nch
            ar.put(b.p, &pr); cache(&mut Rng::new(seed));
            for t in 0..k { b.set_pos(&ar, 1 + t, (pos + t) as u32); }
            run(&gpu, &ar, |e| for t in 0..k { b.pass(e, 1 + t, t, 1, (pos + t + 1 + CH - 1) / CH); });
            let s1 = snap(&b, &ar, k);
            // (c) the same K-token pass with the oversized nch
            ar.put(b.p, &pr); cache(&mut Rng::new(seed));
            run(&gpu, &ar, |e| b.pass(e, 0, 0, k, nch_max));
            let c = snap(&b, &ar, k);
            let kinv = a.ao == s1.ao && a.qr == s1.qr && a.kc == s1.kc && a.vc == s1.vc && a.ksb == s1.ksb && a.vsb == s1.vsb;
            let ninv = a.ao == c.ao;
            println!("[{}] ctx {:4} K {}: qr {}ulp, K {}ulp{}, V {}{}; ao vs f64 CPU max {} ulp ({} of {} differ); K-pass == {}x K=1 bit-exact {}; nch {}=={} {}",
                     b.sfx(), ctx, k, qu, if q8 { kqd } else { ku }, if q8 { " (q-steps)" } else { "" }, vu,
                     if q8 { " (q-steps, scale/min exact unless >=1000)" } else { "ulp" }, worst, on, k * NH * HD, k, kinv, nch, nch_max, ninv);
            assert!(kinv && ninv, "K-invariance");
            assert!(qu <= 1 && ku <= 1, "prep vs CPU");
            if q8 { assert!(kqd <= 1 && vu <= 1, "q8 quantization vs CPU"); } else { assert_eq!(vu, 0, "V copy"); }
            assert!(ou <= 2, "attention vs CPU (f64 ref; ob and ao are two bf16 roundings)");
        }
    }
}

#[test]
fn attn_cpu_ref_kinv_b16() { check(false); }
#[test]
fn attn_cpu_ref_kinv_q8() { check(true); }

// ------------------------------------------------------------------------------------------- MLX vectors
#[test]
fn attn_mlx_vectors() {
    let Some(v) = vectors("attn") else { return };
    let gpu = gpu();
    // ---- prep: K = 2 at offset 37
    {
        let k = 2; let off = 37usize; let maxctx = 64;
        let mut ar = Ar::new(&gpu, Bufs::arena_size(false, k, maxctx, 1, 1));
        let b = Bufs::new(&mut ar, false, k, maxctx, 1, 1);
        let (qp, kp, vp) = (v.u16s("prep.qp"), v.u16s("prep.kp"), v.u16s("prep.vp"));
        let mut p = vec![0u16; k * P];
        for t in 0..k {
            p[t * P..t * P + NH * 2 * HD].copy_from_slice(&qp[t * NH * 2 * HD..(t + 1) * NH * 2 * HD]);
            p[t * P + NH * 2 * HD..t * P + NH * 2 * HD + NKV * HD].copy_from_slice(&kp[t * NKV * HD..(t + 1) * NKV * HD]);
            p[t * P + NH * 2 * HD + NKV * HD..(t + 1) * P].copy_from_slice(&vp[t * NKV * HD..(t + 1) * NKV * HD]);
        }
        ar.put(b.p, &p); ar.put(b.qn, &v.u16s("prep.q_norm")); ar.put(b.kn, &v.u16s("prep.k_norm"));
        b.set_pos(&ar, 0, off as u32);
        run(&gpu, &ar, |e| { b.invf(e); b.prep(e, 0, 0, k, NKV as u32); });
        let qr: Vec<u16> = ar.get(b.qr, k * NH * HD);
        let kc: Vec<u16> = ar.get(b.kc, NKV * maxctx * HD);
        let vc: Vec<u16> = ar.get(b.vc, NKV * maxctx * HD);
        let (mq, mk, mv) = (v.u16s("prep.q"), v.u16s("prep.k"), v.u16s("prep.v")); // [1, heads, K, 256]
        let (mut qu, mut qn, mut ku, mut kn, mut vu) = (0, 0, 0, 0, 0);
        for t in 0..k {
            for h in 0..NH {
                let (u, n) = cmp_bf(&qr[(t * NH + h) * HD..(t * NH + h + 1) * HD], &mq[(h * k + t) * HD..(h * k + t + 1) * HD]);
                qu = qu.max(u); qn += n;
            }
            for kv in 0..NKV {
                let s = (kv * maxctx + off + t) * HD;
                let (u, n) = cmp_bf(&kc[s..s + HD], &mk[(kv * k + t) * HD..(kv * k + t + 1) * HD]); ku = ku.max(u); kn += n;
                vu = vu.max(cmp_bf(&vc[s..s + HD], &mv[(kv * k + t) * HD..(kv * k + t + 1) * HD]).0);
            }
        }
        println!("MLX attn.prep (K 2, off 37): q {} ulp ({} of {} differ), k {} ulp ({} of {}), v {} ulp",
                 qu, qn, k * NH * HD, ku, kn, k * NKV * HD, vu);
        assert!(qu <= 1 && ku <= 1 && vu == 0);
    }
    // ---- sdpa: q given, K/V = kv.k / kv.v [1, 4, 300, 256]; gate 20 -> sigmoid == 1.0 exactly in bf16, so ao == bf(o)
    {
        let n_all = 300; let maxctx = n_all; let nch = 2; let kmax = 8;
        let mut ar = Ar::new(&gpu, Bufs::arena_size(false, kmax, maxctx, nch, kmax));
        let b = Bufs::new(&mut ar, false, kmax, maxctx, nch, kmax);
        ar.put(b.kc, &v.u16s("kv.k")); ar.put(b.vc, &v.u16s("kv.v"));
        assert_eq!(sig_bf(20.0), 1.0);
        let mut prow = vec![0u16; P];
        for h in 0..NH { for d in 0..HD { prow[h * 2 * HD + HD + d] = to_bf(20.0); } }
        ar.put(b.p, &prow.repeat(kmax));
        for (n, kq) in [(1usize, 1usize), (17, 1), (256, 1), (257, 1), (300, 1), (17, 2), (300, 2), (300, 8)] {
            let nm = format!("sdpa_n{}_k{}", n, kq);
            let q = v.u16s(&format!("{}.q", nm)); // [1, 24, K, 256]
            let mut qr = vec![0u16; kq * NH * HD];
            for h in 0..NH { for t in 0..kq { qr[(t * NH + h) * HD..(t * NH + h + 1) * HD].copy_from_slice(&q[(h * kq + t) * HD..(h * kq + t + 1) * HD]); } }
            ar.put(b.qr, &qr);
            b.set_pos(&ar, 0, (n - kq) as u32);
            for t in 0..kq { b.set_pos(&ar, 1 + t % (kmax - 1), (n - kq + t) as u32); }
            let nc = (n + CH - 1) / CH;
            run(&gpu, &ar, |e| { b.partial(e, 0, 0, kq, nc); b.reduce(e, 0, kq, nc); });
            let ao: Vec<u16> = ar.get(b.ao, kq * NH * HD);
            let mo = v.u16s(&format!("{}.o", nm));
            let (mut u, mut cnt) = (0, 0);
            for h in 0..NH { for t in 0..kq {
                let (a, c) = cmp_bf(&ao[(t * NH + h) * HD..(t * NH + h + 1) * HD], &mo[(h * kq + t) * HD..(h * kq + t + 1) * HD]);
                u = u.max(a); cnt += c;
            } }
            // f64 CPU reference (bottom-right causal: query t sees keys 0..=n-K+t) vs GPU and vs MLX
            let kvs: Vec<(Vec<f64>, Vec<f64>)> = (0..NKV).map(|kv| load_kv(&b, &ar, kv, n)).collect();
            let (mut cg, mut cm) = (0, 0);
            for h in 0..NH { for t in 0..kq {
                let (kf, vf) = &kvs[h / GRP];
                let rf = cpu_attn(kf, vf, &qr[(t * NH + h) * HD..(t * NH + h + 1) * HD], n - kq + t, &prow[h * 2 * HD + HD..(h + 1) * 2 * HD]);
                cg = cg.max(cmp_bf(&rf, &ao[(t * NH + h) * HD..(t * NH + h + 1) * HD]).0);
                cm = cm.max(cmp_bf(&rf, &mo[(h * kq + t) * HD..(h * kq + t + 1) * HD]).0);
            } }
            println!("MLX attn.{}: GPU vs MLX max {} ulp ({} of {} differ); f64 CPU vs GPU {} ulp, vs MLX {} ulp", nm, u, cnt, kq * NH * HD, cg, cm);
            assert!(cg <= 2, "{}: GPU vs f64 reference", nm);
        }
    }
    // ---- output gate: one chunk with m = 0, l = 1, o = gate.o  ->  ao = bf(gate.o * sigmoid(gate.gate))
    {
        let k = 2;
        let mut ar = Ar::new(&gpu, Bufs::arena_size(false, k, 1, 1, 1));
        let b = Bufs::new(&mut ar, false, k, 1, 1, 1);
        let (go, gg) = (v.u16s("gate.o"), v.u16s("gate.gate"));
        let mut p = vec![0u16; k * P];
        for t in 0..k { for h in 0..NH { p[t * P + h * 2 * HD + HD..t * P + (h + 1) * 2 * HD].copy_from_slice(&gg[(t * NH + h) * HD..(t * NH + h + 1) * HD]); } }
        ar.put(b.p, &p);
        ar.put(b.po, &go.iter().map(|&h| bf(h)).collect::<Vec<f32>>());
        ar.put(b.pml, &(0..k * NH).flat_map(|_| [0f32, 1f32]).collect::<Vec<f32>>());
        run(&gpu, &ar, |e| b.reduce(e, 0, k, 1));
        let (u, c) = cmp_bf(&ar.get::<u16>(b.ao, k * NH * HD), &v.u16s("gate.y"));
        println!("MLX attn.gate: max {} ulp ({} of {} differ)", u, c, k * NH * HD);
        assert_eq!(u, 0);
    }
}

/// rope.* vectors: (1) the CPU RoPE formula (accurate cos/sin) vs MLX at every offset; (2) the GPU attn_prep q path
/// (nkv = 0, so no KV slot is needed even at offset 131071) vs the CPU norm+RoPE reference at the same offsets.
/// The GPU cannot be fed the raw rope.x without its RMSNorm, so its direct MLX check is attn.prep (offset 37).
#[test]
fn attn_rope_vectors() {
    let Some(v) = vectors("rope") else { return };
    let gpu = gpu();
    let k = 8;
    let mut ar = Ar::new(&gpu, Bufs::arena_size(false, k, 1, 1, 1));
    let b = Bufs::new(&mut ar, false, k, 1, 1, 1);
    run(&gpu, &ar, |e| b.invf(e));
    let invf: Vec<f32> = ar.get(b.invf, 32);
    let xq = v.u16s("x.q"); // [1, 24, 8, 256]
    let mut r = Rng::new(5);
    let w: Vec<u16> = (0..HD).map(|_| to_bf(0.8 + 0.8 * r.unif())).collect();
    ar.put(b.qn, &w);
    let pq = NH * 2 * HD; // with nkv = 0 the kernel's p row stride is nh*2*HD
    let mut p = vec![0u16; k * pq];
    for t in 0..k { for h in 0..NH { p[t * pq + h * 2 * HD..t * pq + h * 2 * HD + HD].copy_from_slice(&xq[(h * k + t) * HD..(h * k + t + 1) * HD]); } }
    ar.put(b.p, &p);
    for off in [0u32, 1, 7, 1000, 4095, 131071] {
        // (1) CPU rope vs MLX
        let mq = v.u16s(&format!("off{}.q", off));
        let mq1 = v.u16s(&format!("off{}.q_k1", off));
        let (mut cu, mut cn) = (0, 0);
        for h in 0..NH { for t in 0..k {
            let x: Vec<f32> = xq[(h * k + t) * HD..(h * k + t + 1) * HD].iter().map(|&u| bf(u)).collect();
            let (u, n) = cmp_bf(&rope(&x, off + t as u32, &invf), &mq[(h * k + t) * HD..(h * k + t + 1) * HD]);
            cu = cu.max(u); cn += n;
        } }
        // (2) GPU prep (q heads only) vs CPU norm+rope
        b.set_pos(&ar, 0, off);
        run(&gpu, &ar, |e| b.prep(e, 0, 0, k, 0));
        let qr: Vec<u16> = ar.get(b.qr, k * NH * HD);
        let (mut gu, mut gn) = (0, 0);
        for h in 0..NH { for t in 0..k {
            let refq = cpu_norm_rope(&xq[(h * k + t) * HD..(h * k + t + 1) * HD], &w, off + t as u32, &invf);
            let (u, n) = cmp_bf(&refq, &qr[(t * NH + h) * HD..(t * NH + h + 1) * HD]); gu = gu.max(u); gn += n;
        } }
        println!("rope off {:6}: CPU rope vs MLX max {} ulp ({} of {} differ; MLX q == q_k1: {}); GPU prep vs CPU max {} ulp ({} differ)",
                 off, cu, cn, NH * k * HD, mq == mq1, gu, gn);
    }
}
