//! Self-test of shaders/attn_hd128_flash.metal (`attn128_flash`, prefill) and shaders/attn_hd128_sdpa.metal
//! (`attn128_sdpa_vec`, MLX-matching decode) at head dim 128, Qwen dense GQA shapes (16/8 Qwen3-1.7B, 12/2
//! Qwen2.5-1.5B, 32/8 Qwen3-8B), contexts 1..1100, prefill T up to 512. No model is loaded; inputs are random bf16
//! written straight into qr and the KV cache (prep is covered by tests/attn_hd128.rs).
//! - vs a CPU f32 reference (sequential dot / sums, libm exp, o / l rounded to bf16 once):
//!   flash and the single-pass mirror (N < 1024): <= 2 bf16 ulp, or abs err <= 2e-5 where |ref| < 1/64 (near-cancelled
//!   sums), as tests/attn_hd128.rs. Two-pass mirror rows (N >= 1024, MLX's per-block partials are rounded to bf16)
//!   additionally accept abs err <= e_d / 256 + 1 bf16 ulp of ref, e_d = sum_j p_j |v_jd| / l (each partial carries
//!   <= 2^-9 relative rounding; x2 margin).
//! - flash vs split-K (attn128_partial_k8_b16 + attn128_reduce over the same rows): <= 2 bf16 ulp, or abs <= 2e-5
//!   where |out| < 1/64. Reported: share of outputs that differ.
//! - attn128_sdpa_vec: a T-row dispatch == T one-row dispatches, bit for bit (K-invariance).
//! - gated flag (bit 0) of both kernels vs the CPU reference with MLX's bf16 sigmoid.
//! - MLX bit-identity: outputs of mx.fast.scaled_dot_product_attention (mlx 0.32.4, L = 1, bf16) on hash-generated
//!   inputs, written by an MLX reference script (not part of this repository); `attn128_sdpa_vec` must match every bit. File path:
//!   $QDENSE_MLX_SDPA128 or the default below; the test is skipped (prints why) when the file is missing.
//!
//! Run: cargo test --release --test attn_hd128_flash -- --nocapture --test-threads=1   (inside engine/dense-engine)
use dense_engine::gpu::{run, Arena, Enc, Gpu, SdCtrl};
use dense_engine::model::{sdpa128_vec_max_n, SDPA128_TWO_PASS_MIN};
/// Library for these tests: the engine's common_bf16.metal (read only) + the three HD-128 attention files.
const SOURCES: [&str; 4] = dense_engine::SHADER_SOURCE;

const HD: usize = 128;
const CH: usize = 256;
const SCALE: f32 = 0.08838834764831845; // 128^-0.5 in f32, as the kernels
const CTRL_STRIDE: usize = 256;
/// default location of the MLX outputs (untracked; set $QDENSE_MLX_SDPA128 to use another path)
const MLX_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../archive/qwen-dense-mlx/mlx_sdpa128.bin");

fn gpu() -> Gpu {
    let mut g = Gpu::from_sources(&SOURCES).unwrap();
    for n in ["attn128_flash", "attn128_sdpa_vec", "attn128_reduce", "attn128_reduce_gate"] { g.pipeline(n).unwrap(); }
    for k in 1..=8 { g.pipeline(&format!("attn128_partial_k{}_b16", k)).unwrap(); }
    g
}

// ------------------------------------------------------------------------------------------- bf16 / rng
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
/// Same generator as mlx_sdpa128.py: splitmix64 finalizer of (i + seed * golden), top 24 bits -> [0, 1),
/// (2u - 1) * amp in f32, rounded to bf16 (RNE).
fn hash_bf(n: usize, seed: u64, amp: f32) -> Vec<u16> {
    let base = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n as u64).map(|i| {
        let mut z = i.wrapping_add(base);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = (z >> 40) as f32 / (1u64 << 24) as f32;
        to_bf((u * 2.0 - 1.0) * amp)
    }).collect()
}

// ------------------------------------------------------------------------------------------- CPU reference
fn sig_bf(x: f32) -> f32 {
    let e = rbf(x.abs().exp());
    let y = rbf(1.0 / rbf(1.0 + e));
    if x < 0.0 { y } else { rbf(1.0 - y) }
}
/// f32 attention of one query over keys 0..n of one kv head (kf / vf f32 [n][HD]). Returns (bf16 out, e) with
/// e[d] = sum_j p_j |v_jd| / l (the scale of the two-pass partial rounding error).
fn cpu_attn(kf: &[f32], vf: &[f32], q: &[u16], n: usize, gate: Option<&[u16]>) -> (Vec<u16>, Vec<f32>) {
    let qf: Vec<f32> = q.iter().map(|&h| SCALE * bf(h)).collect();
    let s: Vec<f32> = (0..n).map(|j| { let mut a = 0f32; for d in 0..HD { a += qf[d] * kf[j * HD + d]; } a }).collect();
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
    let l: f32 = p.iter().sum();
    let (mut o, mut e) = (vec![0f32; HD], vec![0f32; HD]);
    for j in 0..n { for d in 0..HD { o[d] += p[j] * vf[j * HD + d]; e[d] += p[j] * vf[j * HD + d].abs(); } }
    let r: Vec<f32> = o.iter().map(|x| x / l).collect();
    let out = (0..HD).map(|d| match gate { None => to_bf(r[d]), Some(g) => to_bf(rbf(r[d]) * sig_bf(bf(g[d]))) }).collect();
    (out, e.iter().map(|x| x / l).collect())
}

// ------------------------------------------------------------------------------------------- GPU harness
#[derive(Clone, Copy, Debug)]
struct Shape { nh: usize, nkv: usize }
impl Shape { fn grp(&self) -> usize { self.nh / self.nkv } }

/// Buffers for one case: T query rows at positions pos .. pos + T - 1 (row t sees keys 0 ..= pos + t).
struct Case {
    s: Shape, t: usize, pos: usize, maxctx: usize, gated: bool,
    qr: usize, kc: usize, vc: usize, p: usize, ao: usize, ao2: usize, ctrl: usize, po: usize, pml: usize,
    q: Vec<u16>, k: Vec<u16>, v: Vec<u16>, pv: Vec<u16>, ar: Arena,
}
impl Case {
    fn p_stride(&self) -> usize { self.s.nh * 2 * HD + 2 * self.s.nkv * HD }
    fn new(gpu: &Gpu, s: Shape, t: usize, pos: usize, gated: bool, seed: u64, kv_std: f32) -> Case {
        let n = pos + t;
        let maxctx = n + 5; // not a multiple of anything on purpose
        let nch = (n + CH - 1) / CH;
        let nslots = t.div_ceil(8) + 1;
        let ps = s.nh * 2 * HD + 2 * s.nkv * HD;
        let sz = 2 * (t * s.nh * HD * 2) * 2 + 2 * s.nkv * maxctx * HD * 2 + t * ps * 2 + nslots * CTRL_STRIDE
            + 8 * s.nh * nch * (HD * 4 + 8) + (1 << 16);
        let mut ar = Arena::new(gpu, sz).unwrap();
        let (qr, kc, vc) = (ar.alloc(t * s.nh * HD * 2), ar.alloc(s.nkv * maxctx * HD * 2), ar.alloc(s.nkv * maxctx * HD * 2));
        let (p, ao, ao2) = (ar.alloc(t * ps * 2), ar.alloc(t * s.nh * HD * 2), ar.alloc(t * s.nh * HD * 2));
        let ctrl = ar.alloc(nslots * CTRL_STRIDE);
        let (po, pml) = (ar.alloc(8 * s.nh * nch * HD * 4), ar.alloc(8 * s.nh * nch * 8));
        let mut r = Rng::new(seed);
        let q = r.bf_n(t * s.nh * HD, 1.5);
        let k = r.bf_n(s.nkv * n * HD, kv_std);
        let v = r.bf_n(s.nkv * n * HD, kv_std);
        let pv = r.bf_n(t * ps, 1.5);
        let c = Case { s, t, pos, maxctx, gated, qr, kc, vc, p, ao, ao2, ctrl, po, pml, q, k, v, pv, ar };
        c.upload();
        c
    }
    fn upload(&self) {
        let n = self.pos + self.t;
        self.ar.put(self.qr, &self.q);
        self.ar.put(self.p, &self.pv);
        for kvh in 0..self.s.nkv {
            self.ar.put(self.kc + kvh * self.maxctx * HD * 2, &self.k[kvh * n * HD..(kvh + 1) * n * HD]);
            self.ar.put(self.vc + kvh * self.maxctx * HD * 2, &self.v[kvh * n * HD..(kvh + 1) * n * HD]);
        }
        self.set_pos(0, self.pos);
    }
    fn set_pos(&self, slot: usize, pos: usize) {
        self.ar.put(self.ctrl + slot * CTRL_STRIDE, &[SdCtrl { pos: pos as u32, ..Default::default() }]);
    }
    fn flags(&self) -> u32 { self.gated as u32 }
    fn flash(&self, e: &Enc, out: usize) {
        e.pso("attn128_flash").b(0, self.qr).b(1, self.kc).b(2, self.vc).b(3, self.p).b(4, out).b(5, self.ctrl)
            .u(6, self.maxctx as u32).u(7, self.s.grp() as u32).u(8, self.p_stride() as u32).u(9, self.t as u32)
            .u(10, self.flags())
            .groups((self.t.div_ceil(8), self.s.nkv), 32 * self.s.grp());
    }
    /// attn128_sdpa_vec for rows t0 .. t0 + k (ctrl slot `slot` must hold pos + t0).
    fn sdpa(&self, e: &Enc, slot: usize, t0: usize, k: usize, out: usize, lim: usize) {
        let nh = self.s.nh;
        e.pso("attn128_sdpa_vec").b(0, self.qr + t0 * nh * HD * 2).b(1, self.kc).b(2, self.vc)
            .b(3, self.p + t0 * self.p_stride() * 2).b(4, out + t0 * nh * HD * 2).b(5, self.ctrl + slot * CTRL_STRIDE)
            .u(6, self.maxctx as u32).u(7, self.s.grp() as u32).u(8, self.p_stride() as u32).u(9, lim as u32)
            .u(10, SDPA128_TWO_PASS_MIN as u32).u(11, self.flags())
            .groups((nh, k), 1024);
    }
    /// Split-K (attn128_partial_kK_b16 + attn128_reduce[_gate]) over all T rows in passes of <= 8 rows (ctrl slot
    /// 1 + i for pass i), into `out`.
    fn split_k(&self, e: &Enc, out: usize) {
        let nh = self.s.nh;
        for (i, t0) in (0..self.t).step_by(8).enumerate() {
            let k = (self.t - t0).min(8);
            let nch = (self.pos + t0 + k + CH - 1) / CH;
            let ctrl = self.ctrl + (1 + i) * CTRL_STRIDE;
            e.pso(&format!("attn128_partial_k{}_b16", k)).b(0, self.qr + t0 * nh * HD * 2).b(1, self.kc).b(2, self.vc)
                .b(3, self.kc).b(4, self.vc).b(5, self.po).b(6, self.pml).b(7, ctrl)
                .u(8, self.maxctx as u32).u(9, self.s.grp() as u32)
                .groups((nch, nh), 128);
            e.pso(if self.gated { "attn128_reduce_gate" } else { "attn128_reduce" }).b(0, self.po).b(1, self.pml)
                .b(2, self.p + t0 * self.p_stride() * 2).b(3, out + t0 * nh * HD * 2).u(4, nch as u32)
                .u(5, self.p_stride() as u32)
                .groups((nh, k), HD / 4);
        }
    }
    fn set_split_slots(&self) {
        for (i, t0) in (0..self.t).step_by(8).enumerate() { self.set_pos(1 + i, self.pos + t0); }
    }
    fn kv_f32(&self, kvh: usize) -> (Vec<f32>, Vec<f32>) {
        let n = self.pos + self.t;
        (self.k[kvh * n * HD..(kvh + 1) * n * HD].iter().map(|&h| bf(h)).collect(),
         self.v[kvh * n * HD..(kvh + 1) * n * HD].iter().map(|&h| bf(h)).collect())
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct Stats { n: usize, diff: usize, ulp: u32, abs: f32, fail: usize }
impl Stats {
    fn add(&mut self, o: Stats) {
        self.n += o.n; self.diff += o.diff; self.ulp = self.ulp.max(o.ulp); self.abs = self.abs.max(o.abs); self.fail += o.fail;
    }
}

/// GPU output `got` (bf16 [T][nh][HD]) vs the CPU reference on rows `rows`: <= 2 bf16 ulp, or abs <= 2e-5 where
/// |ref| < 1/64; rows with `two_pass(n)` (n keys) also accept abs <= e[d] / 256 + 1 bf16 ulp of ref (MLX 2-pass: each
/// block partial rounded to bf16, rel 2^-9, summed with weights; x2 margin).
fn check_cpu(c: &Case, got: &[u16], rows: &[usize], two_pass: impl Fn(usize) -> bool, what: &str) -> Stats {
    let s = c.s;
    let kvs: Vec<(Vec<f32>, Vec<f32>)> = (0..s.nkv).map(|kv| c.kv_f32(kv)).collect();
    let mut st = Stats::default();
    for &t in rows {
        let n = c.pos + t + 1;
        for h in 0..s.nh {
            let (kf, vf) = &kvs[h / s.grp()];
            let gate = if c.gated { Some(&c.pv[t * c.p_stride() + h * 2 * HD + HD..t * c.p_stride() + (h + 1) * 2 * HD]) } else { None };
            let (oref, ew) = cpu_attn(kf, vf, &c.q[(t * s.nh + h) * HD..(t * s.nh + h + 1) * HD], n, gate);
            for d in 0..HD {
                let g = got[(t * s.nh + h) * HD + d];
                let u = ulp_bf(oref[d], g);
                let abs = (bf(g) - bf(oref[d])).abs();
                st.n += 1;
                if u > 0 { st.diff += 1; }
                st.ulp = st.ulp.max(u);
                st.abs = st.abs.max(abs);
                let ulp1 = f32::from_bits(((oref[d] & 0x7fff) as u32 + 1) << 16) - bf(oref[d]).abs();
                let ok = u <= 2 || (bf(oref[d]).abs() < 1.0 / 64.0 && abs <= 2e-5) || (two_pass(n) && abs <= ew[d] / 256.0 + ulp1);
                if !ok {
                    st.fail += 1;
                    if st.fail <= 5 {
                        println!("    FAIL {} t {} (N {}) h {} d {}: ref {:e} gpu {:e} ({} ulp)", what, t, n, h, d, bf(oref[d]), bf(g), u);
                    }
                }
            }
        }
    }
    st
}
/// Two bf16 outputs against each other: <= 2 ulp, or abs <= 2e-5 where |a| < 1/64.
fn check_pair(a: &[u16], b: &[u16]) -> Stats {
    let mut st = Stats::default();
    for (&x, &y) in a.iter().zip(b) {
        let u = ulp_bf(x, y);
        let abs = (bf(x) - bf(y)).abs();
        st.n += 1;
        if u > 0 { st.diff += 1; }
        st.ulp = st.ulp.max(u);
        st.abs = st.abs.max(abs);
        if !(u <= 2 || (bf(x).abs() < 1.0 / 64.0 && abs <= 2e-5)) { st.fail += 1; }
    }
    st
}
/// Rows to check against the CPU: all for small T, else a strided sample + the tile edges + the last row.
fn sample_rows(t: usize) -> Vec<usize> {
    if t <= 64 { return (0..t).collect(); }
    let mut v: Vec<usize> = (0..t).filter(|&r| r % 13 == 0 || (r % 8 == 7 && r % 5 == 0) || r < 9).collect();
    v.push(t - 1);
    v.dedup();
    v
}

// ------------------------------------------------------------------------------------------- flash tests
/// One flash case: CPU reference on sampled rows, flash vs split-K on every row. Returns (vs CPU, vs split-K).
fn flash_case(gpu: &Gpu, s: Shape, t: usize, pos: usize, gated: bool, seed: u64) -> (Stats, Stats) {
    let c = Case::new(gpu, s, t, pos, gated, seed, 1.0);
    c.set_split_slots();
    run(gpu, &c.ar, |e| { c.flash(e, c.ao); c.split_k(e, c.ao2); }).unwrap();
    let fl: Vec<u16> = c.ar.get(c.ao, t * s.nh * HD);
    let sk: Vec<u16> = c.ar.get(c.ao2, t * s.nh * HD);
    let cpu = check_cpu(&c, &fl, &sample_rows(t), |_| false, "flash");
    let pair = check_pair(&fl, &sk);
    println!("  {:>2}/{} T {:>3} pos {:>4}{}: vs CPU max {} ulp, abs {:.1e}, {} / {} differ | vs split-K max {} ulp, abs {:.1e}, {:.2}% differ",
             s.nh, s.nkv, t, pos, if gated { " gated" } else { "" }, cpu.ulp, cpu.abs, cpu.diff, cpu.n, pair.ulp, pair.abs,
             100.0 * pair.diff as f64 / pair.n as f64);
    assert_eq!(cpu.fail, 0, "flash vs CPU outside tolerance: {:?}", cpu);
    assert_eq!(pair.fail, 0, "flash vs split-K outside tolerance: {:?}", pair);
    (cpu, pair)
}
fn flash_sweep(name: &str, s: Shape, cases: &[(usize, usize)]) {
    let t0 = std::time::Instant::now();
    let gpu = gpu();
    let (mut a, mut b) = (Stats::default(), Stats::default());
    for (i, &(t, pos)) in cases.iter().enumerate() {
        let (x, y) = flash_case(&gpu, s, t, pos, false, 7000 + 31 * s.nh as u64 + i as u64);
        a.add(x);
        b.add(y);
    }
    println!("{}: {} cases OK; vs CPU worst {} ulp / abs {:.1e}; vs split-K worst {} ulp / abs {:.1e}, {:.2}% of {} differ; {:.2} s",
             name, cases.len(), a.ulp, a.abs, b.ulp, b.abs, 100.0 * b.diff as f64 / b.n as f64, b.n, t0.elapsed().as_secs_f64());
}

// (T, pos): contexts pos + T from 1 to 1100, T up to 512
#[test]
fn flash_qwen3_1p7b_16_8() {
    flash_sweep("flash 16/8", Shape { nh: 16, nkv: 8 }, &[(1, 0), (3, 0), (8, 0), (9, 0), (17, 0), (16, 100), (33, 255), (64, 1000), (512, 0), (300, 800), (77, 1023)]);
}
#[test]
fn flash_qwen25_1p5b_12_2() {
    flash_sweep("flash 12/2", Shape { nh: 12, nkv: 2 }, &[(1, 0), (2, 5), (8, 248), (25, 231), (128, 128), (512, 588), (40, 1060)]);
}
#[test]
fn flash_32_8() {
    flash_sweep("flash 32/8", Shape { nh: 32, nkv: 8 }, &[(1, 1099), (7, 0), (100, 3), (256, 512), (512, 100)]);
}
#[test]
fn flash_gated() {
    let gpu = gpu();
    for (i, &(t, pos)) in [(9usize, 0usize), (40, 300)].iter().enumerate() {
        flash_case(&gpu, Shape { nh: 16, nkv: 4 }, t, pos, true, 9100 + i as u64);
    }
}

// ------------------------------------------------------------------------------------------- sdpa_vec tests
/// sdpa_vec over T rows at pos: vs CPU on every row; T-row dispatch vs T one-row dispatches bit-exact.
fn sdpa_case(gpu: &Gpu, s: Shape, t: usize, pos: usize, gated: bool, seed: u64) -> Stats {
    let c = Case::new(gpu, s, t, pos, gated, seed, 1.0);
    let lim = sdpa128_vec_max_n(s.grp());
    for r in 0..t { c.set_pos(1 + r, pos + r); }
    run(gpu, &c.ar, |e| {
        c.sdpa(e, 0, 0, t, c.ao, lim);
        for r in 0..t { c.sdpa(e, 1 + r, r, 1, c.ao2, lim); }
    }).unwrap();
    let all: Vec<u16> = c.ar.get(c.ao, t * s.nh * HD);
    let one: Vec<u16> = c.ar.get(c.ao2, t * s.nh * HD);
    assert_eq!(all, one, "sdpa_vec K-invariance");
        let rows: Vec<usize> = (0..t).collect();
    let st = check_cpu(&c, &all, &rows, |n| n >= SDPA128_TWO_PASS_MIN, "sdpa_vec");
    println!("  {:>2}/{} T {} pos {:>4}{}: vs CPU max {} ulp, abs {:.1e}, {} / {} differ (K-invariant)",
             s.nh, s.nkv, t, pos, if gated { " gated" } else { "" }, st.ulp, st.abs, st.diff, st.n);
    assert_eq!(st.fail, 0, "sdpa_vec vs CPU outside tolerance: {:?}", st);
    st
}
#[test]
fn sdpa_vec_shapes() {
    let t0 = std::time::Instant::now();
    let gpu = gpu();
    let mut i = 0;
    for s in [Shape { nh: 16, nkv: 8 }, Shape { nh: 12, nkv: 2 }, Shape { nh: 32, nkv: 8 }] {
        // single pass (N < 1024), straddling the 2-pass switch (N 1020..1027), 2-pass to N 1100 (blocks 64 or 128)
        for &(t, pos) in &[(1usize, 0usize), (1, 31), (4, 60), (8, 500), (8, 1019), (5, 1095)] {
            sdpa_case(&gpu, s, t, pos, false, 8000 + i);
            i += 1;
        }
    }
    sdpa_case(&gpu, Shape { nh: 16, nkv: 4 }, 3, 1022, true, 8100);
    println!("sdpa_vec: OK, {:.2} s", t0.elapsed().as_secs_f64());
}

// ------------------------------------------------------------------------------------------- MLX bit-identity
#[test]
fn sdpa_vec_mlx_bit_identity() {
    let path = std::env::var("QDENSE_MLX_SDPA128").unwrap_or_else(|_| MLX_FILE.to_string());
    let Ok(data) = std::fs::read(&path) else {
        println!("SKIP sdpa_vec_mlx_bit_identity: {} not found (write it with mlx_sdpa128.py, conda env streamdense-mlx)", path);
        return;
    };
    let t0 = std::time::Instant::now();
    let gpu = gpu();
    let u32at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let ncases = u32at(0) as usize;
    let mut off = 4;
    let (mut exact, mut checked, mut skipped, mut bad_elems) = (0, 0, 0, 0usize);
    let mut variant_note = Vec::new();
    for _ in 0..ncases {
        let (nh, nkv, n, sd) = (u32at(off) as usize, u32at(off + 4) as usize, u32at(off + 8) as usize, u32at(off + 12) as u64);
        let (aq, akv) = (f32::from_bits(u32at(off + 16)), f32::from_bits(u32at(off + 20)));
        off += 24;
        let want: Vec<u16> = (0..nh * HD).map(|i| u16::from_le_bytes([data[off + 2 * i], data[off + 2 * i + 1]])).collect();
        off += nh * HD * 2;
        let s = Shape { nh, nkv };
        let lim = sdpa128_vec_max_n(s.grp());
        // MLX's gqa 2-pass variant (grp 8/12/16, N >= 8192) is not mirrored: run once with lim forced to 8192 to show
        // it differs, and do not count it.
        let not_mirrored = n > lim;
        let mut ar = Arena::new(&gpu, 2 * nkv * n * HD * 2 + 2 * nh * HD * 2 + CTRL_STRIDE + (1 << 12)).unwrap();
        let (qr, kc, vc, ao, ctrl) = (ar.alloc(nh * HD * 2), ar.alloc(nkv * n * HD * 2), ar.alloc(nkv * n * HD * 2), ar.alloc(nh * HD * 2), ar.alloc(CTRL_STRIDE));
        ar.put(qr, &hash_bf(nh * HD, 3 * sd, aq));
        ar.put(kc, &hash_bf(nkv * n * HD, 3 * sd + 1, akv));
        ar.put(vc, &hash_bf(nkv * n * HD, 3 * sd + 2, akv));
        ar.put(ctrl, &[SdCtrl { pos: (n - 1) as u32, ..Default::default() }]);
        run(&gpu, &ar, |e| {
            e.pso("attn128_sdpa_vec").b(0, qr).b(1, kc).b(2, vc).b(3, qr).b(4, ao).b(5, ctrl)
                .u(6, n as u32).u(7, s.grp() as u32).u(8, 0).u(9, if not_mirrored { 8192 } else { lim as u32 })
                .u(10, SDPA128_TWO_PASS_MIN as u32).u(11, 0)
                .groups((nh, 1), 1024);
        }).unwrap();
        let got: Vec<u16> = ar.get(ao, nh * HD);
        let nd = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        if not_mirrored {
            skipped += 1;
            variant_note.push(format!("{}/{} N {}: {} of {} differ", nh, nkv, n, nd, nh * HD));
            continue;
        }
        checked += 1;
        if nd == 0 { exact += 1; } else {
            bad_elems += nd;
            let mu = got.iter().zip(&want).map(|(&a, &b)| ulp_bf(a, b)).max().unwrap();
            println!("  MISMATCH {}/{} N {} seed {} amp ({}, {}): {} of {} outputs differ, max {} ulp", nh, nkv, n, sd, aq, akv, nd, nh * HD, mu);
        }
    }
    println!("MLX bit-identity (mlx 0.32.4 sdpa, D 128, L 1): {} / {} cases bit-identical ({} differing outputs); \
              {} cases in MLX's unmirrored gqa variant skipped [{}]; {:.2} s",
             exact, checked, bad_elems, skipped, variant_note.join("; "), t0.elapsed().as_secs_f64());
    assert_eq!(exact, checked, "attn128_sdpa_vec not bit-identical to MLX");
}
