//! `verify_bench` (2026-11-03): the fused K-row verify matmuls (verify_fused_q4.metal `vf_xprep` + `vf_gemv_*` /
//! `vf_swiglu_*`) vs the CURRENT exact-pass kernels (gemv_mma_q4.metal through the engine's own `GemvImpl::MmaD`
//! choice; K > 8 = chunks of 8 rows, as a verify of more than 8 rows would have to run today) at the real projection
//! shapes of a Qwen3.5/3.8 config (reads `config.json` only, random weights, never the model's weights).
//! Reports:
//! - bit identity per shape and K (1..=16): new == old (m3 f32 bits, m0 / m2 bf16, swiglu bf16), new K rows == new
//!   K = 1 per row (K-invariance), and a CPU reference of the arithmetic definition on sampled rows;
//! - per shape, K and kernel variant (old, new NB 1 / 2 / 4): µs per dispatch (min over `--reps` command buffers,
//!   rotated over the weight copies so every dispatch streams from DRAM), weight GB/s;
//! - the full 27B-shaped matmul pass in ONE command buffer (every layer's mixer in / out projection, gate+up+swiglu,
//!   down, lm_head on all K rows; `--copies` weight sets per kind) for old, new and the best-per-shape plan, with
//!   pass ms, ms per row, weight GB/s and % of the measured read ceiling (`bw_read`) and of the 273 GB/s nominal.
//! Matmuls only: attention / DeltaNet core, norms and argmax are not in the pass (same for old and new).
//! RAM: copies x (one DeltaNet + one attention layer's projections + one FFN) + lm_head + scratch; 27B, 2 copies:
//! ~1.3 GB.
//! Run: cargo run --release -p streamdense-model --example verify_bench -- [--model DIR] [--ks 1,2,3,4,8,12,16]
//!      [--reps 5] [--copies 2] [--no-head] [--check-only]
//!      [--peak]: sustained MMA / FMA rate; [--kern a,b,..]: gate+up µs of named swiglu-layout kernels (vf_abl* ablations).
use streamdense_core::arena::Region;
use streamdense_metal::{commit, contents, gpu_ms, wait, Buffer, Enc, Gpu};

use crate::qwen_config::{LayerKind, QwenConfig};
pub use crate::qwen_layers::{vf_gemv_enc, vf_gemv_kernel, vf_pipelines, vf_prep_bytes, vf_prep_enc, vf_swiglu_enc, vf_swiglu_kernel, vf_tiles,
                             VF_GEMV_NBS, VF_K_MAX, VF_SWIGLU_NBS};
use crate::qwen_layers::{
    gemv, mma_pipelines, mma_swiglu_enc, mma_swiglu_kernel, mma_twin_nt, mma_twin_swiglu_enc, mma_twin_tmem, mma_use_twin,
    mmad_t4, mmad_t4_nt, mmad_t4_tmem, GemvImpl, MMA_NB_SWIGLU, MMA_SK,
};
use crate::weights::Q4b;

/// M4 Pro nominal unified-memory bandwidth (GB/s), for the % column.
const NOMINAL_GBS: f64 = 273.0;

// ---- the current exact-pass kernels, exactly as encode_layer runs them with `GemvImpl::MmaD` (decode_fused), and
// for k > 8 in chunks of 8 rows (no current kernel takes more than 8).
fn old_swiglu8(e: &Enc, gate: &Q4b, up: &Q4b, k: usize, x: usize, out: usize) {
    if mma_use_twin(gate.cols, true, k) {
        if mmad_t4(gate.cols, true, k) {
            let nt = mmad_t4_nt(gate.cols, true);
            e.tgmem(0, mmad_t4_tmem(nt, true));
            e.pso(&format!("qm_swiglutd_k1_t4_s{}", MMA_SK))
                .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off).b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
                .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32)
                .groups((gate.rows.div_ceil(4), 1), nt);
        } else {
            let nt = mma_twin_nt(gate.cols, true, k, MMA_SK, 512);
            e.tgmem(0, mma_twin_tmem(nt, true, k, MMA_SK));
            mma_twin_swiglu_enc(e, &format!("qm_swiglutd_k{}_s{}", k, MMA_SK), nt, gate, up, x, out);
        }
    } else {
        mma_swiglu_enc(e, &mma_swiglu_kernel(MMA_NB_SWIGLU, MMA_SK), MMA_NB_SWIGLU, MMA_SK, gate, up, k, x, out);
    }
}
/// Current verify matmul of `k` rows (any k: chunks of 8). `ob` = output bytes per element of `mode`.
pub fn old_gemv_enc(e: &Enc, m: &Q4b, k: usize, mode: u32, x: usize, out: usize) {
    let ob = if mode == 3 { 4 } else { 2 };
    for t0 in (0..k).step_by(8) {
        gemv(e, GemvImpl::MmaD, m, (k - t0).min(8), mode, x + t0 * m.cols * 2, out + t0 * m.rows * ob);
    }
}
pub fn old_swiglu_enc(e: &Enc, gate: &Q4b, up: &Q4b, k: usize, x: usize, out: usize) {
    for t0 in (0..k).step_by(8) {
        old_swiglu8(e, gate, up, (k - t0).min(8), x + t0 * gate.cols * 2, out + t0 * gate.rows * 2);
    }
}

// ---- host helpers
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 } // [0, 1)
    fn n(&mut self) -> f32 { (0..4).map(|_| self.f()).sum::<f32>() * 1.732 - 3.464 } // ~N(0, 1)
    /// ~N(0, 1), 30% of values scaled by 2^e, e in [-12, 12]: wide exponent range, so any change of summation order
    /// (x sums, the fma chain, the slice adds) shows up in the output bits.
    fn wide(&mut self) -> f32 {
        let v = self.n();
        if self.f() < 0.3 { v * (2.0f32).powi((self.next() % 25) as i32 - 12) } else { v }
    }
}
fn to_bf(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 { return ((u >> 16) | 0x0040) as u16; }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}
fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
fn rbf(f: f32) -> f32 { bf(to_bf(f)) }

struct Arena { buf: Buffer, ptr: *mut u8, top: usize, len: usize }
impl Arena {
    fn new(g: &Gpu, len: usize) -> Result<Arena, String> {
        let buf = g.buffer(len)?;
        Ok(Arena { ptr: contents(&buf), buf, top: 0, len })
    }
    fn alloc(&mut self, n: usize) -> Result<usize, String> {
        let o = self.top;
        self.top = (self.top + n + 255) & !255;
        if self.top > self.len { return Err(format!("bench arena too small ({} > {})", self.top, self.len)); }
        Ok(o)
    }
    #[allow(clippy::mut_from_ref)] // bench arena: disjoint offsets into one shared GPU buffer
    fn sl<T>(&self, off: usize, n: usize) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut T, n) } }
}

fn q4_bytes(r: usize, c: usize) -> usize { r * c / 2 + 2 * (r * c / 64 * 2) }

/// Fill `n` u32 words at `p` with xorshift noise on all cores.
fn fill_u32(p: *mut u8, n: usize, seed: u64) {
    let th = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(12);
    let per = n.div_ceil(th).max(1);
    let base = p as usize;
    std::thread::scope(|s| {
        for t in 0..th {
            let (a, b) = (t * per, ((t + 1) * per).min(n));
            if a >= b { continue; }
            s.spawn(move || {
                let sl = unsafe { std::slice::from_raw_parts_mut((base as *mut u32).add(a), b - a) };
                let mut r = Rng(seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(t as u64 + 1)) | 1);
                for x in sl { *x = r.next() as u32; }
            });
        }
    });
}

fn put_mat(a: &mut Arena, rng: &mut Rng, r: usize, c: usize) -> Result<Q4b, String> {
    let (wl, sl) = (r * c / 2, r * c / 64 * 2);
    let (w, s, b) = (a.alloc(wl)?, a.alloc(sl)?, a.alloc(sl)?);
    fill_u32(unsafe { a.ptr.add(w) }, r * c / 8, rng.next());
    let (ss, bb) = (a.sl::<u16>(s, r * c / 64), a.sl::<u16>(b, r * c / 64));
    for i in 0..r * c / 64 {
        let sc = 0.002 + 0.018 * rng.f();
        ss[i] = to_bf(sc);
        bb[i] = to_bf(-8.0 * sc + 0.01 * rng.n());
    }
    Ok(Q4b { rows: r, cols: c, w: Region { off: w, len: wl }, s: Region { off: s, len: sl }, b: Region { off: b, len: sl } })
}

fn run(g: &Gpu, a: &Arena, f: &dyn Fn(&Enc)) -> Result<f64, String> {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &a.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb)?;
    Ok(gpu_ms(&cb))
}
/// min over `reps` command buffers of (gpu ms / n) for n encodes of f(enc, i).
fn time(g: &Gpu, a: &Arena, reps: usize, n: usize, f: &dyn Fn(&Enc, usize)) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..reps { best = best.min(run(g, a, &|e| for i in 0..n { f(e, i) })? / n as f64); }
    Ok(best)
}

/// CPU reference of the arithmetic definition (verify_fused_q4.metal header) for row `r` of `m`, token row x
/// (bf16 [cols]); SK = 8 slices. Returns v (f32, before the mode's store).
pub fn cpu_row(w: &[u32], s: &[u16], b: &[u16], cols: usize, r: usize, x: &[u16]) -> f32 {
    let (g_n, c8) = (cols / 64, cols / 8);
    let mut v = 0.0f32;
    for k in 0..MMA_SK {
        let mut acc = 0.0f32;
        for g in k * g_n / MMA_SK..(k + 1) * g_n / MMA_SK {
            let mut p = 0.0f32;
            let mut pj = [0.0f32; 4];
            for c in 0..64 {
                let word = w[r * c8 + 8 * g + c / 8];
                let kk = (c % 8) as u32;
                let xv = bf(x[64 * g + c]);
                let a = xv * f32::from_bits((127 - 4 * kk) << 23);
                let q = (word & (0xf << (4 * kk))) as f32;
                p = a.mul_add(q, p);
                pj[(c % 8) / 2] += xv;
            }
            let xsum = (pj[0] + pj[1]) + (pj[2] + pj[3]);
            let (sc, bi) = (bf(s[r * g_n + g]), bf(b[r * g_n + g]));
            acc += sc.mul_add(p, xsum * bi);
        }
        v = if k == 0 { acc } else { v + acc };
    }
    v
}
/// CPU mirror of qm_silu_mul (MLX bf16 swiglu rounding chain).
pub fn cpu_silu_mul(gv: f32, uv: f32) -> u16 {
    let (g, u) = (rbf(gv), rbf(uv));
    let e = rbf(g.abs().exp());
    let d = rbf(1.0 + e);
    let y = rbf(1.0 / d);
    let sg = if g < 0.0 { y } else { rbf(1.0 - y) };
    to_bf(rbf(g * sg) * u)
}

/// One distinct projection shape: rows x cols, how many times a pass runs it, mode (None = swiglu pair).
#[derive(Clone)]
struct Shape { name: String, r: usize, c: usize, count: usize, mode: Option<u32> }

fn shapes_of(cfg: &QwenConfig, head: bool) -> Vec<Shape> {
    let (h, nd, na, nl) = (cfg.hidden, cfg.n_deltanet(), cfg.n_attention(), cfg.layers);
    let mut v: Vec<Shape> = Vec::new();
    let mut add = |name: &str, r: usize, c: usize, count: usize, mode: Option<u32>| {
        if count == 0 { return; }
        match v.iter_mut().find(|s| s.r == r && s.c == c && s.mode == mode) {
            Some(s) => { s.name = format!("{}|{}", s.name, name); s.count += count; }
            None => v.push(Shape { name: name.into(), r, c, count, mode }),
        }
    };
    add("dn_in", cfg.dn_in_rows(), h, nd, Some(0));
    add("dn_out", h, cfg.dn_v_dim(), nd, Some(2));
    add("attn_in", cfg.attn_in_rows(), h, na, Some(0));
    add("o_proj", h, cfg.q_dim(), na, Some(2));
    add("gate+up", cfg.inter, h, nl, None);
    add("down", h, cfg.inter, nl, Some(2));
    if head { add("lm_head", cfg.vocab, h, 1, Some(3)); }
    v
}

/// Kernel choice for one shape at one K: Old (current) or New(nb).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pick { Old, New(usize) }

fn enc_shape(e: &Enc, p: Pick, mode: Option<u32>, m0: &Q4b, m1: &Q4b, k: usize, x: usize, xt: usize, st: usize, out: usize) {
    match (p, mode) {
        (Pick::Old, Some(md)) => old_gemv_enc(e, m0, k, md, x, out),
        (Pick::Old, None) => old_swiglu_enc(e, m0, m1, k, x, out),
        (Pick::New(nb), Some(md)) => vf_gemv_enc(e, nb, m0, k, md, xt, st, out),
        (Pick::New(nb), None) => vf_swiglu_enc(e, nb, m0, m1, k, xt, st, out),
    }
}

fn library() -> Result<Gpu, String> {
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_mma_q4", "verify_fused_q4", "bench_bw"])?;
    for p in mma_pipelines() { g.pipeline(p)?; }
    for m in [0, 2, 3] { g.pipeline(&format!("qm_gemvtd_k1_t4_s{}_m{}", MMA_SK, m))?; }
    g.pipeline(&format!("qm_swiglutd_k1_t4_s{}", MMA_SK))?;
    for p in vf_pipelines() { g.pipeline(&p)?; }
    for p in ["vf_peak_mma", "vf_peak_fma", "bw_read"] { g.pipeline(p)?; }
    Ok(g)
}

/// Bit-identity of one shape (matrices m0 / m1, x [16][c] at `x`): for every K in 1..=16 and every NB, new == old
/// (each checked mode) and new == new K = 1 per row; CPU reference on `cpu_rows` sampled rows for K = 16.
/// Returns failure lines (empty = all identical).
#[allow(clippy::too_many_arguments)]
fn check_shape(g: &Gpu, a: &Arena, sh: &Shape, m0: &Q4b, m1: &Q4b, x: usize, xt: usize, st: usize, o1: usize, o2: usize, o3: usize,
               cpu_rows: usize) -> Result<Vec<String>, String> {
    let (r, c) = (sh.r, sh.c);
    let modes: Vec<Option<u32>> = if sh.mode.is_none() { vec![None] } else { vec![Some(3), Some(0), Some(2)] };
    let nbs: &[usize] = if sh.mode.is_none() { &VF_SWIGLU_NBS } else { &VF_GEMV_NBS };
    let mut bad = Vec::new();
    for md in modes {
        let ob = if md == Some(3) { 4 } else { 2 };
        let tag = md.map_or("swiglu".to_string(), |m| format!("m{}", m));
        let n = VF_K_MAX * r * ob;
        // m2 adds into the output: every run starts from the same residual pattern
        let reset = |o: usize| { let v = a.sl::<u16>(o, n / 2); for (i, e) in v.iter_mut().enumerate() { *e = to_bf(((i % 97) as f32 - 48.0) * 0.01); } };
        // new K = 1 per row (the K-invariance reference)
        reset(o3);
        run(g, a, &|e| for t in 0..VF_K_MAX {
            let ob1 = o3 + t * r * ob;
            vf_prep_enc(e, 1, c, x + t * c * 2, xt, st);
            match md { Some(m) => vf_gemv_enc(e, nbs[0], m0, 1, m, xt, st, ob1),
                       None => vf_swiglu_enc(e, nbs[0], m0, m1, 1, xt, st, ob1) }
        })?;
        for k in 1..=VF_K_MAX {
            if md == Some(2) { reset(o1); } else { a.sl::<u8>(o1, n).fill(0xa5); }
            run(g, a, &|e| match md { Some(m) => old_gemv_enc(e, m0, k, m, x, o1), None => old_swiglu_enc(e, m0, m1, k, x, o1) })?;
            run(g, a, &|e| vf_prep_enc(e, k, c, x, xt, st))?;
            for &nb in nbs {
                if md == Some(2) { reset(o2); } else { a.sl::<u8>(o2, n).fill(0x5a); }
                run(g, a, &|e| match md { Some(m) => vf_gemv_enc(e, nb, m0, k, m, xt, st, o2), None => vf_swiglu_enc(e, nb, m0, m1, k, xt, st, o2) })?;
                let (p, q, z) = (a.sl::<u8>(o1, k * r * ob), a.sl::<u8>(o2, k * r * ob), a.sl::<u8>(o3, k * r * ob));
                let d_old = p.chunks_exact(ob).zip(q.chunks_exact(ob)).filter(|(u, v)| u != v).count();
                let d_k1 = z.chunks_exact(ob).zip(q.chunks_exact(ob)).filter(|(u, v)| u != v).count();
                if d_old > 0 { bad.push(format!("{} {} K={} nb={}: {} / {} differ from the current kernels", sh.name, tag, k, nb, d_old, k * r)); }
                if d_k1 > 0 { bad.push(format!("{} {} K={} nb={}: {} / {} differ from K = 1 per row", sh.name, tag, k, nb, d_k1, k * r)); }
            }
        }
        // CPU reference (o2 holds the last nb's K = 16 output): sampled rows incl. the last
        if md == Some(3) || md.is_none() {
            let (w0, s0, b0) = (a.sl::<u32>(m0.w.off, r * c / 8), a.sl::<u16>(m0.s.off, r * c / 64), a.sl::<u16>(m0.b.off, r * c / 64));
            let (w1, s1, b1) = (a.sl::<u32>(m1.w.off, r * c / 8), a.sl::<u16>(m1.s.off, r * c / 64), a.sl::<u16>(m1.b.off, r * c / 64));
            let xs16 = a.sl::<u16>(x, VF_K_MAX * c);
            let mut nbad = 0;
            let rows: Vec<usize> = (0..cpu_rows).map(|i| if i + 1 == cpu_rows { r - 1 } else { i * r / cpu_rows }).collect();
            for &row in &rows {
                for t in 0..VF_K_MAX {
                    let xr = &xs16[t * c..(t + 1) * c];
                    let v = cpu_row(w0, s0, b0, c, row, xr);
                    let ok = match md {
                        Some(_) => a.sl::<f32>(o2, VF_K_MAX * r)[t * r + row].to_bits() == v.to_bits(),
                        None => a.sl::<u16>(o2, VF_K_MAX * r)[t * r + row] == cpu_silu_mul(v, cpu_row(w1, s1, b1, c, row, xr)),
                    };
                    if !ok { nbad += 1; }
                }
            }
            if nbad > 0 { bad.push(format!("{} {}: {} / {} sampled outputs differ from the CPU reference", sh.name, tag, nbad, rows.len() * VF_K_MAX)); }
        }
    }
    Ok(bad)
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let model = arg("--model").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4").into());
    let ks: Vec<usize> = match arg("--ks") {
        Some(v) => v.split(',').map(|x| x.trim().parse().map_err(|_| format!("--ks {}: want a,b,c", v))).collect::<Result<_, _>>()?,
        None => vec![1, 2, 3, 4, 8, 12, 16],
    };
    if let Some(k) = ks.iter().find(|k| **k == 0 || **k > VF_K_MAX) { return Err(format!("--ks: k {} not in 1..=16", k)); }
    let reps: usize = arg("--reps").map_or(Ok(5), |v| v.parse().map_err(|_| "--reps".to_string()))?;
    let copies: usize = arg("--copies").map_or(Ok(2), |v| v.parse().map_err(|_| "--copies".to_string()))?.max(1);
    let head = !args.iter().any(|a| a == "--no-head");
    let check_only = args.iter().any(|a| a == "--check-only");

    let t0 = std::time::Instant::now();
    let mut g = library()?;
    eprintln!("compiled gemv_mma_q4 + verify_fused_q4 in {:.2} s on {}", t0.elapsed().as_secs_f64(), g.name());
    if args.iter().any(|a| a == "--peak") { return peak(&g); }
    if args.iter().any(|a| a == "--attn") {
        let poss: Vec<usize> = arg("--pos").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or(vec![1024, 4000, 8000]);
        return attn_cli(&poss, &ks, reps);
    }
    if args.iter().any(|a| a == "--attn-pass") {
        let poss: Vec<usize> = arg("--pos").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or(vec![1024, 4096, 8192, 16384, 32768]);
        return attn_pass_cli(&poss, reps);
    }

    let cfg = QwenConfig::from_model_dir(&model)?;
    if let Some(names) = arg("--kern") { return kern(&mut g, &cfg, &names, &ks, reps); }
    let shapes = shapes_of(&cfg, head);
    // ---- arena: per kind `copies` weight sets (+ lm_head once), x, prep, outputs
    let wbytes: usize = shapes.iter().map(|s| {
        let per = q4_bytes(s.r, s.c) * if s.mode.is_none() { 2 } else { 1 };
        // merged shapes (dn_out|o_proj) get `copies` sets per source layer kind
        let sets = if s.name == "lm_head" { 1 } else { copies * s.name.split('|').count() };
        sets * (per + 3 * 256)
    }).sum();
    let (maxr, maxc) = (shapes.iter().map(|s| s.r).max().unwrap(), shapes.iter().map(|s| s.c).max().unwrap());
    let len = wbytes + VF_K_MAX * maxc * 2 * 3 + VF_K_MAX * maxc * 4 + VF_K_MAX * maxc / 16 + 3 * VF_K_MAX * maxr * 4 + (8 << 20);
    eprintln!("bench arena {:.2} GB ({} weight copies per kind{})", len as f64 / 1e9, copies, if head { " + lm_head" } else { "" });
    let mut a = Arena::new(&g, len)?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (x, xf) = (a.alloc(VF_K_MAX * maxc * 2)?, a.alloc(VF_K_MAX * maxc * 2)?);
    for v in a.sl::<u16>(x, VF_K_MAX * maxc) { *v = to_bf(rng.wide()); }
    let (xt, st) = (a.alloc(VF_K_MAX * maxc * 4)?, a.alloc(VF_K_MAX * maxc / 16 + 256)?);
    let (o1, o2, o3) = (a.alloc(VF_K_MAX * maxr * 4)?, a.alloc(VF_K_MAX * maxr * 4)?, a.alloc(VF_K_MAX * maxr * 4)?);
    let wbase = a.top;
    let mut mats: Vec<Vec<(Q4b, Q4b)>> = Vec::new();
    for s in &shapes {
        let sets = if s.name == "lm_head" { 1 } else { copies * s.name.split('|').count() };
        let mut v = Vec::with_capacity(sets);
        for _ in 0..sets {
            let m0 = put_mat(&mut a, &mut rng, s.r, s.c)?;
            let m1 = if s.mode.is_none() { put_mat(&mut a, &mut rng, s.r, s.c)? } else { m0 };
            v.push((m0, m1));
        }
        mats.push(v);
    }
    eprintln!("weights filled: {:.2} GB in {:.1} s", (a.top - wbase) as f64 / 1e9, t0.elapsed().as_secs_f64());

    // ---- bit identity (every shape, K = 1..16, every NB, modes; CPU reference on sampled rows)
    let mut bad: Vec<String> = Vec::new();
    for (si, s) in shapes.iter().enumerate() {
        let (m0, m1) = mats[si][0];
        let tc = std::time::Instant::now();
        let b = check_shape(&g, &a, s, &m0, &m1, x, xt, st, o1, o2, o3, if s.r > 100_000 { 4 } else { 6 })?;
        eprintln!("check {} {} x {}: {} ({:.1} s)", s.name, s.r, s.c, if b.is_empty() { "bit-identical" } else { "DIFFERS" }, tc.elapsed().as_secs_f64());
        bad.extend(b);
    }
    println!("Bit identity, K = 1..16, every NB, modes 3 / 0 / 2 and swiglu: new == current kernels, new K rows == new K = 1");
    println!("per row, CPU reference on sampled rows: {}", if bad.is_empty() { "ALL OK".to_string() } else { format!("FAIL\n  {}", bad.join("\n  ")) });
    if check_only { return if bad.is_empty() { Ok(()) } else { Err("bit identity failed".into()) }; }

    warmup(&g, &a, 700.0)?;
    // ---- read ceiling of this GPU (whole weight region)
    let wwords = (a.top - wbase) / 16;
    let bw_ms = time(&g, &a, reps.max(3), 1, &|e, _| {
        e.pso("bw_read").b(0, wbase).b(1, o3).u(2, wwords as u32).u(3, 0x1234_5678).threads((20 * 1024 * 4, 1), 1024);
    })?;
    let ceil_gbs = (wwords * 16) as f64 / bw_ms / 1e6;
    println!("\nRead ceiling (bw_read over {:.2} GB of weights): {:.1} GB/s (nominal {:.0})", (wwords * 16) as f64 / 1e9, ceil_gbs, NOMINAL_GBS);

    // ---- per shape x K x variant: µs per dispatch (rotated over the copies)
    println!("\n| shape | R x C | per pass | K | kernel | µs | weight GB/s | vs current |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut best: Vec<Vec<(Pick, f64, f64)>> = vec![Vec::new(); shapes.len()]; // per shape, per ks index: (pick, ms, old ms)
    let mut prep_ms: Vec<(usize, usize, f64)> = Vec::new(); // (cols, k, ms)
    for (si, s) in shapes.iter().enumerate() {
        let nc = mats[si].len();
        let n = if nc == 1 { 3 } else { 2 * nc };
        let wb = q4_bytes(s.r, s.c) * if s.mode.is_none() { 2 } else { 1 };
        let nbs: &[usize] = if s.mode.is_none() { &VF_SWIGLU_NBS } else { &VF_GEMV_NBS };
        for &k in &ks {
            let t_old = time(&g, &a, reps, n, &|e, i| { let (m0, m1) = mats[si][i % nc]; enc_shape(e, Pick::Old, s.mode, &m0, &m1, k, x, xt, st, o1) })?;
            println!("| {} | {} x {} | {} | {} | current | {:.0} | {:.0} | 1.00 |", s.name, s.r, s.c, s.count, k, t_old * 1e3, wb as f64 / t_old / 1e6);
            let mut b = (Pick::Old, t_old, t_old);
            for &nb in nbs {
                let t = time(&g, &a, reps, n, &|e, i| { let (m0, m1) = mats[si][i % nc]; enc_shape(e, Pick::New(nb), s.mode, &m0, &m1, k, x, xt, st, o1) })?;
                println!("| {} | {} x {} | {} | {} | vf nb{} | {:.0} | {:.0} | {:.2} |", s.name, s.r, s.c, s.count, k, nb, t * 1e3, wb as f64 / t / 1e6, t_old / t);
                if t < b.1 { b = (Pick::New(nb), t, t_old); }
            }
            best[si].push(b);
        }
    }
    let mut pcols = vec![cfg.hidden, cfg.dn_v_dim(), cfg.q_dim(), cfg.inter];
    pcols.dedup();
    for &c in &pcols {
        for &k in &ks {
            let t = time(&g, &a, reps, 20, &|e, _| vf_prep_enc(e, k, c, x, xt, st))?;
            prep_ms.push((c, k, t));
        }
    }
    println!("\nvf_xprep µs per dispatch (cols: K = {:?}):", ks);
    for &c in &pcols {
        let v: Vec<String> = ks.iter().map(|&k| format!("{:.1}", prep_ms.iter().find(|p| p.0 == c && p.1 == k).unwrap().2 * 1e3)).collect();
        println!("  {}: {}", c, v.join(" / "));
    }

    // ---- full 27B-shaped matmul pass in one command buffer: every layer in model order, copies rotated per kind
    let si_of = |n: &str| shapes.iter().position(|s| s.name.split('|').any(|p| p == n));
    let wtot: f64 = shapes.iter().map(|s| (s.count * q4_bytes(s.r, s.c) * if s.mode.is_none() { 2 } else { 1 }) as f64).sum();
    let pass = |plan: &dyn Fn(usize, usize) -> Pick, k: usize, ki: usize| -> Result<f64, String> {
        let enc_one = |e: &Enc, name: &str, cp: usize, x_in: usize, out: usize, prepped: &mut Option<usize>| {
            let si = si_of(name).unwrap();
            let s = &shapes[si];
            // the copy of this source kind: merged shapes keep their kinds' copies apart
            let part = s.name.split('|').position(|p| p == name).unwrap();
            let (m0, m1) = mats[si][(part * copies + cp) % mats[si].len()];
            let p = plan(si, ki);
            if let Pick::New(_) = p {
                if *prepped != Some(x_in) { vf_prep_enc(e, k, s.c, x_in, xt, st); *prepped = Some(x_in); }
            }
            enc_shape(e, p, s.mode, &m0, &m1, k, x_in, xt, st, out);
        };
        let mut best_ms = f64::MAX;
        for _ in 0..reps.min(3) {
            let ms = run(&g, &a, &|e| {
                let (mut na, mut nd) = (0usize, 0usize);
                for l in 0..cfg.layers {
                    let mut pp: Option<usize> = None;
                    // inputs: layer input = x rows; mixer out / down inputs use distinct buffers (o2 / o3 as x)
                    match cfg.kinds[l] {
                        LayerKind::DeltaNet => {
                            enc_one(e, "dn_in", nd % copies, x, o1, &mut pp);
                            enc_one(e, "dn_out", nd % copies, o2, o1, &mut pp);
                            nd += 1;
                        }
                        LayerKind::Attention => {
                            enc_one(e, "attn_in", na % copies, x, o1, &mut pp);
                            enc_one(e, "o_proj", na % copies, o2, o1, &mut pp);
                            na += 1;
                        }
                    }
                    enc_one(e, "gate+up", l % copies, xf, o1, &mut pp);
                    enc_one(e, "down", l % copies, o3, o1, &mut pp);
                }
                if head { let mut pp = None; enc_one(e, "lm_head", 0, x, o1, &mut pp); }
            })?;
            best_ms = best_ms.min(ms);
        }
        Ok(best_ms)
    };
    // the x / o2 / o3 inputs above are only timing data; make them finite bf16
    for o in [xf, o2, o3] { for v in a.sl::<u16>(o, VF_K_MAX * maxc) { *v = to_bf(0.5 * rng.n()); } }
    println!("\nFull 27B-shaped matmul pass, one command buffer ({} layers{}, {:.2} GB of weights per pass; GPU ms, best of {}):",
             cfg.layers, if head { " + lm_head on all K rows" } else { "" }, wtot / 1e9, reps.min(3));
    println!("| K | current ms | new ms | best-per-shape ms | plan | speedup | ms / row (best) | weight GB/s (best) | % of ceiling | % of {:.0} |", NOMINAL_GBS);
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (ki, &k) in ks.iter().enumerate() {
        let t_old = pass(&|_, _| Pick::Old, k, ki)?;
        let t_new = pass(&|si, _| Pick::New(if shapes[si].mode.is_none() { 1 } else { 2 }), k, ki)?;
        let t_best = pass(&|si, ki| best[si][ki].0, k, ki)?;
        let plan: Vec<String> = shapes.iter().enumerate().map(|(si, s)| format!("{}:{}", s.name.split('|').next().unwrap(),
            match best[si][ki].0 { Pick::Old => "cur".to_string(), Pick::New(nb) => format!("vf{}", nb) })).collect();
        let gbs = wtot / t_best / 1e6;
        println!("| {} | {:.1} | {:.1} | {:.1} | {} | {:.2}x | {:.2} | {:.0} | {:.0}% | {:.0}% |", k, t_old, t_new, t_best, plan.join(" "),
                 t_old / t_best, t_best / k as f64, gbs, 100.0 * gbs / ceil_gbs, 100.0 * gbs / NOMINAL_GBS);
    }
    Ok(())
}

/// Keep the GPU busy for ~`ms` so its clock has ramped up before timing.
fn warmup(g: &Gpu, a: &Arena, ms: f64) -> Result<(), String> {
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() * 1e3 < ms { run(g, a, &|e| e.pso("vf_peak_fma").b(0, 0).u(1, 2048).groups((20 * 64, 1), 256))?; }
    Ok(())
}

/// Kernel experiments on the gate+up shape: µs of the named swiglu-binding kernels (comma list; `vf_swiglu_n1_t1`
/// layout, e.g. the `vf_abl*` ablations) at each K, rotated over 4 weight copies.
fn kern(gg: &mut Gpu, cfg: &QwenConfig, names: &str, ks: &[usize], reps: usize) -> Result<(), String> {
    let (r, c, nc) = (cfg.inter, cfg.hidden, 4usize);
    for n in names.split(',') { gg.pipeline(n.trim())?; }
    let gg = &*gg;
    let len = nc * 2 * (q4_bytes(r, c) + 1024) + VF_K_MAX * c * 8 + VF_K_MAX * r * 4 + (4 << 20);
    let mut a = Arena::new(gg, len)?;
    let mut rng = Rng(7);
    let x = a.alloc(VF_K_MAX * c * 2)?;
    for v in a.sl::<u16>(x, VF_K_MAX * c) { *v = to_bf(rng.n()); }
    let (xt, st, out) = (a.alloc(VF_K_MAX * c * 4)?, a.alloc(VF_K_MAX * c / 16)?, a.alloc(VF_K_MAX * r * 2)?);
    let mats: Vec<(Q4b, Q4b)> = (0..nc).map(|_| Ok((put_mat(&mut a, &mut rng, r, c)?, put_mat(&mut a, &mut rng, r, c)?))).collect::<Result<_, String>>()?;
    warmup(gg, &a, 500.0)?;
    let wb = 2 * q4_bytes(r, c);
    println!("| kernel | K | µs | weight GB/s | MMA TFLOPS (8-row tiles) |");
    println!("|---|---|---|---|---|");
    for n in names.split(',').map(|n| n.trim()) {
        // NB and tiles from the name (`_n{NB}_t{NT}`); prep and FLOPs follow the kernel's tiles
        let num = |tag: &str| n.split('_').find_map(|p| p.strip_prefix(tag).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
        let (nb, nt) = (num("n"), num("t"));
        for &k in ks.iter().filter(|&&k| vf_tiles(k) == nt) {
            run(gg, &a, &|e| vf_prep_enc(e, k, c, x, xt, st))?;
            let t = time(gg, &a, reps, 2 * nc, &|e, i| {
                let (gm, um) = mats[i % nc];
                e.pso(n).b(0, gm.w.off).b(1, gm.s.off).b(2, gm.b.off).b(3, um.w.off).b(4, um.s.off).b(5, um.b.off)
                    .b(6, xt).b(7, out).u(8, r as u32).u(9, c as u32).u(10, k as u32).b(11, st)
                    .groups((r.div_ceil(8 * nb), 1), 32 * MMA_SK);
            })?;
            let fl = 2.0 * (2 * r * c * 8 * nt) as f64;
            println!("| {} | {} | {:.0} | {:.0} | {:.2} |", n, k, t * 1e3, wb as f64 / t / 1e6, fl / t / 1e9);
        }
    }
    Ok(())
}

/// Sustained simdgroup-MMA and scalar-FMA rate of this GPU.
fn peak(g: &Gpu) -> Result<(), String> {
    let a = Arena::new(g, 1 << 20)?;
    for (name, flop_per_iter_lane) in [("vf_peak_mma", 8.0 * 512.0 * 2.0 / 32.0), ("vf_peak_fma", 32.0)] {
        let (n, groups, tpg) = (4096u32, 20 * 64, 256usize);
        let ms = time(g, &a, 5, 1, &|e, _| e.pso(name).b(0, 0).u(1, n).groups((groups, 1), tpg))?;
        println!("{}: {:.3} ms, {:.2} TFLOPS", name, ms, flop_per_iter_lane * n as f64 * (groups * tpg) as f64 / ms / 1e9);
    }
    Ok(())
}

// ---- K-row verify attention (verify_mixers.metal) vs attn_sdpa_vec_b16, 27B attention shapes (nh 24, nkv 4, hd 256)
const AT_NH: usize = 24;
const AT_NKV: usize = 4;
const AT_HD: usize = 256;

/// `q8`: the same-shaped KV in q8 (u8 values k, v + half2 (scale, min) per 32-dim group ks, vs), random contents.
struct AttnRig { g: Gpu, a: Arena, b: crate::qwen_layers::VaBufs, sc: usize, ao2: usize, max_ctx: usize, po: usize, pml: usize, bytes: usize,
                 q8: [usize; 4] }

fn attn_rig(max_ctx: usize, seed: u64) -> Result<AttnRig, String> {
    use crate::qwen_layers::{va_scratch_bytes, vm_pipelines, VaBufs};
    let mut g = Gpu::from_shader_files(&["common_bf16", "deltanet", "attn_gated", "verify_mixers"])?;
    g.pipeline("attn_sdpa_vec_b16")?;
    for p in vm_pipelines() { g.pipeline(&p)?; }
    for t in 1..=8 {
        for q8 in [false, true] { g.pipeline(crate::qwen_layers::partial_name(t, q8))?; g.pipeline(crate::qwen_layers::split_name(t, q8))?; }
    }
    g.pipeline("attn_reduce_gate")?;
    let kvb = AT_NKV * max_ctx * AT_HD * 2;
    let prow = AT_NH * 2 * AT_HD + 2 * AT_NKV * AT_HD;
    let cfg_sc = 8 * AT_NH * crate::qwen_layers::VA_BMAX * (AT_HD * 2 + 8);
    let _ = va_scratch_bytes;
    // split-K partials for 8 rows over every chunk of max_ctx: po f32 [8][nh][nch][hd], pml f32x2 [8][nh][nch]
    let nch = max_ctx.div_ceil(crate::qwen_engine::ATTN_CHUNK);
    let (pob, pmlb) = (8 * AT_NH * nch * AT_HD * 4, 8 * AT_NH * nch * 8);
    let (kqb, ksb) = (AT_NKV * max_ctx * AT_HD, AT_NKV * max_ctx * (AT_HD / 32) * 4);
    let len = 2 * kvb + 2 * (kqb + ksb) + 8 * prow * 2 + 3 * 8 * AT_NH * AT_HD * 2 + cfg_sc + pob + pmlb + (4 << 20);
    let mut a = Arena::new(&g, len)?;
    let (k, v) = (a.alloc(kvb)?, a.alloc(kvb)?);
    let (qr, p, ao, ao2) = (a.alloc(8 * AT_NH * AT_HD * 2)?, a.alloc(8 * prow * 2)?, a.alloc(8 * AT_NH * AT_HD * 2)?, a.alloc(8 * AT_NH * AT_HD * 2)?);
    let ctrl = a.alloc(std::mem::size_of::<streamdense_metal::SdCtrl>())?;
    let sc = a.alloc(cfg_sc)?;
    let (po, pml) = (a.alloc(pob)?, a.alloc(pmlb)?);
    let q8 = [a.alloc(kqb)?, a.alloc(kqb)?, a.alloc(ksb)?, a.alloc(ksb)?];
    let mut rng = Rng(seed);
    // keys / queries ~N(0, 1) with a few large values: peaked softmaxes and wide exponents in the partial sums
    for o in [k, v] { for x in a.sl::<u16>(o, kvb / 2) { *x = to_bf(if rng.f() < 0.02 { 6.0 * rng.n() } else { rng.n() }); } }
    for x in a.sl::<u16>(qr, 8 * AT_NH * AT_HD) { *x = to_bf(2.0 * rng.n()); }
    for x in a.sl::<u16>(p, 8 * prow) { *x = to_bf(3.0 * rng.n()); }
    // q8 KV: random codes; per group fp16 scale ~ range / 255 and min ~ -range / 2 (range 1..13, a few groups with
    // large ranges), plus zero / signed-zero / subnormal scales and mins scattered through
    for o in [q8[0], q8[1]] { for x in a.sl::<u8>(o, kqb) { *x = (rng.next() >> 56) as u8; } }
    let f16 = streamdense_core::f16::f32_to_f16;
    for o in [q8[2], q8[3]] {
        for (i, x) in a.sl::<u16>(o, ksb / 2).chunks_exact_mut(2).enumerate() {
            let r = if rng.f() < 0.02 { 12.0 + 40.0 * rng.f() } else { 1.0 + 12.0 * rng.f() };
            x[0] = f16(r / 255.0);
            x[1] = f16(-r * (0.3 + 0.4 * rng.f()));
            match i % 211 { 3 => x[0] = 0, 5 => x[1] = 0x8000, 7 => { x[0] = 0x0001; x[1] = 0x0000 } 11 => x[0] = 0x03ff, _ => {} }
        }
    }
    Ok(AttnRig { g, a, b: VaBufs { qr, k, v, p, ao, ctrl }, sc, ao2, max_ctx, po, pml, bytes: len, q8 })
}

impl AttnRig {
    fn set_pos(&self, pos: usize) {
        let c = self.a.sl::<streamdense_metal::SdCtrl>(self.b.ctrl, 1);
        c[0] = streamdense_metal::SdCtrl { pos: pos as u32, ..Default::default() };
    }
    fn old(&self, e: &Enc, t: usize, ao: usize) {
        let prow = AT_NH * 2 * AT_HD + 2 * AT_NKV * AT_HD;
        let b = &self.b;
        e.pso("attn_sdpa_vec_b16").b(0, b.qr).b(1, b.k).b(2, b.v).b(3, b.p).b(4, ao).b(5, b.ctrl)
            .u(6, self.max_ctx as u32).u(7, (AT_NH / AT_NKV) as u32).u(8, prow as u32).u(9, crate::qwen_layers::sdpa_vec_max_n().min(8192) as u32)
            .u(10, crate::qwen_layers::sdpa_two_pass_min() as u32)
            .groups((AT_NH, t), 1024);
    }
    fn fused(&self, e: &Enc, t: usize, r: usize, ao: usize) {
        let prow = AT_NH * 2 * AT_HD + 2 * AT_NKV * AT_HD;
        let b = crate::qwen_layers::VaBufs { ao, ..self.b };
        crate::qwen_layers::va_sdpa_enc(e, &b, self.sc, r, t, AT_NH, AT_NKV, AT_HD, self.max_ctx, prow,
            crate::qwen_layers::sdpa_vec_max_n().min(8192), crate::qwen_layers::sdpa_two_pass_min());
    }
    /// Mismatching bf16 outputs of the new path (every R) vs attn_sdpa_vec_b16 for t rows at `pos`; also vs the old
    /// kernel run one row at a time (K-invariance of the reference itself).
    fn check(&self, pos: usize, t: usize) -> Result<Vec<String>, String> {
        let n = t * AT_NH * AT_HD;
        let mut bad = vec![];
        self.set_pos(pos);
        self.a.sl::<u8>(self.b.ao, 2 * n).fill(0xa5);
        run(&self.g, &self.a, &|e| self.old(e, t, self.b.ao))?;
        for &r in &crate::qwen_layers::VA_RS {
            self.a.sl::<u8>(self.ao2, 2 * n).fill(0xa5); // rows with N > lim stay unwritten in both
            run(&self.g, &self.a, &|e| self.fused(e, t, r, self.ao2))?;
            let (x, y) = (self.a.sl::<u16>(self.b.ao, n), self.a.sl::<u16>(self.ao2, n));
            let d = x.iter().zip(y.iter()).filter(|(u, v)| u != v).count();
            if d > 0 { bad.push(format!("attn pos {} T {} R {}: {} / {} outputs differ", pos, t, r, d, n)); }
        }
        Ok(bad)
    }
}

impl AttnRig {
    /// `t` rows of the engine's attention (`attn_sdpa_rows`: split-K for rows with N > lim, then the verify mirror
    /// for rows with N <= lim) at ctrl.pos = `pos`, reading qr / p rows `row0..row0 + t`, writing ao rows from `ao`.
    /// `vm`: verify mixers on (va_sdpa rows for N <= lim) or off (attn_sdpa_vec_b16 rows). `q8`: on the q8 KV.
    #[allow(clippy::too_many_arguments)]
    fn pass(&self, e: &Enc, pos: usize, t: usize, row0: usize, ao: usize, nch: usize, vm: bool, q8: bool) {
        let prow = AT_NH * 2 * AT_HD + 2 * AT_NKV * AT_HD;
        let rb = AT_NH * AT_HD * 2;
        let b = crate::qwen_layers::VaBufs { qr: self.b.qr + row0 * rb, p: self.b.p + row0 * prow * 2, ao, ..self.b };
        let [k, v, ksb, vsb] = if q8 { self.q8 } else { [self.b.k, self.b.v, self.b.k, self.b.v] };
        let b = crate::qwen_layers::VaBufs { k, v, ..b };
        let kv = crate::qwen_layers::KvRef { k, v, ksb, vsb, q8, max_ctx: self.max_ctx };
        crate::qwen_layers::attn_sdpa_rows(e, &kv, &b, self.po, self.pml, pos, t, nch, AT_NH, AT_NKV, AT_HD, prow, vm.then_some(self.sc));
    }
    /// Split-K partials (kernel `name`, `threads` per threadgroup) + attn_reduce_gate for all `t` rows at ctrl.pos,
    /// whatever their N (the split-K kernels are defined for every N).
    /// `q8`: on the q8 KV (kernel `name` must be a q8 variant).
    #[allow(clippy::too_many_arguments)]
    fn splitk(&self, e: &Enc, t: usize, ao: usize, nch: usize, name: &str, threads: usize, q8: bool) {
        let prow = AT_NH * 2 * AT_HD + 2 * AT_NKV * AT_HD;
        let b = &self.b;
        let [k, v, ks, vs] = if q8 { self.q8 } else { [b.k, b.v, b.k, b.v] };
        e.pso(name).b(0, b.qr).b(1, k).b(2, v).b(3, ks).b(4, vs).b(5, self.po).b(6, self.pml).b(7, b.ctrl)
            .u(8, self.max_ctx as u32).u(9, (AT_NH / AT_NKV) as u32).groups((nch, AT_NH), threads);
        e.pso("attn_reduce_gate").b(0, self.po).b(1, self.pml).b(2, b.p).b(3, ao).u(4, nch as u32).u(5, prow as u32)
            .groups((AT_NH, t), AT_HD / 4);
    }
    /// `attn_split_k{t}[_q8]` == the original `attn_partial_k{t}_{b16,q8}`, bit for bit over all t rows, at the minimal
    /// chunk count and with every chunk of max_ctx dispatched.
    #[cfg(test)]
    fn check_splitk(&self, pos: usize, t: usize, q8: bool) -> Result<Vec<String>, String> {
        use crate::qwen_engine::ATTN_CHUNK;
        let n = t * AT_NH * AT_HD;
        let (name, threads) = (crate::qwen_layers::split_name(t, q8), 256);
        let old = crate::qwen_layers::partial_name(t, q8);
        let mut bad = vec![];
        self.set_pos(pos);
        for nch in [(pos + t).div_ceil(ATTN_CHUNK), self.max_ctx.div_ceil(ATTN_CHUNK)] {
            self.a.sl::<u8>(self.ao2, 2 * n).fill(0x5a);
            run(&self.g, &self.a, &|e| self.splitk(e, t, self.ao2, nch, old, 128, q8))?;
            self.a.sl::<u8>(self.b.ao, 2 * n).fill(0xa5);
            run(&self.g, &self.a, &|e| self.splitk(e, t, self.b.ao, nch, name, threads, q8))?;
            let (x, y) = (self.a.sl::<u16>(self.b.ao, n), self.a.sl::<u16>(self.ao2, n));
            let d = x.iter().zip(y.iter()).filter(|(u, v)| u != v).count();
            if d > 0 { bad.push(format!("split-K {} vs {}, pos {} nch {}: {} / {} outputs differ", name, old, pos, nch, d, n)); }
        }
        Ok(bad)
    }
    /// K-invariance of the whole attention path at long contexts: a t-row pass at `pos` (chunks just covering
    /// pos + t, and again with every chunk of max_ctx dispatched) == t one-row passes at pos + i (each with its own
    /// minimal chunk count), bit for bit over all t rows. Returns (failures, max |diff| over mismatching outputs).
    /// `q8`: the same on the q8 KV (split-K for every row).
    fn check_rows(&self, pos: usize, t: usize, q8: bool) -> Result<(Vec<String>, f32), String> {
        use crate::qwen_engine::ATTN_CHUNK;
        let rb = AT_NH * AT_HD * 2;
        let n = t * AT_NH * AT_HD;
        // reference: one row per command buffer (ctrl.pos changes between rows); sentinel differs from the pass
        // buffer's so a row left unwritten on either side shows up as a mismatch
        self.a.sl::<u8>(self.ao2, 2 * n).fill(0x5a);
        for i in 0..t {
            self.set_pos(pos + i);
            run(&self.g, &self.a, &|e| self.pass(e, pos + i, 1, i, self.ao2 + i * rb, (pos + i + 1).div_ceil(ATTN_CHUNK), true, q8))?;
        }
        let mut bad = vec![];
        let mut worst = 0f32;
        self.set_pos(pos);
        for nch in [(pos + t).div_ceil(ATTN_CHUNK), self.max_ctx.div_ceil(ATTN_CHUNK)] {
            self.a.sl::<u8>(self.b.ao, 2 * n).fill(0xa5);
            run(&self.g, &self.a, &|e| self.pass(e, pos, t, 0, self.b.ao, nch, true, q8))?;
            let (x, y) = (self.a.sl::<u16>(self.b.ao, n), self.a.sl::<u16>(self.ao2, n));
            let mut rows = vec![];
            for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                if u != v {
                    worst = worst.max((bf(*u) - bf(*v)).abs());
                    let r = i / (AT_NH * AT_HD);
                    if !rows.contains(&r) { rows.push(r); }
                }
            }
            let d = x.iter().zip(y.iter()).filter(|(u, v)| u != v).count();
            if d > 0 { bad.push(format!("attn rows{} pos {} T {} nch {}: {} / {} outputs differ (rows {:?})", if q8 { " q8" } else { "" }, pos, t, nch, d, n, rows)); }
        }
        Ok((bad, worst))
    }
}

/// Bit identity + GPU µs of the verify attention vs attn_sdpa_vec_b16 (one layer: 24 q heads, 4 kv heads, hd 256).
fn attn_cli(poss: &[usize], ks: &[usize], reps: usize) -> Result<(), String> {
    let max_ctx = poss.iter().max().copied().unwrap_or(1024) + 16;
    let rig = attn_rig(max_ctx, 11)?;
    let mut bad = vec![];
    println!("verify attention rig: arena {:.1} MB, max_ctx {}", rig.bytes as f64 / 1e6, max_ctx);
    for &pos in poss {
        for &t in ks.iter().filter(|&&t| (1..=8).contains(&t)) { bad.extend(rig.check(pos, t)?); bad.extend(rig.check_rows(pos, t, false)?.0); }
    }
    println!("verify attention bit identity (pos {:?}, K {:?}, R {:?}): {}", poss, ks, crate::qwen_layers::VA_RS,
             if bad.is_empty() { "ALL OK".into() } else { format!("FAIL\n  {}", bad.join("\n  ")) });
    println!("| pos | K | attn_sdpa_vec_b16 µs | {} |", crate::qwen_layers::VA_RS.iter().map(|r| format!("va R{} µs", r)).collect::<Vec<_>>().join(" | "));
    println!("|---|---|---|{}", "---|".repeat(crate::qwen_layers::VA_RS.len()));
    for &pos in poss {
        rig.set_pos(pos);
        for &t in ks.iter().filter(|&&t| (1..=8).contains(&t)) {
            let o = time(&rig.g, &rig.a, reps, 16, &|e, _| rig.old(e, t, rig.b.ao))?;
            let v: Vec<String> = crate::qwen_layers::VA_RS.iter().map(|&r| time(&rig.g, &rig.a, reps, 16, &|e, _| rig.fused(e, t, r, rig.ao2))
                .map(|x| format!("{:.0}", x * 1e3))).collect::<Result<_, _>>()?;
            println!("| {} | {} | {:.0} | {} |", pos, t, o * 1e3, v.join(" | "));
            if std::env::var("SD_VA_SPLIT").is_ok() {
                // pass-1 kernel alone per R (dispatch filter drops va_sdpa_p2)
                let v: Vec<String> = crate::qwen_layers::VA_RS.iter().map(|&r| {
                    let mut best = f64::MAX;
                    for _ in 0..reps {
                        let cb = rig.g.command_buffer();
                        let f = |_: usize, n: &str| n.starts_with("va_sdpa_p1");
                        let e = Enc::with_filter(&rig.g, &cb, &rig.a.buf, &f);
                        for _ in 0..16 { rig.fused(&e, t, r, rig.ao2); }
                        e.end();
                        commit(&cb);
                        wait(&cb)?;
                        best = best.min(gpu_ms(&cb) / 16.0);
                    }
                    Ok(format!("{:.0}", best * 1e3))
                }).collect::<Result<_, String>>()?;
                println!("| {} | {} | p1 only | {} |", pos, t, v.join(" | "));
            }
        }
    }
    if bad.is_empty() { Ok(()) } else { Err("verify attention differs".into()) }
}

/// GPU ms of one layer's engine attention path (`attn_sdpa_rows`, as a verify pass of K rows runs it: split-K for
/// rows with N > 8192, else attn_sdpa_vec_b16 / va_sdpa with verify mixers) for K = 1..8 at kv length N ~ pos,
/// median over `reps` command buffers of 4 passes each, then the split-K kernels alone (attn_partial vs
/// attn_split_rows, + attn_reduce_gate). One layer of the 27B (nh 24, nkv 4, hd 256), bf16 KV, then the same for q8 KV.
fn attn_pass_cli(poss: &[usize], reps: usize) -> Result<(), String> {
    use crate::qwen_engine::ATTN_CHUNK;
    let max_ctx = poss.iter().max().copied().unwrap_or(1024) + 16;
    let rig = attn_rig(max_ctx, 11)?;
    println!("attention pass rig: arena {:.1} MB, max_ctx {}, reps {}", rig.bytes as f64 / 1e6, max_ctx, reps.max(3));
    println!("| N (pos) | path | K=1 µs | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=8 / K=1 |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    // median GPU µs of one pass (4 per command buffer, `reps` buffers) for t = 1..8
    let med = |f: &dyn Fn(&Enc, usize)| -> Result<Vec<f64>, String> {
        let mut us = vec![];
        for t in 1..=8usize {
            run(&rig.g, &rig.a, &|e| f(e, t))?; // warm
            let mut v: Vec<f64> = (0..reps.max(3))
                .map(|_| run(&rig.g, &rig.a, &|e| for _ in 0..4 { f(e, t) }).map(|x| x / 4.0))
                .collect::<Result<_, _>>()?;
            v.sort_by(|a, b| a.total_cmp(b));
            us.push(v[v.len() / 2] * 1e3);
        }
        Ok(us)
    };
    let row = |pos: usize, name: &str, us: &[f64]| println!("| {} | {} | {} | {:.2}x |", pos, name,
        us.iter().map(|x| format!("{:.0}", x)).collect::<Vec<_>>().join(" | "), us[7] / us[0]);
    for &pos in poss {
        rig.set_pos(pos);
        for vm in [false, true] {
            let us = med(&|e, t| rig.pass(e, pos, t, 0, rig.b.ao, (pos + t).div_ceil(ATTN_CHUNK), vm, false))?;
            row(pos, if vm { "verify_mixers" } else { "default" }, &us);
        }
        let us = med(&|e, t| rig.pass(e, pos, t, 0, rig.b.ao, (pos + t).div_ceil(ATTN_CHUNK), false, true))?;
        row(pos, "q8 default", &us);
        // the split-K kernels alone (+ attn_reduce_gate) for every row, whatever N: before / after attn_split_rows
        for q8 in [false, true] {
            let kv = if q8 { "q8 " } else { "" };
            let us = med(&|e, t| rig.splitk(e, t, rig.b.ao, (pos + t).div_ceil(ATTN_CHUNK), crate::qwen_layers::partial_name(t, q8), 128, q8))?;
            row(pos, &format!("{}split-K attn_partial", kv), &us);
            let us = med(&|e, t| rig.splitk(e, t, rig.b.ao, (pos + t).div_ceil(ATTN_CHUNK), crate::qwen_layers::split_name(t, q8), 256, q8))?;
            row(pos, &format!("{}split-K attn_split", kv), &us);
        }
    }
    Ok(())
}

/// dn_recur_v_ck{0,1} vs dn_recur_ck{0,1} at the 27B DeltaNet shapes (kh 16, vh 48, dk = dv = 128), T = 1..8 rows,
/// both state roles: y, the state copy in place and the row-0 checkpoint copy must be byte-identical. Returns failures.
pub fn dn_recur_check(seed: u64) -> Result<Vec<String>, String> {
    use crate::qwen_layers::DnDims;
    let (kh, vh, d) = (16usize, 48usize, 128usize);
    let conv = 2 * kh * d + vh * d;
    let dd = DnDims { kh: kh as u32, vh: vh as u32, conv: conv as u32, prow: (conv + vh * d + 2 * vh) as u32, b_off: (conv + vh * d) as u32,
                      a_off: (conv + vh * d + vh) as u32, layers: 1, pad: 0 };
    let mut g = Gpu::from_shader_files(&["common_bf16", "deltanet", "attn_gated", "verify_mixers"])?;
    for p in ["dn_recur_ck0", "dn_recur_ck1", "dn_recur_v_ck0", "dn_recur_v_ck1"] { g.pipeline(p)?; }
    let sb = 2 * vh * d * d * 4;
    let mut a = Arena::new(&g, 3 * sb + 8 * conv * 2 + 8 * vh * 8 + 2 * 8 * vh * d * 2 + (1 << 20))?;
    let (s0, s1, sref) = (a.alloc(sb)?, a.alloc(sb)?, a.alloc(sb)?);
    let (cv, gg, be, y0, y1) = (a.alloc(8 * conv * 2)?, a.alloc(8 * vh * 4)?, a.alloc(8 * vh * 2)?, a.alloc(8 * vh * d * 2)?, a.alloc(8 * vh * d * 2)?);
    let ctrl = a.alloc(std::mem::size_of::<streamdense_metal::SdCtrl>())?;
    let mut rng = Rng(seed);
    for x in a.sl::<f32>(sref, sb / 4) { *x = rng.wide() * 0.1; }
    for x in a.sl::<u16>(cv, 8 * conv) { *x = to_bf(rng.wide() * 0.2); }
    for x in a.sl::<f32>(gg, 8 * vh) { *x = 0.5 + 0.5 * rng.f(); }
    for x in a.sl::<u16>(be, 8 * vh) { *x = to_bf(rng.f()); }
    let mut bad = vec![];
    for role in [0u32, 1] {
        a.sl::<streamdense_metal::SdCtrl>(ctrl, 1)[0] = streamdense_metal::SdCtrl { role, ..Default::default() };
        for ck in [false, true] {
            for t in 1..=8usize {
                for (s, y, name) in [(s0, y0, if ck { "dn_recur_ck1" } else { "dn_recur_ck0" }), (s1, y1, if ck { "dn_recur_v_ck1" } else { "dn_recur_v_ck0" })] {
                    let src = a.sl::<u8>(sref, sb).to_vec();
                    a.sl::<u8>(s, sb).copy_from_slice(&src);
                    a.sl::<u8>(y, 8 * vh * d * 2).fill(0x77);
                    run(&g, &a, &|e| {
                        e.pso(name).b(0, cv).b(1, gg).b(2, be).b(3, s).b(4, y).b(5, ctrl).u(6, t as u32).bytes(7, &dd).groups((4 * vh, 1), 128);
                    })?;
                }
                let ds = a.sl::<u8>(s0, sb).iter().zip(a.sl::<u8>(s1, sb).iter()).filter(|(p, q)| p != q).count();
                let dy = a.sl::<u8>(y0, 8 * vh * d * 2).iter().zip(a.sl::<u8>(y1, 8 * vh * d * 2).iter()).filter(|(p, q)| p != q).count();
                if ds + dy > 0 { bad.push(format!("dn_recur role {} ck {} T {}: state {} / y {} bytes differ", role, ck, t, ds, dy)); }
            }
        }
    }
    Ok(bad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small matrices at the 27B column widths (5120, 17408), odd row counts: the fused kernels are bit-identical to
    /// the current exact-pass kernels and to the CPU definition for K = 1..16, every NB, every mode.
    #[test]
    fn verify_fused_bit_identical() {
        let g = library().expect("compile");
        let len = 64 << 20;
        let mut a = Arena::new(&g, len).unwrap();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let maxc = 17408;
        let x = a.alloc(VF_K_MAX * maxc * 2).unwrap();
        for v in a.sl::<u16>(x, VF_K_MAX * maxc) { *v = to_bf(rng.wide()); }
        let (xt, st) = (a.alloc(VF_K_MAX * maxc * 4).unwrap(), a.alloc(VF_K_MAX * maxc / 16).unwrap());
        let maxr = 72;
        let (o1, o2, o3) = (a.alloc(VF_K_MAX * maxr * 4).unwrap(), a.alloc(VF_K_MAX * maxr * 4).unwrap(), a.alloc(VF_K_MAX * maxr * 4).unwrap());
        let mut bad = Vec::new();
        for (r, c, mode) in [(72, 5120, Some(0)), (40, 17408, Some(2)), (37, 6144, Some(3)), (44, 5120, None)] {
            let sh = Shape { name: format!("{}x{}", r, c), r, c, count: 1, mode };
            let m0 = put_mat(&mut a, &mut rng, r, c).unwrap();
            let m1 = if mode.is_none() { put_mat(&mut a, &mut rng, r, c).unwrap() } else { m0 };
            bad.extend(check_shape(&g, &a, &sh, &m0, &m1, x, xt, st, o1, o2, o3, r).unwrap());
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// dn_recur_v (verify mixers) == dn_recur at the 27B DeltaNet shapes, T 1..8, ck 0 / 1, both roles.
    #[test]
    fn verify_dn_recur_bit_identical() {
        let bad = dn_recur_check(5).unwrap();
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// Verify attention (va_sdpa_p1 + p2, every R) == attn_sdpa_vec_b16 bit for bit, K = 1..8, positions across the
    /// single-pass / two-pass switch (1024), the 64 -> 128 block switch and the lim (8192) edge.
    #[test]
    fn verify_attn_bit_identical() {
        let rig = attn_rig(8200, 3).expect("rig");
        let mut bad = vec![];
        for pos in [0usize, 37, 1015, 1019, 1022, 1023, 1024, 1025, 1500, 4093, 8183, 8186, 8191] {
            for t in 1..=8 { if pos + t <= 8200 { bad.extend(rig.check(pos, t).unwrap()); } }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// Long contexts up to 32768 keys: the full attention path (split-K above lim = 8192, verify mirror below, both
    /// in a straddling pass; q8 KV: split-K for every row) of a K-row pass == K one-row passes, bit for bit, K = 1..8. Positions straddle lim, cross
    /// the 256-key chunk edges (N = 256 c, 256 c + 1), sit mid range and reach N = 32768.
    #[test]
    fn verify_attn_long_bit_identical() {
        const MAX: usize = 32768;
        let rig = attn_rig(MAX, 7).expect("rig");
        println!("verify_attn_long: rig arena {:.1} MB (KV {:.1} MB)", rig.bytes as f64 / 1e6, (2 * AT_NKV * MAX * AT_HD * 2) as f64 / 1e6);
        let mut bad = vec![];
        let mut worst = 0f32;
        let mut checked = 0;
        for pos in [37usize, 1023, 8183, 8185, 8188, 8191, 8192, 8193, 8200, 8447, 8448, 8449, 8703, 8704, 12287, 12288, 12345,
                    16383, 16384, 16385, 20000, 24575, 24576, 24577, 32511, 32512, 32760, 32764, 32767] {
            for t in 1..=8 {
                if pos + t <= MAX {
                    for q8 in [false, true] {
                        let (b, w) = rig.check_rows(pos, t, q8).unwrap();
                        bad.extend(b);
                        worst = worst.max(w);
                        checked += 1;
                    }
                }
            }
        }
        println!("verify_attn_long: {} (pos, T, kv) passes checked, {} failures, max |diff| {}", checked, bad.len(), worst);
        assert!(bad.is_empty(), "max |diff| {}\n{}", worst, bad.join("\n"));
    }
    /// The split-K kernel the engine runs for K rows == the original `attn_partial_k{K}_b16`, bit for bit, K = 1..8,
    /// up to 32768 keys: chunk edges (N = 256 c, 256 c + 1, rows straddling a chunk), the first chunks, partial last
    /// chunks of every fill, and every chunk of max_ctx dispatched (empty chunks).
    #[test]
    fn verify_attn_splitk_bit_identical() {
        const MAX: usize = 32768;
        let rig = attn_rig(MAX, 13).expect("rig");
        // signed zeros and bf16 subnormals (converted with float(bfloat) here, with bit shifts in attn_partial):
        // scattered through K and V, and kv head 2's keys / kv head 3's values all subnormal or zero
        let kvn = AT_NKV * MAX * AT_HD;
        let tiny = [0x0000u16, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0040, 0x0013];
        for (o, all) in [(rig.b.k, 2usize), (rig.b.v, 3)] {
            let x = rig.a.sl::<u16>(o, kvn);
            for (i, w) in tiny.iter().cycle().take(4000).enumerate() { x[(i * 7919 + 13) % kvn] = *w; }
            for (i, y) in x[all * MAX * AT_HD..(all + 1) * MAX * AT_HD].iter_mut().enumerate() { *y = tiny[(i * 31 + i / 7) % tiny.len()]; }
        }
        let mut bad = vec![];
        let mut checked = 0;
        let mut poss = vec![0usize, 1, 2, 5, 37, 250, 252, 253, 254, 255, 256, 257, 511, 1000, 1023, 4093, 8183, 8191, 8192, 8193,
                            8447, 8448, 12287, 12345, 16380, 16383, 16384, 20000, 24575, 24576, 32511, 32512, 32760, 32764, 32767];
        poss.extend((0..16).map(|i| 9000 + 257 * i + i * i)); // assorted fills of the last chunk
        for &pos in &poss {
            for t in 1..=8 {
                if pos + t <= MAX { bad.extend(rig.check_splitk(pos, t, false).unwrap()); checked += 1; }
            }
        }
        println!("verify_attn_splitk: {} (pos, K) passes checked x 2 chunk counts, {} failures", checked, bad.len());
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// q8 KV: `attn_split_k{K}_q8` == `attn_partial_k{K}_q8`, bit for bit, K = 1..8, up to 32768 keys, at the same
    /// positions as verify_attn_splitk_bit_identical (zero / subnormal scales and signed-zero mins in the rig's KV).
    #[test]
    fn verify_attn_splitk_q8_bit_identical() {
        const MAX: usize = 32768;
        let rig = attn_rig(MAX, 17).expect("rig");
        let mut bad = vec![];
        let mut checked = 0;
        let mut poss = vec![0usize, 1, 2, 5, 37, 250, 252, 253, 254, 255, 256, 257, 511, 1000, 1023, 4093, 8183, 8191, 8192, 8193,
                            8447, 8448, 12287, 12345, 16380, 16383, 16384, 20000, 24575, 24576, 32511, 32512, 32760, 32764, 32767];
        poss.extend((0..16).map(|i| 9000 + 257 * i + i * i));
        for &pos in &poss {
            for t in 1..=8 {
                if pos + t <= MAX { bad.extend(rig.check_splitk(pos, t, true).unwrap()); checked += 1; }
            }
        }
        println!("verify_attn_splitk_q8: {} (pos, K) passes checked x 2 chunk counts, {} failures", checked, bad.len());
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
