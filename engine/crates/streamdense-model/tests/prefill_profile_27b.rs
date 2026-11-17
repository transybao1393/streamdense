//! Prompt-read (fast prefill) profile of the REAL 27B (2026-11-17, measurement only; `prefill_profile.rs`).
//! One engine with the gateway's defaults (8-bit KV, MTP + spec auto, gemv mma, fast prefill, draft vocab, activity +
//! read-mask hooks), then:
//! 1. check: the timed prompt read (`pf_prefill_timed`, a copy of `prefill_mtp`'s loop that records per-chunk times)
//!    gives the same first token and rec top-1 / top-2 bits as the engine's own `prefill_mtp` on the first
//!    SD_PF_CHECK tokens (default 1024; 0 = skip);
//! 2. the timed prompt read of the first SD_PF_N tokens (default: the whole file): tok/s per context window
//!    (from the chunks' GPU spans), cumulative tok/s at 2K / 4K / 8K / 16K / 32K, GPU idle between chunks, CPU
//!    staging / encode ms, wall, peak footprint;
//! 3. component profile (`pf_profile_at`) of one chunk of prefill_t rows at each SD_PF_POS position (default
//!    0,2048,8192): GPU ms per class (one command buffer per class via `Enc::with_filter`), share of the whole
//!    chunk, dispatches, TFLOPS for the GEMM / attention classes. Runs last (it leaves the DeltaNet state advanced).
//! Env: SD_PF_PROMPT (u32 ids file, required), SD_PF_MODEL, SD_PF_N, SD_PF_T (prefill_t, default 128; 9..=256),
//! SD_PF_MAX_CTX (default N + 256 rounded up to 256), SD_PF_CAP_GB (engine budget cap, default 17.5), SD_PF_ITERS
//! (profile reps, min taken, default 2), SD_PF_MTP (1 default; 0 = MTP off, `prefill`), SD_PF_OUT (dir for the CSVs).
//! Run (under sd-watchdog, built with CARGO_TARGET_DIR=target-dev cargo test --release -p streamdense-model
//! --test prefill_profile_27b --no-run): prefill_profile_27b-<hash> --ignored --nocapture --test-threads 1 prefill_profile_27b
//!
//! `gpu_peak` (no model): sustained simdgroup-MMA rate of this GPU for fp32 (the engine's own `vf_peak_mma`), fp16
//! and bf16 8x8 fragments, and the scalar fp32 FMA rate (`vf_peak_fma`): register-only loops, the hardware ceiling of
//! any GEMM. Run: prefill_profile_27b-<hash> --ignored --nocapture gpu_peak
use std::io::Write;
use std::time::Instant;

use streamdense_model::{KvFormat, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

#[test]
#[ignore]
fn prefill_profile_27b() {
    let d = std::env::var("SD_PF_MODEL").unwrap_or_else(|_| format!("{}/models/qwen38_27b_mlx4", ROOT));
    let Ok(pf) = std::env::var("SD_PF_PROMPT") else { eprintln!("SKIP: set SD_PF_PROMPT"); return };
    let all: Vec<u32> = std::fs::read_to_string(&pf).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let n = env_usize("SD_PF_N", all.len()).min(all.len());
    let ids = &all[..n];
    let t = env_usize("SD_PF_T", 128);
    let max_ctx = env_usize("SD_PF_MAX_CTX", (n + 256).div_ceil(256) * 256);
    let cap = (std::env::var("SD_PF_CAP_GB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(17.5) * 1e9) as u64;
    let iters = env_usize("SD_PF_ITERS", 2);
    let check = env_usize("SD_PF_CHECK", 1024).min(n);
    let mtp = std::env::var("SD_PF_MTP").map_or(true, |s| s != "0");
    let out = std::env::var("SD_PF_OUT").ok().map(std::path::PathBuf::from);
    if let Some(o) = &out { std::fs::create_dir_all(o).unwrap(); }
    let pos: Vec<usize> = std::env::var("SD_PF_POS").unwrap_or_else(|_| "0,2048,8192".into()).split(',')
        .filter_map(|s| s.trim().parse().ok()).filter(|&p| p + t + 1 <= max_ctx).collect();
    let vk = format!("{}/draft_vocab.keep", d);
    let dt = std::env::temp_dir().join("sd_pf_draft_table");
    let opt = QwenOptions {
        cap, max_ctx, kv: KvFormat::Q8, prefill_t: t, prefill_exact: false, mtp, spec_depth: if mtp { 7 } else { 1 }, spec_auto: mtp,
        activity: true, activity_batch: true, read_mask: true,
        draft_vocab: (mtp && std::path::Path::new(&vk).exists()).then(|| vk.clone().into()),
        draft_table: mtp.then_some(dt),
        ..Default::default()
    };
    let t_open = Instant::now();
    let e = QwenEngine::open(&d, opt).unwrap();
    println!("model {} | prompt {} ({} of {} tokens) | prefill_t {} | max_ctx {} | kv {:?} | mtp {} | open {:.1} s | footprint after open {:.2} GB",
             d, pf, n, all.len(), t, max_ctx, e.opt.kv, mtp, t_open.elapsed().as_secs_f64(), e.footprint() as f64 / 1e9);
    let prefill_ref = |x: &[u32]| -> u32 { if mtp { e.prefill_mtp(x).unwrap().0 } else { e.prefill(x).unwrap() } };

    // 1. timed prompt read == the engine's own
    if check > 0 {
        e.reset();
        let tr = Instant::now();
        let t_ref = prefill_ref(&ids[..check]);
        let s_ref = tr.elapsed().as_secs_f64();
        let r_ref = e.rec_token(check - 1);
        e.reset();
        let tp = Instant::now();
        let (t_pf, _) = e.pf_prefill_timed(&ids[..check]).unwrap();
        let s_pf = tp.elapsed().as_secs_f64();
        let r_pf = e.rec_token(check - 1);
        let same = t_ref == t_pf && r_ref.1.to_bits() == r_pf.1.to_bits() && r_ref.2.to_bits() == r_pf.2.to_bits();
        println!("CHECK first {} tokens: engine {} ({:.1} s, {:.1} tok/s) vs timed {} ({:.1} s, {:.1} tok/s); rec top1/top2 {:?} vs {:?}: {}",
                 check, t_ref, s_ref, check as f64 / s_ref, t_pf, s_pf, check as f64 / s_pf, (r_ref.1, r_ref.2), (r_pf.1, r_pf.2),
                 if same { "EQUAL" } else { "DIFFERENT" });
        assert!(same, "timed prompt read differs from the engine's");
    }

    // 2. the timed prompt read
    e.reset();
    let tw = Instant::now();
    let (tok, ch) = e.pf_prefill_timed(ids).unwrap();
    let wall = tw.elapsed().as_secs_f64();
    let g0 = ch[0].gpu.0;
    let g1 = ch.iter().map(|c| c.gpu.1).fold(0.0, f64::max);
    // GPU busy = union of the chunk spans (a chunk's span starts at its GPU-side wait for the previous chunk)
    let mut sp: Vec<(f64, f64)> = ch.iter().map(|c| c.gpu).collect();
    sp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut busy, mut cur) = (0.0, (sp[0].0, sp[0].0));
    for s in &sp { if s.0 > cur.1 { busy += cur.1 - cur.0; cur = *s; } else { cur.1 = cur.1.max(s.1); } }
    busy += cur.1 - cur.0;
    let stage: f64 = ch.iter().map(|c| c.stage_ms).sum();
    let enc: f64 = ch.iter().map(|c| c.enc_ms).sum();
    let fast_rows: usize = ch.iter().filter(|c| c.k > 8).map(|c| c.k).sum();
    println!("PREFILL {} tokens: first token {} | wall {:.1} s = {:.2} tok/s | GPU first start..last end {:.1} s, busy {:.1} s, idle {:.3} s | host before first GPU start + after last end {:.3} s",
             n, tok, wall, n as f64 / wall, g1 - g0, busy, (g1 - g0) - busy, wall - (g1 - g0));
    println!("  chunks {} ({} rows in fast chunks, tail {} rows) | CPU staging {:.0} ms total ({:.2} ms/chunk) | CPU encode {:.0} ms total ({:.2} ms/chunk) | peak footprint {:.2} GB",
             ch.len(), fast_rows, n - fast_rows, stage, stage / ch.len() as f64, enc, enc / ch.len() as f64, e.peak() as f64 / 1e9);
    // per window and cumulative tok/s (GPU spans; a window ends at its last chunk's GPU end)
    let marks = [2048usize, 4096, 8192, 16384, 32768];
    let end_at = |m: usize| -> Option<f64> { ch.iter().filter(|c| c.i + c.k <= m).map(|c| c.gpu.1).fold(None, |a, x| Some(a.map_or(x, |y: f64| y.max(x)))) };
    let rows_at = |m: usize| -> usize { ch.iter().filter(|c| c.i + c.k <= m).map(|c| c.k).sum() };
    let mut prev = (0usize, g0);
    println!("  | up to token | cumulative tok/s | window | window tok/s | window ms/chunk-row |");
    for &m in marks.iter().chain(std::iter::once(&n)) {
        if m > n || (m == n && marks.contains(&n)) { if m != n { continue; } }
        let (Some(te), r) = (end_at(m), rows_at(m)) else { continue };
        if r <= prev.0 { continue; }
        let wr = r - prev.0;
        println!("  | {} | {:.2} | {}..{} | {:.2} | {:.3} |", r, r as f64 / (te - g0), prev.0, r, wr as f64 / (te - prev.1), (te - prev.1) * 1e3 / wr as f64);
        prev = (r, te);
    }
    if let Some(o) = &out {
        let mut f = std::fs::File::create(o.join(format!("chunks_t{}_n{}.csv", t, n))).unwrap();
        writeln!(f, "i,k,stage_ms,enc_ms,gpu_start_s,gpu_end_s,gpu_ms").unwrap();
        for c in &ch { writeln!(f, "{},{},{:.3},{:.3},{:.6},{:.6},{:.3}", c.i, c.k, c.stage_ms, c.enc_ms, c.gpu.0 - g0, c.gpu.1 - g0, (c.gpu.1 - c.gpu.0) * 1e3).unwrap(); }
    }

    // 3. component profile per position
    let mut csv = String::from("pos,k,class,gpu_ms,share,dispatches,flop,tflops\n");
    for &p in &pos {
        let pr = e.pf_profile_at(&all[..t + 1], t, p, iters).unwrap();
        let sum: f64 = pr.classes.iter().map(|c| c.1).sum();
        println!("PROFILE pos {} chunk {}: whole chunk {:.1} ms GPU ({:.1} ms wall) = {:.1} tok/s; sum of classes {:.1} ms ({:+.1}%)",
                 p, t, pr.full_ms, pr.full_wall_ms, t as f64 / pr.full_ms * 1e3, sum, (sum / pr.full_ms - 1.0) * 100.0);
        println!("  | class | GPU ms | share | dispatches | GFLOP | TFLOPS |");
        for (cl, ms, nd, fl) in &pr.classes {
            let tf = if *fl > 0.0 { format!("{:.2}", fl / ms / 1e9) } else { "-".into() };
            println!("  | {} | {:.2} | {:.1}% | {} | {:.1} | {} |", cl, ms, ms / pr.full_ms * 100.0, nd, fl / 1e9, tf);
            csv += &format!("{},{},{},{:.3},{:.4},{},{:.0},{}\n", p, t, cl, ms, ms / pr.full_ms, nd, fl, if *fl > 0.0 { fl / ms / 1e9 } else { 0.0 });
        }
        csv += &format!("{},{},WHOLE,{:.3},1,0,0,0\n", p, t, pr.full_ms);
    }
    if let Some(o) = &out { std::fs::write(o.join(format!("profile_t{}_n{}.csv", t, n)), csv).unwrap(); }
    println!("peak footprint {:.2} GB", e.peak() as f64 / 1e9);
}

#[test]
#[ignore]
fn gpu_peak() {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::*;
    use streamdense_metal::{commit, gpu_ms, wait, Enc, Gpu, QWEN_SHADERS};
    let mut g = Gpu::from_shader_files(QWEN_SHADERS).unwrap();
    for p in ["vf_peak_mma", "vf_peak_fma"] { g.pipeline(p).unwrap(); }
    println!("GPU: {}", g.name());
    let buf = g.buffer(1 << 20).unwrap();
    let (n, groups, tpg) = (4096u32, 20 * 64, 256usize);
    let lanes = (groups * tpg) as f64;
    for (name, fpl) in [("vf_peak_mma (fp32 8x8 MMA)", 256.0), ("vf_peak_fma (scalar fp32 FMA)", 32.0)] {
        let k = name.split(' ').next().unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let cb = g.command_buffer();
            let e = Enc::new(&g, &cb, &buf);
            e.pso(k).b(0, 0).u(1, n).groups((groups, 1), tpg);
            e.end(); commit(&cb); wait(&cb).unwrap();
            best = best.min(gpu_ms(&cb));
        }
        println!("{}: {:.3} ms, {:.2} TFLOPS", name, best, fpl * n as f64 * lanes / best / 1e9);
    }
    // fp16 / bf16 fragments (same loop shape as vf_peak_mma: 8 independent accumulators)
    let src = r#"
#include <metal_stdlib>
using namespace metal;
#define PK(NAME, M, NA) kernel void NAME(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]], \
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    M a, b, c[NA]; \
    a = M(1); b = M(1); \
    for (uint j = 0; j < NA; j++) c[j] = M(0); \
    for (uint i = 0; i < n; i++) _Pragma("clang loop unroll(full)") for (uint j = 0; j < NA; j++) simdgroup_multiply_accumulate(c[j], a, b, c[j]); \
    float s = 0; for (uint j = 0; j < NA; j++) s += float(c[j].thread_elements()[0]); \
    if (s == 12345.678f) out[tg * 32 + lane] = s; }
PK(pk_f16, simdgroup_half8x8, 8)
PK(pk_f32, simdgroup_float8x8, 8)
PK(pk_f16_16, simdgroup_half8x8, 16)
PK(pk_f32_16, simdgroup_float8x8, 16)
PK(pk_f32_4, simdgroup_float8x8, 4)
#if __METAL_VERSION__ >= 310
PK(pk_bf16, simdgroup_bfloat8x8, 8)
#endif
"#;
    let lib = g.device.newLibraryWithSource_options_error(&NSString::from_str(src), None).expect("peak shader");
    for (k, label, na) in [("pk_f32_4", "fp32 8x8 MMA, 4 acc", 4.0), ("pk_f32", "fp32 8x8 MMA, 8 acc (inline)", 8.0), ("pk_f32_16", "fp32 8x8 MMA, 16 acc", 16.0),
                           ("pk_f16", "fp16 8x8 MMA, fp16 acc, 8 acc", 8.0), ("pk_f16_16", "fp16 8x8 MMA, fp16 acc, 16 acc", 16.0), ("pk_bf16", "bf16 8x8 MMA, bf16 acc, 8 acc", 8.0)] {
        let Some(f) = lib.newFunctionWithName(&NSString::from_str(k)) else { println!("{}: not available", label); continue };
        let pso = g.device.newComputePipelineStateWithFunction_error(&f).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let cb = g.command_buffer();
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
                enc.setBytes_length_atIndex(std::ptr::NonNull::from(&n).cast(), 4, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: groups, height: 1, depth: 1 }, MTLSize { width: tpg, height: 1, depth: 1 });
            enc.endEncoding();
            commit(&cb); wait(&cb).unwrap();
            best = best.min(gpu_ms(&cb));
        }
        let _ = ProtocolObject::<dyn MTLComputePipelineState>::from_ref(&*pso);
        println!("{}: {:.3} ms, {:.2} TFLOPS", label, best, 32.0 * na * n as f64 * lanes / best / 1e9);
    }
}
