//! `decode_bench` (2026-11-03): per-token decode time budget at the real shapes of a Qwen3.5/3.8 config, with random
//! weights (reads `config.json` only, never the model's weights). It runs the ENGINE's own encoders (`encode_layer`,
//! `encode_head`, compact `encode_down`) for one K = 1 decode token over all layers, reusing `--copies` distinct
//! weight sets per layer kind (each layer is ~200 MB on the 27B, far beyond the SLC, so a reused copy streams from
//! DRAM like a distinct one). Reports:
//! - the whole token's GPU time (one command buffer, as `QwenEngine::encode_pass`), full FFN and compact FFN;
//! - per kernel class (one command buffer holding only that class's dispatches, via `Enc::with_filter`): GPU ms,
//!   dispatches, bytes, GB/s; the sum of classes vs the whole token (= dispatch overlap / bubbles);
//! - per-dispatch floor (chains of tiny dispatches), CPU encode time, and the wall per token of the synchronous
//!   loop (encode, commit, wait: `fsel_step` / `feed_stream`) vs an encode-ahead loop.
//!
//! RAM: copies x (DeltaNet + attention layer + full FFN + compact FFN) + lm_head + DeltaNet state of all layers + KV
//! of `copies` attention layers + scratch; 27B, 2 copies, ctx 4096: ~1.7 GB.
//! Run: `cargo run --release -p streamdense-model --example decode_bench -- --model DIR [--pos 1024] [--max-ctx 4096]
//!       [--copies 2] [--iters 15] [--kc 10728] [--fused]`
use std::time::Instant;

use streamdense_core::arena::{Arena, Region};
use streamdense_metal::{commit, contents, gpu_ms, wait, Buffer, CmdBuf, Enc, Gpu, SdCtrl, QWEN_SHADERS};

use crate::ffn_compact::CDown;
use crate::qwen_config::{LayerKind, QwenConfig};
use crate::qwen_engine::ATTN_CHUNK;
use crate::qwen_layers::{encode_head, encode_layer, encode_rope_invf, qwen_pipelines, scratch_sizes, DnState, GemvImpl, KvRef, PassIo};
use crate::weights::{AttnW, DnW, EmbedSrc, FfnW, LayerW, Mixer, Q4b, Weights};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
}
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }

/// Fill `n` u32 words at `p` with xorshift noise on all cores (weights; values do not change GPU time).
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

/// Options of the synthetic decode.
#[derive(Clone, Debug)]
pub struct SynthOpt {
    pub copies: usize,
    pub max_ctx: usize,
    /// ctrl.pos of the timed token (attention reads pos + 1 keys)
    pub pos: usize,
    /// compact FFN neurons per layer (multiple of 8; 0 = no compact set)
    pub kc: usize,
    pub gemv: GemvImpl,
    /// widest verify pass to size scratch for (8 = the engine; > 8 = bench-only wide verify)
    pub kmax: usize,
    /// sequences for the batched verify bench (`--batch`): 1 = today; each extra one gets its own DeltaNet state
    /// (2 copies), KV (one per attention copy) and ctrl
    pub seqs: usize,
}

/// One synthetic engine: arena, weights (full and compact FFN layer lists), state, KV, pass scratch.
pub struct Synth {
    pub gpu: Gpu,
    pub buf: Buffer,
    pub arena: Arena,
    pub cfg: QwenConfig,
    pub opt: SynthOpt,
    pub full: Vec<LayerW>,
    pub cpt: Vec<LayerW>,
    pub w: Weights,
    pub dn: DnState,
    pub kv: Vec<Option<KvRef>>,
    pub io: PassIo,
    pub stage: usize,
    /// deep-spec DeltaNet replay rows (verify passes, `dn_save`), depth 7
    pub replay: usize,
    /// fused-verify prep scratch (xt, st), as `QwenEngine` with `verify_fused`
    pub vf: (usize, usize),
    /// verify-mixers attention scratch, as `QwenEngine` with `verify_mixers`
    pub va: usize,
    /// batched verify bench: every sequence (index 0 = dn / kv / io.ctrl above)
    pub bseqs: Vec<crate::qwen_layers::BenchSeq>,
}

const A16: usize = 256;
fn al(x: usize) -> usize { x.div_ceil(A16) * A16 + A16 }
fn q4_len(r: usize, c: usize) -> usize { al(r * c / 2) + 2 * al(r * c / 64 * 2) + 3 * 16384 }

impl Synth {
    pub fn new(cfg: &QwenConfig, opt: SynthOpt) -> Result<Synth, String> {
        cfg.validate()?;
        let (h, n) = (cfg.hidden, cfg.inter);
        let c = opt.copies.max(1);
        let kc = opt.kc;
        if kc % 8 != 0 || kc > n { return Err(format!("kc {} must be a multiple of 8 <= inter {}", kc, n)); }
        let g = n / 64;
        let nch_max = opt.max_ctx.div_ceil(ATTN_CHUNK);
        let tmax = opt.kmax.max(8);
        let dn_l = q4_len(cfg.dn_in_rows(), h) + q4_len(h, cfg.dn_v_dim()) + 4 * al(cfg.conv_dim() * cfg.conv_k * 2);
        let at_l = q4_len(cfg.attn_in_rows(), h) + q4_len(h, cfg.q_dim()) + 2 * al(cfg.head_dim * 2);
        let ffn_l = 2 * q4_len(n, h) + q4_len(h, n);
        let cpt_l = if kc > 0 { 2 * q4_len(kc, h) + al(h * kc / 2) + 2 * al(h * g * 2) + al(kc * 2) + al((g + 1) * 4) + al(kc / 8 * 4) + 6 * 16384 } else { 0 };
        let kv_l = 2 * al(cfg.kv_heads * opt.max_ctx * cfg.head_dim * 2);
        let scratch: usize = scratch_sizes(cfg, tmax, nch_max).iter().map(|x| al(x.1)).sum::<usize>()
            + al((opt.max_ctx + tmax) * 16) + al(crate::ffn_compact::scratch_bytes(n, tmax)) + 4 * al(crate::embed::row_bytes(h) * tmax);
        let (vxt, vst) = crate::qwen_layers::vf_prep_bytes(tmax, crate::qwen_layers::vf_max_cols(cfg));
        let rpl = crate::spec_deep::replay_bytes(cfg, tmax - 1);
        let vab = crate::qwen_layers::va_scratch_bytes(cfg, 8);
        // two DeltaNet state copies: verify passes write the row-0 checkpoint (ck), as the deep spec loop
        let xs = opt.seqs.max(1) - 1;
        let total = xs * (al(DnState::s_bytes(cfg, 2)) + al(DnState::conv_bytes(cfg, 2)) + c * kv_l + al(std::mem::size_of::<SdCtrl>()))
            + c * (dn_l + at_l + ffn_l + cpt_l + kv_l + 2 * al(h * 2)) + q4_len(cfg.vocab, h) + al(h * 2)
            + al(DnState::s_bytes(cfg, 2)) + al(DnState::conv_bytes(cfg, 2)) + scratch + al(vxt) + al(vst) + al(rpl) + al(vab) + al(tmax * cfg.vocab * 4) + (64 << 20);
        let mut shaders: Vec<&str> = QWEN_SHADERS.to_vec();
        shaders.push("ffn_compact");
        shaders.push("bench_bw");
        let mut gpu = Gpu::from_shader_files(&shaders)?;
        for p in qwen_pipelines() { gpu.pipeline(p)?; }
        for p in crate::ffn_compact::pipelines() { gpu.pipeline(p)?; }
        for p in ["embed_q4", "advance1", "bw_read", "dn_save_rows"] { gpu.pipeline(p)?; }
        for p in crate::qwen_layers::vf_engine_pipelines() { gpu.pipeline(&p)?; }
        for p in crate::qwen_layers::vm_pipelines() { gpu.pipeline(&p)?; }
        for v in ["_t4"] { for m in [0, 2, 3] { gpu.pipeline(&format!("qm_gemvtd_k1{}_s8_m{}", v, m))?; } gpu.pipeline(&format!("qm_swiglutd_k1{}_s8", v))?; }
        if tmax > 8 { for t in (16..=tmax).step_by(8) { for m in [0, 2, 3] { gpu.pipeline(&crate::qwen_layers::vf_gemv_kernel(crate::qwen_layers::VF_NB_GEMV, t, m))?; } gpu.pipeline(&crate::qwen_layers::vf_swiglu_kernel(crate::qwen_layers::VF_NB_SWIGLU, t))?; } }
        let buf = gpu.buffer(total)?;
        let mut arena = Arena::new(contents(&buf), total);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let seed = std::cell::Cell::new(1u64);
        let base = contents(&buf);
        // ---- helpers
        let q4 = |a: &mut Arena, rng: &mut Rng, r: usize, cc: usize| -> Result<Q4b, String> {
            let w = a.alloc("w", r * cc / 2)?;
            let s = a.alloc("w", r * cc / 64 * 2)?;
            let b = a.alloc("w", r * cc / 64 * 2)?;
            seed.set(seed.get() + 1);
            fill_u32(a.ptr(w), r * cc / 8, seed.get());
            let (ss, bb) = (a.slice_mut::<u16>(s), a.slice_mut::<u16>(b));
            for i in 0..r * cc / 64 {
                let sc = 0.002 + 0.018 * rng.f();
                ss[i] = to_bf(sc);
                bb[i] = to_bf(-8.0 * sc + 0.01 * (rng.f() - 0.5));
            }
            Ok(Q4b { rows: r, cols: cc, w, s, b })
        };
        let bf_fill = |a: &mut Arena, rng: &mut Rng, n: usize, v: f32, spread: f32| -> Result<Region, String> {
            let r = a.alloc("w", n * 2)?;
            for x in a.slice_mut::<u16>(r) { *x = to_bf(v + spread * (rng.f() - 0.5)); }
            Ok(r)
        };
        // ---- weight copies
        let mut dns = vec![];
        let mut ats = vec![];
        let mut ffs = vec![];
        let mut cps = vec![];
        let mut norms = vec![];
        let mut keep: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() { let j = (rng.next() % (i as u64 + 1)) as usize; keep.swap(i, j); }
        keep.truncate(kc);
        keep.sort_unstable();
        for _ in 0..c {
            let in_proj = q4(&mut arena, &mut rng, cfg.dn_in_rows(), h)?;
            let conv_w = bf_fill(&mut arena, &mut rng, cfg.conv_dim() * cfg.conv_k, 0.0, 0.5)?;
            let a_log = bf_fill(&mut arena, &mut rng, cfg.dn_vh, 0.5, 0.5)?;
            let dt_bias = bf_fill(&mut arena, &mut rng, cfg.dn_vh, 0.0, 1.0)?;
            let norm_w = bf_fill(&mut arena, &mut rng, cfg.dn_dv, 1.0, 0.1)?;
            let out_proj = q4(&mut arena, &mut rng, h, cfg.dn_v_dim())?;
            dns.push(DnW { in_proj, conv_w, a_log, dt_bias, norm_w, out_proj });
            let in_proj = q4(&mut arena, &mut rng, cfg.attn_in_rows(), h)?;
            let q_norm = bf_fill(&mut arena, &mut rng, cfg.head_dim, 1.0, 0.1)?;
            let k_norm = bf_fill(&mut arena, &mut rng, cfg.head_dim, 1.0, 0.1)?;
            let o_proj = q4(&mut arena, &mut rng, h, cfg.q_dim())?;
            ats.push(AttnW { in_proj, q_norm, k_norm, o_proj });
            let gate = q4(&mut arena, &mut rng, n, h)?;
            let up = q4(&mut arena, &mut rng, n, h)?;
            let down = q4(&mut arena, &mut rng, h, n)?;
            ffs.push(FfnW { gate, up, down, cd: None });
            if kc > 0 {
                let gate = q4(&mut arena, &mut rng, kc, h)?;
                let up = q4(&mut arena, &mut rng, kc, h)?;
                let dw = arena.alloc("w", h * kc / 2)?;
                seed.set(seed.get() + 1);
                fill_u32(arena.ptr(dw), h * kc / 8, seed.get());
                let ds = bf_fill(&mut arena, &mut rng, h * g, 0.01, 0.018)?;
                let db = bf_fill(&mut arena, &mut rng, h * g, -0.08, 0.01)?;
                let (gid, gstart, wmeta) = (arena.alloc("w", kc * 2)?, arena.alloc("w", (g + 1) * 4)?, arena.alloc("w", kc / 8 * 4)?);
                let cd = CDown { kc, groups: g, gid, gstart, wmeta, mma: false };
                crate::ffn_compact::fill_meta(&cd, &keep, n, base);
                cps.push(FfnW { gate, up, down: Q4b { rows: h, cols: kc, w: dw, s: ds, b: db }, cd: Some(cd) });
            }
            norms.push((bf_fill(&mut arena, &mut rng, h, 1.0, 0.1)?, bf_fill(&mut arena, &mut rng, h, 1.0, 0.1)?));
        }
        let lm_head = q4(&mut arena, &mut rng, cfg.vocab, h)?;
        let final_norm = bf_fill(&mut arena, &mut rng, h, 1.0, 0.1)?;
        // ---- state, KV (one KV per attention copy)
        let dn_s = arena.alloc("st", DnState::s_bytes(cfg, 2))?;
        let dn_c = arena.alloc("st", DnState::conv_bytes(cfg, 2))?;
        let dn = DnState { s: dn_s.off, conv: dn_c.off, copies: 2 };
        let (vf_xt, vf_st, replay) = (arena.alloc("vf", vxt)?.off, arena.alloc("vf", vst)?.off, arena.alloc("rpl", rpl.max(16))?.off);
        let va = arena.alloc("va", vab)?.off;
        let mut kvs = vec![];
        for _ in 0..c {
            let nb = cfg.kv_heads * opt.max_ctx * cfg.head_dim * 2;
            let (k, v) = (arena.alloc("kv", nb)?, arena.alloc("kv", nb)?);
            for r in [k, v] { for x in arena.slice_mut::<u16>(r) { *x = to_bf(rng.f() - 0.5); } }
            kvs.push(KvRef { k: k.off, v: v.off, ksb: k.off, vsb: v.off, q8: false, max_ctx: opt.max_ctx });
        }
        // ---- scratch
        let mut sr = std::collections::HashMap::new();
        for (nm, b) in scratch_sizes(cfg, tmax, nch_max) { sr.insert(nm, arena.alloc(nm, b)?.off); }
        let ctrl = arena.alloc("ctrl", std::mem::size_of::<SdCtrl>())?;
        let rec = arena.alloc("rec", (opt.max_ctx + tmax) * 16)?;
        let cfs = arena.alloc("cfs", crate::ffn_compact::scratch_bytes(n, tmax))?;
        let stage = arena.alloc("stage", crate::embed::row_bytes(h) * tmax)?;
        fill_u32(arena.ptr(stage), stage.len / 4, 77);
        { let sl = arena.slice_mut::<u16>(stage); let w = h / 2 / 2; for x in &mut sl[w..w + 2 * (h / 64)] { *x = to_bf(0.01); } }
        // scratch_sizes caps logits at 8 rows (head blocks of 8); a wide verify head writes tmax rows
        if tmax > 8 { sr.insert("logits", arena.alloc("logits", tmax * cfg.vocab * 4)?.off); }
        let o = |k: &str| sr[k];
        let io = PassIo {
            k: 1, h: o("h"), x: o("x"), p: o("p"), cv: o("cv"), g: o("g"), beta: o("beta"), y: o("y"), yn: o("yn"), ao: o("ao"),
            f: o("f"), po: o("po"), pml: o("pml"), hf: o("hf"), ctrl: ctrl.off, nch_max: (opt.pos + 1).div_ceil(ATTN_CHUNK).min(nch_max),
            pos: opt.pos, logits: o("logits"), rec: rec.off, invf: o("invf"), dn_save: None, dn_hold: None, gemv: opt.gemv, act: None, rm: None,
            fsel: None, cfs: cfs.off, vf: None, fxcap: None, vm: None, ban: None,
        };
        let mut full = vec![];
        let mut cpt = vec![];
        let mut kv = vec![];
        let (mut nd, mut na) = (0usize, 0usize);
        for l in 0..cfg.layers {
            let (mixer, ci) = match cfg.kinds[l] {
                LayerKind::DeltaNet => { nd += 1; (Mixer::Dn(dns[(nd - 1) % c]), (nd - 1) % c) }
                LayerKind::Attention => { na += 1; (Mixer::Attn(ats[(na - 1) % c]), (na - 1) % c) }
            };
            kv.push(if cfg.kinds[l] == LayerKind::Attention { Some(kvs[ci]) } else { None });
            let (in_norm, post_norm) = norms[l % c];
            full.push(LayerW { in_norm, post_norm, mixer, ffn: ffs[l % c] });
            if kc > 0 { cpt.push(LayerW { in_norm, post_norm, mixer, ffn: cps[l % c] }); }
        }
        let w = Weights { layers: vec![], final_norm, lm_head, mtp: None, embed: EmbedSrc::default() };
        let mut bseqs = vec![crate::qwen_layers::BenchSeq { dn, kv: kv.clone(), ctrl: ctrl.off }];
        for _ in 0..xs {
            let s0 = arena.alloc("st", DnState::s_bytes(cfg, 2))?;
            let c0 = arena.alloc("st", DnState::conv_bytes(cfg, 2))?;
            arena.slice_mut::<u8>(s0).fill(0);
            arena.slice_mut::<u8>(c0).fill(0);
            let mut kx = vec![];
            for _ in 0..c {
                let nb = cfg.kv_heads * opt.max_ctx * cfg.head_dim * 2;
                let (k, v) = (arena.alloc("kv", nb)?, arena.alloc("kv", nb)?);
                for r in [k, v] { for x in arena.slice_mut::<u16>(r) { *x = to_bf(rng.f() - 0.5); } }
                kx.push(KvRef { k: k.off, v: v.off, ksb: k.off, vsb: v.off, q8: false, max_ctx: opt.max_ctx });
            }
            let (mut na2, mut kvl) = (0usize, vec![]);
            for l in 0..cfg.layers {
                kvl.push(if cfg.kinds[l] == LayerKind::Attention { na2 += 1; Some(kx[(na2 - 1) % c]) } else { None });
            }
            let cx = arena.alloc("ctrl", std::mem::size_of::<SdCtrl>())?;
            bseqs.push(crate::qwen_layers::BenchSeq { dn: DnState { s: s0.off, conv: c0.off, copies: 2 }, kv: kvl, ctrl: cx.off });
        }
        let s = Synth { gpu, buf, arena, cfg: cfg.clone(), opt, full, cpt, w, dn, kv, io, stage: stage.off, replay, vf: (vf_xt, vf_st), va, bseqs };
        {
            let cb = s.gpu.command_buffer();
            let e = Enc::new(&s.gpu, &cb, &s.buf);
            encode_rope_invf(&e, &s.cfg, &s.io);
            e.end();
            commit(&cb);
            wait(&cb)?;
        }
        s.reset_h();
        Ok(s)
    }

    #[allow(clippy::mut_from_ref)] // shared CPU/GPU arena, as QwenEngine::ctrl
    fn ctrl(&self) -> &mut SdCtrl { unsafe { &mut *(contents(&self.buf).add(self.io.ctrl) as *mut SdCtrl) } }
    /// Position back to `opt.pos` (no command buffer in flight).
    pub fn set_pos(&self) {
        *self.ctrl() = SdCtrl { pos: self.opt.pos as u32, ..Default::default() };
        for sq in &self.bseqs[1..] { unsafe { *(contents(&self.buf).add(sq.ctrl) as *mut SdCtrl) = SdCtrl { pos: self.opt.pos as u32, ..Default::default() }; } }
    }
    /// Small finite hidden row (random weights blow it up over many passes; timing does not depend on it).
    pub fn reset_h(&self) {
        let p = unsafe { std::slice::from_raw_parts_mut(contents(&self.buf).add(self.io.h) as *mut u16, self.cfg.hidden * self.opt.kmax.max(8)) };
        let mut r = Rng(99);
        for x in p { *x = to_bf((r.f() - 0.5) * 0.1); }
    }

    /// Encode one K = 1 decode token exactly as `QwenEngine::encode_pass(cb, 1, slot, 1, false)` does.
    pub fn encode_token(&self, e: &Enc, compact: bool) -> Result<(), String> {
        let c = &self.cfg;
        let io = &self.io;
        e.pso("embed_q4").b(0, self.stage).b(1, io.h).u(2, 1);
        e.threads((c.hidden / 8, 1), 64);
        let layers = if compact { &self.cpt } else { &self.full };
        for (l, lw) in layers.iter().enumerate() {
            encode_layer(e, c, lw, Some(&self.dn), self.kv[l].as_ref(), io, false, None, l)?;
        }
        encode_head(e, c, &self.w, io, 1, None)?;
        e.pso("advance1").b(0, io.ctrl);
        e.threads((1, 1), 1);
        Ok(())
    }

    /// Encode one exact VERIFY pass of `k` rows as the deep spec loop does (`encode_verify_deep`: embed k rows, every
    /// layer with the row-0 checkpoint (k > 1) and the DeltaNet replay save (k > 2), head on all k rows; the MTP emb
    /// copy and `spec_conf` are left out, < 0.1 ms). `vf`: fused verify matmuls (`QwenOptions.verify_fused`), `va`:
    /// verify mixers (`QwenOptions.verify_mixers`).
    pub fn encode_verify(&self, e: &Enc, k: usize, vf: bool, va: bool) -> Result<(), String> {
        let c = &self.cfg;
        let io = PassIo { k, nch_max: (self.opt.pos + k).div_ceil(ATTN_CHUNK).min(self.opt.max_ctx.div_ceil(ATTN_CHUNK)),
                          dn_save: if k > 2 { Some(self.replay) } else { None }, vf: if vf { Some(self.vf) } else { None },
                          vm: if va { Some(self.va) } else { None }, ..self.io };
        e.pso("embed_q4").b(0, self.stage).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        for (l, lw) in self.full.iter().enumerate() {
            encode_layer(e, c, lw, Some(&self.dn), self.kv[l].as_ref(), &io, k > 1, None, l)?;
        }
        encode_head(e, c, &self.w, &io, k, None)
    }

    /// one fused verify pass carrying `b` sequences of `r` rows (B*R rows through embed / norms / matmuls / FFN /
    /// head; DeltaNet and attention per sequence on its own state, KV and ctrl.pos), as `encode_verify` otherwise
    /// (row-0 checkpoint per sequence when r > 1, DeltaNet replay save when r > 2, head on all rows).
    pub fn encode_verify_batch(&self, e: &Enc, b: usize, r: usize, vf: bool) -> Result<(), String> {
        let c = &self.cfg;
        let k = b * r;
        if b > self.bseqs.len() { return Err(format!("--batch {}x{}: only {} sequences built (--seqs)", b, r, self.bseqs.len())); }
        let io = PassIo { k, nch_max: (self.opt.pos + r).div_ceil(ATTN_CHUNK).min(self.opt.max_ctx.div_ceil(ATTN_CHUNK)),
                          dn_save: if r > 2 { Some(self.replay) } else { None }, vf: if vf { Some(self.vf) } else { None },
                          vm: None, ..self.io };
        e.pso("embed_q4").b(0, self.stage).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        for (l, lw) in self.full.iter().enumerate() {
            crate::qwen_layers::bench_encode_layer_batch(e, c, lw, &self.bseqs[..b], &io, r, r > 1, l)?;
        }
        encode_head(e, c, &self.w, &io, k, None)
    }

    /// GPU ms of `iters` command buffers of `f` (after 2 warm-up), with an optional dispatch filter.
    pub fn time_with(&self, f: &dyn Fn(&Enc) -> Result<(), String>, filt: Option<&dyn Fn(usize, &str) -> bool>, iters: usize) -> Result<Vec<f64>, String> {
        let mut v = vec![];
        for it in 0..iters + 2 {
            self.set_pos();
            let cb = self.gpu.command_buffer();
            let e = match filt { Some(fl) => Enc::with_filter(&self.gpu, &cb, &self.buf, fl), None => Enc::new(&self.gpu, &cb, &self.buf) };
            f(&e)?;
            e.end();
            commit(&cb);
            wait(&cb)?;
            if it >= 2 { v.push(gpu_ms(&cb)); }
            if it % 4 == 3 { self.reset_h(); }
        }
        Ok(v)
    }

    /// Kernel names of `f`'s dispatches in order (nothing dispatched).
    pub fn names_of(&self, f: &dyn Fn(&Enc) -> Result<(), String>) -> Result<Vec<String>, String> {
        let rec = std::cell::RefCell::new(Vec::new());
        let fl = |_: usize, n: &str| { rec.borrow_mut().push(n.to_string()); false };
        let cb = self.gpu.command_buffer();
        let e = Enc::with_filter(&self.gpu, &cb, &self.buf, &fl);
        f(&e)?;
        e.end();
        Ok(rec.into_inner())
    }

    /// GPU ms of `iters` single-token command buffers (after 2 warm-up), with an optional dispatch filter.
    pub fn time(&self, compact: bool, filt: Option<&dyn Fn(usize, &str) -> bool>, iters: usize) -> Result<Vec<f64>, String> {
        let mut v = vec![];
        for it in 0..iters + 2 {
            self.set_pos();
            let cb = self.gpu.command_buffer();
            let e = match filt { Some(f) => Enc::with_filter(&self.gpu, &cb, &self.buf, f), None => Enc::new(&self.gpu, &cb, &self.buf) };
            self.encode_token(&e, compact)?;
            e.end();
            commit(&cb);
            wait(&cb)?;
            if it >= 2 { v.push(gpu_ms(&cb)); }
            if it % 8 == 7 { self.reset_h(); }
        }
        Ok(v)
    }

    /// Kernel names of one token in dispatch order (nothing dispatched).
    pub fn names(&self, compact: bool) -> Result<Vec<String>, String> {
        let rec = std::cell::RefCell::new(Vec::new());
        let f = |_: usize, n: &str| { rec.borrow_mut().push(n.to_string()); false };
        let cb = self.gpu.command_buffer();
        let e = Enc::with_filter(&self.gpu, &cb, &self.buf, &f);
        self.encode_token(&e, compact)?;
        e.end();
        Ok(rec.into_inner())
    }

    /// Class label per dispatch (projections told apart by position in the layer) and its bytes read + written.
    pub fn labels(&self, names: &[String], compact: bool) -> Vec<(String, f64)> {
        let c = &self.cfg;
        let q = |r: usize, cc: usize| (r * cc / 2 + 2 * (r * cc / 64 * 2)) as f64;
        let (h, n) = (c.hidden, c.inter);
        let kc = self.opt.kc;
        let mut out = vec![];
        let mut norms = 0usize;
        let mut prev = String::new();
        for nm in names {
            let layer = (norms.saturating_sub(1)) / 2;
            let kind = c.kinds.get(layer).copied().unwrap_or(LayerKind::Attention);
            let lab: (String, f64) = if nm == "rmsnorm_b" {
                norms += 1;
                ("rmsnorm_b (in/post/final)".into(), (3 * h * 2) as f64)
            } else if nm.ends_with("_m0") {
                match kind {
                    LayerKind::DeltaNet => ("gemv dn.in_proj".into(), q(c.dn_in_rows(), h)),
                    LayerKind::Attention => ("gemv attn.in_proj".into(), q(c.attn_in_rows(), h)),
                }
            } else if nm.ends_with("_m2") {
                if prev == "rmsnorm_gated" { ("gemv dn.out_proj".into(), q(h, c.dn_v_dim())) }
                else if prev.starts_with("attn_") { ("gemv attn.o_proj".into(), q(h, c.q_dim())) }
                else { ("gemv ffn.down (full)".into(), q(h, n)) }
            } else if nm.ends_with("_m3") {
                ("gemv lm_head".into(), q(c.vocab, h))
            } else if nm.contains("swiglu") {
                (format!("swiglu ffn.gate+up ({})", if compact { "compact" } else { "full" }), 2.0 * q(if compact { kc } else { n }, h))
            } else if nm.starts_with("cd_") && nm != "cd_fsum" {
                ("compact down".into(), (h * kc / 2 + 2 * h * (n / 64) * 2) as f64)
            } else if nm.starts_with("dn_recur") {
                (nm.clone(), 2.0 * (c.dn_vh * c.dn_dv * c.dn_dk * 4) as f64)
            } else if nm == "attn_sdpa_vec_b16" || nm.starts_with("attn_partial") || nm.starts_with("attn_split") || nm.starts_with("va_sdpa_p1") {
                (nm.clone(), (2 * (self.opt.pos + 1) * c.kv_dim() * 2) as f64)
            } else if nm == "dn_conv" {
                (nm.clone(), (c.conv_dim() * (2 + 2 * 3 + 4 * 2 + 2)) as f64)
            } else {
                (nm.clone(), 0.0)
            };
            if nm != "vf_xprep" { prev = nm.clone(); }
            out.push(lab);
        }
        out
    }
}

fn med(v: &[f64]) -> f64 { let mut s = v.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] }

/// Chain of `n` tiny dispatches (`advance1`, 1 thread): GPU µs per dispatch.
fn dispatch_floor(s: &Synth, n: usize, iters: usize) -> Result<f64, String> {
    let mut v = vec![];
    for _ in 0..iters {
        let cb = s.gpu.command_buffer();
        let e = Enc::new(&s.gpu, &cb, &s.buf);
        for _ in 0..n { e.pso("advance1").b(0, s.io.ctrl); e.threads((1, 1), 1); }
        e.end();
        commit(&cb);
        wait(&cb)?;
        v.push(gpu_ms(&cb) * 1e3 / n as f64);
    }
    s.set_pos();
    Ok(med(&v))
}

/// Pure streaming read of the lm_head region (715 MB on the 27B): best GB/s over a few grid sizes.
fn read_ceiling(s: &Synth, iters: usize) -> Result<(f64, usize), String> {
    let r = s.w.lm_head.w;
    let n = r.len / 16;
    let mut best = (0.0, 0);
    for tg in [256usize, 512, 1024] {
        for mult in [8usize, 16, 32, 64] {
            let grid = tg * mult * 20;
            let mut v = vec![];
            for _ in 0..iters {
                let cb = s.gpu.command_buffer();
                let e = Enc::new(&s.gpu, &cb, &s.buf);
                e.pso("bw_read").b(0, r.off).b(1, s.io.logits).u(2, n as u32).u(3, 0x1234_5678);
                e.threads((grid, 1), tg);
                e.end();
                commit(&cb);
                wait(&cb)?;
                v.push(gpu_ms(&cb));
            }
            let gbs = r.len as f64 / med(&v) / 1e6;
            if gbs > best.0 { best = (gbs, grid); }
        }
    }
    Ok(best)
}

/// Loops of `n` tokens: (wall ms/token sync, GPU ms/token, CPU encode ms, wall ms/token encode-ahead).
fn loops(s: &Synth, compact: bool, n: usize) -> Result<(f64, f64, f64, f64), String> {
    let enc = |cb: &CmdBuf| -> Result<f64, String> {
        let t = Instant::now();
        let e = Enc::new(&s.gpu, cb, &s.buf);
        s.encode_token(&e, compact)?;
        e.end();
        Ok(t.elapsed().as_secs_f64() * 1e3)
    };
    // synchronous: encode, commit, wait (fsel_step / feed_stream / decode_sync, minus the embedding pread)
    s.set_pos();
    let (mut gpu, mut encs) = (0.0, vec![]);
    let t0 = Instant::now();
    for i in 0..n {
        if i % 8 == 7 { s.reset_h(); }
        s.ctrl().pos = s.opt.pos as u32;
        let cb = s.gpu.command_buffer();
        encs.push(enc(&cb)?);
        commit(&cb);
        wait(&cb)?;
        gpu += gpu_ms(&cb);
    }
    let sync = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    // encode-ahead: CB(i+1) encoded while CB(i) runs, committed right after CB(i) completes (the token is known then)
    s.set_pos();
    let t0 = Instant::now();
    let mut next = s.gpu.command_buffer();
    enc(&next)?;
    for _ in 0..n {
        let cb = next;
        s.ctrl().pos = s.opt.pos as u32;
        commit(&cb);
        next = s.gpu.command_buffer();
        enc(&next)?;
        wait(&cb)?;
    }
    let ahead = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    s.set_pos();
    Ok((sync, gpu / n as f64, med(&encs), ahead))
}

/// Bit-identity of `GemvImpl::MmaD` vs `Mma` at the real shapes: every projection / FFN / lm_head matrix, K = 1..4,
/// modes 0 and 3 (mode 2 = mode 0 + the same residual rounding), swiglu; random x. Returns the mismatching outputs.
pub fn check_mmad(s: &Synth) -> Result<usize, String> {
    use crate::qwen_layers::{gemv, swiglu_m};
    let io = &s.io;
    let (lw_dn, lw_at) = (s.full.iter().find(|l| matches!(l.mixer, Mixer::Dn(_))), s.full.iter().find(|l| matches!(l.mixer, Mixer::Attn(_))));
    let (Some(ld), Some(la)) = (lw_dn, lw_at) else { return Err("no layers".into()) };
    let (Mixer::Dn(d), Mixer::Attn(a)) = (ld.mixer, la.mixer) else { unreachable!() };
    let mats = [d.in_proj, d.out_proj, a.in_proj, a.o_proj, ld.ffn.down, s.w.lm_head];
    let base = contents(&s.buf);
    let mut bad = 0;
    let mut r = Rng(5);
    let xs = unsafe { std::slice::from_raw_parts_mut(base.add(io.f) as *mut u16, 4 * s.cfg.inter) };
    for x in xs.iter_mut() { *x = to_bf(r.f() - 0.5); }
    let run = |f: &dyn Fn(&Enc)| -> Result<(), String> {
        let cb = s.gpu.command_buffer();
        let e = Enc::new(&s.gpu, &cb, &s.buf);
        f(&e);
        e.end();
        commit(&cb);
        wait(&cb)
    };
    for k in 1..=4 {
        for m in &mats {
            for mode in [0u32, 3] {
                let ob = if mode == 3 { 4 } else { 2 } * k * m.rows;
                let mut outs = vec![];
                for imp in [GemvImpl::Mma, GemvImpl::MmaD] {
                    run(&|e| gemv(e, imp, m, k, mode, io.f, io.logits))?;
                    outs.push(unsafe { std::slice::from_raw_parts(base.add(io.logits), ob) }.to_vec());
                }
                let n = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a != b).count();
                if n > 0 { println!("check MmaD: {}x{} k {} mode {}: {} bytes differ", m.rows, m.cols, k, mode, n); }
                bad += n;
            }
        }
        let mut outs = vec![];
        for imp in [GemvImpl::Mma, GemvImpl::MmaD] {
            run(&|e| swiglu_m(e, imp, &ld.ffn.gate, &ld.ffn.up, k, io.f, io.logits, None))?;
            outs.push(unsafe { std::slice::from_raw_parts(base.add(io.logits), 2 * k * ld.ffn.gate.rows) }.to_vec());
        }
        let n = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a != b).count();
        if n > 0 { println!("check MmaD: swiglu k {}: {} bytes differ", k, n); }
        bad += n;
    }
    println!("check MmaD vs Mma (27B shapes, K = 1..4, 6 matrices x modes 0/3 + swiglu): {} bytes differ", bad);
    Ok(bad)
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: f64| get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let model = get("--model").ok_or("--model DIR (config.json only is read)")?;
    let cfg = QwenConfig::from_model_dir(&model)?;
    let kc = ((num("--kc", (cfg.inter as f64 * 0.6163).round()) as usize) / 8) * 8;
    let opt = SynthOpt { copies: num("--copies", 2.0) as usize, max_ctx: num("--max-ctx", 4096.0) as usize, pos: num("--pos", 1024.0) as usize,
                         kc, kmax: 8, seqs: 1, gemv: if args.iter().any(|a| a == "--mmad") { GemvImpl::MmaD } else { GemvImpl::parse(&get("--gemv").unwrap_or("mma".into()))? } };
    // wide verify: --verify K > 8 sizes the scratch for the widest K and lifts the encoders' K cap (bench only)
    let wide = get("--verify").map(|v| v.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).max().unwrap_or(8)).unwrap_or(8);
    if wide > crate::qwen_layers::VF_K_WIDE_MAX { return Err(format!("--verify K {} > {} (fused verify tiles stop at 4 x 8)", wide, crate::qwen_layers::VF_K_WIDE_MAX)); }
    // --batch BxR,BxR,...: batched verify passes (bench only); sequences built = largest B
    let batch: Vec<(usize, usize)> = match get("--batch") {
        Some(v) => v.split(',').map(|x| { let mut it = x.trim().split('x').map(|y| y.parse::<usize>());
            match (it.next(), it.next()) { (Some(Ok(b)), Some(Ok(r))) if b >= 1 && r >= 1 && b * r <= 8 => Ok((b, r)), _ => Err(format!("--batch {} (BxR, B*R <= 8)", x)) } })
            .collect::<Result<_, _>>()?,
        None => vec![],
    };
    let opt = SynthOpt { kmax: wide.max(8), seqs: batch.iter().map(|x| x.0).max().unwrap_or(1), ..opt };
    if wide > 8 { crate::qwen_layers::BENCH_VERIFY_K_MAX.store(wide, std::sync::atomic::Ordering::Relaxed); }
    let iters = num("--iters", 15.0) as usize;
    let t = Instant::now();
    let s = Synth::new(&cfg, opt.clone())?;
    println!("decode_bench | hidden {} inter {} layers {} ({} DeltaNet) vocab {} | pos {} | copies {} | kc {} | arena {:.2} GB built in {:.1} s",
             cfg.hidden, cfg.inter, cfg.layers, cfg.n_deltanet(), cfg.vocab, opt.pos, opt.copies, kc, s.arena.size() as f64 / 1e9,
             t.elapsed().as_secs_f64());
    if args.iter().any(|a| a == "--check") { check_mmad(&s)?; }
    if !batch.is_empty() { return batch_whole(&s, &batch, args.iter().any(|a| a == "--no-vf"), iters, num("--rounds", 3.0) as usize); }
    if args.iter().any(|a| a == "--shootout") { let mut s = s; return shootout(&mut s, iters); }
    if args.iter().any(|a| a == "--nt") { return nt_sweep(&s, iters); }
    if let Some(v) = get("--verify") {
        let ks: Vec<usize> = v.split(',').map(|x| x.trim().parse::<usize>().map_err(|_| format!("--verify {}", v))).collect::<Result<_, _>>()?;
        // --modes off,vf,va: today's kernels / + fused verify matmuls / + verify mixers (va implies vf)
        let modes: Vec<(bool, bool)> = get("--modes").unwrap_or("off,vf,va".into()).split(',').map(|m| match m.trim() {
            "off" => Ok((false, false)), "vf" => Ok((true, false)), "va" => Ok((true, true)), "va-only" => Ok((false, true)),
            x => Err(format!("--modes: {} (off, vf, va, va-only)", x)) }).collect::<Result<_, _>>()?;
        if args.iter().any(|a| a == "--whole") { return verify_whole(&s, &ks, &modes, iters, num("--rounds", 3.0) as usize); }
        return verify_profile(&s, &ks, &modes, iters, args.iter().any(|a| a == "--classes"));
    }
    let floor = dispatch_floor(&s, 640, 7)?;
    println!("dispatch floor: {:.2} us per dependent 1-thread dispatch (chain of 640)", floor);
    let (bw, grid) = read_ceiling(&s, 5)?;
    println!("read ceiling: {:.0} GB/s (pure streaming read of {:.0} MB, grid {})", bw, s.w.lm_head.w.len as f64 / 1e6, grid);
    for compact in [false, true] {
        if compact && kc == 0 { continue; }
        let mode = if compact { "COMPACT" } else { "FULL" };
        let whole = med(&s.time(compact, None, iters)?);
        let names = s.names(compact)?;
        let labs = s.labels(&names, compact);
        let mut classes: Vec<String> = vec![];
        for (l, _) in &labs { if !classes.contains(l) { classes.push(l.clone()); } }
        println!("\n== {} token: {} dispatches, GPU {:.2} ms (median of {})", mode, names.len(), whole, iters);
        println!("{:<30} {:>5} {:>9} {:>9} {:>8} {:>8} {:>9}", "class", "n", "ms", "MB", "GB/s", "us/disp", "ms@240");
        let mut sum = 0.0;
        let mut tot_b = 0.0;
        for cl in &classes {
            let idx: Vec<usize> = labs.iter().enumerate().filter(|(_, x)| &x.0 == cl).map(|(i, _)| i).collect();
            let bytes: f64 = idx.iter().map(|&i| labs[i].1).sum();
            let f = |i: usize, _: &str| labs.get(i).is_some_and(|x| &x.0 == cl);
            let ms = med(&s.time(compact, Some(&f), iters.min(9))?);
            sum += ms;
            tot_b += bytes;
            let gbs = if bytes > 0.0 { format!("{:.0}", bytes / ms / 1e6) } else { "-".into() };
            println!("{:<30} {:>5} {:>9.3} {:>9.1} {:>8} {:>8.1} {:>9.3}", cl, idx.len(), ms, bytes / 1e6, gbs, ms * 1e3 / idx.len() as f64,
                     bytes / 240e6);
        }
        println!("{:<30} {:>5} {:>9.3} {:>9.1} {:>8.0} {:>8} {:>9.3}", "SUM of classes", names.len(), sum, tot_b / 1e6, tot_b / sum / 1e6, "", tot_b / 240e6);
        println!("whole token {:.3} ms ({:.0} GB/s over {:.2} GB); sum - whole = {:.3} ms", whole, tot_b / whole / 1e6, tot_b / 1e9, sum - whole);
        let (sync, gpu, enc, ahead) = loops(&s, compact, 40)?;
        println!("loop (40 tok): sync wall {:.2} ms/token (GPU {:.2}, CPU gap {:.2}; encode {:.2} ms) | encode-ahead wall {:.2} ms/token",
                 sync, gpu, sync - gpu, enc, ahead);
    }
    Ok(())
}

/// Per-shape kernel shootout among the bit-identical Mma-family choices for K = 1 (SK = 8): the scalar twin (static
/// 16 KB T), the twin on dynamic T (MmaD), and the simdgroup-matrix kernels qm_gemv / qm_swiglu NB = 1, 2. Each
/// shape: 32 dispatches alternating 2 weight copies, median of `iters` command buffers, GB/s.
pub fn shootout(s: &mut Synth, iters: usize) -> Result<(), String> {
    use crate::qwen_layers::{mma_gemv_enc, mma_swiglu_enc, mma_twin_gemv_enc, mma_twin_nt, mma_twin_swiglu_enc, mma_twin_tmem};
    for nb in [1usize, 2] {
        for m in [0u32, 2, 3] { s.gpu.pipeline(&format!("qm_gemv_n{}_s8_m{}", nb, m))?; }
        s.gpu.pipeline(&format!("qm_swiglu_n{}_s8", nb))?;
    }
    let s = &*s;
    let (mut dn, mut at, mut ff) = (vec![], vec![], vec![]);
    for l in s.full.iter() {
        match l.mixer { Mixer::Dn(d) => if dn.len() < 2 { dn.push(d) }, Mixer::Attn(a) => if at.len() < 2 { at.push(a) } }
        if ff.len() < 2 && !ff.iter().any(|f: &FfnW| f.gate.w.off == l.ffn.gate.w.off) { ff.push(l.ffn); }
    }
    let io = &s.io;
    let tm = |f: &dyn Fn(&Enc, usize)| -> Result<f64, String> {
        let mut v = vec![];
        for _ in 0..iters + 1 {
            let cb = s.gpu.command_buffer();
            let e = Enc::new(&s.gpu, &cb, &s.buf);
            for i in 0..32 { f(&e, i % 2); }
            e.end();
            commit(&cb);
            wait(&cb)?;
            v.push(gpu_ms(&cb) / 32.0);
        }
        Ok(med(&v[1..]))
    };
    let shapes: Vec<(&str, [Q4b; 2], u32)> = vec![
        ("dn.in_proj", [dn[0].in_proj, dn[1].in_proj], 0), ("dn.out_proj", [dn[0].out_proj, dn[1].out_proj], 2),
        ("attn.in_proj", [at[0].in_proj, at[1].in_proj], 0), ("attn.o_proj", [at[0].o_proj, at[1].o_proj], 2),
        ("ffn.down", [ff[0].down, ff[1].down], 2), ("lm_head", [s.w.lm_head, s.w.lm_head], 3)];
    println!("\n== K = 1 kernel shootout (bit-identical Mma family, SK 8): ms per dispatch (GB/s)");
    for (nm, ms, mode) in shapes {
        let b = (ms[0].rows * ms[0].cols / 2 + 2 * ms[0].rows * ms[0].cols / 32) as f64;
        let nt = mma_twin_nt(ms[0].cols, false, 1, 8, 512);
        let x = if ms[0].cols == s.cfg.hidden { io.x } else { io.f };
        let r0 = tm(&|e, i| mma_twin_gemv_enc(e, &format!("qm_gemvt_k1_s8_m{}", mode), nt, &ms[i], x, io.logits))?;
        let r1 = tm(&|e, i| { e.tgmem(0, mma_twin_tmem(nt, false, 1, 8)); mma_twin_gemv_enc(e, &format!("qm_gemvtd_k1_s8_m{}", mode), nt, &ms[i], x, io.logits) })?;
        let r2 = tm(&|e, i| mma_gemv_enc(e, &format!("qm_gemv_n1_s8_m{}", mode), 1, 8, &ms[i], 1, x, io.logits))?;
        let r3 = tm(&|e, i| mma_gemv_enc(e, &format!("qm_gemv_n2_s8_m{}", mode), 2, 8, &ms[i], 1, x, io.logits))?;
        let g = |t: f64| format!("{:.3} ({:.0})", t, b / t / 1e6);
        println!("{:<14} {:>5}x{:<5} twin {} | twin-dyn {} | mma n1 {} | mma n2 {}", nm, ms[0].rows, ms[0].cols, g(r0), g(r1), g(r2), g(r3));
    }
    let (g0, g1) = (&ff[0], &ff[1]);
    let b = 2.0 * (g0.gate.rows * g0.gate.cols / 2 + 2 * g0.gate.rows * g0.gate.cols / 32) as f64;
    let nt = mma_twin_nt(g0.gate.cols, true, 1, 8, 512);
    let pair = |i: usize| if i == 0 { g0 } else { g1 };
    let r0 = tm(&|e, i| mma_twin_swiglu_enc(e, "qm_swiglut_k1_s8", nt, &pair(i).gate, &pair(i).up, io.x, io.f))?;
    let r1 = tm(&|e, i| { e.tgmem(0, mma_twin_tmem(nt, true, 1, 8)); mma_twin_swiglu_enc(e, "qm_swiglutd_k1_s8", nt, &pair(i).gate, &pair(i).up, io.x, io.f) })?;
    let r2 = tm(&|e, i| mma_swiglu_enc(e, "qm_swiglu_n1_s8", 1, 8, &pair(i).gate, &pair(i).up, 1, io.x, io.f))?;
    let r3 = tm(&|e, i| mma_swiglu_enc(e, "qm_swiglu_n2_s8", 2, 8, &pair(i).gate, &pair(i).up, 1, io.x, io.f))?;
    let g = |t: f64| format!("{:.3} ({:.0})", t, b / t / 1e6);
    println!("{:<14} {:>5}x{:<5} twin {} | twin-dyn {} | mma n1 {} | mma n2 {}", "ffn.gate+up", g0.gate.rows, g0.gate.cols, g(r0), g(r1), g(r2), g(r3));
    Ok(())
}
/// MmaD twin sweep over rows per register pass (RP), output rows per threadgroup (RT) and threads per group (chunks).
/// None of them is part of the result (a row's group terms and their order never change), so every variant is
/// bit-identical. Prints ms per dispatch and GB/s for each K = 1 matmul shape.
pub fn nt_sweep(s: &Synth, iters: usize) -> Result<(), String> {
    let io = &s.io;
    let mut ff: Vec<FfnW> = vec![];
    let (mut dn, mut at) = (vec![], vec![]);
    for l in s.full.iter() {
        if ff.len() < 2 && !ff.iter().any(|f| f.gate.w.off == l.ffn.gate.w.off) { ff.push(l.ffn); }
        match l.mixer {
            Mixer::Dn(d) => if dn.len() < 2 && !dn.iter().any(|x: &DnW| x.out_proj.w.off == d.out_proj.w.off) { dn.push(d) },
            Mixer::Attn(a) => if at.len() < 2 && !at.iter().any(|x: &AttnW| x.o_proj.w.off == a.o_proj.w.off) { at.push(a) },
        }
    }
    let tm = |f: &dyn Fn(&Enc, usize)| -> Result<f64, String> {
        let mut v = vec![];
        for _ in 0..iters + 1 {
            let cb = s.gpu.command_buffer();
            let e = Enc::new(&s.gpu, &cb, &s.buf);
            for i in 0..32 { f(&e, i % 2); }
            e.end();
            commit(&cb);
            wait(&cb)?;
            v.push(gpu_ms(&cb) / 32.0);
        }
        Ok(med(&v[1..]))
    };
    // (suffix, RP, RT)
    let vars = [("", 4usize, 8usize), ("_t4", 4, 4)];
    println!("\n== MmaD twin sweep (K = 1): ms per dispatch (GB/s) by variant (RP, RT) and threads per group");
    let shapes: Vec<(&str, [Q4b; 2], usize, u32)> = vec![
        ("dn.in_proj", [dn[0].in_proj, dn[1].in_proj], io.x, 0), ("dn.out_proj", [dn[0].out_proj, dn[1].out_proj], io.f, 2),
        ("attn.in_proj", [at[0].in_proj, at[1].in_proj], io.x, 0), ("ffn.down", [ff[0].down, ff[1].down], io.f, 2),
        ("lm_head", [s.w.lm_head, s.w.lm_head], io.x, 3)];
    for (nm, ms, xo, mode) in shapes {
        let g = ms[0].cols / 64;
        let b = (ms[0].rows * ms[0].cols / 2 + 2 * ms[0].rows * ms[0].cols / 32) as f64;
        println!("{} {}x{}", nm, ms[0].rows, ms[0].cols);
        for (sfx, rp, rt) in vars {
            let np = rt / rp;
            let mut line = format!("  rp {} rt {:>2}", rp, rt);
            for chunks in [1usize, 2] {
                let nt = (g.div_ceil(chunks) * np).max(8 * rt);
                if nt > 1024 { continue; }
                let k = format!("qm_gemvtd_k1{}_s8_m{}", sfx, mode);
                let tmem = ((rp * nt).max(8 * rt) * 4).div_ceil(16) * 16;
                let t = tm(&|e, i| {
                    let m = &ms[i];
                    e.tgmem(0, tmem).pso(&k).b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, xo).b(4, io.logits)
                        .u(5, m.rows as u32).u(6, m.cols as u32).b(7, io.logits).b(8, io.logits).groups((m.rows.div_ceil(rt), 1), nt);
                })?;
                line += &format!(" | nt {:>4} {:.3} ({:.0})", nt, t, b / t / 1e6);
            }
            println!("{}", line);
        }
    }
    let g = ff[0].gate.cols / 64;
    let b = 2.0 * (ff[0].gate.rows * ff[0].gate.cols / 2 + 2 * ff[0].gate.rows * ff[0].gate.cols / 32) as f64;
    println!("ffn.gate+up {}x{}", ff[0].gate.rows, ff[0].gate.cols);
    for (sfx, rp, rt) in vars {
        let np = 2 * rt / rp;
        let mut line = format!("  rp {} rt {:>2}", rp, rt);
        for chunks in [1usize, 2] {
            let nt = (g.div_ceil(chunks) * np).max(16 * rt);
            if nt > 1024 { continue; }
            let k = format!("qm_swiglutd_k1{}_s8", sfx);
            let tmem = ((rp * nt).max(16 * rt) * 4).div_ceil(16) * 16;
            let t = tm(&|e, i| {
                let (gt, up) = (&ff[i].gate, &ff[i].up);
                e.tgmem(0, tmem).pso(&k).b(0, gt.w.off).b(1, gt.s.off).b(2, gt.b.off).b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
                    .b(6, io.x).b(7, io.f).u(8, gt.rows as u32).u(9, gt.cols as u32).groups((gt.rows.div_ceil(rt), 1), nt);
            })?;
            line += &format!(" | nt {:>4} {:.3} ({:.0})", nt, t, b / t / 1e6);
        }
        println!("{}", line);
    }
    Ok(())
}

/// Op family of a `labels` class (verify profile).
fn family(cl: &str) -> &'static str {
    if cl.starts_with("swiglu") || cl.starts_with("gemv ffn") || cl == "compact down" { "FFN (gate+up+swiglu, down)" }
    else if cl.starts_with("gemv dn") { "DeltaNet projections (in, out)" }
    else if cl.starts_with("gemv attn") { "attention projections (in, o)" }
    else if cl == "gemv lm_head" { "lm_head" }
    else if cl == "vf_xprep" { "vf_xprep (fused-verify input prep)" }
    else if cl.starts_with("dn_") || cl == "rmsnorm_gated" { "DeltaNet core (conv, prep, recur, gated norm, save)" }
    else if cl.starts_with("attn_") || cl.starts_with("va_sdpa") { "attention core (prep/RoPE/KV append, sdpa)" }
    else { "small (norms, embed, argmax)" }
}

/// Exact verify pass of K rows (27B shapes, random weights): whole-pass GPU ms with the fused verify matmuls off / on,
/// and per op family (each family alone in one command buffer via `Enc::with_filter`) with bytes and % of the read
/// ceiling; sum of families - whole = what overlap between dispatches hides (negative) / bubbles (none to see here).
pub fn verify_profile(s: &Synth, ks: &[usize], modes: &[(bool, bool)], iters: usize, classes: bool) -> Result<(), String> {
    let (ceil, _) = read_ceiling(s, 5)?;
    println!("read ceiling {:.0} GB/s; pos {}; verify pass = embed + {} layers (ck, dn_save) + head on all K rows", ceil, s.opt.pos, s.cfg.layers);
    for &k in ks {
        if !(1..=8).contains(&k) { return Err(format!("verify K {} not in 1..=8", k)); }
        for &(vf, va) in modes {
            // 4 passes back to back per command buffer (GPU clock stays up, as in a decode loop); names of one pass
            const REP: usize = 4;
            let enc = |e: &Enc| s.encode_verify(e, k, vf, va);
            let enc4 = |e: &Enc| { for _ in 0..REP { s.encode_verify(e, k, vf, va)?; } Ok(()) };
            let whole = med(&s.time_with(&enc4, None, iters)?) / REP as f64;
            let names = s.names_of(&enc)?;
            let nd = names.len();
            let labs = s.labels(&names, false);
            let mut cls: Vec<String> = vec![];
            for (l, _) in &labs { if !cls.contains(l) { cls.push(l.clone()); } }
            let mut fams: Vec<(&str, usize, f64, f64)> = vec![]; // family, dispatches, ms, bytes
            let mut cl_lines = vec![format!("{:<34} {:>5} {:>9} {:>9} {:>7} {:>6}", "  class", "n", "ms", "MB", "GB/s", "%ceil")];
            for cl in &cls {
                let idx: Vec<usize> = labs.iter().enumerate().filter(|(_, x)| &x.0 == cl).map(|(i, _)| i).collect();
                let bytes: f64 = idx.iter().map(|&i| labs[i].1).sum();
                let f = |i: usize, _: &str| labs.get(i % nd).is_some_and(|x| &x.0 == cl);
                let ms = med(&s.time_with(&enc4, Some(&f), iters.min(7))?) / REP as f64;
                cl_lines.push(format!("  {:<32} {:>5} {:>9.3} {:>9.1} {:>7.0} {:>5.0}%", cl, idx.len(), ms, bytes / 1e6, bytes / ms / 1e6, 100.0 * bytes / ms / 1e6 / ceil));
                let fm = family(cl);
                match fams.iter_mut().find(|x| x.0 == fm) {
                    Some(x) => { x.1 += idx.len(); x.2 += ms; x.3 += bytes; }
                    None => fams.push((fm, idx.len(), ms, bytes)),
                }
            }
            println!("\n== verify K = {} | fused verify matmuls {} | verify mixers {} | {} dispatches | whole pass GPU {:.2} ms ({:.2} ms/row)", k,
                     if vf { "ON" } else { "off" }, if va { "ON" } else { "off" }, names.len(), whole, whole / k as f64);
            println!("| part | dispatches | ms | GB | GB/s | % of read ceiling |");
            println!("|---|---|---|---|---|---|");
            let (mut sm, mut sb) = (0.0, 0.0);
            for (fm, n, ms, b) in &fams {
                sm += ms;
                sb += b;
                let g = if *b > 0.0 { format!("{:.0} | {:.0}%", b / ms / 1e6, 100.0 * b / ms / 1e6 / ceil) } else { "- | -".into() };
                println!("| {} | {} | {:.2} | {:.3} | {} |", fm, n, ms, b / 1e9, g);
            }
            println!("| sum of parts (each alone) | {} | {:.2} | {:.2} | {:.0} | {:.0}% |", names.len(), sm, sb / 1e9, sb / sm / 1e6, 100.0 * sb / sm / 1e6 / ceil);
            println!("| whole pass | {} | {:.2} | {:.2} | {:.0} | {:.0}% |  (sum - whole = {:.2} ms)", names.len(), whole, sb / 1e9, sb / whole / 1e6,
                     100.0 * sb / whole / 1e6 / ceil, sm - whole);
            if classes { println!("{}", cl_lines.join("\n")); }
        }
    }
    Ok(())
}

/// Whole verify-pass GPU ms only, modes interleaved per K and repeated `rounds` times (A/B on a busy machine):
/// median over rounds of each round's median.
pub fn verify_whole(s: &Synth, ks: &[usize], modes: &[(bool, bool)], iters: usize, rounds: usize) -> Result<(), String> {
    let name = |m: (bool, bool)| match m { (false, false) => "off", (true, false) => "vf", (true, true) => "vf+mixers", (false, true) => "mixers" };
    println!("verify pass GPU ms (pos {}; median of {} rounds, each a command buffer of {} passes back to back; modes interleaved)", s.opt.pos, rounds, iters);
    println!("| K | {} |", modes.iter().map(|&m| name(m)).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---|".repeat(modes.len()));
    for &k in ks {
        let mut r: Vec<Vec<f64>> = vec![vec![]; modes.len()];
        for _ in 0..rounds {
            for (i, &(vf, va)) in modes.iter().enumerate() {
                // `iters` passes back to back in one command buffer (GPU clock stays up, as in a decode loop)
                // K > 8 without fused verify: no single pass exists (MMA kernels stop at 8) -> chunks of <= 8 rows
                // at the same pos (what a K-row check costs on today's non-fused kernels; attention +8 keys ignored)
                let enc1 = |e: &Enc| -> Result<(), String> {
                    if k <= 8 || vf { return s.encode_verify(e, k, vf, va); }
                    let mut r = k;
                    while r > 0 { let c = r.min(8); s.encode_verify(e, c, false, va)?; r -= c; }
                    Ok(())
                };
                let v = s.time_with(&|e: &Enc| { for _ in 0..iters { enc1(e)?; } Ok(()) }, None, 2)?;
                r[i].push(med(&v) / iters as f64);
            }
        }
        println!("| {} | {} |", k, r.iter().map(|v| format!("{:.2}", med(v))).collect::<Vec<_>>().join(" | "));
    }
    Ok(())
}

/// whole batched verify pass GPU ms per B x R config (fused verify matmuls unless `no_vf`), configs interleaved per
/// round; median over rounds of each round's median (2 timed command buffers of `iters` passes back to back). The
/// 1 x K row is also timed through `encode_verify` (today's single-sequence encoder) as a cross-check.
pub fn batch_whole(s: &Synth, cfgs: &[(usize, usize)], no_vf: bool, iters: usize, rounds: usize) -> Result<(), String> {
    let vf = !no_vf;
    println!("batched verify pass GPU ms (pos {}, max_ctx {}; fused verify {}; median of {} rounds, each a command buffer of {} passes; configs interleaved)",
             s.opt.pos, s.opt.max_ctx, if vf { "ON" } else { "off" }, rounds, iters);
    let mut r: Vec<Vec<f64>> = vec![vec![]; cfgs.len()];
    let mut x: Vec<Vec<f64>> = vec![vec![]; cfgs.len()];
    for _ in 0..rounds {
        for (i, &(b, rr)) in cfgs.iter().enumerate() {
            let v = s.time_with(&|e: &Enc| { for _ in 0..iters { s.encode_verify_batch(e, b, rr, vf)?; } Ok(()) }, None, 2)?;
            r[i].push(med(&v) / iters as f64);
            if b == 1 {
                let v = s.time_with(&|e: &Enc| { for _ in 0..iters { s.encode_verify(e, rr, vf, false)?; } Ok(()) }, None, 2)?;
                x[i].push(med(&v) / iters as f64);
            }
        }
    }
    println!("| B x R | rows | ms/pass | ms/pass per sequence | encode_verify (1 x K check) | rounds |");
    println!("|---|---|---|---|---|---|");
    for (i, &(b, rr)) in cfgs.iter().enumerate() {
        let m = med(&r[i]);
        println!("| {}x{} | {} | {:.2} | {:.2} | {} | {} |", b, rr, b * rr, m, m / b as f64,
                 if x[i].is_empty() { "-".to_string() } else { format!("{:.2}", med(&x[i])) },
                 r[i].iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
