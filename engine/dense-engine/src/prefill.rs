//! GEMM prefill for long prompts: every projection and the FFN run as the engine's simdgroup-matrix 4-bit GEMM
//! (engine/shaders/gemm_q4.metal, read-only via include_str!, dispatched as streamdense-model
//! `qwen_layers::encode_layer_prefill` does for the 27B prefill) over T token rows at once, attention as
//! `attn128_flash` (shaders/attn_hd128_flash.metal), instead of decode gemv passes of <= 8 rows.
//!
//! Per chunk of T rows (T <= `t_max`, one command buffer per chunk, waited on), per layer:
//! ```text
//! rmsnorm_b                 groups (T) x 1024                         x = norm(h)
//! q4b_gemm_{BM}x64_m0       groups (ceil(P/64), ceil(T/BM)) x 128     p = [T][q | k | v]   (fused qkv, weights.rs)
//! qd_bias_add (Qwen2/2.5)   threads (P, T)                            p += bias
//! attn128_prep_b16          groups (nh + nkv, T) x 128                QK-norm (Qwen3) + RoPE -> qr; K/V at pos + t
//! attn128_flash             groups (ceil(T/8), nkv) x 32*grp          ao (causal: row t sees keys 0 ..= pos + t)
//! q4b_gemm_{BM}x64_m2       o_proj, h += o ao (residual in place)
//! rmsnorm_b; q4b_gemm_swiglu_{BM}x64 (gate|up, groups (ceil(I/32), ceil(T/BM))); q4b_gemm_{BM}x64_m2 down (h += ..)
//! ```
//! BM = 32 for T <= 32, else 64 (qwen_layers::gemm_tile, measured on M4 Pro). Head after the last chunk (last row
//! only), identical to the decode head: rmsnorm_b -> `qm_gemvt_k1_s8_m3` lm_head (f32 logits; `qd_gemv_b16_f32` when
//! tied to an unquantized embedding) -> argmax_b.
//!
//! Weights and KV cache stay where the engine put them (its arena, `weights.rs` layout unchanged: the GEMM reads the
//! same MLX layout W u32 [R][C/8], bf16 S / B [R][C/64] as the gemv family). The T-row scratch lives in a SEPARATE
//! buffer owned by `Prefill` (sized for `t_max` rows; ~20 MB at Qwen3-1.7B shapes, T = 512), so the engine's arena
//! is untouched. Kernels bind (arena, offset) or (scratch, offset).
//!
//! NOT bit-exact with the decode-pass prefill: GEMM summation order (fp32 simdgroup MMA) and flash attention
//! differ from the MMA gemv / split-K, so bf16 outputs occasionally differ by an ulp and that propagates; held to a
//! tolerance (tests/prefill_gemm.rs).
//!
//! Other dense families: the same per-layer steps as `model::encode_layer` with GEMMs: embedding multiplier
//! (qd_scale_b16), attention kernels of the model's head dim / scale / soft cap / per-layer window (`attn{HD}_prep_b16`,
//! `attn{HD}_flash` with function constants), unfused residuals (o / down bias, post-attention / post-feedforward
//! norm, residual multiplier: GEMM m0 -> qd_bias_add -> rmsnorm_b -> qd_residual), unfused FFN (GeGLU or MLP bias:
//! gate GEMM m0, up GEMM m0, biases, qd_glu), logits scaling / final soft cap (qd_logits_post) before argmax_b.
//! Layer forms, norms, plain MLP, OLMo 2 full-projection QK-norm and the prep variants as `model::encode_layer`
//! (see model.rs), with GEMMs.
use std::ptr::NonNull;

use objc2_metal::*;

use super::config::{rope_inv_freq, su_inv_freq, Block, DenseConfig, NormKind, RopeScaling};
use super::gpu::{Arena, Buffer, Gpu, SdCtrl};
use super::model::{self, Dims};
use super::weights::{LayerW, Q4, Weights};

const BF: usize = 2;
const F32: usize = 4;

/// Library source with every kernel the prefill (and the decode engine) uses: the engine library
/// (`ENGINE_SHADER_SOURCE` of lib.rs, which includes engine gemm_q4.metal and shaders/attn_hd128_flash.metal).
pub const PREFILL_SHADER_SOURCE: [&str; 9] = crate::ENGINE_SHADER_SOURCE;

/// GEMM tiles built in gemm_q4.metal (BM token rows x BN weight rows), 128 threads each.
const GEMM_TILES: [(usize, usize); 2] = [(32, 64), (64, 64)];

/// Tile for T rows (qwen_layers::gemm_tile: 32x64 best up to T = 32, 64x64 above).
pub fn gemm_tile(t: usize) -> (usize, usize) { if t <= 32 { GEMM_TILES[0] } else { GEMM_TILES[1] } }

/// Every kernel the prefill dispatches for `cfg` (attention kernels per head dim / scale / soft cap / window).
pub fn pipelines(cfg: &DenseConfig) -> Vec<String> {
    let mut v = Vec::new();
    for (bm, bn) in GEMM_TILES {
        for m in [0, 2] { v.push(format!("q4b_gemm_{}x{}_m{}", bm, bn, m)); }
        v.push(format!("q4b_gemm_swiglu_{}x{}", bm, bn));
    }
    for n in ["qd_embed_q4", "rmsnorm_b", "qd_bias_add", "attn128_rope_invf", "argmax_b", "qm_gemvt_k1_s8_m3",
              "qd_scale_b16", "qd_residual", "qd_glu", "qd_logits_post", "qd_embed_b16", "qd_gemv_b16_f32",
              "qd_layernorm", "qd_rmsnorm_rows", "qd_act_b16"] {
        v.push(n.into());
    }
    let d = Dims::new(cfg);
    for (w, r) in model::layer_windows(cfg).into_iter().zip(model::layer_rope(cfg)) {
        let av = d.variant(w, r);
        v.push(av.prep());
        v.push(av.flash());
    }
    v.sort();
    v.dedup();
    v
}

/// The model the prefill runs over: config, weights (arena offsets) and per-layer KV caches (bf16
/// [nkv][max_ctx][head_dim], 16-byte aligned) inside `arena`. From an `Engine`: `PrefillModel { cfg: &eng.cfg, w: &eng.w,
/// kv: eng.kv.iter().map(|k| (k.k, k.v)).collect(), max_ctx: eng.max_ctx, arena: &eng.arena.buf }`.
pub struct PrefillModel<'a> {
    pub cfg: &'a DenseConfig,
    pub w: &'a Weights,
    pub kv: Vec<(usize, usize)>,
    pub max_ctx: usize,
    pub arena: &'a Buffer,
}

/// Scratch offsets (in `Prefill::scratch`).
#[derive(Clone, Copy, Debug, Default)]
struct Io {
    tok: usize,
    h: usize,
    x: usize,
    p: usize,
    qr: usize,
    ao: usize,
    f: usize,
    hf: usize,
    logits: usize,
    invf: usize,
    ctrl: usize,
    hctrl: usize,
    rec: usize,
    dummy: usize,
    /// up rows (unfused FFN), residual branch, sliding layers' RoPE table (= invf unless rope_local_theta)
    f2: usize,
    r: usize,
    invf_local: usize,
}

/// GEMM prefill state: scratch for up to `t_max` rows of one model shape.
pub struct Prefill {
    pub t_max: usize,
    pub scratch: Arena,
    io: Io,
    hidden: usize,
    vocab: usize,
    dims: Dims,
    windows: Vec<Option<u32>>,
    rope: Vec<bool>,
}

/// Encoder binding either the engine arena (`a`) or the prefill scratch (`s`).
struct PEnc<'a> {
    e: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>>,
    gpu: &'a Gpu,
    arena: &'a Buffer,
    scratch: &'a Buffer,
}

impl PEnc<'_> {
    fn pso(&self, name: &str) -> &Self { self.e.setComputePipelineState(self.gpu.pso(name)); self }
    fn bind(&self, buf: &Buffer, i: usize, off: usize) -> &Self {
        // SAFETY: offset into a live buffer; kernels index within the regions the caller sized.
        unsafe { self.e.setBuffer_offset_atIndex(Some(buf), off, i) };
        self
    }
    fn a(&self, i: usize, off: usize) -> &Self { self.bind(self.arena, i, off) }
    fn s(&self, i: usize, off: usize) -> &Self { self.bind(self.scratch, i, off) }
    fn u(&self, i: usize, v: u32) -> &Self {
        // SAFETY: 4 bytes copied from a live u32.
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i) };
        self
    }
    fn f(&self, i: usize, v: f32) -> &Self {
        // SAFETY: 4 bytes copied from a live f32.
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i) };
        self
    }
    fn groups(&self, g: (usize, usize), t: usize) {
        self.e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: g.0, height: g.1, depth: 1 },
                                                          MTLSize { width: t, height: 1, depth: 1 });
    }
    fn threads(&self, g: (usize, usize), t: usize) {
        self.e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: g.0, height: g.1, depth: 1 },
                                                     MTLSize { width: t, height: 1, depth: 1 });
    }

    /// 4-bit GEMM of `t` scratch rows x through `m`; MODE 0: out = bf16; 2: out += (residual, in place).
    fn gemm(&self, m: &Q4, t: usize, mode: u32, x: usize, out: usize) {
        let (bm, bn) = gemm_tile(t);
        self.pso(&format!("q4b_gemm_{}x{}_m{}", bm, bn, mode))
            .a(0, m.w).a(1, m.s).a(2, m.b).s(3, x)
            .s(4, out).u(5, m.rows as u32).u(6, m.cols as u32).s(7, out).s(8, out).u(9, t as u32)
            .groups((m.rows.div_ceil(bn), t.div_ceil(bm)), 128);
    }
    fn swiglu(&self, gate: &Q4, up: &Q4, t: usize, x: usize, out: usize) {
        let (bm, bn) = gemm_tile(t);
        self.pso(&format!("q4b_gemm_swiglu_{}x{}", bm, bn))
            .a(0, gate.w).a(1, gate.s).a(2, gate.b).a(3, up.w).a(4, up.s).a(5, up.b)
            .s(6, x).s(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32).u(10, t as u32)
            .groups((gate.rows.div_ceil(bn / 2), t.div_ceil(bm)), 128);
    }
    fn rmsnorm(&self, n: usize, eps: f32, t: usize, x: usize, w: usize, out: usize) {
        self.pso("rmsnorm_b").s(0, x).a(1, w).s(2, out).u(3, n as u32).f(4, eps).u(5, n as u32).groups((t, 1), 1024);
    }
    /// t rows of hidden size `n`: RMSNorm (rmsnorm_b) or LayerNorm with optional bias (qd_layernorm).
    fn norm(&self, kind: NormKind, n: usize, eps: f32, t: usize, x: usize, w: usize, b: Option<usize>, out: usize) {
        match kind {
            NormKind::Rms => self.rmsnorm(n, eps, t, x, w, out),
            NormKind::LayerNorm { .. } => {
                self.pso("qd_layernorm").s(0, x).a(1, w).a(2, b.unwrap_or(w)).s(3, out).u(4, n as u32).f(5, eps)
                    .u(6, b.is_some() as u32).groups((t, 1), 1024);
            }
        }
    }
    /// scratch p bf16 [t][n] += arena bias [n]
    fn bias_add(&self, p: usize, b: usize, n: usize, t: usize) {
        self.pso("qd_bias_add").s(0, p).a(1, b).u(2, n as u32).threads((n, t), 256);
    }
}

/// Threads per group of the K = 1 twin gemv (model.rs / qwen_layers::mma_twin_nt with k = 1, sk = 8, tmax = 512).
fn twin_nt_k1(cols: usize) -> usize {
    let (rp, np, sk) = (4usize, 2usize, 8usize);
    let cap = 512usize.min(4096 / rp).max(np);
    let g = cols / 64;
    let chunks = (g * np).div_ceil(cap);
    let nt = g.div_ceil(chunks) * np;
    nt.max((8 * sk).div_ceil(np) * np)
}

impl Prefill {
    /// Build the pipelines on `gpu` (its library must hold `PREFILL_SHADER_SOURCE`'s kernels) and allocate the
    /// scratch for chunks of up to `t_max` rows.
    pub fn new(gpu: &mut Gpu, cfg: &DenseConfig, t_max: usize) -> Result<Prefill, String> {
        if t_max == 0 { return Err("prefill: t_max 0".into()); }
        if !crate::config::HEAD_DIMS.contains(&cfg.head_dim) {
            return Err(format!("prefill: head_dim {} (need 64, 96, 128, 160 or 256)", cfg.head_dim));
        }
        let grp = cfg.grp();
        if grp == 0 || cfg.heads % cfg.kv_heads != 0 || grp > 32 {
            return Err(format!("prefill: heads {} / kv_heads {} (attn flash needs grp <= 32)", cfg.heads, cfg.kv_heads));
        }
        if cfg.hidden % 64 != 0 || cfg.inter % 64 != 0 { return Err("prefill: hidden / inter must be multiples of 64".into()); }
        for p in pipelines(cfg) { gpu.pipeline(&p)?; }
        let dims = Dims::new(cfg);
        let (h, t) = (cfg.hidden, t_max);
        let sizes = [
            t * 4, t * h * BF, t * h * BF, t * cfg.attn_in_rows() * BF, t * cfg.q_dim() * BF, t * cfg.q_dim() * BF,
            t * cfg.inter * BF, h * BF, cfg.vocab * F32, (cfg.rope_dims / 2) * F32, std::mem::size_of::<SdCtrl>(),
            std::mem::size_of::<SdCtrl>(), 16, 256,
            if dims.ffn_unfused { t * cfg.inter * BF } else { 0 },
            if dims.resid_mult != 1.0 || cfg.o_bias || cfg.mlp_bias || cfg.four_norms || cfg.block != Block::Sequential { t * h * BF } else { 0 },
            if cfg.rope_local_theta.is_some() { (cfg.rope_dims / 2) * F32 } else { 0 },
        ];
        let total = sizes.iter().map(|n| (n + 255) / 256 * 256).sum::<usize>() + 256;
        let mut scratch = Arena::new(gpu, total)?;
        let mut it = sizes.iter().map(|&n| scratch.alloc(n)).collect::<Vec<_>>().into_iter();
        let mut nx = || it.next().unwrap();
        let mut io = Io {
            tok: nx(), h: nx(), x: nx(), p: nx(), qr: nx(), ao: nx(), f: nx(), hf: nx(), logits: nx(), invf: nx(),
            ctrl: nx(), hctrl: nx(), rec: nx(), dummy: nx(), f2: nx(), r: nx(), invf_local: nx(),
        };
        if cfg.rope_local_theta.is_none() { io.invf_local = io.invf; }
        let pf = Prefill { t_max, scratch, io, hidden: h, vocab: cfg.vocab, dims, windows: model::layer_windows(cfg),
                           rope: model::layer_rope(cfg) };
        let n = cfg.rope_dims / 2;
        if let Some(su) = &cfg.rope_su {
            pf.scratch.put(pf.io.invf, &su_inv_freq(cfg.rope_dims, cfg.rope_theta, su));
        } else if cfg.rope_scale == RopeScaling::None {
            let l2 = (cfg.rope_theta as f32).log2();
            pf.run_cb(gpu, &pf.scratch.buf, |e| { e.pso("attn128_rope_invf").s(0, pf.io.invf).f(1, l2).threads((n, 1), n); })?;
        } else {
            pf.scratch.put(pf.io.invf, &rope_inv_freq(cfg.rope_dims, cfg.rope_theta, cfg.rope_scale));
        }
        if let Some(th) = cfg.rope_local_theta {
            let l2 = (th as f32).log2();
            pf.run_cb(gpu, &pf.scratch.buf, |e| { e.pso("attn128_rope_invf").s(0, pf.io.invf_local).f(1, l2).threads((n, 1), n); })?;
        }
        Ok(pf)
    }

    /// Scratch bytes.
    pub fn bytes(&self) -> usize { self.scratch.size }

    fn run_cb(&self, gpu: &Gpu, arena: &Buffer, f: impl FnOnce(&PEnc)) -> Result<(), String> {
        let cb = gpu.command_buffer();
        let e = PEnc { e: cb.computeCommandEncoder().ok_or("prefill: no encoder")?, gpu, arena, scratch: &self.scratch.buf };
        f(&e);
        e.e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if let Some(err) = cb.error() { return Err(format!("prefill: GPU error: {}", err.localizedDescription())); }
        Ok(())
    }

    /// Prefill `tokens` at positions `pos ..` (KV rows pos .. pos + len - 1 of every layer are written), in chunks
    /// of up to `t_max` rows; with `head`, the greedy next token after the last one (logits via `logits()`).
    /// The caller owns the position bookkeeping (Engine: pos += tokens.len()).
    pub fn run(&self, gpu: &Gpu, m: &PrefillModel, tokens: &[u32], pos: usize, head: bool) -> Result<Option<u32>, String> {
        let cfg = m.cfg;
        if tokens.is_empty() { return Err("prefill: no tokens".into()); }
        if cfg.hidden != self.hidden || cfg.vocab != self.vocab || cfg.head_dim != self.dims.hd {
            return Err("prefill: model shape != the one Prefill was built for".into());
        }
        if pos + tokens.len() > m.max_ctx {
            return Err(format!("prefill: context full ({} + {} > max_ctx {})", pos, tokens.len(), m.max_ctx));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= cfg.vocab) {
            return Err(format!("prefill: token {} >= vocab {}", t, cfg.vocab));
        }
        if m.kv.len() != m.w.layers.len() || m.kv.len() != self.windows.len() { return Err("prefill: kv / layers count mismatch".into()); }
        if m.kv.iter().any(|&(k, v)| k % 16 != 0 || v % 16 != 0) { return Err("prefill: KV must be 16-byte aligned".into()); }
        let n = tokens.chunks(self.t_max).count();
        for (i, c) in tokens.chunks(self.t_max).enumerate() {
            let p0 = pos + i * self.t_max;
            let last = head && i + 1 == n;
            self.scratch.put(self.io.tok, c);
            self.scratch.put(self.io.ctrl, &[SdCtrl { pos: p0 as u32, ..Default::default() }]);
            self.scratch.put(self.io.hctrl, &[SdCtrl::default()]);
            self.run_cb(gpu, m.arena, |e| {
                self.encode_chunk(e, m, c.len());
                if last { self.encode_head(e, m, c.len()); }
            })?;
        }
        if !head { return Ok(None); }
        let c: Vec<SdCtrl> = self.scratch.get(self.io.hctrl, 1);
        Ok(Some(c[0].tok[0]))
    }

    /// f32 logits of the last head.
    pub fn logits(&self) -> Vec<f32> { self.scratch.get(self.io.logits, self.vocab) }

    /// Residual stream (bf16 bits) of the last chunk's row `t` after all layers (before the final norm).
    pub fn hidden_row(&self, t: usize) -> Vec<u16> { self.scratch.get(self.io.h + t * self.hidden * BF, self.hidden) }

    fn encode_chunk(&self, e: &PEnc, m: &PrefillModel, t: usize) {
        let (cfg, io) = (m.cfg, &self.io);
        if m.w.embed_bf16 {
            e.pso("qd_embed_b16").a(0, m.w.embed.w).s(1, io.tok).s(2, io.h).u(3, cfg.hidden as u32).threads((cfg.hidden, t), 64);
        } else {
            e.pso("qd_embed_q4").a(0, m.w.embed.w).a(1, m.w.embed.s).a(2, m.w.embed.b).s(3, io.tok).s(4, io.h)
                .u(5, cfg.hidden as u32).threads((cfg.hidden / 8, t), 64);
        }
        if self.dims.embed_scale != 1.0 {
            let n = t * cfg.hidden;
            e.pso("qd_scale_b16").s(0, io.h).u(1, n as u32).f(2, self.dims.embed_scale).threads((n, 1), 256);
        }
        for (((lw, &kv), &win), &rope) in m.w.layers.iter().zip(&m.kv).zip(&self.windows).zip(&self.rope) {
            self.encode_layer(e, m, lw, kv, win, rope, t);
        }
    }

    /// h += mult * post(m x + bias) over t rows (fused GEMM MODE 2 when there is nothing else, as before).
    fn residual(&self, e: &PEnc, mat: &Q4, bias: Option<usize>, post: Option<(usize, Option<usize>)>, t: usize, x: usize) {
        let (d, io) = (&self.dims, &self.io);
        if bias.is_none() && post.is_none() && d.resid_mult == 1.0 {
            e.gemm(mat, t, 2, x, io.h);
            return;
        }
        e.gemm(mat, t, 0, x, io.r);
        if let Some(b) = bias { e.bias_add(io.r, b, d.hidden, t); }
        let r = match post { Some((w, b)) => { e.norm(d.norm, d.hidden, d.eps, t, io.r, w, b, io.x); io.x } None => io.r };
        let n = t * d.hidden;
        e.pso("qd_residual").s(0, io.h).s(1, r).u(2, n as u32).f(3, d.resid_mult).threads((n, 1), 256);
    }

    /// io.f = FFN activation of t rows of `x` (fused SwiGLU GEMM; gate / up GEMMs + biases + qd_glu; or `Mlp::Plain`
    /// up GEMM + bias + qd_act_b16).
    fn ffn(&self, e: &PEnc, lw: &LayerW, t: usize, x: usize) {
        let (d, io) = (&self.dims, &self.io);
        let n = t * d.inter;
        if d.mlp_plain {
            e.gemm(&lw.up, t, 0, x, io.f);
            if let Some(b) = lw.up_bias { e.bias_add(io.f, b, d.inter, t); }
            e.pso("qd_act_b16").s(0, io.f).u(1, n as u32).u(2, d.act).threads((n, 1), 256);
        } else if d.ffn_unfused {
            e.gemm(&lw.gate, t, 0, x, io.f);
            e.gemm(&lw.up, t, 0, x, io.f2);
            if let Some(b) = lw.gate_bias { e.bias_add(io.f, b, d.inter, t); }
            if let Some(b) = lw.up_bias { e.bias_add(io.f2, b, d.inter, t); }
            e.pso("qd_glu").s(0, io.f).s(1, io.f2).s(2, io.f).u(3, n as u32).u(4, d.act).threads((n, 1), 256);
        } else {
            e.swiglu(&lw.gate, &lw.up, t, x, io.f);
        }
    }

    fn encode_layer(&self, e: &PEnc, m: &PrefillModel, lw: &LayerW, (kc, vc): (usize, usize), win: Option<u32>, rope: bool, t: usize) {
        let (d, io) = (&self.dims, &self.io);
        let (h, nh, nkv, eps) = (d.hidden, d.nh, d.nkv, d.eps);
        let prow = lw.qkv.rows;
        let av = d.variant(win, rope);
        let ain = if d.block == Block::PostNorm { io.h } else {
            e.norm(d.norm, h, eps, t, io.h, lw.in_norm, lw.in_norm_bias, io.x);
            io.x
        };
        e.gemm(&lw.qkv, t, 0, ain, io.p);
        if let Some(b) = lw.qkv_bias { e.bias_add(io.p, b, prow, t); }
        if d.qk_full {
            let (qd, kvd) = (nh * d.hd, nkv * d.hd);
            for (off, n, w) in [(0, qd, lw.q_norm), (qd, kvd, lw.k_norm)] {
                e.pso("qd_rmsnorm_rows").s(0, io.p + off * BF).a(1, w.unwrap()).u(2, n as u32).f(3, eps)
                    .u(4, prow as u32).groups((t, 1), 1024);
            }
        }
        let flags = if lw.q_norm.is_some() && !d.qk_full { 1u32 } else { 0 };
        e.pso(&av.prep()).s(0, io.p);
        match (lw.q_norm, lw.k_norm, d.qk_full) {
            (Some(q), Some(k), false) => { e.a(1, q).a(2, k); }
            _ => { e.s(1, io.dummy).s(2, io.dummy); }
        }
        let invf = if win.is_some() { io.invf_local } else { io.invf };
        e.s(3, io.qr).a(4, kc).a(5, vc).s(6, io.dummy).s(7, io.dummy).s(8, invf).s(9, io.ctrl)
            .u(10, m.max_ctx as u32).f(11, eps).u(12, nkv as u32).u(13, d.rdims as u32).u(14, flags)
            .groups((nh + nkv, t), d.hd);
        let grp = nh / nkv;
        e.pso(&av.flash()).s(0, io.qr).a(1, kc).a(2, vc).s(3, io.p).s(4, io.ao).s(5, io.ctrl)
            .u(6, m.max_ctx as u32).u(7, grp as u32).u(8, prow as u32).u(9, t as u32).u(10, 0)
            .groups((t.div_ceil(8), nkv), 32 * grp);
        let post_attn = lw.post_attn_norm.map(|w| (w, lw.post_attn_norm_bias));
        let post_ffw = lw.post_ffw_norm.map(|w| (w, lw.post_ffw_norm_bias));
        match d.block {
            Block::Sequential => {
                self.residual(e, &lw.o, lw.o_bias, post_attn, t, io.ao);
                e.norm(d.norm, h, eps, t, io.h, lw.post_norm, lw.post_norm_bias, io.x);
                self.ffn(e, lw, t, io.x);
                self.residual(e, &lw.down, lw.down_bias, post_ffw, t, io.f);
            }
            Block::PostNorm => {
                self.residual(e, &lw.o, lw.o_bias, post_attn, t, io.ao);
                self.ffn(e, lw, t, io.h);
                self.residual(e, &lw.down, lw.down_bias, post_ffw, t, io.f);
            }
            Block::Parallel => {
                self.residual(e, &lw.o, lw.o_bias, None, t, io.ao);
                self.ffn(e, lw, t, io.x);
                self.residual(e, &lw.down, lw.down_bias, None, t, io.f);
            }
            Block::ParallelBranchesFirst => {
                e.gemm(&lw.o, t, 0, io.ao, io.r);
                if let Some(b) = lw.o_bias { e.bias_add(io.r, b, h, t); }
                self.ffn(e, lw, t, io.x);
                let n = t * h;
                match lw.down_bias {
                    None => e.gemm(&lw.down, t, 2, io.f, io.r),
                    Some(b) => {
                        e.gemm(&lw.down, t, 0, io.f, io.x);
                        e.bias_add(io.x, b, h, t);
                        e.pso("qd_residual").s(0, io.r).s(1, io.x).u(2, n as u32).f(3, 1.0).threads((n, 1), 256);
                    }
                }
                e.pso("qd_residual").s(0, io.h).s(1, io.r).u(2, n as u32).f(3, d.resid_mult).threads((n, 1), 256);
            }
        }
    }

    /// Decode-identical head on row t - 1: hf = final_norm(h), logits = lm_head hf (twin gemv K = 1, f32)
    /// [/ logits_scaling, soft cap], argmax.
    fn encode_head(&self, e: &PEnc, m: &PrefillModel, t: usize) {
        let (cfg, io, lm, d) = (m.cfg, &self.io, &m.w.lm_head, &self.dims);
        e.norm(d.norm, cfg.hidden, cfg.eps, 1, io.h + (t - 1) * cfg.hidden * BF, m.w.final_norm, m.w.final_norm_bias, io.hf);
        if m.w.tied && m.w.embed_bf16 {
            e.pso("qd_gemv_b16_f32").a(0, lm.w).s(1, io.hf).s(2, io.logits).u(3, lm.rows as u32).u(4, lm.cols as u32)
                .groups((lm.rows.div_ceil(8), 1), 256);
        } else {
            e.pso("qm_gemvt_k1_s8_m3")
                .a(0, lm.w).a(1, lm.s).a(2, lm.b).s(3, io.hf)
                .s(4, io.logits).u(5, lm.rows as u32).u(6, lm.cols as u32).s(7, io.logits).s(8, io.logits)
                .groups((lm.rows.div_ceil(8), 1), twin_nt_k1(lm.cols));
        }
        if d.logit_div != 1.0 || d.final_cap > 0.0 {
            e.pso("qd_logits_post").s(0, io.logits).u(1, d.vocab as u32).f(2, d.logit_div).f(3, d.final_cap)
                .threads((d.vocab, 1), 256);
        }
        e.pso("argmax_b").s(0, io.logits).s(1, io.hctrl).s(2, io.rec).u(3, lm.rows as u32).groups((1, 1), 1024);
    }
}
