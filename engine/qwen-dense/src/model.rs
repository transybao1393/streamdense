//! Kernel dispatch of one Qwen dense pass (K = 1..=8 token rows): embed, per layer RMSNorm -> fused q|k|v 4-bit gemv
//! (+ bias) -> attention (prep + core) -> o_proj (+= residual) -> RMSNorm -> SwiGLU -> down (+= residual), and the
//! head (final norm of the last row, lm_head gemv, argmax). Every buffer is an absolute arena offset (`gpu::Enc`).
//!
//! Matmuls: the engine's MMA 4-bit family (engine/shaders/gemv_mma_q4.metal), dispatched exactly as
//! streamdense-model `qwen_layers::gemv` / `swiglu` with `GemvImpl::Mma`: the scalar twins `qm_gemvt_k{K}_s8_m*` /
//! `qm_swiglut_k{K}_s8` where `mma_use_twin`, else `qm_gemv_n2_s8_m*` / `qm_swiglu_n1_s8`. Both are bit-identical per
//! row for the same SK (8), so a row's result does not depend on K: prefill (K-token passes) = token-by-token decode.
//! The helper functions `mma_use_twin` / `mma_twin_nt` are copies of the qwen_layers ones (same rule, same numbers).
//!
//! Attention: `AttnImpl` picks the core kernels after the shared prep (`attn128_prep_b16`: QK-norm flag, RoPE, KV
//! append): `SplitK` (attn128_partial_k{K}_b16 + attn128_reduce), `Flash` (attn128_flash), `Sdpa` (attn128_sdpa_vec,
//! the MLX decode mirror; rows past `sdpa128_vec_max_n` fall back to split-K) and `Auto` (default: passes of > 1 row
//! flash, 1-row passes sdpa, as MLX's prefill / decode split).
//!
//! Pass size: split-K / sdpa passes are 1..=8 rows (attn128_partial_k1..8). Flash / auto passes take up to
//! `PASS_MAX` rows; the matmuls of such a pass run in chunks of <= 8 rows (`gemv` / `swiglu` loop over row chunks).
//! The MMA family is bit-identical per row for any K, so chunking changes nothing numerically.
use crate::gpu::Enc;
use crate::weights::{LayerW, Q4, Weights};

pub const BF: usize = 2;
pub const F32: usize = 4;
pub const HD: usize = 128;
/// attn128_partial key chunk.
pub const CHUNK: usize = 256;
/// Max token rows of one matmul dispatch (MMA K <= 8) and of a split-K / sdpa pass (attn128_partial_k1..8).
pub const K_MAX: usize = 8;
/// Max token rows of one flash / auto pass (prefill); its matmuls run in chunks of K_MAX rows.
pub const PASS_MAX: usize = 128;
/// MLX 0.32.4's first kv length taking its 2-pass decode kernels on this GPU class (M4 Pro, arch "applegpu_g16s"):
/// `two_pass_min` [10] of attn128_sdpa_vec.
pub const SDPA128_TWO_PASS_MIN: usize = 1024;
/// Largest N attn128_sdpa_vec may take (`lim` [9]): 8192 (blocks <= 128), 8191 for grp 8 / 12 / 16 (MLX switches to
/// its unmirrored sdpa_vector_2pass_1_gqa at N >= 8192 there). Rows with N above it run split-K.
pub fn sdpa128_vec_max_n(grp: usize) -> usize { if grp == 8 || grp == 12 || grp == 16 { 8191 } else { 8192 } }

// ---- MMA 4-bit matmul (copies of streamdense-model qwen_layers constants / rules) ----
pub const MMA_SK: usize = 8;
pub const MMA_NB_GEMV: usize = 2;
pub const MMA_NB_SWIGLU: usize = 1;
pub const MMA_TWIN_K_MAX: usize = 4;

fn mma_twin_rp(_k: usize) -> usize { 4 }

/// Twin (no 8-row padding) or MMA kernel for `k` rows of a `cols`-wide matrix (qwen_layers::mma_use_twin).
pub fn mma_use_twin(cols: usize, swiglu: bool, k: usize) -> bool {
    if k <= 2 { return true; }
    if k > MMA_TWIN_K_MAX { return false; }
    let np = (if swiglu { 2 } else { 1 }) * 8 / mma_twin_rp(k);
    let cap = 512usize.min(4096 / (mma_twin_rp(k) * k));
    (cols / 64) * np <= cap
}

/// Threads per group of a twin (qwen_layers::mma_twin_nt).
pub fn mma_twin_nt(cols: usize, swiglu: bool, k: usize, sk: usize, tmax: usize) -> usize {
    let rp = mma_twin_rp(k);
    let nm = if swiglu { 2 } else { 1 };
    let np = nm * 8 / rp;
    let cap = tmax.min(512).min(4096 / (rp * k)).max(np);
    let g = cols / 64;
    let chunks = (g * np).div_ceil(cap);
    let nt = g.div_ceil(chunks) * np;
    nt.max((nm * 8 * sk).div_ceil(np) * np)
}

fn twin_gemv_name(k: usize, mode: u32) -> String { format!("qm_gemvt_k{}_s{}_m{}", k, MMA_SK, mode) }
fn twin_swiglu_name(k: usize) -> String { format!("qm_swiglut_k{}_s{}", k, MMA_SK) }
fn mma_gemv_name(mode: u32) -> String { format!("qm_gemv_n{}_s{}_m{}", MMA_NB_GEMV, MMA_SK, mode) }
fn mma_swiglu_name() -> String { format!("qm_swiglu_n{}_s{}", MMA_NB_SWIGLU, MMA_SK) }

/// k rows of x (bf16 [k][m.cols]) through `m`. MODE 0: out bf16 [k][m.rows]; 2: out += (bf16 residual, in place);
/// 3: out f32 [k][m.rows]. k > 8: dispatched in chunks of 8 rows (per-row results do not depend on the chunking).
pub fn gemv(e: &Enc, m: &Q4, k: usize, mode: u32, x: usize, out: usize) {
    let ob = if mode == 3 { F32 } else { BF };
    for c in (0..k).step_by(K_MAX) {
        gemv_k(e, m, (k - c).min(K_MAX), mode, x + c * m.cols * BF, out + c * m.rows * ob);
    }
}

fn gemv_k(e: &Enc, m: &Q4, k: usize, mode: u32, x: usize, out: usize) {
    assert!((1..=K_MAX).contains(&k), "gemv: k {} not in 1..=8", k);
    if mma_use_twin(m.cols, false, k) {
        let nt = mma_twin_nt(m.cols, false, k, MMA_SK, 512);
        e.pso(&twin_gemv_name(k, mode))
            .b(0, m.w).b(1, m.s).b(2, m.b).b(3, x)
            .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out)
            .groups((m.rows.div_ceil(8), 1), nt);
    } else {
        e.pso(&mma_gemv_name(mode))
            .b(0, m.w).b(1, m.s).b(2, m.b).b(3, x)
            .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out).u(9, k as u32)
            .groups((m.rows.div_ceil(8 * MMA_NB_GEMV), 1), 32 * MMA_SK);
    }
}

/// out bf16 [k][gate.rows] = silu(gate x) * (up x) (MLX bf16 swiglu rounding chain). k > 8: chunks of 8 rows.
pub fn swiglu(e: &Enc, gate: &Q4, up: &Q4, k: usize, x: usize, out: usize) {
    for c in (0..k).step_by(K_MAX) {
        swiglu_k(e, gate, up, (k - c).min(K_MAX), x + c * gate.cols * BF, out + c * gate.rows * BF);
    }
}

fn swiglu_k(e: &Enc, gate: &Q4, up: &Q4, k: usize, x: usize, out: usize) {
    assert!((1..=K_MAX).contains(&k), "swiglu: k {} not in 1..=8", k);
    if mma_use_twin(gate.cols, true, k) {
        let nt = mma_twin_nt(gate.cols, true, k, MMA_SK, 512);
        e.pso(&twin_swiglu_name(k))
            .b(0, gate.w).b(1, gate.s).b(2, gate.b).b(3, up.w).b(4, up.s).b(5, up.b)
            .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32)
            .groups((gate.rows.div_ceil(8), 1), nt);
    } else {
        e.pso(&mma_swiglu_name())
            .b(0, gate.w).b(1, gate.s).b(2, gate.b).b(3, up.w).b(4, up.s).b(5, up.b)
            .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32).u(10, k as u32)
            .groups((gate.rows.div_ceil(8 * MMA_NB_SWIGLU), 1), 32 * MMA_SK);
    }
}

/// k rows: out = rmsnorm(x) * w (engine norm_bf16.metal `rmsnorm_b`, MLX rounding).
pub fn rmsnorm(e: &Enc, n: usize, eps: f32, k: usize, x: usize, w: usize, out: usize) {
    e.pso("rmsnorm_b").b(0, x).b(1, w).b(2, out).u(3, n as u32).f(4, eps).u(5, n as u32).groups((k, 1), 1024);
}

/// Attention core implementation (after the shared prep).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttnImpl {
    /// Passes of > 1 row: `Flash`; 1-row passes (decode, and the last prompt token, as MLX's generate_step): `Sdpa`.
    #[default]
    Auto,
    /// attn128_partial_k{K}_b16 over ceil((pos + K) / 256) chunks + attn128_reduce (shaders/attn_hd128.metal).
    SplitK,
    /// attn128_flash (shaders/attn_hd128_flash.metal) for every pass, any row count up to PASS_MAX.
    Flash,
    /// attn128_sdpa_vec (shaders/attn_hd128_sdpa.metal, bit-identical to mx.fast.scaled_dot_product_attention at
    /// L = 1) for every row; rows with N > sdpa128_vec_max_n(grp) are written by split-K first.
    Sdpa,
}

impl AttnImpl {
    pub fn parse(s: &str) -> Result<AttnImpl, String> {
        match s {
            "auto" => Ok(AttnImpl::Auto),
            "splitk" => Ok(AttnImpl::SplitK),
            "flash" => Ok(AttnImpl::Flash),
            "sdpa" => Ok(AttnImpl::Sdpa),
            _ => Err(format!("attention impl {:?}: want auto | splitk | flash | sdpa", s)),
        }
    }
    pub fn name(self) -> &'static str {
        match self { AttnImpl::Auto => "auto", AttnImpl::SplitK => "splitk", AttnImpl::Flash => "flash", AttnImpl::Sdpa => "sdpa" }
    }
    /// The core a pass of `k` rows runs (Auto resolved).
    pub fn core(self, k: usize) -> AttnImpl {
        match self { AttnImpl::Auto => if k > 1 { AttnImpl::Flash } else { AttnImpl::Sdpa }, x => x }
    }
    /// Max rows of one pass.
    pub fn max_pass(self) -> usize {
        match self { AttnImpl::SplitK | AttnImpl::Sdpa => K_MAX, AttnImpl::Flash | AttnImpl::Auto => PASS_MAX }
    }
    /// Pipelines this impl's core dispatches (split-K is always built: the sdpa fallback for long rows).
    pub fn pipelines(self) -> Vec<String> {
        let mut v: Vec<String> = (1..=K_MAX).map(|k| format!("attn128_partial_k{}_b16", k)).collect();
        v.push("attn128_reduce".into());
        if matches!(self, AttnImpl::Flash | AttnImpl::Auto) { v.push("attn128_flash".into()); }
        if matches!(self, AttnImpl::Sdpa | AttnImpl::Auto) { v.push("attn128_sdpa_vec".into()); }
        v
    }
}

/// Buffers and dims of one attention layer pass (arena offsets).
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnIo {
    /// token rows of the pass (1..=8 split-K / sdpa, 1..=PASS_MAX flash) and the position of row 0 (host copy of ctrl.pos)
    pub k: usize,
    pub pos: usize,
    pub nh: usize,
    pub nkv: usize,
    pub rdims: usize,
    pub eps: f32,
    pub max_ctx: usize,
    /// p bf16 [k][nh*128 | nkv*128 | nkv*128] (q | k | v, after bias)
    pub p: usize,
    /// q_norm / k_norm bf16 [128]; None = no QK-norm (Qwen2)
    pub q_norm: Option<usize>,
    pub k_norm: Option<usize>,
    /// qr bf16 [k][nh][128] (normed + roped q)
    pub qr: usize,
    /// this layer's KV cache, bf16 [nkv][max_ctx][128] each
    pub kc: usize,
    pub vc: usize,
    /// any 256-B-aligned scratch to bind as Ksb / Vsb (unused with bf16 KV)
    pub dummy: usize,
    /// RoPE inverse frequencies f32 [rdims / 2]
    pub invf: usize,
    /// SdCtrl (ctrl.pos = pos)
    pub ctrl: usize,
    /// split-K partials: po f32 [k][nh][nch_max][128], pml f32x2 [k][nh][nch_max]
    pub po: usize,
    pub pml: usize,
    pub nch_max: usize,
    /// output bf16 [k][nh*128]
    pub ao: usize,
}

/// Shared prep: [QK-norm] + RoPE -> qr; K, V appended at pos + t.
pub fn encode_attn_prep(e: &Enc, a: &AttnIo) {
    let flags = if a.q_norm.is_some() { 1u32 } else { 0 };
    let qn = a.q_norm.unwrap_or(a.dummy);
    let kn = a.k_norm.unwrap_or(a.dummy);
    e.pso("attn128_prep_b16")
        .b(0, a.p).b(1, qn).b(2, kn).b(3, a.qr).b(4, a.kc).b(5, a.vc).b(6, a.dummy).b(7, a.dummy)
        .b(8, a.invf).b(9, a.ctrl).u(10, a.max_ctx as u32).f(11, a.eps).u(12, a.nkv as u32)
        .u(13, a.rdims as u32).u(14, flags)
        .groups((a.nh + a.nkv, a.k), HD);
}

fn encode_split_k(e: &Enc, a: &AttnIo) {
    assert!(a.k <= K_MAX, "split-K: pass of {} rows > {}", a.k, K_MAX);
    let nch = (a.pos + a.k).div_ceil(CHUNK);
    assert!(nch <= a.nch_max, "attention: {} chunks > nch_max {}", nch, a.nch_max);
    e.pso(&format!("attn128_partial_k{}_b16", a.k))
        .b(0, a.qr).b(1, a.kc).b(2, a.vc).b(3, a.dummy).b(4, a.dummy).b(5, a.po).b(6, a.pml).b(7, a.ctrl)
        .u(8, a.max_ctx as u32).u(9, (a.nh / a.nkv) as u32)
        .groups((nch, a.nh), 128);
    // partials are indexed with the dispatched nch, so reduce reads the same stride
    e.pso("attn128_reduce").b(0, a.po).b(1, a.pml).b(2, a.p).b(3, a.ao).u(4, nch as u32).u(5, 0)
        .groups((a.nh, a.k), HD / 4);
}

/// Attention core: qr + KV cache (keys 0 ..= pos + t for row t) -> ao.
pub fn encode_attn_core(e: &Enc, imp: AttnImpl, a: &AttnIo) {
    let grp = a.nh / a.nkv;
    let p_stride = (a.nh + 2 * a.nkv) * HD; // read only by the gated variants (flags bit 0, never set here)
    match imp.core(a.k) {
        AttnImpl::SplitK => encode_split_k(e, a),
        AttnImpl::Flash => {
            assert!(grp <= 32, "flash: grp {} > 32", grp);
            e.pso("attn128_flash")
                .b(0, a.qr).b(1, a.kc).b(2, a.vc).b(3, a.p).b(4, a.ao).b(5, a.ctrl)
                .u(6, a.max_ctx as u32).u(7, grp as u32).u(8, p_stride as u32).u(9, a.k as u32).u(10, 0)
                .groups((a.k.div_ceil(8), a.nkv), 32 * grp);
        }
        AttnImpl::Sdpa => {
            let lim = sdpa128_vec_max_n(grp);
            // rows with N = pos + t + 1 > lim: split-K writes every row first, sdpa_vec then overwrites rows <= lim
            if a.pos + a.k > lim { encode_split_k(e, a); }
            e.pso("attn128_sdpa_vec")
                .b(0, a.qr).b(1, a.kc).b(2, a.vc).b(3, a.p).b(4, a.ao).b(5, a.ctrl)
                .u(6, a.max_ctx as u32).u(7, grp as u32).u(8, p_stride as u32).u(9, lim as u32)
                .u(10, SDPA128_TWO_PASS_MIN as u32).u(11, 0)
                .groups((a.nh, a.k), 1024);
        }
        AttnImpl::Auto => unreachable!("core() resolves Auto"),
    }
}

/// Scratch of one pass (arena offsets), row t = 0..k-1.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassIo {
    pub k: usize,
    pub pos: usize,
    /// u32 [rows] token ids of the pass (rows = the engine's max pass)
    pub tok: usize,
    /// residual stream bf16 [rows][hidden]
    pub h: usize,
    /// normed input bf16 [rows][hidden]
    pub x: usize,
    pub p: usize,
    pub qr: usize,
    pub ao: usize,
    /// FFN activation bf16 [rows][inter]
    pub f: usize,
    pub po: usize,
    pub pml: usize,
    pub nch_max: usize,
    /// final-normed last row bf16 [hidden]
    pub hf: usize,
    /// f32 [vocab] (last row)
    pub logits: usize,
    /// argmax record float4 [max_ctx + 8] (argmax_b writes rec[pos + t])
    pub rec: usize,
    pub invf: usize,
    pub ctrl: usize,
    pub dummy: usize,
    pub max_ctx: usize,
}

/// Dims the pass needs (from DenseConfig).
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub hidden: usize,
    pub inter: usize,
    pub nh: usize,
    pub nkv: usize,
    pub rdims: usize,
    pub eps: f32,
    pub vocab: usize,
}

/// Per-layer KV cache offsets.
#[derive(Clone, Copy, Debug, Default)]
pub struct Kv {
    pub k: usize,
    pub v: usize,
}

/// RoPE inverse frequencies into `invf` (once per engine).
pub fn encode_rope_invf(e: &Enc, rdims: usize, theta: f64, invf: usize) {
    let n = rdims / 2;
    e.pso("attn128_rope_invf").b(0, invf).f(1, (theta as f32).log2()).threads((n, 1), n);
}

/// Embed the io.k tokens at io.tok into io.h.
pub fn encode_embed(e: &Enc, d: &Dims, w: &Weights, io: &PassIo) {
    e.pso("qd_embed_q4").b(0, w.embed.w).b(1, w.embed.s).b(2, w.embed.b).b(3, io.tok).b(4, io.h)
        .u(5, d.hidden as u32).threads((d.hidden / 8, io.k), 64);
}

/// One decoder layer for io.k rows (h in place).
pub fn encode_layer(e: &Enc, d: &Dims, lw: &LayerW, kv: &Kv, io: &PassIo, attn: AttnImpl) {
    let k = io.k;
    rmsnorm(e, d.hidden, d.eps, k, io.h, lw.in_norm, io.x);
    gemv(e, &lw.qkv, k, 0, io.x, io.p);
    if let Some(b) = lw.qkv_bias {
        let prow = lw.qkv.rows;
        e.pso("qd_bias_add").b(0, io.p).b(1, b).u(2, prow as u32).threads((prow, k), 256);
    }
    let a = AttnIo {
        k, pos: io.pos, nh: d.nh, nkv: d.nkv, rdims: d.rdims, eps: d.eps, max_ctx: io.max_ctx,
        p: io.p, q_norm: lw.q_norm, k_norm: lw.k_norm, qr: io.qr, kc: kv.k, vc: kv.v, dummy: io.dummy,
        invf: io.invf, ctrl: io.ctrl, po: io.po, pml: io.pml, nch_max: io.nch_max, ao: io.ao,
    };
    encode_attn_prep(e, &a);
    encode_attn_core(e, attn, &a);
    gemv(e, &lw.o, k, 2, io.ao, io.h);
    rmsnorm(e, d.hidden, d.eps, k, io.h, lw.post_norm, io.x);
    swiglu(e, &lw.gate, &lw.up, k, io.x, io.f);
    gemv(e, &lw.down, k, 2, io.f, io.h);
}

/// Head on the last row: hf = final_norm(h[k-1]); logits f32 = lm_head hf; argmax_b -> ctrl.tok[0].
pub fn encode_head(e: &Enc, d: &Dims, w: &Weights, io: &PassIo) {
    let last = io.h + (io.k - 1) * d.hidden * BF;
    rmsnorm(e, d.hidden, d.eps, 1, last, w.final_norm, io.hf);
    gemv(e, &w.lm_head, 1, 3, io.hf, io.logits);
    e.pso("argmax_b").b(0, io.logits).b(1, io.ctrl).b(2, io.rec).u(3, w.lm_head.rows as u32).groups((1, 1), 1024);
}

/// Pipelined decode, first dispatch of a 1-row pass: tok[0] = ctrl.tok[0] (the previous pass's greedy token, written
/// by its argmax_b) and ctrl.pos = io.pos, on the GPU (what `Engine::forward` writes from the host).
pub fn encode_next_tok(e: &Enc, io: &PassIo) {
    e.pso("qd_next_tok").b(0, io.ctrl).b(1, io.tok).u(2, io.pos as u32).threads((1, 1), 1);
}

/// Every pipeline a pass uses with attention impl `attn`.
pub fn pipelines(attn: AttnImpl) -> Vec<String> {
    let mut v: Vec<String> = vec![];
    for k in 1..=MMA_TWIN_K_MAX {
        for m in [0u32, 2, 3] { v.push(twin_gemv_name(k, m)); }
        v.push(twin_swiglu_name(k));
    }
    for m in [0u32, 2, 3] { v.push(mma_gemv_name(m)); }
    v.push(mma_swiglu_name());
    for n in ["rmsnorm_b", "argmax_b", "qd_embed_q4", "qd_bias_add", "qd_next_tok", "attn128_rope_invf", "attn128_prep_b16"] {
        v.push(n.into());
    }
    v.extend(attn.pipelines());
    v
}
