//! Kernel dispatch of one dense pass (K = 1..=8 token rows): embed (x embedding multiplier), per layer RMSNorm -> fused
//! q|k|v 4-bit gemv (+ bias) -> attention (prep + core; head dim 64 / 128 / 256, optional scale / soft cap / sliding
//! window as pipeline function constants) -> o_proj (+ bias, post-attention norm, residual multiplier) -> RMSNorm ->
//! SwiGLU or GeGLU (+ MLP biases) -> down (+ bias, post-feedforward norm, multiplier), and the head (final norm of the
//! last row, lm_head gemv, logits scaling / soft cap, argmax). Without those options (Qwen) the dispatches are exactly
//! the original ones: fused MODE 2 residual gemvs, fused SwiGLU, no extra kernels. Every buffer is an absolute arena offset (`gpu::Enc`).
//!
//! Other layer forms (config `norm` / `block` / `mlp` / `qk_norm_kind`, RoPE variants): LayerNorm (qd_layernorm, with
//! or without bias) wherever an RMSNorm stands; `Block::Parallel` (h += o; h += down, both branches from the one input
//! norm), `ParallelBranchesFirst` (r = o, r += down (MODE 2), h += r), `PostNorm` (attention / FFN read h, outputs
//! post-normed); `Mlp::Plain` (up gemv -> qd_act_b16 -> down); OLMo 2 q / k RMSNorm over the whole projection
//! (qd_rmsnorm_rows on p before the prep); per-layer prep variants (traditional / no RoPE, su multiplier, per-head
//! LayerNorm QK-norm) as function constants of attn{HD}_prep.
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
use crate::config::{Activation, Block, DenseConfig, LayerKind, Mlp, NormKind, QkNorm};
use crate::gpu::{fc_key, fc_key_prep, Enc};
use crate::weights::{LayerW, Q4, Weights};

pub const BF: usize = 2;
pub const F32: usize = 4;
/// Head dim of the original (Qwen) kernels; the engine takes 64 / 96 / 128 / 160 / 256 (`DenseConfig::head_dim`).
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

/// k rows: out = layer_norm(x) * w [+ b] (qd_layernorm, mx.fast.layer_norm rounding).
pub fn layernorm(e: &Enc, n: usize, eps: f32, k: usize, x: usize, w: usize, b: Option<usize>, out: usize) {
    e.pso("qd_layernorm").b(0, x).b(1, w).b(2, b.unwrap_or(w)).b(3, out).u(4, n as u32).f(5, eps)
        .u(6, b.is_some() as u32).groups((k, 1), 1024);
}

/// k rows of hidden size: the model's norm kind (RMSNorm as before, or LayerNorm with optional bias).
fn norm(e: &Enc, d: &Dims, k: usize, x: usize, w: usize, b: Option<usize>, out: usize) {
    match d.norm {
        NormKind::Rms => rmsnorm(e, d.hidden, d.eps, k, x, w, out),
        NormKind::LayerNorm { .. } => layernorm(e, d.hidden, d.eps, k, x, w, b, out),
    }
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
}

/// Attention kernel variant of one layer: head dim + the per-pipeline function constants (attn_hd128.metal).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AttnVariant {
    pub hd: usize,
    /// `None` = hd^-0.5 (the kernel default)
    pub scale: Option<f32>,
    pub softcap: Option<f32>,
    /// sliding window (sliding layers only)
    pub window: Option<u32>,
    /// prep only: RoPE mode bits (1 traditional, 2 no RoPE), su multiplier, QK-norm mode (1 per-head LayerNorm)
    pub rope: u32,
    pub mscale: Option<f32>,
    pub qkn: u32,
}

impl AttnVariant {
    fn key(&self, base: &str) -> String {
        fc_key(&format!("attn{}_{}", self.hd, base), self.scale, self.softcap, self.window)
    }
    pub fn prep(&self) -> String { fc_key_prep(&format!("attn{}_prep_b16", self.hd), self.rope, self.mscale, self.qkn) }
    pub fn partial(&self, k: usize) -> String { self.key(&format!("partial_k{}_b16", k)) }
    pub fn reduce(&self) -> String { format!("attn{}_reduce", self.hd) }
    pub fn flash(&self) -> String { self.key("flash") }
    pub fn sdpa(&self) -> String { self.key("sdpa_vec") }
    /// Pipelines of attention impl `imp` for this variant (split-K is always built: the sdpa fallback for long rows).
    pub fn pipelines(&self, imp: AttnImpl) -> Vec<String> {
        let mut v: Vec<String> = (1..=K_MAX).map(|k| self.partial(k)).collect();
        v.push(self.reduce());
        v.push(self.prep());
        if matches!(imp, AttnImpl::Flash | AttnImpl::Auto) { v.push(self.flash()); }
        if matches!(imp, AttnImpl::Sdpa | AttnImpl::Auto) { v.push(self.sdpa()); }
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
    /// kernels: head dim, scale, soft cap, window
    pub v: AttnVariant,
    /// p bf16 [k][nh*hd | nkv*hd | nkv*hd] (q | k | v, after bias)
    pub p: usize,
    /// q_norm / k_norm bf16 [hd]; None = no QK-norm
    pub q_norm: Option<usize>,
    pub k_norm: Option<usize>,
    /// qr bf16 [k][nh][hd] (normed + roped q)
    pub qr: usize,
    /// this layer's KV cache, bf16 [nkv][max_ctx][hd] each
    pub kc: usize,
    pub vc: usize,
    /// any 256-B-aligned scratch to bind as Ksb / Vsb (unused with bf16 KV)
    pub dummy: usize,
    /// RoPE inverse frequencies f32 [rdims / 2] (this layer's table)
    pub invf: usize,
    /// SdCtrl (ctrl.pos = pos)
    pub ctrl: usize,
    /// split-K partials: po f32 [k][nh][nch_max][hd], pml f32x2 [k][nh][nch_max]
    pub po: usize,
    pub pml: usize,
    pub nch_max: usize,
    /// output bf16 [k][nh*hd]
    pub ao: usize,
}

/// Shared prep: [QK-norm] + RoPE -> qr; K, V appended at pos + t.
pub fn encode_attn_prep(e: &Enc, a: &AttnIo) {
    let flags = if a.q_norm.is_some() { 1u32 } else { 0 };
    let qn = a.q_norm.unwrap_or(a.dummy);
    let kn = a.k_norm.unwrap_or(a.dummy);
    e.pso(&a.v.prep())
        .b(0, a.p).b(1, qn).b(2, kn).b(3, a.qr).b(4, a.kc).b(5, a.vc).b(6, a.dummy).b(7, a.dummy)
        .b(8, a.invf).b(9, a.ctrl).u(10, a.max_ctx as u32).f(11, a.eps).u(12, a.nkv as u32)
        .u(13, a.rdims as u32).u(14, flags)
        .groups((a.nh + a.nkv, a.k), a.v.hd);
}

fn encode_split_k(e: &Enc, a: &AttnIo) {
    assert!(a.k <= K_MAX, "split-K: pass of {} rows > {}", a.k, K_MAX);
    let nch = (a.pos + a.k).div_ceil(CHUNK);
    assert!(nch <= a.nch_max, "attention: {} chunks > nch_max {}", nch, a.nch_max);
    e.pso(&a.v.partial(a.k))
        .b(0, a.qr).b(1, a.kc).b(2, a.vc).b(3, a.dummy).b(4, a.dummy).b(5, a.po).b(6, a.pml).b(7, a.ctrl)
        .u(8, a.max_ctx as u32).u(9, (a.nh / a.nkv) as u32)
        .groups((nch, a.nh), 128);
    // partials are indexed with the dispatched nch, so reduce reads the same stride
    e.pso(&a.v.reduce()).b(0, a.po).b(1, a.pml).b(2, a.p).b(3, a.ao).u(4, nch as u32).u(5, 0)
        .groups((a.nh, a.k), a.v.hd / 4);
}

/// Attention core: qr + KV cache (keys visible to row t: 0 ..= pos + t, or the window) -> ao.
pub fn encode_attn_core(e: &Enc, imp: AttnImpl, a: &AttnIo) {
    let grp = a.nh / a.nkv;
    let p_stride = (a.nh + 2 * a.nkv) * a.v.hd; // read only by the gated variants (flags bit 0, never set here)
    match imp.core(a.k) {
        AttnImpl::SplitK => encode_split_k(e, a),
        AttnImpl::Flash => {
            assert!(grp <= 32, "flash: grp {} > 32", grp);
            e.pso(&a.v.flash())
                .b(0, a.qr).b(1, a.kc).b(2, a.vc).b(3, a.p).b(4, a.ao).b(5, a.ctrl)
                .u(6, a.max_ctx as u32).u(7, grp as u32).u(8, p_stride as u32).u(9, a.k as u32).u(10, 0)
                .groups((a.k.div_ceil(8), a.nkv), 32 * grp);
        }
        AttnImpl::Sdpa => {
            let lim = sdpa128_vec_max_n(grp);
            // rows with N = pos + t + 1 > lim: split-K writes every row first, sdpa_vec then overwrites rows <= lim
            if a.pos + a.k > lim { encode_split_k(e, a); }
            e.pso(&a.v.sdpa())
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
    /// up projection bf16 [rows][inter] (unfused FFN: GeGLU or MLP bias; else unused)
    pub f2: usize,
    /// residual branch bf16 [rows][hidden] (unfused residual: o / down bias, post-norm, residual multiplier)
    pub r: usize,
    pub po: usize,
    pub pml: usize,
    pub nch_max: usize,
    /// final-normed last row bf16 [hidden]
    pub hf: usize,
    /// f32 [vocab] (last row)
    pub logits: usize,
    /// argmax record float4 [max_ctx + 8] (argmax_b writes rec[pos + t])
    pub rec: usize,
    /// RoPE tables f32 [rdims / 2]: full-attention layers, sliding layers (= invf unless `rope_local_theta`)
    pub invf: usize,
    pub invf_local: usize,
    pub ctrl: usize,
    pub dummy: usize,
    pub max_ctx: usize,
}

/// Dims and per-model math switches the pass needs (from DenseConfig, see `Dims::new`).
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub hidden: usize,
    pub inter: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub rdims: usize,
    pub eps: f32,
    pub vocab: usize,
    /// qd_glu act code: 0 SiLU, 1 tanh-GELU
    pub act: u32,
    /// FFN goes through gate / up gemvs + qd_glu (GeGLU or MLP bias) instead of the fused SwiGLU
    pub ffn_unfused: bool,
    /// bf16-rounded embedding multiplier (1 = none)
    pub embed_scale: f32,
    /// bf16-rounded residual multiplier (1 = none)
    pub resid_mult: f32,
    /// logits divisor (1 = none) and final soft cap (0 = none)
    pub logit_div: f32,
    pub final_cap: f32,
    /// attention scale (None = hd^-0.5) and score soft cap
    pub scale: Option<f32>,
    pub softcap: Option<f32>,
    pub norm: NormKind,
    pub block: Block,
    /// `Mlp::Plain`: up -> act -> down
    pub mlp_plain: bool,
    /// q_norm / k_norm over the whole projection (qd_rmsnorm_rows), not in the prep
    pub qk_full: bool,
    /// prep constants: traditional RoPE bit, su multiplier (bf16-rounded), QK-norm mode
    pub rope_trad: bool,
    pub mscale: Option<f32>,
    pub qkn: u32,
}

fn rbf(x: f32) -> f32 { streamdense_core::bf16_to_f32(streamdense_core::f32_to_bf16(x)) }

impl Dims {
    pub fn new(cfg: &DenseConfig) -> Dims {
        let act = match cfg.activation { Activation::GeluTanh => 1, Activation::GeluErf => 2, _ => 0 };
        Dims {
            hidden: cfg.hidden, inter: cfg.inter, nh: cfg.heads, nkv: cfg.kv_heads, hd: cfg.head_dim, rdims: cfg.rope_dims,
            eps: cfg.eps, vocab: cfg.vocab, act, ffn_unfused: act != 0 || cfg.mlp_bias,
            embed_scale: rbf(cfg.embed_scale as f32), resid_mult: rbf(cfg.residual_multiplier as f32),
            logit_div: cfg.logits_scaling as f32, final_cap: cfg.final_softcap.unwrap_or(0.0) as f32,
            scale: cfg.attn_scale.map(|s| s as f32), softcap: cfg.attn_softcap.map(|s| s as f32),
            norm: cfg.norm, block: cfg.block, mlp_plain: cfg.mlp == Mlp::Plain,
            qk_full: cfg.qk_norm && cfg.qk_norm_kind == QkNorm::FullRms, rope_trad: cfg.rope_traditional,
            // mlx_lm SuScaledRoPE multiplies the bf16 q / k by a weakly typed scalar: the scale is rounded to bf16
            mscale: cfg.rope_su.as_ref().map(|s| rbf(s.mscale as f32)),
            qkn: if cfg.qk_norm && cfg.qk_norm_kind == QkNorm::HeadLayerNorm { 1 } else { 0 },
        }
    }
    /// Attention kernels of a layer with sliding window `window` (None = full attention), RoPE on or off.
    pub fn variant(&self, window: Option<u32>, rope: bool) -> AttnVariant {
        let mode = if !rope { 2 } else if self.rope_trad { 1 } else { 0 };
        AttnVariant { hd: self.hd, scale: self.scale, softcap: self.softcap, window, rope: mode,
                      mscale: if rope { self.mscale } else { None }, qkn: self.qkn }
    }
}

/// Per-layer window of `cfg` (`Some(sliding_window)` on sliding layers).
pub fn layer_windows(cfg: &DenseConfig) -> Vec<Option<u32>> {
    cfg.kinds.iter().map(|k| match k {
        LayerKind::Sliding => cfg.sliding_window.map(|w| w as u32),
        LayerKind::Full => None,
    }).collect()
}

/// Per-layer RoPE switch of `cfg` (false on `no_rope_layers`).
pub fn layer_rope(cfg: &DenseConfig) -> Vec<bool> { (0..cfg.layers).map(|l| cfg.rope_on(l)).collect() }

/// Per-layer KV cache offsets.
#[derive(Clone, Copy, Debug, Default)]
pub struct Kv {
    pub k: usize,
    pub v: usize,
}

/// RoPE inverse frequencies into `invf` (once per engine): plain RoPE of `theta` with the kernel (bit-identical to
/// before), any implemented scaling from the host (`config::rope_inv_freq`).
pub fn encode_rope_invf(e: &Enc, rdims: usize, theta: f64, invf: usize) {
    let n = rdims / 2;
    e.pso("attn128_rope_invf").b(0, invf).f(1, (theta as f32).log2()).threads((n, 1), n);
}

/// Embed the io.k tokens at io.tok into io.h (times the embedding multiplier, when set); 4-bit or bf16 embedding.
pub fn encode_embed(e: &Enc, d: &Dims, w: &Weights, io: &PassIo) {
    if w.embed_bf16 {
        e.pso("qd_embed_b16").b(0, w.embed.w).b(1, io.tok).b(2, io.h).u(3, d.hidden as u32).threads((d.hidden, io.k), 64);
    } else {
        e.pso("qd_embed_q4").b(0, w.embed.w).b(1, w.embed.s).b(2, w.embed.b).b(3, io.tok).b(4, io.h)
            .u(5, d.hidden as u32).threads((d.hidden / 8, io.k), 64);
    }
    if d.embed_scale != 1.0 {
        let n = io.k * d.hidden;
        e.pso("qd_scale_b16").b(0, io.h).u(1, n as u32).f(2, d.embed_scale).threads((n, 1), 256);
    }
}

/// p bf16 [k][n] += bias [n].
fn bias_add(e: &Enc, p: usize, b: usize, n: usize, k: usize) {
    e.pso("qd_bias_add").b(0, p).b(1, b).u(2, n as u32).threads((n, k), 256);
}

/// h += m * post(m x + bias): the fused MODE 2 gemv when there is no bias / post-norm / multiplier, else MODE 0 into
/// io.r, bias, [norm -> io.x], qd_residual. `post` = (weight, LayerNorm bias).
fn residual(e: &Enc, d: &Dims, m: &Q4, bias: Option<usize>, post: Option<(usize, Option<usize>)>, k: usize, x: usize, io: &PassIo) {
    if bias.is_none() && post.is_none() && d.resid_mult == 1.0 {
        gemv(e, m, k, 2, x, io.h);
        return;
    }
    gemv(e, m, k, 0, x, io.r);
    if let Some(b) = bias { bias_add(e, io.r, b, d.hidden, k); }
    let r = match post { Some((w, b)) => { norm(e, d, k, io.r, w, b, io.x); io.x } None => io.r };
    let n = k * d.hidden;
    e.pso("qd_residual").b(0, io.h).b(1, r).u(2, n as u32).f(3, d.resid_mult).threads((n, 1), 256);
}

/// io.f = act(gate x + bias) * (up x + bias): fused SwiGLU, or gate / up gemvs + biases + qd_glu; `Mlp::Plain`:
/// io.f = act(up x + bias) (qd_act_b16). x = the FFN input rows.
fn ffn(e: &Enc, d: &Dims, lw: &LayerW, k: usize, x: usize, io: &PassIo) {
    let n = k * d.inter;
    if d.mlp_plain {
        gemv(e, &lw.up, k, 0, x, io.f);
        if let Some(b) = lw.up_bias { bias_add(e, io.f, b, d.inter, k); }
        e.pso("qd_act_b16").b(0, io.f).u(1, n as u32).u(2, d.act).threads((n, 1), 256);
        return;
    }
    if !d.ffn_unfused {
        swiglu(e, &lw.gate, &lw.up, k, x, io.f);
        return;
    }
    gemv(e, &lw.gate, k, 0, x, io.f);
    gemv(e, &lw.up, k, 0, x, io.f2);
    if let Some(b) = lw.gate_bias { bias_add(e, io.f, b, d.inter, k); }
    if let Some(b) = lw.up_bias { bias_add(e, io.f2, b, d.inter, k); }
    e.pso("qd_glu").b(0, io.f).b(1, io.f2).b(2, io.f).u(3, n as u32).u(4, d.act).threads((n, 1), 256);
}

/// One decoder layer for io.k rows (h in place); `window` = this layer's sliding window (None: full attention),
/// `rope` = this layer applies RoPE.
pub fn encode_layer(e: &Enc, d: &Dims, lw: &LayerW, kv: &Kv, io: &PassIo, attn: AttnImpl, window: Option<u32>, rope: bool) {
    let k = io.k;
    // attention input: input norm, or h itself (PostNorm)
    let ain = if d.block == Block::PostNorm { io.h } else { norm(e, d, k, io.h, lw.in_norm, lw.in_norm_bias, io.x); io.x };
    gemv(e, &lw.qkv, k, 0, ain, io.p);
    if let Some(b) = lw.qkv_bias { bias_add(e, io.p, b, lw.qkv.rows, k); }
    let (qd, kvd) = (d.nh * d.hd, d.nkv * d.hd);
    if d.qk_full {
        // OLMo 2: RMSNorm over the whole q and k projections, in place in p (the prep then runs without QK-norm)
        for (off, n, w) in [(0, qd, lw.q_norm), (qd, kvd, lw.k_norm)] {
            e.pso("qd_rmsnorm_rows").b(0, io.p + off * BF).b(1, w.unwrap()).u(2, n as u32).f(3, d.eps)
                .u(4, lw.qkv.rows as u32).groups((k, 1), 1024);
        }
    }
    let (q_norm, k_norm) = if d.qk_full { (None, None) } else { (lw.q_norm, lw.k_norm) };
    let a = AttnIo {
        k, pos: io.pos, nh: d.nh, nkv: d.nkv, rdims: d.rdims, eps: d.eps, max_ctx: io.max_ctx, v: d.variant(window, rope),
        p: io.p, q_norm, k_norm, qr: io.qr, kc: kv.k, vc: kv.v, dummy: io.dummy,
        invf: if window.is_some() { io.invf_local } else { io.invf }, ctrl: io.ctrl, po: io.po, pml: io.pml,
        nch_max: io.nch_max, ao: io.ao,
    };
    encode_attn_prep(e, &a);
    encode_attn_core(e, attn, &a);
    let post_attn = lw.post_attn_norm.map(|w| (w, lw.post_attn_norm_bias));
    let post_ffw = lw.post_ffw_norm.map(|w| (w, lw.post_ffw_norm_bias));
    match d.block {
        Block::Sequential => {
            residual(e, d, &lw.o, lw.o_bias, post_attn, k, io.ao, io);
            norm(e, d, k, io.h, lw.post_norm, lw.post_norm_bias, io.x);
            ffn(e, d, lw, k, io.x, io);
            residual(e, d, &lw.down, lw.down_bias, post_ffw, k, io.f, io);
        }
        Block::PostNorm => {
            residual(e, d, &lw.o, lw.o_bias, post_attn, k, io.ao, io);
            ffn(e, d, lw, k, io.h, io);
            residual(e, d, &lw.down, lw.down_bias, post_ffw, k, io.f, io);
        }
        Block::Parallel => {
            // h = (h + attn) + mlp; io.x (the normed input) is not touched by a residual without post-norm
            residual(e, d, &lw.o, lw.o_bias, None, k, io.ao, io);
            ffn(e, d, lw, k, io.x, io);
            residual(e, d, &lw.down, lw.down_bias, None, k, io.f, io);
        }
        Block::ParallelBranchesFirst => {
            // r = bf(attn [+ bias]); r = bf(r + bf(mlp [+ bias])); h = bf(h + bf(r * m))
            gemv(e, &lw.o, k, 0, io.ao, io.r);
            if let Some(b) = lw.o_bias { bias_add(e, io.r, b, d.hidden, k); }
            ffn(e, d, lw, k, io.x, io);
            let n = k * d.hidden;
            match lw.down_bias {
                None => gemv(e, &lw.down, k, 2, io.f, io.r),
                Some(b) => {
                    gemv(e, &lw.down, k, 0, io.f, io.x);
                    bias_add(e, io.x, b, d.hidden, k);
                    e.pso("qd_residual").b(0, io.r).b(1, io.x).u(2, n as u32).f(3, 1.0).threads((n, 1), 256);
                }
            }
            e.pso("qd_residual").b(0, io.h).b(1, io.r).u(2, n as u32).f(3, d.resid_mult).threads((n, 1), 256);
        }
    }
}

/// Head on the last row: hf = final_norm(h[k-1]); logits f32 = lm_head hf [/ logits_scaling, soft cap];
/// argmax_b -> ctrl.tok[0]. lm_head: 4-bit gemv, or the dense bf16 gemv when tied to a bf16 embedding.
pub fn encode_head(e: &Enc, d: &Dims, w: &Weights, io: &PassIo) {
    let last = io.h + (io.k - 1) * d.hidden * BF;
    norm(e, d, 1, last, w.final_norm, w.final_norm_bias, io.hf);
    if w.tied && w.embed_bf16 {
        let m = &w.lm_head;
        e.pso("qd_gemv_b16_f32").b(0, m.w).b(1, io.hf).b(2, io.logits).u(3, m.rows as u32).u(4, m.cols as u32)
            .groups((m.rows.div_ceil(8), 1), 256);
    } else {
        gemv(e, &w.lm_head, 1, 3, io.hf, io.logits);
    }
    encode_logits_post(e, d, io.logits);
    e.pso("argmax_b").b(0, io.logits).b(1, io.ctrl).b(2, io.rec).u(3, w.lm_head.rows as u32).groups((1, 1), 1024);
}

/// logits f32 [vocab] / logits_scaling, then the final soft cap (nothing dispatched when neither is set).
pub fn encode_logits_post(e: &Enc, d: &Dims, logits: usize) {
    if d.logit_div != 1.0 || d.final_cap > 0.0 {
        e.pso("qd_logits_post").b(0, logits).u(1, d.vocab as u32).f(2, d.logit_div).f(3, d.final_cap)
            .threads((d.vocab, 1), 256);
    }
}

/// Pipelined decode, first dispatch of a 1-row pass: tok[0] = ctrl.tok[0] (the previous pass's greedy token, written
/// by its argmax_b) and ctrl.pos = io.pos, on the GPU (what `Engine::forward` writes from the host).
pub fn encode_next_tok(e: &Enc, io: &PassIo) {
    e.pso("qd_next_tok").b(0, io.ctrl).b(1, io.tok).u(2, io.pos as u32).threads((1, 1), 1);
}

/// Every pipeline a pass uses with attention impl `attn`, for the model's dims and per-layer windows / RoPE switches.
pub fn pipelines(attn: AttnImpl, d: &Dims, windows: &[Option<u32>], rope: &[bool]) -> Vec<String> {
    let mut v: Vec<String> = vec![];
    for k in 1..=MMA_TWIN_K_MAX {
        for m in [0u32, 2, 3] { v.push(twin_gemv_name(k, m)); }
        v.push(twin_swiglu_name(k));
    }
    for m in [0u32, 2, 3] { v.push(mma_gemv_name(m)); }
    v.push(mma_swiglu_name());
    for n in ["rmsnorm_b", "argmax_b", "qd_embed_q4", "qd_bias_add", "qd_next_tok", "attn128_rope_invf", "qd_scale_b16",
              "qd_residual", "qd_glu", "qd_logits_post", "qd_embed_b16", "qd_gemv_b16_f32", "qd_layernorm",
              "qd_rmsnorm_rows", "qd_act_b16"] {
        v.push(n.into());
    }
    let mut seen: Vec<(Option<u32>, bool)> = vec![];
    for (l, w) in windows.iter().enumerate() {
        let r = rope.get(l).copied().unwrap_or(true);
        if !seen.contains(&(*w, r)) { seen.push((*w, r)); v.extend(d.variant(*w, r).pipelines(attn)); }
    }
    if windows.is_empty() { v.extend(d.variant(None, true).pipelines(attn)); }
    v.sort();
    v.dedup();
    v
}
