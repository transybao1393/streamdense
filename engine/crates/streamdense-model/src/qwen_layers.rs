//! Qwen3.5/3.8 layer encoders (plan 6.1): one decoder layer of either kind (Gated DeltaNet or gated full
//! attention) for K = 1..=8 tokens, and the head (final norm, lm_head, argmax). Every buffer is an absolute arena
//! offset; every dim comes from `QwenConfig` or the weight shapes. Positions, the state role and the kv length live
//! in the device `SdCtrl` block, so nothing here depends on the current position (a pass can be encoded early).
//!
//! Buffer bindings (kernels as built):
//! - `q4b_gemv_k{K}[_r{D}]_m{0,2,3}`: W, S, B, x, out, R, C, res, outf; ceil(R/2D) groups x 64 (every index bound;
//!   D = row-dots per simdgroup, 2 without `_r`; see `gemv_d`).
//! - `q4b_swiglu_k{K}[_r{D}]`: Wg/Sg/Bg, Wu/Su/Bu, x, out, R, C; ceil(R/D) groups x 64 (D = 2 without `_r`).
//! - `rmsnorm_b`: x, w, out, n, eps, out_stride; T groups x 1024. `rmsnorm_gated`: y, z, z_stride, w, out, eps;
//!   (vh, T) groups x dv.
//! - `dn_conv` / `dn_prep` / `dn_recur_ck{0,1}` with `DnDims` by value; `attn_prep_*`, `attn_partial_k*_*`,
//!   `attn_split_k*`, `attn_reduce_gate`; `argmax_b`.
//! - `GemvImpl::Mma` (default): `qm_gemvt_k{K}_s8_m*` / `qm_swiglut_k{K}_s8` (`mma_use_twin`) or
//!   `qm_gemv_n2_s8_m*` / `qm_swiglu_n1_s8` (gemv_mma_q4.metal; bindings at `mma_gemv_enc` / `mma_twin_gemv_enc`).
use streamdense_core::DType;
use streamdense_metal::Enc;

use crate::dump::Dumper;
use crate::qwen_config::{LayerKind, QwenConfig};
use crate::weights::{AttnW, DnW, LayerW, Mixer, Q4b, Weights};

/// DeltaNet dims as the `deltanet.metal` kernels take them (`constant DnDims&`), 32 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DnDims {
    pub kh: u32,
    pub vh: u32,
    /// conv channels = q | k | v = conv_dim()
    pub conv: u32,
    /// p row stride = dn_in_rows()
    pub prow: u32,
    /// in_proj_b column inside a p row = conv + vh * dv
    pub b_off: u32,
    /// in_proj_a column inside a p row = b_off + vh
    pub a_off: u32,
    /// DeltaNet layers in the conv state buffer = n_deltanet()
    pub layers: u32,
    pub pad: u32,
}

impl DnDims {
    pub fn from_cfg(cfg: &QwenConfig) -> DnDims {
        let conv = cfg.conv_dim();
        let b_off = conv + cfg.dn_v_dim();
        DnDims {
            kh: cfg.dn_kh as u32,
            vh: cfg.dn_vh as u32,
            conv: conv as u32,
            prow: cfg.dn_in_rows() as u32,
            b_off: b_off as u32,
            a_off: (b_off + cfg.dn_vh) as u32,
            layers: cfg.n_deltanet() as u32,
            pad: 0,
        }
    }
}

/// Scratch of one pass (arena offsets, plan 6.4), rows t = 0..k-1. Row strides: h, x, hf [hidden]; p [max(dn_in_rows,
/// attn_in_rows)] (each layer kind uses its own row width); cv [conv_dim]; y, yn [dn_v_dim]; ao [q_dim]; f [inter];
/// g f32 [vh]; beta bf16 [vh]; po f32 [k][heads][nch_max][head_dim]; pml f32x2 [k][heads][nch_max];
/// logits f32 [k][vocab]; rec f32x4 [max_ctx]; invf f32 [rope_dims / 2]; ctrl = the `SdCtrl` block.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassIo {
    pub k: usize,
    pub h: usize,
    pub x: usize,
    pub p: usize,
    pub cv: usize,
    pub g: usize,
    pub beta: usize,
    pub y: usize,
    pub yn: usize,
    pub ao: usize,
    pub f: usize,
    pub po: usize,
    pub pml: usize,
    pub hf: usize,
    pub ctrl: usize,
    /// attention chunks dispatched (each 256 keys); must cover ctrl.pos + k. Empty chunks add nothing.
    pub nch_max: usize,
    /// host copy of ctrl.pos for this pass (exact: passes are encoded in execution order). Only picks the attention
    /// path (`SDPA_VEC_MAX_N`); the kernels still read the position from ctrl.
    pub pos: usize,
    pub logits: usize,
    pub rec: usize,
    pub invf: usize,
    /// deep spec verify (spec_deep.rs): arena offset of the DeltaNet replay save, bf16 [k-2][n_deltanet][conv + 2*vh];
    /// encode_dn copies rows 1..k-2 of [qkv | b | a] there (k > 2). None = no save.
    pub dn_save: Option<usize>,
    /// context-draft verify (ctx_draft.rs): arena offset of the hold save, bf16 [k][n_deltanet][conv + 2*vh]. Some: encode_dn
    /// saves ALL k rows of [qkv | b | a] there and runs the DeltaNet without writing the state back (dn_conv ck =
    /// DN_HOLD, dn_recur_hold); the accepted rows are replayed afterwards. None = today.
    pub dn_hold: Option<usize>,
    /// matmul family of the exact pass (decode, verify, MTP head, exact prefill chunks); the engine copies it from
    /// `QwenOptions.gemv` into every pass.
    pub gemv: GemvImpl,
    /// activity recorder offsets (activity.rs); None = off. Only model layers record (not the MTP layer).
    pub act: Option<crate::activity::ActIo>,
    /// read mask (readmask.rs): None = no region (today's code). Some: the `rm_seen` recorder runs after every
    /// model layer; with `apply` the exact pass uses the masked Mma kernels (never the MTP layer, never fast prefill).
    pub rm: Option<crate::readmask::RmIo>,
    /// FFN-select test hook (ffn_select.rs): None = off (no dispatch). Some: per-neuron FFN selection after every model-layer swiglu.
    pub fsel: Option<crate::ffn_select::FselIo>,
    /// forheader compact FFN (ffn_compact.rs): arena offset of the per-group f sums f32 [k][inter/64]; 0 = off.
    pub cfs: usize,
    /// fused K-row verify matmuls (verify_fused_q4.metal, `QwenOptions.verify_fused`): arena offsets of the `vf_xprep`
    /// outputs (xt f32 [cols][8], st f32 [cols / 64][8], sized by `vf_prep_bytes(8, widest input)`). Some: unmasked
    /// Mma-family passes of 3..=8 rows run `vf_gemv` / `vf_swiglu` (bit-identical to the Mma / MmaD kernels, see the
    /// shader header). None = today.
    pub vf: Option<(usize, usize)>,
    /// FFN-input capture (`QwenOptions.ffn_x_capture`): arena offset of bf16 [layers][hidden]; Some: every K = 1
    /// model-layer pass copies its FFN input (post-norm x) to row l. None = no dispatch (default).
    pub fxcap: Option<usize>,
    /// K-row verify mixers (verify_mixers.metal, `QwenOptions.verify_mixers`). Some(sc): DeltaNet passes of 2..=8 rows
    /// run dn_recur_v, and (sc = arena offset of `va_scratch_bytes(cfg, 8)`, VM_NO_ATTN = none: q8 KV) bf16-KV sdpa rows of
    /// passes of <= 8 rows run va_sdpa_p1/p3/p2 instead of attn_sdpa_vec_b16. Bit-identical per row. None = today.
    pub vm: Option<usize>,
    /// one-row passes without `vm`: arena offset of `va_scratch_bytes(cfg, 1)` (bf16 KV). Some: K = 1 sdpa rows with
    /// N <= lim run va_sdpa (R = 1) instead of attn_sdpa_vec_b16 (bit-identical per row; 27B shapes, one layer: 4096
    /// keys 173 -> 120 µs, 8000 343 -> 210). None = attn_sdpa_vec_b16.
    pub va1: Option<usize>,
    /// banned ids (gateway `format: plan`): arena offset of u32 [h, id0 .. id222] (bit 31 of h: allow-list mode); Some (h > 0): `ban_logits` sets those
    /// logits to -inf before every head argmax (decode, MTP / deep / DFlash / context-draft verify). None = today.
    pub ban: Option<usize>,
}

/// DeltaNet state of the whole model (arena offsets). The kernels pick the live copy by `ctrl.role`.
/// - `s`: f32 [n_deltanet][copies][vh][dv][dk] (per DeltaNet layer, the copies back to back);
/// - `conv`: bf16 [copies][n_deltanet][conv_k - 1][conv_dim].
/// `copies` = 2 with MTP (rollback checkpoint, plan 7.2), 1 in plain mode: ctrl.role stays 0 and no pass sets `ck`,
/// so the kernels never touch copy 1 (they compute its address but only write it under a checkpoint).
#[derive(Clone, Copy, Debug, Default)]
pub struct DnState {
    pub s: usize,
    pub conv: usize,
    pub copies: usize,
}

impl DnState {
    /// Bytes of one DeltaNet layer's S (all copies).
    pub fn s_layer_bytes(cfg: &QwenConfig, copies: usize) -> usize { copies * cfg.dn_vh * cfg.dn_dv * cfg.dn_dk * 4 }
    pub fn s_bytes(cfg: &QwenConfig, copies: usize) -> usize { cfg.n_deltanet() * Self::s_layer_bytes(cfg, copies) }
    /// Bytes of one conv state copy for one layer.
    pub fn conv_layer_bytes(cfg: &QwenConfig) -> usize { (cfg.conv_k - 1) * cfg.conv_dim() * 2 }
    pub fn conv_bytes(cfg: &QwenConfig, copies: usize) -> usize { copies * cfg.n_deltanet() * Self::conv_layer_bytes(cfg) }
}

/// KV cache of ONE attention layer (arena offsets): bf16 [kv_heads][max_ctx][head_dim] (or u8 when `q8`), plus
/// the q8 scale/min half2 [kv_heads][max_ctx][head_dim / 32] (bound but unused for bf16).
#[derive(Clone, Copy, Debug, Default)]
pub struct KvRef {
    pub k: usize,
    pub v: usize,
    pub ksb: usize,
    pub vsb: usize,
    pub q8: bool,
    pub max_ctx: usize,
}

const BF: usize = 2;
const F32: usize = 4;

/// Largest kv length N handled by `attn_sdpa_vec_b16`, the bit-exact mirror of MLX's decode attention (L = 1):
/// single-pass `sdpa_vector` below `SDPA_TWO_PASS_MIN`, the fused 2-pass mirror from there up to this bound (its
/// `blocks` <= 128). Rows with larger N use the split-K chunk path (`attn_split_rows` + `attn_reduce_gate`).
/// `SD_SDPA_VEC_MAX` overrides it (0 = always split-K).
/// dn_conv `ck` value of a held pass (deltanet.metal DN_HOLD): no checkpoint, conv state not written back.
pub const DN_HOLD: u32 = 0xFFFF_FFFE;
pub const SDPA_VEC_MAX_N: usize = 8192;
/// First kv length at which MLX 0.32.4 switches to its 2-pass kernels (measured: tiny model layer 3,
/// every row N <= 1023 bit-exact with the single-pass mirror, every row N >= 1024 different).
/// `SD_SDPA_TWO_PASS_MIN` overrides it.
pub const SDPA_TWO_PASS_MIN: usize = 1024;

pub fn sdpa_two_pass_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("SD_SDPA_TWO_PASS_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(SDPA_TWO_PASS_MIN))
}

pub fn sdpa_vec_max_n() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("SD_SDPA_VEC_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(SDPA_VEC_MAX_N))
}

/// Stats only (no effect on any dispatch): verify passes whose rows straddle an attention path boundary inside one
/// dispatch. A pass of `k` rows at host position `pos` has row N from pos + 1 to pos + k (`attn_sdpa_rows`).
/// `two_pass`: bf16 KV, rows on both sides of `sdpa_two_pass_min()` (pos + 1 < B <= pos + k).
/// `split_k`: bf16 KV, rows on both sides of the split-K limit lim = `sdpa_vec_max_n().min(8192)` (pos + 1 <= lim < pos + k).
/// `chunk`: the pass's split-K rows (every row with q8 KV; rows with N > lim with bf16) end in different 256-key
/// chunks (`ATTN_CHUNK`), i.e. their last keys N - 1 are not in the same chunk. q8 KV: `two_pass` / `split_k` stay 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttnStraddle { pub passes: usize, pub two_pass: usize, pub split_k: usize, pub chunk: usize }
impl AttnStraddle {
    pub fn note(&mut self, pos: usize, k: usize, q8: bool) {
        if k == 0 { return; }
        self.passes += 1;
        let (lo, hi) = (pos + 1, pos + k);
        let lim = if q8 { 0 } else {
            let b = sdpa_two_pass_min();
            if lo < b && hi >= b { self.two_pass += 1; }
            let lim = sdpa_vec_max_n().min(8192);
            if lo <= lim && hi > lim { self.split_k += 1; }
            lim
        };
        let s = lo.max(lim + 1); // first split-K row's N
        let c = crate::qwen_engine::ATTN_CHUNK;
        if s <= hi && (s - 1) / c != (hi - 1) / c { self.chunk += 1; }
    }
    pub fn add(&mut self, o: &AttnStraddle) {
        self.passes += o.passes; self.two_pass += o.two_pass; self.split_k += o.split_k; self.chunk += o.chunk;
    }
}

/// Every kernel the encoders here use, both matmul families (build these pipelines before encoding).
pub fn qwen_pipelines() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for k in KS {
        v.extend_from_slice(&[gemv_name(k, 0), gemv_name(k, 2), gemv_name(k, 3), swiglu_name(k)]);
        v.extend_from_slice(&[partial_name(k, false), partial_name(k, true), split_name(k, false), split_name(k, true)]);
        v.extend([false, true].iter().filter_map(|&q8| split_gqa(k, q8).map(|x| x.0)));
    }
    v.extend(mma_pipelines());
    v.extend_from_slice(&[
        "rmsnorm_b", "rmsnorm_gated", "dn_conv", "dn_prep", "dn_recur_ck0", "dn_recur_ck1",
        "attn_rope_invf", "attn_prep_b16", "attn_prep_q8", "attn_reduce_gate", "attn_sdpa_vec_b16", "argmax_b", "copy_bytes",
        "ban_logits",
    ]);
    v
}

/// Row counts of the exact (GEMV) pass: every per-K kernel exists for these (GEMV / swiglu in both families,
/// attn_partial k1..k8; the DeltaNet kernels, attn_sdpa_vec_b16, rmsnorm and argmax_b take k at run time).
pub const EXACT_KS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const KS: [usize; 8] = EXACT_KS;

/// Matmul family of the exact passes (decode, MTP verify / deep verify, the MTP head's layer and lm_head, exact
/// prefill chunks). One family per engine, so plain decode and verify always agree bit for bit (each family is
/// K-invariant on its own; the two are NOT bit-identical to each other). The fast prefill GEMM is unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GemvImpl {
    /// gemv_mma_q4.metal: scalar twins `qm_gemvt_k{K}_s8_m*` / `qm_swiglut_k{K}_s8` where `mma_use_twin`, the
    /// simdgroup-matrix kernels `qm_gemv_n2_s8_m*` / `qm_swiglu_n1_s8` above (bit-identical per row either way).
    #[default]
    Mma,
    /// gemv_q4_bf16.metal `q4b_gemv_k{K}[_r{D}]_m*` / `q4b_swiglu_k{K}[_r{D}]` (bit-exact vs MLX's qmv).
    Old,
    /// `Mma` with the twins (K <= 4) on dynamic threadgroup memory sized per dispatch (`qm_gemvtd` / `qm_swiglutd`,
    /// `mma_twin_tmem`): same kernels' arithmetic, BIT-IDENTICAL to `Mma`, higher occupancy. Never an option value:
    /// the engine sets it in `PassIo.gemv` when `QwenOptions.decode_fused` is on with gemv = Mma.
    MmaD,
}
impl GemvImpl {
    pub fn parse(s: &str) -> Result<GemvImpl, String> {
        match s { "mma" => Ok(GemvImpl::Mma), "old" => Ok(GemvImpl::Old), _ => Err(format!("gemv impl {:?}: want mma|old", s)) }
    }
    pub fn name(self) -> &'static str { match self { GemvImpl::Mma => "mma", GemvImpl::Old => "old", GemvImpl::MmaD => "mma-d" } }
    /// The Mma arithmetic family (`Mma` or `MmaD`, bit-identical to each other).
    pub fn is_mma(self) -> bool { matches!(self, GemvImpl::Mma | GemvImpl::MmaD) }
}

/// K values with compiled `q4b_gemv` / `q4b_swiglu` kernels (gemv_q4_bf16.metal). The exact pass runs the ones also
/// in `KS`.
pub const GEMV_KS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// Row-dots per simdgroup (D) compiled for `k`: 2 (the original geometry), 4 (k <= 5, 8), and 8 for k <= 3 (registers).
pub fn gemv_ds(k: usize) -> &'static [usize] { if k <= 3 { &[2, 4, 8] } else if k == 6 || k == 7 { &[2] } else { &[2, 4] } }
/// Default D per k for the exact pass (K = 1 keeps the proven 242 GB/s geometry; K = 2..4 amortise the x side over 4
/// rows; K = 5, 8 spill at D = 4: bench gemv-k full pass K=5 2.75x t1 at D 2 vs 4.92x at D 4, K=8 4.60x vs 6.69x;
/// K = 6, 7 only have D = 2). Per-row numbers do not depend on D. `SD_GEMV_D=k:d,k:d` (e.g. `2:8,4:4`)
/// overrides per k; pick from `streamdense bench gemv-k`.
const GEMV_D_DEFAULT: [(usize, usize); 8] = [(1, 2), (2, 4), (3, 4), (4, 4), (5, 2), (6, 2), (7, 2), (8, 2)];

pub fn gemv_kernel(k: usize, d: usize, mode: u32) -> String {
    if d == 2 { format!("q4b_gemv_k{}_m{}", k, mode) } else { format!("q4b_gemv_k{}_r{}_m{}", k, d, mode) }
}
pub fn swiglu_kernel(k: usize, d: usize) -> String {
    if d == 2 { format!("q4b_swiglu_k{}", k) } else { format!("q4b_swiglu_k{}_r{}", k, d) }
}

/// Resolved D and kernel names per k (built once; names leaked so encoders get &'static str like the other kernels).
struct GemvTab { d: [usize; 9], gemv: [[&'static str; 4]; 9], swiglu: [&'static str; 9] }
fn gemv_tab() -> &'static GemvTab {
    static T: std::sync::OnceLock<GemvTab> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut d = [0usize; 9];
        for (k, v) in GEMV_D_DEFAULT { d[k] = v; }
        if let Ok(s) = std::env::var("SD_GEMV_D") {
            for kv in s.split(',').filter(|x| !x.trim().is_empty()) {
                let p: Vec<usize> = kv.split(':').filter_map(|x| x.trim().parse().ok()).collect();
                match p[..] {
                    [k, v] if GEMV_KS.contains(&k) && gemv_ds(k).contains(&v) => d[k] = v,
                    _ => eprintln!("SD_GEMV_D: ignoring '{}' (want k:d, k in {:?}, d in gemv_ds(k))", kv, GEMV_KS),
                }
            }
        }
        let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
        let (mut gemv, mut swiglu) = ([[""; 4]; 9], [""; 9]);
        for k in GEMV_KS {
            for m in [0u32, 2, 3] { gemv[k][m as usize] = leak(gemv_kernel(k, d[k], m)); }
            swiglu[k] = leak(swiglu_kernel(k, d[k]));
        }
        GemvTab { d, gemv, swiglu }
    })
}
/// D the exact pass uses for `k` rows.
pub fn gemv_d(k: usize) -> usize {
    let d = gemv_tab().d.get(k).copied().unwrap_or(0);
    if d == 0 { panic!("q4b_gemv: no variant for k {}", k) }
    d
}
fn gemv_name(k: usize, mode: u32) -> &'static str {
    gemv_d(k);
    let n = gemv_tab().gemv[k][mode as usize];
    if n.is_empty() { panic!("q4b_gemv: no variant for k {} mode {}", k, mode) }
    n
}
fn swiglu_name(k: usize) -> &'static str { gemv_d(k); gemv_tab().swiglu[k] }
pub(crate) fn partial_name(k: usize, q8: bool) -> &'static str {
    match (k, q8) {
        (1, false) => "attn_partial_k1_b16", (2, false) => "attn_partial_k2_b16",
        (3, false) => "attn_partial_k3_b16", (4, false) => "attn_partial_k4_b16",
        (5, false) => "attn_partial_k5_b16", (6, false) => "attn_partial_k6_b16",
        (7, false) => "attn_partial_k7_b16", (8, false) => "attn_partial_k8_b16",
        (1, true) => "attn_partial_k1_q8", (2, true) => "attn_partial_k2_q8",
        (3, true) => "attn_partial_k3_q8", (4, true) => "attn_partial_k4_q8",
        (5, true) => "attn_partial_k5_q8", (6, true) => "attn_partial_k6_q8",
        (7, true) => "attn_partial_k7_q8", (8, true) => "attn_partial_k8_q8",
        _ => panic!("attn_partial: no variant for k {}", k),
    }
}

/// `attn_split_k{K}` / `attn_split_k{K}_q8` (attn_gated.metal `attn_split_rows`, 256 threads): the split-K partials
/// of a K-row pass, bit-identical per row to `attn_partial_k{K}_b16` / `_q8`
/// (verify_bench::tests::verify_attn_splitk_bit_identical, verify_attn_splitk_q8_bit_identical) and cheaper (27B
/// attention shapes, 32K keys, one layer + reduce, bf16: K = 1 ~0.65x, K = 8 ~0.4x the attn_partial time).
pub(crate) fn split_name(k: usize, q8: bool) -> &'static str {
    match (k, q8) {
        (1, false) => "attn_split_k1", (2, false) => "attn_split_k2", (3, false) => "attn_split_k3",
        (4, false) => "attn_split_k4", (5, false) => "attn_split_k5", (6, false) => "attn_split_k6",
        (7, false) => "attn_split_k7", (8, false) => "attn_split_k8",
        (1, true) => "attn_split_k1_q8", (2, true) => "attn_split_k2_q8", (3, true) => "attn_split_k3_q8",
        (4, true) => "attn_split_k4_q8", (5, true) => "attn_split_k5_q8", (6, true) => "attn_split_k6_q8",
        (7, true) => "attn_split_k7_q8", (8, true) => "attn_split_k8_q8",
        _ => panic!("attn_split: no variant for k {}", k),
    }
}
/// `SD_ATTN_SPLIT=old`: split-K passes run `attn_partial_k*` (A/B; same numbers, slower).
fn split_old() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("SD_ATTN_SPLIT").is_ok_and(|v| v.trim() == "old"))
}
/// GQA split-K variant of a `k`-row pass (attn_gated.metal `attn_split_rows` with G > 1): (kernel, G q heads per
/// threadgroup, dispatch (nch, nh / G)); G divides the q heads per kv head. Same numbers as `split_name(k, q8)`, cheaper
/// (27B shapes, 32K keys + reduce: K = 1 bf16 0.75x, q8 0.77x; K = 2 bf16 0.75x).
pub(crate) fn split_gqa(k: usize, q8: bool) -> Option<(&'static str, usize)> {
    match (k, q8) {
        (1, false) => Some(("attn_split_k1_g3", 3)),
        (1, true) => Some(("attn_split_k1_g2_q8", 2)),
        (2, false) => Some(("attn_split_k2_g2", 2)),
        _ => None,
    }
}
/// Split-K partial kernel of a `k`-row pass, its threads per threadgroup and q heads per threadgroup G (dispatch
/// (nch, nh / G) groups): `split_gqa` when G divides `grp` (q heads per kv head), else `attn_split_k{k}[_q8]` (G = 1),
/// or `attn_partial_k{k}_{b16,q8}` (G = 1) with SD_ATTN_SPLIT=old.
pub(crate) fn split_k_kernel(k: usize, q8: bool, grp: usize) -> (&'static str, usize, usize) {
    if split_old() { return (partial_name(k, q8), 128, 1); }
    match split_gqa(k, q8) {
        Some((n, g)) if grp % g == 0 => (n, 256, g),
        _ => (split_name(k, q8), 256, 1),
    }
}

/// Bench only (decode_bench --verify K > 8): widest exact pass row count the encoders accept. 8 = the engine
/// (never changed outside decode_bench). Above 8 a pass needs the fused verify matmuls (`vf_on`), which take up to
/// VF_K_WIDE_MAX rows; the other per-K pieces of a verify pass (DeltaNet, attn_sdpa_vec_b16 rows, norms, argmax_b)
/// take K at run time. The non-fused MMA kernels, attn_partial_k* (rows past sdpa_vec_max_n) and the verify mixers
/// stay at 8.
pub static BENCH_VERIFY_K_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(8);
fn verify_k_max() -> usize { BENCH_VERIFY_K_MAX.load(std::sync::atomic::Ordering::Relaxed).max(8) }

fn check_k(k: usize) -> Result<(), String> {
    if KS.contains(&k) || (k > 8 && k <= verify_k_max()) { Ok(()) } else { Err(format!("qwen_layers: k = {} not supported ({:?})", k, KS)) }
}

/// 4-bit GEMV of `k` rows of x (bf16 [k][m.cols]) through `m` with family `imp`. MODE 0: out bf16 [k][m.rows]; 2:
/// out += (residual, in place, bf16); 3: out f32 [k][m.rows].
pub(crate) fn gemv(e: &Enc, imp: GemvImpl, m: &Q4b, k: usize, mode: u32, x: usize, out: usize) {
    match imp {
        GemvImpl::Old => gemv_enc(e, gemv_name(k, mode), gemv_d(k), m, x, out),
        GemvImpl::Mma | GemvImpl::MmaD if mma_use_twin(m.cols, false, k) => {
            let nt = mma_twin_nt(m.cols, false, k, MMA_SK, 512);
            if imp == GemvImpl::MmaD && mmad_t4(m.cols, false, k) {
                let nt = mmad_t4_nt(m.cols, false);
                e.tgmem(0, mmad_t4_tmem(nt, false));
                e.pso(mma_tab().t4_gemv[mode as usize])
                    .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x)
                    .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out)
                    .groups((m.rows.div_ceil(4), 1), nt);
            } else if imp == GemvImpl::MmaD {
                e.tgmem(0, mma_twin_tmem(nt, false, k, MMA_SK));
                mma_twin_gemv_enc(e, mma_tab().twind_gemv[k][mode as usize], nt, m, x, out)
            } else {
                mma_twin_gemv_enc(e, mma_tab().twin_gemv[k][mode as usize], nt, m, x, out)
            }
        }
        GemvImpl::Mma | GemvImpl::MmaD => mma_gemv_enc(e, mma_tab().gemv[mode as usize], MMA_NB_GEMV, MMA_SK, m, k, x, out),
    }
}
/// out bf16 [k][gate.rows] = swiglu(gate x, up x) for `k` rows of x, family `imp`.
fn swiglu(e: &Enc, imp: GemvImpl, gate: &Q4b, up: &Q4b, k: usize, x: usize, out: usize) {
    match imp {
        GemvImpl::Old => swiglu_enc(e, swiglu_name(k), gemv_d(k), gate, up, x, out),
        GemvImpl::Mma | GemvImpl::MmaD if mma_use_twin(gate.cols, true, k) => {
            let nt = mma_twin_nt(gate.cols, true, k, MMA_SK, 512);
            if imp == GemvImpl::MmaD && mmad_t4(gate.cols, true, k) {
                let nt = mmad_t4_nt(gate.cols, true);
                e.tgmem(0, mmad_t4_tmem(nt, true));
                e.pso(mma_tab().t4_swiglu)
                    .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off)
                    .b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
                    .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32)
                    .groups((gate.rows.div_ceil(4), 1), nt);
            } else if imp == GemvImpl::MmaD {
                e.tgmem(0, mma_twin_tmem(nt, true, k, MMA_SK));
                mma_twin_swiglu_enc(e, mma_tab().twind_swiglu[k], nt, gate, up, x, out)
            } else {
                mma_twin_swiglu_enc(e, mma_tab().twin_swiglu[k], nt, gate, up, x, out)
            }
        }
        GemvImpl::Mma | GemvImpl::MmaD => mma_swiglu_enc(e, mma_tab().swiglu, MMA_NB_SWIGLU, MMA_SK, gate, up, k, x, out),
    }
}

/// read mask of one matmul (arena offsets; readmask.rs, gemv_mma_q4.metal `qm_*x`): `rb` row tiles (1 bit per
/// 8 rows), `cl` group list (count, then 64-group indices), `fl` the pass's "read everything" flag, `flags` bit 0 =
/// rows masked, bit 1 = columns listed (swiglu: rows always).
#[derive(Clone, Copy, Debug)]
pub struct MatMask { pub rb: usize, pub cl: usize, pub fl: usize, pub flags: u32 }

struct MmaXTab { twin_gemv: [[&'static str; 4]; 9], twin_swiglu: [&'static str; 9], gemv: [&'static str; 4], swiglu: &'static str }
fn mma_x_tab() -> &'static MmaXTab {
    static T: std::sync::OnceLock<MmaXTab> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
        let (mut twin_gemv, mut twin_swiglu, mut gemv) = ([[""; 4]; 9], [""; 9], [""; 4]);
        for k in 1..=MMA_TWIN_K_MAX {
            for m in [0u32, 2, 3] { twin_gemv[k][m as usize] = leak(format!("qm_gemvtx_k{}_s{}_m{}", k, MMA_SK, m)); }
            twin_swiglu[k] = leak(format!("qm_swiglutx_k{}_s{}", k, MMA_SK));
        }
        for m in [0u32, 2, 3] { gemv[m as usize] = leak(format!("qm_gemvx_n{}_s{}_m{}", MMA_NB_GEMV, MMA_SK, m)); }
        MmaXTab { twin_gemv, twin_swiglu, gemv, swiglu: leak(format!("qm_swiglux_n{}_s{}", MMA_NB_SWIGLU, MMA_SK)) }
    })
}
/// Masked kernels of the Mma family (built with `QwenOptions.read_mask`).
pub fn mma_mask_pipelines() -> Vec<&'static str> {
    let t = mma_x_tab();
    let mut v = vec![t.gemv[0], t.gemv[2], t.gemv[3], t.swiglu];
    for k in 1..=MMA_TWIN_K_MAX { v.extend_from_slice(&[t.twin_gemv[k][0], t.twin_gemv[k][2], t.twin_gemv[k][3], t.twin_swiglu[k]]); }
    v
}
/// `gemv` (Mma family) with an optional read mask: None = the plain kernel; Some = its masked variant, same twin / MMA
/// choice and SK (fl set or the all-ones mask: bit-identical to None).
#[allow(clippy::too_many_arguments)]
pub fn gemv_m(e: &Enc, imp: GemvImpl, m: &Q4b, k: usize, mode: u32, x: usize, out: usize, mk: Option<MatMask>) {
    let Some(mk) = mk.filter(|_| imp.is_mma()) else { return gemv(e, imp, m, k, mode, x, out) };
    if mma_use_twin(m.cols, false, k) {
        let nt = mma_twin_nt(m.cols, false, k, MMA_SK, 512);
        e.pso(mma_x_tab().twin_gemv[k][mode as usize])
            .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x).b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out)
            .b(10, mk.rb).b(11, mk.cl).b(12, mk.fl).u(13, mk.flags)
            .groups((m.rows.div_ceil(8), 1), nt);
    } else {
        assert!((1..=MMA_K_MAX).contains(&k), "qm_gemvx: k {} not in 1..=8", k);
        e.pso(mma_x_tab().gemv[mode as usize])
            .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x).b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out)
            .u(9, k as u32).b(10, mk.rb).b(11, mk.cl).b(12, mk.fl).u(13, mk.flags)
            .groups((m.rows.div_ceil(8 * MMA_NB_GEMV), 1), 32 * MMA_SK);
    }
}
/// `swiglu` (Mma family) with an optional row mask (`mk.rb`, `mk.fl`); masked neurons come out as 0.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_m(e: &Enc, imp: GemvImpl, gate: &Q4b, up: &Q4b, k: usize, x: usize, out: usize, mk: Option<MatMask>) {
    let Some(mk) = mk.filter(|_| imp.is_mma()) else { return swiglu(e, imp, gate, up, k, x, out) };
    let bind = |name: &str| e.pso(name)
        .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off).b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
        .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32).b(11, mk.rb).b(12, mk.fl);
    if mma_use_twin(gate.cols, true, k) {
        let nt = mma_twin_nt(gate.cols, true, k, MMA_SK, 512);
        bind(mma_x_tab().twin_swiglu[k]).groups((gate.rows.div_ceil(8), 1), nt);
    } else {
        assert!((1..=MMA_K_MAX).contains(&k), "qm_swiglux: k {} not in 1..=8", k);
        bind(mma_x_tab().swiglu).u(10, k as u32).groups((gate.rows.div_ceil(8 * MMA_NB_SWIGLU), 1), 32 * MMA_SK);
    }
}

/// Row / column masks of model layer `l` for a masked pass (None when the pass is not masked).
fn layer_masks(io: &PassIo, cfg: &QwenConfig, l: usize) -> Option<(MatMask, MatMask, MatMask, MatMask, crate::readmask::RmIo)> {
    let r = io.rm.filter(|r| r.apply && l < cfg.layers)?;
    let rows = |o: usize| MatMask { rb: r.at(l, o), cl: r.at(l, o), fl: r.fl, flags: 1 };
    let cols = |o: usize| MatMask { rb: r.at(l, o), cl: r.at(l, o), fl: r.fl, flags: 2 };
    Some((rows(r.o_rbf), cols(r.o_clf), rows(r.o_rbm), cols(r.o_clm), r))
}

/// `rm_seen` after model layer `l` (no-op without a read-mask region, or for the MTP layer).
fn encode_seen(e: &Enc, cfg: &QwenConfig, io: &PassIo, l: usize) {
    let Some(r) = io.rm.filter(|_| l < cfg.layers) else { return };
    let (mo, nh, hw) = match cfg.kinds[l] {
        LayerKind::DeltaNet => (io.yn, cfg.dn_vh, cfg.dn_dv),
        LayerKind::Attention => (io.ao, cfg.heads, cfg.head_dim),
    };
    e.pso("rm_seen").b(0, io.f).b(1, mo).b(2, r.norm(l)).b(3, r.norm(l) + r.o_on).b(4, r.at(l, r.o_sf)).b(5, r.at(l, r.o_sh))
        .u(6, cfg.inter as u32).u(7, nh as u32).u(8, hw as u32).u(9, io.k as u32).f(10, r.ffn_thr).f(11, r.head_thr)
        .groups((cfg.inter / 64 + nh, 1), 64);
}

// ---- fused K-row verify matmuls (verify_fused_q4.metal, 2026-11-03; synthetic 27B K = 8 matmul pass 122.4 ->
// 94.0 ms). Bit-identical to the Mma / MmaD family for every K (shader header; verify_bench tests), so a pass may
// switch per matmul. Used for unmasked Mma-family passes of K in VF_K_MIN..=8 (`vf_on`); K <= 2 keeps the twins
// (the vf kernels pad to 8 rows: ~flat cost), read-masked passes keep the masked kernels, the compact CDown keeps
// its own kernel. K 9..16 kernels exist (two token tiles) but the engine's other per-K kernels stop at 8.
/// Max token rows of one fused verify dispatch (two 8-row MMA tiles).
pub const VF_K_MAX: usize = 16;
/// Bench only: t3 / t4 instantiations (NB 2 gemv, NB 1 swiglu) let one dispatch take up to 32 rows; timing
/// only, bit identity not tested above 16.
pub const VF_K_WIDE_MAX: usize = 32;
/// Smallest K that takes the fused kernels in the engine.
pub const VF_K_MIN: usize = 3;
/// Weight-row blocks (8 rows each) compiled for `vf_gemv` / `vf_swiglu`; the engine uses NB 2 / 1 (within 1% of the
/// best per shape on the 27B).
pub const VF_GEMV_NBS: [usize; 3] = [1, 2, 4];
pub const VF_SWIGLU_NBS: [usize; 2] = [1, 2];
pub const VF_NB_GEMV: usize = 2;
pub const VF_NB_SWIGLU: usize = 1;
pub fn vf_gemv_kernel(nb: usize, k: usize, mode: u32) -> String { format!("vf_gemv_n{}_t{}_m{}", nb, vf_tiles(k), mode) }
pub fn vf_swiglu_kernel(nb: usize, k: usize) -> String { format!("vf_swiglu_n{}_t{}", nb, vf_tiles(k)) }
/// 8-row token tiles of a `k`-row dispatch.
pub fn vf_tiles(k: usize) -> usize { if k > 8 { k.div_ceil(8) } else { 1 } }
/// Every fused verify kernel (for `Gpu::pipeline`).
pub fn vf_pipelines() -> Vec<String> {
    let mut v = vec!["vf_xprep".to_string()];
    for t in [1, 16] {
        for nb in VF_GEMV_NBS { for m in [0, 2, 3] { v.push(vf_gemv_kernel(nb, t, m)); } }
        for nb in VF_SWIGLU_NBS { v.push(vf_swiglu_kernel(nb, t)); }
    }
    v
}
/// The kernels the engine dispatches (K <= 8, NB 2 / 1).
pub fn vf_engine_pipelines() -> Vec<String> {
    let mut v = vec!["vf_xprep".to_string()];
    for m in [0, 2, 3] { v.push(vf_gemv_kernel(VF_NB_GEMV, 8, m)); }
    v.push(vf_swiglu_kernel(VF_NB_SWIGLU, 8));
    v
}
/// Bytes of the `vf_xprep` outputs for `k` rows of a `cols`-wide input, token-minor with KP = 8 x tiles rows:
/// (xt f32 [cols][KP], st f32 [cols / 64][KP]).
pub fn vf_prep_bytes(k: usize, cols: usize) -> (usize, usize) { let kp = 8 * vf_tiles(k); (kp * cols * 4, kp * (cols / 64) * 4) }
/// Widest matmul input of a pass (hidden, inter, dn v dim, q dim): sizes the engine's vf scratch.
pub fn vf_max_cols(cfg: &QwenConfig) -> usize { cfg.hidden.max(cfg.inter).max(cfg.dn_v_dim()).max(cfg.q_dim()) }
/// xt / st of `k` rows of x (bf16 [k][cols]); once per matmul input, shared by every matrix that reads it (the
/// layout depends on k only through the tile count, so one prep serves every `vf_*` dispatch of the same k).
pub fn vf_prep_enc(e: &Enc, k: usize, cols: usize, x: usize, xt: usize, st: usize) {
    let kp = 8 * vf_tiles(k);
    e.pso("vf_xprep").b(0, x).b(1, xt).b(2, st).u(3, cols as u32).u(4, k as u32).u(5, kp as u32).threads((cols / 64 * kp, 1), 64);
}
/// `k` rows (1..=16) through `m` from the prepped input; modes as the engine's `gemv` (0 out bf16, 2 out += in
/// place, 3 out f32). ceil(rows / (8 nb)) groups x 256 threads.
pub fn vf_gemv_enc(e: &Enc, nb: usize, m: &Q4b, k: usize, mode: u32, xt: usize, st: usize, out: usize) {
    assert!((1..=VF_K_WIDE_MAX).contains(&k), "vf_gemv: k {} not in 1..=32", k);
    e.pso(&vf_gemv_kernel(nb, k, mode))
        .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, xt)
        .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out).u(9, k as u32).b(10, st)
        .groups((m.rows.div_ceil(8 * nb), 1), 32 * MMA_SK);
}
/// out bf16 [k][gate.rows] = swiglu(gate x, up x) from the prepped input.
pub fn vf_swiglu_enc(e: &Enc, nb: usize, gate: &Q4b, up: &Q4b, k: usize, xt: usize, st: usize, out: usize) {
    assert!((1..=VF_K_WIDE_MAX).contains(&k), "vf_swiglu: k {} not in 1..=32", k);
    e.pso(&vf_swiglu_kernel(nb, k))
        .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off)
        .b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
        .b(6, xt).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32).u(10, k as u32).b(11, st)
        .groups((gate.rows.div_ceil(8 * nb), 1), 32 * MMA_SK);
}
// ---- K-row verify mixers (verify_mixers.metal, 2026-11-03). Attention: the MLX two-pass sdpa rows of a pass run as
// va_sdpa_p1 (pass-1 blocks, one simdgroup per (block, q head) for R rows: each key loaded and converted once for R
// rows) + va_sdpa_p3 (pass 2 + gate from the scratch); va_sdpa_p2 = attn_sdpa_vec_b16 for every other row.
// Bit-identical to attn_sdpa_vec_b16 per row (shader header; verify_bench tests). One attention layer, 27B shapes,
// K = 8: pos 1024 346 -> 246 µs, pos 4000 967 -> 651, pos 8000 1761 -> 1204. bf16 KV only; scratch `PassIo.vm`.
// DeltaNet: dn_recur_v = dn_recur with the pass's tokens staged in threadgroup memory (K = 8, 48 layers 3.17 -> 2.88 ms).
/// `PassIo.vm` value without the attention scratch (q8 KV engines): DeltaNet part only.
pub const VM_NO_ATTN: usize = usize::MAX;
/// Most pass-1 blocks of one row (sdpa_blocks, N <= 8192).
pub const VA_BMAX: usize = 128;
/// Rows per va_sdpa_p1 threadgroup for a `k`-row pass (verify_bench --attn, 27B shapes, pos 1024..8000: R 2 best
/// for K 4..8, R 3 for K 3, R 1 for K 1).
pub fn va_r(k: usize) -> usize { match k { 1 => 1, 3 => 3, _ => 2 } }
pub const VA_RS: [usize; 5] = [1, 2, 3, 4, 8];
pub fn vm_pipelines() -> Vec<String> {
    let mut v: Vec<String> = VA_RS.iter().map(|r| format!("va_sdpa_p1_r{}", r)).collect();
    v.extend(["va_sdpa_p2", "va_sdpa_p3", "dn_recur_v_ck0", "dn_recur_v_ck1"].map(String::from));
    v
}
/// Scratch of the verify attention for passes of up to `k` rows: bf16 partials + (max, sum) per row, head, block.
pub fn va_scratch_bytes(cfg: &QwenConfig, k: usize) -> usize { k * cfg.heads * VA_BMAX * (cfg.head_dim * BF + 8) }
/// Buffers of one verify sdpa: (qr, K, V, p (gate), ao, ctrl) arena offsets.
#[derive(Clone, Copy, Debug)]
pub struct VaBufs { pub qr: usize, pub k: usize, pub v: usize, pub p: usize, pub ao: usize, pub ctrl: usize }
/// `t` rows (pass rows, row t at ctrl.pos + t) of bf16 sdpa + gate into ao, as attn_sdpa_vec_b16 over (nh, t) with the
/// same max_ctx / grp / p_stride / lim / two_pass_min. `sc` = scratch (va_scratch_bytes(cfg, t) bytes), r = rows per
/// pass-1 threadgroup.
#[allow(clippy::too_many_arguments)]
pub fn va_sdpa_enc(e: &Enc, b: &VaBufs, sc: usize, r: usize, t: usize, nh: usize, nkv: usize, hd: usize, max_ctx: usize,
                   p_stride: usize, lim: usize, tpm: usize) {
    let grp = nh / nkv;
    // one row chunk per threadgroup (the kernel takes several: 1, 2, 4 per threadgroup measured equal)
    let ch = t.div_ceil(r);
    let part_b = t * nh * VA_BMAX * hd * BF;
    e.pso(&format!("va_sdpa_p1_r{}", r)).b(0, b.qr).b(1, b.k).b(2, b.v).b(3, sc).b(4, sc + part_b).b(5, b.ctrl)
        .u(6, max_ctx as u32).u(7, grp as u32).u(8, t as u32).u(9, lim as u32).u(10, tpm as u32).u(11, nh as u32)
        .groups((VA_BMAX * ch, nkv), 32 * grp);
    e.pso("va_sdpa_p3").b(0, b.p).b(1, b.ao).b(2, b.ctrl).u(3, grp as u32).u(4, p_stride as u32).u(5, lim as u32)
        .u(6, tpm as u32).b(7, sc).b(8, sc + part_b).u(9, t as u32)
        .groups((nh, t), 256);
    e.pso("va_sdpa_p2").b(0, b.qr).b(1, b.k).b(2, b.v).b(3, b.p).b(4, b.ao).b(5, b.ctrl)
        .u(6, max_ctx as u32).u(7, grp as u32).u(8, p_stride as u32).u(9, lim as u32).u(10, tpm as u32)
        .groups((nh, t), 1024);
}

/// The vf scratch when this matmul takes the fused kernels: option on, Mma family, unmasked, K in VF_K_MIN..=8,
/// cols a multiple of 64.
fn vf_on(io: &PassIo, k: usize, cols: usize, masked: bool) -> Option<(usize, usize)> {
    io.vf.filter(|_| io.gemv.is_mma() && !masked && (VF_K_MIN..=verify_k_max()).contains(&k) && cols % 64 == 0)
}
/// Exact-pass matmul of `k` rows: fused verify kernel when `vf_on`, else `gemv_m` (unchanged).
fn mm(e: &Enc, io: &PassIo, m: &Q4b, k: usize, mode: u32, x: usize, out: usize, mk: Option<MatMask>) {
    match vf_on(io, k, m.cols, mk.is_some()) {
        Some((xt, st)) => { vf_prep_enc(e, k, m.cols, x, xt, st); vf_gemv_enc(e, VF_NB_GEMV, m, k, mode, xt, st, out); }
        None => gemv_m(e, io.gemv, m, k, mode, x, out, mk),
    }
}
/// Exact-pass swiglu of `k` rows: fused when `vf_on`, else `swiglu_m`.
fn mm_swiglu(e: &Enc, io: &PassIo, gate: &Q4b, up: &Q4b, k: usize, x: usize, out: usize, mk: Option<MatMask>) {
    match vf_on(io, k, gate.cols, mk.is_some()) {
        Some((xt, st)) => { vf_prep_enc(e, k, gate.cols, x, xt, st); vf_swiglu_enc(e, VF_NB_SWIGLU, gate, up, k, xt, st, out); }
        None => swiglu_m(e, io.gemv, gate, up, k, x, out, mk),
    }
}

// ---- engine choice inside the MMA family (`GemvImpl::Mma`). SK is part of the result (fixed-order split-K sum), so
// it is ONE value for every shape, decode and verify alike; twin vs MMA and NB are not (bit-identical per row), so
// the twin / MMA switch (`mma_use_twin`) is free to depend on K and the static matrix shape (never on positions).
/// Split-K simdgroups (MMA) / slices (twin) of the exact passes.
pub const MMA_SK: usize = 8;
/// Weight-row blocks per simdgroup of the MMA gemv / swiglu kernels.
pub const MMA_NB_GEMV: usize = 2;
pub const MMA_NB_SWIGLU: usize = 1;
/// Largest K that may run the scalar twin (twin pipelines are built for 1..=this).
pub const MMA_TWIN_K_MAX: usize = 4;
/// Twin (no 8-row padding) or MMA kernel for `k` rows of a `cols`-wide matrix. Rule from the synthetic 27B full pass
/// (bench gemv-k --impl both --ks 1..8 --sks 8): the twin wins at K <= 2 on every shape, and at
/// K = 3, 4 exactly where one chunk of its threadgroup covers all column groups (`mma_twin_nt` cap 512 threads,
/// 4096 / (4 K) T terms): K = 3 all but down (cols = inter), K = 4 the gemv shapes with cols = hidden / q_dim but
/// not gate+up; MMA from K = 5 (pass ms twin-only vs MMA-only K = 5: 184.6 vs 115.2).
pub fn mma_use_twin(cols: usize, swiglu: bool, k: usize) -> bool {
    if k <= 2 { return true; }
    if k > MMA_TWIN_K_MAX { return false; }
    let np = (if swiglu { 2 } else { 1 }) * 8 / mma_twin_rp(k); // row passes, as mma_twin_nt
    let cap = 512usize.min(4096 / (mma_twin_rp(k) * k));
    (cols / 64) * np <= cap
}

struct MmaTab { twin_gemv: [[&'static str; 4]; 9], twin_swiglu: [&'static str; 9], gemv: [&'static str; 4], swiglu: &'static str,
                twind_gemv: [[&'static str; 4]; 9], twind_swiglu: [&'static str; 9], t4_gemv: [&'static str; 4], t4_swiglu: &'static str }
fn mma_tab() -> &'static MmaTab {
    static T: std::sync::OnceLock<MmaTab> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
        let (mut twin_gemv, mut twin_swiglu, mut gemv) = ([[""; 4]; 9], [""; 9], [""; 4]);
        let (mut twind_gemv, mut twind_swiglu) = ([[""; 4]; 9], [""; 9]);
        for k in 1..=MMA_TWIN_K_MAX {
            for m in [0u32, 2, 3] { twin_gemv[k][m as usize] = leak(mma_twin_gemv_kernel(k, MMA_SK, m)); }
            twin_swiglu[k] = leak(mma_twin_swiglu_kernel(k, MMA_SK));
            for m in [0u32, 2, 3] { twind_gemv[k][m as usize] = leak(format!("qm_gemvtd_k{}_s{}_m{}", k, MMA_SK, m)); }
            twind_swiglu[k] = leak(format!("qm_swiglutd_k{}_s{}", k, MMA_SK));
        }
        for m in [0u32, 2, 3] { gemv[m as usize] = leak(mma_gemv_kernel(MMA_NB_GEMV, MMA_SK, m)); }
        let mut t4_gemv = [""; 4];
        for m in [0usize, 2, 3] { t4_gemv[m] = leak(format!("qm_gemvtd_k1_t4_s{}_m{}", MMA_SK, m)); }
        MmaTab { twin_gemv, twin_swiglu, gemv, swiglu: leak(mma_swiglu_kernel(MMA_NB_SWIGLU, MMA_SK)), twind_gemv, twind_swiglu, t4_gemv,
                 t4_swiglu: leak(format!("qm_swiglutd_k1_t4_s{}", MMA_SK)) }
    })
}
/// Kernels `GemvImpl::Mma` dispatches (part of `qwen_pipelines`).
pub fn mma_pipelines() -> Vec<&'static str> {
    let t = mma_tab();
    let mut v = vec![t.gemv[0], t.gemv[2], t.gemv[3], t.swiglu];
    for k in 1..=MMA_TWIN_K_MAX { v.extend_from_slice(&[t.twin_gemv[k][0], t.twin_gemv[k][2], t.twin_gemv[k][3], t.twin_swiglu[k]]); }
    for k in 1..=MMA_TWIN_K_MAX { v.extend_from_slice(&[t.twind_gemv[k][0], t.twind_gemv[k][2], t.twind_gemv[k][3], t.twind_swiglu[k]]); }
    v.extend_from_slice(&[t.t4_gemv[0], t.t4_gemv[2], t.t4_gemv[3], t.t4_swiglu]);
    v
}
/// `GemvImpl::MmaD` K = 1 rule (decode_bench sweep, 27B): the RT = 4 twin (`qm_gemvtd_k1_t4_*` / `qm_swiglutd_k1_t4`,
/// 4 output rows per threadgroup, one chunk) for swiglu and for gemv with cols > 8192 (ffn.down 200 -> 219 GB/s,
/// gate+up 232 -> 240); RT = 8 elsewhere (equal there). Rows per group are not part of the result.
pub fn mmad_t4(cols: usize, swiglu: bool, k: usize) -> bool { k == 1 && (swiglu || cols > 8192) && mmad_t4_nt(cols, swiglu) <= 1024 }
/// Threads of a t4 dispatch: one chunk, NP = 1 (gemv) / 2 (swiglu) row passes x G groups, at least the NR x SK reducers.
pub fn mmad_t4_nt(cols: usize, swiglu: bool) -> usize {
    let np = if swiglu { 2 } else { 1 };
    ((cols / 64) * np).max(np * 4 * MMA_SK)
}
/// Dynamic threadgroup bytes of a t4 dispatch (RP 4 x NT floats, at least the SK x NR slice partials).
pub fn mmad_t4_tmem(nt: usize, swiglu: bool) -> usize {
    let nr = if swiglu { 8 } else { 4 };
    ((4 * nt).max(MMA_SK * nr) * 4).div_ceil(16) * 16
}
/// Bytes of dynamic threadgroup memory a `GemvImpl::MmaD` twin dispatch of `nt` threads needs (T terms: RP x NT x K
/// floats, at least the SK x NR x K slice partials), rounded up to 16 B.
pub fn mma_twin_tmem(nt: usize, swiglu: bool, k: usize, sk: usize) -> usize {
    let nr = if swiglu { 16 } else { 8 };
    ((mma_twin_rp(k) * nt * k).max(sk * nr * k) * 4).div_ceil(16) * 16
}
/// `gemv` with an explicit kernel (`gemv_kernel(k, d, mode)`) and its D: ceil(rows / 2D) groups x 64 (2 simdgroups).
pub fn gemv_enc(e: &Enc, name: &str, d: usize, m: &Q4b, x: usize, out: usize) {
    e.pso(name)
        .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x)
        .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out)
        .groups((m.rows.div_ceil(2 * d), 1), 64);
}
/// `swiglu` with an explicit kernel (`swiglu_kernel(k, d)`): D / 2 row pairs per simdgroup, ceil(rows / D) groups x 64.
pub fn swiglu_enc(e: &Enc, name: &str, d: usize, gate: &Q4b, up: &Q4b, x: usize, out: usize) {
    e.pso(name)
        .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off)
        .b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
        .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32)
        .groups((gate.rows.div_ceil(d), 1), 64);
}

// ---- MMA K-row matmul (gemv_mma_q4.metal; `GemvImpl::Mma` of encode_layer / encode_head). One kernel for every
// K = 1..8 (runtime K, token rows padded to 8 inside the simdgroup tile), so a row's output bits do not depend on K.
/// NB (blocks of 8 weight rows per simdgroup) compiled in gemv_mma_q4.metal.
pub const MMA_NBS: [usize; 3] = [1, 2, 4];
/// SK (split-K simdgroups per threadgroup) compiled for `nb`. Part of the result (fixed order partial sum), so the
/// engine must use one SK per matrix shape for decode and verify alike; K never changes it.
pub fn mma_sks(nb: usize) -> &'static [usize] { if nb <= 2 { &[1, 2, 4, 8] } else { &[1, 2, 4] } }
/// Max token rows of one MMA dispatch.
pub const MMA_K_MAX: usize = 8;
pub fn mma_gemv_kernel(nb: usize, sk: usize, mode: u32) -> String { format!("qm_gemv_n{}_s{}_m{}", nb, sk, mode) }
pub fn mma_swiglu_kernel(nb: usize, sk: usize) -> String { format!("qm_swiglu_n{}_s{}", nb, sk) }
/// `k` rows (1..=8) of x (bf16 [k][m.cols]) through `m` with `mma_gemv_kernel(nb, sk, mode)`; modes as `gemv`.
/// ceil(rows / (8 nb)) groups x 32 sk threads. Needs cols % 64 == 0 (and >= 64 sk for every simdgroup to get work).
#[allow(clippy::too_many_arguments)]
pub fn mma_gemv_enc(e: &Enc, name: &str, nb: usize, sk: usize, m: &Q4b, k: usize, x: usize, out: usize) {
    assert!((1..=MMA_K_MAX).contains(&k), "qm_gemv: k {} not in 1..=8", k);
    e.pso(name)
        .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x)
        .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out).u(9, k as u32)
        .groups((m.rows.div_ceil(8 * nb), 1), 32 * sk);
}
/// out bf16 [k][gate.rows] = swiglu(gate x, up x) with `mma_swiglu_kernel(nb, sk)`.
#[allow(clippy::too_many_arguments)]
pub fn mma_swiglu_enc(e: &Enc, name: &str, nb: usize, sk: usize, gate: &Q4b, up: &Q4b, k: usize, x: usize, out: usize) {
    assert!((1..=MMA_K_MAX).contains(&k), "qm_swiglu: k {} not in 1..=8", k);
    e.pso(name)
        .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off)
        .b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
        .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32).u(10, k as u32)
        .groups((gate.rows.div_ceil(8 * nb), 1), 32 * sk);
}

/// Scalar twins of the MMA kernels for k = 1..=8 rows (gemv_mma_q4.metal qm_gemvt / qm_swiglut): per row
/// bit-identical to `mma_gemv_kernel(nb, sk, mode)` / `mma_swiglu_kernel(nb, sk)` with the same SK (any nb, any k),
/// without the 8-row padding (k = 1 is the plain decode kernel). k is compiled in (kernel per k).
pub const MMA_TWIN_KS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// SK values compiled for the twins (an MMA SK without a twin has no unpadded K = 1 path).
pub const MMA_TWIN_SKS: [usize; 4] = [1, 2, 4, 8];
pub fn mma_twin_gemv_kernel(k: usize, sk: usize, mode: u32) -> String { format!("qm_gemvt_k{}_s{}_m{}", k, sk, mode) }
pub fn mma_twin_swiglu_kernel(k: usize, sk: usize) -> String { format!("qm_swiglut_k{}_s{}", k, sk) }
/// Threads per group of a twin: NP row passes (2 gemv, 4 swiglu) x CG groups per chunk, CG = ceil(G / chunks) with
/// the fewest chunks keeping NT <= min(`tmax`, 512, 1024 / k) (threadgroup T terms: 4 NT k floats <= 4096); at least
/// NM * 8 * SK (reducer threads). Not part of the result.
/// Rows per register pass of the twin for k rows (QMT_RP in gemv_mma_q4.metal).
pub fn mma_twin_rp(_k: usize) -> usize { 4 }
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
/// k rows of x through `m` with `mma_twin_gemv_kernel(k, sk, mode)`: ceil(rows / 8) groups x `nt` threads.
pub fn mma_twin_gemv_enc(e: &Enc, name: &str, nt: usize, m: &Q4b, x: usize, out: usize) {
    e.pso(name)
        .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x)
        .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out)
        .groups((m.rows.div_ceil(8), 1), nt);
}
/// k rows: out = swiglu(gate x, up x) with `mma_twin_swiglu_kernel(k, sk)`.
pub fn mma_twin_swiglu_enc(e: &Enc, name: &str, nt: usize, gate: &Q4b, up: &Q4b, x: usize, out: usize) {
    e.pso(name)
        .b(0, gate.w.off).b(1, gate.s.off).b(2, gate.b.off)
        .b(3, up.w.off).b(4, up.s.off).b(5, up.b.off)
        .b(6, x).b(7, out).u(8, gate.rows as u32).u(9, gate.cols as u32)
        .groups((gate.rows.div_ceil(8), 1), nt);
}

fn rmsnorm(e: &Enc, cfg: &QwenConfig, k: usize, x: usize, w: usize, out: usize) {
    e.pso("rmsnorm_b").b(0, x).b(1, w).b(2, out).u(3, cfg.hidden as u32).f(4, cfg.eps).u(5, cfg.hidden as u32)
        .groups((k, 1), 1024);
}

/// Optional dump capture (no-op without a dumper).
fn cap(d: Option<&Dumper>, e: &Enc, name: String, off: usize, dt: DType, shape: &[usize]) -> Result<(), String> {
    match d { Some(d) => d.capture(e, &name, off, dt, shape), None => Ok(()) }
}

/// RoPE inverse frequencies into `io.invf` (once per engine; depends only on the config).
pub fn encode_rope_invf(e: &Enc, cfg: &QwenConfig, io: &PassIo) {
    let n = cfg.rope_dims / 2;
    e.pso("attn_rope_invf").b(0, io.invf).f(1, (cfg.rope_theta as f32).log2()).threads((n, 1), n);
}

/// One decoder layer for `io.k` tokens (plan 6.1). The kind comes from `lw.mixer` (so the MTP attention layer works
/// too); for a DeltaNet layer `l` must be the model layer index, and its DeltaNet index (for the state) is counted
/// from `cfg.kinds`. `st` is required for DeltaNet, `kv` (this layer's cache) for attention. `ck`: also write the
/// rollback checkpoint after token 0 (verify pass, plan 7.2). Dumps use the plan 9.3 names `L{l}.*`.
#[allow(clippy::too_many_arguments)]
pub fn encode_layer(e: &Enc, cfg: &QwenConfig, lw: &LayerW, st: Option<&DnState>, kv: Option<&KvRef>, io: &PassIo,
                    ck: bool, dump: Option<&Dumper>, l: usize) -> Result<(), String> {
    let k = io.k;
    check_k(k)?;
    let hd = cfg.hidden;
    if l == 0 { crate::activity::encode_begin(e, cfg, io); }
    rmsnorm(e, cfg, k, io.h, lw.in_norm.off, io.x);
    cap(dump, e, format!("L{}.x_in", l), io.x, DType::BF16, &[k, hd])?;
    match &lw.mixer {
        Mixer::Dn(dw) => {
            if cfg.kinds.get(l) != Some(&LayerKind::DeltaNet) {
                return Err(format!("qwen_layers: layer {} has DeltaNet weights but kind {:?}", l, cfg.kinds.get(l)));
            }
            let st = st.ok_or_else(|| format!("qwen_layers: DeltaNet layer {} needs a DnState", l))?;
            let dn = cfg.kinds[..l].iter().filter(|x| **x == LayerKind::DeltaNet).count();
            encode_dn(e, cfg, dw, st, io, ck, dump, l, dn)?;
        }
        Mixer::Attn(aw) => {
            let kv = kv.ok_or_else(|| format!("qwen_layers: attention layer {} needs a KvRef", l))?;
            encode_attn(e, cfg, aw, kv, io, dump, l)?;
        }
    }
    cap(dump, e, format!("L{}.h_mid", l), io.h, DType::BF16, &[k, hd])?;
    encode_ffn(e, cfg, lw, io, dump, l)
}

/// The FFN half of a decoder layer over all `io.k` rows (+ the recorder / read-mask hooks after it): x = norm(h);
/// f = silu(gate x) * up x; h += down f. Shared by `encode_layer` and the batched `encode_layer_batch`.
fn encode_ffn(e: &Enc, cfg: &QwenConfig, lw: &LayerW, io: &PassIo, dump: Option<&Dumper>, l: usize) -> Result<(), String> {
    let k = io.k;
    let hd = cfg.hidden;
    rmsnorm(e, cfg, k, io.h, lw.post_norm.off, io.x);
    if let Some(fx) = io.fxcap {
        if k == 1 && l < cfg.layers {
            let nb = hd * BF;
            e.pso("copy_bytes").b(0, io.x).b(1, fx + l * nb).u(2, nb as u32).threads((nb, 1), 256);
        }
    }
    let ff = &lw.ffn;
    let lm = layer_masks(io, cfg, l);
    mm_swiglu(e, io, &ff.gate, &ff.up, k, io.x, io.f, lm.map(|m| m.0));
    crate::ffn_select::encode(e, cfg, io, ff, l);
    cap(dump, e, format!("L{}.ffn.act", l), io.f, DType::BF16, &[k, ff.gate.rows])?;
    match &ff.cd {
        Some(cd) => crate::ffn_compact::encode_down(e, ff, cd, io),
        None => mm(e, io, &ff.down, k, 2, io.f, io.h, lm.map(|m| m.1)),
    }
    cap(dump, e, format!("L{}.h_out", l), io.h, DType::BF16, &[k, hd])?;
    crate::activity::encode_after(e, cfg, io, l);
    encode_seen(e, cfg, io, l);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_dn(e: &Enc, cfg: &QwenConfig, dw: &DnW, st: &DnState, io: &PassIo, ck: bool, dump: Option<&Dumper>,
             l: usize, dn: usize) -> Result<(), String> {
    let k = io.k;
    let prow = cfg.dn_in_rows();
    // p = [qkv | z | b | a]
    let lm = layer_masks(io, cfg, l);
    mm(e, io, &dw.in_proj, k, 0, io.x, io.p, lm.map(|m| m.2));
    cap(dump, e, format!("L{}.dn.proj", l), io.p, DType::BF16, &[k, prow])?;
    encode_dn_mix(e, cfg, dw, st, io, ck, dump, l, dn)?;
    mm(e, io, &dw.out_proj, k, 2, io.yn, io.h, lm.map(|m| m.3));
    Ok(())
}

/// The DeltaNet token mixer of `encode_dn` between its projections: replay / hold saves, conv, prep, the gated delta
/// rule on `st` (live copy ctrl.role), gated norm -> io.yn. Reads io.p rows 0..k (the in_proj output).
#[allow(clippy::too_many_arguments)]
fn encode_dn_mix(e: &Enc, cfg: &QwenConfig, dw: &DnW, st: &DnState, io: &PassIo, ck: bool, dump: Option<&Dumper>,
                 l: usize, dn: usize) -> Result<(), String> {
    let k = io.k;
    let dd = DnDims::from_cfg(cfg);
    let (conv, prow, vh) = (cfg.conv_dim(), cfg.dn_in_rows(), cfg.dn_vh);
    let (kdim, vdim) = (cfg.dn_k_dim(), cfg.dn_v_dim());
    // deep spec verify: keep the replay inputs of rows 1..k-2 (row 0 is the checkpoint, row k-1 is never replayed)
    if let (Some(sv), true) = (io.dn_save, k > 2) {
        e.pso("dn_save_rows").b(0, io.p).b(1, sv).u(2, dn as u32).u(3, k as u32).bytes(4, &dd)
            .threads((conv + 2 * vh, k - 2), 256);
    }
    // context-draft verify: keep every row's replay inputs; the state is only read (held)
    if let Some(sv) = io.dn_hold {
        if ck { return Err("qwen_layers: dn_hold and ck together".into()); }
        e.pso("dn_save_all").b(0, io.p).b(1, sv).u(2, dn as u32).u(3, k as u32).bytes(4, &dd)
            .threads((conv + 2 * vh, k), 256);
    }
    // conv + silu -> cv; state update in place (checkpoint after token 0 when ck)
    e.pso("dn_conv").b(0, io.p).b(1, dw.conv_w.off).b(2, st.conv).b(3, io.cv).b(4, io.ctrl)
        .u(5, dn as u32).u(6, k as u32).u(7, if io.dn_hold.is_some() { DN_HOLD } else if ck { 0 } else { u32::MAX }).bytes(8, &dd)
        .threads((conv, 1), 256);
    cap(dump, e, format!("L{}.dn.conv", l), io.cv, DType::BF16, &[k, conv])?;
    // q, k l2-normed in place; gates
    e.pso("dn_prep").b(0, io.cv).b(1, io.p).b(2, dw.a_log.off).b(3, dw.dt_bias.off).b(4, io.g).b(5, io.beta)
        .u(6, k as u32).bytes(7, &dd)
        .groups((2 * cfg.dn_kh + 1, k), 128);
    if k == 1 {
        // q | k | v are contiguous slices of the single cv row only when k == 1
        cap(dump, e, format!("L{}.dn.q", l), io.cv, DType::BF16, &[k, kdim])?;
        cap(dump, e, format!("L{}.dn.k", l), io.cv + kdim * BF, DType::BF16, &[k, kdim])?;
        cap(dump, e, format!("L{}.dn.v", l), io.cv + 2 * kdim * BF, DType::BF16, &[k, vdim])?;
    }
    cap(dump, e, format!("L{}.dn.g", l), io.g, DType::F32, &[k, vh])?;
    cap(dump, e, format!("L{}.dn.beta", l), io.beta, DType::BF16, &[k, vh])?;
    // gated delta rule -> y
    if ck && st.copies < 2 { return Err("qwen_layers: checkpoint (ck) needs 2 DeltaNet state copies (MTP on)".into()); }
    let s_layer = st.s + dn * DnState::s_layer_bytes(cfg, st.copies);
    // verify mixers (io.vm): 2..=8-row passes stage the tokens' k / q / v / g / beta in threadgroup memory
    let vm = io.vm.is_some() && (2..=8).contains(&k);
    e.pso(if io.dn_hold.is_some() { "dn_recur_hold" } else if vm { if ck { "dn_recur_v_ck1" } else { "dn_recur_v_ck0" } }
          else if ck { "dn_recur_ck1" } else { "dn_recur_ck0" })
        .b(0, io.cv).b(1, io.g).b(2, io.beta).b(3, s_layer).b(4, io.y).b(5, io.ctrl).u(6, k as u32).bytes(7, &dd)
        .groups((4 * vh, 1), 128);
    cap(dump, e, format!("L{}.dn.y", l), io.y, DType::BF16, &[k, vdim])?;
    // State dumps read copy 0 (role is GPU-side; plain passes never flip it).
    cap(dump, e, format!("L{}.dn.S", l), s_layer, DType::F32, &[vh, cfg.dn_dv, cfg.dn_dk])?;
    cap(dump, e, format!("L{}.dn.conv_state", l), st.conv + dn * DnState::conv_layer_bytes(cfg), DType::BF16,
        &[cfg.conv_k - 1, conv])?;
    // yn = rmsnorm(y, w) * silu(z), z = p cols conv..conv + vdim
    e.pso("rmsnorm_gated").b(0, io.y).b(1, io.p + conv * BF).u(2, prow as u32).b(3, dw.norm_w.off).b(4, io.yn)
        .f(5, cfg.eps)
        .groups((vh, k), cfg.dn_dv);
    cap(dump, e, format!("L{}.dn.yn", l), io.yn, DType::BF16, &[k, vdim])?;
    Ok(())
}

fn encode_attn(e: &Enc, cfg: &QwenConfig, aw: &AttnW, kv: &KvRef, io: &PassIo, dump: Option<&Dumper>, l: usize)
               -> Result<(), String> {
    let k = io.k;
    let (nh, hdim) = (cfg.heads, cfg.head_dim);
    let prow = cfg.attn_in_rows();
    let nch = io.nch_max;
    if nch == 0 { return Err("qwen_layers: nch_max = 0".into()); }
    // p = [q|gate per head | k | v]
    let lm = layer_masks(io, cfg, l);
    mm(e, io, &aw.in_proj, k, 0, io.x, io.p, lm.map(|m| m.2));
    cap(dump, e, format!("L{}.attn.proj", l), io.p, DType::BF16, &[k, prow])?;
    encode_attn_mix(e, cfg, aw, kv, io, dump, l)?;
    if let Some((_, _, _, _, r)) = lm {
        // masked q heads ran attention on q = 0: their output is exactly 0, as any masked unit
        e.pso("rm_zero_heads").b(0, io.ao).b(1, r.at(l, r.o_hb)).b(2, r.fl).u(3, hdim as u32).u(4, nh as u32).threads((nh * hdim, k), 256);
    }
    mm(e, io, &aw.o_proj, k, 2, io.ao, io.h, lm.map(|m| m.3));
    Ok(())
}

/// sdpa + gate of a `k`-row pass at host position `pos` (= ctrl.pos) -> b.ao, as `encode_attn_mix` runs it after
/// attn_prep. Row t sees N = pos + t + 1 keys. bf16 KV: rows with N <= lim take the MLX single-pass mirror
/// (bit-exact), longer rows the split-K path (`nch` 256-key chunks, partials in po / pml); a pass that straddles lim
/// runs split-K first and the mirror overwrites its rows. Each row's arithmetic depends only on its own N, so
/// K-invariance holds. `vm` = verify-mixer scratch (see `PassIo::vm`); b.k / b.v must equal kv.k / kv.v.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_sdpa_rows(e: &Enc, kv: &KvRef, b: &VaBufs, po: usize, pml: usize, pos: usize, k: usize, nch: usize,
                             nh: usize, nkv: usize, hdim: usize, prow: usize, vm: Option<usize>) {
    let lim = if kv.q8 { 0 } else { sdpa_vec_max_n().min(8192) }; // 8192: the kernel's blocks <= 128 bound
    let (n_lo, n_hi) = (pos + 1, pos + k);
    if n_hi > lim {
        let (pn, nt, g) = split_k_kernel(k, kv.q8, nh / nkv);
        e.pso(pn)
            .b(0, b.qr).b(1, kv.k).b(2, kv.v).b(3, kv.ksb).b(4, kv.vsb).b(5, po).b(6, pml).b(7, b.ctrl)
            .u(8, kv.max_ctx as u32).u(9, (nh / nkv) as u32)
            .groups((nch, nh / g), nt);
        e.pso("attn_reduce_gate").b(0, po).b(1, pml).b(2, b.p).b(3, b.ao).u(4, nch as u32).u(5, prow as u32)
            .groups((nh, k), hdim);
    }
    if let (true, Some(sc)) = (n_lo <= lim && !kv.q8 && k <= 8, vm.filter(|&o| o != VM_NO_ATTN)) {
        va_sdpa_enc(e, b, sc, va_r(k), k, nh, nkv, hdim, kv.max_ctx, prow, lim, sdpa_two_pass_min());
    } else if n_lo <= lim {
        e.pso("attn_sdpa_vec_b16")
            .b(0, b.qr).b(1, kv.k).b(2, kv.v).b(3, b.p).b(4, b.ao).b(5, b.ctrl)
            .u(6, kv.max_ctx as u32).u(7, (nh / nkv) as u32).u(8, prow as u32).u(9, lim as u32)
            .u(10, sdpa_two_pass_min() as u32)
            .groups((nh, k), 1024);
    }
}

/// The attention token mixer of `encode_attn` between its projections: q/k norm + RoPE, K / V rows at ctrl.pos + t
/// into `kv`, sdpa per row -> io.ao (gated). Reads io.p rows 0..k (the in_proj output); io.pos = host ctrl.pos.
fn encode_attn_mix(e: &Enc, cfg: &QwenConfig, aw: &AttnW, kv: &KvRef, io: &PassIo, dump: Option<&Dumper>, l: usize)
                   -> Result<(), String> {
    let k = io.k;
    let (nh, nkv, hdim) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let prow = cfg.attn_in_rows();
    let nch = io.nch_max;
    if nch == 0 { return Err("qwen_layers: nch_max = 0".into()); }
    // q/k norm + RoPE -> qr; K, V appended at ctrl.pos + t
    e.pso(if kv.q8 { "attn_prep_q8" } else { "attn_prep_b16" })
        .b(0, io.p).b(1, aw.q_norm.off).b(2, aw.k_norm.off).b(3, io.cv)
        .b(4, kv.k).b(5, kv.v).b(6, kv.ksb).b(7, kv.vsb).b(8, io.invf).b(9, io.ctrl)
        .u(10, kv.max_ctx as u32).f(11, cfg.eps).u(12, nkv as u32).u(13, cfg.rope_dims as u32)
        .groups((nh + nkv, k), hdim);
    // qr lives in the cv scratch (bf16 [k][nh][hd]; DeltaNet and attention never share a layer)
    cap(dump, e, format!("L{}.attn.q", l), io.cv, DType::BF16, &[k, nh, hdim])?;
    let vm = io.vm.or(io.va1.filter(|_| k == 1));
    attn_sdpa_rows(e, kv, &VaBufs { qr: io.cv, k: kv.k, v: kv.v, p: io.p, ao: io.ao, ctrl: io.ctrl }, io.po, io.pml, io.pos, k, nch,
                   nh, nkv, hdim, prow, vm);
    cap(dump, e, format!("L{}.attn.o", l), io.ao, DType::BF16, &[k, cfg.q_dim()])?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Prefill token mixers (PF-B). The projections (dn in_proj / out_proj, attn in_proj / o_proj) are NOT encoded here:
// the caller runs them (GEMM over the T rows) before / after. Same PassIo buffer conventions as encode_dn /
// encode_attn, with T = io.k rows (1..=PREFILL_T_MAX, no K = 1/2/4/8 restriction).

/// Largest T for the prefill mixers (`dn_conv_pf`: ceil(T / 8) blocks of 32 threads in one threadgroup).
pub const PREFILL_T_MAX: usize = 256;

/// Below this T, encode_attn_prefill uses the per-row MLX mirror `attn_sdpa_vec_b16` (bit-exact, and faster there:
/// the flash kernel has only ceil(T / 8) x kv_heads threadgroups). Measured at the 27B shapes, pos 1000 (min of runs):
/// T = 8 mirror 266 us vs flash 567; T = 16 530 vs 575; T = 32 1129 vs 578.
pub const PREFILL_FLASH_MIN_T: usize = 17;

/// Kernels the prefill mixers use (build these pipelines before encoding; all but `dn_conv_pf` and `attn_flash_*`
/// are shared with `qwen_pipelines`).
pub fn prefill_pipelines() -> Vec<&'static str> {
    vec!["dn_conv_pf", "dn_prep", "dn_recur_ck0", "rmsnorm_gated", "attn_prep_b16", "attn_prep_q8", "attn_flash_b16", "attn_flash_q8",
         "attn_flash2_q8", "attn_sdpa_vec_b16"]
}

/// `SD_ATTN_V2=0` selects the old 8-bit-KV prefill attention `attn_flash_q8`; default (unset or anything else) is
/// `attn_flash2_q8` (attn_gated.metal; fp16 operands + simdgroup_load, 24-key blocks, 16-row tiles; 27B shapes T 128:
/// 1.7x at 2K-32K keys; outputs equal attn_flash_q8's except rare last-bit differences, metal tests/attn_flash2.rs).
/// Fast prefill only (encode_attn_prefill); decode / verify attention and the exact prefill path never use it.
pub fn attn_v2() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("SD_ATTN_V2").map_or(true, |v| v.trim() != "0"))
}

fn check_prefill_t(t: usize) -> Result<(), String> {
    if (1..=PREFILL_T_MAX).contains(&t) { Ok(()) } else { Err(format!("qwen_layers: prefill T = {} not in 1..={}", t, PREFILL_T_MAX)) }
}

/// DeltaNet mixer for the T = io.k rows of a prefill pass, plain mode (no checkpoint; the live state copy is
/// ctrl.role, updated in place, so decode after prefill continues from it). `l` = model layer index.
/// Reads io.p (in_proj output, bf16 [T][dn_in_rows]); writes io.cv (conv out, q/k normed), io.g, io.beta, io.y and
/// io.yn (bf16 [T][dn_v_dim] = rmsnorm(y) * silu(z), the out_proj input). Bit-exact vs encode_dn's kernels.
pub fn encode_dn_prefill(e: &Enc, cfg: &QwenConfig, dw: &DnW, st: &DnState, io: &PassIo, l: usize) -> Result<(), String> {
    let t = io.k;
    check_prefill_t(t)?;
    if cfg.kinds.get(l) != Some(&LayerKind::DeltaNet) {
        return Err(format!("qwen_layers: prefill layer {} is not DeltaNet ({:?})", l, cfg.kinds.get(l)));
    }
    let dn = cfg.kinds[..l].iter().filter(|x| **x == LayerKind::DeltaNet).count();
    let dd = DnDims::from_cfg(cfg);
    let (conv, prow, vh) = (cfg.conv_dim(), cfg.dn_in_rows(), cfg.dn_vh);
    e.pso("dn_conv_pf").b(0, io.p).b(1, dw.conv_w.off).b(2, st.conv).b(3, io.cv).b(4, io.ctrl)
        .u(5, dn as u32).u(6, t as u32).bytes(7, &dd)
        .groups((conv.div_ceil(32), 1), 32 * t.div_ceil(8));
    e.pso("dn_prep").b(0, io.cv).b(1, io.p).b(2, dw.a_log.off).b(3, dw.dt_bias.off).b(4, io.g).b(5, io.beta)
        .u(6, t as u32).bytes(7, &dd)
        .groups((2 * cfg.dn_kh + 1, t), 128);
    let s_layer = st.s + dn * DnState::s_layer_bytes(cfg, st.copies);
    // dn_recur_ck0 already walks all T tokens with the state in registers (one S read + one write per dispatch)
    e.pso("dn_recur_ck0")
        .b(0, io.cv).b(1, io.g).b(2, io.beta).b(3, s_layer).b(4, io.y).b(5, io.ctrl).u(6, t as u32).bytes(7, &dd)
        .groups((4 * vh, 1), 128);
    e.pso("rmsnorm_gated").b(0, io.y).b(1, io.p + conv * BF).u(2, prow as u32).b(3, dw.norm_w.off).b(4, io.yn)
        .f(5, cfg.eps)
        .groups((vh, t), cfg.dn_dv);
    Ok(())
}

/// Gated attention mixer for the T = io.k rows of a prefill pass. Reads io.p (in_proj output, bf16
/// [T][attn_in_rows]); attn_prep_b16 writes qr into io.cv and K/V rows ctrl.pos .. ctrl.pos + T - 1 into the cache;
/// then io.ao (bf16 [T][q_dim], gated; the o_proj input) comes from attn_flash_b16 (T >= PREFILL_FLASH_MIN_T; not
/// bit-exact vs the per-row mirror: flash summation order, max abs diff ~5e-4 on random data) or from the bit-exact
/// per-row mirror attn_sdpa_vec_b16 (small T, all rows' N = io.pos + t + 1 <= sdpa_vec_max_n()). Causal: row t sees
/// keys 0 ..= pos + t. io.pos must be the host copy of ctrl.pos (as encode_attn).
/// 8-bit KV (kv.q8): attn_prep_q8 (the decode path's KV writer: the same K / V values give the same q8 bytes and
/// scale / min) and attn_flash2_q8 (default; `attn_v2()`) or attn_flash_q8 (SD_ATTN_V2=0) for every T (no per-row
/// mirror for q8; both dequantize the staged tiles to fp16).
pub fn encode_attn_prefill(e: &Enc, cfg: &QwenConfig, aw: &AttnW, kv: &KvRef, io: &PassIo) -> Result<(), String> {
    let t = io.k;
    check_prefill_t(t)?;
    let (nh, nkv, hdim) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let grp = nh / nkv;
    if grp * 32 > 1024 { return Err(format!("qwen_layers: attn_flash_b16 needs heads / kv_heads <= 32 (got {})", grp)); }
    if kv.k % 16 != 0 || kv.v % 16 != 0 { return Err("qwen_layers: attn_flash_b16 needs 16-byte aligned KV".into()); }
    let prow = cfg.attn_in_rows();
    e.pso(if kv.q8 { "attn_prep_q8" } else { "attn_prep_b16" })
        .b(0, io.p).b(1, aw.q_norm.off).b(2, aw.k_norm.off).b(3, io.cv)
        .b(4, kv.k).b(5, kv.v).b(6, kv.ksb).b(7, kv.vsb).b(8, io.invf).b(9, io.ctrl)
        .u(10, kv.max_ctx as u32).f(11, cfg.eps).u(12, nkv as u32).u(13, cfg.rope_dims as u32)
        .groups((nh + nkv, t), hdim);
    let lim = sdpa_vec_max_n().min(8192);
    if !kv.q8 && t < PREFILL_FLASH_MIN_T && io.pos + t <= lim {
        e.pso("attn_sdpa_vec_b16")
            .b(0, io.cv).b(1, kv.k).b(2, kv.v).b(3, io.p).b(4, io.ao).b(5, io.ctrl)
            .u(6, kv.max_ctx as u32).u(7, grp as u32).u(8, prow as u32).u(9, lim as u32)
            .u(10, sdpa_two_pass_min() as u32)
            .groups((nh, t), 1024);
        return Ok(());
    }
    if kv.q8 && attn_v2() && grp <= 16 {
        e.pso("attn_flash2_q8")
            .b(0, io.cv).b(1, kv.k).b(2, kv.v).b(3, io.p).b(4, io.ao).b(5, io.ctrl)
            .u(6, kv.max_ctx as u32).u(7, grp as u32).u(8, prow as u32).u(9, t as u32).b(10, kv.ksb).b(11, kv.vsb)
            .groups((t.div_ceil(16), nkv), 64 * grp);
        return Ok(());
    }
    e.pso(if kv.q8 { "attn_flash_q8" } else { "attn_flash_b16" })
        .b(0, io.cv).b(1, kv.k).b(2, kv.v).b(3, io.p).b(4, io.ao).b(5, io.ctrl)
        .u(6, kv.max_ctx as u32).u(7, grp as u32).u(8, prow as u32).u(9, t as u32).b(10, kv.ksb).b(11, kv.vsb)
        .groups((t.div_ceil(8), nkv), 32 * grp);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Fast prefill pass (PF-A, plan U25): the projections and the FFN run as simdgroup-matrix GEMMs over all T rows
// (`gemm_q4.metal`), the token mixers as PF-B's encode_dn_prefill / encode_attn_prefill. NOT bit-exact with the
// stepwise path (GEMM summation order; flash attention for T >= PREFILL_FLASH_MIN_T): held to the teacher bar.

/// GEMM tiles (BM token rows x BN weight rows) built in gemm_q4.metal; 128 threads each.
const GEMM_TILES: [(usize, usize); 2] = [(32, 64), (64, 64)];

/// Every kernel `encode_layer_prefill` uses besides `prefill_pipelines()` and `rmsnorm_b`.
pub fn gemm_pipelines() -> Vec<String> {
    let mut v = Vec::new();
    for (bm, bn) in GEMM_TILES {
        for m in [0, 2, 3] { v.push(format!("q4b_gemm_{}x{}_m{}", bm, bn, m)); }
        v.push(format!("q4b_gemm_swiglu_{}x{}", bm, bn));
    }
    v
}

/// The fast swiglu GEMMs that also write the gate output (`encode_layer_prefill_g`; streamed prefill fast mode).
pub fn gemm_g_pipelines() -> Vec<String> { GEMM_TILES.iter().map(|(bm, bn)| format!("q4b_gemm_swiglu_g_{}x{}", bm, bn)).collect() }

/// Tile for T rows (measured on M4 Pro, qwen_gemm_bench: 32x64 best up to T = 32, 64x64 above).
fn gemm_tile(t: usize) -> (usize, usize) { if t <= 32 { GEMM_TILES[0] } else { GEMM_TILES[1] } }

/// 4-bit GEMM of `t` rows of x (bf16 [t][m.cols]) through `m`; modes as `gemv` (0 out, 2 residual add, 3 f32).
fn gemm(e: &Enc, m: &Q4b, t: usize, mode: u32, x: usize, out: usize) {
    let (bm, bn) = gemm_tile(t);
    e.pso(&format!("q4b_gemm_{}x{}_m{}", bm, bn, mode))
        .b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x)
        .b(4, out).u(5, m.rows as u32).u(6, m.cols as u32).b(7, out).b(8, out).u(9, t as u32)
        .groups((m.rows.div_ceil(bn), t.div_ceil(bm)), 128);
}

/// One decoder layer for the T = io.k rows of a fast prefill pass (1..=PREFILL_T_MAX rows; used for T > 8).
/// Plain mode only: no checkpoint, no dumps; bf16 or q8 KV. Same buffers as encode_layer.
#[allow(clippy::too_many_arguments)]
pub fn encode_layer_prefill(e: &Enc, cfg: &QwenConfig, lw: &LayerW, st: Option<&DnState>, kv: Option<&KvRef>, io: &PassIo,
                            l: usize) -> Result<(), String> {
    encode_layer_prefill_g(e, cfg, lw, st, kv, io, l, None)
}

/// `encode_layer_prefill` + `gout`: the FFN swiglu GEMM also writes the gate output bf(gate x), bf16 [T][inter], to
/// that offset (`q4b_gemm_swiglu_g_*`; f and every other output bit-identical). Used by the streamed prefill fast mode.
#[allow(clippy::too_many_arguments)]
pub fn encode_layer_prefill_g(e: &Enc, cfg: &QwenConfig, lw: &LayerW, st: Option<&DnState>, kv: Option<&KvRef>, io: &PassIo,
                              l: usize, gout: Option<usize>) -> Result<(), String> {
    let t = io.k;
    check_prefill_t(t)?;
    if l == 0 { crate::activity::encode_begin(e, cfg, io); }
    rmsnorm(e, cfg, t, io.h, lw.in_norm.off, io.x);
    match &lw.mixer {
        Mixer::Dn(dw) => {
            let st = st.ok_or_else(|| format!("qwen_layers: DeltaNet layer {} needs a DnState", l))?;
            gemm(e, &dw.in_proj, t, 0, io.x, io.p);
            encode_dn_prefill(e, cfg, dw, st, io, l)?;
            gemm(e, &dw.out_proj, t, 2, io.yn, io.h);
        }
        Mixer::Attn(aw) => {
            let kv = kv.ok_or_else(|| format!("qwen_layers: attention layer {} needs a KvRef", l))?;
            gemm(e, &aw.in_proj, t, 0, io.x, io.p);
            encode_attn_prefill(e, cfg, aw, kv, io)?;
            gemm(e, &aw.o_proj, t, 2, io.ao, io.h);
        }
    }
    rmsnorm(e, cfg, t, io.h, lw.post_norm.off, io.x);
    let ff = &lw.ffn;
    let (bm, bn) = gemm_tile(t);
    let sw = e.pso(&match gout { Some(_) => format!("q4b_gemm_swiglu_g_{}x{}", bm, bn), None => format!("q4b_gemm_swiglu_{}x{}", bm, bn) })
        .b(0, ff.gate.w.off).b(1, ff.gate.s.off).b(2, ff.gate.b.off)
        .b(3, ff.up.w.off).b(4, ff.up.s.off).b(5, ff.up.b.off)
        .b(6, io.x).b(7, io.f).u(8, ff.gate.rows as u32).u(9, ff.gate.cols as u32).u(10, t as u32);
    if let Some(g) = gout { sw.b(11, g); }
    sw.groups((ff.gate.rows.div_ceil(bn / 2), t.div_ceil(bm)), 128);
    gemm(e, &ff.down, t, 2, io.f, io.h);
    crate::activity::encode_after(e, cfg, io, l);
    encode_seen(e, cfg, io, l);
    Ok(())
}

/// Head: hf = final_norm(h) for all `io.k` rows (kept for MTP), then lm_head (MODE 3, f32) and `argmax_b` on the
/// last `rows` rows (rows = 1 for plain decode / prefill, 2 for the MTP verify pass) -> logits [rows][vocab],
/// ctrl.tok / top1 / top2, rec[ctrl.pos + t]. Dump: `hf` (all k rows).
pub fn encode_head(e: &Enc, cfg: &QwenConfig, w: &Weights, io: &PassIo, rows: usize, dump: Option<&Dumper>)
                   -> Result<(), String> {
    let k = io.k;
    if rows == 0 || rows > k { return Err(format!("qwen_layers: head rows {} not in 1..={}", rows, k)); }
    check_k(rows)?;
    rmsnorm(e, cfg, k, io.h, w.final_norm.off, io.hf);
    cap(dump, e, "hf".to_string(), io.hf, DType::BF16, &[k, cfg.hidden])?;
    let x0 = io.hf + (k - rows) * cfg.hidden * BF;
    mm(e, io, &w.lm_head, rows, 3, x0, io.logits, None);
    if let Some(b) = io.ban {
        e.pso("ban_logits").b(0, io.logits).b(1, b).u(2, w.lm_head.rows as u32).groups((rows, 1), 256);
    }
    e.pso("argmax_b").b(0, io.logits).b(1, io.ctrl).b(2, io.rec).u(3, w.lm_head.rows as u32)
        .groups((rows, 1), 1024);
    Ok(())
}

/// Bytes of `PassIo` scratch rows a caller must provide for `k` rows (per buffer), for sizing: (name, bytes).
pub fn scratch_sizes(cfg: &QwenConfig, k: usize, nch_max: usize) -> Vec<(&'static str, usize)> {
    let pw = cfg.dn_in_rows().max(cfg.attn_in_rows());
    let cvw = cfg.conv_dim().max(cfg.q_dim());
    vec![
        ("h", k * cfg.hidden * BF), ("x", k * cfg.hidden * BF), ("hf", k * cfg.hidden * BF),
        ("p", k * pw * BF), ("cv", k * cvw * BF), ("g", k * cfg.dn_vh * F32), ("beta", k * cfg.dn_vh * BF),
        ("y", k * cfg.dn_v_dim() * BF), ("yn", k * cfg.dn_v_dim() * BF), ("ao", k * cfg.q_dim() * BF),
        ("f", k * cfg.inter * BF),
        // split-K partials and logits only ever serve K <= 8 rows (attn_partial_k*, head / teacher blocks), also when
        // a fast prefill pass has k = T up to PREFILL_T_MAX rows
        ("po", k.min(8) * cfg.heads * nch_max * cfg.head_dim * F32), ("pml", k.min(8) * cfg.heads * nch_max * 2 * F32),
        ("logits", k.min(8) * cfg.vocab * F32), ("invf", cfg.rope_dims / 2 * F32),
    ]
}

// ---------------------------------------------------------------------------------------------------------------
// Bench only (decode_bench --batch): one verify pass carrying B sequences of R rows each. Norms, projections,
// FFN run on all B*R rows together (fused verify matmuls when io.vf and B*R in VF_K_MIN..=8); the token mixers
// (DeltaNet conv/recurrence, attention prep + sdpa) run per sequence on that sequence's own state / KV / ctrl.pos.
// No masks, no dumps, no verify mixers (vm), no hold. Not used by the engine.

/// One sequence of a batched bench pass: its DeltaNet state, KV per layer, ctrl (pos, role).
#[derive(Clone, Debug)]
pub struct BenchSeq { pub dn: DnState, pub kv: Vec<Option<KvRef>>, pub ctrl: usize }

/// One layer of a batched verify pass (see above). `io.k` = B * R rows, `r` rows per sequence; `io.pos` = every
/// sequence's ctrl.pos (same context length; only the lim / nch choice uses it).
pub fn bench_encode_layer_batch(e: &Enc, cfg: &QwenConfig, lw: &LayerW, seqs: &[BenchSeq], io: &PassIo, r: usize, ck: bool, l: usize)
                                -> Result<(), String> {
    let k = io.k;
    if seqs.len() * r != k { return Err(format!("batch: {} seqs x {} rows != k {}", seqs.len(), r, k)); }
    check_k(k)?;
    check_k(r)?;
    rmsnorm(e, cfg, k, io.h, lw.in_norm.off, io.x);
    match &lw.mixer {
        Mixer::Dn(dw) => {
            let dn = cfg.kinds[..l].iter().filter(|x| **x == LayerKind::DeltaNet).count();
            let dd = DnDims::from_cfg(cfg);
            let (conv, prow, vh, vdim) = (cfg.conv_dim(), cfg.dn_in_rows(), cfg.dn_vh, cfg.dn_v_dim());
            mm(e, io, &dw.in_proj, k, 0, io.x, io.p, None);
            for (si, sq) in seqs.iter().enumerate() {
                let o = si * r;
                let (p, cv, g, beta, y, yn) = (io.p + o * prow * BF, io.cv + o * conv * BF, io.g + o * vh * F32,
                                               io.beta + o * vh * BF, io.y + o * vdim * BF, io.yn + o * vdim * BF);
                let st = &sq.dn;
                if let (Some(sv), true) = (io.dn_save, r > 2) {
                    e.pso("dn_save_rows").b(0, p).b(1, sv).u(2, dn as u32).u(3, r as u32).bytes(4, &dd)
                        .threads((conv + 2 * vh, r - 2), 256);
                }
                e.pso("dn_conv").b(0, p).b(1, dw.conv_w.off).b(2, st.conv).b(3, cv).b(4, sq.ctrl)
                    .u(5, dn as u32).u(6, r as u32).u(7, if ck { 0 } else { u32::MAX }).bytes(8, &dd)
                    .threads((conv, 1), 256);
                e.pso("dn_prep").b(0, cv).b(1, p).b(2, dw.a_log.off).b(3, dw.dt_bias.off).b(4, g).b(5, beta)
                    .u(6, r as u32).bytes(7, &dd)
                    .groups((2 * cfg.dn_kh + 1, r), 128);
                let s_layer = st.s + dn * DnState::s_layer_bytes(cfg, st.copies);
                e.pso(if ck { "dn_recur_ck1" } else { "dn_recur_ck0" })
                    .b(0, cv).b(1, g).b(2, beta).b(3, s_layer).b(4, y).b(5, sq.ctrl).u(6, r as u32).bytes(7, &dd)
                    .groups((4 * vh, 1), 128);
                e.pso("rmsnorm_gated").b(0, y).b(1, p + conv * BF).u(2, prow as u32).b(3, dw.norm_w.off).b(4, yn)
                    .f(5, cfg.eps)
                    .groups((vh, r), cfg.dn_dv);
            }
            mm(e, io, &dw.out_proj, k, 2, io.yn, io.h, None);
        }
        Mixer::Attn(aw) => {
            let (nh, nkv, hdim) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
            let (prow, qd) = (cfg.attn_in_rows(), cfg.q_dim());
            let nch = io.nch_max;
            mm(e, io, &aw.in_proj, k, 0, io.x, io.p, None);
            for (si, sq) in seqs.iter().enumerate() {
                let o = si * r;
                let kv = sq.kv[l].as_ref().ok_or_else(|| format!("batch: layer {} needs a KvRef", l))?;
                let (p, qr, ao) = (io.p + o * prow * BF, io.cv + o * qd * BF, io.ao + o * qd * BF);
                e.pso(if kv.q8 { "attn_prep_q8" } else { "attn_prep_b16" })
                    .b(0, p).b(1, aw.q_norm.off).b(2, aw.k_norm.off).b(3, qr)
                    .b(4, kv.k).b(5, kv.v).b(6, kv.ksb).b(7, kv.vsb).b(8, io.invf).b(9, sq.ctrl)
                    .u(10, kv.max_ctx as u32).f(11, cfg.eps).u(12, nkv as u32).u(13, cfg.rope_dims as u32)
                    .groups((nh + nkv, r), hdim);
                let lim = if kv.q8 { 0 } else { sdpa_vec_max_n().min(8192) };
                let (n_lo, n_hi) = (io.pos + 1, io.pos + r);
                if n_hi > lim {
                    let (pn, nt, g) = split_k_kernel(r, kv.q8, nh / nkv);
                    e.pso(pn)
                        .b(0, qr).b(1, kv.k).b(2, kv.v).b(3, kv.ksb).b(4, kv.vsb).b(5, io.po).b(6, io.pml).b(7, sq.ctrl)
                        .u(8, kv.max_ctx as u32).u(9, (nh / nkv) as u32)
                        .groups((nch, nh / g), nt);
                    e.pso("attn_reduce_gate").b(0, io.po).b(1, io.pml).b(2, p).b(3, ao).u(4, nch as u32).u(5, prow as u32)
                        .groups((nh, r), hdim);
                }
                if n_lo <= lim {
                    e.pso("attn_sdpa_vec_b16")
                        .b(0, qr).b(1, kv.k).b(2, kv.v).b(3, p).b(4, ao).b(5, sq.ctrl)
                        .u(6, kv.max_ctx as u32).u(7, (nh / nkv) as u32).u(8, prow as u32).u(9, lim as u32)
                        .u(10, sdpa_two_pass_min() as u32)
                        .groups((nh, r), 1024);
                }
            }
            mm(e, io, &aw.o_proj, k, 2, io.ao, io.h, None);
        }
    }
    rmsnorm(e, cfg, k, io.h, lw.post_norm.off, io.x);
    let ff = &lw.ffn;
    mm_swiglu(e, io, &ff.gate, &ff.up, k, io.x, io.f, None);
    mm(e, io, &ff.down, k, 2, io.f, io.h, None);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Batched exact passes (batch.rs, 2026-11-04): one pass carries several sequences, each a contiguous block of rows.
// Norms, projections, FFN and the head run once over all rows (the same K-invariant matmuls as any exact pass, so
// each row is bit-identical to the row of a pass of that sequence alone); the token mixers (DeltaNet conv / delta
// rule with the row-0 checkpoint and replay save, attention prep + sdpa) and argmax run per sequence on that
// sequence's own state, KV, ctrl and rec table (`encode_dn_mix` / `encode_attn_mix`, the very kernels of
// `encode_layer`). No read mask, no hold, no dumps.

/// One sequence's block of rows in a batched pass.
#[derive(Clone, Copy, Debug)]
pub struct SeqRows<'a> {
    /// first row of the block and its rows (blocks are contiguous and in order)
    pub row0: usize,
    pub k: usize,
    /// host copy of this sequence's ctrl.pos (attention path choice) and the attention chunks covering pos + k
    pub pos: usize,
    pub nch: usize,
    /// arena offsets of this sequence's ctrl block and rec table
    pub ctrl: usize,
    pub rec: usize,
    pub dn: DnState,
    /// KV per model layer (`kv`) and of the MTP layer (`mtp_kv`, used when the layer index is cfg.layers)
    pub kv: &'a [Option<KvRef>],
    pub mtp_kv: Option<KvRef>,
    /// deep spec replay save of this sequence (rows 1..k-2), and the row-0 DeltaNet checkpoint
    pub dn_save: Option<usize>,
    pub ck: bool,
    /// this sequence's banned-id list (`PassIo::ban`); applied by `encode_head_batch`
    pub ban: Option<usize>,
}

/// `io` narrowed to one sequence's block (row offsets of every per-row scratch; the mixer row widths of the kind).
fn seq_io(cfg: &QwenConfig, io: &PassIo, s: &SeqRows, dn: bool) -> PassIo {
    let (prow, cvw) = if dn { (cfg.dn_in_rows(), cfg.conv_dim()) } else { (cfg.attn_in_rows(), cfg.q_dim()) };
    let o = s.row0;
    let hb = cfg.hidden * BF;
    PassIo { k: s.k, pos: s.pos, nch_max: s.nch, ctrl: s.ctrl, rec: s.rec, dn_save: s.dn_save, dn_hold: None,
             h: io.h + o * hb, x: io.x + o * hb, hf: io.hf + o * hb, p: io.p + o * prow * BF, cv: io.cv + o * cvw * BF,
             g: io.g + o * cfg.dn_vh * F32, beta: io.beta + o * cfg.dn_vh * BF, y: io.y + o * cfg.dn_v_dim() * BF,
             yn: io.yn + o * cfg.dn_v_dim() * BF, ao: io.ao + o * cfg.q_dim() * BF, ..*io }
}

fn check_blocks(seqs: &[SeqRows], k: usize) -> Result<(), String> {
    let mut r = 0;
    for s in seqs {
        if s.row0 != r || s.k == 0 { return Err(format!("batch: block at row {} with {} rows, expected row {}", s.row0, s.k, r)); }
        check_k(s.k)?;
        r += s.k;
    }
    if r != k { return Err(format!("batch: blocks cover {} rows, pass has {}", r, k)); }
    Ok(())
}

/// One decoder layer over the `io.k` rows of a batched pass (see above). `l` = model layer index (cfg.layers = the
/// MTP layer: its attention uses each block's `mtp_kv`).
pub fn encode_layer_batch(e: &Enc, cfg: &QwenConfig, lw: &LayerW, seqs: &[SeqRows], io: &PassIo, l: usize) -> Result<(), String> {
    let k = io.k;
    check_k(k)?;
    check_blocks(seqs, k)?;
    if io.rm.is_some() || io.dn_hold.is_some() { return Err("batch: no read mask / hold".into()); }
    if l == 0 { crate::activity::encode_begin(e, cfg, io); }
    rmsnorm(e, cfg, k, io.h, lw.in_norm.off, io.x);
    match &lw.mixer {
        Mixer::Dn(dw) => {
            if cfg.kinds.get(l) != Some(&LayerKind::DeltaNet) {
                return Err(format!("batch: layer {} has DeltaNet weights but kind {:?}", l, cfg.kinds.get(l)));
            }
            let dn = cfg.kinds[..l].iter().filter(|x| **x == LayerKind::DeltaNet).count();
            mm(e, io, &dw.in_proj, k, 0, io.x, io.p, None);
            for s in seqs { encode_dn_mix(e, cfg, dw, &s.dn, &seq_io(cfg, io, s, true), s.ck, None, l, dn)?; }
            mm(e, io, &dw.out_proj, k, 2, io.yn, io.h, None);
        }
        Mixer::Attn(aw) => {
            mm(e, io, &aw.in_proj, k, 0, io.x, io.p, None);
            for s in seqs {
                let kv = if l >= cfg.layers { s.mtp_kv } else { s.kv.get(l).copied().flatten() };
                let kv = kv.ok_or_else(|| format!("batch: attention layer {} needs a KvRef", l))?;
                encode_attn_mix(e, cfg, aw, &kv, &seq_io(cfg, io, s, false), None, l)?;
            }
            mm(e, io, &aw.o_proj, k, 2, io.ao, io.h, None);
        }
    }
    encode_ffn(e, cfg, lw, io, None, l)
}

/// Head of a batched pass: hf = final_norm(h) on all rows, lm_head on all rows, argmax_b per block into that
/// sequence's ctrl.tok / rec[ctrl.pos + t] (as `encode_head` with rows = k for each sequence alone).
pub fn encode_head_batch(e: &Enc, cfg: &QwenConfig, w: &Weights, io: &PassIo, seqs: &[SeqRows]) -> Result<(), String> {
    let k = io.k;
    check_k(k)?;
    check_blocks(seqs, k)?;
    rmsnorm(e, cfg, k, io.h, w.final_norm.off, io.hf);
    mm(e, io, &w.lm_head, k, 3, io.hf, io.logits, None);
    for s in seqs {
        if let Some(b) = s.ban {
            e.pso("ban_logits").b(0, io.logits + s.row0 * w.lm_head.rows * F32).b(1, b).u(2, w.lm_head.rows as u32)
                .groups((s.k, 1), 256);
        }
        e.pso("argmax_b").b(0, io.logits + s.row0 * w.lm_head.rows * F32).b(1, s.ctrl).b(2, s.rec).u(3, w.lm_head.rows as u32)
            .groups((s.k, 1), 1024);
    }
    Ok(())
}

#[cfg(test)]
mod straddle_tests {
    use super::AttnStraddle;
    #[test]
    fn attn_straddle_counts() {
        let (b, lim) = (super::sdpa_two_pass_min(), super::sdpa_vec_max_n().min(8192));
        let mut s = AttnStraddle::default();
        s.note(b - 2, 2, false);   // N = b-1, b: straddles b
        s.note(b - 1, 8, false);   // N = b..: no
        s.note(lim - 1, 2, false); // N = lim, lim+1: straddles lim; split-K row lim+1 only: no chunk edge
        s.note(lim + 250, 8, false); // N = lim+251..lim+258: split-K rows cross a 256 edge (lim % 256 == 0)
        s.note(250, 8, true);      // q8: N = 251..258 cross the 256 edge
        s.note(0, 4, true);        // q8: N = 1..4 same chunk
        let want_chunk = if lim % 256 == 0 { 2 } else { s.chunk };
        assert_eq!((s.passes, s.two_pass, s.split_k, s.chunk), (6, 1, 1, want_chunk));
    }
}
