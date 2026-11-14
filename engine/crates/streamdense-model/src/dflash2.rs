//! DFlash2 block drafter (2026-11-05): the z-lab drafter for this target, one forward per round proposes B - 1 tokens,
//! a candidate selector walks the top-k of each position; the target checks [pending, D1..D(B-1)] in one exact verify
//! pass of K = B rows (<= 8). Every emitted token is the verify pass's argmax, so the output is plain greedy's whatever
//! the drafts (lossless). Off unless `QwenOptions.dflash2` names a drafter directory. Reference: z-lab dflash
//! model_mlx.py (`DFlash2DraftModel`). File: the MLX 4-bit export (config.json +
//! model.safetensors, 179 tensors), read by its own tensor names.
//!
//! Drafter:
//!   taps     the target's residual stream after 0-based layers target_layer_ids[i], per position (no tap norm);
//!   context  feature = hidden_norm(fc(concat taps)); every draft layer turns it into its own K / V (k_proj / v_proj,
//!            k_norm, RoPE at the context position; the drafter's own table: all head_dim dims rotate, the target's
//!            are partial) = the context KV, one row per target position, in a ring;
//!   block    rows = [embed(pending), embed(mask token) x (B - 1)] at positions a .. a + B - 1, embeddings from the
//!            TARGET's table (0 extra bytes); n_layers plain sliding-attention layers (no gate), each with a grouped
//!            dynamic convolution around attention and around the MLP:
//!              x = input_layernorm(h); d = attention_conv.kernel_projection(x);
//!              h += conv(attn(conv(x, d[0], base[0])), d[1], base[1]);  the same with post_attention_layernorm + MLP;
//!            attention keys: context positions p with a + t - p < window for block row t, and the whole block
//!            (bidirectional, is_causal false); final norm and the TARGET's lm_head on rows 1 .. B - 1;
//!   selector candidate_selector (dflash2_select.rs, CPU) walks positions 1 .. B - 1 from the anchor token -> drafts.
//! Engine side:
//!   capture  only with this drafter loaded: every pass through `encode_pass_full` or this module's verify copies the
//!            tap rows (`DFlash2::tap`) and, at the end of the same command buffer, turns them into context KV rows
//!            (`DFlash2::absorb`). Rows past the accepted ones leave stale rows at positions >= the next anchor, which
//!            are never read and which the next pass overwrites.
//!   ring     rows per head = min(max_ctx, window - 1 + B - 1): the window's window - 1 rows, plus the B - 1 stale rows a
//!            verify pass may write past the next anchor (without them a rejected draft would have overwritten the
//!            oldest keys of the next round). `ctx_lo` tracks the oldest position still present.
//!   loop     `feed_stream_dflash2` (the `feed_stream` contract), as `feed_stream_dflash`: draft CB + selector ->
//!            drafts; verify = the context-draft HOLD pass of K = d + 1 rows + taps + absorb; decide j (longest prefix
//!            D_{i+1} == a_i, a_i not eos); emit a_0 .. a_je; replay rows 0 .. je into the DeltaNet state; pending = a_je.
//!   memory   forward weights (file tensors without the selector's) + ring KV (n_layers x 2 x kv_heads x ring x head_dim
//!            bf16) + scratch (tap rows for tmax rows, block rows, B - 1 logits and hidden rows, RoPE table) in the arena;
//!            selector `SelCfg::bytes` on the heap (its codebooks are read per pick, not held). All budget rows.
//! Not covered (drafts still lossless, only weaker): passes that bypass `encode_pass_full` (streamed prefill, snapshot
//! restore, the MTP / context-draft loops) leave no context rows (`Df2Stats::short_ctx` counts such rounds).
use std::cell::Cell;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use streamdense_core::arena::{Arena, Region};
use streamdense_core::{bf16_to_f32, DType, SafeIndex};
use streamdense_metal::{cb_signal, cb_wait, commit, gpu_ms, wait, CmdBuf, Enc, Gpu};

use crate::dflash::DraftHook;
use crate::dflash2_select::{SelCfg, Selector};
use crate::qwen_config::QwenConfig;
use crate::qwen_engine::{QwenEngine, StreamStats};
use crate::qwen_layers::{encode_head, encode_layer, gemv_m, swiglu_m, PassIo};
use crate::weights::Q4b;

const BF: usize = 2;
pub const CONFIG_FILE: &str = "config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const ARCHITECTURE: &str = "DFlash2DraftModel";
/// Largest block: the verify pass has at most 8 rows (EXACT_KS, logits scratch).
pub const BLOCK_MAX: usize = crate::dflash::BLOCK_MAX;
/// Rows per absorb matmul (the K-row GEMVs take 1..=8 rows).
const ABSORB_ROWS: usize = 8;

/// The drafter's config (the file's config.json, refused on anything this forward does not implement).
#[derive(Clone, Debug, PartialEq)]
pub struct Df2Config {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub vocab: usize,
    pub n_layers: usize,
    pub block: usize,
    pub mask_token: u32,
    pub window: usize,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub sel_rank: usize,
    pub sel_top_k: usize,
    pub num_target_layers: usize,
    /// HF hidden_states indices as in `DfConfig` (output of target layer i - 1): the file's target_layer_ids + 1
    pub layer_ids: Vec<usize>,
}
impl Df2Config {
    pub fn q_dim(&self) -> usize { self.heads * self.head_dim }
    pub fn kv_dim(&self) -> usize { self.kv_heads * self.head_dim }
    /// rotary dims: all of a head (the target's are partial, so this table is not the target's)
    pub fn rot_dims(&self) -> usize { self.head_dim }
    pub fn n_taps(&self) -> usize { self.layer_ids.len() }
    /// fc input width
    pub fn fw(&self) -> usize { self.n_taps() * self.hidden }
    /// target layer after which tap i is taken
    pub fn tap_layer(&self, i: usize) -> usize { self.layer_ids[i] - 1 }
    pub fn conv_groups(&self) -> usize { self.hidden / self.conv_group }
    /// kernel_projection rows: [2][K][groups]
    pub fn kern_rows(&self) -> usize { 2 * self.conv_kernel * self.conv_groups() }
    /// context KV rows per head
    pub fn ring_rows(&self, max_ctx: usize) -> usize { max_ctx.min(self.window - 1 + self.block - 1).max(1) }

    pub fn from_json(v: &serde_json::Value) -> Result<Df2Config, String> {
        let err = |m: String| format!("dflash2 config.json: {}", m);
        let u = |x: &serde_json::Value, k: &str| x.get(k).and_then(|y| y.as_u64()).map(|y| y as usize).ok_or_else(|| err(format!("no integer {:?}", k)));
        let f = |x: &serde_json::Value, k: &str| x.get(k).and_then(|y| y.as_f64()).ok_or_else(|| err(format!("no number {:?}", k)));
        let arch: Vec<&str> = v.get("architectures").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        if arch != [ARCHITECTURE] { return Err(err(format!("architectures {:?}, want [{:?}]", arch, ARCHITECTURE))); }
        let want_bool = |k: &str, w: bool| -> Result<(), String> {
            if v.get(k).and_then(|x| x.as_bool()) != Some(w) { return Err(err(format!("{} must be {}", k, w))); }
            Ok(())
        };
        want_bool("is_causal", false)?;
        want_bool("use_sliding_window", true)?;
        want_bool("attention_bias", false)?;
        if v.get("hidden_act").and_then(|x| x.as_str()) != Some("silu") { return Err(err("hidden_act must be silu".into())); }
        if v.get("rope_scaling").is_some_and(|x| !x.is_null()) { return Err(err("rope_scaling is not supported".into())); }
        let rp = v.get("rope_parameters").ok_or_else(|| err("no rope_parameters".into()))?;
        if rp.get("rope_type").and_then(|x| x.as_str()) != Some("default") { return Err(err("rope_type must be default".into())); }
        let q = v.get("quantization").ok_or_else(|| err("no quantization (a 4-bit g64 affine file is required)".into()))?;
        if u(q, "bits")? != 4 || u(q, "group_size")? != 64 || q.get("mode").and_then(|x| x.as_str()) != Some("affine") {
            return Err(err(format!("quantization {}, want 4 bits, group 64, affine", q)));
        }
        let d = v.get("dflash_config").ok_or_else(|| err("no dflash_config".into()))?;
        // z-lab's optional knobs: only their neutral values are implemented
        for (x, k, neutral) in [(d, "input_embedding_scale", 1.0), (d, "output_multiplier", 1.0), (d, "final_logit_softcapping", 0.0), (v, "final_logit_softcapping", 0.0)] {
            if let Some(y) = x.get(k).filter(|y| !y.is_null()) {
                if y.as_f64() != Some(neutral) { return Err(err(format!("{} = {} is not supported", k, y))); }
            }
        }
        let n_layers = u(v, "num_hidden_layers")?;
        let lt = v.get("layer_types").and_then(|x| x.as_array()).ok_or_else(|| err("no layer_types".into()))?;
        if lt.len() != n_layers || lt.iter().any(|x| x.as_str() != Some("sliding_attention")) {
            return Err(err(format!("layer_types {:?}, want {} x sliding_attention", lt, n_layers)));
        }
        let layer_ids: Vec<usize> = d.get("target_layer_ids").and_then(|x| x.as_array()).ok_or_else(|| err("no target_layer_ids".into()))?
            .iter().map(|x| x.as_u64().map(|y| y as usize + 1).ok_or_else(|| err("target_layer_ids not integers".into()))).collect::<Result<_, _>>()?;
        Ok(Df2Config { hidden: u(v, "hidden_size")?, heads: u(v, "num_attention_heads")?, kv_heads: u(v, "num_key_value_heads")?, head_dim: u(v, "head_dim")?,
                       inter: u(v, "intermediate_size")?, eps: f(v, "rms_norm_eps")? as f32, rope_theta: f(rp, "rope_theta")?, vocab: u(v, "vocab_size")?,
                       n_layers, block: u(d, "block_size")?, mask_token: u(d, "mask_token_id")? as u32, window: u(v, "sliding_window")?,
                       conv_kernel: u(d, "conv_kernel_size")?, conv_group: u(d, "conv_group_size")?, sel_rank: u(d, "selector_rank")?,
                       sel_top_k: u(d, "selector_top_k")?, num_target_layers: u(v, "num_target_layers")?, layer_ids })
    }
    /// The config.json fields `from_json` reads (`write_random`).
    pub fn to_json(&self) -> serde_json::Value {
        let ids: Vec<usize> = self.layer_ids.iter().map(|i| i - 1).collect();
        serde_json::json!({"architectures": [ARCHITECTURE], "attention_bias": false, "is_causal": false, "head_dim": self.head_dim,
            "dflash_config": {"block_size": self.block, "conv_group_size": self.conv_group, "conv_kernel_size": self.conv_kernel, "mask_token_id": self.mask_token,
                              "selector_rank": self.sel_rank, "selector_top_k": self.sel_top_k, "target_layer_ids": ids},
            "hidden_act": "silu", "hidden_size": self.hidden, "intermediate_size": self.inter, "layer_types": vec!["sliding_attention"; self.n_layers],
            "num_attention_heads": self.heads, "num_hidden_layers": self.n_layers, "num_key_value_heads": self.kv_heads, "num_target_layers": self.num_target_layers,
            "rms_norm_eps": self.eps, "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"}, "sliding_window": self.window,
            "use_sliding_window": true, "vocab_size": self.vocab, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}})
    }
    /// Shapes the engine kernels and the target need.
    pub fn check(&self, t: &QwenConfig) -> Result<(), String> {
        let e = |m: String| Err(format!("dflash2: {}", m));
        if self.hidden != t.hidden { return e(format!("hidden {} != target {}", self.hidden, t.hidden)); }
        if self.vocab != t.vocab { return e(format!("vocab {} != target {} (the target's embeddings and lm_head are shared)", self.vocab, t.vocab)); }
        if self.num_target_layers != t.layers { return e(format!("num_target_layers {} != target {}", self.num_target_layers, t.layers)); }
        if self.mask_token as usize >= self.vocab { return e(format!("mask token {} >= vocab {}", self.mask_token, self.vocab)); }
        if !(2..=BLOCK_MAX).contains(&self.block) { return e(format!("block {} not in 2..={}", self.block, BLOCK_MAX)); }
        if self.window <= self.block { return e(format!("sliding_window {} <= block {}", self.window, self.block)); }
        if self.head_dim % 32 != 0 || self.head_dim > 256 { return e(format!("head_dim {}: want a multiple of 32, <= 256", self.head_dim)); }
        if self.kv_heads == 0 || self.heads % self.kv_heads != 0 { return e(format!("heads {} / kv_heads {}", self.heads, self.kv_heads)); }
        if self.n_layers == 0 || self.layer_ids.is_empty() { return e("no layers / no taps".into()); }
        if let Some(&i) = self.layer_ids.iter().find(|&&i| i == 0 || i > t.layers) {
            return e(format!("tap after layer {} not in 0..{}", i as i64 - 1, t.layers));
        }
        if self.conv_kernel != 2 { return e(format!("conv_kernel_size {}, this forward was checked for 2 taps", self.conv_kernel)); }
        if self.conv_group == 0 || self.hidden % self.conv_group != 0 { return e(format!("conv_group_size {} does not divide hidden {}", self.conv_group, self.hidden)); }
        if !(self.rope_theta > 0.0) { return e(format!("rope_theta {}", self.rope_theta)); }
        if self.sel_top_k == 0 || self.sel_top_k > self.vocab { return e(format!("selector_top_k {} vs vocab {}", self.sel_top_k, self.vocab)); }
        for (n, x) in [("hidden", self.hidden), ("intermediate", self.inter), ("q_dim", self.q_dim()), ("selector_rank", self.sel_rank)] {
            if x == 0 || x % 64 != 0 { return e(format!("{} {} not a multiple of 64 (4-bit g64)", n, x)); }
        }
        Ok(())
    }
    pub fn sel_cfg(&self) -> SelCfg { SelCfg { vocab: self.vocab, hidden: self.hidden, rank: self.sel_rank, top_k: self.sel_top_k, max_rows: self.block - 1 } }
}

#[derive(Clone, Copy, Debug)]
pub struct Df2Conv {
    pub proj: Q4b,
    /// bf16 [2][K][hidden]
    pub base: Region,
}

#[derive(Clone, Copy, Debug)]
pub struct Df2Layer {
    pub in_norm: Region,
    pub post_norm: Region,
    pub a_conv: Df2Conv,
    pub m_conv: Df2Conv,
    pub q: Q4b,
    pub k: Q4b,
    pub v: Q4b,
    pub q_norm: Region,
    pub k_norm: Region,
    pub o: Q4b,
    pub gate: Q4b,
    pub up: Q4b,
    pub down: Q4b,
    /// context KV of this layer: bf16 [kv_heads][ring][head_dim]
    pub kc: Region,
    pub vc: Region,
}

/// Header-only plan of a drafter directory (budget rows before anything is allocated).
pub struct Df2Plan {
    pub dir: PathBuf,
    pub cfg: Df2Config,
    idx: SafeIndex,
    /// the forward's tensors: read into the arena. The selector's are checked here, read by dflash2_select.rs.
    weight_bytes: usize,
    n_tensors: usize,
}

type TensorList = Vec<(String, DType, Vec<usize>)>;

impl Df2Plan {
    pub fn read(dir: &Path, t: &QwenConfig) -> Result<Df2Plan, String> {
        let cf = dir.join(CONFIG_FILE);
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cf).map_err(|e| format!("{}: {}", cf.display(), e))?)
            .map_err(|e| format!("{}: {}", cf.display(), e))?;
        let cfg = Df2Config::from_json(&v)?;
        cfg.check(t)?;
        let wf = dir.join(WEIGHTS_FILE);
        let idx = SafeIndex::open_file(&wf.display().to_string())?;
        let mut p = Df2Plan { dir: dir.to_path_buf(), cfg, idx, weight_bytes: 0, n_tensors: 0 };
        let want = p.tensors();
        for (name, dt, shape) in &want {
            let ti = p.idx.get(name).ok_or_else(|| format!("dflash2: no tensor {}", name))?;
            if ti.dtype != *dt || ti.shape != *shape {
                return Err(format!("dflash2: {} is {:?} {:?}, want {:?} {:?}", name, ti.dtype, ti.shape, dt, shape));
            }
            if !name.starts_with("candidate_selector.") { p.weight_bytes += ti.nbytes; p.n_tensors += 1; }
        }
        // a tensor this forward would ignore means another model than the one it was checked against
        if let Some(extra) = p.idx.names().find(|n| !want.iter().any(|w| w.0 == *n)) { return Err(format!("dflash2: unexpected tensor {}", extra)); }
        if let Some(x) = p.idx.skipped.first() { return Err(format!("dflash2: unexpected tensor {} (dtype not readable)", x.0)); }
        Ok(p)
    }

    /// Every tensor of the file: (name, dtype, shape).
    fn tensors(&self) -> TensorList {
        let c = &self.cfg;
        let h = c.hidden;
        let mut v: TensorList = Vec::new();
        let q4 = |v: &mut TensorList, n: &str, rows: usize, cols: usize| {
            v.push((format!("{}.weight", n), DType::U32, vec![rows, cols / 8]));
            v.push((format!("{}.scales", n), DType::BF16, vec![rows, cols / 64]));
            v.push((format!("{}.biases", n), DType::BF16, vec![rows, cols / 64]));
        };
        let vecn = |v: &mut TensorList, n: String, len: usize| v.push((n, DType::BF16, vec![len]));
        q4(&mut v, "fc", h, c.fw());
        vecn(&mut v, "hidden_norm.weight".into(), h);
        vecn(&mut v, "norm.weight".into(), h);
        for l in 0..c.n_layers {
            let p = format!("layers.{}", l);
            vecn(&mut v, format!("{}.input_layernorm.weight", p), h);
            vecn(&mut v, format!("{}.post_attention_layernorm.weight", p), h);
            for cv in ["attention_conv", "mlp_conv"] {
                v.push((format!("{}.{}.base_kernel", p, cv), DType::BF16, vec![2, c.conv_kernel, h]));
                q4(&mut v, &format!("{}.{}.kernel_projection", p, cv), c.kern_rows(), h);
            }
            q4(&mut v, &format!("{}.self_attn.q_proj", p), c.q_dim(), h);
            q4(&mut v, &format!("{}.self_attn.k_proj", p), c.kv_dim(), h);
            q4(&mut v, &format!("{}.self_attn.v_proj", p), c.kv_dim(), h);
            vecn(&mut v, format!("{}.self_attn.q_norm.weight", p), c.head_dim);
            vecn(&mut v, format!("{}.self_attn.k_norm.weight", p), c.head_dim);
            q4(&mut v, &format!("{}.self_attn.o_proj", p), h, c.q_dim());
            q4(&mut v, &format!("{}.mlp.gate_proj", p), c.inter, h);
            q4(&mut v, &format!("{}.mlp.up_proj", p), c.inter, h);
            q4(&mut v, &format!("{}.mlp.down_proj", p), h, c.inter);
        }
        q4(&mut v, "candidate_selector.hidden_projection", c.sel_rank, h);
        q4(&mut v, "candidate_selector.predecessor_codebook", c.vocab, c.sel_rank);
        q4(&mut v, "candidate_selector.successor_codebook", c.vocab, c.sel_rank);
        v
    }

    fn kv_bytes(&self, max_ctx: usize) -> usize { self.cfg.n_layers * 2 * self.cfg.kv_heads * self.cfg.ring_rows(max_ctx) * self.cfg.head_dim * BF }

    /// Scratch regions (name, bytes) for passes of up to `tmax` rows.
    fn scratch(&self, tmax: usize) -> Vec<(&'static str, usize)> {
        let c = &self.cfg;
        let (b, h, d) = (c.block, c.hidden, c.block - 1);
        vec![
            ("taps", tmax * c.fw() * BF), ("feat", ABSORB_ROWS * h * BF), ("ctx", ABSORB_ROWS * h * BF),
            ("kr", ABSORB_ROWS * c.kv_dim() * BF), ("vr", ABSORB_ROWS * c.kv_dim() * BF),
            ("h", b * h * BF), ("x", b * h * BF), ("xc", b * h * BF), ("ar", b * h * BF), ("dy", b * c.kern_rows() * BF),
            ("q", b * c.q_dim() * BF), ("qr", b * c.q_dim() * BF), ("ao", b * c.q_dim() * BF), ("f", b * c.inter * BF),
            ("kb", b * c.kv_dim() * BF), ("vb", b * c.kv_dim() * BF),
            ("hf", d * h * BF), ("logits", d * c.vocab * 4), ("hf32", d * h * 4), ("invf", c.rot_dims() / 2 * 4),
        ]
    }

    /// Budget rows of a drafter for passes of up to `tmax` rows.
    pub fn budget_rows(&self, tmax: usize, max_ctx: usize) -> Vec<(String, u64)> {
        let sc: usize = self.scratch(tmax).iter().map(|x| x.1).sum();
        vec![
            (format!("DFlash2 drafter weights ({} layers, 4-bit, {})", self.cfg.n_layers, self.dir.display()), self.weight_bytes as u64),
            ("DFlash2 candidate selector (projection + pick scratch; codebooks read per pick, not held)".to_string(), self.cfg.sel_cfg().bytes()),
            (format!("DFlash2 context KV bf16 ring, {} rows per head @ window {}", self.cfg.ring_rows(max_ctx), self.cfg.window), self.kv_bytes(max_ctx) as u64),
            ("DFlash2 scratch (tap rows, block rows, logits)".to_string(), sc as u64),
        ]
    }
    /// Arena regions it allocates (for the alignment slack row).
    pub fn regions(&self) -> usize { self.n_tensors + self.scratch(1).len() + 2 * self.cfg.n_layers }
}

/// Statistics of one `feed_stream_dflash2` (or summed: `add`).
#[derive(Clone, Debug, Default)]
pub struct Df2Stats {
    /// verify passes (rounds), drafts verified / accepted, passes with a rejected draft
    pub passes: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub rejects: usize,
    /// tokens emitted by the rounds (not the prefill's first token)
    pub tokens: usize,
    /// accept_hist[j] = passes with j drafts accepted
    pub accept_hist: [usize; BLOCK_MAX],
    /// GPU ms: drafter forwards, verify passes, replays
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub replay_ms: f64,
    /// CPU ms: candidate selector walks
    pub select_ms: f64,
    /// selector positions that fell back to logits only (codebook read failed); drafts stay valid, only weaker
    pub select_io_errors: u64,
    /// rounds whose drafter context was shorter than the anchor (positions fed by passes without capture)
    pub short_ctx: usize,
}
impl Df2Stats {
    pub fn add(&mut self, o: &Df2Stats) {
        self.passes += o.passes; self.drafted += o.drafted; self.accepted += o.accepted; self.rejects += o.rejects; self.tokens += o.tokens;
        for i in 0..BLOCK_MAX { self.accept_hist[i] += o.accept_hist[i]; }
        self.draft_ms += o.draft_ms; self.verify_ms += o.verify_ms; self.replay_ms += o.replay_ms; self.select_ms += o.select_ms; self.select_io_errors += o.select_io_errors; self.short_ctx += o.short_ctx;
    }
    pub fn acceptance(&self) -> f64 { if self.drafted > 0 { self.accepted as f64 / self.drafted as f64 } else { 0.0 } }
    pub fn tokens_per_pass(&self) -> f64 { if self.passes > 0 { self.tokens as f64 / self.passes as f64 } else { 0.0 } }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"passes": self.passes, "drafted": self.drafted, "accepted": self.accepted, "rejects": self.rejects,
            "tokens": self.tokens, "acceptance": self.acceptance(), "tokens_per_pass": self.tokens_per_pass(), "accept_hist": self.accept_hist,
            "draft_ms": self.draft_ms, "verify_ms": self.verify_ms, "replay_ms": self.replay_ms, "select_ms": self.select_ms, "select_io_errors": self.select_io_errors, "short_ctx": self.short_ctx})
    }
}

/// A loaded drafter: weights, context KV and scratch in the engine arena.
pub struct DFlash2 {
    pub cfg: Df2Config,
    fc: Q4b,
    hidden_norm: Region,
    norm: Region,
    pub layers: Vec<Df2Layer>,
    sel: Selector,
    taps: Region,
    feat: Region,
    ctx: Region,
    kr: Region,
    vr: Region,
    h: Region,
    x: Region,
    xc: Region,
    ar: Region,
    dy: Region,
    q: Region,
    qr: Region,
    ao: Region,
    f: Region,
    kb: Region,
    vb: Region,
    hf: Region,
    logits: Region,
    hf32: Region,
    invf: Region,
    ring: usize,
    max_ctx: usize,
    tmax: usize,
    /// contiguous prefix end of positions with context rows (host copy, updated at encode time = execution order)
    ctx_len: Cell<usize>,
    /// oldest position whose rows are still in the ring
    ctx_lo: Cell<usize>,
}
/// The brief's name for the drafter.
pub type Df2 = DFlash2;

#[repr(C)]
#[derive(Clone, Copy)]
struct Df2Prep { nq: u32, nkv: u32, hd: u32, rdims: u32, pos0: u32, kv_rows: u32, row0: u32, eps: f32 }
#[repr(C)]
#[derive(Clone, Copy)]
struct Df2Attn { nh: u32, nkv: u32, hd: u32, nb: u32, ring: u32, anchor: u32, ctx_end: u32, lo: u32, window: u32, scale: f32 }
#[repr(C)]
#[derive(Clone, Copy)]
struct Df2ConvArgs { h: u32, gs: u32, groups: u32, kern: u32, side: u32, add: u32 }

pub const PIPELINES: [&str; 4] = ["df2_copy_rows", "df2_prep", "df2_attn", "df2_conv"];
pub const SHADER: &str = "dflash2";

impl DFlash2 {
    /// Allocate + read the drafter (weights, context KV, scratch) into the engine arena; build its pipelines.
    pub fn load(plan: &Df2Plan, arena: &mut Arena, gpu: &mut Gpu, tmax: usize, max_ctx: usize) -> Result<DFlash2, String> {
        let c = plan.cfg.clone();
        let files: Vec<std::fs::File> = plan.idx.files().iter().map(|f| std::fs::File::open(f).map_err(|e| format!("{}: {}", f, e)))
            .collect::<Result<_, _>>()?;
        let rd = |arena: &mut Arena, name: &str| -> Result<Region, String> {
            let t = plan.idx.get(name).ok_or_else(|| format!("dflash2: no tensor {}", name))?;
            let r = arena.alloc(&format!("w:dflash2.{}", name), t.nbytes)?;
            let dst = arena.slice_mut::<u8>(r);
            files[t.file].read_exact_at(&mut dst[..t.nbytes], t.off).map_err(|e| format!("dflash2: read {}: {}", name, e))?;
            Ok(r)
        };
        let q4 = |arena: &mut Arena, n: &str, rows: usize, cols: usize| -> Result<Q4b, String> {
            Ok(Q4b { rows, cols, w: rd(arena, &format!("{}.weight", n))?, s: rd(arena, &format!("{}.scales", n))?, b: rd(arena, &format!("{}.biases", n))? })
        };
        let h = c.hidden;
        let fc = q4(arena, "fc", h, c.fw())?;
        let hidden_norm = rd(arena, "hidden_norm.weight")?;
        let norm = rd(arena, "norm.weight")?;
        let ring = c.ring_rows(max_ctx);
        let kvb = c.kv_heads * ring * c.head_dim * BF;
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            let p = format!("layers.{}", l);
            let conv = |arena: &mut Arena, n: &str| -> Result<Df2Conv, String> {
                Ok(Df2Conv { proj: q4(arena, &format!("{}.{}.kernel_projection", p, n), c.kern_rows(), h)?, base: rd(arena, &format!("{}.{}.base_kernel", p, n))? })
            };
            let in_norm = rd(arena, &format!("{}.input_layernorm.weight", p))?;
            let post_norm = rd(arena, &format!("{}.post_attention_layernorm.weight", p))?;
            let a_conv = conv(arena, "attention_conv")?;
            let m_conv = conv(arena, "mlp_conv")?;
            let qp = q4(arena, &format!("{}.self_attn.q_proj", p), c.q_dim(), h)?;
            let kp = q4(arena, &format!("{}.self_attn.k_proj", p), c.kv_dim(), h)?;
            let vp = q4(arena, &format!("{}.self_attn.v_proj", p), c.kv_dim(), h)?;
            let q_norm = rd(arena, &format!("{}.self_attn.q_norm.weight", p))?;
            let k_norm = rd(arena, &format!("{}.self_attn.k_norm.weight", p))?;
            let o = q4(arena, &format!("{}.self_attn.o_proj", p), h, c.q_dim())?;
            let gate = q4(arena, &format!("{}.mlp.gate_proj", p), c.inter, h)?;
            let up = q4(arena, &format!("{}.mlp.up_proj", p), c.inter, h)?;
            let down = q4(arena, &format!("{}.mlp.down_proj", p), h, c.inter)?;
            let kc = arena.alloc(&format!("kv:dflash2.{}.k", l), kvb)?;
            let vc = arena.alloc(&format!("kv:dflash2.{}.v", l), kvb)?;
            layers.push(Df2Layer { in_norm, post_norm, a_conv, m_conv, q: qp, k: kp, v: vp, q_norm, k_norm, o, gate, up, down, kc, vc });
        }
        let sel = Selector::load(&plan.dir, &c.sel_cfg())?;
        let mut s = std::collections::HashMap::new();
        for (n, b) in plan.scratch(tmax) {
            let r = arena.alloc(&format!("s:dflash2.{}", n), b)?;
            arena.slice_mut::<u8>(r).fill(0);
            s.insert(n, r);
        }
        // the drafter's own RoPE frequencies base^(-i / (dims / 2)); the target's table covers its partial dims only
        let half = c.rot_dims() / 2;
        let log2base = (c.rope_theta as f32).log2();
        for (i, f) in arena.slice_mut::<f32>(s["invf"])[..half].iter_mut().enumerate() { *f = (-(i as f32 / half as f32) * log2base).exp2(); }
        for n in PIPELINES { gpu.pipeline(n)?; }
        Ok(DFlash2 { cfg: c, fc, hidden_norm, norm, layers, sel, taps: s["taps"], feat: s["feat"], ctx: s["ctx"], kr: s["kr"], vr: s["vr"],
                     h: s["h"], x: s["x"], xc: s["xc"], ar: s["ar"], dy: s["dy"], q: s["q"], qr: s["qr"], ao: s["ao"], f: s["f"], kb: s["kb"], vb: s["vb"],
                     hf: s["hf"], logits: s["logits"], hf32: s["hf32"], invf: s["invf"], ring, max_ctx, tmax, ctx_len: Cell::new(0), ctx_lo: Cell::new(0) })
    }

    /// Positions 0 .. ctx_len() have context rows (a contiguous prefix), the ones >= ctx_lo() still present.
    pub fn ctx_len(&self) -> usize { self.ctx_len.get() }
    pub fn ctx_lo(&self) -> usize { self.ctx_lo.get() }
    pub fn ring_rows(&self) -> usize { self.ring }

    fn rmsnorm(&self, e: &Enc, rows: usize, x: usize, w: usize, out: usize) {
        e.pso("rmsnorm_b").b(0, x).b(1, w).b(2, out).u(3, self.cfg.hidden as u32).f(4, self.cfg.eps).u(5, self.cfg.hidden as u32)
            .groups((rows, 1), 1024);
    }
    fn copy_rows(e: &Enc, src: usize, dst: usize, n: usize, rows: usize, ss: usize, ds: usize) {
        e.pso("df2_copy_rows").b(0, src).b(1, dst).u(2, n as u32).u(3, ss as u32).u(4, ds as u32).threads((n, rows), 256);
    }

    /// After target layer `l` of a pass (`io.k` rows): copy the rows of every tap taken after `l` into the tap buffer
    /// [row][tap][hidden].
    pub(crate) fn tap(&self, e: &Enc, io: &PassIo, l: usize) {
        let c = &self.cfg;
        for i in 0..c.n_taps() {
            if c.tap_layer(i) == l { Self::copy_rows(e, io.h, self.taps.off + i * c.hidden * BF, c.hidden, io.k, c.hidden, c.fw()); }
        }
    }

    /// End of a pass of `k` rows at position `pos`: context rows for positions pos .. pos + k (fc + hidden_norm, then per
    /// draft layer k / v projection, k_norm + RoPE, into its ring), in blocks of <= 8 rows.
    pub(crate) fn absorb(&self, e: &Enc, io: &PassIo, pos: usize, k: usize) -> Result<(), String> {
        let c = &self.cfg;
        if k > self.tmax { return Err(format!("dflash2: pass of {} rows > tap rows {}", k, self.tmax)); }
        if pos + k > self.max_ctx { return Err(format!("dflash2: positions {}..{} > max_ctx {}", pos, pos + k, self.max_ctx)); }
        let fw = c.fw();
        let mut r0 = 0;
        while r0 < k {
            let kb = (k - r0).min(ABSORB_ROWS);
            gemv_m(e, io.gemv, &self.fc, kb, 0, self.taps.off + r0 * fw * BF, self.feat.off, None);
            self.rmsnorm(e, kb, self.feat.off, self.hidden_norm.off, self.ctx.off);
            for lw in &self.layers {
                gemv_m(e, io.gemv, &lw.k, kb, 0, self.ctx.off, self.kr.off, None);
                gemv_m(e, io.gemv, &lw.v, kb, 0, self.ctx.off, self.vr.off, None);
                let a = Df2Prep { nq: 0, nkv: c.kv_heads as u32, hd: c.head_dim as u32, rdims: c.rot_dims() as u32, pos0: (pos + r0) as u32,
                                  kv_rows: self.ring as u32, row0: (pos + r0) as u32, eps: c.eps };
                e.pso("df2_prep").b(0, self.kr.off).b(1, self.kr.off).b(2, self.vr.off).b(3, lw.k_norm.off).b(4, lw.k_norm.off)
                    .b(5, self.kr.off).b(6, lw.kc.off).b(7, lw.vc.off).b(8, self.invf.off).bytes(9, &a).groups((c.kv_heads, kb), c.head_dim);
            }
            r0 += kb;
        }
        // contiguous prefix: a pass at or before its end extends / truncates it; a pass past it leaves a gap. A rewind
        // below the ring's oldest position restarts the valid range at the rewind.
        if pos <= self.ctx_len.get() {
            self.ctx_len.set(pos + k);
            if pos < self.ctx_lo.get() { self.ctx_lo.set(pos); }
        }
        self.ctx_lo.set(self.ctx_lo.get().max((pos + k).saturating_sub(self.ring)));
        Ok(())
    }

    /// kernel_projection of `x` into `dy`, then conv side 0 of `x` into `xc`: z-lab `prepare`.
    fn conv_prepare(&self, e: &Enc, io: &PassIo, cv: &Df2Conv) {
        let b = self.cfg.block;
        gemv_m(e, io.gemv, &cv.proj, b, 0, self.x.off, self.dy.off, None);
        self.conv(e, cv, 0, 0, self.x.off, self.xc.off);
    }
    /// conv side 1 of `ar` added onto the residual `h`: z-lab `finish` + the layer's residual add.
    fn conv_finish(&self, e: &Enc, cv: &Df2Conv) { self.conv(e, cv, 1, 1, self.ar.off, self.h.off); }
    fn conv(&self, e: &Enc, cv: &Df2Conv, side: u32, add: u32, x: usize, out: usize) {
        let c = &self.cfg;
        let a = Df2ConvArgs { h: c.hidden as u32, gs: c.conv_group as u32, groups: c.conv_groups() as u32, kern: c.conv_kernel as u32, side, add };
        e.pso("df2_conv").b(0, x).b(1, self.dy.off).b(2, cv.base.off).b(3, out).bytes(4, &a).threads((c.hidden, c.block), 256);
    }

    /// Block forward at anchor position `a` (the embedding rows of the pending and the mask token staged in `stage`):
    /// final-norm hidden rows bf16 [B - 1][hidden] in `hf`, logits f32 [B - 1][vocab] in `logits`.
    fn encode_draft(&self, e: &Enc, io: &PassIo, lm_head: &Q4b, stage: usize, a: usize) -> Result<(), String> {
        let c = &self.cfg;
        let (b, h) = (c.block, c.hidden);
        e.pso("embed_q4").b(0, stage).b(1, self.h.off).u(2, 2);
        e.threads((h / 8, 2), 64);
        if b > 2 { Self::copy_rows(e, self.h.off + h * BF, self.h.off + 2 * h * BF, h, b - 2, 0, h); }
        let g = io.gemv;
        let at = Df2Attn { nh: c.heads as u32, nkv: c.kv_heads as u32, hd: c.head_dim as u32, nb: b as u32, ring: self.ring as u32, anchor: a as u32,
                           ctx_end: a.min(self.ctx_len.get()) as u32, lo: self.ctx_lo.get() as u32, window: c.window as u32,
                           scale: (c.head_dim as f32).powf(-0.5) };
        let pp = Df2Prep { nq: c.heads as u32, nkv: c.kv_heads as u32, hd: c.head_dim as u32, rdims: c.rot_dims() as u32, pos0: a as u32, kv_rows: b as u32, row0: 0, eps: c.eps };
        for lw in &self.layers {
            self.rmsnorm(e, b, self.h.off, lw.in_norm.off, self.x.off);
            self.conv_prepare(e, io, &lw.a_conv);
            gemv_m(e, g, &lw.q, b, 0, self.xc.off, self.q.off, None);
            gemv_m(e, g, &lw.k, b, 0, self.xc.off, self.kr.off, None);
            gemv_m(e, g, &lw.v, b, 0, self.xc.off, self.vr.off, None);
            e.pso("df2_prep").b(0, self.q.off).b(1, self.kr.off).b(2, self.vr.off).b(3, lw.q_norm.off).b(4, lw.k_norm.off)
                .b(5, self.qr.off).b(6, self.kb.off).b(7, self.vb.off).b(8, self.invf.off).bytes(9, &pp).groups((c.heads + c.kv_heads, b), c.head_dim);
            e.pso("df2_attn").b(0, self.qr.off).b(1, lw.kc.off).b(2, lw.vc.off).b(3, self.kb.off).b(4, self.vb.off).b(5, self.ao.off)
                .bytes(6, &at).groups((c.heads, b), 256);
            gemv_m(e, g, &lw.o, b, 0, self.ao.off, self.ar.off, None);
            self.conv_finish(e, &lw.a_conv);
            self.rmsnorm(e, b, self.h.off, lw.post_norm.off, self.x.off);
            self.conv_prepare(e, io, &lw.m_conv);
            swiglu_m(e, g, &lw.gate, &lw.up, b, self.xc.off, self.f.off, None);
            gemv_m(e, g, &lw.down, b, 0, self.f.off, self.ar.off, None);
            self.conv_finish(e, &lw.m_conv);
        }
        // z-lab slices the anchor row off before the final norm and the head
        self.rmsnorm(e, b - 1, self.h.off + h * BF, self.norm.off, self.hf.off);
        gemv_m(e, g, lm_head, b - 1, 3, self.hf.off, self.logits.off, None);
        Ok(())
    }
}

/// Write a random drafter in the file format for the target `t` (tests; deterministic in `seed`). Weights are random
/// 4-bit codes with per-group scale / bias of a ~1/sqrt(cols) spread; norms ~1; conv bases ~identity (tap 0 ~1, tap 1 ~0).
pub fn write_random(dir: &Path, t: &QwenConfig, n_layers: usize, block: usize, window: usize, sel_rank: usize, seed: u64) -> Result<Df2Config, String> {
    let n = 5.min(t.layers.max(1));
    let (lo, hi) = (2usize.min(t.layers), t.layers.saturating_sub(2).max(1));
    let layer_ids: Vec<usize> = (0..n).map(|i| (lo as f64 + i as f64 * (hi as f64 - lo as f64) / (n.max(2) - 1) as f64).round() as usize)
        .map(|x| x.clamp(1, t.layers)).collect();
    let c = Df2Config { hidden: t.hidden, heads: t.heads, kv_heads: t.kv_heads, head_dim: t.head_dim, inter: t.inter, eps: t.eps, rope_theta: t.rope_theta,
                        vocab: t.vocab, n_layers, block, mask_token: (t.vocab - 1) as u32, window, conv_kernel: 2, conv_group: 16, sel_rank,
                        sel_top_k: 16.min(t.vocab), num_target_layers: t.layers, layer_ids };
    c.check(t)?;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let bf = streamdense_core::f32_to_bf16;
    let mut owned: Vec<(String, DType, Vec<usize>, Vec<u8>)> = vec![];
    let q4 = |o: &mut Vec<(String, DType, Vec<usize>, Vec<u8>)>, rnd: &mut dyn FnMut() -> u64, n: &str, rows: usize, cols: usize| {
        let w: Vec<u8> = (0..rows * cols / 2).map(|_| rnd() as u8).collect();
        let a = 1.7 / (cols as f32).sqrt();
        let g = rows * cols / 64;
        let sc: Vec<u8> = (0..g).flat_map(|_| bf(a * (0.8 + 0.4 * (rnd() % 1000) as f32 / 1000.0) * 2.0 / 15.0).to_le_bytes()).collect();
        let bi: Vec<u8> = (0..g).flat_map(|_| bf(-a * (0.8 + 0.4 * (rnd() % 1000) as f32 / 1000.0)).to_le_bytes()).collect();
        o.push((format!("{}.weight", n), DType::U32, vec![rows, cols / 8], w));
        o.push((format!("{}.scales", n), DType::BF16, vec![rows, cols / 64], sc));
        o.push((format!("{}.biases", n), DType::BF16, vec![rows, cols / 64], bi));
    };
    let vecn = |o: &mut Vec<(String, DType, Vec<usize>, Vec<u8>)>, rnd: &mut dyn FnMut() -> u64, n: String, len: usize, base: f32, spread: f32| {
        let d: Vec<u8> = (0..len).flat_map(|_| bf(base + spread * ((rnd() % 2001) as f32 / 1000.0 - 1.0)).to_le_bytes()).collect();
        o.push((n, DType::BF16, vec![len], d));
    };
    let h = c.hidden;
    q4(&mut owned, &mut rnd, "fc", h, c.fw());
    vecn(&mut owned, &mut rnd, "hidden_norm.weight".into(), h, 1.0, 0.2);
    vecn(&mut owned, &mut rnd, "norm.weight".into(), h, 1.0, 0.2);
    for l in 0..n_layers {
        let p = format!("layers.{}", l);
        vecn(&mut owned, &mut rnd, format!("{}.input_layernorm.weight", p), h, 1.0, 0.2);
        vecn(&mut owned, &mut rnd, format!("{}.post_attention_layernorm.weight", p), h, 1.0, 0.2);
        for cv in ["attention_conv", "mlp_conv"] {
            let base: Vec<u8> = (0..2 * c.conv_kernel * h).flat_map(|i| {
                let tap = (i / h) % c.conv_kernel;
                bf(if tap == 0 { 1.0 } else { 0.0 } + 0.05 * ((rnd() % 2001) as f32 / 1000.0 - 1.0)).to_le_bytes()
            }).collect();
            owned.push((format!("{}.{}.base_kernel", p, cv), DType::BF16, vec![2, c.conv_kernel, h], base));
            q4(&mut owned, &mut rnd, &format!("{}.{}.kernel_projection", p, cv), c.kern_rows(), h);
        }
        q4(&mut owned, &mut rnd, &format!("{}.self_attn.q_proj", p), c.q_dim(), h);
        q4(&mut owned, &mut rnd, &format!("{}.self_attn.k_proj", p), c.kv_dim(), h);
        q4(&mut owned, &mut rnd, &format!("{}.self_attn.v_proj", p), c.kv_dim(), h);
        vecn(&mut owned, &mut rnd, format!("{}.self_attn.q_norm.weight", p), c.head_dim, 1.0, 0.2);
        vecn(&mut owned, &mut rnd, format!("{}.self_attn.k_norm.weight", p), c.head_dim, 1.0, 0.2);
        q4(&mut owned, &mut rnd, &format!("{}.self_attn.o_proj", p), h, c.q_dim());
        q4(&mut owned, &mut rnd, &format!("{}.mlp.gate_proj", p), c.inter, h);
        q4(&mut owned, &mut rnd, &format!("{}.mlp.up_proj", p), c.inter, h);
        q4(&mut owned, &mut rnd, &format!("{}.mlp.down_proj", p), h, c.inter);
    }
    q4(&mut owned, &mut rnd, "candidate_selector.hidden_projection", sel_rank, h);
    q4(&mut owned, &mut rnd, "candidate_selector.predecessor_codebook", c.vocab, sel_rank);
    q4(&mut owned, &mut rnd, "candidate_selector.successor_codebook", c.vocab, sel_rank);
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let views: Vec<(&str, DType, &[usize], &[u8])> = owned.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice())).collect();
    streamdense_core::write_safetensors(&dir.join(WEIGHTS_FILE).display().to_string(), &views)?;
    std::fs::write(dir.join(CONFIG_FILE), serde_json::to_string_pretty(&c.to_json()).unwrap()).map_err(|e| format!("{}: {}", dir.display(), e))?;
    Ok(c)
}

impl QwenEngine {
    /// A DFlash2 drafter is loaded (`QwenOptions.dflash2`).
    pub fn dflash2_on(&self) -> bool { self.dflash2.is_some() }
    pub fn dflash2_config(&self) -> Option<&Df2Config> { self.dflash2.as_ref().map(|d| &d.cfg) }
    /// Positions with drafter context rows (0 without a drafter).
    pub fn dflash2_ctx_len(&self) -> usize { self.dflash2.as_ref().map_or(0, |d| d.ctx_len()) }
    /// Oldest position whose context rows are still in the ring (tests).
    pub fn dflash2_ctx_lo(&self) -> usize { self.dflash2.as_ref().map_or(0, |d| d.ctx_lo()) }
    /// Context KV rows per head (tests).
    pub fn dflash2_ring_rows(&self) -> usize { self.dflash2.as_ref().map_or(0, |d| d.ring_rows()) }
    /// Tap rows bf16 [k][fw] left by the last pass of k rows (tests: reference check).
    pub fn dflash2_tap_rows(&self, k: usize) -> Vec<u16> {
        match &self.dflash2 { Some(d) => self.arena.slice_mut::<u16>(d.taps)[..k * d.cfg.fw()].to_vec(), None => vec![] }
    }
    /// Turn given tap rows bf16 [k][fw] into context rows at positions pos .. pos + k without a target pass, k <= the tap
    /// rows (tests: same inputs for the forward check against z-lab).
    pub fn dflash2_feed_taps(&self, pos: usize, taps: &[u16]) -> Result<(), String> {
        let d = self.df2()?;
        let k = taps.len() / d.cfg.fw();
        if k * d.cfg.fw() != taps.len() || k == 0 || k > d.tmax { return Err(format!("dflash2_feed_taps: {} values, {} rows max", taps.len(), d.tmax)); }
        self.arena.slice_mut::<u16>(d.taps)[..taps.len()].copy_from_slice(taps);
        let cb = self.enc_cb(&|e| d.absorb(e, &self.io(), pos, k))?;
        commit(&cb);
        wait(&cb)
    }

    fn df2(&self) -> Result<&DFlash2, String> { self.dflash2.as_ref().ok_or_else(|| "dflash2: no drafter loaded (QwenOptions.dflash2)".to_string()) }

    /// One drafter forward at anchor position `a` with pending token `t` (synchronous; staging slot 1) + the selector
    /// walk: (the B - 1 drafts, GPU ms, selector ms). Reads only the drafter's own state (the context rows < a); the
    /// target's KV / state are untouched.
    pub fn dflash2_draft(&self, a: usize, t: u32) -> Result<(Vec<u32>, f64, f64), String> {
        let d = self.df2()?;
        self.reader.read_rows(&[t, d.cfg.mask_token], self.stage_ptr(1))?;
        let cb = self.enc_cb(&|e| d.encode_draft(e, &self.io(), &self.w.lm_head, self.sl().stage.at(self.slot_bytes), a))?;
        commit(&cb);
        wait(&cb)?;
        let ms = gpu_ms(&cb);
        let n = d.cfg.block - 1;
        let tc = Instant::now();
        let hb = &self.arena.slice_mut::<u16>(d.hf)[..n * d.cfg.hidden];
        let hf = &mut self.arena.slice_mut::<f32>(d.hf32)[..n * d.cfg.hidden];
        for (o, &x) in hf.iter_mut().zip(hb) { *o = bf16_to_f32(x); }
        let lg = &self.arena.slice_mut::<f32>(d.logits)[..n * d.cfg.vocab];
        let dr = d.sel.pick_from(t, lg, hf, n);
        if dr.len() != n { return Err(format!("dflash2: selector returned {} drafts for {} rows", dr.len(), n)); }
        Ok((dr, ms, tc.elapsed().as_secs_f64() * 1e3))
    }
    /// Logits f32 [B - 1][vocab] of the last `dflash2_draft` (tests).
    pub fn dflash2_logits(&self) -> Vec<f32> {
        match &self.dflash2 { Some(d) => self.arena.slice_mut::<f32>(d.logits)[..(d.cfg.block - 1) * d.cfg.vocab].to_vec(), None => vec![] }
    }
    /// Final-norm hidden rows f32 [B - 1][hidden] of the last `dflash2_draft` (tests).
    pub fn dflash2_hidden(&self) -> Vec<f32> {
        match &self.dflash2 { Some(d) => self.arena.slice_mut::<f32>(d.hf32)[..(d.cfg.block - 1) * d.cfg.hidden].to_vec(), None => vec![] }
    }

    /// Verify pass of k rows staged in slot 0 at position p (as dflash.rs `encode_df_verify`): k = 1 the plain decode
    /// pass, k >= 2 a HOLD pass. Head + argmax on all rows: rec[p + i]. ctrl.pos not advanced.
    fn encode_df2_verify(&self, e: &Enc, d: &DFlash2, k: usize, p: usize) -> Result<(), String> {
        let c = &self.cfg;
        let hold = if k > 1 { Some(self.ctx_hold.ok_or("dflash2: no hold save")?.off) } else { None };
        let io = PassIo { k, pos: p, nch_max: self.nch_for(p, k), dn_save: None, dn_hold: hold, rm: self.rm_io(true), ..self.io() };
        e.pso("embed_q4").b(0, self.sl().stage.at(0)).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        for l in 0..c.layers {
            encode_layer(e, c, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, false, None, l)?;
            d.tap(e, &io, l);
        }
        encode_head(e, c, &self.w, &io, k, None)?;
        d.absorb(e, &io, p, k)
    }

    /// Streaming greedy decode with DFlash2 block drafts (`feed_stream` contract, as `feed_stream_dflash`): output ==
    /// `feed_stream` token for token, every emitted token is the target's own argmax. `ids.len() == 1`: that token is
    /// row 0 of the first verify pass (its context rows must exist: the earlier passes ran with the drafter loaded).
    pub fn feed_stream_dflash2(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool)
                               -> Result<(StreamStats, Df2Stats), String> {
        self.feed_stream_dflash2_ov(ids, n_new, stop, on_tok, None)
    }

    /// `feed_stream_dflash2` with a test hook that replaces the drafts of each round (the drafter still runs).
    pub fn feed_stream_dflash2_ov(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                                  hook: Option<DraftHook>) -> Result<(StreamStats, Df2Stats), String> {
        let d = self.df2()?;
        let hold = self.ctx_hold.ok_or("feed_stream_dflash2: no hold save")?.off;
        let mut st = StreamStats { stop_reason: "", ..Default::default() };
        let mut ds = Df2Stats::default();
        if ids.is_empty() { return Err("feed_stream_dflash2: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok((st, ds)); }
        let tp = Instant::now();
        let mut pending = ids[0];
        if ids.len() > 1 {
            let t = self.prefill(ids)?;
            st.gen = 1;
            st.last = Some(t);
            pending = t;
            if !on_tok(t) { st.stop_reason = "cancelled"; }
            else if self.is_eos(t) || stop.contains(&t) { st.stop_reason = "eos"; }
            else if st.gen >= n_new { st.stop_reason = "length"; }
        }
        st.prefill_s = tp.elapsed().as_secs_f64();
        if !st.stop_reason.is_empty() { return Ok((st, ds)); }
        let td = Instant::now();
        let io0 = d.sel.io_errors();
        let r = self.df2_loop(d, pending, hold, n_new, stop, on_tok, hook, &mut st, &mut ds);
        ds.select_io_errors = d.sel.io_errors() - io0;
        st.decode_s = td.elapsed().as_secs_f64();
        st.gpu_ms = ds.draft_ms + ds.verify_ms + ds.replay_ms;
        r?;
        if st.stop_reason.is_empty() { st.stop_reason = "length"; }
        Ok((st, ds))
    }

    #[allow(clippy::too_many_arguments)]
    fn df2_loop(&self, d: &DFlash2, mut pending: u32, hold: usize, n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                hook: Option<DraftHook>, st: &mut StreamStats, ds: &mut Df2Stats) -> Result<(), String> {
        let max_ctx = self.opt.max_ctx;
        let b = d.cfg.block;
        let ev = self.gpu.shared_event();
        let mut nsig = 0u64;
        let mut wait_v: Option<u64> = None;
        let mut last_replay: Option<CmdBuf> = None;
        let fin = |lr: &mut Option<CmdBuf>, ds: &mut Df2Stats| -> Result<(), String> {
            if let Some(cb) = lr.take() { wait(&cb)?; ds.replay_ms += gpu_ms(&cb); }
            Ok(())
        };
        let r = (|| -> Result<(), String> {
            loop {
                let p = self.pos();
                if p + 1 > max_ctx { st.stop_reason = "ctx_full"; break; }
                // ---- drafts: one drafter forward + selector (it reads only its own context rows, so it may overlap the replay)
                let left = n_new.saturating_sub(st.gen).max(1);
                let cap = (b - 1).min(max_ctx - p - 1).min(left - 1);
                let mut drafts: Vec<u32> = vec![];
                if cap > 0 {
                    if d.ctx_len() < p { ds.short_ctx += 1; }
                    let (dr, ms, sel_ms) = self.dflash2_draft(p, pending)?;
                    ds.draft_ms += ms;
                    ds.select_ms += sel_ms;
                    drafts = match hook { Some(f) => f(p, &dr), None => dr };
                    drafts.truncate(cap);
                }
                let nd = drafts.len();
                let k = nd + 1;
                let mut rows = Vec::with_capacity(k);
                rows.push(pending);
                rows.extend_from_slice(&drafts);
                self.reader.read_rows(&rows, self.stage_ptr(0))?;
                // ---- verify (waits on the GPU for the previous replay)
                let vcb = objc2::rc::autoreleasepool(|_| -> Result<CmdBuf, String> {
                    let cb = self.gpu.command_buffer();
                    if let Some(v) = wait_v { cb_wait(&cb, &ev, v); }
                    let e = Enc::new(&self.gpu, &cb, &self.buf);
                    let r = self.encode_df2_verify(&e, d, k, p);
                    e.end();
                    r.map(|_| cb)
                })?;
                self.rm_before_pass();
                commit(&vcb);
                wait(&vcb)?;
                ds.verify_ms += gpu_ms(&vcb);
                fin(&mut last_replay, ds)?;
                // ---- decide: only the target's own argmax is ever emitted
                let a: Vec<u32> = (0..k).map(|i| self.rec_token(p + i).0).collect();
                let mut j = 0;
                while j < nd && a[j] == drafts[j] && !self.is_eos(a[j]) { j += 1; }
                ds.passes += 1;
                ds.drafted += nd;
                if j < nd { ds.rejects += 1; }
                let mut why: Option<&'static str> = None;
                let mut used = 0;
                for &t in &a[..=j] {
                    used += 1;
                    st.gen += 1;
                    ds.tokens += 1;
                    st.last = Some(t);
                    if !on_tok(t) { why = Some("cancelled"); break; }
                    if self.is_eos(t) || stop.contains(&t) { why = Some("eos"); break; }
                    if st.gen >= n_new { why = Some("length"); break; }
                }
                let je = used - 1;
                // the drafts that became output: a stop token or a cancel can end the run inside the accepted ones
                ds.accept_hist[je] += 1;
                ds.accepted += je;
                let gaps: Vec<f32> = (0..=je).map(|i| self.rec_gap(p + i)).collect();
                self.rm_after_pass(&gaps);
                self.ctrl().pos = (p + je + 1) as u32;
                self.arena.slice_mut::<f32>(self.sl().rec)[4 * (p + je + 1)..4 * (p + k)].fill(0.0);
                self.set_pos(p + je + 1);
                // ---- replay of the fed rows (HOLD pass); the next verify waits for its signal on the GPU
                wait_v = None;
                if k > 1 {
                    let cb = self.enc_cb(&|e| self.encode_replay_from(e, hold, je + 1))?;
                    nsig += 1;
                    cb_signal(&cb, &ev, nsig);
                    commit(&cb);
                    wait_v = Some(nsig);
                    last_replay = Some(cb);
                }
                if why.is_none() && self.pos() + 1 > max_ctx { why = Some("ctx_full"); }
                if let Some(w) = why { st.stop_reason = w; break; }
                pending = a[je];
            }
            Ok(())
        })();
        let f = fin(&mut last_replay, ds);
        r.and(f)
    }
}
