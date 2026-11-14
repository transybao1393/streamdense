//! End-to-end self-test of the dense-engine on SYNTHETIC tiny models (no real model is loaded): random MLX
//! 4-bit weights in a `MemSource`, both families:
//! - "qwen3": QK-norm, no bias, tied embeddings, BF16 scales / norms, GQA 4 q heads / 2 kv heads (q dim 512 != hidden);
//! - "qwen2": q/k/v bias, no QK-norm, tied embeddings, F16 scales / biases / norms (converted to bf16 at load, as the
//!   Qwen2.5 MLX conversion stores them), 2 q heads / 1 kv head.
//!
//! Checks:
//! 1. logits after a 300-token prefill (two 256-key attention chunks) and after 3 decode steps vs a CPU f32
//!    reference of the same model (bf16 rounding at the same points as the kernels; summation orders differ):
//!    max |diff| <= 1.5% of max |ref| + 0.01, and the GPU argmax is a CPU near-max (within that tolerance). The check
//!    is shown to discriminate: the CPU model WITHOUT the family feature (q/k/v bias, QK-norm) misses the tolerance.
//!    Run for every attention impl with engine-pass prefill: auto (flash prefill passes of up to 128 rows + the
//!    MLX-mirror sdpa_vec for 1-row passes), splitk, flash, sdpa; and with the default GEMM prefill (auto, splitk).
//! 2. K-invariance: prefill in passes of 8 (twin + MMA matmuls, attn128_partial_k8 / k5 ...) == the same tokens fed
//!    one per pass, logits bit-identical; same for sdpa (8 / 1 / mixed) and flash (128 / 8 / mixed 1..8 rows).
//! 3. sdpa long-row fallback: rows with N > sdpa128_vec_max_n run split-K (prefill across N = 8192).
//! 4. auto engine == flash body + sdpa last token, bit for bit (its prefill split).
//! 5. The local tokenizer.json files (read-only, no weights) round-trip text through encode / decode.
//!
//! 6. Other dense families (`family_specs`: head dim 64 GQA / MHA, 96 / 160 GQA, 256 MQA; Llama q/k/v/o and MLP biases, untied head,
//!    llama3 RoPE; Granite multipliers; Gemma GeGLU, (1 + w) norms, embedding scale; Gemma 2 four norms, attention /
//!    final soft caps, alternating sliding window; Gemma 3 QK-norm, local RoPE theta, linear-scaled global RoPE), every
//!    field set on `DenseConfig` as a family reader would: the same CPU-reference check as 1 for every attention impl
//!    / prefill mode, and per feature an ablated reference (that one feature off) that must miss the tolerance;
//!    K-invariance and pipelined == sync for the new kernel variants; weight layout (every tensor used once).
//! 7. Sliding-window decode past the window: decode == CPU reference == one-shot prefill, window exactly W.
//! 8. Unquantized (F16 / BF16) tied embedding: bf16 lookup + dense lm_head vs the CPU reference.
//!
//! Run: cargo test --release --test engine_synth -- --nocapture --test-threads=1   (inside engine/dense-engine)
use dense_engine::config::{Activation, DenseConfig, Family, LayerKind, RopeScaling};
use dense_engine::engine::PrefillMode;
use dense_engine::model::AttnImpl;
use dense_engine::weights::MemSource;
use dense_engine::{Engine, EngineOptions, Tokenizer};
use streamdense_core::DType;

// ------------------------------------------------------------------------------------------- bf16 / f16 / rng
fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
fn to_bf(f: f32) -> u16 {
    let mut u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 { return ((u >> 16) | 0x0040) as u16; }
    u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (u >> 16) as u16
}
fn rbf(f: f32) -> f32 { bf(to_bf(f)) }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unif(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    fn sym(&mut self) -> f32 { 2.0 * self.unif() - 1.0 }
}

// ------------------------------------------------------------------------------------------- synthetic model
#[derive(Clone)]
struct Spec { qwen3: bool, hidden: usize, inter: usize, layers: usize, heads: usize, kv: usize, vocab: usize, f16: bool, x: Ext }

/// Model math beyond the Qwen pair. `family: None` = the config straight from `DenseConfig::from_json` (qwen2 / qwen3
/// as the spec says); `Some` = fields set on the parsed config the way a family config reader would. The CPU
/// reference reads the same struct, so an ablated copy (one feature off) gives the "feature missing" reference.
#[derive(Clone, Debug)]
struct Ext {
    family: Option<Family>,
    hd: usize,
    qk_norm: bool,
    qkv_bias: bool,
    o_bias: bool,
    mlp_bias: bool,
    gelu: bool,
    plus1: bool,
    four: bool,
    embed_scale: f64,
    resid: f64,
    logit_div: f64,
    scale: Option<f64>,
    attn_cap: Option<f64>,
    final_cap: Option<f64>,
    window: Option<usize>,
    /// per layer: sliding (empty = all full attention)
    sliding: Vec<bool>,
    theta: f64,
    local_theta: Option<f64>,
    rope: RopeScaling,
    untied: bool,
}

impl Ext {
    fn qwen(qwen3: bool) -> Ext {
        Ext { family: None, hd: 128, qk_norm: qwen3, qkv_bias: !qwen3, o_bias: false, mlp_bias: false, gelu: false,
              plus1: false, four: false, embed_scale: 1.0, resid: 1.0, logit_div: 1.0, scale: None, attn_cap: None,
              final_cap: None, window: None, sliding: vec![], theta: 1e6, local_theta: None, rope: RopeScaling::None,
              untied: false }
    }
    fn fam(f: Family, hd: usize) -> Ext { Ext { family: Some(f), hd, qk_norm: false, qkv_bias: false, ..Ext::qwen(false) } }
    fn is_sliding(&self, l: usize) -> bool { self.sliding.get(l).copied().unwrap_or(false) }
    /// One copy per feature that is on, with that feature off (name, ext).
    fn ablations(&self) -> Vec<(&'static str, Ext)> {
        let mut v: Vec<(&'static str, Ext)> = vec![];
        let mut add = |on: bool, name: &'static str, f: &dyn Fn(&mut Ext)| {
            if on { let mut e = self.clone(); f(&mut e); v.push((name, e)); }
        };
        add(self.qk_norm, "qk-norm", &|e| e.qk_norm = false);
        add(self.qkv_bias, "q/k/v bias", &|e| e.qkv_bias = false);
        add(self.o_bias, "o_proj bias", &|e| e.o_bias = false);
        add(self.mlp_bias, "mlp bias", &|e| e.mlp_bias = false);
        add(self.gelu, "GeGLU (as SiLU)", &|e| e.gelu = false);
        add(self.plus1, "(1 + w) norms", &|e| e.plus1 = false);
        add(self.four, "four norms", &|e| e.four = false);
        add(self.embed_scale != 1.0, "embed scale", &|e| e.embed_scale = 1.0);
        add(self.resid != 1.0, "residual multiplier", &|e| e.resid = 1.0);
        add(self.logit_div != 1.0, "logits scaling", &|e| e.logit_div = 1.0);
        add(self.scale.is_some_and(|s| s as f32 != (self.hd as f32).powf(-0.5)), "attention scale", &|e| e.scale = None);
        add(self.attn_cap.is_some(), "attn soft cap", &|e| e.attn_cap = None);
        add(self.final_cap.is_some(), "final soft cap", &|e| e.final_cap = None);
        add(self.sliding.iter().any(|s| *s), "sliding window", &|e| e.sliding.clear());
        add(self.local_theta.is_some(), "local rope theta", &|e| e.local_theta = None);
        add(self.rope != RopeScaling::None, "rope scaling", &|e| e.rope = RopeScaling::None);
        v
    }
}

/// One dequantized matrix (row-major [rows][cols], f32) as the CPU reference sees it.
struct Mat { rows: usize, cols: usize, w: Vec<f32> }

/// Norm weights are RAW (as stored in the file); the reference applies (1 + w) itself when `plus1`. With `four`,
/// post_norm holds pre_feedforward_layernorm, post_attn / post_ffw the post_attention / post_feedforward ones.
struct RefLayer { in_norm: Vec<f32>, post_norm: Vec<f32>, qkv: Mat, bias: Option<Vec<f32>>, q_norm: Option<Vec<f32>>,
                  k_norm: Option<Vec<f32>>, o: Mat, gate: Mat, up: Mat, down: Mat, post_attn: Option<Vec<f32>>,
                  post_ffw: Option<Vec<f32>>, o_bias: Option<Vec<f32>>, gate_bias: Option<Vec<f32>>,
                  up_bias: Option<Vec<f32>>, down_bias: Option<Vec<f32>> }
struct RefModel { embed: Mat, norm: Vec<f32>, layers: Vec<RefLayer>, lm_head: Option<Mat> }

/// bf16-exact float (also exact in f16: |x| kept >= 1e-3).
fn val(r: &mut Rng, center: f32, spread: f32) -> f32 {
    let mut x = center + spread * r.sym();
    if x.abs() < 1e-3 { x = if x < 0.0 { -1e-3 } else { 1e-3 }; }
    rbf(x)
}
fn store(src: &mut MemSource, name: &str, shape: &[usize], v: &[f32], f16: bool) {
    let bytes: Vec<u8> = if f16 {
        v.iter().flat_map(|x| streamdense_core::f16::f32_to_f16(*x).to_le_bytes()).collect()
    } else {
        v.iter().flat_map(|x| to_bf(*x).to_le_bytes()).collect()
    };
    src.insert(name, if f16 { DType::F16 } else { DType::BF16 }, shape, bytes);
}
/// Random 4-bit linear `prefix` [rows][cols] (MLX layout) -> MemSource, returns the dequantized matrix.
fn q4(src: &mut MemSource, r: &mut Rng, prefix: &str, rows: usize, cols: usize, f16: bool) -> Mat {
    let g = cols / 64;
    let mut words = vec![0u32; rows * cols / 8];
    let (mut s, mut b) = (vec![0f32; rows * g], vec![0f32; rows * g]);
    let mut w = vec![0f32; rows * cols];
    let sc0 = 0.9 / (cols as f32).sqrt() / 4.3; // keeps the output std ~ the input std
    for i in 0..rows {
        for gi in 0..g {
            s[i * g + gi] = val(r, sc0, sc0 * 0.3);
            b[i * g + gi] = val(r, -7.5 * s[i * g + gi], sc0);
        }
        for c in 0..cols {
            let q = (r.next() % 16) as u32;
            words[(i * cols + c) / 8] |= q << (4 * (c % 8));
            let gi = c / 64;
            w[i * cols + c] = (q as f32).mul_add(s[i * g + gi], b[i * g + gi]); // fma, as the kernels
        }
    }
    src.insert(&format!("{}.weight", prefix), DType::U32, &[rows, cols / 8], words.iter().flat_map(|x| x.to_le_bytes()).collect());
    store(src, &format!("{}.scales", prefix), &[rows, g], &s, f16);
    store(src, &format!("{}.biases", prefix), &[rows, g], &b, f16);
    Mat { rows, cols, w }
}
fn vecw(src: &mut MemSource, r: &mut Rng, name: &str, n: usize, center: f32, spread: f32, f16: bool) -> Vec<f32> {
    let v: Vec<f32> = (0..n).map(|_| val(r, center, spread)).collect();
    store(src, name, &[n], &v, f16);
    v
}
fn cat(m: &[&Mat]) -> Mat {
    Mat { rows: m.iter().map(|x| x.rows).sum(), cols: m[0].cols, w: m.iter().flat_map(|x| x.w.iter().copied()).collect() }
}

fn build(sp: &Spec, seed: u64) -> (DenseConfig, MemSource, RefModel) {
    let mut r = Rng(seed);
    let mut src = MemSource::default();
    let x = &sp.x;
    let (h, hd) = (sp.hidden, x.hd);
    let (qd, kvd) = (sp.heads * hd, sp.kv * hd);
    let f16 = sp.f16;
    // Gemma-style raw norm weights sit around 0 (used as 1 + w)
    let nc = if x.plus1 { 0.0 } else { 1.0 };
    let embed = q4(&mut src, &mut r, "model.embed_tokens", sp.vocab, h, f16);
    let norm = vecw(&mut src, &mut r, "model.norm.weight", h, nc, 0.2, f16);
    let mut layers = vec![];
    for l in 0..sp.layers {
        let p = format!("model.layers.{}", l);
        let a = format!("{}.self_attn", p);
        let in_norm = vecw(&mut src, &mut r, &format!("{}.input_layernorm.weight", p), h, nc, 0.2, f16);
        let (post_norm, post_attn, post_ffw) = if x.four {
            let pa = vecw(&mut src, &mut r, &format!("{}.post_attention_layernorm.weight", p), h, nc, 0.2, f16);
            let pre = vecw(&mut src, &mut r, &format!("{}.pre_feedforward_layernorm.weight", p), h, nc, 0.2, f16);
            let pf = vecw(&mut src, &mut r, &format!("{}.post_feedforward_layernorm.weight", p), h, nc, 0.2, f16);
            (pre, Some(pa), Some(pf))
        } else {
            (vecw(&mut src, &mut r, &format!("{}.post_attention_layernorm.weight", p), h, nc, 0.2, f16), None, None)
        };
        let q = q4(&mut src, &mut r, &format!("{}.q_proj", a), qd, h, f16);
        let k = q4(&mut src, &mut r, &format!("{}.k_proj", a), kvd, h, f16);
        let v = q4(&mut src, &mut r, &format!("{}.v_proj", a), kvd, h, f16);
        let bias = if x.qkv_bias {
            let mut b = vecw(&mut src, &mut r, &format!("{}.q_proj.bias", a), qd, 0.0, 0.5, f16);
            b.extend(vecw(&mut src, &mut r, &format!("{}.k_proj.bias", a), kvd, 0.0, 0.5, f16));
            b.extend(vecw(&mut src, &mut r, &format!("{}.v_proj.bias", a), kvd, 0.0, 0.5, f16));
            Some(b)
        } else { None };
        let (q_norm, k_norm) = if x.qk_norm {
            (Some(vecw(&mut src, &mut r, &format!("{}.q_norm.weight", a), hd, nc, 0.2, f16)),
             Some(vecw(&mut src, &mut r, &format!("{}.k_norm.weight", a), hd, nc, 0.2, f16)))
        } else { (None, None) };
        let o = q4(&mut src, &mut r, &format!("{}.o_proj", a), h, qd, f16);
        let o_bias = if x.o_bias { Some(vecw(&mut src, &mut r, &format!("{}.o_proj.bias", a), h, 0.0, 0.5, f16)) } else { None };
        let gate = q4(&mut src, &mut r, &format!("{}.mlp.gate_proj", p), sp.inter, h, f16);
        let up = q4(&mut src, &mut r, &format!("{}.mlp.up_proj", p), sp.inter, h, f16);
        let down = q4(&mut src, &mut r, &format!("{}.mlp.down_proj", p), h, sp.inter, f16);
        let (gate_bias, up_bias, down_bias) = if x.mlp_bias {
            (Some(vecw(&mut src, &mut r, &format!("{}.mlp.gate_proj.bias", p), sp.inter, 0.0, 0.5, f16)),
             Some(vecw(&mut src, &mut r, &format!("{}.mlp.up_proj.bias", p), sp.inter, 0.0, 0.5, f16)),
             Some(vecw(&mut src, &mut r, &format!("{}.mlp.down_proj.bias", p), h, 0.0, 0.5, f16)))
        } else { (None, None, None) };
        layers.push(RefLayer { in_norm, post_norm, qkv: cat(&[&q, &k, &v]), bias, q_norm, k_norm, o, gate, up, down,
                               post_attn, post_ffw, o_bias, gate_bias, up_bias, down_bias });
    }
    let lm_head = if x.untied { Some(q4(&mut src, &mut r, "lm_head", sp.vocab, h, f16)) } else { None };
    let mut cfg = serde_json::json!({
        "model_type": if sp.qwen3 { "qwen3" } else { "qwen2" }, "hidden_size": h, "intermediate_size": sp.inter,
        "num_hidden_layers": sp.layers, "num_attention_heads": sp.heads, "num_key_value_heads": sp.kv,
        "vocab_size": sp.vocab, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
        "hidden_act": "silu", "eos_token_id": 1, "max_position_embeddings": 4096,
        "quantization": {"group_size": 64, "bits": 4},
    });
    if sp.qwen3 { cfg["head_dim"] = 128.into(); cfg["attention_bias"] = false.into(); }
    let mut cfg = DenseConfig::from_json(&cfg, None).unwrap();
    if let Some(f) = x.family {
        // what a family config reader does: set the fields of its family on a DenseConfig
        cfg.family = f;
        cfg.head_dim = hd;
        cfg.rope_dims = hd;
        cfg.qk_norm = x.qk_norm;
        cfg.attention_bias = x.qkv_bias;
        cfg.o_bias = x.o_bias;
        cfg.mlp_bias = x.mlp_bias;
        cfg.activation = if x.gelu { Activation::GeluTanh } else { Activation::Silu };
        cfg.hidden_act = if x.gelu { "gelu_pytorch_tanh".into() } else { "silu".into() };
        cfg.norm_plus_one = x.plus1;
        cfg.four_norms = x.four;
        cfg.embed_scale = x.embed_scale;
        cfg.residual_multiplier = x.resid;
        cfg.logits_scaling = x.logit_div;
        cfg.attn_scale = x.scale;
        cfg.attn_softcap = x.attn_cap;
        cfg.final_softcap = x.final_cap;
        cfg.sliding_window = x.window;
        cfg.kinds = (0..sp.layers).map(|l| if x.is_sliding(l) { LayerKind::Sliding } else { LayerKind::Full }).collect();
        cfg.rope_theta = x.theta;
        cfg.rope_local_theta = x.local_theta;
        cfg.rope_scale = x.rope;
        cfg.tie_word_embeddings = !x.untied;
    }
    (cfg, src, RefModel { embed, norm, layers, lm_head })
}

// ------------------------------------------------------------------------------------------- CPU reference
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let r = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| rbf(w * rbf(v * r))).collect()
}
fn matvec(m: &Mat, x: &[f32]) -> Vec<f32> {
    (0..m.rows).map(|i| m.w[i * m.cols..(i + 1) * m.cols].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}
fn silu_mul(g: f32, u: f32) -> f32 {
    let (g, u) = (rbf(g), rbf(u));
    let e = rbf(g.abs().exp());
    let y = rbf(1.0 / rbf(1.0 + e));
    let s = if g < 0.0 { y } else { rbf(1.0 - y) };
    rbf(rbf(g * s) * u)
}
/// tanh-GELU (HF gelu_pytorch_tanh) in f32, rounded once to bf16, times up.
fn gelu_mul(g: f32, u: f32) -> f32 {
    let (g, u) = (rbf(g), rbf(u));
    let a = rbf(0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()));
    rbf(a * u)
}
/// HF `_compute_llama3_parameters` / `_compute_linear_scaling_rope_parameters`, written on inverse frequencies (the
/// engine computes mlx_lm's period form; same math).
fn ref_invf(n: usize, theta: f64, rope: RopeScaling) -> Vec<f32> {
    // plain: the GPU kernel's f32 formula exp2(-(i / n) * log2(theta))
    let plain: Vec<f32> = (0..n).map(|i| (-(i as f32 / n as f32) * (theta as f32).log2()).exp2()).collect();
    match rope {
        RopeScaling::None => plain,
        RopeScaling::Linear { factor } => (0..n).map(|i| (theta.powf(-(i as f64) / n as f64) / factor) as f32).collect(),
        RopeScaling::Llama3 { factor, low_freq_factor: lo, high_freq_factor: hi, original_max_pos } => (0..n).map(|i| {
            let inv = theta.powf(-(i as f64) / n as f64);
            let old = original_max_pos as f64;
            let wl = 2.0 * std::f64::consts::PI / inv;
            let llama = if wl > old / lo { inv / factor } else { inv };
            let smooth = (old / wl - lo) / (hi - lo);
            let medium = wl >= old / hi && wl <= old / lo; // HF: not (wl < high_wl) and not (wl > low_wl)
            (if medium { (1.0 - smooth) * llama / factor + smooth * llama } else { llama }) as f32
        }).collect(),
    }
}

struct CpuRef<'a> { m: &'a RefModel, sp: &'a Spec, x: Ext, kc: Vec<Vec<Vec<f32>>>, vc: Vec<Vec<Vec<f32>>>, invf: Vec<f32>,
                    invf_local: Vec<f32> }

impl<'a> CpuRef<'a> {
    fn new(m: &'a RefModel, sp: &'a Spec) -> CpuRef<'a> { CpuRef::with(m, sp, sp.x.clone()) }
    /// Reference with the math of `x` (an ablation of sp.x) over the same weights.
    fn with(m: &'a RefModel, sp: &'a Spec, x: Ext) -> CpuRef<'a> {
        let n = x.hd / 2;
        let invf = ref_invf(n, x.theta, x.rope);
        let invf_local = match x.local_theta { Some(t) => ref_invf(n, t, RopeScaling::None), None => invf.clone() };
        CpuRef { m, sp, x, kc: vec![vec![]; sp.layers], vc: vec![vec![]; sp.layers], invf, invf_local }
    }
    fn rope(&self, x: &[f32], pos: usize, invf: &[f32]) -> Vec<f32> {
        let half = self.x.hd / 2;
        let mut o = vec![0f32; self.x.hd];
        for i in 0..half {
            let th = pos as f32 * invf[i];
            let (s, c) = th.sin_cos();
            o[i] = rbf(x[i] * c - x[i + half] * s);
            o[i + half] = rbf(x[i] * s + x[i + half] * c);
        }
        o
    }
    /// Norm weight as the math sees it.
    fn nw(&self, w: &[f32]) -> Vec<f32> { if self.x.plus1 { w.iter().map(|v| rbf(1.0 + v)).collect() } else { w.to_vec() } }
    /// h = h + bf(post(bf(r) + bias) * m)
    fn residual(&self, h: &mut [f32], r: Vec<f32>, bias: &Option<Vec<f32>>, post: &Option<Vec<f32>>, use_bias: bool) {
        let mut r: Vec<f32> = r.into_iter().map(rbf).collect();
        if let (Some(b), true) = (bias, use_bias) { for (v, b) in r.iter_mut().zip(b) { *v = rbf(*v + b); } }
        if let (Some(w), true) = (post, self.x.four) { r = rmsnorm(&r, &self.nw(w), 1e-6); }
        let m = rbf(self.x.resid as f32);
        for (a, b) in h.iter_mut().zip(r) { *a = rbf(*a + rbf(b * m)); }
    }
    /// Feed one token; returns logits.
    fn step(&mut self, tok: u32) -> Vec<f32> {
        let (sp, hd) = (self.sp, self.x.hd);
        let (h, nh, nkv) = (sp.hidden, sp.heads, sp.kv);
        let pos = self.kc[0].len();
        let e = &self.m.embed;
        let mut x: Vec<f32> = e.w[tok as usize * h..(tok as usize + 1) * h].iter().map(|v| rbf(*v)).collect();
        if self.x.embed_scale != 1.0 {
            let s = rbf(self.x.embed_scale as f32);
            for v in x.iter_mut() { *v = rbf(*v * s); }
        }
        let scale = self.x.scale.map(|s| s as f32).unwrap_or((hd as f32).powf(-0.5));
        for (l, lw) in self.m.layers.iter().enumerate() {
            let sliding = self.x.is_sliding(l);
            let invf = if sliding { self.invf_local.clone() } else { self.invf.clone() };
            let xn = rmsnorm(&x, &self.nw(&lw.in_norm), 1e-6);
            let mut p: Vec<f32> = matvec(&lw.qkv, &xn).into_iter().map(rbf).collect();
            if let (Some(b), true) = (&lw.bias, self.x.qkv_bias) { for (v, b) in p.iter_mut().zip(b) { *v = rbf(*v + b); } }
            let head = |i: usize| p[i * hd..(i + 1) * hd].to_vec();
            let qk = |v: Vec<f32>, w: &Option<Vec<f32>>| match (w, self.x.qk_norm) { (Some(w), true) => rmsnorm(&v, &self.nw(w), 1e-6), _ => v };
            let qs: Vec<Vec<f32>> = (0..nh).map(|i| self.rope(&qk(head(i), &lw.q_norm), pos, &invf)).collect();
            let mut ks = vec![];
            let mut vs = vec![];
            for j in 0..nkv {
                ks.extend(self.rope(&qk(head(nh + j), &lw.k_norm), pos, &invf));
                vs.extend(head(nh + nkv + j));
            }
            self.kc[l].push(ks);
            self.vc[l].push(vs);
            let n = pos + 1;
            // window: key t visible iff pos - t < W
            let lo = match (sliding, self.x.window) { (true, Some(w)) => n.saturating_sub(w), _ => 0 };
            let mut ao = vec![0f32; nh * hd];
            for i in 0..nh {
                let j = i / (nh / nkv);
                let sc: Vec<f32> = (lo..n).map(|t| {
                    let k = &self.kc[l][t][j * hd..(j + 1) * hd];
                    let s = qs[i].iter().zip(k).map(|(a, b)| scale * a * b).sum::<f32>();
                    match self.x.attn_cap { Some(c) => c as f32 * (s / c as f32).tanh(), None => s }
                }).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let pe: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                let lsum: f32 = pe.iter().sum();
                for d in 0..hd {
                    let o: f32 = (lo..n).map(|t| pe[t - lo] * self.vc[l][t][j * hd + d]).sum();
                    ao[i * hd + d] = rbf(o / lsum);
                }
            }
            self.residual(&mut x, matvec(&lw.o, &ao), &lw.o_bias, &lw.post_attn, self.x.o_bias);
            let xn = rmsnorm(&x, &self.nw(&lw.post_norm), 1e-6);
            let mut g: Vec<f32> = matvec(&lw.gate, &xn).into_iter().map(rbf).collect();
            let mut u: Vec<f32> = matvec(&lw.up, &xn).into_iter().map(rbf).collect();
            if self.x.mlp_bias {
                for (v, b) in g.iter_mut().zip(lw.gate_bias.as_ref().unwrap()) { *v = rbf(*v + b); }
                for (v, b) in u.iter_mut().zip(lw.up_bias.as_ref().unwrap()) { *v = rbf(*v + b); }
            }
            let f: Vec<f32> = g.iter().zip(&u).map(|(g, u)| if self.x.gelu { gelu_mul(*g, *u) } else { silu_mul(*g, *u) }).collect();
            self.residual(&mut x, matvec(&lw.down, &f), &lw.down_bias, &lw.post_ffw, self.x.mlp_bias);
        }
        let hf = rmsnorm(&x, &self.nw(&self.m.norm), 1e-6);
        let mut lg = matvec(self.m.lm_head.as_ref().unwrap_or(&self.m.embed), &hf);
        if self.x.logit_div != 1.0 { for v in lg.iter_mut() { *v /= self.x.logit_div as f32; } }
        if let Some(c) = self.x.final_cap { for v in lg.iter_mut() { *v = c as f32 * (*v / c as f32).tanh(); } }
        lg
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..v.len() { if v[i] > v[b] { b = i; } }
    b
}

/// GPU logits vs CPU: tolerance and argmax agreement (up to near-ties).
fn tol_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32, f32) {
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs()));
    let maxd = gpu.iter().zip(cpu).fold(0f32, |a, (g, c)| a.max((g - c).abs()));
    (scale, maxd, 0.015 * scale + 0.01)
}
fn compare(what: &str, gpu: &[f32], cpu: &[f32], gpu_tok: u32) {
    let (scale, maxd, tol) = tol_diff(gpu, cpu);
    let ct = argmax(cpu);
    println!("  {:<40} max|ref| {:7.3}  max|diff| {:.4} (tol {:.4})  argmax gpu {} cpu {}", what, scale, maxd, tol, gpu_tok, ct);
    assert!(maxd <= tol, "{}: max |diff| {} > {}", what, maxd, tol);
    // argmax_b compares the bf16-rounded logits (MLX logits are bf16), lowest index on ties
    let gb: Vec<f32> = gpu.iter().map(|v| rbf(*v)).collect();
    assert_eq!(argmax(&gb) as u32, gpu_tok, "{}: argmax_b != host argmax of the bf16-rounded logits", what);
    assert!(cpu[gpu_tok as usize] >= cpu[ct] - tol, "{}: gpu token {} is not a cpu near-max", what, gpu_tok);
}

fn specs() -> Vec<(&'static str, Spec)> {
    vec![
        ("qwen3 (qk-norm, bf16)", Spec { qwen3: true, hidden: 256, inter: 512, layers: 2, heads: 4, kv: 2, vocab: 384, f16: false, x: Ext::qwen(true) }),
        ("qwen2 (bias, f16)", Spec { qwen3: false, hidden: 256, inter: 640, layers: 2, heads: 2, kv: 1, vocab: 320, f16: true, x: Ext::qwen(false) }),
    ]
}

/// (attention impl, prefill mode): every impl with engine passes, and the default GEMM prefill with auto / splitk.
const IMPLS: [(AttnImpl, PrefillMode); 6] = [
    (AttnImpl::Auto, PrefillMode::Passes), (AttnImpl::SplitK, PrefillMode::Passes), (AttnImpl::Flash, PrefillMode::Passes),
    (AttnImpl::Sdpa, PrefillMode::Passes), (AttnImpl::Auto, PrefillMode::Gemm), (AttnImpl::SplitK, PrefillMode::Gemm),
];

#[test]
fn engine_vs_cpu_reference() {
    for (name, sp) in specs() {
        let (cfg, src, m) = build(&sp, 0x9e3779b97f4a7c15);
        let mut r = Rng(42);
        let prompt: Vec<u32> = (0..300).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        let mut cpu = CpuRef::new(&m, &sp);
        let mut cl = vec![];
        for &t in &prompt { cl = cpu.step(t); }
        // ablation: the reference without this family's feature must NOT pass
        let mut m2 = build(&sp, 0x9e3779b97f4a7c15).2;
        for l in &mut m2.layers { l.bias = None; l.q_norm = None; l.k_norm = None; }
        let mut cpu2 = CpuRef::new(&m2, &sp);
        let mut cl2 = vec![];
        for &t in &prompt { cl2 = cpu2.step(t); }
        // decode continuation: the CPU's own greedy choices (same sequence for every impl)
        let mut cont = vec![];
        let mut cls = vec![];
        let mut c = cl.clone();
        for _ in 0..3 {
            let t = argmax(&c) as u32;
            c = cpu.step(t);
            cont.push(t);
            cls.push(c.clone());
        }
        for (imp, pm) in IMPLS {
            let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 320, attn: imp, prefill: pm, ..Default::default() }).unwrap();
            println!("{} [{} {}]: load {:.2} s (shader compile + weights), arena {} KB", name, imp.name(), pm.name(),
                     eng.load_s, eng.arena_bytes / 1024);
            let tag = format!("{} {} {}", name, imp.name(), pm.name());
            let tok = eng.prefill(&prompt).unwrap();
            compare(&format!("{} prefill 300", tag), &eng.logits(), &cl, tok);
            let (_, d2, t2) = tol_diff(&eng.logits(), &cl2);
            println!("  {:<40} ablated reference (no bias / QK-norm): max|diff| {:.4} (must exceed {:.4})", tag, d2, t2);
            assert!(d2 > t2, "{}: the comparison does not see the bias / QK-norm", tag);
            for (i, (&t, c)) in cont.iter().zip(&cls).enumerate() {
                let tok = eng.step(t).unwrap();
                compare(&format!("{} decode {}", tag, i + 1), &eng.logits(), c, tok);
            }
            assert_eq!(eng.pos(), 303);
        }
    }
}

/// Synthetic 2-layer models of the other families, every feature set through `DenseConfig` fields. Values are chosen
/// so each feature moves the logits well past the tolerance (soft caps below the typical score / logit size, a window
/// much shorter than the 300-token prompt, a llama3 original context of 256 so most frequencies are rescaled).
fn family_specs() -> Vec<(&'static str, Spec)> {
    let sp = |hidden, inter, heads, kv, vocab, x| Spec { qwen3: true, hidden, inter, layers: 2, heads, kv, vocab, f16: false, x };
    let mut v = vec![];
    let mut x = Ext::fam(Family::Llama, 64);
    x.qk_norm = true;
    v.push(("hd64 gqa 4/2 (qk-norm)", sp(256, 512, 4, 2, 384, x)));
    let mut x = Ext::fam(Family::Llama, 64);
    (x.qkv_bias, x.o_bias, x.untied) = (true, true, true);
    v.push(("hd64 mha 4/4 (q/k/v/o bias, untied)", sp(256, 512, 4, 4, 320, x)));
    let mut x = Ext::fam(Family::Gemma, 256);
    (x.gelu, x.plus1, x.embed_scale) = (true, true, 256f64.sqrt());
    v.push(("hd256 mqa 2/1 (gemma: geglu, 1+w, embed)", sp(256, 512, 2, 1, 384, x)));
    // Gemma 2: even layers sliding (HF is_sliding = not layer_idx % 2), query_pre_attn_scalar 100, both soft caps
    let mut x = Ext::fam(Family::Gemma2, 128);
    (x.gelu, x.plus1, x.four, x.embed_scale) = (true, true, true, 256f64.sqrt());
    (x.scale, x.attn_cap, x.final_cap) = (Some(100f64.powf(-0.5)), Some(1.5), Some(2.0));
    (x.window, x.sliding) = (Some(37), vec![true, false]);
    v.push(("gemma2-style hd128 4/2", sp(256, 512, 4, 2, 384, x)));
    // Gemma 3: qk-norm (1 + w), sliding layer with its own plain RoPE theta, full layer with linear-scaled RoPE
    let mut x = Ext::fam(Family::Gemma3, 256);
    (x.qk_norm, x.gelu, x.plus1, x.four, x.embed_scale) = (true, true, true, true, 256f64.sqrt());
    x.scale = Some(256f64.powf(-0.5));
    (x.window, x.sliding, x.local_theta, x.rope) = (Some(50), vec![true, false], Some(10000.0), RopeScaling::Linear { factor: 4.0 });
    v.push(("gemma3-style hd256 2/1", sp(256, 512, 2, 1, 384, x)));
    // Granite: attention_multiplier, embedding_multiplier, residual_multiplier, logits_scaling
    let mut x = Ext::fam(Family::Granite, 64);
    (x.scale, x.embed_scale, x.resid, x.logit_div) = (Some(0.05), 4.0, 0.5, 2.0);
    v.push(("granite hd64 4/1 (multipliers)", sp(256, 512, 4, 1, 384, x)));
    let mut x = Ext::fam(Family::Llama, 64);
    x.theta = 500000.0;
    x.rope = RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_pos: 256 };
    v.push(("llama3 rope hd64 4/2", sp(256, 512, 4, 2, 384, x)));
    let mut x = Ext::fam(Family::Llama, 128);
    (x.qkv_bias, x.o_bias, x.mlp_bias) = (true, true, true);
    v.push(("llama biases hd128 4/2 (attn + mlp)", sp(256, 640, 4, 2, 320, x)));
    // head dim 96 (Phi-3 / Phi-3.5-mini) and 160 (stablelm-2-12b): split-K runs 3 / 5 32-dim slices per row on 4 / 8
    // lanes; the 160 case also takes the window and soft-cap variants
    let mut x = Ext::fam(Family::Llama, 96);
    x.qk_norm = true;
    v.push(("hd96 gqa 4/2 (qk-norm)", sp(256, 512, 4, 2, 384, x)));
    let mut x = Ext::fam(Family::Llama, 160);
    (x.qkv_bias, x.attn_cap) = (true, Some(1.5));
    (x.window, x.sliding) = (Some(37), vec![true, false]);
    v.push(("hd160 gqa 4/2 (bias, window, attn cap)", sp(256, 512, 4, 2, 384, x)));
    v
}

/// Every family spec vs its CPU reference: logits after a 300-token prefill and 3 decode steps, every attention impl
/// and prefill mode (as `engine_vs_cpu_reference`). Each feature is shown to matter: the reference with that one
/// feature off misses the tolerance.
#[test]
fn families_vs_cpu_reference() {
    for (name, sp) in family_specs() {
        let (cfg, src, m) = build(&sp, 0x51ed_270b_9e37_79b9);
        cfg.validate().unwrap_or_else(|e| panic!("{}: {}", name, e));
        let mut r = Rng(43);
        let prompt: Vec<u32> = (0..300).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        let mut cpu = CpuRef::new(&m, &sp);
        let mut cl = vec![];
        for &t in &prompt { cl = cpu.step(t); }
        let ablated: Vec<(&str, Vec<f32>)> = sp.x.ablations().into_iter().map(|(an, ax)| {
            let mut c = CpuRef::with(&m, &sp, ax);
            let mut l = vec![];
            for &t in &prompt { l = c.step(t); }
            (an, l)
        }).collect();
        let mut cont = vec![];
        let mut cls = vec![];
        let mut c = cl.clone();
        for _ in 0..3 {
            let t = argmax(&c) as u32;
            c = cpu.step(t);
            cont.push(t);
            cls.push(c.clone());
        }
        for (imp, pm) in IMPLS {
            let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 320, attn: imp, prefill: pm, ..Default::default() }).unwrap();
            let tag = format!("{} {} {}", name, imp.name(), pm.name());
            let tok = eng.prefill(&prompt).unwrap();
            compare(&format!("{} prefill 300", tag), &eng.logits(), &cl, tok);
            for (an, al) in &ablated {
                let (_, d2, t2) = tol_diff(&eng.logits(), al);
                if imp == AttnImpl::Auto && pm == PrefillMode::Passes {
                    println!("  {:<40} without {:<20} max|diff| {:.4} (must exceed {:.4})", tag, an, d2, t2);
                }
                assert!(d2 > t2, "{}: the comparison does not see {}", tag, an);
            }
            for (i, (&t, c)) in cont.iter().zip(&cls).enumerate() {
                let tok = eng.step(t).unwrap();
                compare(&format!("{} decode {}", tag, i + 1), &eng.logits(), c, tok);
            }
            assert_eq!(eng.pos(), 303);
        }
    }
}

/// Weight layout of every family spec (names, dtypes, shapes only; nothing loaded): every tensor in the source is used
/// exactly once (four norms, o / MLP biases, QK-norm, untied lm_head), and a missing optional tensor is named.
#[test]
fn families_weight_layout() {
    for (name, sp) in family_specs() {
        let (cfg, src, _) = build(&sp, 5);
        let (w, jobs) = dense_engine::weights::layout(&cfg, &src).unwrap();
        assert_eq!(jobs.len(), src.map.len(), "{}: tensors used != tensors present", name);
        let l0 = &w.layers[0];
        assert_eq!((l0.post_attn_norm.is_some(), l0.post_ffw_norm.is_some()), (sp.x.four, sp.x.four), "{}", name);
        assert_eq!((l0.o_bias.is_some(), l0.down_bias.is_some(), w.tied), (sp.x.o_bias, sp.x.mlp_bias, !sp.x.untied), "{}", name);
        for missing in ["layers.1.pre_feedforward_layernorm", "layers.1.post_feedforward_layernorm", "layers.0.self_attn.o_proj.bias",
                        "layers.1.mlp.down_proj.bias", "layers.0.self_attn.k_norm", "lm_head.scales"] {
            if !src.map.keys().any(|k| k.contains(missing)) { continue; }
            let mut s2 = MemSource::default();
            for (k, v) in &src.map { if !k.contains(missing) { s2.map.insert(k.clone(), v.clone()); } }
            let e = dense_engine::weights::layout(&cfg, &s2).err().unwrap();
            assert!(e.contains(missing), "{}: {}", name, e);
        }
        println!("  {:<44} {} tensors, all used; region {} KB", name, jobs.len(), w.bytes / 1024);
    }
}

/// The new kernel variants keep the engine's invariants: head dim 64 / 96 / 128 (window, soft caps) / 160 (window,
/// soft cap) / 256 (window, two RoPE tables). Prefill in passes of 8 == one token per pass == mixed sizes, bit for bit (split-K, sdpa, flash); and
/// pipelined decode == sync decode, bit for bit.
#[test]
fn families_k_invariant_and_pipelined() {
    use dense_engine::engine::DecodeMode;
    let mixed: Vec<usize> = (1..=8).collect();
    for (name, sp) in family_specs().into_iter().filter(|(n, _)| n.starts_with("hd64 gqa") || n.starts_with("gemma")
                                                                     || n.starts_with("hd96 gqa") || n.starts_with("hd160 gqa")) {
        let (cfg, src, _) = build(&sp, 0x77);
        let mut r = Rng(6);
        let prompt: Vec<u32> = (0..270).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        let cases: [(AttnImpl, Vec<Vec<usize>>); 3] = [
            (AttnImpl::SplitK, vec![vec![8], vec![1], mixed.clone()]),
            (AttnImpl::Sdpa, vec![vec![8], vec![1], mixed.clone()]),
            (AttnImpl::Flash, vec![vec![128], vec![8], vec![1], mixed.clone(), vec![100, 37, 128]]),
        ];
        for (imp, pats) in cases {
            let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 512, attn: imp, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
            let (a, la) = feed(&mut eng, &prompt, &pats[0]);
            for p in &pats[1..] {
                let (b, lb) = feed(&mut eng, &prompt, p);
                assert!(same(&la, &lb) && a == b, "{} {}: passes {:?} logits != passes {:?}", name, imp.name(), pats[0], p);
            }
            println!("  {:<28} {:<6} next token {} identical over pass patterns {:?}", name, imp.name(), a, pats);
        }
        for (imp, pm) in [(AttnImpl::Auto, PrefillMode::Gemm), (AttnImpl::SplitK, PrefillMode::Passes)] {
            let opts = |decode| EngineOptions { max_ctx: 320, attn: imp, prefill: pm, decode };
            let mut sy = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Sync)).unwrap();
            let mut pi = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Pipelined)).unwrap();
            let a = sy.generate(&prompt[..240], 30, &[], |_| {}).unwrap();
            let b = pi.generate(&prompt[..240], 30, &[], |_| {}).unwrap();
            assert!(a.tokens == b.tokens && same(&sy.logits(), &pi.logits()), "{} {} {}: pipelined != sync", name, imp.name(), pm.name());
            println!("  {:<28} {} {} pipelined == sync over 30 tokens {:?}..", name, imp.name(), pm.name(), &a.tokens[..6]);
        }
    }
}

fn same(x: &[f32], y: &[f32]) -> bool { x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()) }

/// Feed `prompt` in passes of the sizes `sizes` (cycled), head on every pass; returns (last next token, logits).
fn feed(eng: &mut Engine, prompt: &[u32], sizes: &[usize]) -> (u32, Vec<f32>) {
    eng.reset();
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < prompt.len() {
        let n = sizes[j % sizes.len()].min(prompt.len() - i);
        c = eng.forward(&prompt[i..i + n], true).unwrap().unwrap();
        i += n;
        j += 1;
    }
    (c, eng.logits())
}

#[test]
fn prefill_k_invariant() {
    let mixed: Vec<usize> = (1..=8).collect();
    for (name, sp) in specs() {
        let (cfg, src, _) = build(&sp, 7);
        let mut r = Rng(5);
        // 270 tokens = 33 passes of 8 + one of 6; crosses the 256-key chunk boundary inside a pass
        let prompt: Vec<u32> = (0..270).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        // (impl, pass-size patterns that must give bit-identical logits)
        let cases: [(AttnImpl, Vec<Vec<usize>>); 3] = [
            (AttnImpl::SplitK, vec![vec![8], vec![1], mixed.clone()]),
            (AttnImpl::Sdpa, vec![vec![8], vec![1], mixed.clone()]),
            (AttnImpl::Flash, vec![vec![128], vec![8], vec![1], mixed.clone(), vec![100, 37, 128]]),
        ];
        for (imp, pats) in cases {
            let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 512, attn: imp, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
            let (a, la) = feed(&mut eng, &prompt, &pats[0]);
            for p in &pats[1..] {
                let (b, lb) = feed(&mut eng, &prompt, p);
                assert!(same(&la, &lb), "{} {}: passes {:?} logits != passes {:?}", name, imp.name(), pats[0], p);
                assert_eq!(a, b);
            }
            println!("  {:<24} {:<6} next token {} identical over pass patterns {:?}", name, imp.name(), a, pats);
            // generate() is prefill + steps
            let st = eng.generate(&prompt[..20], 5, &[], |_| {}).unwrap();
            assert_eq!((st.tokens.len(), st.decode_passes, eng.pos()), (5, 4, 24));
        }
        // auto = flash over prompt[..n-1] (passes of <= 128) + one sdpa pass for the last token
        let mut auto = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 512, attn: AttnImpl::Auto, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
        let ta = auto.prefill(&prompt).unwrap();
        let la = auto.logits();
        let mut fl = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 512, attn: AttnImpl::Flash, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
        let mut sd = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 512, attn: AttnImpl::Sdpa, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
        // the same split by hand: flash passes of <= 128 over prompt[..269], then a 1-row (sdpa) pass
        auto.reset();
        for c in prompt[..269].chunks(128) { auto.forward(c, false).unwrap(); }
        let tb = auto.forward(&prompt[269..], true).unwrap().unwrap();
        assert!(same(&la, &auto.logits()) && ta == tb, "{}: auto prefill != flash body + 1-row pass", name);
        // pure flash / pure sdpa engines are not bit-identical to auto but agree within the CPU tolerance
        let tf = fl.prefill(&prompt).unwrap();
        let ts = sd.prefill(&prompt).unwrap();
        let (_, dfl, tol) = tol_diff(&fl.logits(), &la);
        let (_, dsd, _) = tol_diff(&sd.logits(), &la);
        println!("  {:<24} auto next {} (flash {}, sdpa {}); max|diff| vs flash {:.4}, vs sdpa {:.4} (tol {:.4})",
                 name, ta, tf, ts, dfl, dsd, tol);
        assert!(dfl <= tol && dsd <= tol);
    }
}

/// sdpa long-row fallback: rows with N > sdpa128_vec_max_n(grp) (8192 for grp 2) are written by split-K. A prefill
/// crossing N = 8192 inside a pass, then decode steps past it: logits stay within the CPU tolerance of the split-K
/// engine (without the fallback those rows would keep stale attention output).
#[test]
fn sdpa_long_row_fallback() {
    let (name, sp) = specs().remove(0);
    let (cfg, src, _) = build(&sp, 11);
    let mut r = Rng(9);
    let prompt: Vec<u32> = (0..8190).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
    let mut logits = vec![];
    for imp in [AttnImpl::SplitK, AttnImpl::Sdpa] {
        let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 8208, attn: imp, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
        let mut toks = vec![eng.prefill(&prompt).unwrap()];
        // a 5-row pass with rows N = 8191 .. 8195, then two decode steps (N = 8196, 8197)
        let t5: Vec<u32> = (0..5).map(|i| (i * 7 + 3) as u32 % sp.vocab as u32).collect();
        toks.push(eng.forward(&t5, true).unwrap().unwrap());
        let mut l = vec![eng.logits()];
        for t in [5u32, 9] { toks.push(eng.step(t).unwrap()); l.push(eng.logits()); }
        println!("  {} {:<6} tokens {:?}", name, imp.name(), toks);
        logits.push(l);
    }
    for (i, (a, b)) in logits[0].iter().zip(&logits[1]).enumerate() {
        let (scale, d, tol) = tol_diff(b, a);
        println!("  pass {}: sdpa vs splitk max|diff| {:.4} (max|ref| {:.3}, tol {:.4})", i, d, scale, tol);
        assert!(d <= tol && b.iter().all(|v| v.is_finite()));
        assert!(a[argmax(b)] >= a[argmax(a)] - tol);
    }
}

#[test]
fn errors_reported() {
    let (_, sp) = specs().remove(0);
    let (cfg, src, _) = build(&sp, 3);
    let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 10, attn: AttnImpl::SplitK, prefill: PrefillMode::Passes, ..Default::default() }).unwrap();
    assert!(eng.forward(&[1; 9], true).unwrap_err().contains("1..=8"));
    let mut auto = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 10, ..Default::default() }).unwrap();
    assert!(auto.forward(&[1; 129], true).unwrap_err().contains("1..=128"));
    assert!(auto.forward(&[1; 11], true).unwrap_err().contains("context full"));
    assert!(eng.forward(&[sp.vocab as u32], true).unwrap_err().contains("vocab"));
    eng.prefill(&[1; 10]).unwrap();
    assert!(eng.step(1).unwrap_err().contains("context full"));
    // missing tensor / no quantization block
    let mut src2 = MemSource::default();
    for (k, v) in &src.map { if !k.contains("layers.1.mlp.up_proj.scales") { src2.map.insert(k.clone(), v.clone()); } }
    let e = Engine::from_source(cfg.clone(), &src2, EngineOptions::default()).err().unwrap();
    assert!(e.contains("layers.1.mlp.up_proj.scales"), "{}", e);
    let mut c2 = cfg;
    c2.quant = None;
    assert!(Engine::from_source(c2, &src, EngineOptions::default()).err().unwrap().contains("quantization"));
}

/// Tokenizer of the local models (tokenizer.json only; skipped when missing).
#[test]
fn local_tokenizers_round_trip() {
    for m in ["qwen3-1.7b-mlx4", "qwen2.5-1.5b-instruct-mlx4"] {
        let dir = format!("{}/models/{}", env!("CARGO_MANIFEST_DIR"), m);
        if !std::path::Path::new(&dir).join("tokenizer.json").exists() { println!("  SKIP {}: no tokenizer.json", m); continue; }
        let tk = Tokenizer::from_path(&dir).unwrap();
        let text = "Hello, world! The capital of France is Paris.\n  def f(x):\treturn x**2  # 你好";
        let ids = tk.encode(text);
        assert_eq!(tk.decode(&ids), text, "{}", m);
        assert_eq!(tk.encode("<|im_end|>"), vec![151645], "{}", m);
        println!("  {} : {} tokens, first {:?}", m, ids.len(), &ids[..6]);
    }
}

/// Weight layout of the local models from their safetensors HEADERS only (names, dtypes, shapes; no tensor data is
/// read, no GPU memory): every tensor the engine needs is present with the expected shape. Skipped when missing.
#[test]
fn local_models_layout() {
    for m in ["qwen3-1.7b-mlx4", "qwen2.5-1.5b-instruct-mlx4"] {
        let dir = format!("{}/models/{}", env!("CARGO_MANIFEST_DIR"), m);
        if !std::path::Path::new(&dir).join("config.json").exists() { println!("  SKIP {}: missing", m); continue; }
        let cfg = DenseConfig::from_model_dir(&dir).unwrap();
        let src = dense_engine::weights::SafeSource::open(&dir).unwrap();
        let (w, jobs) = dense_engine::weights::layout(&cfg, &src).unwrap();
        let file_bytes: usize = src.idx.iter().map(|(_, t)| t.nbytes).sum();
        println!("  {}: {} jobs, weight region {:.1} MB (tensors in file {:.1} MB), tied {}, qkv rows {}, bias {}",
                 m, jobs.len(), w.bytes as f64 / 1e6, file_bytes as f64 / 1e6, w.tied, w.layers[0].qkv.rows,
                 w.layers[0].qkv_bias.is_some());
        assert!(w.tied && w.layers.len() == cfg.layers);
        assert_eq!(w.layers[0].qkv.rows, cfg.attn_in_rows());
        // every tensor of the file is used exactly once (no lm_head, nothing left over)
        assert_eq!(jobs.len(), src.idx.len(), "{}", m);
    }
}

/// Pipelined decode (`DecodeMode::Pipelined`: pass t+1 encoded and committed while pass t runs, the token id chained
/// on the GPU) == the synchronous step loop (`DecodeMode::Sync`), bit for bit: same tokens, same final logits, same
/// pos, for every attention impl / prefill mode; across the 256-key chunk boundary during decode; with a stop token
/// (look-ahead pass discarded; the next step continues identically); max_new 1 / 2; context full (same tokens handed
/// out, same error).
#[test]
fn decode_pipelined_bit_identical() {
    use dense_engine::engine::DecodeMode;
    let opts = |imp, pm, decode, max_ctx| EngineOptions { max_ctx, attn: imp, prefill: pm, decode };
    for (name, sp) in specs() {
        let (cfg, src, _) = build(&sp, 0x5eed);
        let mut r = Rng(77);
        let prompt: Vec<u32> = (0..240).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        for (imp, pm) in IMPLS {
            let tag = format!("{} {} {}", name, imp.name(), pm.name());
            let mut sy = Engine::from_source(cfg.clone(), &src, opts(imp, pm, DecodeMode::Sync, 320)).unwrap();
            let mut pi = Engine::from_source(cfg.clone(), &src, opts(imp, pm, DecodeMode::Pipelined, 320)).unwrap();
            let run = |e: &mut Engine, p: &[u32], n: usize, stop: &[u32]| {
                let mut seen = vec![];
                let st = e.generate(p, n, stop, |t| seen.push(t)).unwrap();
                assert_eq!(seen, st.tokens);
                (st.tokens, st.decode_passes, e.pos(), e.logits())
            };
            // 40 tokens from pos 240: decode crosses the 256-key chunk boundary
            let a = run(&mut sy, &prompt, 40, &[]);
            let b = run(&mut pi, &prompt, 40, &[]);
            assert_eq!((&a.0, a.1, a.2), (&b.0, b.1, b.2), "{}: tokens / passes / pos", tag);
            assert!(same(&a.3, &b.3), "{}: final logits differ", tag);
            assert_eq!((a.1, a.2), (39, 279));
            // stop on the token first produced at decode pass >= 10 (both runs stop at its first occurrence)
            let s = a.0[10];
            let first = a.0.iter().position(|&t| t == s).unwrap();
            let a2 = run(&mut sy, &prompt, 40, &[s]);
            let b2 = run(&mut pi, &prompt, 40, &[s]);
            assert_eq!(a2.0, a.0[..=first].to_vec(), "{}: sync stop", tag);
            assert_eq!((&a2.0, a2.1, a2.2), (&b2.0, b2.1, b2.2), "{}: stop: tokens / passes / pos", tag);
            assert!(same(&a2.3, &b2.3), "{}: stop: logits differ", tag);
            // continue after the stop: the discarded look-ahead pass must not leave a trace
            for t in [3u32, s, 1] {
                let (x, y) = (sy.step(t).unwrap(), pi.step(t).unwrap());
                assert!(x == y && same(&sy.logits(), &pi.logits()), "{}: step {} after stop", tag, t);
            }
            for n in [1, 2, 3] {
                let (a3, b3) = (run(&mut sy, &prompt[..20], n, &[]), run(&mut pi, &prompt[..20], n, &[]));
                assert_eq!((&a3.0, a3.1, a3.2), (&b3.0, b3.1, b3.2), "{}: max_new {}", tag, n);
                assert!(same(&a3.3, &b3.3), "{}: max_new {} logits", tag, n);
            }
            // context full: 300-token cache, prompt 240 + 80 wanted -> 61 tokens fit, then the same error
            let full = |e: &mut Engine| {
                let mut seen = vec![];
                let err = e.generate(&prompt, 80, &[], |t| seen.push(t)).unwrap_err();
                (seen, err, e.pos())
            };
            let mut sy3 = Engine::from_source(cfg.clone(), &src, opts(imp, pm, DecodeMode::Sync, 300)).unwrap();
            let mut pi3 = Engine::from_source(cfg.clone(), &src, opts(imp, pm, DecodeMode::Pipelined, 300)).unwrap();
            let (fa, fb) = (full(&mut sy3), full(&mut pi3));
            assert_eq!(fa, fb, "{}: context full", tag);
            assert!(fa.1.contains("context full") && fa.0.len() == 61 && fa.2 == 300, "{}: {:?}", tag, (fa.0.len(), &fa.1, fa.2));
            assert!(same(&sy3.logits(), &pi3.logits()), "{}: context full logits", tag);
            println!("  {:<44} pipelined == sync: 40 tokens {:?}.., stop {} after {}, max_new 1..3, context full",
                     tag, &a.0[..6], s, a2.0.len());
        }
    }
}

/// Sliding-window decode past the window. Gemma-3-shaped model: 5 sliding layers (own RoPE theta)
/// : 1 full layer, hd 256, 4 q heads / 1 kv head, window 16. Prompts longer than the window (37; 530: two GEMM prefill
/// chunks; 1030: sdpa_vec two-pass rows) and shorter (10), then greedy decode steps across / past the window. Every
/// decode step's logits vs the CPU reference; the last step vs a one-shot prefill of prompt + continuation; the
/// reference with window 15 / 17 must miss the tolerance (the check sees an off-by-one in the window); pipelined
/// decode == sync. Every attention impl / prefill mode for the short prompts, auto + GEMM prefill for the long ones.
#[test]
fn sliding_window_decode_past_window() {
    use dense_engine::engine::DecodeMode;
    let w = 16;
    let mut x = Ext::fam(Family::Gemma3, 256);
    (x.qk_norm, x.gelu, x.plus1, x.four, x.embed_scale) = (true, true, true, true, 256f64.sqrt());
    x.scale = Some(256f64.powf(-0.5));
    (x.window, x.sliding, x.local_theta) = (Some(w), (0..6).map(|l| l % 6 != 5).collect(), Some(10000.0));
    let sp = Spec { qwen3: true, hidden: 256, inter: 512, layers: 6, heads: 4, kv: 1, vocab: 384, f16: false, x };
    let (cfg, src, m) = build(&sp, 0x3a5c_1f00_d00d_0001);
    cfg.validate().unwrap();
    let mut r = Rng(17);
    let all: Vec<u32> = (0..1030).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
    for (plen, steps) in [(37usize, 20usize), (10, 20), (530, 6), (1030, 4)] {
        let prompt = &all[..plen];
        let mut cpu = CpuRef::new(&m, &sp);
        let mut c = vec![];
        for &t in prompt { c = cpu.step(t); }
        // the CPU's own greedy continuation, fed to every engine
        let mut cont = vec![];
        let mut cls = vec![];
        for _ in 0..steps {
            let t = argmax(&c) as u32;
            c = cpu.step(t);
            cont.push(t);
            cls.push(c.clone());
        }
        let full: Vec<u32> = prompt.iter().chain(&cont).copied().collect();
        // references with the window one shorter / longer
        let off: Vec<(usize, Vec<f32>)> = [w - 1, w + 1].iter().map(|&w2| {
            let mut x2 = sp.x.clone();
            x2.window = Some(w2);
            let mut c2 = CpuRef::with(&m, &sp, x2);
            let mut l = vec![];
            for &t in &full { l = c2.step(t); }
            (w2, l)
        }).collect();
        let impls: Vec<(AttnImpl, PrefillMode)> = if plen > 512 { vec![(AttnImpl::Auto, PrefillMode::Gemm)] } else { IMPLS.to_vec() };
        for (imp, pm) in impls {
            let tag = format!("window {} prompt {} {} {}", w, plen, imp.name(), pm.name());
            let opts = |decode| EngineOptions { max_ctx: plen + steps + 8, attn: imp, prefill: pm, decode };
            let mut eng = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Sync)).unwrap();
            eng.prefill(prompt).unwrap();
            for (i, (&t, c)) in cont.iter().zip(&cls).enumerate() {
                let tok = eng.step(t).unwrap();
                compare(&format!("{} decode {} (pos {})", tag, i + 1, plen + i), &eng.logits(), c, tok);
            }
            let dec = eng.logits();
            for (w2, l) in &off {
                let (_, d, tol) = tol_diff(&dec, l);
                assert!(d > tol, "{}: decode logits also fit the window-{} reference ({} <= {})", tag, w2, d, tol);
            }
            let mut one = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Sync)).unwrap();
            let t1 = one.prefill(&full).unwrap();
            compare(&format!("{} one-shot prefill {}", tag, full.len()), &one.logits(), cls.last().unwrap(), t1);
            let (_, d, tol) = tol_diff(&dec, &one.logits());
            println!("  {:<40} decode vs one-shot prefill max|diff| {:.4} (tol {:.4})", tag, d, tol);
            assert!(d <= tol, "{}: decode vs one-shot prefill max|diff| {} > {}", tag, d, tol);
            let mut sy = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Sync)).unwrap();
            let mut pi = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Pipelined)).unwrap();
            let a = sy.generate(prompt, steps, &[], |_| {}).unwrap();
            let b = pi.generate(prompt, steps, &[], |_| {}).unwrap();
            assert!(a.tokens == b.tokens && same(&sy.logits(), &pi.logits()), "{}: pipelined != sync", tag);
        }
    }
}

/// Unquantized embedding (MLX 4-bit Gemma 1 keeps `embed_tokens.weight` F16 [vocab][hidden]):
/// Gemma-1-style model (GeGLU, (1 + w) norms, embedding scale, hd 256 MQA) with a tied F16 / BF16 embedding, every
/// other linear 4-bit. Layout: dense bf16 embedding, every tensor used; logits after a 300-token prefill and 3 decode
/// steps vs the CPU reference for every attention impl / prefill mode; a 4-bit embedding still loads 4-bit.
#[test]
fn unquantized_embedding_tied() {
    let mut x = Ext::fam(Family::Gemma, 256);
    (x.gelu, x.plus1, x.embed_scale) = (true, true, 256f64.sqrt());
    let sp = Spec { qwen3: true, hidden: 256, inter: 512, layers: 2, heads: 2, kv: 1, vocab: 384, f16: false, x };
    for dt in [DType::F16, DType::BF16] {
        let (cfg, mut src, mut m) = build(&sp, 0x0e3b_ed00_f16e_0001);
        let (w4, _) = dense_engine::weights::layout(&cfg, &src).unwrap();
        assert!(!w4.embed_bf16);
        // replace the 4-bit embedding with a dense one (bf16-exact values, also exact in f16)
        let mut r = Rng(0xe3b);
        let ew: Vec<f32> = (0..sp.vocab * sp.hidden).map(|_| val(&mut r, 0.0, 0.1)).collect();
        for sfx in ["weight", "scales", "biases"] { src.map.remove(&format!("model.embed_tokens.{}", sfx)); }
        store(&mut src, "model.embed_tokens.weight", &[sp.vocab, sp.hidden], &ew, dt == DType::F16);
        m.embed = Mat { rows: sp.vocab, cols: sp.hidden, w: ew };
        let (w, jobs) = dense_engine::weights::layout(&cfg, &src).unwrap();
        assert!(w.embed_bf16 && w.tied && w.lm_head == w.embed && jobs.len() == src.map.len(), "{:?} layout", dt);
        let mut r = Rng(44);
        let prompt: Vec<u32> = (0..300).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        let mut cpu = CpuRef::new(&m, &sp);
        let mut cl = vec![];
        for &t in &prompt { cl = cpu.step(t); }
        let mut cont = vec![];
        let mut cls = vec![];
        let mut c = cl.clone();
        for _ in 0..3 {
            let t = argmax(&c) as u32;
            c = cpu.step(t);
            cont.push(t);
            cls.push(c.clone());
        }
        for (imp, pm) in IMPLS {
            let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 320, attn: imp, prefill: pm, ..Default::default() }).unwrap();
            let tag = format!("{:?} embedding {} {}", dt, imp.name(), pm.name());
            let tok = eng.prefill(&prompt).unwrap();
            compare(&format!("{} prefill 300", tag), &eng.logits(), &cl, tok);
            for (i, (&t, c)) in cont.iter().zip(&cls).enumerate() {
                let tok = eng.step(t).unwrap();
                compare(&format!("{} decode {}", tag, i + 1), &eng.logits(), c, tok);
            }
        }
    }
}
