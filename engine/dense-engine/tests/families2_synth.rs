//! Self-test of the second batch of dense-family features on SYNTHETIC tiny models (no real model is loaded): random
//! MLX 4-bit weights in a `MemSource`, every feature set on a `DenseConfig` the way a family adapter does (start from
//! `DenseConfig::base`, set fields), tensors named by the config's `TensorNames`. One 2-layer model per family shape:
//! - phi3: fused qkv_proj / gate_up_proj, LongRoPE (su: long factors + q / k multiplier), GQA hd 128; the same at
//!   hd 96 (Phi-3 / Phi-3.5-mini) and hd 96 with partial rotary;
//! - olmo2: post-norm-only layers, q / k RMSNorm over the whole projection, q/k/v/o bias, hd 64;
//! - smollm3: a no-RoPE layer (NoPE), hd 64 GQA 4/1;
//! - glm4: four norms under GLM names, fused gate_up, traditional (interleaved) partial RoPE (half), q/k/v bias;
//! - stablelm (parallel): LayerNorm with bias, parallel residual, per-head LayerNorm QK-norm, partial RoPE (quarter),
//!   q/k/v bias; the same at hd 160 (stablelm-2-12b, rope 40 dims); stablelm (sequential): LayerNorm with bias,
//!   sequential layer, partial RoPE;
//! - cohere2: LayerNorm without bias, parallel block (branches summed first), traditional RoPE on the sliding layer,
//!   no RoPE on the full layer, window 37, tied head, logit scale; a biased variant (o_proj / MLP biases);
//! - starcoder2: LayerNorm with bias, non-gated MLP c_fc -> exact-erf GELU -> c_proj, biases on every linear, tied.
//!
//! Mistral is mlx_lm's llama (covered by the Llama paths of engine_synth.rs).
//!
//! Checks (as engine_synth.rs `families_vs_cpu_reference`): logits after a 300-token prefill and 3 decode steps vs a
//! CPU f32 reference (bf16 rounding at the kernels' points) for every attention impl with engine-pass prefill and the
//! GEMM prefill (auto, splitk); per feature an ablated reference (that one feature off) must miss the tolerance;
//! K-invariance (passes of 8 / 1 / mixed, bit-identical) and pipelined == sync decode for the new prep variants;
//! weight layout (every tensor used once, a missing tensor named); config validation messages.
//!
//! Run: cargo test --release --test families2_synth -- --nocapture --test-threads=1   (inside engine/dense-engine)
use dense_engine::config::{Activation, Block, DenseConfig, Family, LayerKind, Mlp, NormKind, QkNorm, SuRope};
use dense_engine::engine::{DecodeMode, PrefillMode};
use dense_engine::model::AttnImpl;
use dense_engine::weights::MemSource;
use dense_engine::{Engine, EngineOptions};
use streamdense_core::DType;

// ------------------------------------------------------------------------------------------- bf16 / rng
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

// ------------------------------------------------------------------------------------------- synthetic weights
struct Mat { rows: usize, cols: usize, w: Vec<f32> }

fn val(r: &mut Rng, center: f32, spread: f32) -> f32 {
    let mut x = center + spread * r.sym();
    if x.abs() < 1e-3 { x = if x < 0.0 { -1e-3 } else { 1e-3 }; }
    rbf(x)
}
fn store(src: &mut MemSource, name: &str, shape: &[usize], v: &[f32]) {
    src.insert(name, DType::BF16, shape, v.iter().flat_map(|x| to_bf(*x).to_le_bytes()).collect());
}
/// Random 4-bit linear `prefix` [rows][cols] (MLX layout) -> MemSource, returns the dequantized matrix.
fn q4(src: &mut MemSource, r: &mut Rng, prefix: &str, rows: usize, cols: usize) -> Mat {
    let g = cols / 64;
    let mut words = vec![0u32; rows * cols / 8];
    let (mut s, mut b) = (vec![0f32; rows * g], vec![0f32; rows * g]);
    let mut w = vec![0f32; rows * cols];
    let sc0 = 0.9 / (cols as f32).sqrt() / 4.3;
    for i in 0..rows {
        for gi in 0..g {
            s[i * g + gi] = val(r, sc0, sc0 * 0.3);
            b[i * g + gi] = val(r, -7.5 * s[i * g + gi], sc0);
        }
        for c in 0..cols {
            let q = (r.next() % 16) as u32;
            words[(i * cols + c) / 8] |= q << (4 * (c % 8));
            w[i * cols + c] = (q as f32).mul_add(s[i * g + c / 64], b[i * g + c / 64]);
        }
    }
    src.insert(&format!("{}.weight", prefix), DType::U32, &[rows, cols / 8], words.iter().flat_map(|x| x.to_le_bytes()).collect());
    store(src, &format!("{}.scales", prefix), &[rows, g], &s);
    store(src, &format!("{}.biases", prefix), &[rows, g], &b);
    Mat { rows, cols, w }
}
fn vecw(src: &mut MemSource, r: &mut Rng, name: &str, n: usize, center: f32, spread: f32) -> Vec<f32> {
    let v: Vec<f32> = (0..n).map(|_| val(r, center, spread)).collect();
    store(src, name, &[n], &v);
    v
}
fn rows(m: &Mat, r0: usize, n: usize) -> Mat { Mat { rows: n, cols: m.cols, w: m.w[r0 * m.cols..(r0 + n) * m.cols].to_vec() } }
fn cat(m: &[&Mat]) -> Mat { Mat { rows: m.iter().map(|x| x.rows).sum(), cols: m[0].cols, w: m.iter().flat_map(|x| x.w.iter().copied()).collect() } }

/// Norm weight + optional LayerNorm bias.
#[derive(Clone)]
struct Nw { w: Vec<f32>, b: Option<Vec<f32>> }

struct RefLayer {
    in_norm: Option<Nw>,
    /// FFN input norm of a sequential layer (post_attention_layernorm, or pre_ffn with four norms)
    post_norm: Option<Nw>,
    /// on the attention / FFN output (four norms, post-norm)
    post_attn: Option<Nw>,
    post_ffw: Option<Nw>,
    qkv: Mat,
    bias: Option<Vec<f32>>,
    /// HeadRms [hd]; FullRms [q dim] / [kv dim]; HeadLayerNorm [heads][hd]
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    o: Mat,
    gate: Option<Mat>,
    up: Mat,
    down: Mat,
    o_bias: Option<Vec<f32>>,
    gate_bias: Option<Vec<f32>>,
    up_bias: Option<Vec<f32>>,
    down_bias: Option<Vec<f32>>,
}
struct RefModel { embed: Mat, norm: Nw, layers: Vec<RefLayer>, lm_head: Option<Mat> }

/// Every tensor `cfg` names (as an MLX conversion of that family would hold them) into a MemSource + the reference
/// weights.
fn build(cfg: &DenseConfig, seed: u64) -> (MemSource, RefModel) {
    let mut r = Rng(seed);
    let mut src = MemSource::default();
    let nm = &cfg.names;
    let (h, hd, inter) = (cfg.hidden, cfg.head_dim, cfg.inter);
    let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
    let nbias = matches!(cfg.norm, NormKind::LayerNorm { bias: true });
    let norm = |src: &mut MemSource, r: &mut Rng, name: &str| Nw {
        w: vecw(src, r, &format!("{}.weight", name), h, 1.0, 0.2),
        b: if nbias { Some(vecw(src, r, &format!("{}.bias", name), h, 0.0, 0.3)) } else { None },
    };
    let embed = q4(&mut src, &mut r, &nm.embed, cfg.vocab, h);
    let fnorm = norm(&mut src, &mut r, &nm.final_norm);
    let mut layers = vec![];
    for i in 0..cfg.layers {
        let p = format!("{}.{}", nm.layer_prefix, i);
        let t = |n: &str| format!("{}.{}", p, n);
        let (mut in_norm, mut post_norm, mut post_attn, mut post_ffw) = (None, None, None, None);
        match cfg.block {
            Block::Sequential => {
                in_norm = Some(norm(&mut src, &mut r, &t(&nm.input_norm)));
                if cfg.four_norms {
                    post_attn = Some(norm(&mut src, &mut r, &t(&nm.post_attn_norm)));
                    post_norm = Some(norm(&mut src, &mut r, &t(&nm.pre_ffn_norm)));
                    post_ffw = Some(norm(&mut src, &mut r, &t(&nm.post_ffn_norm)));
                } else {
                    post_norm = Some(norm(&mut src, &mut r, &t(&nm.post_attn_norm)));
                }
            }
            Block::Parallel | Block::ParallelBranchesFirst => in_norm = Some(norm(&mut src, &mut r, &t(&nm.input_norm))),
            Block::PostNorm => {
                post_attn = Some(norm(&mut src, &mut r, &t(&nm.post_attn_norm)));
                post_ffw = Some(norm(&mut src, &mut r, &t(&nm.post_ffn_norm)));
            }
        }
        let (qkv, bias) = match &nm.qkv {
            Some(f) => {
                let m = q4(&mut src, &mut r, &t(f), qd + 2 * kvd, h);
                let b = if cfg.attention_bias { Some(vecw(&mut src, &mut r, &format!("{}.bias", t(f)), qd + 2 * kvd, 0.0, 0.5)) } else { None };
                (m, b)
            }
            None => {
                let q = q4(&mut src, &mut r, &t(&nm.q), qd, h);
                let k = q4(&mut src, &mut r, &t(&nm.k), kvd, h);
                let v = q4(&mut src, &mut r, &t(&nm.v), kvd, h);
                let b = if cfg.attention_bias {
                    let mut b = vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.q)), qd, 0.0, 0.5);
                    b.extend(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.k)), kvd, 0.0, 0.5));
                    b.extend(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.v)), kvd, 0.0, 0.5));
                    Some(b)
                } else { None };
                (cat(&[&q, &k, &v]), b)
            }
        };
        let (q_norm, k_norm) = if !cfg.qk_norm { (None, None) } else {
            let (qn, kn) = (t(&nm.q_norm), t(&nm.k_norm));
            match cfg.qk_norm_kind {
                QkNorm::HeadRms => (Some(vecw(&mut src, &mut r, &format!("{}.weight", qn), hd, 1.0, 0.2)),
                                    Some(vecw(&mut src, &mut r, &format!("{}.weight", kn), hd, 1.0, 0.2))),
                // weights well away from 1 so the norm visibly rescales the scores
                QkNorm::FullRms => (Some(vecw(&mut src, &mut r, &format!("{}.weight", qn), qd, 2.0, 1.0)),
                                    Some(vecw(&mut src, &mut r, &format!("{}.weight", kn), kvd, 2.0, 1.0))),
                QkNorm::HeadLayerNorm => {
                    let mut per = |base: &str, n: usize| -> Vec<f32> {
                        (0..n).flat_map(|j| vecw(&mut src, &mut r, &format!("{}.norms.{}.weight", base, j), hd, 1.0, 0.4)).collect()
                    };
                    (Some(per(&qn, cfg.heads)), Some(per(&kn, cfg.kv_heads)))
                }
            }
        };
        let o = q4(&mut src, &mut r, &t(&nm.o), h, qd);
        let o_bias = if cfg.o_bias { Some(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.o)), h, 0.0, 0.5)) } else { None };
        let (mut gate, mut gate_bias, mut up_bias) = (None, None, None);
        let up = match (&nm.gate_up, cfg.mlp) {
            (Some(gu), Mlp::Gated) => {
                let m = q4(&mut src, &mut r, &t(gu), 2 * inter, h);
                if cfg.mlp_bias {
                    let b = vecw(&mut src, &mut r, &format!("{}.bias", t(gu)), 2 * inter, 0.0, 0.5);
                    (gate_bias, up_bias) = (Some(b[..inter].to_vec()), Some(b[inter..].to_vec()));
                }
                gate = Some(rows(&m, 0, inter));
                rows(&m, inter, inter)
            }
            (_, Mlp::Gated) => {
                gate = Some(q4(&mut src, &mut r, &t(&nm.gate), inter, h));
                let u = q4(&mut src, &mut r, &t(&nm.up), inter, h);
                if cfg.mlp_bias {
                    gate_bias = Some(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.gate)), inter, 0.0, 0.5));
                    up_bias = Some(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.up)), inter, 0.0, 0.5));
                }
                u
            }
            (_, Mlp::Plain) => {
                let u = q4(&mut src, &mut r, &t(&nm.up), inter, h);
                if cfg.mlp_bias { up_bias = Some(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.up)), inter, 0.0, 0.5)); }
                u
            }
        };
        let down = q4(&mut src, &mut r, &t(&nm.down), h, inter);
        let down_bias = if cfg.mlp_bias { Some(vecw(&mut src, &mut r, &format!("{}.bias", t(&nm.down)), h, 0.0, 0.5)) } else { None };
        layers.push(RefLayer { in_norm, post_norm, post_attn, post_ffw, qkv, bias, q_norm, k_norm, o, gate, up,
                               down, o_bias, gate_bias, up_bias, down_bias });
    }
    let lm_head = if cfg.tie_word_embeddings { None } else { Some(q4(&mut src, &mut r, &nm.lm_head, cfg.vocab, h)) };
    (src, RefModel { embed, norm: fnorm, layers, lm_head })
}

// ------------------------------------------------------------------------------------------- CPU reference
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let r = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| rbf(w * rbf(v * r))).collect()
}
/// mx.fast.layer_norm on bf16: y = bf((x - m) r), out = bf(w y [+ b]) (w y + b in f32).
fn layernorm(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let m: f32 = x.iter().sum::<f32>() / n;
    let v: f32 = x.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / n;
    let r = 1.0 / (v + eps).sqrt();
    (0..x.len()).map(|i| {
        let y = rbf((x[i] - m) * r);
        rbf(w[i] * y + b.map(|b| b[i]).unwrap_or(0.0))
    }).collect()
}
fn matvec(m: &Mat, x: &[f32]) -> Vec<f32> {
    (0..m.rows).map(|i| m.w[i * m.cols..(i + 1) * m.cols].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}
fn silu(g: f32) -> f32 {
    let e = rbf(g.abs().exp());
    let y = rbf(1.0 / rbf(1.0 + e));
    let s = if g < 0.0 { y } else { rbf(1.0 - y) };
    rbf(g * s)
}
fn gelu_tanh(g: f32) -> f32 { rbf(0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh())) }
/// Single-precision erf (njuffa's erff, as the kernel and MLX's Metal erf).
fn erf32(a: f32) -> f32 {
    let (t, s) = (a.abs(), a * a);
    if t > 0.927734375 {
        let mut r = (-1.72853470e-5f32).mul_add(t, 3.83197126e-4);
        let u = (-3.88396438e-3f32).mul_add(t, 2.42546219e-2);
        r = r.mul_add(s, u);
        r = r.mul_add(t, -1.06777877e-1);
        r = r.mul_add(t, -6.34846687e-1);
        r = r.mul_add(t, -1.28717512e-1);
        r = r.mul_add(t, -t);
        (1.0 - r.exp()).copysign(a)
    } else {
        let mut r = -5.96761703e-4f32;
        for c in [4.99119423e-3, -2.67681349e-2, 1.12819925e-1, -3.76125336e-1, 1.28379166e-1] { r = r.mul_add(s, c); }
        r.mul_add(a, a)
    }
}
/// mlx nn.gelu on bf16: every op rounded (x / bf16(sqrt 2), erf, 1 +, x *, / 2).
fn gelu_erf(g: f32) -> f32 {
    let a = rbf(g / 1.4140625);
    let s = rbf(1.0 + rbf(erf32(a)));
    rbf(rbf(g * s) * 0.5)
}
fn act(a: &Activation, g: f32) -> f32 {
    match a { Activation::GeluTanh => gelu_tanh(g), Activation::GeluErf => gelu_erf(g), _ => silu(g) }
}

struct CpuRef<'a> { m: &'a RefModel, c: DenseConfig, kc: Vec<Vec<Vec<f32>>>, vc: Vec<Vec<Vec<f32>>>, invf: Vec<f32> }

impl<'a> CpuRef<'a> {
    /// Reference with the math of `c` (the engine's config, or an ablated copy) over the weights `m`.
    fn new(m: &'a RefModel, c: DenseConfig) -> CpuRef<'a> {
        let n = c.rope_dims / 2;
        let invf = match &c.rope_su {
            Some(su) => (0..n).map(|i| (1.0 / (c.rope_theta.powf(i as f64 / n as f64) * su.long_factor[i])) as f32).collect(),
            // the GPU kernel's f32 formula exp2(-(i / n) * log2(theta))
            None => (0..n).map(|i| (-(i as f32 / n as f32) * (c.rope_theta as f32).log2()).exp2()).collect(),
        };
        CpuRef { m, kc: vec![vec![]; c.layers], vc: vec![vec![]; c.layers], invf, c }
    }
    fn norm(&self, x: &[f32], n: &Nw) -> Vec<f32> {
        match self.c.norm {
            NormKind::Rms => rmsnorm(x, &n.w, self.c.eps),
            NormKind::LayerNorm { bias } => layernorm(x, &n.w, if bias { n.b.as_deref() } else { None }, self.c.eps),
        }
    }
    /// [su multiplier] + RoPE (rotate-half or traditional) over the first rope_dims dims.
    fn rope(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let rd = self.c.rope_dims;
        let mut x = x.to_vec();
        if let Some(su) = &self.c.rope_su {
            let ms = rbf(su.mscale as f32);
            for v in x[..rd].iter_mut() { *v = rbf(*v * ms); }
        }
        let mut o = x.clone();
        for i in 0..rd / 2 {
            let (s, c) = (pos as f32 * self.invf[i]).sin_cos();
            let (a, b) = if self.c.rope_traditional { (2 * i, 2 * i + 1) } else { (i, i + rd / 2) };
            o[a] = rbf(x[a] * c - x[b] * s);
            o[b] = rbf(x[a] * s + x[b] * c);
        }
        o
    }
    /// out = bf(r [+ bias]) [-> post norm]
    fn branch(&self, r: Vec<f32>, bias: &Option<Vec<f32>>, post: Option<&Nw>) -> Vec<f32> {
        let mut r: Vec<f32> = r.into_iter().map(rbf).collect();
        if let Some(b) = bias { for (v, b) in r.iter_mut().zip(b) { *v = rbf(*v + b); } }
        match post { Some(n) => self.norm(&r, n), None => r }
    }
    fn ffn(&self, lw: &RefLayer, x: &[f32]) -> Vec<f32> {
        let bias = |v: Vec<f32>, b: &Option<Vec<f32>>| -> Vec<f32> {
            let v: Vec<f32> = v.into_iter().map(rbf).collect();
            match (b, self.c.mlp_bias) { (Some(b), true) => v.iter().zip(b).map(|(a, b)| rbf(a + b)).collect(), _ => v }
        };
        let u = bias(matvec(&lw.up, x), &lw.up_bias);
        let f: Vec<f32> = match (self.c.mlp, &lw.gate) {
            (Mlp::Gated, Some(gm)) => {
                let g = bias(matvec(gm, x), &lw.gate_bias);
                g.iter().zip(&u).map(|(g, u)| rbf(act(&self.c.activation, *g) * u)).collect()
            }
            _ => u.iter().map(|v| act(&self.c.activation, *v)).collect(),
        };
        let db = if self.c.mlp_bias { &lw.down_bias } else { &None };
        self.branch(matvec(&lw.down, &f), db, None)
    }
    fn step(&mut self, tok: u32) -> Vec<f32> {
        let c = &self.c;
        let (h, nh, nkv, hd) = (c.hidden, c.heads, c.kv_heads, c.head_dim);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let pos = self.kc[0].len();
        let mut x: Vec<f32> = self.m.embed.w[tok as usize * h..(tok as usize + 1) * h].iter().map(|v| rbf(*v)).collect();
        let scale = (hd as f32).powf(-0.5);
        for (l, lw) in self.m.layers.iter().enumerate() {
            let sliding = c.kinds[l] == LayerKind::Sliding;
            // no input norm in the file (PostNorm, or its ablation to an un-normed parallel layer): h itself
            let ain = match (&lw.in_norm, c.block) { (Some(n), b) if b != Block::PostNorm => self.norm(&x, n), _ => x.clone() };
            let mut p: Vec<f32> = matvec(&lw.qkv, &ain).into_iter().map(rbf).collect();
            if let (Some(b), true) = (&lw.bias, c.attention_bias) { for (v, b) in p.iter_mut().zip(b) { *v = rbf(*v + b); } }
            if c.qk_norm && c.qk_norm_kind == QkNorm::FullRms {
                let q = rmsnorm(&p[..qd], lw.q_norm.as_ref().unwrap(), c.eps);
                let k = rmsnorm(&p[qd..qd + kvd], lw.k_norm.as_ref().unwrap(), c.eps);
                p[..qd].copy_from_slice(&q);
                p[qd..qd + kvd].copy_from_slice(&k);
            }
            let head = |i: usize| p[i * hd..(i + 1) * hd].to_vec();
            let qk = |v: Vec<f32>, w: &Option<Vec<f32>>, j: usize| -> Vec<f32> {
                if !c.qk_norm { return v; }
                match c.qk_norm_kind {
                    QkNorm::HeadRms => rmsnorm(&v, w.as_ref().unwrap(), c.eps),
                    QkNorm::FullRms => v,
                    QkNorm::HeadLayerNorm => layernorm(&v, &w.as_ref().unwrap()[j * hd..(j + 1) * hd], None, c.eps),
                }
            };
            let rope_on = c.rope_on(l);
            let rp = |v: Vec<f32>| if rope_on { self.rope(&v, pos) } else { v };
            let qs: Vec<Vec<f32>> = (0..nh).map(|i| rp(qk(head(i), &lw.q_norm, i))).collect();
            let (mut ks, mut vs) = (vec![], vec![]);
            for j in 0..nkv {
                ks.extend(rp(qk(head(nh + j), &lw.k_norm, j)));
                vs.extend(head(nh + nkv + j));
            }
            self.kc[l].push(ks);
            self.vc[l].push(vs);
            let n = pos + 1;
            let lo = match (sliding, c.sliding_window) { (true, Some(w)) => n.saturating_sub(w), _ => 0 };
            let mut ao = vec![0f32; qd];
            for i in 0..nh {
                let j = i / (nh / nkv);
                let sc: Vec<f32> = (lo..n).map(|t| {
                    let k = &self.kc[l][t][j * hd..(j + 1) * hd];
                    qs[i].iter().zip(k).map(|(a, b)| scale * a * b).sum::<f32>()
                }).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let pe: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                let lsum: f32 = pe.iter().sum();
                for d in 0..hd {
                    let o: f32 = (lo..n).map(|t| pe[t - lo] * self.vc[l][t][j * hd + d]).sum();
                    ao[i * hd + d] = rbf(o / lsum);
                }
            }
            let ob = if c.o_bias { &lw.o_bias } else { &None };
            let add = |x: &mut Vec<f32>, r: &[f32]| for (a, b) in x.iter_mut().zip(r) { *a = rbf(*a + b); };
            match c.block {
                Block::Sequential => {
                    let a = self.branch(matvec(&lw.o, &ao), ob, if c.four_norms { lw.post_attn.as_ref() } else { None });
                    add(&mut x, &a);
                    // ablation Parallel -> Sequential: no post norm in the file, reuse the input norm
                    let fin = self.norm(&x, lw.post_norm.as_ref().or(lw.in_norm.as_ref()).unwrap());
                    let mut f = self.ffn(lw, &fin);
                    if c.four_norms { f = self.norm(&f, lw.post_ffw.as_ref().unwrap()); }
                    add(&mut x, &f);
                }
                Block::PostNorm => {
                    let a = self.branch(matvec(&lw.o, &ao), ob, lw.post_attn.as_ref());
                    add(&mut x, &a);
                    let f = self.ffn(lw, &x);
                    let f = self.norm(&f, lw.post_ffw.as_ref().unwrap());
                    add(&mut x, &f);
                }
                Block::Parallel => {
                    let a = self.branch(matvec(&lw.o, &ao), ob, None);
                    let f = self.ffn(lw, &ain);
                    add(&mut x, &a);
                    add(&mut x, &f);
                }
                Block::ParallelBranchesFirst => {
                    let mut a = self.branch(matvec(&lw.o, &ao), ob, None);
                    let f = self.ffn(lw, &ain);
                    add(&mut a, &f);
                    add(&mut x, &a);
                }
            }
        }
        let hf = self.norm(&x, &self.m.norm);
        let mut lg = matvec(self.m.lm_head.as_ref().unwrap_or(&self.m.embed), &hf);
        if c.logits_scaling != 1.0 { for v in lg.iter_mut() { *v /= c.logits_scaling as f32; } }
        lg
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..v.len() { if v[i] > v[b] { b = i; } }
    b
}
fn tol_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32, f32) {
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs()));
    let maxd = gpu.iter().zip(cpu).fold(0f32, |a, (g, c)| a.max((g - c).abs()));
    (scale, maxd, 0.015 * scale + 0.01)
}
fn compare(what: &str, gpu: &[f32], cpu: &[f32], gpu_tok: u32) {
    let (scale, maxd, tol) = tol_diff(gpu, cpu);
    let ct = argmax(cpu);
    println!("  {:<46} max|ref| {:7.3}  max|diff| {:.4} (tol {:.4})  argmax gpu {} cpu {}", what, scale, maxd, tol, gpu_tok, ct);
    assert!(maxd <= tol, "{}: max |diff| {} > {}", what, maxd, tol);
    let gb: Vec<f32> = gpu.iter().map(|v| rbf(*v)).collect();
    assert_eq!(argmax(&gb) as u32, gpu_tok, "{}: argmax_b != host argmax of the bf16-rounded logits", what);
    assert!(cpu[gpu_tok as usize] >= cpu[ct] - tol, "{}: gpu token {} is not a cpu near-max", what, gpu_tok);
}
fn same(x: &[f32], y: &[f32]) -> bool { x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()) }

// ------------------------------------------------------------------------------------------- family configs
/// A 2-layer model of `family` set up as its adapter would (DenseConfig::base + fields).
fn base(family: Family, mt: &str, heads: usize, kv: usize, hd: usize) -> DenseConfig {
    let mut c = DenseConfig::base(mt, family, 256, 512, 2, heads, kv, hd, 384);
    c.eos = vec![1];
    c.rope_theta = 10000.0;
    c
}

/// (name, config, ablations: (feature, config with that feature off)).
type Case = (&'static str, DenseConfig, Vec<(&'static str, DenseConfig)>);

fn ablate(c: &DenseConfig, f: impl Fn(&mut DenseConfig)) -> DenseConfig { let mut a = c.clone(); f(&mut a); a }

fn cases() -> Vec<Case> {
    let mut v: Vec<Case> = vec![];
    // Phi-3 (128k-style): fused qkv / gate_up, LongRoPE factors 1..4, mscale from max 131072 / original 4096
    let mut c = base(Family::Phi3, "phi3", 4, 2, 128);
    c.names.qkv = Some("self_attn.qkv_proj".into());
    c.names.gate_up = Some("mlp.gate_up_proj".into());
    let mut r = Rng(0x5eed_0001);
    let lf: Vec<f64> = (0..64).map(|i| 1.0 + 3.0 * (i as f64 / 63.0) + 0.1 * r.sym() as f64).collect();
    c.rope_su = Some(SuRope::new(lf, 131072, 4096, None));
    let ab = vec![("su long factors", ablate(&c, |a| a.rope_su.as_mut().unwrap().long_factor.iter_mut().for_each(|f| *f = 1.0))),
                  ("su q/k multiplier", ablate(&c, |a| a.rope_su.as_mut().unwrap().mscale = 1.0))];
    v.push(("phi3 (fused, su rope) hd128 4/2", c, ab));
    // Phi-3 / Phi-3.5-mini head dim 96 (full rotary, LongRoPE over 48 frequencies): the split-K kernel runs 3 32-dim
    // slices per row on 4 lanes
    let mut c = base(Family::Phi3, "phi3", 4, 2, 96);
    c.names.qkv = Some("self_attn.qkv_proj".into());
    c.names.gate_up = Some("mlp.gate_up_proj".into());
    let lf: Vec<f64> = (0..48).map(|i| 1.0 + 3.0 * (i as f64 / 47.0) + 0.1 * r.sym() as f64).collect();
    c.rope_su = Some(SuRope::new(lf, 131072, 4096, None));
    let ab = vec![("su long factors", ablate(&c, |a| a.rope_su.as_mut().unwrap().long_factor.iter_mut().for_each(|f| *f = 1.0))),
                  ("su q/k multiplier", ablate(&c, |a| a.rope_su.as_mut().unwrap().mscale = 1.0))];
    v.push(("phi3.5-mini-style (su rope) hd96 4/2", c, ab));
    // head dim 96 with partial rotary (half: 48 dims, as a partial_rotary_factor 0.5 config would give), MHA
    let mut c = base(Family::Phi3, "phi3", 4, 4, 96);
    c.names.qkv = Some("self_attn.qkv_proj".into());
    c.names.gate_up = Some("mlp.gate_up_proj".into());
    c.rope_dims = 48;
    let ab = vec![("partial rotary", ablate(&c, |a| a.rope_dims = 96))];
    v.push(("phi3-style partial rotary hd96 4/4", c, ab));
    // OLMo 2: post-norm-only layers, whole-projection q / k RMSNorm, attention bias on q/k/v and o
    let mut c = base(Family::Olmo2, "olmo2", 4, 2, 64);
    (c.block, c.qk_norm, c.qk_norm_kind, c.attention_bias, c.o_bias) = (Block::PostNorm, true, QkNorm::FullRms, true, true);
    let ab = vec![("post norms", ablate(&c, |a| a.block = Block::Parallel)),
                  ("full q/k norm", ablate(&c, |a| a.qk_norm = false))];
    v.push(("olmo2 (post-norm, full qk-norm) hd64 4/2", c, ab));
    // SmolLM3: layer 1 without RoPE (config no_rope_layers [1, 0] -> ours [false, true])
    let mut c = base(Family::SmolLM3, "smollm3", 4, 1, 64);
    c.no_rope_layers = vec![false, true];
    c.rope_theta = 5e6;
    let ab = vec![("no-RoPE layer", ablate(&c, |a| a.no_rope_layers.clear()))];
    v.push(("smollm3 (NoPE layer) hd64 4/1", c, ab));
    // GLM-4: four norms under GLM names, fused gate_up, traditional RoPE on half the head dim, q/k/v bias
    let mut c = base(Family::Glm4, "glm4", 4, 2, 128);
    c.four_norms = true;
    c.names.post_attn_norm = "post_self_attn_layernorm".into();
    c.names.pre_ffn_norm = "post_attention_layernorm".into();
    c.names.post_ffn_norm = "post_mlp_layernorm".into();
    c.names.gate_up = Some("mlp.gate_up_proj".into());
    (c.rope_traditional, c.rope_dims, c.attention_bias) = (true, 64, true);
    let ab = vec![("traditional rope", ablate(&c, |a| a.rope_traditional = false)),
                  ("partial rotary", ablate(&c, |a| a.rope_dims = 128)),
                  ("four norms", ablate(&c, |a| a.four_norms = false))];
    v.push(("glm4 (4 norms, trad partial rope) hd128 4/2", c, ab));
    // StableLM 2 (parallel residual + qk_layernorm, as stablelm-2-12b): LayerNorm with bias, partial rotary 0.25
    let mut c = base(Family::StableLm, "stablelm", 4, 2, 64);
    (c.norm, c.block, c.eps) = (NormKind::LayerNorm { bias: true }, Block::Parallel, 1e-5);
    (c.qk_norm, c.qk_norm_kind, c.attention_bias, c.rope_dims) = (true, QkNorm::HeadLayerNorm, true, 16);
    c.names.q_norm = "self_attn.q_layernorm".into();
    c.names.k_norm = "self_attn.k_layernorm".into();
    let ab = vec![("layernorm (as rmsnorm)", ablate(&c, |a| a.norm = NormKind::Rms)),
                  ("layernorm bias", ablate(&c, |a| a.norm = NormKind::LayerNorm { bias: false })),
                  ("parallel block", ablate(&c, |a| a.block = Block::Sequential)),
                  ("per-head qk layernorm", ablate(&c, |a| a.qk_norm = false)),
                  ("partial rotary", ablate(&c, |a| a.rope_dims = 64))];
    v.push(("stablelm parallel (LN bias, qk-LN) hd64 4/2", c, ab));
    // stablelm-2-12b shape: head dim 160 (5 slices on 8 lanes in split-K), partial rotary 0.25 (40 dims), parallel
    // residual, per-head LayerNorm QK-norm, GQA
    let mut c = base(Family::StableLm, "stablelm", 4, 2, 160);
    (c.norm, c.block, c.eps) = (NormKind::LayerNorm { bias: true }, Block::Parallel, 1e-5);
    (c.qk_norm, c.qk_norm_kind, c.rope_dims) = (true, QkNorm::HeadLayerNorm, 40);
    c.names.q_norm = "self_attn.q_layernorm".into();
    c.names.k_norm = "self_attn.k_layernorm".into();
    let ab = vec![("parallel block", ablate(&c, |a| a.block = Block::Sequential)),
                  ("per-head qk layernorm", ablate(&c, |a| a.qk_norm = false)),
                  ("partial rotary", ablate(&c, |a| a.rope_dims = 160))];
    v.push(("stablelm-2-12b-style (qk-LN, rope 40) hd160 4/2", c, ab));
    // StableLM (sequential, e.g. stablelm-2-1_6b / zephyr): LayerNorm with bias, q/k/v bias, partial rotary
    let mut c = base(Family::StableLm, "stablelm", 4, 4, 64);
    (c.norm, c.eps, c.attention_bias, c.rope_dims) = (NormKind::LayerNorm { bias: true }, 1e-5, true, 16);
    let ab = vec![("layernorm bias", ablate(&c, |a| a.norm = NormKind::LayerNorm { bias: false }))];
    v.push(("stablelm sequential (LN bias) hd64 4/4", c, ab));
    // Cohere2: LayerNorm without bias, parallel (branches first), traditional RoPE on the sliding layer, no RoPE on
    // the full layer (pattern: layer 1 of 2 is global), window 37, tied head, logit_scale 0.25
    let cohere = |bias: bool| {
        let mut c = base(Family::Cohere2, "cohere2", 4, 2, 64);
        (c.norm, c.block, c.eps, c.rope_traditional) = (NormKind::LayerNorm { bias: false }, Block::ParallelBranchesFirst, 1e-5, true);
        (c.kinds, c.sliding_window, c.no_rope_layers) = (vec![LayerKind::Sliding, LayerKind::Full], Some(37), vec![false, true]);
        (c.tie_word_embeddings, c.logits_scaling) = (true, 1.0 / 0.25);
        (c.o_bias, c.mlp_bias, c.attention_bias) = (bias, bias, bias);
        c
    };
    let c = cohere(false);
    let ab = vec![("layernorm (as rmsnorm)", ablate(&c, |a| a.norm = NormKind::Rms)),
                  ("parallel block", ablate(&c, |a| a.block = Block::Sequential)),
                  ("traditional rope", ablate(&c, |a| a.rope_traditional = false)),
                  ("no-RoPE full layer", ablate(&c, |a| a.no_rope_layers.clear())),
                  ("sliding window", ablate(&c, |a| a.kinds = vec![LayerKind::Full; 2])),
                  ("logit scale", ablate(&c, |a| a.logits_scaling = 1.0))];
    v.push(("cohere2 (LN, parallel, nope full) hd64 4/2", c, ab));
    let c = cohere(true);
    let ab = vec![("o / mlp biases", ablate(&c, |a| (a.o_bias, a.mlp_bias) = (false, false)))];
    v.push(("cohere2-style with o / mlp biases hd64 4/2", c, ab));
    // StarCoder2: LayerNorm with bias, c_fc -> gelu (erf) -> c_proj, biases on every linear, tied
    let mut c = base(Family::Starcoder2, "starcoder2", 4, 2, 64);
    (c.norm, c.eps, c.mlp, c.activation) = (NormKind::LayerNorm { bias: true }, 1e-5, Mlp::Plain, Activation::GeluErf);
    (c.attention_bias, c.o_bias, c.mlp_bias, c.tie_word_embeddings) = (true, true, true, true);
    c.names.up = "mlp.c_fc".into();
    c.names.down = "mlp.c_proj".into();
    let ab = vec![("gelu (as silu)", ablate(&c, |a| a.activation = Activation::Silu)),
                  ("mlp biases", ablate(&c, |a| a.mlp_bias = false)),
                  ("layernorm bias", ablate(&c, |a| a.norm = NormKind::LayerNorm { bias: false }))];
    v.push(("starcoder2 (LN bias, plain gelu mlp) hd64 4/2", c, ab));
    v
}

const IMPLS: [(AttnImpl, PrefillMode); 6] = [
    (AttnImpl::Auto, PrefillMode::Passes), (AttnImpl::SplitK, PrefillMode::Passes), (AttnImpl::Flash, PrefillMode::Passes),
    (AttnImpl::Sdpa, PrefillMode::Passes), (AttnImpl::Auto, PrefillMode::Gemm), (AttnImpl::SplitK, PrefillMode::Gemm),
];

/// Every case vs its CPU reference: logits after a 300-token prefill and 3 decode steps, every attention impl and
/// prefill mode; each listed feature is shown to matter (the reference with it off misses the tolerance).
#[test]
fn families2_vs_cpu_reference() {
    for (name, cfg, ablations) in cases() {
        cfg.validate().unwrap_or_else(|e| panic!("{}: {}", name, e));
        let (src, m) = build(&cfg, 0x51ed_270b_9e37_79b9);
        let mut r = Rng(43);
        let prompt: Vec<u32> = (0..300).map(|_| (r.next() % cfg.vocab as u64) as u32).collect();
        let mut cpu = CpuRef::new(&m, cfg.clone());
        let mut cl = vec![];
        for &t in &prompt { cl = cpu.step(t); }
        let ablated: Vec<(&str, Vec<f32>)> = ablations.iter().map(|(an, ac)| {
            let mut c = CpuRef::new(&m, ac.clone());
            let mut l = vec![];
            for &t in &prompt { l = c.step(t); }
            (*an, l)
        }).collect();
        let (mut cont, mut cls, mut c) = (vec![], vec![], cl.clone());
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
                    println!("  {:<46} without {:<28} max|diff| {:.4} (must exceed {:.4})", tag, an, d2, t2);
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

/// Weight layout of every case (names, dtypes, shapes; nothing loaded): every tensor in the source is used exactly
/// once, fused gate_up is split into views, and a missing tensor of each new kind is named in the error.
#[test]
fn families2_weight_layout() {
    for (name, cfg, _) in cases() {
        let (src, _) = build(&cfg, 5);
        let (w, jobs) = dense_engine::weights::layout(&cfg, &src).unwrap();
        assert_eq!(jobs.len(), src.map.len(), "{}: tensors used != tensors present", name);
        let l0 = &w.layers[0];
        if cfg.names.gate_up.is_some() {
            assert_eq!((l0.gate.rows, l0.up.rows, l0.up.w - l0.gate.w), (cfg.inter, cfg.inter, cfg.inter * cfg.hidden / 2), "{}", name);
        }
        assert_eq!(l0.in_norm_bias.is_some() || l0.post_attn_norm_bias.is_some(), matches!(cfg.norm, NormKind::LayerNorm { bias: true }), "{}", name);
        for missing in ["layers.1.input_layernorm.bias", "model.norm.bias", "layers.1.self_attn.q_layernorm.norms.3",
                        "layers.0.self_attn.qkv_proj.scales", "layers.1.mlp.gate_up_proj.biases", "layers.1.mlp.c_fc.bias",
                        "layers.0.post_mlp_layernorm", "layers.1.post_feedforward_layernorm", "layers.0.self_attn.k_norm"] {
            if !src.map.keys().any(|k| k.contains(missing)) { continue; }
            let mut s2 = MemSource::default();
            for (k, v) in &src.map { if !k.contains(missing) { s2.map.insert(k.clone(), v.clone()); } }
            let e = dense_engine::weights::layout(&cfg, &s2).err().unwrap();
            assert!(e.contains(missing), "{}: {}", name, e);
        }
        println!("  {:<46} {} tensors, all used; region {} KB", name, jobs.len(), w.bytes / 1024);
    }
}

/// The new prep variants (su multiplier, traditional / partial / no RoPE, per-head LayerNorm, whole-projection
/// QK-norm) and layer forms keep the engine's invariants: prefill in passes of 8 == one token per pass == mixed sizes,
/// bit for bit (split-K, sdpa, flash); pipelined decode == sync decode, bit for bit (GEMM and pass prefill).
#[test]
fn families2_k_invariant_and_pipelined() {
    let mixed: Vec<usize> = (1..=8).collect();
    for (name, cfg, _) in cases() {
        let (src, _) = build(&cfg, 0x77);
        let mut r = Rng(6);
        let prompt: Vec<u32> = (0..270).map(|_| (r.next() % cfg.vocab as u64) as u32).collect();
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
        }
        for (imp, pm) in [(AttnImpl::Auto, PrefillMode::Gemm), (AttnImpl::SplitK, PrefillMode::Passes)] {
            let opts = |decode| EngineOptions { max_ctx: 320, attn: imp, prefill: pm, decode };
            let mut sy = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Sync)).unwrap();
            let mut pi = Engine::from_source(cfg.clone(), &src, opts(DecodeMode::Pipelined)).unwrap();
            let a = sy.generate(&prompt[..240], 30, &[], |_| {}).unwrap();
            let b = pi.generate(&prompt[..240], 30, &[], |_| {}).unwrap();
            assert!(a.tokens == b.tokens && same(&sy.logits(), &pi.logits()), "{} {} {}: pipelined != sync", name, imp.name(), pm.name());
        }
        println!("  {:<46} K-invariant (splitk / sdpa / flash) and pipelined == sync", name);
    }
}

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

/// Validation of the new fields: what is accepted, what is named.
#[test]
fn families2_config_validation() {
    for (name, cfg, _) in cases() { cfg.validate().unwrap_or_else(|e| panic!("{}: {}", name, e)); }
    // SuRope::new: mlx_lm default scale sqrt(1 + ln(32) / ln(4096)); explicit long_mscale wins; factor <= 1 -> 1
    let s = SuRope::new(vec![1.0; 48], 131072, 4096, None);
    assert!((s.mscale - (1.0 + 32f64.ln() / 4096f64.ln()).sqrt()).abs() < 1e-12);
    assert_eq!(SuRope::new(vec![], 131072, 4096, Some(1.25)).mscale, 1.25);
    assert_eq!(SuRope::new(vec![], 4096, 4096, None).mscale, 1.0);
    let mut c = base(Family::Phi3, "phi3", 4, 2, 128);
    c.rope_dims = 96;
    c.validate().unwrap(); // partial rotary on a non-Qwen family
    c.rope_dims = 95;
    assert!(c.validate().unwrap_err().contains("must be even"));
    c.rope_dims = 128;
    c.rope_su = Some(SuRope::new(vec![1.0; 10], 131072, 4096, None));
    assert!(c.validate().unwrap_err().contains("rope_su long_factor has 10 entries, want rope_dims / 2 = 64"));
    c.rope_su = Some(SuRope::new(vec![1.0; 64], 131072, 4096, None));
    c.rope_scale = dense_engine::config::RopeScaling::Linear { factor: 2.0 };
    assert!(c.validate().unwrap_err().contains("rope_su together with rope_scale"));
    let mut c = base(Family::Olmo2, "olmo2", 4, 2, 64);
    (c.block, c.four_norms) = (Block::PostNorm, true);
    assert!(c.validate().unwrap_err().contains("four_norms only with Block::Sequential"));
    let mut c = base(Family::SmolLM3, "smollm3", 4, 2, 64);
    c.no_rope_layers = vec![true];
    assert!(c.validate().unwrap_err().contains("no_rope_layers has 1 entries, layers = 2"));
    let mut c = base(Family::Starcoder2, "starcoder2", 4, 2, 64);
    (c.mlp, c.names.gate_up) = (Mlp::Plain, Some("mlp.gate_up_proj".into()));
    assert!(c.validate().unwrap_err().contains("Mlp::Plain with a fused gate_up"));
    // erf of the reference (and kernel) vs a f64 series: < 2 f32 ulps on [-4, 4]
    for i in -400..=400 {
        let x = i as f64 / 100.0;
        let (mut term, mut sum, mut k) = (x, x, 0);
        while term.abs() > 1e-17 && k < 200 { k += 1; term *= -x * x / k as f64; sum += term / (2 * k + 1) as f64; }
        let e = sum * 2.0 / std::f64::consts::PI.sqrt();
        assert!((erf32(x as f32) as f64 - e).abs() <= 2.5e-7 * e.abs().max(1e-3), "erf({}) {} vs {}", x, erf32(x as f32), e);
    }
}
