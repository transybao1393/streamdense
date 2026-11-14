//! End-to-end self-test of the qwen-dense engine on SYNTHETIC tiny models (no real model is loaded): random MLX
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
//! Run: cargo test --release --test engine_synth -- --nocapture --test-threads=1   (inside engine/qwen-dense)
use qwen_dense::config::DenseConfig;
use qwen_dense::engine::PrefillMode;
use qwen_dense::model::AttnImpl;
use qwen_dense::weights::MemSource;
use qwen_dense::{Engine, EngineOptions, Tokenizer};
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
struct Spec { qwen3: bool, hidden: usize, inter: usize, layers: usize, heads: usize, kv: usize, vocab: usize, f16: bool }

/// One dequantized matrix (row-major [rows][cols], f32) as the CPU reference sees it.
struct Mat { rows: usize, cols: usize, w: Vec<f32> }

struct RefLayer { in_norm: Vec<f32>, post_norm: Vec<f32>, qkv: Mat, bias: Option<Vec<f32>>, q_norm: Option<Vec<f32>>,
                  k_norm: Option<Vec<f32>>, o: Mat, gate: Mat, up: Mat, down: Mat }
struct RefModel { embed: Mat, norm: Vec<f32>, layers: Vec<RefLayer> }

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
    let (h, hd) = (sp.hidden, 128);
    let (qd, kvd) = (sp.heads * hd, sp.kv * hd);
    let f16 = sp.f16;
    let embed = q4(&mut src, &mut r, "model.embed_tokens", sp.vocab, h, f16);
    let norm = vecw(&mut src, &mut r, "model.norm.weight", h, 1.0, 0.2, f16);
    let mut layers = vec![];
    for l in 0..sp.layers {
        let p = format!("model.layers.{}", l);
        let a = format!("{}.self_attn", p);
        let in_norm = vecw(&mut src, &mut r, &format!("{}.input_layernorm.weight", p), h, 1.0, 0.2, f16);
        let post_norm = vecw(&mut src, &mut r, &format!("{}.post_attention_layernorm.weight", p), h, 1.0, 0.2, f16);
        let q = q4(&mut src, &mut r, &format!("{}.q_proj", a), qd, h, f16);
        let k = q4(&mut src, &mut r, &format!("{}.k_proj", a), kvd, h, f16);
        let v = q4(&mut src, &mut r, &format!("{}.v_proj", a), kvd, h, f16);
        let bias = if !sp.qwen3 {
            let mut b = vecw(&mut src, &mut r, &format!("{}.q_proj.bias", a), qd, 0.0, 0.5, f16);
            b.extend(vecw(&mut src, &mut r, &format!("{}.k_proj.bias", a), kvd, 0.0, 0.5, f16));
            b.extend(vecw(&mut src, &mut r, &format!("{}.v_proj.bias", a), kvd, 0.0, 0.5, f16));
            Some(b)
        } else { None };
        let (q_norm, k_norm) = if sp.qwen3 {
            (Some(vecw(&mut src, &mut r, &format!("{}.q_norm.weight", a), hd, 1.0, 0.2, f16)),
             Some(vecw(&mut src, &mut r, &format!("{}.k_norm.weight", a), hd, 1.0, 0.2, f16)))
        } else { (None, None) };
        let o = q4(&mut src, &mut r, &format!("{}.o_proj", a), h, qd, f16);
        let gate = q4(&mut src, &mut r, &format!("{}.mlp.gate_proj", p), sp.inter, h, f16);
        let up = q4(&mut src, &mut r, &format!("{}.mlp.up_proj", p), sp.inter, h, f16);
        let down = q4(&mut src, &mut r, &format!("{}.mlp.down_proj", p), h, sp.inter, f16);
        layers.push(RefLayer { in_norm, post_norm, qkv: cat(&[&q, &k, &v]), bias, q_norm, k_norm, o, gate, up, down });
    }
    let mut cfg = serde_json::json!({
        "model_type": if sp.qwen3 { "qwen3" } else { "qwen2" }, "hidden_size": h, "intermediate_size": sp.inter,
        "num_hidden_layers": sp.layers, "num_attention_heads": sp.heads, "num_key_value_heads": sp.kv,
        "vocab_size": sp.vocab, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
        "hidden_act": "silu", "eos_token_id": 1, "max_position_embeddings": 4096,
        "quantization": {"group_size": 64, "bits": 4},
    });
    if sp.qwen3 { cfg["head_dim"] = 128.into(); cfg["attention_bias"] = false.into(); }
    let cfg = DenseConfig::from_json(&cfg, None).unwrap();
    (cfg, src, RefModel { embed, norm, layers })
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

struct CpuRef<'a> { m: &'a RefModel, sp: &'a Spec, kc: Vec<Vec<Vec<f32>>>, vc: Vec<Vec<Vec<f32>>>, invf: Vec<f32> }

impl<'a> CpuRef<'a> {
    fn new(m: &'a RefModel, sp: &'a Spec) -> CpuRef<'a> {
        let invf = (0..64).map(|i| (-(i as f32 / 64.0) * 1e6f32.log2()).exp2()).collect();
        CpuRef { m, sp, kc: vec![vec![]; sp.layers], vc: vec![vec![]; sp.layers], invf }
    }
    fn rope(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let mut o = vec![0f32; 128];
        for i in 0..64 {
            let th = pos as f32 * self.invf[i];
            let (s, c) = th.sin_cos();
            o[i] = rbf(x[i] * c - x[i + 64] * s);
            o[i + 64] = rbf(x[i] * s + x[i + 64] * c);
        }
        o
    }
    /// Feed one token; returns logits.
    fn step(&mut self, tok: u32) -> Vec<f32> {
        let sp = self.sp;
        let (h, nh, nkv) = (sp.hidden, sp.heads, sp.kv);
        let pos = self.kc[0].len();
        let e = &self.m.embed;
        let mut x: Vec<f32> = e.w[tok as usize * h..(tok as usize + 1) * h].iter().map(|v| rbf(*v)).collect();
        for (l, lw) in self.m.layers.iter().enumerate() {
            let xn = rmsnorm(&x, &lw.in_norm, 1e-6);
            let mut p: Vec<f32> = matvec(&lw.qkv, &xn).into_iter().map(rbf).collect();
            if let Some(b) = &lw.bias { for (v, b) in p.iter_mut().zip(b) { *v = rbf(*v + b); } }
            let head = |i: usize| p[i * 128..(i + 1) * 128].to_vec();
            let qs: Vec<Vec<f32>> = (0..nh).map(|i| {
                let q = head(i);
                let q = match &lw.q_norm { Some(w) => rmsnorm(&q, w, 1e-6), None => q };
                self.rope(&q, pos)
            }).collect();
            let mut ks = vec![];
            let mut vs = vec![];
            for j in 0..nkv {
                let k = head(nh + j);
                let k = match &lw.k_norm { Some(w) => rmsnorm(&k, w, 1e-6), None => k };
                ks.extend(self.rope(&k, pos));
                vs.extend(head(nh + nkv + j));
            }
            self.kc[l].push(ks);
            self.vc[l].push(vs);
            let n = pos + 1;
            let mut ao = vec![0f32; nh * 128];
            let scale = 0.08838834764831845f32;
            for i in 0..nh {
                let j = i / (nh / nkv);
                let sc: Vec<f32> = (0..n).map(|t| {
                    let k = &self.kc[l][t][j * 128..(j + 1) * 128];
                    qs[i].iter().zip(k).map(|(a, b)| scale * a * b).sum::<f32>()
                }).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let pe: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                let lsum: f32 = pe.iter().sum();
                for d in 0..128 {
                    let o: f32 = (0..n).map(|t| pe[t] * self.vc[l][t][j * 128 + d]).sum();
                    ao[i * 128 + d] = rbf(o / lsum);
                }
            }
            let o = matvec(&lw.o, &ao);
            for (a, b) in x.iter_mut().zip(o) { *a = rbf(*a + rbf(b)); }
            let xn = rmsnorm(&x, &lw.post_norm, 1e-6);
            let g = matvec(&lw.gate, &xn);
            let u = matvec(&lw.up, &xn);
            let f: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu_mul(*g, *u)).collect();
            let d = matvec(&lw.down, &f);
            for (a, b) in x.iter_mut().zip(d) { *a = rbf(*a + rbf(b)); }
        }
        let hf = rmsnorm(&x, &self.m.norm, 1e-6);
        matvec(&self.m.embed, &hf)
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
        ("qwen3 (qk-norm, bf16)", Spec { qwen3: true, hidden: 256, inter: 512, layers: 2, heads: 4, kv: 2, vocab: 384, f16: false }),
        ("qwen2 (bias, f16)", Spec { qwen3: false, hidden: 256, inter: 640, layers: 2, heads: 2, kv: 1, vocab: 320, f16: true }),
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
        let src = qwen_dense::weights::SafeSource::open(&dir).unwrap();
        let (w, jobs) = qwen_dense::weights::layout(&cfg, &src).unwrap();
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
    use qwen_dense::engine::DecodeMode;
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
