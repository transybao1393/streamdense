//! GEMM prefill (src/prefill.rs: q4b_gemm projections / FFN + attn128_flash over T rows) vs the engine's decode-pass
//! prefill (passes of <= 8 rows, MMA gemv + split-K), on SYNTHETIC tiny models only (no real model is loaded):
//! - "qwen3": QK-norm, tied, bf16 scales, GQA 4/2;  "qwen2": q/k/v bias, tied, f16 scales, 2/1.
//!
//! For T = 1, 7, 8, 9, 300, 513 (t_max 512, so 513 = two chunks) plus a chunked run (t_max 128) and a continuation
//! (first part by decode passes, rest by GEMM at pos > 0):
//! - KV cache (K = normed + roped key, V = value projection) of every layer at every prompt position — the per-layer
//!   hidden state the decode continues from — within 1.5 % of max |ref| (bf16; summation orders differ);
//! - final logits within 1.5 % of max |logit| + 0.01 (the CPU-reference bar of engine_synth.rs) and the SAME argmax.
//!   The bar is shown to discriminate: changing the prompt's last token moves the logits beyond it.
//!
//! Benches (ignored by default; `cargo test --release --test prefill_gemm -- --ignored --nocapture bench` runs both: GEMM vs
//! pass prefill, and `bench_decode` = decode Sync vs Pipelined): Qwen3-1.7B
//! shapes (hidden 2048, inter 6144, 28 layers, 16/8 heads, vocab 32768 instead of 151936 to stay < 1.2 GB; the head
//! runs once per prefill in both paths), random weights generated straight into the arena (no RAM copy).
use dense_engine::config::DenseConfig;
use dense_engine::weights::{MemSource, TensorSource};
use dense_engine::gpu;
use dense_engine::{Engine, EngineOptions};
use streamdense_core::DType;

use dense_engine::engine::PrefillMode;
use dense_engine::model::AttnImpl;
use dense_engine::prefill::{Prefill, PrefillModel, PREFILL_SHADER_SOURCE};

/// The reference: engine passes of <= 8 rows (MMA gemv + split-K), not the engine's default GEMM prefill.
fn passes(max_ctx: usize) -> EngineOptions { EngineOptions { max_ctx, attn: AttnImpl::SplitK, prefill: PrefillMode::Passes, ..Default::default() } }

// ------------------------------------------------------------------------------------------- helpers (engine_synth)
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

struct Spec { qwen3: bool, hidden: usize, inter: usize, layers: usize, heads: usize, kv: usize, vocab: usize, f16: bool }

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
fn q4(src: &mut MemSource, r: &mut Rng, prefix: &str, rows: usize, cols: usize, f16: bool) {
    let g = cols / 64;
    let mut words = vec![0u32; rows * cols / 8];
    let (mut s, mut b) = (vec![0f32; rows * g], vec![0f32; rows * g]);
    let sc0 = 0.9 / (cols as f32).sqrt() / 4.3;
    for i in 0..rows {
        for gi in 0..g {
            s[i * g + gi] = val(r, sc0, sc0 * 0.3);
            b[i * g + gi] = val(r, -7.5 * s[i * g + gi], sc0);
        }
        for c in 0..cols { words[(i * cols + c) / 8] |= ((r.next() % 16) as u32) << (4 * (c % 8)); }
    }
    src.insert(&format!("{}.weight", prefix), DType::U32, &[rows, cols / 8], words.iter().flat_map(|x| x.to_le_bytes()).collect());
    store(src, &format!("{}.scales", prefix), &[rows, g], &s, f16);
    store(src, &format!("{}.biases", prefix), &[rows, g], &b, f16);
}
fn vecw(src: &mut MemSource, r: &mut Rng, name: &str, n: usize, center: f32, spread: f32, f16: bool) {
    let v: Vec<f32> = (0..n).map(|_| val(r, center, spread)).collect();
    store(src, name, &[n], &v, f16);
}
fn cfg_json(qwen3: bool, h: usize, inter: usize, layers: usize, heads: usize, kv: usize, vocab: usize) -> DenseConfig {
    let mut c = serde_json::json!({
        "model_type": if qwen3 { "qwen3" } else { "qwen2" }, "hidden_size": h, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
        "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
        "hidden_act": "silu", "eos_token_id": 1, "max_position_embeddings": 40960,
        "quantization": {"group_size": 64, "bits": 4},
    });
    if qwen3 { c["head_dim"] = 128.into(); c["attention_bias"] = false.into(); }
    DenseConfig::from_json(&c, None).unwrap()
}
fn build(sp: &Spec, seed: u64) -> (DenseConfig, MemSource) {
    let mut r = Rng(seed);
    let mut src = MemSource::default();
    let (h, hd) = (sp.hidden, 128);
    let (qd, kvd, f16) = (sp.heads * hd, sp.kv * hd, sp.f16);
    q4(&mut src, &mut r, "model.embed_tokens", sp.vocab, h, f16);
    vecw(&mut src, &mut r, "model.norm.weight", h, 1.0, 0.2, f16);
    for l in 0..sp.layers {
        let p = format!("model.layers.{}", l);
        let a = format!("{}.self_attn", p);
        vecw(&mut src, &mut r, &format!("{}.input_layernorm.weight", p), h, 1.0, 0.2, f16);
        vecw(&mut src, &mut r, &format!("{}.post_attention_layernorm.weight", p), h, 1.0, 0.2, f16);
        q4(&mut src, &mut r, &format!("{}.q_proj", a), qd, h, f16);
        q4(&mut src, &mut r, &format!("{}.k_proj", a), kvd, h, f16);
        q4(&mut src, &mut r, &format!("{}.v_proj", a), kvd, h, f16);
        if !sp.qwen3 {
            vecw(&mut src, &mut r, &format!("{}.q_proj.bias", a), qd, 0.0, 0.5, f16);
            vecw(&mut src, &mut r, &format!("{}.k_proj.bias", a), kvd, 0.0, 0.5, f16);
            vecw(&mut src, &mut r, &format!("{}.v_proj.bias", a), kvd, 0.0, 0.5, f16);
        } else {
            vecw(&mut src, &mut r, &format!("{}.q_norm.weight", a), hd, 1.0, 0.2, f16);
            vecw(&mut src, &mut r, &format!("{}.k_norm.weight", a), hd, 1.0, 0.2, f16);
        }
        q4(&mut src, &mut r, &format!("{}.o_proj", a), h, qd, f16);
        q4(&mut src, &mut r, &format!("{}.mlp.gate_proj", p), sp.inter, h, f16);
        q4(&mut src, &mut r, &format!("{}.mlp.up_proj", p), sp.inter, h, f16);
        q4(&mut src, &mut r, &format!("{}.mlp.down_proj", p), h, sp.inter, f16);
    }
    (cfg_json(sp.qwen3, h, sp.inter, sp.layers, sp.heads, sp.kv, sp.vocab), src)
}

fn specs() -> Vec<(&'static str, Spec)> {
    vec![
        ("qwen3 (qk-norm, bf16)", Spec { qwen3: true, hidden: 256, inter: 512, layers: 2, heads: 4, kv: 2, vocab: 384, f16: false }),
        ("qwen2 (bias, f16)", Spec { qwen3: false, hidden: 256, inter: 640, layers: 2, heads: 2, kv: 1, vocab: 320, f16: true }),
    ]
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..v.len() { if v[i] > v[b] { b = i; } }
    b
}

fn model(eng: &Engine) -> PrefillModel<'_> {
    PrefillModel { cfg: &eng.cfg, w: &eng.w, kv: eng.kv.iter().map(|k| (k.k, k.v)).collect(), max_ctx: eng.max_ctx,
                   arena: &eng.arena.buf }
}

/// K and V caches (bf16 as f32) of every layer, positions 0 .. n.
fn kv_dump(eng: &Engine, n: usize) -> Vec<f32> {
    let (nkv, mc) = (eng.cfg.kv_heads, eng.max_ctx);
    let mut out = vec![];
    for kv in &eng.kv {
        for base in [kv.k, kv.v] {
            let all: Vec<u16> = eng.arena.get(base, nkv * mc * 128);
            for j in 0..nkv { out.extend(all[j * mc * 128..(j * mc + n) * 128].iter().map(|&x| bf(x))); }
        }
    }
    out
}
fn clear_kv(eng: &mut Engine) {
    let len = eng.cfg.kv_heads * eng.max_ctx * 128 * 2;
    let offs: Vec<usize> = eng.kv.iter().flat_map(|k| [k.k, k.v]).collect();
    for o in offs { eng.arena.bytes_mut(o, len).fill(0); }
}

/// (max |ref|, max |diff|, fraction of elements that differ)
fn diff(a: &[f32], r: &[f32]) -> (f32, f32, f64) {
    assert_eq!(a.len(), r.len());
    let scale = r.iter().fold(0f32, |m, v| m.max(v.abs()));
    let maxd = a.iter().zip(r).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let nd = a.iter().zip(r).filter(|(x, y)| x != y).count();
    (scale, maxd, nd as f64 / a.len().max(1) as f64)
}

const KV_TOL: f32 = 0.015;
fn logit_tol(scale: f32) -> f32 { 0.015 * scale + 0.01 }

/// One case: decode-pass prefill of `prompt` (reference) vs [decode passes for prompt[..split]] + GEMM prefill of the
/// rest at pos = split, chunks of t_max.
fn case(name: &str, eng: &mut Engine, gpu: &mut gpu::Gpu, prompt: &[u32], split: usize, t_max: usize) {
    let n = prompt.len();
    eng.reset();
    let ta = eng.prefill(prompt).unwrap();
    let la = eng.logits();
    let kva = kv_dump(eng, n);
    clear_kv(eng);
    eng.reset();
    if split > 0 { eng.prefill(&prompt[..split]).unwrap(); }
    let pf = Prefill::new(gpu, &eng.cfg, t_max).unwrap();
    let tb = pf.run(gpu, &model(eng), &prompt[split..], split, true).unwrap().unwrap();
    let lb = pf.logits();
    let kvb = kv_dump(eng, n);
    let (ks, kd, kf) = diff(&kvb, &kva);
    let (ls, ld, lf) = diff(&lb, &la);
    println!("  {:<22} T {:>3} (split {:>3}, t_max {:>3}): KV max|ref| {:6.3} max|diff| {:.4} ({:5.2}% differ) | logits max|ref| {:6.3} max|diff| {:.4} (tol {:.4}, {:5.1}% differ) | argmax {} {}",
             name, n, split, t_max, ks, kd, 100.0 * kf, ls, ld, logit_tol(ls), 100.0 * lf, ta, tb);
    assert!(kd <= KV_TOL * ks, "{} T {}: KV max |diff| {} > {}", name, n, kd, KV_TOL * ks);
    assert!(ld <= logit_tol(ls), "{} T {}: logits max |diff| {} > {}", name, n, ld, logit_tol(ls));
    assert_eq!(ta, tb, "{} T {}: argmax differs", name, n);
    assert_eq!(argmax(&lb) as u32, tb, "{} T {}: argmax_b != host argmax", name, n);
}

#[test]
fn gemm_prefill_vs_decode_passes() {
    for (name, sp) in specs() {
        let (cfg, src) = build(&sp, 0x9e3779b97f4a7c15);
        let mut eng = Engine::from_source(cfg, &src, passes(768)).unwrap();
        let mut gpu = gpu::Gpu::from_sources(&PREFILL_SHADER_SOURCE).unwrap();
        let mut r = Rng(42);
        let prompt: Vec<u32> = (0..513).map(|_| (r.next() % sp.vocab as u64) as u32).collect();
        for t in [1, 7, 8, 9, 300, 513] { case(name, &mut eng, &mut gpu, &prompt[..t], 0, 512); }
        case(name, &mut eng, &mut gpu, &prompt[..300], 0, 128); // 3 chunks, flash at pos 128 / 256
        case(name, &mut eng, &mut gpu, &prompt[..300], 100, 512); // decode passes then GEMM at pos 100
        // the logits bar discriminates: another last token moves the logits beyond it
        let mut p2 = prompt[..300].to_vec();
        eng.reset();
        eng.prefill(&p2).unwrap();
        let la = eng.logits();
        p2[299] = (p2[299] + 1) % sp.vocab as u32;
        eng.reset();
        eng.prefill(&p2).unwrap();
        let (s, d, _) = diff(&eng.logits(), &la);
        println!("  {:<22} last token changed: logits max|diff| {:.4} (must exceed tol {:.4})", name, d, logit_tol(s));
        assert!(d > logit_tol(s));
    }
}

#[test]
fn errors_reported() {
    let (_, sp) = specs().remove(0);
    let (cfg, src) = build(&sp, 3);
    let eng = Engine::from_source(cfg, &src, passes(16)).unwrap();
    let mut gpu = gpu::Gpu::from_sources(&PREFILL_SHADER_SOURCE).unwrap();
    let pf = Prefill::new(&mut gpu, &eng.cfg, 8).unwrap();
    let m = model(&eng);
    assert!(pf.run(&gpu, &m, &[], 0, true).unwrap_err().contains("no tokens"));
    assert!(pf.run(&gpu, &m, &[1; 17], 0, true).unwrap_err().contains("context full"));
    assert!(pf.run(&gpu, &m, &[sp.vocab as u32], 0, true).unwrap_err().contains("vocab"));
    assert_eq!(pf.run(&gpu, &m, &[1; 16], 0, false).unwrap(), None);
}

// ------------------------------------------------------------------------------------------- bench
/// Random MLX 4-bit tensors generated on read (no stored copy): hash of (name, element) -> data.
struct GenSource { map: std::collections::HashMap<String, (DType, Vec<usize>)> }

fn hash(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn name_seed(s: &str) -> u64 { s.bytes().fold(0xcbf29ce484222325u64, |h, c| (h ^ c as u64).wrapping_mul(0x100000001b3)) }
fn hsym(a: u64, i: u64) -> f32 { 2.0 * ((hash(a, i) >> 40) as f32 / (1u64 << 24) as f32) - 1.0 }

impl GenSource {
    fn new() -> GenSource { GenSource { map: Default::default() } }
    fn q4(&mut self, p: &str, rows: usize, cols: usize) {
        self.map.insert(format!("{}.weight", p), (DType::U32, vec![rows, cols / 8]));
        self.map.insert(format!("{}.scales", p), (DType::BF16, vec![rows, cols / 64]));
        self.map.insert(format!("{}.biases", p), (DType::BF16, vec![rows, cols / 64]));
    }
    fn v(&mut self, n: &str, len: usize) { self.map.insert(n.into(), (DType::BF16, vec![len])); }
}

impl TensorSource for GenSource {
    fn info(&self, name: &str) -> Option<(DType, Vec<usize>)> { self.map.get(name).cloned() }
    fn read(&self, name: &str, dst: &mut [u8]) -> Result<(), String> {
        let (dt, shape) = self.map.get(name).ok_or(format!("{} missing", name))?;
        let sd = name_seed(name);
        if *dt == DType::U32 {
            for (i, c) in dst.chunks_exact_mut(8).enumerate() { c.copy_from_slice(&hash(sd, i as u64).to_le_bytes()); }
            return Ok(());
        }
        let wr = |dst: &mut [u8], f: &dyn Fn(u64) -> f32| {
            for (i, c) in dst.chunks_exact_mut(2).enumerate() { c.copy_from_slice(&to_bf(f(i as u64)).to_le_bytes()); }
        };
        if let Some(p) = name.strip_suffix(".scales").or_else(|| name.strip_suffix(".biases")) {
            let sc0 = 0.9 / ((shape[1] * 64) as f32).sqrt() / 4.3;
            let ss = name_seed(&format!("{}.scales", p));
            let s = move |i: u64| rbf(sc0 + 0.3 * sc0 * hsym(ss, i));
            if name.ends_with(".scales") { wr(dst, &s); } else { wr(dst, &|i| -7.5 * s(i) + sc0 * hsym(sd, i)); }
        } else {
            wr(dst, &|i| 1.0 + 0.2 * hsym(sd, i));
        }
        Ok(())
    }
}

/// Synthetic Qwen3-1.7B-shape source (random weights generated on read).
fn qwen3_1p7b_shapes() -> (DenseConfig, GenSource, usize) {
    let (h, inter, layers, heads, kv, vocab) = (2048, 6144, 28, 16, 8, 32768);
    let cfg = cfg_json(true, h, inter, layers, heads, kv, vocab);
    let mut src = GenSource::new();
    src.q4("model.embed_tokens", vocab, h);
    src.v("model.norm.weight", h);
    for l in 0..layers {
        let p = format!("model.layers.{}", l);
        let a = format!("{}.self_attn", p);
        src.v(&format!("{}.input_layernorm.weight", p), h);
        src.v(&format!("{}.post_attention_layernorm.weight", p), h);
        src.q4(&format!("{}.q_proj", a), heads * 128, h);
        src.q4(&format!("{}.k_proj", a), kv * 128, h);
        src.q4(&format!("{}.v_proj", a), kv * 128, h);
        src.v(&format!("{}.q_norm.weight", a), 128);
        src.v(&format!("{}.k_norm.weight", a), 128);
        src.q4(&format!("{}.o_proj", a), h, heads * 128);
        src.q4(&format!("{}.mlp.gate_proj", p), inter, h);
        src.q4(&format!("{}.mlp.up_proj", p), inter, h);
        src.q4(&format!("{}.mlp.down_proj", p), h, inter);
    }
    (cfg, src, vocab)
}

/// Decode tok/s, `DecodeMode::Sync` vs `Pipelined` (Qwen3-1.7B shapes, random weights, vocab 32768 so the head is
/// ~4.6x cheaper than the real one; 128 tokens after a 64-token prompt, best of 3). Run alone, GPU idle:
/// `cargo test --release --test prefill_gemm -- --ignored --nocapture bench_decode`.
#[test]
#[ignore]
fn bench_decode_pipelined_qwen3_1p7b_shapes() {
    use dense_engine::engine::DecodeMode;
    let (cfg, src, vocab) = qwen3_1p7b_shapes();
    let mut r = Rng(3);
    let prompt: Vec<u32> = (0..64).map(|_| (r.next() % vocab as u64) as u32).collect();
    let mut toks = vec![];
    for decode in [DecodeMode::Sync, DecodeMode::Pipelined] {
        let mut eng = Engine::from_source(cfg.clone(), &src, EngineOptions { max_ctx: 256, decode, ..Default::default() }).unwrap();
        eng.generate(&prompt, 8, &[], |_| {}).unwrap(); // warm-up
        let mut best = 0f64;
        let mut t = vec![];
        for _ in 0..3 {
            let st = eng.generate(&prompt, 128, &[], |_| {}).unwrap();
            best = best.max(st.decode_tok_s());
            t = st.tokens;
        }
        println!("  decode {:<9} {:7.1} tok/s (best of 3, 127 passes)", decode.name(), best);
        toks.push(t);
        drop(eng);
    }
    assert_eq!(toks[0], toks[1], "pipelined tokens != sync tokens");
    println!("  peak footprint {:.0} MB", dense_engine::engine::peak_footprint() as f64 / 1e6);
}

/// Qwen3-1.7B shapes, random weights; prefill tok/s GEMM vs decode passes (T = 512, 1024). Run alone, GPU idle.
#[test]
#[ignore]
fn bench_qwen3_1p7b_shapes() {
    let (cfg, src, vocab) = qwen3_1p7b_shapes();
    let mut eng = Engine::from_source(cfg, &src, passes(1024)).unwrap();
    let mut gpu = gpu::Gpu::from_sources(&PREFILL_SHADER_SOURCE).unwrap();
    let pf = Prefill::new(&mut gpu, &eng.cfg, 512).unwrap();
    println!("bench: arena {:.0} MB, prefill scratch {:.1} MB, load {:.1} s", eng.arena_bytes as f64 / 1e6,
             pf.bytes() as f64 / 1e6, eng.load_s);
    let mut r = Rng(9);
    let prompt: Vec<u32> = (0..1024).map(|_| (r.next() % vocab as u64) as u32).collect();
    let _ = pf.run(&gpu, &model(&eng), &prompt[..64], 0, true).unwrap(); // warm-up
    for t in [512usize, 1024] {
        let p = &prompt[..t];
        let (mut tg, mut tm) = (f64::MAX, f64::MAX);
        let (mut a, mut b) = (0, 0);
        for _ in 0..2 {
            eng.reset();
            let t0 = std::time::Instant::now();
            a = eng.prefill(p).unwrap();
            tg = tg.min(t0.elapsed().as_secs_f64());
            let t0 = std::time::Instant::now();
            b = pf.run(&gpu, &model(&eng), p, 0, true).unwrap().unwrap();
            tm = tm.min(t0.elapsed().as_secs_f64());
        }
        println!("  T {:>4}: gemv passes of 8 {:7.1} ms = {:6.0} tok/s | GEMM (t_max 512) {:7.1} ms = {:6.0} tok/s | {:.2}x | next tok {} / {}",
                 t, tg * 1e3, t as f64 / tg, tm * 1e3, t as f64 / tm, tg / tm, a, b);
    }
    println!("  peak footprint {:.0} MB", dense_engine::engine::peak_footprint() as f64 / 1e6);
}

