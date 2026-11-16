//! 8-bit KV fast prefill (attn_prep_q8 + attn_flash_q8 in the GEMM prefill passes) on the tiny model
//! (models/tiny_qwen35), against the bf16 fast prefill of the same prompt:
//! - KV: the first attention layer's K / V inputs are the same bits in both engines (only DeltaNet layers before it),
//!   so its q8 bytes and fp16 scale / min must equal the q8 quantization (attn_prep_q8's arithmetic, on the CPU) of
//!   the bf16 engine's KV, bit for bit. Later attention layers: dequantized q8 within q8 rounding plus the flash
//!   numerics of the earlier layers (reported in q8 steps).
//! - tokens: greedy decode after the q8 fast prefill vs after the bf16 fast prefill (and vs the q8 exact prefill):
//!   first token, first difference and the bf16 top-1 / top-2 margin there; prefill tok/s of the three.
//! Run: cargo test --release --offline -p streamdense-model --test q8_fast_prefill -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_core::f16::{f16_to_f32, f32_to_f16};
use streamdense_model::{KvFormat, LayerKind, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const MAX_CTX: usize = 4096;

fn open(kv: KvFormat, exact: bool) -> Option<QwenEngine> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", d)).exists() { eprintln!("SKIP: {} missing", d); return None; }
    let opt = QwenOptions { max_ctx: MAX_CTX, kv, prefill_exact: exact, ..Default::default() };
    Some(QwenEngine::open(&d, opt).expect("open tiny"))
}

fn prompts() -> Vec<(String, Vec<u32>)> {
    let s = std::fs::read_to_string(format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT)).expect("tiny prompts");
    let j: Value = serde_json::from_str(&s).unwrap();
    let mut v: Vec<(String, Vec<u32>)> = j["prompts"].as_array().unwrap().iter().map(|r| (r["id"].as_str().unwrap().to_string(),
        r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect())).collect();
    let mut s = 0x1234_5678u64;
    v.push(("syn300".into(), (0..300).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 2000) as u32 + 10 }).collect()));
    v
}

/// attn_prep_q8's group quantization on the CPU: (codes, fp16 scale, fp16 min).
fn quant(x: &[f32]) -> ([u8; 32], u16, u16) {
    let (mut lo, mut hi) = (x[0], x[0]);
    for &v in &x[1..] { lo = lo.min(v); hi = hi.max(v); }
    let sc16 = f32_to_f16((hi - lo).max(1e-8) / 255.0);
    let lo16 = f32_to_f16(lo);
    let scf = if f16_to_f32(sc16) > 0.0 { f16_to_f32(sc16) } else { 1.0 };
    let lof = f16_to_f32(lo16);
    let mut q = [0u8; 32];
    for (i, &v) in x.iter().enumerate() { q[i] = ((v - lof) / scf).round_ties_even().clamp(0.0, 255.0) as u8; }
    (q, sc16, lo16)
}

/// KV rows 0..n of every attention layer: (layer, bit-identical groups, groups, max |deq - bf16| in q8 steps).
fn kv_cmp(q: &QwenEngine, b: &QwenEngine, n: usize) -> Vec<(usize, usize, usize, f32)> {
    let (hd, nkv) = (q.cfg.head_dim, q.cfg.kv_heads);
    let (qb, bb) = (q.arena.base() as usize, b.arena.base() as usize);
    let mut out = vec![];
    for l in 0..q.cfg.layers {
        let (Some(rq), Some(rb)) = (q.sl().kv[l].as_ref(), b.sl().kv[l].as_ref()) else { continue };
        let (mut same, mut tot, mut worst) = (0, 0, 0f32);
        for (vq, sq, vb) in [(rq.k, rq.ksb, rb.k), (rq.v, rq.vsb, rb.v)] {
            for h in 0..nkv {
                for p in 0..n {
                    for g in 0..hd / 32 {
                        let e0 = (h * MAX_CTX + p) * hd + g * 32;
                        let gs = (h * MAX_CTX + p) * (hd / 32) + g;
                        let x: Vec<f32> = (0..32).map(|i| unsafe {
                            streamdense_core::bf16_to_f32(std::ptr::read((bb + vb + 2 * (e0 + i)) as *const u16)) }).collect();
                        let gq: Vec<u8> = (0..32).map(|i| unsafe { std::ptr::read((qb + vq + e0 + i) as *const u8) }).collect();
                        let (sc, mn) = unsafe { (std::ptr::read((qb + sq + 4 * gs) as *const u16), std::ptr::read((qb + sq + 4 * gs + 2) as *const u16)) };
                        let (cq, csc, cmn) = quant(&x);
                        tot += 1;
                        if cq[..] == gq[..] && csc == sc && cmn == mn { same += 1; }
                        let (scf, mnf) = (f16_to_f32(sc), f16_to_f32(mn));
                        let step = if scf > 0.0 { scf } else { 1.0 };
                        for i in 0..32 { worst = worst.max((scf * gq[i] as f32 + mnf - x[i]).abs() / step); }
                    }
                }
            }
        }
        out.push((l, same, tot, worst));
    }
    out
}

#[test]
fn q8_fast_prefill_vs_bf16_fast() {
    let (Some(q), Some(b), Some(qx)) = (open(KvFormat::Q8, false), open(KvFormat::Bf16, false), open(KvFormat::Q8, true)) else { return };
    assert!(q.fast_prefill() && b.fast_prefill() && !qx.fast_prefill());
    let first_attn = q.cfg.kinds.iter().position(|k| *k == LayerKind::Attention).unwrap();
    const N: usize = 48;
    let (mut first_ok, mut runs) = (0, 0);
    for (id, ids) in prompts() {
        // KV after the prompt only
        q.reset(); b.reset();
        let (tq, tb) = (q.prefill(&ids).unwrap(), b.prefill(&ids).unwrap());
        for (l, same, tot, worst) in kv_cmp(&q, &b, ids.len()) {
            println!("  {:8} ({:4} tok) layer {}: q8 groups == CPU quant of bf16 KV {} / {}; max |deq - bf16| {:.2} q8 steps",
                     id, ids.len(), l, same, tot, worst);
            if l == first_attn { assert_eq!(same, tot, "{} layer {}: q8 fast prefill KV != q8 quantization of the same values", id, l); }
            // 0.5 step = rounding; later layers add the earlier layers' (q8 vs bf16 attention) differences
            assert!(worst <= if l == first_attn { 0.5001 } else { 8.0 }, "{} layer {}: dequant error {} steps", id, l, worst);
        }
        // greedy tokens
        let gq = q.generate(&ids, N, false).unwrap();
        let gb = b.generate(&ids, N, false).unwrap();
        let gx = qx.generate(&ids, N, false).unwrap();
        let diff = |a: &[u32], c: &[u32]| a.iter().zip(c).position(|(x, y)| x != y);
        let d = diff(&gq.tokens, &gb.tokens);
        println!("  {:8}: first token q8 fast {} bf16 fast {} | first diff vs bf16 fast {:?}{} | vs q8 exact {:?} | q8 exact vs bf16 fast {:?}",
                 id, tq, tb, d, d.map(|i| format!(" (bf16 margin {:.3}, q8 margin {:.3})", gb.margins[i], gq.margins[i])).unwrap_or_default(),
                 diff(&gq.tokens, &gx.tokens), diff(&gx.tokens, &gb.tokens));
        let tps = |s: f64| ids.len() as f64 / s;
        println!("  {:8}: prefill tok/s q8 fast {:.0}, bf16 fast {:.0}, q8 exact {:.0}", id, tps(gq.prefill_s), tps(gb.prefill_s), tps(gx.prefill_s));
        runs += 1;
        first_ok += (tq == tb) as usize;
        assert!(tq == tb || gb.margins[0] < 0.5, "{}: first token differs at bf16 margin {}", id, gb.margins[0]);
    }
    println!("first token q8 fast == bf16 fast: {} / {}", first_ok, runs);
}
