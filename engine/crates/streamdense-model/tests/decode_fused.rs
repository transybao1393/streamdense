//! `QwenOptions.decode_fused` (2026-11-03) on the tiny model (models/tiny_qwen35), Metal: every output must be
//! BIT-IDENTICAL to today's decode (decode_fused off):
//! - full mode and compact mode (`ffn_compact`): per decode step (`fsel_step`, the cpa-single loop) the token, the f32
//!   logits and the hidden row bit for bit, the DeltaNet state after the run; `generate` (decode_sync, and pipelined)
//!   tokens + margins; `feed_stream` tokens; teacher top-k (exact prefill chunks run the MmaD twins too);
//! - MTP engine: plain decode and fixed-depth speculative decode equal with and without decode_fused.
use std::sync::Arc;
use streamdense_core::arena::Region;
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}
fn opt(fused: bool, set: Option<&Arc<CompactSet>>) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: set.cloned(), decode_fused: fused, ..Default::default() }
}
fn bytes(e: &QwenEngine, off: usize, len: usize) -> Vec<u8> { e.arena.slice_mut::<u8>(Region { off, len }).to_vec() }

/// Stepwise compare of a (decode_fused off) and b (on): prefill, then `n` fsel_step decode steps.
fn steps(a: &QwenEngine, b: &QwenEngine, n: usize, what: &str) {
    let v = a.cfg.vocab * 4;
    let hb = a.cfg.hidden * 2;
    a.reset();
    b.reset();
    let (mut ta, mut tb) = (a.prefill(&PROMPT).unwrap(), b.prefill(&PROMPT).unwrap());
    assert_eq!(ta, tb, "{}: prefill token", what);
    for i in 0..n {
        let (xa, ga) = a.fsel_step(ta).unwrap();
        let (xb, gb) = b.fsel_step(tb).unwrap();
        assert_eq!((xa, ga.to_bits()), (xb, gb.to_bits()), "{}: step {} token / margin", what, i);
        assert!(bytes(a, a.s.io.logits, v) == bytes(b, b.s.io.logits, v), "{}: step {} logits differ", what, i);
        assert!(bytes(a, a.s.io.h, hb) == bytes(b, b.s.io.h, hb), "{}: step {} hidden row differs", what, i);
        (ta, tb) = (xa, xb);
    }
    assert_eq!(a.pos(), b.pos(), "{}: host position", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_s()) == b.arena.slice_mut::<u8>(b.dn_s()), "{}: DeltaNet state differs", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_conv()) == b.arena.slice_mut::<u8>(b.dn_conv()), "{}: conv state differs", what);
}

fn loops(a: &QwenEngine, b: &QwenEngine, what: &str) {
    for pipe in [false, true] {
        let (ga, gb) = (a.generate(&PROMPT, 32, pipe).unwrap(), b.generate(&PROMPT, 32, pipe).unwrap());
        assert_eq!(ga.tokens, gb.tokens, "{}: generate (pipe {}) tokens", what, pipe);
        let bits = |m: &[f32]| m.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&ga.margins), bits(&gb.margins), "{}: generate (pipe {}) margins", what, pipe);
    }
    let fs = |e: &QwenEngine| { e.reset(); let mut t = vec![]; e.feed_stream(&PROMPT, 32, &[], &mut |x| { t.push(x); true }).unwrap(); t };
    let (fa, fb) = (fs(a), fs(b));
    assert_eq!(fa, fb, "{}: feed_stream tokens", what);
    // continuation on the same sequence (a cached encode-ahead pass must not leak into the next feed)
    let cont = |e: &QwenEngine, last: u32| { let mut t = vec![]; e.feed_stream(&[last, 11, 323], 16, &[], &mut |x| { t.push(x); true }).unwrap(); t };
    assert_eq!(cont(a, *fa.last().unwrap()), cont(b, *fb.last().unwrap()), "{}: feed_stream continuation", what);
    let seq: Vec<u32> = PROMPT.iter().chain(fa.iter()).copied().collect();
    let (xa, xb) = (a.teacher(&seq, 8).unwrap(), b.teacher(&seq, 8).unwrap());
    for (i, (p, q)) in xa.iter().zip(&xb).enumerate() {
        assert_eq!(p.idx, q.idx, "{}: teacher row {} ids", what, i);
        assert!(p.val.iter().zip(&q.val).all(|(x, y)| x.to_bits() == y.to_bits()), "{}: teacher row {} logits", what, i);
    }
    eprintln!("{}: fsel steps, generate sync / pipe, feed_stream (+ continuation), teacher: bit-identical", what);
}

#[test]
fn decode_fused_full_bit_identical() {
    let Some(d) = dir() else { return };
    let a = QwenEngine::open(&d, opt(false, None)).expect("open tiny");
    let b = QwenEngine::open(&d, opt(true, None)).expect("open tiny fused");
    steps(&a, &b, 40, "full");
    loops(&a, &b, "full");
    // fast (GEMM) prefill engines: decode after a fast prefill is equal too
    let fa = QwenEngine::open(&d, QwenOptions { prefill_exact: false, ..opt(false, None) }).unwrap();
    let fb = QwenEngine::open(&d, QwenOptions { prefill_exact: false, ..opt(true, None) }).unwrap();
    steps(&fa, &fb, 24, "full, fast prefill");
}

#[test]
fn decode_fused_compact_bit_identical() {
    let Some(d) = dir() else { return };
    let probe = QwenEngine::open(&d, opt(false, None)).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let mask: Vec<u8> = (0..l * n).map(|i| { let j = i % n; ((j / 64) % 2 == 0 || j % 64 < 4 || j % 4 != 3) as u8 }).collect();
    let set = Arc::new(CompactSet::from_mask(&mask, l, n).unwrap());
    let a = QwenEngine::open(&d, opt(false, Some(&set))).expect("open tiny compact");
    let b = QwenEngine::open(&d, opt(true, Some(&set))).expect("open tiny compact fused");
    steps(&a, &b, 40, "compact");
    loops(&a, &b, "compact");
}

#[test]
fn decode_fused_mtp_bit_identical() {
    use streamdense_model::spec::DraftOverride;
    use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
    let Some(d) = dir() else { return };
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { eprintln!("SKIP: tiny has no mtp/"); return; }
    let mo = |fused: bool| QwenOptions { mtp: true, spec_depth: 4, ..opt(fused, None) };
    let a = QwenEngine::open(&d, mo(false)).unwrap();
    let b = QwenEngine::open(&d, mo(true)).unwrap();
    a.reset();
    let plain = a.generate(&PROMPT, 40, false).unwrap().tokens;
    b.reset();
    assert_eq!(b.generate(&PROMPT, 40, false).unwrap().tokens, plain, "MTP engine plain decode");
    for depth in 1..=4 {
        let ra = generate_spec_mode(&a, &PROMPT, 40, SpecMode::Fixed(depth), DraftOverride::None).unwrap();
        let rb = generate_spec_mode(&b, &PROMPT, 40, SpecMode::Fixed(depth), DraftOverride::None).unwrap();
        assert_eq!(ra.tokens, plain, "spec d {} (off) != plain", depth);
        assert_eq!(rb.tokens, plain, "spec d {} (decode_fused) != plain", depth);
        assert_eq!(ra.passes, rb.passes, "spec d {}: passes", depth);
    }
    eprintln!("MTP: plain + spec d 1..4 bit-identical with decode_fused");
}
