//! `QwenOptions.verify_mixers` (K-row verify attention + DeltaNet recurrence, verify_mixers.metal) on the tiny model (models/tiny_qwen35),
//! Metal. Every output must be BIT-IDENTICAL to the engine without it. The tiny contexts are short, so this file
//! sets SD_SDPA_TWO_PASS_MIN = 8: every row with N >= 8 keys takes MLX's two-pass sdpa (the rows the new kernels
//! batch), shorter rows the single pass (the copied fallback), and passes straddle the switch. Both engines of a
//! comparison read the same value (one process).
//! - exact prefill (K = 8 chunks): prefill token, DeltaNet + conv state, teacher top-k logits;
//! - plain decode and context drafts (`feed_stream_ctx`, verify passes of 1..=8 rows), with and without verify_fused;
//! - MTP engine: fixed-depth deep spec d = 1..=7 and auto == plain greedy, same passes.
//! (The kernels themselves at the 27B shapes, N up to 8199: verify_bench::tests::verify_attn_bit_identical, verify_dn_recur_bit_identical.)
use std::sync::Arc;
use streamdense_model::ctx_draft::CtxDrafter;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    if std::env::var("SD_VERIFY_MIXERS").is_ok() { eprintln!("SKIP: SD_VERIFY_MIXERS is set (this test compares off vs on)"); return None; }
    // before any engine reads it (OnceLock); every test of this file sets the same value
    std::env::set_var("SD_SDPA_TWO_PASS_MIN", "8");
    Some(d)
}
fn opt(va: bool, vf: bool, ctx: usize) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ctx_draft: ctx, verify_fused: vf, verify_mixers: va, ..Default::default() }
}
fn same_state(a: &QwenEngine, b: &QwenEngine, what: &str) {
    assert_eq!(a.pos(), b.pos(), "{}: host position", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_s()) == b.arena.slice_mut::<u8>(b.dn_s()), "{}: DeltaNet state differs", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_conv()) == b.arena.slice_mut::<u8>(b.dn_conv()), "{}: conv state differs", what);
}
fn plain(e: &QwenEngine, ids: &[u32], n: usize) -> Vec<u32> {
    let mut t = vec![];
    e.feed_stream(ids, n, &[], &mut |x| { t.push(x); true }).unwrap();
    t
}

#[test]
fn verify_mixers_bit_identical() {
    let Some(d) = dir() else { return };
    for vf in [false, true] {
        let what = format!("verify_fused {}", vf);
        let a = QwenEngine::open(&d, opt(false, vf, 0)).unwrap();
        let b = QwenEngine::open(&d, opt(true, vf, 7)).unwrap();
        // prefill (exact K = 8 chunks) + state
        let long: Vec<u32> = (0..61u32).map(|i| PROMPT[i as usize % 13] + i).collect();
        a.reset(); b.reset();
        assert_eq!(a.prefill(&long).unwrap(), b.prefill(&long).unwrap(), "{}: prefill token", what);
        same_state(&a, &b, &format!("{} prefill", what));
        let (xa, xb) = (a.teacher(&long, 8).unwrap(), b.teacher(&long, 8).unwrap());
        for (i, (p, q)) in xa.iter().zip(&xb).enumerate() {
            assert_eq!(p.idx, q.idx, "{}: teacher row {} ids", what, i);
            assert!(p.val.iter().zip(&q.val).all(|(x, y)| x.to_bits() == y.to_bits()), "{}: teacher row {} logits", what, i);
        }
        // plain decode, then context drafts (repeat-heavy prompt: long accepted runs, k up to 8) == plain
        a.reset(); b.reset();
        let pa = plain(&a, &PROMPT, 40);
        assert_eq!(plain(&b, &PROMPT, 40), pa, "{}: plain decode", what);
        same_state(&a, &b, &format!("{} plain", what));
        // drafts that are mostly right (as tests/verify_fused.rs): a history index holding the plain reply with a few
        // tokens changed, so the verify passes have 1..=8 rows, full accepts and rejections
        let mut heavy: Vec<u32> = PROMPT.to_vec();
        heavy.extend_from_slice(&pa);
        heavy.extend_from_slice(&PROMPT);
        a.reset(); b.reset();
        let ha = plain(&a, &heavy, 60);
        let mut hc = streamdense_core::history_index::Config::with_dir(&std::env::temp_dir().join("sd_va_unused"));
        hc.budget_bytes = 1 << 20;
        let hx = Arc::new(streamdense_core::history_index::HistoryIndex::new(hc));
        let mut noisy: Vec<u32> = heavy.iter().chain(ha.iter()).copied().collect();
        for i in (heavy.len() + 5..noisy.len()).step_by(11) { noisy[i] = (noisy[i] + 1) % a.cfg.vocab as u32; }
        hx.ingest(&noisy);
        let mut cx = CtxDrafter::new().with_history(Some(hx));
        a.reset();
        assert_eq!(plain(&a, &heavy, 60), ha);
        let mut hb = vec![];
        let (_, cs) = b.feed_stream_ctx(&heavy, 60, &[], &mut |x| { hb.push(x); true }, &mut cx).unwrap();
        assert_eq!(ha, hb, "{}: ctx drafts tokens", what);
        same_state(&a, &b, &format!("{} ctx feed", what));
        eprintln!("{}: ctx passes {}, k_hist {:?}, accepted {}", what, cs.passes, cs.k_hist, cs.accepted);
        assert!(cs.k_hist[2..].iter().sum::<usize>() > 0, "{}: no verify pass of >= 3 rows", what);
    }
}

#[test]
fn verify_mixers_mtp_deep_spec() {
    use streamdense_model::spec::DraftOverride;
    use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
    let Some(d) = dir() else { return };
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { eprintln!("SKIP: tiny has no mtp/"); return; }
    let mo = |va: bool| QwenOptions { mtp: true, spec_depth: 7, ..opt(va, true, 0) };
    let a = QwenEngine::open(&d, mo(false)).unwrap();
    let b = QwenEngine::open(&d, mo(true)).unwrap();
    a.reset();
    let pl = a.generate(&PROMPT, 48, false).unwrap().tokens;
    b.reset();
    assert_eq!(b.generate(&PROMPT, 48, false).unwrap().tokens, pl, "MTP engine plain decode");
    let seq: Vec<u32> = PROMPT.iter().chain(pl.iter()).copied().collect();
    let mut modes: Vec<SpecMode> = (1..=7).map(SpecMode::Fixed).collect();
    modes.push(SpecMode::Auto { max: 7 });
    for m in modes {
        for ov in [DraftOverride::None, DraftOverride::Oracle { seq: seq.clone(), p: 0.8, seed: 7 }] {
            let ra = generate_spec_mode(&a, &PROMPT, 48, m, ov.clone()).unwrap();
            let rb = generate_spec_mode(&b, &PROMPT, 48, m, ov.clone()).unwrap();
            assert_eq!(ra.tokens, pl, "spec {} (off) != plain", m.name());
            assert_eq!(rb.tokens, pl, "spec {} (verify_mixers) != plain", m.name());
            if !matches!(m, SpecMode::Auto { .. }) { assert_eq!(ra.passes, rb.passes, "spec {}: passes", m.name()); }
        }
    }
    eprintln!("MTP: plain + spec d 1..7 + auto == plain with verify_mixers");
}
