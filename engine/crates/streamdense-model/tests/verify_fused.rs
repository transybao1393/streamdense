//! `QwenOptions.verify_fused` (fused K-row verify matmuls, verify_fused_q4.metal) on the tiny model
//! (models/tiny_qwen35), Metal. Every output must be BIT-IDENTICAL to the engine without it:
//! - exact prefill (K = 8 chunks take the fused kernels): prefill token, DeltaNet + conv state, teacher top-k logits;
//! - full and compact (`ffn_compact`) engines, decode_fused off and on;
//! - context drafts (`feed_stream_ctx`, verify passes of 1..=8 rows) == plain `feed_stream`;
//! - MTP engine: fixed-depth deep spec d = 1..=7 and auto == plain greedy, passes equal to the engine without it.
//! (The bit identity of the kernels themselves: verify_bench::tests::verify_fused_bit_identical.)
use std::sync::Arc;
use streamdense_model::ctx_draft::CtxDrafter;
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    if std::env::var("SD_VERIFY_FUSED").is_ok() { eprintln!("SKIP: SD_VERIFY_FUSED is set (this test compares off vs on)"); return None; }
    Some(d)
}
fn opt(vf: bool, fused: bool, set: Option<&Arc<CompactSet>>, ctx: usize) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: set.cloned(), decode_fused: fused, ctx_draft: ctx, verify_fused: vf,
                  ..Default::default() }
}
fn half_set(d: &str) -> Arc<CompactSet> {
    let probe = QwenEngine::open(d, opt(false, false, None, 0)).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    let mask: Vec<u8> = (0..l * n).map(|i| { let j = i % n; ((j / 64) % 2 == 0 || j % 64 < 4 || j % 4 != 3) as u8 }).collect();
    Arc::new(CompactSet::from_mask(&mask, l, n).unwrap())
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

/// a = without the option (and without ctx drafts), b = with it (ctx drafts 7).
fn compare(d: &str, set: Option<&Arc<CompactSet>>, label: &str) {
    for fused in [false, true] {
        let what = format!("{} decode_fused {}", label, fused);
        let a = QwenEngine::open(d, opt(false, fused, set, 0)).unwrap();
        let b = QwenEngine::open(d, opt(true, fused, set, 7)).unwrap();
        // prefill (exact K = 8 chunks) + state
        let long: Vec<u32> = (0..61u32).map(|i| PROMPT[i as usize % 13] + i).collect();
        a.reset(); b.reset();
        assert_eq!(a.prefill(&long).unwrap(), b.prefill(&long).unwrap(), "{}: prefill token", what);
        same_state(&a, &b, &format!("{} prefill", what));
        // teacher top-k (exact chunks of 8 rows)
        let (xa, xb) = (a.teacher(&long, 8).unwrap(), b.teacher(&long, 8).unwrap());
        for (i, (p, q)) in xa.iter().zip(&xb).enumerate() {
            assert_eq!(p.idx, q.idx, "{}: teacher row {} ids", what, i);
            assert!(p.val.iter().zip(&q.val).all(|(x, y)| x.to_bits() == y.to_bits()), "{}: teacher row {} logits", what, i);
        }
        // context drafts with the fused verify == plain without it, + continuations
        a.reset(); b.reset();
        let rep = plain(&a, &PROMPT, 40);
        let mut heavy: Vec<u32> = PROMPT.to_vec();
        heavy.extend_from_slice(&rep);
        heavy.extend_from_slice(&PROMPT);
        for p in [PROMPT.to_vec(), heavy] {
            a.reset(); b.reset();
            let pa = plain(&a, &p, 40);
            // drafts that are mostly right: a history index (no files) holding the plain reply with a few tokens
            // changed, so the verify passes have 1..=8 rows, full accepts and rejections
            let mut hc = streamdense_core::history_index::Config::with_dir(&std::env::temp_dir().join("sd_vf_unused"));
            hc.budget_bytes = 1 << 20;
            let hx = Arc::new(streamdense_core::history_index::HistoryIndex::new(hc));
            let mut noisy: Vec<u32> = p.iter().chain(pa.iter()).copied().collect();
            for i in (p.len() + 5..noisy.len()).step_by(11) { noisy[i] = (noisy[i] + 1) % a.cfg.vocab as u32; }
            hx.ingest(&noisy);
            let mut cx = CtxDrafter::new().with_history(Some(hx));
            a.reset();
            assert_eq!(plain(&a, &p, 40), pa);
            let mut pb = vec![];
            let (_, cs) = b.feed_stream_ctx(&p, 40, &[], &mut |x| { pb.push(x); true }, &mut cx).unwrap();
            assert_eq!(pa, pb, "{}: ctx drafts tokens", what);
            same_state(&a, &b, &format!("{} ctx feed", what));
            let ca = plain(&a, &[*pa.last().unwrap()], 30);
            let mut cb = vec![];
            let (_, cs2) = b.feed_stream_ctx(&[*pb.last().unwrap()], 30, &[], &mut |x| { cb.push(x); true }, &mut cx).unwrap();
            assert_eq!(ca, cb, "{}: ctx continuation tokens", what);
            same_state(&a, &b, &format!("{} ctx continuation", what));
            eprintln!("{}: ctx passes {} + {}, k_hist {:?} / {:?}, accepted {} + {}", what, cs.passes, cs2.passes, cs.k_hist, cs2.k_hist, cs.accepted, cs2.accepted);
            assert!(cs.k_hist[2..].iter().sum::<usize>() > 0, "{}: no verify pass of >= 3 rows (the fused kernels never ran)", what);
        }
        eprintln!("{}: prefill, state, teacher, ctx drafts: bit-identical with verify_fused", what);
    }
}

#[test]
fn verify_fused_full_bit_identical() {
    let Some(d) = dir() else { return };
    compare(&d, None, "full");
}

#[test]
fn verify_fused_compact_bit_identical() {
    let Some(d) = dir() else { return };
    let set = half_set(&d);
    compare(&d, Some(&set), "compact");
}

#[test]
fn verify_fused_mtp_deep_spec() {
    use streamdense_model::spec::DraftOverride;
    use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
    let Some(d) = dir() else { return };
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { eprintln!("SKIP: tiny has no mtp/"); return; }
    for fused in [false, true] {
        let mo = |vf: bool| QwenOptions { mtp: true, spec_depth: 7, ..opt(vf, fused, None, 0) };
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
                assert_eq!(rb.tokens, pl, "spec {} (verify_fused, decode_fused {}) != plain", m.name(), fused);
                if !matches!(m, SpecMode::Auto { .. }) { assert_eq!(ra.passes, rb.passes, "spec {}: passes", m.name()); }
            }
        }
        eprintln!("MTP decode_fused {}: plain + spec d 1..7 + auto == plain with verify_fused", fused);
    }
}
