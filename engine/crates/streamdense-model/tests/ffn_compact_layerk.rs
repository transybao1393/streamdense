//! the compact FFN with a DIFFERENT kept count per layer ("layer-K shift": budget moved from
//! concentrated to flat layers, same total). Tiny model (models/tiny_qwen35), Metal. No engine change was needed:
//! `CompactSet.keep` is per layer and the arena / kernels size every layer from its own `keep[l].len()`; this pins it.
//! - per-layer counts (multiples of 8, same total as a uniform 0.75 set) == the ffn_select hook in mode Static with the
//!   same mask (same tolerance as tests/ffn_compact.rs: clear rows same top-1, top-k logits within 0.25, greedy equal
//!   up to the first near-tie);
//! - the engine's budget for the shifted set is within page-alignment slack of the uniform set (the FFN row is linear in
//!   the kept count, so the same total = the same bytes);
//! - `ffn_compact_load` takes another set with the same per-layer counts, refuses the uniform one;
//! - with MTP (when the tiny model has mtp/): compact + MTP head == plain compact on the shifted set.
use std::sync::Arc;
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::ffn_select::FselMode;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];
const TIE: f32 = 0.25;

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}
fn opts(fsel: bool, set: Option<CompactSet>) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, ffn_select: fsel, activity: false, ffn_compact: set.map(Arc::new), ..Default::default() }
}
fn gen(e: &QwenEngine) -> Vec<u32> { e.reset(); let mut t = vec![]; e.feed_stream(&PROMPT, 24, &[], &mut |x| { t.push(x); true }).unwrap(); t }

/// Per-layer counts: uniform `k` ± offsets that sum to 0 (multiples of 8), so the total equals the uniform set's.
fn shifted_counts(l: usize, k: usize, n: usize) -> Vec<usize> {
    let step = ((n - k).min(k) / 3).max(8) / 8 * 8; // largest offset 3·step stays inside [8, n]
    let mut c = vec![k; l];
    for p in 0..l / 2 {
        let d = step * (1 + p % 3);
        c[2 * p] = k + d; // even layers gain, odd layers give the same amount
        c[2 * p + 1] = k - d;
    }
    c
}
/// Mask with `counts[l]` kept neurons in layer l, spread unevenly over the 64-groups (deterministic hash order).
fn mask_for(counts: &[usize], n: usize, salt: u64) -> Vec<u8> {
    let mut m = vec![0u8; counts.len() * n];
    for (l, &c) in counts.iter().enumerate() {
        let mut ix: Vec<(u64, usize)> = (0..n).map(|j| {
            let mut h = (l as u64) << 32 ^ j as u64 ^ salt;
            h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15); h ^= h >> 29; h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9); h ^= h >> 32;
            (h, j)
        }).collect();
        ix.sort_unstable();
        for &(_, j) in &ix[..c] { m[l * n + j] = 1; }
    }
    m
}

#[test]
fn per_layer_k_compact_matches_hook_and_budget() {
    let Some(d) = dir() else { return };
    let cfg = streamdense_model::QwenConfig::from_model_dir(&d).unwrap();
    let (l, n) = (cfg.layers, cfg.inter);
    let k = n * 3 / 4 / 8 * 8;
    let counts = shifted_counts(l, k, n);
    assert_eq!(counts.iter().sum::<usize>(), l * k);
    assert!(counts.iter().any(|&c| c != k) && counts.iter().all(|&c| c % 8 == 0 && c >= 8 && c <= n));
    let mask = mask_for(&counts, n, 1);
    let set = CompactSet::from_mask(&mask, l, n).unwrap();
    assert_eq!(set.counts(), counts);

    // budget: same total kept -> same FFN bytes up to page alignment (one ALIGN per region per layer at most)
    let total = |s: &CompactSet| { let b = QwenEngine::budget(&d, &opts(false, Some(s.clone()))).unwrap(); b.arena_bytes() + b.reserve };
    let uni = CompactSet { inter: n, keep: vec![(0..k as u32).collect(); l] };
    let (tu, ts) = (total(&uni), total(&set));
    let slack = (8 * l * streamdense_core::arena::ALIGN) as u64;
    eprintln!("budget: uniform {} B, per-layer {} B (diff {} B, slack allowed {} B)", tu, ts, ts as i64 - tu as i64, slack);
    assert!(ts <= tu + slack && tu <= ts + slack, "per-layer set budget {} vs uniform {}", ts, tu);

    // compact (per-layer K) vs the hook with the same mask
    let hook = QwenEngine::open(&d, opts(true, None)).unwrap();
    hook.fsel_mask().unwrap().copy_from_slice(&mask);
    hook.fsel_set(FselMode::Static, 0, &vec![0; l]).unwrap();
    let cpt = QwenEngine::open(&d, opts(false, Some(set.clone()))).unwrap();
    assert_eq!(cpt.ffn_compact_set().unwrap().counts(), counts);
    let mut seq = PROMPT.to_vec();
    seq.extend(gen(&hook));
    let (ta, tb) = (hook.teacher(&seq, 8).unwrap(), cpt.teacher(&seq, 8).unwrap());
    let (mut clear, mut same, mut md) = (0, 0, 0f32);
    for (x, y) in ta.iter().zip(&tb) {
        if x.val[0] - x.val[1] >= TIE { clear += 1; if x.idx[0] == y.idx[0] { same += 1; } }
        for (i, u) in x.idx.iter().zip(&x.val) { if let Some(q) = y.idx.iter().position(|z| z == i) { md = md.max((u - y.val[q]).abs()); } }
    }
    hook.reset();
    let r = hook.generate(&PROMPT, 24, false).unwrap();
    let ng = r.margins.iter().position(|m| *m < TIE).unwrap_or(r.tokens.len());
    let gc = gen(&cpt);
    eprintln!("per-layer K {:?}: teacher top-1 {} / {} clear rows, max top-k logit diff {}; greedy compared over {} tokens", counts, same, clear, md, ng);
    assert!(clear >= 20, "too few clear rows ({})", clear);
    assert_eq!(same, clear, "per-layer K compact: teacher top-1 differs from the hook on a clear row");
    assert!(md <= TIE, "per-layer K compact: logits differ from the hook by {}", md);
    assert_eq!(&gc[..ng.min(gc.len())], &r.tokens[..ng], "per-layer K compact: greedy differs before the first near-tie");

    // in-place reload: same per-layer counts, other neurons; back gives the first output; uniform refused
    let set2 = Arc::new(CompactSet::from_mask(&mask_for(&counts, n, 2), l, n).unwrap());
    cpt.ffn_compact_load(set2.clone()).unwrap();
    let g2 = gen(&cpt);
    let fresh2 = QwenEngine::open(&d, opts(false, Some((*set2).clone()))).unwrap();
    assert_eq!(g2, gen(&fresh2), "reloaded per-layer set != fresh engine with that set");
    drop(fresh2);
    cpt.ffn_compact_load(Arc::new(set.clone())).unwrap();
    assert_eq!(gen(&cpt), gc, "reload back to the first per-layer set changed the output");
    assert!(cpt.ffn_compact_load(Arc::new(uni)).is_err(), "a set with other per-layer counts must be refused");
    drop(cpt);
    drop(hook);

    // compact + MTP on the per-layer set == plain compact (verify passes through the multi-row compact down)
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() {
        use streamdense_model::spec::DraftOverride;
        use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
        let o = QwenOptions { mtp: true, spec_depth: 4, ..opts(false, Some(set.clone())) };
        let e = QwenEngine::open(&d, o).unwrap();
        e.reset();
        let plain = e.generate(&PROMPT, 32, false).unwrap().tokens;
        for depth in [1usize, 4] {
            let r = generate_spec_mode(&e, &PROMPT, 32, SpecMode::Fixed(depth), DraftOverride::None).unwrap();
            assert_eq!(r.tokens, plain, "per-layer K: compact + MTP (d = {}) != plain compact", depth);
        }
        eprintln!("per-layer K: compact + MTP d = 1, 4 == plain compact ({} tokens)", plain.len());
    } else { eprintln!("SKIP MTP part: tiny has no mtp/"); }
}
