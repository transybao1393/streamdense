//! forheader compact FFN (ffn_compact.rs) on the tiny model (models/tiny_qwen35), Metal:
//! - keep share 1.0 (every neuron, the compact down kernel) == the normal path, and a partial set == the ffn_select hook
//!   in mode Static with the same mask from position 0. Only the fp32 summation order of the down differs, and the
//!   tiny random model has near-ties (its own Mma and Old GEMV families already part at greedy token 7, margin 0.125),
//!   so "matches" = teacher-forced over prompt + 24 greedy tokens: same top-1 on every row whose reference top-1 −
//!   top-2 margin is >= 0.25, top-k logits within 0.25; greedy tokens equal up to the first reference margin < 0.25;
//! - `ffn_compact_load` swaps sets in place: back to the first set gives exactly the fresh engine's output;
//! - the set file round-trips.
use std::sync::Arc;
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::ffn_select::FselMode;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}
fn open(d: &str, fsel: bool, set: Option<CompactSet>) -> QwenEngine {
    let opt = QwenOptions { max_ctx: 1024, prefill_exact: true, ffn_select: fsel, activity: false, ffn_compact: set.map(Arc::new), ..Default::default() };
    QwenEngine::open(d, opt).expect("open tiny")
}
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];
const TIE: f32 = 0.25;
fn gen(e: &QwenEngine) -> Vec<u32> { e.reset(); let mut t = vec![]; e.feed_stream(&PROMPT, 24, &[], &mut |x| { t.push(x); true }).unwrap(); t }
/// Reference `a` vs `b`, teacher-forced over prompt + a's greedy tokens: (rows with a clear reference top-1 (margin >=
/// TIE), of those with the same top-1 in b, max |logit diff| over the top-k ids both have).
fn cmp_teacher(a: &QwenEngine, b: &QwenEngine) -> (usize, usize, f32) {
    let mut seq = PROMPT.to_vec();
    seq.extend(gen(a));
    let (ta, tb) = (a.teacher(&seq, 8).unwrap(), b.teacher(&seq, 8).unwrap());
    let (mut clear, mut same, mut md) = (0, 0, 0f32);
    for (x, y) in ta.iter().zip(&tb) {
        if x.val[0] - x.val[1] >= TIE { clear += 1; if x.idx[0] == y.idx[0] { same += 1; } }
        for (i, u) in x.idx.iter().zip(&x.val) { if let Some(k) = y.idx.iter().position(|z| z == i) { md = md.max((u - y.val[k]).abs()); } }
    }
    (clear, same, md)
}
/// Greedy of `b` equals a's up to (excluding) a's first near-tie token; returns that compared length.
fn cmp_greedy(a: &QwenEngine, b: &QwenEngine, what: &str) -> usize {
    a.reset();
    let r = a.generate(&PROMPT, 24, false).unwrap();
    let n = r.margins.iter().position(|m| *m < TIE).unwrap_or(r.tokens.len());
    let gb = gen(b);
    assert_eq!(&gb[..n.min(gb.len())], &r.tokens[..n], "{}: greedy differs before the first near-tie (token {})", what, n);
    n
}

#[test]
fn compact_all_matches_normal_and_partial_matches_hook() {
    let Some(d) = dir() else { return };
    let base_e = open(&d, false, None);
    let (l, n) = (base_e.cfg.layers, base_e.cfg.inter);
    let base = gen(&base_e);
    // keep share 1.0
    let all = open(&d, false, Some(CompactSet::all(l, n)));
    assert_eq!(all.ffn_compact_set().unwrap().counts(), vec![n; l]);
    let (clear, same, md) = cmp_teacher(&base_e, &all);
    let ng = cmp_greedy(&base_e, &all, "compact keep 1.0 vs normal");
    eprintln!("keep 1.0: teacher top-1 {} / {} clear rows, max top-k logit diff {}; greedy equal over the first {} tokens (before a near-tie)", same, clear, md, ng);
    assert!(clear >= 20, "too few clear rows ({})", clear);
    assert_eq!(same, clear, "compact keep 1.0: teacher top-1 differs on a clear row");
    assert!(md <= TIE, "compact keep 1.0: logits differ by {}", md);
    drop(all);
    // partial set vs the hook (Static, from position 0)
    let mask: Vec<u8> = (0..l * n).map(|i| ((i % n) % 4 != 3) as u8).collect();
    let set = CompactSet::from_mask(&mask, l, n).unwrap();
    assert_eq!(set.counts(), vec![n / 4 * 3; l]);
    let hook = open(&d, true, None);
    hook.fsel_mask().unwrap().copy_from_slice(&mask);
    hook.fsel_set(FselMode::Static, 0, &vec![0; l]).unwrap();
    let cpt = open(&d, false, Some(set.clone()));
    let (clear, same, md) = cmp_teacher(&hook, &cpt);
    let ng = cmp_greedy(&hook, &cpt, "compact partial vs hook Static");
    let gc = gen(&cpt);
    eprintln!("partial 0.75: teacher top-1 {} / {} clear rows, max top-k logit diff {}; greedy equal over the first {} tokens (before a near-tie)", same, clear, md, ng);
    assert_ne!(gc, base, "the partial set should change the output");
    assert!(clear >= 20, "too few clear rows ({})", clear);
    assert_eq!(same, clear, "compact partial: teacher top-1 differs from the hook on a clear row");
    assert!(md <= TIE, "compact partial: logits differ from the hook by {}", md);
    // uneven groups: every other 64-group keeps all, the rest only 4 neurons (packed words then span 3 groups: the
    // kernel's rare per-nibble path)
    let mask3: Vec<u8> = (0..l * n).map(|i| { let j = i % n; ((j / 64) % 2 == 0 || j % 64 < 4) as u8 }).collect();
    hook.fsel_mask().unwrap().copy_from_slice(&mask3);
    hook.fsel_set(FselMode::Static, 0, &vec![0; l]).unwrap();
    let c3 = open(&d, false, Some(CompactSet::from_mask(&mask3, l, n).unwrap()));
    let (clear, same, md) = cmp_teacher(&hook, &c3);
    let ng = cmp_greedy(&hook, &c3, "compact uneven vs hook Static");
    eprintln!("uneven groups: teacher top-1 {} / {} clear rows, max top-k logit diff {}; greedy equal over the first {} tokens", same, clear, md, ng);
    assert_eq!(same, clear, "compact uneven: teacher top-1 differs from the hook on a clear row");
    assert!(md <= TIE, "compact uneven: logits differ from the hook by {}", md);
    drop(c3);
    // in-place reload: another set of the same sizes, then back
    let mask2: Vec<u8> = (0..l * n).map(|i| ((i % n) % 4 != 0) as u8).collect();
    let set2 = Arc::new(CompactSet::from_mask(&mask2, l, n).unwrap());
    cpt.ffn_compact_load(set2.clone()).unwrap();
    let g2 = gen(&cpt);
    let fresh2 = open(&d, false, Some((*set2).clone()));
    assert_eq!(g2, gen(&fresh2), "reloaded set != fresh engine with that set");
    cpt.ffn_compact_load(Arc::new(set.clone())).unwrap();
    assert_eq!(gen(&cpt), gc, "reload back to the first set changed the output");
    assert!(cpt.ffn_compact_load(Arc::new(CompactSet::all(l, n))).is_err(), "a set of another size must be refused");
    // file round trip
    let p = std::env::temp_dir().join(format!("sd_ffn_compact_{}.bin", std::process::id()));
    set.write(&p).unwrap();
    assert_eq!(CompactSet::read(&p).unwrap(), set);
    let _ = std::fs::remove_file(&p);
}

/// Compact + MTP (2026-11-02): speculative decode on a compact engine (MTP layer full, verify passes through the
/// multi-row compact down) == plain compact greedy, token for token (no tolerance: the K-row pass is per row
/// bit-identical to K = 1; the MTP engine's plain decode uses the twin cd_dt), for the real MTP head at depths 1..=7 (sync + pipelined) and oracle drafts p in {0, 0.5,
/// 1} (every rollback length); plain compact in an MTP engine == plain compact without MTP; MTP on another set.
#[test]
fn compact_mtp_equals_plain_compact() {
    use streamdense_model::spec::DraftOverride;
    use streamdense_model::spec_deep::{generate_spec_mode, generate_spec_sync, SpecMode};
    let Some(d) = dir() else { return };
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { eprintln!("SKIP: tiny has no mtp/"); return; }
    let probe = open(&d, false, None);
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let mask: Vec<u8> = (0..l * n).map(|i| ((i % n) % 4 != 3) as u8).collect();
    let mask3: Vec<u8> = (0..l * n).map(|i| { let j = i % n; ((j / 64) % 2 == 0 || j % 64 < 4) as u8 }).collect();
    let n_new = 48;
    for (name, m) in [("partial 0.75", &mask), ("uneven groups", &mask3)] {
        let set = Arc::new(CompactSet::from_mask(m, l, n).unwrap());
        let plain_e = open(&d, false, Some((*set).clone()));
        plain_e.reset();
        let plain_k = plain_e.generate(&PROMPT, n_new, false).unwrap().tokens;
        let opt = QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, mtp: true, spec_depth: 7, ffn_compact: Some(set.clone()), ..Default::default() };
        let e = QwenEngine::open(&d, opt).expect("open tiny compact + mtp");
        // the MTP engine's down family (cd_dm / cd_dt) differs from the plain engine's (cd_downk) in fp32 order only:
        // same tolerance check as above (clear rows same top-1, logits within TIE)
        let (clear, same, md) = cmp_teacher(&plain_e, &e);
        let ng = cmp_greedy(&plain_e, &e, "compact MTP-engine family vs plain-engine family");
        eprintln!("{}: MTP-engine down family vs plain-engine family: teacher top-1 {} / {} clear rows, max top-k logit diff {}; greedy equal over the first {} tokens", name, same, clear, md, ng);
        assert_eq!(same, clear, "{}: down families differ in top-1 on a clear row", name);
        assert!(md <= TIE, "{}: down families differ by {}", name, md);
        drop(plain_e);
        e.reset();
        let plain = e.generate(&PROMPT, n_new, false).unwrap().tokens;
        eprintln!("{}: plain (MTP engine) vs plain (plain engine): {} / {} tokens equal", name, plain.iter().zip(&plain_k).filter(|(a, b)| a == b).count(), plain.len());
        let seq: Vec<u32> = PROMPT.iter().chain(plain.iter()).copied().collect();
        let mut runs = 0;
        for depth in 1..=7usize {
            let r = generate_spec_mode(&e, &PROMPT, n_new, SpecMode::Fixed(depth), DraftOverride::None).unwrap();
            assert_eq!(r.tokens, plain, "{}: compact + MTP head (pipe, d = {}) != plain compact", name, depth);
            eprintln!("{}: d = {} head: {} passes, {:.2} tokens/pass", name, depth, r.passes, r.tokens_per_pass);
            if depth >= 2 {
                let s = generate_spec_sync(&e, &PROMPT, n_new, depth, DraftOverride::None).unwrap();
                assert_eq!(s.tokens, plain, "{}: compact + MTP head (sync, d = {}) != plain compact", name, depth);
            }
            for p in [0.0, 0.5, 1.0] {
                let ov = DraftOverride::Oracle { seq: seq.clone(), p, seed: 7 };
                let r = generate_spec_mode(&e, &PROMPT, n_new, SpecMode::Fixed(depth), ov).unwrap();
                assert_eq!(r.tokens, plain, "{}: compact + oracle p {} (d = {}) != plain compact", name, p, depth);
                runs += 1;
            }
        }
        let a = generate_spec_mode(&e, &PROMPT, n_new, SpecMode::Auto { max: 7 }, DraftOverride::None).unwrap();
        assert_eq!(a.tokens, plain, "{}: compact + MTP auto7 != plain compact", name);
        eprintln!("{}: {} oracle runs + head runs == plain compact ({} tokens)", name, runs, plain.len());
    }
}

/// 2026-11-02: the set size from the budget formula fits the cap and is the largest multiple of 8 that does
/// (checked against the engine's own plan); MTP's extra rows (MTP layer, MTP KV, 2nd DeltaNet state) lower K.
#[test]
fn keep_for_cap_is_largest_fitting_k() {
    let Some(d) = dir() else { return };
    let base = QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ..Default::default() };
    let cfg = streamdense_model::QwenConfig::from_model_dir(&d).unwrap();
    let (l, n) = (cfg.layers, cfg.inter);
    let total = |o: &QwenOptions, k: usize| {
        let mut o = o.clone();
        o.ffn_compact = Some(Arc::new(CompactSet { inter: n, keep: vec![(0..k as u32).collect(); l] }));
        let b = QwenEngine::budget(&d, &o).unwrap();
        b.arena_bytes() + b.reserve
    };
    let k0 = n / 16 * 8; // half the neurons
    let cap = total(&base, k0) + 1;
    let p = QwenEngine::compact_keep_for_cap(&d, &base, cap).unwrap();
    assert_eq!(p.k % 8, 0);
    assert!(p.total <= cap && total(&base, p.k) == p.total, "plan {:?}", p);
    assert!(p.k + 8 > n || total(&base, p.k + 8) > cap, "K + 8 still fits: {:?}", p);
    assert!(p.k >= k0, "{:?}", p);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() {
        let m = QwenOptions { mtp: true, spec_depth: 4, ..base.clone() };
        let capm = total(&m, k0) + 1;
        let pm = QwenEngine::compact_keep_for_cap(&d, &m, capm).unwrap();
        assert!(pm.total <= capm && pm.k >= k0 && (pm.k + 8 > n || total(&m, pm.k + 8) > capm), "{:?}", pm);
        let pp = QwenEngine::compact_keep_for_cap(&d, &base, capm).unwrap();
        assert!(pm.rest > pp.rest && pm.k < pp.k, "MTP rows must lower K: {:?} vs {:?}", pm, pp);
    }
}

/// Single-process streamed prefill (ffn_stream.rs, 2026-11-02) == the two-process path, bit for bit (no tolerance:
/// same kernels on the same rows, only the order of layers vs chunks differs):
/// - process A (compact engine + `stream_prefill`): full-FFN prompt read layer by layer, set rebuilt from the prompt
///   activity in the same pass, then compact greedy decode;
/// - process B (full engine, recorder + ffn-select hook in mode All): prefill, snapshot; its recorder sums == A's `an`, its
///   per-token gate scores summed on the CPU == A's `ag`, its first token and snapshot body == A's;
/// - process C (compact engine opened with A's final set): restore B's snapshot, greedy decode == A's tokens, and every
///   compact arena byte == A's (the in-RAM fill == the file gather).
/// For 1 and 2 slots, and a second prompt on the same engine (sets swapped from the first prompt's). Each also with
/// `ffn_stream_host` (buffers inside the idle lm_head, re-read before the head): same results, smaller budget, and the
/// lm_head bytes == a fresh engine's after every prompt and after the refused calls.
#[test]
fn stream_prefill_equals_two_process() {
    use streamdense_model::ffn_select::FselMode;
    let Some(d) = dir() else { return };
    let probe = open(&d, false, None);
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let kc = n / 16 * 8; // half the neurons
    let kres = kc / 8;
    let init = CompactSet { inter: n, keep: vec![(0..kc as u32).collect(); l] };
    // set rule (the c2 + reserve shape): top kc - kres by an, then the top kres of the rest by ag; sorted
    let rule = |an: &[f32], ag: &[f32]| -> Vec<u32> {
        let mut ix: Vec<u32> = (0..n as u32).collect();
        ix.sort_by(|&a, &b| an[b as usize].total_cmp(&an[a as usize]).then(a.cmp(&b)));
        let mut keep: Vec<u32> = ix[..kc - kres].to_vec();
        let mut rest: Vec<u32> = ix[kc - kres..].to_vec();
        rest.sort_by(|&a, &b| ag[b as usize].total_cmp(&ag[a as usize]).then(a.cmp(&b)));
        keep.extend_from_slice(&rest[..kres]);
        keep.sort();
        keep
    };
    let prompt2: Vec<u32> = PROMPT.iter().rev().copied().chain([42, 7, 1999]).collect();
    let n_new = 24;
    let full = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, ffn_select: true, activity: true, ..Default::default() }).unwrap();
    let tap = full.activity_tap().unwrap();
    let never = usize::MAX >> 1;
    for (host, slots) in [(false, 1usize), (false, 2), (true, 1), (true, 2)] {
        let opt = QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: Some(Arc::new(init.clone())),
                                ffn_stream_tokens: 64, ffn_stream_slots: slots, ffn_stream_host: host, ..Default::default() };
        let b = QwenEngine::budget(&d, &opt).unwrap();
        assert!(b.items.iter().any(|(x, _)| x.starts_with("streamed prefill")), "budget rows missing:\n{}", b.report());
        if host {
            let nb = QwenEngine::budget(&d, &QwenOptions { ffn_stream_host: false, ..opt.clone() }).unwrap();
            assert!(b.arena_bytes() < nb.arena_bytes(), "hosting must lower the budget:\n{}", b.report());
            eprintln!("host, slots {}: arena {} B vs {} B unhosted", slots, b.arena_bytes(), nb.arena_bytes());
        }
        let a = QwenEngine::open(&d, opt).unwrap();
        let lm_ok = |a: &QwenEngine, c: &QwenEngine| [(a.w.lm_head.w, c.w.lm_head.w), (a.w.lm_head.s, c.w.lm_head.s), (a.w.lm_head.b, c.w.lm_head.b)]
            .iter().all(|(x, y)| a.arena.slice_mut::<u8>(*x) == c.arena.slice_mut::<u8>(*y));
        assert!(a.stream_prefill_on());
        for (pi, prompt) in [PROMPT.to_vec(), prompt2.clone()].iter().enumerate() {
            // A
            let (mut an_a, mut ag_a) = (vec![0f32; l * n], vec![0f32; l * n]);
            let mut seen = vec![];
            a.reset();
            let st = a.stream_prefill(prompt, &mut |li, an, ag| {
                an_a[li * n..(li + 1) * n].copy_from_slice(an);
                ag_a[li * n..(li + 1) * n].copy_from_slice(ag);
                seen.push(li);
                Ok(rule(an, ag))
            }).unwrap();
            assert_eq!(seen, (0..l).collect::<Vec<_>>());
            assert_eq!(a.pos(), prompt.len());
            let snap_a = a.snapshot(prompt).unwrap();
            let mut toks_a = vec![st.tok];
            while toks_a.len() < n_new { toks_a.push(a.fsel_step(*toks_a.last().unwrap()).unwrap().0); }
            let set = a.ffn_compact_set().unwrap();
            assert_eq!(set.counts(), vec![kc; l]);
            eprintln!("slots {} prompt {}: {} tok, ssd {} B in {:.4} s, wall {:.4} s, changed {} | {:?}", slots, pi, prompt.len(), st.ssd_bytes,
                      st.ssd_busy_s, st.wall_s, st.changed, &toks_a[..8]);
            assert!(st.ssd_bytes > 0 && st.changed > 0);
            assert_eq!(st.host_reload_bytes > 0, host, "lm_head reload only when hosted");
            // B: full model, recorder sums and per-token gate scores
            // gate scores are written only for rows at / after the hook's start (cpa-prep: mode All, start 0; f untouched)
            full.fsel_set(FselMode::All, 0, &vec![n as u32; l]).unwrap();
            full.reset();
            let mut pg = vec![0f32; l * n];
            for i in 0..prompt.len() {
                full.prefill(&prompt[i..i + 1]).unwrap();
                for (x, y) in pg.iter_mut().zip(full.fsel_gate_scores().unwrap()) { *x += *y; }
            }
            full.reset();
            full.fsel_set(FselMode::All, never, &vec![n as u32; l]).unwrap();
            full.activity_reset();
            let tok_b = full.prefill(prompt).unwrap();
            let pre: Vec<f32> = tap.with_live(|v| v[..l * n].to_vec());
            let snap_b = full.snapshot(prompt).unwrap();
            assert_eq!(st.tok, tok_b, "first token");
            assert!(an_a == pre, "an != recorder sums (max diff {})", an_a.iter().zip(&pre).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
            assert!(ag_a == pg, "ag != per-token gate scores (max diff {})", ag_a.iter().zip(&pg).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
            assert!(snap_a.body == snap_b.body, "prompt state differs from the full engine's");
            // C: compact engine with A's set, B's snapshot
            let cpt = open(&d, false, Some((*set).clone()));
            cpt.reset();
            cpt.restore(&snap_b).unwrap();
            let mut toks_c = vec![tok_b];
            while toks_c.len() < n_new { toks_c.push(cpt.fsel_step(*toks_c.last().unwrap()).unwrap().0); }
            assert_eq!(toks_a, toks_c, "single process != two-process (host {}, slots {}, prompt {})", host, slots, pi);
            assert!(lm_ok(&a, &cpt), "lm_head bytes differ after the prompt (host {}, slots {})", host, slots);
            for li in 0..l {
                let (fa, fc) = (&a.w.layers[li].ffn, &cpt.w.layers[li].ffn);
                let (ca, cc) = (fa.cd.unwrap(), fc.cd.unwrap());
                for (ra, rc) in [(fa.gate.w, fc.gate.w), (fa.gate.s, fc.gate.s), (fa.gate.b, fc.gate.b), (fa.up.w, fc.up.w), (fa.up.s, fc.up.s),
                                 (fa.up.b, fc.up.b), (fa.down.w, fc.down.w), (ca.gid, cc.gid), (ca.gstart, cc.gstart), (ca.wmeta, cc.wmeta)] {
                    assert!(a.arena.slice_mut::<u8>(ra) == cpt.arena.slice_mut::<u8>(rc), "layer {} compact bytes differ", li);
                }
            }
            eprintln!("slots {} prompt {}: an / ag / first token / state / {} decode tokens / compact bytes all equal", slots, pi, n_new);
            // the same prompt in pieces (each but the last a multiple of prefill_k = 8), accumulators seeded from the
            // previous piece; rebuild = set built after every piece, defer = empty set (no fill) until the last piece
            for defer in [false, true] {
                let cuts: Vec<usize> = (8..prompt.len()).step_by(8).chain([prompt.len()]).collect();
                let mut seed: Vec<Vec<f32>> = vec![vec![0f32; 2 * n]; l];
                a.reset();
                let mut b0 = 0;
                let mut last = None;
                for (ci, &b1) in cuts.iter().enumerate() {
                    let fin = ci + 1 == cuts.len();
                    let prev = seed.clone();
                    let st = a.stream_prefill_seeded(&prompt[b0..b1], Some(&prev), &mut |li, an, ag| {
                        seed[li][..n].copy_from_slice(an);
                        seed[li][n..].copy_from_slice(ag);
                        Ok(if defer && !fin { vec![] } else { rule(an, ag) })
                    }).unwrap();
                    if defer && !fin { assert_eq!(st.changed, 0); }
                    b0 = b1;
                    last = Some(st);
                }
                assert!(cuts.len() >= 2);
                assert_eq!(a.pos(), prompt.len());
                for li in 0..l {
                    assert!(seed[li][..n] == an_a[li * n..(li + 1) * n] && seed[li][n..] == ag_a[li * n..(li + 1) * n], "pieces: layer {} sums differ", li);
                }
                // whole state equal except the `rec` rows (snapshot spans: 2 ctrl blocks of 72 B, rec 16 B per position, then
                // DeltaNet / conv / KV): the head also ran on each earlier piece's last row and wrote its rec entry there
                let sb = a.snapshot(prompt).unwrap().body;
                let r = 2 * 72 + 16 * prompt.len();
                assert!(sb[..144] == snap_a.body[..144] && sb[r..] == snap_a.body[r..], "pieces: prompt state differs");
                assert!(*a.ffn_compact_set().unwrap() == *set, "pieces: set differs");
                let mut toks_p = vec![last.unwrap().tok];
                while toks_p.len() < n_new { toks_p.push(a.fsel_step(*toks_p.last().unwrap()).unwrap().0); }
                assert_eq!(toks_p, toks_a, "pieces != one shot (slots {}, prompt {}, defer {})", slots, pi, defer);
                eprintln!("slots {} prompt {}: {} pieces (defer {}) == one shot: sums / state / set / {} tokens", slots, pi, cuts.len(), defer, n_new);
            }
        }
        // too long / wrong policy are refused
        a.reset();
        assert!(a.stream_prefill(&vec![1u32; 65], &mut |_, _, _| Ok(vec![])).is_err());
        a.reset();
        assert!(a.stream_prefill(&PROMPT, &mut |_, _, _| Ok(vec![0, 1])).is_err());
        // the wrong policy fails after layer 0 ran with the hosted slot: the guard must have restored the lm_head
        let fresh = open(&d, false, Some(init.clone()));
        assert!(lm_ok(&a, &fresh), "lm_head not restored after a refused call (host {}, slots {})", host, slots);
    }
    // off by default
    let e = open(&d, false, Some(init.clone()));
    assert!(!e.stream_prefill_on());
}

/// Fast streamed prefill (`ffn_stream_fast`, 2026-11-03): each streamed layer runs the full engine's FAST prefill
/// kernels (GEMM, flash attention) on its fast chunks, the gate score from the swiglu GEMM's own gate output.
/// Reference = the two-process path with a FAST full engine (prefill_exact off, same prefill_t): full fast prefill ->
/// snapshot -> compact engine with A's final set restores it. Same kernels on the same rows and positions, so this is
/// expected (and asserted) bit-identical: first token, snapshot body, an == the fast engine's recorder sums, then 24
/// compact decode tokens equal. The tie tolerance of this file (TIE = 0.25, see the module doc) is NOT needed for these
/// asserts; it would apply only against the EXACT path, which the fast mode is not held to (printed, not asserted:
/// first token / tokens / ag vs the exact streamed path and vs the stepwise per-token gate scores, max relative error).
/// prefill_t = 32 so prompt 1 (69 tokens: 32 + 32 flash GEMM chunks + a 2 / 2 / 1 exact tail) and prompt 2 (45: one
/// flash chunk + a 13-row GEMM chunk on the per-row attention) cover every chunk kind. Pieces cut at multiples of
/// prefill_t == one shot (sums, state, set, tokens), for rebuild and defer. 1 slot unhosted and 2 slots hosted.
#[test]
fn stream_prefill_fast_equals_full_fast() {
    use streamdense_model::ffn_select::FselMode;
    let Some(d) = dir() else { return };
    let probe = open(&d, false, None);
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let kc = n / 16 * 8;
    let kres = kc / 8;
    let init = CompactSet { inter: n, keep: vec![(0..kc as u32).collect(); l] };
    let rule = |an: &[f32], ag: &[f32]| -> Vec<u32> {
        let mut ix: Vec<u32> = (0..n as u32).collect();
        ix.sort_by(|&a, &b| an[b as usize].total_cmp(&an[a as usize]).then(a.cmp(&b)));
        let mut keep: Vec<u32> = ix[..kc - kres].to_vec();
        let mut rest: Vec<u32> = ix[kc - kres..].to_vec();
        rest.sort_by(|&a, &b| ag[b as usize].total_cmp(&ag[a as usize]).then(a.cmp(&b)));
        keep.extend_from_slice(&rest[..kres]);
        keep.sort();
        keep
    };
    let pt = 32usize;
    let p1: Vec<u32> = (0..69u32).map(|i| PROMPT[i as usize % PROMPT.len()].wrapping_add(i * 37) % 2000).collect();
    let p2: Vec<u32> = (0..45u32).map(|i| (i * 131 + 17) % 2000).collect();
    let n_new = 24;
    let relerr = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-6)).fold(0f32, f32::max);
    // full engines: fast (reference, recorder on) and exact stepwise (per-token gate scores, ffn-select hook)
    let ffast = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: false, prefill_t: pt, activity: true, ..Default::default() }).unwrap();
    assert!(ffast.fast_prefill());
    let tap = ffast.activity_tap().unwrap();
    let fexact = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, ffn_select: true, activity: false, ..Default::default() }).unwrap();
    // exact streamed engine (for the printed fast-vs-exact differences)
    let ex = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: Some(Arc::new(init.clone())),
                                                ffn_stream_tokens: 128, ..Default::default() }).unwrap();
    for (host, slots) in [(false, 1usize), (true, 2)] {
        let opt = QwenOptions { max_ctx: 1024, prefill_exact: true, prefill_t: pt, activity: false, ffn_compact: Some(Arc::new(init.clone())),
                                ffn_stream_tokens: 128, ffn_stream_slots: slots, ffn_stream_host: host, ffn_stream_fast: true, ..Default::default() };
        let bf = QwenEngine::budget(&d, &opt).unwrap();
        let be = QwenEngine::budget(&d, &QwenOptions { ffn_stream_fast: false, ..opt.clone() }).unwrap();
        eprintln!("fast host {} slots {}: arena {} B vs {} B exact streamed (bigger pass scratch)", host, slots, bf.arena_bytes(), be.arena_bytes());
        let a = QwenEngine::open(&d, opt).unwrap();
        assert!(!a.fast_prefill(), "the compact engine's own prefill stays exact");
        for (pi, prompt) in [p1.clone(), p2.clone()].iter().enumerate() {
            let (mut an_a, mut ag_a) = (vec![0f32; l * n], vec![0f32; l * n]);
            a.reset();
            let st = a.stream_prefill(prompt, &mut |li, an, ag| {
                an_a[li * n..(li + 1) * n].copy_from_slice(an);
                ag_a[li * n..(li + 1) * n].copy_from_slice(ag);
                Ok(rule(an, ag))
            }).unwrap();
            assert_eq!(a.pos(), prompt.len());
            let snap_a = a.snapshot(prompt).unwrap();
            let mut toks_a = vec![st.tok];
            while toks_a.len() < n_new { toks_a.push(a.fsel_step(*toks_a.last().unwrap()).unwrap().0); }
            let set = a.ffn_compact_set().unwrap();
            // B: full fast prefill + recorder
            ffast.reset();
            ffast.activity_reset();
            let tok_b = ffast.prefill(prompt).unwrap();
            let pre: Vec<f32> = tap.with_live(|v| v[..l * n].to_vec());
            let snap_b = ffast.snapshot(prompt).unwrap();
            let an_err = relerr(&an_a, &pre);
            // stepwise exact per-token gate scores (the exact path's ag) and the exact streamed path
            fexact.fsel_set(FselMode::All, 0, &vec![n as u32; l]).unwrap();
            fexact.reset();
            let mut pg = vec![0f32; l * n];
            for i in 0..prompt.len() {
                fexact.prefill(&prompt[i..i + 1]).unwrap();
                for (x, y) in pg.iter_mut().zip(fexact.fsel_gate_scores().unwrap()) { *x += *y; }
            }
            let (mut an_e, mut ag_e) = (vec![0f32; l * n], vec![0f32; l * n]);
            ex.reset();
            let ste = ex.stream_prefill(prompt, &mut |li, an, ag| {
                an_e[li * n..(li + 1) * n].copy_from_slice(an);
                ag_e[li * n..(li + 1) * n].copy_from_slice(ag);
                Ok(rule(an, ag))
            }).unwrap();
            let mut toks_e = vec![ste.tok];
            while toks_e.len() < n_new { toks_e.push(ex.fsel_step(*toks_e.last().unwrap()).unwrap().0); }
            let same_e = toks_a.iter().zip(&toks_e).take_while(|(x, y)| x == y).count();
            eprintln!("fast host {} slots {} prompt {} ({} tok): an vs fast recorder max rel err {:.3e}; vs exact path: an {:.3e}, ag {:.3e} \
                       (stepwise gate scores {:.3e}); first token {} vs exact {}; {} / {} leading tokens equal the exact streamed path",
                      host, slots, pi, prompt.len(), an_err, relerr(&an_a, &an_e), relerr(&ag_a, &ag_e), relerr(&ag_a, &pg), st.tok, ste.tok, same_e, n_new);
            assert_eq!(st.tok, tok_b, "first token != full fast prefill");
            assert!(snap_a.body == snap_b.body, "prompt state != full fast prefill's");
            assert!(an_a == pre, "an != fast recorder sums (max rel err {:.3e})", an_err);
            assert!(relerr(&ag_a, &pg) < 0.05, "ag far from the stepwise gate scores");
            // C: compact engine with A's set restores B's snapshot
            let cpt = open(&d, false, Some((*set).clone()));
            cpt.reset();
            cpt.restore(&snap_b).unwrap();
            let mut toks_c = vec![tok_b];
            while toks_c.len() < n_new { toks_c.push(cpt.fsel_step(*toks_c.last().unwrap()).unwrap().0); }
            assert_eq!(toks_a, toks_c, "fast single process != full fast two-process (host {}, slots {}, prompt {})", host, slots, pi);
            // pieces at multiples of prefill_t == one shot
            for defer in [false, true] {
                let cuts: Vec<usize> = (pt..prompt.len()).step_by(pt).chain([prompt.len()]).collect();
                assert!(cuts.len() >= 2);
                let mut seed: Vec<Vec<f32>> = vec![vec![0f32; 2 * n]; l];
                a.reset();
                let (mut b0, mut last) = (0, None);
                for (ci, &b1) in cuts.iter().enumerate() {
                    let fin = ci + 1 == cuts.len();
                    let prev = seed.clone();
                    let st = a.stream_prefill_seeded(&prompt[b0..b1], Some(&prev), &mut |li, an, ag| {
                        seed[li][..n].copy_from_slice(an);
                        seed[li][n..].copy_from_slice(ag);
                        Ok(if defer && !fin { vec![] } else { rule(an, ag) })
                    }).unwrap();
                    b0 = b1;
                    last = Some(st);
                }
                for li in 0..l {
                    assert!(seed[li][..n] == an_a[li * n..(li + 1) * n] && seed[li][n..] == ag_a[li * n..(li + 1) * n], "fast pieces: layer {} sums differ", li);
                }
                let sb = a.snapshot(prompt).unwrap().body;
                let r = 2 * 72 + 16 * prompt.len();
                assert!(sb[..144] == snap_a.body[..144] && sb[r..] == snap_a.body[r..], "fast pieces: prompt state differs");
                assert!(*a.ffn_compact_set().unwrap() == *set, "fast pieces: set differs");
                let mut toks_p = vec![last.unwrap().tok];
                while toks_p.len() < n_new { toks_p.push(a.fsel_step(*toks_p.last().unwrap()).unwrap().0); }
                assert_eq!(toks_p, toks_a, "fast pieces != one shot (prompt {}, defer {})", pi, defer);
            }
            eprintln!("fast host {} slots {} prompt {}: == full fast two-process (token, state, an, {} tokens); pieces == one shot", host, slots, pi, n_new);
        }
        if host {
            let fresh = open(&d, false, Some(init.clone()));
            let ok = [(a.w.lm_head.w, fresh.w.lm_head.w), (a.w.lm_head.s, fresh.w.lm_head.s), (a.w.lm_head.b, fresh.w.lm_head.b)]
                .iter().all(|(x, y)| a.arena.slice_mut::<u8>(*x) == fresh.arena.slice_mut::<u8>(*y));
            assert!(ok, "fast hosted: lm_head not restored");
        }
    }
    // option checks
    assert!(QwenEngine::budget(&d, &QwenOptions { prefill_exact: true, activity: false, ffn_compact: Some(Arc::new(init.clone())), ffn_stream_fast: true, ..Default::default() }).is_err(),
            "ffn_stream_fast without ffn_stream_tokens must be refused");
}
