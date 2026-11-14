//! DFlash2 block drafter (`QwenOptions.dflash2`, dflash2.rs) on the tiny model (models/tiny_qwen35) with random drafters,
//! Metal. Never loads 27B weights (`real_file_plan` reads headers only).
//! - lossless: `feed_stream_dflash2` == `feed_stream`, token for token, and the DeltaNet state + conv state + host
//!   position equal after every feed; random drafter (mostly rejected) and an oracle hook with 0..=7 accepted drafts;
//!   exact and fast prefill, fused verify, continuations, token limits, stop tokens, a real EOS, cancel, ctx_full;
//! - ring: a small `window`, so a run passes the ring size: ctx_lo moves, stale verify rows never reach the drafter
//!   (logits after a run == logits after a fresh prefill of the same tokens, bit for bit), reset to position 0;
//! - stats: tokens == passes + accepted on every run;
//! - `ref_forward` (only with SD_DFLASH2_REF_DIR): dump for the z-lab check
//!   (an external reference script, not in the repo: make -> this test -> check).
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use streamdense_core::{DType, SafeIndex};
use streamdense_model::dflash::DraftHook;
use streamdense_model::dflash2::{self, Df2Plan, Df2Stats};
use streamdense_model::dflash2_select::Selector;
use streamdense_model::qwen_config::QwenConfig;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];
static SEQ: AtomicUsize = AtomicUsize::new(0);

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}
fn opt(exact: bool, df: Option<&Path>) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: exact, activity: false, dflash2: df.map(|p| p.to_path_buf()), ..Default::default() }
}
fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd_dflash2_{}_{}_{}", tag, std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    d
}
/// A random drafter for the tiny target in a fresh temp dir.
fn random_drafter(d: &str, tag: &str, block: usize, window: usize) -> PathBuf {
    let cfg = QwenConfig::from_model_dir(d).unwrap();
    let p = tmpdir(tag);
    dflash2::write_random(&p, &cfg, 2, block, window, 64, 7).unwrap();
    p
}

fn plain(e: &QwenEngine, ids: &[u32], n: usize, stop: &[u32]) -> (Vec<u32>, &'static str) {
    let mut t = vec![];
    let s = e.feed_stream(ids, n, stop, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(s.gen, t.len());
    (t, s.stop_reason)
}
fn dfl(e: &QwenEngine, ids: &[u32], n: usize, stop: &[u32], hook: Option<DraftHook>) -> (Vec<u32>, &'static str, Df2Stats) {
    let mut t = vec![];
    let (s, c) = e.feed_stream_dflash2_ov(ids, n, stop, &mut |x| { t.push(x); true }, hook).unwrap();
    assert_eq!(s.gen, t.len());
    assert_stats(&c, if ids.len() > 1 && !t.is_empty() { t.len() - 1 } else { t.len() });
    (t, s.stop_reason, c)
}
/// `rounds`: tokens the rounds emitted (the prefill's first token is not one).
fn assert_stats(c: &Df2Stats, rounds: usize) {
    assert_eq!(c.tokens, rounds, "tokens counted vs emitted: {:?}", c);
    assert_eq!(c.tokens, c.passes + c.accepted, "tokens == passes + accepted: {:?}", c);
    assert_eq!(c.accept_hist.iter().sum::<usize>(), c.passes, "{:?}", c);
    assert!(c.accepted <= c.drafted, "{:?}", c);
}
fn same_state(a: &QwenEngine, b: &QwenEngine, what: &str) {
    assert_eq!(a.pos(), b.pos(), "{}: host position", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_s()) == b.arena.slice_mut::<u8>(b.dn_s()), "{}: DeltaNet state differs", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_conv()) == b.arena.slice_mut::<u8>(b.dn_conv()), "{}: conv state differs", what);
    assert!(b.dflash2_ctx_len() >= b.pos(), "{}: drafter context {} < position {}", what, b.dflash2_ctx_len(), b.pos());
}

/// Oracle drafts from the true sequence `seq` (prompt + plain output): the anchor position picks how many drafts are right
/// (`accept(a)`), the first one after them is wrong.
fn oracle(seq: Vec<u32>, accept: fn(usize) -> usize) -> impl Fn(usize, &[u32]) -> Vec<u32> {
    move |a: usize, dr: &[u32]| {
        let right = accept(a);
        (0..dr.len()).map(|i| {
            let t = seq.get(a + 1 + i).copied().unwrap_or(0);
            if i < right { t } else { (t + 1) % 2048 }
        }).collect()
    }
}
fn cycle8(a: usize) -> usize { a % 8 }
fn all7(_: usize) -> usize { 7 }
fn none(_: usize) -> usize { 0 }
fn mostly3(a: usize) -> usize { if a % 5 == 0 { 7 } else { 3 } }

/// First feed + 2 continuations, plain vs DFlash2; returns summed stats.
fn scenario(a: &QwenEngine, b: &QwenEngine, prompt: &[u32], n: usize, accept: Option<fn(usize) -> usize>, what: &str) -> Df2Stats {
    let mut tot = Df2Stats::default();
    a.reset();
    b.reset();
    let (pa, ra) = plain(a, prompt, n, &[]);
    let mut seq: Vec<u32> = prompt.to_vec();
    seq.extend_from_slice(&pa);
    let hk = accept.map(|x| oracle(seq.clone(), x));
    let (pb, rb, s) = dfl(b, prompt, n, &[], hk.as_ref().map(|f| f as DraftHook));
    assert_eq!(pa, pb, "{}: first feed tokens", what);
    assert_eq!(ra, rb, "{}: first feed stop", what);
    same_state(a, b, &format!("{} first feed", what));
    tot.add(&s);
    // continuation 1: the pending token alone
    let (ca, _) = plain(a, &[*pa.last().unwrap()], n, &[]);
    seq.extend_from_slice(&ca);
    let hk = accept.map(|x| oracle(seq.clone(), x));
    let (cb, _, s) = dfl(b, &[*pb.last().unwrap()], n, &[], hk.as_ref().map(|f| f as DraftHook));
    assert_eq!(ca, cb, "{}: continuation (1 token)", what);
    same_state(a, b, &format!("{} continuation 1", what));
    tot.add(&s);
    // continuation 2: pending token + the prompt again
    let mut f: Vec<u32> = vec![*ca.last().unwrap()];
    f.extend_from_slice(prompt);
    let (da, _) = plain(a, &f, n, &[]);
    seq.extend_from_slice(prompt);
    seq.extend_from_slice(&da);
    let hk = accept.map(|x| oracle(seq.clone(), x));
    let (db, _, s) = dfl(b, &f, n, &[], hk.as_ref().map(|f| f as DraftHook));
    assert_eq!(da, db, "{}: continuation (prompt again)", what);
    same_state(a, b, &format!("{} continuation 2", what));
    tot.add(&s);
    tot
}

fn report(what: &str, s: &Df2Stats) {
    eprintln!("{}: {} passes, drafted {} accepted {} ({:.2}), {:.2} tokens/pass, rejects {}, hist {:?}, short_ctx {}, io errors {}",
              what, s.passes, s.drafted, s.accepted, s.acceptance(), s.tokens_per_pass(), s.rejects, s.accept_hist, s.short_ctx, s.select_io_errors);
}

#[test]
fn lossless_random_and_oracle() {
    let Some(d) = dir() else { return };
    for exact in [true, false] {
        let p = random_drafter(&d, "ll", 8, 2048);
        let a = QwenEngine::open(&d, opt(exact, None)).unwrap();
        let b = QwenEngine::open(&d, opt(exact, Some(&p))).unwrap();
        assert!(b.dflash2_on() && !a.dflash2_on());
        assert_eq!(b.dflash2_config().unwrap().block, 8);
        let rep: Vec<u32> = PROMPT.iter().chain(PROMPT.iter()).chain(PROMPT.iter()).copied().collect();
        let long: Vec<u32> = (0..150u32).map(|i| (i * 37 + 11) % 2000).collect(); // > 8 rows: multi-chunk absorb
        for (pr, n) in [(&PROMPT[..], 40), (&rep[..], 48), (&long[..], 30)] {
            let tag = format!("exact {} prompt {}", exact, pr.len());
            let s = scenario(&a, &b, pr, n, None, &format!("{} random", tag));
            report(&format!("{} random", tag), &s);
            assert!(s.passes > 0 && s.drafted > 0);
            assert_eq!(s.short_ctx, 0, "{} random: every position has context rows", tag);
            assert_eq!(s.select_io_errors, 0);
            for (nm, acc) in [("cycle8", cycle8 as fn(usize) -> usize), ("all7", all7), ("none", none), ("mostly3", mostly3)] {
                let s = scenario(&a, &b, pr, n, Some(acc), &format!("{} oracle {}", tag, nm));
                report(&format!("{} oracle {}", tag, nm), &s);
                if nm == "all7" { assert!(s.accepted * 10 > s.drafted * 9, "{}: oracle drafts almost all accepted", tag); }
                if nm == "none" { assert_eq!(s.accepted, 0, "{}", tag); }
            }
        }
        let _ = std::fs::remove_dir_all(&p);
    }
}

/// Every count of accepted drafts 0..=7 is reached: round r gets r % 8 right drafts.
#[test]
fn accept_hist_covers_0_to_7() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "hist", 8, 2048);
    let a = QwenEngine::open(&d, opt(true, None)).unwrap();
    let b = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    a.reset();
    let n = 120;
    let (x, _) = plain(&a, &PROMPT, n, &[]);
    let mut seq: Vec<u32> = PROMPT.to_vec();
    seq.extend_from_slice(&x);
    let round = std::cell::Cell::new(0usize);
    let hk = |pos: usize, dr: &[u32]| -> Vec<u32> {
        let right = round.get() % 8;
        round.set(round.get() + 1);
        (0..dr.len()).map(|i| {
            let t = seq.get(pos + 1 + i).copied().unwrap_or(0);
            if i < right { t } else { (t + 1) % 2048 }
        }).collect()
    };
    b.reset();
    let (y, _, s) = dfl(&b, &PROMPT, n, &[], Some(&hk as DraftHook));
    report("hist", &s);
    assert_eq!(x, y);
    same_state(&a, &b, "hist");
    assert!(s.accept_hist.iter().all(|&c| c > 0), "{:?}", s.accept_hist);
    let _ = std::fs::remove_dir_all(&p);
}

#[test]
fn limits_stop_eos_cancel_ctx_full() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "stop", 8, 2048);
    for fused in [false, true] {
        let (mut oa, mut ob) = (opt(true, None), opt(true, Some(&p)));
        oa.verify_fused = fused;
        ob.verify_fused = fused;
        let a = QwenEngine::open(&d, oa).unwrap();
        let b = QwenEngine::open(&d, ob).unwrap();
        a.reset();
        let (full, _) = plain(&a, &PROMPT, 40, &[]);
        let mut seq: Vec<u32> = PROMPT.to_vec();
        seq.extend_from_slice(&full);
        for (nm, acc) in [("all7", all7 as fn(usize) -> usize), ("cycle8", cycle8), ("none", none)] {
            let hk = oracle(seq.clone(), acc);
            for n in [1usize, 2, 3, 8, 9, 16] {
                a.reset();
                b.reset();
                let (x, rx) = plain(&a, &PROMPT, n, &[]);
                let (y, ry, _) = dfl(&b, &PROMPT, n, &[], Some(&hk as DraftHook));
                assert_eq!((x, rx), (y, ry), "{} n_new {}", nm, n);
                same_state(&a, &b, &format!("{} n_new {}", nm, n));
            }
            // a stop token inside an accepted run
            for at in [1usize, 4, 11, 20] {
                let stop = full[at];
                a.reset();
                b.reset();
                let (x, rx) = plain(&a, &PROMPT, 40, &[stop]);
                let (y, ry, _) = dfl(&b, &PROMPT, 40, &[stop], Some(&hk as DraftHook));
                assert_eq!((x, rx), (y, ry), "{} stop at {}", nm, at);
                assert_eq!(rx, "eos");
                same_state(&a, &b, &format!("{} stop at {}", nm, at));
            }
            // cancel after 5 tokens (first token from the prefill, 4 from the rounds)
            b.reset();
            let mut got = vec![];
            let (s, c) = b.feed_stream_dflash2_ov(&PROMPT, 40, &[], &mut |t| { got.push(t); got.len() < 5 }, Some(&hk as DraftHook)).unwrap();
            assert_eq!(s.stop_reason, "cancelled");
            assert_eq!(got, full[..5].to_vec());
            assert_stats(&c, 4);
        }
    }
    // the context fills up: same tokens, same reason, same state
    let (mut oa, mut ob) = (opt(true, None), opt(true, Some(&p)));
    oa.max_ctx = 64;
    ob.max_ctx = 64;
    let a = QwenEngine::open(&d, oa).unwrap();
    let b = QwenEngine::open(&d, ob).unwrap();
    a.reset();
    let (x, rx) = plain(&a, &PROMPT, 200, &[]);
    let mut seq: Vec<u32> = PROMPT.to_vec();
    seq.extend_from_slice(&x);
    let hk = oracle(seq, all7);
    for hook in [None, Some(&hk as DraftHook)] {
        a.reset();
        b.reset();
        let (x, rx2) = plain(&a, &PROMPT, 200, &[]);
        let (y, ry, _) = dfl(&b, &PROMPT, 200, &[], hook);
        assert_eq!(rx, "ctx_full");
        assert_eq!((x, rx2), (y, ry), "ctx_full");
        same_state(&a, &b, "ctx_full");
    }
    let _ = std::fs::remove_dir_all(&p);
}

/// A real EOS (the model's config lists a token it emits): tiny copy with only the config changed.
#[test]
fn real_eos_inside_a_run() {
    let Some(d) = dir() else { return };
    let a0 = QwenEngine::open(&d, opt(true, None)).unwrap();
    a0.reset();
    let (full, _) = plain(&a0, &PROMPT, 40, &[]);
    drop(a0);
    let p = random_drafter(&d, "eos", 8, 2048);
    for at in [0usize, 1, 5, 11] {
        let eos = full[at];
        let m = tmpdir("eosmodel");
        std::fs::create_dir_all(&m).unwrap();
        for f in ["model-00001-of-00001.safetensors", "model.safetensors.index.json", "generation_config.json"] {
            std::os::unix::fs::symlink(std::fs::canonicalize(format!("{}/{}", d, f)).unwrap(), m.join(f)).unwrap();
        }
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/config.json", d)).unwrap()).unwrap();
        v["eos_token_id"] = serde_json::json!([eos]);
        v["text_config"]["eos_token_id"] = serde_json::json!(eos);
        std::fs::write(m.join("config.json"), v.to_string()).unwrap();
        std::fs::remove_file(m.join("generation_config.json")).unwrap();
        std::os::unix::fs::symlink(std::fs::canonicalize(format!("{}/mtp", d)).unwrap(), m.join("mtp")).unwrap();
        let ms = m.display().to_string();
        let a = QwenEngine::open(&ms, opt(true, None)).unwrap();
        let b = QwenEngine::open(&ms, opt(true, Some(&p))).unwrap();
        a.reset();
        let (x, rx) = plain(&a, &PROMPT, 40, &[]);
        assert_eq!(rx, "eos", "EOS {} at {}: plain output {:?}", eos, at, x);
        assert_eq!(*x.last().unwrap(), eos);
        let mut seq: Vec<u32> = PROMPT.to_vec();
        seq.extend_from_slice(&x);
        seq.extend_from_slice(&full[x.len().min(full.len())..]);
        let hk = oracle(seq, all7);
        for hook in [None, Some(&hk as DraftHook)] {
            a.reset();
            b.reset();
            let (x2, _) = plain(&a, &PROMPT, 40, &[]);
            let (y, ry, _) = dfl(&b, &PROMPT, 40, &[], hook);
            assert_eq!((x2, "eos"), (y, ry), "EOS at {}", at);
            same_state(&a, &b, &format!("EOS at {}", at));
        }
        let _ = std::fs::remove_dir_all(&m);
    }
    let _ = std::fs::remove_dir_all(&p);
}

/// Blocks other than 8 (fewer keys than simdgroups in the attention kernel at block 2).
#[test]
fn other_block_sizes() {
    let Some(d) = dir() else { return };
    for block in [2usize, 3, 5] {
        let p = random_drafter(&d, "blk", block, 64);
        let a = QwenEngine::open(&d, opt(true, None)).unwrap();
        let b = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
        for acc in [None, Some(all7 as fn(usize) -> usize), Some(cycle8)] {
            let s = scenario(&a, &b, &PROMPT, 30, acc, &format!("block {}", block));
            report(&format!("block {}", block), &s);
        }
        let _ = std::fs::remove_dir_all(&p);
    }
}

/// The older matmul family (`--gemv old`): the drafter's K-row matmuls run on it too.
#[test]
fn gemv_old_family() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "old", 8, 64);
    let g = streamdense_model::qwen_layers::GemvImpl::parse("old").unwrap();
    let (mut oa, mut ob) = (opt(true, None), opt(true, Some(&p)));
    oa.gemv = g;
    ob.gemv = g;
    let a = QwenEngine::open(&d, oa).unwrap();
    let b = QwenEngine::open(&d, ob).unwrap();
    for acc in [None, Some(all7 as fn(usize) -> usize), Some(cycle8)] {
        let s = scenario(&a, &b, &PROMPT, 40, acc, "gemv old");
        report("gemv old", &s);
    }
    let _ = std::fs::remove_dir_all(&p);
}

fn fed_after(prompt: &[u32], out: &[u32]) -> Vec<u32> { prompt.iter().chain(out[..out.len() - 1].iter()).copied().collect() }

/// Small window: the run passes the ring. Stale verify rows, wraps and resets must never change what the drafter reads:
/// after a run its next forward is bit-identical to the one after a fresh exact prefill of the same tokens.
#[test]
fn ring_wrap_stale_rows_and_reset() {
    let Some(d) = dir() else { return };
    let window = 16;
    let block = 8;
    let p = random_drafter(&d, "ring", block, window);
    let a = QwenEngine::open(&d, opt(true, None)).unwrap();
    let b = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    let c = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    let ring = window - 1 + block - 1;
    assert_eq!(b.dflash2_ring_rows(), ring);
    a.reset();
    let n = 120;
    assert!(PROMPT.len() + n > 3 * ring, "the run passes the ring several times");
    let (full, _) = plain(&a, &PROMPT, n, &[]);
    let mut seq: Vec<u32> = PROMPT.to_vec();
    seq.extend_from_slice(&full);
    let fed = fed_after(&PROMPT, &full);
    let pend = *full.last().unwrap();
    // the reference: fresh prefill
    c.reset();
    c.prefill(&fed).unwrap();
    let (dc, _, _) = c.dflash2_draft(c.pos(), pend).unwrap();
    let (lc, hc) = (c.dflash2_logits(), c.dflash2_hidden());
    assert!(lc.iter().all(|x| x.is_finite()) && hc.iter().all(|x| x.is_finite()));
    assert_eq!(c.dflash2_ctx_lo(), c.pos() - ring);
    let mut first: Option<Vec<f32>> = None;
    for (nm, acc) in [("none", Some(none as fn(usize) -> usize)), ("cycle8", Some(cycle8)), ("mostly3", Some(mostly3)), ("all7", Some(all7)), ("random", None)] {
        // same run, twice: the second starts after a reset to position 0 with a wrapped ring
        for run in 0..2 {
            a.reset();
            b.reset();
            let (x, _) = plain(&a, &PROMPT, n, &[]);
            let hk = acc.map(|f| oracle(seq.clone(), f));
            let (y, _, s) = dfl(&b, &PROMPT, n, &[], hk.as_ref().map(|f| f as DraftHook));
            assert_eq!(x, y, "{} run {}", nm, run);
            same_state(&a, &b, &format!("ring {} run {}", nm, run));
            assert_eq!(s.short_ctx, 0);
            report(&format!("ring {} run {}", nm, run), &s);
            assert_eq!(b.pos(), fed.len());
            assert!(b.dflash2_ctx_lo() > 0 && b.dflash2_ctx_lo() + ring >= b.dflash2_ctx_len(), "lo {} len {}", b.dflash2_ctx_lo(), b.dflash2_ctx_len());
            let (db, _, _) = b.dflash2_draft(b.pos(), pend).unwrap();
            let (lb, hb) = (b.dflash2_logits(), b.dflash2_hidden());
            assert_eq!(db, dc, "{} run {}: drafts after a run vs fresh prefill", nm, run);
            assert!(hb == hc, "{} run {}: hidden differs", nm, run);
            assert!(lb == lc, "{} run {}: drafter logits differ, max diff {}", nm, run, lb.iter().zip(&lc).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
            first.get_or_insert(lb);
        }
    }
    // reset to position 0 after a wrapped ring: the anchor sees no context; == a fresh engine, then ctx_lo restarts
    b.reset();
    let (d0, _, _) = b.dflash2_draft(0, PROMPT[0]).unwrap();
    let l0 = b.dflash2_logits();
    let f = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    f.reset();
    let (d1, _, _) = f.dflash2_draft(0, PROMPT[0]).unwrap();
    assert_eq!(d0, d1, "anchor 0 after a reset");
    assert!(l0 == f.dflash2_logits(), "anchor 0 after a reset: logits differ");
    b.prefill(&PROMPT).unwrap();
    assert_eq!(b.dflash2_ctx_lo(), 0);
    assert_eq!(b.dflash2_ctx_len(), PROMPT.len());
    let _ = std::fs::remove_dir_all(&p);
}

/// Capture by K-row verify passes == fresh prefill, no wrap (big window).
#[test]
fn capture_matches_prefill() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "cap", 8, 2048);
    let a = QwenEngine::open(&d, opt(true, None)).unwrap();
    let b = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    let c = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    a.reset();
    let (full, _) = plain(&a, &PROMPT, 30, &[]);
    let mut seq: Vec<u32> = PROMPT.to_vec();
    seq.extend_from_slice(&full);
    for acc in [cycle8 as fn(usize) -> usize, all7, none] {
        b.reset();
        let hk = oracle(seq.clone(), acc);
        let (y, _, _) = dfl(&b, &PROMPT, 30, &[], Some(&hk as DraftHook));
        assert_eq!(y, full);
        let fed = fed_after(&PROMPT, &full);
        assert_eq!(b.pos(), fed.len());
        let pend = *full.last().unwrap();
        let (db, _, _) = b.dflash2_draft(b.pos(), pend).unwrap();
        let lb = b.dflash2_logits();
        c.reset();
        c.prefill(&fed).unwrap();
        let (dc, _, _) = c.dflash2_draft(c.pos(), pend).unwrap();
        assert_eq!(db, dc);
        assert!(lb == c.dflash2_logits(), "drafter logits differ");
    }
    let _ = std::fs::remove_dir_all(&p);
}

#[test]
fn budget_rows_and_off_by_default() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "bud", 8, 2048);
    let rep = QwenEngine::budget(&d, &opt(true, Some(&p))).unwrap().report();
    for row in ["DFlash2 drafter weights", "DFlash2 candidate selector", "DFlash2 context KV", "DFlash2 scratch", "DFlash2 drafts: DeltaNet hold save"] {
        assert!(rep.contains(row), "{}: {}", row, rep);
    }
    let rep0 = QwenEngine::budget(&d, &opt(true, None)).unwrap().report();
    assert!(!rep0.contains("DFlash2"));
    let a = QwenEngine::open(&d, opt(true, None)).unwrap();
    assert!(!a.dflash2_on());
    assert_eq!(a.dflash2_ctx_len(), 0);
    assert!(a.feed_stream_dflash2(&PROMPT, 4, &[], &mut |_| true).is_err(), "no drafter: refused");
    // not with MTP
    let mut o = opt(true, Some(&p));
    o.mtp = true;
    assert!(QwenEngine::budget(&d, &o).err().unwrap().contains("not with"));
    let cfg = QwenConfig::from_model_dir(&d).unwrap();
    // file checks of the plan
    let bad = |what: &str, want: &str, f: &dyn Fn(&Path)| {
        let q = tmpdir("planbad");
        dflash2::write_random(&q, &cfg, 2, 8, 2048, 64, 7).unwrap();
        f(&q);
        let e = Df2Plan::read(&q, &cfg).err().unwrap_or_else(|| panic!("{}: not refused", what));
        assert!(e.contains(want), "{}: {}", what, e);
        let _ = std::fs::remove_dir_all(&q);
    };
    bad("missing file", "No such file", &|q| std::fs::remove_file(q.join(dflash2::WEIGHTS_FILE)).unwrap());
    bad("missing config", "No such file", &|q| std::fs::remove_file(q.join(dflash2::CONFIG_FILE)).unwrap());
    let cfgedit = |q: &Path, f: &dyn Fn(&mut serde_json::Value)| {
        let pth = q.join(dflash2::CONFIG_FILE);
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&pth).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&pth, v.to_string()).unwrap();
    };
    bad("wrong architecture", "architectures", &|q| cfgedit(q, &|v| v["architectures"] = serde_json::json!(["DFlashDraftModel"])));
    bad("causal", "is_causal", &|q| cfgedit(q, &|v| v["is_causal"] = serde_json::json!(true)));
    bad("8 bits", "quantization", &|q| cfgedit(q, &|v| v["quantization"]["bits"] = serde_json::json!(8)));
    bad("hidden", "hidden", &|q| cfgedit(q, &|v| v["hidden_size"] = serde_json::json!(1024)));
    bad("kernel 3", "conv_kernel_size", &|q| cfgedit(q, &|v| v["dflash_config"]["conv_kernel_size"] = serde_json::json!(3)));
    bad("wrong target layers", "num_target_layers", &|q| cfgedit(q, &|v| v["num_target_layers"] = serde_json::json!(64)));
    // config says rank 128, tensors have 64: wrong shape
    bad("wrong shape", "candidate_selector", &|q| cfgedit(q, &|v| v["dflash_config"]["selector_rank"] = serde_json::json!(128)));
    bad("wrong layers", "no tensor", &|q| cfgedit(q, &|v| { v["num_hidden_layers"] = serde_json::json!(3); v["layer_types"] = serde_json::json!(["sliding_attention", "sliding_attention", "sliding_attention"]); }));
    let _ = std::fs::remove_dir_all(&p);
}

/// The real file, headers only: no weight enters a model.
#[test]
fn real_file_plan() {
    let (t, f) = (format!("{}/models/qwen38_27b_mlx4", ROOT), PathBuf::from(format!("{}/models/dflash2_27b_q4", ROOT)));
    if !f.join(dflash2::WEIGHTS_FILE).exists() || !Path::new(&t).join("config.json").exists() { eprintln!("SKIP: real file missing"); return; }
    let cfg = QwenConfig::from_model_dir(&t).unwrap();
    let p = Df2Plan::read(&f, &cfg).unwrap();
    assert_eq!((p.cfg.block, p.cfg.hidden, p.cfg.window, p.cfg.n_layers), (8, 5120, 2048, 5));
    assert_eq!(p.cfg.layer_ids, vec![6, 20, 34, 48, 62]);
    assert_eq!(p.cfg.ring_rows(8192), 2047 + 7);
    let mut sum = 0;
    for (n, b) in p.budget_rows(8, 8192) { eprintln!("{:>16} B  {:8.3} GB  {}", b, b as f64 / 1e9, n); sum += b; }
    eprintln!("sum of drafter rows {} B = {:.3} GB", sum, sum as f64 / 1e9);
    let o = QwenOptions { max_ctx: 8192, activity: false, dflash2: Some(f.clone()), ..Default::default() };
    let bud = QwenEngine::budget(&t, &o).unwrap();
    eprintln!("{}", bud.report());
    eprintln!("needed: arena {} B + reserve {} B = {:.3} GB", bud.arena_bytes(), bud.reserve, (bud.arena_bytes() + bud.reserve) as f64 / 1e9);
    let on = QwenEngine::budget(&t, &QwenOptions { max_ctx: 8192, activity: false, ..Default::default() }).unwrap();
    eprintln!("plain, same options: {:.3} GB", (on.arena_bytes() + on.reserve) as f64 / 1e9);
}

fn read_tensor(idx: &SafeIndex, name: &str) -> (Vec<u8>, Vec<usize>) {
    use std::os::unix::fs::FileExt;
    let t = idx.get(name).unwrap_or_else(|| panic!("no tensor {}", name));
    let f = std::fs::File::open(&idx.files()[t.file]).unwrap();
    let mut v = vec![0u8; t.nbytes];
    f.read_exact_at(&mut v, t.off).unwrap();
    (v, t.shape.clone())
}

/// Dump for an external reference check (skipped without SD_DFLASH2_REF_DIR): the same weights and inputs, this engine's output.
#[test]
fn ref_forward() {
    let Some(d) = dir() else { return };
    let Ok(rd) = std::env::var("SD_DFLASH2_REF_DIR") else { eprintln!("SKIP ref_forward: set SD_DFLASH2_REF_DIR"); return };
    let rd = PathBuf::from(rd);
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(rd.join("cases.json")).unwrap()).unwrap();
    for cs in cases.as_array().unwrap() {
        let name = cs["name"].as_str().unwrap();
        let dp = rd.join(cs["drafter"].as_str().unwrap());
        let e = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, dflash2: Some(dp.clone()), ..Default::default() }).unwrap();
        e.reset();
        let cfg = e.dflash2_config().unwrap().clone();
        let (fw, b) = (cfg.fw(), cfg.block);
        let (a, t, last_taps): (usize, u32, Vec<u16>);
        if let Some(pr) = cs["prompt"].as_array() {
            // through the target: this engine's own taps
            let ids: Vec<u32> = pr.iter().map(|x| x.as_u64().unwrap() as u32).collect();
            t = e.prefill(&ids).unwrap();
            a = e.pos();
            let lastk = a & a.wrapping_neg(); // exact prefill chunks are powers of two <= 8, largest first: the smallest is last
            let lastk = lastk.min(8);
            last_taps = e.dflash2_tap_rows(lastk);
        } else {
            let idx = SafeIndex::open_file(&rd.join(format!("{}_in.safetensors", name)).display().to_string()).unwrap();
            let (tb, sh) = read_tensor(&idx, "taps");
            assert_eq!(sh[1], fw);
            let v: Vec<u16> = tb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            a = sh[0];
            let mut p = 0;
            while p < a {
                let k = (a - p).min(8);
                e.dflash2_feed_taps(p, &v[p * fw..(p + k) * fw]).unwrap();
                p += k;
            }
            t = cs["token"].as_u64().unwrap() as u32;
            last_taps = vec![];
        }
        let (dr, ms, sms) = e.dflash2_draft(a, t).unwrap();
        eprintln!("ref_forward {}: anchor {} token {} ring {} lo {}, drafts {:?}, {:.3} ms GPU, {:.3} ms selector", name, a, t, e.dflash2_ring_rows(), e.dflash2_ctx_lo(), dr, ms, sms);
        let lg: Vec<u8> = e.dflash2_logits().iter().flat_map(|x| x.to_le_bytes()).collect();
        let hf: Vec<u8> = e.dflash2_hidden().iter().flat_map(|x| x.to_le_bytes()).collect();
        let drb: Vec<u8> = dr.iter().flat_map(|x| x.to_le_bytes()).collect();
        let an: Vec<u8> = [a as u32, t].iter().flat_map(|x| x.to_le_bytes()).collect();
        let lt: Vec<u8> = last_taps.iter().flat_map(|x| x.to_le_bytes()).collect();
        let n = b - 1;
        let mut out: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = vec![
            ("logits", DType::F32, vec![n, cfg.vocab], lg), ("hidden", DType::F32, vec![n, cfg.hidden], hf),
            ("drafts", DType::U32, vec![n], drb), ("anchor", DType::U32, vec![2], an),
        ];
        if !lt.is_empty() { let k = last_taps.len() / fw; out.push(("last_taps", DType::BF16, vec![k, fw], lt)); }
        let views: Vec<(&str, DType, &[usize], &[u8])> = out.iter().map(|(n, d, s, b)| (*n, *d, s.as_slice(), b.as_slice())).collect();
        streamdense_core::write_safetensors(&rd.join(format!("{}_out.safetensors", name)).display().to_string(), &views).unwrap();

        // the selector on z-lab's own inputs
        let sp = rd.join(format!("{}_sel.safetensors", name));
        if sp.exists() {
            let idx = SafeIndex::open_file(&sp.display().to_string()).unwrap();
            let (lb, ls) = read_tensor(&idx, "logits"); // trials x rows x vocab f32
            let (hb, _) = read_tensor(&idx, "hidden");
            let (ab, _) = read_tensor(&idx, "anchors");
            let f32s = |v: &[u8]| -> Vec<f32> { v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
            let (lv, hv) = (f32s(&lb), f32s(&hb));
            let anch: Vec<u32> = ab.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let s = Selector::load(&dp, &cfg.sel_cfg()).unwrap();
            let (nt, rows) = (ls[0], ls[1]);
            let mut paths: Vec<u8> = vec![];
            for i in 0..nt {
                let lr = &lv[i * rows * cfg.vocab..(i + 1) * rows * cfg.vocab];
                let hr = &hv[i * rows * cfg.hidden..(i + 1) * rows * cfg.hidden];
                for x in s.pick_from(anch[i], lr, hr, rows) { paths.extend_from_slice(&x.to_le_bytes()); }
            }
            assert_eq!(s.io_errors(), 0);
            streamdense_core::write_safetensors(&rd.join(format!("{}_selout.safetensors", name)).display().to_string(),
                                                &[("paths", DType::U32, &[nt, rows], &paths)]).unwrap();
        }
    }
}
