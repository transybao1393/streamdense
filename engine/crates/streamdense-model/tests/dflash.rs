//! DFlash block drafter (`QwenOptions.dflash`, dflash.rs) on the tiny model (models/tiny_qwen35), Metal.
//! - lossless: `feed_stream_dflash` == `feed_stream` of an engine without a drafter, token for token, and the DeltaNet
//!   state + conv state + host position equal after every feed; with a random drafter (mostly rejected blocks), with
//!   an oracle hook (the true continuation, some drafts corrupted: long accepted runs and partial accepts), exact and
//!   fast prefill, fused verify on, continuations (1 pending token / prompt again), n_new limits, a stop token, cancel;
//! - capture: the drafter context written by K-row verify passes == the one a fresh prefill of the same tokens
//!   writes (drafter logits at the next position bit-identical, exact prefill: K-invariant passes);
//! - `ref_dump` (only with SD_DFLASH_REF_DIR): dump for the Python reference check
//!   (an external reference script: make -> this test -> check).
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use streamdense_core::DType;
use streamdense_model::dflash::{self, DfStats};
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
    QwenOptions { max_ctx: 1024, prefill_exact: exact, activity: false, dflash: df.map(|p| p.to_path_buf()), ..Default::default() }
}
fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd_dflash_{}_{}_{}", tag, std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    d
}
/// A random drafter for the tiny target in a fresh temp dir.
fn random_drafter(d: &str, tag: &str, tap_norm: bool, block: usize) -> PathBuf {
    let probe = QwenEngine::open(d, opt(true, None)).unwrap();
    let p = tmpdir(tag);
    dflash::write_random(&p, &probe.cfg, 2, block, tap_norm, 7).unwrap();
    p
}

fn plain(e: &QwenEngine, ids: &[u32], n: usize, stop: &[u32]) -> (Vec<u32>, &'static str) {
    let mut t = vec![];
    let s = e.feed_stream(ids, n, stop, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(s.gen, t.len());
    (t, s.stop_reason)
}
fn dfl(e: &QwenEngine, ids: &[u32], n: usize, stop: &[u32], hook: Option<dflash::DraftHook>) -> (Vec<u32>, &'static str, DfStats) {
    let mut t = vec![];
    let (s, c) = e.feed_stream_dflash_ov(ids, n, stop, &mut |x| { t.push(x); true }, hook).unwrap();
    assert_eq!(s.gen, t.len());
    (t, s.stop_reason, c)
}
fn same_state(a: &QwenEngine, b: &QwenEngine, what: &str) {
    assert_eq!(a.pos(), b.pos(), "{}: host position", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_s()) == b.arena.slice_mut::<u8>(b.dn_s()), "{}: DeltaNet state differs", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_conv()) == b.arena.slice_mut::<u8>(b.dn_conv()), "{}: conv state differs", what);
    assert!(b.dflash_ctx_len() >= b.pos(), "{}: drafter context {} < position {}", what, b.dflash_ctx_len(), b.pos());
}

/// Oracle drafts: the true tokens at positions a + 1 .. from `seq` (prompt + plain output), every draft whose position
/// is a multiple of `bad` replaced by a wrong token.
fn oracle(seq: Vec<u32>, bad: usize) -> impl Fn(usize, &[u32]) -> Vec<u32> {
    move |a: usize, dr: &[u32]| {
        (0..dr.len()).map(|i| {
            let q = a + 1 + i;
            let t = seq.get(q).copied().unwrap_or(0);
            if bad > 0 && q % bad == 0 { (t + 1) % 2048 } else { t }
        }).collect()
    }
}

/// First feed + 2 continuations, plain vs dflash; returns summed stats.
fn scenario(a: &QwenEngine, b: &QwenEngine, prompt: &[u32], n: usize, bad: Option<usize>, what: &str) -> DfStats {
    let mut tot = DfStats::default();
    a.reset();
    b.reset();
    let (pa, ra) = plain(a, prompt, n, &[]);
    let mut seq: Vec<u32> = prompt.to_vec();
    seq.extend_from_slice(&pa);
    let hk = bad.map(|x| oracle(seq.clone(), x));
    let (pb, rb, s) = dfl(b, prompt, n, &[], hk.as_ref().map(|f| f as dflash::DraftHook));
    assert_eq!(pa, pb, "{}: first feed tokens", what);
    assert_eq!(ra, rb, "{}: first feed stop", what);
    same_state(a, b, &format!("{} first feed", what));
    tot.add(&s);
    // continuation 1: the pending token alone
    let (ca, _) = plain(a, &[*pa.last().unwrap()], n, &[]);
    seq.extend_from_slice(&ca);
    let hk = bad.map(|x| oracle(seq.clone(), x));
    let (cb, _, s) = dfl(b, &[*pb.last().unwrap()], n, &[], hk.as_ref().map(|f| f as dflash::DraftHook));
    assert_eq!(ca, cb, "{}: continuation (1 token)", what);
    same_state(a, b, &format!("{} continuation 1", what));
    tot.add(&s);
    // continuation 2: pending token + the prompt again
    let mut f: Vec<u32> = vec![*ca.last().unwrap()];
    f.extend_from_slice(prompt);
    let (da, _) = plain(a, &f, n, &[]);
    seq.extend_from_slice(prompt);
    seq.extend_from_slice(&da);
    let hk = bad.map(|x| oracle(seq.clone(), x));
    let (db, _, s) = dfl(b, &f, n, &[], hk.as_ref().map(|f| f as dflash::DraftHook));
    assert_eq!(da, db, "{}: continuation (prompt again)", what);
    same_state(a, b, &format!("{} continuation 2", what));
    tot.add(&s);
    tot
}

fn report(what: &str, s: &DfStats) {
    eprintln!("{}: {} passes, drafted {} accepted {} ({:.2}), {:.2} tokens/pass, rejects {}, hist {:?}, short_ctx {}",
              what, s.passes, s.drafted, s.accepted, s.acceptance(), s.tokens_per_pass(), s.rejects, s.accept_hist, s.short_ctx);
}

#[test]
fn lossless_random_and_oracle() {
    let Some(d) = dir() else { return };
    for (exact, tap_norm) in [(true, true), (false, false)] {
        let p = random_drafter(&d, "ll", tap_norm, 8);
        let a = QwenEngine::open(&d, opt(exact, None)).unwrap();
        let b = QwenEngine::open(&d, opt(exact, Some(&p))).unwrap();
        assert!(b.dflash_on() && !a.dflash_on());
        let rep: Vec<u32> = PROMPT.iter().chain(PROMPT.iter()).chain(PROMPT.iter()).copied().collect();
        let long: Vec<u32> = (0..150u32).map(|i| (i * 37 + 11) % 2000).collect(); // > 8 rows: multi-chunk absorb (fast: 1 GEMM pass)
        for (pr, n) in [(&PROMPT[..], 40), (&rep[..], 48), (&long[..], 30)] {
            let tag = format!("exact {} tap_norm {} prompt {}", exact, tap_norm, pr.len());
            let s = scenario(&a, &b, pr, n, None, &format!("{} random", tag));
            report(&format!("{} random", tag), &s);
            assert!(s.passes > 0 && s.drafted > 0);
            for bad in [0usize, 5, 3] {
                let s = scenario(&a, &b, pr, n, Some(bad), &format!("{} oracle bad {}", tag, bad));
                report(&format!("{} oracle bad {}", tag, bad), &s);
                if bad == 0 { assert!(s.accepted * 2 > s.drafted, "oracle drafts mostly accepted"); }
                assert!(s.accept_hist[1..7].iter().sum::<usize>() > 0 || bad == 0, "partial accepts exercised");
            }
        }
        let _ = std::fs::remove_dir_all(&p);
    }
}

#[test]
fn stop_rules_and_fused_verify() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "stop", true, 6);
    let mut oa = opt(true, None);
    let mut ob = opt(true, Some(&p));
    oa.verify_fused = true;
    ob.verify_fused = true;
    let a = QwenEngine::open(&d, oa).unwrap();
    let b = QwenEngine::open(&d, ob).unwrap();
    a.reset();
    let (full, _) = plain(&a, &PROMPT, 40, &[]);
    let mut seq: Vec<u32> = PROMPT.to_vec();
    seq.extend_from_slice(&full);
    let hk = oracle(seq.clone(), 0);
    // n_new limits (inside an accepted run) and a stop token
    for n in [1usize, 2, 3, 7, 9, 17] {
        a.reset();
        b.reset();
        let (x, rx) = plain(&a, &PROMPT, n, &[]);
        let (y, ry, _) = dfl(&b, &PROMPT, n, &[], Some(&hk as dflash::DraftHook));
        assert_eq!((x, rx), (y, ry), "n_new {}", n);
        same_state(&a, &b, &format!("n_new {}", n));
    }
    let stop = full[11];
    a.reset();
    b.reset();
    let (x, rx) = plain(&a, &PROMPT, 40, &[stop]);
    let (y, ry, _) = dfl(&b, &PROMPT, 40, &[stop], Some(&hk as dflash::DraftHook));
    assert_eq!((x.clone(), rx), (y, ry), "stop token");
    assert_eq!(rx, "eos");
    same_state(&a, &b, "stop");
    // cancel after 5 tokens
    b.reset();
    let mut got = vec![];
    let (s, _) = b.feed_stream_dflash_ov(&PROMPT, 40, &[], &mut |t| { got.push(t); got.len() < 5 }, Some(&hk as dflash::DraftHook)).unwrap();
    assert_eq!(s.stop_reason, "cancelled");
    assert_eq!(got, full[..5].to_vec());
    // random drafter, fused verify, longer
    let s = scenario(&a, &b, &PROMPT, 60, None, "fused random");
    report("fused random", &s);
    let s = scenario(&a, &b, &PROMPT, 60, Some(4), "fused oracle bad 4");
    report("fused oracle bad 4", &s);
    let _ = std::fs::remove_dir_all(&p);
}

#[test]
fn capture_matches_prefill() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "cap", true, 8);
    let b = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    let c = QwenEngine::open(&d, opt(true, Some(&p))).unwrap();
    let a = QwenEngine::open(&d, opt(true, None)).unwrap();
    a.reset();
    let (full, _) = plain(&a, &PROMPT, 30, &[]);
    let mut seq: Vec<u32> = PROMPT.to_vec();
    seq.extend_from_slice(&full);
    for bad in [0usize, 3] {
        b.reset();
        let hk = oracle(seq.clone(), bad);
        let (y, _, _) = dfl(&b, &PROMPT, 30, &[], if bad == 0 { None } else { Some(&hk as dflash::DraftHook) });
        assert_eq!(y, full);
        // fed tokens: prompt + every emitted token but the last (pending)
        let fed: Vec<u32> = PROMPT.iter().chain(full[..full.len() - 1].iter()).copied().collect();
        assert_eq!(b.pos(), fed.len());
        let pend = *full.last().unwrap();
        let (db, _) = b.dflash_draft(b.pos(), pend).unwrap();
        let lb = b.dflash_logits();
        c.reset();
        c.prefill(&fed).unwrap();
        assert_eq!(c.pos(), fed.len());
        let (dc, _) = c.dflash_draft(c.pos(), pend).unwrap();
        let lc = c.dflash_logits();
        assert_eq!(db, dc, "drafts after spec decode vs fresh prefill (bad {})", bad);
        assert!(lb == lc, "drafter logits differ: max diff {}", lb.iter().zip(&lc).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
        assert!(lb.iter().all(|x| x.is_finite()));
    }
    let _ = std::fs::remove_dir_all(&p);
}

#[test]
fn budget_rows_and_off_by_default() {
    let Some(d) = dir() else { return };
    let p = random_drafter(&d, "bud", false, 8);
    let rep = QwenEngine::budget(&d, &opt(true, Some(&p))).unwrap().report();
    assert!(rep.contains("DFlash drafter weights") && rep.contains("DFlash drafter context KV") && rep.contains("DFlash drafts: DeltaNet hold save"), "{}", rep);
    let rep0 = QwenEngine::budget(&d, &opt(true, None)).unwrap().report();
    assert!(!rep0.contains("DFlash"));
    // a kit training dir (no "format") is refused with a pointer to the exporter
    let bad = tmpdir("bad");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("dflash.json"), "{\"hidden\": 512}").unwrap();
    let e = QwenEngine::budget(&d, &opt(true, Some(&bad))).err().unwrap();
    assert!(e.contains("export_engine.py"), "{}", e);
    let _ = std::fs::remove_dir_all(&p);
    let _ = std::fs::remove_dir_all(&bad);
}

/// Dump for an external reference check (skipped without SD_DFLASH_REF_DIR).
#[test]
fn ref_dump() {
    let Some(d) = dir() else { return };
    let Ok(rd) = std::env::var("SD_DFLASH_REF_DIR") else { eprintln!("SKIP ref_dump: set SD_DFLASH_REF_DIR (engine_ref.py make)"); return };
    let rd = PathBuf::from(rd);
    let mut o = opt(true, Some(&rd.join("engine")));
    o.dflash_log_taps = true;
    let e = QwenEngine::open(&d, o).unwrap();
    e.reset();
    let prompt: Vec<u32> = PROMPT.iter().chain(PROMPT.iter()).copied().chain([7, 99, 1500]).collect();
    let t = e.prefill(&prompt).unwrap();
    let a = e.pos();
    let (drafts, ms) = e.dflash_draft(a, t).unwrap();
    let logits = e.dflash_logits();
    let taps = e.dflash_tap_log(a).unwrap();
    let cfg = e.dflash_config().unwrap().clone();
    eprintln!("ref_dump: anchor {} token {}, drafts {:?}, draft forward {:.3} ms GPU", a, t, drafts, ms);
    let b = cfg.block;
    let u8s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let tb: Vec<u8> = taps.iter().flat_map(|x| x.to_le_bytes()).collect();
    let lb: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let an = u8s(&[a as u32, t]);
    let db = u8s(&drafts);
    streamdense_core::write_safetensors(&rd.join("dump.safetensors").display().to_string(), &[
        ("taps", DType::BF16, &[a, cfg.fw()], &tb), ("anchor", DType::U32, &[2], &an),
        ("logits", DType::F32, &[b - 1, cfg.vocab], &lb), ("drafts", DType::U32, &[b - 1], &db),
    ]).unwrap();
}
