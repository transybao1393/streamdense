//! Context drafting (`QwenOptions.ctx_draft`, ctx_draft.rs) on the tiny model (models/tiny_qwen35), Metal. Lossless,
//! no tolerance: `feed_stream_ctx` == `feed_stream` of an engine without ctx_draft, token for token, on
//! - full and compact (`ffn_compact`) engines, decode_fused off and on;
//! - repeat-heavy prompts (the context holds the reply-to-be: high acceptance, long accepted runs) and plain ones
//!   (mostly K = 1 passes, rejections);
//! - continuations on the same sequence (pending token + suffix), with `ids.len() == 1` (12 GB mode's first token) and
//!   longer feeds; after each feed the DeltaNet state + conv state + host position equal the plain engine's bit for bit;
//! - with the draft table (temp dir) as a second draft source;
//! - stop rules: n_new limits, a stop token, cancel.
//! Acceptance stats are printed (drafted, accepted, tokens per verify pass).
use std::sync::Arc;
use streamdense_model::ctx_draft::{CtxDrafter, CtxStats};
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}
fn opt(fused: bool, set: Option<&Arc<CompactSet>>, ctx: usize) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: set.cloned(), decode_fused: fused, ctx_draft: ctx, ..Default::default() }
}
fn half_set(d: &str) -> Arc<CompactSet> {
    let probe = QwenEngine::open(d, opt(false, None, 0)).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    let mask: Vec<u8> = (0..l * n).map(|i| { let j = i % n; ((j / 64) % 2 == 0 || j % 64 < 4 || j % 4 != 3) as u8 }).collect();
    Arc::new(CompactSet::from_mask(&mask, l, n).unwrap())
}

fn plain(e: &QwenEngine, ids: &[u32], n: usize, stop: &[u32]) -> (Vec<u32>, &'static str) {
    let mut t = vec![];
    let s = e.feed_stream(ids, n, stop, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(s.gen, t.len());
    (t, s.stop_reason)
}
fn ctx(e: &QwenEngine, cx: &mut CtxDrafter, ids: &[u32], n: usize, stop: &[u32]) -> (Vec<u32>, &'static str, CtxStats) {
    let mut t = vec![];
    let (s, c) = e.feed_stream_ctx(ids, n, stop, &mut |x| { t.push(x); true }, cx).unwrap();
    assert_eq!(s.gen, t.len());
    (t, s.stop_reason, c)
}
fn same_state(a: &QwenEngine, b: &QwenEngine, what: &str) {
    assert_eq!(a.pos(), b.pos(), "{}: host position", what);
    assert_eq!(a.rec_token(a.pos().saturating_sub(1)).0, b.rec_token(b.pos().saturating_sub(1)).0, "{}: last rec", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_s()) == b.arena.slice_mut::<u8>(b.dn_s()), "{}: DeltaNet state differs", what);
    assert!(a.arena.slice_mut::<u8>(a.dn_conv()) == b.arena.slice_mut::<u8>(b.dn_conv()), "{}: conv state differs", what);
}

/// One scenario: a first feed (prompt) + 2 continuations; returns the summed stats.
fn scenario(a: &QwenEngine, b: &QwenEngine, cx: &mut CtxDrafter, prompt: &[u32], n: usize, what: &str) -> CtxStats {
    let mut tot = CtxStats::default();
    a.reset();
    b.reset();
    cx.reset();
    let (pa, ra) = plain(a, prompt, n, &[]);
    let (pb, rb, s) = ctx(b, cx, prompt, n, &[]);
    assert_eq!(pa, pb, "{}: first feed tokens", what);
    assert_eq!(ra, rb, "{}: first feed stop", what);
    same_state(a, b, &format!("{} first feed", what));
    tot.add(&s);
    // continuation 1: the pending token alone (12 GB mode's first token path)
    let (ca, _) = plain(a, &[*pa.last().unwrap()], n, &[]);
    let (cb, _, s) = ctx(b, cx, &[*pb.last().unwrap()], n, &[]);
    assert_eq!(ca, cb, "{}: continuation (1 token) tokens", what);
    same_state(a, b, &format!("{} continuation 1", what));
    tot.add(&s);
    // continuation 2: pending token + the prompt again (repeat-heavy)
    let mut f: Vec<u32> = vec![*ca.last().unwrap()];
    f.extend_from_slice(prompt);
    let (da, _) = plain(a, &f, n, &[]);
    let (db, _, s) = ctx(b, cx, &f, n, &[]);
    assert_eq!(da, db, "{}: continuation (prompt again) tokens", what);
    same_state(a, b, &format!("{} continuation 2", what));
    tot.add(&s);
    assert_eq!(cx.history().len(), b.pos(), "{}: lookup history == tokens in the KV", what);
    tot
}

fn report(what: &str, s: &CtxStats) {
    eprintln!("{}: passes {} (drafted {}), drafts {} accepted {} ({:.2}), rejects {}, tokens {} = {:.2} per pass, lookup {}/{}, table {}/{}, k_hist {:?}, accept_hist {:?}, pre-encoded {}",
              what, s.passes, s.drafted_passes, s.drafted, s.accepted, s.acceptance(), s.rejects, s.tokens, s.tokens_per_pass(),
              s.lookup_acc, s.lookup_drafts, s.table_acc, s.table_drafts, s.k_hist, s.accept_hist, s.preencoded);
}

/// Prompts: plain, and repeat-heavy = prompt + the plain reply + prompt (the context holds the reply-to-be).
fn prompts(a: &QwenEngine) -> Vec<(String, Vec<u32>)> {
    a.reset();
    let (rep, _) = plain(a, &PROMPT, 48, &[]);
    let mut heavy: Vec<u32> = PROMPT.to_vec();
    heavy.extend_from_slice(&rep);
    heavy.extend_from_slice(&PROMPT);
    let other: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 13) % 2000).collect();
    let mut code: Vec<u32> = other.clone();
    code.extend_from_slice(&other[..30]);
    vec![("plain".into(), PROMPT.to_vec()), ("repeat-heavy".into(), heavy), ("synthetic".into(), other), ("synthetic-repeat".into(), code)]
}

fn run(d: &str, set: Option<&Arc<CompactSet>>, label: &str) {
    for fused in [false, true] {
        let a = QwenEngine::open(d, opt(fused, set, 0)).unwrap();
        let b = QwenEngine::open(d, opt(fused, set, 7)).unwrap();
        let mut cx = CtxDrafter::new();
        let mut all = CtxStats::default();
        for (name, p) in prompts(&a) {
            let s = scenario(&a, &b, &mut cx, &p, 40, &format!("{} fused {} {}", label, fused, name));
            report(&format!("{} fused {} {}", label, fused, name), &s);
            all.add(&s);
        }
        assert!(all.accepted > 0, "{}: no draft accepted at all", label);
        assert!(all.passes < all.tokens, "{}: no pass emitted more than one token", label);
        report(&format!("{} fused {} ALL", label, fused), &all);
    }
}

#[test]
fn ctx_draft_full_lossless() {
    let Some(d) = dir() else { return };
    run(&d, None, "full");
}

#[test]
fn ctx_draft_compact_lossless() {
    let Some(d) = dir() else { return };
    let set = half_set(&d);
    run(&d, Some(&set), "compact");
}

#[test]
fn ctx_draft_stops_and_depths() {
    let Some(d) = dir() else { return };
    let set = half_set(&d);
    let a = QwenEngine::open(&d, opt(true, Some(&set), 0)).unwrap();
    let pr = prompts(&a);
    let heavy = &pr[1].1;
    for depth in [1, 3, 7] {
        let b = QwenEngine::open(&d, opt(true, Some(&set), depth)).unwrap();
        let mut cx = CtxDrafter::new();
        // n_new limits: 1, 2, odd lengths cut inside an accepted run
        for n in [1, 2, 5, 13, 33] {
            a.reset(); b.reset(); cx.reset();
            let (x, rx) = plain(&a, heavy, n, &[]);
            let (y, ry, _) = ctx(&b, &mut cx, heavy, n, &[]);
            assert_eq!((x, rx), (y, ry), "depth {} n_new {}", depth, n);
            same_state(&a, &b, &format!("depth {} n_new {}", depth, n));
        }
        // a stop token in the middle of the reply
        a.reset();
        let (full, _) = plain(&a, heavy, 40, &[]);
        let stop = [full[17]];
        a.reset(); b.reset(); cx.reset();
        let (x, rx) = plain(&a, heavy, 40, &stop);
        let (y, ry, _) = ctx(&b, &mut cx, heavy, 40, &stop);
        assert_eq!((x, rx), (y, ry), "depth {} stop token", depth);
        same_state(&a, &b, &format!("depth {} stop", depth));
        // cancel after 9 tokens
        a.reset(); b.reset(); cx.reset();
        let mut t = vec![];
        let s = a.feed_stream(heavy, 40, &[], &mut |x| { t.push(x); t.len() < 9 }).unwrap();
        let mut u = vec![];
        let (s2, _) = b.feed_stream_ctx(heavy, 40, &[], &mut |x| { u.push(x); u.len() < 9 }, &mut cx).unwrap();
        assert_eq!((t, s.stop_reason), (u, s2.stop_reason), "depth {} cancel", depth);
        same_state(&a, &b, &format!("depth {} cancel", depth));
    }
    // context full: max_ctx 64 with a 50-token prompt
    let small = |c: usize| QwenEngine::open(&d, QwenOptions { max_ctx: 64, ..opt(true, Some(&set), c) }).unwrap();
    let (a, b) = (small(0), small(7));
    let mut cx = CtxDrafter::new();
    let p: Vec<u32> = heavy[..50.min(heavy.len())].to_vec();
    a.reset(); b.reset();
    let (x, rx) = plain(&a, &p, 40, &[]);
    let (y, ry, _) = ctx(&b, &mut cx, &p, 40, &[]);
    assert_eq!((x, rx), (y, ry), "ctx_full");
    same_state(&a, &b, "ctx_full");
}

#[test]
fn ctx_draft_with_table() {
    let Some(d) = dir() else { return };
    let set = half_set(&d);
    let td = std::env::temp_dir().join(format!("sd_ctx_draft_table_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&td);
    let a = QwenEngine::open(&d, opt(true, Some(&set), 0)).unwrap();
    let b = QwenEngine::open(&d, QwenOptions { draft_table: Some(td.clone()), draft_table_bytes: 1 << 20, ..opt(true, Some(&set), 7) }).unwrap();
    let mut cx = CtxDrafter::new();
    let mut all = CtxStats::default();
    // twice: the second round finds the first round's replies in the table's history
    for round in 0..2 {
        for (name, p) in prompts(&a) {
            let s = scenario(&a, &b, &mut cx, &p, 40, &format!("table round {} {}", round, name));
            all.add(&s);
        }
        b.flush_draft_source();
    }
    report("compact fused + table", &all);
    assert!(all.accepted > 0);
    let _ = std::fs::remove_dir_all(&td);
}

#[test]
fn lookup_unit() {
    let mut cx = CtxDrafter::new();
    cx.push(&[1, 2, 3, 4, 5, 6, 7, 1, 2, 3]);
    // suffix [.., 2, 3] + pending 4 -> match [1 2 3 4] ending at 4, continuation 5 6 7 1
    let (m, t) = cx.lookup(4, 4);
    assert_eq!((m, t), (4, vec![5, 6, 7, 1]));
    // self-overlapping continuation: a b a b ... pending a -> b a b a
    let mut cx = CtxDrafter::new();
    cx.push(&[9, 8, 9, 8]);
    let (m, t) = cx.lookup(9, 5);
    assert!(m >= 3, "m {}", m);
    assert_eq!(t, vec![8, 9, 8, 9, 8]);
    // nothing to match
    let cx = CtxDrafter::new();
    assert_eq!(cx.lookup(5, 3), (0, vec![]));
}
