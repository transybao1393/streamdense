//! 12 GB mode backend (compact.rs) on the tiny model (models/tiny_qwen35), Metal, against one streamed read of the
//! whole sequence on a second engine (`stream_prefill` + greedy compact decode; that path == the two-process path bit
//! for bit: streamdense-model/tests/ffn_compact.rs). Bit-exact, no tolerance:
//! - a prefix extension (request 1 = prompt, 1 token; request 2 = the pending token + a suffix, seeded with request 1's
//!   sums) == one read of prompt + token + suffix: prompt sums, set, decode tokens;
//! - a feed longer than the piece size (3 streamed passes, set built after the last) == one read;
//! - `reset` drops the sums (the next prompt == a fresh read);
//! - a held prefix whose length is not a multiple of prefill_k: KV, ag sums, set and tokens still equal; the an sums
//!   only up to fp32 order (they are added per chunk);
//! - MTP decode after the streamed read == plain decode (same engine);
//! - status: mode_info, stream stats per feed.
use std::sync::Arc;
use streamdense_gateway::compact::CompactBackend;
use streamdense_gateway::Backend;
use streamdense_model::ffn_set::{SetRule, T1};
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}

/// Deterministic synthetic global score / down norms for the tiny dims.
fn t1(l: usize, n: usize) -> T1 {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    T1 { layers: l, inter: n, global: (0..l * n).map(|_| r() / n as f32).collect(), down_norm: (0..l * n).map(|_| 0.5 + r()).collect() }
}

fn opts(rule: &SetRule, stream: usize, mtp: bool) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, mtp, ffn_compact: Some(Arc::new(rule.initial())), ffn_stream_tokens: stream,
                  ffn_stream_slots: 2, ffn_stream_host: true, ..Default::default() }
}

/// One streamed read of `ids` from a reset on `r` + greedy compact decode of `n` tokens: (tokens, sums [l][an | ag]).
fn reference(r: &QwenEngine, rule: &SetRule, ids: &[u32], n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let inter = r.cfg.inter;
    let mut sums = vec![vec![0f32; 2 * inter]; r.cfg.layers];
    r.reset();
    let st = r.stream_prefill(ids, &mut |li, an, ag| {
        sums[li][..inter].copy_from_slice(an);
        sums[li][inter..].copy_from_slice(ag);
        Ok(rule.layer(li, an, ag))
    }).unwrap();
    let mut t = vec![st.tok];
    while t.len() < n { t.push(r.fsel_step(*t.last().unwrap()).unwrap().0); }
    (t, sums)
}

fn feed(b: &CompactBackend, ids: &[u32], n: usize, mtp: bool) -> (Vec<u32>, streamdense_gateway::FeedStats) {
    let mut t = vec![];
    let f = b.feed(ids, n, &[], mtp, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(f.gen, t.len());
    (t, f)
}

fn ids(n: usize, seed: u32) -> Vec<u32> { (0..n as u32).map(|i| (i * 7919 + seed * 104_729 + 13) % 2000).collect() }

#[test]
fn compact_backend_equals_one_read() {
    let Some(d) = dir() else { return };
    let probe = QwenEngine::open(&d, QwenOptions { max_ctx: 64, ..Default::default() }).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let rule = SetRule::new(t1(l, n), n / 16 * 8).unwrap(); // half the neurons
    // backend under test: 64-token pieces; reference: one pass for everything here
    let ea = QwenEngine::open(&d, opts(&rule, 64, false)).unwrap();
    let er = QwenEngine::open(&d, opts(&rule, 512, false)).unwrap();
    let info = serde_json::json!({"mode": "12gb", "k": rule.k});
    let b = CompactBackend::new(&ea, rule.clone(), info.clone()).unwrap();
    assert_eq!(b.piece, 64);
    assert_eq!(b.mode_info(), Some(info));
    let nn = 16;

    // 1. prefix extension, held length 48 (a multiple of prefill_k)
    let (p1, sfx) = (ids(48, 1), ids(37, 2));
    b.reset();
    let (t0, f1) = feed(&b, &p1, 1, false);
    assert_eq!(t0.len(), 1);
    assert_eq!(b.pos(), 48, "the generated token stays pending");
    let s1 = f1.stream.clone().unwrap();
    assert_eq!((s1.pieces, s1.tokens), (1, 48));
    assert!(s1.ssd_bytes > 0 && s1.wall_s > 0.0 && f1.prefill_s == s1.wall_s);
    let mut f2_ids = vec![t0[0]];
    f2_ids.extend_from_slice(&sfx);
    let (t2, f2) = feed(&b, &f2_ids, nn, false);
    let mut all = p1.clone();
    all.extend_from_slice(&f2_ids);
    let (tr, sr) = reference(&er, &rule, &all, nn);
    assert!(b.sums() == sr, "prefix: sums differ from one read");
    assert!(*ea.ffn_compact_set().unwrap() == *er.ffn_compact_set().unwrap(), "prefix: set differs");
    assert_eq!(t2, tr[..t2.len()], "prefix: tokens differ from one read");
    assert_eq!(f2.stream.as_ref().unwrap().tokens, 38);
    eprintln!("prefix extension == one read: sums / set / {} tokens", t2.len());

    // 2. a long feed in 3 pieces (64 + 64 + 22), set built after the last
    let p3 = ids(150, 3);
    b.reset();
    let (t3, f3) = feed(&b, &p3, nn, false);
    assert_eq!(f3.stream.as_ref().unwrap().pieces, 3);
    let (tr, sr) = reference(&er, &rule, &p3, nn);
    assert!(b.sums() == sr, "pieces: sums differ");
    assert!(*ea.ffn_compact_set().unwrap() == *er.ffn_compact_set().unwrap(), "pieces: set differs");
    assert_eq!(t3, tr[..t3.len()], "pieces: tokens differ");
    eprintln!("3 pieces == one read");

    // 3. reset drops the sums: a new prompt == a fresh read
    let p4 = ids(29, 4);
    b.reset();
    let (t4, _) = feed(&b, &p4, nn, false);
    let (tr, sr) = reference(&er, &rule, &p4, nn);
    assert!(b.sums() == sr && t4 == tr[..t4.len()], "after reset");

    // 4. held length 45 (not a multiple of 8): KV + ag + set + tokens equal; an up to fp32 order
    let p5 = ids(45, 5);
    b.reset();
    let (t0, _) = feed(&b, &p5, 1, false);
    let mut f5 = vec![t0[0]];
    f5.extend_from_slice(&sfx);
    let (t5, _) = feed(&b, &f5, nn, false);
    let mut all = p5.clone();
    all.extend_from_slice(&f5);
    let (tr, sr) = reference(&er, &rule, &all, nn);
    let sa = b.sums();
    let mut an_rel = 0f32;
    for li in 0..l {
        assert!(sa[li][n..] == sr[li][n..], "unaligned: ag sums must be exact (token by token)");
        for j in 0..n { an_rel = an_rel.max((sa[li][j] - sr[li][j]).abs() / sr[li][j].abs().max(1e-20)); }
    }
    assert!(an_rel < 1e-5, "unaligned: an sums differ by {} (relative)", an_rel);
    assert!(*ea.ffn_compact_set().unwrap() == *er.ffn_compact_set().unwrap(), "unaligned: set differs");
    assert_eq!(t5, tr[..t5.len()], "unaligned: tokens differ");
    eprintln!("unaligned held prefix: tokens / set / ag equal, an max relative diff {:e}", an_rel);
    drop(b);
    drop(ea);
    drop(er);

    // 5. MTP decode after the streamed read == plain decode on the same (MTP) engine
    let em = QwenEngine::open(&d, opts(&rule, 64, true)).unwrap();
    let bm = CompactBackend::new(&em, rule.clone(), serde_json::json!({})).unwrap();
    assert!(bm.mtp_available());
    let p6 = ids(40, 6);
    bm.reset();
    let (tp, _) = feed(&bm, &p6, 24, false);
    bm.reset();
    let (tm, fm) = feed(&bm, &p6, 24, true);
    assert!(fm.spec.is_some());
    assert_eq!(tm, tp, "MTP decode != plain decode after the streamed read");
    eprintln!("MTP == plain: {} tokens, spec {:?}", tm.len(), fm.spec);
    drop(bm);
    drop(em);

    // 6. fast streamed read ([forheader] fast, prefill_t 32): pieces of 64 (a multiple of prefill_t) == one fast read
    let fo = |s: usize| QwenOptions { ffn_stream_fast: true, prefill_t: 32, ..opts(&rule, s, false) };
    let (ef, ef1) = (QwenEngine::open(&d, fo(64)).unwrap(), QwenEngine::open(&d, fo(512)).unwrap());
    let bf = CompactBackend::new(&ef, rule.clone(), serde_json::json!({})).unwrap();
    assert_eq!(bf.piece, 64);
    let p7 = ids(150, 7);
    bf.reset();
    let (t7, f7) = feed(&bf, &p7, nn, false);
    assert_eq!(f7.stream.as_ref().unwrap().pieces, 3);
    let (tr, sr) = reference(&ef1, &rule, &p7, nn);
    assert!(bf.sums() == sr, "fast pieces: sums differ");
    assert!(*ef.ffn_compact_set().unwrap() == *ef1.ffn_compact_set().unwrap(), "fast pieces: set differs");
    assert_eq!(t7, tr[..t7.len()], "fast pieces: tokens differ");
    eprintln!("fast read: 3 pieces == one read");
}
