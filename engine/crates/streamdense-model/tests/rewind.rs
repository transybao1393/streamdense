//! Rewind checkpoints (rewind.rs) on the tiny model (models/tiny_qwen35, has mtp/): plain, MTP (depth 1), MTP auto
//! + context drafts; exact and fast prefill. Compared bit for bit: emitted tokens and the recorded per-position
//! argmax / top-1 / top-2 logits (rec table) of every position the continuation computes.
//! 1. turn-start checkpoint (what the gateway takes): A -> reply, checkpoint, [pending] + B -> reply, checkpoint 2,
//!    rewind to checkpoint 1 (2 dropped), [pending] + B' (B changed at its 4th token) -> reply ==
//!    a fresh engine running A -> reply, [pending] + B' -> reply (the same continuation without the detour); with the
//!    exact prefill also == one fresh prefill of the whole prompt A + reply + B'.
//! 2. checkpoint at a prefill-chunk boundary (256): A0, checkpoint, A1 -> reply, rewind, A1' -> reply == one fresh
//!    prefill of A0 + A1' (every mode, fast prefill included: same chunks).
//! 3. validity: reset / snapshot restore invalidate every checkpoint; the budget row is there.
//!
//! Run: cargo test --release -p streamdense-model --test rewind -- --nocapture --test-threads 1
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_model::QwenConfig;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}

fn chunk(n: usize, k: u32) -> Vec<u32> { (0..n as u32).map(|i| (i * 71 + k * 13) % 1500 + 20).collect() }

/// Feed `ids` from the current position and decode up to `n` tokens (the engine's decode mode for `mtp`).
fn run(eng: &QwenEngine, ids: &[u32], n: usize, mtp: bool) -> Vec<u32> {
    let mut out = vec![];
    if mtp { eng.feed_stream_mtp(ids, n, &[], &mut |t| { out.push(t); true }).unwrap(); }
    else { eng.feed_stream(ids, n, &[], &mut |t| { out.push(t); true }).unwrap(); }
    out
}

/// rec entries (token, top-1 bits, top-2 bits) of the positions after a feed ending at `end` that emitted `n` tokens.
fn rec(eng: &QwenEngine, end: usize, n: usize) -> Vec<(u32, u32, u32)> {
    (end - 1..end - 1 + n).map(|i| { let (t, a, b) = eng.rec_token(i); (t, a.to_bits(), b.to_bits()) }).collect()
}

#[test]
fn rewind_exact_tiny() {
    let Some(dir) = model_dir() else { return };
    let cfg = QwenConfig::from_model_dir(&dir).unwrap();
    let per = streamdense_model::rewind::ckpt_bytes(&cfg) as u64;
    let cases: Vec<(&str, QwenOptions, bool)> = vec![
        ("plain exact", QwenOptions { mtp: false, prefill_exact: true, ..Default::default() }, false),
        ("plain fast", QwenOptions { mtp: false, ..Default::default() }, false),
        ("mtp d1 fast", QwenOptions { mtp: true, ..Default::default() }, true),
        ("mtp d1 exact", QwenOptions { mtp: true, prefill_exact: true, ..Default::default() }, true),
        ("mtp auto+ctx fast", QwenOptions { mtp: true, spec_depth: 7, spec_auto: true, mtp_ctx_draft: 7, ..Default::default() }, true),
        ("mtp auto+ctx exact", QwenOptions { mtp: true, spec_depth: 7, spec_auto: true, mtp_ctx_draft: 7, prefill_exact: true, ..Default::default() }, true),
    ];
    for (name, o, mtp) in cases {
        let exact = o.prefill_exact;
        let eng = QwenEngine::open(&dir, QwenOptions { max_ctx: 1024, rewind_bytes: 4 * per, ..o }).expect("open tiny");
        assert_eq!(eng.ckpt_slots(), 4, "{}: 4 checkpoints from the budget", name);
        assert!(eng.budget_report.contains("rewind checkpoints (4 x"), "{}: budget row\n{}", name, eng.budget_report);

        // ---- 1. turn-start checkpoint
        let (a, b) = (chunk(300, 1), chunk(120, 2));
        let mut b2 = b.clone();
        b2[3] = (b2[3] + 1) % 1500 + 20;
        eng.reset();
        let g1 = run(&eng, &a, 8, mtp);
        let pend = *g1.last().unwrap();
        let p = eng.pos();
        assert_eq!(p, a.len() + g1.len() - 1, "{}: pos after turn 1", name);
        eng.ckpt_save(0).unwrap();
        let mut f2 = vec![pend];
        f2.extend_from_slice(&b);
        let g2 = run(&eng, &f2, 8, mtp);
        eng.ckpt_save(1).unwrap();
        let q = eng.pos();
        assert_eq!((eng.ckpt_pos(0), eng.ckpt_pos(1)), (Some(p), Some(q)), "{}", name);
        assert_eq!(eng.ckpt_restore(0).unwrap(), p, "{}: rewound", name);
        assert_eq!((eng.pos(), eng.ckpt_pos(0), eng.ckpt_pos(1)), (p, Some(p), None), "{}: checkpoint past the rewind dropped", name);
        let mut f3 = vec![pend];
        f3.extend_from_slice(&b2);
        let n3 = 24;
        let g3 = run(&eng, &f3, n3, mtp);
        let r3 = rec(&eng, p + f3.len(), g3.len());
        // reference: the same continuation without the detour (fresh engine state)
        eng.reset();
        assert_eq!(run(&eng, &a, 8, mtp), g1, "{}: turn 1 replays", name);
        let h3 = run(&eng, &f3, n3, mtp);
        let eq_cont = h3 == g3 && rec(&eng, p + f3.len(), h3.len()) == r3;
        // one fresh prefill of the whole prompt
        eng.reset();
        let mut whole = a.clone();
        whole.extend_from_slice(&g1[..g1.len() - 1]);
        whole.extend_from_slice(&f3);
        let w3 = run(&eng, &whole, n3, mtp);
        let eq_fresh = w3 == g3 && rec(&eng, whole.len(), w3.len()) == r3;
        println!("  {:18} turn-start rewind to {} (of {}; turn 2 reply {:?}..): == continuation {} | == one fresh prefill {}{}",
                 name, p, q, &g2[..2], eq_cont, eq_fresh, if exact { "" } else { " (fast prefill: chunking differs, not required)" });
        assert!(eq_cont, "{}: rewind != the same continuation from a fresh state", name);
        if exact { assert!(eq_fresh, "{}: exact prefill: rewind != one fresh prefill", name); }

        // ---- 2. checkpoint on a prefill chunk boundary (prefill_t 128): == one fresh prefill in every mode
        let (a0, a1) = (chunk(256, 3), chunk(100, 4));
        let mut a1b = a1.clone();
        a1b[0] = (a1b[0] + 7) % 1500 + 20;
        eng.reset();
        run(&eng, &a0, 1, mtp);
        assert_eq!(eng.pos(), 256, "{}", name);
        eng.ckpt_save(2).unwrap();
        assert_eq!(eng.ckpt_pos(0), None, "{}: reset invalidated the old checkpoints", name);
        run(&eng, &a1, 12, mtp);
        assert_eq!(eng.ckpt_restore(2).unwrap(), 256, "{}", name);
        let g = run(&eng, &a1b, n3, mtp);
        let r = rec(&eng, 256 + a1b.len(), g.len());
        eng.reset();
        let mut whole = a0.clone();
        whole.extend_from_slice(&a1b);
        let w = run(&eng, &whole, n3, mtp);
        let eq = w == g && rec(&eng, whole.len(), w.len()) == r;
        println!("  {:18} chunk-boundary rewind to 256: == one fresh prefill {} ({} tokens)", name, eq, g.len());
        assert!(eq, "{}: aligned rewind != one fresh prefill", name);

        // ---- 3. a snapshot restore starts a new epoch; the checkpoints of the snapshot's own lineage stay valid
        eng.reset();
        run(&eng, &a0, 1, mtp);
        let p3 = eng.pos();
        eng.ckpt_save(3).unwrap();
        let s = eng.snapshot(&a0).unwrap();
        eng.reset();
        assert_eq!(eng.ckpt_pos(3), None, "{}: reset invalidates", name);
        eng.restore(&s).unwrap();
        assert_eq!(eng.ckpt_pos(3), Some(p3), "{}: restore of the same lineage keeps the checkpoint", name);
        // a snapshot of another lineage (same length, other ids) does not
        eng.reset();
        let other: Vec<u32> = a0.iter().map(|&t| if t > 300 { t - 1 } else { t }).collect();
        run(&eng, &other, 1, mtp);
        let so = eng.snapshot(&other).unwrap();
        eng.reset();
        eng.restore(&so).unwrap();
        assert_eq!(eng.ckpt_pos(3), None, "{}: restore of another lineage invalidates", name);
        assert!(eng.ckpt_restore(3).is_err(), "{}", name);
    }
}
