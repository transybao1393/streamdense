//! 12 GB mode backend (compact.rs) with context drafts (`QwenOptions.ctx_draft`, engine ctx_draft.rs) on the tiny
//! model (models/tiny_qwen35), Metal, the gateway's defaults (fast streamed read + fused decode). Lossless, no
//! tolerance: a ctx_draft backend (lookup + draft table in a temp dir), a dry one (`ctx_dry`: same RAM, plain decode)
//! and a plain backend (ctx_draft 0) emit the same tokens on every feed of a multi-turn session:
//! - first prompt, a continuation (pending token + suffix), a continuation that repeats earlier context, a long feed in
//!   pieces, reset + a new prompt;
//! - a snapshot save / reset / restore in the middle (the lookup history is replaced by the snapshot's tokens);
//! - FeedStats.ctx_draft / spec are set only on the drafting backend; mode_info unchanged.
use std::sync::Arc;
use streamdense_gateway::compact::CompactBackend;
use streamdense_gateway::Backend;
use streamdense_model::ffn_set::{SetRule, T1};
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}

fn t1(l: usize, n: usize) -> T1 {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    T1 { layers: l, inter: n, global: (0..l * n).map(|_| r() / n as f32).collect(), down_norm: (0..l * n).map(|_| 0.5 + r()).collect() }
}

fn opts(rule: &SetRule, ctx: usize, table: Option<&std::path::Path>) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: Some(Arc::new(rule.initial())), ffn_stream_tokens: 64,
                  ffn_stream_slots: 2, ffn_stream_host: true, ffn_stream_fast: true, prefill_t: 32, decode_fused: true, ctx_draft: ctx,
                  draft_table: table.map(|p| p.to_path_buf()), draft_table_bytes: 1 << 20, ..Default::default() }
}

fn feed(b: &CompactBackend, ids: &[u32], n: usize) -> (Vec<u32>, streamdense_gateway::FeedStats) {
    let mut t = vec![];
    let f = b.feed(ids, n, &[], false, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(f.gen, t.len());
    (t, f)
}

fn ids(n: usize, seed: u32) -> Vec<u32> { (0..n as u32).map(|i| (i * 7919 + seed * 104_729 + 13) % 2000).collect() }

#[test]
fn compact_ctx_draft_equals_plain() {
    let Some(d) = dir() else { return };
    let probe = QwenEngine::open(&d, QwenOptions { max_ctx: 64, ..Default::default() }).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let rule = SetRule::new(t1(l, n), n / 16 * 8).unwrap();
    let td = std::env::temp_dir().join(format!("sd_ctx_gw_table_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&td);
    let ep = QwenEngine::open(&d, opts(&rule, 0, None)).unwrap();
    let ec = QwenEngine::open(&d, opts(&rule, 7, Some(&td))).unwrap();
    let ed = QwenEngine::open(&d, opts(&rule, 7, None)).unwrap();
    let info = serde_json::json!({"mode": "12gb"});
    let bp = CompactBackend::new(&ep, rule.clone(), info.clone()).unwrap();
    let bc = CompactBackend::new(&ec, rule.clone(), info.clone()).unwrap();
    let mut bd = CompactBackend::new(&ed, rule.clone(), info.clone()).unwrap();
    bd.ctx_dry = true;
    assert!(!bp.ctx_live() && bc.ctx_live() && !bd.ctx_live());
    assert_eq!(bc.mode_info(), Some(info));
    let snap = std::env::temp_dir().join(format!("sd_ctx_gw_snap_{}.sdsnap", std::process::id()));
    let nn = 40;

    for round in 0..2 {
        let all = [&bp, &bc, &bd];
        for b in all { b.reset(); }
        // the same prompt both rounds: round 1 finds round 0's replies in the draft table (accepted drafts)
        let p1 = ids(70, 1);
        let mut seq: Vec<u32> = vec![];
        // turn 1: prompt (2 pieces)
        let mut outs = vec![];
        for b in all { outs.push(feed(b, &p1, nn)); }
        assert_eq!(outs[0].0, outs[1].0, "round {} turn 1: ctx != plain", round);
        assert_eq!(outs[0].0, outs[2].0, "round {} turn 1: dry != plain", round);
        assert!(outs[0].1.ctx_draft.is_none() && outs[2].1.ctx_draft.is_none() && outs[1].1.ctx_draft.is_some());
        seq.extend_from_slice(&p1);
        let r1 = outs[0].0.clone();
        seq.extend_from_slice(&r1[..r1.len() - 1]); // decoded tokens in the KV (the last one is pending)
        // turn 2: pending + suffix that repeats the prompt and the reply (repeat-heavy)
        let mut f2 = vec![*r1.last().unwrap()];
        f2.extend_from_slice(&p1[..30]);
        f2.extend_from_slice(&r1[..r1.len() - 1]);
        f2.extend_from_slice(&p1[..20]);
        let mut outs = vec![];
        for b in all { outs.push(feed(b, &f2, nn)); }
        assert_eq!(outs[0].0, outs[1].0, "round {} turn 2: ctx != plain", round);
        assert_eq!(outs[0].0, outs[2].0, "round {} turn 2: dry != plain", round);
        seq.extend_from_slice(&f2);
        let r2 = outs[0].0.clone();
        eprintln!("round {} turn 2 ctx_draft: {}", round, outs[1].1.ctx_draft.as_ref().unwrap());
        // snapshot of the drafting backend (and the plain one) after turn 2, reset, restore
        let mut held = seq.clone();
        held.extend_from_slice(&r2[..r2.len() - 1]);
        assert_eq!(held.len(), bc.pos());
        let snap_p = snap.with_extension(format!("p{}.sdsnap", round));
        bp.save_snapshot(&snap_p, &held).unwrap();
        bc.save_snapshot(&snap, &held).unwrap();
        bp.reset();
        bc.reset();
        assert_eq!(bp.restore_snapshot(&snap_p).unwrap(), held);
        assert_eq!(bc.restore_snapshot(&snap).unwrap(), held);
        // turn 3: pending + the same suffix again
        let mut f3 = vec![*r2.last().unwrap()];
        f3.extend_from_slice(&f2[1..]);
        let mut outs = vec![];
        for b in all { outs.push(feed(b, &f3, nn)); }
        assert_eq!(outs[0].0, outs[1].0, "round {} turn 3 (after restore): ctx != plain", round);
        assert_eq!(outs[0].0, outs[2].0, "round {} turn 3: dry != plain", round);
        eprintln!("round {} turn 3 ctx_draft: {}", round, outs[1].1.ctx_draft.as_ref().unwrap());
        let _ = std::fs::remove_file(&snap_p);
    }
    let tot = bc.ctx_totals().unwrap();
    eprintln!("ctx totals: passes {}, drafts {} accepted {} ({:.2}), tokens {} = {:.2} per pass, lookup {}/{}, table {}/{}",
              tot.passes, tot.drafted, tot.accepted, tot.acceptance(), tot.tokens, tot.tokens_per_pass(), tot.lookup_acc, tot.lookup_drafts,
              tot.table_acc, tot.table_drafts);
    assert!(tot.passes > 0);
    assert!(tot.accepted > 0, "round 1 should accept the table's drafts of round 0's replies");
    let _ = std::fs::remove_file(&snap);
    let _ = std::fs::remove_dir_all(&td);
}

/// History index as the third draft source (temp dir, 1 MB budget) + the fused verify matmuls: the backend with both
/// emits the plain backend's tokens; round 1 finds round 0's replies in the index (pushed by the backend after each
/// feed through the updater) and accepts its drafts; the updater is never left paused.
#[test]
fn compact_ctx_draft_history_index() {
    use streamdense_core::history_index::{Config, HistoryIndex, Updater};
    use streamdense_gateway::compact::HistoryHook;
    let Some(d) = dir() else { return };
    let probe = QwenEngine::open(&d, QwenOptions { max_ctx: 64, ..Default::default() }).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    let rule = SetRule::new(t1(l, n), n / 16 * 8).unwrap();
    let td = std::env::temp_dir().join(format!("sd_ctx_gw_hist_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&td);
    let mut c = Config::with_dir(&td);
    c.budget_bytes = 1 << 20;
    let idx = Arc::new(HistoryIndex::open(c).unwrap().0);
    let enc: streamdense_core::history_index::Encoder = Arc::new(|s: &str| s.bytes().map(|b| b as u32).collect());
    let up = Updater::spawn(idx.clone(), enc, Default::default()).unwrap();
    let ep = QwenEngine::open(&d, opts(&rule, 0, None)).unwrap();
    let eh = QwenEngine::open(&d, QwenOptions { verify_fused: true, history_bytes: 1 << 20, ..opts(&rule, 7, None) }).unwrap();
    let info = serde_json::json!({"mode": "12gb"});
    let bp = CompactBackend::new(&ep, rule.clone(), info.clone()).unwrap();
    let mut bh = CompactBackend::new(&eh, rule.clone(), info.clone()).unwrap();
    bh.set_history(HistoryHook { index: idx.clone(), updater: up });
    let nn = 40;
    let mut last = None;
    for round in 0..2 {
        for b in [&bp, &bh] { b.reset(); }
        for (t, seed) in [(0, 3u32), (1, 4)] {
            let p = ids(50 + 10 * t, seed);
            let (a, _) = feed(&bp, &p, nn);
            let (b, f) = feed(&bh, &p, nn);
            assert_eq!(a, b, "round {} turn {}: history + verify_fused != plain", round, t);
            assert!(!bh.history().unwrap().updater.is_paused(), "updater left paused");
            eprintln!("round {} turn {} ctx_draft: {}", round, t, f.ctx_draft.as_ref().unwrap());
            // continuation (pending token alone)
            let (a2, _) = feed(&bp, &[*a.last().unwrap()], nn);
            let (b2, _) = feed(&bh, &[*b.last().unwrap()], nn);
            assert_eq!(a2, b2, "round {} turn {} continuation", round, t);
        }
        assert!(bh.history().unwrap().updater.wait_idle(std::time::Duration::from_secs(10)), "updater did not drain");
        last = Some(bh.ctx_totals().unwrap());
    }
    let tot = last.unwrap();
    eprintln!("history totals: passes {}, drafts {} accepted {}, history {}/{}, lookup {}/{}; index {} tokens",
              tot.passes, tot.drafted, tot.accepted, tot.history_acc, tot.history_drafts, tot.lookup_acc, tot.lookup_drafts, idx.stats().tokens);
    assert!(idx.stats().tokens > 0, "no reply reached the index");
    assert!(tot.history_acc > 0, "round 1 should accept history drafts of round 0's replies");
    assert!(idx.stats().feedback.iter().any(|f| f.0 > 0), "no feedback reached the index");
    drop(bh);
    let _ = std::fs::remove_dir_all(&td);
}
