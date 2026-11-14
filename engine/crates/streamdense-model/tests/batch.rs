//! Batched decode (batch.rs) on the tiny model (models/tiny_qwen35, has mtp/): every sequence decoded in a batch is
//! token-identical to plain greedy decoding of it alone.
//! - 2 and 3 sequences, different prompts and lengths (1 .. 300 tokens), MTP head drafts and oracle drafts (p 0.5 / 0.9:
//!   long accepted runs, every replay length), adaptive depth (auto7) and fixed depth (7, capped by the row budget).
//! - one finishing early (short n_new, a stop token, a cancel), one joining mid-way, a solo run that yields to a
//!   batch and resumes alone after it.
//! - the slot's exit state: a follow-up turn on the batched slot == the same session decoded plain alone.
//! - q8 KV.
//! Run: cargo test --release --offline -p streamdense-model --test batch -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_model::batch::{row_budget, BatchSeq};
use streamdense_model::spec::DraftOverride;
use streamdense_model::{KvFormat, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}

fn open(seqs: usize, auto: bool, kv: KvFormat) -> Option<QwenEngine> {
    let dir = model_dir()?;
    let opt = QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, spec_auto: auto, seqs, kv, ..Default::default() };
    Some(QwenEngine::open(&dir, opt).expect("open tiny"))
}

fn prompts() -> Vec<Vec<u32>> {
    let p = format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT);
    let mut v: Vec<Vec<u32>> = match std::fs::read_to_string(&p) {
        Ok(s) => {
            let j: Value = serde_json::from_str(&s).unwrap();
            j["prompts"].as_array().unwrap().iter().take(3).map(|r| r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()).collect()
        }
        Err(_) => vec![],
    };
    let mut s = 0x1234_5678u64;
    let mut rnd = |n: usize| -> Vec<u32> { (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 2000) as u32 + 10 }).collect() };
    v.push(rnd(1));
    v.push(rnd(300));
    v.push(rnd(37));
    v
}

/// Plain greedy reference (its own engine, one slot).
struct Ref { eng: QwenEngine }
impl Ref {
    fn plain(&self, prompt: &[u32], n: usize) -> Vec<u32> { self.eng.generate(prompt, n, false).unwrap().tokens }
    /// Plain turns on one fresh sequence: turn 0 feeds its ids, later turns the pending last token + their ids.
    fn session(&self, turns: &[(Vec<u32>, usize)]) -> Vec<Vec<u32>> {
        self.eng.reset();
        let mut out: Vec<Vec<u32>> = vec![];
        for (ids, n) in turns {
            let mut feed = match out.last() { Some(t) => vec![*t.last().unwrap()], None => vec![] };
            feed.extend_from_slice(ids);
            let mut got = vec![];
            self.eng.feed_stream(&feed, *n, &[], &mut |t| { got.push(t); true }).unwrap();
            out.push(got);
        }
        out
    }
}

fn oracle(prompt: &[u32], plain: &[u32], p: f64, seed: u64) -> DraftOverride {
    if p <= 0.0 { return DraftOverride::None; }
    DraftOverride::Oracle { seq: prompt.iter().chain(plain.iter()).copied().collect(), p, seed }
}

/// Rounds until every sequence is done; returns the number of rounds.
fn drain(eng: &QwenEngine, seqs: &mut [&mut BatchSeq]) -> usize {
    let mut n = 0;
    while seqs.iter().any(|s| s.active()) {
        eng.batch_round(seqs, &mut |_, _| true).unwrap();
        n += 1;
        assert!(n < 10_000, "no progress");
    }
    n
}

#[test]
fn row_budgets() {
    assert_eq!(row_budget(1), vec![8]);
    assert_eq!(row_budget(2), vec![4, 4]);
    assert_eq!(row_budget(3), vec![3, 3, 2]);
    assert_eq!(row_budget(4), vec![2, 2, 2, 2]);
}

#[test]
fn batch2_equals_plain() {
    let (Some(eng), Some(r)) = (open(2, true, KvFormat::Bf16), open(1, false, KvFormat::Bf16)) else { return };
    let r = Ref { eng: r };
    let ps = prompts();
    let pairs = [(0usize, 4usize, 48usize, 40usize), (1, 3, 33, 64), (2, 5, 50, 20)];
    for (pi, &(a, b, na, nb)) in pairs.iter().enumerate() {
        let (a, b) = (&ps[a % ps.len()], &ps[b % ps.len()]);
        let (ga, gb) = (r.plain(a, na), r.plain(b, nb));
        for p in [0.0, 0.5, 0.9] {
            eng.select_seq(0); eng.reset();
            eng.select_seq(1); eng.reset();
            let mut sa = eng.batch_begin(0, a, na, &[], &oracle(a, &ga, p, 11), &mut |_| true).unwrap();
            let mut sb = eng.batch_begin(1, b, nb, &[], &oracle(b, &gb, p, 12), &mut |_| true).unwrap();
            let rounds = drain(&eng, &mut [&mut sa, &mut sb]);
            assert_eq!(sa.r.tokens, ga, "pair {} p {}: seq A != plain", pi, p);
            assert_eq!(sb.r.tokens, gb, "pair {} p {}: seq B != plain", pi, p);
            eprintln!("pair {} p {}: A {} tok ({}), B {} tok ({}) in {} rounds; A acc {}/{} B acc {}/{}, accept hist A {:?}",
                      pi, p, ga.len(), sa.done.unwrap(), gb.len(), sb.done.unwrap(), rounds, sa.r.accepted, sa.r.passes, sb.r.accepted, sb.r.passes, sa.r.accept_hist);
        }
    }
}

#[test]
fn batch3_early_finish_and_join() {
    let (Some(eng), Some(r)) = (open(3, true, KvFormat::Bf16), open(1, false, KvFormat::Bf16)) else { return };
    let r = Ref { eng: r };
    let ps = prompts();
    let (a, b, c) = (&ps[0], &ps[4 % ps.len()], &ps[1 % ps.len()]);
    let (na, nb, nc) = (70, 12, 45);
    let (ga, gb, gc) = (r.plain(a, na), r.plain(b, nb), r.plain(c, nc));
    for p in [0.0, 0.9] {
        for s in 0..3 { eng.select_seq(s); eng.reset(); }
        let mut sa = eng.batch_begin(0, a, na, &[], &oracle(a, &ga, p, 21), &mut |_| true).unwrap();
        let mut sb = eng.batch_begin(1, b, nb, &[], &oracle(b, &gb, p, 22), &mut |_| true).unwrap();
        // a few rounds of two, then C joins (its prefill between passes)
        for _ in 0..3 { eng.batch_round(&mut [&mut sa, &mut sb], &mut |_, _| true).unwrap(); }
        let mut sc = eng.batch_begin(2, c, nc, &[], &oracle(c, &gc, p, 23), &mut |_| true).unwrap();
        let mut three = 0;
        while sa.active() || sb.active() || sc.active() {
            if sa.active() && sb.active() && sc.active() { three += 1; }
            eng.batch_round(&mut [&mut sa, &mut sb, &mut sc], &mut |_, _| true).unwrap();
        }
        assert_eq!(sa.r.tokens, ga, "p {}: A != plain", p);
        assert_eq!(sb.r.tokens, gb, "p {}: B (early finish) != plain", p);
        assert_eq!(sc.r.tokens, gc, "p {}: C (joined mid-way) != plain", p);
        assert!(sb.r.tokens.len() < sa.r.tokens.len());
        eprintln!("p {}: A {} B {} C {} tokens; rounds with 3 active {}; batched passes A {} B {} C {}", p, ga.len(), gb.len(), gc.len(), three,
                  sa.batched, sb.batched, sc.batched);
    }
}

#[test]
fn stop_cancel_and_followup() {
    let (Some(eng), Some(r)) = (open(2, true, KvFormat::Bf16), open(1, false, KvFormat::Bf16)) else { return };
    let r = Ref { eng: r };
    let ps = prompts();
    let (a, b) = (&ps[2 % ps.len()], &ps[5 % ps.len()]);
    let (ga, gb) = (r.plain(a, 50), r.plain(b, 50));
    // A stops on a stop token (its 9th generated token), B is cancelled after 20 tokens
    let stop = ga[8];
    let ia = ga.iter().position(|&t| t == stop).unwrap();
    for p in [0.0, 0.9] {
        for s in 0..2 { eng.select_seq(s); eng.reset(); }
        let mut sa = eng.batch_begin(0, a, 50, &[stop], &oracle(a, &ga, p, 31), &mut |_| true).unwrap();
        let mut sb = eng.batch_begin(1, b, 50, &[], &oracle(b, &gb, p, 32), &mut |_| true).unwrap();
        let mut nb = 1;
        while sa.active() || sb.active() {
            eng.batch_round(&mut [&mut sa, &mut sb], &mut |i, _| { if i == 1 { nb += 1; nb <= 20 } else { true } }).unwrap();
        }
        assert_eq!(sa.done, Some("eos"));
        assert_eq!(sa.r.tokens, ga[..=ia].to_vec(), "p {}: A (stop token) != plain prefix", p);
        assert_eq!(sb.done, Some("cancelled"));
        assert_eq!(sb.r.tokens, gb[..21].to_vec(), "p {}: B (cancelled on its 21st token, which is emitted) != plain prefix", p);
        // follow-up turns on both slots (plain feed: the pending last token first, then an MTP turn) == the same
        // session decoded plain alone (feed_stream turns on one engine: same prefill chunks, same passes)
        let extra: Vec<u32> = vec![17, 99, 1234, 5];
        for (slot, prompt, toks) in [(0usize, a, &sa.r.tokens), (1, b, &sb.r.tokens)] {
            eng.select_seq(slot);
            assert_eq!(eng.pos(), prompt.len() + toks.len() - 1, "slot {}: position after the batch", slot);
            let mut feed = vec![*toks.last().unwrap()];
            feed.extend_from_slice(&extra);
            let mut got = vec![];
            eng.feed_stream(&feed, 24, &[], &mut |t| { got.push(t); true }).unwrap();
            let mut got2 = vec![];
            let mut feed2 = vec![*got.last().unwrap()];
            feed2.extend_from_slice(&[7, 8, 9]);
            eng.feed_stream_mtp(&feed2, 16, &[], &mut |t| { got2.push(t); true }).unwrap();
            let want = r.session(&[(prompt.clone(), toks.len()), (extra.clone(), 24), (vec![7, 8, 9], 16)]);
            assert_eq!(&want[0], toks, "p {} slot {}: reference session turn 1", p, slot);
            assert_eq!(got, want[1], "p {} slot {}: plain follow-up after the batch != plain session", p, slot);
            assert_eq!(got2, want[2], "p {} slot {}: MTP follow-up after the batch != plain session", p, slot);
        }
        eprintln!("p {}: stop at A token {}, B cancelled at 20; follow-ups plain + MTP == plain sessions", p, ia);
    }
}

#[test]
fn solo_yield_batch_resume() {
    let (Some(eng), Some(r)) = (open(2, true, KvFormat::Bf16), open(1, false, KvFormat::Bf16)) else { return };
    let r = Ref { eng: r };
    let ps = prompts();
    let (a, b) = (&ps[1 % ps.len()], &ps[3 % ps.len()]);
    let (ga, gb) = (r.plain(a, 90), r.plain(b, 30));
    for p in [0.0, 0.9] {
        for s in 0..2 { eng.select_seq(s); eng.reset(); }
        let mut sa = eng.batch_begin(0, a, 90, &[], &oracle(a, &ga, p, 41), &mut |_| true).unwrap();
        // alone until 20 tokens are out, then yield (a request arrived)
        let n = std::cell::Cell::new(1usize);
        eng.solo_run(&mut sa, &mut |_| { n.set(n.get() + 1); true }, &|| n.get() >= 20).unwrap();
        assert!(sa.active(), "A finished before the yield");
        let solo1 = sa.r.passes;
        let mut sb = eng.batch_begin(1, b, 30, &[], &oracle(b, &gb, p, 42), &mut |_| true).unwrap();
        while sb.active() { eng.batch_round(&mut [&mut sa, &mut sb], &mut |_, _| true).unwrap(); }
        let batched = sa.batched;
        // B is done: A alone again
        eng.solo_run(&mut sa, &mut |_| true, &|| false).unwrap();
        assert!(!sa.active());
        assert_eq!(sa.r.tokens, ga, "p {}: A (solo / batch / solo) != plain", p);
        assert_eq!(sb.r.tokens, gb, "p {}: B != plain", p);
        assert!(batched > 0 && solo1 > 0);
        eprintln!("p {}: A {} solo passes, {} batched, {} total; B {} tokens", p, solo1, batched, sa.r.passes, gb.len());
    }
}

#[test]
fn fixed_depth_and_q8() {
    for (auto, kv, name) in [(false, KvFormat::Bf16, "fixed7 bf16"), (true, KvFormat::Q8, "auto7 q8")] {
        let (Some(eng), Some(r)) = (open(3, auto, kv), open(1, false, kv)) else { return };
        let r = Ref { eng: r };
        let ps = prompts();
        let (a, b, c) = (&ps[0], &ps[4 % ps.len()], &ps[5 % ps.len()]);
        let (ga, gb, gc) = (r.plain(a, 40), r.plain(b, 25), r.plain(c, 33));
        for p in [0.0, 0.9] {
            for s in 0..3 { eng.select_seq(s); eng.reset(); }
            let mut sa = eng.batch_begin(0, a, 40, &[], &oracle(a, &ga, p, 51), &mut |_| true).unwrap();
            let mut sb = eng.batch_begin(1, b, 25, &[], &oracle(b, &gb, p, 52), &mut |_| true).unwrap();
            let mut sc = eng.batch_begin(2, c, 33, &[], &oracle(c, &gc, p, 53), &mut |_| true).unwrap();
            drain(&eng, &mut [&mut sa, &mut sb, &mut sc]);
            assert_eq!(sa.r.tokens, ga, "{} p {}: A != plain", name, p);
            assert_eq!(sb.r.tokens, gb, "{} p {}: B != plain", name, p);
            assert_eq!(sc.r.tokens, gc, "{} p {}: C != plain", name, p);
            eprintln!("{} p {}: 3 sequences == plain (A acc {}/{} passes)", name, p, sa.r.accepted, sa.r.passes);
        }
    }
}

#[test]
fn fused_mtp_step_equals_serial() {
    // the fused chained draft step (one fc / MTP layer / lm_head over all sequences' rows) gives the same drafts as
    // one step per sequence: same passes, accepted drafts and depth choices, on the MTP head's own drafts
    let Some(eng) = open(3, false, KvFormat::Bf16) else { return }; // fixed depth 7: every sequence chains to its row budget
    let ps = prompts();
    let (a, b, c) = (&ps[0], &ps[1 % ps.len()], &ps[4 % ps.len()]);
    let mut runs = vec![];
    for serial in [false, true] {
        if serial { std::env::set_var("SD_BATCH_MTP", "serial"); } else { std::env::remove_var("SD_BATCH_MTP"); }
        for s in 0..3 { eng.select_seq(s); eng.reset(); }
        let mut sa = eng.batch_begin(0, a, 40, &[], &DraftOverride::None, &mut |_| true).unwrap();
        let mut sb = eng.batch_begin(1, b, 40, &[], &DraftOverride::None, &mut |_| true).unwrap();
        let mut sc = eng.batch_begin(2, c, 40, &[], &DraftOverride::None, &mut |_| true).unwrap();
        drain(&eng, &mut [&mut sa, &mut sb, &mut sc]);
        let st = |s: &BatchSeq| (s.r.tokens.clone(), s.r.passes, s.r.accepted, s.r.accept_hist.clone(), s.r.depth_hist.clone(), s.r.drafts.clone(), s.chained);
        runs.push((st(&sa), st(&sb), st(&sc)));
    }
    std::env::remove_var("SD_BATCH_MTP");
    assert_eq!(runs[0], runs[1], "fused MTP step != serial steps");
    let d = runs[0].0 .6 + runs[0].1 .6 + runs[0].2 .6;
    assert!(d > 0, "no chained draft steps");
    eprintln!("fused == serial: {} chained draft steps over 3 sequences", d);
}
