//! S1 self-test (PF-A / PF-B wiring): fast prefill (GEMM + prefill mixers, `prefill_t` row chunks) vs the exact
//! prefill (`prefill_exact`, K <= 8 GEMV chunks, bit-exact vs MLX stepwise) on the tiny model (models/tiny_qwen35).
//! The fast path is NOT bit-exact, so most checks are structural (they catch wiring bugs, not 1-ulp noise):
//! - teacher fast vs exact on ctx1k / readme4k: top-1 agreement, logit error in bf16 ulp of the common top-32 tokens,
//!   and the error at chunk-boundary rows (p % T in 0..4) vs interior rows (a state / position hand-off bug at a
//!   boundary shows up there); also with prefill_t = 9 (a boundary every 9 rows).
//! - lm_head on the last row only: the fast prefill's first token / margin == the fast teacher's last-row top-1 and
//!   top1 - top2 bitwise (same chunks; the K = 1 head is K-invariant), so prefill runs the head on the right row.
//! - chunk boundaries: prompt lengths 8, 9, 127, 128, 129, 300: <= 8 rows is bit-identical (no fast pass); else any
//!   token difference must sit at a near-tie (exact margin <= 3 bf16 ulp at that step, via the exact teacher).
//! - serve continue (feed_stream, the `qwen serve` feed/pending logic) with fast prefill: 3 turns (suffix > 8 rows
//!   and <= 8 rows) vs a fresh run on the full history; positions equal; token diffs only at near-ties; the same
//!   session with the exact prefill stays bit-identical to its fresh run.
//! Run: cargo test --release --offline -p streamdense-model --test qwen_prefill -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_model::{QwenEngine, QwenOptions, TopK};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const NEAR_TIE_ULP: f32 = 3.0;

fn open(exact: bool, prefill_t: usize) -> Option<QwenEngine> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", d)).exists() { eprintln!("SKIP: {} missing", d); return None; }
    let opt = QwenOptions { max_ctx: 4096, prefill_exact: exact, prefill_t, ..Default::default() };
    Some(QwenEngine::open(&d, opt).expect("open tiny"))
}

fn prompt(id: &str) -> Vec<u32> {
    let s = std::fs::read_to_string(format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT)).expect("tiny prompts");
    let j: Value = serde_json::from_str(&s).unwrap();
    j["prompts"].as_array().unwrap().iter().find(|r| r["id"] == id).map(|r| r["ids"].as_array().unwrap().iter()
        .map(|x| x.as_u64().unwrap() as u32).collect()).unwrap_or_else(|| panic!("prompt {}", id))
}

/// bf16 grid spacing at |x|.
fn ulp(x: f32) -> f32 { f32::from_bits(((x.abs().max(1e-30).to_bits() >> 23) << 23) - (7 << 23)) }

/// Max |logit diff| over tokens in both top-k lists, in ulp at the exact value.
fn logit_err(e: &TopK, f: &TopK) -> f32 {
    let mut m = 0f32;
    for (i, &t) in e.idx.iter().enumerate() {
        if let Some(j) = f.idx.iter().position(|&u| u == t) { m = m.max((f.val[j] - e.val[i]).abs() / ulp(e.val[i])); }
    }
    m
}
fn margin_ulp(t: &TopK) -> f32 { (t.val[0] - t.val[1]) / ulp(t.val[0]) }

struct Cmp { top1: f64, err_mean: f64, err_max: f32, bnd_mean: f64, int_mean: f64, flips_at_tie: usize, flips: usize }

fn teacher_cmp(e: &QwenEngine, f: &QwenEngine, ids: &[u32], t: usize) -> Cmp {
    let te = e.teacher(ids, 32).unwrap();
    let tf = f.teacher(ids, 32).unwrap();
    assert_eq!(te.len(), ids.len());
    assert_eq!(tf.len(), ids.len());
    let (mut agree, mut es, mut em, mut bs, mut bn, mut is_, mut inn, mut flips, mut tie) = (0, 0f64, 0f32, 0f64, 0, 0f64, 0, 0, 0);
    for p in 0..ids.len() {
        let err = logit_err(&te[p], &tf[p]);
        if te[p].idx[0] == tf[p].idx[0] { agree += 1; } else {
            flips += 1;
            if margin_ulp(&te[p]) <= NEAR_TIE_ULP { tie += 1; }
        }
        es += err as f64; em = em.max(err);
        if p % t < 4 { bs += err as f64; bn += 1; } else { is_ += err as f64; inn += 1; }
    }
    let n = ids.len() as f64;
    Cmp { top1: agree as f64 / n, err_mean: es / n, err_max: em, bnd_mean: bs / bn.max(1) as f64, int_mean: is_ / inn.max(1) as f64,
          flips_at_tie: tie, flips }
}

#[test]
fn teacher_fast_vs_exact_structure() {
    let (Some(e), Some(f), Some(f9)) = (open(true, 128), open(false, 128), open(false, 9)) else { return };
    for id in ["ctx1k", "readme4k"] {
        let ids = prompt(id);
        for (eng, t) in [(&f, 128usize), (&f9, 9)] {
            let c = teacher_cmp(&e, eng, &ids, t);
            println!("teacher {:8} ({} pos) prefill_t {:3}: top-1 fast == exact {:.2}% ({} flips, {} at exact margin <= {} ulp), logit err mean {:.3} max {:.1} ulp; boundary rows (p % T < 4) mean {:.3} vs interior {:.3} ulp",
                     id, ids.len(), t, 100.0 * c.top1, c.flips, c.flips_at_tie, NEAR_TIE_ULP, c.err_mean, c.err_max, c.bnd_mean, c.int_mean);
            assert!(c.bnd_mean <= 2.0 * c.int_mean + 0.5, "chunk-boundary rows have a larger error: hand-off bug?");
            assert!(c.err_max <= 16.0, "logit error {} ulp: beyond GEMM-order noise", c.err_max);
            assert!(c.top1 >= 0.9, "top-1 agreement {:.3}", c.top1);
        }
    }
    // lm_head row: fast prefill token / margin == the fast teacher's last row (same chunks, K-invariant K = 1 head)
    for id in ["short", "ctx1k", "readme4k"] {
        let ids = prompt(id);
        let tf = f.teacher(&ids, 32).unwrap();
        let g = f.generate(&ids, 1, false).unwrap();
        let last = tf.last().unwrap();
        println!("head row {:8}: prefill token {} margin {} | teacher last row top-1 {} margin {}", id, g.tokens[0], g.margins[0],
                 last.idx[0], last.val[0] - last.val[1]);
        assert_eq!(g.tokens[0], last.idx[0], "fast prefill head row != teacher last row");
        assert_eq!(g.margins[0], last.val[0] - last.val[1], "fast prefill margin != teacher last row");
    }
}

/// First token difference of a vs b, and the exact engine's margin (ulp) at that step of b's sequence.
fn first_diff_margin(e: &QwenEngine, prompt: &[u32], a: &[u32], b: &[u32]) -> Option<(usize, f32)> {
    let i = (0..a.len().min(b.len())).find(|&i| a[i] != b[i])?;
    let seq: Vec<u32> = prompt.iter().chain(b[..i].iter()).copied().collect();
    let t = e.teacher(&seq, 32).unwrap();
    Some((i, margin_ulp(t.last().unwrap())))
}

#[test]
fn chunk_boundaries() {
    let (Some(e), Some(f)) = (open(true, 128), open(false, 128)) else { return };
    let src = prompt("readme4k");
    let n_new = 24;
    let (mut n_ident, mut n_tie) = (0, 0);
    for n in [8usize, 9, 127, 128, 129, 300] {
        let p = &src[..n];
        let ge = e.generate(p, n_new, true).unwrap();
        let gf = f.generate(p, n_new, true).unwrap();
        let gs = f.generate(p, n_new, false).unwrap();
        assert_eq!(gf.tokens, gs.tokens, "fast prefill: pipelined != sync (n {})", n);
        let d = first_diff_margin(&e, p, &gf.tokens, &ge.tokens);
        println!("prompt len {:3}: fast {} exact ({} tokens){}", n, if d.is_none() { "==" } else { "!=" }, n_new,
                 d.map(|(i, m)| format!(", first diff @{}: exact margin {:.1} ulp (fast margin {:.4})", i, m, gf.margins[i])).unwrap_or_default());
        if n <= 8 {
            assert_eq!(gf.tokens, ge.tokens, "<= 8 rows must be bit-exact");
            assert_eq!(gf.margins, ge.margins, "<= 8 rows must be bit-exact (margins)");
        }
        match d { None => n_ident += 1, Some((_, m)) => { assert!(m <= NEAR_TIE_ULP, "token diff at a {:.1}-ulp margin (n {})", m, n); n_tie += 1; } }
    }
    println!("chunk boundaries: {} identical, {} differ at a near-tie", n_ident, n_tie);
}

/// The `qwen serve` gen logic: continue = feed [pending] + tokens on the kept state, else reset.
fn serve_session(eng: &QwenEngine, turns: &[Vec<u32>], n_new: usize) -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut pending: Option<u32> = None;
    let (mut outs, mut poss) = (vec![], vec![]);
    for (k, toks) in turns.iter().enumerate() {
        if k == 0 { eng.reset(); }
        let mut feed = vec![];
        if k > 0 { if let Some(p) = pending { feed.push(p); } }
        feed.extend_from_slice(toks);
        let mut got = vec![];
        let st = eng.feed_stream(&feed, n_new, &[], &mut |t| { got.push(t); true }).unwrap();
        pending = st.last;
        outs.push(got);
        poss.push(eng.pos());
    }
    (outs, poss)
}

#[test]
fn serve_continue_fast() {
    let (Some(e), Some(f)) = (open(true, 128), open(false, 128)) else { return };
    let src = prompt("ctx1k");
    // turn 1: 200 rows (fast chunk + tail), turn 2: 40-row suffix (fast pass), turn 3: 5-row suffix (exact tail only)
    let turns = vec![src[..200].to_vec(), src[200..240].to_vec(), src[240..245].to_vec()];
    let n_new = 16;
    for (name, eng) in [("exact", &e), ("fast", &f)] {
        let (outs, poss) = serve_session(eng, &turns, n_new);
        // full history for the fresh run of turn 3: turn1 + reply1 + turn2 + reply2 + turn3
        let hist: Vec<u32> = turns[0].iter().chain(&outs[0]).chain(&turns[1]).chain(&outs[1]).chain(&turns[2]).copied().collect();
        eng.reset();
        let mut fresh = vec![];
        let st = eng.feed_stream(&hist, n_new, &[], &mut |t| { fresh.push(t); true }).unwrap();
        let _ = st;
        assert_eq!(eng.pos(), poss[2], "{}: position after the continue session != fresh", name);
        let d = first_diff_margin(&e, &hist, &outs[2], &fresh);
        println!("serve continue ({}): positions {:?}, reply 3 {} fresh run on the {}-token history{}", name, poss,
                 if d.is_none() { "==" } else { "!=" }, hist.len(),
                 d.map(|(i, m)| format!(", first diff @{}: exact margin {:.1} ulp", i, m)).unwrap_or_default());
        if name == "exact" { assert!(d.is_none(), "exact prefill: continue != fresh (must be bit-identical)"); }
        else if let Some((_, m)) = d { assert!(m <= NEAR_TIE_ULP, "fast continue vs fresh: diff at a {:.1}-ulp margin", m); }
    }
    // fast vs exact session, reply by reply
    let (oe, pe) = serve_session(&e, &turns, n_new);
    let (of, pf) = serve_session(&f, &turns, n_new);
    // info only: once a reply differs (near-tie), the later turns see different histories
    for k in 0..3 {
        println!("  reply {}: fast {} exact (pos {} / {})", k + 1, if oe[k] == of[k] { "==" } else { "!=" }, pf[k], pe[k]);
    }
}
