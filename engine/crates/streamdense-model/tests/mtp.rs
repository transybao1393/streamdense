//! S2 self-test (P2 U19-U22): MTP speculative decode on the tiny model (models/tiny_qwen35, has mtp/).
//! - generate_mtp == plain greedy, bit-exact: real head (pipe + sync) and oracle drafts p in {0, 0.5, 0.9, 1}
//!   (pipe + sync); accepted and rejected passes both > 0; p = 1 never rejects (unless an eos is emitted).
//! - The same with the exact prefill (K <= 8 GEMV chunks) and across position 1024 (1-pass / 2-pass sdpa switch).
//! - serve emulation (same feed / pending logic as `qwen serve`): continue sessions mixing MTP and plain turns, with
//!   stop tokens, == an all-plain session in tokens, stop reasons and positions.
//! - Rollback: a reply that stops on an accepted draft d (length, stop token, cancel) leaves the engine where plain
//!   leaves it (pos, state): the next turn (MTP or plain) == plain. Uses `feed_stream_mtp_ov` (oracle p = 1).
//! - ctx_full: MTP stops at most 1 token earlier than plain, its tokens a prefix of plain's.
//! - Deep spec (spec_deep.rs, d = 2..7 drafts per K = d + 1 pass, gemv mma) with oracle drafts p in {0, 0.5, 0.8, 0.9, 1}: == plain,
//!   every replay length 1..d-1 and full accepts hit; sessions mixing deep and plain turns (incl. stops) == plain.
//! - Adaptive depth + pipelined loop (spec_deep.rs `SpecMode::Auto`, encoded-ahead verify / outcome / chain buffers):
//!   oracle p in {0, 0.5, 0.9, 1} == plain and == the sync loop at fixed depth; with a draft source (in-test stubs:
//!   noisy oracle table, empty table; and the real DraftTable in a temp dir) == plain, table drafts used; sessions
//!   (stops, cancel, mixed with plain / fixed-depth turns) == plain.
//! Run: cargo test --release --offline -p streamdense-model --test mtp -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_model::spec::{generate_mtp, DraftOverride, SpecResult};
use streamdense_model::spec_deep::{generate_spec_mode, generate_spec_ov, generate_spec_sync, DraftSource, SpecMode};
use streamdense_model::qwen_layers::GemvImpl;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}

fn open(max_ctx: usize, exact: bool) -> Option<QwenEngine> {
    let dir = model_dir()?;
    let opt = QwenOptions { max_ctx, mtp: true, prefill_exact: exact, ..Default::default() };
    Some(QwenEngine::open(&dir, opt).expect("open tiny"))
}

fn prompts() -> Vec<(String, Vec<u32>)> {
    let p = format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT);
    let mut v: Vec<(String, Vec<u32>)> = match std::fs::read_to_string(&p) {
        Ok(s) => {
            let j: Value = serde_json::from_str(&s).unwrap();
            j["prompts"].as_array().unwrap().iter().map(|r| (r["id"].as_str().unwrap().to_string(),
                r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect())).collect()
        }
        Err(_) => { eprintln!("note: {} missing, synthetic prompts only", p); vec![] }
    };
    // synthetic: 1 token; 300 tokens (fast prefill GEMM chunks + MTP warm-up over > 8-row chunks)
    let mut s = 0x1234_5678u64;
    let mut rnd = |n: usize| -> Vec<u32> { (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 2000) as u32 + 10 }).collect() };
    v.push(("syn1".into(), rnd(1)));
    v.push(("syn300".into(), rnd(300)));
    v
}
fn get<'a>(p: &'a [(String, Vec<u32>)], id: &str) -> Option<&'a Vec<u32>> { p.iter().find(|x| x.0 == id).map(|x| &x.1) }

fn has_eos(t: &[u32]) -> bool {
    let c = streamdense_model::QwenConfig::from_model_dir(&model_dir().unwrap()).unwrap();
    t.iter().any(|x| c.eos.contains(x))
}

/// plain sync + pipe (must agree), then every MTP run; returns (equal runs, runs, accepted, rejected)
fn compare(eng: &QwenEngine, id: &str, prompt: &[u32], n: usize, ps: &[f64]) -> (usize, usize, usize, usize) {
    let g = eng.generate(prompt, n, true).unwrap();
    let gs = eng.generate(prompt, n, false).unwrap();
    assert_eq!(g.tokens, gs.tokens, "{}: plain pipe != sync", id);
    let seq: Vec<u32> = prompt.iter().chain(g.tokens.iter()).copied().collect();
    let mut runs: Vec<(String, DraftOverride, bool)> = vec![("head pipe".into(), DraftOverride::None, true), ("head sync".into(), DraftOverride::None, false)];
    for &p in ps {
        for pipe in [true, false] {
            runs.push((format!("oracle {} {}", p, if pipe { "pipe" } else { "sync" }), DraftOverride::Oracle { seq: seq.clone(), p, seed: 7 }, pipe));
        }
    }
    let (mut eq_n, mut acc, mut rej) = (0, 0, 0);
    let eos_in = has_eos(&g.tokens);
    for (name, ov, pipe) in &runs {
        let m: SpecResult = generate_mtp(eng, prompt, g.tokens.len(), ov.clone(), *pipe).unwrap();
        let eq = m.tokens == g.tokens;
        let p1_ok = !(name.starts_with("oracle 1 ") && m.rejects > 0 && !eos_in);
        println!("  {:8} P={:4} n={:3} {:16}: == plain {} | passes {:3} acc {:3} rej {:3} ({:.3}, {:.2} tok/pass, {} overridden){}",
                 id, prompt.len(), g.tokens.len(), name, eq, m.passes, m.accepted, m.rejects, m.acceptance, m.tokens_per_pass,
                 m.overridden, if p1_ok { "" } else { "  <- p = 1 rejected" });
        if eq && p1_ok { eq_n += 1; }
        acc += m.accepted; rej += m.rejects;
    }
    (eq_n, runs.len(), acc, rej)
}

#[test]
fn mtp_equals_plain_head_and_oracle() {
    let Some(eng) = open(2048, false) else { return; };
    let ps = prompts();
    let (mut e, mut n, mut a, mut r) = (0, 0, 0, 0);
    for id in ["short", "code", "math", "chinese", "syn1", "syn300"] {
        let Some(p) = get(&ps, id) else { continue };
        let (e1, n1, a1, r1) = compare(&eng, id, p, 64, &[0.0, 0.5, 0.9, 1.0]);
        e += e1; n += n1; a += a1; r += r1;
    }
    println!("fast prefill: MTP == plain on {} / {} runs, accepted {} rejected {}", e, n, a, r);
    assert!(e == n && n > 0 && a > 0 && r > 0);
}

#[test]
fn mtp_equals_plain_exact_prefill() {
    let Some(eng) = open(2048, true) else { return; };
    let ps = prompts();
    let (mut e, mut n, mut a, mut r) = (0, 0, 0, 0);
    for id in ["code", "syn300"] {
        let Some(p) = get(&ps, id) else { continue };
        let (e1, n1, a1, r1) = compare(&eng, id, p, 40, &[0.5, 1.0]);
        e += e1; n += n1; a += a1; r += r1;
    }
    println!("exact prefill: MTP == plain on {} / {} runs, accepted {} rejected {}", e, n, a, r);
    assert!(e == n && n > 0 && a > 0 && r > 0);
}

#[test]
fn mtp_crosses_1024() {
    let Some(eng) = open(1280, false) else { return; };
    let ps = prompts();
    // ctx1k (973 tokens) + 120 new: decode crosses 1024; synthetic 1000 if the ref prompts are missing
    let p: Vec<u32> = get(&ps, "ctx1k").cloned().unwrap_or_else(|| (0..1000u32).map(|i| (i * 37) % 2000 + 10).collect());
    let (e, n, a, r) = compare(&eng, "ctx1k", &p, 120, &[0.5, 1.0]);
    println!("across 1024 (P = {} .. {}): MTP == plain on {} / {} runs, accepted {} rejected {}", p.len(), p.len() + 120, e, n, a, r);
    assert!(e == n && a > 0 && r > 0);
}

// ------------------------------------------------------------------------------------------------ serve emulation
#[derive(Clone)]
enum Mode { Plain, Mtp(DraftOverride), Deep(DraftOverride, usize), Spec(DraftOverride, SpecMode) }

#[derive(Debug, Clone, PartialEq)]
struct Turn { tokens: Vec<u32>, reason: String, pos: usize, accepted: usize, passes: usize }

/// One `qwen serve` gen op (same feed / pending / reset logic as the CLI).
fn turn(eng: &QwenEngine, pending: &mut Option<u32>, toks: &[u32], cont: bool, mode: &Mode, nnew: usize, stop: &[u32],
        cancel_at: Option<usize>) -> Turn {
    let cont = cont && eng.pos() > 0;
    let mut feed = Vec::new();
    if cont { if let Some(p) = *pending { feed.push(p); } }
    feed.extend_from_slice(toks);
    if !cont { eng.reset(); *pending = None; }
    let mut out = Vec::new();
    let mut on = |t: u32| -> bool { out.push(t); cancel_at.map_or(true, |c| out.len() < c) };
    let (st, spec) = match mode {
        Mode::Plain => (eng.feed_stream(&feed, nnew, stop, &mut on).unwrap(), None),
        Mode::Mtp(ov) => { let (s, m) = eng.feed_stream_mtp_ov(&feed, nnew, stop, &mut on, ov).unwrap(); (s, Some(m)) }
        Mode::Deep(ov, d) => { let (s, m) = eng.feed_stream_spec_ov(&feed, nnew, stop, &mut on, *d, ov).unwrap(); (s, Some(m)) }
        Mode::Spec(ov, md) => { let (s, m) = eng.feed_stream_spec_mode(&feed, nnew, stop, &mut on, *md, ov).unwrap(); (s, Some(m)) }
    };
    *pending = st.last;
    Turn { tokens: out, reason: st.stop_reason.to_string(), pos: eng.pos(),
           accepted: spec.as_ref().map_or(0, |m| m.accepted), passes: spec.as_ref().map_or(0, |m| m.passes) }
}

struct Script { toks: Vec<Vec<u32>>, nnew: Vec<usize>, stop: Vec<Vec<u32>>, cancel: Vec<Option<usize>> }

/// Runs a session; `modes[i]` per turn; turn 0 is fresh, later turns continue. Returns turns + the absolute sequence.
fn session(eng: &QwenEngine, s: &Script, modes: &[Mode]) -> (Vec<Turn>, Vec<u32>) {
    let mut pending = None;
    let mut seq: Vec<u32> = Vec::new();
    let mut v = Vec::new();
    for i in 0..s.toks.len() {
        let t = turn(eng, &mut pending, &s.toks[i], i > 0, &modes[i], s.nnew[i], &s.stop[i], s.cancel[i]);
        seq.extend_from_slice(&s.toks[i]);
        seq.extend_from_slice(&t.tokens);
        v.push(t);
    }
    (v, seq)
}

fn user_turns() -> Vec<Vec<u32>> {
    let ps = prompts();
    let mut v: Vec<Vec<u32>> = ["short", "code", "math", "chinese"].iter().filter_map(|id| get(&ps, id).cloned()).collect();
    if v.len() < 4 { v = (0..4).map(|k| (0..20u32).map(|i| (i * 53 + k * 101) % 2000 + 10).collect()).collect(); }
    v
}

#[test]
fn serve_sessions_mix_mtp_and_plain() {
    let Some(eng) = open(2048, false) else { return; };
    let u = user_turns();
    // probe (plain, no stops): pick a stop token for turn 1 = probe reply token 9 (first occurrence)
    let base = Script { toks: u.clone(), nnew: vec![40, 33, 40, 24], stop: vec![vec![]; 4], cancel: vec![None; 4] };
    let (probe, _) = session(&eng, &base, &vec![Mode::Plain; 4]);
    let x = probe[1].tokens[9];
    let s = Script { stop: vec![vec![], vec![x], vec![], vec![]], ..base };
    let (plain, seq) = session(&eng, &s, &vec![Mode::Plain; 4]);
    println!("  plain session: {:?}", plain.iter().map(|t| (t.tokens.len(), t.reason.clone(), t.pos)).collect::<Vec<_>>());
    let ok_stop = plain[1].reason == "eos" && plain[1].tokens.last() == Some(&x);
    let m_h = Mode::Mtp(DraftOverride::None);
    let m_o = |p: f64| Mode::Mtp(DraftOverride::Oracle { seq: seq.clone(), p, seed: 3 });
    let p = Mode::Plain;
    let pats: Vec<(&str, Vec<Mode>)> = vec![
        ("MMMM head", vec![m_h.clone(); 4]),
        ("MMMM o0.5", vec![m_o(0.5); 4]),
        ("MMMM o1", vec![m_o(1.0); 4]),
        ("PMPM o0.9", vec![p.clone(), m_o(0.9), p.clone(), m_o(0.9)]),
        ("MPMP o1", vec![m_o(1.0), p.clone(), m_o(1.0), p.clone()]),
        ("PPMM head", vec![p.clone(), p.clone(), m_h.clone(), m_h.clone()]),
        ("MPPM o0.5", vec![m_o(0.5), p.clone(), p.clone(), m_o(0.5)]),
    ];
    let mut ok = ok_stop;
    for (name, modes) in &pats {
        let (t, _) = session(&eng, &s, modes);
        let strip = |v: &[Turn]| v.iter().map(|x| (x.tokens.clone(), x.reason.clone(), x.pos)).collect::<Vec<_>>();
        let eq = strip(&t) == strip(&plain);
        let (acc, pas): (usize, usize) = t.iter().fold((0, 0), |a, x| (a.0 + x.accepted, a.1 + x.passes));
        println!("  {:10}: == plain session {} (tokens, stop reasons, positions) | MTP passes {} accepted {}", name, eq, pas, acc);
        if !eq {
            for (i, (a, b)) in t.iter().zip(&plain).enumerate() { if a != b { println!("    turn {}: {:?}\n      plain {:?}", i, a, b); } }
        }
        ok &= eq;
    }
    assert!(ok, "serve sessions: MTP/plain mix != plain (stop token used: {})", ok_stop);
}

#[test]
fn rollback_after_accepted_draft_stop() {
    let Some(eng) = open(2048, false) else { return; };
    let u = user_turns();
    let mut ok = true;
    // probe for a stop token sitting at an odd reply index (a draft slot when every draft is accepted)
    let base = Script { toks: u[..3].to_vec(), nnew: vec![30, 30, 30], stop: vec![vec![]; 3], cancel: vec![None; 3] };
    let (probe, _) = session(&eng, &base, &vec![Mode::Plain; 3]);
    let r0 = &probe[0].tokens;
    let stop_ix = (3..r0.len()).step_by(2).find(|&i| !r0[..i].contains(&r0[i])).expect("no usable stop token");
    let cases: Vec<(&str, Script)> = vec![
        ("length 8 (even)", Script { toks: base.toks.clone(), nnew: vec![8, 12, 10], stop: vec![vec![]; 3], cancel: vec![None; 3] }),
        ("stop token @odd", Script { toks: base.toks.clone(), nnew: vec![30, 30, 6], stop: vec![vec![r0[stop_ix]], vec![], vec![]], cancel: vec![None; 3] }),
        ("cancel @4 tokens", Script { toks: base.toks.clone(), nnew: vec![30, 30, 6], stop: vec![vec![]; 3], cancel: vec![Some(4), None, Some(6)] }),
    ];
    for (name, s) in &cases {
        let (plain, seq) = session(&eng, s, &vec![Mode::Plain; 3]);
        let o1 = Mode::Mtp(DraftOverride::Oracle { seq: seq.clone(), p: 1.0, seed: 1 });
        for (pn, modes) in [("M M M", vec![o1.clone(); 3]), ("M P M", vec![o1.clone(), Mode::Plain, o1.clone()]),
                            ("M M P", vec![o1.clone(), o1.clone(), Mode::Plain])] {
            let (t, _) = session(&eng, s, &modes);
            // turn 0 must have ended on an accepted draft: every pass accepted and an even number of tokens
            let on_d = t[0].passes > 0 && t[0].accepted == t[0].passes && t[0].tokens.len() % 2 == 0;
            let eq = t.iter().zip(&plain).all(|(a, b)| a.tokens == b.tokens && a.reason == b.reason && a.pos == b.pos);
            println!("  {:17} {}: turn 0 {} tok '{}' pos {} (plain pos {}), stopped on accepted d {} | session == plain {}",
                     name, pn, t[0].tokens.len(), t[0].reason, t[0].pos, plain[0].pos, on_d, eq);
            if !eq { for (i, (a, b)) in t.iter().zip(&plain).enumerate() { if a.tokens != b.tokens || a.pos != b.pos { println!("    turn {}: {:?}\n      plain {:?}", i, a, b); } } }
            ok &= eq && on_d;
        }
    }
    assert!(ok, "rollback after an accepted-draft stop");
}

#[test]
fn ctx_full_mtp_vs_plain() {
    let Some(eng) = open(96, false) else { return; };
    let u = user_turns();
    let p: Vec<u32> = u.iter().flatten().copied().take(70).collect();
    let mut pend = None;
    let a = turn(&eng, &mut pend, &p, false, &Mode::Plain, 500, &[], None);
    let mut ok = true;
    for ov in [DraftOverride::None, DraftOverride::Oracle { seq: p.iter().chain(&a.tokens).copied().collect(), p: 1.0, seed: 0 }] {
        let mut pend = None;
        let b = turn(&eng, &mut pend, &p, false, &Mode::Mtp(ov), 500, &[], None);
        let prefix = a.tokens.starts_with(&b.tokens) && a.tokens.len() - b.tokens.len() <= 1;
        println!("  max_ctx 96, P {}: plain {} tok '{}' pos {} | MTP {} tok '{}' pos {} (prefix, <= 1 shorter: {})",
                 p.len(), a.tokens.len(), a.reason, a.pos, b.tokens.len(), b.reason, b.pos, prefix);
        ok &= prefix && b.pos <= 96 && (a.reason != "ctx_full" || b.reason == "ctx_full");
    }
    assert!(ok);
}

#[test]
fn deep_spec_oracle_replay() {
    let Some(dir) = model_dir() else { return; };
    let eng = QwenEngine::open(&dir, QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, gemv: GemvImpl::Mma, ..Default::default() }).expect("open tiny");
    let ps = prompts();
    let mut ok = true;
    let mut seen = vec![vec![false; 8]; 8]; // seen[d][je]: replay of je rows (1..d-1) / full accept (je = d)
    for id in ["short", "code", "math", "syn300"] {
        let Some(p) = get(&ps, id) else { continue };
        let g = eng.generate(p, 64, false).unwrap();
        let seq: Vec<u32> = p.iter().chain(g.tokens.iter()).copied().collect();
        for d in 2..=7 {
            for (i, &pr) in [0.0, 0.5, 0.8, 0.9, 1.0].iter().enumerate() {
                let r = generate_spec_ov(&eng, p, g.tokens.len(), d, DraftOverride::Oracle { seq: seq.clone(), p: pr, seed: 11 + i as u64 + 10 * d as u64 }).unwrap();
                let eq = r.tokens == g.tokens;
                for &(je, _) in &r.replay { seen[d][je] = true; }
                if r.accept_hist.get(d).copied().unwrap_or(0) > 0 { seen[d][d] = true; }
                println!("  {:6} d={} oracle {:.1}: == plain {} | passes {:3} acc {:3} hist {:?} replays {}", id, d, pr, eq,
                         r.passes, r.accepted, r.accept_hist, r.replay.len());
                ok &= eq;
            }
        }
    }
    for d in 2..=7 { for je in 1..=d { if !seen[d][je] { println!("  d={}: {} never hit", d, if je == d { "full accept".to_string() } else { format!("replay of {} rows", je) }); ok = false; } } }
    // sessions: deep turns (stops included) leave the engine where plain does
    let u = user_turns();
    let base = Script { toks: u.clone(), nnew: vec![30, 25, 30, 20], stop: vec![vec![]; 4], cancel: vec![None; 4] };
    let (probe, _) = session(&eng, &base, &vec![Mode::Plain; 4]);
    let s = Script { stop: vec![vec![], vec![probe[1].tokens[7]], vec![], vec![]], cancel: vec![None, None, Some(5), None], ..base };
    let (plain, seq) = session(&eng, &s, &vec![Mode::Plain; 4]);
    for d in 2..=7 {
        let o = |p: f64| Mode::Deep(DraftOverride::Oracle { seq: seq.clone(), p, seed: 5 }, d);
        for (name, modes) in [("DDDD o0.7", vec![o(0.7); 4]), ("DPDP o1", vec![o(1.0), Mode::Plain, o(1.0), Mode::Plain]),
                              ("PDMD o0.5", vec![Mode::Plain, o(0.5), Mode::Mtp(DraftOverride::Oracle { seq: seq.clone(), p: 0.5, seed: 5 }), o(0.5)])] {
            let (t, _) = session(&eng, &s, &modes);
            let eq = t.iter().zip(&plain).all(|(a, b)| a.tokens == b.tokens && a.reason == b.reason && a.pos == b.pos);
            println!("  session d={} {}: == plain {}", d, name, eq);
            if !eq { for (i, (a, b)) in t.iter().zip(&plain).enumerate() { if a != b { println!("    turn {}: {:?}\n      plain {:?}", i, a, b); } } }
            ok &= eq;
        }
    }
    assert!(ok, "deep spec != plain or replay lengths not covered");
}

/// The previous matmul family (`--gemv old`, q4b_gemv k1..k8) through the same deep verify: d = 2..7 == plain greedy
/// of the same engine (K-invariant within the family; K = 6, 7 kernels included).
#[test]
fn deep_spec_gemv_old() {
    let Some(dir) = model_dir() else { return; };
    let eng = QwenEngine::open(&dir, QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, gemv: GemvImpl::Old, ..Default::default() }).expect("open tiny");
    let ps = prompts();
    let mut ok = true;
    for id in ["short", "code"] {
        let Some(p) = get(&ps, id) else { continue };
        let g = eng.generate(p, 48, false).unwrap();
        let seq: Vec<u32> = p.iter().chain(g.tokens.iter()).copied().collect();
        for d in 2..=7 {
            for pr in [0.5, 0.9] {
                let r = generate_spec_ov(&eng, p, g.tokens.len(), d, DraftOverride::Oracle { seq: seq.clone(), p: pr, seed: 3 + d as u64 }).unwrap();
                let eq = r.tokens == g.tokens;
                println!("  old {:6} d={} oracle {:.1}: == plain {} | passes {:3} hist {:?}", id, d, pr, eq, r.passes, r.accept_hist);
                ok &= eq;
            }
        }
    }
    assert!(ok, "gemv old: deep spec != plain");
}

/// Pruned draft vocab (`QwenOptions.draft_vocab`, head_pruned.metal), both matmul families, keep-list = ~60% of the
/// ids (pseudo-random, so real drafts often want a removed id): (1) the pruned head's logits are bit-identical to the
/// full head's kept rows and its draft == the full argmax restricted to kept rows (bf16-rounded, lowest id), on random
/// rows and on real MTP rows; (2) MTP d = 1 (pipe + sync) and deep spec d = 2..7 with the pruned draft head == plain.
#[test]
fn draft_vocab_pruned_head() {
    let Some(dir) = model_dir() else { return; };
    let vocab = streamdense_model::QwenConfig::from_model_dir(&dir).unwrap().vocab;
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let ids: Vec<u32> = (0..vocab as u32).filter(|_| rnd() % 10 < 6).collect();
    let path = std::env::temp_dir().join(format!("sd_tiny_draft_vocab_{}.keep", std::process::id()));
    streamdense_model::draft_vocab::KeepList::new(vocab as u32, ids.clone()).unwrap().write(&path).unwrap();
    let to_bf = |f: f32| -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 };
    let bfr = |f: f32| f32::from_bits((to_bf(f) as u32) << 16);
    let check = |pr: &streamdense_model::mtp::DraftProbe, tag: &str| -> bool {
        let nbad = pr.ids.iter().zip(&pr.kept).filter(|(&t, k)| pr.full[t as usize].to_bits() != k.to_bits()).count();
        let argmax = |it: &mut dyn Iterator<Item = u32>| -> u32 {
            let mut b = (f32::NEG_INFINITY, u32::MAX);
            for t in it { let v = bfr(pr.full[t as usize]); if b.1 == u32::MAX || v > b.0 { b = (v, t); } }
            b.1
        };
        let (rf, rk) = (argmax(&mut (0..pr.full.len() as u32)), argmax(&mut pr.ids.iter().copied()));
        let ok = nbad == 0 && pr.ids == ids && pr.draft_kept == rk && pr.draft_full == rf;
        println!("  {}: kept logits bit-identical {} ({} / {} differ) | draft pruned {} == restricted argmax {} | full {} == {} | {}",
                 tag, nbad == 0, nbad, pr.ids.len(), pr.draft_kept, rk, pr.draft_full, rf, if ok { "ok" } else { "FAIL" });
        ok
    };
    let ps = prompts();
    let mut ok = true;
    for gemv in [GemvImpl::Mma, GemvImpl::Old] {
        let opt = QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, gemv, draft_vocab: Some(path.clone()), ..Default::default() };
        let eng = QwenEngine::open(&dir, opt).expect("open tiny");
        println!("gemv {}: keep {} / {} ids", gemv.name(), ids.len(), vocab);
        let h = eng.cfg.hidden;
        for r in 0..4 {
            let row: Vec<u16> = (0..h).map(|_| to_bf(((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0 * (1 + r) as f32)).collect();
            ok &= check(&eng.draft_head_probe(Some(&row)).unwrap(), &format!("random row {}", r));
        }
        for id in ["short", "code", "syn300"] {
            let Some(p) = get(&ps, id) else { continue };
            eng.reset();
            let (t1, hf) = eng.prefill_mtp(p).unwrap();
            let d = eng.mtp_first_draft(t1, hf, 0).unwrap();
            let pr = eng.draft_head_probe(None).unwrap();
            ok &= check(&pr, &format!("real MTP row ({})", id)) && d == pr.draft_kept;
        }
        for id in ["short", "code", "math", "syn300"] {
            let Some(p) = get(&ps, id) else { continue };
            eng.reset();
            let g = eng.generate(p, 48, false).unwrap();
            for pipe in [true, false] {
                eng.reset();
                let r = generate_mtp(&eng, p, g.tokens.len(), DraftOverride::None, pipe).unwrap();
                let eq = r.tokens == g.tokens;
                println!("  {} {:6} mtp d=1 {}: == plain {} | passes {:3} acc {:3} rej {:3}", gemv.name(), id, if pipe { "pipe" } else { "sync" },
                         eq, r.passes, r.accepted, r.rejects);
                ok &= eq;
            }
            for d in 2..=7 {
                eng.reset();
                let r = streamdense_model::spec_deep::generate_spec(&eng, p, g.tokens.len(), d).unwrap();
                let eq = r.tokens == g.tokens;
                println!("  {} {:6} deep d={}: == plain {} | passes {:3} hist {:?}", gemv.name(), id, d, eq, r.passes, r.accept_hist);
                ok &= eq;
            }
        }
    }
    let _ = std::fs::remove_file(&path);
    assert!(ok, "pruned draft vocab: head not bit-identical or spec != plain");
}

// ------------------------------------------------------------------------------------------- adaptive + pipelined
/// Test draft source: proposes the true continuation (session sequence `seq`), each token replaced by a wrong one
/// with probability `miss` (deterministic). Tracks the absolute position through begin_sequence / observe.
struct OracleTable { seq: Vec<u32>, pos: usize, miss: f64, s: u64, max: usize, proposed: usize }
impl OracleTable {
    fn new(seq: &[u32], miss: f64, seed: u64, max: usize) -> Self { OracleTable { seq: seq.to_vec(), pos: 0, miss, s: seed, max, proposed: 0 } }
    fn rnd(&mut self) -> f64 { self.s ^= self.s << 13; self.s ^= self.s >> 7; self.s ^= self.s << 17; (self.s >> 11) as f64 / (1u64 << 53) as f64 }
}
impl DraftSource for OracleTable {
    fn propose(&mut self, _tail: &[u32], max: usize) -> Vec<(u32, f32)> {
        let mut v = vec![];
        let mut c = 1.0f32;
        for i in 0..max.min(self.max) {
            let Some(&t) = self.seq.get(self.pos + i) else { break };
            let t = if self.rnd() < self.miss { t.wrapping_add(17) % 2000 + 10 } else { t };
            c *= 0.9;
            v.push((t, c));
        }
        self.proposed += v.len();
        v
    }
    fn observe(&mut self, _tail: &[u32], accepted: &[u32], top3: &[[u32; 3]]) {
        assert!(top3.len() == accepted.len() + 1 || top3.len() == accepted.len(), "top3 rows {} for {} accepted", top3.len(), accepted.len());
        self.pos += accepted.len() + (top3.len() > accepted.len()) as usize;
    }
    fn begin_sequence(&mut self, prompt: &[u32]) { self.pos = prompt.len(); }
    fn flush(&mut self) {}
}
/// Proposes nothing (the planner must fall back to MTP).
struct EmptyTable;
impl DraftSource for EmptyTable {
    fn propose(&mut self, _: &[u32], _: usize) -> Vec<(u32, f32)> { vec![] }
    fn observe(&mut self, _: &[u32], _: &[u32], _: &[[u32; 3]]) {}
    fn begin_sequence(&mut self, _: &[u32]) {}
    fn flush(&mut self) {}
}

fn open_auto(dir: &str) -> QwenEngine {
    QwenEngine::open(dir, QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, spec_auto: true, ..Default::default() }).expect("open tiny")
}

#[test]
fn adaptive_pipelined_oracle() {
    let Some(dir) = model_dir() else { return; };
    let eng = open_auto(&dir);
    assert_eq!(eng.spec_mode(), SpecMode::Auto { max: 7 });
    let ps = prompts();
    let mut ok = true;
    let mut dh = vec![0usize; 8];
    let (mut pre, mut reenc, mut pads) = (0, 0, 0);
    for id in ["short", "code", "math", "chinese", "syn300"] {
        let Some(p) = get(&ps, id) else { continue };
        let g = eng.generate(p, 64, false).unwrap();
        let seq: Vec<u32> = p.iter().chain(g.tokens.iter()).copied().collect();
        for (i, &pr) in [0.0, 0.5, 0.9, 1.0].iter().enumerate() {
            let ov = DraftOverride::Oracle { seq: seq.clone(), p: pr, seed: 21 + i as u64 };
            let r = generate_spec_mode(&eng, p, g.tokens.len(), SpecMode::Auto { max: 7 }, ov.clone()).unwrap();
            let eq = r.tokens == g.tokens;
            for (d, c) in r.depth_hist.iter().enumerate() { dh[d] += c; }
            pre += r.preencoded; reenc += r.reencode_ms.len(); pads += r.drafts_pad;
            let idle = r.idle_ms.iter().sum::<f64>() / r.idle_ms.len().max(1) as f64;
            println!("  {:7} auto oracle {:.1}: == plain {} | passes {:3} acc {:3} {:.2} tok/pass depth {:?} pad {} pre {} reenc {} idle {:.3} ms/round",
                     id, pr, eq, r.passes, r.accepted, r.tokens_per_pass, r.depth_hist, r.drafts_pad, r.preencoded, r.reencode_ms.len(), idle);
            ok &= eq;
            // fixed depth: pipelined == sync reference == plain
            for d in [2, 5, 7] {
                let a = generate_spec_mode(&eng, p, g.tokens.len(), SpecMode::Fixed(d), ov.clone()).unwrap();
                let b = generate_spec_sync(&eng, p, g.tokens.len(), d, ov.clone()).unwrap();
                let e2 = a.tokens == g.tokens && b.tokens == g.tokens && a.accept_hist == b.accept_hist;
                if !e2 { println!("    fixed d={}: pipe == plain {} sync == plain {} hist {:?} vs {:?}", d, a.tokens == g.tokens, b.tokens == g.tokens, a.accept_hist, b.accept_hist); }
                ok &= e2;
            }
        }
    }
    println!("  auto depth histogram (draft rows per pass) {:?}, verify encoded ahead {} / re-encoded {}, pad rows {}", dh, pre, reenc, pads);
    assert!(dh.iter().filter(|&&c| c > 0).count() >= 3, "auto used fewer than 3 depths: {:?}", dh);
    assert!(pre > 0, "no encoded-ahead verify was used");
    assert!(ok, "adaptive / pipelined spec != plain");
}

#[test]
fn adaptive_table_drafts() {
    let Some(dir) = model_dir() else { return; };
    let eng = open_auto(&dir);
    let ps = prompts();
    let mut ok = true;
    let mut table_acc = 0;
    for id in ["short", "code", "math", "syn300"] {
        let Some(p) = get(&ps, id) else { continue };
        let g = eng.generate(p, 64, false).unwrap();
        let seq: Vec<u32> = p.iter().chain(g.tokens.iter()).copied().collect();
        for (name, miss) in [("table exact", 0.0), ("table 30% miss", 0.3), ("table 90% miss", 0.9), ("empty", -1.0)] {
            if miss < 0.0 { eng.set_draft_source(Some(Box::new(EmptyTable))); }
            else { eng.set_draft_source(Some(Box::new(OracleTable::new(&seq, miss, 99, 8)))); }
            for (k, ov) in [DraftOverride::None, DraftOverride::Oracle { seq: seq.clone(), p: 0.5, seed: 4 }].into_iter().enumerate() {
                let r = generate_spec_mode(&eng, p, g.tokens.len(), SpecMode::Auto { max: 7 }, ov).unwrap();
                let eq = r.tokens == g.tokens;
                println!("  {:7} {:15} {}: == plain {} | passes {:3} {:.2} tok/pass drafts mtp {} (acc {}) table {} (acc {}) pad {} depth {:?}",
                         id, name, if k == 0 { "head  " } else { "o0.5  " }, eq, r.passes, r.tokens_per_pass, r.drafts_mtp, r.acc_mtp,
                         r.drafts_table, r.acc_table, r.drafts_pad, r.depth_hist);
                if miss == 0.0 && k == 0 { table_acc += r.acc_table; }
                if miss < 0.0 { ok &= r.drafts_table == 0; }
                ok &= eq;
            }
        }
    }
    eng.set_draft_source(None);
    // the real learning table (temp dir): two passes over the prompts, the second may draft from history
    let td = std::env::temp_dir().join(format!("sd_draft_table_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&td);
    let t = streamdense_model::draft_table::DraftTable::open(&td, 4 << 20).expect("draft table");
    eng.set_draft_source(Some(Box::new(t)));
    let mut real_table = 0;
    for round in 0..2 {
        for id in ["short", "code", "math"] {
            let Some(p) = get(&ps, id) else { continue };
            let g = eng.generate(p, 48, false).unwrap();
            let r = generate_spec_mode(&eng, p, g.tokens.len(), SpecMode::Auto { max: 7 }, DraftOverride::None).unwrap();
            let eq = r.tokens == g.tokens;
            println!("  real table round {} {:6}: == plain {} | {:.2} tok/pass table drafts {} (acc {}) mtp {} (acc {})", round, id, eq,
                     r.tokens_per_pass, r.drafts_table, r.acc_table, r.drafts_mtp, r.acc_mtp);
            real_table += r.drafts_table;
            ok &= eq;
        }
    }
    eng.set_draft_source(None);
    let _ = std::fs::remove_dir_all(&td);
    println!("  exact-table accepted table drafts {}, real-table drafts {}", table_acc, real_table);
    assert!(table_acc > 0, "an exact table was never used");
    assert!(ok, "table-drafted spec != plain");
}

#[test]
fn adaptive_sessions() {
    let Some(dir) = model_dir() else { return; };
    let eng = open_auto(&dir);
    let u = user_turns();
    let base = Script { toks: u.clone(), nnew: vec![30, 25, 30, 20], stop: vec![vec![]; 4], cancel: vec![None; 4] };
    let (probe, _) = session(&eng, &base, &vec![Mode::Plain; 4]);
    let s = Script { stop: vec![vec![], vec![probe[1].tokens[7]], vec![], vec![]], cancel: vec![None, None, Some(5), None], ..base };
    let (plain, seq) = session(&eng, &s, &vec![Mode::Plain; 4]);
    let a = |p: f64| Mode::Spec(DraftOverride::Oracle { seq: seq.clone(), p, seed: 8 }, SpecMode::Auto { max: 7 });
    let mut ok = true;
    for (name, modes, table) in [
        ("AAAA head", vec![Mode::Spec(DraftOverride::None, SpecMode::Auto { max: 7 }); 4], false),
        ("AAAA o0.7", vec![a(0.7); 4], false),
        ("APAD o1", vec![a(1.0), Mode::Plain, a(1.0), Mode::Deep(DraftOverride::Oracle { seq: seq.clone(), p: 0.5, seed: 8 }, 4)], false),
        ("AAAA table", vec![a(0.0); 4], true),
    ] {
        if table { eng.set_draft_source(Some(Box::new(OracleTable::new(&seq, 0.2, 5, 8)))); }
        let (t, _) = session(&eng, &s, &modes);
        eng.set_draft_source(None);
        let eq = t.iter().zip(&plain).all(|(a, b)| a.tokens == b.tokens && a.reason == b.reason && a.pos == b.pos);
        println!("  session {}: == plain {}", name, eq);
        if !eq { for (i, (x, y)) in t.iter().zip(&plain).enumerate() { if x != y { println!("    turn {}: {:?}\n      plain {:?}", i, x, y); } } }
        ok &= eq;
    }
    assert!(ok, "adaptive sessions != plain");
}

/// DraftOverride::External (oracle-drafter bench hook): drafts from a scripted sequence (true greedy tokens with wrong
/// tokens injected at chosen positions), chained MTP steps skipped. Must equal plain greedy at every depth, and the
/// accepted drafts per pass must follow the script exactly (the verify is exact, every other draft is correct).
#[test]
fn deep_spec_external_drafts() {
    let Some(dir) = model_dir() else { return; };
    let eng = QwenEngine::open(&dir, QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, gemv: GemvImpl::Mma, ..Default::default() }).expect("open tiny");
    let ps = prompts();
    let mut ok = true;
    for id in ["short", "code", "math"] {
        let Some(p) = get(&ps, id) else { continue };
        let g = eng.generate(p, 64, false).unwrap();
        let truth: Vec<u32> = p.iter().chain(g.tokens.iter()).copied().collect();
        for d in 2..=7usize {
            for want in [0usize, 1, d / 2, d] {
                // pass schedule: each pass accepts `want` drafts (wrong token at p + 1 + want), full passes advance d + 1
                let mut seq = truth.clone();
                let mut pos = p.len();
                while pos + 1 + want < seq.len() {
                    if want < d { let q = pos + 1 + want; seq[q] = if seq[q] != 13 { 13 } else { 11 }; pos += want + 1; } else { pos += d + 1; }
                }
                let r = generate_spec_mode(&eng, p, g.tokens.len(), SpecMode::Fixed(d), DraftOverride::External { seq }).unwrap();
                let eq = r.tokens == g.tokens;
                // every pass but possibly the last (cut by the length / eos) accepts exactly `want`
                let full = r.accept_hist.iter().enumerate().filter(|&(j, _)| j != want).map(|(_, c)| c).sum::<usize>() <= 1;
                println!("  {:6} d={} want {}: == plain {} | passes {:3} acc {:3} hist {:?} mtp-drafts {}", id, d, want, eq, r.passes, r.accepted,
                         r.accept_hist, r.drafts_mtp);
                ok &= eq && (full || has_eos(&g.tokens));
            }
        }
    }
    assert!(ok, "External drafts: output != plain or acceptance off-script");
}
