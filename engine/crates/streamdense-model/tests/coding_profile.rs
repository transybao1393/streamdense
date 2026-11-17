//! The "coding" profile's engine options (streamdense-core config.rs `CODING_PROFILE`) on the tiny model
//! (models/tiny_qwen35): q8 KV, MTP auto depth + context drafts (mtp_ctx_draft 7), fused verify matmuls (the
//! profile's decode speed-up: every check pass of a coding session has 3..8 rows), 4 rewind checkpoints.
//! - MTP auto + context drafts with fused verify == plain greedy of the same engine, tokens and recorded per-position
//!   top-1 / top-2 logits bit for bit, on copy-heavy prompts (a "file" read back = write / edit tool arguments that
//!   copy context text), real head and oracle drafts; lookup drafts verified and accepted.
//! - The same replies == an engine without fused verify (the default options), so the profile changes speed only.
//! Run: cargo test --release -p streamdense-model --test coding_profile -- --nocapture --test-threads 1
use streamdense_model::spec::DraftOverride;
use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
use streamdense_model::{KvFormat, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}

/// The profile's engine options (the gateway maps [gateway] max_ctx / kv / mtp / spec_depth / mtp_ctx_draft /
/// verify_fused / rewind_gb to these); `vf` = verify_fused.
fn coding_opts(dir: &str, vf: bool) -> QwenOptions {
    let cfg = streamdense_model::QwenConfig::from_model_dir(dir).unwrap();
    let per = streamdense_model::rewind::ckpt_bytes(&cfg) as u64;
    QwenOptions { max_ctx: 2048, kv: KvFormat::Q8, mtp: true, spec_depth: 7, spec_auto: true, mtp_ctx_draft: 7, verify_fused: vf,
                  rewind_bytes: 4 * per, ..Default::default() }
}

#[test]
fn coding_profile_equals_plain() {
    let Some(dir) = model_dir() else { return; };
    let eng = QwenEngine::open(&dir, coding_opts(&dir, true)).expect("open tiny (coding profile)");
    let base = QwenEngine::open(&dir, coding_opts(&dir, false)).expect("open tiny (no fused verify)");
    assert_eq!(eng.ckpt_slots(), 4);
    let chunk = |n: usize, k: u32| -> Vec<u32> { (0..n as u32).map(|i| (i * 71 + k * 13) % 1500 + 20).collect() };
    // "file" = a chunk; the reply-to-be copies it: file + plain reply + file (a write of text already in context),
    // and file + reply + the file's second half (an edit that copies the tail)
    let copy = |x: &[u32], n: usize, tail: bool| -> Vec<u32> {
        let g = eng.generate(x, n, false).unwrap().tokens;
        let back: &[u32] = if tail { &x[x.len() / 2..] } else { x };
        x.iter().chain(g.iter()).chain(back.iter()).copied().collect()
    };
    let cases: Vec<(&str, Vec<u32>, usize)> = vec![
        ("write 60", copy(&chunk(60, 1), 64, false), 96),
        ("edit 200", copy(&chunk(200, 2), 48, true), 80),
        ("write 480", copy(&chunk(480, 3), 48, false), 64), // > 1000 prompt tokens: decode crosses position 1024
        ("fresh 300", chunk(300, 9), 64),
    ];
    let rec = |e: &QwenEngine, p: usize, n: usize| -> Vec<(u32, u32, u32)> {
        (p - 1..p - 1 + n).map(|i| { let (t, a, b) = e.rec_token(i); (t, a.to_bits(), b.to_bits()) }).collect()
    };
    let (mut ok, mut drafts, mut acc) = (true, 0usize, 0usize);
    for (id, p, n) in &cases {
        let g = eng.generate(p, *n, false).unwrap();
        let gr = rec(&eng, p.len(), g.tokens.len());
        // plain decode passes have 1 row (fused verify applies to 3..8): the default engine gives the same reply
        let gb = base.generate(p, *n, false).unwrap();
        let same_base = gb.tokens == g.tokens && rec(&base, p.len(), gb.tokens.len()) == gr;
        let seq: Vec<u32> = p.iter().chain(g.tokens.iter()).copied().collect();
        for (name, ov) in [("head", DraftOverride::None), ("o0.7", DraftOverride::Oracle { seq: seq.clone(), p: 0.7, seed: 11 })] {
            let r = generate_spec_mode(&eng, p, g.tokens.len(), SpecMode::Auto { max: 7 }, ov.clone()).unwrap();
            let eq = r.tokens == g.tokens && rec(&eng, p.len(), r.tokens.len()) == gr;
            // the same spec run without fused verify: same tokens (only the matmul kernel differs, bit-identical)
            let rb = generate_spec_mode(&base, p, g.tokens.len(), SpecMode::Auto { max: 7 }, ov).unwrap();
            let eqb = rb.tokens == r.tokens && rec(&base, p.len(), rb.tokens.len()) == gr;
            println!("  {:9} {} (P {}, {} new): == plain {} | == no-vf plain {} spec {} | passes {:3} {:.2} tok/pass lookup/table {} (acc {}) mtp {} (acc {})",
                     id, name, p.len(), g.tokens.len(), eq, same_base, eqb, r.passes, r.tokens_per_pass, r.drafts_table, r.acc_table, r.drafts_mtp, r.acc_mtp);
            ok &= eq && same_base && eqb;
            drafts += r.drafts_table;
            acc += r.acc_table;
        }
    }
    println!("coding profile: lookup/table drafts {} accepted {}", drafts, acc);
    assert!(ok, "coding profile options: output != plain greedy");
    assert!(drafts > 0 && acc > 0, "lookup drafts never verified / accepted ({} / {})", drafts, acc);
}
