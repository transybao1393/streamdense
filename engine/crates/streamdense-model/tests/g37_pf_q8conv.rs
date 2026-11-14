//! q8-conversion prototype check (bench-only switch SD_PF_Q8CONV=1, not an engine feature): fast bf16 prefill + in-place KV
//! conversion to q8, then q8 decode, on the tiny model (models/tiny_qwen35).
//! - prompt <= 8 tokens: the fast path runs only exact passes, so its bf16 KV is the exact KV and the CPU conversion
//!   must reproduce `attn_prep_q8` bit for bit: tokens and recorded top-1 / top-2 logits == the exact q8 engine.
//! - ctx1k / readme4k: report decode-token agreement vs bf16 fast and q8 exact (fast prefill is not bit-exact).
//! Run: cargo test --release --offline -p streamdense-model --test g37_pf_q8conv -- --nocapture --test-threads 1
use streamdense_model::{KvFormat, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn open(kv: KvFormat, exact: bool, conv: bool) -> Option<QwenEngine> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", d)).exists() { eprintln!("SKIP: {} missing", d); return None; }
    if conv { std::env::set_var("SD_PF_Q8CONV", "1"); } else { std::env::remove_var("SD_PF_Q8CONV"); }
    let e = QwenEngine::open(&d, QwenOptions { max_ctx: 4096, kv, prefill_exact: exact, ..Default::default() }).expect("open tiny");
    std::env::remove_var("SD_PF_Q8CONV");
    Some(e)
}

fn prompt(id: &str) -> Vec<u32> {
    let s = std::fs::read_to_string(format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT)).expect("tiny prompts");
    let j: serde_json::Value = serde_json::from_str(&s).unwrap();
    j["prompts"].as_array().unwrap().iter().find(|r| r["id"] == id).map(|r| r["ids"].as_array().unwrap().iter()
        .map(|x| x.as_u64().unwrap() as u32).collect()).unwrap()
}

/// Greedy run from a fresh sequence: (tokens, rec (tok, top1 bits, top2 bits) for the decoded positions).
fn run(e: &QwenEngine, ids: &[u32], n: usize) -> (Vec<u32>, Vec<(u32, u32, u32)>) {
    e.reset();
    let mut out = vec![];
    e.feed_stream(ids, n, &[], &mut |t| { out.push(t); true }).unwrap();
    let rec = (ids.len() - 1..ids.len() - 1 + out.len()).map(|p| { let (t, a, b) = e.rec_token(p); (t, a.to_bits(), b.to_bits()) }).collect();
    (out, rec)
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> { a.iter().zip(b).position(|(x, y)| x != y) }

#[test]
fn g37_pf_q8conv() {
    let (Some(conv), Some(q8x), Some(bff)) = (open(KvFormat::Q8, false, true), open(KvFormat::Q8, true, false), open(KvFormat::Bf16, false, false))
        else { return; };
    let n = 64;
    // <= 8-token prompts: bit-identical to the exact q8 engine
    for id in ["short", "code", "math", "chinese"] {
        let ids = &prompt(id)[..8];
        let (tc, rc) = run(&conv, ids, n);
        let (tx, rx) = run(&q8x, ids, n);
        assert_eq!(tc, tx, "{}: tokens differ (8-token prompt)", id);
        assert_eq!(rc, rx, "{}: recorded logits differ (8-token prompt)", id);
        eprintln!("{} (8 tokens): {} decode tokens + top-1/top-2 logits bit-identical to q8 exact", id, tc.len());
    }
    for id in ["short", "code", "math", "chinese", "ctx1k", "readme4k"] {
        let ids = prompt(id);
        let (tc, _) = run(&conv, &ids, n);
        let (tx, _) = run(&q8x, &ids, n);
        let (tb, _) = run(&bff, &ids, n);
        eprintln!("{} ({} tokens): first diff vs q8 exact {:?}, vs bf16 fast {:?}; q8 exact vs bf16 fast {:?}", id, ids.len(),
                  first_diff(&tc, &tx), first_diff(&tc, &tb), first_diff(&tx, &tb));
    }
}
