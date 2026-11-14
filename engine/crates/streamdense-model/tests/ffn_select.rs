//! FFN-select test hook (ffn_select.rs) on the tiny model (models/tiny_qwen35), Metal: hook on in mode All == hook off bit
//! for bit (greedy tokens and teacher top-k); Oracle with K = inter == All; Oracle with a small K changes the output
//! and keeps exactly K neurons per layer on the recorded row.
use streamdense_model::ffn_select::FselMode;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn open(fsel: bool) -> Option<QwenEngine> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    let opt = QwenOptions { max_ctx: 1024, prefill_exact: true, ffn_select: fsel, ..Default::default() };
    Some(QwenEngine::open(&d, opt).expect("open tiny"))
}

#[test]
fn hook_all_is_bit_identical_and_oracle_keeps_k() {
    let (Some(off), Some(on)) = (open(false), open(true)) else { return };
    let prompt: Vec<u32> = vec![760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];
    let gen = |e: &QwenEngine| { e.reset(); let mut t = vec![]; e.feed_stream(&prompt, 24, &[], &mut |x| { t.push(x); true }).unwrap(); t };
    let base = gen(&off);
    let (l, n) = (on.cfg.layers, on.cfg.inter);
    on.fsel_set(FselMode::All, 0, &vec![n as u32; l]).unwrap();
    assert_eq!(gen(&on), base, "mode All != hook off");
    let ta = off.teacher(&prompt, 8).unwrap();
    let tb = on.teacher(&prompt, 8).unwrap();
    for (a, b) in ta.iter().zip(&tb) { assert_eq!((&a.idx, &a.val, a.lse.to_bits()), (&b.idx, &b.val, b.lse.to_bits())); }
    on.fsel_set(FselMode::Oracle, 0, &vec![n as u32; l]).unwrap();
    assert_eq!(gen(&on), base, "Oracle K = inter != hook off");
    on.fsel_dn().unwrap().fill(1.0);
    let k = n / 8;
    on.fsel_set(FselMode::Oracle, 0, &vec![k as u32; l]).unwrap();
    let small = gen(&on);
    let c = on.fsel_cov().unwrap();
    for li in 0..l { assert!(c[li * 4 + 2] as usize >= k && (c[li * 4 + 2] as usize) <= k + 2, "layer {} kept {}", li, c[li * 4 + 2]); }
    assert_eq!(on.fsel_sel().unwrap().iter().map(|&x| x as usize).sum::<usize>() as f32, (0..l).map(|li| c[li * 4 + 2]).sum::<f32>());
    eprintln!("oracle K = {} of {}: {} / {} tokens equal base", k, n, small.iter().zip(&base).filter(|(a, b)| a == b).count(), base.len());
    on.fsel_set(FselMode::Gate, 0, &vec![k as u32; l]).unwrap();
    let _ = gen(&on);
    // Fetch with K = 0 == Static with the same mask; Fetch with K > 0 fetches exactly K per layer
    for (j, x) in on.fsel_mask().unwrap().iter_mut().enumerate() { *x = (j % 3 == 0) as u8; }
    on.fsel_set(FselMode::Static, 0, &vec![0; l]).unwrap();
    let st = gen(&on);
    on.fsel_set(FselMode::Fetch, 0, &vec![0; l]).unwrap();
    assert_eq!(gen(&on), st, "Fetch K = 0 != Static");
    on.fsel_set(FselMode::Fetch, 0, &vec![16; l]).unwrap();
    let _ = gen(&on);
    let c = on.fsel_cov().unwrap();
    for li in 0..l { assert!(c[li * 4 + 3] >= 16.0 && c[li * 4 + 3] <= 18.0, "layer {} fetched {}", li, c[li * 4 + 3]); }
    // GateIn with K = inter == Static with the same mask; with a small K it keeps ~K neurons, all inside the mask
    on.fsel_set(FselMode::GateIn, 0, &vec![n as u32; l]).unwrap();
    assert_eq!(gen(&on), st, "GateIn K = inter != Static");
    let kin = n / 6;
    on.fsel_set(FselMode::GateIn, 0, &vec![kin as u32; l]).unwrap();
    let _ = gen(&on);
    let c = on.fsel_cov().unwrap();
    for li in 0..l { assert!(c[li * 4 + 2] as usize >= kin && (c[li * 4 + 2] as usize) <= kin + 2, "layer {} kept {}", li, c[li * 4 + 2]); }
    let (sel, mask) = (on.fsel_sel().unwrap(), on.fsel_mask().unwrap());
    assert!(sel.iter().zip(mask.iter()).all(|(s, m)| *s == 0 || *m != 0), "GateIn kept a neuron outside the mask");
}
