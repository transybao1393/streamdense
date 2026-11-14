//! Margin-gated recheck (gate_recheck.rs) on the tiny model (models/tiny_qwen35), Metal, no tolerance:
//! - tau = +inf, full mode (every decode step held, then redone with the full FFN in every layer): the greedy tokens
//!   are exactly the FULL engine's (exact prefill, fused decode) — the streamed prefill already equals it, and every
//!   recheck rewrites KV / DeltaNet state of its position from the same past;
//! - tau = 0 (hold pass + replay every step, never a recheck): bit-for-bit the plain compact decode (`fsel_step`):
//!   same tokens and same margins;
//! - "last" mode (full FFN in the last layer only) runs, is deterministic, and full_from >= layers is refused.
//! - missing-only load (`gate_recheck_ex(.., true)`): every recheck's logits bit-identical to the full-read recheck
//!   (both in full and last mode, for a scattered set and merge gaps 0 / 4 / 1000), tau = +inf == the full model, and it
//!   reads fewer bytes; same with the mmap copy (SD_RECHECK_MMAP, default on) and pread, with the resident rows copied by the
//!   CPU or by the GPU (`cs_rowscatter`, SD_RECHECK_GPU_ROWS, default on).
//! - zero-copy recheck (`SD_RECHECK_ZEROCOPY`, default on; residency set `SD_RECHECK_RESIDENT=1`, default off, also
//!   checked on): the full-FFN layers read gate / up / down W from no-copy
//!   Metal buffers over the mapped model file; every recheck's logits bit-identical to the copy path (= full read), and
//!   the 16-byte alignment check falls back to the copy path on misaligned files (the tiny model's are).
use std::sync::Arc;
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const PROMPT: [u32; 13] = [760, 1512, 314, 938, 369, 279, 1974, 11, 323, 1036, 374, 525, 204];
const N: usize = 24;
/// The tests below that set SD_RECHECK_* env vars hold this (the env is process-wide; tests run on threads).
static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&d).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}

fn compact(d: &str, set: &CompactSet, gate: bool) -> QwenEngine {
    QwenEngine::open(d, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: Some(Arc::new(set.clone())),
                                      ffn_stream_tokens: 64, ffn_stream_slots: 2, ffn_stream_host: true, decode_fused: true,
                                      gate_hold: gate, ..Default::default() }).unwrap()
}

/// Gated greedy decode: `recheck(step, margin)` decides; `full_from` = first full-FFN layer of a recheck.
/// Returns (tokens, margins of the hold passes, rechecks).
fn gated(e: &QwenEngine, full_from: usize, recheck: &dyn Fn(usize, f32) -> bool) -> (Vec<u32>, Vec<f32>, usize) {
    let (t, m, n, _, _) = gated_ex(e, full_from, false, recheck);
    (t, m, n)
}

/// `gated` with the missing-only load switch; also returns every recheck's logits and its bytes read.
fn gated_ex(e: &QwenEngine, full_from: usize, missing: bool, recheck: &dyn Fn(usize, f32) -> bool)
            -> (Vec<u32>, Vec<f32>, usize, Vec<Vec<f32>>, u64) {
    e.reset();
    let st = e.stream_prefill(&PROMPT, &mut |_, _, _| Ok(vec![])).unwrap();
    let (mut toks, mut ms, mut pending, mut nr, mut lg, mut by) = (vec![st.tok], vec![], false, 0, vec![], 0u64);
    while toks.len() < N {
        let last = *toks.last().unwrap();
        let (x, m) = e.gate_hold_pass(last, pending).unwrap();
        ms.push(m);
        if recheck(toks.len(), m) {
            let p = e.pos();
            let r = e.gate_recheck_ex(last, full_from, missing).unwrap();
            assert_eq!(e.pos(), p + 1);
            lg.push(e.last_logits().to_vec());
            by += r.ssd_bytes;
            toks.push(r.tok);
            pending = false;
            nr += 1;
        } else {
            toks.push(x);
            pending = true;
        }
    }
    (toks, ms, nr, lg, by)
}

#[test]
fn gate_recheck_full_and_zero() {
    let Some(d) = dir() else { return };
    let full = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, decode_fused: true, ..Default::default() }).unwrap();
    let (l, n) = (full.cfg.layers, full.cfg.inter);
    full.reset();
    let mut want = vec![];
    full.feed_stream(&PROMPT, N, &[], &mut |x| { want.push(x); true }).unwrap();
    drop(full);
    let set = CompactSet { inter: n, keep: vec![(0..(n / 16 * 8) as u32).collect(); l] };
    let b = QwenEngine::budget(&d, &QwenOptions { ffn_compact: Some(Arc::new(set.clone())), ffn_stream_tokens: 64, gate_hold: true, prefill_exact: true, activity: false, ..Default::default() }).unwrap();
    assert!(b.items.iter().any(|(x, _)| x.starts_with("margin gate")), "hold row missing from the budget:\n{}", b.report());

    // plain compact reference
    let ep = compact(&d, &set, false);
    ep.reset();
    let st = ep.stream_prefill(&PROMPT, &mut |_, _, _| Ok(vec![])).unwrap();
    let (mut plain, mut pm) = (vec![st.tok], vec![]);
    while plain.len() < N { let (x, m) = ep.fsel_step(*plain.last().unwrap()).unwrap(); plain.push(x); pm.push(m); }
    drop(ep);

    let eg = compact(&d, &set, true);
    // tau = +inf, full: == the full model
    let (t_inf, _, nr) = gated(&eg, 0, &|_, _| true);
    eprintln!("full model {:?}\ntau inf    {:?}\ncompact    {:?}", want, t_inf, plain);
    assert_eq!(nr, N - 1);
    assert_eq!(t_inf, want, "tau = +inf (every step rechecked, full FFN) != full model");
    assert_ne!(plain, want, "the half set should change the tiny model's output (else the test proves nothing)");
    // tau = 0: == plain compact, bit for bit (tokens and margins)
    let (t0, m0, nr0) = gated(&eg, 0, &|_, m| m < 0.0);
    assert_eq!(nr0, 0);
    assert_eq!(t0, plain, "tau = 0 != plain compact");
    assert_eq!(m0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), pm.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "margins differ");
    // mixed (every 3rd step rechecked) and last mode: run, deterministic
    let (a, _, _) = gated(&eg, 0, &|i, _| i % 3 == 0);
    let (b2, _, _) = gated(&eg, 0, &|i, _| i % 3 == 0);
    assert_eq!(a, b2);
    let (la, _, nl) = gated(&eg, l - 1, &|_, _| true);
    let (lb, _, _) = gated(&eg, l - 1, &|_, _| true);
    eprintln!("mixed      {:?}\nlast 1     {:?}", a, la);
    assert_eq!(nl, N - 1);
    assert_eq!(la, lb);
    // after all that, tau = +inf again from a reset == the full model (no state leaks between sequences)
    assert_eq!(gated(&eg, 0, &|_, _| true).0, want);
    eg.reset();
    eg.stream_prefill(&PROMPT, &mut |_, _, _| Ok(vec![])).unwrap();
    eg.gate_hold_pass(want[0], false).unwrap();
    assert!(eg.gate_recheck(want[0], l).is_err());
}

#[test]
fn gate_recheck_missing_only() {
    let Some(d) = dir() else { return };
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("SD_RECHECK_ZEROCOPY", "0");
    let full = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, decode_fused: true, ..Default::default() }).unwrap();
    let (l, n) = (full.cfg.layers, full.cfg.inter);
    full.reset();
    let mut want = vec![];
    full.feed_stream(&PROMPT, N, &[], &mut |x| { want.push(x); true }).unwrap();
    drop(full);
    // a scattered half set (runs of kept and missing rows of varying length, per layer different), multiple of 8
    let keep: Vec<Vec<u32>> = (0..l).map(|li| {
        let v: Vec<u32> = (0..n as u32).filter(|&j| ((j as usize * 7 + li * 3) % 11) < 5 || j % 13 == 0).collect();
        v[..v.len() / 8 * 8].to_vec()
    }).collect();
    let set = CompactSet { inter: n, keep };
    set.validate().unwrap();
    let eg = compact(&d, &set, true);
    let bits = |v: &[Vec<f32>]| v.iter().map(|x| x.iter().map(|f| f.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>();
    for ff in [0, l - 1] {
        let (tf, _, nf, lf, bf) = gated_ex(&eg, ff, false, &|_, _| true);
        for (gap, mm, gr) in [("0", "0", "0"), ("4", "0", "0"), ("1000", "0", "0"), ("0", "1", "0"), ("4", "1", "0"),
                             ("0", "0", "1"), ("4", "0", "1"), ("1000", "0", "1"), ("0", "1", "1"), ("4", "1", "1")] {
            std::env::set_var("SD_RECHECK_GAP", gap);
            std::env::set_var("SD_RECHECK_MMAP", mm);
            std::env::set_var("SD_RECHECK_GPU_ROWS", gr);
            let (tm, _, nm, lm, bm) = gated_ex(&eg, ff, true, &|_, _| true);
            eprintln!("full_from {} gap {} mmap {} gpu rows {}: full read {} B, missing {} B ({:.3})", ff, gap, mm, gr, bf, bm, bm as f64 / bf as f64);
            assert_eq!(nm, nf);
            assert_eq!(tm, tf, "missing-only tokens != full-read recheck (full_from {}, gap {})", ff, gap);
            assert!(bits(&lm) == bits(&lf), "missing-only logits not bit-identical (full_from {}, gap {})", ff, gap);
            if gap != "1000" { assert!(bm < bf, "missing-only read {} B, not fewer than {} B", bm, bf); }
            if ff == 0 { assert_eq!(tm, want, "missing-only tau = +inf != full model (gap {})", gap); }
        }
        std::env::remove_var("SD_RECHECK_GAP");
        std::env::remove_var("SD_RECHECK_MMAP");
        std::env::remove_var("SD_RECHECK_GPU_ROWS");
    }
    // mixed gating, missing (defaults: mmap copy, GPU resident rows) == full read
    let (a, _, _, la, _) = gated_ex(&eg, 0, false, &|i, _| i % 3 == 0);
    let (b, _, _, lb, _) = gated_ex(&eg, 0, true, &|i, _| i % 3 == 0);
    assert_eq!(a, b);
    assert!(bits(&la) == bits(&lb));
    std::env::remove_var("SD_RECHECK_ZEROCOPY");
}

#[test]
fn gate_recheck_zero_copy() {
    let Some(d) = dir() else { return };
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let full = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, decode_fused: true, ..Default::default() }).unwrap();
    let (l, n) = (full.cfg.layers, full.cfg.inter);
    full.reset();
    let mut want = vec![];
    full.feed_stream(&PROMPT, N, &[], &mut |x| { want.push(x); true }).unwrap();
    drop(full);
    let keep: Vec<Vec<u32>> = (0..l).map(|li| {
        let v: Vec<u32> = (0..n as u32).filter(|&j| ((j as usize * 5 + li) % 9) < 4 || j % 7 == 0).collect();
        v[..v.len() / 8 * 8].to_vec()
    }).collect();
    let set = CompactSet { inter: n, keep };
    set.validate().unwrap();
    let bits = |v: &[Vec<f32>]| v.iter().map(|x| x.iter().map(|f| f.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mixed_gate = |i: usize, _: f32| i % 3 == 0;
    // reference: the copy path (default) on its own engine
    std::env::set_var("SD_RECHECK_ZEROCOPY", "0");
    let ec = compact(&d, &set, true);
    let refs: Vec<_> = [0, l - 1].iter().map(|&ff| gated_ex(&ec, ff, true, &|_, _| true)).collect();
    let mixed = gated_ex(&ec, 0, true, &mixed_gate);
    drop(ec);
    // zero-copy (default on): the tiny model's FFN tensors sit at 16 k and 16 k + 4 file offsets -> both direct binding
    // and the GPU re-align copy run
    std::env::remove_var("SD_RECHECK_ZEROCOPY");
    let ez = compact(&d, &set, true);
    for (&ff, r) in [0, l - 1].iter().zip(&refs) {
        let (tz, _, nz, lz, bz) = gated_ex(&ez, ff, true, &|_, _| true);
        eprintln!("zero-copy full_from {}: {} rechecks, {} B read (copy path {} B)", ff, nz, bz, r.4);
        assert_eq!(nz, r.2);
        assert_eq!(bz, 0, "zero-copy read bytes from the file (fell back to the copy path?)");
        assert_eq!(tz, r.0, "zero-copy tokens != copy path (full_from {})", ff);
        assert!(bits(&lz) == bits(&r.3), "zero-copy logits not bit-identical (full_from {})", ff);
        if ff == 0 { assert_eq!(tz, want, "zero-copy tau = +inf != full model"); }
    }
    let (a, _, _, la, _) = gated_ex(&ez, 0, true, &mixed_gate);
    assert_eq!(a, mixed.0);
    assert!(bits(&la) == bits(&mixed.3));
    // switched off on the same engine: back to the copy path, same logits
    std::env::set_var("SD_RECHECK_ZEROCOPY", "0");
    let (b, _, _, lb, bb) = gated_ex(&ez, 0, true, &mixed_gate);
    std::env::remove_var("SD_RECHECK_ZEROCOPY");
    assert!(bb > 0);
    assert_eq!(b, mixed.0);
    assert!(bits(&lb) == bits(&mixed.3));
    drop(ez);
    // zero-copy WITH the residency set (SD_RECHECK_RESIDENT=1, opt-in) on its own engine: same bits as the default
    // (no set) above
    std::env::set_var("SD_RECHECK_RESIDENT", "1");
    let en = compact(&d, &set, true);
    let (tn, _, _, ln, bn) = gated_ex(&en, 0, true, &mixed_gate);
    let (tf, _, _, lf, _) = gated_ex(&en, l - 1, true, &|_, _| true);
    drop(en);
    std::env::remove_var("SD_RECHECK_RESIDENT");
    assert_eq!(bn, 0);
    assert_eq!(tn, mixed.0);
    assert!(bits(&ln) == bits(&mixed.3), "zero-copy with residency set not bit-identical");
    assert_eq!(tf, refs[1].0);
    assert!(bits(&lf) == bits(&refs[1].3));
    // any byte alignment: copies of the tiny model with every MLP tensor at file offset 16 k + shift (odd / 2 mod 4: the
    // 27B's files 2 and 3 are like that) -> cs_bytecopy; shift 0 + SD_RECHECK_ZC_COPYALL: the copy kernel on aligned
    // tensors; shift 0: all direct. Zero-copy == copy path bit for bit.
    for (shift, all) in [(0usize, false), (0, true), (1, false), (2, false), (3, false), (7, false)] {
        let sd = shifted_copy(&d, shift);
        std::env::set_var("SD_RECHECK_ZEROCOPY", "0");
        let ec = compact(&sd, &set, true);
        let (tc, _, _, lc, bc) = gated_ex(&ec, 0, true, &mixed_gate);
        drop(ec);
        std::env::remove_var("SD_RECHECK_ZEROCOPY");
        if all { std::env::set_var("SD_RECHECK_ZC_COPYALL", "1"); }
        let ez = compact(&sd, &set, true);
        let (tz, _, _, lz, bz) = gated_ex(&ez, 0, true, &mixed_gate);
        drop(ez);
        std::env::remove_var("SD_RECHECK_ZC_COPYALL");
        let _ = std::fs::remove_dir_all(&sd);
        eprintln!("file shift {} copy-all {}: copy path {} B read, zero-copy {} B", shift, all, bc, bz);
        assert!(bc > 0 && bz == 0);
        assert_eq!(tc, mixed.0, "re-laid-out copy changed the model (shift {})", shift);
        assert_eq!(tz, tc, "zero-copy tokens != copy path at file shift {}", shift);
        assert!(bits(&lz) == bits(&lc), "zero-copy logits not bit-identical at file shift {}", shift);
    }
}

/// A copy of the tiny model dir whose safetensors data is laid out again: every non-MLP tensor at a 16-byte aligned
/// file offset, every MLP tensor at (16-byte aligned + `shift`), so only the FFN weights the zero-copy path wraps are
/// misaligned. Returns the copy's path (temp dir).
fn shifted_copy(d: &str, shift: usize) -> String {
    let out = std::env::temp_dir().join(format!("sd_zc_shift{}_{}", shift, std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(out.join("mtp")).unwrap();
    for f in ["config.json", "generation_config.json", "model.safetensors.index.json", "mtp/weights.safetensors"] {
        let src = format!("{}/{}", d, f);
        if std::path::Path::new(&src).exists() { std::fs::copy(&src, out.join(f)).unwrap(); }
    }
    let name = "model-00001-of-00001.safetensors";
    let b = std::fs::read(format!("{}/{}", d, name)).unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let mut h: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let names: Vec<String> = h.keys().filter(|k| *k != "__metadata__").cloned().collect();
    let mut data: Vec<u8> = Vec::new();
    for k in &names {
        let t = h.get_mut(k).unwrap();
        let o = &t["data_offsets"];
        let (a, z) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        // header length is padded to a multiple of 16 below, so data offset == file offset mod 16
        let pos = data.len().div_ceil(16) * 16 + if k.contains(".mlp.") { shift } else { 0 };
        data.resize(pos, 0);
        data.extend_from_slice(&b[8 + n + a..8 + n + z]);
        t["data_offsets"] = serde_json::json!([pos, pos + (z - a)]);
    }
    let mut hdr = serde_json::to_vec(&h).unwrap();
    while (8 + hdr.len()) % 16 != 0 { hdr.push(b' '); }
    let mut v = Vec::with_capacity(8 + hdr.len() + data.len());
    v.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    v.extend_from_slice(&hdr);
    v.extend_from_slice(&data);
    std::fs::write(out.join(name), v).unwrap();
    out.to_string_lossy().into_owned()
}
