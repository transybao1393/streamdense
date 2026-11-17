//! MTP prefill warm-up v2 (2026-11-17, `SD_MTP_WARM_V2` = 0 | fc | 1; mtp.rs `WarmMode`) on the tiny model.
//! - `fc` (whole-chunk fc in one `bf16_gemv_g4` dispatch): MTP KV [0..P) and the first draft bit-identical to `0`
//!   (today's path), fast and exact prefill, bf16 and q8 KV.
//! - `1` (fast MTP layer on fast chunks): exact prefill engine bit-identical to `0` (exact chunks keep the exact
//!   layer); fast prefill: MTP KV differences reported (bytes differing, max |dK|, |dV| in bf16).
//! - Output never changes: generate_mtp and deep spec (fixed depth 3, auto) == plain greedy in every mode;
//!   acceptance per mode reported.
//! - Prompt-read wall time per mode (tiny model: not representative of 27B; see the report).
//! Env is set per run inside this file: run with --test-threads 1.
//! Run: cargo test --release --offline -p streamdense-model --test mtp_warm -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_model::qwen_engine::KvFormat;
use streamdense_model::spec::{generate_mtp, DraftOverride};
use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
use streamdense_model::{QwenEngine, QwenOptions};
use std::time::Instant;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const MODES: [&str; 3] = ["0", "fc", "1"];

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}

fn prompts() -> Vec<(String, Vec<u32>)> {
    let p = format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT);
    let j: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    j["prompts"].as_array().unwrap().iter()
        .map(|x| (x["id"].as_str().unwrap().to_string(), x["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()))
        .collect()
}

fn set_mode(m: &str) { std::env::set_var("SD_MTP_WARM_V2", m); }

/// MTP KV bytes of rows [0..n) (K and V per head; q8 adds the scale / min rows).
fn mtp_kv_bytes(eng: &QwenEngine, n: usize) -> (Vec<u8>, Vec<u8>) {
    let kv = eng.mtp_kv().expect("mtp kv");
    let (hd, nkv) = (eng.cfg.head_dim, eng.cfg.kv_heads);
    let eb = if kv.q8 { 1 } else { 2 };
    let base = eng.arena.base();
    let rd = |off: usize, len: usize| unsafe { std::slice::from_raw_parts(base.add(off), len).to_vec() };
    let mut out = [Vec::new(), Vec::new()];
    for (i, b) in [kv.k, kv.v].into_iter().enumerate() {
        for h in 0..nkv { out[i].extend(rd(b + h * kv.max_ctx * hd * eb, n * hd * eb)); }
    }
    if kv.q8 {
        let sb = (hd / 32) * 4;
        for (i, b) in [kv.ksb, kv.vsb].into_iter().enumerate() {
            for h in 0..nkv { out[i].extend(rd(b + h * kv.max_ctx * sb, n * sb)); }
        }
    }
    (out[0].clone(), out[1].clone())
}

fn bf(x: u16) -> f32 { f32::from_bits((x as u32) << 16) }
fn max_bf_diff(a: &[u8], b: &[u8]) -> (f32, f32) {
    let (mut d, mut m) = (0f32, 0f32);
    for (x, y) in a.chunks(2).zip(b.chunks(2)) {
        let (p, q) = (bf(u16::from_le_bytes([x[0], x[1]])), bf(u16::from_le_bytes([y[0], y[1]])));
        d = d.max((p - q).abs());
        m = m.max(p.abs());
    }
    (d, m)
}

struct Warm { kv: (Vec<u8>, Vec<u8>), draft: u32, ms: f64 }

/// reset, prefill_mtp(prompt) (timed, min of `reps`), then the first draft; MTP KV rows [0..P).
fn warm(eng: &QwenEngine, p: &[u32], reps: usize) -> Warm {
    let mut ms = f64::MAX;
    let mut last = (0, 0);
    for _ in 0..reps {
        eng.reset();
        let t = Instant::now();
        last = eng.prefill_mtp(p).unwrap();
        ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    // rows [0..P): the prompt read writes [0..P-1), the first draft row P-1 (read after it: no stale row)
    let draft = eng.mtp_first_draft(last.0, last.1, 0).unwrap();
    let kv = mtp_kv_bytes(eng, p.len());
    Warm { kv, draft, ms }
}

fn open(exact: bool, kv: KvFormat) -> Option<QwenEngine> {
    let dir = model_dir()?;
    Some(QwenEngine::open(&dir, QwenOptions { max_ctx: 8192, mtp: true, spec_depth: 7, prefill_exact: exact, kv, ..Default::default() })
        .expect("open tiny"))
}

#[test]
fn warm_modes_kv_identity_and_diff() {
    let ps = prompts();
    let mut ok = true;
    for (exact, kv) in [(false, KvFormat::Bf16), (false, KvFormat::Q8), (true, KvFormat::Bf16), (true, KvFormat::Q8)] {
        let Some(eng) = open(exact, kv) else { return };
        for (id, p) in &ps {
            if exact && p.len() > 1000 { continue; } // exact prefill of 4K on tiny is slow and adds nothing
            let r: Vec<Warm> = MODES.iter().map(|m| { set_mode(m); warm(&eng, p, 1) }).collect();
            let fc_eq = r[1].kv == r[0].kv && r[1].draft == r[0].draft;
            let v2_eq = r[2].kv == r[0].kv && r[2].draft == r[0].draft;
            let nd = r[2].kv.0.iter().zip(&r[0].kv.0).filter(|(a, b)| a != b).count()
                   + r[2].kv.1.iter().zip(&r[0].kv.1).filter(|(a, b)| a != b).count();
            let tot = r[0].kv.0.len() + r[0].kv.1.len();
            let diff = if matches!(kv, KvFormat::Bf16) && !v2_eq {
                let (dk, mk) = max_bf_diff(&r[2].kv.0, &r[0].kv.0);
                let (dv, mv) = max_bf_diff(&r[2].kv.1, &r[0].kv.1);
                format!(" max|dK| {:.3e} (|K| max {:.2}) max|dV| {:.3e} (|V| max {:.2})", dk, mk, dv, mv)
            } else { String::new() };
            println!("  {:5} {:4} {:8} P={:4}: fc == 0 {} | v2 == 0 {} ({} / {} bytes differ, draft {} vs {}){}",
                     if exact { "exact" } else { "fast" }, if matches!(kv, KvFormat::Q8) { "q8" } else { "bf16" }, id, p.len(),
                     fc_eq, v2_eq, nd, tot, r[2].draft, r[0].draft, diff);
            ok &= fc_eq;
            if exact { ok &= v2_eq; }
        }
    }
    set_mode("1");
    assert!(ok, "fc mode (any prefill) or v2 on the exact prefill not bit-identical to mode 0");
}

#[test]
fn warm_modes_output_and_acceptance() {
    let ps = prompts();
    let Some(eng) = open(false, KvFormat::Q8) else { return };
    let mut ok = true;
    let mut tot: Vec<(usize, usize)> = vec![(0, 0); MODES.len() * 3];
    let mut agree: Vec<(usize, usize)> = vec![(0, 0); MODES.len()];
    for (id, p) in &ps {
        let n = 64;
        let plain = eng.generate(p, n, true).unwrap().tokens;
        let mut d0: Vec<u32> = vec![];
        for (mi, m) in MODES.iter().enumerate() {
            set_mode(m);
            for (si, (name, sm)) in [("mtp d1", None), ("fixed d3", Some(SpecMode::Fixed(3))), ("auto d7", Some(SpecMode::Auto { max: 7 }))].iter().enumerate() {
                let r = match sm {
                    None => generate_mtp(&eng, p, plain.len(), DraftOverride::None, true).unwrap(),
                    Some(s) => generate_spec_mode(&eng, p, plain.len(), *s, DraftOverride::None).unwrap(),
                };
                let eq = r.tokens == plain;
                ok &= eq;
                if si == 0 {
                    // MTP head drafts (d1, sync-free pipe run) vs mode 0: how often the warm-up change moves a draft
                    if mi == 0 { d0 = r.drafts.clone(); }
                    let a = &mut agree[mi];
                    a.0 += r.drafts.iter().zip(&d0).filter(|(x, y)| x == y).count();
                    a.1 += r.drafts.len().min(d0.len());
                }
                let t = &mut tot[mi * 3 + si];
                t.0 += r.accepted; t.1 += r.passes;
                println!("  {:8} P={:4} mode {:2} {:8}: == plain {} | passes {:3} accepted {:3} ({:.3}/pass, {:.2} tok/pass)",
                         id, p.len(), m, name, eq, r.passes, r.accepted, r.acceptance, r.tokens_per_pass);
            }
        }
    }
    for (mi, m) in MODES.iter().enumerate() {
        println!("drafts (mtp d1) equal to mode 0: mode {:2} {} / {}", m, agree[mi].0, agree[mi].1);
        for (si, name) in ["mtp d1", "fixed d3", "auto d7"].iter().enumerate() {
            let t = tot[mi * 3 + si];
            println!("total mode {:2} {:8}: accepted {} / passes {} = {:.3}", m, name, t.0, t.1, t.0 as f64 / t.1.max(1) as f64);
        }
    }
    set_mode("1");
    assert!(ok, "an MTP / deep spec run differs from plain greedy");
}

#[test]
fn warm_modes_prefill_time() {
    let ps = prompts();
    let Some(eng) = open(false, KvFormat::Q8) else { return };
    for (id, p) in ps.iter().filter(|x| x.1.len() > 500) {
        for m in MODES {
            set_mode(m);
            let w = warm(&eng, p, 5);
            println!("  {:8} P={:4} mode {:2}: prefill_mtp {:.2} ms (min of 5)", id, p.len(), m, w.ms);
        }
        eng.reset();
        let mut plain = f64::MAX;
        for _ in 0..5 { eng.reset(); let t = Instant::now(); eng.prefill(p).unwrap(); plain = plain.min(t.elapsed().as_secs_f64() * 1e3); }
        println!("  {:8} P={:4} plain prefill (no MTP warm-up) {:.2} ms", id, p.len(), plain);
    }
    set_mode("1");
}
