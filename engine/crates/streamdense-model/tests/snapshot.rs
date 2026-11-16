//! Engine state snapshots (src/snapshot.rs) on the tiny model (models/tiny_qwen35, with mtp/).
//! - size formula: `body_bytes` == the bytes `snapshot()` produces; the 27B number at pos 1259 from its config only
//!   (models/qwen38_27b_mlx4/config.json, nothing loaded).
//! - bit-exact restore, plain (feed_stream) and MTP (feed_stream_mtp), fast prefill and --prefill-exact:
//!   ref = reset, feed prefix P (n_new = 1, token t0 held back), snapshot (file + RAM), then
//!     (a) decode N from t0 and (b) feed a remainder R then decode N;
//!   each replayed after: dirtying the engine with another prompt then restore (file), restore (RAM), and a second
//!   freshly opened engine (restart) restoring the file. Equal: tokens, stop reasons, positions, rec records
//!   (argmax, top-1, top-2 bitwise) for every generated position, and (plain) the full logits row per token.
//! - the same with 8-bit KV (q8: exact prefill), plain and MTP.
//! - cross: an MTP snapshot restored into a plain engine == the plain reference.
//! - errors: bad magic, truncated body, tokens.len() != pos; cache key / longest-prefix lookup.
//! - timing: save / restore (file, RAM) on the tiny model, and the 27B estimate at ~3 GB/s.
//! Run: cargo test --release --offline -p streamdense-model --test snapshot -- --nocapture --test-threads 1
use std::path::PathBuf;
use std::time::Instant;

use streamdense_model::snapshot::{self, SnapshotCache};
use streamdense_model::{KvFormat, QwenConfig, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const P: usize = 1259; // prefix length (the Claude Code prefix size)
const R: usize = 37;   // remainder
const N: usize = 40;   // decoded tokens

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}
fn open(mtp: bool, exact: bool, max_ctx: usize) -> Option<QwenEngine> { open_kv(mtp, exact, max_ctx, KvFormat::Bf16) }
fn open_kv(mtp: bool, exact: bool, max_ctx: usize, kv: KvFormat) -> Option<QwenEngine> {
    let opt = QwenOptions { max_ctx, mtp, prefill_exact: exact, kv, ..Default::default() };
    Some(QwenEngine::open(&model_dir()?, opt).expect("open tiny"))
}
fn rnd(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 2000) as u32 + 10 }).collect()
}
fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd_snapshot_test_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// What a continuation produced: tokens, stop reason, end position, per generated position the rec record, and
/// (plain only) the logits row per emitted decode token.
#[derive(Debug, PartialEq)]
struct Run { toks: Vec<u32>, stop: String, pos: usize, rec: Vec<(u32, u32, u32)>, logits: Vec<Vec<u32>> }

/// Continue from the current state: feed `ids`, decode up to N tokens (plain or MTP).
fn cont(eng: &QwenEngine, ids: &[u32], mtp: bool) -> Run {
    let p0 = eng.pos();
    let mut toks = Vec::new();
    let mut logits: Vec<Vec<u32>> = Vec::new();
    let stop = if mtp {
        let (st, _) = eng.feed_stream_mtp(ids, N, &[], &mut |t| { toks.push(t); true }).unwrap();
        st.stop_reason.to_string()
    } else {
        let st = eng.feed_stream(ids, N, &[], &mut |t| {
            toks.push(t);
            logits.push(eng.head_logits(0).iter().map(|x| x.to_bits()).collect());
            true
        }).unwrap();
        st.stop_reason.to_string()
    };
    // generated token i was produced at position p0 + ids.len() - 1 + i
    let rec = (0..toks.len()).map(|i| {
        let (t, a, b) = eng.record(p0 + ids.len() - 1 + i);
        (t, a.to_bits(), b.to_bits())
    }).collect();
    Run { toks, stop, pos: eng.pos(), rec, logits }
}

/// reset + feed the prefix (one generated token held back); returns it.
fn feed_prefix(eng: &QwenEngine, prefix: &[u32], mtp: bool) -> u32 {
    eng.reset();
    let mut t0 = None;
    if mtp { eng.feed_stream_mtp(prefix, 1, &[], &mut |t| { t0 = Some(t); true }).unwrap(); }
    else { eng.feed_stream(prefix, 1, &[], &mut |t| { t0 = Some(t); true }).unwrap(); }
    assert_eq!(eng.pos(), prefix.len());
    t0.unwrap()
}

fn check_mode(mtp: bool, exact: bool) { check_mode_kv(mtp, exact, KvFormat::Bf16) }
fn check_mode_kv(mtp: bool, exact: bool, kv: KvFormat) {
    let Some(eng) = open_kv(mtp, exact, 4096, kv) else { return };
    let tag = format!("{} {}{}", if mtp { "mtp" } else { "plain" }, if exact { "exact" } else { "fast" }, if kv == KvFormat::Q8 { " q8" } else { "" });
    let prefix = rnd(0x51A9_5EED, P);
    let rem = rnd(0xBEEF_0001, R);
    let other = rnd(0x0DD_BA11, 300);
    let path = tmp(&format!("{}.sdsnap", tag.replace(' ', "_")));

    // reference runs (no snapshot involved)
    let t0 = feed_prefix(&eng, &prefix, mtp);
    let ref_a = cont(&eng, &[t0], mtp);
    feed_prefix(&eng, &prefix, mtp);
    let ref_b = cont(&eng, &rem, mtp);
    assert!(ref_a.toks.len() > 1 && ref_b.toks.len() > 1, "{}: too few tokens", tag);

    // snapshot after the prefix
    assert_eq!(feed_prefix(&eng, &prefix, mtp), t0);
    let ts = Instant::now();
    eng.save_snapshot(&path, &prefix).unwrap();
    let save_s = ts.elapsed().as_secs_f64();
    let mem = eng.snapshot(&prefix).unwrap();
    let file_len = std::fs::metadata(&path).unwrap().len();
    let mtp_len = if mtp { P } else { 0 };
    assert_eq!(mem.bytes() as u64, snapshot::body_bytes(&eng.cfg, eng.opt.kv, P, mtp_len, mtp), "{}: body size formula", tag);
    assert_eq!(mem.header.mtp_len, mtp_len, "{}: mtp_len after feed_stream_mtp", tag);

    let dirty = |e: &QwenEngine| { e.reset(); cont(e, &other, mtp); };
    let mut restore_file_s = 0.0;
    let mut restore_mem_s = 0.0;
    for (case, want) in [("a", &ref_a), ("b", &ref_b)] {
        let ids: Vec<u32> = if case == "a" { vec![t0] } else { rem.clone() };
        // same engine, file
        dirty(&eng);
        let tr = Instant::now();
        let got_tokens = eng.restore_snapshot(&path).unwrap();
        restore_file_s = tr.elapsed().as_secs_f64();
        assert_eq!(got_tokens, prefix);
        assert_eq!(eng.pos(), P);
        let r = cont(&eng, &ids, mtp);
        assert_eq!(&r, want, "{} case {}: file restore != reference", tag, case);
        // same engine, RAM
        dirty(&eng);
        let tr = Instant::now();
        eng.restore(&mem).unwrap();
        restore_mem_s = tr.elapsed().as_secs_f64();
        let r = cont(&eng, &ids, mtp);
        assert_eq!(&r, want, "{} case {}: RAM restore != reference", tag, case);
        println!("  {:12} case {}: {} tokens, stop {}, pos {} — file + RAM restore bit-exact", tag, case, r.toks.len(), r.stop, r.pos);
    }
    // restart: a second engine (different max_ctx: rows re-strided) from the file
    drop(mem);
    let Some(e2) = open_kv(mtp, exact, 2048, kv) else { return };
    for (case, want) in [("a", &ref_a), ("b", &ref_b)] {
        let ids: Vec<u32> = if case == "a" { vec![t0] } else { rem.clone() };
        e2.restore_snapshot(&path).unwrap();
        let r = cont(&e2, &ids, mtp);
        assert_eq!(&r, want, "{} case {}: restart (max_ctx 2048) restore != reference", tag, case);
    }
    println!("  {:12} snapshot {} B (file {} B); save {:.2} ms, restore file {:.2} ms ({:.2} GB/s), RAM {:.2} ms; restart restore bit-exact",
             tag, mem_len(&eng, mtp), file_len, save_s * 1e3, restore_file_s * 1e3, file_len as f64 / restore_file_s / 1e9, restore_mem_s * 1e3);
    let _ = std::fs::remove_file(&path);
}
fn mem_len(eng: &QwenEngine, mtp: bool) -> u64 { snapshot::body_bytes(&eng.cfg, eng.opt.kv, P, if mtp { P } else { 0 }, mtp) }

#[test]
fn restore_plain_fast() { check_mode(false, false); }
#[test]
fn restore_plain_exact() { check_mode(false, true); }
#[test]
fn restore_mtp_fast() { check_mode(true, false); }
#[test]
fn restore_mtp_exact() { check_mode(true, true); }
/// 8-bit KV (exact prefill): K / V bytes and their scale / min rows are saved and restored bit-exact
#[test]
fn restore_plain_q8() { check_mode_kv(false, true, KvFormat::Q8); }
#[test]
fn restore_mtp_q8() { check_mode_kv(true, true, KvFormat::Q8); }

/// MTP-engine snapshot restored into a plain engine == plain reference (MTP == plain is the spec contract).
#[test]
fn cross_mtp_to_plain() {
    let (Some(m), Some(p)) = (open(true, true, 4096), open(false, true, 4096)) else { return };
    let prefix = rnd(0x51A9_5EED, 300);
    let t0 = feed_prefix(&p, &prefix, false);
    let want = cont(&p, &[t0], false);
    assert_eq!(feed_prefix(&m, &prefix, true), t0);
    let s = m.snapshot(&prefix).unwrap();
    p.reset();
    p.restore(&s).unwrap();
    assert_eq!(cont(&p, &[t0], false), want, "MTP snapshot in a plain engine != plain");
    // and back: plain snapshot into the MTP engine: tokens equal plain (MTP KV stale only)
    feed_prefix(&p, &prefix, false);
    let s = p.snapshot(&prefix).unwrap();
    m.reset();
    m.restore(&s).unwrap();
    assert_eq!(cont(&m, &[t0], true).toks, want.toks, "plain snapshot in an MTP engine: tokens != plain");
    println!("  cross plain <-> mtp: ok");
}

#[test]
fn errors_and_cache() {
    let Some(eng) = open(false, true, 4096) else { return };
    let prefix = rnd(7, 64);
    feed_prefix(&eng, &prefix, false);
    assert!(eng.snapshot(&prefix[..63]).is_err(), "tokens.len() != pos must fail");
    let path = tmp("err.sdsnap");
    eng.save_snapshot(&path, &prefix).unwrap();
    // truncated
    let b = std::fs::read(&path).unwrap();
    let bad = tmp("trunc.sdsnap");
    std::fs::write(&bad, &b[..b.len() - 100]).unwrap();
    assert!(eng.restore_snapshot(&bad).is_err());
    let mut m = b.clone();
    m[0] = b'X';
    std::fs::write(&bad, &m).unwrap();
    assert!(eng.restore_snapshot(&bad).is_err());
    // cache: longest prefix
    let cache = SnapshotCache::open(tmp("cache"), "tiny_qwen35-test").unwrap();
    feed_prefix(&eng, &prefix[..32], false);
    let p32 = cache.save(&eng, &prefix[..32]).unwrap();
    feed_prefix(&eng, &prefix, false);
    let p64 = cache.save(&eng, &prefix).unwrap();
    let mut long = prefix.clone();
    long.extend(rnd(9, 10));
    assert_eq!(cache.longest_prefix(&eng, &long), Some((p64.clone(), 64)));
    assert_eq!(cache.longest_prefix(&eng, &prefix[..40]), Some((p32, 32)));
    let mut miss = prefix.clone();
    miss[0] ^= 1;
    assert_eq!(cache.longest_prefix(&eng, &miss), None);
    let other = SnapshotCache::open(tmp("cache"), "other-model").unwrap();
    assert_eq!(other.longest_prefix(&eng, &long), None, "model id must be part of the key");
    let _ = std::fs::remove_dir_all(tmp("cache"));
    for f in ["err.sdsnap", "trunc.sdsnap"] { let _ = std::fs::remove_file(tmp(f)); }
    println!("  errors + cache: ok");
}

/// Snapshot size of the 27B at pos 1259 from its config (nothing loaded), and the restore estimate.
#[test]
fn size_27b() {
    let d = format!("{}/models/qwen38_27b_mlx4", ROOT);
    let Ok(cfg) = QwenConfig::from_model_dir(&d) else { eprintln!("SKIP: {} config missing", d); return };
    let plain = snapshot::body_bytes(&cfg, KvFormat::Bf16, P, 0, false);
    let mtp = snapshot::body_bytes(&cfg, KvFormat::Bf16, P, P, true);
    println!("  27B @ {}: plain {} B ({:.1} MB), mtp {} B ({:.1} MB); restore @3 GB/s ≈ {:.0} ms", P, plain, plain as f64 / 1e6,
             mtp, mtp as f64 / 1e6, mtp as f64 / 3e9 * 1e3);
    if cfg.n_deltanet() == 48 && cfg.n_attention() == 16 && cfg.kv_heads == 4 && cfg.head_dim == 256 {
        assert_eq!(plain, 236_474_176);
        assert_eq!(mtp, 241_631_040);
    }
}
