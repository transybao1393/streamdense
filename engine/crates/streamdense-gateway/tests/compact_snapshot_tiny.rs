//! 12 GB-mode snapshots (compact.rs + snap.rs, file format 2 in streamdense-model snapshot.rs) on the tiny model
//! (models/tiny_qwen35), Metal. Bit-exact, no tolerance:
//! 1. backend: prompt read, snapshot, a different prompt (other set + sums), restore, then the prefix extension ==
//!    the same extension in memory without the snapshot: sums, held length, set, tokens;
//! 2. worker over real HTTP (27B tokenizer, ids folded): a coding-agent request with snapshot cuts (stable prefix +
//!    rolling state, aligned to the read's chunks), its re-send (restore of the rolling state) and a new conversation
//!    on the same system prompt (restore of the stable prefix) give the same replies as a gateway without snapshots;
//! 3. format: full-mode files stay format 1 (the old header, no trailing section) and load as before; a full-mode file
//!    is refused by 12 GB mode and a 12 GB-mode file by the full model (clear messages); another alignment is refused;
//!    the two modes' cache keys never meet.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::client::request;
use streamdense_gateway::compact::CompactBackend;
use streamdense_gateway::{Backend, CacheHook, GatewayConfig, SnapshotHook, WorkerConfig};
use streamdense_model::ffn_set::{SetRule, T1};
use streamdense_model::snapshot::{SnapHeader, Snapshot, SnapshotCache, FORMAT, FORMAT_COMPACT};
use streamdense_model::{QwenEngine, QwenOptions};
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { eprintln!("SKIP: needs models/tiny_qwen35"); return None; }
    Some(d)
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd_csnap_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_file(&d);
    d
}

/// Deterministic synthetic global score / down norms for the tiny dims.
fn t1(l: usize, n: usize) -> T1 {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    T1 { layers: l, inter: n, global: (0..l * n).map(|_| r() / n as f32).collect(), down_norm: (0..l * n).map(|_| 0.5 + r()).collect() }
}

fn opts(rule: &SetRule, stream: usize, max_ctx: usize) -> QwenOptions {
    QwenOptions { max_ctx, prefill_exact: true, activity: false, mtp: false, ffn_compact: Some(Arc::new(rule.initial())), ffn_stream_tokens: stream,
                  ffn_stream_slots: 2, ffn_stream_host: true, ..Default::default() }
}

fn rule_for(d: &str) -> SetRule {
    let probe = QwenEngine::open(d, QwenOptions { max_ctx: 64, ..Default::default() }).unwrap();
    let (l, n) = (probe.cfg.layers, probe.cfg.inter);
    drop(probe);
    SetRule::new(t1(l, n), n / 16 * 8).unwrap() // half the neurons
}

fn feed(b: &CompactBackend, ids: &[u32], n: usize) -> Vec<u32> {
    let mut t = vec![];
    let f = b.feed(ids, n, &[], false, &mut |x| { t.push(x); true }).unwrap();
    assert_eq!(f.gen, t.len());
    t
}

fn ids(n: usize, seed: u32) -> Vec<u32> { (0..n as u32).map(|i| (i * 7919 + seed * 104_729 + 13) % 2000).collect() }

fn hook_for(b: &CompactBackend, d: &Path, model: &str) -> CacheHook {
    let mut cache = SnapshotCache::open(d.to_path_buf(), &CacheHook::model_id(model)).unwrap();
    cache.mode = b.snap_mode();
    CacheHook::new(cache, 8, 1 << 30)
}

#[test]
fn compact_snapshot_restore_equals_in_memory() {
    let Some(d) = dir() else { return };
    let rule = rule_for(&d);
    let ea = QwenEngine::open(&d, opts(&rule, 64, 1024)).unwrap();
    let b = CompactBackend::new(&ea, rule.clone(), json!({})).unwrap();
    let cd = tmp("backend");
    let mut hook = hook_for(&b, &cd, &d);
    let (p1, sfx, nn) = (ids(96, 1), ids(41, 2), 16);

    // in memory: prompt (1 token), then the pending token + suffix
    b.reset();
    let t0 = feed(&b, &p1, 1);
    let s1 = b.sums();
    assert_eq!(b.held(), 96);
    let mut ext = vec![t0[0]];
    ext.extend_from_slice(&sfx);
    let ta = feed(&b, &ext, nn);
    let (sa, seta, ha) = (b.sums(), (*ea.ffn_compact_set().unwrap()).clone(), b.held());

    // snapshot after the prompt; another prompt in between (other set + sums); restore; the same extension
    b.reset();
    assert_eq!(feed(&b, &p1, 1), t0);
    let ev = SnapshotHook::<CompactBackend>::save(&mut hook, &b, &p1, false, None).unwrap();
    assert!(ev.error.is_none(), "{:?}", ev.error);
    assert!(ev.bytes > (ea.cfg.layers * 2 * ea.cfg.inter * 4) as u64, "file holds the sums: {} B", ev.bytes);
    b.reset();
    let _ = feed(&b, &ids(70, 9), 6);
    assert!(b.sums() != s1);
    b.reset();
    let mut all = p1.clone();
    all.extend_from_slice(&ext);
    let ev = SnapshotHook::<CompactBackend>::restore(&mut hook, &b, &all[..all.len() - 1]).unwrap();
    assert!(ev.error.is_none(), "{:?}", ev.error);
    assert_eq!((ev.n, b.pos(), b.held()), (96, 96, 96));
    assert!(b.sums() == s1, "restored sums != in-memory sums after the prompt");
    let tb = feed(&b, &ext, nn);
    assert!(b.sums() == sa, "restore + extension: sums differ from in memory");
    assert_eq!(b.held(), ha);
    assert!(*ea.ffn_compact_set().unwrap() == seta, "restore + extension: set differs from in memory");
    assert_eq!(tb, ta, "restore + extension: tokens differ from in memory");
    eprintln!("restore + extension == in memory: sums / held {} / set / {} tokens", ha, tb.len());
    let _ = std::fs::remove_dir_all(&cd);
}

// ---- 2. worker over HTTP: snapshots change no reply

fn start(tiny: String, tok: String, snaps: Option<PathBuf>, log: String) -> u16 {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(&tok).unwrap());
        let rule = rule_for(&tiny);
        let eng = QwenEngine::open(&tiny, opts(&rule, 64, 4096)).unwrap();
        let vocab = eng.cfg.vocab as u32;
        let b = CompactBackend::new(&eng, rule, json!({"mode": "12gb"})).unwrap();
        let hook = snaps.map(|d| Box::new(hook_for(&b, &d, &tiny)) as Box<dyn SnapshotHook<CompactBackend>>);
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: false, fold: Some(vocab) }, ..Default::default() };
        streamdense_gateway::serve(&b, tk, cfg, hook, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    rx.recv_timeout(Duration::from_secs(120)).unwrap()
}

fn body(user: &str) -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut b = v["body"].clone();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(12);
    if !user.is_empty() { b["messages"] = json!([{"role": "user", "content": user}]); }
    b
}

fn post(port: u16, b: &Value) -> String {
    let r = request(port, "POST", "/v1/messages", Some(b)).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()["content"].to_string()
}

fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}
fn kinds(l: &Value) -> Vec<String> { l["snap"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect() }

#[test]
fn compact_gateway_snapshots_change_no_reply() {
    let Some(tiny) = dir() else { return };
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !Path::new(&format!("{}/tokenizer.json", tok)).exists() { eprintln!("SKIP: 27B tokenizer.json missing"); return; }
    let al = { let r = rule_for(&tiny); let e = QwenEngine::open(&tiny, opts(&r, 64, 256)).unwrap(); CompactBackend::new(&e, r, json!({})).unwrap().align };
    let cd = tmp("gw");
    let (log_s, log_n) = (tmp("gw_s.jsonl").display().to_string(), tmp("gw_n.jsonl").display().to_string());
    let ps = start(tiny.clone(), tok.clone(), Some(cd.clone()), log_s.clone());
    let pn = start(tiny.clone(), tok.clone(), None, log_n.clone());
    let (b1, b3) = (body(""), body("Something else entirely."));

    // 1. cold: full read with two cuts (stable prefix, rolling state), both on whole chunks
    let r1 = post(ps, &b1);
    assert_eq!(r1, post(pn, &b1), "cuts changed the reply");
    let l = log_lines(&log_s);
    assert_eq!(l[0]["mode"], "full", "{}", l[0]);
    assert_eq!(kinds(&l[0]), ["save_stable", "save_roll"], "{}", l[0]);
    let n_prompt = l[0]["n_prompt"].as_u64().unwrap() as usize;
    let (n_stable, n_roll) = (l[0]["snap"][0]["n"].as_u64().unwrap() as usize, l[0]["snap"][1]["n"].as_u64().unwrap() as usize);
    assert!(n_stable % al == 0 && n_roll % al == 0, "cuts {} / {} not on {}-token chunks", n_stable, n_roll, al);
    assert!(n_roll <= n_prompt - 1 && n_prompt - 1 - n_roll < al && n_stable < n_roll, "{} {} {}", n_stable, n_roll, n_prompt);
    assert_eq!(l[0]["ram_mode"]["stream"]["tokens"].as_u64().unwrap() as usize, n_prompt, "every prompt token read once");

    // 2. re-send: restore of the rolling state, the rest read seeded with its sums
    let r2 = post(ps, &b1);
    assert_eq!(r2, post(pn, &b1), "restore (rolling) changed the reply");
    assert_eq!(r2, r1);
    let l = log_lines(&log_s);
    assert_eq!(l[1]["mode"], "restore", "{}", l[1]);
    assert_eq!(kinds(&l[1]), ["restore"], "{}", l[1]);
    assert_eq!(l[1]["n_reused"].as_u64().unwrap() as usize, n_roll);

    // 3. new conversation, same system + tools: restore of the stable prefix
    let r3 = post(ps, &b3);
    assert_eq!(r3, post(pn, &b3), "restore (stable) changed the reply");
    let l = log_lines(&log_s);
    assert_eq!(l[2]["mode"], "restore", "{}", l[2]);
    assert_eq!(l[2]["n_reused"].as_u64().unwrap() as usize, n_stable);
    for f in std::fs::read_dir(&cd).unwrap().flatten() {
        let h = SnapHeader::read_file(&f.path()).unwrap();
        assert_eq!(h.format, FORMAT_COMPACT, "{}", f.path().display());
        assert_eq!(h.compact.as_ref().unwrap().held, h.pos);
    }
    eprintln!("prompt {} tokens: cuts {} / {} (chunks of {}), replies equal to the gateway without snapshots", n_prompt, n_stable, n_roll, al);
    let _ = std::fs::remove_dir_all(&cd);
    let _ = std::fs::remove_file(&log_s);
    let _ = std::fs::remove_file(&log_n);
}

// ---- 3. format compatibility + refusals

#[test]
fn snapshot_formats_and_refusals() {
    let Some(d) = dir() else { return };
    let rule = rule_for(&d);
    let cd = tmp("fmt");
    std::fs::create_dir_all(&cd).unwrap();
    let toks = ids(40, 3);

    // full mode: format 1, the old header keys, no trailing section; loads into a fresh engine
    let ef = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, ..Default::default() }).unwrap();
    ef.reset();
    ef.prefill(&toks).unwrap();
    let pf = cd.join("full.sdsnap");
    ef.save_snapshot(&pf, &toks).unwrap();
    let raw = std::fs::read(&pf).unwrap();
    let hl = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    let hj: Value = serde_json::from_slice(&raw[16..16 + hl]).unwrap();
    let mut keys: Vec<&str> = hj.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    keys.sort();
    assert_eq!(keys, ["body", "engine", "fingerprint", "format", "kv", "mtp", "mtp_len", "pos", "tokens"], "full-mode header changed");
    assert_eq!(hj["format"], json!(FORMAT));
    assert_eq!(raw.len() as u64, 16 + hl as u64 + hj["body"].as_u64().unwrap(), "full-mode file has a trailing section");
    let h = SnapHeader::read_file(&pf).unwrap();
    assert!(h.compact.is_none());
    let e2 = QwenEngine::open(&d, QwenOptions { max_ctx: 1024, ..Default::default() }).unwrap();
    assert_eq!(e2.restore_snapshot(&pf).unwrap(), toks);
    let s = Snapshot::read(&pf).unwrap();
    e2.reset();
    e2.restore(&s).unwrap();
    assert_eq!(e2.pos(), toks.len());
    drop(e2);

    // 12 GB mode: format 2 (+ sums); refuses the full-mode file before touching anything
    let ea = QwenEngine::open(&d, opts(&rule, 64, 1024)).unwrap();
    let b = CompactBackend::new(&ea, rule.clone(), json!({})).unwrap();
    b.reset();
    let _ = feed(&b, &toks, 1);
    let pc = cd.join("compact.sdsnap");
    b.save_snapshot(&pc, &toks).unwrap();
    let hc = SnapHeader::read_file(&pc).unwrap();
    let m = hc.compact.clone().unwrap();
    assert_eq!((hc.format, m.held, m.align, m.layers, m.width), (FORMAT_COMPACT, 40, b.align, ea.cfg.layers, 2 * ea.cfg.inter));
    let rc = std::fs::read(&pc).unwrap();
    let hlc = u64::from_le_bytes(rc[8..16].try_into().unwrap());
    assert_eq!(hc.body, h.body, "same state body as the full-mode file");
    assert_eq!(rc.len() as u64, 16 + hlc + hc.body + m.sums_bytes(), "format 2 = header + format-1 body + sums");
    let mode_c = b.snap_mode();
    let e = b.restore_snapshot(&pf).unwrap_err();
    assert!(e.contains("full model") && e.contains("refused"), "{}", e);
    assert_eq!((b.pos(), b.held()), (0, 0), "a refused restore leaves the backend reset");
    // ... and the full model refuses the 12 GB-mode file (file + in-RAM paths)
    let e = ef.restore_snapshot(&pc).unwrap_err();
    assert!(e.contains("12 GB mode") && e.contains("refused"), "{}", e);
    let e = ef.restore(&Snapshot::read(&pc).unwrap()).unwrap_err();
    assert!(e.contains("12 GB mode") && e.contains("refused"), "{}", e);
    // the 12 GB-mode file restores here
    assert_eq!(b.restore_snapshot(&pc).unwrap(), toks);
    assert_eq!((b.pos(), b.held()), (40, 40));
    drop(b);
    drop(ea);

    // another alignment (fast read, prefill_t 32) is refused
    let fo = QwenOptions { ffn_stream_fast: true, prefill_t: 32, ..opts(&rule, 64, 1024) };
    let ex = QwenEngine::open(&d, fo).unwrap();
    let bx = CompactBackend::new(&ex, rule.clone(), json!({})).unwrap();
    assert_ne!(bx.snap_mode(), mode_c);
    let e = bx.restore_snapshot(&pc).unwrap_err();
    assert!(e.contains("another 12 GB-mode setup") && e.contains("refused"), "{}", e);

    // cache keys: the full-mode cache never offers the 12 GB-mode file and the reverse
    let mid = CacheHook::model_id(&d);
    let fullc = SnapshotCache::open(cd.join("cache"), &mid).unwrap();
    let pff = fullc.save(&ef, &toks).unwrap();
    let mut cc = SnapshotCache::open(cd.join("cache"), &mid).unwrap();
    cc.mode = bx.snap_mode();
    assert_ne!(cc.key_prefix(&ex), fullc.key_prefix(&ef));
    assert!(!fullc.key_prefix(&ef).starts_with(&cc.key_prefix(&ex)) && !cc.key_prefix(&ex).starts_with(&fullc.key_prefix(&ef)));
    let mut hook = CacheHook::new(cc, 8, 1 << 30);
    assert!(SnapshotHook::<CompactBackend>::restore(&mut hook, &bx, &toks).is_none(), "12 GB-mode cache offered a full-mode file");
    assert!(pff.exists(), "the full-mode file is left alone");
    assert_eq!(fullc.longest_prefix(&ef, &toks).map(|x| x.1), Some(40));
    eprintln!("formats: full = 1 (unchanged), 12 GB mode = 2 (+{} B sums); refusals both ways + other alignment", m.sums_bytes());
    let _ = std::fs::remove_dir_all(&cd);
}
