//! `streamdense gateway` in 12 GB mode ([forheader] mode = "12gb") end to end over real HTTP on the tiny model
//! (models/tiny_qwen35, vocab 2048) with the 27B tokenizer and --fold-ids, a synthetic global score file, and a cap
//! that keeps about half the FFN neurons (K read from the ready line). Never loads the 27B weights; --port 0.
//! 1. a request == the two-process reference: full engine prefill (recorder sums + per-token gate scores) + snapshot,
//!    set built by the same rule, a compact engine with that set restores the snapshot and decodes greedily;
//! 2. a 1-token request (prompt length a multiple of 8) then its extension (planner prefix / followup: only the new
//!    tokens are stream-read, seeded with the held sums) == one streamed read of the whole sequence;
//! 3. status: ready line / health (mode, K, resident GB), per-request log (stream bytes / time), snapshots on (temp
//!    --snapshot-dir; these short prompts make no cut), no neuron store in the watcher dir (no recorder), MTP
//!    off although --mtp is passed ([forheader] mtp = false).
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use streamdense_gateway::chatmap::{parse_tool_calls, render, reply_text};
use streamdense_gateway::client::request;
use streamdense_gateway::planner::KvState;
use streamdense_model::ffn_compact::CompactSet;
use streamdense_model::ffn_select::FselMode;
use streamdense_model::ffn_set::{SetRule, T1};
use streamdense_model::qwen_config::QwenConfig;
use streamdense_model::{QwenEngine, QwenOptions};
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const STREAM: usize = 64;

struct Gw { child: Child, port: u16, ready: Value, files: Vec<std::path::PathBuf> }
impl Drop for Gw {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for f in &self.files { let _ = std::fs::remove_file(f); let _ = std::fs::remove_dir_all(f); }
    }
}

fn t1(l: usize, n: usize) -> T1 {
    let mut s = 0x5DEE_CE66_DBA5_1D3Bu64;
    let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    T1 { layers: l, inter: n, global: (0..l * n).map(|_| r() / n as f32).collect(), down_norm: (0..l * n).map(|_| 0.5 + r()).collect() }
}

/// The gateway's engine options in 12 GB mode for this test (the keep formula input), K = None (any set).
fn opts(stream: usize) -> QwenOptions {
    QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, mtp: false, ffn_stream_tokens: stream, ffn_stream_slots: 2,
                  ffn_stream_host: true, ..Default::default() }
}

/// `tag` keeps the temp files of the tests in this binary apart; `extra` = more [forheader] lines, `flags` = more
/// gateway flags.
fn start(tiny: &str, tok: &str, t1p: &std::path::Path, cap_gb: f64, tag: &str, extra: &str, flags: &[&str]) -> Gw {
    let pid = format!("{}{}", std::process::id(), tag);
    let log = std::env::temp_dir().join(format!("sd_gw12_{}.jsonl", pid));
    let wdir = std::env::temp_dir().join(format!("sd_gw12_watcher_{}", pid));
    let sdir = std::env::temp_dir().join(format!("sd_gw12_snaps_{}", pid));
    let _ = std::fs::remove_dir_all(&wdir);
    let _ = std::fs::remove_dir_all(&sdir);
    let conf = std::env::temp_dir().join(format!("sd_gw12_{}.toml", pid));
    let _ = std::fs::remove_file(&log);
    let st = if extra.contains("stream_tokens") { String::new() } else { format!("stream_tokens = {}\n", STREAM) };
    std::fs::write(&conf, format!("[forheader]\nmode = \"12gb\"\ncap_gb = {}\n{}t1 = \"{}\"\n{}", cap_gb, st, t1p.display(), extra)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_streamdense"))
        .args(["gateway", "--model", tiny, "--tokenizer", tok, "--max-ctx", "1024", "--port", "0", "--fold-ids", "--mtp", "--log", &log.display().to_string(),
               "--watcher-dir", &wdir.display().to_string(), "--snapshot-dir", &sdir.display().to_string()]).args(flags)
        .env("SD_CONFIG", &conf).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let out = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { for l in BufReader::new(out).lines() { let Ok(l) = l else { break }; let _ = tx.send(l); } });
    let line = rx.recv_timeout(Duration::from_secs(120)).expect("gateway: no ready line within 120 s");
    let ready: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("ready line {:?}: {}", line, e));
    assert_eq!(ready["ready"], json!(true), "{}", ready);
    Gw { child, port: ready["port"].as_u64().unwrap() as u16, ready, files: vec![log, wdir, conf, sdir] }
}

fn log_lines(g: &Gw) -> Vec<Value> {
    std::fs::read_to_string(&g.files[0]).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// Rendered prompt ids of a request body (unfolded), as the gateway's worker builds them.
fn prompt_ids(tk: &Tokenizer, body: &Value) -> (Vec<u32>, Vec<streamdense_gateway::chatmap::ChatMsg>) {
    let cr = streamdense_gateway::anthropic::convert(body).unwrap();
    (tk.encode(&render(&cr.msgs, &cr.tools, false, true).unwrap()), cr.msgs)
}

/// Visible reply text of `toks` as the gateway reports it (non-stream JSON content).
fn visible(tk: &Tokenizer, toks: &[u32]) -> String { parse_tool_calls(&reply_text(&tk.decode(toks), false)).0 }

/// Greedy tokens up to `n` or the model's eos (included), as the engine's decode loop stops.
fn cut_eos(t: &[u32], eos: &[u32], n: usize) -> Vec<u32> {
    let mut v = vec![];
    for &x in t.iter().take(n) { v.push(x); if eos.contains(&x) { break; } }
    v
}

fn reply(port: u16, body: &Value) -> (String, u64) {
    let r = request(port, "POST", "/v1/messages", Some(body)).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let j = r.json();
    let text = j["content"].as_array().unwrap().iter().filter(|b| b["type"] == "text").map(|b| b["text"].as_str().unwrap().to_string()).collect::<String>();
    (text, j["usage"]["output_tokens"].as_u64().unwrap())
}

#[test]
fn gateway_12gb_tiny_http() {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", tok)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer.json missing");
        return;
    }
    let cfg = QwenConfig::from_model_dir(&tiny).unwrap();
    let (l, n, vocab) = (cfg.layers, cfg.inter, cfg.vocab as u32);
    let t1p = std::env::temp_dir().join(format!("sd_gw12_{}.sdt1", std::process::id()));
    let t = t1(l, n);
    t.write(&t1p).unwrap();
    // a cap that keeps about half the neurons: the formula's own terms at a large cap, then rest + fixed + K x per neuron
    let p = QwenEngine::compact_keep_for_cap(&tiny, &opts(STREAM), 16_000_000_000).unwrap();
    assert_eq!(p.k, n / 8 * 8);
    let cap = p.rest as f64 + p.ffn_fixed + (l * (n / 2 + 4)) as f64 * p.per_neuron;
    // the exact prompt read (fast = false) so request A equals the exact two-process reference; the fused decode stays
    // on (its default; bit-identical), so the reference checks it through the gateway too
    let mut g = start(&tiny, &tok, &t1p, cap / 1e9, "", "fast = false\n", &[]);
    g.files.push(t1p.clone());
    let port = g.port;
    let fh = &g.ready["forheader"];
    assert_eq!(fh["mode"], "12gb", "{}", g.ready);
    assert_eq!((&fh["fast"], &fh["fused"]), (&json!(false), &json!(true)), "{}", g.ready);
    let k = fh["k"].as_u64().unwrap() as usize;
    assert!(k >= 8 && k < n && k % 8 == 0, "K {} of {}", k, n);
    assert!((k as i64 - (n / 2) as i64).abs() <= 64, "K {} far from half of {}", k, n);
    assert_eq!(g.ready["mtp_decode"], json!(false), "[forheader] mtp = false wins over --mtp");
    assert_eq!(g.ready["snapshots"], json!(true));
    // no recorder: the watcher keeps its text state only, no neuron store / predictor files
    assert!(g.files[1].is_dir(), "watcher dir {}", g.files[1].display());
    assert!(!g.files[1].join("store").exists(), "12 GB mode created a neuron store in {}", g.files[1].display());
    assert_eq!(g.ready["watcher"]["activity"], "text features only");
    let rule = SetRule::new(t, k).unwrap();
    assert_eq!(fh["kres"].as_u64().unwrap() as usize, rule.kres);

    // health: mode, K, resident + live footprint
    let h = request(port, "GET", "/health", None).unwrap().json();
    assert_eq!(h["ram_mode"]["mode"], "12gb", "{}", h);
    assert_eq!(h["ram_mode"]["k"].as_u64(), Some(k as u64));
    assert!(h["ram_mode"]["resident_gb"].as_f64().unwrap() > 0.0 && h["ram_mode"]["footprint_gb"].as_f64().unwrap() > 0.0, "{}", h);
    assert!(h["ram_mode"]["peak_footprint_gb"].as_f64().unwrap() >= h["ram_mode"]["resident_gb"].as_f64().unwrap() - 0.05, "{}", h);
    assert!(h["ram_mode"]["planned_gb"].as_f64().unwrap() <= cap / 1e9 + 1e-9, "{}", h);
    assert_eq!(h["info"]["forheader"]["mode"], "12gb");

    let tk = Tokenizer::from_path(&tok).unwrap();
    let fold = |v: &[u32]| -> Vec<u32> { v.iter().map(|x| x % vocab).collect() };
    let nn = 12;

    // ---- 1. request A vs the two-process reference
    let ba = json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": "Write a short poem about the sea and the wind."}], "max_tokens": nn});
    let (text_a, out_a) = reply(port, &ba);
    let (pa, _) = prompt_ids(&tk, &ba);
    let pa = fold(&pa);
    // B: full engine; recorder sums (an) + per-token gate scores (ag, the gate-score hook in mode All from row 0), snapshot
    let full = QwenEngine::open(&tiny, QwenOptions { max_ctx: 1024, prefill_exact: true, ffn_select: true, activity: true, ..Default::default() }).unwrap();
    let tap = full.activity_tap().unwrap();
    full.fsel_set(FselMode::All, 0, &vec![n as u32; l]).unwrap();
    full.reset();
    let mut ag = vec![0f32; l * n];
    for i in 0..pa.len() {
        full.prefill(&pa[i..i + 1]).unwrap();
        for (x, y) in ag.iter_mut().zip(full.fsel_gate_scores().unwrap()) { *x += *y; }
    }
    full.reset();
    full.fsel_set(FselMode::All, usize::MAX >> 1, &vec![n as u32; l]).unwrap();
    full.activity_reset();
    let tok_b = full.prefill(&pa).unwrap();
    let an: Vec<f32> = tap.with_live(|v| v[..l * n].to_vec());
    let snap = full.snapshot(&pa).unwrap();
    drop(full);
    let set = CompactSet { inter: n, keep: (0..l).map(|li| rule.layer(li, &an[li * n..(li + 1) * n], &ag[li * n..(li + 1) * n])).collect() };
    // C: compact engine with that set restores B's state and decodes
    let c = QwenEngine::open(&tiny, QwenOptions { max_ctx: 1024, prefill_exact: true, activity: false, ffn_compact: Some(Arc::new(set)), ..Default::default() }).unwrap();
    c.reset();
    c.restore(&snap).unwrap();
    let mut tc = vec![tok_b];
    while tc.len() < nn { tc.push(c.fsel_step(*tc.last().unwrap()).unwrap().0); }
    drop(c);
    let tc = cut_eos(&tc, &cfg.eos, nn);
    assert_eq!(out_a as usize, tc.len());
    assert_eq!(text_a, visible(&tk, &tc), "12 GB gateway != two-process reference");
    eprintln!("request A == two-process reference ({} tokens)", tc.len());

    // ---- 2. a 1-token request (prompt length a multiple of 8), then its extension
    let mut word = String::from("Explain how tides work");
    let (mut bb1, mut pb1, mut msgs1);
    loop {
        bb1 = json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": word}], "max_tokens": 1});
        (pb1, msgs1) = prompt_ids(&tk, &bb1);
        if pb1.len() % 8 == 0 { break; }
        word.push_str(" ok");
    }
    let (text_b1, out_b1) = reply(port, &bb1);
    assert_eq!(out_b1, 1);
    let r = QwenEngine::open(&tiny, QwenOptions { ffn_compact: Some(Arc::new(rule.initial())), ..opts(512) }).unwrap();
    let one_read = |ids: &[u32], n_tok: usize| -> Vec<u32> {
        r.reset();
        let st = r.stream_prefill(ids, &mut |li, a, g| Ok(rule.layer(li, a, g))).unwrap();
        let mut t = vec![st.tok];
        while t.len() < n_tok { t.push(r.fsel_step(*t.last().unwrap()).unwrap().0); }
        cut_eos(&t, &cfg.eos, n_tok)
    };
    let t0 = one_read(&fold(&pb1), 1)[0];
    assert_eq!(text_b1, visible(&tk, &[t0]));
    let mut bb2 = bb1.clone();
    bb2["messages"].as_array_mut().unwrap().push(json!({"role": "assistant", "content": text_b1}));
    bb2["messages"].as_array_mut().unwrap().push(json!({"role": "user", "content": "And the moon?"}));
    bb2["max_tokens"] = json!(nn);
    let (text_b2, out_b2) = reply(port, &bb2);
    // the sequence the engine holds after request 2's prompt read: request 1's prompt + its token + the planned tokens
    let (full2, msgs2) = prompt_ids(&tk, &bb2);
    let mut seq = pb1.clone();
    seq.push(t0);
    let kv = KvState { seq: seq.clone(), msgs: msgs1, tools: vec![], think: false };
    let plan = kv.plan(&msgs2, &[], false, full2, &|m, gp| render(m, &[], false, gp), &|s| tk.encode(s));
    seq.extend_from_slice(&plan.tokens);
    let tr = one_read(&fold(&seq), nn);
    assert_eq!(out_b2 as usize, tr.len());
    assert_eq!(text_b2, visible(&tk, &tr), "prefix extension != one streamed read of the whole sequence");
    eprintln!("1-token request + extension ({}) == one read ({} tokens)", plan.mode.as_str(), tr.len());

    // ---- 3. request log: stream stats per request, no snapshot cut / restore for these short prompts, reuse on the
    // extension; after 3 jobs still no neuron store (the text watcher saw them)
    assert!(!g.files[1].join("store").exists() && g.files[1].join("state.json").is_file(), "{}", g.files[1].display());
    let lg = log_lines(&g);
    assert_eq!(lg.len(), 3, "{:?}", lg);
    for x in &lg {
        let s = &x["ram_mode"]["stream"];
        assert_eq!(x["ram_mode"]["mode"], "12gb", "{}", x);
        assert_eq!(x["ram_mode"]["k"].as_u64(), Some(k as u64));
        assert!(s["ssd_gb"].as_f64().unwrap() > 0.0 && s["wall_s"].as_f64().unwrap() > 0.0 && s["pieces"].as_u64().unwrap() >= 1, "{}", x);
        assert_eq!(s["tokens"], x["n_feed"], "{}", x);
        assert_eq!(x["snap"], json!([]), "{}", x);
        assert_eq!(x["mtp"], json!(false));
    }
    assert_eq!(lg[0]["mode"], "full");
    assert!(lg[2]["mode"] == "prefix" || lg[2]["mode"] == "followup", "{}", lg[2]);
    assert_eq!(lg[2]["mode"], plan.mode.as_str());
    assert!(lg[2]["n_reused"].as_u64().unwrap() == pb1.len() as u64, "{}", lg[2]);
    assert!(lg[0]["ram_mode"]["stream"]["pieces"].as_u64().unwrap() as usize == pa.len().div_ceil(STREAM), "{}", lg[0]);

    // ---- 4. snapshots through the CLI wiring: a long single-turn prompt saves one state (its first turn = the "stable"
    // prefix, cut on a whole chunk; 12 GB-mode file, own key), its
    // re-send restores it and gives the same reply
    let bl = json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": "Describe the water cycle in detail. ".repeat(60)}], "max_tokens": 6});
    let (r1, _) = reply(port, &bl);
    let (r2, _) = reply(port, &bl);
    let lg = log_lines(&g);
    let kinds = |l: &Value| l["snap"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    assert_eq!(kinds(&lg[3]).len(), 1, "{}", lg[3]);
    assert_eq!(lg[3]["snap"][0]["n"].as_u64().unwrap() % 8, 0, "cut on a whole chunk: {}", lg[3]);
    assert!(lg[3]["snap"][0]["error"].is_null(), "{}", lg[3]);
    assert_eq!(lg[4]["mode"], "restore", "{}", lg[4]);
    assert_eq!(lg[4]["n_reused"], lg[3]["snap"][0]["n"], "{}", lg[4]);
    assert_eq!(r1, r2, "restored 12 GB-mode state gave another reply");
    // snapshot files only (the tier index tiers.json lives beside them)
    let files: Vec<String> = std::fs::read_dir(&g.files[3]).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".sdsnap")).collect();
    assert!(files.len() == 1 && files[0].contains("f2c") && files[0].ends_with(".sdsnap"), "{:?}", files);
}

/// Defaults (2026-11-03): the 12 GB mode runs the fast prompt read and the fused decode unless [forheader] fast / fused
/// = false or --no-fast / --no-fused. With the defaults a request answers, and a long prompt saves a snapshot (cut on the
/// fast read's alignment) whose re-send restores it with the same reply; the flags / config keys turn both off; stream
/// pieces shorter than prefill_t (128) fall back to the exact read.
#[test]
fn gateway_12gb_tiny_defaults() {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", tok)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer.json missing");
        return;
    }
    let cfg = QwenConfig::from_model_dir(&tiny).unwrap();
    let (l, n) = (cfg.layers, cfg.inter);
    let t1p = std::env::temp_dir().join(format!("sd_gw12_{}d.sdt1", std::process::id()));
    t1(l, n).write(&t1p).unwrap();
    let fo = QwenOptions { ffn_stream_fast: true, ..opts(128) };
    let p = QwenEngine::compact_keep_for_cap(&tiny, &fo, 16_000_000_000).unwrap();
    let cap = p.rest as f64 + p.ffn_fixed + (l * (n / 2 + 4)) as f64 * p.per_neuron;
    {
        let g = start(&tiny, &tok, &t1p, cap / 1e9, "d", "stream_tokens = 128\n", &[]);
        let fh = &g.ready["forheader"];
        assert_eq!((&fh["fast"], &fh["fused"]), (&json!(true), &json!(true)), "defaults: {}", g.ready);
        let (text, out) = reply(g.port, &json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": "Write a short poem about the sea and the wind."}], "max_tokens": 12}));
        assert!(out > 0, "no output tokens: {:?}", text);
        let bl = json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": "Describe the water cycle in detail. ".repeat(60)}], "max_tokens": 6});
        let (r1, _) = reply(g.port, &bl);
        let (r2, _) = reply(g.port, &bl);
        let lg = log_lines(&g);
        assert_eq!(lg.len(), 3, "{:?}", lg);
        assert_eq!(lg[1]["snap"].as_array().map(|a| a.len()), Some(1), "{}", lg[1]);
        assert!(lg[1]["snap"][0]["error"].is_null(), "{}", lg[1]);
        assert_eq!(lg[2]["mode"], "restore", "{}", lg[2]);
        assert_eq!(r1, r2, "restored fast-read state gave another reply");
    }
    {
        let g = start(&tiny, &tok, &t1p, cap / 1e9, "n", "", &["--no-fast", "--no-fused"]);
        let fh = &g.ready["forheader"];
        assert_eq!((&fh["fast"], &fh["fused"]), (&json!(false), &json!(false)), "--no-fast --no-fused: {}", g.ready);
    }
    {
        let g = start(&tiny, &tok, &t1p, cap / 1e9, "c", "fast = false\nfused = false\n", &[]);
        let fh = &g.ready["forheader"];
        assert_eq!((&fh["fast"], &fh["fused"]), (&json!(false), &json!(false)), "config off: {}", g.ready);
    }
    {
        let g = start(&tiny, &tok, &t1p, cap / 1e9, "s", "", &[]); // stream_tokens 64 < prefill_t 128
        let fh = &g.ready["forheader"];
        assert_eq!((&fh["fast"], &fh["fused"]), (&json!(false), &json!(true)), "short pieces: {}", g.ready);
    }
    let _ = std::fs::remove_file(&t1p);
}
