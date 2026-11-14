//! `streamdense gateway --dflash2 DIR`: refusals through --plan (nothing loaded, no port bound) and one allowed run on the
//! tiny model (models/tiny_qwen35, a random drafter) over real HTTP with --port 0: the text equals a plain gateway's, a
//! failed draft ends that request with an error, the next request works. The 27B tokenizer.json is read (as
//! gateway_http.rs); `refuse_real_budget` hashes and reads the headers of the real drafter, no weights enter a model.
//! Never touches 8898, ~/.streamdense or the snapshot cache: scratch dirs only; only children spawned here are killed.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use streamdense_core::{write_safetensors, SafeIndex};
use streamdense_gateway::client::{request, Response};
use streamdense_model::{dflash2, QwenConfig};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tiny() -> String { format!("{}/models/tiny_qwen35", ROOT) }
fn tok() -> String { format!("{}/models/qwen38_27b_mlx4", ROOT) }
fn have_tiny() -> bool {
    let ok = Path::new(&format!("{}/config.json", tiny())).exists() && Path::new(&format!("{}/tokenizer.json", tok())).exists();
    if !ok { eprintln!("SKIP: tiny model or 27B tokenizer.json missing"); }
    ok
}
struct Tmp(PathBuf);
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("sd_gw_df2_{}_{}_{}", tag, std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}
fn drafter(base: &Path) -> PathBuf {
    let d = base.join("drafter");
    dflash2::write_random(&d, &QwenConfig::from_model_dir(&tiny()).unwrap(), 2, 8, 2048, 64, 7).unwrap();
    d
}
fn sha(p: &Path) -> String {
    let o = Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(p).output().unwrap();
    String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap().to_string()
}
fn conf(base: &Path, body: &str) -> String {
    let c = base.join(format!("conf_{}.toml", SEQ.fetch_add(1, Ordering::SeqCst)));
    std::fs::write(&c, format!("# gateway_dflash2 test\n{}\n", body)).unwrap();
    c.display().to_string()
}
/// A --plan run: (exit ok, stderr).
fn plan(c: &str, args: &[&str]) -> (bool, String) {
    let mut a = vec!["gateway", "--plan", "--port", "0", "--max-ctx", "1024"];
    a.extend_from_slice(args);
    let o = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(&a).env("SD_CONFIG", c).stdin(Stdio::null()).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(!err.contains("opened in") && !err.contains("listening"), "--plan started the engine: {}", err);
    assert!(o.stdout.is_empty(), "--plan wrote a ready line: {}", String::from_utf8_lossy(&o.stdout));
    (o.status.success(), err)
}
fn refused(c: &str, args: &[&str], want: &str) -> String {
    let (ok, err) = plan(c, args);
    assert!(!ok, "not refused: {:?}\n{}", args, err);
    assert!(err.contains(want), "{:?}: want {:?} in\n{}", args, want, err);
    assert!(!err.contains("--plan: budget only"), "{}", err);
    let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").to_string();
    println!("REFUSAL {}", last);
    last
}

#[test]
fn refusals_and_flags_plan_only() {
    if !have_tiny() { return; }
    let t = tmp("plan");
    let d = drafter(&t.0);
    let ds = d.display().to_string();
    let good = sha(&d.join("model.safetensors"));
    let c = conf(&t.0, &format!("[dflash2]\nexpected_sha256 = \"{}\"", good));
    let tn = tiny();
    // allowed: the rows show, the note shows, MTP gives way
    let (ok, err) = plan(&c, &["--model", &tn, "--dflash2", &ds, "--mtp"]);
    assert!(ok, "{}", err);
    for s in ["DFlash2 drafter weights", "DFlash2 context KV", "--plan: budget only", "experimental", "MTP off for this run", "checking the sha256"] { assert!(err.contains(s), "{}: {}", s, err); }
    // through the config; --no-dflash2 wins and then no file is looked at
    let cc = conf(&t.0, &format!("[gateway]\ndflash2 = \"{}\"\n[dflash2]\nexpected_sha256 = \"{}\"", ds, good));
    let (ok, err) = plan(&cc, &["--model", &tn]);
    assert!(ok && err.contains("DFlash2 drafter weights"), "{}", err);
    let (ok, err) = plan(&cc, &["--model", &tn, "--no-dflash2"]);
    assert!(ok && !err.contains("DFlash2") && !err.contains("sha256"), "{}", err);
    // off by default
    let c0 = conf(&t.0, "");
    let (ok, err) = plan(&c0, &["--model", &tn]);
    assert!(ok && !err.to_ascii_lowercase().contains("dflash2"), "{}", err);
    // refusals
    refused(&c, &["--model", &tn, "--dflash2", &ds, "--drafter", &ds], "DFlash2 refused: --dflash2 and --drafter cannot be combined");
    refused(&c, &["--model", &tn, "--dflash2", &ds, "--forheader", "12gb"], "DFlash2 refused: --dflash2 cannot be combined with the 12 GB mode");
    refused(&c0, &["--model", &tn, "--dflash2", &ds], "sha256 mismatch");
    let wrong = "0".repeat(64);
    refused(&conf(&t.0, &format!("[dflash2]\nexpected_sha256 = \"{}\"", wrong)), &["--model", &tn, "--dflash2", &ds], "sha256 mismatch");
    refused(&conf(&t.0, "[dflash2]\nexpected_sha256 = \"abc\""), &["--model", &tn], "want 64 hex digits");
    refused(&c, &["--model", &tn, "--dflash2", &ds, "--abort-gb", "0.1"], "DFlash2 refused: the budget needs");
    // missing file
    let miss = t.0.join("nothing");
    refused(&c, &["--model", &tn, "--dflash2", &miss.display().to_string()], "DFlash2 refused: cannot check the weights");
    // the right file with a wrong shape (the hash fits, the tensors do not): config says rank 128
    let bad = t.0.join("badshape");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::copy(d.join("model.safetensors"), bad.join("model.safetensors")).unwrap();
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    v["dflash_config"]["selector_rank"] = json!(128);
    std::fs::write(bad.join("config.json"), v.to_string()).unwrap();
    refused(&c, &["--model", &tn, "--dflash2", &bad.display().to_string()], "candidate_selector");
    // a wrong-shape tensor with its own hash
    let bad2 = t.0.join("badfile");
    dflash2::write_random(&bad2, &QwenConfig::from_model_dir(&tn).unwrap(), 2, 8, 2048, 128, 7).unwrap();
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(bad2.join("config.json")).unwrap()).unwrap();
    v["dflash_config"]["selector_rank"] = json!(64);
    std::fs::write(bad2.join("config.json"), v.to_string()).unwrap();
    let c2 = conf(&t.0, &format!("[dflash2]\nexpected_sha256 = \"{}\"", sha(&bad2.join("model.safetensors"))));
    refused(&c2, &["--model", &tn, "--dflash2", &bad2.display().to_string()], "candidate_selector");
}

/// The real target and the real file at the default 16 GB: refused, plain words; the budget rows are printed first.
#[test]
fn refuse_real_budget() {
    let (m, f) = (tok(), format!("{}/models/dflash2_27b_q4", ROOT));
    if !Path::new(&f).join("model.safetensors").exists() || !Path::new(&m).join("config.json").exists() { eprintln!("SKIP: real file"); return; }
    let t = tmp("real");
    let c = conf(&t.0, "");
    let o = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(["gateway", "--plan", "--port", "0", "--model", &m, "--dflash2", &f]).env("SD_CONFIG", &c).stdin(Stdio::null()).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    println!("{}", err);
    assert!(!o.status.success() && err.contains("DFlash2 refused: the budget needs") && err.contains("DFlash2 drafter weights"), "{}", err);
    assert!(!err.contains("opened in") && !err.contains("listening"), "{}", err);
}

struct Gw { child: Child, port: u16, log: PathBuf, ready: Value }
impl Drop for Gw { fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); } }

fn start(base: &Path, model: &str, c: &str, extra: &[&str]) -> Gw {
    let id = SEQ.fetch_add(1, Ordering::SeqCst);
    let log = base.join(format!("log_{}.jsonl", id));
    let (snaps, wdir) = (base.join(format!("snaps_{}", id)), base.join(format!("watcher_{}", id)));
    let mut a: Vec<String> = ["gateway", "--model", model, "--tokenizer", &tok(), "--max-ctx", "1024", "--port", "0", "--fold-ids", "--log"].iter().map(|s| s.to_string()).collect();
    a.push(log.display().to_string());
    a.extend(["--snapshot-dir".to_string(), snaps.display().to_string(), "--watcher-dir".to_string(), wdir.display().to_string()]);
    a.extend(extra.iter().map(|s| s.to_string()));
    let elog = std::fs::File::create(base.join(format!("stderr_{}.log", id))).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(&a).env("SD_CONFIG", c).env("SD_WATCHER_DIR", &wdir)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(elog).spawn().unwrap();
    let out = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { for l in BufReader::new(out).lines() { let Ok(l) = l else { break }; let _ = tx.send(l); } });
    let line = match rx.recv_timeout(Duration::from_secs(120)) {
        Ok(l) => l,
        Err(e) => { let _ = child.kill(); let _ = child.wait(); panic!("no ready line: {}: {}", e, std::fs::read_to_string(base.join(format!("stderr_{}.log", id))).unwrap_or_default()); }
    };
    let ready: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(ready["ready"], json!(true));
    assert_ne!(ready["port"], json!(8898));
    Gw { child, port: ready["port"].as_u64().unwrap() as u16, log, ready }
}
fn logs(g: &Gw) -> Vec<Value> { std::fs::read_to_string(&g.log).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect() }
fn body(text: &str, stream: bool, n: usize) -> Value {
    json!({"model": "qwen38-27b", "system": [{"type": "text", "text": "Be brief."}], "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}], "max_tokens": n, "stream": stream})
}
fn reply(r: &Response) -> String {
    assert_eq!(r.status, 200, "{}", r.text());
    r.json()["content"][0]["text"].as_str().unwrap().to_string()
}

/// A copy of the tiny target whose embedding codes are the file's last tensor, so the last bytes of the file are the
/// mask token's row (vocab - 1): cutting them makes the drafter's row read fail and nothing else.
fn target_with_embed_last(base: &Path) -> (PathBuf, PathBuf, u64) {
    let tn = tiny();
    let m = base.join("target");
    std::fs::create_dir_all(&m).unwrap();
    let file = "model-00001-of-00001.safetensors";
    let idx = SafeIndex::open_file(&format!("{}/{}", tn, file)).unwrap();
    assert!(idx.skipped.is_empty());
    let raw = std::fs::read(format!("{}/{}", tn, file)).unwrap();
    let last = "language_model.model.embed_tokens.weight";
    let mut names: Vec<&str> = idx.names().filter(|n| *n != last).collect();
    names.push(last);
    let views: Vec<(&str, streamdense_core::DType, &[usize], &[u8])> = names.iter().map(|n| {
        let t = idx.get(n).unwrap();
        (*n, t.dtype, t.shape.as_slice(), &raw[t.off as usize..t.off as usize + t.nbytes])
    }).collect();
    let row = idx.get(last).unwrap().nbytes as u64 / idx.get(last).unwrap().shape[0] as u64;
    let w = m.join(file);
    write_safetensors(&w.display().to_string(), &views).unwrap();
    for f in ["config.json", "model.safetensors.index.json"] { std::fs::copy(format!("{}/{}", tn, f), m.join(f)).unwrap(); }
    if Path::new(&format!("{}/generation_config.json", tn)).exists() { std::fs::copy(format!("{}/generation_config.json", tn), m.join("generation_config.json")).unwrap(); }
    std::os::unix::fs::symlink(std::fs::canonicalize(format!("{}/mtp", tn)).unwrap(), m.join("mtp")).unwrap();
    (m, w, row)
}

#[test]
fn allowed_run_failure_and_next_request() {
    if !have_tiny() { return; }
    let t = tmp("run");
    let d = drafter(&t.0);
    let ds = d.display().to_string();
    let c = conf(&t.0, &format!("[dflash2]\nexpected_sha256 = \"{}\"", sha(&d.join("model.safetensors"))));
    let (m, wf, row) = target_with_embed_last(&t.0);
    let ms = m.display().to_string();
    let prompts = ["Hello there", "Write a sorted list in Rust", "What is the capital of France?"];
    // the reference: a plain gateway on the same target
    let plain_texts: Vec<String> = {
        let g = start(&t.0, &ms, &c, &["--no-mtp"]);
        assert_eq!(g.ready["dflash2"], json!({"state": "off"}), "off by default: {}", g.ready);
        assert!(!g.ready["mtp_decode"].as_bool().unwrap_or(true));
        let tx: Vec<String> = prompts.iter().map(|p| reply(&request(g.port, "POST", "/v1/messages", Some(&body(p, false, 24))).unwrap())).collect();
        assert!(request(g.port, "GET", "/health", None).unwrap().json().get("dflash2").is_none(), "off: no dflash2 in /health");
        let lg = logs(&g);
        assert!(lg.iter().all(|l| l.get("dflash2").is_none() && l.get("watcher").is_some()), "off: watcher field at the top level: {:?}", lg);
        tx
    };
    let g = start(&t.0, &ms, &c, &["--dflash2", &ds]);
    assert_eq!(g.ready["dflash2"]["state"], json!("on"), "{}", g.ready);
    assert_eq!(g.ready["dflash2"]["experimental"], json!(true));
    assert_eq!(g.ready["snapshots"], json!(false));
    assert_eq!(g.ready["mtp_available"], json!(false));
    let p = g.port;
    let t1 = reply(&request(p, "POST", "/v1/messages", Some(&body(prompts[0], false, 24))).unwrap());
    assert_eq!(t1, plain_texts[0], "same text as the plain gateway");
    let t2 = reply(&request(p, "POST", "/v1/messages", Some(&body(prompts[1], false, 24))).unwrap());
    assert_eq!(t2, plain_texts[1], "same text as the plain gateway (2nd request)");
    let lg = logs(&g);
    let l = lg.last().unwrap();
    assert_eq!(l["dflash2"]["state"], json!("on"), "{}", l);
    assert!(l["dflash2"]["watcher"].is_string(), "on: watcher summary under dflash2.watcher: {}", l);
    assert!(l.get("watcher").is_none(), "{}", l);
    let st = &l["dflash2"]["stats"];
    assert_eq!(st["errors"], json!(0), "{}", st);
    println!("stats {}", st);
    // forced failure: the drafter's mask token row can no longer be read
    let saved = std::fs::read(&wf).unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(&wf).unwrap();
    f.set_len(saved.len() as u64 - row).unwrap();
    let r = request(p, "POST", "/v1/messages", Some(&body(prompts[2], false, 24))).unwrap();
    assert_eq!(r.status, 500, "{}", r.text());
    assert!(r.text().contains("DFlash2 decode failed, request ended"), "{}", r.text());
    println!("FORCED FAILURE (non-stream) {}", r.text());
    let r = request(p, "POST", "/v1/messages", Some(&body("A second failing prompt", true, 24))).unwrap();
    let ev: Vec<(String, Value)> = r.events();
    assert!(r.status != 200 || ev.iter().any(|(n, _)| n == "error"), "{}", r.text());
    assert!(r.text().contains("DFlash2 decode failed"), "{}", r.text());
    println!("FORCED FAILURE (stream) status {} events {:?}", r.status, ev.iter().map(|x| x.0.clone()).collect::<Vec<_>>());
    let stats = logs(&g).last().unwrap()["dflash2"]["stats"].clone();
    assert_eq!(stats["errors"], json!(2), "{}", stats);
    // the file comes back: the next requests work, same text as plain
    std::io::Write::write_all(&mut std::fs::OpenOptions::new().append(true).open(&wf).unwrap(), &saved[saved.len() - row as usize..]).unwrap();
    assert_eq!(std::fs::read(&wf).unwrap(), saved);
    let t3 = reply(&request(p, "POST", "/v1/messages", Some(&body(prompts[2], false, 24))).unwrap());
    assert_eq!(t3, plain_texts[2], "after the failure: same text as the plain gateway");
    let t1b = reply(&request(p, "POST", "/v1/messages", Some(&body(prompts[0], false, 24))).unwrap());
    assert_eq!(t1b, plain_texts[0]);
    let h = request(p, "GET", "/health", None).unwrap();
    assert_eq!(h.status, 200);
    assert_eq!(h.json()["dflash2"]["state"], json!("on"));
    assert!(h.json()["dflash2"]["watcher"].is_string() && h.json().get("watcher").is_none(), "{}", h.text());
    let s = logs(&g).last().unwrap()["dflash2"]["stats"].clone();
    println!("stats after {}", s);
    assert!(s["total"]["passes"].as_u64().unwrap_or(0) > 0, "{}", s);
}
