//! Domain watcher over real HTTP: `streamdense gateway` on the tiny model (models/tiny_qwen35, --fold-ids, 27B
//! tokenizer; never the 27B weights), a temp config (every job / fold closes the decision window) and a temp --watcher-dir (never
//! ~/.streamdense). Checks the routes, notes, per-job observation (turns, tokens, recorder activity when the recorder is
//! on), the request-log / health field, the files on disk, the audit queue (every reply) and the heat thread.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use streamdense_gateway::client::request;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Gw { child: Child, port: u16, dir: String }
impl Drop for Gw {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); let _ = std::fs::remove_dir_all(&self.dir); }
}

fn start() -> Option<Gw> { start_with("") }

fn start_with(extra: &str) -> Option<Gw> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    let tok = format!("{}/models/qwen38_27b_mlx4", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", tok)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer.json missing");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("sd_watcher_http_{}_{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst))).display().to_string();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let conf = format!("{}/streamdense.toml", dir);
    // + the 27B draft-vocab keep-list (vocab 248,320 vs the tiny model's 2048): must be a warning + off, not a crash
    let keep = format!("{}/models/qwen38_27b_mlx4/draft_vocab.keep", ROOT);
    let gw = if std::path::Path::new(&keep).exists() { format!("[gateway]\ndraft_vocab = \"{}\"\n", keep) } else { String::new() };
    std::fs::write(&conf, format!("{}[watcher]\ndecide_window_folds = 1\ndecide_window_min_turns = 0\nfold_every = 1\n{}", gw, extra)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_streamdense"))
        .args(["gateway", "--model", &tiny, "--tokenizer", &tok, "--max-ctx", "1024", "--port", "0", "--fold-ids", "--no-snapshots",
               "--log", &format!("{}/req.jsonl", dir), "--watcher-dir", &format!("{}/w", dir)])
        .env("SD_CONFIG", &conf).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let out = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { for l in BufReader::new(out).lines() { let Ok(l) = l else { break }; let _ = tx.send(l); } });
    let line = rx.recv_timeout(Duration::from_secs(120)).expect("gateway: no ready line within 120 s");
    let ready: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("ready line {:?}: {}", line, e));
    assert_eq!(ready["draft_vocab"], Value::Null, "a keep-list for another vocab is switched off: {}", ready);
    assert_eq!(ready["ready"], json!(true), "{}", ready);
    Some(Gw { child, port: ready["port"].as_u64().unwrap() as u16, dir })
}

const RUST: &str = "fn main() {\n    let mut v: Vec<u32> = Vec::new();\n    println!(\"{:?}\", v);\n}\nimpl Foo { pub fn bar(&mut self) -> Result<(), String> { Ok(()) } }\n";

#[test]
fn watcher_tiny_http() {
    let Some(g) = start() else { return };
    let p = g.port;
    let st = request(p, "GET", "/v1/streamdense/watcher", None).unwrap();
    assert_eq!(st.status, 200, "{}", st.text());
    let s = st.json();
    assert_eq!(s["domain"], "unknown");
    assert_eq!(s["note"], "dry run: nothing unloaded yet");
    assert_eq!(s["config"]["decide_window_min_turns"], 0);
    assert_eq!(s["config"]["cut_languages"], json!(["zh", "ja", "ko", "tl", "id", "ms"]));
    assert_eq!(s["mode"], "observe");

    // notes: a fold decides at once
    let r = request(p, "POST", "/v1/streamdense/watcher/note", Some(&json!({"kind": "fold", "text": RUST, "project": "/x/engine", "mode": "code"}))).unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let n = r.json();
    assert_eq!(n["domain"], "code-rust", "{}", n);
    assert_eq!(n["decision"]["trigger"], "fold");
    assert_eq!(request(p, "POST", "/v1/streamdense/watcher/note", Some(&json!({"kind": "nope", "text": "x"}))).unwrap().status, 400);
    assert_eq!(request(p, "GET", "/v1/streamdense/watcher/note", None).unwrap().status, 405);
    assert_eq!(request(p, "GET", "/v1/streamdense/other", None).unwrap().status, 404);

    // two jobs: observed after the reply (turns, tokens, activity), one decision each
    let b = json!({"model": "qwen38-27b", "system": "Be brief.", "messages": [{"role": "user", "content": RUST}], "max_tokens": 3, "stream": false});
    for _ in 0..2 { assert_eq!(request(p, "POST", "/v1/messages", Some(&b)).unwrap().status, 200); }
    std::thread::sleep(Duration::from_millis(300)); // the hook runs after the reply went out
    let s = request(p, "GET", "/v1/streamdense/watcher", None).unwrap().json();
    assert_eq!(s["turn"], 2, "{}", s);
    assert!(s["tokens_observed"].as_u64().unwrap() > 0, "{}", s);
    assert_eq!(s["domain"], "code-rust", "{}", s);
    assert_eq!(s["decisions_recent"].as_array().unwrap().len(), 3);
    assert_eq!(s["decisions_recent"][0]["sessions"].as_array().unwrap().len(), 1, "system prompt hash: {}", s["decisions_recent"][0]);
    let jobs = s["activity"]["jobs"].as_u64().unwrap();
    eprintln!("activity: {} | plan: {}", s["activity"], s["unload_plan"]);
    if jobs > 0 {
        // activity recorder on: the rust profile grew and the plan covers every layer of the tiny model (8)
        let rust = s["profiles"].as_array().unwrap().iter().find(|x| x["domain"] == "code-rust").unwrap().clone();
        assert!(rust["tokens"].as_u64().unwrap() > 0, "{}", rust);
        assert_eq!(s["unload_plan"]["per_layer_cold"].as_array().unwrap().len(), 8, "{}", s["unload_plan"]);
    }

    // request log + health carry the one-line summary
    let log = std::fs::read_to_string(format!("{}/req.jsonl", g.dir)).unwrap();
    let last: Value = serde_json::from_str(log.lines().last().unwrap()).unwrap();
    assert!(last["watcher"].as_str().unwrap().starts_with("code-rust"), "{}", last);
    let h = request(p, "GET", "/health", None).unwrap().json();
    assert!(h["watcher"].as_str().unwrap().contains("[dry run]"), "{}", h);

    // on disk: state, decisions, notes (no request text kept: only the note text)
    let w = format!("{}/w", g.dir);
    let st: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/state.json", w)).unwrap()).unwrap();
    assert_eq!(st["domain"], "code-rust");
    assert_eq!(std::fs::read_to_string(format!("{}/decisions.jsonl", w)).unwrap().lines().count(), 3);
    assert_eq!(std::fs::read_to_string(format!("{}/notes.jsonl", w)).unwrap().lines().count(), 1);
    assert!(!std::fs::read_to_string(format!("{}/decisions.jsonl", w)).unwrap().contains("Vec::new"));
}

#[test]
fn watcher_audits_heat_http() {
    let Some(g) = start() else { return };
    let p = g.port;
    // a fold makes code-rust the domain; then two jobs: both replies are queued for their audit
    let r = request(p, "POST", "/v1/streamdense/watcher/note", Some(&json!({"kind": "fold", "text": RUST, "mode": "code"}))).unwrap();
    assert_eq!(r.json()["domain"], "code-rust");
    let b = json!({"model": "qwen38-27b", "messages": [{"role": "user", "content": RUST}], "max_tokens": 6, "stream": false});
    let ids: Vec<String> = (0..2).map(|_| request(p, "POST", "/v1/messages", Some(&b)).unwrap().json()["id"].as_str().unwrap().to_string()).collect();
    assert!(ids.iter().all(|i| i.starts_with("msg_")));

    // votes are gone: 410 for old clients, whatever the method
    let r = request(p, "POST", "/v1/streamdense/watcher/vote", Some(&json!({"message_id": ids[0], "vote": "like"}))).unwrap();
    assert_eq!(r.status, 410, "{}", r.text());
    assert!(r.text().contains("audited automatically"));
    assert_eq!(request(p, "GET", "/v1/streamdense/watcher/vote", None).unwrap().status, 410);

    // the heat thread drains the queue in idle time (newest first) and writes its summary
    let t0 = std::time::Instant::now();
    let s = loop {
        let s = request(p, "GET", "/v1/streamdense/watcher", None).unwrap().json();
        if s["audit_queue"]["done"].as_u64() == Some(2) && s["heat"]["all"].is_object() { break s; }
        assert!(t0.elapsed() < Duration::from_secs(20), "no audits / heat after 20 s: {} {}", s["audit_queue"], s["heat"]);
        std::thread::sleep(Duration::from_millis(200));
    };
    let au = s["audits_recent"].as_array().unwrap();
    assert_eq!(au.len(), 2);
    assert_eq!((au[0]["message_id"].as_str(), au[1]["message_id"].as_str()), (Some(ids[1].as_str()), Some(ids[0].as_str())), "newest first");
    assert_eq!((au[0]["result"].clone(), au[0]["cause"].clone()), (json!("skipped"), Value::Null));
    assert!(au[0].get("vote").is_none() && au[0]["reason"].as_str().unwrap().contains("full model"));
    let rust = s["scorecard"].as_array().unwrap().iter().find(|r| r["domain"] == "code-rust").unwrap_or_else(|| panic!("{}", s["scorecard"])).clone();
    assert_eq!((rust["skipped"].as_u64(), rust["clean_audits"].as_u64(), rust["live"].as_bool()), (Some(2), Some(0), Some(false)), "{}", rust);
    assert!(rust.get("likes").is_none() && rust["agreement"].is_null());
    assert_eq!(s["audit_queue"]["pending"], 0);

    let h = &s["heat"];
    eprintln!("heat: cpu {} s ({}% of one core), summary {} ms, qos {}, ram {} MB, samples {} ({} active)", h["cpu_s"], h["cpu_pct_one_core"],
              h["summary_cpu_ms"], h["qos"], h["ram_mb"], h["samples"], h["active_samples"]);
    eprintln!("heat all: tokens {} hot {} mtp {} vocab {} embed {}", h["all"]["tokens"], h["all"]["hot_frac_mean"], h["mtp"]["all"], h["vocab"]["all"], h["embed"]["all"]);
    assert_eq!(h["qos"], "background");
    if s["activity"]["jobs"].as_u64().unwrap_or(0) > 0 {
        // heat all-time rows = the rows of the jobs (the thread started before them; no reset involved)
        let learned: u64 = s["profiles"].as_array().unwrap().iter().map(|x| x["tokens"].as_u64().unwrap()).sum();
        assert_eq!(h["all"]["tokens"].as_f64().unwrap() as u64, learned, "{}", h["all"]["tokens"]);
        assert_eq!(h["all"]["per_layer_hot"].as_array().unwrap().len(), 8);
        assert!(h["vocab"]["all"]["tokens"].as_f64().unwrap() >= 2.0, "emitted tokens counted: {}", h["vocab"]);
        assert!(h["embed"]["all"]["tokens"].as_f64().unwrap() > h["vocab"]["all"]["tokens"].as_f64().unwrap(), "prompt rows read: {}", h["embed"]);
        let w = format!("{}/w", g.dir);
        let jf: Vec<_> = std::fs::read_dir(format!("{}/heat", w)).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|x| x == "json")).collect();
        assert_eq!(jf.len(), 1);
        // heat on disk: ONE fixed-size store file (<= 64,000,000 B); the jobs' label holds a slot
        let sd = std::fs::read_dir(format!("{}/store", w)).unwrap().next().unwrap().unwrap().path();
        let len = std::fs::metadata(sd.join("store.bin")).unwrap().len();
        assert!(len > 0 && len <= 64_000_000, "{}", len);
        assert!(s["slots"].as_array().unwrap().iter().any(|x| x["label"] == "code-rust" && x["turns"].as_u64() > Some(0)), "{}", s["slots"]);
    }
    let w = format!("{}/w", g.dir);
    assert_eq!(std::fs::read_to_string(format!("{}/audits.jsonl", w)).unwrap().lines().count(), 2);
    assert!(!std::path::Path::new(&format!("{}/votes.jsonl", w)).exists());
}

/// Read mask: mask = "off" (default) changes nothing: no profile, no predictor file, the GET view says off, and the reply is
/// identical to a gateway with mask = "dry" (dry never applies a mask). Dry: the predictor scores every job and the
/// profile is built at idle (numbers in the GET view).
#[test]
fn watcher_mask_off_dry_http() {
    let b = json!({"model": "qwen38-27b", "system": "Be brief.", "messages": [{"role": "user", "content": RUST}], "max_tokens": 8, "stream": false});
    let reply = |p: u16| { let r = request(p, "POST", "/v1/messages", Some(&b)).unwrap(); assert_eq!(r.status, 200, "{}", r.text()); r.json()["content"].clone() };
    let pred_files = |dir: &str| std::fs::read_dir(format!("{}/w/store", dir)).map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().join("predict.bin").exists()).count()).unwrap_or(0);
    let (off_reply, off_view) = {
        let Some(g) = start() else { return };
        let r = reply(g.port);
        std::thread::sleep(Duration::from_millis(1500)); // past the idle point: sd-heat would build a profile if mask were on
        let s = request(g.port, "GET", "/v1/streamdense/watcher", None).unwrap().json();
        assert_eq!(s["mask"], json!({"mode": "off", "applied": false}), "{}", s["mask"]);
        assert_eq!(s["config"]["mask"], "off");
        assert_eq!(pred_files(&g.dir), 0, "mask off writes no predictor file");
        (r, s)
    };
    let Some(g) = start_with("mask = \"dry\"\nexplore_frac = 0.05\n") else { return };
    let r = reply(g.port);
    assert_eq!(r, off_reply, "dry never applies a mask: same reply as off");
    let t0 = std::time::Instant::now();
    let s = loop {
        let s = request(g.port, "GET", "/v1/streamdense/watcher", None).unwrap().json();
        if s["mask"]["profile"]["blocks"].as_u64().is_some() || off_view["activity"]["jobs"].as_u64() == Some(0) { break s; }
        assert!(t0.elapsed() < Duration::from_secs(20), "no profile after 20 s: {}", s["mask"]);
        std::thread::sleep(Duration::from_millis(200));
    };
    let m = &s["mask"];
    eprintln!("mask dry: {}", m);
    assert_eq!((m["mode"].clone(), m["applied"].clone(), m["effective"].clone()), (json!("dry"), json!(false), json!("dry")));
    if s["activity"]["jobs"].as_u64().unwrap_or(0) > 0 {
        assert_eq!(pred_files(&g.dir), 1);
        assert_eq!(m["predictor"]["scored"], 1, "{}", m["predictor"]);
        assert!(m["predictor"]["last"]["tokens_mean"].as_f64().unwrap() > 0.0, "new prompt tokens seen: {}", m["predictor"]["last"]);
        assert_eq!(m["profile"]["per_layer_masked"].as_array().unwrap().len(), 8);
        assert!(m["profile"]["blocks"].as_u64().unwrap() > 0);
    }
}

/// Answer library (2026-11-03): OFF by default (no file, GET view {"on": false}, nothing in /health). On: one record per
/// turn with >= 8 generated tokens (shorter turns skipped), a ring capped at 2 records (oldest dropped), no prompt text
/// unless `library_text = true`; `streamdense library stats|export` read it back.
#[test]
fn watcher_library_http() {
    const MARK: &str = "zebra-quartz-71";
    let body = |n: u32| json!({"model": "qwen38-27b", "system": "Be brief.", "messages": [{"role": "user", "content": format!("{} {}", MARK, RUST)}], "max_tokens": n, "stream": false});
    let job = |p: u16, n: u32| { let r = request(p, "POST", "/v1/messages", Some(&body(n))).unwrap(); assert_eq!(r.status, 200, "{}", r.text()); };
    let wait = |p: u16, written: u64| {
        let t0 = std::time::Instant::now();
        loop {
            let s = request(p, "GET", "/v1/streamdense/watcher", None).unwrap().json();
            if s["library"]["written"].as_u64() >= Some(written) || s["activity"]["jobs"].as_u64() == Some(0) { break s; }
            assert!(t0.elapsed() < Duration::from_secs(20), "library: {} records not written after 20 s: {}", written, s["library"]);
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    let cli = |args: &[&str]| { let o = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(args).output().unwrap(); assert!(o.status.success(), "{:?}: {}", args, String::from_utf8_lossy(&o.stderr)); String::from_utf8(o.stdout).unwrap() };
    let file_of = |dir: &str| std::fs::read_dir(format!("{}/w/library", dir)).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "sdl")).unwrap();
    let has = |hay: &[u8], n: &str| hay.windows(n.len()).any(|w| w == n.as_bytes());
    {
        let Some(g) = start() else { return };
        job(g.port, 10);
        std::thread::sleep(Duration::from_millis(800));
        let s = request(g.port, "GET", "/v1/streamdense/watcher", None).unwrap().json();
        assert_eq!(s["library"], json!({"on": false}), "default off");
        assert!(!std::path::Path::new(&format!("{}/w/library", g.dir)).exists(), "off: no library dir");
        assert!(!request(g.port, "GET", "/health", None).unwrap().json()["watcher"].as_str().unwrap().contains("library"));
    }
    // on, cap 90,000 B = 2 tiny records (40,960 B each + 4,096 B header), no text
    let Some(g) = start_with("library = true\nlibrary_cap_gb = 0.00009\n") else { return };
    let p = g.port;
    for n in [12, 12, 3, 12] { job(p, n); }
    let s = wait(p, 3);
    if s["activity"]["jobs"].as_u64() == Some(0) { eprintln!("SKIP: no recorder"); return; }
    let l = &s["library"];
    eprintln!("library (tiny): {}", l);
    assert_eq!((l["on"].clone(), l["records"].clone(), l["slots"].clone()), (json!(true), json!(2), json!(2)), "{}", l);
    assert_eq!((l["written"].as_u64(), l["evicted"].as_u64(), l["skipped_short"].as_u64(), l["errors"].as_u64()), (Some(3), Some(1), Some(1), Some(0)), "{}", l);
    assert_eq!((l["record_bytes"].as_u64(), l["cap_bytes"].as_u64()), (Some(40_960), Some(4096 + 2 * 40_960)));
    assert!(l["disk_bytes"].as_u64() <= l["cap_bytes"].as_u64(), "{}", l);
    assert_eq!(l["text"], json!(false));
    assert_eq!(l["cost"]["marks"].as_u64(), Some(4), "one mark per job: {}", l["cost"]);
    assert!(request(p, "GET", "/health", None).unwrap().json()["watcher"].as_str().unwrap().contains("library 2 rec"));
    let f = file_of(&g.dir);
    assert!(!has(&std::fs::read(&f).unwrap(), MARK), "no prompt text stored by default");
    let st: Value = serde_json::from_str(&cli(&["library", "stats", "--file", f.to_str().unwrap(), "--verify", "--json"])).unwrap();
    assert_eq!((st["records"].as_u64(), st["bad"].as_u64(), st["by_label"]["code-rust"].as_u64()), (Some(2), Some(0), Some(2)), "{}", st);
    assert!(st["mean"]["answer_tokens"].as_f64().unwrap() >= 8.0 && st["mean"]["prompt_rows"].as_f64().unwrap() > 0.0, "{}", st);
    let ex = cli(&["library", "export", "--dir", &format!("{}/w/library", g.dir), "--text", "--verify"]);
    let rows: Vec<Value> = ex.lines().map(|x| serde_json::from_str(x).unwrap()).collect();
    assert_eq!(rows.iter().map(|r| r["seq"].as_u64().unwrap()).collect::<Vec<_>>(), vec![2, 3], "oldest record dropped first (the short turn got no seq)");
    assert!(rows.iter().all(|r| r["checksum_ok"] == json!(true) && r["has_text"] == json!(false) && r.get("text").is_none() && r["answer_rows"].as_u64() > Some(0)));
    drop(g);
    // library_text = true: the newest user message is kept
    let Some(g) = start_with("library = true\nlibrary_text = true\n") else { return };
    job(g.port, 12);
    let s = wait(g.port, 1);
    assert_eq!(s["library"]["text"], json!(true));
    assert!(has(&std::fs::read(file_of(&g.dir)).unwrap(), MARK), "text stored when switched on");
    let ex = cli(&["library", "export", "--dir", &format!("{}/w/library", g.dir), "--text"]);
    assert!(ex.contains(MARK), "{}", ex);
}
