//! Batched serving (`--seqs N`, worker.rs `run_batched`, model batch.rs):
//! 1. scripted fake engine with 3 slots (20 ms per pass, one token per sequence per pass): concurrent requests decode
//!    in shared passes (wall time ~ one request's, not the sum); a request arriving during a lone decode waits only for
//!    the current pass (the lone sequence yields); a 4th request waits for a free slot; a warm-up waits until nothing
//!    decodes; a follow-up of a conversation goes back to its own slot and reuses its KV while another one ran.
//! 2. tiny model (models/tiny_qwen35, ids folded) + the 27B tokenizer: replies of a 3-slot gateway (2 concurrent, a
//!    third joining mid-way, different lengths) == the replies of a 1-slot gateway serving them one by one.
use serde_json::{json, Value};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use streamdense_gateway::client::request;
use streamdense_gateway::{Backend, DecSeq, FeedStats, GatewayConfig, WorkerConfig};
use streamdense_model::qwen_engine::{QwenEngine, QwenOptions};
use streamdense_tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("sd_gwbatch_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_file(&d);
    d
}

fn log_lines(p: &str) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// The agent fixture with these messages (non-streaming).
fn body(msgs: Value, max_tokens: u64) -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/anthropic_agent_req.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let mut b = v["body"].clone();
    b["stream"] = json!(false);
    b["max_tokens"] = json!(max_tokens);
    b["messages"] = msgs;
    b
}

fn user(text: &str) -> Value { json!([{"role": "user", "content": text}]) }

fn post_bg(port: u16, b: Value) -> std::thread::JoinHandle<(Value, f64)> {
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = request(port, "POST", "/v1/messages", Some(&b)).unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        (r.json(), t0.elapsed().as_secs_f64())
    })
}

// ---------------------------------------------------------------- scheduler (fake engine)

const PASS: Duration = Duration::from_millis(20);
const PER_TOKEN: Duration = Duration::from_micros(50);
const SLOTS: usize = 3;

#[derive(Default)]
struct Counters { rounds: AtomicUsize, max_round: AtomicUsize, solo_passes: AtomicUsize, yields: AtomicUsize }

/// Per-slot positions (last token pending, as the engine); every pass emits token 13 (".") per sequence.
struct Multi { pos: Vec<Cell<usize>>, cur: Cell<usize>, c: Arc<Counters> }
struct FakeSeq { n_new: usize, gen: usize, prefill_s: f64, t0: Instant }

impl Multi {
    fn step(&self, d: &mut DecSeq, on_tok: &mut dyn FnMut(u32) -> bool) {
        let slot = d.slot;
        let s = d.inner.downcast_mut::<FakeSeq>().unwrap();
        self.pos[slot].set(self.pos[slot].get() + 1);
        s.gen += 1;
        Self::tok(d, on_tok);
    }
    /// emit the newest token of `d` (gen already counted); mark it done on cancel / length
    fn tok(d: &mut DecSeq, on_tok: &mut dyn FnMut(u32) -> bool) {
        let s = d.inner.downcast_ref::<FakeSeq>().unwrap();
        let (gen, n_new, prefill_s, t0) = (s.gen, s.n_new, s.prefill_s, s.t0);
        let why = if !on_tok(13) { Some("cancelled") } else if gen >= n_new { Some("length") } else { None };
        if let Some(w) = why {
            d.done = Some(w.into());
            d.stats = FeedStats { gen, last: Some(13), stop_reason: w.into(), prefill_s, decode_s: t0.elapsed().as_secs_f64(), ..Default::default() };
        }
    }
}

impl Backend for Multi {
    fn reset(&self) { self.pos[self.cur.get()].set(0); }
    fn pos(&self) -> usize { self.pos[self.cur.get()].get() }
    fn max_ctx(&self) -> usize { 32768 }
    fn vocab(&self) -> usize { 248320 }
    fn mtp_available(&self) -> bool { true }
    fn warm_chunk(&self) -> usize { 64 }
    /// warm-ups / snapshot segments only (the batched scheduler decodes through dec_*)
    fn feed(&self, ids: &[u32], n_new: usize, _stop: &[u32], _mtp: bool, on_tok: &mut dyn FnMut(u32) -> bool) -> Result<FeedStats, String> {
        assert_eq!(n_new, 1, "batched mode decodes through dec_begin");
        std::thread::sleep(PER_TOKEN * ids.len() as u32);
        let p = &self.pos[self.cur.get()];
        p.set(p.get() + ids.len());
        on_tok(13);
        Ok(FeedStats { gen: 1, last: Some(13), stop_reason: "length".into(), ..Default::default() })
    }
    fn seqs(&self) -> usize { SLOTS }
    fn select(&self, i: usize) { self.cur.set(i); }
    fn dec_begin(&self, slot: usize, ids: &[u32], n_new: usize, _stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool) -> Result<DecSeq, String> {
        assert_eq!(self.cur.get(), slot, "the worker selects the slot before its prefill");
        let t = PER_TOKEN * ids.len() as u32;
        std::thread::sleep(t);
        self.pos[slot].set(self.pos[slot].get() + ids.len());
        let mut d = DecSeq { slot, done: None, stats: FeedStats::default(),
                             inner: Box::new(FakeSeq { n_new, gen: 1, prefill_s: t.as_secs_f64(), t0: Instant::now() }) };
        Self::tok(&mut d, on_tok);
        Ok(d)
    }
    fn dec_solo(&self, d: &mut DecSeq, on_tok: &mut dyn FnMut(u32) -> bool, yield_now: &dyn Fn() -> bool) -> Result<(), String> {
        while d.done.is_none() {
            if yield_now() { self.c.yields.fetch_add(1, Ordering::SeqCst); break; }
            std::thread::sleep(PASS);
            self.c.solo_passes.fetch_add(1, Ordering::SeqCst);
            self.step(d, on_tok);
        }
        Ok(())
    }
    fn dec_round(&self, s: &mut [&mut DecSeq], on_tok: &mut dyn FnMut(usize, u32) -> bool) -> Result<(), String> {
        assert!(s.len() >= 2 && s.len() <= SLOTS && s.iter().all(|d| d.done.is_none()));
        let mut slots: Vec<usize> = s.iter().map(|d| d.slot).collect();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), s.len(), "one slot per sequence");
        self.c.rounds.fetch_add(1, Ordering::SeqCst);
        self.c.max_round.fetch_max(s.len(), Ordering::SeqCst);
        std::thread::sleep(PASS);
        for (i, d) in s.iter_mut().enumerate() { self.step(d, &mut |t| on_tok(i, t)); }
        Ok(())
    }
}

fn start_multi(log: &str, c: Arc<Counters>) -> Option<u16> {
    if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer.json missing"); return None; }
    let (tx, rx) = std::sync::mpsc::channel();
    let log = log.to_string();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let eng = Multi { pos: (0..SLOTS).map(|_| Cell::new(0)).collect(), cur: Cell::new(0), c };
        let cfg = GatewayConfig { port: 0, log: Some(log), worker: WorkerConfig { think: false, mtp: true, fold: None }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, None, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(60)).unwrap())
}

fn out_tokens(r: &Value) -> u64 { r["usage"]["output_tokens"].as_u64().unwrap() }

#[test]
fn batched_scheduler_fake() {
    let log = tmp("sched.jsonl").display().to_string();
    let c = Arc::new(Counters::default());
    let Some(port) = start_multi(&log, c.clone()) else { return };

    // 1. three concurrent requests share passes: ~40 passes in all, not 3 x 40
    let t0 = Instant::now();
    let hs: Vec<_> = ["one", "two", "three"].iter().map(|t| post_bg(port, body(user(&format!("concurrent {}", t)), 40))).collect();
    for h in hs { let (r, _) = h.join().unwrap(); assert_eq!(out_tokens(&r), 40, "{}", r); }
    let wall = t0.elapsed().as_secs_f64();
    assert_eq!(c.max_round.load(Ordering::SeqCst), 3);
    assert!(wall < 1.6, "3 x 40 tokens in shared passes took {:.2} s (serial would be ~2.4 s)", wall);

    // 2. a request arriving during a lone decode waits only for the current pass (the lone sequence yields), and the
    //    long one still gets all its tokens
    let y0 = c.yields.load(Ordering::SeqCst);
    let long = post_bg(port, body(user("long running"), 100));
    std::thread::sleep(Duration::from_millis(300));
    let (q, q_s) = post_bg(port, body(user("quick one"), 4)).join().unwrap();
    let (l, _) = long.join().unwrap();
    assert_eq!((out_tokens(&q), out_tokens(&l)), (4, 100));
    assert!(c.yields.load(Ordering::SeqCst) > y0, "the lone decode yielded");
    assert!(q_s < 0.4, "quick request took {:.2} s next to a running one", q_s);
    let ll = log_lines(&log);
    let ql = ll.iter().find(|x| x["n_new"] == 4).unwrap();
    assert!(ql["queue_s"].as_f64().unwrap() < 0.1, "queued past a pass: {}", ql);

    // 3. a 4th request waits for a free slot; never more than SLOTS in a pass
    let hs: Vec<_> = (0..4).map(|i| post_bg(port, body(user(&format!("slot test {}", i)), 20 + 10 * i))).collect();
    let res: Vec<(Value, f64)> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    for (i, (r, _)) in res.iter().enumerate() { assert_eq!(out_tokens(r), 20 + 10 * i as u64); }
    assert_eq!(c.max_round.load(Ordering::SeqCst), 3);
    let ll = log_lines(&log);
    let waited = ll[ll.len() - 4..].iter().filter(|x| x["queue_s"].as_f64().unwrap() > 0.2).count();
    assert_eq!(waited, 1, "exactly one request waited for a slot: {:?}", ll[ll.len() - 4..].iter().map(|x| x["queue_s"].clone()).collect::<Vec<_>>());

    // 4. a warm-up waits until nothing decodes; real requests queued after it still run first
    let a = post_bg(port, body(user("decoding alpha"), 60));
    std::thread::sleep(Duration::from_millis(100));
    let w = post_bg(port, body(json!([{"role": "user", "content": "warm beta"}, {"role": "assistant", "content": "ok"}]), 0));
    std::thread::sleep(Duration::from_millis(50));
    let b = post_bg(port, body(user("decoding gamma"), 10));
    let (wr, _) = w.join().unwrap();
    let _ = (a.join().unwrap(), b.join().unwrap());
    assert_eq!(wr["streamdense"]["warm"], "done", "{}", wr);
    let ll = log_lines(&log);
    let tail: Vec<bool> = ll[ll.len() - 3..].iter().map(|x| x["warm"].as_bool().unwrap()).collect();
    assert_eq!(tail, [false, false, true], "both real requests finished before the warm-up ran");

    // 5. per-slot KV: conversation X, then Y next to it, then X's follow-up reuses X's slot (prefix / followup)
    let x = user("conversation X: list the files");
    let rx = post_bg(port, body(x.clone(), 30));
    std::thread::sleep(Duration::from_millis(60));
    let ry = post_bg(port, body(user("conversation Y: something else"), 30));
    let (xr, _) = rx.join().unwrap();
    let _ = ry.join().unwrap();
    let reply = xr["content"][0]["text"].as_str().unwrap_or("").to_string();
    let mut next = x.as_array().unwrap().clone();
    next.push(json!({"role": "assistant", "content": reply}));
    next.push(json!({"role": "user", "content": "and now?"}));
    let (_n, _) = post_bg(port, body(Value::Array(next), 4)).join().unwrap();
    let last = log_lines(&log).pop().unwrap();
    assert_ne!(last["mode"], "full", "{}", last);
    assert!(last["n_reused"].as_u64().unwrap() > 0, "{}", last);
    println!("  fake batch: 3x40 in {:.2} s ({} rounds, {} solo passes, {} yields); quick next to long {:.2} s; follow-up {} reused {}",
             wall, c.rounds.load(Ordering::SeqCst), c.solo_passes.load(Ordering::SeqCst), c.yields.load(Ordering::SeqCst), q_s,
             last["mode"], last["n_reused"]);
    let _ = std::fs::remove_file(&log);
}

// ---------------------------------------------------------------- tiny model: batched == one by one

fn start_tiny(seqs: usize) -> Option<u16> {
    let tiny = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/mtp/weights.safetensors", tiny)).exists() || !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() {
        eprintln!("SKIP: tiny model or 27B tokenizer missing");
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tk = Arc::new(Tokenizer::from_path(TOK_DIR).unwrap());
        let opt = QwenOptions { max_ctx: 4096, mtp: true, spec_depth: 7, spec_auto: true, seqs, ..Default::default() };
        let eng = QwenEngine::open(&tiny, opt).unwrap();
        assert_eq!(eng.seqs(), seqs);
        let vocab = eng.cfg.vocab as u32;
        let cfg = GatewayConfig { port: 0, worker: WorkerConfig { think: false, mtp: true, fold: Some(vocab) }, ..Default::default() };
        streamdense_gateway::serve(&eng, tk, cfg, None, |p| { tx.send(p).unwrap(); }).unwrap();
    });
    Some(rx.recv_timeout(Duration::from_secs(120)).unwrap())
}

fn reply(r: &Value) -> (String, u64, String) {
    let text = r["content"].as_array().unwrap().iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join("|");
    (text, out_tokens(r), r["stop_reason"].as_str().unwrap_or("").to_string())
}

#[test]
fn batched_equals_serial_tiny() {
    let reqs = [(user("Write a short poem about rivers."), 48u64), (user("List three prime numbers and explain why."), 80),
                (user("What is the capital of France?"), 24)];
    let Some(solo) = start_tiny(1) else { return };
    let want: Vec<_> = reqs.iter().map(|(m, n)| reply(&post_bg(solo, body(m.clone(), *n)).join().unwrap().0)).collect();
    let Some(port) = start_tiny(3) else { return };
    // two at once, the third joins while they decode; then all three at once again (slots reused)
    for round in 0..2 {
        let a = post_bg(port, body(reqs[0].0.clone(), reqs[0].1));
        let b = post_bg(port, body(reqs[1].0.clone(), reqs[1].1));
        if round == 0 { std::thread::sleep(Duration::from_millis(150)); }
        let c = post_bg(port, body(reqs[2].0.clone(), reqs[2].1));
        let got = [a, b, c].map(|h| reply(&h.join().unwrap().0));
        for i in 0..3 { assert_eq!(got[i], want[i], "round {} request {}", round, i); }
    }
    println!("  tiny batch == serial: tokens {:?}", want.iter().map(|w| w.1).collect::<Vec<_>>());
}
