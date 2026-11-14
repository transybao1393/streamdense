//! streamdense index build|stats|query   (history index: drafts from text the user already has; see
//! streamdense-core/src/history_index.rs). No model load; `build` and `query --text` load only the tokenizer.
//!   build  [--project DIR] [--sessions PATH] [--dir D] [--budget-mb 32] [--threads N] [--max-file-kb 256] [--tokenizer DIR]
//!   stats  [--dir D] [--budget-mb 32]
//!   query  (--text "..." [--tokenizer DIR] | --tokens 1,2,3) [--max-len 8] [--branches 4] [--reps 1000] [--dir D] [--budget-mb 32]
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::{Duration, Instant};
use streamdense_core::history_index::{Config, Control, Drafts, Encoder, HistoryIndex, OpenNote, Stats};
use streamdense_tokenizer::{Tokenizer, DEFAULT_TOKENIZER_DIR};
use crate::Args;

const USAGE: &str = "usage: streamdense index build [--project DIR] [--sessions PATH] [--dir D] [--budget-mb 32] [--threads N] [--max-file-kb 256] [--tokenizer DIR]
       streamdense index stats [--dir D] [--budget-mb 32]
       streamdense index query (--text \"...\" | --tokens 1,2,3) [--max-len 8] [--branches 4] [--reps 1000] [--dir D] [--budget-mb 32] [--tokenizer DIR]";

fn fail(e: impl std::fmt::Display) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }
fn mb(b: f64) -> f64 { b / 1048576.0 }

fn config(a: &Args) -> Config {
    let mut c = Config::default();
    if let Some(d) = a.s("--dir") { c.dir = crate::watcher::expand_home(&d) }
    c.budget_bytes = (a.f("--budget-mb", 32.0) * 1048576.0) as usize;
    c.threads = a.f("--threads", 0.0) as usize;
    c.max_file_bytes = (a.f("--max-file-kb", 256.0) * 1024.0) as u64;
    c
}
fn open(c: Config) -> HistoryIndex {
    let dir = c.dir.clone();
    let (idx, note) = HistoryIndex::open(c).unwrap_or_else(|e| fail(e));
    match note {
        OpenNote::Loaded => eprintln!("loaded {}", dir.display()),
        OpenNote::New => eprintln!("new index (nothing in {})", dir.display()),
        OpenNote::GeometryChanged => eprintln!("WARNING: {} was built with another --budget-mb; starting empty", dir.display()),
        OpenNote::Corrupt(e) => eprintln!("WARNING: {}; starting empty", e),
    }
    idx
}
fn tokenizer(a: &Args) -> Arc<Tokenizer> {
    let p = a.sd("--tokenizer", DEFAULT_TOKENIZER_DIR);
    let t0 = Instant::now();
    let t = Tokenizer::from_path(&p).unwrap_or_else(|e| fail(e));
    eprintln!("tokenizer {} ({:.2} s)", p, t0.elapsed().as_secs_f64());
    Arc::new(t)
}
fn encoder(t: Arc<Tokenizer>) -> Encoder { Arc::new(move |s: &str| t.encode(s)) }

fn print_stats(s: &Stats) {
    println!("budget         {:.1} MB | table {:.1} MB | RAM (table + manifest) {:.1} MB", mb(s.budget_bytes as f64), mb(s.table_bytes as f64), mb(s.ram_bytes as f64));
    println!("slots          {} used of {} ({:.1}%) | orders {:?}", s.used_slots, s.slots, 100.0 * s.used_slots as f64 / s.slots.max(1) as f64, s.orders);
    println!("ingested       {} tokens in {} docs | {} n-gram observations | {} evictions | {} files tracked", s.tokens, s.docs, s.inserts, s.evictions, s.files_tracked);
    let fb: Vec<String> = s.feedback.iter().enumerate().filter(|(_, f)| f.0 + f.1 > 0).map(|(n, f)| format!("n{} {}/{}", n, f.0, f.0 + f.1)).collect();
    if !fb.is_empty() { println!("feedback       {}", fb.join(" | ")) }
}

/// progress line every 0.5 s on stderr while `job` runs on another thread
fn with_progress<T: Send>(what: &str, ctl: &Control, job: impl FnOnce() -> T + Send) -> T {
    let t0 = Instant::now();
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|sc| {
        let h = sc.spawn(|| { let r = job(); done.store(true, Relaxed); r });
        while !done.load(Relaxed) {
            std::thread::sleep(Duration::from_millis(500));
            eprint!("\r{}: {}/{} files, {:.1} MB, {} tokens, {:.0} s   ", what, ctl.files_done.load(Relaxed), ctl.files_found.load(Relaxed),
                mb(ctl.bytes_done.load(Relaxed) as f64), ctl.tokens_done.load(Relaxed), t0.elapsed().as_secs_f64());
        }
        eprintln!();
        h.join().unwrap_or_else(|_| fail("worker panicked"))
    })
}

pub fn main(a: &Args) {
    match a.0.get(2).map(String::as_str) {
        Some("build") => build(a),
        Some("stats") => {
            let idx = open(config(a));
            println!("dir            {}", idx.config().dir.display());
            print_stats(&idx.stats());
        }
        Some("query") => query(a),
        _ => { eprintln!("{}", USAGE); std::process::exit(2) }
    }
}

fn build(a: &Args) {
    let (project, sessions) = (a.s("--project").map(PathBuf::from), a.s("--sessions").map(PathBuf::from));
    if project.is_none() && sessions.is_none() { eprintln!("ERROR: give --project DIR and/or --sessions PATH\n{}", USAGE); std::process::exit(2) }
    for p in project.iter().chain(sessions.iter()) { if !p.exists() { fail(format!("{} does not exist", p.display())) } }
    let c = config(a);
    eprintln!("budget {:.1} MB, {} build threads (background QoS), files ≤ {} KB, dir {}", mb(c.budget_bytes as f64), c.build_threads(), c.max_file_bytes / 1024, c.dir.display());
    let idx = open(c);
    let enc = encoder(tokenizer(a));
    let t0 = Instant::now();
    if let Some(p) = &project {
        let ctl = Control::default();
        let r = with_progress("project", &ctl, || idx.index_dir(p, &enc, &ctl));
        println!("project        {} indexed, {} unchanged, {} ignored, {} binary, {} large, {} errors | {:.1} MB, {} tokens",
            r.files_indexed, r.files_unchanged, r.skipped_ignored, r.skipped_binary, r.skipped_large, r.errors, mb(r.bytes as f64), r.tokens);
    }
    if let Some(p) = &sessions {
        let ctl = Control::default();
        let r = with_progress("sessions", &ctl, || { streamdense_core::history_index::set_background_qos(); idx.index_sessions(p, &enc, &ctl) });
        println!("sessions       {} indexed, {} unchanged, {} large, {} errors | {:.1} MB, {} tokens", r.files_indexed, r.files_unchanged, r.skipped_large, r.errors, mb(r.bytes as f64), r.tokens);
    }
    let w = idx.save().unwrap_or_else(|e| fail(e));
    println!("saved          {:.1} MB to {} in {:.1} s total", mb(w as f64), idx.config().dir.display(), t0.elapsed().as_secs_f64());
    print_stats(&idx.stats());
}

fn query(a: &Args) {
    let idx = open(config(a));
    let tk = if a.s("--text").is_some() || a.s("--tokens").is_none() { Some(tokenizer(a)) } else { None };
    let tail: Vec<u32> = match (a.s("--tokens"), a.s("--text"), &tk) {
        (Some(t), _, _) => t.split(',').map(|x| x.trim().parse().unwrap_or_else(|_| fail(format!("bad token id {:?}", x)))).collect(),
        (None, Some(s), Some(t)) => t.encode(&s),
        _ => { eprintln!("{}", USAGE); std::process::exit(2) }
    };
    let (max_len, branches, reps) = (a.f("--max-len", 8.0) as usize, a.f("--branches", 4.0) as usize, a.f("--reps", 1000.0).max(1.0) as usize);
    let mut d = Drafts::default();
    let mut us: Vec<f64> = (0..reps).map(|_| { let t = Instant::now(); idx.query_into(&tail, max_len, branches, &mut d); t.elapsed().as_secs_f64() * 1e6 }).collect();
    us.sort_by(|a, b| a.total_cmp(b));
    println!("tail {:?} → {} rows, {} lookups | latency p50 {:.1} us, p99 {:.1} us over {} reps", &tail[tail.len().saturating_sub(8)..], d.rows.len(), d.lookups,
        us[us.len() / 2], us[us.len() * 99 / 100], reps);
    for (i, r) in d.rows.iter().enumerate() {
        let text = tk.as_ref().map(|t| format!(" {:?}", t.decode(&r.tokens))).unwrap_or_default();
        println!("row {} (parent {}, shared {}): {:?}{} conf {:?} orders {:?}", i, r.parent, r.shared, r.tokens, text,
            r.conf.iter().map(|c| (c * 1000.0).round() / 1000.0).collect::<Vec<_>>(), r.order);
    }
}
