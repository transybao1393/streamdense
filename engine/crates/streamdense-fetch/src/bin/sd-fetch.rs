//! sd-fetch file | repo | tensors | bench   (parallel ranged HTTP downloads from Hugging Face, streamed to disk)
//!   common:  --threads N (8) --chunk-mb M (32) --piece-kb K (1024) --rev REV (main); token from $HF_TOKEN
//!   file:    --repo R --file F --out PATH [--limit-mb B] [--resume]
//!   repo:    --repo R --out DIR [--include GLOB]... [--exclude GLOB]... [--resume]
//!   tensors: --repo R --file F --match SUBSTR --out PATH
//!   bench:   --repo R --file F --threads 1,4,8,16 --limit-mb 1024 --out-json PATH [--tmp PATH]
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;
use streamdense_fetch::{fetch_file, hf, FetchConfig, Stats};

struct Args(Vec<String>);
impl Args {
    fn s(&self, k: &str) -> Option<String> { self.0.iter().position(|x| x == k).and_then(|i| self.0.get(i + 1).cloned()) }
    fn sd(&self, k: &str, d: &str) -> String { self.s(k).unwrap_or(d.to_string()) }
    fn f(&self, k: &str, d: f64) -> f64 { self.s(k).map(|v| v.parse().unwrap_or_else(|_| die(&format!("bad value for {}: {}", k, v)))).unwrap_or(d) }
    fn has(&self, k: &str) -> bool { self.0.iter().any(|x| x == k) }
    fn req(&self, k: &str) -> String { self.s(k).unwrap_or_else(|| die(&format!("missing {}", k))) }
    fn all(&self, k: &str) -> Vec<String> {
        self.0.iter().enumerate().filter(|(_, x)| *x == k).filter_map(|(i, _)| self.0.get(i + 1).cloned()).collect()
    }
}
fn die(msg: &str) -> ! { eprintln!("ERROR: {}", msg); std::process::exit(2) }
const MB: f64 = 1048576.0;

fn final_line(bytes: u64, secs: f64, threads: usize) {
    let mbps = if secs > 0.0 { bytes as f64 / MB / secs } else { 0.0 };
    println!("done: {} bytes ({:.1} MB) in {:.2} s = {:.1} MB/s, threads {}", bytes, bytes as f64 / MB, secs, mbps, threads);
}

/// UTC date-time string from the system clock (no chrono dependency).
fn now_utc() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (t.div_euclid(86400), t.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

fn stats_json(s: &Stats) -> Value {
    json!({"bytes": s.bytes, "secs": s.secs, "mb_per_s": s.mb_per_s, "chunks": s.chunks, "retries": s.retries,
        "threads": s.threads, "per_thread_bytes": s.per_thread_bytes, "first_byte_ms": s.first_byte_ms})
}

fn main() {
    let a = Args(std::env::args().collect());
    let cmd = a.0.get(1).cloned().unwrap_or_default();
    let mut cfg = FetchConfig::default();
    cfg.chunk_bytes = (a.f("--chunk-mb", 32.0) * MB) as u64;
    cfg.piece_bytes = (a.f("--piece-kb", 1024.0) * 1024.0) as usize;
    cfg.token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    if cmd != "bench" { cfg.threads = a.f("--threads", 8.0) as usize; }
    let rev = a.sd("--rev", "main");
    let limit = a.s("--limit-mb").map(|v| (v.parse::<f64>().unwrap_or_else(|_| die("bad --limit-mb")) * MB) as u64);
    let resume = a.has("--resume");

    match cmd.as_str() {
        "file" => {
            let (repo, file, out) = (a.req("--repo"), a.req("--file"), PathBuf::from(a.req("--out")));
            if let Some(p) = out.parent() { if !p.as_os_str().is_empty() { std::fs::create_dir_all(p).unwrap_or_else(|e| die(&e.to_string())); } }
            let url = hf::resolve_url(&repo, &rev, &file);
            eprintln!("{} -> {}", url, out.display());
            let s = fetch_file(&url, &out, &cfg, limit, resume).unwrap_or_else(|e| die(&e));
            eprintln!("chunks {}, retries {}, first byte {:.1} ms", s.chunks, s.retries, s.first_byte_ms);
            final_line(s.bytes, s.secs, s.threads);
        }
        "repo" => {
            let (repo, dir) = (a.req("--repo"), PathBuf::from(a.req("--out")));
            let (inc, exc) = (a.all("--include"), a.all("--exclude"));
            let files = hf::list_files(&repo, &rev, &cfg).unwrap_or_else(|e| die(&e));
            let pick: Vec<_> = files.into_iter()
                .filter(|f| inc.is_empty() || inc.iter().any(|g| hf::glob_match(g, &f.path)))
                .filter(|f| !exc.iter().any(|g| hf::glob_match(g, &f.path)))
                .collect();
            let total: u64 = pick.iter().map(|f| f.size).sum();
            eprintln!("{}@{}: {} files, {:.1} MB", repo, rev, pick.len(), total as f64 / MB);
            let t0 = Instant::now();
            let (mut done, mut fetched, mut retries) = (0u64, 0u64, 0u32);
            let mut last = Instant::now() - std::time::Duration::from_secs(2);
            for (i, f) in pick.iter().enumerate() {
                let out = dir.join(&f.path);
                if let Some(p) = out.parent() { std::fs::create_dir_all(p).unwrap_or_else(|e| die(&format!("{}: {}", p.display(), e))); }
                let have = std::fs::metadata(&out).map(|m| m.len()).ok();
                if resume && have == Some(f.size) {
                    done += f.size;
                    eprintln!("[{}/{}] {} already complete, skipped", i + 1, pick.len(), f.path);
                    continue;
                }
                let url = hf::resolve_url(&repo, &rev, &f.path);
                let s = fetch_file(&url, &out, &cfg, None, resume).unwrap_or_else(|e| die(&format!("{}: {}", f.path, e)));
                done += f.size; fetched += s.bytes; retries += s.retries;
                if last.elapsed().as_secs_f64() >= 1.0 || i + 1 == pick.len() {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!("[{}/{}] {:.1} / {:.1} MB, {:.1} MB/s", i + 1, pick.len(), done as f64 / MB, total as f64 / MB,
                        fetched as f64 / MB / el.max(1e-9));
                    last = Instant::now();
                }
            }
            eprintln!("retries {}", retries);
            final_line(fetched, t0.elapsed().as_secs_f64(), cfg.threads);
        }
        "tensors" => {
            let (repo, file, pat, out) = (a.req("--repo"), a.req("--file"), a.req("--match"), PathBuf::from(a.req("--out")));
            if let Some(p) = out.parent() { if !p.as_os_str().is_empty() { std::fs::create_dir_all(p).unwrap_or_else(|e| die(&e.to_string())); } }
            let filter = |name: &str| name.contains(pat.as_str());
            let (s, n) = hf::fetch_tensors(&repo, &rev, &file, &filter, &out, &cfg).unwrap_or_else(|e| die(&e));
            eprintln!("{} tensors matching {:?} -> {}", n, pat, out.display());
            final_line(s.bytes, s.secs, s.threads);
        }
        "bench" => {
            let (repo, file, out_json) = (a.req("--repo"), a.req("--file"), PathBuf::from(a.req("--out-json")));
            let tlist: Vec<usize> = a.sd("--threads", "1,4,8,16").split(',')
                .map(|x| x.trim().parse().unwrap_or_else(|_| die(&format!("bad thread count {}", x)))).collect();
            let limit = Some(limit.unwrap_or((1024.0 * MB) as u64));
            let jdir = out_json.parent().filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
            std::fs::create_dir_all(&jdir).unwrap_or_else(|e| die(&e.to_string()));
            let tmp = a.s("--tmp").map(PathBuf::from).unwrap_or_else(|| jdir.join(".bench.tmp"));
            let url = hf::resolve_url(&repo, &rev, &file);
            eprintln!("bench {} (first {:.0} MB) -> tmp {}", url, limit.unwrap() as f64 / MB, tmp.display());
            let mut rows = vec![];
            let (mut tb, mut ts) = (0u64, 0f64);
            for &t in &tlist {
                cfg.threads = t;
                let _ = std::fs::remove_file(&tmp);
                let r = fetch_file(&url, &tmp, &cfg, limit, false);
                let _ = std::fs::remove_file(&tmp);
                let s = r.unwrap_or_else(|e| die(&format!("threads {}: {}", t, e)));
                eprintln!("threads {:2}: {:.1} MB in {:.2} s = {:.1} MB/s", t, s.bytes as f64 / MB, s.secs, s.mb_per_s);
                tb += s.bytes; ts += s.secs;
                rows.push(s);
            }
            println!("{:>7} {:>9} {:>8} {:>9} {:>14} {:>7}", "threads", "MB", "s", "MB/s", "first-byte ms", "retries");
            for s in &rows {
                println!("{:>7} {:>9.1} {:>8.2} {:>9.1} {:>14.1} {:>7}", s.threads, s.bytes as f64 / MB, s.secs, s.mb_per_s, s.first_byte_ms, s.retries);
            }
            let date = now_utc();
            let jrows: Vec<Value> = rows.iter().map(|s| {
                let mut v = stats_json(s);
                v["date"] = json!(date); v["repo"] = json!(repo); v["file"] = json!(file); v["rev"] = json!(rev);
                v["chunk_bytes"] = json!(cfg.chunk_bytes); v["piece_bytes"] = json!(cfg.piece_bytes);
                v
            }).collect();
            let doc = json!({"date": date, "repo": repo, "file": file, "rev": rev, "url": url, "limit_bytes": limit,
                "chunk_bytes": cfg.chunk_bytes, "piece_bytes": cfg.piece_bytes, "rows": jrows});
            std::fs::write(&out_json, serde_json::to_string_pretty(&doc).unwrap()).unwrap_or_else(|e| die(&e.to_string()));
            eprintln!("wrote {}", out_json.display());
            final_line(tb, ts, tlist.iter().copied().max().unwrap_or(0));
        }
        _ => {
            eprintln!("usage: sd-fetch file|repo|tensors|bench [flags]  (see source header)");
            std::process::exit(2);
        }
    }
}
