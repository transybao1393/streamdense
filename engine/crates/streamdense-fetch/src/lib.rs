//! Parallel ranged HTTP fetcher (std threads + ureq).
//!
//! Workers pull chunks (<= `chunk_bytes`) from a shared queue, issue
//! `Range: bytes=a-b` GETs (206 required), and stream the body in
//! `piece_bytes` buffers straight into the output file with `write_at`.
//! RAM stays at about `threads * piece_bytes`.

pub mod hf;

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ureq::ResponseExt;

const MIB: u64 = 1 << 20;

pub struct FetchConfig {
    pub threads: usize,
    pub chunk_bytes: u64,
    pub piece_bytes: usize,
    pub retries: u32,
    pub token: Option<String>,
    pub timeout_s: u64,
}

impl Default for FetchConfig {
    fn default() -> Self {
        FetchConfig {
            threads: 8,
            chunk_bytes: 32 * MIB,
            piece_bytes: MIB as usize,
            retries: 5,
            token: default_token(),
            timeout_s: 60,
        }
    }
}

/// `HF_TOKEN` env, else `~/.cache/huggingface/token`.
fn default_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join(".cache/huggingface/token");
    let t = std::fs::read_to_string(p).ok()?.trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

#[derive(Debug, Clone)]
pub struct Head {
    pub final_url: String,
    pub size: u64,
    pub accept_ranges: bool,
    pub etag: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub src_offset: u64,
    pub len: u64,
    pub dst_offset: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub bytes: u64,
    pub secs: f64,
    pub mb_per_s: f64,
    pub chunks: usize,
    pub retries: u32,
    pub threads: usize,
    pub per_thread_bytes: Vec<u64>,
    pub first_byte_ms: f64,
}

pub fn agent(cfg: &FetchConfig) -> ureq::Agent {
    let t = Duration::from_secs(cfg.timeout_s.max(1));
    // Whole-body budget: timeout plus the chunk at a floor of 256 KiB/s.
    // A stalled chunk then fails and is retried from the bytes already written.
    let body_t = t + Duration::from_secs(cfg.chunk_bytes / (256 * 1024) + 1);
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(10)
        .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
        .timeout_connect(Some(t))
        .timeout_recv_response(Some(t))
        .timeout_recv_body(Some(body_t))
        .max_idle_connections_per_host(cfg.threads.max(1) * 2)
        .max_idle_connections(cfg.threads.max(1) * 4)
        .build()
        .into()
}

fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let auth = rest.split(['/', '?', '#']).next().unwrap_or("");
    let hostport = auth.rsplit('@').next().unwrap_or(auth);
    hostport.split(':').next().unwrap_or(hostport)
}

fn is_hf_host(url: &str) -> bool {
    let h = host_of(url).to_ascii_lowercase();
    h == "huggingface.co" || h.ends_with(".huggingface.co")
}

fn auth_header(url: &str, cfg: &FetchConfig) -> Option<String> {
    match &cfg.token {
        Some(t) if is_hf_host(url) => Some(format!("Bearer {t}")),
        _ => None,
    }
}

fn header_str(resp: &ureq::http::Response<ureq::Body>, name: &str) -> Option<String> {
    resp.headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Parse total size from `Content-Range: bytes a-b/total`.
fn content_range_total(v: &str) -> Option<u64> {
    v.rsplit('/').next()?.trim().parse().ok()
}

/// HEAD with redirect following; falls back to a `bytes=0-0` GET when HEAD
/// does not yield the size.
pub fn head(url: &str, cfg: &FetchConfig) -> Result<Head, String> {
    let ag = agent(cfg);
    let mut req = ag.head(url);
    if let Some(a) = auth_header(url, cfg) {
        req = req.header("Authorization", a);
    }
    let resp = req.call().map_err(|e| format!("HEAD {url}: {e}"))?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(format!("HEAD {url}: status {status}"));
    }
    let final_url = resp.get_uri().to_string();
    let mut accept_ranges = header_str(&resp, "accept-ranges")
        .map(|v| v.to_ascii_lowercase().contains("bytes"))
        .unwrap_or(false);
    let mut etag = header_str(&resp, "x-linked-etag").or_else(|| header_str(&resp, "etag"));
    let mut size = header_str(&resp, "content-length").and_then(|v| v.trim().parse::<u64>().ok());
    drop(resp);

    if size.is_none() || size == Some(0) || !accept_ranges {
        let mut req = ag.get(&final_url).header("Range", "bytes=0-0");
        if let Some(a) = auth_header(&final_url, cfg) {
            req = req.header("Authorization", a);
        }
        let resp = req.call().map_err(|e| format!("GET 0-0 {final_url}: {e}"))?;
        let status = resp.status().as_u16();
        if status == 206 {
            accept_ranges = true;
            if let Some(t) = header_str(&resp, "content-range").and_then(|v| content_range_total(&v)) {
                size = Some(t);
            }
        } else if (200..300).contains(&status) {
            if size.is_none() {
                size = header_str(&resp, "content-length").and_then(|v| v.trim().parse().ok());
            }
        } else {
            return Err(format!("GET 0-0 {final_url}: status {status}"));
        }
        if etag.is_none() {
            etag = header_str(&resp, "etag");
        }
        // Do not read the body (it may be the full file on a 200).
    }
    let size = size.ok_or_else(|| format!("{url}: server gave no size"))?;
    Ok(Head { final_url, size, accept_ranges, etag })
}

/// Fetch `[start, end_excl)` into memory. Requires 206. For small reads
/// (headers, indexes); use `fetch_ranges` for bulk data.
pub fn get_range(url: &str, start: u64, end_excl: u64, cfg: &FetchConfig) -> Result<Vec<u8>, String> {
    if end_excl <= start {
        return Ok(Vec::new());
    }
    let ag = agent(cfg);
    let want = end_excl - start;
    let mut last = String::new();
    for attempt in 0..=cfg.retries {
        if attempt > 0 {
            std::thread::sleep(backoff(attempt));
        }
        let mut req = ag.get(url).header("Range", format!("bytes={}-{}", start, end_excl - 1));
        if let Some(a) = auth_header(url, cfg) {
            req = req.header("Authorization", a);
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(e) => {
                last = format!("GET {url} [{start},{end_excl}): {e}");
                continue;
            }
        };
        let status = resp.status().as_u16();
        if status != 206 {
            last = format!("GET {url} [{start},{end_excl}): status {status}, want 206");
            if status == 429 || status >= 500 {
                continue;
            }
            return Err(last);
        }
        let mut buf = Vec::with_capacity(want as usize);
        let mut rd = resp.into_body().into_reader().take(want);
        match rd.read_to_end(&mut buf) {
            Ok(_) if buf.len() as u64 == want => return Ok(buf),
            Ok(_) => last = format!("GET {url}: short body {} of {want}", buf.len()),
            Err(e) => last = format!("GET {url}: read: {e}"),
        }
    }
    Err(last)
}

fn backoff(attempt: u32) -> Duration {
    Duration::from_millis((250u64 << attempt.min(5)).min(8000))
}

/// One unit of work: a contiguous source span written at `dst`.
#[derive(Clone, Copy)]
struct Chunk {
    src: u64,
    len: u64,
    dst: u64,
}

fn split_ranges(ranges: &[Range], chunk_bytes: u64) -> Vec<Chunk> {
    let cb = chunk_bytes.max(1);
    let mut out = Vec::new();
    for r in ranges {
        let mut off = 0;
        while off < r.len {
            let len = (r.len - off).min(cb);
            out.push(Chunk { src: r.src_offset + off, len, dst: r.dst_offset + off });
            off += len;
        }
    }
    out
}

/// Shared state for a parallel fetch.
struct Shared<'a> {
    url: &'a str,
    cfg: &'a FetchConfig,
    agent: ureq::Agent,
    out: &'a File,
    chunks: &'a [(usize, Chunk)],
    next: AtomicUsize,
    abort: AtomicBool,
    err: Mutex<Option<String>>,
    retries: AtomicU32,
    done_bytes: AtomicU64,
    total_bytes: u64,
    t0: Instant,
    first_byte_ns: AtomicU64,
    progress: Option<&'a (dyn Fn(u64, u64) + Sync)>,
    on_chunk_done: Option<&'a (dyn Fn(usize) + Sync)>,
}

impl Shared<'_> {
    fn add_bytes(&self, n: u64) {
        let d = self.done_bytes.fetch_add(n, Ordering::Relaxed) + n;
        if let Some(p) = self.progress {
            p(d, self.total_bytes);
        }
    }

    fn fail(&self, e: String) {
        let mut g = self.err.lock().unwrap();
        if g.is_none() {
            *g = Some(e);
        }
        self.abort.store(true, Ordering::SeqCst);
    }

    /// One attempt at `[c.src + done, c.src + c.len)`. Advances `done` as
    /// pieces land on disk. Err((msg, retryable)).
    fn attempt(&self, c: &Chunk, done: &mut u64, buf: &mut [u8], my_bytes: &mut u64) -> Result<(), (String, bool)> {
        let a = c.src + *done;
        let b = c.src + c.len - 1;
        let mut req = self.agent.get(self.url).header("Range", format!("bytes={a}-{b}"));
        if let Some(h) = auth_header(self.url, self.cfg) {
            req = req.header("Authorization", h);
        }
        let resp = req.call().map_err(|e| (format!("GET bytes={a}-{b}: {e}"), true))?;
        let status = resp.status().as_u16();
        if status != 206 {
            let retry = status == 429 || status == 408 || status >= 500;
            return Err((format!("GET bytes={a}-{b}: status {status}, want 206"), retry));
        }
        let remaining = c.len - *done;
        let mut rd = resp.into_body().into_reader().take(remaining);
        while *done < c.len {
            if self.abort.load(Ordering::Relaxed) {
                return Err(("aborted".into(), false));
            }
            // Fill up to one piece (or the rest of the chunk), then write it.
            let want = ((c.len - *done) as usize).min(buf.len());
            let mut got = 0;
            while got < want {
                match rd.read(&mut buf[got..want]) {
                    Ok(0) => break,
                    Ok(n) => {
                        if got == 0 && *done == 0 {
                            let ns = self.t0.elapsed().as_nanos() as u64;
                            self.first_byte_ns.fetch_min(ns, Ordering::Relaxed);
                        }
                        got += n;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => {
                        self.flush_piece(c, done, &buf[..got], my_bytes)?;
                        return Err((format!("read bytes={a}-{b}: {e}"), true));
                    }
                }
            }
            self.flush_piece(c, done, &buf[..got], my_bytes)?;
            if got < want {
                return Err((format!("short body bytes={a}-{b}: {} of {}", *done, c.len), true));
            }
        }
        Ok(())
    }

    fn flush_piece(&self, c: &Chunk, done: &mut u64, piece: &[u8], my_bytes: &mut u64) -> Result<(), (String, bool)> {
        if piece.is_empty() {
            return Ok(());
        }
        self.out
            .write_all_at(piece, c.dst + *done)
            .map_err(|e| (format!("write_at {}: {e}", c.dst + *done), false))?;
        let n = piece.len() as u64;
        *done += n;
        *my_bytes += n;
        self.add_bytes(n);
        Ok(())
    }

    fn worker(&self) -> u64 {
        let mut buf = vec![0u8; self.cfg.piece_bytes.max(4096)];
        let mut my_bytes = 0u64;
        loop {
            if self.abort.load(Ordering::Relaxed) {
                break;
            }
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(&(idx, c)) = self.chunks.get(i) else { break };
            let mut done = 0u64;
            let mut attempt = 0u32;
            loop {
                match self.attempt(&c, &mut done, &mut buf, &mut my_bytes) {
                    Ok(()) => {
                        if let Some(f) = self.on_chunk_done {
                            f(idx);
                        }
                        break;
                    }
                    Err((e, retryable)) => {
                        if self.abort.load(Ordering::Relaxed) {
                            break;
                        }
                        if !retryable || attempt >= self.cfg.retries {
                            self.fail(format!("chunk {idx} after {attempt} retries: {e}"));
                            break;
                        }
                        attempt += 1;
                        self.retries.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(backoff(attempt));
                    }
                }
            }
        }
        my_bytes
    }
}

fn run_chunks(
    url: &str,
    chunks: &[(usize, Chunk)],
    out: &File,
    cfg: &FetchConfig,
    progress: Option<&(dyn Fn(u64, u64) + Sync)>,
    on_chunk_done: Option<&(dyn Fn(usize) + Sync)>,
) -> Result<Stats, String> {
    let threads = cfg.threads.max(1).min(chunks.len().max(1));
    let sh = Shared {
        url,
        cfg,
        agent: agent(cfg),
        out,
        chunks,
        next: AtomicUsize::new(0),
        abort: AtomicBool::new(false),
        err: Mutex::new(None),
        retries: AtomicU32::new(0),
        done_bytes: AtomicU64::new(0),
        total_bytes: chunks.iter().map(|(_, c)| c.len).sum(),
        t0: Instant::now(),
        first_byte_ns: AtomicU64::new(u64::MAX),
        progress,
        on_chunk_done,
    };
    let per_thread: Vec<u64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|_| s.spawn(|| sh.worker())).collect();
        hs.into_iter().map(|h| h.join().unwrap_or(0)).collect()
    });
    let secs = sh.t0.elapsed().as_secs_f64();
    if let Some(e) = sh.err.lock().unwrap().take() {
        return Err(e);
    }
    let bytes = sh.done_bytes.load(Ordering::Relaxed);
    let fb = sh.first_byte_ns.load(Ordering::Relaxed);
    Ok(Stats {
        bytes,
        secs,
        mb_per_s: if secs > 0.0 { bytes as f64 / MIB as f64 / secs } else { 0.0 },
        chunks: chunks.len(),
        retries: sh.retries.load(Ordering::Relaxed),
        threads,
        per_thread_bytes: per_thread,
        first_byte_ms: if fb == u64::MAX { 0.0 } else { fb as f64 / 1e6 },
    })
}

/// Fetch `ranges` of `url` (use `Head::final_url`) into `out` in parallel.
/// `progress(done_bytes, total_bytes)` is called from worker threads.
pub fn fetch_ranges(
    url: &str,
    ranges: &[Range],
    out: &File,
    cfg: &FetchConfig,
    progress: Option<&(dyn Fn(u64, u64) + Sync)>,
) -> Result<Stats, String> {
    let chunks: Vec<(usize, Chunk)> = split_ranges(ranges, cfg.chunk_bytes).into_iter().enumerate().collect();
    run_chunks(url, &chunks, out, cfg, progress, None)
}

fn sidecar_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_os_string();
    s.push(".sdfetch");
    PathBuf::from(s)
}

/// Done chunk indexes from a sidecar that matches this url/size/chunk size.
fn load_sidecar(p: &Path, url: &str, size: u64, chunk: u64) -> HashSet<usize> {
    let Ok(txt) = std::fs::read_to_string(p) else { return HashSet::new() };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else { return HashSet::new() };
    if v["url"].as_str() != Some(url) || v["size"].as_u64() != Some(size) || v["chunk_bytes"].as_u64() != Some(chunk) {
        return HashSet::new();
    }
    v["done"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default()
}

fn write_sidecar(p: &Path, url: &str, size: u64, chunk: u64, done: &HashSet<usize>) -> Result<(), String> {
    let mut d: Vec<usize> = done.iter().copied().collect();
    d.sort_unstable();
    let v = serde_json::json!({ "url": url, "size": size, "chunk_bytes": chunk, "done": d });
    let tmp = {
        let mut s = p.as_os_str().to_os_string();
        s.push(".tmp");
        PathBuf::from(s)
    };
    std::fs::write(&tmp, v.to_string()).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, p).map_err(|e| format!("rename {}: {e}", p.display()))
}

/// Download `url` to `out_path`. `limit_bytes` fetches only the first B
/// bytes (benchmarks). With `resume`, a `<out>.sdfetch` sidecar records
/// finished chunks; it is removed on success.
pub fn fetch_file(
    url: &str,
    out_path: &Path,
    cfg: &FetchConfig,
    limit_bytes: Option<u64>,
    resume: bool,
) -> Result<Stats, String> {
    let h = head(url, cfg)?;
    if !h.accept_ranges {
        return Err(format!("{url}: server does not support byte ranges"));
    }
    let target = limit_bytes.map_or(h.size, |l| l.min(h.size));
    let chunk = cfg.chunk_bytes.max(1);
    let side = sidecar_path(out_path);

    let done0 = if resume && out_path.exists() {
        load_sidecar(&side, url, target, chunk)
    } else {
        HashSet::new()
    };
    let out = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(done0.is_empty())
        .open(out_path)
        .map_err(|e| format!("open {}: {e}", out_path.display()))?;
    out.set_len(target).map_err(|e| format!("set_len {}: {e}", out_path.display()))?;

    let all = split_ranges(&[Range { src_offset: 0, len: target, dst_offset: 0 }], chunk);
    let todo: Vec<(usize, Chunk)> = all.into_iter().enumerate().filter(|(i, _)| !done0.contains(i)).collect();

    let done = Mutex::new(done0);
    let side_err: Mutex<Option<String>> = Mutex::new(None);
    let on_done = |idx: usize| {
        if !resume {
            return;
        }
        let mut d = done.lock().unwrap();
        d.insert(idx);
        if let Err(e) = write_sidecar(&side, url, target, chunk, &d) {
            side_err.lock().unwrap().get_or_insert(e);
        }
    };
    if resume {
        write_sidecar(&side, url, target, chunk, &done.lock().unwrap())?;
    }
    let stats = run_chunks(&h.final_url, &todo, &out, cfg, None, Some(&on_done))?;
    if let Some(e) = side_err.into_inner().unwrap() {
        eprintln!("streamdense-fetch: sidecar warning: {e}");
    }
    out.sync_all().map_err(|e| format!("fsync {}: {e}", out_path.display()))?;
    if resume {
        let _ = std::fs::remove_file(&side);
    }
    Ok(stats)
}
