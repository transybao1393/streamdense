//! Answer library collector + `streamdense library stats|export` (watcher `library = true`; OFF by default).
//! Every gateway turn with >= `MIN_ANSWER_TOKENS` generated tokens becomes one record (format:
//! streamdense_core::library): the prompt's and the answer's per-neuron FFN contribution |a|·‖W_down col‖ as u8 log
//! codes + a count sketch of the prompt for kNN, the turn's coarse label, token / row counts. No prompt text unless
//! `library_text = true`.
//! Cost (no GPU work added; the activity recorder already sums every row):
//! - worker thread, once per turn at the first generated token: `ActivityTap::mark_prompt` copies the FFN sums
//!   (layers·inter f32; 4.5 MB on the 27B, one memcpy);
//! - sd-heat (background QoS), recorder gate closed: swap the mark out (O(1)) + one pass answer = live − mark;
//! - sd-heat after the gate opened: encode (log codes, sketch, checksum) + one pwrite of the slot.
//! Heap (27B): mark 4.5 MB (tap) + prompt 4.5 + answer 4.5 + slot 2.4 + scratch 0.14 = 16 MB, fixed at open.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use streamdense_core::library::{Dims, Encoder, Lib, Meta, Ring};
use streamdense_model::activity::ActivityTap;

/// Turns with fewer generated tokens are not recorded (a few-token answer says little about the answer set).
pub const MIN_ANSWER_TOKENS: u32 = 8;
/// A record that would grow the file is skipped while the disk has less free space than this.
pub const MIN_FREE_BYTES: u64 = 2_000_000_000;

/// What the job hook hands over (worker thread). `text` only with `library_text = true`.
#[derive(Clone, Debug, Default)]
pub struct LibJob {
    pub n_prompt: u32,
    pub n_reused: u32,
    pub n_feed: u32,
    pub n_new: u32,
    pub mtp: bool,
    pub spec: Option<(usize, usize, usize, f64)>,
    pub text: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct LibStats {
    pub written: u64,
    pub evicted: u64,
    pub skipped_short: u64,
    pub skipped_nomark: u64,
    pub skipped_disk: u64,
    pub errors: u64,
    pub last_error: Option<String>,
    /// (count, total ms, max ms) of capture (gate closed), encode, write
    pub capture: (u64, f64, f64),
    pub encode: (u64, f64, f64),
    pub write: (u64, f64, f64),
}
fn add(t: &mut (u64, f64, f64), ms: f64) { t.0 += 1; t.1 += ms; t.2 = t.2.max(ms); }
fn mean(t: &(u64, f64, f64)) -> f64 { if t.0 > 0 { t.1 / t.0 as f64 } else { 0.0 } }
fn r3(x: f64) -> f64 { (x * 1000.0).round() / 1000.0 }

/// Free bytes on the file system holding `p` (u64::MAX when unknown).
pub fn free_bytes(p: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(p.as_os_str().as_bytes()) else { return u64::MAX };
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 { return u64::MAX; }
    s.f_bavail as u64 * s.f_frsize as u64
}

pub struct Collector {
    pub ring: Ring,
    enc: Encoder,
    tap: Arc<ActivityTap>,
    down: Box<dyn FnMut(usize, &mut [f32]) + Send>,
    pro: Vec<f32>,
    ans: Vec<f32>,
    slot: Vec<u8>,
    pending: Option<Meta>,
    pub text: bool,
    pub dir: PathBuf,
    pub min_free: u64,
    pub st: LibStats,
}

impl Collector {
    /// Open the ring under `dir` (cap bytes), enable the tap's prompt mark. `down(l, out)`: layer l's down column norms.
    pub fn open(dir: &Path, cap: u64, text: bool, tap: Arc<ActivityTap>, down: Box<dyn FnMut(usize, &mut [f32]) + Send>) -> Result<Collector, String> {
        let t = tap.template();
        let d = Dims::new(t.layers, t.inter);
        let ring = Ring::open(dir, d, cap, crate::watcher::now_s()).map_err(|e| format!("library {}: {}", dir.display(), e))?;
        tap.enable_prompt_mark();
        Ok(Collector { enc: Encoder::new(d), tap, down, pro: vec![0.0; d.n()], ans: vec![0.0; d.n()], slot: vec![0u8; d.slot_bytes()], pending: None,
                       text, dir: dir.to_path_buf(), min_free: MIN_FREE_BYTES, st: LibStats::default(), ring })
    }
    /// Host heap held (vectors + slot + scratch; the tap's mark copy counted too).
    pub fn heap_bytes(&self) -> usize { 4 * (3 * self.pro.len()) + self.slot.len() + self.enc.heap_bytes() }

    /// sd-heat, recorder gate closed (the recorder holds exactly this job): take the job's prompt mark and the answer
    /// sums (live − mark). `label` / `active`: the watcher's label of the turn and the active domain.
    pub fn capture(&mut self, job: &LibJob, label: &str, active: &str) {
        let t0 = std::time::Instant::now();
        let Some(prow) = self.tap.take_prompt_mark(&mut self.pro) else { self.st.skipped_nomark += 1; return };
        if job.n_new < MIN_ANSWER_TOKENS { self.st.skipped_short += 1; return; }
        let rows = self.tap.counters().0;
        let (pro, ans) = (&self.pro, &mut self.ans);
        self.tap.with_live(|x| for ((a, &l), &p) in ans.iter_mut().zip(&x[..pro.len()]).zip(pro) { *a = (l - p).max(0.0); });
        let (sp, sa, sr) = job.spec.map_or((0, 0, 0), |s| (s.0 as u32, s.1 as u32, s.2 as u32));
        self.pending = Some(Meta { seq: 0, ts: crate::watcher::now_s(), label: label.into(), active: active.into(), prompt_rows: prow,
                                   answer_rows: rows.wrapping_sub(prow), n_prompt: job.n_prompt, n_reused: job.n_reused, n_feed: job.n_feed, n_new: job.n_new,
                                   mtp: job.mtp, spec_passes: sp, spec_accepted: sa, spec_rejects: sr,
                                   text: if self.text { job.text.clone() } else { None } });
        add(&mut self.st.capture, t0.elapsed().as_secs_f64() * 1e3);
    }

    /// sd-heat, after the gate opened: encode + write the pending record (false = none pending / skipped / error).
    pub fn flush(&mut self) -> bool {
        let Some(mut m) = self.pending.take() else { return false };
        let (_, replaces) = self.ring.next_slot();
        if !replaces && free_bytes(&self.dir) < self.min_free { self.st.skipped_disk += 1; return false; }
        m.seq = self.ring.next_seq();
        let t0 = std::time::Instant::now();
        self.enc.encode(&m, &self.pro, &self.ans, &mut *self.down, &mut self.slot);
        add(&mut self.st.encode, t0.elapsed().as_secs_f64() * 1e3);
        let t1 = std::time::Instant::now();
        match self.ring.append(&self.slot) {
            Ok((_, ev)) => { self.st.written += 1; if ev { self.st.evicted += 1; } }
            Err(e) => { self.st.errors += 1; self.st.last_error = Some(e.to_string()); eprintln!("library: write: {}", e); return false; }
        }
        add(&mut self.st.write, t1.elapsed().as_secs_f64() * 1e3);
        true
    }

    /// The tap's mark cost on the worker thread: (marks, mean ms, max ms).
    pub fn mark_cost(&self) -> (u64, f64, f64) {
        let n = self.tap.mark_n.load(Relaxed);
        (n, if n > 0 { self.tap.mark_ns.load(Relaxed) as f64 / n as f64 / 1e6 } else { 0.0 }, self.tap.mark_max_ns.load(Relaxed) as f64 / 1e6)
    }
    pub fn line(&self) -> String {
        format!("library {} rec, {:.0} MB / {:.2} GB", self.ring.count(), self.ring.disk_bytes() as f64 / 1e6, self.ring.cap_bytes() as f64 / 1e9)
    }
    pub fn json(&self) -> Value {
        let s = &self.st;
        let (mn, mm, mx) = self.mark_cost();
        json!({"on": true, "file": self.ring.path.display().to_string(), "records": self.ring.count(), "slots": self.ring.slots,
               "disk_bytes": self.ring.disk_bytes(), "cap_bytes": self.ring.cap_bytes(), "record_bytes": self.ring.d.slot_bytes(), "text": self.text,
               "written": s.written, "evicted": s.evicted, "skipped_short": s.skipped_short, "skipped_nomark": s.skipped_nomark,
               "skipped_disk": s.skipped_disk, "errors": s.errors, "last_error": s.last_error, "min_answer_tokens": MIN_ANSWER_TOKENS,
               "cost": {"mark_worker_ms_mean": r3(mm), "mark_worker_ms_max": r3(mx), "marks": mn,
                        "capture_gate_ms_mean": r3(mean(&s.capture)), "capture_gate_ms_max": r3(s.capture.2),
                        "encode_bg_ms_mean": r3(mean(&s.encode)), "encode_bg_ms_max": r3(s.encode.2),
                        "write_bg_ms_mean": r3(mean(&s.write)), "write_bg_ms_max": r3(s.write.2), "heap_bytes": self.heap_bytes()},
               "line": self.line()})
    }
}

// ---- offline CLI ---------------------------------------------------------------------------------------------------

fn lib_path(a: &crate::Args) -> PathBuf {
    if let Some(f) = a.s("--file") { return PathBuf::from(f); }
    a.s("--dir").map(|d| crate::watcher::expand_home(&d))
        .unwrap_or_else(|| crate::watcher::config_dir(&crate::load_config().cfg.watcher.dir).join("library"))
}

/// One record's metadata as JSON (`text` only when stored and asked for).
pub fn meta_json(m: &Meta, slot: usize, ok: Option<bool>, text: bool) -> Value {
    let mut v = json!({"seq": m.seq, "ts": m.ts, "slot": slot, "label": m.label, "active": m.active, "prompt_rows": m.prompt_rows,
        "answer_rows": m.answer_rows, "n_prompt": m.n_prompt, "n_reused": m.n_reused, "n_feed": m.n_feed, "n_new": m.n_new, "mtp": m.mtp,
        "spec": [m.spec_passes, m.spec_accepted, m.spec_rejects], "has_text": m.text.is_some()});
    if let Some(o) = ok { v["checksum_ok"] = json!(o); }
    if text { if let Some(t) = &m.text { v["text"] = json!(t); } }
    v
}

/// `streamdense library stats [--dir DIR | --file F] [--verify] [--json]` / `export [--out FILE] [--text] [--verify]`
pub fn main(a: &crate::Args) {
    let sub = a.0.get(2).cloned().unwrap_or_default();
    let p = lib_path(a);
    let lib = match Lib::open(&p) { Ok(l) => l, Err(e) => { eprintln!("ERROR: {}: {}", p.display(), e); std::process::exit(2) } };
    let recs = lib.records();
    let verify = a.has("--verify");
    let t0 = std::time::Instant::now();
    match sub.as_str() {
        "stats" => {
            let (mut by, mut bad, mut texts) = (std::collections::BTreeMap::<String, u64>::new(), 0u64, 0u64);
            let (mut pr, mut ar, mut nn, mut fed) = (0f64, 0f64, 0f64, 0f64);
            let (mut t_old, mut t_new) = (f64::INFINITY, 0f64);
            for (k, &(_, i)) in recs.iter().enumerate() {
                let Ok((m, _)) = lib.meta(i) else { bad += 1; continue };
                if verify && !lib.verify(i) { bad += 1; }
                *by.entry(m.label.clone()).or_default() += 1;
                if m.text.is_some() { texts += 1; }
                pr += m.prompt_rows as f64; ar += m.answer_rows as f64; nn += m.n_new as f64; fed += m.n_feed as f64;
                t_old = t_old.min(m.ts); t_new = t_new.max(m.ts);
                if verify && k % 50 == 49 { eprintln!("library: verified {}/{} ({:.1} s)", k + 1, recs.len(), t0.elapsed().as_secs_f64()); }
            }
            let n = recs.len().max(1) as f64;
            let v = json!({"file": lib.path.display().to_string(), "dims": lib.d.tag(), "records": recs.len(), "slots": lib.slots,
                "record_bytes": lib.d.slot_bytes(), "disk_bytes": lib.disk_bytes(), "cap_bytes": streamdense_core::library::FILE_HDR + (lib.slots * lib.d.slot_bytes()) as u64,
                "by_label": by, "with_text": texts, "bad": if verify { json!(bad) } else { json!("not checked (--verify)") },
                "mean": {"prompt_rows": pr / n, "answer_rows": ar / n, "answer_tokens": nn / n, "fed_tokens": fed / n,
                         "answer_rows_per_token": if nn > 0.0 { ar / nn } else { 0.0 }},
                "oldest_ts": if recs.is_empty() { Value::Null } else { json!(t_old) }, "newest_ts": if recs.is_empty() { Value::Null } else { json!(t_new) }});
            if a.has("--json") { println!("{}", serde_json::to_string_pretty(&v).unwrap()); return; }
            println!("library {} ({})", v["file"].as_str().unwrap(), v["dims"].as_str().unwrap());
            println!("  records {} / {} slots | {:.1} MB on disk, cap {:.2} GB | {:.2} MB per record", recs.len(), lib.slots, lib.disk_bytes() as f64 / 1e6,
                     v["cap_bytes"].as_u64().unwrap() as f64 / 1e9, lib.d.slot_bytes() as f64 / 1e6);
            println!("  by label: {}", serde_json::to_string(&v["by_label"]).unwrap());
            println!("  mean per record: prompt rows {:.0}, answer rows {:.0}, answer tokens {:.0} (rows per token {:.2}: > 1 = MTP verify rows incl. rejected drafts)",
                     pr / n, ar / n, nn / n, if nn > 0.0 { ar / nn } else { 0.0 });
            println!("  with text: {} | checksum: {}", texts, if verify { format!("{} bad", bad) } else { "not checked (--verify)".into() });
        }
        "export" => {
            use std::io::Write;
            let mut out: Box<dyn Write> = match a.s("--out") {
                Some(f) => Box::new(std::io::BufWriter::new(std::fs::File::create(&f).unwrap_or_else(|e| { eprintln!("ERROR: {}: {}", f, e); std::process::exit(2) }))),
                None => Box::new(std::io::BufWriter::new(std::io::stdout())),
            };
            for (k, &(_, i)) in recs.iter().enumerate() {
                let Ok((m, _)) = lib.meta(i) else { continue };
                let ok = verify.then(|| lib.verify(i));
                let _ = writeln!(out, "{}", meta_json(&m, i, ok, a.has("--text")));
                if k % 200 == 199 { eprintln!("library: exported {}/{}", k + 1, recs.len()); }
            }
            eprintln!("library: exported {} records (metadata JSONL; vectors stay in {})", recs.len(), lib.path.display());
        }
        _ => { eprintln!("usage: streamdense library stats|export [--dir DIR | --file F] [--verify] [--json] [--out FILE] [--text]"); std::process::exit(2) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use streamdense_core::library::{Lib, FILE_HDR};

    /// Cost at 27B sizes (synthetic sums, no model): the worker-thread mark (one memcpy of layers·inter f32), the
    /// gate-closed capture pass (answer = live − mark) and the background encode + write per record.
    /// cargo test --release -p streamdense-cli --bin streamdense library::tests::cost_27b -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost_27b() {
        let d = Dims::new(64, 17408);
        let n = d.n();
        let mut s = 7u64;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32).powi(6) * 50.0 };
        let live: Vec<f32> = (0..n).map(|_| r() * 2.0).collect();
        let mut mark = vec![0f32; n];
        let mut ans = vec![0f32; n];
        let dir = std::env::temp_dir().join(format!("sd_library_cost_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut ring = Ring::open(&dir, d, FILE_HDR + 6 * d.slot_bytes() as u64, 1.0).unwrap();
        let mut enc = Encoder::new(d);
        let mut slot = vec![0u8; d.slot_bytes()];
        let dn: Vec<f32> = (0..d.inter).map(|j| 0.5 + (j % 7) as f32 * 0.1).collect();
        let (mut tm, mut tc, mut te, mut tw) = (0f64, 0f64, 0f64, 0f64);
        let reps = 8;
        for k in 0..reps {
            let t0 = std::time::Instant::now();
            mark.copy_from_slice(&live); // worker thread, first token
            tm += t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            for ((a, &l), &p) in ans.iter_mut().zip(&live).zip(&mark) { *a = (l * 1.7 - p).max(0.0); } // sd-heat, gate closed
            tc += t1.elapsed().as_secs_f64();
            crate::heat::set_background_qos();
            let m = Meta { seq: ring.next_seq(), ts: k as f64, label: "code-rust".into(), n_new: 100, ..Default::default() };
            let t2 = std::time::Instant::now();
            enc.encode(&m, &mark, &ans, &mut |_, o| o.copy_from_slice(&dn), &mut slot);
            te += t2.elapsed().as_secs_f64();
            let t3 = std::time::Instant::now();
            ring.append(&slot).unwrap();
            tw += t3.elapsed().as_secs_f64();
        }
        let ms = |t: f64| t * 1e3 / reps as f64;
        eprintln!("27B library record {} B ({:.2} MB; {:.0} per GB, {:.2} GB per 1,000 turns): mark (worker) {:.2} ms, capture (gate closed) {:.2} ms, \
                   encode (background) {:.1} ms, write {:.1} ms; heap {:.1} MB", d.slot_bytes(), d.slot_bytes() as f64 / 1e6, 1e9 / d.slot_bytes() as f64,
                  d.slot_bytes() as f64 * 1000.0 / 1e9, ms(tm), ms(tc), ms(te), ms(tw), (3 * 4 * n + d.slot_bytes() + enc.heap_bytes()) as f64 / 1e6);
        assert_eq!((ring.count(), ring.disk_bytes()), (6, FILE_HDR + 6 * d.slot_bytes() as u64), "ring full at its cap");
        let lib = Lib::open(&dir).unwrap();
        assert!(lib.records().iter().all(|x| lib.verify(x.1)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
