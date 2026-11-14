//! Low-level watcher: one dedicated thread at background QoS (macOS QOS_CLASS_BACKGROUND: the
//! efficiency cores; never the engine / encode / HTTP threads). Heat maps of every FFN neuron (contribution
//! |a|·‖W_down col‖), DeltaNet / attention head and layer resid, the MTP draft layer (FFN |a| + heads), output tokens
//! emitted (lm_head rows) and embedding rows read (per token id), on three time scales: now (tau 5 min), session
//! (tau 4 h), all (no decay).
//! Heat on disk: the maps ARE the disk store (`store.rs`): now / session = the pooled
//! decayed sets, all = the per-label all-time sets (+ the held slot) summed while streaming. Off the job hook
//! (2026-11-01): the hook only closes the recorder gate and hands the job over (`Shared::submit`); THIS thread scores,
//! labels and gates it from the recorder in place, adds it into the batch region (model memory; one disk fold per
//! batch: label change, `fold_every` jobs, `BATCH_IDLE_S` idle, SIGINT / SIGTERM), zeroes the recorder and opens the
//! gate; the next job's first recorder write waits only if that is not done yet (`next_job_waits`). This
//! thread ticks every `sample_ms` (default 100) on two live GPU counters (activity / idle), drains the audit queue
//! when the engine is idle (>= 0.5 s without a pass), and writes the summary (30 s after activity, or at idle) by
//! streaming the sets (exact radix selection, `store::hot`) straight into a JSON buffer on its own stack, then
//! heat/<tag>.json. Host heap for the numbers: 0 (fixed stack arrays; `heap_bytes` in the summary).
//! `streamdense watcher heat` prints the summary; the GET view reads the file.
use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use streamdense_model::activity::{ActivityTap, SCRIPT_NAMES};

use crate::store::{self, Buf, Hist, Rec, Sel, Store, CH, MAXH, MAXL, NOW, SESS, TAUS_S};
use crate::watcher::{Shared, NP};

pub const SCALES: [&str; 3] = ["now", "session", "all"];
pub const SUMMARY_S: f64 = 30.0;
pub const IDLE_S: f64 = 0.5;
const NS: usize = SCRIPT_NAMES.len();
/// the summary JSON buffer (sd-heat stack)
pub const JSON_MAX: usize = 16 << 10;
/// stack arrays of one summary (JSON 16 KB + chunk 8 KB + record 8 KB + radix 12 KB + per-layer / scale results)
pub const SUMMARY_STACK: usize = JSON_MAX + CH * 4 + (CH * 4 + 16) + 12 * 1024 + 3 * MAXL * 8 + MAXH * 4 + 3 * 2 * (NS + 4) * 8;

fn now_s() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0) }

/// JSON into a fixed buffer (no allocation); `ok` false when it overflowed.
pub struct Jw<'a> { pub b: &'a mut [u8], pub n: usize, pub ok: bool }
impl std::fmt::Write for Jw<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let e = self.n + s.len();
        if e > self.b.len() { self.ok = false; return Ok(()); }
        self.b[self.n..e].copy_from_slice(s.as_bytes());
        self.n = e;
        Ok(())
    }
}
impl<'a> Jw<'a> {
    pub fn new(b: &'a mut [u8]) -> Jw<'a> { Jw { b, n: 0, ok: true } }
    /// 4 decimals (null when not finite)
    pub fn f(&mut self, x: f64) { if x.is_finite() { let _ = write!(self, "{:.4}", x); } else { let _ = self.write_str("null"); } }
    /// a count as a float (`5.0`, as the old summary)
    pub fn t(&mut self, x: f64) { if x.is_finite() { let _ = write!(self, "{:.1}", x.round()); } else { let _ = self.write_str("null"); } }
    /// a JSON string (no escaping needed for the paths / names written here except quotes and backslashes)
    pub fn s(&mut self, x: &str) {
        let _ = self.write_str("\"");
        for c in x.chars() { let _ = match c { '"' => self.write_str("\\\""), '\\' => self.write_str("\\\\"), c => self.write_char(c) }; }
        let _ = self.write_str("\"");
    }
    #[cfg(test)]
    pub fn bytes(&self) -> &[u8] { &self.b[..self.n] }
}

/// Token-row results of one scale.
#[derive(Clone, Copy, Default)]
struct Tok { tokens: f64, hot: usize, frac: f64, seen: usize, by: [f64; NS], scripts: bool }
#[derive(Clone, Copy, Default)]
struct Mtp { tokens: f64, ffn: f64, head: f64 }

fn frac(p: &store::Pick, len: usize) -> f64 { if p.total > 0.0 && len > 0 { p.n as f64 / len as f64 } else { f64::NAN } }

fn tok_part(st: &Store, sel: Sel, base: usize, keep: f64, script: Option<&dyn Fn(u32) -> u8>, buf: &mut Buf, rec: &mut Rec, h: &mut Hist) -> Tok {
    let v = st.d.vocab;
    let mut t = Tok { scripts: script.is_some(), ..Default::default() };
    if v == 0 { return t; }
    let p = store::hot(v, keep, &mut |o, b| st.read_sel(sel, base + o, b, rec), buf, h);
    (t.tokens, t.hot, t.frac, t.seen) = (p.total, p.n, if p.total > 0.0 { p.n as f64 / v as f64 } else { f64::NAN }, p.seen);
    if let Some(sf) = script {
        let mut id = 0u32;
        store::each(v, &mut |o, b| st.read_sel(sel, base + o, b, rec), buf, &mut |x| { if x > 0.0 { t.by[(sf(id) as usize).min(NS - 1)] += x as f64; } id += 1; });
    }
    t
}

fn tok_json(w: &mut Jw, t: &Tok) {
    let _ = w.write_str("{\"tokens\":"); w.t(t.tokens);
    let _ = write!(w, ",\"rows_hot\":{},\"rows_hot_frac\":", t.hot); w.f(t.frac);
    let _ = write!(w, ",\"rows_seen\":{}", t.seen);
    if t.scripts && t.tokens > 0.0 {
        let _ = w.write_str(",\"scripts\":{");
        let mut first = true;
        for (i, n) in SCRIPT_NAMES.iter().enumerate() {
            if !(t.by[i] > 0.0) { continue; }
            if !first { let _ = w.write_str(","); }
            first = false;
            w.s(n); let _ = w.write_str(":"); w.f(t.by[i] / t.tokens);
        }
        let _ = w.write_str("}");
    }
    let _ = w.write_str("}");
}
fn mtp_json(w: &mut Jw, m: &Mtp) {
    let _ = w.write_str("{\"tokens\":"); w.t(m.tokens);
    let _ = w.write_str(",\"ffn_hot_frac\":"); w.f(m.ffn);
    let _ = w.write_str(",\"head_hot_frac\":"); w.f(m.head);
    let _ = w.write_str("}");
}

/// The summary fields (one JSON object, open: the caller adds its own fields and the closing brace) at clock `t`:
/// per scale (now / session = the pooled decayed sets, all = every label set) per-layer hot fractions, heads, MTP,
/// vocab / embed rows (+ script shares). Streams the sets through stack buffers; no allocation.
pub fn summary(st: &mut Store, keep: f64, t: f64, script: Option<&dyn Fn(u32) -> u8>, w: &mut Jw) {
    st.t = t;
    let (layers, inter, heads) = (st.d.layers.min(MAXL), st.d.inter, st.d.tmpl.heads);
    let mut buf = [0f32; CH];
    let mut rec: Rec = [0u8; CH * 4 + 16];
    let mut h = Hist::default();
    let mut mtp = [Mtp::default(); 3];
    let mut voc = [Tok::default(); 3];
    let mut emb = [Tok::default(); 3];
    let _ = w.write_str("{");
    for k in 0..3 {
        let sel = [Sel::Set(NOW), Sel::Set(SESS), Sel::Labels][k];
        let (tokens, mtp_tokens) = if k < 2 {
            let s = [NOW, SESS][k];
            let hd = st.hdr(s);
            let f = (-(t - hd.t0).max(0.0) / TAUS_S[k]).exp();
            (hd.rows_d * f, hd.mtp_d * f)
        } else {
            (0..NP).chain([store::HELD]).fold((0.0, 0.0), |a, s| { let hd = st.hdr(s); (a.0 + hd.tokens as f64, a.1 + hd.mtp_rows as f64) })
        };
        let (mut per, mut hh) = ([f64::NAN; MAXL], [f64::NAN; MAXL]);
        for l in 0..layers {
            let p = store::hot(inter, keep, &mut |o, b| st.read_sel(sel, l * inter + o, b, &mut rec), &mut buf, &mut h);
            per[l] = frac(&p, inter);
            let hr = st.heads_of(l);
            let mut hv = [0f32; MAXH];
            let nh = hr.len().min(MAXH);
            st.read_sel(sel, hr.start, &mut hv[..nh], &mut rec);
            let p = store::hot(nh, keep, &mut |o, b| b.copy_from_slice(&hv[o..o + b.len()]), &mut buf, &mut h);
            hh[l] = frac(&p, nh);
        }
        let om = st.o_mtp();
        let pf = store::hot(inter, keep, &mut |o, b| st.read_sel(sel, om + o, b, &mut rec), &mut buf, &mut h);
        let ph = store::hot(heads, keep, &mut |o, b| st.read_sel(sel, om + inter + o, b, &mut rec), &mut buf, &mut h);
        mtp[k] = Mtp { tokens: mtp_tokens, ffn: frac(&pf, inter), head: frac(&ph, heads) };
        voc[k] = tok_part(st, sel, st.o_out(), keep, script, &mut buf, &mut rec, &mut h);
        emb[k] = tok_part(st, sel, st.o_emb(), keep, script, &mut buf, &mut rec, &mut h);
        let mean = |v: &[f64]| { let (s, n) = v.iter().filter(|x| x.is_finite()).fold((0.0, 0), |a, x| (a.0 + x, a.1 + 1)); if n == 0 { f64::NAN } else { s / n as f64 } };
        if k > 0 { let _ = w.write_str(","); }
        w.s(SCALES[k]);
        let _ = w.write_str(":{\"tokens\":"); w.t(tokens);
        let _ = w.write_str(",\"hot_frac_mean\":"); w.f(mean(&per[..layers]));
        let _ = w.write_str(",\"head_hot_frac_mean\":"); w.f(mean(&hh[..layers]));
        let _ = w.write_str(",\"per_layer_hot\":[");
        for l in 0..layers { if l > 0 { let _ = w.write_str(","); } w.f(per[l]); }
        let _ = w.write_str("],\"coldest_layers\":[");
        // 5 coldest (finite) layers, ascending, ties by index: selection on the stack
        let mut used = [false; MAXL];
        for c in 0..5 {
            let mut best: Option<usize> = None;
            for l in 0..layers { if !used[l] && per[l].is_finite() && best.is_none_or(|b| per[l] < per[b]) { best = Some(l); } }
            let Some(b) = best else { break };
            used[b] = true;
            if c > 0 { let _ = w.write_str(","); }
            let _ = write!(w, "[{},", b); w.f(per[b]); let _ = w.write_str("]");
        }
        let _ = w.write_str("],\"mtp\":"); mtp_json(w, &mtp[k]);
        let _ = w.write_str(",\"vocab\":"); tok_json(w, &voc[k]);
        let _ = w.write_str(",\"embed\":"); tok_json(w, &emb[k]);
        let _ = w.write_str("}");
    }
    // "heat.vocab" / "heat.embed" / "heat.mtp" (per scale) as well, for readers of either shape
    for (name, part) in [("mtp", 0), ("vocab", 1), ("embed", 2)] {
        let _ = w.write_str(","); w.s(name); let _ = w.write_str(":{");
        for k in 0..3 {
            if k > 0 { let _ = w.write_str(","); }
            w.s(SCALES[k]); let _ = w.write_str(":");
            match part { 0 => mtp_json(w, &mtp[k]), 1 => tok_json(w, &voc[k]), _ => tok_json(w, &emb[k]) }
        }
        let _ = w.write_str("}");
    }
    let _ = w.write_str(",\"keep_frac\":"); w.f(keep);
    let _ = write!(w, ",\"hot\":\"share of entries holding {} of the mass (FFN: |a| x ||W_down col||)\"", keep);
    let _ = write!(w, ",\"taus_s\":[{:.1},{:.1},null]", TAUS_S[0], TAUS_S[1]);
    let labels = (0..NP).filter(|&s| st.has(s)).count();
    let _ = w.write_str(",\"all_from\":{\"dir\":"); w.s(&st.dir.to_string_lossy()); let _ = write!(w, ",\"labels\":{}}}", labels);
    let p = &st.prog;
    let _ = write!(w, ",\"jobs_folded\":{},\"fold\":{{\"last_ms\":", p.folded.load(Relaxed));
    w.f(p.last_fold_us.load(Relaxed) as f64 / 1e3);
    let _ = write!(w, ",\"last_bytes_written\":{},\"bytes_written_total\":{},\"repaired_chunks\":{},\"lost_chunks\":{}}}",
                   p.last_fold_bytes.load(Relaxed), p.bytes_written.load(Relaxed), p.repaired.load(Relaxed), p.lost.load(Relaxed));
    let _ = write!(w, ",\"disk_bytes\":{},\"disk_cap_bytes\":{},\"heap_bytes\":0,\"stack_bytes\":{}", st.disk_bytes(), st.cap, SUMMARY_STACK);
}

/// This thread -> background QoS (efficiency cores on Apple silicon). Ok = the call succeeded.
pub fn set_background_qos() -> bool {
    #[cfg(target_os = "macos")]
    unsafe { return libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0) == 0; }
    #[allow(unreachable_code)]
    false
}
/// CPU time of the calling thread (s).
pub fn thread_cpu_s() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts); }
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

pub fn heat_dir(watcher_dir: &Path) -> PathBuf { watcher_dir.join("heat") }

pub struct HeatCfg { pub sample_ms: u64, pub keep: f64, pub dir: PathBuf }

/// Start the low-level watcher thread (runs for the life of the process). `st`: a read-only handle on the watcher's
/// store (None without a recorder: audits only); `script`: script class per token id (from the tokenizer, on the fly).
pub fn spawn(tap: Option<Arc<ActivityTap>>, st: Option<Store>, script: Option<Box<dyn Fn(u32) -> u8 + Send>>, ws: Arc<Shared>, c: HeatCfg)
             -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new().name("sd-heat".into()).stack_size(256 << 10).spawn(move || run(tap, st, script, ws, c))
}

/// Write `b` to `p` atomically (tmp + rename; prebuilt paths: no allocation).
fn write_file(tmp: &Path, p: &Path, b: &[u8]) -> std::io::Result<()> {
    std::fs::write(tmp, b)?;
    std::fs::rename(tmp, p)
}

#[allow(clippy::too_many_arguments)]
fn run(tap: Option<Arc<ActivityTap>>, mut st: Option<Store>, script: Option<Box<dyn Fn(u32) -> u8 + Send>>, ws: Arc<Shared>, c: HeatCfg) {
    let qos = set_background_qos();
    let t_start = now_s();
    let cpu0 = thread_cpu_s();
    let hd = heat_dir(&c.dir);
    let _ = std::fs::create_dir_all(&hd);
    let tag = st.as_ref().map_or_else(|| "none".to_string(), |s| s.dir.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default());
    let (js, tmp) = (hd.join(format!("{}.json", tag)), hd.join(format!("{}.json.tmp", tag)));
    let _ = ws.heat_json.set(js.clone());
    let counters = |tp: &ActivityTap| { let (_, p, _) = tp.counters(); (p, tp.mtp_counters().1) };
    let mut last_c = tap.as_deref().map(counters);
    let mut last_jobs = ws.jobs.load(Relaxed);
    let (mut last_act, mut last_sum, mut dirty_sum) = (0.0f64, f64::NEG_INFINITY, true);
    let (mut samples, mut active_n, mut audits) = (0u64, 0u64, 0u64);
    let mut out = [0u8; JSON_MAX];
    // read mask: the profile mask is rebuilt on this thread's read-only store after each summary (mask != off; one box, setup)
    let mut scratch = match (&st, ws.mask_enabled()) { (Some(s), true) => Some(crate::watcher::predict::ReadMask::boxed(s)), _ => None };
    loop {
        // a handed-over job wakes the thread at once; else one sample period
        let t_wait = std::time::Instant::now();
        while let Some(m) = ws.next_job(std::time::Duration::from_millis(c.sample_ms).saturating_sub(t_wait.elapsed())) {
            ws.process(m);
            ws.library_flush(); // answer library: encode + write this job's record (gate already open)
            if t_wait.elapsed().as_millis() as u64 >= c.sample_ms { break; }
        }
        if SHUTDOWN.load(Relaxed) {
            ws.flush();
            eprintln!("heat: shutdown: open batch folded");
            std::process::exit(0);
        }
        let t = now_s();
        samples += 1;
        // in place: two GPU counters (no copy) + the job counter
        let cn = tap.as_deref().map(counters);
        let jobs = ws.jobs.load(Relaxed);
        let active = cn != last_c || jobs != last_jobs;
        (last_c, last_jobs) = (cn, jobs);
        if active { active_n += 1; last_act = t; dirty_sum = true; }
        let last_job = ws.last_job_ms.load(Relaxed) as f64 / 1e3;
        let folding = ws.prog.get().is_some_and(|p| p.phase.load(Relaxed) == 1);
        let idle = !active && !folding && t - last_act >= IDLE_S && (tap.is_some() || t - last_job >= 1.0);
        if dirty_sum && (t - last_sum >= SUMMARY_S || idle) {
            let c0 = thread_cpu_s();
            let mut w = Jw::new(&mut out);
            match &mut st {
                Some(s) => {
                    s.refresh();
                    s.prog.phase.store(2, Relaxed);
                    summary(s, c.keep, t, script.as_deref().map(|f| f as &dyn Fn(u32) -> u8), &mut w);
                    s.prog.phase.store(0, Relaxed);
                    let sat = tap.as_deref().map_or(0, |tp| tp.out_counts.saturated() + tp.embed_counts.saturated());
                    let _ = write!(w, ",\"count_saturated\":{}", sat);
                }
                None => { let _ = w.write_str("{\"state\":\"no recorder (--no-activity): heat off, audits only\""); }
            }
            let cpu_sum_s = thread_cpu_s() - c0;
            let cpu = thread_cpu_s() - cpu0;
            let _ = write!(w, ",\"jobs_summarized\":{},\"samples\":{},\"active_samples\":{},\"sample_ms\":{},\"qos\":\"{}\",\"cpu_s\":", jobs, samples, active_n,
                           c.sample_ms, if qos { "background" } else { "default (QoS call failed)" });
            w.f(cpu);
            let _ = w.write_str(",\"cpu_pct_one_core\":"); w.f(100.0 * cpu / (t - t_start).max(1e-3));
            let _ = w.write_str(",\"summary_cpu_ms\":"); w.f(cpu_sum_s * 1e3);
            let _ = write!(w, ",\"audits_done\":{},\"file\":", audits);
            w.s(&js.to_string_lossy());
            let _ = write!(w, ",\"ts\":{:.3},\"truncated\":{}}}", t, !w.ok);
            let n = w.n;
            if let Err(e) = write_file(&tmp, &js, &out[..n]) { eprintln!("heat: {}: {}", js.display(), e); }
            ws.summarized.store(jobs, Relaxed);
            dirty_sum = false;
            last_sum = t;
            if let (Some(s), Some(m)) = (st.as_mut(), scratch.as_deref_mut()) { ws.rebuild_mask(s, m, t); }
        }
        // a batch idle for BATCH_IDLE_S: fold it (then the next summary includes it)
        if !active && ws.flush_idle(BATCH_IDLE_S) { dirty_sum = true; }
        // audits after the summary: a client that sees an audit done also sees the summary of that job
        if idle && ws.audit_idle() { audits += 1; }
        // exploration (mask dry / on; every EXPLORE_EVERY_S at idle; the GPU read is the engine hook)
        if idle { ws.explore_idle(t); }
    }
}

/// The live part of the GET view: phase / progress / lag (from atomics).
/// A batch whose newest job is this old is folded to disk.
pub const BATCH_IDLE_S: f64 = 30.0;
/// Set by SIGINT / SIGTERM (`install_shutdown`): the sd-heat thread folds the open batch and exits the process.
pub static SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
extern "C" fn on_signal(_: libc::c_int) {
    // a second signal (or a stuck flush) exits at once
    if SHUTDOWN.swap(true, Relaxed) { unsafe { libc::_exit(130) } }
}
/// SIGINT / SIGTERM -> fold the open batch, then exit (gateway only; within ~one sample period + one fold).
pub fn install_shutdown() {
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
}

pub fn live_json(ws: &Shared) -> Value {
    let jobs = ws.jobs.load(Relaxed);
    let lag = jobs.saturating_sub(ws.summarized.load(Relaxed));
    let queued = ws.submitted.load(Relaxed).saturating_sub(jobs);
    let g = &streamdense_model::activity::GATE;
    let (gw, gms) = (g.waits.load(Relaxed), g.wait_us.load(Relaxed) as f64 / 1e3);
    let Some(p) = ws.prog.get() else { return serde_json::json!({"phase": "off", "lag_jobs": lag, "queued_jobs": queued, "line": "heat: off (no recorder)"}) };
    let ph = p.phase.load(Relaxed) as usize;
    let (job, k, n) = (p.job.load(Relaxed), p.chunk.load(Relaxed), p.chunks.load(Relaxed));
    let line = match ph {
        1 => format!("heat: folding job {} ({}/{} chunks); summary lag {} jobs", job, k + 1, n, lag),
        2 => format!("heat: summary running; lag {} jobs", lag),
        _ => format!("heat: {}; summary lag {} jobs", store::PHASES[ph.min(store::PHASES.len() - 1)], lag),
    };
    serde_json::json!({"phase": store::PHASES[ph.min(store::PHASES.len() - 1)], "job": job, "chunk": k, "chunks": n, "lag_jobs": lag, "queued_jobs": queued,
        "next_job_waits": gw, "next_job_wait_ms": gms, "jobs_folded": p.folded.load(Relaxed), "line": line})
}

/// `streamdense watcher heat [--dir DIR] [--layers]`: the last summary written by the gateway's heat thread.
pub fn print(dir: &Path, layers: bool) -> Result<(), String> {
    let hd = heat_dir(dir);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&hd).map_err(|e| format!("{}: {}", hd.display(), e))?.filter_map(|e| e.ok())
        .map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    if files.is_empty() { return Err(format!("{}: no heat summary yet (the gateway writes one 30 s after activity)", hd.display())); }
    for f in files {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {}", f.display(), e))?;
        println!("{} (tick {} ms, {} ticks, {} active, {} jobs folded; thread CPU {} s = {}% of one core; QoS {}; heap {} B, disk {} / {} B; last fold {} ms, {} B written; audits {})",
                 f.display(), v["sample_ms"], v["samples"], v["active_samples"], v["jobs_folded"], v["cpu_s"], v["cpu_pct_one_core"],
                 v["qos"].as_str().unwrap_or("?"), v["heap_bytes"], v["disk_bytes"], v["disk_cap_bytes"], v["fold"]["last_ms"], v["fold"]["last_bytes_written"], v["audits_done"]);
        println!("hot = {}", v["hot"].as_str().unwrap_or(""));
        println!("| scale | tokens | FFN hot (mean) | heads hot | coldest layers | MTP FFN / heads hot | out vocab rows hot | embed rows hot | out scripts |");
        println!("|---|---|---|---|---|---|---|---|---|");
        for s in SCALES {
            let x = &v[s];
            let cold: Vec<String> = x["coldest_layers"].as_array().into_iter().flatten().map(|p| format!("L{} {}", p[0], p[1])).collect();
            let sc: Vec<String> = x["vocab"]["scripts"].as_object().into_iter().flatten().map(|(k, f)| format!("{} {}", k, f)).collect();
            println!("| {} | {} | {} | {} | {} | {} / {} | {} ({}) | {} ({}) | {} |", s, x["tokens"], x["hot_frac_mean"], x["head_hot_frac_mean"], cold.join(", "),
                     x["mtp"]["ffn_hot_frac"], x["mtp"]["head_hot_frac"], x["vocab"]["rows_hot"], x["vocab"]["rows_hot_frac"],
                     x["embed"]["rows_hot"], x["embed"]["rows_hot_frac"], sc.join(", "));
        }
        if layers {
            println!("| layer | now | session | all |");
            println!("|---|---|---|---|");
            let n = v["all"]["per_layer_hot"].as_array().map_or(0, |a| a.len());
            for l in 0..n { println!("| {} | {} | {} | {} |", l, v["now"]["per_layer_hot"][l], v["session"]["per_layer_hot"][l], v["all"]["per_layer_hot"][l]); }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{JobSrc, CAP_BYTES};
    use crate::watcher::{label_names, Dims, MIXED};
    use serde_json::json;
    use streamdense_model::activity::ActivityStats;
    use streamdense_model::LayerKind;

    fn tmpdir(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_heat_{}_{}", n, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn dims(layers: usize, inter: usize, vocab: usize) -> Dims {
        let kinds = (0..layers).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let t = ActivityStats { enabled: true, layers, hidden: 128, inter, kinds, dn_kh: 1, dn_dk: 128, dn_vh: 2, dn_dv: 128, heads: 3, kv_heads: 1,
                                head_dim: 128, conv_k: 4, group: 64, bits: 4, ..Default::default() };
        let mut d = Dims::from_stats(&t, 64);
        d.vocab = vocab;
        d
    }
    fn open(dir: &Path, d: &Dims) -> Store { Store::open(dir, d.clone(), label_names(), MIXED, false).unwrap() }
    /// a job as one unified vector (test only)
    struct V { v: Vec<f32>, rows: (u64, u64), panic_at: usize }
    impl JobSrc for V {
        fn rows(&self) -> (u64, u64) { self.rows }
        fn read(&self, off: usize, out: &mut [f32]) {
            assert!(off < self.panic_at, "simulated crash in the middle of a fold");
            for (j, o) in out.iter_mut().enumerate() { *o = self.v.get(off + j).copied().unwrap_or(0.0); }
        }
    }
    fn summary_json(st: &mut Store, t: f64, script: Option<&dyn Fn(u32) -> u8>) -> Value {
        let mut b = [0u8; JSON_MAX];
        let mut w = Jw::new(&mut b);
        summary(st, 0.995, t, script, &mut w);
        let _ = w.write_str("}");
        assert!(w.ok);
        serde_json::from_slice(w.bytes()).unwrap()
    }

    /// The old in-RAM fold test on the disk store: same job, same summary numbers on every scale.
    #[test]
    fn fold_summary_on_disk() {
        let d = dims(4, 8, 16);
        let dir = tmpdir("fold");
        let mut st = open(&dir, &d);
        assert_eq!((st.n_ffn(), st.n_hr(), st.n_mtp(), st.n()), (32, 13, 11, 88));
        // layer 0: neuron 0 holds all FFN mass; other layers spread evenly; down norms 2 (contributions = |a| x 2)
        let mut v = vec![0f32; st.n()];
        v[0] = 200.0;
        for x in v[8..32].iter_mut() { *x = 2.0; }
        for (i, x) in [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0].iter().enumerate() { v[32 + i] = *x; }
        v[st.o_mtp()] = 1.0;
        v[st.o_mtp() + 8] = 1.0;
        (v[st.o_out() + 3], v[st.o_out() + 4]) = (3.0, 1.0);
        (v[st.o_emb() + 1], v[st.o_emb() + 2], v[st.o_emb() + 3]) = (1.0, 1.0, 2.0);
        let job = V { v, rows: (5, 2), panic_at: usize::MAX };
        st.fold_job(0, &job, 1, 0.0).unwrap();
        let h = st.hdr(0);
        assert_eq!((h.tokens, h.turns, h.mtp_rows, h.out_tok, h.emb_rows), (5, 1, 2, 4, 4));
        let script = |id: u32| if id == 4 { 3u8 } else { 0 };
        let v = summary_json(&mut st, 0.0, Some(&script));
        for k in SCALES {
            assert_eq!(v[k]["per_layer_hot"][0], json!(0.125), "{}", k);
            assert_eq!(v[k]["per_layer_hot"][1], json!(1.0), "{}", k);
            assert_eq!(v[k]["coldest_layers"][0], json!([0, 0.125]), "{}", k);
            assert_eq!(v[k]["vocab"]["rows_hot"], json!(2), "{}", k);
            assert_eq!(v[k]["vocab"]["scripts"], json!({"ascii": 0.75, "han": 0.25}), "{}", k);
            assert_eq!(v[k]["embed"]["rows_seen"], json!(3), "{}", k);
            assert_eq!(v[k]["mtp"]["ffn_hot_frac"], json!(0.125), "{}", k);
            assert_eq!(v[k]["mtp"]["tokens"], json!(2.0), "{}", k);
            assert_eq!(v["vocab"][k]["tokens"], json!(4.0), "{}", k);
            assert_eq!(v["embed"][k]["tokens"], json!(4.0), "{}", k);
            assert_eq!(v[k]["tokens"], json!(5.0), "{}", k);
        }
        assert_eq!(v["all"]["head_hot_frac_mean"], v["now"]["head_hot_frac_mean"]);
        assert_eq!(v["all_from"]["labels"], json!(1));
        assert_eq!(v["heap_bytes"], json!(0));
        // one tau later: now x e^-1, session barely, all unchanged
        let v = summary_json(&mut st, TAUS_S[0], None);
        assert_eq!(v["now"]["tokens"].as_f64().unwrap(), (5.0 * (-1.0f64).exp()).round()); // token counts print rounded
        assert_eq!(v["session"]["tokens"].as_f64().unwrap(), (5.0 * (-TAUS_S[0] / TAUS_S[1]).exp()).round());
        assert_eq!(v["all"]["tokens"], json!(5.0));
        assert_eq!(v["now"]["per_layer_hot"][0], json!(0.125)); // decay keeps the shape
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Disk sets = the exact (f64) result of the old per-job semantics on a fixed sequence: all-time sums, now / session
    /// with the decay of every job to the read clock (the in-RAM f16 maps approximated exactly this).
    #[test]
    fn disk_equals_exact_reference() {
        let d = dims(4, 1024, 300);
        let dir = tmpdir("exact");
        let mut st = open(&dir, &d);
        let n = st.n();
        assert_eq!(st.nch(), 3);
        let times = [0.0, 40.0, 41.0, 200.0, 900.0];
        let mut x = 12345u32;
        let jobs: Vec<Vec<f32>> = times.iter().map(|_| (0..n).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; if x % 5 == 0 { 0.0 } else { (x % 1000) as f32 / 7.0 } }).collect()).collect();
        for (i, (t, v)) in times.iter().zip(&jobs).enumerate() {
            st.fold_job(i % 2, &V { v: v.clone(), rows: (10, 1), panic_at: usize::MAX }, i as u64, *t).unwrap();
        }
        let tr = 1000.0;
        st.t = tr;
        let mut rec = [0u8; CH * 4 + 16];
        let mut got = vec![0f32; n];
        for (sel, tau) in [(Sel::Set(NOW), Some(TAUS_S[0])), (Sel::Set(SESS), Some(TAUS_S[1])), (Sel::Labels, None)] {
            st.read_sel(sel, 0, &mut got, &mut rec);
            let mut worst = 0f64;
            for i in 0..n {
                let want: f64 = times.iter().zip(&jobs).map(|(t, v)| v[i] as f64 * tau.map_or(1.0, |tau| (-(tr - t) / tau).exp())).sum();
                if want > 1e-3 { worst = worst.max((got[i] as f64 / want - 1.0).abs()); }
            }
            eprintln!("{:?}: worst relative error {:.2e}", sel, worst);
            assert!(worst < 2e-6, "{:?}: {}", sel, worst);
        }
        // per label all-time: label 0 = jobs 0, 2, 4
        st.read_sel(Sel::Set(0), 0, &mut got, &mut rec);
        assert!((0..n).all(|i| got[i] == jobs[0][i] + jobs[2][i] + jobs[4][i]));
        assert_eq!((st.hdr(0).turns, st.hdr(1).turns, st.hdr(NOW).turns), (3, 2, 5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A crash in the middle of a fold leaves every chunk consistent; a torn chunk is restored from the journal, else
    /// zeroed and counted.
    #[test]
    fn crash_mid_fold_and_torn_chunk() {
        let d = dims(4, 1024, 300);
        let dir = tmpdir("crash");
        let mut st = open(&dir, &d);
        let n = st.n();
        let one = V { v: vec![1.0; n], rows: (7, 0), panic_at: usize::MAX };
        st.fold_job(0, &one, 1, 0.0).unwrap();
        // crash while reading the job's 2nd chunk: chunk 0 has job 2, chunks 1, 2 do not; header not advanced
        let two = V { v: vec![2.0; n], rows: (9, 0), panic_at: CH };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.fold_job(0, &two, 2, 1.0)));
        assert!(r.is_err());
        drop(st);
        let mut st = open(&dir, &d);
        let mut rec = [0u8; CH * 4 + 16];
        let mut v = vec![0f32; n];
        st.read_sel(Sel::Set(0), 0, &mut v, &mut rec);
        assert!(v[..CH].iter().all(|x| *x == 3.0) && v[CH..].iter().all(|x| *x == 1.0));
        assert_eq!(st.hdr(0).tokens, 7);
        assert_eq!((st.prog.repaired.load(Relaxed), st.prog.lost.load(Relaxed)), (0, 0));
        // torn chunk 0 (half new bytes): the journal holds its pre-image (job 1 state) -> restored
        let f = std::fs::OpenOptions::new().read(true).write(true).open(st.path()).unwrap();
        use std::os::unix::fs::FileExt;
        let base = st.slot_base(st.slot_of(0).unwrap());
        f.write_all_at(&[0x55u8; 4000], base + (crate::store::HDR + 4000) as u64).unwrap();
        st.read_sel(Sel::Set(0), 0, &mut v, &mut rec);
        assert!(v[..CH].iter().all(|x| *x == 1.0), "restored from the journal: {}", v[0]);
        assert_eq!(st.prog.repaired.load(Relaxed), 1);
        // torn chunk 2 (no journal for it): zeroed + counted, never silently wrong
        f.write_all_at(&[0x55u8; 100], base + (crate::store::HDR + 2 * (CH * 4 + 16) + 8) as u64).unwrap();
        st.read_sel(Sel::Set(0), 0, &mut v, &mut rec);
        assert!(v[2 * CH..].iter().all(|x| *x == 0.0) && v[CH] == 1.0);
        assert_eq!(st.prog.lost.load(Relaxed), 1);
        // the store keeps working
        st.fold_job(0, &one, 3, 2.0).unwrap();
        st.read_sel(Sel::Set(0), 0, &mut v, &mut rec);
        assert_eq!((v[0], v[CH], v[2 * CH]), (2.0, 2.0, 1.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Steady state: fold, summary, plan and score allocate nothing (counting allocator, per thread).
    #[test]
    fn zero_heap_steady_state() {
        let d = dims(8, 512, 1000);
        let dir = tmpdir("zero");
        let mut st = open(&dir, &d);
        let n = st.n();
        let job = V { v: (0..n).map(|i| (i % 13) as f32).collect(), rows: (3, 1), panic_at: usize::MAX };
        let script = |id: u32| (id % 10) as u8;
        // first calls: set creation (paths, files) = one-time setup
        st.fold_job(0, &job, 1, 0.0).unwrap();
        st.fold_job(4, &job, 2, 0.0).unwrap();
        let _ = summary_json(&mut st, 1.0, Some(&script));
        let mut out = [0u8; JSON_MAX];
        let a0 = streamdense_core::alloc::thread_allocs();
        st.fold_job(0, &job, 3, 2.0).unwrap();
        let mut w = Jw::new(&mut out);
        summary(&mut st, 0.995, 3.0, Some(&script), &mut w);
        let p = st.plan(Sel::Labels, 0.95, 64);
        let p1 = st.plan(Sel::Set(0), 0.95, 1);
        let mut use_l = [false; NP];
        (use_l[0], use_l[4]) = (true, true);
        let cos = st.score(&job, &use_l);
        // similarity gate path: slot similarity, centroid cosine, pooled-only fold of a held job
        let sm = st.sim(0, &job).unwrap();
        let cc = st.cent_cos(0, 4);
        st.fold_pooled(&job, 4, 3.0).unwrap();
        // batched path: per-job centroid + one batch fold of 8 jobs
        st.update_cent(0, &job, 3).unwrap();
        st.fold_to(Some(0), &job, 8, 5, 4.0).unwrap();
        let a1 = streamdense_core::alloc::thread_allocs();
        assert_eq!(a1 - a0, 0, "heap allocations in fold + summary + plan + score + gate");
        assert!(sm.min() > 0.99 && cc > 0.99, "{:?} {}", sm, cc);
        // even 64-blocks each hold ~1/8 of a layer (> the 5 % budget): none cold; per neuron the coldest 5 % are
        assert!(w.ok && p.neurons == 8 * 512 && p.cold_neurons == 0 && p1.cold_neurons > 0 && cos[0] > 0.99);
        assert!(SUMMARY_STACK <= 64 << 10, "{}", SUMMARY_STACK);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The GET view's live part: progress of a running fold and the summary lag.
    #[test]
    fn progress_and_lag() {
        let dir = tmpdir("live");
        let w = crate::watcher::Watcher::open(dir.clone(), crate::watcher::WCfg::from_config(&Default::default()), Some(dims(4, 8, 16))).unwrap();
        let ws = Shared::new(w, || 0);
        let p = ws.prog.get().unwrap().clone();
        p.phase.store(1, Relaxed);
        (p.job.store(812, Relaxed), p.chunk.store(2, Relaxed), p.chunks.store(18, Relaxed));
        ws.jobs.store(5, Relaxed);
        ws.summarized.store(3, Relaxed);
        let v = live_json(&ws);
        assert_eq!(v["line"], json!("heat: folding job 812 (3/18 chunks); summary lag 2 jobs"));
        assert_eq!((v["lag_jobs"].clone(), v["phase"].clone()), (json!(2), json!("folding")));
        p.phase.store(0, Relaxed);
        assert!(live_json(&ws)["line"].as_str().unwrap().starts_with("heat: idle"));
        assert_eq!(ws.heat_view()["state"], json!("starting"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cost at 27B sizes (synthetic source, no model): fold time + bytes written per job, summary and plan time, disk.
    /// cargo test --release -p streamdense-cli --bin streamdense heat::tests::cost_27b -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost_27b() {
        struct P(usize);
        impl JobSrc for P {
            fn rows(&self) -> (u64, u64) { (300, 100) }
            fn read(&self, off: usize, out: &mut [f32]) { for (j, o) in out.iter_mut().enumerate() { let g = off + j; *o = if g >= self.0 && g % 97 != 0 { 0.0 } else { ((g * 7) % 13) as f32 }; } }
        }
        let kinds: Vec<LayerKind> = (0..64).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let t = ActivityStats { enabled: true, layers: 64, hidden: 5120, inter: 17408, kinds, dn_kh: 16, dn_dk: 128, dn_vh: 48, dn_dv: 128, heads: 24,
                                kv_heads: 4, head_dim: 256, conv_k: 4, group: 64, bits: 4, ..Default::default() };
        let mut d = Dims::from_stats(&t, 64);
        d.vocab = 248_320;
        let dir = tmpdir("cost");
        let mut st = open(&dir, &d);
        let src = P(st.o_out());
        set_background_qos();
        let mut wb = 0;
        for lab in 0..NP { if lab != MIXED { st.fold_job(lab, &src, lab as u64, 0.0).unwrap(); } }
        let t0 = std::time::Instant::now();
        for i in 0..5 { wb = st.fold_job(0, &src, 10 + i, 1.0 + i as f64).unwrap(); }
        let fold_ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
        let t1 = std::time::Instant::now();
        let mut out = [0u8; JSON_MAX];
        let mut w = Jw::new(&mut out);
        summary(&mut st, 0.995, 10.0, None, &mut w);
        let sum_ms = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = std::time::Instant::now();
        let _ = st.plan(Sel::Labels, 0.995, 64);
        let plan_ms = t2.elapsed().as_secs_f64() * 1e3;
        let t3 = std::time::Instant::now();
        let _ = st.sim(0, &src);
        let sim_ms = t3.elapsed().as_secs_f64() * 1e3;
        eprintln!("27B store.bin {} B (cap {}): table {} B, {} label slots + 1 held slot x {} B, now / session {} B each, down {} B, journal {} B",
                  st.disk_bytes(), CAP_BYTES, crate::store::HDR, st.nslots(), st.lsz(), st.nsz(), st.dsz(), st.jsz());
        eprintln!("27B: fold {:.1} ms/job (background QoS), {} B written/job; summary {:.0} ms; plan (union) {:.0} ms; similarity gate {:.0} ms; json ok {}",
                  fold_ms, wb, sum_ms, plan_ms, sim_ms, w.ok);
        assert_eq!(st.disk_bytes(), st.total_bytes());
        assert!(st.disk_bytes() <= CAP_BYTES && st.nslots() == NP - 1, "6 label slots + held + now + session + down + journal fit the cap");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Job-end delay + bytes per job at 27B sizes (synthetic recorder, no model). Before: the hook scored, gated and
    /// folded every job inline. After: the hook only closes the gate and hands the job over; sd-heat scores, gates and
    /// adds it to the batch (here a host Vec stands in for the model-memory batch region), one disk fold per
    /// `fold_every` jobs.
    /// cargo test --release -p streamdense-cli --bin streamdense heat::tests::cost_27b_hook -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost_27b_hook() {
        use crate::store::Batcher;
        use crate::watcher::{Feat, JobMsg, Shared, WCfg, Watcher};
        use std::sync::Mutex;
        struct P(usize, f32);
        impl JobSrc for P {
            fn rows(&self) -> (u64, u64) { (300, 100) }
            fn read(&self, off: usize, out: &mut [f32]) { for (j, o) in out.iter_mut().enumerate() { let g = off + j; *o = if g >= self.0 && g % 97 != 0 { 0.0 } else { ((g * 7) % 13) as f32 * self.1 }; } }
        }
        struct VB(Mutex<Vec<f32>>);
        impl Batcher for VB {
            fn add(&self, job: &dyn JobSrc) { let mut v = self.0.lock().unwrap(); let mut b = [0f32; CH]; let n = v.len(); let mut o = 0; while o < n { let l = CH.min(n - o); job.read(o, &mut b[..l]); for j in 0..l { v[o + j] += b[j]; } o += l; } }
            fn clear(&self) { self.0.lock().unwrap().fill(0.0); }
            fn read(&self, off: usize, out: &mut [f32]) { let v = self.0.lock().unwrap(); for (j, o) in out.iter_mut().enumerate() { *o = v.get(off + j).copied().unwrap_or(0.0); } }
        }
        let kinds: Vec<LayerKind> = (0..64).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let t = ActivityStats { enabled: true, layers: 64, hidden: 5120, inter: 17408, kinds, dn_kh: 16, dn_dk: 128, dn_vh: 48, dn_dv: 128, heads: 24,
                                kv_heads: 4, head_dim: 256, conv_k: 4, group: 64, bits: 4, ..Default::default() };
        let mut d = Dims::from_stats(&t, 64);
        d.vocab = 248_320;
        set_background_qos();
        let f = Feat::of("fn main() { let x = vec![1, 2, 3]; println!(\"{:?}\", x); } // rust code");
        let mut res = vec![];
        for batched in [false, true] {
            let dir = tmpdir(if batched { "hook_b" } else { "hook_a" });
            let w = Watcher::open(dir.clone(), WCfg::from_config(&Default::default()), Some(d.clone())).unwrap();
            let ws = Shared::new(w, || 0);
            let src = { let w = ws.w.lock().unwrap(); P(w.store.as_ref().unwrap().o_out(), 1.0) };
            if batched { let n = ws.w.lock().unwrap().store.as_ref().unwrap().n(); ws.w.lock().unwrap().batcher = Some(Arc::new(VB(Mutex::new(vec![0.0; n])))); }
            // warm-up (slot claim, centroid) then 16 jobs
            ws.on_job(&f, 300, Some(&src), "k", "m0");
            ws.flush();
            let p = ws.prog.get().unwrap().clone();
            let w0 = p.bytes_written.load(Relaxed);
            let (mut hook_ms, mut heat_ms) = (0f64, 0f64);
            for i in 0..16 {
                if batched {
                    let t0 = std::time::Instant::now();
                    ws.submit(JobMsg { f, tokens: 300, key: "k".into(), msg_id: format!("m{}", i), t_end: crate::watcher::now_s(), prompt: None, lib: None });
                    hook_ms += t0.elapsed().as_secs_f64() * 1e3;
                    let m = ws.next_job(std::time::Duration::from_millis(1)).unwrap();
                    let t1 = std::time::Instant::now();
                    // sd-heat's work on this job (the gateway's `process`, with the synthetic recorder as the source)
                    { let mut w = ws.w.lock().unwrap(); let a = w.domain; w.observe_src(&m.f, m.tokens, Some(&src), &m.key); w.remember_job(&m.msg_id, a); }
                    streamdense_model::activity::GATE.open();
                    heat_ms += t1.elapsed().as_secs_f64() * 1e3;
                } else {
                    let t0 = std::time::Instant::now();
                    ws.on_job(&f, 300, Some(&src), "k", &format!("m{}", i));
                    hook_ms += t0.elapsed().as_secs_f64() * 1e3;
                }
            }
            ws.flush();
            let per_job = (p.bytes_written.load(Relaxed) - w0) / 16;
            let wb = ws.w.lock().unwrap().store.as_ref().unwrap().disk_bytes();
            eprintln!("27B {}: job-end delay (hook) {:.3} ms/job; sd-heat {:.1} ms/job; {} B written/job = {:.2} GB/day at 300 replies; store {} B",
                      if batched { "after (hand-off + batch, fold_every 8)" } else { "before (inline fold)" }, hook_ms / 16.0, heat_ms / 16.0, per_job,
                      per_job as f64 * 300.0 / 1e9, wb);
            res.push((hook_ms / 16.0, per_job));
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert!(res[1].0 < 1.0 && res[1].1 * 4 < res[0].1, "{:?}", res);
    }
}
