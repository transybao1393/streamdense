//! HTTP front: routes, per-request handler threads, Anthropic SSE / JSON assembly, request log.
use crate::anthropic::{self as an, ChatReq};
use crate::chatmap::{coerce_args, parse_tool_calls, render, reply_text, Visible};
use crate::http::{self, Request};
use crate::worker::{Ev, Job, Outcome};
use serde_json::{json, Value};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use streamdense_tokenizer::Tokenizer;

/// While a request waits (queue, long prefill), an SSE `ping` goes out this often; a failed write cancels it.
pub const PING_EVERY: Duration = Duration::from_secs(10);
/// A queued warm-up's handler checks this often whether its client closed the connection (then it is dropped).
pub const WARM_POLL: Duration = Duration::from_millis(100);

/// The client closed its side of the connection (peek reads EOF or fails; a pending next request is not EOF).
fn client_gone(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() { return false; }
    let mut b = [0u8; 1];
    let r = s.peek(&mut b);
    let _ = s.set_nonblocking(false);
    match r {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => !matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted),
    }
}

pub struct Shared {
    pub tk: Arc<Tokenizer>,
    pub model_name: String,
    pub max_ctx: usize,
    /// Thinking honored (`--think`).
    pub think: bool,
    pub jobs: Mutex<Sender<Job>>,
    pub log: Logger,
    pub info: Value,
    /// Extra routes under /v1/streamdense/ + a request-log field (the watcher).
    pub ext: Option<Arc<dyn crate::Extension>>,
    /// Activity recorder (GET /v1/streamdense/activity, served here before `ext` routes)
    pub activity: Option<Arc<streamdense_model::activity::ActivityTap>>,
    /// RAM mode of the backend (`Backend::mode_info`; Some in 12 GB mode): /health + the request log
    pub mode: Option<Value>,
    /// process peak footprint tracker (with `mode`)
    pub peak: Option<Arc<std::sync::atomic::AtomicU64>>,
    /// snapshot tiers (snap_tier.rs): pending-request counter (stops idle work) + status for /health
    pub tiers: Option<Arc<crate::snap_tier::TierShared>>,
    /// the running warm-up (worker.rs `WarmGate`): a real request for another conversation cancels it
    pub warm: Arc<crate::worker::WarmGate>,
    /// idle-sleep assertion (awake.rs): held while a request is queued/running and `LINGER` after the last one
    pub awake: Arc<crate::awake::Awake>,
}

impl Shared {
    fn ext_field(&self) -> Option<(&'static str, Value)> { self.ext.as_ref().and_then(|e| e.log_field()) }
}

/// One JSON line per request to a file (`--log FILE`, appended) or stderr.
pub struct Logger { file: Option<Mutex<std::fs::File>> }
impl Logger {
    pub fn new(path: Option<&str>) -> Result<Logger, String> {
        Ok(Logger { file: match path {
            Some(p) => Some(Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).map_err(|e| format!("--log {}: {}", p, e))?)),
            None => None,
        } })
    }
    pub fn line(&self, v: &Value) {
        match &self.file {
            Some(f) => { if let Ok(mut f) = f.lock() { let _ = writeln!(f, "{}", v); let _ = f.flush(); } }
            None => eprintln!("{}", v),
        }
    }
}

fn now_s() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0) }

/// Accept loop (blocks): one thread per connection.
pub fn accept_loop(listener: TcpListener, sh: Arc<Shared>) {
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        let sh = sh.clone();
        std::thread::spawn(move || { let _ = connection(s, &sh); });
    }
}

fn json_resp(s: &mut TcpStream, code: u16, v: &Value, ka: bool, head: bool) -> std::io::Result<()> {
    http::write_response(s, code, "application/json", v.to_string().as_bytes(), ka, head, &[("request-id", an::new_id("req_"))])
}

fn connection(mut s: TcpStream, sh: &Shared) -> std::io::Result<()> {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(600)));
    let mut r = http::reader(&s)?;
    loop {
        let req = match http::read_request(&mut r) {
            Ok(Some(q)) => q,
            Ok(None) => return Ok(()),
            Err(e) => { let _ = json_resp(&mut s, 400, &an::error_json("invalid_request_error", &e), false, false); return Ok(()); }
        };
        let ka = req.keep_alive();
        let head = req.method == "HEAD";
        let p = req.path.trim_end_matches('/');
        match (req.method.as_str(), p) {
            ("GET" | "HEAD", "" | "/health" | "/healthz") => {
                let mut v = json!({"status": "ok", "model": sh.model_name, "info": sh.info});
                v["info"]["warm"] = json!(true); // warm-ups supported (clients check this before sending warm-ups)
                if let Some(m) = &sh.mode { v["ram_mode"] = ram_mode(m, sh.peak.as_deref()); }
                if let Some(t) = &sh.tiers { v["snapshots"] = t.health(); }
                if let Some((k, f)) = sh.ext_field() { v[k] = f; }
                json_resp(&mut s, 200, &v, ka, head)?
            }
            ("GET" | "HEAD", "/v1/models") => json_resp(&mut s, 200, &json!({"data": [model_obj(sh)], "has_more": false,
                "first_id": sh.model_name, "last_id": sh.model_name}), ka, head)?,
            ("GET" | "HEAD", m) if m.starts_with("/v1/models/") => json_resp(&mut s, 200, &model_obj(sh), ka, head)?,
            ("GET" | "HEAD", "/v1/streamdense/activity") => {
                let v = streamdense_model::activity::handle_request(sh.activity.as_deref(), &activity_query(&req.query));
                json_resp(&mut s, 200, &v, ka, head)?;
            }
            ("POST", "/v1/messages/count_tokens") => {
                let (code, v) = match parse_body(&req) {
                    Ok(cr) => match render(&cr.msgs, &cr.tools, sh.think && cr.thinking_requested, true) {
                        Ok(t) => (200, json!({"input_tokens": sh.tk.encode(&t).len()})),
                        Err(e) => (400, an::error_json("invalid_request_error", &e)),
                    },
                    Err(e) => (400, an::error_json("invalid_request_error", &e)),
                };
                json_resp(&mut s, code, &v, ka, false)?;
            }
            ("POST", "/v1/messages") => {
                let keep = messages(&mut s, &req, sh, ka)?;
                if !keep { return Ok(()); }
            }
            (m, x) if x.starts_with("/v1/streamdense/") && sh.ext.is_some() => {
                let (code, v) = sh.ext.as_ref().and_then(|e| e.route(m, x, &req.body))
                    .unwrap_or_else(|| (404, an::error_json("not_found_error", &format!("no route {} {}", req.method, req.path))));
                json_resp(&mut s, code, &v, ka, head)?
            }
            (_, "/v1/messages" | "/v1/messages/count_tokens" | "/v1/models" | "" | "/health") =>
                json_resp(&mut s, 405, &an::error_json("invalid_request_error", &format!("{} not allowed on {}", req.method, req.path)), ka, head)?,
            _ => json_resp(&mut s, 404, &an::error_json("not_found_error", &format!("no route {} {}", req.method, req.path)), ka, head)?,
        }
        if !ka { return Ok(()); }
    }
}

fn model_obj(sh: &Shared) -> Value {
    json!({"id": sh.model_name, "type": "model", "display_name": sh.model_name, "created_at": "2026-10-29T00:00:00Z"})
}

fn parse_body(req: &Request) -> Result<ChatReq, String> {
    let v: Value = serde_json::from_slice(&req.body).map_err(|e| format!("request body is not JSON: {}", e))?;
    an::convert(&v)
}

/// The finished reply as Anthropic content: (visible text, [(tool name, coerced input)], stop_reason). A stop
/// sequence / banned-id stop ("stop_sequence" / "format", worker.rs) wins over a tool call cut by it.
fn finish_reply(o: &Outcome, req: &ChatReq) -> (String, Vec<(String, Value)>, &'static str) {
    let (visible, calls) = parse_tool_calls(&reply_text(&o.raw, o.think));
    let calls: Vec<(String, Value)> = calls.into_iter().map(|c| { let a = coerce_args(&c.args, req.tool_schema(&c.name)); (c.name, a) }).collect();
    let stop = if o.stop_reason == "stop_sequence" { "stop_sequence" } else if o.stop_reason == "format" { "format" }
        else if !calls.is_empty() { "tool_use" } else if o.stop_reason == "length" || (o.stop_reason == "ctx_full" && o.n_new > 0) { "max_tokens" } else { "end_turn" };
    (visible, calls, stop)
}

/// Which prefix this request started from: "kv" (the live engine state: prefix / followup), "own" / "shared" (a
/// snapshot of the request's conversation / of the project's system prompt + tools), "cold" (prefilled from zero).
fn prefix_used(o: &Outcome) -> &'static str {
    match o.mode.as_str() {
        "prefix" | "followup" => "kv",
        "restore" => o.snaps.iter().find(|e| e.kind == "restore" && e.error.is_none()).map_or("cold", |e| if e.src.is_empty() { "own" } else { e.src }),
        "full" => "cold",
        _ => "",
    }
}

fn log_line(sh: &Shared, req: &Request, cr: &ChatReq, o: &Outcome, stop: &str, calls: &[(String, Value)], total: f64) {
    let mut v = json!({"ts": now_s(), "path": req.path, "stream": cr.stream, "n_msgs": cr.msgs.len(), "n_tools": cr.tools.len(),
        "n_prompt": o.n_prompt, "mode": o.mode, "n_feed": o.n_feed, "n_reused": o.n_reused, "n_new": o.n_new,
        "queue_s": o.queue_s, "tokenize_s": o.tokenize_s, "ttft_s": o.ttft_s, "prefill_s": o.prefill_s, "decode_tok_s": o.decode_tok_s,
        "total_s": total, "stop_reason": stop, "engine_stop": o.stop_reason, "tools": calls.iter().map(|c| c.0.clone()).collect::<Vec<_>>(),
        "think": o.think, "mtp": o.mtp, "spec": o.spec.map(|s| json!({"passes": s.0, "accepted": s.1, "rejects": s.2, "acceptance": s.3})),
        "ctx_draft": o.ctx_draft,
        "snap": o.snaps.iter().map(|e| json!({"kind": e.kind, "n": e.n, "ms": (e.ms * 10.0).round() / 10.0, "mb": (e.bytes as f64 / 1e5).round() / 10.0,
            "tier": e.tier, "src": e.src, "error": e.error})).collect::<Vec<_>>(),
        "prefix": prefix_used(o),
        "tiers": o.tiers,
        "warm": cr.warm,
        "error": o.error});
    if let Some(d) = o.drop_reason { v["drop"] = json!(d); }
    if let Some(m) = &sh.mode {
        let mut r = ram_mode(m, sh.peak.as_deref());
        if let Some(s) = &o.stream {
            r["stream"] = json!({"pieces": s.pieces, "tokens": s.tokens, "ssd_gb": s.ssd_bytes as f64 / 1e9, "ssd_busy_s": s.ssd_busy_s, "wall_s": s.wall_s,
                "gpu_s": s.gpu_s, "load_wait_s": s.load_wait_s, "policy_s": s.policy_s, "fill_s": s.fill_s, "changed": s.changed,
                "host_reload_mb": s.host_reload_bytes as f64 / 1e6});
        }
        v["ram_mode"] = r;
    }
    if let Some((k, f)) = sh.ext_field() { v[k] = f; }
    sh.log.line(&v);
}

/// The RAM mode fields for /health and the log: the backend's static info (mode, K, planned / resident GB, ..) + the
/// live footprint.
fn ram_mode(m: &Value, peak: Option<&std::sync::atomic::AtomicU64>) -> Value {
    let mut r = m.clone();
    let f = streamdense_core::footprint::self_footprint();
    r["footprint_gb"] = json!(f as f64 / 1e9);
    if let Some(p) = peak { r["peak_footprint_gb"] = json!(p.load(std::sync::atomic::Ordering::Relaxed).max(f) as f64 / 1e9); }
    r
}

/// Write SSE bytes (the head first if not sent yet). false = the client is gone.
fn wr(s: &mut TcpStream, head_sent: &mut bool, data: &str) -> bool {
    let r = (|| -> std::io::Result<()> {
        if !*head_sent { http::write_sse_head(s, &[("request-id", an::new_id("req_"))])?; *head_sent = true; }
        s.write_all(data.as_bytes())?;
        s.flush()
    })();
    r.is_ok()
}

/// POST /v1/messages. Returns whether the connection may stay open.
fn messages(s: &mut TcpStream, req: &Request, sh: &Shared, ka: bool) -> std::io::Result<bool> {
    let t0 = Instant::now();
    let _awake = sh.awake.begin(); // no idle sleep until this handler returns + awake::LINGER
    let cr = match parse_body(req) {
        Ok(mut c) => {
            if req.header("x-streamdense-warm").is_some_and(|v| matches!(v.trim(), "1" | "true")) { c.warm = true; }
            c.project = req.header("x-streamdense-project").and_then(an::parse_project);
            c.lane_plan = req.header("x-streamdense-lane").is_some_and(|v| v.trim().eq_ignore_ascii_case("plan"));
            c
        }
        Err(e) => { sh.log.line(&json!({"ts": now_s(), "path": req.path, "error": e})); json_resp(s, 400, &an::error_json("invalid_request_error", &e), ka, false)?; return Ok(ka); }
    };
    let (tx, rx) = channel::<Ev>();
    let cancel = Arc::new(AtomicBool::new(false));
    let id = an::new_id("msg_");
    let conv = cr.conv_key();
    let job = Job { req: cr.clone(), warm: cr.warm, conv, id: id.clone(), events: tx, cancel: cancel.clone(), t_recv: t0 };
    if let Some(t) = &sh.tiers { t.arrived(); } // idle snapshot work stops before the job is even queued
    if sh.jobs.lock().map(|j| j.send(job).is_err()).unwrap_or(true) {
        json_resp(s, 529, &an::error_json("overloaded_error", "the engine worker is gone"), false, false)?;
        return Ok(false);
    }
    // after the send (see WarmGate): a warm-up of another conversation in progress stops at its next piece
    if !cr.warm { sh.warm.real_arrived(conv, cr.lane_key()); }
    let model = cr.model.clone();
    if cr.warm { return warm_reply(s, req, sh, &cr, &id, rx, &cancel, t0, ka); }
    if !cr.stream {
        let mut out: Option<Outcome> = None;
        for ev in rx.iter() { if let Ev::Done(o) = ev { out = Some(o); break; } }
        let o = out.unwrap_or_else(|| Outcome { stop_reason: "error".into(), error: Some("worker dropped the request".into()), ..Default::default() });
        if let Some(e) = &o.error {
            log_line(sh, req, &cr, &o, "error", &[], t0.elapsed().as_secs_f64());
            let code = if o.n_prompt == 0 && o.mode.is_empty() { 400 } else { 500 };
            json_resp(s, code, &an::error_json(if code == 400 { "invalid_request_error" } else { "api_error" }, e), ka, false)?;
            return Ok(ka);
        }
        let (mut visible, calls, stop) = finish_reply(&o, &cr);
        if o.stop_reason == "ctx_full" && o.n_new == 0 { visible = an::ctx_full_text(o.n_prompt, o.max_ctx); }
        let mut content = vec![];
        if !visible.is_empty() || calls.is_empty() {
            content.push(json!({"type": "text", "text": if visible.is_empty() { "(no output)".to_string() } else { visible }}));
        }
        for (n, a) in &calls { content.push(json!({"type": "tool_use", "id": an::new_id("toolu_"), "name": n, "input": a})); }
        let u = an::usage(o.n_prompt - o.n_reused.min(o.n_prompt), o.n_reused, o.n_new);
        log_line(sh, req, &cr, &o, stop, &calls, t0.elapsed().as_secs_f64());
        json_resp(s, 200, &an::message_json(&id, &model, content, stop, o.stop_sequence.as_deref(), u), ka, false)?;
        return Ok(ka);
    }

    // streaming: the SSE head goes out with the first event (or ping) so an early error can still be a plain 4xx
    let mut head_sent = false;
    let mut started = false;
    let mut text_open = false;
    let mut decoding = false;
    let mut hinted: Option<(usize, bool)> = None;
    let mut vis: Option<Visible> = None;
    let mut broken = false;
    let outcome = loop {
        let ev = match rx.recv_timeout(PING_EVERY) {
            Ok(ev) => ev,
            Err(RecvTimeoutError::Timeout) => {
                if !broken && !wr(s, &mut head_sent, &an::ping()) { broken = true; cancel.store(true, Ordering::SeqCst); }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break Outcome { stop_reason: "error".into(), error: Some("worker dropped the request".into()), ..Default::default() },
        };
        match ev {
            Ev::Start { n_prompt, n_reused, think, .. } => {
                vis = Some(Visible::new(think));
                started = true;
                if !broken && !wr(s, &mut head_sent, &an::message_start(&id, &model, an::usage(n_prompt - n_reused.min(n_prompt), n_reused, 0))) {
                    broken = true; cancel.store(true, Ordering::SeqCst);
                }
                // phase hints for the client's companion (a plain ping to any other client): reading the prompt now
                if !broken && !wr(s, &mut head_sent, &an::ping_phase("prefill")) { broken = true; cancel.store(true, Ordering::SeqCst); }
            }
            Ev::Text(piece) => {
                let Some(v) = vis.as_mut() else { continue };
                // the first generated token (visible or not, e.g. a tool call): the prefill is over
                if !decoding {
                    decoding = true;
                    if !broken && !wr(s, &mut head_sent, &an::ping_phase("decode")) { broken = true; cancel.store(true, Ordering::SeqCst); }
                }
                let t = v.push(&piece);
                // a tool call being written: its name, then its path once complete (the client shows WRITING main.py)
                if let Some((k, name, path)) = v.call_hint() {
                    let h = (k, path.is_some());
                    if hinted != Some(h) && !broken {
                        hinted = Some(h);
                        if !wr(s, &mut head_sent, &an::ping_tool(&name, path.as_deref().unwrap_or(""))) { broken = true; cancel.store(true, Ordering::SeqCst); }
                    }
                }
                if t.is_empty() || broken { continue; }
                let mut out = String::new();
                if !text_open { out += &an::text_block_start(0); text_open = true; }
                out += &an::text_delta(0, &t);
                if !wr(s, &mut head_sent, &out) { broken = true; cancel.store(true, Ordering::SeqCst); }
            }
            Ev::Done(o) => break o,
        }
    };
    let total = t0.elapsed().as_secs_f64();
    if let Some(e) = &outcome.error {
        log_line(sh, req, &cr, &outcome, "error", &[], total);
        if broken { return Ok(false); }
        if !head_sent {
            let code = if outcome.mode.is_empty() { 400 } else { 500 };
            json_resp(s, code, &an::error_json(if code == 400 { "invalid_request_error" } else { "api_error" }, e), false, false)?;
        } else {
            wr(s, &mut head_sent, &an::sse_error("api_error", e));
        }
        return Ok(false);
    }
    let (_, calls, stop) = finish_reply(&outcome, &cr);
    log_line(sh, req, &cr, &outcome, stop, &calls, total);
    if broken { return Ok(false); }
    let mut out = String::new();
    if !started {
        out += &an::message_start(&id, &model, an::usage(outcome.n_prompt - outcome.n_reused.min(outcome.n_prompt), outcome.n_reused, 0));
    }
    let tail = vis.as_mut().map(|v| v.finish()).unwrap_or_default();
    let mut extra = tail;
    if outcome.stop_reason == "ctx_full" && outcome.n_new == 0 { extra = an::ctx_full_text(outcome.n_prompt, outcome.max_ctx); }
    if !extra.is_empty() || (!text_open && calls.is_empty()) {
        if !text_open { out += &an::text_block_start(0); text_open = true; }
        out += &an::text_delta(0, if extra.is_empty() { "(no output)" } else { &extra });
    }
    let mut idx = 0;
    if text_open { out += &an::block_stop(0); idx = 1; }
    for (n, a) in &calls { out += &an::tool_use_block(idx, &an::new_id("toolu_"), n, a); idx += 1; }
    out += &an::message_delta(stop, outcome.stop_sequence.as_deref(), an::usage(outcome.n_prompt - outcome.n_reused.min(outcome.n_prompt), outcome.n_reused, outcome.n_new));
    out += &an::message_stop();
    wr(s, &mut head_sent, &out);
    Ok(false) // SSE responses are close-delimited
}

/// A warm-up's reply (always JSON, `stream` ignored): no content, the usage of the prefill, and a `streamdense`
/// object with the status ("done" | "cancelled" | "ctx_full") and the snapshot saves.
#[allow(clippy::too_many_arguments)]
/// While it is queued (no Start yet) the client closing the connection cancels it: the worker drops it before it
/// starts (`worker::prune_warm`, drop reason "client_gone"). A running warm-up is left to finish (its prefill is the
/// snapshot the next request restores).
fn warm_reply(s: &mut TcpStream, req: &Request, sh: &Shared, cr: &ChatReq, id: &str, rx: std::sync::mpsc::Receiver<Ev>, cancel: &AtomicBool,
              t0: Instant, ka: bool) -> std::io::Result<bool> {
    let (mut started, mut gone) = (false, false);
    let out = loop {
        match rx.recv_timeout(WARM_POLL) {
            Ok(Ev::Done(o)) => break Some(o),
            Ok(Ev::Start { .. }) => started = true,
            Ok(_) => {}
            Err(RecvTimeoutError::Timeout) => if !started && !gone && client_gone(s) { gone = true; cancel.store(true, Ordering::SeqCst); },
            Err(RecvTimeoutError::Disconnected) => break None,
        }
    };
    let mut o = out.unwrap_or_else(|| Outcome { stop_reason: "error".into(), error: Some("worker dropped the request".into()), warm: true, ..Default::default() });
    if gone && o.drop_reason.is_some() { o.drop_reason = Some("client_gone"); }
    let total = t0.elapsed().as_secs_f64();
    if let Some(e) = &o.error {
        log_line(sh, req, cr, &o, "error", &[], total);
        let code = if o.n_prompt == 0 && o.mode.is_empty() { 400 } else { 500 };
        json_resp(s, code, &an::error_json(if code == 400 { "invalid_request_error" } else { "api_error" }, e), ka, false)?;
        return Ok(ka);
    }
    let status = match o.stop_reason.as_str() { "warm" => "done", "ctx_full" => "ctx_full", _ => "cancelled" };
    log_line(sh, req, cr, &o, status, &[], total);
    let mut v = an::message_json(id, &cr.model, vec![], "max_tokens", None, an::usage(o.n_prompt - o.n_reused.min(o.n_prompt), o.n_reused, 0));
    v["streamdense"] = json!({"warm": status, "n_prompt": o.n_prompt, "n_reused": o.n_reused, "n_feed": o.n_feed, "mode": o.mode,
        "prefix": prefix_used(&o), "drop": o.drop_reason, "queue_s": o.queue_s, "prefill_s": o.prefill_s, "total_s": total,
        "snap": o.snaps.iter().map(|e| json!({"kind": e.kind, "n": e.n, "error": e.error})).collect::<Vec<_>>()});
    json_resp(s, 200, &v, ka, false)?;
    Ok(ka)
}

/// GET /v1/streamdense/activity query -> `activity::handle_request` JSON: `plan=1` (or keep= / granularity= / layers=1)
/// adds the dry-run unload plan, `path=FILE` writes the full snapshot, `reset=1` starts a new window after reading.
fn activity_query(q: &str) -> Value {
    let mut v = json!({});
    let mut plan = json!({});
    let on = |x: &str| x == "1" || x == "true";
    for kv in q.split('&').filter(|x| !x.is_empty()) {
        let (k, x) = kv.split_once('=').unwrap_or((kv, "1"));
        match k {
            "path" => v["path"] = json!(x),
            "reset" => v["reset"] = json!(on(x)),
            "plan" => if on(x) { v["plan"] = plan.clone(); },
            "keep" => if let Ok(f) = x.parse::<f64>() { plan["keep"] = json!(f); },
            "granularity" => if let Ok(g) = x.parse::<u64>() { plan["granularity"] = json!(g); },
            "layers" => plan["layers"] = json!(on(x)),
            _ => {}
        }
    }
    if v.get("plan").is_some() || !plan.as_object().unwrap().is_empty() {
        let mut p = v["plan"].as_object().cloned().unwrap_or_default();
        for (k, x) in plan.as_object().unwrap() { p.insert(k.clone(), x.clone()); }
        v["plan"] = Value::Object(p);
    }
    v
}
