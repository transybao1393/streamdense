//! streamdense-gateway: the Anthropic Messages API on 127.0.0.1, served by the Qwen3.8-27B engine in-process, so
//! Claude Code can use the local model (`streamdense gateway`, docs/GATEWAY.md).
//!
//! Endpoints: POST /v1/messages (`?beta=true` ignored; SSE or JSON; `max_tokens: 0` / `x-streamdense-warm: 1` = warm-up), POST /v1/messages/count_tokens,
//! GET /v1/models[/id], GET|HEAD / and /health; /v1/streamdense/* via an `Extension` (the domain watcher).
//! One generation at a time: requests queue for the single worker (the thread that owns the engine). KV reuse across requests (prefix / followup / full, see planner.rs).
//!
//! Library API:
//!   serve(&engine, tokenizer, GatewayConfig, Option<snapshot hook>, on_ready(port)) -> blocks
//!   serve_ext(.., Ext { ext: Extension (extra routes + log field), on_job: JobHook (worker thread, after each job) }, ..)
//!   Backend (implemented for QwenEngine), SnapshotHook (no-op NoSnapshots by default; CacheHook = snapshot files,
//!   snap.rs), print_env(..)
//! Modules: http (std HTTP/1.1), anthropic (request conversion, SSE framing), chatmap (template with tools,
//! tool-call parsing, arg coercion, visible-text filter: the only tokenizer-feature boundary), planner (KV modes),
//! worker (engine loop + snapshot cut points + idle steps), snap (snapshot cache policy), snap_tier (hot / warm / cold tiers,
//! idle-time save + pre-read), server (routes, SSE assembly, request log).
pub mod anthropic;
pub mod awake;
pub mod chatmap;
pub mod client;
pub mod compact;
pub mod http;
pub mod planner;
pub mod server;
pub mod snap;
pub mod snap_tier;
pub mod worker;

pub use snap::{CacheHook, SnapTarget};
pub use anthropic::ChatReq;
pub use worker::{Backend, DecSeq, FeedStats, NoSnapshots, Outcome, SnapEvent, SnapshotHook, StreamFeed, WorkerConfig};
use serde_json::Value;
use std::sync::{mpsc, Arc, Mutex};
use streamdense_tokenizer::Tokenizer;

pub const DEFAULT_PORT: u16 = 8898;
pub const MODEL_NAME: &str = "qwen38-27b";

#[derive(Clone, Debug)]
pub struct GatewayConfig {
    pub host: String,
    /// 0 = any free port (reported to `on_ready`).
    pub port: u16,
    pub model_name: String,
    pub worker: WorkerConfig,
    /// Request log file (JSON lines, appended); None = stderr.
    pub log: Option<String>,
    /// Extra fields for the health endpoint.
    pub info: Value,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        GatewayConfig { host: "127.0.0.1".into(), port: DEFAULT_PORT, model_name: MODEL_NAME.into(),
            worker: WorkerConfig { think: false, mtp: true, fold: None }, log: None, info: Value::Null }
    }
}

/// Extra routes + a request-log field from outside the gateway (the domain watcher, streamdense-cli/src/watcher.rs).
/// Called on HTTP handler threads.
pub trait Extension: Send + Sync {
    /// Handle `method path` (paths under /v1/streamdense/); None = not mine (404).
    fn route(&self, method: &str, path: &str, body: &[u8]) -> Option<(u16, Value)>;
    /// (key, value) added to every request-log line and to /health.
    fn log_field(&self) -> Option<(&'static str, Value)> { None }
}

/// Called on the worker thread after each /v1/messages job (after its Done event went out), with the engine.
pub type JobHook<'a, B> = Box<dyn FnMut(&B, &ChatReq, &Outcome) + 'a>;

/// `serve` add-ons: extra routes / log field and the per-job hook.
pub struct Ext<'a, B: ?Sized> {
    pub ext: Option<Arc<dyn Extension>>,
    pub on_job: Option<JobHook<'a, B>>,
}
impl<'a, B: ?Sized> Default for Ext<'a, B> { fn default() -> Self { Ext { ext: None, on_job: None } } }

/// Bind, start the accept thread, call `on_ready(port)`, then run the generation worker on this thread (forever).
pub fn serve<B: Backend + ?Sized>(eng: &B, tk: Arc<Tokenizer>, cfg: GatewayConfig, hook: Option<Box<dyn SnapshotHook<B> + '_>>,
                                  on_ready: impl FnOnce(u16)) -> Result<(), String> {
    serve_ext(eng, tk, cfg, hook, Ext::default(), on_ready)
}

/// `serve` with an `Ext` (watcher routes, log field, per-job hook).
pub fn serve_ext<'a, B: Backend + ?Sized>(eng: &'a B, tk: Arc<Tokenizer>, cfg: GatewayConfig, hook: Option<Box<dyn SnapshotHook<B> + 'a>>,
                                          ext: Ext<'a, B>, on_ready: impl FnOnce(u16)) -> Result<(), String> {
    let listener = std::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).map_err(|e| format!("bind {}:{}: {}", cfg.host, cfg.port, e))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let tiers = hook.as_ref().and_then(|h| h.tiers());
    let (tx, rx) = mpsc::channel();
    let gate = Arc::new(worker::WarmGate::default());
    let sh = Arc::new(server::Shared {
        tk: tk.clone(), model_name: cfg.model_name.clone(), max_ctx: eng.max_ctx(), think: cfg.worker.think,
        jobs: Mutex::new(tx), log: server::Logger::new(cfg.log.as_deref())?, info: cfg.info.clone(), activity: eng.activity(), ext: ext.ext, mode: eng.mode_info(), peak: eng.peak_footprint(), tiers,
        warm: gate.clone(), awake: awake::Awake::start(awake::LINGER),
    });
    std::thread::spawn(move || server::accept_loop(listener, sh));
    let mut w = worker::Worker::new(eng, tk, cfg.worker.clone());
    if let Some(h) = hook { w.hook = h; }
    w.on_job = ext.on_job;
    w.gate = gate;
    on_ready(port);
    w.run(rx);
    Ok(())
}

/// Shell exports that point Claude Code at the gateway (`streamdense gateway --print-env`).
pub fn print_env(host: &str, port: u16, model: &str) -> String {
    format!("export ANTHROPIC_BASE_URL=http://{h}:{p}\n\
export ANTHROPIC_AUTH_TOKEN=dummy\n\
export ANTHROPIC_API_KEY=\n\
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1\n\
export ANTHROPIC_MODEL={m}\n\
export ANTHROPIC_SMALL_FAST_MODEL={m}\n\
export ANTHROPIC_DEFAULT_HAIKU_MODEL={m}\n\
export ANTHROPIC_DEFAULT_SONNET_MODEL={m}\n\
export ANTHROPIC_DEFAULT_OPUS_MODEL={m}\n", h = host, p = port, m = model)
}
