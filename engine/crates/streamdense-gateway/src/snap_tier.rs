//! Snapshot tiers: which engine-state snapshots live where, and the idle-time work that keeps the next
//! turn / a restart warm. Policy + bookkeeping only; `snap::CacheHook` does the engine side (save / restore / RAM copy).
//!
//! Tiers
//! - hot:  in-process RAM copies (`streamdense_model::snapshot::Snapshot`) of recent session states, restored with a
//!         memcpy. Admitted only while `footprint + copy + hot_headroom <= ram_cap` (the live cap the watchdog enforces;
//!         the CPU heap limit is moved by the copy's size, `alloc::shift_limit`) and the hot total <= `hot_bytes`;
//!         dropped first (LRU) when the footprint rises. Never in 12 GB mode (`SnapTarget::hot_capable` false: no
//!         spare RAM there).
//! - warm: `.sdsnap` files in the cache dir (restored straight into the arena). Index `tiers.json` (same dir): kind,
//!         session keys, last use, hit count per file, so hits and recency survive a restart.
//! - cold: deleted by the disk policy: after each save, lowest score first until <= `max_files`, <= `max_bytes`, and
//!         free disk >= `min_free` (the file just written is never deleted); no save at all while free < `min_free`.
//!         score = last use (unix s) + 600 s per hit (max 6) + 1800 s for a stable prefix.
//!
//! Sessions: `project` = hash of the tokens through the first `<|im_end|>` (system prompt + tools), `conv` = through
//! the second (+ the first user turn: one conversation). Per conv the newest `CONV_KEEP` non-stable files are kept
//! (request-time "roll" + idle "idle"); `TierConfig::legacy` (`CacheHook::new`) keeps the old one-rolling-per-gateway rule.
//!
//! Idle work (gateway idle >= `idle`, nothing running or queued; worker thread, one step per wake-up so a request is
//! picked up between steps): 1. save the live session's state (prompt + reply, the engine minus its pending token) when
//! it is >= `idle_save_min` tokens past that conv's newest file; 2. hot copy of it (if room); 3. hand the pre-read plan
//! (most likely next files, not the live conv's) to the `sd-snaptier` thread (background QoS) which reads them through
//! the OS file cache in 4 MB chunks (no F_NOCACHE: the point is the cache; the restore reads through it), checking
//! `TierShared::interrupted` before every chunk. The engine save itself is one call (~0.35 s at 240 MB on the 27B,
//! more for long states) and cannot be cut mid-file; the time a request waited on it is logged (`blocked_ms`).
//! Step 2b (`hot_stable`): a hot copy of the live project's shared prefix (stable file) when it fits without dropping
//! another copy.
//!
//! Multi-user (`switch_save`): when a request of another conversation is about to replace the live state (no idle gap
//! between interleaved users), the live state is saved first ("switch" file, counted like an idle save per conv) when it
//! is >= `switch_save_min` tokens past that conv's newest file (or its project's stable prefix). Restores report where
//! the prefix came from (`restore_src`: "own" = a file of the request's conversation, "shared" = the project prefix).
//! Stable prefixes (several per project: e.g. a client's agent system turn + its planner system turn): each is keyed by its
//! own token hash (`stable_key`; `conv` of a stable entry / hot copy), so two system turns never share a key, not even
//! when a 12 GB-mode cut ends before the `<|im_end|>`. A newer stable prefix of the same family (`same_family`: shares
//! >= `SUPERSEDE_SHARE` of the shorter one's tokens, in `FP_BLOCK` blocks: the planner turn with changed project
//! facts) replaces the older one (file + hot copy). Disk policy: stable files go only after every non-stable file
//! (unless unused for `STABLE_PROTECT_S`); hot tier: a non-stable copy never drops a stable copy, LRU drops non-stable
//! copies first. A warm-up's stable save is copied to the hot tier at once (`hot_stable_file`).
//! Family by header (`StableTag`): with `x-streamdense-project` the entry stores its project id and lane (planner:
//! `x-streamdense-lane: plan` or `streamdense.format == "plan"`; else agent); a newer stable prefix supersedes an
//! older one exactly when both carry the same project + lane. The token-overlap rule is only the fallback when the
//! older entry has no tag; an untagged new prefix never replaces a tagged one, and two tagged entries of different
//! projects or lanes never supersede each other. The static part of the planner turn (before `PROJECT FACTS`,
//! `worker::static_prefix_len`) is a stable prefix of its own: tag `StableTag::plan_static` (lane `plan-static`, no
//! project, shared by every project): it and the facts cut of a project coexist; a newer static cut replaces only an
//! older static one.
//! Eviction first (N8, `evict_for_save` / `stable_ensure`, off by default): switch / idle saves of the live chat (>=
//! `evict_for_save_min` tokens) and new stable prefixes delete files in `room_plan` order until the file fits above the
//! floor (`make_room`) instead of being skipped; the live chat's newest file and its stable prefix are `protected`
//! from both that and the disk policy. Sizes estimated by token count (`est_bytes_for`).
//! Helper heap: index (~200 B / file) + one 4 MB read buffer while pre-reading + the token list of the live state.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use streamdense_model::snapshot::{tokens_hash, SnapHeader, Snapshot};
use streamdense_tokenizer::IM_END;

/// Non-stable files kept per conversation (newest first): the request-time rolling state + the idle save.
pub const CONV_KEEP: usize = 2;
/// Pre-read chunk (the helper's only big buffer).
pub const PREREAD_CHUNK: usize = 4 << 20;
/// A file paged in less than this long ago is not pre-read again.
pub const PREREAD_TTL: Duration = Duration::from_secs(600);
pub const INDEX_FILE: &str = "tiers.json";
const HIT_CREDIT_S: f64 = 600.0;
const STABLE_CREDIT_S: f64 = 1800.0;
/// Fingerprint block (tokens) of a stable prefix: one hash per whole block, compared in order.
pub const FP_BLOCK: usize = 64;
/// A newer stable prefix sharing at least this share of the shorter one's tokens replaces the older one.
pub const SUPERSEDE_SHARE: f64 = 0.75;
/// A stable file unused this long is evicted like any other file (else only after every non-stable file).
pub const STABLE_PROTECT_S: f64 = 86400.0;

#[derive(Clone, Debug, PartialEq)]
pub struct TierConfig {
    pub max_files: usize,
    pub max_bytes: u64,
    /// keep this much free disk (0 = no check)
    pub min_free: u64,
    /// idle work after this long without a request; None = no idle work
    pub idle: Option<Duration>,
    pub idle_save: bool,
    pub idle_save_min: usize,
    pub preread: bool,
    pub preread_files: usize,
    pub preread_bytes: u64,
    /// hot tier budget (0 = off) and the headroom kept under `ram_cap` (0 = no cap known: hot off)
    pub hot_bytes: u64,
    pub hot_headroom: u64,
    pub ram_cap: u64,
    /// one rolling file per conversation (true) or one per gateway (legacy)
    pub per_conv: bool,
    /// keep the index file (`tiers.json`: kinds, hits, last use across restarts); legacy: none (the dir holds only .sdsnap)
    pub index: bool,
    /// idle: also a hot copy of the live project's stable prefix (when it fits without dropping another copy)
    pub hot_stable: bool,
    /// save the live state before another conversation's request replaces it (>= `switch_save_min` tokens new)
    pub switch_save: bool,
    pub switch_save_min: usize,
    /// eviction first (`[snapshots] evict_for_save`): a switch / idle save of a live state >= `evict_for_save_min`
    /// tokens deletes the lowest-score files (`room_plan`; never the protected ones) instead of being skipped by the
    /// free-disk floor; sizes estimated by token count (`est_bytes_for`)
    pub evict_for_save: bool,
    pub evict_for_save_min: usize,
    /// stable prefix always on disk (`[snapshots] stable_ensure`): stable saves evict first, and the live lineage's
    /// stable prefix file is never deleted by the disk policy
    pub stable_ensure: bool,
}

impl TierConfig {
    /// The pre-tier policy (`CacheHook::new`): LRU caps, one rolling file per gateway, no idle work, no hot tier.
    pub fn legacy(max_files: usize, max_bytes: u64) -> TierConfig {
        TierConfig { max_files: max_files.max(1), max_bytes, min_free: 0, idle: None, idle_save: false, idle_save_min: 256, preread: false,
                     preread_files: 0, preread_bytes: 0, hot_bytes: 0, hot_headroom: 0, ram_cap: 0, per_conv: false, index: false,
                     hot_stable: false, switch_save: false, switch_save_min: 64, evict_for_save: false, evict_for_save_min: 2048, stable_ensure: false }
    }
    /// From `[snapshots]` + the live RAM cap in bytes (`[ram] cap_gb` / --abort-gb).
    pub fn from_config(s: &streamdense_core::config::Snapshots, ram_cap: u64) -> TierConfig {
        let gb = |x: f64| (x * 1e9) as u64;
        TierConfig { max_files: s.max_files.max(1), max_bytes: gb(s.max_disk_gb), min_free: gb(s.min_free_disk_gb), idle: Some(Duration::from_secs_f64(s.idle_s)),
                     idle_save: s.idle_save, idle_save_min: s.idle_save_min_tokens, preread: s.preread, preread_files: s.preread_files,
                     preread_bytes: gb(s.preread_gb), hot_bytes: gb(s.hot_gb), hot_headroom: gb(s.hot_headroom_gb), ram_cap, per_conv: true, index: true,
                     hot_stable: s.hot_stable, switch_save: s.switch_save, switch_save_min: s.switch_save_min_tokens,
                     evict_for_save: s.evict_for_save, evict_for_save_min: s.evict_for_save_min_tokens, stable_ensure: s.stable_ensure }
    }
}

/// Shared with the HTTP threads: pending requests (idle work stops at once), the published status (/health).
#[derive(Default)]
pub struct TierShared {
    pending: AtomicUsize,
    busy: AtomicBool,
    status: Mutex<Value>,
}
impl TierShared {
    /// HTTP thread, before a job is queued.
    pub fn arrived(&self) { self.pending.fetch_add(1, Ordering::SeqCst); }
    /// A request is queued or running: idle work must stop.
    pub fn interrupted(&self) -> bool { self.pending.load(Ordering::SeqCst) > 0 || self.busy.load(Ordering::SeqCst) }
    pub fn health(&self) -> Value { self.status.lock().map(|s| s.clone()).unwrap_or(Value::Null) }
    fn taken(&self) { let _ = self.pending.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |p| Some(p.saturating_sub(1))); self.busy.store(true, Ordering::SeqCst); }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind { Stable, Roll, Idle, Switch, Unknown }
impl Kind {
    pub fn as_str(&self) -> &'static str { match self { Kind::Stable => "stable", Kind::Roll => "roll", Kind::Idle => "idle", Kind::Switch => "switch", Kind::Unknown => "?" } }
    fn parse(s: &str) -> Kind { match s { "stable" => Kind::Stable, "roll" => Kind::Roll, "idle" => Kind::Idle, "switch" => Kind::Switch, _ => Kind::Unknown } }
}

/// One warm file.
#[derive(Clone, Debug)]
pub struct Entry {
    pub n: usize,
    pub bytes: u64,
    pub kind: Kind,
    pub project: u64,
    pub conv: u64,
    /// unix s
    pub last_use: f64,
    pub created: f64,
    pub hits: u32,
    /// stable prefixes: `fingerprint` of the tokens (family / supersede check); empty otherwise
    pub blocks: Vec<u64>,
    /// stable prefixes: project id + lane from the request headers (None: no header, overlap fallback)
    pub tag: Option<StableTag>,
}
impl Entry {
    pub fn score(&self) -> f64 { self.last_use + self.hits.min(6) as f64 * HIT_CREDIT_S + if self.kind == Kind::Stable { STABLE_CREDIT_S } else { 0.0 } }
}

/// (project, conv) keys of a token prefix.
pub fn session_keys(tokens: &[u32]) -> (u64, u64) {
    let mut ends = tokens.iter().enumerate().filter(|(_, &t)| t == IM_END).map(|(i, _)| i + 1);
    let p_end = ends.next();
    let project = tokens_hash(&tokens[..p_end.unwrap_or(tokens.len().min(64))]);
    let conv = ends.next().map(|e| tokens_hash(&tokens[..e])).unwrap_or(project);
    (project, conv)
}

/// Project id + lane of a stable prefix (from the request headers; `ChatReq::stable_tag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StableTag {
    /// `x-streamdense-project` (16 hex); 0 for a shared (`stat`) prefix
    pub project: u64,
    /// planner system turn (`x-streamdense-lane: plan` / `format: plan`); false = agent
    pub plan: bool,
    /// the static part of the planner turn (before `worker::PROJECT_FACTS_MARKER`): identical for every project, so
    /// shared (no project), lane `plan-static`; one copy (a newer one supersedes only an older static one)
    pub stat: bool,
}
impl StableTag {
    /// The tag of the static planner cut.
    pub fn plan_static() -> StableTag { StableTag { project: 0, plan: true, stat: true } }
    pub fn lane(&self) -> &'static str { if self.stat { "plan-static" } else if self.plan { "plan" } else { "agent" } }
    /// Inverse of `lane` (index load).
    pub fn from_lane(project: u64, lane: &str) -> StableTag {
        if lane == "plan-static" { StableTag::plan_static() } else { StableTag { project, plan: lane == "plan", stat: false } }
    }
}

/// Does a newer stable prefix (`new`: tag, fingerprint, length) replace an older one (`old`)? Both tagged: same
/// project and lane. An untagged new one never replaces a tagged one (a request without the header cannot claim a
/// project's prefix). Otherwise (old untagged: no header, or an older index) the token-overlap fallback `same_family`.
pub fn supersedes(new: (Option<StableTag>, &[u64], usize), old: (Option<StableTag>, &[u64], usize)) -> bool {
    match (new.0, old.0) {
        (Some(a), Some(b)) => a == b,
        (None, Some(_)) => false,
        // the static planner cut is a token prefix of every facts cut: overlap must not make it replace one
        (Some(a), None) if a.stat => false,
        _ => same_family(new.1, new.2, old.1, old.2),
    }
}

/// Key of a stable prefix (entry `conv` / hot copy key): its own token hash.
pub fn stable_key(tokens: &[u32]) -> u64 { tokens_hash(tokens) }

/// Prefix fingerprint: one hash per whole `FP_BLOCK` block.
pub fn fingerprint(tokens: &[u32]) -> Vec<u64> { tokens.chunks_exact(FP_BLOCK).map(tokens_hash).collect() }

/// Two stable prefixes (fingerprints + lengths) of one family: they share >= `SUPERSEDE_SHARE` of the shorter one
/// (and >= one block). Agent and planner system turns share little; a planner turn with changed facts shares most.
pub fn same_family(a: &[u64], an: usize, b: &[u64], bn: usize) -> bool {
    let shared = a.iter().zip(b).take_while(|(x, y)| x == y).count() * FP_BLOCK;
    let m = an.min(bn);
    m > 0 && shared >= FP_BLOCK && shared as f64 >= SUPERSEDE_SHARE * m as f64
}

pub fn now_unix() -> f64 { SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0) }

/// Free bytes on the file system holding `dir` (None = unknown).
pub fn disk_free(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 { return None; }
    Some(s.f_bavail as u64 * s.f_frsize as u64)
}

/// This thread -> background QoS (efficiency cores, lowest I/O priority). Ok = the call succeeded.
fn background_qos() -> bool {
    #[cfg(target_os = "macos")]
    unsafe { return libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0) == 0; }
    #[allow(unreachable_code)]
    false
}

/// One hot (RAM) copy.
pub struct Hot {
    pub snap: Snapshot,
    /// conversation key; a stable prefix copy: `stable_key`
    pub conv: u64,
    pub stable: bool,
    /// a stable copy: its entry's `tag`
    pub tag: Option<StableTag>,
    pub last_use: Instant,
    pub hits: u32,
    /// bytes charged (body + header tokens)
    pub bytes: u64,
}

/// Counters (since start) and the idle work since the previous job (request log).
#[derive(Clone, Debug, Default)]
pub struct Counters {
    pub hit_hot: u64,
    pub hit_warm: u64,
    pub miss: u64,
    pub saves: u64,
    pub saves_skipped_disk: u64,
    pub evicted: u64,
    pub evicted_bytes: u64,
    pub hot_admitted: u64,
    pub hot_refused: u64,
    pub hot_dropped: u64,
    pub idle_saves: u64,
    pub idle_save_ms: f64,
    pub switch_saves: u64,
    /// stable prefixes replaced by a newer one of their family
    pub superseded: u64,
    pub preread_files: u64,
    pub preread_bytes: u64,
    pub preread_cancelled: u64,
    pub blocked_ms: f64,
}

#[derive(Default)]
struct PrereadStats { files: u64, bytes: u64, cancelled: u64, ms: f64, running: bool, done: Vec<String> }

struct Preread { tx: Sender<Vec<(PathBuf, u64)>>, stats: Arc<Mutex<PrereadStats>> }

fn spawn_preread(shared: Arc<TierShared>) -> Option<Preread> {
    let (tx, rx) = channel::<Vec<(PathBuf, u64)>>();
    let stats = Arc::new(Mutex::new(PrereadStats::default()));
    let st = stats.clone();
    std::thread::Builder::new().name("sd-snaptier".into()).spawn(move || {
        background_qos();
        for plan in rx {
            let t0 = Instant::now();
            let mut buf = vec![0u8; PREREAD_CHUNK];
            let (mut files, mut bytes, mut cancelled, mut done) = (0u64, 0u64, false, vec![]);
            'plan: for (p, _) in plan {
                let Ok(mut f) = std::fs::File::open(&p) else { continue };
                loop {
                    if shared.interrupted() { cancelled = true; break 'plan; }
                    match f.read(&mut buf) { Ok(0) => break, Ok(k) => bytes += k as u64, Err(_) => break }
                }
                files += 1;
                if let Some(n) = p.file_name() { done.push(n.to_string_lossy().to_string()); }
            }
            drop(buf);
            if let Ok(mut s) = st.lock() {
                s.files += files; s.bytes += bytes; s.cancelled += cancelled as u64; s.ms += t0.elapsed().as_secs_f64() * 1e3;
                s.running = false; s.done.extend(done);
            }
        }
    }).ok()?;
    Some(Preread { tx, stats })
}

/// Idle steps of one idle period, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step { Save, Hot, HotStable, Preread }

pub struct Tiers {
    pub cfg: TierConfig,
    pub dir: PathBuf,
    pub entries: HashMap<String, Entry>,
    pub hot: Vec<Hot>,
    pub shared: Arc<TierShared>,
    pub c: Counters,
    /// idle work since the previous job (moved into the request log at the next job)
    since: Counters,
    /// idle steps left in this idle period; when it started (job end)
    pub steps: Vec<Step>,
    idle_from: Option<Instant>,
    /// last blocking idle step (start, end): a request that arrived during it waited
    last_block: Option<(Instant, Instant)>,
    preread: Option<Preread>,
    paged: HashMap<String, Instant>,
    /// the live engine state: (conv, n) and its project
    pub live: Option<(u64, usize)>,
    pub live_project: Option<u64>,
    /// the longest stable file that is a prefix of the live state (file name)
    pub live_stable: Option<String>,
    /// tests: a fake free-disk figure used instead of statvfs (a save takes its bytes, a deletion gives them back);
    /// shared so a test can move it between requests while the gateway owns the hook
    pub fake_free: Option<Arc<AtomicU64>>,
    dirty: bool,
}

impl Tiers {
    /// Open over the cache dir: load the index, add files it does not know (keys from their headers), drop stale rows.
    pub fn open(dir: &Path, cfg: TierConfig) -> Tiers {
        let mut t = Tiers { cfg, dir: dir.to_path_buf(), entries: HashMap::new(), hot: vec![], shared: Arc::new(TierShared::default()), c: Counters::default(),
                            since: Counters::default(), steps: vec![], idle_from: None, last_block: None, preread: None, paged: HashMap::new(), live: None, live_project: None, live_stable: None, fake_free: None, dirty: false };
        let idx = if t.cfg.index { std::fs::read_to_string(dir.join(INDEX_FILE)).ok() } else { None };
        if let Some(s) = idx {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                for (name, e) in v["files"].as_object().into_iter().flatten() {
                    let u = |k: &str| e[k].as_u64().unwrap_or(0);
                    let f = |k: &str| e[k].as_f64().unwrap_or(0.0);
                    let h = |k: &str| e[k].as_str().and_then(|x| u64::from_str_radix(x, 16).ok()).unwrap_or(0);
                    let blocks = e["blocks"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().and_then(|x| u64::from_str_radix(x, 16).ok())).collect()).unwrap_or_default();
                    let tag = e["pid"].as_str().and_then(|x| u64::from_str_radix(x, 16).ok())
                        .map(|project| StableTag::from_lane(project, e["lane"].as_str().unwrap_or("")));
                    t.entries.insert(name.clone(), Entry { n: u("n") as usize, bytes: u("bytes"), kind: Kind::parse(e["kind"].as_str().unwrap_or("")),
                        project: h("project"), conv: h("conv"), last_use: f("last_use"), created: f("created"), hits: u("hits") as u32, blocks, tag });
                }
            }
        }
        t.sync_dir();
        // stable entries of an older index: no fingerprint yet -> from the file header (the key: the own token hash)
        let dir = t.dir.clone();
        for (name, e) in t.entries.iter_mut().filter(|(_, e)| e.kind == Kind::Stable && e.blocks.is_empty()) {
            if let Ok(h) = SnapHeader::read_file(&dir.join(name)) { e.blocks = fingerprint(&h.tokens); e.conv = stable_key(&h.tokens); t.dirty = true; }
        }
        t.publish();
        t
    }

    /// Re-list the dir: unknown files get an entry (header read), vanished ones are dropped, sizes refreshed.
    pub fn sync_dir(&mut self) {
        let mut seen = std::collections::HashSet::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if !name.ends_with(".sdsnap") { continue; }
                let Ok(m) = e.metadata() else { continue };
                seen.insert(name.clone());
                if let Some(x) = self.entries.get_mut(&name) { x.bytes = m.len(); continue; }
                let mt = m.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0);
                let (n, project, conv) = match SnapHeader::read_file(&e.path()) {
                    Ok(h) => { let (p, c) = session_keys(&h.tokens); (h.tokens.len(), p, c) }
                    Err(_) => (0, 0, 0),
                };
                self.entries.insert(name, Entry { n, bytes: m.len(), kind: Kind::Unknown, project, conv, last_use: mt, created: mt, hits: 0, blocks: vec![], tag: None });
                self.dirty = true;
            }
        }
        let before = self.entries.len();
        self.entries.retain(|k, _| seen.contains(k));
        if self.entries.len() != before { self.dirty = true; }
    }

    pub fn usage(&self) -> (usize, u64) { (self.entries.len(), self.entries.values().map(|e| e.bytes).sum()) }
    fn hot_bytes(&self) -> u64 { self.hot.iter().map(|h| h.bytes).sum() }

    /// A file was written (`kind`, its tokens). Per-conv / per-gateway rolling replacement, then the disk policy.
    pub fn saved(&mut self, path: &Path, tokens: &[u32], kind: Kind) -> Vec<String> { self.saved_tagged(path, tokens, kind, None) }
    /// `saved` with the request's stable tag (project + lane; kept on a stable entry, used for the supersede check).
    pub fn saved_tagged(&mut self, path: &Path, tokens: &[u32], kind: Kind, tag: Option<StableTag>) -> Vec<String> {
        let name = file_name(path);
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let (project, conv) = session_keys(tokens);
        let now = now_unix();
        let old = self.entries.get(&name).cloned();
        let kind = match old.as_ref().map(|o| o.kind) { Some(Kind::Stable) => Kind::Stable, _ => kind }; // a stable file stays stable
        // a stable prefix is keyed by its own tokens: two system turns of one project never share a key
        let (conv, blocks) = if kind == Kind::Stable { (stable_key(tokens), fingerprint(tokens)) } else { (conv, vec![]) };
        let tag = if kind == Kind::Stable { tag.or(old.as_ref().and_then(|o| o.tag)) } else { None };
        // the static planner cut belongs to no project (shared by all)
        let project = if tag.is_some_and(|g| g.stat) { 0 } else { project };
        self.entries.insert(name.clone(), Entry { n: tokens.len(), bytes, kind, project, conv, last_use: now, created: now,
                                                   hits: old.as_ref().map(|o| o.hits).unwrap_or(0), blocks: blocks.clone(), tag });
        self.paged.insert(name.clone(), Instant::now()); // just written: in the file cache
        self.c.saves += 1;
        if let Some(f) = &self.fake_free { let d = bytes.saturating_sub(old.as_ref().map_or(0, |o| o.bytes)); let _ = f.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(x.saturating_sub(d))); }
        // the prefix a request just read is its lineage's: protected from the disk policy from now on
        if kind == Kind::Stable && self.cfg.stable_ensure { self.live_stable = Some(name.clone()); }
        self.dirty = true;
        let mut gone = vec![];
        if kind != Kind::Stable {
            // replaced rolling states: legacy = every other roll; per conv = beyond the newest CONV_KEEP of this conv
            let mut same: Vec<(String, f64)> = self.entries.iter()
                .filter(|(k, e)| **k != name && e.kind != Kind::Stable && if self.cfg.per_conv { e.conv == conv && e.kind != Kind::Unknown } else { e.kind == Kind::Roll })
                .map(|(k, e)| (k.clone(), e.created)).collect();
            same.sort_by(|a, b| b.1.total_cmp(&a.1));
            let keep = if self.cfg.per_conv { CONV_KEEP - 1 } else { 0 };
            for (k, _) in same.into_iter().skip(keep) { if self.remove(&k, "replaced") { gone.push(k); } }
        } else {
            gone.extend(self.supersede(&name, conv, &blocks, tokens.len(), tag));
        }
        gone.extend(self.evict(Some(&name)));
        gone
    }

    /// A new stable prefix (`name`, key `key`, `tag`): older stable files of its family (`supersedes`) go, with their hot copies.
    fn supersede(&mut self, name: &str, key: u64, blocks: &[u64], n: usize, tag: Option<StableTag>) -> Vec<String> {
        let stale: Vec<(String, u64)> = self.entries.iter()
            .filter(|(k, e)| k.as_str() != name && e.kind == Kind::Stable && e.conv != key && supersedes((tag, blocks, n), (e.tag, &e.blocks, e.n)))
            .map(|(k, e)| (k.clone(), e.conv)).collect();
        let mut gone = vec![];
        for (k, old) in stale {
            if self.remove(&k, "superseded") { self.c.superseded += 1; self.drop_hot_key(old, true); gone.push(k); }
        }
        // a hot copy of a family member without a file entry (file evicted earlier) goes too
        let olds: Vec<usize> = self.hot.iter().enumerate()
            .filter(|(_, h)| h.stable && h.conv != key && supersedes((tag, blocks, n), (h.tag, &fingerprint(h.snap.tokens()), h.snap.tokens().len()))).map(|(i, _)| i).collect();
        for i in olds.into_iter().rev() { self.drop_hot(i); }
        gone
    }

    fn remove(&mut self, name: &str, why: &str) -> bool {
        let p = self.dir.join(name);
        let ok = std::fs::remove_file(&p).is_ok() || !p.exists();
        if ok {
            if let Some(e) = self.entries.remove(name) {
                if why != "replaced" && why != "superseded" { self.c.evicted += 1; self.c.evicted_bytes += e.bytes; }
                if let Some(f) = &self.fake_free { f.fetch_add(e.bytes, Ordering::SeqCst); }
            }
            self.paged.remove(name);
            self.dirty = true;
            eprintln!("gateway: snapshot tiers: deleted {} ({})", p.display(), why);
        }
        ok
    }

    /// Disk policy: lowest score first until within max_files / max_bytes / min_free; never `keep`. Stable files
    /// (used within `STABLE_PROTECT_S`) only after every other file: two stable prefixes never evict each other
    /// while a rolling state is left to delete.
    pub fn evict(&mut self, keep: Option<&str>) -> Vec<String> {
        self.sync_dir();
        let now = now_unix();
        let protect = self.protected();
        let mut order: Vec<(String, (bool, f64), u64)> = self.entries.iter().filter(|(k, _)| Some(k.as_str()) != keep && !protect.contains(k))
            .map(|(k, e)| (k.clone(), (e.kind == Kind::Stable && now - e.last_use < STABLE_PROTECT_S, e.score()), e.bytes)).collect();
        order.sort_by(|a, b| a.1.0.cmp(&b.1.0).then(a.1.1.total_cmp(&b.1.1)));
        let (mut n, mut bytes) = self.usage();
        let mut free = if self.cfg.min_free > 0 { self.free() } else { None };
        let mut gone = vec![];
        for (k, _, b) in order {
            let low = free.is_some_and(|f| f < self.cfg.min_free);
            if n <= self.cfg.max_files && bytes <= self.cfg.max_bytes && !low { break; }
            let why = if low { "free disk below min_free_disk_gb" } else if n > self.cfg.max_files { "file cap" } else { "disk cap" };
            if self.remove(&k, why) {
                n -= 1; bytes = bytes.saturating_sub(b);
                if let Some(f) = free.as_mut() { *f += b; }
                gone.push(k);
            }
        }
        gone
    }

    /// May a file of about `est` bytes be written now? (frees cold files first when the disk is low)
    pub fn disk_ok(&mut self, est: u64) -> Result<(), String> {
        if self.cfg.min_free == 0 { return Ok(()); }
        let Some(f) = self.free() else { return Ok(()) };
        if f >= self.cfg.min_free + est { return Ok(()); }
        self.evict(None);
        let f = self.free().unwrap_or(f);
        if f >= self.cfg.min_free + est { return Ok(()); }
        self.c.saves_skipped_disk += 1;
        Err(format!("disk low: {:.2} GB free < min_free_disk_gb {:.2} + {:.2} GB (save skipped)", f as f64 / 1e9, self.cfg.min_free as f64 / 1e9, est as f64 / 1e9))
    }

    /// Size estimate of the next file: the largest file we know (a fresh cache: 0).
    pub fn est_bytes(&self) -> u64 { self.entries.values().map(|e| e.bytes).max().unwrap_or(0) }

    /// Size estimate of a file of `n` tokens (sizes grow with the token count): the smallest known file of >= n
    /// tokens, else the largest known file scaled by n / its tokens (an overestimate: its fixed part scales too);
    /// a fresh cache: 0. A 1.9K-token stable prefix is no longer charged the 1.4 GB of a 20K-token state.
    pub fn est_bytes_for(&self, n: usize) -> u64 {
        let known = || self.entries.values().filter(|e| e.n > 0 && e.bytes > 0);
        if let Some(e) = known().filter(|e| e.n >= n).min_by_key(|e| e.n) { return e.bytes; }
        known().max_by_key(|e| e.n).map_or(0, |e| (e.bytes as f64 * n as f64 / e.n as f64).ceil() as u64)
    }

    /// Free bytes on the cache disk (`fake_free` in tests).
    pub fn free(&self) -> Option<u64> { match &self.fake_free { Some(f) => Some(f.load(Ordering::SeqCst)), None => disk_free(&self.dir) } }

    /// Files the disk policy must not delete: with `stable_ensure` the live lineage's stable prefix; with
    /// `evict_for_save` also the newest file of the live conversation (a request of another conversation runs: its
    /// saves never take the state that conversation comes back to).
    pub fn protected(&self) -> Vec<String> {
        let mut p = vec![];
        if self.cfg.stable_ensure { if let Some(s) = &self.live_stable { p.push(s.clone()); } }
        if self.cfg.evict_for_save {
            if let Some((conv, _)) = self.live {
                if let Some((k, _)) = self.entries.iter().filter(|(_, e)| e.conv == conv && e.kind != Kind::Stable).max_by_key(|(_, e)| e.n) { p.push(k.clone()); }
            }
        }
        p
    }

    /// Eviction first: make `est` bytes of room above the `min_free` floor for a save of `n` tokens (`what`: kind,
    /// for the log) by deleting files in `room_plan` order, never `protect` (+ `protected()`). Ok(deleted names) when
    /// the floor holds after it (nothing deleted when it held already); Err (nothing deleted, counted as skipped)
    /// when deleting every candidate could not make the room.
    pub fn make_room(&mut self, est: u64, n: usize, what: &str, protect: &[String]) -> Result<Vec<String>, String> {
        if self.cfg.min_free == 0 { return Ok(vec![]); }
        self.sync_dir();
        let Some(f) = self.free() else { return Ok(vec![]) };
        let need = self.cfg.min_free + est;
        if f >= need { return Ok(vec![]); }
        let mut keep = self.protected();
        keep.extend(protect.iter().cloned());
        let now = now_unix();
        let cands: Vec<RoomCand> = self.entries.iter().filter(|(k, _)| !keep.contains(k))
            .map(|(k, e)| RoomCand { name: k.clone(), stable: e.kind == Kind::Stable, score: e.score(), bytes: e.bytes, age_s: now - e.last_use }).collect();
        let gb = |b: u64| b as f64 / 1e9;
        let Some(plan) = room_plan(&cands, f, need) else {
            self.c.saves_skipped_disk += 1;
            let all: u64 = cands.iter().map(|c| c.bytes).sum();
            return Err(format!("disk low: {:.2} GB free < min_free_disk_gb {:.2} + {:.2} GB, and deleting every other file ({} files, {:.2} GB; {} protected) cannot make room (save skipped)",
                               gb(f), gb(self.cfg.min_free), gb(est), cands.len(), gb(all), keep.len()));
        };
        let mut gone = vec![];
        let mut desc = vec![];
        for name in plan {
            let e = self.entries.get(&name).cloned();
            if self.remove(&name, "room for a save (eviction first)") {
                if let Some(e) = e { desc.push(format!("{} ({}, {} tokens, {:.2} GB, age {:.0} s)", name, e.kind.as_str(), e.n, gb(e.bytes), (now - e.last_use).max(0.0))); }
                gone.push(name);
            }
        }
        let f2 = self.free().unwrap_or(f);
        eprintln!("gateway: snapshot tiers: eviction first for a {} save ({} tokens, est {:.2} GB): free {:.2} -> {:.2} GB (floor {:.2}); evicted {}",
                  what, n, gb(est), gb(f), gb(f2), gb(self.cfg.min_free), desc.join(", "));
        if f2 >= need { Ok(gone) } else {
            self.c.saves_skipped_disk += 1;
            Err(format!("disk low after eviction first: {:.2} GB free < min_free_disk_gb {:.2} + {:.2} GB (save skipped)", gb(f2), gb(self.cfg.min_free), gb(est)))
        }
    }

    /// A warm file was restored.
    pub fn hit_warm(&mut self, path: &Path) {
        if let Some(e) = self.entries.get_mut(&file_name(path)) { e.hits += 1; e.last_use = now_unix(); self.dirty = true; }
        self.c.hit_warm += 1;
    }
    pub fn miss(&mut self) { self.c.miss += 1; }

    /// Longest hot copy whose tokens are a prefix of `tokens`: (index, n).
    pub fn hot_best(&self, tokens: &[u32]) -> Option<(usize, usize)> {
        self.hot.iter().enumerate().filter(|(_, h)| { let t = h.snap.tokens(); !t.is_empty() && t.len() <= tokens.len() && t == &tokens[..t.len()] })
            .map(|(i, h)| (i, h.snap.tokens().len())).max_by_key(|x| x.1)
    }
    pub fn hit_hot(&mut self, i: usize) { if let Some(h) = self.hot.get_mut(i) { h.hits += 1; h.last_use = Instant::now(); } self.c.hit_hot += 1; }

    /// Hot admission for `bytes` of a non-stable copy (`hot_room_for(bytes, false)`).
    pub fn hot_room(&mut self, bytes: u64) -> Result<(), String> { self.hot_room_for(bytes, false) }
    /// Hot admission for `bytes`: budget (LRU non-stable copies make room; a stable copy is dropped only for another
    /// stable one, never for a conversation copy), then the live footprint against the cap.
    pub fn hot_room_for(&mut self, bytes: u64, stable: bool) -> Result<(), String> {
        if self.cfg.hot_bytes == 0 { return Err("hot_gb = 0".into()); }
        if self.cfg.ram_cap == 0 { return Err("no RAM cap known".into()); }
        if bytes > self.cfg.hot_bytes { return Err(format!("copy {:.2} GB > hot_gb {:.2}", bytes as f64 / 1e9, self.cfg.hot_bytes as f64 / 1e9)); }
        let fixed: u64 = if stable { 0 } else { self.hot.iter().filter(|h| h.stable).map(|h| h.bytes).sum() };
        if fixed + bytes > self.cfg.hot_bytes {
            return Err(format!("copy {:.2} GB + stable copies {:.2} > hot_gb {:.2}", bytes as f64 / 1e9, fixed as f64 / 1e9, self.cfg.hot_bytes as f64 / 1e9));
        }
        while self.hot_bytes() + bytes > self.cfg.hot_bytes { self.drop_hot_lru(); }
        let fp = streamdense_core::footprint::self_footprint();
        if fp + bytes + self.cfg.hot_headroom > self.cfg.ram_cap {
            return Err(format!("footprint {:.2} GB + copy {:.2} + headroom {:.2} > cap {:.2}", fp as f64 / 1e9, bytes as f64 / 1e9,
                               self.cfg.hot_headroom as f64 / 1e9, self.cfg.ram_cap as f64 / 1e9));
        }
        Ok(())
    }
    /// Charge a new hot copy of a conversation state (the heap limit was raised by `bytes` before it was read).
    pub fn hot_add(&mut self, snap: Snapshot, bytes: u64) { self.hot_add_kind(snap, bytes, false) }
    /// Charge a new hot copy; `stable`: a stable prefix (keyed by its own tokens; its family's older copies go).
    pub fn hot_add_kind(&mut self, snap: Snapshot, bytes: u64, stable: bool) {
        let conv = if stable { stable_key(snap.tokens()) } else { session_keys(snap.tokens()).1 };
        // an older copy of the same conversation (same kind) is superseded
        self.drop_hot_key(conv, stable);
        let tag = if stable { self.stable_tag_of(conv) } else { None };
        if stable {
            let (fp, n) = (fingerprint(snap.tokens()), snap.tokens().len());
            let olds: Vec<usize> = self.hot.iter().enumerate()
                .filter(|(_, h)| h.stable && supersedes((tag, &fp, n), (h.tag, &fingerprint(h.snap.tokens()), h.snap.tokens().len()))).map(|(i, _)| i).collect();
            for i in olds.into_iter().rev() { self.drop_hot(i); }
        }
        self.hot.push(Hot { snap, conv, stable, tag, last_use: Instant::now(), hits: 0, bytes });
        self.c.hot_admitted += 1;
    }
    /// Drop the hot copies with key `key` of this kind.
    fn drop_hot_key(&mut self, key: u64, stable: bool) {
        let olds: Vec<usize> = self.hot.iter().enumerate().filter(|(_, h)| h.conv == key && h.stable == stable).map(|(i, _)| i).collect();
        for i in olds.into_iter().rev() { self.drop_hot(i); }
    }
    /// The tag of the stable file entry with key `key` (None: no entry / untagged).
    fn stable_tag_of(&self, key: u64) -> Option<StableTag> {
        self.entries.values().find(|e| e.kind == Kind::Stable && e.conv == key).and_then(|e| e.tag)
    }
    /// Is this stable prefix (key) in the hot tier?
    pub fn hot_has_stable(&self, key: u64) -> bool { self.hot.iter().any(|h| h.stable && h.conv == key) }
    /// Hot copy of a stable prefix file (warm-up save, idle step 2b): read it into RAM when the tier admits it.
    /// `drop_others`: may drop non-stable copies (LRU) to make room; else only when it fits next to them.
    /// Ok(true) = admitted, Ok(false) = there already / no room (counted refused), Err = read error.
    pub fn hot_stable_file(&mut self, path: &Path, drop_others: bool) -> Result<bool, String> {
        let name = file_name(path);
        let key = match self.entries.get(&name) { Some(e) if e.kind == Kind::Stable => e.conv, _ => return Ok(false) };
        if self.hot_has_stable(key) { return Ok(false); }
        let bytes = std::fs::metadata(path).map_err(|e| format!("{}: {}", path.display(), e))?.len();
        // older copies of this family go first (they are replaced by this one)
        let (blocks, n, tag) = { let e = &self.entries[&name]; (e.blocks.clone(), e.n, e.tag) };
        let olds: Vec<usize> = self.hot.iter().enumerate()
            .filter(|(_, h)| h.stable && supersedes((tag, &blocks, n), (h.tag, &fingerprint(h.snap.tokens()), h.snap.tokens().len()))).map(|(i, _)| i).collect();
        for i in olds.into_iter().rev() { self.drop_hot(i); }
        if !drop_others && self.hot_bytes() + bytes > self.cfg.hot_bytes { return Ok(false); }
        if let Err(e) = self.hot_room_for(bytes, true) {
            self.hot_counted(false);
            eprintln!("gateway: snapshot tiers: no hot copy of the stable prefix: {}", e);
            return Ok(false);
        }
        streamdense_core::alloc::shift_limit(bytes as isize); // budgeted heap: the CPU heap limit moves with the copy
        match Snapshot::read(path) {
            Ok(s) => { self.hot_add_kind(s, bytes, true); self.hot_counted(true); Ok(true) }
            Err(e) => { streamdense_core::alloc::shift_limit(-(bytes as isize)); Err(format!("hot copy {}: {}", path.display(), e)) }
        }
    }
    pub fn drop_hot(&mut self, i: usize) {
        if i < self.hot.len() {
            let h = self.hot.remove(i);
            let b = h.bytes;
            drop(h);
            streamdense_core::alloc::shift_limit(-(b as isize));
            self.c.hot_dropped += 1;
        }
    }
    /// LRU drop: non-stable copies first, then stable ones.
    fn drop_hot_lru(&mut self) {
        if let Some(i) = self.hot.iter().enumerate().min_by_key(|(_, h)| (h.stable, h.last_use)).map(|(i, _)| i) { self.drop_hot(i); }
    }
    /// Footprint pressure: drop hot copies (LRU) until footprint + headroom <= cap.
    pub fn hot_pressure(&mut self) {
        if self.hot.is_empty() || self.cfg.ram_cap == 0 { return; }
        while !self.hot.is_empty() && streamdense_core::footprint::self_footprint() + self.cfg.hot_headroom > self.cfg.ram_cap { self.drop_hot_lru(); }
    }

    // ---- job / idle cycle (worker thread)

    /// A job left the queue: idle work stops; returns the request-log fields (idle work since the previous job).
    pub fn job_begin(&mut self, t_recv: Instant) -> Value {
        self.shared.taken();
        self.steps.clear();
        self.idle_from = None;
        if let Some((s, e)) = self.last_block.take() {
            if t_recv < e { let from = if t_recv > s { t_recv } else { s }; let ms = e.duration_since(from).as_secs_f64() * 1e3; self.since.blocked_ms += ms; self.c.blocked_ms += ms; }
        }
        self.poll_preread();
        self.hot_pressure();
        let s = std::mem::take(&mut self.since);
        let mut v = json!({"idle_saves": s.idle_saves, "idle_save_ms": (s.idle_save_ms * 10.0).round() / 10.0, "preread_files": s.preread_files,
                           "preread_mb": (s.preread_bytes as f64 / 1e5).round() / 10.0, "preread_cancelled": s.preread_cancelled,
                           "hot_admitted": s.hot_admitted, "blocked_ms": (s.blocked_ms * 10.0).round() / 10.0});
        if s.evicted > 0 { v["evicted"] = json!(s.evicted); }
        v
    }

    /// The job is done: arm the idle period (`live` = the engine's state keys).
    pub fn job_end(&mut self, fed: Option<&[u32]>) {
        let keys = fed.filter(|f| !f.is_empty()).map(|f| (session_keys(f), f.len()));
        self.live = keys.map(|((_, c), n)| (c, n));
        self.live_project = keys.map(|((p, _), _)| p);
        self.live_stable = fed.and_then(|f| self.stable_prefix_of(f).map(|(k, _)| k));
        self.shared.busy.store(false, Ordering::SeqCst);
        if self.cfg.idle.is_some() {
            self.steps = vec![Step::Save, Step::Hot, Step::HotStable, Step::Preread];
            self.idle_from = Some(Instant::now());
        }
        self.save_index();
        self.publish();
    }

    /// How long the worker may block before the next idle step (None = nothing left).
    pub fn idle_wait(&self) -> Option<Duration> {
        let (d, from) = (self.cfg.idle?, self.idle_from?);
        if self.steps.is_empty() { return None; }
        if self.shared.interrupted() { return Some(Duration::from_millis(50)); } // a request is on its way into the queue
        Some(d.saturating_sub(from.elapsed()))
    }

    /// The next idle step if it is due and nothing is waiting (the caller runs it, then `step_done`).
    pub fn next_step(&mut self) -> Option<Step> {
        if self.shared.interrupted() { return None; }
        let (d, from) = (self.cfg.idle?, self.idle_from?);
        if from.elapsed() < d || self.steps.is_empty() { return None; }
        Some(self.steps.remove(0))
    }
    /// Record a blocking idle step's span (a request arriving inside it waited).
    pub fn step_span(&mut self, t0: Instant) { self.last_block = Some((t0, Instant::now())); }

    /// Count an idle save (+ its ms).
    pub fn idle_saved(&mut self, ms: f64) { for c in [&mut self.c, &mut self.since] { c.idle_saves += 1; c.idle_save_ms += ms; } }
    pub fn hot_counted(&mut self, ok: bool) { if ok { self.since.hot_admitted += 1; } else { self.c.hot_refused += 1; } }

    /// Newest file of `conv` (n tokens).
    pub fn conv_newest(&self, conv: u64) -> Option<usize> { self.entries.values().filter(|e| e.conv == conv && e.kind != Kind::Stable).map(|e| e.n).max() }

    /// The stable (shared prefix) file of `project`: (path, n), the longest if several.
    pub fn stable_of(&self, project: u64) -> Option<(PathBuf, usize)> {
        self.entries.iter().filter(|(_, e)| e.kind == Kind::Stable && e.project == project && e.n > 0).max_by_key(|(_, e)| e.n)
            .map(|(k, e)| (self.dir.join(k), e.n))
    }

    /// The longest stable file whose tokens are a prefix of `tokens` (exact: its own token hash): (file name, n).
    pub fn stable_prefix_of(&self, tokens: &[u32]) -> Option<(String, usize)> {
        self.entries.iter().filter(|(_, e)| e.kind == Kind::Stable && e.n > 0 && e.n <= tokens.len() && stable_key(&tokens[..e.n]) == e.conv)
            .max_by_key(|(_, e)| e.n).map(|(k, e)| (k.clone(), e.n))
    }
    /// The live state's stable prefix file: (path, n).
    pub fn live_stable_file(&self) -> Option<(PathBuf, usize)> {
        let k = self.live_stable.as_ref()?;
        self.entries.get(k).map(|e| (self.dir.join(k), e.n))
    }

    /// The tokens of a session already covered by files: its conversation's newest file, else its project's stable prefix.
    pub fn covered(&self, project: u64, conv: u64) -> usize {
        self.conv_newest(conv).or_else(|| self.stable_of(project).map(|s| s.1)).unwrap_or(0)
    }

    /// A switch save (live state saved before another conversation's request): counted with the saves.
    pub fn switch_saved(&mut self) { self.c.switch_saves += 1; }

    /// The pre-read plan: files most likely needed next, by score, not the live conversation's (the engine holds it),
    /// not paged within PREREAD_TTL, not covered by a hot copy, within preread_files / preread_bytes.
    pub fn preread_plan(&self) -> Vec<(PathBuf, u64)> {
        let live = self.live.map(|l| l.0);
        let mut c: Vec<(&String, &Entry)> = self.entries.iter()
            .filter(|(k, e)| Some(e.conv) != live && e.n > 0 && !self.paged.get(*k).is_some_and(|t| t.elapsed() < PREREAD_TTL) && !self.hot.iter().any(|h| h.snap.tokens().len() == e.n && h.conv == e.conv))
            .collect();
        c.sort_by(|a, b| b.1.score().total_cmp(&a.1.score()));
        let mut out = vec![];
        let mut total = 0;
        for (k, e) in c {
            if out.len() >= self.cfg.preread_files || total + e.bytes > self.cfg.preread_bytes { break; }
            total += e.bytes;
            out.push((self.dir.join(k), e.bytes));
        }
        out
    }
    /// Hand a plan to the pre-read thread (started on first use).
    pub fn start_preread(&mut self, plan: Vec<(PathBuf, u64)>) {
        if plan.is_empty() { return; }
        if self.preread.is_none() { self.preread = spawn_preread(self.shared.clone()); }
        let Some(p) = &self.preread else { return };
        if let Ok(mut s) = p.stats.lock() { s.running = true; }
        let _ = p.tx.send(plan);
    }
    /// Pre-read results so far into the counters / paged set.
    pub fn poll_preread(&mut self) {
        let Some(p) = &self.preread else { return };
        let Ok(mut s) = p.stats.lock() else { return };
        let (files, bytes, cancelled, done) = (s.files, s.bytes, s.cancelled, std::mem::take(&mut s.done));
        s.files = 0; s.bytes = 0; s.cancelled = 0;
        drop(s);
        for c in [&mut self.c, &mut self.since] { c.preread_files += files; c.preread_bytes += bytes; c.preread_cancelled += cancelled; }
        for d in done { self.paged.insert(d, Instant::now()); }
    }
    /// Is the pre-read thread still reading (tests / health)?
    pub fn preread_running(&self) -> bool { self.preread.as_ref().and_then(|p| p.stats.lock().ok().map(|s| s.running)).unwrap_or(false) }

    /// Index file (atomic; only when something changed).
    pub fn save_index(&mut self) {
        if !self.dirty || !self.cfg.index { return; }
        let mut files = serde_json::Map::new();
        for (k, e) in &self.entries {
            let mut v = json!({"n": e.n, "bytes": e.bytes, "kind": e.kind.as_str(), "project": format!("{:016x}", e.project),
                "conv": format!("{:016x}", e.conv), "last_use": e.last_use, "created": e.created, "hits": e.hits});
            if !e.blocks.is_empty() { v["blocks"] = json!(e.blocks.iter().map(|b| format!("{:016x}", b)).collect::<Vec<_>>()); }
            if let Some(g) = e.tag { v["pid"] = json!(format!("{:016x}", g.project)); v["lane"] = json!(g.lane()); }
            files.insert(k.clone(), v);
        }
        let p = self.dir.join(INDEX_FILE);
        let tmp = self.dir.join(format!("{}.tmp.{}", INDEX_FILE, std::process::id()));
        if std::fs::write(&tmp, json!({"version": 1, "files": files}).to_string()).and_then(|_| std::fs::rename(&tmp, &p)).is_ok() { self.dirty = false; }
        else { let _ = std::fs::remove_file(&tmp); }
    }

    /// Status for /health (published; the HTTP threads read the copy).
    pub fn publish(&mut self) {
        self.poll_preread();
        let (n, bytes) = self.usage();
        let now = now_unix();
        let mut top: Vec<&Entry> = self.entries.values().collect();
        top.sort_by(|a, b| b.score().total_cmp(&a.score()));
        let top: Vec<Value> = top.iter().take(8).map(|e| json!({"n": e.n, "mb": (e.bytes as f64 / 1e5).round() / 10.0, "kind": e.kind.as_str(),
            "conv": format!("{:08x}", e.conv >> 32), "hits": e.hits, "age_s": (now - e.last_use).max(0.0).round()})).collect();
        // per-kind counts: warm files, hot copies, and every stable prefix (both of a project kept: agent + planner)
        let mut kinds = serde_json::Map::new();
        for e in self.entries.values() {
            let k = kinds.entry(e.kind.as_str()).or_insert(json!(0));
            *k = json!(k.as_u64().unwrap_or(0) + 1);
        }
        let hot_stable = self.hot.iter().filter(|h| h.stable).count();
        let mut stables: Vec<&Entry> = self.entries.values().filter(|e| e.kind == Kind::Stable).collect();
        stables.sort_by(|a, b| b.last_use.total_cmp(&a.last_use));
        let stables: Vec<Value> = stables.iter().map(|e| json!({"key": format!("{:08x}", e.conv >> 32), "n": e.n, "mb": (e.bytes as f64 / 1e5).round() / 10.0,
            "hits": e.hits, "age_s": (now - e.last_use).max(0.0).round(), "hot": self.hot_has_stable(e.conv),
            "project": e.tag.map(|g| format!("{:016x}", g.project)), "lane": e.tag.map(|g| g.lane())})).collect();
        let c = &self.c;
        let gb = |b: u64| (b as f64 / 1e7).round() / 100.0;
        let v = json!({
            "dir": self.dir.display().to_string(),
            "warm": {"files": n, "gb": gb(bytes), "max_files": self.cfg.max_files, "max_gb": gb(self.cfg.max_bytes), "kinds": kinds},
            "hot": {"copies": self.hot.len(), "kinds": {"stable": hot_stable, "conv": self.hot.len() - hot_stable}, "gb": gb(self.hot_bytes()), "max_gb": gb(self.cfg.hot_bytes), "headroom_gb": gb(self.cfg.hot_headroom), "ram_cap_gb": gb(self.cfg.ram_cap)},
            "disk": {"free_gb": self.free().map(gb), "min_free_gb": gb(self.cfg.min_free), "evict_for_save": self.cfg.evict_for_save, "stable_ensure": self.cfg.stable_ensure},
            "hits": {"hot": c.hit_hot, "warm": c.hit_warm, "miss": c.miss},
            "saves": c.saves, "switch_saves": c.switch_saves, "superseded": c.superseded, "stable": stables, "saves_skipped_disk": c.saves_skipped_disk, "evicted": c.evicted, "evicted_gb": gb(c.evicted_bytes),
            "hot_admitted": c.hot_admitted, "hot_refused": c.hot_refused, "hot_dropped": c.hot_dropped,
            "idle": {"on": self.cfg.idle.is_some(), "idle_s": self.cfg.idle.map(|d| d.as_secs_f64()), "saves": c.idle_saves, "save_ms": c.idle_save_ms.round(),
                     "preread_files": c.preread_files, "preread_gb": gb(c.preread_bytes), "preread_cancelled": c.preread_cancelled,
                     "blocked_ms": c.blocked_ms.round(), "pending_steps": self.steps.len()},
            "live": self.live.map(|(conv, n)| json!({"conv": format!("{:08x}", conv >> 32), "n": n, "saved_n": self.conv_newest(conv)})),
            "top": top,
        });
        if let Ok(mut s) = self.shared.status.lock() { *s = v; }
    }
}

impl Drop for Tiers {
    fn drop(&mut self) { while !self.hot.is_empty() { self.drop_hot(0); } }
}

/// A file eviction first may delete (`room_plan`).
#[derive(Clone, Debug)]
pub struct RoomCand { pub name: String, pub stable: bool, pub score: f64, pub bytes: u64, pub age_s: f64 }

/// Eviction-first order: which files to delete so that `free` + their bytes >= `need`. Non-stable files first, then
/// stable prefixes unused for `STABLE_PROTECT_S`, then the other stable prefixes; lowest score (oldest, fewest hits)
/// first within each group; stop as soon as the room is there. None (delete nothing) when all of them together are
/// not enough. Some(empty) when `free` >= `need` already.
pub fn room_plan(cands: &[RoomCand], free: u64, need: u64) -> Option<Vec<String>> {
    if free >= need { return Some(vec![]); }
    let mut order: Vec<&RoomCand> = cands.iter().collect();
    let group = |c: &RoomCand| if !c.stable { 0 } else if c.age_s >= STABLE_PROTECT_S { 1 } else { 2 };
    order.sort_by(|a, b| group(a).cmp(&group(b)).then(a.score.total_cmp(&b.score)));
    let mut f = free;
    let mut out = vec![];
    for c in order {
        if f >= need { break; }
        f += c.bytes;
        out.push(c.name.clone());
    }
    if f >= need { Some(out) } else { None }
}

pub fn file_name(p: &Path) -> String { p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default() }

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_snaptier_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn file(d: &Path, name: &str, bytes: usize) -> PathBuf { let p = d.join(format!("{}.sdsnap", name)); std::fs::write(&p, vec![0u8; bytes]).unwrap(); p }
    fn toks(sys: u32, user: u32, n: usize) -> Vec<u32> {
        let mut t = vec![sys; 70]; t.push(IM_END); t.extend(std::iter::repeat_n(user, 10)); t.push(IM_END);
        while t.len() < n { t.push(7); }
        t
    }

    #[test]
    fn keys_split_project_and_conv() {
        let (p1, c1) = session_keys(&toks(1, 2, 100));
        let (p2, c2) = session_keys(&toks(1, 3, 100));
        let (p3, _) = session_keys(&toks(4, 2, 100));
        assert_eq!(p1, p2);
        assert_ne!(c1, c2);
        assert_ne!(p1, p3);
        // a stable prefix (one <|im_end|>): conv = project
        let s = &toks(1, 2, 100)[..71];
        assert_eq!(session_keys(s), (p1, p1));
        assert_eq!(session_keys(&toks(1, 2, 100)[..90]).1, c1);
    }

    #[test]
    fn per_conv_rolling_and_caps() {
        let d = tmp("caps");
        let mut cfg = TierConfig::legacy(4, 1 << 30);
        cfg.per_conv = true;
        cfg.index = true;
        let mut t = Tiers::open(&d, cfg);
        let a = toks(1, 2, 300);
        let b = toks(1, 3, 300);
        let st = file(&d, "stable", 100);
        t.saved(&st, &a[..71], Kind::Stable);
        for (i, n) in [(0, 200), (1, 250), (2, 300)] {
            let p = file(&d, &format!("a{}", i), 100);
            t.saved(&p, &a[..n], if i == 2 { Kind::Idle } else { Kind::Roll });
            std::thread::sleep(Duration::from_millis(5));
        }
        // conv A keeps its newest 2 non-stable files; the stable prefix stays
        assert!(!d.join("a0.sdsnap").exists() && d.join("a1.sdsnap").exists() && d.join("a2.sdsnap").exists() && st.exists());
        let pb = file(&d, "b0", 100);
        t.saved(&pb, &b[..280], Kind::Roll);
        assert_eq!(t.usage().0, 4);
        // a 5th file: the lowest score goes (a1: oldest, no hits); the stable prefix has the stable credit
        t.hit_warm(&d.join("a2.sdsnap"));
        let pb2 = file(&d, "c0", 100);
        t.saved(&pb2, &toks(5, 6, 200), Kind::Roll);
        assert_eq!(t.usage().0, 4);
        assert!(!d.join("a1.sdsnap").exists(), "lowest score evicted");
        assert!(st.exists() && pb2.exists());
        // byte cap: 250 B max -> down to 2 files, the newest kept
        t.cfg.max_bytes = 250;
        let pz = file(&d, "z", 100);
        t.saved(&pz, &toks(8, 9, 120), Kind::Roll);
        assert!(t.usage().1 <= 250 && pz.exists(), "{:?}", t.usage());
        // index round trip: hits / kinds survive a reopen
        t.save_index();
        let t2 = Tiers::open(&d, TierConfig { index: true, ..TierConfig::legacy(4, 1 << 30) });
        assert_eq!(t2.entries.len(), t.entries.len());
        for (k, e) in &t.entries { assert_eq!((t2.entries[k].kind, t2.entries[k].hits, t2.entries[k].conv), (e.kind, e.hits, e.conv)); }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn legacy_one_roll_per_gateway() {
        let d = tmp("legacy");
        let mut t = Tiers::open(&d, TierConfig::legacy(4, 1 << 30));
        let r1 = file(&d, "r1", 10);
        t.saved(&r1, &toks(1, 2, 100), Kind::Roll);
        let r2 = file(&d, "r2", 10);
        t.saved(&r2, &toks(1, 3, 100), Kind::Roll);
        assert!(!r1.exists() && r2.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn disk_floor() {
        let d = tmp("disk");
        let mut cfg = TierConfig::legacy(32, 1 << 40);
        let free = disk_free(&d).unwrap();
        // a floor above the free space: no save, cold files evicted trying to make room
        cfg.min_free = free + (1 << 30);
        let mut t = Tiers::open(&d, cfg);
        let old = file(&d, "old", 10);
        t.sync_dir();
        assert!(t.disk_ok(0).is_err());
        assert!(!old.exists(), "cold file deleted under the disk floor");
        assert_eq!(t.c.saves_skipped_disk, 1);
        t.cfg.min_free = 1;
        assert!(t.disk_ok(1 << 20).is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn idle_cycle_and_cancel() {
        let d = tmp("idle");
        let mut cfg = TierConfig::legacy(8, 1 << 30);
        cfg.idle = Some(Duration::from_millis(30));
        cfg.preread = true; cfg.preread_files = 2; cfg.preread_bytes = 1 << 30;
        let mut t = Tiers::open(&d, cfg);
        let big = file(&d, "big", 3 * PREREAD_CHUNK + 5);
        t.saved(&big, &toks(1, 2, 100), Kind::Roll);
        t.paged.clear();
        // job ends -> idle due after 30 ms; a pending request stops everything
        t.job_end(Some(&toks(4, 5, 90)));
        assert!(t.idle_wait().unwrap() <= Duration::from_millis(30));
        assert!(t.next_step().is_none(), "not idle long enough");
        std::thread::sleep(Duration::from_millis(40));
        t.shared.arrived();
        assert!(t.next_step().is_none(), "a queued request blocks idle work");
        let _ = t.job_begin(Instant::now());
        t.job_end(Some(&toks(4, 5, 90)));
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(t.next_step(), Some(Step::Save));
        assert_eq!(t.next_step(), Some(Step::Hot));
        assert_eq!(t.next_step(), Some(Step::HotStable));
        assert_eq!(t.next_step(), Some(Step::Preread));
        let plan = t.preread_plan();
        assert_eq!(plan.len(), 1, "the other conv's file");
        t.start_preread(plan);
        let t0 = Instant::now();
        while t.preread_running() && t0.elapsed() < Duration::from_secs(10) { std::thread::sleep(Duration::from_millis(5)); }
        let v = t.job_begin(Instant::now());
        assert_eq!((v["preread_files"].as_u64(), v["preread_cancelled"].as_u64()), (Some(1), Some(0)), "{}", v);
        assert!(t.preread_plan().is_empty(), "paged: not again within the TTL");
        // cancellation: a request arriving while the thread reads stops it before the next 4 MB chunk
        t.paged.clear();
        t.job_end(None);
        t.shared.busy.store(true, Ordering::SeqCst); // a job is running: the thread stops at its first check
        t.start_preread(t.preread_plan());
        let t0 = Instant::now();
        while t.preread_running() && t0.elapsed() < Duration::from_secs(10) { std::thread::sleep(Duration::from_millis(5)); }
        let v = t.job_begin(Instant::now());
        assert_eq!((v["preread_files"].as_u64(), v["preread_cancelled"].as_u64()), (Some(0), Some(1)), "{}", v);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn snap(tokens: &[u32], body: usize) -> Snapshot {
        use streamdense_model::snapshot::{SnapHeader, FORMAT};
        let header = SnapHeader { format: FORMAT, engine: "test".into(), fingerprint: 0, kv: streamdense_model::KvFormat::Bf16, pos: tokens.len(),
                                  mtp_len: 0, mtp: false, body: body as u64, tokens: tokens.to_vec(), compact: None };
        Snapshot { header, body: vec![0u8; body] }
    }
    fn hot_cfg(max_files: usize, hot_bytes: u64) -> TierConfig {
        let mut cfg = TierConfig::legacy(max_files, 1 << 40);
        cfg.per_conv = true; cfg.index = true; cfg.hot_bytes = hot_bytes; cfg.ram_cap = u64::MAX / 4;
        cfg
    }
    /// A system turn: `head` shared tokens (value 3), then `own` tokens of value `v` (no <|im_end|>: a 12 GB-mode cut).
    fn sys_cut(head: usize, v: u32, own: usize) -> Vec<u32> { let mut t = vec![3; head]; t.extend(std::iter::repeat_n(v, own)); t }

    #[test]
    fn two_stable_prefixes_of_a_project_keyed_apart_and_kept() {
        let d = tmp("two_stable");
        let mut t = Tiers::open(&d, hot_cfg(4, 1 << 20));
        // agent + planner system turns, cut before <|im_end|>: session_keys collide (first 64 tokens), stable keys do not
        let (agent, planner) = (sys_cut(64, 1, 40), sys_cut(64, 2, 40));
        assert_eq!(session_keys(&agent).0, session_keys(&planner).0, "the old project key collides");
        let (pa, pp) = (file(&d, "agent", 100), file(&d, "planner", 100));
        t.saved(&pa, &agent, Kind::Stable);
        t.saved(&pp, &planner, Kind::Stable);
        assert_ne!(t.entries["agent.sdsnap"].conv, t.entries["planner.sdsnap"].conv);
        assert!(pa.exists() && pp.exists(), "the planner turn does not supersede the agent turn");
        assert_eq!(t.stable_prefix_of(&[agent.clone(), vec![9; 20]].concat()).map(|x| x.0), Some("agent.sdsnap".into()));
        assert_eq!(t.stable_prefix_of(&[planner.clone(), vec![9; 20]].concat()).map(|x| x.0), Some("planner.sdsnap".into()));
        // rolling files past the file cap (even with hits): the rolls go, never a stable prefix
        for i in 0..5u32 {
            let p = file(&d, &format!("r{}", i), 100);
            t.saved(&p, &toks(10 + i, 20 + i, 120), Kind::Roll);
            t.hit_warm(&p);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(t.usage().0, 4);
        assert!(pa.exists() && pp.exists(), "both stable prefixes survive the disk policy");
        // hot tier: both stable copies fit -> both kept; a conversation copy never drops them (refused instead)
        t.hot_add_kind(snap(&agent, 300), 300, true);
        t.hot_add_kind(snap(&planner, 300), 300, true);
        assert_eq!(t.hot.len(), 2, "keyed apart: the second copy does not replace the first");
        let conv = toks(1, 2, 200);
        assert!(t.hot_room(1 << 20).is_err(), "a conv copy that needs a stable copy's room is refused");
        assert_eq!(t.hot.iter().filter(|h| h.stable).count(), 2);
        assert!(t.hot_room(1000).is_ok());
        t.hot_add(snap(&conv, 1000), 1000);
        // budget pressure from another stable copy: the conversation copy (LRU, non-stable) goes first
        let other = sys_cut(0, 5, 100);
        assert!(t.hot_room_for(t.cfg.hot_bytes - 600, true).is_ok());
        t.hot_add_kind(snap(&other, 10), t.cfg.hot_bytes - 600, true);
        assert!(t.hot_has_stable(stable_key(&agent)) && t.hot_has_stable(stable_key(&planner)), "both stable copies kept");
        assert!(!t.hot.iter().any(|h| !h.stable), "the conv copy was dropped first");
        // /health: per-kind counts
        t.publish();
        let h = t.shared.health();
        assert_eq!(h["warm"]["kinds"]["stable"].as_u64(), Some(2), "{}", h);
        assert_eq!(h["hot"]["kinds"]["stable"].as_u64(), Some(3), "{}", h);
        assert_eq!(h["stable"].as_array().map(|a| a.len()), Some(2));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn newer_planner_turn_supersedes_the_older() {
        let d = tmp("supersede");
        let mut t = Tiers::open(&d, hot_cfg(16, 1 << 20));
        // planner instructions (256 tokens) + project facts (64) + <|im_end|>; the facts change between v1 and v2
        let planner = |facts: u32| { let mut v = vec![5; 256]; v.extend(std::iter::repeat_n(facts, 64)); v.push(IM_END); v };
        let mut agent = vec![8; 300]; agent.push(IM_END);
        let pa = file(&d, "agent", 100);
        t.saved(&pa, &agent, Kind::Stable);
        let p1 = file(&d, "plan1", 100);
        t.saved(&p1, &planner(6), Kind::Stable);
        t.hot_add_kind(snap(&planner(6), 100), 100, true);
        t.hot_add_kind(snap(&agent, 100), 100, true);
        std::thread::sleep(Duration::from_millis(2));
        let p2 = file(&d, "plan2", 100);
        let gone = t.saved(&p2, &planner(7), Kind::Stable);
        assert_eq!(gone, vec!["plan1.sdsnap".to_string()]);
        assert!(!p1.exists() && p2.exists() && pa.exists(), "stale planner file dropped, agent prefix kept");
        assert_eq!(t.c.superseded, 1);
        assert_eq!(t.c.evicted, 0, "a supersede is not an eviction");
        assert!(!t.hot_has_stable(stable_key(&planner(6))), "the stale planner hot copy is dropped too");
        assert!(t.hot_has_stable(stable_key(&agent)));
        assert_eq!(t.entries.values().filter(|e| e.kind == Kind::Stable).count(), 2, "the cache does not grow");
        // the fingerprint survives a reopen (index), so a later v3 still finds v2 after a restart
        t.save_index();
        let mut t2 = Tiers::open(&d, hot_cfg(16, 1 << 20));
        let p3 = file(&d, "plan3", 100);
        t2.saved(&p3, &planner(4), Kind::Stable);
        assert!(!p2.exists() && p3.exists() && pa.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn project_header_decides_the_supersede_family() {
        let d = tmp("tagged");
        let mut t = Tiers::open(&d, hot_cfg(16, 1 << 20));
        let (pa, pb) = (0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888u64);
        let tag = |project, plan| Some(StableTag { project, plan, stat: false });
        // planner v1 / v2 share only 1 of 5 blocks (far below the 75% fallback): the tag still supersedes
        let planner = |facts: u32| { let mut v = vec![5; 64]; v.extend(std::iter::repeat_n(facts, 256)); v.push(IM_END); v };
        let mut agent = vec![8; 300]; agent.push(IM_END);
        let pg = file(&d, "agent", 100);
        t.saved_tagged(&pg, &agent, Kind::Stable, tag(pa, false));
        let p1 = file(&d, "plan1", 100);
        t.saved_tagged(&p1, &planner(6), Kind::Stable, tag(pa, true));
        t.hot_add_kind(snap(&planner(6), 100), 100, true);
        assert_eq!(t.hot[0].tag, tag(pa, true), "a hot copy carries its entry's tag");
        let p2 = file(&d, "plan2", 100);
        assert_eq!(t.saved_tagged(&p2, &planner(7), Kind::Stable, tag(pa, true)), vec!["plan1.sdsnap".to_string()]);
        assert!(!p1.exists() && p2.exists() && pg.exists(), "same project + plan lane: the older planner turn goes; the agent turn stays");
        assert!(!t.hot_has_stable(stable_key(&planner(6))), "its hot copy too");
        // another project with an almost identical planner turn (shares >= 75%): kept apart by the header
        let mut near = planner(7); near[300] = 9;
        assert!(same_family(&fingerprint(&near), near.len(), &fingerprint(&planner(7)), planner(7).len()));
        let pn = file(&d, "planB", 100);
        assert!(t.saved_tagged(&pn, &near, Kind::Stable, tag(pb, true)).is_empty());
        assert!(p2.exists() && pn.exists(), "different project: never superseded, overlap or not");
        // same project, other lane, overlapping tokens: the agent lane never replaces the planner lane
        let pg2 = file(&d, "agent2", 100);
        let mut agent2 = planner(7); agent2[310] = 2;
        assert!(t.saved_tagged(&pg2, &agent2, Kind::Stable, tag(pa, false)).iter().all(|g| g == "agent.sdsnap"));
        assert!(p2.exists() && !pg.exists() && pg2.exists(), "agent v2 replaced agent v1 only (same project + lane)");
        // no header: never replaces a tagged prefix; the 75% overlap fallback among untagged ones
        let pu = file(&d, "untagged", 100);
        let mut un = planner(7); un[305] = 1;
        assert!(t.saved(&pu, &un, Kind::Stable).is_empty(), "an untagged prefix does not claim a project's prefix");
        assert!(p2.exists() && pn.exists() && pg2.exists());
        let pu2 = file(&d, "untagged2", 100);
        let mut un2 = un.clone(); un2[306] = 1;
        assert_eq!(t.saved(&pu2, &un2, Kind::Stable), vec!["untagged.sdsnap".to_string()], "fallback between untagged prefixes");
        // a tagged prefix migrates an untagged (older index) one of its family by overlap
        let p3 = file(&d, "plan3", 100);
        let mut v3 = un2.clone(); v3[307] = 1;
        let gone = t.saved_tagged(&p3, &v3, Kind::Stable, tag(pa, true));
        assert!(gone.contains(&"plan2.sdsnap".to_string()) && gone.contains(&"untagged2.sdsnap".to_string()), "{:?}", gone);
        assert!(pn.exists() && pg2.exists(), "other project / other lane kept");
        // the tag survives the index (a restart keeps the family)
        t.save_index();
        let mut t2 = Tiers::open(&d, hot_cfg(16, 1 << 20));
        assert_eq!(t2.entries["planB.sdsnap"].tag, tag(pb, true));
        assert_eq!(t2.entries["agent2.sdsnap"].tag, tag(pa, false));
        assert_eq!(t2.entries["plan3.sdsnap"].tag, tag(pa, true));
        let h = { t2.publish(); t2.shared.health() };
        assert!(h["stable"].as_array().unwrap().iter().any(|e| e["project"] == "5555666677778888" && e["lane"] == "plan"), "{}", h);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn static_planner_cut_and_facts_cut_coexist() {
        let d = tmp("static");
        let mut t = Tiers::open(&d, hot_cfg(16, 1 << 20));
        let (pa, pb) = (0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888u64);
        let facts_tag = |project| Some(StableTag { project, plan: true, stat: false });
        // static part (512 tokens) + facts (64) + <|im_end|>: the static cut is a token prefix of each facts cut
        let stat: Vec<u32> = vec![5; 512];
        let planner = |facts: u32| { let mut v = stat.clone(); v.extend(std::iter::repeat_n(facts, 64)); v.push(IM_END); v };
        let ps = file(&d, "static", 100);
        assert!(t.saved_tagged(&ps, &stat, Kind::Stable, Some(StableTag::plan_static())).is_empty());
        assert_eq!(t.entries["static.sdsnap"].project, 0, "the static cut belongs to no project");
        t.hot_add_kind(snap(&stat, 100), 100, true);
        let f1 = file(&d, "facts1", 100);
        assert!(t.saved_tagged(&f1, &planner(6), Kind::Stable, facts_tag(pa)).is_empty(), "facts cut does not replace the static cut");
        t.hot_add_kind(snap(&planner(6), 100), 100, true);
        assert!(ps.exists() && f1.exists());
        assert!(t.hot_has_stable(stable_key(&stat)) && t.hot_has_stable(stable_key(&planner(6))), "both hot copies kept");
        // newer facts of the same project: only the older facts cut goes
        let f2 = file(&d, "facts2", 100);
        assert_eq!(t.saved_tagged(&f2, &planner(7), Kind::Stable, facts_tag(pa)), vec!["facts1.sdsnap".to_string()]);
        assert!(ps.exists() && !f1.exists() && f2.exists());
        assert!(t.hot_has_stable(stable_key(&stat)) && !t.hot_has_stable(stable_key(&planner(6))));
        // another project: its facts cut sits next to the one shared static cut
        let fb = file(&d, "factsB", 100);
        assert!(t.saved_tagged(&fb, &planner(8), Kind::Stable, facts_tag(pb)).is_empty());
        // an untagged facts cut (no header) never replaces the static one, and vice versa by overlap
        let fu = file(&d, "factsU", 100);
        assert!(t.saved(&fu, &planner(9), Kind::Stable).is_empty());
        // a newer static part (new instructions) replaces only the older static cut
        let mut stat2 = stat.clone(); stat2[500] = 1;
        let ps2 = file(&d, "static2", 100);
        assert_eq!(t.saved_tagged(&ps2, &stat2, Kind::Stable, Some(StableTag::plan_static())), vec!["static.sdsnap".to_string()]);
        assert!(!ps.exists() && ps2.exists() && f2.exists() && fb.exists() && fu.exists());
        assert_eq!(t.entries.values().filter(|e| e.tag.is_some_and(|g| g.stat)).count(), 1, "one static copy");
        // the lane survives the index
        t.save_index();
        let t2 = Tiers::open(&d, hot_cfg(16, 1 << 20));
        assert_eq!(t2.entries["static2.sdsnap"].tag, Some(StableTag::plan_static()));
        assert_eq!(t2.entries["facts2.sdsnap"].tag, facts_tag(pa));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn static_prefix_len_finds_the_marker() {
        use crate::worker::static_prefix_len;
        // 10 = "\n\n", 11 = "\n", 12 = " " (whitespace), 20 21 = "PROJECT FACTS"
        let ws = |t: u32| match t { 10 => Some(2), 11 => Some(1), 12 => Some(0), _ => None };
        let words = [20, 21];
        let mut p = vec![7; 100]; p.extend([10, 20, 21, 30, 31, IM_END, 40]);
        assert_eq!(static_prefix_len(&p, &words, ws), Some(100), "cut before the newline run");
        let mut p = vec![7; 100]; p.extend([11, 12, 11, 20, 21, IM_END]);
        assert_eq!(static_prefix_len(&p, &words, ws), Some(100), "split newline run");
        let mut p = vec![7; 100]; p.extend([11, 20, 21, IM_END]);
        assert_eq!(static_prefix_len(&p, &words, ws), None, "one newline: not the marker");
        let mut p = vec![7; 100]; p.extend([IM_END, 10, 20, 21]);
        assert_eq!(static_prefix_len(&p, &words, ws), None, "after the system turn");
        let mut p = vec![7; 10]; p.extend([10, 20, 21, IM_END]);
        assert_eq!(static_prefix_len(&p, &words, ws), None, "shorter than STABLE_MIN");
    }

    #[test]
    fn project_header_parsing() {
        use crate::anthropic::parse_project;
        assert_eq!(parse_project(" 00ff00ff00ff00ff "), Some(0x00ff00ff00ff00ff));
        assert_eq!(parse_project("00ff00ff00ff00f"), None);
        assert_eq!(parse_project("00ff00ff00ff00fg"), None);
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "streamdense": {"format": "plan"}});
        let mut r = crate::anthropic::convert(&body).unwrap();
        assert_eq!(r.stable_tag(), None, "no project header: no tag");
        r.project = Some(7);
        assert_eq!(r.stable_tag(), Some(StableTag { project: 7, plan: true, stat: false }), "format plan -> plan lane");
        r.plan = false;
        assert_eq!(r.stable_tag().map(|g| g.plan), Some(false));
        r.lane_plan = true;
        assert_eq!(r.stable_tag().map(|g| g.plan), Some(true), "x-streamdense-lane: plan");
    }

    #[test]
    fn warmup_stable_save_goes_hot() {
        let d = tmp("warm_hot");
        let mut t = Tiers::open(&d, hot_cfg(8, 4000));
        let mut sys = vec![5; 100]; sys.push(IM_END);
        let p = d.join("plan.sdsnap");
        snap(&sys, 1000).write(&p).unwrap();
        t.saved(&p, &sys, Kind::Stable);
        // the hot budget is full of a conversation copy: a warm-up's stable save may drop it (LRU, non-stable)
        let conv = toks(1, 2, 200);
        t.hot_add(snap(&conv, 3500), 3500);
        let bytes = std::fs::metadata(&p).unwrap().len();
        assert_eq!(t.hot_stable_file(&p, false), Ok(false), "idle step 2b never drops another copy");
        assert_eq!(t.hot_stable_file(&p, true), Ok(true));
        assert!(t.hot_has_stable(stable_key(&sys)));
        assert!(!t.hot.iter().any(|h| !h.stable), "the conv copy made room");
        assert_eq!(t.hot.iter().map(|h| h.bytes).sum::<u64>(), bytes);
        // the next plan call (same system turn + a new user turn) restores it from RAM
        let mut call = sys.clone(); call.extend([4, 4, 4, IM_END, 7]);
        assert_eq!(t.hot_best(&call).map(|x| x.1), Some(sys.len()));
        assert_eq!(t.hot_stable_file(&p, true), Ok(false), "already hot: no second copy");
        // the live state's stable prefix (idle step 2b's source) is found exactly
        t.job_end(Some(&call));
        assert_eq!(t.live_stable_file().map(|x| x.1), Some(sys.len()));
        let _ = std::fs::remove_dir_all(&d);
    }

    fn cand(name: &str, stable: bool, score: f64, gb: u64, age_s: f64) -> RoomCand { RoomCand { name: name.into(), stable, score, bytes: gb, age_s } }

    #[test]
    fn room_plan_order_and_refusal() {
        // fake disk in units of 10 MB: free 1277, floor 1200 + est 105 -> need 1305 (the logged shape: 12.77 GB free <
        // 12.00 + 1.05 GB)
        let c = vec![cand("stable_old", true, 10.0, 2, 2e5), cand("stable_new", true, 5.0, 2, 10.0),
                     cand("roll_b", false, 300.0, 14, 50.0), cand("roll_a", false, 100.0, 3, 900.0), cand("idle_c", false, 200.0, 20, 60.0)];
        // lowest score non-stable first, stop once the room is there (28 short: 3 + 20 is not enough, + 14 is)
        assert_eq!(room_plan(&c, 1277, 1305), Some(vec!["roll_a".to_string(), "idle_c".to_string(), "roll_b".to_string()]));
        assert_eq!(room_plan(&c, 1277, 1300), Some(vec!["roll_a".to_string(), "idle_c".to_string()]));
        // already enough: nothing
        assert_eq!(room_plan(&c, 1400, 1305), Some(vec![]));
        // every non-stable file is not enough: the stale stable prefix (unused > STABLE_PROTECT_S) goes before a fresh one
        assert_eq!(room_plan(&c, 1260, 1298), Some(vec!["roll_a".into(), "idle_c".into(), "roll_b".into(), "stable_old".into()]));
        // all of them cannot make the room: None (nothing is deleted for nothing)
        assert_eq!(room_plan(&c, 100, 1305), None);
        assert_eq!(room_plan(&[], 1277, 1305), None);
    }

    #[test]
    fn eviction_first_fake_disk() {
        let d = tmp("evict_first");
        let mut cfg = hot_cfg(32, 1 << 40);
        cfg.min_free = 12_000; cfg.evict_for_save = true; cfg.stable_ensure = true; cfg.evict_for_save_min = 100;
        let mut t = Tiers::open(&d, cfg);
        let ff = Arc::new(AtomicU64::new(20_000));
        t.fake_free = Some(ff.clone());
        let set = |v: u64| ff.store(v, Ordering::SeqCst);
        let get = || ff.load(Ordering::SeqCst);
        let main = toks(1, 2, 400);
        let sys = &main[..71];
        // the main conversation's stable prefix + an old file of another conversation + an old file of a third one
        let ps = file(&d, "main_stable", 300);
        t.saved_tagged(&ps, sys, Kind::Stable, None);
        assert_eq!(t.live_stable.as_deref(), Some("main_stable.sdsnap"), "stable_ensure: the new prefix is the lineage's");
        let po = file(&d, "other", 1500);
        t.saved(&po, &toks(4, 5, 300), Kind::Roll);
        std::thread::sleep(Duration::from_millis(5));
        let px = file(&d, "third", 1600);
        t.saved(&px, &toks(6, 7, 320), Kind::Roll);
        let pm = file(&d, "main_roll", 1000);
        t.saved(&pm, &main[..250], Kind::Roll);
        assert_eq!(get(), 20_000 - 300 - 1500 - 1600 - 1000);
        t.job_end(Some(&main));
        assert_eq!(t.live_stable.as_deref(), Some("main_stable.sdsnap"));
        // size estimate by tokens: 400 tokens -> the largest known (320 tokens, 1600 B) scaled; 71 tokens -> the stable file
        assert_eq!(t.est_bytes_for(400), 2000);
        assert_eq!(t.est_bytes_for(71), 300);
        assert_eq!(t.est_bytes_for(260), 1500);
        assert_eq!(t.est_bytes(), 1600, "the old estimate: the largest file");
        // the floor blocks a 2000 B save (15600 free >= 12000 + 2000 holds) -> lower the fake free under the floor
        set(12_500);
        // disk_ok (the former rule) skips; eviction first deletes the lowest-score unprotected files and saves
        let gone = t.make_room(2000, 400, "switch", &[]).unwrap();
        assert_eq!(gone, vec!["other.sdsnap".to_string()], "oldest first; enough after one");
        assert!(!po.exists() && px.exists() && ps.exists() && pm.exists(), "live stable prefix + live conv newest kept");
        assert_eq!(get(), 14_000);
        // not enough even after deleting every unprotected file: nothing deleted, counted skipped
        set(5_000);
        let skipped = t.c.saves_skipped_disk;
        let e = t.make_room(2000, 400, "idle", &[]).unwrap_err();
        assert!(e.contains("cannot make room") && px.exists(), "{}", e);
        assert_eq!(t.c.saves_skipped_disk, skipped + 1);
        // the disk policy (after a save) never deletes the protected files either
        set(11_000);
        let gone = t.evict(None);
        assert!(gone.contains(&"third.sdsnap".to_string()) && ps.exists() && pm.exists(), "{:?}", gone);
        // without stable_ensure / evict_for_save nothing is protected (the former policy)
        t.cfg.stable_ensure = false; t.cfg.evict_for_save = false;
        assert!(t.protected().is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }
}
