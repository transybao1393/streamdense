//! `CacheHook`: the gateway's snapshot policy on top of streamdense-model's `SnapshotCache` (files in
//! ~/Library/Caches/streamdense/snapshots or $SD_SNAPSHOT_DIR, or `--snapshot-dir`) and the tier manager
//! (`snap_tier::Tiers`: hot RAM copies / warm files / cold = deleted, idle-time work; see snap_tier.rs).
//! - restore: the longest cached prefix of the prompt, hot copy first when it covers at least as much as the best file
//!   (the file's mtime is touched + its hit count / last use in the tier index).
//! - save stable: kept (the system prompt + tool defs prefix shared by every Claude Code session).
//! - save rolling: `CacheHook::new` (legacy): one per gateway, the previous one deleted after the new one is written;
//!   `CacheHook::with_tiers` (the gateway): the newest 2 per conversation (request-time + idle save).
//! - cap: after each save the tier policy deletes the lowest-score files until <= max_files, <= max_bytes and free disk
//!   >= min_free (the file just written is never deleted); no save while the disk is below min_free.
//! - idle (with_tiers): save the live session, hot copy of it (+ of its shared prefix), pre-read the likely next files.
//! - switch (with_tiers, multi-user): save the live state before another conversation's request replaces it.
//! - restore events carry `src`: "own" (a file of the request's conversation) | "shared" (the project prefix).
//! - 12 GB mode: the same policy over `compact::CompactBackend` (`SnapTarget`): format-2 files (state + prompt sums)
//!   under their own keys (`SnapshotCache.mode` = `CompactBackend::snap_mode`; a mismatch saves / restores nothing); no
//!   hot tier (`hot_capable` false: no spare RAM in that mode).
use crate::compact::CompactBackend;
use crate::snap_tier::{Kind, StableTag, Step, TierConfig, TierShared, Tiers};
use crate::worker::{SnapEvent, SnapshotHook, STABLE_MIN};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use streamdense_model::snapshot::{Snapshot, SnapshotCache};
use streamdense_model::QwenEngine;

pub struct CacheHook {
    pub cache: SnapshotCache,
    pub max_files: usize,
    pub max_bytes: u64,
    pub tiers: Tiers,
}

impl CacheHook {
    /// The pre-tier policy (LRU caps, one rolling file per gateway, no idle work, no hot tier).
    pub fn new(cache: SnapshotCache, max_files: usize, max_bytes: u64) -> CacheHook {
        Self::with_tiers(cache, TierConfig::legacy(max_files, max_bytes))
    }

    /// The tier manager with `cfg` (`TierConfig::from_config(&[snapshots], ram cap)`).
    pub fn with_tiers(cache: SnapshotCache, cfg: TierConfig) -> CacheHook {
        let tiers = Tiers::open(&cache.dir, cfg);
        CacheHook { max_files: tiers.cfg.max_files, max_bytes: tiers.cfg.max_bytes, cache, tiers }
    }

    /// The model id for cache keys: canonical model dir + size and mtime of every file in it.
    pub fn model_id(model_dir: &str) -> String {
        let canon = std::fs::canonicalize(model_dir).map(|p| p.display().to_string()).unwrap_or_else(|_| model_dir.to_string());
        let mut parts = vec![canon.clone()];
        let mut walk = vec![PathBuf::from(&canon)];
        while let Some(d) = walk.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let mut es: Vec<_> = rd.flatten().collect();
            es.sort_by_key(|e| e.file_name());
            for e in es {
                let Ok(m) = e.metadata() else { continue };
                if m.is_dir() { walk.push(e.path()); continue; }
                let t = m.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                parts.push(format!("{}:{}:{}", e.path().display(), m.len(), t));
            }
        }
        parts.join("|")
    }

    /// (files, bytes) of *.sdsnap in the cache directory.
    pub fn usage(&self) -> (usize, u64) {
        let Ok(rd) = std::fs::read_dir(&self.cache.dir) else { return (0, 0) };
        rd.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".sdsnap")).filter_map(|e| e.metadata().ok())
            .fold((0, 0), |(n, b), m| (n + 1, b + m.len()))
    }

    /// Save under the cache key + tier bookkeeping (disk floor checked first).
    fn save_kind<B: SnapTarget + ?Sized>(&mut self, b: &B, tokens: &[u32], kind: Kind) -> Result<PathBuf, String> { self.save_tagged(b, tokens, kind, None) }
    /// `save_kind` with the request's stable tag (project + lane: the supersede family of a stable prefix).
    fn save_tagged<B: SnapTarget + ?Sized>(&mut self, b: &B, tokens: &[u32], kind: Kind, tag: Option<StableTag>) -> Result<PathBuf, String> {
        self.mode_ok(b)?;
        self.tiers.disk_ok(self.tiers.est_bytes())?;
        let p = self.cache.path_for(b.engine(), tokens);
        b.save_snap(&p, tokens)?;
        self.tiers.saved_tagged(&p, tokens, kind, tag);
        Ok(p)
    }

    /// Idle step 1: the live state (prompt + reply, `fed` = ids in the engine) to a file when it is far enough past
    /// its conversation's newest file. Some(ms) = saved.
    fn idle_save<B: SnapTarget + ?Sized>(&mut self, b: &B, fed: &[u32]) -> Option<f64> {
        if !self.tiers.cfg.idle_save || fed.len() < STABLE_MIN || self.mode_ok(b).is_err() { return None; }
        let (_, conv) = crate::snap_tier::session_keys(fed);
        if self.tiers.conv_newest(conv).is_some_and(|n| n + self.tiers.cfg.idle_save_min > fed.len()) { return None; }
        if self.cache.path_for(b.engine(), fed).exists() { return None; }
        let t0 = Instant::now();
        match self.save_kind(b, fed, Kind::Idle) {
            Ok(_) => Some(t0.elapsed().as_secs_f64() * 1e3),
            Err(e) => { eprintln!("gateway: snapshot tiers: idle save: {}", e); None }
        }
    }

    /// Idle step 2b: a hot copy of the live state's stable prefix file (the exact one: agent or planner system turn),
    /// only when it fits in `hot_gb` next to the copies already there (never drops another copy for it, but replaces
    /// an older copy of its family) and the RAM rule admits it.
    fn idle_hot_stable<B: SnapTarget + ?Sized>(&mut self, b: &B) -> bool {
        if !self.tiers.cfg.hot_stable || !b.hot_capable() || self.tiers.cfg.hot_bytes == 0 || self.mode_ok(b).is_err() { return false; }
        let Some((p, _)) = self.tiers.live_stable_file() else { return false };
        match self.tiers.hot_stable_file(&p, false) {
            Ok(ok) => ok,
            Err(e) => { eprintln!("gateway: snapshot tiers: {}", e); false }
        }
    }

    /// Idle step 2: a hot (RAM) copy of the live state's file, when the tier admits it.
    fn idle_hot<B: SnapTarget + ?Sized>(&mut self, b: &B, fed: &[u32]) -> bool {
        if !b.hot_capable() || self.tiers.cfg.hot_bytes == 0 || self.mode_ok(b).is_err() { return false; }
        if self.tiers.hot.iter().any(|h| h.snap.tokens() == fed) { return false; }
        let p = self.cache.path_for(b.engine(), fed);
        let Ok(m) = std::fs::metadata(&p) else { return false };
        let bytes = m.len();
        if let Err(e) = self.tiers.hot_room(bytes) {
            self.tiers.hot_counted(false);
            eprintln!("gateway: snapshot tiers: no hot copy: {}", e);
            return false;
        }
        streamdense_core::alloc::shift_limit(bytes as isize); // the copy is budgeted heap: the CPU heap limit moves with it
        match Snapshot::read(&p) {
            Ok(s) => { self.tiers.hot_add(s, bytes); self.tiers.hot_counted(true); true }
            Err(e) => { streamdense_core::alloc::shift_limit(-(bytes as isize)); eprintln!("gateway: snapshot tiers: hot copy {}: {}", p.display(), e); false }
        }
    }
}

/// Where a restored prefix of `n` tokens came from: "own" = it reaches past the system turn into the request's
/// conversation (a file of that conversation), "shared" = only the project prefix (system prompt + tools) or less.
pub fn restore_src(tokens: &[u32], n: usize) -> &'static str {
    let (project, conv) = crate::snap_tier::session_keys(&tokens[..n.min(tokens.len())]);
    if conv == project { "shared" } else { "own" }
}

fn touch(p: &PathBuf) {
    if let Ok(f) = std::fs::File::options().write(true).open(p) { let _ = f.set_modified(SystemTime::now()); }
}

/// What the snapshot cache saves / restores: the full model (`QwenEngine`, format 1) or the 12 GB-mode backend
/// (`compact::CompactBackend`, format 2: + prompt sums). The cache's `mode` tag must equal `snap_mode` (keys).
pub trait SnapTarget {
    fn engine(&self) -> &QwenEngine;
    /// `SnapshotCache.mode` for this target ("" = full model).
    fn snap_mode(&self) -> String;
    fn save_snap(&self, path: &Path, tokens: &[u32]) -> Result<(), String>;
    fn restore_snap(&self, path: &Path) -> Result<Vec<u32>, String>;
    /// Hot tier (RAM copies restored with `restore_ram`) possible for this target.
    fn hot_capable(&self) -> bool { false }
    fn restore_ram(&self, _s: &Snapshot) -> Result<(), String> { Err("no hot tier for this backend".into()) }
}
impl SnapTarget for QwenEngine {
    fn engine(&self) -> &QwenEngine { self }
    fn snap_mode(&self) -> String { String::new() }
    fn save_snap(&self, path: &Path, tokens: &[u32]) -> Result<(), String> { self.save_snapshot(path, tokens) }
    fn restore_snap(&self, path: &Path) -> Result<Vec<u32>, String> { self.restore_snapshot(path) }
    fn hot_capable(&self) -> bool { true }
    fn restore_ram(&self, s: &Snapshot) -> Result<(), String> { self.restore(s) }
}
impl SnapTarget for CompactBackend<'_> {
    fn engine(&self) -> &QwenEngine { self.eng }
    fn snap_mode(&self) -> String { CompactBackend::snap_mode(self) }
    fn save_snap(&self, path: &Path, tokens: &[u32]) -> Result<(), String> { self.save_snapshot(path, tokens) }
    fn restore_snap(&self, path: &Path) -> Result<Vec<u32>, String> { self.restore_snapshot(path) }
}

impl CacheHook {
    /// The cache was opened for another mode than this target's (a wiring bug): nothing is saved / restored.
    fn mode_ok<B: SnapTarget + ?Sized>(&self, b: &B) -> Result<(), String> {
        let m = b.snap_mode();
        if m == self.cache.mode { Ok(()) } else { Err(format!("snapshot cache mode {:?} != backend {:?}", self.cache.mode, m)) }
    }
}

impl<B: SnapTarget + ?Sized> SnapshotHook<B> for CacheHook {
    fn enabled(&self) -> bool { true }

    fn restore(&mut self, b: &B, tokens: &[u32]) -> Option<SnapEvent> {
        let t0 = Instant::now();
        if let Err(e) = self.mode_ok(b) { return Some(SnapEvent { kind: "restore", error: Some(e), ..Default::default() }); }
        let warm = self.cache.longest_prefix(b.engine(), tokens);
        let hot = if b.hot_capable() { self.tiers.hot_best(tokens) } else { None };
        let src = |n: usize| restore_src(tokens, n);
        if let Some((i, n)) = hot.filter(|h| h.1 >= warm.as_ref().map_or(0, |w| w.1)) {
            let r = b.restore_ram(&self.tiers.hot[i].snap);
            let bytes = self.tiers.hot[i].bytes;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            return Some(match r {
                Ok(()) => { self.tiers.hit_hot(i); SnapEvent { kind: "restore", tier: "hot", n, ms, bytes, error: None, src: src(n) } }
                Err(e) => { self.tiers.drop_hot(i); SnapEvent { kind: "restore", tier: "hot", n, ms, bytes, error: Some(e), src: src(n) } }
            });
        }
        let Some((p, n)) = warm else { self.tiers.miss(); return None };
        let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        let r = b.restore_snap(&p);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        touch(&p);
        self.tiers.hit_warm(&p);
        Some(match r {
            Ok(t) if t.len() == n => SnapEvent { kind: "restore", tier: "warm", n, ms, bytes, error: None, src: src(n) },
            Ok(t) => SnapEvent { kind: "restore", tier: "warm", n, ms, bytes, error: Some(format!("restored {} tokens, wanted {}", t.len(), n)), src: src(n) },
            Err(e) => { let _ = std::fs::remove_file(&p); SnapEvent { kind: "restore", tier: "warm", n, ms, bytes, error: Some(e), src: src(n) } }
        })
    }

    fn has(&self, b: &B, tokens: &[u32]) -> bool { self.mode_ok(b).is_ok() && self.cache.path_for(b.engine(), tokens).exists() }

    fn save(&mut self, b: &B, tokens: &[u32], stable: bool, tag: Option<StableTag>) -> Option<SnapEvent> {
        let kind = if stable { "save_stable" } else { "save_roll" };
        let t0 = Instant::now();
        let r = self.save_tagged(b, tokens, if stable { Kind::Stable } else { Kind::Roll }, tag);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        Some(match r {
            Ok(p) => SnapEvent { kind, n: tokens.len(), ms, bytes: std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), ..Default::default() },
            Err(e) => SnapEvent { kind, n: tokens.len(), ms, error: Some(e), ..Default::default() },
        })
    }

    fn stable_hot(&mut self, b: &B, tokens: &[u32]) -> Option<SnapEvent> {
        if !b.hot_capable() || self.tiers.cfg.hot_bytes == 0 || self.mode_ok(b).is_err() { return None; }
        let p = self.cache.path_for(b.engine(), tokens);
        let t0 = Instant::now();
        let r = self.tiers.hot_stable_file(&p, true);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        match r {
            Ok(true) => { self.tiers.publish(); Some(SnapEvent { kind: "hot_stable", tier: "hot", n: tokens.len(), ms, bytes, ..Default::default() }) }
            Ok(false) => None,
            Err(e) => Some(SnapEvent { kind: "hot_stable", tier: "hot", n: tokens.len(), ms, bytes, error: Some(e), ..Default::default() }),
        }
    }

    fn switch(&mut self, b: &B, live: &[u32], next: &[u32]) -> Option<SnapEvent> {
        if !self.tiers.cfg.switch_save || live.len() < STABLE_MIN || self.mode_ok(b).is_err() { return None; }
        let (project, conv) = crate::snap_tier::session_keys(live);
        // only a conversation state (past the system turn), and only when the next request is another conversation
        if conv == project || crate::snap_tier::session_keys(next).1 == conv { return None; }
        if self.tiers.covered(project, conv) + self.tiers.cfg.switch_save_min > live.len() { return None; }
        if self.cache.path_for(b.engine(), live).exists() { return None; }
        let t0 = Instant::now();
        let r = self.save_kind(b, live, Kind::Switch);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        Some(match r {
            Ok(p) => { self.tiers.switch_saved(); SnapEvent { kind: "save_switch", n: live.len(), ms, bytes: std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), ..Default::default() } }
            Err(e) => SnapEvent { kind: "save_switch", n: live.len(), ms, error: Some(e), ..Default::default() },
        })
    }

    fn tiers(&self) -> Option<Arc<TierShared>> { Some(self.tiers.shared.clone()) }

    fn job_begin(&mut self, t_recv: Instant) -> Option<Value> { Some(self.tiers.job_begin(t_recv)) }

    fn job_end(&mut self, _b: &B, fed: Option<&[u32]>) { self.tiers.job_end(fed); }

    fn idle_wait(&self) -> Option<Duration> { self.tiers.idle_wait() }

    fn idle(&mut self, b: &B, fed: Option<&[u32]>) {
        let Some(step) = self.tiers.next_step() else { return };
        let t0 = Instant::now();
        match (step, fed) {
            (Step::Save, Some(f)) => {
                if let Some(ms) = self.idle_save(b, f) { self.tiers.idle_saved(ms); }
                self.tiers.step_span(t0);
            }
            (Step::Hot, Some(f)) => { if self.idle_hot(b, f) { self.tiers.step_span(t0); } }
            (Step::HotStable, Some(_)) => { if self.idle_hot_stable(b) { self.tiers.step_span(t0); } }
            (Step::Preread, _) if self.tiers.cfg.preread => { let plan = self.tiers.preread_plan(); self.tiers.start_preread(plan); }
            _ => {}
        }
        self.tiers.save_index();
        self.tiers.publish();
    }
}
