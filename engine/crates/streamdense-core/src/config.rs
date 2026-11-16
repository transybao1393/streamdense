//! streamdense.toml: RAM limits + gateway defaults shared by sd-watchdog, `streamdense gateway|qwen serve|up|config`
//! and the Swift app (via `streamdense config --json`). Precedence everywhere: CLI flag > config file > built-in default.
//! Search order: $SD_CONFIG, ./streamdense.toml, the first streamdense.toml walking up from the executable's directory
//! (the repo root), ~/.config/streamdense/config.toml. No file -> `Config::default()` (the same values as the repo file).
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "streamdense.toml";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ram {
    /// hard cap on the model process footprint (sd-watchdog --cap-gb, engine --abort-gb)
    pub cap_gb: f64,
    /// system swap GROWTH limit since the process started (sd-watchdog + engine --swap-gb); not absolute swap
    pub swap_gb: f64,
    /// sd-watchdog --margin-gb
    pub margin_gb: f64,
    /// `nice -n` of the model process
    pub nice: i32,
}
impl Default for Ram {
    fn default() -> Self { Ram { cap_gb: 16.0, swap_gb: 7.0, margin_gb: 0.0, nice: 10 } }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gateway {
    /// model dir; relative paths resolve against the repo root
    pub model: String,
    pub port: u16,
    pub max_ctx: usize,
    /// KV cache format (gateway --kv): "bf16" (default) or "q8" (8-bit, 32-dim groups with an fp16 scale / min: 0.5625x
    /// the bf16 KV bytes, MTP KV included). q8 takes the fast prompt read as bf16 does (not the 12 GB mode's streamed one).
    pub kv: String,
    pub mtp: bool,
    /// MTP drafts verified per pass (1 = draft 1 / verify 2; 2..=7 = chained drafts, one exact pass of
    /// spec_depth + 1 rows; "auto" = adaptive 0..=7 per pass; needs mtp)
    pub spec_depth: SpecDepth,
    /// matmul family of the exact passes (gateway --gemv): "mma" (gemv_mma_q4.metal, default) or "old" (gemv_q4_bf16.metal)
    pub gemv: String,
    /// pruned MTP draft vocab keep-list (gateway --draft-vocab; `streamdense vocab keep`); "" = off (full-vocab drafts).
    /// Relative paths resolve against the repo root. Output is unchanged either way (verify keeps the full vocab).
    pub draft_vocab: String,
    /// learning draft table directory for spec_depth = "auto" (gateway --draft-table; "~/" = home; "" = off)
    pub draft_table: String,
    /// fused K-row verify matmuls (engine `QwenOptions.verify_fused`, verify_fused_q4.metal): bit-identical, faster
    /// verify passes of 3..=8 rows. Default OFF until the 27B check passes; gateway --verify-fused / --no-verify-fused.
    pub verify_fused: bool,
    /// DFlash block drafter directory (engine `QwenOptions.dflash`, dflash.rs; kit export_engine.py output; gateway
    /// --drafter DIR / --no-drafter). "" = off (default). Relative paths resolve against the repo root. Full mode only:
    /// replaces the MTP drafts (MTP is not loaded) and turns snapshots off (they hold no drafter context). Output
    /// identical to plain decode.
    pub drafter: String,
    /// DFlash2 block drafter directory (engine `QwenOptions.dflash2`, dflash2.rs; gateway --dflash2 DIR / --no-dflash2).
    /// "" = off (default). Relative paths resolve against the repo root. Experimental, not yet proven identical to plain
    /// decoding on the real 27B in this engine. Full mode only, refused with `drafter`; turns MTP and snapshots off;
    /// the whole budget must fit `[ram] cap_gb` / --abort-gb; the weights must match `[dflash2] expected_sha256`.
    pub dflash2: String,
    /// combined MTP + context drafts (engine `QwenOptions.mtp_ctx_draft`, ctx_draft.rs `CtxSource`): with MTP and
    /// spec_depth = "auto", each verify pass takes the context-lookup run (up to this many drafts) or the MTP chain,
    /// whichever the planner expects to give more tokens per ms. 0 = off (default). Output identical to plain decode.
    pub mtp_ctx_draft: usize,
}
impl Default for Gateway {
    fn default() -> Self {
        Gateway { model: "models/qwen38_27b_mlx4".into(), port: 8898, max_ctx: 8192, kv: "bf16".into(), mtp: true, spec_depth: SpecDepth::N(1), gemv: "mma".into(),
                  draft_vocab: String::new(), draft_table: String::new(), verify_fused: false, drafter: String::new(), dflash2: String::new(),
                  mtp_ctx_draft: 0 }
    }
}

/// sha256 of the tested DFlash2 weights (models/dflash2_27b_q4/model.safetensors, 1,082,881,511 B).
pub const DFLASH2_SHA256: &str = "ac61dedfd68c6524b71fd96cbe50a2e20749edfc0fbb25f964f61b93efcd022e";

/// `[dflash2]`: the file check of the DFlash2 option (`[gateway] dflash2`). Read only when that option is on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dflash2 {
    /// sha256 (64 hex digits) the weights file must have; the gateway refuses to start on a mismatch
    pub expected_sha256: String,
}
impl Default for Dflash2 {
    fn default() -> Self { Dflash2 { expected_sha256: DFLASH2_SHA256.into() } }
}

/// Languages cut from the draft vocab and treated as "opposite" by the watcher.
pub const CUT_LANGUAGES: [&str; 6] = ["zh", "ja", "ko", "tl", "id", "ms"];
/// `[watcher] mask` values (read mask).
pub const WATCHER_MASKS: [&str; 3] = ["off", "dry", "on"];
/// `[watcher] profile_window` values (read-mask profile: union of the parts' keep sets).
pub const PROFILE_WINDOWS: [&str; 6] = ["now", "session", "all", "now+all", "session+all", "now+session+all"];

/// `[watcher]`: the always-on domain watcher (gateway, streamdense-cli/src/watcher.rs). There is no off switch:
/// `mode = "observe"` (default) only decides + logs what would unload; `"unload"` is reserved (acts like observe).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Watcher {
    /// "observe" | "unload" (reserved; observe until then)
    pub mode: String,
    /// decision window: closes after this many fold notes (compactions) once it has `decide_window_min_turns` turns;
    /// a client that never folds closes it at 2 x min_turns. 0 = turns only.
    pub decide_window_folds: usize,
    /// decision window: at least this many turns (jobs). Learning still happens every turn.
    pub decide_window_min_turns: usize,
    /// similarity gate (heat store, 2026-11-01): a job whose activity profile is below this similarity to its label's
    /// slot (min of the block-fingerprint and head-mass cosines; ~0.5 = unrelated, 1 = same) is "almost opposite": it
    /// is held out of the slot (pooled now / session only) until it persists over the decision window. 0..=1
    pub opp_similarity: f64,
    /// heat store: fold the GPU recorder's batch to disk every this many consecutive same-label jobs (also on a label
    /// change, after 30 s idle and at shutdown). 1 = every job. 1..=64
    pub fold_every: usize,
    /// a window's top domain must lead the runner-up by this share (mean per-turn scores); also the per-turn margin a
    /// turn needs to be learned into its own label (else the unassigned pool)
    pub margin: f64,
    /// switch only after the same new domain leads this many windows in a row (sub-margin windows don't reset it)
    pub hysteresis: usize,
    /// go-live: clean-audit points in a row (clean audit = 1.0, mask-caused mismatch = reset + tighten)
    pub go_live_points: f64,
    /// Low-level watcher: sample period of the GPU recorder buffer (ms; background-QoS thread)
    pub sample_ms: u64,
    /// unload plan: keep the hottest FFN blocks / heads covering this share of a domain's activity per layer
    pub keep_frac: f64,
    /// unload plan + activity profiles: FFN neurons per block
    pub granularity: usize,
    /// state, per-domain profiles, decisions.jsonl, notes.jsonl ("~/" = home). One run elsewhere (tests, smoke runs):
    /// gateway / up `--watcher-dir DIR` or `$SD_WATCHER_DIR` (streamdense-cli watcher::config_dir); this default stays.
    pub dir: String,
    /// languages whose weights are always hot (never unload candidates); language is never a domain.
    /// "all-except-cut" = every language but `cut_languages`
    pub keep_languages: String,
    /// languages that are always "opposite": their specialist weights are unload-first candidates (found later by
    /// contrast probes; reserved). ISO 639-1 codes: zh ja ko tl id ms
    pub cut_languages: Vec<String>,
    /// Read mask (activity-only, one profile pooled over all labels): "off" (default: no profile, no predictor,
    /// nothing written) | "dry" (build the profile + score the CPU prompt predictor + schedule exploration; report
    /// what would be skipped; no mask applied) | "on" (apply the mask; off by default; acts as "dry"
    /// until the engine's mask hook exists)
    pub mask: String,
    /// Read-mask profile window: "now" | "session" | "all" | "now+all" | "session+all" | "now+session+all"; keep = the
    /// union of each part's keep set (blocks / heads holding `keep_frac` of that part's mass per layer)
    pub profile_window: String,
    /// Exploration: share of the masked blocks / heads re-read per idle round (CPU-ranked, rotating). 0..=1
    pub explore_frac: f64,
    /// answer library (2026-11-03): one compact record per turn (prompt FFN activity -> answer FFN activity, u8 log
    /// codes + a prompt sketch; 2.37 MB per turn on the 27B) in a ring file `<dir>/library/library_<dims>.sdl`, oldest
    /// dropped first. OFF by default; the user turns it on. No prompt text unless `library_text`.
    pub library: bool,
    /// ring file cap (GB = 1e9 B; ~420 turns per GB on the 27B)
    pub library_cap_gb: f64,
    /// also keep the newest user message text (<= 3,840 B) in each record (privacy: off by default)
    pub library_text: bool,
    /// retired 2026-11-01 (per-window margin rule): accepted so older files still load, ignored
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decide_every_turns: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// retired 2026-11-01 (votes removed; go-live = audits only): accepted so older files still load, ignored
    #[serde(skip_serializing_if = "Option::is_none")]
    pub like_weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harm_multiplier: Option<f64>,
}
impl Default for Watcher {
    fn default() -> Self {
        Watcher { mode: "observe".into(), decide_window_folds: 3, decide_window_min_turns: 50, opp_similarity: 0.25, fold_every: 8, margin: 0.15, hysteresis: 1,
                  go_live_points: 50.0, sample_ms: 100, keep_frac: 0.995, granularity: 64,
                  dir: "~/.streamdense/watcher".into(), keep_languages: "all-except-cut".into(),
                  cut_languages: CUT_LANGUAGES.iter().map(|s| s.to_string()).collect(),
                  mask: "off".into(), profile_window: "all".into(), explore_frac: 0.02, library: false, library_cap_gb: 1.0, library_text: false, decide_every_turns: None, confidence: None, like_weight: None, harm_multiplier: None }
    }
}

/// `[snapshots]`: the gateway's snapshot tiers (streamdense-gateway snap_tier.rs, docs/GATEWAY.md "Snapshot tiers").
/// hot = in-process RAM copies (only while the live footprint leaves room under `[ram] cap_gb`; never in 12 GB mode),
/// warm = `.sdsnap` files on SSD, cold = deleted by the disk policy. Idle work (gateway idle >= `idle_s`): save the
/// live session's state, page in the likely next files; stopped as soon as a request arrives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Snapshots {
    /// max total size of the snapshot files on disk (GB = 1e9 B; gateway --snapshot-max-gb wins)
    pub max_disk_gb: f64,
    /// max number of snapshot files (gateway --snapshot-max-files wins)
    pub max_files: usize,
    /// keep at least this much free space on the snapshot disk: no save below it, files evicted until it holds
    pub min_free_disk_gb: f64,
    /// gateway idle this long (no request running or queued) before idle work starts (s)
    pub idle_s: f64,
    /// idle work: save the live session's state (prompt + reply) so the next turn / a restart restores it
    pub idle_save: bool,
    /// ... only when the live state is at least this many tokens past that session's newest file
    pub idle_save_min_tokens: usize,
    /// idle work: page in (OS file cache) the snapshot files most likely needed next
    pub preread: bool,
    /// ... at most this many files and this many GB per idle period
    pub preread_files: usize,
    pub preread_gb: f64,
    /// hot tier: RAM copies of recent session states, at most this many GB (0 = off). Admitted only while
    /// footprint + copy + `hot_headroom_gb` <= `[ram] cap_gb`; dropped first when the footprint rises
    pub hot_gb: f64,
    pub hot_headroom_gb: f64,
    /// hot tier: also keep a RAM copy of the live project's shared prefix (system prompt + tools) when it fits
    /// without dropping another copy (same admission rule; a 32 GB machine: raise `[ram] cap_gb` and `hot_gb`)
    pub hot_stable: bool,
    /// multi-user: before a request of another conversation replaces the live state, save the live state (prompt +
    /// reply) when it is at least `switch_save_min_tokens` past that conversation's newest file (or its shared prefix),
    /// so interleaved users each restore their own turn instead of re-reading it
    pub switch_save: bool,
    pub switch_save_min_tokens: usize,
}
impl Default for Snapshots {
    fn default() -> Self {
        Snapshots { max_disk_gb: 8.0, max_files: 32, min_free_disk_gb: 20.0, idle_s: 5.0, idle_save: true, idle_save_min_tokens: 256,
                    preread: true, preread_files: 2, preread_gb: 1.5, hot_gb: 1.0, hot_headroom_gb: 1.0,
                    hot_stable: true, switch_save: true, switch_save_min_tokens: 64 }
    }
}

/// `[history]`: the history index (streamdense-core history_index.rs): drafts from text the user already has (project
/// files, finished replies). OFF by default. Used by the 12 GB mode's context drafts ([forheader] ctx_draft) as a
/// third draft source; lossless (every draft is verified). Its RAM (`budget_mb`) is counted in the 12 GB keep formula.
/// Build it ahead with `streamdense index build --project DIR [--dir D]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct History {
    pub enabled: bool,
    /// index dir ("~/" = home)
    pub dir: String,
    /// hard RAM bound of the index (table + manifest + build buffers), MB (2^20 B)
    pub budget_mb: f64,
    /// project dir scanned (incrementally, background QoS) at gateway start; "" = no scan
    pub project: String,
    /// session logs (file or dir of JSON / JSONL) scanned at gateway start; "" = no scan
    pub sessions: String,
}
impl Default for History {
    fn default() -> Self { History { enabled: false, dir: "~/.streamdense/history_index".into(), budget_mb: 32.0, project: String::new(), sessions: String::new() } }
}

/// `[forheader] mode` values.
pub const FORHEADER_MODES: [&str; 2] = ["off", "12gb"];

/// `[forheader]`: the 12 GB mode (research; 2026-11-03). OFF by default: the gateway runs the full model. "12gb":
/// every prompt is read with the full model streamed from SSD and the answer is decoded with a smaller resident FFN
/// whose size comes from `cap_gb` by formula (gateway, streamdense-gateway/src/compact.rs). The activity recorder, the
/// answer library are off in this mode (snapshots work: their own file format, see streamdense-model snapshot.rs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Forheader {
    /// "off" | "12gb"
    pub mode: String,
    /// RAM the resident set is sized for (GB = 1e9 B; the keep formula's cap). <= ram.cap_gb
    pub cap_gb: f64,
    /// MTP decode in this mode (default off: the MTP layer + its KV cost resident neurons)
    pub mtp: bool,
    /// full-layer read slots of the streamed prompt read (1..=4; 2 overlaps the next layer's read with the GPU)
    pub slots: usize,
    /// place the read slots inside the idle lm_head (no extra RAM; the lm_head part is re-read per prompt)
    pub host: bool,
    /// max prompt tokens per streamed pass (longer prompts go in several passes, each a full FFN read); capped at
    /// gateway.max_ctx. 8..=262144
    pub stream_tokens: usize,
    /// fast streamed prompt read (GEMM prefill kernels; 27B TTFT 21.9/32.4/39.4 s -> 11.96/17.03/21.46 s at
    /// 1029/1506/1833 tokens, first token same 13/13, not bit-exact with the exact read). Default ON (2026-11-03);
    /// false (or gateway --no-fast) = the exact read. Needs bf16 KV (off with a warning under --kv q8)
    pub fast: bool,
    /// fused decode (engine `decode_fused`: MmaD GEMV twins + encode-ahead; bit-identical output; 27B 56.70 -> 51.02
    /// ms/token, 13/13 identical). Default ON (2026-11-03); false (or gateway --no-fused) = today's decode loop
    pub fused: bool,
    /// global score file (`streamdense forheader export|install`; "~/" = home)
    pub t1: String,
    /// context drafts (engine ctx_draft.rs): draft from text already in the context (+ the [gateway]
    /// draft_table when set) and check each block in one exact pass; output identical to plain decode. Default OFF; gateway --ctx-draft / --no-ctx-draft. Its RAM (hold save + lookup index + table) is
    /// counted in the keep formula, so K is a little smaller than without it.
    pub ctx_draft: bool,
    /// max drafts per verify pass with ctx_draft (1..=7; K = d + 1 rows)
    pub ctx_draft_max: usize,
    /// fused K-row verify matmuls in this mode (as `[gateway] verify_fused`; the 12 GB mode reads this one). Default OFF.
    pub verify_fused: bool,
}
impl Default for Forheader {
    fn default() -> Self {
        Forheader { mode: "off".into(), cap_gb: 12.0, mtp: false, slots: 2, host: true, stream_tokens: 8192, fast: true, fused: true, t1: "~/.streamdense/forheader/t1.sdt1".into(),
                    ctx_draft: false, ctx_draft_max: 7, verify_fused: false }
    }
}
impl Forheader {
    pub fn on(&self) -> bool { self.mode == "12gb" }
}

/// `spec_depth`: a number (fixed depth) or "auto" (adaptive, max 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SpecDepth {
    N(usize),
    #[serde(with = "auto_str")]
    Auto,
}
impl SpecDepth {
    /// Flag form: "auto" or the number.
    pub fn arg(&self) -> String { match self { SpecDepth::N(n) => n.to_string(), SpecDepth::Auto => "auto".into() } }
    /// Fixed depth > 1 or auto (the deep spec loop).
    pub fn deep(&self) -> bool { !matches!(self, SpecDepth::N(0) | SpecDepth::N(1)) }
}
mod auto_str {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(s: S) -> Result<S::Ok, S::Error> { s.serialize_str("auto") }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<(), D::Error> {
        let v = String::deserialize(d)?;
        if v == "auto" { Ok(()) } else { Err(serde::de::Error::custom(format!("spec_depth {:?}: want 1..=7 or \"auto\"", v))) }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub ram: Ram,
    pub gateway: Gateway,
    pub watcher: Watcher,
    pub forheader: Forheader,
    pub snapshots: Snapshots,
    pub history: History,
    pub dflash2: Dflash2,
}

/// A resolved config and the file it came from (`None` = built-in defaults).
#[derive(Debug, Clone)]
pub struct Loaded {
    pub cfg: Config,
    pub source: Option<PathBuf>,
}

impl Loaded {
    /// `gateway.model` as a path: absolute as is, else joined to `root` (the repo root).
    pub fn model_path(&self, root: &Path) -> PathBuf {
        let m = Path::new(&self.cfg.gateway.model);
        if m.is_absolute() { m.to_path_buf() } else { root.join(m) }
    }
    pub fn source_str(&self) -> String {
        self.source.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "built-in defaults (no config file found)".into())
    }
}

/// Parse one file; errors name the file (and, for unknown keys, the key and its line).
pub fn parse_file(path: &Path) -> Result<Config, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("config {}: {}", path.display(), e))?;
    parse_str(&text).map_err(|e| format!("config {}: {}", path.display(), e))
}

/// `[gateway] kv` values (gateway --kv).
pub const KV_FORMATS: [&str; 2] = ["bf16", "q8"];

pub fn parse_str(text: &str) -> Result<Config, String> {
    let cfg: Config = toml::from_str(text).map_err(|e| e.to_string().trim_end().to_string())?;
    validate(&cfg)?;
    Ok(cfg)
}

fn validate(c: &Config) -> Result<(), String> {
    let r = &c.ram;
    if !(r.cap_gb > 0.0 && r.cap_gb.is_finite()) { return Err(format!("ram.cap_gb = {}: want > 0", r.cap_gb)); }
    if !(r.swap_gb >= 0.0 && r.swap_gb.is_finite()) { return Err(format!("ram.swap_gb = {}: want >= 0", r.swap_gb)); }
    if !(r.margin_gb >= 0.0 && r.margin_gb.is_finite()) { return Err(format!("ram.margin_gb = {}: want >= 0", r.margin_gb)); }
    if !(-20..=20).contains(&r.nice) { return Err(format!("ram.nice = {}: want -20..=20", r.nice)); }
    if c.gateway.max_ctx == 0 { return Err("gateway.max_ctx = 0".into()); }
    if !KV_FORMATS.contains(&c.gateway.kv.as_str()) { return Err(format!("gateway.kv = {:?}: want \"bf16\" or \"q8\"", c.gateway.kv)); }
    if c.gateway.model.is_empty() { return Err("gateway.model is empty".into()); }
    if let SpecDepth::N(d) = c.gateway.spec_depth {
        if !(1..=7).contains(&d) { return Err(format!("gateway.spec_depth = {}: want 1..=7 or \"auto\"", d)); }
    }
    if c.gateway.mtp_ctx_draft > 7 { return Err(format!("gateway.mtp_ctx_draft = {}: want 0..=7", c.gateway.mtp_ctx_draft)); }
    if !["mma", "old"].contains(&c.gateway.gemv.as_str()) { return Err(format!("gateway.gemv = {:?}: want \"mma\" or \"old\"", c.gateway.gemv)); }
    let e = &c.dflash2.expected_sha256;
    if e.len() != 64 || !e.chars().all(|c| c.is_ascii_hexdigit()) { return Err(format!("dflash2.expected_sha256 = {:?}: want 64 hex digits", e)); }
    let w = &c.watcher;
    if !["observe", "unload"].contains(&w.mode.as_str()) { return Err(format!("watcher.mode = {:?}: want \"observe\" or \"unload\" (the watcher cannot be switched off)", w.mode)); }
    if w.decide_window_folds > 20 { return Err(format!("watcher.decide_window_folds = {}: want 0..=20", w.decide_window_folds)); }
    if !(1..=64).contains(&w.fold_every) { return Err(format!("watcher.fold_every = {}: want 1..=64", w.fold_every)); }
    if !(0.0..=1.0).contains(&w.opp_similarity) { return Err(format!("watcher.opp_similarity = {}: want 0..=1", w.opp_similarity)); }
    if w.decide_window_min_turns > 10_000 { return Err(format!("watcher.decide_window_min_turns = {}: want 0..=10000", w.decide_window_min_turns)); }
    if !(w.margin >= 0.0 && w.margin < 1.0) { return Err(format!("watcher.margin = {}: want 0..1", w.margin)); }
    if !(1..=10).contains(&w.hysteresis) { return Err(format!("watcher.hysteresis = {}: want 1..=10 (windows)", w.hysteresis)); }
    if !(w.go_live_points >= 1.0 && w.go_live_points.is_finite()) { return Err(format!("watcher.go_live_points = {}: want >= 1", w.go_live_points)); }
    if !(10..=10_000).contains(&w.sample_ms) { return Err(format!("watcher.sample_ms = {}: want 10..=10000", w.sample_ms)); }
    if !(w.keep_frac > 0.5 && w.keep_frac <= 1.0) { return Err(format!("watcher.keep_frac = {}: want (0.5, 1]", w.keep_frac)); }
    if !(1..=4096).contains(&w.granularity) { return Err(format!("watcher.granularity = {}: want 1..=4096", w.granularity)); }
    if w.keep_languages != "all-except-cut" { return Err(format!("watcher.keep_languages = {:?}: want \"all-except-cut\"", w.keep_languages)); }
    if let Some(l) = w.cut_languages.iter().find(|l| !(2..=3).contains(&l.len()) || !l.chars().all(|c| c.is_ascii_lowercase())) {
        return Err(format!("watcher.cut_languages: {:?} is not an ISO 639 code (e.g. \"zh\")", l));
    }
    if w.dir.is_empty() { return Err("watcher.dir is empty".into()); }
    if !WATCHER_MASKS.contains(&w.mask.as_str()) { return Err(format!("watcher.mask = {:?}: want \"off\", \"dry\" or \"on\"", w.mask)); }
    if !PROFILE_WINDOWS.contains(&w.profile_window.as_str()) { return Err(format!("watcher.profile_window = {:?}: want one of {:?}", w.profile_window, PROFILE_WINDOWS)); }
    if !(0.0..=1.0).contains(&w.explore_frac) { return Err(format!("watcher.explore_frac = {}: want 0..=1", w.explore_frac)); }
    if !(w.library_cap_gb > 0.0 && w.library_cap_gb <= 10_000.0) { return Err(format!("watcher.library_cap_gb = {}: want (0, 10000]", w.library_cap_gb)); }
    let f = &c.forheader;
    if !FORHEADER_MODES.contains(&f.mode.as_str()) { return Err(format!("forheader.mode = {:?}: want \"off\" or \"12gb\"", f.mode)); }
    if !(f.cap_gb > 0.0 && f.cap_gb.is_finite()) { return Err(format!("forheader.cap_gb = {}: want > 0", f.cap_gb)); }
    if f.cap_gb > r.cap_gb { return Err(format!("forheader.cap_gb = {} is above ram.cap_gb = {} (the hard cap)", f.cap_gb, r.cap_gb)); }
    if !(1..=4).contains(&f.slots) { return Err(format!("forheader.slots = {}: want 1..=4", f.slots)); }
    if !(8..=262_144).contains(&f.stream_tokens) { return Err(format!("forheader.stream_tokens = {}: want 8..=262144", f.stream_tokens)); }
    if !(1..=7).contains(&f.ctx_draft_max) { return Err(format!("forheader.ctx_draft_max = {}: want 1..=7", f.ctx_draft_max)); }
    if f.on() && f.t1.is_empty() { return Err("forheader.t1 is empty (mode \"12gb\" needs the global score file)".into()); }
    let s = &c.snapshots;
    if !(s.max_disk_gb > 0.0 && s.max_disk_gb.is_finite()) { return Err(format!("snapshots.max_disk_gb = {}: want > 0", s.max_disk_gb)); }
    if s.max_files == 0 { return Err("snapshots.max_files = 0: want >= 1".into()); }
    if !(s.min_free_disk_gb >= 0.0 && s.min_free_disk_gb.is_finite()) { return Err(format!("snapshots.min_free_disk_gb = {}: want >= 0", s.min_free_disk_gb)); }
    if !(s.idle_s >= 0.5 && s.idle_s <= 3600.0) { return Err(format!("snapshots.idle_s = {}: want 0.5..=3600", s.idle_s)); }
    if !(s.preread_gb >= 0.0 && s.preread_gb.is_finite()) { return Err(format!("snapshots.preread_gb = {}: want >= 0", s.preread_gb)); }
    if !(s.hot_gb >= 0.0 && s.hot_gb.is_finite()) { return Err(format!("snapshots.hot_gb = {}: want >= 0", s.hot_gb)); }
    if !(s.hot_headroom_gb >= 0.0 && s.hot_headroom_gb.is_finite()) { return Err(format!("snapshots.hot_headroom_gb = {}: want >= 0", s.hot_headroom_gb)); }
    let h = &c.history;
    if !(h.budget_mb >= 0.25 && h.budget_mb <= 1024.0) { return Err(format!("history.budget_mb = {}: want 0.25..=1024", h.budget_mb)); }
    if h.enabled && h.dir.is_empty() { return Err("history.dir is empty".into()); }
    if s.hot_gb > r.cap_gb { return Err(format!("snapshots.hot_gb = {} is above ram.cap_gb = {}", s.hot_gb, r.cap_gb)); }
    Ok(())
}

/// The file `load()` would read, if any. $SD_CONFIG must exist when set.
pub fn find() -> Result<Option<PathBuf>, String> {
    if let Some(p) = std::env::var_os("SD_CONFIG").filter(|p| !p.is_empty()) {
        let p = PathBuf::from(p);
        if !p.is_file() { return Err(format!("SD_CONFIG={}: no such file", p.display())); }
        return Ok(Some(p));
    }
    if let Ok(cwd) = std::env::current_dir() {
        let p = cwd.join(FILE_NAME);
        if p.is_file() { return Ok(Some(p)); }
    }
    if let Ok(exe) = std::env::current_exe() {
        let exe = exe.canonicalize().unwrap_or(exe);
        for d in exe.ancestors().skip(1) {
            let p = d.join(FILE_NAME);
            if p.is_file() { return Ok(Some(p)); }
        }
    }
    if let Some(h) = std::env::var_os("HOME") {
        let p = PathBuf::from(h).join(".config/streamdense/config.toml");
        if p.is_file() { return Ok(Some(p)); }
    }
    Ok(None)
}

/// Find + parse; no file -> defaults.
pub fn load() -> Result<Loaded, String> {
    match find()? {
        Some(p) => Ok(Loaded { cfg: parse_file(&p)?, source: Some(p) }),
        None => Ok(Loaded { cfg: Config::default(), source: None }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // the repo file parses; its [ram] limits are the built-in defaults (the 16 GB cap is fixed). [gateway] holds the
    // user's chosen settings (spec_depth, draft_vocab, draft_table, ..) and is not compared.
    fn repo_file_parses_ram_is_default() {
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../streamdense.toml")).unwrap();
        let c = parse_str(&t).unwrap();
        assert_eq!(c.ram, Ram::default());
        assert_eq!(c.ram.cap_gb, 16.0);
        // the live repo file may run the watcher's read mask as "dry" (report only); every other key stays default
        assert!(["off", "dry"].contains(&c.watcher.mask.as_str()), "{}", c.watcher.mask);
        assert_eq!(c.watcher, Watcher { mask: c.watcher.mask.clone(), ..Watcher::default() });
    }
    #[test]
    fn partial_and_unknown() {
        let c = parse_str("[ram]\ncap_gb = 12\n").unwrap();
        assert_eq!(c.ram.cap_gb, 12.0);
        assert_eq!(c.gateway, Gateway::default());
        let e = parse_str("[ram]\ncap_gbb = 12\n").unwrap_err();
        assert!(e.contains("cap_gbb"), "{}", e);
        assert!(parse_str("[rams]\n").unwrap_err().contains("rams"));
        assert_eq!(parse_str("[gateway]\nspec_depth = \"auto\"\n").unwrap().gateway.spec_depth, SpecDepth::Auto);
        assert_eq!(parse_str("[gateway]\nspec_depth = 4\n").unwrap().gateway.spec_depth, SpecDepth::N(4));
        assert!(parse_str("[gateway]\nspec_depth = \"fast\"\n").is_err());
        assert!(parse_str("[gateway]\nspec_depth = 9\n").is_err());
        let w = parse_str("[watcher]\ndecide_window_folds = 2\ndecide_window_min_turns = 40\nmargin = 0.2\nhysteresis = 2\ngo_live_points = 30\nsample_ms = 250\n").unwrap().watcher;
        assert_eq!((w.decide_window_folds, w.decide_window_min_turns, w.margin, w.hysteresis), (2, 40, 0.2, 2));
        assert_eq!((w.go_live_points, w.sample_ms), (30.0, 250));
        let d = Watcher::default();
        assert_eq!((d.decide_window_folds, d.decide_window_min_turns, d.margin, d.hysteresis, d.go_live_points, d.sample_ms), (3, 50, 0.15, 1, 50.0, 100));
        // retired vote keys (2026-11-01) still load, ignored
        let v = parse_str("[watcher]\nlike_weight = 0.1\nharm_multiplier = 2\n").unwrap().watcher;
        assert_eq!((v.like_weight, v.harm_multiplier, v.go_live_points), (Some(0.1), Some(2.0), 50.0));
        assert!(!toml::to_string(&Config::default()).unwrap().contains("like_weight"));
        // retired keys of the first watcher still load (ignored), so an older file never breaks `streamdense up`
        let old = parse_str("[watcher]\ndecide_every_turns = 3\nconfidence = 0.5\nhysteresis = 2\n").unwrap().watcher;
        assert_eq!((old.decide_every_turns, old.confidence, old.hysteresis, old.margin), (Some(3), Some(0.5), 2, 0.15));
        assert_eq!(d.cut_languages, vec!["zh", "ja", "ko", "tl", "id", "ms"]);
        assert_eq!(d.keep_languages, "all-except-cut");
        assert_eq!(parse_str("[watcher]\ncut_languages = [\"zh\", \"ja\"]\n").unwrap().watcher.cut_languages, vec!["zh", "ja"]);
        assert!(parse_str("[watcher]\ncut_languages = [\"Chinese\"]\n").is_err());
        assert!(parse_str("[watcher]\nkeep_languages = \"en\"\n").is_err());
        assert!(parse_str("[watcher]\nmargin = 1.5\n").is_err());
        assert!(parse_str("[watcher]\nhysteresis = 0\n").is_err());
        assert!(parse_str("[watcher]\nsample_ms = 1\n").is_err());
        assert!(!toml::to_string(&Config::default()).unwrap().contains("decide_every_turns"));
        assert!(parse_str("[watcher]\nenabled = false\n").unwrap_err().contains("enabled")); // no off switch
        assert!(parse_str("[watcher]\nmode = \"off\"\n").is_err());
        assert!(parse_str("[watcher]\nkeep_frac = 0.2\n").is_err());
        // read mask: default off; dry / on + window + exploration share validated
        assert_eq!((d.mask.as_str(), d.profile_window.as_str(), d.explore_frac), ("off", "all", 0.02));
        let m = parse_str("[watcher]\nmask = \"dry\"\nprofile_window = \"now+all\"\nexplore_frac = 0.1\n").unwrap().watcher;
        assert_eq!((m.mask.as_str(), m.profile_window.as_str(), m.explore_frac), ("dry", "now+all", 0.1));
        assert!(parse_str("[watcher]\nmask = \"yes\"\n").unwrap_err().contains("watcher.mask"));
        assert!(parse_str("[watcher]\nprofile_window = \"week\"\n").unwrap_err().contains("profile_window"));
        assert!(parse_str("[watcher]\nexplore_frac = 2\n").is_err());
        // answer library: default off, 1 GB, no text; switch + cap + text parse; a zero / negative cap is refused
        assert_eq!((d.library, d.library_cap_gb, d.library_text), (false, 1.0, false));
        let l = parse_str("[watcher]\nlibrary = true\nlibrary_cap_gb = 4\n").unwrap().watcher;
        assert_eq!((l.library, l.library_cap_gb, l.library_text), (true, 4.0, false));
        assert!(parse_str("[watcher]\nlibrary_text = true\n").unwrap().watcher.library_text);
        assert!(parse_str("[watcher]\nlibrary_cap_gb = 0\n").unwrap_err().contains("library_cap_gb"));
    }
    #[test]
    fn gateway_kv() {
        assert_eq!(Gateway::default().kv, "bf16");
        assert_eq!(parse_str("[ram]\ncap_gb = 16\n").unwrap().gateway.kv, "bf16");
        assert_eq!(parse_str("[gateway]\nkv = \"q8\"\n").unwrap().gateway.kv, "q8");
        assert_eq!(parse_str("[gateway]\nkv = \"bf16\"\n").unwrap().gateway.kv, "bf16");
        for bad in ["\"q4\"", "\"Q8\"", "\"\"", "8"] {
            assert!(parse_str(&format!("[gateway]\nkv = {}\n", bad)).is_err(), "kv = {} accepted", bad);
        }
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../streamdense.toml")).unwrap();
        assert_eq!(parse_str(&t).unwrap().gateway.kv, "bf16", "the repo file keeps bf16 KV");
    }
    #[test]
    fn forheader_section() {
        // default off; the repo file / any file without the section stays off
        let d = Forheader::default();
        assert!(!d.on());
        assert_eq!((d.mode.as_str(), d.cap_gb, d.mtp, d.slots, d.host, d.stream_tokens, d.fast, d.fused), ("off", 12.0, false, 2, true, 8192, true, true));
        assert!(!parse_str("[forheader]\nfast = false\n").unwrap().forheader.fast);
        assert!(!parse_str("[forheader]\nfused = false\n").unwrap().forheader.fused);
        assert!(!d.ctx_draft);
        assert_eq!(d.ctx_draft_max, 7);
        let c = parse_str("[forheader]\nctx_draft = true\nctx_draft_max = 3\n").unwrap().forheader;
        assert!(c.ctx_draft && c.ctx_draft_max == 3);
        assert!(parse_str("[forheader]\nctx_draft_max = 0\n").unwrap_err().contains("ctx_draft_max"));
        assert!(parse_str("[forheader]\nctx_draft_max = 8\n").unwrap_err().contains("ctx_draft_max"));
        assert_eq!(parse_str("[ram]\ncap_gb = 16\n").unwrap().forheader, d);
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../streamdense.toml")).unwrap();
        assert!(!parse_str(&t).unwrap().forheader.on(), "the repo file must keep the 12 GB mode off");
        let f = parse_str("[forheader]\nmode = \"12gb\"\ncap_gb = 11.5\nmtp = true\nslots = 1\nhost = false\nstream_tokens = 2048\nt1 = \"/x/t1.sdt1\"\n").unwrap().forheader;
        assert!(f.on());
        assert_eq!((f.cap_gb, f.mtp, f.slots, f.host, f.stream_tokens, f.t1.as_str()), (11.5, true, 1, false, 2048, "/x/t1.sdt1"));
        for (bad, key) in [("mode = \"10gb\"", "forheader.mode"), ("cap_gb = 0", "forheader.cap_gb"), ("cap_gb = -1", "forheader.cap_gb"), ("cap_gb = 20", "above ram.cap_gb"),
                           ("slots = 0", "forheader.slots"), ("slots = 5", "forheader.slots"), ("stream_tokens = 4", "stream_tokens"),
                           ("mode = \"12gb\"\nt1 = \"\"", "forheader.t1"), ("kk = 1", "kk")] {
            let e = parse_str(&format!("[forheader]\n{}\n", bad)).unwrap_err();
            assert!(e.contains(key), "{:?}: {}", bad, e);
        }
        // the cap may sit at the hard cap; a lower hard cap moves the limit
        assert!(parse_str("[forheader]\ncap_gb = 16\n").is_ok());
        assert!(parse_str("[ram]\ncap_gb = 12\n[forheader]\ncap_gb = 12.5\n").is_err());
    }
    #[test]
    fn history_and_verify_fused() {
        let d = History::default();
        assert_eq!((d.enabled, d.dir.as_str(), d.budget_mb, d.project.as_str()), (false, "~/.streamdense/history_index", 32.0, ""));
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../streamdense.toml")).unwrap();
        let c = parse_str(&t).unwrap();
        assert!(!c.history.enabled && !c.gateway.verify_fused && !c.forheader.verify_fused, "the repo file keeps history / verify_fused off");
        let h = parse_str("[history]\nenabled = true\ndir = \"/tmp/x\"\nbudget_mb = 4\nproject = \"/p\"\n").unwrap().history;
        assert!(h.enabled && h.budget_mb == 4.0 && h.project == "/p");
        assert!(parse_str("[history]\nbudget_mb = 0.1\n").unwrap_err().contains("budget_mb"));
        assert!(parse_str("[history]\nfoo = 1\n").is_err());
        assert!(parse_str("[gateway]\nverify_fused = true\n").unwrap().gateway.verify_fused);
        assert_eq!(c.gateway.mtp_ctx_draft, 0, "the repo file keeps mtp_ctx_draft off");
        assert_eq!(parse_str("[gateway]\nmtp_ctx_draft = 7\n").unwrap().gateway.mtp_ctx_draft, 7);
        assert!(parse_str("[gateway]\nmtp_ctx_draft = 8\n").is_err());
        assert!(parse_str("[forheader]\nverify_fused = true\n").unwrap().forheader.verify_fused);
    }
    #[test]
    fn snapshots_section() {
        let d = Snapshots::default();
        assert_eq!((d.max_disk_gb, d.max_files, d.min_free_disk_gb, d.idle_s, d.idle_save, d.preread, d.hot_gb), (8.0, 32, 20.0, 5.0, true, true, 1.0));
        assert_eq!((d.hot_stable, d.switch_save, d.switch_save_min_tokens), (true, true, 64));
        assert_eq!(parse_str("[ram]\ncap_gb = 16\n").unwrap().snapshots, d);
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../streamdense.toml")).unwrap();
        assert_eq!(parse_str(&t).unwrap().snapshots, d, "the repo file documents the defaults");
        let s = parse_str("[snapshots]\nmax_disk_gb = 2\nmax_files = 3\nidle_s = 1\nidle_save = false\nhot_gb = 0\n").unwrap().snapshots;
        assert_eq!((s.max_disk_gb, s.max_files, s.idle_s, s.idle_save, s.hot_gb), (2.0, 3, 1.0, false, 0.0));
        for (bad, key) in [("max_disk_gb = 0", "max_disk_gb"), ("max_files = 0", "max_files"), ("idle_s = 0.1", "idle_s"), ("hot_gb = -1", "hot_gb"),
                           ("hot_gb = 17", "above ram.cap_gb"), ("min_free_disk_gb = -2", "min_free_disk_gb"), ("ttl = 1", "ttl")] {
            let e = parse_str(&format!("[snapshots]\n{}\n", bad)).unwrap_err();
            assert!(e.contains(key), "{:?}: {}", bad, e);
        }
    }
}
