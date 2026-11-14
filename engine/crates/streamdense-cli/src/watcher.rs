//! Domain watcher: always on, CPU only, inside `streamdense gateway`. No off switch: `[watcher]
//! mode = "observe"` (default) decides + logs; "unload" is reserved for a later unload stage and observes until then.
//!
//! Inputs: (a) notes from clients: POST /v1/streamdense/watcher/note {"kind": "state_card"|"fold"|"turn", "text",
//! "project", "mode": "chat"|"code"} (a client's state cards / fold summaries); (b) every /v1/messages job (newest user
//! text + trailing tool results + the generated reply -> cheap features; text itself is never kept, only features +
//! short hashes); (c) engine activity per job (activity recorder: `engine_activity`, contribution mass per neuron + head mass).
//! Two clocks:
//! - Every turn: the job's own scores (rules, blended 75 / 25 with activity-fingerprint centroids once two domains
//!   have >= 3 learned turns) -> the turn's own label if its top domain leads the runner-up by >= `margin`, else
//!   "unassigned". The turn's activity is learned into THAT label (never into the active domain), and the reply's
//!   audit is attributed to it.
//! - Decision window (`decide_window_folds` folds and >= `decide_window_min_turns` turns, ~2-3 compactions, 50+ turns;
//!   no folds from a client: 2 x min_turns): mean per-turn / per-note scores; the top domain must lead the runner-up
//!   by >= `margin` (no absolute threshold). A new domain is adopted after `hysteresis` such windows in a row; a
//!   sub-margin window keeps the count (never resets it). No margin and the active domain not in the top two -> mixed
//!   (the union profile). Fast topic changes are the fallback's job (mask off), not quick switches.
//! Domains are topics only: language never picks the domain (kept languages are always hot; cut languages =
//! `cut_languages`, unload-first, identified later by contrast probes). State cards / folds: their prose is neutral (cards
//! read the same whatever the topic); tool results
//! count code markers double and their prose not at all; code fences / tool use on code lift the code share.
//! Plans (dry run): per domain, `unload::unload_plan` on the domain's accumulated profile. A wrong domain can only
//! cost speed (fallback), so plans are advisory. Plans are cached per domain (recomputed after the profile learns).
//! Go-live (audits only, no user votes): every reply is queued for an audit (bounded, newest first,
//! drained by the low-level watcher thread while the engine is idle, heat.rs); per profile, clean-audit points in a
//! row: clean audit 1.0, mask-caused mismatch = reset + tighten. Live = points >= go_live_points while masked. No
//! masks yet: every audit is recorded "skipped" (the real re-check is a TODO hook in `audit_next`).
//! Disk (dir, default ~/.streamdense/watcher): state.json (incl. the open window + go-live state + held labels),
//! store/<dims tag>_V<vocab>/store.bin (`store.rs`, heat on disk 2026-11-01: ONE fixed-size file <= 64,000,000 B; per
//! label one slot of full f32: per-neuron contribution sums, heads / resid, MTP layer, emitted tokens, embedding rows,
//! the activity centroid, counters; plus the pooled now / session regions), decisions.jsonl + notes.jsonl +
//! audits.jsonl (append-only, compacted).
//! Similarity gate (2026-11-01): before a job folds into its label's slot it is compared with the slot (`Store::sim`:
//! min of the block-fingerprint cosine vs the slot centroid and the head-mass cosine). Below `opp_similarity`
//! (almost opposite) it is NOT written into the slot: it folds into the pooled now / session only and is counted as
//! held for that label. A label without a slot and no free slot is held the same way. Held turns that persist over
//! the decision window (`decide_window_folds` folds since the first hold and >= `decide_window_min_turns` held turns,
//! or 2 x min_turns without folds) are confirmed: a slotted label then folds again (the gate stays open until the
//! slot is similar again); a label without a slot replaces the least-used AND most-dissimilar slot (score = its share
//! of the max slot tokens + its similarity to the job; never the active domain, the window's top label or a slot
//! whose centroid cosine to either is >= `RELATED_COS`). Every hold and replace is logged to notes.jsonl with the
//! numbers. One off-topic burst never wipes or dilutes an established slot.
//! The old profiles/<model tag>/<label>.bin / .neuron / .vocab files are migrated into the store on open (renamed
//! *.migrated; block mass without per-neuron data is spread evenly over its block, so no count is lost).
//! RAM: no profile data in RAM: the per-turn activity score (centroid cosine), the learning (fold) and the plans stream
//! the store and the GPU recorder through fixed stack arrays (store::CH f32 + one record); `profiles` keeps two counters
//! per label.
use serde_json::{json, Value};
use std::collections::VecDeque;
#[cfg(test)]
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use streamdense_model::activity::ActivityStats;
use streamdense_model::unload::HARD_CAP;
#[cfg(test)]
use streamdense_model::unload;
use streamdense_model::LayerKind;

use crate::store::{BatchView, Batcher, JobSrc, PlanOut, Sel, Store, CH, MAXL};
pub use crate::store::Progress;
/// Read mask / prompt predictor / exploration, CLI side (src/predict.rs; declared here so main.rs stays untouched).
#[path = "predict.rs"]
pub mod predict;

/// Topics only (language is not a domain; the same topic in English or Vietnamese is one domain).
pub const DOMAINS: [&str; 6] = ["code-rust", "code-swift", "code-other", "agent-tools", "chat-general", "mixed"];
const CHAT: usize = 4;
/// Labels of the first watcher (language buckets), merged into chat-general when read (state, audits, profiles).
pub const OLD_CHAT: [&str; 3] = ["chat-en", "chat-vi", "chat-other"];
pub const ND: usize = DOMAINS.len();
pub const MIXED: usize = ND - 1;
/// the pool of turns whose own label had too small a margin (profile index; never an active domain)
pub const UNASSIGNED: usize = ND;
/// profiles: the domains (mixed slot stays empty) + unassigned
pub const NP: usize = ND + 1;
const NONE: usize = usize::MAX;
const SCAN_MAX: usize = 16 << 10;
const NOTE_MAX: usize = 2048;
const NOTES_KEEP: usize = 50;
const RECENT: usize = 20;
const DECISIONS_MAX_LINES: usize = 5000;
const DECISIONS_KEEP: usize = 2000;
/// Activity fingerprints vote once >= 2 domains have centroids from this many learned turns.
const CENT_MIN_TURNS: u64 = 3;
/// a slot whose centroid cosine to the active domain (or the window's top label) is at least this is "related": never
/// replaced
pub const RELATED_COS: f32 = 0.8;
const ACT_WEIGHT: f32 = 0.25;
/// Plans from fewer tokens are reported with ready = false.
pub const PLAN_MIN_TOKENS: u64 = 4096;
/// Profiles below this many tokens are not representative yet (curve report).
pub const REPRESENTATIVE_TOKENS: u64 = 200_000;
// audits: bounded queue / lists / files
const ID_MAX: usize = 128;
/// replies waiting for their audit (newest first; past this the oldest is dropped)
pub const AUDIT_QUEUE_MAX: usize = 64;
const AUDITS_KEEP: usize = 200;
const AUDITS_MAX_LINES: usize = 400;
const AUDITS_SENT: usize = 20;
pub const AUDIT_SKIP_REASON: &str = "dry run: no mask active, reply came from the full model";

pub fn now_s() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0) }
fn r3(x: f64) -> f64 { (x * 1000.0).round() / 1000.0 }

/// FNV-1a 64 -> 8 hex chars.
pub fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    format!("{:08x}", (h >> 32) as u32)
}

/// Longest prefix of `s` at most `max` bytes, on a char boundary.
pub fn trunc(s: &str, max: usize) -> &str {
    if s.len() <= max { return s; }
    let mut i = max;
    while !s.is_char_boundary(i) { i -= 1; }
    &s[..i]
}

// ---- features ------------------------------------------------------------------------------------------------------

/// Cheap text features (marker counts, line shapes, letters). Nothing of the text is kept.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Feat {
    pub rust: f32,
    pub swift: f32,
    pub other: f32,
    pub lines: u32,
    pub code_lines: u32,
    /// ``` fences (pairs, rounded up)
    pub fences: u32,
    pub letters: u32,
    pub tools: u32,
    pub tool_msgs: u32,
    /// client hint: 1 = code, -1 = chat, 0 = none
    pub mode: i8,
    /// state card / fold summary: its prose says nothing about the topic
    pub neutral: bool,
}

const RUST: [&str; 23] = ["fn ", "let mut ", "impl ", "pub fn", "pub struct", "&mut ", ".rs", "cargo ", "Cargo.toml", "unwrap()",
    "#[derive", "Vec<", "Option<", "Result<", "println!", "&self", "use std::", "crate::", "error[E", "rustc", "clippy", "Some(", "Ok(()"];
const SWIFT: [&str; 14] = ["func ", "guard ", ".swift", "SwiftUI", "import Foundation", "@State", "@Published", "xcodebuild",
    "var body", "some View", "Xcode", "DispatchQueue", "@MainActor", "NSWindow"];
// ambiguous markers left out: "):" (prose), "=> {" (Rust match arms), "const " (Rust too); ".js" is counted minus ".json"
const OTHER: [&str; 13] = ["def ", ".py", "python", ".ts", "function ", "#include", "console.log", "npm ", "pip ",
    "SELECT ", "package main", "public class", "print("];


impl Feat {
    /// Features of the first 16 KB of `text`.
    pub fn of(text: &str) -> Feat {
        let t = trunc(text, SCAN_MAX);
        let mut f = Feat::default();
        let n = |ms: &[&str]| ms.iter().map(|m| t.matches(m).count()).sum::<usize>() as f32;
        f.rust = n(&RUST) + 2.0 * t.matches("```rust").count() as f32;
        f.swift = n(&SWIFT) + 2.0 * t.matches("```swift").count() as f32;
        f.other = n(&OTHER) + t.matches(".js").count().saturating_sub(t.matches(".json").count()) as f32
            + 2.0 * ["```python", "```js", "```ts", "```c", "```go", "```java", "```sql"].iter().map(|m| t.matches(m).count()).sum::<usize>() as f32;
        f.fences = (t.matches("```").count() as u32).div_ceil(2);
        for l in t.lines() {
            let s = l.trim_end();
            if s.trim().is_empty() { continue; }
            f.lines += 1;
            let st = s.trim_start();
            let code = s.ends_with(['{', '}', ';', ')', ']', ',']) && !s.ends_with(". )") || st.starts_with("//") || st.starts_with("```")
                || st.starts_with("fn ") || st.starts_with("def ") || st.starts_with("let ") || st.starts_with("use ") || st.starts_with("import ")
                || s.starts_with("    ") || s.starts_with('\t') || st.starts_with("$ ");
            if code { f.code_lines += 1; }
        }
        for c in t.chars() {
            if !c.is_alphabetic() { continue; }
            f.letters += 1;
        }
        f
    }

    /// A tool result: code markers count double, its prose (cargo / ls output, file text) is language-neutral.
    pub fn tool_result(text: &str) -> Feat {
        let mut f = Feat::of(text);
        f.rust *= 2.0; f.swift *= 2.0; f.other *= 2.0;
        f.letters = 0;
        f
    }

    pub fn add(&mut self, o: &Feat) {
        self.rust += o.rust; self.swift += o.swift; self.other += o.other; self.lines += o.lines; self.code_lines += o.code_lines;
        self.fences += o.fences; self.letters += o.letters; self.tools = self.tools.max(o.tools);
        self.tool_msgs += o.tool_msgs; if o.mode != 0 { self.mode = o.mode; }
        self.neutral |= o.neutral;
    }

    /// Domain probabilities (mixed = 0). None = no signal (empty text, or a neutral card with prose only).
    pub fn probs(&self) -> Option<[f32; ND]> {
        let mut s = [0f32; ND];
        let hits = self.rust + self.swift + self.other;
        if self.letters == 0 && hits == 0.0 && self.tools == 0 && self.fences == 0 { return None; }
        let cl = if self.lines > 0 { self.code_lines as f32 / self.lines as f32 } else { 0.0 };
        let mut code = (hits / 6.0).min(1.0) * 0.6 + (cl * 1.5).min(1.0) * 0.4;
        // strong code evidence: a code block, or tools used on code (file reads / edits / cargo with code markers)
        if self.fences > 0 { code = code.max(0.6); }
        if self.tools > 0 && hits >= 1.0 { code = code.max(0.75); }
        if self.mode > 0 { code = code.max(0.6); } else if self.mode < 0 { code *= 0.5; }
        let (r, w, o) = if hits > 0.0 { (self.rust / hits, self.swift / hits, self.other / hits) } else { (0.0, 0.0, 1.0) };
        s[0] = code * r; s[1] = code * w; s[2] = code * o;
        let agentic = if self.tools > 0 { 0.3 + 0.05 * self.tool_msgs.min(6) as f32 } else { 0.0 };
        s[3] = agentic * (1.0 - 0.6 * code);
        let chat = (1.0 - code) * (if self.tools > 0 { 0.6 } else { 1.0 }) / (1.0 + 0.5 * self.tool_msgs.min(6) as f32);
        // prose in any language is chat; the language itself never picks the domain
        if self.letters > 0 && !self.neutral { s[CHAT] = chat; }
        let sum: f32 = s.iter().sum();
        if sum < 1e-6 { return None; }
        for x in s.iter_mut() { *x /= sum; }
        Some(s)
    }
}

/// Domains (mixed excluded) ranked by score, best first.
pub fn ranked(p: &[f32; ND]) -> Vec<(usize, f32)> {
    let mut c: Vec<(usize, f32)> = (0..MIXED).map(|d| (d, p[d])).collect();
    c.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    c
}

/// Top-2 -> (label, confidence): near-tie of two substantial domains = mixed.
#[cfg(test)]
pub fn label_of(p: &[f32; ND]) -> (usize, f32, Vec<(usize, f32)>) {
    let c = ranked(p);
    let ((d1, p1), (_, p2)) = (c[0], c[1]);
    if p1 - p2 < 0.1 && p2 >= 0.25 { (MIXED, (p1 + p2).min(1.0), c) } else { (d1, p1, c) }
}

// ---- model dims, activity, profiles, plans --------------------------------------------------------------------------

/// Model dims for profiles + plans: the `ActivityStats` template (dims, q4 layout, resident bytes) + block size.
#[derive(Clone, Debug)]
pub struct Dims {
    pub layers: usize,
    pub inter: usize,
    pub gran: usize,
    /// FFN blocks per layer = ceil(inter / gran)
    pub blocks: usize,
    /// most prunable heads of any layer (DeltaNet value heads / attention q heads)
    pub hmax: usize,
    pub tmpl: ActivityStats,
    /// model vocab for the per-label vocab files (0 = not recorded)
    pub vocab: usize,
}

impl Dims {
    /// `tmpl`: `ActivityStats::dims(cfg)` / `QwenEngine::activity_template` (or a snapshot: only dims / layout / byte
    /// fields are kept; every data vector is dropped, so each Dims copy is ~1 KB, not the 2 MB of vocab / MTP sums a
    /// snapshot carries: 2026-11-01 RAM fix).
    pub fn from_stats(t: &ActivityStats, gran: usize) -> Dims {
        let gran = gran.max(1);
        let hmax = t.kinds.iter().map(|k| match k { LayerKind::Attention => t.heads, LayerKind::DeltaNet => t.dn_vh }).max().unwrap_or(0);
        let tmpl = ActivityStats { enabled: true, layers: t.layers, hidden: t.hidden, inter: t.inter, kinds: t.kinds.clone(), dn_kh: t.dn_kh,
                                   dn_dk: t.dn_dk, dn_vh: t.dn_vh, dn_dv: t.dn_dv, heads: t.heads, kv_heads: t.kv_heads, head_dim: t.head_dim,
                                   conv_k: t.conv_k, group: t.group, bits: t.bits, weight_bytes: t.weight_bytes,
                                   resident_bytes: t.resident_bytes, ..ActivityStats::default() };
        Dims { layers: t.layers, inter: t.inter, gran, blocks: t.inter.div_ceil(gran), hmax, tmpl, vocab: 0 }
    }
    pub fn tag(&self) -> String { format!("L{}_I{}_g{}_H{}", self.layers, self.inter, self.gran, self.hmax) }
    pub fn block_neurons(&self, b: usize) -> usize { self.gran.min(self.inter - b * self.gran) }
}

/// Engine activity of one job, in profile form: per (layer, FFN block) contribution mass (Σ|a_j|·‖W_down[:, j]‖
/// summed over the block), per (layer, head) output-norm mass, the per-neuron contributions (empty when not
/// available) and the rows (tokens) it covers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Act {
    pub tokens: u64,
    /// layers x blocks
    pub ffn: Vec<f32>,
    /// layers x hmax
    pub heads: Vec<f32>,
    /// layers x inter (or empty)
    pub neuron: Vec<f32>,
    /// Output tokens emitted and embedding rows read during the job, sparse (id, count), ascending ids (the
    /// recorder's per-window counters, `ActivityTap::take_tokens`; empty without a recorder)
    pub out: Vec<(u32, u32)>,
    pub embed: Vec<(u32, u32)>,
}

impl Act {
    #[cfg(test)]
    pub fn zero(d: &Dims) -> Act { Act { tokens: 0, ffn: vec![0.0; d.layers * d.blocks], heads: vec![0.0; d.layers * d.hmax], ..Default::default() } }
    /// Activity stats -> profile form (None without data).
    #[cfg(test)]
    pub fn from_stats(s: &ActivityStats, d: &Dims) -> Option<Act> { Act::from_stats_norms(s.clone(), d, Some(&s.down_norm)) }
    #[cfg(test)]
    /// Same with the down norms borrowed (`ActivityTap::down_norms`; None or a wrong length = |a| sums only).
    pub fn from_stats_norms(mut s: ActivityStats, d: &Dims, down_norm: Option<&[f32]>) -> Option<Act> {
        if !s.enabled || s.layers != d.layers || s.inter != d.inter || s.neuron.len() != s.layers * s.inter { return None; }
        let mut a = Act::zero(d);
        a.tokens = s.rows;
        for l in 0..d.layers {
            let hm = s.head_mass(l);
            a.heads[l * d.hmax..l * d.hmax + hm.len()].copy_from_slice(hm);
        }
        let mut n = std::mem::take(&mut s.neuron);
        if let Some(dn) = down_norm.filter(|v| v.len() == n.len()) { for (x, w) in n.iter_mut().zip(dn) { *x *= w; } }
        drop(s);
        for l in 0..d.layers {
            for j in 0..d.inter { a.ffn[l * d.blocks + j / d.gran] += n[l * d.inter + j]; }
        }
        a.neuron = n;
        Some(a)
    }
    /// 1 bit per block: hotter than its layer's mean.
    #[cfg(test)]
    pub fn fingerprint(&self, d: &Dims) -> Vec<f32> {
        let mut fp = vec![0f32; self.ffn.len()];
        for l in 0..d.layers {
            let row = &self.ffn[l * d.blocks..(l + 1) * d.blocks];
            let mean = row.iter().sum::<f32>() / d.blocks.max(1) as f32;
            for (b, &x) in row.iter().enumerate() { if x > mean { fp[l * d.blocks + b] = 1.0; } }
        }
        fp
    }
}

/// An owned job (tests, migration) as a store job source: FFN = `neuron` (else the block masses spread evenly over
/// their neurons), heads into the recorder head order (resid 0), MTP 0, the sparse token counts. Does not borrow the store.
pub struct ActSrc<'a> { a: &'a Act, d: Dims, n_ffn: usize, o_mtp: usize, o_out: usize, o_emb: usize, n: usize, hr: Vec<f32> }
impl<'a> ActSrc<'a> {
    pub fn new(a: &'a Act, st: &Store) -> ActSrc<'a> {
        let d = st.d.clone();
        let mut hr = vec![0f32; st.n_hr()];
        if a.heads.len() == d.layers * d.hmax {
            for l in 0..d.layers { let r = st.heads_of(l); for (h, i) in r.clone().enumerate() { hr[i - st.n_ffn()] = a.heads[l * d.hmax + h]; } }
        }
        ActSrc { a, n_ffn: st.n_ffn(), o_mtp: st.o_mtp(), o_out: st.o_out(), o_emb: st.o_emb(), n: st.n(), hr, d }
    }
}
fn sparse_get(v: &[(u32, u32)], i: usize) -> f32 { v.binary_search_by_key(&(i as u32), |x| x.0).map_or(0.0, |k| v[k].1 as f32) }
impl JobSrc for ActSrc<'_> {
    fn rows(&self) -> (u64, u64) { (self.a.tokens, 0) }
    fn read(&self, off: usize, out: &mut [f32]) {
        let d = &self.d;
        for (j, o) in out.iter_mut().enumerate() {
            let g = off + j;
            *o = if g < self.n_ffn {
                if self.a.neuron.len() == self.n_ffn { self.a.neuron[g] } else {
                    let (l, b) = (g / d.inter, (g % d.inter) / d.gran);
                    self.a.ffn.get(l * d.blocks + b).copied().unwrap_or(0.0) / d.block_neurons(b) as f32
                }
            } else if g < self.o_mtp { self.hr[g - self.n_ffn] }
            else if g < self.o_out { 0.0 }
            else if g < self.o_emb { sparse_get(&self.a.out, g - self.o_out) }
            else if g < self.n { sparse_get(&self.a.embed, g - self.o_emb) } else { 0.0 };
        }
    }
}

/// One label's accumulated activity + fingerprint centroid. `windows` counts learned turns (file format unchanged).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub tokens: u64,
    pub windows: u64,
    pub ffn: Vec<f64>,
    pub heads: Vec<f64>,
    pub cent: Vec<f32>,
}

const MAGIC: &[u8; 8] = b"SDWPROF1";
pub const NMAGIC: &[u8; 8] = b"SDWNEUR1";
pub const NHEAD: usize = 8 + 4 * 8;

impl Profile {
    #[cfg(test)]
    pub fn zero(d: &Dims) -> Profile { Profile { tokens: 0, windows: 0, ffn: vec![0.0; d.layers * d.blocks], heads: vec![0.0; d.layers * d.hmax], cent: vec![0.0; d.layers * d.blocks] } }
    #[cfg(test)]
    fn learn(&mut self, a: &Act, d: &Dims) {
        for (x, y) in self.ffn.iter_mut().zip(&a.ffn) { *x += *y as f64; }
        for (x, y) in self.heads.iter_mut().zip(&a.heads) { *x += *y as f64; }
        self.tokens += a.tokens;
        self.windows += 1;
        let alpha = (1.0 / self.windows as f32).max(0.05);
        for (c, f) in self.cent.iter_mut().zip(a.fingerprint(d)) { *c += alpha * (f - *c); }
    }
    #[cfg(test)]
    pub fn to_bytes(&self, d: &Dims) -> Vec<u8> {
        let mut v = Vec::with_capacity(48 + self.ffn.len() * 12 + self.heads.len() * 8);
        v.extend_from_slice(MAGIC);
        for x in [d.layers as u64, d.blocks as u64, d.hmax as u64, self.tokens, self.windows] { v.extend_from_slice(&x.to_le_bytes()); }
        for x in &self.ffn { v.extend_from_slice(&x.to_le_bytes()); }
        for x in &self.heads { v.extend_from_slice(&x.to_le_bytes()); }
        for x in &self.cent { v.extend_from_slice(&x.to_le_bytes()); }
        v
    }
    pub fn from_bytes(b: &[u8], d: &Dims) -> Option<Profile> {
        let (nf, nh) = (d.layers * d.blocks, d.layers * d.hmax);
        if b.len() != 8 + 40 + nf * 12 + nh * 8 || &b[..8] != MAGIC { return None; }
        let u = |i: usize| u64::from_le_bytes(b[8 + i * 8..16 + i * 8].try_into().unwrap());
        if (u(0), u(1), u(2)) != (d.layers as u64, d.blocks as u64, d.hmax as u64) { return None; }
        let mut o = 48;
        let mut f64s = |n: usize| { let v: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(b[o + i * 8..o + i * 8 + 8].try_into().unwrap())).collect(); o += n * 8; v };
        let ffn = f64s(nf);
        let heads = f64s(nh);
        let cent = (0..nf).map(|i| f32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect();
        Some(Profile { tokens: u(3), windows: u(4), ffn, heads, cent })
    }
}

/// Per-neuron sums file: "SDWNEUR1", layers, inter, tokens, turns (u64 LE), then f32 LE [layers][inter].
/// Adds `v` (layers x inter) in a stream (old file -> tmp -> rename; 16 KB chunks, 64 KB buffers). A file of
/// other dims or size starts again from zero.
#[cfg(test)]
pub fn add_neuron_file(p: &Path, v: &[f32], layers: usize, inter: usize, tokens: u64) -> std::io::Result<()> {
    if v.len() != layers * inter { return Ok(()); }
    add_file_with(p, layers, inter, tokens, |i| v[i])
}

/// The stream behind both: `add(i)` is called for i = 0 .. layers x inter in order (64 KB buffers).
#[cfg(test)]
fn add_file_with(p: &Path, layers: usize, inter: usize, tokens: u64, mut add: impl FnMut(usize) -> f32) -> std::io::Result<()> {
    let n = layers * inter;
    if n == 0 { return Ok(()); }
    let want = (NHEAD + n * 4) as u64;
    let mut old = None;
    if let Ok(f) = std::fs::File::open(p) {
        if f.metadata().map(|m| m.len()).unwrap_or(0) == want {
            let mut r = std::io::BufReader::with_capacity(1 << 16, f);
            let mut h = [0u8; NHEAD];
            if r.read_exact(&mut h).is_ok() && &h[..8] == NMAGIC {
                let u = |i: usize| u64::from_le_bytes(h[8 + i * 8..16 + i * 8].try_into().unwrap());
                if (u(0), u(1)) == (layers as u64, inter as u64) { old = Some((r, u(2), u(3))); }
            }
        }
    }
    let tmp = p.with_extension("ntmp");
    let mut w = std::io::BufWriter::with_capacity(1 << 16, std::fs::File::create(&tmp)?);
    let (t0, n0) = old.as_ref().map(|o| (o.1, o.2)).unwrap_or((0, 0));
    w.write_all(NMAGIC)?;
    for x in [layers as u64, inter as u64, t0 + tokens, n0 + 1] { w.write_all(&x.to_le_bytes())?; }
    let mut buf = [0u8; 4 * 4096];
    for c0 in (0..n).step_by(4096) {
        let len = 4096.min(n - c0);
        let nb = len * 4;
        match old.as_mut() { Some(o) => o.0.read_exact(&mut buf[..nb])?, None => buf[..nb].fill(0) }
        for i in 0..len {
            let y = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap()) + add(c0 + i);
            buf[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes());
        }
        w.write_all(&buf[..nb])?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, p)
}

/// Read a per-neuron sums file: (sums [layers][inter], tokens, turns); None when missing or of other dims.
/// The same file opened for streaming (header checked, size = NHEAD + layers x inter x 4): (file, tokens, turns).
pub fn open_neuron_file(p: &Path, layers: usize, inter: usize) -> Option<(std::fs::File, u64, u64)> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(p).ok()?;
    if f.metadata().ok()?.len() != (NHEAD + layers * inter * 4) as u64 { return None; }
    let mut h = [0u8; NHEAD];
    f.read_exact_at(&mut h, 0).ok()?;
    let u = |i: usize| u64::from_le_bytes(h[8 + i * 8..16 + i * 8].try_into().unwrap());
    if &h[..8] != NMAGIC || (u(0), u(1)) != (layers as u64, inter as u64) { return None; }
    Some((f, u(2), u(3)))
}

/// Migration job source streamed from the old profile files (2026-11-01 RAM fix; same values as the in-RAM `ActSrc`
/// over the `Act` the old migration built, `migrate_stream_matches_in_ram`): FFN = the .neuron per-neuron sums read
/// per chunk + the .bin block mass not covered by them spread evenly (`add`, one f32 per block, NaN = none); heads
/// in the recorder head order; MTP 0; token counts = the rounded .vocab rows (count > 0).
pub struct OldSrc { neu: Option<std::fs::File>, voc: Option<std::fs::File>, add: Vec<f32>, hr: Vec<f32>, inter: usize, gran: usize, blocks: usize,
                    vocab: usize, n_ffn: usize, o_mtp: usize, o_out: usize, o_emb: usize, n: usize }
impl OldSrc {
    pub fn new(st: &Store, prof: Option<&Profile>, neu: Option<std::fs::File>, voc: Option<std::fs::File>) -> OldSrc {
        use std::os::unix::fs::FileExt;
        let d = &st.d;
        let mut add = vec![f32::NAN; if prof.is_some() { d.layers * d.blocks } else { 0 }];
        let mut hr = vec![0f32; st.n_hr()];
        if let Some(p) = prof {
            // per block: the .neuron mass (sequential f64 sum, one pass over the file in stack chunks), the rest spread
            let mut have = vec![0f64; d.layers * d.blocks];
            if let Some(f) = &neu {
                let mut b = [0u8; CH * 4];
                let n = d.layers * d.inter;
                let mut k = 0;
                while k < n {
                    let len = CH.min(n - k);
                    if f.read_exact_at(&mut b[..len * 4], (NHEAD + k * 4) as u64).is_err() { b[..len * 4].fill(0); }
                    for j in 0..len {
                        let g = k + j;
                        have[(g / d.inter) * d.blocks + (g % d.inter) / d.gran] += f32::from_le_bytes(b[j * 4..j * 4 + 4].try_into().unwrap()) as f64;
                    }
                    k += len;
                }
            }
            for l in 0..d.layers {
                for b in 0..d.blocks {
                    let i = l * d.blocks + b;
                    let rest = p.ffn[i] - have[i];
                    if rest > 0.0 { add[i] = (rest / d.block_neurons(b) as f64) as f32; }
                }
            }
            if p.heads.len() == d.layers * d.hmax {
                for l in 0..d.layers { for (h, i) in st.heads_of(l).enumerate() { hr[i - st.n_ffn()] = p.heads[l * d.hmax + h] as f32; } }
            }
        }
        OldSrc { neu, voc, add, hr, inter: d.inter, gran: d.gran, blocks: d.blocks, vocab: d.vocab, n_ffn: st.n_ffn(), o_mtp: st.o_mtp(),
                 o_out: st.o_out(), o_emb: st.o_emb(), n: st.n() }
    }
    /// f32 LE values [i, i + out.len()) of a streamed file (after its header) into `out`; 0 when missing / unreadable.
    fn file(f: Option<&std::fs::File>, i: usize, out: &mut [f32]) {
        use std::os::unix::fs::FileExt;
        let Some(f) = f else { out.fill(0.0); return };
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        if f.read_exact_at(bytes, (NHEAD + i * 4) as u64).is_err() { out.fill(0.0); return; }
        #[cfg(target_endian = "big")]
        for x in out.iter_mut() { *x = f32::from_bits(x.to_bits().swap_bytes()); }
    }
}
impl JobSrc for OldSrc {
    fn rows(&self) -> (u64, u64) { (0, 0) }
    fn read(&self, off: usize, out: &mut [f32]) {
        let end = off + out.len();
        // FFN
        let (a, b) = (off.min(self.n_ffn), end.min(self.n_ffn));
        if a < b {
            let o = &mut out[..b - a];
            OldSrc::file(self.neu.as_ref(), a, o);
            if !self.add.is_empty() {
                for (j, x) in o.iter_mut().enumerate() {
                    let g = a + j;
                    let e = self.add[(g / self.inter) * self.blocks + (g % self.inter) / self.gran];
                    if !e.is_nan() { *x += e; }
                }
            }
        }
        for g in off.max(self.n_ffn)..end {
            let o = &mut out[g - off];
            if g < self.o_mtp { *o = self.hr[g - self.n_ffn]; } else if g < self.o_out { *o = 0.0; }
        }
        // token counts: .vocab rows 0 (emitted) / 1 (embedding rows), rounded, > 0 only
        for (lo, hi, row) in [(self.o_out, self.o_emb, 0), (self.o_emb, self.n, 1)] {
            let (a, b) = (off.max(lo), end.min(hi));
            if a >= b { continue; }
            let o = &mut out[a - off..b - off];
            OldSrc::file(self.voc.as_ref(), row * self.vocab + (a - lo), o);
            for x in o.iter_mut() { *x = if *x > 0.0 { x.round() as u32 as f32 } else { 0.0 }; }
        }
        // past the layout
        if end > self.n { out[self.n.saturating_sub(off)..].fill(0.0); }
    }
}

#[cfg(test)]
pub fn read_neuron_file(p: &Path, layers: usize, inter: usize) -> Option<(Vec<f32>, u64, u64)> {
    let b = std::fs::read(p).ok()?;
    let n = layers * inter;
    if b.len() != NHEAD + n * 4 || &b[..8] != NMAGIC { return None; }
    let u = |i: usize| u64::from_le_bytes(b[8 + i * 8..16 + i * 8].try_into().unwrap());
    if (u(0), u(1)) != (layers as u64, inter as u64) { return None; }
    Some((b[NHEAD..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(), u(2), u(3)))
}

/// "What would unload" for one domain (streamed from the store: `Store::plan`, the `unload_plan` semantics).
/// Fixed arrays (no heap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub domain: &'static str,
    pub tokens: u64,
    /// tokens >= PLAN_MIN_TOKENS
    pub ready: bool,
    pub cold_ffn_frac: f64,
    pub cold_heads: usize,
    /// stage B (real unload): bytes freed
    pub gb_unload: f64,
    /// stage A (read mask, next step): pass time saved; stage B pass time saved
    pub ms_saved_per_pass: f64,
    pub ms_saved_unload: f64,
    /// resident after stage B (recorder estimate; current footprint - unload when the engine gave none)
    pub footprint_gb: f64,
    pub fits_cap: bool,
    pub layers: usize,
    /// cold FFN neurons per layer (aligned `granularity` blocks), cold heads per layer
    pub per_layer_cold: [u32; MAXL],
    pub per_layer_cold_heads: [u16; MAXL],
}

impl Default for Plan {
    fn default() -> Plan {
        Plan { domain: "", tokens: 0, ready: false, cold_ffn_frac: 0.0, cold_heads: 0, gb_unload: 0.0, ms_saved_per_pass: 0.0, ms_saved_unload: 0.0,
               footprint_gb: 0.0, fits_cap: true, layers: 0, per_layer_cold: [0; MAXL], per_layer_cold_heads: [0; MAXL] }
    }
}

impl Plan {
    pub fn cold(&self) -> &[u32] { &self.per_layer_cold[..self.layers] }
    pub fn cold_heads_per_layer(&self) -> &[u16] { &self.per_layer_cold_heads[..self.layers] }
    pub fn json(&self) -> Value {
        json!({"domain": self.domain, "tokens": self.tokens, "ready": self.ready, "cold_ffn_frac": r3(self.cold_ffn_frac),
            "cold_heads": self.cold_heads, "gb_unload": r3(self.gb_unload), "ms_saved_per_pass": r3(self.ms_saved_per_pass),
            "ms_saved_unload": r3(self.ms_saved_unload), "footprint_gb": r3(self.footprint_gb), "fits_cap": self.fits_cap,
            "per_layer_cold": self.cold(), "per_layer_cold_heads": self.cold_heads_per_layer()})
    }
    /// From a streamed plan (`footprint_gb` = the process now; used when the engine gave no resident estimate).
    pub fn from_out(domain: &'static str, po: &PlanOut, d: &Dims, footprint_gb: f64) -> Plan {
        let gb_unload = po.unload_bytes() as f64 / 1e9;
        let rb = d.tmpl.resident_bytes;
        let footprint_gb = if rb > 0 { rb.saturating_sub(po.unload_bytes()) as f64 / 1e9 } else { (footprint_gb - gb_unload).max(0.0) };
        Plan { domain, tokens: po.tokens, ready: po.tokens >= PLAN_MIN_TOKENS, cold_ffn_frac: po.cold_neurons as f64 / po.neurons.max(1) as f64,
               cold_heads: po.cold_heads, gb_unload, ms_saved_per_pass: po.ms_a(), ms_saved_unload: po.ms_b(), footprint_gb,
               fits_cap: footprint_gb * 1e9 <= HARD_CAP as f64, layers: po.layers, per_layer_cold: po.per_layer_cold, per_layer_cold_heads: po.per_layer_cold_heads }
    }
}

/// Profile counts -> activity stats: block mass spread evenly over its neurons (so `unload_plan` at granularity = gran
/// sees the same unit masses), no down norms (already folded in), heads back per kind.
#[cfg(test)]
pub fn stats_of(ffn: &[f64], heads: &[f64], tokens: u64, d: &Dims) -> ActivityStats {
    let mut s = d.tmpl.clone();
    s.rows = tokens;
    s.neuron = vec![0.0; d.layers * d.inter];
    s.dn_head = vec![0.0; s.n_deltanet() * s.dn_vh];
    s.attn_head = vec![0.0; s.n_attention() * s.heads];
    s.resid = vec![0.0; d.layers];
    let ok_f = ffn.len() == d.layers * d.blocks;
    let ok_h = heads.len() == d.layers * d.hmax;
    for l in 0..d.layers {
        if ok_f {
            for b in 0..d.blocks {
                let n = d.block_neurons(b);
                let v = (ffn[l * d.blocks + b] / n as f64) as f32;
                for x in &mut s.neuron[l * d.inter + b * d.gran..l * d.inter + b * d.gran + n] { *x = v; }
            }
        }
        if ok_h {
            let i = s.kind_index(l);
            let (dst, nh) = match s.kinds[l] { LayerKind::DeltaNet => (&mut s.dn_head, s.dn_vh), LayerKind::Attention => (&mut s.attn_head, s.heads) };
            for h in 0..nh { dst[i * nh + h] = heads[l * d.hmax + h] as f32; }
        }
    }
    s
}

/// The plan for a domain's block counts (`unload_plan` on `stats_of`; heap: the reference for `Store::plan`).
#[cfg(test)]
pub fn plan_for(domain: &'static str, ffn: &[f64], heads: &[f64], tokens: u64, d: &Dims, keep: f64, footprint_gb: f64) -> Plan {
    let s = stats_of(ffn, heads, tokens, d);
    let u = unload::unload_plan(&s, keep as f32, d.gran);
    let gb_unload = u.unload_bytes as f64 / 1e9;
    let footprint_gb = if u.resident_before > 0 { u.resident_after_b as f64 / 1e9 } else { (footprint_gb - gb_unload).max(0.0) };
    let mut p = Plan { domain, tokens, ready: tokens >= PLAN_MIN_TOKENS, cold_ffn_frac: u.cold_neurons as f64 / u.neurons.max(1) as f64,
           cold_heads: u.dn_heads_cold + u.attn_heads_cold, gb_unload, ms_saved_per_pass: u.pass_ms_saved_a, ms_saved_unload: u.pass_ms_saved_b,
           footprint_gb, fits_cap: footprint_gb * 1e9 <= u.cap as f64, layers: u.layers.len().min(MAXL), ..Default::default() };
    for (i, l) in u.layers.iter().take(MAXL).enumerate() { p.per_layer_cold[i] = l.cold_neurons as u32; p.per_layer_cold_heads[i] = l.cold_heads as u16; }
    p
}

// ---- the watcher ----------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WCfg {
    pub mode: String,
    /// decision window: folds needed (0 = turns only) and minimum turns
    pub folds: usize,
    pub min_turns: usize,
    /// disk fold every this many same-label jobs (batch in model memory; also on label change, idle, shutdown)
    pub fold_every: usize,
    /// similarity gate: a job below this similarity to its label's slot is held out of it (module doc)
    pub opp: f32,
    /// top-1 over top-2 lead needed (window decisions and per-turn learning)
    pub margin: f32,
    /// windows in a row
    pub hysteresis: usize,
    pub go_live: f64,
    /// Low-level watcher: sample period of the recorder buffer
    pub sample_ms: u64,
    pub keep_frac: f64,
    pub gran: usize,
    /// always-hot languages ("all-except-cut") and the always-"opposite" ones (unload-first; reserved)
    pub keep_languages: String,
    pub cut_languages: Vec<String>,
    /// Read mask: "off" | "dry" | "on", the profile window, the exploration share (predict.rs)
    pub mask: String,
    pub profile_window: String,
    pub explore_frac: f64,
}

impl WCfg {
    pub fn from_config(w: &streamdense_core::config::Watcher) -> WCfg {
        WCfg { mode: w.mode.clone(), folds: w.decide_window_folds, min_turns: w.decide_window_min_turns, fold_every: w.fold_every, opp: w.opp_similarity as f32, margin: w.margin as f32,
               hysteresis: w.hysteresis, go_live: w.go_live_points,
               sample_ms: w.sample_ms,
               keep_frac: w.keep_frac, gran: w.granularity, keep_languages: w.keep_languages.clone(), cut_languages: w.cut_languages.clone(),
               mask: w.mask.clone(), profile_window: w.profile_window.clone(), explore_frac: w.explore_frac }
    }
}

/// The open decision window: summed per-turn / per-note scores, counts, per-turn labels, tokens learned per label.
#[derive(Clone, Debug, Default, PartialEq)]
struct Win { sum: [f32; ND], n: u32, turns: u32, folds: u32, labels: [u32; NP], learned: [u64; NP], keys: Vec<String> }

impl Win {
    fn add(&mut self, p: &[f32; ND]) { for (s, x) in self.sum.iter_mut().zip(p) { *s += x; } self.n += 1; }
    fn mean(&self) -> Option<[f32; ND]> {
        if self.n == 0 { return None; }
        let mut p = self.sum;
        for x in p.iter_mut() { *x /= self.n as f32; }
        Some(p)
    }
    fn json(&self) -> Value {
        json!({"sum": self.sum.to_vec(), "n": self.n, "turns": self.turns, "folds": self.folds, "labels": self.labels.to_vec(),
               "learned": self.learned.to_vec(), "keys": self.keys})
    }
    fn from_json(v: &Value) -> Win {
        let mut w = Win { n: v["n"].as_u64().unwrap_or(0) as u32, turns: v["turns"].as_u64().unwrap_or(0) as u32,
                          folds: v["folds"].as_u64().unwrap_or(0) as u32, ..Default::default() };
        for (i, x) in v["sum"].as_array().into_iter().flatten().take(ND).enumerate() { w.sum[i] = x.as_f64().unwrap_or(0.0) as f32; }
        for (i, x) in v["labels"].as_array().into_iter().flatten().take(NP).enumerate() { w.labels[i] = x.as_u64().unwrap_or(0) as u32; }
        for (i, x) in v["learned"].as_array().into_iter().flatten().take(NP).enumerate() { w.learned[i] = x.as_u64().unwrap_or(0); }
        w.keys = v["keys"].as_array().into_iter().flatten().filter_map(|k| k.as_str().map(String::from)).take(8).collect();
        w
    }
    /// {label: n} of the non-zero entries
    fn named<T: Copy + PartialEq + Default + Into<Value>>(v: &[T]) -> Value {
        Value::Object(v.iter().enumerate().filter(|(_, x)| **x != T::default()).map(|(i, x)| (dname(i).to_string(), (*x).into())).collect())
    }
}

/// Go-live state of one profile (see the module doc). Audits only (no votes).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoLive {
    /// clean-audit points since the last reset
    pub points: f64,
    pub resets: u64,
    pub tightened: u64,
    pub clean: u64,
}

impl GoLive {
    /// One audit outcome: result match | mismatch | skipped. A mismatch under a mask is mask-caused by definition (the
    /// audit re-runs the same reply with the full model): reset + tighten.
    pub fn apply(&mut self, result: &str) {
        match result {
            "match" => { self.points += 1.0; self.clean += 1; }
            "mismatch" => { self.points = 0.0; self.resets += 1; self.tightened += 1; }
            _ => {} // skipped / pending: no mask was active, nothing to score
        }
    }
    pub fn required(&self, c: &WCfg) -> f64 { c.go_live }
    pub fn reached(&self, c: &WCfg) -> bool { self.points + 1e-9 >= c.go_live }
    fn json(&self) -> Value { json!({"points": self.points, "resets": self.resets, "tightened": self.tightened, "clean": self.clean}) }
    /// (old state files: "harm" / "model_limit" of the vote era are ignored)
    fn from_json(v: &Value) -> GoLive {
        let u = |k: &str| v[k].as_u64().unwrap_or(0);
        GoLive { points: v["points"].as_f64().unwrap_or(0.0), resets: u("resets"), tightened: u("tightened"), clean: u("clean") }
    }
}

pub struct Watcher {
    pub cfg: WCfg,
    pub dir: PathBuf,
    pub dims: Option<Dims>,
    pub domain: usize,
    pub since_turn: u64,
    pub turn: u64,
    pending: Option<(usize, usize)>,
    pub tokens_observed: u64,
    win: Win,
    /// mean scores of the last closed window
    last_win: Option<[f32; ND]>,
    /// the newest turn's own label (UNASSIGNED when its margin was too small) and margin
    last_turn: (usize, f32),
    cands: Vec<(usize, f32)>,
    recent: VecDeque<Value>,
    /// NP labels: the domains (mixed: only after a cap merge) + unassigned; counters only (the data is in `store`)
    pub profiles: Vec<Meta>,
    /// the disk store (None without dims)
    pub store: Option<Store>,
    /// per label: turns held out of its slot by the similarity gate (see the module doc)
    pub held: [Held; NP],
    /// fold notes seen (all time; the held turns' decision window counts folds from here)
    pub folds_total: u64,
    /// same-label jobs summed before one disk fold (None: every job folds straight to disk)
    pub batcher: Option<Arc<dyn Batcher>>,
    batch: Batch,
    /// cached plan per label (mixed / unknown share the union), dropped when a profile learns
    plans: Vec<Option<Plan>>,
    /// jobs that brought recorder activity
    pub act_jobs: u64,
    /// Replies waiting for their audit: (message id, the turn's own label, domain active when it was generated,
    /// turn), newest last; <= AUDIT_QUEUE_MAX (the oldest is dropped). Drained newest first in idle time.
    audit_q: VecDeque<(String, usize, usize, u64)>,
    /// Replies dropped from a full queue, audits done (this run)
    pub audit_dropped: u64,
    pub audit_done: u64,
    /// Recent audits (newest last, <= AUDITS_KEEP), JSON in the shape `audits_recent` sends
    audits: VecDeque<Value>,
    /// Go-live state per label
    pub golive: Vec<GoLive>,
    /// The read mask, the CPU prompt predictor and exploration (None: mask = off; predict.rs)
    pub mk: Option<Box<predict::MaskCtl>>,
}

fn dname(d: usize) -> &'static str { if d < ND { DOMAINS[d] } else if d == UNASSIGNED { "unassigned" } else { "unknown" } }
fn dindex(s: &str) -> usize {
    if s == "unassigned" { UNASSIGNED } else if OLD_CHAT.contains(&s) { CHAT } else { DOMAINS.iter().position(|d| *d == s).unwrap_or(NONE) }
}
/// labels that hold a profile: the domains but mixed, then unassigned
fn labels() -> impl Iterator<Item = usize> { (0..MIXED).chain([UNASSIGNED]) }
/// Store label names by index (the domains incl. mixed, then unassigned).
pub fn label_names() -> [&'static str; NP] { let mut n = [""; NP]; for (i, x) in n.iter_mut().enumerate() { *x = dname(i); } n }

/// A label's held turns (similarity gate): turns / tokens held, `fold0` = `folds_total` at the first hold, the last
/// similarity numbers, `open` = confirmed (the gate stays open until the slot is similar again).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Held { pub turns: u64, pub tokens: u64, pub fold0: u64, pub sim: f32, pub open: bool }
impl Held {
    fn json(&self) -> Value { json!({"turns": self.turns, "tokens": self.tokens, "fold0": self.fold0, "sim": r3(self.sim as f64), "open": self.open}) }
    fn from_json(v: &Value) -> Held {
        Held { turns: v["turns"].as_u64().unwrap_or(0), tokens: v["tokens"].as_u64().unwrap_or(0), fold0: v["fold0"].as_u64().unwrap_or(0),
               sim: v["sim"].as_f64().unwrap_or(0.0) as f32, open: v["open"].as_bool().unwrap_or(false) }
    }
}

/// Where a learned job goes: its label's slot, the held slot (owner label), or the pooled now / session only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target { Slot(usize), Held(usize), Pooled }
impl Target {
    /// the store set (None = pooled only)
    pub fn set(&self) -> Option<usize> { match self { Target::Slot(l) => Some(*l), Target::Held(_) => Some(crate::store::HELD), Target::Pooled => None } }
}
/// The open batch (sums in the batcher; counters here): one target, `jobs` jobs, rows, the rows-weighted job time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Batch { pub target: Option<Target>, pub jobs: u64, pub rows: u64, pub mtp_rows: u64, pub t_w: f64, pub w: f64, pub t_last: f64, pub turn: u64 }

/// A label's counters (the data lives in the store).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Meta { pub tokens: u64, pub windows: u64 }

/// `~/x` -> $HOME/x.
pub fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) { (Some(r), Some(h)) => PathBuf::from(h).join(r), _ => PathBuf::from(p) }
}

/// Env var that points the watcher at another directory for one run (tests, smoke runs): `--watcher-dir` / `--dir`
/// > `$SD_WATCHER_DIR` > `[watcher] dir`. Inherited through `streamdense up` (sd-watchdog + nice exec the gateway).
pub const WATCHER_DIR_ENV: &str = "SD_WATCHER_DIR";

/// The watcher directory without a flag: `$SD_WATCHER_DIR` when set (non-empty), else `[watcher] dir` (`~/` expanded).
pub fn config_dir(cfg_dir: &str) -> PathBuf {
    match std::env::var(WATCHER_DIR_ENV) { Ok(d) if !d.is_empty() => expand_home(&d), _ => expand_home(cfg_dir) }
}

fn write_atomic(p: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, p)
}

/// Append a line; past `max` lines keep the last `keep` (atomic rewrite).
fn append_compact(p: &Path, line: &str, max: usize, keep: usize) {
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) { let _ = writeln!(f, "{}", line); }
    let Ok(t) = std::fs::read_to_string(p) else { return };
    let n = t.lines().count();
    if n > max {
        let tail: Vec<&str> = t.lines().skip(n - keep).collect();
        let _ = write_atomic(p, (tail.join("\n") + "\n").as_bytes());
    }
}

fn ktok(n: u64) -> String { if n >= 1000 { format!("{:.1}K", n as f64 / 1e3) } else { n.to_string() } }

impl Watcher {
    /// Open (or create) the watcher state in `dir`. `dims` None = no engine activity (text features only).
    #[cfg(test)]
    pub fn open(dir: PathBuf, cfg: WCfg, dims: Option<Dims>) -> Result<Watcher, String> { Watcher::open_opts(dir, cfg, dims, true) }

    /// `open` with `store` = false: the neuron store (`store/<tag>_V<vocab>/store.bin`) and the predictor table
    /// (`predict.bin` next to it) are neither opened nor created. For a gateway without the activity recorder (12 GB mode,
    /// `--no-activity`): nothing would ever be written there. The text watcher (state, decisions, audits) runs as usual.
    pub fn open_opts(dir: PathBuf, cfg: WCfg, dims: Option<Dims>, store: bool) -> Result<Watcher, String> {
        std::fs::create_dir_all(&dir).map_err(|e| format!("watcher dir {}: {}", dir.display(), e))?;
        let mut w = Watcher { cfg, dir, dims, domain: NONE, since_turn: 0, turn: 0, pending: None, tokens_observed: 0, win: Win::default(),
            last_win: None, last_turn: (UNASSIGNED, 0.0), cands: vec![], recent: VecDeque::new(), profiles: vec![], store: None, held: [Held::default(); NP], folds_total: 0, batcher: None, batch: Batch::default(),
            plans: vec![None; NP], act_jobs: 0, audit_q: VecDeque::new(), audit_dropped: 0, audit_done: 0, audits: VecDeque::new(), golive: vec![GoLive::default(); NP], mk: None };
        w.load_audits();
        if let Ok(t) = std::fs::read_to_string(w.dir.join("state.json")) {
            if let Ok(s) = serde_json::from_str::<Value>(&t) {
                w.domain = dindex(s["domain"].as_str().unwrap_or(""));
                if w.domain == UNASSIGNED { w.domain = NONE; }
                w.since_turn = s["since_turn"].as_u64().unwrap_or(0);
                w.turn = s["turn"].as_u64().unwrap_or(0);
                w.tokens_observed = s["tokens_observed"].as_u64().unwrap_or(0);
                if let (Some(d), Some(n)) = (s["pending"][0].as_str(), s["pending"][1].as_u64()) { if dindex(d) < ND { w.pending = Some((dindex(d), n as usize)); } }
                if s["window"].is_object() { w.win = Win::from_json(&s["window"]); }
                if let Some(a) = s["last_window"].as_array().filter(|a| a.len() == ND) {
                    let mut p = [0f32; ND];
                    for (i, x) in a.iter().enumerate() { p[i] = x.as_f64().unwrap_or(0.0) as f32; }
                    w.last_win = Some(p);
                }
                if let Some(g) = s["golive"].as_object() { for (k, v) in g { let i = dindex(k); if i < NP { w.golive[i] = GoLive::from_json(v); } } }
                if let Some(h) = s["held"].as_object() { for (k, v) in h { let i = dindex(k); if i < NP { w.held[i] = Held::from_json(v); } } }
                w.folds_total = s["folds_total"].as_u64().unwrap_or(0);
            }
        }
        if let Ok(t) = std::fs::read_to_string(w.dir.join("decisions.jsonl")) {
            let ls: Vec<&str> = t.lines().collect();
            for l in &ls[ls.len().saturating_sub(RECENT)..] { if let Ok(v) = serde_json::from_str::<Value>(l) { w.recent.push_back(v); } }
        }
        if let Some(d) = w.dims.clone().filter(|_| store) {
            let st = Store::open(&w.dir, d.clone(), label_names(), MIXED, false).map_err(|e| format!("watcher store in {}: {}", w.dir.display(), e))?;
            w.store = Some(st);
            w.migrate_old(&d);
            w.reconcile_held();
            let c = &w.cfg;
            w.mk = w.store.as_ref().and_then(|st| predict::MaskCtl::open(st, &c.mask, &c.profile_window, c.keep_frac, c.explore_frac));
        }
        w.load_meta();
        Ok(w)
    }

    /// The held slot and the held counters agree after a restart: a held slot whose owner has no held turns is dropped;
    /// held counters of a label that does not own the held slot are reset (its jobs are in now / session only).
    fn reconcile_held(&mut self) {
        let Some(st) = self.store.as_mut() else { return };
        let owner = st.held_owner();
        if let Some(o) = owner { if self.held[o].turns == 0 { let _ = st.drop_held(); } }
        for l in 0..NP { if owner != Some(l) && self.held[l].turns > 0 { self.held[l] = Held { open: self.held[l].open, ..Held::default() }; } }
    }
    /// Counters per label from the store headers.
    fn load_meta(&mut self) {
        self.profiles = (0..NP).map(|i| self.store.as_ref().map(|s| { let h = s.hdr(i); Meta { tokens: h.tokens, windows: h.turns } }).unwrap_or_default()).collect();
    }

    /// One-time migration (2026-11-01, heat on disk) of the old profiles/<tag>/ files into the store: per label the
    /// .neuron per-neuron sums + the .bin block mass not covered by them (spread evenly over the block's neurons), the
    /// .bin heads, tokens, turns and centroid, the .vocab counts. chat-en / chat-vi / chat-other go into chat-general.
    /// Originals are renamed *.migrated (kept). Setup only; streamed (2026-11-01 RAM fix): the .neuron / .vocab files
    /// are read in `CH` chunks straight into the fold's stack chunk (`OldSrc`), never whole into RAM; host heap = the
    /// .bin profile (233 KB on the 27B) + per-block adds (layers x blocks f32, 70 KB) while one label migrates.
    fn migrate_old(&mut self, d: &Dims) {
        let pd = self.prof_dir();
        if !pd.is_dir() { return; }
        let names: Vec<(&str, usize)> = (0..MIXED).chain([UNASSIGNED]).map(|i| (dname(i), i)).chain(OLD_CHAT.iter().map(|n| (*n, CHAT))).collect();
        let Some(st) = self.store.as_mut() else { return };
        let mut done = vec![];
        for (name, to) in names {
            let (pb, pn, pv) = (pd.join(format!("{}.bin", name)), pd.join(format!("{}.neuron", name)), pd.join(format!("{}.vocab", name)));
            let prof = std::fs::read(&pb).ok().and_then(|b| Profile::from_bytes(&b, d));
            let neu = open_neuron_file(&pn, d.layers, d.inter);
            let voc = if d.vocab > 0 { open_neuron_file(&pv, 2, d.vocab) } else { None };
            if prof.is_none() && neu.is_none() && voc.is_none() { continue; }
            let (tokens, turns) = match (&prof, &neu) { (Some(p), _) => (p.tokens, p.windows), (None, Some(n)) => (n.1, n.2), _ => (0, 0) };
            let src = OldSrc::new(st, prof.as_ref(), neu.map(|n| n.0), voc.map(|v| v.0));
            if let Err(e) = st.add_raw(to, &src, tokens, turns, prof.as_ref().map(|p| p.cent.as_slice())) { eprintln!("watcher: migrate {}: {}", name, e); return; }
            drop(src);
            for p in [&pb, &pn, &pv] { if p.exists() { let _ = std::fs::rename(p, p.with_extension(format!("{}.migrated", p.extension().unwrap().to_string_lossy()))); } }
            done.push(format!("{} -> {} ({} tokens)", name, dname(to), tokens));
        }
        if !done.is_empty() { eprintln!("watcher: migrated into the store: {}", done.join(", ")); }
    }

    pub fn prof_dir(&self) -> PathBuf { self.dir.join("profiles").join(self.dims.as_ref().map(|d| d.tag()).unwrap_or_else(|| "none".into())) }
    pub fn domain_name(&self) -> &'static str { dname(self.domain) }

    fn save_state(&self) {
        let g: serde_json::Map<String, Value> = (0..NP).filter(|&i| self.golive[i] != GoLive::default()).map(|i| (dname(i).to_string(), self.golive[i].json())).collect();
        let s = json!({"domain": dname(self.domain), "since_turn": self.since_turn, "turn": self.turn, "tokens_observed": self.tokens_observed,
            "pending": self.pending.map(|(d, n)| json!([dname(d), n])), "model": self.dims.as_ref().map(|d| d.tag()),
            "window": self.win.json(), "last_window": self.last_win.map(|p| p.to_vec()), "golive": g,
            "held": self.held_json(), "folds_total": self.folds_total});
        let _ = write_atomic(&self.dir.join("state.json"), s.to_string().as_bytes());
    }

    /// A turn's own scores: rules blended with the activity centroids (cosine streamed from the store; no heap).
    pub fn turn_scores_src(&self, f: &Feat, src: Option<&dyn JobSrc>) -> Option<[f32; ND]> {
        let mut p = f.probs()?;
        if let (Some(s), Some(st)) = (src, self.store.as_ref()) {
            let mut use_l = [false; NP];
            for d in 0..MIXED { use_l[d] = self.profiles[d].windows >= CENT_MIN_TURNS; }
            if use_l.iter().filter(|x| **x).count() >= 2 {
                let cos = st.score(s, &use_l);
                let (mut e, mut z) = ([0f32; NP], 0f32);
                for d in 0..MIXED { if use_l[d] { e[d] = (cos[d] * 8.0).exp(); z += e[d]; } }
                for x in p.iter_mut() { *x *= 1.0 - ACT_WEIGHT; }
                for d in 0..MIXED { if use_l[d] { p[d] += ACT_WEIGHT * e[d] / z; } }
            }
        }
        Some(p)
    }
    /// Same for an owned job (tests).
    #[cfg(test)]
    pub fn turn_scores(&self, f: &Feat, act: Option<&Act>) -> Option<[f32; ND]> {
        let src = match (act, self.store.as_ref()) { (Some(a), Some(st)) => Some(ActSrc::new(a, st)), _ => None };
        self.turn_scores_src(f, src.as_ref().map(|s| s as &dyn JobSrc))
    }

    fn act_ok(&self, a: &Act) -> bool { self.dims.as_ref().is_some_and(|d| a.ffn.len() == d.layers * d.blocks && a.heads.len() == d.layers * d.hmax) }

    /// One /v1/messages job (or a "turn" note) with an owned job activity (tests).
    pub fn observe(&mut self, f: &Feat, tokens: u64, act: Option<Act>, key: &str) -> Option<Value> {
        let act = act.filter(|a| self.act_ok(a));
        let src = match (&act, self.store.as_ref()) { (Some(a), Some(st)) => Some(ActSrc::new(a, st)), _ => None };
        self.observe_src(f, tokens, src.as_ref().map(|s| s as &dyn JobSrc), key)
    }
    /// One job: features, tokens through the engine, engine activity (the GPU recorder in place: `store::TapSrc`).
    pub fn observe_src(&mut self, f: &Feat, tokens: u64, src: Option<&dyn JobSrc>, key: &str) -> Option<Value> {
        let p = self.turn_scores_src(f, src);
        self.observe_core(p, tokens, src, key)
    }
    /// `observe` after scoring (owned job activity; tests).
    #[cfg(test)]
    fn observe_scores(&mut self, p: Option<[f32; ND]>, tokens: u64, act: Option<Act>, key: &str) -> Option<Value> {
        let act = act.filter(|a| self.act_ok(a));
        let src = match (&act, self.store.as_ref()) { (Some(a), Some(st)) => Some(ActSrc::new(a, st)), _ => None };
        self.observe_core(p, tokens, src.as_ref().map(|s| s as &dyn JobSrc), key)
    }

    /// Label the turn (own top label if it leads by >= margin, else unassigned), learn its activity into that label,
    /// add it to the window; decide when the window is full.
    fn observe_core(&mut self, p: Option<[f32; ND]>, tokens: u64, src: Option<&dyn JobSrc>, key: &str) -> Option<Value> {
        if !key.is_empty() && !self.win.keys.iter().any(|k| k == key) && self.win.keys.len() < 8 { self.win.keys.push(key.into()); }
        self.turn += 1;
        self.tokens_observed += tokens;
        let (label, margin) = match p {
            Some(p) => { let c = ranked(&p); let m = c[0].1 - c[1].1; (if m >= self.cfg.margin { c[0].0 } else { UNASSIGNED }, m) }
            None => (UNASSIGNED, 0.0),
        };
        if let Some(p) = p { self.win.add(&p); }
        self.win.turns += 1;
        self.win.labels[label] += 1;
        if let Some(a) = src {
            self.act_jobs += 1;
            self.win.learned[label] += a.rows().0;
            self.learn(label, a);
        }
        self.last_turn = (label, margin);
        let r = if self.window_full() { self.decide("turns") } else { None };
        self.save_state();
        r
    }

    /// Learn one turn's activity into `label` (a domain or unassigned) through the similarity gate (module doc), then
    /// into the batch (`accumulate`). Held jobs go to the held slot (one owner label at a time; others wait in now /
    /// session only, logged); on confirm the held slot is added into the label's slot (`Store::move_held`), so no
    /// held turn is lost. Streamed, no heap on the fold path (notes are written only on hold / confirm / replace).
    fn learn(&mut self, label: usize, src: &dyn JobSrc) {
        if label >= NP || label == MIXED || self.store.is_none() { return; }
        let (turn, t, opp) = (self.turn, now_s(), self.cfg.opp);
        let (rows, _) = src.rows();
        let protect = self.protected();
        let (slot, sim, owner, free) = {
            let st = self.store.as_ref().unwrap();
            (st.slot_of(label), st.sim(label, src), st.held_owner(), st.free_slot())
        };
        let mut h = self.held[label];
        let below = label != UNASSIGNED && sim.is_some_and(|x| x.min() < opp);
        let mut notes: [Option<Value>; 2] = [None, None];
        // a similar job ends a hold / an open gate (the opposite profile did not persist, or the slot caught up); the
        // held jobs stay in now / session only (never diluting the slot)
        if slot.is_some() && sim.is_some() && !below && h != Held::default() {
            if owner == Some(label) && h.turns > 0 {
                self.flush_batch();
                if let Err(e) = self.store.as_mut().unwrap().drop_held() { eprintln!("watcher: store drop held: {}", e); }
                notes[1] = Some(json!({"ts": t, "kind": "hold-ended", "label": dname(label), "turn": turn, "held_turns": h.turns, "held_tokens": h.tokens,
                    "text": format!("{}: a similar turn ended the hold; {} held turns kept in now / session only", dname(label), h.turns)}));
            }
            h = Held::default();
        }
        let hold = (slot.is_some() && below && !h.open) || (slot.is_none() && free.is_none());
        let mut target = Target::Slot(label);
        if hold {
            let simj = sim.map(|x| json!({"min": r3(x.min() as f64), "blocks": r3(x.blocks as f64), "heads": r3(x.heads as f64)}));
            if owner.is_some_and(|o| o != label) {
                target = Target::Pooled;
                notes[0] = Some(json!({"ts": t, "kind": "hold-wait", "label": dname(label), "turn": turn, "held_slot": dname(owner.unwrap()), "sim": simj,
                    "text": format!("{} turn {} held in now / session only: the held slot is busy with {}", dname(label), turn, dname(owner.unwrap()))}));
            } else {
                if owner.is_none() {
                    self.flush_batch();
                    if let Err(e) = self.store.as_mut().unwrap().set_held_owner(label) { eprintln!("watcher: store held slot: {}", e); }
                }
                if h.turns == 0 { h.fold0 = self.folds_total; }
                h.turns += 1;
                h.tokens += rows;
                h.sim = sim.map_or(0.0, |x| x.min());
                let need = self.cfg.min_turns as u64;
                let confirmed = h.turns >= need.max(1) && (self.folds_total - h.fold0 >= self.cfg.folds as u64 || h.turns >= (2 * need).max(1));
                if !confirmed {
                    target = Target::Held(label);
                    notes[0] = Some(json!({"ts": t, "kind": "hold", "label": dname(label), "turn": turn, "held_turns": h.turns, "held_tokens": h.tokens,
                        "folds_since": self.folds_total - h.fold0, "opp_similarity": opp, "reason": if slot.is_some() { "opposite to its slot" } else { "no free slot" },
                        "sim": simj, "text": format!("held {} turn {} out of its slot ({}; {} held turns)", dname(label), turn,
                            sim.map_or("no slot".into(), |x| format!("similarity {:.3} < {:.2}", x.min(), opp)), h.turns)}));
                } else {
                    // confirmed: the held jobs (batch flushed first) go into the label's slot, then this job folds there
                    self.flush_batch();
                    let held = h;
                    let st = self.store.as_mut().unwrap();
                    let mut ok = slot.is_some();
                    if slot.is_none() {
                        // replace the least-used AND most-dissimilar unprotected slot
                        let maxtok = (0..st.nslots()).filter_map(|i| st.slot_label(i)).map(|v| st.hdr(v).tokens).max().unwrap_or(0).max(1);
                        let mut best: Option<(f32, usize, usize, f32, u64)> = None;
                        for i in 0..st.nslots() {
                            let Some(v) = st.slot_label(i) else { continue };
                            if protect.iter().any(|&p| p == v) || protect.iter().any(|&p| p < NP && p != label && st.cent_cos(v, p) >= RELATED_COS) { continue; }
                            let tok = st.hdr(v).tokens;
                            let sv = st.sim(v, src).map_or(0.0, |x| x.min());
                            let score = tok as f32 / maxtok as f32 + sv;
                            if best.is_none_or(|b| score < b.0) { best = Some((score, i, v, sv, tok)); }
                        }
                        match best {
                            Some((score, i, v, sv, tok)) => match st.claim(i, label) {
                                Ok(_) => {
                                    ok = true;
                                    self.profiles[v] = Meta::default();
                                    self.golive[v] = GoLive::default();
                                    notes[1] = Some(json!({"ts": t, "kind": "replace", "label": dname(label), "replaced": dname(v), "slot": i, "turn": turn,
                                        "held_turns": held.turns, "held_tokens": held.tokens, "replaced_tokens": tok, "replaced_usage": r3(tok as f64 / maxtok as f64),
                                        "replaced_sim": r3(sv as f64), "score": r3(score as f64), "protected": protect.iter().filter(|&&p| p < NP).map(|&p| dname(p)).collect::<Vec<_>>(),
                                        "text": format!("slot {}: {} replaced by {} (held {} turns moved in; {} usage {:.3}, similarity {:.3})", i, dname(v), dname(label), held.turns, dname(v), tok as f64 / maxtok as f64, sv)}));
                                }
                                Err(e) => eprintln!("watcher: store replace: {}", e),
                            },
                            None => {
                                target = Target::Held(label);
                                notes[0] = Some(json!({"ts": t, "kind": "hold", "label": dname(label), "turn": turn, "held_turns": held.turns, "reason": "confirmed but every slot is protected",
                                    "text": format!("held {}: confirmed, but every slot is the active / hot domain or related to it", dname(label))}));
                            }
                        }
                    }
                    if ok {
                        match st.move_held(label) {
                            Ok(()) => {
                                h = Held { open: slot.is_some(), ..Held::default() };
                                if slot.is_some() {
                                    notes[0] = Some(json!({"ts": t, "kind": "hold-confirmed", "label": dname(label), "turn": turn, "held_turns": held.turns, "held_tokens": held.tokens,
                                        "sim": r3(held.sim as f64), "text": format!("{}: the opposite profile persisted {} turns; held turns moved into its slot", dname(label), held.turns)}));
                                }
                            }
                            Err(e) => { eprintln!("watcher: store move held: {}", e); target = Target::Pooled; }
                        }
                    }
                }
            }
        }
        self.held[label] = h;
        // a label without a slot and a free one: claim it (batch flushed first: slot changes never cross a batch)
        if target == Target::Slot(label) && self.store.as_ref().unwrap().slot_of(label).is_none() {
            self.flush_batch();
            let st = self.store.as_mut().unwrap();
            match st.free_slot() { Some(i) => { if let Err(e) = st.claim(i, label) { eprintln!("watcher: store claim: {}", e); target = Target::Pooled; } } None => target = Target::Pooled }
        }
        self.accumulate(target, src, turn, t);
        for n in notes.into_iter().flatten() {
            eprintln!("watcher: {}", n["text"].as_str().unwrap_or(""));
            append_compact(&self.dir.join("notes.jsonl"), &n.to_string(), 2 * NOTES_KEEP, NOTES_KEEP);
        }
        self.plans.fill(None);
    }

    /// One job into its target: the centroid now (per job, small), the sums into the batch (one disk fold per batch:
    /// target change, `fold_every` jobs, idle, shutdown) or, without a batcher, straight to disk. No heap.
    fn accumulate(&mut self, target: Target, src: &dyn JobSrc, turn: u64, t: f64) {
        let b = self.batcher.clone();
        if b.is_some() && self.batch.jobs > 0 && (self.batch.target != Some(target) || self.batch.jobs >= self.cfg.fold_every.max(1) as u64) { self.flush_batch(); }
        let pending = self.batch.jobs;
        let Some(st) = self.store.as_mut() else { return };
        if let Some(s) = target.set() { if let Err(e) = st.update_cent(s, src, pending) { eprintln!("watcher: store centroid: {}", e); } }
        match b {
            Some(b) => {
                b.add(src);
                let (r, m) = src.rows();
                let w = r.max(1) as f64;
                let bt = &mut self.batch;
                bt.target = Some(target);
                bt.jobs += 1;
                (bt.rows, bt.mtp_rows, bt.t_w, bt.w, bt.turn) = (bt.rows + r, bt.mtp_rows + m, bt.t_w + t * w, bt.w + w, turn);
                bt.t_last = t;
                if bt.jobs >= self.cfg.fold_every.max(1) as u64 { self.flush_batch(); }
            }
            None => {
                if let Err(e) = st.fold_to(target.set(), src, 1, turn, t) { eprintln!("watcher: store fold: {}", e); }
                self.refresh_meta(target);
            }
        }
    }
    /// Fold the batch to disk (one write of its target + now / session, at the rows-weighted mean job time), then
    /// zero it. A crash before this loses at most the batch (model memory), never corrupts the store.
    pub fn flush_batch(&mut self) {
        if self.batch.jobs == 0 { return; }
        let bt = std::mem::take(&mut self.batch);
        let (Some(b), Some(st)) = (self.batcher.clone(), self.store.as_mut()) else { return };
        let view = BatchView { b: &*b, rows: (bt.rows, bt.mtp_rows) };
        let target = bt.target.unwrap_or(Target::Pooled);
        if let Err(e) = st.fold_to(target.set(), &view, bt.jobs, bt.turn, bt.t_w / bt.w.max(1e-9)) { eprintln!("watcher: store batch fold: {}", e); }
        b.clear();
        self.refresh_meta(target);
    }
    /// Flush a batch whose newest job is >= `idle_s` old (the sd-heat thread, idle).
    pub fn flush_if_idle(&mut self, t: f64, idle_s: f64) -> bool {
        if self.batch.jobs > 0 && t - self.batch.t_last >= idle_s { self.flush_batch(); true } else { false }
    }
    /// The open batch: {target, jobs, rows, fold_every}.
    pub fn batch_json(&self) -> Value {
        let b = &self.batch;
        let tg = match b.target { Some(Target::Slot(l)) => json!(dname(l)), Some(Target::Held(l)) => json!(format!("held:{}", dname(l))), Some(Target::Pooled) => json!("pooled"), None => Value::Null };
        json!({"target": tg, "jobs": b.jobs, "rows": b.rows, "fold_every": self.cfg.fold_every, "batched": self.batcher.is_some()})
    }
    fn refresh_meta(&mut self, target: Target) {
        if let (Target::Slot(l), Some(st)) = (target, self.store.as_ref()) { let h = st.hdr(l); self.profiles[l] = Meta { tokens: h.tokens, windows: h.turns }; }
    }

    /// Labels whose slots are never replaced: the active domain and the open window's top label.
    fn protected(&self) -> [usize; 2] {
        let top = (0..NP).filter(|&i| self.win.labels[i] > 0).max_by_key(|&i| self.win.labels[i]).unwrap_or(NONE);
        [self.domain, top]
    }
    /// The store's slot table: [{slot, label, tokens, turns, bytes}] (free slots: label null).
    pub fn slots_json(&self) -> Value {
        let Some(st) = self.store.as_ref() else { return json!([]) };
        Value::Array((0..st.nslots()).map(|i| match st.slot_label(i) {
            Some(v) => { let h = st.hdr(v); json!({"slot": i, "label": dname(v), "tokens": h.tokens, "turns": h.turns, "bytes": st.lsz()}) }
            None => json!({"slot": i, "label": null, "bytes": st.lsz()}),
        }).collect())
    }
    fn held_json(&self) -> Value {
        Value::Object((0..NP).filter(|&i| self.held[i] != Held::default()).map(|i| (dname(i).to_string(), self.held[i].json())).collect())
    }

    /// A client note. state_card / fold: kept (text truncated to 2 KB), scored into the window (prose
    /// neutral); a fold counts toward the window's folds. turn: observed like a job.
    pub fn note(&mut self, kind: &str, text: &str, project: &str, mode: &str) -> Result<Option<Value>, String> {
        if !["state_card", "fold", "turn"].contains(&kind) { return Err(format!("kind {:?}: want state_card | fold | turn", kind)); }
        let mut f = Feat::of(text);
        f.mode = match mode { "code" => 1, "chat" => -1, "" => 0, m => return Err(format!("mode {:?}: want chat | code", m)) };
        if kind == "turn" { return Ok(self.observe(&f, 0, None, "")); }
        let rec = json!({"ts": now_s(), "kind": kind, "project": trunc(project, 256), "mode": mode, "text": trunc(text, NOTE_MAX)});
        append_compact(&self.dir.join("notes.jsonl"), &rec.to_string(), 2 * NOTES_KEEP, NOTES_KEEP);
        f.neutral = true;
        if let Some(p) = f.probs() { self.win.add(&p); }
        if kind == "fold" { self.win.folds += 1; self.folds_total += 1; }
        let r = if self.window_full() { self.decide(kind) } else { None };
        self.save_state();
        Ok(r)
    }

    /// The window has a signal, >= min_turns turns, and the folds (or 2 x min_turns turns without them).
    fn window_full(&self) -> bool {
        let (t, need) = (self.win.turns as usize, self.cfg.min_turns);
        self.win.n > 0 && t >= need && (self.win.folds as usize >= self.cfg.folds || t >= (2 * need).max(1))
    }

    /// Close the window and decide (trigger: "turns" | "fold" | "state_card"). None without signal.
    pub fn decide(&mut self, trigger: &str) -> Option<Value> {
        let w = std::mem::take(&mut self.win);
        let p = w.mean()?;
        let c = ranked(&p);
        let ((d1, p1), (d2, p2)) = (c[0], c[1]);
        let margin = p1 - p2;
        let confident = margin >= self.cfg.margin;
        // no margin: hold the active domain if it is one of the top two, else the union (mixed)
        let cand = if confident { Some(d1) }
            else if self.domain == NONE || (self.domain != MIXED && self.domain != d1 && self.domain != d2) { Some(MIXED) } else { None };
        let before = self.domain;
        match cand {
            Some(x) if x == self.domain => self.pending = None,
            Some(x) => {
                let n = match self.pending { Some((d, n)) if d == x => n + 1, _ => 1 };
                // first decision is adopted at once (nothing to protect yet)
                if n >= self.cfg.hysteresis || self.domain == NONE { self.domain = x; self.since_turn = self.turn; self.pending = None; }
                else { self.pending = Some((x, n)); }
            }
            None => {} // sub-margin window: progress toward a pending switch is kept, not reset
        }
        self.last_win = Some(p);
        self.cands = c;
        let plan = self.plan(self.domain, 0.0).map(|p| p.json());
        let learned: Vec<String> = labels().filter(|&i| w.learned[i] > 0).map(|i| format!("{} {}", dname(i), ktok(w.learned[i]))).collect();
        let rec = json!({"ts": now_s(), "turn": self.turn, "trigger": trigger, "label": dname(if confident { d1 } else { MIXED }),
            "confidence": r3(p1 as f64), "margin": r3(margin as f64), "runner_up": dname(d2), "confident": confident,
            "domain": dname(self.domain), "switched": self.domain != before, "since_turn": self.since_turn,
            "pending": self.pending.map(|(d, n)| json!([dname(d), n])), "candidates": self.cands_json(4),
            "window": {"turns": w.turns, "folds": w.folds, "samples": w.n, "labels": Win::named(&w.labels), "learned": Win::named(&w.learned)},
            "tokens_window": w.learned.iter().sum::<u64>(), "sessions": w.keys,
            "learned": if learned.is_empty() { Value::Null } else { Value::from(learned.join(", ")) },
            "mode": self.cfg.mode, "plan": plan});
        append_compact(&self.dir.join("decisions.jsonl"), &rec.to_string(), DECISIONS_MAX_LINES, DECISIONS_KEEP);
        self.recent.push_back(rec.clone());
        while self.recent.len() > RECENT { self.recent.pop_front(); }
        Some(rec)
    }

    fn cands_json(&self, n: usize) -> Value {
        Value::Array(self.cands.iter().take(n).map(|&(d, s)| json!({"domain": dname(d), "score": r3(s as f64)})).collect())
    }

    /// The active domain's real standing: (its share, its lead over the best other (< 0 = behind), the top domain, the
    /// top's lead, from the open window (true) or the last closed one). None before any signal.
    pub fn standing(&self) -> Option<(f32, f32, usize, f32, bool)> {
        let (p, open) = match self.win.mean() { Some(p) => (p, true), None => (self.last_win?, false) };
        let c = ranked(&p);
        let (conf, lead) = match self.domain {
            d if d < MIXED => (p[d], p[d] - c.iter().find(|x| x.0 != d).map(|x| x.1).unwrap_or(0.0)),
            MIXED => (c[0].1 + c[1].1, 0.0),
            _ => (0.0, 0.0),
        };
        Some((conf, lead, c[0].0, c[0].1 - c[1].1, open))
    }

    /// Dry-run plan for label `d` (cached until its profile learns). `footprint_gb` (the process now) is used only
    /// when the engine gave no resident estimate.
    pub fn plan(&mut self, d: usize, footprint_gb: f64) -> Option<Plan> {
        let i = if d < MIXED || d == UNASSIGNED { d } else { MIXED };
        if self.plans[i].is_none() {
            let (dm, st) = (self.dims.as_ref()?, self.store.as_ref()?);
            let sel = if i == MIXED { Sel::Labels } else { Sel::Set(i) };
            st.prog.phase.store(3, Relaxed);
            let po = st.plan(sel, self.cfg.keep_frac, dm.gran);
            st.prog.phase.store(0, Relaxed);
            self.plans[i] = Some(Plan::from_out(dname(i), &po, dm, 0.0));
        }
        let mut p = self.plans[i]?;
        p.domain = dname(d);
        if self.dims.as_ref().is_some_and(|m| m.tmpl.resident_bytes == 0) { p.footprint_gb = (footprint_gb - p.gb_unload).max(0.0); }
        Some(p)
    }

    /// GET /v1/streamdense/watcher.
    pub fn status(&mut self, footprint_gb: f64) -> Value {
        let plan = self.plan(self.domain, footprint_gb).map(|p| p.json()).unwrap_or_else(|| json!({"domain": dname(self.domain),
            "tokens": 0, "ready": false, "cold_ffn_frac": 0.0, "cold_heads": 0, "gb_unload": 0.0, "ms_saved_per_pass": 0.0,
            "ms_saved_unload": 0.0, "footprint_gb": r3(footprint_gb), "fits_cap": true, "per_layer_cold": [], "per_layer_cold_heads": []}));
        let profiles: Vec<Value> = if self.profiles.is_empty() { vec![] } else {
            labels().chain((self.profiles[MIXED].tokens > 0).then_some(MIXED)).collect::<Vec<_>>().into_iter().map(|d| { let g = self.plan(d, footprint_gb).map(|p| p.gb_unload).unwrap_or(0.0);
                json!({"domain": dname(d), "tokens": self.profiles[d].tokens, "turns": self.profiles[d].windows, "gb_unload": r3(g),
                       "representative": self.profiles[d].tokens >= REPRESENTATIVE_TOKENS}) }).collect()
        };
        let activity = if self.act_jobs > 0 { json!({"source": "activity recorder", "jobs": self.act_jobs}) }
            else { json!({"source": "none yet (recorder off or no job yet): text features only", "jobs": 0}) };
        let st = self.standing();
        let wm = self.win.mean().map(|p| { let c = ranked(&p); json!({"top": dname(c[0].0), "margin": r3((c[0].1 - c[1].1) as f64)}) });
        json!({"domain": dname(self.domain), "confidence": r3(st.map_or(0.0, |s| s.0) as f64), "margin": r3(st.map_or(0.0, |s| s.1) as f64),
            "confidence_from": match st { Some((.., true)) => "open window", Some(_) => "last window", None => "no signal yet" },
            "since_turn": self.since_turn, "turn": self.turn,
            "pending": self.pending.map(|(d, n)| json!({"domain": dname(d), "wins": n})), "candidates": self.cands_json(ND),
            "window": {"turns": self.win.turns, "folds": self.win.folds, "samples": self.win.n, "need_turns": self.cfg.min_turns,
                       "need_folds": self.cfg.folds, "labels": Win::named(&self.win.labels), "learned": Win::named(&self.win.learned),
                       "scores": wm, "last_turn": {"label": dname(self.last_turn.0), "margin": r3(self.last_turn.1 as f64)}},
            "decisions_recent": self.recent.iter().rev().take(10).cloned().collect::<Vec<_>>(), "tokens_observed": self.tokens_observed,
            "unload_plan": plan, "profiles": profiles, "mode": self.cfg.mode, "activity": activity,
            "held": self.held_json(), "held_slot": self.store.as_ref().and_then(|st| st.held_owner()).map(dname), "slots": self.slots_json(), "batch": self.batch_json(),
            "config": {"decide_window_folds": self.cfg.folds, "decide_window_min_turns": self.cfg.min_turns, "opp_similarity": self.cfg.opp, "fold_every": self.cfg.fold_every, "margin": self.cfg.margin,
                "hysteresis": self.cfg.hysteresis, "go_live_points": self.cfg.go_live,
                "keep_frac": self.cfg.keep_frac, "granularity": self.cfg.gran, "keep_languages": self.cfg.keep_languages, "cut_languages": self.cfg.cut_languages,
                "sample_ms": self.cfg.sample_ms, "mask": self.cfg.mask, "profile_window": self.cfg.profile_window, "explore_frac": self.cfg.explore_frac, "dir": self.dir.display().to_string()},
            "scorecard": self.scorecard(), "audits_recent": self.audits.iter().rev().take(AUDITS_SENT).rev().cloned().collect::<Vec<_>>(), "audit_queue": self.audit_queue_json(),
            "mask": self.mk.as_ref().map_or_else(|| json!({"mode": "off", "applied": false}), |m| m.json()),
            "note": "dry run: nothing unloaded yet"})
    }

    // ---- read mask, prompt predictor, exploration (predict.rs) ------------------------------------------------------

    /// Score + learn the CPU prompt predictor on one finished job (the recorder in place: activity + new tokens). Mask
    /// off: nothing. An aggregate goes to notes.jsonl every `predict::NOTE_EVERY` requests.
    pub fn score_job(&mut self, src: &dyn JobSrc, toks: &dyn predict::Tokens, prompt: Option<&streamdense_model::readmask::ReadMask>) -> Option<predict::Score> {
        let mk = self.mk.as_mut()?;
        let (s, note) = mk.score_job(src, toks, prompt);
        if let Some((a, n)) = note {
            let mut v = a.json(n);
            v["ts"] = json!(now_s());
            v["kind"] = json!("predict");
            v["turn"] = json!(self.turn);
            v["text"] = json!(format!("predictor (stage A, scored only): {} requests: precision {} recall {}, masked precision {} recall {}, {} B predicted / {} B missed per request",
                n, v["precision"], v["recall"], v["masked_precision"], v["masked_recall"], v["bytes_predicted_per_request"], v["bytes_missed_per_request"]));
            append_compact(&self.dir.join("notes.jsonl"), &v.to_string(), 2 * NOTES_KEEP, NOTES_KEEP);
        }
        s
    }
    /// Rebuild the profile mask from the watcher's own store (tests / no sd-heat); the gateway builds on sd-heat's
    /// read-only store (`Shared::rebuild_mask`).
    #[cfg(test)]
    pub fn rebuild_mask(&mut self, t: f64) -> bool {
        let (Some(mk), Some(st)) = (self.mk.as_mut(), self.store.as_mut()) else { return false };
        mk.rebuild(st, t);
        true
    }
    /// An exploration round when one is due (idle; mask dry / on). A round that found active units is noted.
    pub fn explore_idle(&mut self, t: f64) -> bool {
        let Some(mk) = self.mk.as_mut().filter(|m| m.explore_due(t)) else { return false };
        let Some(x) = mk.explore(t) else { return false };
        if x.found > 0 {
            let v = json!({"ts": t, "kind": "explore", "round": x.round, "found_active": x.found, "joined": x.joined, "scheduled": x.n,
                "text": format!("exploration round {}: {} of {} re-read masked units were active; they rejoin the profile", x.round, x.found, x.n)});
            append_compact(&self.dir.join("notes.jsonl"), &v.to_string(), 2 * NOTES_KEEP, NOTES_KEEP);
        }
        true
    }

    // ---- audits (every reply) ----------------------------------------------------------------------------------------

    /// Queue reply `msg_id` for its audit (called by the job hook after `observe`): its turn's own label and the domain
    /// `active` when it was generated (whose mask the audit checks). Bounded: a full queue drops its oldest reply.
    pub fn remember_job(&mut self, msg_id: &str, active: usize) {
        if msg_id.is_empty() { return; }
        self.audit_q.push_back((trunc(msg_id, ID_MAX).to_string(), self.last_turn.0, active, self.turn));
        while self.audit_q.len() > AUDIT_QUEUE_MAX { self.audit_q.pop_front(); self.audit_dropped += 1; }
    }

    /// The recent audits from audits.jsonl (a votes.jsonl of the vote era is left on disk, unread).
    fn load_audits(&mut self) {
        if let Ok(t) = std::fs::read_to_string(self.dir.join("audits.jsonl")) {
            let ls: Vec<&str> = t.lines().collect();
            for l in &ls[ls.len().saturating_sub(AUDITS_KEEP)..] { if let Ok(v) = serde_json::from_str::<Value>(l) { self.audits.push_back(v); } }
        }
    }

    /// A stage-A read mask is active for domain `d`. No masked kernels exist yet: always false, so every reply
    /// so far came from the full model and there is nothing to compare.
    fn mask_active(&self, _d: usize) -> bool { self.mk.as_ref().is_some_and(|m| m.applied()) }

    /// Keep an audit (file + recent list) and score it into its label's go-live state.
    fn record_audit(&mut self, d: usize, a: Value) {
        append_compact(&self.dir.join("audits.jsonl"), &a.to_string(), AUDITS_MAX_LINES, AUDITS_KEEP);
        if d < NP { self.golive[d].apply(a["result"].as_str().unwrap_or("")); }
        self.audits.push_back(a);
        while self.audits.len() > AUDITS_KEEP { self.audits.pop_front(); }
        self.audit_done += 1;
        self.save_state();
    }

    /// Audit the newest queued reply (caller: the low-level watcher thread, only while the engine is idle). False when
    /// the queue is empty.
    pub fn audit_next(&mut self) -> bool {
        let Some((id, d, active, turn)) = self.audit_q.pop_back() else { return false };
        if self.mask_active(active) {
            // TODO(once read masks exist): the real audit. Re-check the reply with the FULL model in idle GPU time:
            // teacher-forced, exact K = 8 passes over the reply's tokens (same kernel family as decode, so a match is
            // bit-exact), pausing whenever a new request arrives (push the reply back). Result: match | mismatch
            // (+ diverge_at, the full model's continuation as full_answer, cause "mask"); then `record_audit`.
            unreachable!("no read masks yet");
        }
        // honest dry run: the reply came from the full model, so there is nothing to compare against
        let a = json!({"ts": now_s(), "message_id": id, "domain": dname(d), "active": dname(active), "turn": turn,
            "result": "skipped", "reason": AUDIT_SKIP_REASON, "cause": null, "tokens": 0, "diverge_at": null, "full_answer": null});
        self.record_audit(d, a);
        true
    }

    fn audit_queue_json(&self) -> Value {
        json!({"pending": self.audit_q.len(), "max": AUDIT_QUEUE_MAX, "dropped": self.audit_dropped, "done": self.audit_done,
               "order": "newest first, engine idle only"})
    }

    /// Per-label scorecard (labels with audits, plus the current domain) + the go-live points. agreement is null until a
    /// real audit compared tokens.
    pub fn scorecard(&self) -> Vec<Value> {
        let mut ds: Vec<usize> = self.audits.iter().map(|a| dindex(a["domain"].as_str().unwrap_or(""))).collect();
        if self.domain < ND { ds.push(self.domain); }
        ds.retain(|&d| d < NP);
        ds.sort_unstable();
        ds.dedup();
        ds.iter().map(|&d| {
            let au: Vec<&Value> = self.audits.iter().filter(|a| dindex(a["domain"].as_str().unwrap_or("")) == d).collect();
            let res = |r: &str| au.iter().filter(|a| a["result"] == r).count();
            let audited: u64 = au.iter().filter(|a| a["result"] == "match" || a["result"] == "mismatch").map(|a| a["tokens"].as_u64().unwrap_or(0)).sum();
            let agreed: u64 = au.iter().map(|a| match a["result"].as_str() {
                Some("match") => a["tokens"].as_u64().unwrap_or(0),
                Some("mismatch") => a["diverge_at"].as_u64().unwrap_or(0),
                _ => 0,
            }).sum();
            let g = &self.golive[d];
            json!({"domain": dname(d), "audited_tokens": audited, "agreement": if audited > 0 { json!(r3(agreed as f64 / audited as f64)) } else { Value::Null },
                "mismatches": res("mismatch"), "skipped": res("skipped"), "clean_audits": res("match"), "go_live_at": self.cfg.go_live.round() as u64,
                "points": r3(g.points), "required": r3(g.required(&self.cfg)), "resets": g.resets, "tightened": g.tightened,
                "live": g.reached(&self.cfg) && self.mask_active(d)})
        }).collect()
    }

    /// One line for the request log / health (cached plan only: no plan work per request), with the real standing:
    /// `chat-general 0.12 (-0.31 vs code-rust; window 23/50 turns 1/3 folds) since t51 (t90); would unload .. [dry run]`.
    pub fn summary(&self) -> String {
        let plan = self.plans[if self.domain < MIXED || self.domain == UNASSIGNED { self.domain } else { MIXED }].as_ref().filter(|p| p.tokens > 0)
            .map(|p| format!("; would unload {:.2} GB{}", p.gb_unload, if p.ready { "" } else { " (few tokens)" })).unwrap_or_default();
        let st = match self.standing() {
            Some((conf, lead, top, tm, _)) => {
                let vs = if top == self.domain || self.domain >= MIXED { format!("top {} +{:.2}", dname(top), tm) } else { format!("{:+.2} vs {}", lead, dname(top)) };
                format!("{:.2} ({}; window {}/{} turns {}/{} folds)", conf, vs, self.win.turns, self.cfg.min_turns, self.win.folds, self.cfg.folds)
            }
            None => format!("- (window {}/{} turns {}/{} folds)", self.win.turns, self.cfg.min_turns, self.win.folds, self.cfg.folds),
        };
        format!("{} {} since t{} (t{}){} [dry run]", self.domain_name(), st, self.since_turn, self.turn, plan)
    }
}

/// Features of one /v1/messages job: the newest user text, the tool results after it (<= 3; markers double, prose
/// neutral), the generated reply.
pub fn job_feat(msgs: &[streamdense_gateway::chatmap::ChatMsg], n_tools: usize, reply: &str) -> Feat {
    let mut f = Feat::of(reply);
    let mut tool = 0;
    for m in msgs.iter().rev() {
        match m.role.as_str() {
            "tool" => { tool += 1; if tool <= 3 { f.add(&Feat::tool_result(&m.content)); } }
            "user" => { f.add(&Feat::of(&m.content)); break; }
            "system" => break,
            _ => {}
        }
    }
    f.tools = n_tools as u32;
    f.tool_msgs = tool;
    f
}

// ---- gateway glue ---------------------------------------------------------------------------------------------------

/// The watcher as a gateway `Extension` (routes + log field); the job hook goes through `on_job`. `heat_json` = the
/// summary file the low-level watcher thread writes (heat.rs; the GET view reads it), `prog` = the store's live
/// progress, `jobs` / `summarized` = jobs seen / jobs covered by the last summary (lag), `last_job_ms` = end of the last
/// job (idle detection without a recorder).
/// One finished job handed from the job hook to the sd-heat thread (the recorder holds exactly this job until then).
#[derive(Clone, Debug)]
pub struct JobMsg { pub f: Feat, pub tokens: u64, pub key: String, pub msg_id: String, pub t_end: f64,
    /// Mask dry / on: the engine's prompt-active set of this sequence (`QwenEngine::prompt_active`), the predictor's truth
    pub prompt: Option<streamdense_model::readmask::ReadMask>,
    /// answer library (`library = true`): the turn's token counts (+ text only with `library_text`)
    pub lib: Option<crate::library::LibJob> }

pub struct Shared {
    /// the job hook's hand-off (one job at a time: the recorder gate holds the next job back) and its wake-up
    pub queue: Mutex<Option<JobMsg>>,
    pub qcv: Condvar,
    /// the recorder as a job source (set by the gateway when the recorder is on)
    pub src: OnceLock<crate::store::TapSrc>,
    pub submitted: AtomicU64,
    pub w: Mutex<Watcher>,
    pub footprint: fn() -> u64,
    pub heat_json: OnceLock<PathBuf>,
    pub prog: OnceLock<Arc<Progress>>,
    pub jobs: AtomicU64,
    pub summarized: AtomicU64,
    pub last_job_ms: AtomicU64,
    /// answer library collector (`[watcher] library = true`; set by the gateway)
    pub lib: OnceLock<Mutex<crate::library::Collector>>,
}

impl Shared {
    pub fn new(w: Watcher, footprint: fn() -> u64) -> Shared {
        let prog = OnceLock::new();
        if let Some(st) = &w.store { let _ = prog.set(st.prog.clone()); }
        Shared { queue: Mutex::new(None), qcv: Condvar::new(), src: OnceLock::new(), submitted: AtomicU64::new(0), w: Mutex::new(w), footprint, heat_json: OnceLock::new(), prog, jobs: AtomicU64::new(0), summarized: AtomicU64::new(0), last_job_ms: AtomicU64::new(0), lib: OnceLock::new() }
    }
    /// `src`: the job's activity (the GPU recorder in place; the caller resets it after); `msg_id`: the reply's message
    /// id (queued for its audit, attributed to this turn's own label).
    pub fn on_job(&self, f: &Feat, tokens: u64, src: Option<&dyn JobSrc>, key: &str, msg_id: &str) {
        if let Ok(mut w) = self.w.lock() { let active = w.domain; w.observe_src(f, tokens, src, key); w.remember_job(msg_id, active); }
        self.jobs.fetch_add(1, Relaxed);
        self.last_job_ms.store((now_s() * 1e3) as u64, Relaxed);
    }
    /// The job hook (worker thread, recorder on): close the recorder gate and hand the job to sd-heat. Returns at once
    /// (no score, no fold); the next job's first recorder write waits only if sd-heat has not released it yet.
    pub fn submit(&self, m: JobMsg) {
        streamdense_model::activity::GATE.close();
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        if q.replace(m).is_some() { eprintln!("watcher: a job was handed over before the previous one was taken (dropped)"); }
        self.submitted.fetch_add(1, Relaxed);
        self.qcv.notify_one();
    }
    /// sd-heat: the next handed-over job, waiting up to `wait`.
    pub fn next_job(&self, wait: std::time::Duration) -> Option<JobMsg> {
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        if q.is_none() { q = self.qcv.wait_timeout(q, wait).map(|r| r.0).unwrap_or_else(|e| e.into_inner().0); }
        q.take()
    }
    /// sd-heat: score + label + gate + batch one handed-over job from the recorder in place, then zero the recorder and
    /// open the gate (also when anything above fails).
    pub fn process(&self, m: JobMsg) {
        struct Open<'a>(Option<&'a crate::store::TapSrc>);
        impl Drop for Open<'_> { fn drop(&mut self) { if let Some(s) = self.0 { s.tap.reset(); } streamdense_model::activity::GATE.open(); } }
        let src = self.src.get();
        let _open = Open(src);
        let mut names = ("unknown", "unknown");
        if let Ok(mut w) = self.w.lock() {
            let active = w.domain;
            w.observe_src(&m.f, m.tokens, src.map(|s| s as &dyn JobSrc), &m.key);
            w.remember_job(&m.msg_id, active);
            // Stage A: predictor scored on the job's own activity + new tokens (before the recorder is zeroed)
            if let Some(s) = src { w.score_job(s, s, m.prompt.as_ref()); }
            names = (dname(w.last_turn.0), dname(active));
        }
        // answer library: this job's prompt mark + answer sums, still under the closed gate (encode + write: `library_flush`)
        if let (Some(j), Some(l)) = (m.lib.as_ref(), self.lib.get()) { l.lock().unwrap_or_else(|e| e.into_inner()).capture(j, names.0, names.1); }
        self.jobs.fetch_add(1, Relaxed);
        self.last_job_ms.store((m.t_end * 1e3) as u64, Relaxed);
    }
    /// sd-heat, after `process` (gate open): encode + write the pending library record (no-op when off / none).
    pub fn library_flush(&self) -> bool { self.lib.get().is_some_and(|l| l.lock().unwrap_or_else(|e| e.into_inner()).flush()) }
    /// The library status (GET view) and its one-line form (health / request log); off = {"on": false}.
    pub fn library_json(&self) -> Value { self.lib.get().map_or_else(|| json!({"on": false}), |l| l.lock().unwrap_or_else(|e| e.into_inner()).json()) }
    /// sd-heat: rebuild the profile mask on its read-only store into `scratch` (outside the watcher lock), then install
    /// it under the lock (mask = profile | joined). False with mask = off. No heap.
    pub fn rebuild_mask(&self, st: &mut Store, scratch: &mut predict::ReadMask, t: f64) -> bool {
        let Some((win, keep)) = self.w.lock().ok().and_then(|w| w.mk.as_ref().map(|m| (m.win, m.keep))) else { return false };
        let t0 = std::time::Instant::now();
        let tok = predict::build_profile(st, win, keep, t, scratch);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let Ok(mut w) = self.w.lock() else { return false };
        let Some(mk) = w.mk.as_mut() else { return false };
        mk.install(scratch, tok, ms, t);
        true
    }
    /// sd-heat, idle: one exploration round when due.
    pub fn explore_idle(&self, t: f64) -> bool { self.w.lock().map(|mut w| w.explore_idle(t)).unwrap_or(false) }
    /// mask != off (sd-heat builds the profile then)
    pub fn mask_enabled(&self) -> bool { self.w.lock().map(|w| w.mk.is_some()).unwrap_or(false) }
    /// sd-heat: fold a batch idle for >= `idle_s`.
    pub fn flush_idle(&self, idle_s: f64) -> bool { self.w.lock().map(|mut w| w.flush_if_idle(now_s(), idle_s)).unwrap_or(false) }
    /// Shutdown: fold the open batch.
    pub fn flush(&self) { if let Ok(mut w) = self.w.lock() { w.flush_batch(); } }
    /// One queued audit (newest first); false when none is waiting.
    pub fn audit_idle(&self) -> bool { self.w.lock().map(|mut w| w.audit_next()).unwrap_or(false) }
    /// The heat summary file + the live progress / lag.
    pub fn heat_view(&self) -> Value {
        let mut v = self.heat_json.get().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .unwrap_or_else(|| json!({"state": "starting"}));
        v["live"] = crate::heat::live_json(self);
        v
    }
}

impl streamdense_gateway::Extension for Shared {
    fn route(&self, method: &str, path: &str, body: &[u8]) -> Option<(u16, Value)> {
        let err = |code: u16, m: &str| Some((code, json!({"type": "error", "error": {"type": "invalid_request_error", "message": m}})));
        match (method, path.trim_end_matches('/')) {
            ("GET" | "HEAD", "/v1/streamdense/watcher") => {
                let fp = (self.footprint)() as f64 / 1e9;
                let mut st = self.w.lock().ok()?.status(fp);
                st["heat"] = self.heat_view();
                st["library"] = self.library_json();
                Some((200, st))
            }
            ("POST", "/v1/streamdense/watcher/note") => {
                let v: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return err(400, &format!("body is not JSON: {}", e)) };
                let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
                let mut w = self.w.lock().ok()?;
                match w.note(&s("kind"), &s("text"), &s("project"), &s("mode")) {
                    Ok(d) => Some((200, json!({"ok": true, "decision": d, "domain": w.domain_name(), "turn": w.turn}))),
                    Err(e) => err(400, &e),
                }
            }
            // no votes: every reply is audited automatically
            (_, "/v1/streamdense/watcher/vote") => Some((410, json!({"type": "error", "error": {"type": "gone",
                "message": "votes were removed: every reply is audited automatically in idle time (see GET /v1/streamdense/watcher)"}}))),
            (_, "/v1/streamdense/watcher" | "/v1/streamdense/watcher/note") => err(405, &format!("{} not allowed on {}", method, path)),
            _ => None,
        }
    }
    fn log_field(&self) -> Option<(&'static str, Value)> {
        let s = self.w.lock().ok()?.summary();
        let lib = self.lib.get().map(|l| format!(" | {}", l.lock().unwrap_or_else(|e| e.into_inner()).line())).unwrap_or_default();
        Some(("watcher", Value::from(format!("{}{}", s, lib))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sd_watcher_{}_{}", n, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }
    /// every turn / note closes the window; switch after 2 windows
    fn cfg() -> WCfg {
        WCfg { mode: "observe".into(), folds: 0, min_turns: 0, fold_every: 8, opp: 0.25, margin: 0.15, hysteresis: 2, go_live: 50.0, sample_ms: 100,
               keep_frac: 0.995, gran: 64, keep_languages: "all-except-cut".into(), cut_languages: vec!["zh".into()],
               mask: "off".into(), profile_window: "session+all".into(), explore_frac: 0.02 }
    }
    fn top(t: &str) -> &'static str { let p = Feat::of(t).probs().unwrap(); dname(label_of(&p).0) }
    fn margin(f: &Feat) -> (usize, f32) { let c = ranked(&f.probs().unwrap()); (c[0].0, c[0].1 - c[1].1) }
    /// Recorder-shaped template: layer 3 of every 4 is attention (3 q heads), the rest DeltaNet (2 value heads).
    fn tmpl(layers: usize, inter: usize) -> ActivityStats {
        let kinds = (0..layers).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        ActivityStats { enabled: true, layers, hidden: 128, inter, kinds, dn_kh: 1, dn_dk: 128, dn_vh: 2, dn_dv: 128, heads: 3, kv_heads: 1,
            head_dim: 128, conv_k: 4, group: 64, bits: 4, weight_bytes: 9_000_000_000, resident_bytes: 10_000_000_000, ..Default::default() }
    }
    /// 4 layers x 256 neurons (4 blocks of 64); hmax 3.
    fn dims() -> Dims { Dims::from_stats(&tmpl(4, 256), 64) }
    /// scores: (domain, share) pairs, rest 0
    fn sc(v: &[(usize, f32)]) -> Option<[f32; ND]> { let mut p = [0f32; ND]; for &(d, x) in v { p[d] = x; } Some(p) }

    const RUST_T: &str = "fn main() {\n    let mut v: Vec<u32> = Vec::new();\n    v.push(1);\n    println!(\"{:?}\", v);\n}\nimpl Foo { pub fn bar(&mut self) -> Result<(), String> { Ok(()) } }\n";
    const SWIFT_T: &str = "import SwiftUI\nstruct ContentView: View {\n    @State private var n = 0\n    var body: some View {\n        Text(\"hi\")\n    }\n}\nfunc go() { guard let x = y else { return } }\n";
    const EN_T: &str = "Could you tell me what a good weekend trip near the coast would be? I like quiet towns and long walks by the sea.";
    const VI_T: &str = "Bạn có thể giới thiệu cho tôi một chuyến đi cuối tuần gần biển không? Tôi thích những thị trấn yên tĩnh và đi bộ dọc bờ biển.";
    const PROSE: &str = "The test failed because the parser kept the old token after the error. I changed the loop so it resets the \
        position before it retries, and now the whole suite passes. The remaining warning is about an unused import, which I left alone.";
    const CARGO: &str = "   Compiling stockroom v0.1.0 (/x/stockroom)\nerror[E0502]: cannot borrow `v` as mutable\n --> src/main.rs:4:5\n  |\n4 |     v.push(1);\n  |     ^^^^^^^^^\n\
        For more information about this error, try `rustc --explain E0502`.\n";

    #[test]
    fn features() {
        assert_eq!(top(RUST_T), "code-rust");
        assert_eq!(top(SWIFT_T), "code-swift");
        assert_eq!(top("def f(x):\n    return x + 1\nimport numpy as np\nprint(np.zeros(3))\n"), "code-other");
        // language is not a domain: the same topic in English, Vietnamese or Russian is chat-general
        assert_eq!(top(EN_T), "chat-general");
        assert_eq!(top(VI_T), "chat-general");
        assert_eq!(top("Расскажите, пожалуйста, о хорошем маршруте на выходные у моря."), "chat-general");
        assert_eq!(Feat::of(EN_T).probs(), Feat::of(VI_T).probs());
        let mut f = Feat::of("ok, listing the files now");
        f.tools = 12; f.tool_msgs = 4;
        assert_eq!(dname(label_of(&f.probs().unwrap()).0), "agent-tools");
        assert!(Feat::of("").probs().is_none());
        // mode hint pushes a neutral text toward code
        let mut g = Feat::of("please look at the parser and fix the bug"); g.mode = 1;
        assert!(dname(label_of(&g.probs().unwrap()).0).starts_with("code-"));
        // the job features: newest user text + trailing tool results + reply; older turns ignored
        use streamdense_gateway::chatmap::ChatMsg;
        let msgs = vec![ChatMsg::new("user", VI_T), ChatMsg::new("assistant", "ok"), ChatMsg::new("user", RUST_T), ChatMsg::tool("src/main.rs")];
        let jf = job_feat(&msgs, 3, "Done.");
        assert_eq!((jf.tools, jf.tool_msgs), (3, 1));
        // regression: a prose reply after tool use on Rust is still clearly code-rust (was ~0.46 vs chat)
        let msgs = vec![ChatMsg::new("user", "please fix the failing test in the parser"), ChatMsg::tool(CARGO), ChatMsg::tool(RUST_T)];
        let (d, m) = margin(&job_feat(&msgs, 2, PROSE));
        assert_eq!(dname(d), "code-rust");
        assert!(m >= 0.3, "rust margin {}", m);
        // tool results: prose is neutral (cargo output does not pull toward chat), markers count double
        assert_eq!(Feat::tool_result(CARGO).letters, 0);
        assert_eq!(Feat::tool_result(RUST_T).rust, 2.0 * Feat::of(RUST_T).rust);
        // ambiguous markers are gone: a Rust match arm is not code-other
        assert_eq!(top("match x {\n    Some(v) => { println!(\"{}\", v) }\n    None => {}\n}\n"), "code-rust");
        // state cards / folds: their prose is neutral; code in them still counts
        let mut card = Feat::of(EN_T); card.neutral = true;
        assert!(card.probs().is_none());
        let mut card = Feat::of(&format!("{}\n{}", EN_T, RUST_T)); card.neutral = true;
        assert_eq!(dname(label_of(&card.probs().unwrap()).0), "code-rust");
        assert_eq!(trunc("aé", 2), "a");
        assert_eq!(short_hash("x").len(), 8);
    }

    #[test]
    fn hysteresis_no_flip_flop() {
        let d = tmpdir("hyst");
        let mut w = Watcher::open(d.clone(), cfg(), None).unwrap();
        let rust = Feat::of(RUST_T);
        let chat = Feat::of(EN_T);
        for _ in 0..4 { w.observe(&rust, 100, None, "k"); }
        assert_eq!(w.domain_name(), "code-rust");
        let s0 = w.since_turn;
        // single off-topic windows between rust windows: no switch
        w.observe(&chat, 10, None, "k");
        assert_eq!(w.domain_name(), "code-rust");
        w.observe(&rust, 10, None, "k");
        w.observe(&chat, 10, None, "k");
        w.observe(&rust, 10, None, "k");
        assert_eq!((w.domain_name(), w.since_turn), ("code-rust", s0));
        // two in a row: switch
        w.observe(&chat, 10, None, "k");
        w.observe(&chat, 10, None, "k");
        assert_eq!(w.domain_name(), "chat-general");
        assert_eq!(w.since_turn, w.turn);
        let dec = std::fs::read_to_string(d.join("decisions.jsonl")).unwrap();
        assert_eq!(dec.lines().count(), 10);
        assert_eq!(dec.lines().filter(|l| l.contains("\"switched\":true")).count(), 2); // first adoption + the real switch
        // window of 6 turns + 1 fold: nothing before both; a client without folds closes at 12 turns
        let mut c = cfg(); c.min_turns = 6; c.folds = 1;
        let d2 = tmpdir("every");
        let mut w = Watcher::open(d2.clone(), c, None).unwrap();
        for i in 0..5 { assert!(w.observe(&rust, 1, None, "k").is_none(), "turn {}", i); }
        assert!(w.note("fold", RUST_T, "/x/proj", "code").unwrap().is_none()); // 5 < 6 turns
        assert!(w.note("bogus", "", "", "").is_err());
        let dec = w.observe(&rust, 1, None, "k").unwrap(); // 6 turns + 1 fold
        assert_eq!((dec["window"]["turns"].as_u64(), dec["window"]["folds"].as_u64(), dec["trigger"].as_str()), (Some(6), Some(1), Some("turns")));
        for i in 0..11 { assert!(w.observe(&rust, 1, None, "k").is_none(), "turn {}", i); }
        assert!(w.observe(&rust, 1, None, "k").is_some()); // 12 turns, no fold
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }

    /// Regression: the winner is consistently ahead but below 0.5 -> it must still switch (margin rule), and a
    /// sub-margin window must not reset progress toward it.
    #[test]
    fn margin_windows_no_stuck() {
        let d = tmpdir("margin");
        let mut c = cfg(); c.min_turns = 10; c.folds = 1; c.hysteresis = 1;
        let mut w = Watcher::open(d.clone(), c.clone(), None).unwrap();
        let chat = sc(&[(CHAT, 0.9), (3, 0.1)]);
        let rust = sc(&[(0, 0.46), (CHAT, 0.29), (2, 0.15), (3, 0.10)]); // observed range 0.415-0.493
        for _ in 0..10 { w.observe_scores(chat, 100, None, "k"); }
        w.note("fold", "", "", "").unwrap(); // empty note: no signal, but the fold counts
        assert_eq!(w.domain_name(), "chat-general");
        for i in 0..9 { assert!(w.observe_scores(rust, 100, None, "k").is_none(), "{}", i); }
        // the open window already shows the real standing: chat-general well behind code-rust
        let (conf, lead, top, _, open) = w.standing().unwrap();
        assert!(open && (conf - 0.29).abs() < 1e-4 && lead < -0.15 && top == 0, "{} {} {}", conf, lead, top);
        assert!(w.summary().starts_with("chat-general 0.29 (-0.17 vs code-rust; window 9/10 turns 0/1 folds)"), "{}", w.summary());
        w.observe_scores(rust, 100, None, "k");
        let r = w.note("fold", "", "", "").unwrap().unwrap();
        assert_eq!((r["label"].as_str(), r["switched"].as_bool(), r["confident"].as_bool()), (Some("code-rust"), Some(true), Some(true)), "{}", r);
        assert_eq!(w.domain_name(), "code-rust");
        assert!((r["confidence"].as_f64().unwrap() - 0.46).abs() < 1e-3 && (r["margin"].as_f64().unwrap() - 0.17).abs() < 1e-3);
        assert_eq!(w.status(0.0)["confidence"], json!(0.46));

        // hysteresis 2: confident win -> pending 1; a sub-margin window (0.40 vs 0.33) keeps it; next confident win switches
        let mut c2 = c.clone(); c2.hysteresis = 2;
        let d2 = tmpdir("margin2");
        let mut w = Watcher::open(d2.clone(), c2, None).unwrap();
        let win = |w: &mut Watcher, p: Option<[f32; ND]>| { for _ in 0..10 { w.observe_scores(p, 1, None, ""); } w.note("fold", "", "", "").unwrap().unwrap() };
        win(&mut w, chat);
        assert_eq!(w.domain_name(), "chat-general");
        assert_eq!(win(&mut w, rust)["pending"], json!(["code-rust", 1]));
        let r = win(&mut w, sc(&[(0, 0.40), (CHAT, 0.33), (2, 0.27)]));
        assert_eq!((r["label"].as_str(), r["confident"].as_bool(), r["domain"].as_str()), (Some("mixed"), Some(false), Some("chat-general")));
        assert_eq!(r["pending"], json!(["code-rust", 1]), "sub-margin window must not reset: {}", r);
        let r = win(&mut w, rust);
        assert_eq!((r["domain"].as_str(), r["switched"].as_bool()), (Some("code-rust"), Some(true)));
        // no margin and the active domain not in the top two -> mixed (union)
        let r = win(&mut w, sc(&[(1, 0.36), (2, 0.34), (0, 0.30)]));
        assert_eq!(r["pending"], json!(["mixed", 1]));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }

    /// Values [off, off + len) of store set `s` (all-time).
    fn rd(w: &Watcher, s: usize, off: usize, len: usize) -> Vec<f32> {
        let st = w.store.as_ref().unwrap();
        let mut v = vec![0f32; len];
        let mut rec = [0u8; crate::store::CH * 4 + 16];
        st.read_sel(Sel::Set(s), off, &mut v, &mut rec);
        v
    }
    /// Synthetic job activity: `hot` blocks of every layer get the mass (and per neuron 100 / 64); heads: head 0 only.
    fn act(d: &Dims, hot: &[usize], tokens: u64) -> Act {
        let mut a = Act::zero(d);
        a.tokens = tokens;
        a.neuron = vec![0.0; d.layers * d.inter];
        for l in 0..d.layers {
            for &b in hot { a.ffn[l * d.blocks + b] = 100.0; for j in 0..64 { a.neuron[l * d.inter + b * 64 + j] = 100.0 / 64.0; } }
            a.heads[l * d.hmax] = 50.0;
        }
        a
    }

    /// Each turn is learned into its own label (never the active domain); small margins go to unassigned; audits
    /// follow the turn's label.
    #[test]
    fn learn_into_own_label_and_audits() {
        let dm = dims();
        let d = tmpdir("ownlabel");
        let mut c = cfg(); c.min_turns = 10; c.folds = 1; c.hysteresis = 1;
        let mut w = Watcher::open(d.clone(), c, Some(dm.clone())).unwrap();
        for _ in 0..10 { w.observe_scores(sc(&[(CHAT, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[3], 100)), "k"); }
        w.note("fold", "", "", "").unwrap();
        assert_eq!((w.domain_name(), w.profiles[CHAT].tokens), ("chat-general", 1000));
        // rust turns while chat-general is active: learned into code-rust, chat-general unchanged
        let active = w.domain;
        w.observe_scores(sc(&[(0, 0.46), (CHAT, 0.29), (2, 0.25)]), 1, Some(act(&dm, &[0], 200)), "k");
        w.remember_job("msg_rust", active);
        assert_eq!((w.profiles[0].tokens, w.profiles[CHAT].tokens, w.domain_name()), (200, 1000, "chat-general"));
        // ambiguous turn (0.40 vs 0.33): unassigned pool
        w.observe_scores(sc(&[(0, 0.40), (CHAT, 0.33), (2, 0.27)]), 1, Some(act(&dm, &[1], 50)), "k");
        w.remember_job("msg_amb", active);
        assert_eq!((w.profiles[UNASSIGNED].tokens, w.profiles[0].tokens), (50, 200));
        let st = w.store.as_ref().unwrap();
        assert!(st.has(UNASSIGNED) && st.has(0) && !st.has(1));
        // per-neuron sums on disk (the label's store set)
        let h = st.hdr(0);
        assert_eq!((h.tokens, h.turns), (200, 1));
        let n = rd(&w, 0, 0, 65);
        assert!((n[5] - 100.0 / 64.0).abs() < 1e-6 && n[64] == 0.0);
        // heads: head 0 of every layer, in the recorder head order
        let st = w.store.as_ref().unwrap();
        assert_eq!(rd(&w, 0, st.heads_of(3).start, 3), vec![50.0, 0.0, 0.0]);
        w.observe_scores(sc(&[(0, 0.9), (2, 0.1)]), 1, Some(act(&dm, &[0], 300)), "k");
        let h = w.store.as_ref().unwrap().hdr(0);
        assert_eq!((h.tokens, h.turns), (500, 2));
        assert!((rd(&w, 0, 5, 1)[0] - 200.0 / 64.0).abs() < 1e-5);
        // the pooled now / session sets got every job (chat 10 x, rust 2 x, unassigned 1 x)
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.hdr(crate::store::NOW).turns, st.hdr(crate::store::SESS).tokens), (13, 1550));
        let st = w.status(0.0);
        assert_eq!(st["window"]["labels"], json!({"code-rust": 2, "unassigned": 1}));
        assert_eq!(st["window"]["learned"], json!({"code-rust": 500, "unassigned": 50}));
        assert!(st["profiles"].as_array().unwrap().iter().any(|p| p["domain"] == "unassigned" && p["tokens"] == 50));
        // audits: every reply is queued; drained newest first, attributed to the turn's own label (active beside it)
        assert_eq!(w.status(0.0)["audit_queue"]["pending"], json!(2));
        assert!(w.audit_next());
        let a = w.audits.back().unwrap().clone();
        assert_eq!((a["message_id"].as_str(), a["domain"].as_str(), a["active"].as_str(), a["result"].as_str()),
                   (Some("msg_amb"), Some("unassigned"), Some("chat-general"), Some("skipped")));
        assert!(a.get("vote").is_none());
        assert!(w.audit_next());
        assert_eq!(w.audits.back().unwrap()["domain"], "code-rust");
        assert!(!w.audit_next());
        let sc = w.scorecard();
        assert!(sc.iter().any(|r| r["domain"] == "unassigned" && r["skipped"] == 1));
        assert!(sc.iter().any(|r| r["domain"] == "code-rust" && r["skipped"] == 1 && r["points"] == json!(0.0) && r["required"] == json!(50.0)
            && r.get("likes").is_none() && r["agreement"].is_null()));
        // bounded queue: the oldest replies are dropped
        for i in 0..AUDIT_QUEUE_MAX + 5 { w.remember_job(&format!("m{}", i), 0); }
        let q = &w.status(0.0)["audit_queue"];
        assert_eq!((q["pending"].as_u64(), q["dropped"].as_u64()), (Some(AUDIT_QUEUE_MAX as u64), Some(5)));
        assert!(w.audit_next());
        assert_eq!(w.audits.back().unwrap()["message_id"], format!("m{}", AUDIT_QUEUE_MAX + 4)); // newest first
        // mixed plan = union incl. the unassigned pool
        assert_eq!(w.plan(MIXED, 0.0).unwrap().tokens, 1000 + 500 + 50);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn go_live_scoring() {
        let c = cfg();
        let mut g = GoLive::default();
        for _ in 0..49 { g.apply("match"); }
        assert!(!g.reached(&c));
        for _ in 0..9 { g.apply("skipped"); } // no mask: nothing to score
        assert!(!g.reached(&c));
        g.apply("match");
        assert!(g.reached(&c) && g.points == 50.0 && g.clean == 50);
        // mask-caused mismatch: hard reset + tighten
        g.apply("mismatch");
        assert_eq!((g.points, g.resets, g.tightened, g.reached(&c)), (0.0, 1, 1, false));
        for _ in 0..50 { g.apply("match"); }
        assert!(g.reached(&c) && g.required(&c) == 50.0);
        // persisted in state.json; vote-era fields of an old state file are ignored
        let d = tmpdir("golive");
        let mut w = Watcher::open(d.clone(), cfg(), None).unwrap();
        w.golive[0] = g.clone();
        w.save_state();
        assert_eq!(Watcher::open(d.clone(), cfg(), None).unwrap().golive[0], g);
        std::fs::write(d.join("state.json"), r#"{"golive":{"code-rust":{"points":3,"harm":true,"model_limit":2,"clean":3}}}"#).unwrap();
        let w = Watcher::open(d.clone(), cfg(), None).unwrap();
        assert_eq!((w.golive[0].points, w.golive[0].clean), (3.0, 3));
        assert!(w.golive[0].reached(&c) == false);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// per-label vocab file: output tokens | embedding rows of each learned job
    #[test]
    fn vocab_per_label() {
        let mut dm = dims();
        dm.vocab = 10;
        let d = tmpdir("vocab");
        let mut w = Watcher::open(d.clone(), cfg(), Some(dm.clone())).unwrap();
        let mut a = act(&dm, &[0], 100);
        a.out = vec![(1, 2), (2, 1), (99, 1)];
        a.embed = vec![(5, 3)];
        w.observe_scores(sc(&[(0, 0.9), (2, 0.1)]), 1, Some(a), "k");
        let st = w.store.as_ref().unwrap();
        let v = rd(&w, 0, st.o_out(), 20);
        let h = st.hdr(0);
        assert_eq!((h.out_tok, h.emb_rows, h.turns), (3, 3, 1)); // id 99 is past the vocab: ignored
        assert_eq!((v[1], v[2], v[10 + 5], v.iter().sum::<f32>()), (2.0, 1.0, 3.0, 6.0));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// chat-en / chat-vi profiles of the first watcher are merged into chat-general (nothing lost, originals kept).
    #[test]
    fn migrate_language_profiles() {
        let dm = dims();
        let d = tmpdir("migrate");
        let pd = d.join("profiles").join(dm.tag());
        std::fs::create_dir_all(&pd).unwrap();
        let mut en = Profile::zero(&dm); en.learn(&act(&dm, &[0], 100), &dm);
        let mut vi = Profile::zero(&dm); vi.learn(&act(&dm, &[1], 300), &dm); vi.learn(&act(&dm, &[1], 300), &dm);
        std::fs::write(pd.join("chat-en.bin"), en.to_bytes(&dm)).unwrap();
        std::fs::write(pd.join("chat-vi.bin"), vi.to_bytes(&dm)).unwrap();
        add_neuron_file(&pd.join("chat-vi.neuron"), &act(&dm, &[1], 0).neuron, dm.layers, dm.inter, 600).unwrap();
        std::fs::write(d.join("state.json"), r#"{"domain":"chat-vi","pending":["chat-en",1],"turn":7}"#).unwrap();
        std::fs::write(d.join("audits.jsonl"), "{\"message_id\":\"m1\",\"result\":\"skipped\",\"domain\":\"chat-vi\"}\n").unwrap();
        let w = Watcher::open(d.clone(), cfg(), Some(dm.clone())).unwrap();
        let g = &w.profiles[CHAT];
        assert_eq!((g.tokens, g.windows), (700, 3));
        // block masses kept: en's block 0 (no .neuron: spread), vi's block 1 (100 per-neuron + 100 from the .bin rest)
        let l0 = rd(&w, CHAT, 0, 256);
        let blk = |b: usize| l0[b * 64..b * 64 + 64].iter().sum::<f32>();
        assert!((blk(0) - 100.0).abs() < 1e-3 && (blk(1) - 200.0).abs() < 1e-3, "{} {}", blk(0), blk(1));
        let mut c = [0f32; 4];
        assert!(w.store.as_ref().unwrap().read_cent(CHAT, 0, &mut c));
        assert!((c[1] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!((w.domain_name(), w.pending, w.turn), ("chat-general", Some((CHAT, 1)), 7));
        assert!(w.scorecard().iter().any(|r| r["domain"] == "chat-general" && r["skipped"] == 1));
        assert!(pd.join("chat-en.bin.migrated").exists() && pd.join("chat-vi.neuron.migrated").exists());
        assert!(!pd.join("chat-en.bin").exists() && !pd.join("chat-general.bin").exists());
        drop(w);
        assert_eq!(Watcher::open(d.clone(), cfg(), Some(dm)).unwrap().profiles[CHAT].tokens, 700); // once only
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn persistence_reopen() {
        let d = tmpdir("persist");
        {
            let mut w = Watcher::open(d.clone(), cfg(), Some(dims())).unwrap();
            for _ in 0..3 { w.observe(&Feat::of(RUST_T), 100, Some(act(&dims(), &[0, 1], 5000)), "k"); }
            w.note("state_card", &"é".repeat(3000), "proj", "code").unwrap();
            assert_eq!(w.profiles[0].tokens, 15000);
        }
        let w = Watcher::open(d.clone(), cfg(), Some(dims())).unwrap();
        assert_eq!(w.domain_name(), "code-rust");
        assert_eq!(w.profiles[0].tokens, 15000);
        assert_eq!(w.profiles[0].windows, 3);
        assert_eq!(rd(&w, 0, 0, 64).iter().sum::<f32>(), 300.0);
        assert_eq!(w.tokens_observed, 300);
        assert_eq!(w.turn, 3);
        assert!(!w.recent.is_empty());
        let notes = std::fs::read_to_string(d.join("notes.jsonl")).unwrap();
        let n: Value = serde_json::from_str(notes.lines().last().unwrap()).unwrap();
        assert!(n["text"].as_str().unwrap().len() <= NOTE_MAX);
        // other model dims: separate profiles (nothing carried over)
        let w2 = Watcher::open(d.clone(), cfg(), Some(Dims::from_stats(&tmpl(3, 128), 64))).unwrap();
        assert_eq!(w2.profiles[0].tokens, 0);
        // notes compaction keeps the last 50
        let mut w3 = Watcher::open(d.clone(), cfg(), None).unwrap();
        for i in 0..120 { w3.note("fold", &format!("note {}", i), "", "").unwrap(); }
        let k = std::fs::read_to_string(d.join("notes.jsonl")).unwrap().lines().count();
        assert!(k <= 2 * NOTES_KEEP && k >= NOTES_KEEP, "{}", k);
        // an open window survives a restart
        let mut c = cfg(); c.min_turns = 50;
        let d2 = tmpdir("persist_win");
        {
            let mut w = Watcher::open(d2.clone(), c.clone(), None).unwrap();
            for _ in 0..3 { w.observe(&Feat::of(RUST_T), 1, None, "k"); }
            w.note("fold", RUST_T, "", "code").unwrap();
        }
        let w = Watcher::open(d2.clone(), c, None).unwrap();
        assert_eq!((w.win.turns, w.win.folds, w.win.n, w.win.labels[0]), (3, 1, 4, 3));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }

    #[test]
    fn stats_round_trip() {
        let dm = dims();
        let mut s = tmpl(4, 256);
        s.rows = 7;
        s.neuron = (0..4 * 256).map(|i| (i % 7) as f32).collect();
        s.dn_head = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        s.attn_head = vec![7.0, 8.0, 9.0];
        let a = Act::from_stats(&s, &dm).unwrap();
        assert_eq!(a.tokens, 7);
        assert_eq!(a.ffn[0], s.neuron[..64].iter().sum::<f32>());
        assert_eq!(a.neuron, s.neuron);
        assert_eq!(&a.heads[..3], &[1.0, 2.0, 0.0]); // DeltaNet layer 0: 2 heads of hmax 3
        assert_eq!(&a.heads[9..12], &[7.0, 8.0, 9.0]); // attention layer 3
        // with down norms: neuron = Σ|a| · ‖W_down[:, j]‖ (in place)
        let mut s2 = s.clone();
        s2.down_norm = vec![2.0; 4 * 256];
        let a2 = Act::from_stats(&s2, &dm).unwrap();
        assert_eq!((a2.neuron[3], a2.ffn[0]), (6.0, 2.0 * a.ffn[0]));
        let f: Vec<f64> = a.ffn.iter().map(|x| *x as f64).collect();
        let h: Vec<f64> = a.heads.iter().map(|x| *x as f64).collect();
        let back = Act::from_stats(&stats_of(&f, &h, 7, &dm), &dm).unwrap();
        for (x, y) in back.ffn.iter().zip(&a.ffn) { assert!((x - y).abs() < 1e-3); }
        assert_eq!(back.heads, a.heads);
        // recorder off: no activity
        assert!(Act::from_stats(&ActivityStats { enabled: false, ..tmpl(4, 256) }, &dm).is_none());
    }

    #[test]
    fn plan_aggregation() {
        let dm = dims();
        // layer 0: blocks 0,1 hot, 2 a sliver, 3 dead; other layers: no data (all kept)
        let mut f = vec![0.0; 16];
        f[0] = 600.0; f[1] = 399.0; f[2] = 1.0;
        let mut h = vec![0.0; 12];
        h[0] = 10.0;
        let p = plan_for("code-rust", &f, &h, 5000, &dm, 0.995, 0.0);
        assert_eq!(p.cold(), [128, 0, 0, 0]); // blocks 2 + 3 (0.1% <= 0.5%)
        assert_eq!(p.cold_heads_per_layer(), [1, 0, 0, 0]);
        let u = unload::unload_plan(&stats_of(&f, &h, 5000, &dm), 0.995, 64);
        assert!((p.gb_unload - u.unload_bytes as f64 / 1e9).abs() < 1e-12 && u.unload_bytes > 0);
        assert!((p.ms_saved_per_pass - u.pass_ms_saved_a).abs() < 1e-12);
        assert!((p.cold_ffn_frac - 128.0 / 1024.0).abs() < 1e-12);
        assert!((p.footprint_gb - (10e9 - u.unload_bytes as f64) / 1e9).abs() < 1e-9);
        assert!(p.ready && p.fits_cap);
        assert_eq!(plan_for("x", &f, &h, 5000, &dm, 1.0, 0.0).cold(), [0; 4]); // keep all: nothing cold
        // partial last block: inter 100 / gran 64 -> blocks of 64 + 36
        let d2 = Dims::from_stats(&tmpl(1, 100), 64);
        assert_eq!(plan_for("x", &[1.0, 0.0], &[0.0; 2], 1, &d2, 0.995, 0.0).cold(), [36]);
        // per domain accumulation + mixed = union of all labels
        let d = tmpdir("plan");
        let mut w = Watcher::open(d.clone(), cfg(), Some(dm.clone())).unwrap();
        for _ in 0..3 { w.observe(&Feat::of(RUST_T), 1, Some(act(&dm, &[0], 2000)), "k"); }
        for _ in 0..3 { w.observe(&Feat::of(SWIFT_T), 1, Some(act(&dm, &[3], 2000)), "k"); }
        let pr = w.plan(0, 0.0).unwrap();
        let ps = w.plan(1, 0.0).unwrap();
        assert_eq!((pr.cold(), ps.cold()), (&[192u32; 4][..], &[192u32; 4][..]));
        assert_eq!(pr.cold_heads_per_layer(), [1, 1, 1, 2]);
        let pm = w.plan(MIXED, 0.0).unwrap();
        assert_eq!(pm.cold(), [128; 4]);
        // the streamed plan = the unload_plan reference on the same per-neuron data (a label's set)
        let st = w.store.as_ref().unwrap();
        let mut s = stats_of(&[], &[], 0, &dm);
        s.neuron = rd(&w, 0, 0, st.n_ffn());
        for l in 0..4 { let r = st.heads_of(l); let hv = rd(&w, 0, r.start, r.len()); let i = s.kind_index(l);
            match s.kinds[l] { LayerKind::DeltaNet => s.dn_head[i * 2..i * 2 + 2].copy_from_slice(&hv), LayerKind::Attention => s.attn_head[i * 3..i * 3 + 3].copy_from_slice(&hv) } }
        for g in [1usize, 16, 64] {
            for keep in [0.999, 0.995, 0.95, 0.8] {
                let u = unload::unload_plan(&s, keep as f32, g);
                let po = st.plan(Sel::Set(0), keep as f32 as f64, g);
                assert_eq!((po.cold_neurons, po.cold_heads, po.ffn_bytes, po.head_bytes, po.skip_bytes),
                           (u.cold_neurons, u.dn_heads_cold + u.attn_heads_cold, u.ffn_bytes, u.head_bytes, u.skip_bytes_per_pass), "g {} keep {}", g, keep);
            }
        }
        assert_eq!(pm.tokens, pr.tokens + ps.tokens);
        assert_eq!(w.domain_name(), "code-swift");
        let st = w.status(12.0);
        assert_eq!(st["note"], "dry run: nothing unloaded yet");
        assert_eq!(st["profiles"].as_array().unwrap().len(), MIXED + 1);
        assert_eq!(st["unload_plan"]["per_layer_cold"], json!([192, 192, 192, 192]));
        assert_eq!(st["activity"]["jobs"], 6);
        assert_eq!(st["config"]["cut_languages"], json!(["zh"]));
        assert!(w.summary().contains("would unload"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn activity_centroids_learn() {
        let dm = dims();
        let d = tmpdir("learn");
        let mut w = Watcher::open(d.clone(), cfg(), Some(dm.clone())).unwrap();
        for _ in 0..4 { w.observe(&Feat::of(RUST_T), 1, Some(act(&dm, &[0], 100)), "k"); }
        for _ in 0..4 { w.observe(&Feat::of(EN_T), 1, Some(act(&dm, &[3], 100)), "k"); }
        assert!(w.profiles[0].windows >= 3 && w.profiles[CHAT].windows >= 3);
        // same text, different activity: the activity of each kind moves the scores toward its domain
        let mut amb = Feat::of(RUST_T); amb.add(&Feat::of(EN_T)); amb.add(&Feat::of(EN_T));
        let with_rust = w.turn_scores(&amb, Some(&act(&dm, &[0], 100))).unwrap();
        // the centroid cosine streams the store and the job: no heap after setup
        let (a, st) = (act(&dm, &[0], 100), w.store.as_ref().unwrap());
        let src = ActSrc::new(&a, st);
        let mut use_l = [false; NP];
        (use_l[0], use_l[CHAT]) = (true, true);
        let _ = st.score(&src, &use_l);
        let n0 = streamdense_core::alloc::thread_allocs();
        let cos = st.score(&src, &use_l);
        assert_eq!(streamdense_core::alloc::thread_allocs(), n0, "score allocated");
        assert!(cos[0] > 0.99 && cos[CHAT] < 0.5, "{:?}", cos);
        let with_chat = w.turn_scores(&amb, Some(&act(&dm, &[3], 100))).unwrap();
        assert!(with_rust[0] > with_chat[0] && with_chat[CHAT] > with_rust[CHAT]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Similarity gate (2026-11-01) at tiny dims, 2 label slots + the held slot: an A job opposite to A's slot goes
    /// to the held slot (slot A unchanged) and a similar A job ends that hold (held dropped, A unchanged); then an
    /// opposite burst B (no slot, none free) is held in the held slot; an A opposite job meanwhile waits in now only
    /// (held slot busy); B persists over the window -> B replaces the least-used dissimilar slot (C) and B's slot holds
    /// ALL held B jobs + the confirming one; A kept; a similar drift A -> A' folds in; file size constant; notes.
    #[test]
    fn similarity_gate_holds_and_replaces() {
        let dm = dims();
        let d = tmpdir("gate");
        let mut c = cfg(); c.min_turns = 3; c.folds = 1; c.hysteresis = 1;
        let mut w = Watcher::open(d.clone(), c, Some(dm.clone())).unwrap();
        // 2 label slots: reopen the store at a cap holding exactly two
        let path = w.store.as_ref().unwrap().path().to_path_buf();
        let cap = { let st = w.store.as_ref().unwrap(); st.fixed_bytes() + 2 * st.lsz() };
        w.store = None;
        std::fs::remove_file(&path).unwrap();
        w.store = Some(Store::open_cap(&d, dm.clone(), label_names(), MIXED, false, cap).unwrap());
        let total = w.store.as_ref().unwrap().total_bytes();
        let size = || std::fs::metadata(&path).unwrap().len();
        let nf = dm.layers * dm.inter;
        let (a, cc, b) = (0usize, 1usize, 2usize);
        let held_slot = |w: &Watcher| w.store.as_ref().unwrap().held_owner();
        for _ in 0..3 { w.observe_scores(sc(&[(a, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[0], 100)), "k"); }
        w.observe_scores(sc(&[(cc, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[1], 100)), "k");
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.slot_of(a), st.slot_of(cc), st.free_slot(), size()), (Some(0), Some(1), None, total));
        let a_before = rd(&w, a, 0, nf);
        // a job labelled A but opposite to A's slot: held (held slot), slot A unchanged; a similar A job ends the hold
        let sim = w.store.as_ref().unwrap().sim(a, &ActSrc::new(&act(&dm, &[2, 3], 100), w.store.as_ref().unwrap())).unwrap();
        let same = w.store.as_ref().unwrap().sim(a, &ActSrc::new(&act(&dm, &[0], 100), w.store.as_ref().unwrap())).unwrap();
        eprintln!("similarity to slot A: opposite job {:?}, same-profile job {:?}", sim, same);
        assert!(sim.min() < 0.25 && same.min() > 0.99);
        w.observe_scores(sc(&[(a, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[2, 3], 100)), "k");
        assert_eq!((w.held[a].turns, w.store.as_ref().unwrap().hdr(a).turns, held_slot(&w)), (1, 3, Some(a)));
        assert_eq!(rd(&w, a, 0, nf), a_before, "slot A unchanged by its own opposite job");
        assert_eq!(w.store.as_ref().unwrap().hdr(crate::store::HELD).turns, 1);
        w.observe_scores(sc(&[(a, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[0], 100)), "k");
        assert_eq!((w.held[a], w.store.as_ref().unwrap().hdr(a).turns, held_slot(&w)), (Held::default(), 4, None));
        let a_before = rd(&w, a, 0, nf);
        let now0 = w.store.as_ref().unwrap().hdr(crate::store::NOW).turns;
        // opposite burst B (blocks 2, 3): no slot, none free -> held slot
        for i in 1..=2u64 {
            w.observe_scores(sc(&[(b, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[2, 3], 100)), "k");
            assert_eq!(w.held[b].turns, i);
        }
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.slot_of(b), held_slot(&w), st.hdr(a).turns, st.hdr(crate::store::NOW).turns), (None, Some(b), 4, now0 + 2));
        assert_eq!(rd(&w, a, 0, nf), a_before, "slot A unchanged");
        // A opposite while B holds the held slot: now / session only, not counted, logged
        w.observe_scores(sc(&[(a, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[2, 3], 100)), "k");
        assert_eq!((w.held[a].turns, held_slot(&w), w.store.as_ref().unwrap().hdr(a).turns), (0, Some(b), 4));
        assert_eq!(rd(&w, a, 0, nf), a_before);
        // B persists: a fold, then its 3rd held turn -> confirmed -> replaces C (least used, dissimilar); A kept;
        // B's slot = the 2 held jobs + the confirming one (nothing lost)
        w.note("fold", "", "", "").unwrap();
        w.observe_scores(sc(&[(b, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[2, 3], 100)), "k");
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.slot_of(b), st.slot_of(cc), st.slot_of(a), held_slot(&w)), (Some(1), None, Some(0), None));
        assert_eq!((st.hdr(b).turns, st.hdr(b).tokens, st.hdr(a).turns, w.held[b]), (3, 300, 4, Held::default()));
        assert_eq!((w.profiles[cc], size()), (Meta::default(), total));
        let n = rd(&w, b, 0, nf);
        assert!((n[2 * 64] - 3.0 * 100.0 / 64.0).abs() < 1e-5 && n[0] == 0.0 && n[64] == 0.0, "B's slot = 3 B jobs: {}", n[2 * 64]);
        // similar drift A -> A' (blocks 0, 1: cosine 0.71 to A's centroid) folds in normally
        w.observe_scores(sc(&[(a, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[0, 1], 100)), "k");
        assert_eq!((w.store.as_ref().unwrap().hdr(a).turns, w.held[a]), (5, Held::default()));
        assert_eq!(size(), total);
        let notes = std::fs::read_to_string(d.join("notes.jsonl")).unwrap();
        let ns: Vec<Value> = notes.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let kinds = |k: &str| ns.iter().filter(|n| n["kind"] == k).cloned().collect::<Vec<Value>>();
        let (holds, rep) = (kinds("hold"), kinds("replace"));
        assert_eq!((holds.len(), kinds("hold-ended").len(), kinds("hold-wait").len()), (3, 1, 1));
        assert!(holds[0]["sim"]["min"].as_f64().unwrap() < 0.25 && holds[1]["reason"] == "no free slot");
        assert_eq!((rep.len(), rep[0]["label"].as_str(), rep[0]["replaced"].as_str(), rep[0]["held_turns"].as_u64()), (1, Some("code-other"), Some("code-swift"), Some(3)));
        assert!(rep[0]["replaced_usage"].as_f64().unwrap() < 0.5 && rep[0]["replaced_sim"].as_f64().unwrap() < 0.25);
        eprintln!("replace note: {}", rep[0]);
        // the held state survives a restart
        w.observe_scores(sc(&[(cc, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[1], 100)), "k");
        assert_eq!((w.held[cc].turns, held_slot(&w)), (1, Some(cc)));
        let st = w.status(0.0);
        assert_eq!((st["held"]["code-swift"]["turns"].clone(), st["held_slot"].clone()), (json!(1), json!("code-swift")));
        assert_eq!(st["slots"].as_array().unwrap().len(), 2);
        drop(w);
        let mut c = cfg(); c.min_turns = 3; c.folds = 1;
        let w = Watcher::open(d.clone(), c, Some(dm.clone())).unwrap();
        assert_eq!((w.held[cc].turns, held_slot(&w)), (1, Some(cc)));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Test batcher: the batch as a host Vec (the gateway's lives in model memory, `store::TapBatch`).
    struct VecBatch(Mutex<Vec<f32>>);
    impl Batcher for VecBatch {
        fn add(&self, job: &dyn JobSrc) {
            let mut v = self.0.lock().unwrap();
            let mut x = vec![0f32; v.len()];
            job.read(0, &mut x);
            for (a, b) in v.iter_mut().zip(&x) { *a += *b; }
        }
        fn clear(&self) { self.0.lock().unwrap().fill(0.0); }
        fn read(&self, off: usize, out: &mut [f32]) { let v = self.0.lock().unwrap(); for (j, o) in out.iter_mut().enumerate() { *o = v.get(off + j).copied().unwrap_or(0.0); } }
    }

    /// Batching (2026-11-01): consecutive same-label jobs are summed and folded once per `fold_every` jobs or on a
    /// label change; per-label sums stay exact (never two labels in one fold); a crash loses only the open batch;
    /// bytes written per job drop ~fold_every x.
    #[test]
    fn batch_folds_same_label_exactly() {
        let dm = dims();
        let run = |name: &str, fold_every: usize, batched: bool| -> (Watcher, PathBuf, u64) {
            let d = tmpdir(name);
            let mut c = cfg(); c.fold_every = fold_every;
            let mut w = Watcher::open(d.clone(), c, Some(dm.clone())).unwrap();
            if batched { let n = w.store.as_ref().unwrap().n(); w.batcher = Some(Arc::new(VecBatch(Mutex::new(vec![0.0; n])))); }
            let w0 = w.store.as_ref().unwrap().prog.bytes_written.load(Relaxed);
            for i in 0..5 { w.observe_scores(sc(&[(0, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[i % 2], 100 + i as u64)), "k"); }
            let wb = w.store.as_ref().unwrap().prog.bytes_written.load(Relaxed) - w0;
            (w, d, wb)
        };
        let (plain, d0, wb_plain) = run("batch_off", 4, false);
        let (mut w, d, wb_batch) = run("batch_on", 4, true);
        eprintln!("bytes written for 5 jobs: per job {} B, batched (fold_every 4) {} B", wb_plain, wb_batch);
        // 4 jobs folded as one batch, the 5th open
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.hdr(0).turns, st.hdr(0).tokens, w.batch.jobs), (4, 100 + 101 + 102 + 103, 1));
        assert!(wb_batch * 2 < wb_plain, "{} vs {}", wb_batch, wb_plain);
        // label change: the open A batch is folded first, B starts its own (never mixed)
        w.observe_scores(sc(&[(2, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[3], 50)), "k");
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.hdr(0).turns, st.hdr(0).tokens, st.hdr(2).turns, w.batch.jobs), (5, 510, 0, 1));
        w.flush_batch();
        let nf = dm.layers * dm.inter;
        assert_eq!(rd(&w, 0, 0, nf), rd(&plain, 0, 0, nf), "batched sums = per-job sums");
        let b = rd(&w, 2, 0, nf);
        assert!(b[3 * 64] > 0.0 && b[..3 * 64].iter().all(|x| *x == 0.0), "B only in B's slot");
        assert_eq!(w.store.as_ref().unwrap().hdr(crate::store::NOW).turns, 6);
        // crash with an open batch: the store keeps every folded batch, the open one is lost (never half-written)
        w.observe_scores(sc(&[(0, 0.9), (3, 0.1)]), 1, Some(act(&dm, &[0], 100)), "k");
        drop(w);
        let w = Watcher::open(d.clone(), cfg(), Some(dm.clone())).unwrap();
        let st = w.store.as_ref().unwrap();
        assert_eq!((st.hdr(0).turns, st.hdr(2).turns), (5, 1));
        assert_eq!((st.prog.repaired.load(Relaxed), st.prog.lost.load(Relaxed)), (0, 0));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d0);
    }

    /// The pre-2026-11-01 migration of one label, all in RAM (the reference for `OldSrc`): whole files read, the
    /// .neuron vector cloned, the .bin rest spread in place, sparse token counts.
    fn migrate_ref(pd: &Path, name: &str, d: &Dims) -> (Act, Option<Profile>) {
        let prof = std::fs::read(pd.join(format!("{}.bin", name))).ok().and_then(|b| Profile::from_bytes(&b, d));
        let neu = read_neuron_file(&pd.join(format!("{}.neuron", name)), d.layers, d.inter);
        let voc = read_neuron_file(&pd.join(format!("{}.vocab", name)), 2, d.vocab);
        let mut a = Act::zero(d);
        a.neuron = neu.as_ref().map(|x| x.0.clone()).unwrap_or_else(|| vec![0.0; d.layers * d.inter]);
        if let Some(p) = &prof {
            for l in 0..d.layers {
                for b in 0..d.blocks {
                    let r = l * d.inter + b * d.gran..l * d.inter + b * d.gran + d.block_neurons(b);
                    let have: f64 = a.neuron[r.clone()].iter().map(|x| *x as f64).sum();
                    let rest = p.ffn[l * d.blocks + b] - have;
                    if rest > 0.0 { let e = (rest / r.len() as f64) as f32; for x in &mut a.neuron[r] { *x += e; } }
                }
            }
            a.heads = p.heads.iter().map(|x| *x as f32).collect();
        }
        if let Some((v, _, _)) = &voc {
            a.out = (0..d.vocab).filter(|&i| v[i] > 0.0).map(|i| (i as u32, v[i].round() as u32)).collect();
            a.embed = (0..d.vocab).filter(|&i| v[d.vocab + i] > 0.0).map(|i| (i as u32, v[d.vocab + i].round() as u32)).collect();
        }
        (a, prof)
    }
    /// Old-format files of one label: .bin (block masses above / below the .neuron sums, heads), .neuron, .vocab.
    fn old_files(pd: &Path, name: &str, d: &Dims, bin: bool, neu: bool, voc: bool) {
        std::fs::create_dir_all(pd).unwrap();
        let v: Vec<f32> = (0..d.layers * d.inter).map(|i| ((i * 37) % 101) as f32 * 0.37).collect();
        if bin {
            let mut p = Profile::zero(d);
            for (i, x) in p.ffn.iter_mut().enumerate() { *x = if i % 3 == 0 { 0.0 } else { 3000.0 + (i % 17) as f64 * 11.3 }; }
            for (i, x) in p.heads.iter_mut().enumerate() { *x = (i % 5) as f64 * 1.7; }
            for (i, x) in p.cent.iter_mut().enumerate() { *x = (i % 2) as f32; }
            (p.tokens, p.windows) = (1234, 5);
            std::fs::write(pd.join(format!("{}.bin", name)), p.to_bytes(d)).unwrap();
        }
        if neu { add_neuron_file(&pd.join(format!("{}.neuron", name)), &v, d.layers, d.inter, 77).unwrap(); }
        if voc {
            let c: Vec<f32> = (0..2 * d.vocab).map(|i| match i % 4 { 0 => 0.0, 1 => 2.5, 2 => 0.4, _ => (i % 9) as f32 }).collect();
            add_neuron_file(&pd.join(format!("{}.vocab", name)), &c, 2, d.vocab, 0).unwrap();
        }
    }

    /// 2026-11-01 RAM fix: the streamed migration source gives bit-identical values to the old in-RAM one, for every
    /// file combination, at chunk-aligned and odd offsets.
    #[test]
    fn migrate_stream_matches_in_ram() {
        let mut dm = Dims::from_stats(&tmpl(4, 1000), 64); // last block of each layer partial (40 neurons)
        dm.vocab = 3001;
        let d = tmpdir("migstream");
        let st = Store::open(&d, dm.clone(), label_names(), MIXED, false).unwrap();
        for (k, (bin, neu, voc)) in [(true, true, true), (true, false, false), (false, true, true), (false, false, true), (true, true, false)].into_iter().enumerate() {
            let pd = d.join(format!("p{}", k));
            old_files(&pd, "x", &dm, bin, neu, voc);
            let (a, prof) = migrate_ref(&pd, "x", &dm);
            let neu_f = open_neuron_file(&pd.join("x.neuron"), dm.layers, dm.inter).map(|x| x.0);
            let voc_f = open_neuron_file(&pd.join("x.vocab"), 2, dm.vocab).map(|x| x.0);
            let new = OldSrc::new(&st, prof.as_ref(), neu_f, voc_f);
            let old = ActSrc::new(&a, &st);
            assert_eq!(JobSrc::rows(&new), (0, 0));
            let n = st.n();
            for step in [CH, 777, n + 5] {
                let mut off = 0;
                while off < n {
                    let len = step.min(n + 3 - off);
                    let (mut x, mut y) = (vec![0f32; len], vec![1f32; len]);
                    JobSrc::read(&old, off, &mut x);
                    JobSrc::read(&new, off, &mut y);
                    for j in 0..len { assert_eq!(x[j].to_bits(), y[j].to_bits(), "case {} off {} j {}", k, off, j); }
                    off += len;
                }
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 2026-11-01 RAM fix, measured at the 27B dims (64 x 17408, vocab 248320; no model): phys_footprint left behind by
    /// one label's migration, old in-RAM path vs the streamed one. macOS libmalloc keeps freed large blocks dirty (the
    /// large-block cache), so the old path's whole-file reads and clones stay in the footprint after they are freed.
    #[test]
    fn migrate_footprint_27b() {
        use streamdense_core::footprint::self_footprint;
        let kinds = (0..64).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let t = ActivityStats { enabled: true, layers: 64, hidden: 5120, inter: 17408, kinds, dn_kh: 16, dn_dk: 128, dn_vh: 48, dn_dv: 128,
                                heads: 24, kv_heads: 4, head_dim: 256, conv_k: 4, group: 64, bits: 4, ..Default::default() };
        let mut dm = Dims::from_stats(&t, 64);
        dm.vocab = 248320;
        let d = tmpdir("mig27b");
        let st = Store::open(&d, dm.clone(), label_names(), MIXED, false).unwrap();
        let pd = d.join("p");
        old_files(&pd, "x", &dm, true, true, true);
        // streamed (first: nothing freed before it to reuse)
        let (h0, f0) = (streamdense_core::alloc::heap_now(), self_footprint());
        let prof = std::fs::read(pd.join("x.bin")).ok().and_then(|b| Profile::from_bytes(&b, &dm));
        let src = OldSrc::new(&st, prof.as_ref(), open_neuron_file(&pd.join("x.neuron"), 64, 17408).map(|x| x.0),
                              open_neuron_file(&pd.join("x.vocab"), 2, 248320).map(|x| x.0));
        let mut sum = 0f64;
        let mut c = vec![0f32; CH];
        for off in (0..st.n()).step_by(CH) { let l = CH.min(st.n() - off); JobSrc::read(&src, off, &mut c[..l]); sum += c[..l].iter().map(|x| *x as f64).sum::<f64>(); }
        let h_new = streamdense_core::alloc::heap_now() as i64 - h0 as i64;
        drop((src, prof, c));
        let f_new = self_footprint() as i64 - f0 as i64;
        // old in-RAM path
        let f1 = self_footprint();
        let (a, prof) = migrate_ref(&pd, "x", &dm);
        let src = ActSrc::new(&a, &st);
        let mut sum2 = 0f64;
        let mut c = vec![0f32; CH];
        for off in (0..st.n()).step_by(CH) { let l = CH.min(st.n() - off); JobSrc::read(&src, off, &mut c[..l]); sum2 += c[..l].iter().map(|x| *x as f64).sum::<f64>(); }
        drop(src);
        drop((a, prof, c));
        let f_old = self_footprint() as i64 - f1 as i64;
        eprintln!("migration of one 27B-size label: footprint left +{:.2} MB streamed (heap live during it {:.3} MB) vs +{:.2} MB in RAM",
                  f_new as f64 / 1e6, h_new as f64 / 1e6, f_old as f64 / 1e6);
        assert_eq!(sum, sum2);
        assert!(h_new < 600_000, "streamed migration heap {} B", h_new);
        assert!(f_new < 2_000_000, "streamed migration left {} B in the footprint", f_new);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// No activity recorder (12 GB mode, --no-activity): `open_opts(.., store = false)` creates no neuron store and no
    /// predictor table (mask "dry" would build one), the text watcher still labels, decides and saves its state; an
    /// activity handed in anyway is ignored. Control: the same dir with store = true creates them.
    #[test]
    fn no_store_without_recorder() {
        let dm = dims();
        let d = tmpdir("no_store");
        let mut c = cfg(); c.mask = "dry".into();
        let mut w = Watcher::open_opts(d.clone(), c.clone(), Some(dm.clone()), false).unwrap();
        assert!(w.store.is_none() && w.mk.is_none());
        for _ in 0..3 { w.observe_scores(sc(&[(0, 0.9), (CHAT, 0.1)]), 10, Some(act(&dm, &[0], 100)), "k"); }
        assert_eq!(w.turn, 3);
        assert_eq!(w.act_jobs, 0, "no store: activity is not learned");
        let s = w.status(0.0);
        assert_eq!(s["turn"], json!(3), "{}", s);
        assert!(d.join("state.json").is_file());
        assert!(!d.join("store").exists(), "store dir created without the recorder");
        assert!(std::fs::read_dir(&d).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().ends_with(".bin")));
        drop(w);
        // reopen without the store keeps the text state
        let w = Watcher::open_opts(d.clone(), c.clone(), Some(dm.clone()), false).unwrap();
        assert_eq!(w.turn, 3);
        assert!(!d.join("store").exists());
        drop(w);
        // control: with the store the files appear (the path the gateway takes with the recorder on)
        let w = Watcher::open_opts(d.clone(), c, Some(dm.clone()), true).unwrap();
        let sd = w.store.as_ref().unwrap().dir.clone();
        assert!(sd.join("store.bin").is_file() && sd.join("predict.bin").is_file(), "{}", sd.display());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Read mask through the watcher: mask off = nothing built or written; dry = one pooled profile (blocks never active in any
    /// label are masked), the predictor scores each job, exploration schedules at idle; audits stay "skipped" (not applied).
    #[test]
    fn mask_off_and_dry() {
        let dm = dims();
        let d = tmpdir("mask_off");
        let mut w = Watcher::open(d.clone(), cfg(), Some(dm.clone())).unwrap();
        w.observe_scores(sc(&[(0, 0.9), (CHAT, 0.1)]), 1, Some(act(&dm, &[0], 100)), "k");
        assert!(!w.rebuild_mask(1.0) && w.mk.is_none());
        assert_eq!(w.status(0.0)["mask"], json!({"mode": "off", "applied": false}));
        assert!(!w.store.as_ref().unwrap().dir.join("predict.bin").exists());
        let _ = std::fs::remove_dir_all(&d);

        let d = tmpdir("mask_dry");
        let mut c = cfg(); c.mask = "dry".into(); c.keep_frac = 0.995;
        let mut w = Watcher::open(d.clone(), c, Some(dm.clone())).unwrap();
        // code-rust uses block 0, chat-general block 1 (every layer); blocks 2, 3 never active
        for (lab, b) in [(0usize, 0usize), (CHAT, 1), (0, 0)] {
            let a = act(&dm, &[b], 100);
            w.observe_scores(sc(&[(lab, 0.9), (3, 0.1)]), 1, Some(a.clone()), "k");
            let src = ActSrc::new(&a, w.store.as_ref().unwrap());
            let s = w.score_job(&src, &predict::TokList(&[7, 8, b as u32 + 100]), None).unwrap();
            assert_eq!(s.active, dm.layers + dm.layers, "1 block + head 0 per layer: {:?}", s);
        }
        assert!(w.rebuild_mask(now_s()));
        let m = w.status(0.0)["mask"].clone();
        assert_eq!((m["mode"].clone(), m["applied"].clone()), (json!("dry"), json!(false)));
        assert_eq!(m["profile"]["masked_blocks"], json!(2 * dm.layers), "{}", m["profile"]);
        assert_eq!(m["profile"]["per_layer_masked"], json!(vec![2; dm.layers]));
        assert!(m["profile"]["skip_bytes_per_pass"].as_u64().unwrap() > 0);
        assert_eq!(m["predictor"]["scored"], json!(3));
        let mk = w.mk.as_ref().unwrap();
        assert!(mk.mask.ffn(0, 0) && mk.mask.ffn(0, 1) && !mk.mask.ffn(0, 2) && !mk.mask.ffn(3, 3));
        assert!(w.explore_idle(now_s()));
        assert_eq!(w.status(0.0)["mask"]["explore"]["last"]["read"], json!(false), "dry: schedule only");
        assert!(!w.mask_active(0));
        let _ = std::fs::remove_dir_all(&d);
    }
}
