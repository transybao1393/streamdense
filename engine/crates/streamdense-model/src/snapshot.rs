//! Engine state snapshots (prefix cache): save the full decode state of a `QwenEngine` after a token prefix and
//! restore it later (same engine, a new session, or a fresh process after a restart), so only the remainder of a
//! prompt is prefilled. Built for the ~1.25K-token Claude Code prefix (system prompt + tool defs) on Qwen3.8-27B.
//!
//! # API (stable; streamdense-gateway calls these)
//! ```ignore
//! use streamdense_model::snapshot::{self, Snapshot, SnapshotCache};
//! // file (preferred: no extra RAM, the arena is read/written in place)
//! eng.save_snapshot(&path, &tokens)?;            // tokens = every id fed so far; tokens.len() must == eng.pos()
//! let tokens: Vec<u32> = eng.restore_snapshot(&path)?;   // eng.pos() == tokens.len() afterwards
//! // in RAM (a heap copy: counts against the footprint cap, see SIZE)
//! let s: Snapshot = eng.snapshot(&tokens)?;  eng.restore(&s)?;  s.bytes();  s.write(&path)?;  Snapshot::read(&path)?;
//! // size without an engine (config only)
//! snapshot::body_bytes(&cfg, kv, pos, mtp_len, with_mtp) -> u64
//! // disk cache: ~/Library/Caches/streamdense/snapshots (or $SD_SNAPSHOT_DIR), keyed by
//! //   model id + structural fingerprint + kv format + mtp + engine/format version + FNV-64 of the token ids + length
//! let cache = SnapshotCache::open_default(model_id)?;      // model_id: e.g. canonical model dir (+ weights mtime)
//! cache.path_for(&eng, &tokens) -> PathBuf;   cache.save(&eng, &tokens)? -> PathBuf
//! cache.longest_prefix(&eng, &prompt[..prompt.len() - 1]) -> Option<(PathBuf, usize)>   // n tokens covered
//! // 12 GB mode (format 2 = format-1 header + body, a "compact" header object, then the prompt sums; gateway compact.rs)
//! eng.save_snapshot_compact(&path, &tokens, &CompactMeta {..}, &sums)?;
//! let (tokens, meta) = eng.restore_snapshot_compact(&path, &want_meta, &mut sums)?;   // refuses format 1 / another align
//! cache.mode = "c8x".into();   // 12 GB-mode keys (`..f2c8x-..`): never mixed with full-mode files
//! ```
//! Full mode is unchanged (format 1, same bytes; old files load). `restore_snapshot` / `restore` refuse format 2 and
//! `restore_snapshot_compact` refuses format 1, each with a clear error.
//! Typical gateway turn: `reset()`; if `longest_prefix(prompt[..len-1])` = (p, n): `restore_snapshot(p)` and
//! `feed_stream(_mtp)(&prompt[n..], ..)`; else feed the whole prompt. To populate: `reset()`, feed exactly the shared
//! prefix with `feed_stream(_mtp)(prefix, 1, ..)` (the one generated token is discarded), `cache.save(eng, prefix)`.
//!
//! # Contract
//! - Call only between turns (no command buffer in flight). Restore overwrites ctrl (pos, role, mtp_len, tok, draft,
//!   ...), the host position, rec[0..pos), the live DeltaNet S + conv copy, KV rows [0..pos) of every attention layer,
//!   and (both sides MTP) mctrl + MTP KV rows [0..mtp_len). Everything else decode reads is rewritten before it is
//!   read (KV rows >= pos, the DeltaNet checkpoint copy, scratch, staging), so restore is bit-exact: restore(P) +
//!   decode N == prefill P + decode N, token for token and logit for logit (tests/snapshot.rs).
//! - The state after a snapshot is the state after `feed_stream(_mtp)` returned: the last emitted token is NOT fed
//!   (callers keep it, as for any continuation). ctrl.tok[0] / draft are restored too.
//! - Portable across max_ctx (rows are re-strided), across plain <-> MTP engines (the live DeltaNet copy is stored
//!   role-free and restored into copy 0 with role 0; MTP state is skipped when either side lacks it — a plain
//!   snapshot in an MTP engine leaves the MTP KV stale: lower acceptance on the prefix, output still exact), NOT
//!   across models / dims / kv format (structural fingerprint checked) — use `SnapshotCache` keys for model identity.
//!
//! # SIZE (body; header adds ~5 B/token of JSON token ids + ~300 B)
//! ```text
//! kv_tok   = 2 (K,V) * kv_heads * (head_dim * eb + q8 ? head_dim/32 * 4 : 0)        eb = 2 bf16, 1 q8
//! body     = 2*72 (ctrl, mctrl)  + 16 * pos (rec)
//!          + n_deltanet * dn_vh * dn_dv * dn_dk * 4      (S, f32, live copy only)
//!          + n_deltanet * (conv_k - 1) * conv_dim * 2    (conv, bf16; conv_dim = 2*dn_kh*dn_dk + dn_vh*dn_dv)
//!          + n_attention * kv_tok * pos  + (mtp ? kv_tok * mtp_len : 0)
//! ```
//! Qwen3.8-27B (config: 48 DeltaNet + 16 attention, kv_heads 4, head_dim 256, dn 16k/48v x 128, conv_k 4), bf16 KV:
//! S 150,994,944 + conv 2,949,120 (fixed, 153.9 MB) + KV 65,536 B/token (+4,096 MTP) + rec 16 B/token.
//! At pos = 1259: plain 236,474,176 B (236.5 MB); with MTP (mtp_len = 1259) 241,631,040 B (241.6 MB).
//! An in-RAM copy of that is ~1/4 of the ~1 GB headroom under the 16.5 GB cap: prefer the file path (restore
//! preads straight into the arena: 0 extra RAM; ~0.08 s at ~3 GB/s SSD, faster from the page cache).
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use streamdense_metal::SdCtrl;

use crate::qwen_config::{LayerKind, QwenConfig};
use crate::qwen_engine::{KvFormat, QwenEngine};
use crate::qwen_layers::{DnState, KvRef};

/// File magic + format version. Bump `FORMAT` when the layout or any state semantics change.
pub const MAGIC: &[u8; 8] = b"SDSNAP\0\0";
pub const FORMAT: u32 = 1;
/// Format of a 12 GB-mode snapshot (streamdense-gateway compact.rs): the format-1 header + body, a `"compact"` header
/// object (`CompactMeta`) and, after the body, the streamed prefill's prompt sums (f32 LE, `layers` x `width`).
/// Full mode keeps writing format 1 (unchanged; old files load). A format-2 file is refused by the full-model
/// restore (`restore_snapshot` / `restore`) and a format-1 file by `restore_snapshot_compact`, each with a clear error;
/// older builds refuse format 2 by its number.
pub const FORMAT_COMPACT: u32 = 2;

/// The optional 12 GB-mode section: what a streamed prefill needs to resume exactly as in-memory reuse does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactMeta {
    pub layers: usize,
    /// floats per layer row of the sums (2 x inter: an | ag)
    pub width: usize,
    /// piece alignment of the streamed read (prefill_k, or lcm(prefill_t, prefill_k) with the fast read)
    pub align: usize,
    /// fast streamed read (`QwenOptions.ffn_stream_fast`)
    pub fast: bool,
    /// tokens per streamed pass when saved (informational: pieces are multiples of `align`)
    pub piece: usize,
    /// stream-read tokens in the sums (decoded tokens sit in the KV only); <= pos
    pub held: usize,
}

impl CompactMeta {
    fn json(&self) -> Value {
        json!({"layers": self.layers, "width": self.width, "align": self.align, "fast": self.fast, "piece": self.piece, "held": self.held})
    }
    fn from_json(j: &Value) -> Result<CompactMeta, String> {
        let u = |k: &str| j[k].as_u64().map(|v| v as usize).ok_or(format!("snapshot header: compact.{}", k));
        Ok(CompactMeta { layers: u("layers")?, width: u("width")?, align: u("align")?, fast: j["fast"].as_bool().unwrap_or(false),
                         piece: u("piece")?, held: u("held")? })
    }
    /// Bytes of the sums section after the body.
    pub fn sums_bytes(&self) -> u64 { (self.layers * self.width * 4) as u64 }
}

const REFUSE_COMPACT: &str = "made in 12 GB mode (it carries the compact FFN's prompt sums); the full model cannot resume from it: refused";
const REFUSE_FULL: &str = "made by the full model (format 1, no prompt sums); 12 GB mode cannot resume from it: refused";
/// Engine version baked into cache keys (a new build may change numerics: fresh prefill != old snapshot).
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
const CTRL: usize = std::mem::size_of::<SdCtrl>();

/// FNV-1a 64.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    h
}
/// FNV-1a 64 of token ids (little-endian u32).
pub fn tokens_hash(t: &[u32]) -> u64 {
    let mut v = Vec::with_capacity(t.len() * 4);
    for x in t { v.extend_from_slice(&x.to_le_bytes()); }
    fnv64(&v)
}

fn kv_name(kv: KvFormat) -> &'static str { match kv { KvFormat::Bf16 => "bf16", KvFormat::Q8 => "q8" } }

/// Structural fingerprint: every dim the state layout depends on + kv format (not model identity).
pub fn fingerprint(cfg: &QwenConfig, kv: KvFormat) -> u64 {
    let s = format!("{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{}", cfg.kinds, kv, cfg.hidden, cfg.kv_heads, cfg.head_dim, cfg.dn_kh,
                    cfg.dn_vh, cfg.dn_dk, cfg.dn_dv, cfg.conv_k, cfg.vocab);
    fnv64(s.as_bytes())
}

/// KV bytes per token per attention layer (K + V, + q8 scale/min).
pub fn kv_token_bytes(cfg: &QwenConfig, kv: KvFormat) -> usize {
    let hd = cfg.head_dim;
    2 * cfg.kv_heads * match kv { KvFormat::Bf16 => hd * 2, KvFormat::Q8 => hd + (hd / 32) * 4 }
}

/// Snapshot body bytes (see SIZE in the module doc).
pub fn body_bytes(cfg: &QwenConfig, kv: KvFormat, pos: usize, mtp_len: usize, with_mtp: bool) -> u64 {
    let ndn = cfg.n_deltanet();
    let fixed = 2 * CTRL + ndn * DnState::s_layer_bytes(cfg, 1) + ndn * DnState::conv_layer_bytes(cfg);
    let kvt = kv_token_bytes(cfg, kv);
    (fixed + 16 * pos + cfg.n_attention() * kvt * pos + if with_mtp { kvt * mtp_len } else { 0 }) as u64
}

/// Parsed snapshot header (JSON in the file, after MAGIC + u64 length).
#[derive(Clone, Debug)]
pub struct SnapHeader {
    pub format: u32,
    pub engine: String,
    pub fingerprint: u64,
    pub kv: KvFormat,
    pub pos: usize,
    pub mtp_len: usize,
    pub mtp: bool,
    pub body: u64,
    pub tokens: Vec<u32>,
    /// 12 GB mode (format 2 only): the streamed prefill's resume state; the sums follow the body.
    pub compact: Option<CompactMeta>,
}

impl SnapHeader {
    fn to_bytes(&self) -> Vec<u8> {
        let mut j = json!({
            "format": self.format, "engine": self.engine, "fingerprint": format!("{:016x}", self.fingerprint),
            "kv": kv_name(self.kv), "pos": self.pos, "mtp_len": self.mtp_len, "mtp": self.mtp, "body": self.body,
            "tokens": self.tokens,
        });
        if let Some(c) = &self.compact { j["compact"] = c.json(); } // format 1: no key (byte-identical to before)
        let js = serde_json::to_vec(&j).unwrap();
        let mut v = Vec::with_capacity(16 + js.len());
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&(js.len() as u64).to_le_bytes());
        v.extend_from_slice(&js);
        v
    }
    fn from_json(b: &[u8]) -> Result<SnapHeader, String> {
        let j: Value = serde_json::from_slice(b).map_err(|e| format!("snapshot header: {}", e))?;
        let u = |k: &str| j[k].as_u64().ok_or(format!("snapshot header: missing {}", k));
        let kv = match j["kv"].as_str() { Some("bf16") => KvFormat::Bf16, Some("q8") => KvFormat::Q8,
                                          o => return Err(format!("snapshot header: kv {:?}", o)) };
        let fp = j["fingerprint"].as_str().and_then(|s| u64::from_str_radix(s, 16).ok()).ok_or("snapshot header: fingerprint")?;
        let tokens = j["tokens"].as_array().ok_or("snapshot header: tokens")?.iter()
            .map(|x| x.as_u64().map(|v| v as u32).ok_or("snapshot header: token id".to_string())).collect::<Result<Vec<_>, _>>()?;
        Ok(SnapHeader { format: u("format")? as u32, engine: j["engine"].as_str().unwrap_or("").to_string(), fingerprint: fp, kv,
                        pos: u("pos")? as usize, mtp_len: u("mtp_len")? as usize, mtp: j["mtp"].as_bool().unwrap_or(false),
                        body: u("body")?, tokens,
                        compact: if j["compact"].is_object() { Some(CompactMeta::from_json(&j["compact"])?) } else { None } })
    }
    /// Read MAGIC + header from a reader; leaves it at the body.
    pub fn read_from(r: &mut dyn Read) -> Result<SnapHeader, String> {
        let mut m = [0u8; 16];
        r.read_exact(&mut m).map_err(|e| format!("snapshot: {}", e))?;
        if &m[..8] != MAGIC { return Err("snapshot: bad magic".into()); }
        let n = u64::from_le_bytes(m[8..].try_into().unwrap()) as usize;
        if n > 64 << 20 { return Err(format!("snapshot: header {} B too large", n)); }
        let mut b = vec![0u8; n];
        r.read_exact(&mut b).map_err(|e| format!("snapshot header: {}", e))?;
        let h = Self::from_json(&b)?;
        match (h.format, h.compact.is_some()) {
            (FORMAT, false) | (FORMAT_COMPACT, true) => {}
            (FORMAT_COMPACT, false) => return Err("snapshot: format 2 without its compact section".into()),
            (FORMAT, true) => return Err("snapshot: format 1 with a compact section".into()),
            (f, _) => return Err(format!("snapshot: format {} (this build reads {} and {})", f, FORMAT, FORMAT_COMPACT)),
        }
        if h.tokens.len() != h.pos { return Err(format!("snapshot: {} tokens != pos {}", h.tokens.len(), h.pos)); }
        Ok(h)
    }
    pub fn read_file(path: &Path) -> Result<SnapHeader, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Self::read_from(&mut f)
    }
}

/// In-RAM snapshot: header + body bytes (the body is `bytes()` long; heap memory, counted by phys_footprint).
pub struct Snapshot {
    pub header: SnapHeader,
    pub body: Vec<u8>,
}

impl Snapshot {
    pub fn bytes(&self) -> usize { self.body.len() }
    pub fn tokens(&self) -> &[u32] { &self.header.tokens }
    /// Write in the file format (`restore_snapshot` / `Snapshot::read` can load it).
    pub fn write(&self, path: &Path) -> Result<(), String> {
        if self.header.compact.is_some() { return Err("snapshot: an in-RAM copy holds no prompt sums; cannot write a 12 GB-mode file".into()); }
        atomic_write(path, |f| {
            f.write_all(&self.header.to_bytes()).map_err(|e| e.to_string())?;
            f.write_all(&self.body).map_err(|e| e.to_string())
        })
    }
    pub fn read(path: &Path) -> Result<Snapshot, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let header = SnapHeader::read_from(&mut f)?;
        let mut body = vec![0u8; header.body as usize];
        f.read_exact(&mut body).map_err(|e| format!("{}: body: {}", path.display(), e))?;
        Ok(Snapshot { header, body })
    }
}

/// Write via `path.tmp.<pid>` + rename (a concurrent reader never sees a partial file).
fn atomic_write(path: &Path, w: impl FnOnce(&mut File) -> Result<(), String>) -> Result<(), String> {
    if let Some(d) = path.parent() { if !d.as_os_str().is_empty() { std::fs::create_dir_all(d).map_err(|e| format!("{}: {}", d.display(), e))?; } }
    let tmp = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
    let r = (|| {
        let mut f = File::create(&tmp).map_err(|e| format!("{}: {}", tmp.display(), e))?;
        w(&mut f).map_err(|e| format!("{}: {}", tmp.display(), e))?;
        f.sync_all().map_err(|e| format!("{}: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
    })();
    if r.is_err() { let _ = std::fs::remove_file(&tmp); }
    r
}

// The sums section is f32 little-endian; Apple silicon (the only target) is little-endian.
#[cfg(not(target_endian = "little"))]
compile_error!("snapshot sums are written as native f32: little-endian only");
fn f32_bytes(v: &[f32]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) } }
fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) } }

/// One contiguous arena range of the body, in body order: (arena offset, bytes).
type Span = (usize, usize);

/// KV rows [0..n) of one attention layer: per K / V (and q8 K / V scales), per head, a contiguous run.
fn kv_spans(cfg: &QwenConfig, kv: &KvRef, n: usize, out: &mut Vec<Span>) {
    let hd = cfg.head_dim;
    let eb = if kv.q8 { 1 } else { 2 };
    for base in [kv.k, kv.v] {
        for h in 0..cfg.kv_heads { out.push((base + h * kv.max_ctx * hd * eb, n * hd * eb)); }
    }
    if kv.q8 {
        let sb = (hd / 32) * 4;
        for base in [kv.ksb, kv.vsb] {
            for h in 0..cfg.kv_heads { out.push((base + h * kv.max_ctx * sb, n * sb)); }
        }
    }
}

impl QwenEngine {
    fn mctrl_off(&self) -> Option<usize> { self.mtp_s().as_ref().map(|m| m.mctrl) }

    /// Body layout for this engine: (spans, which ones exist). `role` = the live DeltaNet copy to read/write;
    /// `mtp` = include mctrl + MTP KV (both snapshot and engine have MTP). Spans absent on this side (mctrl / MTP KV
    /// when `mtp` is false) are returned as `None` so the body offsets stay those of the file.
    fn spans(&self, pos: usize, mtp_len: usize, role: usize, snap_mtp: bool) -> Vec<(Option<usize>, usize)> {
        let c = &self.cfg;
        let mut v: Vec<(Option<usize>, usize)> = Vec::new();
        v.push((Some(self.sl().ctrl.off), CTRL));
        let use_mtp = snap_mtp && self.mtp_kv().is_some();
        v.push((if use_mtp { self.mctrl_off() } else { None }, CTRL));
        v.push((Some(self.sl().rec.off), 16 * pos));
        let one = DnState::s_layer_bytes(c, 1);
        let copies = self.dn().copies;
        let ndn = c.n_deltanet();
        for d in 0..ndn { v.push((Some(self.dn().s + d * one * copies + role * one), one)); }
        let cl = DnState::conv_layer_bytes(c) * ndn;
        v.push((Some(self.dn().conv + role * cl), cl));
        let mut ks = Vec::new();
        for l in 0..c.layers {
            if c.kinds[l] == LayerKind::Attention { kv_spans(c, self.kv()[l].as_ref().unwrap(), pos, &mut ks); }
        }
        v.extend(ks.into_iter().map(|(o, n)| (Some(o), n)));
        if snap_mtp {
            let mut ms = Vec::new();
            match self.mtp_kv().as_ref() {
                Some(kv) => { kv_spans(c, kv, mtp_len, &mut ms); v.extend(ms.into_iter().map(|(o, n)| (Some(o), n))); }
                None => {   // engine without MTP reading an MTP snapshot: skip those bytes
                    let n = kv_token_bytes(c, self.opt.kv) * mtp_len;
                    if n > 0 { v.push((None, n)); }
                }
            }
        }
        v
    }

    fn snap_header(&self, tokens: &[u32]) -> Result<SnapHeader, String> {
        let c = self.ctrl_read();
        let pos = self.pos();
        if c.pos as usize != pos { return Err(format!("snapshot: host pos {} != ctrl.pos {} (command buffer in flight?)", pos, c.pos)); }
        if tokens.len() != pos { return Err(format!("snapshot: {} tokens != pos {}", tokens.len(), pos)); }
        let mtp = self.mtp_kv().is_some();
        let mtp_len = if mtp { (c.mtp_len as usize).min(pos) } else { 0 };
        Ok(SnapHeader { format: FORMAT, engine: ENGINE_VERSION.to_string(), fingerprint: fingerprint(&self.cfg, self.opt.kv),
                        kv: self.opt.kv, pos, mtp_len, mtp, body: body_bytes(&self.cfg, self.opt.kv, pos, mtp_len, mtp), tokens: tokens.to_vec(),
                        compact: None })
    }

    /// Source spans of the current state (live DeltaNet copy = ctrl.role); `None` = zeros (mctrl of a plain engine).
    fn save_spans(&self, h: &SnapHeader) -> Vec<(Option<usize>, usize)> {
        let role = (self.ctrl_read().role as usize).min(self.dn().copies - 1);
        self.spans(h.pos, h.mtp_len, role, h.mtp)
    }
    fn span_bytes(&self, o: Option<usize>, n: usize) -> &[u8] {
        static ZERO: [u8; CTRL] = [0; CTRL];
        match o {
            Some(o) => unsafe { std::slice::from_raw_parts(self.arena.base().add(o), n) },
            None => &ZERO[..n], // only the mctrl slot is ever absent on the saving side
        }
    }

    fn check_restore(&self, h: &SnapHeader) -> Result<(), String> {
        let fp = fingerprint(&self.cfg, self.opt.kv);
        if h.fingerprint != fp { return Err(format!("snapshot: fingerprint {:016x} != engine {:016x} (other model dims or kv format)", h.fingerprint, fp)); }
        if h.kv != self.opt.kv { return Err("snapshot: kv format differs".into()); }
        if h.pos == 0 || h.pos > self.opt.max_ctx { return Err(format!("snapshot: pos {} not in 1..={}", h.pos, self.opt.max_ctx)); }
        if h.mtp_len > h.pos { return Err(format!("snapshot: mtp_len {} > pos {}", h.mtp_len, h.pos)); }
        let want = body_bytes(&self.cfg, h.kv, h.pos, h.mtp_len, h.mtp);
        if h.body != want { return Err(format!("snapshot: body {} B != expected {} B", h.body, want)); }
        Ok(())
    }

    /// After the body is in place: live DeltaNet copy is copy 0 → role 0; host position; plain-target fix-ups.
    fn finish_restore(&self, h: &SnapHeader) {
        let c = self.ctrl();
        c.role = 0;
        c.pos = h.pos as u32;
        if !(h.mtp && self.mtp_kv().is_some()) {
            // no MTP state restored: an MTP engine's prefill_mtp moves mtp_len up to pos (stale MTP KV, exact output)
            c.mtp_len = if self.mtp_kv().is_some() { 0 } else { c.mtp_len };
        }
        self.set_pos(h.pos);
    }

    /// In-RAM snapshot of the current state. `tokens` = every id fed so far (len == pos()). Nothing may be in flight.
    pub fn snapshot(&self, tokens: &[u32]) -> Result<Snapshot, String> {
        let h = self.snap_header(tokens)?;
        let mut body = Vec::with_capacity(h.body as usize);
        for (o, n) in self.save_spans(&h) { body.extend_from_slice(self.span_bytes(o, n)); }
        debug_assert_eq!(body.len() as u64, h.body);
        Ok(Snapshot { header: h, body })
    }

    /// Restore an in-RAM snapshot. Nothing may be in flight.
    pub fn restore(&self, s: &Snapshot) -> Result<(), String> {
        let h = &s.header;
        if h.compact.is_some() { return Err(format!("snapshot: {}", REFUSE_COMPACT)); }
        self.check_restore(h)?;
        if s.body.len() as u64 != h.body { return Err(format!("snapshot: body {} B != header {} B", s.body.len(), h.body)); }
        let mut at = 0;
        for (o, n) in self.spans(h.pos, h.mtp_len, 0, h.mtp) {
            if let Some(o) = o {
                unsafe { std::ptr::copy_nonoverlapping(s.body.as_ptr().add(at), self.arena.base().add(o), n); }
            }
            at += n;
        }
        self.finish_restore(h);
        Ok(())
    }

    /// Write the current state to `path` (atomic: tmp + rename), straight from the arena (no RAM copy).
    pub fn save_snapshot(&self, path: &Path, tokens: &[u32]) -> Result<(), String> {
        let h = self.snap_header(tokens)?;
        let spans = self.save_spans(&h);
        atomic_write(path, |f| {
            f.write_all(&h.to_bytes()).map_err(|e| e.to_string())?;
            for (o, n) in spans { f.write_all(self.span_bytes(o, n)).map_err(|e| e.to_string())?; }
            Ok(())
        })
    }

    /// Load a snapshot file into the engine (read straight into the arena; no RAM copy). Returns the token ids the
    /// state covers (pos() == their count). On error the engine state is undefined: `reset()` before reuse.
    pub fn restore_snapshot(&self, path: &Path) -> Result<Vec<u32>, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let h = SnapHeader::read_from(&mut f)?;
        if h.compact.is_some() { return Err(format!("{}: snapshot {}", path.display(), REFUSE_COMPACT)); }
        self.check_restore(&h)?;
        self.read_body(&mut f, &h, path)?;
        self.finish_restore(&h);
        Ok(h.tokens)
    }

    /// The body of `h` from `f` straight into the arena (reset on a short read).
    fn read_body(&self, f: &mut File, h: &SnapHeader, path: &Path) -> Result<(), String> {
        let mut skip = Vec::new();
        for (o, n) in self.spans(h.pos, h.mtp_len, 0, h.mtp) {
            let dst: &mut [u8] = match o {
                Some(o) => unsafe { std::slice::from_raw_parts_mut(self.arena.base().add(o), n) },
                None => { skip.resize(n, 0); &mut skip[..] }
            };
            if let Err(e) = f.read_exact(dst) {
                self.reset();
                return Err(format!("{}: body: {}", path.display(), e));
            }
        }
        Ok(())
    }

    /// 12 GB mode: write the current state + the streamed prefill's resume state (`meta`, `sums` [layers][width])
    /// as a format-2 file (atomic, straight from the arena; the sums are written from the caller's rows, no copy).
    pub fn save_snapshot_compact(&self, path: &Path, tokens: &[u32], meta: &CompactMeta, sums: &[Vec<f32>]) -> Result<(), String> {
        if sums.len() != meta.layers || sums.iter().any(|r| r.len() != meta.width) {
            return Err(format!("snapshot: sums are not {} x {}", meta.layers, meta.width));
        }
        if meta.held > tokens.len() { return Err(format!("snapshot: held {} > {} tokens", meta.held, tokens.len())); }
        let mut h = self.snap_header(tokens)?;
        h.format = FORMAT_COMPACT;
        h.compact = Some(meta.clone());
        let spans = self.save_spans(&h);
        atomic_write(path, |f| {
            f.write_all(&h.to_bytes()).map_err(|e| e.to_string())?;
            for (o, n) in spans { f.write_all(self.span_bytes(o, n)).map_err(|e| e.to_string())?; }
            for r in sums { f.write_all(f32_bytes(r)).map_err(|e| e.to_string())?; }
            Ok(())
        })
    }

    /// 12 GB mode: load a format-2 file (state into the arena, sums into `sums`, read in place). `want`: layers,
    /// width, align and fast must match (piece and held are the file's). A format-1 (full-model) file is refused
    /// before anything is touched. Returns (tokens, the file's meta). On a body / sums error the engine is reset and
    /// `sums` is undefined (the caller zeroes it).
    pub fn restore_snapshot_compact(&self, path: &Path, want: &CompactMeta, sums: &mut [Vec<f32>]) -> Result<(Vec<u32>, CompactMeta), String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let h = SnapHeader::read_from(&mut f)?;
        let Some(m) = h.compact.clone() else { return Err(format!("{}: snapshot {}", path.display(), REFUSE_FULL)) };
        if (m.layers, m.width, m.align, m.fast) != (want.layers, want.width, want.align, want.fast) {
            return Err(format!("{}: snapshot of another 12 GB-mode setup (layers {} width {} align {} fast {}; this engine {} / {} / {} / {}): refused",
                               path.display(), m.layers, m.width, m.align, m.fast, want.layers, want.width, want.align, want.fast));
        }
        if m.held > h.pos { return Err(format!("{}: snapshot: held {} > pos {}", path.display(), m.held, h.pos)); }
        if sums.len() != m.layers || sums.iter().any(|r| r.len() != m.width) { return Err(format!("snapshot: sums buffer is not {} x {}", m.layers, m.width)); }
        self.check_restore(&h)?;
        self.read_body(&mut f, &h, path)?;
        for r in sums.iter_mut() {
            if let Err(e) = f.read_exact(f32_bytes_mut(r)) {
                self.reset();
                return Err(format!("{}: sums: {}", path.display(), e));
            }
        }
        self.finish_restore(&h);
        Ok((h.tokens, m))
    }

    /// Full logits row `row` (f32 [vocab]) of the last head pass (test / debug hook: after a plain K = 1 decode pass
    /// row 0 holds the logits of the token just emitted).
    pub fn head_logits(&self, row: usize) -> Vec<f32> {
        let v = self.cfg.vocab;
        let p = unsafe { self.arena.base().add(self.io().logits) as *const f32 };
        unsafe { std::slice::from_raw_parts(p.add(row * v), v).to_vec() }
    }

    /// Record of position p: (argmax token, bf16 top-1 logit, top-2 logit) as the head wrote it.
    pub fn record(&self, p: usize) -> (u32, f32, f32) {
        let r = self.arena.slice_mut::<f32>(self.sl().rec);
        (r[4 * p].to_bits(), r[4 * p + 1], r[4 * p + 2])
    }
}

/// `~/Library/Caches/streamdense/snapshots`, or `$SD_SNAPSHOT_DIR`.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("SD_SNAPSHOT_DIR") { if !d.is_empty() { return PathBuf::from(d); } }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("Library/Caches/streamdense/snapshots")
}

/// Disk snapshot cache. File name: `{model:016x}-{fingerprint:016x}-{kv}-{plain|mtp}-e{ENGINE_VERSION}f{FORMAT}-{tokhash:016x}-{n}.sdsnap`.
pub struct SnapshotCache {
    pub dir: PathBuf,
    /// FNV-64 of the caller's model id (e.g. canonical model dir + weights mtime)
    pub model: u64,
    /// "" = full model (format 1 keys, unchanged); else the 12 GB-mode tag (e.g. "c8x"): keys `..f2<mode>-..`, so
    /// the two modes never see each other's files.
    pub mode: String,
}

impl SnapshotCache {
    pub fn open(dir: PathBuf, model_id: &str) -> Result<SnapshotCache, String> {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
        Ok(SnapshotCache { dir, model: fnv64(model_id.as_bytes()), mode: String::new() })
    }
    pub fn open_default(model_id: &str) -> Result<SnapshotCache, String> { Self::open(default_cache_dir(), model_id) }

    /// Everything of the key except the tokens.
    pub fn key_prefix(&self, eng: &QwenEngine) -> String {
        let (f, m) = if self.mode.is_empty() { (FORMAT, "") } else { (FORMAT_COMPACT, self.mode.as_str()) };
        format!("{:016x}-{:016x}-{}-{}-e{}f{}{}-", self.model, fingerprint(&eng.cfg, eng.opt.kv), kv_name(eng.opt.kv),
                if eng.mtp_kv().is_some() { "mtp" } else { "plain" }, ENGINE_VERSION, f, m)
    }
    pub fn path_for(&self, eng: &QwenEngine, tokens: &[u32]) -> PathBuf {
        self.dir.join(format!("{}{:016x}-{}.sdsnap", self.key_prefix(eng), tokens_hash(tokens), tokens.len()))
    }
    /// Save the engine's current state under its key (tokens.len() == eng.pos()).
    pub fn save(&self, eng: &QwenEngine, tokens: &[u32]) -> Result<PathBuf, String> {
        let p = self.path_for(eng, tokens);
        eng.save_snapshot(&p, tokens)?;
        Ok(p)
    }
    /// The cached snapshot covering the longest prefix of `tokens` (n <= tokens.len()): (path, n). Only files of this
    /// engine's key prefix; the header's token ids are compared (hash collisions cannot match). Pass
    /// `&prompt[..len - 1]` so at least one token remains to feed.
    pub fn longest_prefix(&self, eng: &QwenEngine, tokens: &[u32]) -> Option<(PathBuf, usize)> {
        let kp = self.key_prefix(eng);
        let mut best: Option<(PathBuf, usize)> = None;
        for e in std::fs::read_dir(&self.dir).ok()?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with(&kp) || !name.ends_with(".sdsnap") { continue; }
            let n: usize = match name.trim_end_matches(".sdsnap").rsplit('-').next().and_then(|s| s.parse().ok()) { Some(n) => n, None => continue };
            if n == 0 || n > tokens.len() || best.as_ref().is_some_and(|b| b.1 >= n) { continue; }
            if !name.contains(&format!("-{:016x}-{}.sdsnap", tokens_hash(&tokens[..n]), n)) { continue; }
            match SnapHeader::read_file(&e.path()) {
                Ok(h) if h.tokens[..] == tokens[..n] => best = Some((e.path(), n)),
                _ => {}
            }
        }
        best
    }
}
