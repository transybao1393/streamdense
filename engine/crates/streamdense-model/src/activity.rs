//! activity recorder (`QwenOptions.activity`, on by default; CLI `--no-activity`).
//! After every layer of every main-model pass (decode, MTP verify, deep verify, exact and fast prefill, teacher; not
//! the MTP draft layer) `act_layer` (activity.metal) reads the rows the layer left behind and adds f32 sums into its
//! own arena region. It writes nothing another kernel reads, so outputs are unchanged bit for bit.
//! - neuron[l][j] = Σ_t |a_tj|, a = silu(gate x) · up x (the bf16 FFN activation io.f, the down projection input).
//!   The contribution weight down_norm[l][j] = ‖W_down[:, j]‖ (dequantized q4 column norm) is computed once on the
//!   CPU at the first snapshot; neuron · down_norm bounds the neuron's share of the residual updates.
//! - dn_head[d][h] = Σ_t ‖yn_t[h·dv ..]‖ (gated, normed DeltaNet value-head output = out_proj input block);
//!   attn_head[a][h] = Σ_t ‖ao_t[h·hd ..]‖ (gated attention output of q head h = o_proj input block).
//! - resid[l] = Σ_t ‖h_out − h_in‖ over passes of <= PREV_ROWS rows only (decode / verify / exact chunks; fast
//!   prefill passes skip it, `resid_rows` counts the rows that have it).
//! - rows / passes: GPU counters bumped by layer 0, so passes encoded ahead and never committed do not count. Verify
//!   passes count all their rows (rejected drafts included); prefill rows count too.
//! Cost: one `act_begin` (exact passes) + one `act_layer` per layer, ~ceil(inter / 256) + heads + 1 groups reading
//! k·(inter + mixer + 2·hidden) bf16 (27B, k = 1: ~60 KB per layer vs ~200 MB of weights): expected < 1% of a pass;
//! measure with `--no-activity` (same binary).
//! RAM (arena): f32 [layers·inter + n_dn·dn_vh + n_attn·heads + layers] + 16 B counters + MTP layer region + u32
//! token counters [2][vocab] + bf16 prev [PREV_ROWS][hidden] = 6.6 MB on the 27B (all model memory); host: down_norm
//! (4.5 MB) only when `snapshot` is used (the gateway's watcher reads the norms from its disk store instead).
//! Reset (2026-11-01, heat-on-disk) ZEROES the region in place: no host baseline, no f32 stagnation across windows.
//! Call it when the engine is idle (the gateway: the job hook, after the job); a pass in flight would lose its adds.
//! Reads are not synchronized with the GPU (a pass in flight may be partly counted).
//! coverage (all deltas since reset like the above):
//! - mtp_neuron [inter] / mtp_head [heads]: the MTP draft layer (act_layer on its own small region, MTP passes only:
//!   warm-up, first draft, every draft step); mtp_rows / mtp_passes its counters; mtp_down_norm = its down column norms.
//! - vocab[id]: emitted output tokens, counted on the CPU in `log_out` (the gateway worker's on_tok, i.e. plain, MTP
//!   depth 1 and deep spec alike; prefill tokens are never passed there). Zero GPU cost.
//! - embed_rows[id]: embedding rows read from SSD (EmbedReader::read_rows after a successful read: prompt, fed and
//!   draft tokens as actually read). Both are per-window u32 atomic counters in the recorder's arena region (model
//!   memory, 4 B per id each: 1 MB each for 248,320 ids; exact up to 4.29e9 per id per window, `saturated` counts adds
//!   lost past that), zeroed by `reset` / `take_tokens` (one window = one gateway job). `snapshot` returns them dense,
//!   `snapshot_lite` leaves them empty; the watcher reads them in place (`TokCounts::with_counts`).
//! Answer library (watcher `library = true`; off = nothing allocated): `mark_prompt` (worker, first generated token) copies
//!   the FFN sums to a host buffer (layers·inter f32, 4.5 MB on the 27B); `take_prompt_mark` swaps it out (sd-heat).
//! - scripts[id]: script class per id (`SCRIPT_NAMES`), set by the caller from the tokenizer (`set_scripts`), used by
//!   `vocab_by_script` / `by_script`.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use serde_json::{json, Value};
use streamdense_core::{bf16_to_f32, write_safetensors, DType, SafeIndex};
use streamdense_metal::{Buffer, Enc};

use crate::qwen_config::{LayerKind, QwenConfig};
use crate::qwen_layers::PassIo;
use crate::weights::Q4b;

/// Recorder hand-off gate (2026-11-01, fold off the job hook): the gateway's job hook CLOSES it after a job (the
/// recorder now holds exactly that job), the consumer thread (sd-heat) reads + resets the recorder and OPENS it. The
/// next job's first recorder write waits only while it is closed: every pass encode with the recorder on
/// (`encode_begin` / the MTP layer's `encode_after`) and every arena token-counter add. Open by default (nothing else
/// ever closes it), so `qwen` / tests never wait. `waits` / `wait_us` count the waits.
pub struct RecGate { closed: AtomicBool, m: Mutex<bool>, cv: Condvar, pub waits: AtomicU64, pub wait_us: AtomicU64 }
impl RecGate {
    pub const fn new() -> RecGate { RecGate { closed: AtomicBool::new(false), m: Mutex::new(false), cv: Condvar::new(), waits: AtomicU64::new(0), wait_us: AtomicU64::new(0) } }
    pub fn close(&self) { let mut g = self.m.lock().unwrap_or_else(|e| e.into_inner()); *g = true; self.closed.store(true, Ordering::SeqCst); }
    pub fn open(&self) { let mut g = self.m.lock().unwrap_or_else(|e| e.into_inner()); *g = false; self.closed.store(false, Ordering::SeqCst); self.cv.notify_all(); }
    pub fn is_closed(&self) -> bool { self.closed.load(Ordering::SeqCst) }
    /// Block while closed (fast path: one atomic load).
    pub fn wait_open(&self) {
        if !self.closed.load(Ordering::SeqCst) { return; }
        let t0 = std::time::Instant::now();
        let mut g = self.m.lock().unwrap_or_else(|e| e.into_inner());
        while *g { g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner()); }
        self.waits.fetch_add(1, Ordering::Relaxed);
        self.wait_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}
impl Default for RecGate { fn default() -> Self { RecGate::new() } }
/// The process's recorder gate (one engine per gateway process).
pub static GATE: RecGate = RecGate::new();

/// Rows of the residual snapshot `prev` (exact passes have <= 8 rows).
pub const PREV_ROWS: usize = 8;
/// MLX q4 layout of every projection: 4-bit values, one bf16 scale + one bf16 bias per 64 input columns.
pub const Q4_GROUP: usize = 64;
pub const Q4_BITS: usize = 4;
const TG: usize = 256;

/// Arena offsets of the recorder (absolute), carried in `PassIo.act`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ActIo {
    pub neuron: usize,
    pub dn_head: usize,
    pub attn_head: usize,
    pub resid: usize,
    pub cnt: usize,
    pub prev: usize,
    /// MTP draft layer: f32 [inter] FFN | [heads] attention heads | [1] unused resid, then 4 u32 counters
    pub mtp: usize,
    /// u32 token counters [2][vocab]: emitted output tokens | embedding rows read
    pub tok: usize,
}

/// f32 values of the accumulator region (before the 4 u32 counters).
fn n_f32(cfg: &QwenConfig) -> usize {
    cfg.layers * cfg.inter + cfg.n_deltanet() * cfg.dn_vh + cfg.n_attention() * cfg.heads + cfg.layers
}
/// f32 values of the MTP layer region (before its 4 u32 counters).
pub fn n_mtp(cfg: &QwenConfig) -> usize { cfg.inter + cfg.heads + 1 }
/// (accumulator bytes incl. the MTP layer region and the token counters, prev bytes).
pub fn sizes(cfg: &QwenConfig) -> (usize, usize) { (n_f32(cfg) * 4 + 16 + n_mtp(cfg) * 4 + 16 + 2 * cfg.vocab * 4, PREV_ROWS * cfg.hidden * 2) }
pub fn pipelines() -> [&'static str; 2] { ["act_begin", "act_layer"] }

impl ActIo {
    pub fn new(cfg: &QwenConfig, acc: usize, prev: usize) -> ActIo {
        let dn_head = acc + cfg.layers * cfg.inter * 4;
        let attn_head = dn_head + cfg.n_deltanet() * cfg.dn_vh * 4;
        let resid = attn_head + cfg.n_attention() * cfg.heads * 4;
        let cnt = resid + cfg.layers * 4;
        let mtp = cnt + 16;
        ActIo { neuron: acc, dn_head, attn_head, resid, cnt, prev, mtp, tok: mtp + n_mtp(cfg) * 4 + 16 }
    }
}

/// Before layer 0: keep h_in for the residual sum (passes of <= PREV_ROWS rows).
pub(crate) fn encode_begin(e: &Enc, cfg: &QwenConfig, io: &PassIo) {
    let Some(a) = io.act else { return };
    GATE.wait_open();
    if io.k > PREV_ROWS { return; }
    let n = io.k * cfg.hidden;
    e.pso("act_begin").b(0, io.h).b(1, a.prev).u(2, n as u32).threads((n, 1), TG);
}

/// After model layer `l` (no-op without a recorder). l == cfg.layers = the MTP draft layer (attention): its own
/// region and counters (rows / passes of MTP passes), no residual sum.
pub(crate) fn encode_after(e: &Enc, cfg: &QwenConfig, io: &PassIo, l: usize) {
    let Some(a) = io.act else { return };
    if l == cfg.layers {
        GATE.wait_open();
        let (hm, nm) = (a.mtp + cfg.inter * 4, n_mtp(cfg));
        e.pso("act_layer").b(0, io.f).b(1, io.ao).b(2, io.h).b(3, a.prev)
            .b(4, a.mtp).b(5, hm).b(6, a.mtp + (nm - 1) * 4).b(7, a.mtp + nm * 4)
            .u(8, cfg.inter as u32).u(9, cfg.heads as u32).u(10, cfg.head_dim as u32).u(11, cfg.hidden as u32).u(12, io.k as u32).u(13, 2)
            .groups((cfg.inter.div_ceil(TG) + cfg.heads + 1, 1), TG);
        return;
    }
    if l > cfg.layers { return; }
    let kind = cfg.kinds[l];
    let idx = cfg.kinds[..l].iter().filter(|x| **x == kind).count();
    let (mo, heads, hd, ah) = match kind {
        LayerKind::DeltaNet => (io.yn, cfg.dn_vh, cfg.dn_dv, a.dn_head + idx * cfg.dn_vh * 4),
        LayerKind::Attention => (io.ao, cfg.heads, cfg.head_dim, a.attn_head + idx * cfg.heads * 4),
    };
    let flags = (io.k <= PREV_ROWS) as u32 | (((l == 0) as u32) << 1);
    e.pso("act_layer").b(0, io.f).b(1, mo).b(2, io.h).b(3, a.prev)
        .b(4, a.neuron + l * cfg.inter * 4).b(5, ah).b(6, a.resid + l * 4).b(7, a.cnt)
        .u(8, cfg.inter as u32).u(9, heads as u32).u(10, hd as u32).u(11, cfg.hidden as u32).u(12, io.k as u32).u(13, flags)
        .groups((cfg.inter.div_ceil(TG) + heads + 1, 1), TG);
}

/// Usage statistics since the last reset (see the module doc). Per-layer vectors are row-major [layer][index].
#[derive(Clone, Debug, Default)]
pub struct ActivityStats {
    pub enabled: bool,
    pub rows: u64,
    pub passes: u64,
    pub resid_rows: u64,
    pub layers: usize,
    pub hidden: usize,
    pub inter: usize,
    pub kinds: Vec<LayerKind>,
    pub dn_kh: usize,
    pub dn_dk: usize,
    pub dn_vh: usize,
    pub dn_dv: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub conv_k: usize,
    /// q4 layout (Q4_GROUP, Q4_BITS)
    pub group: usize,
    pub bits: usize,
    /// resident weight bytes (arena "w:" regions) and the whole resident estimate (arena + process reserve)
    pub weight_bytes: u64,
    pub resident_bytes: u64,
    /// [layers][inter] Σ|a|
    pub neuron: Vec<f32>,
    /// [layers][inter] ‖W_down[:, j]‖ (empty when not computed)
    pub down_norm: Vec<f32>,
    /// [n_deltanet][dn_vh], [n_attention][heads], [layers]
    pub dn_head: Vec<f32>,
    pub attn_head: Vec<f32>,
    pub resid: Vec<f32>,
    /// [vocab] emitted output tokens per id, [vocab] embedding rows read per id (empty in `dims` and `snapshot_lite`)
    pub vocab: Vec<u32>,
    pub embed_rows: Vec<u32>,
    /// MTP draft layer: rows / passes, [inter] Σ|a|, [inter] ‖W_down[:, j]‖ (empty when not computed / no MTP), [heads]
    pub mtp_rows: u64,
    pub mtp_passes: u64,
    pub mtp_neuron: Vec<f32>,
    pub mtp_down_norm: Vec<f32>,
    pub mtp_head: Vec<f32>,
    /// [vocab] script class per id (index into SCRIPT_NAMES); empty = unknown (`ActivityTap::set_scripts` not called)
    pub scripts: Vec<u8>,
}

impl ActivityStats {
    /// Empty stats with the model dims (no data).
    pub fn dims(cfg: &QwenConfig) -> ActivityStats {
        ActivityStats { layers: cfg.layers, hidden: cfg.hidden, inter: cfg.inter, kinds: cfg.kinds.clone(), dn_kh: cfg.dn_kh,
                        dn_dk: cfg.dn_dk, dn_vh: cfg.dn_vh, dn_dv: cfg.dn_dv, heads: cfg.heads, kv_heads: cfg.kv_heads,
                        head_dim: cfg.head_dim, conv_k: cfg.conv_k, group: Q4_GROUP, bits: Q4_BITS, ..Default::default() }
    }
    pub fn n_deltanet(&self) -> usize { self.kinds.iter().filter(|k| **k == LayerKind::DeltaNet).count() }
    pub fn n_attention(&self) -> usize { self.layers - self.n_deltanet() }
    /// Index of layer `l` among the layers of its kind.
    pub fn kind_index(&self, l: usize) -> usize { self.kinds[..l].iter().filter(|x| **x == self.kinds[l]).count() }
    /// FFN contribution of layer `l`: Σ|a_j| · ‖W_down[:, j]‖ (Σ|a_j| alone without down norms).
    pub fn ffn_contrib(&self, l: usize) -> Vec<f32> {
        let n = &self.neuron[l * self.inter..(l + 1) * self.inter];
        if self.down_norm.len() != self.neuron.len() { return n.to_vec(); }
        n.iter().zip(&self.down_norm[l * self.inter..(l + 1) * self.inter]).map(|(a, w)| a * w).collect()
    }
    /// Mixer head sums of layer `l` (DeltaNet value heads or attention q heads).
    pub fn head_mass(&self, l: usize) -> &[f32] {
        let i = self.kind_index(l);
        match self.kinds[l] {
            LayerKind::DeltaNet => &self.dn_head[i * self.dn_vh..(i + 1) * self.dn_vh],
            LayerKind::Attention => &self.attn_head[i * self.heads..(i + 1) * self.heads],
        }
    }

    /// Compact summary (the serve op / gateway endpoint): counts, and per layer the fraction of FFN neurons / heads
    /// that holds 99.5% of the contribution mass (lower = more concentrated), mean ‖Δh‖ per row.
    pub fn summary(&self) -> Value {
        if !self.enabled { return json!({"enabled": false}); }
        let r4 = |x: f64| if x.is_finite() { json!((x * 1e4).round() / 1e4) } else { Value::Null };
        let ffn: Vec<f64> = (0..self.layers).map(|l| frac_for_mass(&self.ffn_contrib(l), 0.995)).collect();
        let top1: Vec<f64> = (0..self.layers).map(|l| top_share(&self.ffn_contrib(l), 0.01)).collect();
        let heads: Vec<f64> = (0..self.layers).map(|l| frac_for_mass(self.head_mass(l), 0.995)).collect();
        let mean = |v: &[f64]| { let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
                                 if f.is_empty() { f64::NAN } else { f.iter().sum::<f64>() / f.len() as f64 } };
        let of = |k: LayerKind| -> Vec<f64> { (0..self.layers).filter(|&l| self.kinds[l] == k).map(|l| heads[l]).collect() };
        json!({"enabled": true, "rows": self.rows, "passes": self.passes, "resid_rows": self.resid_rows,
            "layers": self.layers, "inter": self.inter, "dn_heads": self.dn_vh, "attn_heads": self.heads,
            "down_norm": !self.down_norm.is_empty(),
            "ffn_frac_995_mean": r4(mean(&ffn)), "ffn_top1pct_share_mean": r4(mean(&top1)),
            "dn_head_frac_995_mean": r4(mean(&of(LayerKind::DeltaNet))), "attn_head_frac_995_mean": r4(mean(&of(LayerKind::Attention))),
            "ffn_frac_995": ffn.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "head_frac_995": heads.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "resid_per_row": self.resid.iter().map(|x| r4(if self.resid_rows > 0 { *x as f64 / self.resid_rows as f64 } else { f64::NAN }))
                .collect::<Vec<_>>(),
            "out_tokens": self.vocab.iter().map(|&x| x as u64).sum::<u64>(),
            "embed_rows": self.embed_rows.iter().map(|&x| x as u64).sum::<u64>(),
            "mtp_rows": self.mtp_rows, "mtp_passes": self.mtp_passes,
            "vocab_by_script": self.vocab_by_script().into_iter().filter(|(_, n)| *n > 0).map(|(s, n)| (s, json!(n))).collect::<serde_json::Map<_, _>>()})
    }

    /// Per-script totals of `vocab` (all SCRIPT_NAMES in order, zeros included); [("unknown", total)] without scripts.
    pub fn vocab_by_script(&self) -> Vec<(String, u64)> { self.by_script(&self.vocab) }
    /// Same for any per-id count vector (e.g. `embed_rows`).
    pub fn by_script(&self, v: &[u32]) -> Vec<(String, u64)> {
        if self.scripts.len() != v.len() || v.is_empty() {
            return vec![("unknown".to_string(), v.iter().map(|&x| x as u64).sum())];
        }
        let mut t = [0u64; SCRIPT_NAMES.len()];
        for (&n, &c) in v.iter().zip(&self.scripts) { t[(c as usize).min(SCRIPT_NAMES.len() - 1)] += n as u64; }
        SCRIPT_NAMES.iter().zip(t).map(|(s, n)| (s.to_string(), n)).collect()
    }

    /// Full snapshot as one safetensors file (F32 tensors + `meta` U32 + `kinds` U8, 0 = DeltaNet, 1 = attention).
    pub fn write(&self, path: &str) -> Result<(), String> {
        let lo = |x: u64| x as u32;
        let hi = |x: u64| (x >> 32) as u32;
        let meta: Vec<u32> = vec![lo(self.rows), hi(self.rows), lo(self.passes), hi(self.passes), lo(self.resid_rows), hi(self.resid_rows),
            self.layers as u32, self.hidden as u32, self.inter as u32, self.dn_kh as u32, self.dn_dk as u32, self.dn_vh as u32,
            self.dn_dv as u32, self.heads as u32, self.kv_heads as u32, self.head_dim as u32, self.conv_k as u32, self.group as u32,
            self.bits as u32, lo(self.weight_bytes), hi(self.weight_bytes), lo(self.resident_bytes), hi(self.resident_bytes),
            lo(self.mtp_rows), hi(self.mtp_rows), lo(self.mtp_passes), hi(self.mtp_passes)];
        let kinds: Vec<u8> = self.kinds.iter().map(|k| (*k == LayerKind::Attention) as u8).collect();
        let b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mb: Vec<u8> = meta.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (nb, db, dh, ah, rb) = (b(&self.neuron), b(&self.down_norm), b(&self.dn_head), b(&self.attn_head), b(&self.resid));
        let li = [self.layers, self.inter];
        let (sd, sa, sl, sm, sk) = ([self.n_deltanet(), self.dn_vh], [self.n_attention(), self.heads], [self.layers], [meta.len()], [kinds.len()]);
        let dn_shape = [if self.down_norm.is_empty() { 0 } else { self.layers }, self.inter];
        let u = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (vb, eb, mn, md, mh) = (u(&self.vocab), u(&self.embed_rows), b(&self.mtp_neuron), b(&self.mtp_down_norm), b(&self.mtp_head));
        let (svb, seb, smn, smd, smh, ssc) = ([self.vocab.len()], [self.embed_rows.len()], [self.mtp_neuron.len()], [self.mtp_down_norm.len()],
                                              [self.mtp_head.len()], [self.scripts.len()]);
        write_safetensors(path, &[("meta", DType::U32, &sm, &mb), ("kinds", DType::U8, &sk, &kinds), ("neuron", DType::F32, &li, &nb),
            ("down_norm", DType::F32, &dn_shape, &db), ("dn_head", DType::F32, &sd, &dh), ("attn_head", DType::F32, &sa, &ah),
            ("resid", DType::F32, &sl, &rb), ("vocab", DType::U32, &svb, &vb), ("embed_rows", DType::U32, &seb, &eb),
            ("mtp_neuron", DType::F32, &smn, &mn), ("mtp_down_norm", DType::F32, &smd, &md), ("mtp_head", DType::F32, &smh, &mh),
            ("scripts", DType::U8, &ssc, &self.scripts)])
    }

    /// Read a snapshot written by `write` (files from before the coverage counters: the new fields come back empty / 0).
    pub fn read(path: &str) -> Result<ActivityStats, String> {
        let idx = SafeIndex::open_file(path)?;
        let data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        let raw = |n: &str| -> Result<&[u8], String> {
            let t = idx.get(n).ok_or_else(|| format!("{}: no tensor {}", path, n))?;
            Ok(&data[t.off as usize..t.off as usize + t.nbytes])
        };
        let f = |n: &str| -> Result<Vec<f32>, String> { Ok(raw(n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()) };
        let m: Vec<u32> = raw("meta")?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if m.len() < 23 { return Err(format!("{}: meta has {} values", path, m.len())); }
        let w = |i: usize| m[i] as u64 | (m[i + 1] as u64) << 32;
        let kinds = raw("kinds")?.iter().map(|&k| if k == 1 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let has = |n: &str| idx.get(n).is_some();
        let fo = |n: &str| -> Result<Vec<f32>, String> { if has(n) { f(n) } else { Ok(Vec::new()) } };
        let uo = |n: &str| -> Result<Vec<u32>, String> {
            if !has(n) { return Ok(Vec::new()); }
            Ok(raw(n)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let wo = |i: usize| if m.len() >= i + 2 { w(i) } else { 0 };
        Ok(ActivityStats { enabled: true, rows: w(0), passes: w(2), resid_rows: w(4), layers: m[6] as usize, hidden: m[7] as usize,
            inter: m[8] as usize, kinds, dn_kh: m[9] as usize, dn_dk: m[10] as usize, dn_vh: m[11] as usize, dn_dv: m[12] as usize,
            heads: m[13] as usize, kv_heads: m[14] as usize, head_dim: m[15] as usize, conv_k: m[16] as usize, group: m[17] as usize,
            bits: m[18] as usize, weight_bytes: w(19), resident_bytes: w(21), neuron: f("neuron")?, down_norm: f("down_norm")?,
            dn_head: f("dn_head")?, attn_head: f("attn_head")?, resid: f("resid")?, vocab: uo("vocab")?, embed_rows: uo("embed_rows")?,
            mtp_rows: wo(23), mtp_passes: wo(25), mtp_neuron: fo("mtp_neuron")?, mtp_down_norm: fo("mtp_down_norm")?,
            mtp_head: fo("mtp_head")?, scripts: if has("scripts") { raw("scripts")?.to_vec() } else { Vec::new() } })
    }
}

/// Fraction of entries (largest first) needed to reach `keep` of the total mass; NaN when the total is 0.
pub fn frac_for_mass(v: &[f32], keep: f64) -> f64 {
    let total: f64 = v.iter().map(|x| *x as f64).sum();
    if v.is_empty() || !(total > 0.0) { return f64::NAN; }
    let mut s: Vec<f32> = v.to_vec();
    s.sort_unstable_by(|a, b| b.total_cmp(a));
    let (mut c, mut n) = (0f64, 0usize);
    for x in s { if c >= keep * total { break; } c += x as f64; n += 1; }
    n as f64 / v.len() as f64
}
/// Share of the mass held by the top `frac` of entries; NaN when the total is 0.
pub fn top_share(v: &[f32], frac: f64) -> f64 {
    let total: f64 = v.iter().map(|x| *x as f64).sum();
    if v.is_empty() || !(total > 0.0) { return f64::NAN; }
    let mut s: Vec<f32> = v.to_vec();
    s.sort_unstable_by(|a, b| b.total_cmp(a));
    let n = ((v.len() as f64 * frac).ceil() as usize).max(1);
    s[..n].iter().map(|x| *x as f64).sum::<f64>() / total
}

/// Per token id u32 counters of one window (`vocab`, `embed_rows`): bumped from any thread (relaxed atomics; ids
/// >= len are ignored), saturating at u32::MAX (`saturated` counts the adds lost there), zeroed by `clear` / `take`.
/// 4 B per id, in the recorder's arena region (`in_arena`, model memory) or on the heap (`new`: tests, no recorder).
pub struct TokCounts { p: *const AtomicU32, n: usize, _own: Option<Box<[AtomicU32]>>, _mem: Option<Mem>, saturated: AtomicU64 }
unsafe impl Send for TokCounts {}
unsafe impl Sync for TokCounts {}
impl std::fmt::Debug for TokCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "TokCounts({} ids, arena {})", self.n, self._mem.is_some()) }
}
impl TokCounts {
    pub fn new(n: usize) -> TokCounts {
        let own: Box<[AtomicU32]> = (0..n).map(|_| AtomicU32::new(0)).collect();
        TokCounts { p: own.as_ptr(), n, _own: Some(own), _mem: None, saturated: AtomicU64::new(0) }
    }
    /// Counters at byte offset `off` of the arena buffer (zeroed by the caller; 4-byte aligned).
    pub(crate) fn in_arena(buf: &Buffer, off: usize, n: usize) -> TokCounts {
        let p = (streamdense_metal::contents(buf) as usize + off) as *const AtomicU32;
        TokCounts { p, n, _own: None, _mem: Some(Mem(buf.clone())), saturated: AtomicU64::new(0) }
    }
    fn c(&self) -> &[AtomicU32] { if self.n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(self.p, self.n) } } }
    /// The counters in place (no copy).
    pub fn with_counts<R>(&self, f: impl FnOnce(&[AtomicU32]) -> R) -> R { f(self.c()) }
    pub fn add(&self, ids: &[u32]) {
        if self._mem.is_some() { GATE.wait_open(); }
        let c = self.c();
        for &i in ids {
            let Some(c) = c.get(i as usize) else { continue };
            if c.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1)).is_err() { self.saturated.fetch_add(1, Ordering::Relaxed); }
        }
    }
    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn get(&self, id: u32) -> u32 { self.c().get(id as usize).map_or(0, |c| c.load(Ordering::Relaxed)) }
    /// Dense copy (u32 per id).
    pub fn load(&self) -> Vec<u32> { self.c().iter().map(|c| c.load(Ordering::Relaxed)).collect() }
    /// Total of the window.
    pub fn total(&self) -> u64 { self.c().iter().map(|c| c.load(Ordering::Relaxed) as u64).sum() }
    /// Adds lost to saturation since the engine opened.
    pub fn saturated(&self) -> u64 { self.saturated.load(Ordering::Relaxed) }
    pub fn clear(&self) { for c in self.c() { c.store(0, Ordering::Relaxed); } }
    /// Non-zero counts as (id, count), ascending ids, zeroing each (an add racing the take lands in the next window).
    pub fn take(&self) -> Vec<(u32, u32)> {
        let mut v = Vec::new();
        for (i, c) in self.c().iter().enumerate() {
            if c.load(Ordering::Relaxed) == 0 { continue; }
            let n = c.swap(0, Ordering::Relaxed);
            if n > 0 { v.push((i as u32, n)); }
        }
        v
    }
}
pub type SharedCounts = Arc<TokCounts>;

struct Mem(#[allow(dead_code)] Buffer);
unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}

/// Thread-safe handle on the recorder (holds a reference to the arena MTLBuffer, so it outlives nothing it reads).
pub struct ActivityTap {
    _mem: Mem,
    base: usize,
    io: ActIo,
    n: usize,
    tmpl: ActivityStats,
    down: Vec<Q4b>,
    down_norm: OnceLock<Vec<f32>>,
    /// MTP layer down projection (None without MTP) and its column norms
    mtp_down: Option<Q4b>,
    mtp_down_norm: OnceLock<Vec<f32>>,
    /// MTP layer region length (f32 values before its counters)
    n_mtp: usize,
    /// model vocab (lm_head / embedding rows)
    pub vocab: usize,
    /// per id counts of the window: emitted output tokens (`log_out`) and embedding rows read from SSD (EmbedReader)
    pub out_counts: SharedCounts,
    pub embed_counts: SharedCounts,
    scripts: OnceLock<Vec<u8>>,
    /// batch region (same layout as the recorder, `QwenOptions.activity_batch`): byte distance from the recorder
    batch: Option<usize>,
    acc_bytes: usize,
    /// answer library (watcher `library = true`): host copy of the FFN sums at the prompt / answer boundary
    /// (`mark_prompt`); None = off (no allocation, `mark_prompt` returns at once)
    mark: OnceLock<Mutex<PromptMark>>,
    /// mark cost on the worker thread: count, total ns, max ns
    pub mark_n: AtomicU64,
    pub mark_ns: AtomicU64,
    pub mark_max_ns: AtomicU64,
}

/// The FFN sums [layers·inter] (raw Σ|a|) at the first generated token of a job, and the main rows counted by then.
#[derive(Default)]
pub struct PromptMark { pub v: Vec<f32>, pub rows: u32, pub valid: bool }

impl ActivityTap {
    /// `acc` must be zeroed by the caller; `down` = the FFN down projection of each layer, `mtp_down` = the MTP layer's;
    /// `embed_counts` = what the EmbedReader bumps.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(buf: &Buffer, cfg: &QwenConfig, io: ActIo, down: Vec<Q4b>, mtp_down: Option<Q4b>, weight_bytes: u64,
                      resident_bytes: u64, embed_counts: SharedCounts, batch: Option<usize>) -> ActivityTap {
        let base = streamdense_metal::contents(buf) as usize;
        let tmpl = ActivityStats { enabled: true, weight_bytes, resident_bytes, ..ActivityStats::dims(cfg) };
        ActivityTap { _mem: Mem(buf.clone()), base, io, n: n_f32(cfg), tmpl, down, down_norm: OnceLock::new(), mtp_down,
                      mtp_down_norm: OnceLock::new(), n_mtp: n_mtp(cfg), vocab: cfg.vocab,
                      out_counts: Arc::new(TokCounts::in_arena(buf, io.tok, cfg.vocab)), embed_counts, scripts: OnceLock::new(),
                      batch: batch.map(|b| b - io.neuron), acc_bytes: sizes(cfg).0, mark: OnceLock::new(),
                      mark_n: AtomicU64::new(0), mark_ns: AtomicU64::new(0), mark_max_ns: AtomicU64::new(0) }
    }
    /// Answer library: allocate the prompt-mark copy (layers·inter f32 host heap; 4.5 MB on the 27B). Idempotent.
    pub fn enable_prompt_mark(&self) {
        let _ = self.mark.get_or_init(|| Mutex::new(PromptMark { v: vec![0.0; self.tmpl.layers * self.tmpl.inter], rows: 0, valid: false }));
    }
    pub fn prompt_mark_enabled(&self) -> bool { self.mark.get().is_some() }
    /// Answer library, worker thread at the first generated token of a job (the prompt's passes are complete and no
    /// decode pass is encoded yet): copy the FFN sums (one memcpy, no GPU work). No-op unless enabled.
    pub fn mark_prompt(&self) {
        let Some(m) = self.mark.get() else { return };
        let t0 = std::time::Instant::now();
        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
        let n = g.v.len();
        self.with_live(|x| g.v.copy_from_slice(&x[..n]));
        g.rows = self.counters().0;
        g.valid = true;
        drop(g);
        let ns = t0.elapsed().as_nanos() as u64;
        self.mark_n.fetch_add(1, Ordering::Relaxed);
        self.mark_ns.fetch_add(ns, Ordering::Relaxed);
        self.mark_max_ns.fetch_max(ns, Ordering::Relaxed);
    }
    /// Answer library (sd-heat, gate closed): swap this job's mark into `out` (same length; O(1), no copy) and return
    /// the rows counted at the mark. None = no mark since the last take (no token generated / not enabled).
    pub fn take_prompt_mark(&self, out: &mut Vec<f32>) -> Option<u32> {
        let m = self.mark.get()?;
        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
        if !g.valid || out.len() != g.v.len() { g.valid = false; return None; }
        std::mem::swap(&mut g.v, out);
        g.valid = false;
        Some(g.rows)
    }
    /// Emitted output tokens (caller: the decode loop's token callback): per id `vocab` counts.
    pub fn log_out(&self, ids: &[u32]) { self.out_counts.add(ids); }
    /// The window's token counts, sparse (out, embed), zeroing both counters (the per-job path; `reset` zeroes them too).
    pub fn take_tokens(&self) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) { (self.out_counts.take(), self.embed_counts.take()) }
    /// Script class per token id (`script_classes`, len = vocab); first call wins. False when refused.
    pub fn set_scripts(&self, classes: Vec<u8>) -> bool { classes.len() == self.vocab && self.scripts.set(classes).is_ok() }
    pub fn scripts(&self) -> Option<&Vec<u8>> { self.scripts.get() }
    /// The MTP draft layer's live sums: [inter FFN | heads], read in place (all zero when MTP never ran).
    pub fn with_mtp_live<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        f(unsafe { std::slice::from_raw_parts((self.base + self.io.mtp) as *const f32, self.n_mtp - 1) })
    }
    /// MTP layer counters (rows, passes), absolute.
    pub fn mtp_counters(&self) -> (u32, u32) {
        let c = |i: usize| unsafe { std::ptr::read_volatile((self.base + self.io.mtp + self.n_mtp * 4 + 4 * i) as *const u32) };
        (c(0), c(1))
    }
    /// ‖W_down[:, j]‖ of one layer into `out` (len inter), computed now (one thread; the watcher's disk store).
    pub fn down_norm_layer(&self, l: usize, out: &mut [f32]) { col_norms(self.base, &self.down[l], out); }
    /// batch (2026-11-01): a second region of the recorder's layout in model memory where consecutive jobs of one
    /// label are summed before one disk fold. None unless opened with `QwenOptions.activity_batch`.
    pub fn has_batch(&self) -> bool { self.batch.is_some() }
    /// Add the recorder (one job) into the batch: f32 sums, MTP sums, u32 token counters (the GPU counters are not
    /// summed: the caller counts rows). Recorder idle only (gate closed). No allocation.
    pub fn add_to_batch(&self) {
        let Some(d) = self.batch else { return };
        unsafe {
            let add_f = |off: usize, n: usize| {
                let a = std::slice::from_raw_parts((self.base + off) as *const f32, n);
                let b = std::slice::from_raw_parts_mut((self.base + off + d) as *mut f32, n);
                for (y, x) in b.iter_mut().zip(a) { *y += *x; }
            };
            add_f(self.io.neuron, self.n);
            add_f(self.io.mtp, self.n_mtp);
            let a = std::slice::from_raw_parts((self.base + self.io.tok) as *const u32, 2 * self.vocab);
            let b = std::slice::from_raw_parts_mut((self.base + self.io.tok + d) as *mut u32, 2 * self.vocab);
            for (y, x) in b.iter_mut().zip(a) { *y = y.saturating_add(*x); }
        }
    }
    /// Zero the batch region.
    pub fn reset_batch(&self) {
        if let Some(d) = self.batch { unsafe { std::ptr::write_bytes((self.base + self.io.neuron + d) as *mut u8, 0, self.acc_bytes); } }
    }
    /// The batch's [neuron | dn_head | attn_head | resid] sums (raw |a|, like `with_live`), in place.
    pub fn with_batch<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let d = self.batch.unwrap_or(0);
        f(unsafe { std::slice::from_raw_parts((self.base + self.io.neuron + d) as *const f32, if self.batch.is_some() { self.n } else { 0 }) })
    }
    /// The batch's MTP layer sums [inter | heads].
    pub fn with_mtp_batch<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let d = self.batch.unwrap_or(0);
        f(unsafe { std::slice::from_raw_parts((self.base + self.io.mtp + d) as *const f32, if self.batch.is_some() { self.n_mtp - 1 } else { 0 }) })
    }
    /// The batch's token counts: (emitted output tokens, embedding rows read), u32 per id.
    pub fn with_batch_counts<R>(&self, f: impl FnOnce(&[u32], &[u32]) -> R) -> R {
        let (d, v) = (self.batch.unwrap_or(0), if self.batch.is_some() { self.vocab } else { 0 });
        let p = (self.base + self.io.tok + d) as *const u32;
        f(unsafe { std::slice::from_raw_parts(p, v) }, unsafe { std::slice::from_raw_parts(p.add(v), v) })
    }
    /// Layers with a down projection (= cfg.layers).
    pub fn down_layers(&self) -> usize { self.down.len() }
    /// Down norms if already computed (never computes them).
    pub fn down_norms_ready(&self) -> Option<&Vec<f32>> { self.down_norm.get() }
    /// Everything since the last reset (the live sums: reset zeroes them), without any down norms; `tokens`: the dense
    /// token counts + scripts too.
    fn delta(&self, tokens: bool) -> ActivityStats {
        let mut v = Vec::with_capacity(self.n);
        self.with_live(|x| v.extend_from_slice(x));
        let (r, p, rr) = self.counters();
        let s = &self.tmpl;
        let (a, bb, c) = (s.layers * s.inter, s.n_deltanet() * s.dn_vh, s.n_attention() * s.heads);
        v.truncate(a + bb + c + s.layers);
        let resid = v.split_off(a + bb + c);
        let attn_head = v.split_off(a + bb);
        let dn_head = v.split_off(a);
        let mut mtp_neuron = self.with_mtp_live(|x| x.to_vec());
        let (mr, mp) = self.mtp_counters();
        let mtp_head = mtp_neuron.split_off(s.inter);
        let (vocab, embed_rows, scripts) = if tokens {
            (self.out_counts.load(), self.embed_counts.load(), self.scripts.get().cloned().unwrap_or_default())
        } else { (Vec::new(), Vec::new(), Vec::new()) };
        ActivityStats { rows: r as u64, passes: p as u64, resid_rows: rr as u64, neuron: v, dn_head, attn_head, resid,
                        vocab, embed_rows, mtp_rows: mr as u64, mtp_passes: mp as u64, mtp_neuron, mtp_head, scripts, ..s.clone() }
    }
    /// Stats since the last reset (computes the down column norms on first use: a CPU pass over the down weights,
    /// ~1/3 of the FFN bytes; ~1 s on the 27B with 8 threads), with the dense token counts and scripts.
    pub fn snapshot(&self) -> ActivityStats {
        ActivityStats { down_norm: self.down_norms().clone(), mtp_down_norm: self.mtp_down_norms().clone(), ..self.delta(true) }
    }
    /// Same, without the down norms (the caller borrows `down_norms()` instead of a 4.5 MB clone per job) and without
    /// `vocab` / `embed_rows` / `scripts` (empty: the per-job path takes the token counts sparse, `take_tokens`).
    pub fn snapshot_lite(&self) -> ActivityStats { self.delta(false) }
    /// New window: the sums, the GPU counters and the token counters are zeroed in place (no host copy). Engine idle
    /// only (a pass in flight would lose its adds).
    pub fn reset(&self) {
        unsafe {
            std::ptr::write_bytes((self.base + self.io.neuron) as *mut u8, 0, self.n * 4);
            std::ptr::write_bytes((self.base + self.io.cnt) as *mut u8, 0, 16);
            std::ptr::write_bytes((self.base + self.io.mtp) as *mut u8, 0, self.n_mtp * 4 + 16);
        }
        self.out_counts.clear();
        self.embed_counts.clear();
    }

    /// low-level watcher: the live accumulators since the last reset: [neuron layers·inter | dn_head | attn_head |
    /// resid], read in place (no copy, no allocation). Reads race the GPU like `snapshot` (a pass in flight may be
    /// partly counted).
    pub fn with_live<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        f(unsafe { std::slice::from_raw_parts((self.base + self.io.neuron) as *const f32, self.n) })
    }
    /// Live GPU counters (rows, passes, resid_rows), absolute, u32 wrapping.
    pub fn counters(&self) -> (u32, u32, u32) {
        let c = |i: usize| unsafe { std::ptr::read_volatile((self.base + self.io.cnt + 4 * i) as *const u32) };
        (c(0), c(1), c(2))
    }
    /// Dims template (no data).
    pub fn template(&self) -> &ActivityStats { &self.tmpl }

    /// ‖W_down[:, j]‖ per layer and neuron (dequantized q·s + b in f32, as the GEMV kernels), computed once.
    pub fn down_norms(&self) -> &Vec<f32> {
        self.down_norm.get_or_init(|| {
            let inter = self.tmpl.inter;
            let mut out = vec![0f32; self.down.len() * inter];
            let nt = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8);
            let per = self.down.len().div_ceil(nt).max(1);
            std::thread::scope(|sc| {
                for (ci, chunk) in out.chunks_mut(per * inter).enumerate() {
                    let base = self.base;
                    let down = &self.down;
                    sc.spawn(move || for (li, o) in chunk.chunks_mut(inter).enumerate() {
                        col_norms(base, &down[ci * per + li], o);
                    });
                }
            });
            out
        })
    }
    /// Same for the MTP layer (empty without MTP), computed once.
    pub fn mtp_down_norms(&self) -> &Vec<f32> {
        self.mtp_down_norm.get_or_init(|| match &self.mtp_down {
            Some(m) => { let mut o = vec![0f32; self.tmpl.inter]; col_norms(self.base, m, &mut o); o }
            None => Vec::new(),
        })
    }
}

/// Script classes of `script_class` / `ActivityStats::vocab_by_script`.
pub const SCRIPT_NAMES: [&str; 10] = ["ascii", "latin", "cyrillic", "han", "kana", "hangul", "thai", "arabic", "special", "other"];
const SC_LATIN: u8 = 1;
const SC_SPECIAL: u8 = 8;
const SC_OTHER: u8 = 9;
/// Code point ranges per class (Han / kana / Hangul = the keep-list tool's cjk / ja / ko sets, streamdense-cli
/// vocab.rs; duplicated here so the model crate needs no tokenizer).
const SCRIPT_RANGES: &[(u32, u32, u8)] = &[
    (0xC0, 0x2AF, 1), (0x1E00, 0x1EFF, 1), (0x2C60, 0x2C7F, 1), (0xA720, 0xA7FF, 1), (0xAB30, 0xAB6F, 1), (0xFB00, 0xFB06, 1),
    (0x400, 0x52F, 2), (0x1C80, 0x1C8F, 2), (0x2DE0, 0x2DFF, 2), (0xA640, 0xA69F, 2),
    (0x2E80, 0x2FFF, 3), (0x3000, 0x303F, 3), (0x3100, 0x312F, 3), (0x3190, 0x319F, 3), (0x31A0, 0x31BF, 3), (0x31C0, 0x31EF, 3),
    (0x3200, 0x33FF, 3), (0x3400, 0x4DBF, 3), (0x4DC0, 0x4DFF, 3), (0x4E00, 0x9FFF, 3), (0xF900, 0xFAFF, 3), (0xFE10, 0xFE1F, 3),
    (0xFE30, 0xFE6F, 3), (0xFF00, 0xFF64, 3), (0xFFE0, 0xFFEF, 3), (0x20000, 0x3FFFF, 3),
    (0x3040, 0x309F, 4), (0x30A0, 0x30FF, 4), (0x31F0, 0x31FF, 4), (0xFF65, 0xFF9F, 4), (0x1AFF0, 0x1B16F, 4),
    (0x1100, 0x11FF, 5), (0x3130, 0x318F, 5), (0xA960, 0xA97F, 5), (0xAC00, 0xD7AF, 5), (0xD7B0, 0xD7FF, 5), (0xFFA0, 0xFFDF, 5),
    (0xE00, 0xE7F, 6),
    (0x600, 0x6FF, 7), (0x750, 0x77F, 7), (0x870, 0x8FF, 7), (0xFB50, 0xFDFF, 7), (0xFE70, 0xFEFF, 7),
];
fn cp_class(c: u32) -> Option<u8> { SCRIPT_RANGES.iter().find(|&&(a, b, _)| (a..=b).contains(&c)).map(|r| r.2) }
/// Neutral code points: decide nothing (combining marks count as Latin only when nothing else decides).
fn cp_neutral(c: u32) -> bool { c < 0x80 || (0x300..0x370).contains(&c) || (0x2000..0x2070).contains(&c) || (0xFE00..0xFE10).contains(&c) }
/// Class whose ranges (merged when adjacent) cover all of [lo, hi].
fn range_class(lo: u32, hi: u32) -> Option<u8> {
    let cls = cp_class(lo)?;
    let mut iv: Vec<(u32, u32)> = SCRIPT_RANGES.iter().filter(|r| r.2 == cls).map(|r| (r.0, r.1)).collect();
    iv.sort();
    let mut c = lo;
    for (a, b) in iv {
        if a > c { break; }
        if b >= c { if b >= hi { return Some(cls); } c = b + 1; }
    }
    None
}

/// Script class of one token (bytes as the byte-level BPE piece decodes; `special` = added / control token):
/// special; ascii = all-ASCII or single-byte (digits, code, byte fallback); else the first complete code point of a
/// known script (Latin incl. Vietnamese 1E00-1EFF, Cyrillic, Han, kana, Hangul, Thai, Arabic), or a truncated last
/// character whose possible code points all lie in one script (e.g. a lone Han lead byte); combining marks alone =
/// latin; anything else (emoji, symbols, other scripts, pieces starting mid-character) = other.
pub fn script_class(b: &[u8], special: bool) -> u8 {
    if special { return SC_SPECIAL; }
    if b.is_empty() { return SC_OTHER; }
    if b.len() == 1 || b.is_ascii() { return 0; }
    let cont = |x: u8| (0x80..=0xBF).contains(&x);
    let mut i = 0;
    while i < b.len() && cont(b[i]) { i += 1; }
    let mut comb = false;
    while i < b.len() {
        let l = b[i];
        let n = match l { 0x00..=0x7F => 1, 0xC2..=0xDF => 2, 0xE0..=0xEF => 3, 0xF0..=0xF4 => 4, _ => { i += 1; continue; } };
        let have = (1..n).take_while(|&k| i + k < b.len() && cont(b[i + k])).count() + 1;
        let lead = match n { 1 => l as u32, 2 => (l & 0x1F) as u32, 3 => (l & 0x0F) as u32, _ => (l & 0x07) as u32 };
        let (mut lo, mut hi) = (lead, lead);
        for k in 1..n {
            let (bl, bh) = if k < have { ((b[i + k] & 0x3F) as u32, (b[i + k] & 0x3F) as u32) } else { (0, 0x3F) };
            lo = (lo << 6) | bl;
            hi = (hi << 6) | bh;
        }
        if have == n {
            if (0x300..0x370).contains(&lo) { comb = true; }
            if !cp_neutral(lo) { if let Some(c) = cp_class(lo) { return c; } }
        } else if i + have == b.len() {
            let lo = lo.max([0, 0, 0x80, 0x800, 0x10000][n]);
            let hi = hi.min(0x10FFFF);
            if lo <= hi { if let Some(c) = range_class(lo, hi) { return c; } }
        }
        i += have;
    }
    if comb { SC_LATIN } else { SC_OTHER }
}

/// Script class per token id 0..vocab (`piece(id)` = (bytes, special); ids past the tokenizer = empty bytes = other).
/// Caller with a tokenizer: `script_classes(cfg.vocab, |id| (tk.id_to_bytes(id), tk.is_special(id)))`.
pub fn script_classes<'a>(vocab: usize, piece: impl Fn(u32) -> (&'a [u8], bool)) -> Vec<u8> {
    (0..vocab as u32).map(|id| { let (b, s) = piece(id); script_class(b, s) }).collect()
}

/// Column norms of one q4 matrix in the arena at `base` (w u32 [rows][cols / 8], s / b bf16 [rows][cols / 64]).
pub(crate) fn col_norms(base: usize, m: &Q4b, out: &mut [f32]) {
    let (rows, cols) = (m.rows, m.cols);
    let (wc, gc) = (cols / 8, cols / Q4_GROUP);
    let w = unsafe { std::slice::from_raw_parts((base + m.w.off) as *const u32, rows * wc) };
    let s = unsafe { std::slice::from_raw_parts((base + m.s.off) as *const u16, rows * gc) };
    let b = unsafe { std::slice::from_raw_parts((base + m.b.off) as *const u16, rows * gc) };
    // accumulate in `out` itself (no host buffer; same order and arithmetic as before: bit-identical norms)
    let acc = &mut out[..cols];
    acc.fill(0.0);
    for r in 0..rows {
        for g in 0..gc {
            let (sc, bi) = (bf16_to_f32(s[r * gc + g]), bf16_to_f32(b[r * gc + g]));
            for wi in 0..Q4_GROUP / 8 {
                let word = w[r * wc + g * (Q4_GROUP / 8) + wi];
                let j0 = g * Q4_GROUP + wi * 8;
                for q in 0..8 {
                    let x = ((word >> (4 * q)) & 15) as f32 * sc + bi;
                    acc[j0 + q] += x * x;
                }
            }
        }
    }
    for a in acc.iter_mut() { *a = a.sqrt(); }
}


/// One activity request (serve op `{"op":"activity",..}` and gateway GET /v1/streamdense/activity), JSON in, JSON out:
/// in  {"path": FILE (write the full snapshot, safetensors), "plan": true | {"keep": 0.995, "granularity": 64,
///      "layers": false}, "reset": bool (after reading)}
/// out {"activity": summary, "written": FILE?, "plan": UnloadPlan JSON?, "reset": bool} or {"activity": {"enabled": false}}.
pub fn handle_request(tap: Option<&ActivityTap>, req: &Value) -> Value {
    let Some(t) = tap else { return json!({"activity": {"enabled": false}}) };
    let s = t.snapshot();
    let mut out = json!({"activity": s.summary()});
    if let Some(p) = req["path"].as_str().filter(|p| !p.is_empty()) {
        match s.write(p) { Ok(()) => out["written"] = json!(p), Err(e) => out["error"] = json!(e) }
    }
    let pl = &req["plan"];
    if pl.as_bool() == Some(true) || pl.is_object() {
        let keep = pl["keep"].as_f64().unwrap_or(0.995) as f32;
        let g = pl["granularity"].as_u64().unwrap_or(crate::unload::MASK_BLOCK as u64).max(1) as usize;
        out["plan"] = crate::unload::unload_plan(&s, keep, g).to_json(pl["layers"].as_bool().unwrap_or(false));
    }
    if req["reset"].as_bool() == Some(true) { t.reset(); out["reset"] = json!(true); }
    out
}
