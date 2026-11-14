//! MTP draft head (plan 7.3 enc B, 7.4 prefill warm-up; U19).
//! - One MTP "row" pairs the target hidden h_p (after the final norm) with emb(t_{p+1}) and sits at MTP KV slot /
//!   RoPE position p = ctrl.mtp_len + row. `mtp_prep` builds cat = [rms(emb) | rms(hf)] and writes the second control
//!   block `mctrl` (pos = mtp_len, mtp_rows); `mctrl` is the ctrl of the MTP attention layer and of `select_row`.
//! - Draft: fc (bf16 GEMV, pairs of rows) → the MTP decoder layer (own KV, `encode_layer` with `mctrl`) →
//!   select_row(mtp_rows - 1) → mtp.norm → target lm_head (K = 1) → `mtp_draft` (ctrl.draft, ctrl.mtp_len += rows).
//!   Warm-up chunks end with `mtp_advance` instead (no head).
//! - Invariant kept by every caller: after a step / prefill, ctrl.mtp_len == ctrl.pos.
//! - Host positions only choose the attention path near `sdpa_vec_max_n()` (8192); the 1-pass / 2-pass choice at
//!   1024 is made per row on the GPU from ctrl.pos. When the host only knows bounds [lo, hi] (encode-ahead),
//!   `path_pos` returns a position that makes `encode_attn` dispatch every kernel any position in range needs.
use streamdense_core::arena::Region;
use streamdense_metal::{cb_signal, cb_wait, commit, wait, Enc, SdCtrl};

use crate::embed::row_bytes;
use crate::qwen_config::QwenConfig;
use crate::qwen_engine::{QwenEngine, ATTN_CHUNK};
use crate::qwen_layers::{encode_layer, gemv, gemv_d, mma_twin_nt, mma_use_twin, sdpa_vec_max_n, GemvImpl, KvRef, PassIo, MMA_SK};
use crate::weights::{MtpW, Q4b};

/// Shader files the MTP path adds to `QWEN_SHADERS` (U17 / U18; head_pruned = the pruned-vocab draft head).
pub const MTP_SHADERS: &[&str] = &["gemv_bf16", "spec", "head_pruned", "spec_adapt"];
const KS: [usize; 4] = [1, 2, 4, 8];
const BF: usize = 2;
/// `spec_conf` rows (8 verify rows + the MTP draft row) and bytes per row ([idx0..2, v0..2, lse, 0] f32).
pub const CONF_ROWS: usize = 9;
pub const CONF_BYTES: usize = 32;

/// Pipelines used by `mtp.rs` / `spec.rs` (beyond `qwen_pipelines()`).
pub fn pipelines() -> Vec<&'static str> {
    vec!["bf16_gemv_k1", "bf16_gemv_k2", "mtp_prep", "mtp_draft", "mtp_advance", "spec_decide", "select_row",
         "embed_q4", "copy_bytes", "rmsnorm_b", "q4b_gemv_k1_m3", "spec_conf"]
}

/// MTP scratch (arena offsets). `emb` [tmax][H] bf16 (spec mode: row 0 = emb(d), row 1 = emb(new t1)); `cat`
/// [tmax][2H]; `h` [tmax][H] (fc out = MTP layer residual); `row`, `hn` [H]; `stage` 2 slots of tmax embedding rows
/// (prefill warm-up: emb(ids[i+1..])); `eos` u32 [n_eos].
#[derive(Clone, Copy, Debug, Default)]
pub struct MtpScratch {
    pub mctrl: usize,
    pub emb: usize,
    pub cat: usize,
    pub h: usize,
    pub row: usize,
    pub hn: usize,
    pub stage: usize,
    pub slot_bytes: usize,
    pub eos: Region,
    pub n_eos: usize,
    pub tmax: usize,
    /// pruned draft vocab (`QwenOptions.draft_vocab`): arena offset of the kept token ids u32 [n_keep] (ascending);
    /// n_keep = 0: the draft head runs the full lm_head
    pub keep: usize,
    pub n_keep: usize,
    /// `spec_conf` output (spec_adapt.metal): [CONF_ROWS][8] f32: rows 0..8 = verify rows, row 8 = the last MTP draft
    pub conf: usize,
}

/// MTP KV of the engine (plan interface name).
#[derive(Clone, Copy, Debug)]
pub struct MtpState {
    pub kv: KvRef,
}

/// Scratch rows the MTP path needs, for the budget / allocation (names get the engine's `s:` prefix).
pub fn scratch_sizes(cfg: &QwenConfig, tmax: usize) -> Vec<(&'static str, usize)> {
    let h = cfg.hidden;
    vec![
        ("mtp.ctrl", std::mem::size_of::<SdCtrl>()), ("mtp.emb", tmax * h * BF), ("mtp.cat", tmax * 2 * h * BF),
        ("mtp.h", tmax * h * BF), ("mtp.row", h * BF), ("mtp.hn", h * BF), ("mtp.stage", 2 * tmax * row_bytes(h)),
        ("mtp.eos", cfg.eos.len().max(1) * 4), ("mtp.conf", CONF_ROWS * CONF_BYTES),
    ]
}

impl MtpScratch {
    /// From the engine's scratch regions (`get(name)` = the region allocated for a `scratch_sizes` name).
    pub fn from_regions(cfg: &QwenConfig, tmax: usize, get: &dyn Fn(&str) -> Region) -> MtpScratch {
        MtpScratch {
            mctrl: get("mtp.ctrl").off, emb: get("mtp.emb").off, cat: get("mtp.cat").off, h: get("mtp.h").off,
            row: get("mtp.row").off, hn: get("mtp.hn").off, stage: get("mtp.stage").off, slot_bytes: tmax * row_bytes(cfg.hidden),
            eos: get("mtp.eos"), n_eos: cfg.eos.len(), tmax, keep: 0, n_keep: 0, conf: get("mtp.conf").off,
        }
    }
}

/// Host position for `PassIo.pos` when the real ctrl position is only known to lie in [lo, hi] (k rows): the
/// returned value makes `encode_attn` dispatch the MLX mirror and/or the split-K path exactly as every position in
/// the range needs (a straddling range gets both; each row's kernel then picks by its own N on the GPU).
pub fn path_pos(lo: usize, hi: usize, k: usize, q8: bool) -> Result<usize, String> {
    if lo == hi || q8 { return Ok(lo); }
    let lim = sdpa_vec_max_n().min(8192);
    if hi + k <= lim || lo + 1 > lim { return Ok(lo); }
    if k >= 2 { return Ok(lim - 1); }
    Err(format!("mtp: k = 1 pass with position range [{}, {}] straddles the sdpa limit {}", lo, hi, lim))
}

/// Where the MTP rows come from.
#[derive(Clone, Copy, Debug)]
pub enum MtpRows {
    /// Step mode (enc B): 2 rows dispatched, ctrl.mtp_rows (1 or 2) of them real; `ms.emb` rows 0/1 = emb(d) /
    /// emb(new t1); hf = the verify pass's 2 final-norm rows (`io.hf`).
    Spec,
    /// Prefill warm-up / first row: `t` rows (1, 2, 4 or 8) of emb [t][H] at `emb` and hf [t][H] at `hf`.
    Direct { t: usize, emb: usize, hf: usize },
}

/// Encode one MTP pass (plan 7.3 enc B / 7.4): mtp_prep → fc → MTP layer (K rows, ctrl = mctrl) → (draft: select_row
/// → norm → lm_head → mtp_draft) | (warm-up: mtp_advance). `io` = the engine's main pass scratch (x, p, cv, ao, f,
/// po, pml, logits are reused; h is replaced by `ms.h`). `pos` = host value of mctrl.pos (from `path_pos`), `nch`
/// = attention chunks covering mctrl.pos + K.
#[allow(clippy::too_many_arguments)]
pub fn encode_mtp(e: &Enc, cfg: &QwenConfig, w: &MtpW, head: &Q4b, kv: &KvRef, io: &PassIo, ms: &MtpScratch,
                  rows: MtpRows, pos: usize, nch: usize, draft: bool) -> Result<(), String> {
    let h = cfg.hidden;
    let (k, emb, hf, t_arg) = match rows {
        MtpRows::Spec => (2, ms.emb, io.hf, 0usize),
        MtpRows::Direct { t, emb, hf } => (t, emb, hf, t),
    };
    if !KS.contains(&k) || k > ms.tmax { return Err(format!("mtp: {} rows not in {:?} or > {}", k, KS, ms.tmax)); }
    e.pso("mtp_prep").b(0, emb).b(1, hf).b(2, w.pre_e.off).b(3, w.pre_h.off).b(4, ms.cat).b(5, io.ctrl)
        .f(6, cfg.eps).u(7, h as u32).u(8, t_arg as u32).b(9, ms.mctrl)
        .groups((k, 2), 1024);
    // fc: bf16 [H][2H], rows in pairs (K-invariant kernel)
    let c = 2 * h;
    let mut t = 0;
    while t < k {
        let n = (k - t).min(2);
        e.pso(if n == 2 { "bf16_gemv_k2" } else { "bf16_gemv_k1" })
            .b(0, w.fc.off).b(1, ms.cat + t * c * BF).b(2, ms.h + t * h * BF).u(3, h as u32).u(4, c as u32)
            .groups((h.div_ceil(32), 1), 256);
        t += n;
    }
    let mio = PassIo { k, h: ms.h, ctrl: ms.mctrl, pos, nch_max: nch.max(1), ..*io };
    encode_layer(e, cfg, &w.layer, None, Some(kv), &mio, false, None, cfg.layers)?;
    if draft {
        e.pso("select_row").b(0, ms.h).b(1, ms.row).u(2, h as u32).b(3, ms.mctrl).threads((h, 1), 256);
        e.pso("rmsnorm_b").b(0, ms.row).b(1, w.norm.off).b(2, ms.hn).u(3, h as u32).f(4, cfg.eps).u(5, h as u32)
            .groups((1, 1), 1024);
        if ms.n_keep > 0 {
            // pruned draft vocab: kept rows only (logits[0..n_keep]), argmax mapped back to the token id
            head_pruned(e, io.gemv, head, ms.keep, ms.n_keep, ms.hn, io.logits);
            e.pso("mtp_draft_idx").b(0, io.logits).b(1, io.ctrl).b(2, ms.mctrl).u(3, ms.n_keep as u32).b(4, ms.keep)
                .groups((1, 1), 1024);
        } else {
            gemv(e, io.gemv, head, 1, 3, ms.hn, io.logits); // lm_head K = 1, the engine's matmul family
            e.pso("mtp_draft").b(0, io.logits).b(1, io.ctrl).b(2, ms.mctrl).u(3, head.rows as u32).groups((1, 1), 1024);
        }
    } else {
        e.pso("mtp_advance").b(0, io.ctrl).b(1, ms.mctrl).threads((1, 1), 1);
    }
    Ok(())
}

/// Pruned-vocab head kernels of family `imp` (head_pruned.metal; built with `QwenOptions.draft_vocab`).
pub fn pruned_pipelines(imp: GemvImpl) -> Vec<&'static str> { vec![pruned_kernel(imp), "mtp_draft_idx"] }
fn pruned_kernel(imp: GemvImpl) -> &'static str {
    match imp {
        GemvImpl::Mma | GemvImpl::MmaD => match MMA_SK { 1 => "qm_gemvt_idx_k1_s1_m3", 2 => "qm_gemvt_idx_k1_s2_m3", 4 => "qm_gemvt_idx_k1_s4_m3", _ => "qm_gemvt_idx_k1_s8_m3" },
        GemvImpl::Old => match gemv_d(1) { 4 => "q4b_gemv_idx_k1_r4_m3", 8 => "q4b_gemv_idx_k1_r8_m3", _ => "q4b_gemv_idx_k1_r2_m3" },
    }
}

/// out f32 [n] = lm_head rows idx[0..n] (u32 token ids at arena offset `idx`) . x (bf16 [cols]), K = 1, family
/// `imp`: per kept row bit-identical to `gemv(e, imp, head, 1, 3, ..)` on that row (the Mma family at K = 1 is the
/// twin, `mma_use_twin`; Old uses the decode D). Reads n / rows of the head's bytes; no weight copy.
pub fn head_pruned(e: &Enc, imp: GemvImpl, head: &Q4b, idx: usize, n: usize, x: usize, out: usize) {
    let enc = |name: &str, groups: usize, nt: usize| {
        e.pso(name).b(0, head.w.off).b(1, head.s.off).b(2, head.b.off).b(3, x).b(4, out).u(5, n as u32)
            .u(6, head.cols as u32).b(7, idx).groups((groups, 1), nt);
    };
    match imp {
        GemvImpl::Mma | GemvImpl::MmaD => {
            debug_assert!(mma_use_twin(head.cols, false, 1));
            enc(pruned_kernel(imp), n.div_ceil(8), mma_twin_nt(head.cols, false, 1, MMA_SK, 512))
        }
        GemvImpl::Old => { let d = gemv_d(1); enc(pruned_kernel(imp), n.div_ceil(2 * d), 64) }
    }
}

/// Split n rows into kernel sizes (8 / 4 / 2 / 1, at most `max`): (offset, rows).
pub(crate) fn splits(n: usize, max: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut i = 0;
    while i < n {
        let k = *KS.iter().rev().find(|&&k| k <= max && k <= n - i).unwrap();
        v.push((i, k));
        i += k;
    }
    v
}

impl QwenEngine {
    pub(crate) fn mtp_parts(&self) -> Result<(&MtpW, &KvRef, &MtpScratch), String> {
        match (self.w.mtp.as_ref(), self.mtp_kv().as_ref(), self.mtp_s().as_ref()) {
            (Some(w), Some(kv), Some(ms)) => Ok((w, kv, ms)),
            _ => Err("MTP is off: open the engine with QwenOptions.mtp (--mtp)".into()),
        }
    }
    pub(crate) fn nch_for(&self, hi: usize, k: usize) -> usize { (hi + k).div_ceil(ATTN_CHUNK).clamp(1, self.nch_max) }

    /// Prefill with MTP warm-up (plan 7.4), from the CURRENT position (continuation-safe): target chunks exactly as
    /// `prefill` (same chunking, same passes), then per chunk final norm → hf and MTP rows (hf_p, emb(ids[p+1])) for
    /// every p whose next token is known. Returns (t1 = greedy next token, arena offset of the last row's hf). The
    /// last MTP row (hf_last, emb(t1)) is `mtp_first_draft`'s job. If a plain turn left ctrl.mtp_len behind ctrl.pos,
    /// mtp_len is moved up to pos first (those MTP KV slots stay stale: lower acceptance only, output is exact).
    pub fn prefill_mtp(&self, ids: &[u32]) -> Result<(u32, usize), String> {
        let (w, kv, ms) = self.mtp_parts()?;
        if ids.is_empty() { return Err("prefill_mtp: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx {
            return Err(format!("prefill_mtp: {} + {} tokens > max_ctx {}", self.pos(), ids.len(), self.opt.max_ctx));
        }
        if self.dumper.is_some() { return Err("prefill_mtp: dump mode is not supported with MTP".into()); }
        self.rm_prefill_begin(); // read mask: the prompt is always read in full
        {
            let c = self.ctrl();
            if c.mtp_len != c.pos { c.mtp_len = c.pos; }
        }
        let cfg = &self.cfg;
        let hb = cfg.hidden * BF;
        let io = self.io();
        let p0 = self.pos();
        self.pf_begin(); // q8-conversion prototype (no-op unless SD_PF_Q8CONV=1 with --kv q8)
        let ch = self.chunks(ids.len());
        let mut inflight: std::collections::VecDeque<streamdense_metal::CmdBuf> = Default::default();
        let mut hf_last = io.hf;
        let mut res: Result<(), String> = Ok(());
        let order = self.gpu.shared_event(); // CB c+1 waits for CB c on the GPU (untracked arena; see `prefill`)
        for (ci, &(i, k)) in ch.iter().enumerate() {
            let slot = ci % 2;
            if inflight.len() >= 2 { if let Err(e) = wait(&inflight.pop_front().unwrap()) { res = Err(e); break; } }
            let last = ci + 1 == ch.len();
            let m = if i + k < ids.len() { k } else { k - 1 };
            let mstage = unsafe { self.arena.base().add(ms.stage + slot * ms.slot_bytes) };
            let r = (|| -> Result<streamdense_metal::CmdBuf, String> {
                self.reader.read_rows(&ids[i..i + k], self.stage_ptr(slot))?;
                if m > 0 { self.reader.read_rows(&ids[i + 1..i + 1 + m], mstage)?; }
                let cb = self.gpu.command_buffer();
                if ci > 0 { cb_wait(&cb, &order, ci as u64); }
                self.encode_pass(&cb, k, slot, last as usize, false)?;
                let e = Enc::new(&self.gpu, &cb, &self.buf);
                let r = (|| -> Result<(), String> {
                    if !last {
                        e.pso("rmsnorm_b").b(0, io.h).b(1, self.w.final_norm.off).b(2, io.hf).u(3, cfg.hidden as u32)
                            .f(4, cfg.eps).u(5, cfg.hidden as u32).groups((k, 1), 1024);
                    }
                    if m > 0 {
                        e.pso("embed_q4").b(0, ms.stage + slot * ms.slot_bytes).b(1, ms.emb).u(2, m as u32);
                        e.threads((cfg.hidden / 8, m), 64);
                        for (o, s) in splits(m, self.tmax) {
                            let pos = p0 + i + o;
                            encode_mtp(&e, cfg, w, &self.w.lm_head, kv, &io, ms,
                                       MtpRows::Direct { t: s, emb: ms.emb + o * hb, hf: io.hf + o * hb },
                                       pos, self.nch_for(pos, s), false)?;
                        }
                    }
                    Ok(())
                })();
                e.end();
                r?;
                cb_signal(&cb, &order, ci as u64 + 1);
                Ok(cb)
            })();
            match r {
                Ok(cb) => { commit(&cb); inflight.push_back(cb); }
                Err(e) => { res = Err(e); break; }
            }
            if last { hf_last = io.hf + (k - 1) * hb; }
        }
        for cb in &inflight { let w = wait(cb); if res.is_ok() { res = w; } }
        if res.is_err() { self.pf_on.set(false); }
        res?;
        self.pf_end();
        self.rm_prefill_end();
        Ok((self.ctrl_read().tok[0], hf_last))
    }

    /// The MTP row of the last fed position: (hf at `hf_off`, emb(t1)) → draft (ctrl.draft, mtp_len += 1). emb(t1)
    /// is staged in row 0 of staging slot `slot` (the first verify pass reads [t1, d] from that slot). Returns the
    /// GPU draft (the caller may override it, then stages emb(d) in row 1).
    pub fn mtp_first_draft(&self, t1: u32, hf_off: usize, slot: usize) -> Result<u32, String> {
        let (w, kv, ms) = self.mtp_parts()?;
        let cfg = &self.cfg;
        self.reader.read_rows(&[t1], self.stage_ptr(slot))?;
        let pos = self.ctrl_read().mtp_len as usize; // exact here (nothing in flight)
        let cb = self.gpu.command_buffer();
        let e = Enc::new(&self.gpu, &cb, &self.buf);
        let r = (|| -> Result<(), String> {
            e.pso("embed_q4").b(0, self.sl().stage.at(slot * self.slot_bytes)).b(1, ms.emb).u(2, 1);
            e.threads((cfg.hidden / 8, 1), 64);
            encode_mtp(&e, cfg, w, &self.w.lm_head, kv, &self.io(), ms, MtpRows::Direct { t: 1, emb: ms.emb, hf: hf_off },
                       pos, self.nch_for(pos, 1), true)
        })();
        e.end();
        r?;
        commit(&cb);
        wait(&cb)?;
        Ok(self.ctrl_read().draft)
    }

    /// spec-bench: arena offsets for chained drafting: (hn = this step's mtp.norm output, row = its pre-norm row,
    /// save_hn, save_row = emb rows 2 / 3, where `mtp_save_step` keeps a copy of hn / row).
    pub fn mtp_chain_offsets(&self) -> Result<(usize, usize, usize, usize), String> {
        let (_, _, ms) = self.mtp_parts()?;
        if ms.tmax < 4 { return Err(format!("mtp chain: needs 4 MTP emb rows, tmax {}", ms.tmax)); }
        let hb = self.cfg.hidden * BF;
        Ok((ms.hn, ms.row, ms.emb + 2 * hb, ms.emb + 3 * hb))
    }

    /// spec-bench: copy the last MTP step's hn / row to the save slots (emb rows 2 / 3), so several chain variants
    /// can start from the same first step.
    pub fn mtp_save_step(&self) -> Result<(), String> {
        let (hn, row, shn, srow) = self.mtp_chain_offsets()?;
        let hb = self.cfg.hidden * BF;
        let cb = self.gpu.command_buffer();
        let e = Enc::new(&self.gpu, &cb, &self.buf);
        e.pso("copy_bytes").b(0, hn).b(1, shn).u(2, hb as u32).threads((hb, 1), 256);
        e.pso("copy_bytes").b(0, row).b(1, srow).u(2, hb as u32).threads((hb, 1), 256);
        e.end();
        commit(&cb);
        wait(&cb)
    }

    /// spec-bench: one CHAINED MTP draft step (multi-step MTP drafting): row (hidden at `src`, emb(d)) at MTP
    /// position ctrl.mtp_len → MTP layer (writes its KV slot mtp_len) → norm → lm_head → argmax. `src` is the previous
    /// MTP step's output hidden (hn after mtp.norm, the vLLM convention, or the pre-norm row). It is copied to emb
    /// row 1 first (the step overwrites hn / row). ctrl.draft = the new draft, ctrl.mtp_len += 1 (the caller moves
    /// mtp_len back after the chain). Returns (draft, GPU ms).
    pub fn mtp_chain_step(&self, d: u32, src: usize) -> Result<(u32, f64), String> {
        let (_, _, ms) = self.mtp_parts()?;
        self.mtp_chain_offsets()?;
        self.reader.read_rows(&[d], unsafe { self.arena.base().add(ms.stage) })?;
        let pos = self.ctrl_read().mtp_len as usize;
        if pos + 1 > self.opt.max_ctx { return Err(format!("mtp chain: position {} >= max_ctx {}", pos, self.opt.max_ctx)); }
        let cb = self.gpu.command_buffer();
        let e = Enc::new(&self.gpu, &cb, &self.buf);
        let r = self.encode_chain_step(&e, src, ms.stage, pos);
        e.end();
        r?;
        commit(&cb);
        wait(&cb)?;
        Ok((self.ctrl_read().draft, streamdense_metal::gpu_ms(&cb)))
    }

    /// The GPU body of `mtp_chain_step` (also encoded ahead by spec_deep.rs's pipelined loop): hidden at `src` →
    /// emb row 1, the embedding row staged at arena offset `stage` → emb row 0, one MTP draft row at host position
    /// `pos` (= ctrl.mtp_len when it runs). Any draft-head variant (e.g. the pruned vocab) belongs in `encode_mtp`.
    pub(crate) fn encode_chain_step(&self, e: &Enc, src: usize, stage: usize, pos: usize) -> Result<(), String> {
        let (w, kv, ms) = self.mtp_parts()?;
        let cfg = &self.cfg;
        let hb = cfg.hidden * BF;
        e.pso("copy_bytes").b(0, src).b(1, ms.emb + hb).u(2, hb as u32).threads((hb, 1), 256);
        e.pso("embed_q4").b(0, stage).b(1, ms.emb).u(2, 1);
        e.threads((cfg.hidden / 8, 1), 64);
        encode_mtp(e, cfg, w, &self.w.lm_head, kv, &self.io(), ms, MtpRows::Direct { t: 1, emb: ms.emb, hf: ms.emb + hb },
                   pos, self.nch_for(pos, 1), true)
    }
}

/// `QwenEngine::draft_head_probe` result.
#[derive(Clone, Debug, Default)]
pub struct DraftProbe {
    /// full lm_head logits f32 [vocab] (the draft head without a keep-list)
    pub full: Vec<f32>,
    /// pruned head logits f32 [n_keep] (row i = token ids[i])
    pub kept: Vec<f32>,
    pub ids: Vec<u32>,
    /// GPU drafts: `mtp_draft` over `full`, `mtp_draft_idx` over `kept`
    pub draft_full: u32,
    pub draft_kept: u32,
}

impl QwenEngine {
    /// Test probe of the pruned draft head (needs `QwenOptions.draft_vocab`): runs the full and the pruned lm_head
    /// (the engine's family) + both draft argmaxes on the bf16 row `hn` (None: the current mtp.norm output, e.g.
    /// after `mtp_first_draft`). ctrl is restored afterwards; io.logits is clobbered.
    pub fn draft_head_probe(&self, hn: Option<&[u16]>) -> Result<DraftProbe, String> {
        let (_, _, ms) = self.mtp_parts()?;
        if ms.n_keep == 0 { return Err("draft_head_probe: no draft vocab (QwenOptions.draft_vocab)".into()); }
        let (h, v, n) = (self.cfg.hidden, self.cfg.vocab, ms.n_keep);
        let base = self.arena.base();
        if let Some(x) = hn {
            if x.len() != h { return Err(format!("draft_head_probe: hn has {} values, hidden {}", x.len(), h)); }
            unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), base.add(ms.hn) as *mut u16, h) };
        }
        let saved = self.ctrl_read();
        let io = &self.io();
        let head = &self.w.lm_head;
        let run = |f: &dyn Fn(&Enc)| -> Result<(), String> {
            let cb = self.gpu.command_buffer();
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            f(&e);
            e.end();
            commit(&cb);
            wait(&cb)
        };
        let read_f32 = |off: usize, k: usize| -> Vec<f32> { unsafe { std::slice::from_raw_parts(base.add(off) as *const f32, k).to_vec() } };
        run(&|e| {
            gemv(e, io.gemv, head, 1, 3, ms.hn, io.logits);
            e.pso("mtp_draft").b(0, io.logits).b(1, io.ctrl).b(2, ms.mctrl).u(3, head.rows as u32).groups((1, 1), 1024);
        })?;
        let full = read_f32(io.logits, v);
        let draft_full = self.ctrl_read().draft;
        unsafe { std::ptr::write_bytes(base.add(io.logits), 0xa5, n * 4) };
        run(&|e| {
            head_pruned(e, io.gemv, head, ms.keep, n, ms.hn, io.logits);
            e.pso("mtp_draft_idx").b(0, io.logits).b(1, io.ctrl).b(2, ms.mctrl).u(3, n as u32).b(4, ms.keep).groups((1, 1), 1024);
        })?;
        let kept = read_f32(io.logits, n);
        let draft_kept = self.ctrl_read().draft;
        *self.ctrl() = saved;
        let ids = unsafe { std::slice::from_raw_parts(base.add(ms.keep) as *const u32, n).to_vec() };
        Ok(DraftProbe { full, kept, ids, draft_full, draft_kept })
    }
}
