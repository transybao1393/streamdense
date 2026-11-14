//! Qwen3.5/3.8 decode engine (P1: plain greedy 4-bit decode). Mirrors the 7B `engine.rs` structure:
//! budget (every arena row listed, refuse with the shortfall) → one arena MTLBuffer → F_NOCACHE pread of the MLX
//! safetensors (`weights::load`) → state / KV / ctrl / scratch → passes (`qwen_layers`) → prefill in K-row chunks →
//! greedy generate (one command buffer per token, SSD embedding hand-off thread commits the next one) → stats.
//! - Token ids in and out (text/tokenizer lives in streamdense-tokenizer). Greedy only.
//! - Control lives on the GPU (`SdCtrl`): kernels read `ctrl.pos` / `ctrl.role`, `argmax_b` writes `ctrl.tok`, and
//!   `advance1` moves `ctrl.pos`. The CPU writes ctrl only in `reset()`. A host-side copy of the position (`pos`)
//!   only sizes the attention chunk dispatch (`nch_max`); passes are encoded in execution order, so it is exact.
//! - DeltaNet state: 1 copy in plain mode (role 0 only), 2 copies (S[2], conv[2]) with MTP for the rollback
//!   checkpoint. MTP weights / KV are only planned and loaded with `opt.mtp` (default off).
//! - Self-monitor: optional `footprint::Monitor` (abort above `opt.abort_at`, e.g. 18.0 GB). Off by default because
//!   the CLI runs its own monitor.
use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use streamdense_core::arena::{Arena, Region, ALIGN};
use streamdense_core::budget::Budget;
use streamdense_core::footprint::{self, Monitor};
use streamdense_core::{alloc, gb, write_safetensors, DType, SafeIndex};
use streamdense_metal::{cb_signal, cb_wait, commit, contents, gpu_ms, wait, Buffer, CmdBuf, Enc, Gpu, SdCtrl, QWEN_SHADERS};

use crate::dump::Dumper;
use crate::embed::{CommitFn, EmbedReader, Handoff, HandoffKind, HandoffPolicy, LatStats};
use crate::qwen_config::{LayerKind, QwenConfig};
use crate::qwen_layers::{encode_head, encode_layer, encode_layer_prefill, encode_rope_invf, gemm_pipelines, prefill_pipelines,
                         qwen_pipelines, scratch_sizes, DnState, GemvImpl, KvRef, PassIo, EXACT_KS, PREFILL_T_MAX};
use crate::weights::{self, Weights};

/// Keys per attention split-K chunk: a constant of `attn_gated.metal` (plan 4.2), not a model dim.
pub const ATTN_CHUNK: usize = 256;
/// Prefill chunk sizes (exact passes also take 3 and 5 rows: `EXACT_KS`, the deep spec verify).
const KS: [usize; 4] = [1, 2, 4, 8];
/// `topk32` writes 32 entries per row.
const TOPK_MAX: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat { Bf16, Q8 }

/// Layer-by-layer capture (plan 9.3) during `prefill`: `layers` empty = all layers. The prefill then runs stepwise
/// (k = 1, as the MLX ref) and writes one safetensors file to `path` with every tensor as [T, ...] (state tensors
/// `L{l}.dn.S` / `L{l}.dn.conv_state`: after the last token).
#[derive(Clone, Debug)]
pub struct DumpSpec {
    pub layers: Vec<usize>,
    pub path: String,
}

#[derive(Clone, Debug)]
pub struct QwenOptions {
    pub cap: u64,            // hard RAM cap for the budget (bytes)
    pub reserve: u64,        // process reserve inside the cap (Metal runtime, shader lib, CPU heap, binary)
    pub max_ctx: usize,
    pub kv: KvFormat,
    pub mtp: bool,           // P2: MTP weights, MTP KV and the 2nd state copy
    pub spec_depth: usize,   // MTP drafts verified per pass (1 = draft 1 / verify 2; d >= 2: spec_deep.rs, K = d + 1)
    pub gemv: GemvImpl,      // matmul family of every exact pass (decode, verify, MTP head, exact prefill); default Mma
    pub prefill_k: usize,   // exact prefill chunk rows (1, 2, 4 or 8)
    pub prefill_t: usize,    // fast prefill chunk rows (GEMM + prefill mixers, 9..=PREFILL_T_MAX), plan U25
    pub prefill_exact: bool, // true: K <= 8 GEMV chunks only (bit-exact vs MLX stepwise); false: fast prefill
    pub load_threads: usize,
    pub embed_threads: usize,
    pub depth: usize,        // decode command buffers encoded ahead of the last committed one
    pub abort_at: u64,       // self-monitor: abort above this phys_footprint (0 = no engine monitor; the CLI has one)
    pub swap_abort: u64,     // self-monitor: abort above this system swap
    pub dump: Option<DumpSpec>,
    /// pruned MTP draft vocab: keep-list file of `streamdense vocab keep` (needs mtp). The draft head computes only
    /// those lm_head rows (gathered, no weight copy; + 4 B per kept id); verify stays full-vocab, output unchanged.
    pub draft_vocab: Option<std::path::PathBuf>,
    /// adaptive speculative depth (spec_deep.rs `SpecMode::Auto`): 0..=spec_depth drafts per verify pass, chosen per
    /// round from the MTP draft confidence, the acceptance history and the draft table (needs mtp, spec_depth >= 2)
    pub spec_auto: bool,
    /// cost model of the adaptive planner (verify pass ms by K, MTP step ms, fixed ms per round)
    pub spec_cost: crate::spec_deep::SpecCost,
    /// learning draft table directory (draft_table.rs; None = off) and its RAM budget (bytes)
    pub draft_table: Option<std::path::PathBuf>,
    pub draft_table_bytes: usize,
    /// activity recorder (activity.rs): per FFN neuron / head / layer usage sums on every pass, ~4.5 MB arena on the
    /// 27B; outputs unchanged. Default on (CLI `--no-activity`).
    pub activity: bool,
    /// batch region (activity.rs `add_to_batch`): one more recorder-sized region (6.5 MB on the 27B) where the
    /// gateway sums consecutive jobs of one label before a disk fold. Default off (the gateway turns it on).
    pub activity_batch: bool,
    /// the RAM cap enforced live (sd-watchdog + --abort-gb), shown as "the cap" in the budget report; 0 = show `cap`
    pub live_cap: u64,
    /// read mask (readmask.rs): allocate the mask region (mask words + bf16 column norms, ~2.9 MB on the 27B)
    /// and build the masked kernels. Default OFF; with it on the mask itself is still off until `set_read_mask`.
    pub read_mask: bool,
    /// FFN-select test hook (ffn_select.rs): per-neuron FFN selection region + kernel. Default OFF (no region, no dispatch,
    /// outputs unchanged); with it on the hook is in mode All (keeps everything) until `fsel_set`.
    pub ffn_select: bool,
    /// forheader step 3 (ffn_compact.rs): hold ONLY these FFN neurons of every main layer (gate / up rows, down columns
    /// repacked with the original scales). Default None (today's full FFN). Needs prefill_exact, recorder / read mask /
    /// ffn_select off. MTP works (the MTP layer stays full; verify passes use the multi-row compact down). `ffn_compact_load` swaps in another set of the same sizes.
    pub ffn_compact: Option<std::sync::Arc<crate::ffn_compact::CompactSet>>,
    /// single-process streamed prefill (ffn_stream.rs, 2026-11-02): max prompt tokens of `stream_prefill` (0 = off, the
    /// default: no region, no kernel). Needs ffn_compact. Adds `ffn_stream_slots` full-layer FFN slots + the prompt's
    /// hidden rows to the budget (so the keep formula counts them).
    pub ffn_stream_tokens: usize,
    /// full-layer FFN slots of the streamed prefill (2 = the read of layer l+1 overlaps the GPU on layer l; 1 = serial)
    pub ffn_stream_slots: usize,
    /// place the streamed-prefill buffers inside the lm_head (idle until the head, re-read from the model file after the
    /// last layer; ffn_stream.rs). Hosted bytes leave the budget, so the keep formula gives them back to K. Default off.
    pub ffn_stream_host: bool,
    /// fast streamed prefill (ffn_stream.rs, 2026-11-03): each streamed layer runs the fast prefill kernels (GEMM
    /// projections + FFN, flash attention) on chunks of up to `prefill_t` rows (the full engine's fast `chunks`), the
    /// prompt-gate score from the swiglu GEMM's own gate output. Not bit-exact with the exact streamed path (GEMM /
    /// flash summation order); equal to the full engine's fast prefill. Needs ffn_stream_tokens > 0 and bf16 KV; the
    /// pass scratch grows to prefill_t rows (budget). Decode and `prefill` stay exact. Default off.
    pub ffn_stream_fast: bool,
    /// decode speed-ups that keep every output bit-identical (2026-11-03, decode_bench): the Mma twins on dynamic
    /// threadgroup memory (`GemvImpl::MmaD`, exact-pass matmuls only; gemv = Mma) and the encode-ahead synchronous
    /// decode (`feed_stream` / `decode_sync` / `fsel_step` encode the next pass while the GPU runs this one).
    /// Default off.
    pub decode_fused: bool,
    /// context drafting (ctx_draft.rs, 2026-11-03, queue item 0b "D"): max drafts per verify pass (0 = off, the default;
    /// 1..=SPEC_DEPTH_MAX). Drafts come from the context (n-gram lookup over every token in the KV) and the draft table
    /// when one is open; no MTP head. Verify = one exact pass of d + 1 rows that only reads the DeltaNet state, then the
    /// accepted rows are replayed in place: no second state copy. Adds the hold save ((d + 1) x replay row, 7.9 MB at
    /// d = 7 on the 27B) and, with a draft table, its RAM budget to the plan (so the 12 GB keep formula counts them).
    pub ctx_draft: usize,
    /// fused K-row verify matmuls (verify_fused_q4.metal, 2026-11-03): exact passes of 3..=8 rows with the Mma family
    /// and no read mask run `vf_gemv` / `vf_swiglu` (bit-identical to the Mma / MmaD kernels; synthetic 27B K = 8
    /// matmul pass 122.4 -> 94.0 ms). K <= 2, read-masked passes and the compact CDown keep today's kernels. Adds the
    /// `vf_xprep` scratch (~0.6 MB on the 27B). The planner's cost table is unchanged (`SpecCost::verify_fused` exists
    /// but is not applied: it over-drafted on the 27B). Default off.
    pub verify_fused: bool,
    /// K-row verify mixers (verify_mixers.metal, 2026-11-03), bit-identical per row to today's kernels: bf16-KV sdpa
    /// rows of every pass of <= 8 rows run va_sdpa_p1 / p3 / p2 instead of attn_sdpa_vec_b16 (synthetic 27B, one
    /// attention layer, K = 8: pos 1024 346 -> 246 µs, pos 4000 967 -> 651), and DeltaNet passes of 2..=8 rows run
    /// dn_recur_v (tokens staged in threadgroup memory). Scratch `va_scratch_bytes(cfg, 8)` (12.8 MB on the 27B) when
    /// the KV is bf16 (q8 KV keeps today's sdpa). `SD_VERIFY_MIXERS=1|0` forces it. Default off (needs a 27B check).
    pub verify_mixers: bool,
    /// CPU RAM of a history index the caller keeps next to the engine for context drafts (history_index.rs; the
    /// gateway's `[history]`). Only counted in the plan (so the 12 GB keep formula leaves room for it); 0 = none.
    pub history_bytes: usize,
    /// margin-gated recheck (gate_recheck.rs): a 1-row DeltaNet hold save so a decode pass can only
    /// READ the state (`gate_hold_pass`) and either replay its row (accept) or redo the position with the full FFN
    /// (`gate_recheck`). 0.99 MB on the 27B, a budget row (the keep formula counts it). Default off.
    pub gate_hold: bool,
    /// FFN-input capture (research harness): a bf16 [layers][hidden] region; each K = 1 pass copies every
    /// layer's FFN input there (`ffn_x_capture()`). 0.66 MB on the 27B (a scratch row). Default off.
    pub ffn_x_capture: bool,
    /// DFlash block drafter (dflash.rs, 2026-11-03): directory with dflash.json + dflash.safetensors (kit
    /// export_engine.py). Some: drafter weights + context KV + scratch join the plan, every `encode_pass_full` pass
    /// captures the tap rows into the drafter's context, and `feed_stream_dflash` drafts blocks verified in one exact
    /// pass (lossless). None = off (default: no region, no kernel, no capture).
    pub dflash: Option<std::path::PathBuf>,
    /// tests: also log the raw tap rows of every position (bf16 [max_ctx][taps x hidden], `dflash_tap_log`).
    pub dflash_log_taps: bool,
    /// DFlash2 block drafter (dflash2.rs, experimental): directory with config.json + model.safetensors. Some: weights +
    /// context KV + scratch join the plan, every `encode_pass_full` pass captures the tap rows, `feed_stream_dflash2`
    /// drafts blocks verified in one exact pass. None = off (default: no region, no kernel, no capture). Refused with
    /// `dflash`, MTP or the streamed prefill.
    pub dflash2: Option<std::path::PathBuf>,
    /// decode sequence slots (batch.rs, 2026-11-04): 1 = one sequence, today (default). n > 1 allocates n - 1 more
    /// slots (own KV, MTP KV, DeltaNet state, ctrl, rec, staging, replay, MTP rows; a budget row) so up to n sequences
    /// live at once and decode together in one batched verify pass (`QwenEngine::batch_round`). Needs MTP; no dump,
    /// read mask, ffn select, DFlash, context drafts, streamed prefill or the q8-conversion prototype.
    pub seqs: usize,
}
impl Default for QwenOptions {
    fn default() -> Self {
        QwenOptions { cap: 17_000_000_000, reserve: 400_000_000, max_ctx: 8192, kv: KvFormat::Bf16, mtp: false, spec_depth: 1, gemv: GemvImpl::Mma, prefill_k: 8,
                      prefill_t: 128, prefill_exact: false, load_threads: 4, embed_threads: 4, depth: 2, abort_at: 0, swap_abort: 7_000_000_000, dump: None,
                      draft_vocab: None, spec_auto: false, spec_cost: Default::default(), draft_table: None,
                      draft_table_bytes: crate::spec_deep::DRAFT_TABLE_BYTES, activity: true, activity_batch: false, live_cap: 0, read_mask: false, ffn_select: false,
                      ffn_compact: None, ffn_stream_tokens: 0, ffn_stream_slots: 2, ffn_stream_host: false, ffn_stream_fast: false,
                      decode_fused: false, ctx_draft: 0, verify_fused: false, verify_mixers: false, history_bytes: 0, gate_hold: false, ffn_x_capture: false,
                      dflash: None, dflash_log_taps: false, dflash2: None, seqs: 1 }
    }
}

/// Teacher-forced head output of one position: top-k (idx, bf16-rounded logit) by value, and logsumexp.
#[derive(Clone, Debug, Default)]
pub struct TopK {
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
    pub lse: f32,
}

#[derive(Default, Debug, Clone)]
pub struct GenResult {
    pub tokens: Vec<u32>,
    /// bf16 top-1 − top-2 logit per generated token
    pub margins: Vec<f32>,
    /// GPU time per decode command buffer (the first token comes from the prefill)
    pub gpu_ms: Vec<f64>,
    /// decode wall time (s), first decode commit → last token
    pub gen_wall_s: f64,
    pub prefill_s: f64,
    /// peak phys_footprint seen by the engine monitor, else the footprint sampled at the end (bytes)
    pub peak: u64,
    /// pipelined only: hand-off latency (GPU signal → next CB committed) and its pread part
    pub handoff: Option<LatStats>,
    pub pread: Option<LatStats>,
}

/// `feed_stream` result. `decode_s` runs from the first token (out of the prefill) to the last; `gen` counts the
/// prefill's token too, so decode ms/token = decode_s / (gen - 1).
#[derive(Default, Debug, Clone)]
pub struct StreamStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub gpu_ms: f64,
    pub gen: usize,
    pub last: Option<u32>,
    pub stop_reason: &'static str,
}

/// Arena regions outside the weights.
pub struct QwenScratch {
    /// `k = tmax`, `nch_max` = the max_ctx bound; passes copy it with their own k / nch_max
    pub io: PassIo,
    pub tk_idx: Region,
    pub tk_val: Region,
    pub tk_lse: Region,
    /// bound at the q8 scale slots of a bf16 KV (unused there)
    pub dummy: Region,
    pub dump: Option<Region>,
}

/// Every size derived from the config and the options; shared by the budget and the allocation.
struct Layout {
    /// pruned draft vocab ids (opt.draft_vocab)
    keep: Option<Vec<u32>>,
    tmax: usize,
    nch_max: usize,
    row_bytes: usize,
    slot_bytes: usize,
    scratch: Vec<(String, usize)>,
    kv_vals: usize,  // per K (or V) per attention layer
    kv_sb: usize,    // q8 scale/min per K (or V) per attention layer, 0 for bf16
    dump_bytes: usize,
    s_bytes: usize,
    conv_bytes: usize,
    replay_bytes: usize, // spec depth >= 2: DeltaNet replay inputs (spec_deep.rs), 0 otherwise
    hold_bytes: usize,   // ctx_draft > 0: hold save of a context-draft verify (ctx_draft.rs), 0 otherwise
    act_bytes: (usize, usize), // activity recorder (accumulators, prev), (0, 0) when off
    rm_bytes: usize,  // read mask region, 0 when off
    dflash: Option<crate::dflash::DfPlan>, // DFlash drafter plan (opt.dflash), None when off
    dflash2: Option<crate::dflash2::Df2Plan>, // DFlash2 drafter plan (opt.dflash2), None when off
    regions: usize,  // arena regions (weights + ours), for the alignment slack row
    seq_bytes: usize, // one extra sequence slot (opt.seqs > 1, batch.rs)
}

fn weight_regions(cfg: &QwenConfig, mtp: bool) -> usize {
    // per layer: 2 norms + FFN 3×(W, S, B); DeltaNet: in/out 2×3 + conv, A_log, dt_bias, norm; attention: 2×3 + q/k norm
    let per = |k: LayerKind| 2 + 9 + match k { LayerKind::DeltaNet => 10, LayerKind::Attention => 8 };
    let main: usize = cfg.kinds.iter().map(|k| per(*k)).sum::<usize>() + 1 + 3;
    main + if mtp { 4 + per(LayerKind::Attention) } else { 0 }
}

/// q8-conversion prototype switch (bench / env only, default off): `SD_PF_Q8CONV=1` with `--kv q8` and the fast prefill on =
/// fast bf16 prefill of a prompt that starts at position 0, then one in-place conversion of its KV to q8, then q8
/// decode. Prefills that start at pos > 0 take today's exact q8 prefill.
pub fn pf_q8conv() -> bool { std::env::var("SD_PF_Q8CONV").map(|v| v == "1").unwrap_or(false) }

fn layout(cfg: &QwenConfig, opt: &QwenOptions) -> Result<Layout, String> {
    if !KS.contains(&opt.prefill_k) { return Err(format!("prefill_k {} not in {:?}", opt.prefill_k, KS)); }
    if opt.max_ctx == 0 { return Err("max_ctx = 0".into()); }
    let stream_fast = opt.ffn_stream_fast && opt.ffn_stream_tokens > 0;
    if (!opt.prefill_exact || stream_fast) && !(9..=PREFILL_T_MAX).contains(&opt.prefill_t) {
        return Err(format!("prefill_t {} not in 9..={}", opt.prefill_t, PREFILL_T_MAX));
    }
    if !(1..=crate::spec_deep::SPEC_DEPTH_MAX).contains(&opt.spec_depth) {
        return Err(format!("spec_depth {} not in 1..={}", opt.spec_depth, crate::spec_deep::SPEC_DEPTH_MAX));
    }
    if opt.spec_depth > 1 && !opt.mtp { return Err(format!("spec_depth {} needs MTP (--mtp)", opt.spec_depth)); }
    if opt.spec_auto && opt.spec_depth < 2 { return Err("spec_auto needs spec_depth >= 2 (the largest depth; replay buffer sized for it)".into()); }
    if opt.spec_depth > 1 && !EXACT_KS.contains(&(opt.spec_depth + 1)) {
        return Err(format!("spec_depth {}: no exact pass of {} rows (exact K: {:?})", opt.spec_depth, opt.spec_depth + 1, EXACT_KS));
    }
    // MTP verify needs 2 rows (deep verify d + 1, and 4 MTP emb rows for the chained drafts); fast prefill passes have
    // up to prefill_t rows (po / pml / logits stay at 8 rows)
    if opt.ctx_draft > crate::spec_deep::SPEC_DEPTH_MAX {
        return Err(format!("ctx_draft {} not in 0..={}", opt.ctx_draft, crate::spec_deep::SPEC_DEPTH_MAX));
    }
    let spec_rows = if opt.spec_depth > 1 { (opt.spec_depth + 1).max(4) } else { 2 };
    // context-draft verify: d + 1 rows
    let spec_rows = spec_rows.max(opt.ctx_draft + 1);
    // DFlash drafter: verify passes of up to `block` rows (dflash.rs)
    let df = match &opt.dflash { Some(p) => Some(crate::dflash::DfPlan::read(p, cfg)?), None => None };
    if opt.dflash2.is_some() && (opt.dflash.is_some() || opt.mtp || opt.ffn_stream_tokens > 0) {
        return Err("dflash2: not with the DFlash drafter, MTP or the streamed prefill (12 GB mode)".into());
    }
    let df2 = match &opt.dflash2 { Some(p) => Some(crate::dflash2::Df2Plan::read(p, cfg)?), None => None };
    let df_d = df.as_ref().map_or(0, |d| d.cfg.block - 1).max(df2.as_ref().map_or(0, |d| d.cfg.block - 1));
    let spec_rows = spec_rows.max(df_d + 1);
    // fast streamed prefill (ffn_stream_fast): its passes have up to prefill_t rows (at most the prompt cap)
    let tmax = (if opt.prefill_exact && !stream_fast { opt.prefill_k }
                else if opt.prefill_exact { opt.prefill_k.max(opt.prefill_t.min(opt.ffn_stream_tokens)) }
                else { opt.prefill_k.max(opt.prefill_t) }).max(spec_rows);
    let nch_max = opt.max_ctx.div_ceil(ATTN_CHUNK);
    let row_bytes = crate::embed::row_bytes(cfg.hidden);
    let slot_bytes = tmax * row_bytes;
    let mut scratch: Vec<(String, usize)> = scratch_sizes(cfg, tmax, nch_max).into_iter().map(|(n, b)| (n.to_string(), b)).collect();
    scratch.extend([
        ("ctrl".to_string(), CTRL_REGION),
        ("rec".to_string(), (opt.max_ctx + tmax) * 16),
        ("stage".to_string(), 2 * slot_bytes),
        ("tk_idx".to_string(), tmax * TOPK_MAX * 4),
        ("tk_val".to_string(), tmax * TOPK_MAX * 4),
        ("tk_lse".to_string(), tmax * 4),
        ("dummy".to_string(), 256),
    ]);
    if opt.ffn_x_capture { scratch.push(("fxcap".to_string(), cfg.layers * cfg.hidden * 2)); }
    if opt.mtp { scratch.extend(crate::mtp::scratch_sizes(cfg, tmax).into_iter().map(|(n, b)| (n.to_string(), b))); }
    // batch.rs: the fused MTP draft step gathers one row per sequence (cat, layer residual, normed hidden)
    if opt.mtp && opt.seqs > 1 {
        let h = cfg.hidden * 2;
        scratch.extend([("mtp.bcat".to_string(), opt.seqs * 2 * h), ("mtp.bh".to_string(), opt.seqs * h), ("mtp.bhn".to_string(), opt.seqs * h)]);
    }
    if opt.verify_fused {
        let (xt, st) = crate::qwen_layers::vf_prep_bytes(8, crate::qwen_layers::vf_max_cols(cfg));
        scratch.extend([("vf_xt".to_string(), xt), ("vf_st".to_string(), st)]);
    }
    if opt.verify_mixers && opt.kv == KvFormat::Bf16 { scratch.push(("va".to_string(), crate::qwen_layers::va_scratch_bytes(cfg, 8))); }
    let n = cfg.kv_heads * opt.max_ctx * cfg.head_dim;
    let (kv_vals, kv_sb) = match opt.kv {
        KvFormat::Bf16 => (n * 2, 0),
        // q8-conversion prototype (pf_q8conv): q8 KV whose K / V regions are bf16-sized, so the fast (bf16) prefill can write
        // them and `pf_convert` turns them into q8 in place (vals in the first half, scales in ks / vs as usual)
        // (scales in the same region after the n q8 bytes: n / 8 bytes <= n, so peak RAM = the bf16 KV's)
        KvFormat::Q8 if pf_q8conv() && !opt.prefill_exact => (n * 2, 0),
        KvFormat::Q8 => (n, cfg.kv_heads * opt.max_ctx * (cfg.head_dim / 32) * 4),
    };
    let copies = if opt.mtp { 2 } else { 1 };
    let s_bytes = DnState::s_bytes(cfg, copies);
    let conv_bytes = DnState::conv_bytes(cfg, copies);
    let dump_bytes = opt.dump.as_ref().map_or(0, |d| dump_bytes(cfg, &d.layers));
    let replay_bytes = crate::spec_deep::replay_bytes(cfg, opt.spec_depth);
    let hold_bytes = crate::ctx_draft::hold_bytes(cfg, opt.ctx_draft.max(df_d))
        .max(if opt.gate_hold { crate::spec_deep::replay_row_bytes(cfg) } else { 0 });
    let keep = match &opt.draft_vocab {
        Some(p) if opt.mtp => Some(crate::draft_vocab::KeepList::read_for(p, cfg.vocab)?.ids),
        Some(p) => return Err(format!("draft_vocab {}: needs MTP (--mtp)", p.display())),
        None => None,
    };
    if let Some(c) = &opt.ffn_compact {
        // MTP is allowed (2026-11-02): the MTP layer keeps its full FFN; verify passes run the compact layers over K
        // rows with the multi-row down (ffn_compact.rs `downk_kernel`), per row bit-identical to K = 1 decode.
        if !opt.prefill_exact || opt.activity || opt.read_mask || opt.ffn_select {
            return Err("ffn_compact needs prefill_exact and activity / read_mask / ffn_select off".into());
        }
        if c.keep.len() != cfg.layers || c.inter != cfg.inter { return Err(format!("ffn_compact: set is {:?}, model has {} layers x {}", c, cfg.layers, cfg.inter)); }
        c.validate()?;
    }
    if opt.ffn_stream_tokens > 0 {
        if opt.ffn_compact.is_none() { return Err("ffn_stream_tokens needs ffn_compact".into()); }
        if !(1..=4).contains(&opt.ffn_stream_slots) { return Err(format!("ffn_stream_slots {} not in 1..=4", opt.ffn_stream_slots)); }
        if opt.ffn_stream_tokens > opt.max_ctx { return Err(format!("ffn_stream_tokens {} > max_ctx {}", opt.ffn_stream_tokens, opt.max_ctx)); }
    }
    if opt.ffn_stream_host && opt.ffn_stream_tokens == 0 { return Err("ffn_stream_host needs ffn_stream_tokens > 0".into()); }
    if opt.ffn_stream_fast {
        if opt.ffn_stream_tokens == 0 { return Err("ffn_stream_fast needs ffn_stream_tokens > 0".into()); }
        if opt.kv != KvFormat::Bf16 { return Err("ffn_stream_fast needs bf16 KV (the fast prefill attention)".into()); }
        if opt.dump.is_some() { return Err("ffn_stream_fast: no dump mode".into()); }
    }
    let host_pool = opt.ffn_stream_host.then(|| crate::ffn_stream::host_pool(cfg.vocab, cfg.hidden));
    if opt.seqs == 0 || opt.seqs > crate::batch::SEQS_MAX { return Err(format!("seqs {} not in 1..={}", opt.seqs, crate::batch::SEQS_MAX)); }
    if opt.seqs > 1 {
        if !opt.mtp { return Err("seqs > 1 (batched decode) needs MTP (--mtp)".into()); }
        if opt.dump.is_some() || opt.read_mask || opt.ffn_select || opt.dflash.is_some() || opt.dflash2.is_some() || opt.ctx_draft > 0 || opt.ffn_stream_tokens > 0
            || (opt.kv == KvFormat::Q8 && pf_q8conv() && !opt.prefill_exact) {
            return Err("seqs > 1: no dump, read mask, ffn select, DFlash, context drafts, streamed prefill or SD_PF_Q8CONV".into());
        }
    }
    let act_bytes = if opt.activity { crate::activity::sizes(cfg) } else { (0, 0) };
    let rm_bytes = if opt.read_mask { crate::readmask::layout(cfg).1 } else { 0 };
    let kv_regions = if kv_sb > 0 { 4 } else { 2 };
    let regions = weight_regions(cfg, opt.mtp) + scratch.len() + (cfg.n_attention() + opt.mtp as usize) * kv_regions + 2
        + opt.dump.is_some() as usize + (replay_bytes > 0) as usize + (hold_bytes > 0) as usize + keep.is_some() as usize + 2 * opt.activity as usize + (opt.activity && opt.activity_batch) as usize + opt.read_mask as usize + opt.ffn_select as usize
        + opt.ffn_compact.as_ref().map_or(0, |_| 1 + 4 * cfg.layers)
        + if opt.ffn_stream_tokens > 0 { crate::ffn_stream::regions_ex(cfg.hidden, cfg.inter, tmax, opt.ffn_stream_tokens, opt.ffn_stream_slots, host_pool) } else { 0 }
        + df.as_ref().map_or(0, |d| d.regions(opt.dflash_log_taps))
        + df2.as_ref().map_or(0, |d| d.regions());
    let seq_sc = seq_scratch(cfg, opt, tmax);
    let extra = opt.seqs.max(1) - 1;
    let regions = regions + extra * ((cfg.n_attention() + opt.mtp as usize) * kv_regions + 2 + (replay_bytes > 0) as usize + seq_sc.len());
    let seq_bytes = (cfg.n_attention() + opt.mtp as usize) * 2 * (kv_vals + kv_sb) + s_bytes + conv_bytes + replay_bytes
        + seq_sc.iter().map(|x| x.1).sum::<usize>();
    Ok(Layout { keep, tmax, nch_max, row_bytes, slot_bytes, scratch, kv_vals, kv_sb, dump_bytes, s_bytes, conv_bytes, replay_bytes, hold_bytes, act_bytes, rm_bytes, dflash: df, dflash2: df2, regions, seq_bytes })
}

/// Dump region for one k = 1 pass: the largest per-layer capture set (plan 9.3 names, as `qwen_layers` captures
/// them) times the enabled layers, plus `emb` / `hf`, each slot padded to 16 B.
fn dump_bytes(cfg: &QwenConfig, layers: &[usize]) -> usize {
    let (h, bf, f4) = (cfg.hidden, 2, 4);
    let common = (3 * h + cfg.inter) * bf + 4 * 16; // x_in, h_mid, h_out, ffn.act
    let dn = common + (cfg.dn_in_rows() + cfg.conv_dim() + 2 * cfg.dn_k_dim() + 3 * cfg.dn_v_dim() + cfg.dn_vh
        + (cfg.conv_k - 1) * cfg.conv_dim()) * bf + (cfg.dn_vh + cfg.dn_vh * cfg.dn_dv * cfg.dn_dk) * f4 + 11 * 16;
    let at = common + (cfg.attn_in_rows() + 2 * cfg.q_dim()) * bf + 3 * 16;
    let n = if layers.is_empty() { cfg.layers } else { layers.iter().filter(|&&l| l < cfg.layers).count() };
    n * dn.max(at) + 2 * (h * bf + 16) + 2 * (TOPK_MAX * 4 + 16) + (4 + 16) + 4096
}

/// Concatenate per-step dump files along dim 0 (state tensors: the last step's), write `out`, remove the parts.
fn merge_dumps(parts: &[String], out: &str) -> Result<(), String> {
    if parts.is_empty() { return Err("dump: no passes".into()); }
    let idx: Vec<SafeIndex> = parts.iter().map(|p| SafeIndex::open_file(p)).collect::<Result<_, _>>()?;
    let data: Vec<Vec<u8>> = parts.iter().map(|p| std::fs::read(p).map_err(|e| format!("{}: {}", p, e))).collect::<Result<_, _>>()?;
    let li = idx.len() - 1;
    let mut owned: Vec<(String, DType, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, t) in idx[li].iter() {
        let at = |i: usize, t: &streamdense_core::TensorInfo| data[i][t.off as usize..t.off as usize + t.nbytes].to_vec();
        if name.ends_with(".dn.S") || name.ends_with(".dn.conv_state") {
            owned.push((name.to_string(), t.dtype, t.shape.clone(), at(li, t)));
            continue;
        }
        let (mut rows, mut bytes) = (0usize, Vec::new());
        for (i, ix) in idx.iter().enumerate() {
            if let Some(ti) = ix.get(name) { rows += ti.shape[0]; bytes.extend_from_slice(&at(i, ti)); }
        }
        let mut shape = t.shape.clone();
        shape[0] = rows;
        owned.push((name.to_string(), t.dtype, shape, bytes));
    }
    let views: Vec<(&str, DType, &[usize], &[u8])> = owned.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice())).collect();
    write_safetensors(out, &views)?;
    for p in parts { let _ = std::fs::remove_file(p); }
    Ok(())
}

fn open_model(dir: &str) -> Result<(SafeIndex, QwenConfig), String> {
    let cfg = QwenConfig::from_model_dir(dir)?;
    cfg.validate()?;
    let idx = SafeIndex::open_mlx_dir(dir)?;
    Ok((idx, cfg))
}

/// Every arena row of plan section 2, from the headers and the options (nothing allocated, nothing read).
fn plan_budget(idx: &SafeIndex, cfg: &QwenConfig, opt: &QwenOptions) -> Result<(Budget, Layout), String> {
    let lay = layout(cfg, opt)?;
    let mut bud = Budget::new(opt.cap, opt.reserve);
    bud.live_cap = opt.live_cap;
    weights::plan_ex(idx, cfg, opt.mtp, opt.ffn_compact.as_deref(), &mut bud)?;
    bud.add("embedding staging (SSD rows)", lay.slot_bytes as u64 * 2);
    bud.add(if opt.mtp { "DeltaNet state x2 + conv x2 (MTP rollback)" } else { "DeltaNet state + conv (1 copy, plain)" },
            (lay.s_bytes + lay.conv_bytes) as u64);
    if lay.replay_bytes > 0 {
        bud.add(&format!("DeltaNet replay inputs (spec depth {})", opt.spec_depth), lay.replay_bytes as u64);
    }
    if lay.hold_bytes > 0 && opt.ctx_draft == 0 && lay.dflash.is_none() && lay.dflash2.is_none() {
        bud.add("margin gate: DeltaNet hold save (1 row)", lay.hold_bytes as u64);
    }
    if lay.hold_bytes > 0 && opt.ctx_draft > 0 {
        bud.add(&format!("context drafts: DeltaNet hold save (ctx_draft {}: {} rows)", opt.ctx_draft, opt.ctx_draft + 1), lay.hold_bytes as u64);
        // CPU-heap helpers (lookup index, draft table): counted here so the 12 GB keep formula leaves room for them
        bud.add(&format!("context drafts: lookup index (CPU, {} ctx)", opt.max_ctx), crate::ctx_draft::lookup_bytes(opt.max_ctx) as u64);
        if opt.draft_table.is_some() { bud.add("context drafts: draft table (CPU RAM budget)", opt.draft_table_bytes as u64); }
        if opt.history_bytes > 0 { bud.add("context drafts: history index (CPU RAM budget)", opt.history_bytes as u64); }
    }
    let kvname = match opt.kv { KvFormat::Bf16 => "bf16", KvFormat::Q8 => "8-bit g32" };
    let kv_layer = 2 * (lay.kv_vals + lay.kv_sb);
    bud.add(&format!("KV {} @ {} ctx", kvname, opt.max_ctx), (cfg.n_attention() * kv_layer) as u64);
    if opt.mtp { bud.add(&format!("MTP KV {} @ {} ctx", kvname, opt.max_ctx), kv_layer as u64); }
    let sc: usize = lay.scratch.iter().map(|x| x.1).sum();
    bud.add("scratch (pass rows, logits, partials, ctrl, rec)", sc as u64);
    if opt.dump.is_some() { bud.add("dump region", lay.dump_bytes as u64); }
    if opt.activity { bud.add("activity recorder (usage sums + residual rows)", (lay.act_bytes.0 + lay.act_bytes.1) as u64); }
    if opt.activity && opt.activity_batch { bud.add("activity batch (same-label jobs before a disk fold)", lay.act_bytes.0 as u64); }
    if opt.read_mask { bud.add("read mask (mask words + bf16 column norms)", lay.rm_bytes as u64); }
    if opt.ffn_select { bud.add("ffn select (test hook)", crate::ffn_select::layout(cfg, lay.tmax).0 as u64); }
    if opt.ffn_compact.is_some() { bud.add("ffn compact f-sum scratch", crate::ffn_compact::scratch_bytes(cfg.inter, lay.tmax) as u64); }
    if opt.ffn_stream_tokens > 0 {
        for (n, b) in crate::ffn_stream::budget_rows_ex(cfg.hidden, cfg.inter, lay.tmax, opt.ffn_stream_tokens, opt.ffn_stream_slots,
                                                         opt.ffn_stream_host.then(|| crate::ffn_stream::host_pool(cfg.vocab, cfg.hidden))) { bud.add(&n, b); }
    }
    if let Some(d) = &lay.dflash {
        if opt.ctx_draft == 0 { bud.add(&format!("DFlash drafts: DeltaNet hold save ({} rows)", d.cfg.block), lay.hold_bytes as u64); }
        for (n, b) in d.budget_rows(lay.tmax, opt.max_ctx, opt.dflash_log_taps) { bud.add(&n, b); }
    }
    if let Some(d) = &lay.dflash2 {
        if opt.ctx_draft == 0 { bud.add(&format!("DFlash2 drafts: DeltaNet hold save ({} rows)", d.cfg.block), lay.hold_bytes as u64); }
        for (n, b) in d.budget_rows(lay.tmax, opt.max_ctx) { bud.add(&n, b); }
    }
    if opt.seqs > 1 {
        bud.add(&format!("sequence slots 2..{} (batch.rs: KV + MTP KV + DeltaNet state x2 + replay + ctrl / rec / staging / MTP rows, {:.3} GB each)",
                         opt.seqs, lay.seq_bytes as f64 / 1e9), ((opt.seqs - 1) * lay.seq_bytes) as u64);
    }
    if let Some(k) = &lay.keep { bud.add(&format!("MTP draft vocab keep-list ({} ids)", k.len()), (k.len() * 4) as u64); }
    bud.add(&format!("arena alignment slack ({} regions)", lay.regions), (lay.regions * ALIGN) as u64);
    Ok((bud, lay))
}

/// Command buffer handed to the hand-off thread, which commits it.
struct SendCb(CmdBuf);
unsafe impl Send for SendCb {}

#[derive(Default)]
struct PendState { cbs: HashMap<u64, SendCb>, committed: u64, closed: bool }
#[derive(Default)]
struct Pending { m: Mutex<PendState>, cv: Condvar }

/// Bytes of each slot's ctrl region: the `SdCtrl` block, then the banned-id list u32 [n, id0 .. id30] at `BAN_OFF`.
pub const CTRL_REGION: usize = 256;
pub const BAN_OFF: usize = 128;
pub const BAN_MAX: usize = 31;
const _: () = assert!(std::mem::size_of::<SdCtrl>() <= BAN_OFF && BAN_OFF + 4 * (BAN_MAX + 1) <= CTRL_REGION);

/// The arena regions one decode sequence owns (batching, batch.rs): KV per attention layer, MTP KV, DeltaNet state
/// (both copies with MTP), the ctrl block, the rec table, the embedding staging slots, the deep-spec replay save and
/// the MTP scratch (mctrl, emb / hn / stage / conf rows; the eos ids and the draft keep-list are shared). Slot 0 is
/// the engine's classic state; `QwenOptions.seqs` adds more. The engine works on the slot `select_seq` picked.
pub struct SeqSlot {
    /// per model layer: Some for attention layers
    pub kv: Vec<Option<KvRef>>,
    /// P2: the MTP layer's own KV
    pub mtp_kv: Option<KvRef>,
    /// model-wide DeltaNet state (S[n_dn][2], conv[2][n_dn])
    pub dn: DnState,
    pub dn_s: Region,
    pub dn_conv: Region,
    pub ctrl: Region,
    pub rec: Region,
    /// 2 staging slots of `tmax` embedding rows
    pub stage: Region,
    /// spec depth >= 2: DeltaNet replay inputs (spec_deep.rs)
    pub dn_replay: Option<Region>,
    /// P2: MTP scratch (Some with opt.mtp)
    pub mtp_s: Option<crate::mtp::MtpScratch>,
    /// host copy of ctrl.pos at the next encoded pass
    pub pos: Cell<usize>,
}

/// Scratch regions each extra sequence slot allocates for itself (slot 0 has them in the shared scratch list).
fn seq_scratch(cfg: &QwenConfig, opt: &QwenOptions, tmax: usize) -> Vec<(&'static str, usize)> {
    let mut v = vec![("ctrl", CTRL_REGION), ("rec", (opt.max_ctx + tmax) * 16),
                     ("stage", 2 * tmax * crate::embed::row_bytes(cfg.hidden))];
    if opt.mtp { v.extend(crate::mtp::scratch_sizes(cfg, tmax).into_iter().filter(|x| x.0 != "mtp.eos")); }
    v
}

pub struct QwenEngine {
    pub gpu: Gpu,
    pub buf: Buffer,
    pub arena: Arena,
    pub cfg: QwenConfig,
    pub opt: QwenOptions,
    pub w: Weights,
    pub s: QwenScratch,
    /// per-sequence state (KV, MTP KV, DeltaNet state, ctrl, rec, staging, replay, MTP scratch): slot 0 always, more
    /// with `opt.seqs` (batch.rs). Every pass, prefill, snapshot and spec loop works on the selected slot `cur`.
    pub(crate) slots: Vec<SeqSlot>,
    pub(crate) cur: Cell<usize>,
    pub reader: Arc<EmbedReader>,
    pub dumper: Option<Dumper>,
    pub(crate) tmax: usize,
    pub(crate) nch_max: usize,
    pub(crate) slot_bytes: usize,
    /// ctx_draft > 0: hold save of the context-draft verify (ctx_draft.rs)
    pub(crate) ctx_hold: Option<Region>,
    pub monitor: Option<Monitor>,
    pub load_s: f64,
    pub loaded_bytes: u64,
    pub budget_report: String,
    /// extra draft source of the adaptive spec loop (the learning `DraftTable` with opt.draft_table, or a test stub
    /// via `set_draft_source`); proposals are only drafts, the verify pass decides every token
    pub(crate) draft_src: std::cell::RefCell<Option<Box<dyn crate::spec_deep::DraftSource>>>,
    /// activity recorder handle (Some with opt.activity); thread-safe, see `activity_tap`
    pub activity: Option<Arc<crate::activity::ActivityTap>>,
    /// read mask state (Some with opt.read_mask; readmask.rs)
    pub(crate) rm: Option<std::cell::RefCell<crate::readmask::RmState>>,
    /// forheader compact FFN state (Some with opt.ffn_compact; ffn_compact.rs)
    pub(crate) cfc: Option<crate::ffn_compact::CompactState>,
    /// streamed prefill regions (Some with opt.ffn_stream_tokens > 0; ffn_stream.rs)
    pub(crate) cst: Option<crate::ffn_stream::StreamRegions>,
    /// zero-copy recheck: no-copy Metal buffers over the mapped model file's FFN tensors, built on the first
    /// missing-only recheck (None inside = mapping / buffer creation failed, the copy path is used; ffn_stream.rs)
    pub(crate) zc: std::cell::OnceCell<Option<crate::ffn_stream::ZeroCopy>>,
    /// decode_fused encode-ahead: a K = 1 decode pass (stage slot 0, head on the row) already encoded for host position
    /// `.0` from stage slot `.1`, not committed (`decode_cb` / `encode_ahead`)
    pub(crate) ahead: std::cell::RefCell<Option<(usize, usize, CmdBuf)>>,
    /// DFlash drafter (Some with opt.dflash; dflash.rs)
    pub(crate) dflash: Option<crate::dflash::DFlash>,
    /// DFlash2 drafter (Some with opt.dflash2; dflash2.rs)
    pub(crate) dflash2: Option<crate::dflash2::DFlash2>,
    /// q8-conversion prototype: bf16 views of `kv` (Some when pf_q8conv is on) and "a converting fast prefill is running"
    pub(crate) pf_bf: Option<Vec<Option<KvRef>>>,
    pub(crate) pf_on: Cell<bool>,
    /// batch.rs fused MTP draft step: arena offsets of (cat [seqs][2H], residual [seqs][H], normed [seqs][H]); seqs > 1
    pub(crate) bmtp: Option<(usize, usize, usize)>,
}

impl QwenEngine {
    /// Dry run: the budget table from the headers (callers print `report()` and call `check()`).
    /// Check a draft-vocab keep-list against the model in `dir` (the model vocab from its config): Err = unreadable or
    /// made for another vocab (callers warn and run without it: output is the same, drafts use the full vocab).
    pub fn check_draft_vocab(dir: &str, path: &std::path::Path) -> Result<usize, String> {
        let (_, cfg) = open_model(dir)?;
        Ok(crate::draft_vocab::KeepList::read_for(path, cfg.vocab)?.ids.len())
    }

    pub fn budget(dir: &str, opt: &QwenOptions) -> Result<Budget, String> {
        let (idx, cfg) = open_model(dir)?;
        Ok(plan_budget(&idx, &cfg, opt)?.0)
    }

    pub fn open(dir: &str, opt: QwenOptions) -> Result<QwenEngine, String> {
        let mut opt = opt;
        // A/B and test runs: SD_VERIFY_FUSED=1|0 forces the option (whole test files with the fused verify kernels)
        if let Ok(v) = std::env::var("SD_VERIFY_FUSED") { opt.verify_fused = v == "1"; }
        if let Ok(v) = std::env::var("SD_VERIFY_MIXERS") { opt.verify_mixers = v == "1"; }
        // NOT auto-applied: SpecCost::verify_fused() (K = 4 cheaper than K = 3, -28 ms at K >= 5) made the 27B 12 GB
        // ctx-draft planner draft more on context-heavy prompts (K 4..8 passes 116 -> 1256, acceptance 0.30 -> 0.21) and lose 10% there
        // (2026-11-03 connector); the measured 12 GB saving is ~15 ms per K = 8 pass. Pass it via spec_cost to try.
        let monitor = if opt.abort_at > 0 { Some(Monitor::start(opt.abort_at, opt.swap_abort, 200)) } else { None };
        let (idx, cfg) = open_model(dir)?;
        // ---- budget (everything the arena will hold), checked before any allocation
        let (bud, lay) = plan_budget(&idx, &cfg, &opt)?;
        let arena_bytes = bud.check()? as usize;
        let report = bud.report();
        eprintln!("budget:\n{}", report);

        let mut shaders: Vec<&str> = QWEN_SHADERS.to_vec();
        if opt.mtp { for f in crate::mtp::MTP_SHADERS { if !shaders.contains(f) { shaders.push(f); } } }
        if opt.activity { shaders.push("activity"); }
        if opt.read_mask { shaders.push("readmask"); }
        if opt.ffn_select { shaders.push("ffn_select"); }
        if opt.ffn_compact.is_some() { shaders.push("ffn_compact"); }
        if opt.dflash.is_some() { shaders.push(crate::dflash::SHADER); }
        if opt.dflash2.is_some() { shaders.push(crate::dflash2::SHADER); }
        let mut gpu = Gpu::from_shader_files(&shaders)?;
        if arena_bytes > gpu.max_buffer() { return Err(format!("arena {} B > maxBufferLength {}", arena_bytes, gpu.max_buffer())); }
        let buf = gpu.buffer(arena_bytes)?;
        let mut arena = Arena::new(contents(&buf), arena_bytes);

        // ---- weights: allocate + read (F_NOCACHE pread, several threads, no mmap)
        let t0 = Instant::now();
        let (w, loaded, cjobs) = weights::load_ex(&idx, &cfg, &mut arena, opt.mtp, opt.load_threads, opt.ffn_compact.as_deref())?;
        let cfc = opt.ffn_compact.as_ref().map(|c| crate::ffn_compact::CompactState {
            files: idx.files().to_vec(), jobs: cjobs, set: std::cell::RefCell::new(c.clone()), hidden: cfg.hidden, inter: cfg.inter,
            threads: opt.load_threads });
        let load_s = t0.elapsed().as_secs_f64();

        // ---- KV per attention layer (+ MTP)
        let q8 = opt.kv == KvFormat::Q8;
        let mut sr: HashMap<String, Region> = HashMap::new();
        for (n, b) in &lay.scratch { sr.insert(n.clone(), arena.alloc(&format!("s:{}", n), *b)?); }
        let dummy = sr["dummy"].off;
        let kv_ref = |arena: &mut Arena, name: &str| -> Result<KvRef, String> {
            let k = arena.alloc(&format!("kv:{}.k", name), lay.kv_vals)?.off;
            let v = arena.alloc(&format!("kv:{}.v", name), lay.kv_vals)?.off;
            let conv = q8 && lay.kv_sb == 0; // pf_q8conv layout: [q8 vals n | scales n / 8 | unused] in a 2n region
            let (ksb, vsb) = if conv { (k + lay.kv_vals / 2, v + lay.kv_vals / 2) } else if q8 {
                (arena.alloc(&format!("kv:{}.ks", name), lay.kv_sb)?.off, arena.alloc(&format!("kv:{}.vs", name), lay.kv_sb)?.off)
            } else { (dummy, dummy) };
            Ok(KvRef { k, v, ksb, vsb, q8, max_ctx: opt.max_ctx })
        };
        let mut kv = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            kv.push(if cfg.kinds[l] == LayerKind::Attention { Some(kv_ref(&mut arena, &l.to_string())?) } else { None });
        }
        let mtp_kv = if opt.mtp { Some(kv_ref(&mut arena, "mtp")?) } else { None };
        let pf_bf = (q8 && pf_q8conv() && !opt.prefill_exact).then(|| {
            eprintln!("pf_q8conv: fast bf16 prefill from pos 0, then in-place KV -> q8 (prototype)");
            kv.iter().map(|r: &Option<KvRef>| r.map(|r| KvRef { q8: false, ..r })).collect::<Vec<_>>()
        });

        // ---- DeltaNet state (both copies; the kernels pick ctrl.role, plain mode stays at role 0)
        let dn_s = arena.alloc("st:dn.S", lay.s_bytes)?;
        let dn_conv = arena.alloc("st:dn.conv", lay.conv_bytes)?;
        let dn = DnState { s: dn_s.off, conv: dn_conv.off, copies: if opt.mtp { 2 } else { 1 } };
        let dn_replay = if lay.replay_bytes > 0 { Some(arena.alloc("st:dn.replay", lay.replay_bytes)?) } else { None };
        let ctx_hold = if lay.hold_bytes > 0 { Some(arena.alloc("st:dn.hold", lay.hold_bytes)?) } else { None };
        let dump_r = match &opt.dump { Some(_) => Some(arena.alloc("s:dump", lay.dump_bytes)?), None => None };
        let dumper = match (&opt.dump, dump_r) { (Some(d), Some(r)) => Some(Dumper::new(r, d.layers.clone())), _ => None };
        let act_r = if opt.activity {
            let acc = arena.alloc("s:activity", lay.act_bytes.0)?;
            arena.slice_mut::<u8>(acc).fill(0);
            Some((acc, arena.alloc("s:activity.prev", lay.act_bytes.1)?))
        } else { None };
        let act = act_r.map(|(a, p)| crate::activity::ActIo::new(&cfg, a.off, p.off));
        let act_batch = if opt.activity && opt.activity_batch {
            let b = arena.alloc("s:activity.batch", lay.act_bytes.0)?;
            arena.slice_mut::<u8>(b).fill(0);
            Some(b.off)
        } else { None };

        let o = |n: &str| sr[n].off;
        let io = PassIo {
            k: lay.tmax, h: o("h"), x: o("x"), p: o("p"), cv: o("cv"), g: o("g"), beta: o("beta"), y: o("y"), yn: o("yn"),
            ao: o("ao"), f: o("f"), po: o("po"), pml: o("pml"), hf: o("hf"), ctrl: o("ctrl"), nch_max: lay.nch_max,
            logits: o("logits"), rec: o("rec"), invf: o("invf"), pos: 0, dn_save: None, dn_hold: None, gemv: if opt.decode_fused && opt.gemv == GemvImpl::Mma { GemvImpl::MmaD } else { opt.gemv }, act, rm: None, fsel: None,
            cfs: 0, vf: if opt.verify_fused { Some((o("vf_xt"), o("vf_st"))) } else { None },
            fxcap: if opt.ffn_x_capture { Some(o("fxcap")) } else { None },
            vm: if opt.verify_mixers { Some(if opt.kv == KvFormat::Bf16 { o("va") } else { crate::qwen_layers::VM_NO_ATTN }) } else { None },
            ban: None,
        };
        let mut io = io;
        if opt.ffn_select {
            let (n, f) = crate::ffn_select::layout(&cfg, lay.tmax);
            let r = arena.alloc("s:ffn_select", n)?;
            arena.slice_mut::<u8>(r).fill(0);
            io.fsel = Some(crate::ffn_select::at_offset(f, r.off));
            gpu.pipeline("ffn_select")?;
        }
        if opt.ffn_compact.is_some() {
            let r = arena.alloc("s:ffn_compact", crate::ffn_compact::scratch_bytes(cfg.inter, lay.tmax))?;
            io.cfs = r.off;
            for n in crate::ffn_compact::pipelines() { gpu.pipeline(n)?; }
        }
        let cst = if opt.ffn_stream_tokens > 0 {
            for n in crate::ffn_stream::pipelines() { gpu.pipeline(n)?; }
            if opt.ffn_stream_fast {
                for n in prefill_pipelines() { gpu.pipeline(n)?; }
                for n in gemm_pipelines().into_iter().chain(crate::qwen_layers::gemm_g_pipelines()) { gpu.pipeline(&n)?; }
            }
            let host = if opt.ffn_stream_host {
                let src = |s: &str| -> Result<(usize, u64), String> {
                    let n = format!("{}.{}", crate::weights::head_prefix(&cfg), s);
                    idx.get(&n).map(|t| (t.file, t.off)).ok_or_else(|| format!("ffn_stream_host: {} not found", n))
                };
                Some(crate::ffn_stream::HostPool { regions: [w.lm_head.w, w.lm_head.s, w.lm_head.b], src: [src("weight")?, src("scales")?, src("biases")?] })
            } else { None };
            Some(crate::ffn_stream::alloc_ex(&mut arena, cfg.hidden, cfg.inter, lay.tmax, opt.ffn_stream_tokens, opt.ffn_stream_slots, cfg.vocab, host)?)
        } else { None };
        let s = QwenScratch { io, tk_idx: sr["tk_idx"], tk_val: sr["tk_val"],
                              tk_lse: sr["tk_lse"], dummy: sr["dummy"], dump: dump_r };

        for n in qwen_pipelines() { gpu.pipeline(n)?; }
        if !opt.prefill_exact {
            for n in prefill_pipelines() { gpu.pipeline(n)?; }
            for n in gemm_pipelines() { gpu.pipeline(&n)?; }
        }
        if opt.mtp { for n in crate::mtp::pipelines() { gpu.pipeline(n)?; } }
        if dn_replay.is_some() { gpu.pipeline("dn_save_rows")?; }
        if ctx_hold.is_some() { for n in ["dn_save_all", "dn_recur_hold"] { gpu.pipeline(n)?; } }
        if opt.verify_fused { for n in crate::qwen_layers::vf_engine_pipelines() { gpu.pipeline(&n)?; } }
        if opt.verify_mixers { for n in crate::qwen_layers::vm_pipelines() { gpu.pipeline(&n)?; } }
        if act.is_some() { for n in crate::activity::pipelines() { gpu.pipeline(n)?; } }
        let rm = if opt.read_mask {
            let r = arena.alloc("s:readmask", lay.rm_bytes)?;
            arena.slice_mut::<u8>(r).fill(0);
            for n in crate::readmask::pipelines() { gpu.pipeline(&n)?; }
            Some(std::cell::RefCell::new(crate::readmask::RmState::new(&cfg, crate::readmask::at_offset(&cfg, r.off))))
        } else { None };
        let mut mtp_s = if opt.mtp { Some(crate::mtp::MtpScratch::from_regions(&cfg, lay.tmax, &|n: &str| sr[n])) } else { None };
        if let (Some(ids), Some(m)) = (&lay.keep, mtp_s.as_mut()) {
            let r = arena.alloc("s:mtp.keep", ids.len() * 4)?;
            arena.slice_mut::<u32>(r)[..ids.len()].copy_from_slice(ids);
            (m.keep, m.n_keep) = (r.off, ids.len());
            for n in crate::mtp::pruned_pipelines(opt.gemv) { gpu.pipeline(n)?; }
        }
        if let Some(m) = &mtp_s { for (i, t) in cfg.eos.iter().enumerate() { arena.slice_mut::<u32>(m.eos)[i] = *t; } }
        for n in ["embed_q4", "advance1", "topk32"] { gpu.pipeline(n)?; }
        let dflash = match &lay.dflash {
            Some(p) => Some(crate::dflash::DFlash::load(p, &mut arena, &mut gpu, lay.tmax, opt.max_ctx, opt.dflash_log_taps)?),
            None => None,
        };
        let dflash2 = match &lay.dflash2 {
            Some(p) => Some(crate::dflash2::DFlash2::load(p, &mut arena, &mut gpu, lay.tmax, opt.max_ctx)?),
            None => None,
        };
        let mut reader = EmbedReader::open(&w.embed, &cfg, opt.embed_threads)?;
        // token counters live in the recorder's arena region (model memory; zeroed with it above)
        let embed_counts: crate::activity::SharedCounts = Arc::new(match &act {
            Some(a) => crate::activity::TokCounts::in_arena(&buf, a.tok + cfg.vocab * 4, cfg.vocab),
            None => crate::activity::TokCounts::new(0),
        });
        if act.is_some() { reader.counts = Some(embed_counts.clone()); }
        let reader = Arc::new(reader);
        let draft_src = crate::spec_deep::open_draft_table(&opt)?;
        let activity = act.map(|a| {
            let down = w.layers.iter().map(|lw| lw.ffn.down).collect();
            let mtp_down = w.mtp.as_ref().map(|m| m.layer.ffn.down);
            Arc::new(crate::activity::ActivityTap::new(&buf, &cfg, a, down, mtp_down, arena.total("w:") as u64, (arena.size() as u64) + opt.reserve, embed_counts, act_batch))
        });

        // ---- sequence slots: slot 0 = the regions above; slots 1.. (opt.seqs, batch.rs) get their own copies
        let mut slots = vec![SeqSlot { kv, mtp_kv, dn, dn_s, dn_conv, ctrl: sr["ctrl"], rec: sr["rec"], stage: sr["stage"], dn_replay, mtp_s,
                                       pos: Cell::new(0) }];
        for i in 1..opt.seqs.max(1) {
            let mut kv = Vec::with_capacity(cfg.layers);
            for l in 0..cfg.layers {
                kv.push(if cfg.kinds[l] == LayerKind::Attention { Some(kv_ref(&mut arena, &format!("{}#{}", l, i))?) } else { None });
            }
            let mtp_kv = if opt.mtp { Some(kv_ref(&mut arena, &format!("mtp#{}", i))?) } else { None };
            let dn_s = arena.alloc(&format!("st:dn.S#{}", i), lay.s_bytes)?;
            let dn_conv = arena.alloc(&format!("st:dn.conv#{}", i), lay.conv_bytes)?;
            let dn = DnState { s: dn_s.off, conv: dn_conv.off, copies: if opt.mtp { 2 } else { 1 } };
            let dn_replay = if lay.replay_bytes > 0 { Some(arena.alloc(&format!("st:dn.replay#{}", i), lay.replay_bytes)?) } else { None };
            let mut own: HashMap<String, Region> = HashMap::new();
            for (n, b) in seq_scratch(&cfg, &opt, lay.tmax) { own.insert(n.to_string(), arena.alloc(&format!("s:{}#{}", n, i), b)?); }
            let mtp_s = slots[0].mtp_s.map(|m0| {
                let m = crate::mtp::MtpScratch::from_regions(&cfg, lay.tmax, &|n: &str| own.get(n).copied().unwrap_or(sr[n]));
                crate::mtp::MtpScratch { keep: m0.keep, n_keep: m0.n_keep, eos: m0.eos, ..m }
            });
            slots.push(SeqSlot { kv, mtp_kv, dn, dn_s, dn_conv, ctrl: own["ctrl"], rec: own["rec"], stage: own["stage"], dn_replay, mtp_s,
                                 pos: Cell::new(0) });
        }
        let bmtp_off = sr.get("mtp.bcat").map(|r| (r.off, sr["mtp.bh"].off, sr["mtp.bhn"].off));
        let eng = QwenEngine { gpu, buf, arena, cfg, opt, w, s, slots, cur: Cell::new(0), reader, dumper,
                               tmax: lay.tmax, nch_max: lay.nch_max, slot_bytes: lay.slot_bytes, ctx_hold, monitor, load_s,
                               loaded_bytes: loaded, budget_report: report,
                               draft_src: std::cell::RefCell::new(draft_src), activity, rm, cfc, cst,
                               ahead: std::cell::RefCell::new(None), zc: std::cell::OnceCell::new(), dflash, dflash2, pf_bf, pf_on: Cell::new(false),
                               bmtp: bmtp_off };
        debug_assert_eq!(lay.slot_bytes, lay.tmax * lay.row_bytes);
        {   // RoPE inverse frequencies (config only)
            let cb = eng.gpu.command_buffer();
            let e = Enc::new(&eng.gpu, &cb, &eng.buf);
            encode_rope_invf(&e, &eng.cfg, &eng.s.io);
            e.end();
            commit(&cb);
            wait(&cb)?;
        }
        eng.rm_init_norms();
        for i in (0..eng.slots.len()).rev() { eng.select_seq(i); eng.reset(); }
        Ok(eng)
    }

    /// The selected sequence slot (batch.rs; slot 0 unless `select_seq` moved it).
    pub fn sl(&self) -> &SeqSlot { &self.slots[self.cur.get()] }
    /// Sequence slots of this engine (`QwenOptions.seqs`, at least 1).
    pub fn n_seqs(&self) -> usize { self.slots.len() }
    /// The selected slot index.
    pub fn cur_seq(&self) -> usize { self.cur.get() }
    /// Make slot `i` the engine's state for every following call (passes, prefill, spec loops, snapshots). Nothing
    /// may be in flight for the old slot that the caller still expects to read through the engine.
    pub fn select_seq(&self, i: usize) {
        assert!(i < self.slots.len(), "select_seq {} of {} slots", i, self.slots.len());
        if i != self.cur.get() { self.ahead.borrow_mut().take(); }
        self.cur.set(i);
    }
    pub(crate) fn kv(&self) -> &[Option<KvRef>] { &self.sl().kv }
    pub fn mtp_kv(&self) -> &Option<KvRef> { &self.sl().mtp_kv }
    pub(crate) fn dn(&self) -> DnState { self.sl().dn }
    pub fn dn_s(&self) -> Region { self.sl().dn_s }
    pub fn dn_conv(&self) -> Region { self.sl().dn_conv }
    pub(crate) fn dn_replay(&self) -> Option<Region> { self.sl().dn_replay }
    pub fn mtp_s(&self) -> &Option<crate::mtp::MtpScratch> { &self.sl().mtp_s }
    /// Pass scratch of the selected slot: the shared rows with this slot's ctrl and rec.
    pub fn io(&self) -> PassIo { let s = self.sl(); PassIo { ctrl: s.ctrl.off, rec: s.rec.off, ban: self.ban_io(self.cur.get()), ..self.s.io } }

    /// Banned ids of slot `slot` for the head (`PassIo::ban`): Some(arena offset) when its list is non-empty.
    pub fn ban_io(&self, slot: usize) -> Option<usize> {
        let r = self.slots[slot].ctrl;
        let n = unsafe { std::ptr::read_volatile(self.arena.ptr(r).add(BAN_OFF) as *const u32) };
        (n > 0).then_some(r.off + BAN_OFF)
    }

    /// Ban `ids` on the selected slot: from the next pass on, every head argmax (decode, prefill's last row, MTP / deep /
    /// DFlash / context-draft verify, batched verify) skips them (`ban_logits`: logit -inf). Empty = no ban (today's
    /// passes, no extra dispatch). Persists across `reset`; the caller clears it. At most `BAN_MAX` ids.
    pub fn set_ban(&self, ids: &[u32]) -> Result<(), String> {
        if ids.len() > BAN_MAX { return Err(format!("ban: {} ids > {}", ids.len(), BAN_MAX)); }
        // the GPU may still read the list of an in-flight pass of this slot: only change it between requests
        self.ahead.borrow_mut().take();
        let p = unsafe { self.arena.ptr(self.sl().ctrl).add(BAN_OFF) as *mut u32 };
        unsafe {
            for (i, &t) in ids.iter().enumerate() { std::ptr::write_volatile(p.add(1 + i), t); }
            std::ptr::write_volatile(p, ids.len() as u32);
        }
        Ok(())
    }

    pub(crate) fn ctrl(&self) -> &mut SdCtrl { unsafe { &mut *(self.arena.ptr(self.sl().ctrl) as *mut SdCtrl) } }
    pub(crate) fn ctrl_read(&self) -> SdCtrl { unsafe { std::ptr::read_volatile(self.arena.ptr(self.sl().ctrl) as *const SdCtrl) } }
    pub(crate) fn stage_ptr(&self, slot: usize) -> *mut u8 { unsafe { self.arena.ptr(self.sl().stage).add(slot * self.slot_bytes) } }
    fn rec_at(&self, p: usize) -> (u32, f32, f32) {
        let r = self.arena.slice_mut::<f32>(self.sl().rec);
        (r[4 * p].to_bits(), r[4 * p + 1], r[4 * p + 2])
    }
    pub(crate) fn is_eos(&self, t: u32) -> bool { self.cfg.eos.contains(&t) }
    /// Position (ctrl.pos) of the next encoded pass.
    pub fn pos(&self) -> usize { self.sl().pos.get() }
    /// P2 (spec.rs): resync the host position with ctrl.pos after an MTP run (nothing in flight).
    pub(crate) fn set_pos(&self, p: usize) { self.sl().pos.set(p); }

    /// Bench only (spec-bench timing): move ctrl.pos, ctrl.mtp_len and the host position back to `p` (nothing in
    /// flight). KV rows >= p are simply overwritten by the next pass; the DeltaNet state is NOT rolled back, so the
    /// following passes compute on an advanced state (fine for timing, not for outputs).
    pub fn rewind_to(&self, p: usize) {
        let c = self.ctrl();
        c.pos = p as u32;
        c.mtp_len = p as u32;
        self.sl().pos.set(p);
    }
    /// rec[p] of the last pass that wrote it: (argmax token, bf16 top-1, top-2).
    pub fn rec_token(&self, p: usize) -> (u32, f32, f32) { self.rec_at(p) }

    /// Start a new sequence: ctrl zeroed (pos 0, role 0), DeltaNet S and conv state zeroed. KV needs no clearing
    /// (attention only reads keys 0..pos + t, all written before). No command buffer may be in flight.
    pub fn reset(&self) {
        self.ahead.borrow_mut().take();
        *self.ctrl() = unsafe { std::mem::zeroed() };
        self.arena.slice_mut::<u8>(self.dn_s()).fill(0);
        self.arena.slice_mut::<u8>(self.dn_conv()).fill(0);
        self.arena.slice_mut::<u8>(self.sl().rec).fill(0);
        self.sl().pos.set(0);
        self.rm_reset();
    }

    /// One pass over `k` rows staged in `stage_slot`: embed_q4 → all layers → (head on the last `head_rows` rows) →
    /// ctrl.pos += k. `ck`: write the DeltaNet rollback checkpoint after row 0 (MTP verify, P2).
    pub fn encode_pass(&self, cb: &CmdBuf, k: usize, stage_slot: usize, head_rows: usize, ck: bool) -> Result<(), String> {
        self.encode_pass_ex(cb, k, stage_slot, head_rows, ck, false)
    }

    /// `encode_pass` + optional `topk32` over the head rows (teacher mode).
    pub fn encode_pass_ex(&self, cb: &CmdBuf, k: usize, stage_slot: usize, head_rows: usize, ck: bool, topk: bool) -> Result<(), String> {
        self.encode_pass_full(cb, k, stage_slot, head_rows, ck, topk, false)
    }

    /// `encode_pass_ex` with `gemm`: force the fast prefill pass (GEMM projections + prefill mixers) for any k,
    /// also k <= 8 (spec-bench verify timing). A fast pass with head_rows == k runs the head on all k rows in blocks
    /// of <= 8 rows (logits hold 8 rows; argmax_b writes rec[pos + t] for every row), topk32 only with `topk`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_pass_full(&self, cb: &CmdBuf, k: usize, stage_slot: usize, head_rows: usize, ck: bool, topk: bool, gemm: bool)
                            -> Result<(), String> {
        // fast prefill pass (plan U25): k > 8 rows, GEMM projections + prefill mixers; head on 0 / 1 rows, or on all
        // k rows in blocks of 8 (teacher with topk, verify timing without)
        let fast = gemm || !EXACT_KS.contains(&k);
        if fast {
            if !self.fast_prefill() || k > self.tmax {
                return Err(format!("pass k {}: fast prefill off or k > {} (exact passes take {:?})", k, self.tmax, EXACT_KS));
            }
            if ck { return Err("fast prefill pass: no checkpoint".into()); }
            if !(head_rows <= 1 || head_rows == k) { return Err(format!("fast pass: head rows {} for k {}", head_rows, k)); }
        } else {
            if k > self.tmax { return Err(format!("pass k {} > {}", k, self.tmax)); }
            if head_rows > k || (head_rows > 0 && !EXACT_KS.contains(&head_rows)) {
                return Err(format!("head rows {} invalid for k {}", head_rows, k));
            }
        }
        if stage_slot > 1 { return Err(format!("stage slot {}", stage_slot)); }
        let pos = self.sl().pos.get();
        if pos + k > self.opt.max_ctx { return Err(format!("position {} + {} > max_ctx {}", pos, k, self.opt.max_ctx)); }
        let c = &self.cfg;
        let io = PassIo { k, pos, nch_max: (pos + k).div_ceil(ATTN_CHUNK).min(self.nch_max), rm: self.rm_io(!fast), ..self.io() };
        let dump = self.dumper.as_ref();
        let e = Enc::new(&self.gpu, cb, &self.buf);
        let r = (|| -> Result<(), String> {
            e.pso("embed_q4").b(0, self.sl().stage.at(stage_slot * self.slot_bytes)).b(1, io.h).u(2, k as u32);
            e.threads((c.hidden / 8, k), 64);
            if let Some(d) = dump { d.capture(&e, "emb", io.h, streamdense_core::DType::BF16, &[k, c.hidden])?; }
            for l in 0..c.layers {
                if fast {
                    encode_layer_prefill(&e, c, &self.w.layers[l], Some(&self.dn()), self.kv_at(l), &io, l)?;
                } else {
                    encode_layer(&e, c, &self.w.layers[l], Some(&self.dn()), self.kv_at(l), &io, ck, dump, l)?;
                }
                if let Some(d) = &self.dflash { d.tap(&e, &io, l); }
                if let Some(d) = &self.dflash2 { d.tap(&e, &io, l); }
            }
            // argmax_b writes rec[ctrl.pos + t] for the head rows: move pos to the first head row first
            let adv = |n: usize| for _ in 0..n { e.pso("advance1").b(0, io.ctrl); e.threads((1, 1), 1); };
            if fast && head_rows > 1 {
                // teacher over a fast pass: head + topk32 per block of <= 8 rows (logits hold 8 rows)
                let hb = c.hidden * 2;
                let mut r0 = 0;
                while r0 < k {
                    let kb = *KS.iter().rev().find(|&&b| b <= k - r0).unwrap();
                    let iob = PassIo { k: kb, h: io.h + r0 * hb, hf: io.hf + r0 * hb, ..io };
                    encode_head(&e, c, &self.w, &iob, kb, None)?;
                    if topk {
                        e.pso("topk32").b(0, io.logits).b(1, self.s.tk_idx.off + r0 * TOPK_MAX * 4)
                            .b(2, self.s.tk_val.off + r0 * TOPK_MAX * 4).b(3, self.s.tk_lse.off + r0 * 4).u(4, c.vocab as u32);
                        e.groups((kb, 1), 1024);
                    }
                    adv(kb);
                    r0 += kb;
                }
            } else if head_rows > 0 {
                adv(k - head_rows);
                encode_head(&e, c, &self.w, &io, head_rows, dump)?;
                if topk {
                    e.pso("topk32").b(0, io.logits).b(1, self.s.tk_idx.off).b(2, self.s.tk_val.off).b(3, self.s.tk_lse.off)
                        .u(4, c.vocab as u32);
                    e.groups((head_rows, 1), 1024);
                    if let Some(d) = dump {
                        use streamdense_core::DType;
                        d.capture(&e, "logits_top32", self.s.tk_val.off, DType::F32, &[head_rows, TOPK_MAX])?;
                        d.capture(&e, "logits_top32.idx", self.s.tk_idx.off, DType::U32, &[head_rows, TOPK_MAX])?;
                        d.capture(&e, "lse", self.s.tk_lse.off, DType::F32, &[head_rows])?;
                    }
                }
                adv(head_rows);
            } else { adv(k); }
            if let Some(d) = &self.dflash { d.absorb(&e, &io, pos, k)?; }
            if let Some(d) = &self.dflash2 { d.absorb(&e, &io, pos, k)?; }
            Ok(())
        })();
        e.end();
        r?;
        self.sl().pos.set(pos + k);
        Ok(())
    }

    /// decode_fused: the encode-ahead applies (no read-mask hooks between passes, no dump).
    fn ahead_on(&self) -> bool { self.opt.decode_fused && self.rm.is_none() && self.dumper.is_none() }

    /// Command buffer of the K = 1 decode pass at the host position from stage `slot` (head on the row; moves the host
    /// position by 1 as `encode_pass`): the one `encode_ahead` made for this position and slot, else encoded now. A
    /// decode pass is fully determined by (position, slot) and the engine, so the early copy is the same command stream.
    pub(crate) fn decode_cb(&self, slot: usize) -> Result<CmdBuf, String> {
        let cached = self.ahead.borrow_mut().take();
        if let Some((p, s, cb)) = cached {
            if p == self.sl().pos.get() && s == slot { self.sl().pos.set(p + 1); return Ok(cb); }
        }
        let cb = self.gpu.command_buffer();
        self.encode_pass(&cb, 1, slot, 1, false)?;
        Ok(cb)
    }

    /// decode_fused: encode the decode pass at the host position from stage `slot` now (while the GPU runs the previous
    /// one) without committing it; the host position stays. No-op when `ahead_on` is false or the context is full.
    pub(crate) fn encode_ahead(&self, slot: usize) -> Result<(), String> {
        let p = self.sl().pos.get();
        if !self.ahead_on() || p + 1 > self.opt.max_ctx { return Ok(()); }
        let cb = self.gpu.command_buffer();
        self.encode_pass(&cb, 1, slot, 1, false)?;
        self.sl().pos.set(p);
        *self.ahead.borrow_mut() = Some((p, slot, cb));
        Ok(())
    }

    /// Fast prefill (plan U25) is used for passes of > 8 rows unless `opt.prefill_exact`; it needs bf16 KV and is
    /// off in dump mode (dumps are stepwise).
    pub fn fast_prefill(&self) -> bool { !self.opt.prefill_exact && (self.opt.kv == KvFormat::Bf16 || self.pf_on.get()) && self.dumper.is_none() }

    /// the KV refs of the main layers for the next pass (bf16 views while a converting prefill runs).
    pub(crate) fn kv_at(&self, l: usize) -> Option<&KvRef> {
        if self.pf_on.get() { if let Some(v) = &self.pf_bf { return v[l].as_ref(); } }
        self.kv()[l].as_ref()
    }

    /// start a converting prefill when the prototype is on and the prompt starts at position 0.
    pub(crate) fn pf_begin(&self) -> bool {
        let on = self.pf_bf.is_some() && self.sl().pos.get() == 0 && self.dumper.is_none();
        self.pf_on.set(on);
        on
    }

    /// after a converting prefill (nothing in flight): KV rows 0..pos of every main attention layer, bf16 ->
    /// q8 in place, with the same arithmetic as `attn_prep_q8` (per 32-dim group: fp16 scale = (max - min) / 255, fp16
    /// bias = min, q = clamp(rint((x - bias) / scale), 0, 255)); one CPU thread per K / V region. The MTP layer's KV
    /// is written by exact (<= 8-row) passes and is already q8.
    pub(crate) fn pf_end(&self) {
        if !self.pf_on.replace(false) { return; }
        let t0 = Instant::now();
        let n = self.sl().pos.get();
        let (hd, nkv) = (self.cfg.head_dim, self.cfg.kv_heads);
        let base = self.arena.base() as usize;
        let mut jobs = vec![];
        for r in self.kv().iter().flatten() { jobs.push((r.k, r.ksb, r.max_ctx)); jobs.push((r.v, r.vsb, r.max_ctx)); }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (next, jobs) = (&next, &jobs);
        std::thread::scope(|sc| {
            for _ in 0..8.min(jobs.len()) {
                sc.spawn(move || loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if j >= jobs.len() { break; }
                    let (vals, sb, mc) = jobs[j];
                    // scales may lie inside the bf16 data still to be read (conv layout): keep them aside, write last
                    let mut sbuf = vec![0u16; 2 * nkv * n * (hd / 32)];
                    unsafe {
                    let src = (base + vals) as *const u16;
                    let dst = (base + vals) as *mut u8;
                    let sbp = (base + sb) as *mut u16;
                    // ascending element order: byte e (q8) only overlaps bf16 elements <= e already read
                    for h in 0..nkv {
                        for p in 0..n {
                            for g in 0..hd / 32 {
                                let e0 = (h * mc + p) * hd + g * 32;
                                let mut x = [0f32; 32];
                                for (i, xi) in x.iter_mut().enumerate() { *xi = streamdense_core::bf16_to_f32(std::ptr::read(src.add(e0 + i))); }
                                let (mut lo, mut hi) = (x[0], x[0]);
                                for &xi in &x[1..] { lo = lo.min(xi); hi = hi.max(xi); }
                                let sc16 = streamdense_core::f16::f32_to_f16((hi - lo).max(1e-8) / 255.0);
                                let lo16 = streamdense_core::f16::f32_to_f16(lo);
                                let scv = streamdense_core::f16::f16_to_f32(sc16);
                                let scf = if scv > 0.0 { scv } else { 1.0 };
                                let lof = streamdense_core::f16::f16_to_f32(lo16);
                                for (i, &xi) in x.iter().enumerate() {
                                    std::ptr::write(dst.add(e0 + i), ((xi - lof) / scf).round_ties_even().clamp(0.0, 255.0) as u8);
                                }
                                let gi = (h * n + p) * (hd / 32) + g;
                                sbuf[2 * gi] = sc16;
                                sbuf[2 * gi + 1] = lo16;
                            }
                        }
                    }
                    for h in 0..nkv {
                        for p in 0..n {
                            for g in 0..hd / 32 {
                                let (gi, gs) = ((h * n + p) * (hd / 32) + g, (h * mc + p) * (hd / 32) + g);
                                std::ptr::write(sbp.add(2 * gs), sbuf[2 * gi]);
                                std::ptr::write(sbp.add(2 * gs + 1), sbuf[2 * gi + 1]);
                            }
                        }
                    }
                    }
                });
            }
        });
        eprintln!("pf_q8conv: converted {} rows x {} regions to q8 in {:.3} s", n, jobs.len(), t0.elapsed().as_secs_f64());
    }

    /// Chunk sizes for n rows. Exact: prefill_k chunks, the remainder in 4 / 2 / 1 (K-invariance keeps it bit-exact).
    /// Fast: chunks of up to prefill_t rows while more than 8 rows remain (GEMM passes), then a tail of <= 8 rows in
    /// K <= 2 GEMV passes (measured: the K = 4 / 8 GEMV is slower per row than K = 2; the 32-row GEMM tile wastes
    /// rows below 9).
    pub(crate) fn chunks(&self, n: usize) -> Vec<(usize, usize)> {
        self.chunks_ex(n, self.fast_prefill())
    }

    /// `chunks` with the mode given (`fast`: the fast prefill's chunks; the streamed prefill fast mode uses them on an
    /// engine whose own `prefill` stays exact).
    pub(crate) fn chunks_ex(&self, n: usize, fast: bool) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        let mut i = 0;
        while i < n {
            if fast && n - i > 8 {
                let k = self.opt.prefill_t.min(n - i);
                v.push((i, k));
                i += k;
                continue;
            }
            let kmax = if fast { self.opt.prefill_k.min(2) } else { self.opt.prefill_k };
            let k = *KS.iter().rev().find(|&&k| k <= kmax && k <= n - i).unwrap();
            v.push((i, k));
            i += k;
        }
        v
    }

    /// Feed `ids` from the current position in K-row chunks (rows of chunk c+1 are staged while c runs); the head
    /// runs on the last row only. Returns the greedy next token (ctrl.tok[0]); rec[pos_last] holds its top-1/top-2.
    pub fn prefill(&self, ids: &[u32]) -> Result<u32, String> {
        if ids.is_empty() { return Err("prefill: no tokens".into()); }
        if self.sl().pos.get() + ids.len() > self.opt.max_ctx {
            return Err(format!("prefill: {} + {} tokens > max_ctx {}", self.sl().pos.get(), ids.len(), self.opt.max_ctx));
        }
        self.rm_prefill_begin(); // read mask: the prompt is always read in full
        if let (Some(d), Some(spec)) = (self.dumper.as_ref(), self.opt.dump.as_ref()) {
            let t = self.prefill_dump(ids, d, &spec.path)?;
            self.rm_prefill_end();
            return Ok(t);
        }
        self.pf_begin();
        let ch = self.chunks(ids.len());
        let mut inflight: VecDeque<CmdBuf> = VecDeque::new();
        // The arena is not hazard-tracked: with two chunk CBs in flight, CB c+1 could start on the GPU before CB c
        // has finished (a race on KV / DeltaNet state; seen as non-deterministic prefill results on the 27B). Each CB
        // waits on the GPU for the previous one's signal; the CPU still stages chunk c+1 while chunk c runs.
        let order = self.gpu.shared_event();
        for (ci, &(i, k)) in ch.iter().enumerate() {
            let slot = ci % 2;
            if inflight.len() >= 2 { wait(&inflight.pop_front().unwrap())?; } // slot free again
            self.reader.read_rows(&ids[i..i + k], self.stage_ptr(slot))?;
            let cb = self.gpu.command_buffer();
            if ci > 0 { cb_wait(&cb, &order, ci as u64); }
            let last = ci + 1 == ch.len();
            if let Err(e) = self.encode_pass(&cb, k, slot, last as usize, false) {
                for cb in &inflight { wait(cb)?; }
                self.pf_on.set(false);
                return Err(e);
            }
            cb_signal(&cb, &order, ci as u64 + 1);
            commit(&cb);
            inflight.push_back(cb);
        }
        for cb in &inflight { wait(cb)?; }
        self.pf_end();
        self.rm_prefill_end();
        Ok(self.ctrl_read().tok[0])
    }

    /// bf16 top-1 − top-2 logit recorded for position `p` (rec table), the unsure-token signal.
    pub(crate) fn rec_gap(&self, p: usize) -> f32 { let (_, a, b) = self.rec_at(p); a - b }

    /// Dump mode: stepwise (k = 1) passes, one capture set per step, merged into `path` at the end. The head and
    /// `topk32` run on every step, so `hf`, `logits_top32` (+ `.idx`) and `lse` are [T, ...] like the MLX ref.
    fn prefill_dump(&self, ids: &[u32], d: &Dumper, path: &str) -> Result<u32, String> {
        let mut parts = Vec::with_capacity(ids.len());
        for (i, &t) in ids.iter().enumerate() {
            d.reset();
            self.reader.read_rows(&[t], self.stage_ptr(0))?;
            let cb = self.gpu.command_buffer();
            self.encode_pass_ex(&cb, 1, 0, 1, false, true)?;
            commit(&cb);
            wait(&cb)?;
            let part = format!("{}.part{}", path, i);
            d.write(&self.arena, &part)?;
            parts.push(part);
        }
        merge_dumps(&parts, path)?;
        Ok(self.ctrl_read().tok[0])
    }

    /// Greedy decode from a fresh state: prefill → first token, then one command buffer per token. `pipe`: the
    /// hand-off thread waits for CB(n)'s event, reads ctrl.tok, preads the embedding row and commits CB(n+1), which
    /// the main thread has already encoded; else synchronous (commit, wait, read, stage). Both are bit-identical.
    pub fn generate(&self, prompt: &[u32], n_new: usize, pipe: bool) -> Result<GenResult, String> {
        let p_len = prompt.len();
        if p_len + n_new > self.opt.max_ctx { return Err(format!("prompt {} + {} new > max_ctx {}", p_len, n_new, self.opt.max_ctx)); }
        let mut r = GenResult::default();
        self.reset();
        if n_new == 0 {
            if p_len > 0 { self.prefill(prompt)?; }
            r.peak = self.peak();
            return Ok(r);
        }
        let tp = Instant::now();
        let t0 = self.prefill(prompt)?;
        r.prefill_s = tp.elapsed().as_secs_f64();
        r.tokens.push(t0);
        if n_new > 1 && !self.is_eos(t0) {
            // CB(j) (j >= 1) reads staging slot j % 2; emb(t0) goes to slot 1
            self.reader.read_rows(&[t0], self.stage_ptr(1))?;
            let steps = n_new - 1;
            if pipe { self.decode_pipelined(steps, &mut r)?; } else { self.decode_sync(steps, &mut r)?; }
        }
        for (i, &t) in r.tokens.iter().enumerate() {
            let (rt, b1, b2) = self.rec_at(p_len - 1 + i);
            if rt != t { return Err(format!("token {}: record {} != emitted {}", i, rt, t)); }
            r.margins.push(b1 - b2);
        }
        r.peak = self.peak();
        Ok(r)
    }

    /// Streaming greedy decode for `qwen serve`: feed `ids` from the CURRENT position (no reset: the caller resets
    /// for a new sequence, or continues a previous one), then decode synchronously, calling `on_tok` per token as it
    /// is produced. `on_tok` returns false to cancel (checked between tokens). Stops on the model's eos, on a token in
    /// `stop`, after `n_new` tokens, or when the next pass would exceed max_ctx. The last emitted token is NOT fed
    /// (it is `last`); a continuation feeds it first. Same passes as `generate(.., pipe = false)`, so same tokens.
    pub fn feed_stream(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool)
        -> Result<StreamStats, String> {
        let mut st = StreamStats { stop_reason: "length", ..Default::default() };
        if ids.is_empty() { return Err("feed_stream: no tokens".into()); }
        if self.sl().pos.get() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok(st); }
        let tp = Instant::now();
        let mut tok = self.prefill(ids)?;
        st.prefill_s = tp.elapsed().as_secs_f64();
        let td = Instant::now();
        loop {
            st.gen += 1;
            st.last = Some(tok);
            if !on_tok(tok) { st.stop_reason = "cancelled"; break; }
            if self.is_eos(tok) || stop.contains(&tok) { st.stop_reason = "eos"; break; }
            if st.gen >= n_new { st.stop_reason = "length"; break; }
            if self.sl().pos.get() + 1 > self.opt.max_ctx { st.stop_reason = "ctx_full"; break; }
            self.reader.read_rows(&[tok], self.stage_ptr(0))?;
            self.rm_before_pass();
            let cb = self.decode_cb(0)?;
            commit(&cb);
            if st.gen + 1 < n_new { self.encode_ahead(0)?; }
            wait(&cb)?;
            self.rm_after_pass(&[self.rec_gap(self.sl().pos.get() - 1)]);
            st.gpu_ms += gpu_ms(&cb);
            tok = self.ctrl_read().tok[0];
        }
        st.decode_s = td.elapsed().as_secs_f64();
        Ok(st)
    }

    fn decode_sync(&self, steps: usize, r: &mut GenResult) -> Result<(), String> {
        let t0 = Instant::now();
        for j in 1..=steps {
            self.rm_before_pass();
            let cb = self.decode_cb(j % 2)?;
            commit(&cb);
            if j < steps { self.encode_ahead((j + 1) % 2)?; }
            wait(&cb)?;
            self.rm_after_pass(&[self.rec_gap(self.sl().pos.get() - 1)]);
            r.gpu_ms.push(gpu_ms(&cb));
            let tok = self.ctrl_read().tok[0];
            r.tokens.push(tok);
            if self.is_eos(tok) || j == steps { break; }
            self.reader.read_rows(&[tok], self.stage_ptr((j + 1) % 2))?;
        }
        r.gen_wall_s = t0.elapsed().as_secs_f64();
        Ok(())
    }

    fn decode_pipelined(&self, steps: usize, r: &mut GenResult) -> Result<(), String> {
        let ev = self.gpu.shared_event(); // fresh per call (an aborted hand-off leaves its events released)
        let enc = |j: u64| -> Result<CmdBuf, String> {
            let cb = self.gpu.command_buffer();
            self.encode_pass(&cb, 1, (j % 2) as usize, 1, false)?;
            cb_signal(&cb, &ev, j);
            Ok(cb)
        };
        let pend = Arc::new(Pending::default());
        let pc = pend.clone();
        let commit_fn: CommitFn = Box::new(move |n| {
            let mut g = pc.m.lock().unwrap();
            loop {
                if let Some(cb) = g.cbs.remove(&n) {
                    commit(&cb.0);
                    g.committed = n;
                    pc.cv.notify_all();
                    return;
                }
                if g.closed { return; }
                g = pc.cv.wait(g).unwrap();
            }
        });
        let policy = HandoffPolicy::new(HandoffKind::Plain, 1, self.tmax, self.cfg.eos.clone(), steps);
        let cb1 = enc(1)?;
        let mut all: Vec<CmdBuf> = vec![cb1.clone()];
        let t0 = Instant::now();
        commit(&cb1);
        pend.m.lock().unwrap().committed = 1;
        let ctrl_ptr = self.arena.ptr(self.sl().ctrl) as *mut SdCtrl;
        let ho = Handoff::spawn(self.reader.clone(), ev.clone(), ctrl_ptr, self.stage_ptr(0), policy, Some(commit_fn));
        let close = |p: &Pending| { let mut g = p.m.lock().unwrap(); g.closed = true; g.cbs.clear(); p.cv.notify_all(); };
        let depth = self.opt.depth.max(1) as u64;
        for j in 2..=steps as u64 {
            {   // encode at most `depth` CBs ahead of the last committed one
                let mut g = pend.m.lock().unwrap();
                while g.committed + depth < j && !ho.is_finished() {
                    g = pend.cv.wait_timeout(g, Duration::from_millis(1)).unwrap().0;
                }
            }
            if ho.is_finished() { break; }
            let cb = match enc(j) {
                Ok(cb) => cb,
                Err(e) => { ho.abort(); close(&pend); let _ = ho.join(); for cb in &all { let _ = wait(cb); } return Err(e); }
            };
            all.push(cb.clone());
            let mut g = pend.m.lock().unwrap();
            g.cbs.insert(j, SendCb(cb));
            pend.cv.notify_all();
        }
        let out = ho.join();
        close(&pend);
        let committed = pend.m.lock().unwrap().committed;
        for (i, cb) in all.iter().enumerate() {
            if (i + 1) as u64 > committed { break; }
            wait(cb)?;
            r.gpu_ms.push(gpu_ms(cb));
        }
        r.gen_wall_s = t0.elapsed().as_secs_f64();
        let out = out?;
        r.tokens.extend_from_slice(&out.tokens);
        r.handoff = Some(out.handoff);
        r.pread = Some(out.pread);
        // the encode-ahead advanced the host position past the committed passes
        self.sl().pos.set(self.sl().pos.get() - (all.len() as u64 - committed) as usize);
        Ok(())
    }

    /// Teacher-forced pass over `ids` from a fresh state: head + topk32 on every row. Returns per position the top
    /// `topk` (<= 32) of the bf16-rounded logits for the token after it, and the logsumexp.
    pub fn teacher(&self, ids: &[u32], topk: usize) -> Result<Vec<TopK>, String> {
        if topk == 0 || topk > TOPK_MAX { return Err(format!("teacher: topk {} not in 1..={}", topk, TOPK_MAX)); }
        if ids.len() > self.opt.max_ctx { return Err(format!("teacher: {} tokens > max_ctx {}", ids.len(), self.opt.max_ctx)); }
        self.reset();
        let mut out = Vec::with_capacity(ids.len());
        for (i, k) in self.chunks(ids.len()) {
            self.reader.read_rows(&ids[i..i + k], self.stage_ptr(0))?;
            let cb = self.gpu.command_buffer();
            self.encode_pass_ex(&cb, k, 0, k, false, true)?;
            commit(&cb);
            wait(&cb)?;
            let (ix, vv, ls) = (self.arena.slice_mut::<u32>(self.s.tk_idx), self.arena.slice_mut::<f32>(self.s.tk_val),
                                self.arena.slice_mut::<f32>(self.s.tk_lse));
            for t in 0..k {
                out.push(TopK { idx: ix[t * TOPK_MAX..t * TOPK_MAX + topk].to_vec(), val: vv[t * TOPK_MAX..t * TOPK_MAX + topk].to_vec(),
                                lse: ls[t] });
            }
        }
        Ok(out)
    }

    pub fn arena_report(&self) -> String {
        format!("arena {:.3} GB used of {:.3} GB: weights {:.3}, KV {:.3}, state {:.3}, scratch {:.3}",
            gb(self.arena.used() as u64), gb(self.arena.size() as u64), gb(self.arena.total("w:") as u64),
            gb(self.arena.total("kv:") as u64), gb(self.arena.total("st:") as u64), gb(self.arena.total("s:") as u64))
    }
    pub fn footprint(&self) -> u64 { footprint::self_footprint() }
    /// Engine monitor peak when it runs, else the current footprint.
    pub fn peak(&self) -> u64 { self.monitor.as_ref().map_or_else(footprint::self_footprint, |m| m.peak()) }
    pub fn heap_mb(&self) -> f64 { alloc::heap_now() as f64 / 1e6 }

    /// usage stats since the last reset (`enabled: false` and no data when the recorder is off). First call also
    /// computes the down column norms (CPU, once). Any thread may use `activity_tap()` instead.
    pub fn activity_snapshot(&self) -> crate::activity::ActivityStats {
        match &self.activity { Some(t) => t.snapshot(), None => crate::activity::ActivityStats::dims(&self.cfg) }
    }
    /// The recorder's dims template (dims, q4 layout, resident / weight bytes; `enabled` = recorder on), no data and no
    /// host copy: what a watcher needs for its dims. Unlike `activity_snapshot` it never computes the down norms
    /// (4.5 MB kept for the life of the tap on the 27B) nor copies the sums (2026-11-01 RAM fix).
    pub fn activity_template(&self) -> crate::activity::ActivityStats {
        match &self.activity { Some(t) => t.template().clone(), None => crate::activity::ActivityStats::dims(&self.cfg) }
    }
    /// Start a new usage window (baseline; no GPU sync needed).
    pub fn activity_reset(&self) { if let Some(t) = &self.activity { t.reset(); } }
    /// Thread-safe recorder handle (gateway endpoint, watcher).
    pub fn activity_tap(&self) -> Option<Arc<crate::activity::ActivityTap>> { self.activity.clone() }
}
