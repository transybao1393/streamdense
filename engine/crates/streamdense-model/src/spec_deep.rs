//! Deep MTP speculative decode: d = 2..=SPEC_DEPTH_MAX chained MTP drafts verified in ONE exact pass of K = d + 1
//! rows, greedy and lossless. Sync loop; one step at position p, ctrl.mtp_len == p:
//!   CB verify: embed_q4 [t1, D1..Dd] (staging slot 0) → copy emb(D1..Dd) to mtp.emb rows 0..d-1 → 64 layers over
//!              K rows with the row-0 DeltaNet checkpoint (ck) and the replay save → final norm + lm_head + argmax_b
//!              on all K rows: rec[p + i] = a_i, the greedy token after row i.
//!   CPU:       j = longest prefix with D_{i+1} == a_i and a_i not eos; emit a_0..a_j (stop / cancel may cut it at
//!              je <= j: the emitted tokens are a_0..a_je, a_je is held, rows 0..je are fed); ctrl.pos = p + je + 1;
//!              rec[p + je + 1 ..] zeroed. je < d: ctrl.role ^= 1 (live copy = state after row 0).
//!   CB replay  (0 < je < d): per DeltaNet layer dn_conv / dn_prep / dn_recur_ck0 over the saved rows 1..je.
//!   CB mtp:    emb(a_je) → mtp.emb row je; MTP rows (hf_q, emb(token q + 1)) for q = p..p+je (1/2/4 splits; the
//!              last split drafts) → D1, mtp_len = new pos.
//!   chain:     D_i (i >= 2) = one MTP step on (hn of the previous step, emb(D_{i-1})) at MTP slot mtp_len + i - 2
//!              (`mtp_chain_step`, the "norm" convention); then mtp_len back to pos, so those speculative MTP
//!              KV slots are overwritten by the next step's real rows.
//! Attention needs no rollback: rows past the accepted ones left K/V at positions >= the new pos, and every later
//! pass writes its own K/V rows before it reads them (row t of a pass sees keys 0 ..= ctrl.pos + t).
//!
//! DeltaNet rollback = checkpoint + REPLAY. A full DeltaNet state copy is ~151 MB (48 layers x 48 v heads x 128 x 128
//! fp32 + conv), so one copy per row does not fit the cap. The existing checkpoint (state after row 0, written by the
//! ck pass into copy role ^ 1) is kept; a later row r needs only its raw in_proj columns [qkv | b | a] (conv_dim +
//! 2*vh bf16 = 20 672 B per DeltaNet layer on the 27B): dn_conv rebuilds the conv window from the checkpoint's conv
//! state, dn_prep the l2-normed q/k and the gates, dn_recur the state, with the very same kernels and per-token code,
//! so the result is bit-identical (K-invariance, deltanet.metal header). Rows 1..d-1 are saved (row 0 is the
//! checkpoint; accepting all d drafts keeps the in-place state, no rollback): (d - 1) x 48 x 20 672 B = 0.99 MB per
//! row, 2.98 MB at d = 4. Checkpointing BEFORE row 0 would save one more row (row 0 is always accepted) and replay it
//! on every rollback; a state ring per row would cost 151 MB per row.
//!
//! Losslessness rests on the K-row pass being bit-identical per row to K = 1 passes (K-invariant GEMV, per-row
//! attention, per-row head): then every emitted token is plain greedy's and the state after a step equals plain
//! decoding's, whatever the drafts were.
use std::time::{Duration, Instant};

use streamdense_metal::{cb_done, cb_signal, commit, ev_value, gpu_ms, gpu_span, wait, CmdBuf, Enc, Event};

use crate::draft_table::DraftTable;
use crate::mtp::{encode_mtp, path_pos, splits, MtpRows, CONF_BYTES};
use crate::qwen_config::{LayerKind, QwenConfig};
use crate::qwen_engine::{KvFormat, QwenEngine, QwenOptions, StreamStats};
use crate::qwen_layers::{encode_head, encode_layer, DnDims, DnState, PassIo, EXACT_KS};
use crate::spec::{generate_mtp, DraftOverride, Oracle, SpecResult};
use crate::weights::Mixer;

const BF: usize = 2;
/// K = d + 1 <= 8: logits / split-K partial scratch hold 8 rows; usable depths are those with d + 1 in EXACT_KS
/// (all of 1..=7).
pub const SPEC_DEPTH_MAX: usize = 7;

/// Bytes of one saved replay row: [qkv | b | a] bf16 for every DeltaNet layer.
pub fn replay_row_bytes(cfg: &QwenConfig) -> usize { cfg.n_deltanet() * (cfg.conv_dim() + 2 * cfg.dn_vh) * BF }

/// Replay buffer for verify passes of up to `depth` drafts: rows 1..depth-1, bf16 [depth - 1][n_deltanet][conv + 2*vh].
pub fn replay_bytes(cfg: &QwenConfig, depth: usize) -> usize {
    if depth >= 2 { (depth - 1) * replay_row_bytes(cfg) } else { 0 }
}

impl QwenEngine {
    /// Depth `d >= 2` usable on this engine (replay rows, verify rows, MTP emb rows).
    pub(crate) fn check_spec_depth(&self, d: usize) -> Result<(), String> {
        self.mtp_parts()?;
        if d < 2 || d > SPEC_DEPTH_MAX { return Err(format!("spec depth {} not in 2..={}", d, SPEC_DEPTH_MAX)); }
        if d > self.opt.spec_depth {
            return Err(format!("spec depth {} > the engine's spec_depth {} (replay buffer sized at open)", d, self.opt.spec_depth));
        }
        if !EXACT_KS.contains(&(d + 1)) { return Err(format!("spec depth {}: no exact pass of {} rows (exact K: {:?})", d, d + 1, EXACT_KS)); }
        if d + 1 > self.tmax { return Err(format!("spec depth {}: {} verify rows > tmax {}", d, d + 1, self.tmax)); }
        Ok(())
    }

    /// Verify pass of k rows at a position in [lo, hi] (exact when lo == hi; a range when encoded ahead, `path_pos`)
    /// (see the module doc). Rows are staged in slot 0. Ends with `spec_conf` over the k logits rows (conf rows 0..k).
    fn encode_verify_deep(&self, e: &Enc, k: usize, lo: usize, hi: usize) -> Result<(), String> {
        let (_, _, ms) = self.mtp_parts()?;
        let c = &self.cfg;
        let hb = c.hidden * BF;
        let save = if k > 2 { Some(self.dn_replay().ok_or("spec: no DeltaNet replay buffer (engine spec_depth 1)")?.off) } else { None };
        let pos = path_pos(lo, hi, k, self.opt.kv == KvFormat::Q8)?;
        let io = PassIo { k, pos, nch_max: self.nch_for(hi, k), dn_save: save, rm: self.rm_io(true), ..self.io() };
        e.pso("embed_q4").b(0, self.sl().stage.at(0)).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        // emb(D1..Dd) = rows 1.. of the verify input, before the layers update h in place (the MTP rows reuse them)
        let nb = (k - 1) * hb;
        if nb > 0 { e.pso("copy_bytes").b(0, io.h + hb).b(1, ms.emb).u(2, nb as u32).threads((nb, 1), 256); }
        for l in 0..c.layers {
            encode_layer(e, c, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, k > 1, None, l)?;
        }
        encode_head(e, c, &self.w, &io, k, None)?;
        e.pso("spec_conf").b(0, io.logits).b(1, ms.conf).u(2, self.w.lm_head.rows as u32).groups((k, 1), 1024);
        Ok(())
    }

    /// DeltaNet replay of saved rows 1..=rows from the live state copy (the row-0 checkpoint after the role flip),
    /// in place. Same kernels and per-token code as the verify pass; the saved rows stand in for p with
    /// prow = layers * w, b at conv, a at conv + vh.
    fn encode_replay(&self, e: &Enc, rows: usize) -> Result<(), String> {
        self.encode_replay_from(e, self.dn_replay().ok_or("spec: no DeltaNet replay buffer")?.off, rows)
    }

    /// `encode_replay` over `rows` saved rows at arena offset `sv` (bf16 [rows][n_deltanet][conv + 2*vh]): the deep spec
    /// replay buffer, or the context-draft hold save (ctx_draft.rs, rows 0..).
    pub(crate) fn encode_replay_from(&self, e: &Enc, sv: usize, rows: usize) -> Result<(), String> {
        let c = &self.cfg;
        let io = &self.io();
        let dd = DnDims::from_cfg(c);
        let (conv, vh) = (c.conv_dim(), c.dn_vh);
        let w = conv + 2 * vh;
        let rd = DnDims { prow: dd.layers * w as u32, b_off: conv as u32, a_off: (conv + vh) as u32, ..dd };
        let mut dn = 0usize;
        for l in 0..c.layers {
            if c.kinds[l] != LayerKind::DeltaNet { continue; }
            let Mixer::Dn(dw) = &self.w.layers[l].mixer else { return Err(format!("spec replay: layer {} has no DeltaNet weights", l)) };
            let p0 = sv + dn * w * BF;
            e.pso("dn_conv").b(0, p0).b(1, dw.conv_w.off).b(2, self.dn().conv).b(3, io.cv).b(4, io.ctrl)
                .u(5, dn as u32).u(6, rows as u32).u(7, u32::MAX).bytes(8, &rd)
                .threads((conv, 1), 256);
            e.pso("dn_prep").b(0, io.cv).b(1, p0).b(2, dw.a_log.off).b(3, dw.dt_bias.off).b(4, io.g).b(5, io.beta)
                .u(6, rows as u32).bytes(7, &rd)
                .groups((2 * c.dn_kh + 1, rows), 128);
            let s_layer = self.dn().s + dn * DnState::s_layer_bytes(c, self.dn().copies);
            e.pso("dn_recur_ck0")
                .b(0, io.cv).b(1, io.g).b(2, io.beta).b(3, s_layer).b(4, io.y).b(5, io.ctrl).u(6, rows as u32).bytes(7, &rd)
                .groups((4 * vh, 1), 128);
            dn += 1;
        }
        Ok(())
    }

    /// MTP rows q = p..p+n-1 of a step (hf rows 0..n-1 of the verify pass, mtp.emb rows 0..n-1 = emb of the token
    /// after each), row n-1's emb from mtp.stage; the last split drafts (ctrl.draft, hn), mtp_len += n.
    fn encode_mtp_rows(&self, e: &Enc, n: usize, p: usize) -> Result<(), String> { self.encode_mtp_rows_hf(e, n, p, self.io().hf) }

    /// `encode_mtp_rows` with the verify pass's hf rows of this sequence at arena offset `hf0` (a batched pass: the
    /// sequence's block of rows, batch.rs).
    pub(crate) fn encode_mtp_rows_hf(&self, e: &Enc, n: usize, p: usize, hf0: usize) -> Result<(), String> {
        let (w, kv, ms) = self.mtp_parts()?;
        let c = &self.cfg;
        let hb = c.hidden * BF;
        let io = self.io();
        e.pso("embed_q4").b(0, ms.stage).b(1, ms.emb + (n - 1) * hb).u(2, 1);
        e.threads((c.hidden / 8, 1), 64);
        let sp = splits(n, self.tmax);
        for (i, &(o, s)) in sp.iter().enumerate() {
            let pos = p + o;
            encode_mtp(e, c, w, &self.w.lm_head, kv, &io, ms, MtpRows::Direct { t: s, emb: ms.emb + o * hb, hf: hf0 + o * hb },
                       pos, self.nch_for(pos, s), i + 1 == sp.len())?;
        }
        Ok(())
    }

    /// ctrl.draft (D1, its hidden in mtp hn) + `depth - 1` chained MTP steps → [D1..Dd], GPU ms of the chain.
    /// ctrl.mtp_len is restored (only real rows stay in the MTP KV).
    fn chain_drafts(&self, depth: usize) -> Result<(Vec<u32>, f64), String> {
        let first = self.ctrl_read().draft;
        let mut ds = Vec::with_capacity(depth);
        ds.push(first);
        let mut g = 0.0;
        if depth > 1 {
            let keep = self.ctrl_read().mtp_len;
            let (hn, _, _, _) = self.mtp_chain_offsets()?;
            let mut d = first;
            for _ in 2..=depth {
                let (nd, ms) = self.mtp_chain_step(d, hn)?;
                g += ms;
                ds.push(nd);
                d = nd;
            }
            self.ctrl().mtp_len = keep;
        }
        Ok((ds, g))
    }

    pub(crate) fn run_cb(&self, f: &dyn Fn(&Enc) -> Result<(), String>) -> Result<f64, String> {
        let cb = self.gpu.command_buffer();
        let e = Enc::new(&self.gpu, &cb, &self.buf);
        let r = f(&e);
        e.end();
        r?;
        commit(&cb);
        wait(&cb)?;
        Ok(gpu_ms(&cb))
    }

    /// Sync deep step loop from the state `feed_stream_spec` leaves (t1 emitted and counted in `st.gen`, D1 in
    /// ctrl.draft, mtp_len == pos). Same stop rules as `spec_loop_sync`; on exit the last emitted token is not fed
    /// and the MTP row of the last fed position is fed with its emb, so a later feed can continue.
    /// `oracle` (test hook): replaces draft i of the step at position p by seq[p + 1 + i] (each with probability p).
    #[allow(clippy::too_many_arguments)]
    fn spec_loop_deep(&self, depth: usize, n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                      st: &mut StreamStats, r: &mut SpecResult, oracle: &mut Option<Oracle>) -> Result<(), String> {
        let ov = |ds: &mut Vec<u32>, p: usize, o: &mut Option<Oracle>| if let Some(o) = o { for (i, d) in ds.iter_mut().enumerate() { *d = o.apply(p + 1 + i, *d); } };
        let (_, _, ms) = self.mtp_parts()?;
        let k = depth + 1;
        let mstage = unsafe { self.arena.base().add(ms.stage) };
        r.accept_hist = vec![0; depth + 1];
        let mut t1 = st.last.ok_or("spec: no first token")?;
        if self.pos() + k > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok(()); }
        let (mut drafts, mut gd) = self.chain_drafts(depth)?;
        ov(&mut drafts, self.pos(), oracle);
        loop {
            let p = self.pos();
            let mut rows = Vec::with_capacity(k);
            rows.push(t1);
            rows.extend_from_slice(&drafts);
            self.reader.read_rows(&rows, self.stage_ptr(0))?;
            self.rm_before_pass();
            let gv = self.run_cb(&|e| self.encode_verify_deep(e, k, p, p))?;
            r.verify_ms.push(gv);
            let a: Vec<u32> = (0..k).map(|i| self.rec_token(p + i).0).collect();
            let mut j = 0;
            while j < depth && a[j] == drafts[j] && !self.is_eos(a[j]) { j += 1; }
            r.passes += 1;
            r.straddle.note(p, k, self.opt.kv == KvFormat::Q8);
            r.accepted += j;
            if j < depth { r.rejects += 1; }
            r.accept_hist[j] += 1;
            r.drafts.push(drafts[0]);
            r.draft_pos.push(p as u32);
            let mut why: Option<&'static str> = None;
            let mut used = 0;
            for &t in &a[..=j] {
                used += 1;
                st.gen += 1;
                st.last = Some(t);
                r.tokens.push(t);
                if !on_tok(t) { why = Some("cancelled"); break; }
                if self.is_eos(t) || stop.contains(&t) { why = Some("eos"); break; }
                if st.gen >= n_new { why = Some("length"); break; }
            }
            let je = used - 1;
            let held = a[je];
            let gaps: Vec<f32> = (0..=je).map(|i| self.rec_gap(p + i)).collect();
            self.rm_after_pass(&gaps);
            {   // rows 0..je fed; the pass's later rows are dropped (their K/V and rec entries are stale)
                let cw = self.ctrl();
                cw.pos = (p + je + 1) as u32;
                if je < depth { cw.role ^= 1; }
            }
            self.arena.slice_mut::<f32>(self.sl().rec)[4 * (p + je + 1)..4 * (p + k)].fill(0.0);
            self.set_pos(p + je + 1);
            let mut gr = 0.0;
            if je >= 1 && je < depth {
                gr = self.run_cb(&|e| self.encode_replay(e, je))?;
                r.replay.push((je, gr));
            }
            self.reader.read_rows(&[held], mstage)?;
            gd += self.run_cb(&|e| self.encode_mtp_rows(e, je + 1, p))?;
            if why.is_none() && self.pos() + k > self.opt.max_ctx { why = Some("ctx_full"); }
            if let Some(w) = why {
                r.draft_ms.push(gd);
                r.gpu_ms.push(gv + gr + gd);
                st.stop_reason = w;
                break;
            }
            let (mut ds, gc) = self.chain_drafts(depth)?;
            ov(&mut ds, self.pos(), oracle);
            gd += gc;
            r.draft_ms.push(gd);
            r.gpu_ms.push(gv + gr + gd);
            gd = 0.0;
            drafts = ds;
            t1 = held;
        }
        Ok(())
    }

    /// Streaming deep MTP decode (`feed_stream_mtp` contract) with `depth` drafts per verify pass; depth 1 = the
    /// draft 1 / verify 2 loop of spec.rs. Verify passes need pos + depth + 1 <= max_ctx ("ctx_full" otherwise).
    pub fn feed_stream_spec(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool, depth: usize)
                            -> Result<(StreamStats, SpecResult), String> {
        self.feed_stream_spec_ov(ids, n_new, stop, on_tok, depth, &DraftOverride::None)
    }

    /// `feed_stream_spec` with a draft override (test hook; the oracle's `seq` is indexed by absolute position): the
    /// pipelined loop at fixed depth (`feed_stream_spec_mode`).
    pub fn feed_stream_spec_ov(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool, depth: usize,
                               ov: &DraftOverride) -> Result<(StreamStats, SpecResult), String> {
        self.feed_stream_spec_mode(ids, n_new, stop, on_tok, SpecMode::Fixed(depth.max(1)), ov)
    }

    /// The synchronous reference loop (`spec_loop_deep`): same tokens and exit state as the pipelined one.
    pub fn feed_stream_spec_sync(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool, depth: usize,
                                 ov: &DraftOverride) -> Result<(StreamStats, SpecResult), String> {
        if depth <= 1 { return self.feed_stream_mtp_ov(ids, n_new, stop, on_tok, ov); }
        self.check_spec_depth(depth)?;
        let mut st = StreamStats { stop_reason: "", ..Default::default() };
        let mut r = SpecResult { depth, ..Default::default() };
        if ids.is_empty() { return Err("feed_stream_spec: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok((st, r)); }
        let tp = Instant::now();
        let (t1, hf) = self.prefill_mtp(ids)?;
        self.mtp_first_draft(t1, hf, 0)?;
        st.prefill_s = tp.elapsed().as_secs_f64();
        let td = Instant::now();
        st.gen = 1;
        st.last = Some(t1);
        r.tokens.push(t1);
        if !on_tok(t1) { st.stop_reason = "cancelled"; }
        else if self.is_eos(t1) || stop.contains(&t1) { st.stop_reason = "eos"; }
        else if n_new <= 1 { st.stop_reason = "length"; }
        else {
            let mut oracle = Oracle::new(ov);
            self.spec_loop_deep(depth, n_new, stop, on_tok, &mut st, &mut r, &mut oracle)?;
            r.overridden += oracle.as_ref().map_or(0, |o| o.used);
        }
        if st.stop_reason.is_empty() { st.stop_reason = "length"; }
        st.decode_s = td.elapsed().as_secs_f64();
        st.gpu_ms = r.gpu_ms.iter().sum();
        r.wall = st.decode_s;
        r.prefill_s = st.prefill_s;
        r.finish();
        Ok((st, r))
    }
}

/// Greedy deep MTP generation from a fresh state: `depth` chained drafts verified per pass (depth 1 = `generate_mtp`
/// sync). Tokens must equal `QwenEngine::generate` bit-exactly.
pub fn generate_spec(eng: &QwenEngine, prompt: &[u32], n_new: usize, depth: usize) -> Result<SpecResult, String> {
    generate_spec_ov(eng, prompt, n_new, depth, DraftOverride::None)
}

/// `generate_spec` with a draft override (test hook, as `generate_mtp`): the pipelined loop at fixed depth.
pub fn generate_spec_ov(eng: &QwenEngine, prompt: &[u32], n_new: usize, depth: usize, ov: DraftOverride) -> Result<SpecResult, String> {
    generate_spec_mode(eng, prompt, n_new, SpecMode::Fixed(depth.max(1)), ov)
}

// ================================================================================================================
// Adaptive depth + pipelined rounds (`SpecMode`, `spec_loop_pipe`).
//
// Round at position p (t1 held in stage slot 0 row 0, D1 from the previous round's MTP rows, mtp_len == p):
//   drafts  Auto: one `DraftSource::propose` per round (context tail ending in t1). Table-first when its expected
//           tokens per ms beat the MTP chain's; else the MTP chain, one pre-encoded command buffer per step
//           (`encode_chain_step` + `spec_conf` → the step's top-1 prob), stopped by the planner (below); a table run
//           that agrees with the chain may extend it for free (no MTP steps). Fixed(d): d MTP drafts, no table.
//           Each draft is pread into stage slot 0 row i as soon as it is known (chain step i + 1 embeds it there).
//   verify  K = rows pass encoded AHEAD (during the previous verify, position range [p_prev + 1, p_prev + K_prev]).
//           If the round ends with fewer rows and the cost table says the extra rows are ~free (<= PAD_MS), the
//           spare rows are padded (table tokens, else the last row's token: still exact drafts, just unlikely);
//           otherwise the pass is re-encoded (CPU time on the critical path, `reencode_ms`).
//   overlap while the verify runs (~117 ms) the CPU encodes the MTP-rows command buffer for every possible outcome
//           je = 0..K-1 (replay je rows + MTP rows + D1 + spec_conf) and the next round's verify; while the MTP-rows
//           buffer runs it encodes the next round's chain steps. After each GPU event the CPU only decides, preads
//           one embedding row and commits an encoded buffer, so GPU idle per hand-off = wake-up + pread + commit.
//           Drafting cannot overlap the verify itself: D1 needs the verify's final hidden of the last accepted row
//           and the bonus token's embedding (SSD pread of a GPU-chosen id).
// Planner (deterministic: only GPU-computed confidences and acceptance history, never timings):
//   q_s (P(draft s accepted | drafts < s accepted)) = MTP: 0.5 cal_m[bin(top-1 prob)] + 0.5 ewma_s (each term at the
//   weight of its observations, up to N_FULL, the rest from the prior blend: `Plan::q_m`); table: 0.5
//   cal_t[bin(conditional conf)] + 0.5 conf. E(d) = 1 + Σ_{i<=d} Π_{s<=i} q_s, T(d) = round_ms + pass_ms[d] (K = d+1)
//   + mtp_ms * chain steps + attention(K, N); choose d = argmax E/T (d = 0: a one-row pass, no drafts).
//   Context-aware attention term: every verify row reads the whole KV, so a K-row pass costs ~K x one row's
//   attention (split-K above 8192 keys: K = 8 ~7x K = 1). attention(K, N) = N x attn_layers x kv scale x
//   us_per_key[K] (+ one MTP-layer row per chain step and per round); negligible at short contexts, it makes the
//   planner pick shallower passes (down to d = 0) as N grows. SD_SPEC_CTX_COST=0 drops the term (the previous
//   policy; bench comparison only). Speed only: the output is the verify argmax either way. During the chain, future q use ewma; the chain continues while the
//   argmax lies beyond the drafts computed so far and Π q >= MIN_PREFIX. ewma / cal update after each verify up to
//   the first rejected draft (conditional rates).
// Every emitted token is the verify pass's argmax, as in `spec_loop_deep`: lossless whatever the drafts.

/// Default RAM budget of the draft table (`QwenOptions.draft_table_bytes`).
pub const DRAFT_TABLE_BYTES: usize = 8 << 20;
/// Spare verify rows cost at most this (ms, from the cost table) → pad instead of re-encoding the pass.
const PAD_MS: f64 = 1.5;
/// Stop the chain when the predicted probability that all drafts so far are accepted falls below this.
pub(crate) const MIN_PREFIX: f64 = 0.03;
const NBIN: usize = 5;
const EW_ALPHA: f64 = 0.15;
/// Observations after which a calibration bin / a depth's EWMA counts fully in `Plan::q_m` (before: blended with
/// its prior in proportion).
const N_FULL: u32 = 8;
pub(crate) const TAIL: usize = 16;

/// Speculative depth policy of a generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecMode {
    /// d chained MTP drafts per verify pass (1 = spec.rs draft 1 / verify 2)
    Fixed(usize),
    /// 0..=max drafts per pass, chosen per round (needs max <= the engine's spec_depth)
    Auto { max: usize },
}
impl SpecMode {
    /// "auto" | N
    pub fn parse(s: &str, max: usize) -> Result<SpecMode, String> {
        if s == "auto" { return Ok(SpecMode::Auto { max }); }
        s.parse().map(SpecMode::Fixed).map_err(|_| format!("spec depth {:?}: want auto or 1..={}", s, SPEC_DEPTH_MAX))
    }
    pub fn max_depth(&self) -> usize { match *self { SpecMode::Fixed(d) => d, SpecMode::Auto { max } => max } }
    pub fn name(&self) -> String { match *self { SpecMode::Fixed(d) => d.to_string(), SpecMode::Auto { max } => format!("auto{}", max) } }
}

/// Cost model of the adaptive planner. Defaults: measured synthetic full-pass GEMV ms (mma family) for
/// K = 1..8, the measured chained MTP step GPU median, and a fixed per-round part (replay + MTP rows + hand-offs);
/// plus the attention cost per key of a K-row pass (`attn_us_key*`, below).
#[derive(Clone, Debug, PartialEq)]
pub struct SpecCost {
    pub pass_ms: [f64; 8],
    pub mtp_ms: f64,
    pub round_ms: f64,
    /// µs per KV key per attention layer of a K-row pass (K = 1..8), bf16 split-K path (`attn_split_rows` +
    /// `attn_reduce_gate`: rows with N > 8192).
    pub attn_us_key: [f64; 8],
    /// Same for q8 KV (every row split-K, `attn_split_k*_q8` / GQA variants).
    pub attn_us_key_q8: [f64; 8],
    /// Same for the single / two-pass sdpa rows (bf16 KV, N <= 8192).
    pub attn_us_key_short: [f64; 8],
}
/// Attention cost constants (`SpecCost::attn_us_key*`): GPU median of one layer's attention path (`attn_sdpa_rows`)
/// for a K-row pass at 27B attention shapes (24 q heads, 4 kv heads, head dim 256, bf16 KV) divided by N, on an
/// M4 Pro: `verify_bench --attn-pass` (split-K at N = 32760, mean of the default and verify_mixers rows, linear in N
/// within 4% from 8K to 32K; short path at N = 8000, default kernels). Other shapes are scaled by kv_dim / 1024.
/// Split-K = `attn_split_rows` (K = 5 / 6 and 7 / 8 cost about the same: two score passes, and 3 / 2-row value
/// sweeps). Re-measured 2026-11-16 after the q8 split-K kernel (u8 conversion) and the GQA-shared K = 1 / 2 kernels:
/// bf16 = mean of default + verify_mixers rows, q8 = the "q8 default" row, two runs at N = 32768 each (12288: within
/// 10%); bf16 K = 1 / 2 fell from 0.0305 / 0.0451, every q8 K from 0.0475 .. 0.3343 (the old `attn_partial`
/// estimate, 2.4-3x too steep: the planner over-charged deep passes at long context with q8 KV). Short path
/// unchanged (N = 8000 re-measured within 6%).
const ATTN_US_KEY_SPLIT: [f64; 8] = [0.0240, 0.0342, 0.0557, 0.0622, 0.1031, 0.1019, 0.1280, 0.1220];
const ATTN_US_KEY_SPLIT_Q8: [f64; 8] = [0.0199, 0.0351, 0.0537, 0.0607, 0.1046, 0.1020, 0.1338, 0.1218];
const ATTN_US_KEY_SHORT: [f64; 8] = [0.0443, 0.0618, 0.0954, 0.1135, 0.1534, 0.1709, 0.2059, 0.2250];
impl Default for SpecCost {
    fn default() -> Self {
        SpecCost { pass_ms: [63.2, 76.4, 97.5, 114.7, 115.2, 115.9, 116.4, 116.6], mtp_ms: 4.2, round_ms: 7.0,
                   attn_us_key: ATTN_US_KEY_SPLIT, attn_us_key_q8: ATTN_US_KEY_SPLIT_Q8, attn_us_key_short: ATTN_US_KEY_SHORT }
    }
}
impl SpecCost {
    /// Defaults with the fused verify matmuls (`QwenOptions.verify_fused`): pass ms for K >= 3 minus the matmul-pass
    /// savings measured on synthetic 27B shapes (verify_bench, 2026-11-03: K = 3 106.0 -> 94.2, K = 4 121.0 -> 90.2,
    /// K = 8 122.4 -> 94.0; K = 5..7 taken as K = 8). Estimate; only steers the planner, never the output.
    /// Not applied by default (27B 12 GB mode: over-drafting, -10% measured); see QwenEngine::open.
    pub fn verify_fused() -> SpecCost {
        let mut c = SpecCost::default();
        let save = [0.0, 0.0, 11.8, 30.8, 28.4, 28.4, 28.4, 28.4];
        for k in 0..8 { c.pass_ms[k] -= save[k]; }
        c
    }
    /// "t1,..,t8[;mtp=MS][;round=MS]" (fewer than 8 pass values: the last one repeats).
    pub fn parse(s: &str) -> Result<SpecCost, String> {
        let mut c = SpecCost::default();
        for (i, part) in s.split(';').map(str::trim).filter(|x| !x.is_empty()).enumerate() {
            let num = |v: &str| v.trim().parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0).ok_or(format!("spec cost {:?}: bad number {:?}", s, v));
            if let Some(v) = part.strip_prefix("mtp=") { c.mtp_ms = num(v)?; }
            else if let Some(v) = part.strip_prefix("round=") { c.round_ms = num(v)?; }
            else if i == 0 {
                let v: Vec<f64> = part.split(',').map(num).collect::<Result<_, _>>()?;
                if v.is_empty() || v.len() > 8 { return Err(format!("spec cost {:?}: want 1..8 pass ms", s)); }
                for k in 0..8 { c.pass_ms[k] = v[k.min(v.len() - 1)]; }
            } else { return Err(format!("spec cost {:?}: unknown part {:?}", s, part)); }
        }
        Ok(c)
    }
}

/// Extra draft source of the adaptive loop (`DraftTable`; tests plug stubs via `QwenEngine::set_draft_source`).
/// Same contract as draft_table.rs: `propose` = next tokens after `context_tail` with a running-product confidence;
/// `observe(tail, accepted drafts, verifier top-3 rows 0..=accepted.len())` after each verify with the same tail.
pub trait DraftSource {
    fn propose(&mut self, context_tail: &[u32], max: usize) -> Vec<(u32, f32)>;
    fn observe(&mut self, context_tail: &[u32], accepted: &[u32], verifier_top3: &[[u32; 3]]);
    fn begin_sequence(&mut self, prompt: &[u32]);
    fn flush(&mut self);
    /// `begin_sequence` of a feed that starts at KV position `pos` (`seq` = the fed tokens + the first output token);
    /// sources that index the whole KV (ctx_draft.rs `CtxSource`) keep their history up to `pos`.
    fn begin_at(&mut self, pos: usize, seq: &[u32]) { let _ = pos; self.begin_sequence(seq) }
    /// The last `propose` returned the context lookup's run (ctx_draft.rs `CtxSource`), not a table run. Stats only.
    fn last_from_lookup(&self) -> bool { false }
}
impl DraftSource for DraftTable {
    fn propose(&mut self, t: &[u32], max: usize) -> Vec<(u32, f32)> { DraftTable::propose(self, t, max) }
    fn observe(&mut self, t: &[u32], a: &[u32], v: &[[u32; 3]]) { DraftTable::observe(self, t, a, v) }
    fn begin_sequence(&mut self, p: &[u32]) { DraftTable::begin_sequence(self, p) }
    fn flush(&mut self) { DraftTable::flush(self) }
}

/// `QwenOptions.draft_table` → the engine's draft source ("~/" expanded). A table that fails to open is reported
/// and left off (drafting only: the engine still works).
pub(crate) fn open_draft_table(opt: &QwenOptions) -> Result<Option<Box<dyn DraftSource>>, String> {
    let Some(d) = &opt.draft_table else { return Ok(None) };
    let s = d.to_string_lossy();
    let dir = match (s.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(h)) => std::path::PathBuf::from(h).join(rest),
        _ => d.clone(),
    };
    match DraftTable::open(&dir, opt.draft_table_bytes) {
        Ok(t) => { eprintln!("draft table: {} ({:.1} MB RAM)", dir.display(), t.ram_bytes() as f64 / 1e6); Ok(Some(Box::new(t))) }
        Err(e) => { eprintln!("draft table: {}: {} (off)", dir.display(), e); Ok(None) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Src { Mtp, Table, Pad }

fn bin(p: f64) -> usize { ((p.clamp(0.0, 1.0) * NBIN as f64) as usize).min(NBIN - 1) }

/// The adaptive planner (see the section header).
pub(crate) struct Plan {
    pub(crate) cost: SpecCost,
    ew: [f64; SPEC_DEPTH_MAX],
    cal_m: [f64; NBIN],
    /// observations so far of each ew depth / cal_m bin (saturating at N_FULL)
    n_ew: [u32; SPEC_DEPTH_MAX],
    n_cal: [u32; NBIN],
    cal_t: [f64; NBIN],
    /// attention layers of the main model x kv scale (0 = no attention term: the context-blind policy)
    attn_w: f64,
    /// kv scale alone (the MTP layer: one attention layer)
    attn_mtp: f64,
    /// every row takes the split-K path (q8 KV)
    split_all: bool,
    /// keys before the pass (the position p of the round)
    pub(crate) n: usize,
}
impl Plan {
    pub(crate) fn new(cost: SpecCost) -> Plan {
        let c = [0.1, 0.3, 0.5, 0.7, 0.9];
        Plan { cost, ew: [0.8; SPEC_DEPTH_MAX], cal_m: c, n_ew: [0; SPEC_DEPTH_MAX], n_cal: [0; NBIN], cal_t: c, attn_w: 0.0, attn_mtp: 0.0,
               split_all: false, n: 0 }
    }
    /// Context-aware planner: `attn_layers` attention layers with `kv_dim` (27B: 16, 1024), q8 KV or not.
    pub(crate) fn with_attn(mut self, attn_layers: usize, kv_dim: usize, q8: bool) -> Plan {
        let sc = kv_dim as f64 / 1024.0;
        self.attn_w = attn_layers as f64 * sc;
        self.attn_mtp = sc;
        self.split_all = q8;
        self
    }
    /// ms of one layer's attention (kv scale 1) for a K-row pass at the current n.
    fn attn1(&self, k: usize) -> f64 {
        let k = k.clamp(1, 8);
        let lim = crate::qwen_layers::sdpa_vec_max_n().min(8192);
        let tab = if self.split_all { &self.cost.attn_us_key_q8 }
                  else if self.n + k > lim { &self.cost.attn_us_key } else { &self.cost.attn_us_key_short };
        tab[k - 1] * (self.n + k) as f64 * 1e-3
    }
    /// ms of a K-row verify pass at the current n (matmuls + main-model attention).
    pub(crate) fn pass(&self, k: usize) -> f64 { self.cost.pass_ms[k.clamp(1, 8) - 1] + self.attn_w * self.attn1(k) }
    /// MTP draft s (1-based), top-1 prob p if known: 0.5 cal_m[bin(p)] + 0.5 ew[s], each term taken at the weight of
    /// its evidence (observations / N_FULL) and the rest from the prior blend. A bin without data does not drag a
    /// well-measured high acceptance down (copy-heavy text: the chain otherwise stops at the first draft whose MTP
    /// probability lands in a not-yet-seen low bin), and vice versa; with both measured it is the plain 50 / 50 blend.
    pub(crate) fn q_m(&self, s: usize, p: Option<f64>) -> f64 {
        let i = (s - 1).min(SPEC_DEPTH_MAX - 1);
        let e = self.ew[i];
        let Some(p) = p else { return e };
        let b = bin(p);
        let c = self.cal_m[b];
        let (wc, we) = (0.5 * self.n_cal[b] as f64 / N_FULL as f64, 0.5 * self.n_ew[i] as f64 / N_FULL as f64);
        wc * c + we * e + (1.0 - wc - we) * (0.5 * c + 0.5 * e)
    }
    pub(crate) fn q_t(&self, c: f64) -> f64 { 0.5 * self.cal_t[bin(c)] + 0.5 * c.clamp(0.0, 1.0) }
    pub(crate) fn t(&self, d: usize, steps: usize) -> f64 {
        // MTP layer attention: one row per chain step + the round's MTP rows (counted as one)
        let mtp_attn = if self.attn_mtp > 0.0 { self.attn_mtp * self.attn1(1) * (steps + 1) as f64 } else { 0.0 };
        self.cost.round_ms + self.pass(d.min(7) + 1) + self.cost.mtp_ms * steps as f64 + mtp_attn
    }
    /// argmax over d in 0..=hi (hi <= q.len()) of E(d) / T(d, steps(d)); (d, ratio).
    pub(crate) fn best(&self, q: &[f64], hi: usize, steps: &dyn Fn(usize) -> usize) -> (usize, f64) {
        let (mut e, mut pr) = (1.0, 1.0);
        let mut b = (0, e / self.t(0, steps(0)));
        for d in 1..=hi.min(q.len()) {
            pr *= q[d - 1];
            e += pr;
            let r = e / self.t(d, steps(d));
            if r > b.1 + 1e-12 { b = (d, r); }
        }
        b
    }
    /// Chain decision after s = qs.len() MTP drafts (q of each known): run one more chain step when the best depth
    /// over 0..=cap (later drafts at their ew, s - 1 steps already sunk) lies beyond s and Π q >= MIN_PREFIX.
    pub(crate) fn go_on(&self, qs: &[f64], cap: usize) -> bool {
        let s = qs.len();
        let mut qq = qs.to_vec();
        for t in s + 1..=cap { qq.push(self.q_m(t, None)); }
        let pre: f64 = qs.iter().product();
        let (b, _) = self.best(&qq, cap, &|d| d.max(s) - 1);
        b > s && pre >= MIN_PREFIX
    }
    /// After a verify: drafts (source, MTP prob or table conditional conf) with j accepted.
    pub(crate) fn update(&mut self, rows: &[(Src, f64)], j: usize) {
        for (i, &(src, c)) in rows.iter().enumerate().take(j + 1) {
            let ok = if i < j { 1.0 } else { 0.0 };
            match src {
                Src::Mtp => {
                    let s = i.min(SPEC_DEPTH_MAX - 1);
                    self.ew[s] += EW_ALPHA * (ok - self.ew[s]);
                    self.n_ew[s] = (self.n_ew[s] + 1).min(N_FULL);
                    if c >= 0.0 {
                        let b = bin(c);
                        self.cal_m[b] += EW_ALPHA * (ok - self.cal_m[b]);
                        self.n_cal[b] = (self.n_cal[b] + 1).min(N_FULL);
                    }
                }
                Src::Table => { let b = bin(c); self.cal_t[b] += EW_ALPHA * (ok - self.cal_t[b]); }
                Src::Pad => {}
            }
        }
    }
}

/// Wait for `ev` >= v (GPU signal at the end of `cb`): spin, yield, then 20 µs sleeps; errors if `cb` finished
/// without it (GPU error) or after 60 s.
fn wait_sig(ev: &Event, v: u64, cb: &CmdBuf) -> Result<(), String> {
    let t0 = Instant::now();
    let mut i = 0u32;
    loop {
        if ev_value(ev) >= v { std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire); return Ok(()); }
        i += 1;
        if i % 64 == 0 && cb_done(cb) {
            if ev_value(ev) >= v { return Ok(()); }
            wait(cb)?;
            return Err("spec: command buffer finished without its signal".into());
        }
        if i < 2000 { std::hint::spin_loop(); }
        else if i < 4000 { std::thread::yield_now(); }
        else {
            if t0.elapsed() > Duration::from_secs(60) { return Err(format!("spec: timeout waiting for GPU signal {}", v)); }
            std::thread::sleep(Duration::from_micros(20));
        }
    }
}

/// Committed command buffers of a generation: signal values and GPU idle accounting.
struct Queue { ev: Event, n: u64, last_end: f64, round: Vec<CmdBuf> }
impl Queue {
    /// Signal + commit an encoded buffer, return its signal value.
    fn go(&mut self, cb: &CmdBuf) -> u64 {
        self.n += 1;
        cb_signal(cb, &self.ev, self.n);
        commit(cb);
        self.round.push(cb.clone());
        self.n
    }
    fn run(&mut self, cb: &CmdBuf) -> Result<(), String> { let v = self.go(cb); wait_sig(&self.ev, v, cb) }
    /// Finish the round's buffers: (GPU busy ms, GPU idle ms between them incl. the gap before the first).
    fn close_round(&mut self) -> Result<(f64, f64), String> {
        let (mut busy, mut idle) = (0.0, 0.0);
        for cb in self.round.drain(..) {
            wait(&cb)?;
            let (s, e) = gpu_span(&cb);
            if self.last_end > 0.0 && s > self.last_end { idle += (s - self.last_end) * 1e3; }
            busy += (e - s) * 1e3;
            self.last_end = self.last_end.max(e);
        }
        Ok((busy, idle))
    }
}

pub(crate) fn conf_row(eng: &QwenEngine, r: usize) -> ([u32; 3], [f32; 3], f32) {
    let ms = eng.mtp_s().as_ref().unwrap();
    let f = unsafe { std::slice::from_raw_parts(eng.arena.base().add(ms.conf + r * CONF_BYTES) as *const f32, 8) };
    ([f[0].to_bits(), f[1].to_bits(), f[2].to_bits()], [f[3], f[4], f[5]], f[6])
}
/// Top-1 probability of the last MTP draft (conf row 8).
pub(crate) fn mtp_prob(eng: &QwenEngine) -> f64 {
    let (_, v, lse) = conf_row(eng, 8);
    if v[0].is_finite() && lse.is_finite() { ((v[0] - lse) as f64).exp().clamp(0.0, 1.0) } else { 0.0 }
}

impl QwenEngine {
    /// Replace the adaptive loop's extra draft source (None = off); returns the previous one.
    pub fn set_draft_source(&self, s: Option<Box<dyn DraftSource>>) -> Option<Box<dyn DraftSource>> { self.draft_src.replace(s) }
    /// Flush the draft source to disk (between generations).
    pub fn flush_draft_source(&self) { if let Some(s) = self.draft_src.borrow_mut().as_mut() { s.flush(); } }
    /// The adaptive planner of a generation: `opt.spec_cost` + the context-aware attention term (off with
    /// SD_SPEC_CTX_COST=0, the context-blind policy, for bench comparisons).
    pub(crate) fn new_plan(&self) -> Plan {
        let p = Plan::new(self.opt.spec_cost.clone());
        if std::env::var("SD_SPEC_CTX_COST").is_ok_and(|v| v == "0") { return p; }
        p.with_attn(self.cfg.n_attention(), self.cfg.kv_dim(), self.opt.kv != KvFormat::Bf16)
    }
    /// The engine's default spec policy: Auto { max: spec_depth } with opt.spec_auto, else Fixed(spec_depth).
    pub fn spec_mode(&self) -> SpecMode {
        if self.opt.spec_auto { SpecMode::Auto { max: self.opt.spec_depth } } else { SpecMode::Fixed(self.opt.spec_depth) }
    }

    /// Encode one command buffer inside its own autorelease pool: `commandBuffer` / `computeCommandEncoder` return
    /// autoreleased objects (the encoder retains its buffer), and a thread without a pool drains only at exit, so
    /// the encoded-ahead buffers that are never committed (spare verify Ks, unused outcomes) would otherwise stay
    /// alive and exhaust the queue's 64-buffer limit -> `commandBuffer` blocks forever.
    pub(crate) fn enc_cb(&self, f: &dyn Fn(&Enc) -> Result<(), String>) -> Result<CmdBuf, String> {
        objc2::rc::autoreleasepool(|_| {
            let cb = self.gpu.command_buffer();
            let e = Enc::new(&self.gpu, &cb, &self.buf);
            let r = f(&e);
            e.end();
            r.map(|_| cb)
        })
    }
    pub(crate) fn stage_row(&self, r: usize) -> *mut u8 { unsafe { self.stage_ptr(0).add(r * self.reader.row_bytes()) } }
    pub(crate) fn mtp_logit_rows(&self) -> usize {
        let ms = self.mtp_s().as_ref().unwrap();
        if ms.n_keep > 0 { ms.n_keep } else { self.w.lm_head.rows }
    }
    /// MTP-rows buffer of outcome je for a pass of kd drafts at p: replay (0 < je < kd) + MTP rows je + 1 (D1) +
    /// spec_conf of D1's logits (conf row 8).
    fn encode_after(&self, e: &Enc, je: usize, kd: usize, p: usize) -> Result<(), String> { self.encode_after_hf(e, je, kd, p, self.io().hf) }
    /// `encode_after` with the sequence's hf rows at `hf0` (batch.rs).
    pub(crate) fn encode_after_hf(&self, e: &Enc, je: usize, kd: usize, p: usize, hf0: usize) -> Result<(), String> {
        let ms = self.mtp_s().as_ref().unwrap();
        if je >= 1 && je < kd { self.encode_replay(e, je)?; }
        self.encode_mtp_rows_hf(e, je + 1, p, hf0)?;
        e.pso("spec_conf").b(0, self.io().logits).b(1, ms.conf + 8 * CONF_BYTES).u(2, self.mtp_logit_rows() as u32).groups((1, 1), 1024);
        Ok(())
    }
    /// Chain step i (>= 2) at MTP position pos: embeds stage row i - 1 (= D_{i-1}) + spec_conf (conf row 8).
    pub(crate) fn encode_chain_conf(&self, e: &Enc, i: usize, pos: usize) -> Result<(), String> {
        let ms = self.mtp_s().as_ref().unwrap();
        self.encode_chain_step(e, ms.hn, self.sl().stage.off + (i - 1) * self.reader.row_bytes(), pos)?;
        e.pso("spec_conf").b(0, self.io().logits).b(1, ms.conf + 8 * CONF_BYTES).u(2, self.mtp_logit_rows() as u32).groups((1, 1), 1024);
        Ok(())
    }

    /// Pipelined deep loop (section header) from the state `feed_stream_spec_mode` leaves: t1 emitted (in `st.gen`)
    /// and staged in slot 0 row 0, D1 in ctrl.draft (+ its conf row), mtp_len == pos. Same stop rules and exit state
    /// as `spec_loop_deep`. `hist`: the sequence so far (context tails for the draft source).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spec_loop_pipe(&self, mode: SpecMode, n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                      st: &mut StreamStats, r: &mut SpecResult, oracle: &mut Option<Oracle>, hist: &mut Vec<u32>,
                      plan: &mut Plan, yield_now: &dyn Fn() -> bool) -> Result<(), String> {
        let (_, _, ms) = self.mtp_parts()?;
        let dmax = mode.max_depth();
        let auto = matches!(mode, SpecMode::Auto { .. });
        let mstage = unsafe { self.arena.base().add(ms.stage) };
        let rb = self.reader.row_bytes();
        let mut src_b = self.draft_src.borrow_mut();
        let mut src = if auto { src_b.as_mut() } else { None };
        let mut q = Queue { ev: self.gpu.shared_event(), n: 0, last_end: 0.0, round: vec![] };
        // a resumed sequence (batch.rs `solo_run`) keeps its counters
        if r.accept_hist.len() != dmax + 1 { r.accept_hist = vec![0; dmax + 1]; }
        if r.depth_hist.len() != dmax + 1 { r.depth_hist = vec![0; dmax + 1]; }
        let mut t1 = st.last.ok_or("spec: no first token")?;
        let mut d1 = self.ctrl_read().draft;
        let mut p1 = mtp_prob(self);
        let mut vpre: Vec<(CmdBuf, usize, usize, usize)> = vec![]; // verify candidates (cb, k, lo, hi) encoded ahead
        let mut chain: Vec<CmdBuf> = vec![];                        // chain steps 2.. of this round, encoded ahead
        let mut gd = 0.0;
        let mut tr = Instant::now();
        {   // first round's chain steps (later rounds encode them while the MTP rows run)
            let p = self.pos();
            for i in 2..=dmax.min(self.opt.max_ctx.saturating_sub(p)) { chain.push(self.enc_cb(&|e| self.encode_chain_conf(e, i, p + i - 2))?); }
        }
        loop {
            let p = self.pos();
            plan.n = p;
            // ---- drafts
            let cap = if auto { dmax.min(self.opt.max_ctx.saturating_sub(p + 1)) } else { dmax };
            if p + 1 + if auto { 0 } else { dmax } > self.opt.max_ctx { st.stop_reason = "ctx_full"; break; }
            let tail: Vec<u32> = hist[hist.len().saturating_sub(TAIL)..].to_vec();
            let prop: Vec<(u32, f64)> = match src.as_mut() {
                Some(s) if cap > 0 => s.propose(&tail, cap).into_iter().map(|(t, c)| (t, c as f64)).collect(),
                _ => vec![],
            };
            let prop_lookup = !prop.is_empty() && src.as_ref().is_some_and(|s| s.last_from_lookup());
            // table conditional q per position
            let qt: Vec<f64> = prop.iter().enumerate().map(|(i, &(_, c))| {
                let prev = if i == 0 { 1.0 } else { prop[i - 1].1.max(1e-6) };
                plan.q_t((c / prev).min(1.0))
            }).collect();
            let mut drafts: Vec<u32> = vec![];
            let mut rows: Vec<(Src, f64)> = vec![]; // per draft: source, MTP prob / table conditional conf
            let qm1 = plan.q_m(1, Some(p1));
            let mut steps_run = 0usize;
            let table_first = auto && !prop.is_empty() && {
                let mut qq = qt.clone();
                if prop[0].0 == d1 { qq[0] = qq[0].max(qm1); }
                let (_, rt) = plan.best(&qq, cap, &|_| 0);
                let qe: Vec<f64> = (1..=cap).map(|s| if s == 1 { qm1 } else { plan.q_m(s, None) }).collect();
                let (_, rm) = plan.best(&qe, cap, &|d| d.saturating_sub(1));
                rt > rm
            };
            if table_first {
                let mut qq = qt.clone();
                if prop[0].0 == d1 { qq[0] = qq[0].max(qm1); }
                let (d, _) = plan.best(&qq, cap, &|_| 0);
                for i in 0..d {
                    let c = if i == 0 { prop[0].1 } else { prop[i].1 / prop[i - 1].1.max(1e-6) };
                    drafts.push(prop[i].0);
                    rows.push((Src::Table, c.min(1.0)));
                }
            } else if cap > 0 {
                drafts.push(d1);
                rows.push((Src::Mtp, p1));
                let mut qs = vec![qm1];
                loop {
                    let s = drafts.len();
                    let last = match oracle.as_mut() { Some(o) => o.apply(p + s, drafts[s - 1]), None => drafts[s - 1] };
                    drafts[s - 1] = last;
                    self.reader.read_rows(&[last], self.stage_row(s))?;
                    if s >= cap { break; }
                    let go_on = !auto || plan.go_on(&qs, cap);
                    if !go_on || s - 1 >= chain.len() { break; }
                    if oracle.as_ref().is_some_and(|o| o.skip_chain) {
                        // DraftOverride::External: no chain step; the placeholder is replaced (and staged) by the
                        // oracle at the top of the next iteration
                        drafts.push(drafts[s - 1]);
                        rows.push((Src::Mtp, 1.0));
                        qs.push(1.0);
                        continue;
                    }
                    let cb = chain[s - 1].clone();
                    q.run(&cb)?;
                    steps_run += 1;
                    let nd = self.ctrl_read().draft;
                    let pr = mtp_prob(self);
                    drafts.push(nd);
                    rows.push((Src::Mtp, pr));
                    qs.push(plan.q_m(s + 1, Some(pr)));
                }
                // final depth among the computed drafts (chain steps are sunk), then a free table extension
                let s = drafts.len();
                let (mut d, _) = if auto { plan.best(&qs, s, &|_| steps_run) } else { (s, 0.0) };
                if auto && !prop.is_empty() && prop.len() > d && prop[..d].iter().map(|x| x.0).eq(drafts[..d].iter().copied()) {
                    let mut qq = qs[..d].to_vec();
                    qq.extend_from_slice(&qt[d..]);
                    let (e, _) = plan.best(&qq, cap, &|_| steps_run);
                    if e > d {
                        drafts.truncate(d);
                        rows.truncate(d);
                        for i in d..e { drafts.push(prop[i].0); rows.push((Src::Table, (prop[i].1 / prop[i - 1].1.max(1e-6)).min(1.0))); }
                        d = e;
                    }
                }
                drafts.truncate(d);
                rows.truncate(d);
            }
            if steps_run > 0 { self.ctrl().mtp_len = p as u32; }
            chain.clear();
            // table rows (and their oracle draws): pread into stage rows 1..
            for i in 0..drafts.len() {
                if rows[i].0 != Src::Mtp {
                    if let Some(o) = oracle.as_mut() { drafts[i] = o.apply(p + 1 + i, drafts[i]); }
                    self.reader.read_rows(&[drafts[i]], self.stage_row(i + 1))?;
                }
            }
            let nd = drafts.len();
            // ---- verify pass: the one encoded ahead, padded, or re-encoded
            let mut k = nd + 1;
            // smallest encoded-ahead K that fits exactly, or with spare rows costing <= PAD_MS
            let pick = vpre.iter().enumerate()
                .filter(|(_, x)| x.2 <= p && p <= x.3 && p + x.1 <= self.opt.max_ctx && x.1 >= k
                        && (x.1 == k || (auto && plan.pass(x.1) - plan.pass(k) <= PAD_MS)))
                .min_by_key(|(_, x)| x.1).map(|(i, _)| i);
            let vcb = match pick {
                Some(i) => {
                    let (cb, ke, _, _) = vpre.swap_remove(i);
                    // spare rows: the table's continuation when it agrees with the drafts, else the last row's token
                    let agree = prop.len() >= nd && prop[..nd].iter().map(|x| x.0).eq(drafts.iter().copied());
                    for i in k..ke {
                        let t = if agree && i - 1 < prop.len() { prop[i - 1].0 } else { drafts.last().copied().unwrap_or(t1) };
                        let t = match oracle.as_mut() { Some(o) => o.apply(p + i, t), None => t };
                        drafts.push(t);
                        rows.push((Src::Pad, 0.0));
                        self.reader.read_rows(&[t], self.stage_row(i))?;
                    }
                    k = ke;
                    r.preencoded += 1;
                    cb
                }
                None => {
                    let te = Instant::now();
                    let cb = self.enc_cb(&|e| self.encode_verify_deep(e, k, p, p))?;
                    r.reencode_ms.push(te.elapsed().as_secs_f64() * 1e3);
                    cb
                }
            };
            vpre.clear();
            let kd = k - 1;
            self.rm_before_pass(); // the pass's "read everything" flag (encoded-ahead buffers read it at run time)
            let vv = q.go(&vcb);
            // overlap: outcome buffers of this pass + the next round's verify, encoded while it runs
            let te = Instant::now();
            let mut after: Vec<CmdBuf> = Vec::with_capacity(k);
            for je in 0..k { after.push(self.enc_cb(&|e| self.encode_after(e, je, kd, p))?); }
            // next round's verify candidates: Fixed = K d + 1; Auto = the planner's K from history and the cheap tiers
            // (K <= 3 differ by > PAD_MS) + 8 (a long table run), each encoded for the position range [p + 1, p + k]
            let mut kc: Vec<usize> = if auto {
                let qe: Vec<f64> = (1..=dmax).map(|s| plan.q_m(s, None)).collect();
                vec![1, 2, 3, plan.best(&qe, dmax, &|d| d.saturating_sub(1)).0 + 1, dmax + 1]
            } else { vec![dmax + 1] };
            kc.sort_unstable();
            kc.dedup();
            let (lo, hi) = (p + 1, p + k);
            for kn in kc {
                if hi + kn > self.opt.max_ctx || kn - 1 > self.opt.spec_depth || !EXACT_KS.contains(&kn) { continue; }
                if let Ok(cb) = self.enc_cb(&|e| self.encode_verify_deep(e, kn, lo, hi)) { vpre.push((cb, kn, lo, hi)); }
            }
            r.overlap_encode_ms.push(te.elapsed().as_secs_f64() * 1e3);
            wait_sig(&q.ev, vv, &vcb)?;
            // ---- decide (as spec_loop_deep)
            let a: Vec<u32> = (0..k).map(|i| self.rec_token(p + i).0).collect();
            let mut j = 0;
            while j < kd && a[j] == drafts[j] && !self.is_eos(a[j]) { j += 1; }
            r.passes += 1;
            r.straddle.note(p, k, self.opt.kv == KvFormat::Q8);
            r.accepted += j;
            if j < kd { r.rejects += 1; }
            let nreal = rows.iter().filter(|x| x.0 != Src::Pad).count();
            r.accept_hist[j.min(dmax)] += 1;
            r.depth_hist[nreal.min(dmax)] += 1;
            for (i, x) in rows.iter().enumerate() {
                let acc = i < j;
                match x.0 { Src::Mtp => { r.drafts_mtp += 1; r.acc_mtp += acc as usize; }
                            Src::Table => { r.drafts_table += 1; r.acc_table += acc as usize; }
                            Src::Pad => { r.drafts_pad += 1; r.acc_pad += acc as usize; } }
                if x.0 == Src::Table && prop_lookup { r.drafts_lookup += 1; r.acc_lookup += acc as usize; }
            }
            {   // passes by draft source (stats only)
                let (ns, nm) = rows.iter().fold((0, 0), |(s, m), x| (s + (x.0 == Src::Table) as usize, m + (x.0 == Src::Mtp) as usize));
                match (ns > 0, nm > 0) { (true, false) => r.passes_src += 1, (true, true) => r.passes_mixed += 1,
                                         (false, true) => r.passes_mtp += 1, (false, false) => r.passes_nodraft += 1 }
            }
            r.drafts.push(d1);
            r.draft_pos.push(p as u32);
            plan.update(&rows, j);
            let mut why: Option<&'static str> = None;
            let mut used = 0;
            for &t in &a[..=j] {
                used += 1;
                st.gen += 1;
                st.last = Some(t);
                r.tokens.push(t);
                if !on_tok(t) { why = Some("cancelled"); break; }
                if self.is_eos(t) || stop.contains(&t) { why = Some("eos"); break; }
                if st.gen >= n_new { why = Some("length"); break; }
            }
            let je = used - 1;
            let held = a[je];
            let gaps: Vec<f32> = (0..=je).map(|i| self.rec_gap(p + i)).collect();
            self.rm_after_pass(&gaps);
            if let Some(s) = src.as_mut() {
                let top3: Vec<[u32; 3]> = (0..=je).map(|i| { let mut t = conf_row(self, i).0; t[0] = a[i]; t }).collect();
                s.observe(&tail, &a[..je], &top3);
            }
            hist.extend_from_slice(&a[..=je]);
            if hist.len() > 4 * TAIL { hist.drain(..hist.len() - TAIL); }
            {
                let cw = self.ctrl();
                cw.pos = (p + je + 1) as u32;
                if je < kd { cw.role ^= 1; }
            }
            self.arena.slice_mut::<f32>(self.sl().rec)[4 * (p + je + 1)..4 * (p + k)].fill(0.0);
            self.set_pos(p + je + 1);
            // ---- MTP rows (+ replay) of outcome je; the next round's chain steps are encoded while it runs
            self.reader.read_rows(&[held], self.stage_row(0))?;
            unsafe { std::ptr::copy_nonoverlapping(self.stage_row(0), mstage, rb) };
            let acb = after.swap_remove(je);
            drop(after);
            let av = q.go(&acb);
            let np = p + je + 1;
            if why.is_none() {
                for i in 2..=dmax.min(self.opt.max_ctx.saturating_sub(np)) { chain.push(self.enc_cb(&|e| self.encode_chain_conf(e, i, np + i - 2))?); }
            }
            wait_sig(&q.ev, av, &acb)?;
            let (busy, idle) = q.close_round()?;
            gd += busy;
            r.gpu_ms.push(busy);
            r.verify_ms.push(gpu_ms(&vcb));
            r.draft_ms.push(busy - gpu_ms(&vcb));
            r.idle_ms.push(idle);
            r.round_ms.push(tr.elapsed().as_secs_f64() * 1e3);
            tr = Instant::now();
            if je >= 1 && je < kd { r.replay.push((je, gpu_ms(&acb))); } // replay + MTP rows buffer
            if why.is_none() && !auto && self.pos() + dmax + 1 > self.opt.max_ctx { why = Some("ctx_full"); }
            if why.is_none() && self.pos() + 1 > self.opt.max_ctx { why = Some("ctx_full"); }
            // batch.rs: another sequence wants the GPU; the state is the exit state of any stop (round boundary)
            if why.is_none() && yield_now() { why = Some("yield"); }
            if let Some(w) = why { st.stop_reason = w; break; }
            d1 = self.ctrl_read().draft;
            p1 = mtp_prob(self);
            t1 = held;
        }
        let _ = (t1, gd);
        Ok(())
    }

    /// Streaming deep MTP decode with a depth policy (`feed_stream_mtp` contract). Fixed(1) = spec.rs's loop.
    pub fn feed_stream_spec_mode(&self, ids: &[u32], n_new: usize, stop: &[u32], on_tok: &mut dyn FnMut(u32) -> bool,
                                 mode: SpecMode, ov: &DraftOverride) -> Result<(StreamStats, SpecResult), String> {
        if mode == SpecMode::Fixed(1) || mode.max_depth() == 0 { return self.feed_stream_mtp_ov(ids, n_new, stop, on_tok, ov); }
        let d = mode.max_depth();
        self.check_spec_depth(d)?;
        let mut st = StreamStats { stop_reason: "", ..Default::default() };
        let mut r = SpecResult { depth: d, mode: mode.name(), ..Default::default() };
        if ids.is_empty() { return Err("feed_stream_spec: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { st.stop_reason = "ctx_full"; return Ok((st, r)); }
        let tp = Instant::now();
        let p0 = self.pos();
        let (t1, hf) = self.prefill_mtp(ids)?;
        self.mtp_first_draft(t1, hf, 0)?;
        let (_, _, ms) = self.mtp_parts()?;
        self.run_cb(&|e| { e.pso("spec_conf").b(0, self.io().logits).b(1, ms.conf + 8 * CONF_BYTES).u(2, self.mtp_logit_rows() as u32).groups((1, 1), 1024); Ok(()) })?;
        let mut hist: Vec<u32> = ids[ids.len().saturating_sub(TAIL)..].to_vec();
        hist.push(t1);
        if matches!(mode, SpecMode::Auto { .. }) {
            if let Some(s) = self.draft_src.borrow_mut().as_mut() {
                let mut seq = ids.to_vec();
                seq.push(t1);
                s.begin_at(p0, &seq);
            }
        }
        st.prefill_s = tp.elapsed().as_secs_f64();
        let td = Instant::now();
        st.gen = 1;
        st.last = Some(t1);
        r.tokens.push(t1);
        if !on_tok(t1) { st.stop_reason = "cancelled"; }
        else if self.is_eos(t1) || stop.contains(&t1) { st.stop_reason = "eos"; }
        else if n_new <= 1 { st.stop_reason = "length"; }
        else {
            let mut oracle = Oracle::new(ov);
            let mut plan = self.new_plan();
            self.spec_loop_pipe(mode, n_new, stop, on_tok, &mut st, &mut r, &mut oracle, &mut hist, &mut plan, &|| false)?;
            r.overridden += oracle.as_ref().map_or(0, |o| o.used);
        }
        if st.stop_reason.is_empty() { st.stop_reason = "length"; }
        st.decode_s = td.elapsed().as_secs_f64();
        st.gpu_ms = r.gpu_ms.iter().sum();
        r.wall = st.decode_s;
        r.prefill_s = st.prefill_s;
        r.finish();
        Ok((st, r))
    }
}

/// Greedy generation from a fresh state with a depth policy (tokens must equal `QwenEngine::generate`); the draft
/// source is flushed afterwards.
pub fn generate_spec_mode(eng: &QwenEngine, prompt: &[u32], n_new: usize, mode: SpecMode, ov: DraftOverride) -> Result<SpecResult, String> {
    if mode == SpecMode::Fixed(1) { return generate_mtp(eng, prompt, n_new, ov, false); }
    let d = mode.max_depth();
    eng.check_spec_depth(d)?;
    let p_len = prompt.len();
    if p_len == 0 { return Err("generate_spec: empty prompt".into()); }
    if p_len + n_new + d > eng.opt.max_ctx {
        return Err(format!("prompt {} + {} new (+{} verify rows) > max_ctx {}", p_len, n_new, d, eng.opt.max_ctx));
    }
    eng.reset();
    if n_new == 0 { eng.prefill(prompt)?; return Ok(SpecResult { depth: d, ..Default::default() }); }
    let (_, mut r) = eng.feed_stream_spec_mode(prompt, n_new, &[], &mut |_| true, mode, &ov)?;
    eng.flush_draft_source();
    r.tokens.truncate(n_new);
    eng.margins_check(p_len, &mut r)?;
    r.finish();
    Ok(r)
}

/// The synchronous reference loop (`spec_loop_deep`, one command buffer per stage, no overlap), fixed depth.
pub fn generate_spec_sync(eng: &QwenEngine, prompt: &[u32], n_new: usize, depth: usize, ov: DraftOverride) -> Result<SpecResult, String> {
    if depth <= 1 { return generate_mtp(eng, prompt, n_new, ov, false); }
    eng.check_spec_depth(depth)?;
    let p_len = prompt.len();
    if p_len == 0 || p_len + n_new + depth > eng.opt.max_ctx { return Err("generate_spec_sync: bad prompt / max_ctx".into()); }
    eng.reset();
    let (_, mut r) = eng.feed_stream_spec_sync(prompt, n_new, &[], &mut |_| true, depth, &ov)?;
    r.tokens.truncate(n_new);
    eng.margins_check(p_len, &mut r)?;
    r.finish();
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_27b(n: usize, q: f64) -> (Plan, Vec<f64>) {
        let mut p = Plan::new(SpecCost::default()).with_attn(16, 1024, false);
        p.n = n;
        (p, vec![q; SPEC_DEPTH_MAX])
    }

    /// Within each attention path (the MLX-mirror rows up to N = 8192, split-K above) the chosen depth never rises
    /// as the context grows, and it falls below the short-context depth at 32K for q 0.7 (high acceptance, q >= 0.8,
    /// still pays for deep passes at 32K with attn_split_rows; low already picks d <= 1). Split-K rows cost less per
    /// key than the mirror rows, so the depth may rise again where a pass leaves the mirror path.
    #[test]
    fn depth_falls_with_context() {
        for q in [0.5, 0.7, 0.8, 0.9] {
            let mut prev = usize::MAX;
            let mut ds = vec![];
            for n in [100usize, 1024, 4096, 8183, 8191, 8192, 12288, 16384, 24576, 32768] {
                let (p, qs) = plan_27b(n, q);
                let (d, _) = p.best(&qs, SPEC_DEPTH_MAX, &|d| d.saturating_sub(1));
                if n + 8 > 8192 && n <= 8192 { prev = usize::MAX; } // first N where a deep pass is costed as split-K
                assert!(d <= prev, "q {}: depth rose to {} at N {} ({:?})", q, d, n, ds);
                prev = d;
                ds.push(d);
            }
            if q == 0.7 { assert!(ds[ds.len() - 1] < ds[0], "q {}: no fall {:?}", q, ds); }
            println!("q {}: depth by N {:?}", q, ds);
        }
        // the measured 32K acceptance (~0.37): at most one draft pays off (with attn_split_rows a 2-row pass costs
        // ~1.5x a 1-row pass in attention; with attn_partial it was ~2.3x and the planner chose 0)
        let (p, qs) = plan_27b(32560, 0.37);
        assert!(p.best(&qs, SPEC_DEPTH_MAX, &|d| d.saturating_sub(1)).0 <= 1);
        // without the attention term (old policy) the same stats still draft
        let mut old = Plan::new(SpecCost::default());
        old.n = 32560;
        assert!(old.best(&qs, SPEC_DEPTH_MAX, &|d| d.saturating_sub(1)).0 > 0);
    }

    /// Never below plain: the chosen depth's expected tokens per ms >= the one-row pass's, for any stats / N.
    #[test]
    fn never_below_plain() {
        for n in [0usize, 500, 8191, 8192, 20000, 32768] {
            for q in [0.0, 0.1, 0.3, 0.5, 0.9, 1.0] {
                let (p, qs) = plan_27b(n, q);
                let st = |d: usize| d.saturating_sub(1);
                let (d, r) = p.best(&qs, SPEC_DEPTH_MAX, &st);
                assert!(r >= 1.0 / p.t(0, st(0)) - 1e-15, "N {} q {}: d {} rate {} < plain", n, q, d, r);
                if q == 0.0 { assert_eq!(d, 0); }
            }
        }
    }

    /// The chain of one round as `spec_loop_pipe` runs it: MTP drafts with top-1 probs `probs` (one per chain step,
    /// as far as the planner continues), then the final depth among the computed drafts (steps sunk). Returns d.
    fn round_depth(p: &Plan, probs: &[f64]) -> usize {
        let cap = SPEC_DEPTH_MAX;
        let mut qs = vec![p.q_m(1, Some(probs[0]))];
        while qs.len() < cap && p.go_on(&qs, cap) { let s = qs.len(); qs.push(p.q_m(s + 1, Some(probs[s]))); }
        let steps = qs.len() - 1;
        p.best(&qs, qs.len(), &|_| steps).0
    }
    fn rng(s: &mut u64) -> f64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; (*s >> 11) as f64 / (1u64 << 53) as f64 }

    /// Synthetic acceptance histories of the two measured regimes (27B, 32K context):
    /// copy-heavy edit (q8 KV): ~0.9 conditional acceptance at every depth, MTP probs mostly high with some mid / low
    /// ones that are accepted too -> the planner keeps going deep (>= 5 drafts) even through a low-prob draft, like
    /// the context-blind policy (measured: all depth 7, 28.59 tok/s vs 23.49 with depth 1 in 10 / 34 rounds).
    /// prose summary (bf16 KV): ~0.5 acceptance, roughly calibrated MTP probs -> shallow (<= 2 drafts) on typical
    /// drafts, as the context-aware policy measured (12.79 vs 11.77 tok/s context-blind).
    #[test]
    fn regimes_copy_deep_prose_shallow() {
        let mut s = 0x9e37_79b9u64;
        // copy: 30 rounds of 7 MTP drafts, ~0.88 conditional acceptance
        let mut cp = Plan::new(SpecCost::default()).with_attn(16, 1024, true);
        cp.n = 32600;
        let cyc = [0.95, 0.9, 0.5, 0.97, 0.3, 0.9, 0.95, 0.6, 0.92];
        for r in 0..30 {
            let rows: Vec<(Src, f64)> = (0..7).map(|i| (Src::Mtp, cyc[(r + i) % cyc.len()])).collect();
            let mut j = 0;
            while j < 7 && rng(&mut s) > 0.12 { j += 1; }
            cp.update(&rows, j);
        }
        // incl. drafts whose MTP prob falls in a bin the history never saw (< 0.2)
        let copy_rounds: [&[f64]; 5] = [&[0.95, 0.9, 0.95, 0.9, 0.97, 0.9, 0.95], &[0.9, 0.3, 0.95, 0.9, 0.9, 0.95, 0.9],
                                        &[0.5, 0.9, 0.9, 0.6, 0.9, 0.95, 0.9], &[0.15, 0.95, 0.9, 0.9, 0.5, 0.9, 0.9],
                                        &[0.9, 0.15, 0.9, 0.9, 0.9, 0.9, 0.9]];
        let dc: Vec<usize> = copy_rounds.iter().map(|pr| round_depth(&cp, pr)).collect();
        // prose: 60 rounds of 1..2 drafts, accepted with probability ~ the MTP prob (overall ~0.5)
        let mut pp = Plan::new(SpecCost::default()).with_attn(16, 1024, false);
        pp.n = 32560;
        let (mut acc, mut tot) = (0, 0);
        for r in 0..60 {
            let k = 1 + r % 2;
            let rows: Vec<(Src, f64)> = (0..k).map(|_| (Src::Mtp, 0.1 + 0.85 * rng(&mut s))).collect();
            let mut j = 0;
            while j < k && rng(&mut s) < rows[j].1 { j += 1; }
            acc += j; tot += (j + 1).min(k);
            pp.update(&rows, j);
        }
        let prose_rounds: [&[f64]; 4] = [&[0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5], &[0.6, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5],
                                         &[0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], &[0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]];
        let dp: Vec<usize> = prose_rounds.iter().map(|pr| round_depth(&pp, pr)).collect();
        println!("copy 32K q8: ew {:?} cal {:?} -> depths {:?}", cp.ew, cp.cal_m, dc);
        println!("prose 32K bf16 (acceptance {:.2}): ew {:?} cal {:?} -> depths {:?}", acc as f64 / tot as f64, pp.ew, pp.cal_m, dp);
        assert!(dc.iter().all(|&d| d >= 5), "copy regime not deep: {:?}", dc);
        assert!(dp.iter().all(|&d| d <= 2), "prose regime not shallow: {:?}", dp);
        // a fresh plan (no history) with a low-prob first draft: stays cautious at 32K
        let mut f = Plan::new(SpecCost::default()).with_attn(16, 1024, false);
        f.n = 32560;
        assert!(round_depth(&f, &[0.2; 7]) <= 1);
    }

    /// Short contexts: the attention term barely moves the plan (same depth as the context-blind planner).
    #[test]
    fn short_context_unchanged() {
        for q in [0.5, 0.7, 0.8, 0.9] {
            let (p, qs) = plan_27b(200, q);
            let mut old = Plan::new(SpecCost::default());
            old.n = 200;
            let st = |d: usize| d.saturating_sub(1);
            assert_eq!(p.best(&qs, SPEC_DEPTH_MAX, &st).0, old.best(&qs, SPEC_DEPTH_MAX, &st).0, "q {}", q);
        }
    }
}
