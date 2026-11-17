//! Batched decode (2026-11-04): 2..=SEQS_MAX sequences decode together, one exact verify pass per round.
//!
//! Each sequence lives in its own engine slot (`QwenOptions.seqs`, `SeqSlot`: KV, MTP KV, DeltaNet state x2, ctrl,
//! rec, staging, replay save, MTP rows). A round:
//!   drafts  per sequence: D1 from its last MTP rows (ctrl.draft), then chained MTP steps; one command buffer per
//!           chain step carries the step of every sequence still drafting (`encode_chain_conf` per slot). The depth
//!           of each sequence is its own planner's choice (`spec_deep::Plan`, as the solo loop) capped by its row
//!           budget (`row_budget`: 8 rows per pass shared out, 2 -> 4 + 4, 3 -> 3 + 3 + 2, 4 -> 2 each).
//!   verify  ONE pass over all blocks [t1, D1..Dd] of every sequence (`encode_layer_batch` / `encode_head_batch`):
//!           matmuls over all rows, DeltaNet / attention / argmax per sequence on its own state, KV, ctrl, rec.
//!   decide  per sequence exactly as `spec_loop_pipe` (longest accepted prefix, stop rules, ctrl.pos / role, rec).
//!   after   ONE command buffer: per sequence its DeltaNet replay (0 < je < d) + MTP rows + next D1 (`encode_after_hf`
//!           on that sequence's hf rows).
//! Lossless: every emitted token is the argmax of an exact verify row, and each row of a batched pass is bit-identical
//! to the same row of that sequence's own pass (K-invariant matmuls, per-sequence mixers), so each sequence's tokens
//! and its state after a round are those of plain greedy decoding alone, whatever the drafts and the batch.
//!
//! A sequence can also run alone (`solo_run`: the solo pipelined loop `spec_loop_pipe`, encode-ahead and draft table
//! included) until a `yield_now` check says another sequence wants the GPU; it stops at a round boundary in the same
//! exit state, so it can join a batch, leave it, and run alone again (the scheduler in streamdense-gateway).
use std::time::Instant;

use streamdense_metal::{commit, gpu_ms, wait};

use crate::mtp::CONF_BYTES;
use crate::qwen_engine::{QwenEngine, StreamStats};
use crate::qwen_layers::{encode_head_batch, encode_layer_batch, PassIo, SeqRows};
use crate::spec::{DraftOverride, Oracle, SpecResult};
use crate::spec_deep::{mtp_prob, Plan, Src, SpecMode, TAIL};

/// Most sequence slots an engine allocates (`QwenOptions.seqs`).
pub const SEQS_MAX: usize = 4;
/// Rows of one batched verify pass (the exact pass K cap).
pub const BATCH_ROWS: usize = 8;
const BF: usize = 2;

/// Rows per sequence of a batched pass for `n` active sequences: BATCH_ROWS shared out, earlier ones first.
pub fn row_budget(n: usize) -> Vec<usize> {
    if n == 0 { return vec![]; }
    let (b, x) = (BATCH_ROWS / n, BATCH_ROWS % n);
    (0..n).map(|i| b + (i < x) as usize).collect()
}

/// One sequence decoding (solo or batched) between rounds. Its engine state lives in slot `slot`.
pub struct BatchSeq {
    pub slot: usize,
    pub n_new: usize,
    pub stop: Vec<u32>,
    /// `feed_stream` contract: gen counts every emitted token (t1 included), last = the held (unfed) last token
    pub st: StreamStats,
    pub r: SpecResult,
    /// Some(stop reason) once finished (eos | length | cancelled | ctx_full); the engine slot then holds the solo exit
    /// state (last emitted token not fed, MTP row of the last fed position fed), so a later feed continues it
    pub done: Option<&'static str>,
    /// verify passes run batched with other sequences / alone (`r.passes` = both)
    pub batched: usize,
    /// chained MTP draft steps run in batched rounds (fused with other sequences when they chain too)
    pub chained: usize,
    mode: SpecMode,
    hist: Vec<u32>,
    plan: Plan,
    oracle: Option<Oracle>,
    t_dec: Instant,
}

impl BatchSeq {
    pub fn active(&self) -> bool { self.done.is_none() }
    fn finish(&mut self, why: &'static str) {
        if self.done.is_some() { return; }
        self.done = Some(why);
        self.st.stop_reason = why;
        self.st.decode_s = self.t_dec.elapsed().as_secs_f64();
        self.st.gpu_ms = self.r.gpu_ms.iter().sum();
        self.r.wall = self.st.decode_s;
        self.r.prefill_s = self.st.prefill_s;
        self.r.overridden += self.oracle.as_ref().map_or(0, |o| o.used);
        self.r.finish();
    }
}

/// Per-round work of one sequence.
struct Work {
    i: usize,
    p: usize,
    cap: usize,
    drafts: Vec<u32>,
    rows: Vec<(Src, f64)>,
    qs: Vec<f64>,
    steps: usize,
    chaining: bool,
    row0: usize,
    k: usize,
    je: usize,
    why: Option<&'static str>,
}

impl QwenEngine {
    /// Start a sequence in slot `slot` (selected from here on): prefill `ids` with the MTP warm-up from the slot's
    /// current position (the caller resets the slot for a new sequence), first draft, emit t1, exactly as
    /// `feed_stream_spec_mode` starts. Needs MTP with spec_depth >= 2 (the engine's `spec_mode`). `ov`: test hook.
    pub fn batch_begin(&self, slot: usize, ids: &[u32], n_new: usize, stop: &[u32], ov: &DraftOverride,
                       on_tok: &mut dyn FnMut(u32) -> bool) -> Result<BatchSeq, String> {
        self.select_seq(slot);
        let mode = self.spec_mode();
        let d = mode.max_depth();
        if d < 2 { return Err(format!("batch: needs MTP spec depth >= 2 (engine spec_depth {})", self.opt.spec_depth)); }
        self.check_spec_depth(d)?;
        if ids.is_empty() { return Err("batch_begin: no tokens".into()); }
        let mut s = BatchSeq { slot, n_new, stop: stop.to_vec(), st: StreamStats { stop_reason: "", ..Default::default() },
                               r: SpecResult { depth: d, mode: mode.name(), ..Default::default() }, done: None, batched: 0, chained: 0, mode,
                               hist: vec![], plan: self.new_plan(), oracle: Oracle::new(ov), t_dec: Instant::now() };
        if self.pos() + ids.len() > self.opt.max_ctx { s.finish("ctx_full"); return Ok(s); }
        let tp = Instant::now();
        let p0 = self.pos();
        let (t1, hf) = self.prefill_mtp(ids)?;
        self.mtp_first_draft(t1, hf, 0)?;
        let ms = self.mtp_s().as_ref().ok_or("batch: MTP is off")?;
        self.run_cb(&|e| { e.pso("spec_conf").b(0, self.io().logits).b(1, ms.conf + 8 * CONF_BYTES).u(2, self.mtp_logit_rows() as u32).groups((1, 1), 1024); Ok(()) })?;
        s.hist = ids[ids.len().saturating_sub(TAIL)..].to_vec();
        s.hist.push(t1);
        if matches!(mode, SpecMode::Auto { .. }) {
            if let Some(src) = self.draft_src.borrow_mut().as_mut() {
                let mut seq = ids.to_vec();
                seq.push(t1);
                src.begin_at(p0, &seq);
            }
        }
        s.st.prefill_s = tp.elapsed().as_secs_f64();
        s.t_dec = Instant::now();
        s.st.gen = 1;
        s.st.last = Some(t1);
        s.r.tokens.push(t1);
        if !on_tok(t1) { s.finish("cancelled"); }
        else if self.is_eos(t1) || stop.contains(&t1) { s.finish("eos"); }
        else if n_new <= 1 { s.finish("length"); }
        Ok(s)
    }

    /// Run `s` alone with the solo pipelined loop (encode-ahead, draft table) until it finishes or `yield_now()` turns
    /// true at a round boundary (then `s` stays active, in the exit state, ready for `batch_round` / `solo_run`).
    pub fn solo_run(&self, s: &mut BatchSeq, on_tok: &mut dyn FnMut(u32) -> bool, yield_now: &dyn Fn() -> bool) -> Result<(), String> {
        if s.done.is_some() { return Ok(()); }
        self.select_seq(s.slot);
        let t1 = s.st.last.ok_or("solo_run: no held token")?;
        self.reader.read_rows(&[t1], self.stage_ptr(0))?;
        s.st.stop_reason = "";
        let stop = s.stop.clone();
        let r = self.spec_loop_pipe(s.mode, s.n_new, &stop, on_tok, &mut s.st, &mut s.r, &mut s.oracle, &mut s.hist, &mut s.plan, yield_now);
        if let Err(e) = r { s.finish("error"); return Err(e); }
        match s.st.stop_reason {
            "yield" => s.st.stop_reason = "",
            "" => s.finish("length"),
            w => s.finish(w),
        }
        Ok(())
    }

    /// One batched round over the active sequences of `seqs` (each in its own slot; see the module doc). `on_tok(i, t)`
    /// gets every emitted token of `seqs[i]` in order and returns false to cancel that sequence. Finished sequences are
    /// skipped; a sequence that stops this round is finished (`done`). The selected slot afterwards is unspecified.
    pub fn batch_round(&self, seqs: &mut [&mut BatchSeq], on_tok: &mut dyn FnMut(usize, u32) -> bool) -> Result<(), String> {
        let act: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].done.is_none()).collect();
        if act.is_empty() { return Ok(()); }
        {
            let mut sl: Vec<usize> = act.iter().map(|&i| seqs[i].slot).collect();
            sl.sort_unstable();
            sl.dedup();
            if sl.len() != act.len() { return Err("batch_round: two active sequences share a slot".into()); }
        }
        let budget = row_budget(act.len());
        let max_ctx = self.opt.max_ctx;
        let rb = self.reader.row_bytes();
        let hb = self.cfg.hidden * BF;
        // ---- set-up: D1 per sequence (solo stop rules first)
        let mut ws: Vec<Work> = vec![];
        for (bi, &i) in act.iter().enumerate() {
            let s = &mut *seqs[i];
            self.select_seq(s.slot);
            let p = self.pos();
            let dmax = s.mode.max_depth();
            let auto = matches!(s.mode, SpecMode::Auto { .. });
            if p + 1 + if auto { 0 } else { dmax } > max_ctx { s.finish("ctx_full"); continue; }
            let cap = (if auto { dmax.min(max_ctx.saturating_sub(p + 1)) } else { dmax }).min(budget[bi] - 1);
            s.plan.n = p;
            let mut w = Work { i, p, cap, drafts: vec![], rows: vec![], qs: vec![], steps: 0, chaining: cap > 0, row0: 0, k: 0, je: 0, why: None };
            if cap > 0 {
                let d1 = self.ctrl_read().draft;
                let p1 = mtp_prob(self);
                let d = match s.oracle.as_mut() { Some(o) => o.apply(p + 1, d1), None => d1 };
                self.reader.read_rows(&[d], self.stage_row(1))?;
                w.drafts.push(d);
                w.rows.push((Src::Mtp, p1));
                w.qs.push(s.plan.q_m(1, Some(p1)));
                s.r.drafts.push(d1);
                s.r.draft_pos.push(p as u32);
            }
            ws.push(w);
        }
        if ws.is_empty() { return Ok(()); }
        // ---- chained MTP drafts: one command buffer per step for every sequence still drafting
        let mut gchain = 0.0;
        loop {
            for w in ws.iter_mut() {
                if !w.chaining { continue; }
                let s = &*seqs[w.i];
                let n = w.drafts.len();
                let go_on = n < w.cap && match s.mode {
                    SpecMode::Auto { .. } => s.plan.go_on(&w.qs, w.cap),
                    SpecMode::Fixed(_) => true,
                };
                w.chaining = go_on;
            }
            let run: Vec<usize> = (0..ws.len()).filter(|&x| ws[x].chaining).collect();
            if run.is_empty() { break; }
            let serial = run.len() == 1 || mtp_serial();
            let cb = self.enc_cb(&|e| {
                if !serial {
                    let items: Vec<(usize, usize, usize)> = run.iter().map(|&x| { let w = &ws[x]; let n = w.drafts.len(); (seqs[w.i].slot, n + 1, w.p + n - 1) }).collect();
                    return self.encode_chain_batch(e, &items);
                }
                for &x in &run {
                    let w = &ws[x];
                    self.select_seq(seqs[w.i].slot);
                    let n = w.drafts.len();
                    self.encode_chain_conf(e, n + 1, w.p + n - 1)?;
                }
                Ok(())
            })?;
            commit(&cb);
            wait(&cb)?;
            gchain += gpu_ms(&cb);
            for &x in &run {
                let w = &mut ws[x];
                let s = &mut *seqs[w.i];
                self.select_seq(s.slot);
                let nd = self.ctrl_read().draft;
                let pr = mtp_prob(self);
                let n = w.drafts.len() + 1;
                let d = match s.oracle.as_mut() { Some(o) => o.apply(w.p + n, nd), None => nd };
                self.reader.read_rows(&[d], self.stage_row(n))?;
                w.drafts.push(d);
                w.rows.push((Src::Mtp, pr));
                w.qs.push(s.plan.q_m(n, Some(pr)));
                w.steps += 1;
                s.chained += 1;
            }
        }
        // final depth per sequence (chain steps are sunk), MTP KV length back to the position
        let mut k_tot = 0;
        for w in ws.iter_mut() {
            let s = &*seqs[w.i];
            let n = w.drafts.len();
            let d = if matches!(s.mode, SpecMode::Auto { .. }) && n > 0 { s.plan.best(&w.qs, n, &|_| w.steps).0 } else { n };
            w.drafts.truncate(d);
            w.rows.truncate(d);
            if w.steps > 0 { self.select_seq(s.slot); self.ctrl().mtp_len = w.p as u32; }
            w.row0 = k_tot;
            w.k = d + 1;
            k_tot += w.k;
        }
        // ---- the batched verify pass
        let blocks: Vec<SeqRows> = ws.iter().map(|w| {
            let sl = &self.slots[seqs[w.i].slot];
            SeqRows { row0: w.row0, k: w.k, pos: w.p, nch: self.nch_for(w.p, w.k), ctrl: sl.ctrl.off, rec: sl.rec.off, dn: sl.dn,
                      kv: &sl.kv, mtp_kv: sl.mtp_kv, dn_save: if w.k > 2 { sl.dn_replay.map(|r| r.off) } else { None }, ck: w.k > 1,
                      ban: self.ban_io(seqs[w.i].slot) }
        }).collect();
        if blocks.iter().any(|b| b.k > 2 && b.dn_save.is_none()) { return Err("batch: no DeltaNet replay buffer".into()); }
        let io = PassIo { k: k_tot, nch_max: blocks.iter().map(|b| b.nch).max().unwrap_or(1), dn_save: None, rm: None, ..self.s.io };
        let vocab = self.w.lm_head.rows;
        let vcb = self.enc_cb(&|e| {
            let c = &self.cfg;
            for w in &ws {
                let sl = &self.slots[seqs[w.i].slot];
                let ms = sl.mtp_s.as_ref().ok_or("batch: MTP is off")?;
                e.pso("embed_q4").b(0, sl.stage.at(0)).b(1, io.h + w.row0 * hb).u(2, w.k as u32);
                e.threads((c.hidden / 8, w.k), 64);
                // emb(D1..Dd) for the MTP rows after the pass (as encode_verify_deep)
                let nb = (w.k - 1) * hb;
                if nb > 0 { e.pso("copy_bytes").b(0, io.h + (w.row0 + 1) * hb).b(1, ms.emb).u(2, nb as u32).threads((nb, 1), 256); }
            }
            for l in 0..c.layers { encode_layer_batch(e, c, &self.w.layers[l], &blocks, &io, l)?; }
            encode_head_batch(e, c, &self.w, &io, &blocks)?;
            for w in &ws {
                let ms = self.slots[seqs[w.i].slot].mtp_s.as_ref().ok_or("batch: MTP is off")?;
                e.pso("spec_conf").b(0, io.logits + w.row0 * vocab * 4).b(1, ms.conf).u(2, vocab as u32).groups((w.k, 1), 1024);
            }
            Ok(())
        })?;
        commit(&vcb);
        wait(&vcb)?;
        let gv = gpu_ms(&vcb);
        // ---- decide per sequence (as spec_loop_pipe)
        for w in ws.iter_mut() {
            let s = &mut *seqs[w.i];
            self.select_seq(s.slot);
            let (p, k, kd) = (w.p, w.k, w.k - 1);
            let dmax = s.mode.max_depth();
            let a: Vec<u32> = (0..k).map(|i| self.rec_token(p + i).0).collect();
            let mut j = 0;
            while j < kd && a[j] == w.drafts[j] && !self.is_eos(a[j]) { j += 1; }
            s.r.passes += 1;
            s.r.straddle.note(p, k, self.opt.kv == crate::qwen_engine::KvFormat::Q8);
            if act.len() > 1 { s.batched += 1; }
            s.r.accepted += j;
            if j < kd { s.r.rejects += 1; }
            if s.r.accept_hist.len() != dmax + 1 { s.r.accept_hist = vec![0; dmax + 1]; }
            if s.r.depth_hist.len() != dmax + 1 { s.r.depth_hist = vec![0; dmax + 1]; }
            s.r.accept_hist[j.min(dmax)] += 1;
            s.r.depth_hist[kd.min(dmax)] += 1;
            s.r.drafts_mtp += kd;
            s.r.acc_mtp += j;
            if kd > 0 { s.r.passes_mtp += 1; } else { s.r.passes_nodraft += 1; }
            s.plan.update(&w.rows, j);
            let mut used = 0;
            for &t in &a[..=j] {
                used += 1;
                s.st.gen += 1;
                s.st.last = Some(t);
                s.r.tokens.push(t);
                if !on_tok(w.i, t) { w.why = Some("cancelled"); break; }
                if self.is_eos(t) || s.stop.contains(&t) { w.why = Some("eos"); break; }
                if s.st.gen >= s.n_new { w.why = Some("length"); break; }
            }
            let je = used - 1;
            w.je = je;
            s.hist.extend_from_slice(&a[..=je]);
            if s.hist.len() > 4 * TAIL { s.hist.drain(..s.hist.len() - TAIL); }
            {
                let cw = self.ctrl();
                cw.pos = (p + je + 1) as u32;
                if je < kd { cw.role ^= 1; }
            }
            self.arena.slice_mut::<f32>(self.sl().rec)[4 * (p + je + 1)..4 * (p + k)].fill(0.0);
            self.sl().pos.set(p + je + 1);
            // the held token: stage row 0 (next verify) and the MTP stage (its MTP row below)
            let ms = self.mtp_s().as_ref().ok_or("batch: MTP is off")?;
            self.reader.read_rows(&[a[je]], self.stage_row(0))?;
            unsafe { std::ptr::copy_nonoverlapping(self.stage_row(0), self.arena.base().add(ms.stage), rb) };
        }
        // ---- replay + MTP rows + next D1 of every sequence, one command buffer
        let acb = self.enc_cb(&|e| {
            for w in &ws {
                self.select_seq(seqs[w.i].slot);
                self.encode_after_hf(e, w.je, w.k - 1, w.p, io.hf + w.row0 * hb)?;
            }
            Ok(())
        })?;
        commit(&acb);
        wait(&acb)?;
        let ga = gpu_ms(&acb);
        let n = ws.len() as f64;
        for w in &ws {
            let s = &mut *seqs[w.i];
            self.select_seq(s.slot);
            s.r.verify_ms.push(gv);
            s.r.gpu_ms.push((gv + ga + gchain) / n);
            let auto = matches!(s.mode, SpecMode::Auto { .. });
            let mut why = w.why;
            if why.is_none() && !auto && self.pos() + s.mode.max_depth() + 1 > max_ctx { why = Some("ctx_full"); }
            if why.is_none() && self.pos() + 1 > max_ctx { why = Some("ctx_full"); }
            if let Some(y) = why { s.finish(y); }
        }
        Ok(())
    }
}


/// `SD_BATCH_MTP=serial`: chain steps of a batched round run one sequence after the other in the step's command
/// buffer (`encode_chain_conf` per slot) instead of the fused step (A/B and tests; drafts are identical).
fn mtp_serial() -> bool { std::env::var("SD_BATCH_MTP").map(|v| v == "serial").unwrap_or(false) }

impl QwenEngine {
    /// One FUSED chained MTP draft step for several sequences, one row each: `items` = (slot, i, pos) with i = the
    /// chain step (stage row i - 1 = D_{i-1} of that slot) at MTP position pos (= that slot's ctrl.mtp_len when it
    /// runs). Per sequence: copy hn -> emb row 1, embed D_{i-1} -> emb row 0, mtp_prep into its row of the gather
    /// buffer; then fc over all rows, the MTP layer over all rows (attention per sequence on its MTP KV, ctrl =
    /// its mctrl), per sequence select_row + mtp.norm -> its hn (the next step's source) and its gather row, the
    /// lm_head over all rows, per sequence mtp_draft + spec_conf (conf row 8). Each row is bit-identical to
    /// `encode_chain_conf` of that sequence alone (K-invariant kernels); a pruned draft vocab keeps the per-sequence head.
    fn encode_chain_batch(&self, e: &streamdense_metal::Enc, items: &[(usize, usize, usize)]) -> Result<(), String> {
        let (bcat, bh, bhn) = self.bmtp.ok_or("batch: no fused MTP scratch (seqs = 1)")?;
        let w = self.w.mtp.as_ref().ok_or("batch: MTP is off")?;
        let c = &self.cfg;
        let (h, hb, b) = (c.hidden, c.hidden * BF, items.len());
        let io = self.s.io;
        let head = &self.w.lm_head;
        let q8 = self.opt.kv == crate::qwen_engine::KvFormat::Q8;
        for (r, &(slot, i, _)) in items.iter().enumerate() {
            let sl = &self.slots[slot];
            let ms = sl.mtp_s.as_ref().ok_or("batch: MTP is off")?;
            e.pso("copy_bytes").b(0, ms.hn).b(1, ms.emb + hb).u(2, hb as u32).threads((hb, 1), 256);
            e.pso("embed_q4").b(0, sl.stage.off + (i - 1) * self.reader.row_bytes()).b(1, ms.emb).u(2, 1);
            e.threads((h / 8, 1), 64);
            e.pso("mtp_prep").b(0, ms.emb).b(1, ms.emb + hb).b(2, w.pre_e.off).b(3, w.pre_h.off).b(4, bcat + r * 2 * hb).b(5, sl.ctrl.off)
                .f(6, c.eps).u(7, h as u32).u(8, 1).b(9, ms.mctrl)
                .groups((1, 2), 1024);
        }
        // fc: bf16 [H][2H], rows in pairs (the K-invariant kernel of encode_mtp)
        let cc = 2 * h;
        let mut t = 0;
        while t < b {
            let n = (b - t).min(2);
            e.pso(if n == 2 { "bf16_gemv_k2" } else { "bf16_gemv_k1" })
                .b(0, w.fc.off).b(1, bcat + t * cc * BF).b(2, bh + t * hb).u(3, h as u32).u(4, cc as u32)
                .groups((h.div_ceil(32), 1), 256);
            t += n;
        }
        let blocks: Vec<SeqRows> = items.iter().enumerate().map(|(r, &(slot, _, pos))| {
            let sl = &self.slots[slot];
            let ms = sl.mtp_s.as_ref().unwrap();
            SeqRows { row0: r, k: 1, pos: crate::mtp::path_pos(pos, pos, 1, q8).unwrap_or(pos), nch: self.nch_for(pos, 1), ctrl: ms.mctrl,
                      rec: sl.rec.off, dn: sl.dn, kv: &[], mtp_kv: sl.mtp_kv, dn_save: None, ck: false, ban: None }
        }).collect();
        let mio = PassIo { k: b, h: bh, nch_max: blocks.iter().map(|x| x.nch).max().unwrap_or(1), dn_save: None, rm: None, ..io };
        encode_layer_batch(e, c, &w.layer, &blocks, &mio, c.layers)?;
        let pruned = self.slots[items[0].0].mtp_s.as_ref().map_or(0, |m| m.n_keep) > 0;
        for (r, &(slot, _, _)) in items.iter().enumerate() {
            let sl = &self.slots[slot];
            let ms = sl.mtp_s.as_ref().unwrap();
            e.pso("select_row").b(0, bh + r * hb).b(1, ms.row).u(2, h as u32).b(3, ms.mctrl).threads((h, 1), 256);
            e.pso("rmsnorm_b").b(0, ms.row).b(1, w.norm.off).b(2, ms.hn).u(3, h as u32).f(4, c.eps).u(5, h as u32)
                .groups((1, 1), 1024);
            if pruned {
                crate::mtp::head_pruned(e, io.gemv, head, ms.keep, ms.n_keep, ms.hn, io.logits);
                e.pso("mtp_draft_idx").b(0, io.logits).b(1, sl.ctrl.off).b(2, ms.mctrl).u(3, ms.n_keep as u32).b(4, ms.keep)
                    .groups((1, 1), 1024);
                e.pso("spec_conf").b(0, io.logits).b(1, ms.conf + 8 * CONF_BYTES).u(2, ms.n_keep as u32).groups((1, 1), 1024);
            } else {
                e.pso("copy_bytes").b(0, ms.hn).b(1, bhn + r * hb).u(2, hb as u32).threads((hb, 1), 256);
            }
        }
        if !pruned {
            crate::qwen_layers::gemv(e, io.gemv, head, b, 3, bhn, io.logits);
            for (r, &(slot, _, _)) in items.iter().enumerate() {
                let sl = &self.slots[slot];
                let ms = sl.mtp_s.as_ref().unwrap();
                let lg = io.logits + r * head.rows * 4;
                e.pso("mtp_draft").b(0, lg).b(1, sl.ctrl.off).b(2, ms.mctrl).u(3, head.rows as u32).groups((1, 1), 1024);
                e.pso("spec_conf").b(0, lg).b(1, ms.conf + 8 * CONF_BYTES).u(2, head.rows as u32).groups((1, 1), 1024);
            }
        }
        Ok(())
    }
}
