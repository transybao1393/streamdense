//! `prefill_profile` (2026-11-17, bench only): where the fast prompt read (fast prefill, plan U25) of the real engine
//! spends its time. Nothing here is used by the engine; no kernel or engine code path changes.
//! - `pf_prefill_timed`: the engine's own prompt read (`prefill_mtp` with MTP on, else `prefill`): same chunks, same
//!   passes (`encode_pass`), same MTP warm-up encoder, same event ordering, plus per chunk: CPU staging (embedding
//!   rows) ms, CPU encode ms, and the command buffer's GPU span. The test checks its first token / rec bits against
//!   the real `prefill_mtp` / `prefill`.
//! - `pf_profile_at`: one fast chunk of `k` rows at kv position `p` (a middle chunk: no head, MTP warm-up on all k
//!   rows), re-encoded with `Enc::with_filter` per component class (one command buffer holding only that class's
//!   dispatches, `rewind_to(p)` before each), GPU ms per class vs the whole chunk. Content-independent timing: the
//!   DeltaNet state is NOT restored (as `rewind_to` says), so call it only after the outputs that matter were read.
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::time::Instant;

use streamdense_metal::{cb_signal, cb_wait, commit, gpu_ms, gpu_span, wait, CmdBuf, Enc};

use crate::qwen_engine::{QwenEngine, ATTN_CHUNK};
use crate::qwen_layers::{encode_layer_prefill, PassIo};
use crate::weights::Mixer;


/// One chunk of `pf_prefill_timed`.
#[derive(Clone, Debug, Default)]
pub struct ChunkTime {
    /// first row (prompt index) and rows
    pub i: usize,
    pub k: usize,
    /// CPU: embedding rows staged (target + MTP rows), encode (both encoders)
    pub stage_ms: f64,
    pub enc_ms: f64,
    /// GPU span of the chunk's command buffer (host time base, s); includes its GPU-side wait for the previous chunk
    pub gpu: (f64, f64),
}

/// `pf_profile_at` result: per class (label, GPU ms min over iters, dispatches, FLOP of the class in this chunk).
#[derive(Clone, Debug, Default)]
pub struct PfProfile {
    pub p: usize,
    pub k: usize,
    pub full_ms: f64,
    pub full_wall_ms: f64,
    pub classes: Vec<(String, f64, usize, f64)>,
}

impl QwenEngine {
    /// MTP warm-up part of `prefill_mtp` for chunk (i, k): final norm → hf (not last), embed of ids[i+1..i+1+m] and the
    /// MTP layer on those rows (`prefill_mtp`'s own `encode_mtp_warm`, mode from SD_MTP_WARM_V2). `m` rows were staged
    /// in MTP stage `slot`.
    #[allow(clippy::too_many_arguments)]
    fn pf_mtp_part(&self, e: &Enc, k: usize, slot: usize, i: usize, p0: usize, last: bool, m: usize) -> Result<(), String> {
        self.encode_mtp_warm(e, k, slot, p0 + i, last, m, crate::mtp::warm_mode())
    }

    /// The engine's prompt read with per-chunk timing (see the module doc). Returns (greedy next token, chunks).
    pub fn pf_prefill_timed(&self, ids: &[u32]) -> Result<(u32, Vec<ChunkTime>), String> {
        if ids.is_empty() { return Err("pf_prefill_timed: no tokens".into()); }
        if self.pos() + ids.len() > self.opt.max_ctx { return Err("pf_prefill_timed: > max_ctx".into()); }
        let mtp = self.opt.mtp;
        self.rm_prefill_begin();
        if mtp {
            let c = self.ctrl();
            if c.mtp_len != c.pos { c.mtp_len = c.pos; }
        }
        let p0 = self.pos();
        let ch = self.chunks(ids.len());
        let mut inflight: VecDeque<(usize, CmdBuf)> = VecDeque::new();
        let mut out: Vec<ChunkTime> = Vec::with_capacity(ch.len());
        let order = self.gpu.shared_event();
        for (ci, &(i, k)) in ch.iter().enumerate() {
            let slot = ci % 2;
            if inflight.len() >= 2 {
                let (j, cb) = inflight.pop_front().unwrap();
                wait(&cb)?;
                out[j].gpu = gpu_span(&cb);
            }
            let last = ci + 1 == ch.len();
            let ts = Instant::now();
            self.reader.read_rows(&ids[i..i + k], self.stage_ptr(slot))?;
            let m = if i + k < ids.len() { k } else { k - 1 };
            if mtp && m > 0 {
                let ms = self.mtp_s().as_ref().unwrap();
                let mstage = unsafe { self.arena.base().add(ms.stage + slot * ms.slot_bytes) };
                self.reader.read_rows(&ids[i + 1..i + 1 + m], mstage)?;
            }
            let stage_ms = ts.elapsed().as_secs_f64() * 1e3;
            let te = Instant::now();
            let cb = self.gpu.command_buffer();
            if ci > 0 { cb_wait(&cb, &order, ci as u64); }
            self.encode_pass(&cb, k, slot, last as usize, false)?;
            if mtp {
                let e = Enc::new(&self.gpu, &cb, &self.buf);
                let r = self.pf_mtp_part(&e, k, slot, i, p0, last, m);
                e.end();
                r?;
            }
            cb_signal(&cb, &order, ci as u64 + 1);
            let enc_ms = te.elapsed().as_secs_f64() * 1e3;
            commit(&cb);
            out.push(ChunkTime { i, k, stage_ms, enc_ms, gpu: (0.0, 0.0) });
            inflight.push_back((ci, cb));
        }
        for (j, cb) in &inflight { wait(cb)?; out[*j].gpu = gpu_span(cb); }
        self.rm_prefill_end();
        Ok((self.ctrl_read().tok[0], out))
    }

    /// One fast middle chunk (k rows at position `p`: embed, all layers, no head, ctrl.pos += k, then the MTP warm-up
    /// part on k rows) on `e` / `e2` (two encoders, as the engine).
    fn pf_encode_mid_chunk(&self, e: &Enc, k: usize, p: usize) -> Result<(), String> {
        let c = &self.cfg;
        let io = PassIo { k, pos: p, nch_max: (p + k).div_ceil(ATTN_CHUNK).min(self.nch_max), rm: self.rm_io(false), ..self.io() };
        e.pso("embed_q4").b(0, self.sl().stage.at(0)).b(1, io.h).u(2, k as u32);
        e.threads((c.hidden / 8, k), 64);
        for l in 0..c.layers {
            encode_layer_prefill(e, c, &self.w.layers[l], Some(&self.dn()), self.kv_at(l), &io, l)?;
        }
        for _ in 0..k { e.pso("advance1").b(0, io.ctrl); e.threads((1, 1), 1); }
        Ok(())
    }

    fn pf_run_chunk(&self, k: usize, p: usize, filt: Option<&dyn Fn(usize, &str) -> bool>, phase: &Cell<u8>) -> Result<(f64, f64), String> {
        self.rewind_to(p);
        let t = Instant::now();
        let cb = self.gpu.command_buffer();
        phase.set(0);
        let e = match filt { Some(f) => Enc::with_filter(&self.gpu, &cb, &self.buf, f), None => Enc::new(&self.gpu, &cb, &self.buf) };
        let r = self.pf_encode_mid_chunk(&e, k, p);
        e.end();
        r?;
        if self.opt.mtp {
            phase.set(1);
            let e = match filt { Some(f) => Enc::with_filter(&self.gpu, &cb, &self.buf, f), None => Enc::new(&self.gpu, &cb, &self.buf) };
            let r = self.pf_mtp_part(&e, k, 0, 0, p, false, k);
            e.end();
            r?;
        }
        commit(&cb);
        wait(&cb)?;
        Ok((gpu_ms(&cb), t.elapsed().as_secs_f64() * 1e3))
    }

    /// Component profile of one fast middle chunk of `k` rows (k = opt.prefill_t normally) at kv position `p`; `ids`
    /// gives k + 1 token ids to stage (contents do not change the timing). Min GPU ms over `iters` per class.
    pub fn pf_profile_at(&self, ids: &[u32], k: usize, p: usize, iters: usize) -> Result<PfProfile, String> {
        if ids.len() < k + 1 { return Err("pf_profile_at: need k + 1 ids".into()); }
        if p + k > self.opt.max_ctx { return Err(format!("pf_profile_at: {} + {} > max_ctx", p, k)); }
        if !self.fast_prefill() || !(9..=self.tmax).contains(&k) { return Err("pf_profile_at: fast prefill chunk only".into()); }
        self.reader.read_rows(&ids[..k], self.stage_ptr(0))?;
        if self.opt.mtp {
            let ms = self.mtp_s().as_ref().unwrap();
            self.reader.read_rows(&ids[1..k + 1], unsafe { self.arena.base().add(ms.stage) })?;
        }
        // dispatch names (nothing dispatched); phase 0 = target pass, 1 = MTP warm-up part
        let phase = Cell::new(0u8);
        let names: RefCell<Vec<(u8, String)>> = RefCell::new(vec![]);
        let rec = |_: usize, n: &str| { names.borrow_mut().push((phase.get(), n.to_string())); false };
        self.pf_run_chunk(k, p, Some(&rec), &phase)?;
        let names = names.into_inner();
        let labels = pf_labels(&names);
        // FLOP per class for this chunk (GEMMs from the weight shapes; attention QK + PV over the causal keys)
        let mut flop: std::collections::HashMap<&str, f64> = Default::default();
        let kf = k as f64;
        for lw in &self.w.layers {
            let mm = |m: &crate::weights::Q4b| 2.0 * kf * m.rows as f64 * m.cols as f64;
            match &lw.mixer {
                Mixer::Dn(d) => { *flop.entry("dn_in_proj").or_default() += mm(&d.in_proj); *flop.entry("dn_out_proj").or_default() += mm(&d.out_proj); }
                Mixer::Attn(a) => {
                    *flop.entry("attn_qkv_proj").or_default() += mm(&a.in_proj);
                    *flop.entry("attn_o_proj").or_default() += mm(&a.o_proj);
                    let keys: f64 = (0..k).map(|t| (p + t + 1) as f64).sum();
                    *flop.entry("attn_flash").or_default() += 4.0 * self.cfg.heads as f64 * self.cfg.head_dim as f64 * keys;
                }
            }
            *flop.entry("ffn_gate_up").or_default() += mm(&lw.ffn.gate) + mm(&lw.ffn.up);
            *flop.entry("ffn_down").or_default() += mm(&lw.ffn.down);
        }
        let mut full = (f64::MAX, f64::MAX);
        for _ in 0..iters {
            let r = self.pf_run_chunk(k, p, None, &phase)?;
            full = (full.0.min(r.0), full.1.min(r.1));
        }
        let mut order: Vec<&str> = vec![];
        for l in &labels { if !order.contains(&l.as_str()) { order.push(l); } }
        let mut classes = vec![];
        for cl in order {
            let n = labels.iter().filter(|l| l.as_str() == cl).count();
            let cnt = Cell::new(0usize);
            let f = |_: usize, _: &str| { let i = cnt.get(); cnt.set(i + 1); labels.get(i).is_some_and(|l| l == cl) };
            let mut best = f64::MAX;
            for _ in 0..iters {
                cnt.set(0);
                best = best.min(self.pf_run_chunk(k, p, Some(&f), &phase)?.0);
            }
            classes.push((cl.to_string(), best, n, flop.get(cl).copied().unwrap_or(0.0)));
        }
        self.rewind_to(p);
        Ok(PfProfile { p, k, full_ms: full.0, full_wall_ms: full.1, classes })
    }
}

/// Class label per dispatch of a fast chunk (+ MTP warm-up part): projections told apart by their neighbours.
pub fn pf_labels(names: &[(u8, String)]) -> Vec<String> {
    let mut out = Vec::with_capacity(names.len());
    for (i, (ph, n)) in names.iter().enumerate() {
        let next = names.get(i + 1).map(|x| x.1.as_str()).unwrap_or("");
        let prev = if i > 0 { names[i - 1].1.as_str() } else { "" };
        let l = if *ph == 1 {
            let first = i == 0 || names[i - 1].0 == 0;
            if n.starts_with("bf16_gemv") { "mtp_fc" } else if n == "embed_q4" || n == "mtp_prep" || (n == "rmsnorm_b" && first) { "mtp_prep" }
            else { "mtp_layer" }
        } else if n == "embed_q4" { "embed" }
        else if n == "rmsnorm_b" { "norm" }
        else if n.starts_with("q4b_gemm_swiglu") { "ffn_gate_up" }
        else if n.starts_with("q4b_gemm_") && n.ends_with("_m0") {
            if next.starts_with("dn_") { "dn_in_proj" } else if next.starts_with("attn_") { "attn_qkv_proj" } else { "gemm_other" }
        } else if n.starts_with("q4b_gemm_") && n.ends_with("_m2") {
            if prev == "rmsnorm_gated" { "dn_out_proj" } else if prev.starts_with("attn_") { "attn_o_proj" } else if prev.starts_with("q4b_gemm_swiglu") { "ffn_down" } else { "gemm_other" }
        } else if n == "dn_conv_pf" { "dn_conv" }
        else if n == "dn_prep" { "dn_prep" }
        else if n.starts_with("dn_recur") { "dn_recur" }
        else if n == "rmsnorm_gated" { "dn_gated_norm" }
        else if n.starts_with("attn_prep") { "attn_prep_rope_kvwrite" }
        else if n.starts_with("attn_flash") || n.starts_with("attn_sdpa") { "attn_flash" }
        else if n.starts_with("act_") { "activity" }
        else if n.starts_with("rm_") { "readmask" }
        else if n == "advance1" { "advance" }
        else { "other" };
        out.push(l.to_string());
    }
    out
}
