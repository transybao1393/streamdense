//! `streamdense qwen spec-bench`: the two numbers behind deeper speculative decoding.
//! (a) `time_verify`: wall / GPU time of ONE verify pass over k new rows at context c, with the head (final norm,
//!     lm_head, argmax) on ALL k rows. GEMV path (k in 1 / 2 / 4 / 8, the decode / MTP-verify kernels, optional
//!     DeltaNet checkpoint after row 0 as the K = 2 MTP verify writes) or the fast-prefill GEMM path (any k). Each
//!     rep rewinds ctrl.pos to c (the DeltaNet state keeps advancing: timing only).
//! (b) `mtp_chain_accept`: plain greedy decode (k = 1 passes, same tokens as `generate`) and, at every position,
//!     the MTP head CHAINED `depth` steps: step 1 is the normal MTP draft (target hf, emb(t1)); step j >= 2 feeds
//!     the previous step's own hidden and its own draft token at the next MTP position. Two hidden conventions:
//!     `norm` (after mtp.norm, as vLLM's MTP proposer feeds it back) and `prenorm` (the MTP layer output before the
//!     norm). After each chain ctrl.mtp_len goes back to the post-step-1 value: the MTP KV holds only real rows,
//!     exactly as in the real MTP path.
use std::time::Instant;

use streamdense_metal::{commit, gpu_ms, wait};

use crate::qwen_engine::QwenEngine;

/// Per rep: (wall ms commit → completed, GPU ms).
pub type Times = (Vec<f64>, Vec<f64>);

#[derive(Default, Debug, Clone)]
pub struct ChainRun {
    /// greedy tokens; gen[0] comes from the prefill
    pub gen: Vec<u32>,
    /// per position i < n: the `depth` chained drafts (draft j predicts gen[i + j]), per hidden convention
    pub drafts_norm: Vec<Vec<u32>>,
    pub drafts_prenorm: Vec<Vec<u32>>,
    /// wall ms of each step-1 draft (mtp_first_draft: pread + CB), GPU ms of each chained step (j >= 2)
    pub first_wall_ms: Vec<f64>,
    pub chain_gpu_ms: Vec<f64>,
    /// GPU ms of each k = 1 target pass
    pub target_gpu_ms: Vec<f64>,
    pub prefill_s: f64,
    pub wall_s: f64,
}

impl QwenEngine {
    /// Time `warmup + reps` verify passes over `toks` (k = toks.len() rows) at context `c` (rewound each rep).
    pub fn time_verify(&self, c: usize, toks: &[u32], gemm: bool, ck: bool, warmup: usize, reps: usize) -> Result<Times, String> {
        let k = toks.len();
        if c + k > self.opt.max_ctx { return Err(format!("time_verify: {} + {} > max_ctx {}", c, k, self.opt.max_ctx)); }
        let (mut w, mut g) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
        for i in 0..warmup + reps {
            self.rewind_to(c);
            self.reader.read_rows(toks, self.stage_ptr(0))?;
            let cb = self.gpu.command_buffer();
            self.encode_pass_full(&cb, k, 0, k, ck, false, gemm)?;
            let t0 = Instant::now();
            commit(&cb);
            wait(&cb)?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            if i >= warmup { w.push(dt); g.push(gpu_ms(&cb)); }
        }
        self.rewind_to(c);
        Ok((w, g))
    }

    /// Correctness probe for the timing paths: fresh state, prefill `ctx`, one verify pass over `toks`; returns the
    /// argmax of each of the k rows (rec[c + t]).
    pub fn verify_rows(&self, ctx: &[u32], toks: &[u32], gemm: bool) -> Result<Vec<u32>, String> {
        self.reset();
        self.prefill(ctx)?;
        let c = self.pos();
        self.reader.read_rows(toks, self.stage_ptr(0))?;
        let cb = self.gpu.command_buffer();
        self.encode_pass_full(&cb, toks.len(), 0, toks.len(), false, false, gemm)?;
        commit(&cb);
        wait(&cb)?;
        Ok((0..toks.len()).map(|t| self.rec_token(c + t).0).collect())
    }

    /// Teacher-forced chained MTP acceptance (see the module doc): greedy decode until `n + depth` tokens (or eos /
    /// max_ctx), chains of `depth` drafts at every position i < n.
    pub fn mtp_chain_accept(&self, prompt: &[u32], n: usize, depth: usize) -> Result<ChainRun, String> {
        self.mtp_parts()?;
        if depth == 0 { return Err("mtp chain: depth 0".into()); }
        let (hn, row, shn, srow) = self.mtp_chain_offsets()?;
        let mut r = ChainRun::default();
        self.reset();
        let tp = Instant::now();
        let (mut t1, mut hf) = self.prefill_mtp(prompt)?;
        r.prefill_s = tp.elapsed().as_secs_f64();
        let t0 = Instant::now();
        r.gen.push(t1);
        while r.gen.len() < n + depth && !self.is_eos(t1) && self.pos() + depth + 1 <= self.opt.max_ctx {
            let i = r.gen.len() - 1;
            if i < n {
                // step 1 = the real MTP draft row (writes the MTP KV slot of this position); stages emb(t1) in slot 0
                let tf = Instant::now();
                let d1 = self.mtp_first_draft(t1, hf, 0)?;
                r.first_wall_ms.push(tf.elapsed().as_secs_f64() * 1e3);
                let keep = self.ctrl_read().mtp_len;
                if depth > 1 { self.mtp_save_step()?; }
                for (first_src, src, out) in [(shn, hn, &mut r.drafts_norm), (srow, row, &mut r.drafts_prenorm)] {
                    let mut ds = vec![d1];
                    let mut d = d1;
                    for j in 2..=depth {
                        let (nd, g) = self.mtp_chain_step(d, if j == 2 { first_src } else { src })?;
                        r.chain_gpu_ms.push(g);
                        ds.push(nd);
                        d = nd;
                    }
                    self.ctrl().mtp_len = keep;
                    out.push(ds);
                }
            } else {
                self.reader.read_rows(&[t1], self.stage_ptr(0))?;
            }
            let cb = self.gpu.command_buffer();
            self.encode_pass(&cb, 1, 0, 1, false)?;
            commit(&cb);
            wait(&cb)?;
            r.target_gpu_ms.push(gpu_ms(&cb));
            t1 = self.ctrl_read().tok[0];
            hf = self.io().hf;
            r.gen.push(t1);
        }
        r.wall_s = t0.elapsed().as_secs_f64();
        Ok(r)
    }
}
