//! Margin-gated recheck. OFF by default (`QwenOptions.gate_hold = false`: no region, no
//! pipeline, no change). For the 12 GB mode (compact FFN + streamed prefill, MTP off).
//!
//! Step at host position p, token t (the caller tracks whether the previous step's row is still pending):
//!   hold     `gate_hold_pass(t, replay_prev)`: [replay of the pending row p-1 (dn_conv / dn_prep / dn_recur_ck0 over the
//!            1-row hold save, the ctx_draft replay: bit-identical to the in-place pass) + ctrl.pos = p] then ONE exact
//!            K = 1 compact pass at p with the DeltaNet state only READ (dn_conv DN_HOLD, dn_recur_hold; row saved by
//!            dn_save_all). KV row p is written, rec[p] / logits hold the compact answer, ctrl.pos stays p.
//!   accept   margin >= tau: the next `gate_hold_pass(.., true)` replays row p first (one state read + write).
//!   recheck  margin < tau: `gate_recheck(t, full_from)` drops the hold row and redoes position p from the same past:
//!            `stream_pass([t])` (ffn_stream.rs) with layers 0..full_from on the resident set and full_from.. on their
//!            FULL FFN streamed from the model file through the existing slots (no new buffers). full_from = 0: every
//!            layer full = the full model's decode pass at p given this past. KV row p, DeltaNet state and conv are
//!            written by that pass, so the state matches whichever token the caller continues with (the input at p is t
//!            either way). Logits of the recheck: `last_logits()`.
//! RAM: 1 replay row (0.99 MB on the 27B) as a budget row; the recheck reuses the streamed prefill's slots / hrows.
use streamdense_metal::{commit, wait, Enc};

use crate::ffn_stream::StreamStats;
use crate::mtp::path_pos;
use crate::qwen_engine::{KvFormat, QwenEngine};
use crate::qwen_layers::{encode_head, encode_layer, PassIo};

impl QwenEngine {
    /// One K = 1 HOLD pass feeding `tok` (module doc). `replay_prev`: the row of the previous hold pass (at host
    /// position p) is accepted: replay it and move to p + 1 first. Returns (argmax, bf16 top-1 − top-2) of this pass;
    /// the host position stays at this pass's position (its row pending). Engine idle only.
    pub fn gate_hold_pass(&self, tok: u32, replay_prev: bool) -> Result<(u32, f32), String> {
        let hold = self.ctx_hold.ok_or("gate_hold_pass: QwenOptions.gate_hold off (no hold save)")?.off;
        self.ahead.borrow_mut().take();
        let c = &self.cfg;
        if replay_prev {
            let p = self.pos() + 1;
            self.ctrl().pos = p as u32;
            self.set_pos(p);
        }
        let p = self.pos();
        if p + 1 > self.opt.max_ctx { return Err(format!("gate_hold_pass: position {} at max_ctx", p)); }
        self.reader.read_rows(&[tok], self.stage_ptr(0))?;
        let cb = self.gpu.command_buffer();
        let e = Enc::new(&self.gpu, &cb, &self.buf);
        let r = (|| -> Result<(), String> {
            if replay_prev { self.encode_replay_from(&e, hold, 1)?; }
            let pos = path_pos(p, p, 1, self.opt.kv == KvFormat::Q8)?;
            let io = PassIo { k: 1, pos, nch_max: self.nch_for(p, 1), dn_save: None, dn_hold: Some(hold), rm: None, ..self.io() };
            e.pso("embed_q4").b(0, self.sl().stage.at(0)).b(1, io.h).u(2, 1);
            e.threads((c.hidden / 8, 1), 64);
            for l in 0..c.layers {
                encode_layer(&e, c, &self.w.layers[l], Some(&self.dn()), self.kv()[l].as_ref(), &io, false, None, l)?;
            }
            encode_head(&e, c, &self.w, &io, 1, None)
        })();
        e.end();
        r?;
        commit(&cb);
        wait(&cb)?;
        let (t, a, b) = self.rec_token(p);
        Ok((t, a - b))
    }

    /// Accept the pending hold row without a following pass (replay it, host position + 1). Engine idle only.
    pub fn gate_flush(&self) -> Result<(), String> {
        let hold = self.ctx_hold.ok_or("gate_flush: QwenOptions.gate_hold off")?.off;
        let p = self.pos() + 1;
        self.ctrl().pos = p as u32;
        self.set_pos(p);
        let cb = self.gpu.command_buffer();
        let e = Enc::new(&self.gpu, &cb, &self.buf);
        let r = self.encode_replay_from(&e, hold, 1);
        e.end();
        r?;
        commit(&cb);
        wait(&cb)
    }

    /// Redo the pending position with the full FFN in layers `full_from..` (module doc): feeds `tok` at the host
    /// position, the hold row is dropped. On return the host position is p + 1, the state is this pass's, the
    /// recheck's greedy token is `.tok` and its logits `last_logits()`. Needs the streamed prefill (`ffn_stream_tokens`).
    pub fn gate_recheck(&self, tok: u32, full_from: usize) -> Result<StreamStats, String> { self.gate_recheck_ex(tok, full_from, false) }

    /// `gate_recheck` with `missing`: each full layer's slot gets the resident set's gate / up rows copied from
    /// RAM and only the missing rows (+ gate / up S, B and the down W) read from the model file (ffn_stream.rs
    /// `load_layer_missing`). The slot bytes are the same as the full read, so the result is bit-identical; fewer bytes.
    /// default copy path: the file part is a memcpy from a read-only mmap of the model files (`SD_RECHECK_MMAP=0`: F_NOCACHE
    /// pread; pread also when mapping fails), and the resident rows are copied by the GPU at the start of each layer's
    /// command buffer (`cs_rowscatter`, keep indices in the hrows tail, no new memory; `SD_RECHECK_GPU_ROWS=0`: CPU), so
    /// the CPU loader of layer l+1 only copies the file part while the GPU runs layer l.
    /// Default (`SD_RECHECK_ZEROCOPY=0` = the copy path above; automatic fallback if mapping / buffer creation fails):
    /// zero-copy, no CPU copy at all: the full-FFN layers read gate / up / down W from no-copy Metal buffers over the
    /// mapped model file (16-byte aligned tensors bound directly, misaligned ones re-aligned into the slot by the GPU,
    /// `cs_bytecopy`); ffn_stream.rs `ZeroCopy`. Bit-identical to the copy path.
    pub fn gate_recheck_ex(&self, tok: u32, full_from: usize, missing: bool) -> Result<StreamStats, String> {
        self.ahead.borrow_mut().take();
        self.ctrl().pos = self.pos() as u32;
        self.stream_pass(&[tok], None, &mut |_, _, _| Ok(Vec::new()), full_from, missing)
    }
}
