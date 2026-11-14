//! FFN-SELECT TEST HOOK: per-neuron FFN selection for quality-ceiling tests. OFF by default
//! (`QwenOptions.ffn_select = false`: no arena region, no shader, no dispatch, outputs unchanged bit for bit).
//! With it on, after the swiglu of every model layer (exact passes: decode, exact prefill chunks, teacher; not the
//! MTP layer, not fast prefill) `ffn_select.metal` keeps a set of neurons of io.f and zeroes the rest, for rows at
//! absolute position >= `start` only (the prompt can stay unmasked). Modes (`FselMode`): All (record only), Static
//! (CPU mask per layer and neuron), Oracle (top-K per layer by |a|·‖W_down col‖ of the same row, exact in-pass),
//! Gate (top-K by |silu(gate x)| of the same row; the hook runs an extra gate GEMV into its scratch), Fetch (static
//! mask plus the top-K non-mask neurons by |silu(gate x)|: resident set + per-token fetch; cov[3] = fetched count), GateIn (top-K of the
//! static-mask neurons by |silu(gate x)|: per-token selection inside the resident set). The last row of
//! each pass records the kept set, the per-neuron scores and [kept mass, total mass, kept count, fetched count] per layer.
//! No bytes are skipped (all rows are still read and computed): quality only, no speed.
//! RAM (arena): prm 16 + 4·layers B, static mask + sel u8 [layers][inter], dn + sc f32 [layers][inter], cov f32
//! [layers][4], gate scratch bf16 [tmax][inter], gate scores f32 [layers][inter] (~18 MB on the 27B).
use crate::qwen_config::QwenConfig;
use crate::qwen_engine::QwenEngine;
use crate::qwen_layers::PassIo;
use streamdense_metal::Enc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FselMode { All = 0, Static = 1, Oracle = 2, Gate = 3, Fetch = 4, GateIn = 5 }

/// Arena offsets of the hook region.
#[derive(Clone, Copy, Debug)]
pub struct FselIo { pub prm: usize, pub mask: usize, pub dn: usize, pub sel: usize, pub sc: usize, pub cov: usize, pub gscr: usize, pub gs: usize }

const TG: usize = 1024;

/// (region bytes, offsets relative to the region start) for `tmax` rows per pass.
pub fn layout(cfg: &QwenConfig, tmax: usize) -> (usize, FselIo) {
    let (l, n) = (cfg.layers, cfg.inter);
    let a16 = |x: usize| x.div_ceil(16) * 16;
    let prm = 0;
    let mask = a16(16 + 4 * l);
    let dn = a16(mask + l * n);
    let sel = a16(dn + 4 * l * n);
    let sc = a16(sel + l * n);
    let cov = a16(sc + 4 * l * n);
    let gscr = a16(cov + 16 * l);
    let gs = a16(gscr + 2 * tmax * n);
    (a16(gs + 4 * l * n), FselIo { prm, mask, dn, sel, sc, cov, gscr, gs })
}
pub fn at_offset(io: FselIo, base: usize) -> FselIo {
    FselIo { prm: io.prm + base, mask: io.mask + base, dn: io.dn + base, sel: io.sel + base, sc: io.sc + base, cov: io.cov + base, gscr: io.gscr + base, gs: io.gs + base }
}

/// After swiglu of model layer `l` (no-op without the hook or for the MTP layer).
pub(crate) fn encode(e: &Enc, cfg: &QwenConfig, io: &PassIo, ff: &crate::weights::FfnW, l: usize) {
    let Some(s) = io.fsel.filter(|_| l < cfg.layers) else { return };
    let n = cfg.inter;
    // gate x for mode Gate (always computed with the hook on: the mode lives in GPU memory)
    crate::qwen_layers::gemv_m(e, io.gemv, &ff.gate, io.k, 0, io.x, s.gscr, None);
    e.pso("ffn_select").b(0, io.f).b(1, s.gscr).b(2, s.dn + 4 * l * n).b(3, s.mask + l * n).b(4, s.sel + l * n).b(5, s.sc + 4 * l * n)
        .b(6, s.cov + 16 * l).b(7, s.prm).u(8, n as u32).u(9, io.k as u32).u(10, l as u32).u(11, io.pos as u32).b(12, s.gs + 4 * l * n)
        .groups((io.k, 1), TG);
}

impl QwenEngine {
    fn fsel_io(&self) -> Result<FselIo, String> { self.io().fsel.ok_or_else(|| "ffn_select: hook off (QwenOptions.ffn_select)".to_string()) }
    // views of the engine-owned arena region, written only while the engine is idle (as Arena::slice_mut)
    #[allow(clippy::mut_from_ref)]
    fn fsel_slice<T>(&self, off: usize, n: usize) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.arena.base().add(off) as *mut T, n) } }
    /// Mode, first masked absolute position, and K per layer (Oracle / Gate). Engine idle only.
    pub fn fsel_set(&self, mode: FselMode, start: usize, keep: &[u32]) -> Result<(), String> {
        let s = self.fsel_io()?;
        if keep.len() != self.cfg.layers { return Err(format!("ffn_select: keep has {} layers", keep.len())); }
        let p = self.fsel_slice::<u32>(s.prm, 4 + self.cfg.layers);
        p[0] = mode as u32;
        p[1] = start as u32;
        p[4..].copy_from_slice(keep);
        Ok(())
    }
    /// Static mask [layers][inter] (1 = keep), written in place. Engine idle only.
    pub fn fsel_mask(&self) -> Result<&mut [u8], String> { let s = self.fsel_io()?; Ok(self.fsel_slice(s.mask, self.cfg.layers * self.cfg.inter)) }
    /// Per-neuron score weights (‖W_down col‖) [layers][inter]. Engine idle only.
    pub fn fsel_dn(&self) -> Result<&mut [f32], String> { let s = self.fsel_io()?; Ok(self.fsel_slice(s.dn, self.cfg.layers * self.cfg.inter)) }
    /// Last row of the last pass: kept set, scores before masking, [kept mass, total mass, kept count, 1] per layer.
    pub fn fsel_sel(&self) -> Result<&[u8], String> { let s = self.fsel_io()?; Ok(self.fsel_slice(s.sel, self.cfg.layers * self.cfg.inter)) }
    pub fn fsel_scores(&self) -> Result<&[f32], String> { let s = self.fsel_io()?; Ok(self.fsel_slice(s.sc, self.cfg.layers * self.cfg.inter)) }
    /// Last row of the last pass: |silu(gate x)| per layer and neuron.
    pub fn fsel_gate_scores(&self) -> Result<&[f32], String> { let s = self.fsel_io()?; Ok(self.fsel_slice(s.gs, self.cfg.layers * self.cfg.inter)) }
    pub fn fsel_cov(&self) -> Result<&[f32], String> { let s = self.fsel_io()?; Ok(self.fsel_slice(s.cov, self.cfg.layers * 4)) }

    /// unsure-token test: one synchronous decode pass feeding `tok` at the current position (the same pass as
    /// `feed_stream`'s decode loop). Returns (argmax token, bf16 top-1 − top-2 logit). Works with the hook on or off.
    pub fn fsel_step(&self, tok: u32) -> Result<(u32, f32), String> {
        self.reader.read_rows(&[tok], self.stage_ptr(0))?;
        let cb = self.decode_cb(0)?;
        streamdense_metal::commit(&cb);
        self.encode_ahead(0)?; // decode_fused: the next step's pass is encoded while this one runs
        streamdense_metal::wait(&cb)?;
        let (t, a, b) = self.rec_token(self.pos() - 1);
        Ok((t, a - b))
    }
    /// forced-token scoring: the f32 logits row 0 of the last 1-row pass (`fsel_step`), lm_head rows wide.
    /// Valid until the next pass runs. Engine idle only.
    pub fn last_logits(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.arena.base().add(self.io().logits) as *const f32, self.w.lm_head.rows) }
    }
    /// the per-layer FFN inputs (post-norm x, bf16 bits [layers][hidden]) of the last K = 1 pass; None when
    /// `QwenOptions.ffn_x_capture` is off. Valid until the next pass runs. Engine idle only.
    pub fn ffn_x_capture(&self) -> Option<&[u16]> {
        self.io().fxcap.map(|o| unsafe { std::slice::from_raw_parts(self.arena.base().add(o) as *const u16, self.cfg.layers * self.cfg.hidden) })
    }
    /// row 0 of the residual stream after the last layer (pre final norm, bf16 bits [hidden]) of the last 1-row
    /// pass. Valid until the next pass runs. Engine idle only.
    pub fn last_hidden(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.arena.base().add(self.io().h) as *const u16, self.cfg.hidden) }
    }
    /// copy the live DeltaNet state (S + conv, plain mode) into `buf` (resized once, reused). Engine idle only.
    pub fn fsel_dn_save(&self, buf: &mut Vec<u8>) {
        let (s, c) = (self.arena.slice_mut::<u8>(self.dn_s()), self.arena.slice_mut::<u8>(self.dn_conv()));
        buf.resize(s.len() + c.len(), 0);
        buf[..s.len()].copy_from_slice(s);
        buf[s.len()..].copy_from_slice(c);
    }
    /// put back a `fsel_dn_save` copy and move the position to `pos` (redo the passes from there: KV rows >= pos
    /// and rec are rewritten by the next pass). Plain mode only (no MTP). Engine idle only.
    pub fn fsel_dn_restore(&self, buf: &[u8], pos: usize) {
        let (s, c) = (self.arena.slice_mut::<u8>(self.dn_s()), self.arena.slice_mut::<u8>(self.dn_conv()));
        s.copy_from_slice(&buf[..s.len()]);
        c.copy_from_slice(&buf[s.len()..]);
        self.rewind_to(pos);
    }
}
