//! In-RAM rewind checkpoints (`QwenOptions.rewind_bytes`): rewind the live sequence to an earlier position instead of
//! a reset + snapshot restore when a client rewrites an earlier message of a chat.
//!
//! The attention layers' KV (and the MTP layer's KV, and the `rec` table) are append-only: rows < p are never
//! rewritten while the sequence only grows from p, so going back to position p needs nothing for them. The Gated
//! DeltaNet layers are recurrent: their state S and conv state at position p cannot be recomputed from the KV. A
//! checkpoint therefore holds exactly what a snapshot holds besides the append-only rows: the ctrl block, the MTP
//! ctrl block (mctrl) and the live DeltaNet S + conv copy (27B: 150,994,944 + 2,949,120 B = 153.9 MB each; 0.62 GB
//! = 4 checkpoints). Save and restore are GPU blits inside the arena (no host copy of the state), once per request.
//!
//! Exactness: `ckpt_restore(i)` leaves the engine in the state it had when `ckpt_save(i)` ran (same ctrl, mctrl,
//! DeltaNet state, KV rows [0..p), MTP KV rows [0..mtp_len), rec [0..p)), i.e. the state a snapshot taken there
//! and restored now would give (snapshot.rs contract: restore + decode == the original continuation, bit for bit).
//!
//! Validity: a checkpoint is valid while the slot's sequence has only grown since it was taken. `reset`, any
//! snapshot restore and `rewind_to` start a new epoch (all checkpoints of the slot invalid); restoring checkpoint i
//! at position p invalidates every checkpoint past p (their rows get overwritten by the next feed).
use std::cell::{Cell, RefCell};

use streamdense_core::arena::Region;
use streamdense_metal::{blit_copy, commit, wait, SdCtrl};

use crate::qwen_config::QwenConfig;
use crate::qwen_engine::QwenEngine;
use crate::qwen_layers::DnState;

/// Bytes of one checkpoint's DeltaNet part (one S copy + one conv copy of every DeltaNet layer).
pub fn ckpt_bytes(cfg: &QwenConfig) -> usize {
    cfg.n_deltanet() * (DnState::s_layer_bytes(cfg, 1) + DnState::conv_layer_bytes(cfg))
}

/// Checkpoints a RAM budget of `bytes` holds (0 = off).
pub fn ckpt_count(cfg: &QwenConfig, bytes: u64) -> usize {
    bytes.checked_div(ckpt_bytes(cfg) as u64).unwrap_or(0) as usize
}

#[derive(Clone)]
struct Meta {
    pos: usize,
    slot: usize,
    /// epochs (KV lineages) in which this checkpoint is valid
    epochs: Vec<u64>,
    ctrl: SdCtrl,
    mctrl: Option<SdCtrl>,
}

/// Snapshots saved in this process remembered for re-validating checkpoints on their restore.
const SAVED_MAX: usize = 512;

/// The checkpoint region (K x `ckpt_bytes`) and the host side of every checkpoint.
///
/// Epochs: an epoch is one KV lineage (the sequence only grew within it). Ids are unique in the process. A new epoch
/// starts on reset, `rewind_to`, a snapshot restore and a checkpoint restore. A checkpoint restore at p carries the
/// checkpoints at <= p into the new epoch (their KV rows are untouched). A snapshot restore carries the checkpoints
/// valid in the epoch the snapshot was SAVED in, at <= its length: that snapshot holds the very KV rows [0..n) those
/// checkpoints were taken on (within one epoch the sequence only grows, so a checkpoint at p <= n of that epoch
/// precedes the save), so the restored state + the checkpoint's DeltaNet state == the state at p back then.
pub struct Rewind {
    pub region: Region,
    pub per: usize,
    pub k: usize,
    meta: RefCell<Vec<Option<Meta>>>,
    /// current epoch per sequence slot
    epoch: RefCell<Vec<u64>>,
    next: Cell<u64>,
    /// (length, FNV of the token ids) of each snapshot saved in this process -> the epoch it was saved in
    saved: RefCell<std::collections::HashMap<(usize, u64), u64>>,
}

impl Rewind {
    pub(crate) fn new(region: Region, per: usize, k: usize, slots: usize) -> Rewind {
        Rewind { region, per, k, meta: RefCell::new(vec![None; k]), epoch: RefCell::new((0..slots as u64).collect()), next: Cell::new(slots as u64),
                 saved: RefCell::new(std::collections::HashMap::new()) }
    }
    fn epoch(&self, slot: usize) -> u64 { self.epoch.borrow().get(slot).copied().unwrap_or(u64::MAX) }
    /// Start a new epoch on `slot`; checkpoints of that slot valid in the old one with `pos <= carry` (if any) and,
    /// with `also`, those valid in epoch `also.0` with `pos <= also.1`, stay valid.
    fn new_epoch(&self, slot: usize, carry: Option<usize>, also: Option<(u64, usize)>) {
        let old = self.epoch(slot);
        let e = self.next.get();
        self.next.set(e + 1);
        for m in self.meta.borrow_mut().iter_mut().flatten() {
            if m.slot != slot { continue; }
            let keep = carry.is_some_and(|c| m.pos <= c && m.epochs.contains(&old))
                || also.is_some_and(|(ae, n)| m.pos <= n && m.epochs.contains(&ae));
            if keep { m.epochs.push(e); if m.epochs.len() > 16 { m.epochs.remove(0); } }
        }
        if let Some(x) = self.epoch.borrow_mut().get_mut(slot) { *x = e; }
    }
}

impl QwenEngine {
    /// Rewind checkpoint slots of this engine (0 = off).
    pub fn ckpt_slots(&self) -> usize { self.rw.as_ref().map_or(0, |r| r.k) }
    /// Bytes of one checkpoint (0 = off).
    pub fn ckpt_size(&self) -> usize { self.rw.as_ref().map_or(0, |r| r.per) }

    /// A new epoch of the selected slot: every checkpoint taken on it before is invalid (reset, `rewind_to`).
    pub(crate) fn ckpt_new_epoch(&self) { if let Some(r) = &self.rw { r.new_epoch(self.cur_seq(), None, None); } }

    /// A snapshot of `tokens` (the current state) is being saved: remember the epoch it belongs to.
    pub(crate) fn ckpt_note_save(&self, tokens: &[u32]) {
        let Some(r) = &self.rw else { return };
        let mut s = r.saved.borrow_mut();
        if s.len() >= SAVED_MAX { s.clear(); }
        s.insert((tokens.len(), crate::snapshot::tokens_hash(tokens)), r.epoch(self.cur_seq()));
    }

    /// A snapshot of `tokens` was just restored: a new epoch, keeping the checkpoints of the epoch that snapshot was
    /// saved in (this process) at positions <= its length.
    pub(crate) fn ckpt_after_restore(&self, tokens: &[u32]) {
        let Some(r) = &self.rw else { return };
        let src = r.saved.borrow().get(&(tokens.len(), crate::snapshot::tokens_hash(tokens))).copied();
        r.new_epoch(self.cur_seq(), None, src.map(|e| (e, tokens.len())));
    }

    /// Position of checkpoint `i` when it is still valid on the selected slot (taken in this epoch, at or before
    /// the current position).
    pub fn ckpt_pos(&self, i: usize) -> Option<usize> {
        let r = self.rw.as_ref()?;
        let meta = r.meta.borrow();
        let m = meta.get(i)?.as_ref()?;
        (m.slot == self.cur_seq() && m.epochs.contains(&r.epoch(m.slot)) && m.pos <= self.pos()).then_some(m.pos)
    }

    /// (src, dst) blit spans between the live DeltaNet copy `role` and checkpoint `i` (`to_ckpt`: live -> checkpoint).
    fn ckpt_spans(&self, r: &Rewind, i: usize, role: usize, to_ckpt: bool) -> Vec<(usize, usize, usize)> {
        let c = &self.cfg;
        let dn = self.dn();
        let one = DnState::s_layer_bytes(c, 1);
        let ndn = c.n_deltanet();
        let base = r.region.off + i * r.per;
        let mut v = Vec::with_capacity(ndn + 1);
        for d in 0..ndn {
            let live = dn.s + d * one * dn.copies + role * one;
            let ck = base + d * one;
            v.push(if to_ckpt { (live, ck, one) } else { (ck, live, one) });
        }
        let cl = DnState::conv_layer_bytes(c) * ndn;
        let (live, ck) = (dn.conv + role * cl, base + ndn * one);
        v.push(if to_ckpt { (live, ck, cl) } else { (ck, live, cl) });
        v
    }

    fn ckpt_blit(&self, spans: &[(usize, usize, usize)]) -> Result<(), String> {
        objc2::rc::autoreleasepool(|_| {
            let cb = self.gpu.command_buffer();
            blit_copy(&cb, &self.buf, spans);
            commit(&cb);
            wait(&cb)
        })
    }

    /// Save the current state of the selected slot (position `pos()`) into checkpoint `i` (overwrites it). Between
    /// turns only (nothing in flight). Checkpoints of this slot past the current position are dropped (stale).
    pub fn ckpt_save(&self, i: usize) -> Result<(), String> {
        let r = self.rw.as_ref().ok_or("rewind checkpoints off (rewind_bytes = 0)")?;
        if i >= r.k { return Err(format!("checkpoint {} of {}", i, r.k)); }
        self.ahead.borrow_mut().take();
        let c = self.ctrl_read();
        let pos = self.pos();
        if c.pos as usize != pos { return Err(format!("checkpoint: host pos {} != ctrl.pos {} (command buffer in flight?)", pos, c.pos)); }
        let role = (c.role as usize).min(self.dn().copies - 1);
        r.meta.borrow_mut()[i] = None;
        self.ckpt_blit(&self.ckpt_spans(r, i, role, true))?;
        let mctrl = self.mtp_s().as_ref().map(|m| unsafe { std::ptr::read_volatile(self.arena.base().add(m.mctrl) as *const SdCtrl) });
        let slot = self.cur_seq();
        let ep = r.epoch(slot);
        let mut meta = r.meta.borrow_mut();
        for m in meta.iter_mut() { if m.as_ref().is_some_and(|m| m.slot == slot && m.pos > pos && m.epochs.contains(&ep)) { *m = None; } }
        meta[i] = Some(Meta { pos, slot, epochs: vec![ep], ctrl: c, mctrl });
        Ok(())
    }

    /// Go back to checkpoint `i` (valid on the selected slot): the DeltaNet state into copy 0 (role 0, as a snapshot
    /// restore), ctrl + mctrl as saved, host position = its position. Checkpoints past it are dropped. Returns the
    /// position. Nothing may be in flight.
    pub fn ckpt_restore(&self, i: usize) -> Result<usize, String> {
        let r = self.rw.as_ref().ok_or("rewind checkpoints off (rewind_bytes = 0)")?;
        let p = self.ckpt_pos(i).ok_or_else(|| format!("checkpoint {} not valid on this sequence", i))?;
        let m = r.meta.borrow()[i].clone().ok_or("checkpoint gone")?;
        self.ahead.borrow_mut().take();
        if let Err(e) = self.ckpt_blit(&self.ckpt_spans(r, i, 0, false)) {
            // the live state is half written: nothing of it may be reused
            self.reset();
            return Err(e);
        }
        let mut c = m.ctrl;
        c.role = 0;
        *self.ctrl() = c;
        if let (Some(ms), Some(mc)) = (self.mtp_s().as_ref(), m.mctrl) {
            unsafe { std::ptr::write_volatile(self.arena.base().add(ms.mctrl) as *mut SdCtrl, mc) };
        }
        self.set_pos(p);
        // a new lineage from p on: the checkpoints at <= p carry over, the later ones are left in the old epoch only
        // (valid again only if a snapshot saved in that epoch is restored)
        r.new_epoch(self.cur_seq(), Some(p), None);
        Ok(p)
    }
}
