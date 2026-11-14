//! Named tensor capture for layer-by-layer bisection against the MLX reference (plan 9.3).
//! `capture` encodes a `copy_bytes` dispatch from a live arena offset into the next free slot of a dump region;
//! after the command buffer completes, `write` reads the slots back from the shared arena into one safetensors file.
use std::cell::{Cell, RefCell};

use streamdense_core::arena::{Arena, Region};
use streamdense_core::{write_safetensors, DType};
use streamdense_metal::Enc;

const SLOT_ALIGN: usize = 16;
const TG: usize = 256;

struct Slot {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    off: usize, // absolute arena offset
    nbytes: usize,
}

pub struct Dumper {
    region: Region,
    /// Layers whose `L{l}.*` captures are kept; empty = all layers. Non-layer names (`emb`, `hf`, `mtp.*`, ...) are always kept.
    enabled_layers: Vec<usize>,
    cursor: Cell<usize>, // bytes used within `region`
    slots: RefCell<Vec<Slot>>,
}

/// `L{l}.rest` -> Some(l); anything else -> None.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix('L')?;
    let dot = rest.find('.')?;
    rest[..dot].parse().ok()
}

impl Dumper {
    pub fn new(region: Region, enabled_layers: Vec<usize>) -> Dumper {
        Dumper { region, enabled_layers, cursor: Cell::new(0), slots: RefCell::new(Vec::new()) }
    }

    pub fn enabled(&self, name: &str) -> bool {
        match layer_of(name) {
            Some(l) => self.enabled_layers.is_empty() || self.enabled_layers.contains(&l),
            None => true,
        }
    }

    /// Encode a copy of `prod(shape) * dtype.size()` bytes at arena offset `src_off` into the next dump slot.
    /// Skipped (Ok) for layers not enabled; Err on region overflow or a duplicate name. The caller's encoder
    /// pipeline state is changed, so callers set their own `.pso` before the next dispatch (as every encode does).
    pub fn capture(&self, e: &Enc, name: &str, src_off: usize, dtype: DType, shape: &[usize]) -> Result<(), String> {
        if !self.enabled(name) { return Ok(()); }
        let nbytes = shape.iter().product::<usize>() * dtype.size();
        if nbytes == 0 { return Err(format!("dump '{}': empty shape {:?}", name, shape)); }
        if nbytes > u32::MAX as usize { return Err(format!("dump '{}': {} B exceeds u32", name, nbytes)); }
        let mut slots = self.slots.borrow_mut();
        if slots.iter().any(|s| s.name == name) { return Err(format!("dump '{}' captured twice", name)); }
        let start = (self.cursor.get() + SLOT_ALIGN - 1) / SLOT_ALIGN * SLOT_ALIGN;
        if start + nbytes > self.region.len {
            return Err(format!("dump region full: '{}' needs {} B, {} B free of {}", name, nbytes,
                               self.region.len.saturating_sub(start), self.region.len));
        }
        let dst = self.region.at(start);
        e.pso("copy_bytes").b(0, src_off).b(1, dst).u(2, nbytes as u32).threads((nbytes, 1), TG);
        self.cursor.set(start + nbytes);
        slots.push(Slot { name: name.to_string(), dtype, shape: shape.to_vec(), off: dst, nbytes });
        Ok(())
    }

    pub fn len(&self) -> usize { self.slots.borrow().len() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn names(&self) -> Vec<String> { self.slots.borrow().iter().map(|s| s.name.clone()).collect() }

    /// Forget all captures (reuse the region for another pass).
    pub fn reset(&self) {
        self.cursor.set(0);
        self.slots.borrow_mut().clear();
    }

    /// Read every slot back from the shared arena and write them as one safetensors file.
    /// Call only after the command buffer that holds the captures has completed.
    pub fn write(&self, arena: &Arena, path: &str) -> Result<(), String> {
        let slots = self.slots.borrow();
        if self.region.off + self.region.len > arena.size() {
            return Err(format!("dump region {:?} outside arena of {} B", self.region, arena.size()));
        }
        let data: Vec<&[u8]> = slots
            .iter()
            .map(|s| unsafe { std::slice::from_raw_parts(arena.base().add(s.off) as *const u8, s.nbytes) })
            .collect();
        let tensors: Vec<(&str, DType, &[usize], &[u8])> = slots
            .iter()
            .zip(data.iter())
            .map(|(s, d)| (s.name.as_str(), s.dtype, s.shape.as_slice(), *d))
            .collect();
        write_safetensors(path, &tensors)
    }
}
