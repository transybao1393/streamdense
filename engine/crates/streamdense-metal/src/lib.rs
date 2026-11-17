//! streamdense-metal: device, shader library (the shared engine/shaders/*.metal, compiled at startup), pipelines,
//! the arena MTLBuffer and a small compute-encoder helper.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;
use std::ptr::NonNull;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type Event = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// The shared shader files, in concatenation order (same list the Swift KernelLab uses).
pub const SHADER_FILES: &[(&str, &str)] = &[
    ("misc.metal", include_str!("../../../shaders/misc.metal")),
    ("gemv_q4.metal", include_str!("../../../shaders/gemv_q4.metal")),
    ("gate.metal", include_str!("../../../shaders/gate.metal")),
    ("compact.metal", include_str!("../../../shaders/compact.metal")),
    ("sparse_up.metal", include_str!("../../../shaders/sparse_up.metal")),
    ("axpy_down.metal", include_str!("../../../shaders/axpy_down.metal")),
    ("attn_kv8.metal", include_str!("../../../shaders/attn_kv8.metal")),
    ("head_q8.metal", include_str!("../../../shaders/head_q8.metal")),
];
pub fn shader_source() -> String {
    // SD_SHADER_DIR=<dir> reads the files at run time instead (kernel iteration without a rebuild)
    if let Ok(dir) = std::env::var("SD_SHADER_DIR") {
        return SHADER_FILES.iter().map(|(n, _)| std::fs::read_to_string(format!("{}/{}", dir, n)).expect("shader file")).collect::<Vec<_>>().join("\n");
    }
    SHADER_FILES.iter().map(|(_, s)| *s).collect::<Vec<_>>().join("\n")
}

/// Qwen3.8-27B shader set, in concatenation order (common_bf16 first). gemv_bf16 + spec are the P2 (MTP) kernels;
/// gemm_q4 is the P3 prefill GEMM (U25).
pub const QWEN_SHADERS: &[&str] =
    &["common_bf16", "gemv_mma_q4", "verify_fused_q4", "gemv_q4_bf16", "norm_bf16", "deltanet", "attn_gated", "verify_mixers", "head_sample", "gemv_bf16", "spec", "gemm_q4"];

/// Directory read by `Gpu::from_shader_files`: `SD_SHADER_DIR` if set, else `engine/shaders/` of this checkout.
pub fn shader_dir() -> String {
    std::env::var("SD_SHADER_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../shaders").to_string())
}

/// Reads `<shader_dir>/<name>.metal` for each name (".metal" optional) at run time and joins them in order.
pub fn shader_source_files(names: &[&str]) -> Result<String, String> {
    let dir = shader_dir();
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        let file = if n.ends_with(".metal") { n.to_string() } else { format!("{}.metal", n) };
        let path = format!("{}/{}", dir, file);
        out.push(std::fs::read_to_string(&path).map_err(|e| format!("shader file {}: {}", path, e))?);
    }
    Ok(out.join("\n"))
}

/// GPU control block (plan 7.1); same layout as `struct SdCtrl` in `common_bf16.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SdCtrl {
    pub pos: u32,
    pub role: u32,
    pub mtp_len: u32,
    pub mtp_rows: u32,
    pub n_emit: u32,
    pub emit: [u32; 2],
    pub draft: u32,
    pub tok: [u32; 2],
    pub top1: [f32; 2],
    pub top2: [f32; 2],
    pub stop: u32,
    pub step: u32,
    pub pad: [u32; 2],
}
const _: () = {
    assert!(std::mem::size_of::<SdCtrl>() == 72);
    assert!(std::mem::align_of::<SdCtrl>() == 4);
    assert!(std::mem::offset_of!(SdCtrl, n_emit) == 16);
    assert!(std::mem::offset_of!(SdCtrl, emit) == 20);
    assert!(std::mem::offset_of!(SdCtrl, draft) == 28);
    assert!(std::mem::offset_of!(SdCtrl, tok) == 32);
    assert!(std::mem::offset_of!(SdCtrl, top1) == 40);
    assert!(std::mem::offset_of!(SdCtrl, top2) == 48);
    assert!(std::mem::offset_of!(SdCtrl, stop) == 56);
    assert!(std::mem::offset_of!(SdCtrl, step) == 60);
    assert!(std::mem::offset_of!(SdCtrl, pad) == 64);
};

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    psos: HashMap<String, Pso>,
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        Gpu::with_source(device, queue, &shader_source(), false)
    }
    /// Library built from `engine/shaders/<name>.metal` concatenated at run time (`SD_SHADER_DIR` honoured).
    /// Use `QWEN_SHADERS` for the 27B engine; kernel tests pass `["common_bf16", "<their file>"]`.
    /// Compiled with `MTLMathMode::Safe` (IEEE, no reassociation; MLX builds its kernels without fast-math), so the
    /// test libraries and the engine library share one math mode. The 7B library (`Gpu::new`) keeps the default.
    pub fn from_shader_files(names: &[&str]) -> Result<Gpu, String> {
        let src = shader_source_files(names)?;
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        Gpu::with_source(device, queue, &src, true)
    }
    fn with_source(device: Retained<ProtocolObject<dyn MTLDevice>>, queue: Retained<ProtocolObject<dyn MTLCommandQueue>>, src: &str, safe_math: bool) -> Result<Gpu, String> {
        let src = NSString::from_str(src);
        let opts = if safe_math {
            let o = MTLCompileOptions::new();
            o.setMathMode(MTLMathMode::Safe);
            Some(o)
        } else { None };
        let lib = device.newLibraryWithSource_options_error(&src, opts.as_deref()).map_err(|e| format!("shader compile: {}", e.localizedDescription()))?;
        Ok(Gpu { device, queue, lib, psos: HashMap::new() })
    }
    pub fn name(&self) -> String { self.device.name().to_string() }
    pub fn max_buffer(&self) -> usize { self.device.maxBufferLength() }
    pub fn pipeline(&mut self, name: &str) -> Result<(), String> {
        if self.psos.contains_key(name) { return Ok(()); }
        let f = self.lib.newFunctionWithName(&NSString::from_str(name)).ok_or(format!("no kernel {}", name))?;
        let p = self.device.newComputePipelineStateWithFunction_error(&f).map_err(|e| format!("pipeline {}: {}", name, e.localizedDescription()))?;
        self.psos.insert(name.to_string(), p);
        Ok(())
    }
    pub fn pso(&self, name: &str) -> &Pso { self.psos.get(name).unwrap_or_else(|| panic!("pipeline {} not built", name)) }
    pub fn buffer(&self, len: usize) -> Result<Buffer, String> {
        self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(format!("cannot allocate a {} B Metal buffer", len))
    }
    /// A buffer over existing memory, no copy (`newBufferWithBytesNoCopy`, shared storage, no deallocator: the caller keeps
    /// the memory alive and unchanged while the buffer is in use). `ptr` and `len` must be multiples of the VM page size
    /// (`page_size()`); used to wrap ranges of a read-only file mapping (the GPU reads the file pages directly).
    ///
    /// # Safety
    /// `ptr..ptr + len` must stay mapped for the buffer's lifetime.
    pub unsafe fn buffer_no_copy(&self, ptr: *mut u8, len: usize) -> Result<Buffer, String> {
        let pg = page_size();
        if ptr.is_null() || (ptr as usize) % pg != 0 || len == 0 || len % pg != 0 {
            return Err(format!("no-copy buffer: {:p} / {} B not page aligned ({} B)", ptr, len, pg));
        }
        let p = NonNull::new(ptr as *mut std::ffi::c_void).ok_or("no-copy buffer: null")?;
        unsafe { self.device.newBufferWithBytesNoCopy_length_options_deallocator(p, len, MTLResourceOptions::StorageModeShared, None) }
            .ok_or(format!("cannot wrap {} B as a no-copy Metal buffer", len))
    }
    /// An MTLResidencySet holding `bufs`, committed, `requestResidency` called and attached to the queue (every command
    /// buffer of the queue then has them resident without per-commit residency work). Keep the returned set alive;
    /// dropping it ends residency and detaches it (`ResidencySet::drop`).
    pub fn residency_set(&self, bufs: &[Buffer]) -> Result<ResidencySet, String> {
        let d = MTLResidencySetDescriptor::new();
        unsafe { d.setInitialCapacity(bufs.len()) };
        let set = self.device.newResidencySetWithDescriptor_error(&d)
            .map_err(|e| format!("residency set: {}", e.localizedDescription()))?;
        for b in bufs { set.addAllocation(ProtocolObject::from_ref(&**b)); }
        set.commit();
        set.requestResidency();
        self.queue.addResidencySet(&set);
        Ok(ResidencySet { set, queue: self.queue.clone() })
    }
    pub fn command_buffer(&self) -> CmdBuf { self.queue.commandBuffer().expect("command buffer") }
    pub fn shared_event(&self) -> Event { self.device.newSharedEvent().expect("shared event") }
}

/// A committed, resident MTLResidencySet attached to a queue (`Gpu::residency_set`).
pub struct ResidencySet {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}
impl ResidencySet {
    /// Bytes the set holds (`allocatedSize`).
    pub fn allocated_size(&self) -> u64 { self.set.allocatedSize() }
}
impl Drop for ResidencySet {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.set);
        self.set.endResidency();
    }
}

/// VM page size (16 KB on Apple silicon): alignment of `Gpu::buffer_no_copy`.
pub fn page_size() -> usize {
    extern "C" { fn getpagesize() -> std::ffi::c_int; }
    unsafe { getpagesize() as usize }
}

/// External buffers of an encoder (`Enc::with_ext`): an offset >= `EXT_BASE` passed to `Enc::b` binds
/// `ext[(off - EXT_BASE) >> EXT_SHIFT]` at offset `off & EXT_MASK` instead of the arena, so weight regions can point into
/// other buffers (e.g. no-copy buffers over the model file) without changing the encode code that takes offsets.
pub const EXT_BASE: usize = 1 << 60;
pub const EXT_SHIFT: u32 = 38;
pub const EXT_MASK: usize = (1 << EXT_SHIFT) - 1;
/// Offset that `Enc::b` resolves to byte `inner` of external buffer `idx` (inner < 256 GB).
pub fn ext_off(idx: usize, inner: usize) -> usize { debug_assert!(inner <= EXT_MASK); EXT_BASE + (idx << EXT_SHIFT) + inner }

pub fn contents(b: &Buffer) -> *mut u8 { b.contents().as_ptr() as *mut u8 }

pub fn sz(w: usize, h: usize, d: usize) -> MTLSize { MTLSize { width: w, height: h, depth: d } }

/// Bench-only dispatch filter: called once per `pso` with (dispatch ordinal in this encoder, kernel name); false
/// skips that dispatch (its bindings are still set, harmlessly). Lets a bench time subsets of the real encode path.
pub type EncFilter<'a> = &'a dyn Fn(usize, &str) -> bool;

/// Compute encoder bound to the arena: buffers are (arena, offset).
pub struct Enc<'a> {
    pub e: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    gpu: &'a Gpu,
    arena: &'a Buffer,
    ext: &'a [Buffer],
    filt: Option<EncFilter<'a>>,
    n: std::cell::Cell<usize>,
    on: std::cell::Cell<bool>,
}
impl<'a> Enc<'a> {
    pub fn new(gpu: &'a Gpu, cb: &CmdBuf, arena: &'a Buffer) -> Enc<'a> {
        Enc { e: cb.computeCommandEncoder().expect("encoder"), gpu, arena, ext: &[], filt: None, n: std::cell::Cell::new(0), on: std::cell::Cell::new(true) }
    }
    /// Bench only: an encoder whose dispatches pass through `f` (see `EncFilter`).
    pub fn with_filter(gpu: &'a Gpu, cb: &CmdBuf, arena: &'a Buffer, f: EncFilter<'a>) -> Enc<'a> {
        Enc { filt: Some(f), ..Enc::new(gpu, cb, arena) }
    }
    /// An encoder that also resolves `ext_off` offsets to `ext` (see `EXT_BASE`).
    pub fn with_ext(gpu: &'a Gpu, cb: &CmdBuf, arena: &'a Buffer, ext: &'a [Buffer]) -> Enc<'a> {
        Enc { ext, ..Enc::new(gpu, cb, arena) }
    }
    pub fn pso(&self, name: &str) -> &Self {
        if let Some(f) = self.filt {
            let i = self.n.get();
            self.n.set(i + 1);
            self.on.set(f(i, name));
        }
        self.e.setComputePipelineState(self.gpu.pso(name));
        self
    }
    pub fn b(&self, i: usize, off: usize) -> &Self {
        if off >= EXT_BASE {
            let x = &self.ext[(off - EXT_BASE) >> EXT_SHIFT]; // panics on an ext offset without `with_ext` (a bug)
            unsafe { self.e.setBuffer_offset_atIndex(Some(x), off & EXT_MASK, i) };
        } else {
            unsafe { self.e.setBuffer_offset_atIndex(Some(self.arena), off, i) };
        }
        self
    }
    pub fn u(&self, i: usize, v: u32) -> &Self {
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i) }; self
    }
    pub fn f(&self, i: usize, v: f32) -> &Self {
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i) }; self
    }
    /// setBytes of any plain value (e.g. a small param struct) at index i.
    pub fn bytes<T: Copy>(&self, i: usize, v: &T) -> &Self {
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(v).cast(), std::mem::size_of::<T>(), i) }; self
    }
    /// Dynamic threadgroup memory `[[threadgroup(i)]]` of the following dispatches (bytes, multiple of 16).
    pub fn tgmem(&self, i: usize, bytes: usize) -> &Self { unsafe { self.e.setThreadgroupMemoryLength_atIndex(bytes, i) }; self }
    pub fn groups(&self, g: (usize, usize), t: usize) {
        if !self.on.get() { return; }
        self.e.dispatchThreadgroups_threadsPerThreadgroup(sz(g.0, g.1, 1), sz(t, 1, 1));
    }
    pub fn threads(&self, g: (usize, usize), t: usize) {
        if !self.on.get() { return; }
        self.e.dispatchThreads_threadsPerThreadgroup(sz(g.0, g.1, 1), sz(t, 1, 1));
    }
    pub fn indirect(&self, off: usize, t: usize) {
        if !self.on.get() { return; }
        unsafe { self.e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(self.arena, off, sz(t, 1, 1)) }
    }
    pub fn end(self) { self.e.endEncoding(); }
}

/// Encode one blit pass on `cb` copying `(src, dst, bytes)` ranges inside `buf` (no other encoder may be open on `cb`).
pub fn blit_copy(cb: &CmdBuf, buf: &Buffer, spans: &[(usize, usize, usize)]) {
    let e = cb.blitCommandEncoder().expect("blit encoder");
    for &(s, d, n) in spans {
        if n > 0 { unsafe { e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(buf, s, buf, d, n) }; }
    }
    e.endEncoding();
}

pub fn gpu_ms(cb: &CmdBuf) -> f64 { (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 }
/// (GPUStartTime, GPUEndTime) in seconds, host time base (gaps between command buffers = GPU idle on this queue).
pub fn gpu_span(cb: &CmdBuf) -> (f64, f64) { (cb.GPUStartTime(), cb.GPUEndTime()) }
/// The command buffer finished (completed or error); does not block.
pub fn cb_done(cb: &CmdBuf) -> bool { cb.status().0 >= MTLCommandBufferStatus::Completed.0 }
pub fn wait(cb: &CmdBuf) -> Result<(), String> {
    cb.waitUntilCompleted();
    if let Some(e) = cb.error() { return Err(format!("GPU error: {}", e.localizedDescription())); }
    Ok(())
}
pub fn commit(cb: &CmdBuf) { cb.commit(); }

/// Encode a signal of `ev` to `value` on `cb` (no encoder may be open on `cb`).
pub fn cb_signal(cb: &CmdBuf, ev: &Event, value: u64) {
    cb.encodeSignalEvent_value(ProtocolObject::from_ref(&**ev), value);
}
/// `ev.signaledValue` (CPU read of a shared event).
pub fn ev_value(ev: &Event) -> u64 { ev.signaledValue() }
/// `ev.signaledValue = v` (CPU-side signal; values must not decrease).
pub fn ev_set(ev: &Event, v: u64) { ev.setSignaledValue(v) }
/// Encode a GPU wait until `ev` reaches `value` on `cb` (no encoder may be open on `cb`).
pub fn cb_wait(cb: &CmdBuf, ev: &Event, value: u64) {
    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&**ev), value);
}
