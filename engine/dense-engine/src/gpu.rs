//! Minimal Metal harness for the dense-engine kernels (same objc2 / objc2-metal versions as the engine). Mirrors the
//! shape of streamdense-metal's `Gpu` / `Enc`: one shared arena buffer, kernels bound as (arena, offset), compiled
//! with `MTLMathMode::Safe` as the engine's `Gpu::from_shader_files`.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;
use std::ptr::NonNull;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// GPU control block; same layout as `struct SdCtrl` in engine/shaders/common_bf16.metal (72 B). Only `pos` is read
/// by the attention kernels.
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
const _: () = assert!(std::mem::size_of::<SdCtrl>() == 72);

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    psos: HashMap<String, Pso>,
}

impl Gpu {
    /// Library from the given sources joined with newlines (e.g. `crate::SHADER_SOURCE`), safe math mode.
    pub fn from_sources(srcs: &[&str]) -> Result<Gpu, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let src = NSString::from_str(&srcs.join("\n"));
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        let lib = device.newLibraryWithSource_options_error(&src, Some(&opts))
            .map_err(|e| format!("shader compile: {}", e.localizedDescription()))?;
        Ok(Gpu { device, queue, lib, psos: HashMap::new() })
    }
    /// Build the pipeline `key` (once). A key is a kernel name, optionally followed by function constant values:
    /// `name#s=<f32 bits hex>,c=<f32 bits hex>,w=<u32>` (any subset; see `fc_key`): s = attention scale
    /// (function_constant 20), c = score soft cap (21), w = sliding window (22), as declared in attn_hd128.metal.
    /// A plain name specializes with no constant defined (the kernels' defaults).
    pub fn pipeline(&mut self, key: &str) -> Result<(), String> {
        if self.psos.contains_key(key) { return Ok(()); }
        let (name, consts) = key.split_once('#').unwrap_or((key, ""));
        let vals = MTLFunctionConstantValues::new();
        for kv in consts.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or(format!("pipeline key {}: bad constant {:?}", key, kv))?;
            let bad = || format!("pipeline key {}: bad value {:?}", key, kv);
            let (idx, ty, bits) = match k {
                "s" => (20usize, MTLDataType::Float, u32::from_str_radix(v, 16).map_err(|_| bad())?),
                "c" => (21, MTLDataType::Float, u32::from_str_radix(v, 16).map_err(|_| bad())?),
                "w" => (22, MTLDataType::UInt, v.parse::<u32>().map_err(|_| bad())?),
                _ => return Err(format!("pipeline key {}: unknown constant {:?}", key, k)),
            };
            // SAFETY: 4 bytes of a live u32 (f32 bits or u32), copied by Metal; ty matches the declared constant type.
            unsafe { vals.setConstantValue_type_atIndex(NonNull::from(&bits).cast(), ty, idx) };
        }
        let f = self.lib.newFunctionWithName_constantValues_error(&NSString::from_str(name), &vals)
            .map_err(|e| format!("kernel {}: {}", key, e.localizedDescription()))?;
        let p = self.device.newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| format!("pipeline {}: {}", key, e.localizedDescription()))?;
        self.psos.insert(key.to_string(), p);
        Ok(())
    }
    pub fn pso(&self, name: &str) -> &Pso { self.psos.get(name).unwrap_or_else(|| panic!("pipeline {} not built", name)) }
    pub fn buffer(&self, len: usize) -> Result<Buffer, String> {
        self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(format!("cannot allocate a {} B Metal buffer", len))
    }
    pub fn command_buffer(&self) -> CmdBuf { self.queue.commandBuffer().expect("command buffer") }
    /// Largest single MTLBuffer the device allows.
    pub fn max_buffer(&self) -> usize { self.device.maxBufferLength() }
}

/// Pipeline key of kernel `name` with the attention function constants that are set (`Gpu::pipeline` syntax): scale,
/// score soft cap, sliding window. All `None` = the plain name.
pub fn fc_key(name: &str, scale: Option<f32>, softcap: Option<f32>, window: Option<u32>) -> String {
    let mut parts: Vec<String> = vec![];
    if let Some(s) = scale { parts.push(format!("s={:08x}", s.to_bits())); }
    if let Some(c) = softcap { parts.push(format!("c={:08x}", c.to_bits())); }
    if let Some(w) = window { parts.push(format!("w={}", w)); }
    if parts.is_empty() { name.to_string() } else { format!("{}#{}", name, parts.join(",")) }
}

/// Bump allocator over one shared MTLBuffer (256 B aligned offsets) with typed copy-in / copy-out.
pub struct Arena {
    pub buf: Buffer,
    pub size: usize,
    used: usize,
}
impl Arena {
    pub fn new(gpu: &Gpu, size: usize) -> Result<Arena, String> { Ok(Arena { buf: gpu.buffer(size)?, size, used: 0 }) }
    pub fn alloc(&mut self, len: usize) -> usize {
        let off = (self.used + 255) / 256 * 256;
        assert!(off + len <= self.size, "arena full ({} + {} > {})", off, len, self.size);
        self.used = off + len;
        off
    }
    fn base(&self) -> *mut u8 { self.buf.contents().as_ptr() as *mut u8 }
    pub fn put<T: Copy>(&self, off: usize, v: &[T]) {
        let n = std::mem::size_of_val(v);
        assert!(off + n <= self.size);
        // SAFETY: shared-storage buffer of `size` bytes, range checked above; no GPU work is in flight while the
        // CPU writes (pipelined decode writes only before its first commit; everything else waits on every command buffer).
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, self.base().add(off), n) }
    }
    /// Bytes allocated so far (end of the last region).
    pub fn used(&self) -> usize { self.used }
    /// Mutable view of `len` bytes at `off` (e.g. to read weights straight from a file into the buffer).
    pub fn bytes_mut(&mut self, off: usize, len: usize) -> &mut [u8] {
        assert!(off + len <= self.size);
        // SAFETY: shared-storage buffer of `size` bytes, range checked above; `&mut self` keeps other CPU views out
        // and no GPU work is in flight while the CPU writes (callers wait on every command buffer).
        unsafe { std::slice::from_raw_parts_mut(self.base().add(off), len) }
    }
    pub fn get<T: Copy + Default>(&self, off: usize, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        let bytes = n * std::mem::size_of::<T>();
        assert!(off + bytes <= self.size);
        // SAFETY: shared-storage buffer, range checked above. Pipelined decode reads rec[p] while the next pass may
        // run; that pass writes other bytes only (rec[p + 1], its own rows), so the read sees completed data.
        unsafe { std::ptr::copy_nonoverlapping(self.base().add(off), v.as_mut_ptr() as *mut u8, bytes) }
        v
    }
}

/// Compute encoder bound to the arena: buffers are (arena, offset).
pub struct Enc<'a> {
    pub e: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    gpu: &'a Gpu,
    arena: &'a Buffer,
}
impl<'a> Enc<'a> {
    pub fn new(gpu: &'a Gpu, cb: &CmdBuf, arena: &'a Buffer) -> Enc<'a> {
        Enc { e: cb.computeCommandEncoder().expect("encoder"), gpu, arena }
    }
    pub fn pso(&self, name: &str) -> &Self { self.e.setComputePipelineState(self.gpu.pso(name)); self }
    pub fn b(&self, i: usize, off: usize) -> &Self {
        // SAFETY: offset into the arena buffer; kernels index within the regions the caller allocated.
        unsafe { self.e.setBuffer_offset_atIndex(Some(self.arena), off, i) };
        self
    }
    pub fn u(&self, i: usize, v: u32) -> &Self {
        // SAFETY: 4 bytes copied from a live u32.
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i) };
        self
    }
    pub fn f(&self, i: usize, v: f32) -> &Self {
        // SAFETY: 4 bytes copied from a live f32.
        unsafe { self.e.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i) };
        self
    }
    pub fn groups(&self, g: (usize, usize), t: usize) {
        self.e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: g.0, height: g.1, depth: 1 }, MTLSize { width: t, height: 1, depth: 1 });
    }
    pub fn threads(&self, g: (usize, usize), t: usize) {
        self.e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: g.0, height: g.1, depth: 1 }, MTLSize { width: t, height: 1, depth: 1 });
    }
    pub fn end(self) { self.e.endEncoding(); }
}

/// Encode with `f` and commit without waiting (the GPU runs it while the CPU encodes the next one); pair with `wait`.
/// Command buffers of the one queue run in commit order, and the arena is a hazard-tracked buffer, so a later
/// command buffer sees every write of an earlier one.
pub fn commit(gpu: &Gpu, arena: &Arena, f: impl FnOnce(&Enc)) -> CmdBuf {
    let cb = gpu.command_buffer();
    let e = Enc::new(gpu, &cb, &arena.buf);
    f(&e);
    e.end();
    cb.commit();
    cb
}

/// Wait for a committed command buffer; error if it failed.
pub fn wait(cb: &CmdBuf) -> Result<(), String> {
    cb.waitUntilCompleted();
    if let Some(err) = cb.error() { return Err(format!("GPU error: {}", err.localizedDescription())); }
    Ok(())
}

/// Encode with `f`, commit, wait; error if the command buffer failed.
pub fn run(gpu: &Gpu, arena: &Arena, f: impl FnOnce(&Enc)) -> Result<(), String> {
    wait(&commit(gpu, arena, f))
}
