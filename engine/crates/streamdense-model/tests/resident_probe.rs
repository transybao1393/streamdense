//! Probe (run by hand, `--ignored --nocapture --test-threads=1`): which way of giving pages back lowers
//! phys_footprint for a Metal shared buffer on macOS. 1 GB buffers; prints one line per case.
use objc2_foundation::NSRange;
use objc2_metal::*;
use std::ptr::NonNull;
use streamdense_core::footprint::self_footprint;
use streamdense_metal::{Buffer, Gpu};

const N: usize = 1 << 30;
const H: usize = N / 2;

fn settle() -> u64 { std::thread::sleep(std::time::Duration::from_millis(100)); self_footprint() }
fn mb(a: u64, b: u64) -> f64 { (a as f64 - b as f64) / 1e6 }

fn gpu_touch(g: &Gpu, b: &Buffer, v: u8) { gpu_fill(g, b, b.length(), v) }
fn gpu_fill(g: &Gpu, b: &Buffer, n: usize, v: u8) { objc2::rc::autoreleasepool(|_| {
    let cb = g.queue.commandBuffer().unwrap();
    let e = cb.blitCommandEncoder().unwrap();
    e.fillBuffer_range_value(b, NSRange::new(0, n), v);
    e.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}) }
fn gpu_copy(g: &Gpu, src: &Buffer, so: usize, dst: &Buffer, n: usize) { objc2::rc::autoreleasepool(|_| {
    let cb = g.queue.commandBuffer().unwrap();
    let e = cb.blitCommandEncoder().unwrap();
    unsafe { e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(src, so, dst, 0, n) };
    e.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}) }
fn madv(p: *mut u8, n: usize, a: i32) -> i32 { unsafe { libc::madvise(p as *mut libc::c_void, n, a) } }

#[test]
#[ignore]
fn probe_footprint() {
    let g = Gpu::from_shader_files(&[]).unwrap();
    let small = g.buffer(1 << 20).unwrap();
    // A: copied buffer (newBufferWithLength), as the arena today
    for (name, adv) in [("REUSABLE", libc::MADV_FREE_REUSABLE), ("DONTNEED", libc::MADV_DONTNEED), ("FREE", libc::MADV_FREE)] {
        let f0 = settle();
        let b = g.buffer(N).unwrap();
        let p = streamdense_metal::contents(&b);
        unsafe { std::ptr::write_bytes(p, 7, N) };
        gpu_touch(&g, &b, 9);
        let f1 = settle();
        let r = madv(unsafe { p.add(H) }, H, adv);
        let f2 = settle();
        eprintln!("copied buffer + {:<8}: alloc+touch {:+.1} MB, madvise(half) rc {} -> {:+.1} MB", name, mb(f1, f0), r, mb(f2, f1));
        drop(b);
        eprintln!("  drop buffer -> {:+.1} MB vs start", mb(settle(), f0));
    }
    // B: NoCopy over mmap anon
    for (name, adv) in [("REUSABLE", libc::MADV_FREE_REUSABLE), ("DONTNEED", libc::MADV_DONTNEED)] {
        let f0 = settle();
        let p = unsafe { libc::mmap(std::ptr::null_mut(), N, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) } as *mut u8;
        let b = unsafe {
            g.device.newBufferWithBytesNoCopy_length_options_deallocator(NonNull::new(p as *mut _).unwrap(), N, MTLResourceOptions::StorageModeShared, None)
        }.unwrap();
        unsafe { std::ptr::write_bytes(p, 7, N) };
        gpu_touch(&g, &b, 9);
        let f1 = settle();
        let r = madv(unsafe { p.add(H) }, H, adv);
        let f2 = settle();
        // GPU use after the advise (the other half), then re-take the freed half and check the GPU sees CPU writes
        gpu_fill(&g, &b, H, 3);
        let f3 = settle();
        let r2 = madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSE);
        unsafe { std::ptr::write_bytes(p.add(H), 0x5a, H) };
        let f4 = settle();
        gpu_copy(&g, &b, H + 12345, &small, 1 << 20);
        let seen = unsafe { std::slice::from_raw_parts(streamdense_metal::contents(&small), 1 << 20) };
        let ok = seen.iter().all(|&x| x == 0x5a);
        eprintln!("nocopy mmap + {:<8}: alloc+touch {:+.1} MB, madvise(half) rc {} -> {:+.1} MB, gpu fill after -> {:+.1} MB, REUSE rc {} + rewrite -> {:+.1} MB, gpu sees rewrite {}",
            name, mb(f1, f0), r, mb(f2, f1), mb(f3, f1), r2, mb(f4, f1), ok);
        drop(b);
        unsafe { libc::munmap(p as *mut _, N) };
        eprintln!("  drop + munmap -> {:+.1} MB vs start", mb(settle(), f0));
    }
}

/// (wired, free + speculative + inactive-file) bytes, system wide
#[allow(deprecated)]
fn sys() -> (i64, i64) {
    let mut s: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut n = libc::HOST_VM_INFO64_COUNT;
    unsafe { libc::host_statistics64(libc::mach_host_self(), libc::HOST_VM_INFO64, &mut s as *mut _ as *mut i32, &mut n) };
    let pg = 16384i64;
    (s.wire_count as i64 * pg, (s.free_count as i64 + s.speculative_count as i64) * pg)
}

#[test]
#[ignore]
fn probe_footprint2() {
    let g = Gpu::from_shader_files(&[]).unwrap();
    let line = |tag: &str, f0: u64, s0: (i64, i64)| {
        let f = settle();
        let s = sys();
        eprintln!("  {:<34} footprint {:+8.1} MB  wired {:+8.1} MB  free {:+8.1} MB", tag, mb(f, f0), (s.0 - s0.0) as f64 / 1e6, (s.1 - s0.1) as f64 / 1e6);
    };
    eprintln!("C1 plain mmap (no Metal)");
    {
        let (f0, s0) = (settle(), sys());
        let p = unsafe { libc::mmap(std::ptr::null_mut(), N, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) } as *mut u8;
        unsafe { std::ptr::write_bytes(p, 7, N) };
        line("touch", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSABLE);
        line("REUSABLE half", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSE);
        unsafe { std::ptr::write_bytes(p.add(H), 1, H) };
        line("REUSE + rewrite", f0, s0);
        unsafe { libc::munmap(p as *mut _, N) };
        line("munmap", f0, s0);
    }
    for gpu_used in [false, true] {
        eprintln!("C2 nocopy mmap, gpu used before advise: {}", gpu_used);
        let (f0, s0) = (settle(), sys());
        let p = unsafe { libc::mmap(std::ptr::null_mut(), N, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) } as *mut u8;
        let b = unsafe {
            g.device.newBufferWithBytesNoCopy_length_options_deallocator(NonNull::new(p as *mut _).unwrap(), N, MTLResourceOptions::StorageModeShared, None)
        }.unwrap();
        unsafe { std::ptr::write_bytes(p, 7, N) };
        if gpu_used { gpu_touch(&g, &b, 9); }
        line("touch", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSABLE);
        line("REUSABLE half", f0, s0);
        if gpu_used { gpu_fill(&g, &b, H, 3); line("gpu fill first half", f0, s0); }
        unsafe { std::ptr::write_bytes(p.add(H), 1, H) };
        line("rewrite (no REUSE)", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSE);
        line("REUSE", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSABLE);
        line("REUSABLE again", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSE);
        unsafe { std::ptr::write_bytes(p.add(H), 2, H) };
        line("REUSE then rewrite", f0, s0);
        objc2::rc::autoreleasepool(|_| drop(b));
        line("drop buffer", f0, s0);
        unsafe { libc::munmap(p as *mut _, N) };
        line("munmap", f0, s0);
    }
    eprintln!("D copied buffer drop");
    {
        let (f0, s0) = (settle(), sys());
        let b = objc2::rc::autoreleasepool(|_| g.buffer(N).unwrap());
        unsafe { std::ptr::write_bytes(streamdense_metal::contents(&b), 7, N) };
        line("alloc + touch", f0, s0);
        objc2::rc::autoreleasepool(|_| drop(b));
        line("drop", f0, s0);
        std::thread::sleep(std::time::Duration::from_millis(1000));
        line("drop + 1 s", f0, s0);
    }
}

/// Same as probe 2 with long settles and the system wired count as the "really returned" check.
#[test]
#[ignore]
fn probe_wired() {
    let g = Gpu::from_shader_files(&[]).unwrap();
    let slow = || { std::thread::sleep(std::time::Duration::from_millis(1500)); (self_footprint(), sys()) };
    let line = |tag: &str, f0: u64, s0: (i64, i64)| {
        let (f, s) = slow();
        eprintln!("  {:<30} footprint {:+8.1} MB  wired {:+8.1} MB", tag, mb(f, f0), (s.0 - s0.0) as f64 / 1e6);
    };
    for round in 0..2 {
        eprintln!("nocopy mmap 1 GiB (round {})", round);
        let (f0, s0) = slow();
        let p = unsafe { libc::mmap(std::ptr::null_mut(), N, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) } as *mut u8;
        unsafe { std::ptr::write_bytes(p, 7, N) };
        line("mmap + CPU touch", f0, s0);
        let b = unsafe {
            g.device.newBufferWithBytesNoCopy_length_options_deallocator(NonNull::new(p as *mut _).unwrap(), N, MTLResourceOptions::StorageModeShared, None)
        }.unwrap();
        line("wrap NoCopy", f0, s0);
        gpu_fill(&g, &b, H, 9);
        line("GPU fill first half", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSABLE);
        line("REUSABLE second half", f0, s0);
        objc2::rc::autoreleasepool(|_| drop(b));
        line("drop buffer", f0, s0);
        madv(unsafe { p.add(H) }, H, libc::MADV_FREE_REUSABLE);
        line("REUSABLE again (unwrapped)", f0, s0);
        unsafe { libc::munmap(p as *mut _, N) };
        line("munmap", f0, s0);
        eprintln!("copied buffer 1 GiB (round {})", round);
        let (f0, s0) = slow();
        let b = objc2::rc::autoreleasepool(|_| g.buffer(N).unwrap());
        line("alloc", f0, s0);
        unsafe { std::ptr::write_bytes(streamdense_metal::contents(&b), 7, N) };
        line("CPU touch", f0, s0);
        gpu_fill(&g, &b, H, 9);
        line("GPU fill first half", f0, s0);
        objc2::rc::autoreleasepool(|_| drop(b));
        line("drop", f0, s0);
    }
}
