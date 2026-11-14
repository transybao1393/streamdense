//! Counting global allocator. Binaries install it with `#[global_allocator]`; `set_limit` arms a CPU heap limit
//! (abort on any allocation that would cross it), used after loading so the decode loop cannot grow the heap.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);
thread_local! {
    /// allocations made by this thread (const init: reading it never allocates)
    static TA: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
/// Allocations made by the calling thread so far (counts with `Counting` installed; 0 otherwise).
pub fn thread_allocs() -> u64 { TA.try_with(|c| c.get()).unwrap_or(0) }

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = TA.try_with(|n| n.set(n.get() + 1));
        let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        if c > LIMIT.load(Ordering::Relaxed) {
            LIMIT.store(usize::MAX, Ordering::Relaxed);
            eprintln!("CPU heap limit exceeded: {} B in use", c);
            std::process::abort();
        }
        PEAK.fetch_max(c, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { CUR.fetch_sub(l.size(), Ordering::Relaxed); System.dealloc(p, l) }
}
pub fn heap_now() -> usize { CUR.load(Ordering::Relaxed) }
pub fn heap_peak() -> usize { PEAK.load(Ordering::Relaxed) }
/// limit = current use + `extra` bytes
pub fn set_limit_extra(extra: usize) { LIMIT.store(heap_now() + extra, Ordering::Relaxed); }
pub fn clear_limit() { LIMIT.store(usize::MAX, Ordering::Relaxed); }
/// Move an armed limit by `delta` bytes (a budgeted heap block, e.g. the gateway's hot snapshot copies: raise before
/// allocating, lower after freeing). No limit armed (usize::MAX) = no-op.
pub fn shift_limit(delta: isize) {
    let _ = LIMIT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |l| {
        if l == usize::MAX { None } else { Some(if delta >= 0 { l.saturating_add(delta as usize).min(usize::MAX - 1) } else { l.saturating_sub(delta.unsigned_abs()) }) }
    });
}
