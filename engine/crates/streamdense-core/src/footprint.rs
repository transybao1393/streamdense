//! phys_footprint (the number Activity Monitor / jetsam use; includes Metal shared buffers) and system swap, plus a
//! monitor thread that aborts the process above cap + margin or when system swap has GROWN by more than the swap
//! limit since the monitor started. The swap limit is a growth limit, not an absolute one: swap other programs held
//! before we started (13.6 GB on a busy Mac, 2026-11-14) does not count against a model.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub fn phys_footprint(pid: i32) -> Option<u64> {
    let mut ri: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, &mut ri as *mut _ as *mut libc::rusage_info_t) };
    if r == 0 { Some(ri.ri_phys_footprint) } else { None }
}
pub fn self_footprint() -> u64 { phys_footprint(std::process::id() as i32).unwrap_or(0) }
pub fn swap_used() -> u64 {
    let mut x: libc::xsw_usage = unsafe { std::mem::zeroed() };
    let mut n = std::mem::size_of::<libc::xsw_usage>();
    let name = std::ffi::CString::new("vm.swapusage").unwrap();
    let r = unsafe { libc::sysctlbyname(name.as_ptr(), &mut x as *mut _ as *mut libc::c_void, &mut n, std::ptr::null_mut(), 0) };
    if r == 0 { x.xsu_used } else { 0 }
}

/// System swap grown since `baseline` (the value when the monitor started); 0 when it shrank.
pub fn swap_growth(now: u64, baseline: u64) -> u64 { now.saturating_sub(baseline) }

/// The abort decision of `Monitor` / sd-watchdog: footprint above `limit`, or swap grown more than `swap_limit` over
/// `swap_base`.
pub fn over_limit(footprint: u64, limit: u64, swap_now: u64, swap_base: u64, swap_limit: u64) -> bool {
    footprint > limit || swap_growth(swap_now, swap_base) > swap_limit
}

pub struct Monitor { pub peak: Arc<AtomicU64>, stop: Arc<AtomicBool>, h: Option<std::thread::JoinHandle<()>> }
impl Monitor {
    /// Checks every `every_ms`; prints and exits(86) if footprint > limit or system swap grew by more than
    /// `swap_limit` since this call (the baseline, printed on the start line).
    pub fn start(limit: u64, swap_limit: u64, every_ms: u64) -> Monitor {
        let base = swap_used();
        eprintln!("FOOTPRINT MONITOR: start, footprint limit {:.3} GB, swap growth limit {:.3} GB over baseline {:.3} GB (system swap now)",
            crate::gb(limit), crate::gb(swap_limit), crate::gb(base));
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let h = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let f = self_footprint();
                p.fetch_max(f, Ordering::Relaxed);
                let sw = swap_used();
                if over_limit(f, limit, sw, base, swap_limit) {
                    eprintln!("FOOTPRINT MONITOR: footprint {:.3} GB (limit {:.3}), swap {:.3} GB = baseline {:.3} + growth {:.3} (growth limit {:.3}) -> abort",
                        crate::gb(f), crate::gb(limit), crate::gb(sw), crate::gb(base), crate::gb(swap_growth(sw, base)), crate::gb(swap_limit));
                    std::process::exit(86);
                }
                std::thread::sleep(std::time::Duration::from_millis(every_ms));
            }
        });
        Monitor { peak, stop, h: Some(h) }
    }
    pub fn peak(&self) -> u64 { self.peak.load(Ordering::Relaxed).max(self_footprint()) }
}
impl Drop for Monitor {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); if let Some(h) = self.h.take() { let _ = h.join(); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    const G: u64 = 1_000_000_000;

    /// swap other programs held before the start does not abort; growth past the limit does; footprint cap as before.
    #[test]
    fn swap_limit_is_growth_over_baseline() {
        // 13.6 GB system swap at start, 7 GB limit, 0.2 GB model: no abort
        assert!(!over_limit(G / 5, 16 * G, 13_600_000_000, 13_600_000_000, 7 * G));
        assert!(!over_limit(G / 5, 16 * G, 20_600_000_000, 13_600_000_000, 7 * G), "growth == limit is allowed");
        assert!(over_limit(G / 5, 16 * G, 20_700_000_000, 13_600_000_000, 7 * G), "growth 7.1 GB > 7");
        // swap shrank below the baseline: growth 0
        assert_eq!(swap_growth(2 * G, 5 * G), 0);
        assert!(!over_limit(G, 16 * G, 2 * G, 5 * G, 0));
        // from zero swap the old absolute rule and the new one agree
        assert!(over_limit(G, 16 * G, 8 * G, 0, 7 * G));
        // footprint cap unchanged
        assert!(over_limit(16 * G + 1, 16 * G, 0, 0, 7 * G));
        assert!(!over_limit(16 * G, 16 * G, 0, 0, 7 * G));
    }
}
