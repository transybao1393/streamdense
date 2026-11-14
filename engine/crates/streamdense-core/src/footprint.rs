//! phys_footprint (the number Activity Monitor / jetsam use; includes Metal shared buffers) and system swap, plus a
//! monitor thread that aborts the process above cap + margin or above a swap limit.
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

pub struct Monitor { pub peak: Arc<AtomicU64>, stop: Arc<AtomicBool>, h: Option<std::thread::JoinHandle<()>> }
impl Monitor {
    /// Checks every `every_ms`; prints and exits(86) if footprint > limit or swap > swap_limit.
    pub fn start(limit: u64, swap_limit: u64, every_ms: u64) -> Monitor {
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let h = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let f = self_footprint();
                p.fetch_max(f, Ordering::Relaxed);
                let sw = swap_used();
                if f > limit || sw > swap_limit {
                    eprintln!("FOOTPRINT MONITOR: footprint {:.3} GB (limit {:.3}), swap {:.3} GB (limit {:.3}) -> abort",
                        crate::gb(f), crate::gb(limit), crate::gb(sw), crate::gb(swap_limit));
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
