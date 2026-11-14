//! sd-watchdog [--cap-gb X] [--margin-gb M] [--swap-gb S] [--log FILE] -- <command> [args...]
//! Runs the command and SIGKILLs it if its phys_footprint exceeds cap + margin or system swap has GROWN by more than
//! --swap-gb since the watchdog started (baseline = system swap at start, printed on the start line; before
//! 2026-11-14 it was the absolute system swap, so other programs' swap killed small models).
//! Prints the peak footprint at exit. Defaults for --cap-gb / --margin-gb / --swap-gb: [ram] in streamdense.toml.
//! SIGTERM / SIGHUP to the watchdog are forwarded to the command (`streamdense up` execs into this process,
//! so stopping its pid stops the gateway; before 2026-11-03 the gateway was left running), and it waits for its exit;
//! SIGINT only if the command still runs 3 s later (a terminal Ctrl-C already reached the whole process group).
use std::process::Command;
use std::sync::atomic::{AtomicI32, Ordering};
use streamdense_core::footprint::{over_limit, phys_footprint, swap_growth, swap_used};
use streamdense_core::{config, gb};

/// Last stop signal received (0 = none); forwarded by the poll loop.
static PENDING: AtomicI32 = AtomicI32::new(0);
extern "C" fn on_signal(sig: libc::c_int) { PENDING.store(sig, Ordering::SeqCst); }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(sep) = a.iter().position(|x| x == "--").filter(|&s| s + 1 < a.len()) else {
        eprintln!("usage: sd-watchdog [--cap-gb X] [--margin-gb M] [--swap-gb S] [--log FILE] -- cmd ...\n  \
                   --swap-gb S: kill when system swap has grown by more than S GB since the watchdog started (not absolute swap)");
        std::process::exit(2);
    };
    let conf = config::load().unwrap_or_else(|e| { eprintln!("[sd-watchdog] ERROR: {}", e); std::process::exit(2) });
    let get = |k: &str, d: f64| a[..sep].iter().position(|x| x == k).map(|i| {
        a.get(i + 1).and_then(|v| v.parse::<f64>().ok()).unwrap_or_else(|| { eprintln!("[sd-watchdog] ERROR: {} wants a number", k); std::process::exit(2) })
    }).unwrap_or(d);
    let cap = (get("--cap-gb", conf.cfg.ram.cap_gb) * 1e9) as u64;
    let margin = (get("--margin-gb", conf.cfg.ram.margin_gb) * 1e9) as u64;
    let swap_lim = (get("--swap-gb", conf.cfg.ram.swap_gb) * 1e9) as u64;
    let log = a[..sep].iter().position(|x| x == "--log").map(|i| a[i + 1].clone());
    let swap_base = swap_used();
    for s in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] { unsafe { libc::signal(s, on_signal as *const () as libc::sighandler_t); } }
    let mut child = Command::new(&a[sep + 1]).args(&a[sep + 2..]).spawn().expect("spawn");
    let pid = child.id() as i32;
    let (mut peak, mut peak_swap) = (0u64, 0u64);
    let t0 = std::time::Instant::now();
    let note = |s: String| { eprintln!("[sd-watchdog] {}", s); if let Some(l) = &log { use std::io::Write; let _ = writeln!(std::fs::OpenOptions::new().create(true).append(true).open(l).unwrap(), "{}", s); } };
    note(format!("config: {}", conf.source_str()));
    note(format!("pid {} cap {:.2} GB + margin {:.2} GB, swap growth limit {:.2} GB over baseline {:.2} GB (system swap at start)",
                 pid, gb(cap), gb(margin), gb(swap_lim), gb(swap_base)));
    let mut int_at: Option<std::time::Instant> = None;
    let status = loop {
        // SIGTERM / SIGHUP: forwarded at once. SIGINT: a terminal's Ctrl-C reaches the whole process group (the
        // command too, and a second signal makes the gateway exit without its last fold), so it is forwarded only if
        // the command is still running 3 s later (`kill -INT <pid>` from a script)
        let sig = PENDING.swap(0, Ordering::SeqCst);
        if sig == libc::SIGINT { int_at.get_or_insert(std::time::Instant::now()); }
        else if sig != 0 {
            note(format!("signal {} at {:.1} s: forwarded to pid {}", sig, t0.elapsed().as_secs_f64(), pid));
            unsafe { libc::kill(pid, sig); }
        }
        if int_at.is_some_and(|t| t.elapsed().as_secs_f64() >= 3.0) {
            int_at = None;
            note(format!("SIGINT at {:.1} s: command still running after 3 s, forwarded to pid {}", t0.elapsed().as_secs_f64(), pid));
            unsafe { libc::kill(pid, libc::SIGINT); }
        }
        if let Some(st) = child.try_wait().unwrap() { break st; }
        let f = phys_footprint(pid).unwrap_or(0);
        let sw = swap_used();
        peak = peak.max(f); peak_swap = peak_swap.max(sw);
        if over_limit(f, cap + margin, sw, swap_base, swap_lim) {
            note(format!("KILL at {:.1} s: footprint {:.3} GB, swap {:.3} GB (growth {:.3} GB over baseline)", t0.elapsed().as_secs_f64(), gb(f), gb(sw), gb(swap_growth(sw, swap_base))));
            let _ = child.kill();
            break child.wait().unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    note(format!("exit {:?} after {:.1} s; peak footprint {:.3} GB, peak swap {:.3} GB", status.code(), t0.elapsed().as_secs_f64(), gb(peak), gb(peak_swap)));
    std::process::exit(status.code().unwrap_or(1));
}
