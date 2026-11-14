//! Keep the Mac out of idle sleep while the gateway is busy: a `PreventUserIdleSystemSleep` power assertion
//! (IOKit `IOPMAssertionCreateWithName`) is held while any request (real or warm-up) is queued or running, and for
//! `LINGER` after the last one ends (the client runs its tools between requests). Display sleep is not blocked.
//! No-op on other OSes. Pure hold/release logic is `HoldState` (unit-tested); `Awake` runs it on one small thread.
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long the assertion stays after the last request ends.
pub const LINGER: Duration = Duration::from_secs(120);

/// Requests in flight + when the last one ended.
#[derive(Debug, Default, Clone)]
pub struct HoldState { active: usize, last_end: Option<Instant> }

impl HoldState {
    pub fn begin(&mut self) { self.active += 1; }
    pub fn end(&mut self, now: Instant) {
        self.active = self.active.saturating_sub(1);
        if self.active == 0 { self.last_end = Some(now); }
    }
    /// Whether the assertion should be held at `now`.
    pub fn wants_hold(&self, now: Instant, linger: Duration) -> bool {
        self.active > 0 || self.last_end.is_some_and(|t| now.saturating_duration_since(t) < linger)
    }
    /// When the hold runs out if nothing new arrives (None: requests in flight, or nothing to release).
    pub fn release_at(&self, linger: Duration) -> Option<Instant> {
        if self.active > 0 { return None; }
        self.last_end.map(|t| t + linger)
    }
}

pub struct Awake { st: Mutex<HoldState>, cv: Condvar, linger: Duration }

/// Held for the lifetime of one request handler.
pub struct AwakeGuard(Arc<Awake>);
impl Drop for AwakeGuard {
    fn drop(&mut self) {
        if let Ok(mut s) = self.0.st.lock() { s.end(Instant::now()); }
        self.0.cv.notify_all();
    }
}

impl Awake {
    /// Starts the assertion thread (it sleeps on a condvar; no polling).
    pub fn start(linger: Duration) -> Arc<Awake> {
        let a = Arc::new(Awake { st: Mutex::new(HoldState::default()), cv: Condvar::new(), linger });
        let t = a.clone();
        let _ = std::thread::Builder::new().name("awake".into()).spawn(move || t.run());
        a
    }
    pub fn begin(self: &Arc<Self>) -> AwakeGuard {
        if let Ok(mut s) = self.st.lock() { s.begin(); }
        self.cv.notify_all();
        AwakeGuard(self.clone())
    }
    fn run(&self) {
        let mut held: Option<pm::Assertion> = None;
        let Ok(mut s) = self.st.lock() else { return };
        loop {
            let now = Instant::now();
            let want = s.wants_hold(now, self.linger);
            if want && held.is_none() { held = pm::Assertion::take("streamdense gateway: request in flight"); }
            if !want { held = None; } // Drop releases
            s = match s.release_at(self.linger).filter(|_| want) {
                Some(at) => match self.cv.wait_timeout(s, at.saturating_duration_since(now)) { Ok((g, _)) => g, Err(_) => return },
                None => match self.cv.wait(s) { Ok(g) => g, Err(_) => return },
            };
        }
    }
}

#[cfg(target_os = "macos")]
mod pm {
    use std::ffi::{c_char, c_void, CString};
    type CFStringRef = *const c_void;
    const UTF8: u32 = 0x0800_0100; // kCFStringEncodingUTF8
    const LEVEL_ON: u32 = 255; // kIOPMAssertionLevelOn
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(alloc: *const c_void, s: *const c_char, enc: u32) -> CFStringRef;
        fn CFRelease(p: *const c_void);
    }
    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOPMAssertionCreateWithName(kind: CFStringRef, level: u32, name: CFStringRef, id: *mut u32) -> i32;
        fn IOPMAssertionRelease(id: u32) -> i32;
    }
    fn cfstr(s: &str) -> CFStringRef {
        let c = CString::new(s).unwrap_or_default();
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
    }
    pub struct Assertion(u32);
    impl Assertion {
        pub fn take(name: &str) -> Option<Assertion> {
            let (k, n) = (cfstr("PreventUserIdleSystemSleep"), cfstr(name)); // kIOPMAssertPreventUserIdleSystemSleep
            if k.is_null() || n.is_null() { return None; }
            let mut id = 0u32;
            let r = unsafe { IOPMAssertionCreateWithName(k, LEVEL_ON, n, &mut id) };
            unsafe { CFRelease(k); CFRelease(n); }
            if r == 0 { Some(Assertion(id)) } else { eprintln!("awake: IOPMAssertionCreateWithName failed ({:#x})", r); None }
        }
    }
    impl Drop for Assertion { fn drop(&mut self) { unsafe { IOPMAssertionRelease(self.0); } } }
}

#[cfg(not(target_os = "macos"))]
mod pm {
    pub struct Assertion;
    impl Assertion { pub fn take(_: &str) -> Option<Assertion> { None } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_release_timer() {
        let l = Duration::from_secs(120);
        let t0 = Instant::now();
        let mut s = HoldState::default();
        assert!(!s.wants_hold(t0, l), "idle at start");
        assert_eq!(s.release_at(l), None);
        s.begin(); // real request
        s.begin(); // warm-up queued behind it
        assert!(s.wants_hold(t0 + Duration::from_secs(1000), l), "held while in flight, however long");
        assert_eq!(s.release_at(l), None);
        s.end(t0 + Duration::from_secs(10));
        assert!(s.wants_hold(t0 + Duration::from_secs(500), l), "one still queued");
        s.end(t0 + Duration::from_secs(20));
        assert_eq!(s.release_at(l), Some(t0 + Duration::from_secs(140)));
        assert!(s.wants_hold(t0 + Duration::from_secs(139), l), "linger covers client tool time");
        assert!(!s.wants_hold(t0 + Duration::from_secs(140), l), "released 120 s after the last end");
        s.begin(); // next request inside the linger: no gap, timer restarts on its end
        s.end(t0 + Duration::from_secs(200));
        assert!(s.wants_hold(t0 + Duration::from_secs(319), l));
        assert!(!s.wants_hold(t0 + Duration::from_secs(320), l));
        s.end(t0 + Duration::from_secs(400)); // stray end never underflows
        assert_eq!(s.release_at(l), Some(t0 + Duration::from_secs(520)));
    }

    #[test]
    fn guard_counts_and_assertion_thread_runs() {
        let a = Awake::start(Duration::from_millis(50));
        let g1 = a.begin();
        let g2 = a.begin();
        assert_eq!(a.st.lock().unwrap().active, 2);
        drop(g1);
        drop(g2);
        let now = Instant::now();
        let s = a.st.lock().unwrap().clone();
        assert_eq!(s.active, 0);
        assert!(s.wants_hold(now, a.linger));
        assert!(!s.wants_hold(now + Duration::from_millis(60), a.linger));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn assertion_visible_to_pmset() {
        let name = format!("streamdense awake test {}", std::process::id());
        let a = pm::Assertion::take(&name).expect("IOPMAssertionCreateWithName");
        let out = |n: &str| String::from_utf8_lossy(&std::process::Command::new("pmset").args(["-g", "assertions"]).output().unwrap().stdout).contains(n);
        assert!(out(&name), "assertion listed while held");
        drop(a);
        assert!(!out(&name), "assertion gone after release");
    }
}
