//! File reads with F_NOCACHE (streamed / loaded weights do not fill the macOS file cache) via pread; no mmap.
use std::ffi::CString;

pub struct NoCacheFile { pub fd: i32, pub len: u64 }
unsafe impl Send for NoCacheFile {}
unsafe impl Sync for NoCacheFile {}

impl NoCacheFile {
    pub fn open(path: &str) -> Result<Self, String> {
        let c = CString::new(path).unwrap();
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY) };
        if fd < 0 { return Err(format!("cannot open {}", path)); }
        if unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) } != 0 { return Err(format!("F_NOCACHE failed on {}", path)); }
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        unsafe { libc::fstat(fd, &mut st) };
        Ok(NoCacheFile { fd, len: st.st_size as u64 })
    }
    /// pread exactly `len` bytes at `off` into `dst`
    /// `dst` must be valid for `len` writable bytes (a slot in the weight arena); kept safe-signature like the other
    /// arena-pointer helpers, so the obligation sits with the caller that computed the region.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn read_at(&self, dst: *mut u8, len: usize, off: u64) -> Result<(), String> {
        let mut done = 0usize;
        while done < len {
            let r = unsafe { libc::pread(self.fd, dst.add(done) as *mut libc::c_void, (len - done).min(1 << 30), (off + done as u64) as libc::off_t) };
            if r <= 0 { return Err(format!("pread failed at {} (+{}): {}", off, done, std::io::Error::last_os_error())); }
            done += r as usize;
        }
        Ok(())
    }
    /// Many (dst, len, off) reads split into `chunk`-byte pieces, run on `threads` threads.
    pub fn read_many(&self, jobs: &[(usize, usize, u64)], threads: usize, chunk: usize) -> Result<u64, String> {
        let mut pieces: Vec<(usize, usize, u64)> = Vec::new();
        for &(dst, len, off) in jobs {
            let mut d = 0;
            while d < len { let n = chunk.min(len - d); pieces.push((dst + d, n, off + d as u64)); d += n; }
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let err = std::sync::Mutex::new(None::<String>);
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= pieces.len() { break; }
                    let (dst, n, off) = pieces[i];
                    if let Err(e) = self.read_at(dst as *mut u8, n, off) { *err.lock().unwrap() = Some(e); break; }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap() { return Err(e); }
        Ok(jobs.iter().map(|j| j.1 as u64).sum())
    }
}
impl Drop for NoCacheFile { fn drop(&mut self) { unsafe { libc::close(self.fd); } } }
