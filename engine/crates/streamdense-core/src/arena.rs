//! One fixed region of memory (backed by one MTLBuffer, storageModeShared), sub-allocated once at startup.
//! Nothing ever grows it: an allocation that does not fit is an error, so the GPU side cannot exceed the cap.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub off: usize,
    pub len: usize,
}
impl Region {
    pub fn at(&self, extra: usize) -> usize { self.off + extra }
}

pub struct Arena {
    base: *mut u8,
    size: usize,
    used: usize,
    pub table: Vec<(String, Region)>,
}
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

pub const ALIGN: usize = 16384; // page size on Apple silicon: keeps F_NOCACHE reads into the arena page aligned

impl Arena {
    /// `base` must stay valid for `size` bytes for the arena's lifetime (the MTLBuffer contents pointer).
    pub fn new(base: *mut u8, size: usize) -> Self { Arena { base, size, used: 0, table: Vec::new() } }
    pub fn alloc(&mut self, name: &str, len: usize) -> Result<Region, String> {
        let align = if len >= ALIGN { ALIGN } else { 256 };
        let off = (self.used + align - 1) / align * align;
        if off + len > self.size {
            return Err(format!("arena full: '{}' needs {} B, {} B free of {}", name, len, self.size.saturating_sub(off), self.size));
        }
        self.used = off + len.max(16);
        let r = Region { off, len };
        self.table.push((name.to_string(), r));
        Ok(r)
    }
    pub fn size(&self) -> usize { self.size }
    pub fn used(&self) -> usize { self.used }
    pub fn base(&self) -> *mut u8 { self.base }
    pub fn ptr(&self, r: Region) -> *mut u8 { unsafe { self.base.add(r.off) } }
    // The arena is one shared CPU/GPU buffer handed out as disjoint `Region`s; callers hold at most one live slice per
    // region (the GPU writes the same bytes between command buffers), so `&self -> &mut` is by design here.
    #[allow(clippy::mut_from_ref)]
    pub fn slice_mut<T>(&self, r: Region) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.base.add(r.off) as *mut T, r.len / std::mem::size_of::<T>()) }
    }
    /// Sum of allocations whose name starts with `prefix`.
    pub fn total(&self, prefix: &str) -> usize { self.table.iter().filter(|(n, _)| n.starts_with(prefix)).map(|(_, r)| r.len).sum() }
}
