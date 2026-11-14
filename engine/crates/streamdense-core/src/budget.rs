//! Startup budget: every part of the arena is listed before anything is allocated; if the sum (plus the process
//! reserve) exceeds the cap, refuse with the shortfall.

pub struct Budget {
    pub cap: u64,
    pub reserve: u64,
    pub items: Vec<(String, u64)>,
    /// the RAM cap enforced live on the real footprint (sd-watchdog + --abort-gb); 0 = none. Shown in `report` as
    /// "the cap"; `cap` stays the startup estimate limit (the estimate includes a pessimistic process reserve).
    pub live_cap: u64,
}
impl Budget {
    pub fn new(cap: u64, reserve: u64) -> Self { Budget { cap, reserve, items: Vec::new(), live_cap: 0 } }
    pub fn add(&mut self, name: &str, bytes: u64) { self.items.push((name.to_string(), bytes)); }
    pub fn arena_bytes(&self) -> u64 {
        // each item rounded up to the arena page alignment
        self.items.iter().map(|(_, b)| (b + crate::arena::ALIGN as u64 - 1) / crate::arena::ALIGN as u64 * crate::arena::ALIGN as u64).sum()
    }
    pub fn check(&self) -> Result<u64, String> {
        let need = self.arena_bytes() + self.reserve;
        if need > self.cap {
            return Err(format!("RAM cap {:.3} GB too small: need {:.3} GB (arena {:.3} + reserve {:.3}), short by {:.3} GB\n{}",
                crate::gb(self.cap), crate::gb(need), crate::gb(self.arena_bytes()), crate::gb(self.reserve), crate::gb(need - self.cap), self.report()));
        }
        Ok(self.arena_bytes())
    }
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (n, b) in &self.items { s += &format!("  {:<28} {:>9.3} GB\n", n, crate::gb(*b)); }
        let cap = if self.live_cap > 0 {
            format!("cap {:.3} GB, enforced live on the real footprint; startup estimate limit {:.3} GB", crate::gb(self.live_cap), crate::gb(self.cap))
        } else { format!("cap {:.3} GB", crate::gb(self.cap)) };
        s += &format!("  {:<28} {:>9.3} GB\n  {:<28} {:>9.3} GB ({})", "process reserve", crate::gb(self.reserve), "total",
            crate::gb(self.arena_bytes() + self.reserve), cap);
        s
    }
}
