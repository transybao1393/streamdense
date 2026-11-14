//! Pruned draft vocabulary: the keep-list file of `streamdense vocab keep` (`QwenOptions.draft_vocab`). The MTP
//! draft head computes logits only for these lm_head rows (head_pruned.metal, gathered, no weight copy); the verify
//! pass keeps the full vocab, so output is unchanged (a removed token can never be drafted: lower acceptance only).
//! File: magic "SDKEEP01", u32 LE model vocab, u32 LE count n, then n u32 LE token ids (strictly ascending, < vocab).
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"SDKEEP01";
const HDR: usize = 16;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeepList {
    pub vocab: u32,
    pub ids: Vec<u32>,
}

impl KeepList {
    pub fn new(vocab: u32, mut ids: Vec<u32>) -> Result<KeepList, String> {
        ids.sort_unstable();
        ids.dedup();
        let k = KeepList { vocab, ids };
        k.check()?;
        Ok(k)
    }
    fn check(&self) -> Result<(), String> {
        if self.ids.is_empty() { return Err("keep-list: no ids".into()); }
        if self.ids.windows(2).any(|w| w[0] >= w[1]) { return Err("keep-list: ids not strictly ascending".into()); }
        if let Some(&l) = self.ids.last() { if l >= self.vocab { return Err(format!("keep-list: id {} >= vocab {}", l, self.vocab)); } }
        Ok(())
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HDR + 4 * self.ids.len());
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&self.vocab.to_le_bytes());
        b.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for i in &self.ids { b.extend_from_slice(&i.to_le_bytes()); }
        b
    }
    pub fn from_bytes(b: &[u8]) -> Result<KeepList, String> {
        if b.len() < HDR || &b[..8] != MAGIC { return Err("keep-list: bad magic (want SDKEEP01)".into()); }
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (vocab, n) = (u(8), u(12) as usize);
        if b.len() != HDR + 4 * n { return Err(format!("keep-list: {} bytes, header says {} ids", b.len(), n)); }
        let k = KeepList { vocab, ids: (0..n).map(|i| u(HDR + 4 * i)).collect() };
        k.check()?;
        Ok(k)
    }
    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("{}: {}", path.display(), e))
    }
    pub fn read(path: &Path) -> Result<KeepList, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        KeepList::from_bytes(&b).map_err(|e| format!("{}: {}", path.display(), e))
    }
    /// Read and check against the model vocab.
    pub fn read_for(path: &Path, vocab: usize) -> Result<KeepList, String> {
        let k = KeepList::read(path)?;
        if k.vocab as usize != vocab { return Err(format!("{}: keep-list for vocab {}, model has {}", path.display(), k.vocab, vocab)); }
        Ok(k)
    }
}
