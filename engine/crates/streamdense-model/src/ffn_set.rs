//! Per-prompt FFN set rule of the 12 GB mode (forheader, 2026-11-03): the product side of the research harness
//! (`forheader cpa-single`).
//!
//! Inputs:
//! - `T1`: a global per-neuron score [layers][inter] (mean over the T1 prompts of the per-layer normalised
//!   |a|·‖W_down‖) and the down-column norms ‖W_down‖ [layers][inter], one binary file (`T1::read` / `write`,
//!   `T1::from_research_dir` builds it from a research harness' t1 output directory, bit for bit as the harness'
//!   `global_scores`);
//! - per layer, the prompt sums the streamed prefill hands its policy (`an` = Σ|a|, `ag` = Σ|silu(gate x)|).
//!
//! Set "A" of one layer (`SetRule::layer`, the harness' `c2_mask` + `add_fetch` on one layer, same code, same tie
//! order): the top K − R neurons by (per-layer normalised an·‖W_down‖ + global score), then the top R of the rest by
//! ag, R = round(0.10 · K); returned sorted. `SetRule::initial` = global top K per layer (the set loaded at open).
//! K comes from the RAM cap (`QwenEngine::compact_keep_for_cap`), never hardcoded.
use crate::ffn_compact::CompactSet;
use std::path::Path;

const MAGIC: &[u8; 8] = b"SDFHT1\0\0";

/// Global score + down norms, f32 [layers][inter] each.
#[derive(Clone, PartialEq)]
pub struct T1 {
    pub layers: usize,
    pub inter: usize,
    pub global: Vec<f32>,
    pub down_norm: Vec<f32>,
}

impl std::fmt::Debug for T1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "T1 {{ layers: {}, inter: {} }}", self.layers, self.inter)
    }
}

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

impl T1 {
    /// Bytes of the file for these dims.
    pub fn file_bytes(layers: usize, inter: usize) -> usize { 16 + 8 * layers * inter }

    pub fn check(&self) -> Result<(), String> {
        let n = self.layers * self.inter;
        if self.layers == 0 || self.inter == 0 || self.global.len() != n || self.down_norm.len() != n {
            return Err(format!("T1: want 2 x {} x {} values, have {} + {}", self.layers, self.inter, self.global.len(), self.down_norm.len()));
        }
        if let Some(i) = self.global.iter().chain(&self.down_norm).position(|x| !x.is_finite()) {
            return Err(format!("T1: value {} is not finite", i));
        }
        Ok(())
    }

    /// The model's dims must match.
    pub fn check_dims(&self, layers: usize, inter: usize) -> Result<(), String> {
        if self.layers != layers || self.inter != inter {
            return Err(format!("T1 score file is for {} layers x {} neurons, the model has {} x {} (made for another model?)",
                               self.layers, self.inter, layers, inter));
        }
        Ok(())
    }

    /// File: magic, layers u32, inter u32, global f32 [layers * inter], down_norm f32 [layers * inter] (little endian).
    pub fn write(&self, path: &Path) -> Result<(), String> {
        self.check()?;
        let mut b = Vec::with_capacity(Self::file_bytes(self.layers, self.inter));
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&(self.layers as u32).to_le_bytes());
        b.extend_from_slice(&(self.inter as u32).to_le_bytes());
        for x in self.global.iter().chain(&self.down_norm) { b.extend_from_slice(&x.to_le_bytes()); }
        if let Some(d) = path.parent() { if !d.as_os_str().is_empty() { std::fs::create_dir_all(d).map_err(|e| format!("{}: {}", d.display(), e))?; } }
        // write next to it, then rename: a reader never sees half a file
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &b).map_err(|e| format!("{}: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
    }

    pub fn read(path: &Path) -> Result<T1, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        if b.len() < 16 || &b[..8] != MAGIC { return Err(format!("{}: not a T1 score file", path.display())); }
        let (l, n) = (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize);
        if b.len() != Self::file_bytes(l, n) { return Err(format!("{}: {} bytes, want {} for {} x {}", path.display(), b.len(), Self::file_bytes(l, n), l, n)); }
        let m = 16 + 4 * l * n;
        let t = T1 { layers: l, inter: n, global: f32s(&b[16..m]), down_norm: f32s(&b[m..]) };
        t.check().map_err(|e| format!("{}: {}", path.display(), e))?;
        Ok(t)
    }

    /// From the research T1 directory (meta.json {layers, inter, prompts: [{i}]}, pNN_total.f32, down_norm.f32): the
    /// harness' `global_scores`, same arithmetic in the same order (so the sets equal the harness' sets).
    pub fn from_research_dir(dir: &Path) -> Result<T1, String> {
        let rd = |f: &str| std::fs::read(dir.join(f)).map_err(|e| format!("{}: {}", dir.join(f).display(), e));
        let m: serde_json::Value = serde_json::from_slice(&rd("meta.json")?).map_err(|e| format!("{}/meta.json: {}", dir.display(), e))?;
        let u = |k: &str| m[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("{}/meta.json: no {}", dir.display(), k));
        let (l, inter) = (u("layers")?, u("inter")?);
        let dn = f32s(&rd("down_norm.f32")?);
        if dn.len() < l * inter { return Err(format!("{}/down_norm.f32: {} values, want {}", dir.display(), dn.len(), l * inter)); }
        let ps = m["prompts"].as_array().filter(|p| !p.is_empty()).ok_or_else(|| format!("{}/meta.json: no prompts", dir.display()))?;
        let mut g = vec![0f32; l * inter];
        for p in ps {
            let i = p["i"].as_u64().ok_or_else(|| format!("{}/meta.json: prompt without i", dir.display()))?;
            let v = f32s(&rd(&format!("p{:02}_total.f32", i))?);
            if v.len() < l * inter { return Err(format!("{}/p{:02}_total.f32: {} values, want >= {}", dir.display(), i, v.len(), l * inter)); }
            for li in 0..l {
                let r = li * inter..(li + 1) * inter;
                let t: f64 = r.clone().map(|j| v[j] as f64 * dn[j] as f64).sum();
                for j in r { g[j] += (v[j] as f64 * dn[j] as f64 / t.max(1e-30) / ps.len() as f64) as f32; }
            }
        }
        let t = T1 { layers: l, inter, global: g, down_norm: dn[..l * inter].to_vec() };
        t.check()?;
        Ok(t)
    }
}

/// Indices of the top `k` of `s` (descending, `total_cmp`) among `ix` (in place: ix[..k] after the call).
fn top_into(s: &[f32], ix: &mut [u32], k: usize) {
    if k < ix.len() { ix.select_nth_unstable_by(k, |&a, &b| s[b as usize].total_cmp(&s[a as usize])); }
}

/// The rule with its sizes: K per layer, R of them from the prompt gate.
#[derive(Clone, Debug)]
pub struct SetRule {
    pub t1: T1,
    pub k: usize,
    pub kres: usize,
}

impl SetRule {
    /// R = round(0.10 · K) (the research setting).
    pub fn reserve_for(k: usize) -> usize { (0.10 * k as f64).round() as usize }

    pub fn new(t1: T1, k: usize) -> Result<SetRule, String> {
        t1.check()?;
        if k < 8 || k % 8 != 0 || k > t1.inter { return Err(format!("set rule: K = {} (want a multiple of 8 in 8..={})", k, t1.inter)); }
        Ok(SetRule { kres: Self::reserve_for(k), t1, k })
    }

    /// Global top K per layer (sorted).
    pub fn initial(&self) -> CompactSet {
        let n = self.t1.inter;
        let keep = (0..self.t1.layers).map(|li| {
            let s = &self.t1.global[li * n..(li + 1) * n];
            let mut ix: Vec<u32> = (0..n as u32).collect();
            top_into(s, &mut ix, self.k);
            let mut k = ix[..self.k.min(n)].to_vec();
            k.sort_unstable();
            k
        }).collect();
        CompactSet { inter: n, keep }
    }

    /// Set A of layer `li` from the prompt sums (sorted, exactly K neurons).
    pub fn layer(&self, li: usize, an: &[f32], ag: &[f32]) -> Vec<u32> {
        let n = self.t1.inter;
        let r = li * n..(li + 1) * n;
        let (dn, gs) = (&self.t1.down_norm[r.clone()], &self.t1.global[r]);
        // c2: per-layer normalised an·‖W_down‖ + global score, top K − R
        let tt: f64 = (0..n).map(|j| an[j] as f64 * dn[j] as f64).sum();
        let sc: Vec<f32> = (0..n).map(|j| (an[j] as f64 * dn[j] as f64 / tt.max(1e-30)) as f32 + gs[j]).collect();
        let mut ix: Vec<u32> = (0..n as u32).collect();
        let core = self.k - self.kres;
        top_into(&sc, &mut ix, core);
        let mut m = vec![0u8; n];
        for &j in &ix[..core.min(n)] { m[j as usize] = 1; }
        // reserve: top R of the rest by the prompt gate
        ix.clear();
        ix.extend((0..n as u32).filter(|&j| m[j as usize] == 0));
        let kk = self.kres.min(ix.len());
        top_into(ag, &mut ix, kk);
        for &j in &ix[..kk] { m[j as usize] = 2; }
        (0..n as u32).filter(|&j| m[j as usize] != 0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t1(l: usize, n: usize) -> T1 {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
        T1 { layers: l, inter: n, global: (0..l * n).map(|_| r() / n as f32).collect(), down_norm: (0..l * n).map(|_| 0.5 + r()).collect() }
    }

    #[test]
    fn file_round_trip_and_errors() {
        let d = std::env::temp_dir().join(format!("sd_t1_{}", std::process::id()));
        let p = d.join("t1.sdt1");
        let a = t1(3, 64);
        a.write(&p).unwrap();
        assert_eq!(T1::read(&p).unwrap(), a);
        assert!(a.check_dims(3, 64).is_ok());
        assert!(a.check_dims(4, 64).unwrap_err().contains("another model"));
        std::fs::write(&p, b"nope").unwrap();
        assert!(T1::read(&p).unwrap_err().contains("not a T1"));
        assert!(T1::read(&d.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn research_dir_is_the_harness_mean() {
        let d = std::env::temp_dir().join(format!("sd_t1dir_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (l, n) = (2usize, 16usize);
        let w = |f: &str, v: &[f32]| std::fs::write(d.join(f), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let dn: Vec<f32> = (0..l * n).map(|j| 1.0 + j as f32 / 10.0).collect();
        w("down_norm.f32", &dn);
        // per-prompt totals with extra head values after the FFN part (as the harness writes them)
        let p0: Vec<f32> = (0..l * n + 5).map(|j| (j % 7) as f32).collect();
        let p1: Vec<f32> = (0..l * n + 5).map(|j| (j % 3) as f32 + 0.5).collect();
        w("p00_total.f32", &p0);
        w("p03_total.f32", &p1);
        std::fs::write(d.join("meta.json"), format!("{{\"layers\": {}, \"inter\": {}, \"prompts\": [{{\"i\": 0}}, {{\"i\": 3}}]}}", l, n)).unwrap();
        let t = T1::from_research_dir(&d).unwrap();
        assert_eq!((t.layers, t.inter), (l, n));
        assert_eq!(t.down_norm, dn);
        // per layer the global score sums to 1 (mean of normalised distributions)
        for li in 0..l { let s: f32 = t.global[li * n..(li + 1) * n].iter().sum(); assert!((s - 1.0).abs() < 1e-5, "{}", s); }
        std::fs::remove_file(d.join("p03_total.f32")).unwrap();
        assert!(T1::from_research_dir(&d).unwrap_err().contains("p03_total"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rule_sizes_and_reserve() {
        let (l, n) = (2usize, 256usize);
        let r = SetRule::new(t1(l, n), 96).unwrap();
        assert_eq!(r.kres, 10);
        assert!(SetRule::new(t1(l, n), 90).is_err());
        let init = r.initial();
        init.validate().unwrap();
        assert_eq!(init.counts(), vec![96; l]);
        // an concentrated on neurons 0..50, ag on 200..256: the core comes from an (+ global), the reserve from ag
        let an: Vec<f32> = (0..n).map(|j| if j < 50 { 100.0 } else { 0.0 }).collect();
        let ag: Vec<f32> = (0..n).map(|j| if j >= 200 { 1.0 + j as f32 } else { 0.0 }).collect();
        for li in 0..l {
            let k = r.layer(li, &an, &ag);
            assert_eq!(k.len(), 96);
            assert!(k.windows(2).all(|w| w[0] < w[1]));
            assert!((0..50).all(|j| k.binary_search(&j).is_ok()), "prompt-active neurons kept");
            // the 10 reserve picks are the highest-ag neurons not in the core
            let top_ag: Vec<u32> = (246..256).collect();
            assert!(top_ag.iter().all(|j| k.binary_search(j).is_ok()), "{:?}", k);
        }
        // same input, same output (deterministic tie order)
        assert_eq!(r.layer(1, &an, &ag), r.layer(1, &an, &ag));
    }
}
