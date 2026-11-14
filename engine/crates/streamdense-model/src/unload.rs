//! unload PLAN (dry run, CPU only): which FFN neurons and mixer heads a usage profile (`ActivityStats`, activity recorder) could
//! drop, the read mask kernels would use (stage A), and the bytes / time / footprint either stage would gain.
//! Nothing here touches the engine; the plan is pure computation and serializes as JSON (`UnloadPlan::to_json`).
//!
//! Selection, per layer: contribution c_j = Σ|a_j| · ‖W_down[:, j]‖ (Σ|a_j| when no down norms). Units are aligned
//! blocks of `granularity` neurons [u·g, (u+1)·g); units are removed coldest first while the removed mass stays
//! <= (1 − keep_frac) · total (keep_frac >= 1, or no recorded mass: nothing cold). Heads (DeltaNet value heads,
//! attention q heads) the same way, whole heads only, on their output-norm sums. `*_free` fields give the neuron-level
//! (g = 1) answer for comparison: what alignment costs.
//!
//! Why granularity 64: gate / up are output rows (any row can go: its packed values and its own scale / bias row go
//! with it). down is the transposed case: neuron j is input column j, and the q4 scale / bias are shared by 64
//! consecutive input columns (MLX group 64). Dropping a whole aligned 64-block removes its packed nibbles and the
//! block's scale / bias column of every row, and what stays is again a regular q4 matrix (inter' = 64·hot blocks):
//! the existing q4 GEMV / swiglu kernels run on it unchanged, kept values bit-identical, no requantization. Finer
//! sets (g = 1) need either requantizing down (changes the kept neurons' numbers) or a gather-aware down kernel
//! with a per-column scale index (the partial blocks keep their scales: +320 B per kept 27B neuron). 64 is the
//! smallest granularity that needs neither; the `*_free` numbers show what g = 1 would add.
//!
//! Two stages:
//! - A, READ MASK (next): all weights stay resident; a per-domain mask makes the kernels skip cold work. RAM is
//!   unchanged; the saving is bytes not read per pass (`skip_bytes_per_pass`). Fallback = mask off (today's speed).
//!   Mask format (`LayerPlan::hot_blocks`, `head_hot`): per layer a sorted u32 list of hot 64-neuron block indices
//!   (always 64-blocks, whatever `granularity`: a block is cold only if every neuron in it is), and per layer a bit
//!   mask of hot heads (bit h = head h, u64 words). Design for the kernels (not built here):
//!   * swiglu (gate / up, rows = neurons): dispatch over hot blocks only: threadgroup / row-tile index i maps to
//!     neuron rows hot_blocks[i]·64 + r (tiles of 8 / 16 rows divide 64, so a tile never straddles blocks); cold
//!     rows are not computed and io.f of cold neurons is written as 0 (or never read, below).
//!   * down (input columns = neurons): each simdgroup walks the K dimension in 64-column groups; walk hot_blocks
//!     instead of 0..inter/64 (group index g = hot_blocks[i]; packed word offset g·8, scale / bias index g). The
//!     summation order over the kept groups changes, so output is "full model with cold neurons zeroed", not bit
//!     identical to the full model (quality bar). Twin / MMA kernels: same group walk, chunk split over the hot
//!     list length.
//!   * heads: DeltaNet / attention out_proj columns h·dv.. (dv = 128 = 2 groups, hd = 256 = 4 groups) are skipped
//!     like down blocks; in_proj rows of a cold head (v, z, b, a / q, gate) are skipped like swiglu rows, and the
//!     mixer kernels (dn_recur, rmsnorm_gated, attention) early-exit for cold heads (their output block is 0).
//!     Shared q / k (DeltaNet key heads) and kv heads are never skipped.
//!   * buffers: hot_blocks u32 [layers][≤ inter/64] + count per layer and head bits: < 20 KB on the 27B; switching
//!     the mask = rewriting those bytes between passes.
//! - B, REAL UNLOAD (later): TODO. Repacked per-profile resident weight buffers built layer by layer in the
//!   background under the cap (gate / up: hot rows; down: hot 64-blocks with their scale / bias columns; mixer
//!   in/out projections without cold heads), the originals stay in the safetensors on SSD (F_NOCACHE pread, as
//!   weights::load) for the out-of-profile fallback (stream the missing blocks for that stretch). Frees
//!   `unload_bytes` of RAM (plus per-head conv weights; the DeltaNet state of cold value heads could go too, not
//!   counted).
use serde_json::{json, Value};

use crate::activity::ActivityStats;
use crate::qwen_config::LayerKind;

/// Hard RAM cap of the 27B setup, bytes.
pub const HARD_CAP: u64 = 16_000_000_000;
/// Weight read bandwidth for the pass-time estimate (M4 Pro, bytes / s).
pub const READ_BW: f64 = 240e9;
/// Neurons per down-projection q4 group = mask block.
pub const MASK_BLOCK: usize = 64;

#[derive(Clone, Debug)]
pub struct LayerPlan {
    pub layer: usize,
    pub kind: LayerKind,
    /// cold neurons at `granularity` / at neuron level (g = 1)
    pub cold_neurons: usize,
    pub cold_neurons_free: usize,
    /// cold units (indices of `granularity` blocks), sorted
    pub cold_units: Vec<u32>,
    /// stage A mask: hot 64-neuron blocks (sorted) and hot-head bits (bit h of word h / 64)
    pub hot_blocks: Vec<u32>,
    pub head_hot: Vec<u64>,
    pub heads: usize,
    pub cold_heads: usize,
    /// share of the contribution / head mass kept (1 with no data)
    pub ffn_kept: f64,
    pub head_kept: f64,
    /// stage B bytes freed (FFN, heads) and stage A bytes not read per pass
    pub ffn_bytes: u64,
    pub head_bytes: u64,
    pub skip_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct UnloadPlan {
    pub keep_frac: f32,
    pub granularity: usize,
    pub rows: u64,
    pub layers: Vec<LayerPlan>,
    pub neurons: usize,
    pub cold_neurons: usize,
    pub cold_neurons_free: usize,
    pub dn_heads: usize,
    pub dn_heads_cold: usize,
    pub attn_heads: usize,
    pub attn_heads_cold: usize,
    /// stage B: RAM freed (FFN + heads), and what g = 1 neuron-level unload would free in the FFN
    pub ffn_bytes: u64,
    pub head_bytes: u64,
    pub unload_bytes: u64,
    pub ffn_bytes_free: u64,
    /// stage A: weight bytes not read per pass with the mask (RAM unchanged)
    pub skip_bytes_per_pass: u64,
    pub weight_bytes: u64,
    pub pass_ms_full: f64,
    pub pass_ms_saved_a: f64,
    pub pass_ms_saved_b: f64,
    pub resident_before: u64,
    pub resident_after_b: u64,
    pub cap: u64,
}

/// Bytes of one q4 row of `cols` input columns: packed values + bf16 scale and bias per group.
pub fn q4_row_bytes(cols: usize, group: usize, bits: usize) -> u64 { (cols * bits / 8 + cols.div_ceil(group) * 4) as u64 }

/// Cold mask over `mass`: aligned units of `g`, coldest first, while the removed mass <= (1 − keep) · total.
pub fn cold_mask(mass: &[f32], g: usize, keep: f64) -> Vec<bool> {
    let n = mass.len();
    let mut cold = vec![false; n];
    let total: f64 = mass.iter().map(|x| *x as f64).sum();
    if keep >= 1.0 || n == 0 || g == 0 || !(total > 0.0) { return cold; }
    let budget = (1.0 - keep.max(0.0)) * total * (1.0 + 1e-9);
    let mut units: Vec<(f64, usize)> = (0..n.div_ceil(g)).map(|u| (mass[u * g..((u + 1) * g).min(n)].iter().map(|x| *x as f64).sum(), u)).collect();
    units.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut c = 0f64;
    for (m, u) in units {
        if c + m > budget { break; }
        c += m;
        for x in &mut cold[u * g..((u + 1) * g).min(n)] { *x = true; }
    }
    cold
}

/// Aligned `group`-blocks of `mask` that are entirely cold.
fn whole_groups(mask: &[bool], group: usize) -> usize { mask.chunks(group).filter(|c| c.len() == group && c.iter().all(|x| *x)).count() }

fn kept(mass: &[f32], cold: &[bool]) -> f64 {
    let t: f64 = mass.iter().map(|x| *x as f64).sum();
    if !(t > 0.0) { return 1.0; }
    1.0 - mass.iter().zip(cold).filter(|(_, c)| **c).map(|(x, _)| *x as f64).sum::<f64>() / t
}

/// Per-head weight bytes: (stage A bytes skipped per pass, stage B bytes freed) of one head of `kind`.
pub fn head_bytes(s: &ActivityStats, kind: LayerKind) -> (u64, u64) {
    let row = q4_row_bytes(s.hidden, s.group, s.bits);
    let (in_rows, cols, extra) = match kind {
        // v + z rows, b, a rows of in_proj; dv out_proj input columns; conv weights of the head's v channels (bf16)
        LayerKind::DeltaNet => (2 * s.dn_dv + 2, s.dn_dv, (s.dn_dv * s.conv_k * 2) as u64),
        // q + gate rows of the fused q_proj; hd o_proj input columns
        LayerKind::Attention => (2 * s.head_dim, s.head_dim, 0),
    };
    let out = (s.hidden * cols * s.bits / 8) as u64 + (s.hidden * (cols / s.group) * 4) as u64;
    let a = in_rows as u64 * row + out;
    (a, a + extra)
}

/// The plan (see the module doc). `keep_frac_of_contribution` e.g. 0.995; `granularity` e.g. 64 (>= 1).
pub fn unload_plan(stats: &ActivityStats, keep_frac_of_contribution: f32, granularity: usize) -> UnloadPlan {
    let s = stats;
    let g = granularity.max(1);
    let keep = keep_frac_of_contribution as f64;
    let row = q4_row_bytes(s.hidden, s.group, s.bits);
    let per_neuron = 2 * row + (s.hidden * s.bits / 8) as u64; // gate + up rows, down packed column
    let per_group = (s.hidden * 4) as u64; // down scale + bias column of one 64-group
    let ffn_b = |m: &[bool]| m.iter().filter(|x| **x).count() as u64 * per_neuron + whole_groups(m, s.group) as u64 * per_group;
    let has = s.enabled && s.neuron.len() == s.layers * s.inter;
    let mut p = UnloadPlan { keep_frac: keep_frac_of_contribution, granularity: g, rows: s.rows, layers: Vec::new(), neurons: s.layers * s.inter,
        cold_neurons: 0, cold_neurons_free: 0, dn_heads: s.n_deltanet() * s.dn_vh, dn_heads_cold: 0, attn_heads: s.n_attention() * s.heads,
        attn_heads_cold: 0, ffn_bytes: 0, head_bytes: 0, unload_bytes: 0, ffn_bytes_free: 0, skip_bytes_per_pass: 0, weight_bytes: s.weight_bytes,
        pass_ms_full: s.weight_bytes as f64 / READ_BW * 1e3, pass_ms_saved_a: 0.0, pass_ms_saved_b: 0.0, resident_before: s.resident_bytes,
        resident_after_b: s.resident_bytes, cap: HARD_CAP };
    for l in 0..s.layers {
        let kind = s.kinds[l];
        let nh = match kind { LayerKind::DeltaNet => s.dn_vh, LayerKind::Attention => s.heads };
        let (c, hm): (Vec<f32>, Vec<f32>) = if has { (s.ffn_contrib(l), s.head_mass(l).to_vec()) } else { (vec![0.0; s.inter], vec![0.0; nh]) };
        let cm = cold_mask(&c, g, keep);
        let cf = cold_mask(&c, 1, keep);
        let hc = cold_mask(&hm, 1, keep);
        let nblk = s.inter.div_ceil(MASK_BLOCK);
        let hot_blocks: Vec<u32> = (0..nblk).filter(|&b| !cm[b * MASK_BLOCK..((b + 1) * MASK_BLOCK).min(s.inter)].iter().all(|x| *x))
            .map(|b| b as u32).collect();
        let mut head_hot = vec![0u64; nh.div_ceil(64)];
        for h in 0..nh { if !hc[h] { head_hot[h / 64] |= 1 << (h % 64); } }
        let cold_heads = hc.iter().filter(|x| **x).count();
        let (ha, hb) = head_bytes(s, kind);
        // stage A skips whole 64-blocks only (a block with any hot neuron is read in full)
        let mut hot = vec![false; nblk];
        for &b in &hot_blocks { hot[b as usize] = true; }
        let blk_mask: Vec<bool> = (0..s.inter).map(|j| !hot[j / MASK_BLOCK]).collect();
        let lp = LayerPlan { layer: l, kind, cold_neurons: cm.iter().filter(|x| **x).count(), cold_neurons_free: cf.iter().filter(|x| **x).count(),
            cold_units: (0..s.inter.div_ceil(g)).filter(|&u| cm[u * g]).map(|u| u as u32).collect(), hot_blocks, head_hot, heads: nh, cold_heads,
            ffn_kept: kept(&c, &cm), head_kept: kept(&hm, &hc), ffn_bytes: ffn_b(&cm), head_bytes: cold_heads as u64 * hb,
            skip_bytes: ffn_b(&blk_mask) + cold_heads as u64 * ha };
        p.cold_neurons += lp.cold_neurons;
        p.cold_neurons_free += lp.cold_neurons_free;
        match kind { LayerKind::DeltaNet => p.dn_heads_cold += cold_heads, LayerKind::Attention => p.attn_heads_cold += cold_heads }
        p.ffn_bytes += lp.ffn_bytes;
        p.ffn_bytes_free += ffn_b(&cf);
        p.head_bytes += lp.head_bytes;
        p.skip_bytes_per_pass += lp.skip_bytes;
        p.layers.push(lp);
    }
    p.unload_bytes = p.ffn_bytes + p.head_bytes;
    p.pass_ms_saved_a = p.skip_bytes_per_pass as f64 / READ_BW * 1e3;
    p.pass_ms_saved_b = p.unload_bytes as f64 / READ_BW * 1e3;
    p.resident_after_b = p.resident_before.saturating_sub(p.unload_bytes);
    p
}

impl UnloadPlan {
    /// JSON (bytes as integers, GB / ms rounded); `layers: false` leaves out the per-layer masks.
    pub fn to_json(&self, layers: bool) -> Value {
        let gb = |b: u64| (b as f64 / 1e7).round() / 100.0;
        let ms = |x: f64| (x * 100.0).round() / 100.0;
        let mut v = json!({"keep_frac": (self.keep_frac as f64 * 1e6).round() / 1e6, "granularity": self.granularity, "mask_block": MASK_BLOCK, "rows": self.rows,
            "neurons": self.neurons, "cold_neurons": self.cold_neurons, "cold_neurons_free": self.cold_neurons_free,
            "dn_heads": self.dn_heads, "dn_heads_cold": self.dn_heads_cold, "attn_heads": self.attn_heads, "attn_heads_cold": self.attn_heads_cold,
            "a_read_mask": {"skip_bytes_per_pass": self.skip_bytes_per_pass, "skip_gb": gb(self.skip_bytes_per_pass),
                            "pass_ms_saved": ms(self.pass_ms_saved_a), "ram_change_bytes": 0},
            "b_unload": {"ffn_bytes": self.ffn_bytes, "head_bytes": self.head_bytes, "unload_bytes": self.unload_bytes,
                         "unload_gb": gb(self.unload_bytes), "ffn_bytes_neuron_level": self.ffn_bytes_free,
                         "pass_ms_saved": ms(self.pass_ms_saved_b), "resident_after_gb": gb(self.resident_after_b),
                         "fits_cap": self.resident_after_b <= self.cap},
            "weight_gb": gb(self.weight_bytes), "pass_ms_full": ms(self.pass_ms_full), "resident_before_gb": gb(self.resident_before),
            "cap_gb": gb(self.cap), "fits_cap_now": self.resident_before <= self.cap, "read_bw_gbs": READ_BW / 1e9});
        if layers {
            v["layers"] = self.layers.iter().map(|l| json!({"layer": l.layer,
                "kind": match l.kind { LayerKind::DeltaNet => "deltanet", LayerKind::Attention => "attention" },
                "cold_neurons": l.cold_neurons, "cold_neurons_free": l.cold_neurons_free, "cold_units": l.cold_units,
                "hot_blocks": l.hot_blocks, "heads": l.heads, "cold_heads": l.cold_heads, "head_hot": l.head_hot,
                "ffn_kept": (l.ffn_kept * 1e6).round() / 1e6, "head_kept": (l.head_kept * 1e6).round() / 1e6,
                "ffn_bytes": l.ffn_bytes, "head_bytes": l.head_bytes, "skip_bytes": l.skip_bytes})).collect();
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(layers: usize, inter: usize, hidden: usize) -> ActivityStats {
        let kinds: Vec<LayerKind> = (0..layers).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        let mut s = ActivityStats { enabled: true, rows: 10, layers, hidden, inter, kinds, dn_kh: 16, dn_dk: 128, dn_vh: 48, dn_dv: 128,
            heads: 24, kv_heads: 4, head_dim: 256, conv_k: 4, group: 64, bits: 4, weight_bytes: 14_400_000_000,
            resident_bytes: 15_900_000_000, ..Default::default() };
        s.neuron = vec![1.0; layers * inter];
        s.dn_head = vec![1.0; s.n_deltanet() * s.dn_vh];
        s.attn_head = vec![1.0; s.n_attention() * s.heads];
        s.resid = vec![1.0; layers];
        s
    }

    #[test]
    fn keep_one_nothing_cold() {
        let s = stats(4, 256, 512);
        let p = unload_plan(&s, 1.0, 64);
        assert_eq!((p.cold_neurons, p.cold_neurons_free, p.dn_heads_cold, p.attn_heads_cold, p.unload_bytes, p.skip_bytes_per_pass), (0, 0, 0, 0, 0, 0));
        assert!(p.layers.iter().all(|l| l.hot_blocks == vec![0, 1, 2, 3] && l.head_hot.iter().map(|w| w.count_ones()).sum::<u32>() as usize == l.heads));
        // no data: nothing cold either
        let mut z = s.clone();
        z.neuron.iter_mut().for_each(|x| *x = 0.0);
        assert_eq!(unload_plan(&z, 0.9, 64).cold_neurons, 0);
    }

    #[test]
    fn concentrated_and_rounding() {
        let mut s = stats(1, 256, 512);
        // block 0 holds the mass; block 1 is all tiny except one hot neuron; blocks 2, 3 tiny
        for j in 0..256 { s.neuron[j] = if j < 64 { 1.0 } else { 1e-7 }; }
        s.neuron[64 + 5] = 1.0;
        s.dn_head = (0..48).map(|h| if h < 6 { 1.0 } else { 1e-7 }).collect();
        let p = unload_plan(&s, 0.995, 64);
        let l = &p.layers[0];
        assert_eq!(l.cold_units, vec![2, 3]);
        assert_eq!(l.cold_neurons, 128);
        assert_eq!(l.cold_neurons_free, 256 - 65); // neuron level also drops the 63 tiny neurons of block 1
        assert_eq!(l.hot_blocks, vec![0, 1]);
        assert_eq!(l.cold_heads, 42);
        assert_eq!(l.head_hot, vec![0b11_1111]);
        assert!(l.ffn_kept >= 0.995 && l.head_kept >= 0.995);
        // granularity 128: blocks {0,1} hot, {2,3} cold -> same 128; granularity 256: one unit, hot
        assert_eq!(unload_plan(&s, 0.995, 128).cold_neurons, 128);
        assert_eq!(unload_plan(&s, 0.995, 256).cold_neurons, 0);
        // granularity 32: cold units are 32-blocks; the stage A mask still uses whole 64-blocks
        let p32 = unload_plan(&s, 0.995, 32);
        assert_eq!(p32.layers[0].cold_neurons, 128 + 32); // block 1's second half (64+32..128) is tiny too
        assert_eq!(p32.layers[0].hot_blocks, vec![0, 1]);
    }

    #[test]
    fn byte_math_27b() {
        let s = stats(64, 17408, 5120);
        let p = unload_plan(&s, 0.0, 64); // keep nothing: everything cold
        assert_eq!(p.cold_neurons, 64 * 17408);
        // 27B FFN per neuron: gate + up rows 2 x (2560 + 80 x 4) + down 2560 packed + 320 (its share of a 64-group)
        assert_eq!(p.ffn_bytes, 64 * 17408 * 8640);
        assert_eq!(p.ffn_bytes_free, p.ffn_bytes);
        // DeltaNet value head: 258 in_proj rows x 2880 + out_proj 5120 x (64 + 2 x 4) + conv 128 x 4 x 2
        let dn: u64 = 258 * 2880 + 5120 * 72 + 1024;
        // attention q head: 512 rows x 2880 + o_proj 5120 x (128 + 4 x 4)
        let at: u64 = 512 * 2880 + 5120 * 144;
        assert_eq!(p.head_bytes, 48 * 48 * dn + 16 * 24 * at);
        assert_eq!(p.skip_bytes_per_pass, p.ffn_bytes + p.head_bytes - 48 * 48 * 1024);
        assert!((p.pass_ms_saved_b - p.unload_bytes as f64 / 240e9 * 1e3).abs() < 1e-9);
        assert_eq!(p.resident_after_b, 15_900_000_000u64.saturating_sub(p.unload_bytes));
        // a partially cold 64-group frees no scale / bias column (neuron level, g = 1)
        let mut t = stats(1, 128, 5120);
        for j in 0..128 { t.neuron[j] = if j < 3 { 1.0 } else { 0.0 }; }
        let q = unload_plan(&t, 0.995, 1);
        assert_eq!(q.layers[0].cold_neurons, 125);
        assert_eq!(q.ffn_bytes, 125 * 8320 + 20480); // block 1 whole (+ its scales), block 0: 61 neurons without scales
        assert_eq!(q.skip_bytes_per_pass, 64 * 8320 + 20480 + q.head_bytes); // A: block 1 only (heads all hot here)
        let j = q.to_json(true);
        assert_eq!(j["layers"][0]["hot_blocks"], json!([0]));
        assert_eq!(j["b_unload"]["unload_bytes"], json!(q.unload_bytes));
    }
}
