//! DFlash2 candidate selector, CPU (z-lab `CandidateSelector.select`, dflash/model_mlx.py and model.py, temperature 0).
//!
//! LOSSLESS: the selector only chooses which drafts the target verifies. The exact target verify decides acceptance and
//! every emitted token is the verify pass's own argmax, so the selector can change the acceptance rate, never the output.
//! That is also why a failed codebook read here degrades to the plain logit choice instead of failing the round.
//!
//! Walk, per position t = 0 .. rows - 1, with a = the previous selected token (t = 0: the ANCHOR, `inputs[:, 0]`):
//!   candidates  top `top_k` tokens of the row's logits;
//!   score(b)    logits[t][b] + sum_r pred[a][r] * proj(hidden[t])[r] * succ[b][r];
//!   choice      argmax of score (temperature 0); the choice becomes the next a.
//! `hidden` and `logits` are the block rows after the anchor (`logits_start = 1`): `hidden` is the drafter's output AFTER
//! its final norm (the same rows the lm_head reads), `logits` that head's output, full vocab.
//!
//! Tie-break: candidate set = highest logits, a tie at the boundary keeps the lower token id; choice = highest score, a tie
//! takes the lowest token id. Deterministic: one thread, fixed summation order. z-lab takes the first maximum (torch
//! documents this for argmax) in candidate order, but that order comes from `argpartition` / `topk(sorted=False)` and is
//! unspecified, so z-lab has no stable token-level rule to copy; this one is a canonical instance of it. NaN never wins.
//!
//! Differences from z-lab (none changes the formula): arithmetic in f32 where MLX on this file likely uses bf16 (its
//! dequantize output should follow the bf16 scales; not run), so near-ties can resolve differently; codebook rows read
//! from the file per candidate rather than held in memory; greedy only, no sampling (the engine verifies at greedy).
//!
//! File: the community 4-bit export (models/dflash2_27b_q4), where BOTH codebooks and the projection are MLX affine g64:
//! W u32 [rows][cols / 8] (low nibble first), S and B bf16 [rows][cols / 64], value = nibble * S[group] + B[group]. A file
//! with bf16 codebooks is refused, not guessed at.
//!
//! Memory: resident = the packed projection (rank x hidden: 737,280 B on the 27B) + per-pick scratch (see `SelCfg::bytes`).
//! The codebooks (71.5 MB in the file) are not held: one pick reads rows x (1 + top_k) rows of 144 B each with `pread`, so
//! only the touched pages enter the OS file cache (reclaimable; not counted in `bytes`, which is what the process owns).
//! Cost per pick, by count: rows x vocab logit scans, rows x rank x hidden MACs, rows x (1 + top_k) x 3 preads. Time: no
//! number yet (never run).
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use streamdense_core::{bf16_to_f32, DType, SafeIndex, TensorInfo};

pub const CONFIG_FILE: &str = "config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// On-disk layout this reader implements; anything else is refused.
const Q_BITS: usize = 4;
const Q_GROUP: usize = 64;
/// Values per packed u32 word (32 bits / Q_BITS) and bf16 bytes per S or B value.
const WORD_VALS: usize = 8;
const BF: usize = 2;
const PREFIX: &str = "candidate_selector.";

/// Selector shape, all from the drafter's config.json.
#[derive(Clone, Debug, PartialEq)]
pub struct SelCfg {
    pub vocab: usize,
    pub hidden: usize,
    pub rank: usize,
    pub top_k: usize,
    /// Most rows one pick is meant to see: block_size - 1 (only sizes `bytes`, the path buffer).
    pub max_rows: usize,
}

impl SelCfg {
    pub fn from_dir(dir: &Path) -> Result<SelCfg, String> {
        let p = dir.join(CONFIG_FILE);
        let txt = std::fs::read_to_string(&p).map_err(|e| format!("{}: {}", p.display(), e))?;
        let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", p.display(), e))?;
        SelCfg::from_json(&v)
    }

    /// Hard refusal unless the config is a DFlash2 drafter quantized as affine, 4 bits, group 64.
    pub fn from_json(v: &serde_json::Value) -> Result<SelCfg, String> {
        let is_d2 = v["architectures"].as_array().is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlash2DraftModel")));
        if !is_d2 { return Err("selector: config architectures has no DFlash2DraftModel".into()); }
        let num = |o: &serde_json::Value, n: &str| o[n].as_u64().map(|x| x as usize).ok_or_else(|| format!("selector: config needs {}", n));
        let d = &v["dflash_config"];
        for key in ["quantization", "quantization_config"] {
            let q = &v[key];
            if key == "quantization_config" && q.is_null() { continue; }
            if num(q, "bits")? != Q_BITS || num(q, "group_size")? != Q_GROUP || q["mode"].as_str() != Some("affine") {
                return Err(format!("selector: {} must be affine, {} bits, group {}: {}", key, Q_BITS, Q_GROUP, q));
            }
        }
        let c = SelCfg {
            vocab: num(v, "vocab_size")?,
            hidden: num(v, "hidden_size")?,
            rank: num(d, "selector_rank")?,
            top_k: num(d, "selector_top_k")?,
            max_rows: num(d, "block_size")?.saturating_sub(1),
        };
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<(), String> {
        if self.vocab == 0 || self.vocab > u32::MAX as usize { return Err(format!("selector: bad vocab {}", self.vocab)); }
        if self.top_k == 0 || self.top_k > self.vocab { return Err(format!("selector: top_k {} not in 1..={}", self.top_k, self.vocab)); }
        if self.rank == 0 || self.rank % Q_GROUP != 0 { return Err(format!("selector: rank {} not a multiple of {}", self.rank, Q_GROUP)); }
        if self.hidden == 0 || self.hidden % Q_GROUP != 0 { return Err(format!("selector: hidden {} not a multiple of {}", self.hidden, Q_GROUP)); }
        Ok(())
    }

    /// Bytes of one packed row with `cols` values: (W, S, B).
    fn packed(cols: usize) -> (usize, usize, usize) { (cols / WORD_VALS * 4, cols / Q_GROUP * BF, cols / Q_GROUP * BF) }

    /// Process memory the selector holds or allocates per pick; codebooks excluded (not held, see the file comment).
    pub fn bytes(&self) -> u64 {
        let (pw, ps, pb) = SelCfg::packed(self.hidden);
        let (cw, cs, cb) = SelCfg::packed(self.rank);
        let resident = self.rank * (pw + ps + pb);
        let scratch = self.top_k * 8 + self.top_k * 4 + 3 * self.rank * 4 + self.hidden / Q_GROUP * 4 + cw + cs + cb + self.max_rows * 4;
        (resident + scratch) as u64
    }
}

/// File location of one packed codebook.
#[derive(Clone, Copy, Debug)]
struct Book { w: u64, s: u64, b: u64 }

pub struct Selector {
    cfg: SelCfg,
    file: std::fs::File,
    pred: Book,
    succ: Book,
    /// hidden_projection, packed: W [rank][hidden / 8], S and B [rank][hidden / 64]
    pw: Vec<u32>,
    ps: Vec<u16>,
    pb: Vec<u16>,
    io_errors: AtomicU64,
}

/// One pick's work buffers.
struct Scratch {
    cand: Vec<(f32, u32)>,
    edges: Vec<f32>,
    pred: Vec<f32>,
    y: Vec<f32>,
    row: Vec<f32>,
    gsum: Vec<f32>,
    raw: Vec<u8>,
}

impl Selector {
    /// Exact name, dtype and shape checks on all nine tensors; any other tensor under `candidate_selector.` is refused too.
    pub fn load(dir: &Path, cfg: &SelCfg) -> Result<Selector, String> {
        cfg.check()?;
        let path = dir.join(WEIGHTS_FILE).display().to_string();
        let idx = SafeIndex::open_file(&path)?;
        let want = |n: &str, dt: DType, shape: [usize; 2]| -> Result<TensorInfo, String> {
            let t = idx.get(n).ok_or_else(|| format!("selector: {}: no tensor {}", path, n))?;
            if t.dtype != dt || t.shape != shape {
                return Err(format!("selector: {}: {} is {} {:?}, need {} {:?}", path, n, t.dtype.name(), t.shape, dt.name(), shape));
            }
            Ok(t.clone())
        };
        let triple = |n: &str, rows: usize, cols: usize| -> Result<[TensorInfo; 3], String> {
            Ok([
                want(&format!("{}{}.weight", PREFIX, n), DType::U32, [rows, cols / WORD_VALS])?,
                want(&format!("{}{}.scales", PREFIX, n), DType::BF16, [rows, cols / Q_GROUP])?,
                want(&format!("{}{}.biases", PREFIX, n), DType::BF16, [rows, cols / Q_GROUP])?,
            ])
        };
        let (rank, vocab) = (cfg.rank, cfg.vocab);
        let [pw, ps, pb] = triple("hidden_projection", rank, cfg.hidden)?;
        let book = |n: &str| -> Result<Book, String> {
            let [w, s, b] = triple(n, vocab, rank)?;
            Ok(Book { w: w.off, s: s.off, b: b.off })
        };
        let (pred, succ) = (book("predecessor_codebook")?, book("successor_codebook")?);
        let known: Vec<String> = ["hidden_projection", "predecessor_codebook", "successor_codebook"].iter()
            .flat_map(|n| ["weight", "scales", "biases"].map(|p| format!("{}{}.{}", PREFIX, n, p))).collect();
        for n in idx.names() {
            if n.starts_with(PREFIX) && !known.iter().any(|k| k == n) { return Err(format!("selector: {}: unexpected tensor {}", path, n)); }
        }
        let file = std::fs::File::open(&path).map_err(|e| format!("{}: {}", path, e))?;
        let bytes = |t: &TensorInfo| -> Result<Vec<u8>, String> {
            let mut v = vec![0u8; t.nbytes];
            file.read_exact_at(&mut v, t.off).map_err(|e| format!("selector: {}: read: {}", path, e))?;
            Ok(v)
        };
        let pw = bytes(&pw)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let u16s = |v: Vec<u8>| -> Vec<u16> { v.chunks_exact(BF).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
        let (ps, pb) = (u16s(bytes(&ps)?), u16s(bytes(&pb)?));
        Ok(Selector { cfg: cfg.clone(), file, pred, succ, pw, ps, pb, io_errors: AtomicU64::new(0) })
    }

    /// The z-lab walk: the first position's predecessor is `anchor`, the block's first token (the pending token). This is
    /// the call the drafter should use.
    pub fn pick_from(&self, anchor: u32, logits: &[f32], hidden: &[f32], rows: usize) -> Vec<u32> {
        self.walk(Some(anchor), logits, hidden, rows)
    }

    /// `logits` rows x vocab, `hidden` rows x hidden. No anchor: position 0 is logit-only, later positions as z-lab, so
    /// position 0 can differ from z-lab's and acceptance can be lower. Prefer `pick_from`.
    pub fn pick(&self, logits: &[f32], hidden: &[f32], rows: usize) -> Vec<u32> {
        self.walk(None, logits, hidden, rows)
    }

    /// Process memory for the budget: see `SelCfg::bytes`.
    pub fn bytes(&self) -> u64 { self.cfg.bytes() }

    /// Positions that fell back to logits only because a codebook row could not be read (or the anchor is out of range).
    pub fn io_errors(&self) -> u64 { self.io_errors.load(Ordering::Relaxed) }

    fn walk(&self, anchor: Option<u32>, logits: &[f32], hidden: &[f32], rows: usize) -> Vec<u32> {
        let c = &self.cfg;
        assert_eq!(logits.len(), rows * c.vocab, "selector: logits must be rows x vocab");
        assert_eq!(hidden.len(), rows * c.hidden, "selector: hidden must be rows x hidden");
        let (cw, cs, cb) = SelCfg::packed(c.rank);
        let mut sc = Scratch {
            cand: Vec::with_capacity(c.top_k),
            edges: vec![0.0; c.top_k],
            pred: vec![0.0; c.rank],
            y: vec![0.0; c.rank],
            row: vec![0.0; c.rank],
            gsum: vec![0.0; c.hidden / Q_GROUP],
            raw: vec![0; cw + cs + cb],
        };
        let mut path = Vec::with_capacity(rows);
        let mut pred = anchor;
        for t in 0..rows {
            top_k(&logits[t * c.vocab..(t + 1) * c.vocab], c.top_k, &mut sc.cand);
            sc.edges.fill(0.0);
            if let Some(a) = pred {
                if let Err(e) = self.edges(a, &hidden[t * c.hidden..(t + 1) * c.hidden], &mut sc) {
                    if self.io_errors.fetch_add(1, Ordering::Relaxed) == 0 { eprintln!("selector: logits only at this position: {}", e); }
                    sc.edges.fill(0.0);
                }
            }
            let picked = best(&sc.cand, &sc.edges);
            path.push(picked);
            pred = Some(picked);
        }
        path
    }

    /// edges[i] = sum_r pred[a][r] * proj(h_t)[r] * succ[cand[i]][r].
    fn edges(&self, a: u32, h_t: &[f32], sc: &mut Scratch) -> io::Result<()> {
        let c = &self.cfg;
        self.read_row(&self.pred, a, &mut sc.raw, &mut sc.pred)?;
        gemv(&self.pw, &self.ps, &self.pb, c.rank, c.hidden, h_t, &mut sc.gsum, &mut sc.y);
        for (p, y) in sc.pred.iter_mut().zip(&sc.y) { *p *= *y; }
        for i in 0..sc.cand.len() {
            self.read_row(&self.succ, sc.cand[i].1, &mut sc.raw, &mut sc.row)?;
            sc.edges[i] = sc.pred.iter().zip(&sc.row).map(|(p, s)| p * s).sum();
        }
        Ok(())
    }

    /// Read and unpack one codebook row.
    fn read_row(&self, book: &Book, tok: u32, raw: &mut [u8], out: &mut [f32]) -> io::Result<()> {
        if tok as usize >= self.cfg.vocab {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("token {} out of vocab {}", tok, self.cfg.vocab)));
        }
        let (cw, cs, cb) = SelCfg::packed(self.cfg.rank);
        let (w, rest) = raw.split_at_mut(cw);
        let (s, b) = rest.split_at_mut(cs);
        let tok = tok as u64;
        self.file.read_exact_at(w, book.w + tok * cw as u64)?;
        self.file.read_exact_at(s, book.s + tok * cs as u64)?;
        self.file.read_exact_at(&mut b[..cb], book.b + tok * cb as u64)?;
        decode_row(w, s, b, out);
        Ok(())
    }
}

/// Top `k` of `row` into `cand`, sorted by ascending token id. Strictly greater replaces, and ids are scanned upward, so a
/// tie at the boundary keeps the lower id. NaN ranks lowest.
fn top_k(row: &[f32], k: usize, cand: &mut Vec<(f32, u32)>) {
    let key = |l: f32| if l.is_nan() { f32::NEG_INFINITY } else { l };
    cand.clear();
    cand.extend(row[..k].iter().enumerate().map(|(i, &l)| (key(l), i as u32)));
    // worst = lowest logit, among equals the highest id
    let worst = |c: &[(f32, u32)]| {
        let mut w = 0;
        for (i, e) in c.iter().enumerate() {
            if e.0 < c[w].0 || (e.0 == c[w].0 && e.1 > c[w].1) { w = i; }
        }
        w
    };
    let mut w = worst(cand);
    for (i, &l) in row.iter().enumerate().skip(k) {
        if l > cand[w].0 {
            cand[w] = (l, i as u32);
            w = worst(cand);
        }
    }
    cand.sort_unstable_by_key(|e| e.1);
}

/// argmax of logit + edge over `cand` (ascending id): the first maximum, so a tie takes the lowest id. A NaN score never
/// wins; with no finite score at all the lowest id is taken.
fn best(cand: &[(f32, u32)], edges: &[f32]) -> u32 {
    let (mut id, mut top) = (cand[0].1, f32::NEG_INFINITY);
    for (e, x) in cand.iter().zip(edges) {
        let s = e.0 + x;
        if s > top { (id, top) = (e.1, s); }
    }
    id
}

/// One packed row (little-endian bytes): out[g * 64 + i] = nibble(i) * S[g] + B[g], low nibble first in each u32.
fn decode_row(w: &[u8], s: &[u8], b: &[u8], out: &mut [f32]) {
    let bf = |x: &[u8], g: usize| bf16_to_f32(u16::from_le_bytes([x[g * BF], x[g * BF + 1]]));
    let word_bytes = 4;
    for (i, c) in w.chunks_exact(word_bytes).enumerate() {
        let word = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let j0 = i * WORD_VALS;
        let g = j0 / Q_GROUP;
        let (sc, bi) = (bf(s, g), bf(b, g));
        for q in 0..WORD_VALS {
            out[j0 + q] = ((word >> (Q_BITS * q)) & 15) as f32 * sc + bi;
        }
    }
}

/// y = M x, M packed [rows, cols]: per group S * sum(nibble * x) + B * sum(x) = sum((nibble * S + B) * x).
#[allow(clippy::too_many_arguments)]
fn gemv(w: &[u32], s: &[u16], b: &[u16], rows: usize, cols: usize, x: &[f32], gsum: &mut [f32], y: &mut [f32]) {
    let (wc, gc, wg) = (cols / WORD_VALS, cols / Q_GROUP, Q_GROUP / WORD_VALS);
    for (g, o) in gsum.iter_mut().enumerate().take(gc) { *o = x[g * Q_GROUP..(g + 1) * Q_GROUP].iter().sum(); }
    for (r, o) in y.iter_mut().enumerate().take(rows) {
        let mut acc = 0.0f32;
        for g in 0..gc {
            let mut q_x = 0.0f32;
            for wi in 0..wg {
                let word = w[r * wc + g * wg + wi];
                let j0 = g * Q_GROUP + wi * WORD_VALS;
                for q in 0..WORD_VALS {
                    q_x += ((word >> (Q_BITS * q)) & 15) as f32 * x[j0 + q];
                }
            }
            acc += bf16_to_f32(s[r * gc + g]) * q_x + bf16_to_f32(b[r * gc + g]) * gsum[g];
        }
        *o = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use streamdense_core::{f32_to_bf16, write_safetensors};

    type T = (String, DType, Vec<usize>, Vec<u8>);

    // Tiny file: vocab 6, top 3, rank 64, hidden 64, S = 1, B = 0. Row 0, column 0 of the projection is 1 and all else 0, so
    // proj(h)[0] = h[0] and proj(h)[r > 0] = 0. Only value 0 of each codebook row is nonzero (QP / QS below), therefore:
    //   score(b) = logit[b] + QP[a] * h[0] * QS[b]
    const QP: [u8; 6] = [1, 2, 0, 3, 1, 0];
    const QS: [u8; 6] = [0, 1, 2, 3, 4, 5];
    const V: usize = 6;
    const R: usize = 64;

    fn cfg() -> SelCfg { SelCfg { vocab: V, hidden: R, rank: R, top_k: 3, max_rows: 7 } }

    fn le16(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|&x| f32_to_bf16(x).to_le_bytes()).collect() }

    /// `rows` rows of 64 values with only value 0 set: (W, S, B) tensors under `name`.
    fn tensors(name: &str, q0: &[u8]) -> Vec<T> {
        let rows = q0.len();
        let w: Vec<u8> = q0.iter().flat_map(|&q| { let mut r = vec![0u32; R / 8]; r[0] = q as u32; r }).flat_map(u32::to_le_bytes).collect();
        vec![
            (format!("{}{}.weight", PREFIX, name), DType::U32, vec![rows, R / 8], w),
            (format!("{}{}.scales", PREFIX, name), DType::BF16, vec![rows, 1], le16(&vec![1.0; rows])),
            (format!("{}{}.biases", PREFIX, name), DType::BF16, vec![rows, 1], le16(&vec![0.0; rows])),
        ]
    }

    fn model() -> Vec<T> {
        let mut proj = [0u8; R];
        proj[0] = 1;
        let mut t = tensors("hidden_projection", &proj);
        t.extend(tensors("predecessor_codebook", &QP));
        t.extend(tensors("successor_codebook", &QS));
        // a drafter tensor outside the selector must not matter
        t.push(("norm.weight".into(), DType::BF16, vec![R], le16(&[1.0; R])));
        t
    }

    struct TmpDir(std::path::PathBuf);
    impl Drop for TmpDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

    fn write(ts: &[T]) -> TmpDir {
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!("sd_dflash2_select_{}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        let views: Vec<(&str, DType, &[usize], &[u8])> = ts.iter().map(|(n, dt, s, b)| (n.as_str(), *dt, s.as_slice(), b.as_slice())).collect();
        write_safetensors(&d.join(WEIGHTS_FILE).display().to_string(), &views).unwrap();
        TmpDir(d)
    }

    /// `h0` per row becomes hidden[row][0], everything else 0.
    fn hid(h0: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0; h0.len() * R];
        for (t, x) in h0.iter().enumerate() { h[t * R] = *x; }
        h
    }

    fn sel(d: &TmpDir) -> Selector { Selector::load(&d.0, &cfg()).unwrap() }

    #[test]
    fn hand_computed_path() {
        let d = write(&model());
        let s = sel(&d);
        let logits = [
            9.0, 1.0, 8.0, 0.0, 7.0, 2.0, // top 3 = ids 0, 2, 4; a = 1, QP 2, h 0.5 -> factor 1: 9+0, 8+2, 7+4 -> id 4
            0.0, 5.0, 0.0, 6.0, 0.0, 3.5, // top 3 = ids 1, 3, 5; a = 4, QP 1, h 1 -> factor 1: 5+1, 6+3, 3.5+5 -> id 3
        ];
        assert_eq!(s.pick_from(1, &logits, &hid(&[0.5, 1.0]), 2), vec![4, 3]);
        assert_eq!(s.io_errors(), 0);
    }

    #[test]
    fn codebook_forces_non_argmax() {
        let d = write(&model());
        let s = sel(&d);
        // top 3 = ids 0, 1, 2; a = 3, QP 3, h -1 -> factor -3: 1 - 0, 2 - 3, 3 - 6 -> id 0, though the logit argmax is id 2
        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        assert_eq!(s.pick_from(3, &logits, &hid(&[-1.0]), 1), vec![0]);
    }

    #[test]
    fn rows_one_and_zero() {
        let d = write(&model());
        let s = sel(&d);
        // a = 0, QP 1, h 2 -> factor 2: 1 + 0, 2 + 2, 3 + 4 -> id 2
        assert_eq!(s.pick_from(0, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &hid(&[2.0]), 1), vec![2]);
        assert_eq!(s.pick_from(0, &[], &[], 0), Vec::<u32>::new());
    }

    #[test]
    fn predecessor_changes_the_choice() {
        let d = write(&model());
        let s = sel(&d);
        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        // a = 2, QP 0 -> no edge -> logit argmax id 2; a = 3 (above) gives id 0. Same row, only the anchor differs.
        assert_eq!(s.pick_from(2, &logits, &hid(&[-1.0]), 1), vec![2]);
        assert_eq!(s.pick_from(3, &logits, &hid(&[-1.0]), 1), vec![0]);
    }

    #[test]
    fn score_tie_takes_lowest_id() {
        let d = write(&model());
        let s = sel(&d);
        // all edges 0, three equal logits -> id 0
        assert_eq!(s.pick_from(0, &[5.0, 5.0, 5.0, 1.0, 1.0, 1.0], &hid(&[0.0]), 1), vec![0]);
        // top 3 = ids 0, 2, 3; a = 1, factor 1: 1 + 0, 5 + 2, 4 + 3 -> 7 = 7 -> id 2 (different logits, equal score)
        assert_eq!(s.pick_from(1, &[1.0, 0.0, 5.0, 4.0, 0.0, 0.0], &hid(&[0.5]), 1), vec![2]);
    }

    #[test]
    fn candidate_tie_keeps_lower_ids() {
        let d = write(&model());
        let s = sel(&d);
        // ids 1..=4 tie at 7, top 3 keeps 1, 2, 3. a = 1, factor 1: 8, 9, 10 -> id 3 (id 4 would score 11 if it were kept)
        assert_eq!(s.pick_from(1, &[1.0, 7.0, 7.0, 7.0, 7.0, 0.0], &hid(&[0.5]), 1), vec![3]);
    }

    #[test]
    fn no_anchor_is_logits_only_at_position_zero() {
        let d = write(&model());
        let s = sel(&d);
        let logits = [
            9.0, 1.0, 8.0, 0.0, 7.0, 2.0, // no predecessor: id 0 although a = 1 would give id 4
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, // a = 0, QP 1, h 2 -> factor 2 -> id 2
        ];
        assert_eq!(s.pick(&logits, &hid(&[0.5, 2.0]), 2), vec![0, 2]);
    }

    #[test]
    fn bad_anchor_and_nan_fall_back() {
        let d = write(&model());
        let s = sel(&d);
        assert_eq!(s.pick_from(V as u32, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], &hid(&[-1.0]), 1), vec![2]);
        assert_eq!(s.io_errors(), 1);
        // NaN logit never wins or enters while real values remain
        assert_eq!(s.pick_from(2, &[f32::NAN, 2.0, 3.0, 1.0, 0.0, 0.0], &hid(&[0.0]), 1), vec![2]);
    }

    #[test]
    fn top_k_all_candidates() {
        let mut c = vec![];
        top_k(&[3.0, 1.0, 2.0], 3, &mut c);
        assert_eq!(c, vec![(3.0, 0), (1.0, 1), (2.0, 2)]);
        // ascending input, every element replaces: top 2 of 5
        top_k(&[1.0, 2.0, 3.0, 4.0, 5.0], 2, &mut c);
        assert_eq!(c, vec![(4.0, 3), (5.0, 4)]);
    }

    #[test]
    fn decode_row_affine_groups_and_nibble_order() {
        // rank 128: 2 groups, S = [0.5, 2], B = [-1, 1]; nibble 9 = 3 (word 1, second nibble), nibble 64 = 15
        let mut w = [0u32; 16];
        w[1] = 3 << 4;
        w[8] = 15;
        let wb: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut out = vec![0.0; 128];
        decode_row(&wb, &le16(&[0.5, 2.0]), &le16(&[-1.0, 1.0]), &mut out);
        assert_eq!(out[0], -1.0);
        assert_eq!(out[9], 0.5); // 3 * 0.5 - 1
        assert_eq!(out[64], 31.0); // 15 * 2 + 1
        assert_eq!(out[65], 1.0);
    }

    #[test]
    fn gemv_two_groups() {
        // 2 x 128. Row 0: group 0 S 0.5 B 1, nibble 1 = 2; group 1 S 1 B -1, all nibbles 0. Row 1: S 1, B 0, all nibbles 0.
        let mut w = vec![0u32; 32];
        w[0] = 2 << 4;
        let s = [f32_to_bf16(0.5), f32_to_bf16(1.0), f32_to_bf16(1.0), f32_to_bf16(1.0)];
        let b = [f32_to_bf16(1.0), f32_to_bf16(-1.0), f32_to_bf16(0.0), f32_to_bf16(0.0)];
        let mut x = vec![0.0; 128];
        x[1] = 3.0;
        x[64] = 2.0;
        let (mut g, mut y) = (vec![0.0; 2], vec![0.0; 2]);
        gemv(&w, &s, &b, 2, 128, &x, &mut g, &mut y);
        assert_eq!(y, vec![4.0, 0.0]); // (2 * 0.5 + 1) * 3 + (0 + -1) * 2
    }

    #[test]
    fn load_refuses() {
        let err = |ts: Vec<T>| Selector::load(&write(&ts).0, &cfg()).err().expect("must refuse");
        // missing tensor
        let mut t = model();
        t.retain(|x| x.0 != "candidate_selector.successor_codebook.biases");
        assert!(err(t).contains("no tensor"));
        // wrong shape
        let mut t = model();
        let i = t.iter().position(|x| x.0 == "candidate_selector.predecessor_codebook.weight").unwrap();
        t[i].2 = vec![V * 2, R / 16];
        assert!(err(t).contains("need"));
        // wrong dtype, same byte size
        let mut t = model();
        let i = t.iter().position(|x| x.0 == "candidate_selector.hidden_projection.scales").unwrap();
        t[i].1 = DType::F16;
        assert!(err(t).contains("need"));
        // codebooks stored as plain bf16 (the unquantized name z-lab loads): the 4-bit names are gone
        let mut t = model();
        t.retain(|x| !x.0.starts_with("candidate_selector.predecessor_codebook"));
        t.push(("candidate_selector.predecessor_codebook".into(), DType::BF16, vec![V, R], vec![0; V * R * 2]));
        assert!(err(t).contains("no tensor"));
        // stray selector tensor
        let mut t = model();
        t.push(("candidate_selector.extra".into(), DType::BF16, vec![1], vec![0; 2]));
        assert!(err(t).contains("unexpected"));
        // no file
        assert!(Selector::load(Path::new("/nonexistent_sd_dflash2"), &cfg()).is_err());
    }

    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlash2DraftModel"], "hidden_size": 5120, "vocab_size": 248320,
            "dflash_config": { "block_size": 8, "selector_rank": 256, "selector_top_k": 16 },
            "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
            "quantization_config": { "group_size": 64, "bits": 4, "mode": "affine" },
        })
    }

    #[test]
    fn config_by_formula() {
        let c = SelCfg::from_json(&real_config()).unwrap();
        assert_eq!(c, SelCfg { vocab: 248320, hidden: 5120, rank: 256, top_k: 16, max_rows: 7 });
        // resident 256 x (640 x 4 + 80 x 2 + 80 x 2) = 737,280; scratch 128 + 64 + 3072 + 320 + 144 + 28 = 3,756
        assert_eq!(c.bytes(), 737_280 + 3_756);
    }

    #[test]
    fn config_refuses() {
        let with = |f: &dyn Fn(&mut serde_json::Value)| { let mut v = real_config(); f(&mut v); SelCfg::from_json(&v).is_err() };
        assert!(with(&|v| v["architectures"] = serde_json::json!(["DFlashDraftModel"])));
        assert!(with(&|v| v["quantization"]["bits"] = serde_json::json!(8)));
        assert!(with(&|v| v["quantization"]["group_size"] = serde_json::json!(32)));
        assert!(with(&|v| v["quantization"]["mode"] = serde_json::json!("mxfp4")));
        assert!(with(&|v| v["quantization_config"]["bits"] = serde_json::json!(8)));
        assert!(with(&|v| v.as_object_mut().unwrap().remove("quantization").map(|_| ()).unwrap()));
        assert!(with(&|v| v["dflash_config"]["selector_rank"] = serde_json::json!(100)));
        assert!(with(&|v| v["hidden_size"] = serde_json::json!(5000)));
        assert!(with(&|v| v["dflash_config"]["selector_top_k"] = serde_json::json!(0)));
    }
}
