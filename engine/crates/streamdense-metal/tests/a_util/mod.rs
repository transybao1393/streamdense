//! Shared helpers for the gemv_q4_bf16 / norm_bf16 / head_sample kernel tests (self-test agent A; copied from b_util).
//! A bump allocator over one shared MTLBuffer (Enc binds everything as (arena, offset)), bf16 helpers that mirror
//! common_bf16.metal, a minimal safetensors reader for the MLX vectors (no extra crates), and ulp metrics.
#![allow(dead_code)]
use std::collections::HashMap;
use streamdense_metal::{contents, Buffer, Gpu};

pub const VEC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/vectors");
pub const READY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/READY");

// ---------------------------------------------------------------- bf16 (mirror common_bf16.metal)
pub fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
pub fn to_bf(f: f32) -> u16 {
    let mut u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 { return ((u >> 16) | 0x0040) as u16; }
    u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (u >> 16) as u16
}
pub fn rbf(f: f32) -> f32 { bf(to_bf(f)) }

/// Monotone integer key of a bf16 bit pattern (for ulp distance).
fn key16(u: u16) -> i32 { if u >= 0x8000 { 0x8000 - u as i32 } else { u as i32 } }
pub fn ulp_bf(a: u16, b: u16) -> u32 { (key16(a) - key16(b)).unsigned_abs() }
fn key32(f: f32) -> i64 { let u = f.to_bits(); if u >= 0x8000_0000 { 0x8000_0000i64 - u as i64 } else { u as i64 } }
pub fn ulp_f32(a: f32, b: f32) -> u64 { (key32(a) - key32(b)).unsigned_abs() }

/// (max ulp, number of differing elements)
pub fn cmp_bf(a: &[u16], b: &[u16]) -> (u32, usize) {
    assert_eq!(a.len(), b.len());
    let mut m = 0; let mut n = 0;
    for (x, y) in a.iter().zip(b) { let d = ulp_bf(*x, *y); if d > 0 { n += 1; } m = m.max(d); }
    (m, n)
}
pub fn cmp_f32(a: &[f32], b: &[f32]) -> (u64, usize, f32) {
    assert_eq!(a.len(), b.len());
    let mut m = 0; let mut n = 0; let mut ad = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = ulp_f32(*x, *y); if d > 0 { n += 1; } m = m.max(d); ad = ad.max((x - y).abs());
    }
    (m, n, ad)
}

// ---------------------------------------------------------------- deterministic random
pub struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Rng { Rng(s.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1) }
    pub fn u64(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    pub fn unif(&mut self) -> f32 { (self.u64() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unif().max(1e-7); let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    pub fn bf_n(&mut self, n: usize, std: f32) -> Vec<u16> { (0..n).map(|_| to_bf(std * self.normal())).collect() }
}

// ---------------------------------------------------------------- arena
pub struct Ar {
    pub buf: Buffer,
    pub size: usize,
    used: usize,
}
impl Ar {
    pub fn new(gpu: &Gpu, size: usize) -> Ar { Ar { buf: gpu.buffer(size).unwrap(), size, used: 0 } }
    pub fn alloc(&mut self, len: usize) -> usize {
        let off = (self.used + 255) / 256 * 256;
        assert!(off + len <= self.size, "test arena full");
        self.used = off + len;
        off
    }
    pub fn base(&self) -> *mut u8 { contents(&self.buf) }
    pub fn put<T: Copy>(&self, off: usize, v: &[T]) {
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, self.base().add(off), std::mem::size_of_val(v)) }
    }
    pub fn get<T: Copy + Default>(&self, off: usize, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        unsafe { std::ptr::copy_nonoverlapping(self.base().add(off), v.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>()) }
        v
    }
    pub fn fill(&self, off: usize, len: usize, byte: u8) { unsafe { std::ptr::write_bytes(self.base().add(off), byte, len) } }
    pub fn alloc_put<T: Copy>(&mut self, v: &[T]) -> usize { let o = self.alloc(std::mem::size_of_val(v).max(4)); self.put(o, v); o }
}

// ---------------------------------------------------------------- minimal safetensors reader
pub struct St { pub t: HashMap<String, (String, Vec<usize>, Vec<u8>)> }
impl St {
    pub fn open(path: &str) -> Option<St> {
        let b = std::fs::read(path).ok()?;
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&b[8..8 + n]).ok()?;
        let data = &b[8 + n..];
        let mut t = HashMap::new();
        let mut p = Json { s: hdr.as_bytes(), i: 0 };
        let root = p.value();
        if let JV::Obj(m) = root {
            for (k, v) in m {
                if k == "__metadata__" { continue; }
                if let JV::Obj(f) = v {
                    let get = |n: &str| f.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()).unwrap();
                    let dt = match get("dtype") { JV::Str(s) => s, _ => panic!() };
                    let shape = match get("shape") { JV::Arr(a) => a.iter().map(|x| x.num() as usize).collect(), _ => panic!() };
                    let off = match get("data_offsets") { JV::Arr(a) => (a[0].num() as usize, a[1].num() as usize), _ => panic!() };
                    t.insert(k, (dt, shape, data[off.0..off.1].to_vec()));
                }
            }
        }
        Some(St { t })
    }
    pub fn has(&self, n: &str) -> bool { self.t.contains_key(n) }
    pub fn shape(&self, n: &str) -> Vec<usize> { self.t.get(n).unwrap_or_else(|| panic!("no tensor {}", n)).1.clone() }
    pub fn u16s(&self, n: &str) -> Vec<u16> {
        let (dt, _, d) = self.t.get(n).unwrap_or_else(|| panic!("no tensor {}", n));
        assert!(dt == "BF16" || dt == "F16", "{} is {}", n, dt);
        d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
    pub fn f32s(&self, n: &str) -> Vec<f32> {
        let (dt, _, d) = self.t.get(n).unwrap_or_else(|| panic!("no tensor {}", n));
        assert_eq!(dt, "F32", "{}", n);
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

#[derive(Clone, Debug)]
enum JV { Obj(Vec<(String, JV)>), Arr(Vec<JV>), Str(String), Num(f64), Other }
impl JV { fn num(&self) -> f64 { match self { JV::Num(x) => *x, _ => panic!("not a number") } } }
struct Json<'a> { s: &'a [u8], i: usize }
impl<'a> Json<'a> {
    fn ws(&mut self) { while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() { self.i += 1; } }
    fn value(&mut self) -> JV {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1; let mut m = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b'}' { self.i += 1; break; }
                    if self.s[self.i] == b',' { self.i += 1; continue; }
                    let k = match self.value() { JV::Str(s) => s, _ => panic!("key") };
                    self.ws(); assert_eq!(self.s[self.i], b':'); self.i += 1;
                    let v = self.value(); m.push((k, v));
                }
                JV::Obj(m)
            }
            b'[' => {
                self.i += 1; let mut a = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b']' { self.i += 1; break; }
                    if self.s[self.i] == b',' { self.i += 1; continue; }
                    a.push(self.value());
                }
                JV::Arr(a)
            }
            b'"' => {
                self.i += 1; let st = self.i;
                while self.s[self.i] != b'"' { if self.s[self.i] == b'\\' { self.i += 1; } self.i += 1; }
                let r = String::from_utf8_lossy(&self.s[st..self.i]).to_string(); self.i += 1; JV::Str(r)
            }
            c if c == b'-' || c.is_ascii_digit() => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { self.i += 1; }
                JV::Num(std::str::from_utf8(&self.s[st..self.i]).unwrap().parse().unwrap())
            }
            _ => { while self.i < self.s.len() && self.s[self.i].is_ascii_alphabetic() { self.i += 1; } JV::Other }
        }
    }
}

/// The MLX vector file, or None (with a skip message) if the vectors are missing.
pub fn vectors(file: &str) -> Option<St> {
    let p = format!("{}/{}.safetensors", VEC_DIR, file);
    match St::open(&p) {
        Some(s) => {
            if !std::path::Path::new(READY).exists() { eprintln!("note: {} present but testdata/READY missing", p); }
            Some(s)
        }
        None => { eprintln!("SKIP MLX-vector part: {} missing", p); None }
    }
}

// ---------------------------------------------------------------- agent A extras
impl St {
    pub fn u32s(&self, n: &str) -> Vec<u32> {
        let (dt, _, d) = self.t.get(n).unwrap_or_else(|| panic!("no tensor {}", n));
        assert!(dt == "U32", "{} is {}", n, dt);
        d.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    pub fn dtype(&self, n: &str) -> String { self.t.get(n).unwrap_or_else(|| panic!("no tensor {}", n)).0.clone() }
}

/// Random MLX-affine 4-bit matrix [r][c]: W u32 [r][c/8], S/B bf16 [r][c/64] (scale ~ U(0.002, 0.01), bias ~ -8 s).
pub fn rand_q4(rng: &mut Rng, r: usize, c: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let w: Vec<u32> = (0..r * c / 8).map(|_| rng.u64() as u32).collect();
    let g = r * c / 64;
    let mut s = Vec::with_capacity(g);
    let mut b = Vec::with_capacity(g);
    for _ in 0..g {
        let sc = 0.002 + 0.008 * rng.unif();
        s.push(to_bf(sc));
        b.push(to_bf(-8.0 * sc + 0.002 * rng.normal()));
    }
    (w, s, b)
}

/// MLX dequantize of row `row` (fp32 q*s + b, one bf16 rounding) as f32 values.
pub fn deq_row(w: &[u32], s: &[u16], b: &[u16], c: usize, row: usize) -> Vec<f32> {
    let (wc, gc) = (c / 8, c / 64);
    (0..c).map(|j| {
        let q = (w[row * wc + j / 8] >> (4 * (j % 8))) & 15;
        let (sc, bi) = (bf(s[row * gc + j / 64]), bf(b[row * gc + j / 64]));
        rbf(q as f32 * sc + bi)
    }).collect()
}

/// Exact affine row dot in f64 (w = s*q + b without bf16 rounding of the product, as the GEMV kernels do).
pub fn dot_affine_f64(w: &[u32], s: &[u16], b: &[u16], c: usize, row: usize, x: &[u16]) -> f64 {
    let (wc, gc) = (c / 8, c / 64);
    let mut acc = 0f64;
    for g in 0..gc {
        let (sc, bi) = (bf(s[row * gc + g]) as f64, bf(b[row * gc + g]) as f64);
        let (mut qx, mut xs) = (0f64, 0f64);
        for j in g * 64..g * 64 + 64 {
            let q = ((w[row * wc + j / 8] >> (4 * (j % 8))) & 15) as f64;
            let xv = bf(x[j]) as f64;
            qx += q * xv; xs += xv;
        }
        acc += sc * qx + bi * xs;
    }
    acc
}

/// Round an f64 to bf16 (via f32 then RNE: f64 -> f32 rounding first can double-round, so do it directly).
pub fn f64_to_bf(v: f64) -> u16 {
    let f = v as f32;
    // correct a possible double rounding: pick the bf16 neighbour nearest to v in f64
    let a = to_bf(f);
    let cand = [a.wrapping_sub(1), a, a.wrapping_add(1)];
    let mut best = a; let mut bd = f64::INFINITY;
    for c in cand { let d = (bf(c) as f64 - v).abs(); if bf(c).is_finite() && d < bd { bd = d; best = c; } }
    best
}
