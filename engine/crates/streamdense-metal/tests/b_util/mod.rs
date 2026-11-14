//! Shared helpers for the deltanet / attn_gated kernel tests (self-test agent B).
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
