//! attn_flash2_q8 (SD_ATTN_V2) vs attn_flash_q8: speed at 27B prefill shapes (nh 24, nkv 4, HD 256, q8 KV, T 128 / 256
//! rows at pos 2K / 8K / 16K / 32K) and accuracy vs an f64 reference (same q8 KV values) and vs the old kernel.
//! Bench (prints ms / TFLOPS): SD_AF2_BENCH=1 cargo test --release --offline -p streamdense-metal --test attn_flash2 -- --nocapture --test-threads 1
//! Without SD_AF2_BENCH only the accuracy test runs (small max_ctx).
mod b_util;
use b_util::*;
use streamdense_metal::{commit, gpu_ms, wait, Enc, Gpu, SdCtrl};

const HD: usize = 256;
const NH: usize = 24;
const NKV: usize = 4;
const GRP: usize = NH / NKV;

/// (kernel, R row tiles)
const VARIANTS: &[(&str, usize)] = &[("attn_flash2_q8", 2), ("attn_flash2_q8_r1", 1)];

struct Bufs { maxctx: usize, prow: usize, qr: usize, k8: usize, v8: usize, ks: usize, vs: usize, p: usize, ao: usize, ctrl: usize }

struct Fix { gpu: Gpu, ar: Ar, b: Bufs, kf: Vec<f32>, vf: Vec<f32> }

/// f32 -> fp16 bits, round to nearest even (normal range only; enough for the q8 scale / min here).
fn f16b(x: f32) -> u16 {
    let b = x.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&e), "f16b out of range {}", x);
    let m = b & 0x7f_ffff;
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1; }
    (s | h) as u16
}
fn f16f(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let (e, m) = (((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    assert!(e > 0 && e < 31);
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}
/// q8 KV like attn_prep_q8: per 32-dim group scale = (max - min) / 255, min, both fp16; codes round((x - min) / scale).
fn quant(r: &mut Rng, n: usize, bias: &[f32]) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
    let mut codes = vec![0u8; n];
    let mut sb = vec![0u16; n / 32 * 2];
    let mut deq = vec![0f32; n];
    let mut x = [0f32; 32];
    for gi in 0..n / 32 {
        for (i, v) in x.iter_mut().enumerate() { *v = rbf(r.normal() + bias[(gi * 32 + i) % HD]); }
        let (mn, mx) = x.iter().fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let sc = f16b(((mx - mn) / 255.0).max(1e-4));
        let mh = f16b(mn);
        sb[2 * gi] = sc; sb[2 * gi + 1] = mh;
        let (scf, mhf) = (f16f(sc), f16f(mh));
        for i in 0..32 {
            let c = ((x[i] - mhf) / scf).round().clamp(0.0, 255.0) as u8;
            codes[gi * 32 + i] = c;
            deq[gi * 32 + i] = scf * c as f32 + mhf;
        }
    }
    (codes, sb, deq)
}

fn fixture(maxctx: usize, tmax: usize, names: &[&str]) -> Fix {
    let mut gpu = Gpu::from_shader_files(&["common_bf16", "attn_gated"]).unwrap();
    for k in names { gpu.pipeline(k).unwrap(); }
    let prow = NH * 2 * HD + 2 * NKV * HD;
    let nkvv = NKV * maxctx * HD;
    let mut ar = Ar::new(&gpu, tmax * prow * 2 + tmax * NH * HD * 4 + nkvv * 2 + nkvv / 32 * 8 + (1 << 22));
    let b = Bufs { maxctx, prow, qr: ar.alloc(tmax * NH * HD * 2), k8: ar.alloc(nkvv), v8: ar.alloc(nkvv), ks: ar.alloc(nkvv / 32 * 4),
                   vs: ar.alloc(nkvv / 32 * 4), p: ar.alloc(tmax * prow * 2), ao: ar.alloc(tmax * NH * HD * 2), ctrl: ar.alloc(256) };
    let mut r = Rng::new(17);
    // K with per-dim offsets (real K caches have strong per-channel means), V centered
    let kbias: Vec<f32> = (0..HD).map(|_| 1.5 * r.normal()).collect();
    let (kc, ksb, kf) = quant(&mut r, nkvv, &kbias);
    let (vc, vsb, vf) = quant(&mut r, nkvv, &vec![0.0; HD]);
    ar.put(b.k8, &kc); ar.put(b.ks, &ksb); ar.put(b.v8, &vc); ar.put(b.vs, &vsb);
    ar.put(b.p, &r.bf_n(tmax * prow, 1.5));
    Fix { gpu, ar, b, kf, vf }
}

fn run(gpu: &Gpu, ar: &Ar, reps: usize, f: impl Fn(&Enc)) -> f64 {
    let cb = gpu.command_buffer();
    let e = Enc::new(gpu, &cb, &ar.buf);
    for _ in 0..reps { f(&e); }
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
    gpu_ms(&cb) / reps as f64
}

fn enc_old(e: &Enc, b: &Bufs, t: usize) {
    e.pso("attn_flash_q8").b(0, b.qr).b(1, b.k8).b(2, b.v8).b(3, b.p).b(4, b.ao).b(5, b.ctrl)
        .u(6, b.maxctx as u32).u(7, GRP as u32).u(8, b.prow as u32).u(9, t as u32).b(10, b.ks).b(11, b.vs)
        .groups((t.div_ceil(8), NKV), 32 * GRP);
}
fn enc_v2(e: &Enc, b: &Bufs, t: usize, name: &str, r: usize) {
    e.pso(name).b(0, b.qr).b(1, b.k8).b(2, b.v8).b(3, b.p).b(4, b.ao).b(5, b.ctrl)
        .u(6, b.maxctx as u32).u(7, GRP as u32).u(8, b.prow as u32).u(9, t as u32).b(10, b.ks).b(11, b.vs)
        .groups((t.div_ceil(8 * r), NKV), 32 * GRP * r);
}

/// f64 reference of rows `rows` (all heads), gated and rounded as the kernels' epilogue: (row, head, dim) -> bf16 value.
fn reference(fx: &Fix, pos: usize, rows: &[usize]) -> Vec<(usize, Vec<f32>)> {
    let b = &fx.b;
    let mut out = vec![];
    for &tt in rows {
        let nk = pos + tt + 1;
        let mut o = vec![0f32; NH * HD];
        for hh in 0..NH {
            let kvh = hh / GRP;
            let q: Vec<u16> = fx.ar.get(b.qr + (tt * NH + hh) * HD * 2, HD);
            let qf: Vec<f64> = q.iter().map(|&x| bf(x) as f64 / 16.0).collect();
            let kk = &fx.kf[kvh * b.maxctx * HD..];
            let vv = &fx.vf[kvh * b.maxctx * HD..];
            let sc: Vec<f64> = (0..nk).map(|j| (0..HD).map(|e| qf[e] * kk[j * HD + e] as f64).sum()).collect();
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = w.iter().sum();
            let gate: Vec<u16> = fx.ar.get(b.p + (tt * b.prow + hh * 2 * HD + HD) * 2, HD);
            let mut acc = vec![0f64; HD];
            for j in 0..nk { let wj = w[j]; for dd in 0..HD { acc[dd] += wj * vv[j * HD + dd] as f64; } }
            for dd in 0..HD {
                let sg = { let x = bf(gate[dd]); let e = rbf(x.abs().exp()); let y = rbf(1.0 / rbf(1.0 + e)); if x < 0.0 { y } else { rbf(1.0 - y) } };
                o[hh * HD + dd] = rbf(rbf((acc[dd] / l) as f32) * sg);
            }
        }
        out.push((tt, o));
    }
    out
}

/// (max abs, max rel (|ref| >= 1/16), outputs != ref) of `got` rows vs the reference.
fn err(got: &[u16], refr: &[(usize, Vec<f32>)]) -> (f32, f32, usize) {
    let (mut a, mut rl, mut d) = (0f32, 0f32, 0usize);
    for (tt, o) in refr {
        for i in 0..NH * HD {
            let x = bf(got[tt * NH * HD + i]);
            let e = (x - o[i]).abs();
            assert!(x.is_finite());
            if e > 0.0 { d += 1; }
            a = a.max(e);
            if o[i].abs() >= 1.0 / 16.0 { rl = rl.max(e / o[i].abs()); }
        }
    }
    (a, rl, d)
}

#[test]
fn attn_flash2_accuracy() {
    let (maxctx, tmax) = (4352usize, 256usize);
    let mut names = vec!["attn_flash_q8"];
    names.extend(VARIANTS.iter().map(|v| v.0));
    let fx = fixture(maxctx, tmax, &names);
    let b = &fx.b;
    let mut r = Rng::new(5);
    let mut worst_old = (0f32, 0f32);
    let mut worst_new = (0f32, 0f32);
    let mut worst_nd = 0usize;
    for sharp in [false, true] {
        fx.ar.put(b.qr, &r.bf_n(tmax * NH * HD, if sharp { 2.5 } else { 1.0 }));
        for (pos, t) in [(0usize, 128usize), (0, 37), (5, 131), (1000, 128), (4096, 256)] {
            fx.ar.put(b.ctrl, &[SdCtrl { pos: pos as u32, ..Default::default() }]);
            let n = t * NH * HD;
            let rows: Vec<usize> = { let mut v = vec![0, 1, 7, 8, t / 2, t - 1]; v.dedup(); v };
            let refr = reference(&fx, pos, &rows);
            fx.ar.fill(b.ao, tmax * NH * HD * 2, 0xff);
            run(&fx.gpu, &fx.ar, 1, |e| enc_old(e, b, t));
            let old: Vec<u16> = fx.ar.get(b.ao, n);
            let eo = err(&old, &refr);
            worst_old = (worst_old.0.max(eo.0), worst_old.1.max(eo.1));
            println!("pos {:5} T {:3} {}: old       vs f64: max abs {:.2e} max rel {:.2e} differ {}", pos, t, if sharp { "sharp " } else { "normal" }, eo.0, eo.1, eo.2);
            for &(name, rr) in VARIANTS {
                {
                    fx.ar.fill(b.ao, tmax * NH * HD * 2, 0xff);
                    run(&fx.gpu, &fx.ar, 1, |e| enc_v2(e, b, t, name, rr));
                    let nw: Vec<u16> = fx.ar.get(b.ao, n);
                    let past: Vec<u16> = fx.ar.get(b.ao + n * 2, (tmax - t) * NH * HD);
                    assert!(past.iter().all(|&x| x == 0xffff), "{} wrote past T", name);
                    let en = err(&nw, &refr);
                    let (mut am, mut nd) = (0f32, 0usize);
                    for i in 0..n { let a = (bf(nw[i]) - bf(old[i])).abs(); assert!(bf(nw[i]).is_finite(), "{} non-finite", name); if nw[i] != old[i] { nd += 1; } am = am.max(a); }
                    println!("    {:18}: vs f64 max abs {:.2e} max rel {:.2e} differ {:4} | vs old max abs {:.2e}, {} of {} differ",
                             name, en.0, en.1, en.2, am, nd, n);
                    worst_new = (worst_new.0.max(en.0), worst_new.1.max(en.1));
                    worst_nd = worst_nd.max(nd);
                    // no accuracy loss vs the old kernel against f64; outputs differ from the old kernel only rarely
                    assert!(en.0 <= eo.0 && en.1 <= eo.1 && en.2 <= eo.2 + 8, "{} worse than old vs f64: {:?} vs {:?}", name, en, eo);
                    assert!(nd * 1000 <= n, "{}: {} of {} outputs differ from attn_flash_q8", name, nd, n);
                }
            }
        }
    }
    println!("worst vs f64: old max abs {:.2e} rel {:.2e}; new max abs {:.2e} rel {:.2e}; most outputs != old in one case: {}",
             worst_old.0, worst_old.1, worst_new.0, worst_new.1, worst_nd);
}

#[test]
fn attn_flash2_bench() {
    if std::env::var("SD_AF2_BENCH").is_err() { return; }
    let tmax = 256usize;
    let maxctx = 32768 + tmax;
    let mut names = vec!["attn_flash_q8"];
    names.extend(VARIANTS.iter().map(|v| v.0));
    let only: Option<Vec<String>> = std::env::var("SD_AF2_ONLY").ok().map(|s| s.split(',').map(String::from).collect());
    let fx = fixture(maxctx, tmax, &names);
    let b = &fx.b;
    let mut r = Rng::new(9);
    fx.ar.put(b.qr, &r.bf_n(tmax * NH * HD, 1.0));
    let reps = 4;
    let ts: Vec<usize> = std::env::var("SD_AF2_T").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![128]);
    for &t in &ts {
        for pos in [2048usize, 8192, 16384, 32768 - t] {
            fx.ar.put(b.ctrl, &[SdCtrl { pos: pos as u32, ..Default::default() }]);
            let flop = 4.0 * (NH * HD) as f64 * (0..t).map(|i| (pos + i + 1) as f64).sum::<f64>();
            let best = |f: &dyn Fn(&Enc)| -> f64 { run(&fx.gpu, &fx.ar, 1, f); (0..3).map(|_| run(&fx.gpu, &fx.ar, reps, f)).fold(f64::MAX, f64::min) };
            let mo = best(&|e| enc_old(e, b, t));
            println!("T {:3} pos {:5}: old            {:7.3} ms {:5.2} TFLOPS", t, pos, mo, flop / mo / 1e9);
            for &(name, rr) in VARIANTS {
                if let Some(o) = &only { if !o.iter().any(|x| x == name) { continue; } }
                let ms = best(&|e| enc_v2(e, b, t, name, rr));
                println!("              {:18}: {:7.3} ms {:5.2} TFLOPS  {:.2}x", name, ms, flop / ms / 1e9, mo / ms);
            }
        }
    }
}
