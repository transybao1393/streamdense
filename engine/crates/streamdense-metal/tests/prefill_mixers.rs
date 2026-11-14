//! S1 self-test (PF-B): the prefill token-mixer kernels vs the existing per-row kernels.
//! - `dn_conv_pf` (deltanet.metal) vs T one-token `dn_conv` dispatches (the stepwise path): conv output and the conv
//!   state of the live copy bit-exact, the other copy untouched; tiny (kh 2, vh 6) and real (kh 16, vh 48) dims,
//!   role 0 and 1, T in {1, 2, 3, 7, 8, 9, 37, 128, 131, 256}.
//! - `attn_flash_b16` (attn_gated.metal) vs the bit-exact MLX mirror `attn_sdpa_vec_b16` run on the same T rows: tiny
//!   (4 / 2 heads) and real (24 / 4) shapes, pos {0, 5, 1000 (rows cross the 2-pass switch at N = 1024), 2900},
//!   T {1, 8, 17, 33, 128, 131}. Bar (ulp at max(|flash|, |ref|), where |ref| >= 1/16): <= 2 ulp
//!   (normal softmax; the output is bf(bf(o) * sigmoid(gate)), so a 1-ulp flip of o in a higher binade than the gated
//!   output is 2 output ulp: measured case o ~0.134, gate 2.16 -> 0.1187 vs 0.1196) / <= 3 ulp (sharp softmax,
//!   q std x 2.5) on rows with N < 1024 (the mirror's single-pass path). Rows with N >= 1024 use MLX's 2-pass mirror,
//!   which is itself less accurate (flash is the closer one to f64 there), so they are printed only. Plus an f64
//!   reference on rows 0, T/2, T-1 (all heads): flash <= 2 ulp. Always finite, no writes past T rows.
//! Run: cargo test --release --offline -p streamdense-metal --test prefill_mixers -- --nocapture --test-threads 1
mod b_util;
use b_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu, SdCtrl};

const D: usize = 128;
const HD: usize = 256;

fn run(gpu: &Gpu, ar: &Ar, f: impl Fn(&Enc)) {
    let cb = gpu.command_buffer();
    let e = Enc::new(gpu, &cb, &ar.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

// ------------------------------------------------------------------------------------------- DeltaNet conv
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DnDims { kh: u32, vh: u32, conv: u32, prow: u32, b_off: u32, a_off: u32, layers: u32, pad: u32 }
fn dims(kh: usize, vh: usize, layers: usize) -> DnDims {
    let conv = 2 * kh * D + vh * D;
    let b_off = conv + vh * D;
    DnDims { kh: kh as u32, vh: vh as u32, conv: conv as u32, prow: (b_off + 2 * vh) as u32, b_off: b_off as u32,
             a_off: (b_off + vh) as u32, layers: layers as u32, pad: 0 }
}

#[test]
fn dn_conv_pf_vs_dn_conv() {
    let mut gpu = Gpu::from_shader_files(&["common_bf16", "deltanet"]).unwrap();
    for k in ["dn_conv", "dn_conv_pf"] { gpu.pipeline(k).unwrap(); }
    let tmax = 256;
    let mut n_cases = 0;
    for (kh, vh) in [(2usize, 6usize), (16, 48)] {
        let (layers, layer) = (3usize, 1usize);
        let dd = dims(kh, vh, layers);
        let (c, prow) = (dd.conv as usize, dd.prow as usize);
        let st_len = 2 * layers * 3 * c;
        let mut ar = Ar::new(&gpu, tmax * prow * 2 + c * 8 + 2 * st_len * 2 + 2 * tmax * c * 2 + (1 << 16));
        let (p, w, st, out, ctrl) = (ar.alloc(tmax * prow * 2), ar.alloc(c * 8), ar.alloc(st_len * 2), ar.alloc(tmax * c * 2), ar.alloc(256));
        let mut r = Rng::new(kh as u64);
        let (pv, wv, sv) = (r.bf_n(tmax * prow, 1.0), r.bf_n(c * 4, 0.5), r.bf_n(st_len, 1.0));
        ar.put(p, &pv); ar.put(w, &wv);
        for role in [0u32, 1] {
            ar.put(ctrl, &[SdCtrl { role, ..Default::default() }]);
            for t_n in [1usize, 2, 3, 7, 8, 9, 37, 128, 131, 256] {
                // stepwise reference: T one-token dn_conv dispatches (ck off)
                ar.put(st, &sv); ar.fill(out, tmax * c * 2, 0);
                run(&gpu, &ar, |e| for t in 0..t_n {
                    e.pso("dn_conv").b(0, p + t * prow * 2).b(1, w).b(2, st).b(3, out + t * c * 2).b(4, ctrl)
                        .u(5, layer as u32).u(6, 1).u(7, !0).bytes(8, &dd).threads((c, 1), 256);
                });
                let (o_ref, s_ref): (Vec<u16>, Vec<u16>) = (ar.get(out, t_n * c), ar.get(st, st_len));
                ar.put(st, &sv); ar.fill(out, tmax * c * 2, 0);
                run(&gpu, &ar, |e| {
                    e.pso("dn_conv_pf").b(0, p).b(1, w).b(2, st).b(3, out).b(4, ctrl).u(5, layer as u32).u(6, t_n as u32)
                        .bytes(7, &dd).groups((c.div_ceil(32), 1), 32 * t_n.div_ceil(8));
                });
                let (o_pf, s_pf): (Vec<u16>, Vec<u16>) = (ar.get(out, t_n * c), ar.get(st, st_len));
                let past: Vec<u16> = ar.get(out + t_n * c * 2, (tmax - t_n) * c);
                let (ou, on) = cmp_bf(&o_pf, &o_ref);
                let (su, sn) = cmp_bf(&s_pf, &s_ref);
                // the other role's copy (and other layers) must equal the input
                let other = ((1 - role as usize) * layers) * 3 * c;
                let untouched = s_pf[other..other + layers * 3 * c] == sv[other..other + layers * 3 * c];
                let clean = past.iter().all(|&x| x == 0);
                if t_n == 1 || t_n == 9 || t_n == 256 || ou > 0 || su > 0 || !untouched || !clean {
                    println!("  dn_conv_pf kh {:2} vh {:2} role {} T {:3}: out {} ulp ({} differ), state {} ulp ({} differ), other copy untouched {}, no write past T {}",
                             kh, vh, role, t_n, ou, on, su, sn, untouched, clean);
                }
                assert!(ou == 0 && su == 0 && untouched && clean, "dn_conv_pf != stepwise dn_conv (kh {} role {} T {})", kh, role, t_n);
                n_cases += 1;
            }
        }
    }
    println!("dn_conv_pf: {} cases, all bit-exact vs T one-token dn_conv dispatches", n_cases);
}

// ------------------------------------------------------------------------------------------- attention
/// bf16 grid spacing at magnitude |x| (a 1-step flip across a binade edge counts as 1 ulp of the larger side).
fn sig_bf(x: f32) -> f32 {
    let e = rbf(x.abs().exp());
    let y = rbf(1.0 / rbf(1.0 + e));
    if x < 0.0 { y } else { rbf(1.0 - y) }
}
fn ulp_at(x: f32) -> f32 { f32::from_bits(((x.abs().to_bits() >> 23) << 23).saturating_sub(7 << 23)) }

struct AttnCase { nh: usize, nkv: usize }

fn sdpa(e: &Enc, b: &AB, t: usize, lim: u32) {
    e.pso("attn_sdpa_vec_b16").b(0, b.qr).b(1, b.kc).b(2, b.vc).b(3, b.p).b(4, b.ao).b(5, b.ctrl)
        .u(6, b.maxctx as u32).u(7, (b.nh / b.nkv) as u32).u(8, b.prow as u32).u(9, lim).u(10, 1024)
        .groups((b.nh, t), 1024);
}
fn flash(e: &Enc, b: &AB, t: usize) {
    let grp = b.nh / b.nkv;
    e.pso("attn_flash_b16").b(0, b.qr).b(1, b.kc).b(2, b.vc).b(3, b.p).b(4, b.ao).b(5, b.ctrl)
        .u(6, b.maxctx as u32).u(7, grp as u32).u(8, b.prow as u32).u(9, t as u32)
        .groups((t.div_ceil(8), b.nkv), 32 * grp);
}
struct AB { nh: usize, nkv: usize, maxctx: usize, prow: usize, qr: usize, kc: usize, vc: usize, p: usize, ao: usize, ctrl: usize }

#[test]
fn attn_flash_vs_sdpa_mirror() {
    let mut gpu = Gpu::from_shader_files(&["common_bf16", "attn_gated"]).unwrap();
    for k in ["attn_sdpa_vec_b16", "attn_flash_b16"] { gpu.pipeline(k).unwrap(); }
    let maxctx = 3072;
    let tmax = 131;
    let (mut worst_n, mut worst_s, mut absmax, mut nd, mut ntot) = (0u32, 0u32, 0f32, 0usize, 0usize);
    let (mut worst_2p, mut worst_f, mut worst_m) = (0u32, 0u32, 0u32);
    for case in [AttnCase { nh: 4, nkv: 2 }, AttnCase { nh: 24, nkv: 4 }] {
        let (nh, nkv) = (case.nh, case.nkv);
        let prow = nh * 2 * HD + 2 * nkv * HD;
        let mut ar = Ar::new(&gpu, tmax * prow * 2 + tmax * nh * HD * 4 + 2 * nkv * maxctx * HD * 2 + (1 << 20));
        let b = AB { nh, nkv, maxctx, prow, qr: ar.alloc(tmax * nh * HD * 2), kc: ar.alloc(nkv * maxctx * HD * 2),
                     vc: ar.alloc(nkv * maxctx * HD * 2), p: ar.alloc(tmax * prow * 2), ao: ar.alloc(tmax * nh * HD * 2),
                     ctrl: ar.alloc(256) };
        let mut r = Rng::new(nh as u64 * 7);
        ar.put(b.kc, &r.bf_n(nkv * maxctx * HD, 1.0));
        ar.put(b.vc, &r.bf_n(nkv * maxctx * HD, 1.0));
        ar.put(b.p, &r.bf_n(tmax * prow, 1.5));
        for sharp in [false, true] {
            ar.put(b.qr, &r.bf_n(tmax * nh * HD, if sharp { 2.5 } else { 1.0 }));
            for pos in [0usize, 5, 1000, 2900] {
                for t in [1usize, 8, 17, 33, 128, 131] {
                    if pos + t > maxctx { continue; }
                    ar.put(b.ctrl, &[SdCtrl { pos: pos as u32, ..Default::default() }]);
                    let n = t * nh * HD;
                    ar.fill(b.ao, tmax * nh * HD * 2, 0xff);
                    run(&gpu, &ar, |e| sdpa(e, &b, t, 8192));
                    let refo: Vec<u16> = ar.get(b.ao, n);
                    ar.fill(b.ao, tmax * nh * HD * 2, 0xff);
                    run(&gpu, &ar, |e| flash(e, &b, t));
                    let fl: Vec<u16> = ar.get(b.ao, n);
                    let past: Vec<u16> = ar.get(b.ao + n * 2, (tmax - t) * nh * HD);
                    // vs the mirror, split by the mirror's path: N < 1024 single-pass sdpa_vector, N >= 1024 MLX 2-pass
                    let (mut wu, mut wu2, mut am, mut d, mut fin) = (0u32, 0u32, 0f32, 0usize, true);
                    for i in 0..n {
                        let (x, y) = (bf(fl[i]), bf(refo[i]));
                        if !x.is_finite() { fin = false; continue; }
                        if fl[i] != refo[i] { d += 1; }
                        let a = (x - y).abs();
                        am = am.max(a);
                        let two = pos + i / (nh * HD) + 1 >= 1024;
                        if y.abs() >= 1.0 / 16.0 {
                            let u = (a / ulp_at(x.abs().max(y.abs()))).round() as u32;
                            if two { wu2 = wu2.max(u); } else { wu = wu.max(u); }
                        } else { assert!(a <= 1.6e-2, "attn_flash small-output abs diff {} (pos {} T {})", a, pos, t); }
                    }
                    // f64 reference on rows 0, T/2, T-1 (all heads): flash and mirror errors, gated as attn_reduce_gate
                    let (mut ef, mut em) = (0u32, 0u32);
                    let mut rows = vec![0, t / 2, t - 1]; rows.dedup();
                    for &tt in &rows {
                        let nk = pos + tt + 1;
                        for hh in 0..nh {
                            let kvh = hh / (nh / nkv);
                            let q: Vec<u16> = ar.get(b.qr + (tt * nh + hh) * HD * 2, HD);
                            let kk: Vec<u16> = ar.get(b.kc + kvh * maxctx * HD * 2, nk * HD);
                            let vv: Vec<u16> = ar.get(b.vc + kvh * maxctx * HD * 2, nk * HD);
                            let sc: Vec<f64> = (0..nk).map(|j| (0..HD).map(|e| bf(q[e]) as f64 / 16.0 * bf(kk[j * HD + e]) as f64).sum()).collect();
                            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let w: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                            let l: f64 = w.iter().sum();
                            let gate: Vec<u16> = ar.get(b.p + (tt * prow + hh * 2 * HD + HD) * 2, HD);
                            for dd in 0..HD {
                                let o = (0..nk).map(|j| w[j] * bf(vv[j * HD + dd]) as f64).sum::<f64>() / l;
                                let g = rbf(rbf(o as f32) * sig_bf(bf(gate[dd])));
                                if g.abs() < 1.0 / 16.0 { continue; }
                                let i = (tt * nh + hh) * HD + dd;
                                let uf = ((bf(fl[i]) - g).abs() / ulp_at(g.abs().max(bf(fl[i]).abs()))).round() as u32;
                                let um = ((bf(refo[i]) - g).abs() / ulp_at(g.abs().max(bf(refo[i]).abs()))).round() as u32;
                                ef = ef.max(uf); em = em.max(um);
                            }
                        }
                    }
                    let clean = past.iter().all(|&x| x == 0xffff);
                    println!("  attn_flash nh {:2}/{} {} pos {:4} T {:3}: vs mirror max {} ulp (N<1024) / {} ulp (N>=1024, 2-pass), max abs {:.1e}, {:.2}% differ | vs f64 (3 rows): flash {} ulp, mirror {} ulp{}",
                             nh, nkv, if sharp { "sharp " } else { "normal" }, pos, t, wu, wu2, am, 100.0 * d as f64 / n as f64, ef, em,
                             if fin && clean { "" } else { "  NON-FINITE / WROTE PAST T" });
                    assert!(fin && clean, "attn_flash non-finite or wrote past T rows");
                    assert!(wu <= if sharp { 3 } else { 2 }, "attn_flash vs single-pass mirror: {} ulp (sharp {}, pos {}, T {})", wu, sharp, pos, t);
                    assert!(ef <= 2, "attn_flash vs f64: {} ulp (sharp {}, pos {}, T {})", ef, sharp, pos, t);
                    if sharp { worst_s = worst_s.max(wu); } else { worst_n = worst_n.max(wu); }
                    worst_2p = worst_2p.max(wu2); worst_f = worst_f.max(ef); worst_m = worst_m.max(em);
                    absmax = absmax.max(am); nd += d; ntot += n;
                }
            }
        }
    }
    println!("attn_flash_b16 vs attn_sdpa_vec_b16: worst {} ulp (normal) / {} ulp (sharp) on N < 1024 rows, {} ulp on N >= 1024 rows (MLX 2-pass mirror), max abs {:.1e}, {:.3}% of {} outputs differ; vs f64: flash worst {} ulp, mirror worst {} ulp",
             worst_n, worst_s, worst_2p, absmax, 100.0 * nd as f64 / ntot as f64, ntot, worst_f, worst_m);
}
