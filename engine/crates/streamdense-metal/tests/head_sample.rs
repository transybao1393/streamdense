//! U8 self-test: embed_q4, argmax_b, topk32, copy_bytes, select_row, advance1 (engine/shaders/head_sample.metal).
//! - embed_q4 vs CPU dequant and MLX QuantizedEmbedding rows (embed.safetensors).
//! - argmax_b: ties -> lowest index (MLX tie rows + planted ties at V = 248320), top1/top2, rec record.
//! - lm_head: q4b_gemv m3 -> argmax_b / topk32 vs MLX (head.safetensors); topk32 + lse vs CPU at V = 248320.
//! Run: cargo test --release --offline -p streamdense-metal --test head_sample -- --nocapture --test-threads 1
mod a_util;
use a_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu, SdCtrl};

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16", "head_sample"]).expect("compile");
    for k in ["embed_q4", "argmax_b", "topk32", "copy_bytes", "select_row", "advance1", "q4b_gemv_k1_m3", "q4b_gemv_k4_m3"] {
        g.pipeline(k).unwrap();
    }
    g
}

fn run(g: &Gpu, ar: &Ar, f: impl FnOnce(&Enc)) {
    let cb = g.command_buffer();
    let e = Enc::new(g, &cb, &ar.buf);
    f(&e);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
}

fn argmax(g: &Gpu, ar: &Ar, lg: usize, ctrl: usize, rec: usize, v: usize, k: usize) {
    run(g, ar, |e| { e.pso("argmax_b").b(0, lg).b(1, ctrl).b(2, rec).u(3, v as u32); e.groups((k, 1), 1024); });
}
fn topk(g: &Gpu, ar: &Ar, lg: usize, idx: usize, val: usize, lse: usize, v: usize, k: usize) {
    run(g, ar, |e| { e.pso("topk32").b(0, lg).b(1, idx).b(2, val).b(3, lse).u(4, v as u32); e.groups((k, 1), 1024); });
}

/// CPU greedy on bf16-rounded logits: (argmax lowest index, top1, top2 = best of the rest)
fn cpu_argmax(l: &[f32]) -> (u32, f32, f32) {
    let r: Vec<f32> = l.iter().map(|&x| rbf(x)).collect();
    let mut bi = 0;
    for i in 1..r.len() { if r[i] > r[bi] { bi = i; } }
    let t2 = r.iter().enumerate().filter(|(i, _)| *i != bi).map(|(_, &v)| v).fold(f32::NEG_INFINITY, f32::max);
    (bi as u32, r[bi], t2)
}
/// CPU top-32 (value desc, index asc) + lse (f64) on bf16-rounded logits
fn cpu_topk(l: &[f32]) -> (Vec<u32>, Vec<f32>, f64) {
    let r: Vec<f32> = l.iter().map(|&x| rbf(x)).collect();
    let mut ix: Vec<u32> = (0..r.len() as u32).collect();
    ix.sort_by(|&a, &b| r[b as usize].partial_cmp(&r[a as usize]).unwrap().then(a.cmp(&b)));
    ix.truncate(32);
    let mx = r[ix[0] as usize] as f64;
    let s: f64 = r.iter().map(|&v| (v as f64 - mx).exp()).sum();
    let vals = ix.iter().map(|&i| r[i as usize]).collect();
    (ix, vals, mx + s.ln())
}

#[test]
fn embed_rows_vs_mlx() {
    let g = gpu();
    let Some(v) = vectors("embed") else { return; };
    let (w, s, b) = (v.u32s("tab.w"), v.u16s("tab.s"), v.u16s("tab.b"));
    assert_eq!(v.dtype("rows.ids"), "U32", "U32 index decision");
    let ids = v.u32s("rows.ids");
    let y = v.u16s("rows.y");
    let h = v.shape("rows.y")[1];
    let (wc, gc) = (h / 8, h / 64);
    let rb = h / 2 + gc * 4;
    assert_eq!(rb, 2880);
    let mut stage = vec![0u8; ids.len() * rb];
    for (t, &id) in ids.iter().enumerate() {
        let row = &mut stage[t * rb..(t + 1) * rb];
        let id = id as usize;
        for j in 0..wc { row[j * 4..j * 4 + 4].copy_from_slice(&w[id * wc + j].to_le_bytes()); }
        for j in 0..gc {
            row[h / 2 + j * 2..h / 2 + j * 2 + 2].copy_from_slice(&s[id * gc + j].to_le_bytes());
            row[h / 2 + gc * 2 + j * 2..h / 2 + gc * 2 + j * 2 + 2].copy_from_slice(&b[id * gc + j].to_le_bytes());
        }
    }
    let mut ar = Ar::new(&g, 1 << 20);
    let st = ar.alloc_put(&stage);
    let out = ar.alloc(ids.len() * h * 2);
    let t = ids.len();
    run(&g, &ar, |e| { e.pso("embed_q4").b(0, st).b(1, out).u(2, t as u32); e.threads((h / 8, t), 64); });
    let gk = ar.get::<u16>(out, t * h);
    let cpu: Vec<u16> = ids.iter().flat_map(|&id| deq_row(&w, &s, &b, h, id as usize).into_iter().map(to_bf)).collect();
    let (u, n) = cmp_bf(&gk, &y);
    println!("  embed_q4 8 rows x {}: vs MLX max ulp {} ({} / {} differ), == CPU dequant {}", h, u, n, y.len(), gk == cpu);
    assert!(u == 0 && gk == cpu);
}

#[test]
fn argmax_ties_and_mlx() {
    let g = gpu();
    let mut ar = Ar::new(&g, 16 << 20);
    let ctrl = ar.alloc(std::mem::size_of::<SdCtrl>());
    let rec = ar.alloc(4096 * 16);
    if let Some(v) = vectors("head") {
        let tl = v.u16s("tie.logits");
        let sh = v.shape("tie.logits");
        let am = v.u32s("tie.argmax");
        let lf: Vec<f32> = tl.iter().map(|&x| bf(x)).collect();
        let lo = ar.alloc_put(&lf);
        // 3 rows: process 2 rows per dispatch (ctrl holds tok[0..2]) -> rows 0,1 then row 2 alone
        ar.put(ctrl, &[SdCtrl::default()]);
        argmax(&g, &ar, lo, ctrl, rec, sh[1], 2);
        let c01 = ar.get::<SdCtrl>(ctrl, 1)[0];
        argmax(&g, &ar, lo + 2 * sh[1] * 4, ctrl, rec, sh[1], 1);
        let c2 = ar.get::<SdCtrl>(ctrl, 1)[0];
        let got = [c01.tok[0], c01.tok[1], c2.tok[0]];
        println!("  MLX tie rows: engine {:?}, MLX {:?}, expected lowest {:?}", got, am, [3, 500, 10]);
        assert_eq!(got.to_vec(), am);
    }
    // planted ties at V = 248320 (values equal only after bf16 rounding), 2 rows, rec at ctrl.pos
    let vv = 248320;
    let mut rng = Rng::new(11);
    let mut l: Vec<f32> = (0..2 * vv).map(|_| rng.normal() * 3.0).collect();
    for (i, &p) in [200_001usize, 17, 248_319].iter().enumerate() { l[p] = 40.0 + i as f32 * 0.01; }  // all bf16 40.0
    for (i, &p) in [vv + 248_000, vv + 248_300, vv + 99_999].iter().enumerate() { l[p] = 41.0 + i as f32 * 0.02; }
    let lo = ar.alloc_put(&l);
    ar.put(ctrl, &[SdCtrl { pos: 37, ..Default::default() }]);
    argmax(&g, &ar, lo, ctrl, rec, vv, 2);
    let c = ar.get::<SdCtrl>(ctrl, 1)[0];
    let r = ar.get::<[f32; 4]>(rec + 37 * 16, 2);
    let mut ok = true;
    for t in 0..2 {
        let (ti, t1, t2) = cpu_argmax(&l[t * vv..(t + 1) * vv]);
        let rt = r[t];
        let e = c.tok[t] == ti && c.top1[t] == t1 && c.top2[t] == t2 && rt[0].to_bits() == ti && rt[1] == t1 && rt[2] == t2;
        println!("  V=248320 row {}: engine tok {} top1 {} top2 {} | CPU {} {} {} | rec ok {}", t, c.tok[t], c.top1[t], c.top2[t], ti, t1, t2,
                 rt[0].to_bits() == ti);
        ok &= e;
    }
    assert_eq!(c.tok, [17, 99_999]);
    assert!(ok);
    // random rows without planted ties, 20 trials
    let mut bad = 0;
    for trial in 0..20 {
        let l: Vec<f32> = (0..vv).map(|_| rng.normal() * (1.0 + trial as f32)).collect();
        ar.put(lo, &l);
        ar.put(ctrl, &[SdCtrl::default()]);
        argmax(&g, &ar, lo, ctrl, rec, vv, 1);
        let c = ar.get::<SdCtrl>(ctrl, 1)[0];
        let (ti, t1, t2) = cpu_argmax(&l);
        if (c.tok[0], c.top1[0], c.top2[0]) != (ti, t1, t2) { bad += 1; }
    }
    println!("  20 random V=248320 rows: {} mismatches vs CPU", bad);
    assert_eq!(bad, 0);
}

#[test]
fn lm_head_mlx_and_topk() {
    let g = gpu();
    let mut ar = Ar::new(&g, 32 << 20);
    let ctrl = ar.alloc(std::mem::size_of::<SdCtrl>());
    let rec = ar.alloc(64 * 16);
    let idx = ar.alloc(8 * 32 * 4);
    let val = ar.alloc(8 * 32 * 4);
    let lse = ar.alloc(8 * 4);
    if let Some(v) = vectors("head") {
        let (w, s, b, x) = (v.u32s("lm.w"), v.u16s("lm.s"), v.u16s("lm.b"), v.u16s("lm.x"));
        let (r, c) = (v.shape("lm.w")[0], v.shape("lm.x")[1]);
        let k = v.shape("lm.x")[0];
        let (wo, so, bo, xo) = (ar.alloc_put(&w), ar.alloc_put(&s), ar.alloc_put(&b), ar.alloc_put(&x));
        let lg = ar.alloc(k * r * 4);
        run(&g, &ar, |e| {
            e.pso("q4b_gemv_k4_m3").b(0, wo).b(1, so).b(2, bo).b(3, xo).b(4, lg).u(5, r as u32).u(6, c as u32).b(7, lg).b(8, lg);
            e.groups(((r + 3) / 4, 1), 64);
        });
        let lf = ar.get::<f32>(lg, k * r);
        let lb: Vec<u16> = lf.iter().map(|&x| to_bf(x)).collect();
        let (u1, n1) = cmp_bf(&lb, &v.u16s("lm.logits_k1"));
        let mut toks = Vec::new();
        for t0 in (0..k).step_by(2) {
            ar.put(ctrl, &[SdCtrl::default()]);
            argmax(&g, &ar, lg + t0 * r * 4, ctrl, rec, r, 2);
            let cc = ar.get::<SdCtrl>(ctrl, 1)[0];
            toks.extend_from_slice(&cc.tok);
        }
        let am = v.u32s("lm.argmax");
        topk(&g, &ar, lg, idx, val, lse, r, k);
        let (gi, gv) = (ar.get::<u32>(idx, k * 32), ar.get::<f32>(val, k * 32));
        let (mi, mv) = (v.u32s("lm.top32_idx"), v.u16s("lm.top32_val"));
        // compare values exactly; indices as sets within equal-value runs
        let mut vals_eq = true;
        let mut sets_eq = true;
        for t in 0..k {
            for j in 0..32 { vals_eq &= to_bf(gv[t * 32 + j]) == mv[t * 32 + j]; }
            let mut a: Vec<(u16, u32)> = (0..32).map(|j| (to_bf(gv[t * 32 + j]), gi[t * 32 + j])).collect();
            let mut m: Vec<(u16, u32)> = (0..32).map(|j| (mv[t * 32 + j], mi[t * 32 + j])).collect();
            // drop the last value run (may be cut differently at position 32)
            let last = a[31].0;
            a.retain(|p| p.0 != last); m.retain(|p| p.0 != last);
            a.sort(); m.sort();
            sets_eq &= a == m;
        }
        println!("  lm 4x512: logits vs MLX k1 max ulp {} ({} differ); argmax {:?} vs MLX {:?}; top32 vals == {} idx sets == {}",
                 u1, n1, toks, am, vals_eq, sets_eq);
        assert!(u1 == 0 && toks == am && vals_eq && sets_eq);
    }
    // topk32 + lse at V = 248320 vs CPU, with planted ties
    let vv = 248320;
    let mut rng = Rng::new(5);
    let mut l: Vec<f32> = (0..2 * vv).map(|_| rng.normal() * 2.5).collect();
    for &p in &[1000usize, 5, 248_000, 77_777] { l[p] = 12.0 + (p % 7) as f32 * 0.001; }
    let lg = ar.alloc_put(&l);
    topk(&g, &ar, lg, idx, val, lse, vv, 2);
    let (gi, gv, gl) = (ar.get::<u32>(idx, 64), ar.get::<f32>(val, 64), ar.get::<f32>(lse, 2));
    let mut ok = true;
    for t in 0..2 {
        let (ci, cv, cl) = cpu_topk(&l[t * vv..(t + 1) * vv]);
        let e = gi[t * 32..(t + 1) * 32] == ci[..] && gv[t * 32..(t + 1) * 32] == cv[..];
        let rel = ((gl[t] as f64 - cl) / cl).abs();
        println!("  topk32 V=248320 row {}: idx+val == CPU {}, first idx {:?}, lse {} vs {:.6} (rel {:.1e})", t, e, &gi[t * 32..t * 32 + 5], gl[t], cl, rel);
        ok &= e && rel < 1e-5;
    }
    assert_eq!(&gi[..4], &[5, 1000, 77_777, 248_000]);
    // V < 32
    let small: Vec<f32> = (0..20).map(|i| (i as f32 * 0.37).sin()).collect();
    ar.put(lg, &small);
    topk(&g, &ar, lg, idx, val, lse, 20, 1);
    let (gi, gv) = (ar.get::<u32>(idx, 32), ar.get::<f32>(val, 32));
    let (ci, cv, _) = cpu_topk(&small);
    let sm = gi[..20] == ci[..20] && gv[..20] == cv[..20] && gi[20..].iter().all(|&x| x == u32::MAX) && gv[20..].iter().all(|&x| x == f32::NEG_INFINITY);
    println!("  topk32 V=20: == CPU + padding {}", sm);
    assert!(ok && sm);
}

#[test]
fn copy_select_advance() {
    let g = gpu();
    let mut ar = Ar::new(&g, 1 << 20);
    let src: Vec<u8> = (0..1001u32).map(|i| (i * 7 + 3) as u8).collect();
    let so = ar.alloc_put(&src);
    let d = ar.alloc(2048);
    ar.fill(d, 2048, 0);
    run(&g, &ar, |e| { e.pso("copy_bytes").b(0, so + 3).b(1, d + 5).u(2, 997); e.threads((997, 1), 256); });
    let got = ar.get::<u8>(d, 1010);
    let cok = got[5..1002] == src[3..1000] && got[..5].iter().all(|&x| x == 0) && got[1002..].iter().all(|&x| x == 0);
    // select_row
    let n = 5120;
    let rows: Vec<u16> = (0..2 * n as u32).map(|i| i as u16).collect();
    let ro = ar.alloc_put(&rows);
    let dst = ar.alloc(n * 2);
    let ctrl = ar.alloc(72);
    let mut sok = true;
    for (mr, want) in [(0u32, 0usize), (1, 0), (2, 1)] {
        ar.put(ctrl, &[SdCtrl { mtp_rows: mr, ..Default::default() }]);
        run(&g, &ar, |e| { e.pso("select_row").b(0, ro).b(1, dst).u(2, n as u32).b(3, ctrl); e.threads((n, 1), 256); });
        sok &= ar.get::<u16>(dst, n) == rows[want * n..(want + 1) * n];
    }
    ar.put(ctrl, &[SdCtrl { pos: 41, step: 9, ..Default::default() }]);
    run(&g, &ar, |e| { e.pso("advance1").b(0, ctrl); e.threads((1, 1), 1); });
    let c = ar.get::<SdCtrl>(ctrl, 1)[0];
    let aok = c.pos == 42 && c.step == 9;
    println!("  copy_bytes (unaligned) {}, select_row {}, advance1 {}", cok, sok, aok);
    assert!(cok && sok && aok);
}
