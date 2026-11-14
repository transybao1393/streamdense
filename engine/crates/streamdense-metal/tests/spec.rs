//! S2 self-test (P2 U18): spec.metal - spec_decide, mtp_prep, mtp_draft, mtp_advance.
//! - spec_decide truth table (accept / reject / eos at a0 or a1, a0 eos == d, 2-entry eos list, n_eos = 0), rec
//!   zeroing (only rec[p + 1], only on reject), role flip, step counter, fields it must not touch.
//! - mtp_prep: direct mode (T = 1, 2, 4, 8) and spec mode (accept rows 2, reject rows 1) == two rmsnorm_b calls
//!   writing the cat halves (out_stride 2H), bitwise; mctrl = {pos: mtp_len, mtp_rows, role 0}; rows past `rows` and
//!   ctrl untouched. H = 5120 (27B) and 512 (tiny).
//! - mtp_draft == argmax_b (row 0) incl. planted bf16 ties, all-equal rows, -inf, max at V - 1, at V = 248320 / 2048;
//!   ctrl.mtp_len += mctrl.mtp_rows; nothing else in ctrl changes.
//! - mtp_advance: ctrl.mtp_len += mctrl.mtp_rows only.
//! Run: cargo test --release --offline -p streamdense-metal --test spec -- --nocapture --test-threads 1
mod a_util;
use a_util::*;
use streamdense_metal::{commit, wait, Enc, Gpu, SdCtrl};

fn gpu() -> Gpu {
    let mut g = Gpu::from_shader_files(&["common_bf16", "norm_bf16", "head_sample", "spec"]).expect("compile");
    for k in ["rmsnorm_b", "argmax_b", "mtp_prep", "mtp_draft", "mtp_advance", "spec_decide"] { g.pipeline(k).unwrap(); }
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

fn put_ctrl(ar: &Ar, off: usize, c: &SdCtrl) { ar.put(off, std::slice::from_ref(c)); }
fn get_ctrl(ar: &Ar, off: usize) -> SdCtrl { ar.get::<SdCtrl>(off, 1)[0] }

// ------------------------------------------------------------------------------------------------ spec_decide
#[test]
fn spec_decide_truth_table() {
    let g = gpu();
    let mut ar = Ar::new(&g, 1 << 20);
    let max_ctx = 64;
    let (ctrl, rec) = (ar.alloc(72), ar.alloc(max_ctx * 16));
    let eos_l = [2046u32, 2044];
    let eos = ar.alloc_put(&eos_l);
    // (name, a0, a1, d, n_eos, accept, stop)
    let cases: &[(&str, u32, u32, u32, u32, bool, bool)] = &[
        ("accept", 5, 7, 5, 2, true, false),
        ("reject", 6, 7, 5, 2, false, false),
        ("accept, a1 eos (1st)", 5, 2046, 5, 2, true, true),
        ("accept, a1 eos (2nd)", 5, 2044, 5, 2, true, true),
        ("a0 eos == d -> reject path", 2044, 7, 2044, 2, false, true),
        ("reject, a0 eos", 2046, 7, 5, 2, false, true),
        ("reject, a1 eos ignored", 6, 2046, 5, 2, false, false),
        ("accept, n_eos 0 (a1 = 2046 not eos)", 5, 2046, 5, 0, true, false),
        ("a0 == d == 2044, n_eos 0 -> accept", 2044, 9, 2044, 0, true, false),
        ("accept, a0 == a1 == d", 9, 9, 9, 2, true, false),
    ];
    let mut ok = true;
    for (i, &(name, a0, a1, d, n_eos, acc, stop)) in cases.iter().enumerate() {
        for role in [0u32, 1] {
            let p = 10 + i as u32;
            let c0 = SdCtrl { pos: p, role, mtp_len: p, mtp_rows: 9, n_emit: 9, emit: [111, 222], draft: d, tok: [a0, a1],
                              top1: [1.5, 2.5], top2: [0.5, 0.25], stop: 7, step: 41, pad: [3, 4] };
            put_ctrl(&ar, ctrl, &c0);
            let rec0: Vec<f32> = (0..max_ctx * 4).map(|j| j as f32 + 0.5).collect();
            ar.put(rec, &rec0);
            run(&g, &ar, |e| { e.pso("spec_decide").b(0, ctrl).b(1, rec).b(2, eos).u(3, n_eos).threads((1, 1), 1); });
            let c = get_ctrl(&ar, ctrl);
            let r = ar.get::<f32>(rec, max_ctx * 4);
            let mut want = c0;
            want.step = 42;
            want.stop = stop as u32;
            let mut want_rec = rec0.clone();
            if acc {
                want.emit = [d, a1]; want.n_emit = 2; want.pos = p + 2; want.mtp_rows = 2;
            } else {
                want.emit = [a0, 0]; want.n_emit = 1; want.pos = p + 1; want.mtp_rows = 1; want.role = role ^ 1;
                for j in 0..4 { want_rec[(p as usize + 1) * 4 + j] = 0.0; }
            }
            let good = c == want && r == want_rec;
            if !good || role == 0 {
                println!("  {:38} role {}: n_emit {} emit {:?} pos {} rows {} role {} stop {} step {} rec ok {} -> {}",
                         name, role, c.n_emit, c.emit, c.pos - p, c.mtp_rows, c.role, c.stop, c.step, r == want_rec,
                         if good { "ok" } else { "FAIL" });
            }
            if !good { println!("    got  {:?}\n    want {:?}", c, want); }
            ok &= good;
        }
    }
    assert!(ok, "spec_decide truth table");
}

// ------------------------------------------------------------------------------------------------ mtp_prep
fn rms_ref(g: &Gpu, ar: &Ar, x: usize, w: usize, out: usize, h: usize, rows: usize, eps: f32) {
    run(g, ar, |e| { e.pso("rmsnorm_b").b(0, x).b(1, w).b(2, out).u(3, h as u32).f(4, eps).u(5, (2 * h) as u32); e.groups((rows, 1), 1024); });
}

#[test]
fn mtp_prep_modes_vs_rmsnorm_b() {
    let g = gpu();
    let eps = 1e-6f32;
    let mut ok = true;
    for &h in &[5120usize, 512] {
        let mut rng = Rng::new(h as u64);
        let tmax = 8;
        let mut ar = Ar::new(&g, 64 << 20);
        let mut emb_v = rng.bf_n(tmax * h, 0.02);
        let mut hf_v = rng.bf_n(tmax * h, 1.0);
        emb_v[3] = to_bf(40.0); hf_v[h + 7] = to_bf(-300.0); // outliers
        let mut pe = rng.bf_n(h, 0.3); let mut ph = rng.bf_n(h, 0.3);
        for v in pe.iter_mut().chain(ph.iter_mut()) { *v = to_bf(1.0 + bf(*v)); } // weights carry +1
        let (emb, hf, pre_e, pre_h) = (ar.alloc_put(&emb_v), ar.alloc_put(&hf_v), ar.alloc_put(&pe), ar.alloc_put(&ph));
        let (cat, refc) = (ar.alloc(tmax * 2 * h * 2), ar.alloc(tmax * 2 * h * 2));
        let (ctrl, mctrl) = (ar.alloc(72), ar.alloc(72));
        let prep = |ar: &Ar, t: u32, groups: usize| {
            run(&g, ar, |e| {
                e.pso("mtp_prep").b(0, emb).b(1, hf).b(2, pre_e).b(3, pre_h).b(4, cat).b(5, ctrl).f(6, eps).u(7, h as u32)
                    .u(8, t).b(9, mctrl);
                e.groups((groups, 2), 1024);
            });
        };
        let c0 = SdCtrl { pos: 77, role: 1, mtp_len: 75, mtp_rows: 2, n_emit: 2, emit: [1, 2], draft: 3, tok: [4, 5],
                          top1: [1.0, 2.0], top2: [0.0, 1.0], stop: 0, step: 9, pad: [0, 0] };
        let m0 = SdCtrl { pos: 1, role: 1, mtp_len: 123, mtp_rows: 5, n_emit: 6, emit: [7, 8], draft: 9, tok: [10, 11],
                          top1: [0.0; 2], top2: [0.0; 2], stop: 1, step: 2, pad: [3, 4] };
        // direct mode
        for t in [1usize, 2, 4, 8] {
            put_ctrl(&ar, ctrl, &c0); put_ctrl(&ar, mctrl, &m0);
            ar.fill(cat, tmax * 2 * h * 2, 0xab);
            ar.fill(refc, tmax * 2 * h * 2, 0xab);
            prep(&ar, t as u32, t);
            rms_ref(&g, &ar, emb, pre_e, refc, h, t, eps);
            rms_ref(&g, &ar, hf, pre_h, refc + h * 2, h, t, eps);
            let (a, b) = (ar.get::<u16>(cat, tmax * 2 * h), ar.get::<u16>(refc, tmax * 2 * h));
            let m = get_ctrl(&ar, mctrl);
            let want_m = SdCtrl { pos: 75, mtp_rows: t as u32, role: 0, ..m0 };
            let good = a == b && m == want_m && get_ctrl(&ar, ctrl) == c0;
            println!("  H {:4} direct T={}: cat == 2x rmsnorm_b {}, mctrl {}, ctrl untouched {}", h, t, a == b, m == want_m, get_ctrl(&ar, ctrl) == c0);
            ok &= good;
        }
        // spec mode: accept (rows 2), reject (rows 1)
        for rows in [2u32, 1] {
            let c = SdCtrl { mtp_rows: rows, ..c0 };
            put_ctrl(&ar, ctrl, &c); put_ctrl(&ar, mctrl, &m0);
            ar.fill(cat, tmax * 2 * h * 2, 0xab);
            ar.fill(refc, tmax * 2 * h * 2, 0xab);
            prep(&ar, 0, 2);
            if rows == 2 {
                rms_ref(&g, &ar, emb, pre_e, refc, h, 2, eps);
            } else {
                rms_ref(&g, &ar, emb + h * 2, pre_e, refc, h, 1, eps); // emb(new t1) -> row 0
            }
            rms_ref(&g, &ar, hf, pre_h, refc + h * 2, h, rows as usize, eps);
            let (a, b) = (ar.get::<u16>(cat, tmax * 2 * h), ar.get::<u16>(refc, tmax * 2 * h));
            let m = get_ctrl(&ar, mctrl);
            let want_m = SdCtrl { pos: 75, mtp_rows: rows, role: 0, ..m0 };
            let good = a == b && m == want_m && get_ctrl(&ar, ctrl) == c;
            println!("  H {:4} spec rows={} ({}): cat == rmsnorm_b {} (row 1 untouched on reject incl.), mctrl {}, ctrl untouched {}",
                     h, rows, if rows == 2 { "accept" } else { "reject" }, a == b, m == want_m, get_ctrl(&ar, ctrl) == c);
            ok &= good;
        }
    }
    assert!(ok, "mtp_prep vs rmsnorm_b");
}

// ------------------------------------------------------------------------------------------------ mtp_draft / advance
#[test]
fn mtp_draft_vs_argmax_b_and_advance() {
    let g = gpu();
    let mut ok = true;
    for &v in &[248_320usize, 2048] {
        let mut ar = Ar::new(&g, 8 << 20);
        let (lg, rec) = (ar.alloc(v * 4), ar.alloc(64 * 16));
        let (ca, cd, mc) = (ar.alloc(72), ar.alloc(72), ar.alloc(72));
        let mut rng = Rng::new(v as u64 + 3);
        let base: Vec<f32> = (0..v).map(|_| 3.0 * rng.normal()).collect();
        let mut cases: Vec<(String, Vec<f32>)> = vec![("random".into(), base.clone())];
        // planted bf16 tie: two f32 values that round to the same bf16, the larger f32 at the higher index
        for &(i, j) in &[(5usize, v - 3), (1023, 1024), (31, 32), (0, v - 1)] {
            let mut l = base.clone();
            l[i] = 20.0; l[j] = f32::from_bits(20.0f32.to_bits() + 3); // same bf16, larger f32
            cases.push((format!("bf16 tie @{} / @{}", i, j), l));
        }
        { let mut l = base.clone(); l[v - 1] = 50.0; cases.push(("max at V-1".into(), l)); }
        cases.push(("all equal".into(), vec![1.25; v]));
        { let mut l = vec![f32::NEG_INFINITY; v]; l[v / 2] = -1e30; cases.push(("-inf but one".into(), l)); }
        { let l: Vec<f32> = base.iter().map(|&x| -x.abs() - 1.0).collect(); cases.push(("all negative".into(), l)); }
        for (name, l) in &cases {
            ar.put(lg, l);
            let c0 = SdCtrl { pos: 3, role: 1, mtp_len: 40, mtp_rows: 2, n_emit: 2, emit: [5, 6], draft: 999, tok: [7, 8],
                              top1: [0.5, 0.5], top2: [0.25, 0.25], stop: 0, step: 4, pad: [0, 0] };
            for rows in [1u32, 2] {
                let m0 = SdCtrl { mtp_rows: rows, pos: 40, ..Default::default() };
                put_ctrl(&ar, ca, &c0); put_ctrl(&ar, cd, &c0); put_ctrl(&ar, mc, &m0);
                run(&g, &ar, |e| {
                    e.pso("argmax_b").b(0, lg).b(1, ca).b(2, rec).u(3, v as u32); e.groups((1, 1), 1024);
                    e.pso("mtp_draft").b(0, lg).b(1, cd).b(2, mc).u(3, v as u32); e.groups((1, 1), 1024);
                });
                let (a, d) = (get_ctrl(&ar, ca), get_ctrl(&ar, cd));
                // CPU: argmax on bf16-rounded logits, lowest index on ties
                let r: Vec<f32> = l.iter().map(|&x| rbf(x)).collect();
                let mut bi = 0; for i in 1..v { if r[i] > r[bi] { bi = i; } }
                let want = SdCtrl { draft: a.tok[0], mtp_len: 40 + rows, ..c0 };
                let good = d == want && a.tok[0] == bi as u32 && get_ctrl(&ar, mc) == m0;
                if rows == 1 || !good {
                    println!("  V {:6} {:24}: draft {} argmax_b {} cpu {} | mtp_len +{} | other fields untouched {} -> {}",
                             v, name, d.draft, a.tok[0], bi, d.mtp_len - 40, d == want, if good { "ok" } else { "FAIL" });
                }
                ok &= good;
            }
        }
        // mtp_advance
        for rows in [1u32, 2, 8] {
            let c0 = SdCtrl { pos: 3, mtp_len: 40, draft: 5, ..Default::default() };
            let m0 = SdCtrl { mtp_rows: rows, pos: 40, ..Default::default() };
            put_ctrl(&ar, cd, &c0); put_ctrl(&ar, mc, &m0);
            run(&g, &ar, |e| { e.pso("mtp_advance").b(0, cd).b(1, mc).threads((1, 1), 1); });
            let good = get_ctrl(&ar, cd) == SdCtrl { mtp_len: 40 + rows, ..c0 } && get_ctrl(&ar, mc) == m0;
            if v == 2048 { println!("  mtp_advance rows {}: {}", rows, if good { "ok" } else { "FAIL" }); }
            ok &= good;
        }
    }
    assert!(ok, "mtp_draft / mtp_advance");
}
