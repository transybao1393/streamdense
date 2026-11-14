//! PF-B microbench + diff check (synthetic data, real 27B shapes from config.json; no weights are read).
//! cargo run --offline --release -p streamdense-model --example pfb_bench [-- runs]
use streamdense_metal::{commit, contents, gpu_ms, wait, Enc, Gpu, SdCtrl, QWEN_SHADERS};
use streamdense_core::arena::Region;
use streamdense_model::qwen_layers::{encode_attn_prefill, encode_dn_prefill, DnDims, DnState, KvRef, PassIo};
use streamdense_model::{AttnW, DnW, LayerKind};
use streamdense_model::QwenConfig;

struct Arena { off: usize }
impl Arena { fn take(&mut self, n: usize) -> usize { let o = self.off; self.off = (o + n + 255) & !255; o } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }
}
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16 }
fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
fn key16(u: u16) -> i32 { if u >= 0x8000 { 0x8000 - u as i32 } else { u as i32 } }
fn ulp(a: u16, b: u16) -> u32 { (key16(a) - key16(b)).unsigned_abs() }

fn fill_bf(base: *mut u8, off: usize, n: usize, r: &mut Rng, sc: f32) {
    let p = unsafe { std::slice::from_raw_parts_mut(base.add(off) as *mut u16, n) };
    for x in p { *x = to_bf(r.f() * sc); }
}
fn fill_f32(base: *mut u8, off: usize, n: usize, r: &mut Rng, sc: f32) {
    let p = unsafe { std::slice::from_raw_parts_mut(base.add(off) as *mut f32, n) };
    for x in p { *x = r.f() * sc; }
}
fn rd<T: Copy>(base: *mut u8, off: usize, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(base.add(off) as *const T, n).to_vec() }
}
fn wr<T: Copy>(base: *mut u8, off: usize, v: &[T]) {
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), base.add(off) as *mut T, v.len()) }
}
fn cmp_bf(a: &[u16], b: &[u16]) -> (u32, usize) {
    let mut m = 0; let mut n = 0;
    for (x, y) in a.iter().zip(b) { let d = ulp(*x, *y); if d > 0 { n += 1; } m = m.max(d); }
    (m, n)
}
fn cmp_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut n = 0; let mut m = 0f32;
    for (x, y) in a.iter().zip(b) { if x.to_bits() != y.to_bits() { n += 1; m = m.max((x - y).abs()); } }
    (n, m)
}

fn main() {
    let runs: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(5);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");
    let cfg = QwenConfig::from_model_dir(dir).unwrap();
    cfg.validate().unwrap();
    let mut gpu = Gpu::from_shader_files(QWEN_SHADERS).unwrap();
    for k in ["dn_conv", "dn_prep", "dn_recur_ck0", "attn_rope_invf", "attn_prep_b16", "attn_sdpa_vec_b16",
              "dn_conv_pf", "attn_flash_b16", "rmsnorm_gated"] {
        gpu.pipeline(k).unwrap();
    }
    let dd = DnDims::from_cfg(&cfg);
    let tmax = 128usize;
    let max_ctx = 4096usize;
    let (conv, prow, vh) = (cfg.conv_dim(), cfg.dn_in_rows(), cfg.dn_vh);
    let (nh, nkv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let aprow = cfg.attn_in_rows();
    let mut a = Arena { off: 0 };
    let p = a.take(tmax * prow.max(aprow) * 2);
    let convw = a.take(conv * 4 * 2);
    let alog = a.take(vh * 2);
    let dtb = a.take(vh * 2);
    let cst = a.take(dd.layers as usize * 3 * conv * 2);
    let cst0 = a.take(dd.layers as usize * 3 * conv * 2);
    let cv = a.take(tmax * conv.max(cfg.q_dim()) * 2);
    let g = a.take(tmax * vh * 4);
    let beta = a.take(tmax * vh * 2);
    let s_n = vh * 128 * 128;
    let s_all = a.take(3 * s_n * 4);
    let s = s_all + 2 * s_n * 4; // DeltaNet layer 2
    let s0 = a.take(s_n * 4);
    let y = a.take(tmax * vh * 128 * 2);
    let ao = a.take(tmax * cfg.q_dim() * 2);
    let kc = a.take(nkv * max_ctx * hd * 2);
    let vc = a.take(nkv * max_ctx * hd * 2);
    let qnw = a.take(hd * 2);
    let knw = a.take(hd * 2);
    let invf = a.take(64 * 4);
    let nctrl = 1 + tmax;
    let ctrl = a.take(nctrl * 72);
    let arena = gpu.buffer(a.off).unwrap();
    let base = contents(&arena);
    let mut r = Rng(0x9e3779b97f4a7c15);
    fill_bf(base, p, tmax * prow.max(aprow), &mut r, 1.0);
    fill_bf(base, convw, conv * 4, &mut r, 0.5);
    fill_bf(base, alog, vh, &mut r, 1.0);
    fill_bf(base, dtb, vh, &mut r, 1.0);
    fill_bf(base, cst0, dd.layers as usize * 3 * conv, &mut r, 1.0);
    fill_f32(base, s0, s_n, &mut r, 0.1);
    fill_bf(base, kc, nkv * max_ctx * hd, &mut r, 1.0);
    fill_bf(base, vc, nkv * max_ctx * hd, &mut r, 1.0);
    // PFB_SHARP=x scales the q/k norm weights (sharper softmax for the attention diff)
    let wsc: f32 = std::env::var("PFB_SHARP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    fill_bf(base, qnw, hd, &mut r, wsc);
    fill_bf(base, knw, hd, &mut r, wsc);
    let set_ctrl = |pos: u32| {
        for i in 0..nctrl { let c = SdCtrl { pos: pos + i as u32, ..Default::default() }; wr(base, ctrl + i * 72, &[c]); }
    };
    let reset_state = || {
        unsafe {
            std::ptr::copy_nonoverlapping(base.add(cst0), base.add(cst), dd.layers as usize * 3 * conv * 2);
            std::ptr::copy_nonoverlapping(base.add(s0), base.add(s), s_n * 4);
        }
    };
    let layer = 2u32;
    let run = |f: &dyn Fn(&Enc)| -> f64 {
        let cb = gpu.command_buffer();
        let e = Enc::new(&gpu, &cb, &arena);
        f(&e);
        e.end();
        commit(&cb);
        wait(&cb).unwrap();
        gpu_ms(&cb) * 1e3
    };
    // ---- DeltaNet: existing kernels on rows [r0, r0 + k)
    let dn_old = |e: &Enc, r0: usize, k: usize| {
        e.pso("dn_conv").b(0, p + r0 * prow * 2).b(1, convw).b(2, cst).b(3, cv + r0 * conv * 2).b(4, ctrl)
            .u(5, layer).u(6, k as u32).u(7, u32::MAX).bytes(8, &dd).threads((conv, 1), 256);
        e.pso("dn_prep").b(0, cv + r0 * conv * 2).b(1, p + r0 * prow * 2).b(2, alog).b(3, dtb).b(4, g + r0 * vh * 4)
            .b(5, beta + r0 * vh * 2).u(6, k as u32).bytes(7, &dd).groups((2 * cfg.dn_kh + 1, k), 128);
        e.pso("dn_recur_ck0").b(0, cv + r0 * conv * 2).b(1, g + r0 * vh * 4).b(2, beta + r0 * vh * 2).b(3, s)
            .b(4, y + r0 * vh * 128 * 2).b(5, ctrl).u(6, k as u32).bytes(7, &dd).groups((4 * vh, 1), 128);
        e.pso("rmsnorm_gated").b(0, y + r0 * vh * 128 * 2).b(1, p + (r0 * prow + conv) * 2).u(2, prow as u32).b(3, qnw)
            .b(4, ao + r0 * vh * 128 * 2).f(5, cfg.eps).groups((vh, k), 128);
    };
    // model layer whose DeltaNet index is `layer`
    let lmodel = (0..cfg.layers).filter(|&i| cfg.kinds[i] == LayerKind::DeltaNet).nth(layer as usize).unwrap();
    let mut dw = DnW::default();
    dw.conv_w = Region { off: convw, len: 0 }; dw.a_log = Region { off: alog, len: 0 };
    dw.dt_bias = Region { off: dtb, len: 0 }; dw.norm_w = Region { off: qnw, len: 0 };
    let dst = DnState { s: s_all, conv: cst, copies: 1 };
    let mk_io = |k: usize, pos: usize| PassIo { k, p, cv, g, beta, y, yn: ao, ao, ctrl, invf, pos, ..Default::default() };
    let dn_new = |e: &Enc, k: usize| { encode_dn_prefill(e, &cfg, &dw, &dst, &mk_io(k, 0), lmodel).unwrap(); };
    let dn_outputs = || -> (Vec<u16>, Vec<f32>, Vec<u16>, Vec<u16>, Vec<f32>, Vec<u16>) {
        (rd::<u16>(base, cv, tmax * conv), rd::<f32>(base, g, tmax * vh), rd::<u16>(base, beta, tmax * vh),
         rd::<u16>(base, y, tmax * vh * 128), rd::<f32>(base, s, s_n), rd::<u16>(base, cst, dd.layers as usize * 3 * conv))
    };
    // diff: row-by-row old (T=1 dispatches) vs new (one dispatch of T)
    let bench_only = std::env::var("PFB_BENCH_ONLY").is_ok();
    set_ctrl(0);
    for &t in if bench_only { &[][..] } else { &[1usize, 3, 8, 37, 128][..] } {
        reset_state();
        run(&|e| for i in 0..t { dn_old(e, i, 1) });
        let o = dn_outputs();
        reset_state();
        run(&|e| dn_new(e, t));
        let n = dn_outputs();
        let c1 = cmp_bf(&o.0[..t * conv], &n.0[..t * conv]);
        let c2 = cmp_f32(&o.1[..t * vh], &n.1[..t * vh]);
        let c3 = cmp_bf(&o.2[..t * vh], &n.2[..t * vh]);
        let c4 = cmp_bf(&o.3[..t * vh * 128], &n.3[..t * vh * 128]);
        let c5 = cmp_f32(&o.4, &n.4);
        let c6 = cmp_bf(&o.5, &n.5);
        let fin = n.3[..t * vh * 128].iter().all(|v| bf(*v).is_finite());
        println!("dn diff T={:3}: cv {:?} g {:?} beta {:?} y {:?} S {:?} conv_state {:?} (max ulp, n diff) finite {}",
                 t, c1, c2, c3, c4, c5, c6, fin);
    }
    // ---- attention: prep (existing, T rows) + per-row sdpa mirror vs flash
    run(&|e| { e.pso("attn_rope_invf").b(0, invf).f(1, (cfg.rope_theta as f32).log2()).threads((cfg.rope_dims / 2, 1), cfg.rope_dims / 2); });
    let grp = nh / nkv;
    let prep = |e: &Enc, r0: usize, k: usize, c: usize| {
        e.pso("attn_prep_b16").b(0, p + r0 * aprow * 2).b(1, qnw).b(2, knw).b(3, cv + r0 * nh * hd * 2).b(4, kc).b(5, vc)
            .b(6, kc).b(7, vc).b(8, invf).b(9, c).u(10, max_ctx as u32).f(11, cfg.eps).u(12, nkv as u32)
            .u(13, cfg.rope_dims as u32).groups((nh + nkv, k), hd);
    };
    let sdpa_old = |e: &Enc, r0: usize, k: usize, c: usize| {
        e.pso("attn_sdpa_vec_b16").b(0, cv + r0 * nh * hd * 2).b(1, kc).b(2, vc).b(3, p + r0 * aprow * 2)
            .b(4, ao + r0 * nh * hd * 2).b(5, c).u(6, max_ctx as u32).u(7, grp as u32).u(8, aprow as u32)
            .u(9, 8192).u(10, 1024).groups((nh, k), 1024);
    };
    let mut aw = AttnW::default();
    aw.q_norm = Region { off: qnw, len: 0 }; aw.k_norm = Region { off: knw, len: 0 };
    let kvr = KvRef { k: kc, v: vc, ksb: kc, vsb: vc, q8: false, max_ctx };
    let flash = |e: &Enc, k: usize, pos: usize| { encode_attn_prefill(e, &cfg, &aw, &kvr, &mk_io(k, pos)).unwrap(); };
    // the flash kernel alone (whatever T), after attn_prep_b16
    let flash_raw = |e: &Enc, k: usize| {
        e.pso("attn_flash_b16").b(0, cv).b(1, kc).b(2, vc).b(3, p).b(4, ao).b(5, ctrl).u(6, max_ctx as u32)
            .u(7, grp as u32).u(8, aprow as u32).u(9, k as u32).groups((k.div_ceil(8), nkv), 32 * grp);
    };
    for &pos in if bench_only { &[][..] } else { &[0usize, 5, 1000, 2900][..] } {
        for &t in &[1usize, 8, 37, 128] {
            set_ctrl(pos as u32);
            run(&|e| prep(e, 0, t, ctrl));
            run(&|e| sdpa_old(e, 0, t, ctrl));
            let o = rd::<u16>(base, ao, t * nh * hd);
            wr(base, ao, &vec![0u16; t * nh * hd]);
            for raw in [false, true] {
                wr(base, ao, &vec![0u16; t * nh * hd]);
                if raw { run(&|e| flash_raw(e, t)); } else { run(&|e| flash(e, t, pos)); }
                let n = rd::<u16>(base, ao, t * nh * hd);
                let mut md = 0f32; let mut mr = 0f32; let mut big = 0u32; let mut fin = true;
                for (x, y) in o.iter().zip(&n) {
                    md = md.max((bf(*x) - bf(*y)).abs()); mr = mr.max(bf(*x).abs()); fin &= bf(*y).is_finite();
                    if bf(*x).abs() >= 0.0625 { big = big.max(ulp(*x, *y)); }
                }
                println!("attn diff pos={:4} T={:3} {}: max abs {:.3e} (max |ref| {:.3}; ulp where |ref|>=1/16: {})  finite {}",
                         pos, t, if raw { "flash kernel " } else { "encode_attn_prefill" }, md, mr, big, fin);
            }
        }
    }
    // ---- bench
    println!("\nbench (GPU us per layer, min of {} runs; 'old K=8' = existing kernels in 8-row chunks)", runs);
    println!("| T | DN old K=8 | DN old 1xT | DN new | attn old K=8 | attn old 1xT | attn new (encode_attn_prefill) |");
    println!("|---|---|---|---|---|---|---|");
    let pos = 1000usize;
    const REP: usize = 10;
    set_ctrl(pos as u32);
    let t_w = std::time::Instant::now();
    while t_w.elapsed().as_secs_f64() < 2.0 { run(&|e| for _ in 0..REP { flash(e, 128, pos); dn_old(e, 0, 128) }); }
    {
        let rep = |f: &dyn Fn(&Enc)| -> f64 { let mut b = f64::MAX; for _ in 0..runs { b = b.min(run(&|e| for _ in 0..REP { f(e) }) / REP as f64); } b };
        let t = 128usize;
        let dsp = |e: &Enc, name: &str| match name {
            "dn_conv" => { e.pso("dn_conv").b(0, p).b(1, convw).b(2, cst).b(3, cv).b(4, ctrl).u(5, layer).u(6, t as u32).u(7, u32::MAX).bytes(8, &dd).threads((conv, 1), 256); }
            "dn_conv_pf" => { e.pso("dn_conv_pf").b(0, p).b(1, convw).b(2, cst).b(3, cv).b(4, ctrl).u(5, layer).u(6, t as u32).bytes(7, &dd).groups((conv.div_ceil(32), 1), 32 * t.div_ceil(8)); }
            "dn_prep" => { e.pso("dn_prep").b(0, cv).b(1, p).b(2, alog).b(3, dtb).b(4, g).b(5, beta).u(6, t as u32).bytes(7, &dd).groups((2 * cfg.dn_kh + 1, t), 128); }
            "dn_recur_ck0" => { e.pso(name).b(0, cv).b(1, g).b(2, beta).b(3, s).b(4, y).b(5, ctrl).u(6, t as u32).bytes(7, &dd).groups((4 * vh, 1), 128); }
            "rmsnorm_gated" => { e.pso("rmsnorm_gated").b(0, y).b(1, p + conv * 2).u(2, prow as u32).b(3, qnw).b(4, ao).f(5, cfg.eps).groups((vh, t), 128); }
            "attn_prep_b16" => { prep(e, 0, t, ctrl); }
            _ => unreachable!(),
        };
        let mut line = String::from("T=128 per kernel (us):");
        for n in ["dn_conv", "dn_conv_pf", "dn_prep", "dn_recur_ck0", "rmsnorm_gated", "attn_prep_b16"] {
            line += &format!(" {} {:.0};", n, rep(&|e| dsp(e, n)));
        }
        println!("{}", line);
    }
    for &t in &[8usize, 16, 32, 64, 128] {
        let mut best = [f64::MAX; 6];
        let rep = |f: &dyn Fn(&Enc)| -> f64 { run(&|e| for _ in 0..REP { f(e) }) / REP as f64 };
        for _ in 0..runs {
            best[0] = best[0].min(rep(&|e| { let mut i = 0; while i < t { dn_old(e, i, 8.min(t - i)); i += 8; } }));
            best[1] = best[1].min(rep(&|e| dn_old(e, 0, t)));
            best[2] = best[2].min(rep(&|e| dn_new(e, t)));
            // per-chunk ctrl blocks: chunk at row i uses the ctrl at index i (pos + i)
            best[3] = best[3].min(rep(&|e| { let mut i = 0; while i < t { let k = 8.min(t - i); prep(e, i, k, ctrl + i * 72); sdpa_old(e, i, k, ctrl + i * 72); i += 8; } }));
            best[4] = best[4].min(rep(&|e| { prep(e, 0, t, ctrl); sdpa_old(e, 0, t, ctrl); }));
            best[5] = best[5].min(rep(&|e| flash(e, t, pos)));
        }
        println!("| {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} |", t, best[0], best[1], best[2], best[3], best[4], best[5]);
    }
}
