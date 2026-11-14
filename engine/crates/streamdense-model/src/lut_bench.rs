//! `streamdense bench lut-k`: research bench for lookup-table 4-bit matmul (shaders/gemv_lut_q4.metal).
//! `--prims`: primitive throughput probes (fp32 / fp16 / half2 / int / mixed issue, threadgroup lookups).
//! No model weights are loaded.
use streamdense_metal::{commit, contents, gpu_ms, wait, Enc, Gpu};

fn time(g: &Gpu, buf: &streamdense_metal::Buffer, reps: usize, f: &dyn Fn(&Enc)) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let cb = g.command_buffer();
        let e = Enc::new(g, &cb, buf);
        f(&e);
        e.end();
        commit(&cb);
        wait(&cb)?;
        best = best.min(gpu_ms(&cb));
    }
    Ok(best)
}

fn prims(reps: usize, arg_n: u32) -> Result<(), String> {
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_lut_q4"])?;
    let names = ["lt_p_f32", "lt_p_f16", "lt_p_h2", "lt_p_mix", "lt_p_fint", "lt_p_int", "lt_p_tg0", "lt_p_tg1", "lt_p_tg2", "lt_p_tg3"];
    for n in names { g.pipeline(n)?; }
    let threads = 1usize << 18;
    let buf = g.buffer(threads * 4 + 4096)?;
    let _ = contents(&buf);
    let n_it: u32 = arg_n;
    println!("GPU {}: {} threads x {} iterations, min of {} reps", g.name(), threads, n_it, reps);
    println!("| probe | ops / iteration | ms | G ops/s | ops/s vs fp32 fma |");
    println!("|---|---|---|---|---|");
    let mut f32rate = 0.0;
    for (n, ops, desc) in [("lt_p_f32", 8, "8 fp32 fma"), ("lt_p_f16", 8, "8 fp16 fma"), ("lt_p_h2", 16, "8 half2 fma (16 lanes-ops)"),
                           ("lt_p_mix", 16, "8 fp32 + 8 fp16 fma"), ("lt_p_fint", 24, "8 fp32 fma + 8 x (iadd, xor)"),
                           ("lt_p_int", 16, "8 x (iadd, xor)"),
                           ("lt_p_tg0", 8, "8 tg lookups, per-lane table (conflict-free) + fadd"),
                           ("lt_p_tg1", 8, "8 tg lookups, shared 256-entry table, random idx + fadd"),
                           ("lt_p_tg2", 8, "8 tg lookups, broadcast idx + fadd"),
                           ("lt_p_tg3", 8, "same ALU work, no load (float(idx))")] {
        let tg = if n.starts_with("lt_p_tg") { 1024 } else { 256 };
        let ms = time(&g, &buf, reps, &|e| { e.pso(n).b(0, 0).u(1, n_it).f(2, 0.999).groups((threads / tg, 1), tg); })?;
        let rate = threads as f64 * n_it as f64 * ops as f64 / ms / 1e6; // G ops/s
        if n == "lt_p_f32" { f32rate = rate; }
        println!("| {} ({}) | {} | {:.2} | {:.0} | {:.2} |", n, desc, ops, ms, rate, rate / f32rate);
    }
    Ok(())
}

// ---- `--gemv` (default): lq_gemv (bit-plane LUT) at the 27B shapes, K = 1..8, vs q4b_gemv K = 1 (D = 2, m3).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    fn n(&mut self) -> f32 { (0..4).map(|_| self.f()).sum::<f32>() * 1.732 - 3.464 }
}
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }
fn q4_bytes(r: usize, c: usize) -> usize { r * c / 2 + 2 * (r * c / 64 * 2) }

struct Mat { w: usize, s: usize, b: usize }
struct A { ptr: *mut u8, top: usize, len: usize }
impl A {
    fn alloc(&mut self, n: usize) -> Result<usize, String> {
        let o = self.top;
        self.top = (self.top + n + 255) & !255;
        if self.top > self.len { return Err("lut bench arena too small".into()); }
        Ok(o)
    }
    #[allow(clippy::mut_from_ref)] // bench arena: disjoint offsets into one shared GPU buffer
    fn sl<T>(&self, off: usize, n: usize) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut T, n) } }
}
fn put_mat(a: &mut A, rng: &mut Rng, r: usize, c: usize) -> Result<Mat, String> {
    let (w, s, b) = (a.alloc(r * c / 2)?, a.alloc(r * c / 32)?, a.alloc(r * c / 32)?);
    for v in a.sl::<u32>(w, r * c / 8) { *v = rng.next() as u32; }
    let (ss, bb) = (a.sl::<u16>(s, r * c / 64), a.sl::<u16>(b, r * c / 64));
    for i in 0..r * c / 64 {
        let sc = 0.002 + 0.018 * rng.f();
        ss[i] = to_bf(sc);
        bb[i] = to_bf(-8.0 * sc + 0.01 * rng.n());
    }
    Ok(Mat { w, s, b })
}

fn gemv(reps: usize, model: &str, ks: &[usize], head: bool) -> Result<(), String> {
    use crate::qwen_config::QwenConfig;
    let mut g = Gpu::from_shader_files(&["common_bf16", "gemv_q4_bf16", "gemv_lut_q4"])?;
    g.pipeline("q4b_gemv_k1_m3")?;
    for k in 1..=8 { g.pipeline(&format!("lq_gemv_k{}", k))?; }
    let cfg = QwenConfig::from_model_dir(model)?;
    let (h, nd, na, nl) = (cfg.hidden, cfg.n_deltanet(), cfg.n_attention(), cfg.layers);
    // (name, rows, cols, dispatches per pass); gate+up = 2 GEMVs of inter x h (the swiglu pair, timed as two gemvs)
    let mut shapes: Vec<(String, usize, usize, usize)> = Vec::new();
    for (n, r, c, cnt) in [("dn_in", cfg.dn_in_rows(), h, nd), ("dn_out", h, cfg.dn_v_dim(), nd), ("attn_in", cfg.attn_in_rows(), h, na),
                           ("o_proj", h, cfg.q_dim(), na), ("gate,up", cfg.inter, h, 2 * nl), ("down", h, cfg.inter, nl)] {
        if cnt == 0 { continue; }
        match shapes.iter_mut().find(|s| s.1 == r && s.2 == c) { Some(s) => { s.0 = format!("{}|{}", s.0, n); s.3 += cnt; }
                                                                  None => shapes.push((n.into(), r, c, cnt)) }
    }
    if head { shapes.push(("lm_head".into(), cfg.vocab, h, 1)); }
    let rot = 256e6;
    let copies = |r: usize, c: usize| ((rot / q4_bytes(r, c) as f64).ceil() as usize).max(1);
    let wmax = shapes.iter().map(|s| copies(s.1, s.2) * (q4_bytes(s.1, s.2) + 1024)).max().unwrap();
    let (maxr, maxc) = (shapes.iter().map(|s| s.1).max().unwrap(), shapes.iter().map(|s| s.2).max().unwrap());
    let len = wmax + 3 * (8 * maxc * 2 + 256) + 4 * (8 * maxr * 4 + 256) + (1 << 20);
    let buf = g.buffer(len)?;
    eprintln!("lut bench arena {:.3} GB on {}", len as f64 / 1e9, g.name());
    let mut a = A { ptr: contents(&buf), top: 0, len };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let (x, x2) = (a.alloc(8 * maxc * 2)?, a.alloc(8 * maxc * 2)?);
    for v in a.sl::<u16>(x, 8 * maxc) { *v = to_bf(rng.n()); }
    let (oref, olq, otst) = (a.alloc(8 * maxr * 4)?, a.alloc(8 * maxr * 4)?, a.alloc(8 * maxr * 4)?);
    let wbase = a.top;
    let enc_lq = |e: &Enc, k: usize, m: &Mat, r: usize, c: usize, xo: usize, o: usize| {
        e.pso(&format!("lq_gemv_k{}", k)).b(0, m.w).b(1, m.s).b(2, m.b).b(3, xo).b(4, o).u(5, r as u32).u(6, c as u32)
            .groups((r.div_ceil(256), 1), 256);
    };
    let enc_old = |e: &Enc, m: &Mat, r: usize, c: usize, xo: usize, o: usize| {
        e.pso("q4b_gemv_k1_m3").b(0, m.w).b(1, m.s).b(2, m.b).b(3, xo).b(4, o).u(5, r as u32).u(6, c as u32).b(7, o).b(8, o)
            .groups((r.div_ceil(4), 1), 64);
    };
    let run1 = |f: &dyn Fn(&Enc)| -> Result<(), String> {
        let cb = g.command_buffer(); let e = Enc::new(&g, &cb, &buf); f(&e); e.end(); commit(&cb); wait(&cb)
    };
    let mut per: Vec<(usize, usize, f64)> = Vec::new(); // (shape, k, ms); k = 0: q4b K=1
    let (mut inv_ok, mut cl) = (true, Vec::new());
    println!("| shape | R x C | per pass | K | lq µs | q4b K=1 µs | lq / q4b K=1 |");
    println!("|---|---|---|---|---|---|---|");
    for (si, (name, r, c, cnt)) in shapes.iter().enumerate() {
        let (r, c) = (*r, *c);
        a.top = wbase;
        let nc = copies(r, c);
        let mats: Vec<Mat> = (0..nc).map(|_| put_mat(&mut a, &mut rng, r, c)).collect::<Result<_, _>>()?;
        let m0 = &mats[0];
        // references: q4b K=1 and lq K=1 per x row
        run1(&|e| for t in 0..8 { enc_old(e, m0, r, c, x + t * c * 2, oref + t * r * 4); enc_lq(e, 1, m0, r, c, x + t * c * 2, olq + t * r * 4); })?;
        // closeness lq vs q4b (f32 and bf16-rounded)
        {
            let (p, q) = (a.sl::<f32>(oref, 8 * r), a.sl::<f32>(olq, 8 * r));
            let (mut eq32, mut eqbf, mut mx, mut ss) = (0usize, 0usize, 0f64, 0f64);
            for (u, v) in p.iter().zip(q.iter()) {
                if u.to_bits() == v.to_bits() { eq32 += 1; }
                if to_bf(*u) == to_bf(*v) { eqbf += 1; }
                mx = mx.max((*u as f64 - *v as f64).abs());
                ss += (*u as f64) * (*u as f64);
            }
            let n = (8 * r) as f64;
            cl.push(format!("| {} | {:.3}% | {:.3}% | {:.2e} |", name, 100.0 * eq32 as f64 / n, 100.0 * eqbf as f64 / n, mx / (ss / n).sqrt()));
        }
        // K-invariance: every K, rows t < K == lq K=1 row t; K = 8 with odd / even rows replaced
        for k in 1..=8 {
            a.sl::<u8>(otst, 8 * r * 4).fill(0xa5);
            run1(&|e| enc_lq(e, k, m0, r, c, x, otst))?;
            if a.sl::<u32>(olq, k * r) != a.sl::<u32>(otst, k * r) { inv_ok = false; eprintln!("K-invariance FAIL {} K={}", name, k); }
        }
        for keep in [0usize, 1] {
            { let (xs, xn) = (a.sl::<u16>(x, 8 * maxc), a.sl::<u16>(x2, 8 * maxc)); xn.copy_from_slice(xs);
              for t in (0..8).filter(|t| t % 2 != keep) { for v in &mut xn[t * c..(t + 1) * c] { *v = to_bf(300.0 * rng.n()); } } }
            run1(&|e| enc_lq(e, 8, m0, r, c, x2, otst))?;
            for t in (0..8).filter(|t| t % 2 == keep) {
                if a.sl::<u32>(olq + t * r * 4, r) != a.sl::<u32>(otst + t * r * 4, r) { inv_ok = false; eprintln!("neighbour FAIL {} row {}", name, t); }
            }
        }
        let n = if nc == 1 { 3 } else { 2 * nc };
        let tm = |f: &dyn Fn(&Enc, usize)| -> Result<f64, String> {
            time(&g, &buf, reps, &|e| for i in 0..n { f(e, i) }).map(|ms| ms / n as f64)
        };
        let t_old = tm(&|e, i| enc_old(e, &mats[i % nc], r, c, x, otst))?;
        per.push((si, 0, t_old));
        for &k in ks {
            let ms = tm(&|e, i| enc_lq(e, k, &mats[i % nc], r, c, x, otst))?;
            per.push((si, k, ms));
            println!("| {} | {} x {} | {} | {} | {:.0} | {:.0} | {:.2} |", name, r, c, cnt, k, ms * 1e3, t_old * 1e3, ms / t_old);
        }
    }
    let pass = |k: usize| -> f64 { shapes.iter().enumerate().map(|(si, s)| s.3 as f64 * per.iter().find(|p| p.0 == si && p.1 == k).unwrap().2).sum() };
    let p1 = pass(0);
    println!("\nFull-pass estimate{} (gate+up as 2 gemvs): q4b K=1 (this run) {:.2} ms", if head { " incl. lm_head" } else { "" }, p1);
    println!("| K | lq pass ms | / q4b K=1 |");
    println!("|---|---|---|");
    for &k in ks { let p = pass(k); println!("| {} | {:.2} | {:.2} |", k, p, p / p1); }
    println!("\nlq K=1 vs q4b_gemv K=1 (8 x rows per shape, f32 outputs):");
    println!("| shape | f32 bit-equal | bf16-rounded equal | max |d| / rms |");
    println!("|---|---|---|---|");
    for l in &cl { println!("{}", l); }
    println!("\nlq K-invariance (K=1..8 rows and K=8 with neighbours replaced == per-row K=1): {}", if inv_ok { "ALL OK" } else { "FAILED" });
    Ok(())
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let reps: usize = arg("--reps").map_or(Ok(5), |v| v.parse().map_err(|_| "--reps".to_string()))?;
    if args.iter().any(|a| a == "--prims") { return prims(reps, arg("--iters").map_or(8192, |v| v.parse().unwrap_or(8192))); }
    let model = arg("--model").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4").into());
    let ks: Vec<usize> = arg("--ks").unwrap_or_else(|| "1,2,3,4,5,8".into()).split(',').map(|s| s.trim().parse().map_err(|_| "--ks".to_string()))
        .collect::<Result<_, _>>()?;
    if ks.iter().any(|k| *k == 0 || *k > 8) { return Err("--ks: 1..=8".into()); }
    gemv(reps, &model, &ks, !args.iter().any(|a| a == "--no-head"))
}
