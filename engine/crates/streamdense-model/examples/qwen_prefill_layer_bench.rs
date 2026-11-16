//! U25 synthetic layer bench (no model load): one DeltaNet layer and one attention layer at the real 27B shapes
//! (QwenConfig from config.json) with random weights. Times the fast prefill layer (`encode_layer_prefill`, GEMM +
//! prefill mixers) over T rows against the exact path (`encode_layer` in K-row passes), at kv position `--pos`, and
//! compares the layer outputs h (fast vs exact, same inputs and state). Projects whole-model prefill tok/s from the
//! layer times (n_deltanet * dn + n_attention * attn; head and embedding not included).
//! Run: cargo run --release --offline -p streamdense-model --example qwen_prefill_layer_bench -- [--t 128] [--pos 1000]
//!      [--reps 3] [--k 8] [--kv bf16|q8]   (q8: 8-bit KV, random codes and fp16 scale / min)
use streamdense_core::arena::Region;
use streamdense_metal::{commit, contents, gpu_ms, wait, Enc, Gpu, SdCtrl, QWEN_SHADERS};
use streamdense_model::qwen_layers::{encode_layer, encode_layer_prefill, encode_rope_invf, gemm_pipelines, prefill_pipelines,
                                     qwen_pipelines, scratch_sizes, DnState, KvRef, PassIo};
use streamdense_model::{AttnW, DnW, FfnW, LayerKind, LayerW, Mixer, Q4b, QwenConfig};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    fn n(&mut self) -> f32 { (0..4).map(|_| self.f()).sum::<f32>() * 1.732 - 3.464 }
}
fn to_bf(f: f32) -> u16 { let u = f.to_bits(); ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }
fn bf(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }

struct A { ptr: *mut u8, top: usize, len: usize }
impl A {
    fn alloc(&mut self, n: usize) -> Region {
        let o = self.top;
        self.top = (self.top + n + 255) & !255;
        assert!(self.top <= self.len, "bench arena too small");
        Region { off: o, len: n }
    }
    #[allow(clippy::mut_from_ref)] // bench arena: disjoint regions of one shared GPU buffer
    fn sl<T>(&self, r: Region) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.add(r.off) as *mut T, r.len / std::mem::size_of::<T>()) } }
    fn bf_fill(&mut self, n: usize, rng: &mut Rng, f: &dyn Fn(&mut Rng) -> f32) -> Region {
        let r = self.alloc(n * 2);
        for v in self.sl::<u16>(r) { *v = to_bf(f(rng)); }
        r
    }
    fn q4b(&mut self, rng: &mut Rng, rows: usize, cols: usize) -> Q4b {
        let (w, s, b) = (self.alloc(rows * cols / 2), self.alloc(rows * cols / 32), self.alloc(rows * cols / 32));
        for v in self.sl::<u32>(w) { *v = rng.next() as u32; }
        let (ss, bb) = (self.sl::<u16>(s), self.sl::<u16>(b));
        for i in 0..ss.len() {
            let sc = (0.002 + 0.018 * rng.f()) * 0.1;
            ss[i] = to_bf(sc);
            bb[i] = to_bf(-7.5 * sc + 0.001 * rng.n());
        }
        Q4b { rows, cols, w, s, b }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    let model = arg("--model").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4").into());
    let t: usize = arg("--t").map_or(128, |v| v.parse().unwrap());
    let pos: usize = arg("--pos").map_or(1000, |v| v.parse().unwrap());
    let reps: usize = arg("--reps").map_or(3, |v| v.parse().unwrap());
    let kx: usize = arg("--k").map_or(8, |v| v.parse().unwrap());
    let q8 = arg("--kv").is_some_and(|v| v == "q8");
    let cfg = QwenConfig::from_model_dir(&model).expect("config");
    cfg.validate().expect("validate");
    let max_ctx = (pos + t).next_power_of_two().max(1024);
    let ldn = cfg.kinds.iter().position(|k| *k == LayerKind::DeltaNet).unwrap();
    let lat = cfg.kinds.iter().position(|k| *k == LayerKind::Attention).unwrap();

    let mut g = Gpu::from_shader_files(QWEN_SHADERS).expect("compile");
    for n in qwen_pipelines() { g.pipeline(n).unwrap(); }
    for n in prefill_pipelines() { g.pipeline(n).unwrap(); }
    for n in gemm_pipelines() { g.pipeline(&n).unwrap(); }
    g.pipeline("advance1").unwrap();
    let len = 1_100_000_000usize;
    let buf = g.buffer(len).expect("buffer");
    let mut a = A { ptr: contents(&buf), top: 0, len };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let h = cfg.hidden;
    let one = |r: &mut Rng| 1.0 + 0.05 * r.n();
    // ---- weights (random)
    let ffn = |a: &mut A, rng: &mut Rng| FfnW { gate: a.q4b(rng, cfg.inter, h), up: a.q4b(rng, cfg.inter, h), down: a.q4b(rng, h, cfg.inter), cd: None };
    let dnw = LayerW {
        in_norm: a.bf_fill(h, &mut rng, &one), post_norm: a.bf_fill(h, &mut rng, &one),
        mixer: Mixer::Dn(DnW {
            in_proj: a.q4b(&mut rng, cfg.dn_in_rows(), h), conv_w: a.bf_fill(cfg.conv_dim() * cfg.conv_k, &mut rng, &|r| 0.3 * r.n()),
            a_log: a.bf_fill(cfg.dn_vh, &mut rng, &|r| 0.5 * r.n()), dt_bias: a.bf_fill(cfg.dn_vh, &mut rng, &|r| 0.5 * r.n()),
            norm_w: a.bf_fill(cfg.dn_dv, &mut rng, &one), out_proj: a.q4b(&mut rng, h, cfg.dn_v_dim()),
        }),
        ffn: ffn(&mut a, &mut rng),
    };
    let atw = LayerW {
        in_norm: a.bf_fill(h, &mut rng, &one), post_norm: a.bf_fill(h, &mut rng, &one),
        mixer: Mixer::Attn(AttnW {
            in_proj: a.q4b(&mut rng, cfg.attn_in_rows(), h), q_norm: a.bf_fill(cfg.head_dim, &mut rng, &one),
            k_norm: a.bf_fill(cfg.head_dim, &mut rng, &one), o_proj: a.q4b(&mut rng, h, cfg.q_dim()),
        }),
        ffn: ffn(&mut a, &mut rng),
    };
    // ---- state, KV, scratch
    let s_r = a.alloc(DnState::s_bytes(&cfg, 1));
    let c_r = a.alloc(DnState::conv_bytes(&cfg, 1));
    let st = DnState { s: s_r.off, conv: c_r.off, copies: 1 };
    let kvn = cfg.kv_heads * max_ctx * cfg.head_dim;
    let dummy = a.alloc(256);
    let kv = if q8 {
        let (kr, vr) = (a.alloc(kvn), a.alloc(kvn));
        for r in [kr, vr] { for v in a.sl::<u32>(r) { *v = rng.next() as u32; } }
        // fp16 (scale, min) per 32-dim group: scale ~ 6 / 255, min ~ -3 (codes span about +-3, as r.n())
        let sb = |a: &mut A, rng: &mut Rng| {
            let r = a.alloc(kvn / 32 * 4);
            for (i, v) in a.sl::<u16>(r).iter_mut().enumerate() {
                let x: f32 = if i % 2 == 0 { (6.0 + 0.5 * rng.n()) / 255.0 } else { -3.0 + 0.1 * rng.n() };
                *v = streamdense_core::f16::f32_to_f16(x);
            }
            r
        };
        let (ks, vs) = (sb(&mut a, &mut rng), sb(&mut a, &mut rng));
        KvRef { k: kr.off, v: vr.off, ksb: ks.off, vsb: vs.off, q8: true, max_ctx }
    } else {
        let (kr, vr) = (a.bf_fill(kvn, &mut rng, &|r| r.n()), a.bf_fill(kvn, &mut rng, &|r| r.n()));
        KvRef { k: kr.off, v: vr.off, ksb: dummy.off, vsb: dummy.off, q8: false, max_ctx }
    };
    let nch_max = max_ctx.div_ceil(256);
    let mut sr = std::collections::HashMap::new();
    for (n, b) in scratch_sizes(&cfg, t, nch_max) { sr.insert(n, a.alloc(b)); }
    let ctrl = a.alloc(std::mem::size_of::<SdCtrl>());
    let rec = a.alloc((max_ctx + t) * 16);
    let o = |n: &str| sr[n].off;
    let io0 = PassIo { k: t, h: o("h"), x: o("x"), p: o("p"), cv: o("cv"), g: o("g"), beta: o("beta"), y: o("y"), yn: o("yn"),
                       ao: o("ao"), f: o("f"), po: o("po"), pml: o("pml"), hf: o("hf"), ctrl: ctrl.off, nch_max, pos,
                       logits: o("logits"), rec: rec.off, invf: o("invf"), dn_save: None, dn_hold: None, gemv: Default::default(), act: None, rm: None, fsel: None, cfs: 0, vf: None, fxcap: None, vm: None, va1: None, ban: None };
    let hin = a.bf_fill(t * h, &mut rng, &|r| r.n());
    let hbytes = t * h * 2;
    {
        let cb = g.command_buffer();
        let e = Enc::new(&g, &cb, &buf);
        encode_rope_invf(&e, &cfg, &io0);
        e.end(); commit(&cb); wait(&cb).unwrap();
    }
    let reset = |a: &A| {
        a.sl::<u8>(s_r).fill(0);
        a.sl::<u8>(c_r).fill(0);
        let c = unsafe { &mut *(a.ptr.add(ctrl.off) as *mut SdCtrl) };
        *c = unsafe { std::mem::zeroed() };
        c.pos = pos as u32;
        unsafe { std::ptr::copy_nonoverlapping(a.ptr.add(hin.off), a.ptr.add(io0.h), hbytes) };
    };
    let run = |a: &A, lw: &LayerW, l: usize, fast: bool, k: usize| -> f64 {
        reset(a);
        let cb = g.command_buffer();
        let e = Enc::new(&g, &cb, &buf);
        if fast {
            encode_layer_prefill(&e, &cfg, lw, Some(&st), Some(&kv), &io0, l).unwrap();
        } else {
            for c in 0..t / k {
                let io = PassIo { k, pos: pos + c * k, nch_max: (pos + c * k + k).div_ceil(256), h: io0.h + c * k * h * 2, ..io0 };
                encode_layer(&e, &cfg, lw, Some(&st), Some(&kv), &io, false, None, l).unwrap();
                for _ in 0..k { e.pso("advance1").b(0, ctrl.off); e.threads((1, 1), 1); }
            }
        }
        e.end(); commit(&cb); wait(&cb).unwrap();
        gpu_ms(&cb)
    };
    println!("T = {}, pos = {}, exact K = {} ({} passes), KV {}, GPU {}", t, pos, kx, t / kx, if q8 { "q8" } else { "bf16" }, g.name());
    println!("| layer | fast ms | exact ms | speedup | fast ms/token |");
    println!("|---|---|---|---|---|");
    let mut ms = [0.0f64; 2];
    for (i, (name, lw, l)) in [("DeltaNet", &dnw, ldn), ("attention", &atw, lat)].into_iter().enumerate() {
        let tf = (0..reps).map(|_| run(&a, lw, l, true, kx)).fold(f64::MAX, f64::min);
        let hf: Vec<u16> = a.sl::<u16>(Region { off: io0.h, len: hbytes }).to_vec();
        let te = (0..reps).map(|_| run(&a, lw, l, false, kx)).fold(f64::MAX, f64::min);
        let he: Vec<u16> = a.sl::<u16>(Region { off: io0.h, len: hbytes }).to_vec();
        ms[i] = tf;
        println!("| {} | {:.2} | {:.2} | {:.1}x | {:.3} |", name, tf, te, te / tf, tf / t as f64);
        let (mut md, mut nd, mut rms, mut nan) = (0f32, 0usize, 0f64, 0usize);
        for (&p, &q) in hf.iter().zip(&he) {
            let (x, y) = (bf(p), bf(q));
            if !x.is_finite() || !y.is_finite() { nan += 1; continue; }
            md = md.max((x - y).abs());
            nd += (p != q) as usize;
            rms += (y as f64).powi(2);
        }
        println!("  h fast vs exact: max abs {:.3e}, rms(h) {:.3e}, differ {:.2}%, non-finite {}", md,
                 (rms / hf.len() as f64).sqrt(), nd as f64 * 100.0 / hf.len() as f64, nan);
    }
    let per_pass = cfg.n_deltanet() as f64 * ms[0] + cfg.n_attention() as f64 * ms[1];
    println!("projected layers-only prefill: {:.0} ms per {} tokens = {:.1} tok/s", per_pass, t, t as f64 * 1e3 / per_pass);
}
