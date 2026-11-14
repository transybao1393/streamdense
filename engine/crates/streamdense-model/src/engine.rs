//! In-RAM decode engine (Phase 1): every weight in one arena MTLBuffer under a hard cap, loaded with F_NOCACHE
//! preads; per-position passes ported from decode7b.swift (encodePass / generate / prefill), pipelined by default.
use crate::config::{Act, Config, Index};
use crate::ffn::{gemv4, DenseGated, Ffn, Q4, SparseFatRelu};
use std::time::Instant;
use streamdense_core::arena::{Arena, Region};
use streamdense_core::budget::Budget;
use streamdense_core::io::NoCacheFile;
use streamdense_core::{alloc, gb};
use streamdense_core::f16::{f16_to_f32, f32_to_f16};
use streamdense_metal::{commit, contents, gpu_ms, wait, Buffer, CmdBuf, Enc, Gpu};

pub const KMAX: usize = 4; // largest verify pass the scratch is sized for (k-token kernels, Phase 5)
pub const SPLITS: usize = 64;

#[derive(Clone, Debug)]
pub struct Options {
    pub cap: u64,           // hard RAM cap (bytes)
    pub reserve: u64,       // process reserve inside the cap (Metal / code / CPU heap)
    pub max_ctx: usize,
    pub ffn: String,        // "sparse" | "dense"
    pub ordered: bool,      // ordered (deterministic) compaction; false = decode7b atomic compact_thr
    pub head: String,       // "q8" | "fp16"
    pub depth: usize,       // pipelined command buffers in flight
    pub thr: f32,
    pub load_threads: usize,
}
impl Default for Options {
    fn default() -> Self {
        Options { cap: 5_500_000_000, reserve: 300_000_000, max_ctx: 2176, ffn: "sparse".into(), ordered: true, head: "q8".into(),
                  depth: 2, thr: f16_to_f32(f32_to_f16(0.01)), load_threads: 4 }
    }
}

struct LayerW { attn_norm: Region, ffn_norm: Region, qkv: Q4, o: Q4 }
struct KvL { k8: Region, ksb: Region, v8: Region, vsb: Region }
enum Head { Q8 { w: Region, s: Region, b: Region }, F16(Region) }

pub struct Scratch {
    pub ha: Region, pub hb: Region, pub xa: Region, pub xf: Region, pub qkv: Region, pub qr: Region, pub ao: Region,
    pub g: Region, pub hbuf: Region, pub idx: Region, pub am: Region, pub partial: Region, pub po: Region, pub pml: Region,
    pub counts: Region, pub args: Region, pub xfin: Region, pub logits: Region, pub outtok: Region, pub tokbuf: Region,
    pub rec: Region, pub actrec: Region, pub invf: Region, pub dummy: Region,
}

pub struct Engine {
    pub gpu: Gpu,
    pub buf: Buffer,
    pub arena: Arena,
    pub cfg: Config,
    pub opt: Options,
    layers: Vec<LayerW>,
    kv: Vec<KvL>,
    embed: Region,
    norm: Region,
    head: Head,
    pub s: Scratch,
    ffn: Box<dyn Ffn>,
    ncount: usize,
    pub load_s: f64,
    pub loaded_bytes: u64,
    pub budget_report: String,
}

#[derive(Default, Debug, Clone)]
pub struct GenResult {
    pub tokens: Vec<u32>,
    pub margins: Vec<f32>,
    pub gpu_ms: Vec<f64>,
    pub gen_wall_s: f64,
    pub act_prompt: Vec<f64>,
    pub act_gen: Vec<f64>,
    pub n_prompt: usize,
    pub n_gen: usize,
}

fn kv_bytes(cfg: &Config, max_ctx: usize) -> (usize, usize) {
    let n = cfg.kv_heads * max_ctx * cfg.head_dim;            // 8-bit values per layer per K (or V)
    let sb = cfg.kv_heads * max_ctx * (cfg.head_dim / 32) * 4; // half2 (scale, min) per group of 32
    (n, sb)
}

impl Engine {
    pub fn open(export_dir: &str, opt: Options) -> Result<Engine, String> {
        let idx = Index::open(export_dir)?;
        let cfg = idx.cfg.clone();
        match cfg.q_scale_folded {
            Some(v) => eprintln!("checkpoint: W_q scale {} folded (streamdense_q_scale_folded)", v),
            None => eprintln!("WARNING: checkpoint has no streamdense_q_scale_folded flag (W_q x0.9 win #2 not present)"),
        }
        let (h, d, nl, v) = (cfg.hidden, cfg.inter, cfg.layers, cfg.vocab);
        let sparse = opt.ffn == "sparse";
        if sparse && cfg.act != Act::Relu { return Err("sparse FFN needs a ReLU-family model".into()); }
        // ---- budget (everything the arena will hold), checked before any allocation
        let mut bud = Budget::new(opt.cap, opt.reserve);
        let lw: usize = (0..nl).map(|l| {
            let p = format!("l{}", l);
            idx.q4_bytes(&format!("{}.qkv", p)).unwrap() + idx.q4_bytes(&format!("{}.o", p)).unwrap()
        }).sum();
        bud.add("attention q/k/v/o (4-bit)", lw as u64);
        let ffn_names: Vec<&str> = if sparse { vec!["gate", "up", "downT"] } else { vec!["gate", "up", "down"] };
        let fw: usize = (0..nl).map(|l| ffn_names.iter().map(|n| idx.q4_bytes(&format!("l{}.{}", l, n)).unwrap()).sum::<usize>()).sum();
        bud.add(&format!("ffn {} (4-bit)", ffn_names.join("/")), fw as u64);
        let norms = (0..nl).map(|l| idx.t(&format!("l{}.attn_norm", l)).unwrap().nbytes * 2).sum::<usize>() + idx.t("norm")?.nbytes;
        bud.add("norms", norms as u64);
        bud.add("embeddings (fp16, resident in Phase 1)", idx.t("embed")?.nbytes as u64);
        let head_bytes = if opt.head == "q8" { v * h + 2 * v * (h / 64) * 2 } else { idx.t("lm_head")?.nbytes };
        bud.add(&format!("lm_head {}", opt.head), head_bytes as u64);
        let (kvn, kvsb) = kv_bytes(&cfg, opt.max_ctx);
        bud.add(&format!("KV cache 8-bit g32 @ {} ctx", opt.max_ctx), (nl * 2 * (kvn + kvsb)) as u64);
        let nchmax = (opt.max_ctx + KMAX + 15) / 16;
        let ncount = 2 * nl + 8;
        let qd = cfg.qdim();
        let sc: Vec<(&str, usize)> = vec![
            ("ha", KMAX * h * 4), ("hb", KMAX * h * 4), ("xa", KMAX * h * 2), ("xf", KMAX * h * 2),
            ("qkv", KMAX * (qd + 2 * cfg.kvdim()) * 2), ("qr", KMAX * qd * 2), ("ao", KMAX * qd * 2),
            ("g", KMAX * d * 2), ("hbuf", KMAX * d * 2), ("idx", d * 4), ("am", d.max(16)), ("partial", SPLITS * KMAX * h * 4),
            ("po", KMAX * cfg.heads * nchmax * cfg.head_dim * 4), ("pml", KMAX * cfg.heads * nchmax * 8), ("counts", ncount * 4),
            ("args", nl * 12), ("xfin", KMAX * h * 2), ("logits", KMAX * v * 4), ("outtok", KMAX * 4),
            ("tokbuf", (opt.max_ctx + 16) * 4), ("rec", (opt.max_ctx + 16) * 16), ("actrec", (opt.max_ctx + 16) * nl * 4),
            ("invf", 64 * 4), ("dummy", 64),
        ];
        let scratch_bytes: usize = sc.iter().map(|x| (x.1 + 255) / 256 * 256).sum::<usize>() + 16384 * 4;
        bud.add("scratch (activations, lists, records)", scratch_bytes as u64);
        let arena_bytes = bud.check()? as usize + 16 * 16384 * 64; // + alignment slack
        let report = bud.report();
        eprintln!("budget:\n{}", report);

        let mut gpu = Gpu::new()?;
        if arena_bytes > gpu.max_buffer() { return Err(format!("arena {} B > maxBufferLength {}", arena_bytes, gpu.max_buffer())); }
        let buf = gpu.buffer(arena_bytes)?;
        let mut arena = Arena::new(contents(&buf), arena_bytes);

        // ---- allocate + read (F_NOCACHE pread, several threads, no mmap)
        let file = NoCacheFile::open(&idx.bin)?;
        let mut jobs: Vec<(usize, usize, u64)> = Vec::new();
        let base = arena.base() as usize;
        let take = |arena: &mut Arena, jobs: &mut Vec<(usize, usize, u64)>, name: &str| -> Result<Region, String> {
            let t = idx.t(name)?;
            let r = arena.alloc(&format!("w:{}", name), t.nbytes)?;
            jobs.push((base + r.off, t.nbytes, t.off));
            Ok(r)
        };
        let q4 = |arena: &mut Arena, jobs: &mut Vec<(usize, usize, u64)>, name: &str| -> Result<Q4, String> {
            let sh = idx.t(&format!("{}.w", name))?.shape.clone();
            Ok(Q4 { r: sh[0], c: sh[1] * 8, w: take(arena, jobs, &format!("{}.w", name))?, s: take(arena, jobs, &format!("{}.s", name))?,
                    b: take(arena, jobs, &format!("{}.b", name))? })
        };
        let t0 = Instant::now();
        let embed = take(&mut arena, &mut jobs, "embed")?;
        let norm = take(&mut arena, &mut jobs, "norm")?;
        let mut layers = Vec::new();
        let mut ffw: Vec<[Q4; 3]> = Vec::new();
        for l in 0..nl {
            let p = format!("l{}", l);
            layers.push(LayerW { attn_norm: take(&mut arena, &mut jobs, &format!("{}.attn_norm", p))?,
                                 ffn_norm: take(&mut arena, &mut jobs, &format!("{}.ffn_norm", p))?,
                                 qkv: q4(&mut arena, &mut jobs, &format!("{}.qkv", p))?, o: q4(&mut arena, &mut jobs, &format!("{}.o", p))? });
            ffw.push([q4(&mut arena, &mut jobs, &format!("{}.{}", p, ffn_names[0]))?, q4(&mut arena, &mut jobs, &format!("{}.{}", p, ffn_names[1]))?,
                      q4(&mut arena, &mut jobs, &format!("{}.{}", p, ffn_names[2]))?]);
        }
        let head = if opt.head == "q8" {
            Head::Q8 { w: arena.alloc("w:lm_head_q8.w", v * h)?, s: arena.alloc("w:lm_head_q8.s", v * (h / 64) * 2)?,
                       b: arena.alloc("w:lm_head_q8.b", v * (h / 64) * 2)? }
        } else { Head::F16(take(&mut arena, &mut jobs, "lm_head")?) };
        let mut loaded = file.read_many(&jobs, opt.load_threads, 8 << 20)?;
        if let Head::Q8 { w, s, b } = &head {
            loaded += quant8_head(&file, idx.t("lm_head")?.off, &arena, *w, *s, *b, v, h, opt.load_threads)?;
        }
        let load_s = t0.elapsed().as_secs_f64();

        let mut kv = Vec::new();
        for l in 0..nl {
            kv.push(KvL { k8: arena.alloc(&format!("kv:{}.k", l), kvn)?, ksb: arena.alloc(&format!("kv:{}.ks", l), kvsb)?,
                          v8: arena.alloc(&format!("kv:{}.v", l), kvn)?, vsb: arena.alloc(&format!("kv:{}.vs", l), kvsb)? });
        }
        let mut sr = std::collections::HashMap::new();
        for (n, b) in &sc { sr.insert(*n, arena.alloc(&format!("s:{}", n), *b)?); }
        let s = Scratch { ha: sr["ha"], hb: sr["hb"], xa: sr["xa"], xf: sr["xf"], qkv: sr["qkv"], qr: sr["qr"], ao: sr["ao"], g: sr["g"],
            hbuf: sr["hbuf"], idx: sr["idx"], am: sr["am"], partial: sr["partial"], po: sr["po"], pml: sr["pml"], counts: sr["counts"],
            args: sr["args"], xfin: sr["xfin"], logits: sr["logits"], outtok: sr["outtok"], tokbuf: sr["tokbuf"], rec: sr["rec"],
            actrec: sr["actrec"], invf: sr["invf"], dummy: sr["dummy"] };
        {   // HF: 1 / theta ** (arange(0, 128, 2) / 128) in fp32
            let p = arena.slice_mut::<f32>(s.invf);
            for i in 0..64 { p[i] = 1.0 / cfg.rope_theta.powf((2 * i) as f32 / cfg.head_dim as f32); }
        }
        let ffn: Box<dyn Ffn> = if sparse {
            Box::new(SparseFatRelu { w: ffw, g: s.g.off, hbuf: s.hbuf.off, idx: s.idx.off, partial: s.partial.off, counts: s.counts.off,
                                     args: s.args.off, thr: opt.thr, h, d, ordered: opt.ordered, dummy: s.dummy.off })
        } else {
            let mode = if cfg.act == Act::Silu { 4 } else { 1 };
            Box::new(DenseGated { w: ffw, g: s.g.off, hbuf: s.hbuf.off, thr: opt.thr, mode, dummy: s.dummy.off })
        };
        let mut names: Vec<&str> = vec!["gemv4_m0", "gemv4_m2", "zero_u32", "embed", "rmsnorm", "rope_kv_q8", "attn2_partial_q8_k1",
                                        "attn_reduce", "argmax", "copy_u32", "gemv8_k1", "gemv_f16_k1"];
        names.extend(ffn.kernels());
        for n in names { gpu.pipeline(n)?; }
        Ok(Engine { gpu, buf, arena, cfg, opt, layers, kv, embed, norm, head, s, ffn, ncount, load_s, loaded_bytes: loaded, budget_report: report })
    }

    fn attn_chunk(n: usize) -> usize { let mut ch = 256; while ch > 32 && (n + ch - 1) / ch < 4 { ch /= 2; } ch }

    /// one pass over positions pos..pos+k (tokens from tokbuf); decode7b encodePass
    pub fn encode_pass(&self, cb: &CmdBuf, pos: usize, k: usize, plen: usize, head: bool, write_tok: bool) {
        assert!(k == 1, "Phase 1: k = 1 passes only");
        let c = &self.cfg;
        let s = &self.s;
        let e = Enc::new(&self.gpu, cb, &self.buf);
        let dm = s.dummy.off;
        e.pso("zero_u32").b(0, s.counts.off).u(1, self.ncount as u32);
        e.threads((self.ncount, 1), 64);
        e.pso("embed").b(0, self.embed.off).b(1, s.tokbuf.off).b(2, s.ha.off).u(3, pos as u32).u(4, c.hidden as u32);
        e.threads((c.hidden, k), 256);
        let grp = (c.heads / c.kv_heads) as u32;
        for l in 0..c.layers {
            let lw = &self.layers[l];
            self.norm_enc(&e, s.ha.off, lw.attn_norm.off, s.xa.off, k);
            gemv4(&e, &lw.qkv, s.xa.off, 0, s.qkv.off, dm, dm, dm, self.opt.thr);
            let kv = &self.kv[l];
            e.pso("rope_kv_q8").b(0, s.qkv.off).b(1, s.qr.off).b(2, dm).b(3, dm).b(4, kv.k8.off).b(5, kv.ksb.off).b(6, kv.v8.off)
                .b(7, kv.vsb.off).b(8, s.invf.off).u(9, pos as u32).u(10, self.opt.max_ctx as u32).u(11, c.kv_heads as u32);
            e.groups((c.heads, k), c.head_dim);
            let n0 = pos + 1;
            let ch = Self::attn_chunk(n0 + k - 1);
            let nch = (n0 + k - 1 + ch - 1) / ch;
            e.pso("attn2_partial_q8_k1").b(0, s.qr.off).b(1, dm).b(2, dm).b(3, kv.k8.off).b(4, kv.ksb.off).b(5, kv.v8.off).b(6, kv.vsb.off)
                .b(7, s.po.off).b(8, s.pml.off).u(9, n0 as u32).u(10, self.opt.max_ctx as u32).u(11, ch as u32).u(12, grp);
            e.groups((nch, c.heads), 128);
            e.pso("attn_reduce").b(0, s.po.off).b(1, s.pml.off).b(2, s.ao.off).u(3, nch as u32);
            e.groups((c.heads, k), 32);
            gemv4(&e, &lw.o, s.ao.off, 2, dm, dm, s.ha.off, s.hb.off, self.opt.thr);
            self.norm_enc(&e, s.hb.off, lw.ffn_norm.off, s.xf.off, k);
            self.ffn.encode(&e, l, s.xf.off, s.hb.off, s.ha.off);
        }
        if head {
            self.norm_enc(&e, s.ha.off, self.norm.off, s.xfin.off, k);
            match &self.head {
                Head::Q8 { w, s: hs, b } => {
                    e.pso("gemv8_k1").b(0, w.off).b(1, hs.off).b(2, b.off).b(3, s.xfin.off).b(4, s.logits.off).u(5, c.vocab as u32).u(6, c.hidden as u32);
                    e.groups(((c.vocab + 7) / 8, 1), 256);
                }
                Head::F16(w) => {
                    e.pso("gemv_f16_k1").b(0, w.off).b(1, s.xfin.off).b(2, s.logits.off).u(3, c.vocab as u32).u(4, c.hidden as u32);
                    e.groups(((c.vocab + 7) / 8, 1), 256);
                }
            }
            e.pso("argmax").b(0, s.logits.off).b(1, s.outtok.off).b(2, s.tokbuf.off).u(3, c.vocab as u32).u(4, pos as u32)
                .u(5, plen as u32).u(6, write_tok as u32).b(7, s.rec.off);
            e.groups((k, 1), 1024);
        }
        if self.ffn.sparse() && pos < self.opt.max_ctx {
            e.pso("copy_u32").b(0, s.counts.off).b(1, s.actrec.off + pos * c.layers * 4).u(2, c.layers as u32);
            e.threads((c.layers, 1), 32);
        }
        e.end();
    }
    fn norm_enc(&self, e: &Enc, x: usize, w: usize, out: usize, k: usize) {
        e.pso("rmsnorm").b(0, x).b(1, w).b(2, out).u(3, self.cfg.hidden as u32).f(4, self.cfg.eps);
        e.groups((k, 1), 1024);
    }

    fn tok(&self) -> &mut [u32] { self.arena.slice_mut::<u32>(self.s.tokbuf) }
    pub fn set_prompt(&self, ids: &[u32]) {
        let t = self.tok();
        t.iter_mut().for_each(|x| *x = 0);
        t[..ids.len()].copy_from_slice(ids);
    }
    fn rec_at(&self, p: usize) -> (u32, f32, f32) {
        let r = self.arena.slice_mut::<f32>(self.s.rec);
        (r[4 * p].to_bits(), r[4 * p + 1], r[4 * p + 2])
    }
    fn act_at(&self, p: usize) -> &[u32] { &self.arena.slice_mut::<u32>(self.s.actrec)[p * self.cfg.layers..(p + 1) * self.cfg.layers] }

    /// positions 0..upto without waiting per token (no CPU dependency), lm_head skipped
    pub fn prefill(&self, upto: usize, plen: usize) -> Result<(), String> {
        let mut last: Option<CmdBuf> = None;
        for p in 0..upto {
            let cb = self.gpu.command_buffer();
            self.encode_pass(&cb, p, 1, plen, false, true);
            commit(&cb);
            if p % 64 == 63 { wait(&cb)?; }
            last = Some(cb);
        }
        if let Some(cb) = last { wait(&cb)?; }
        Ok(())
    }

    /// greedy decode (decode7b generate): pipelined (up to depth command buffers in flight, the GPU argmax writes the
    /// next token) or one synchronous pass per position
    pub fn generate(&self, prompt: &[u32], n_new: usize, pipe: bool) -> Result<GenResult, String> {
        let nl = self.cfg.layers;
        let mut r = GenResult { act_prompt: vec![0.0; nl], act_gen: vec![0.0; nl], ..Default::default() };
        let p_len = prompt.len();
        if p_len + n_new + 1 > self.opt.max_ctx { return Err(format!("prompt {} + {} new > max_ctx {}", p_len, n_new, self.opt.max_ctx)); }
        self.set_prompt(prompt);
        let sparse = self.ffn.sparse();
        let add_active = |r: &mut GenResult, p: usize| {
            if !sparse { return; }
            let c = self.act_at(p);
            for l in 0..nl { if p < p_len { r.act_prompt[l] += c[l] as f64 } else { r.act_gen[l] += c[l] as f64 } }
            if p < p_len { r.n_prompt += 1 } else { r.n_gen += 1 }
        };
        let eos = self.cfg.eos;
        if pipe {
            self.prefill(p_len - 1, p_len)?;
            for p in 0..p_len - 1 { add_active(&mut r, p); }
            let mut inflight: std::collections::VecDeque<(CmdBuf, usize)> = Default::default();
            let mut stop = false;
            let mut p = p_len - 1;
            let t0 = Instant::now();
            let mut t_end = t0;
            let mut retire = |inflight: &mut std::collections::VecDeque<(CmdBuf, usize)>, r: &mut GenResult, stop: &mut bool| -> Result<(), String> {
                let (cb, q) = inflight.pop_front().unwrap();
                wait(&cb)?;
                if *stop { return Ok(()); }
                t_end = Instant::now();
                r.gpu_ms.push(gpu_ms(&cb));
                add_active(r, q);
                let (tok, b1, b2) = self.rec_at(q);
                r.tokens.push(tok);
                r.margins.push(b1 - b2);
                if tok == eos { *stop = true; }
                Ok(())
            };
            while p < p_len - 1 + n_new && !stop {
                let cb = self.gpu.command_buffer();
                self.encode_pass(&cb, p, 1, p_len, true, true);
                commit(&cb);
                inflight.push_back((cb, p));
                p += 1;
                if inflight.len() >= self.opt.depth { retire(&mut inflight, &mut r, &mut stop)?; }
            }
            while !inflight.is_empty() { retire(&mut inflight, &mut r, &mut stop)?; }
            r.gen_wall_s = (t_end - t0).as_secs_f64();
            return Ok(r);
        }
        let mut wall = 0.0;
        for p in 0..p_len - 1 + n_new {
            let is_gen = p >= p_len - 1;
            let t0 = Instant::now();
            let cb = self.gpu.command_buffer();
            self.encode_pass(&cb, p, 1, p_len, is_gen, true);
            commit(&cb);
            wait(&cb)?;
            add_active(&mut r, p);
            if is_gen {
                wall += t0.elapsed().as_secs_f64();
                r.gpu_ms.push(gpu_ms(&cb));
                let (tok, b1, b2) = self.rec_at(p);
                let out = self.arena.slice_mut::<u32>(self.s.outtok)[0];
                if out != tok { return Err(format!("GPU outtok {} != record {}", out, tok)); }
                r.tokens.push(tok);
                r.margins.push(b1 - b2);
                if tok == eos { break; }
            }
        }
        r.gen_wall_s = wall;
        Ok(r)
    }

    /// decode7b MODE=bench row: prefill P-1, NGEN synchronous passes (wall + GPU), then the same NGEN positions
    /// re-decoded pipelined (ms/token = total wall / NGEN); returns (sync wall ms, sync gpu ms, pipelined ms, tokens equal)
    pub fn bench_ctx(&self, prompt: &[u32], ngen: usize) -> Result<(f64, f64, f64, bool, f64), String> {
        let p_len = prompt.len();
        self.set_prompt(prompt);
        self.prefill(p_len - 1, p_len)?;
        let (mut wall, mut gpu, mut toks, mut act) = (0.0, 0.0, vec![], 0.0);
        for p in (p_len - 1)..(p_len - 1 + ngen) {
            let t0 = Instant::now();
            let cb = self.gpu.command_buffer();
            self.encode_pass(&cb, p, 1, p_len, true, true);
            commit(&cb);
            wait(&cb)?;
            wall += t0.elapsed().as_secs_f64() * 1e3;
            gpu += gpu_ms(&cb);
            toks.push(self.arena.slice_mut::<u32>(self.s.outtok)[0]);
            if self.ffn.sparse() { act += self.arena.slice_mut::<u32>(self.s.counts)[..self.cfg.layers].iter().map(|&x| x as f64).sum::<f64>(); }
        }
        let t0 = Instant::now();
        let mut inflight: std::collections::VecDeque<CmdBuf> = Default::default();
        for p in (p_len - 1)..(p_len - 1 + ngen) {
            let cb = self.gpu.command_buffer();
            self.encode_pass(&cb, p, 1, p_len, true, true);
            commit(&cb);
            inflight.push_back(cb);
            if inflight.len() >= self.opt.depth { wait(&inflight.pop_front().unwrap())?; }
        }
        while let Some(cb) = inflight.pop_front() { wait(&cb)?; }
        let pipe_ms = t0.elapsed().as_secs_f64() * 1e3 / ngen as f64;
        let tp = self.tok();
        let same = (0..ngen).all(|i| tp[p_len + i] == toks[i]);
        let af = act / (ngen * self.cfg.layers * self.cfg.inter) as f64;
        Ok((wall / ngen as f64, gpu / ngen as f64, pipe_ms, same, af))
    }

    pub fn arena_report(&self) -> String {
        format!("arena {:.3} GB used of {:.3} GB: weights {:.3}, KV {:.3}, scratch {:.3}", gb(self.arena.used() as u64), gb(self.arena.size() as u64),
            gb(self.arena.total("w:") as u64), gb(self.arena.total("kv:") as u64), gb(self.arena.total("s:") as u64))
    }
    pub fn heap_mb(&self) -> f64 { alloc::heap_now() as f64 / 1e6 }
}

/// lm_head 8-bit g64 quantized at load from the fp16 head, bit-exact with decode7b.swift quant8 (fp32 min / max,
/// scale = max(hi - lo, 1e-8) / 255, round half to even, clamp, fp16 scale + min). Read in row chunks (small heap).
fn quant8_head(file: &NoCacheFile, off: u64, arena: &Arena, w: Region, s: Region, b: Region, rows: usize, cols: usize, threads: usize) -> Result<u64, String> {
    let g = cols / 64;
    let chunk = 256usize;
    let nchunks = (rows + chunk - 1) / chunk;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let err = std::sync::Mutex::new(None::<String>);
    let (wp, sp, bp) = (arena.ptr(w) as usize, arena.ptr(s) as usize, arena.ptr(b) as usize);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut tmp = vec![0u16; chunk * cols];
                loop {
                    let ci = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if ci >= nchunks { break; }
                    let r0 = ci * chunk;
                    let nr = chunk.min(rows - r0);
                    if let Err(e) = file.read_at(tmp.as_mut_ptr() as *mut u8, nr * cols * 2, off + (r0 * cols * 2) as u64) {
                        *err.lock().unwrap() = Some(e); break;
                    }
                    for r in 0..nr {
                        let row = r0 + r;
                        for gi in 0..g {
                            let x = &tmp[r * cols + gi * 64..r * cols + gi * 64 + 64];
                            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                            for &v in x { let f = f16_to_f32(v); lo = lo.min(f); hi = hi.max(f); }
                            let scale = (hi - lo).max(1e-8) / 255.0;
                            for (i, &v) in x.iter().enumerate() {
                                let q = ((f16_to_f32(v) - lo) / scale).round_ties_even().max(0.0).min(255.0);
                                unsafe { *((wp + row * cols + gi * 64 + i) as *mut u8) = q as u8; }
                            }
                            unsafe {
                                *((sp + (row * g + gi) * 2) as *mut u16) = f32_to_f16(scale);
                                *((bp + (row * g + gi) * 2) as *mut u16) = f32_to_f16(lo);
                            }
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() { return Err(e); }
    Ok((rows * cols * 2) as u64)
}
