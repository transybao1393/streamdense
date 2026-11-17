//! MTP warm-up v2 cost at the 27B shapes, synthetic (random 4-bit weights, decode_bench `Synth`; only the 27B
//! config.json is read, no weights loaded): the MTP decoder layer (an attention layer) over the 128 rows of one
//! prefill chunk, today (16 exact `encode_layer` passes of 8 rows, ctrl.pos advanced between them as mtp_prep does)
//! vs v2 (one `encode_layer_prefill` of 128 rows), at kv positions 0 / 8192 / 16384 / 32000. bf16 KV (Synth).
//! The fc part is timed at the real shape in streamdense-metal tests/gemv_bf16.rs (`bf16_gemv_wide_k_kinv_and_time`).
//! Run: cargo test --release --offline -p streamdense-model --test mtp_warm_synth -- --nocapture --test-threads 1
use streamdense_metal::Enc;
use streamdense_model::decode_bench::{Synth, SynthOpt};
use streamdense_model::qwen_engine::ATTN_CHUNK;
use streamdense_model::qwen_layers::{encode_layer, encode_layer_prefill, gemm_pipelines, prefill_pipelines, scratch_sizes, GemvImpl, PassIo};
use streamdense_model::{LayerKind, QwenConfig};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

#[test]
fn mtp_layer_exact_vs_fast_27b_shapes() {
    let dir = format!("{}/models/qwen38_27b_mlx4", ROOT);
    let Ok(cfg) = QwenConfig::from_model_dir(&dir) else { eprintln!("SKIP: no 27B config.json"); return };
    let max_ctx = 32768 + 256;
    let opt = SynthOpt { copies: 1, max_ctx, pos: 0, kc: 0, gemv: GemvImpl::Mma, kmax: 8, seqs: 1 };
    let mut s = Synth::new(&cfg, opt).expect("synth");
    for p in prefill_pipelines() { s.gpu.pipeline(p).unwrap(); }
    for p in gemm_pipelines() { s.gpu.pipeline(&p).unwrap(); }
    // 128-row pass scratch from the arena slack (po / pml / logits / invf stay the Synth's: exact passes use <= 8 rows)
    let t = 128;
    let nch = max_ctx.div_ceil(ATTN_CHUNK);
    let mut off = std::collections::HashMap::new();
    for (n, b) in scratch_sizes(&cfg, t, nch) {
        if ["po", "pml", "logits", "invf"].contains(&n) { continue; }
        off.insert(n, s.arena.alloc(n, b.div_ceil(256) * 256).expect("arena slack").off);
    }
    let io = PassIo { h: off["h"], x: off["x"], hf: off["hf"], p: off["p"], cv: off["cv"], g: off["g"], beta: off["beta"],
                      y: off["y"], yn: off["yn"], ao: off["ao"], f: off["f"], ..s.io };
    let l = (0..cfg.layers).find(|&l| cfg.kinds[l] == LayerKind::Attention).unwrap();
    let lw = &s.full[l];
    let kv = s.kv[l];
    let hb = cfg.hidden * 2;
    let opos = s.opt.pos;
    for pos in [0usize, 8192, 16384, 32000] {
        s.opt.pos = pos;
        let exact = |e: &Enc| -> Result<(), String> {
            for b in 0..t / 8 {
                let p = pos + b * 8;
                let bio = PassIo { k: 8, pos: p, nch_max: (p + 8).div_ceil(ATTN_CHUNK).min(nch), h: io.h + b * 8 * hb, ..io };
                encode_layer(e, &cfg, lw, None, kv.as_ref(), &bio, false, None, cfg.layers)?;
                for _ in 0..8 { e.pso("advance1").b(0, io.ctrl); e.threads((1, 1), 1); }
            }
            Ok(())
        };
        let fast = |e: &Enc| -> Result<(), String> {
            let fio = PassIo { k: t, pos, nch_max: (pos + t).div_ceil(ATTN_CHUNK).min(nch), ..io };
            encode_layer_prefill(e, &cfg, lw, None, kv.as_ref(), &fio, cfg.layers)
        };
        let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let te = med(s.time_with(&exact, None, 7).unwrap());
        let tf = med(s.time_with(&fast, None, 7).unwrap());
        println!("  MTP layer, 128 rows at pos {:5}: exact 16 x 8 rows {:6.2} ms | fast prefill layer {:6.2} ms | saves {:6.2} ms",
                 pos, te, tf, te - tf);
    }
    s.opt.pos = opos;
}
