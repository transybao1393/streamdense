//! qd_compare: run the qwen-dense engine on the exact prompt token ids of an MLX reference file and compare the
//! greedy outputs token by token.
//!
//! cargo run --release --example qd_compare -- --model DIR --ref MLX.json --out OURS.json [--attn auto|splitk|flash|sdpa]
//!     [--prefill gemm|passes] [--decode pipelined|sync] [--prefill-reps N]
//!
//! --prefill-reps N (default 3): after the comparison, every prompt's prefill (reset + prefill, waited) is timed N
//! more times; the minimum goes to `prefill_min_s` (single runs are noisy).
//!
//! MLX.json (written by an MLX reference script, not part of this repository): {"prompts": [{"name", "prompt_ids": [..], "tokens": [..]
//! (greedy, no eos stop), "top": [[[id, logit] x 5] per generated position], ...}, ...]}.
//! Per prompt, in one process (the model is loaded once):
//! 1. free run: `generate(prompt_ids, n, no stop)`: tokens, prefill time, decode tok/s;
//! 2. teacher forced: prefill + steps fed MLX's tokens; at every position our greedy pick vs MLX's token, and our
//!    bf16-rounded top-2 (argmax_b compares bf16-rounded logits).
//!
//! Reported: identical prefix (free run vs MLX), position-wise identity, teacher-forced agreement, first divergence
//! (both tokens, our top-1 - top-2 gap, our logit gap between the two tokens, MLX's gap between them when MLX's top-5
//! has our token), peak phys_footprint of the process.
use qwen_dense::engine::peak_footprint;
use qwen_dense::engine::{DecodeMode, PrefillMode};
use qwen_dense::model::AttnImpl;
use qwen_dense::{Engine, EngineOptions, Tokenizer};
use serde_json::{json, Value};

fn main() {
    if let Err(e) = run() {
        eprintln!("qd_compare: {}", e);
        std::process::exit(1);
    }
}

fn rbf(f: f32) -> f32 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 { return f; }
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// (top-1 id, top-1, top-2 id, top-2) of the bf16-rounded logits, lowest index on ties (argmax_b's rule).
fn top2(l: &[f32]) -> (u32, f32, u32, f32) {
    let (mut b, mut bv, mut s, mut sv) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        let v = rbf(v);
        if v > bv { s = b; sv = bv; b = i; bv = v; } else if v > sv { s = i; sv = v; }
    }
    (b as u32, bv, s as u32, sv)
}

fn ids(v: &Value) -> Vec<u32> { v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default() }

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut rf, mut out, mut attn, mut pmode) = (None, None, None, AttnImpl::Auto, PrefillMode::Gemm);
    let mut reps = 3usize;
    let mut decode = DecodeMode::Pipelined;
    let mut i = 0;
    while i < args.len() {
        let val = |i: usize| args.get(i + 1).cloned().ok_or(format!("{} needs a value", args[i]));
        match args[i].as_str() {
            "--model" => { model = Some(val(i)?); i += 1; }
            "--ref" => { rf = Some(val(i)?); i += 1; }
            "--out" => { out = Some(val(i)?); i += 1; }
            "--attn" => { attn = AttnImpl::parse(&val(i)?)?; i += 1; }
            "--prefill" => { pmode = PrefillMode::parse(&val(i)?)?; i += 1; }
            "--decode" => { decode = DecodeMode::parse(&val(i)?)?; i += 1; }
            "--prefill-reps" => { reps = val(i)?.parse().map_err(|e| format!("--prefill-reps: {}", e))?; i += 1; }
            a => return Err(format!("unknown argument {:?}", a)),
        }
        i += 1;
    }
    let model = model.ok_or("--model DIR is required")?;
    let rf = rf.ok_or("--ref MLX.json is required")?;
    let r: Value = serde_json::from_str(&std::fs::read_to_string(&rf).map_err(|e| format!("{}: {}", rf, e))?)
        .map_err(|e| format!("{}: {}", rf, e))?;
    let prompts = r["prompts"].as_array().ok_or("ref: no prompts")?;
    let need = prompts.iter().map(|p| ids(&p["prompt_ids"]).len() + ids(&p["tokens"]).len()).max().unwrap_or(0);
    let ctx = (need + 8).div_ceil(256) * 256;
    let tk = Tokenizer::from_path(&model)?;
    let mut eng = Engine::load(&model, EngineOptions { max_ctx: ctx, attn, prefill: pmode, decode })?;
    println!("model {}  attn {}  prefill {}  decode {}  max_ctx {}  load {:.2} s  arena {:.3} GB", model, attn.name(), pmode.name(), decode.name(), ctx, eng.load_s,
             eng.arena_bytes as f64 / 1e9);
    let mut res = vec![];
    for p in prompts {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let pid = ids(&p["prompt_ids"]);
        let mt = ids(&p["tokens"]);
        let n = mt.len();
        // 1. free run
        let st = eng.generate(&pid, n, &[], |_| {})?;
        let ours = st.tokens.clone();
        let prefix = ours.iter().zip(&mt).take_while(|(a, b)| a == b).count();
        let same_pos = ours.iter().zip(&mt).filter(|(a, b)| a == b).count();
        // 2. teacher forced on MLX's tokens
        eng.reset();
        let mut tf = vec![];
        let mut pred = eng.prefill(&pid)?;
        for j in 0..n {
            let l = eng.logits();
            let (a, av, b, bv) = top2(&l);
            debug_assert_eq!(a, pred);
            tf.push((pred, av, b, bv, rbf(l[mt[j] as usize])));
            if j + 1 < n { pred = eng.step(mt[j])?; }
        }
        let tf_agree = tf.iter().zip(&mt).filter(|(t, m)| t.0 == **m).count();
        let div = if prefix < n {
            let (o, ov, o2, o2v, lm) = tf[prefix];
            let mtop = p["top"].get(prefix).and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let mlx_l = |id: u32| mtop.iter().find(|e| e[0].as_u64() == Some(id as u64)).and_then(|e| e[1].as_f64());
            let mg = match (mlx_l(mt[prefix]), mlx_l(o)) { (Some(a), Some(b)) => Some(a - b), _ => None };
            json!({"pos": prefix, "mlx_tok": mt[prefix], "our_tok": o, "mlx_tok_text": tk.decode(&[mt[prefix]]),
                   "our_tok_text": tk.decode(&[o]), "our_top1": ov, "our_top2_tok": o2, "our_top2": o2v,
                   "our_top1_minus_top2": ov - o2v, "our_gap_ours_minus_mlx_tok": ov - lm, "mlx_gap_mlx_minus_ours": mg})
        } else { Value::Null };
        let text40 = tk.decode(&ours[..ours.len().min(40)]);
        println!("\n[{}] prompt {} tok | prefix {}/{} | same pos {:.1}% | teacher-forced {:.1}% | prefill {:.3} s | decode {:.2} tok/s",
                 name, pid.len(), prefix, n, 100.0 * same_pos as f64 / n as f64, 100.0 * tf_agree as f64 / n as f64,
                 st.prefill_s, st.decode_tok_s());
        if !div.is_null() { println!("  first divergence {}", div); }
        println!("  ours: {:?}", text40);
        res.push(json!({"name": name, "prompt_tokens": pid.len(), "n": n, "tokens": ours, "prefix": prefix,
                        "same_pos": same_pos, "tf_agree": tf_agree, "divergence": div, "prefill_s": st.prefill_s,
                        "decode_s": st.decode_s, "decode_passes": st.decode_passes, "decode_tok_s": st.decode_tok_s(),
                        "text40": text40}));
    }
    for (p, r) in prompts.iter().zip(res.iter_mut()) {
        let pid = ids(&p["prompt_ids"]);
        let mut best = r["prefill_s"].as_f64().unwrap_or(f64::MAX);
        for _ in 0..reps {
            eng.reset();
            let t0 = std::time::Instant::now();
            eng.prefill(&pid)?;
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!("[{}] prefill min of {} runs: {:.3} s ({:.0} tok/s)", r["name"].as_str().unwrap_or("?"), reps + 1, best,
                 pid.len() as f64 / best);
        r["prefill_min_s"] = json!(best);
    }
    let peak = peak_footprint();
    println!("\npeak footprint {:.3} GB", peak as f64 / 1e9);
    if let Some(o) = out {
        let j = json!({"model": model, "attn": attn.name(), "prefill": pmode.name(), "decode": decode.name(), "max_ctx": ctx, "load_s": eng.load_s,
                       "arena_bytes": eng.arena_bytes, "peak_footprint_bytes": peak, "prompts": res});
        std::fs::write(&o, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| format!("{}: {}", o, e))?;
        println!("wrote {}", o);
    }
    Ok(())
}
