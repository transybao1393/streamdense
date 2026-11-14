//! qd_run: greedy generation with the dense-engine.
//!
//! cargo run --release --example qd_run -- --model DIR --prompt "..." --max-tokens N [--tokens-out FILE.json]
//!     [--max-ctx N] [--attn auto|splitk|flash|sdpa] [--prefill gemm|passes] [--decode pipelined|sync] [--no-stop]
//!     (attn auto = flash prefill passes + MLX-mirror sdpa decode; prefill gemm = GEMM prefill for prompts >= 16 tokens)
//!
//! The prompt is tokenized raw (no chat template, no BOS). Generation stops at max-tokens or after an eos id of the
//! config (unless --no-stop). Prints the token ids, the text, load / prefill time, decode tok/s and the peak
//! phys_footprint; --tokens-out also writes them as JSON.
use dense_engine::engine::{footprint, peak_footprint};
use dense_engine::engine::{DecodeMode, PrefillMode};
use dense_engine::model::AttnImpl;
use dense_engine::{Engine, EngineOptions, Tokenizer};
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("qd_run: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut prompt, mut max_new, mut out, mut max_ctx, mut attn, mut stop) =
        (None, None, 64usize, None, None, AttnImpl::Auto, true);
    let mut pmode = PrefillMode::Gemm;
    let mut decode = DecodeMode::Pipelined;
    let mut i = 0;
    while i < args.len() {
        let val = |i: usize| args.get(i + 1).cloned().ok_or(format!("{} needs a value", args[i]));
        match args[i].as_str() {
            "--model" => { model = Some(val(i)?); i += 1; }
            "--prompt" => { prompt = Some(val(i)?); i += 1; }
            "--max-tokens" => { max_new = val(i)?.parse().map_err(|e| format!("--max-tokens: {}", e))?; i += 1; }
            "--tokens-out" => { out = Some(val(i)?); i += 1; }
            "--max-ctx" => { max_ctx = Some(val(i)?.parse::<usize>().map_err(|e| format!("--max-ctx: {}", e))?); i += 1; }
            "--attn" => { attn = AttnImpl::parse(&val(i)?)?; i += 1; }
            "--prefill" => { pmode = PrefillMode::parse(&val(i)?)?; i += 1; }
            "--decode" => { decode = DecodeMode::parse(&val(i)?)?; i += 1; }
            "--no-stop" => stop = false,
            "-h" | "--help" => {
                println!("qd_run --model DIR --prompt TEXT --max-tokens N [--tokens-out FILE.json] [--max-ctx N] [--attn auto|splitk|flash|sdpa] [--prefill gemm|passes] [--decode pipelined|sync] [--no-stop]");
                return Ok(());
            }
            a => return Err(format!("unknown argument {:?} (try --help)", a)),
        }
        i += 1;
    }
    let model = model.ok_or("--model DIR is required")?;
    let prompt = prompt.ok_or("--prompt TEXT is required")?;

    let tk = Tokenizer::from_path(&model)?;
    let ids = tk.encode(&prompt);
    if ids.is_empty() { return Err("prompt encodes to no tokens".into()); }
    let ctx = max_ctx.unwrap_or((ids.len() + max_new + 8).div_ceil(256) * 256);
    println!("model {}  prompt tokens {}  max_ctx {}  attn {}  prefill {}  decode {}", model, ids.len(), ctx, attn.name(), pmode.name(), decode.name());

    let mut eng = Engine::load(&model, EngineOptions { max_ctx: ctx, attn, prefill: pmode, decode })?;
    println!("load {:.2} s  ({} layers, hidden {}, heads {}/{}, bias {}, qk_norm {}, tied {})  arena {:.3} GB  footprint {:.3} GB",
             eng.load_s, eng.cfg.layers, eng.cfg.hidden, eng.cfg.heads, eng.cfg.kv_heads, eng.cfg.attention_bias,
             eng.cfg.qk_norm, eng.w.tied, eng.arena_bytes as f64 / 1e9, footprint() as f64 / 1e9);
    let stops: Vec<u32> = if stop { eng.cfg.eos.clone() } else { vec![] };

    let mut dec = dense_engine::StreamDecoder::new();
    print!("\n{}", prompt);
    std::io::stdout().flush().ok();
    let st = eng.generate(&ids, max_new, &stops, |t| {
        print!("{}", dec.push(&tk, t));
        std::io::stdout().flush().ok();
    })?;
    println!("{}\n", dec.finish());

    let text = tk.decode(&st.tokens);
    let peak = peak_footprint();
    println!("prompt ids {:?}", ids);
    println!("tokens {:?}", st.tokens);
    println!("text {:?}", text);
    println!("prefill {} tokens in {:.3} s ({:.1} tok/s)", st.prompt_tokens, st.prefill_s, st.prefill_tok_s());
    println!("decode {} passes in {:.3} s ({:.2} tok/s)", st.decode_passes, st.decode_s, st.decode_tok_s());
    println!("peak footprint {:.3} GB", peak as f64 / 1e9);

    if let Some(path) = out {
        let j = serde_json::json!({
            "model": model, "prompt": prompt, "prompt_tokens": ids, "tokens": st.tokens, "text": text,
            "max_ctx": ctx, "attn": attn.name(), "prefill": pmode.name(), "decode": decode.name(), "load_s": eng.load_s, "prefill_s": st.prefill_s,
            "prefill_tok_s": st.prefill_tok_s(), "decode_passes": st.decode_passes, "decode_s": st.decode_s,
            "decode_tok_s": st.decode_tok_s(), "peak_footprint_bytes": peak,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| format!("{}: {}", path, e))?;
        println!("wrote {}", path);
    }
    Ok(())
}
