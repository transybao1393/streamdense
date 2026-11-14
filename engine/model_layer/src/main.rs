//! sd-model-layer inspect <model_dir>                                     report: family, engine, what applies
//! sd-model-layer run <model_dir> (--prompt TEXT | --prompt-ids 1,2,3) --max-tokens N [--max-ctx N]   greedy generation
//! `run` tokenizes the prompt raw (no BOS, no chat template) and loads the weights. RAM limits: [ram] of streamdense.toml.
//! `--prompt-ids` skips the tokenizer (folders without tokenizer.json, e.g. small test models): ids in, ids out.
use std::io::Write;
use streamdense_core::config;
use streamdense_model_layer::engine::check_text_run;
use streamdense_model_layer::report::inspect_text;
use streamdense_model_layer::{max_ctx_for, LoadOptions, Model};
use streamdense_tokenizer::{StreamDecoder, Tokenizer};

const USAGE: &str = "usage: sd-model-layer inspect <model_dir> | run <model_dir> (--prompt TEXT | --prompt-ids 1,2,3) --max-tokens N [--max-ctx N]";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match (args.first().map(|s| s.as_str()), args.get(1)) {
        (Some("inspect"), Some(dir)) if args.len() == 2 => inspect(dir),
        (Some("run"), Some(dir)) => run(dir, &args[2..]),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = r {
        eprintln!("ERROR: {}", e);
        std::process::exit(2);
    }
}

fn inspect(dir: &str) -> Result<(), String> {
    let conf = config::load()?;
    print!("{}", inspect_text(dir, &conf)?);
    Ok(())
}

fn run(dir: &str, args: &[String]) -> Result<(), String> {
    let (mut prompt, mut prompt_ids, mut max_new, mut max_ctx) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or(format!("{} needs a value", a));
        match a.as_str() {
            "--prompt" => prompt = Some(val()?),
            "--prompt-ids" => prompt_ids = Some(val()?.split(',').map(|x| x.trim().parse::<u32>()).collect::<Result<Vec<u32>, _>>()
                .map_err(|e| format!("--prompt-ids: {}", e))?),
            "--max-tokens" => max_new = Some(val()?.parse::<usize>().map_err(|e| format!("--max-tokens: {}", e))?),
            "--max-ctx" => max_ctx = Some(val()?.parse::<usize>().map_err(|e| format!("--max-ctx: {}", e))?),
            _ => return Err(format!("unknown argument {:?}\n{}", a, USAGE)),
        }
    }
    let max_new = max_new.ok_or("--max-tokens N is required")?;
    // family and tokenizer first: a bad folder fails before any weight is read
    check_text_run(dir, &streamdense_model_layer::detect(dir)?)?;
    let (tk, ids) = match (prompt, prompt_ids) {
        (Some(p), None) => {
            let tk = Tokenizer::from_path(dir)?;
            let ids = tk.encode(&p);
            (Some(tk), ids)
        }
        (None, Some(ids)) => (None, ids),
        _ => return Err(format!("give exactly one of --prompt TEXT, --prompt-ids LIST\n{}", USAGE)),
    };
    if ids.is_empty() { return Err("prompt encodes to no tokens".into()); }
    let conf = config::load()?;
    let opts = LoadOptions::from_config(max_ctx.unwrap_or(max_ctx_for(ids.len(), max_new)), &conf.cfg.ram);
    let mut m = Model::load(dir, &opts)?;
    eprintln!("family {} -> {}, prompt tokens {}, max_ctx {}", m.detected.entry.name, m.detected.entry.engine.name(), ids.len(), opts.max_ctx);
    let mut dec = StreamDecoder::new();
    let g = m.generate(&ids, max_new, &mut |t| {
        match &tk { Some(tk) => print!("{}", dec.push(tk, t)), None => print!("{} ", t) }
        let _ = std::io::stdout().flush();
    })?;
    println!("{}", if tk.is_some() { dec.finish() } else { String::new() });
    eprintln!("{} tokens, stop {}, prefill {:.3} s, decode {:.3} s, peak footprint {:.3} GB",
              g.tokens.len(), g.stop_reason, g.prefill_s, g.decode_s, m.peak_footprint() as f64 / 1e9);
    Ok(())
}
