//! streamdense forheader export --data DIR [--out FILE] [--model DIR]   global score file from a research T1 directory
//! streamdense forheader install FILE [--out FILE] [--model DIR]        check a score file and copy it into place
//! streamdense forheader info [FILE] [--model DIR]                      dims + whether it fits the model
//! The 12 GB mode's input (`[forheader] t1`, default ~/.streamdense/forheader/t1.sdt1). CPU only: reads config.json /
//! safetensors headers of the model, never its weights.
use crate::{config_root, load_config, watcher::expand_home, Args};
use std::path::{Path, PathBuf};
use streamdense_model::ffn_set::T1;
use streamdense_model::qwen_config::QwenConfig;

fn die(e: impl std::fmt::Display) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }

/// `--out` or the configured `[forheader] t1`.
fn target(a: &Args) -> PathBuf { a.s("--out").map(PathBuf::from).unwrap_or_else(|| expand_home(&load_config().cfg.forheader.t1)) }

fn model_dir(a: &Args) -> String {
    a.s("--model").unwrap_or_else(|| { let c = load_config(); config_root(&c).join(&c.cfg.gateway.model).display().to_string() })
}

/// Dims check against the model (warning only when the model dir is missing: the file may be meant for another box).
fn check_model(t: &T1, model: &str) -> Result<(), String> {
    if !Path::new(model).join("config.json").exists() { eprintln!("note: {} not found: dims not checked against a model", model); return Ok(()); }
    let c = QwenConfig::from_model_dir(model)?;
    t.check_dims(c.layers, c.inter)
}

pub fn main(a: &Args) {
    let sub = a.0.get(2).map(String::as_str).unwrap_or("");
    match sub {
        "export" => {
            let d = a.s("--data").unwrap_or_else(|| die("--data DIR (the research T1 directory: meta.json, pNN_total.f32, down_norm.f32)"));
            let t = T1::from_research_dir(Path::new(&d)).unwrap_or_else(|e| die(e));
            check_model(&t, &model_dir(a)).unwrap_or_else(|e| die(e));
            let out = target(a);
            t.write(&out).unwrap_or_else(|e| die(e));
            println!("wrote {} ({} layers x {} neurons, {:.1} MB)", out.display(), t.layers, t.inter, T1::file_bytes(t.layers, t.inter) as f64 / 1e6);
        }
        "install" => {
            let src = a.0.get(3).filter(|s| !s.starts_with("--")).unwrap_or_else(|| die("install FILE"));
            let t = T1::read(Path::new(src)).unwrap_or_else(|e| die(e));
            check_model(&t, &model_dir(a)).unwrap_or_else(|e| die(e));
            let out = target(a);
            t.write(&out).unwrap_or_else(|e| die(e));
            println!("installed {} -> {} ({} layers x {} neurons)", src, out.display(), t.layers, t.inter);
        }
        "info" => {
            let p = a.0.get(3).filter(|s| !s.starts_with("--")).map(PathBuf::from).unwrap_or_else(|| target(a));
            let t = T1::read(&p).unwrap_or_else(|e| die(format!("{} (export one: streamdense forheader export --data DIR)", e)));
            let model = model_dir(a);
            println!("{}: {} layers x {} neurons", p.display(), t.layers, t.inter);
            match check_model(&t, &model) {
                Ok(()) => println!("fits {}", model),
                Err(e) => die(e),
            }
        }
        _ => die("usage: streamdense forheader export --data DIR [--out FILE] | install FILE [--out FILE] | info [FILE]  (--model DIR)"),
    }
}
