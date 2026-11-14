//! What applies to a model folder, as facts from the existing code and the files present. Reads config.json, file
//! names, the tokenizer files and the small per-model files (keep-list, score file, drafter header); never the weights.
//! No speed or memory claim: `FOOTER` says so on every report.
//!
//! `optimizations` is the one list of existing optimizations: name, the engines that have it, the state there.
use crate::family::{detect, Detected, EngineKind, Family};
use qwen_dense::config::DenseConfig;
use serde_json::Value;
use std::path::{Path, PathBuf};
use streamdense_core::config;
use streamdense_model::dflash::{DfPlan, CONFIG_FILE, WEIGHTS_FILE};
use streamdense_model::draft_vocab::KeepList;
use streamdense_model::ffn_set::T1;
use streamdense_model::QwenConfig;
use streamdense_tokenizer::Tokenizer;

const FOOTER: &str = "a detected family does not prove a model runs; speed and memory of a new model have to be measured.\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    Available,
    /// file name
    NeedsFile(String),
    OffUntilTested,
    OtherEngine,
}

impl State {
    pub fn label(&self) -> String {
        match self {
            State::Available => "available".into(),
            State::NeedsFile(n) => format!("needs per-model file {}, missing", n),
            State::OffUntilTested => "off by default until tested".into(),
            State::OtherEngine => "only in the other engine".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptLine {
    pub name: &'static str,
    pub state: State,
    /// where it is set / why this state
    pub note: String,
}

pub struct Report {
    pub dir: String,
    /// streamdense.toml in use (or built-in defaults)
    pub config_source: String,
    pub detected: Detected,
    pub quantization: String,
    /// what the chosen engine's config reader / validator refuses, by name (empty = nothing)
    pub config_problems: Vec<String>,
    /// file (or alternatives) -> present
    pub files: Vec<(&'static str, bool)>,
    /// "loads (N ids)" or the loader's error
    pub tokenizer: String,
    pub optimizations: Vec<OptLine>,
}

/// `quantization` / `quantization_config` block of config.json (the fields `DenseConfig::from_json` reads).
fn quantization(root: &Value) -> String {
    let q = if root["quantization"].is_object() { &root["quantization"] } else { &root["quantization_config"] };
    if !q.is_object() { return "none in config.json".into(); }
    match (q["bits"].as_u64(), q["group_size"].as_u64()) {
        (Some(b), Some(g)) => format!("{}-bit, group {}", b, g),
        _ => format!("block present, bits / group_size not readable: {}", q),
    }
}

/// "~/" = home, relative = against the dir of the streamdense.toml in use.
fn conf_path(p: &str, source: &Option<PathBuf>) -> PathBuf {
    if let Some(r) = p.strip_prefix("~/") { return PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(r); }
    match source.as_ref().and_then(|s| s.parent()) { Some(d) => d.join(p), None => PathBuf::from(p) }
}

/// MTP head: `mtp_file` of config.json (safetensors.rs `open_mlx_dir`) and `mtp_num_hidden_layers` = 1 (weights.rs).
fn mtp_state(dir: &Path, root: &Value, q: Option<&QwenConfig>) -> (State, String) {
    let name = root["mtp_file"].as_str();
    let layers = q.map_or(0, |c| c.mtp_layers);
    match name {
        Some(f) if dir.join(f).exists() && layers == 1 => (State::Available, format!("{} present; [gateway] mtp / QwenOptions.mtp", f)),
        Some(f) if dir.join(f).exists() => (State::NeedsFile(f.to_string()), format!("file present, but mtp_num_hidden_layers = {} (weights.rs wants 1)", layers)),
        Some(f) => (State::NeedsFile(f.to_string()), "named by config.json mtp_file".into()),
        None => (State::NeedsFile("<config.json mtp_file>".into()), "config.json has no mtp_file".into()),
    }
}

fn draft_vocab_state(dir: &Path, q: Option<&QwenConfig>) -> (State, String) {
    let name = "draft_vocab.keep";
    let p = dir.join(name);
    if !p.exists() { return (State::NeedsFile(name.into()), "`streamdense vocab keep` writes it; needs MTP".into()); }
    match q.ok_or("config not readable".to_string()).and_then(|c| KeepList::read_for(&p, c.vocab)) {
        Ok(k) => (State::Available, format!("{} ids; needs MTP; [gateway] draft_vocab", k.ids.len())),
        Err(e) => (State::NeedsFile(name.into()), format!("file present but not usable: {}", e)),
    }
}

fn drafter_state(conf: &config::Loaded, q: Option<&QwenConfig>) -> (State, String) {
    let name = format!("{} + {}", CONFIG_FILE, WEIGHTS_FILE);
    let d = &conf.cfg.gateway.drafter;
    if d.is_empty() { return (State::NeedsFile(name), "[gateway] drafter is empty; trained per model by the kit".into()); }
    let p = conf_path(d, &conf.source);
    match q.ok_or("config not readable".to_string()).and_then(|c| DfPlan::read(&p, c)) {
        Ok(_) => (State::Available, format!("[gateway] drafter = {}; replaces MTP", p.display())),
        Err(e) => (State::NeedsFile(name), format!("[gateway] drafter = {}: {}", p.display(), e)),
    }
}

fn score_file_state(conf: &config::Loaded, q: Option<&QwenConfig>) -> (State, String) {
    let p = conf_path(&conf.cfg.forheader.t1, &conf.source);
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    if !p.exists() { return (State::NeedsFile(name), "`streamdense forheader export` writes it; [forheader] t1".into()); }
    let r = q.ok_or("config not readable".to_string()).and_then(|c| T1::read(&p).and_then(|t| t.check_dims(c.layers, c.inter)));
    match r {
        Ok(()) => (State::Available, format!("{} fits this model's dims", p.display())),
        Err(e) => (State::NeedsFile(name), format!("{}: {}", p.display(), e)),
    }
}

/// Every existing optimization for a model of `engine`.
fn optimizations(engine: EngineKind, dir: &Path, root: &Value, q: Option<&QwenConfig>, conf: &config::Loaded) -> Vec<OptLine> {
    use EngineKind::{Main, QwenDense};
    let g = &conf.cfg.gateway;
    let off = |note: &str| (State::OffUntilTested, note.to_string());
    let on = |note: &str| (State::Available, note.to_string());
    let mtp = mtp_state(dir, root, q);
    let spec = match &mtp.0 {
        State::Available => (State::Available, format!("[gateway] spec_depth = {}, needs MTP", g.spec_depth.arg())),
        s => (s.clone(), "needs MTP".to_string()),
    };
    let rows: Vec<(&'static str, &[EngineKind], (State, String))> = vec![
        ("MTP draft head", &[Main], mtp),
        ("adaptive spec depth + draft table", &[Main], spec),
        ("draft vocab", &[Main], draft_vocab_state(dir, q)),
        ("DFlash drafter", &[Main], drafter_state(conf, q)),
        ("forheader score file", &[Main], score_file_state(conf, q)),
        ("forheader 12 GB mode", &[Main], off("[forheader] mode, research, default \"off\"; needs the score file")),
        ("fused decode", &[Main], on("QwenOptions.decode_fused default off; only the 12 GB mode turns it on ([forheader] fused)")),
        ("fast prompt read", &[Main, QwenDense], match engine {
            Main => on("QwenOptions.prefill_exact = false (default); streamed form only in the 12 GB mode ([forheader] fast)"),
            QwenDense => on("PrefillMode::Gemm (default), prompts of >= 16 tokens"),
        }),
        ("pipelined decode", &[Main, QwenDense], match engine {
            Main => on("QwenEngine::generate(.., pipe = true); this layer's run uses feed_stream, as qwen serve does"),
            QwenDense => on("DecodeMode::Pipelined (default)"),
        }),
        ("verify_fused", &[Main], off("QwenOptions.verify_fused; streamdense.toml: default OFF until the 27B check")),
        ("verify mixers", &[Main], off("QwenOptions.verify_mixers; default off, needs a 27B check")),
        ("ctx_draft", &[Main], off("[forheader] ctx_draft, 12 GB mode; default OFF until the 27B test")),
        ("ffn select", &[Main], off("QwenOptions.ffn_select, test hook; default off")),
        ("read mask", &[Main], off("[watcher] mask; default off, \"on\" only after a 27B check")),
    ];
    rows.into_iter().map(|(name, owners, (state, note))| {
        if owners.contains(&engine) { OptLine { name, state, note } } else { OptLine { name, state: State::OtherEngine, note: String::new() } }
    }).collect()
}

/// The report of a folder, from its family's own code: the one place reports are routed. Reads config, tokenizer and
/// safetensors headers; never the weights, no Metal.
pub fn inspect_text(dir: &str, conf: &config::Loaded) -> Result<String, String> {
    match detect(dir)?.entry.family {
        Family::Qwen => Ok(Report::build(dir, conf)?.text()),
    }
}

impl Report {
    pub fn build(dir: &str, conf: &config::Loaded) -> Result<Report, String> {
        let detected = detect(dir)?;
        if detected.entry.family != Family::Qwen { return Err(format!("{}: has its own report (report::inspect_text)", detected.entry.name)); }
        let p = Path::new(dir);
        let path = p.join("config.json");
        let txt = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let root: Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path.display(), e))?;
        let engine = detected.entry.engine;
        let (mut config_problems, mut qwen) = (vec![], None);
        match engine {
            EngineKind::QwenDense => match DenseConfig::from_model_dir(dir) {
                Ok(c) => {
                    config_problems = c.unsupported();
                    if c.quant.is_none() { config_problems.push("no quantization block: the weight loader wants an MLX 4-bit model".into()); }
                }
                Err(e) => config_problems.push(e),
            },
            EngineKind::Main => match QwenConfig::from_model_dir(dir) {
                Ok(c) => {
                    if let Err(e) = c.validate() { config_problems.push(e); }
                    qwen = Some(c);
                }
                Err(e) => config_problems.push(e),
            },
        }
        let has = |n: &str| p.join(n).exists();
        let weights: (&'static str, bool) = match engine {
            EngineKind::QwenDense => ("model.safetensors.index.json or model.safetensors", has("model.safetensors.index.json") || has("model.safetensors")),
            EngineKind::Main => ("model.safetensors.index.json (the main engine has no single-file path)", has("model.safetensors.index.json")),
        };
        let files = vec![weights, ("tokenizer.json", has("tokenizer.json")), ("chat_template.jinja (run does not apply it)", has("chat_template.jinja"))];
        let tokenizer = match Tokenizer::from_path(dir) {
            Ok(t) => format!("loads ({} ids)", t.vocab_size()),
            Err(e) => format!("does not load: {}", e),
        };
        let optimizations = optimizations(engine, p, &root, qwen.as_ref(), conf);
        Ok(Report { dir: dir.to_string(), config_source: conf.source_str(), detected, quantization: quantization(&root), config_problems, files, tokenizer, optimizations })
    }

    pub fn text(&self) -> String {
        let mut s = format!("model      {}\nconfig     {}\nfamily     {} (model_type {})\nengine     {}\nquant      {}\ntokenizer  {}\n",
                            self.dir, self.config_source, self.detected.entry.name, self.detected.model_type, self.detected.entry.engine.name(), self.quantization, self.tokenizer);
        s += "files\n";
        for (n, ok) in &self.files { s += &format!("  {:8} {}\n", if *ok { "present" } else { "missing" }, n); }
        s += "engine config check\n";
        if self.config_problems.is_empty() { s += "  nothing refused\n"; }
        for p in &self.config_problems { s += &format!("  refused: {}\n", p); }
        s += "optimizations\n";
        for o in &self.optimizations {
            s += &format!("  {:34} {}{}\n", o.name, o.state.label(), if o.note.is_empty() { String::new() } else { format!("  [{}]", o.note) });
        }
        s += FOOTER;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Folder with only a config.json, and a config whose per-machine paths are not there.
    fn setup(tag: &str, cfg: &Value) -> (PathBuf, config::Loaded) {
        let d = std::env::temp_dir().join(format!("sd_model_layer_{}_{}", tag, std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), cfg.to_string()).unwrap();
        let mut c = config::Config::default();
        c.forheader.t1 = d.join("no_such.sdt1").display().to_string();
        (d, config::Loaded { cfg: c, source: None })
    }
    fn state(r: &Report, name: &str) -> String { r.optimizations.iter().find(|o| o.name == name).unwrap().state.label() }

    #[test]
    fn dense_report() {
        let cfg = json!({"model_type": "qwen3", "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 256, "eos_token_id": 1, "hidden_act": "silu",
                         "quantization": {"group_size": 64, "bits": 4}});
        let (d, conf) = setup("dense", &cfg);
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert_eq!(r.detected.entry.engine, EngineKind::QwenDense);
        assert_eq!(r.quantization, "4-bit, group 64");
        assert!(r.config_problems.is_empty(), "{:?}", r.config_problems);
        assert_eq!(state(&r, "MTP draft head"), "only in the other engine");
        assert_eq!(state(&r, "verify_fused"), "only in the other engine");
        assert_eq!(state(&r, "fast prompt read"), "available");
        assert!(!r.files[0].1);
        assert!(r.tokenizer.starts_with("does not load"));
        assert!(r.text().contains("does not prove a model runs"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn hybrid_report() {
        let cfg = json!({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text", "hidden_size": 512, "intermediate_size": 1024,
                         "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256, "vocab_size": 256,
                         "eos_token_id": 1, "linear_num_key_heads": 4, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
                         "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.25}});
        let (d, conf) = setup("hybrid", &cfg);
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert_eq!(r.detected.entry.engine, EngineKind::Main);
        assert_eq!(r.quantization, "none in config.json");
        assert_eq!(state(&r, "draft vocab"), "needs per-model file draft_vocab.keep, missing");
        assert_eq!(state(&r, "forheader score file"), "needs per-model file no_such.sdt1, missing");
        assert!(state(&r, "DFlash drafter").contains("dflash.json"));
        assert!(state(&r, "MTP draft head").starts_with("needs per-model file"));
        assert_eq!(state(&r, "verify_fused"), "off by default until tested");
        assert_eq!(state(&r, "ctx_draft"), "off by default until tested");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn rejected_folder_has_no_report() {
        let (d, conf) = setup("moe", &json!({"model_type": "qwen3_moe"}));
        assert!(Report::build(d.to_str().unwrap(), &conf).err().unwrap().contains("dense models only"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
