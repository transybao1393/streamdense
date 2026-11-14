//! What applies to a model folder, as facts from the existing code and the files present. Reads config.json, file
//! names, the tokenizer files, the small per-model files (keep-list, score file, drafter header) and the safetensors
//! headers; never the tensor data. The headers go through the chosen engine's own weight walk (`weight_check`), so a
//! dtype / shape the loader refuses is refused here too.
//! No speed or memory claim: `FOOTER` says so on every report.
//!
//! `optimizations` is the one list of existing optimizations: name, the engines that have it, the state there.
use crate::family::{detect, Detected, EngineKind, Family};
use dense_engine::config::DenseConfig;
use serde_json::Value;
use std::path::{Path, PathBuf};
use streamdense_core::budget::Budget;
use streamdense_core::{config, SafeIndex};
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
    /// non-Qwen families: ("generation", ..), ("chat", template name or refusal)
    pub family_lines: Vec<(&'static str, String)>,
    /// what the safetensors header check did (refusals go to `config_problems`), and the ignored vision tensors
    pub weights: String,
}

/// Tensors of a vision tower next to the language model (multimodal MLX folders): never looked up by either loader.
fn is_vision(name: &str) -> bool { name.starts_with("vision_tower.") || name.contains(".vision_tower.") }

/// The chosen engine's weight checks on the safetensors headers, nothing else read: main engine `weights::plan` (its
/// load walk without an arena), dense-engine `weights::layout` (offsets and copy jobs; `Engine::from_source` validates
/// the config first, so this does too). Returns the report line and the refusal, if any.
fn weight_check(dir: &Path, engine: EngineKind, qwen: Option<&QwenConfig>, dense: Option<&DenseConfig>) -> (String, Option<String>) {
    let d = dir.to_string_lossy();
    let opened: Result<(SafeIndex, Result<(), String>), String> = match engine {
        EngineKind::Main => {
            let Some(c) = qwen else { return ("not checked: config not accepted".into(), None) };
            if !dir.join("model.safetensors.index.json").exists() { return ("not checked: no safetensors index".into(), None); }
            SafeIndex::open_mlx_dir(&d).map(|idx| {
                let r = streamdense_model::weights::plan(&idx, c, false, &mut Budget::new(u64::MAX, 0)).map(|_| ());
                (idx, r)
            })
        }
        EngineKind::DenseEngine => {
            let Some(c) = dense else { return ("not checked: config not accepted".into(), None) };
            if !dir.join("model.safetensors.index.json").exists() && !dir.join("model.safetensors").exists() {
                return ("not checked: no safetensors".into(), None);
            }
            dense_engine::weights::SafeSource::open(&d).map(|src| {
                let r = c.validate().and_then(|_| dense_engine::weights::layout(c, &src).map(|_| ()));
                (src.idx, r)
            })
        }
    };
    let (idx, r) = match opened {
        Ok(x) => x,
        Err(e) => return ("headers not readable".into(), Some(format!("weights: {}", e))),
    };
    let vision = idx.names().filter(|n| is_vision(n)).count();
    let mut line = match &r {
        Ok(()) => format!("headers of {} tensors: dtypes / shapes accepted by the {} loader", idx.len(), engine.name()),
        Err(_) => format!("headers of {} tensors: refused by the {} loader", idx.len(), engine.name()),
    };
    if vision > 0 { line += &format!("; ignored (vision): {} vision_tower.* tensors", vision); }
    for (f, n) in &idx.skipped { line += &format!("; ignored: {} ({} tensors, file absent, no language_model.*)", f, n); }
    (line, r.err())
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
    use EngineKind::{DenseEngine, Main};
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
        ("fast prompt read", &[Main, DenseEngine], match engine {
            Main => on("QwenOptions.prefill_exact = false (default); streamed form only in the 12 GB mode ([forheader] fast)"),
            DenseEngine => on("PrefillMode::Gemm (default), prompts of >= 16 tokens"),
        }),
        ("pipelined decode", &[Main, DenseEngine], match engine {
            Main => on("QwenEngine::generate(.., pipe = true); this layer's run uses feed_stream, as qwen serve does"),
            DenseEngine => on("DecodeMode::Pipelined (default)"),
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
        Family::Qwen | Family::Llama | Family::Granite | Family::Gemma | Family::Mistral | Family::Phi3 | Family::Olmo2
        | Family::SmolLM3 | Family::Glm4 | Family::StableLm | Family::Cohere2 | Family::Starcoder2 => Ok(Report::build(dir, conf)?.text()),
    }
}

/// Generation of a family member, from the parsed engine config (or the config.json when it does not parse).
fn generation(family: Family, root: &Value, c: Option<&DenseConfig>) -> String {
    use dense_engine::config::{Family as F, RopeScaling};
    match (family, c) {
        (Family::Qwen, _) => String::new(),
        (Family::Gemma, Some(c)) => match c.family { F::Gemma => "Gemma 1".into(), F::Gemma2 => "Gemma 2".into(), F::Gemma3 => "Gemma 3 (text)".into(), f => format!("{:?}", f) },
        (Family::Llama, Some(c)) => match c.rope_scale {
            RopeScaling::Llama3 { .. } => "Llama 3.1 / 3.2 (llama3 RoPE scaling)".into(),
            _ => format!("Llama-architecture (plain RoPE theta {}, head dim {})", c.rope_theta, c.head_dim),
        },
        (Family::Granite, Some(_)) => "Granite (dense, attention only)".into(),
        (f, Some(c)) => format!("{:?} (dense-engine {:?} layers)", f, c.family),
        (_, None) => format!("not determined (config not accepted), model_type {}", root["model_type"].as_str().unwrap_or("?")),
    }
}

impl Report {
    pub fn build(dir: &str, conf: &config::Loaded) -> Result<Report, String> {
        let detected = detect(dir)?;
        let p = Path::new(dir);
        let path = p.join("config.json");
        let txt = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let root: Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path.display(), e))?;
        let engine = detected.entry.engine;
        let (mut config_problems, mut qwen, mut dense) = (vec![], None, None);
        let family = detected.entry.family;
        if family != Family::Qwen {
            config_problems = match family {
                Family::Llama => crate::llama::blockers(&root),
                Family::Granite => crate::granite::blockers(&root),
                Family::Mistral => crate::mistral::blockers(&root),
                Family::Phi3 => crate::phi3::blockers(&root),
                Family::Olmo2 => crate::olmo2::blockers(&root),
                Family::SmolLM3 => crate::smollm3::blockers(&root),
                Family::Glm4 => crate::glm4::blockers(&root),
                Family::StableLm => crate::stablelm::blockers(&root),
                Family::Cohere2 => crate::cohere2::blockers(&root),
                Family::Starcoder2 => crate::starcoder2::blockers(&root),
                Family::Gemma => crate::gemma::blockers(&root),
                Family::Qwen => vec![],
            };
        }
        match engine {
            EngineKind::DenseEngine => match crate::engine::dense_config(dir, &detected) {
                Ok(c) => {
                    if family == Family::Qwen { config_problems = crate::qwen::blockers(&c); }
                    if c.quant.is_none() && !config_problems.iter().any(|p| p.contains("quantization")) {
                        config_problems.push("no quantization block: the weight loader wants an MLX 4-bit model".into());
                    }
                    dense = Some(c);
                }
                Err(e) => if !config_problems.contains(&e) { config_problems.push(e) },
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
            EngineKind::DenseEngine => ("model.safetensors.index.json or model.safetensors", has("model.safetensors.index.json") || has("model.safetensors")),
            EngineKind::Main => ("model.safetensors.index.json (the main engine has no single-file path)", has("model.safetensors.index.json")),
        };
        let tpl_file = "chat_template.jinja (else tokenizer_config.json chat_template; run --chat renders it)";
        let files = vec![weights, ("tokenizer.json", has("tokenizer.json")), (tpl_file, has("chat_template.jinja"))];
        let family_lines = if family == Family::Qwen { vec![] } else {
            let tpl = crate::engine::template_name(dir, &detected).unwrap_or_else(|e| format!("refused: {}", e));
            vec![("generation", generation(family, &root, dense.as_ref())), ("chat", tpl)]
        };
        let tokenizer = match Tokenizer::from_path(dir) {
            Ok(t) => format!("loads ({} ids)", t.vocab_size()),
            Err(e) => format!("does not load: {}", e),
        };
        // weights: only once the config is accepted (a refused config already blocks the load)
        let weights = if config_problems.is_empty() {
            let (line, refused) = weight_check(p, engine, qwen.as_ref(), dense.as_ref());
            config_problems.extend(refused);
            line
        } else { "not checked: config refused".into() };
        let optimizations = optimizations(engine, p, &root, qwen.as_ref(), conf);
        Ok(Report { dir: dir.to_string(), config_source: conf.source_str(), detected, quantization: quantization(&root), config_problems, files, tokenizer, optimizations, family_lines, weights })
    }

    pub fn text(&self) -> String {
        let mut s = format!("model      {}\nconfig     {}\nfamily     {} (model_type {})\nengine     {}\nquant      {}\ntokenizer  {}\n",
                            self.dir, self.config_source, self.detected.entry.name, self.detected.model_type, self.detected.entry.engine.name(), self.quantization, self.tokenizer);
        for (k, v) in &self.family_lines { s += &format!("{:10} {}\n", k, v); }
        s += "files\n";
        for (n, ok) in &self.files { s += &format!("  {:8} {}\n", if *ok { "present" } else { "missing" }, n); }
        s += &format!("weights    {}\n", self.weights);
        s += "engine config check\n";
        if self.config_problems.is_empty() { s += "  nothing refused\n"; }
        if !self.config_problems.is_empty() && self.detected.entry.family != Family::Qwen { s += "  blockers (the model does not run until these are gone):\n"; }
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
        assert_eq!(r.detected.entry.engine, EngineKind::DenseEngine);
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

    /// Header-only safetensors: real header, data region left as a sparse hole (`set_len`), so tests stay tiny on disk.
    fn write_headers(path: &Path, tensors: &[(String, &str, Vec<usize>)]) {
        use std::io::Write;
        let size = |dt: &str| match dt { "U32" | "F32" => 4, "BF16" | "F16" => 2, _ => 1 };
        let mut hdr = serde_json::Map::new();
        let mut off = 0usize;
        for (n, dt, shape) in tensors {
            let len = shape.iter().product::<usize>() * size(dt);
            hdr.insert(n.clone(), json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + len]}));
            off += len;
        }
        let mut h = Value::Object(hdr).to_string().into_bytes();
        while !h.len().is_multiple_of(8) { h.push(b' '); }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.set_len((8 + h.len() + off) as u64).unwrap();
    }
    /// `{p}.weight` U32 [r, c/8] + scales / biases BF16 [r, c/64] (MLX 4-bit, group 64).
    fn q4(out: &mut Vec<(String, &'static str, Vec<usize>)>, p: &str, r: usize, c: usize) {
        out.push((format!("{}.weight", p), "U32", vec![r, c / 8]));
        out.push((format!("{}.scales", p), "BF16", vec![r, c / 64]));
        out.push((format!("{}.biases", p), "BF16", vec![r, c / 64]));
    }
    fn hybrid_cfg() -> Value {
        json!({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text", "hidden_size": 512, "intermediate_size": 1024,
               "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256, "vocab_size": 256,
               "eos_token_id": 1, "linear_num_key_heads": 4, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
               "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.25},
               "quantization": {"group_size": 64, "bits": 4}})
    }
    /// Every tensor the main engine loads for `c` (MTP off), `A_log` in `a_log_dtype`, plus 2 vision_tower tensors.
    fn hybrid_tensors(c: &QwenConfig, a_log_dtype: &'static str) -> Vec<(String, &'static str, Vec<usize>)> {
        use streamdense_model::qwen_config::LayerKind;
        let (h, mut t) = (c.hidden, vec![]);
        for (l, k) in c.kinds.iter().enumerate() {
            let p = format!("language_model.model.layers.{}.", l);
            t.push((format!("{}input_layernorm.weight", p), "BF16", vec![h]));
            t.push((format!("{}post_attention_layernorm.weight", p), "BF16", vec![h]));
            match k {
                LayerKind::DeltaNet => {
                    let a = format!("{}linear_attn.", p);
                    for (n, r) in [("in_proj_qkv", c.conv_dim()), ("in_proj_z", c.dn_v_dim()), ("in_proj_b", c.dn_vh), ("in_proj_a", c.dn_vh)] {
                        q4(&mut t, &format!("{}{}", a, n), r, h);
                    }
                    q4(&mut t, &format!("{}out_proj", a), h, c.dn_v_dim());
                    t.push((format!("{}conv1d.weight", a), "BF16", vec![c.conv_dim(), c.conv_k, 1]));
                    t.push((format!("{}A_log", a), a_log_dtype, vec![c.dn_vh]));
                    t.push((format!("{}dt_bias", a), "BF16", vec![c.dn_vh]));
                    t.push((format!("{}norm.weight", a), "F32", vec![c.dn_dv]));
                }
                LayerKind::Attention => {
                    let a = format!("{}self_attn.", p);
                    for (n, r) in [("q_proj", 2 * c.q_dim()), ("k_proj", c.kv_dim()), ("v_proj", c.kv_dim())] { q4(&mut t, &format!("{}{}", a, n), r, h); }
                    q4(&mut t, &format!("{}o_proj", a), h, c.q_dim());
                    t.push((format!("{}q_norm.weight", a), "BF16", vec![c.head_dim]));
                    t.push((format!("{}k_norm.weight", a), "BF16", vec![c.head_dim]));
                }
            }
            for (n, r, cols) in [("gate_proj", c.inter, h), ("up_proj", c.inter, h), ("down_proj", h, c.inter)] { q4(&mut t, &format!("{}mlp.{}", p, n), r, cols); }
        }
        t.push(("language_model.model.norm.weight".into(), "BF16", vec![h]));
        q4(&mut t, "language_model.lm_head", c.vocab, h);
        q4(&mut t, "language_model.model.embed_tokens", c.vocab, h);
        t.push(("vision_tower.blocks.0.attn.qkv.weight".into(), "F32", vec![8, 8]));
        t.push(("vision_tower.merger.mlp.0.bias".into(), "BF16", vec![8]));
        t
    }
    fn write_index(d: &Path, tensors: &[(String, &str, Vec<usize>)]) {
        let wm: serde_json::Map<String, Value> = tensors.iter().map(|t| (t.0.clone(), json!("model.safetensors"))).collect();
        std::fs::write(d.join("model.safetensors.index.json"), json!({"weight_map": wm}).to_string()).unwrap();
        write_headers(&d.join("model.safetensors"), tensors);
    }

    /// the main engine's header walk runs in inspect: F32 `A_log` (small Qwen3.5) accepted, vision tower
    /// reported as ignored, a dtype the loader refuses (F16 `A_log`) refused by name.
    #[test]
    fn hybrid_weight_headers() {
        let (d, conf) = setup("hybrid_w", &hybrid_cfg());
        let c = QwenConfig::from_model_dir(d.to_str().unwrap()).unwrap();
        write_index(&d, &hybrid_tensors(&c, "F32"));
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert!(r.config_problems.is_empty(), "{:?}", r.config_problems);
        assert!(r.weights.contains("accepted") && r.weights.contains("ignored (vision): 2 vision_tower.* tensors"), "{}", r.weights);
        assert!(r.text().contains("nothing refused"));
        write_index(&d, &hybrid_tensors(&c, "F16"));
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert_eq!(r.config_problems.len(), 1, "{:?}", r.config_problems);
        assert!(r.config_problems[0].contains("A_log is F16"), "{}", r.config_problems[0]);
        assert!(r.text().contains("refused: weights:"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// tied hybrid (small Qwen3.5: `tie_word_embeddings` true, no lm_head tensor) is accepted by the header walk;
    /// the same folder untied is refused by the lm_head name.
    #[test]
    fn hybrid_tied_head() {
        let mut cfg = hybrid_cfg();
        cfg["text_config"]["tie_word_embeddings"] = json!(true);
        let (d, conf) = setup("hybrid_tied", &cfg);
        let c = QwenConfig::from_model_dir(d.to_str().unwrap()).unwrap();
        assert!(c.tie_word_embeddings);
        let t: Vec<_> = hybrid_tensors(&c, "F32").into_iter().filter(|t| !t.0.starts_with("language_model.lm_head.")).collect();
        write_index(&d, &t);
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert!(r.config_problems.is_empty(), "{:?}", r.config_problems);
        assert!(r.weights.contains("accepted"), "{}", r.weights);
        cfg["text_config"]["tie_word_embeddings"] = json!(false);
        std::fs::write(d.join("config.json"), cfg.to_string()).unwrap();
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert_eq!(r.config_problems.len(), 1, "{:?}", r.config_problems);
        assert!(r.config_problems[0].contains("language_model.lm_head.weight not found"), "{}", r.config_problems[0]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// dense-engine folders go through `dense_engine::weights::layout` on the headers.
    #[test]
    fn dense_weight_headers() {
        let cfg = json!({"model_type": "qwen3", "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 256, "eos_token_id": 1, "hidden_act": "silu",
                         "tie_word_embeddings": true, "quantization": {"group_size": 64, "bits": 4}});
        let (d, conf) = setup("dense_w", &cfg);
        let c = DenseConfig::from_model_dir(d.to_str().unwrap()).unwrap();
        let tensors = |qdt: &'static str| {
            let (h, mut t) = (c.hidden, vec![]);
            q4(&mut t, "model.embed_tokens", c.vocab, h);
            t.push(("model.norm.weight".into(), "BF16", vec![h]));
            for l in 0..c.layers {
                let p = format!("model.layers.{}", l);
                t.push((format!("{}.input_layernorm.weight", p), "F16", vec![h]));
                t.push((format!("{}.post_attention_layernorm.weight", p), "BF16", vec![h]));
                for (n, r) in [("q_proj", c.q_dim()), ("k_proj", c.kv_dim()), ("v_proj", c.kv_dim())] { q4(&mut t, &format!("{}.self_attn.{}", p, n), r, h); }
                q4(&mut t, &format!("{}.self_attn.o_proj", p), h, c.q_dim());
                if c.qk_norm {
                    t.push((format!("{}.self_attn.q_norm.weight", p), "BF16", vec![c.head_dim]));
                    t.push((format!("{}.self_attn.k_norm.weight", p), "BF16", vec![c.head_dim]));
                }
                for (n, r, cols) in [("gate_proj", c.inter, h), ("up_proj", c.inter, h), ("down_proj", h, c.inter)] { q4(&mut t, &format!("{}.mlp.{}", p, n), r, cols); }
            }
            let i = t.iter().position(|x| x.0 == "model.layers.1.mlp.up_proj.weight").unwrap();
            t[i].1 = qdt;
            t
        };
        write_headers(&d.join("model.safetensors"), &tensors("U32"));
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert!(r.config_problems.is_empty(), "{:?}", r.config_problems);
        assert!(r.weights.contains("accepted by the dense-engine loader") && !r.weights.contains("vision"), "{}", r.weights);
        write_headers(&d.join("model.safetensors"), &tensors("F32"));
        let r = Report::build(d.to_str().unwrap(), &conf).unwrap();
        assert!(r.config_problems.len() == 1 && r.config_problems[0].contains("up_proj.weight is F32"), "{:?}", r.config_problems);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn rejected_folder_has_no_report() {
        let (d, conf) = setup("moe", &json!({"model_type": "qwen3_moe"}));
        assert!(Report::build(d.to_str().unwrap(), &conf).err().unwrap().contains("dense models only"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
