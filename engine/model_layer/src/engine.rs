//! `Model`: load a model folder and run greedy token generation through the engine its family goes to.
//!
//! Both engines are greedy and take token ids, so one `generate` fits both. What differs is not hidden:
//! - dense-engine: `Engine::generate` (pipelined decode). No RAM budget before load; the footprint monitor is the only cap.
//! - main engine: `QwenEngine::feed_stream` after `reset`, the plain decode of `qwen serve` (budget checked in `open`).
//!   MTP, drafters, the 12 GB mode and the other options are not turned on here: they are the gateway's wiring.
//!
//! Prompts are token ids. The caller tokenizes (`streamdense_tokenizer::Tokenizer`); `chat_text` renders a family's
//! chat template and says how to encode the result.
//! Every family is listed in `family::FAMILIES`; a family with its own checks adds them to `check_text_run`.
//! Llama, Granite and Gemma run on dense-engine with the `DenseConfig` their folder builds (`dense_config`).
use crate::family::{detect, Detected, EngineKind, Family};
use dense_engine::config::DenseConfig;
use dense_engine::weights::SafeSource;
use dense_engine::{Engine, EngineOptions};
use serde_json::Value;
use std::path::Path;
use streamdense_core::{config, footprint};
use streamdense_model::{QwenEngine, QwenOptions};

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// KV cache length (tokens); prompt + generated must fit
    pub max_ctx: usize,
    /// abort above this phys_footprint (GB = 1e9 B)
    pub cap_gb: f64,
    /// abort when system swap grows by more than this since load (footprint::Monitor baseline)
    pub swap_gb: f64,
}

impl LoadOptions {
    /// `max_ctx` + the `[ram]` limits of streamdense.toml (`qwen serve` takes the same two).
    pub fn from_config(max_ctx: usize, ram: &config::Ram) -> LoadOptions { LoadOptions { max_ctx, cap_gb: ram.cap_gb, swap_gb: ram.swap_gb } }
}

/// Default KV length for a request: prompt + new + 8 rounded up to 256 (qd_run's rule).
pub fn max_ctx_for(prompt_tokens: usize, max_new: usize) -> usize { (prompt_tokens + max_new + 8).div_ceil(256) * 256 }

/// Result of `Model::generate`.
#[derive(Clone, Debug, Default)]
pub struct Generation {
    /// generated tokens, the stop token included if hit
    pub tokens: Vec<u32>,
    /// "eos" | "length" | "ctx_full"
    pub stop_reason: String,
    pub prefill_s: f64,
    pub decode_s: f64,
}

/// One per `Model`, never moved in a loop: boxing buys nothing (the engine workspace allows this lint too).
#[allow(clippy::large_enum_variant)]
enum Loaded {
    DenseEngine(Engine),
    Main(QwenEngine),
}

pub struct Model {
    pub detected: Detected,
    eng: Loaded,
    mon: footprint::Monitor,
}

/// Root object of `dir/config.json`.
pub fn read_config(dir: &str) -> Result<Value, String> {
    let path = format!("{}/config.json", dir);
    let txt = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path, e))?;
    serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path, e))
}

/// The dense-engine config of a folder, from its family's reader (Qwen: `DenseConfig::from_model_dir`, not validated
/// here; the other families validate). Err for a family that does not run on dense-engine.
pub fn dense_config(dir: &str, d: &Detected) -> Result<DenseConfig, String> {
    let root = || read_config(dir);
    match d.entry.family {
        Family::Qwen => crate::qwen::dense_config(dir, d),
        Family::Llama => crate::llama::dense_config(&root()?, Path::new(dir)),
        Family::Granite => crate::granite::dense_config(&root()?, Path::new(dir)),
        Family::Gemma => crate::gemma::dense_config(&root()?, Path::new(dir)),
    }
}

/// How a rendered chat string is tokenized: `Tokenizer::encode` when the template text already holds the BOS text
/// (Llama 3 `<|begin_of_text|>`, Gemma `<bos>`) or the model has no BOS (ChatML), as HF `apply_chat_template` does
/// (`add_special_tokens=False`).
pub struct ChatText {
    pub text: String,
    /// template name, for logs
    pub template: String,
}

/// Name of the chat template a folder's family renders, or why `--chat` is refused for it.
pub fn template_name(dir: &str, d: &Detected) -> Result<String, String> {
    let tpl = crate::llama::template::template_text(Path::new(dir))?;
    let has = |m: &str| tpl.as_deref().is_some_and(|t| t.contains(m));
    match d.entry.family {
        Family::Qwen => Ok(crate::qwen::template_kind(Path::new(dir))?.name().to_string()),
        Family::Llama => Ok(crate::llama::template_kind(Path::new(dir))?.name().to_string()),
        Family::Granite if has("<|im_start|>") => Ok("granite chatml (thinking on, template default)".into()),
        Family::Granite => Err("chat template: Granite template without <|im_start|> (older <|start_of_role|> form): not ported".into()),
        Family::Gemma if has("<start_of_turn>") => Ok("gemma turns".into()),
        Family::Gemma => Err("chat template: Gemma folder without a <start_of_turn> template (base model?): not ported".into()),
    }
}

/// Today's date as "%d %b %Y" (local time is not known without libc tz; UTC), what transformers' `strftime_now`
/// writes into the Llama 3.1 / 3.2 system block.
fn today_dmy() -> String {
    let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() / 86400).unwrap_or(0) as i64;
    // civil_from_days (H. Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    const M: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{:02} {} {}", day, M[(month - 1) as usize], year)
}

/// One user message in the folder's chat template, with the generation prompt. Encode `text` with
/// `Tokenizer::encode` (no post-processor BOS).
pub fn chat_text(dir: &str, d: &Detected, user: &str) -> Result<ChatText, String> {
    let template = template_name(dir, d)?;
    let msgs = vec![("user".to_string(), user.to_string())];
    let text = match d.entry.family {
        Family::Qwen => crate::qwen::chat_prompt(Path::new(dir), &msgs, true)?,
        Family::Llama => match crate::llama::template_kind(Path::new(dir))? {
            crate::llama::TemplateKind::Llama3Dated { knowledge, .. } =>
                crate::llama::TemplateKind::Llama3Dated { knowledge, date: today_dmy() }.render(&msgs, true)?,
            k => k.render(&msgs, true)?,
        },
        Family::Granite => crate::granite::chat_prompt(&msgs, true)?,
        Family::Gemma => crate::gemma::chat_prompt(&msgs, true)?,
    };
    Ok(ChatText { text, template })
}

/// What stops a text prompt from reaching this family's engine, before any weight is read. The family's own reasons
/// come first, so a family without a tokenizer is not reported as a tokenizer load error.
pub fn check_text_run(dir: &str, d: &Detected) -> Result<(), String> {
    match d.entry.family {
        Family::Qwen => Ok(()),
        Family::Llama | Family::Granite | Family::Gemma => dense_config(dir, d).map(|_| ()),
    }
}

impl Model {
    /// Detect the family, start the footprint monitor, load through the family's engine. Reads the weights.
    pub fn load(dir: &str, o: &LoadOptions) -> Result<Model, String> {
        let detected = detect(dir)?;
        let mon = footprint::Monitor::start((o.cap_gb * 1e9) as u64, (o.swap_gb * 1e9) as u64, 250);
        let dense = EngineOptions { max_ctx: o.max_ctx, ..Default::default() };
        let eng = match (detected.entry.family, detected.entry.engine) {
            (Family::Qwen, EngineKind::DenseEngine) => Loaded::DenseEngine(Engine::load(dir, dense)?),
            (Family::Llama | Family::Granite | Family::Gemma, EngineKind::DenseEngine) => {
                let cfg = dense_config(dir, &detected)?;
                Loaded::DenseEngine(Engine::from_source(cfg, &SafeSource::open(dir)?, dense)?)
            }
            (f, EngineKind::Main) if f != Family::Qwen => return Err(format!("{:?} family has no main-engine path", f)),
            (_, EngineKind::Main) => {
                // `cap` stays at its default like `qwen serve`; the live limit is the monitor above. no activity recorder: nothing reads it here
                let opt = QwenOptions { max_ctx: o.max_ctx, activity: false, live_cap: (o.cap_gb * 1e9) as u64, ..Default::default() };
                Loaded::Main(QwenEngine::open(dir, opt)?)
            }
        };
        Ok(Model { detected, eng, mon })
    }

    /// Greedy generation from `prompt` (state reset first): up to `max_new` tokens, stopping after an eos id of the
    /// model's config. `on_token` sees every token as it is produced.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, on_token: &mut dyn FnMut(u32)) -> Result<Generation, String> {
        if max_new == 0 { return Ok(Generation { stop_reason: "length".into(), ..Default::default() }); }
        match &mut self.eng {
            Loaded::DenseEngine(e) => {
                let stop = e.cfg.eos.clone();
                let st = e.generate(prompt, max_new, &stop, &mut *on_token)?;
                let eos = st.tokens.last().is_some_and(|t| stop.contains(t));
                Ok(Generation { stop_reason: if eos { "eos" } else { "length" }.into(), prefill_s: st.prefill_s, decode_s: st.decode_s, tokens: st.tokens })
            }
            Loaded::Main(e) => {
                e.reset();
                let mut tokens = Vec::with_capacity(max_new);
                let st = e.feed_stream(prompt, max_new, &[], &mut |t| { tokens.push(t); on_token(t); true })?;
                Ok(Generation { tokens, stop_reason: st.stop_reason.into(), prefill_s: st.prefill_s, decode_s: st.decode_s })
            }
        }
    }

    /// Peak phys_footprint seen by the monitor (bytes).
    pub fn peak_footprint(&self) -> u64 { self.mon.peak() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refusals of `check_text_run`, from a folder holding only a config.json: nothing loads.
    #[test]
    fn text_run_gives_the_family_reason() {
        let check = |tag: &str, cfg: &str| {
            let d = std::env::temp_dir().join(format!("sd_model_layer_run_{}_{}", tag, std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("config.json"), cfg).unwrap();
            let dir = d.to_str().unwrap();
            let r = check_text_run(dir, &detect(dir).unwrap());
            std::fs::remove_dir_all(&d).unwrap();
            r
        };
        let shape = r#""hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 256, "bos_token_id": 0, "eos_token_id": 1"#;
        assert!(check("qwen", &format!(r#"{{"model_type": "qwen3", {}}}"#, shape)).is_ok());
    }
}
