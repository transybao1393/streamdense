//! `Model`: load a model folder and run greedy token generation through the engine its family goes to.
//!
//! Both engines are greedy and take token ids, so one `generate` fits both. What differs is not hidden:
//! - qwen-dense: `Engine::generate` (pipelined decode). No RAM budget before load; the footprint monitor is the only cap.
//! - main engine: `QwenEngine::feed_stream` after `reset`, the plain decode of `qwen serve` (budget checked in `open`).
//!   MTP, drafters, the 12 GB mode and the other options are not turned on here: they are the gateway's wiring.
//!
//! Prompts are raw token ids: no BOS, no chat template. The caller tokenizes (`streamdense_tokenizer::Tokenizer`).
//! Every family is listed in `family::FAMILIES`; a family with its own checks adds them to `check_text_run`.
use crate::family::{detect, Detected, EngineKind, Family};
use qwen_dense::{Engine, EngineOptions};
use streamdense_core::{config, footprint};
use streamdense_model::{QwenEngine, QwenOptions};

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// KV cache length (tokens); prompt + generated must fit
    pub max_ctx: usize,
    /// abort above this phys_footprint (GB = 1e9 B)
    pub cap_gb: f64,
    /// abort above this system swap
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
    QwenDense(Engine),
    Main(QwenEngine),
}

pub struct Model {
    pub detected: Detected,
    eng: Loaded,
    mon: footprint::Monitor,
}

/// What stops a text prompt from reaching this family's engine, before any weight is read. The family's own reasons
/// come first, so a family without a tokenizer is not reported as a tokenizer load error.
pub fn check_text_run(_dir: &str, d: &Detected) -> Result<(), String> {
    match d.entry.family {
        Family::Qwen => Ok(()),
    }
}

impl Model {
    /// Detect the family, start the footprint monitor, load through the family's engine. Reads the weights.
    pub fn load(dir: &str, o: &LoadOptions) -> Result<Model, String> {
        let detected = detect(dir)?;
        let mon = footprint::Monitor::start((o.cap_gb * 1e9) as u64, (o.swap_gb * 1e9) as u64, 250);
        let dense = EngineOptions { max_ctx: o.max_ctx, ..Default::default() };
        let eng = match (detected.entry.family, detected.entry.engine) {
            (Family::Qwen, EngineKind::QwenDense) => Loaded::QwenDense(Engine::load(dir, dense)?),
            (Family::Qwen, EngineKind::Main) => {
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
            Loaded::QwenDense(e) => {
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
