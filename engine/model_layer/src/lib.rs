//! streamdense-model-layer: one entry point in front of the existing engines. Calls them; holds no inference code.
//!
//! - `family`: config.json -> family + engine (`FAMILIES`, the only place families are listed); rejects MoE / unknown.
//! - `engine`: `Model` (load a model folder, greedy token generation) over `dense_engine::Engine` / `QwenEngine`.
//! - `qwen`, `llama`, `granite`, `gemma`: per-family config.json -> `DenseConfig` mapping, chat template, refusals.
//! - `report`: what applies to a model folder (family, engine, quantization, state of every existing optimization).
pub mod engine;
pub mod family;
pub mod gemma;
pub mod granite;
pub mod llama;
pub mod qwen;
pub mod report;

pub use engine::{max_ctx_for, Generation, LoadOptions, Model};
pub use family::{detect, detect_json, Detected, EngineKind, Family, FamilyEntry, Shape, FAMILIES};
pub use report::{OptLine, Report, State};
