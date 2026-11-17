//! streamdense-model: config, weights, layers, KV cache, decode loop.
pub mod activity;
pub mod batch;
pub mod config;
pub mod ctx_draft;
pub mod draft_table;
pub mod draft_vocab;
pub mod ffn_select;
pub mod ffn_compact;
pub mod ffn_set;
pub mod ffn_stream;
pub mod gate_recheck;
pub mod dflash;
pub mod dflash2;
pub mod dflash2_select;
pub mod dump;
pub mod embed;
pub mod engine;
pub mod ffn;
pub mod decode_bench;
pub mod gemv_bench;
pub mod lut_bench;
pub mod qwen_config;
// ctrl() hands out `&mut SdCtrl` from `&self`: the shared CPU/GPU arena pattern (see streamdense-core arena.rs
// slice_mut). Scoped to this module only; TODO move onto the fn in qwen_engine.rs when that file is free.
#[allow(clippy::mut_from_ref)]
pub mod qwen_engine;
pub mod qwen_layers;
pub mod prefill_profile;
pub mod readmask;
pub mod resident;
pub mod rewind;
pub mod mtp;
pub mod snapshot;
pub mod spec;
pub mod spec_bench;
pub mod spec_deep;
pub mod unload;
pub mod verify_bench;
pub mod weights;
pub use engine::{Engine, GenResult, Options};
pub use qwen_config::{LayerKind, QwenConfig};
pub use qwen_engine::{DumpSpec, KvFormat, QwenEngine, QwenOptions, SeqSlot, TopK};
pub use weights::{AttnW, DnW, EmbedSrc, FfnW, LayerW, Mixer, MtpW, Q4b, Weights};
