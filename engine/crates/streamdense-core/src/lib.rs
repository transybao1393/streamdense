//! streamdense-core: model-independent memory / I/O / monitoring pieces (later shared with moeflux).
pub mod alloc;
pub mod arena;
pub mod budget;
pub mod config;
pub mod f16;
pub mod footprint;
pub mod history_index;
pub mod io;
pub mod library;
pub mod safetensors;

pub use f16::{bf16_to_f32, f32_to_bf16};
pub use safetensors::{write_safetensors, DType, SafeIndex, TensorInfo};

pub const GB: f64 = 1e9;
pub fn gb(b: u64) -> f64 { b as f64 / GB }
