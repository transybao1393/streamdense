//! dense-engine: a minimal engine for ordinary dense decoder models (Qwen2 / 2.5 / 3 parsed here; Llama, Granite, Gemma
//! 1 / 2 / 3-text, Mistral, Phi-3, OLMo 2, SmolLM3, GLM-4, StableLM, Cohere2, StarCoder2 through config fields: head dim
//! 64 / 96 / 128 / 160 / 256, GQA or MHA, MLX 4-bit weights) on the streamdense
//! Metal kernels. Not wired into the engine workspace (see README.md).
//! - `config`: config.json reader + validator (`DenseConfig`).
//! - `gpu`: a minimal Metal harness (device, library from sources, pipelines, one arena buffer, encoder).
//! - `weights`: MLX 4-bit safetensors loader (fused q|k|v, f16 -> bf16, tied embeddings).
//! - `model`: kernel dispatch of one K-token pass (embed, layers, head) and the attention impl switch.
//! - `engine`: `Engine` (load, prefill, decode step, greedy generate) and the tokenizer re-export.
//! - `prefill`: GEMM prefill for long prompts (q4b_gemm projections / FFN + attn128_flash over up to t_max rows).
pub mod config;
pub mod engine;
pub mod gpu;
pub mod model;
pub mod prefill;
pub mod weights;

pub use engine::{Engine, EngineOptions, GenStats};
pub use streamdense_tokenizer::{StreamDecoder, Tokenizer};

/// Library source for the attention kernels (head dim 64 / 96 / 128 / 160 / 256): the engine's common_bf16.metal (bf, to_bf, SdCtrl; read,
/// not modified) followed by shaders/attn_hd128.metal (split-K, prep), attn_hd128_flash.metal (prefill flash) and
/// attn_hd128_sdpa.metal (MLX decode mirror).
pub const SHADER_SOURCE: [&str; 4] = [
    include_str!("../../shaders/common_bf16.metal"),
    include_str!("../shaders/attn_hd128.metal"),
    include_str!("../shaders/attn_hd128_flash.metal"),
    include_str!("../shaders/attn_hd128_sdpa.metal"),
];

/// Library source of the whole engine: `SHADER_SOURCE` + the engine's MMA 4-bit matmuls (gemv_mma_q4.metal),
/// RMSNorm (norm_bf16.metal), greedy head (head_sample.metal: argmax_b), the prefill GEMM (gemm_q4.metal) - all
/// read-only via include_str! - and shaders/qd_misc.metal (4-bit embedding gather, q/k/v bias add).
pub const ENGINE_SHADER_SOURCE: [&str; 9] = [
    include_str!("../../shaders/common_bf16.metal"),
    include_str!("../shaders/attn_hd128.metal"),
    include_str!("../shaders/attn_hd128_flash.metal"),
    include_str!("../shaders/attn_hd128_sdpa.metal"),
    include_str!("../../shaders/gemv_mma_q4.metal"),
    include_str!("../../shaders/norm_bf16.metal"),
    include_str!("../../shaders/head_sample.metal"),
    include_str!("../shaders/qd_misc.metal"),
    include_str!("../../shaders/gemm_q4.metal"),
];
