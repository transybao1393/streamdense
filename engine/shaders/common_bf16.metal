// common_bf16.metal - streamdense shared MSL for the Qwen3.8-27B engine.
// Must be the FIRST file of the concatenated Qwen shader source (Gpu::from_shader_files / QWEN_SHADERS).
// bf16 is carried as ushort; math is fp32. SdCtrl mirrors the Rust #[repr(C)] struct (plan 7.1).
#include <metal_stdlib>
using namespace metal;

// bf16 -> f32 (exact)
static inline float bf(ushort h) { return as_type<float>(uint(h) << 16); }

// f32 -> bf16, round to nearest even; NaN kept as a (quiet) NaN with its sign.
static inline ushort to_bf(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}

// GPU control block (72 B). Only GPU kernels advance it; the CPU writes at reset / test override.
struct SdCtrl {
    uint pos;       // position of t1 (first row of the next pass)
    uint role;      // which DeltaNet/conv state copy is current (0/1)
    uint mtp_len;   // MTP KV length (= positions fed to MTP)
    uint mtp_rows;  // rows fed to MTP this step (1 or 2)
    uint n_emit;
    uint emit[2];
    uint draft;     // MTP draft d
    uint tok[2];    // argmax of the pass rows
    float top1[2];
    float top2[2];
    uint stop;
    uint step;
    uint pad[2];
};
