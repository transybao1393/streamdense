// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (quantized.h load_vector / qdot method, applied to gathered rows).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// sparse_up.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// Sparse up (gather): decode7b gemv_gather_act (k = 1) and gemv_gather_k (k-token), rows by neuron id.
#include <metal_stdlib>
using namespace metal;

#ifndef SD_NIB_SCALE
#define SD_NIB_SCALE
constant float4 NIB_SCALE = float4(1.0f, 0.0625f, 0.00390625f, 0.000244140625f);
#endif
inline float dot_row(device const uint* W, device const half* S, device const half* B,
                     device const half* x, uint row, uint C, uint lane) {
    uint C8 = C / 8, G = C / 64;
    device const uint* w = W + row * C8;
    device const half* s = S + row * G;
    device const half* b = B + row * G;
    float acc = 0;
    #pragma unroll(8)
    for (uint i = lane; i < C8; i += 32) {
        uint q = w[i];
        device const half* xx = x + i * 8;
        float xq = 0, xs = 0;
        for (uint k = 0; k < 8; k++) {
            float xv = xx[k];
            xq += float((q >> (4 * k)) & 15u) * xv;
            xs += xv;
        }
        acc += float(s[i / 8]) * xq + float(b[i / 8]) * xs;
    }
    return simd_sum(acc);
}
kernel void gemv_gather_act(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                            device const half* B [[buffer(2)]], device const half* x [[buffer(3)]],
                            device const uint* idx [[buffer(4)]], device const uint* count [[buffer(5)]],
                            device const half* g [[buffer(6)]], device half* out [[buffer(7)]],
                            constant uint& C [[buffer(8)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint t = tg * 8 + sg;
    if (t >= count[0]) return;
    uint row = idx[t];
    float v = dot_row(W, S, B, x, row, C, lane);
    if (lane == 0) out[t] = half(max(float(g[row]), 0.0f) * v);
}
template <uint K>
kernel void gemv_gather_k(device const uint4* W [[buffer(0)]], device const half* S [[buffer(1)]],
                          device const half* B [[buffer(2)]], device const half* x [[buffer(3)]],
                          device const uint* idx [[buffer(4)]], device const uint* count [[buffer(5)]],
                          device const half* g [[buffer(6)]], device const uchar* am [[buffer(7)]],
                          device half* out [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& R [[buffer(10)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint t = tg * 8 + sg;
    if (t >= count[0]) return;
    uint row = idx[t];
    uint C32 = C / 32, G = C / 64;
    float acc[K];
    for (uint k = 0; k < K; k++) acc[k] = 0;
    for (uint i = lane; i < C32; i += 32) {
        uint4 q = W[row * C32 + i];
        float xq[K], xs[K];
        for (uint k = 0; k < K; k++) { xq[k] = 0; xs[k] = 0; }
        for (uint j = 0; j < 8; j++) {
            uint sh = (j & 1) * 16, qw = q[j >> 1];
            float4 sv = NIB_SCALE * ((j & 1) ? 1.0f / 65536.0f : 1.0f);
            float4 wv = float4(float(qw & (15u << sh)), float(qw & (15u << (sh + 4))),
                               float(qw & (15u << (sh + 8))), float(qw & (15u << (sh + 12))));
            for (uint k = 0; k < K; k++) {
                float4 v = float4(((device const half4*)(x + k * C + i * 32))[j]);
                xs[k] += v.x + v.y + v.z + v.w;
                xq[k] += dot(wv, v * sv);
            }
        }
        float sc = float(S[row * G + i / 2]), bi = float(B[row * G + i / 2]);
        for (uint k = 0; k < K; k++) acc[k] += sc * xq[k] + bi * xs[k];
    }
    uint bits = am[t];
    for (uint k = 0; k < K; k++) {
        float v = simd_sum(acc[k]);
        if (lane == 0) out[k * R + t] = ((bits >> k) & 1u) ? half(max(float(g[k * R + row]), 0.0f) * v) : half(0);
    }
}
#define KTOKU(K_) template [[host_name("gemv_gather_k" #K_)]] kernel void gemv_gather_k<K_>(device const uint4*, device const half*, device const half*, device const half*, device const uint*, device const uint*, device const half*, device const uchar*, device half*, constant uint&, constant uint&, uint, uint, uint);
KTOKU(2) KTOKU(3) KTOKU(4)
