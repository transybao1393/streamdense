// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (quantized.h load_vector / qdot method: x pre-scaled by 1, 1/16, 1/256, 1/4096, unshifted nibble masks, scale*accum + sum*bias).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// gemv_q4.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// 4-bit g64 GEMV ported verbatim from decode7b.swift: gemv4 = ffnbench r2_d1_s0 tpt64 (the tuned gate kernel, also used for q/k/v/o), MODE 4 (SiLU, SwiGLU) added; gemv_k = k-token GEMV.
#include <metal_stdlib>
using namespace metal;

// ---- 4-bit GEMV: ffnbench gemv_gate_t<RPS 2, DEC 1, SB 0> (uint4 loads, in-place nibble decode), 64 threads per
// threadgroup. MODE 0: out = v (half); 1: out = act(g) * v (half, act = FATReLU); 2: outf = res + half(v); 3: outf = v
template <uint MODE>
kernel void gemv4(device const uint4* W [[buffer(0)]], device const half* S [[buffer(1)]],
                  device const half* B [[buffer(2)]], device const half* x [[buffer(3)]],
                  device half* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                  device const half* g [[buffer(7)]], device const float* res [[buffer(8)]], device float* outf [[buffer(9)]],
                  constant float& thr [[buffer(10)]],
                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    const uint RPS = 2;
    uint row0 = (tg * nsg + sg) * RPS;
    if (row0 >= R) return;
    uint C32 = C / 32, G = C / 64;
    float acc[RPS] = {0, 0};
    for (uint i = lane; i < C32; i += 32) {
        device const half4* xx = (device const half4*)(x + i * 32);
        float xv[32];
        float xs = 0;
        for (uint j = 0; j < 8; j++) {
            float4 v = float4(xx[j]);
            xs += v.x + v.y + v.z + v.w;
            float sc = (j & 1) ? 1.0f / 65536.0f : 1.0f;
            xv[4 * j] = v.x * sc; xv[4 * j + 1] = v.y * sc * 0.0625f;
            xv[4 * j + 2] = v.z * sc * 0.00390625f; xv[4 * j + 3] = v.w * sc * 0.000244140625f;
        }
        for (uint r = 0; r < RPS; r++) {
            uint row = row0 + r;
            if (row >= R) break;
            uint4 q4 = W[row * C32 + i];
            float xq = 0;
            for (uint w = 0; w < 4; w++) {
                uint q = q4[w];
                for (uint k = 0; k < 8; k++) xq += float(q & (15u << (4 * k))) * xv[8 * w + k];
            }
            acc[r] += float(S[row * G + i / 2]) * xq + float(B[row * G + i / 2]) * xs;
        }
    }
    for (uint r = 0; r < RPS; r++) {
        float v = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < R) {
            if (MODE == 0) out[row] = half(v);
            else if (MODE == 1) { float gv = float(g[row]); out[row] = half((gv > thr ? gv : 0.0f) * v); }
            else if (MODE == 2) outf[row] = res[row] + float(half(v));
            else if (MODE == 3) outf[row] = v;
            else { float gv = float(g[row]); out[row] = half(gv / (1.0f + exp(-gv)) * v); }   // MODE 4: SwiGLU silu(g) * up
        }
    }
}
#define G4(M_) template [[host_name("gemv4_m" #M_)]] kernel void gemv4<M_>(device const uint4*, device const half*, \
    device const half*, device const half*, device half*, constant uint&, constant uint&, device const half*, \
    device const float*, device float*, constant float&, uint, uint, uint, uint);
G4(0) G4(1) G4(2) G4(3) G4(4)

// ---- k-token GEMV (ffnbench SPEC=1), token-major x [K, C] / out [K, R]
#ifndef SD_NIB_SCALE
#define SD_NIB_SCALE
constant float4 NIB_SCALE = float4(1.0f, 0.0625f, 0.00390625f, 0.000244140625f);
#endif
template <uint K, uint RPS, bool OUTF>
kernel void gemv_k(device const uint4* W [[buffer(0)]], device const half* S [[buffer(1)]],
                   device const half* B [[buffer(2)]], device const half* x [[buffer(3)]],
                   device half* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                   device float* outf [[buffer(7)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint row0 = (tg * nsg + sg) * RPS;
    if (row0 >= R) return;
    uint C32 = C / 32, G = C / 64;
    float acc[RPS][K];
    for (uint r = 0; r < RPS; r++) for (uint t = 0; t < K; t++) acc[r][t] = 0;
    for (uint i = lane; i < C32; i += 32) {
        uint4 q[RPS];
        for (uint r = 0; r < RPS; r++) q[r] = W[min(row0 + r, R - 1) * C32 + i];
        float xq[RPS][K], xs[K];
        for (uint t = 0; t < K; t++) { xs[t] = 0; for (uint r = 0; r < RPS; r++) xq[r][t] = 0; }
        for (uint j = 0; j < 8; j++) {
            uint sh = (j & 1) * 16;
            float4 sv = NIB_SCALE * ((j & 1) ? 1.0f / 65536.0f : 1.0f);
            float4 wv[RPS];
            for (uint r = 0; r < RPS; r++) {
                uint qw = q[r][j >> 1];
                wv[r] = float4(float(qw & (15u << sh)), float(qw & (15u << (sh + 4))),
                               float(qw & (15u << (sh + 8))), float(qw & (15u << (sh + 12))));
            }
            for (uint t = 0; t < K; t++) {
                float4 v = float4(((device const half4*)(x + t * C + i * 32))[j]);
                xs[t] += v.x + v.y + v.z + v.w;
                v *= sv;
                for (uint r = 0; r < RPS; r++) xq[r][t] += dot(wv[r], v);
            }
        }
        for (uint r = 0; r < RPS; r++) {
            uint row = min(row0 + r, R - 1);
            float sc = float(S[row * G + i / 2]), bi = float(B[row * G + i / 2]);
            for (uint t = 0; t < K; t++) acc[r][t] += sc * xq[r][t] + bi * xs[t];
        }
    }
    for (uint r = 0; r < RPS; r++) for (uint t = 0; t < K; t++) {
        float v = simd_sum(acc[r][t]);
        if (lane == 0 && row0 + r < R) { if (OUTF) outf[t * R + row0 + r] = v; else out[t * R + row0 + r] = half(v); }
    }
}
#define GEMV_K(K_, R_, F_, N_) template [[host_name(N_ #K_ "_r" #R_)]] kernel void gemv_k<K_, R_, F_>(device const uint4*, device const half*, device const half*, device const half*, device half*, constant uint&, constant uint&, device float*, uint, uint, uint, uint);
GEMV_K(2, 2, false, "gemv_k") GEMV_K(3, 2, false, "gemv_k") GEMV_K(4, 2, false, "gemv_k")
GEMV_K(2, 2, true, "gemv_kf") GEMV_K(3, 2, true, "gemv_kf") GEMV_K(4, 2, true, "gemv_kf")
