// head_q8.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// lm_head: decode7b gemv8_k (8-bit g64), gemv_f16 (fp16 head), argmax + top-2 record.
#include <metal_stdlib>
using namespace metal;

// ---- lm_head 8-bit (LMHEAD=q8): affine groups of 64 along the input, w = sc * q + bias (fp16 sc / bias), K tokens
// (x [K, C], out [K, R] fp32); one row per simdgroup, 8 simdgroups per threadgroup, one 16-byte load per lane
template <uint K>
kernel void gemv8_k(device const uint4* W [[buffer(0)]], device const half* S [[buffer(1)]], device const half* B [[buffer(2)]],
                    device const half* x [[buffer(3)]], device float* out [[buffer(4)]], constant uint& R [[buffer(5)]],
                    constant uint& C [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint row = tg * nsg + sg;
    if (row >= R) return;
    uint C16 = C / 16, G = C / 64;
    float acc[K];
    for (uint t = 0; t < K; t++) acc[t] = 0;
    device const uint4* w = W + row * C16;
    for (uint i = lane; i < C16; i += 32) {
        uint4 q = w[i];
        float sc = float(S[row * G + i / 4]), bi = float(B[row * G + i / 4]);
        for (uint t = 0; t < K; t++) {
            device const half4* xx = (device const half4*)(x + t * C + i * 16);
            float d = 0, xs = 0;
            for (uint u = 0; u < 4; u++) {
                float4 xv = float4(xx[u]);
                d += dot(float4(as_type<uchar4>(q[u])), xv);
                xs += xv.x + xv.y + xv.z + xv.w;
            }
            acc[t] += sc * d + bi * xs;
        }
    }
    for (uint t = 0; t < K; t++) {
        float v = simd_sum(acc[t]);
        if (lane == 0) out[t * R + row] = v;
    }
}
#define G8(K_) template [[host_name("gemv8_k" #K_)]] kernel void gemv8_k<K_>(device const uint4*, device const half*, device const half*, \
    device const half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
G8(1) G8(2) G8(3) G8(4)

// ---- lm_head fp16, K tokens (x [K, C], out [K, R] fp32); one row per simdgroup, 8 simdgroups per threadgroup
template <uint K>
kernel void gemv_f16(device const half4* W [[buffer(0)]], device const half4* x [[buffer(1)]], device float* out [[buffer(2)]],
                     constant uint& R [[buffer(3)]], constant uint& C [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint row = tg * nsg + sg;
    if (row >= R) return;
    uint C4 = C / 4;
    float acc[K];
    for (uint t = 0; t < K; t++) acc[t] = 0;
    device const half4* w = W + row * C4;
    for (uint i = lane; i < C4; i += 32) {
        float4 wv = float4(w[i]);
        for (uint t = 0; t < K; t++) acc[t] += dot(wv, float4(x[t * C4 + i]));
    }
    for (uint t = 0; t < K; t++) {
        float v = simd_sum(acc[t]);
        if (lane == 0) out[t * R + row] = v;
    }
}
#define GF(K_) template [[host_name("gemv_f16_k" #K_)]] kernel void gemv_f16<K_>(device const half4*, device const half4*, \
    device float*, constant uint&, constant uint&, uint, uint, uint, uint);
GF(1) GF(2) GF(3) GF(4)

// ---- greedy argmax + top-2 per token row (lowest index on ties); 1024 threads per threadgroup, threadgroup = row t.
// out[t] = token; if wtok != 0 also tokbuf[pos + 1 + t] = token when pos + 1 + t >= plen (prompt tokens kept).
// rec[pos + t] = (token bits, top-1 logit, top-2 logit, 0): read by the CPU after the pass, so passes can be pipelined.
kernel void argmax(device const float* logits [[buffer(0)]], device uint* out [[buffer(1)]], device uint* tokbuf [[buffer(2)]],
                   constant uint& V [[buffer(3)]], constant uint& pos [[buffer(4)]], constant uint& plen [[buffer(5)]],
                   constant uint& wtok [[buffer(6)]], device float4* rec [[buffer(7)]],
                   uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32], bs[32];
    threadgroup uint bi[32];
    device const float* lg = logits + t * V;
    float best = -INFINITY, sec = -INFINITY;
    uint bidx = 0xffffffffu;
    for (uint i = tid; i < V; i += 1024) {
        float v = lg[i];
        if (v > best) { sec = best; best = v; bidx = i; } else if (v > sec) sec = v;
    }
    float m = simd_max(best);
    uint mi = simd_min(best == m ? bidx : 0xffffffffu);
    float s2 = simd_max(bidx == mi ? sec : best);           // every lane but the winner offers its best
    if (lane == 0) { bv[sg] = m; bi[sg] = mi; bs[sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = bv[lane], sv = bs[lane];
        uint ix = bi[lane];
        float m2 = simd_max(v);
        uint r = simd_min(v == m2 ? ix : 0xffffffffu);
        float sec2 = simd_max(ix == r ? sv : v);
        if (lane == 0) {
            out[t] = r;
            if (wtok != 0 && pos + 1 + t >= plen) tokbuf[pos + 1 + t] = r;
            rec[pos + t] = float4(as_type<float>(r), m2, sec2, 0.0f);
        }
    }
}
