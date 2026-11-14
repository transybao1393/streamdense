// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (quantized.h qmv_fast / qmv, load_vector, qdot: per-lane value set, fp32 expression forms, simd_sum tree).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// gemv_q4_bf16.metal - streamdense Qwen3.8-27B 4-bit GEMV on the MLX affine layout (plan sections 1, 4, 5).
// Needs common_bf16.metal first (bf, to_bf). Adapted from gemv_q4.metal gemv_k (the 7B k-token GEMV, 226-240 GB/s);
// the per-lane split and fp32 order are MLX's qmv_fast / qmv (see q4b_dot_v).
// Layout: W u32 [R][C/8] (8 nibbles per u32, low nibble first) read as uint4 [R][C/32]; scales / biases bf16 [R][C/64];
// w = s*q + b; x [K][C] and out / res [K][R] are bf16 as ushort. Math is fp32.
// K-invariance (hard rule): every K and every D (row-dots per simdgroup) comes from the same q4b_dot template with the
// same lane mapping and the same per-(row, token) expression sequence (no cross-token or cross-row sums), so row t of
// a K-token dispatch equals a K = 1 one bit for bit (proof sketch at q4b_dot_v; checked by `streamdense bench gemv-k`).
//
// Why the old K-token loop scaled badly (K=4 3.7x, K=8 18x the K=1 pass): it built
// xt[K][V] (K x 16 pre-scaled x floats) for ALL tokens before touching the weights, so the live set was ~16K + 2K
// acc + pointers: ~60 regs at K=2, ~100 at K=4 (occupancy collapse, no latency hiding for the weight stream), ~170 at
// K=8 (spill). And 2 rows per simdgroup amortised the x side (bf convert, 2^-4j scale, block sum: ~3 ops per x value
// per token) over only 2 rows, i.e. ~1.5 ops per weight per token on top of the ~1.25-2 of the dot itself.
// Now: the loop is word-major (i = 16-bit word of the lane block); per word each row's 4 nibbles are converted once
// (wf, shared by all K tokens), then each token's 4 x values are loaded, scaled and summed once (shared by all D rows)
// and FMA'd into every row. Live set ~ 2D (words) + 4D (wf) + 2DK (accum, acc) + K + 4, ~64 regs at K=4, D=4.

// bf16 x4 -> f32 x4 (exact)
static inline float4 bf4(ushort4 h) { return as_type<float4>(uint4(h) << 16); }
// round f32 to bf16 and back (an MLX bf16 store point)
static inline float rbf(float f) { return bf(to_bf(f)); }

// acc[n][t] = sum_c w_n[c] * x[t][c] for this lane's share (no simd_sum yet). NR rows, each with its own W/S/B base.
// Mirrors MLX's M = 1 quantized_matmul (quantized.h qmv_fast_impl / qmv_impl + load_vector + qdot, 4-bit) exactly, so
// the per-lane value set, the fp32 expression forms and the simd_sum tree are MLX's (bit-exact vs the stepwise ref):
// - V = 16 values per lane per block of 32 * V = 512 when C % 512 == 0 (qmv_fast), else V = 8 per 256 (qmv, with the
//   partial last block of qdot_safe: only `rem` values, rem a multiple of 4).
// - sum = x0 + x1 + x2 + x3 per 4 values (left to right) added to the block sum; x_thread = x * 2^-4j (exact).
// - accum += (x0*(w & 0xf) + x1*(w & 0xf0) + x2*(w & 0xf00) + x3*(w & 0xf000)) per 16-bit word;
//   result += scale * accum + sum * bias, once per lane block (scale / bias of the block's group).
// Losslessness of the word-major order (vs the old token-major one and across K / NR): for a fixed (row n, token t)
// the values combined and the order are unchanged - sum[t] still gets its words i = 0..V/4-1 in order, accum[n][t]
// starts at 0 and gets word i's 4-product expression in order i = 0..V/4-1, acc[n][t] gets sc*accum + sum*bi once per
// lane block in block order, simd_sum over the same 32 lanes. Only WHERE things are computed moved: float(w & mask)
// (exact: <= 0xf000 < 2^24) and the x scaling (exact power-of-two divide) are computed once and reused, which yields
// the same fp32 value as recomputing them. Every source expression is the K = 1 one verbatim, so the compiler's fp
// contraction choices (Safe math mode: no reassociation) are the same per (n, t) and nothing mixes tokens or rows.
template <uint K, uint NR, uint V>
static inline void q4b_dot_v(thread device const uint* const (&wr)[NR], thread device const ushort* const (&sr)[NR],
                             thread device const ushort* const (&br)[NR], device const ushort* x, uint C, uint lane,
                             thread float (&acc)[NR][K]) {
    // C % 64 == 0 (QwenConfig::validate) and the block start is a multiple of V, so every lane block is full
    // (MLX's qdot_safe tail then does the same math as qdot).
    constexpr uint NW = V / 8;   // u32 weight words per lane block (2 or 1)
    for (uint n = 0; n < NR; n++) for (uint t = 0; t < K; t++) acc[n][t] = 0.0f;
    for (uint base = lane * V; base < C; base += 32 * V) {
        uint wq[NR][NW];
        for (uint n = 0; n < NR; n++) {
            if (NW == 2) { uint2 q = *(device const uint2*)(wr[n] + base / 8); wq[n][0] = q.x; wq[n][NW - 1] = q.y; }
            else wq[n][0] = wr[n][base / 8];
        }
        float accum[NR][K], sum[K];
        for (uint t = 0; t < K; t++) { sum[t] = 0.0f; for (uint n = 0; n < NR; n++) accum[n][t] = 0.0f; }
        for (uint i = 0; i < V / 4; i++) {
            // weight side once per word, shared by the K tokens
            float wf[NR][4];
            for (uint n = 0; n < NR; n++) {
                uint w = (wq[n][i / 2] >> (16 * (i & 1))) & 0xffffu;  // 16-bit word i (little endian)
                wf[n][0] = float(w & 0x000f); wf[n][1] = float(w & 0x00f0);
                wf[n][2] = float(w & 0x0f00); wf[n][3] = float(w & 0xf000);
            }
            // x side once per token, shared by the NR rows
            for (uint t = 0; t < K; t++) {
                ushort4 hv = ((device const ushort4*)(x + t * C + base))[i];
                float a0 = bf(hv.x), a1 = bf(hv.y), a2 = bf(hv.z), a3 = bf(hv.w);
                sum[t] += a0 + a1 + a2 + a3;
                float x0 = a0, x1 = a1 / 16.0f, x2 = a2 / 256.0f, x3 = a3 / 4096.0f;
                for (uint n = 0; n < NR; n++)
                    accum[n][t] += (x0 * wf[n][0] + x1 * wf[n][1] + x2 * wf[n][2] + x3 * wf[n][3]);
            }
        }
        for (uint n = 0; n < NR; n++) {
            float sc = bf(sr[n][base / 64]), bi = bf(br[n][base / 64]);
            for (uint t = 0; t < K; t++) acc[n][t] += sc * accum[n][t] + sum[t] * bi;
        }
    }
}
// Row bases as u32 weight words: W u32 [R][C/8].
template <uint K, uint NR>
static inline void q4b_dot(thread device const uint4* const (&wr4)[NR], thread device const ushort* const (&sr)[NR],
                           thread device const ushort* const (&br)[NR], device const ushort* x, uint C, uint lane,
                           thread float (&acc)[NR][K]) {
    device const uint* wr[NR];
    for (uint n = 0; n < NR; n++) wr[n] = (device const uint*)wr4[n];
    if (C % 512 == 0) q4b_dot_v<K, NR, 16>(wr, sr, br, x, C, lane, acc);
    else q4b_dot_v<K, NR, 8>(wr, sr, br, x, C, lane, acc);
}

// ---- q4b_gemv: groups ceil(R / (2 * RPS)) x 64 threads, RPS rows per simdgroup.
// MODE 0: out = bf(v). MODE 2: res = bf(bf(res) + bf(v)) in place (residual, MLX bf16 add of the bf16 matmul output).
// MODE 3: outf = v (fp32 logits).
// Names: q4b_gemv_k{K}_m{M} = RPS 2 (the original geometry, ceil(R/4) groups; tests / MTP / examples use these);
// q4b_gemv_k{K}_r{D}_m{M} = RPS D (4, or 8 for K <= 3), ceil(R / (2D)) groups. Same per-row numbers for every D.
template <uint K, uint RPS, uint MODE>
kernel void q4b_gemv(device const uint4* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                     device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                     device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                     device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint row0 = (tg * nsg + sg) * RPS;
    if (row0 >= R) return;
    uint C32 = C / 32, G = C / 64;
    device const uint4* wr[RPS];
    device const ushort* sr[RPS];
    device const ushort* br[RPS];
    for (uint n = 0; n < RPS; n++) {
        uint row = min(row0 + n, R - 1);
        wr[n] = W + row * C32; sr[n] = S + row * G; br[n] = B + row * G;
    }
    float acc[RPS][K];
    q4b_dot<K, RPS>(wr, sr, br, x, C, lane, acc);
    for (uint n = 0; n < RPS; n++) for (uint t = 0; t < K; t++) {
        float v = simd_sum(acc[n][t]);
        uint row = row0 + n;
        if (lane == 0 && row < R) {
            uint o = t * R + row;
            if (MODE == 0) out[o] = to_bf(v);
            else if (MODE == 2) res[o] = to_bf(bf(res[o]) + rbf(v));
            else outf[o] = v;
        }
    }
}
#define Q4B_GEMV_T(NAME_, K_, D_, M_) template [[host_name(NAME_)]] kernel void q4b_gemv<K_, D_, M_>( \
    device const uint4*, device const ushort*, device const ushort*, device const ushort*, device ushort*, \
    constant uint&, constant uint&, device ushort*, device float*, uint, uint, uint, uint);
#define Q4B_GEMV(K_) Q4B_GEMV_T("q4b_gemv_k" #K_ "_m0", K_, 2, 0) Q4B_GEMV_T("q4b_gemv_k" #K_ "_m2", K_, 2, 2) \
    Q4B_GEMV_T("q4b_gemv_k" #K_ "_m3", K_, 2, 3)
#define Q4B_GEMV_D(K_, D_) Q4B_GEMV_T("q4b_gemv_k" #K_ "_r" #D_ "_m0", K_, D_, 0) \
    Q4B_GEMV_T("q4b_gemv_k" #K_ "_r" #D_ "_m2", K_, D_, 2) Q4B_GEMV_T("q4b_gemv_k" #K_ "_r" #D_ "_m3", K_, D_, 3)
Q4B_GEMV(1) Q4B_GEMV(2) Q4B_GEMV(3) Q4B_GEMV(4) Q4B_GEMV(5) Q4B_GEMV(6) Q4B_GEMV(7) Q4B_GEMV(8)
Q4B_GEMV_D(1, 4) Q4B_GEMV_D(2, 4) Q4B_GEMV_D(3, 4) Q4B_GEMV_D(4, 4) Q4B_GEMV_D(5, 4) Q4B_GEMV_D(8, 4)
Q4B_GEMV_D(1, 8) Q4B_GEMV_D(2, 8) Q4B_GEMV_D(3, 8)

// ---- q4b_swiglu: groups ceil(R / (2 NP)) x 64 threads; a simdgroup computes NP (gate row, up row) pairs, i.e.
// D = 2 NP row-dots through the same q4b_dot (so per-row numbers equal q4b_gemv's).
// out = MLX swiglu = silu(g) * u on bf16 arrays, rounding after each op as MLX's bfloat Sigmoid / Multiply do:
// g = bf(gv), u = bf(uv); e = bf(exp|g|); d = bf(1 + e); y = bf(1 / d); s = g < 0 ? y : bf(1 - y); out = bf(bf(g*s) * u).
static inline ushort q4b_silu_mul(float gv, float uv) {
    float g = rbf(gv), u = rbf(uv);
    float e = rbf(precise::exp(fabs(g)));
    float d = rbf(1.0f + e);
    float y = rbf(precise::divide(1.0f, d));
    float s = (g < 0.0f) ? y : rbf(1.0f - y);
    return to_bf(rbf(g * s) * u);
}
// Names: q4b_swiglu_k{K} = NP 1 (ceil(R/2) groups, the original); q4b_swiglu_k{K}_r{D} = NP D/2, ceil(R/D) groups.
template <uint K, uint NP>
kernel void q4b_swiglu(device const uint4* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                       device const ushort* Bg [[buffer(2)]], device const uint4* Wu [[buffer(3)]],
                       device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                       device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                       constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint row0 = (tg * nsg + sg) * NP;
    if (row0 >= R) return;
    uint C32 = C / 32, G = C / 64;
    device const uint4* wr[2 * NP];
    device const ushort* sr[2 * NP];
    device const ushort* br[2 * NP];
    for (uint p = 0; p < NP; p++) {
        uint row = min(row0 + p, R - 1);
        wr[2 * p] = Wg + row * C32; sr[2 * p] = Sg + row * G; br[2 * p] = Bg + row * G;
        wr[2 * p + 1] = Wu + row * C32; sr[2 * p + 1] = Su + row * G; br[2 * p + 1] = Bu + row * G;
    }
    float acc[2 * NP][K];
    q4b_dot<K, 2 * NP>(wr, sr, br, x, C, lane, acc);
    for (uint p = 0; p < NP; p++) for (uint t = 0; t < K; t++) {
        float gv = simd_sum(acc[2 * p][t]), uv = simd_sum(acc[2 * p + 1][t]);
        if (lane == 0 && row0 + p < R) out[t * R + row0 + p] = q4b_silu_mul(gv, uv);
    }
}
#define Q4B_SWIGLU_T(NAME_, K_, NP_) template [[host_name(NAME_)]] kernel void q4b_swiglu<K_, NP_>( \
    device const uint4*, device const ushort*, device const ushort*, device const uint4*, device const ushort*, \
    device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, uint, uint, uint, uint);
#define Q4B_SWIGLU(K_) Q4B_SWIGLU_T("q4b_swiglu_k" #K_, K_, 1)
#define Q4B_SWIGLU_D(K_, D_) Q4B_SWIGLU_T("q4b_swiglu_k" #K_ "_r" #D_, K_, D_ / 2)
Q4B_SWIGLU(1) Q4B_SWIGLU(2) Q4B_SWIGLU(3) Q4B_SWIGLU(4) Q4B_SWIGLU(5) Q4B_SWIGLU(6) Q4B_SWIGLU(7) Q4B_SWIGLU(8)
Q4B_SWIGLU_D(1, 4) Q4B_SWIGLU_D(2, 4) Q4B_SWIGLU_D(3, 4) Q4B_SWIGLU_D(4, 4) Q4B_SWIGLU_D(5, 4) Q4B_SWIGLU_D(8, 4)
Q4B_SWIGLU_D(1, 8) Q4B_SWIGLU_D(2, 8) Q4B_SWIGLU_D(3, 8)
