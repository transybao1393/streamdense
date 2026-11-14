// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (gemv.h GEMVKernel: lane mapping, accumulation order, load_safe tail, simd reduction).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// gemv_bf16.metal - streamdense Qwen3.8-27B bf16-weight GEMV (P2, U17): the MTP fc, out[t] = W x[t].
// Needs common_bf16.metal first (bf, to_bf). mtp.fc.weight is BF16 [R = 5120][C = 10240] (read from the
// mtp/weights.safetensors header), so this is a plain bf16 GEMV, not the 4-bit one.
//
// Numerics: mirrors MLX `gemv` (mlx/backend/metal/kernels/gemv.h, GEMVKernel), the kernel `x @ W.T` dispatches at
// M = 1 with W row-major (non-transposed mat, matrix_ld = C). MLX picks BM = 8 (out >= 4096; 4 below), BN = 1, SM = 1,
// SN = 32, TM = 4, TN = 4. BM / BN do not change the math (BN = 1: no threadgroup reduction), so the per-row value
// is fixed by: lane l of the simdgroup owning rows [r0, r0 + 4) covers cols 128 i + 4 l .. + 3 for i = 0, 1, ...;
//   acc[tm] += float(W[r0 + tm][c]) * float(x[c])   for tn = 0..3 in order (fp32, one accumulator per row),
// a `load_safe` tail (cols past C read as 0), then acc += simd_shuffle_down(acc, s) for s = 16, 8, 4, 2, 1, and lane 0
// stores bf16(acc) (RNE). The tail simdgroup is shifted inwards (r0 = R - 4) like MLX, so it rewrites equal values.
// K-invariance (hard rule): each token t runs exactly the K = 1 sequence on its own accumulators; the weight loads
// are shared. Row t of a K = 2 dispatch is bit-equal to a K = 1 dispatch.
//
// Buffers: W ushort[R][C] [0], x ushort[K][C] [1], out ushort[K][R] [2], R [3], C [4].
// Dispatch: groups((ceil(R / 32), 1), 256) - 8 simdgroups x 4 rows per threadgroup. Requires R >= 4.

template <uint K>
static inline void bgemv_body(device const ushort* W, device const ushort* x, device ushort* out, uint R, uint C,
                              uint tg, uint sg, uint lane) {
    constexpr uint TM = 4, TN = 4, BLK = 32 * TN;
    uint r0 = (tg * 8 + sg) * TM;
    if (r0 >= R) return;
    r0 = r0 + TM <= R ? r0 : R - TM;
    device const ushort* m = W + (ulong)r0 * C;
    float acc[K][TM];
    for (uint t = 0; t < K; t++) for (uint i = 0; i < TM; i++) acc[t][i] = 0.0f;
    uint bn = lane * TN;
    uint n_iter = C / BLK, last = n_iter * BLK;
    bool vec4 = (C & 3u) == 0;   // rows 8-byte aligned: ushort4 loads (same values, same math order)
    for (uint it = 0; it < n_iter; it++, bn += BLK) {
        float v[K][TN];
        for (uint t = 0; t < K; t++) {
            if (vec4) {
                ushort4 q = *(device const ushort4*)(x + t * C + bn);
                v[t][0] = bf(q.x); v[t][1] = bf(q.y); v[t][2] = bf(q.z); v[t][3] = bf(q.w);
            } else {
                for (uint j = 0; j < TN; j++) v[t][j] = bf(x[t * C + bn + j]);
            }
        }
        for (uint tm = 0; tm < TM; tm++) {
            float w[TN];
            if (vec4) {
                ushort4 q = *(device const ushort4*)(m + (ulong)tm * C + bn);
                w[0] = bf(q.x); w[1] = bf(q.y); w[2] = bf(q.z); w[3] = bf(q.w);
            } else {
                for (uint j = 0; j < TN; j++) w[j] = bf(m[(ulong)tm * C + bn + j]);
            }
            for (uint t = 0; t < K; t++)
                for (uint j = 0; j < TN; j++) acc[t][tm] += w[j] * v[t][j];
        }
    }
    if (C > last) {   // MLX leftover: load_safe zero-fills cols >= C (bn here = last + 4 * lane)
        float v[K][TN];
        for (uint t = 0; t < K; t++)
            for (uint j = 0; j < TN; j++) v[t][j] = bn + j < C ? bf(x[t * C + bn + j]) : 0.0f;
        for (uint tm = 0; tm < TM; tm++) {
            float w[TN];
            for (uint j = 0; j < TN; j++) w[j] = bn + j < C ? bf(m[(ulong)tm * C + bn + j]) : 0.0f;
            for (uint t = 0; t < K; t++)
                for (uint j = 0; j < TN; j++) acc[t][tm] += w[j] * v[t][j];
        }
    }
    for (uint t = 0; t < K; t++)
        for (uint tm = 0; tm < TM; tm++)
            for (ushort s = 16; s >= 1; s >>= 1) acc[t][tm] += simd_shuffle_down(acc[t][tm], s);
    if (lane == 0) {
        for (uint t = 0; t < K; t++)
            for (uint tm = 0; tm < TM; tm++) out[t * R + r0 + tm] = to_bf(acc[t][tm]);
    }
}

kernel void bf16_gemv_k1(device const ushort* W [[buffer(0)]], device const ushort* x [[buffer(1)]],
                         device ushort* out [[buffer(2)]], constant uint& R [[buffer(3)]], constant uint& C [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    bgemv_body<1>(W, x, out, R, C, tg, sg, lane);
}

kernel void bf16_gemv_k2(device const ushort* W [[buffer(0)]], device const ushort* x [[buffer(1)]],
                         device ushort* out [[buffer(2)]], constant uint& R [[buffer(3)]], constant uint& C [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    bgemv_body<2>(W, x, out, R, C, tg, sg, lane);
}
