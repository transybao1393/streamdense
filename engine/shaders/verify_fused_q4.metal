// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (steel BaseMMAFrag fragment coordinate mapping; affine 4-bit layout and swiglu rounding chain).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// verify_fused_q4.metal - streamdense Qwen3.8-27B K-row 4-bit matmul for the exact VERIFY pass, K = 1..16 token rows
// (2026-11-03, verify_bench). Needs common_bf16.metal and gemv_mma_q4.metal first (bf, to_bf, qm_fm, qm_fn, qm_rbf,
// qm_silu_mul, QM_UNROLL).
// Layout as qm_gemv: W u32 [R][C/8] (8 nibbles per u32, low nibble first); scales / biases bf16 [R][C/64];
// w = s*q + b; out / res [K][R] bf16 (ushort); outf [K][R] f32. C % 64 == 0, any R, 1 <= K <= 16.
//
// SAME ARITHMETIC as gemv_mma_q4.metal with SK = 8 (the engine's MMA_SK): per output row r, token t,
//   per group g: P = 0; for c = 0..63: P = fma(x[t][64g + c] * 2^-4(c % 8), float(w nibble c & 0xf << 4(c % 8)), P)
//                xs = (p0 + p1) + (p2 + p3), p_j = 0 + x[2j] + x[2j + 1] + x[8 + 2j] + x[9 + 2j] + ... (in group)
//                acc = acc + fma(sc, P, xs * bi)       over slice k's groups [k G / 8, (k + 1) G / 8) in order
//   v = acc_0 + acc_1 + ... + acc_7 (left to right); stores as qm_gemv (m0 / m2 / m3, swiglu MLX chain).
// so every row is BIT-IDENTICAL to qm_gemv_n*_s8 / qm_swiglu_n*_s8 and their twins qm_gemvt*_s8 / qm_swiglut*_s8
// (the current exact passes), for every K (K-invariant: a row never depends on K or on the other rows).
// What changes is where the work happens (verify_bench: 27B K = 8 gate+up 812 -> ~595 µs):
// - vf_xprep, once per matmul INPUT (shared by every matrix that reads it): xt[c][KP] = x[t][c] * 2^-4(c % 8) as f32
//   (exact: power-of-two scale) and st[g][KP] = xs of group g (the order above), TOKEN-MINOR, KP = 8 NT; padding
//   tokens K..KP-1 are written as 0. The matmuls load their x fragments as one float2 per lane per MMA step instead
//   of unpacking, scaling and summing x in every threadgroup (qm_dot: ~50 ALU ops per lane per group).
// - Operand roles swapped vs qm_dot: D[row][token] = W (A: 8 rows x 8 columns) x X (B: 8 columns x 8 tokens) + D.
//   The MMA is still, per output element, the sequential fma chain over the 8 columns with the same exact products
//   x * q (fma is symmetric in its factors), so the bits do not change. A lane (fm, fn) now holds A[fm][fn],
//   A[fm][fn + 1] = nibbles fn, fn + 1 of ONE word of ONE row: one 32 B row-group load per block per group and one
//   scale + one bias (qm_dot: two rows, two of each; load issue was the measured limiter, not the dequant ALU).
// - NT = 1 or 2 token tiles of 8 rows: at K = 9..16 the dequantized W fragment of a step feeds both tiles (one
//   weight read and one dequant per 16 tokens; the current kernels need two dispatches of <= 8 rows).
// An MMA output column depends only on its own B column, so padding tokens cannot touch the real ones.
// Kernels (SK = 8 simdgroups per threadgroup = the 8 split-K slices, slice k reduced in order as qm_reduce):
// vf_xprep: x bf16 [K][C] [0], xt f32 [C][KP] [1], st f32 [C / 64][KP] [2], C [3], K [4], KP [5];
//   threads C / 64 * KP (1-D), thread i = (group i / KP, token i % KP): neighbouring threads write neighbouring tokens.
// vf_gemv_n{NB}_t{NT}_m{M}: W[0] S[1] B[2] xt[3] out[4] R[5] C[6] res[7] outf[8] K[9] st[10];
//   groups ceil(R / (8 NB)) x 256 threads. MODE 0: out = bf(v); 2: res = bf(bf(res) + bf(v)) in place; 3: outf = v.
// vf_swiglu_n{NB}_t{NT}: Wg/Sg/Bg[0-2] Wu/Su/Bu[3-5] xt[6] out[7] R[8] C[9] K[10] st[11]; NB blocks of 8 outputs
//   (gate and up rows of the same outputs), same grid.
// vf_abl{A}_swiglu_n1_t1: bench ablations (verify_bench --kern), parts of the work removed, outputs meaningless.

constant constexpr uint VF_SK = 8;  // split-K slices: MUST equal the engine's MMA_SK (part of the result)

kernel void vf_xprep(device const ushort* x [[buffer(0)]], device float* xt [[buffer(1)]], device float* st [[buffer(2)]],
                     constant uint& C [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& KP [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    const uint g = id / KP, t = id % KP, G = C / 64;
    if (g >= G) return;
    device float* ap = xt + 64 * g * KP + t;
    if (t >= K) {
        for (uint c = 0; c < 64; c++) ap[c * KP] = 0.0f;
        st[g * KP + t] = 0.0f;
        return;
    }
    device const uint4* xp = (device const uint4*)(x + t * C + 64 * g);
    float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    QM_UNROLL for (uint s = 0; s < 8; s++) {
        uint4 v = xp[s];
        uint u[4] = {v.x, v.y, v.z, v.w};
        QM_UNROLL for (uint j = 0; j < 4; j++) {
            float lo = as_type<float>(u[j] << 16), hi = as_type<float>(u[j] & 0xffff0000u);
            p[j] += lo;
            p[j] += hi;
            ap[(8 * s + 2 * j) * KP] = lo * as_type<float>((127u - 4 * (2 * j)) << 23);
            ap[(8 * s + 2 * j + 1) * KP] = hi * as_type<float>((127u - 4 * (2 * j + 1)) << 23);
        }
    }
    st[g * KP + t] = (p[0] + p[1]) + (p[2] + p[3]);
}

// P[tt][b] = the 64-column fma chains of group g (row of block b, tokens of tile tt), from zero.
template <uint NB, uint NT, uint ABL>
static inline void vf_group(thread device const uint* const (&wr)[NB], device const float* xp, uint g, uint lane,
                            uint m0, uint m1, thread simdgroup_float8x8 (&P)[NT][NB]) {
    constexpr uint KP = 8 * NT;
    uint4 w[NB][2];
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        device const uint4* p = (device const uint4*)(wr[b] + 8 * g);
        if (ABL & 4) { w[b][0] = uint4(g * 77u + lane + b, lane, g, b); w[b][1] = w[b][0] ^ 0x5555u; }
        else { w[b][0] = p[0]; w[b][1] = p[1]; }
    }
    QM_UNROLL for (uint tt = 0; tt < NT; tt++) QM_UNROLL for (uint b = 0; b < NB; b++)
        P[tt][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    QM_UNROLL for (uint s = 0; s < 8; s++) {
        simdgroup_float8x8 X[NT];
        QM_UNROLL for (uint tt = 0; tt < NT; tt++) {
            float2 a = (ABL & 1) ? float2(float(lane + s) * 1e-3f, float(g) * 1e-3f)
                                 : *(device const float2*)(xp + (64 * g + 8 * s) * KP + 8 * tt);
            X[tt].thread_elements()[0] = a.x;
            X[tt].thread_elements()[1] = a.y;
        }
        QM_UNROLL for (uint b = 0; b < NB; b++) {
            const uint ws = w[b][s / 4][s % 4];
            simdgroup_float8x8 Wm;
            Wm.thread_elements()[0] = (ABL & 2) ? as_type<float>(ws) : float(ws & m0);
            Wm.thread_elements()[1] = (ABL & 2) ? as_type<float>(ws ^ 1u) : float(ws & m1);
            QM_UNROLL for (uint tt = 0; tt < NT; tt++) simdgroup_multiply_accumulate(P[tt][b], Wm, X[tt], P[tt][b]);
        }
    }
}

// acc[tt][b][c] = acc + fma(sc, P, xs * bi) of group g (the lane's row of block b, token 8 tt + fn + c).
template <uint NB, uint NT, uint ABL>
static inline void vf_epi(thread device const ushort* const (&sr)[NB], thread device const ushort* const (&br)[NB],
                          device const float* st, uint g, uint fn, thread simdgroup_float8x8 (&P)[NT][NB],
                          thread float (&acc)[NT][NB][2]) {
    constexpr uint KP = 8 * NT;
    float2 xs[NT];
    QM_UNROLL for (uint tt = 0; tt < NT; tt++)
        xs[tt] = (ABL & 8) ? float2(float(g), 1.0f) : *(device const float2*)(st + g * KP + 8 * tt + fn);
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        float sc = (ABL & 8) ? 0.5f : bf(sr[b][g]), bi = (ABL & 8) ? 0.25f : bf(br[b][g]);
        QM_UNROLL for (uint tt = 0; tt < NT; tt++) {
            thread auto& pe = P[tt][b].thread_elements();
            acc[tt][b][0] = acc[tt][b][0] + fma(sc, pe[0], xs[tt].x * bi);
            acc[tt][b][1] = acc[tt][b][1] + fma(sc, pe[1], xs[tt].y * bi);
        }
    }
}

// acc[tt][b][c] = row (row0 + 8 b + fm) . x[8 tt + fn + c] over the groups [g0, g1) in order. wr / sr / br: the
// lane's row per block. (Two groups per iteration, NB 2 / 4 or NT 2 chains: no faster, verify_bench 2026-11-03.)
template <uint NB, uint NT, uint ABL = 0>
static inline void vf_dot(thread device const uint* const (&wr)[NB], thread device const ushort* const (&sr)[NB],
                          thread device const ushort* const (&br)[NB], device const float* xt, device const float* st,
                          uint g0, uint g1, uint lane, thread float (&acc)[NT][NB][2]) {
    constexpr uint KP = 8 * NT;
    const uint fm = qm_fm(lane), fn = qm_fn(lane);
    const uint m0 = 0xfu << (4 * fn), m1 = 0xfu << (4 * (fn + 1));
    device const float* xp = xt + fm * KP + fn;
    QM_UNROLL for (uint tt = 0; tt < NT; tt++) QM_UNROLL for (uint b = 0; b < NB; b++) { acc[tt][b][0] = 0.0f; acc[tt][b][1] = 0.0f; }
    for (uint g = g0; g < g1; g++) {
        simdgroup_float8x8 P[NT][NB];
        vf_group<NB, NT, ABL>(wr, xp, g, lane, m0, m1, P);
        vf_epi<NB, NT, ABL>(sr, br, st, g, fn, P, acc);
    }
}

// qm_reduce over the 8 slices for NT x NB x 2 values per lane: simdgroup 0 adds slices 1..7 in order.
template <uint N>
static inline void vf_reduce(thread float* acc, threadgroup float* red, uint sg, uint lane) {
    if (sg > 0) QM_UNROLL for (uint i = 0; i < N; i++) red[((sg - 1) * N + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (uint k = 1; k < VF_SK; k++) QM_UNROLL for (uint i = 0; i < N; i++) acc[i] += red[((k - 1) * N + i) * 32 + lane];
}

template <uint NB, uint NT, uint MODE>
kernel void vf_gemv(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                    device const ushort* B [[buffer(2)]], device const float* xt [[buffer(3)]],
                    device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                    device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]], constant uint& K [[buffer(9)]],
                    device const float* st [[buffer(10)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(VF_SK - 1) * NT * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    const uint fm = qm_fm(lane), fn = qm_fn(lane), C8 = C / 8, G = C / 64;
    device const uint* wr[NB];
    device const ushort* sr[NB];
    device const ushort* br[NB];
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        uint row = min(row0 + 8 * b + fm, R - 1);
        wr[b] = W + row * C8; sr[b] = S + row * G; br[b] = B + row * G;
    }
    float acc[NT][NB][2];
    vf_dot<NB, NT>(wr, sr, br, xt, st, sg * G / VF_SK, (sg + 1) * G / VF_SK, lane, acc);
    vf_reduce<NT * NB * 2>(&acc[0][0][0], red, sg, lane);
    if (sg != 0) return;
    QM_UNROLL for (uint tt = 0; tt < NT; tt++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        const uint t = 8 * tt + fn + c;
        if (t >= K) continue;
        QM_UNROLL for (uint b = 0; b < NB; b++) {
            uint row = row0 + 8 * b + fm;
            if (row >= R) continue;
            uint o = t * R + row;
            float v = acc[tt][b][c];
            if (MODE == 0) out[o] = to_bf(v);
            else if (MODE == 2) res[o] = to_bf(bf(res[o]) + qm_rbf(v));
            else outf[o] = v;
        }
    }
}

template <uint NB, uint NT, uint ABL = 0>
kernel void vf_swiglu(device const uint* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                      device const ushort* Bg [[buffer(2)]], device const uint* Wu [[buffer(3)]],
                      device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                      device const float* xt [[buffer(6)]], device ushort* out [[buffer(7)]],
                      constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& K [[buffer(10)]],
                      device const float* st [[buffer(11)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(VF_SK - 1) * NT * 2 * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    const uint fm = qm_fm(lane), fn = qm_fn(lane), C8 = C / 8, G = C / 64;
    device const uint* wr[2 * NB];
    device const ushort* sr[2 * NB];
    device const ushort* br[2 * NB];
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        uint row = min(row0 + 8 * b + fm, R - 1);
        wr[b] = Wg + row * C8; sr[b] = Sg + row * G; br[b] = Bg + row * G;
        wr[NB + b] = Wu + row * C8; sr[NB + b] = Su + row * G; br[NB + b] = Bu + row * G;
    }
    float acc[NT][2 * NB][2];
    vf_dot<2 * NB, NT, ABL>(wr, sr, br, xt, st, sg * G / VF_SK, (sg + 1) * G / VF_SK, lane, acc);
    vf_reduce<NT * 2 * NB * 2>(&acc[0][0][0], red, sg, lane);
    if (sg != 0) return;
    QM_UNROLL for (uint tt = 0; tt < NT; tt++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        const uint t = 8 * tt + fn + c;
        if (t >= K) continue;
        QM_UNROLL for (uint b = 0; b < NB; b++) {
            uint row = row0 + 8 * b + fm;
            if (row < R) out[t * R + row] = qm_silu_mul(acc[tt][b][c], acc[tt][NB + b][c]);
        }
    }
}

#define VF_GEMV_T(NB_, NT_, M_) template [[host_name("vf_gemv_n" #NB_ "_t" #NT_ "_m" #M_)]] kernel void \
    vf_gemv<NB_, NT_, M_>(device const uint*, device const ushort*, device const ushort*, device const float*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, constant uint&, device const float*, \
    uint, uint, uint);
#define VF_SWIGLU_T(NB_, NT_, A_, SFX_) template [[host_name("vf" SFX_ "_swiglu_n" #NB_ "_t" #NT_)]] kernel void \
    vf_swiglu<NB_, NT_, A_>(device const uint*, device const ushort*, device const ushort*, device const uint*, \
    device const ushort*, device const ushort*, device const float*, device ushort*, constant uint&, constant uint&, \
    constant uint&, device const float*, uint, uint, uint);
#define VF_ALL(NB_, NT_) VF_GEMV_T(NB_, NT_, 0) VF_GEMV_T(NB_, NT_, 2) VF_GEMV_T(NB_, NT_, 3) VF_SWIGLU_T(NB_, NT_, 0, "")
VF_ALL(1, 1) VF_ALL(1, 2) VF_ALL(2, 1) VF_ALL(2, 2)
VF_GEMV_T(4, 1, 0) VF_GEMV_T(4, 1, 2) VF_GEMV_T(4, 1, 3) VF_GEMV_T(4, 2, 0) VF_GEMV_T(4, 2, 2) VF_GEMV_T(4, 2, 3)
// bench only (decode_bench --verify K 17..32): 3 / 4 token tiles at the engine's NB (timing; bit identity untested)
VF_GEMV_T(2, 3, 0) VF_GEMV_T(2, 3, 2) VF_GEMV_T(2, 3, 3) VF_SWIGLU_T(1, 3, 0, "")
VF_GEMV_T(2, 4, 0) VF_GEMV_T(2, 4, 2) VF_GEMV_T(2, 4, 3) VF_SWIGLU_T(1, 4, 0, "")
VF_SWIGLU_T(1, 1, 2, "_abl2") VF_SWIGLU_T(1, 1, 4, "_abl4") VF_SWIGLU_T(1, 1, 8, "_abl8") VF_SWIGLU_T(1, 1, 15, "_abl15")

// ---- probes (bench only): sustained simdgroup MMA and scalar FMA rate of this GPU (verify_bench --peak).
kernel void vf_peak_mma(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 a, b, c[8];
    a.thread_elements()[0] = float(lane) * 1e-3f; a.thread_elements()[1] = 1.0f;
    b.thread_elements()[0] = 0.5f; b.thread_elements()[1] = float(tg) * 1e-4f;
    for (uint j = 0; j < 8; j++) c[j] = make_filled_simdgroup_matrix<float, 8, 8>(float(j));
    for (uint i = 0; i < n; i++) QM_UNROLL for (uint j = 0; j < 8; j++) simdgroup_multiply_accumulate(c[j], a, b, c[j]);
    float s = 0;
    for (uint j = 0; j < 8; j++) s += c[j].thread_elements()[0] + c[j].thread_elements()[1];
    if (s == 12345.678f) out[tg * 32 + lane] = s;
}
kernel void vf_peak_fma(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float a = float(lane) * 1e-3f, b = float(tg) * 1e-4f;
    float c[16];
    for (uint j = 0; j < 16; j++) c[j] = float(j);
    for (uint i = 0; i < n; i++) QM_UNROLL for (uint j = 0; j < 16; j++) c[j] = fma(a, b, c[j]);
    float s = 0;
    for (uint j = 0; j < 16; j++) s += c[j];
    if (s == 12345.678f) out[tg * 32 + lane] = s;
}
