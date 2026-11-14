// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (steel BaseMMAFrag fragment coordinate mapping; affine 4-bit layout and swiglu rounding chain).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// gemv_mma_q4.metal - streamdense Qwen3.8-27B 4-bit matmul for K = 1..8 token rows (exact decode / verify pass
// candidate, not wired into encode_layer yet). Needs common_bf16.metal first (bf, to_bf); every helper has a qm_
// prefix, so the file also compiles alone after common_bf16 (bench / test libraries).
// Layout as q4b_gemv: W u32 [R][C/8] (8 nibbles per u32, low nibble first); scales / biases bf16 [R][C/64];
// w = s*q + b; x [K][C] and out / res [K][R] bf16 (ushort); outf [K][R] f32. C % 64 == 0, any R, 1 <= K <= 8.
//
// Two kernel families, ONE arithmetic definition (per output row r, token t, with SK = split-K slices):
//   per group g: P = 0; for s = 0..7 (word s), k = 0..7 (nibble k): P = fma(x[t][64g+8s+k] * 2^-4k,
//                float(w_s & 0xf << 4k), P)           (= x * q exactly: both scalings are powers of two)
//                xs = (p0 + p1) + (p2 + p3), p_j = 0 + x[2j] + x[2j+1] + x[8+2j] + x[9+2j] + ... (s ascending)
//                acc = acc + fma(sc, P, xs * bi)       over the slice's groups [k G/SK, (k+1) G/SK) in order
//   v = acc_0 + acc_1 + ... + acc_{SK-1} (left to right); stores as q4b_gemv (m0 / m2 / m3, swiglu MLX chain).
// v depends only on x[t], the weights and SK - never on K, on the other rows, on NB or on which family ran it.
// - qm_gemv / qm_swiglu (simdgroup_matrix): a simdgroup owns NB blocks of 8 weight rows x 8 token rows (rows >= K
//   are zero padding, K is a runtime value), SK simdgroups split C. P is 8 MMAs (8x8x8) per group from zero.
//   On Apple GPUs an 8x8x8 float MMA is, per element, the sequential fma chain d = C; d = fma(a_k, b_k, d) for
//   k = 0..7 (measured on M4 Pro: 100% of 524288 random-exponent elements), so P is the chain above.
// - qm_gemvt / qm_swiglut (scalar "twins", KT = 1..8 compiled in): same expressions, no padding; KT = 1 is the
//   plain decode kernel at GEMV speed. Checked bit-identical to the MMA kernels by `streamdense bench gemv-k
//   --impl mma` (every K, every NB, neighbour rows replaced).
// NOT bit-exact with q4b_gemv (different summation order): closeness reported by the bench.

#define QM_UNROLL _Pragma("clang loop unroll(full)")

static inline float qm_rbf(float f) { return bf(to_bf(f)); }

// == q4b_silu_mul (gemv_q4_bf16.metal): MLX bf16 swiglu rounding chain.
static inline ushort qm_silu_mul(float gv, float uv) {
    float g = qm_rbf(gv), u = qm_rbf(uv);
    float e = qm_rbf(precise::exp(fabs(g)));
    float d = qm_rbf(1.0f + e);
    float y = qm_rbf(precise::divide(1.0f, d));
    float s = (g < 0.0f) ? y : qm_rbf(1.0f - y);
    return to_bf(qm_rbf(g * s) * u);
}

static inline uint qm_fm(uint lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
static inline uint qm_fn(uint lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

// ---- qm_gemv / qm_swiglu (MMA). Fragments are written straight into simdgroup_float8x8 thread_elements() (MLX
// steel BaseMMAFrag layout: lane holds [fm][fn], [fm][fn + 1]); no threadgroup memory except the SK partials.
// Kernels: qm_gemv_n{NB}_s{SK}_m{0,2,3}: W[0] S[1] B[2] x[3] out[4] R[5] C[6] res[7] outf[8] K[9]; groups
//   ceil(R / (8 NB)) x 32 SK threads. MODE 0: out = bf(v); 2: res = bf(bf(res) + bf(v)) in place; 3: outf = v.
// qm_swiglu_n{NB}_s{SK}: Wg/Sg/Bg[0-2] Wu/Su/Bu[3-5] x[6] out[7] R[8] C[9] K[10]; NB blocks of 8 outputs (gate and
//   up rows of the same outputs) per simdgroup, groups ceil(R / (8 NB)) x 32 SK.
// acc[b][c] = row (block b, fn + c) . x[fm] over the groups [g0, g1). wr/sr/br: row bases of the lane's 2 rows per
// block. MMA step s = word s of the group (columns 64g + 8s .. +7), k = nibble index:
//   A[t][k] = x[t][64g + 8s + k] * 2^-4k  (lane (fm, fn): k = fn, fn + 1: one u32 = 2 bf16 per step),
//   B[k][n] = float(w_s(row n) & 0xf << 4k) = q * 2^4k  (lane: k = fm, rows fn, fn + 1; the 8 lanes fm of a row
//   read the same 2 x 16 B of the group, a broadcast load).
// Both scalings are exact powers of two, so every product A*B is the float x * q. With the MMA's per-element fma
// chain over k, P accumulates columns 64g + 0, 1, ..., 63 in natural order (what the K = 1 twin does).
template <uint NB>
static inline void qm_dot(thread device const uint* const (&wr)[NB][2], thread device const ushort* const (&sr)[NB][2],
                          thread device const ushort* const (&br)[NB][2], device const ushort* x, uint K, uint C,
                          uint g0, uint g1, uint lane, thread float (&acc)[NB][2], device const uint* gl, bool lst) {
    const uint fm = qm_fm(lane), fn = qm_fn(lane);
    const bool xv = fm < K;
    device const uint* xp = (device const uint*)(x + min(fm, K - 1) * C + fn);  // fn even: 4-byte aligned
    const float sa0 = as_type<float>((127u - 4 * fn) << 23), sa1 = as_type<float>((127u - 4 * (fn + 1)) << 23);
    const uint mk = 0xfu << (4 * fm);
    QM_UNROLL for (uint b = 0; b < NB; b++) { acc[b][0] = 0.0f; acc[b][1] = 0.0f; }
    for (uint i = g0; i < g1; i++) {
        const uint g = lst ? gl[i] : i;  // read mask: list position i -> 64-group g (plain kernels: g = i)
        uint4 w[NB][2][2];
        QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
            device const uint4* p = (device const uint4*)(wr[b][c] + 8 * g);
            w[b][c][0] = p[0]; w[b][c][1] = p[1];
        }
        float x0[8], x1[8];
        QM_UNROLL for (uint s = 0; s < 8; s++) {
            uint u = xv ? xp[32 * g + 4 * s] : 0u;
            x0[s] = as_type<float>(u << 16); x1[s] = as_type<float>(u & 0xffff0000u);
        }
        // xsum: lane (fn) sums its 16 values (s ascending, k = fn then fn + 1), then (p0 + p1) + (p2 + p3) over fn
        float xs = 0.0f;
        QM_UNROLL for (uint s = 0; s < 8; s++) { xs += x0[s]; xs += x1[s]; }
        xs = xs + simd_shuffle_xor(xs, 1);
        xs = xs + simd_shuffle_xor(xs, 8);

        simdgroup_float8x8 P[NB];
        QM_UNROLL for (uint b = 0; b < NB; b++) P[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        QM_UNROLL for (uint s = 0; s < 8; s++) {
            simdgroup_float8x8 A;
            A.thread_elements()[0] = x0[s] * sa0;
            A.thread_elements()[1] = x1[s] * sa1;
            QM_UNROLL for (uint b = 0; b < NB; b++) {
                simdgroup_float8x8 Bm;
                Bm.thread_elements()[0] = float(w[b][0][s / 4][s % 4] & mk);
                Bm.thread_elements()[1] = float(w[b][1][s / 4][s % 4] & mk);
                simdgroup_multiply_accumulate(P[b], A, Bm, P[b]);
            }
        }
        QM_UNROLL for (uint b = 0; b < NB; b++) {
            thread auto& pe = P[b].thread_elements();
            QM_UNROLL for (uint c = 0; c < 2; c++) {
                float sc = bf(sr[b][c][g]), bi = bf(br[b][c][g]);
                acc[b][c] = acc[b][c] + fma(sc, pe[c], xs * bi);  // explicit: the K = 1 twin (qm_gemv1) repeats it
            }
        }
    }
}

// Split-K over the SK simdgroups of a threadgroup (all on the same 8 NB rows): simdgroup k takes groups
// [k G / SK, (k + 1) G / SK); simdgroup 0 then adds the partials in order k = 1..SK-1 (fixed for a given SK, so
// SK is part of the kernel name and the result; K never changes it).
template <uint NB, uint SK>
static inline void qm_reduce(thread float (&acc)[NB][2], threadgroup float* red, uint sg, uint lane) {
    if (SK == 1) return;
    if (sg > 0) QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++)
        red[(((sg - 1) * NB + b) * 2 + c) * 32 + lane] = acc[b][c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (uint k = 1; k < SK; k++) QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++)
        acc[b][c] += red[(((k - 1) * NB + b) * 2 + c) * 32 + lane];
}

template <uint NB, uint SK, uint MODE>
kernel void qm_gemv(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                    device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                    device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                    device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]], constant uint& K [[buffer(9)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(SK > 1 ? SK - 1 : 1) * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    const uint fm = qm_fm(lane), fn = qm_fn(lane), C8 = C / 8, G = C / 64;
    device const uint* wr[NB][2];
    device const ushort* sr[NB][2];
    device const ushort* br[NB][2];
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = min(row0 + 8 * b + fn + c, R - 1);
        wr[b][c] = W + row * C8; sr[b][c] = S + row * G; br[b][c] = B + row * G;
    }
    float acc[NB][2];
    qm_dot<NB>(wr, sr, br, x, K, C, sg * G / SK, (sg + 1) * G / SK, lane, acc, W, false);
    qm_reduce<NB, SK>(acc, red, sg, lane);
    if (sg != 0 || fm >= K) return;
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = row0 + 8 * b + fn + c;
        if (row >= R) continue;
        uint o = fm * R + row;
        float v = acc[b][c];
        if (MODE == 0) out[o] = to_bf(v);
        else if (MODE == 2) res[o] = to_bf(bf(res[o]) + qm_rbf(v));
        else outf[o] = v;
    }
}

template <uint NB, uint SK>
kernel void qm_swiglu(device const uint* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                      device const ushort* Bg [[buffer(2)]], device const uint* Wu [[buffer(3)]],
                      device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                      device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                      constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& K [[buffer(10)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(SK > 1 ? SK - 1 : 1) * 2 * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    const uint fm = qm_fm(lane), fn = qm_fn(lane), C8 = C / 8, G = C / 64;
    device const uint* wr[2 * NB][2];
    device const ushort* sr[2 * NB][2];
    device const ushort* br[2 * NB][2];
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = min(row0 + 8 * b + fn + c, R - 1);
        wr[b][c] = Wg + row * C8; sr[b][c] = Sg + row * G; br[b][c] = Bg + row * G;
        wr[NB + b][c] = Wu + row * C8; sr[NB + b][c] = Su + row * G; br[NB + b][c] = Bu + row * G;
    }
    float acc[2 * NB][2];
    qm_dot<2 * NB>(wr, sr, br, x, K, C, sg * G / SK, (sg + 1) * G / SK, lane, acc, Wg, false);
    qm_reduce<2 * NB, SK>(acc, red, sg, lane);
    if (sg != 0 || fm >= K) return;
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = row0 + 8 * b + fn + c;
        if (row < R) out[fm * R + row] = qm_silu_mul(acc[b][c], acc[NB + b][c]);
    }
}

#define QM_GEMV_T(NB_, SK_, M_) template [[host_name("qm_gemv_n" #NB_ "_s" #SK_ "_m" #M_)]] kernel void \
    qm_gemv<NB_, SK_, M_>(device const uint*, device const ushort*, device const ushort*, device const ushort*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, constant uint&, uint, uint, uint);
#define QM_SWIGLU_T(NB_, SK_) template [[host_name("qm_swiglu_n" #NB_ "_s" #SK_)]] kernel void qm_swiglu<NB_, SK_>( \
    device const uint*, device const ushort*, device const ushort*, device const uint*, device const ushort*, \
    device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, constant uint&, \
    uint, uint, uint);
#define QM_ALL(NB_, SK_) QM_GEMV_T(NB_, SK_, 0) QM_GEMV_T(NB_, SK_, 2) QM_GEMV_T(NB_, SK_, 3) QM_SWIGLU_T(NB_, SK_)
QM_ALL(1, 1) QM_ALL(1, 2) QM_ALL(1, 4) QM_ALL(1, 8)
QM_ALL(2, 1) QM_ALL(2, 2) QM_ALL(2, 4) QM_ALL(2, 8)
QM_ALL(4, 1) QM_ALL(4, 2) QM_ALL(4, 4)

// ---- qm_gemvt / qm_swiglut: scalar "twins" of qm_gemv / qm_swiglu for KT = 1..8 token rows (no 8-row padding),
// BIT-IDENTICAL to them for the same SK (any NB). Basis: on Apple GPUs (measured on M4 Pro: 100% of 524288
// random-exponent elements) simdgroup_multiply_accumulate(D, A, B, C) on float8x8 is, per element, the sequential
// fma chain d = C; d = fma(A[i][k], B[k][j], d) for k = 0..7. So per (row, token, group) qm_gemv computes
//   P = 0; for s = 0..7: for k = 0..7: P = fma(x[64g + 8s + k] * 2^-4k, float(w_s & 0xf << 4k), P)
//   xs = (p0 + p1) + (p2 + p3), p_j = 0 + x[2j] + x[2j + 1] + x[8 + 2j] + x[9 + 2j] + ... (s ascending, in group)
//   acc = acc + fma(sc, P, xs * bi) over the slice's groups in order; slices added in order k = 0..SK-1;
// and the twin evaluates exactly those expressions in that order, only with a different work split: a thread owns
// one group of RP rows x KT tokens at a time (64 serial fmas per (row, token), one live weight uint4 per row), writes
// the per-group terms T to threadgroup memory, and reducer threads (row, slice) add them in group order.
// Dispatch: groups ceil(R / 8) x NT threads, NT = NP * CG (NP = NM * 8 / RP row passes, CG groups per chunk),
// NT >= NM * 8 * SK, RP * NT * KT <= QMT_TMAX (host: mma_twin_nt). RT = 8 output rows per group.
// Names: qm_gemvt_k{KT}_s{SK}_m{M} (buffers as qm_gemv, [9] ignored), qm_swiglut_k{KT}_s{SK} (as qm_swiglu).
constant constexpr uint QMT_RT = 8;      // output rows per threadgroup
constant constexpr uint QMT_RP = 4;      // rows per register pass
constant constexpr uint QMT_TMAX = 4096; // threadgroup floats for the T terms (16 KB)

// T terms of group g for NR rows (bases wr / sr / br) and KT tokens (x rows at xg + t * C, xs[t] their sums).
template <uint NR, uint KT>
static inline void qmt_terms(thread device const uint* const (&wr)[NR], thread device const ushort* const (&sr)[NR],
                             thread device const ushort* const (&br)[NR], device const ushort* xg, uint C,
                             thread const float (&xs)[KT], uint g, thread float (&tv)[NR][KT]) {
    float P[NR][KT];
    QM_UNROLL for (uint n = 0; n < NR; n++) QM_UNROLL for (uint t = 0; t < KT; t++) P[n][t] = 0.0f;
    for (uint h = 0; h < 2; h++) {
        uint4 wv[NR];
        QM_UNROLL for (uint n = 0; n < NR; n++) wv[n] = ((device const uint4*)(wr[n] + 8 * g))[h];
        QM_UNROLL for (uint j = 0; j < 4; j++) {
            uint xu[KT][4];
            QM_UNROLL for (uint t = 0; t < KT; t++) {
                uint4 xq = ((device const uint4*)(xg + t * C + 32 * h + 8 * j))[0];
                xu[t][0] = xq.x; xu[t][1] = xq.y; xu[t][2] = xq.z; xu[t][3] = xq.w;
            }
            uint wj[NR];
            QM_UNROLL for (uint n = 0; n < NR; n++) wj[n] = wv[n][j];
            QM_UNROLL for (uint k = 0; k < 8; k++) {
                float bq[NR];
                QM_UNROLL for (uint n = 0; n < NR; n++) bq[n] = float(wj[n] & (0xfu << (4 * k)));
                QM_UNROLL for (uint t = 0; t < KT; t++) {
                    float a = as_type<float>((k & 1) ? (xu[t][k / 2] & 0xffff0000u) : (xu[t][k / 2] << 16)) *
                              as_type<float>((127u - 4 * k) << 23);
                    QM_UNROLL for (uint n = 0; n < NR; n++) P[n][t] = fma(a, bq[n], P[n][t]);
                }
            }
        }
    }
    QM_UNROLL for (uint n = 0; n < NR; n++) {
        float sc = bf(sr[n][g]), bi = bf(br[n][g]);
        QM_UNROLL for (uint t = 0; t < KT; t++) tv[n][t] = fma(sc, P[n][t], xs[t] * bi);
    }
}

// NM = 1 (gemv) or 2 (swiglu: rows 0..7 gate, 8..15 up of the same outputs). Result: fin[r * KT + t] in tg mem.
template <uint KT, uint SK, uint NM, uint RP = QMT_RP, uint RT = QMT_RT>
static inline void qmt_rows(device const uint* W0, device const ushort* S0, device const ushort* B0,
                            device const uint* W1, device const ushort* S1, device const ushort* B1,
                            device const ushort* x, uint R, uint C, uint row0, uint tid, uint NT,
                            threadgroup float* T, threadgroup float* fin, device const uint* gls, uint NL, bool lst) {
    constexpr uint NR = NM * RT, NP = NR / RP;
    const uint G = C / 64, C8 = C / 8;
    // read mask: positions 0..N-1 of the group list gls (lst), else of 0..G-1 (plain kernels: N = G, g = i)
    const uint N = lst ? NL : G;
    // reducer role: thread tid < NR * SK owns (row rr, slice rk), all KT tokens
    const uint rr = tid / SK, rk = tid % SK;
    const bool red = tid < NR * SK;
    const uint ga = rk * N / SK, gb = (rk + 1) * N / SK;
    float acc[KT];
    QM_UNROLL for (uint t = 0; t < KT; t++) acc[t] = 0.0f;
    // work items: thread tid -> (position gc + tid % CG, row pass tid / CG); a chunk of CG positions covers all NP passes
    const uint CG = NT / NP, gl = tid % CG, pass = tid / CG;
    for (uint gc = 0; gc < N; gc += CG) {
        const uint gi = gc + gl;
        if (pass < NP && gi < N) {
            const uint g = lst ? gls[gi] : gi;
            device const ushort* xg = x + 64 * g;
            // xsum per token as the MMA lanes (see above)
            float xs[KT];
            QM_UNROLL for (uint t = 0; t < KT; t++) {
                float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                QM_UNROLL for (uint s = 0; s < 8; s++) {
                    uint4 v = ((device const uint4*)(xg + t * C + 8 * s))[0];
                    uint u[4] = {v.x, v.y, v.z, v.w};
                    QM_UNROLL for (uint j = 0; j < 4; j++) { p[j] += as_type<float>(u[j] << 16); p[j] += as_type<float>(u[j] & 0xffff0000u); }
                }
                xs[t] = (p[0] + p[1]) + (p[2] + p[3]);
            }
            device const uint* wr[RP];
            device const ushort* sr[RP];
            device const ushort* br[RP];
            QM_UNROLL for (uint n = 0; n < RP; n++) {
                uint r = pass * RP + n;
                bool up = NM == 2 && r >= RT;
                uint row = min(row0 + (r % RT), R - 1);
                wr[n] = (up ? W1 : W0) + row * C8; sr[n] = (up ? S1 : S0) + row * G; br[n] = (up ? B1 : B0) + row * G;
            }
            float tv[RP][KT];
            qmt_terms<RP, KT>(wr, sr, br, xg, C, xs, g, tv);
            QM_UNROLL for (uint n = 0; n < RP; n++) QM_UNROLL for (uint t = 0; t < KT; t++)
                T[((pass * RP + n) * KT + t) * CG + gl] = tv[n][t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (red) {
            uint lo = max(ga, gc), hi = min(gb, min(gc + CG, N));
            for (uint gg = lo; gg < hi; gg++) QM_UNROLL for (uint t = 0; t < KT; t++)
                acc[t] = acc[t] + T[(rr * KT + t) * CG + (gg - gc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // slices in order k = 0..SK-1 (as qm_reduce: acc_0 + acc_1, then + acc_2, ...)
    if (red) QM_UNROLL for (uint t = 0; t < KT; t++) T[(rk * NR + rr) * KT + t] = acc[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NR) QM_UNROLL for (uint t = 0; t < KT; t++) {
        float v = T[tid * KT + t];
        for (uint k = 1; k < SK; k++) v += T[(k * NR + tid) * KT + t];
        fin[tid * KT + t] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint KT, uint SK, uint MODE>
kernel void qm_gemvt(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                     device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                     device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                     device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint NT [[threads_per_threadgroup]]) {
    threadgroup float T[QMT_TMAX];
    threadgroup float fin[QMT_RT * KT];
    const uint row0 = tg * QMT_RT;
    qmt_rows<KT, SK, 1>(W, S, B, W, S, B, x, R, C, row0, tid, NT, T, fin, W, 0, false);
    const uint row = row0 + tid;
    if (tid >= QMT_RT || row >= R) return;
    QM_UNROLL for (uint t = 0; t < KT; t++) {
        float v = fin[tid * KT + t];
        uint o = t * R + row;
        if (MODE == 0) out[o] = to_bf(v);
        else if (MODE == 2) res[o] = to_bf(bf(res[o]) + qm_rbf(v));
        else outf[o] = v;
    }
}

template <uint KT, uint SK>
kernel void qm_swiglut(device const uint* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                       device const ushort* Bg [[buffer(2)]], device const uint* Wu [[buffer(3)]],
                       device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                       device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                       constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]],
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint NT [[threads_per_threadgroup]]) {
    threadgroup float T[QMT_TMAX];
    threadgroup float fin[2 * QMT_RT * KT];
    const uint row0 = tg * QMT_RT;
    qmt_rows<KT, SK, 2>(Wg, Sg, Bg, Wu, Su, Bu, x, R, C, row0, tid, NT, T, fin, Wg, 0, false);
    const uint row = row0 + tid;
    if (tid >= QMT_RT || row >= R) return;
    QM_UNROLL for (uint t = 0; t < KT; t++) out[t * R + row] = qm_silu_mul(fin[tid * KT + t], fin[(QMT_RT + tid) * KT + t]);
}

#define QMT_GEMV_T(KT_, SK_, M_) template [[host_name("qm_gemvt_k" #KT_ "_s" #SK_ "_m" #M_)]] kernel void \
    qm_gemvt<KT_, SK_, M_>(device const uint*, device const ushort*, device const ushort*, device const ushort*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, uint, uint, uint);
#define QMT_SWIGLU_T(KT_, SK_) template [[host_name("qm_swiglut_k" #KT_ "_s" #SK_)]] kernel void qm_swiglut<KT_, SK_>( \
    device const uint*, device const ushort*, device const ushort*, device const uint*, device const ushort*, \
    device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, uint, uint, uint);
#define QMT_K(KT_, SK_) QMT_GEMV_T(KT_, SK_, 0) QMT_GEMV_T(KT_, SK_, 2) QMT_GEMV_T(KT_, SK_, 3) QMT_SWIGLU_T(KT_, SK_)
#define QMT_ALL(KT_) QMT_K(KT_, 1) QMT_K(KT_, 2) QMT_K(KT_, 4) QMT_K(KT_, 8)
QMT_ALL(1) QMT_ALL(2) QMT_ALL(3) QMT_ALL(4) QMT_ALL(5) QMT_ALL(6) QMT_ALL(7) QMT_ALL(8)

// ---- qm_gemvtd / qm_swiglutd (2026-11-03, decode_bench; QwenOptions.decode_fused): the twins above with the T
// terms in DYNAMIC threadgroup memory [[threadgroup(0)]] sized by the host to what the dispatch uses,
// max(RP * NT * KT, SK * NR * KT) floats (mma_twin_tmem), instead of the fixed QMT_TMAX = 16 KB array. The K = 1
// decode shapes need 2.5-5 KB, so more threadgroups fit per core (occupancy) on these memory-bound kernels.
// Same qmt_rows code, same work split, same order: BIT-IDENTICAL to qm_gemvt / qm_swiglut of the same KT, SK.
// Names: qm_gemvtd_k{KT}_s8_m{M}, qm_swiglutd_k{KT}_s8 (KT = 1..4); buffers as the twins.
template <uint KT, uint SK, uint MODE, uint RP, uint RT>
kernel void qm_gemvtd(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                      device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                      device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                      device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]],
                      threadgroup float* T [[threadgroup(0)]],
                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint NT [[threads_per_threadgroup]]) {
    threadgroup float fin[RT * KT];
    const uint row0 = tg * RT;
    qmt_rows<KT, SK, 1, RP, RT>(W, S, B, W, S, B, x, R, C, row0, tid, NT, T, fin, W, 0, false);
    const uint row = row0 + tid;
    if (tid >= RT || row >= R) return;
    QM_UNROLL for (uint t = 0; t < KT; t++) {
        float v = fin[tid * KT + t];
        uint o = t * R + row;
        if (MODE == 0) out[o] = to_bf(v);
        else if (MODE == 2) res[o] = to_bf(bf(res[o]) + qm_rbf(v));
        else outf[o] = v;
    }
}

template <uint KT, uint SK, uint RP, uint RT>
kernel void qm_swiglutd(device const uint* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                        device const ushort* Bg [[buffer(2)]], device const uint* Wu [[buffer(3)]],
                        device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                        device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                        constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]],
                        threadgroup float* T [[threadgroup(0)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint NT [[threads_per_threadgroup]]) {
    threadgroup float fin[2 * RT * KT];
    const uint row0 = tg * RT;
    qmt_rows<KT, SK, 2, RP, RT>(Wg, Sg, Bg, Wu, Su, Bu, x, R, C, row0, tid, NT, T, fin, Wg, 0, false);
    const uint row = row0 + tid;
    if (tid >= RT || row >= R) return;
    QM_UNROLL for (uint t = 0; t < KT; t++) out[t * R + row] = qm_silu_mul(fin[tid * KT + t], fin[(RT + tid) * KT + t]);
}

// RP (rows per register pass) and RT (output rows per threadgroup) are template parameters; neither is part of the
// result (a row's group terms and their order never change). Plain names: RP 4, RT 8 (the twins' split).
#define QMTD_GEMV_T(KT_, SK_, M_, RP_, RT_, SFX_) template [[host_name("qm_gemvtd_k" #KT_ SFX_ "_s" #SK_ "_m" #M_)]] kernel void \
    qm_gemvtd<KT_, SK_, M_, RP_, RT_>(device const uint*, device const ushort*, device const ushort*, device const ushort*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, threadgroup float*, uint, uint, uint);
#define QMTD_SWIGLU_T(KT_, SK_, RP_, RT_, SFX_) template [[host_name("qm_swiglutd_k" #KT_ SFX_ "_s" #SK_)]] kernel void \
    qm_swiglutd<KT_, SK_, RP_, RT_>(device const uint*, device const ushort*, device const ushort*, device const uint*, \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, \
    threadgroup float*, uint, uint, uint);
#define QMTD_K(KT_, RP_, RT_, SFX_) QMTD_GEMV_T(KT_, 8, 0, RP_, RT_, SFX_) QMTD_GEMV_T(KT_, 8, 2, RP_, RT_, SFX_) \
    QMTD_GEMV_T(KT_, 8, 3, RP_, RT_, SFX_) QMTD_SWIGLU_T(KT_, 8, RP_, RT_, SFX_)
QMTD_K(1, 4, 8, "") QMTD_K(2, 4, 8, "") QMTD_K(3, 4, 8, "") QMTD_K(4, 4, 8, "")

// _t4: RT 4, K = 1 (decode_bench sweep, 27B: ffn.down 200 -> 219 GB/s, gate+up 232 -> 240-247; one chunk, NT = G
// for gemv, 2 G for swiglu). The engine uses it for swiglu and for gemv with cols > 8192 (mmad_twin).
QMTD_K(1, 4, 4, "_t4")

// ---- READ MASK (streamdense-model readmask.rs): masked variants of the engine's kernels (qm_gemv n2 s8,
// qm_swiglu n1 s8, the twins k1..4 s8), same buffers as their plain kernel plus:
//   rb [10] (swiglu [11]): row tiles, 1 bit per 8 weight rows (bit t of word t / 32 = rows 8t .. 8t + 7);
//   cl [11]: input 64-groups to read, cl[0] = n, cl[1 .. n] ascending;
//   fl [12]: fl[0] != 0 = "read everything" for this pass (unsure token);
//   mf [13] (gemv only): bit 0 = rows masked by rb, bit 1 = columns restricted to cl (swiglu: rows always).
// Rows: a threadgroup whose row tiles are all off reads no weights and writes 0 to its outputs (MODE 0 / 3, swiglu:
// the masked neurons are inactive, a = 0) or nothing (MODE 2, the residual add); a tile with any row on is computed
// in full. Columns: the split-K slices run over list POSITIONS, slice k = [k n / SK, (k + 1) n / SK) in order, so
// skipped groups contribute exactly nothing. With fl[0] != 0, or mf = 0, or the all-groups list 0..G-1 with every
// tile on, every quantity is the plain kernel's (same groups, same slices, same order): BIT-IDENTICAL output.
static inline bool qm_on(device const uint* rb, uint t) { return ((rb[t >> 5] >> (t & 31u)) & 1u) != 0u; }

template <uint NB, uint SK, uint MODE>
kernel void qm_gemvx(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                     device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                     device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                     device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]], constant uint& K [[buffer(9)]],
                     device const uint* rb [[buffer(10)]], device const uint* cl [[buffer(11)]],
                     device const uint* fl [[buffer(12)]], constant uint& mf [[buffer(13)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(SK > 1 ? SK - 1 : 1) * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    const bool full = fl[0] != 0u;
    if (!full && (mf & 1u) != 0u) {
        bool on = false;
        QM_UNROLL for (uint b = 0; b < NB; b++) on = on || (row0 + 8 * b < R && qm_on(rb, row0 / 8 + b));
        if (!on) {
            if (MODE != 2) for (uint e = sg * 32 + lane; e < 8 * NB * K; e += 32 * SK) {
                uint row = row0 + e % (8 * NB), t = e / (8 * NB);
                if (row < R) { if (MODE == 0) out[t * R + row] = 0; else outf[t * R + row] = 0.0f; }
            }
            return;
        }
    }
    const bool lst = !full && (mf & 2u) != 0u;
    const uint fm = qm_fm(lane), fn = qm_fn(lane), C8 = C / 8, G = C / 64, N = lst ? cl[0] : G;
    device const uint* wr[NB][2];
    device const ushort* sr[NB][2];
    device const ushort* br[NB][2];
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = min(row0 + 8 * b + fn + c, R - 1);
        wr[b][c] = W + row * C8; sr[b][c] = S + row * G; br[b][c] = B + row * G;
    }
    float acc[NB][2];
    qm_dot<NB>(wr, sr, br, x, K, C, sg * N / SK, (sg + 1) * N / SK, lane, acc, cl + 1, lst);
    qm_reduce<NB, SK>(acc, red, sg, lane);
    if (sg != 0 || fm >= K) return;
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = row0 + 8 * b + fn + c;
        if (row >= R) continue;
        uint o = fm * R + row;
        float v = acc[b][c];
        if (MODE == 0) out[o] = to_bf(v);
        else if (MODE == 2) res[o] = to_bf(bf(res[o]) + qm_rbf(v));
        else outf[o] = v;
    }
}

template <uint NB, uint SK>
kernel void qm_swiglux(device const uint* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                       device const ushort* Bg [[buffer(2)]], device const uint* Wu [[buffer(3)]],
                       device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                       device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                       constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& K [[buffer(10)]],
                       device const uint* rb [[buffer(11)]], device const uint* fl [[buffer(12)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(SK > 1 ? SK - 1 : 1) * 2 * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    if (fl[0] == 0u) {
        bool on = false;
        QM_UNROLL for (uint b = 0; b < NB; b++) on = on || (row0 + 8 * b < R && qm_on(rb, row0 / 8 + b));
        if (!on) {
            for (uint e = sg * 32 + lane; e < 8 * NB * K; e += 32 * SK) {
                uint row = row0 + e % (8 * NB), t = e / (8 * NB);
                if (row < R) out[t * R + row] = 0;
            }
            return;
        }
    }
    const uint fm = qm_fm(lane), fn = qm_fn(lane), C8 = C / 8, G = C / 64;
    device const uint* wr[2 * NB][2];
    device const ushort* sr[2 * NB][2];
    device const ushort* br[2 * NB][2];
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = min(row0 + 8 * b + fn + c, R - 1);
        wr[b][c] = Wg + row * C8; sr[b][c] = Sg + row * G; br[b][c] = Bg + row * G;
        wr[NB + b][c] = Wu + row * C8; sr[NB + b][c] = Su + row * G; br[NB + b][c] = Bu + row * G;
    }
    float acc[2 * NB][2];
    qm_dot<2 * NB>(wr, sr, br, x, K, C, sg * G / SK, (sg + 1) * G / SK, lane, acc, Wg, false);
    qm_reduce<2 * NB, SK>(acc, red, sg, lane);
    if (sg != 0 || fm >= K) return;
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = row0 + 8 * b + fn + c;
        if (row < R) out[fm * R + row] = qm_silu_mul(acc[b][c], acc[NB + b][c]);
    }
}

template <uint KT, uint SK, uint MODE>
kernel void qm_gemvtx(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                      device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                      device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                      device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]],
                      device const uint* rb [[buffer(10)]], device const uint* cl [[buffer(11)]],
                      device const uint* fl [[buffer(12)]], constant uint& mf [[buffer(13)]],
                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint NT [[threads_per_threadgroup]]) {
    threadgroup float T[QMT_TMAX];
    threadgroup float fin[QMT_RT * KT];
    const uint row0 = tg * QMT_RT;
    const bool full = fl[0] != 0u;
    if (!full && (mf & 1u) != 0u && !qm_on(rb, tg)) {
        if (MODE != 2) for (uint e = tid; e < QMT_RT * KT; e += NT) {
            uint row = row0 + e % QMT_RT, t = e / QMT_RT;
            if (row < R) { if (MODE == 0) out[t * R + row] = 0; else outf[t * R + row] = 0.0f; }
        }
        return;
    }
    const bool lst = !full && (mf & 2u) != 0u;
    qmt_rows<KT, SK, 1>(W, S, B, W, S, B, x, R, C, row0, tid, NT, T, fin, cl + 1, lst ? cl[0] : 0u, lst);
    const uint row = row0 + tid;
    if (tid >= QMT_RT || row >= R) return;
    QM_UNROLL for (uint t = 0; t < KT; t++) {
        float v = fin[tid * KT + t];
        uint o = t * R + row;
        if (MODE == 0) out[o] = to_bf(v);
        else if (MODE == 2) res[o] = to_bf(bf(res[o]) + qm_rbf(v));
        else outf[o] = v;
    }
}

template <uint KT, uint SK>
kernel void qm_swiglutx(device const uint* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                        device const ushort* Bg [[buffer(2)]], device const uint* Wu [[buffer(3)]],
                        device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                        device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                        constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]],
                        device const uint* rb [[buffer(11)]], device const uint* fl [[buffer(12)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint NT [[threads_per_threadgroup]]) {
    threadgroup float T[QMT_TMAX];
    threadgroup float fin[2 * QMT_RT * KT];
    const uint row0 = tg * QMT_RT;
    if (fl[0] == 0u && !qm_on(rb, tg)) {
        for (uint e = tid; e < QMT_RT * KT; e += NT) {
            uint row = row0 + e % QMT_RT, t = e / QMT_RT;
            if (row < R) out[t * R + row] = 0;
        }
        return;
    }
    qmt_rows<KT, SK, 2>(Wg, Sg, Bg, Wu, Su, Bu, x, R, C, row0, tid, NT, T, fin, Wg, 0, false);
    const uint row = row0 + tid;
    if (tid >= QMT_RT || row >= R) return;
    QM_UNROLL for (uint t = 0; t < KT; t++) out[t * R + row] = qm_silu_mul(fin[tid * KT + t], fin[(QMT_RT + tid) * KT + t]);
}

#define QMX_GEMV_T(NB_, SK_, M_) template [[host_name("qm_gemvx_n" #NB_ "_s" #SK_ "_m" #M_)]] kernel void \
    qm_gemvx<NB_, SK_, M_>(device const uint*, device const ushort*, device const ushort*, device const ushort*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, constant uint&, device const uint*, \
    device const uint*, device const uint*, constant uint&, uint, uint, uint);
#define QMX_SWIGLU_T(NB_, SK_) template [[host_name("qm_swiglux_n" #NB_ "_s" #SK_)]] kernel void qm_swiglux<NB_, SK_>( \
    device const uint*, device const ushort*, device const ushort*, device const uint*, device const ushort*, \
    device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, constant uint&, \
    device const uint*, device const uint*, uint, uint, uint);
#define QMTX_GEMV_T(KT_, SK_, M_) template [[host_name("qm_gemvtx_k" #KT_ "_s" #SK_ "_m" #M_)]] kernel void \
    qm_gemvtx<KT_, SK_, M_>(device const uint*, device const ushort*, device const ushort*, device const ushort*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, device const uint*, \
    device const uint*, device const uint*, constant uint&, uint, uint, uint);
#define QMTX_SWIGLU_T(KT_, SK_) template [[host_name("qm_swiglutx_k" #KT_ "_s" #SK_)]] kernel void \
    qm_swiglutx<KT_, SK_>(device const uint*, device const ushort*, device const ushort*, device const uint*, \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, \
    device const uint*, device const uint*, uint, uint, uint);
#define QMTX_K(KT_) QMTX_GEMV_T(KT_, 8, 0) QMTX_GEMV_T(KT_, 8, 2) QMTX_GEMV_T(KT_, 8, 3) QMTX_SWIGLU_T(KT_, 8)
QMX_GEMV_T(2, 8, 0) QMX_GEMV_T(2, 8, 2) QMX_GEMV_T(2, 8, 3) QMX_SWIGLU_T(1, 8)
QMTX_K(1) QMTX_K(2) QMTX_K(3) QMTX_K(4)
