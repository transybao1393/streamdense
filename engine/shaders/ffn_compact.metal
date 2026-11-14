// ffn_compact.metal - forheader step 3 (streamdense-model ffn_compact.rs; only with QwenOptions.ffn_compact, off by
// default). FFN down of a COMPACT layer: kc kept neurons, the down matrix repacked per kept neuron (column) with the
// ORIGINAL per-(row, 64-group) scales / biases. Needs common_bf16 (bf, to_bf) first.
//   W u32 [R][kc/8] (nibble i of row r = original column keep[i], low nibble first), S / B bf16 [R][G] (G = inter/64),
//   gid u16 [kc] (original group of kept column i), gstart u32 [G+1], f bf16 [k][kc] (swiglu of the kept rows).
//   y_r = sum_g S[r][g] * sum_{i: gid[i] = g} f_i * q_ri + B[r][g] * fs[g],  fs[g] = sum_{i: gid[i] = g} f_i.
//   h[t][r] = bf(bf(h) + bf(y))  (MODE 2 of the q4 GEMVs: residual add in place).
// fp32 math; the order differs from the full kernels (which sum zeroed neurons too), so results are close, not equal.

// fs[t][g] = sum of f over group g's kept columns. (G, k) groups x 32 threads.
kernel void cd_fsum(device const ushort* f [[buffer(0)]], device const uint* gstart [[buffer(1)]], device float* fs [[buffer(2)]],
                    constant uint& kc [[buffer(3)]], constant uint& G [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint g = tg.x, t = tg.y;
    uint a = gstart[g], b = gstart[g + 1];
    float s = 0.0f;
    for (uint j = a + lane; j < b; j += 32u) s += bf(f[t * kc + j]);
    s = simd_sum(s);
    if (lane == 0) fs[t * G + g] = s;
}

// cd_down_r{NR}: NR rows per simdgroup, 2 simdgroups per threadgroup: ceil(R / (2 NR)) groups x 64 threads. Lanes take
// single 32-bit words (8 kept columns), strided by 32 (coalesced). BRANCH-FREE per word (a lane's consecutive words are
// 32 words apart, i.e. in other groups, and ~1 word in 6 straddles a group boundary, so a running sum + per-nibble
// path would diverge on almost every simdgroup iteration): wmeta[word] = g0 | g7 << 12 | p << 24 | complex << 28 (g0 /
// g7 = groups of the word's first / last column, p = columns in g0, complex = 3+ groups: rare per-nibble path). Per
// word the f side (bf16 -> f32, pre-divided by 16^j as MLX's qdot so nibbles need no shift; split at p into xa / xb)
// is computed once and shared by the NR rows; per row y += S[r][g0] * (xa . q) + S[r][g7] * (xb . q).
template <uint NR>
static inline void cd_down_impl(device const uint* W, device const ushort* S, device const ushort* B, device const ushort* f,
                                device const ushort* gid, device const uint* wmeta, device const float* fs, device ushort* h,
                                uint R, uint kc, uint G, uint k, uint tg, uint sg, uint lane) {
    uint r0 = (tg * 2u + sg) * NR;
    if (r0 >= R) return;                          // uniform per simdgroup
    uint nw = kc / 8u;
    uint rr[NR];
    for (uint n = 0; n < NR; n++) rr[n] = min(r0 + n, R - 1u);
    for (uint t = 0; t < k; t++) {
        device const ushort* ft = f + t * kc;
        float y[NR];
        for (uint n = 0; n < NR; n++) y[n] = 0.0f;
        for (uint wi = lane; wi < nw; wi += 32u) {
            uint c0 = wi * 8u;
            uint wv[NR];
            for (uint n = 0; n < NR; n++) wv[n] = W[rr[n] * nw + wi];
            ushort4 fa = *(device const ushort4*)(ft + c0);
            ushort4 fb = *(device const ushort4*)(ft + c0 + 4u);
            float xs[8] = { bf(fa.x), bf(fa.y) / 16.0f, bf(fa.z) / 256.0f, bf(fa.w) / 4096.0f,
                            bf(fb.x), bf(fb.y) / 16.0f, bf(fb.z) / 256.0f, bf(fb.w) / 4096.0f };
            uint m = wmeta[wi];
            if ((m >> 28) == 0u) {
                uint g0 = m & 0xfffu, g7 = (m >> 12) & 0xfffu, p = (m >> 24) & 0xfu;
                float xa[8], xb[8];
                for (uint i = 0; i < 8u; i++) { xa[i] = i < p ? xs[i] : 0.0f; xb[i] = xs[i] - xa[i]; }
                for (uint n = 0; n < NR; n++) {
                    uint w = wv[n];
                    float q[8] = { float(w & 0x000fu), float(w & 0x00f0u), float(w & 0x0f00u), float(w & 0xf000u),
                                   float((w >> 16) & 0x000fu), float((w >> 16) & 0x00f0u), float((w >> 16) & 0x0f00u), float((w >> 16) & 0xf000u) };
                    float da = 0.0f, db = 0.0f;
                    for (uint i = 0; i < 8u; i++) { da += xa[i] * q[i]; db += xb[i] * q[i]; }
                    y[n] += bf(S[rr[n] * G + g0]) * da + bf(S[rr[n] * G + g7]) * db;
                }
            } else {                              // 3+ groups in this word (a group with < 8 kept columns): rare
                for (uint i = 0; i < 8u; i++) {
                    uint g = gid[c0 + i];
                    uint sh = 16u * (i / 4u), mk = 0xfu << (4u * (i % 4u));
                    for (uint n = 0; n < NR; n++) y[n] += bf(S[rr[n] * G + g]) * (xs[i] * float((wv[n] >> sh) & mk));
                }
            }
        }
        device const float* fst = fs + t * G;
        for (uint g = lane; g < G; g += 32u) {
            float x = fst[g];
            for (uint n = 0; n < NR; n++) y[n] += bf(B[rr[n] * G + g]) * x;
        }
        for (uint n = 0; n < NR; n++) {
            float v = simd_sum(y[n]);
            if (lane == 0 && r0 + n < R) {
                uint o = t * R + r0 + n;
                h[o] = to_bf(bf(h[o]) + bf(to_bf(v)));
            }
        }
    }
}

#define CD_DOWN(NAME, NR) \
kernel void NAME(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]], device const ushort* B [[buffer(2)]], \
                 device const ushort* f [[buffer(3)]], device const ushort* gid [[buffer(4)]], device const float* fs [[buffer(5)]], \
                 device ushort* h [[buffer(6)]], constant uint& R [[buffer(7)]], constant uint& kc [[buffer(8)]], \
                 constant uint& G [[buffer(9)]], constant uint& k [[buffer(10)]], device const uint* wmeta [[buffer(11)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { cd_down_impl<NR>(W, S, B, f, gid, wmeta, fs, h, R, kc, G, k, tg, sg, lane); }
CD_DOWN(cd_down_r4, 4u)
CD_DOWN(cd_down_r8, 8u)

// cd_downk_r4_k{KT} (2026-11-02, compact + MTP verify): the multi-row twin of cd_down_r4 for KT = 1..8 token rows.
// Each lane loads its W words (and the 2 scales of each row) ONCE and applies them to all KT tokens, so the down
// matrix is read once per pass instead of once per row (cd_down_* loop the tokens outside the word loop). Per
// (row, token) the arithmetic is cd_down_r4's, in the same order (same word sequence per lane, same expressions,
// Safe math: no reassociation), so the result of a row never depends on KT or on the other rows: a K-row verify
// pass is bit-identical per row to K = 1 decode (the losslessness basis of spec_deep.rs). The engine uses this
// family for every compact pass (KT = k).
template <uint NR, uint KT>
static inline void cd_downk_impl(device const uint* W, device const ushort* S, device const ushort* B, device const ushort* f,
                                 device const ushort* gid, device const uint* wmeta, device const float* fs, device ushort* h,
                                 uint R, uint kc, uint G, uint tg, uint sg, uint lane) {
    uint r0 = (tg * 2u + sg) * NR;
    if (r0 >= R) return;                          // uniform per simdgroup
    uint nw = kc / 8u;
    uint rr[NR];
    for (uint n = 0; n < NR; n++) rr[n] = min(r0 + n, R - 1u);
    float y[KT][NR];
    for (uint t = 0; t < KT; t++) for (uint n = 0; n < NR; n++) y[t][n] = 0.0f;
    for (uint wi = lane; wi < nw; wi += 32u) {
        uint c0 = wi * 8u;
        uint wv[NR];
        for (uint n = 0; n < NR; n++) wv[n] = W[rr[n] * nw + wi];
        uint m = wmeta[wi];
        if ((m >> 28) == 0u) {
            uint g0 = m & 0xfffu, g7 = (m >> 12) & 0xfffu, p = (m >> 24) & 0xfu;
            float s0[NR], s7[NR];
            for (uint n = 0; n < NR; n++) { s0[n] = bf(S[rr[n] * G + g0]); s7[n] = bf(S[rr[n] * G + g7]); }
            for (uint t = 0; t < KT; t++) {
                device const ushort* ft = f + t * kc;
                ushort4 fa = *(device const ushort4*)(ft + c0);
                ushort4 fb = *(device const ushort4*)(ft + c0 + 4u);
                float xs[8] = { bf(fa.x), bf(fa.y) / 16.0f, bf(fa.z) / 256.0f, bf(fa.w) / 4096.0f,
                                bf(fb.x), bf(fb.y) / 16.0f, bf(fb.z) / 256.0f, bf(fb.w) / 4096.0f };
                float xa[8], xb[8];
                for (uint i = 0; i < 8u; i++) { xa[i] = i < p ? xs[i] : 0.0f; xb[i] = xs[i] - xa[i]; }
                for (uint n = 0; n < NR; n++) {
                    uint w = wv[n];
                    float q[8] = { float(w & 0x000fu), float(w & 0x00f0u), float(w & 0x0f00u), float(w & 0xf000u),
                                   float((w >> 16) & 0x000fu), float((w >> 16) & 0x00f0u), float((w >> 16) & 0x0f00u), float((w >> 16) & 0xf000u) };
                    float da = 0.0f, db = 0.0f;
                    for (uint i = 0; i < 8u; i++) { da += xa[i] * q[i]; db += xb[i] * q[i]; }
                    y[t][n] += s0[n] * da + s7[n] * db;
                }
            }
        } else {                                  // 3+ groups in this word: rare per-nibble path (as cd_down_r4)
            for (uint t = 0; t < KT; t++) {
                device const ushort* ft = f + t * kc;
                ushort4 fa = *(device const ushort4*)(ft + c0);
                ushort4 fb = *(device const ushort4*)(ft + c0 + 4u);
                float xs[8] = { bf(fa.x), bf(fa.y) / 16.0f, bf(fa.z) / 256.0f, bf(fa.w) / 4096.0f,
                                bf(fb.x), bf(fb.y) / 16.0f, bf(fb.z) / 256.0f, bf(fb.w) / 4096.0f };
                for (uint i = 0; i < 8u; i++) {
                    uint g = gid[c0 + i];
                    uint sh = 16u * (i / 4u), mk = 0xfu << (4u * (i % 4u));
                    for (uint n = 0; n < NR; n++) y[t][n] += bf(S[rr[n] * G + g]) * (xs[i] * float((wv[n] >> sh) & mk));
                }
            }
        }
    }
    for (uint t = 0; t < KT; t++) {
        device const float* fst = fs + t * G;
        for (uint g = lane; g < G; g += 32u) {
            float x = fst[g];
            for (uint n = 0; n < NR; n++) y[t][n] += bf(B[rr[n] * G + g]) * x;
        }
        for (uint n = 0; n < NR; n++) {
            float v = simd_sum(y[t][n]);
            if (lane == 0 && r0 + n < R) {
                uint o = t * R + r0 + n;
                h[o] = to_bf(bf(h[o]) + bf(to_bf(v)));
            }
        }
    }
}

#define CD_DOWNK(NAME, NR, KT) \
kernel void NAME(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]], device const ushort* B [[buffer(2)]], \
                 device const ushort* f [[buffer(3)]], device const ushort* gid [[buffer(4)]], device const float* fs [[buffer(5)]], \
                 device ushort* h [[buffer(6)]], constant uint& R [[buffer(7)]], constant uint& kc [[buffer(8)]], \
                 constant uint& G [[buffer(9)]], constant uint& k [[buffer(10)]], device const uint* wmeta [[buffer(11)]], \
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { cd_downk_impl<NR, KT>(W, S, B, f, gid, wmeta, fs, h, R, kc, G, tg, sg, lane); }
CD_DOWNK(cd_downk_r4_k1, 4u, 1u) CD_DOWNK(cd_downk_r4_k2, 4u, 2u) CD_DOWNK(cd_downk_r4_k3, 4u, 3u) CD_DOWNK(cd_downk_r4_k4, 4u, 4u)
CD_DOWNK(cd_downk_r4_k5, 4u, 5u) CD_DOWNK(cd_downk_r4_k6, 4u, 6u) CD_DOWNK(cd_downk_r4_k7, 4u, 7u) CD_DOWNK(cd_downk_r4_k8, 4u, 8u)

// ---- cd_dm (simdgroup MMA, K = 1..8 rows) and cd_dt (scalar K = 1 twin): the compact down for verify passes.
// ONE arithmetic definition per (output row r, token t), split-K over SK word slices [s nw / SK, (s + 1) nw / SK):
//   walk the slice's words in order; a "run" = consecutive kept columns of one 64-group g. Per run
//     P = 0; for each word of the run (columns in order, masked to the run): for k = 0..7:
//         P = fma(m_k ? x[t][8w + k] * 2^-4k : 0, float(w_rk & 0xf << 4k), P)      (= x * q, exact scalings)
//     acc = acc + fma(S[r][g], P, first ? fs[t][g] * B[r][g] : 0)    first = the group's first kept column is in
//                                                                       this slice (its bias counted once)
//   v = acc_0 + acc_1 + ... + acc_{SK-1}; h[t][r] = bf(bf(h) + bf(v)).
// cd_dm: an 8x8x8 float MMA is the sequential fma chain over k per element (gemv_mma_q4.metal header, measured on
// M4 Pro), so a run's MMAs give exactly P; rows of A >= K are zero padding, so a row's v never depends on K. cd_dt
// computes the same chain per thread for K = 1 (decode at GEMV speed). Both need cd_fsum's fs. Run boundaries come
// from wmeta (uniform per simdgroup: every lane walks the same word, so no divergence); 3+ group words use gid.
static inline void cd_flush1(thread float& acc, float P, device const ushort* S, device const ushort* B, uint rowG, uint g,
                             device const uint* gstart, uint cb, device const float* fst) {
    float bt = gstart[g] >= cb ? fst[g] * bf(B[rowG + g]) : 0.0f;
    acc = acc + fma(bf(S[rowG + g]), P, bt);
}

// Slice bounds (shared by cd_dm and cd_dt: part of the arithmetic): in 4-word units, the last slice takes the tail.
static inline uint2 cd_slice(uint nw, uint sg, uint SK) {
    uint n4 = nw / 4u;
    return uint2(4u * (sg * n4 / SK), sg + 1u == SK ? nw : 4u * ((sg + 1u) * n4 / SK));
}

template <uint NB>
struct CdDm {
    simdgroup_float8x8 P[NB];
    float acc[NB][2];
    uint rows[NB][2];
    uint cur;
};

template <uint NB>
static inline void cd_dm_flush(thread CdDm<NB>& st, uint g, device const ushort* S, device const ushort* B,
                               device const uint* gstart, uint cb, device const float* fst, uint G) {
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        thread auto& pe = st.P[b].thread_elements();
        QM_UNROLL for (uint c = 0; c < 2; c++) cd_flush1(st.acc[b][c], pe[c], S, B, st.rows[b][c] * G, g, gstart, cb, fst);
        st.P[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
}

// one word: x pair (a0, a1) of the lane, weights wq[b][c] (the word of row (b, c)), wmeta m, word index w
template <uint NB>
static inline void cd_dm_word(thread CdDm<NB>& st, float a0, float a1, thread const uint (&wq)[NB][2], uint m, uint w, uint fn, uint mk,
                              device const ushort* S, device const ushort* B, device const ushort* gid, device const uint* gstart,
                              uint cb, device const float* fst, uint G) {
    simdgroup_float8x8 Bm[NB];
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        Bm[b].thread_elements()[0] = float(wq[b][0] & mk);
        Bm[b].thread_elements()[1] = float(wq[b][1] & mk);
    }
    if ((m >> 28) == 0u) {
        uint g0 = m & 0xfffu, g7 = (m >> 12) & 0xfffu, p = (m >> 24) & 0xfu;
        if (g0 != st.cur) { if (st.cur != 0xffffffffu) cd_dm_flush<NB>(st, st.cur, S, B, gstart, cb, fst, G); st.cur = g0; }
        simdgroup_float8x8 A;
        A.thread_elements()[0] = fn < p ? a0 : 0.0f;
        A.thread_elements()[1] = fn + 1 < p ? a1 : 0.0f;
        QM_UNROLL for (uint b = 0; b < NB; b++) simdgroup_multiply_accumulate(st.P[b], A, Bm[b], st.P[b]);
        if (p < 8u) {
            cd_dm_flush<NB>(st, st.cur, S, B, gstart, cb, fst, G); st.cur = g7;
            A.thread_elements()[0] = fn < p ? 0.0f : a0;
            A.thread_elements()[1] = fn + 1 < p ? 0.0f : a1;
            QM_UNROLL for (uint b = 0; b < NB; b++) simdgroup_multiply_accumulate(st.P[b], A, Bm[b], st.P[b]);
        }
    } else {
        uint i0 = 0;
        while (i0 < 8u) {
            uint g = gid[8u * w + i0], i1 = i0 + 1;
            while (i1 < 8u && gid[8u * w + i1] == g) i1++;
            if (g != st.cur) { if (st.cur != 0xffffffffu) cd_dm_flush<NB>(st, st.cur, S, B, gstart, cb, fst, G); st.cur = g; }
            simdgroup_float8x8 A;
            A.thread_elements()[0] = (fn >= i0 && fn < i1) ? a0 : 0.0f;
            A.thread_elements()[1] = (fn + 1 >= i0 && fn + 1 < i1) ? a1 : 0.0f;
            QM_UNROLL for (uint b = 0; b < NB; b++) simdgroup_multiply_accumulate(st.P[b], A, Bm[b], st.P[b]);
            i0 = i1;
        }
    }
}

template <uint NB, uint SK>
kernel void cd_dm(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]], device const ushort* B [[buffer(2)]],
                  device const ushort* f [[buffer(3)]], device const ushort* gid [[buffer(4)]], device const float* fs [[buffer(5)]],
                  device ushort* h [[buffer(6)]], constant uint& R [[buffer(7)]], constant uint& kc [[buffer(8)]],
                  constant uint& G [[buffer(9)]], constant uint& K [[buffer(10)]], device const uint* wmeta [[buffer(11)]],
                  device const uint* gstart [[buffer(12)]],
                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[(SK > 1 ? SK - 1 : 1) * NB * 2 * 32];
    const uint row0 = tg * 8 * NB;
    const uint fm = qm_fm(lane), fn = qm_fn(lane), nw = kc / 8u;
    const bool xv = fm < K;
    const uint tk = min(fm, K - 1);
    device const ushort* xr = f + tk * kc + fn;
    device const float* fst = fs + tk * G;
    const float sa0 = as_type<float>((127u - 4 * fn) << 23), sa1 = as_type<float>((127u - 4 * (fn + 1)) << 23);
    const uint mk = 0xfu << (4 * fm);
    CdDm<NB> st;
    device const uint* wr[NB][2];
    QM_UNROLL for (uint b = 0; b < NB; b++) {
        QM_UNROLL for (uint c = 0; c < 2; c++) { st.rows[b][c] = min(row0 + 8 * b + fn + c, R - 1); wr[b][c] = W + st.rows[b][c] * nw; }
        st.acc[b][0] = 0.0f; st.acc[b][1] = 0.0f;
        st.P[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    st.cur = 0xffffffffu;
    const uint2 sl = cd_slice(nw, sg, SK);
    const uint cb = sl.x * 8u;
    const bool vec = (nw % 4u) == 0u;     // uint4 rows (16 B aligned): 4 words per step
    uint w = sl.x;
    if (vec) {
        for (; w + 4u <= sl.y; w += 4u) {
            uint4 mm = *(device const uint4*)(wmeta + w);
            uint4 wv[NB][2];
            QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) wv[b][c] = *(device const uint4*)(wr[b][c] + w);
            QM_UNROLL for (uint j = 0; j < 4u; j++) {
                uint u = xv ? *(device const uint*)(xr + 8u * (w + j)) : 0u;
                uint wq[NB][2];
                QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) wq[b][c] = wv[b][c][j];
                cd_dm_word<NB>(st, as_type<float>(u << 16) * sa0, as_type<float>(u & 0xffff0000u) * sa1, wq, mm[j], w + j, fn, mk,
                               S, B, gid, gstart, cb, fst, G);
            }
        }
    }
    for (; w < sl.y; w++) {
        uint u = xv ? *(device const uint*)(xr + 8u * w) : 0u;
        uint wq[NB][2];
        QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) wq[b][c] = wr[b][c][w];
        cd_dm_word<NB>(st, as_type<float>(u << 16) * sa0, as_type<float>(u & 0xffff0000u) * sa1, wq, wmeta[w], w, fn, mk,
                       S, B, gid, gstart, cb, fst, G);
    }
    if (st.cur != 0xffffffffu) cd_dm_flush<NB>(st, st.cur, S, B, gstart, cb, fst, G);
    qm_reduce<NB, SK>(st.acc, red, sg, lane);
    if (sg != 0 || fm >= K) return;
    QM_UNROLL for (uint b = 0; b < NB; b++) QM_UNROLL for (uint c = 0; c < 2; c++) {
        uint row = row0 + 8 * b + fn + c;
        if (row >= R) continue;
        uint o = fm * R + row;
        h[o] = to_bf(bf(h[o]) + qm_rbf(st.acc[b][c]));
    }
}

static inline void cd_dt_word(thread float& acc, thread float& P, thread uint& cur, uint wv, uint m, uint w, device const ushort* f,
                              device const ushort* S, device const ushort* B, device const ushort* gid, device const uint* gstart,
                              uint rowG, uint cb, device const float* fs) {
    uint4 fx = *(device const uint4*)(f + 8u * w);
    uint fu[4] = { fx.x, fx.y, fx.z, fx.w };
    float a[8], q[8];
    for (uint j = 0; j < 4u; j++) {
        a[2 * j] = as_type<float>(fu[j] << 16) * as_type<float>((127u - 8u * j) << 23);
        a[2 * j + 1] = as_type<float>(fu[j] & 0xffff0000u) * as_type<float>((127u - 4u * (2u * j + 1u)) << 23);
    }
    for (uint k = 0; k < 8u; k++) q[k] = float(wv & (0xfu << (4u * k)));
    if ((m >> 28) == 0u) {
        uint g0 = m & 0xfffu, g7 = (m >> 12) & 0xfffu, p = (m >> 24) & 0xfu;
        if (g0 != cur) { if (cur != 0xffffffffu) { cd_flush1(acc, P, S, B, rowG, cur, gstart, cb, fs); P = 0.0f; } cur = g0; }
        for (uint k = 0; k < 8u; k++) P = fma(k < p ? a[k] : 0.0f, q[k], P);
        if (p < 8u) {
            cd_flush1(acc, P, S, B, rowG, cur, gstart, cb, fs); P = 0.0f; cur = g7;
            for (uint k = 0; k < 8u; k++) P = fma(k < p ? 0.0f : a[k], q[k], P);
        }
    } else {
        uint i0 = 0;
        while (i0 < 8u) {
            uint g = gid[8u * w + i0], i1 = i0 + 1;
            while (i1 < 8u && gid[8u * w + i1] == g) i1++;
            if (g != cur) { if (cur != 0xffffffffu) { cd_flush1(acc, P, S, B, rowG, cur, gstart, cb, fs); P = 0.0f; } cur = g; }
            for (uint k = 0; k < 8u; k++) P = fma((k >= i0 && k < i1) ? a[k] : 0.0f, q[k], P);
            i0 = i1;
        }
    }
}

// Scalar twin for K = 1: thread = (row, slice); threadgroup = 32 rows x SK simdgroups (slice = simdgroup), partials
// added in slice order (as qm_reduce: acc_0 + acc_1 + ...). Same chain as cd_dm per element.
template <uint SK>
kernel void cd_dt(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]], device const ushort* B [[buffer(2)]],
                  device const ushort* f [[buffer(3)]], device const ushort* gid [[buffer(4)]], device const float* fs [[buffer(5)]],
                  device ushort* h [[buffer(6)]], constant uint& R [[buffer(7)]], constant uint& kc [[buffer(8)]],
                  constant uint& G [[buffer(9)]], constant uint& K [[buffer(10)]], device const uint* wmeta [[buffer(11)]],
                  device const uint* gstart [[buffer(12)]],
                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[SK * 32];
    const uint nw = kc / 8u;
    const uint row = min(tg * 32u + lane, R - 1u), rowG = row * G;
    const uint2 sl = cd_slice(nw, sg, SK);
    const uint wb = sl.x, we = sl.y, cb = wb * 8u;
    device const uint* wr = W + row * nw;
    float acc = 0.0f, P = 0.0f;
    uint cur = 0xffffffffu;
    uint w = wb;
    if ((nw % 4u) == 0u) {
        for (; w + 4u <= we; w += 4u) {
            uint4 wv4 = *(device const uint4*)(wr + w);
            uint4 mm = *(device const uint4*)(wmeta + w);
            for (uint j = 0; j < 4u; j++) cd_dt_word(acc, P, cur, wv4[j], mm[j], w + j, f, S, B, gid, gstart, rowG, cb, fs);
        }
    }
    for (; w < we; w++) cd_dt_word(acc, P, cur, wr[w], wmeta[w], w, f, S, B, gid, gstart, rowG, cb, fs);
    if (cur != 0xffffffffu) cd_flush1(acc, P, S, B, rowG, cur, gstart, cb, fs);
    if (SK > 1) {
        red[sg * 32u + lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0) return;
        for (uint s = 1; s < SK; s++) acc += red[s * 32u + lane];
    }
    uint r = tg * 32u + lane;
    if (r < R) h[r] = to_bf(bf(h[r]) + qm_rbf(acc));
}

#define CD_DM_T(NB_, SK_) template [[host_name("cd_dm_n" #NB_ "_s" #SK_)]] kernel void cd_dm<NB_, SK_>( \
    device const uint*, device const ushort*, device const ushort*, device const ushort*, device const ushort*, \
    device const float*, device ushort*, constant uint&, constant uint&, constant uint&, constant uint&, \
    device const uint*, device const uint*, uint, uint, uint);
#define CD_DT_T(SK_) template [[host_name("cd_dt_s" #SK_)]] kernel void cd_dt<SK_>( \
    device const uint*, device const ushort*, device const ushort*, device const ushort*, device const ushort*, \
    device const float*, device ushort*, constant uint&, constant uint&, constant uint&, constant uint&, \
    device const uint*, device const uint*, uint, uint, uint);
CD_DM_T(1, 1) CD_DM_T(1, 2) CD_DM_T(1, 4) CD_DM_T(2, 1) CD_DM_T(2, 2) CD_DM_T(2, 4) CD_DM_T(4, 2) CD_DM_T(4, 4)
CD_DM_T(4, 1) CD_DM_T(8, 1) CD_DM_T(8, 2)
CD_DT_T(1) CD_DT_T(2) CD_DT_T(4)

// ---- single-process streamed prefill (streamdense-model ffn_stream.rs; only with QwenOptions.ffn_stream_tokens > 0) ----
// The prompt is read layer by layer with each main layer's FULL FFN (streamed from the model file into a slot); these
// kernels (+ cs_rowscatter for the margin-gate recheck) are the only additions to the normal exact pass kernels.
// cs_setpos: ctrl.pos = p (every layer's chunk passes start again at the prompt's first position). 1 thread.
kernel void cs_setpos(device SdCtrl* ctrl [[buffer(0)]], constant uint& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) ctrl->pos = p;
}
// cs_rowscatter (missing-only margin-gate recheck, gate_recheck.rs): dst row idx[i] = src row i, rows of rw uint4 (one
// 4-bit gate or up W row): the compact layer's resident rows into the full-layer slot. A plain copy (same bytes as the CPU
// copy it replaces). Grid (rw, n).
kernel void cs_rowscatter(device const uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]], device const uint* idx [[buffer(2)]],
                          constant uint& rw [[buffer(3)]], constant uint& n [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= rw || p.y >= n) return;
    dst[(ulong)idx[p.y] * rw + p.x] = src[(ulong)p.y * rw + p.x];
}
// cs_bytecopy (zero-copy margin-gate recheck): dst (16-byte aligned) = n16 x 16 bytes of src starting at byte `shift`
// (any alignment). src = a no-copy buffer over the mapped model file bound at its page-aligned start; the kernels need
// >= 4-byte aligned weights, the 27B's files 2 / 3 put tensors at odd offsets, so those tensors are re-aligned into the
// slot by the GPU (same bytes). Reads only the 4-byte words that hold the copied bytes (never past the tensor's last
// word). One uint4 per thread.
kernel void cs_bytecopy(device const uint* src [[buffer(0)]], device uint4* dst [[buffer(1)]], constant uint& shift [[buffer(2)]],
                        constant uint& n16 [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n16) return;
    ulong s = (ulong)shift + (ulong)i * 16;
    ulong w = s >> 2;
    uint r = (uint)(s & 3) * 8;
    uint a0 = src[w], a1 = src[w + 1], a2 = src[w + 2], a3 = src[w + 3];
    if (r == 0) { dst[i] = uint4(a0, a1, a2, a3); return; }
    uint a4 = src[w + 4], q = 32 - r;
    dst[i] = uint4((a0 >> r) | (a1 << q), (a1 >> r) | (a2 << q), (a2 >> r) | (a3 << q), (a3 >> r) | (a4 << q));
}
// cs_acc: prompt activity of one layer, per neuron j (one thread each):
//   an[j] += sum_t |f[t][j]|                      one add per pass, the activity recorder's arithmetic (act_layer);
//   ag[j] += |silu(g[t][j])| for t = 0..k-1       one add per row in row order, the ffn-select hook's per-row gate score
//                                                 (ffn_select) summed on the CPU token by token.
kernel void cs_acc(device const ushort* f [[buffer(0)]], device const ushort* g [[buffer(1)]], device float* an [[buffer(2)]],
                   device float* ag [[buffer(3)]], constant uint& inter [[buffer(4)]], constant uint& k [[buffer(5)]],
                   uint j [[thread_position_in_grid]]) {
    if (j >= inter) return;
    float s = 0.0f;
    for (uint t = 0; t < k; t++) s += fabs(bf(f[t * inter + j]));
    an[j] += s;
    float a = ag[j];
    for (uint t = 0; t < k; t++) { float x = bf(g[t * inter + j]); float q = fabs(x / (1.0f + exp(-x))); a += q; }
    ag[j] = a;
}
