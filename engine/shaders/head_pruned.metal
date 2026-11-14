// head_pruned.metal - pruned-vocab MTP draft head: lm_head rows gathered through a keep-list idx[NK] (sorted
// ascending token ids, `streamdense vocab keep`), no weight copy. Needs common_bf16 (bf, to_bf, SdCtrl),
// gemv_mma_q4 (qmt_terms, QMT_*) and gemv_q4_bf16 (q4b_dot) first; spec.metal is not needed.
// Per kept row i the value is the full-head kernel's expression sequence for row idx[i] (neither family mixes rows),
// so outf[i] == the full head's logits[idx[i]] bit for bit:
//   qm_gemvt_idx_k1_s{SK}_m3   == qm_gemvt_k1_s{SK}_m3 / qm_gemv_n*_s{SK}_m3 (GemvImpl::Mma). W[0] S[1] B[2] x[3]
//                                 outf[4] NK[5] C[6] idx[7]; groups ceil(NK / 8) x NT (host: mma_twin_nt, as the twin).
//   q4b_gemv_idx_k1_r{D}_m3    == q4b_gemv_k1[_r{D}]_m3 (GemvImpl::Old). Same buffers; groups ceil(NK / 2D) x 64.
//   mtp_draft_idx              == mtp_draft over the kept logits (bf16-rounded, lowest i on ties = lowest token id since
//                                 idx ascends): ctrl.draft = idx[i], ctrl.mtp_len += mctrl.mtp_rows. logits[0] ctrl[1]
//                                 mctrl[2] NK[3] idx[4]; groups (1, 1) x 1024.

// qmt_rows (gemv_mma_q4.metal) for NM = 1 with the row of item r = idx[min(i0 + r, NK - 1)]; every other line is
// qmt_rows verbatim (same terms, same group-order reduction, same slice order).
template <uint KT, uint SK>
static inline void qmt_rows_idx(device const uint* W0, device const ushort* S0, device const ushort* B0,
                                device const uint* idx, device const ushort* x, uint NK, uint C, uint i0, uint tid,
                                uint NT, threadgroup float* T, threadgroup float* fin) {
    constexpr uint NR = QMT_RT, NP = NR / QMT_RP;
    const uint G = C / 64, C8 = C / 8;
    const uint rr = tid / SK, rk = tid % SK;
    const bool red = tid < NR * SK;
    const uint ga = rk * G / SK, gb = (rk + 1) * G / SK;
    float acc[KT];
    QM_UNROLL for (uint t = 0; t < KT; t++) acc[t] = 0.0f;
    const uint CG = NT / NP, gl = tid % CG, pass = tid / CG;
    for (uint gc = 0; gc < G; gc += CG) {
        const uint g = gc + gl;
        if (pass < NP && g < G) {
            device const ushort* xg = x + 64 * g;
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
            device const uint* wr[QMT_RP];
            device const ushort* sr[QMT_RP];
            device const ushort* br[QMT_RP];
            QM_UNROLL for (uint n = 0; n < QMT_RP; n++) {
                uint r = pass * QMT_RP + n;
                uint row = idx[min(i0 + r, NK - 1)];
                wr[n] = W0 + row * C8; sr[n] = S0 + row * G; br[n] = B0 + row * G;
            }
            float tv[QMT_RP][KT];
            qmt_terms<QMT_RP, KT>(wr, sr, br, xg, C, xs, g, tv);
            QM_UNROLL for (uint n = 0; n < QMT_RP; n++) QM_UNROLL for (uint t = 0; t < KT; t++)
                T[((pass * QMT_RP + n) * KT + t) * CG + gl] = tv[n][t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (red) {
            uint lo = max(ga, gc), hi = min(gb, min(gc + CG, G));
            for (uint gg = lo; gg < hi; gg++) QM_UNROLL for (uint t = 0; t < KT; t++)
                acc[t] = acc[t] + T[(rr * KT + t) * CG + (gg - gc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (red) QM_UNROLL for (uint t = 0; t < KT; t++) T[(rk * NR + rr) * KT + t] = acc[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NR) QM_UNROLL for (uint t = 0; t < KT; t++) {
        float v = T[tid * KT + t];
        for (uint k = 1; k < SK; k++) v += T[(k * NR + tid) * KT + t];
        fin[tid * KT + t] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint SK>
kernel void qm_gemvt_idx(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                         device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                         device float* outf [[buffer(4)]], constant uint& NK [[buffer(5)]], constant uint& C [[buffer(6)]],
                         device const uint* idx [[buffer(7)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint NT [[threads_per_threadgroup]]) {
    threadgroup float T[QMT_TMAX];
    threadgroup float fin[QMT_RT];
    const uint i0 = tg * QMT_RT;
    qmt_rows_idx<1, SK>(W, S, B, idx, x, NK, C, i0, tid, NT, T, fin);
    const uint i = i0 + tid;
    if (tid >= QMT_RT || i >= NK) return;
    outf[i] = fin[tid];
}

template <uint RPS>
kernel void q4b_gemv_idx(device const uint4* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                         device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                         device float* outf [[buffer(4)]], constant uint& NK [[buffer(5)]], constant uint& C [[buffer(6)]],
                         device const uint* idx [[buffer(7)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint i0 = (tg * nsg + sg) * RPS;
    if (i0 >= NK) return;
    uint C32 = C / 32, G = C / 64;
    device const uint4* wr[RPS];
    device const ushort* sr[RPS];
    device const ushort* br[RPS];
    for (uint n = 0; n < RPS; n++) {
        uint row = idx[min(i0 + n, NK - 1)];
        wr[n] = W + row * C32; sr[n] = S + row * G; br[n] = B + row * G;
    }
    float acc[RPS][1];
    q4b_dot<1, RPS>(wr, sr, br, x, C, lane, acc);
    for (uint n = 0; n < RPS; n++) {
        float v = simd_sum(acc[n][0]);
        if (lane == 0 && i0 + n < NK) outf[i0 + n] = v;
    }
}

kernel void mtp_draft_idx(device const float* logits [[buffer(0)]], device SdCtrl* ctrl [[buffer(1)]],
                          device const SdCtrl* mctrl [[buffer(2)]], constant uint& NK [[buffer(3)]],
                          device const uint* idx [[buffer(4)]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float best = -INFINITY;
    uint bidx = 0xffffffffu;
    for (uint i = tid; i < NK; i += 1024) {
        float v = bf(to_bf(logits[i]));
        if (bidx == 0xffffffffu || v > best) { best = v; bidx = i; }
    }
    float m = simd_max(best);
    uint mi = simd_min(best == m ? bidx : 0xffffffffu);
    if (lane == 0) { bv[sg] = m; bi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = bv[lane];
        uint ix = bi[lane];
        float m2 = simd_max(v);
        uint r = simd_min(v == m2 ? ix : 0xffffffffu);
        if (lane == 0) {
            ctrl->draft = idx[r];
            ctrl->mtp_len += mctrl->mtp_rows;
        }
    }
}

#define QMT_IDX_T(SK_) template [[host_name("qm_gemvt_idx_k1_s" #SK_ "_m3")]] kernel void qm_gemvt_idx<SK_>( \
    device const uint*, device const ushort*, device const ushort*, device const ushort*, device float*, \
    constant uint&, constant uint&, device const uint*, uint, uint, uint);
QMT_IDX_T(1) QMT_IDX_T(2) QMT_IDX_T(4) QMT_IDX_T(8)
#define Q4B_IDX_T(D_) template [[host_name("q4b_gemv_idx_k1_r" #D_ "_m3")]] kernel void q4b_gemv_idx<D_>( \
    device const uint4*, device const ushort*, device const ushort*, device const ushort*, device float*, \
    constant uint&, constant uint&, device const uint*, uint, uint, uint, uint);
Q4B_IDX_T(2) Q4B_IDX_T(4) Q4B_IDX_T(8)
