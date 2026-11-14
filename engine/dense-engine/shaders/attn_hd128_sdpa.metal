// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (sdpa_vector / sdpa_vector_2pass: thread layout, reduction order, block partial rounding).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// attn_hd128_sdpa.metal - dense-engine: MLX-matching decode attention at head dim D = 64, 96, 128, 160 or 256
// (attn<D>_sdpa_vec; D = 160: same arithmetic, MLX itself has no sdpa_vector for it).
// Needs engine/shaders/common_bf16.metal (bf, to_bf, SdCtrl) and attn_hd128.metal (function constants QD_FC_*,
// qd_scale, qd_softcap, qd_key_lo) first in the concatenated source.
// Port of engine/shaders/attn_gated.metal's attn_sdpa_vec_b16 (D = 256) to a template over D. Optional per-pipeline
// scale / score soft cap (after simd_sum) / sliding window (each simdgroup starts at its first key >= N - W) via the
// function constants; with none set the D = 128 code is the original MLX mirror.
//
// ---- attn128_sdpa_vec: MLX's decode attention (mlx/backend/metal/kernels/sdpa_vector.h, T = bf16, D = V = 128, L = 1,
// no mask, no sinks) mirrored per query row. MLX 0.32.4 host logic (backend/metal/scaled_dot_product_attention.cpp,
// device class 's' = M4 Pro "applegpu_g16s"):
//   - N < 1024: sdpa_vector<bf16, 128>: 32 simdgroups (BN) x 32 lanes (BD); simdgroup g takes keys g, g + 32, ...;
//     lane l holds dims 4l..4l+3 (qk_per_thread = v_per_thread = 128 / 32 = 4); online softmax with fast::exp; the 32
//     partials combined through threadgroup memory (transpose + simd_sum, 4 rounds).
//   - N >= 1024: sdpa_vector_2pass_1 (TK = 4 prefetch: loads only, the per-key arithmetic order is unchanged) +
//     sdpa_vector_2pass_2, fused here: simdgroup g runs pass-1 blocks b = g, g + 32, ... (keys b, b + blocks, ...;
//     o rounded to bf16 as MLX's T partials, sums / maxs fp32), then pass 2. blocks = 64, or 128 when N > 1024 and
//     gqa > 4 (n_simds = gqa * L, L = 1), N <= 8192.
//   - NOT mirrored: sdpa_vector_2pass_1_gqa (MLX takes it for D = 128, gqa in {8, 12, 16}, N >= 8192): the host must
//     keep lim <= 8191 for those gqa factors (dense_engine::sdpa128_vec_max_n), larger rows go to split-K.
// Exactly the same arithmetic as the D = 256 mirror (which is bit-exact vs MLX in the engine's own tests) with D = 128; the
// bit-identity at D = 128 is checked against mx.fast.scaled_dot_product_attention by tests/attn_hd128_flash.rs.
// Scale: D^-0.5 rounded to f32 unless QD_FC_SCALE is set (MLX gets head_dim**-0.5 as a float); q = bf(q) * scale in fp32, as MLX.
// Output: ob = bf(o / sum_exp) (sdpa output). flags bit 0 clear: ao = ob (Qwen dense). flags bit 0 set: ao =
// bf(ob * sigmoid_bf16(gate)) with MLX's bf16 unary rounding, gate of head h, row t at p + t * p_stride + h*2*HD + HD.
// Per query row t the key count is N = ctrl.pos + t + 1, so a row's arithmetic depends only on N (K-invariance).
// Rows with N > lim are skipped (written by the split-K path, dispatched before this).
// Buffers: qr [0] (bf16 [T][nh][D]), Kc [1], Vc [2] (bf16 [nkv][max_ctx][D]), p [3] (gate, read only if gated),
// ao [4] (bf16 [T][nh*D]), ctrl [5], max_ctx [6], grp [7] (nh / nkv), p_stride [8] (gated only), lim [9],
// two_pass_min [10] (MLX 0.32.4: 1024), flags [11]. Dispatch (nh, T) groups x 1024 threads.
constant uint AV_MAXBB = 4;                        // pass-1 blocks per simdgroup (blocks <= 128, N <= 8192)
constant uint AV_GATED = 1u;

static inline int attn128_sdpa_blocks(int N, int n_simds) {
    if (N > 1024 && n_simds > 4) return N <= 8192 ? 128 : (N <= 32768 ? 256 : (N <= 65536 ? 512 : 1024));
    return 64;
}

template <int D>
kernel void attn_sdpa_vec(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
                             device const ushort* Vc [[buffer(2)]], device const ushort* p [[buffer(3)]],
                             device ushort* ao [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                             constant uint& maxctx [[buffer(6)]], constant uint& grp [[buffer(7)]],
                             constant uint& p_stride [[buffer(8)]], constant uint& lim [[buffer(9)]],
                             constant uint& two_pass_min [[buffer(10)]], constant uint& flags [[buffer(11)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                             uint simd_gid [[simdgroup_index_in_threadgroup]],
                             uint simd_lid [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr int BN = 32, BD = 32;
    constexpr int qk_per_thread = D / BD, v_per_thread = D / BD;
    typedef float U;
    const uint h = tg.x, t = tg.y, nh = ntgs.x;
    const int N = int(ctrl->pos + t + 1);
    if (uint(N) > lim) return;  // whole threadgroup leaves together (no barrier reached)
    const int blocks = attn128_sdpa_blocks(N, int(grp));
    const bool two = uint(N) >= two_pass_min;
    if (two && blocks > int(BN * AV_MAXBB)) return;  // host never sets lim that high
    const uint kvh = h / grp;
    thread U q[qk_per_thread], k[qk_per_thread], o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN * AV_MAXBB];
    threadgroup U sum_exp_scores[BN * AV_MAXBB];

    device const ushort* queries = qr + (t * nh + h) * D + simd_lid * qk_per_thread;
    device const ushort* kbase = Kc + kvh * maxctx * D + simd_lid * qk_per_thread;
    device const ushort* vbase = Vc + kvh * maxctx * D + simd_lid * v_per_thread;
    const U scale = qd_scale(D);
    const int lo = int(qd_key_lo(uint(N - 1)));        // first visible key (0 without a window)
    for (int i = 0; i < qk_per_thread; i++) q[i] = scale * bf(queries[i]);

    U max_score, sum_exp_score, factor;
    if (!two) {
        for (int i = 0; i < v_per_thread; i++) o[i] = 0;
        max_score = -FLT_MAX;  // Limits<float>::finite_min
        sum_exp_score = 0;
        int i0 = simd_gid;                               // window: first key >= lo of this simdgroup's stride
        if (lo > i0) i0 += (lo - i0 + BN - 1) / BN * BN;
        device const ushort* keys = kbase + i0 * D;
        device const ushort* values = vbase + i0 * D;
        for (int i = i0; i < N; i += BN) {
            for (int j = 0; j < qk_per_thread; j++) k[j] = bf(keys[j]);
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
            score = simd_sum(score);
            score = qd_softcap(score);
            U new_max = max(max_score, score);
            U fac = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * fac + exp_score;
            for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * fac + exp_score * bf(values[j]);
            keys += BN * D;
            values += BN * D;
        }
        if (simd_lid == 0) {
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[simd_lid];
        U new_max = simd_max(max_score);
        factor = fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
    } else {
        // ---- pass 1: blocks b = simd_gid + 32 bb, keys b, b + blocks, ... (sdpa_vector_2pass_1)
        const int nbb = blocks / BN;
        U part[AV_MAXBB][v_per_thread];
        for (int bb = 0; bb < nbb; bb++) {
            const int b = simd_gid + BN * bb;
            U ob[v_per_thread];
            for (int j = 0; j < v_per_thread; j++) ob[j] = 0;
            U mxs = -FLT_MAX, se = 0;
            int ib = b;
            if (lo > ib) ib += (lo - ib + blocks - 1) / blocks * blocks;
            device const ushort* keys = kbase + ib * D;
            device const ushort* values = vbase + ib * D;
            for (int i = ib; i < N; i += blocks) {
                U score = 0;
                for (int j = 0; j < qk_per_thread; j++) score += q[j] * bf(keys[j]);
                score = simd_sum(score);
                score = qd_softcap(score);
                U new_max = max(mxs, score);
                U fac = fast::exp(mxs - new_max);
                U exp_score = fast::exp(score - new_max);
                mxs = new_max;
                se = se * fac + exp_score;
                for (int j = 0; j < v_per_thread; j++) ob[j] = ob[j] * fac + exp_score * bf(values[j]);
                keys += blocks * D;
                values += blocks * D;
            }
            for (int j = 0; j < v_per_thread; j++) part[bb][j] = bf(to_bf(ob[j]));  // MLX partials are T (bf16)
            if (simd_lid == 0) { sum_exp_scores[b] = se; max_scores[b] = mxs; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- pass 2 (sdpa_vector_2pass_2)
        sum_exp_score = 0.0;
        max_score = -FLT_MAX;
        for (int b = 0; b < nbb; ++b) max_score = max(max_score, max_scores[simd_lid + BN * b]);
        max_score = simd_max(max_score);
        for (int b = 0; b < nbb; ++b) {
            U fac = fast::exp(max_scores[simd_lid + BN * b] - max_score);
            sum_exp_score += fac * sum_exp_scores[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);
        for (int i = 0; i < v_per_thread; i++) o[i] = 0;
        for (int b = 0; b < nbb; ++b) {
            U fac = fast::exp(max_scores[simd_gid + BN * b] - max_score);
            for (int i = 0; i < v_per_thread; i++) o[i] += fac * part[b][i];
        }
        factor = 1.0f;  // pass 2's transpose sums the already weighted partials
    }
    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = two ? simd_sum(outputs[simd_gid * BD + simd_lid]) : simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        device ushort* out = ao + t * nh * D + h * D + simd_gid * v_per_thread;
        if (flags & AV_GATED) {
            device const ushort* g = p + t * p_stride + h * 2 * D + D + simd_gid * v_per_thread;
            for (int i = 0; i < v_per_thread; i++) {
                float ob = bf(to_bf(o[i]));
                float gv = bf(g[i]);
                float e = bf(to_bf(precise::exp(fabs(gv))));
                float y = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
                float s = gv < 0.0f ? y : bf(to_bf(1.0f - y));
                out[i] = to_bf(ob * s);
            }
        } else {
            for (int i = 0; i < v_per_thread; i++) out[i] = to_bf(o[i]);
        }
    }
}
#define AVS(D_) template [[host_name("attn" #D_ "_sdpa_vec")]] kernel void attn_sdpa_vec<D_>(device const ushort*, \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, device const SdCtrl*, \
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint2, uint, \
    uint);
AVS(128) AVS(64) AVS(256) AVS(96) AVS(160)
