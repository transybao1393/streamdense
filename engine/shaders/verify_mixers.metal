// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (sdpa_vector / sdpa_vector_2pass: partition, reduction order, rounding). Full MIT license text: THIRD_PARTY_NOTICES.md.
//
// verify_mixers.metal - streamdense K-row exact VERIFY token mixers (2026-11-03, decode_bench --verify,
// QwenOptions.verify_mixers): the attention sdpa for bf16 KV (va_sdpa_*) and the DeltaNet recurrence (dn_recur_v, at
// the end of the file). Needs common_bf16.metal, deltanet.metal and attn_gated.metal first (bf, to_bf, SdCtrl, DN_D,
// DnDims, AG_*, sdpa_blocks).
//
// attn_sdpa_vec_b16 runs one 1024-thread threadgroup per (q head, row): every one of the nh x K threadgroups streams
// the whole K/V of its kv head (48 times the same bytes at K = 8, GQA 6), and each simdgroup walks its keys as one
// dependent chain (load -> dot -> simd_sum -> exp -> update), so a verify pass costs ~K x a decode row (27B, pos 4000,
// K = 8: 15.4 ms per pass vs 2.9 at K = 1). Here the MLX two-pass rows are split the way MLX itself splits them:
// - va_sdpa_p1<R>: threadgroup (block b, kv head, row chunk) = grp simdgroups (one per q head of the kv group);
//   simdgroup = pass-1 block b of ONE q head for R rows: each key of the block is loaded ONCE (into registers) and
//   feeds R independent online-softmax chains (rows), and the grp simdgroups of the threadgroup share it through L1.
//   Writes MLX's pass-1 outputs (o rounded to bf16 as MLX's T partials, max and sum fp32) to the scratch.
// - va_sdpa_p3: pass 2 + division + sigmoid gate of those rows from the scratch, 256 threads per (head, row) with
//   lane = block set (attn_sdpa_vec_b16 gets there through a threadgroup transpose; same values, same simd_sums).
// - va_sdpa_p2: every other row (single pass, N > lim, or a block count different from the last row's) runs the
//   unchanged attn_sdpa_vec_b16 code (copied; its threadgroups of batched rows return at once).
// BIT-IDENTICAL to attn_sdpa_vec_b16 per row: a block's chain is the same sequence of the same operations (key order
// b, b + blocks, ...; lane l holds dims 8l..8l+7; same expressions), only which simdgroup runs it and how many other
// rows run beside it change. Rows never interact (each row keeps its own N), so K-invariance holds.
// A row is batched iff two_pass_min <= N <= lim and sdpa_blocks(N) == sdpa_blocks(N_last), N_last = pos + T (both
// kernels test it the same way from ctrl.pos).
// Scratch (bf16 KV only): part ushort [T][nh][VA_BMAX][256] then ms float2 (max, sum) [T][nh][VA_BMAX]
//   = T * nh * VA_BMAX * (512 + 8) bytes (27B, T = 8: 12.8 MB).
// p1 buffers: qr [0], Kc [1], Vc [2], part [3], ms [4], ctrl [5], max_ctx [6], grp [7], T [8], lim [9],
//   two_pass_min [10], nh [11]. Dispatch (VA_BMAX * ceil(T / (R C)), nkv) groups x 32 grp C: C row chunks of R rows
//   per threadgroup (simdgroup s: q head s % grp, chunk s / grp; x = block + VA_BMAX * threadgroup chunk).
// p2 buffers: as attn_sdpa_vec_b16 [0..10]. Dispatch (nh, T) groups x 1024. p3: see the kernel.

constant constexpr int VA_BMAX = 128;  // most pass-1 blocks of a row (sdpa_blocks for N <= 8192)

static inline bool va_batched(int N, int Nlast, int grp, uint lim, uint tpm) {
    return uint(N) >= tpm && uint(N) <= lim && sdpa_blocks(N, grp) == sdpa_blocks(Nlast, grp)
           && sdpa_blocks(N, grp) <= VA_BMAX;
}

#define VA_UNROLL _Pragma("clang loop unroll(full)")

template <uint R>
kernel void va_sdpa_p1(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
                       device const ushort* Vc [[buffer(2)]], device ushort* part [[buffer(3)]],
                       device float2* ms [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                       constant uint& maxctx [[buffer(6)]], constant uint& grp [[buffer(7)]],
                       constant uint& T [[buffer(8)]], constant uint& lim [[buffer(9)]],
                       constant uint& two_pass_min [[buffer(10)]], constant uint& nh [[buffer(11)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
                       uint nsg [[simdgroups_per_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr int D = 256, PT = 8;  // dims per lane
    typedef float U;
    const int pos = int(ctrl->pos);
    const int Nlast = pos + int(T);
    const int blocks = sdpa_blocks(Nlast, int(grp));
    const int b = int(tg.x % VA_BMAX);
    if (b >= blocks) return;
    const uint cpt = nsg / grp;  // row chunks per threadgroup
    const uint kvh = tg.y, h = kvh * grp + simd_gid % grp;
    const uint t0 = ((tg.x / VA_BMAX) * cpt + simd_gid / grp) * R;
    const U scale = AG_SCALE;
    // rows t0 .. t0 + R - 1 of this chunk: active ones are a prefix-free set; N grows with t, so the batched rows
    // that see key i are exactly the active rows with N > i
    int N[R];
    bool act[R];
    int nmax = 0;
    VA_UNROLL for (uint r = 0; r < R; r++) {
        const uint t = t0 + r;
        N[r] = pos + int(t) + 1;
        act[r] = t < T && va_batched(N[r], Nlast, int(grp), lim, two_pass_min);
        if (act[r]) nmax = max(nmax, N[r]);
    }
    if (nmax == 0) return;
    U q[R][PT], ob[R][PT], mxs[R], se[R];
    VA_UNROLL for (uint r = 0; r < R; r++) {
        device const ushort* queries = qr + (min(t0 + r, T - 1) * nh + h) * D + simd_lid * PT;
        VA_UNROLL for (int j = 0; j < PT; j++) { q[r][j] = scale * bf(queries[j]); ob[r][j] = 0; }
        mxs[r] = -FLT_MAX;
        se[r] = 0;
    }
    device const ushort* keys = Kc + (kvh * maxctx + uint(b)) * D + simd_lid * PT;
    device const ushort* values = Vc + (kvh * maxctx + uint(b)) * D + simd_lid * PT;
    // keys / values of the next key of the block are loaded one iteration ahead (latency hiding; the loop is a
    // dependent chain per row otherwise). Raw bf16 bits, converted at use.
    uint4 kn = *(device const uint4*)keys, vn = *(device const uint4*)values;
    for (int i = b; i < nmax; i += blocks) {
        const uint4 kc = kn, vc = vn;
        keys += blocks * D;
        values += blocks * D;
        if (i + blocks < nmax) { kn = *(device const uint4*)keys; vn = *(device const uint4*)values; }
        U k[PT], v[PT];
        const uint kw[4] = {kc.x, kc.y, kc.z, kc.w}, vw[4] = {vc.x, vc.y, vc.z, vc.w};
        VA_UNROLL for (int j = 0; j < 4; j++) {
            k[2 * j] = as_type<float>(kw[j] << 16); k[2 * j + 1] = as_type<float>(kw[j] & 0xffff0000u);
            v[2 * j] = as_type<float>(vw[j] << 16); v[2 * j + 1] = as_type<float>(vw[j] & 0xffff0000u);
        }
        VA_UNROLL for (uint r = 0; r < R; r++) {
            if (act[r] && i < N[r]) {
                U score = 0;
                VA_UNROLL for (int j = 0; j < PT; j++) score += q[r][j] * k[j];
                score = simd_sum(score);
                U new_max = max(mxs[r], score);
                U fac = fast::exp(mxs[r] - new_max);
                U exp_score = fast::exp(score - new_max);
                mxs[r] = new_max;
                se[r] = se[r] * fac + exp_score;
                VA_UNROLL for (int j = 0; j < PT; j++) ob[r][j] = ob[r][j] * fac + exp_score * v[j];
            }
        }
    }
    VA_UNROLL for (uint r = 0; r < R; r++) {
        if (!act[r]) continue;
        const uint row = ((t0 + r) * nh + h) * VA_BMAX + uint(b);
        device ushort* po = part + row * D + simd_lid * PT;
        VA_UNROLL for (int j = 0; j < PT; j++) po[j] = to_bf(ob[r][j]);  // MLX partials are T (bf16)
        if (simd_lid == 0) ms[row] = float2(mxs[r], se[r]);
    }
}

#define VA_P1(R_) template [[host_name("va_sdpa_p1_r" #R_)]] kernel void va_sdpa_p1<R_>( \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, device float2*, \
    device const SdCtrl*, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, \
    constant uint&, uint2, uint, uint, uint);
VA_P1(1) VA_P1(2) VA_P1(3) VA_P1(4) VA_P1(8)

kernel void va_sdpa_p2(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
                       device const ushort* Vc [[buffer(2)]], device const ushort* p [[buffer(3)]],
                       device ushort* ao [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                       constant uint& maxctx [[buffer(6)]], constant uint& grp [[buffer(7)]],
                       constant uint& p_stride [[buffer(8)]], constant uint& lim [[buffer(9)]],
                       constant uint& two_pass_min [[buffer(10)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                       uint simd_gid [[simdgroup_index_in_threadgroup]],
                       uint simd_lid [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr int BN = 32, BD = 32, D = 256;
    constexpr int qk_per_thread = D / BD, v_per_thread = D / BD;
    typedef float U;
    const uint h = tg.x, t = tg.y, nh = ntgs.x;
    const int N = int(ctrl->pos + t + 1);
    if (uint(N) > lim) return;  // whole threadgroup leaves together (no barrier reached)
    const int blocks = sdpa_blocks(N, int(grp));
    const bool two = uint(N) >= two_pass_min;
    if (two && blocks > int(BN * AG_MAXBB)) return;  // host never sets lim that high (checked there)
    if (va_batched(N, int(ctrl->pos + ntgs.y), int(grp), lim, two_pass_min)) return;  // va_sdpa_p3's row
    const uint kvh = h / grp;
    thread U q[qk_per_thread], k[qk_per_thread], o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN * AG_MAXBB];
    threadgroup U sum_exp_scores[BN * AG_MAXBB];

    device const ushort* queries = qr + (t * nh + h) * D + simd_lid * qk_per_thread;
    device const ushort* kbase = Kc + kvh * maxctx * D + simd_lid * qk_per_thread;
    device const ushort* vbase = Vc + kvh * maxctx * D + simd_lid * v_per_thread;
    const U scale = AG_SCALE;
    for (int i = 0; i < qk_per_thread; i++) q[i] = scale * bf(queries[i]);

    U max_score, sum_exp_score, factor;
    if (!two) {
        for (int i = 0; i < v_per_thread; i++) o[i] = 0;
        max_score = -FLT_MAX;  // Limits<float>::finite_min
        sum_exp_score = 0;
        device const ushort* keys = kbase + simd_gid * D;
        device const ushort* values = vbase + simd_gid * D;
        for (int i = simd_gid; i < N; i += BN) {
            for (int j = 0; j < qk_per_thread; j++) k[j] = bf(keys[j]);
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
            score = simd_sum(score);
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
        U part[AG_MAXBB][v_per_thread];
        for (int bb = 0; bb < nbb; bb++) {
            const int b = simd_gid + BN * bb;
            U ob[v_per_thread];
            for (int j = 0; j < v_per_thread; j++) ob[j] = 0;
            U mxs = -FLT_MAX, se = 0;
            device const ushort* keys = kbase + b * D;
            device const ushort* values = vbase + b * D;
            for (int i = b; i < N; i += blocks) {
                U score = 0;
                for (int j = 0; j < qk_per_thread; j++) score += q[j] * bf(keys[j]);
                score = simd_sum(score);
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
        device const ushort* g = p + t * p_stride + h * 2 * D + D + simd_gid * v_per_thread;
        device ushort* out = ao + t * nh * D + h * D + simd_gid * v_per_thread;
        for (int i = 0; i < v_per_thread; i++) {
            float ob = bf(to_bf(o[i]));
            float gv = bf(g[i]);
            float e = bf(to_bf(precise::exp(fabs(gv))));
            float y = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
            float s = gv < 0.0f ? y : bf(to_bf(1.0f - y));
            out[i] = to_bf(ob * s);
        }
    }
}

// va_sdpa_p3: pass 2 of the batched rows (sdpa_vector_2pass_2 as attn_sdpa_vec_b16's two-pass branch) + division +
// sigmoid gate, from the va_sdpa_p1 scratch. attn_sdpa_vec_b16 computes, per output dim d, simd_sum over the 32
// block sets g (lane = g after its threadgroup transpose) of o_g[d] = sum over b of fac(g + 32 b) * part[g + 32 b][d]
// (b ascending, from 0); here lane = g directly (no transpose): simdgroup s owns dims 32 s .. 32 s + 31, 8 at a time.
// Same values into the same simd_sum, same max / sum_exp chain (lane = block set), same epilogue.
// Buffers: p [0] (gate), ao [1], ctrl [2], grp [3], p_stride [4], lim [5], two_pass_min [6], part [7], ms [8], T [9].
// Dispatch (nh, T) groups x 256.
kernel void va_sdpa_p3(device const ushort* p [[buffer(0)]], device ushort* ao [[buffer(1)]],
                       device const SdCtrl* ctrl [[buffer(2)]], constant uint& grp [[buffer(3)]],
                       constant uint& p_stride [[buffer(4)]], constant uint& lim [[buffer(5)]],
                       constant uint& two_pass_min [[buffer(6)]], device const ushort* vpart [[buffer(7)]],
                       device const float2* vms [[buffer(8)]], constant uint& T [[buffer(9)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                       uint simd_gid [[simdgroup_index_in_threadgroup]],
                       uint simd_lid [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr int BN = 32, D = 256;
    typedef float U;
    const uint h = tg.x, t = tg.y, nh = ntgs.x;
    const int N = int(ctrl->pos + t + 1);
    if (!va_batched(N, int(ctrl->pos + T), int(grp), lim, two_pass_min)) return;
    const int blocks = sdpa_blocks(N, int(grp));
    const int nbb = blocks / BN;
    device const float2* m = vms + (t * nh + h) * VA_BMAX;
    U sum_exp_score = 0.0, max_score = -FLT_MAX;
    for (int b = 0; b < nbb; ++b) max_score = max(max_score, m[simd_lid + BN * b].x);
    max_score = simd_max(max_score);
    U fac[AG_MAXBB];
    for (int b = 0; b < nbb; ++b) {
        fac[b] = fast::exp(m[simd_lid + BN * b].x - max_score);
        sum_exp_score += fac[b] * m[simd_lid + BN * b].y;
    }
    sum_exp_score = simd_sum(sum_exp_score);
    device const ushort* pp = vpart + (t * nh + h) * VA_BMAX * D;
    device const ushort* g = p + t * p_stride + h * 2 * D + D;
    device ushort* out = ao + t * nh * D + h * D;
    VA_UNROLL for (int c = 0; c < 4; c++) {
        const int d0 = 32 * int(simd_gid) + 8 * c;
        U o[8];
        VA_UNROLL for (int i = 0; i < 8; i++) o[i] = 0;
        for (int b = 0; b < nbb; ++b) {
            const uint4 w = *(device const uint4*)(pp + (simd_lid + BN * b) * D + d0);
            const uint ww[4] = {w.x, w.y, w.z, w.w};
            VA_UNROLL for (int j = 0; j < 4; j++) {
                o[2 * j] += fac[b] * as_type<float>(ww[j] << 16);
                o[2 * j + 1] += fac[b] * as_type<float>(ww[j] & 0xffff0000u);
            }
        }
        U r = 0;
        VA_UNROLL for (int i = 0; i < 8; i++) {
            U v = simd_sum(o[i]);
            v = sum_exp_score == 0 ? v : (v / sum_exp_score);
            if (simd_lid == uint(i)) r = v;
        }
        if (simd_lid < 8) {
            const int d = d0 + int(simd_lid);
            float ob = bf(to_bf(r));
            float gv = bf(g[d]);
            float e = bf(to_bf(precise::exp(fabs(gv))));
            float y = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
            float s = gv < 0.0f ? y : bf(to_bf(1.0f - y));
            out[d] = to_bf(ob * s);
        }
    }
}

// ---- dn_recur_v<CK> (verify passes, T <= 8): dn_recur with the T tokens' k / q (this v head's k head), v (this
// threadgroup's 32 dv rows), g and beta staged in threadgroup memory before the token loop, while the state rows are
// in flight; the loop then reads threadgroup memory instead of issuing device loads on its dependent chain. Same
// arithmetic, same order, same values (bf16 bits copied) as dn_recur, so the state / y / checkpoint are bit-identical.
// Needs the engine's dispatch: (4 vh, 1) groups x 128 threads (4 simdgroups = 32 dv rows of ONE v head).
constant constexpr uint VD_TMAX = 8;
template <bool CK>
kernel void dn_recur_v(device const ushort* cv [[buffer(0)]], device const float* g [[buffer(1)]],
                       device const ushort* beta [[buffer(2)]], device float* S [[buffer(3)]],
                       device ushort* y [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                       constant uint& T [[buffer(6)]], constant DnDims& dd [[buffer(7)]],
                       uint2 gid [[threadgroup_position_in_grid]], uint2 ngr [[threadgroups_per_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr uint lanes_per_row = 4;
    constexpr uint rows_per_simdgroup = 32 / lanes_per_row;  // 8
    constexpr uint values_per_lane = DN_D / lanes_per_row;   // 32
    constexpr uint partials_per_lane = values_per_lane / 4;  // 8
    constexpr uint row_groups = DN_D / rows_per_simdgroup;   // 16
    threadgroup ushort tk[VD_TMAX * DN_D], tq[VD_TMAX * DN_D], tv[VD_TMAX * 32], tb[VD_TMAX];
    threadgroup float tgt[VD_TMAX];

    const uint tg0 = (gid.y * ngr.x + gid.x) * nsg;  // first simdgroup of this threadgroup
    uint sgi = tg0 + sg;
    if (tg0 >= dd.vh * row_groups) return;  // whole threadgroup (vh * 16 is a multiple of 4)
    uint hv = sgi / row_groups;
    uint hk = hv / (dd.vh / dd.kh);
    uint row_in_simdgroup = lane / lanes_per_row;
    uint lane_in_row = lane & (lanes_per_row - 1);
    uint dv = (sgi % row_groups) * rows_per_simdgroup + row_in_simdgroup;
    const uint dv0 = (tg0 % row_groups) * rows_per_simdgroup;  // first dv row of the threadgroup

    const uint kdim = dd.kh * DN_D;
    uint role = ctrl->role & 1u;
    const uint copy = dd.vh * DN_D * DN_D;
    device float* st = S + role * copy + (hv * DN_D + dv) * DN_D + lane_in_row * values_per_lane;
    device float* stc = S + (role ^ 1u) * copy + (hv * DN_D + dv) * DN_D + lane_in_row * values_per_lane;

    float state[values_per_lane];
    for (uint i = 0; i < values_per_lane; ++i) state[i] = st[i];

    for (uint i = tid; i < T * DN_D; i += 128) {
        const uint t = i / DN_D, d = i % DN_D;
        tq[i] = cv[t * dd.conv + hk * DN_D + d];
        tk[i] = cv[t * dd.conv + kdim + hk * DN_D + d];
    }
    for (uint i = tid; i < T * 32; i += 128) {
        const uint t = i / 32, r = i % 32;
        tv[i] = cv[t * dd.conv + 2 * kdim + hv * DN_D + dv0 + r];
    }
    if (tid < T) { tgt[tid] = g[tid * dd.vh + hv]; tb[tid] = beta[tid * dd.vh + hv]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const ushort* q_ = tq + lane_in_row * values_per_lane;
    threadgroup const ushort* k_ = tk + lane_in_row * values_per_lane;
    device ushort* y_ = y + hv * DN_D;

    for (uint t = 0; t < T; ++t) {
        float gt = tgt[t];

        float part[partials_per_lane];
        for (uint pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                uint e = pb * 4 + i;
                state[e] = state[e] * gt;
                acc += state[e] * bf(k_[e]);
            }
            part[pb] = acc;
        }
        float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);

        float delta = (bf(tv[t * 32 + dv - dv0]) - kv_mem) * bf(tb[t]);

        for (uint pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                uint e = pb * 4 + i;
                state[e] = state[e] + bf(k_[e]) * delta;
                acc += state[e] * bf(q_[e]);
            }
            part[pb] = acc;
        }
        float out = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        if (lane_in_row == 0) y_[dv] = to_bf(out);  // static_cast<InT>(out), bf16 RNE

        if (CK && t == 0) {
            for (uint i = 0; i < values_per_lane; ++i) stc[i] = state[i];
        }

        q_ += DN_D;
        k_ += DN_D;
        y_ += dd.vh * DN_D;
    }

    for (uint i = 0; i < values_per_lane; ++i) st[i] = state[i];
}

template [[host_name("dn_recur_v_ck0")]] kernel void dn_recur_v<false>(
    device const ushort*, device const float*, device const ushort*, device float*, device ushort*,
    device const SdCtrl*, constant uint&, constant DnDims&, uint2, uint2, uint, uint, uint, uint);
template [[host_name("dn_recur_v_ck1")]] kernel void dn_recur_v<true>(
    device const ushort*, device const float*, device const ushort*, device float*, device ushort*,
    device const SdCtrl*, constant uint&, constant DnDims&, uint2, uint2, uint, uint, uint, uint);
