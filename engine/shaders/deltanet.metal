// Portions adapted from mlx-lm (https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
// (models/gated_delta.py packed Dk=128 gated-delta kernel and qwen3_5 GatedDeltaNet), and from MLX
// (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc. (rms_norm, unary Sigmoid, log1p).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// deltanet.metal - streamdense Qwen3.8-27B Gated DeltaNet kernels (linear-attention layers).
// Needs common_bf16.metal first in the concatenated source (bf, to_bf, SdCtrl).
//
// Math source of truth: mlx-lm qwen3_5.GatedDeltaNet.__call__ + gated_delta.py (normalize_qk, compute_g,
// gated_delta_update -> the "packed" Dk=128 kernel _make_gated_delta_packed_kernel). Dtypes mirror MLX:
// conv out, q/k after l2norm, v, beta and y are bf16; g and the state S are fp32. Every bf16 rounding point is
// an explicit to_bf() at the place MLX stores (or, inside a bf16 elementwise op, computes) a bf16 value.
//
// Model dims are runtime arguments (DnDims, below) so the same kernels run the 27B and the tiny test model.
// Only dk = dv = 128 (the MLX packed layout) and conv k = 4 are compile-time; QwenConfig.validate() enforces them.
//
// Math mode: every kernel body starts with `#pragma METAL fp math_mode(safe)` (MLX compiles its kernels without
// fast-math). Under the default fast mode dn_recur misses MLX by 1 bf16 ulp in y and ~6e-8 in S (self-test B).
//
// K-invariance: every kernel walks the T tokens sequentially with the same per-token code, carrying state in
// registers (fp32 S, bf16 conv rows) exactly as it would be stored, so a T=2 dispatch is bit-identical to two
// T=1 dispatches (plan section 4.2).
//
// Rollback (plan 7.2): state copy `role = ctrl.role` is updated in place; with a checkpoint (dn_conv ck = token
// index, dn_recur_ck1) the state after token ck / token 0 is also written to copy role ^ 1.

constant constexpr uint DN_D = 128;  // head_k_dim == head_v_dim (validated on the Rust side)
constant constexpr uint DN_KS = 3;   // conv_kernel_size - 1 conv state rows
constant constexpr uint DN_HOLD = 0xFFFFFFFEu; // dn_conv ck: hold the state (no write-back, ctx_draft.rs)

// DeltaNet dims, bound as `constant DnDims&` (dn_conv [8], dn_prep [7], dn_recur [7]).
// Rust mirror (U14), 32 B:
//   #[repr(C)] pub struct DnDims { pub kh: u32, pub vh: u32, pub conv: u32, pub prow: u32,
//                                  pub b_off: u32, pub a_off: u32, pub layers: u32, pub pad: u32 }
//   kh = dn_kh, vh = dn_vh, conv = conv_dim() (= 2*kh*128 + vh*128), prow = dn_in_rows() (p row stride),
//   b_off = conv + vh*128 (in_proj_b inside a p row), a_off = b_off + vh, layers = n_deltanet(), pad = 0.
//   27B: {16, 48, 10240, 16480, 16384, 16432, 48, 0}; tiny: {2, 6, 1280, 2060, 2048, 2054, 6, 0}.
struct DnDims {
    uint kh;      // linear k heads
    uint vh;      // linear v heads (multiple of kh, <= 128)
    uint conv;    // conv channels = q | k | v = 2*kh*128 + vh*128
    uint prow;    // dn_in GEMV row stride (qkv | z | b | a)
    uint b_off;   // in_proj_b column offset within a p row
    uint a_off;   // in_proj_a column offset within a p row
    uint layers;  // DeltaNet layers in the conv state buffer
    uint pad;
};

// MLX unary Sigmoid with T = bfloat16_t (unary_ops.h: y = 1 / (1 + precise::exp(abs(x))); x < 0 ? y : 1 - y),
// every bfloat op rounded to bf16. x is already a bf16 value.
static inline float dn_sigmoid_bf(float x) {
    float e = bf(to_bf(precise::exp(fabs(x))));
    float d = bf(to_bf(1.0f + e));
    float y = bf(to_bf(precise::divide(1.0f, d)));
    return x < 0.0f ? y : bf(to_bf(1.0f - y));
}

// ---------------------------------------------------------------------------------------------------------------
// dn_conv: depthwise causal conv1d (k = 4, no bias) + silu over the dd.conv qkv channels, one thread per channel.
// p ushort[T][prow] [0] (qkv = cols 0..conv), w ushort[conv][4] [1] (conv1d.weight [C,4,1]),
// st ushort[2][layers][3][conv] [2], out ushort[T][conv] [3], ctrl [4], layer [5] (DeltaNet layer index
// 0..layers-1, not the model layer), T [6], ck [7] (token index after which the state is also copied to
// st[role ^ 1]; ~0 = none; DN_HOLD = no checkpoint and the state is NOT written back: context-draft verify,
// ctx_draft.rs), dd [8].
// Dispatch: dd.conv threads (any threadgroup width).
//
// MLX: conv_input = [conv_state ; qkv] (bf16), conv_out = silu(conv1d(conv_input)); depthwise_conv_1d keeps a
// fp32 accumulator over the 4 taps oldest -> newest (acc += x * w), stores bf16; nn.silu (compiled,
// x * sigmoid(x)) runs in bf16. New conv state = the last 3 rows of conv_input (bf16, unchanged values).
kernel void dn_conv(device const ushort* p [[buffer(0)]], device const ushort* w [[buffer(1)]],
                    device ushort* st [[buffer(2)]], device ushort* out [[buffer(3)]],
                    device const SdCtrl* ctrl [[buffer(4)]], constant uint& layer [[buffer(5)]],
                    constant uint& T [[buffer(6)]], constant uint& ck [[buffer(7)]],
                    constant DnDims& dd [[buffer(8)]], uint c [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)  // MLX builds kernels without fast-math; needed for bit-exact S/y
    if (c >= dd.conv) return;
    uint role = ctrl->role & 1u;
    device ushort* s = st + ((role * dd.layers + layer) * DN_KS) * dd.conv + c;
    device ushort* sc = st + (((role ^ 1u) * dd.layers + layer) * DN_KS) * dd.conv + c;
    float w0 = bf(w[c * 4 + 0]), w1 = bf(w[c * 4 + 1]), w2 = bf(w[c * 4 + 2]), w3 = bf(w[c * 4 + 3]);
    ushort r0 = s[0], r1 = s[dd.conv], r2 = s[2 * dd.conv];
    for (uint t = 0; t < T; ++t) {
        ushort x = p[t * dd.prow + c];
        float acc = 0.0f;
        acc += bf(r0) * w0;
        acc += bf(r1) * w1;
        acc += bf(r2) * w2;
        acc += bf(x) * w3;
        float cv = bf(to_bf(acc));                          // conv1d output, bf16
        out[t * dd.conv + c] = to_bf(cv * dn_sigmoid_bf(cv));  // silu in bf16: bf(x * bf(sigmoid(x)))
        r0 = r1; r1 = r2; r2 = x;
        if (t == ck) { sc[0] = r0; sc[dd.conv] = r1; sc[2 * dd.conv] = r2; }
    }
    if (ck != DN_HOLD) { s[0] = r0; s[dd.conv] = r1; s[2 * dd.conv] = r2; }
}

// ---------------------------------------------------------------------------------------------------------------
// dn_prep: normalize_qk + gates.
// cv ushort[T][conv] [0] (in place: q cols 0..kh*128, k cols kh*128..2*kh*128), p ushort[T][prow] [1]
// (b at b_off, a at a_off), A_log ushort[vh] [2] (bf16 in the checkpoint), dt_bias ushort[vh] [3],
// g float[T][vh] [4], beta ushort[T][vh] [5], T [6], dd [7].
// Dispatch: (2*kh + 1, T) groups x 128 threads. Group x < kh: q head x; kh..2kh-1: k head x - kh;
// 2kh: the vh gates (needs vh <= 128).
//
// normalize_qk(q, k, inv_scale = 128^-0.5, eps = 1e-6):
//   rms_eps = eps * inv_scale^2 (double, passed to mx.fast.rms_norm as float)
//   q = (inv_scale^2) * rms_norm(q)  -> python scalar weak-typed to bf16: bf16(1/128) = 2^-7
//   k = inv_scale * rms_norm(k)      -> bf16(0.0883883..) = 0x3db5 = 0.08837890625
//   rms_norm (no weight, bf16): one simdgroup, lane l sums x[4l..4l+3]^2 sequentially (N_READS = 4), simd_sum,
//   inv = precise::rsqrt(sum / 128 + rms_eps), out = bf16(x * inv).
// compute_g (compiled): g = exp(-exp(f32(A_log)) * softplus(a + dt_bias)), a + dt_bias and softplus in bf16
//   (softplus = logaddexp(x, 0) on bfloat), product and exp in fp32 (precise::exp).
// beta = sigmoid(b) in bf16 (unary op on bfloat).
kernel void dn_prep(device ushort* cv [[buffer(0)]], device const ushort* p [[buffer(1)]],
                    device const ushort* A_log [[buffer(2)]], device const ushort* dt_bias [[buffer(3)]],
                    device float* g [[buffer(4)]], device ushort* beta [[buffer(5)]],
                    constant uint& T [[buffer(6)]], constant DnDims& dd [[buffer(7)]],
                    uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)  // MLX builds kernels without fast-math; needed for bit-exact S/y
    uint t = gid.y;
    if (t >= T) return;
    const uint kdim = dd.kh * DN_D;
    if (gid.x < 2 * dd.kh) {
        if (tid >= 32) return;  // MLX rms_norm for axis 128 runs one 32-lane simdgroup, 4 reads per lane
        bool is_q = gid.x < dd.kh;
        uint col = is_q ? gid.x * DN_D : kdim + (gid.x - dd.kh) * DN_D;
        device ushort* x = cv + t * dd.conv + col + tid * 4;
        float x0 = bf(x[0]), x1 = bf(x[1]), x2 = bf(x[2]), x3 = bf(x[3]);
        float acc = 0.0f;
        acc += x0 * x0;
        acc += x1 * x1;
        acc += x2 * x2;
        acc += x3 * x3;
        acc = simd_sum(acc);  // MLX's second (cross-simdgroup) simd_sum adds only zeros for one simdgroup
        const float rms_eps = as_type<float>(0x320637bdu);  // float(1e-6 * (128^-0.5)^2)
        float inv = precise::rsqrt(acc / float(DN_D) + rms_eps);
        // q: bf16(2^-7) ; k: bf16(128^-0.5) = 0x3db5
        float sc = is_q ? bf(ushort(0x3c00)) : bf(ushort(0x3db5));
        x[0] = to_bf(bf(to_bf(x0 * inv)) * sc);
        x[1] = to_bf(bf(to_bf(x1 * inv)) * sc);
        x[2] = to_bf(bf(to_bf(x2 * inv)) * sc);
        x[3] = to_bf(bf(to_bf(x3 * inv)) * sc);
        return;
    }
    uint h = tid;
    if (h >= dd.vh) return;
    device const ushort* pr = p + t * dd.prow;
    // beta = sigmoid(b), bf16
    beta[t * dd.vh + h] = to_bf(dn_sigmoid_bf(bf(pr[dd.b_off + h])));
    // softplus(a + dt_bias) in bf16: LogAddExp<bfloat>(x, 0) = max + log1p(exp(min - max)), MLX log1p(bfloat)
    float x = bf(to_bf(bf(pr[dd.a_off + h]) + bf(dt_bias[h])));
    float sp;
    if (isnan(x)) {
        sp = x;
    } else {
        float mx = fmax(x, 0.0f), mn = fmin(x, 0.0f);
        if (mn == -INFINITY || mx == INFINITY) {
            sp = mx;
        } else {
            float e = bf(to_bf(precise::exp(bf(to_bf(mn - mx)))));  // metal::exp(bfloat) -> bf16
            float xp1 = 1.0f + e;                                     // MLX log1p(bfloat16_t)
            float l = (xp1 == 1.0f) ? e : bf(to_bf(e * precise::divide(precise::log(xp1), xp1 - 1.0f)));
            sp = bf(to_bf(mx + l));
        }
    }
    float ea = precise::exp(bf(A_log[h]));  // A_log.astype(f32), exp in fp32
    g[t * dd.vh + h] = precise::exp(-ea * sp);
}

// ---------------------------------------------------------------------------------------------------------------
// dn_recur<CK>: gated delta rule over T tokens, MLX "packed" Dk=128 layout and reduction order
// (gated_delta_step_packed_btree): a simdgroup owns 8 dv rows of one v head, 4 lanes per row, each lane holds
// 32 contiguous dk state elements in registers; dot products = 8 sequential 4-element partials, local pairwise
// tree ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)), then shuffle_xor 1, 2 within the 4-lane row group.
// cv ushort[T][conv] [0] (q at 0, k at kh*128, v at 2*kh*128; v head h uses k/q head h / (vh/kh)),
// g float[T][vh] [1], beta ushort[T][vh] [2], S float[2][vh][128][128] of this layer [3] ([role][hv][dv][dk]),
// y ushort[T][vh][128] [4], ctrl [5], T [6], dd [7].
// Dispatch: vh x 16 simdgroups = 4*vh threadgroups x 128 threads (4 simdgroups each), e.g. groups (4, vh) or
// (4*vh, 1); simdgroup s = linear threadgroup index * 4 + simdgroup index, hv = s / 16, row group = s % 16.
// State is read from S[role] once, kept in registers across the T tokens and written back in place;
// CK: the state after token 0 is also written to S[role ^ 1]. HOLD (dn_recur_hold, ctx_draft.rs): the state is NOT
// written back (the pass only reads it; the accepted rows are replayed in place afterwards).
template <bool CK, bool HOLD>
kernel void dn_recur(device const ushort* cv [[buffer(0)]], device const float* g [[buffer(1)]],
                     device const ushort* beta [[buffer(2)]], device float* S [[buffer(3)]],
                     device ushort* y [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                     constant uint& T [[buffer(6)]], constant DnDims& dd [[buffer(7)]],
                     uint2 gid [[threadgroup_position_in_grid]], uint2 ngr [[threadgroups_per_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)  // MLX builds kernels without fast-math; needed for bit-exact S/y
    constexpr uint lanes_per_row = 4;
    constexpr uint rows_per_simdgroup = 32 / lanes_per_row;  // 8
    constexpr uint values_per_lane = DN_D / lanes_per_row;   // 32
    constexpr uint partials_per_lane = values_per_lane / 4;  // 8
    constexpr uint row_groups = DN_D / rows_per_simdgroup;   // 16

    uint sgi = (gid.y * ngr.x + gid.x) * nsg + sg;
    if (sgi >= dd.vh * row_groups) return;
    uint hv = sgi / row_groups;
    uint hk = hv / (dd.vh / dd.kh);
    uint row_in_simdgroup = lane / lanes_per_row;
    uint lane_in_row = lane & (lanes_per_row - 1);
    uint dv = (sgi % row_groups) * rows_per_simdgroup + row_in_simdgroup;

    const uint kdim = dd.kh * DN_D;
    uint role = ctrl->role & 1u;
    const uint copy = dd.vh * DN_D * DN_D;
    device float* st = S + role * copy + (hv * DN_D + dv) * DN_D + lane_in_row * values_per_lane;
    device float* stc = S + (role ^ 1u) * copy + (hv * DN_D + dv) * DN_D + lane_in_row * values_per_lane;

    device const ushort* q_ = cv + hk * DN_D + lane_in_row * values_per_lane;
    device const ushort* k_ = cv + kdim + hk * DN_D + lane_in_row * values_per_lane;
    device const ushort* v_ = cv + 2 * kdim + hv * DN_D;
    device ushort* y_ = y + hv * DN_D;

    float state[values_per_lane];
    for (uint i = 0; i < values_per_lane; ++i) state[i] = st[i];

    for (uint t = 0; t < T; ++t) {
        float gt = g[t * dd.vh + hv];

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

        float delta = (bf(v_[dv]) - kv_mem) * bf(beta[t * dd.vh + hv]);

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

        q_ += dd.conv;
        k_ += dd.conv;
        v_ += dd.conv;
        y_ += dd.vh * DN_D;
    }

    if (!HOLD) { for (uint i = 0; i < values_per_lane; ++i) st[i] = state[i]; }
}

template [[host_name("dn_recur_ck0")]] kernel void dn_recur<false, false>(
    device const ushort*, device const float*, device const ushort*, device float*, device ushort*,
    device const SdCtrl*, constant uint&, constant DnDims&, uint2, uint2, uint, uint, uint);
template [[host_name("dn_recur_ck1")]] kernel void dn_recur<true, false>(
    device const ushort*, device const float*, device const ushort*, device float*, device ushort*,
    device const SdCtrl*, constant uint&, constant DnDims&, uint2, uint2, uint, uint, uint);
template [[host_name("dn_recur_hold")]] kernel void dn_recur<false, true>(
    device const ushort*, device const float*, device const ushort*, device float*, device ushort*,
    device const SdCtrl*, constant uint&, constant DnDims&, uint2, uint2, uint, uint, uint);

// ===============================================================================================================
// Prefill (PF-B). dn_prep ((2kh+1, T) groups) and dn_recur_ck0 (state in registers across all T tokens: one read
// and one write of S per dispatch) already take T rows in one dispatch and are reused as is. A dn_recur variant
// with q/k/v/g/beta staged through threadgroup memory measured the same (357 vs 359 us at T = 128, 27B shapes:
// the per-token ALU chain bounds it, not loads) and a 2-lanes-per-row variant spilled (3x slower), so neither is
// kept. dn_conv_pf below replaces dn_conv's T-long serial loop. Role convention unchanged: state copy ctrl.role is
// read and updated in place; copy role ^ 1 is never written (no checkpoint in prefill).

// dn_conv_pf: dn_conv parallel over (channel, 8-token block). Thread (c, blk) rebuilds its 3-row history from the
// conv state (blk 0) or from p rows t0-3..t0-1 (same bf16 values dn_conv would carry), then runs dn_conv's per-token
// code for its 8 tokens. All blocks of a channel sit in one threadgroup: the barrier orders the state read (blk 0)
// before the state write (the block holding token T-1).
// Buffers as dn_conv minus ck: p [0], w [1], st [2], out [3], ctrl [4], layer [5], T [6], dd [7].
// Dispatch: groups (ceil(dd.conv / 32), 1) x (32 * ceil(T / 8)) threads (T <= 256).
kernel void dn_conv_pf(device const ushort* p [[buffer(0)]], device const ushort* w [[buffer(1)]],
                       device ushort* st [[buffer(2)]], device ushort* out [[buffer(3)]],
                       device const SdCtrl* ctrl [[buffer(4)]], constant uint& layer [[buffer(5)]],
                       constant uint& T [[buffer(6)]], constant DnDims& dd [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    uint c = tg.x * 32 + (tid & 31u);
    uint t0 = (tid >> 5) * 8;
    bool live = c < dd.conv && t0 < T;
    uint t1 = min(t0 + 8, T);
    uint role = ctrl->role & 1u;
    device ushort* s = st + ((role * dd.layers + layer) * DN_KS) * dd.conv + c;
    ushort r0 = 0, r1 = 0, r2 = 0;
    float w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    if (live) {
        w0 = bf(w[c * 4 + 0]); w1 = bf(w[c * 4 + 1]); w2 = bf(w[c * 4 + 2]); w3 = bf(w[c * 4 + 3]);
        // X = [state rows 0..2 ; p rows 0..T-1]; the window before token t0 is X[t0], X[t0 + 1], X[t0 + 2]
        r0 = t0 >= 3 ? p[(t0 - 3) * dd.prow + c] : s[t0 * dd.conv];
        r1 = t0 >= 2 ? p[(t0 - 2) * dd.prow + c] : s[(t0 + 1) * dd.conv];
        r2 = t0 >= 1 ? p[(t0 - 1) * dd.prow + c] : s[(t0 + 2) * dd.conv];
        for (uint t = t0; t < t1; ++t) {
            ushort x = p[t * dd.prow + c];
            float acc = 0.0f;
            acc += bf(r0) * w0;
            acc += bf(r1) * w1;
            acc += bf(r2) * w2;
            acc += bf(x) * w3;
            float cv = bf(to_bf(acc));
            out[t * dd.conv + c] = to_bf(cv * dn_sigmoid_bf(cv));
            r0 = r1; r1 = r2; r2 = x;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (live && t1 == T) { s[0] = r0; s[dd.conv] = r1; s[2 * dd.conv] = r2; }
}

// ===============================================================================================================
// Deep speculative verify (spec_deep.rs): per-row rollback by REPLAY. A verify pass over K = d + 1 rows runs with the
// row-0 checkpoint (dn_conv ck = 0, dn_recur_ck1) and keeps, per DeltaNet layer, the raw in_proj columns [qkv | b | a]
// of rows 1..K-2 (dn_save_rows). After accepting j < d drafts the host flips ctrl.role to the row-0 checkpoint and
// replays rows 1..j with the unchanged dn_conv / dn_prep / dn_recur_ck0 over the saved rows (a DnDims with
// prow = layers * w, b_off = conv, a_off = conv + vh). By K-invariance the result is bit-identical to feeding
// those rows one by one; no matmul is re-run.

// dn_save_rows: p ushort[T][prow] [0] (this layer's in_proj output), sv ushort[T-2][layers][w] [1] (w = conv + 2*vh,
// row r = pass row r + 1), layer [2] (DeltaNet index), T [3], dd [4]. Needs a_off == b_off + vh (DnDims::from_cfg).
// Dispatch: threads (w, T - 2).
kernel void dn_save_rows(device const ushort* p [[buffer(0)]], device ushort* sv [[buffer(1)]],
                         constant uint& layer [[buffer(2)]], constant uint& T [[buffer(3)]],
                         constant DnDims& dd [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint w = dd.conv + 2 * dd.vh;
    uint c = gid.x, r = gid.y;
    if (c >= w || r + 2 > T) return;
    uint src = c < dd.conv ? c : dd.b_off + (c - dd.conv);
    sv[((ulong)r * dd.layers + layer) * w + c] = p[(ulong)(r + 1) * dd.prow + src];
}

// ===============================================================================================================
// Context-draft verify (ctx_draft.rs): a K-row pass that only READS the DeltaNet state (dn_conv ck = DN_HOLD,
// dn_recur_hold) and keeps the raw in_proj columns [qkv | b | a] of ALL K rows; after the CPU picks the accepted
// rows 0..je, those je + 1 rows are replayed in place with dn_conv / dn_prep / dn_recur_ck0 over the saved rows (as
// the deep spec replay). No second state copy: the live state is never overwritten by rejected rows.
// dn_save_all: p ushort[T][prow] [0], sv ushort[T][layers][w] [1] (w = conv + 2*vh, row r = pass row r), layer [2],
// T [3], dd [4]. Dispatch: threads (w, T).
kernel void dn_save_all(device const ushort* p [[buffer(0)]], device ushort* sv [[buffer(1)]],
                        constant uint& layer [[buffer(2)]], constant uint& T [[buffer(3)]],
                        constant DnDims& dd [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint w = dd.conv + 2 * dd.vh;
    uint c = gid.x, r = gid.y;
    if (c >= w || r >= T) return;
    uint src = c < dd.conv ? c : dd.b_off + (c - dd.conv);
    sv[((ulong)r * dd.layers + layer) * w + c] = p[(ulong)r * dd.prow + src];
}
