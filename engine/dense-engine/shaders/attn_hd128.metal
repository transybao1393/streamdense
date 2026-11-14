// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (rms_norm and rope kernels, via engine/shaders/attn_gated.metal: partition, reduction order, rounding).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// attn_hd128.metal - dense-engine: full / sliding-window attention for dense models at head dim HD = 64, 96, 128, 160
// or 256:
// optional q/k RMSNorm + RoPE (any rdims <= HD) + KV append, split-K attention over fixed 256-key chunks (bf16 KV at
// every HD; q8 KV at HD 128 only), and the reduce, plain or (HD 128) fused with the sigmoid output gate.
// Needs engine/shaders/common_bf16.metal first in the concatenated source (bf, to_bf, SdCtrl). Must come before
// attn_hd128_flash.metal and attn_hd128_sdpa.metal, which use the function constants and helpers declared here.
// Port of engine/shaders/attn_gated.metal (attn_rope_invf, attn_prep, attn_partial, attn_reduce_gate) from HD = 256
// to a template over HD. Kernel names: attn{HD}_... (the HD 128 names are the original attn128_ ones).
//
// Per-pipeline function constants (all optional; undefined = the Qwen behaviour, compiled out):
//   QD_FC_SCALE   [[function_constant(20)]] float  attention scale (default HD^-0.5 rounded to f32)
//   QD_FC_SOFTCAP [[function_constant(21)]] float  score soft cap: s = cap * tanh(s / cap) after scaling
//   QD_FC_WINDOW  [[function_constant(22)]] uint   sliding window W: key j visible to the query at position i iff
//                                                  j <= i and i - j < W (HF Gemma 2 / 3 sliding mask)
// attn{HD}_prep only (other families; src/gpu.rs fc_key_prep):
//   QD_FC_ROPE    [[function_constant(23)]] uint   bit 0: traditional RoPE (pairs (2i, 2i + 1), mlx traditional=True);
//                                                  bit 1: no RoPE on this layer (q / k pass through)
//   QD_FC_MSCALE  [[function_constant(24)]] float  q / k dims < rdims multiplied by it (bf16) before RoPE (mlx_lm
//                                                  SuScaledRoPE, Phi-3 LongRoPE)
//   QD_FC_QKN     [[function_constant(25)]] uint   1: QK-norm is a LayerNorm (no bias) per head with one weight per head
//                                                  (q_norm [nh][HD], k_norm [nkv][HD]; StableLM qk_layernorm)
// Set by the host through the pipeline key "name#s=<f32 bits hex>,c=<bits>,w=<W>" (src/gpu.rs Gpu::pipeline).
//
// Differences from attn_gated (everything else, rounding points included, is the same):
//   - scale = HD^-0.5 by default (128: f32 0.08838834764831845, as MLX passes head_dim**-0.5 as a float).
//   - attn{HD}_prep: HD threads per (head, token); flags [14]: bit 0 = QK-norm (else q/k pass through unnormed),
//     bit 1 = gated layout (q | gate interleaved per head; HD 128 only). rdims may be HD (full rotary) or less.
//   - attn{HD}_partial: NG = HD / 32 slices of 32 dims per cached row (one q8 group each) on NL = NG rounded up to a
//     power of two lanes (HD 96: 4, HD 160: 8; lanes sub >= NG idle), 32 / NL rows per simdgroup, 128 / NL per step.
//   - attn{HD}_reduce (no gate) and attn128_reduce_gate: same buffer layout; the plain one ignores p / p_stride.
// Layouts: p bf16 [T][P], P = nh*QS + 2*nkv*HD with QS = 2*HD (gated: q|gate per head) or HD (plain), then k, then v;
// qr bf16 [T][nh][HD]; KV [nkv][max_ctx][HD] (bf16, or u8 + half2 (scale, min) per 32-dim group [nkv][max_ctx][HD/32]);
// po float4 [T][nh][nch][HD/4], pml float2 (m, l) [T][nh][nch]; ao bf16 [T][nh*HD].
// K-invariance as attn_gated: per query row the arithmetic does not depend on K or nch.

constant float QD_FC_SCALE [[function_constant(20)]];
constant float QD_FC_SOFTCAP [[function_constant(21)]];
constant uint QD_FC_WINDOW [[function_constant(22)]];
constant bool QD_HAS_SCALE = is_function_constant_defined(QD_FC_SCALE);
constant bool QD_HAS_SOFTCAP = is_function_constant_defined(QD_FC_SOFTCAP);
constant bool QD_HAS_WINDOW = is_function_constant_defined(QD_FC_WINDOW);
constant uint QD_FC_ROPE [[function_constant(23)]];
constant float QD_FC_MSCALE [[function_constant(24)]];
constant uint QD_FC_QKN [[function_constant(25)]];
constant bool QD_HAS_ROPE = is_function_constant_defined(QD_FC_ROPE);
constant bool QD_HAS_MSCALE = is_function_constant_defined(QD_FC_MSCALE);
constant bool QD_HAS_QKN = is_function_constant_defined(QD_FC_QKN);

/// HD^-0.5 rounded to f32 (MLX gets head_dim**-0.5 as a float), or the QD_FC_SCALE constant.
static inline float qd_scale(uint hd) {
    if (QD_HAS_SCALE) return QD_FC_SCALE;
    return hd == 64 ? 0.125f : (hd == 256 ? 0.0625f : (hd == 96 ? 0.10206207261596575f
         : (hd == 160 ? 0.07905694150420949f : 0.08838834764831845f)));
}
/// Score soft cap (identity unless QD_FC_SOFTCAP is set).
static inline float qd_softcap(float s) { return QD_HAS_SOFTCAP ? QD_FC_SOFTCAP * precise::tanh(s / QD_FC_SOFTCAP) : s; }
/// Key j visible to the query at position qp: causal, and inside the sliding window when QD_FC_WINDOW is set.
static inline bool qd_key_ok(uint j, uint qp) { return j <= qp && (!QD_HAS_WINDOW || j + QD_FC_WINDOW > qp); }
/// First key visible to the query at position qp.
static inline uint qd_key_lo(uint qp) { return (QD_HAS_WINDOW && qp + 1 > QD_FC_WINDOW) ? qp + 1 - QD_FC_WINDOW : 0u; }

constant uint AH_CH = 256;        // fixed split-K chunk
constant uint AH_PAD = 33;        // padded slice stride in threadgroup memory
constant uint AH_QKNORM = 1u;     // prep flag bits
constant uint AH_GATED = 2u;

// ---- RoPE inverse frequencies (mx.fast.rope): d = i / (rdims/2), invf = exp2(-d * log2(base)). One thread per
// frequency (rdims/2 threads).
kernel void attn128_rope_invf(device float* invf [[buffer(0)]], constant float& log2base [[buffer(1)]],
                              uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
#pragma METAL fp math_mode(safe)
    float d = float(i) / float(n);
    invf[i] = exp2(-d * log2base);
}

// ---- attn{HD}_prep: threadgroup (head hh, token t), HD threads (thread d = one dim).
// hh < nh: q head -> [RMSNorm(q_norm)] -> RoPE -> qr. hh >= nh: kv head hh - nh -> [RMSNorm(k_norm)] -> RoPE -> K slot,
// V copied, both at pos = ctrl.pos + t. Dispatch (nh + nkv, T) groups x HD.
// Buffers: p [0], q_norm [1], k_norm [2] (ignored without QK-norm), qr [3], Kc [4], Vc [5], Ksb [6], Vsb [7] (Q8 only),
// invf [8], ctrl [9], max_ctx [10], eps [11], nkv [12], rdims [13], flags [14].
// RMSNorm (mx.fast.rms_norm): r = precise::rsqrt(sum(x^2)/HD + eps); n = bf(bf(w) * bf(x * r)); without QK-norm n = x.
// (Gemma's (1 + w) weights are folded at load.) RoPE non-traditional over the first rdims dims, pairs (i, i + rdims/2);
// dims >= rdims pass through. Function constants 23..25 (see the top): traditional pairs or no RoPE, the su multiplier
// (applied after the norm, before RoPE), per-head LayerNorm QK-norm (mean / variance over HD, n = bf(w * bf((x - m) r))).
template <bool Q8, uint HD>
kernel void attn_prep(device const ushort* p [[buffer(0)]], device const ushort* qnw [[buffer(1)]],
                      device const ushort* knw [[buffer(2)]], device ushort* qr [[buffer(3)]],
                      device uchar* Kc [[buffer(4)]], device uchar* Vc [[buffer(5)]],
                      device half2* Ksb [[buffer(6)]], device half2* Vsb [[buffer(7)]],
                      device const float* invf [[buffer(8)]], device const SdCtrl* ctrl [[buffer(9)]],
                      constant uint& maxctx [[buffer(10)]], constant float& eps [[buffer(11)]],
                      constant uint& nkv [[buffer(12)]], constant uint& rdims [[buffer(13)]],
                      constant uint& flags [[buffer(14)]],
                      uint2 ntgs [[threadgroups_per_grid]], uint2 tg [[threadgroup_position_in_grid]],
                      uint d [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    threadgroup float part[HD / 32];
    threadgroup float tn[HD];
    const bool qkn = (flags & AH_QKNORM) != 0, gated = (flags & AH_GATED) != 0;  // uniform per dispatch
    const uint QS = gated ? 2 * HD : HD;                    // q row stride per head in p
    uint nh = ntgs.x - nkv, hh = tg.x, t = tg.y;
    uint pos = ctrl->pos + t;
    uint P = nh * QS + 2 * nkv * HD;
    device const ushort* row = p + t * P;
    bool isq = hh < nh;                                     // uniform per threadgroup
    uint kv = hh - nh;
    device const ushort* src = isq ? row + hh * QS : row + nh * QS + kv * HD;
    float x = bf(src[d]);
    float nv = x;
    if (qkn && QD_HAS_QKN && QD_FC_QKN == 1u) {
        // per-head LayerNorm, no bias (mx.fast.layer_norm(x, None, None, eps), then w * that in bf16)
        float w = bf(isq ? qnw[hh * HD + d] : knw[kv * HD + d]);
        float sx = simd_sum(x);
        if (lane == 0) part[sg] = sx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint s2 = 0; s2 < HD / 32; s2++) tot += part[s2];
        float xc = x - tot / float(HD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sv = simd_sum(xc * xc);
        if (lane == 0) part[sg] = sv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float var = 0;
        for (uint s2 = 0; s2 < HD / 32; s2++) var += part[s2];
        float r = precise::rsqrt(var / float(HD) + eps);
        nv = bf(to_bf(w * bf(to_bf(xc * r))));
    } else if (qkn) {
        float w = bf(isq ? qnw[d] : knw[d]);
        // sum of squares as MLX rms_single_row (N_READS = 4): thread d < HD / 4 sums dims 4d .. 4d + 3
        float acc = 0;
        if (d * 4 + 4 <= HD) {
            for (uint i = 0; i < 4; i++) { float xi = bf(src[d * 4 + i]); acc += xi * xi; }
        }
        float ss = simd_sum(acc);
        if (lane == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint s2 = 0; s2 < HD / 32; s2++) tot += part[s2];
        float r = precise::rsqrt(tot / float(HD) + eps);
        nv = bf(to_bf(w * bf(to_bf(x * r))));
    }
    if (QD_HAS_MSCALE && d < rdims) nv = bf(to_bf(nv * QD_FC_MSCALE));
    tn[d] = nv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint half_r = rdims / 2;
    float o = nv;
    if (QD_HAS_ROPE && (QD_FC_ROPE & 2u) != 0) {
        // no RoPE on this layer
    } else if (QD_HAS_ROPE && (QD_FC_ROPE & 1u) != 0) {
        if (d < rdims) {
            // traditional: pair (2i, 2i + 1) rotated by frequency i
            float theta = float(pos) * invf[d / 2];
            float c = fast::cos(theta), s = fast::sin(theta);
            float x1 = tn[d & ~1u], x2 = tn[d | 1u];
            o = (d & 1u) == 0 ? x1 * c - x2 * s : x1 * s + x2 * c;
        }
    } else if (d < rdims) {
        uint i = d < half_r ? d : d - half_r;
        float theta = float(pos) * invf[i];
        float c = fast::cos(theta), s = fast::sin(theta);
        float x1 = tn[i], x2 = tn[i + half_r];
        o = d < half_r ? x1 * c - x2 * s : x1 * s + x2 * c;
    }
    ushort ob = to_bf(o);
    if (isq) { qr[(t * nh + hh) * HD + d] = ob; return; }
    ushort vb = row[nh * QS + nkv * HD + kv * HD + d];
    uint slot = (kv * maxctx + pos) * HD + d;
    if (!Q8) { ((device ushort*)Kc)[slot] = ob; ((device ushort*)Vc)[slot] = vb; return; }
    uint gslot = (kv * maxctx + pos) * (HD / 32) + sg;
    for (uint u = 0; u < 2; u++) {
        float xv = bf(u == 0 ? ob : vb);
        float lo = simd_min(xv), hi = simd_max(xv);
        half sc16 = half(max(hi - lo, 1e-8f) / 255.0f), lo16 = half(lo);
        float scf = float(sc16) > 0.0f ? float(sc16) : 1.0f;
        uchar qv = uchar(clamp(rint((xv - float(lo16)) / scf), 0.0f, 255.0f));
        if (u == 0) { Kc[slot] = qv; if (lane == 0) Ksb[gslot] = half2(sc16, lo16); }
        else { Vc[slot] = qv; if (lane == 0) Vsb[gslot] = half2(sc16, lo16); }
    }
}
#define AHPREP(NAME_, Q_, HD_) template [[host_name(NAME_)]] kernel void attn_prep<Q_, HD_>(device const ushort*, \
    device const ushort*, device const ushort*, device ushort*, device uchar*, device uchar*, device half2*, \
    device half2*, device const float*, device const SdCtrl*, constant uint&, constant float&, constant uint&, \
    constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
AHPREP("attn128_prep_b16", false, 128) AHPREP("attn128_prep_q8", true, 128)
AHPREP("attn64_prep_b16", false, 64) AHPREP("attn256_prep_b16", false, 256)
AHPREP("attn96_prep_b16", false, 96) AHPREP("attn160_prep_b16", false, 160)

// ---- attn{HD}_partial: threadgroup (chunk c, q head h), 128 threads = 4 simdgroups; NG = HD/32 slices of 32 dims per
// cached row (one q8 group each) on NL lanes (NG rounded up to a power of two for the xor butterflies: NL = NG at HD
// 64 / 128 / 256, 4 at HD 96, 8 at HD 160; lanes with sub >= NG hold zeros and write nothing), RPS = 32/NL rows per
// simdgroup at a time -> 4 * RPS rows per step over the chunk's 256 keys.
// Query t (position pos + t, pos = ctrl.pos) sees keys qd_key_ok(j, pos + t) (0 .. pos + t, or the window).
// Keys of chunk c: [256c, min(256c + 256, pos + K)).
// Buffers: qr [0], Kc [1], Vc [2] (bf16 as ushort, or u8), Ksb [3], Vsb [4] (Q8), po [5], pml [6], ctrl [7],
// max_ctx [8], grp [9] (q heads per kv head). Dispatch (nch, nh) groups x 128; nch >= ceil((pos + K) / 256).
// Scores: s = sum_e (scale * q_e) * k_e in fp32 (fma chain over the lane's 32 dims, then an xor butterfly over the
// NG lanes); Q8: s = sc * (q . kq) + bias * sum(q); then the soft cap if set. Chunk softmax: m = max, p =
// fast::exp(s - m), l = sum p, o = sum p * v (fp32). Chunks with no visible key for query t write m = -inf, l = 0, o = 0.
template <bool Q8, uint K, uint HD>
kernel void attn_partial(device const ushort* qr [[buffer(0)]], device const uchar* Kc [[buffer(1)]],
                         device const uchar* Vc [[buffer(2)]], device const half2* Ksb [[buffer(3)]],
                         device const half2* Vsb [[buffer(4)]], device float4* po [[buffer(5)]],
                         device float2* pml [[buffer(6)]], device const SdCtrl* ctrl [[buffer(7)]],
                         constant uint& maxctx [[buffer(8)]], constant uint& grp [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr uint NG = HD / 32, PD = AH_PAD;
    constexpr uint NL = NG <= 2 ? NG : (NG <= 4 ? 4 : 8);   // lanes per row (power of two >= NG)
    constexpr uint RPS = 32 / NL, STEP = 4 * RPS;
    constexpr uint TB = K < 2 ? K : 2;                      // queries per V sweep
    threadgroup float tq[K][NG * AH_PAD];                   // scaled q, slice-padded
    threadgroup float sc[K][AH_CH];
    threadgroup float redm[K][4];
    threadgroup float red[4][NG * AH_PAD];
    threadgroup float redl[4];
    uint c = tg.x, h = tg.y, nch = ntgs.x, nh = ntgs.y, kvh = h / grp;
    uint pos = ctrl->pos;                                   // position of query 0
    uint nmax = pos + K;                                    // keys visible to the last query
    uint j0 = c * AH_CH, j1 = min(j0 + AH_CH, nmax);
    uint rs = lane / NL, sub = lane % NL;                   // row slot in the simdgroup, 32-dim slice of the row
    const bool act = sub < NG;                              // always true when NL == NG
    const float scale = qd_scale(HD);
    for (uint t = 0; t < K; t++)
        for (uint e = tid; e < HD; e += 128) tq[t][(e / 32) * PD + e % 32] = scale * bf(qr[(t * nh + h) * HD + e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs[K], mloc[K];
    for (uint t = 0; t < K; t++) {
        mloc[t] = -INFINITY; qs[t] = 0;
        if (Q8 && act) for (uint e = 0; e < 32; e++) qs[t] += tq[t][sub * PD + e];
    }
    for (uint j = j0 + sg * RPS + rs; j < j1; j += STEP) {
        uint row = kvh * maxctx + j;
        float kf[32];
        float2 sb = float2(1.0f, 0.0f);
        if (!act) {
            for (uint e = 0; e < 32; e++) kf[e] = 0.0f;
        } else if (Q8) {
            device const uint4* kp = (device const uint4*)(Kc + row * HD + sub * 32);
            sb = float2(Ksb[row * NG + sub]);
            for (uint u = 0; u < 2; u++) {
                uint4 wv = kp[u];
                for (uint v = 0; v < 4; v++) {
                    float4 f = float4(as_type<uchar4>(wv[v]));
                    kf[16 * u + 4 * v] = f.x; kf[16 * u + 4 * v + 1] = f.y; kf[16 * u + 4 * v + 2] = f.z; kf[16 * u + 4 * v + 3] = f.w;
                }
            }
        } else {
            device const uint4* kp = (device const uint4*)(Kc + (row * HD + sub * 32) * 2);
            for (uint u = 0; u < 4; u++) {
                uint4 wv = kp[u];
                for (uint v = 0; v < 4; v++) {
                    kf[8 * u + 2 * v] = as_type<float>(wv[v] << 16);
                    kf[8 * u + 2 * v + 1] = as_type<float>(wv[v] & 0xffff0000u);
                }
            }
        }
        for (uint t = 0; t < K; t++) {
            float dd = 0;
            if (act) for (uint e = 0; e < 32; e++) dd = fma(tq[t][sub * PD + e], kf[e], dd);
            if (Q8) dd = fma(sb.x, dd, sb.y * qs[t]);
            for (uint s2 = 1; s2 < NL; s2 <<= 1) dd += simd_shuffle_xor(dd, s2);
            dd = qd_softcap(dd);
            if (qd_key_ok(j, pos + t)) {
                mloc[t] = max(mloc[t], dd);
                if (sub == 0) sc[t][j - j0] = dd;
            }
        }
    }
    for (uint t = 0; t < K; t++) { float mm = simd_max(mloc[t]); if (lane == 0) redm[t][sg] = mm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m[K];
    for (uint t = 0; t < K; t++) m[t] = max(max(redm[t][0], redm[t][1]), max(redm[t][2], redm[t][3]));
    for (uint tb = 0; tb < K; tb += TB) {
        float acc[TB][32], l[TB], ab[TB];
        for (uint u = 0; u < TB; u++) { l[u] = 0; ab[u] = 0; for (uint e = 0; e < 32; e++) acc[u][e] = 0; }
        for (uint j = j0 + sg * RPS + rs; j < j1; j += STEP) {
            uint row = kvh * maxctx + j;
            float vf[32];
            float2 sb = float2(1.0f, 0.0f);
            if (!act) {
                for (uint e = 0; e < 32; e++) vf[e] = 0.0f;
            } else if (Q8) {
                device const uint4* vp = (device const uint4*)(Vc + row * HD + sub * 32);
                sb = float2(Vsb[row * NG + sub]);
                for (uint u = 0; u < 2; u++) {
                    uint4 wv = vp[u];
                    for (uint v = 0; v < 4; v++) {
                        float4 f = float4(as_type<uchar4>(wv[v]));
                        vf[16 * u + 4 * v] = f.x; vf[16 * u + 4 * v + 1] = f.y; vf[16 * u + 4 * v + 2] = f.z; vf[16 * u + 4 * v + 3] = f.w;
                    }
                }
            } else {
                device const uint4* vp = (device const uint4*)(Vc + (row * HD + sub * 32) * 2);
                for (uint u = 0; u < 4; u++) {
                    uint4 wv = vp[u];
                    for (uint v = 0; v < 4; v++) {
                        vf[8 * u + 2 * v] = as_type<float>(wv[v] << 16);
                        vf[8 * u + 2 * v + 1] = as_type<float>(wv[v] & 0xffff0000u);
                    }
                }
            }
            for (uint u = 0; u < TB; u++) {
                uint t = tb + u;
                if (t < K && qd_key_ok(j, pos + t)) {       // visible key for query t; odd K: no t = K
                    float pe = fast::exp(sc[t][j - j0] - m[t]);
                    l[u] += pe;
                    float ps = Q8 ? pe * sb.x : pe;
                    for (uint e = 0; e < 32; e++) acc[u][e] = fma(ps, vf[e], acc[u][e]);
                    if (Q8) ab[u] = fma(pe, sb.y, ab[u]);
                }
            }
        }
        for (uint u = 0; u < TB; u++) {
            uint t = tb + u;
            if (t >= K) break;                              // odd K: the last sweep has one query (uniform per group)
            // sum the RPS row slots (lanes sub, sub + NL, ..., sub + 32 - NL)
            for (uint e = 0; e < 32; e++) {
                for (uint s2 = NL; s2 < 32; s2 <<= 1) acc[u][e] += simd_shuffle_xor(acc[u][e], s2);
            }
            for (uint s2 = NL; s2 < 32; s2 <<= 1) l[u] += simd_shuffle_xor(l[u], s2);
            if (Q8) { for (uint s2 = NL; s2 < 32; s2 <<= 1) ab[u] += simd_shuffle_xor(ab[u], s2); }
            if (rs == 0 && act) for (uint e = 0; e < 32; e++) red[sg][sub * PD + e] = Q8 ? acc[u][e] + ab[u] : acc[u][e];
            if (lane == 0) redl[sg] = l[u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint slot = (t * nh + h) * nch + c;
            if (tid < HD / 4) {
                uint s0 = (tid / 8) * PD + (tid % 8) * 4;   // dims 4 tid .. 4 tid + 3
                float4 o = 0;
                for (uint s2 = 0; s2 < 4; s2++) o += float4(red[s2][s0], red[s2][s0 + 1], red[s2][s0 + 2], red[s2][s0 + 3]);
                po[slot * (HD / 4) + tid] = o;
            }
            if (tid == 0) pml[slot] = float2(m[t], ((redl[0] + redl[1]) + redl[2]) + redl[3]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
#define AHP(Q_, N_, K_, HD_) template [[host_name("attn" #HD_ "_partial_k" #K_ "_" #N_)]] kernel void \
    attn_partial<Q_, K_, HD_>(device const ushort*, device const uchar*, device const uchar*, device const half2*, \
    device const half2*, device float4*, device float2*, device const SdCtrl*, constant uint&, constant uint&, uint2, \
    uint2, uint, uint, uint);
#define AHP8(Q_, N_, HD_) AHP(Q_, N_, 1, HD_) AHP(Q_, N_, 2, HD_) AHP(Q_, N_, 3, HD_) AHP(Q_, N_, 4, HD_) \
    AHP(Q_, N_, 5, HD_) AHP(Q_, N_, 6, HD_) AHP(Q_, N_, 7, HD_) AHP(Q_, N_, 8, HD_)
AHP8(false, b16, 128) AHP8(true, q8, 128) AHP8(false, b16, 64) AHP8(false, b16, 256)
AHP8(false, b16, 96) AHP8(false, b16, 160)

// ---- attn{HD}_reduce / attn128_reduce_gate: threadgroup (q head h, token t), HD/4 threads x float4.
// Combine the chunks (skipping empty ones, l = 0): M = max m, w = fast::exp(m - M), L = sum w l, o = sum w o
// (chunk order, fp32), ob = bf(o / L) (sdpa output, bf16). Plain: ao = ob. Gated: MLX bf16 sigmoid on the gate
// (e = bf(precise::exp(|g|)), y = bf(1 / bf(1 + e)), s = g < 0 ? y : bf(1 - y)), ao = bf(ob * s).
// Buffers: po [0], pml [1], p [2] (gated only: gate of head h, token t at p + t * p_stride + h * 2 * HD + HD), ao [3]
// (bf16 [T][nh * HD]), nch [4], p_stride [5] (gated only: elements per p row = nh*2*HD + 2*nkv*HD).
template <bool G, uint HD>
kernel void attn_reduce(device const float4* po [[buffer(0)]], device const float2* pml [[buffer(1)]],
                        device const ushort* p [[buffer(2)]], device ushort* ao [[buffer(3)]],
                        constant uint& nch [[buffer(4)]], constant uint& p_stride [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    uint h = tg.x, t = tg.y, nh = ntgs.x, base = (t * nh + h) * nch;
    float M = -INFINITY;
    for (uint c = 0; c < nch; c++) { float2 ml = pml[base + c]; if (ml.y > 0.0f) M = max(M, ml.x); }
    float L = 0;
    float4 o = 0;
    for (uint c = 0; c < nch; c++) {
        float2 ml = pml[base + c];
        if (ml.y > 0.0f) {
            float w = fast::exp(ml.x - M);
            L = fma(w, ml.y, L);
            o = fma(float4(w), po[(base + c) * (HD / 4) + lane], o);
        }
    }
    float4 r = L == 0.0f ? o : o / L;
    device ushort* out = ao + t * nh * HD + h * HD + lane * 4;
    if (!G) { for (uint i = 0; i < 4; i++) out[i] = to_bf(r[i]); return; }
    device const ushort* g = p + t * p_stride + h * 2 * HD + HD + lane * 4;
    for (uint i = 0; i < 4; i++) {
        float ob = bf(to_bf(r[i]));
        float gv = bf(g[i]);
        float e = bf(to_bf(precise::exp(fabs(gv))));
        float y = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
        float s = gv < 0.0f ? y : bf(to_bf(1.0f - y));
        out[i] = to_bf(ob * s);
    }
}
#define AHR(NAME_, G_, HD_) template [[host_name(NAME_)]] kernel void attn_reduce<G_, HD_>(device const float4*, \
    device const float2*, device const ushort*, device ushort*, constant uint&, constant uint&, uint2, uint2, uint);
AHR("attn128_reduce", false, 128) AHR("attn128_reduce_gate", true, 128)
AHR("attn64_reduce", false, 64) AHR("attn256_reduce", false, 256)
AHR("attn96_reduce", false, 96) AHR("attn160_reduce", false, 160)
