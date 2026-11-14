// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (rms_norm and rope kernels, via engine/shaders/attn_gated.metal: partition, reduction order, rounding).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// attn_hd128.metal - qwen-dense: full attention at head dim 128 for ordinary Qwen dense models (Qwen2 / 2.5 / 3):
// optional q/k RMSNorm (Qwen3 yes, Qwen2 no) + RoPE (full rotary for these models) + KV append, split-K attention over
// fixed 256-key chunks (bf16 or q8 KV), and the reduce, plain (Qwen dense) or fused with the sigmoid output gate.
// Needs engine/shaders/common_bf16.metal first in the concatenated source (bf, to_bf, SdCtrl).
// Port of engine/shaders/attn_gated.metal (attn_rope_invf, attn_prep, attn_partial, attn_reduce_gate) from HD = 256
// to HD = 128. Kernel names carry an `attn128_` prefix so both files can live in one library.
// Not ported: attn_sdpa_vec_b16 (MLX bit-exact decode mirror) and attn_flash_b16 (prefill).
//
// Differences from attn_gated (everything else, rounding points included, is the same):
//   - HD = 128, scale = 128^-0.5 (f32 0.08838834764831845, as MLX passes head_dim**-0.5 as a float; not exact,
//     unlike 1/16 at HD = 256).
//   - attn128_prep: 128 threads per (head, token); flags [14]: bit 0 = QK-norm (else q/k pass through unnormed),
//     bit 1 = gated layout (q | gate interleaved per head; else plain q). rdims may be HD (full rotary) or less.
//   - attn128_partial: 4 lanes per cached row (32 dims each = one q8 group), 8 rows per simdgroup, 32 rows per step.
//   - attn128_reduce (no gate) and attn128_reduce_gate: same buffer layout; the plain one ignores p / p_stride.
// Layouts: p bf16 [T][P], P = nh*QS + 2*nkv*HD with QS = 2*HD (gated: q|gate per head) or HD (plain), then k, then v;
// qr bf16 [T][nh][HD]; KV [nkv][max_ctx][HD] (bf16, or u8 + half2 (scale, min) per 32-dim group [nkv][max_ctx][HD/32]);
// po float4 [T][nh][nch][HD/4], pml float2 (m, l) [T][nh][nch]; ao bf16 [T][nh*HD].
// K-invariance as attn_gated: per query row the arithmetic does not depend on K or nch.

constant uint AH_HD = 128;        // head dim; the lane mapping below assumes it
constant uint AH_CH = 256;        // fixed split-K chunk
constant uint AH_NG = AH_HD / 32; // 32-dim groups per head (q8 KV) = row slices per row (4 lanes x 32 dims)
constant uint AH_PAD = 33;        // padded slice stride in threadgroup memory
constant float AH_SCALE = 0.08838834764831845f; // HD^-0.5 for HD = 128 (rounded to f32)
constant uint AH_QKNORM = 1u;     // prep flag bits
constant uint AH_GATED = 2u;

// ---- RoPE inverse frequencies (mx.fast.rope): d = i / (rdims/2), invf = exp2(-d * log2(base)). One thread per
// frequency (rdims/2 threads; 64 for full rotary at HD 128).
kernel void attn128_rope_invf(device float* invf [[buffer(0)]], constant float& log2base [[buffer(1)]],
                              uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
#pragma METAL fp math_mode(safe)
    float d = float(i) / float(n);
    invf[i] = exp2(-d * log2base);
}

// ---- attn128_prep: threadgroup (head hh, token t), HD = 128 threads (thread d = one dim).
// hh < nh: q head -> [RMSNorm(q_norm)] -> RoPE -> qr. hh >= nh: kv head hh - nh -> [RMSNorm(k_norm)] -> RoPE -> K slot,
// V copied, both at pos = ctrl.pos + t. Dispatch (nh + nkv, T) groups x 128.
// Buffers: p [0], q_norm [1], k_norm [2] (ignored without QK-norm), qr [3], Kc [4], Vc [5], Ksb [6], Vsb [7] (Q8 only),
// invf [8], ctrl [9], max_ctx [10], eps [11], nkv [12], rdims [13], flags [14].
// RMSNorm (mx.fast.rms_norm): r = precise::rsqrt(sum(x^2)/HD + eps); n = bf(bf(w) * bf(x * r)); without QK-norm n = x.
// RoPE non-traditional over the first rdims dims, pairs (i, i + rdims/2); dims >= rdims pass through.
template <bool Q8>
kernel void attn128_prep(device const ushort* p [[buffer(0)]], device const ushort* qnw [[buffer(1)]],
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
    const uint HD = AH_HD;
    threadgroup float part[AH_HD / 32];
    threadgroup float tn[AH_HD];
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
    if (qkn) {
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
    tn[d] = nv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint half_r = rdims / 2;
    float o = nv;
    if (d < rdims) {
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
template [[host_name("attn128_prep_b16")]] kernel void attn128_prep<false>(device const ushort*, device const ushort*,
    device const ushort*, device ushort*, device uchar*, device uchar*, device half2*, device half2*, device const float*,
    device const SdCtrl*, constant uint&, constant float&, constant uint&, constant uint&, constant uint&, uint2, uint2,
    uint, uint, uint);
template [[host_name("attn128_prep_q8")]] kernel void attn128_prep<true>(device const ushort*, device const ushort*,
    device const ushort*, device ushort*, device uchar*, device uchar*, device half2*, device half2*, device const float*,
    device const SdCtrl*, constant uint&, constant float&, constant uint&, constant uint&, constant uint&, uint2, uint2,
    uint, uint, uint);

// ---- attn128_partial: threadgroup (chunk c, q head h), 128 threads = 4 simdgroups; 4 lanes per cached row (32 dims
// per lane = one q8 group), 8 rows per simdgroup at a time -> 32 rows per step over the chunk's 256 keys.
// Query t (position pos + t, pos = ctrl.pos) sees keys 0 .. pos + t. Keys of chunk c: [256c, min(256c + 256, pos + K)).
// Buffers: qr [0], Kc [1], Vc [2] (bf16 as ushort, or u8), Ksb [3], Vsb [4] (Q8), po [5], pml [6], ctrl [7],
// max_ctx [8], grp [9] (q heads per kv head). Dispatch (nch, nh) groups x 128; nch >= ceil((pos + K) / 256).
// Scores: s = sum_e (scale * q_e) * k_e in fp32 (fma chain over the lane's 32 dims, then a 2-step xor butterfly over
// the 4 lanes); Q8: s = sc * (q . kq) + bias * sum(q). Chunk softmax: m = max, p = fast::exp(s - m), l = sum p,
// o = sum p * v (fp32). Chunks with no valid key for query t write m = -inf, l = 0, o = 0.
template <bool Q8, uint K>
kernel void attn128_partial(device const ushort* qr [[buffer(0)]], device const uchar* Kc [[buffer(1)]],
                            device const uchar* Vc [[buffer(2)]], device const half2* Ksb [[buffer(3)]],
                            device const half2* Vsb [[buffer(4)]], device float4* po [[buffer(5)]],
                            device float2* pml [[buffer(6)]], device const SdCtrl* ctrl [[buffer(7)]],
                            constant uint& maxctx [[buffer(8)]], constant uint& grp [[buffer(9)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint HD = AH_HD, NG = AH_NG, PD = AH_PAD;
    constexpr uint TB = K < 2 ? K : 2;                      // queries per V sweep
    threadgroup float tq[K][AH_NG * AH_PAD];                // scaled q, slice-padded
    threadgroup float sc[K][AH_CH];
    threadgroup float redm[K][4];
    threadgroup float red[4][AH_NG * AH_PAD];
    threadgroup float redl[4];
    uint c = tg.x, h = tg.y, nch = ntgs.x, nh = ntgs.y, kvh = h / grp;
    uint n0 = ctrl->pos + 1;                                // keys visible to query 0
    uint nmax = n0 + K - 1;
    uint j0 = c * AH_CH, j1 = min(j0 + AH_CH, nmax);
    uint rs = lane / 4, sub = lane % 4;                     // row slot in the simdgroup, 32-dim slice of the row
    for (uint t = 0; t < K; t++)
        for (uint e = tid; e < HD; e += 128) tq[t][(e / 32) * PD + e % 32] = AH_SCALE * bf(qr[(t * nh + h) * HD + e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs[K], mloc[K];
    for (uint t = 0; t < K; t++) {
        mloc[t] = -INFINITY; qs[t] = 0;
        if (Q8) for (uint e = 0; e < 32; e++) qs[t] += tq[t][sub * PD + e];
    }
    for (uint j = j0 + sg * 8 + rs; j < j1; j += 32) {
        uint row = kvh * maxctx + j;
        float kf[32];
        float2 sb = float2(1.0f, 0.0f);
        if (Q8) {
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
            for (uint e = 0; e < 32; e++) dd = fma(tq[t][sub * PD + e], kf[e], dd);
            if (Q8) dd = fma(sb.x, dd, sb.y * qs[t]);
            dd += simd_shuffle_xor(dd, 1); dd += simd_shuffle_xor(dd, 2);
            if (j < n0 + t) {
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
        for (uint j = j0 + sg * 8 + rs; j < j1; j += 32) {
            uint row = kvh * maxctx + j;
            float vf[32];
            float2 sb = float2(1.0f, 0.0f);
            if (Q8) {
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
                if (t < K && j < n0 + t) {                  // valid key for query t; odd K: no t = K
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
            // sum the 8 row slots (lanes sub, sub + 4, ..., sub + 28)
            for (uint e = 0; e < 32; e++) {
                acc[u][e] += simd_shuffle_xor(acc[u][e], 4); acc[u][e] += simd_shuffle_xor(acc[u][e], 8);
                acc[u][e] += simd_shuffle_xor(acc[u][e], 16);
            }
            l[u] += simd_shuffle_xor(l[u], 4); l[u] += simd_shuffle_xor(l[u], 8); l[u] += simd_shuffle_xor(l[u], 16);
            if (Q8) { ab[u] += simd_shuffle_xor(ab[u], 4); ab[u] += simd_shuffle_xor(ab[u], 8); ab[u] += simd_shuffle_xor(ab[u], 16); }
            if (rs == 0) for (uint e = 0; e < 32; e++) red[sg][sub * PD + e] = Q8 ? acc[u][e] + ab[u] : acc[u][e];
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
#define AHP(Q_, N_, K_) template [[host_name("attn128_partial_k" #K_ "_" #N_)]] kernel void attn128_partial<Q_, K_>( \
    device const ushort*, device const uchar*, device const uchar*, device const half2*, device const half2*, \
    device float4*, device float2*, device const SdCtrl*, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
AHP(false, b16, 1) AHP(false, b16, 2) AHP(false, b16, 3) AHP(false, b16, 4) AHP(false, b16, 5) AHP(false, b16, 6)
AHP(false, b16, 7) AHP(false, b16, 8)
AHP(true, q8, 1) AHP(true, q8, 2) AHP(true, q8, 3) AHP(true, q8, 4) AHP(true, q8, 5) AHP(true, q8, 6) AHP(true, q8, 7)
AHP(true, q8, 8)

// ---- attn128_reduce / attn128_reduce_gate: threadgroup (q head h, token t), 32 threads x float4 (HD = 128).
// Combine the chunks (skipping empty ones, l = 0): M = max m, w = fast::exp(m - M), L = sum w l, o = sum w o
// (chunk order, fp32), ob = bf(o / L) (sdpa output, bf16). Plain: ao = ob. Gated: MLX bf16 sigmoid on the gate
// (e = bf(precise::exp(|g|)), y = bf(1 / bf(1 + e)), s = g < 0 ? y : bf(1 - y)), ao = bf(ob * s).
// Buffers: po [0], pml [1], p [2] (gated only: gate of head h, token t at p + t * p_stride + h * 2 * HD + HD), ao [3]
// (bf16 [T][nh * HD]), nch [4], p_stride [5] (gated only: elements per p row = nh*2*HD + 2*nkv*HD).
template <bool G>
kernel void attn128_reduce(device const float4* po [[buffer(0)]], device const float2* pml [[buffer(1)]],
                           device const ushort* p [[buffer(2)]], device ushort* ao [[buffer(3)]],
                           constant uint& nch [[buffer(4)]], constant uint& p_stride [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint HD = AH_HD;
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
template [[host_name("attn128_reduce")]] kernel void attn128_reduce<false>(device const float4*, device const float2*,
    device const ushort*, device ushort*, constant uint&, constant uint&, uint2, uint2, uint);
template [[host_name("attn128_reduce_gate")]] kernel void attn128_reduce<true>(device const float4*, device const float2*,
    device const ushort*, device ushort*, constant uint&, constant uint&, uint2, uint2, uint);
