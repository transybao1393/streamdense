// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (rms_norm, rope, sdpa_vector / sdpa_vector_2pass and unary Sigmoid kernels: partition, reduction order, rounding;
// steel BaseMMAFrag fragment layout in attn_flash_b16).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// attn_gated.metal - streamdense Qwen3.8-27B full-attention layers (16 of 64): q/k RMSNorm + partial RoPE + KV append,
// split-K attention over fixed 256-key chunks, and the reduce fused with the sigmoid output gate.
// Needs common_bf16.metal first in the concatenated source (bf, to_bf, SdCtrl).
// Adapted from attn_kv8.metal (rope_kv, attn2_partial, attn_reduce) for head dim 256, bf16 activations, GQA.
//
// Mirrors mlx-lm qwen3_next.Qwen3NextAttention (used by qwen3_5):
//   q_proj out reshaped [L, nh, 2*HD] and split -> queries = cols [0, HD), gate = cols [HD, 2*HD) of each head
//   queries = q_norm(queries), keys = k_norm(k_proj) (mx.fast.rms_norm, bf16 out)
//   queries, keys = nn.RoPE(rdims = 64, traditional = False, base = 1e7)(., offset)   (mx.fast.rope, bf16 out)
//   o = sdpa(q, K, V, scale = HD^-0.5) (bf16 out) ; ao = o * mx.sigmoid(gate) (bf16 ops)
// Rounding points copied from the MLX Metal kernels (mlx/backend/metal/kernels: rms_norm, rope, sdpa_vector,
// unary_ops Sigmoid). MLX builds its kernels with -fno-fast-math, so every kernel body here starts with
// `#pragma METAL fp math_mode(safe)` (scoped to the body, so the other files in the concatenated source keep the
// default fast mode) and uses fast:: exactly where MLX does (fast::cos/sin in rope, fast::exp in sdpa).
// sdpa, two paths (qwen_layers::encode_attn picks per pass from the host position):
//   - attn_sdpa_vec_b16 (bf16 KV, N <= 8192): MLX's sdpa_vector / sdpa_vector_2pass mirrored per query row, bit-exact
//     vs MLX L = 1 decode (measured). K-invariant because each row's arithmetic depends only on its own N.
//   - attn_partial + attn_reduce_gate (q8 KV, or N > 8192): fixed 256-key chunks, each with fp32 partials (m, l, o)
//     and an exact max-then-exp softmax per chunk, combined in chunk order. Not bit-exact vs MLX. The engine runs
//     attn_split_rows instead of attn_partial (same partials bit for bit, cheaper per row; bf16 and q8 KV).
//   q/k RMSNorm uses MLX's rms_single_row partition (4 consecutive dims per thread, see attn_prep).
// Layouts: p bf16 [T][P] with P = nh*2*HD + 2*nkv*HD (q|gate interleaved per head, then k, then v);
// qr bf16 [T][nh][HD]; KV [nkv][max_ctx][HD] (bf16, or u8 + half2 (scale, min) per 32-dim group [nkv][max_ctx][HD/32]);
// po float4 [T][nh][nch][HD/4], pml float2 (m, l) [T][nh][nch]; ao bf16 [T][nh*HD].
// K-invariance: per query row t the arithmetic (order included) does not depend on K or nch: masked keys and
// empty chunks are skipped, not added as zeros, so even signed zeros match a K = 1 dispatch.

constant uint AG_HD = 256;        // head dim (QwenConfig::validate checks 256); the lane mapping below assumes it
constant uint AG_CH = 256;        // fixed split-K chunk (plan section 4)
constant uint AG_NG = AG_HD / 32; // 32-dim groups per head (q8 KV) = row slices per row (8 lanes x 32 dims)
constant uint AG_PAD = 33;        // padded slice stride in threadgroup memory (no bank conflicts across the 8 slices)
constant float AG_SCALE = 0.0625f; // HD^-0.5 for HD = 256 (exact)

// ---- RoPE inverse frequencies, as mx.fast.rope computes them: d = i / (rdims/2), invf = exp2(-d * log2(base)).
// One thread per frequency (rdims/2 = 32 threads). log2base = log2(rope_theta) in f32 (MLX: std::log2 on the
// float base, host side).
kernel void attn_rope_invf(device float* invf [[buffer(0)]], constant float& log2base [[buffer(1)]],
                           uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
#pragma METAL fp math_mode(safe)
    float d = float(i) / float(n);
    invf[i] = exp2(-d * log2base);
}

// ---- attn_prep: threadgroup (head hh, token t), HD = 256 threads (thread d = one dim).
// hh < nh: q head -> RMSNorm(q_norm) -> RoPE -> qr. hh >= nh: kv head hh - nh -> RMSNorm(k_norm) -> RoPE -> K slot,
// V copied, both at pos = ctrl.pos + t. Dispatch (nh + nkv, T) groups x 256.
// Buffers: p [0], q_norm [1], k_norm [2], qr [3], Kc [4], Vc [5] (bf16 as ushort, or u8 when Q8), Ksb [6], Vsb [7]
// (Q8 only), invf [8], ctrl [9], max_ctx [10], eps [11], nkv [12], rdims [13].
// RMSNorm (mx.fast.rms_norm): r = precise::rsqrt(sum(x^2)/HD + eps); n = bf(bf(w) * bf(x * r)).
// RoPE (mx.fast.rope, non-traditional, first rdims dims, pairs (i, i + rdims/2)):
//   theta = float(pos) * invf[i]; c = fast::cos(theta); s = fast::sin(theta)
//   out[i] = bf(x1 * c - x2 * s); out[i + rdims/2] = bf(x1 * s + x2 * c) with x1 = n[i], x2 = n[i + rdims/2];
//   dims >= rdims pass through (= n).
// Q8: K (after RoPE) and V quantized per 32-dim group (one simdgroup) as attn_kv8.rope_kv:
//   fp16 scale = (max - min) / 255, fp16 bias = min; q = clamp(rint((x - bias) / scale), 0, 255).
template <bool Q8>
kernel void attn_prep(device const ushort* p [[buffer(0)]], device const ushort* qnw [[buffer(1)]],
                      device const ushort* knw [[buffer(2)]], device ushort* qr [[buffer(3)]],
                      device uchar* Kc [[buffer(4)]], device uchar* Vc [[buffer(5)]],
                      device half2* Ksb [[buffer(6)]], device half2* Vsb [[buffer(7)]],
                      device const float* invf [[buffer(8)]], device const SdCtrl* ctrl [[buffer(9)]],
                      constant uint& maxctx [[buffer(10)]], constant float& eps [[buffer(11)]],
                      constant uint& nkv [[buffer(12)]], constant uint& rdims [[buffer(13)]],
                      uint2 ntgs [[threadgroups_per_grid]], uint2 tg [[threadgroup_position_in_grid]],
                      uint d [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint HD = AG_HD;
    threadgroup float part[AG_HD / 32];
    threadgroup float tn[AG_HD];
    uint nh = ntgs.x - nkv, hh = tg.x, t = tg.y;
    uint pos = ctrl->pos + t;
    uint P = nh * 2 * HD + 2 * nkv * HD;
    device const ushort* row = p + t * P;
    bool isq = hh < nh;                                     // uniform per threadgroup
    uint kv = hh - nh;
    float x = bf(isq ? row[hh * 2 * HD + d] : row[nh * 2 * HD + kv * HD + d]);
    float w = bf(isq ? qnw[d] : knw[d]);
    // sum of squares as MLX rms_single_row (N_READS = 4): thread d < HD / 4 sums dims 4d .. 4d + 3
    device const ushort* src = isq ? row + hh * 2 * HD : row + nh * 2 * HD + kv * HD;
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
    float nv = bf(to_bf(w * bf(to_bf(x * r))));
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
    ushort vb = row[nh * 2 * HD + nkv * HD + kv * HD + d];
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
template [[host_name("attn_prep_b16")]] kernel void attn_prep<false>(device const ushort*, device const ushort*,
    device const ushort*, device ushort*, device uchar*, device uchar*, device half2*, device half2*, device const float*,
    device const SdCtrl*, constant uint&, constant float&, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
template [[host_name("attn_prep_q8")]] kernel void attn_prep<true>(device const ushort*, device const ushort*,
    device const ushort*, device ushort*, device uchar*, device uchar*, device half2*, device half2*, device const float*,
    device const SdCtrl*, constant uint&, constant float&, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);

// ---- attn_partial: threadgroup (chunk c, q head h), 128 threads = 4 simdgroups; 8 lanes per cached row (32 dims
// per lane = one q8 group), 4 rows per simdgroup at a time -> 16 rows per step over the chunk's 256 keys.
// Query t (position pos + t, pos = ctrl.pos) sees keys 0 .. pos + t. Keys of chunk c: [256c, min(256c + 256, pos + K)).
// Buffers: qr [0], Kc [1], Vc [2] (bf16 as ushort, or u8), Ksb [3], Vsb [4] (Q8), po [5], pml [6], ctrl [7],
// max_ctx [8], grp [9] (q heads per kv head). Dispatch (nch, nh) groups x 128; nch >= ceil((pos + K) / 256).
// Scores: s = sum_e (scale * q_e) * k_e in fp32 (fma chain over the lane's 32 dims, then a 3-step xor butterfly
// over the 8 lanes); Q8: s = sc * (q . kq) + bias * sum(q) (the lane's 32 dims are one group).
// Chunk softmax: m = max over valid keys, p = fast::exp(s - m), l = sum p, o = sum p * v (fp32).
// Chunks with no valid key for query t write m = -inf, l = 0, o = 0.
// The V pass handles 2 queries per sweep (registers); each query's arithmetic is the same for every K.
template <bool Q8, uint K>
kernel void attn_partial(device const ushort* qr [[buffer(0)]], device const uchar* Kc [[buffer(1)]],
                         device const uchar* Vc [[buffer(2)]], device const half2* Ksb [[buffer(3)]],
                         device const half2* Vsb [[buffer(4)]], device float4* po [[buffer(5)]],
                         device float2* pml [[buffer(6)]], device const SdCtrl* ctrl [[buffer(7)]],
                         constant uint& maxctx [[buffer(8)]], constant uint& grp [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint HD = AG_HD, NG = AG_NG, PD = AG_PAD;
    constexpr uint TB = K < 2 ? K : 2;                      // queries per V sweep
    threadgroup float tq[K][AG_NG * AG_PAD];                // scaled q, slice-padded
    threadgroup float sc[K][AG_CH];
    threadgroup float redm[K][4];
    threadgroup float red[4][AG_NG * AG_PAD];
    threadgroup float redl[4];
    uint c = tg.x, h = tg.y, nch = ntgs.x, nh = ntgs.y, kvh = h / grp;
    uint n0 = ctrl->pos + 1;                                // keys visible to query 0
    uint nmax = n0 + K - 1;
    uint j0 = c * AG_CH, j1 = min(j0 + AG_CH, nmax);
    uint rs = lane / 8, sub = lane % 8;                     // row slot in the simdgroup, 32-dim slice of the row
    for (uint t = 0; t < K; t++)
        for (uint e = tid; e < HD; e += 128) tq[t][(e / 32) * PD + e % 32] = AG_SCALE * bf(qr[(t * nh + h) * HD + e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs[K], mloc[K];
    for (uint t = 0; t < K; t++) {
        mloc[t] = -INFINITY; qs[t] = 0;
        if (Q8) for (uint e = 0; e < 32; e++) qs[t] += tq[t][sub * PD + e];
    }
    for (uint j = j0 + sg * 4 + rs; j < j1; j += 16) {
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
            dd += simd_shuffle_xor(dd, 1); dd += simd_shuffle_xor(dd, 2); dd += simd_shuffle_xor(dd, 4);
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
        for (uint j = j0 + sg * 4 + rs; j < j1; j += 16) {
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
                if (t < K && j < n0 + t) {                  // valid key for query t (implies m[t] finite); odd K: no t = K
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
            if (t >= K) break;                              // odd K: the last sweep has one query (t uniform per group)
            // sum the 4 row slots (lanes sub, sub + 8, + 16, + 24)
            for (uint e = 0; e < 32; e++) { acc[u][e] += simd_shuffle_xor(acc[u][e], 8); acc[u][e] += simd_shuffle_xor(acc[u][e], 16); }
            l[u] += simd_shuffle_xor(l[u], 8); l[u] += simd_shuffle_xor(l[u], 16);
            if (Q8) { ab[u] += simd_shuffle_xor(ab[u], 8); ab[u] += simd_shuffle_xor(ab[u], 16); }
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
#define AGP(Q_, N_, K_) template [[host_name("attn_partial_k" #K_ "_" #N_)]] kernel void attn_partial<Q_, K_>( \
    device const ushort*, device const uchar*, device const uchar*, device const half2*, device const half2*, \
    device float4*, device float2*, device const SdCtrl*, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
AGP(false, b16, 1) AGP(false, b16, 2) AGP(false, b16, 3) AGP(false, b16, 4) AGP(false, b16, 5) AGP(false, b16, 6) AGP(false, b16, 7)
AGP(false, b16, 8)
AGP(true, q8, 1) AGP(true, q8, 2) AGP(true, q8, 3) AGP(true, q8, 4) AGP(true, q8, 5) AGP(true, q8, 6) AGP(true, q8, 7)
AGP(true, q8, 8)

#define AG_UNROLL _Pragma("clang loop unroll(full)")
// 4 u8 codes (one uint, little endian) -> exact floats via half: (0x6400 | b) is the half 1024 + b, minus 1024 = b.
static inline void ag_u8x4(uint w, thread float* f) {
    half2 lo = as_type<half2>((w & 0x00ff00ffu) | 0x64006400u) - half2(1024.0h);
    half2 hi = as_type<half2>(((w >> 8) & 0x00ff00ffu) | 0x64006400u) - half2(1024.0h);
    f[0] = lo.x; f[1] = hi.x; f[2] = lo.y; f[3] = hi.y;
}
// ---- attn_split_rows: the attn_partial of a K-row pass, cheaper per row. BIT-IDENTICAL per row to
// attn_partial<Q8, K'> for every K' (so to the one-row pass): the same scores, max, p, chunk sums, written to po / pml
// in the same layout; attn_reduce_gate is unchanged. (qwen_layers::split_k_kernel picks it; SD_ATTN_SPLIT=old keeps
// attn_partial.)
// Why attn_partial costs ~K x per row (27B shapes, 32K keys: K = 8 ~7x K = 1): every score reads the row's scaled q
// from threadgroup memory (one 4-byte load per FMA; threadgroup bandwidth, not the FMAs, bounds the score loop), each
// V sweep (2 rows) re-reads and re-converts the values, and q + scores + 4 x 1 KB reduce buffers per threadgroup (21 KB
// at K = 8) leave one 4-simdgroup threadgroup per core.
// Here (256 threads = 8 simdgroups per (chunk c, q head h), dispatch (nch, nh) groups x 256; with G > 1 per (chunk c,
// heads h0 .. h0 + G - 1 of one kv head), dispatch (nch, nh / G), the K x G (row t, head h0 + g) pairs as rows v = G t + g):
// - scores: lane octet (rs = lane / 8) x sub (lane % 8, dims 32 sub .. 32 sub + 31). An octet holds the 32 q dims of
//   ONE row (t = RP * pass + rs % RP) in registers and walks keys with key slot rs / RP (KO slots); octets of
//   different rows read the same key bytes in the same step (one coalesced load). Score = the same chain as
//   attn_partial: dd = fma(q_e, k_e, dd) for e = 0..31 from 0, then the xor butterfly 1, 2, 4. Rows > RP: NP passes
//   of 8 / NP simdgroups. Keys are converted with float(bfloat) (exact, same value as the bit shift).
// - m = max over the row's valid keys (order-free), p = fast::exp(s - m) once per (row, key) into the score buffer.
// - o, l: attn_partial's 16 key streams (sgo = 0..3, rs = 0..3: keys j0 + 4 sgo + rs + 16 i, in order) with 16 dims
//   per lane: simdgroup sg runs streams (sgo = sg / 2, rs = 2 (sg % 2) + rl), lane = 16 rl + 2 sub + hf holds dims
//   32 sub + 16 hf + e, TB rows per sweep, l += p; acc_e = fma(p, v_e, acc_e) as there. Combine in attn_partial's
//   order: Q = r_{2a} + r_{2a+1} (xor 16 here, xor 8 there), S_sgo = Q_{rs 0,1} + Q_{rs 2,3}, o = (((0 + S_0) + S_1) +
//   S_2) + S_3, l likewise. With TB >= 4 the 4 KB pair-sum buffer reuses the finished sweep's p rows.
// Q8 (u8 KV + half2 (scale, min) per 32-dim group): the lane's 32 key dims are one group, score = attn_partial's
//   dd = fma(sc, dd, min * qs) with qs = q_0 + ... + q_31 (in order, from 0) before the butterfly. V: acc_e =
//   fma(p * sc, v_e, acc_e), ab = fma(p, min, ab) per lane (both 16-dim halves of a group hold the same ab), combined as
//   attn_partial: o_e = (Q_e + Q'_e) + (ab_Q + ab_Q') per key-stream quad, then the same chunk-order sum. Codes go
//   to float with ag_u8x4 (half bit trick, exact, the same values as float(uchar)): the u8 -> float conversions, not
//   the bytes, bounded the q8 kernel (it was slower than bf16 from K = 3 up); the score converts 4 codes at a time
//   between its FMAs (same chain order).
// Masked keys (j >= pos + 1 + t) and empty chunks are skipped as in attn_partial, so K-invariance holds.
template <bool Q8, uint K, uint G, uint RP, uint KO, uint TB>
kernel void attn_split_rows(device const ushort* qr [[buffer(0)]], device const uchar* Kc [[buffer(1)]],
                            device const uchar* Vc [[buffer(2)]], device const half2* Ksb [[buffer(3)]],
                            device const half2* Vsb [[buffer(4)]], device float4* po [[buffer(5)]],
                            device float2* pml [[buffer(6)]], device const SdCtrl* ctrl [[buffer(7)]],
                            constant uint& maxctx [[buffer(8)]], constant uint& grp [[buffer(9)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint HD = AG_HD;
    constexpr uint V = K * G;                               // rows of this threadgroup: (row t, head h0 + g), v = G t + g
    // G = 1 keeps the plain index expressions: the same code as before G (a hoisted min(v, V - 1) cost q8 K = 4 8%)
    constexpr bool GQ = G > 1;
    static_assert(RP * KO <= 4 && 8 % ((V + RP - 1) / RP) == 0, "attn_split_rows geometry");
    constexpr uint NP = (V + RP - 1) / RP;                  // score passes (row sets), SGP simdgroups each
    constexpr uint SGP = 8 / NP;
    constexpr uint ST = SGP * KO;                           // key stride of a lane octet
    constexpr bool AL = TB >= 4;                            // pair sums go to the finished sweep's p rows
    constexpr uint KP = AL ? (V + TB - 1) / TB * TB : V;    // p rows (padded so the last sweep has TB rows)
    threadgroup float sc[KP][AG_CH];                        // scores, then p = exp(s - m) of the valid keys
    threadgroup float redm[8][V];                           // per simdgroup row maxima
    threadgroup float prx[AL ? 1 : 4 * AG_HD];
    threadgroup float pl[8];
    threadgroup float pab[Q8 ? 4 * AG_NG : 1];             // Q8: ab of the odd simdgroup per (stream quad, group)
    uint c = tg.x, h0 = GQ ? tg.y * G : tg.y, nch = ntgs.x, nh = GQ ? ntgs.y * G : ntgs.y, kvh = h0 / grp;
    uint n0 = ctrl->pos + 1;                                // keys visible to row 0
    uint nmax = n0 + K - 1;
    uint j0 = c * AG_CH, j1 = min(j0 + AG_CH, nmax);
    {
        uint rs = lane / 8, sub = lane % 8;
        uint v = (sg / SGP) * RP + rs % RP, ko = rs / RP, t = GQ ? v / G : v;
        bool act = rs < RP * KO && v < V;
        float q[32];
        float mloc = -INFINITY;
        AG_UNROLL for (uint e = 0; e < 32; e++)
            q[e] = AG_SCALE * bf(qr[(GQ ? (min(v, V - 1) / G) * nh + h0 + min(v, V - 1) % G : min(v, V - 1) * nh + h0) * HD + sub * 32 + e]);
        float qs = 0;
        if (Q8) AG_UNROLL for (uint e = 0; e < 32; e++) qs += q[e];
        uint jf = j0 + (sg % SGP) * KO + ko;
        if (Q8) {
            device const uint4* kb = (device const uint4*)(Kc + (kvh * maxctx) * HD + sub * 32);
            device const half2* sbp = Ksb + (kvh * maxctx) * AG_NG + sub;
            uint4 kn[2];
            half2 sbn = half2(0);
            if (jf < j1) { kn[0] = kb[jf * (HD / 16)]; kn[1] = kb[jf * (HD / 16) + 1]; sbn = sbp[jf * AG_NG]; }
            for (uint j = jf; j < j1; j += ST) {
                uint4 kc[2] = {kn[0], kn[1]};
                float2 sb = float2(sbn);
                if (j + ST < j1) { kn[0] = kb[(j + ST) * (HD / 16)]; kn[1] = kb[(j + ST) * (HD / 16) + 1]; sbn = sbp[(j + ST) * AG_NG]; }
                float dd = 0;
                AG_UNROLL for (uint g = 0; g < 8; g++) {           // 4 codes at a time, the chain in dim order
                    float f4[4];
                    ag_u8x4(kc[g / 4][g % 4], f4);
                    AG_UNROLL for (uint e = 0; e < 4; e++) dd = fma(q[4 * g + e], f4[e], dd);
                }
                dd = fma(sb.x, dd, sb.y * qs);
                dd += simd_shuffle_xor(dd, 1); dd += simd_shuffle_xor(dd, 2); dd += simd_shuffle_xor(dd, 4);
                if (act && j < n0 + t) {
                    mloc = max(mloc, dd);
                    if (sub == 0) sc[v][j - j0] = dd;
                }
            }
        } else {
        device const uint4* kb = (device const uint4*)(Kc + ((kvh * maxctx) * HD + sub * 32) * 2);
        float kf[32];
        uint4 kn[4];
        if (jf < j1) AG_UNROLL for (uint i = 0; i < 4; i++) kn[i] = kb[jf * (HD / 8) + i];
        for (uint j = jf; j < j1; j += ST) {
            AG_UNROLL for (uint i = 0; i < 4; i++) {
                float4 a = float4(as_type<bfloat4>(kn[i].xy)), b = float4(as_type<bfloat4>(kn[i].zw));
                kf[8 * i] = a.x; kf[8 * i + 1] = a.y; kf[8 * i + 2] = a.z; kf[8 * i + 3] = a.w;
                kf[8 * i + 4] = b.x; kf[8 * i + 5] = b.y; kf[8 * i + 6] = b.z; kf[8 * i + 7] = b.w;
            }
            if (j + ST < j1) AG_UNROLL for (uint i = 0; i < 4; i++) kn[i] = kb[(j + ST) * (HD / 8) + i];
            float dd = 0;
            AG_UNROLL for (uint e = 0; e < 32; e++) dd = fma(q[e], kf[e], dd);
            dd += simd_shuffle_xor(dd, 1); dd += simd_shuffle_xor(dd, 2); dd += simd_shuffle_xor(dd, 4);
            if (act && j < n0 + t) {
                mloc = max(mloc, dd);
                if (sub == 0) sc[v][j - j0] = dd;
            }
        }
        }
        // octets of the same row in this simdgroup: rs, rs + RP, ... (key slots)
        AG_UNROLL for (uint o = RP; o < RP * KO; o *= 2) mloc = max(mloc, simd_shuffle_xor(mloc, (ushort)(8 * o)));
        if (sub == 0 && rs < RP && v < V) redm[sg][v] = mloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < V * AG_CH; i += 256) {
        uint v = i / AG_CH, jj = i % AG_CH;
        if (j0 + jj < n0 + (GQ ? v / G : v)) {
            float m = -INFINITY;
            AG_UNROLL for (uint s2 = 0; s2 < SGP; s2++) m = max(m, redm[(v / RP) * SGP + s2][v]);
            sc[v][jj] = fast::exp(sc[v][jj] - m);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rl = lane / 16, sub = (lane / 2) % 8, hf = lane % 2;
    uint jv = j0 + 4 * (sg / 2) + 2 * (sg % 2) + rl;
    uint dof = 32 * sub + 16 * hf;
    device const uint4* vb = (device const uint4*)(Vc + ((kvh * maxctx) * HD + dof) * (Q8 ? 1 : 2));
    device const half2* vsp = Vsb + (kvh * maxctx) * AG_NG + sub;
    constexpr uint VW = Q8 ? HD / 16 : HD / 8;              // uint4 per cached V row
    for (uint tb = 0; tb < V; tb += TB) {
        float acc[TB][16], l[TB], ab[TB];
        AG_UNROLL for (uint u = 0; u < TB; u++) { l[u] = 0; ab[u] = 0; AG_UNROLL for (uint e = 0; e < 16; e++) acc[u][e] = 0; }
        uint4 vn[2];
        half2 sbn = half2(0);
        if (jv < j1) { vn[0] = vb[jv * VW]; if (Q8) sbn = vsp[jv * AG_NG]; else vn[1] = vb[jv * VW + 1]; }
        for (uint j = jv; j < j1; j += 16) {
            float vf[16];
            float2 sb = float2(sbn);
            if (Q8) {
                AG_UNROLL for (uint v = 0; v < 4; v++) ag_u8x4(vn[0][v], &vf[4 * v]);
                if (j + 16 < j1) { vn[0] = vb[(j + 16) * VW]; sbn = vsp[(j + 16) * AG_NG]; }
            } else {
                AG_UNROLL for (uint i = 0; i < 2; i++) {
                    float4 a = float4(as_type<bfloat4>(vn[i].xy)), b = float4(as_type<bfloat4>(vn[i].zw));
                    vf[8 * i] = a.x; vf[8 * i + 1] = a.y; vf[8 * i + 2] = a.z; vf[8 * i + 3] = a.w;
                    vf[8 * i + 4] = b.x; vf[8 * i + 5] = b.y; vf[8 * i + 6] = b.z; vf[8 * i + 7] = b.w;
                }
                if (j + 16 < j1) { vn[0] = vb[(j + 16) * VW]; vn[1] = vb[(j + 16) * VW + 1]; }
            }
            if (j < n0) {                                   // key visible to every row
                AG_UNROLL for (uint u = 0; u < TB; u++) {
                    if (tb + u < V) {
                        float pe = sc[tb + u][j - j0];
                        l[u] += pe;
                        float ps = Q8 ? pe * sb.x : pe;
                        AG_UNROLL for (uint e = 0; e < 16; e++) acc[u][e] = fma(ps, vf[e], acc[u][e]);
                        if (Q8) ab[u] = fma(pe, sb.y, ab[u]);
                    }
                }
            } else {
                AG_UNROLL for (uint u = 0; u < TB; u++) {
                    uint v = tb + u;
                    if (v < V && j < n0 + (GQ ? v / G : v)) {
                        float pe = sc[v][j - j0];
                        l[u] += pe;
                        float ps = Q8 ? pe * sb.x : pe;
                        AG_UNROLL for (uint e = 0; e < 16; e++) acc[u][e] = fma(ps, vf[e], acc[u][e]);
                        if (Q8) ab[u] = fma(pe, sb.y, ab[u]);
                    }
                }
            }
        }
        threadgroup float* pr = AL ? &sc[tb][0] : &prx[0];
        if (AL) threadgroup_barrier(mem_flags::mem_threadgroup);   // every simdgroup done reading these p rows
        AG_UNROLL for (uint u = 0; u < TB; u++) {
            uint v = tb + u;
            if (v >= V) break;                              // uniform
            AG_UNROLL for (uint e = 0; e < 16; e++) acc[u][e] += simd_shuffle_xor(acc[u][e], 16);
            l[u] += simd_shuffle_xor(l[u], 16);
            if (Q8) ab[u] += simd_shuffle_xor(ab[u], 16);
            // S_sgo = Q_{sg = 2 sgo} + Q_{sg = 2 sgo + 1}; pr[sgo] = Q of the odd simdgroup, then S
            if (rl == 0 && sg % 2 == 1) AG_UNROLL for (uint e = 0; e < 16; e++) pr[(sg / 2) * AG_HD + dof + e] = acc[u][e];
            if (Q8 && rl == 0 && hf == 0 && sg % 2 == 1) pab[(sg / 2) * AG_NG + sub] = ab[u];
            if (lane == 0) pl[sg] = l[u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rl == 0 && sg % 2 == 0) {
                // Q8: attn_partial's red = acc + ab after the 4-stream sum, i.e. (Q + Q') + (ab_Q + ab_Q')
                float abs_ = Q8 ? ab[u] + pab[(sg / 2) * AG_NG + sub] : 0.0f;
                AG_UNROLL for (uint e = 0; e < 16; e++) {
                    float s = acc[u][e] + pr[(sg / 2) * AG_HD + dof + e];
                    pr[(sg / 2) * AG_HD + dof + e] = Q8 ? s + abs_ : s;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint slot = (GQ ? (v / G) * nh + h0 + v % G : v * nh + h0) * nch + c;
            if (tid < HD / 4) {
                float4 o = 0;
                AG_UNROLL for (uint s2 = 0; s2 < 4; s2++) o += ((threadgroup float4*)(pr + s2 * AG_HD))[tid];
                po[slot * (HD / 4) + tid] = o;
            }
            if (tid == 0) {
                float m = -INFINITY;
                AG_UNROLL for (uint s2 = 0; s2 < SGP; s2++) m = max(m, redm[(v / RP) * SGP + s2][v]);
                pml[slot] = float2(m, (((pl[0] + pl[1]) + (pl[2] + pl[3])) + (pl[4] + pl[5])) + (pl[6] + pl[7]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
// (K, RP, KO, TB) per K: measured best of the valid geometries at 32K keys (verify_bench --attn-pass)
#define ASRG(Q_, N_, K_, G_, RP_, KO_, TB_) template [[host_name(N_)]] kernel void attn_split_rows<Q_, K_, G_, RP_, KO_, TB_>( \
    device const ushort*, device const uchar*, device const uchar*, device const half2*, device const half2*, \
    device float4*, device float2*, device const SdCtrl*, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
#define ASR(Q_, S_, K_, RP_, KO_, TB_) template [[host_name("attn_split_k" #K_ S_)]] kernel void attn_split_rows<Q_, K_, 1, RP_, KO_, TB_>( \
    device const ushort*, device const uchar*, device const uchar*, device const half2*, device const half2*, \
    device float4*, device float2*, device const SdCtrl*, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
ASR(false, "", 1, 1, 4, 1) ASR(false, "", 2, 2, 2, 2) ASR(false, "", 3, 4, 1, 3) ASR(false, "", 4, 4, 1, 4)
ASR(false, "", 5, 4, 1, 3) ASR(false, "", 6, 4, 1, 3) ASR(false, "", 7, 4, 1, 2) ASR(false, "", 8, 4, 1, 2)
// q8 KV: attn_split_k{K}_q8, bit-identical to attn_partial_k{K}_q8 (the bf16 geometries; re-measured after ag_u8x4: the other
// valid ones, incl. 2 rows per lane octet, were slower)
ASR(true, "_q8", 1, 1, 4, 1) ASR(true, "_q8", 2, 2, 2, 2) ASR(true, "_q8", 3, 4, 1, 3) ASR(true, "_q8", 4, 4, 1, 4)
ASR(true, "_q8", 5, 4, 1, 3) ASR(true, "_q8", 6, 4, 1, 3) ASR(true, "_q8", 7, 4, 1, 2) ASR(true, "_q8", 8, 4, 1, 2)
// GQA (qwen_layers::split_gqa): G q heads of one kv head per threadgroup, dispatch (nch, nh / G); rows v = G t + g share
// each key load. Same per-row arithmetic as above. Measured at the 27B shapes (24 q / 4 kv heads), 32K keys, + reduce:
// K = 1 bf16 G = 3 986 -> 741 µs (G = 6 806, G = 2 831), q8 G = 2 793 -> 612 (G = 3 673, G = 6 624); K = 2 bf16 G = 2
// 1480 -> 1113 (q8 G = 2 and K >= 3 with G > 1: no gain).
ASRG(false, "attn_split_k1_g3", 1, 3, 4, 1, 3) ASRG(true, "attn_split_k1_g2_q8", 1, 2, 2, 2, 2) ASRG(false, "attn_split_k2_g2", 2, 2, 4, 1, 4)

// ---- attn_reduce_gate: threadgroup (q head h, token t), HD threads, one dim each (64 threads x float4 before: 32K keys
// 38 -> 22 µs at K = 1; the chunk loads are issued 4 at a time).
// Combine the chunks (skipping empty ones, l = 0): M = max m, w = fast::exp(m - M), L = sum w l, o = sum w o
// (chunk order, fp32, per dim), then ob = bf(o / L) (sdpa output, bf16), then mx.sigmoid on the bf16 gate as MLX's
// bf16 unary op (each bf16 op rounded): e = bf(precise::exp(|g|)), y = bf(1 / bf(1 + e)), s = g < 0 ? y : bf(1 - y);
// ao = bf(ob * s).
// Buffers: po [0], pml [1], p [2] (gate of head h, token t at p + t * p_stride + h * 2 * HD + HD), ao [3]
// (bf16 [T][nh * HD]), nch [4], p_stride [5] (elements per p row = nh*2*HD + 2*nkv*HD).
kernel void attn_reduce_gate(device const float* po [[buffer(0)]], device const float2* pml [[buffer(1)]],
                             device const ushort* p [[buffer(2)]], device ushort* ao [[buffer(3)]],
                             constant uint& nch [[buffer(4)]], constant uint& p_stride [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint HD = AG_HD;
    uint h = tg.x, t = tg.y, nh = ntgs.x, base = (t * nh + h) * nch;
    float M = -INFINITY;
    for (uint c = 0; c < nch; c++) { float2 ml = pml[base + c]; if (ml.y > 0.0f) M = max(M, ml.x); }
    float L = 0, o = 0;
    uint c = 0;
    for (; c + 4 <= nch; c += 4) {
        float2 ml[4];
        float pv[4];
        AG_UNROLL for (uint i = 0; i < 4; i++) { ml[i] = pml[base + c + i]; pv[i] = po[(base + c + i) * HD + lane]; }
        AG_UNROLL for (uint i = 0; i < 4; i++) {
            if (ml[i].y > 0.0f) {
                float w = fast::exp(ml[i].x - M);
                L = fma(w, ml[i].y, L);
                o = fma(w, pv[i], o);
            }
        }
    }
    for (; c < nch; c++) {
        float2 ml = pml[base + c];
        if (ml.y > 0.0f) {
            float w = fast::exp(ml.x - M);
            L = fma(w, ml.y, L);
            o = fma(w, po[(base + c) * HD + lane], o);
        }
    }
    float r = L == 0.0f ? o : o / L;
    float ob = bf(to_bf(r));
    float gv = bf(p[t * p_stride + h * 2 * HD + HD + lane]);
    float e = bf(to_bf(precise::exp(fabs(gv))));
    float y = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
    float s = gv < 0.0f ? y : bf(to_bf(1.0f - y));
    ao[t * nh * HD + h * HD + lane] = to_bf(ob * s);
}

// ---- attn_sdpa_vec_b16: MLX's decode attention (mlx/backend/metal/kernels/sdpa_vector.h, T = bf16, D = V = 256, L = 1,
// no mask) mirrored exactly per query row, fused with the sigmoid output gate of attn_reduce_gate.
// Per query row t the key count is N = ctrl.pos + t + 1 (MLX L = 1 with the cache holding keys 0..pos + t), so a row's
// arithmetic depends only on N (K-invariance).
// - N < two_pass_min (MLX: 1024): single-pass sdpa_vector. 32 simdgroups (BN) x 32 lanes (BD): simdgroup g takes keys
//   g, g + 32, ...; lane l holds dims 8l..8l+7; online softmax with fast::exp; the 32 partials are combined through
//   threadgroup memory.
// - N >= two_pass_min: sdpa_vector_2pass_1 + sdpa_vector_2pass_2 with `blocks` partials, fused: simdgroup g runs
//   pass-1 blocks b = g, g + 32, ... (keys b, b + blocks, ...; o stored rounded to bf16 as MLX's T partials, sums /
//   maxs fp32), then pass 2 (max, sum_exp, weighted partial sum per simdgroup over its blocks, transpose + simd_sum).
//   blocks = sdpa_blocks(N, gqa) mirrors MLX's choice for this GPU class.
// o / sum_exp is rounded to bf16 (sdpa output), then ao = bf(ob * sigmoid(gate)) with MLX's bf16 unary rounding.
// Rows with N > lim are skipped (written by attn_partial + attn_reduce_gate, dispatched before this).
// Buffers: qr [0], Kc [1], Vc [2] (bf16 [nkv][max_ctx][HD]), p [3] (gate), ao [4], ctrl [5], max_ctx [6], grp [7],
// p_stride [8], lim [9], two_pass_min [10]. Dispatch (nh, T) groups x 1024.
constant uint AG_MAXBB = 4;  // pass-1 blocks per simdgroup (blocks <= 128, i.e. N <= 8192)
static inline int sdpa_blocks(int N, int n_simds) {
    if (N > 1024 && n_simds > 4) return N <= 8192 ? 128 : (N <= 32768 ? 256 : (N <= 65536 ? 512 : 1024));
    return 64;
}
kernel void attn_sdpa_vec_b16(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
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

// ---- attn_flash_b16 / attn_flash_q8 (PF-B, prefill): causal flash attention for T query rows against the KV cache (keys
// 0 .. ctrl.pos + t for row t; attn_prep_b16 / _q8 has already written the T new K/V rows), fused with the sigmoid gate.
// Threadgroup = one kv head x one 8-row query tile; simdgroup s = q head kvh * grp + s (GQA: the grp heads share
// every staged K/V block). Per simdgroup: Q (8 x 256, prescaled by 1/16) and O (8 x 256, fp32) live in registers as
// 32 + 32 simdgroup 8x8 fragments; key blocks of AF_BK keys are staged as bf16 in threadgroup memory (row stride
// padded to 264), S = Q K^T and O += P V on simdgroup_matrix<float>, online softmax (fast::exp, MLX-style) per row.
// Output: ob = bf(o / l), ao = bf(ob * sigmoid_bf16(gate)) exactly as attn_reduce_gate.
// Not bit-exact vs attn_sdpa_vec_b16 (different summation order); fp32 throughout.
// Buffers: qr [0], Kc [1], Vc [2] (bf16 [nkv][max_ctx][HD], or u8 when Q8), p [3] (gate), ao [4], ctrl [5], max_ctx [6],
// grp [7], p_stride [8], T [9], Ksb [10], Vsb [11] (Q8: half2 (scale, min) [nkv][max_ctx][HD / 32]; bf16: unused).
// Dispatch: groups (ceil(T / 8), nkv) x (32 * grp) threads (grp <= 32).
// attn_flash_q8 (8-bit KV, as written by attn_prep_q8): the staging loop dequantizes each 8-dim slice to fp16 in
// threadgroup memory, x = half(scale * q + min) (fp32 fma, then one fp16 rounding: 2^-11 relative, far below the q8
// step (max - min) / 255; q8 scale / min are fp16 already, so the range fits), the rest is the bf16 kernel with fp16
// unpacks. Not bit-exact vs the q8 decode attention (its s = scale * (q . kq) + min * sum(q) chain).
constant uint AF_BK = 16;            // keys per staged block
constant uint AF_KS = AG_HD + 8;     // padded threadgroup row stride (ushort); 528 B keeps uint4 alignment
template <bool Q8>
kernel void attn_flash(device const ushort* qr [[buffer(0)]], device const uchar* Kc [[buffer(1)]],
                       device const uchar* Vc [[buffer(2)]], device const ushort* p [[buffer(3)]],
                       device ushort* ao [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                       constant uint& maxctx [[buffer(6)]], constant uint& grp [[buffer(7)]],
                       constant uint& p_stride [[buffer(8)]], constant uint& T [[buffer(9)]],
                       device const half2* Ksb [[buffer(10)]], device const half2* Vsb [[buffer(11)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr uint HD = AG_HD, ND = AG_HD / 8;
    threadgroup ushort tK[AF_BK * AF_KS];
    threadgroup ushort tV[AF_BK * AF_KS];
    const uint t0 = tg.x * 8, kvh = tg.y, nh = ntgs.y * grp, h = kvh * grp + sg, ntid = 32 * grp;
    const uint pos = ctrl->pos;
    const uint nkeys = pos + min(t0 + 8, T);   // keys visible to the tile's last row
    // simdgroup 8x8 fragment coordinates of this lane (elements (fm, fn) and (fm, fn + 1))
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint trow = t0 + fm;
    const bool rv = trow < T;
    const uint qlim = pos + trow;               // last key index row fm may see

    // Q kept as packed bf16 pairs (32 registers instead of 64); the 1/16 scale is exact, applied on unpack
    uint qpk[ND];
    simdgroup_float8x8 of[ND];
    {
        device const ushort* qp = qr + (trow * nh + h) * HD + fn;
        _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
            qpk[d] = rv ? *(device const uint*)(qp + 8 * d) : 0u;
            of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    float m = -INFINITY, l = 0.0f;
    device const ushort* kbase = (device const ushort*)Kc + kvh * maxctx * HD;
    device const ushort* vbase = (device const ushort*)Vc + kvh * maxctx * HD;
    const uint rbase = kvh * maxctx;            // first KV row of this kv head
    for (uint kb = 0; kb < nkeys; kb += AF_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < AF_BK * (HD / 8); i += ntid) {
            uint r = i / (HD / 8), c8 = (i % (HD / 8)) * 8, key = kb + r;
            uint4 kv = 0, vv = 0;
            if (key < nkeys && Q8) {            // dequantize 8 codes of one 32-dim group to fp16
                uint row = rbase + key;
                uint2 kw = *(device const uint2*)(Kc + row * HD + c8);
                uint2 vw = *(device const uint2*)(Vc + row * HD + c8);
                float2 ks = float2(Ksb[row * AG_NG + c8 / 32]), vs = float2(Vsb[row * AG_NG + c8 / 32]);
                _Pragma("clang loop unroll(full)") for (uint w = 0; w < 2; w++) {
                    float4 kq = float4(as_type<uchar4>(kw[w])), vq = float4(as_type<uchar4>(vw[w]));
                    half4 kh = half4(fma(float4(ks.x), kq, float4(ks.y))), vh = half4(fma(float4(vs.x), vq, float4(vs.y)));
                    kv[2 * w] = as_type<uint>(kh.xy); kv[2 * w + 1] = as_type<uint>(kh.zw);
                    vv[2 * w] = as_type<uint>(vh.xy); vv[2 * w + 1] = as_type<uint>(vh.zw);
                }
            } else if (key < nkeys) {           // keys past the tile's range are zero (never NaN garbage)
                kv = *(device const uint4*)(kbase + key * HD + c8);
                vv = *(device const uint4*)(vbase + key * HD + c8);
            }
            *(threadgroup uint4*)(tK + r * AF_KS + c8) = kv;
            *(threadgroup uint4*)(tV + r * AF_KS + c8) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sf[AF_BK / 8];
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < AF_BK / 8; j++) {
            sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            threadgroup const ushort* k0 = tK + (8 * j + fn) * AF_KS + fm;
            _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
                simdgroup_float8x8 qf, kt;                   // K^T block: (dim fm, key fn / fn + 1)
                qf.thread_elements()[0] = AG_SCALE * as_type<float>(qpk[d] << 16);
                qf.thread_elements()[1] = AG_SCALE * as_type<float>(qpk[d] & 0xffff0000u);
                kt.thread_elements()[0] = Q8 ? float(as_type<half>(k0[8 * d])) : bf(k0[8 * d]);
                kt.thread_elements()[1] = Q8 ? float(as_type<half>(k0[AF_KS + 8 * d])) : bf(k0[AF_KS + 8 * d]);
                simdgroup_multiply_accumulate(sf[j], qf, kt, sf[j]);
            }
        }
        // online softmax over this block (row fm lives in lanes {x, x^1, x^8, x^9})
        float mx = -INFINITY;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < AF_BK / 8; j++) {
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) {
                uint key = kb + 8 * j + fn + i;
                float sv = sf[j].thread_elements()[i];
                sv = (rv && key > qlim) ? -INFINITY : sv;
                sf[j].thread_elements()[i] = sv;
                mx = max(mx, sv);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float mn = max(m, mx);                              // finite: key 0 is in the first block of every row
        float fac = fast::exp(m - mn);
        float rs = 0.0f;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < AF_BK / 8; j++) {
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) {
                float pe = fast::exp(sf[j].thread_elements()[i] - mn);
                sf[j].thread_elements()[i] = pe;
                rs += pe;
            }
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        l = l * fac + rs;
        m = mn;
        _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
            of[d].thread_elements()[0] *= fac;
            of[d].thread_elements()[1] *= fac;
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < AF_BK / 8; j++) {
                simdgroup_float8x8 vf;                       // V block: (key fm, dim fn / fn + 1)
                uint vw = *(threadgroup const uint*)(tV + (8 * j + fm) * AF_KS + 8 * d + fn);
                vf.thread_elements()[0] = Q8 ? float(as_type<half2>(vw).x) : as_type<float>(vw << 16);
                vf.thread_elements()[1] = Q8 ? float(as_type<half2>(vw).y) : as_type<float>(vw & 0xffff0000u);
                simdgroup_multiply_accumulate(of[d], sf[j], vf, of[d]);
            }
        }
    }
    if (!rv) return;
    float inv_l = l;  // o / l below (division, as MLX's o / sum_exp)
    device const ushort* gp = p + trow * p_stride + h * 2 * HD + HD + fn;
    device ushort* out = ao + (trow * nh + h) * HD + fn;
    _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
        _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) {
            float ob = bf(to_bf(of[d].thread_elements()[i] / inv_l));
            float gv = bf(gp[8 * d + i]);
            float e = bf(to_bf(precise::exp(fabs(gv))));
            float yy = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
            float sgm = gv < 0.0f ? yy : bf(to_bf(1.0f - yy));
            out[8 * d + i] = to_bf(ob * sgm);
        }
    }
}
#define AFL(Q_, N_) template [[host_name(N_)]] kernel void attn_flash<Q_>(device const ushort*, device const uchar*, \
    device const uchar*, device const ushort*, device ushort*, device const SdCtrl*, constant uint&, constant uint&, \
    constant uint&, constant uint&, device const half2*, device const half2*, uint2, uint2, uint, uint, uint);
AFL(false, "attn_flash_b16")
AFL(true, "attn_flash_q8")
#undef AFL
