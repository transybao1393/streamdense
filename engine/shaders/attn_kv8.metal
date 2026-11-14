// attn_kv8.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// RoPE + 8-bit g32 KV append (rope_kv) and attention v2 (attn2_partial + attn_reduce), ported from decode7b.swift.
// Only change vs decode7b: addressing generalized for GQA (nh q heads from the grid, nkv / grp kv heads); for
// nkv == nh (Llama-2-7B) every load / store address and every fp operation is the same as decode7b.
#include <metal_stdlib>
using namespace metal;

// ---- RoPE (HF rotate_half, cos/sin in fp32 then fp16) + KV append; threadgroup (head, token), 128 threads.
// Q8: K (after RoPE) and V quantized to 8 bits in groups of 32 along the head dim (one simdgroup = one group),
// fp16 scale = (max - min) / 255 and bias = min per group; q from the fp16-rounded scale / bias.
template <bool Q8>
kernel void rope_kv(device const half* qkv [[buffer(0)]], device half* qr [[buffer(1)]],
                    device half* Kc [[buffer(2)]], device half* Vc [[buffer(3)]],
                    device uchar* K8 [[buffer(4)]], device half2* Ksb [[buffer(5)]],
                    device uchar* V8 [[buffer(6)]], device half2* Vsb [[buffer(7)]],
                    device const float* invf [[buffer(8)]], constant uint& pos0 [[buffer(9)]], constant uint& maxctx [[buffer(10)]], constant uint& nkv [[buffer(11)]],
                    uint2 ntgs [[threadgroups_per_grid]],
                    uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 128;
    uint nh = ntgs.x, QD = nh * HD, KD = nkv * HD;          // GQA: q heads nh, kv heads nkv (llama2: nkv = nh)
    uint h = tg.x, t = tg.y, pos = pos0 + t;
    device const half* src = qkv + t * (QD + 2 * KD);
    float ang = float(pos) * invf[d & 63];
    float c = float(half(precise::cos(ang))), s = float(half(precise::sin(ang)));
    uint base = h * HD;
    float sign = d < 64 ? -1.0f : 1.0f;
    float q = float(src[base + d]), qp = float(src[base + (d ^ 64)]);
    qr[t * QD + base + d] = half(q * c + sign * qp * s);
    if (h >= nkv) return;                                   // uniform per threadgroup
    float k = float(src[QD + base + d]), kp = float(src[QD + base + (d ^ 64)]);
    float v = float(src[QD + KD + base + d]);
    half kh = half(k * c + sign * kp * s), vh = half(v);
    uint slot = (h * maxctx + pos) * HD + d;
    if (!Q8) { Kc[slot] = kh; Vc[slot] = vh; return; }
    uint gslot = (h * maxctx + pos) * 4 + sg;
    for (uint u = 0; u < 2; u++) {
        float x = float(u == 0 ? kh : vh);
        float lo = simd_min(x), hi = simd_max(x);
        half sc16 = half(max(hi - lo, 1e-8f) / 255.0f), lo16 = half(lo);
        float scf = float(sc16) > 0.0f ? float(sc16) : 1.0f;
        uchar qv = uchar(clamp(rint((x - float(lo16)) / scf), 0.0f, 255.0f));
        if (u == 0) { K8[slot] = qv; if (lane == 0) Ksb[gslot] = half2(sc16, lo16); }
        else { V8[slot] = qv; if (lane == 0) Vsb[gslot] = half2(sc16, lo16); }
    }
}
template [[host_name("rope_kv_f16")]] kernel void rope_kv<false>(device const half*, device half*, device half*, device half*,
    device uchar*, device half2*, device uchar*, device half2*, device const float*, constant uint&, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
template [[host_name("rope_kv_q8")]] kernel void rope_kv<true>(device const half*, device half*, device half*, device half*,
    device uchar*, device half2*, device uchar*, device half2*, device const float*, constant uint&, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);

// ---- attention v2: split-K as attn_partial, but 8 lanes per cached row (16 dims per lane,
// one 16-byte load per lane for 8-bit, two for fp16), 4 rows per simdgroup at a time, so 16 rows per threadgroup step;
// row dot products reduced over 8 lanes (3 shuffles) instead of 32 (5); the 8-bit dequant is folded out of the inner
// loops: q.k = sc * (q . kq) + bias * sum(q) (the lane's 16 dims lie in one group of 32), and for V the lane
// accumulates (p * sc) * vq plus one scalar p * bias. The chunk max comes from simd / threadgroup reductions (not a
// per-thread scan of the scores). Chunk length ch (<= 256) is a runtime argument (split-K count).
constant uint CHMAX = 256;
template <bool Q8, uint K>
kernel void attn2_partial(device const half* qr [[buffer(0)]], device const half* Kc [[buffer(1)]], device const half* Vc [[buffer(2)]],
                          device const uchar* K8 [[buffer(3)]], device const half2* Ksb [[buffer(4)]],
                          device const uchar* V8 [[buffer(5)]], device const half2* Vsb [[buffer(6)]],
                          device float4* po [[buffer(7)]], device float2* pml [[buffer(8)]],
                          constant uint& n0 [[buffer(9)]], constant uint& maxctx [[buffer(10)]], constant uint& ch [[buffer(11)]],
                          constant uint& grp [[buffer(12)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint c = tg.x, h = tg.y, nch = ntgs.x, nh = ntgs.y, kvh = h / grp;   // GQA: grp q heads per kv head
    uint nmax = n0 + K - 1;
    uint j0 = c * ch, j1 = min(j0 + ch, nmax);
    uint rs = lane / 8, sub = lane % 8;                     // row slot in the simdgroup, 16-dim slice of the row
    threadgroup float sc[K][CHMAX];
    threadgroup float red[K][4][128];
    threadgroup float redl[K][4], redm[K][4];
    float q[K][16], qs[K], mloc[K];
    for (uint t = 0; t < K; t++) {
        device const half4* qp = (device const half4*)(qr + t * nh * 128 + h * 128 + sub * 16);
        qs[t] = 0; mloc[t] = -INFINITY;
        for (uint u = 0; u < 4; u++) {
            float4 v = float4(qp[u]) * 0.08838834764831845f;
            q[t][4 * u] = v.x; q[t][4 * u + 1] = v.y; q[t][4 * u + 2] = v.z; q[t][4 * u + 3] = v.w;
            qs[t] += v.x + v.y + v.z + v.w;
        }
    }
    for (uint j = j0 + sg * 4 + rs; j < j1; j += 16) {
        uint row = kvh * maxctx + j;
        float kf[16];
        float2 sb = float2(1.0f, 0.0f);
        if (Q8) {
            uint4 w = ((device const uint4*)(K8 + row * 128))[sub];
            sb = float2(Ksb[row * 4 + sub / 2]);
            for (uint u = 0; u < 4; u++) {
                float4 f = float4(as_type<uchar4>(w[u]));
                kf[4 * u] = f.x; kf[4 * u + 1] = f.y; kf[4 * u + 2] = f.z; kf[4 * u + 3] = f.w;
            }
        } else {
            device const half4* kp = (device const half4*)(Kc + row * 128 + sub * 16);
            for (uint u = 0; u < 4; u++) {
                float4 f = float4(kp[u]);
                kf[4 * u] = f.x; kf[4 * u + 1] = f.y; kf[4 * u + 2] = f.z; kf[4 * u + 3] = f.w;
            }
        }
        for (uint t = 0; t < K; t++) {
            float d = 0;
            for (uint e = 0; e < 16; e++) d += q[t][e] * kf[e];
            if (Q8) d = sb.x * d + sb.y * qs[t];
            d += simd_shuffle_xor(d, 1); d += simd_shuffle_xor(d, 2); d += simd_shuffle_xor(d, 4);
            d = j < n0 + t ? d : -INFINITY;
            mloc[t] = max(mloc[t], d);
            if (sub == 0) sc[t][j - j0] = d;
        }
    }
    for (uint t = 0; t < K; t++) { float mm = simd_max(mloc[t]); if (lane == 0) redm[t][sg] = mm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m[K], l[K], ab[K];
    float acc[K][16];
    for (uint t = 0; t < K; t++) {
        m[t] = max(max(redm[t][0], redm[t][1]), max(redm[t][2], redm[t][3])); l[t] = 0; ab[t] = 0;
        for (uint e = 0; e < 16; e++) acc[t][e] = 0;
    }
    for (uint j = j0 + sg * 4 + rs; j < j1; j += 16) {
        uint row = kvh * maxctx + j;
        float vf[16];
        float2 sb = float2(1.0f, 0.0f);
        if (Q8) {
            uint4 w = ((device const uint4*)(V8 + row * 128))[sub];
            sb = float2(Vsb[row * 4 + sub / 2]);
            for (uint u = 0; u < 4; u++) {
                float4 f = float4(as_type<uchar4>(w[u]));
                vf[4 * u] = f.x; vf[4 * u + 1] = f.y; vf[4 * u + 2] = f.z; vf[4 * u + 3] = f.w;
            }
        } else {
            device const half4* vp = (device const half4*)(Vc + row * 128 + sub * 16);
            for (uint u = 0; u < 4; u++) {
                float4 f = float4(vp[u]);
                vf[4 * u] = f.x; vf[4 * u + 1] = f.y; vf[4 * u + 2] = f.z; vf[4 * u + 3] = f.w;
            }
        }
        for (uint t = 0; t < K; t++) {
            float p = m[t] == -INFINITY ? 0.0f : exp(sc[t][j - j0] - m[t]);
            l[t] += p;
            float ps = Q8 ? p * sb.x : p;
            for (uint e = 0; e < 16; e++) acc[t][e] += ps * vf[e];
            if (Q8) ab[t] += p * sb.y;
        }
    }
    for (uint t = 0; t < K; t++) {                          // sum the 4 row slots (lanes sub, sub + 8, + 16, + 24)
        for (uint e = 0; e < 16; e++) { acc[t][e] += simd_shuffle_xor(acc[t][e], 8); acc[t][e] += simd_shuffle_xor(acc[t][e], 16); }
        l[t] += simd_shuffle_xor(l[t], 8); l[t] += simd_shuffle_xor(l[t], 16);
        if (Q8) { ab[t] += simd_shuffle_xor(ab[t], 8); ab[t] += simd_shuffle_xor(ab[t], 16); }
        if (rs == 0) for (uint e = 0; e < 16; e++) red[t][sg][sub * 16 + e] = acc[t][e] + ab[t];
        if (lane == 0) redl[t][sg] = l[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < K) {
        uint t = sg;
        float4 o = 0;
        for (uint s2 = 0; s2 < 4; s2++) o += float4(red[t][s2][4 * lane], red[t][s2][4 * lane + 1], red[t][s2][4 * lane + 2], red[t][s2][4 * lane + 3]);
        uint slot = (t * nh + h) * nch + c;
        po[slot * 32 + lane] = o;
        if (lane == 0) pml[slot] = float2(m[t], redl[t][0] + redl[t][1] + redl[t][2] + redl[t][3]);
    }
}
#define ATT2(Q_, N_, K_) template [[host_name("attn2_partial_" #N_ "_k" #K_)]] kernel void attn2_partial<Q_, K_>(device const half*, \
    device const half*, device const half*, device const uchar*, device const half2*, device const uchar*, device const half2*, \
    device float4*, device float2*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint2, uint, uint);
ATT2(false, f16, 1) ATT2(false, f16, 2) ATT2(false, f16, 3) ATT2(false, f16, 4)
ATT2(true, q8, 1) ATT2(true, q8, 2) ATT2(true, q8, 3) ATT2(true, q8, 4)

// combine the chunks: threadgroup (head, query), 32 threads x float4
kernel void attn_reduce(device const float4* po [[buffer(0)]], device const float2* pml [[buffer(1)]], device half* out [[buffer(2)]],
                        constant uint& nch [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint h = tg.x, t = tg.y;
    uint nh = ntgs.x, base = (t * nh + h) * nch;
    float M = -INFINITY;
    for (uint c = 0; c < nch; c++) M = max(M, pml[base + c].x);
    float L = 0;
    float4 o = 0;
    for (uint c = 0; c < nch; c++) {
        float2 ml = pml[base + c];
        float w = ml.x == -INFINITY ? 0.0f : exp(ml.x - M);
        L += w * ml.y;
        o += w * po[(base + c) * 32 + lane];
    }
    ((device half4*)(out + t * nh * 128 + h * 128))[lane] = half4(o / L);
}
