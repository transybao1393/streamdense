// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (via engine/shaders/attn_gated.metal: MLX sdpa partition and rounding; steel BaseMMAFrag fragment layout).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// attn_hd128_flash.metal - dense-engine: prefill flash attention at head dim HD = 64, 96, 128, 160 or 256
// (attn{HD}_flash).
// Needs engine/shaders/common_bf16.metal (bf, to_bf, SdCtrl) and attn_hd128.metal (function constants QD_FC_*,
// qd_scale, qd_softcap, qd_key_ok, qd_key_lo) first in the concatenated source.
// Port of engine/shaders/attn_gated.metal's attn_flash_b16 (HD = 256, always gated) to a template over HD, ungated by
// default; optional per-pipeline scale / score soft cap / sliding window (function constants, see attn_hd128.metal).
//
// ---- attn{HD}_flash: causal flash attention for T query rows against the bf16 KV cache (keys qd_key_ok(j, ctrl.pos + t)
// for row t: 0 .. ctrl.pos + t, or the window; attn{HD}_prep_b16 has already written the T new K/V rows at
// ctrl.pos .. ctrl.pos + T - 1).
// Threadgroup = one kv head x one 8-row query tile; simdgroup s = q head kvh * grp + s (GQA: the grp heads share every
// staged K/V block). Per simdgroup: Q (8 x HD, bf16 pairs packed in HD/8 registers, scaled on unpack by the scale in
// f32, as MLX's q * scale) and O (8 x HD, fp32) as HD/8 simdgroup 8x8 fragments; key blocks of AQ_BK keys staged as
// bf16 in threadgroup memory (row stride padded to HD + 8), S = Q K^T (+ soft cap) and O += P V on
// simdgroup_matrix<float>, online softmax (fast::exp, MLX-style) per row. ob = bf(o / l). With a window, key blocks
// before the tile's first visible key are skipped, and a row with no visible key so far keeps m = -inf, l = 0.
// flags bit 0 clear: ao = ob (p / p_stride unused). Set: ao = bf(ob * sigmoid_bf16(gate)) exactly as
// attn128_reduce_gate, gate of head h, row t at p + t * p_stride + h * 2 * HD + HD.
// Not bit-exact vs attn{HD}_sdpa_vec / MLX (different summation order); fp32 throughout.
// Buffers: qr [0] (bf16 [T][nh][HD]), Kc [1], Vc [2] (bf16 [nkv][max_ctx][HD], 16-byte aligned), p [3], ao [4]
// (bf16 [T][nh*HD]), ctrl [5], max_ctx [6], grp [7], p_stride [8], T [9], flags [10].
// Dispatch: groups (ceil(T / 8), nkv) x (32 * grp) threads (grp <= 32).
constant uint AQ_BK = 16;               // keys per staged block
constant uint AQ_GATED = 1u;

template <uint HD>
kernel void attn_flash(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
                       device const ushort* Vc [[buffer(2)]], device const ushort* p [[buffer(3)]],
                       device ushort* ao [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                       constant uint& maxctx [[buffer(6)]], constant uint& grp [[buffer(7)]],
                       constant uint& p_stride [[buffer(8)]], constant uint& T [[buffer(9)]],
                       constant uint& flags [[buffer(10)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint2 ntgs [[threadgroups_per_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    constexpr uint ND = HD / 8;
    constexpr uint AQ_KS = HD + 8;      // padded threadgroup row stride (ushort); 2 * AQ_KS B keeps uint4 alignment
    threadgroup ushort tK[AQ_BK * AQ_KS];
    threadgroup ushort tV[AQ_BK * AQ_KS];
    const float scale = qd_scale(HD);
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
    device const ushort* kbase = Kc + kvh * maxctx * HD;
    device const ushort* vbase = Vc + kvh * maxctx * HD;
    const uint kb0 = qd_key_lo(pos + t0) / AQ_BK * AQ_BK;    // 0 without a window
    for (uint kb = kb0; kb < nkeys; kb += AQ_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < AQ_BK * (HD / 8); i += ntid) {
            uint r = i / (HD / 8), c8 = (i % (HD / 8)) * 8, key = kb + r;
            uint4 kv = 0, vv = 0;
            if (key < nkeys) {                  // keys past the tile's range are zero (never NaN garbage)
                kv = *(device const uint4*)(kbase + key * HD + c8);
                vv = *(device const uint4*)(vbase + key * HD + c8);
            }
            *(threadgroup uint4*)(tK + r * AQ_KS + c8) = kv;
            *(threadgroup uint4*)(tV + r * AQ_KS + c8) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sf[AQ_BK / 8];
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < AQ_BK / 8; j++) {
            sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            threadgroup const ushort* k0 = tK + (8 * j + fn) * AQ_KS + fm;
            _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
                simdgroup_float8x8 qf, kt;                   // K^T block: (dim fm, key fn / fn + 1)
                qf.thread_elements()[0] = scale * as_type<float>(qpk[d] << 16);
                qf.thread_elements()[1] = scale * as_type<float>(qpk[d] & 0xffff0000u);
                kt.thread_elements()[0] = bf(k0[8 * d]);
                kt.thread_elements()[1] = bf(k0[AQ_KS + 8 * d]);
                simdgroup_multiply_accumulate(sf[j], qf, kt, sf[j]);
            }
        }
        // online softmax over this block (row fm lives in lanes {x, x^1, x^8, x^9})
        float mx = -INFINITY;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < AQ_BK / 8; j++) {
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) {
                uint key = kb + 8 * j + fn + i;
                float sv = qd_softcap(sf[j].thread_elements()[i]);
                sv = (rv && !qd_key_ok(key, qlim)) ? -INFINITY : sv;
                sf[j].thread_elements()[i] = sv;
                mx = max(mx, sv);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float mn = max(m, mx);       // finite without a window: key 0 is in the first block of every row
        const bool none = QD_HAS_WINDOW && mn == -INFINITY;  // window: no visible key for this row yet
        float fac = none ? 1.0f : fast::exp(m - mn);
        float rs = 0.0f;
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < AQ_BK / 8; j++) {
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) {
                float pe = none ? 0.0f : fast::exp(sf[j].thread_elements()[i] - mn);
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
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < AQ_BK / 8; j++) {
                simdgroup_float8x8 vf;                       // V block: (key fm, dim fn / fn + 1)
                uint vw = *(threadgroup const uint*)(tV + (8 * j + fm) * AQ_KS + 8 * d + fn);
                vf.thread_elements()[0] = as_type<float>(vw << 16);
                vf.thread_elements()[1] = as_type<float>(vw & 0xffff0000u);
                simdgroup_multiply_accumulate(of[d], sf[j], vf, of[d]);
            }
        }
    }
    if (!rv) return;
    device ushort* out = ao + (trow * nh + h) * HD + fn;
    if (flags & AQ_GATED) {
        device const ushort* gp = p + trow * p_stride + h * 2 * HD + HD + fn;
        _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) {
                float ob = bf(to_bf(of[d].thread_elements()[i] / l));
                float gv = bf(gp[8 * d + i]);
                float e = bf(to_bf(precise::exp(fabs(gv))));
                float yy = bf(to_bf(1.0f / bf(to_bf(1.0f + e))));
                float sgm = gv < 0.0f ? yy : bf(to_bf(1.0f - yy));
                out[8 * d + i] = to_bf(ob * sgm);
            }
        }
    } else {
        _Pragma("clang loop unroll(full)") for (uint d = 0; d < ND; d++) {
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < 2; i++) out[8 * d + i] = to_bf(of[d].thread_elements()[i] / l);
        }
    }
}
#define AQF(HD_) template [[host_name("attn" #HD_ "_flash")]] kernel void attn_flash<HD_>(device const ushort*, \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, device const SdCtrl*, \
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint2, uint, uint, uint);
AQF(128) AQF(64) AQF(256) AQF(96) AQF(160)
