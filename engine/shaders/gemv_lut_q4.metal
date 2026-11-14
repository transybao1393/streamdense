// gemv_lut_q4.metal - streamdense research prototype (NOT wired into the engine): lookup-table (T-MAC / LUT-GEMM
// style) 4-bit K-row matmul for the speculative verify pass, plus primitive-throughput probes used to cost it.
// Needs common_bf16.metal first (bf, to_bf). All helpers are lt_ / lq_ prefixed. Driven by `streamdense bench lut-k`
// (crates/streamdense-model/src/lut_bench.rs).

#define LT_UNROLL _Pragma("clang loop unroll(full)")

// ---------------------------------------------------------------------------------------------------------------
// Part 1: primitive throughput probes. Every thread runs N iterations of 8 independent chains; out[gid] keeps the
// result live. Ops/s = threads * N * (ops per iteration) / time.
// ---------------------------------------------------------------------------------------------------------------
// 8 fp32 fma chains
kernel void lt_p_f32(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    float a[8];
    LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = float(gid + j) * 1e-6f;
    for (uint i = 0; i < N; i++) LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = fma(a[j], m, 0.5f);
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += a[j];
    out[gid] = s;
}
// 8 fp16 fma chains
kernel void lt_p_f16(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    half a[8]; half hm = half(m);
    LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = half(float(gid % 64 + j) * 1e-3f);
    for (uint i = 0; i < N; i++) LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = fma(a[j], hm, half(0.5h));
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += float(a[j]);
    out[gid] = s;
}
// 8 half2 fma chains (16 fp16 fmas per iteration if packed)
kernel void lt_p_h2(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    half2 a[8]; half2 hm = half2(half(m));
    LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = half2(half(float(gid % 64 + j) * 1e-3f), half(float(j) * 1e-3f));
    for (uint i = 0; i < N; i++) LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = fma(a[j], hm, half2(0.5h));
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += float(a[j].x) + float(a[j].y);
    out[gid] = s;
}
// 8 fp32 + 8 fp16 fma chains (16 fmas per iteration): same time as lt_p_f32 => fp16 / fp32 co-issue
kernel void lt_p_mix(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    float a[8]; half b[8]; half hm = half(m);
    LT_UNROLL for (uint j = 0; j < 8; j++) { a[j] = float(gid + j) * 1e-6f; b[j] = half(float(gid % 64 + j) * 1e-3f); }
    for (uint i = 0; i < N; i++) LT_UNROLL for (uint j = 0; j < 8; j++) { a[j] = fma(a[j], m, 0.5f); b[j] = fma(b[j], hm, half(0.5h)); }
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += a[j] + float(b[j]);
    out[gid] = s;
}
// 8 fp32 fma + 8 int (xor-shift style) chains (16 ops per iteration): int / fp32 co-issue
kernel void lt_p_fint(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    float a[8]; uint b[8];
    LT_UNROLL for (uint j = 0; j < 8; j++) { a[j] = float(gid + j) * 1e-6f; b[j] = gid * 7 + j; }
    for (uint i = 0; i < N; i++) LT_UNROLL for (uint j = 0; j < 8; j++) { a[j] = fma(a[j], m, 0.5f); b[j] = (b[j] + 0x9e37u) ^ i; }
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += a[j] + float(b[j] & 0xff);
    out[gid] = s;
}
// 8 int chains alone (8 x 2 ops per iteration)
kernel void lt_p_int(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint b[8];
    LT_UNROLL for (uint j = 0; j < 8; j++) b[j] = gid * 7 + j;
    for (uint i = 0; i < N; i++) LT_UNROLL for (uint j = 0; j < 8; j++) b[j] = (b[j] + 0x9e37u) ^ i;
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += float(b[j] & 0xff);
    out[gid] = s;
}

// Threadgroup lookups: 8 lookups per iteration, index bytes taken from a per-lane register word that changes once per
// iteration (1 int op), so per lookup ~ extract + address + load + fadd. MODE: 0 = per-lane table, conflict-free
// (entry e of lane l at e * 32 + l: lane l always hits bank l); 1 = all 32 lanes of a simdgroup share one 256-entry
// table, random indices (bank conflicts); 2 = broadcast (index uniform over the simdgroup); 3 = no load (ALU part
// only: acc += float(idx)). Threadgroup memory: 32 KB (8192 floats), 1024 threads per group.
template <uint MODE>
kernel void lt_p_tg(device float* out [[buffer(0)]], constant uint& N [[buffer(1)]], constant float& m [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tab[8192];
    for (uint i = tid; i < 8192; i += 1024) tab[i] = float(i) * m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a[8];
    LT_UNROLL for (uint j = 0; j < 8; j++) a[j] = 0.0f;
    uint w = (MODE == 2) ? (sg * 0x9e3779b9u) : (gid * 0x9e3779b9u + 0x7f4a7c15u);
    // MODE 1: 32 tables of 256 (one per simdgroup of the 32 in the group)
    threadgroup const float* tb = tab + (MODE == 0 ? lane : (sg % 32) * 256);
    for (uint i = 0; i < N; i++) {
        LT_UNROLL for (uint j = 0; j < 8; j++) {
            uint idx = (w >> (j * 3)) & 0xffu;
            if (MODE == 0) a[j] += tb[idx * 32];
            else if (MODE == 3) a[j] += float(idx);
            else a[j] += tb[idx];
        }
        w = w * 1664525u + 1013904223u;
    }
    float s = 0; LT_UNROLL for (uint j = 0; j < 8; j++) s += a[j];
    out[gid] = s;
}
template [[host_name("lt_p_tg0")]] kernel void lt_p_tg<0>(device float*, constant uint&, constant float&, uint, uint, uint, uint);
template [[host_name("lt_p_tg1")]] kernel void lt_p_tg<1>(device float*, constant uint&, constant float&, uint, uint, uint, uint);
template [[host_name("lt_p_tg2")]] kernel void lt_p_tg<2>(device float*, constant uint&, constant float&, uint, uint, uint, uint);
template [[host_name("lt_p_tg3")]] kernel void lt_p_tg<3>(device float*, constant uint&, constant float&, uint, uint, uint, uint);

// ---------------------------------------------------------------------------------------------------------------
// Part 2: lq_gemv - T-MAC style bit-plane LUT 4-bit GEMV for KT = 1..8 token rows (MLX layout, no repack).
// Layout as q4b_gemv: W u32 [R][C/8] (low nibble first), scales / biases bf16 [R][C/64], w = s*q + b, x [KT][C] bf16,
// outf [KT][R] f32. C % 64 == 0. One thread per output row, LQ_TG rows per threadgroup.
// Per 64-column group g, per token t, the threadgroup builds 16 tables (one per 4-column block j) of 16 entries:
//   L[t][j][p] = ((p&1 ? x0 : 0) + (p&2 ? x1 : 0)) + ((p&4 ? x2 : 0) + (p&8 ? x3 : 0))   (x_i = x[t][64g + 4j + i])
// Row r: for block j, q_i = nibble i of the block, plane b index p_b = sum_i ((q_i >> b) & 1) << i, so
//   sum_i q_i x_i = sum_b 2^b L[p_b]   (bit-serial decomposition; 4 lookups per 4 weights per token)
//   a_b[t] += L[t][j][p_b] over j = 0..15 in order; P = ((a0 + 2 a1) + 4 a2) + 8 a3; xs = L[t][j][15] summed over j
//   acc[t] = acc[t] + fma(sc, P, xs * bi) over groups in order.
// Every per-(row, token) value depends only on x[t] and the row: K-invariant by construction.
// Lookups: all lanes of a simdgroup index the same 16-entry table (bank conflicts / broadcast merges) - by design of
// the probe: this is the layout that shares one table over many rows.
// ---------------------------------------------------------------------------------------------------------------
constant constexpr uint LQ_TG = 256;

template <uint KT>
kernel void lq_gemv(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                    device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                    device float* outf [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float L[2][KT * 256];   // double-buffered tables: [t][j][p]
    const uint row = min(tg * LQ_TG + tid, R - 1), G = C / 64, C8 = C / 8;
    device const uint* wr = W + row * C8;
    device const ushort* sr = S + row * G;
    device const ushort* br = B + row * G;
    float acc[KT];
    LT_UNROLL for (uint t = 0; t < KT; t++) acc[t] = 0.0f;
    // table build for group g into buffer g & 1: thread tid builds entry (j = tid / 16, p = tid % 16) for every token
    auto build = [&](uint g) {
        const uint j = tid / 16, p = tid % 16;
        LT_UNROLL for (uint t = 0; t < KT; t++) {
            device const ushort* xp = x + t * C + 64 * g + 4 * j;
            float x0 = (p & 1) ? bf(xp[0]) : 0.0f, x1 = (p & 2) ? bf(xp[1]) : 0.0f;
            float x2 = (p & 4) ? bf(xp[2]) : 0.0f, x3 = (p & 8) ? bf(xp[3]) : 0.0f;
            L[g & 1][t * 256 + tid] = (x0 + x1) + (x2 + x3);
        }
    };
    build(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < G; g++) {
        if (g + 1 < G) build(g + 1);
        threadgroup const float* Lg = L[g & 1];
        uint4 w0 = ((device const uint4*)(wr + 8 * g))[0], w1 = ((device const uint4*)(wr + 8 * g))[1];
        uint wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float a[KT][4], xs[KT];
        LT_UNROLL for (uint t = 0; t < KT; t++) { xs[t] = 0.0f; LT_UNROLL for (uint b = 0; b < 4; b++) a[t][b] = 0.0f; }
        LT_UNROLL for (uint j = 0; j < 16; j++) {
            uint h = (wv[j / 2] >> (16 * (j & 1))) & 0xffffu;   // 4 nibbles of block j
            uint pb[4];
            LT_UNROLL for (uint b = 0; b < 4; b++)
                pb[b] = ((h >> b) & 1u) | ((h >> (b + 3)) & 2u) | ((h >> (b + 6)) & 4u) | ((h >> (b + 9)) & 8u);
            LT_UNROLL for (uint t = 0; t < KT; t++) {
                threadgroup const float* Lt = Lg + t * 256 + 16 * j;
                LT_UNROLL for (uint b = 0; b < 4; b++) a[t][b] += Lt[pb[b]];
                xs[t] += Lt[15];
            }
        }
        float sc = bf(sr[g]), bi = bf(br[g]);
        LT_UNROLL for (uint t = 0; t < KT; t++) {
            float P = ((a[t][0] + 2.0f * a[t][1]) + 4.0f * a[t][2]) + 8.0f * a[t][3];
            acc[t] = acc[t] + fma(sc, P, xs[t] * bi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tg * LQ_TG + tid >= R) return;
    LT_UNROLL for (uint t = 0; t < KT; t++) outf[t * R + row] = acc[t];
}
#define LQ_T(KT_) template [[host_name("lq_gemv_k" #KT_)]] kernel void lq_gemv<KT_>(device const uint*, \
    device const ushort*, device const ushort*, device const ushort*, device float*, constant uint&, constant uint&, uint, uint);
LQ_T(1) LQ_T(2) LQ_T(3) LQ_T(4) LQ_T(5) LQ_T(6) LQ_T(7) LQ_T(8)
