// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (steel BaseMMAFrag fragment coordinate mapping; affine 4-bit layout and swiglu rounding chain).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// gemm_q4.metal - streamdense Qwen3.8-27B prefill GEMM (plan P3 U25): MLX affine 4-bit g64 weights times T bf16
// activation rows with simdgroup_matrix 8x8 tiles. Needs common_bf16.metal first (bf, to_bf); every other helper here
// has a gq_ prefix, so the file also compiles alone after common_bf16 (test libraries).
// Layout as q4b_gemv: W u32 [R][C/8] (8 nibbles per u32, low nibble first); scales / biases bf16 [R][C/64];
// w = s*q + b; x [T][C] and out / res [T][R] bf16 (ushort); outf [T][R] f32. C % 64 == 0, any T >= 1, any R.
//
// Tiling: BN = WN * NF * 8 weight rows x BM = WM * MF * 8 token rows per threadgroup, BK = 32 per step
// (WN = 2, simdgroups WM x WN, 64 * WM threads). Each step the loader dequantizes the BN x 32 weight tile ONCE into
// threadgroup memory as fp32 fma(q, s, b) (exact affine value; MLX's qmm instead stores bf16(w)), transposed (k-major),
// and stages the x tile as fp32; every token row of the tile reuses it. Fragments come from simdgroup_load; the device
// loads of step k+1 are issued into registers before the MMAs of step k (one-deep software pipeline).
// Simdgroup (wm, wn) owns MF x NF fp32 accumulator frags (MF*8 tokens x NF*8 weight rows).
// Math: fp32 simdgroup MMA. The summation order differs from q4b_gemv (and MLX's qmv), so this is NOT bit-exact with
// the stepwise path: prefill with it is held to the teacher bar (plan U25); the exact K<=8 GEMV path stays selectable.
//
// Kernels (host names), {BM}x{BN} in {32x64, 64x64}:
//   q4b_gemm_{BM}x{BN}_m{0,2,3}: W[0] S[1] B[2] x[3] out[4] R[5] C[6] res[7] outf[8] T[9]
//     dispatch groups (ceil(R/BN), ceil(T/BM)) x (64*WM) threads. MODE 0: out = bf(v); 2: res = bf(bf(res) + bf(v))
//     in place (residual); 3: outf = v (f32). Bind every index (as q4b_gemv).
//   q4b_gemm_swiglu_{BM}x{BN}: Wg/Sg/Bg[0-2] Wu/Su/Bu[3-5] x[6] out[7] R[8] C[9] T[10]
//     dispatch groups (ceil(R/(BN/2)), ceil(T/BM)) x (64*WM); the tile holds BN/2 gate rows and the BN/2 up rows of the
//     same outputs, so each thread holds matching gate/up elements; out = q4b_silu_mul rounding chain (MLX swiglu).
//   q4b_gemm_swiglu_g_{BM}x{BN}: as q4b_gemm_swiglu + gout[11] = bf(gate x) (streamed prefill fast mode, ffn_stream.rs).

constant constexpr uint GQ_BK = 32;  // K step (half a quant group)
constant constexpr uint GQ_XP = GQ_BK + 4;  // x tile row pitch (f32)

#define GQ_UNROLL _Pragma("clang loop unroll(full)")

static inline float gq_rbf(float f) { return bf(to_bf(f)); }

// == q4b_silu_mul (gemv_q4_bf16.metal): MLX bf16 swiglu rounding chain.
static inline ushort gq_silu_mul(float gv, float uv) {
    float g = gq_rbf(gv), u = gq_rbf(uv);
    float e = gq_rbf(precise::exp(fabs(g)));
    float d = gq_rbf(1.0f + e);
    float y = gq_rbf(precise::divide(1.0f, d));
    float s = (g < 0.0f) ? y : gq_rbf(1.0f - y);
    return to_bf(gq_rbf(g * s) * u);
}

// Tile row (= Wt column) of B fragment j for simdgroup column wn. Plain: NF*8 consecutive rows per wn. SW: frags
// j < NF/2 are gate rows wn*NF*4 + 8j, frags j >= NF/2 the up rows of the same outputs (+ BN/2).
template <uint NF, bool SW>
static inline uint gq_nrow(uint j, uint wn) {
    constexpr uint H = NF / 2;
    return SW ? (j / H) * (NF * 8) + wn * H * 8 + (j % H) * 8 : wn * NF * 8 + j * 8;
}

// MLX steel BaseMMAFrag::get_coord: lane holds elements [fm][fn], [fm][fn + 1] of an 8x8 fragment.
static inline uint gq_fm(uint lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
static inline uint gq_fn(uint lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

// acc[i][j] = x tile rows . W tile rows^T over all of C. Matrices: W0 (plain, or gate) and W1 (up, SW only).
// n0: first weight row of the tile (plain: BN rows, SW: BN/2 rows of each matrix). t0: first token row.
// Threadgroup tiles: Xs [BM][GQ_XP] (x as f32, exact), Wt [GQ_BK][BN + 4] (dequantized W tile, transposed).
template <uint MF, uint NF, uint WM, bool SW>
static inline void gq_mma(device const uint2* W0, device const ushort* S0, device const ushort* B0,
                          device const uint2* W1, device const ushort* S1, device const ushort* B1,
                          device const ushort* x, uint R, uint C, uint T, uint n0, uint t0,
                          threadgroup float* Xs, threadgroup float* Wt,
                          uint tid, uint sg, thread simdgroup_float8x8 (&acc)[MF][NF]) {
    constexpr uint NT = 64 * WM;              // threads
    constexpr uint BN = 2 * NF * 8;           // weight rows per tile
    constexpr uint WP = BN + 4;               // Wt row pitch (f32)
    constexpr uint WLD = BN * 2 / NT;         // uint2 (16 nibbles) weight loads per thread per step
    constexpr uint XL = MF / 2;               // uint4 (8 bf16) x loads per thread per step: BM * 32 / 8 / NT
    static_assert(WLD >= 1 && WLD * NT == BN * 2, "weight loader split");
    const uint nk = C / GQ_BK, C16 = C / 16, G = C / 64;
    GQ_UNROLL for (uint i = 0; i < MF; i++) GQ_UNROLL for (uint j = 0; j < NF; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    // weight loader: entry e = tid + i * NT -> tile row lr = e / 2, nibbles 16 hw .. 16 hw + 15 of the step
    device const uint2* wp[WLD];
    device const ushort* sp[WLD];
    device const ushort* bp[WLD];
    GQ_UNROLL for (uint i = 0; i < WLD; i++) {
        uint e = tid + i * NT, lr = e / 2, hw = e & 1;
        bool up = SW && lr >= BN / 2;
        uint wrow = min(n0 + (SW ? lr % (BN / 2) : lr), R - 1);
        wp[i] = (up ? W1 : W0) + wrow * C16 + hw;
        sp[i] = (up ? S1 : S0) + wrow * G;
        bp[i] = (up ? B1 : B0) + wrow * G;
    }
    // x loader: entry e = tid + i * NT -> tile row e / 4, 8-column block e % 4 (rows >= T load zeros)
    device const uint4* xp[XL];
    bool xv[XL];
    GQ_UNROLL for (uint i = 0; i < XL; i++) {
        uint e = tid + i * NT, t = t0 + e / 4;
        xv[i] = t < T;
        xp[i] = (device const uint4*)(x + min(t, T - 1) * C + (e % 4) * 8);
    }

    uint2 wreg[WLD];
    ushort sreg[WLD], breg[WLD];
    uint4 xreg[XL];
    GQ_UNROLL for (uint i = 0; i < WLD; i++) { wreg[i] = wp[i][0]; sreg[i] = sp[i][0]; breg[i] = bp[i][0]; }
    GQ_UNROLL for (uint i = 0; i < XL; i++) xreg[i] = xv[i] ? xp[i][0] : uint4(0);

    const uint wm = sg / 2, wn = sg % 2;
    for (uint kt = 0; kt < nk; kt++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        GQ_UNROLL for (uint i = 0; i < WLD; i++) {
            uint e = tid + i * NT, lr = e / 2, hw = e & 1;
            float s = bf(sreg[i]), b = bf(breg[i]);
            threadgroup float* wd = Wt + (hw * 16) * WP + lr;
            GQ_UNROLL for (uint c = 0; c < 2; c++) {
                uint w = wreg[i][c];
                GQ_UNROLL for (uint n = 0; n < 8; n++) wd[(c * 8 + n) * WP] = fma(float((w >> (4 * n)) & 0xf), s, b);
            }
        }
        GQ_UNROLL for (uint i = 0; i < XL; i++) {
            uint e = tid + i * NT;
            threadgroup float4* xd = (threadgroup float4*)(Xs + (e / 4) * GQ_XP + (e % 4) * 8);
            uint4 v = xreg[i];
            xd[0] = float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xffff0000u),
                           as_type<float>(v.y << 16), as_type<float>(v.y & 0xffff0000u));
            xd[1] = float4(as_type<float>(v.z << 16), as_type<float>(v.z & 0xffff0000u),
                           as_type<float>(v.w << 16), as_type<float>(v.w & 0xffff0000u));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kt + 1 < nk) {  // prefetch step kt + 1 (in flight during the MMAs)
            uint k1 = kt + 1;
            GQ_UNROLL for (uint i = 0; i < WLD; i++) { wreg[i] = wp[i][k1 * 2]; sreg[i] = sp[i][k1 / 2]; breg[i] = bp[i][k1 / 2]; }
            GQ_UNROLL for (uint i = 0; i < XL; i++) xreg[i] = xv[i] ? xp[i][k1 * 4] : uint4(0);
        }
        GQ_UNROLL for (uint kk = 0; kk < GQ_BK; kk += 8) {
            simdgroup_float8x8 a[MF], b[NF];
            GQ_UNROLL for (uint i = 0; i < MF; i++) simdgroup_load(a[i], Xs + (wm * MF * 8 + i * 8) * GQ_XP + kk, GQ_XP);
            GQ_UNROLL for (uint j = 0; j < NF; j++) simdgroup_load(b[j], Wt + kk * WP + gq_nrow<NF, SW>(j, wn), WP);
            GQ_UNROLL for (uint i = 0; i < MF; i++)
                GQ_UNROLL for (uint j = 0; j < NF; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
}

template <uint MF, uint NF, uint WM, uint MODE>
kernel void q4b_gemm(device const uint4* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                     device const ushort* B [[buffer(2)]], device const ushort* x [[buffer(3)]],
                     device ushort* out [[buffer(4)]], constant uint& R [[buffer(5)]], constant uint& C [[buffer(6)]],
                     device ushort* res [[buffer(7)]], device float* outf [[buffer(8)]], constant uint& T [[buffer(9)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = WM * MF * 8, BN = 2 * NF * 8;
    threadgroup float Xs[BM * GQ_XP];
    threadgroup float Wt[GQ_BK * (BN + 4)];
    const uint n0 = tg.x * BN, t0 = tg.y * BM;
    simdgroup_float8x8 acc[MF][NF];
    gq_mma<MF, NF, WM, false>((device const uint2*)W, S, B, (device const uint2*)W, S, B, x, R, C, T, n0, t0, Xs, Wt,
                              tid, sg, acc);
    const uint wm = sg / 2, wn = sg % 2, fm = gq_fm(lane), fn = gq_fn(lane);
    GQ_UNROLL for (uint i = 0; i < MF; i++) {
        uint t = t0 + wm * MF * 8 + i * 8 + fm;
        if (t >= T) continue;
        GQ_UNROLL for (uint j = 0; j < NF; j++) {
            thread auto& te = acc[i][j].thread_elements();
            GQ_UNROLL for (uint c = 0; c < 2; c++) {
                uint r = n0 + gq_nrow<NF, false>(j, wn) + fn + c;
                if (r >= R) continue;
                uint o = t * R + r;
                float v = te[c];
                if (MODE == 0) out[o] = to_bf(v);
                else if (MODE == 2) res[o] = to_bf(bf(res[o]) + gq_rbf(v));
                else outf[o] = v;
            }
        }
    }
}

template <uint MF, uint NF, uint WM>
kernel void q4b_gemm_swiglu(device const uint4* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                            device const ushort* Bg [[buffer(2)]], device const uint4* Wu [[buffer(3)]],
                            device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                            device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                            constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& T [[buffer(10)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = WM * MF * 8, BN = 2 * NF * 8, H = NF / 2;
    threadgroup float Xs[BM * GQ_XP];
    threadgroup float Wt[GQ_BK * (BN + 4)];
    const uint n0 = tg.x * (BN / 2), t0 = tg.y * BM;
    simdgroup_float8x8 acc[MF][NF];
    gq_mma<MF, NF, WM, true>((device const uint2*)Wg, Sg, Bg, (device const uint2*)Wu, Su, Bu, x, R, C, T, n0, t0, Xs, Wt,
                             tid, sg, acc);
    const uint wm = sg / 2, wn = sg % 2, fm = gq_fm(lane), fn = gq_fn(lane);
    GQ_UNROLL for (uint i = 0; i < MF; i++) {
        uint t = t0 + wm * MF * 8 + i * 8 + fm;
        if (t >= T) continue;
        GQ_UNROLL for (uint j = 0; j < H; j++) {
            thread auto& tgv = acc[i][j].thread_elements();
            thread auto& tuv = acc[i][j + H].thread_elements();
            GQ_UNROLL for (uint c = 0; c < 2; c++) {
                uint r = n0 + wn * H * 8 + j * 8 + fn + c;
                if (r < R) out[t * R + r] = gq_silu_mul(tgv[c], tuv[c]);
            }
        }
    }
}


// q4b_gemm_swiglu_g_{BM}x{BN} (ffn_stream.rs fast mode, 2026-11-03): q4b_gemm_swiglu + the gate output itself,
// gout[t][r] = bf(gate x) (what a MODE 0 gate GEMM would write, the value gq_silu_mul rounds first), so the streamed
// prefill's prompt-gate score Σ|silu(g)| comes from the gate already computed (no extra gate pass). Buffers as
// q4b_gemm_swiglu + gout[11]; same dispatch. `out` is bit-identical to q4b_gemm_swiglu's.
template <uint MF, uint NF, uint WM>
kernel void q4b_gemm_swiglu_g(device const uint4* Wg [[buffer(0)]], device const ushort* Sg [[buffer(1)]],
                              device const ushort* Bg [[buffer(2)]], device const uint4* Wu [[buffer(3)]],
                              device const ushort* Su [[buffer(4)]], device const ushort* Bu [[buffer(5)]],
                              device const ushort* x [[buffer(6)]], device ushort* out [[buffer(7)]],
                              constant uint& R [[buffer(8)]], constant uint& C [[buffer(9)]], constant uint& T [[buffer(10)]],
                              device ushort* gout [[buffer(11)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = WM * MF * 8, BN = 2 * NF * 8, H = NF / 2;
    threadgroup float Xs[BM * GQ_XP];
    threadgroup float Wt[GQ_BK * (BN + 4)];
    const uint n0 = tg.x * (BN / 2), t0 = tg.y * BM;
    simdgroup_float8x8 acc[MF][NF];
    gq_mma<MF, NF, WM, true>((device const uint2*)Wg, Sg, Bg, (device const uint2*)Wu, Su, Bu, x, R, C, T, n0, t0, Xs, Wt,
                             tid, sg, acc);
    const uint wm = sg / 2, wn = sg % 2, fm = gq_fm(lane), fn = gq_fn(lane);
    GQ_UNROLL for (uint i = 0; i < MF; i++) {
        uint t = t0 + wm * MF * 8 + i * 8 + fm;
        if (t >= T) continue;
        GQ_UNROLL for (uint j = 0; j < H; j++) {
            thread auto& tgv = acc[i][j].thread_elements();
            thread auto& tuv = acc[i][j + H].thread_elements();
            GQ_UNROLL for (uint c = 0; c < 2; c++) {
                uint r = n0 + wn * H * 8 + j * 8 + fn + c;
                if (r < R) { out[t * R + r] = gq_silu_mul(tgv[c], tuv[c]); gout[t * R + r] = to_bf(tgv[c]); }
            }
        }
    }
}

#define Q4B_GEMM(BM_, BN_, MF_, NF_, WM_, M_) template [[host_name("q4b_gemm_" #BM_ "x" #BN_ "_m" #M_)]] kernel void \
    q4b_gemm<MF_, NF_, WM_, M_>(device const uint4*, device const ushort*, device const ushort*, device const ushort*, \
    device ushort*, constant uint&, constant uint&, device ushort*, device float*, constant uint&, uint2, uint, uint, uint);
#define Q4B_GEMM_SW(BM_, BN_, MF_, NF_, WM_) template [[host_name("q4b_gemm_swiglu_" #BM_ "x" #BN_)]] kernel void \
    q4b_gemm_swiglu<MF_, NF_, WM_>(device const uint4*, device const ushort*, device const ushort*, device const uint4*, \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, \
    constant uint&, uint2, uint, uint, uint);
#define Q4B_GEMM_SWG(BM_, BN_, MF_, NF_, WM_) template [[host_name("q4b_gemm_swiglu_g_" #BM_ "x" #BN_)]] kernel void \
    q4b_gemm_swiglu_g<MF_, NF_, WM_>(device const uint4*, device const ushort*, device const ushort*, device const uint4*, \
    device const ushort*, device const ushort*, device const ushort*, device ushort*, constant uint&, constant uint&, \
    constant uint&, device ushort*, uint2, uint, uint, uint);
#define Q4B_GEMM_ALL(BM_, BN_, MF_, NF_, WM_) Q4B_GEMM(BM_, BN_, MF_, NF_, WM_, 0) Q4B_GEMM(BM_, BN_, MF_, NF_, WM_, 2) \
    Q4B_GEMM(BM_, BN_, MF_, NF_, WM_, 3) Q4B_GEMM_SW(BM_, BN_, MF_, NF_, WM_) Q4B_GEMM_SWG(BM_, BN_, MF_, NF_, WM_)
// 32x128 / 64x128 (NF = 8) and a 128-row (WM = 4) tile were measured slower (register pressure); not built.
Q4B_GEMM_ALL(32, 64, 2, 4, 2)
Q4B_GEMM_ALL(64, 64, 4, 4, 2)
