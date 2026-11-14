// qd_misc.metal - dense-engine glue kernels (needs engine/shaders/common_bf16.metal first: bf, to_bf).
//   qd_embed_q4  token ids -> h bf16 rows, dequantized from the MLX 4-bit embedding (tied lm_head source)
//   qd_bias_add  p += bias per row (q/k/v projection bias of Qwen2 / 2.5), bf16
//   qd_next_tok  pipelined decode: previous pass greedy token (ctrl.tok[0]) -> token row, ctrl.pos = pos
//   qd_scale_b16 x *= s in bf16 (embedding multiplier: Gemma sqrt(hidden), Granite embedding_multiplier)
//   qd_residual  h = h + r * m in bf16 (unfused residual: o_proj / down with bias, post-norm or Granite multiplier)
//   qd_glu       out = act(g) * u in bf16 (SiLU or tanh-GELU; FFN with biases or GeGLU)
//   qd_logits_post  f32 logits / divisor (Granite logits_scaling), then cap * tanh(x / cap) (Gemma 2 final soft cap)
//   qd_embed_b16 token ids -> h bf16 rows copied from an unquantized (bf16) embedding
//   qd_gemv_b16_f32  f32 logits = bf16 matrix x bf16 vector (tied lm_head over an unquantized embedding)
//   qd_layernorm mx.fast.layer_norm rows with weight and optional bias (StableLM, StarCoder2, Cohere2)
//   qd_rmsnorm_rows  in-place RMSNorm of strided row segments (OLMo 2 q / k norm over the whole projection)
//   qd_act_b16   x = act(x) in bf16 (non-gated MLP: StarCoder2 c_fc -> gelu)

// ---- qd_embed_q4: W u32 [V][H/8] (8 nibbles per u32, low nibble first), S / B bf16 [V][H/64] (w = s*q + b),
// tok u32 [T], h bf16 [T][H]. One thread per u32 (8 values). Dispatch threads (H/8, T).
// MLX dequantize rounding: fp32 fma(q, s, b), rounded once to bf16 (as engine head_sample.metal embed_q4).
// Buffers: W [0], S [1], B [2], tok [3], h [4], H [5].
kernel void qd_embed_q4(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                        device const ushort* B [[buffer(2)]], device const uint* tok [[buffer(3)]],
                        device ushort* h [[buffer(4)]], constant uint& H [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    uint W8 = H / 8, G = H / 64;
    if (i >= W8) return;
    uint id = tok[t];
    uint q = W[id * W8 + i];
    uint g = i / 8;
    float s = bf(S[id * G + g]), b = bf(B[id * G + g]);
    device ushort* o = h + t * H + i * 8;
    for (uint k = 0; k < 8; k++) o[k] = to_bf(fma(float((q >> (4 * k)) & 0xfu), s, b));
}

// ---- qd_bias_add: p bf16 [T][P] += bias bf16 [P] (MLX: bf16 matmul output + bf16 bias, one rounding).
// Buffers: p [0], bias [1], P [2]. Dispatch threads (P, T).
kernel void qd_bias_add(device ushort* p [[buffer(0)]], device const ushort* bias [[buffer(1)]],
                        constant uint& P [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= P) return;
    uint o = t * P + i;
    p[o] = to_bf(bf(p[o]) + bf(bias[i]));
}

// ---- qd_next_tok: start of a pipelined decode pass (src/engine.rs DecodeMode::Pipelined). The previous pass's
// argmax_b left its greedy token in ctrl.tok[0]; copy it to the pass's token row and set ctrl.pos, i.e. exactly what
// the host writes before a synchronous pass (ctrl.tok / top1 / top2 are written by argmax_b before anything reads
// them, so only pos and the token matter). Lets the CPU encode pass t+1 while pass t runs.
// Buffers: ctrl [0] (SdCtrl, common_bf16.metal), tok [1] (u32 [rows]), pos [2]. Dispatch threads (1, 1).
kernel void qd_next_tok(device SdCtrl* ctrl [[buffer(0)]], device uint* tok [[buffer(1)]],
                        constant uint& pos [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    tok[0] = ctrl->tok[0];
    ctrl->pos = pos;
}

// ---- qd_scale_b16: x bf16 [n] = bf(x * s) in place (s already rounded to bf16 by the host, as MLX multiplies a bf16
// array by a weakly typed scalar). Buffers: x [0], n [1], s [2]. Dispatch threads (n, 1).
kernel void qd_scale_b16(device ushort* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& s [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    x[i] = to_bf(bf(x[i]) * s);
}

// ---- qd_residual: h bf16 [n] = bf(h + bf(r * m)) (m rounded to bf16 by the host; m = 1: bf(h + r), the same rounding
// as the fused gemv / GEMM MODE 2 residual). Buffers: h [0], r [1], n [2], m [3]. Dispatch threads (n, 1).
kernel void qd_residual(device ushort* h [[buffer(0)]], device const ushort* r [[buffer(1)]], constant uint& n [[buffer(2)]],
                        constant float& m [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    h[i] = to_bf(bf(h[i]) + bf(to_bf(bf(r[i]) * m)));
}

// erf as MLX's Metal erf (njuffa's single-precision erff, max error < 1 ulp): fma polynomial in t = |a| / a^2.
static inline float qd_erf(float a) {
    float t = fabs(a), s = a * a, r;
    if (t > 0.927734375f) {
        r = fma(-1.72853470e-5f, t, 3.83197126e-4f);
        float u = fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = fma(r, s, u);
        r = fma(r, t, -1.06777877e-1f);
        r = fma(r, t, -6.34846687e-1f);
        r = fma(r, t, -1.28717512e-1f);
        r = fma(r, t, -t);
        r = copysign(1.0f - precise::exp(r), a);
    } else {
        r = -5.96761703e-4f;
        r = fma(r, s, 4.99119423e-3f);
        r = fma(r, s, -2.67681349e-2f);
        r = fma(r, s, 1.12819925e-1f);
        r = fma(r, s, -3.76125336e-1f);
        r = fma(r, s, 1.28379166e-1f);
        r = fma(r, a, a);
    }
    return r;
}

// mlx nn.gelu on a bf16 array: x * (1 + erf(x / sqrt(2))) / 2, every op rounded to bf16 (sqrt(2) as bf16 1.4140625,
// the weakly typed scalar), checked against MLX 0.x on 200k values. Returns the bf16-exact result.
static inline float qd_gelu_erf(float g) {
    float a = bf(to_bf(precise::divide(g, 1.4140625f)));
    float e = bf(to_bf(qd_erf(a)));
    float s = bf(to_bf(1.0f + e));
    float p = bf(to_bf(g * s));
    return bf(to_bf(p * 0.5f));
}

// ---- qd_glu: out bf16 [n] = act(g) * u. act 0 = SiLU with MLX's bf16 chain (= gemm_q4.metal gq_silu_mul: e =
// bf(exp|g|), y = bf(1 / bf(1 + e)), s = g < 0 ? y : bf(1 - y), out = bf(bf(g * s) * u)); act 1 = tanh-GELU
// 0.5 g (1 + tanh(sqrt(2/pi) (g + 0.044715 g^3))) in f32, rounded once to bf16, out = bf(that * u); act 2 = exact-erf
// GELU (qd_gelu_erf), out = bf(that * u). out may alias g. Buffers: g [0], u [1], out [2], n [3], act [4]. Dispatch threads (n, 1).
kernel void qd_glu(device const ushort* g [[buffer(0)]], device const ushort* u [[buffer(1)]], device ushort* out [[buffer(2)]],
                   constant uint& n [[buffer(3)]], constant uint& act [[buffer(4)]], uint i [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
    if (i >= n) return;
    float gv = bf(g[i]), uv = bf(u[i]);
    float a;
    if (act == 0) {
        float e = bf(to_bf(precise::exp(fabs(gv))));
        float d = bf(to_bf(1.0f + e));
        float y = bf(to_bf(precise::divide(1.0f, d)));
        float s = (gv < 0.0f) ? y : bf(to_bf(1.0f - y));
        a = bf(to_bf(gv * s));
    } else if (act == 1) {
        float in = 0.7978845608028654f * (gv + 0.044715f * gv * gv * gv);
        a = bf(to_bf(0.5f * gv * (1.0f + precise::tanh(in))));
    } else {
        a = qd_gelu_erf(gv);
    }
    out[i] = to_bf(a * uv);
}

// ---- qd_act_b16: x bf16 [n] = act(x) in place (non-gated MLP). act 0 = SiLU (MLX bf16 chain as qd_glu: bf(g * s)),
// 1 = tanh-GELU (f32, one rounding), 2 = exact-erf GELU (qd_gelu_erf). Buffers: x [0], n [1], act [2]. Threads (n, 1).
kernel void qd_act_b16(device ushort* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& act [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
    if (i >= n) return;
    float gv = bf(x[i]);
    float a;
    if (act == 0) {
        float e = bf(to_bf(precise::exp(fabs(gv))));
        float d = bf(to_bf(1.0f + e));
        float y = bf(to_bf(precise::divide(1.0f, d)));
        float s = (gv < 0.0f) ? y : bf(to_bf(1.0f - y));
        a = gv * s;
    } else if (act == 1) {
        float in = 0.7978845608028654f * (gv + 0.044715f * gv * gv * gv);
        a = 0.5f * gv * (1.0f + precise::tanh(in));
    } else {
        a = qd_gelu_erf(gv);
    }
    x[i] = to_bf(a);
}

// Threadgroup sum: every thread gets the same total (simd_sum per simdgroup, partials in `part`, then every
// simdgroup sums the partials in the same order). Safe to call repeatedly with the same `part`.
static inline float qd_tg_sum(float v, threadgroup float* part, uint ntg, uint sg, uint lane) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a = lane < (ntg + 31) / 32 ? part[lane] : 0.0f;
    return simd_sum(a);
}

// ---- qd_layernorm: one threadgroup per row t (T groups x up to 1024 threads). mx.fast.layer_norm on bf16:
// m = sum(x) / n, v = sum((x - m)^2) / n, r = precise::rsqrt(v + eps), y = bf((x - m) r), out = bf(w y [+ b])
// (w y + b in f32, one rounding; checked against MLX on random rows of 64..4096). out may alias x.
// Buffers: x [0] (rows of n), w [1], b [2] (read when hasb), out [3] (rows of n), n [4], eps [5], hasb [6].
kernel void qd_layernorm(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                         device const ushort* b [[buffer(2)]], device ushort* out [[buffer(3)]],
                         constant uint& n [[buffer(4)]], constant float& eps [[buffer(5)]], constant uint& hasb [[buffer(6)]],
                         uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    threadgroup float part[32];
    device const ushort* xr = x + t * n;
    float s = 0;
    for (uint i = tid; i < n; i += ntg) s += bf(xr[i]);
    float mean = qd_tg_sum(s, part, ntg, sg, lane) / float(n);
    float v = 0;
    for (uint i = tid; i < n; i += ntg) { float c = bf(xr[i]) - mean; v += c * c; }
    float var = qd_tg_sum(v, part, ntg, sg, lane) / float(n);
    float r = precise::rsqrt(var + eps);
    device ushort* o = out + t * n;
    for (uint i = tid; i < n; i += ntg) {
        float y = bf(to_bf((bf(xr[i]) - mean) * r));
        float z = bf(w[i]) * y;
        if (hasb != 0) z += bf(b[i]);
        o[i] = to_bf(z);
    }
}

// ---- qd_rmsnorm_rows: in place, row t = x + t * stride (n values): x = bf(bf(w) * bf(x * r)), r = rsqrt(sum(x^2) / n
// + eps), sum partitioned as rmsnorm_b (MLX rms_single_row / rms_looped, N_READS = 4, n % 4 == 0). OLMo 2 q_norm /
// k_norm over the whole q / k projection of p rows. Buffers: x [0], w [1], n [2], eps [3], stride [4].
// Dispatch groups (T) x 1024.
kernel void qd_rmsnorm_rows(device ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                            constant uint& n [[buffer(2)]], constant float& eps [[buffer(3)]], constant uint& stride [[buffer(4)]],
                            uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    device ushort* xr = x + t * stride;
    float s = 0;
    for (uint r = 0; r < n; r += ntg * 4) {
        if (r + tid * 4 + 4 <= n) {
            for (uint i = 0; i < 4; i++) { float v = bf(xr[r + tid * 4 + i]); s += v * v; }
        }
    }
    float tot = qd_tg_sum(s, part, ntg, sg, lane);
    float r = precise::rsqrt(tot / float(n) + eps);
    for (uint i = tid; i < n; i += ntg) xr[i] = to_bf(bf(w[i]) * bf(to_bf(bf(xr[i]) * r)));
}

// ---- qd_logits_post: f32 logits [n]: x = x / div (div != 1), then x = cap * tanh(x / cap) (cap > 0).
// Buffers: x [0], n [1], div [2], cap [3]. Dispatch threads (n, 1).
kernel void qd_logits_post(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& div [[buffer(2)]],
                           constant float& cap [[buffer(3)]], uint i [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
    if (i >= n) return;
    float v = x[i];
    if (div != 1.0f) v = v / div;
    if (cap > 0.0f) v = cap * precise::tanh(v / cap);
    x[i] = v;
}

// ---- qd_embed_b16: E bf16 [V][H] (embed_tokens kept unquantized in the file, converted to bf16 at load), tok u32 [T],
// h bf16 [T][H] = E[tok[t]]. Buffers: E [0], tok [1], h [2], H [3]. Dispatch threads (H, T).
kernel void qd_embed_b16(device const ushort* E [[buffer(0)]], device const uint* tok [[buffer(1)]],
                         device ushort* h [[buffer(2)]], constant uint& H [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= H) return;
    h[t * H + i] = E[tok[t] * H + i];
}

// ---- qd_gemv_b16_f32: out f32 [R] = W bf16 [R][C] . x bf16 [C], fp32 products and sums (lane strided over the
// columns, then simd_sum). One simdgroup per row, 8 rows per threadgroup.
// Buffers: W [0], x [1], out [2], R [3], C [4]. Dispatch groups (ceil(R / 8)) x 256.
kernel void qd_gemv_b16_f32(device const ushort* W [[buffer(0)]], device const ushort* x [[buffer(1)]],
                            device float* out [[buffer(2)]], constant uint& R [[buffer(3)]], constant uint& C [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    uint r = tg * 8 + sg;
    if (r >= R) return;  // whole simdgroup leaves together (no barrier)
    device const ushort* w = W + ulong(r) * C;
    float acc = 0.0f;
    for (uint c = lane; c < C; c += 32) acc = fma(bf(w[c]), bf(x[c]), acc);
    acc = simd_sum(acc);
    if (lane == 0) out[r] = acc;
}
