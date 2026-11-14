// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (rms_norm kernel rms_single_row / rms_looped: per-thread partition, reduction tree, rounding).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// norm_bf16.metal - streamdense Qwen3.8-27B RMSNorms on bf16 (ushort) rows; math in fp32.
// Needs common_bf16.metal first in the concatenated source (bf, to_bf).
//
// rmsnorm_b      : MLX mx.fast.rms_norm on bf16 x / bf16 w (weights already carry +1 where the plan says so).
//                  r = precise::rsqrt(sum(x^2)/N + eps); out = bf(bf(w) * bf(x * r))
//                  (MLX: out = w * static_cast<T>(x * inv_mean), the bf16*bf16 product rounded to bf16).
// rmsnorm_gated  : mlx-lm Qwen3NextRMSNormGated(y, z) over each 128-dim DeltaNet v head, no +1:
//                  n = bf(bf(w) * bf(y * r))              (mx.fast.rms_norm, bf16 out)
//                  out = bf(silu(f32 z) * f32 n)          (precise_swiglu: gate in fp32, product in fp32, one rounding)
//                  silu(z) = z * sigmoid(z), sigmoid as MLX: s = 1/(1+exp(|z|)); z < 0 ? s : 1 - s.

// ---- rmsnorm_b: one threadgroup per token row (T groups x 1024 threads)
// x ushort[T][N] [0], w ushort[N] [1], out (row t at out + t*out_stride) [2], N [3], eps [4], out_stride [5]
kernel void rmsnorm_b(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                      device ushort* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                      constant float& eps [[buffer(4)]], constant uint& out_stride [[buffer(5)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    device const ushort* xr = x + t * n;
    // MLX rms_single_row / rms_looped (N_READS = 4): thread tid sums x[r + 4 tid .. r + 4 tid + 3] for r = 0,
    // 4 ntg, ... (n % 4 == 0); threads past the row add nothing. Same per-lane partition and simd_sum tree as MLX
    // for n <= 4096 (single row, any ntg >= n / 4) and for n > 4096 with ntg = 1024 (looped, lsize = 1024).
    float s = 0;
    for (uint r = 0; r < n; r += ntg * 4) {
        if (r + tid * 4 + 4 <= n) {
            for (uint i = 0; i < 4; i++) { float v = bf(xr[r + tid * 4 + i]); s += v * v; }
        }
    }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (ntg + 31) / 32 ? part[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) part[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = precise::rsqrt(part[0] / float(n) + eps);
    device ushort* o = out + t * out_stride;
    for (uint i = tid; i < n; i += ntg) o[i] = to_bf(bf(w[i]) * bf(to_bf(bf(xr[i]) * r)));
}

// ---- rmsnorm_gated: one threadgroup of 128 threads per (v head, token); dispatch (n_vh, T) groups x 128.
// Head dim = threads per threadgroup (128). n_vh = threadgroups_per_grid.x.
// y ushort[T][n_vh][128] [0], z base [1] (head h of token t at z + t*z_stride + h*128), z_stride (elements) [2],
// w ushort[128] [3] (no +1), out ushort[T][n_vh*128] [4], eps [5]
kernel void rmsnorm_gated(device const ushort* y [[buffer(0)]], device const ushort* z [[buffer(1)]],
                          constant uint& z_stride [[buffer(2)]], device const ushort* w [[buffer(3)]],
                          device ushort* out [[buffer(4)]], constant float& eps [[buffer(5)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint2 tg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    uint h = g.x, t = g.y, d = tg.x, ntg = tg.x;
    uint row = t * ng.x + h;
    float v = bf(y[row * d + tid]);
    // sum of squares as MLX rms_single_row: thread i < d / 4 sums 4 consecutive values, then simd_sum
    float s = 0;
    if (tid * 4 + 4 <= d) {
        for (uint i = 0; i < 4; i++) { float u = bf(y[row * d + tid * 4 + i]); s += u * u; }
    }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float a = lane < (ntg + 31) / 32 ? part[lane] : 0.0f;
        a = simd_sum(a);
        if (lane == 0) part[0] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = precise::rsqrt(part[0] / float(d) + eps);
    float nrm = bf(to_bf(bf(w[tid]) * bf(to_bf(v * r))));
    float zf = bf(z[t * z_stride + h * d + tid]);
    float e = 1.0f / (1.0f + exp(fabs(zf)));
    float sig = zf < 0.0f ? e : 1.0f - e;
    out[row * d + tid] = to_bf(zf * sig * nrm);
}
