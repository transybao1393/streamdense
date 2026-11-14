// misc.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// decode7b embed, rmsnorm, add_res, copy_u32.
#include <metal_stdlib>
using namespace metal;

kernel void add_res(device const half* v [[buffer(0)]], device const float* res [[buffer(1)]], device float* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]], uint j [[thread_position_in_grid]]) {
    if (j < n) out[j] = res[j] + float(v[j]);
}

// ---- embedding, RMSNorm (one threadgroup per token row)
kernel void embed(device const half* E [[buffer(0)]], device const uint* tokbuf [[buffer(1)]], device float* h [[buffer(2)]],
                  constant uint& pos [[buffer(3)]], constant uint& H [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= H) return;
    uint tok = tokbuf[pos + t];
    h[t * H + i] = float(E[tok * H + i]);
}
kernel void rmsnorm(device const float* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]], constant float& eps [[buffer(4)]],
                    uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    device const float* xr = x + t * n;
    float s = 0;
    for (uint i = tid; i < n; i += ntg) { float v = xr[i]; s += v * v; }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < ntg / 32 ? part[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) part[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rsqrt(part[0] / float(n) + eps);
    for (uint i = tid; i < n; i += ntg) out[t * n + i] = half(float(w[i]) * float(half(xr[i] * r)));
}
// copy n uint32 (per-pass active counts into a per-position record)
kernel void copy_u32(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]], constant uint& n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}
