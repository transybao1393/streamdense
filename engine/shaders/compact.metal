// compact.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// Active-list compaction: compact_thr / compact_union_thr (decode7b, atomic order) and the ordered versions (deterministic); make_args / zero_u32 helpers.
#include <metal_stdlib>
using namespace metal;

kernel void make_args(device const uint* count [[buffer(0)]], device uint* args [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) { args[0] = max((count[0] + 7) / 8, 1u); args[1] = 1; args[2] = 1; }
}
kernel void zero_u32(device uint* p [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) p[i] = 0;
}
// real gate mask: active iff relu(g) > thr (thr > 0, so iff g > thr)
kernel void compact_thr(device const half* g [[buffer(0)]], device uint* idx [[buffer(1)]],
                        device atomic_uint* count [[buffer(2)]], constant uint& R [[buffer(3)]],
                        constant float& thr [[buffer(4)]], uint i [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    bool a = i < R && float(g[i]) > thr;
    uint one = a ? 1u : 0u;
    uint c = simd_sum(one);
    uint pre = simd_prefix_exclusive_sum(one);
    uint base = 0;
    if (lane == 0 && c > 0) base = atomic_fetch_add_explicit(count, c, memory_order_relaxed);
    base = simd_broadcast(base, 0);
    if (a) idx[base + pre] = i;
}
// union over K tokens of the real gate masks (g > thr): slot list + per-slot token bit set
kernel void compact_union_thr(device const half* g [[buffer(0)]], device uint* idx [[buffer(1)]], device atomic_uint* count [[buffer(2)]],
                              device uchar* am [[buffer(3)]], constant uint& R [[buffer(4)]], constant uint& K [[buffer(5)]],
                              constant float& thr [[buffer(6)]], device atomic_uint* tokcount [[buffer(7)]],
                              uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint bits = 0;
    if (i < R) for (uint t = 0; t < K; t++) if (float(g[t * R + i]) > thr) bits |= 1u << t;
    uint one = bits ? 1u : 0u;
    uint c = simd_sum(one);
    uint pre = simd_prefix_exclusive_sum(one);
    uint base = 0;
    if (lane == 0 && c > 0) base = atomic_fetch_add_explicit(count, c, memory_order_relaxed);
    base = simd_broadcast(base, 0);
    if (bits) { idx[base + pre] = i; am[base + pre] = uchar(bits); }
    uint pc = simd_sum(popcount(bits));                     // per-token active count (sum over the K tokens)
    if (lane == 0 && pc > 0) atomic_fetch_add_explicit(tokcount, pc, memory_order_relaxed);
}

// ---- ordered compaction (streamdense, deterministic order): one threadgroup of 1024 threads walks the gate in
// blocks of 1024 rows; simd prefix sum + a threadgroup scan over the simdgroup counts give each active row its rank,
// so idx[] is in increasing neuron order (deterministic, same set as compact_thr). count[0] = total (plain store).
kernel void compact_thr_ordered(device const half* g [[buffer(0)]], device uint* idx [[buffer(1)]],
                                device uint* count [[buffer(2)]], constant uint& R [[buffer(3)]],
                                constant float& thr [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sgoff[32];
    threadgroup uint total;
    uint nsg = (ntg + 31) / 32, base = 0;
    for (uint r0 = 0; r0 < R; r0 += ntg) {
        uint i = r0 + tid;
        bool a = i < R && float(g[i]) > thr;
        uint one = a ? 1u : 0u;
        uint pre = simd_prefix_exclusive_sum(one);
        uint c = simd_sum(one);
        if (lane == 0) sgoff[sg] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            uint v = lane < nsg ? sgoff[lane] : 0u;
            uint ex = simd_prefix_exclusive_sum(v), tot = simd_sum(v);
            if (lane < nsg) sgoff[lane] = ex;
            if (lane == 0) total = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (a) idx[base + sgoff[sg] + pre] = i;
        base += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) count[0] = base;
}
// ordered union over K tokens (k-token verify pass): as compact_thr_ordered, plus the per-slot token bit set and the
// sum of per-token counts in tokcount[0]
kernel void compact_union_ordered(device const half* g [[buffer(0)]], device uint* idx [[buffer(1)]], device uint* count [[buffer(2)]],
                                  device uchar* am [[buffer(3)]], constant uint& R [[buffer(4)]], constant uint& K [[buffer(5)]],
                                  constant float& thr [[buffer(6)]], device uint* tokcount [[buffer(7)]],
                                  uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sgoff[32], sgpc[32];
    threadgroup uint total, totpc;
    uint nsg = (ntg + 31) / 32, base = 0, pcs = 0;
    for (uint r0 = 0; r0 < R; r0 += ntg) {
        uint i = r0 + tid, bits = 0;
        if (i < R) for (uint t = 0; t < K; t++) if (float(g[t * R + i]) > thr) bits |= 1u << t;
        uint one = bits ? 1u : 0u;
        uint pre = simd_prefix_exclusive_sum(one);
        uint c = simd_sum(one), pc = simd_sum(popcount(bits));
        if (lane == 0) { sgoff[sg] = c; sgpc[sg] = pc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            uint v = lane < nsg ? sgoff[lane] : 0u, p = lane < nsg ? sgpc[lane] : 0u;
            uint ex = simd_prefix_exclusive_sum(v), tot = simd_sum(v), tp = simd_sum(p);
            if (lane < nsg) sgoff[lane] = ex;
            if (lane == 0) { total = tot; totpc = tp; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (bits) { idx[base + sgoff[sg] + pre] = i; am[base + sgoff[sg] + pre] = uchar(bits); }
        base += total; pcs += totpc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { count[0] = base; tokcount[0] = pcs; }
}

