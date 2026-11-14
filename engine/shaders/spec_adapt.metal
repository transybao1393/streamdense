// spec_adapt.metal - streamdense adaptive speculative decode (spec_deep.rs): draft confidence and verifier top-3.
// Needs common_bf16.metal first (bf, to_bf). Read only by the CPU planner: nothing here feeds a token decision, so
// these values never change what is emitted (the exact verify pass decides every token).
//   spec_conf  per logits row: top-3 (idx, value) of the bf16-rounded logits (value descending, lowest index on
//              ties, as argmax_b) and the logsumexp over the row -> out[t * 8 + 0..7] =
//              [idx0, idx1, idx2 (uint bits), v0, v1, v2, lse, 0]. top-1 prob = exp(v0 - lse).
//
// Dispatch groups((rows, 1), 1024): threadgroup t = logits row t (row stride V). One pass over the row: each thread
// keeps its own top-3 and an online (max, sum exp) pair; three selection rounds over the 1024 thread heads pick the
// global top-3, and the (max, sum) pairs are merged by simd / threadgroup reductions. NaN logits are skipped.

kernel void spec_conf(device const float* logits [[buffer(0)]], device float* out [[buffer(1)]], constant uint& V [[buffer(2)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float bv[32], bm[32], bs[32];
    threadgroup uint bi[32];
    threadgroup float pick_v;
    threadgroup uint pick_i;
    device const float* lg = logits + (ulong)t * V;
    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY;
    uint i0 = 0xffffffffu, i1 = 0xffffffffu, i2 = 0xffffffffu;
    float m = -INFINITY, s = 0.0f;
    for (uint i = tid; i < V; i += 1024) {
        float v = bf(to_bf(lg[i]));
        if (isnan(v)) continue;
        // indices grow per thread: an equal value ranks after the kept ones (lowest index first)
        if (i0 == 0xffffffffu || v > v0) { v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = v; i0 = i; }
        else if (i1 == 0xffffffffu || v > v1) { v2 = v1; i2 = i1; v1 = v; i1 = i; }
        else if (i2 == 0xffffffffu || v > v2) { v2 = v; i2 = i; }
        if (v > m) { s = s * exp(m - v) + 1.0f; m = v; } else { s += exp(v - m); }
    }
    // ---- logsumexp: merge (m, s) over the threadgroup
    float M = simd_max(m);
    float S = simd_sum(m == -INFINITY ? 0.0f : s * exp(m - M));
    if (lane == 0) { bm[sg] = M; bs[sg] = S; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lse = 0.0f;
    if (sg == 0) {
        float mm = lane < nsg ? bm[lane] : -INFINITY;
        float ss = lane < nsg ? bs[lane] : 0.0f;
        float M2 = simd_max(mm);
        float S2 = simd_sum(mm == -INFINITY ? 0.0f : ss * exp(mm - M2));
        lse = M2 == -INFINITY ? -INFINITY : M2 + log(S2);
    }
    // ---- top-3: three rounds, each picks the best thread head; the winner pops it
    for (uint r = 0; r < 3; r++) {
        float hv = v0;
        uint hi = i0;
        float mx = simd_max(hi == 0xffffffffu ? -INFINITY : hv);
        uint mi = simd_min((hi != 0xffffffffu && hv == mx) ? hi : 0xffffffffu);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) { bv[sg] = mx; bi[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = lane < nsg ? bv[lane] : -INFINITY;
            uint ix = lane < nsg ? bi[lane] : 0xffffffffu;
            float m2 = simd_max(ix == 0xffffffffu ? -INFINITY : v);
            uint w = simd_min((ix != 0xffffffffu && v == m2) ? ix : 0xffffffffu);
            if (lane == 0) {
                pick_v = w == 0xffffffffu ? -INFINITY : m2;
                pick_i = w;
                out[t * 8 + r] = as_type<float>(w);
                out[t * 8 + 3 + r] = pick_v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pick_i != 0xffffffffu && hi == pick_i) { v0 = v1; i0 = i1; v1 = v2; i1 = i2; v2 = -INFINITY; i2 = 0xffffffffu; }
    }
    if (sg == 0 && lane == 0) { out[t * 8 + 6] = lse; out[t * 8 + 7] = 0.0f; }
}
