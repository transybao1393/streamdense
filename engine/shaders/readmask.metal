// readmask.metal - read mask helpers (streamdense-model readmask.rs). Needs common_bf16 (bf) first. The masked
// matmul kernels themselves live in gemv_mma_q4.metal (qm_gemvx / qm_swiglux / qm_gemvtx / qm_swiglutx).

// Masked attention q heads: x [k][nh * hw] (the o_proj input ao) is zeroed on the columns of every head whose bit in
// hb is 0, unless fl[0] != 0 (read everything). The head's q / gate rows were not computed (zeros), so attention ran
// on q = 0; this makes the head's output exactly 0, as for any masked unit. Grid (nh * hw, k).
kernel void rm_zero_heads(device ushort* x [[buffer(0)]], device const uint* hb [[buffer(1)]],
                          device const uint* fl [[buffer(2)]], constant uint& hw [[buffer(3)]], constant uint& nh [[buffer(4)]],
                          uint2 i [[thread_position_in_grid]]) {
    if (fl[0] != 0u || i.x >= nh * hw) return;
    uint h = i.x / hw;
    if (((hb[h >> 5] >> (h & 31u)) & 1u) == 0u) x[i.y * nh * hw + i.x] = 0;
}

// "Seen" recorder of one layer, k rows: a 64-neuron FFN block b is active when some row t and neuron j of it has
// |a_tj| * ||W_down[:, j]|| > tf (the per-neuron contribution of activity.rs); a mixer head h when some row has
// sum_i |o_t,i| * ||W_out[:, i]|| > th over its hw output-projection input columns (o = DeltaNet yn / attention ao).
// Bits are OR-ed (atomic) into sf [nblk / 32 words] / sh [heads / 32 words]; the CPU reads and clears them between
// passes. Writes nothing any other kernel reads. Groups: inter / 64 + heads, 64 threads.
kernel void rm_seen(device const ushort* f [[buffer(0)]], device const ushort* mo [[buffer(1)]],
                    device const ushort* dn [[buffer(2)]], device const ushort* on [[buffer(3)]],
                    device atomic_uint* sf [[buffer(4)]], device atomic_uint* sh [[buffer(5)]],
                    constant uint& inter [[buffer(6)]], constant uint& heads [[buffer(7)]], constant uint& hw [[buffer(8)]],
                    constant uint& k [[buffer(9)]], constant float& tf [[buffer(10)]], constant float& th [[buffer(11)]],
                    uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[2];
    const uint nb = inter / 64;
    if (g < nb) {
        uint j = g * 64 + tid;
        float w = bf(dn[j]), m = 0.0f;
        for (uint t = 0; t < k; t++) m = max(m, fabs(bf(f[t * inter + j])) * w);
        m = simd_max(m);
        if ((tid & 31u) == 0u) part[tid >> 5] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u && max(part[0], part[1]) > tf) atomic_fetch_or_explicit(&sf[g >> 5], 1u << (g & 31u), memory_order_relaxed);
        return;
    }
    const uint h = g - nb, w = heads * hw;
    float m = 0.0f;
    for (uint t = 0; t < k; t++) {
        float s = 0.0f;
        for (uint i = tid; i < hw; i += 64u) s += fabs(bf(mo[t * w + h * hw + i])) * bf(on[h * hw + i]);
        s = simd_sum(s);
        if ((tid & 31u) == 0u) part[tid >> 5] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m = max(m, part[0] + part[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u && m > th) atomic_fetch_or_explicit(&sh[h >> 5], 1u << (h & 31u), memory_order_relaxed);
}
