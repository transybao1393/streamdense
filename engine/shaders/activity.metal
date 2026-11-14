// activity.metal - activity recorder (streamdense-model activity.rs). Reads the rows a layer left behind (FFN
// activation io.f, mixer output io.yn / io.ao, residual io.h) and adds usage sums to its own f32 region. No other
// kernel reads what these write, so every model output is unchanged bit for bit. Needs common_bf16 (bf) first.

// Threadgroup sum of v, valid in thread 0 (ntg a multiple of 32, <= 1024). Every thread of the group must call it.
static inline float act_sum(float v, threadgroup float* part, uint tid, uint ntg) {
    v = simd_sum(v);
    if ((tid & 31u) == 0) part[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    if (tid < 32u) { s = tid < ntg / 32u ? part[tid] : 0.0f; s = simd_sum(s); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// prev = h (n = k * hidden values), before layer 0 of an exact pass (k <= ACT_PREV_ROWS).
kernel void act_begin(device const ushort* h [[buffer(0)]], device ushort* prev [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) prev[i] = h[i];
}

// After one layer, k rows. Groups [0, ng), ng = ceil(inter / ntg): an[j] += sum_t |f[t][j]| (one thread per neuron).
// Groups ng .. ng + heads - 1: ah[h] += sum_t ||mo[t][h * hd .. + hd]||. Last group: flags & 1: ar[0] += sum_t
// ||h[t] - prev[t]||, then prev = h (the next layer's input); flags & 2: cnt[0] += k rows, cnt[1] += 1 pass, cnt[2] += k
// rows with the residual sum (flags & 1).
kernel void act_layer(device const ushort* f [[buffer(0)]], device const ushort* mo [[buffer(1)]],
                      device const ushort* h [[buffer(2)]], device ushort* prev [[buffer(3)]],
                      device float* an [[buffer(4)]], device float* ah [[buffer(5)]], device float* ar [[buffer(6)]],
                      device uint* cnt [[buffer(7)]],
                      constant uint& inter [[buffer(8)]], constant uint& heads [[buffer(9)]], constant uint& hd [[buffer(10)]],
                      constant uint& hidden [[buffer(11)]], constant uint& k [[buffer(12)]], constant uint& flags [[buffer(13)]],
                      uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    uint ng = (inter + ntg - 1) / ntg;
    if (g < ng) {
        uint j = g * ntg + tid;
        if (j >= inter) return;
        float s = 0.0f;
        for (uint t = 0; t < k; t++) s += fabs(bf(f[t * inter + j]));
        an[j] += s;
        return;
    }
    if (g < ng + heads) {
        uint hh = g - ng, w = heads * hd;
        float acc = 0.0f;
        for (uint t = 0; t < k; t++) {
            float q = 0.0f;
            for (uint i = tid; i < hd; i += ntg) { float x = bf(mo[t * w + hh * hd + i]); q += x * x; }
            acc += sqrt(act_sum(q, part, tid, ntg));
        }
        if (tid == 0) ah[hh] += acc;
        return;
    }
    if (flags & 1u) {
        float acc = 0.0f;
        for (uint t = 0; t < k; t++) {
            float q = 0.0f;
            for (uint i = tid; i < hidden; i += ntg) { float d = bf(h[t * hidden + i]) - bf(prev[t * hidden + i]); q += d * d; }
            acc += sqrt(act_sum(q, part, tid, ntg));
        }
        // each thread rewrites only the elements it read above
        for (uint t = 0; t < k; t++) for (uint i = tid; i < hidden; i += ntg) prev[t * hidden + i] = h[t * hidden + i];
        if (tid == 0) ar[0] += acc;
    }
    if ((flags & 2u) && tid == 0) { cnt[0] += k; cnt[1] += 1u; if (flags & 1u) cnt[2] += k; }
}
