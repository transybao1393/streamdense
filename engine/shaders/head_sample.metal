// head_sample.metal - streamdense Qwen3.8-27B: SSD embedding dequant, greedy head, teacher top-k, small copies.
// Concatenate after common_bf16.metal (bf, to_bf, SdCtrl). Plan sections 4.3, 5, 6.2.
//   embed_q4   staged 4-bit embedding rows -> h (bf16), MLX `dequantize` rounding
//   argmax_b   bf16-rounded logits -> ctrl.tok[t], top1[t], top2[t], rec[ctrl.pos + t] (lowest index on ties)
//   topk32     bf16-rounded logits -> top-32 (idx, val) per row + logsumexp (teacher-forced check)
//   copy_bytes raw byte copy (dumps)
//   select_row copy bf16 row (ctrl.mtp_rows - 1) of [rows][n] to [n] (MTP row select)
//   advance1   ctrl.pos += 1 after a plain pass

// ---- embedding: one staged row per token t, laid out [u32 w[H/8] | bf16 s[H/64] | bf16 b[H/64]]
// (H = 5120 -> 2560 + 160 + 160 = 2880 B). One thread per u32 (8 values).
// Dispatch EXACTLY (H/8, T) threads (Enc::threads((H / 8, T), tg), or groups that divide H/8): H is taken from the
// grid width, so no dim is hardcoded here. Row stride = H/2 + H/16 bytes.
// MLX affine_dequantize computes q * scale + bias in fp32 and rounds once to bf16: out = bf(q * s + b)
// (q * s is exact in fp32: 4-bit q times a bf16 scale). Verified against mx.dequantize (gemv.safetensors deq.*,
// embed.safetensors rows.y): the two-rounding form bf(bf(q * s) + b) misses ~half the values. Low nibble first.
kernel void embed_q4(device const uchar* stage [[buffer(0)]], device ushort* h [[buffer(1)]], constant uint& T [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
    uint i = gid.x, t = gid.y;
    uint W = grid.x;                 // u32 words per row = H / 8
    if (t >= T) return;
    uint H = W * 8, G = H / 64;
    device const uchar* row = stage + t * (H / 2 + G * 4);
    uint q = ((device const uint*)row)[i];
    device const ushort* sb = (device const ushort*)(row + H / 2);
    uint g = i / 8;                  // 8 words of 8 nibbles = one group of 64
    float s = bf(sb[g]), b = bf(sb[G + g]);
    device ushort* o = h + t * H + i * 8;
    for (uint k = 0; k < 8; k++) {
        o[k] = to_bf(fma(float((q >> (4 * k)) & 0xfu), s, b));
    }
}

// ---- greedy argmax + top-2 per row (adapted from head_q8 `argmax`). 1024 threads per threadgroup, threadgroup = row t.
// Each fp32 logit is rounded to bf16 before comparing (MLX logits are bf16); lowest index wins ties.
// ctrl.tok[t], ctrl.top1[t], ctrl.top2[t] for t < 2; rec[ctrl.pos + t] = (token bits, top-1, top-2, 0).
kernel void argmax_b(device const float* logits [[buffer(0)]], device SdCtrl* ctrl [[buffer(1)]], device float4* rec [[buffer(2)]],
                     constant uint& V [[buffer(3)]],
                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32], bs[32];
    threadgroup uint bi[32];
    device const float* lg = logits + t * V;
    float best = -INFINITY, sec = -INFINITY;
    uint bidx = 0xffffffffu;
    for (uint i = tid; i < V; i += 1024) {
        float v = bf(to_bf(lg[i]));
        if (bidx == 0xffffffffu || v > best) { sec = best; best = v; bidx = i; } else if (v > sec) sec = v;
    }
    float m = simd_max(best);
    uint mi = simd_min(best == m ? bidx : 0xffffffffu);
    float s2 = simd_max(bidx == mi ? sec : best);           // every lane but the winner offers its best
    if (lane == 0) { bv[sg] = m; bi[sg] = mi; bs[sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = bv[lane], sv = bs[lane];
        uint ix = bi[lane];
        float m2 = simd_max(v);
        uint r = simd_min(v == m2 ? ix : 0xffffffffu);
        float sec2 = simd_max(ix == r ? sv : v);
        if (lane == 0) {
            if (t < 2) { ctrl->tok[t] = r; ctrl->top1[t] = m2; ctrl->top2[t] = sec2; }
            rec[ctrl->pos + t] = float4(as_type<float>(r), m2, sec2, 0.0f);
        }
    }
}

// ---- banned ids (gateway `format: plan`): logits[t][ban[1 + i]] = -inf for i < min(ban[0], 31), before argmax_b.
// ban = u32 [n, id0 .. id30] (128 B). Ids >= V are skipped. Threadgroup = row t, 32 threads. Only dispatched when n > 0,
// so an unbanned pass is the same command stream as before; a banned id can never be the argmax (lossless greedy over
// the allowed ids: when no banned id would win, tok / top1 are unchanged).
kernel void ban_logits(device float* logits [[buffer(0)]], device const uint* ban [[buffer(1)]], constant uint& V [[buffer(2)]],
                       uint t [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    uint n = min(ban[0], 31u);
    if (i < n) { uint id = ban[1 + i]; if (id < V) logits[t * V + id] = -INFINITY; }
}

// ---- top-32 per row + logsumexp (teacher-forced check). 1024 threads per threadgroup, threadgroup = row t.
// Values are the bf16-rounded logits (as argmax_b); order = value descending, index ascending on ties.
// 32 selection rounds over the row: round r takes the best element strictly after the round r-1 pick in that order.
// NaN logits are never selected. lse[t] = max + log(sum exp(v - max)) over the bf16-rounded row, fp32.
// If V < 32 the missing slots get idx 0xffffffff, val -inf.
kernel void topk32(device const float* logits [[buffer(0)]], device uint* idx [[buffer(1)]], device float* val [[buffer(2)]],
                   device float* lse [[buffer(3)]], constant uint& V [[buffer(4)]],
                   uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    threadgroup float pick_v;
    threadgroup uint pick_i;
    device const float* lg = logits + t * V;
    float pv = INFINITY;
    uint pi = 0xffffffffu;           // "nothing picked yet": every non-NaN element qualifies
    bool first = true;
    float mx = -INFINITY;            // the round-0 pick is the row max
    for (uint r = 0; r < 32; r++) {
        float best = -INFINITY;
        uint bidx = 0xffffffffu;
        for (uint i = tid; i < V; i += 1024) {
            float v = bf(to_bf(lg[i]));
            if (isnan(v)) continue;
            bool after = first || v < pv || (v == pv && i > pi);
            if (after && (bidx == 0xffffffffu || v > best)) { best = v; bidx = i; }
        }
        float m = simd_max(best);
        uint mi = simd_min(best == m ? bidx : 0xffffffffu);
        if (lane == 0) { bv[sg] = m; bi[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = bv[lane];
            uint ix = bi[lane];
            float m2 = simd_max(v);
            uint w = simd_min(v == m2 ? ix : 0xffffffffu);
            if (lane == 0) {
                pick_v = w == 0xffffffffu ? -INFINITY : m2;
                pick_i = w;
                idx[t * 32 + r] = w;
                val[t * 32 + r] = pick_v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pv = pick_v;
        pi = pick_i;
        if (first) mx = pv;
        first = false;
        if (pi == 0xffffffffu) {     // row exhausted (V < 32 or NaNs): fill the rest
            if (tid == 0) for (uint q = r + 1; q < 32; q++) { idx[t * 32 + q] = 0xffffffffu; val[t * 32 + q] = -INFINITY; }
            break;
        }
    }
    // logsumexp around the row max
    float s = 0;
    if (mx > -INFINITY) {
        for (uint i = tid; i < V; i += 1024) {
            float v = bf(to_bf(lg[i]));
            if (!isnan(v)) s += exp(v - mx);
        }
    }
    s = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) bv[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = simd_sum(bv[lane]);
        if (lane == 0) lse[t] = mx > -INFINITY ? mx + log(v) : -INFINITY;
    }
}

// ---- raw byte copy (dumps): dispatch nbytes threads. Byte-wise, so any src/dst offset alignment works.
kernel void copy_bytes(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant uint& nbytes [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < nbytes) dst[i] = src[i];
}

// ---- MTP row select: dst[n] = src[row][n] with row = ctrl.mtp_rows - 1 (0 if mtp_rows == 0); dispatch n threads.
kernel void select_row(device const ushort* src [[buffer(0)]], device ushort* dst [[buffer(1)]], constant uint& n [[buffer(2)]],
                       device const SdCtrl* ctrl [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint rows = ctrl->mtp_rows;
    uint row = rows > 0 ? rows - 1 : 0;
    dst[i] = src[row * n + i];
}

// ---- plain pass: ctrl.pos += 1 (1 thread, after argmax_b)
kernel void advance1(device SdCtrl* ctrl [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) ctrl->pos += 1;
}
