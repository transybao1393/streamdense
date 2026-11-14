// dflash2.metal - DFlash2 block drafter kernels (streamdense-model dflash2.rs, 2026-11-05). Built only when the drafter
// is loaded (QwenOptions.dflash2); the target's passes never dispatch them. Needs common_bf16.metal first (bf, to_bf).
// The drafter's matmuls are the engine's 4-bit K-row GEMVs (qwen_layers gemv_m / swiglu_m) and its norms rmsnorm_b;
// these kernels add what is specific to DFlash2: strided row copies, q/k norm + RoPE from separate q / k / v rows with
// the drafter's own table, a ring context KV, attention with a per-row window and no gate, the grouped dynamic conv.
// Layouts: bf16 as ushort; q rows [t][nq][hd]; k, v rows [t][nkv][hd]; context KV [nkv][ring][hd] (slot = position %
// ring); block KV [nkv][B][hd]; qr [t][nh][hd]; ao [t][nh * hd]; conv base [2][K][hidden]; conv dyn [t][2][K][groups].

// ---- df2_copy_rows: dst[t * ds + i] = src[t * ss + i] for i < n. Grid (n, rows). ss = 0 repeats one source row.
kernel void df2_copy_rows(device const ushort* src [[buffer(0)]], device ushort* dst [[buffer(1)]],
                          constant uint& n [[buffer(2)]], constant uint& ss [[buffer(3)]], constant uint& ds [[buffer(4)]],
                          uint2 g [[thread_position_in_grid]]) {
    if (g.x < n) dst[g.y * ds + g.x] = src[g.y * ss + g.x];
}

struct Df2Prep {
    uint nq;       // q heads (0 = context rows: k / v only)
    uint nkv;
    uint hd;
    uint rdims;    // rotary dims (first rdims of each head, pairs (i, i + rdims / 2))
    uint pos0;     // absolute position of row 0
    uint kv_rows;  // rows per kv head in Kd / Vd (ring, or B for the block)
    uint row0;     // Kd / Vd row of p row 0, before the modulo (the position for the ring)
    float eps;
};

// ---- df2_prep: threadgroup (head hh, row t), hd threads (thread d = one dim; hd multiple of 32, <= 1024).
// hh < nq: q head hh -> RMSNorm(qn) -> RoPE -> qr[t][hh]. hh >= nq: kv head j = hh - nq -> RMSNorm(kn) -> RoPE -> Kd, v
// copied -> Vd, both at row (row0 + t) % kv_rows of head j. Rounding as attn_prep; cos / sin precise (position pos0 + t).
kernel void df2_prep(device const ushort* qs [[buffer(0)]], device const ushort* ks [[buffer(1)]],
                     device const ushort* vs [[buffer(2)]], device const ushort* qnw [[buffer(3)]],
                     device const ushort* knw [[buffer(4)]], device ushort* qr [[buffer(5)]],
                     device ushort* Kd [[buffer(6)]], device ushort* Vd [[buffer(7)]],
                     device const float* invf [[buffer(8)]], constant Df2Prep& a [[buffer(9)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float tn[1024];
    uint hh = tg.x, t = tg.y, hd = a.hd;
    bool isq = hh < a.nq;                                   // uniform per threadgroup
    uint j = hh - a.nq;
    device const ushort* src = isq ? qs + (t * a.nq + hh) * hd : ks + (t * a.nkv + j) * hd;
    float x = bf(src[d]);
    float sq = simd_sum(x * x);
    if (lane == 0) part[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint g = 0; g < hd / 32; g++) tot += part[g];
    float r = precise::rsqrt(tot / float(hd) + a.eps);
    float w = bf(isq ? qnw[d] : knw[d]);
    float nv = bf(to_bf(w * bf(to_bf(x * r))));
    tn[d] = nv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = nv;
    uint hr = a.rdims / 2;
    if (d < a.rdims) {
        uint i = d < hr ? d : d - hr;
        float th = float(a.pos0 + t) * invf[i];
        float c = precise::cos(th), sn = precise::sin(th);
        float x1 = tn[i], x2 = tn[i + hr];
        o = d < hr ? x1 * c - x2 * sn : x1 * sn + x2 * c;
    }
    ushort ob = to_bf(o);
    if (isq) { qr[(t * a.nq + hh) * hd + d] = ob; return; }
    uint slot = (j * a.kv_rows + (a.row0 + t) % a.kv_rows) * hd + d;
    Kd[slot] = ob;
    Vd[slot] = vs[(t * a.nkv + j) * hd + d];
}

struct Df2Attn {
    uint nh;
    uint nkv;
    uint hd;       // multiple of 32, <= 256
    uint nb;       // block keys (all visible to every block row: bidirectional, is_causal false)
    uint ring;     // context KV rows per head
    uint anchor;   // position of block row 0
    uint ctx_end;  // context positions < ctx_end exist (<= anchor)
    uint lo;       // oldest context position still in the ring
    uint window;   // sliding window: block row t sees context position p iff anchor + t - p < window
    float scale;   // hd^-0.5
};

// ---- df2_attn: threadgroup (q head h, block row t), 256 threads = 8 simdgroups. Simdgroup g takes keys g, g + 8, ... of
// [the row's context window | block] with an online softmax (fp32); lane owns dims lane*D .. lane*D + D - 1 (D = hd /
// 32). The 8 partial (m, l, o) are combined at the end; ao = bf(o), no gate. GQA: q head h reads kv head h / (nh / nkv).
kernel void df2_attn(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
                     device const ushort* Vc [[buffer(2)]], device const ushort* Kb [[buffer(3)]],
                     device const ushort* Vb [[buffer(4)]], device ushort* ao [[buffer(5)]],
                     constant Df2Attn& a [[buffer(6)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8], sl[8];
    threadgroup float sacc[8 * 256];
    uint h = tg.x, t = tg.y, hd = a.hd, D = hd / 32;
    uint kvh = h / (a.nh / a.nkv);
    float q[8], acc[8];
    device const ushort* qp = qr + (t * a.nh + h) * hd + lane * D;
    for (uint i = 0; i < 8; i++) { q[i] = i < D ? bf(qp[i]) : 0.0f; acc[i] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    // this row's first visible context position; rows further into the block see fewer old keys
    uint first = max(int(a.lo), int(a.anchor) + int(t) + 1 - int(a.window));
    uint n_ctx = a.ctx_end > first ? a.ctx_end - first : 0;
    uint n = n_ctx + a.nb;
    for (uint j = sg; j < n; j += 8) {
        device const ushort* kp;
        device const ushort* vp;
        if (j < n_ctx) { uint o = (kvh * a.ring + (first + j) % a.ring) * hd + lane * D; kp = Kc + o; vp = Vc + o; }
        else { uint o = (kvh * a.nb + (j - n_ctx)) * hd + lane * D; kp = Kb + o; vp = Vb + o; }
        float s = 0.0f;
        for (uint i = 0; i < D; i++) s = fma(q[i], bf(kp[i]), s);
        s = simd_sum(s) * a.scale;
        float mn = max(m, s);
        float c = exp(m - mn), e = exp(s - mn);
        l = l * c + e;
        for (uint i = 0; i < D; i++) acc[i] = acc[i] * c + e * bf(vp[i]);
        m = mn;
    }
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    for (uint i = 0; i < D; i++) sacc[sg * 256 + lane * D + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hd) {
        float M = -INFINITY;
        for (uint g = 0; g < 8; g++) M = max(M, sm[g]);
        float L = 0.0f, o = 0.0f;
        for (uint g = 0; g < 8; g++) {
            float c = sm[g] == -INFINITY ? 0.0f : exp(sm[g] - M);
            L += sl[g] * c;
            o += sacc[g * 256 + tid] * c;
        }
        ao[(t * a.nh + h) * hd + tid] = to_bf(o / L);
    }
}

struct Df2Conv {
    uint h;        // hidden
    uint gs;       // group size
    uint groups;   // h / gs
    uint kern;     // taps K
    uint side;     // 0 = prepare (before the mixer), 1 = finish (after it)
    uint add;      // 0: out = conv; 1: out = out + conv (the layer's residual)
};

// ---- df2_conv: z-lab _grouped_dynamic_convolve. Grid (h, rows), one thread per element i of row t, group g = i / gs:
// conv[t][i] = sum over tap o <= t of (base[side][o][i] + dyn[t][side][o][g]) * x[t - o][i]. Rows before the block are
// zeros in z-lab (the drafter's convolution stays inside the block), so taps past row 0 are skipped. bf16 per operation
// in z-lab's order: base product, add, dynamic product, add.
kernel void df2_conv(device const ushort* x [[buffer(0)]], device const ushort* dy [[buffer(1)]],
                     device const ushort* base [[buffer(2)]], device ushort* out [[buffer(3)]],
                     constant Df2Conv& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    uint i = g.x, t = g.y, gr = i / a.gs;
    float o = 0.0f;
    uint nt = min(a.kern, t + 1);
    for (uint tap = 0; tap < nt; tap++) {
        float v = bf(x[(t - tap) * a.h + i]);
        uint sk = a.side * a.kern + tap;
        o = bf(to_bf(o + bf(to_bf(bf(base[sk * a.h + i]) * v))));
        o = bf(to_bf(o + bf(to_bf(bf(dy[t * 2 * a.kern * a.groups + sk * a.groups + gr]) * v))));
    }
    uint s = t * a.h + i;
    out[s] = a.add != 0 ? to_bf(bf(out[s]) + o) : to_bf(o);
}
