// dflash.metal - DFlash block drafter kernels (streamdense-model dflash.rs, 2026-11-03). Built only when a drafter is
// loaded (QwenOptions.dflash); the target's passes never dispatch them. Needs common_bf16.metal first (bf, to_bf).
// The drafter's matmuls are the engine's 4-bit K-row GEMVs (qwen_layers gemv_m / swiglu_m) and its norms rmsnorm_b;
// these kernels add what the target does not have: strided row copies (taps, mask rows), q/k norm + RoPE with an
// explicit position and a free KV layout (context KV and the block's own KV), attention over [context | block] with
// the block bidirectional, and an argmax that writes only the drafter's own output.
// Layouts: bf16 as ushort; p rows [t][P] = q|gate interleaved per head, then k heads, then v heads (as attn_gated);
// context KV [nkv][max_ctx][hd]; block KV [nkv][B][hd]; qr [t][nh][hd]; ao [t][nh * hd].

// ---- df_copy_rows: dst[t * ds + i] = src[t * ss + i] for i < n. Grid (n, rows). ss = 0 repeats one source row.
kernel void df_copy_rows(device const ushort* src [[buffer(0)]], device ushort* dst [[buffer(1)]],
                         constant uint& n [[buffer(2)]], constant uint& ss [[buffer(3)]], constant uint& ds [[buffer(4)]],
                         uint2 g [[thread_position_in_grid]]) {
    if (g.x < n) dst[g.y * ds + g.x] = src[g.y * ss + g.x];
}

struct DfPrep {
    uint P;        // p row stride (elements)
    uint k_off;    // first k head in the row
    uint v_off;    // first v head in the row
    uint nq;       // q heads (0 = context rows: k / v only)
    uint nkv;
    uint hd;
    uint rdims;    // rotary dims (first rdims of each head, pairs (i, i + rdims / 2))
    uint pos0;     // absolute position of row 0
    uint kv_rows;  // rows per kv head in Kd / Vd (max_ctx, or B for the block)
    uint row0;     // Kd / Vd row of p row 0
    float eps;
};

// ---- df_prep: threadgroup (head hh, row t), hd threads (thread d = one dim; hd multiple of 32, <= 1024).
// hh < nq: q head hh (row cols hh*2*hd ..) -> RMSNorm(qn) -> RoPE -> qr[t][hh]. hh >= nq: kv head j = hh - nq
// (cols k_off + j*hd) -> RMSNorm(kn) -> RoPE -> Kd, v (cols v_off + j*hd) copied -> Vd, both at row row0 + t of head j.
// Rounding as attn_prep: n = bf(bf(w) * bf(x * r)), RoPE out bf16; cos / sin precise (position pos0 + t).
kernel void df_prep(device const ushort* p [[buffer(0)]], device const ushort* qnw [[buffer(1)]],
                    device const ushort* knw [[buffer(2)]], device ushort* qr [[buffer(3)]],
                    device ushort* Kd [[buffer(4)]], device ushort* Vd [[buffer(5)]],
                    device const float* invf [[buffer(6)]], constant DfPrep& a [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float tn[1024];
    uint hh = tg.x, t = tg.y, hd = a.hd;
    bool isq = hh < a.nq;                                   // uniform per threadgroup
    uint j = hh - a.nq;
    device const ushort* row = p + t * a.P;
    device const ushort* src = isq ? row + hh * 2 * hd : row + a.k_off + j * hd;
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
    uint slot = (j * a.kv_rows + a.row0 + t) * hd + d;
    Kd[slot] = ob;
    Vd[slot] = row[a.v_off + j * hd + d];
}

struct DfAttn {
    uint nh;
    uint nkv;
    uint hd;       // multiple of 32, <= 256
    uint n_ctx;    // context keys 0 .. n_ctx - 1 (positions < the block's first position)
    uint nb;       // block keys (all visible to every block row: bidirectional)
    uint max_ctx;  // context KV rows per head
    uint P;        // p row stride (gate of head h at h*2*hd + hd)
    float scale;   // hd^-0.5
};

// ---- df_attn: threadgroup (q head h, block row t), 256 threads = 8 simdgroups. Simdgroup g takes keys g, g + 8, ...
// of [context | block] with an online softmax (fp32); lane owns dims lane*D .. lane*D + D - 1 (D = hd / 32). The 8
// partial (m, l, o) are combined at the end; ao = bf(o * sigmoid(gate)). GQA: q head h reads kv head h / (nh / nkv).
kernel void df_attn(device const ushort* qr [[buffer(0)]], device const ushort* Kc [[buffer(1)]],
                    device const ushort* Vc [[buffer(2)]], device const ushort* Kb [[buffer(3)]],
                    device const ushort* Vb [[buffer(4)]], device const ushort* p [[buffer(5)]],
                    device ushort* ao [[buffer(6)]], constant DfAttn& a [[buffer(7)]],
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
    uint n = a.n_ctx + a.nb;
    for (uint j = sg; j < n; j += 8) {
        device const ushort* kp;
        device const ushort* vp;
        if (j < a.n_ctx) { uint o = (kvh * a.max_ctx + j) * hd + lane * D; kp = Kc + o; vp = Vc + o; }
        else { uint o = (kvh * a.nb + (j - a.n_ctx)) * hd + lane * D; kp = Kb + o; vp = Vb + o; }
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
        o /= L;
        float gt = bf(p[t * a.P + h * 2 * hd + hd + tid]);
        float sgm = 1.0f / (1.0f + exp(-gt));
        ao[(t * a.nh + h) * hd + tid] = to_bf(o * sgm);
    }
}

// ---- df_argmax: threadgroup = logits row t (f32 [rows][V]); tok[t] = first index of the max, val[t] = the max.
kernel void df_argmax(device const float* lg [[buffer(0)]], device uint* tok [[buffer(1)]], device float* val [[buffer(2)]],
                      constant uint& V [[buffer(3)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    device const float* r = lg + t * V;
    const uint NONE = 0xffffffffu;
    float best = -INFINITY;
    uint idx = NONE;
    for (uint i = tid; i < V; i += ntg) { float v = r[i]; if (idx == NONE || v > best) { best = v; idx = i; } }
    for (uint o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_down(best, o);
        uint oi = simd_shuffle_down(idx, o);
        if (lane + o < 32 && oi != NONE && (idx == NONE || ov > best || (ov == best && oi < idx))) { best = ov; idx = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint ns = (ntg + 31) / 32;
        float b = bv[0];
        uint bx = bi[0];
        for (uint g = 1; g < ns; g++) {
            if (bi[g] != NONE && (bx == NONE || bv[g] > b || (bv[g] == b && bi[g] < bx))) { b = bv[g]; bx = bi[g]; }
        }
        tok[t] = bx;
        val[t] = b;
    }
}
