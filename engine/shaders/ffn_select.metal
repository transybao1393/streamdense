// ffn_select.metal - FFN-SELECT TEST HOOK (streamdense-model ffn_select.rs; only with QwenOptions.ffn_select, off by default).
// After the swiglu of model layer l, per row t of the pass (one threadgroup per row), rows at absolute position
// pos0 + t >= prm[1] only: keep a set of neurons of io.f and write 0 to the rest. score s_j = |f_j| * dn_j.
//   prm[0] = 0: keep all (f untouched), 1: keep m[j] != 0, 2: top-K by s (exact in-pass oracle),
//            3: top-K by |silu(g_j)| (g = gate x of the same row, gemv'd into the scratch), 4: m[j] != 0 plus the top-K of the
//            rest by |silu(g_j)| (resident set + per-token fetch), 5: top-K of the m[j] != 0 neurons by |silu(g_j)|
//            (per-token selection inside the resident set), K = prm[4 + l].
// The last row also writes sel[j] (1 = kept), sc[j] = s_j (before masking), gs[j] = |silu(g_j)| and cov = [kept s, total s, kept count, fetched count (mode 4)].
// Rows before prm[1] are untouched and record nothing. Needs common_bf16 (bf, to_bf) first.

#define FS_MAXV 32u   // values per thread: inter <= 1024 * 32

static inline float fs_sum(float v, threadgroup float* part, uint tid, uint ntg) {
    v = simd_sum(v);
    if ((tid & 31u) == 0) part[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    if (tid < 32u) { s = tid < ntg / 32u ? part[tid] : 0.0f; s = simd_sum(s); }
    if (tid == 0) part[0] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}
static inline float fs_max(float v, threadgroup float* part, uint tid, uint ntg) {
    v = simd_max(v);
    if ((tid & 31u) == 0) part[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    if (tid < 32u) { s = tid < ntg / 32u ? part[tid] : 0.0f; s = simd_max(s); }
    if (tid == 0) part[0] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

kernel void ffn_select(device ushort* f [[buffer(0)]], device const ushort* g [[buffer(1)]], device const float* dn [[buffer(2)]],
                       device const uchar* m [[buffer(3)]], device uchar* sel [[buffer(4)]], device float* sc [[buffer(5)]],
                       device float* cov [[buffer(6)]], device const uint* prm [[buffer(7)]],
                       constant uint& inter [[buffer(8)]], constant uint& k [[buffer(9)]], constant uint& l [[buffer(10)]],
                       constant uint& pos0 [[buffer(11)]], device float* gs [[buffer(12)]],
                       uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    if (pos0 + t < prm[1]) return;               // uniform per threadgroup
    uint mode = prm[0], kk = prm[4u + l];
    device ushort* fr = f + t * inter;
    device const ushort* gr = g + t * inter;
    float s[FS_MAXV], r[FS_MAXV];
    uint nv = 0;
    float mx = 0.0f;
    for (uint j = tid; j < inter; j += ntg) {
        float v = fabs(bf(fr[j])) * dn[j];
        float q = v;
        if (mode >= 3u) { float x = bf(gr[j]); q = fabs(x / (1.0f + exp(-x))); }
        if (mode == 4u && m[j] != 0) q = -1.0f;   // resident: always kept, not a fetch candidate
        if (mode == 5u && m[j] == 0) q = -1.0f;   // not resident: never kept (thr >= 0)
        s[nv] = v; r[nv] = q; nv++;
        mx = max(mx, q);
    }
    // threshold: largest thr with count(r >= thr) >= kk (bisection; count(r >= 0) = inter >= kk)
    float thr = 0.0f;
    if (mode >= 2u) {
        if (kk == 0u) thr = INFINITY;
        else if (kk < inter) {
            float hi = fs_max(mx, part, tid, ntg) * 1.0001f + 1e-30f, lo = 0.0f;
            for (uint it = 0; it < 48u; it++) {
                float mid = 0.5f * (lo + hi);
                float c = 0.0f;
                for (uint i = 0; i < nv; i++) c += r[i] >= mid ? 1.0f : 0.0f;
                c = fs_sum(c, part, tid, ntg);
                if (c >= float(kk)) lo = mid; else hi = mid;
            }
            thr = lo;
        }
    }
    bool last = t + 1u == k;
    float km = 0.0f, tm = 0.0f, kc = 0.0f, fc = 0.0f;
    uint i = 0;
    for (uint j = tid; j < inter; j += ntg, i++) {
        bool keep = mode == 0u ? true : (mode == 1u ? m[j] != 0 : (mode == 4u && m[j] != 0) || r[i] >= thr);
        if (mode == 4u && keep && m[j] == 0) fc += 1.0f;
        if (!keep) fr[j] = 0;
        if (last) { sel[j] = keep ? 1 : 0; sc[j] = s[i]; float x = bf(gr[j]); gs[j] = fabs(x / (1.0f + exp(-x))); }
        tm += s[i];
        if (keep) { km += s[i]; kc += 1.0f; }
    }
    if (!last) return;
    km = fs_sum(km, part, tid, ntg);
    tm = fs_sum(tm, part, tid, ntg);
    kc = fs_sum(kc, part, tid, ntg);
    fc = fs_sum(fc, part, tid, ntg);
    if (tid == 0) { cov[0] = km; cov[1] = tm; cov[2] = kc; cov[3] = fc; }
}
