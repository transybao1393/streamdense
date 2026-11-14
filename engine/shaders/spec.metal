// Portions adapted from MLX (https://github.com/ml-explore/mlx), MIT License, Copyright © 2023 Apple Inc.
// (rms_norm kernel rms_single_row / rms_looped: per-thread partition and reduction tree).
// Full MIT license text: THIRD_PARTY_NOTICES.md at the repository root.
//
// spec.metal - streamdense Qwen3.8-27B MTP speculative decode control (P2, U18). Plan 7.1 / 7.3.
// Needs common_bf16.metal first (bf, to_bf, SdCtrl). `advance1` lives in head_sample.metal (not redefined here).
//   mtp_prep    cat = [rms(emb, pre_fc_norm_embedding) | rms(hf, pre_fc_norm_hidden)] for the rows chosen by ctrl
//               (spec mode) or for T given rows (direct mode); writes mctrl (MTP-layer control: pos = mtp_len)
//   mtp_draft   argmax of the MTP logits row -> ctrl.draft; ctrl.mtp_len += mctrl.mtp_rows
//   mtp_advance ctrl.mtp_len += mctrl.mtp_rows (warm-up chunk without a draft)
//   spec_decide accept / reject / eos on the 2 verify rows -> emit list, pos, role flip, mtp_rows, stop
//
// MTP row semantics (plan 1 / 7.3): an MTP row pairs the target hidden h_p (after the final norm) with emb(t_{p+1})
// and sits at RoPE position / MTP KV slot p = mtp_len + row. emb goes FIRST in cat. The MTP norms already carry +1.

// ---- rms half: out[i] = bf(bf(w[i]) * bf(x[i] * r)), r = precise::rsqrt(sum(x^2)/n + eps).
// Same per-thread partition and reduction tree as rmsnorm_b (MLX rms_single_row / rms_looped, N_READS = 4,
// 1024 threads), so the result is bit-equal to rmsnorm_b on the same row.
static inline void spec_rms_row(device const ushort* xr, device const ushort* w, device ushort* o, uint n, float eps,
                                uint tid, uint ntg, uint sg, uint lane, threadgroup float* part) {
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
    for (uint i = tid; i < n; i += ntg) o[i] = to_bf(bf(w[i]) * bf(to_bf(bf(xr[i]) * r)));
}

// ---- mtp_prep: dispatch groups((T == 0 ? 2 : T, 2), 1024). Group (r, half): half 0 = emb part, 1 = hidden part.
// emb ushort[2][H] [0], hf ushort[2][H] [1], pre_e w[H] [2], pre_h w[H] [3], cat ushort[rows][2H] [4], ctrl [5],
// eps [6], H [7], T [8], mctrl [9].
// T == 0 (spec mode, enc B): rows = ctrl.mtp_rows; emb[0] = emb(draft d), emb[1] = emb(new t1); hf = verify rows.
//   accept (rows 2): cat0 = [emb[0] | hf0], cat1 = [emb[1] | hf1];  reject (rows 1): cat0 = [emb[1] | hf0].
// T > 0 (direct mode, prefill warm-up / first row): rows = T, cat[t] = [emb[t] | hf[t]].
// Group (0, 0) thread 0 also writes mctrl: pos = ctrl.mtp_len, mtp_rows = rows, role = 0 (bind mctrl as the ctrl of
// the MTP layer's attention kernels and of select_row).
kernel void mtp_prep(device const ushort* emb [[buffer(0)]], device const ushort* hf [[buffer(1)]],
                     device const ushort* pre_e [[buffer(2)]], device const ushort* pre_h [[buffer(3)]],
                     device ushort* cat [[buffer(4)]], device const SdCtrl* ctrl [[buffer(5)]],
                     constant float& eps [[buffer(6)]], constant uint& H [[buffer(7)]], constant uint& T [[buffer(8)]],
                     device SdCtrl* mctrl [[buffer(9)]],
                     uint2 g [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                     uint2 tg2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    uint tid = tid2.x, ntg = tg2.x;   // 1-D threadgroups (1024 x 1)
    uint r = g.x, half_ = g.y;
    uint rows = T == 0 ? min(ctrl->mtp_rows, 2u) : T;
    if (g.x == 0 && g.y == 0 && tid == 0) {
        mctrl->pos = ctrl->mtp_len;
        mctrl->mtp_rows = rows;
        mctrl->role = 0;
    }
    if (r >= rows) return;
    device ushort* o = cat + (ulong)r * 2 * H + half_ * H;
    if (half_ == 0) {
        uint er = T == 0 ? (rows == 2 ? r : 1u) : r;
        spec_rms_row(emb + (ulong)er * H, pre_e, o, H, eps, tid, ntg, sg, lane, part);
    } else {
        spec_rms_row(hf + (ulong)r * H, pre_h, o, H, eps, tid, ntg, sg, lane, part);
    }
}

// ---- mtp_draft: dispatch groups((1, 1), 1024). Same argmax as argmax_b (each fp32 logit rounded to bf16, lowest
// index on ties) on logits float[V] [0]; ctrl.draft = winner; ctrl.mtp_len += mctrl.mtp_rows. Writes nothing else.
kernel void mtp_draft(device const float* logits [[buffer(0)]], device SdCtrl* ctrl [[buffer(1)]],
                      device const SdCtrl* mctrl [[buffer(2)]], constant uint& V [[buffer(3)]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float best = -INFINITY;
    uint bidx = 0xffffffffu;
    for (uint i = tid; i < V; i += 1024) {
        float v = bf(to_bf(logits[i]));
        if (bidx == 0xffffffffu || v > best) { best = v; bidx = i; }
    }
    float m = simd_max(best);
    uint mi = simd_min(best == m ? bidx : 0xffffffffu);
    if (lane == 0) { bv[sg] = m; bi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = bv[lane];
        uint ix = bi[lane];
        float m2 = simd_max(v);
        uint r = simd_min(v == m2 ? ix : 0xffffffffu);
        if (lane == 0) {
            ctrl->draft = r;
            ctrl->mtp_len += mctrl->mtp_rows;
        }
    }
}

// ---- mtp_advance: threads((1, 1), 1). ctrl.mtp_len += mctrl.mtp_rows.
kernel void mtp_advance(device SdCtrl* ctrl [[buffer(0)]], device const SdCtrl* mctrl [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i == 0) ctrl->mtp_len += mctrl->mtp_rows;
}

// ---- spec_decide: threads((1, 1), 1), after argmax_b of the K = 2 verify pass [t1, d].
// ctrl [0], rec float4[max_ctx] [1], eos uint[n_eos] [2], n_eos [3]. a0 = tok[0], a1 = tok[1], d = draft, p = pos.
//   accept (a0 == d, a0 not eos): emit [d, a1], n_emit 2, pos += 2, mtp_rows 2, role kept (S[role] = post-d state),
//                                 stop = a1 is eos.
//   else (reject, or a0 is eos):  emit [a0], n_emit 1, pos += 1, mtp_rows 1, role ^= 1 (post-t1 checkpoint),
//                                 rec[p + 1] = 0 (stale d-row record), stop = a0 is eos.
// The eos-at-a0 case uses the reject state on purpose: the state then equals plain greedy after emitting a0
// (t1 fed, a0 not fed). step += 1.
static inline bool spec_is_eos(uint t, device const uint* eos, uint n_eos) {
    for (uint i = 0; i < n_eos; i++) if (eos[i] == t) return true;
    return false;
}

kernel void spec_decide(device SdCtrl* ctrl [[buffer(0)]], device float4* rec [[buffer(1)]],
                        device const uint* eos [[buffer(2)]], constant uint& n_eos [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint a0 = ctrl->tok[0], a1 = ctrl->tok[1], d = ctrl->draft, p = ctrl->pos;
    bool e0 = spec_is_eos(a0, eos, n_eos);
    if (a0 == d && !e0) {
        ctrl->emit[0] = d;
        ctrl->emit[1] = a1;
        ctrl->n_emit = 2;
        ctrl->pos = p + 2;
        ctrl->mtp_rows = 2;
        ctrl->stop = spec_is_eos(a1, eos, n_eos) ? 1u : 0u;
    } else {
        ctrl->emit[0] = a0;
        ctrl->emit[1] = 0;
        ctrl->n_emit = 1;
        ctrl->pos = p + 1;
        ctrl->mtp_rows = 1;
        ctrl->role ^= 1u;
        rec[p + 1] = float4(0.0f);
        ctrl->stop = e0 ? 1u : 0u;
    }
    ctrl->step += 1;
}
