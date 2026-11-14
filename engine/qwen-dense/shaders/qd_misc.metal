// qd_misc.metal - qwen-dense engine glue kernels (needs engine/shaders/common_bf16.metal first: bf, to_bf).
//   qd_embed_q4  token ids -> h bf16 rows, dequantized from the MLX 4-bit embedding (tied lm_head source)
//   qd_bias_add  p += bias per row (q/k/v projection bias of Qwen2 / 2.5), bf16
//   qd_next_tok  pipelined decode: previous pass greedy token (ctrl.tok[0]) -> token row, ctrl.pos = pos

// ---- qd_embed_q4: W u32 [V][H/8] (8 nibbles per u32, low nibble first), S / B bf16 [V][H/64] (w = s*q + b),
// tok u32 [T], h bf16 [T][H]. One thread per u32 (8 values). Dispatch threads (H/8, T).
// MLX dequantize rounding: fp32 fma(q, s, b), rounded once to bf16 (as engine head_sample.metal embed_q4).
// Buffers: W [0], S [1], B [2], tok [3], h [4], H [5].
kernel void qd_embed_q4(device const uint* W [[buffer(0)]], device const ushort* S [[buffer(1)]],
                        device const ushort* B [[buffer(2)]], device const uint* tok [[buffer(3)]],
                        device ushort* h [[buffer(4)]], constant uint& H [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    uint W8 = H / 8, G = H / 64;
    if (i >= W8) return;
    uint id = tok[t];
    uint q = W[id * W8 + i];
    uint g = i / 8;
    float s = bf(S[id * G + g]), b = bf(B[id * G + g]);
    device ushort* o = h + t * H + i * 8;
    for (uint k = 0; k < 8; k++) o[k] = to_bf(fma(float((q >> (4 * k)) & 0xfu), s, b));
}

// ---- qd_bias_add: p bf16 [T][P] += bias bf16 [P] (MLX: bf16 matmul output + bf16 bias, one rounding).
// Buffers: p [0], bias [1], P [2]. Dispatch threads (P, T).
kernel void qd_bias_add(device ushort* p [[buffer(0)]], device const ushort* bias [[buffer(1)]],
                        constant uint& P [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= P) return;
    uint o = t * P + i;
    p[o] = to_bf(bf(p[o]) + bf(bias[i]));
}

// ---- qd_next_tok: start of a pipelined decode pass (src/engine.rs DecodeMode::Pipelined). The previous pass's
// argmax_b left its greedy token in ctrl.tok[0]; copy it to the pass's token row and set ctrl.pos, i.e. exactly what
// the host writes before a synchronous pass (ctrl.tok / top1 / top2 are written by argmax_b before anything reads
// them, so only pos and the token matter). Lets the CPU encode pass t+1 while pass t runs.
// Buffers: ctrl [0] (SdCtrl, common_bf16.metal), tok [1] (u32 [rows]), pos [2]. Dispatch threads (1, 1).
kernel void qd_next_tok(device SdCtrl* ctrl [[buffer(0)]], device uint* tok [[buffer(1)]],
                        constant uint& pos [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    tok[0] = ctrl->tok[0];
    ctrl->pos = pos;
}
