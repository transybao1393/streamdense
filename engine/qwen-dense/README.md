# qwen-dense

Standalone engine for ordinary **Qwen dense** models (Qwen2 / Qwen2.5 / Qwen3 dense: every layer full attention,
head dim 128, GQA) on Metal. Own crate (empty `[workspace]` table in `Cargo.toml`), not a member of the engine
workspace: `cargo` in `engine/` never builds it. The model layer ([../model_layer](../model_layer/README.md),
`sd-model-layer`) is the user-facing entry point that routes these models here; this README is the engine's own
reference.

## What is here

| file | what |
|---|---|
| `src/config.rs` | `DenseConfig`: config.json reader for `model_type` qwen2 / qwen3 (also inside `text_config`, and the newer `rope_parameters` form). `validate()` / `unsupported()` name every unsupported feature. Hybrid configs (Qwen3.5 / 3.8: `linear_attention` layers, `linear_*` fields, `full_attention_interval`) are refused with "hybrid ... use the main engine"; MoE configs are refused. Unit tests embed Qwen2.5-0.5B / 7B / 14B and Qwen3-8B / 14B / 32B config shapes (values not checked against the published config are marked `unverified` in comments). |
| `shaders/attn_hd128.metal` | Port of `engine/shaders/attn_gated.metal`'s split-K pipeline to head dim 128: `attn128_rope_invf`, `attn128_prep_{b16,q8}` (optional QK-norm, RoPE with any rdims (full rotary for these models), KV append, q8 per-32-dim quantization), `attn128_partial_k{1..8}_{b16,q8}` (256-key chunks, GQA), `attn128_reduce` (plain, Qwen dense) and `attn128_reduce_gate` (sigmoid output gate). Scale 128^-0.5. Same buffer indices and grid shapes as the attn_gated kernels; prep has one extra uniform, `flags` [14] (bit 0 QK-norm, bit 1 gated q layout). |
| `src/gpu.rs` | Minimal Metal harness (same objc2 / objc2-metal versions as the engine): device, library from `SHADER_SOURCE` (the engine's `common_bf16.metal`, read via `include_str!`, + `attn_hd128.metal`, `attn_hd128_flash.metal`, `attn_hd128_sdpa.metal`), pipelines, one arena buffer, encoder. Safe math mode, as the engine's `Gpu::from_shader_files`. |
| `tests/attn_hd128.rs` | GPU kernels vs a CPU f32 reference at Qwen dense shapes; K-invariance. |
| `src/weights.rs` | MLX 4-bit loader (see "Engine" below): `TensorSource` (`SafeSource` = model dir via streamdense-core `SafeIndex` + pread, `MemSource` = in-memory for tests), `layout()` + `load_into()`, `Weights` / `LayerW` / `Q4`. |
| `src/model.rs` | Kernel dispatch of one K-token pass (embed, layer, head), MMA gemv dispatch rule (chunks of <= 8 rows), `AttnImpl` switch (prep + core: auto / splitk / flash / sdpa), `SDPA128_TWO_PASS_MIN`, `sdpa128_vec_max_n`. |
| `src/engine.rs` | `Engine` (load, `forward`, `prefill` (GEMM or passes), `step`, `generate`, `logits`), `PrefillMode`, `GenStats`, `peak_footprint()`. |
| `src/prefill.rs` | GEMM prefill (q4b_gemm + attn128_flash over up to 512 rows per chunk); see its section. |
| `shaders/qd_misc.metal` | `qd_embed_q4` (token ids -> bf16 rows from the 4-bit embedding, MLX dequant rounding), `qd_bias_add`. |
| `examples/qd_run.rs` | CLI: greedy generation, timings, peak footprint, optional JSON. |
| `examples/qd_compare.rs` | Real-model check: runs the exact prompt ids of an MLX reference JSON (free run + teacher forced), reports identical prefix, first divergence and logit gaps, timings, peak footprint. |
| `tests/engine_synth.rs` | Whole engine on synthetic tiny Qwen3 / Qwen2 models vs a CPU reference, every attention impl; K-invariance per impl; sdpa long-row fallback; tokenizers; real-model weight layout from headers only. |

Dispatch shape (same as `qwen_layers::encode_attn` for attn_gated, HD 128):

```
attn128_rope_invf   threads (rdims/2, 1) x rdims/2               once per model
attn128_prep_{b16,q8}   groups (nh + nkv, K) x 128               buffers 0..13 as attn_prep, + flags [14]
attn128_partial_kK_{b16,q8}  groups (nch, nh) x 128              nch >= ceil((pos + K) / 256)
attn128_reduce / _gate  groups (nh, K) x 32                      p [2] and p_stride [5] read only by _gate
```

p layout: `[T][nh*HD | nkv*HD | nkv*HD]` (q | k | v) plain, or `[T][nh*2*HD | ...]` (q | gate interleaved per head)
with the gated flag.

## What is NOT done

- Wiring into the main engine (`qwen_layers` / `qwen_engine`, gateway). The standalone engine here (section
  "Engine") runs Qwen dense models on its own.
- Converter for dense Qwen checkpoints into the main engine's weight format (the standalone engine reads MLX 4-bit
  safetensors directly).
- YaRN or any other `rope_scaling` (reported as unsupported); sliding-window layers.
- The split-K path is accurate to the tolerances below but not bit-exact vs MLX (the `sdpa` path is; see the
  next section).
- Head dim 64 (Qwen2.5-0.5B: 896 / 14 = 64) — `validate()` rejects it.

## Run the tests

```
cd engine/qwen-dense
cargo test --release -- --nocapture --test-threads=1     # config unit tests + GPU kernel and synthetic engine tests
cargo clippy --all-targets -- -D warnings
```

The GPU tests need a Metal device and use small buffers (largest arena ~ 12 MB); no model is loaded (the engine tests
read only config.json / tokenizer.json / safetensors headers of the local models, and skip when they are missing).

Tolerances (attention output vs the CPU f32 reference over the GPU's own qr and KV cache): <= 2 bf16 ulp, or abs err
<= 2e-5 for near-cancelled outputs (|ref| < 1/64). Prep (qr, K): <= 1 bf16 ulp, or abs <= 1e-6 for near-cancelled
RoPE outputs; V copy exact; q8 codes within 1 of the CPU quantization, V codes and scales exact. K-token passes are
bit-identical to K one-token passes.

## HD-128 prefill flash + MLX decode mirror

`shaders/attn_hd128_flash.metal` (`attn128_flash`) and `shaders/attn_hd128_sdpa.metal` (`attn128_sdpa_vec`): ports of
the engine's `attn_flash_b16` / `attn_sdpa_vec_b16` to HD 128, ungated by default (`flags` bit 0 = sigmoid gate, q|gate
interleaved p layout). Both read the bf16 cache written by `attn128_prep_b16`; neither depends on `attn_hd128.metal`'s
constants. The library must contain them (`common_bf16.metal` first); both are in `SHADER_SOURCE` / `ENGINE_SHADER_SOURCE`
(src/lib.rs) and wired into the engine as `AttnImpl::Flash` / `Sdpa` / `Auto` (see "Real-model test").

```
attn128_flash     groups (ceil(T/8), nkv) x 32*grp (grp <= 32)    qr [0] Kc [1] Vc [2] p [3] ao [4] ctrl [5] max_ctx [6]
                                                                  grp [7] p_stride [8] T [9] flags [10]
attn128_sdpa_vec  groups (nh, T) x 1024                           qr [0] Kc [1] Vc [2] p [3] ao [4] ctrl [5] max_ctx [6]
                                                                  grp [7] p_stride [8] lim [9] two_pass_min [10] flags [11]
```

Row t sees keys 0 ..= ctrl.pos + t. `attn128_sdpa_vec` skips rows with N = pos + t + 1 > lim (run split-K for those
first). lim <= 8192, and <= 8191 for grp 8 / 12 / 16 (MLX 0.32.4 switches to its `sdpa_vector_2pass_1_gqa` there at
N >= 8192, not mirrored); two_pass_min = 1024 (MLX on M4 Pro). Kv buffers must be 16-byte aligned for the flash kernel.

Results (`cargo test --release --test attn_hd128_flash -- --nocapture --test-threads=1`, ~2 s total):
- `attn128_sdpa_vec` vs `mx.fast.scaled_dot_product_attention` (mlx 0.32.4, bf16, L = 1, D 128): **141 / 141 cases
  bit-identical** (shapes 16/8, 12/2, 32/8, 28/4, 40/8, 64/8, 48/4, 64/4; N 1 .. 8192, single- and two-pass, two input
  scales). The 3 cases N = 8192 with grp 8 / 12 / 16 differ (MLX gqa variant) and are excluded by lim. Inputs come from
  a splitmix64 hash generator (in the test) so only MLX's outputs are stored; the test skips if the data file
  ($QDENSE_MLX_SDPA128; not part of this repository) is missing.
- vs CPU f32: flash <= 2 bf16 ulp (or abs <= 2e-5 where |ref| < 1/64) on 16/8, 12/2, 32/8, T 1..512, ctx 1..1100;
  sdpa_vec same bar for N < 1024, two-pass rows within the bf16-partial bound (e_d / 256 + 1 ulp). T-row sdpa_vec ==
  T one-row dispatches bit for bit. flash vs split-K: <= 2 ulp / abs 2e-5 bar holds, ~0.07 % of outputs differ.

## Engine: minimal all-attention Qwen dense decode + prefill

Runs MLX 4-bit Qwen2 / 2.5 / 3 dense models (head dim 128) end to end on Metal: greedy only, bf16 KV cache, one
command buffer per pass; `generate` pipelines decode passes (see "Pipelined decode" below). No MTP, speculation, chat
template or sampling. Tested models (MLX 4-bit conversions, placed under `models/` of this folder, not tracked):
`models/qwen3-1.7b-mlx4` = Qwen/Qwen3-1.7B (QK-norm, tied, bf16 scales), `models/qwen2.5-1.5b-instruct-mlx4` =
Qwen/Qwen2.5-1.5B-Instruct (q/k/v bias, tied, **f16** scales / biases / norms in the file).

```
cargo run --release --example qd_run -- --model models/qwen3-1.7b-mlx4 --prompt "The capital of France is" \
    --max-tokens 32 [--tokens-out out.json] [--max-ctx N] [--attn auto|splitk|flash|sdpa] [--prefill gemm|passes] [--decode pipelined|sync] [--no-stop]
```

Prints prompt ids, generated ids, text, load time, prefill time (tok/s), decode tok/s and the lifetime peak
phys_footprint. Raw prompt tokens (no BOS, no template); stops after an eos id of config.json unless `--no-stop`.
`max_ctx` defaults to prompt + max-tokens + 8 rounded up to 256.

API (`qwen_dense::`):

```rust
Engine::load(dir, EngineOptions { max_ctx, attn: AttnImpl::Auto, prefill: PrefillMode::Gemm, decode: DecodeMode::Pipelined }) -> Result<Engine, String>  // defaults
Engine::from_source(cfg: DenseConfig, src: &dyn TensorSource, opts)   // tests: weights::MemSource
eng.forward(&tokens /* 1..=pass_max */, head: bool) -> Result<Option<u32>>   // one pass at eng.pos(); greedy next if head
eng.prefill(&tokens) -> Result<u32>     // passes of up to pass_max (8 splitk/sdpa, 128 flash/auto), head on the last only;
                                        // auto: last prompt token as its own 1-row pass (mlx_lm generate_step order)
eng.step(tok) -> Result<u32>            // one decode pass
eng.generate(&prompt, max_new, &stop, |tok| ..) -> Result<GenStats>   // reset + prefill + steps
eng.logits() -> Vec<f32>                // f32 logits of the last head pass;  eng.reset(), eng.pos()
Tokenizer::from_path(dir)               // = streamdense-tokenizer (read-only path dep): encode / decode, StreamDecoder
engine::peak_footprint(), engine::footprint()
```

Pass (src/model.rs), K = 1..=8 rows (1..=128 for flash / auto; matmuls then run in chunks of <= 8 rows), every buffer
an offset in one shared arena (weights | KV per layer | scratch):

```
qd_embed_q4                   threads (H/8, K)                  tok ids -> h bf16 (4-bit tied embedding, MLX rounding)
per layer:
  rmsnorm_b                   groups (K) x 1024                 x = norm(h)                     (engine norm_bf16.metal)
  qkv gemv MODE 0             MMA family, see below             p = [K][q | k | v] (q/k/v fused row-wise at load)
  qd_bias_add (Qwen2/2.5)     threads (P, K)                    p += bias (fused q|k|v bias)
  attn128_prep_b16            groups (nh+nkv, K) x 128          flags bit 0 = QK-norm (Qwen3); RoPE full rotary; KV append
  AttnImpl core               splitk: attn128_partial_kK_b16 (nch, nh) x 128 + attn128_reduce (nh, K) x 32
                              flash:  attn128_flash (ceil(K/8), nkv) x 32*grp, flags 0
                              sdpa:   attn128_sdpa_vec (nh, K) x 1024, lim = sdpa128_vec_max_n(grp), two_pass_min 1024
                                      (rows N > lim: split-K first, sdpa skips them)
                              auto (default): K > 1 flash, K = 1 sdpa
  o_proj gemv MODE 2          h += o ao (residual in place)
  rmsnorm_b; SwiGLU (gate, up) -> f; down gemv MODE 2 (h += down f)
head (last row): rmsnorm_b -> hf; lm_head gemv MODE 3 (f32 logits; tied = the embedding matrix); argmax_b -> ctrl.tok[0]
```

- Matmuls: engine `gemv_mma_q4.metal` (include_str!, unmodified), same rule as `qwen_layers::gemv` with
  `GemvImpl::Mma`: twins `qm_gemvt_k{K}_s8_m*` / `qm_swiglut_k{K}_s8` where `mma_use_twin` (K <= 2 always; K 3..4 by
  shape), else `qm_gemv_n2_s8_m*` / `qm_swiglu_n1_s8`. Bit-identical per row, so prefill == token-by-token decode.
- Swapping attention: add an `AttnImpl` variant in src/model.rs, list its kernels in `AttnImpl::pipelines`, add a
  match arm in `encode_attn_core` (reads `AttnIo`: qr [k][nh][128], bf16 Kc / Vc [nkv][max_ctx][128], p, ao, ctrl,
  max_ctx, pos, k), add its .metal file to `ENGINE_SHADER_SOURCE` (src/lib.rs) and a name to `AttnImpl::parse`
  (qd_run `--attn`). The prep (`encode_attn_prep`) is shared. Flash needs 16-byte-aligned KV (the arena aligns to 256).
- Weights (`weights::layout`): q/k/v weight, scales, biases and bias concatenated row-wise; F16 / F32 float tensors
  converted to bf16 at load (RNE); lm_head = embed when tied (else `lm_head.*`). Group size must be 64, bits 4.

Tests (`tests/engine_synth.rs`, ~2 s, no model weights read): synthetic 2-layer Qwen3-style (QK-norm, GQA 4/2, bf16)
and Qwen2-style (bias, 2/1, f16 scales) models vs a CPU reference: logits after a 300-token prefill (2 attention
chunks) and 3 decode steps within 1.5 % of max |logit| + 0.01 (measured max diff 0.017 - 0.028 at max |logit| 2.7 -
4.1), same argmax; an ablated reference (no bias / no QK-norm) misses by 1.1 - 4.2, so the check sees both features.
Prefill in passes of 8 == one token per pass == mixed pass sizes 1..8, logits bit-identical. Tokenizers of both local
models round-trip text. Weight layout of both local models from their safetensors headers: every tensor used once.

Not done / unsure:
- Real-model run: see "Real-model test" below.
- Qwen2.5 MLX files keep f16 scales / biases; converting them to bf16 adds rounding MLX does not have (MLX computes
  that model in f16 end to end; this engine computes in bf16). Expect small logit differences vs MLX for Qwen2.5.
- One arena = one MTLBuffer: weights + KV must fit `maxBufferLength` (checked, error otherwise).
- Prefill: default GEMM path (src/prefill.rs, prompts >= 16 tokens). `--prefill passes`: the K <= 8 decode matmul
  kernels in chunks (weights re-read per 8 rows), flash attention for passes of up to 128 rows (auto / flash).

## Real-model test (2026-11-02)

Both tested models, 6 prompts each (Python code, Rust code, English prose, a fact pattern, a chat-templated question
built with the tokenizer's own chat template, and the first 1,000 tokens of a long Markdown text), 128 greedy tokens,
exact same prompt token ids on both sides (no BOS). MLX reference: mlx 0.32.4 / mlx_lm 0.32.2 `generate_step` (prompt
minus its last token in one prefill call, then the decode path; greedy = argmax of logprobs; no eos stop). Ours:
`examples/qd_compare.rs` with three engine settings: `auto+gemm` (the default: GEMM prefill of prompt[..n-1] for
prompts >= 16 tokens, then the last token and decode through the MLX-mirror sdpa), `auto+passes` (flash prefill passes
of <= 128 rows), `splitk+passes`. One model process at a time, GPU otherwise idle, M4 Pro. Prefill times are the
minimum of 4 runs on both sides; decode tok/s are single 128-token runs (MLX's vary by up to ~10 % run to run).

Identical-token prefix out of 128 (auto+gemm / auto+passes / splitk+passes):

| model | code_py | code_rust | prose | fact | chat | long_1000 | teacher-forced |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B (bf16 both) | 128 / 128 / 128 | 128 / 128 / 128 | 128 / 128 / 128 | 128 / 128 / 128 | 128 / 86 / 37 | 11 / 36 / 11 | 96.9 - 100 % |
| Qwen2.5-1.5B (MLX f16, ours bf16) | 128 / 128 / 128 | 91 / 128 / 91 | 35 / 35 / 35 | 128 / 128 / 128 | 56 / 60 / 46 | 5 / 6 / 5 | 96.9 - 100 % |

Speed (1,000-token prompt prefill, min of 4; decode tok/s over the 6 prompts):

| model | prefill auto+gemm | auto+passes | splitk+passes | MLX | decode ours | decode MLX |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.616 s (1,624 tok/s) | 0.951 s (1,051) | 1.314 s (761) | 0.567 s (1,764) | 127 - 145 | 163 - 198 |
| Qwen2.5-1.5B | 0.585 s (1,710 tok/s) | 0.912 s (1,096) | 1.209 s (827) | 0.591 s (1,692) | 143 - 156 | 186 - 211 |

- Outputs are correct text from the first token (no layout / RoPE / QK-norm / bias / tied-head bug); no model-code
  fix was needed.
- Every first divergence is a rounding tie: our top-1 - top-2 gap is 0 or 0.125 (= 1 bf16 ulp at |logit| 16..32) and
  MLX's own gap between the two tokens is 0 .. 0.25. After the split the two greedy paths continue on different text,
  so the prefix length says little past that point; teacher-forced agreement (our pick == MLX's token when both are
  fed MLX's history) stays >= 96.9 % in every run.
- GEMM prefill: 1.54x (Qwen3) / 1.56x (Qwen2.5) faster than flash passes on the 1,000-token prompt, now 0.92x / 1.01x
  of MLX's prefill speed. Token agreement is in the same tie-limited range as the pass prefill (better on Qwen3 chat,
  worse on the long prompts: different ties). Prompts < 16 tokens and the last prompt token always run as passes.
- Decode is 0.69 - 0.82x MLX per prompt (one waited command buffer per token, no encode / GPU overlap).
- Peak phys_footprint: 1.28 GB (Qwen3) / 1.10 GB (Qwen2.5) with GEMM prefill (its scratch adds ~19 MB) vs MLX's
  mx.get_peak_memory 1.76 / 1.63 GB.
- The MLX "gap" comes from its logprobs (logits - logsumexp, bf16 for Qwen3, f16 for Qwen2.5); mlx_lm's greedy
  sampler takes the argmax of those, ours takes the argmax of bf16-rounded logits (lowest index on ties).

Reproduce: `examples/qd_compare.rs` takes an MLX reference JSON (prompt ids, MLX's 128 greedy tokens, top-5 logprobs
per step, timings, made with mlx_lm `generate_step` on the same prompt ids; the reference scripts are not part of this
repository) and reports identical prefix, first divergence, teacher-forced agreement, timings and peak footprint:

```
cargo build --release --examples
./target/release/examples/qd_compare --model models/qwen3-1.7b-mlx4 --ref mlx_qwen3.json --attn auto --prefill gemm --out ours.json
```

## GEMM prefill: `src/prefill.rs`

Long prompts through the engine's 4-bit simdgroup-matrix GEMM (`engine/shaders/gemm_q4.metal`, read-only, dispatched
as `qwen_layers::encode_layer_prefill`: tiles 32x64 for T <= 32 else 64x64, 128 threads, groups (ceil(R/64),
ceil(T/BM)); swiglu groups (ceil(I/32), ceil(T/BM))) + `attn128_flash`, instead of decode passes of <= 8 rows.
Per chunk of T <= t_max rows (one command buffer each): embed; per layer rmsnorm_b (T rows) -> qkv GEMM m0 -> bias
(Qwen2) -> attn128_prep_b16 (T rows) -> attn128_flash (groups (ceil(T/8), nkv) x 32*grp) -> o_proj GEMM m2 (+=) ->
rmsnorm_b -> gate|up GEMM swiglu -> down GEMM m2 (+=). Head (last row) = the decode head (rmsnorm_b,
`qm_gemvt_k1_s8_m3`, argmax_b). Weight layout unchanged (the GEMM reads the same MLX layout as the gemv family).
Weights / KV stay in the engine arena; the T-row scratch is a separate buffer owned by `Prefill` (19 MB at Qwen3-1.7B
shapes, t_max 512; ~47 MB at 7B shapes, so use t_max 256 there for the <= 32 MB helper budget).

```rust
let mut pf = Prefill::new(&mut gpu, &cfg, 512)?;           // gpu library must hold PREFILL_SHADER_SOURCE's kernels
pf.run(&gpu, &PrefillModel { cfg, w, kv: [(k, v) per layer], max_ctx, arena: &arena.buf }, &tokens, pos, head)?;
pf.logits(); pf.hidden_row(t);                              // caller advances its own pos by tokens.len()
```

Not bit-exact with the decode-pass prefill (GEMM / flash summation order). `tests/prefill_gemm.rs` (synthetic tiny
Qwen3 QK-norm and Qwen2 bias models, `use qwen_dense::prefill`): T = 1, 7, 8, 9,
300, 513 (t_max 512: 513 = 2 chunks), T 300 at t_max 128 (3 chunks) and T 300 continued at pos 100 after decode passes.
KV caches of all layers / positions within 1.5 % of max |ref| (measured max diff 0.0156 - 0.047 at max |ref| 2.9 -
5.4, i.e. <= 1.5 bf16 ulp; 0 - 5.7 % of elements differ; T = 1 bit-identical), logits within 1.5 % of max |logit|
+ 0.01 (measured 0 - 0.015, tol 0.05 - 0.067), same argmax in all 16 cases; changing the last prompt token moves the
logits by 1.8 - 3.7 (the bar discriminates).

Bench (`cargo test --release --test prefill_gemm -- --ignored --nocapture bench`, ~6 s, GPU idle; Qwen3-1.7B shapes:
hidden 2048, inter 6144, 28 layers, 16/8 heads, vocab 32768 (head runs once in both paths), random weights, arena
953 MB, peak footprint 1067 MB), M4 Pro, min of 2:

| T | gemv passes of 8 | GEMM (t_max 512) | speed-up |
|---|---|---|---|
| 512 | 477 ms = 1073 tok/s | 295 ms = 1739 tok/s | 1.62x |
| 1024 | 973 ms = 1053 tok/s | 608 ms = 1685 tok/s | 1.60x |

At T = 512 the GEMM path does ~1.44 TFLOP of 4-bit matmul in 0.29 s (~5 TFLOP/s fp32 MMA; estimate, not profiled): likely compute-bound, so more
probably needs a faster GEMM kernel (e.g. half-precision MMA), not dispatch changes.

Wired into `Engine`: `EngineOptions.prefill: PrefillMode::{Gemm (default), Passes}`; qd_run / qd_compare
`--prefill gemm|passes`. With Gemm and a prompt of >= 16 tokens (`GEMM_MIN_TOKENS`), `Engine::prefill` runs
prompt[..n-1] through `Prefill::run` (built lazily, t_max `GEMM_T_MAX` = 512, own scratch buffer) and the last token as
a 1-row pass of the selected attention (auto: sdpa), as mlx_lm does. `gemm_q4.metal` is in `ENGINE_SHADER_SOURCE`;
`PREFILL_SHADER_SOURCE` = `ENGINE_SHADER_SOURCE`. Real-model numbers: section "Real-model test".

## Pipelined decode: `DecodeMode`

The decode gap in the real-model test (0.69 - 0.82x MLX) was put down to one waited command buffer per token: the CPU encodes
pass t+1 only after pass t has finished and its token was read back, so the GPU idles for the encode time each token.
`EngineOptions.decode` (qd_run / qd_compare `--decode pipelined|sync`) picks how `generate` runs tokens 2..:

- `Pipelined` (default): up to `PIPELINE_DEPTH` = 2 command buffers in flight. Pass t+1 is encoded and committed
  while pass t runs; its first dispatch, `qd_next_tok` (shaders/qd_misc.metal, 1 thread), copies the previous pass's
  greedy token `ctrl.tok[0]` (written by `argmax_b`) into the token row and sets `ctrl.pos`, which is exactly what the
  host writes before a `Sync` pass. Then the same embed / layers / head dispatches as `forward`. The host reads each
  token afterwards from the argmax record `rec[pos]` (argmax_b writes it per position, so the running look-ahead pass
  does not overwrite it), commits the next pass, then calls `on_token`. Ordering: one queue, one hazard-tracked arena
  buffer, so a command buffer sees every write of the previous one. Logits alternate between two f32 [vocab] slots so
  `logits()` stays the last accepted pass's. On a stop token the queued look-ahead pass is waited for and discarded
  (it wrote KV at `pos()`, which the next pass overwrites); no pass beyond `max_new` / `max_ctx` is encoded; context
  full returns the same error after the same tokens as `Sync`.
- `Sync`: the old `step` loop (commit, wait, read ctrl.tok, next).

`forward` / `prefill` / `step` stay synchronous. Same kernels, same order, same buffers: tokens and logits are
bit-identical to `Sync` (`tests/engine_synth.rs` `decode_pipelined_bit_identical`: both synthetic families, every
attention impl / prefill mode, 40 tokens across the 256-key chunk boundary, stop token + continued steps, max_new 1..3,
context full; a deliberately broken token chain fails it). Speed vs `Sync` on real models: not measured yet. Synthetic bench:
`cargo test --release --test prefill_gemm -- --ignored --nocapture bench_decode` (Qwen3-1.7B shapes, vocab 32768);
real models: `qd_run ... --decode sync` vs `--decode pipelined`.
