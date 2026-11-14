# dense-engine

Standalone engine for ordinary **dense decoder** models on Metal: **Qwen** dense (Qwen2 / Qwen2.5 / Qwen3, config.json
parsed here) and, through explicit `DenseConfig` fields filled by family config readers, the math of **Llama** (incl.
SmolLM2, Llama 3), **Granite**, **Gemma 1 / 2 / 3-text**, **Mistral**, **Phi-3 / Phi-4-mini**, **OLMo 2**, **SmolLM3**,
**GLM-4**, **StableLM 2**, **Cohere2** and **StarCoder2**: head dim 64 / 96 / 128 / 160 / 256, GQA or plain multi-head
(see "Other dense families" below). Own crate (empty `[workspace]` table in `Cargo.toml`), not a member of the engine
workspace: `cargo` in `engine/` never builds it. The model layer ([../model_layer](../model_layer/README.md),
`sd-model-layer`) is the user-facing entry point that routes these models here; this README is the engine's own
reference.

## What is here

| file | what |
|---|---|
| `src/config.rs` | `DenseConfig`: every math switch as an explicit field (defaults = Qwen; see "Other dense families"), `DenseConfig::base` for family readers, `rope_inv_freq` (plain / linear / llama3 tables). config.json reader for `model_type` qwen2 / qwen3 (also inside `text_config`, and the newer `rope_parameters` form). `validate()` / `unsupported()` name every unsupported feature. Hybrid configs (Qwen3.5 / 3.8: `linear_attention` layers, `linear_*` fields, `full_attention_interval`) are refused with "hybrid ... use the main engine"; MoE configs are refused. Unit tests embed Qwen2.5-0.5B / 7B / 14B and Qwen3-8B / 14B / 32B config shapes (values not checked against the published config are marked `unverified` in comments). |
| `shaders/attn_hd128.metal` | Port of `engine/shaders/attn_gated.metal`'s split-K pipeline as templates over the head dim HD = 64 / 96 / 128 / 160 / 256 (kernel names `attn{HD}_...`; the HD-128 names are the original `attn128_` ones): `attn128_rope_invf`, `attn{HD}_prep_b16` (+ `attn128_prep_q8`) (optional QK-norm, RoPE with any rdims, KV append, q8 per-32-dim quantization), `attn{HD}_partial_k{1..8}_b16` (+ `attn128_..._q8`) (256-key chunks, GQA), `attn{HD}_reduce` and `attn128_reduce_gate` (sigmoid output gate). Declares the per-pipeline function constants (scale [20], score soft cap [21], sliding window [22]; undefined = HD^-0.5, no cap, full causal) and helpers that the flash / sdpa files use, so it must come before them. Same buffer indices and grid shapes as the attn_gated kernels; prep has one extra uniform, `flags` [14] (bit 0 QK-norm, bit 1 gated q layout). |
| `src/gpu.rs` | Minimal Metal harness (same objc2 / objc2-metal versions as the engine): device, library from `SHADER_SOURCE` (the engine's `common_bf16.metal`, read via `include_str!`, + `attn_hd128.metal`, `attn_hd128_flash.metal`, `attn_hd128_sdpa.metal`), pipelines (key `name#s=..,c=..,w=..` sets function constants, `fc_key`), one arena buffer, encoder. Safe math mode, as the engine's `Gpu::from_shader_files`. |
| `tests/attn_hd128.rs` | GPU kernels vs a CPU f32 reference at Qwen dense shapes; K-invariance. |
| `src/weights.rs` | MLX 4-bit loader (see "Engine" below): `TensorSource` (`SafeSource` = model dir via streamdense-core `SafeIndex` + pread, `MemSource` = in-memory for tests), `layout()` + `load_into()`, `Weights` / `LayerW` / `Q4`. |
| `src/model.rs` | Kernel dispatch of one K-token pass (embed, layer, head; `Dims::new(cfg)` holds the math switches, `AttnVariant` the attention kernel names per head dim / scale / soft cap / window), MMA gemv dispatch rule (chunks of <= 8 rows), `AttnImpl` switch (prep + core: auto / splitk / flash / sdpa), `SDPA128_TWO_PASS_MIN`, `sdpa128_vec_max_n`. |
| `src/engine.rs` | `Engine` (load, `forward`, `prefill` (GEMM or passes), `step`, `generate`, `logits`), `PrefillMode`, `GenStats`, `peak_footprint()`. |
| `src/prefill.rs` | GEMM prefill (q4b_gemm + attn128_flash over up to 512 rows per chunk); see its section. |
| `shaders/qd_misc.metal` | `qd_embed_q4` (token ids -> bf16 rows from the 4-bit embedding, MLX dequant rounding), `qd_bias_add`, `qd_next_tok`, `qd_scale_b16` (embedding multiplier), `qd_residual` (h + r * m), `qd_glu` (SiLU / tanh-GELU gate times up), `qd_logits_post` (logits scaling, final soft cap). |
| `examples/qd_run.rs` | CLI: greedy generation, timings, peak footprint, optional JSON. |
| `examples/qd_compare.rs` | Real-model check: runs the exact prompt ids of an MLX reference JSON (free run + teacher forced), reports identical prefix, first divergence and logit gaps, timings, peak footprint. |
| `tests/engine_synth.rs` | Whole engine on synthetic tiny Qwen3 / Qwen2 models and eight other-family models vs a CPU reference, every attention impl; per-feature ablations; K-invariance per impl; pipelined == sync; sdpa long-row fallback; tokenizers; weight layout (real models from headers only, family models from the synthetic source). |
| `tests/families2_synth.rs` | The same checks for the second batch of family features on synthetic models: Phi-3 (fused qkv / gate_up, LongRoPE, hd 96, partial rotary), OLMo 2 (post-norm, full-projection QK-norm), SmolLM3 (no-RoPE layers), GLM-4 (GLM norms, interleaved partial RoPE), StableLM (LayerNorm, parallel / sequential, per-head LayerNorm QK-norm, hd 160), Cohere2 (parallel block, sliding + no-RoPE layers, logit scale), StarCoder2 (non-gated exact-erf GELU MLP, biases); weight layout; config validation. |

Dispatch shape (same as `qwen_layers::encode_attn` for attn_gated; shown for HD 128, `attn{HD}_` with HD threads for prep and HD/4 for reduce at the other head dims):

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
- YaRN, dynamic or any `rope_scaling` other than linear / llama3 / LongRoPE (su) (reported as unsupported); Qwen's own
  sliding window (`use_sliding_window`, refused for the Qwen families); partial rotary for Qwen (mlx_lm rotates the full
  head dim there). LongRoPE past 4096 tokens is not exercised on a real model.
- The split-K path is accurate to the tolerances below but not bit-exact vs MLX (the `sdpa` path is; see the
  next section).
- Head dims other than 64 / 96 / 128 / 160 / 256 (160 is synthetic-tested only, no real model run); exact-erf GELU from
  `hidden_act` "gelu" (`GeluErf` is set only where mlx_lm calls `nn.gelu`, StarCoder2). Config readers, tokenizers and
  chat templates of the other families live in the model layer (`engine/model_layer/src/<family>`), not here.

## Run the tests

```
cd engine/dense-engine
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

Runs MLX 4-bit Qwen2 / 2.5 / 3 dense models end to end on Metal (real models tested: head dim 128; other families: section "Other dense families" below): greedy only, bf16 KV cache, one
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

API (`dense_engine::`):

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

optional steps (other dense families; none of them is dispatched for a Qwen config):
  after qd_embed_q4           qd_scale_b16 (embed_scale)
  attention kernels           attn{HD}_* of the head dim; pipeline function constants scale / soft cap / window
                              (sliding layers: own pipelines, own RoPE table when rope_local_theta is set)
  o_proj / down               unfused when bias, post-norm or residual multiplier: gemv MODE 0 -> io.r, qd_bias_add,
                              rmsnorm_b (post_attention / post_feedforward norm) -> io.x, qd_residual (h += r * m)
  FFN                         unfused when GeGLU or MLP bias: gate gemv MODE 0 -> f, up gemv MODE 0 -> f2, biases, qd_glu
  head                        qd_logits_post (/ logits_scaling, final soft cap) before argmax_b
```

- Matmuls: engine `gemv_mma_q4.metal` (include_str!, unmodified), same rule as `qwen_layers::gemv` with
  `GemvImpl::Mma`: twins `qm_gemvt_k{K}_s8_m*` / `qm_swiglut_k{K}_s8` where `mma_use_twin` (K <= 2 always; K 3..4 by
  shape), else `qm_gemv_n2_s8_m*` / `qm_swiglu_n1_s8`. Bit-identical per row, so prefill == token-by-token decode.
- Swapping attention: add an `AttnImpl` variant in src/model.rs, list its kernels in `AttnVariant::pipelines`, add a
  match arm in `encode_attn_core` (reads `AttnIo`: qr [k][nh][hd], bf16 Kc / Vc [nkv][max_ctx][hd], p, ao, ctrl,
  max_ctx, pos, k, the variant `v`), add its .metal file to `ENGINE_SHADER_SOURCE` (src/lib.rs) and a name to `AttnImpl::parse`
  (qd_run `--attn`). The prep (`encode_attn_prep`) is shared. Flash needs 16-byte-aligned KV (the arena aligns to 256).
- Weights (`weights::layout`): q/k/v weight, scales, biases and bias concatenated row-wise; F16 / F32 float tensors
  converted to bf16 at load (RNE); lm_head = embed when tied (else `lm_head.*`). Group size must be 64, bits 4.
  Optional by config: `self_attn.o_proj.bias`, `mlp.{gate,up,down}_proj.bias`, `pre_feedforward_layernorm` /
  `post_feedforward_layernorm` (four_norms); with `norm_plus_one` every norm weight is stored as bf16(1 + w).

Tests (`tests/engine_synth.rs`, Qwen part ~2 s of ~20 s, no model weights read): synthetic 2-layer Qwen3-style (QK-norm, GQA 4/2, bf16)
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

## Other dense families: config-driven math

The engine runs the math of other dense decoder families when a config reader fills these `DenseConfig` fields
(start from `DenseConfig::base(model_type, family, hidden, inter, layers, heads, kv_heads, head_dim, vocab)` or a
parsed config; `DenseConfig::from_json` stays the qwen2 / qwen3 reader). Every default is the Qwen behaviour, and with
the defaults the engine dispatches exactly the kernels it did before (Qwen logits bit-identical to the previous code:
prefill + decode + pipelined generate, every attention impl / prefill mode, checked once against the previous commit).

| field | values | used by |
|---|---|---|
| `family` | `Qwen2`, `Qwen3`, `Llama`, `Granite`, `Gemma`, `Gemma2`, `Gemma3`, `Mistral`, `Phi3`, `Olmo2`, `SmolLM3`, `Glm4`, `StableLm`, `Cohere2`, `Starcoder2` | informational; Qwen sliding window refused |
| `head_dim`, `rope_dims` | 64 / 96 / 128 / 160 / 256; `rope_dims` even, <= `head_dim` (partial rotary; Qwen: equal) | all; partial: Phi-4-mini, GLM-4, StableLM |
| `heads`, `kv_heads` | GQA, MQA or plain MHA; group <= 32 | all |
| `attn_scale` | `None` = head_dim^-0.5 | Granite `attention_multiplier`, Gemma 2 / 3 `query_pre_attn_scalar`^-0.5 |
| `attn_softcap`, `final_softcap` | `Some(cap)`: cap * tanh(x / cap) | Gemma 2 |
| `activation` | `Silu` (SwiGLU), `GeluTanh` (GeGLU); `Activation::from_hf` | Gemma `hidden_activation` gelu_pytorch_tanh |
| `norm_plus_one` | RMSNorm weight (1 + w), incl. q_norm / k_norm (folded at load, bf16) | Gemma 1 / 2 / 3 |
| `four_norms` | input, post_attention (on the attention output), pre_feedforward, post_feedforward | Gemma 2 / 3 |
| `qk_norm` | per-head RMSNorm on q / k before RoPE | Qwen3, Gemma 3 |
| `embed_scale` | embeddings * m (bf16) | Gemma sqrt(hidden), Granite `embedding_multiplier` |
| `residual_multiplier` | h = h + r * m (bf16) | Granite |
| `logits_scaling` | logits / s | Granite |
| `kinds`, `sliding_window` | per layer `Full` / `Sliding`; key j visible to query i iff i - j < window | Gemma 2 (even layers), Gemma 3 (5 of 6) |
| `rope_theta`, `rope_scale` | `RopeScaling::None` / `Linear { factor }` / `Llama3 { factor, low_freq_factor, high_freq_factor, original_max_pos }` | Llama 3.x (llama3), Gemma 3 global layers (linear) |
| `rope_local_theta` | sliding layers' plain RoPE theta | Gemma 3 `rope_local_base_freq` |
| `attention_bias`, `o_bias`, `mlp_bias` | q/k/v bias / o_proj bias / gate, up, down bias | Qwen2 (q/k/v); Llama `attention_bias` (q/k/v + o), `mlp_bias` |
| `tie_word_embeddings` | lm_head = embed_tokens, else `lm_head.*` | all |
| `norm` | `Rms` / `LayerNorm { bias }` | StableLM, StarCoder2 (bias), Cohere2 (no bias) |
| `block` | `Sequential` / `Parallel` / `ParallelBranchesFirst` / `PostNorm` | StableLM parallel residual, Cohere2, OLMo 2 |
| `qk_norm_kind` | `HeadRms` / `FullRms` (whole projection) / `HeadLayerNorm` (weight per head) | Qwen3, Gemma 3 / OLMo 2 / StableLM |
| `mlp` | `Gated` / `Plain` (down(act(up x))) | StarCoder2 (exact-erf GELU) |
| `rope_traditional` | interleaved RoPE pairs | GLM-4, Cohere2 |
| `no_rope_layers` | per layer: no RoPE | SmolLM3, Cohere2 full layers |
| `rope_su` | LongRoPE / su as mlx_lm (long factors, q / k mscale) | Phi-3 / 3.5 / 4-mini |
| `names` | tensor names, fused `qkv` / `gate_up` | Phi-3, GLM-4, StarCoder2, Cohere2 |

How (kernels): the attention kernels are templates over the head dim (`attn{64,128,256}_prep_b16`, `_partial_k1..8_b16`,
`_reduce`, `_flash`, `_sdpa_vec`); scale, score soft cap and window are Metal function constants set per pipeline
(`Gpu::pipeline` key `name#s=..,c=..,w=..`), undefined = compiled out. A sliding layer gets its own pipelines; the
KV cache stays full length (no ring buffer), the kernels skip / mask keys outside the window. The flash kernel skips
key blocks before a tile's first visible key and keeps a fully masked row at m = -inf, l = 0. Biases, post-norms,
the residual multiplier, GeGLU and MLP biases use unfused steps (`qd_bias_add`, `rmsnorm_b`, `qd_residual`, `qd_glu`);
logits scaling / final soft cap run in `qd_logits_post` before the argmax, so `Engine::logits()` returns the
post-processed values. RoPE tables: plain ones from `attn128_rope_invf` as before, scaled ones from
`config::rope_inv_freq` on the host (mlx_lm Llama3RoPE / linear). The GEMM prefill does the same steps.

Status and tests (synthetic 2-layer models, random MLX 4-bit weights, no real model weights read; `tests/engine_synth.rs`
`families_vs_cpu_reference`: CPU reference, logits after a 300-token prefill + 3 decode steps for every attention impl
and prefill mode, 24 comparisons per model, tolerance 1.5 % of max |logit| + 0.01 as for Qwen; each feature also
checked by an ablated reference with that one feature off, which must miss the tolerance):

| feature | status | synthetic model(s) | max diff (worst diff / tol) | ablation miss (diff vs tol) |
|---|---|---|---|---|
| head dim 64, GQA | done | hd64 gqa 4/2 (qk-norm) | 0.023 (0.42) | qk-norm 0.70 vs 0.06 |
| head dim 64, plain MHA | done | hd64 mha 4/4 | 0.021 (0.37) | — |
| head dim 256, MQA | done | hd256 mqa 2/1, gemma3-style 2/1 | 0.016 / 0.028 (0.07 / 0.19) | — |
| attention scale | done | gemma2-style (query_pre_attn_scalar 100), granite (0.05) | 0.015 / 0.008 | 0.077 vs 0.040; 0.27 vs 0.08 |
| GeGLU (tanh GELU) | done | hd256 mqa, gemma2-, gemma3-style | see above | 0.28 vs 0.22; 0.28 vs 0.04; 0.32 vs 0.14 |
| (1 + w) RMSNorm incl. QK-norm | done | hd256 mqa, gemma2-, gemma3-style | see above | 13.8 / 2.1 / 9.0 vs <= 0.02 |
| four norms per layer | done | gemma2-, gemma3-style | see above | 2.1 vs 0.04; 4.5 vs 0.21 |
| embedding scale | done | hd256 mqa (16), gemma2/3 (16), granite (4) | see above | 13.4 / 2.5 / 7.2 / 3.2 |
| residual multiplier | done | granite (0.5) | 0.008 (0.10) | 1.74 vs 0.05 |
| logits scaling | done | granite (2) | see above | 4.48 vs 0.14 |
| tied / untied lm_head | done | all tied except hd64 mha (untied) | see above | — |
| attention soft cap | done | gemma2-style (cap 1.5) | 0.015 (0.37) | 0.82 vs 0.04 |
| final logit soft cap | done | gemma2-style (cap 2) | see above | 6.74 vs 0.14 |
| sliding window per layer | done | gemma2-style (window 37, layer 0), gemma3-style (window 50, layer 0) | see above | 1.89 vs 0.04; 2.50 vs 0.13 |
| local RoPE theta (sliding layers) | done | gemma3-style (10000 local, 1e6 global) | see above | 1.07 vs 0.14 |
| RoPE scaling linear | done | gemma3-style global layer (factor 4) | see above | 0.49 vs 0.14 |
| RoPE scaling llama3 | done | llama3 rope (factor 8, low 1, high 4, original 256) | 0.018 (0.26) | 1.37 vs 0.06 |
| q/k/v + o_proj bias | done | hd64 mha, llama biases | 0.021 / 0.023 | q/k/v 3.0 vs 0.05, o 3.4 vs 0.07 (hd64); 3.0 / 2.3 (hd128) |
| MLP bias (unfused SiLU) | done | llama biases | 0.023 (0.29) | 1.50 vs 0.06 |
| YaRN / dynamic RoPE | refused, named by `unsupported()` | — | — | — |
| partial rotary, LongRoPE, fused tensors, LayerNorm, parallel / post-norm blocks, no-RoPE layers, interleaved RoPE, non-gated MLP, hd 96 / 160 | done | `tests/families2_synth.rs` (same method, own ablations) | asserted within tolerance | asserted to miss |
| exact-erf GELU from `hidden_act` "gelu" | refused, named (`GeluErf` only for StarCoder2, as mlx_lm) | — | — | — |

All argmax tokens equal the reference's in every comparison. `families_k_invariant_and_pipelined`: for hd64 gqa,
gemma2-style (hd 128, window, soft caps) and gemma3-style (hd 256, window, two RoPE tables), prefill in passes of 8 ==
1 == mixed (split-K, sdpa; flash also 128 / 100+37+128) bit for bit, and pipelined decode == sync over 30 tokens.
`families_weight_layout`: every tensor of each synthetic model is used exactly once; a missing pre / post-feedforward
norm, o_proj / down bias, k_norm or lm_head tensor is named in the error. `config::tests::{family_fields_validate,
rope_scalings}`: field validation and RoPE tables. Real checkpoints: see "Real models of the other families" below.
Not tested: MLX bit-exactness of the new kernels (the head-dim-64 / 256 sdpa mirrors MLX's
sdpa_vector structure but is only checked against the CPU reference), speed of the new variants.

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

## Real models of the other families (2026-11-14)

Run through the model layer (`sd-model-layer run`, default engine settings: GEMM prefill for prompts >= 16 tokens,
pipelined decode), greedy, 32 new tokens, same prompt ids as mlx_lm greedy. Verified models (MLX 4-bit, group 64):

| model | family / features exercised | short prompt (10-11 tokens) | ~690-token prompt |
|---|---|---|---|
| SmolLM-135M-Instruct | Llama, hd 64, GQA 9/3, tied lm_head | 32 / 32 identical | 32 / 32 |
| Llama-3.2-1B-Instruct | Llama, hd 64, llama3 RoPE scaling, tied | 32 / 32 | 3 / 32, divergence a tie (MLX gap 0.047 at \|logit\| ~19) |
| granite-4.2-3b | Granite, hd 64, attention_multiplier 1/64, untied | 32 / 32 | 32 / 32 |
| gemma-3-270m-it | Gemma 3 text, hd 256, (1 + w) norms, four norms, QK-norm, GeGLU, sliding 512 + two RoPE tables | 32 / 32 | 15 / 32, divergence 1 bf16 step (MLX gap 0.125 at \|logit\| ~20) |
| gemma-1.1-2b-it | Gemma 1, F16 embedding, tanh GELU (per config `gelu_pytorch_tanh`) | chat 9 / 9 to EOS | 685 tokens: 32 / 32 |
| gemma-2-2b-it | Gemma 2, soft caps, sliding 4096 | 32 / 32 (teacher-forced 32 / 32) | 685 tokens: 32 / 32 (teacher-forced 32 / 32); past 4096 not verified |

No engine change was needed. Speeds, footprints and the full comparison: the
[model layer README](../model_layer/README.md#llama-granite-gemma-vs-mlx-2026-11-14). Gemma 2 past its 4096-token
window (2026-11-15): 16 / 16 at 4331 prompt tokens vs a correctly windowed reference (mlx_lm with per-layer sliding
masks). Gemma 3 past 512: the misses are bf16 near-ties (layer-by-layer check, no bug found).

Second batch (2026-11-15, same method, short / ~690-707-token prompt, identical prefix out of 32):

| model | family / features exercised | short prompt | long prompt |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | Mistral (Llama math), hd 128 | 32 / 32 | 32 / 32 |
| Phi-4-mini-instruct | Phi-3, fused tensors, partial rotary 0.75, LongRoPE, tied | 6 / 32, near-tie (gap 0.125; teacher-forced 29 / 32) | 32 / 32 |
| Phi-3.5-mini-instruct | Phi-3, hd 96, LongRoPE | 32 / 32 | 32 / 32 |
| OLMo-2-1124-7B-Instruct | OLMo 2, post-norm, full-projection QK-norm | 32 / 32 | 30 / 32, near-tie (gap 0.023; teacher-forced 32 / 32) |
| SmolLM3-3B | SmolLM3, no-RoPE layers | 32 / 32 | 32 / 32 |
| GLM-4-9B-0414 | GLM-4, four norms, fused gate_up, interleaved partial RoPE | 32 / 32 | 32 / 32 |
| stablelm-2-1_6b-chat | StableLM, LayerNorm, per-head LayerNorm QK-norm, partial RoPE | 13 / 32 (gap 0.031) | 30 / 32 (gap 0.109) |
| c4ai-command-r7b-12-2024 | Cohere2, parallel block, sliding + no-RoPE layers, logit scale | 20 / 20 to EOS (chat) | 32 / 32 |
| starcoder2-3b | StarCoder2, non-gated exact-erf GELU MLP, biases | 32 / 32 | 32 / 32 |

StableLM: mlx computes its F16 norms in f16, this engine in bf16 (teacher-forced 30 / 32, 31 / 32). Head dim 160 has
no real-model run. Speeds and footprints: the
[model layer README](../model_layer/README.md#more-families-vs-mlx-2026-11-15).

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
Qwen3 QK-norm and Qwen2 bias models, `use dense_engine::prefill`): T = 1, 7, 8, 9,
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
