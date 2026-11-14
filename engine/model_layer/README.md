# sd-model-layer: run dense models through one command

`sd-model-layer` is one entry point in front of the streamdense engines. Point it at a model folder (an MLX 4-bit
conversion with `config.json`, safetensors and `tokenizer.json`): it reads `config.json`, picks the model family and
the engine that runs it, and either reports what applies to that model (`inspect`) or generates text (`run`). It holds
no inference code of its own; it calls the existing engines.

The gateway (`streamdense up` / `streamdense gateway`: Anthropic Messages API, MTP speculative decoding, the 12 GB mode)
is built for Qwen3.8-27B. Every other supported model runs here, as plain greedy decoding. Besides Qwen, the Llama,
Granite, Gemma, Mistral, Phi-3, OLMo 2, SmolLM3, GLM-4, StableLM, Cohere2 and StarCoder2 families run on
[`engine/dense-engine`](../dense-engine/README.md): each family's folder in `src/` maps its `config.json` to the
engine's `DenseConfig` and renders its chat template.

## Supported families

Only families that were run on real models are listed. Anything else is refused by name before any weight is read.

| Family | `model_type` in config.json | Examples | Engine | Tested on real models |
|---|---|---|---|---|
| Qwen2 / Qwen2.5 dense | `qwen2` | Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct | [`engine/dense-engine`](../dense-engine/README.md) | Qwen2.5-1.5B-Instruct vs MLX (below) |
| Qwen3 dense | `qwen3` | Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, Qwen/Qwen3-14B | [`engine/dense-engine`](../dense-engine/README.md) | Qwen3-1.7B and Qwen3-0.6B vs MLX (below) |
| Qwen3.5 / Qwen3.8 hybrid | `qwen3_5`, `qwen3_5_text` | Qwen3.8-27B (MLX 4-bit) | main engine (`engine/crates/streamdense-model`) | Qwen3.8-27B (the gateway's model); Qwen3.5-0.8B (tied embeddings, no MTP weights) vs MLX through this layer (below) |
| Llama (Llama 3.x, SmolLM) | `llama` | meta-llama/Llama-3.2-1B-Instruct, HuggingFaceTB/SmolLM-135M-Instruct | [`engine/dense-engine`](../dense-engine/README.md) | Llama-3.2-1B-Instruct and SmolLM-135M-Instruct vs MLX (below) |
| Granite (dense) | `granite` | ibm-granite Granite 4.x 3B dense | [`engine/dense-engine`](../dense-engine/README.md) | granite-4.2-3b (MLX 4-bit) vs MLX (below) |
| Gemma 1 | `gemma` | google/gemma-1.1-2b-it | [`engine/dense-engine`](../dense-engine/README.md) | gemma-1.1-2b-it vs MLX (below) |
| Gemma 2 | `gemma2` | google/gemma-2-2b-it | [`engine/dense-engine`](../dense-engine/README.md) | gemma-2-2b-it vs MLX (below) |
| Gemma 3 (text) | `gemma3_text` | google/gemma-3-270m-it | [`engine/dense-engine`](../dense-engine/README.md) | gemma-3-270m-it vs MLX (below) |
| Mistral | `mistral` | mistralai/Mistral-7B-Instruct-v0.3, Mistral-Nemo-Instruct-2407, Ministral-8B-Instruct-2410 | [`engine/dense-engine`](../dense-engine/README.md) | Mistral-7B-Instruct-v0.3 vs MLX (below) |
| Phi-3 / Phi-3.5 / Phi-4-mini | `phi3` | microsoft/Phi-4-mini-instruct, microsoft/Phi-3.5-mini-instruct | [`engine/dense-engine`](../dense-engine/README.md) | Phi-4-mini-instruct and Phi-3.5-mini-instruct (hd 96, LongRoPE) vs MLX (below) |
| OLMo 2 | `olmo2` | allenai/OLMo-2-1124-7B-Instruct | [`engine/dense-engine`](../dense-engine/README.md) | OLMo-2-1124-7B-Instruct vs MLX (below) |
| SmolLM3 | `smollm3` | HuggingFaceTB/SmolLM3-3B | [`engine/dense-engine`](../dense-engine/README.md) | SmolLM3-3B vs MLX (below) |
| GLM-4 (0414) | `glm4` | THUDM/GLM-4-9B-0414 | [`engine/dense-engine`](../dense-engine/README.md) | GLM-4-9B-0414 vs MLX (below) |
| StableLM 2 | `stablelm` | stabilityai/stablelm-2-1_6b-chat | [`engine/dense-engine`](../dense-engine/README.md) | stablelm-2-1_6b-chat vs MLX (below) |
| Command R7B | `cohere2` | CohereForAI/c4ai-command-r7b-12-2024 | [`engine/dense-engine`](../dense-engine/README.md) | c4ai-command-r7b-12-2024 vs MLX (below) |
| StarCoder2 (base) | `starcoder2` | bigcode/starcoder2-3b | [`engine/dense-engine`](../dense-engine/README.md) | starcoder2-3b vs MLX (below), `--prompt` only |

Gemma 1 and Gemma 2 are verified on real models (2026-11-14, below): gemma-1.1-2b-it (F16 embedding) chat 9 / 9
tokens to EOS, 685-token prompt 32 / 32; gemma-2-2b-it short and 685-token prompts 32 / 32 (teacher-forced 32 / 32
both), chat string equal to HF. Gemma 2 past its 4096-token window (2026-11-15): 16 / 16 at 4331 prompt tokens
against a correctly windowed reference (mlx_lm with per-layer sliding masks; mlx_lm 0.32.2's own `gemma2.py` has no
sliding window and gives 1 / 16 there, so it is not a valid reference). A multimodal `gemma3` folder whose weights are the text model only is
routed but was not run on a real checkpoint. Gemma 1 GELU: this layer follows the model config (`hidden_activation`
`gelu_pytorch_tanh`, tanh GELU, as transformers); mlx_lm's `gemma.py` uses exact-erf GELU, yet the greedy tokens
matched on gemma-1.1-2b-it.

Qwen dense models take head dim 64 or 128 (Qwen2.5-0.5B has head dim 64 and runs: 2026-11-14 short prompt 32 / 32 vs
MLX, long prompt first divergence at a near-tie, gap 0.047). The other families take head dim 64, 96, 128, 160 or 256
(96: Phi-3 / Phi-3.5-mini, checked on a real model; 160: e.g. stablelm-2-12b, synthetic-tested only). Weights must be
MLX 4-bit, group size 64.

Verification: `engine/dense-engine` was checked against MLX (mlx 0.32.4 / mlx_lm 0.32.2) on Qwen/Qwen3-1.7B and
Qwen/Qwen2.5-1.5B-Instruct, 6 prompts x 128 greedy tokens: correct text from the first token, teacher-forced agreement
with MLX >= 96.9 % on every prompt, every first divergence a bf16 rounding tie. Details: the
[dense-engine README](../dense-engine/README.md#real-model-test-2026-11-02). The hybrid path is the main engine that runs
Qwen3.8-27B; this layer runs it with plain greedy decode.

### Llama, Granite, Gemma vs MLX (2026-11-14)

MLX reference: mlx_lm (same environment as above) greedy, temperature 0, the same prompt token ids on both sides
(checked equal), no chat template, prompt in one call then token-by-token decode. Ours: `sd-model-layer run --prompt`
(default engine settings), 32 new tokens. Two prompts: a short one ("Water boils at a lower temperature at high
altitude because", 10-11 tokens, prefill in passes) and a ~690-token one (a repeated paragraph, GEMM prefill; for
Gemma 3 it is longer than the 512-token sliding window). M4 Pro, one process at a time.

| model (MLX 4-bit, group 64) | short prompt: identical prefix | ~690-token prompt: identical prefix | first divergence | chat string == `apply_chat_template` |
|---|---|---|---|---|
| SmolLM-135M-Instruct (llama, hd 64, GQA 9/3, tied) | 32 / 32 | 32 / 32 | none | equal |
| Llama-3.2-1B-Instruct (llama, hd 64, llama3 RoPE) | 32 / 32 | 3 / 32 | pos 3: MLX " deltas" vs ours " fertile", MLX's own logit gap 0.047 (top1-top2 0.0625), \|logit\| ~19: a tie below one bf16 step | equal (date line = today, as transformers) |
| granite-4.2-3b (granite, attention_multiplier 1/64) | 32 / 32 | 32 / 32 | none | equal (thinking on, the template default) |
| gemma-3-270m-it (gemma3_text, hd 256, sliding 512) | 32 / 32 | 15 / 32 | pos 15: MLX " accumulation" vs ours " flow", MLX gap 0.125 = 1 bf16 step at \|logit\| 16..32 | equal |

Short-prompt runs of `sd-model-layer` (32 tokens; prefill of 10-11 tokens, decode tok/s over tokens 2..32; peak
phys_footprint of the process):

| model | prefill | decode | peak footprint |
|---|---|---|---|
| SmolLM-135M-Instruct | 0.021 s | 463 tok/s | 0.19 GB |
| Llama-3.2-1B-Instruct | 0.042 s | 221 tok/s | 0.92 GB |
| granite-4.2-3b | 0.055 s | 89 tok/s | 2.22 GB |
| gemma-3-270m-it | 0.025 s | 307 tok/s | 0.47 GB |

Further runs the same day (same method; "chat" = `run --chat` vs mlx_lm on the same chat prompt ids, "to EOS" =
the whole answer up to the end-of-turn token):

| model (MLX 4-bit, group 64) | result |
|---|---|
| Qwen2.5-1.5B-Instruct | short 32 / 32, long 32 / 32; chat string == HF; chat 12 / 12 to EOS |
| Qwen2.5-0.5B-Instruct (hd 64) | short 32 / 32; long diverges at a near-tie (gap 0.047) |
| Qwen3-0.6B | short 32 / 32, long 32 / 32 |
| Qwen3-1.7B | chat string == HF; chat 26 / 32 (near-tie, gap 0.25; teacher-forced 31 / 32) |
| Qwen3.5-0.8B (hybrid, main engine, tied embeddings, no MTP weights) | short 32 / 32, 640-token 32 / 32, `--chat` 32 / 32, chat string == HF; 177-199 tok/s, 0.69 GB |
| Llama-3.2-1B-Instruct | chat 18 / 18 to EOS; long 32 / 32 |
| SmolLM-135M-Instruct | divergences only at rounding ties (gap <= 0.125; teacher-forced 30-31 / 32) |
| granite-4.2-3b | short 32 / 32, long 32 / 32 |
| gemma-1.1-2b-it (gemma, F16 embedding, tanh GELU per config) | chat 9 / 9 to EOS; 685-token 32 / 32; 66-73 tok/s, 3.49 GB |
| gemma-2-2b-it (gemma2) | short 32 / 32, 685-token 32 / 32, teacher-forced 32 / 32 both; chat string == HF; 95-98 tok/s, 1.87 GB |
| gemma-3-270m-it (gemma3_text) | up to ~500 tokens 31-32 / 32; past the 512 window, see below |

Gemma 3 past its 512-token window, against mlx_lm's no-cache windowed one-shot reference (mlx_lm's cached greedy is
not a valid reference there): prompts of 490 / 600 / 650 tokens 32 / 32, 500 / 530 tokens 31 / 32, 686 tokens 29 / 32
(1 near-tie, 2 misses of ~1.1 logit). A layer-by-layer check (2026-11-15) found no bug: the misses are bf16 near-ties.

~690-token prompt prefill: SmolLM 0.048 s, Llama-3.2-1B 0.298 s (2,315 tok/s), granite 1.002 s (688 tok/s), Gemma 3
0.081 s. Tokenizers: the four `tokenizer.json` files encode 10-11 test strings (digits, Unicode, control bytes, special
tokens) to the same ids as HF `tokenizers`, with and without the post-processor BOS.

### More families vs MLX (2026-11-15)

Same method as above: greedy, 32 new tokens, quantized mlx_lm 0.32.2 on identical prompt ids; a short prompt and a
long one (~690-707 tokens). "chat = HF": the `--chat` string equals `apply_chat_template`. Where a run diverges, the
gap is mlx_lm's own top-1 - top-2 logit gap at the first divergence, and "teacher-forced" is our pick == MLX's token
when both are fed MLX's history. tok/s: decode, short / long prompt; peak RAM: peak phys_footprint. M4 Pro, one
process at a time.

| family | model (MLX 4-bit, group 64) | short | long (~690-707 tokens) | notes | tok/s | peak RAM |
|---|---|---|---|---|---|---|
| mistral | Mistral-7B-Instruct-v0.3-4bit | 32 / 32 | 32 / 32 | chat = HF | 51 / 50 | 4.32 GB |
| phi3 | Phi-4-mini-instruct-4bit | 6 / 32 (near-tie, gap 0.125; teacher-forced 29 / 32, every miss is MLX's #2) | 32 / 32 | chat = HF | 81 / 77 | 2.41 GB |
| phi3 | Phi-3.5-mini-instruct-4bit (hd 96, LongRoPE) | 32 / 32 | 32 / 32 | chat = HF; prompts > 4096 tokens not exercised | 86 / 79 | 2.59 GB |
| olmo2 | OLMo-2-1124-7B-Instruct-4bit | 32 / 32 | 30 / 32 (near-tie, gap 0.023; teacher-forced 32 / 32) | chat = HF; no MLX 4-bit 1B exists | 50 / 47 | 4.67 GB |
| smollm3 | SmolLM3-3B-4bit | 32 / 32 | 32 / 32 | chat = HF (local date) | 85 / 79 | 2.01 GB |
| glm4 | GLM-4-9B-0414-4bit | 32 / 32 | 32 / 32 | chat = HF | 42 / 40 | 5.54 GB |
| stablelm | stablelm-2-1_6b-chat-4bit | 13 / 32 (gap 0.031) | 30 / 32 (gap 0.109) | chat = HF; F16 norms computed in f16 by mlx, in bf16 here; teacher-forced 30 / 32, 31 / 32 | 169 / 155 | 1.90 GB |
| cohere2 | c4ai-command-r7b-12-2024-4bit | 20 / 20 to EOS | 32 / 32 | chat text = HF; BOS added as mlx_lm's Python `generate()` does (the CLI and HF add none) | 43 / 45 | 4.82 GB |
| starcoder2 | starcoder2-3b-4bit (base, `--prompt` only) | 32 / 32 | 32 / 32 | no chat template | 87 / 83 | 2.40 GB |

Head dims now 64 / 96 / 128 / 160 / 256; head dim 160 (e.g. stablelm-2-12b) is synthetic-tested only.

## Known limits

- Greedy decoding only: no sampling, temperature or top-p.
- Weights: MLX 4-bit, group size 64 only.
- `--prompt`: the text as given, no chat template. Qwen: no BOS (as before). Llama / Granite / Gemma: the tokenizer's
  own BOS when it adds one (Llama 3 `<|begin_of_text|>`, Gemma `<bos>`; SmolLM and Granite add none), as HF / mlx_lm.
- `--chat`: one user message in the family's chat template with the generation prompt, template defaults. Qwen
  (`src/qwen/`): kind read from the folder's template; Qwen2.5 ChatML with its default system message, Qwen3 dense
  thinking on (nothing after `<|im_start|>assistant\n`), Qwen3.8 via the gateway's renderer (thinking on, effort xhigh,
  `<think>\n`). No system message, no history, no tools from the command line.
- Mixture-of-experts models (`qwen2_moe`, `qwen3_moe`, any config with expert keys) are rejected: dense models only.
- `rope_scaling` other than linear / llama3 (YaRN, dynamic) is refused by name; Phi-3 LongRoPE / su runs as mlx_lm
  does (always the long factors), prompts over 4096 tokens are untested. Qwen's own sliding window is refused (Gemma 2 /
  3, Mistral `layer_types` and Cohere2 sliding layers are supported).
- Head dim 160 (e.g. stablelm-2-12b) is synthetic-tested only, not on a real model.
- Per family:
  - Llama: chat templates: Llama 3
    header, Llama 3.1 / 3.2 dated system block (today's local date, as transformers' `strftime_now`), ChatML
    (SmolLM / SmolLM2). Llama 2 / Mistral `[INST]`, Zephyr `<|user|>` (TinyLlama-Chat), ChatML with tools and
    folders without a template are refused for `--chat` by name (`--prompt` still works).
  - Granite: dense `granite` only. `granitemoe`, `granitemoeshared` and `granitemoehybrid` with experts: "dense models
    only"; `granitemoehybrid` with Mamba layers: "layer type mamba not supported yet"; attention-only
    `granitemoehybrid` (FFN under `shared_mlp`): "not supported yet"; vision / speech wrappers (`text_config`):
    "not supported yet". The multipliers (`attention_multiplier`, `embedding_multiplier`, `residual_multiplier`,
    `logits_scaling`) must be in config.json (nothing guessed). `--chat` renders the Granite 4.x ChatML template with
    thinking on (its default); older `<|start_of_role|>` templates are refused.
  - Gemma: exact-erf GELU (`hidden_activation` "gelu") refused; a multimodal `gemma3` folder is taken only when its
    weights are the text model alone (`model.*`), a `language_model.*` / vision checkpoint is refused by name.
    `--chat` renders the Gemma turn template (a system message is merged into the first user turn). Gemma 2 past its
    4096-token window: 16 / 16 at 4331 tokens vs a correctly windowed reference. Gemma 3 past its 512-token window:
    686-token prompt 29 / 32 vs the windowed reference, the misses are bf16 near-ties (no bug found).
  - Mistral: `[INST]` templates (v0.3, Nemo, Tekken); tools refused. Ministral 3 / Mistral Small 3.x (`mistral3`) and
    Mixtral (MoE) are refused by name.
  - Phi-3: MoE (`phimoe`) and multimodal (`phi3_v`, `phi4mm`) refused. LongRoPE past 4096 tokens untested.
  - Cohere2: the BOS is added as mlx_lm's Python `generate()` does (its CLI and HF add none).
  - StarCoder2: base model, no chat template: `--chat` is refused, use `--prompt`.
- MTP, drafters, the 12 GB mode and the other gateway options are not used by `run`. In `inspect`, "available" means
  the engine has the feature and its files are present, not that `run` uses it.
- The main (hybrid) engine needs `model.safetensors.index.json`; it has no single-file path.
- The tokenizer is `streamdense-tokenizer` (BPE: byte-level or SentencePiece-style with byte fallback; the HF
  normalizers / pre-tokenizers listed in its `src/pipeline.rs`). Other tokenizers fail with its error, naming the part.
- RAM: dense-engine has no budget check before load; the footprint monitor aborts above `[ram] cap_gb` /
  `swap_gb` of `streamdense.toml` (`swap_gb` = system swap growth since start, not absolute swap). For a hard cap, run under `sd-watchdog` (see the main README).

## Build

Own Cargo workspace (it depends on `engine/dense-engine`, which is its own workspace root, so neither can be a member
of `engine/Cargo.toml`):

```sh
cd engine/model_layer && cargo build --release
# binary: engine/model_layer/target/release/sd-model-layer
```

## Usage

```
sd-model-layer inspect <model_dir>
sd-model-layer run <model_dir> (--prompt TEXT | --chat TEXT | --prompt-ids 1,2,3) --max-tokens N [--max-ctx N] [--show-ids]
```

- `inspect` reads `config.json`, the tokenizer files and safetensors headers only (never the weights, no GPU).
- `run` loads the weights and generates, greedy, stopping at an eos id of the model (config.json, generation_config.json;
  Gemma also `<end_of_turn>`). `--prompt` encodes the text as given, `--chat` wraps it in the chat template first (see
  Known limits). `--prompt-ids` skips the tokenizer (for folders without `tokenizer.json`); it then prints token ids
  instead of text. `--max-ctx` defaults to prompt + max-tokens + 8, rounded up to 256. `--show-ids` prints the prompt
  and generated ids (and the rendered chat text) on stderr.
- Both read RAM limits and per-model file paths from `streamdense.toml` (found as the main binary finds it; built-in
  defaults if none).
- Exit code 0 on success, 2 with `ERROR: <reason>` on stderr otherwise.

### `inspect` examples (real output)

A Qwen2.5 dense model (`models/Qwen2.5-Coder-7B-Instruct-4bit`, MLX 4-bit of Qwen/Qwen2.5-Coder-7B-Instruct):

```
model      models/Qwen2.5-Coder-7B-Instruct-4bit
config     <repo>/streamdense.toml
family     qwen2 (model_type qwen2)
engine     dense-engine
quant      4-bit, group 64
tokenizer  loads (151665 ids)
files
  present  model.safetensors.index.json or model.safetensors
  present  tokenizer.json
  missing  chat_template.jinja (else tokenizer_config.json chat_template; run --chat renders it)
engine config check
  nothing refused
optimizations
  MTP draft head                     only in the other engine
  ...
  fast prompt read                   available  [PrefillMode::Gemm (default), prompts of >= 16 tokens]
  pipelined decode                   available  [DecodeMode::Pipelined (default)]
  ...
a detected family does not prove a model runs; speed and memory of a new model have to be measured.
```

The Qwen3.8-27B hybrid model (`models/qwen38_27b_mlx4`):

```
family     qwen3.8-hybrid (model_type qwen3_5_text)
engine     main engine (streamdense-model)
quant      4-bit, group 64
tokenizer  loads (248077 ids)
files
  present  model.safetensors.index.json (the main engine has no single-file path)
  present  tokenizer.json
  present  chat_template.jinja (else tokenizer_config.json chat_template; run --chat renders it)
engine config check
  nothing refused
optimizations
  MTP draft head                     available  [mtp/weights.safetensors present; [gateway] mtp / QwenOptions.mtp]
  adaptive spec depth + draft table  available  [[gateway] spec_depth = auto, needs MTP]
  draft vocab                        available  [181544 ids; needs MTP; [gateway] draft_vocab]
  DFlash drafter                     needs per-model file dflash.json + dflash.safetensors, missing  [...]
  forheader score file               available  [... fits this model's dims]
  forheader 12 GB mode               off by default until tested  [[forheader] mode, research, default "off"; ...]
  ...
```

A folder of an unsupported family:

```
ERROR: model_type "<type>" is not supported yet (supported: qwen2, qwen3, qwen3_5, qwen3_5_text, llama, granite, gemma, gemma2, gemma3_text)
```

A Llama 3.2 model (`Llama-3.2-1B-Instruct-4bit`, MLX 4-bit; the other families print the same two extra lines):

```
family     llama (model_type llama)
engine     dense-engine
quant      4-bit, group 64
tokenizer  loads (128256 ids)
generation Llama 3.1 / 3.2 (llama3 RoPE scaling)
chat       llama3-dated (3.1 / 3.2)
files
  present  model.safetensors.index.json or model.safetensors
  present  tokenizer.json
  missing  chat_template.jinja (else tokenizer_config.json chat_template; run --chat renders it)
engine config check
  nothing refused
...
```

`generation` / `chat` for the other tested models: `Granite (dense, attention only)` / `granite chatml (thinking on,
template default)`; `Gemma 3 (text)` / `gemma turns`; SmolLM: `Llama-architecture (plain RoPE theta 10000, head dim 64)`
/ `chatml (SmolLM / SmolLM2)`. A Granite hybrid folder:

```
ERROR: granite: layer type mamba not supported yet (model_type "granitemoehybrid", hybrid model): the engine has attention layers only
```

### `run` examples (real output)

Plain prompt and chat on the four tested models (32 / 16 new tokens, greedy; stderr lines shortened):

```
$ sd-model-layer run <Llama-3.2-1B-Instruct-4bit> --prompt "Water boils at a lower temperature at high altitude because" --max-tokens 32
family llama -> dense-engine, prompt tokens 11, max_ctx 256
 the air pressure is lower.
The boiling point of water is 100\({}^{\circ}\)C or 212\({}^{\circ
32 tokens, stop length, prefill 0.042 s, decode 0.140 s, peak footprint 0.924 GB

$ sd-model-layer run <Llama-3.2-1B-Instruct-4bit> --chat "Name three primary colors." --max-tokens 16
chat template llama3-dated (3.1 / 3.2)
family llama -> dense-engine, prompt tokens 40, max_ctx 256
Here are three primary colors:

1. Red
2. Blue
3.
16 tokens, stop length, prefill 0.053 s, decode 0.071 s, peak footprint 0.946 GB

$ sd-model-layer run <SmolLM-135M-Instruct-4bit> --chat "Name three primary colors." --max-tokens 16
chat template chatml (SmolLM / SmolLM2)
Primary colors are red, green, and blue. They are the colors that we

$ sd-model-layer run <granite-4.2-3b-MLX-4bit> --prompt "Water boils at a lower temperature at high altitude because" --max-tokens 32
family granite -> dense-engine, prompt tokens 10, max_ctx 256
 the boiling point decreases with decreasing atmospheric pressure. At sea level, water boils at 100°C. At higher altitudes, atmospheric pressure is lower, so
32 tokens, stop length, prefill 0.055 s, decode 0.348 s, peak footprint 2.219 GB

$ sd-model-layer run <granite-4.2-3b-MLX-4bit> --chat "Name three primary colors." --max-tokens 16
chat template granite chatml (thinking on, template default)
Okay, the user asked, "Name three primary colors." It's a straightforward

$ sd-model-layer run <gemma-3-270m-it-4bit> --prompt "Water boils at a lower temperature at high altitude because" --max-tokens 32
family gemma -> dense-engine, prompt tokens 11, max_ctx 256
 of the constant evaporation of water from the atmosphere. This is a fundamental principle of the Earth's atmosphere. The process is continuous, and the rate of evaporation
32 tokens, stop length, prefill 0.025 s, decode 0.101 s, peak footprint 0.465 GB

$ sd-model-layer run <gemma-3-270m-it-4bit> --chat "Name three primary colors." --max-tokens 16
chat template gemma turns
There are three primary colors:
*   Red: F (F)
```

(Text quality is the model's own: the 270M / 135M models are small; their tokens equal mlx_lm's.)

A small hybrid test model (`models/tiny_qwen35`: 8 layers, hidden 512, vocab 2048, 4-bit; it ships no tokenizer, so
the prompt is given as ids and the output is ids). This checks the load + generate path of the layer, not text quality:

```
$ sd-model-layer run models/tiny_qwen35 --prompt-ids 1,2,3,4,5,6,7,8 --max-tokens 16
budget: ...  total 0.424 GB (cap 16.000 GB, enforced live on the real footprint; ...)
family qwen3.8-hybrid -> main engine (streamdense-model), prompt tokens 8, max_ctx 256
1294 1781 1517 330 131 1728 1144 1308 1163 360 1696 861 578 1552 832 327
16 tokens, stop length, prefill 0.012 s, decode 0.040 s, peak footprint 0.119 GB
```

(2.4 s wall time for the whole process including load and Metal pipeline setup, M4 Pro.)

A text prompt on a dense model:

```sh
sd-model-layer run models/Qwen2.5-Coder-7B-Instruct-4bit --prompt "def fibonacci(n):" --max-tokens 48
```

prints the family line, streams the generated text to stdout, and ends with token count, stop reason (`eos`,
`length`), prefill and decode seconds and the peak footprint on stderr.

## Library use

```rust
use streamdense_model_layer::{detect, max_ctx_for, LoadOptions, Model};
let mut m = Model::load(dir, &LoadOptions { max_ctx: max_ctx_for(prompt.len(), 64), cap_gb: 16.0, swap_gb: 7.0 })?;
let g = m.generate(&prompt_ids, 64, &mut |tok| { /* stream */ })?;   // g.tokens, g.stop_reason, g.prefill_s, g.decode_s
```

## Code

- `src/family.rs`: `FAMILIES`, the only list of families; `detect` reads `config.json`, rejects MoE ("dense models
  only") and unknown `model_type`s ("not supported yet", naming the type), and checks dense vs hybrid shape.
- `src/engine.rs`: `Model::load` + `Model::generate` (greedy, token ids) over `dense_engine::Engine` or `QwenEngine`.
- `src/report.rs`: the `inspect` report: family, engine, quantization, what the engine's config check refuses, files
  present, the state of each existing optimization.
- `src/llama/`, `src/granite/`, `src/gemma/`, `src/mistral/`, `src/phi3/`, `src/olmo2/`, `src/smollm3/`, `src/glm4/`,
  `src/stablelm/`, `src/cohere2/`, `src/starcoder2/`: one folder per non-Qwen family (below).
- `src/localdate.rs`: local calendar date for the dated chat templates (Llama 3.1 / 3.2, SmolLM3).
- `src/main.rs`: the `sd-model-layer` CLI.

Tests: `cargo test` (config fixtures in a temp folder; no weights, no GPU).

## Add a family

A family that runs on dense-engine is one folder `src/<family>/` plus its wiring:

1. `src/<family>/mod.rs` with this interface (see `src/llama`, `src/granite`, `src/gemma`):
   - `MODEL_TYPES: &[&str]`: the config.json `model_type` strings it takes.
   - `dense_config(cfg: &Value, model_dir: &Path) -> Result<DenseConfig, String>`: config.json (+ generation_config.json
     etc. from the folder) -> `dense_engine::config::DenseConfig`, starting from `DenseConfig::base(..)` and filling the
     math fields (table in the [dense-engine README](../dense-engine/README.md#other-dense-families-config-driven-math));
     refuse what the engine would get wrong, by name; call `validate()`. Never guess a field that changes the math.
   - `chat_prompt(messages, add_generation_prompt) -> Result<String, String>`: the chat template as fixed strings,
     checked against the Jinja template's render (`apply_chat_template(.., tokenize=False)`); say in the doc whether the
     rendered text holds the BOS text (then it is encoded with `Tokenizer::encode`).
   - `blockers(cfg: &Value) -> Vec<String>`: what stops the config, config.json only (for `inspect`).
   - `tests.rs`: config fixtures (real config.json shapes), template strings, refusals.
2. Wiring: `pub mod <family>;` in `src/lib.rs`; a `Family` value and a `FAMILIES` entry in `src/family.rs`
   (`model_types: crate::<family>::MODEL_TYPES`, `Shape::Dense`, `EngineKind::DenseEngine`); its arms in
   `src/engine.rs` (`dense_config`, `template_name`, `chat_text`, `check_text_run`) and `src/report.rs`
   (`generation`, blockers in `Report::build`). The compiler lists every match that needs an arm.
3. Tokenizer: check the folder's `tokenizer.json` loads in `streamdense-tokenizer` and encodes like HF `tokenizers`.
4. Real test before listing it as supported: `inspect`; `run --prompt` greedy vs mlx_lm on the same prompt ids
   (identical prefix, first divergence at a rounding tie); `run --chat` text vs `apply_chat_template`.

A family on an engine that is not listed also needs an `EngineKind` value and its arms in `Model::load` /
`Model::generate` and `Report::build` / `optimizations`.

Third-party and license notices: [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md), [`../../LICENSE`](../../LICENSE).
