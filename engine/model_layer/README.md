# sd-model-layer: run Qwen models through one command

`sd-model-layer` is one entry point in front of the streamdense engines. Point it at a model folder (an MLX 4-bit
conversion with `config.json`, safetensors and `tokenizer.json`): it reads `config.json`, picks the model family and
the engine that runs it, and either reports what applies to that model (`inspect`) or generates text (`run`). It holds
no inference code of its own; it calls the existing engines.

The gateway (`streamdense up` / `streamdense gateway`: Anthropic Messages API, MTP speculative decoding, the 12 GB mode)
is built for Qwen3.8-27B. Every other supported model runs here, as plain greedy decoding.

## Supported families

Only families that were run on real models are listed. Anything else is refused by name before any weight is read.

| Family | `model_type` in config.json | Examples | Engine | Tested on real models |
|---|---|---|---|---|
| Qwen2 / Qwen2.5 dense | `qwen2` | Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct | [`engine/qwen-dense`](../qwen-dense/README.md) | Qwen2.5-1.5B-Instruct vs MLX (below) |
| Qwen3 dense | `qwen3` | Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, Qwen/Qwen3-14B | [`engine/qwen-dense`](../qwen-dense/README.md) | Qwen3-1.7B vs MLX (below) |
| Qwen3.5 / Qwen3.8 hybrid | `qwen3_5`, `qwen3_5_text` | Qwen3.8-27B (MLX 4-bit) | main engine (`engine/crates/streamdense-model`) | Qwen3.8-27B (the gateway's model), small test model through this layer |

Dense models need head dim 128 (every Qwen2.5 / Qwen3 dense size from 1.5B up; Qwen2.5-0.5B has head dim 64 and is
refused). Weights must be MLX 4-bit, group size 64.

Verification: `engine/qwen-dense` was checked against MLX (mlx 0.32.4 / mlx_lm 0.32.2) on Qwen/Qwen3-1.7B and
Qwen/Qwen2.5-1.5B-Instruct, 6 prompts x 128 greedy tokens: correct text from the first token, teacher-forced agreement
with MLX >= 96.9 % on every prompt, every first divergence a bf16 rounding tie. Details: the
[qwen-dense README](../qwen-dense/README.md#real-model-test-2026-11-02). The hybrid path is the main engine that runs
Qwen3.8-27B; this layer runs it with plain greedy decode.

## Known limits

- Greedy decoding only: no sampling, temperature or top-p.
- Raw prompt: `run` tokenizes the text as given. No BOS token, no chat template (wrap the prompt in the model's chat
  format yourself if you want chat behaviour).
- Head dim 64 dense models (e.g. Qwen2.5-0.5B) are rejected by the qwen-dense config check.
- Mixture-of-experts models (`qwen2_moe`, `qwen3_moe`, any config with expert keys) are rejected: dense models only.
- `rope_scaling` (YaRN etc.) and sliding-window layers are not supported (reported by name).
- MTP, drafters, the 12 GB mode and the other gateway options are not used by `run`. In `inspect`, "available" means
  the engine has the feature and its files are present, not that `run` uses it.
- The main (hybrid) engine needs `model.safetensors.index.json`; it has no single-file path.
- The tokenizer is `streamdense-tokenizer` (byte-level BPE, NFC normalizer). Other tokenizers fail with its error.
- RAM: qwen-dense has no budget check before load; the footprint monitor aborts above `[ram] cap_gb` /
  `swap_gb` of `streamdense.toml`. For a hard cap, run under `sd-watchdog` (see the main README).

## Build

Own Cargo workspace (it depends on `engine/qwen-dense`, which is its own workspace root, so neither can be a member
of `engine/Cargo.toml`):

```sh
cd engine/model_layer && cargo build --release
# binary: engine/model_layer/target/release/sd-model-layer
```

## Usage

```
sd-model-layer inspect <model_dir>
sd-model-layer run <model_dir> (--prompt TEXT | --prompt-ids 1,2,3) --max-tokens N [--max-ctx N]
```

- `inspect` reads `config.json`, the tokenizer files and safetensors headers only (never the weights, no GPU).
- `run` loads the weights and generates. `--prompt-ids` skips the tokenizer (for folders without `tokenizer.json`);
  it then prints token ids instead of text. `--max-ctx` defaults to prompt + max-tokens + 8, rounded up to 256.
- Both read RAM limits and per-model file paths from `streamdense.toml` (found as the main binary finds it; built-in
  defaults if none).
- Exit code 0 on success, 2 with `ERROR: <reason>` on stderr otherwise.

### `inspect` examples (real output)

A Qwen2.5 dense model (`models/Qwen2.5-Coder-7B-Instruct-4bit`, MLX 4-bit of Qwen/Qwen2.5-Coder-7B-Instruct):

```
model      models/Qwen2.5-Coder-7B-Instruct-4bit
config     <repo>/streamdense.toml
family     qwen2 (model_type qwen2)
engine     qwen-dense
quant      4-bit, group 64
tokenizer  loads (151665 ids)
files
  present  model.safetensors.index.json or model.safetensors
  present  tokenizer.json
  missing  chat_template.jinja (run does not apply it)
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
  present  chat_template.jinja (run does not apply it)
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
ERROR: model_type "<type>" is not supported yet (supported: qwen2, qwen3, qwen3_5, qwen3_5_text)
```

### `run` example (real output)

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
- `src/engine.rs`: `Model::load` + `Model::generate` (greedy, token ids) over `qwen_dense::Engine` or `QwenEngine`.
- `src/report.rs`: the `inspect` report: family, engine, quantization, what the engine's config check refuses, files
  present, the state of each existing optimization.
- `src/main.rs`: the `sd-model-layer` CLI.

Tests: `cargo test` (config fixtures in a temp folder; no weights, no GPU).

## Add a family

Add one entry to `FAMILIES` in `src/family.rs`. If it goes to an engine that is not listed, also add an `EngineKind`
value and its arms in `Model::load` / `Model::generate` and `Report::build` / `optimizations`; the compiler lists them.
Then add a fixture test, and measure on a real model before listing it as supported.

Third-party and license notices: [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md), [`../../LICENSE`](../../LICENSE).
