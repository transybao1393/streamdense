# Third-party notices

This repository's own code is under the terms in [LICENSE](LICENSE). The parts listed below come from, or are
adapted from, other projects and keep their own licenses. No third-party model weights are included in this repository.

## 1. Source code adapted from other projects

### MLX and mlx-lm (Apple), MIT License

Several Metal kernels reproduce the thread partition, reduction order and bf16 rounding points of MLX's Metal
kernels (and of mlx-lm's gated-delta kernel), so that the engine matches MLX bit for bit. The kernels are written
for this engine (different buffers, fused ops, multi-token dispatch), but the numerical structure follows MLX.

| File | Origin |
|---|---|
| [engine/shaders/attn_gated.metal](engine/shaders/attn_gated.metal) | MLX `mlx/backend/metal/kernels`: `rms_norm`, `rope`, `sdpa_vector.h` (`sdpa_vector`, `sdpa_vector_2pass`), unary `Sigmoid`; steel `BaseMMAFrag` fragment layout (`attn_flash_b16`) |
| [engine/shaders/gemv_q4.metal](engine/shaders/gemv_q4.metal) | MLX `quantized.h` `load_vector` / `qdot` method (pre-scaled x, unshifted nibble masks) |
| [engine/shaders/sparse_up.metal](engine/shaders/sparse_up.metal) | MLX `quantized.h` `load_vector` / `qdot` method, on gathered rows |
| [engine/qwen-dense/shaders/attn_hd128_sdpa.metal](engine/qwen-dense/shaders/attn_hd128_sdpa.metal) | MLX `sdpa_vector.h` (`sdpa_vector`, `sdpa_vector_2pass`) at head dim 128, and the host block-count rule of `scaled_dot_product_attention.cpp` |
| [engine/qwen-dense/shaders/attn_hd128.metal](engine/qwen-dense/shaders/attn_hd128.metal), [attn_hd128_flash.metal](engine/qwen-dense/shaders/attn_hd128_flash.metal) | Via `attn_gated.metal`: MLX `rms_norm`, `rope`, sdpa partition and rounding, steel `BaseMMAFrag` layout |
| [engine/shaders/deltanet.metal](engine/shaders/deltanet.metal) | mlx-lm `models/gated_delta.py` (packed Dk=128 gated-delta kernel), `models/qwen3_5.py` `GatedDeltaNet`; MLX `rms_norm`, `Sigmoid`, `log1p` |
| [engine/shaders/gemv_bf16.metal](engine/shaders/gemv_bf16.metal) | MLX `gemv.h` `GEMVKernel` |
| [engine/shaders/gemv_q4_bf16.metal](engine/shaders/gemv_q4_bf16.metal) | MLX `quantized.h` `qmv_fast` / `qmv`, `load_vector`, `qdot` |
| [engine/shaders/norm_bf16.metal](engine/shaders/norm_bf16.metal) | MLX `rms_norm` (`rms_single_row` / `rms_looped`) |
| [engine/shaders/spec.metal](engine/shaders/spec.metal) | MLX `rms_norm` (`rms_single_row` / `rms_looped`) |
| [engine/shaders/gemm_q4.metal](engine/shaders/gemm_q4.metal) | MLX steel `BaseMMAFrag` fragment coordinates; MLX affine 4-bit layout |
| [engine/shaders/gemv_mma_q4.metal](engine/shaders/gemv_mma_q4.metal) | MLX steel `BaseMMAFrag` fragment coordinates; MLX affine 4-bit layout |

Each of these files starts with a header naming the origin and license.

Sources: MLX <https://github.com/ml-explore/mlx>, mlx-lm <https://github.com/ml-explore/mlx-lm>. Both projects use
the same license text and copyright line (checked 2026-11-02):

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Qwen3.8 chat template text, Apache-2.0

[engine/crates/streamdense-tokenizer/src/chat.rs](engine/crates/streamdense-tokenizer/src/chat.rs) renders the
Qwen3.8 chat format in Rust (no template engine) and reproduces the template's fixed strings word for word
(the tool-calling instructions `TOOLS_HEAD` / `TOOLS_TAIL` and the template's error messages), because the prompt must
tokenize exactly as the model expects. Source: `chat_template.jinja` of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (the copy in the
[p4ik/Qwen3.8-27B-MLX-4bit](https://huggingface.co/p4ik/Qwen3.8-27B-MLX-4bit) package says it is adapted from unsloth),
licensed under the Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

### Reimplemented behavior (no code copied)

These parts are written from scratch to give the same results as another project. No source code was copied,
so no license notice is required. They are listed for transparency.

| File | Matches the behavior of | License of the reference |
|---|---|---|
| [engine/crates/streamdense-tokenizer/src/tokenizer.rs](engine/crates/streamdense-tokenizer/src/tokenizer.rs) | Hugging Face `tokenizers` byte-level BPE; the pre-tokenizer regex string that transformers' `Qwen3_5Tokenizer` installs (copied, a short regex); the GPT-2 byte-to-unicode table (OpenAI GPT-2, MIT) | Apache-2.0 ([tokenizers](https://github.com/huggingface/tokenizers), [transformers](https://github.com/huggingface/transformers)) |
| `engine/shaders/gemv_lut_q4.metal` | The T-MAC / LUT-GEMM lookup-table method (from the papers) | n/a (method only) |

## 2. Rust crate dependencies

Crates are fetched from crates.io at build time; this repository does not include their source. If you ship a
compiled binary, include the license texts of these crates with it (for example with `cargo about`).

Three separate Cargo workspaces (from `cargo metadata`, every target platform; `engine/` counted 2026-11-02, the other two 2026-11-14):

| Workspace | Direct dependencies | Crates in the lock graph |
|---|---|---|
| [engine/](engine/Cargo.toml) | libc, objc2, objc2-foundation, objc2-metal, regex, serde, serde_json, toml, unicode-normalization, ureq | 70 |
| [engine/qwen-dense/](engine/qwen-dense/Cargo.toml) | libc, objc2, objc2-foundation, objc2-metal, serde_json, streamdense-core / streamdense-tokenizer (local) | 36 |
| [engine/model_layer/](engine/model_layer/Cargo.toml) | serde_json, qwen-dense / streamdense-core / streamdense-model / streamdense-tokenizer (local) | 36 |

Almost every crate is `MIT OR Apache-2.0`, `MIT`, `Apache-2.0 OR MIT`, or `Unlicense OR MIT`. No GPL, LGPL, MPL
or other copyleft license appears. Crates under a license other than plain MIT / Apache-2.0 (all permissive):

| Crate | License | Note |
|---|---|---|
| ring 0.17 | Apache-2.0 AND ISC | TLS crypto (via ureq / rustls) |
| rustls-webpki 0.103, untrusted 0.9 | ISC | TLS (via ureq) |
| subtle 2.6 | BSD-3-Clause | via rustls |
| webpki-roots 1.0 | CDLA-Permissive-2.0 | Mozilla CA root list as data; permissive data license |
| foldhash 0.1 / 0.2, zlib-rs 0.6 | Zlib | |
| unicode-ident 1.0, unicode-normalization 0.1 | (MIT OR Apache-2.0) AND Unicode-3.0 | Unicode data tables |
| ryu 1.0 | Apache-2.0 OR BSL-1.0 | |
| adler2 2.0 | 0BSD OR MIT OR Apache-2.0 | |

Regenerate with, in each workspace:
`cargo metadata --format-version 1 | jq -r '.packages[] | select(.source != null) | "\(.name) \(.version) \(.license)"'`

## 3. Swift packages

[app/Package.swift](app/Package.swift) has no package dependencies (Apple system frameworks only: SwiftUI, AppKit, IOKit).

## 4. Models

No model weights are distributed in this repository (`models/` is git-ignored). Users download them from Hugging Face
themselves (for example with `sd-fetch`) and must follow each model's license. Licenses below are the `license` field
of each model card, checked 2026-11-02.

| Model | Used for | License |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Main target (base model) | Apache-2.0 |
| [p4ik/Qwen3.8-27B-MLX-4bit](https://huggingface.co/p4ik/Qwen3.8-27B-MLX-4bit) | 4-bit MLX package the engine loads (`models/qwen38_27b_mlx4`) | Apache-2.0 (base_model Qwen/Qwen3.8-27B) |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Config shapes in `engine/qwen-dense` | Apache-2.0 |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Real-model tests of `engine/qwen-dense` vs MLX (MLX 4-bit conversions; weights not included) | Apache-2.0 |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | Config shapes in `engine/qwen-dense` | Apache-2.0 (note: some other Qwen2.5 sizes, e.g. 3B and 72B, use different Qwen licenses) |

Test data: [engine/testdata/vectors/](engine/testdata/vectors) holds random inputs at real shapes run through MLX
ops (no model weights); [engine/refs/](engine/refs) holds token ids, logits and activation dumps produced with MLX / mlx-lm from a
randomly initialised tiny model and from Qwen3.8-27B (model outputs, not weights).
