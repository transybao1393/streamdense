# streamdense: architecture (Rust + Metal + Swift)

Decode engine for dense-architecture LLMs on Apple Silicon (first target: ProSparse-Llama-2-7B, 4-bit export).
Weights are always streamed from the SSD under a hard RAM cap.
Status: build started 2026-10-29.

Numbers marked (m) were measured in this repo; numbers marked (e) are estimates.

## 0. Top 5 measured wins carried into the build

All five are required in Phase 1 and must stay on in the streaming path. They are what make today's 12.15 ms/token sparse decode (vs 17.82 dense) at ctx 1024 (m).

| # | Win | Measured result | Where in streamdense | Phase |
|---|---|---|---|---|
| 1 | Exact-gate sparse FFN (FATReLU 0.01, per neuron) | 11.2% of neurons active; FFN ~2.05x faster; full decode 1.47x vs dense (m) | `gate.metal` + `compact.metal` + `sparse_up` / `axpy_down`; it is also what makes the LRU neuron cache possible | 1 (in RAM), 2 (through slots) |
| 2 | W_q ×0.9 folded into the checkpoint | ppl 19.73 → 12.55 at ctx 512, 8.89 → 7.69 at ctx 2048 (m) | Already baked into the q4 export; the converter keeps it and refuses an unfolded checkpoint (`streamdense_q_scale_folded` flag) | 1 |
| 3 | 8-bit KV cache g32 + attention v2 kernel | no quality loss; 213–223 GB/s; part of a −10% decode gain (m) | `attn_kv8.metal`; KV allocated in the arena at max context (0.295 MB/token) | 1 |
| 4 | Tuned gate kernel r2_d1_s0 tpt64 | gate 117.1 → 103.8 µs (−11%) (m) | `gate.metal` default | 1 |
| 5 | 8-bit lm_head | ppl −0.01%, same output on 36/40 prompts; head 1.04 → ~0.58 ms (m) | `head_q8.metal`, pinned | 1 |

Pipelining (−0.6 ms/token (m)) is also kept. With streaming, the CPU acts between layers, so it becomes one command buffer per token with shared events, and the next token's encoding overlaps the GPU tail.

## 1. Who does what

| Layer | Language | Job | Why there |
|---|---|---|---|
| GPU math | **Metal (MSL)** | All matrix work: 4-bit GEMV, gate, sparse up / axpy-down, 8-bit KV attention, norms, RoPE, lm_head, argmax, ordered compaction | Decode is limited by memory bandwidth; our kernels reach ~240 GB/s (m) |
| Engine core | **Rust** | Hard RAM cap, memory arena, SSD reader threads, LRU / pin cache, scheduling, model logic, CLI, C API | Precise memory control, safe threads, shareable with moeflux (streamdense-core) |
| Kernel lab + oracle | **Swift** | Prototype and benchmark new kernels; `decode7b.swift` stays the token-for-token reference | Native Metal, fastest to iterate, already verified |
| App (later) | **Swift (SwiftUI)** | Chat window, live RAM / tok/s, cap slider; links the Rust library over a C ABI | Best Mac UI option |

Rule: one set of `.metal` shader files, used by both Rust (engine) and Swift (lab), so a kernel proven in the lab drops into the engine unchanged.

## 2. Repository layout

```
streamdense/
  engine/                         Cargo workspace (Rust)
    shaders/                      MSL, shared by Rust and Swift
      gemv_q4.metal               attention projections (4-bit g64)
      gate.metal                  tuned gate r2_d1_s0 tpt64 + threshold
      compact.metal               ordered active-list compaction (deterministic)
      sparse_up.metal             up rows through a slot table
      axpy_down.metal             down columns through a slot table (neuron-major)
      attn_kv8.metal              rope + kv8 g32 write, attention v2
      head_q8.metal               8-bit lm_head + argmax
      misc.metal                  rmsnorm, residual, embed row
    crates/
      streamdense-core/           model-independent, later shared with moeflux
        arena.rs                  one fixed MTLBuffer, sub-allocation, hard cap
        budget.rs                 startup budget: pinned + KV + scratch + cache
        io.rs                     pread + F_NOCACHE thread pool (QD 32-64)
        cache.rs                  CachePolicy trait: Lru (neurons), Pin/Ring (whole tensors)
        footprint.rs              phys_footprint monitor, abort over cap + margin
      streamdense-metal/          device, pipelines, command buffers, MTLSharedEvent
      streamdense-model/          config, weight index, layers, KV cache, decode loop, sampler
      streamdense-convert/        q4 export -> streaming file (.sdx)
      streamdense-cli/            run | bench | check
      streamdense-ffi/            C ABI + generated header for Swift
  swift/
    KernelLab/                    SwiftPM: ffnbench, gemv3bench, new kernel trials (loads ../engine/shaders)
    RefDecoder/                   decode7b.swift (oracle)
    StreamDenseApp/               SwiftUI app (later)
  exports/q4_prosparse_llama2_7b/ current 4-bit export (input to the converter)
```

## 3. Memory under the hard cap

All model memory lives in **one Metal buffer** (`storageModeShared`, unified RAM), allocated once at startup and never grown:

```
+--------------------------- arena = cap - process reserve ---------------------------+
| pinned weights      | KV cache (8-bit g32)  | scratch          | neuron cache (LRU)    |
| attn 1.208 GB (m)   | 0.295 MB x max_ctx(m) | activations,     | slots of 4608 B       |
| gate 0.812 GB (m)   | preallocated          | active lists,    | (up row + down col)   |
| head q8 ~0.14 GB    |                       | slot table       | = everything left     |
| norms               |                       |                  |                       |
+--------------------------------------------------------------------------------------+
embeddings (fp16, 262 MB): streamed, one 8 KB row per token, not resident
```

How the cap is enforced:
1. **Budget check at start.** If pinned + KV(max_ctx) + scratch + a minimum cache won't fit, refuse with the shortfall.
2. **Fixed arena.** The GPU side cannot allocate past it.
3. **No hot-loop heap growth.** CPU buffers are preallocated, and a counting global allocator enforces a small CPU heap limit.
4. **F_NOCACHE on every read.** Streamed weights don't fill the macOS file cache, so system RAM is spared as well.
5. **Footprint monitor thread.** Checks `phys_footprint` every 1 s and aborts above cap + margin.

Example, cap 3.5 GB at max context 2048 (e):

| Part | Size |
|---|---|
| Pinned | 2.16 GB |
| KV | 0.60 GB |
| Scratch + reserve | ~0.2 GB |
| Neuron cache | ~0.54 GB ≈ 117k of 352k neurons (33%) |

Previous estimate for an LRU covering 30% of neurons: ~13 tok/s (`tier_est.py`).

## 4. Streaming file (.sdx)

```
header (JSON: config, tensor table, offsets, quant format)
pinned section     4 KiB-aligned tensors (attention, gate, head, norms, embeddings)
neuron section     per layer, per neuron id: [up row q4 + scale/bias | down column q4 + scale/bias] = 4608 B
```

- One neuron is one record, so one read fetches it.
- The SSD is limited by reads per second (~220k/s at 4-8 KiB, QD 64, (m)), not by MB/s. A later converter option can reorder neurons by co-activation so that neighbours share reads.

## 5. One token, step by step

One command buffer per token. The GPU and the Rust I/O thread hand off per layer with `MTLSharedEvent`:

```
CPU: read embedding row (pread 8 KB) -> scratch
for layer l in 0..32:
  GPU  rmsnorm -> q/k/v GEMV -> rope + kv8 write -> attention v2 -> o GEMV -> residual
  GPU  rmsnorm -> gate GEMV + threshold -> ordered compaction -> active list   signal E[l]
  CPU  wait E[l]: for each active neuron, look up in the cache
         hit  -> slot id
         miss -> take an LRU slot (never one used in this layer), pread record into the slot
       write the slot table                                                      signal F[l]
  GPU  wait F[l]: sparse up (slots) -> FATReLU 0.01 -> axpy down (slots) -> residual
GPU: final norm -> head q8 -> argmax -> token (CPU reads 4 bytes)
```

Speed-ups, added after correctness:
- **Prefetch layer l+1 during layer l.** Use the previous token's active set for that layer: neurons reused token to token are 47% (m).
- **Batch the misses for each layer** into one pread wave at full queue depth.
- **Speculative decoding (prompt lookup).** One verify pass covers several tokens, so SSD reads per accepted token drop.

Cost floor: at least one SSD read latency per layer that has misses, about 0.1–0.5 ms × 32 (e).

## 6. Cache policies

| Policy | Used for | Behaviour |
|---|---|---|
| **LRU (neuron)** | Sparse FFN up/down (ProSparse: ~11% active per token (m)) | Cache slots hold recently active neurons; misses come from the SSD |
| **Pin** | Attention, gate, head, norms | Always resident |
| **Pin + Ring** | Non-sparse dense models (e.g. SwiGLU): every weight is read every token | Pin a fixed part, then stream the rest through a double-buffered ring with large sequential reads (~3.3 GB/s (m)). LRU would always miss on this access pattern, so it is not used here |

The model config picks the policy per tensor, so the same engine runs sparse and non-sparse dense models.

Stream by layer (the Ring) costs ~35–50 ms per streamed 7B layer per token (114 MB at ~2.3–3.3 GB/s, estimate), against 0.37 ms to compute it. So it is used only:
- for prefill, where one read serves every prompt token;
- for non-sparse dense models;
- for attention, when the cap is too small to pin it.

For the sparse FFN, the LRU neuron cache reads ~5x less (e).

Speculative decoding (prompt lookup, k ≤ 2) runs through the same pass. A k-token pass reads the weights and fetches the misses once for all k tokens. k = 2 was measured as the sweet spot of the current k-token kernels; a matrix-unit kernel is needed beyond that. TEAL on W_o was measured and dropped: break-even at 50% sparsity.

## 7. Correctness and tests

- **Reference:** greedy tokens from `decode7b.swift` on 6 prompts (default settings), with the prompts in `engine/refs/prompts7b.json`; the reference token file is not shipped and is passed to `streamdense check --ref FILE`.
- **Bar:**
  - identical tokens with everything in RAM;
  - identical tokens at every cap, because the cache must never change the result.
- Ordered compaction makes the active list deterministic. The current atomic version is not.
- **`streamdense check`:** token compare against the reference.
- **`streamdense bench`:** reports ms/token, cache hit %, SSD MB/token, reads/token and phys_footprint.

## 8. Swift ↔ Rust boundary (C ABI)

```c
sd_handle* sd_open(const char* dfx_path, uint64_t ram_cap_bytes, uint32_t max_ctx);
int        sd_prompt(sd_handle*, const int32_t* tokens, uint32_t n);
int32_t    sd_next(sd_handle*);                 // greedy next token
void       sd_stats(sd_handle*, sd_stats* out); // tok/s, footprint, hit %, SSD MB/token
void       sd_close(sd_handle*);
```

Swift owns the UI and the tokenizer display. Rust owns everything else.

## 9. Build order

| Phase | Deliverable | Pass bar | Estimate |
|---|---|---|---|
| 1 | Workspace, shaders ported, in-RAM decode with arena + budget + F_NOCACHE load, all 5 wins on (section 0) | Tokens identical to `decode7b.swift`; ms/token near 12.15 at ctx 1024 (m) | ~2 h |
| 2 | `.sdx` converter, I/O pool, LRU + pin, slot-table kernels, event hand-off | Identical tokens at every cap | ~2-3 h |
| 3 | Bench at caps 3.0 / 3.5 / 4.5 GB, footprint monitor, README | Measured table vs `tier_est.py` | ~1 h |
| 4 | KernelLab moved onto the shared shaders; C ABI; SwiftUI app shell | App runs the 7B under a chosen cap | ~2–3 h |
| 5 | Pin + Ring path (prefill through the ring; forced layer streaming on the 7B to test it), prompt lookup k ≤ 2 through the streaming pass | Identical tokens; measured ms/token | ~2 h |

Machine rules for every test run:
- One model process at a time, under the footprint monitor.

## 10. Open points

- Tokenizer in Rust (`tokenizers` crate offline availability not checked; Llama-2 SentencePiece).
- Neuron record alignment: 4608 B packed vs padded to 8 KiB. Measure both.
- The sparse path uses neuron-major W_down rounding, which is a slightly different 4-bit model. The decision is still open.
- Cap 3.0 GB is tight: about 0.2 GB of cache at max context ~1152 (e).
