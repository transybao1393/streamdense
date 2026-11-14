# `streamdense qwen serve`: JSON-lines protocol

The engine runs as one child process. The client (the SwiftUI app) writes one JSON object per line to its
**stdin** and reads one JSON object per line from its **stdout**. Logs go only to **stderr** (free text; show
them in a log pane, don't parse them). The engine handles one request at a time. The only exception is
`{"op":"stop"}`, which is read on a separate thread so it can cancel a request that is running.

The client sends text only. Tokenizing, the chat template and detokenizing all happen in Rust
(`engine/crates/streamdense-tokenizer`).

## Launch

```
engine/target/release/sd-watchdog --cap-gb 16 --margin-gb 0 --swap-gb 7 -- \
  nice -n 10 engine/target/release/streamdense qwen serve \
    --model models/qwen38_27b_mlx4 --max-ctx 4096 --abort-gb 16 --mtp
```

| flag | default | meaning |
|---|---|---|
| `--model DIR` | `models/qwen38_27b_mlx4` | weights + `config.json` |
| `--tokenizer DIR` | the `--model` dir | needs `tokenizer.json`. If the model dir has none, the default 27B dir is tried |
| `--max-ctx N` | 4096 | KV capacity in tokens (prompt + reply) |
| `--abort-gb X` | 16 (`[ram] cap_gb`) | footprint monitor aborts the process above this |
| `--mtp` | off | load the MTP head (about +0.5 GB). Without it, `"mtp":true` requests run plain decode |
| `--fold-ids` | off | **tests only.** Chat prompt ids are fed modulo the model vocab, so the tiny test model can run the chat plumbing with the 27B tokenizer. The output text is meaningless |

Loading the 27B takes tens of seconds. Wait for the `ready` line before you send anything. Lines sent earlier
are buffered by the OS and handled after `ready`, so sending early is safe but gives no benefit.

## Startup line (exactly once)

```json
{"ready":true,"pid":12345,"model":"/abs/models/qwen38_27b_mlx4","max_ctx":4096,"open_s":21.4,
 "arena_gb":15.39,"footprint_gb":15.41,"peak_footprint_gb":15.41,"prefill_k":8,"prefill_t":128,
 "prefill_exact":false,"decode":"sync, greedy","mtp_available":true,
 "chat_available":true,"tokenizer":"/abs/models/qwen38_27b_mlx4/tokenizer.json","vocab_size":248077,
 "tokenizer_load_ms":230.0,"stop_ids":[248046,248044]}
```

- `chat_available:false` means the tokenizer did not load. A `tokenizer_error` string then says why.
  `chat` requests fail, but the token-level `gen` op still works. The tokenizer is also refused when it has
  more ids than the model has embedding rows. This happens with the tiny test model unless you pass `--fold-ids`.
- **Fatal startup errors** (missing weights, the budget does not fit, a Metal error) happen **before** `ready`.
  The process writes `ERROR: <message>` to stderr and exits with code 2. With `sd-watchdog`, a footprint or
  swap kill shows as `[sd-watchdog] ... KILL ...` on stderr and a non-zero exit. In both cases, if stdout closes
  before a `ready` line arrives, report the last `ERROR`/`KILL` stderr line to the user.

## Requests (stdin)

### `chat`: generate an assistant reply from text messages

```json
{"op":"chat",
 "messages":[{"role":"system","content":"You are helpful."},
             {"role":"user","content":"Hi"},
             {"role":"assistant","content":"Hello! How can I help?"},
             {"role":"user","content":"Write a haiku."}],
 "think":false,"nnew":2048,"mtp":true,"continue":true}
```

| field | type | default | meaning |
|---|---|---|---|
| `messages` | array, required | | `role` is `system`, `user` or `assistant`. `content` is a string. System messages may only appear at the start (they are merged with `\n`). The list must not be empty |
| `think` | bool | `false` | `enable_thinking` of the Qwen template. `true` adds the reasoning-effort system text and opens `<think>\n`. `false` pre-fills an empty `<think>\n\n</think>\n\n` |
| `reasoning_effort` | string | `"xhigh"` | only used when `think` is true: `xhigh` (or `high`), `medium` or `low` |
| `nnew` | int | 2048 | the most reply tokens to generate |
| `mtp` | bool | `false` | use MTP speculative decode, if the engine was started with `--mtp`. Gives the same greedy tokens either way |
| `continue` | bool | `true` | allow reuse of the KV/DeltaNet state from the previous request (see *KV reuse*). `false` always resets and runs a full prefill |

**Assistant turns in history.** Send each earlier assistant reply back **exactly as you received it**: the
`text` pieces concatenated, or `done.reply`. When the content contains `</think>`, the server splits it at the
**last** `</think>`. The part before it, without a leading `<think>`, becomes the template's
`reasoning_content`. The part after it is the visible content. You may instead send
`{"role":"assistant","content":"answer","reasoning_content":"thoughts"}` yourself. The template keeps reasoning
in earlier turns (`preserve_thinking` defaults to true). So in the template, an assistant turn always renders
as `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n`, and the reasoning part
can be empty.

**Output lines for a chat request, in order:**

1. One `start` line, after tokenizing and before the prefill:
   ```json
   {"start":true,"op":"chat","n_prompt":812,"kv_reused":790,"n_feed":22,"kv_mode":"prefix","think":false,"tokenize_ms":0.4}
   ```
   - `kv_mode`: `"prefix"` means the KV already held a token prefix of the new prompt. `"followup"` means
     Python-UI-style reuse: the previous request's messages plus one assistant reply plus new turns. `"full"`
     means a reset and a full prefill.
   - `kv_reused`: tokens already in the KV that are not run again. `n_feed`: tokens run now.
     `n_prompt = kv_reused + n_feed`.
2. Zero or more text pieces. Each piece is valid UTF-8; a character that spans tokens is held back until it is
   complete. The stop token (`<|im_end|>` / `<|endoftext|>`) is **not** sent as text.
   ```json
   {"text":"Hello"}
   ```
   With `think:true` the stream starts inside the reasoning and contains a literal `</think>` piece. Text
   before it is reasoning and text after it is the answer.
3. Exactly one `done` line (also after `stop`, a full context, or an error):
   ```json
   {"done":true,"op":"chat","stop_reason":"eos","n_prompt":812,"n_new":57,"kv_reused":790,"n_feed":22,
    "kv_mode":"prefix","reply":"Hello! ...","prefill_ms":95.1,"prefill_tok_s":231.3,"ttft_ms":101.7,
    "gen_tokens":57,"decode_ms":3420.0,"ms_per_tok":61.1,"decode_tok_s":16.4,"gpu_ms_per_tok":58.0,
    "pos":868,"ctx_used":869,"max_ctx":4096,"footprint_gb":15.42,"peak_footprint_gb":15.47,
    "mtp":true,"mtp_requested":true,"mtp_available":true,
    "spec":{"passes":30,"accepted":27,"rejects":3,"acceptance":0.9,"tokens_per_pass":1.9},
    "think":false,"tokenize_ms":0.4,"continue_requested":true}
   ```
   - `stop_reason`: `"eos"` (the model produced `<|im_end|>`/`<|endoftext|>`), `"length"` (reached `nnew`),
     `"cancelled"` (`{"op":"stop"}`), `"ctx_full"` (`max_ctx` reached), or `"error"` (then an `"error"` string
     is also present).
   - When the prompt alone does not fit in `max_ctx`, no `start` line is sent and nothing runs. The `done` line
     then has `stop_reason:"ctx_full"`, `n_new:0`, `reply:""` and `n_prompt` (the size needed). Trim the
     history and resend.
   - `n_new` = `gen_tokens` = tokens generated, including the stop token when there is one. `reply` = the full
     decoded reply without the stop token.
   - `spec` is `null` for plain decode. `ms_per_tok`/`decode_tok_s` are `null` when fewer than 2 tokens were
     generated.

**Errors in a chat request.** For a bad `messages` array, a system message that is not at the start, an empty
list, or a tokenizer that did not load, the server sends no `start` line. It sends
`{"done":true,"op":"chat","stop_reason":"error","error":"<message>","n_prompt":0,"n_new":0,"kv_reused":0}`.
The KV state is left unchanged.

### `stop`: cancel the running request

`{"op":"stop"}`: the running `chat` or `gen` ends after the current token with `stop_reason:"cancelled"` and
its usual `done` line. `stop` itself gets no reply, and does nothing when no request is running. The KV still
holds everything generated so far, so the next `chat` with `continue:true` can reuse it.

### `reset`: drop the KV / DeltaNet state

`{"op":"reset"}` → `{"reset":true,"pos":0}`. Send it for "new conversation". `continue:false` on the next chat
does the same.

### `stats`

`{"op":"stats"}` →
`{"stats":true,"pos":868,"pending":248046,"kv_tokens":869,"max_ctx":4096,"footprint_gb":15.42,"peak_footprint_gb":15.47,"heap_mb":3.1}`
(`kv_tokens` = tokens the next request can reuse, including the pending last token.)

### `gen`: token-level (kept for tools and tests)

`{"op":"gen","tokens":[..ids..],"nnew":N,"stop":[ids],"continue":bool,"mtp":bool}` → `{"tok":id}` per token,
then a `done` line (same fields as chat, but without `n_prompt`/`n_new`/`reply`/`kv_mode`; it uses
`prompt_tokens`). A `gen` request makes the chat KV bookkeeping invalid, so the next `chat` runs in `"full"`
mode.

### Other errors

- A line that is not JSON → `{"error":"bad JSON line: ..."}` (no `done`).
- An unknown op → `{"error":"unknown op \"x\""}` (no `done`).
- Closing stdin ends the process (exit 0) after the current request.

## KV reuse (`continue:true`)

The server remembers the token sequence in the KV (prompt + generated tokens, including the last generated
token, which is not yet fed) and the `messages`/`think`/`reasoning_effort` of the last chat. For a new chat:

1. **prefix**: the new prompt is rendered and tokenized. If the KV sequence is a token prefix of it, only the
   rest is fed.
   This also covers a reply that stopped at `length`/`cancelled`, as long as you send its text back unchanged.
   The missing `<|im_end|>` is then simply part of the rest that gets fed.
2. **followup** (what the old Python UI did): if the earlier rendering does not match token for token (for
   example, you sent back a trimmed or edited reply, or the reply tokenizes differently), the server checks whether
   the new `messages` are the previous request's messages, then one `assistant` message, then new message(s),
   with the same `think`/`reasoning_effort`. If so, the server feeds `<|im_end|>` (only when the reply did not
   end with it), then `\n` plus the template text of the new turns plus the generation prompt. The model's own
   reply tokens stay in the KV, and the content of the assistant message you sent is not used.
3. **full** in every other case: an edited earlier message, a different `think`, a different system prompt,
   `continue:false`, a `gen` in between, `reset`, a request that ended in `error` or `ctx_full`, or a reply with
   no tokens.

If the reused prompt does not fit in `max_ctx`, the server tries a full prefill. If that does not fit either,
the request ends with `ctx_full`.

## Stop tokens / ids

`<|im_end|>` = 248046, `<|endoftext|>` = 248044 (both end a reply). `<think>` = 248068, `</think>` = 248069.
