# streamdense gateway: Claude Code on the local Qwen3.8-27B

`streamdense gateway` serves the Anthropic Messages API on `127.0.0.1` and runs the 27B engine in the same process
(crate `engine/crates/streamdense-gateway`, CLI wiring in `engine/crates/streamdense-cli/src/gateway.rs`).
Claude Code talks to it like it talks to the Anthropic API.

## Start

```sh
cd engine && cargo build --release
engine/target/release/streamdense up --log /tmp/gw.jsonl     # all limits from streamdense.toml (see "RAM limits")
```

`streamdense up` is the same as the long form below with the values from `streamdense.toml`:

```sh
B=engine/target/release
$B/sd-watchdog --cap-gb 16 --margin-gb 0 --swap-gb 7 -- nice -n 10 \
  $B/streamdense gateway --model models/qwen38_27b_mlx4 --port 8898 --max-ctx 8192 --mtp --log /tmp/gw.jsonl
```

Check RAM and disk first (`df -h /`, `sysctl vm.swapusage`): the engine holds about 15.6 GB. The process also
aborts itself above 16 GB phys_footprint (`--abort-gb`) or when system swap has grown by more than 7 GB since it
started (`--swap-gb`; the baseline is printed on the `FOOTPRINT MONITOR: start` line), like `qwen serve`. Since
2026-11-14 `--swap-gb` / `swap_gb` is a growth limit for the engine and `sd-watchdog` alike, not absolute system swap:
swap other programs already held does not stop a model.
stdout gets one `{"ready":true,"port":..}` line when the port is open.

Options: `--port N` (0 = any free port), `--host`, `--max-ctx N` (default 8192; 8K ≈ 15.9 GB, 12K ≈ 16.2 GB peak), `--mtp` (load the MTP head; MTP
decode is then on, `--no-mtp-decode` turns it off), `--think` (honor the client's `thinking` request; default off,
for speed), `--log FILE` (request log, else stderr), `--tokenizer DIR`, `--name qwen38-27b` (model id reported),
engine options as in `qwen serve` (`--cap`, `--reserve`, `--kv`, `--prefill-k`, `--prefill-t`, `--prefill-exact`,
`--threads`).
Snapshots (on by default, see "Snapshots" below): `--no-snapshots`, `--snapshot-dir DIR` (default
`~/Library/Caches/streamdense/snapshots`, or `$SD_SNAPSHOT_DIR`), `--snapshot-max-files N`, `--snapshot-max-gb GB` (defaults from `[snapshots]`: 32 files, 8 GB; see "Snapshot tiers").
Watcher state (see "Domain watcher"): `--watcher-dir DIR` or `$SD_WATCHER_DIR` (default `[watcher] dir`); test and smoke runs
should pass both `--watcher-dir` and `--snapshot-dir` (scratch dirs) so they never touch the user's state. `streamdense up`
passes every gateway flag through; `kill <pid of up>` (SIGTERM) now stops the gateway too (sd-watchdog forwards it).

## RAM limits

The RAM cap for every run lives in `streamdense.toml` at the repo root (read by `sd-watchdog`, `streamdense gateway`,
`streamdense qwen serve`, `streamdense up` and the Swift app):

```toml
[ram]
cap_gb    = 16.0   # sd-watchdog kill + engine self-abort (--abort-gb)
swap_gb   = 7      # stop when system swap grows by more than this since start (not absolute swap)
margin_gb = 0      # sd-watchdog margin
nice      = 10
[gateway]
model = "models/qwen38_27b_mlx4"   # port = 8898, max_ctx = 8192, mtp = true
```

Edit the file, then restart with `streamdense up` (extra args go to the gateway and win over the file, e.g.
`streamdense up --max-ctx 12288 --no-mtp`). `streamdense config` prints the resolved values and which file they came
from (`--json` for scripts). Search order: `$SD_CONFIG`, `./streamdense.toml`, the repo root (found from the binary),
`~/.config/streamdense/config.toml`; no file means the same built-in values. Unknown keys are an error. Precedence:
CLI flag > config > default.

## Point Claude Code at it

```sh
eval "$(engine/target/release/streamdense gateway --print-env --port 8898)"
claude --bare
```

`--print-env` loads nothing; it prints:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8898
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_API_KEY=
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_MODEL=qwen38-27b
export ANTHROPIC_SMALL_FAST_MODEL=qwen38-27b
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen38-27b
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen38-27b
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen38-27b
```

Use `claude --bare`: it keeps the system prompt and tool list small (about 1.3K tokens with Bash / Edit / Read),
so the first prefill takes about 16 s and the context limit (`--max-ctx 8192`) lasts several turns. When the
conversation no longer fits, the reply is the text `[streamdense] Context full ... Run /clear to start fresh.`

## Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /v1/messages` (`?beta=true` ignored) | SSE when `stream: true`, else one JSON message |
| `POST /v1/messages` with `max_tokens: 0` or header `x-streamdense-warm: 1` | warm-up: prefill only, JSON reply without content (below) |
| `POST /v1/messages/count_tokens` | `{"input_tokens": n}` of the rendered prompt |
| `GET /v1/models`, `GET /v1/models/{id}` | one model, `qwen38-27b` |
| `GET` / `HEAD` `/`, `/health` | `{"status":"ok",..}`; `info.warm: true` = warm-ups supported |
| `GET /v1/streamdense/watcher`, `POST /v1/streamdense/watcher/note` | domain watcher (below); `/v1/streamdense/watcher/vote` = 410 Gone |

One generation runs at a time; concurrent requests wait in a queue (an SSE `ping` goes out every 10 s while a
request waits; a client that went away cancels its request). Errors use the Anthropic error JSON
(`{"type":"error","error":{"type":"invalid_request_error",..}}`).

## Warm-up (prefill-only requests)

For a client that knows its next prompt prefix before the user types (e.g. right after it folds its history:
the next request would otherwise re-prefill 1.6-6.6K tokens, median 37 s). The client sends the folded request in
the background; the gateway reads it into the KV and the snapshot files, and the user's next request reuses it.

* **Call:** a normal `POST /v1/messages` body with `max_tokens: 0`, and/or the header `x-streamdense-warm: 1`
  (either one is enough; a client may send both, non-streaming). `stream` is ignored: the reply is always JSON.
  `GET /health` advertises `info.warm: true`.
* **What it does:** renders the messages **without** the generation prompt (the history ends in `<|im_end|>\n`,
  a token prefix of the same messages + a new turn + the generation prompt), plans KV reuse as usual (`prefix` /
  `restore` / `full`; never `followup`), prefills all but the last prompt token (left pending, as after a reply)
  and saves the same snapshots a real request would (stable prefix when new; the rolling state, prompt - 1 token,
  when the feed is at least 256 tokens). It decodes nothing. The live KV then holds the prompt, so the next real
  request with that prefix is a `prefix` hit; if another conversation ran in between, the rolling file restores it
  (`restore`, `prefix: own`). The per-job hook (watcher, answer library) does not see warm-ups.
* **Reply:** `200` `{"type":"message","content":[],"stop_reason":"max_tokens","usage":{"input_tokens": fed,
  "cache_read_input_tokens": reused, "output_tokens": 0},"streamdense":{"warm":"done"|"cancelled"|"ctx_full",
  "n_prompt","n_reused","n_feed","mode","prefix","queue_s","prefill_s","total_s","snap":[..]}}`. A cancelled
  warm-up is still a 200 (nothing to retry); errors are the normal error JSON. The request log line has
  `"warm": true` and `stop_reason` done / cancelled / ctx_full.
* **Priority (lowest):** the worker takes every queued job before picking one; real requests run first in arrival
  order, a warm-up only when no real request waits. **Any** real request cancels a warm-up in progress: the
  warm-up feeds in pieces (`Backend::warm_chunk`: `WARM_CHUNK` 48 tokens rounded up to whole prefill chunks, i.e.
  one 128-row chunk at prefill_t 128 / prefill_k 8, ~1.2-1.4 s on the 27B; 12 GB mode: one streamed piece) and
  checks its flag between pieces, so the real request waits at most one piece. A warm-up cancelled only by its own
  conversation keeps its fed pieces in the live KV (no file; that request plans `prefix` against them, same chunk
  boundaries as a one-shot prefill); cancelled by another conversation it resets the engine (its stable snapshot,
  if saved, stays). Picking a real request skips queued warm-ups of its own
  conversation (it prefills that prefix itself). Conversation = system prompt + tool names + first non-system
  message (`ChatReq::conv_key`, the message-level twin of the snapshot conv key).
* **Example:** `curl -s localhost:8898/v1/messages -H 'x-streamdense-warm: 1' -H 'content-type: application/json'
  -d '{"model":"qwen38-27b","max_tokens":0,"system":"..","messages":[..folded history..]}'`
* Tests: `tests/warm_tiny.rs` (tiny model: warm -> real = `prefix` with reused = warm prompt - 1, a fresh gateway
  restores the warm file, both replies == cold; fake engine: cancel by another conversation, queue order, same
  conversation reuse).

## Request mapping

* `system` blocks are joined with "\n"; the `x-anthropic-billing-header...` block is dropped (it changes with each
  Claude Code version and would break prefix reuse).
* user text -> user turn; `tool_result` (string or list content; `is_error` adds `[error] `) -> `tool` turn;
  assistant text + `tool_use` -> assistant turn with tool calls; `thinking` blocks -> reasoning; `tools` -> tool
  definitions in the template (server tools without `input_schema` are skipped).
* `thinking` (`enabled` / `adaptive`) -> `enable_thinking` only with `--think`; reasoning is never streamed.

## Streaming

Anthropic event order: `message_start` -> text `content_block_start` / `text_delta`... (visible text per token;
think blocks hidden; stops at `<tool_call>`; partial tags and incomplete UTF-8 are held back) ->
`content_block_stop` -> one `tool_use` block per call (`input_json_delta` with the arguments typed by the tool
schema) -> `message_delta` (`stop_reason` `end_turn` | `tool_use` | `max_tokens`; usage) -> `message_stop`.
Usage: `input_tokens` = prompt tokens fed this request, `cache_read_input_tokens` = tokens reused from the KV.

## KV reuse

The gateway keeps the token list of the engine state and the last request's messages:

* `prefix`: the new prompt's tokens start with the state's tokens -> only the rest is fed (the usual case with
  Claude Code: steps 2+ start in about 0.7-1.6 s instead of about 16 s, measured with the Python prototype);
* `followup`: same history up to our reply, then new turns, same tools and thinking setting -> feed `<|im_end|>`
  (if the reply did not end with it) + "\n" + the new turns; the model's own reply tokens stay in the state;
* `restore`: the planner said full, but a snapshot file covers a prefix of the prompt -> restore it, feed the rest;
* `full`: reset + prefill.

## Snapshots

The engine state (DeltaNet state + KV + MTP KV, `streamdense-model/src/snapshot.rs`) is saved to files and restored
before a full prefill, so a new Claude Code session or a retry does not prefill from zero
(`streamdense-gateway/src/snap.rs`, cut points in `worker.rs`).

* **Restore**: before a full prefill, the file covering the longest prefix of the prompt (minus its last token) is
  read straight into the engine (file mtime touched for LRU). Bit-exact (tested on the tiny model).
* **Stable prefix** (kept): the tokens through the first `<|im_end|>` = system prompt + tool defs, the same for
  every Claude Code session in the same setup. When a full prefill has no snapshot of it, the gateway feeds that
  prefix first, saves it, then feeds the rest.
* **Rolling last turn** (the newest 2 per conversation: this one + the idle save): when a request prefills at least 256
  tokens, the gateway stops one token before the end of the prompt and saves; an identical re-send (retry) then
  restores it and feeds 1 token. Shorter feeds are not saved (cheap to redo). Switching between two conversations no
  longer deletes the other one's file (before 2026-11-03: one rolling file per gateway).
* **Cap**: after each save the lowest-score `*.sdsnap` files are deleted until <= `[snapshots] max_files` (32) files,
  <= `max_disk_gb` (8 GB) and free disk >= `min_free_disk_gb` (20 GB) (`--snapshot-max-files` / `--snapshot-max-gb` win).
  No save at all while the disk is below the floor (logged as a `snap` error).
* Keys: model dir + file sizes/mtimes, structural fingerprint, KV format, MTP, engine version, token hash. A rebuild
  with a new version or changed weights never restores an old file.
* Size on the 27B: ~150 MB fixed + ~69 KB per token (239 MB at 1223 tokens). Measured: save 340-380 ms,
  restore 19-30 ms from the page cache (~80 ms expected from SSD).
* Each `/v1/messages` log line has `snap`: `[{kind: restore|save_stable|save_roll|save_switch, n, ms, mb, tier, src, error}]`
  (`tier`: restore from `hot` RAM copy or `warm` file; `src`: `own` = a file of the request's conversation, `shared` = the
  project prefix), `prefix` (see Multi-user) and `tiers` (idle work since the previous request, see below).

### Snapshot tiers

`streamdense-gateway/src/snap_tier.rs` (policy, index, pre-read thread) + `snap.rs` (engine side), config `[snapshots]`
in streamdense.toml:

| tier | where | rule |
|---|---|---|
| hot | RAM copy in the gateway process | only while live footprint + copy + `hot_headroom_gb` (1.0) <= `[ram] cap_gb`, total <= `hot_gb` (1.0); dropped first (LRU) when the footprint rises; the CPU heap limit moves by the copy size. None in 12 GB mode (no spare RAM). On the 27B (peak ~15.9 GB of 16) this is normally empty |
| warm | `.sdsnap` file on SSD | restore reads it into the engine (~0.4-0.7 s measured on the 27B; faster when pre-read into the file cache) |
| cold | deleted | score = last use + 600 s per hit (max 6) + 1800 s for a stable prefix; lowest first when over a cap or under the disk floor |

Index `tiers.json` in the snapshot dir keeps kind (stable / roll / idle), session keys, hits and last use across restarts.
Sessions: project = system turn (system prompt + tools), conversation = + the first user turn.

**Idle work** (gateway idle >= `idle_s` = 5 s: nothing running or queued), one step per wake-up of the worker:
1. `idle_save`: save the live state (prompt + reply) when it is >= `idle_save_min_tokens` (256) past that conversation's
   newest file, so the next turn after another session / a client side request / a restart restores it warm;
2. hot copy of it (if the RAM rule allows);
3. `preread`: the `sd-snaptier` thread (background QoS) reads the `preread_files` (2) highest-score files of other
   conversations (<= `preread_gb` 1.5) through the OS file cache in 4 MB chunks (deliberately no F_NOCACHE: the point
   is the cache, which does not count in the process footprint).

A request stops idle work at once: the HTTP thread bumps a pending counter before queueing; the pre-read checks it
before every 4 MB chunk, the worker before every step. The engine save itself is one call (~0.35 s at 240 MB, longer
for long states) and is not cut mid-file; a request that arrived during it shows the wait in `tiers.blocked_ms`.
Helper heap: the index (~200 B per file), one 4 MB buffer while pre-reading.

`/health` has `snapshots`: warm files / GB / caps, hot copies / GB / cap, free disk, hits (hot / warm / miss), saves, switch saves,
evictions, hot admitted / refused / dropped, idle (saves, save ms, pre-read files / GB / cancelled, blocked ms, pending
steps), live state (conversation, tokens, newest saved), top 8 files by score. Request log `tiers`: `idle_saves`,
`idle_save_ms`, `preread_files`, `preread_mb`, `preread_cancelled`, `hot_admitted`, `blocked_ms` (+ `evicted`).

### Multi-user (several agents, one gateway)

Several Claude Code style clients with the same system prompt + tools and their own conversations, interleaved
(A1 B1 C1 A2 ...), each start from the nearest prefix: own conversation > shared project prefix > cold.

* **Shared prefix**: the first full prefill of a project saves the system turn (`save_stable`); every other conversation
  with the same system + tools restores it, whatever the client. Only token-identical system turns share it (a cwd /
  date block inside the system prompt makes a new project).
* **Switch save** (`switch_save`, default on): when a request of another conversation is about to replace the live state
  (back to back: no idle period, so no idle save), the live state (prompt + reply) is saved first when it is >=
  `switch_save_min_tokens` (64) past that conversation's newest file (or the shared prefix). Costs one save (~0.15-0.6 s,
  ~300 MB at ~2K tokens on the 27B), paid by the incoming request. Kind `switch` in `tiers.json`, counted per
  conversation like the idle save (newest 2 kept); other users' files are never deleted except by the caps.
* **Parked conversations**: the worker keeps the KV bookkeeping (tokens + messages + tools) of up to 16 conversations
  whose state is in a file. A request that continues one (prefix or followup against it) restores that file and feeds
  only the new turns, exactly as if the user had been alone (the model's own reply tokens stay, as in `followup`; the
  re-rendered history need not re-tokenize to them). Otherwise the longest token-prefix file is restored.
* **Hot shared prefix** (`hot_stable`, default on): idle step after the live hot copy: a RAM copy of the live project's
  stable file, only if it fits in `hot_gb` next to the other copies and under the RAM rule. On this Mac (27B peak ~15.85
  of 16 GB) it is refused; on a 32 GB machine e.g. `[ram] cap_gb = 28`, `[snapshots] hot_gb = 2` keeps the shared
  prefix (~290 MB) + a live turn in RAM.
* Request log `prefix`: `kv` (live engine state: prefix / followup), `own` / `shared` (snapshot restore), `cold`.
* Measured (27B, 3 users x 4 turns, prefix 1,930 tokens): first user cold 23.2 s TTFT; the other two first turns
  0.53 / 1.18 s (shared, 28-32 tokens fed); turns 2-4 1.0-3.2 s (own, 43-84 fed, restore 21-118 ms). Peak footprint
  15.852 GB; 7 files / 2.16 GB on disk.

## Request log

One JSON line per `/v1/messages` request: `n_prompt`, `mode`, `n_feed`, `n_reused`, `n_new`, `queue_s`,
`ttft_s`, `prefill_s`, `decode_tok_s`, `total_s`, `stop_reason`, `engine_stop`, `tools` (names called), `mtp`,
`spec` (MTP acceptance), `snap` (snapshot restore / saves with ms, MB, tier and src), `prefix` (kv | own | shared | cold), `tiers` (idle snapshot work since
the previous request), `error`, `watcher` (one-line summary).

## 12 GB mode (research)

Off by default. Serves the 27B under a ~12 GB footprint instead of the 16 GB cap:

```toml
[forheader]
mode = "12gb"      # or: streamdense gateway --forheader 12gb
cap_gb = 12.0      # memory the model is sized for; [ram] cap_gb stays the hard kill limit
mtp = false        # MTP in this mode costs model memory (default off)
fast = true        # fast prompt read (default on; not bit-exact with the exact read): false or --no-fast = exact read
fused = true       # fused decode (default on; bit-identical output): false or --no-fused = old decode loop
```

It needs one input file, made once on the machine that has the research data and copied to the others:
`streamdense forheader export --data DIR` (or `streamdense forheader install FILE`) writes
`~/.streamdense/forheader/t1.sdt1`; the gateway stops with a clear error if it is missing or made for another model.
`streamdense gateway --forheader 12gb --plan` prints the memory plan and exits without loading the model.

Results so far (research harness, 27B, the same engine path the gateway uses):

| | 12 GB mode | full model |
|---|---|---|
| peak footprint, whole run incl. prompt reading | 11.76 GB (gateway, 27B smoke 2026-11-03) | 15.86 GB (live gateway, MTP on) |
| decode, plain (no MTP) | 19.7 tok/s with `fast` + `fused` (K 10,696, gateway, 101 prompts); 17.6 with both off (K 10,728, smoke) | 14.3 tok/s (same machine) |
| quality, deployed config: two prompt sets, 101 / 100 checkable prompts | 96 / 101 · 92 / 100 (every extra miss is code) | 100 / 101 · 95 / 100 |
| with MTP on (smaller resident part, K 9,776) | quality 91 / 101; peak 12.51 GB, over the 12 GB target | — |

Speed-ups on by default in this mode (2026-11-03, 27B, research harness, 13 prompts; full mode unchanged):

| | off | on |
|---|---|---|
| `fused` decode | 56.70 ms/token (17.6 tok/s) | 51.02 ms/token (19.6 tok/s); outputs identical 13/13, peak 11.900 GB |
| `fast` prompt read, TTFT at 1029 / 1506 / 1833 tokens | 21.9 / 32.4 / 39.4 s | 11.96 / 17.03 / 21.46 s; first token identical 13/13, peak 11.934 GB |

`fast` is not bit-exact with the exact read (GEMM summation order), and a snapshot made with one read is refused by the other.

Every prompt is read with the full model, so time to first token is longer than in the normal mode and each request
reads model data from SSD (the request log shows how much). Snapshots work as in the normal mode (separate files; a
snapshot of one mode is refused by the other). Turned off in this mode: the activity
recorder and the answer library (the domain watcher runs on text features only; no neuron store is created). KV reuse (`prefix` / `followup`)
works as usual. `/health` and every request-log line carry `ram_mode`: `mode`, `k`, `planned_gb`, `resident_gb`,
live `footprint_gb` and `peak_footprint_gb`, and per request `stream` (`pieces`, `tokens`, `ssd_gb`, `wall_s`, `gpu_s`, …). The gateway in this
mode has passed the tiny-model tests and a 27B smoke (2026-11-03, 101 everyday prompts: peak 11.76 GB, 19.7 tok/s with `fast` + `fused`, text identical to the research harness 101 / 101).

### Context drafts, fused verify, history index (2026-11-03, default off)

- `[forheader] ctx_draft = true` (or `--ctx-draft`): drafts from text already in the context (+ `[gateway] draft_table`),
  one exact check pass per block; output identical (27B: 111/111 prompts). Repeat-heavy prompts 19.3 -> 35.1 tok/s;
  everyday prompts unchanged. `--ctx-draft-dry` keeps its RAM (same K) and decodes plainly, for A/B.
- `[gateway] verify_fused` / `[forheader] verify_fused` (or `--verify-fused`): fused matmuls for check passes of 3..8
  rows; bit-identical. Full mode MTP auto 24.4 -> 27.2 tok/s; 12 GB repeat-heavy 35.1 -> 38.4 tok/s.
- `[history]` (`enabled`, `dir`, `budget_mb` 32, `project`, `sessions`; or `--history --history-dir D --history-project P`):
  a history index (`streamdense index build --project DIR --dir D`) as a third draft source of the 12 GB context
  drafts. Its RAM is in the 12 GB plan (K 10,664 -> 10,600). Not used by the full mode.

### DFlash drafter (2026-11-03, default off; full mode)

- `[gateway] drafter = "path/to/drafter"` (or `--drafter DIR`; `--no-drafter` wins): a trained DFlash block drafter
  (drafter weights trained separately) drafts 7 tokens in one forward from the target's own hidden states; the
  block is checked in ONE exact pass of up to 8 rows, so the text is identical to plain decode (engine `dflash.rs`).
- The drafter directory holds `dflash.json` + `dflash.safetensors` (4-bit g64 weights, norms as 1 + w), written by the
  training side's exporter. A raw training checkpoint dir is refused.
- With a drafter: MTP is not loaded (the drafter replaces it; `[gateway] mtp` / `--mtp` and `spec_depth` do not apply),
  and snapshots are off (snapshot files hold no drafter context). The 12 GB mode ignores it (its own decode loop).
- RAM (budget rows, so `--plan` shows them): drafter weights, its context KV (layers x 2 x kv heads x max_ctx x head dim,
  bf16), tap / block scratch, and an 8-row DeltaNet hold save. Request-log `spec` = (passes, accepted, rejects, acceptance).
- Status: tiny-model tests only (lossless vs plain greedy; drafter logits vs a numpy reference); no 27B run yet.

### DFlash2 drafter (2026-11-05, default off; experimental)

- `[gateway] dflash2 = "models/dflash2_27b_q4"` (or `--dflash2 DIR`; `--no-dflash2` wins): the z-lab drafter, MLX 4-bit
  file, one forward + candidate selector per round, one exact check of up to 8 rows (engine `dflash2.rs`). Text as plain
  decode; a failed draft ends that request with an error, with no fallback to another mode.
- Refused at start, before anything loads: with `--drafter`, with `--forheader 12gb`, weights not matching
  `[dflash2] expected_sha256`, wrong file shape, or a budget over the cap (`--abort-gb` / `[ram] cap_gb`; the cap is never
  raised). On the 27B the budget at 8192 ctx is 16.660 GB, so the default 16 GB cap refuses it. `--plan` shows the rows.
- With it on: MTP is not loaded, snapshots are off. `/health` and the request log get a `dflash2` field (state, note,
  live stats); the watcher summary moves from `watcher` to `dflash2.watcher` only while the option is on.
- Status: tiny-model tests only (lossless vs plain greedy; forward vs z-lab's MLX code,
  `tests/dflash2_accept/forward_check.py`, local test work, not in the repo); no 27B run in the engine yet.

## Domain watcher

Always on (`crates/streamdense-cli/src/watcher.rs`); there is no off switch, only `[watcher] mode = "observe"`
(default; `"unload"` is reserved for a later unload stage and observes until then). It decides the user's domain and reports, per
domain, what would unload. Nothing is unloaded yet.

- Inputs: every `/v1/messages` job after its reply went out (features of the newest user text, the tool results after
  it and the reply; only features + an 8-hex hash of the system prompt are kept), the activity recorder's data for
  that job (snapshot + reset; `--no-activity` = text features only), and notes from clients:
  `POST /v1/streamdense/watcher/note` `{"kind": "state_card"|"fold"|"turn", "text": "..", "project": "..", "mode": "chat"|"code"}`
  (`project` / `mode` optional) -> `{"ok": true, "decision": {..}|null, "domain": "..", "turn": n}`; 400 on a bad kind / mode.
- Domains are topics, not languages (2026-11-01): code-rust, code-swift, code-other, agent-tools, chat-general, mixed (union). The
  same topic in any language lands in the same domain. Kept languages (`keep_languages = "all-except-cut"`) are always hot; the
  cut ones (`cut_languages`, default zh ja ko tl id ms) are reserved as unload-first. Old chat-en / chat-vi / chat-other
  profiles are merged into chat-general on open (originals renamed `*.migrated`); old names in state / audits map to it.
- Every turn (job): rule scores of the job (tool results: code markers x2, prose neutral; fences / tool use on code lift code)
  blended 75 / 25 with the activity fingerprint centroids once two domains have >= 3 learned turns -> the turn's own label if
  top-1 leads top-2 by >= `margin` (0.15), else `unassigned`. The turn's activity is learned into that label only (never the
  active domain); its reply's audit is attributed to it, its emitted tokens + embedding rows go to `<label>.vocab`.
- Decision window: closes after `decide_window_folds` (3) fold notes once it has >= `decide_window_min_turns` (50) turns (no
  folds: at 2 x min turns). Samples = turns + state cards / folds (card / fold prose neutral). Mean scores; the top domain must
  lead the runner-up by >= `margin`; a new domain is adopted after `hysteresis` (1) such windows in a row; a sub-margin window
  keeps the count; no margin with the active domain outside the top two -> mixed. Fast topic changes = fallback (mask off).
- Plan: `unload::unload_plan(profile, keep_frac, granularity)`; mixed / unknown use the union of all labels incl. unassigned.
- `GET /v1/streamdense/watcher` -> `{domain, confidence (the active domain's share in the open window, else the last one),
  margin (its lead, < 0 = behind), confidence_from, since_turn, turn, pending, candidates: [{domain, score}], window: {turns,
  folds, samples, need_turns, need_folds, labels, learned, scores: {top, margin}, last_turn: {label, margin}},
  decisions_recent (newest first, <= 10; each with label, confidence, margin, runner_up, confident, window, learned),
  tokens_observed, unload_plan: {domain, tokens, ready, cold_ffn_frac, cold_heads, gb_unload (stage B), ms_saved_per_pass
  (stage A read mask), ms_saved_unload (stage B), footprint_gb (resident after B), fits_cap, per_layer_cold (neurons),
  per_layer_cold_heads}, profiles: [{domain, tokens, turns, gb_unload, representative (>= 200 K tokens)}] (+ unassigned),
  mode, activity: {source, jobs}, config, note: "dry run: nothing unloaded yet"}`.
- Audits (no user votes): every reply's message id is queued (bounded 64, a full queue drops its oldest;
  `domain` = the reply's own turn label, `active` = the domain active when it was generated) and audited newest first by the
  low-level watcher thread while the engine is idle (no recorder pass for >= 0.5 s). No read masks exist yet, so every
  audit is recorded `result: "skipped"`, `reason: "dry run: no mask active, reply came from the full model"`, `cause: null`; the
  real audit (exact K = 8 teacher-forced re-check with the full model, pausing on new requests) plugs in at `Watcher::audit_next` /
  `mask_active()`. Go-live: clean audit +1, (mask-caused) mismatch = reset + tighten; live = points >= `go_live_points` while masked.
  The GET view adds `scorecard: [{domain, audited_tokens, agreement (null until real audits), mismatches, skipped, clean_audits,
  go_live_at, points, required, resets, tightened, live}]`, `audits_recent` (oldest first, <= 20: `{message_id, domain, active, turn,
  result, reason, cause, tokens, diverge_at, full_answer}`) and `audit_queue: {pending, max, dropped, done, order}`.
  `/v1/streamdense/watcher/vote` (any method) -> 410 Gone. `like_weight` / `harm_multiplier` in old configs still load, ignored.
- Low-level watcher (`crates/streamdense-cli/src/heat.rs`; **the RAM maps below are replaced by
  the disk store, next bullet**): heat maps on three scales, now
  (tau 5 min), session (tau 4 h), all (no decay), for every FFN neuron (contribution |a| x ||W_down col||) / DeltaNet + attention
  head / layer resid, the MTP draft layer (FFN |a| + heads), output tokens emitted (lm_head rows, per id) and embedding rows read
  from SSD (per id). One data path: each job's single recorder window (`engine_activity`: `snapshot_lite` + the per-window
  u16 token counters taken sparse, `ActivityTap::take_tokens`, then reset) is folded into the now / session maps by the job
  hook (`Heat::fold`, ~7-9 ms per job at 27B sizes on the worker thread) and learned into the label files; no second baseline, no
  `prev` copy, no token logs. now / session = f16 maps, lazy decay (true = h x s x e^-(t-t0)/tau, renormalized past 16384) with
  stochastic rounding (unbiased: 5000 adds of ~1/5000 of the sum, mean ratio 0.998, sd 1.3%). "all" is not held in RAM: FFN rows and
  vocab / embed rows are summed from the per-label `<label>.neuron` / `<label>.vocab` files while streaming (70 KB buffers); heads +
  resid and the MTP layer keep small f32 running sums (in `heat/<tag>.bin`). Hot fractions come from a 4096-bin log histogram (one
  pass, no sort; exact for equal values). Thread `sd-heat` at background QoS (`pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND)`,
  efficiency cores) ticks every `[watcher] sample_ms` (100) on the live GPU counters only (activity / idle, in place), writes the
  summary 30 s after activity or at idle (~35 ms at 27B sizes on an E-core), drains the audit queue at idle (>= 0.5 s without a pass
  or fold) and saves the maps 5 min after activity at idle. RAM on the 27B (exact capacities): maps 6,653,632 B (main 2 x 1,116,864
  x 2 B, MTP 2 x 17,432 x 2 B, vocab + embed 4 x 248,320 x 2 B, all-time heads / resid + MTP 80,736 B, histogram 49,152 B) +
  recorder u16 token counters 993,280 B + scripts 248,320 B + MTP baseline 69,732 B = 7,964,964 B (was ~30.7 MB before the merge).
  Token counters saturate at 65,535 per id per window (`count_saturated`). Script classes: `streamdense_model::activity::
  script_classes` (10 classes: ascii latin cyrillic han kana hangul thai arabic special other; Vietnamese is latin), set once at
  gateway start (`tap.set_scripts`). `qwen serve` counts its emitted tokens too (`log_out` in both on_tok paths).
  GET view: `heat: {now|session|all: {tokens, hot_frac_mean, head_hot_frac_mean, per_layer_hot, coldest_layers: [[layer, frac]] (5),
  mtp: {tokens, ffn_hot_frac, head_hot_frac}, vocab, embed}, mtp|vocab|embed: {now|session|all: ..}, keep_frac, jobs_folded,
  all_from: {dir, neuron_labels}, samples, active_samples, cpu_s, cpu_pct_one_core, summary_cpu_ms, qos, ram_bytes, ram_mb,
  ram_total_bytes, ram_total_mb, count_saturated, loaded, audits_done, ..}`; hot = share of entries holding `keep_frac` of the mass;
  vocab / embed: `tokens, rows_hot, rows_hot_frac, rows_seen, scripts` (share per class). Files: `heat/<tag>.json` (summary),
  `heat/<tag>.bin` ("SDWHEAT2": f16 maps + scales + all-time sums; old SDWHEAT1 files are ignored). `streamdense watcher heat
  [--dir ..] [--layers]` prints the last summary.
- Heat store on disk (`crates/streamdense-cli/src/store.rs`): the watcher's and the heat
  maps' only store is disk, full f32, fixed size, updated in place, capped at 64 MB; host-heap RAM for the numbers = 0 (fixed
  stack buffers only: one 2048-f32 chunk + one 8,208 B record + a 12 KB radix histogram per streaming caller). All calculation /
  prediction (fold, summary, plan, centroid score, curve) streams from disk on a background thread. The GPU recorder region (model
  arena) is the only live accumulator.
  - Off the job hook (2026-11-01): the gateway's job hook only closes the recorder gate (`activity::GATE`) and hands the job to
    the `sd-heat` thread (`Shared::submit`, ~0.004 ms). sd-heat scores, labels and gates the job from the recorder in place, adds
    it into the batch region, zeroes the recorder and opens the gate. The next job's first recorder write (pass encode, token
    counter add) waits only while the gate is closed; `heat.live.next_job_waits` / `next_job_wait_ms` count those waits.
  - Batching (`[watcher] fold_every`, default 8): consecutive jobs with the same target are summed in a batch region in model
    memory (`QwenOptions.activity_batch`, one more recorder-sized region: 6.5 MB on the 27B). The batch is folded to disk in one
    write when the target changes, after `fold_every` jobs, after 30 s idle, and on SIGINT / SIGTERM (a second signal exits at
    once). Two labels are never mixed in one batch. The centroid is still updated per job. now / session take the batch at the
    rows-weighted mean job time. A crash loses at most the open batch (model memory) and never leaves the file half-written.
  - One file of fixed size (<= 64,000,000 B, written once at creation, never grows): `<watcher dir>/store/<Dims tag>_V<vocab>/store.bin`.
    Regions: slot table (A / B copies: slot -> label map, held-slot owner, unfinished-move state) | label slots | the held slot |
    now | session | down norms | journal. Label slots are all-time sums with no decay. now / session pool all labels, decayed with
    tau 5 min / 4 h, lazily per chunk. Exact offsets: see the module doc of `store.rs`.
  - Unified index layout of a set: [FFN contribution |a| x ||W_down col|| (layers x inter) | heads + resid (DeltaNet heads,
    attention heads, resid per layer) | MTP layer (|a| inter + heads) | emitted tokens (vocab, lm_head rows) | embedding rows
    read (vocab)].
  - Set = 4096 B header + chunks + (label / held slots only) the centroid.
    - Header: two copies A / B (offsets 0 / 2048), each `[magic | n | n_cent | set id | seq | tokens | turns | mtp_rows |
      out_tok | emb_rows (u64) | last_ts | t0 | rows_d | mtp_d | out_d | emb_d (f64) | checksum u64]`. A write goes to copy
      `seq % 2`; the reader takes the valid copy with the highest seq.
    - Chunk: 2048 f32 + a 16 B trailer `[t0 f64 | checksum u64]` (8,208 B).
    - Centroid: f32 [layers x blocks], EMA with alpha = max(1 / turns, 0.05).
  - Similarity gate (`[watcher] opp_similarity`, default 0.25):
    - `sim` = min(cosine of the job's block fingerprint vs the slot centroid, cosine of its head masses vs the slot's).
    - Below the threshold, the job is held: it goes into the held slot (+ now / session), not into the label's slot. A label
      without a slot is held the same way when no slot is free.
    - The held slot has one owner label at a time. Another label's held jobs wait in now / session only and are logged
      (`hold-wait`), not counted.
    - Confirmed after >= `decide_window_min_turns` held turns plus `decide_window_folds` folds since the first hold (or
      2 x min_turns). On confirm, the held slot is added into the label's slot (`Store::move_held`): every held job plus the
      confirming one ends up there. A label without a slot first replaces the slot with the lowest (usage share + similarity),
      never the active domain, the window's top label, or a slot whose centroid cosine to either is >= 0.8.
    - A similar job ends a hold: the held jobs stay in now / session only (`hold-ended`).
    - notes.jsonl gets `hold`, `hold-wait`, `hold-ended`, `hold-confirmed` and `replace`, each with its numbers. GET adds
      `held`, `held_slot`, `slots`, `batch`.
  - Crash safety:
    - Journal: `[region base | chunk | checksum]` + the pre-image of the label / held chunk being rewritten (written journal ->
      chunk). A torn chunk is restored from it when it matches, else zeroed and counted (`repaired` / `lost`).
    - Slot replace: unmap + journal invalidated -> region zeroed -> map.
    - Held move: the target's pre-move header goes into the table, then each held chunk is added and zeroed. An add whose held
      chunk was not zeroed yet is undone from the journal and redone. The move resumes on reopen, with no double count.
  - Down norms: "SDDOWN01" header + f32 [layers x inter], computed once at gateway start; layer 0 is re-checked on every start.
  - Migration: old `profiles/<tag>/*.bin` / `*.neuron` / `*.vocab` files are folded into the store once and renamed
    `*.migrated` (kept).
  - Sizes on the 27B (measured, `heat::tests::cost_27b`): 63,873,144 B, 126,856 B under the cap.

    | Region | Bytes |
    |---|---|
    | Table | 4,096 |
    | 6 label slots | 6,615,504 each |
    | Held slot | 6,615,504 |
    | now / session | 6,545,872 each |
    | Down norms | 4,460,544 |
    | Journal | 8,232 |

    Mixed is no longer a label, so 6 labels (5 domains + unassigned) + the held slot use the 7 slots.
  - Cost on the 27B (synthetic, release, `cost_27b_hook`):

    | | Inline fold (before) | Hand-off + batch, fold_every 8 (after) |
    |---|---|---|
    | Job-end delay | 115.9 ms/job | 0.004 ms/job |
    | sd-heat work | — | 39.8 ms/job (background QoS) |
    | Written per job | 26,275,424 B | 3,345,356 B |
    | At 300 replies/day | 7.88 GB/day | 1.00 GB/day |
- Read mask, CLI side (activity-only; `crates/streamdense-cli/src/predict.rs`, declared inside watcher.rs):
  `[watcher] mask = "off" | "dry" | "on"` (default "off"), `profile_window` (default "session+all"), `explore_frac` (default 0.02).
  - off: nothing is built, scored or written (no `predict.bin`); GET `mask` = `{"mode": "off", "applied": false}`.
  - dry: profile + predictor score + exploration schedule, reported; no mask applied (replies identical to off).
  - on: off by default; acts as dry until the engine's mask hook is wired (`MaskCtl::applied()` = false).
  - Profile rule: ONE profile pooled over all labels (domains are not an input). For each part of the window (now / session =
    the pooled decayed sets at the build clock, all = every label slot + the held slot), per layer, keep the 64-neuron FFN
    blocks and the heads holding `keep_frac` of that part's mass (`store::cold_units`, the `Store::plan` rule: coldest
    first while the removed mass <= (1 - keep) x total). Keep = union over the parts + the units exploration found active
    (`joined`). A part with no mass in a layer adds nothing; a layer with no mass anywhere keeps everything. Masked = never
    active in the window. Built by sd-heat on its read-only store after each summary (outside the watcher lock), installed
    under it. Mask format = the engine's (bit set = read): per layer u64 words of 64-blocks and of heads; `hot_blocks(l)` /
    `head_bits(l)` give the `unload.rs` list form. Dry numbers: masked blocks / heads, bytes skipped per pass (the
    `Store::plan` byte rule), ms per pass at 240 GB/s, per-layer counts.
  - CPU prompt predictor (stage A: scored only): `<store dir>/predict.bin`, fixed size <= 16,000,000 B (27B: 20,096 units
    = 17,408 FFN 64-blocks + 2,688 heads, 787 token buckets, 15.99 MB). Token ids hash into buckets; per bucket the requests
    it was in (u32) and per unit an u8 count of those where the unit was active (halved with the bucket count at 255); per
    unit an u32 count over all requests = P(u). Predicted: max(P(u), max over the request's 64 rarest known buckets
    (n >= 3) of (c + 2 P(u)) / (n + 2)) >= 0.5. Input = the job's new prompt tokens: today embedding rows read minus
    emitted tokens from the recorder (the fed, non-reused part; rejected draft rows included); truth = the engine's
    `prompt_active()` (this sequence's prefills; the gateway opens the engine with `read_mask` = true when mask != off, +2.9 MB
    model memory on the 27B, no profile set) when present, else the job's own keep set at `keep_frac` (recorder window). Each job: predict, score (precision / recall overall
    and on the masked units, bytes of the predicted masked units = what the CPU would add back / prefetch, bytes of the
    missed ones), then learn (P(u) + the 64 rarest buckets' rows: ~1.3 MB read + written per job on the 27B). GET
    `mask.predictor` (totals + `last`), notes.jsonl `kind: "predict"` every 8 scored requests (aggregate).
  - Exploration: at idle, every 60 s, a rotating share (`explore_frac` of the masked units, 1..=256) ranked by
    P(u) (most likely active first); a unit explored in round r is eligible again at r + ceil(masked / share), so the whole
    masked set rotates. Bookkeeping per unit in predict.bin (last round, found bit, joined round); found active -> joined
    (kept) for 240 rounds, then the store's own numbers decide. The GPU read is the `predict::Explorer` hook: never called
    with mask off or dry (dry reads every block anyway: no blind spot); with "on" and no hook the rounds are scheduled, not
    read. GET `mask.explore` (rounds, scheduled, read, found, expired, joined, last round with its first 8 units),
    notes.jsonl `kind: "explore"` when a round found active units.
  - `streamdense watcher [mask] [--dir DIR] [--model DIR] [--window W] [--keep K] [--sweep] [--json]`: the same profile
    built read only from the store (+ the joined units), the predictor's totals and the exploration counters. CPU only.
  - RAM: four fixed `ReadMask` sets in `MaskCtl` + one in sd-heat (~37 KB each, allocated once); 0 host heap on the build /
    score / learn / explore paths (`predict::tests::zero_heap_steady_state`).
- Request log + `/health`: `"watcher": "chat-general 0.29 (-0.17 vs code-rust; window 9/50 turns 0/3 folds) since t12 (t30); would unload 0.05 GB [dry run]"`.
- Disk (`dir`, default `~/.streamdense/watcher`, `--watcher-dir DIR` or `$SD_WATCHER_DIR`): `state.json` (incl. the open window, last window, go-live state),
  `profiles/<model tag>/<label>.bin` (64-blocks + heads + centroid; fixed size, atomic rewrite), `<label>.neuron` (per-neuron contribution
  sums f32, 4.46 MB per label on the 27B, streamed add per turn), `decisions.jsonl` (every decision + plan; last 2000 kept past 5000),
  `notes.jsonl` (note text <= 2 KB; last 50 kept past 100), `<label>.vocab` (f32 [2][vocab]: emitted-token | embedding-row counts, 2 MB per label on the 27B), `heat/` (heat maps), `audits.jsonl` (last 200 kept past
  400). RAM: block profiles ~0.2 MB per label on the 27B; per job ~4.5 MB transient (the recorder snapshot without a down-norm copy, reused in place; the recorder baseline is refilled in place) + sparse token lists (8 B per distinct id) + 2 x 64 KB file buffers.
- Unload curve (CPU only, read only, no model weights; `crates/streamdense-cli/src/watcher_curve.rs`, rewritten 2026-11-01 over
  the disk store): `streamdense watcher curve [--dir ~/.streamdense/watcher] [--model DIR] [--granularity 64] [--out FILE.json]
  [--min-tokens 200000]`. Opens the store of the model's dims read only (`--granularity` = `[watcher] granularity`, part of the
  store dir tag) and streams every number from disk through fixed stack buffers (no heap in the compute path; a unit test checks
  0 allocations). Per label (domains, mixed, unassigned) and pooled (all labels all-time, now, session; now / session decayed to
  the current time), keep 0.999 / 0.995 / 0.99 / 0.98 / 0.95 / 0.90 / 0.80 at g = 1 (exact per neuron, radix select) and g = 64
  (`Store::plan`): cold FFN share + GB, cold heads, A ms per pass (bytes / 240 GB/s; g = 1 assumes a gather kernel), B GB freed,
  tokens, turns, representative. Plus (all labels pooled): lm_head rows cold per keep level (rows, GB, per script),
  embedding rows cold (SSD GB), MTP draft layer cold FFN / head share (`store::hot`: the fewest largest rows holding `keep` of the
  mass; the rest is cold). Progress on stderr. `--out` streams the same as JSON. Reads the model dir's config + tokenizer only.
- Answer library (2026-11-03; `crates/streamdense-cli/src/library.rs`, format `crates/streamdense-core/src/library.rs`): **off by
  default**. Turn it on with `[watcher] library = true` (+ `library_cap_gb`, default 1.0; `library_text`, default false). One record
  per turn with >= 8 generated tokens: the prompt's and the answer's per-neuron FFN contribution |a| x ||W_down col|| (64 x 17,408
  on the 27B) as u8 log codes (16 per octave, per-layer max scale), a 1024-bucket count sketch per layer of the normalised prompt (for
  kNN), the turn's label and active domain, token / row counts and the MTP counters. No prompt text unless `library_text = true`
  (then the newest user message, <= 3,840 B).
  - Prompt / answer split: the worker calls `ActivityTap::mark_prompt` at the first generated token (prefill done, no decode pass
    encoded): one memcpy of the FFN sums (27B: 0.38 ms per turn). The answer = everything after it, i.e. decode + MTP verify rows
    (rejected drafts included: the recorder sums all rows of a pass; `answer_rows` / `n_new` shows their share). No GPU work added.
  - sd-heat (background QoS): under the closed recorder gate it swaps the mark out and computes answer = live - mark (27B: 0.7 ms);
    after the gate opens it encodes (27B: ~42 ms on an E-core) and writes the slot with one pwrite (~1 ms).
  - Disk: one ring file `<watcher dir>/library/library_L<layers>_I<inter>_S1024.sdl` (default `~/.streamdense/watcher/library/`):
    4 KB header + fixed slots of 2,367,488 B on the 27B (2.37 MB per turn, ~420 turns per GB, 2.37 GB per 1,000 turns). It grows to
    the cap and never past it; then the oldest record is replaced. Changing the cap resizes in place (shrink keeps the newest). Each
    slot has a payload checksum (torn writes are skipped by readers). A record that would grow the file is skipped while the disk
    has < 2 GB free.
  - RAM: ~16 MB host heap on the 27B, fixed at start (mark 4.5 + prompt 4.5 + answer 4.5 + slot 2.4 MB).
  - Status: GET `/v1/streamdense/watcher` -> `library: {on, file, records, slots, disk_bytes, cap_bytes, record_bytes, text,
    written, evicted, skipped_short, skipped_nomark, skipped_disk, errors, cost: {mark_worker_ms_*, capture_gate_ms_*,
    encode_bg_ms_*, write_bg_ms_*, heap_bytes}, line}` (`{"on": false}` when off); `/health` and the request log append
    `| library N rec, X MB / Y GB` to the watcher line.
  - Offline: `streamdense library stats [--dir DIR | --file F] [--verify] [--json]` (records, disk, labels, rows per token) and
    `streamdense library export [--out FILE] [--verify] [--text]` (metadata JSONL, no vectors). Checked offline on 12 real 27B prompts:
    the u8 library gives the same answer-mass scores as the f32 vectors to 3-4 decimals (H 0.7932 / P 0.7297 / LOO 0.7165 vs
    0.7932 / 0.7298 / 0.7165); sketch neighbours change kNN mass by <= 0.0004 vs exact cosines.
