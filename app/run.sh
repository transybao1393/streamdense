#!/bin/bash
# Native macOS chat app for the streamdense Qwen engine.
#   app/run.sh [--model DIR] [--max-ctx 4096] [--no-mtp] [--skip-gate] [--autoload] [--autosend TEXT]
#   --autoload loads the model at startup; --autosend TEXT (debug) also sends TEXT as the first message.
#   The app log (engine stderr, watchdog, per-reply done lines) is mirrored to stdout.
# 1) RAM gate (available = free+inactive+speculative+purgeable >= 20 GB, swap used <= 3 GB); --skip-gate opens the
#    window anyway (the app repeats the gate before every model load)
# 2) builds the app (swift build -c release) if the binary is missing or older than the sources
# 3) launches it. Nothing is loaded until "Load model" or the first Send; the engine runs under
#    sd-watchdog --cap-gb 16.5 --margin-gb 0 --swap-gb 7 with nice -n 10. Quitting the app (Cmd+Q, closing the
#    window, Ctrl+C here) stops the engine and its watchdog so the RAM is freed.
# Env overrides: STREAMDENSE_ROOT (repo root), STREAMDENSE_BIN (engine binaries dir), STREAMDENSE_MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export STREAMDENSE_ROOT="${STREAMDENSE_ROOT:-$(cd "$HERE/.." && pwd)}"
BIN_DIR="${STREAMDENSE_BIN:-$STREAMDENSE_ROOT/engine/target/release}"
APP_BIN="$HERE/.build/release/StreamdenseChat"

SKIP_GATE=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--skip-gate" ]; then SKIP_GATE=1; else ARGS+=("$a"); fi
done

for b in streamdense sd-watchdog; do
  [ -x "$BIN_DIR/$b" ] || echo "warning: missing $BIN_DIR/$b (run 'cargo build --release --offline' in engine/); the app opens but cannot load"
done

pg=$(vm_stat | awk '/page size of/ {print $8}')
read -r free inact spec purg < <(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s); gsub(/\./,"",p); print f, i, s, p}')
avail=$(echo "scale=2; ($free + $inact + $spec + $purg) * $pg / 1000000000" | bc)
swap=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M)
swap_gb=$(echo "scale=2; $swap * 1048576 / 1000000000" | bc)
echo "RAM gate: available ${avail} GB (need >= 20), swap used ${swap_gb} GB (need <= 3)"
if [ "$(echo "$avail < 20" | bc)" = 1 ] || [ "$(echo "$swap_gb > 3" | bc)" = 1 ]; then
  if [ "$SKIP_GATE" = 1 ]; then
    echo "gate STOP, opening the window anyway (--skip-gate); loading stays blocked until RAM frees up"
  else
    echo "STOP: not enough free RAM for the 27B. Close apps and try again (or --skip-gate to open the window only)."; exit 2
  fi
fi

if [ ! -x "$APP_BIN" ] || [ -n "$(find "$HERE/Sources" "$HERE/Package.swift" -newer "$APP_BIN" -print -quit)" ]; then
  echo "building the app (swift build -c release)..."
  (cd "$HERE" && swift build -c release)
fi

echo "launching $APP_BIN (quit the app or Ctrl+C here to stop the engine and free its RAM)"
exec "$APP_BIN" ${ARGS[@]+"${ARGS[@]}"}
