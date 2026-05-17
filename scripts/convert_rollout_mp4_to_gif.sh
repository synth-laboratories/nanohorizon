#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  cat >&2 <<'EOF'
usage: scripts/convert_rollout_mp4_to_gif.sh INPUT.mp4 OUTPUT.gif [WIDTH]

Converts a rollout MP4 into a README-friendly GIF using ffmpeg palettegen/paletteuse.
Default WIDTH is 352 to match the current README asset's native width.
EOF
  exit 2
fi

INPUT_MP4="$1"
OUTPUT_GIF="$2"
WIDTH="${3:-352}"
FFMPEG_BIN="${FFMPEG_BIN:-$(command -v ffmpeg || true)}"

if [[ -z "$FFMPEG_BIN" ]]; then
  echo "ffmpeg not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$INPUT_MP4" ]]; then
  echo "missing input mp4: $INPUT_MP4" >&2
  exit 1
fi

PALETTE_PATH="$(mktemp "${TMPDIR:-/tmp}/nanohorizon_rollout_palette.XXXXXX.png")"
cleanup() {
  rm -f "$PALETTE_PATH"
}
trap cleanup EXIT

"$FFMPEG_BIN" -y \
  -i "$INPUT_MP4" \
  -vf "fps=8,scale=${WIDTH}:-1:flags=lanczos,palettegen" \
  -frames:v 1 \
  -update 1 \
  "$PALETTE_PATH"

"$FFMPEG_BIN" -y \
  -i "$INPUT_MP4" \
  -i "$PALETTE_PATH" \
  -lavfi "fps=8,scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  "$OUTPUT_GIF"

echo "$OUTPUT_GIF"
