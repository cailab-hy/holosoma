#!/usr/bin/env bash
# Replay a retargeted LAFAN clip window in the MuJoCo viewer.
#
# Converts <clip> frames [start, end] (inclusive, input fps) to the mj format and
# plays it with the real G1 model. No holosoma config registration needed -- use
# this to eyeball a window before committing it to a task config.
#
#   scripts/view_lafan_motion.sh dance1_subject3 172 412        # loop until closed
#   ONCE=1 scripts/view_lafan_motion.sh dance1_subject3 172 412  # play once, exit
#   OUT=/path/keep.npz scripts/view_lafan_motion.sh dance1_subject3 172 412
#
# The converted npz is a throwaway under ${TMPDIR:-/tmp} unless OUT is set. It is
# NOT re-centred -- scripts/recenter_wbt_motion.py does that before a clip ships.
set -Eeuo pipefail

CLIP="${1:?usage: $0 <clip-name> <start-frame> <end-frame>}"
START="${2:?missing start frame}"
END="${3:?missing end frame}"

RETARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/holosoma_retargeting/holosoma_retargeting" && pwd)"
PYTHON="${PYTHON:-$HOME/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
OUT="${OUT:-${TMPDIR:-/tmp}/view_${CLIP}_${START}_${END}_mj.npz}"

SRC="demo_results_parallel/g1/robot_only/lafan/${CLIP}_original.npz"
[[ -x "$PYTHON" ]] || { echo "[error] python not found: $PYTHON (set PYTHON=)" >&2; exit 1; }
[[ -f "$RETARGET_DIR/$SRC" ]] || {
  echo "[error] clip not found: $SRC" >&2
  echo "[hint] available:" >&2
  ls "$RETARGET_DIR/demo_results_parallel/g1/robot_only/lafan/" | sed 's/_original.npz//' | paste -sd' ' >&2
  exit 1
}

frames=$(( END - START + 1 ))
echo "[view] $CLIP f$START-$END  = $frames frames @ ${INPUT_FPS}fps = $(awk "BEGIN{printf \"%.2f\", ($frames-1)/$INPUT_FPS}")s"
echo "[view] close the viewer window to exit${ONCE:+ (ONCE set: exits after one pass)}"

cd "$RETARGET_DIR"
exec "$PYTHON" data_conversion/convert_data_format_mj.py \
  --input-file "$SRC" \
  --robot g1 --data-format lafan --object-name ground \
  --input-fps "$INPUT_FPS" --output-fps "$OUTPUT_FPS" \
  --line-range "$START" "$END" \
  ${ONCE:+--once} \
  --output-name "$OUT"
