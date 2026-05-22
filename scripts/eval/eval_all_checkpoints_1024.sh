#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval/eval_all_checkpoints_1024.sh <checkpoint_dir> [output_dir] [num_envs] [max_eval_steps] [--limit N] [--retries N] [--retry-sleep SEC]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="${2:-}"
NUM_ENVS="${3:-1024}"
MAX_EVAL_STEPS="${4:-120}"
shift $(( $# >= 4 ? 4 : $# )) || true

LIMIT=0
RETRIES=2
RETRY_SLEEP=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --retry-sleep)
      RETRY_SLEEP="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint_dir not found: $CHECKPOINT_DIR" >&2
  exit 2
fi

CHECKPOINT_DIR="$(cd "$CHECKPOINT_DIR" && pwd)"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$CHECKPOINT_DIR/eval_1024_results"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

mapfile -t CHECKPOINTS < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.pt' | sort -V)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No .pt checkpoints found in: $CHECKPOINT_DIR" >&2
  exit 2
fi

COUNT=0
FAILURES=0
for CKPT in "${CHECKPOINTS[@]}"; do
  COUNT=$((COUNT + 1))
  if [[ "$LIMIT" -gt 0 && "$COUNT" -gt "$LIMIT" ]]; then
    break
  fi

  STEM="$(basename "$CKPT" .pt)"
  RESULT_DIR="$OUTPUT_DIR/$STEM"
  mkdir -p "$RESULT_DIR"

  STEP="$(python - <<'PY' "$CKPT"
from pathlib import Path
import re, sys
name = Path(sys.argv[1]).name
m = re.search(r'(\d+)', name)
print(int(m.group(1)) if m else -1)
PY
)"

  python - <<'PY' "$RESULT_DIR/eval_invocation.json" "$CKPT" "$NUM_ENVS" "$MAX_EVAL_STEPS" "$STEP"
import json, sys
out, ckpt, num_envs, max_eval_steps, step = sys.argv[1:6]
with open(out, 'w') as f:
    json.dump(
        {
            'checkpoint_path': ckpt,
            'num_envs': int(num_envs),
            'max_eval_steps': int(max_eval_steps),
            'checkpoint_step': int(step),
        },
        f,
        indent=2,
    )
PY

  ATTEMPT=1
  MAX_ATTEMPTS=$((RETRIES + 1))
  SUCCESS=0
  LAST_CMD_STATUS=1
  LAST_FAILURE_REASON="unknown"

  while [[ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]]; do
    ATTEMPT_LOG="$RESULT_DIR/console.attempt_${ATTEMPT}.log"
    echo "running eval for $CKPT (attempt $ATTEMPT/$MAX_ATTEMPTS)"

    set +e
    python -m holosoma.eval_agent \
      --checkpoint "$CKPT" \
      --checkpoint-step "$STEP" \
      --single-episode-per-env \
      --save-eval-results \
      --eval-results-dir "$RESULT_DIR" \
      --eval-overrides.num-envs "$NUM_ENVS" \
      --eval-overrides.headless True \
      --training.torch-deterministic True \
      --training.max-eval-steps "$MAX_EVAL_STEPS" \
      2>&1 | tee "$ATTEMPT_LOG"
    LAST_CMD_STATUS=${PIPESTATUS[0]}
    set -e

    cp "$ATTEMPT_LOG" "$RESULT_DIR/console.log"

    if [[ "$LAST_CMD_STATUS" -eq 0 && -s "$RESULT_DIR/eval_summary.json" ]]; then
      SUCCESS=1
      LAST_FAILURE_REASON=""
      break
    fi

    if [[ ! -s "$RESULT_DIR/eval_summary.json" ]]; then
      LAST_FAILURE_REASON="missing_eval_summary"
    else
      LAST_FAILURE_REASON="nonzero_exit_code"
    fi

    if [[ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]]; then
      sleep "$RETRY_SLEEP"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done

  if [[ "$SUCCESS" -eq 1 ]]; then
    echo "ok" > "$RESULT_DIR/eval_status.txt"
    rm -f "$RESULT_DIR/eval_failure.json"
  else
    echo "fail" > "$RESULT_DIR/eval_status.txt"
    FAILURES=$((FAILURES + 1))

    python - <<'PY' \
      "$RESULT_DIR/eval_failure.json" \
      "$CKPT" \
      "$STEP" \
      "$NUM_ENVS" \
      "$MAX_EVAL_STEPS" \
      "$LAST_CMD_STATUS" \
      "$MAX_ATTEMPTS" \
      "$LAST_FAILURE_REASON" \
      "$RESULT_DIR/console.log"
import json, re, sys
from pathlib import Path

(
    out_path,
    ckpt,
    step,
    num_envs,
    max_eval_steps,
    cmd_status,
    attempts,
    failure_reason,
    console_log,
) = sys.argv[1:]

def tail_error(path: Path) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    keep = []
    pat = re.compile(r"(Traceback|Exception|Error|ArgumentError|RuntimeError)")
    for line in lines:
        if pat.search(line):
            keep.append(line.strip())
    if not keep:
        return ""
    return keep[-1][:1000]

payload = {
    "checkpoint_path": ckpt,
    "checkpoint_step": int(step),
    "num_envs": int(num_envs),
    "max_eval_steps": int(max_eval_steps),
    "cmd_status": int(cmd_status),
    "attempts": int(attempts),
    "failure_reason": failure_reason,
    "error_excerpt": tail_error(Path(console_log)),
}

Path(out_path).write_text(json.dumps(payload, indent=2))
PY

    echo "[Batch Eval] checkpoint failed after retries: $CKPT" >&2
  fi

done

python scripts/eval/collect_checkpoint_eval_summaries.py "$OUTPUT_DIR"

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[Batch Eval] completed with failures: $FAILURES" >&2
  exit 1
fi

echo "[Batch Eval] completed successfully: $OUTPUT_DIR"
