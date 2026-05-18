#!/usr/bin/env bash
# Paired eval runner: SMQR(anchor-only) vs vanilla CQL at multiple checkpoints.
#
# Prerequisite
# ------------
#   * Both runs must have been trained with the SAME offline dataset.
#   * Checkpoints are expected at:
#       runs/smqr_anchor_only/ckpt_<STEP>.pt
#       runs/vanilla_cql/ckpt_<STEP>.pt
#     If your paths differ, override via SMQR_CKPT_DIR / VANILLA_CKPT_DIR below.
#   * Run from repo root:  bash scripts/run_paired_eval.sh
#
# Output
# ------
#   eval_out/<algo>/step_<STEP>/eval_results.csv
#   eval_out/<algo>/step_<STEP>/eval_summary.json
#   eval_out/paired/step_<STEP>/{paired_per_env.csv,paired_summary.json,replay_candidates.json}
#
# Determinism
# -----------
#   Same EVAL_SEED + EVAL_NUM_ENVS across all 6 runs ⇒ env_id is a valid
#   paired key between SMQR and vanilla at each step.

set -euo pipefail

# ── Config (override via env vars) ───────────────────────────────
: "${EVAL_SEED:=20240601}"
: "${EVAL_NUM_ENVS:=256}"
: "${EVAL_MAX_STEPS:=600}"
: "${EVAL_GRASP_RADIUS:=0.12}"
: "${EVAL_LIFT_HEIGHT_MARGIN:=0.05}"
: "${EVAL_HEADLESS:=True}"
# Run without rendering (True/False).  Set to False to open the viewer.

: "${SMQR_CKPT_DIR:=logs/hv-g1-manager/exp_08_smqr}"
: "${VANILLA_CKPT_DIR:=logs/hv-g1-manager/exp_08_cql}"
: "${OUT_ROOT:=eval_out}"
: "${STEPS:=5000 10000 20000}"
# Checkpoint filename pattern, e.g. model_0005000.pt → zero-padded width 7.
: "${CKPT_PAD:=7}"
: "${CKPT_PREFIX:=model_}"
: "${CKPT_SUFFIX:=.pt}"

# ── Derived ──────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMMON_ARGS=(
  --single-episode-per-env
  --save-eval-results
  --eval-grasp-radius "$EVAL_GRASP_RADIUS"
  --eval-lift-height-margin "$EVAL_LIFT_HEIGHT_MARGIN"
)

# These are ExperimentConfig overrides parsed AFTER EvalRunConfig.
# ExperimentConfig is parsed with FlagConversionOff, so bools need True/False.
OVERRIDE_ARGS=(
  --eval-overrides.num-envs "$EVAL_NUM_ENVS"
  --eval-overrides.headless "$EVAL_HEADLESS"
  --training.seed "$EVAL_SEED"
  --training.torch-deterministic True
  --training.max-eval-steps "$EVAL_MAX_STEPS"
)

echo "=============================================================="
echo " Paired eval"
echo "   seed=$EVAL_SEED  num_envs=$EVAL_NUM_ENVS  max_steps=$EVAL_MAX_STEPS"
echo "   grasp_radius=$EVAL_GRASP_RADIUS  lift_margin=$EVAL_LIFT_HEIGHT_MARGIN"
echo "   headless=$EVAL_HEADLESS"
echo "   steps: $STEPS"
echo "=============================================================="

run_eval () {
  local algo="$1" ckpt="$2" step="$3" outdir="$4"
  echo ""
  echo "── eval [$algo @ step=$step] ─────────────────────────────"
  echo "    ckpt=$ckpt"
  echo "    out =$outdir"
  if [[ ! -f "$ckpt" ]]; then
    echo "  [SKIP] checkpoint not found: $ckpt" >&2
    return 0
  fi
  mkdir -p "$outdir"
  python -m holosoma.eval_agent \
    --checkpoint      "$ckpt" \
    --eval-results-dir "$outdir" \
    --algo-name       "$algo" \
    --checkpoint-step "$step" \
    "${COMMON_ARGS[@]}" \
    "${OVERRIDE_ARGS[@]}"
}

# ── 1. Run eval for each (algo, step) ────────────────────────────
for STEP in $STEPS; do
  PADDED="$(printf "%0${CKPT_PAD}d" "$STEP")"
  CKPT_NAME="${CKPT_PREFIX}${PADDED}${CKPT_SUFFIX}"

  run_eval "smqr_anchor_only" \
    "${SMQR_CKPT_DIR}/${CKPT_NAME}" \
    "$STEP" \
    "${OUT_ROOT}/smqr_anchor_only/step_${STEP}"

  run_eval "vanilla_cql" \
    "${VANILLA_CKPT_DIR}/${CKPT_NAME}" \
    "$STEP" \
    "${OUT_ROOT}/vanilla_cql/step_${STEP}"
done

# ── 2. Paired analysis per step ──────────────────────────────────
echo ""
echo "=============================================================="
echo " Paired analysis"
echo "=============================================================="
for STEP in $STEPS; do
  A="${OUT_ROOT}/smqr_anchor_only/step_${STEP}/eval_results.csv"
  B="${OUT_ROOT}/vanilla_cql/step_${STEP}/eval_results.csv"
  OUT="${OUT_ROOT}/paired/step_${STEP}"
  if [[ ! -f "$A" || ! -f "$B" ]]; then
    echo "  [SKIP] missing csv at step=$STEP (A=$A, B=$B)" >&2
    continue
  fi
  python scripts/eval_paired_analysis.py \
    --a       "$A" \
    --b       "$B" \
    --a-name  smqr_anchor_only \
    --b-name  vanilla_cql \
    --out     "$OUT" \
    --replay-top-k 10
done

echo ""
echo "Done.  Results under: ${OUT_ROOT}/"
