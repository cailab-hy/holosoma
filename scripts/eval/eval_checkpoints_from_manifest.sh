#!/usr/bin/env bash
# =====================================================================
#  scripts/eval/eval_checkpoints_from_manifest.sh
#  ---------------------------------------------------------------
#  Read an eval manifest (YAML) and run `python -m holosoma.eval_agent`
#  for every (run, checkpoint_step) combination.
#
#  KEY GUARANTEES
#  --------------
#  * Always uses the canonical  --checkpoint <path>  flag.
#  * Refuses to launch if the generated command does NOT contain
#    "--checkpoint " (regression guard for the historical
#    "ValueError: No checkpoint provided" pitfall when users
#    accidentally passed --algo.config.policy-path).
#  * Missing checkpoints  →  STATUS = skip   (NOT failure).
#  * Failed eval invocations →  STATUS = fail   (recorded in
#    retry_failed.txt for re-run via --retry).
#  * Already-evaluated entries (eval_summary.json present and non-empty)
#    are skipped unless --force is given.
#
#  USAGE
#  -----
#    bash scripts/eval/eval_checkpoints_from_manifest.sh \
#         --manifest configs/eval/offline_rl_eval_manifest.yaml \
#         --out-root eval_out/step1 \
#         [--dry-run]               # show commands, do not execute
#         [--force]                 # re-eval even if results exist
#         [--retry]                 # only re-run entries from retry_failed.txt
#         [--only-experiment NAME]  # filter to one experiment_name
#         [--max-runs N]            # cap number of eval invocations
#
#  OUTPUTS  (under  <out-root>/_pipeline/ )
#  ----------------------------------------
#    STATUS.log              human-readable progress trace
#    eval_index.csv          one row per (experiment, step) with status
#    retry_failed.txt        manifest indices to retry
#    ALL_DONE.flag           created iff every non-skipped entry succeeded
#    logs/<exp>__<step>.log  stdout+stderr for each individual eval
#
#  Per-run eval results live under each entry's ``eval_output_dir``
#  (set in the manifest), in the standard
#  ``step_<STEP>/{eval_results.csv,eval_summary.json}`` layout
#  produced by holosoma.eval_agent.
# =====================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────
MANIFEST=""
OUT_ROOT=""
DRY_RUN=0
FORCE=0
RETRY=0
ONLY_EXP=""
MAX_RUNS=0

# ── CLI parsing ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)         MANIFEST="$2";   shift 2 ;;
    --out-root)         OUT_ROOT="$2";   shift 2 ;;
    --dry-run)          DRY_RUN=1;       shift ;;
    --force)            FORCE=1;         shift ;;
    --retry)            RETRY=1;         shift ;;
    --only-experiment)  ONLY_EXP="$2";   shift 2 ;;
    --max-runs)         MAX_RUNS="$2";   shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "[ERROR] --manifest is required" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] manifest not found: $MANIFEST" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="eval_out/step1"
fi
PIPE_DIR="${OUT_ROOT}/_pipeline"
mkdir -p "${PIPE_DIR}/logs"

STATUS_LOG="${PIPE_DIR}/STATUS.log"
INDEX_CSV="${PIPE_DIR}/eval_index.csv"
RETRY_TXT="${PIPE_DIR}/retry_failed.txt"
DONE_FLAG="${PIPE_DIR}/ALL_DONE.flag"

# Reset done flag at start of every invocation.
rm -f "$DONE_FLAG"

# Retry mode reads the prior retry_failed.txt before we touch it.
RETRY_KEYS=""
if [[ "$RETRY" -eq 1 ]]; then
  if [[ ! -f "$RETRY_TXT" ]]; then
    echo "[ERROR] --retry requested but $RETRY_TXT not found" >&2
    exit 2
  fi
  RETRY_KEYS="$(cat "$RETRY_TXT")"
  echo "[INFO] retry mode: $(echo "$RETRY_KEYS" | wc -l) entries to retry"
fi

# Initialise / append STATUS.log
{
  echo "================================================================="
  echo " eval pipeline run @ $(date -Iseconds)"
  echo "   manifest    : $MANIFEST"
  echo "   out_root    : $OUT_ROOT"
  echo "   dry_run     : $DRY_RUN"
  echo "   force       : $FORCE"
  echo "   retry       : $RETRY"
  echo "   only_exp    : ${ONLY_EXP:-<all>}"
  echo "================================================================="
} | tee -a "$STATUS_LOG"

# Initialise eval_index.csv header on first write.
if [[ ! -f "$INDEX_CSV" ]]; then
  echo "experiment,algorithm_name,group,step,status,checkpoint,eval_dir,log,timestamp" > "$INDEX_CSV"
fi

# Truncate retry_failed.txt at start; we rebuild it.
: > "$RETRY_TXT"

# ── Parse manifest via Python (yaml is already a project dep) ────
PARSED_PLAN="$(python3 - "$MANIFEST" "$ONLY_EXP" "$RETRY" "$RETRY_TXT" <<'PYEOF'
import sys, yaml, json, os
manifest_path, only_exp, retry_flag, retry_txt = sys.argv[1:5]
with open(manifest_path) as f:
    M = yaml.safe_load(f)
defaults = M.get("defaults", {}) or {}
plan = []
retry_keys = set()
if int(retry_flag) and os.path.exists(retry_txt):
    with open(retry_txt) as f:
        retry_keys = {ln.strip() for ln in f if ln.strip()}
for run in M.get("runs", []) or []:
    name = run["experiment_name"]
    if only_exp and name != only_exp:
        continue
    pat = run.get("checkpoint_pattern", "model_{step:07d}.pt")
    eval_root = run["eval_output_dir"]
    extra = run.get("eval_agent_args", []) or []
    for step in run.get("checkpoint_steps", []) or []:
        key = f"{name}::{step}"
        if retry_keys and key not in retry_keys:
            continue
        ck_name = pat.format(step=step)
        ck_path = os.path.join(run["run_dir"], ck_name)
        out_dir = os.path.join(eval_root, f"step_{step}")
        plan.append({
            "experiment_name": name,
            "algorithm_name": run["algorithm_name"],
            "algorithm_group": run["algorithm_group"],
            "checkpoint": ck_path,
            "step": step,
            "out_dir": out_dir,
            "num_envs":      run.get("num_eval_envs",     defaults.get("num_eval_envs", 256)),
            "num_episodes":  run.get("num_eval_episodes", defaults.get("num_eval_episodes", 600)),
            "seed":          run.get("eval_seed",         defaults.get("eval_seed", 20240601)),
            "headless":      defaults.get("headless", True),
            "torch_det":     defaults.get("torch_deterministic", True),
            "grasp_radius":  defaults.get("eval_grasp_radius", 0.12),
            "lift_margin":   defaults.get("eval_lift_height_margin", 0.05),
            "single_ep":     defaults.get("single_episode_per_env", True),
            "save_results":  defaults.get("save_eval_results", True),
            "extra_args":    extra,
        })
print(json.dumps(plan))
PYEOF
)"

# Each line is one JSON entry.
N=$(echo "$PARSED_PLAN" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "[INFO] planned $N (experiment,step) entries" | tee -a "$STATUS_LOG"
if [[ "$N" -eq 0 ]]; then
  echo "[WARN] empty plan — nothing to do" | tee -a "$STATUS_LOG"
  touch "$DONE_FLAG"
  exit 0
fi

count_ok=0
count_skip=0
count_fail=0
count_done=0
count_attempted=0

# Iterate using process-substitution to keep the parent shell counters.
while IFS= read -r ITEM; do
  count_done=$((count_done + 1))
  if [[ "$MAX_RUNS" -gt 0 && "$count_attempted" -ge "$MAX_RUNS" ]]; then
    echo "[INFO] --max-runs=$MAX_RUNS reached, stopping" | tee -a "$STATUS_LOG"
    break
  fi

  EXP=$(echo "$ITEM"      | python3 -c "import json,sys;print(json.load(sys.stdin)['experiment_name'])")
  ALGO=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['algorithm_name'])")
  GROUP=$(echo "$ITEM"    | python3 -c "import json,sys;print(json.load(sys.stdin)['algorithm_group'])")
  CKPT=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['checkpoint'])")
  STEP=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['step'])")
  OUTD=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['out_dir'])")
  NENV=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['num_envs'])")
  NEPS=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['num_episodes'])")
  SEED=$(echo "$ITEM"     | python3 -c "import json,sys;print(json.load(sys.stdin)['seed'])")
  HEADL=$(echo "$ITEM"    | python3 -c "import json,sys;print('True' if json.load(sys.stdin)['headless'] else 'False')")
  DET=$(echo "$ITEM"      | python3 -c "import json,sys;print('True' if json.load(sys.stdin)['torch_det'] else 'False')")
  GRADIUS=$(echo "$ITEM"  | python3 -c "import json,sys;print(json.load(sys.stdin)['grasp_radius'])")
  LMARGIN=$(echo "$ITEM"  | python3 -c "import json,sys;print(json.load(sys.stdin)['lift_margin'])")
  SINGLE_EP=$(echo "$ITEM"| python3 -c "import json,sys;print(json.load(sys.stdin)['single_ep'])")
  SAVE_RES=$(echo "$ITEM" | python3 -c "import json,sys;print(json.load(sys.stdin)['save_results'])")
  EXTRA_JSON=$(echo "$ITEM" | python3 -c "import json,sys;print(json.dumps(json.load(sys.stdin)['extra_args']))")

  KEY="${EXP}::${STEP}"
  LOG_FILE="${PIPE_DIR}/logs/${EXP}__step_${STEP}.log"
  TS=$(date -Iseconds)

  echo "" | tee -a "$STATUS_LOG"
  echo "── [$count_done/$N] ${EXP}  step=${STEP}" | tee -a "$STATUS_LOG"
  echo "    ckpt = $CKPT" | tee -a "$STATUS_LOG"
  echo "    out  = $OUTD" | tee -a "$STATUS_LOG"

  # 1) checkpoint existence
  if [[ ! -f "$CKPT" ]]; then
    echo "    STATUS=skip (checkpoint not found)" | tee -a "$STATUS_LOG"
    echo "${EXP},${ALGO},${GROUP},${STEP},skip,${CKPT},${OUTD},,${TS}" >> "$INDEX_CSV"
    count_skip=$((count_skip + 1))
    continue
  fi

  # 2) already evaluated?
  SUMMARY_FILE="${OUTD}/eval_summary.json"
  if [[ -s "$SUMMARY_FILE" && "$FORCE" -eq 0 ]]; then
    echo "    STATUS=cached (eval_summary.json present; use --force to re-run)" | tee -a "$STATUS_LOG"
    echo "${EXP},${ALGO},${GROUP},${STEP},cached,${CKPT},${OUTD},,${TS}" >> "$INDEX_CSV"
    count_ok=$((count_ok + 1))
    continue
  fi

  mkdir -p "$OUTD"

  # 3) build the command
  CMD=( python -m holosoma.eval_agent
        --checkpoint "$CKPT"
        --eval-results-dir "$OUTD"
        --checkpoint-step "$STEP"
        --eval-grasp-radius "$GRADIUS"
        --eval-lift-height-margin "$LMARGIN" )
  if [[ "$SINGLE_EP" == "True" ]]; then CMD+=( --single-episode-per-env );  fi
  if [[ "$SAVE_RES"  == "True" ]]; then CMD+=( --save-eval-results );        fi
  CMD+=( --eval-overrides.num-envs "$NENV"
         --eval-overrides.headless "$HEADL"
         --training.seed "$SEED"
         --training.torch-deterministic "$DET"
         --training.max-eval-steps "$NEPS" )

  # Append manifest extra args.
  EXTRA_ARGS=()
  while IFS= read -r a; do
    [[ -n "$a" ]] && EXTRA_ARGS+=( "$a" )
  done < <(python3 -c "import json,sys;print('\n'.join(json.loads(sys.argv[1])))" "$EXTRA_JSON")
  CMD+=( "${EXTRA_ARGS[@]}" )

  # 4) GUARD: --checkpoint must be in the command.  Also reject
  #    accidental --algo.config.policy-path overrides.
  CMD_STR="${CMD[*]}"
  if ! [[ "$CMD_STR" == *"--checkpoint "* ]]; then
    echo "    [GUARD-FAIL] generated command is missing '--checkpoint'." | tee -a "$STATUS_LOG"
    echo "    cmd: $CMD_STR" | tee -a "$STATUS_LOG"
    echo "${EXP},${ALGO},${GROUP},${STEP},fail_guard,${CKPT},${OUTD},${LOG_FILE},${TS}" >> "$INDEX_CSV"
    echo "$KEY" >> "$RETRY_TXT"
    count_fail=$((count_fail + 1))
    continue
  fi
  if [[ "$CMD_STR" == *"--algo.config.policy-path"* ]]; then
    echo "    [GUARD-FAIL] command contains --algo.config.policy-path; this is not a checkpoint flag. Use --checkpoint instead." | tee -a "$STATUS_LOG"
    echo "${EXP},${ALGO},${GROUP},${STEP},fail_guard,${CKPT},${OUTD},${LOG_FILE},${TS}" >> "$INDEX_CSV"
    echo "$KEY" >> "$RETRY_TXT"
    count_fail=$((count_fail + 1))
    continue
  fi

  echo "    cmd: $CMD_STR" >> "$LOG_FILE"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    STATUS=dry-run (would execute)" | tee -a "$STATUS_LOG"
    echo "${EXP},${ALGO},${GROUP},${STEP},dry_run,${CKPT},${OUTD},${LOG_FILE},${TS}" >> "$INDEX_CSV"
    continue
  fi

  # 5) execute
  count_attempted=$((count_attempted + 1))
  echo "    executing …" | tee -a "$STATUS_LOG"
  if "${CMD[@]}" >>"$LOG_FILE" 2>&1; then
    if [[ -s "$SUMMARY_FILE" ]]; then
      echo "    STATUS=ok" | tee -a "$STATUS_LOG"
      echo "${EXP},${ALGO},${GROUP},${STEP},ok,${CKPT},${OUTD},${LOG_FILE},${TS}" >> "$INDEX_CSV"
      count_ok=$((count_ok + 1))
    else
      echo "    STATUS=fail (eval_summary.json not produced)" | tee -a "$STATUS_LOG"
      echo "${EXP},${ALGO},${GROUP},${STEP},fail_nosummary,${CKPT},${OUTD},${LOG_FILE},${TS}" >> "$INDEX_CSV"
      echo "$KEY" >> "$RETRY_TXT"
      count_fail=$((count_fail + 1))
    fi
  else
    echo "    STATUS=fail (eval_agent returned non-zero)" | tee -a "$STATUS_LOG"
    echo "${EXP},${ALGO},${GROUP},${STEP},fail_rc,${CKPT},${OUTD},${LOG_FILE},${TS}" >> "$INDEX_CSV"
    echo "$KEY" >> "$RETRY_TXT"
    count_fail=$((count_fail + 1))
  fi
done < <(echo "$PARSED_PLAN" | python3 -c "import sys,json
for x in json.load(sys.stdin): print(json.dumps(x))")

echo ""                                                  | tee -a "$STATUS_LOG"
echo "================================================="  | tee -a "$STATUS_LOG"
echo " summary: ok=$count_ok  skip=$count_skip  fail=$count_fail (attempted=$count_attempted)" | tee -a "$STATUS_LOG"
echo "================================================="  | tee -a "$STATUS_LOG"

# ALL_DONE.flag iff no failures.
if [[ "$count_fail" -eq 0 ]]; then
  touch "$DONE_FLAG"
  echo "[OK] ALL_DONE.flag written → $DONE_FLAG" | tee -a "$STATUS_LOG"
  exit 0
else
  echo "[FAIL] $count_fail entries failed; see $RETRY_TXT" | tee -a "$STATUS_LOG"
  exit 1
fi
