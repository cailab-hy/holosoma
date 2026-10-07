#!/usr/bin/env bash
# Shared helpers for the paper validation sweeps (validation/**/*.sh).
#
# Each leaf script calls `run_seeds <run_name> <exp> [extra tyro args...]`, which launches
# offline_train_agent.py once per seed with `--training.name <run_name>_seed<N>`, one run at
# a time (or JOBS at a time), and skips seeds whose final checkpoint already exists under
# logs/WholeBodyTracking/*-<run_name>_seed<N>-locomotion/.
#
# Environment overrides (all optional):
#   SEEDS="1 2 3"        seeds to run (each leaf script sets its paper default)
#   JOBS=1               concurrent runs (an offline run needs roughly 10-18 GB of GPU memory)
#   SIMULATOR=isaacsim   simulator:<...> subcommand
#   LOGGER=wandb         logger:<...> subcommand (wandb | disabled)
#   EXTRA_ARGS=""        appended verbatim to every command, e.g. "--training.num-envs 1024"
#   DRY_RUN=1            print the commands without running them
#   FORCE=1              rerun even if the final checkpoint exists
#   FINAL_STEP=100000    checkpoint step that marks a finished run
#   ENTRY=...            training entry point (default: offline_train_agent.py)
#   RUN_LOG_DIR=...      where per-run console logs go (default: validation/logs)
set -euo pipefail

VALIDATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOSOMA_ROOT="$(cd "${VALIDATION_DIR}/.." && pwd)"
ENTRY="${ENTRY:-src/holosoma/holosoma/offline_train_agent.py}"
SIMULATOR="${SIMULATOR:-isaacsim}"
LOGGER="${LOGGER:-wandb}"
JOBS="${JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
FINAL_STEP="${FINAL_STEP:-100000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${VALIDATION_DIR}/logs}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${HOLOSOMA_ROOT}/logs/WholeBodyTracking}"

# Datasets (paths relative to the repo root, as the experiment configs expect).
G1_H5="offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5"
LAFAN_H5="offline_data/g1_29dof_wbt_lafan_dance1_fastsac_1m_episode_env256_dataset.h5"
KICK_H5="offline_data/g1_29dof_wbt_lafan_kick_fastsac_1m_episode_env256_dataset.h5"
KICK2_H5="offline_data/g1_29dof_wbt_lafan_kick2_fastsac_1m_episode_env256_dataset.h5"

FAIL_FILE="$(mktemp -t holosoma_validation_failed.XXXXXX)"
trap 'rm -f "${FAIL_FILE}"' EXIT

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

activate_env() {
  cd "${HOLOSOMA_ROOT}"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-hssim}" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${HOLOSOMA_ROOT}/scripts/source_isaacsim_setup.sh"
    set -u
  fi
  export OMNI_KIT_ACCEPT_EULA=1
  # Repo-local wandb account (see scripts/source_isaacsim_setup.sh); also applied when hssim was already active.
  if [[ -f "${HOLOSOMA_ROOT}/.wandb.env" ]]; then
    set -a; . "${HOLOSOMA_ROOT}/.wandb.env"; set +a
    log "wandb: repo-local account from .wandb.env (entity: ${WANDB_ENTITY:-<key default>})"
  fi
}

require_file() {  # <path> <hint>
  if [[ ! -f "$1" ]]; then
    log "missing required file: $1"
    log "  $2"
    exit 1
  fi
}

ensure_acl_sidecar() {  # <h5>
  local h5="$1" out="$1.acl_quality.npz"
  if [[ -f "${out}" ]]; then return 0; fi
  log "ACL quality sidecar missing, computing: ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "python scripts/acl_precompute_quality.py ${h5}"; return 0; fi
  python scripts/acl_precompute_quality.py "${h5}"
}

ensure_aw_sidecar() {  # <h5> <H> <out>
  local h5="$1" horizon="$2" out="$3"
  if [[ -f "${out}" ]]; then return 0; fi
  log "AW weight sidecar missing, computing (H=${horizon}): ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "python scripts/aw_precompute_weights.py ${h5} --H ${horizon} --out ${out}"; return 0; fi
  python scripts/aw_precompute_weights.py "${h5}" --H "${horizon}" --out "${out}"
}

ensure_aw_sidecar_global() {  # <h5> <H> <out>   global-baseline control (ignores phase)
  local h5="$1" horizon="$2" out="$3"
  if [[ -f "${out}" ]]; then return 0; fi
  log "AW global-baseline sidecar missing, computing (H=${horizon}): ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "python scripts/aw_precompute_weights.py ${h5} --H ${horizon} --baseline global --out ${out}"; return 0; fi
  python scripts/aw_precompute_weights.py "${h5}" --H "${horizon}" --baseline global --out "${out}"
}

ensure_vh_sidecar() {  # <h5> <H>   frozen V_H baseline ablation: writes <h5>.aw_weights.H<H>.vh.npz and .vh_presigma.npz
  local h5="$1" horizon="$2"
  local out="${h5}.aw_weights.H${horizon}.vh.npz"
  if [[ -f "${out}" && -f "${out/.vh.npz/.vh_presigma.npz}" ]]; then return 0; fi
  require_file "${h5}.aw_weights.H${horizon}.npz" "PRe sidecar needed for the shared G^H / sigma: run ensure_aw_sidecar first"
  log "V_H baseline sidecars missing, fitting cross-fitted V_H (H=${horizon}): ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "python scripts/vh_precompute_weights.py ${h5} --H ${horizon} --pre ${h5}.aw_weights.H${horizon}.npz"; return 0; fi
  python scripts/vh_precompute_weights.py "${h5}" --H "${horizon}" --pre "${h5}.aw_weights.H${horizon}.npz"
}

ensure_oper_sidecar() {  # <h5> <mode: odpr|ess|raw> <out>   ODPR-CQL weight sidecar built from OPER-A seeds 1-3
  local h5="$1" mode="$2" out="$3" seed
  if [[ -f "${out}" ]]; then return 0; fi
  local -a opers=() extra=()
  for seed in 1 2 3; do
    require_file "${h5}.oper_a.seed${seed}.npz" "run: python scripts/oper_precompute_weights.py ${h5} --seed ${seed} --out ${h5}.oper_a.seed${seed}.npz"
    opers+=("${h5}.oper_a.seed${seed}.npz")
  done
  [[ "${mode}" == "ess" ]] && extra=(--match-ess "${h5}.aw_weights.H50.npz")
  log "OPER-A sidecar missing, building (mode=${mode}): ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "python scripts/odpr_make_sidecar.py --oper ${opers[*]} --mode ${mode} ${extra[*]} --out ${out}"; return 0; fi
  python scripts/odpr_make_sidecar.py --oper "${opers[@]}" --mode "${mode}" "${extra[@]}" --out "${out}"
}

ensure_aw_sidecar_bins() {  # <h5> <H> <K> <out>   phase-baseline sidecar with K progress bins (bin-count sensitivity)
  local h5="$1" horizon="$2" k="$3" out="$4"
  if [[ -f "${out}" ]]; then return 0; fi
  log "AW sidecar with ${k} bins missing, computing (H=${horizon}): ${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "python scripts/aw_precompute_weights.py ${h5} --H ${horizon} --n-bins ${k} --out ${out}"; return 0; fi
  python scripts/aw_precompute_weights.py "${h5}" --H "${horizon}" --n-bins "${k}" --out "${out}"
}

run_finished() {  # <run_name>  -> 0 if a run dir with the final checkpoint exists
  local name="$1" d step5 step7
  step5="$(printf 'model_%05d.pt' "${FINAL_STEP}")"
  step7="$(printf 'model_%07d.pt' "${FINAL_STEP}")"
  for d in "${TRAIN_LOG_ROOT}"/*-"${name}"-locomotion; do
    [[ -d "${d}" ]] || continue
    if [[ -f "${d}/${step5}" || -f "${d}/${step7}" ]]; then return 0; fi
  done
  return 1
}

_launch() {  # <run_name> <log> <cmd...>
  local run="$1" logfile="$2"; shift 2
  if "$@" > "${logfile}" 2>&1; then
    log "[done] ${run}"
  else
    log "[FAIL] ${run} (see ${logfile})"
    echo "${run}" >> "${FAIL_FILE}"
  fi
}

run_seeds() {  # <run_name> <exp> [extra tyro args...]
  local name="$1" exp="$2"; shift 2
  local seed run logfile active=0
  local -a cmd extra
  # shellcheck disable=SC2206
  extra=(${EXTRA_ARGS})
  mkdir -p "${RUN_LOG_DIR}"
  for seed in ${SEEDS}; do
    run="${name}_seed${seed}"
    if [[ "${FORCE}" != "1" ]] && run_finished "${run}"; then
      log "[skip] ${run} (final checkpoint model_${FINAL_STEP} exists)"
      continue
    fi
    cmd=(python "${ENTRY}" "exp:${exp}" "simulator:${SIMULATOR}" "logger:${LOGGER}"
         --training.seed "${seed}" --training.name "${run}" "$@" "${extra[@]}")
    logfile="${RUN_LOG_DIR}/${run}.$(date +%Y%m%d_%H%M%S).log"
    log "[run ] ${run}: ${cmd[*]}"
    [[ "${DRY_RUN}" == "1" ]] && continue
    if [[ "${JOBS}" -gt 1 ]]; then
      _launch "${run}" "${logfile}" "${cmd[@]}" &
      active=$((active + 1))
      if [[ ${active} -ge ${JOBS} ]]; then wait -n; active=$((active - 1)); fi
    else
      if "${cmd[@]}" 2>&1 | tee "${logfile}"; then
        log "[done] ${run}"
      else
        log "[FAIL] ${run} (see ${logfile})"
        echo "${run}" >> "${FAIL_FILE}"
      fi
    fi
  done
  wait
}

run_scripts() {  # <script.sh>...  run leaf scripts sequentially, keep going on failure
  local s
  for s in "$@"; do
    log "===== $(basename "${s}") ====="
    if ! bash "${s}"; then echo "$(basename "${s}")" >> "${FAIL_FILE}"; fi
  done
}

finish() {
  if [[ -s "${FAIL_FILE}" ]]; then
    log "FAILED runs:"; sed 's/^/  /' "${FAIL_FILE}"
    exit 1
  fi
  log "all runs finished (or skipped)"
}
