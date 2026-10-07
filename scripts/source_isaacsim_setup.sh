# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate $CONDA_ENV_NAME
export OMNI_KIT_ACCEPT_EULA=1

# Repo-local Weights & Biases account: if <repo>/.wandb.env exists (gitignored), its WANDB_API_KEY /
# WANDB_ENTITY override the machine-wide login in ~/.netrc for anything launched from this repo's env.
_HOLOSOMA_WANDB_ENV="$(cd "${SCRIPT_DIR}/.." && pwd)/.wandb.env"
if [ -f "${_HOLOSOMA_WANDB_ENV}" ]; then
  set -a; . "${_HOLOSOMA_WANDB_ENV}"; set +a
  echo "wandb: using repo-local account settings from ${_HOLOSOMA_WANDB_ENV} (entity: ${WANDB_ENTITY:-<key default>})"
fi
