#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a fresh clone and yellow-cube LeRobot dataset for pi0.5 LoRA fine-tuning,
# then runs the README-based training steps.
#
# Usage:
#   ./train.sh
#   RESET_STATE=1 ./train.sh   # rerun all steps from scratch (ignores saved step state)
#
# W&B setup (optional):
#   wandb login
#   # or: export WANDB_API_KEY=...
#   # Training in this script uses repo defaults (W&B enabled unless overridden).
#   # To disable W&B, add `--wandb-enabled=false` to the train command below.
#
# Optional overrides:
#   REPO_URL=https://github.com/Paul-LiPu/openpi.git \
#   BRANCH=exp/finetune \
#   REPO_DIR=openpi \
#   DATASET_URL=https://huggingface.co/datasets/zuichucai/pick-place-yellow_cube \
#   DATASET_DIR_NAME=pick-place-yellow_cube \
#   CONFIG_NAME=pi05_so101_low_mem_finetune \
#   EXP_NAME=my_yellow_cube_lora \
#   XLA_MEM_FRACTION=0.9 \
#   STATE_FILE=.openpi-train-state \
#   RESET_STATE=1 \
#   ./train.sh

REPO_URL="${REPO_URL:-https://github.com/Paul-LiPu/openpi.git}"
BRANCH="${BRANCH:-exp/finetune}"
DATASET_URL="${DATASET_URL:-https://huggingface.co/datasets/zuichucai/pick-place-yellow_cube}"
REPO_DIR="${REPO_DIR:-openpi}"
DATASET_DIR_NAME="${DATASET_DIR_NAME:-pick-place-yellow_cube}"
CONFIG_NAME="${CONFIG_NAME:-pi05_so101_low_mem_finetune}"
EXP_NAME="${EXP_NAME:-my_yellow_cube_lora}"
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.9}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
KEEP_PERIOD="${KEEP_PERIOD:-2500}"
CALLER_DIR="${PWD}"
STATE_FILE="${STATE_FILE:-${CALLER_DIR}/.${REPO_DIR}.train_state}"
RESET_STATE="${RESET_STATE:-0}"

mkdir -p "$(dirname "${STATE_FILE}")"
if [[ "${RESET_STATE}" == "1" ]]; then
  rm -f "${STATE_FILE}"
fi
touch "${STATE_FILE}"

step_done() {
  local step="$1"
  grep -Fxq "${step}" "${STATE_FILE}"
}

mark_step_done() {
  local step="$1"
  step_done "${step}" || echo "${step}" >> "${STATE_FILE}"
}

run_step() {
  local step="$1"
  local desc="$2"
  shift 2
  if step_done "${step}"; then
    echo "==> Skip (${step}): ${desc} [already completed]"
    return 0
  fi
  echo "==> ${desc}"
  "$@"
  mark_step_done "${step}"
}

clone_repo_step() {
  if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "Repo directory '${REPO_DIR}' already exists; skipping clone."
  else
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi
}

checkout_branch_step() {
  git fetch origin "${BRANCH}" || true
  if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
    git checkout -B "${BRANCH}" "origin/${BRANCH}"
  else
    git checkout "${BRANCH}"
  fi
}

clone_dataset_step() {
  mkdir -p data
  cd data
  if [[ -d "${DATASET_DIR_NAME}/.git" ]]; then
    echo "Dataset directory '${DATASET_DIR_NAME}' already exists; skipping clone."
  else
    git clone "${DATASET_URL}" "${DATASET_DIR_NAME}"
  fi
  cd ..
}

run_step "clone_repo" "Clone repo: ${REPO_URL}" clone_repo_step

cd "${REPO_DIR}"

run_step "checkout_branch" "Checkout branch: ${BRANCH}" checkout_branch_step

run_step "install_git_xet" "Install git-xet (required for Hugging Face git-xet repos)" \
  bash -lc 'curl -sSfL https://hf.co/git-xet/install.sh | sh'

run_step "clone_dataset" "Clone yellow-cube dataset into data/${DATASET_DIR_NAME}" clone_dataset_step

echo
echo "==> Setup complete"
echo "Repo root: $(pwd)"
echo "Dataset path: $(pwd)/data/${DATASET_DIR_NAME}"
echo "State file: ${STATE_FILE}"
echo
echo "Running: GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --group rlds --group dev"
run_step "install_deps" "Install Python dependencies" env GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --group rlds --group dev

echo "Running: uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}"
run_step "compute_norm_stats" "Compute norm stats (${CONFIG_NAME})" \
  uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

echo "Running: XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION} uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --resume --log-interval=${LOG_INTERVAL} --save-interval=${SAVE_INTERVAL} --keep-period=${KEEP_PERIOD}"
echo "==> Run training (${CONFIG_NAME}, exp=${EXP_NAME})"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION}" \
  uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --resume \
    --log-interval="${LOG_INTERVAL}" \
    --save-interval="${SAVE_INTERVAL}" \
    --keep-period="${KEEP_PERIOD}"

echo
echo "==> Training command finished"
echo "To serve a checkpoint later (README step 3), run:"
echo "uv run scripts/serve_policy.py policy:checkpoint --policy.config=${CONFIG_NAME} --policy.dir=checkpoints/${CONFIG_NAME}/${EXP_NAME}/<step>"
