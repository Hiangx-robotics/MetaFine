#!/usr/bin/env bash
set -euo pipefail

# Three-stage PI0.5 evaluation launcher for one task:
# 1) clean + DR profiles
# 2) instruction swap (clean only)
# 3) object swap (clean only)
#
# Example:
# bash core/policies/pi05/run_eval_three_stage.sh \
#   --policy-path /nat/demos/pi05/outputs/pi0_grasppart/checkpoints/030000/pretrained_model \
#   --env-id grasp_part \
#   --object-name 3558 \
#   --part-name cap \
#   --task "Grasp the cap of the bottle" \
#   --instruction-part handle \
#   --instruction-task "Grasp the handle of the bottle" \
#   --swap-object-name 100920 \
#   --swap-part-name button \
#   --swap-task "Press the button"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/evaluate.py"

usage() {
  cat <<'EOF'
Usage:
  bash core/policies/pi05/run_eval_three_stage.sh \
    --policy-path PATH \
    --env-id ENV_ID \
    --object-name OBJECT \
    --part-name PART \
    --task TASK_TEXT \
    [--instruction-part PART2] \
    [--instruction-task TASK2] \
    [--swap-object-name OBJECT2] \
    [--swap-part-name PART3] \
    [--swap-task TASK3] \
    [--n-episodes N] \
    [--device DEV] \
    [--obs-mode MODE] \
    [--control-mode MODE] \
    [--record-dir DIR] \
    [--max-episode-steps N] \
    [--camera-pos-levels "0.03 0.06 0.12"] \
    [--camera-rot-levels-deg "2 6 12"] \
    [--light-ambient-delta-levels "0.10 0.25 0.40"] \
    [--extra-eval-args "..."]

Required:
  --policy-path, --env-id, --object-name, --part-name, --task

Stage behavior:
  Stage 1 (always): clean + DR profiles
  Stage 2 (optional): only runs if --instruction-part or --instruction-task provided; clean only
  Stage 3 (optional): only runs if --swap-object-name provided; clean only

Notes:
  - Stage 2 defaults:
      instruction-part = current --part-name
      instruction-task = current --task with old part substring replaced by instruction-part (if possible)
  - Stage 3 defaults:
      swap-part-name = current --part-name
      swap-task = current --task
EOF
}

# ---------------------------
# Defaults
# ---------------------------
POLICY_PATH=""
ENV_ID=""
OBJECT_NAME=""
PART_NAME=""
TASK_TEXT=""

INSTRUCTION_PART=""
INSTRUCTION_TASK=""
SWAP_OBJECT_NAME=""
SWAP_PART_NAME=""
SWAP_TASK=""

N_EPISODES=10
DEVICE="cuda"
OBS_MODE="rgb"
CONTROL_MODE="pd_joint_delta_pos"
RECORD_DIR="./"
MAX_EPISODE_STEPS=""

CAMERA_POS_LEVELS="0.03 0.06 0.12"
CAMERA_ROT_LEVELS_DEG="2 6 12"
LIGHT_AMBIENT_DELTA_LEVELS="0.10 0.25 0.40"
EXTRA_EVAL_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) POLICY_PATH="${2:-}"; shift 2 ;;
    --env-id) ENV_ID="${2:-}"; shift 2 ;;
    --object-name) OBJECT_NAME="${2:-}"; shift 2 ;;
    --part-name) PART_NAME="${2:-}"; shift 2 ;;
    --task) TASK_TEXT="${2:-}"; shift 2 ;;

    --instruction-part) INSTRUCTION_PART="${2:-}"; shift 2 ;;
    --instruction-task) INSTRUCTION_TASK="${2:-}"; shift 2 ;;
    --swap-object-name) SWAP_OBJECT_NAME="${2:-}"; shift 2 ;;
    --swap-part-name) SWAP_PART_NAME="${2:-}"; shift 2 ;;
    --swap-task) SWAP_TASK="${2:-}"; shift 2 ;;

    --n-episodes) N_EPISODES="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --obs-mode) OBS_MODE="${2:-}"; shift 2 ;;
    --control-mode) CONTROL_MODE="${2:-}"; shift 2 ;;
    --record-dir) RECORD_DIR="${2:-}"; shift 2 ;;
    --max-episode-steps) MAX_EPISODE_STEPS="${2:-}"; shift 2 ;;
    --camera-pos-levels) CAMERA_POS_LEVELS="${2:-}"; shift 2 ;;
    --camera-rot-levels-deg) CAMERA_ROT_LEVELS_DEG="${2:-}"; shift 2 ;;
    --light-ambient-delta-levels) LIGHT_AMBIENT_DELTA_LEVELS="${2:-}"; shift 2 ;;
    --extra-eval-args) EXTRA_EVAL_ARGS="${2:-}"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${POLICY_PATH}" || -z "${ENV_ID}" || -z "${OBJECT_NAME}" || -z "${PART_NAME}" || -z "${TASK_TEXT}" ]]; then
  echo "[ERROR] Missing required args." >&2
  usage
  exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] evaluate.py not found at: ${EVAL_PY}" >&2
  exit 1
fi

run_eval() {
  local stage_name="$1"
  local object_name="$2"
  local part_name="$3"
  local task_text="$4"
  local enable_dr="$5"
  local out_dir="$6"

  local cmd=(
    python "${EVAL_PY}"
    --policy-path "${POLICY_PATH}"
    --env-id "${ENV_ID}"
    --object-name "${object_name}"
    --part-name "${part_name}"
    --obs-mode "${OBS_MODE}"
    --control-mode "${CONTROL_MODE}"
    --n-episodes "${N_EPISODES}"
    --device "${DEVICE}"
    --task "${task_text}"
    --record-dir "${out_dir}"
    --save-video
  )

  if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    cmd+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
  fi

  if [[ "${enable_dr}" == "1" ]]; then
    cmd+=(--enable-dr-eval)
    # shellcheck disable=SC2206
    local _cam_pos=(${CAMERA_POS_LEVELS})
    # shellcheck disable=SC2206
    local _cam_rot=(${CAMERA_ROT_LEVELS_DEG})
    # shellcheck disable=SC2206
    local _light=(${LIGHT_AMBIENT_DELTA_LEVELS})
    cmd+=(--camera-pos-levels "${_cam_pos[@]}")
    cmd+=(--camera-rot-levels-deg "${_cam_rot[@]}")
    cmd+=(--light-ambient-delta-levels "${_light[@]}")
  fi

  if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local _extra=(${EXTRA_EVAL_ARGS})
    cmd+=("${_extra[@]}")
  fi

  echo "============================================================"
  echo "[${stage_name}] object=${object_name} part=${part_name}"
  echo "[${stage_name}] task=${task_text}"
  echo "[${stage_name}] output=${out_dir}"
  echo "============================================================"
  "${cmd[@]}"
}

# ---------------------------
# Stage 1: clean + DR profiles
# ---------------------------
STAGE1_DIR="${RECORD_DIR}/stage1_clean_and_dr"
run_eval "stage1_clean_and_dr" "${OBJECT_NAME}" "${PART_NAME}" "${TASK_TEXT}" "1" "${STAGE1_DIR}"

# ---------------------------
# Stage 2: instruction swap (clean only)
# ---------------------------
if [[ -n "${INSTRUCTION_PART}" || -n "${INSTRUCTION_TASK}" ]]; then
  local_instruction_part="${INSTRUCTION_PART:-${PART_NAME}}"
  if [[ -n "${INSTRUCTION_TASK}" ]]; then
    local_instruction_task="${INSTRUCTION_TASK}"
  else
    # best-effort replace; if no match, fallback to original task
    local_instruction_task="${TASK_TEXT/${PART_NAME}/${local_instruction_part}}"
  fi
  STAGE2_DIR="${RECORD_DIR}/stage2_instruction_swap_clean"
  run_eval "stage2_instruction_swap_clean" "${OBJECT_NAME}" "${local_instruction_part}" "${local_instruction_task}" "0" "${STAGE2_DIR}"
fi

# ---------------------------
# Stage 3: object swap (clean only)
# ---------------------------
if [[ -n "${SWAP_OBJECT_NAME}" ]]; then
  local_swap_part="${SWAP_PART_NAME:-${PART_NAME}}"
  local_swap_task="${SWAP_TASK:-${TASK_TEXT}}"
  STAGE3_DIR="${RECORD_DIR}/stage3_object_swap_clean"
  run_eval "stage3_object_swap_clean" "${SWAP_OBJECT_NAME}" "${local_swap_part}" "${local_swap_task}" "0" "${STAGE3_DIR}"
fi

echo "All requested stages finished."

# 3763