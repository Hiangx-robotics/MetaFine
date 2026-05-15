#!/usr/bin/env bash
set -euo pipefail

# New grasp_part checkpoint evaluation launcher.
# Stages:
#   - step1: clean + DR
#   - step2: object swap (clean only)
#   - step3: instruction/part perturbation (clean only)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SH="${SCRIPT_DIR}/eval_fixed.sh"

if [[ ! -f "${EVAL_SH}" ]]; then
  echo "[ERROR] eval_fixed.sh not found: ${EVAL_SH}" >&2
  exit 1
fi

N_EPISODES="${N_EPISODES:-10}"
OUT_ROOT="${OUT_ROOT:-/nat/demos/openvla/eval_manual_new_ckpt/press_part_100979_button_30000}"

CKPT="${CKPT:-/nat/demos/openvla/openvla-7b+press_switch_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--30000_chkpt}"

# Step1 base case
STEP1_OBJECT="${STEP1_OBJECT:-100979}"
STEP1_PART="${STEP1_PART:-button}"

# Step2 object swap (clean only)
STEP2_OBJECT="${STEP2_OBJECT:-100952}"
STEP2_PART="${STEP2_PART:-button}"

CAMERA_POS_LEVELS="${CAMERA_POS_LEVELS:-[0.03, 0.06, 0.12]}"
CAMERA_ROT_LEVELS_DEG="${CAMERA_ROT_LEVELS_DEG:-[2.0, 6.0, 12.0]}"
LIGHT_AMBIENT_DELTA_LEVELS="${LIGHT_AMBIENT_DELTA_LEVELS:-[0.10, 0.25, 0.40]}"

mkdir -p "${OUT_ROOT}"

echo "[RUN] press_part step1 (clean+DR)"
OUTPUT_ROOT="${OUT_ROOT}/stage1" \
ENABLE_DR_EVAL=1 \
CAMERA_POS_LEVELS="${CAMERA_POS_LEVELS}" \
CAMERA_ROT_LEVELS_DEG="${CAMERA_ROT_LEVELS_DEG}" \
LIGHT_AMBIENT_DELTA_LEVELS="${LIGHT_AMBIENT_DELTA_LEVELS}" \
bash "${EVAL_SH}" "${CKPT}" "toggle_switch_table" "${N_EPISODES}" "${STEP1_OBJECT}" "${STEP1_PART}"

echo "[RUN] grasp_part step2 (clean only, object swap)"
OUTPUT_ROOT="${OUT_ROOT}/stage2" \
ENABLE_DR_EVAL=0 \
bash "${EVAL_SH}" "${CKPT}" "toggle_switch_table" "${N_EPISODES}" "${STEP2_OBJECT}" "${STEP2_PART}"

echo "[DONE] Results under: ${OUT_ROOT}"
