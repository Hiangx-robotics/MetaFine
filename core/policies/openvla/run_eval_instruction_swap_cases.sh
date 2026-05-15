#!/usr/bin/env bash
set -euo pipefail

# Instruction-variation evaluation launcher (clean only).
# Focus: instruction text perturbation while keeping env/object/part fixed.
#
# Covers (clean-only):
# - grasp_part
# - toggle_switch_table (toggle case)
# - toggle_switch_table (press case)
# - rotate (base + clockwise + counterclockwise)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SH="${SCRIPT_DIR}/eval_fixed.sh"

if [[ ! -f "${EVAL_SH}" ]]; then
  echo "[ERROR] eval_fixed.sh not found at: ${EVAL_SH}" >&2
  exit 1
fi

N_EPISODES="${N_EPISODES:-10}"
OUT_ROOT="${OUT_ROOT:-/nat/demos/openvla/eval_instruction_swap_runs}"
RUN_CURRENT_INSTRUCTION="${RUN_CURRENT_INSTRUCTION:-1}"

# Override these before running when needed.
CKPT_GRASP_PART="${CKPT_GRASP_PART:-/nat/demos/openvla/openvla-7b+grasp_part_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--30000_chkpt}"
CKPT_TOGGLE_SWITCH="${CKPT_TOGGLE_SWITCH:-/nat/demos/openvla/openvla-7b+toggle_switch_2_rlds_100+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--30000_chkpt}"
CKPT_PRESS_SWITCH="${CKPT_PRESS_SWITCH:-/nat/demos/openvla/openvla-7b+press_switch_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--30000_chkpt}"
CKPT_ROTATE_ALONG="${CKPT_ROTATE_ALONG:-/nat/demos/openvla-7b+rotate_along_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt}"

mkdir -p "${OUT_ROOT}"

run_case() {
  local case_id="$1"
  local ckpt="$2"
  local env_id="$3"
  local object_name="$4"
  local part_name="$5"
  local instruction="$6"

  if [[ -z "${ckpt}" || ! -d "${ckpt}" ]]; then
    echo "[SKIP] ${case_id}: checkpoint not found -> ${ckpt}"
    return 0
  fi

  local out_dir="${OUT_ROOT}/${case_id}"
  mkdir -p "${out_dir}"
  echo "[RUN] ${case_id} | env=${env_id} | instruction='${instruction}'"
  OUTPUT_ROOT="${out_dir}" \
  ENABLE_DR_EVAL=0 \
  TASK_DESCRIPTION_OVERRIDE="${instruction}" \
  bash "${EVAL_SH}" "${ckpt}" "${env_id}" "${N_EPISODES}" "${object_name}" "${part_name}"
}

# 1) grasp_part: current instruction + part-name swap (env stays 3558/cap)
if [[ "${RUN_CURRENT_INSTRUCTION}" == "1" ]]; then
  run_case \
    "task1_grasp_part_instruction_current" \
    "${CKPT_GRASP_PART}" \
    "grasp_part" \
    "3558" \
    "cap" \
    "grasp the cap of the bottle"
fi

run_case \
  "task1_grasp_part_instruction_swap" \
  "${CKPT_GRASP_PART}" \
  "grasp_part" \
  "3558" \
  "cap" \
  "grasp the body of the bottle"

# 2) toggle_switch_table (toggle): current instruction + part-name swap
if [[ "${RUN_CURRENT_INSTRUCTION}" == "1" ]]; then
  run_case \
    "task2_toggle_switch_table_102812_instruction_current" \
    "${CKPT_TOGGLE_SWITCH}" \
    "toggle_switch_table" \
    "102812" \
    "button" \
    "toggle the button on the table"
fi

run_case \
  "task2_toggle_switch_table_102812_instruction_swap" \
  "${CKPT_TOGGLE_SWITCH}" \
  "toggle_switch_table" \
  "102812" \
  "button" \
  "toggle the lever on the table"

# 3) toggle_switch_table (press): current instruction + part-name swap
if [[ "${RUN_CURRENT_INSTRUCTION}" == "1" ]]; then
  run_case \
    "task3_toggle_switch_table_100979_instruction_current" \
    "${CKPT_PRESS_SWITCH}" \
    "toggle_switch_table" \
    "100979" \
    "button" \
    "press the button of the switch on the table"
fi

run_case \
  "task3_toggle_switch_table_100979_instruction_swap" \
  "${CKPT_PRESS_SWITCH}" \
  "toggle_switch_table" \
  "100979" \
  "button" \
  "press the top of theswitch on the table"

# 4) rotate: current instruction + clockwise / counterclockwise variants
if [[ "${RUN_CURRENT_INSTRUCTION}" == "1" ]]; then
  run_case \
    "task4_rotate_103062_instruction_current" \
    "${CKPT_ROTATE_ALONG}" \
    "rotate" \
    "103062" \
    "knob" \
    "rotate the knob clockwise by 90 degrees"
fi

run_case \
  "task4_rotate_103062_clockwise" \
  "${CKPT_ROTATE_ALONG}" \
  "rotate" \
  "103062" \
  "knob" \
  "rotate the knob clockwise"

run_case \
  "task5_rotate_103062_counterclockwise" \
  "${CKPT_ROTATE_ALONG}" \
  "rotate" \
  "103062" \
  "knob" \
  "rotate the knob counterclockwise"

echo "[DONE] Instruction-swap evaluation finished."
echo "[OUT] ${OUT_ROOT}"
