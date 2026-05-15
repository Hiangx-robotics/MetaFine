#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for OpenVLA fixed evaluation.
# Stages:
#   - step1: clean + DR profiles
#   - step2: object swap (clean only)
#   - step3: instruction/part perturbation (reserved; not used in current 4 tasks)
#
# Outputs:
#   /nat/demos/openvla/eval_batch_runs/<task_id>/stage{1,2}/metrics_summary.json
#   /nat/demos/openvla/eval_batch_runs/summary_live.json
#   /nat/demos/openvla/eval_batch_runs/summary_live.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SH="${SCRIPT_DIR}/eval_fixed.sh"


OUT_ROOT="/nat/demos/openvla/eval_batch_runs"
SUMMARY_JSON="${OUT_ROOT}/summary_live.json"
SUMMARY_JSONL="${OUT_ROOT}/summary_live.jsonl"

N_EPISODES=10

# Optional checkpoint overrides (export before running).
# Keep defaults aligned with current OpenVLA naming convention.
CKPT_PEG_IN_HOLE=${CKPT_PEG_IN_HOLE:-"/nat/demos/openvla-7b+peg_in_hole_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt"}
CKPT_TOGGLE_SWITCH=${CKPT_TOGGLE_SWITCH:-"/nat/demos/openvla/openvla-7b+toggle_switch_rlds_100+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt"}
CKPT_PRESS_SWITCH=${CKPT_PRESS_SWITCH:-"/nat/demos/openvla-7b+press_switch_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt"}
CKPT_ROTATE_ALONG=${CKPT_ROTATE_ALONG:-"/nat/demos/openvla-7b+rotate_along_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt"}

# DR levels for step1
CAMERA_POS_LEVELS="[0.03, 0.06, 0.12]"
CAMERA_ROT_LEVELS_DEG="[2.0, 6.0, 12.0]"
LIGHT_AMBIENT_DELTA_LEVELS="[0.10, 0.25, 0.40]"

usage() {
  cat <<'EOF'
Usage:
  bash core/policies/openvla-oft/run_eval_batch_cases.sh [--n-episodes N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-episodes) N_EPISODES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${EVAL_SH}" ]]; then
  echo "[ERROR] eval_fixed.sh not found at: ${EVAL_SH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  cat > "${SUMMARY_JSON}" <<'EOF'
{
  "updated_at": "",
  "tasks": {}
}
EOF
fi

run_eval_once() {
  local ckpt="$1"
  local env_id="$2"
  local episodes="$3"
  local object_name="$4"     # "" => skip
  local part_name="$5"       # "" => skip
  local out_dir="$6"
  local enable_dr="$7"       # 1/0

  local cmd_env=(
    OUTPUT_ROOT="${out_dir}"
    OPENVLA_SUMMARY_JSON="${out_dir}/_internal_summary.json"
    OPENVLA_SUMMARY_JSONL="${out_dir}/_internal_summary.jsonl"
  )

  if [[ "${enable_dr}" == "1" ]]; then
    cmd_env+=(
      ENABLE_DR_EVAL=1
      CAMERA_POS_LEVELS="${CAMERA_POS_LEVELS}"
      CAMERA_ROT_LEVELS_DEG="${CAMERA_ROT_LEVELS_DEG}"
      LIGHT_AMBIENT_DELTA_LEVELS="${LIGHT_AMBIENT_DELTA_LEVELS}"
    )
  else
    cmd_env+=(ENABLE_DR_EVAL=0)
  fi

  mkdir -p "${out_dir}"

  if [[ -n "${object_name}" || -n "${part_name}" ]]; then
    env "${cmd_env[@]}" bash "${EVAL_SH}" "${ckpt}" "${env_id}" "${episodes}" "${object_name}" "${part_name}"
  else
    env "${cmd_env[@]}" bash "${EVAL_SH}" "${ckpt}" "${env_id}" "${episodes}"
  fi
}

update_live_summary() {
  local task_id="$1"
  local task_desc="$2"
  local task_root="$3"
  local stage1_metrics="$4"
  local stage2_metrics="$5"
  local stage3_metrics="$6"

  python - "${SUMMARY_JSON}" "${SUMMARY_JSONL}" "${task_id}" "${task_desc}" "${task_root}" "${stage1_metrics}" "${stage2_metrics}" "${stage3_metrics}" <<'PY'
import json
import os
import sys
from datetime import datetime

summary_json = sys.argv[1]
summary_jsonl = sys.argv[2]
task_id = sys.argv[3]
task_desc = sys.argv[4]
task_root = sys.argv[5]
stage1_path = sys.argv[6]
stage2_path = sys.argv[7]
stage3_path = sys.argv[8]

def load_json(path):
    if not path or path == "__NONE__" or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if os.path.exists(summary_json):
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {"updated_at": "", "tasks": {}}

now = datetime.now().isoformat(timespec="seconds")
entry = {
    "task_id": task_id,
    "description": task_desc,
    "task_root": task_root,
    "stage1_metrics_path": stage1_path if stage1_path != "__NONE__" else None,
    "stage2_metrics_path": stage2_path if stage2_path != "__NONE__" else None,
    "stage3_metrics_path": stage3_path if stage3_path != "__NONE__" else None,
    "stage1_metrics": load_json(stage1_path),
    "stage2_metrics": load_json(stage2_path),
    "stage3_metrics": load_json(stage3_path),
    "updated_at": now,
}

data.setdefault("tasks", {})[task_id] = entry
data["updated_at"] = now

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

with open(summary_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

echo "Live summary: ${SUMMARY_JSON}"

########################################
# Task 1: toggle_switch_table (step1 + step2)
########################################
TASK_ID="task1_toggle_switch_table_102812"
TASK_DESC="toggle_switch_table | step1(102812/button, clean+DR) + step2(100849/button, clean)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_TOGGLE_SWITCH}" \
  "toggle_switch_table" \
  "${N_EPISODES}" \
  "102812" \
  "button" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "${CKPT_TOGGLE_SWITCH}" \
  "toggle_switch_table" \
  "${N_EPISODES}" \
  "100849" \
  "button" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "${STAGE2_DIR}/metrics_summary.json" \
  "__NONE__"

########################################
# Task 2: peg_in_hole (step1 only)
########################################
TASK_ID="task2_peg_in_hole"
TASK_DESC="peg_in_hole | step1 only (clean+DR)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_PEG_IN_HOLE}" \
  "peg_in_hole" \
  "${N_EPISODES}" \
  "" \
  "" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

########################################
# Task 3: toggle_switch_table (step1 + step2)
########################################
TASK_ID="task3_toggle_switch_table_100979"
TASK_DESC="toggle_switch_table | step1(100979/button, clean+DR) + step2(100937/button, clean)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_PRESS_SWITCH}" \
  "toggle_switch_table" \
  "${N_EPISODES}" \
  "100979" \
  "button" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "${CKPT_PRESS_SWITCH}" \
  "toggle_switch_table" \
  "${N_EPISODES}" \
  "100937" \
  "button" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "${STAGE2_DIR}/metrics_summary.json" \
  "__NONE__"

########################################
# Task 4: rotate (step1 only)
########################################
TASK_ID="task4_rotate_103062"
TASK_DESC="rotate | step1 only (clean+DR, object=103062)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_ROTATE_ALONG}" \
  "rotate" \
  "${N_EPISODES}" \
  "103062" \
  "" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

echo "All tasks finished."
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Summary JSONL: ${SUMMARY_JSONL}"
