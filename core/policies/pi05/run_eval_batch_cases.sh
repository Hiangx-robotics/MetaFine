#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for 5 PI0.5 evaluation tasks.
# - All outputs go under /nat/demos/pi05
# - After each task finishes, live summary is updated immediately
#
# Usage:
#   bash core/policies/pi05/run_eval_batch_cases.sh
#   bash core/policies/pi05/run_eval_batch_cases.sh --n-episodes 10 --device cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/evaluate.py"

OUT_ROOT="/nat/demos/pi05/eval_batch_runs"
SUMMARY_JSON="${OUT_ROOT}/summary_live.json"
SUMMARY_JSONL="${OUT_ROOT}/summary_live.jsonl"

N_EPISODES=10
DEVICE="cuda"
OBS_MODE="rgb"
CONTROL_MODE="pd_joint_delta_pos"
MAX_EPISODE_STEPS=""
EXTRA_EVAL_ARGS=""

CAMERA_POS_LEVELS="0.03 0.06 0.12"
CAMERA_ROT_LEVELS_DEG="2 6 12"
LIGHT_AMBIENT_DELTA_LEVELS="0.10 0.25 0.40"

usage() {
  cat <<'EOF'
Usage:
  bash core/policies/pi05/run_eval_batch_cases.sh \
    [--n-episodes N] \
    [--device DEV] \
    [--obs-mode MODE] \
    [--control-mode MODE] \
    [--max-episode-steps N] \
    [--camera-pos-levels "0.03 0.06 0.12"] \
    [--camera-rot-levels-deg "2 6 12"] \
    [--light-ambient-delta-levels "0.10 0.25 0.40"] \
    [--extra-eval-args "..."]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-episodes) N_EPISODES="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --obs-mode) OBS_MODE="${2:-}"; shift 2 ;;
    --control-mode) CONTROL_MODE="${2:-}"; shift 2 ;;
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

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] evaluate.py not found at: ${EVAL_PY}" >&2
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
  local policy_path="$1"
  local env_id="$2"
  local object_name="$3"    # "" means do not pass
  local part_name="$4"      # "" means do not pass
  local task_text="$5"
  local record_dir="$6"
  local enable_dr="$7"      # 1/0

  local cmd=(
    python "${EVAL_PY}"
    --policy-path "${policy_path}"
    --env-id "${env_id}"
    --obs-mode "${OBS_MODE}"
    --control-mode "${CONTROL_MODE}"
    --n-episodes "${N_EPISODES}"
    --device "${DEVICE}"
    --task "${task_text}"
    --record-dir "${record_dir}"
    --save-video
  )

  if [[ -n "${object_name}" ]]; then
    cmd+=(--object-name "${object_name}")
  fi
  if [[ -n "${part_name}" ]]; then
    cmd+=(--part-name "${part_name}")
  fi
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

  mkdir -p "${record_dir}"
  "${cmd[@]}"
}

update_live_summary() {
  local task_id="$1"
  local task_desc="$2"
  local task_root="$3"
  local stage1_metrics="$4"
  local stage2_metrics="$5"

  python - "${SUMMARY_JSON}" "${SUMMARY_JSONL}" "${task_id}" "${task_desc}" "${task_root}" "${stage1_metrics}" "${stage2_metrics}" <<'PY'
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
    "stage1_metrics": load_json(stage1_path),
    "stage2_metrics": load_json(stage2_path),
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
# Task 1: step1 only (no object/part)
########################################
TASK_ID="task1_peg_in_hole"
TASK_DESC="peg_in_hole | step1 only | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR=""
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_peginhole/checkpoints/020000/pretrained_model" \
  "peg_in_hole" \
  "" \
  "" \
  "Insert the peg into the hole." \
  "${STAGE1_DIR}" \
  "1"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/peg_in_hole/metrics_summary.json" \
  "__NONE__"

########################################
# Task 2: step1 + step2 (object swap)
########################################
TASK_ID="task2_toggle_switch_table_togglepart"
TASK_DESC="toggle_switch_table(102812/button) | step1+step2(object->100849)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_togglepart/checkpoints/030000/pretrained_model" \
  "toggle_switch_table" \
  "102812" \
  "button" \
  "toggle the button of the switch" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_togglepart/checkpoints/030000/pretrained_model" \
  "toggle_switch_table" \
  "100849" \
  "button" \
  "toggle the button of the switch" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/toggle_switch_table/metrics_summary.json" \
  "${STAGE2_DIR}/toggle_switch_table/metrics_summary.json"

########################################
# Task 3: step1 + step2 (object swap)
########################################
TASK_ID="task3_toggle_switch_table_press"
TASK_DESC="toggle_switch_table(100979/button) | step1+step2(object->100937)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_press/checkpoints/020000/pretrained_model" \
  "toggle_switch_table" \
  "100979" \
  "button" \
  "Press the button of the switch." \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_press/checkpoints/020000/pretrained_model" \
  "toggle_switch_table" \
  "100937" \
  "button" \
  "Press the button of the switch." \
  "${STAGE2_DIR}" \
  "0"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/toggle_switch_table/metrics_summary.json" \
  "${STAGE2_DIR}/toggle_switch_table/metrics_summary.json"

########################################
# Task 4: step1 + step2 (object swap)
########################################
TASK_ID="task4_rotate_knob"
TASK_DESC="rotate(103062/knob) | step1+step2(object->102901)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_rotate_along/checkpoints/020000/pretrained_model" \
  "rotate" \
  "103062" \
  "knob" \
  "Rotate the knob to complete the task." \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_rotate_along/checkpoints/020000/pretrained_model" \
  "rotate" \
  "102901" \
  "knob" \
  "Rotate the knob to complete the task." \
  "${STAGE2_DIR}" \
  "0"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/rotate/metrics_summary.json" \
  "${STAGE2_DIR}/rotate/metrics_summary.json"

########################################
# Task 5: step1 only (no object/part)
########################################
TASK_ID="task5_stack_pyramid"
TASK_DESC="stack_pyramid | step1 only | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR=""
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "/nat/demos/pi05/outputs/pi0_stackpyramid/checkpoints/020000/pretrained_model" \
  "stack_pyramid" \
  "" \
  "" \
  "Stack the cubes into a stable pyramid." \
  "${STAGE1_DIR}" \
  "1"

update_live_summary \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/stack_pyramid/metrics_summary.json" \
  "__NONE__"

echo "All tasks finished."
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Summary JSONL: ${SUMMARY_JSONL}"
