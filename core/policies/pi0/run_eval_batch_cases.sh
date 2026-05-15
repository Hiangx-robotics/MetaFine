#!/usr/bin/env bash
# 不用 set -e：任一 evaluate / update_live_summary 非零退出不应中断后续任务（之前会表现为「跑完某一截就停」）。
set -uo pipefail

# Batch launcher for PI0 evaluation (LeRobot checkpoints under /nat/demos/pi0).
# - Default: 10 episodes per run (10 rollout videos when --save-video)
# - Checkpoints: .../checkpoints/030000/pretrained_model (30k steps), except
#   peginhole/ which only has 020000 in this workspace — override with PEG_CKPT_TAG if you add 030000.
# - Outputs: /nat/demos/pi0/runs/ (per-task subdirs, same layout idea as pi05)
#
# Usage:
#   bash core/policies/pi0/run_eval_batch_cases.sh
#   bash core/policies/pi0/run_eval_batch_cases.sh --n-episodes 10 --device cuda
#   CKPT_TAG=030000 OUT_ROOT=/nat/demos/pi0/runs bash core/policies/pi0/run_eval_batch_cases.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/evaluate.py"

PI0_ROOT="${PI0_ROOT:-/nat/demos/pi0}"
OUT_ROOT="${OUT_ROOT:-${PI0_ROOT}/runs}"
SUMMARY_JSON="${OUT_ROOT}/summary_live.json"
SUMMARY_JSONL="${OUT_ROOT}/summary_live.jsonl"

# 30k training step folder name in LeRobot outputs (030000, not 30000).
CKPT_TAG="${CKPT_TAG:-030000}"
# peginhole 目录若只有 020000，保持默认；若你后来训了 030000，可 export PEG_CKPT_TAG=030000
PEG_CKPT_TAG="${PEG_CKPT_TAG:-020000}"

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
  bash core/policies/pi0/run_eval_batch_cases.sh \
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

  if [[ ! -d "${policy_path}" ]]; then
    echo "[WARN] Skip (missing checkpoint): ${policy_path}" >&2
    return 0
  fi

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
  if ! "${cmd[@]}"; then
    echo "[WARN] evaluate.py exited non-zero (env_id=${env_id}, record_dir=${record_dir}). Continuing batch." >&2
  fi
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

update_live_summary_safe() {
  if ! update_live_summary "$@"; then
    echo "[WARN] update_live_summary failed (task may still have videos on disk). Continuing." >&2
  fi
}

echo "PI0_ROOT=${PI0_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "CKPT_TAG=${CKPT_TAG} (PEG_CKPT_TAG=${PEG_CKPT_TAG} for peginhole)"
echo "N_EPISODES=${N_EPISODES} (videos per sub-run)"
echo "Live summary: ${SUMMARY_JSON}"

########################################
# Task 1: peg in hole (peginhole) — 已暂时跳过，需要时删除 if false 整块即可恢复
########################################
if false; then
TASK_ID="task1_peginhole"
TASK_DESC="peg_in_hole | ${PI0_ROOT}/peginhole | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR=""
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${PI0_ROOT}/peginhole/checkpoints/${PEG_CKPT_TAG}/pretrained_model" \
  "peg_in_hole" \
  "" \
  "" \
  "Grasp the stick on the table and peg it into the hold" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/peg_in_hole/metrics_summary.json" \
  "__NONE__"
fi

########################################
# Task 2: toggle — 已暂时跳过（从 Task 3 开始跑）；恢复时删除 if false 整块
########################################
if false; then
TASK_ID="task2_toggle_switch_table"
TASK_DESC="toggle_switch_table | ${PI0_ROOT}/toggle_part | step1+step2(object swap)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${PI0_ROOT}/toggle_part/checkpoints/${CKPT_TAG}/pretrained_model" \
  "toggle_switch_table" \
  "102812" \
  "button" \
  "Toggle the switch with the button on the table" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "${PI0_ROOT}/toggle_part/checkpoints/${CKPT_TAG}/pretrained_model" \
  "toggle_switch_table" \
  "100849" \
  "button" \
  "Toggle the switch with the button on the table" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/toggle_switch_table/metrics_summary.json" \
  "${STAGE2_DIR}/toggle_switch_table/metrics_summary.json"
fi

########################################
# Task 3: press — two object IDs (press_part @ 30k)
########################################
TASK_ID="task3_toggle_switch_table_press"
TASK_DESC="toggle_switch_table(press) | ${PI0_ROOT}/press_part | step1+step2(object swap)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${PI0_ROOT}/press_part/checkpoints/${CKPT_TAG}/pretrained_model" \
  "toggle_switch_table" \
  "100979" \
  "button" \
  "Press the switch on the table with the button" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "${PI0_ROOT}/press_part/checkpoints/${CKPT_TAG}/pretrained_model" \
  "toggle_switch_table" \
  "100937" \
  "button" \
  "Press the switch on the table with the button" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/toggle_switch_table/metrics_summary.json" \
  "${STAGE2_DIR}/toggle_switch_table/metrics_summary.json"

########################################
# Task 4: grasp_part — bottle cap (matches gras_part_1 lerobot task text)
########################################
TASK_ID="task4_grasp_part"
TASK_DESC="grasp_part | ${PI0_ROOT}/grasp_part | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR=""
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${PI0_ROOT}/grasp_part/checkpoints/${CKPT_TAG}/pretrained_model" \
  "grasp_part" \
  "3558" \
  "cap" \
  "Grasp the cap of the bottle" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/grasp_part/metrics_summary.json" \
  "__NONE__"

########################################
# Task 5: long / put_blocks_into_boxes (green cube — default env special_cube)
########################################
TASK_ID="task5_put_blocks_into_boxes"
TASK_DESC="put_blocks_into_boxes | ${PI0_ROOT}/long | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR=""
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${PI0_ROOT}/long/checkpoints/${CKPT_TAG}/pretrained_model" \
  "put_blocks_into_boxes" \
  "" \
  "" \
  "Place the green cube in the left box and the other cubes in the right box." \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/put_blocks_into_boxes/metrics_summary.json" \
  "__NONE__"

echo "All task blocks finished (see [WARN] lines above for any skipped checkpoints or failed evals)."
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Summary JSONL: ${SUMMARY_JSONL}"
