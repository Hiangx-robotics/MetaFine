#!/usr/bin/env bash
# 不用 set -e：单任务失败时继续跑完 train_rlds_batch 里其余任务。
set -uo pipefail

# 默认使用 GPU 0；需要别的卡时：CUDA_VISIBLE_DEVICES=1 bash ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Batch launcher for vanilla OpenVLA (LeRobot LoRA under /nat/demos/openvla, naming:
#   openvla-7b+<dataset_name>+b4+lr-0.0005+lora-r32+dropout-0.0--fgmanip-openvla--image_aug--<N>_chkpt
# )
# 与 train_rlds_batch.sh 中 TASKS 一一对应；调用 evaluate_fixed.py（支持 DR 多 profile）+
# 离散动作：--use_l1_regression False --use_proprio False --num_open_loop_steps 1
#
# DR：单阶段任务默认开启；toggle/press 的 stage1 开 DR，stage2 仅 clean（与旧 batch 一致）。
#
# Outputs（视频在 OUT_ROOT/<task>/stage*/eval_videos/）:
#   ${OUT_ROOT}/<task_id>/stage{1,2}/metrics_summary.json
#   ${OUT_ROOT}/summary_live.json
#   ${OUT_ROOT}/summary_live.jsonl
#
# Usage:
#   bash core/policies/openvla/run_eval_batch_cases.sh
#   CHKPT_STEP=30000 N_EPISODES=10 bash core/policies/openvla/run_eval_batch_cases.sh
# 换输出子目录：BATCH_RUN_DIR=0415（默认）或 OUT_ROOT=/path/to/custom

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_FIXED_PY="${SCRIPT_DIR}/evaluate_fixed.py"

OPENVLA_ROOT="${OPENVLA_ROOT:-/nat/demos/openvla}"
# 默认把本批评测与视频存到 openvla/0415/ 下（可 export BATCH_RUN_DIR 或 OUT_ROOT 覆盖）
BATCH_RUN_DIR="${BATCH_RUN_DIR:-0415}"
OUT_ROOT="${OUT_ROOT:-${OPENVLA_ROOT}/${BATCH_RUN_DIR}/eval_batch_runs}"
SUMMARY_JSON="${OUT_ROOT}/summary_live.json"
SUMMARY_JSONL="${OUT_ROOT}/summary_live.jsonl"

N_EPISODES=10
# 与磁盘目录名一致：30000_chkpt（非 030000）
CHKPT_STEP="${CHKPT_STEP:-30000}"

# 与 train_rlds.sh / train_rlds_batch 默认 RUN_ROOT_DIR 下生成的目录后缀一致
CKPT_SUFFIX="b4+lr-0.0005+lora-r32+dropout-0.0--fgmanip-openvla--image_aug--${CHKPT_STEP}_chkpt"

# 各任务默认 checkpoint（可按需 export 覆盖整个路径）
CKPT_GRASP_PART=${CKPT_GRASP_PART:-"${OPENVLA_ROOT}/openvla-7b+grasp_part_rlds+${CKPT_SUFFIX}"}
CKPT_PEG_IN_HOLE=${CKPT_PEG_IN_HOLE:-"${OPENVLA_ROOT}/openvla-7b+peg_in_hole_rlds+${CKPT_SUFFIX}"}
CKPT_TOGGLE_SWITCH_2=${CKPT_TOGGLE_SWITCH_2:-"${OPENVLA_ROOT}/openvla-7b+toggle_switch_2_rlds_100+${CKPT_SUFFIX}"}
CKPT_PRESS_SWITCH=${CKPT_PRESS_SWITCH:-"${OPENVLA_ROOT}/openvla-7b+press_switch_rlds+${CKPT_SUFFIX}"}
CKPT_ROTATE_ALONG=${CKPT_ROTATE_ALONG:-"${OPENVLA_ROOT}/openvla-7b+rotate_along_rlds+${CKPT_SUFFIX}"}
CKPT_PLUG_CHARGER=${CKPT_PLUG_CHARGER:-"${OPENVLA_ROOT}/openvla-7b+plug_chargere_rlds+${CKPT_SUFFIX}"}
CKPT_STACK_PYRAMID=${CKPT_STACK_PYRAMID:-"${OPENVLA_ROOT}/openvla-7b+stack_pyramid_rlds_100+${CKPT_SUFFIX}"}

# DR 档位（与 eval_fixed.sh / pi0 批量一致）
CAMERA_POS_LEVELS="${CAMERA_POS_LEVELS:-[0.03, 0.06, 0.12]}"
CAMERA_ROT_LEVELS_DEG="${CAMERA_ROT_LEVELS_DEG:-[2.0, 6.0, 12.0]}"
LIGHT_AMBIENT_DELTA_LEVELS="${LIGHT_AMBIENT_DELTA_LEVELS:-[0.10, 0.25, 0.40]}"

usage() {
  cat <<'EOF'
Usage:
  bash core/policies/openvla/run_eval_batch_cases.sh [--n-episodes N]
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

if [[ ! -f "${EVAL_FIXED_PY}" ]]; then
  echo "[ERROR] evaluate_fixed.py not found at: ${EVAL_FIXED_PY}" >&2
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
  local unnorm_key="$3"
  local episodes="$4"
  local object_name="$5"
  local part_name="$6"
  local out_dir="$7"
  local enable_dr="${8:-1}"

  if [[ ! -d "${ckpt}" ]]; then
    echo "[WARN] Skip (missing checkpoint): ${ckpt}" >&2
    return 0
  fi

  mkdir -p "${out_dir}"
  local -a cmd
  cmd=(
    python evaluate_fixed.py
    --pretrained_checkpoint "${ckpt}"
    --task_name "${env_id}"
    --unnorm_key "${unnorm_key}"
    --num_episodes "${episodes}"
    --use_l1_regression False
    --use_proprio False
    --num_open_loop_steps 1
    --lora_rank 32
    --center_crop True
    --control_mode pd_joint_delta_pos
    --num_images_in_input 1
    --video_dir "${out_dir}/eval_videos"
    --output_dir "${out_dir}"
    --max_steps 500
  )
  if [[ "${enable_dr}" == "1" ]]; then
    cmd+=(
      --enable_dr_eval True
      --camera_pos_levels "${CAMERA_POS_LEVELS}"
      --camera_rot_levels_deg "${CAMERA_ROT_LEVELS_DEG}"
      --light_ambient_delta_levels "${LIGHT_AMBIENT_DELTA_LEVELS}"
    )
  else
    cmd+=(--enable_dr_eval False)
  fi
  if [[ -n "${object_name}" ]]; then
    cmd+=(--object_name "${object_name}")
  fi
  if [[ -n "${part_name}" ]]; then
    cmd+=(--part_name "${part_name}")
  fi
  if ! (
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    cd "${SCRIPT_DIR}" || exit 1
    "${cmd[@]}"
  ); then
    echo "[WARN] evaluate_fixed.py failed for env_id=${env_id} out_dir=${out_dir} (continuing batch)." >&2
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

update_live_summary_safe() {
  if ! update_live_summary "$@"; then
    echo "[WARN] update_live_summary failed for task (metrics may be missing)." >&2
  fi
}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OPENVLA_ROOT=${OPENVLA_ROOT}"
echo "BATCH_RUN_DIR=${BATCH_RUN_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "CHKPT_STEP=${CHKPT_STEP} (checkpoint dir suffix: ...--${CHKPT_STEP}_chkpt)"
echo "DR levels: pos=${CAMERA_POS_LEVELS} rot_deg=${CAMERA_ROT_LEVELS_DEG} light=${LIGHT_AMBIENT_DELTA_LEVELS}"
echo "Live summary: ${SUMMARY_JSON}"

########################################
# Task 1: grasp_part_rlds (train_rlds_batch item 1)
########################################
TASK_ID="task1_grasp_part"
TASK_DESC="grasp_part | grasp_part_rlds ckpt | object 3558/cap | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_GRASP_PART}" \
  "grasp_part" \
  "grasp_part_rlds" \
  "${N_EPISODES}" \
  "3558" \
  "cap" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

########################################
# Task 2: peg_in_hole_rlds (item 2)
########################################
TASK_ID="task2_peg_in_hole"
TASK_DESC="peg_in_hole | peg_in_hole_rlds ckpt | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_PEG_IN_HOLE}" \
  "peg_in_hole" \
  "peg_in_hole_rlds" \
  "${N_EPISODES}" \
  "" \
  "" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

########################################
# Task 3: toggle_switch_2_rlds_100 (item 3) — two object IDs
########################################
TASK_ID="task3_toggle_switch_2"
TASK_DESC="toggle_switch_table | toggle_switch_2_rlds_100 | stage1(102812,clean+DR)+stage2(100849,clean)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_TOGGLE_SWITCH_2}" \
  "toggle_switch_table" \
  "toggle_switch_2_rlds_100" \
  "${N_EPISODES}" \
  "102812" \
  "button" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "${CKPT_TOGGLE_SWITCH_2}" \
  "toggle_switch_table" \
  "toggle_switch_2_rlds_100" \
  "${N_EPISODES}" \
  "100849" \
  "button" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "${STAGE2_DIR}/metrics_summary.json" \
  "__NONE__"

########################################
# Task 4: press_switch_rlds (item 4) — two object IDs
########################################
TASK_ID="task4_press_switch"
TASK_DESC="toggle_switch_table | press_switch_rlds | stage1(100979,clean+DR)+stage2(100937,clean)"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
STAGE2_DIR="${TASK_ROOT}/stage2"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_PRESS_SWITCH}" \
  "toggle_switch_table" \
  "press_switch_rlds" \
  "${N_EPISODES}" \
  "100979" \
  "button" \
  "${STAGE1_DIR}" \
  "1"

run_eval_once \
  "${CKPT_PRESS_SWITCH}" \
  "toggle_switch_table" \
  "press_switch_rlds" \
  "${N_EPISODES}" \
  "100937" \
  "button" \
  "${STAGE2_DIR}" \
  "0"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "${STAGE2_DIR}/metrics_summary.json" \
  "__NONE__"

########################################
# Task 5: rotate_along_rlds (item 5)
########################################
TASK_ID="task5_rotate_along"
TASK_DESC="rotate | rotate_along_rlds | object 103062/knob | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_ROTATE_ALONG}" \
  "rotate" \
  "rotate_along_rlds" \
  "${N_EPISODES}" \
  "103062" \
  "knob" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

########################################
# Task 6: plug_chargere_rlds (item 6) — 新增
########################################
TASK_ID="task6_plug_charger"
TASK_DESC="plug_charger | plug_chargere_rlds ckpt | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_PLUG_CHARGER}" \
  "plug_charger" \
  "plug_chargere_rlds" \
  "${N_EPISODES}" \
  "" \
  "" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

########################################
# Task 7: stack_pyramid_rlds_100 (item 7) — 新增
########################################
TASK_ID="task7_stack_pyramid"
TASK_DESC="stack_pyramid | stack_pyramid_rlds_100 ckpt | clean+DR"
TASK_ROOT="${OUT_ROOT}/${TASK_ID}"
STAGE1_DIR="${TASK_ROOT}/stage1"
mkdir -p "${TASK_ROOT}"

run_eval_once \
  "${CKPT_STACK_PYRAMID}" \
  "stack_pyramid" \
  "stack_pyramid_rlds_100" \
  "${N_EPISODES}" \
  "" \
  "" \
  "${STAGE1_DIR}" \
  "1"

update_live_summary_safe \
  "${TASK_ID}" \
  "${TASK_DESC}" \
  "${TASK_ROOT}" \
  "${STAGE1_DIR}/metrics_summary.json" \
  "__NONE__" \
  "__NONE__"

echo "All task blocks finished (see [WARN] for any skipped/failed steps)."
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Summary JSONL: ${SUMMARY_JSONL}"
