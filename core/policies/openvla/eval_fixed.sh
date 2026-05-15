#!/bin/bash
set -euo pipefail
# =============================================================================
# FGManip OpenVLA evaluation（evaluate_fixed.py）：任务参数 + 可选 DR + 可选 L1/proprio（OFT）
#
# Vanilla OpenVLA（与 train_rlds.sh 离散微调一致）：
#   OPENVLA_VANILLA=1 bash eval_fixed.sh /path/to/chkpt peg_in_hole 10
#
# Usage:
#   bash core/policies/openvla/eval_fixed.sh [CHECKPOINT] [TASK] [EPISODES] [OBJECT_NAME] [PART_NAME]
#
# Examples:
#   bash core/policies/openvla/eval_fixed.sh /path/to/ckpt peg_in_hole 10
#   bash core/policies/openvla/eval_fixed.sh /path/to/ckpt grasp_part 10 3558 cap
#   ENABLE_DR_EVAL=1 OPENVLA_VANILLA=1 bash eval_fixed.sh /path/to/ckpt grasp_part 10 3558 cap
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Point to a saved checkpoint directory (not the run dir itself).
# Priority: $1 > $OPENVLA_CKPT > local default (if exists)
DEFAULT_CKPT="${SCRIPT_DIR}/runs/openvla-7b+fgmanip_rlds+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt"
if [ -n "${1:-}" ]; then
  CKPT="$1"
elif [ -n "${OPENVLA_CKPT:-}" ]; then
  CKPT="$OPENVLA_CKPT"
elif [ -d "$DEFAULT_CKPT" ]; then
  CKPT="$DEFAULT_CKPT"
else
  echo "[ERROR] checkpoint path is required."
  echo "Usage: bash core/policies/openvla/eval_fixed.sh /path/to/checkpoint [TASK] [EPISODES] [OBJECT_NAME] [PART_NAME]"
  echo "Or set OPENVLA_CKPT=/path/to/checkpoint"
  exit 1
fi
TASK=${2:-peg_in_hole}
EPISODES=${3:-10}
OBJECT_NAME=${4:-}
PART_NAME=${5:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/nat/demos/openvla/${TASK}"}
VIDEO_DIR=${VIDEO_DIR:-"${OUTPUT_ROOT}/eval_videos"}
OPENVLA_SUMMARY_JSON=${OPENVLA_SUMMARY_JSON:-"/nat/demos/openvla/summary_live.json"}
OPENVLA_SUMMARY_JSONL=${OPENVLA_SUMMARY_JSONL:-"/nat/demos/openvla/summary_live.jsonl"}

mkdir -p "$VIDEO_DIR"
mkdir -p "$(dirname "$OPENVLA_SUMMARY_JSON")"

if [ ! -d "$CKPT" ]; then
  echo "[ERROR] checkpoint directory not found: $CKPT"
  exit 1
fi

# Allow manual override: export UNNORM_KEY=xxx
UNNORM_KEY=${UNNORM_KEY:-}
if [ -z "$UNNORM_KEY" ]; then
  ckpt_base="$(basename "$CKPT")"
  if [[ "$ckpt_base" =~ \+([A-Za-z0-9_]+_rlds)\+ ]]; then
    UNNORM_KEY="${BASH_REMATCH[1]}"
  else
    UNNORM_KEY="${TASK}_rlds"
  fi
fi

# DR eval switch and levels
ENABLE_DR_EVAL=${ENABLE_DR_EVAL:-0}
CAMERA_POS_LEVELS=${CAMERA_POS_LEVELS:-"[0.03, 0.06, 0.12]"}
CAMERA_ROT_LEVELS_DEG=${CAMERA_ROT_LEVELS_DEG:-"[2.0, 6.0, 12.0]"}
LIGHT_AMBIENT_DELTA_LEVELS=${LIGHT_AMBIENT_DELTA_LEVELS:-"[0.10, 0.25, 0.40]"}
MAX_STEPS=${MAX_STEPS:-500}
TASK_DESCRIPTION_OVERRIDE=${TASK_DESCRIPTION_OVERRIDE:-}

# Only these tasks accept object_name / part_name.
WITH_OBJ_PART_TASKS=(
  "grasp_part"
  "align_to_part"
  "stand_up"
  "toggle_switch"
  "toggle_switch_table"
  "lid_opening"
  "slide_along"
  "rotate"
  "door_env"
)

supports_obj_part=false
for t in "${WITH_OBJ_PART_TASKS[@]}"; do
  if [ "$TASK" = "$t" ]; then
    supports_obj_part=true
    break
  fi
done

OBJ_ARGS=""
if [ "$supports_obj_part" = true ]; then
  if [ -n "$OBJECT_NAME" ]; then
    OBJ_ARGS="$OBJ_ARGS --object_name $OBJECT_NAME"
  fi
  if [ -n "$PART_NAME" ]; then
    OBJ_ARGS="$OBJ_ARGS --part_name $PART_NAME"
  fi
else
  if [ -n "$OBJECT_NAME" ] || [ -n "$PART_NAME" ]; then
    echo "[WARN] task '$TASK' does not support --object_name/--part_name, ignored."
  fi
fi

cd "$SCRIPT_DIR"
DR_ARGS=()
if [ "$ENABLE_DR_EVAL" = "1" ]; then
  DR_ARGS+=(
    --enable_dr_eval True
    --camera_pos_levels "$CAMERA_POS_LEVELS"
    --camera_rot_levels_deg "$CAMERA_ROT_LEVELS_DEG"
    --light_ambient_delta_levels "$LIGHT_AMBIENT_DELTA_LEVELS"
  )
fi
DESC_ARGS=()
if [ -n "$TASK_DESCRIPTION_OVERRIDE" ]; then
  DESC_ARGS+=(--task_description_override "$TASK_DESCRIPTION_OVERRIDE")
fi

# Vanilla OpenVLA（离散动作）：与 OFT 连续头配置区分
VANILLA_ARGS=()
if [ "${OPENVLA_VANILLA:-0}" = "1" ]; then
  VANILLA_ARGS+=(--use_l1_regression False --use_proprio False --num_open_loop_steps 1)
else
  VANILLA_ARGS+=(--use_l1_regression True --use_proprio True --num_open_loop_steps 8)
fi

python evaluate_fixed.py \
  --pretrained_checkpoint "$CKPT" \
  --task_name "$TASK" \
  --unnorm_key "$UNNORM_KEY" \
  --num_episodes "$EPISODES" \
  --video_dir "$VIDEO_DIR" \
  --output_dir "$OUTPUT_ROOT" \
  $OBJ_ARGS \
  "${DR_ARGS[@]}" \
  "${DESC_ARGS[@]}" \
  --control_mode pd_joint_delta_pos \
  --num_images_in_input 1 \
  --lora_rank 32 \
  --center_crop True \
  --max_steps "$MAX_STEPS" \
  "${VANILLA_ARGS[@]}"

TASK_SUMMARY_JSON="${OUTPUT_ROOT}/metrics_summary.json"
python - "$OPENVLA_SUMMARY_JSON" "$OPENVLA_SUMMARY_JSONL" "$TASK" "$CKPT" "$TASK_SUMMARY_JSON" "$OUTPUT_ROOT" <<'PY'
import json
import os
import sys
from datetime import datetime

summary_json = sys.argv[1]
summary_jsonl = sys.argv[2]
task_name = sys.argv[3]
ckpt = sys.argv[4]
task_summary_path = sys.argv[5]
task_root = sys.argv[6]

def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

now = datetime.now().isoformat(timespec="seconds")
task_metrics = load_json(task_summary_path)

if os.path.exists(summary_json):
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {"updated_at": "", "tasks": {}}

entry = {
    "task_name": task_name,
    "checkpoint": ckpt,
    "task_root": task_root,
    "task_metrics_path": task_summary_path if os.path.exists(task_summary_path) else None,
    "task_metrics": task_metrics,
    "updated_at": now,
}

data.setdefault("tasks", {})[task_name] = entry
data["updated_at"] = now

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
with open(summary_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Updated live summary: {summary_json}")
print(f"Appended live summary line: {summary_jsonl}")
PY
