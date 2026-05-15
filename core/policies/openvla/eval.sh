#!/bin/bash
# =============================================================================
# FGManip OpenVLA evaluation in ManiSkill environments
#
# Usage:
#   bash core/policies/openvla/eval.sh [CHECKPOINT] [TASK] [EPISODES] [OBJECT_NAME] [PART_NAME]
#
# Examples:
#   bash core/policies/openvla/eval.sh
#   bash core/policies/openvla/eval.sh /path/to/ckpt grasp_part 10 3763 cap
#   bash core/policies/openvla/eval.sh /path/to/ckpt plug_charger 10 "" ""
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Point to a saved checkpoint directory (not the run dir itself)
CKPT=${1:-${SCRIPT_DIR}/runs/openvla-7b+grasp_part_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--fgmanip-openvla--image_aug--5000_chkpt}
TASK=${2:-grasp_part}
EPISODES=${3:-10}
OBJECT_NAME=${4:-3558}
PART_NAME=${5:-cap}

OBJ_ARGS=""
if [ -n "$OBJECT_NAME" ]; then
  OBJ_ARGS="$OBJ_ARGS --object_name $OBJECT_NAME"
fi
if [ -n "$PART_NAME" ]; then
  OBJ_ARGS="$OBJ_ARGS --part_name $PART_NAME"
fi

cd "$SCRIPT_DIR"
python evaluate.py \
  --pretrained_checkpoint "$CKPT" \
  --task_name "$TASK" \
  --num_episodes "$EPISODES" \
  $OBJ_ARGS \
  --control_mode pd_joint_delta_pos \
  --num_images_in_input 1 \
  --num_open_loop_steps 1 \
  --center_crop True \
  --max_steps 200
