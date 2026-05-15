#!/bin/bash
# =============================================================================
# FGManip OpenVLA-OFT evaluation in ManiSkill environments
#
# Usage:
#   bash core/policies/openvla-oft/eval.sh [CHECKPOINT] [TASK] [EPISODES] [OBJECT_NAME] [PART_NAME]
#
# Examples:
#   bash core/policies/openvla-oft/eval.sh                                        # defaults (grasp_part, obj=3558, part=cap)
#   bash core/policies/openvla-oft/eval.sh /path/to/ckpt grasp_part 10 3763 cap   # custom object & part
#   bash core/policies/openvla-oft/eval.sh /path/to/ckpt plug_charger 10 "" ""    # task without object/part
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Point to a saved checkpoint directory (not the run dir itself)
CKPT=${1:-${SCRIPT_DIR}/runs/openvla-7b+fgmanip_rlds+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--fgmanip--8_acts_chunk--L1_regression--proprio_state--20000_chkpt}
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
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 1 \
  --num_open_loop_steps 8 \
  --lora_rank 32 \
  --center_crop True \
  --max_steps 200

# Tasks that accept object_name & part_name:
#   grasp_part, align_to_part, stand_up, toggle_switch,
#   toggle_switch_table, lid_opening, slide_along, rotate, door_env
#
# Tasks without object_name / part_name (pass "" "" to skip):
#   peg_in_hole, plug_charger, stack_pyramid, draw_triangle
