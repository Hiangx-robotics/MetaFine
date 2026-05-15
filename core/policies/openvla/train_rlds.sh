#!/bin/bash
# =============================================================================
# FGManip OpenVLA training via official LoRA finetune.py (RLDS data pipeline)
#
# The RLDS dataset root should contain TFDS-style folders such as:
#   /nat/demos/openvla/data/grasp_part_rlds
#
# Usage:
#   bash core/policies/openvla/train_rlds.sh
#
# Override defaults via environment variables:
#   DATA_ROOT_DIR=/nat/demos/openvla/data \
#   DATASET_NAME=grasp_part_rlds \
#   NUM_GPUS=1 \
#   CUDA_VISIBLE_DEVICES=1 \
#     bash core/policies/openvla/train_rlds.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
NUM_GPUS=${NUM_GPUS:-1}

DATA_ROOT_DIR=${DATA_ROOT_DIR:-/nat/demos/openvla/data}
DATASET_NAME=${DATASET_NAME:-grasp_part_rlds}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-${SCRIPT_DIR}/runs}
ADAPTER_TMP_DIR=${ADAPTER_TMP_DIR:-${SCRIPT_DIR}/adapter-tmp}
VLA_PATH=${VLA_PATH:-openvla/openvla-7b}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-30005}
SAVE_STEPS=${SAVE_STEPS:-5000}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
LORA_RANK=${LORA_RANK:-32}

cd "$SCRIPT_DIR"
torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" \
  vla-scripts/finetune.py \
  --vla_path "$VLA_PATH" \
  --data_root_dir "$DATA_ROOT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --adapter_tmp_dir "$ADAPTER_TMP_DIR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps 1 \
  --learning_rate "$LEARNING_RATE" \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank "$LORA_RANK" \
  --wandb_entity "${WANDB_ENTITY:-YOUR_WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT:-openvla-fgmanip}" \
  --run_id_note "${RUN_ID_NOTE:-fgmanip-openvla}"
