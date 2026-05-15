#!/bin/bash
# =============================================================================
# FGManip OpenVLA-OFT training via official finetune.py (RLDS data pipeline)
#
# Step 1 - Convert data (run once):
#   python utils/convert_to_rlds.py \
#     -i /nat/demos/grasp_part/grasp_part_1 \
#     -o /nat/demos/datasets/rlds \
#     --image-size 256
#
# Step 2 - Train:
#   bash core/policies/openvla-oft/train_rlds.sh
#
# Override defaults via environment variables:
#   DATA_ROOT_DIR=/my/rlds DATASET_NAME=my_ds NUM_GPUS=1 \
#     bash core/policies/openvla-oft/train_rlds.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
NUM_GPUS=${NUM_GPUS:-1}

DATA_ROOT_DIR=${DATA_ROOT_DIR:-/nat/demos/openvla/data/plug_charger}

DATASET_NAME=${DATASET_NAME:-plug_charger_rlds}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-${SCRIPT_DIR}/runs}
VLA_PATH=${VLA_PATH:-openvla/openvla-7b}

cd "$SCRIPT_DIR"
torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS \
  vla-scripts/finetune.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT_DIR \
  --dataset_name $DATASET_NAME \
  --run_root_dir $RUN_ROOT_DIR \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --lr_warmup_steps 500 \
  --num_steps_before_decay 15000 \
  --max_steps ${MAX_STEPS:-30005} \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "${WANDB_ENTITY:-YOUR_WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT:-openvla-oft-fgmanip}" \
  --run_id_note fgmanip--8_acts_chunk--L1_regression--proprio_state
  # --use_val_set True \
  # --val_freq 2000 \
