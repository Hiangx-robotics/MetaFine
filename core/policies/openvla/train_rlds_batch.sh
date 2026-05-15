#!/usr/bin/env bash
set -euo pipefail

# 完全复刻 train_rlds.sh 的参数，仅循环不同任务数据（OpenVLA）
ROOT="/export/xuhy/EAI/git"
SCRIPT="${ROOT}/core/policies/openvla/train_rlds.sh"
RUN_ROOT_DIR="/nat/demos/openvla"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "${ROOT}"

TASKS=(
  "/nat/demos/openvla/data|grasp_part_rlds"
  "/nat/demos/openvla/data/peg_insertion_side|peg_in_hole_rlds"
  "/nat/demos/openvla/data/toggle_switch_2|toggle_switch_2_rlds_100"
  "/nat/demos/openvla/data/press_switch_7|press_switch_rlds"
  "/nat/demos/openvla/data/rotate_along_2|rotate_along_rlds"
  "/nat/demos/openvla/data/plug_charger|plug_chargere_rlds"
  "/nat/demos/openvla/data/stack_pyramid|stack_pyramid_rlds_100"
)

for item in "${TASKS[@]}"; do
  if [[ "$item" != *"|"* ]]; then
    echo "[ERROR] Invalid TASKS item format: $item"
    echo "        Expected: /path/to/data_root|dataset_name"
    exit 1
  fi
  DATA_ROOT_DIR="${item%%|*}"
  DATASET_NAME="${item##*|}"

  echo "============================================================"
  echo "[RUN] DATA_ROOT_DIR=${DATA_ROOT_DIR}"
  echo "[RUN] DATASET_NAME=${DATASET_NAME}"
  echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DATA_ROOT_DIR="${DATA_ROOT_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
  bash "${SCRIPT}"
done


