#!/usr/bin/env bash
# Batch evaluation for DP3: loads checkpoints from /nat/demos/dp3 and runs eval for each task.
# Usage:
#   bash eval_all_7_tasks.sh [exp_id] [seed] [gpu_id]
# Args:
#   exp_id: experiment id used in checkpoint folder name (e.g. 0305)
#   seed: training/eval seed used in checkpoint folder name (e.g. 0)
#   gpu_id: CUDA device index (e.g. 0 means cuda:0)
# Example:
#   bash eval_all_7_tasks.sh 0305 0 0
#
# Checkpoint dirs expected under CHECKPOINT_ROOT: <task>-dp3-<exp_id>_seed<seed>/
# You can later override env_id / object / part per task via task config or env.

set -euo pipefail

ALG_NAME="dp3"
EXP_ID="${1:-0305}"
SEED="${2:-0}"
GPU_ID="${3:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/nat/demos/dp3}"
VIDEO_ROOT="${VIDEO_ROOT:-/nat/demos/dp3/out}"
ENABLE_DR_EVAL="${ENABLE_DR_EVAL:-1}"
DR_SWEEP_ALL_LEVELS="${DR_SWEEP_ALL_LEVELS:-1}"
DR_LEVEL_IDX="${DR_LEVEL_IDX:-0}"
RUN_CLEAN_FIRST="${RUN_CLEAN_FIRST:-1}"
RUN_OBJECT_SWAP="${RUN_OBJECT_SWAP:-1}"

# Reference from ACT eval settings.
CAMERA_POS_LEVELS=(0.03 0.06 0.12)
CAMERA_ROT_LEVELS_DEG=(2 6 12)
LIGHT_AMBIENT_DELTA_LEVELS=(0.10 0.25 0.40)

# Step2 object swap (clean only) for tasks with object/part env kwargs.
# Tasks not listed here are skipped for swap stage.
declare -A SWAP_OBJECT_MAP=(
  ["toggle_switch_2"]="100849"
  ["press_switch_7"]="100937"
  ["grasp_part_1"]="3763"
)
declare -A SWAP_PART_MAP=(
  ["toggle_switch_2"]="button"
  ["press_switch_7"]="button"
  ["grasp_part_1"]="cap"
)

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"
fi

# Tasks that have checkpoint dirs under /nat/demos/dp3 (match train_all_7_tasks.sh task names)
TASKS=(
  "toggle_switch_2"
#   "peg_in_hole_motionplanning"
  "press_switch_7"
#   "plug_charger_motionplanning"
#   "stack_pyramid_motionplanning"
  "grasp_part_1"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ "${ENABLE_DR_EVAL}" = "1" ]; then
    if [ "${DR_SWEEP_ALL_LEVELS}" = "1" ]; then
        DR_LEVEL_LIST=(0 1 2)
        echo "[DR] enabled, sweep all levels: ${DR_LEVEL_LIST[*]}"
    else
        if [ "${DR_LEVEL_IDX}" -lt 0 ] || [ "${DR_LEVEL_IDX}" -ge "${#CAMERA_POS_LEVELS[@]}" ]; then
            echo "[ERROR] DR_LEVEL_IDX=${DR_LEVEL_IDX} out of range [0, $((${#CAMERA_POS_LEVELS[@]} - 1))]"
            exit 1
        fi
        DR_LEVEL_LIST=("${DR_LEVEL_IDX}")
        echo "[DR] enabled, single level: ${DR_LEVEL_IDX}"
    fi
else
    echo "[DR] disabled"
fi

if [ "${RUN_CLEAN_FIRST}" = "1" ]; then
    echo "[CLEAN] enabled: each task will run clean eval before DR eval."
else
    echo "[CLEAN] disabled"
fi

if [ "${RUN_OBJECT_SWAP}" = "1" ]; then
    echo "[SWAP] enabled: object swap clean-only stage will run where configured."
else
    echo "[SWAP] disabled"
fi

for task in "${TASKS[@]}"; do
    run_dir="${CHECKPOINT_ROOT}/${task}-${ALG_NAME}-${EXP_ID}_seed${SEED}"

    if [ ! -d "${run_dir}" ]; then
        echo "[SKIP] ${task}: run_dir not found -> ${run_dir}"
        continue
    fi
    if [ ! -f "${run_dir}/checkpoints/latest.ckpt" ]; then
        echo "[SKIP] ${task}: latest.ckpt not found in ${run_dir}/checkpoints/"
        continue
    fi

    echo "[EVAL] task=${task}, exp_id=${EXP_ID}, seed=${SEED}, gpu=${GPU_ID}"
    if [ "${RUN_CLEAN_FIRST}" = "1" ]; then
        VIDEO_ROOT_CLEAN="${VIDEO_ROOT}/clean"
        echo "[EVAL-CLEAN] task=${task}"
        bash eval_policy.sh "${ALG_NAME}" "${task}" "${EXP_ID}" "${SEED}" "${GPU_ID}" "${run_dir}" \
            +task.env_runner.video_root_dir="${VIDEO_ROOT_CLEAN}" \
            +task.env_runner.dr_eval=false
    fi

    if [ "${RUN_OBJECT_SWAP}" = "1" ]; then
        swap_object="${SWAP_OBJECT_MAP[$task]:-}"
        swap_part="${SWAP_PART_MAP[$task]:-}"
        if [ -n "${swap_object}" ] && [ -n "${swap_part}" ]; then
            VIDEO_ROOT_SWAP="${VIDEO_ROOT}/swap_clean"
            echo "[EVAL-SWAP] task=${task}, object=${swap_object}, part=${swap_part}"
            bash eval_policy.sh "${ALG_NAME}" "${task}" "${EXP_ID}" "${SEED}" "${GPU_ID}" "${run_dir}" \
                +task.env_runner.video_root_dir="${VIDEO_ROOT_SWAP}" \
                +task.env_runner.dr_eval=false \
                "task.env_runner.env_kwargs.object_name='${swap_object}'" \
                "task.env_runner.env_kwargs.part_name='${swap_part}'"
        else
            echo "[EVAL-SWAP] skip task=${task} (no swap mapping)"
        fi
    fi

    if [ "${ENABLE_DR_EVAL}" = "1" ]; then
        for level in "${DR_LEVEL_LIST[@]}"; do
            CAMERA_POS_JITTER="${CAMERA_POS_LEVELS[${level}]}"
            CAMERA_ROT_JITTER_DEG="${CAMERA_ROT_LEVELS_DEG[${level}]}"
            LIGHT_AMBIENT_DELTA="${LIGHT_AMBIENT_DELTA_LEVELS[${level}]}"
            VIDEO_ROOT_LEVEL="${VIDEO_ROOT}/dr_level_${level}"
            echo "[EVAL-DR] task=${task}, level=${level}, pos=${CAMERA_POS_JITTER}, rot_deg=${CAMERA_ROT_JITTER_DEG}, light_delta=${LIGHT_AMBIENT_DELTA}"
            bash eval_policy.sh "${ALG_NAME}" "${task}" "${EXP_ID}" "${SEED}" "${GPU_ID}" "${run_dir}" \
                +task.env_runner.video_root_dir="${VIDEO_ROOT_LEVEL}" \
                +task.env_runner.dr_eval=true \
                +task.env_runner.camera_pos_jitter="${CAMERA_POS_JITTER}" \
                +task.env_runner.camera_rot_jitter_deg="${CAMERA_ROT_JITTER_DEG}" \
                +task.env_runner.light_ambient_delta="${LIGHT_AMBIENT_DELTA}"
        done
    else
        bash eval_policy.sh "${ALG_NAME}" "${task}" "${EXP_ID}" "${SEED}" "${GPU_ID}" "${run_dir}" \
            +task.env_runner.video_root_dir="${VIDEO_ROOT}" \
            +task.env_runner.dr_eval=false
    fi
done

echo "[DONE] Batch eval finished."
