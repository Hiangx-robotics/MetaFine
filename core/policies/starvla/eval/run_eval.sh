#!/bin/bash
# starVLA/examples/FGManip/run_eval.sh
# Usage: ./run_eval.sh <skill_type> <part_name> <object_name> [num_episodes]
export CUDA_VISIBLE_DEVICES=1

set -e

export Python=/export/anaconda3/envs/maniskill/bin/python
export FGManip_HOME=/export/xuhy/zpy/FGManip
export FGManip_CONFIG_PATH=${FGManip_HOME}/core
export PYTHONPATH=$PYTHONPATH:${FGManip_HOME} # let eval_fgmanip find the FGManip tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let FGManip find the websocket tools from main repo

# SKILL_TYPE="${1:-stand_up}"
SKILL_TYPE="${1:-plug_charger}"
PART_NAME="${2:-body}"
OBJECT_NAME="${3:-bottle}"
NUM_EPISODES="${4:-10}"
PORT="${5:-10093}"
CKPT_PATH="${6:-/export/xuhy/zpy/starVLA/results/Checkpoints/FGManip_GR00T_20ksteps/checkpoints/steps_15000/adapter_model.safetensors
}"
RECORD_DIR="${7:-eval_videos/${SKILL_TYPE}_${PART_NAME}_${OBJECT_NAME}}"

echo "🎯 Evaluating Qwen-GR00T VLA on FGManip"
echo "   Skill: $SKILL_TYPE"
echo "   Part: $PART_NAME"
echo "   Object: $OBJECT_NAME"
echo "   Episodes: $NUM_EPISODES"
echo "   Checkpoint: $CKPT_PATH"
echo "   Server Port: $PORT"
echo "   Record Dir: $RECORD_DIR"
echo ""

# Step 1: Start model server in background (if not already running)
echo "🚀 Starting VLA model server (in starVLA environment)..."
echo "   Run this in a SEPARATE TERMINAL with starVLA environment activated:"
echo "   cd \$STARVLA_ROOT"
echo "   python deployment/model_server/server_policy.py \\"
echo "       --ckpt_path $CKPT_PATH \\"
echo "       --port $PORT"
echo ""

# Step 2: Run evaluation in FGManip environment
echo "▶️ Starting evaluation (in FGManip environment)..."
${Python} eval_fgmanip.py \
    --skill-type "$SKILL_TYPE" \
    --part-name "$PART_NAME" \
    --object-name "$OBJECT_NAME" \
    --num-episodes "$NUM_EPISODES" \
    --port "$PORT" \
    --policy-ckpt-path ${CKPT_PATH} \
    --camera-keys base_camera hand_camera \
    --record-dir "$RECORD_DIR" \
    --control-mode pd_joint_delta_pos \
    --save-camera-images \

echo "✅ Evaluation complete! Results saved to $RECORD_DIR"