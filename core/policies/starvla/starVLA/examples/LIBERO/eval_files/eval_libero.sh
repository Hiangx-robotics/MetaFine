#!/bin/bash

# cd /export/xuhy/zpy/starVLA
# conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/export/xuhy/zpy/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/export/anaconda3/envs/libero/bin/python
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


host="127.0.0.1"
base_port=5694
unnorm_key="franka"
# your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/126_libero4in1_qwen3GR00T_lr1e-4_gradac8_30ksteps/checkpoints/steps_10000/adapter_model.safetensors
# your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/126_libero4in1_qwen3GR00T_lr1e-4_gradac8_30ksteps/checkpoints/steps_10000/adapter_model.safetensors
# your_ckpt=
your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/2_10_libero4in1_LatentVLA_reconDependLatent_LoraQkvo_lr1e-4_gradac8_15ksteps/final_model/adapter_model.safetensors
# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
