#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/export/anaconda3/envs/starVLA/bin/python
your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/FGManip_GR00T_20ksteps/checkpoints/steps_15000/adapter_model.safetensors
gpu_id=1
port=10093
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} ../starVLA/deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
