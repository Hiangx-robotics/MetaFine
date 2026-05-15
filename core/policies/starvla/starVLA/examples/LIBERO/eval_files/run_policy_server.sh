#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/export/anaconda3/envs/starVLA/bin/python
# your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/1229_libero4in1_qwen3oft/checkpoints/steps_30000/adapter_model.safetensors
# your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/126_libero4in1_qwen3GR00T_lr1e-4_gradac8_30ksteps/checkpoints/steps_10000/adapter_model.safetensors
your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/2_10_libero4in1_LatentVLA_reconLatent_LoraQkvo_lr1e-4_gradac8_15ksteps/final_model/adapter_model.safetensors
# your_ckpt=/export/xuhy/zpy/starVLA/results/Checkpoints/2_3_libero4in1_LatentVLA_SpecialToken_lr1e-4_gradac8_30ksteps/final_model/adapter_model.safetensors
gpu_id=1
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
