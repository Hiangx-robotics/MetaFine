# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=enp97s0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=enp97s0          # 必须：指定有效网卡
# # export NCCL_IB_HCA=...                    # 删除！你没有 IB/RoCE
# export TORCH_NCCL_BLOCKING_WAIT=1          # 替代 NCCL_BLOCKING_WAIT
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=3600                   # 单位秒，1小时足够（原10000秒≈2.7小时，可保留）
# export NCCL_SOCKET_TIMEOUT_MS=3600000      # 建议与 NCCL_TIMEOUT 一致

# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=1
export NCCL_SOCKET_IFNAME=enp97s0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT_MS=3600000

# === 新增/修改的核心配置 ===
export NCCL_P2P_DISABLE=1   # 必须加！强制走 CPU 内存中转，解决 PCIe 死锁
export NCCL_IB_DISABLE=1    # 必须加！禁用 InfiniBand/RoCE，强制走 Ethernet
export WANDB_MODE=disabled  # 建议暂时加上，排除网络干扰
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/export/xuhy/zpy/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct # -with-Action-Query-Better64
config_yaml=/export/xuhy/zpy/starVLA/examples/FGManip/train_files/starvla_cotrain_FGManip.yaml
libero_data_root=/export/xuhy/zpy/starVLA/playground/Datasets/fgmanip
data_mix=fgmanip
run_root_dir=./results/Checkpoints
run_id=FGManip_GR00T_standup_10ksteps
# run_id=2_12_libero4in1_LatentVLA_maskLatent0.75Weight0.1_LoraQkvo_lr1e-4_gradac8_10ksteps
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file ../starVLA/starVLA/config/deepseeds/deepspeed_single_gpu.yaml \
  --num_processes 1 \
  ../starVLA/starVLA/training/train_starvla_lora.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 1000 \
  --trainer.num_warmup_steps 100 \
  --trainer.save_interval 500 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity arrosw-southeast-university \
  --compute_environment LOCAL_MACHINE \
  --distributed_type NO_DEEPSPEED
  # --trainer.freeze_modules "qwen_vl_interface.model.model.visual" \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
