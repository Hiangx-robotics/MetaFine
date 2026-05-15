# use the same command as training except the script
# for example:
#   bash eval_policy.sh dp3 PlugCharger-v1 001 0 0
#   bash eval_policy.sh dp3 plug_charger_motionplanning 0305 0 0 "" /nat/demos/dp3/plug_charger_motionplanning-dp3-0305_seed0
#
# Args: alg_name task_name exp_id seed gpu_id [run_dir] [hydra_overrides...]
#   run_dir: optional; if set, load checkpoint from run_dir/checkpoints/latest.ckpt

DEBUG=False
wandb_mode=${WANDB_MODE:-offline}
save_ckpt=${SAVE_CKPT:-false}

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir=${6:-"./data/outputs/${exp_name}_seed${seed}"}
extra_overrides=("${@:7}")

gpu_id=${5}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
fi
python eval.py --config-name=${config_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    "${extra_overrides[@]}"



