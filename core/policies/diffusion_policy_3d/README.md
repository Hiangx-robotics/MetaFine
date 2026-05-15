# 3D Diffusion Policy (DP3) for FGManip

Code for running the 3D Diffusion Policy algorithm based on ["3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations"](https://arxiv.org/abs/2403.03954). Adapted from [linden713/ManiSkill - diffusion_policy_3d](https://github.com/linden713/ManiSkill/tree/main/examples/baselines/diffusion_policy_3d) and the [original DP3 repo](https://github.com/YanjieZe/3D-Diffusion-Policy).

## Directory Structure

```
diffusion_policy_3d/
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── train_policy.sh             # Training script
├── eval_policy.sh              # Evaluation script
├── setup.py                    # Package installation
├── data/                       # Zarr data directory
├── dataset/
│   ├── convert_hdf5_to_zarr.py # HDF5 → Zarr conversion
│   └── load_trajectories.py    # HDF5 loading utility
└── diffusion_policy_3d/
    ├── config/
    │   ├── dp3.yaml            # Main training config (hyperparameters)
    │   └── task/               # Per-task config files
    │       ├── PickCube-v1.yaml
    │       ├── PlugCharger-v1.yaml
    │       └── PushCube-v1.yaml
    ├── policy/dp3.py           # DP3 policy
    ├── model/                  # Model components (UNet, PointNet, etc.)
    ├── dataset/                # Dataset classes
    ├── env_runner/             # Evaluation environment runner
    └── env/                    # ManiSkill environment wrappers
```

## 1. Installation

```bash
conda create -n mani_skill python=3.10 -y
conda activate mani_skill

pip install torch torchvision torchaudio
pip install mani_skill

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
cd ..

cd core/policies/diffusion_policy_3d
pip install -e .

conda install -y libstdcxx-ng -c conda-forge
pip install "setuptools<81"
```

## 2. Replay Trajectories

Convert raw trajectories to the target control mode with pointcloud observations:

```bash
cd /home/DMP/nmi/FGManip

python utils/replay_trajectory.py \
    --traj-path demos/PlugCharger-v1/motionplanning/trajectory.h5 \
    --use-first-env-state \
    -c pd_joint_delta_pos \
    -o pointcloud \
    --save-traj \
    --num-envs 20 \
    -b physx_cpu
```

**Parameters:**
| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `-c` | Control mode | `pd_joint_delta_pos` (8-dim), `pd_ee_delta_pose` (7-dim), `pd_ee_delta_pos` (4-dim) |
| `-o` | Observation mode | `pointcloud` |
| `--num-envs` | Parallel envs | `20` |
| `-b` | Simulation backend | `physx_cpu` |

> **Important**: The `-c` control mode determines the action dimension. The task config must match.

Output: `demos/PlugCharger-v1/motionplanning/trajectory.pointcloud.pd_joint_delta_pos.cpu.h5`

## 3. Convert to Zarr Format

```bash
cd core/policies/diffusion_policy_3d

python dataset/convert_hdf5_to_zarr.py \
    --env_name PlugCharger-v1 \
    --num_demos 100 \
    --hdf5_path /home/DMP/nmi/FGManip/demos/PlugCharger-v1/motionplanning/trajectory.pointcloud.pd_joint_delta_pos.cpu.h5 \
    --zarr_dir ./data/
```

Expected output:
```
action shape: (24142, 8)           # depends on control mode
state shape: (24142, 25)           # qpos[9] + qvel[9] + tcp_pose[7]
point_cloud shape: (24142, 512, 6) # xyz[3] + rgb[3]
```

Verify data: `python -c "import zarr; z=zarr.open('./data/maniskill_PlugCharger-v1_expert.zarr','r'); print(z['data']['action'].shape)"`

## 4. Write Task Config

Create a YAML file under `diffusion_policy_3d/config/task/`. **All dimensions must match the zarr data.**

Example `PlugCharger-v1.yaml`:

```yaml
name: PlugCharger-v1
task_name: ${name}

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [512, 3]        # num_points, xyz channels
      type: point_cloud
    agent_pos:
      shape: [25]            # state dim = qpos(9) + qvel(9) + tcp_pose(7)
      type: low_dim
  action:
    shape: [8]               # must match zarr action dim

env_runner:
  _target_: diffusion_policy_3d.env_runner.maniskill_runner.ManiSkillRunner
  eval_episodes: 20
  max_steps: 200
  n_obs_steps: ${n_obs_steps}
  n_action_steps: ${n_action_steps}
  control_mode: pd_joint_delta_pos  # must match replay -c flag
  task_name: ${task_name}
  num_eval_envs: 1
  sim_backend: cpu
  device: ${training.device}
  use_point_crop: ${policy.use_point_crop}
  use_pc_color: ${policy.use_pc_color}

dataset:
  _target_: diffusion_policy_3d.dataset.maniskill_dataset.ManiSkillDataset
  zarr_path: ./data/maniskill_PlugCharger-v1_expert.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: 90
```

**Control mode ↔ action dimension reference:**

| Control Mode (`-c` / `control_mode`) | Action Dim | Description |
|---------------------------------------|------------|-------------|
| `pd_joint_delta_pos` | 8 | 7 joint deltas + 1 gripper |
| `pd_ee_delta_pose` | 7 | 6 EE pose deltas + 1 gripper |
| `pd_ee_delta_pos` | 4 | 3 EE position deltas + 1 gripper |

## 5. Training

```bash
cd core/policies/diffusion_policy_3d
conda activate mani_skill

# bash train_policy.sh <alg_name> <task_name> <exp_id> <seed> <gpu_id>
bash train_policy.sh dp3 PlugCharger-v1 001 0 0
```

**Arguments:**
1. Algorithm name: `dp3` (corresponds to `dp3.yaml`)
2. Task name: must match a file in `config/task/`
3. Experiment ID: custom string (e.g., date)
4. Seed: random seed
5. GPU ID: GPU device index

Outputs are saved to `data/outputs/<task>-<alg>-<id>_seed<seed>/`. Training logs are tracked via [WandB](https://wandb.ai).

## 6. Evaluation

```bash
# Use the same arguments as training
bash eval_policy.sh dp3 PlugCharger-v1 001 0 0
DR_SWEEP_ALL_LEVELS=1 ENABLE_DR_EVAL=1 bash eval_all_7_tasks.sh 0305 0 0
DR_SWEEP_ALL_LEVELS=0 DR_LEVEL_IDX=2 bash eval_all_7_tasks.sh 0305 0 0

ENABLE_DR_EVAL=0 bash eval_all_7_tasks.sh 0305 0 0
```

## Troubleshooting

### `CXXABI_1.3.15 not found`
```bash
conda install -y libstdcxx-ng -c conda-forge
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
```
This is already set automatically in `train_policy.sh`.

### `ModuleNotFoundError: No module named 'pkg_resources'`
```bash
pip install "setuptools<81"
```

## Citation

```bibtex
@inproceedings{ze20243d,
  title={3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations},
  author={Ze, Yanjie and Zhang, Gu and Zhang, Kangning and Hu, Chenyuan and Wang, Muhan and Xu, Huazhe},
  booktitle={ICRA 2024 Workshop on 3D Visual Representations for Robot Manipulation},
  year={2024}
}
```
