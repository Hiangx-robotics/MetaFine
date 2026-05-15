# OpenVLA-OFT Integration for FGManip

This directory integrates [OpenVLA-OFT](https://github.com/moojink/openvla-oft) with the FGManip simulation platform, providing an end-to-end pipeline for **data conversion**, **fine-tuning**, and **evaluation** on ManiSkill-based manipulation tasks.

## Directory Layout

```
core/policies/openvla-oft/
├── train_rlds.sh        # Training entry point (official finetune.py + RLDS data pipeline)
├── eval.sh              # Evaluation entry point
├── evaluate.py          # Evaluation script — runs rollouts in ManiSkill environments
├── prismatic/           # Extended OXE dataset configs with FGManip registration
├── dlimp/               # Local dlimp dependency
├── runs/                # Training checkpoints (auto-created)
├── eval_videos/         # Evaluation rollout videos (auto-created)
└── SETUP.md             # Environment installation instructions
```

## Prerequisites

Refer to [SETUP.md](SETUP.md) for full installation details. A minimal setup is shown below:

```bash
conda activate maniskill
pip3 install torch torchvision torchaudio
cd core/policies/openvla-oft && pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Data Preparation

Use `utils/convert_to_rlds.py` to convert ManiSkill replay HDF5 trajectories into the RLDS (TensorFlow Datasets) format expected by the OpenVLA-OFT training pipeline.

```bash
# Single task
python utils/convert_to_rlds.py \
  -i /path/to/demos/grasp_part/grasp_part_1 \
  -o /path/to/datasets/rlds \
  --dataset-name grasp_part_rlds \
  --image-size 256

# Multiple source directories merged into one dataset
python utils/convert_to_rlds.py \
  -i /path/to/demos/grasp_part/run_1 /path/to/demos/grasp_part/run_2 \
  -o /path/to/datasets/rlds \
  --dataset-name fgmanip_rlds
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-i / --input-dirs` | Directories containing replay HDF5 files (accepts multiple) | *required* |
| `-o / --output-dir` | RLDS output root (maps to `data_root_dir` at training time) | `/nat/demos/datasets/rlds` |
| `--dataset-name` | Dataset identifier (maps to `dataset_name` at training time) | `fgmanip_rlds` |
| `--image-size` | Target image resolution | `256` |
| `--val-ratio` | Fraction of episodes reserved for validation | `0.0` |
| `--all-episodes` | Include failed episodes (default: successful only) | `False` |

## Training

Launch fine-tuning with the default configuration:

```bash
bash core/policies/openvla-oft/train_rlds.sh
```

Override defaults via environment variables:

```bash
DATA_ROOT_DIR=/path/to/datasets/rlds \
DATASET_NAME=grasp_part_rlds \
NUM_GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
  bash core/policies/openvla-oft/train_rlds.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VLA_PATH` | `openvla/openvla-7b` | Pre-trained model path or HuggingFace model ID |
| `DATA_ROOT_DIR` | `/nat/demos/datasets/rlds` | RLDS data root directory |
| `DATASET_NAME` | `fgmanip_rlds` | Dataset name registered in OXE configs |
| `--batch_size` | `8` | Per-GPU batch size |
| `--learning_rate` | `5e-4` | Peak learning rate |
| `--max_steps` | `20005` | Total training steps |
| `--save_freq` | `5000` | Checkpoint save interval (steps) |
| `--lora_rank` | `32` | LoRA rank |
| `--use_l1_regression` | `True` | Use L1 regression action head |
| `--use_proprio` | `True` | Condition on proprioceptive state (joint positions) |
| `--image_aug` | `True` | Enable image augmentation |

Checkpoints are saved under `runs/` with auto-generated names:

```
openvla-7b+fgmanip_rlds+b8+lr-0.0005+lora-r32+...--<step>_chkpt
```

## Evaluation

Run evaluation with default settings or specify arguments positionally:

```bash
# Defaults: grasp_part task, object=3558, part=cap, 10 episodes
bash core/policies/openvla-oft/eval.sh

# Custom checkpoint, task, episode count, object, and part
bash core/policies/openvla-oft/eval.sh /path/to/checkpoint grasp_part 20 3763 cap

# Tasks without object/part (pass empty strings to skip)
bash core/policies/openvla-oft/eval.sh /path/to/checkpoint plug_charger 10 "" ""
```

The evaluation script performs rollouts in ManiSkill environments, reports the success rate, and saves video recordings to `eval_videos/`.

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained_checkpoint` | — | Path to checkpoint directory or HuggingFace model ID |
| `--task_name` | `plug_charger` | Task identifier (see table below) |
| `--object_name` | `None` | Object asset name (e.g., `3558`). Only for tasks that accept it |
| `--part_name` | `None` | Part to manipulate (e.g., `cap`). Only for tasks that accept it |
| `--num_episodes` | `10` | Number of evaluation episodes |
| `--max_steps` | `200` | Maximum steps per episode |
| `--control_mode` | `pd_joint_delta_pos` | ManiSkill control mode |
| `--num_open_loop_steps` | `8` | Action chunk length (open-loop execution steps) |
| `--use_l1_regression` | `True` | Must match the training configuration |
| `--use_proprio` | `True` | Must match the training configuration |
| `--lora_rank` | `32` | Must match the training configuration |
| `--save_video` | `True` | Save rollout videos to `eval_videos/` |

When `object_name` and `part_name` are provided, the task description prompt sent to the model is automatically templated (e.g., `"grasp the cap"` instead of the generic `"grasp the handle"`).

## Supported Tasks

Tasks that accept `object_name` and `part_name`:

| Task ID | Description |
|---------|-------------|
| `grasp_part` | Grasp a specified part of the object |
| `align_to_part` | Align the gripper to a specified part |
| `stand_up` | Pick up the object and make it stand upright |
| `toggle_switch` | Toggle the switch |
| `toggle_switch_table` | Toggle the switch on the table |
| `lid_opening` | Open the lid of a bottle |
| `slide_along` | Slide the object along a surface |
| `rotate` | Rotate the object |
| `door_env` | Open the door by the handle |

Tasks with fixed scenes (no `object_name` / `part_name`):

| Task ID | Description |
|---------|-------------|
| `plug_charger` | Pick up the charger and plug it into the receptacle |
| `peg_in_hole` | Pick up the peg and insert it into the hole |
| `stack_pyramid` | Stack the blue cube on top of the red and green cubes |
| `draw_triangle` | Draw a triangle connecting the vertices |

## OXE Dataset Registration

FGManip dataset configurations are registered in `prismatic/vla/datasets/rlds/oxe/configs.py`. All tasks share a unified observation and action layout:

- **Image observation**: `primary` — base camera RGB
- **State observation**: 7-DoF joint angles + 1-D gripper width
- **Action encoding**: `JOINT_POS` (joint-space position targets)

Per-task datasets (e.g., `grasp_part_rlds`, `plug_charger_rlds`) and the combined `fgmanip_rlds` dataset are all supported.
