# OpenVLA Integration for FGManip

This directory integrates [OpenVLA](https://github.com/openvla/openvla) with the FGManip simulation platform, providing an end-to-end path for **RLDS data loading**, **LoRA fine-tuning**, and **ManiSkill evaluation**.

The code in `core/policies/openvla/` is intentionally aligned with the official OpenVLA training/inference semantics:

- RLDS batches use a single primary image, a language instruction, and a single action target.
- Fine-tuning defaults to the official OpenVLA LoRA recipe.
- OFT-only defaults such as continuous action heads, proprio inputs, and action chunking are no longer enabled by default here.

## Directory Layout

```text
core/policies/openvla/
├── train_rlds.sh        # Training entry point (official OpenVLA LoRA finetune)
├── eval.sh              # Evaluation entry point
├── evaluate.py          # ManiSkill rollout evaluation
├── prismatic/           # OpenVLA code + FGManip RLDS/OXE registration
├── runs/                # Training checkpoints (auto-created)
├── eval_videos/         # Evaluation rollout videos (auto-created)
└── SETUP.md             # Environment installation instructions
```

## Prerequisites

Refer to `SETUP.md` for environment details. A minimal setup is:

```bash
conda activate maniskill
pip3 install torch torchvision torchaudio
cd core/policies/openvla && pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Data Layout

The training script expects TFDS/RLDS datasets under a root directory such as:

```text
/nat/demos/openvla/data/
└── grasp_part_rlds/
    └── 1.0.0/
        ├── dataset_info.json
        ├── features.json
        └── ...
```

For the current FGManip setup, the default dataset is:

- `data_root_dir=/nat/demos/openvla/data`
- `dataset_name=grasp_part_rlds`

FGManip dataset registration lives in:

- `prismatic/vla/datasets/rlds/oxe/configs.py`
- `prismatic/vla/datasets/rlds/oxe/transforms.py`

All FGManip tasks currently share:

- Image observation: `primary`
- State observation: 7-DoF joint angles + 1-D gripper width
- Action encoding: `JOINT_POS`

## Training

Launch fine-tuning with the default configuration:

```bash
bash core/policies/openvla/train_rlds.sh
```

Override defaults via environment variables:

```bash
DATA_ROOT_DIR=/nat/demos/openvla/data \
DATASET_NAME=grasp_part_rlds \
NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=1 \
  bash core/policies/openvla/train_rlds.sh
```

### Important Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VLA_PATH` | `openvla/openvla-7b` | Base OpenVLA checkpoint |
| `DATA_ROOT_DIR` | `/nat/demos/openvla/data` | RLDS dataset root |
| `DATASET_NAME` | `grasp_part_rlds` | FGManip dataset registered in OXE configs |
| `BATCH_SIZE` | `4` | Per-GPU batch size in the wrapper script |
| `LEARNING_RATE` | `5e-4` | LoRA fine-tuning learning rate |
| `MAX_STEPS` | `30005` | Total gradient steps |
| `SAVE_STEPS` | `5000` | Checkpoint save interval |
| `LORA_RANK` | `32` | LoRA rank |

Checkpoints are saved under `runs/` with names like:

```text
openvla-7b+grasp_part_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--fgmanip-openvla--image_aug--5000_chkpt
```

## Evaluation

Run evaluation with default settings or specify arguments positionally:

```bash
# Defaults: grasp_part task, object=3558, part=cap, 10 episodes
bash core/policies/openvla/eval.sh

# Custom checkpoint, task, episode count, object, and part
bash core/policies/openvla/eval.sh /path/to/checkpoint grasp_part 20 3763 cap

# Tasks without object/part (pass empty strings to skip)
bash core/policies/openvla/eval.sh /path/to/checkpoint plug_charger 10 "" ""
```

The evaluation script performs rollouts in ManiSkill environments, reports success rate, and saves rollout videos to `eval_videos/`.

### Evaluation Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained_checkpoint` | user-provided | Path to a merged checkpoint directory or HF model ID |
| `--task_name` | `grasp_part` | Task identifier |
| `--unnorm_key` | `grasp_part_rlds` | Dataset statistics key used for action unnormalization |
| `--num_open_loop_steps` | `1` | Standard OpenVLA emits one action at a time |
| `--center_crop` | `True` | Matches image augmentation recipe used in OpenVLA docs |
| `--save_video` | `True` | Save rollout videos to `eval_videos/` |

When `object_name` and `part_name` are provided, the task prompt is templated automatically (for example, `"grasp the cap"` instead of the generic `"grasp the handle"`).

## CPU-Only Verification

When GPU capacity is tight, prefer CPU-safe checks before running full training:

```bash
CUDA_VISIBLE_DEVICES="" python -c "from prismatic.vla.datasets import RLDSDataset"
CUDA_VISIBLE_DEVICES="" python -c "import tensorflow_datasets as tfds; print(tfds.builder('grasp_part_rlds', data_dir='/nat/demos/openvla/data').info.name)"
```

These checks verify imports and TFDS metadata without starting training or touching `GPU0`.
