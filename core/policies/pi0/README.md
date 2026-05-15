# 基于 LeRobot 的 PI0 推理脚本

与 `core/policies/pi05` 对齐：使用同仓库内 `pi05/Lerobot` 中的 `lerobot.policies.pi0` 加载 checkpoint，在 FGManip / ManiSkill 环境中闭环评测。本目录不自带 `Lerobot` 子树时，会自动把 `sys.path` 指到相邻目录 `../pi05/Lerobot/src`。

## 1 评估脚本

- `evaluate.py`：标准成功率与视频录制。
- `evaluate_path.py`：在 `evaluate.py` 基础上增加末端 TCP 轨迹保存（`tcp_pose.npy`），便于路径分析。

### 1.1 通用任务

```bash
python core/policies/pi0/evaluate.py \
  --policy-path /Your/path/to/pretrained_model \
  --env-id peg_in_hole \
  --obs-mode rgb \
  --control-mode pd_joint_delta_pos \
  --n-episodes 50 \
  --device cuda \
  --task "Description of this task" \
  --record-dir /Your/path/to/save \
  --save-video
```

### 1.2 需指定物体 / 部件的任务

```bash
python core/policies/pi0/evaluate.py \
  --policy-path /Your/path/to/pretrained_model \
  --env-id toggle_switch \
  --object-name 100920 \
  --part-name button \
  --obs-mode rgb \
  --control-mode pd_joint_delta_pos \
  --n-episodes 50 \
  --device cuda \
  --task "Description of this task" \
  --record-dir /Your/path/to/save \
  --save-video
```

### 1.3 域随机 / 相机与光照扰动（`evaluate_path.py`）

```bash
CUDA_VISIBLE_DEVICES=1 \
python core/policies/pi0/evaluate_path.py \
  --policy-path /Your/path/to/pretrained_model \
  --env-id grasp_part \
  --object-name 3558 \
  --part-name cap \
  --obs-mode rgb \
  --control-mode pd_joint_delta_pos \
  --n-episodes 10 \
  --device cuda \
  --task "Grasp the cap of the bottle" \
  --record-dir ./ \
  --save-video \
  --enable-dr-eval \
  --camera-pos-levels 0.03 0.06 0.12 \
  --camera-rot-levels-deg 2 6 12 \
  --light-ambient-delta-levels 0.10 0.25 0.40
```

**参数说明**

- `--policy-path`：LeRobot 训练输出的 `pretrained_model` 目录（含 `config.json` 与权重）。
- `--task`：与训练时一致的自然语言指令（或通过 `train_config.json` 自动从数据集 `tasks.parquet` 推断）。

## 2 数据与归一化

PI0 默认使用 **Quantile Normalization**；数据集需包含 `q01` / `q99` 等统计量，否则训练预处理可能报错 `QUANTILES normalization mode requires q01 and q99 stats`。说明与 `pi05` 相同。

## 3 环境依赖

需要已安装 **ManiSkill**、**LeRobot（含 `[pi]` 依赖）**，并与训练 PI0 时一致。若使用本仓库 `pi05` 下 vendored 的 LeRobot，请保持该路径可用；否则可安装独立 `lerobot` 包并确保其在 `PYTHONPATH` 中优先于旧版本。

```bash
pip install --upgrade mani_skill
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```
