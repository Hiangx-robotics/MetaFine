# 基于LeRobot的pi05推理脚本

## 1 评估脚本用法

评估脚本位于 `FGManip/core/policies/pi05/evaluate.py`。该脚本支持加载 LeRobot 微调后的 Checkpoint，并在 FGManip 仿真环境中进行闭环测试。

根据任务类型（通用任务 vs 特定物体/部件任务），有两种主要的调用方式：

### 1.1 通用任务评估

适用于不需要指定特定物体 ID 或部件名称的任务，例如 `plug_charger`。

```bash
python core/policies/pi05/evaluate.py \
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

**参数说明：**

* `--policy-path`: 指向 LeRobot 训练输出的 `pretrained_model` 目录（包含 config.json 和 safetensors）。
* `--env-id`: FGManip 环境 ID。
* `--task`: 提供与训练时语义一致的自然语言任务描述。

### 1.2 特定物体任务评估

适用于需要操作特定物体或部件的任务，例如 `toggle_switch`，需要通过 `--object-name` 和 `--part-name` 指定具体的仿真资产。

```bash
python FGManip/core/policies/pi05/evaluate.py \
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


CUDA_VISIBLE_DEVICES =1  \
python core/policies/pi05/evaluate_path.py \
  --policy-path /nat/demos/pi05/outputs/pi0_grasppart/checkpoints/030000/pretrained_model \
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

**差异说明：**

* 增加 `--object-name`: 指定物体的 ID。
* 增加 `--part-name`: 指定操作的部件。

---

## 2 数据转换脚本修改说明

### 归一化统计量计算

* **修改内容**：脚本集成了统计量计算逻辑，生成包含 `q01` (1% 分位数) 和 `q99` (99% 分位数) 的 `stats.json` 文件。例如：
    ```bash
    'q01': np.percentile(actions, 1, axis=0).tolist(),
    'q99': np.percentile(actions, 99, axis=0).tolist()
    ```
* **原因**：Pi05 默认使用 **Quantile Normalization**（分位数归一化）来处理 Action 和 State。如果数据集中缺少这些统计信息，训练脚本会在预处理阶段抛出 `ValueError: QUANTILES normalization mode requires q01 and q99 stats` 错误。

---

## 3. 环境安装

**由于本测试脚本是基于LeRobot中的pi05模型进行实现，需要同时安装maniskill和lerobot中的库**

***maniskill***
```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
```
***

***lerobot***
```bash
pip install -e ".[pi]"
```
For lerobot 0.4.0, if you want to install pi tag, you will have to do: 
```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```

***tips***
```bash
pip install numpy==1.26.4
```
50系显卡记得
```bash
pip install torch==2.10.0 torchvision torchaudio
```