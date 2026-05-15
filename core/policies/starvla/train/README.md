# 基于starVLA codebase的training

---

```bash
conda activate starVLA
```

将数据如lerobot_xx放在FGManip/core/policies/starvla/starVLA/playground/Dataset/

修改core/policies/vla/starvla/starVLA/dataloader/gr00t_lerobot/mixtures.py里的fgmanip

starVLA不太兼容lerobot3.0格式，可以用lerobot_v30_to_v21转成2.1格式, 需要在meta中加入modality.json

```bash
bash run_libero_train.sh
```