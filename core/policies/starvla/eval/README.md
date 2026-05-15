# 基于starVLA codebase的eval
---

### 要改的地方
因为咱还没封装成库，所以开两个环境分别运行VLA和FGManip

两个bash文件需要修改
run_eval.sh
```bash
export Python=/export/anaconda3/envs/maniskill/bin/python
export FGManip_HOME=/export/xuhy/zpy/FGManip
CKPT_PATH=这里需要加载个VLA的权重
```
因为Qwen-GR00T train的时候是ee delta，所以代码里是control_mode是ee，action是7维，不同模型可能需要改control_mode

run_policy_server.sh:
同上

---

### 用法

开两个环境，环境A用于加载基于starVLA codebase的模型，环境B用于FGManip env

环境A: 
```bash
conda activate starVLA
cd core/vla/eval
bash run_policy_server.sh
```

环境B:
```bash
conda activate maniskill
cd core/vla/eval
bash starVLA/examples/FGManip/eval_files/run_eval.sh
```