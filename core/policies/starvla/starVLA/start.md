## StarVLA + LIBERO training + eval

## 安装 StarVLA 环境

- **创建 conda 环境并安装依赖**

```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA

pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .

python playground/Pretrained_models/download.py # 下载https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
python starVLA/model/framework/add_token.py # 注册latent token
```

- **快速自检（可选）**

```bash
python starVLA/model/framework/LatentVLA.py
```

若能正常运行并打印模型结构，说明 StarVLA 环境基本就绪。

---

## 2. 准备 LIBERO 仿真环境
ps：这里有点烦，**eval**的时候要开starVLA和libero两个环境😡。

- **安装 LIBERO 环境**

```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
cd LIBERO

conda create -n libero python=3.10 -y
conda activate libero

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4

cd ../starVLA
```
---

## 3. 下载 LIBERO 训练数据

```bash
mkdir ../dataset
bash examples/LIBERO/data_preparation.sh ../dataset 
```
这里我每次下载一段时间就会断开，说下载太多了，得再bash一下，我下载了很多次😡

完成后检查：

```bash
ls playground/Datasets/LEROBOT_LIBERO_DATA
# 应看到 libero_spatial_no_noops_1.0.0_lerobot 等四个子目录
```

---

## 4. LIBERO 训练

打开 `examples/LIBERO/train_files/run_libero_train.sh`，根据自己环境修改里面的配置

这里设计到多卡的训练我不是很懂😭，里面有些设置我不太懂所以没改，可能得商讨一下🧎

改好了调用
```bash
conda activate starVLA
bash examples/LIBERO/train_files/run_libero_train.sh
```

---

## 5. 在 LIBERO 仿真中评估 / 验证

评估需要 **两个终端 + 两个环境**：

- 终端 A
```bash
conda activate starVLA
bash examples/LIBERO/eval_files/run_policy_server.sh`
```
- 终端 B  
```bash
conda activate starVLA
bash examples/LIBERO/eval_files/eval_libero.sh`
```