# LeRobot Dataset v3.0 to v2.1 Converter

将 LeRobot v3.0 格式数据集转换为 v2.1 格式，支持单个转换和批量转换。

转换后的数据集按 **机型 / 数据集ID** 两级目录组织：

```
output_dir/
├── astribot_s1/
│   ├── 8d85f98d687942d28af78efea1257f32/
│   ├── 02420e0e72bb4891b4e8916bbdc05fdc/
│   └── ...
├── astribot_s2/
│   └── ...
└── ...
```

## 格式差异

| 特性 | v3.0 | v2.1 |
|------|------|------|
| 数据文件 | 合并 parquet（`data/chunk-000/file-000.parquet`） | 每 episode 独立（`data/chunk-000/episode_000000.parquet`） |
| 视频文件 | 合并 mp4（`videos/{cam}/chunk-000/file-000.mp4`） | 每 episode 独立（`videos/chunk-000/{cam}/episode_000000.mp4`） |
| Episode 元数据 | parquet（`meta/episodes/chunk-000/file-000.parquet`） | JSONL（`meta/episodes.jsonl`） |
| 统计信息 | 内嵌在 episodes parquet + `meta/stats.json` | JSONL（`meta/episodes_stats.jsonl`） |
| 任务信息 | parquet（`meta/tasks.parquet`） | JSONL（`meta/tasks.jsonl`） |

## 环境准备

### 1. 安装 lerobot

需要 lerobot v3.0 版本（提供底层工具函数）：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 2. 降级 datasets 库

```bash
pip install "datasets<4.0.0"
```

> `datasets>=4.0.0` 引入了不兼容的 `List`/`Column` 类型，需要使用旧版本。

### 3. 安装 ffmpeg

视频切分依赖 ffmpeg：

```bash
# Ubuntu/Debian
apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 转换单个数据集

```bash
python convert.py \
    --input /path/to/lerobot_v30/8d85f98d687942d28af78efea1257f32 \
    --output-dir /path/to/lerobot_v21
```

输出目录结构（自动按 `robot_type/dataset_id` 组织）：

```
/path/to/lerobot_v21/
└── astribot_s1/                                  # robot_type（自动从 info.json 读取）
    └── 8d85f98d687942d28af78efea1257f32/          # dataset_id
        ├── meta/
        │   ├── info.json
        │   ├── episodes.jsonl
        │   ├── episodes_stats.jsonl
        │   └── tasks.jsonl
        ├── data/
        │   └── chunk-000/
        │       ├── episode_000000.parquet
        │       ├── episode_000001.parquet
        │       └── ...
        ├── videos/
        │   └── chunk-000/
        │       ├── observation.images.head/
        │       │   ├── episode_000000.mp4
        │       │   └── ...
        │       ├── observation.images.torso/
        │       ├── observation.images.wrist_left/
        │       └── observation.images.wrist_right/
        └── images/  (如果原数据集有)
```

### 批量转换

扫描目录下所有 v3.0 数据集并逐个转换：

```bash
python convert.py \
    --input /path/to/lerobot_v30 \
    --output-dir /path/to/lerobot_v21 \
    --batch
```

### 不按机型分组（平铺模式）

如果不需要按机型分子文件夹，加 `--no-group-by-robot`：

```bash
python convert.py \
    --input /path/to/lerobot_v30 \
    --output-dir /path/to/lerobot_v21 \
    --batch --no-group-by-robot
```

输出变为：
```
/path/to/lerobot_v21/
├── 8d85f98d687942d28af78efea1257f32/
├── 02420e0e72bb4891b4e8916bbdc05fdc/
└── ...
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--input` | 是 | 单个 v3.0 数据集路径，或包含多个数据集的父目录（配合 `--batch`） |
| `--output-dir` | 是 | 输出根目录 |
| `--batch` | 否 | 批量模式：扫描 `--input` 下所有 v3.0 数据集 |
| `--repo-id-prefix` | 否 | HuggingFace repo ID 前缀（默认：`astribot`） |
| `--no-group-by-robot` | 否 | 不按机型分组，直接 `<output-dir>/<dataset_id>/` |

## 示例

```bash
# 以青龙数据集为例

# 单个转换
python convert.py \
    --input /qinglong_datasets/qinglong/lerobot/8d85f98d687942d28af78efea1257f32 \
    --output-dir /workspace/lerobot_v21

# 批量转换所有青龙数据集
python convert.py \
    --input /qinglong_datasets/qinglong/lerobot \
    --output-dir /workspace/lerobot_v21 \
    --batch
```

## 注意事项

- 机型信息从每个数据集的 `meta/info.json` 中的 `robot_type` 字段自动读取
- 视频切分使用 `ffmpeg -c copy`（stream copy），速度快且无质量损失
- 如果输出目录中已存在同名数据集，会**先删除再重新转换**
- 批量模式下单个数据集转换失败不会中断整体流程，最终会汇报成功/失败数量
- 转换耗时主要取决于视频数量和大小，200 episode x 4 相机约需 2-3 分钟

## 项目结构

```
lerobot_v30_to_v21/
├── README.md                        # 本文档
├── requirements.txt                 # Python 依赖
├── convert.py                       # 入口脚本（支持单个/批量、按机型+ID命名）
└── convert_dataset_v30_to_v21.py    # 核心转换逻辑
```
