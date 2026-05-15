"""PI0.5 policy evaluation on FGManip/ManiSkill environments.

使用与 lerobot 官方 lerobot_eval.py 一致的方式加载 checkpoint（模型 + preprocessor +
postprocessor），然后在 FGManip 自定义环境中运行 rollout 并统计成功率。

Usage:
    CUDA_VISIBLE_DEVICES=0 python FGManip/core/policies/pi05/evaluate.py \
        --policy-path /path/to/checkpoint/pretrained_model \
        --env-id plug_charger \
        --n-episodes 10 --device cuda \
        --task "Pick up the charger plug, precisely align it with the socket, ..."
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from random import randint
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from tqdm import tqdm
from mani_skill.utils import gym_utils

# ============================= Dependency Stubs ==============================
# LeRobot import 链会触发 serial / deepdiff / av 等可选依赖，
# 纯仿真评测不需要它们，这里用最小 stub 跳过 ImportError。
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def _install_stubs() -> None:
    """Install minimal stubs for optional LeRobot dependencies."""

    # --- serial (pyserial) ---
    if "serial" not in sys.modules:
        try:
            import serial  # noqa: F401
        except ImportError:
            s = types.ModuleType("serial")
            t = types.ModuleType("serial.tools")
            lp = types.ModuleType("serial.tools.list_ports")
            lp.comports = lambda: []  # type: ignore[attr-defined]
            t.list_ports = lp  # type: ignore[attr-defined]
            s.tools = t  # type: ignore[attr-defined]
            for k, v in {"serial": s, "serial.tools": t, "serial.tools.list_ports": lp}.items():
                sys.modules.setdefault(k, v)

    # --- deepdiff ---
    if "deepdiff" not in sys.modules:
        try:
            from deepdiff import DeepDiff  # noqa: F401
        except ImportError:
            m = types.ModuleType("deepdiff")

            class _DD:
                def __init__(self, *a, **kw): pass
                def to_dict(self): return {}

            m.DeepDiff = _DD  # type: ignore[attr-defined]
            sys.modules.setdefault("deepdiff", m)

    # --- av (PyAV) ---
    if "av" not in sys.modules:
        try:
            import av  # noqa: F401
        except ImportError:
            av_mod = types.ModuleType("av")
            vf_mod = types.ModuleType("av.video.frame")
            v_mod = types.ModuleType("av.video")

            class _Logging:
                ERROR = 40
                restore_default_callback = staticmethod(lambda: None)
                set_level = staticmethod(lambda *a, **k: None)

            class _VF:
                pict_type = None
                from_image = staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("av not installed")))
                from_ndarray = staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("av not installed")))

            class _PT:
                NONE = 0

            class _AVErr(Exception):
                pass

            vf_mod.VideoFrame = _VF  # type: ignore[attr-defined]
            vf_mod.PictureType = _PT  # type: ignore[attr-defined]
            v_mod.frame = vf_mod  # type: ignore[attr-defined]
            av_mod.logging = _Logging  # type: ignore[attr-defined]
            av_mod.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("av not installed"))  # type: ignore[attr-defined]
            av_mod.VideoFrame = _VF  # type: ignore[attr-defined]
            av_mod.AudioFrame = type("_AF", (), {"from_ndarray": staticmethod(lambda *a, **k: None)})  # type: ignore[attr-defined]
            av_mod.AVError = _AVErr  # type: ignore[attr-defined]
            av_mod.FFmpegError = _AVErr  # type: ignore[attr-defined]
            av_mod.video = v_mod  # type: ignore[attr-defined]
            av_mod.time_base = 1.0  # type: ignore[attr-defined]
            for k, v in {"av": av_mod, "av.video": v_mod, "av.video.frame": vf_mod}.items():
                sys.modules.setdefault(k, v)


# ============================= Path Setup ====================================

def _setup_paths() -> None:
    """Ensure Lerobot (local) and FGManip/ are on sys.path.
    
    优先使用本地封装的 Lerobot（在 evaluate.py 同目录下的 Lerobot/src/），
    如果不存在则回退到项目根目录的 lerobot/src。
    """
    script_dir = Path(__file__).resolve().parent
    
    # 优先使用本地封装的 Lerobot
    local_lerobot_src = script_dir / "Lerobot" / "src"
    if local_lerobot_src.exists() and local_lerobot_src.is_dir():
        lerobot_path = str(local_lerobot_src)
        if lerobot_path not in sys.path:
            sys.path.insert(0, lerobot_path)
    else:
        # 回退到项目根目录的 lerobot/src
        eai_root = Path(__file__).resolve().parents[4]
        lerobot_path = str(eai_root / "lerobot" / "src")
        if os.path.isdir(lerobot_path) and lerobot_path not in sys.path:
            sys.path.insert(0, lerobot_path)
    
    # 优先当前仓库（.../git）中的 core/skill/env，避免误导入到并行目录 FGManip/core
    repo_root = Path(__file__).resolve().parents[3]  # /export/xuhy/EAI/git
    repo_root_str = str(repo_root)
    if (repo_root / "core").is_dir() and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # 兼容兜底：仅在当前仓库不存在 core 包时，才回退到 /export/xuhy/EAI/FGManip
    eai_root = Path(__file__).resolve().parents[4]
    fgmanip_path = str(eai_root / "FGManip")
    if not (repo_root / "core").is_dir():
        if os.path.isdir(fgmanip_path) and fgmanip_path not in sys.path:
            sys.path.insert(0, fgmanip_path)


_install_stubs()
_setup_paths()

# ============================= LeRobot Imports ===============================
# 必须在 stubs 和 path setup 之后 import

import core.env   # noqa: E402,F401  — 注册 FGManip gymnasium 环境
import core.skill  # noqa: E402,F401
from utils.util import compute_mad, select_action_subspace, summarize_metric  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline, PolicyAction  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402
from mani_skill.utils.wrappers import CPUGymWrapper  # noqa: E402


# ============================= Configuration =================================

@dataclass
class EvalArgs:
    """命令行参数。"""
    policy_path: str                        # checkpoint 的 pretrained_model 目录
    env_id: str = "plug_charger"
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_joint_delta_pos"
    obs_mode: str = "rgb"

    object_name: str | None = None          # 仅 toggle_switch 等需要
    part_name: str | None = None

    n_episodes: int = 10
    max_episode_steps: int | None = None
    seed: int | None = None

    device: str = "cuda"
    tokenizer_path: str | None = None

    record_dir: str = "demos/pi05_eval"
    save_video: bool = True
    traj_name: str = "trajectory"

    task: str | None = None                 # language task 文本
    default_last_action: float = 0.0        # action 维度不足时补齐末尾
    enable_dr_eval: bool = False
    camera_pos_levels: list[float] | None = None
    camera_rot_levels_deg: list[float] | None = None
    light_ambient_delta_levels: list[float] | None = None
    save_cam_overlay: bool = True
    cam_overlay_alpha: float = 0.45
    cam_video_name: str = "trajectory_cam_overlay"
    obs_video_name: str = "trajectory_obs"


def parse_args() -> EvalArgs:
    p = argparse.ArgumentParser(description="Evaluate PI0.5 on FGManip envs")
    p.add_argument("--policy-path", required=True)
    p.add_argument("--env-id", default=EvalArgs.env_id)
    p.add_argument("--sim-backend", default=EvalArgs.sim_backend)
    p.add_argument("--control-mode", default=EvalArgs.control_mode)
    p.add_argument("--obs-mode", default=EvalArgs.obs_mode)
    p.add_argument("--object-name", default=None)
    p.add_argument("--part-name", default=None)
    p.add_argument("--n-episodes", type=int, default=EvalArgs.n_episodes)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=EvalArgs.device)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--record-dir", default=EvalArgs.record_dir)
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--traj-name", default=EvalArgs.traj_name)
    p.add_argument("--task", default=None,
                   help="task 文本；不指定则自动从训练数据集的 tasks.parquet 中提取")
    p.add_argument("--default-last-action", type=float, default=0.0)
    p.add_argument("--enable-dr-eval", action="store_true", default=False)
    p.add_argument("--camera-pos-levels", type=float, nargs="*", default=[0.03, 0.06, 0.12])
    p.add_argument("--camera-rot-levels-deg", type=float, nargs="*", default=[2.0, 6.0, 12.0])
    p.add_argument("--light-ambient-delta-levels", type=float, nargs="*", default=[0.10, 0.25, 0.40])
    p.add_argument("--save-cam-overlay", action="store_true", default=True)
    p.add_argument("--no-cam-overlay", action="store_true")
    p.add_argument("--cam-overlay-alpha", type=float, default=0.45)
    p.add_argument("--cam-video-name", default="trajectory_cam_overlay")
    p.add_argument("--obs-video-name", default="trajectory_obs")
    ns = p.parse_args()
    return EvalArgs(
        policy_path=ns.policy_path, env_id=ns.env_id, sim_backend=ns.sim_backend,
        control_mode=ns.control_mode, obs_mode=ns.obs_mode,
        object_name=ns.object_name, part_name=ns.part_name,
        n_episodes=ns.n_episodes, max_episode_steps=ns.max_episode_steps,
        seed=ns.seed, device=ns.device, tokenizer_path=ns.tokenizer_path,
        record_dir=ns.record_dir, save_video=(not ns.no_video and ns.save_video),
        traj_name=ns.traj_name, task=ns.task,
        default_last_action=ns.default_last_action,
        enable_dr_eval=ns.enable_dr_eval,
        camera_pos_levels=ns.camera_pos_levels,
        camera_rot_levels_deg=ns.camera_rot_levels_deg,
        light_ambient_delta_levels=ns.light_ambient_delta_levels,
        save_cam_overlay=(ns.save_cam_overlay and (not ns.no_cam_overlay)),
        cam_overlay_alpha=ns.cam_overlay_alpha,
        cam_video_name=ns.cam_video_name,
        obs_video_name=ns.obs_video_name,
    )


# ============================= Model Loading =================================
# 与官方 lerobot_eval.py 的 make_policy + make_pre_post_processors 完全一致

def load_policy(
    policy_path: str,
    device: str,
    tokenizer_path: str | None = None,
) -> tuple[PI05Policy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """加载 PI05 模型、preprocessor、postprocessor。

    流程与 lerobot 官方 eval 完全一致：
    1. PreTrainedConfig.from_pretrained → 读取 config.json
    2. PI05Policy.from_pretrained    → 加载模型权重
    3. make_pre_post_processors      → 加载 preprocessor/postprocessor
    """
    # ---- 加载 config ----
    cfg: PreTrainedConfig = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device

    # ---- 加载模型（与 make_policy 内部逻辑一致）----
    policy = PI05Policy.from_pretrained(policy_path, config=cfg)
    policy.to(device)
    policy.eval()

    # ---- 修复 tied weight（PaliGemma embed_tokens ↔ lm_head）----
    try:
        pg = policy.model.paligemma_with_expert.paligemma
        if not torch.equal(pg.lm_head.weight.data,
                           pg.model.language_model.embed_tokens.weight.data):
            pg.model.language_model.embed_tokens.weight = pg.lm_head.weight
            print("✓ Fixed tied weight: embed_tokens = lm_head")
    except AttributeError:
        pass

    # ---- 加载 preprocessor / postprocessor（与 make_pre_post_processors 一致）----
    overrides: dict[str, Any] = {"device_processor": {"device": device}}
    if tokenizer_path:
        overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    return policy, preprocessor, postprocessor


# ==================== ManiSkill Observation → Policy Batch ===================

def maniskill_obs_to_batch(
    obs: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """将 ManiSkill 环境的 observation 转换为 LeRobot policy 输入格式。

    ManiSkill obs 结构:
        obs["agent"]["qpos"]               → shape (1, 9)   → observation.state
        obs["sensor_data"][cam]["rgb"]      → shape (1,H,W,3) → observation.images.{cam}

    LeRobot policy 期望:
        {"observation.state": Tensor(1, 9),
         "observation.images.base_camera": Tensor(1, 3, H, W),
         "observation.images.hand_camera":  Tensor(1, 3, H, W),
         "task": str}
    """
    # ---- State ----
    qpos = obs["agent"]["qpos"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu()
    state = torch.as_tensor(np.asarray(qpos, dtype=np.float32))

    # ---- Images ----
    images: dict[str, torch.Tensor] = {}
    for cam_name in ("base_camera", "hand_camera"):
        rgb = obs["sensor_data"][cam_name]["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu()
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        # (1, H, W, 3) → (1, 3, H, W), uint8 → float32 [0,1]
        if rgb_np.ndim == 3:                       # (H, W, 3) 无 batch
            rgb_np = rgb_np[np.newaxis]
        img_t = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).contiguous().float() / 255.0
        images[f"{OBS_IMAGES}.{cam_name}"] = img_t

    return {OBS_STATE: state, "task": task, **images}


# ==================== Action Adaptation ======================================

def adapt_action(action: np.ndarray, env_dim: int, default_last: float) -> np.ndarray:
    """将 policy 输出的 action 适配到 env.action_space 维度。"""
    policy_dim = action.shape[-1]
    if policy_dim == env_dim:
        return action
    if policy_dim == env_dim - 1:
        pad = np.full((*action.shape[:-1], 1), default_last, dtype=action.dtype)
        return np.concatenate([action, pad], axis=-1)
    if policy_dim > env_dim:
        return action[..., :env_dim]
    raise ValueError(f"action 维度不匹配: policy={policy_dim}, env={env_dim}")


# ==================== Task Text ==============================================

def resolve_task_text(args: EvalArgs) -> str:
    """确定 task 语言指令：优先用户指定 → 自动从训练数据读取 → 兜底用 env_id。"""
    if args.task:
        return args.task

    # 尝试从 train_config.json → tasks.parquet 自动读取
    try:
        import json, pandas as pd
        train_cfg_path = osp.join(args.policy_path, "train_config.json")
        if osp.exists(train_cfg_path):
            with open(train_cfg_path) as f:
                root = json.load(f).get("dataset", {}).get("root", "")
            tasks_path = osp.join(root, "meta", "tasks.parquet")
            if osp.exists(tasks_path):
                texts = pd.read_parquet(tasks_path).index.tolist()
                if texts and isinstance(texts[0], str) and texts[0].strip():
                    print(f"Auto-detected task: '{texts[0]}'")
                    return texts[0]
    except Exception as e:
        print(f"Warning: 无法自动提取 task 文本: {e}")

    fallback = args.env_id
    print(f"Using fallback task: '{fallback}' (建议通过 --task 指定训练时使用的 task 文本)")
    return fallback


# ==================== Camera Resolution ======================================

def get_training_image_hw(cfg: PreTrainedConfig) -> tuple[int, int]:
    """从 checkpoint config 中提取训练时的图像分辨率 (H, W)。"""
    for feat in cfg.input_features.values():
        if hasattr(feat, "type") and str(getattr(feat, "type", "")).upper() == "VISUAL":
            shape = feat.shape  # (C, H, W)
            if len(shape) == 3:
                return shape[1], shape[2]
    return 480, 640  # 默认值


def _extract_success_at_end(final_info: Any) -> bool:
    """与 ACT evaluate 一致：优先 success_at_end，其次 success。"""
    try:
        if isinstance(final_info, dict):
            ep_info = final_info.get("episode", {})
            s = ep_info.get("success_at_end", ep_info.get("success", False))
        else:
            # 单环境通常不会走到这里；做兼容处理
            first = final_info[0] if len(final_info) > 0 else {}
            ep_info = first.get("episode", {}) if isinstance(first, dict) else {}
            s = ep_info.get("success_at_end", ep_info.get("success", False))

        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        if isinstance(s, np.ndarray):
            s = np.asarray(s).reshape(-1)[0] if s.size else False
        return bool(s)
    except Exception:
        return False


def compute_ausc(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, num=arr.size)
    return float(np.trapz(arr, x))


def _to_uint8_rgb(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.max(arr) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_min = float(x.min())
    x_max = float(x.max())
    return (x - x_min) / max(x_max - x_min, eps)


def _colorize_heatmap(heatmap_01: np.ndarray) -> np.ndarray:
    h = np.clip(heatmap_01, 0.0, 1.0)
    r = (255.0 * h).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(h - 0.5) * 2.0) * 0.75).astype(np.uint8)
    b = (255.0 * (1.0 - h)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@torch.no_grad()
def _overlay_pi05_cam(policy: PI05Policy, image_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    image_rgb = _to_uint8_rgb(image_rgb)
    device = next(policy.parameters()).device
    vision_dtype = next(policy.model.paligemma_with_expert.paligemma.vision_tower.parameters()).dtype

    target_h, target_w = policy.config.image_resolution
    img = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).to(device)[None]  # [1, H, W, C]
    if img.shape[1:3] != (target_h, target_w):
        img = resize_with_pad_torch(img, target_h, target_w)
    img = img * 2.0 - 1.0
    img = img.permute(0, 3, 1, 2).contiguous().to(dtype=vision_dtype)  # [1, 3, H, W]

    token_features = policy.model.paligemma_with_expert.embed_image(img)  # [1, N, D]
    token_features = token_features[0]
    if token_features.dim() != 2:
        return image_rgb

    n_tokens = token_features.shape[0]
    n = int(np.sqrt(n_tokens))
    if n * n != n_tokens and n_tokens > 1:
        n2 = int(np.sqrt(n_tokens - 1))
        if n2 * n2 == (n_tokens - 1):
            token_features = token_features[1:]
            n_tokens = token_features.shape[0]
            n = int(np.sqrt(n_tokens))
    if n * n != n_tokens:
        return image_rgb

    token_scores = torch.linalg.vector_norm(token_features, ord=2, dim=-1).view(n, n)
    score_map = F.interpolate(
        token_scores[None, None].float(),
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()
    score_map = _normalize_01(score_map)
    heat_rgb = _colorize_heatmap(score_map).astype(np.float32)
    base_rgb = image_rgb.astype(np.float32)
    out = np.clip((1.0 - alpha) * base_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)
    return out


def _save_cam_video(frames: list[np.ndarray], video_path: str, fps: int = 30) -> None:
    if not frames:
        return
    os.makedirs(osp.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in frames:
        writer.append_data(_to_uint8_rgb(frame))
    writer.close()


def _get_tcp_position_xyz(env: Any) -> np.ndarray:
    """Read current TCP position xyz from the unwrapped env."""
    tcp_p = env.unwrapped.agent.tcp.pose.p
    if isinstance(tcp_p, torch.Tensor):
        tcp_p = tcp_p.detach().cpu().numpy()
    tcp_p = np.asarray(tcp_p, dtype=np.float32)
    if tcp_p.ndim == 2:
        tcp_p = tcp_p[0]
    return tcp_p.reshape(-1)[:3].copy()


def build_eval_profiles(base_env_kwargs: dict[str, Any], args: EvalArgs) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {"clean": dict(base_env_kwargs)}
    if not args.enable_dr_eval:
        return profiles

    cam_pos_levels = list(args.camera_pos_levels or [0.03, 0.06, 0.12])[:3]
    cam_rot_levels = list(args.camera_rot_levels_deg or [2.0, 6.0, 12.0])[:3]
    while len(cam_pos_levels) < 3:
        cam_pos_levels.append(cam_pos_levels[-1] if cam_pos_levels else 0.0)
    while len(cam_rot_levels) < 3:
        cam_rot_levels.append(cam_rot_levels[-1] if cam_rot_levels else 0.0)

    for idx, (pos_j, rot_j) in enumerate(zip(cam_pos_levels, cam_rot_levels), start=1):
        profiles[f"cam_l{idx}"] = {
            **base_env_kwargs,
            "eval_randomize_camera": True,
            "eval_camera_pos_jitter": float(pos_j),
            "eval_camera_rot_jitter_deg": float(rot_j),
            "eval_randomize_light": False,
        }

    light_levels = list(args.light_ambient_delta_levels or [0.10, 0.25, 0.40])[:3]
    while len(light_levels) < 3:
        light_levels.append(light_levels[-1] if light_levels else 0.0)

    for idx, delta in enumerate(light_levels, start=1):
        delta = max(0.0, float(delta))
        low = max(0.0, 0.5 - delta)
        high = min(1.0, 0.5 + delta)
        profiles[f"light_l{idx}"] = {
            **base_env_kwargs,
            "eval_randomize_light": True,
            "eval_ambient_low": low,
            "eval_ambient_high": high,
            "eval_randomize_camera": False,
        }

    return profiles


# ============================= Main ==========================================

def main() -> None:
    args = parse_args()

    if args.tokenizer_path:
        os.environ["LEROBOT_TOKENIZER_PATH"] = args.tokenizer_path

    # ---- 加载模型（与官方 lerobot_eval.py 一致）----
    policy, preprocessor, postprocessor = load_policy(
        args.policy_path, args.device, args.tokenizer_path
    )

    # ---- 确定 task 文本 ----
    task = resolve_task_text(args)

    # ---- 确定相机分辨率（匹配训练数据）----
    cam_h, cam_w = get_training_image_hw(policy.config)
    print(f"Camera resolution: {cam_w}×{cam_h}")

    # ---- 构建环境参数 ----
    env_kwargs: dict[str, Any] = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="rt", width=cam_w, height=cam_h),
        human_render_camera_configs=dict(shader_pack="rt"),
        viewer_camera_configs=dict(shader_pack="rt"),
        sim_backend=args.sim_backend,
    )
    if args.object_name is not None:
        env_kwargs["object_name"] = args.object_name
    if args.part_name is not None:
        env_kwargs["part_name"] = args.part_name
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    # ---- Evaluation loop (clean + optional DR profiles) ----
    profiles = build_eval_profiles(env_kwargs, args)
    profile_results: dict[str, dict[str, Any]] = {}

    episode_seeds = [
        (args.seed + i if args.seed is not None else randint(0, 1_000_000))
        for i in range(args.n_episodes)
    ]

    for profile_name, profile_env_kwargs in profiles.items():
        successes: list[bool] = []
        exp_neg_mad_all_list: list[float] = []
        exp_neg_mad_success_list: list[float] = []
        pbar = tqdm(range(args.n_episodes), desc=f"Eval {args.env_id} [{profile_name}]")

        for ep_idx in pbar:
            trial_dir = osp.join(
                args.record_dir, args.env_id, profile_name, f"trial_{ep_idx + 1:04d}"
            )
            os.makedirs(trial_dir, exist_ok=True)

            env = gym.make(args.env_id, **profile_env_kwargs)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            if args.save_video:
                env = RecordEpisode(
                    env, output_dir=trial_dir, trajectory_name=args.traj_name,
                    # Keep trajectory recording but disable render-camera mp4.
                    # We save observation-view videos manually from sensor_data below.
                    save_video=False, record_reward=False, video_fps=30, save_on_reset=False,
                )

            obs, info = env.reset(seed=int(episode_seeds[ep_idx]))
            policy.reset()

            # 固定执行步数：优先用户指定，否则取环境 max_episode_steps
            target_steps = args.max_episode_steps or gym_utils.find_max_episode_steps_value(env)
            last_info: dict[str, Any] = {}
            env_act_dim = int(np.prod(env.action_space.shape))
            episode_actions: list[np.ndarray] = []
            tcp_rollout_positions: list[np.ndarray] = []
            obs_rollout_frames: list[np.ndarray] = []
            obs_rollout_frames_wrist: list[np.ndarray] = []
            cam_rollout_frames: list[np.ndarray] = []
            cam_rollout_frames_wrist: list[np.ndarray] = []

            try:
                tcp_rollout_positions.append(_get_tcp_position_xyz(env))
            except Exception as tcp_e:
                print(f"Warning: tcp pose collection failed in ep={ep_idx} at reset: {tcp_e}")

            for _ in range(int(target_steps)):
                if args.save_video:
                    try:
                        base_rgb = obs["sensor_data"]["base_camera"]["rgb"]
                        obs_rollout_frames.append(_to_uint8_rgb(base_rgb))
                        if "hand_camera" in obs.get("sensor_data", {}):
                            wrist_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
                            obs_rollout_frames_wrist.append(_to_uint8_rgb(wrist_rgb))
                    except Exception as obs_e:
                        print(f"Warning: obs-view frame collection failed in ep={ep_idx}: {obs_e}")

                # 1. 转换观测 → policy batch（ManiSkill 格式 → LeRobot 格式）
                if args.save_video and args.save_cam_overlay:
                    try:
                        base_rgb = obs["sensor_data"]["base_camera"]["rgb"]
                        cam_frame = _overlay_pi05_cam(
                            policy=policy,
                            image_rgb=_to_uint8_rgb(base_rgb),
                            alpha=float(args.cam_overlay_alpha),
                        )
                        cam_rollout_frames.append(cam_frame)
                        if "hand_camera" in obs.get("sensor_data", {}):
                            wrist_rgb = obs["sensor_data"]["hand_camera"]["rgb"]
                            wrist_frame = _overlay_pi05_cam(
                                policy=policy,
                                image_rgb=_to_uint8_rgb(wrist_rgb),
                                alpha=float(args.cam_overlay_alpha),
                            )
                            cam_rollout_frames_wrist.append(wrist_frame)
                    except Exception as cam_e:
                        print(f"Warning: PI05 CAM overlay failed in ep={ep_idx}, step={len(cam_rollout_frames)}: {cam_e}")

                batch = maniskill_obs_to_batch(obs, task)

                # 2. preprocessor → policy → postprocessor（与官方 rollout 流程一致）
                batch = preprocessor(batch)
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)

                # 3. 转为 numpy 并适配维度
                act_np = action.detach().cpu().numpy()
                if act_np.ndim == 1:
                    act_np = act_np[np.newaxis]
                act_np = adapt_action(act_np, env_act_dim, args.default_last_action)

                # 4. 记录用于 smoothness 的动作子空间（默认不含 gripper）
                episode_actions.append(
                    select_action_subspace(
                        act_np[0],
                        control_mode=args.control_mode,
                        include_gripper=False,
                    ).copy()
                )

                # 5. 执行
                obs, _rew, _terminated, _truncated, info = env.step(act_np[0])
                try:
                    tcp_rollout_positions.append(_get_tcp_position_xyz(env))
                except Exception as tcp_e:
                    print(
                        f"Warning: tcp pose collection failed in ep={ep_idx}, "
                        f"step={len(tcp_rollout_positions)}: {tcp_e}"
                    )
                if isinstance(info, dict):
                    last_info = info

            # ---- Episode 结束 ----
            try:
                tcp_pose_path = osp.join(trial_dir, "tcp_pose.npy")
                np.save(tcp_pose_path, np.asarray(tcp_rollout_positions, dtype=np.float32))
            except Exception as tcp_save_e:
                print(f"Warning: tcp pose saving failed in ep={ep_idx}: {tcp_save_e}")

            if args.save_video:
                try:
                    env.flush_trajectory()
                    env.flush_video()
                    if args.save_cam_overlay:
                        cam_video_path = osp.join(trial_dir, f"{args.cam_video_name}.mp4")
                        _save_cam_video(cam_rollout_frames, cam_video_path, fps=30)
                        if cam_rollout_frames_wrist:
                            cam_wrist_video_path = osp.join(
                                trial_dir, f"{args.cam_video_name}_wrist.mp4"
                            )
                            _save_cam_video(cam_rollout_frames_wrist, cam_wrist_video_path, fps=30)
                    obs_video_path = osp.join(trial_dir, f"{args.obs_video_name}.mp4")
                    _save_cam_video(obs_rollout_frames, obs_video_path, fps=30)
                    if obs_rollout_frames_wrist:
                        obs_wrist_video_path = osp.join(trial_dir, f"{args.obs_video_name}_wrist.mp4")
                        _save_cam_video(obs_rollout_frames_wrist, obs_wrist_video_path, fps=30)
                except Exception:
                    pass
            try:
                env.close()
            except Exception:
                pass

            ep_success = False
            if isinstance(last_info, dict) and "final_info" in last_info:
                ep_success = _extract_success_at_end(last_info["final_info"])
            elif isinstance(last_info, dict) and "success" in last_info:
                ep_success = bool(last_info["success"])

            ep_mad = compute_mad(
                np.asarray(episode_actions, dtype=np.float64), norm="l2", dt=1.0
            )
            if np.isfinite(ep_mad):
                exp_neg_mad = float(np.exp(-float(ep_mad)))
                exp_neg_mad_all_list.append(exp_neg_mad)
                if ep_success:
                    exp_neg_mad_success_list.append(exp_neg_mad)

            successes.append(ep_success)
            pbar.set_postfix(success_rate=f"{np.mean(successes):.3f}")

        exp_neg_mad_all_stats = summarize_metric(exp_neg_mad_all_list)
        exp_neg_mad_success_stats = summarize_metric(exp_neg_mad_success_list)
        result = {
            "episodes": int(len(successes)),
            "successes": int(np.sum(successes)),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
            "exp_neg_mad_all": exp_neg_mad_all_stats,
            "exp_neg_mad_success": exp_neg_mad_success_stats,
        }
        profile_results[profile_name] = result
        print(f"\n{'='*48}")
        print(f"PI0.5 Eval on {args.env_id} [{profile_name}]")
        print(f"  episodes:     {result['episodes']}")
        print(f"  successes:    {result['successes']}")
        print(f"  success_rate: {result['success_rate']:.3f}")
        print(
            f"  exp(-mad)_all:     "
            f"{exp_neg_mad_all_stats['mean']:.6f} (count={exp_neg_mad_all_stats['count']})"
        )
        print(
            f"  exp(-mad)_success: "
            f"{exp_neg_mad_success_stats['mean']:.6f} "
            f"(count={exp_neg_mad_success_stats['count']})"
        )
        print(f"{'='*48}")

    if args.enable_dr_eval:
        ausc_values: list[float] = []
        if all(name in profile_results for name in ["clean", "cam_l1", "cam_l2", "cam_l3"]):
            cam_srs = [
                profile_results[name]["success_rate"]
                for name in ["clean", "cam_l1", "cam_l2", "cam_l3"]
            ]
            cam_ausc = compute_ausc(cam_srs)
            print(f"AUSC(camera): {cam_ausc:.6f}")
            profile_results["ausc_camera"] = {"value": cam_ausc}
            if np.isfinite(cam_ausc):
                ausc_values.append(float(cam_ausc))
        if all(name in profile_results for name in ["clean", "light_l1", "light_l2", "light_l3"]):
            light_srs = [
                profile_results[name]["success_rate"]
                for name in ["clean", "light_l1", "light_l2", "light_l3"]
            ]
            light_ausc = compute_ausc(light_srs)
            print(f"AUSC(light): {light_ausc:.6f}")
            profile_results["ausc_light"] = {"value": light_ausc}
            if np.isfinite(light_ausc):
                ausc_values.append(float(light_ausc))
        if ausc_values:
            ausc_mean = float(np.mean(np.asarray(ausc_values, dtype=np.float64)))
            print(f"AUSC(mean): {ausc_mean:.6f}")
            profile_results["ausc_mean"] = {"value": ausc_mean}

    summary_dir = osp.join(args.record_dir, args.env_id)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = osp.join(summary_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(profile_results, f, ensure_ascii=False, indent=2)
    print(f"Saved eval summary to: {summary_path}")


if __name__ == "__main__":
    main()
