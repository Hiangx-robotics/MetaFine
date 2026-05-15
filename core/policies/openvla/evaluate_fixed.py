"""
evaluate_fixed.py

FGManip OpenVLA 评测入口：支持 **域随机（DR）** 多 profile（clean / 相机扰动 / 光照），
以及可选的连续动作头（L1）与 proprio（与 OFT 训练对齐）。

Vanilla OpenVLA（离散 token、无 L1 头）请使用：
  --use_l1_regression False --use_proprio False --num_open_loop_steps 1
并由 `run_eval_batch_cases.sh` / `eval_fixed.sh`（OPENVLA_VANILLA=1）传入。
"""

import os
import sys
import json
import numpy as np
import torch
import draccus
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import gymnasium as gym
from collections import deque
from PIL import Image
import imageio
import torch.nn.functional as F

# Add root to sys.path to ensure core.env can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Add current directory to sys.path to import experiments
CURR_DIR = os.path.dirname(__file__)
if CURR_DIR not in sys.path:
    sys.path.append(CURR_DIR)

try:
    import core.env  # noqa: F401
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Warning: Could not import core.env from {ROOT_DIR}. Error: {e}")

from experiments.robot.robot_utils import (
    get_model,
    get_action,
    get_image_resize_size,
    set_seed_everywhere,
    DATE_TIME,
)
from experiments.robot.openvla_utils import (
    get_processor,
    get_action_head,
    get_noisy_action_projector,
    get_proprio_projector,
    resize_image_for_policy,
)

from utils.util import select_action_subspace, summarize_metric


TASK_DESCRIPTIONS = {
    "grasp_part": "grasp the handle",
    "align_to_part": "align the gripper to the handle",
    "stand_up": "pick up the object and make it stand up",
    "toggle_switch": "toggle the switch",
    "toggle_switch_table": "toggle the switch on the table",
    "lid_opening": "open the lid of the bottle",
    "slide_along": "slide the object along the surface",
    "rotate": "rotate the object",
    "door_env": "open the door by the handle",
    "plug_charger": "pick up the charger and plug it into the receptacle",
    "peg_in_hole": "pick up the peg and insert it into the box with a hole",
    "stack_pyramid": "stack the blue cube on top of the red and green cubes",
    "draw_triangle": "draw a triangle connecting the vertices",
    "multi_skill": "complete the manipulation task",
}

PART_NAME_TEMPLATES = {
    "grasp_part": "grasp the {part}",
    "align_to_part": "align the gripper to the {part}",
    "toggle_switch": "toggle the {part}",
    "toggle_switch_table": "toggle the {part} on the table",
    "lid_opening": "open the {part} of the bottle",
    "slide_along": "slide the {part}",
    "rotate": "rotate the {part}",
    "door_env": "open the door by the {part}",
}

TASKS_WITH_OBJECT_PART = {
    "grasp_part",
    "align_to_part",
    "stand_up",
    "toggle_switch",
    "toggle_switch_table",
    "lid_opening",
    "slide_along",
    "rotate",
    "door_env",
}


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    task_name: str = "plug_charger"
    object_name: Optional[str] = None
    part_name: Optional[str] = None
    task_description_override: Optional[str] = None
    num_episodes: int = 10
    max_steps: int = 200

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_proprio: bool = True

    use_film: bool = False
    num_images_in_input: int = 1
    lora_rank: int = 32

    seed: int = 7
    # Preferred normalization key in dataset_statistics.json; auto-resolved if None/invalid.
    unnorm_key: Optional[str] = None

    num_open_loop_steps: int = 8
    control_mode: str = "pd_joint_delta_pos"
    enable_dr_eval: bool = False
    camera_pos_levels: Optional[List[float]] = None
    camera_rot_levels_deg: Optional[List[float]] = None
    light_ambient_delta_levels: Optional[List[float]] = None

    save_video: bool = True
    video_dir: str = os.path.join(os.path.dirname(__file__), "eval_videos")
    output_dir: Optional[str] = None
    save_cam_overlay: bool = False
    cam_overlay_alpha: float = 0.45
    video_width: int = 640
    video_height: int = 480


def _compute_mad(actions: np.ndarray, norm: str = "l2", dt: float = 1.0, eps: float = 1e-12) -> float:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.nan
    da = np.diff(arr, axis=0) / max(float(dt), eps)
    if norm == "l2":
        return float(np.mean(np.linalg.norm(da, ord=2, axis=1)))
    if norm == "l1":
        return float(np.mean(np.linalg.norm(da, ord=1, axis=1)))
    raise ValueError("norm must be 'l1' or 'l2'")


def _extract_success_at_end(final_info: Any) -> bool:
    try:
        if isinstance(final_info, dict):
            ep_info = final_info.get("episode", {})
            s = ep_info.get("success_at_end", ep_info.get("success", False))
        else:
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


def _resolve_unnorm_key(cfg: GenerateConfig, model: Any) -> None:
    """Resolve cfg.unnorm_key against loaded model.norm_stats."""
    norm_stats = getattr(model, "norm_stats", None)
    if not isinstance(norm_stats, dict) or len(norm_stats) == 0:
        print("Warning: model.norm_stats is missing or empty; keep provided unnorm_key as-is.")
        return

    available = list(norm_stats.keys())
    if cfg.unnorm_key and cfg.unnorm_key in norm_stats:
        print(f"Using unnorm_key: {cfg.unnorm_key}")
        return

    candidates = []
    if cfg.task_name:
        candidates.append(f"{cfg.task_name}_rlds")
        candidates.append(cfg.task_name)
    candidates.append("fgmanip_rlds")

    selected = None
    for k in candidates:
        if k in norm_stats:
            selected = k
            break
    if selected is None:
        selected = available[0]

    if cfg.unnorm_key and cfg.unnorm_key not in norm_stats:
        print(
            f"Warning: requested unnorm_key '{cfg.unnorm_key}' not found. "
            f"Switching to '{selected}'. Available keys: {available}"
        )
    else:
        print(f"Auto-selected unnorm_key: {selected}. Available keys: {available}")
    cfg.unnorm_key = selected


def compute_ausc(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, num=arr.size)
    return float(np.trapz(arr, x))


def build_eval_profiles(base_env_kwargs: Dict[str, Any], cfg: GenerateConfig) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {"clean": dict(base_env_kwargs)}
    if not cfg.enable_dr_eval:
        return profiles

    cam_pos_levels = list(cfg.camera_pos_levels or [0.03, 0.06, 0.12])[:3]
    cam_rot_levels = list(cfg.camera_rot_levels_deg or [2.0, 6.0, 12.0])[:3]
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

    light_levels = list(cfg.light_ambient_delta_levels or [0.10, 0.25, 0.40])[:3]
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


def get_current_gripper_action(obs: Dict[str, Any], threshold: float = 0.02) -> float:
    try:
        qpos = obs["agent"]["qpos"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        qpos = np.asarray(qpos)
        if qpos.ndim == 2:
            qpos = qpos[0]
        return 1.0 if float(np.mean(qpos[-2:])) > threshold else -1.0
    except Exception:
        return 1.0


def adapt_action(action: Any, env_act_dim: int, obs: Dict[str, Any], control_mode: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 2:
        action = action[0]
    if action.ndim != 1:
        raise ValueError(f"Expected 1D action after squeezing, got shape {action.shape}")

    if control_mode == "pd_joint_pos" and action.shape[0] == env_act_dim - 1:
        gripper = get_current_gripper_action(obs)
        action = np.concatenate([action, np.array([gripper], dtype=np.float32)], axis=0)

    if action.shape[0] != env_act_dim:
        raise ValueError(
            f"Model produced action dim {action.shape[0]}, but env expects {env_act_dim}. "
            f"control_mode={control_mode}"
        )

    return action[None, :]


def get_maniskill_observation(
    obs: Dict[str, Any], cfg: GenerateConfig, resize_size
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray | None]:
    processed_obs = {}
    images_dict = obs.get("image", {})
    if not images_dict and "sensor_data" in obs:
        images_dict = obs["sensor_data"]

    if "base_camera" in images_dict:
        primary_img = images_dict["base_camera"]["rgb"]
    elif images_dict:
        keys = list(images_dict.keys())
        valid_keys = [k for k in keys if isinstance(images_dict[k], dict) and "rgb" in images_dict[k]]
        if valid_keys:
            primary_img = images_dict[valid_keys[0]]["rgb"]
        else:
            primary_img = images_dict[keys[0]]["rgb"]
    else:
        raise ValueError(f"No camera found in observation. Keys: {list(obs.keys())}")

    if isinstance(primary_img, torch.Tensor):
        primary_img = primary_img.cpu().numpy()
    if len(primary_img.shape) == 4:
        primary_img = primary_img[0]
    original_img = primary_img.copy()
    if primary_img.shape[-1] == 4:
        primary_img = primary_img[..., :3]
    if original_img.shape[-1] == 4:
        original_img = original_img[..., :3]
    if original_img.dtype != np.uint8:
        if np.max(original_img) <= 1.0:
            original_img = np.clip(original_img * 255.0, 0, 255).astype(np.uint8)
        else:
            original_img = np.clip(original_img, 0, 255).astype(np.uint8)
    primary_img_resized = resize_image_for_policy(primary_img, resize_size)
    processed_obs["full_image"] = primary_img_resized

    wrist_original_img: np.ndarray | None = None
    if cfg.num_images_in_input > 1:
        if "hand_camera" in images_dict:
            wrist_img = images_dict["hand_camera"]["rgb"]
            if isinstance(wrist_img, torch.Tensor):
                wrist_img = wrist_img.cpu().numpy()
            if len(wrist_img.shape) == 4:
                wrist_img = wrist_img[0]
            wrist_original_img = wrist_img.copy()
            if wrist_img.shape[-1] == 4:
                wrist_img = wrist_img[..., :3]
            if wrist_original_img is not None and wrist_original_img.shape[-1] == 4:
                wrist_original_img = wrist_original_img[..., :3]
            if wrist_original_img is not None and wrist_original_img.dtype != np.uint8:
                if np.max(wrist_original_img) <= 1.0:
                    wrist_original_img = np.clip(wrist_original_img * 255.0, 0, 255).astype(np.uint8)
                else:
                    wrist_original_img = np.clip(wrist_original_img, 0, 255).astype(np.uint8)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
            processed_obs["wrist_image"] = wrist_img_resized
        else:
            print("Warning: 'hand_camera' not found, duplicating 'full_image' as 'wrist_image'.")
            processed_obs["wrist_image"] = primary_img_resized

    # Also try to capture wrist raw frame for CAM video even when model uses single-image input.
    if wrist_original_img is None and "hand_camera" in images_dict:
        wrist_img = images_dict["hand_camera"]["rgb"]
        if isinstance(wrist_img, torch.Tensor):
            wrist_img = wrist_img.cpu().numpy()
        if len(wrist_img.shape) == 4:
            wrist_img = wrist_img[0]
        wrist_original_img = wrist_img.copy()
        if wrist_original_img.shape[-1] == 4:
            wrist_original_img = wrist_original_img[..., :3]
        if wrist_original_img.dtype != np.uint8:
            if np.max(wrist_original_img) <= 1.0:
                wrist_original_img = np.clip(wrist_original_img * 255.0, 0, 255).astype(np.uint8)
            else:
                wrist_original_img = np.clip(wrist_original_img, 0, 255).astype(np.uint8)

    if cfg.use_proprio:
        qpos = None
        if "agent" in obs and "qpos" in obs["agent"]:
            qpos = obs["agent"]["qpos"]
        if qpos is None:
            raise ValueError("Could not find 'qpos' in observation for proprioception.")
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()
        if len(qpos.shape) == 2:
            qpos = qpos[0]

        gripper_val = np.mean(qpos[-2:], keepdims=True)
        processed_obs["state"] = np.concatenate([qpos[:7], gripper_val]).astype(np.float32)

    return processed_obs, original_img, wrist_original_img


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    video_dir,
    object_name=None,
    part_name=None,
    stream_tag: str = "base_cam_overlay",
):
    os.makedirs(video_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    obj_tag = f"--obj={object_name}" if object_name else ""
    part_tag = f"--part={part_name}" if part_name else ""
    mp4_path = (
        f"{video_dir}/{DATE_TIME}--openvla_oft--stream={stream_tag}--episode={idx}"
        f"--success={success}{obj_tag}{part_tag}--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
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
    h = 1.0 - np.clip(heatmap_01, 0.0, 1.0)
    # h = np.clip(heatmap_01, 0.0, 1.0)
    r = (255.0 * h).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(h - 0.5) * 2.0) * 0.75).astype(np.uint8)
    b = (255.0 * (1.0 - h)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _get_openvla_vision_backbone(model: Any):
    if hasattr(model, "vision_backbone"):
        return model.vision_backbone
    if hasattr(model, "model") and hasattr(model.model, "vision_backbone"):
        return model.model.vision_backbone
    raise AttributeError("Cannot find `vision_backbone` on OpenVLA model.")


def _get_openvla_image_transform(vision: Any, processor: Any = None):
    if hasattr(vision, "image_transform"):
        return vision.image_transform
    if hasattr(vision, "get_image_transform"):
        return vision.get_image_transform()
    # HF processor fallback (PrismaticVisionBackbone does not expose image_transform)
    if processor is not None and hasattr(processor, "image_processor"):
        image_processor = processor.image_processor
        if hasattr(image_processor, "apply_transform"):
            return image_processor.apply_transform
    if hasattr(processor, "image_processor") and callable(processor.image_processor):
        # last resort: processor image_processor __call__
        return lambda img: processor.image_processor(img, return_tensors="pt")["pixel_values"][0]
    raise AttributeError("Cannot find image transform for OpenVLA vision backbone.")


@torch.no_grad()
def _overlay_openvla_cam(
    model: Any,
    image_rgb: np.ndarray,
    processor: Any = None,
    alpha: float = 0.45,
) -> np.ndarray:
    image_rgb = _to_uint8_rgb(image_rgb)
    vision = _get_openvla_vision_backbone(model)
    image_pil = Image.fromarray(image_rgb).convert("RGB")
    image_transform = _get_openvla_image_transform(vision, processor=processor)
    pixel_values = image_transform(image_pil)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_dtype = None
    try:
        vision_dtype = next(vision.parameters()).dtype
    except StopIteration:
        vision_dtype = None

    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None].to(model_device)
        if vision_dtype is not None and pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(dtype=vision_dtype)
        try:
            token_features = vision(pixel_values)
        except TypeError:
            # FiLM wrapper needs language embeddings; CAM fallback uses zero embedding.
            lang = torch.zeros((1, 1, getattr(model, "llm_dim", 4096)), device=model_device, dtype=vision_dtype)
            token_features = vision(pixel_values, lang)
    elif isinstance(pixel_values, dict):
        casted = {}
        for k, v in pixel_values.items():
            vi = v[None].to(model_device)
            if vision_dtype is not None and vi.dtype != vision_dtype:
                vi = vi.to(dtype=vision_dtype)
            casted[k] = vi
        pixel_values = casted
        try:
            token_features = vision(pixel_values)
        except TypeError:
            lang = torch.zeros((1, 1, getattr(model, "llm_dim", 4096)), device=model_device, dtype=vision_dtype)
            token_features = vision(pixel_values, lang)
    else:
        return image_rgb

    if token_features.dim() == 3:
        token_features = token_features[0]
    if token_features.dim() != 2:
        return image_rgb

    expected_num_patches = getattr(vision, "num_patches", None)
    if expected_num_patches is not None and token_features.shape[0] >= expected_num_patches:
        if token_features.shape[0] != expected_num_patches:
            token_features = token_features[-expected_num_patches:]

    n_tokens = token_features.shape[0]
    n = int(np.sqrt(n_tokens))
    if n * n != n_tokens:
        if n_tokens > 1:
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


@draccus.wrap()
def eval_policy(cfg: GenerateConfig) -> None:
    print(f"Evaluating OpenVLA (evaluate_fixed) on {cfg.task_name}...")
    set_seed_everywhere(cfg.seed)

    try:
        model = get_model(cfg)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    _resolve_unnorm_key(cfg, model)

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
        except Exception as e:
            print(f"Warning: Could not load action head: {e}")

    proprio_projector = None
    if cfg.use_proprio:
        try:
            proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
        except Exception as e:
            print(f"Warning: Could not load proprio projector: {e}")

    noisy_action_projector = None
    if cfg.use_diffusion:
        try:
            noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
        except Exception as e:
            print(f"Warning: Could not load noisy action projector: {e}")

    resize_size = get_image_resize_size(cfg)

    env_kwargs = dict(
        obs_mode="rgbd",
        control_mode=cfg.control_mode,
        robot_uids="panda_wristcam",
        num_envs=1,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="rt", width=cfg.video_width, height=cfg.video_height),
        human_render_camera_configs=dict(shader_pack="rt"),
        viewer_camera_configs=dict(shader_pack="rt"),
    )
    if cfg.task_name in TASKS_WITH_OBJECT_PART:
        if cfg.object_name is not None:
            env_kwargs["object_name"] = cfg.object_name
        if cfg.part_name is not None:
            env_kwargs["part_name"] = cfg.part_name
    task_description = TASK_DESCRIPTIONS.get(cfg.task_name, "do the task")
    if cfg.part_name is not None and cfg.task_name in PART_NAME_TEMPLATES:
        task_description = PART_NAME_TEMPLATES[cfg.task_name].format(part=cfg.part_name)
    if cfg.task_description_override is not None and str(cfg.task_description_override).strip() != "":
        task_description = str(cfg.task_description_override).strip()
    print(f"Task Description: {task_description}")
    profiles = build_eval_profiles(env_kwargs, cfg)
    profile_results: Dict[str, Dict[str, Any]] = {}
    episode_error_logs: List[Dict[str, Any]] = []

    for profile_name, profile_env_kwargs in profiles.items():
        env = gym.make(cfg.task_name, **profile_env_kwargs)
        env_act_dim = int(np.prod(env.action_space.shape))

        successes: List[bool] = []
        exp_neg_mad_all_list: List[float] = []
        exp_neg_mad_success_list: List[float] = []

        print(f"\n==== Profile: {profile_name} ====")
        for ep in range(cfg.num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            success = False
            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            rollout_images_base = []
            rollout_images_wrist = []
            episode_actions: List[np.ndarray] = []
            last_info: Dict[str, Any] = {}

            print(f"Episode {ep + 1}/{cfg.num_episodes} started.")

            while not done and step < cfg.max_steps:
                try:
                    processed_obs, original_img, wrist_img = get_maniskill_observation(obs, cfg, resize_size)
                    if cfg.save_video:
                        frame = original_img
                        if cfg.save_cam_overlay:
                            try:
                                frame = _overlay_openvla_cam(
                                    model=model,
                                    image_rgb=original_img,
                                    processor=processor,
                                    alpha=float(cfg.cam_overlay_alpha),
                                )
                            except Exception as cam_e:
                                print(f"Warning: CAM overlay failed at ep={ep}, step={step}: {cam_e}")
                                episode_error_logs.append(
                                    {
                                        "profile": profile_name,
                                        "episode": int(ep),
                                        "step": int(step),
                                        "stage": "cam_overlay",
                                        "error": str(cam_e),
                                    }
                                )
                        rollout_images_base.append(frame)
                        if wrist_img is not None:
                            wrist_frame = wrist_img
                            if cfg.save_cam_overlay:
                                try:
                                    wrist_frame = _overlay_openvla_cam(
                                        model=model,
                                        image_rgb=wrist_img,
                                        processor=processor,
                                        alpha=float(cfg.cam_overlay_alpha),
                                    )
                                except Exception:
                                    wrist_frame = wrist_img
                            rollout_images_wrist.append(wrist_frame)
                except Exception as e:
                    print(f"Error processing observation: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_error_logs.append(
                        {
                            "profile": profile_name,
                            "episode": int(ep),
                            "step": int(step),
                            "stage": "observation",
                            "error": str(e),
                        }
                    )
                    break

                if len(action_queue) == 0:
                    actions = get_action(
                        cfg,
                        model,
                        processed_obs,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                    if isinstance(actions, np.ndarray) and actions.ndim == 2:
                        for a in actions:
                            action_queue.append(a)
                    elif isinstance(actions, list):
                        for a in actions:
                            action_queue.append(np.asarray(a))
                    else:
                        action_queue.append(np.asarray(actions))

                action = action_queue.popleft()
                action = adapt_action(action, env_act_dim, obs, cfg.control_mode)
                episode_actions.append(
                    select_action_subspace(
                        action[0], control_mode=cfg.control_mode, include_gripper=False
                    ).copy()
                )

                obs, _reward, terminated, truncated, info = env.step(action)
                if isinstance(info, dict):
                    last_info = info
                if isinstance(terminated, torch.Tensor):
                    terminated = terminated.item()
                if isinstance(truncated, torch.Tensor):
                    truncated = truncated.item()
                # Keep evaluation horizon behavior consistent with ACT evaluation:
                # do not stop early on termination (e.g., success); stop on truncation
                # (time-limit) or when reaching cfg.max_steps via loop condition.
                done = bool(truncated)
                step += 1

            if isinstance(last_info, dict) and "final_info" in last_info:
                success = _extract_success_at_end(last_info["final_info"])
            elif isinstance(last_info, dict) and "success" in last_info:
                success = bool(last_info["success"])

            successes.append(success)
            if len(episode_actions) >= 2:
                ep_mad = _compute_mad(np.asarray(episode_actions, dtype=np.float64), dt=1.0, norm="l2")
            else:
                ep_mad = np.nan
            if np.isfinite(ep_mad):
                exp_neg_mad = float(np.exp(-float(ep_mad)))
                exp_neg_mad_all_list.append(exp_neg_mad)
                if success:
                    exp_neg_mad_success_list.append(exp_neg_mad)

            print(f"Episode {ep + 1} finished. Success: {success}")

            if cfg.save_video and rollout_images_base:
                profile_video_dir = cfg.video_dir if profile_name == "clean" else os.path.join(cfg.video_dir, profile_name)
                save_rollout_video(
                    rollout_images_base,
                    ep,
                    success,
                    task_description,
                    profile_video_dir,
                    object_name=cfg.object_name,
                    part_name=cfg.part_name,
                    stream_tag="base_cam_overlay",
                )
                if rollout_images_wrist:
                    save_rollout_video(
                        rollout_images_wrist,
                        ep,
                        success,
                        task_description,
                        profile_video_dir,
                        object_name=cfg.object_name,
                        part_name=cfg.part_name,
                        stream_tag="wrist_cam_overlay",
                    )

        success_rate = np.mean(successes) if successes else 0.0
        exp_neg_mad_all_stats = summarize_metric(exp_neg_mad_all_list)
        exp_neg_mad_success_stats = summarize_metric(exp_neg_mad_success_list)
        profile_results[profile_name] = {
            "episodes": int(len(successes)),
            "successes": int(np.sum(successes)),
            "success_rate": float(success_rate),
            "exp_neg_mad_all": exp_neg_mad_all_stats,
            "exp_neg_mad_success": exp_neg_mad_success_stats,
        }
        print(
            f"[{profile_name}] Success Rate: {success_rate * 100:.2f}% "
            f"({np.sum(successes)}/{cfg.num_episodes})"
        )
        print(
            f"[{profile_name}] exp(-mad)_all: {exp_neg_mad_all_stats['mean']:.6f} "
            f"(count={exp_neg_mad_all_stats['count']}) | "
            f"exp(-mad)_success: {exp_neg_mad_success_stats['mean']:.6f} "
            f"(count={exp_neg_mad_success_stats['count']})"
        )
        env.close()

    if cfg.enable_dr_eval:
        if all(name in profile_results for name in ["clean", "cam_l1", "cam_l2", "cam_l3"]):
            cam_srs = [profile_results[name]["success_rate"] for name in ["clean", "cam_l1", "cam_l2", "cam_l3"]]
            cam_ausc = compute_ausc(cam_srs)
            profile_results["ausc_camera"] = {"value": cam_ausc}
            print(f"AUSC(camera): {cam_ausc:.6f}")
        if all(name in profile_results for name in ["clean", "light_l1", "light_l2", "light_l3"]):
            light_srs = [profile_results[name]["success_rate"] for name in ["clean", "light_l1", "light_l2", "light_l3"]]
            light_ausc = compute_ausc(light_srs)
            profile_results["ausc_light"] = {"value": light_ausc}
            print(f"AUSC(light): {light_ausc:.6f}")

    if cfg.output_dir:
        summary_dir = cfg.output_dir
    else:
        summary_dir = os.path.dirname(cfg.video_dir) if os.path.basename(cfg.video_dir) == "eval_videos" else cfg.video_dir
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(profile_results, f, ensure_ascii=False, indent=2)
    print(f"Saved eval summary to: {summary_path}")

    if episode_error_logs:
        err_path = os.path.join(summary_dir, "episode_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(episode_error_logs, f, ensure_ascii=False, indent=2)
        print(f"Saved episode error log to: {err_path}")


if __name__ == "__main__":
    eval_policy()
