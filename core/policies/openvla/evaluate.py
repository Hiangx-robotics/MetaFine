"""
evaluate.py

Evaluates an OpenVLA policy in ManiSkill environments.
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
import time

# Add root to sys.path to ensure core.env can be imported
# Assumes this file is at core/policies/openvla/evaluate.py
# Root is ../../../
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Add current directory to sys.path to import experiments
CURR_DIR = os.path.dirname(__file__)
if CURR_DIR not in sys.path:
    sys.path.append(CURR_DIR)

try:
    import core.env  # Register PlugChargerEnv and others
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Warning: Could not import core.env from {ROOT_DIR}. Error: {e}")

from experiments.robot.robot_utils import (
    get_model,
    get_action,
    get_image_resize_size,
    set_seed_everywhere,
    DATE,
    DATE_TIME,
)
from experiments.robot.openvla_utils import (
    get_processor,
    get_action_head,
    get_noisy_action_projector,
    get_proprio_projector,
    resize_image_for_policy,
)

from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_axis_angle
from utils.util import compute_mad, select_action_subspace, summarize_metric

# Task Descriptions for OpenVLA (used as the language prompt fed to the model)
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

# Tasks whose description should be templated with the actual part_name.
# {part} is replaced at runtime when cfg.part_name is set.
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

@dataclass
class GenerateConfig:
    # Model configuration
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True 
    
    # Environment configuration
    task_name: str = "grasp_part"
    object_name: Optional[str] = None
    part_name: Optional[str] = None
    num_episodes: int = 10
    max_steps: int = 200
    
    # Action Head / Projectors (must match training config)
    use_l1_regression: bool = False
    use_diffusion: bool = False
    use_proprio: bool = False
    
    # Other OpenVLA params
    use_film: bool = False
    num_images_in_input: int = 1 
    lora_rank: int = 32
    
    seed: int = 7
    unnorm_key: Optional[str] = "grasp_part_rlds"

    # Eval params
    num_open_loop_steps: int = 1
    control_mode: str = "pd_joint_delta_pos"
    
    # Logging
    save_video: bool = True
    video_dir: str = os.path.join(os.path.dirname(__file__), "eval_videos")

    # Task-graph mode (new). When set, the env, object, part and success
    # predicate are taken from the YAML and the env_id field is ignored.
    task_graph: Optional[str] = None
    results_json: Optional[str] = None
    # If set, writes metrics_summary.json here (for batch eval scripts); else next to eval_videos parent.
    output_dir: Optional[str] = None


def get_current_gripper_action(obs: Dict[str, Any], threshold: float = 0.02) -> float:
    """Infer a binary gripper command from current observation."""
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
    """Adapt model output to the action shape expected by ManiSkill."""
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


def get_maniskill_observation(obs: Dict[str, Any], cfg: GenerateConfig, resize_size) -> tuple[Dict[str, Any], np.ndarray]:
    """
    Extracts and processes observations from ManiSkill observation dict to OpenVLA format.
    Returns: (processed_obs, original_primary_image)
    """
    processed_obs = {}

    # 1. Extract Images
    # obs['image'] usually contains 'base_camera', 'hand_camera' etc.
    # We expect 'base_camera' as the main image.
    
    images_dict = obs.get('image', {})
    
    # Fallback to 'sensor_data' if 'image' is missing
    if not images_dict and 'sensor_data' in obs:
        images_dict = obs['sensor_data']
    
    # Find primary image
    # PlugChargerEnv has "base_camera"
    if 'base_camera' in images_dict:
        primary_img = images_dict['base_camera']['rgb']
    elif images_dict:
        # Fallback to first available camera
        keys = list(images_dict.keys())
        # Filter for keys that have 'rgb'
        valid_keys = [k for k in keys if isinstance(images_dict[k], dict) and 'rgb' in images_dict[k]]
        if valid_keys:
             primary_img = images_dict[valid_keys[0]]['rgb']
        else:
             # Try first key blindly if structure is different
             primary_img = images_dict[keys[0]]['rgb']
    else:
         raise ValueError(f"No camera found in observation. Keys: {list(obs.keys())}. Image dict keys: {list(images_dict.keys())}")
        
    # Resize primary image
    # ManiSkill images are typically (H, W, 3) or (H, W, 4) uint8
    if isinstance(primary_img, torch.Tensor):
        primary_img = primary_img.cpu().numpy()

    # Remove batch dim if present (1, H, W, C)
    if len(primary_img.shape) == 4:
        primary_img = primary_img[0]
        
    original_img = primary_img.copy() # Save copy for video

    if primary_img.shape[-1] == 4:
        primary_img = primary_img[..., :3] # Drop alpha
        
    primary_img_resized = resize_image_for_policy(primary_img, resize_size)
    
    processed_obs["full_image"] = primary_img_resized
    
    # 2. Extract Wrist Image (if needed)
    if cfg.num_images_in_input > 1:
        if 'hand_camera' in images_dict:
            wrist_img = images_dict['hand_camera']['rgb']
            if isinstance(wrist_img, torch.Tensor):
                wrist_img = wrist_img.cpu().numpy()
            if len(wrist_img.shape) == 4:
                wrist_img = wrist_img[0]
            if wrist_img.shape[-1] == 4:
                wrist_img = wrist_img[..., :3]
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
            processed_obs["wrist_image"] = wrist_img_resized
        else:
             # Duplicate base image to satisfy model inputs if missing (suboptimal)
             print("Warning: 'hand_camera' not found, duplicating 'full_image' as 'wrist_image'.")
             processed_obs["wrist_image"] = primary_img_resized

    # 3. Extract Proprioception (if needed)
    if cfg.use_proprio:
        qpos = None
        if 'agent' in obs and 'qpos' in obs['agent']:
            qpos = obs['agent']['qpos']
        if qpos is None:
            raise ValueError("Could not find 'qpos' in observation for proprioception.")
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()
        if len(qpos.shape) == 2:
            qpos = qpos[0]

        gripper_val = np.mean(qpos[-2:], keepdims=True)
        processed_obs["state"] = np.concatenate([qpos[:7], gripper_val]).astype(np.float32)

    return processed_obs, original_img

def save_rollout_video(rollout_images, idx, success, task_description, video_dir,
                       object_name=None, part_name=None):
    """Saves an MP4 replay of an episode."""
    os.makedirs(video_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    obj_tag = f"--obj={object_name}" if object_name else ""
    part_tag = f"--part={part_name}" if part_name else ""
    mp4_path = f"{video_dir}/{DATE_TIME}--openvla--episode={idx}--success={success}{obj_tag}{part_tag}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        # Ensure image is uint8
        if img.dtype != np.uint8:
            # If float [0, 1], convert to [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


# ============================= Task-graph eval ===============================

def _run_task_graph_eval(cfg: GenerateConfig, model, processor, action_head,
                         proprio_projector, noisy_action_projector, resize_size) -> None:
    """Rollout policy against a YAML task graph; emit results.json.

    Mirrors the inner loop of eval_policy below (get_maniskill_observation →
    get_action → action_queue → adapt_action → env.step) but builds the env
    via utils.eval_setup.make_eval_env so success comes from a compiled
    goal_predicate and per-stage progress from stage_predicates.
    """
    from utils.eval_setup import make_eval_env
    from utils.eval_metrics import EpisodeResult, EvalSummary, compute_smoothness

    base_env_kwargs = dict(
        obs_mode="rgbd",
        control_mode=cfg.control_mode,
        robot_uids="panda_wristcam",
        num_envs=1,
    )
    env, task_graph = make_eval_env(cfg, extra_env_kwargs=base_env_kwargs)
    stage_names = [s["name"] for s in (task_graph.stages or [])] if task_graph else []
    env_act_dim = int(np.prod(env.action_space.shape))

    task_description = (
        TASK_DESCRIPTIONS.get(task_graph.env if task_graph else "", None)
        or TASK_DESCRIPTIONS.get(cfg.task_name, "do the task")
    )
    if task_graph and task_graph.part and task_graph.env in PART_NAME_TEMPLATES:
        task_description = PART_NAME_TEMPLATES[task_graph.env].format(part=task_graph.part)
    print(f"[task-graph] task: {cfg.task_graph} | description: {task_description}")

    episodes_out = []
    for ep in range(cfg.num_episodes):
        obs, _ = env.reset()
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        rollout_images = []
        episode_actions = []
        stage_reached = {name: False for name in stage_names}
        ep_success = False
        step = 0
        done = False

        while not done and step < cfg.max_steps:
            try:
                processed_obs, original_img = get_maniskill_observation(obs, cfg, resize_size)
                if cfg.save_video:
                    rollout_images.append(original_img)
            except Exception as obs_e:
                print(f"[task-graph] obs processing failed: {obs_e}")
                break

            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, processed_obs, task_description,
                    processor=processor, action_head=action_head,
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
            episode_actions.append(np.asarray(action, dtype=np.float64))

            obs, _reward, terminated, truncated, info = env.step(action)
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()
            done = terminated or truncated

            # Sample stage flags + success directly from the env so partial
            # progress (engaged then drifted off) is still captured as a
            # 'best reached' running maximum.
            try:
                step_eval = env.unwrapped.evaluate()
            except Exception:
                step_eval = {}
            for name in stage_names:
                key = f"stage_{name}"
                v = step_eval.get(key)
                if v is not None and bool(v.item() if hasattr(v, "item") else v):
                    stage_reached[name] = True
            v = step_eval.get("success")
            if v is not None and bool(v.item() if hasattr(v, "item") else v):
                ep_success = True

            step += 1

        if cfg.save_video and rollout_images:
            try:
                save_rollout_video(
                    rollout_images, ep, ep_success, task_description,
                    cfg.video_dir, object_name=task_graph.object if task_graph else None,
                    part_name=task_graph.part if task_graph else None,
                )
            except Exception:
                pass

        episodes_out.append(EpisodeResult(
            seed=cfg.seed + ep,
            success=bool(ep_success),
            episode_length=int(step),
            stage_flags=dict(stage_reached),
            smoothness=compute_smoothness(np.asarray(episode_actions, dtype=np.float64))
                       if episode_actions else {},
            info={},
        ))
        print(f"[task-graph] ep {ep+1}/{cfg.num_episodes}: success={ep_success} stages={stage_reached}")

    try:
        env.close()
    except Exception:
        pass

    summary = EvalSummary.from_episodes(
        episodes_out,
        policy="openvla",
        checkpoint=str(cfg.pretrained_checkpoint),
        task_graph=cfg.task_graph,
        env_id=task_graph.env if task_graph else None,
        object_name=task_graph.object if task_graph else None,
        part_name=task_graph.part if task_graph else None,
        stage_names=stage_names if stage_names else None,
    )
    out_path = cfg.results_json or os.path.join(cfg.video_dir, "results.json")
    summary.write(out_path)
    print(f"\n{'='*48}")
    print(f"openvla task-graph eval: {cfg.task_graph}")
    print(f"  episodes:     {summary.n_episodes}")
    print(f"  success_rate: {summary.success_rate:.3f}")
    for name, rate in summary.stage_rates.items():
        print(f"  stage[{name:24s}]: {rate:.3f}")
    print(f"  smoothness:   jerk_rms={summary.smoothness_mean.get('jerk_rms', 0):.4f}")
    print(f"  written to:   {out_path}")
    print(f"{'='*48}")


@draccus.wrap()
def eval_policy(cfg: GenerateConfig) -> None:
    print(f"Evaluating OpenVLA on {cfg.task_name}...")
    
    # Task-graph mode bypasses the single-env path entirely.
    if cfg.task_graph is not None:
        # make_eval_env reads .env_id; alias .task_name here.
        cfg.env_id = cfg.task_name
        set_seed_everywhere(cfg.seed)
        try:
            model = get_model(cfg)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        processor = get_processor(cfg) if cfg.model_family == "openvla" else None
        action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            try:
                action_head = get_action_head(cfg, model.llm_dim)
            except Exception as e:
                print(f"Warning: action head load failed: {e}")
        proprio_projector = None
        if cfg.use_proprio:
            try:
                proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
            except Exception as e:
                print(f"Warning: proprio projector load failed: {e}")
        noisy_action_projector = None
        if cfg.use_diffusion:
            try:
                noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
            except Exception as e:
                print(f"Warning: noisy action projector load failed: {e}")
        resize_size = get_image_resize_size(cfg)
        _run_task_graph_eval(cfg, model, processor, action_head,
                             proprio_projector, noisy_action_projector, resize_size)
        return

    # set seed
    set_seed_everywhere(cfg.seed)
    
    # 1. Load Model
    # Wrap in try-except to handle potential model loading errors gracefully
    try:
        model = get_model(cfg)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load components
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    
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
    
    # 2. Setup Environment
    # Note: obs_mode='rgbd' to get images.
    # Only 1 env instance for evaluation
    env_kwargs = dict(
        obs_mode='rgbd',
        control_mode=cfg.control_mode,
        robot_uids='panda_wristcam',
        num_envs=1,
    )
    if cfg.object_name is not None:
        env_kwargs["object_name"] = cfg.object_name
    if cfg.part_name is not None:
        env_kwargs["part_name"] = cfg.part_name
    env = gym.make(cfg.task_name, **env_kwargs)
    env_act_dim = int(np.prod(env.action_space.shape))
    
    # 3. Evaluation Loop
    successes = []
    mad_all_list = []
    mad_success_list = []
    
    # Get task description — substitute actual part_name when available
    task_description = os.environ.get("TASK_DESCRIPTION_OVERRIDE") or TASK_DESCRIPTIONS.get(cfg.task_name, "do the task")
    if cfg.part_name is not None and cfg.task_name in PART_NAME_TEMPLATES:
        task_description = PART_NAME_TEMPLATES[cfg.task_name].format(part=cfg.part_name)
    print(f"Task Description: {task_description}")
    
    for ep in range(cfg.num_episodes):
        obs, _ = env.reset()
        
        # Debug: Print observation structure
        # if ep == 0:
        #     print(f"DEBUG: Observation keys: {obs.keys()}")
        #     if 'image' in obs:
        #         print(f"DEBUG: Image keys: {obs['image'].keys()}")
        #     else:
        #         print("DEBUG: 'image' key NOT found in observation.")
        
        done = False
        step = 0
        success = False
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        episode_actions = []

        rollout_images = []

        print(f"Episode {ep+1}/{cfg.num_episodes} started.")

        while not done and step < cfg.max_steps:
            try:
                processed_obs, original_img = get_maniskill_observation(obs, cfg, resize_size)
                if cfg.save_video:
                    rollout_images.append(original_img)
            except Exception as e:
                print(f"Error processing observation: {e}")
                import traceback
                traceback.print_exc()
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
            episode_actions.append(action[0].copy())

            obs, reward, terminated, truncated, info = env.step(action)

            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()
            done = terminated or truncated

            if done:
                s = info.get("success", False)
                if isinstance(s, torch.Tensor):
                    success = s.item()
                elif isinstance(s, np.ndarray):
                    success = s.item()
                else:
                    success = bool(s)

            step += 1
            
        successes.append(success)

        if len(episode_actions) >= 2:
            ep_mad = compute_mad(np.stack(episode_actions), dt=1.0, norm="l2")
        else:
            ep_mad = np.nan
        mad_all_list.append(ep_mad)
        if success:
            mad_success_list.append(ep_mad)

        print(f"Episode {ep+1} finished. Success: {success}")
        
        if cfg.save_video and rollout_images:
             save_rollout_video(rollout_images, ep, success, task_description, cfg.video_dir,
                                object_name=cfg.object_name, part_name=cfg.part_name)
                
    success_rate = np.mean(successes) if successes else 0.0
    mad_all_stats = summarize_metric(mad_all_list)
    mad_success_stats = summarize_metric(mad_success_list)
    print(f"Evaluation Complete. Success Rate: {success_rate*100:.2f}% ({np.sum(successes)}/{cfg.num_episodes})")
    print(
        f"MAD(all): {mad_all_stats['mean']:.6f} (count={mad_all_stats['count']}) | "
        f"MAD(success): {mad_success_stats['mean']:.6f} (count={mad_success_stats['count']})"
    )
    env.close()

    exp_neg_mad_all_list = [
        float(np.exp(-float(x))) for x in mad_all_list if np.isfinite(x)
    ]
    exp_neg_mad_success_list = [
        float(np.exp(-float(x))) for x in mad_success_list if np.isfinite(x)
    ]
    exp_neg_mad_all_stats = summarize_metric(exp_neg_mad_all_list)
    exp_neg_mad_success_stats = summarize_metric(exp_neg_mad_success_list)
    profile_results = {
        "clean": {
            "episodes": int(len(successes)),
            "successes": int(np.sum(successes)),
            "success_rate": float(success_rate),
            "exp_neg_mad_all": exp_neg_mad_all_stats,
            "exp_neg_mad_success": exp_neg_mad_success_stats,
        }
    }
    if cfg.output_dir:
        summary_dir = cfg.output_dir
    else:
        vd = cfg.video_dir.rstrip("/")
        summary_dir = (
            os.path.dirname(vd) if os.path.basename(vd) == "eval_videos" else vd
        )
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(profile_results, f, ensure_ascii=False, indent=2)
    print(f"Saved eval summary to: {summary_path}")

if __name__ == "__main__":
    eval_policy()
