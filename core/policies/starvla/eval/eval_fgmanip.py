"""
Evaluation script for Qwen-GR00T VLA on FGManip tasks.
Runs in FGManip environment (ManiSkill3) and communicates with starVLA model server via WebSocket.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from tqdm import tqdm


import core.env  
import core.skill

from model2fgmanip_interface import FGManipModelClient, get_model, reset_model

# Instruction mapping for FGManip tasks
TASK_INSTRUCTIONS = {
    "grasp_part": {
        "cap": "grasp the cap firmly",
        "lid": "grasp the lid carefully",
        "handle": "grasp the handle",
        "knob": "grasp the knob",
        "button": "press the button",
        "base": "grasp the base",
    },
    "stand_up": {
        "bottle": "stand up the bottle upright",
        "cup": "stand up the cup",
        "default": "stand up the object vertically",
    },
    "press_switch": {
        "button": "press the button down until it clicks",
        "default": "press the switch down",
    },
    "toggle_switch": {
        "switch": "toggle the switch to the opposite state",
        "default": "toggle the switch",
    },
    "rotate": {
        "knob": "rotate the knob clockwise by 90 degrees",
        "default": "rotate the object",
    },
    "draw_triangle": {
        "default": "draw a precise triangle pattern",
    },
    "plug_charger": {
        "defaule": "Pick up the charger plug, precisely align it with the socket, and insert it into the port until it is fully seated."
    }
}

def get_instruction(skill_type: str, part_name: str = "default") -> str:
    """Get language instruction for a given skill/part combination."""
    if skill_type in TASK_INSTRUCTIONS:
        if part_name in TASK_INSTRUCTIONS[skill_type]:
            return TASK_INSTRUCTIONS[skill_type][part_name]
        elif "default" in TASK_INSTRUCTIONS[skill_type]:
            return TASK_INSTRUCTIONS[skill_type]["default"]
    
    # Fallback generic instructions
    fallbacks = {
        "grasp_part": f"grasp the {part_name} carefully",
        "stand_up": f"stand up the {part_name} object",
        "press_switch": "press the switch",
        "toggle_switch": "toggle the switch",
        "rotate": "rotate the object smoothly",
        "draw_triangle": "draw a triangle",
        "plug_charger": "Pick up the charger plug, precisely align it with the socket, and insert it into the port until it is fully seated."
    }
    return fallbacks.get(skill_type, f"perform the {skill_type} task on {part_name}")

def create_fgmanip_env(
    skill_type: str,
    part_name: str,
    object_name: str,
    num_envs: int = 1,
    obs_mode: str = "rgb",
    control_mode: str = "pd_joint_pos",
    render_mode: str = "rgb_array",
    sim_backend: str = "auto",
    record_dir: Optional[str] = None,
) -> gym.Env:
    """
    Create FGManip environment with proper configuration.
    
    Note: FGManip registers environments with names like "{skill_type}_env"
    """
    # Determine environment ID based on skill type
    # FGManip convention: skill_type_env (e.g., "grasp_part_env")

    env_id = f"{skill_type}"
    
    print(f"🔧 Creating environment: {env_id}")
    print(f"   Part: {part_name} | Object: {object_name}")
    print(f"   Obs mode: {obs_mode} | Control mode: {control_mode}")
    
    # Create base environment
    env_kwargs = {
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "render_mode": render_mode,
        "sim_backend": sim_backend,
        "sensor_configs": dict(shader_pack="rt", width=224, height=224),
        # "human_render_camera_configs": dict(shader_pack="rt"),
        # "viewer_camera_configs": dict(shader_pack="rt"),
    }
    
    # Add FGManip-specific kwargs if applicable
    if part_name != "default":
        env_kwargs["part_name"] = part_name
    if object_name != "default":
        env_kwargs["object_name"] = object_name
    
    try:
        env = gym.make(env_id, num_envs=num_envs, **env_kwargs)
    except TypeError as e:
        # 如果环境不支持 part_name 或 object_name 参数，移除它们后重试
        if "part_name" in str(e) or "object_name" in str(e):
            print(f"⚠️ Environment '{env_id}' does not support part_name/object_name parameters, retrying without them...")
            env_kwargs.pop("part_name", None)
            env_kwargs.pop("object_name", None)
            env = gym.make(env_id, num_envs=num_envs, **env_kwargs)
        else:
            raise
    except gym.error.NameNotFound:
        # Fallback: try without "_env" suffix (some FGManip versions)
        env_id_fallback = skill_type
        print(f"⚠️ Environment '{env_id}' not found, trying '{env_id_fallback}'")
        try:
            env = gym.make(env_id_fallback, num_envs=num_envs, **env_kwargs)
        except TypeError as e:
            # 同样处理 fallback 环境不支持参数的情况
            if "part_name" in str(e) or "object_name" in str(e):
                print(f"⚠️ Environment '{env_id_fallback}' does not support part_name/object_name parameters, retrying without them...")
                env_kwargs.pop("part_name", None)
                env_kwargs.pop("object_name", None)
                env = gym.make(env_id_fallback, num_envs=num_envs, **env_kwargs)
            else:
                raise
    
    # Wrap with video recording if requested
    if record_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=record_dir,
            trajectory_name=f"{skill_type}_{part_name}_{object_name}",
            save_video=True,
            save_trajectory=False,
            video_fps=30,
        )
    
    return env

def evaluate_fgmanip(
    skill_type: str,
    part_name: str,
    object_name: str,
    num_episodes: int,
    host: str,
    port: int,
    policy_ckpt_path: str,
    unnorm_key: Optional[str],
    camera_keys: List[str],
    max_steps: int = 200,
    seed: int = 42,
    record_dir: Optional[str] = None,
    save_camera_images: bool = False,  # 新增
    camera_save_dir: Optional[str] = None,  # 新增
    **env_kwargs,
) -> Dict[str, float]:
    """
    Evaluate Qwen-GR00T VLA model on FGManip task.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get task instruction
    instruction = get_instruction(skill_type, part_name)
    print(f"\n📝 Task instruction: '{instruction}'")
    
    # Create environment
    env = create_fgmanip_env(
        skill_type=skill_type,
        part_name=part_name,
        object_name=object_name,
        record_dir=record_dir,
        **env_kwargs,
    )
    
    if save_camera_images and camera_save_dir is None:
        camera_save_dir = Path(record_dir) / "camera_images" if record_dir else "eval_camera_images"

    # Initialize model client
    print(f"\n🔌 Connecting to VLA server at {host}:{port}")
    model = FGManipModelClient(
        policy_ckpt_path=policy_ckpt_path,
        unnorm_key=unnorm_key,
        host=host,
        port=port,
        camera_keys=camera_keys,
        save_images=save_camera_images,  # 新增
        save_dir=camera_save_dir,  # 新增
    )
    
    # Reset model with instruction
    model.reset(instruction)
    
    # Evaluation loop
    successes = []
    episode_lengths = []
    
    print(f"\n▶️ Starting evaluation ({num_episodes} episodes)...")
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        # Reset environment and model
        obs, info = env.reset(seed=seed + ep)
        model.reset(instruction)
        
        success = False
        for step in range(max_steps):
            # Get action from model and execute
            try:
                action = model.get_action(obs, instruction)
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"⚠️ Step {step} error: {str(e)}")
                break
            
            # Check termination conditions
            if terminated or truncated:
                success = info.get("success", False)
                episode_lengths.append(step + 1)
                break
        
        successes.append(success)
        status = "✓ Success" if success else "✗ Failure"
        tqdm.write(f"  Episode {ep+1}/{num_episodes} | {status} | Steps: {step+1}")
    
    # Cleanup
    env.close()
    model.close()
    
    # Compute metrics
    success_rate = np.mean(successes)
    avg_steps = np.mean(episode_lengths) if episode_lengths else 0
    
    metrics = {
        "skill_type": skill_type,
        "part_name": part_name,
        "object_name": object_name,
        "success_rate": float(success_rate),
        "avg_episode_length": float(avg_steps),
        "num_episodes": num_episodes,
        "num_successes": int(np.sum(successes)),
        "camera_keys": camera_keys,
        "instruction": instruction,
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen-GR00T VLA on FGManip tasks")
    
    # Task configuration
    parser.add_argument("--skill-type", type=str, required=True,
                        choices=["grasp_part", "stand_up", "plug_charger", "press_switch", "toggle_switch", "rotate", "draw_triangle"],
                        help="FGManip skill type to evaluate")
    parser.add_argument("--part-name", type=str, default="default",
                        help="Part name for the skill (e.g., cap, lid, button)")
    parser.add_argument("--object-name", type=str, default="default",
                        help="Object name (e.g., bottle, 102812)")
    
    # Evaluation parameters
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model server configuration
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Model server hostname")
    parser.add_argument("--port", type=int, default=10093,
                        help="Model server port")
    parser.add_argument("--policy-ckpt-path", type=str, required=True,
                        help="Path to Qwen-GR00T checkpoint (for loading stats)")
    parser.add_argument("--unnorm-key", type=str, default=None,
                        help="Dataset key for unnormalization stats (auto-detected if omitted)")
    
    # Observation configuration
    parser.add_argument("--camera-keys", type=str, nargs='+', 
                        default=["base_camera", "hand_camera"],
                        help="Camera keys to use for observation (order must match training)")
    parser.add_argument("--obs-mode", type=str, default="rgb",
                        choices=["rgb", "rgbd"],
                        help="Observation mode (must match training)")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos",
                        choices=["pd_joint_pos", "pd_joint_delta_pos", "pd_ee_delta_pos", "pd_ee_delta_pose"],
                        help="Control mode (must match training)")
    
    # Recording
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Directory to save evaluation videos (optional)")
    parser.add_argument("--shader", type=str, default="rt",
                        choices=["default", "rt", "rt-fast"],
                        help="Shader for rendering")
    parser.add_argument("--sim-backend", type=str, default="auto",
                        choices=["auto", "cpu", "gpu"],
                        help="Simulation backend")

                        # Recording
    parser.add_argument("--save-camera-images", action="store_true",  # 新增
                        help="Save base_camera and hand_camera images during evaluation")
    parser.add_argument("--camera-save-dir", type=str, default=None,  # 新增
                        help="Directory to save camera images (default: record_dir/camera_images)")

    
    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_fgmanip(
        skill_type=args.skill_type,
        part_name=args.part_name,
        object_name=args.object_name,
        num_episodes=args.num_episodes,
        host=args.host,
        port=args.port,
        policy_ckpt_path=args.policy_ckpt_path,
        unnorm_key=args.unnorm_key,
        camera_keys=args.camera_keys,
        max_steps=args.max_steps,
        seed=args.seed,
        record_dir=args.record_dir,
        save_camera_images=args.save_camera_images,  # 新增
        camera_save_dir=args.camera_save_dir,  # 新增
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
    )
    
    # Print results
    print("\n" + "="*60)
    print("📊 FGManip EVALUATION RESULTS")
    print("="*60)
    print(f"Task: {metrics['skill_type']} | Part: {metrics['part_name']} | Object: {metrics['object_name']}")
    print(f"Instruction: '{metrics['instruction']}'")
    print(f"Cameras: {metrics['camera_keys']}")
    print(f"Success Rate: {metrics['success_rate']:.2%} ({metrics['num_successes']}/{metrics['num_episodes']})")
    print(f"Average Episode Length: {metrics['avg_episode_length']:.1f} steps")
    print("="*60)
    
    # Save metrics to JSON
    if args.record_dir:
        metrics_path = Path(args.record_dir) / "metrics.json"
    else:
        metrics_path = Path(f"eval_results/fgmanip_{args.skill_type}_{args.part_name}_{args.object_name}.json")
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()