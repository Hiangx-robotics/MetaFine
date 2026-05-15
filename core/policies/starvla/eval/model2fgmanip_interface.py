# starVLA/examples/FGManip/model2fgmanip_interface.py
"""
WebSocket client for FGManip evaluation - adapted from LIBERO/RoboTwin interfaces
with FGManip-specific observation handling and action processing.
"""
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List
from collections import deque

sys.path.insert(0, '../starVLA')

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config


class FGManipModelClient:
    """WebSocket client specialized for FGManip environments."""
    
    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 10093,
        image_size: List[int] = [224, 224],
        camera_keys: List[str] = ["base_camera", "hand_camera"],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        save_images: bool = False,
        save_dir: Optional[str] = None,
    ):
        # Connect to model server
        self.client = WebsocketClientPolicy(host, port)
        self.policy_ckpt_path = policy_ckpt_path
        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.camera_keys = camera_keys
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        
        # Load action normalization stats from training dataset
        self.action_norm_stats = self._load_action_stats(policy_ckpt_path, unnorm_key)
        self.action_chunk_size = self._get_action_chunk_size(policy_ckpt_path)
        
        # State tracking
        self.task_instruction = None
        self.raw_actions_buffer = None  # Buffer for action chunking
        self.step_in_chunk = 0
        
        print(f"✅ FGManipModelClient initialized | cameras: {camera_keys} | chunk_size: {self.action_chunk_size}")

        # 新增：图像保存相关
        self.save_images = save_images
        self.save_dir = save_dir
        self.step_counter = 0
        self.episode_counter = 0
        if self.save_images and self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            print(f"📸 Image saving enabled | save_dir: {self.save_dir}")

    def _load_action_stats(self, ckpt_path: str, unnorm_key: Optional[str]) -> Dict:
        """Load action normalization statistics from LeRobot dataset."""
        ckpt_path = Path(ckpt_path)
        _, norm_stats = read_mode_config(ckpt_path)
        
        # Auto-select unnorm_key if not provided
        if unnorm_key is None:
            unnorm_key = next(iter(norm_stats.keys())) if len(norm_stats) == 1 else "default"
        
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"unnorm_key '{unnorm_key}' not found in stats. Available keys: {list(norm_stats.keys())}"
            )
        
        return norm_stats[unnorm_key]["action"]
    
    def _get_action_chunk_size(self, ckpt_path: str) -> int:
        """Get action chunk size from model config."""
        ckpt_path = Path(ckpt_path)
        model_config, _ = read_mode_config(ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1
    
    def reset(self, instruction: str) -> None:
        """Reset model state with new task instruction."""
        self.task_instruction = instruction
        self.raw_actions_buffer = None
        self.step_in_chunk = 0
        print(f"🔄 Reset model with instruction: '{instruction}'")

           # 新增：重置步数计数器，增加episode计数
        if self.save_images:
            self.step_counter = 0
            self.episode_counter += 1
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to model input size with correct interpolation."""
        # Handle CHW -> HWC conversion if needed
        # if image.shape[0] == 3 and image.ndim == 3:  # CHW format
        #     image = np.transpose(image, (1, 2, 0))
        
        # Convert float [0,1] to uint8 [0,255] if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize maintaining aspect ratio (center crop if needed)
        return cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
    
    def _extract_images_from_obs(self, obs: Dict) -> List[np.ndarray]:
        """
        Extract RGB images from FGManip's hierarchical observation structure:
        obs -> sensor_data -> {camera_name} -> rgb
        
        FGManip structure example:
        obs['sensor_data']['hand_camera']['rgb']  # shape [B, H, W, 3]
        obs['sensor_data']['base_camera']['rgb']  # shape [B, H, W, 3]
        """
        # First-time debug output
        if not hasattr(self, '_obs_debug_printed'):
            print("\n🔍 FGManip Observation Structure:")
            print(f"   Top-level keys: {list(obs.keys())}")
            if 'sensor_data' in obs:
                print(f"   sensor_data keys: {list(obs['sensor_data'].keys())}")
                for cam in obs['sensor_data'].keys():
                    if isinstance(obs['sensor_data'][cam], dict):
                        print(f"     → {cam}: {list(obs['sensor_data'][cam].keys())}")
            self._obs_debug_printed = True
        
        # Validate required keys exist
        if 'sensor_data' not in obs:
            raise ValueError(
                f"❌ 'sensor_data' missing in observation. Keys: {list(obs.keys())}\n"
                "   Ensure env created with obs_mode='rgb' or 'rgbd'"
            )
        
        sensor_data = obs['sensor_data']
        images = []
        
        for cam_key in self.camera_keys:
            # FGManip structure: sensor_data -> {camera} -> rgb
            if cam_key not in sensor_data:
                available = list(sensor_data.keys())
                raise ValueError(
                    f"❌ Camera '{cam_key}' not found in sensor_data. Available: {available}\n"
                    f"   Expected structure: obs['sensor_data']['{cam_key}']['rgb']"
                )
            
            if 'rgb' not in sensor_data[cam_key]:
                available_subkeys = list(sensor_data[cam_key].keys())
                raise ValueError(
                    f"❌ 'rgb' key missing under camera '{cam_key}'. Available: {available_subkeys}\n"
                    f"   Expected: obs['sensor_data']['{cam_key}']['rgb']"
                )
            
            # Extract RGB tensor: shape [B, H, W, 3]
            img_tensor = sensor_data[cam_key]['rgb']
            
            # Handle batch dimension (B=1 for single env)
            if img_tensor.ndim == 4:
                img = img_tensor[0]  # [H, W, 3]
            else:
                img = img_tensor  # Already [H, W, 3]
            
            # Convert to numpy if torch tensor
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            
            # Ensure HWC layout (should already be HWC)
            if img.shape[0] == 3 and img.ndim == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            
            # Convert float [0,1] to uint8 [0,255]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            img = self._resize_image(img) 
            # img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
            images.append(img)
        
        if not images:
            raise ValueError(f"❌ No images extracted for cameras: {self.camera_keys}")
        
        return images
    
    def unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """
        Convert normalized actions [-1, 1] to raw action space using dataset statistics.
        FGManip action space: [7 joint deltas + 1 gripper] = 8D
        """
        action_high = np.array(self.action_norm_stats["max"])
        action_low = np.array(self.action_norm_stats["min"])
        mask = self.action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        
        # Clip to valid range first
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        
        # Unnormalize masked dimensions (typically all except gripper might be masked)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
            normalized_actions  # Pass through unmasked dimensions (e.g., gripper might be binary)
        )
        
        # Special handling for gripper (index 7): binarize if needed
        if actions.shape[-1] > 7:  # Has gripper dimension
            # Training often uses continuous gripper [-1, 1] where >0 = open
            actions[..., 7] = np.where(actions[..., 7] > 0.0, 1.0, -1.0)
        
        return actions
    
    def get_action(self, obs: Dict, instruction: Optional[str] = None) -> np.ndarray:
        """
        Get next action from VLA model.
        
        Args:
            obs: ManiSkill observation dictionary containing 'image' key
            instruction: Optional new instruction (triggers model reset if changed)
        
        Returns:
            Raw action array [8] for Franka (7 joint deltas + 1 gripper)
        """
        # Handle instruction change
        if instruction is not None and instruction != self.task_instruction:
            self.reset(instruction)
        
        # Extract and preprocess images
        images = self._extract_images_from_obs(obs)

        # Extract and preprocess images

        # 新增：保存原始图像（在resize之前保存更高质量的图像）
        if self.save_images and self.save_dir:
            self._save_camera_images(obs, self.step_counter, self.episode_counter)
            self.step_counter += 1
        
        qpos = obs["agent"]["qpos"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu()
        state = np.asarray(qpos, dtype=np.float32)
        # Prepare VLA input
        example = {
            "image": images,  # List of [H, W, 3] uint8 arrays
            "lang": self.task_instruction,
            # "state": state
        }
        
        # Get new action chunk when buffer is empty or at chunk boundary
        if self.raw_actions_buffer is None or self.step_in_chunk >= self.action_chunk_size:
            vla_input = {
                "examples": [example],
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            
            # Get prediction from server
            response = self.client.predict_action(vla_input)
            normalized_actions = response["data"]["normalized_actions"][0]  # [chunk_size, action_dim]
            
            # Unnormalize entire chunk
            self.raw_actions_buffer = self.unnormalize_actions(normalized_actions)
            self.step_in_chunk = 0
        
        # Return current action from buffer
        action = self.raw_actions_buffer[self.step_in_chunk]
        self.step_in_chunk += 1
        
        return action
    
    def close(self):
        """Close WebSocket connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            print("🔌 WebSocket connection closed")

    def _save_camera_images(self, obs: Dict, step: int, episode: int) -> None:
        """
        保存base_camera和hand_camera的原始图像
        
        Args:
            obs: 环境观测
            step: 当前步数
            episode: 当前episode编号
        """
        if 'sensor_data' not in obs:
            return
        
        sensor_data = obs['sensor_data']
        
        for cam_key in self.camera_keys:
            if cam_key not in sensor_data or 'rgb' not in sensor_data[cam_key]:
                continue
            
            # 提取原始RGB图像
            img_tensor = sensor_data[cam_key]['rgb']
            
            # 处理batch维度
            if img_tensor.ndim == 4:
                img = img_tensor[0]
            else:
                img = img_tensor
            
            # 转换为numpy
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            
            # 确保HWC格式
            if img.shape[0] == 3 and img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
            
            # 转换为uint8
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            # 保存图像：episode_{ep}_step_{step}_{camera}.png
            save_path = Path(self.save_dir) / f"episode_{episode:03d}_step_{step:04d}_{cam_key}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_model(usr_args: Dict) -> FGManipModelClient:
    """Factory function for model instantiation (required by eval framework)."""
    return FGManipModelClient(
        policy_ckpt_path=usr_args["policy_ckpt_path"],
        unnorm_key=usr_args.get("unnorm_key"),
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 10093),
        image_size=usr_args.get("image_size", [224, 224]),
        camera_keys=usr_args.get("camera_keys", ["base_camera", "hand_camera"]),
        use_ddim=usr_args.get("use_ddim", True),
        num_ddim_steps=usr_args.get("num_ddim_steps", 10),
    )


def reset_model(model: FGManipModelClient, instruction: str):
    """Reset model with new instruction."""
    model.reset(instruction)


def eval_step(env, model: FGManipModelClient, obs: Dict) -> np.ndarray:
    """
    Single evaluation step - gets action from model and executes in environment.
    
    Args:
        env: FGManip environment instance
        model: FGManipModelClient instance
        obs: Current observation from environment
    
    Returns:
        Action executed in environment
    """
    # Get instruction from environment (FGManip-specific)
    instruction = getattr(env.unwrapped, "get_instruction", lambda: "perform the task")()
    
    # Get action from model
    action = model.get_action(obs, instruction)
    
    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)
    
    return obs, reward, terminated, truncated, info, action