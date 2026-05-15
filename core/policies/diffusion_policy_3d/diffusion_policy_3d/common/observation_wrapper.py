import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

class FlattenPoindCloudObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the pointcloud mode observations into a dictionary with two keys, "poindcloud" and "state"

    Args:
        pointcloud (bool): Whether to include pointcloud in the observation
        state (bool): Whether to include state data in the observation
    """

    def __init__(self, env: gym.Env, pointcloud=True, state=True, rgb=True) -> None:
        super().__init__(env)
        self.include_pointcloud = pointcloud
        self.include_state = state
        self.include_rgb = rgb
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation: Dict):
        pointcoud_data = observation.pop("pointcloud")
        pointcloud = []
        rgb = []
        # assert len(pointcoud_data) == 1
        pointcloud.append(pointcoud_data['xyzw'])
        rgb.append(pointcoud_data['rgb'])

        if "sensor_param" in observation:
            del observation["sensor_param"]
        if "sensor_data" in observation:
            del observation["sensor_data"]

        if 'is_grasped' in observation['extra']:
            del observation['extra']['is_grasped']
        if 'goal_pos' in observation['extra']:
            del observation['extra']['goal_pos']
            
        state = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        ) # [29] including joint positions [7+2], velocities [7+2], goal position [3], end-effector pose [7], is_grasped [1]

        # qpos = observation['agent']['qpos']
        # tcp_pose = observation['extra']['tcp_pose']
        # state = torch.cat((qpos[:,:7],tcp_pose[:,:3]), dim=1)

        ret = dict()
        if self.include_state:
            ret["state"] = state.clone() # Tensor
        if self.include_pointcloud:
            ret["pointcloud"] = pointcloud[0].clone() # Tensor
        if self.include_rgb:
            ret["rgb"] = rgb[0].clone() # Tensor
        return ret
