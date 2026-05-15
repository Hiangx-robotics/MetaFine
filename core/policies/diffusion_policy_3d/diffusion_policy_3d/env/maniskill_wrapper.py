from termcolor import cprint
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from mani_skill.utils import gym_utils
from mani_skill.utils import common
from diffusion_policy_3d.common.utils import downsample_with_fps

from mani_skill.envs.sapien_env import BaseEnv

# the range to crop poindcloud
TASK_BOUDNS = {
    'GraspCup-v1': [-0.7, -1, 0.01, 1, 1, 100],
    'PickCube-v1': [-0.7, -1, 0.01, 1, 1, 100],
    'default': [-0.7, -1, 0.01, 1, 1, 100],
}


class ManiSkillEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 task_name,
                 use_point_crop=True,
                 use_pc_color=True,
                 num_points=1024,
                 ):
        super().__init__(env)

        self.use_point_crop = use_point_crop
        self.use_pc_color = use_pc_color
        cprint("[ManiSkillEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        cprint("[ManiSkillEnv] use_pc_color: {}".format(self.use_pc_color), "cyan")
        self.num_points = num_points

        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        self.episode_length = self._max_episode_steps = 300 # lch fixed？

        # TODO: check the following action, state, observation space dimension
        self.action_space = self.base_env.action_space
        self.obs_state_dim = self.base_env.observation_space["state"].shape[1] # TODO 29 or 25
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_state_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3), # TODO: check if shape[1] = 3 or 6 (when with rgb)
                dtype=np.float32
            ),
        })

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def get_point_cloud(self, raw_obs, use_point_crop=True, use_rgb=True):

        xyzw = raw_obs['pointcloud']
        filtered_xyzw = xyzw[xyzw[:, -1] == 1] # filter invalid pointcloud, e.g., too far away
        point_cloud = filtered_xyzw[...,:3]

        if use_rgb:
            rgb = raw_obs['rgb']
            filtered_rgb = rgb[xyzw[:, -1] == 1]
            point_cloud = np.concatenate((point_cloud, filtered_rgb), axis=1)

        # if self.pc_transform is not None:
        #     point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        # if self.pc_scale is not None:
        #     point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        # if self.pc_offset is not None:
        #     point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        # Downsample or pad pointclouds to fixed size
        if point_cloud.shape[0] < self.num_points:
            # Pad with repetition of existing points
            shortage = self.num_points - point_cloud.shape[0]
            if point_cloud.shape[0] > 0:
                pad_indices = np.random.choice(point_cloud.shape[0], shortage, replace=True)
                padding = point_cloud[pad_indices]
            else:
                padding = np.zeros((shortage, point_cloud.shape[1]), dtype=point_cloud.dtype)
            point_cloud = np.concatenate([point_cloud, padding], axis=0)
        elif point_cloud.shape[0] > self.num_points:
            point_cloud = downsample_with_fps(point_cloud, num_points=self.num_points)

        return point_cloud
        # return np.stack(point_cloud, axis=0) # convert to ndarray

    def step(self, action):

        raw_obs, reward, done, truncated, env_info = self.env.step(action)
        self.cur_step += 1

        # raw_obs: dict with 3 items
        #           raw_obs["state"]: dict with state [state_shape]
        #           raw_obs["pointcloud"]: dict with pointcloud [num, 4]
        #           raw_obs["rgb"]: dict with rgb [num, 3]
        robot_state = raw_obs["state"]
        point_cloud = self.get_point_cloud(raw_obs, use_point_crop=self.use_point_crop, use_rgb=self.use_pc_color)

        obs_dict = {
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }

        done = done or self.cur_step >= self.episode_length

        return obs_dict, reward, done, truncated, env_info

    def reset(self, **kwargs):
        raw_obs, env_info = self.env.reset(**kwargs)
        self.cur_step = 0

        robot_state = raw_obs["state"]
        point_cloud = self.get_point_cloud(raw_obs, use_point_crop=self.use_point_crop, use_rgb=self.use_pc_color)

        obs_dict = {
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }

        return obs_dict, env_info

    def render(self):
        ret = self.env.render()
        if self.render_mode in ["rgb_array", "sensors", "all"]:
            return common.unbatch(common.to_numpy(ret))