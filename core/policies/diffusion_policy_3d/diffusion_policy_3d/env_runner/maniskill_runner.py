import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env.maniskill_wrapper import ManiSkillEnv
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.utils import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_utils as logger_util
from termcolor import cprint

import os
import time

from typing import Optional

import gymnasium as gym
from mani_skill.utils import gym_utils
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


from diffusion_policy_3d.common.observation_wrapper import FlattenPoindCloudObservationWrapper
from diffusion_policy_3d.common.multistep_wrapper import MultiStepWrapper


def _compute_mad(actions: np.ndarray, eps: float = 1e-12) -> float:
    """ACT-style MAD: mean L2 norm of first-order action differences."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")
    da = np.diff(arr, axis=0) / max(1.0, eps)
    return float(np.mean(np.linalg.norm(da, ord=2, axis=1)))


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    n_obs_steps: int,
    n_action_steps: int,
    max_episode_steps: int,
    reward_agg_method: str,
    device: str,
    use_point_crop: bool,
    use_pc_color: bool,
    num_points: int,
    env_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                # This wrapper wraps any maniskill env created via gym.
                # make to ensure the outputs of env. render, env. reset, env. step are all numpy arrays and are not batched.
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                env = ManiSkillEnv(env, task_name=env_id, use_point_crop=use_point_crop, use_pc_color=use_pc_color, num_points=num_points)

                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="3d_diffusion_policy",
                        source_desc="3d_diffusion_policy evaluation rollout",
                    )

                env = MultiStepWrapper(env=env,
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        reward_agg_method=reward_agg_method)

                cprint("[ManiSkillEnv] observation mode: {}.".format(env_kwargs["obs_mode"]), "red")
                cprint("[ManiSkillEnv] action space: {}.".format(env.action_space.shape), "red")
                cprint("[ManiSkillEnv] observation space: agent_post{}, point_cloud{}.".format(env.observation_space["agent_pos"].shape, env.observation_space["point_cloud"].shape), "red")

                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk()

        assert num_envs == 1
        seed = num_envs - 1
        env = cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                )
    else:
        # TODO: The following code should be modified
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        # TODO: check FrameStack wrapper, if we need to change
        # Answer: Yes. This is used to stack # of 'obs_horizon' history observations
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="3d_diffusion_policy",
                source_desc="3d_diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillEnv(
            MultiStepWrapper(env=env,
                             n_obs_steps=n_obs_steps,
                             n_action_steps=n_action_steps,
                             max_episode_steps=max_episode_steps,
                             reward_agg_method=reward_agg_method),
            task_name=env_id,
            use_point_crop=use_point_crop,
            num_points=num_points,
        )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env

class ManiSkillRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 control_mode,
                 eval_episodes=20,
                 max_steps=300,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 num_eval_envs=1,
                 sim_backend='cpu',
                 exp_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 use_pc_color=True,
                 num_points=512,
                 capture_video=True,
                 env_kwargs: Optional[dict] = None,
                 video_root_dir: str = "/nat/demos/dp3/out",
                 dr_eval: bool = False,
                 camera_pos_jitter: float = 0.0,
                 camera_rot_jitter_deg: float = 0.0,
                 light_ambient_delta: float = 0.0,
                 allow_cpu_fallback: bool = True,
                 cuda_retry_on_cublas: int = 2,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        reward_agg_method='sum'

        default_env_kwargs = dict(
            control_mode=control_mode,
            reward_mode="sparse",
            obs_mode="pointcloud",
            render_mode="all",
        )
        # Allow task yaml to pass custom env kwargs (e.g., object_name/part_name).
        if env_kwargs is None:
            env_kwargs = {}
        merged_env_kwargs = {**default_env_kwargs, **env_kwargs}
        # ACT-style eval-time perturbation controls.
        if dr_eval:
            delta = max(0.0, float(light_ambient_delta))
            ambient_low = max(0.0, 0.5 - delta)
            ambient_high = min(1.0, 0.5 + delta)
            merged_env_kwargs.update(
                eval_randomize_camera=True,
                eval_camera_pos_jitter=float(camera_pos_jitter),
                eval_camera_rot_jitter_deg=float(camera_rot_jitter_deg),
                eval_randomize_light=True,
                eval_ambient_low=ambient_low,
                eval_ambient_high=ambient_high,
            )

        seed = 1
        if exp_name is None:
            exp_name = os.path.basename(__file__)[: -len(".py")]
            run_name = f"{task_name}__{exp_name}__{seed}__{int(time.time())}"
        else:
            run_name = exp_name

        self.env = make_eval_envs(
            task_name,
            num_eval_envs,
            sim_backend,
            n_obs_steps,
            n_action_steps,
            max_steps,
            reward_agg_method,
            device,
            use_point_crop,
            use_pc_color,
            num_points,
            merged_env_kwargs,
            video_dir=os.path.join(video_root_dir, run_name, "videos") if capture_video else None,
            wrappers=[FlattenPoindCloudObservationWrapper],
        )

        self.eval_episodes = eval_episodes

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.cuda_retry_on_cublas = max(0, int(cuda_retry_on_cublas))
        self.policy_device = torch.device(device)

    def run(self, policy: BasePolicy):
        device = self.policy_device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        all_mad = []
        all_exp_neg_mad = []
        env = self.env

        for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                     desc=f"Eval in ManiSkill {self.task_name} Pointcloud Env", leave=False,
                                     mininterval=self.tqdm_interval_sec):

            # start rollout
            obs, _ = env.reset(seed=0)
            # Try to (re)move policy to target device each episode.
            # This lets us recover GPU inference if CUDA becomes healthy again.
            try:
                policy = policy.to(device=self.policy_device)
                device = self.policy_device
                if str(device).startswith("cuda"):
                    warm_a = torch.randn((64, 64), device=device, dtype=torch.float32)
                    warm_b = torch.randn((64, 64), device=device, dtype=torch.float32)
                    _ = warm_a @ warm_b
                    torch.cuda.synchronize(device)
            except Exception as e:
                if self.allow_cpu_fallback:
                    cprint(f"[Warn] Could not move policy to {self.policy_device}; using CPU for this episode. ({e})", "yellow")
                    policy = policy.cpu()
                    device = torch.device("cpu")
                    dtype = torch.float32
                else:
                    raise
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            count = 0
            episode_actions = []
            while not done:
                count += 1
                if str(device).startswith("cuda"):
                    torch.cuda.set_device(device)
                np_obs_dict = dict(obs)
                def _to_policy_tensor(x):
                    arr = np.asarray(x)
                    tensor = torch.from_numpy(arr).to(device=device)
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype=dtype)
                    return tensor

                obs_dict = dict_apply(np_obs_dict, _to_policy_tensor)

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    try:
                        action_dict = policy.predict_action(obs_dict_input)
                    except RuntimeError as e:
                        # Some environments occasionally fail CUDA GEMM init.
                        # Retry on CUDA first, optionally fallback to CPU.
                        if ("CUBLAS_STATUS_NOT_INITIALIZED" in str(e)) and str(device).startswith("cuda"):
                            retried_ok = False
                            last_err = e
                            for _ in range(self.cuda_retry_on_cublas):
                                try:
                                    # Recreate CUDA context-dependent handles.
                                    policy = policy.cpu()
                                    torch.cuda.empty_cache()
                                    policy = policy.cuda()
                                    warm_a = torch.randn((64, 64), device=device, dtype=torch.float32)
                                    warm_b = torch.randn((64, 64), device=device, dtype=torch.float32)
                                    _ = warm_a @ warm_b
                                    torch.cuda.synchronize(device)
                                    action_dict = policy.predict_action(obs_dict_input)
                                    retried_ok = True
                                    break
                                except RuntimeError as e_retry:
                                    last_err = e_retry
                                    if "CUBLAS_STATUS_NOT_INITIALIZED" not in str(e_retry):
                                        raise
                            if retried_ok:
                                pass
                            elif self.allow_cpu_fallback:
                                cprint("[Warn] CUDA cublas init failed during eval; switching policy inference to CPU.", "yellow")
                                policy = policy.cpu()
                                device = torch.device("cpu")
                                dtype = torch.float32
                                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype))
                                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                                obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                                action_dict = policy.predict_action(obs_dict_input)
                            else:
                                raise last_err
                        else:
                            raise

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                episode_actions.append(np.asarray(action, dtype=np.float64).reshape(-1).copy())

                obs, reward, done, _, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            if len(episode_actions) >= 2:
                mad = _compute_mad(np.asarray(episode_actions, dtype=np.float64))
                exp_neg_mad = float(np.exp(-mad)) if np.isfinite(mad) else 0.0
            else:
                mad = float("nan")
                exp_neg_mad = 0.0

            cprint(
                f"[Episode {episode_idx + 1:03d}] steps={count}, success={int(is_success)}, "
                f"reward={float(traj_reward):.4f}, mad={mad:.6f}, exp(-mad)={exp_neg_mad:.6f}",
                "cyan",
            )

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_mad.append(mad)
            all_exp_neg_mad.append(exp_neg_mad)

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['mean_mad'] = float(np.nanmean(np.asarray(all_mad, dtype=np.float64)))
        log_data['mean_exp_neg_mad'] = float(np.mean(np.asarray(all_exp_neg_mad, dtype=np.float64)))

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        #
        # if save_video:
        #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        #     log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset(seed=0)
        # videos = None

        return log_data


