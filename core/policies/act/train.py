ALGO_NAME = 'BC_ACT_state'

import argparse
import os
import random
from distutils.util import strtobool
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os, sys
# Add paths for imports BEFORE any other imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'act'))  # ACT modules
sys.path.insert(0, '/home/robot/workspace/FGManip')  # FGManip root

# Now do all imports
from torch.utils.tensorboard import SummaryWriter
from act.evaluate import evaluate
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from core.env import GraspPartEnv, SlideAlongEnv, StandUpEnv, ToggleSwitchEnv, DrawTriangleEnv, PlugChargerEnv, PegInHoleEnv
from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import tyro

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = 'pickcube.trajectory.state.pd_joint_delta_pos.cpu.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # ACT specific arguments
    lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    kl_weight: float = 10
    """weight for the kl loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 4
    num_queries: int = 30
    pre_norm: bool = False

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "physx_cpu" or "physx_cuda" """
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

    # FGManip environment specific arguments
    object_name: str = "cabinet"
    """the name of the object to manipulate (for FGManip environments)"""
    part_name: str = "handle"
    """the name of the part to manipulate (for FGManip environments)"""
    robot_init_qpos_noise: float = 0.02
    """noise added to robot initial joint positions (for FGManip environments)"""

    # Eval-time domain randomization (OFF by default to preserve existing behavior)
    enable_dr_eval: bool = False
    """if enabled, evaluate clean + camera/light perturbation profiles at eval time"""
    camera_pos_levels: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.06])
    """camera position jitter levels (meters) for L1/L2/L3"""
    camera_rot_levels_deg: List[float] = field(default_factory=lambda: [2.0, 6.0, 12.0])
    """camera rotation jitter levels (degrees) for L1/L2/L3"""
    light_ambient_delta_levels: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.40])
    """ambient light +/- delta around 0.5 for L1/L2/L3"""

class SmallDemoDataset_ACTPolicy(Dataset): # Load everything into GPU memory
    def __init__(self, data_path, num_queries, device, num_traj):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from act.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['obs'] is a list of dicts containing observation data
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        # Handle different observation formats (FGManip uses 'obs', original ACT uses 'observations')
        if 'obs' in trajectories:
            # FGManip format: trajectories['obs'] is list of dicts (one per trajectory)
            # Each dict contains observation arrays: {'state': array, 'sensor_param': array, ...}
            trajectories['observations'] = trajectories['obs']

        # Convert to tensors (skip dict fields)
        for k, v in trajectories.items():
            if k in ['obs']:  # Skip the original obs key, we'll handle observations separately
                continue
            if k == 'observations':
                # Handle FGManip observation dicts: convert state arrays to tensors
                for i in range(len(v)):
                    if isinstance(v[i], dict) and 'state' in v[i]:
                        v[i]['state'] = torch.Tensor(v[i]['state']).to(device)
            elif isinstance(v, list) and len(v) > 0:
                for i in range(len(v)):
                    if isinstance(v[i], (np.ndarray, list)):
                        trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in range(self.num_traj):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [
                (traj_idx, ts) for ts in range(episode_len)
            ]

        print(f"Length of Dataset: {len(self.slices)}")

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in args.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]
        # import ipdb; ipdb.set_trace()
        # get observation at ts only
        # For FGManip, trajectories['observations'][traj_idx] is a dict with keys like 'state', 'sensor_param', etc.
        # Each key contains an array of shape (traj_length, ...)
        obs = self.trajectories['observations'][traj_idx]['state'][ts]

        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        # normalize obs and act_seq
        if not self.delta_control:
            obs = (obs - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return {
            'observations': obs,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)

    def get_norm_stats(self):
        traj_idx, ts = self.slices[index]

        # get observation at start_ts only
        obs = self.trajectories['observations'][traj_idx][ts]
        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        # normalize obs and act_seq
        if not self.delta_control:
            obs = (obs - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return {
            'observations': obs,
            'actions': act_seq,
        }

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space.shape) == 1 # (obs_dim,)
        assert len(env.single_action_space.shape) == 1 # (act_dim,)
        #assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()

        self.kl_weight = args.kl_weight
        self.state_dim = env.single_observation_space.shape[0]
        self.act_dim = env.single_action_space.shape[0]

        # CNN backbone
        backbones = None

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def compute_loss(self, obs, action_seq):
        # forward pass
        a_hat, (mu, logvar) = self.model(obs, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        # store all loss
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def get_action(self, obs):
        # forward pass
        a_hat, (_, _) = self.model(obs) # no action, sample from prior
        return a_hat

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')


def compute_ausc(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, num=arr.size)
    return float(np.trapz(arr, x))


def build_eval_profiles(base_env_kwargs: Dict, args: Args) -> Dict[str, Dict]:
    profiles: Dict[str, Dict] = {"clean": dict(base_env_kwargs)}
    if not args.enable_dr_eval:
        return profiles

    cam_pos_levels = list(args.camera_pos_levels)[:3]
    cam_rot_levels = list(args.camera_rot_levels_deg)[:3]
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

    light_levels = list(args.light_ambient_delta_levels)[:3]
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array")

    if args.env_id in ["grasp_part", "stand_up", "toggle_switch", "toggle_switch_table", "lid_opening", "slide_along"]:
        env_kwargs.update({
            "object_name": args.object_name,
            "part_name": args.part_name,
            "robot_init_qpos_noise": args.robot_init_qpos_noise,
        })

    # Add environment-specific parameters
    if args.env_id == "lid_opening":
        env_kwargs.update({
            "robot_stiffness": 1000.0,
            "robot_damping": 100.0,
            "robot_force_limit": 100.0,
        })
    elif args.env_id == "slide_along":
        env_kwargs.update({
            "success_delta_frac": 0.30,
            "static_threshold": 0.20,
        })

    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    eval_profiles = build_eval_profiles(env_kwargs, args)
    eval_envs = {}
    for profile_name, profile_env_kwargs in eval_profiles.items():
        video_dir = None
        if args.capture_video:
            video_dir = f"runs/{run_name}/videos/{profile_name}"
        eval_envs[profile_name] = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            profile_env_kwargs,
            other_kwargs,
            video_dir=video_dir,
        )
    envs = eval_envs["clean"]

    # dataloader setup
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, args.num_queries, device, num_traj=args.num_demos)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT",
            tags=["act"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # agent setup
    agent = Agent(envs, args).to(device)

    # optimizer setup
    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    # LR drop by a factor of 10 after lr_drop iters
    lr_drop = int((2/3)*args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    # Evaluation
    #eval_kwargs = dict(
    #    stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
    #    max_timesteps=gym_utils.find_max_episode_steps_value(envs), device=device, sim_backend=args.sim_backend
    #)
    eval_kwargs = dict(
        stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps, device=device, sim_backend=args.sim_backend
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    print("Training begins...")
    agent.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        last_tick = time.time()
        # forward and compute loss
        loss_dict = agent.compute_loss(
            obs=data_batch['observations'],  # (B, obs_dim)
            action_seq=data_batch['actions'],  # (B, num_queries, act_dim)
        )
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        timings["update"] += time.time() - last_tick

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            last_tick = time.time()

            ema.copy_to(ema_agent.parameters())

            profile_metrics = {}
            for profile_name, profile_env in eval_envs.items():
                raw_metrics = evaluate(args.num_eval_episodes, ema_agent, profile_env, eval_kwargs)
                mean_metrics = {k: float(np.mean(v)) for k, v in raw_metrics.items()}
                profile_metrics[profile_name] = mean_metrics

                if profile_name == "clean":
                    print(f"Evaluated {len(raw_metrics['success_at_end'])} episodes")
                for k, v in mean_metrics.items():
                    writer.add_scalar(f"eval/{profile_name}/{k}", v, cur_iter)
                print(f"[{profile_name}] " + ", ".join([f"{k}: {v:.4f}" for k, v in mean_metrics.items()]))

            # Aggregate AUSC across perturbation levels if DR eval is enabled.
            if args.enable_dr_eval:
                if all(name in profile_metrics for name in ["clean", "cam_l1", "cam_l2", "cam_l3"]):
                    cam_srs = [profile_metrics[name].get("success_at_end", np.nan) for name in ["clean", "cam_l1", "cam_l2", "cam_l3"]]
                    ausc_cam = compute_ausc(cam_srs)
                    writer.add_scalar("eval/ausc/camera", ausc_cam, cur_iter)
                    print(f"AUSC(camera): {ausc_cam:.4f}")
                if all(name in profile_metrics for name in ["clean", "light_l1", "light_l2", "light_l3"]):
                    light_srs = [profile_metrics[name].get("success_at_end", np.nan) for name in ["clean", "light_l1", "light_l2", "light_l3"]]
                    ausc_light = compute_ausc(light_srs)
                    writer.add_scalar("eval/ausc/light", ausc_light, cur_iter)
                    print(f"AUSC(light): {ausc_light:.4f}")

            timings["eval"] += time.time() - last_tick

            # Keep checkpoint selection based on clean profile for backward compatibility.
            clean_metrics = profile_metrics["clean"]
            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in clean_metrics and clean_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = clean_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best clean/{k}_rate: {clean_metrics[k]:.4f}. Saving checkpoint.')

        if cur_iter % args.log_freq == 0:
            print(f"Iteration {cur_iter}, loss: {total_loss.item()}")
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)
        # Checkpoint
        if args.save_freq is not None and cur_iter % args.save_freq == 0:
            save_ckpt(run_name, str(cur_iter))

    for _name, _env in eval_envs.items():
        _env.close()
    writer.close()
