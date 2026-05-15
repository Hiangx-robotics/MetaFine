from typing import Any, Union, List
import json
from pathlib import Path
from mani_skill.envs.scene import ManiSkillScene
import numpy as np
import sapien
import torch
import math
from mani_skill.utils import gym_utils

from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import common, sapien_utils
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.utils.structs.types import SimConfig, SceneConfig
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import io_utils
from mani_skill.utils.structs.types import GPUMemoryConfig
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.types import Device
from transforms3d.euler import euler2quat
from .scene import (
    GraspPartSceneBuilder, ToggleSwitchSceneBuilder, ToggleSwitchTableSceneBuilder,
    StandUpSceneBuilder, SlideAlongSceneBuilder, DoorSceneBuilder, RotateSceneBuilder,
)
from .env_mixins import EvalDREnvMixin

@register_env("grasp_part", max_episode_steps=300)
class GraspPartEnv(EvalDREnvMixin, BaseEnv):
    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
    ]
    agent: Union[Panda, PandaWristCam]
    goal_thresh = 0.1
    cube_spawn_center = (0, 0)

    def __init__(self,
                 *args,
                 robot_uids="panda_wristcam",
                 robot_init_qpos_noise=0.02,
                 object_name="cabinet_01",
                 part_name="handle",
                 **kwargs):
        
        # Cache object + part choice for the rest of __init__.
        self.object_name = object_name
        self.part_name = part_name
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Validate part_name exists in this object's grasp_poses.json.
        self._validate_part_exists()

        # Load the object's metadata (scale, initial pose, grasp parts).
        self._load_object_config()

        # Pull the default Panda camera config.
        cfg = PICK_CUBE_CONFIGS["panda"]

        # Forward env-shaping params from the config.
        self.goal_thresh = 0.1  # success distance threshold
        self.cube_spawn_center = cfg["cube_spawn_center"]  # object spawn centre
        self.max_goal_height = cfg["max_goal_height"]  # max goal z

        # Sensor (policy-observation) camera config.
        self.sensor_cam_eye_pos = [0.7, 0, 0.9]  # sensor camera eye position
        self.sensor_cam_target_pos = [-0.2, 0, 0.1]  # sensor camera target point
        # Render (human-view) camera config — used when saving videos.
        self.human_cam_eye_pos = self.sensor_cam_eye_pos
        self.human_cam_target_pos = self.sensor_cam_target_pos
        self.verbose = False
        # Engagement state (Phase E). Set by interaction skills, cleared by
        # release skills and at the start of every episode.
        self._engaged_part = None
        # Forward to BaseEnv.__init__.
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def _is_episode_ending(self):
        """Return True on the final step of the current episode."""
        current_steps = self.elapsed_steps
        max_steps = gym_utils.find_max_episode_steps_value(self)
        # Multi-env: compare per-env step counters.
        if self.num_envs > 1:
            return current_steps >= max_steps - 1  # final step
        else:
            return current_steps.item() >= max_steps - 1

    def _validate_part_exists(self):
        """Validate that ``self.part_name`` is present in this object's grasp annotations.

        Reads ``assets/{object_name}/grasp_poses.json`` and checks for an
        exact or fuzzy substring match. On fuzzy match the first hit wins
        and ``self.part_name`` is rewritten so downstream skill code sees
        the canonical name.

        Raises:
            FileNotFoundError: when grasp_poses.json is missing.
            ValueError: when no match (exact or fuzzy) can be found.
        """
        grasp_poses_path = Path(__file__).parent.parent / "assets" / self.object_name / "grasp_poses.json"

        if not grasp_poses_path.exists():
            raise FileNotFoundError(f"Grasp poses file not found: {grasp_poses_path}")

        try:
            with open(grasp_poses_path, 'r', encoding='utf-8') as f:
                grasp_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in grasp poses file: {e}")

        available_parts = list(grasp_data.get("grasp_parts", {}).keys())

        if not available_parts:
            raise ValueError(f"No grasp parts found in {grasp_poses_path}")

        # Exact match first; fall back to fuzzy substring matching.
        if self.part_name not in available_parts:
            # Fuzzy match by substring containment.
            fuzzy_matches = [part for part in available_parts if self.part_name in part]
            if not fuzzy_matches:
                raise ValueError(
                    f"Part '{self.part_name}' not found in available parts: {available_parts}"
                )
            else:
                # Use the first fuzzy hit and pin it on self.
                self.part_name = fuzzy_matches[0]
                

    def _load_object_config(self):
        """Load object metadata from ``assets/{object_name}/model_data.json``.

        Reads scale, initial pose, joint qpos, and the grasp_parts dict
        from the model_data file. Falls back to :func:`_get_default_config`
        when the file is missing or unparseable.
        """
        config_path = Path(__file__).parent.parent / "assets" / self.object_name / "model_data.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.object_config = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in config file {config_path}: {e}")
                self.object_config = self._get_default_config()
        else:
            # if self.verbose:
            #     print(f"Warning: Config file not found {config_path}, using defaults")
            self.object_config = self._get_default_config()

    def _get_default_config(self):
        """Return a benign default object config used when model_data.json is missing."""
        return {
            "scale": 1.0,
            "transform_matrix": [[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]],
            "init_qpos": [],
            "grasp_parts": {}
        }

    @property
    def _default_sensor_configs(self):
        """Return the default sensor (policy-input) camera config list."""
        # Compose the look_at view matrix from eye + target.
        eye = np.array(self.sensor_cam_eye_pos, dtype=np.float32)
        target = np.array(self.sensor_cam_target_pos, dtype=np.float32)

        eye, target = self._maybe_jitter_camera(eye, target)

        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        # Wrap in the CameraConfig the env expects.
        return [CameraConfig("base_camera", pose, 224, 224, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        """Return the default render (video-output) camera config."""
        # Compose the look_at view matrix from eye + target.
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        # Wrap in CameraConfig.
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        """Load the scene.

        Builds the table workspace and the articulated object via the
        GraspPartSceneBuilder.

        Args:
            options (dict): scene loading options (unused today).
        """
        # Build via GraspPartSceneBuilder.
        self.scene_builder = GraspPartSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Pick up the constructed target object.
        self.target_object = self.scene_builder.target_object

        # Spawn a green sphere as a goal-position marker.
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh/4,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset state for a fresh episode.

        Args:
            env_idx (torch.Tensor): which env indices to (re)initialise.
            options (dict): reserved for future use.
        """
        with torch.device(self.device):
            # Batch size.
            b = len(env_idx)

            # Initialise the scene with fixed object placement.
            self.scene_builder.initialize(env_idx)

        # Reset engagement state at the start of every episode (Phase E).
        self.disengage()

    # ------------------------------------------------------------------ #
    # Engagement state (Phase E)                                          #
    # ------------------------------------------------------------------ #
    # ``_engaged_part`` records which annotated part the gripper currently
    # has in its grasp (or is in contact with, for non-grasping engagements
    # like a press). It is set by an interaction-phase skill at the moment
    # of contact and cleared by a release skill. Continuation skills (e.g.
    # ``pure_slide``) read it to know what to operate on without needing
    # the caller to plumb the part name through every step.

    def engage(self, part_name: str) -> None:
        """Record that ``part_name`` is currently engaged with the gripper.

        Called by interaction-phase skill solvers (``grasp_part``, etc.)
        after the contact-establishing motion succeeds. No-op for envs that
        do not maintain articulated-object state.
        """
        self._engaged_part = part_name

    def disengage(self) -> None:
        """Clear the engagement record. Called by release-style skills."""
        self._engaged_part = None

    @property
    def engaged_part(self):
        """Currently engaged part name, or ``None`` if not engaged."""
        return getattr(self, "_engaged_part", None)

    def _get_obs_extra(self, info: dict):
        """Return per-step extra observation data.

        Base implementation only forwards the TCP pose; subclasses extend
        this with task-specific fields.
        """
        # Base extras: just expose the TCP pose.
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)  # end-effector pose
        return obs

    def evaluate(self):
        """Heuristic grasp success based on gripper angle.

        Bucketed: fully closed (< 0.01) = nothing in the jaws, partially
        closed (0.01–0.03) = holding something, fully open (> 0.03) = released.
        The middle band signals a successful grasp.
        """
        # Read gripper joint angles.
        qpos = self.agent.robot.get_qpos()
        left_gripper_angle = qpos[0, -2]   # second-to-last qpos slot
        right_gripper_angle = qpos[0, -1]  # last qpos slot
        gripper_total_angle = abs(left_gripper_angle) 
        # print(f"gripper_total_angle: {gripper_total_angle}")
        # Heuristic thresholds.
        fully_closed_threshold = 0.01  # below this = jaws bottomed out (empty)
        open_threshold = 0.03          # above this = gripper released
        is_fully_closed = gripper_total_angle.item() < fully_closed_threshold
        is_open = gripper_total_angle.item() > open_threshold

        # Grasp iff we're between fully-closed and fully-open.
        grasped_something = not is_fully_closed and not is_open

        # Success = something is held.
        success = torch.tensor([grasped_something], device=self.device)

        # Whether the arm itself has come to rest.
        is_robot_static = self.agent.is_static(threshold=0.2)

        return {
            "success": success,
            "grasped_something": torch.tensor([grasped_something], device=self.device),
            "is_gripper_fully_closed": torch.tensor([is_fully_closed], device=self.device),
            "is_gripper_open": torch.tensor([is_open], device=self.device),
            "gripper_total_angle": torch.tensor([gripper_total_angle], device=self.device),
            "left_gripper_angle": torch.tensor([left_gripper_angle], device=self.device),
            "right_gripper_angle": torch.tensor([right_gripper_angle], device=self.device),
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Compute the dense per-step reward.

        Base implementation returns zeros; subclasses with reward shaping
        override this method.
        """
        # Base: zero reward; subclasses provide task-specific shaping.
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Normalised dense reward — same contract as compute_dense_reward but in [0, 1]."""
        # Base: zero.
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

@register_env("align_to_part", max_episode_steps=500)
class AlignToPartEnv(GraspPartEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
@register_env("stand_up", max_episode_steps=250)
class StandUpEnv(GraspPartEnv):
    """
    
    Pick up a toppled object and stand it upright. Inherits all config from GraspPartEnv.
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _load_scene(self, options: dict):
        self.scene_builder = StandUpSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        
        self.target_object = self.scene_builder.target_object

    def evaluate(self):
        # TODO: also check grasp success + placement on the table.
        
        obj_pose = self.target_object.pose  # shape (N, 7)

        R = obj_pose.to_transformation_matrix()[:, :3, :3]  # (N, 3)
        
        obj_z_axis = R @ torch.tensor([0, 0, 1.0])
        
        obj_z_axis = obj_z_axis / np.linalg.norm(obj_z_axis)
        cos_angle = obj_z_axis[:, 2]
        
        angle_deg = torch.tensor(10.0, device=obj_z_axis.device)
        angle_rad = angle_deg * torch.pi / 180.0
        cos_threshold = torch.cos(angle_rad) 
        is_upright = cos_angle > cos_threshold
        
        is_robot_static = self.agent.is_static(threshold=0.2)

        success = is_upright & is_robot_static
        return {
            "success": success,
            "is_upright": is_upright,
            # "is_obj_placed": is_on_table,
            # "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
        }

@register_env("peg_in_hole", max_episode_steps=250)
class PegInHoleEnv(EvalDREnvMixin, BaseEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and insert the orange end into the box with a hole in it.

    **Randomizations:**
    - Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value. (during reconfiguration)
    - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.003m of clearance. (during reconfiguration)
    - Peg is laid flat on table and has it's xy position and z-axis rotation randomized
    - Box is laid flat on table and has it's xy position and z-axis rotation randomized

    **Success Conditions:**
    - The white end of the peg is within 0.015m of the center of the box (inserted mid way).
    """
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    _clearance = 0.003
    
    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ): 
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    def _build_box_with_hole(self,
        scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
    ):
        builder = scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            sapien.Pose([0, offset + half_center[0], 0]),
            sapien.Pose([0, -offset + half_center[0], 0]),
            sapien.Pose([0, 0, offset + half_center[1]]),
            sapien.Pose([0, 0, -offset + half_center[1]]),
        ]

        mat = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
        )

        for half_size, pose in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder
    @property
    def _default_sim_config(self):
        return SimConfig()
    @property
    def _default_sensor_configs(self):
        eye = np.array([0.5, -0.5, 0.85], dtype=np.float32)
        target = np.array([0.05, -0.1, 0.4], dtype=np.float32)

        eye, target = self._maybe_jitter_camera(eye, target)

        pose = sapien_utils.look_at(eye.tolist(), target.tolist())
        return [CameraConfig("base_camera", pose, 224, 224, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = self._batched_episode_rng.uniform(0.085, 0.125)
            radii = self._batched_episode_rng.uniform(0.015, 0.025)
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)
                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = self._build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            xy = randomization.uniform(
                low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            xy = randomization.uniform(
                low=torch.tensor([-0.05, 0.2]),
                high=torch.tensor([0.05, 0.4]),
                size=(b, 2),
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose * (offset)
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        reward = reaching_reward + is_grasped

        # Stage 3: Orient the grasped peg properly towards the hole

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )

        # Stage 4: Insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5 * (
            1
            - torch.tanh(
                5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
            )
        )
        reward += insertion_reward * (is_grasped & pre_inserted)

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10

@register_env("toggle_switch", max_episode_steps=500)
class ToggleSwitchEnv(GraspPartEnv):
    """
    Generic env for lever / toggle-switch manipulation.

    Drives a Panda arm to flip switch-type articulations.
    Inherits GraspPartEnv for config loading and agent init.

    Args:
        object_name (str): asset folder under assets/ (e.g. '100367').
        part_name (str): name of the part to manipulate.
    """

    def __init__(self,
                 *args,
                 robot_uids="panda_wristcam",
                 robot_init_qpos_noise=0.02,
                 object_name="100920",
                 part_name="button",
                 **kwargs):
        """
        Initialise the env.

        Args:
            object_name (str): asset folder, default "100920".
            part_name (str): part to flip, default "button".
        """
        self.sensor_cam_eye_pos = [-0.8, -0.5, 0.6]  # sensor camera eye
        self.sensor_cam_target_pos = [0.0, 0.0, 0.2]  # sensor camera target
        # Render (human-view) camera config — used when saving videos.
        self.human_cam_eye_pos = self.sensor_cam_eye_pos
        self.human_cam_target_pos = self.sensor_cam_target_pos
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            object_name=object_name,
            part_name=part_name,
            **kwargs
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Cache initial qpos so evaluate() can compare against it.
        if not hasattr(self, 'init_qpos') or self.init_qpos.shape[0] != self.num_envs:
            # Initialise the storage tensor (ensure shape matches).
            current_qpos = self.target_object.get_qpos()
            # self.init_qpos = torch.zeros_like(current_qpos)
            self.init_qpos = current_qpos.clone()

        # Update init_qpos for the envs being reset.
        self.init_qpos[env_idx] = self.target_object.get_qpos()[env_idx]

    @property
    def _default_human_render_camera_configs(self):
        # Camera looks from behind the arm toward the switch (arm at x=-0.615, switch near x=0).
        # Camera is above + behind the arm, aimed at the switch.
        pose = sapien_utils.look_at([-0.8, -0.5, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.scene_builder = ToggleSwitchSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.target_object = self.scene_builder.target_object

        # goal_site is intentionally not built; the toggle task has no goal marker.
        # self.goal_site = actors.build_sphere(
        #     self.scene,
        #     radius=self.goal_thresh,
        #     color=[0, 1, 0, 1],
        #     name="goal_site",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(),
        # )
        # self._hidden_objects.append(self.goal_site)

    def evaluate(self):
        """Evaluate success: switch was flipped past 40% of its stroke."""
        # Read the joint positions of the target object.
        qpos = self.target_object.get_qpos() # shape (N, dof)
        
        # Use the first joint as the toggle joint.
        joints = self.target_object.get_active_joints()
        if not joints:
             return {
                "success": torch.tensor([False], device=self.device),
                "is_toggled": torch.tensor([False], device=self.device),
                "is_robot_static": torch.tensor([True], device=self.device),
            }
            
        joint = joints[0]
        # output is [[min, max]]
        limits = joint.get_limits() 
        joint_min = limits[0][0]
        joint_max = limits[0][1]
        stroke = joint_max - joint_min
        
        # Toggle when displacement exceeds 40% of the total stroke (relative to init_qpos).
        if hasattr(self, 'init_qpos'):
            # Compute displacement.
            delta_qpos = torch.abs(qpos[:, 0] - self.init_qpos[:, 0])
            # 40% of stroke = toggled.
            is_toggled = delta_qpos > (stroke * 0.4)
        else:
            # Fallback (unexpected)
            is_toggled = torch.abs(qpos[:, 0]) > (stroke * 0.2)
        
        is_robot_static = self.agent.is_static(threshold=0.2)
        
        success = is_toggled  # is_robot_static intentionally omitted
        
        return {
            "success": success,
            "is_toggled": is_toggled,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # TODO: implement the reward function.
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

@register_env("toggle_switch_table", max_episode_steps=500)
class ToggleSwitchTableEnv(GraspPartEnv):
    """
    Table-mounted lever / toggle-switch env.

    Drives a Panda arm to flip switch-type articulations.
    Inherits GraspPartEnv. Differs only in using ToggleSwitchTableSceneBuilder.

    Args:
        object_name (str): asset folder under assets/ (e.g. '100367').
        part_name (str): name of the part to manipulate.
    """

    def __init__(self,
                 *args,
                 robot_uids="panda_wristcam",
                 robot_init_qpos_noise=0.02,
                 object_name="100920",
                 part_name="button",
                 **kwargs):
        """
        Initialise the env.

        Args:
            object_name (str): asset folder, default "100920".
            part_name (str): part to flip, default "button".
        """
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            object_name=object_name,
            part_name=part_name,
            **kwargs
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Cache initial qpos so evaluate() can compare against it.
        if not hasattr(self, 'init_qpos') or self.init_qpos.shape[0] != self.num_envs:
            # Initialise the storage tensor (ensure shape matches).
            current_qpos = self.target_object.get_qpos()
            # self.init_qpos = torch.zeros_like(current_qpos)
            self.init_qpos = current_qpos.clone()
            
        # Update init_qpos for the envs being reset.
        self.init_qpos[env_idx] = self.target_object.get_qpos()[env_idx]

    def _load_scene(self, options: dict):
        self.scene_builder = ToggleSwitchTableSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.target_object = self.scene_builder.target_object

        # goal_site is intentionally not built; the toggle task has no goal marker.
        # self.goal_site = actors.build_sphere(
        #     self.scene,
        #     radius=self.goal_thresh/4,
        #     color=[0, 1, 0, 1],
        #     name="goal_site",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(),
        # )
        # self._hidden_objects.append(self.goal_site)
        
    def evaluate(self):
        """Evaluate success: switch was flipped past 40% of its stroke."""
        # Read the joint positions of the target object.
        qpos = self.target_object.get_qpos() # shape (N, dof)
        
        # Total joint stroke.
        joints = self.target_object.get_active_joints()
        if not joints:
             return {
                "success": torch.tensor([False], device=self.device),
                "is_toggled": torch.tensor([False], device=self.device),
                "is_robot_static": torch.tensor([True], device=self.device),
            }
        
        joint = joints[0]
        limits = joint.get_limits()
        joint_min = limits[0][0]
        joint_max = limits[0][1]
        stroke = joint_max - joint_min

        # Toggle when displacement exceeds 40% of the total stroke (relative to init_qpos).
        if hasattr(self, 'init_qpos'):
            delta_qpos = torch.abs(qpos[:, 0] - self.init_qpos[:, 0])
            is_toggled = delta_qpos > (stroke * 0.4)
        else:
            is_toggled = torch.abs(qpos[:, 0]) > (stroke * 0.2)
        
        is_robot_static = self.agent.is_static(threshold=0.2)
        success = is_toggled  # is_robot_static intentionally omitted
        
        return {
            "success": success,
            "is_toggled": is_toggled,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # TODO: implement the reward function.
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

@register_env("lid_opening", max_episode_steps=500)
class LidOpeningEnv(GraspPartEnv):
    """
    Lid-opening env.
    
    Inherits GraspPartEnv; specialised for lid-opening tasks.
    Adds tunable robot drive parameters and auto-discovery of joint + gripper info.
    
    Features:
    - Auto-detect + log every joint on the target object.
    - Auto-detect + log left/right gripper finger positions.
    - Expose stiffness / damping / force_limit knobs.
    """
    
    def __init__(self, 
                 *args,
                 robot_uids="panda_wristcam",
                 robot_init_qpos_noise=0.02,
                 object_name="bottle",
                 part_name="lid",
                 robot_stiffness=1000.0,
                 robot_damping=100.0,
                 robot_force_limit=100.0,
                 camera_eye=None,
                 camera_target=None,
                 **kwargs):
        """
        Initialise the lid-opening env.
        
        Args:
            *args: positional args forwarded to the parent.
            robot_uids (str): robot model, default "panda_wristcam".
            robot_init_qpos_noise (float): initial qpos noise std.
            object_name (str): asset folder for the target object.
            part_name (str): name of the part to manipulate.
            robot_stiffness (float): drive stiffness.
            robot_damping (float): drive damping.
            robot_force_limit (float): drive force limit.
            camera_eye (list): custom [x, y, z] camera eye; None for default.
            camera_target (list): custom [x, y, z] camera target; None for default.
            **kwargs: extra kwargs forwarded to the parent.
        """
        # Cache drive parameters.
        self.robot_stiffness = robot_stiffness
        self.robot_damping = robot_damping
        self.robot_force_limit = robot_force_limit
        
        # Cache the optional camera overrides BEFORE super().__init__.
        self._custom_camera_eye = camera_eye
        self._custom_camera_target = camera_target
        
        # Forward to parent.
        super().__init__(*args, 
                        robot_uids=robot_uids,
                        robot_init_qpos_noise=robot_init_qpos_noise,
                        object_name=object_name,
                        part_name=part_name,
                        **kwargs)
        
        # Will hold detected joint + gripper info after build.
        self.object_joints_info = {}
        self.gripper_info = {}
        self.verbose = False
    
    @property
    def _default_human_render_camera_configs(self):
        """
        Return the render-camera config, honouring optional self._custom_camera_eye/_target.
        
        Returns:
            CameraConfig: render-camera config.
        """
        # Prefer the explicit overrides when present.
        if self._custom_camera_eye is not None:
            eye = self._custom_camera_eye
        else:
            eye = self.human_cam_eye_pos
        
        if self._custom_camera_target is not None:
            target = self._custom_camera_target
        else:
            target = self.human_cam_target_pos
        
        pose = sapien_utils.look_at(eye=eye, target=target)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    def _load_agent(self, options: dict):
        """
        Load the robot agent and apply the configured drive parameters.
        
        Args:
            options (dict): agent-load options.
        """
        # Load the agent via the parent implementation.
        super()._load_agent(options)
        
        # Push drive parameters onto every active joint.
        try:
            robot = self.agent.robot
            joints = robot.get_active_joints()
            
            configured_count = 0
            for joint in joints:
                try:
                    # Note: the API is set_drive_properties (plural).
                    joint.set_drive_properties(
                        stiffness=self.robot_stiffness,
                        damping=self.robot_damping,
                        force_limit=self.robot_force_limit
                    )
                    configured_count += 1
                except AttributeError:
                    # Fall back to the older API on environments that lack it.
                    try:
                        joint.set_drive_property(
                            stiffness=self.robot_stiffness,
                            damping=self.robot_damping
                        )
                        configured_count += 1
                    except Exception:
                        pass
            if self.verbose:    
                print(f"OK: drive parameters set on {configured_count}/{len(joints)} joints:")
                print(f"  stiffness: {self.robot_stiffness}")
                print(f"  damping: {self.robot_damping}")
                print(f"  force_limit: {self.robot_force_limit}")
            
        except Exception as e:
            print(f"WARN: failed to set drive parameters: {e}")
    
    def _after_reconfigure(self, options: dict):
        """
        Post-reconfigure hook: discover joint info + gripper info.
        
        Args:
            options (dict): reconfiguration options.
        """
        # Forward to parent if implemented.
        if hasattr(super(), '_after_reconfigure'):
            super()._after_reconfigure(options)
        
        # Discover + log target-object joints.
        self._detect_object_joints()
        
        # Discover + log gripper finger positions.
        self._detect_gripper_info()
    
    def _read_urdf_joint_axis(self, model_id: str, joint_name: str) -> np.ndarray:
        """
        Read a joint axis vector directly from the URDF.
        
        Args:
            model_id: asset folder name.
            joint_name: name of the joint.
        
        Returns:
            np.ndarray axis in the joint's local frame, or None on failure.
        """
        import os
        from pathlib import Path
        
        # Asset root: env var MS_ASSET_DIR overrides the default.
        asset_dir = os.environ.get("MS_ASSET_DIR", "./assets")
        urdf_path = Path(asset_dir) / model_id / 'mobility.urdf'
        
        if not urdf_path.exists():
            return None
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(urdf_path))
            root = tree.getroot()
            
            for joint_elem in root.findall('joint'):
                if joint_elem.get('name') == joint_name:
                    axis_elem = joint_elem.find('axis')
                    if axis_elem is not None and axis_elem.get('xyz'):
                        axis = np.fromstring(axis_elem.get('xyz'), sep=' ')
                        return axis
        except Exception:
            pass
        
        return None
    
    def _detect_object_joints(self):
        """Detect and print every joint on the target object."""
        try:
            obj = self.target_object
            if obj is None:
                print("WARN: target object not found")
                return
            
            joints = obj.get_joints()
            self.object_joints_info = {}
            
            if self.verbose:    
                print(f"\n=== Target object joint info ===")
                print(f"object: {obj.get_name()}")
                print(f"joints: {len(joints)}")
            
            for i, joint in enumerate(joints):
                joint_name = joint.get_name()
                joint_type = joint.get_type()
                
                # Joint pose.
                try:
                    joint_pose = joint.get_global_pose()
                    joint_pos = np.asarray(joint_pose.p).ravel()[:3]
                    
                    # Rotation matrix from the joint pose.
                    full_tf = np.asarray(joint_pose.to_transformation_matrix())
                    if full_tf.ndim == 3 and full_tf.shape[0] == 1:
                        full_tf = full_tf[0]
                    rotation_matrix = full_tf[:3, :3]
                    
                    # Pull the joint axis from the URDF (default: X).
                    local_axis = np.array([1.0, 0.0, 0.0], dtype=float)
                    
                    # Try reading the actual axis from URDF.
                    try:
                        urdf_axis = self._read_urdf_joint_axis(self.object_name, joint_name)
                        if urdf_axis is not None:
                            local_axis = np.asarray(urdf_axis).ravel()[:3].astype(float)
                            local_axis = local_axis / (np.linalg.norm(local_axis) + 1e-12)
                    except Exception:
                        pass
                    
                    # Transform the local axis to the world frame.
                    joint_axis = rotation_matrix @ local_axis
                    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-12)
                    
                    self.object_joints_info[joint_name] = {
                        'index': i,
                        'type': joint_type,
                        'position': joint_pos,
                        'axis': joint_axis,
                        'joint_object': joint
                    }
                    if self.verbose:    
                        print(f"  [{i}] {joint_name}")
                        print(f"      type: {joint_type}")
                        print(f"      position: {np.round(joint_pos, 4)}")
                        print(f"      axis: {np.round(joint_axis, 4)}")
                    
                except Exception as e:
                    print(f"  [{i}] {joint_name} (type: {joint_type}) — couldn't read details: {e}")
                    self.object_joints_info[joint_name] = {
                        'index': i,
                        'type': joint_type,
                        'joint_object': joint
                    }
            
            print(f"=" * 50)
            
        except Exception as e:
            print(f"FAILED to detect joints: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_gripper_info(self):
        """Detect and print left/right gripper finger info."""
        try:
            robot = self.agent.robot
            all_links = robot.get_links()
            if self.verbose:    
                print(f"\n=== Gripper finger info ===")
                print(f"robot: {robot.get_name()}")
                print(f"links: {len(all_links)}")
            
            left_finger_pos = None
            right_finger_pos = None
            left_finger_link = None
            right_finger_link = None
            
            # Layered matching strategy.
            # Pass 1: exact name match.
            exact_patterns = {
                'left': ['panda_leftfinger', 'gripper_left_finger', 'left_finger_tip', 'leftfinger'],
                'right': ['panda_rightfinger', 'gripper_right_finger', 'right_finger_tip', 'rightfinger']
            }
            
            for link in all_links:
                link_name = link.get_name().lower()
                
                # Match against left-finger patterns.
                for pattern in exact_patterns['left']:
                    if link_name == pattern:
                        try:
                            pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                            left_finger_pos = np.asarray(pose.p).ravel()[:3]
                            left_finger_link = link
                            break
                        except Exception:
                            pass
                
                # Match against right-finger patterns.
                for pattern in exact_patterns['right']:
                    if link_name == pattern:
                        try:
                            pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                            right_finger_pos = np.asarray(pose.p).ravel()[:3]
                            right_finger_link = link
                            break
                        except Exception:
                            pass
            
            # Pass 2: keyword match.
            if left_finger_pos is None or right_finger_pos is None:
                for link in all_links:
                    link_name = link.get_name().lower()
                    
                    if 'finger' in link_name:
                        try:
                            pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                            pos = np.asarray(pose.p).ravel()[:3]
                            
                            if left_finger_pos is None and ('left' in link_name or 'l_' in link_name):
                                left_finger_pos = pos
                                left_finger_link = link
                            elif right_finger_pos is None and ('right' in link_name or 'r_' in link_name):
                                right_finger_pos = pos
                                right_finger_link = link
                        except Exception:
                            pass
            
            # Persist detected gripper info.
            if left_finger_pos is not None and right_finger_pos is not None:
                self.gripper_info = {
                    'left_finger': {
                        'position': left_finger_pos,
                        'link': left_finger_link,
                        'name': left_finger_link.get_name()
                    },
                    'right_finger': {
                        'position': right_finger_pos,
                        'link': right_finger_link,
                        'name': right_finger_link.get_name()
                    },
                    'center': (left_finger_pos + right_finger_pos) / 2.0,
                    'distance': np.linalg.norm(left_finger_pos - right_finger_pos)
                }
                if verbose:
                    print(f"\nOK: detected gripper:")
                    print(f"  left finger: {left_finger_link.get_name()}")
                    print(f"    position: {np.round(left_finger_pos, 4)}")
                    print(f"  right finger: {right_finger_link.get_name()}")
                    print(f"    position: {np.round(right_finger_pos, 4)}")
                    print(f"  gripper centre: {np.round(self.gripper_info['center'], 4)}")
                    print(f"  gripper span: {self.gripper_info['distance']:.4f}")
                
            else:
                
                print(f"WARN: could not detect both fingers")
                if left_finger_pos is not None:
                    print(f"  left only: {left_finger_link.get_name()}")
                if right_finger_pos is not None:
                    print(f"  right only: {right_finger_link.get_name()}")
            
            print(f"=" * 50)
            
        except Exception as e:
            print(f"FAILED to detect gripper: {e}")
            import traceback
            traceback.print_exc()

@register_env("stack_pyramid", max_episode_steps=250)
class StackPyramidEnv(EvalDREnvMixin, BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    agent: Union[Panda]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        eye = np.array([0.6, 0.7, 0.6], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        eye, target = self._maybe_jitter_camera(eye, target)
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [CameraConfig("base_camera", pose, 224, 224, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.2]),
        )
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="cubeC",
            initial_pose=sapien.Pose(p=[-1, 0, 0.2]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            xy = xyz[:, :2]
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02]))
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

            # Cube A
            xyz[:, :2] = cubeA_xy

            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )

            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube B
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube C
            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(2 * self.cube_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2]) > 0.02
            if top_or_next == "top":
                is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            elif top_or_next == "next_to":
                is_cubeA_on_cubeB = xy_flag
            else:
                return NotImplementedError(
                    f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}"
                )

            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)
            return success.bool()

        success_A_B = evaluate_cube_distance(
            offset_AB, self.cubeA, self.cubeB, "next_to"
        )
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(
            success_A_B, torch.logical_and(success_C_B, success_C_A)
        )
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
            )
        return obs

@register_env("slide_along", max_episode_steps=500)
class SlideAlongEnv(GraspPartEnv):
    def __init__(
        self,
        *args,
        object_load_mode: str = "auto",
        success_delta_frac: float = 0.30,
        pull_dist: float | None = None,
        random_xy_low: tuple[float, float] = (0.35, -0.1),
        random_xy_high: tuple[float, float] = (0.45, 0.1),
        randomize_xy: bool = True,
        static_threshold: float = 0.20,
        **kwargs,
    ):
        # SlideAlong adds an object-load-mode parameter (see SlideAlongSceneBuilder).
        self.object_load_mode = object_load_mode

        # Success: an active joint moves past success_delta_frac of its stroke.
        self.success_delta_frac = float(success_delta_frac)

        # Optional: absolute pull_dist threshold (matches skill.slide_along's pull_dist).
        # - None: fall back to the success_delta_frac heuristic.
        # - positive: require max_delta > pull_dist.
        try:
            self.pull_dist = None if pull_dist is None else float(pull_dist)
        except Exception:
            self.pull_dist = None

        # Initial randomisation: jitter the object base pose's XY only; Z + rotation untouched.
        # Range adjustment:
        # - default code values: random_xy_low / random_xy_high
        # - or override via gym.make("slide_along", random_xy_low=(...), random_xy_high=(...), randomize_xy=True)
        self.randomize_xy = bool(randomize_xy)
        self.random_xy_low = tuple(float(x) for x in random_xy_low)
        self.random_xy_high = tuple(float(x) for x in random_xy_high)

        # Success: robot must be (nearly) static.
        self.static_threshold = float(static_threshold)

        # Camera config (scoped to this env):
        self._slide_sensor_cam_eye_pos = [-0.6, -0.6, 0.6]
        self._slide_sensor_cam_target_pos = [0, 0, 0.35]
        self._slide_human_cam_eye_pos = [-0.6, -0.6, 0.6]
        self._slide_human_cam_target_pos = [0, 0, 0.35]
        # The GUI viewer mirrors the human-render camera by default.
        self._slide_viewer_cam_eye_pos = list(self._slide_human_cam_eye_pos)
        self._slide_viewer_cam_target_pos = list(self._slide_human_cam_target_pos)

        super().__init__(*args, **kwargs)

        # Debug/docs only: also write the camera fields onto self so they show up in repr.
        self.sensor_cam_eye_pos = list(self._slide_sensor_cam_eye_pos)
        self.sensor_cam_target_pos = list(self._slide_sensor_cam_target_pos)
        self.human_cam_eye_pos = list(self._slide_human_cam_eye_pos)
        self.human_cam_target_pos = list(self._slide_human_cam_target_pos)

    @property
    def _default_sensor_configs(self):
        eye = np.array(self._slide_sensor_cam_eye_pos, dtype=np.float32)
        target = np.array(self._slide_sensor_cam_target_pos, dtype=np.float32)
        eye, target = self._maybe_jitter_camera(eye, target)
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self._slide_human_cam_eye_pos, target=self._slide_human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    @property
    def _default_viewer_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self._slide_viewer_cam_eye_pos, target=self._slide_viewer_cam_target_pos
        )
        return CameraConfig("viewer", pose, 512, 512, 1, 0.01, 100)

    # ---------------------------------------------------------------------
    # Episode lifecycle
    # ---------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # XY-only randomisation: after scene_builder.initialize, overwrite the object's XY.
        # That keeps the scene-builder's mode logic intact (e.g. large mode's table-snap z).
        if getattr(self, "randomize_xy", False) and hasattr(self, "target_object"):
            try:
                b = int(len(env_idx))
                xy = randomization.uniform(
                    list(self.random_xy_low),
                    list(self.random_xy_high),
                    size=(b, 2),
                )

                # Read current pose; only XY changes.
                pose = (
                    self.target_object.pose
                    if hasattr(self.target_object, "pose")
                    else self.target_object.get_pose()
                )
                p = pose.p
                q = pose.q

                # batched torch pose
                if hasattr(p, "shape") and len(getattr(p, "shape", [])) == 2:
                    p_new = p.clone()
                    # env_idx may be a torch tensor.
                    idx = env_idx.to(dtype=torch.long) if hasattr(env_idx, "to") else env_idx
                    p_new[idx, 0:2] = xy.to(device=p_new.device, dtype=p_new.dtype)
                    self.target_object.set_pose(Pose.create_from_pq(p_new, q))
                else:
                    # Single-env case (typical for non-vectorised recording).
                    p_np = p.cpu().numpy() if hasattr(p, "cpu") else np.array(p)
                    q_np = q.cpu().numpy() if hasattr(q, "cpu") else np.array(q)
                    xy_np = xy[0].cpu().numpy() if hasattr(xy, "cpu") else np.array(xy)[0]
                    p_np = np.array(p_np, dtype=np.float32).reshape(-1)
                    p_np[0] = float(xy_np[0])
                    p_np[1] = float(xy_np[1])
                    self.target_object.set_pose(Pose.create_from_pq(p_np, q_np))
            except Exception:
                # Randomisation must not break episode init; fall back silently on failure.
                pass

        # Cache initial qpos so evaluate() can compute the delta.
        if not hasattr(self, "init_qpos") or self.init_qpos.shape[0] != self.num_envs:
            current_qpos = self.target_object.get_qpos()
            self.init_qpos = torch.zeros_like(current_qpos)
        self.init_qpos[env_idx] = self.target_object.get_qpos()[env_idx]

    def _load_scene(self, options: dict):
        """slide_along-specific scene load — uses SlideAlongSceneBuilder."""
        print(self.object_name)

        self.scene_builder = SlideAlongSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
            object_load_mode=self.object_load_mode,
        )
        self.scene_builder.build()

        # Pick up the constructed target object.
        self.target_object = self.scene_builder.target_object

        # Spawn a green sphere as a goal-position marker.
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh / 4,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def evaluate(self):
        """Evaluate success: drawer joint moved enough + robot is static.

        - When pull_dist is configured: require max_delta > pull_dist (absolute).
        - Otherwise: require max_delta > success_delta_frac * stroke (relative).
        """
        qpos = self.target_object.get_qpos()  # shape (N, dof)
        joints = self.target_object.get_active_joints()
        if not joints:
            return {
                "success": torch.tensor([False], device=self.device),
                "drawer_delta": torch.tensor([0.0], device=self.device),
                "is_drawer_moved": torch.tensor([False], device=self.device),
                "is_robot_static": torch.tensor([True], device=self.device),
                "moved_joint_idx": torch.tensor(
                    [-1], device=self.device, dtype=torch.long
                ),
            }

        # Compute each active joint's stroke.
        strokes = []
        for j in joints:
            lim = j.get_limits()
            jmin = float(lim[0][0])
            jmax = float(lim[0][1])
            strokes.append(jmax - jmin)
        strokes = torch.tensor(strokes, device=self.device, dtype=qpos.dtype)  # (dof,)

        # Per-joint displacement vs init_qpos.
        if (
            hasattr(self, "init_qpos")
            and hasattr(self.init_qpos, "shape")
            and self.init_qpos.shape == qpos.shape
        ):
            delta_all = torch.abs(qpos - self.init_qpos)  # (N, dof)
        else:
            delta_all = torch.abs(qpos)  # (N, dof)

        # Skip near-zero-stroke joints to avoid noise-driven false positives.
        stroke_eps = 1e-6
        valid = strokes > stroke_eps  # (dof,)
        if bool(torch.any(valid).item()):
            delta_all = delta_all * valid[None, :].to(delta_all.dtype)

        # Auto-detect the most-displaced active joint.
        moved_joint_idx = torch.argmax(delta_all, dim=1).to(torch.long)  # (N,)
        max_delta = torch.gather(delta_all, 1, moved_joint_idx[:, None]).squeeze(1)  # (N,)

        # Threshold: prefer the absolute pull_dist; fall back to stroke fraction.
        if getattr(self, "pull_dist", None) is not None:
            thr = float(abs(self.pull_dist))
            sel_threshold = torch.full_like(max_delta, thr)
        else:
            frac = max(0.0, min(1.0, float(self.success_delta_frac)))
            thresholds = strokes * float(frac)  # (dof,)
            sel_threshold = torch.gather(
                thresholds[None, :], 1, moved_joint_idx[:, None]
            ).squeeze(1)  # (N,)

        # Success is decided solely by the auto-detected joint's displacement.
        # Extra epsilon so all-zero deltas don't trip a false positive.
        delta_eps = 1e-6
        is_drawer_moved = (max_delta > delta_eps) & (max_delta > sel_threshold)
        is_robot_static = self.agent.is_static(threshold=float(self.static_threshold))
        success = is_drawer_moved & is_robot_static

        return {
            "success": success,
            "drawer_delta": max_delta,
            "drawer_threshold": sel_threshold,
            "is_drawer_moved": is_drawer_moved,
            "is_robot_static": is_robot_static,
            "moved_joint_idx": moved_joint_idx,
        }

@register_env("plug_charger", max_episode_steps=200)
class PlugChargerEnv(EvalDREnvMixin, BaseEnv):
    """
    **Task Description:**
    The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

    **Randomizations:**
    - The charger position is randomized on the XY plane on top of the table. The rotation is also randomized
    - The receptacle position is randomized on the XY plane and the rotation is also randomized. Note that the human render camera has its pose
    fixed relative to the receptacle.

    **Success Conditions:**
    - The charger is inserted into the receptacle
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlugCharger-v1_rt.mp4"

    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        eye = np.array([0.3, 0.4, 0.1], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)
        eye, target = self._maybe_jitter_camera(eye, target)
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize agent
            if self.agent.uid == "panda_wristcam":
                qpos = torch.tensor(
                    [
                        0.0,
                        np.pi / 8,
                        0,
                        -np.pi * 5 / 8,
                        0,
                        np.pi * 3 / 4,
                        np.pi / 4,
                        0.04,
                        0.04,
                    ]
                )
                qpos = (
                    torch.normal(
                        0,
                        self.robot_init_qpos_noise,
                        (b, len(qpos)),
                        device=self.device,
                    )
                    + qpos
                )
                qpos[:, -2:] = 0.04
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            xy = randomization.uniform(
                [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._base_size[2]
            ori = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
            )
            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.1
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                receptacle_pose=self.receptacle.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
            )
        return obs

@register_env("multi_skill", max_episode_steps=500)
class MultiSkillEnv(GraspPartEnv):
    """Generic env for composing motion-planning skills into a task graph.

    The env loads a single articulated object through GraspPartSceneBuilder
    (multi-object scenes are future work — the chain schema already accepts
    a per-step ``object`` field so the upgrade path is preserved). The
    caller — today a human via record.py / a YAML task graph, tomorrow an
    AI planner — supplies a ``skill_chain`` and an optional
    ``goal_predicate``; the env stays agnostic to skill semantics and only
    exposes universal state predicates the caller composes against.

    Skill chain shape (list of dicts)::

        [
            {"skill": "grasp_part", "object": "3398", "part_name": "cap"},
            {"skill": "rotate",      "object": "3398", "part_name": "cap", "angle": 60},
            ...
        ]

    Validation: on construction the chain is checked against
    ``assets/<object_name>/capabilities.json`` via
    :func:`core.skill_registry.validate_task_graph`. Issues are warn-only
    because auto-derived capabilities are deliberately conservative.

    Success: when ``goal_predicate`` is supplied, ``evaluate()['success']`` is
    the predicate's return value; otherwise success defers to the parent
    ``GraspPartEnv.evaluate()``.
    """

    def __init__(
        self,
        *args,
        skill_chain=None,
        goal_predicate=None,
        validate_chain: bool = True,
        **kwargs,
    ):
        self.skill_chain = list(skill_chain or [])
        self.goal_predicate = goal_predicate
        # stage_predicates is a list of {"name": str, "predicate": Callable}.
        # The eval helper compiles task-graph ``stages: [...]`` into this
        # shape and assigns it after gym.make. Default to empty so envs
        # constructed without a task graph stay no-op.
        self.stage_predicates = []
        self._validate_chain_on_init = bool(validate_chain)
        super().__init__(*args, **kwargs)
        if self._validate_chain_on_init and self.skill_chain:
            self._validate_skill_chain()

    # ------------------------------------------------------------------ #
    # Construction-time validation                                       #
    # ------------------------------------------------------------------ #

    def _validate_skill_chain(self) -> None:
        """Warn (don't raise) when the chain violates registry rules.

        Lazily imports :mod:`core.skill` to ensure the ``@register_skill``
        decorators have run; otherwise the registry would be empty when
        ``gym.make("multi_skill", ...)`` is the first thing the caller does.
        """
        import core.skill  # noqa: F401  populate SKILL_REGISTRY
        from .skill_registry import validate_task_graph
        capabilities = self._load_capabilities()
        issues = validate_task_graph(self.skill_chain, capabilities)
        for msg in issues:
            print(f"[multi_skill] WARN: {msg}")

    def _load_capabilities(self):
        path = Path(__file__).parent.parent / "assets" / self.object_name / "capabilities.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[multi_skill] WARN: failed to read {path}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Universal state predicates                                         #
    # ------------------------------------------------------------------ #
    # Predicates intentionally take object identifiers as strings (joint   #
    # names, part names) so a YAML success spec can name them without      #
    # holding live SAPIEN handles. A goal_predicate composes these.        #

    def grasped(self, part_name=None) -> bool:
        """Whether the gripper currently holds a specific (or any) part.

        Combines two signals so that lying about engagement is hard:

        1. ``self._engaged_part`` (set by interaction-phase skills, cleared
           by release skills and on episode reset). When ``part_name`` is
           supplied it must match.
        2. The gripper-angle heuristic — useful for catching cases where
           the engagement record is stale because the gripper opened
           without an explicit ``release_gripper`` call.
        """
        engaged = getattr(self, "_engaged_part", None)
        if engaged is None:
            return False
        if part_name is not None and part_name != engaged:
            return False
        qpos = self.agent.robot.get_qpos()
        gripper_total = abs(qpos[0, -2]).item()
        return 0.01 < gripper_total < 0.03

    def joint_value(self, joint_name: str):
        """Return the qpos value of a named articulation joint, or None."""
        obj = getattr(self, "target_object", None)
        if obj is None:
            return None
        active_joints = obj.get_active_joints()
        qpos = obj.get_qpos()
        for idx, j in enumerate(active_joints):
            if j.get_name() == joint_name:
                return float(qpos[0, idx].item())
        return None

    def lifted(self, min_height: float, ref: str = "tcp") -> bool:
        """Whether the TCP (or target object center) is at least ``min_height`` above the table."""
        if ref == "tcp":
            z = self.agent.tcp.pose.p[0, 2].item()
        elif ref == "object":
            obj = getattr(self, "target_object", None)
            if obj is None:
                return False
            z = obj.pose.p[0, 2].item()
        else:
            raise ValueError(f"lifted: ref must be 'tcp' or 'object', got {ref!r}")
        return z >= float(min_height)

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(self):
        base = super().evaluate()
        # Evaluate every stage predicate independently. The result is a
        # boolean list (``stage_flags``) plus a flat ``stage_<name>`` key
        # per stage so the caller can read it like any other info field.
        # We deliberately treat stages as independent — no "must have
        # reached i-1 to count i" coupling — so partial progress is
        # visible even when a policy skips ahead.
        if self.stage_predicates:
            obs_extra = self._get_obs_extra({})
            flags = []
            for stage in self.stage_predicates:
                try:
                    flag = bool(stage["predicate"](self, obs_extra))
                except Exception as exc:
                    print(f"[multi_skill] stage {stage['name']!r} predicate raised: {exc}")
                    flag = False
                flags.append(flag)
                base[f"stage_{stage['name']}"] = torch.tensor([flag], device=self.device)
            base["stage_flags"] = torch.tensor([flags], device=self.device)

        if self.goal_predicate is None:
            return base
        try:
            obs_extra = self._get_obs_extra({})
            success = bool(self.goal_predicate(self, obs_extra))
        except Exception as exc:
            print(f"[multi_skill] goal_predicate raised: {exc}")
            success = False
        base["success"] = torch.tensor([success], device=self.device)
        return base

@register_env("draw_triangle", max_episode_steps=300)
class DrawTriangleEnv(EvalDREnvMixin, BaseEnv):
    r"""
    **Task Description:**
    Instantiates a table with a white canvas on it and a goal triangle with an outline. A robot with a stick is to draw the triangle with a red line.

    **Randomizations:**
    - the goal triangle's position on the xy-plane is randomized
    - the goal triangle's z-rotation is randomized in range [0, 2 $\pi$]

    **Success Conditions:**
    - the drawn points by the robot are within a euclidean distance of 0.05m with points on the goal triangle
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/DrawTriangle-v1_rt.mp4"

    MAX_DOTS = 300
    """
    The total "ink" available to use and draw with before you need to call env.reset. NOTE that on GPU simulation it is not recommended to have a very high value for this as it can slow down rendering
    when too many objects are being rendered in many scenes.
    """
    DOT_THICKNESS = 0.003
    """thickness of the paint drawn on to the canvas"""
    CANVAS_THICKNESS = 0.02
    """How thick the canvas on the table is"""
    BRUSH_RADIUS = 0.01
    """The brushes radius"""
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    """The colors of the brushes. If there is more than one color, each parallel environment will have a randomly sampled color."""
    THRESHOLD = 0.025

    SUPPORTED_REWARD_MODES = ["sparse"]

    SUPPORTED_ROBOTS: ["panda_stick"]  # type: ignore
    agent: PandaStick

    def __init__(
        self,
        *args,
        robot_uids="panda_stick",
        **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # we set contact_offset to a small value as we are not expecting to make any contacts really apart from the brush hitting the canvas too hard.
        # We set solver iterations very low as this environment is not doing a ton of manipulation (the brush is attached to the robot after all)
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        eye = np.array([0.3, 0, 0.8], dtype=np.float32)
        target = np.array([0, 0, 0.1], dtype=np.float32)
        eye, target = self._maybe_jitter_camera(eye, target)
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=320,
                height=240,
                fov=1.2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=960,
            fov=1.2,
            near=0.01,
            far=100,
        )

    def _load_scene(self, options: dict):

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        def create_goal_triangle(name="tri", base_color=None):

            box1_half_w = 0.3 / 2
            box1_half_h = 0.01 / 2
            half_thickness = 0.001 / 2

            radius = (box1_half_w) / math.sqrt(3)

            theta = np.pi / 2

            # define centers and compute verticies, might need to adjust how centers are calculated or add a theta arg for variation
            c1 = np.array([radius * math.cos(theta), radius * math.sin(theta), 0.01])
            c2 = np.array(
                [
                    radius * math.cos(theta + (2 * np.pi / 3)),
                    radius * math.sin(theta + (2 * np.pi / 3)),
                    0.01,
                ]
            )
            c3 = np.array(
                [
                    radius * math.cos((theta + (4 * np.pi / 3))),
                    radius * math.sin(theta + (4 * np.pi / 3)),
                    0.01,
                ]
            )
            self.original_verts = np.array(
                [(c1 + c3) - c2, (c1 + c2) - c3, (c2 + c3) - c1]
            )

            builder = self.scene.create_actor_builder()
            first_block_pose = sapien.Pose(
                list(c1), euler2quat(0, 0, theta - (np.pi / 2))
            )
            first_block_size = [box1_half_w, box1_half_h, half_thickness]
            builder.add_box_visual(
                pose=first_block_pose,
                half_size=first_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )

            second_block_pose = sapien.Pose(
                list(c2), euler2quat(0, 0, theta - (5 * np.pi / 6))
            )
            second_block_size = [box1_half_w, box1_half_h, half_thickness]
            # builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(
                pose=second_block_pose,
                half_size=second_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )

            third_block_pose = sapien.Pose(
                list(c3), euler2quat(0, 0, theta - (np.pi / 6))
            )
            third_block_size = [box1_half_w, box1_half_h, half_thickness]
            # builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(
                pose=third_block_pose,
                half_size=third_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            return builder.build_kinematic(name=name)

        # build a white canvas on the table
        self.canvas = self.scene.create_actor_builder()
        self.canvas.add_box_visual(
            half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
            material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
        )
        self.canvas.add_box_collision(
            half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
        )
        self.canvas.initial_pose = sapien.Pose(p=[-0.1, 0, self.CANVAS_THICKNESS / 2])
        self.canvas = self.canvas.build_static(name="canvas")

        self.dots = []
        color_choices = torch.randint(0, len(self.BRUSH_COLORS), (self.num_envs,))
        for i in range(self.MAX_DOTS):
            actors = []
            if len(self.BRUSH_COLORS) > 1:
                for env_idx in range(self.num_envs):
                    builder = self.scene.create_actor_builder()
                    builder.add_cylinder_visual(
                        radius=self.BRUSH_RADIUS,
                        half_length=self.DOT_THICKNESS / 2,
                        material=sapien.render.RenderMaterial(
                            base_color=self.BRUSH_COLORS[color_choices[env_idx]]
                        ),
                    )
                    builder.set_scene_idxs([env_idx])
                    builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                    actor = builder.build_kinematic(name=f"dot_{i}_{env_idx}")
                    actors.append(actor)
                self.dots.append(Actor.merge(actors))
            else:
                builder = self.scene.create_actor_builder()
                builder.add_cylinder_visual(
                    radius=self.BRUSH_RADIUS,
                    half_length=self.DOT_THICKNESS / 2,
                    material=sapien.render.RenderMaterial(
                        base_color=self.BRUSH_COLORS[0]
                    ),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                actor = builder.build_kinematic(name=f"dot_{i}")
                self.dots.append(actor)
        self.goal_tri = create_goal_triangle(
            name="goal_tri",
            base_color=np.array([10, 10, 10, 255]) / 255,
        )
        self.dots_dist = torch.ones((self.num_envs, 300), device=self.device) * -1
        self.ref_dist = torch.zeros((self.num_envs, 153), device=self.device).to(bool)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.draw_step = 0
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            target_pos = torch.zeros((b, 3))

            target_pos[:, :2] = torch.rand((b, 2)) * 0.02 - 0.1
            target_pos[:, -1] = 0.01
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            mats = quaternion_to_matrix(qs).to(self.device)
            self.goal_tri.set_pose(Pose.create_from_pq(p=target_pos, q=qs))

            if hasattr(self, "vertices"):
                self.vertices[env_idx] = torch.from_numpy(
                    np.tile(self.original_verts, (b, 1, 1))
                ).to(
                    self.device
                )  # b, 3, 3
            else:
                self.vertices = torch.from_numpy(
                    np.tile(self.original_verts, (b, 1, 1))
                ).to(self.device)

            self.vertices[env_idx] = (
                mats.double() @ self.vertices[env_idx].transpose(-1, -2).double()
            ).transpose(
                -1, -2
            )  # apply rotation matrix
            self.vertices[env_idx] += target_pos.unsqueeze(1)

            self.triangles = self.generate_triangle_with_points(
                50, self.vertices[:, :, :-1]
            )

            self.dots_dist[env_idx] = torch.ones((b, 300)) * -1
            self.ref_dist[env_idx] = torch.zeros((b, 153)).to(bool)

            for dot in self.dots:
                # initially spawn dots in the table so they aren't seen
                dot.set_pose(
                    sapien.Pose(
                        p=[0, 0, -self.DOT_THICKNESS], q=euler2quat(0, np.pi / 2, 0)
                    )
                )

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        # This is the actual, GPU parallelized, drawing code.
        # This is not real drawing but seeks to mimic drawing by placing dots on the canvas whenever the robot is close enough to the canvas surface
        # We do not actually check if the robot contacts the table (although that is possible) and instead use a fast method to check.
        # We add a 0.005 meter of leeway to make it easier for the robot to get close to the canvas and start drawing instead of having to be super close to the table.
        robot_touching_table = (
            self.agent.tcp.pose.p[:, 2]
            < self.CANVAS_THICKNESS + self.DOT_THICKNESS + 0.005
        )
        robot_brush_pos = torch.zeros((self.num_envs, 3), device=self.device)
        robot_brush_pos[:, 2] = -self.DOT_THICKNESS
        robot_brush_pos[robot_touching_table, :2] = self.agent.tcp.pose.p[
            robot_touching_table, :2
        ]
        robot_brush_pos[robot_touching_table, 2] = (
            self.DOT_THICKNESS / 2 + self.CANVAS_THICKNESS
        )
        # move the next unused dot to the robot's brush position. All unused dots are initialized inside the table so they aren't visible
        new_dot_pos = Pose.create_from_pq(robot_brush_pos, euler2quat(0, np.pi / 2, 0))
        self.dots[self.draw_step].set_pose(new_dot_pos)

        self.draw_step += 1

        # on GPU sim we have to call _gpu_apply_all() to apply the changes we make to object poses.
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        out = self.success_check()
        return {"success": out}

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                goal_pose=self.goal_tri.pose.raw_pose.reshape(self.num_envs, -1),
                tcp_to_verts_pos=(
                    self.vertices - self.agent.tcp.pose.p.unsqueeze(1)
                ).reshape(self.num_envs, -1),
                goal_pos=self.goal_tri.pose.p.reshape(self.num_envs, -1),
                vertices=self.vertices.reshape(self.num_envs, -1),
            )

        return obs

    def generate_triangle_with_points(self, n, vertices):
        # interpolates a triangle from vertices to have n points. total
        batch_size = vertices.shape[0]

        all_points = []

        for i in range(vertices.shape[1]):
            start_vertex = vertices[:, i, :]
            end_vertex = vertices[:, (i + 1) % vertices.shape[1], :]
            t = torch.linspace(0, 1, n + 2, device=vertices.device)[:-1]
            t = t.view(1, -1, 1).repeat(batch_size, 1, 2)
            intermediate_points = (
                start_vertex.unsqueeze(1) * (1 - t) + end_vertex.unsqueeze(1) * t
            )
            all_points.append(intermediate_points)
        all_points = torch.cat(all_points, dim=1)

        return all_points

    def success_check(self):

        if self.draw_step > 0:
            current_dot = self.dots[self.draw_step - 1].pose.p.reshape(
                self.num_envs, 1, 3
            )  # b,3
            z_mask = current_dot[:, :, 2] < 0

            # distance for newly added pointed to all ref points
            dist = (
                torch.sqrt(
                    torch.sum(
                        (current_dot[:, :, None, :2] - self.triangles[:, None, :, :2])
                        ** 2,
                        dim=-1,
                    )
                )
                < self.THRESHOLD
            )

            # if a reference point has a draw point near it
            self.ref_dist = torch.logical_or(
                self.ref_dist, (1 - z_mask.int()) * dist.reshape((self.num_envs, 153))
            )

            # if current drawn point is close to a reference point. -1 if the drawn point hasn't actually been drawn yet
            self.dots_dist[:, self.draw_step - 1] = torch.where(
                z_mask, -1, torch.any(dist, dim=-1)
            ).reshape(
                self.num_envs,
            )

            mask = self.dots_dist > -1
            # for valid drawn points
            return torch.logical_and(
                torch.all(self.dots_dist[mask], dim=-1),
                torch.all(self.ref_dist, dim=-1),
            )
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

@register_env("rotate", max_episode_steps=500)
class RotateEnv(GraspPartEnv):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.sensor_cam_eye_pos = [-0.9, 0.0, 0.8]
        self.sensor_cam_target_pos = [0.1, 0.0, 0.3]
        self.human_cam_eye_pos = self.sensor_cam_eye_pos
        self.human_cam_target_pos = self.sensor_cam_target_pos

    @property
    def _camera_pose(self):
        return sapien_utils.look_at(
            eye=[-0.4, 0, 0.6],    # camera eye [x, y, z]
            target=[0.1, 0.0, 0.3]   # camera target [x, y, z]
        )

    @property
    def _default_sensor_configs(self):
        eye = np.array([-0.4, 0, 0.6], dtype=np.float32)
        target = np.array([0.1, 0.0, 0.3], dtype=np.float32)
        eye, target = self._maybe_jitter_camera(eye, target)
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera", self._camera_pose, 512, 512, 1, 0.01, 100)
    


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not hasattr(self, 'init_qpos') or self.init_qpos.shape[0] != self.num_envs:
            current_qpos = self.target_object.get_qpos()
            # self.init_qpos = torch.zeros_like(current_qpos)
            self.init_qpos = current_qpos.clone()

        self.init_qpos[env_idx] = self.target_object.get_qpos()[env_idx]

    def _load_scene(self, options: dict):
        """Load the scene for a rotation task."""
        self.scene_builder = RotateSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.target_object = self.scene_builder.target_object
        
        # Spawn a goal-position marker.
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh/4,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def evaluate(self):
        """Evaluate success: at least one revolute joint moved past its threshold."""
        # 1. Read joint state.
        qpos = self.target_object.get_qpos() # shape (N, dof)
        joints = self.target_object.get_active_joints()
        
        if not joints:
             return {
                "success": torch.tensor([False] * self.num_envs, device=self.device),
                "is_rotated": torch.tensor([False] * self.num_envs, device=self.device),
                "is_robot_static": torch.tensor([True] * self.num_envs, device=self.device),
            }

        # 2. Did any revolute joint move?
        is_rotated_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for i, joint in enumerate(joints):
            # joint.type may be a string or a list (e.g. ["continuous"]); handle both.
            # Empirically: joint.type == ["continuous"] is the form to compare against.
            j_type = joint.type
            if isinstance(j_type, list):
                j_type = j_type[0]
            
            # Only rotational joints (revolute / continuous) are interesting.
            if j_type not in ["revolute", "continuous"]:
                continue

            # Compute delta vs init_qpos.
            delta = torch.abs(qpos[:, i])
            if hasattr(self, 'init_qpos'):
                delta = torch.abs(qpos[:, i] - self.init_qpos[:, i])
            
            # Criteria:
            # absolute change > 0.1 rad (~5.7°) → ensure sensitivity.
            cond_absolute = delta > 0.3
            
            # Relative-stroke check for bounded joints.
            limits = joint.get_limits()
            joint_range = limits[0][1] - limits[0][0]
            cond_relative = torch.zeros_like(cond_absolute)
            
            # For bounded joints, also require the change exceeds a fraction of the stroke.
            if -1e3 < limits[0][0] and limits[0][1] < 1e3 and joint_range > 1e-3:
                cond_relative = delta > (joint_range * 0.2)
                
            is_rotated_any = is_rotated_any | cond_absolute | cond_relative

        # 3. Auxiliary status checks.
        is_robot_static = self.agent.is_static(threshold=0.2)
        
        # 4. Grasp check.
        is_grasped = torch.tensor([False] * self.num_envs, device=self.device)
        if hasattr(self.agent, 'is_grasping'):
            # Workaround for AttributeError: Articulation has no _bodies attr.
            # Must call is_grasping against individual Links, not the Articulation.
            for link in self.target_object.get_links():
                # Only count links that are actually being grasped (target-link filtering optional).
                # Lenient: any grasped link + rotation counts as success.
                
                if self.agent.is_grasping(link):
                    is_grasped = torch.tensor([True] * self.num_envs, device=self.device)
                    break
        
        # Final success = rotation criterion AND robot is grasping.
        success = is_rotated_any & is_grasped

        return {
            "success": success,
            "is_rotated": is_rotated_any,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped
        }

@register_env("door_env", max_episode_steps=500)
class DoorEnv(GraspPartEnv):

    def __init__(self,
                 *args,
                 robot_uids="panda_wristcam",
                 robot_init_qpos_noise=0.02,
                 object_name="door",
                 part_name="handle",
                 **kwargs):
        """
        Initialise the door env.

        Args:
            *args: positional args forwarded to the parent.
            robot_uids (str): robot model, default "panda_wristcam".
            robot_init_qpos_noise (float): initial qpos noise std, default 0.02.
            object_name (str): asset folder, default "door".
            part_name (str): part to grasp, default "handle".
            **kwargs: extra kwargs forwarded to the parent.
        """
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            object_name=object_name,
            part_name=part_name,
            **kwargs
        )

    def _load_scene(self, options: dict):
        """
        Build the door scene.

        Uses DoorSceneBuilder instead of GraspPartSceneBuilder.

        Args:
            options (dict): scene-load options.
        """
        # Build via DoorSceneBuilder.
        self.scene_builder = DoorSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Pick up the constructed target object.
        self.target_object = self.scene_builder.target_object

        # Spawn a green sphere as a goal-position marker.
        from mani_skill.utils.building import actors
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh/4,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

@register_env("take_out_and_grasp_part_into_box", max_episode_steps=800)
class TakeOutAndGraspPartIntoBoxEnv(GraspPartEnv):
    """
    Long-horizon task: remove a cube from the box, then grasp the target object and place it inside.

    Stages:
    - Stage 0: remove the cube from the box (place it outside on the table).
    - Stage 1: grasp the target object and put it inside the box.
    - Stage 2: task complete.
    """

    def __init__(
        self,
        *args,
        object_name="cabinet_01",
        part_name="handle",
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        # GraspPartEnv.__init__ handles object_name, part_name, validation, config loading
        super().__init__(
            *args,
            object_name=object_name,
            part_name=part_name,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("base_camera", pose, 224, 224, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        """Build the scene: load the target via GraspPartSceneBuilder + add box and cube."""
        from .scene import GraspPartSceneBuilder
        self.scene_builder = GraspPartSceneBuilder(
            self,
            object_name=self.object_name,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.target_object = self.scene_builder.target_object

        self.box = self._build_box()

        self.cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0.4, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )

    def _build_box(self):
        """Build a box with the opening facing up."""
        builder = self.scene.create_actor_builder()
        box_size = [0.15, 0.15, 0.06]
        wall_thickness = 0.005

        builder.add_box_collision(
            sapien.Pose([0, 0, -box_size[2] / 2]),
            [box_size[0] / 2, box_size[1] / 2, wall_thickness / 2]
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, -box_size[2] / 2]),
            [box_size[0] / 2, box_size[1] / 2, wall_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
        )

        walls = [
            (sapien.Pose([box_size[0] / 2, 0, 0]),  [wall_thickness / 2, box_size[1] / 2, box_size[2] / 2]),
            (sapien.Pose([-box_size[0] / 2, 0, 0]), [wall_thickness / 2, box_size[1] / 2, box_size[2] / 2]),
            (sapien.Pose([0, box_size[1] / 2, 0]),  [box_size[0] / 2, wall_thickness / 2, box_size[2] / 2]),
            (sapien.Pose([0, -box_size[1] / 2, 0]), [box_size[0] / 2, wall_thickness / 2, box_size[2] / 2]),
        ]
        for pose, half_size in walls:
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(
                pose, half_size,
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
            )

        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_static(name="box")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset per-episode state."""
        with torch.device(self.device):
            b = len(env_idx)
            super()._initialize_episode(env_idx, options)

            import random
            from utils.util import rotation_quaternion_z as _rot_z
            if random.random() < 0.6:
                xy = randomization.uniform(
                    low=torch.tensor([-0.2, -0.1]),
                    high=torch.tensor([0.2, 0.2]),
                    size=(b, 2)
                )
            else:
                xy = randomization.uniform(
                    low=torch.tensor([-0.3, -0.2]),
                    high=torch.tensor([0.35, -0.1]),
                    size=(b, 2)
                )
            pos = torch.zeros((b, 3), device=self.device)
            pos[:, :2] = xy.to(self.device)
            pos[:, 2] = 0.2
            angle = np.random.uniform(0, 60)
            quat = _rot_z(angle)
            if b > 0:
                self.target_object.set_pose(
                    Pose.create_from_pq(pos[0].cpu().numpy(), quat)
                )

            box_pos = torch.zeros((b, 3), device=self.device)
            box_pos[:, 0] = 0.03
            box_pos[:, 1] = -0.25
            box_pos[:, 2] = 0.05
            box_quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.box.set_pose(Pose.create_from_pq(box_pos, box_quat))

            cube_pos = box_pos.clone()
            cube_pos[:, 2] = box_pos[:, 2] + 0.03
            cube_quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_pos, cube_quat))

            # Initialise per-env task stage.
            if not hasattr(self, 'stage_id'):
                self.stage_id = torch.zeros(b, dtype=torch.long, device=self.device)
            else:
                self.stage_id[env_idx] = 0

    def _is_in_box(self, obj) -> torch.Tensor:
        """Whether the object is inside the box."""
        obj_pos = obj.pose.p
        box_pos = self.box.pose.p
        xy_in = (
            (torch.abs(obj_pos[:, 0] - box_pos[:, 0]) < 0.12) &
            (torch.abs(obj_pos[:, 1] - box_pos[:, 1]) < 0.12)
        )
        z_in = (
            (obj_pos[:, 2] > box_pos[:, 2] - 0.03) &
            (obj_pos[:, 2] < box_pos[:, 2] + 0.20)
        )
        return xy_in & z_in

    def _is_outside_box(self, obj) -> torch.Tensor:
        """Whether the object is clearly outside the box."""
        obj_pos = obj.pose.p
        box_pos = self.box.pose.p
        xy_outside = (
            (torch.abs(obj_pos[:, 0] - box_pos[:, 0]) > 0.15) |
            (torch.abs(obj_pos[:, 1] - box_pos[:, 1]) > 0.15)
        )
        return xy_outside

    def _is_grasping_target(self) -> torch.Tensor:
        """Whether the arm currently grasps the target object (per-link check)."""
        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for link in self.target_object.get_links():
            result = result | self.agent.is_grasping(link)
        return result

    def evaluate(self):
        """
        Stage 0 → 1: cube has left the box and is at rest.
        Stage 1 → 2: target object is inside the box and released — success.
        """
        cube_outside = self._is_outside_box(self.cube)
        cube_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        cube_not_grasped = ~self.agent.is_grasping(self.cube)

        target_in_box = self._is_in_box(self.target_object)
        target_not_grasped = ~self._is_grasping_target()
        is_robot_static = self.agent.is_static(threshold=0.2)

        # Stage 0 -> 1
        stage_0_done = cube_outside & cube_static & cube_not_grasped
        self.stage_id = torch.where(
            (self.stage_id == 0) & stage_0_done,
            torch.ones_like(self.stage_id),
            self.stage_id,
        )

        # Stage 1 -> 2
        stage_1_done = target_in_box & target_not_grasped & is_robot_static
        self.stage_id = torch.where(
            (self.stage_id == 1) & stage_1_done,
            torch.ones_like(self.stage_id) * 2,
            self.stage_id,
        )

        success = self.stage_id == 2

        return {
            "success": success,
            "stage_id": self.stage_id,
            "cube_outside": cube_outside,
            "target_in_box": target_in_box,
            "stage_0_done": stage_0_done,
            "stage_1_done": stage_1_done,
        }

@register_env("put_blocks_in_box_and_take_out_memory", max_episode_steps=800)
class put_blocks_in_box_and_take_out_memory(EvalDREnvMixin, BaseEnv):
    """
    Long-horizon: place red + blue cubes in the box, then retrieve the red cube.
    
    Stages:
    - Stage 0: put the red cube in the box.
    - Stage 1: put the blue cube in the box.
    - Stage 2: take the red cube back out.
    - Stage 3: task complete.
    """
    
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]
    
    def __init__(
        self,
        *args,
        object_name=None,
        part_name=None,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0, 0.8], target=[0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 224, 224, 1, 0.01, 100)]
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.5, 0.8], target=[0, 0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    def _load_scene(self, options: dict):
        """Build the scene: table + box + red cube + blue cube."""
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        self.box = self._build_box()
        
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="red_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
    
    def _build_box(self):
        """Build a box with the opening facing up."""
        builder = self.scene.create_actor_builder()
        
        box_size = [0.15, 0.15, 0.06]  
        wall_thickness = 0.005
        
        # Box floor.
        builder.add_box_collision(
            sapien.Pose([0, 0, -box_size[2]/2]),
            [box_size[0]/2, box_size[1]/2, wall_thickness/2]
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, -box_size[2]/2]),
            [box_size[0]/2, box_size[1]/2, wall_thickness/2],
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
        )
        
        walls = [
            # front
            (sapien.Pose([box_size[0]/2, 0, 0]), [wall_thickness/2, box_size[1]/2, box_size[2]/2]),
            # back
            (sapien.Pose([-box_size[0]/2, 0, 0]), [wall_thickness/2, box_size[1]/2, box_size[2]/2]),
            # left
            (sapien.Pose([0, box_size[1]/2, 0]), [box_size[0]/2, wall_thickness/2, box_size[2]/2]),
            # right
            (sapien.Pose([0, -box_size[1]/2, 0]), [box_size[0]/2, wall_thickness/2, box_size[2]/2]),
        ]
        
        for pose, half_size in walls:
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(
                pose, half_size,
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
            )
        
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_static(name="box")
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset per-episode state."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            box_xy = randomization.uniform(
                [0.01, 0.05], [0.05, 0.05], size=(b, 2)
            )
            box_pos = torch.zeros((b, 3))
            box_pos[:, :2] = box_xy
            box_pos[:, 2] = 0.05
            box_quat = randomization.random_quaternions(
                b, lock_x=True, lock_y=True
            )
            self.box.set_pose(Pose.create_from_pq(box_pos, box_quat))
            
            red_xy = randomization.uniform(
                [-0.15, -0.2], [-0.05, -0.1], size=(b, 2)
            )
            red_pos = torch.zeros((b, 3))
            red_pos[:, :2] = red_xy
            red_pos[:, 2] = 0.02
            red_quat = randomization.random_quaternions(
                b, lock_x=True, lock_y=True
            )
            self.red_cube.set_pose(Pose.create_from_pq(red_pos, red_quat))
            
            blue_xy = randomization.uniform(
                [0.05, -0.2], [0.15, -0.1], size=(b, 2)
            )
            blue_pos = torch.zeros((b, 3))
            blue_pos[:, :2] = blue_xy
            blue_pos[:, 2] = 0.02
            blue_quat = randomization.random_quaternions(
                b, lock_x=True, lock_y=True
            )
            self.blue_cube.set_pose(Pose.create_from_pq(blue_pos, blue_quat))
            
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04
            ])
            qpos = self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
            
            # At which stage is the task
            if not hasattr(self, 'stage_id'):
                self.stage_id = torch.zeros(b, dtype=torch.long, device=self.device)
            else:
                self.stage_id[env_idx] = 0
            
            # Randomise put-in order: 0 = red first then blue, 1 = blue first then red.
            if not hasattr(self, 'put_order'):
                self.put_order = torch.zeros(b, dtype=torch.long, device=self.device)
            
            # Record which cube must be retrieved (always the one placed first).
            if not hasattr(self, 'take_out_order'):
                self.take_out_order = torch.zeros(b, dtype=torch.long, device=self.device)
            self.take_out_order[env_idx] = self.put_order[env_idx]
    
    def _is_cube_in_box(self, cube: Actor, box=None) -> torch.Tensor:
        """Whether the cube is inside the box (defaults to self.box)."""
        if box is None:
            box = self.box
        cube_pos = cube.pose.p
        box_pos = box.pose.p
        
        xy_in_box = (
            (torch.abs(cube_pos[:, 0] - box_pos[:, 0]) < 0.12) &
            (torch.abs(cube_pos[:, 1] - box_pos[:, 1]) < 0.12)
        )
        
        z_in_box = (
            (cube_pos[:, 2] > box_pos[:, 2] - 0.03) &
            (cube_pos[:, 2] < box_pos[:, 2] + 0.15)
        )
        
        return xy_in_box & z_in_box
    
    def _is_cube_outside_box(self, cube: Actor) -> torch.Tensor:
        """Whether the cube is outside the box."""
        cube_pos = cube.pose.p
        box_pos = self.box.pose.p
        
        xy_outside = (
            (torch.abs(cube_pos[:, 0] - box_pos[:, 0]) > 0.15) |
            (torch.abs(cube_pos[:, 1] - box_pos[:, 1]) > 0.15)
        )
        
        return xy_outside
    
    def evaluate(self):
        """Evaluate task completion (supports the randomised put-order)."""
        red_in_box = self._is_cube_in_box(self.red_cube)
        blue_in_box = self._is_cube_in_box(self.blue_cube)
        red_outside_box = self._is_cube_outside_box(self.red_cube)
        blue_outside_box = self._is_cube_outside_box(self.blue_cube)
        
        is_robot_static = self.agent.is_static(threshold=0.2)
        red_is_static = self.red_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        blue_is_static = self.blue_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        red_not_grasped = ~self.agent.is_grasping(self.red_cube)
        blue_not_grasped = ~self.agent.is_grasping(self.blue_cube)
        
        # put_order=0: red first then blue; retrieve red.
        # put_order=1: blue first then red; retrieve blue.
        
        # Stage 0 → 1: the first cube is in the box.
        stage_0_complete_red_first = red_in_box & red_is_static & red_not_grasped
        stage_0_complete_blue_first = blue_in_box & blue_is_static & blue_not_grasped
        stage_0_complete = torch.where(
            self.put_order == 0,
            stage_0_complete_red_first,
            stage_0_complete_blue_first
        )
        self.stage_id = torch.where(
            (self.stage_id == 0) & stage_0_complete,
            torch.ones_like(self.stage_id),
            self.stage_id
        )
        
        # Stage 1 → 2: both cubes are in the box.
        stage_1_complete = red_in_box & blue_in_box & red_is_static & blue_is_static & red_not_grasped & blue_not_grasped
        self.stage_id = torch.where(
            (self.stage_id == 1) & stage_1_complete,
            torch.ones_like(self.stage_id) * 2,
            self.stage_id
        )
        
        # Stage 2 → 3: the first cube has been taken out again.
        stage_2_complete_red_out = (
            red_outside_box & blue_in_box & 
            red_is_static & blue_is_static & 
            red_not_grasped & is_robot_static
        )
        stage_2_complete_blue_out = (
            blue_outside_box & red_in_box & 
            red_is_static & blue_is_static & 
            blue_not_grasped & is_robot_static
        )
        stage_2_complete = torch.where(
            self.put_order == 0,
            stage_2_complete_red_out,
            stage_2_complete_blue_out
        )
        self.stage_id = torch.where(
            (self.stage_id == 2) & stage_2_complete,
            torch.ones_like(self.stage_id) * 3,
            self.stage_id
        )
        
        success = (self.stage_id == 3)
        
        return {
            "success": success,
            "stage_id": self.stage_id,
            "put_order": self.put_order,
            "red_in_box": red_in_box,
            "blue_in_box": blue_in_box,
            "red_outside_box": red_outside_box,
            "blue_outside_box": blue_outside_box,
            "stage_0_complete": stage_0_complete,
            "stage_1_complete": stage_1_complete,
            "stage_2_complete": stage_2_complete,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Compute the dense per-step reward.

        Base implementation returns zeros; subclasses with reward shaping
        override this method.
        """
        # Base: zero reward; subclasses provide task-specific shaping.
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Normalised dense reward — same contract as compute_dense_reward but in [0, 1]."""
        # Base: zero.
        return torch.zeros(len(obs) if hasattr(obs, '__len__') else 1)

@register_env("put_blocks_into_boxes", max_episode_steps=800)
class PutBlocksInBox(put_blocks_in_box_and_take_out_memory):
    """
    Long-horizon: put the chosen-color cube in the left box, the other two in the right box.

    Instruction example: "Place the {special_cube} cube in the left-hand box and the remaining cubes in the right-hand box."

    Stages:
    - Stage 0: place special_cube in left_box.
    - Stage 1: place the second cube in right_box.
    - Stage 2: place the third cube in right_box.
    - Stage 3: task complete.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    BOX_X = 0.10
    BOX_LEFT_Y  =  -0.20
    BOX_RIGHT_Y = 0.20
    BOX_Z = 0.05

    def __init__(
        self,
        *args,
        object_name=None,
        part_name=None,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        special_cube: str = "green",   # put into left box
        **kwargs
    ):
        assert special_cube in ("red", "blue", "green"), "special_cube : 'red'、'blue' or 'green'"
        self.special_cube = special_cube
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super(put_blocks_in_box_and_take_out_memory, self).__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        """Build the scene: table + left box + right box + red / blue / green cubes."""
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.left_box = self._build_named_box("left_box")
        self.right_box = self._build_named_box("right_box")

        self.box = self.left_box
        self.box2 = self.right_box

        self.red_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1],
            name="red_cube", initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self.blue_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1],
            name="blue_cube", initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0.8, 0, 1],
            name="green_cube", initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )

    def _build_named_box(self, name: str):
        """Build an open-top box with the given actor name."""
        builder = self.scene.create_actor_builder()
        box_size = [0.15, 0.15, 0.06]
        wall_thickness = 0.005
        builder.add_box_collision(
            sapien.Pose([0, 0, -box_size[2] / 2]),
            [box_size[0] / 2, box_size[1] / 2, wall_thickness / 2]
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, -box_size[2] / 2]),
            [box_size[0] / 2, box_size[1] / 2, wall_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
        )
        walls = [
            (sapien.Pose([box_size[0] / 2, 0, 0]),  [wall_thickness / 2, box_size[1] / 2, box_size[2] / 2]),
            (sapien.Pose([-box_size[0] / 2, 0, 0]), [wall_thickness / 2, box_size[1] / 2, box_size[2] / 2]),
            (sapien.Pose([0, box_size[1] / 2, 0]),  [box_size[0] / 2, wall_thickness / 2, box_size[2] / 2]),
            (sapien.Pose([0, -box_size[1] / 2, 0]), [box_size[0] / 2, wall_thickness / 2, box_size[2] / 2]),
        ]
        for pose, half_size in walls:
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(
                pose, half_size,
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
            )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_static(name=name)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Per-episode init: fixed left/right boxes; cubes spawn on the arm side, randomly placed."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ------------------------------------------------------------------
            # Boxes are pinned along the Y axis (perpendicular to the arm), no rotation.
            # ------------------------------------------------------------------
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, -1)

            left_pos = torch.zeros((b, 3), device=self.device)
            left_pos[:, 0] = self.BOX_X
            left_pos[:, 1] = self.BOX_LEFT_Y
            left_pos[:, 2] = self.BOX_Z
            self.left_box.set_pose(Pose.create_from_pq(left_pos, identity_quat))

            right_pos = torch.zeros((b, 3), device=self.device)
            right_pos[:, 0] = self.BOX_X
            right_pos[:, 1] = self.BOX_RIGHT_Y
            right_pos[:, 2] = self.BOX_Z
            self.right_box.set_pose(Pose.create_from_pq(right_pos, identity_quat))

            self.box = self.left_box
            self.box2 = self.right_box

            offsets = [
                randomization.uniform([-0.10, -0.18], [-0.05, -0.10], size=(b, 2)),
                randomization.uniform([-0.10, -0.04], [-0.05,  0.04], size=(b, 2)),
                randomization.uniform([-0.10,  0.10], [-0.05,  0.18], size=(b, 2)),
            ]
            cube_z = 0.02

            red_pos = torch.zeros((b, 3))
            red_pos[:, :2] = offsets[0]
            red_pos[:, 2] = cube_z
            red_quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.red_cube.set_pose(Pose.create_from_pq(red_pos, red_quat))

            blue_pos = torch.zeros((b, 3))
            blue_pos[:, :2] = offsets[1]
            blue_pos[:, 2] = cube_z
            blue_quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.blue_cube.set_pose(Pose.create_from_pq(blue_pos, blue_quat))

            green_pos = torch.zeros((b, 3))
            green_pos[:, :2] = offsets[2]
            green_pos[:, 2] = cube_z
            green_quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.green_cube.set_pose(Pose.create_from_pq(green_pos, green_quat))

            # Robot init.
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04
            ])
            qpos = self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Per-env task stage.
            if not hasattr(self, 'stage_id'):
                self.stage_id = torch.zeros(b, dtype=torch.long, device=self.device)
            else:
                self.stage_id[env_idx] = 0

    def _get_cube_by_order_idx(self, order_idx: int):
        """
        order_idx=0 -> red, 1 -> blue, 2 -> green
          0: [0,1,2]  R-B-G
          1: [0,2,1]  R-G-B
          2: [1,0,2]  B-R-G
          3: [1,2,0]  B-G-R
          4: [2,0,1]  G-R-B
          5: [2,1,0]  G-B-R
        """
        PERM_TABLE = torch.tensor([
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ], dtype=torch.long, device=self.device)  # shape (6, 3)
        color_idx = PERM_TABLE[self.put_order, order_idx]  # shape (b,)
        return color_idx

    def _cube_for_color(self, color_idx_tensor: torch.Tensor):
        """Return pose.p of the cube selected by the per-env color_idx tensor."""
        red_p = self.red_cube.pose.p
        blue_p = self.blue_cube.pose.p
        green_p = self.green_cube.pose.p
        color_idx = color_idx_tensor.unsqueeze(1)  
        pos = torch.where(color_idx == 0, red_p,
               torch.where(color_idx == 1, blue_p, green_p))
        return pos

    def _is_grasping_color(self, color_idx_tensor: torch.Tensor) -> torch.Tensor:
        """Whether the arm is grasping the cube identified by color_idx."""
        r = self.agent.is_grasping(self.red_cube)
        b = self.agent.is_grasping(self.blue_cube)
        g = self.agent.is_grasping(self.green_cube)
        ci = color_idx_tensor
        return torch.where(ci == 0, r, torch.where(ci == 1, b, g))

    def _get_special_cube(self):
        """Return the Actor for special_cube."""
        return {"red": self.red_cube, "blue": self.blue_cube, "green": self.green_cube}[self.special_cube]

    def _get_other_cubes(self):
        """Return the two Actors that are not special_cube."""
        return [c for name, c in [("red", self.red_cube), ("blue", self.blue_cube), ("green", self.green_cube)]
                if name != self.special_cube]

    def _is_in_box(self, cube: Actor, box) -> torch.Tensor:
        return self._is_cube_in_box(cube, box=box)

    def _is_static(self, cube: Actor) -> torch.Tensor:
        return cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    def evaluate(self):
        special = self._get_special_cube()
        others  = self._get_other_cubes()

        special_in_left  = self._is_in_box(special,   self.left_box)
        other0_in_right  = self._is_in_box(others[0], self.right_box)
        other1_in_right  = self._is_in_box(others[1], self.right_box)

        special_static   = self._is_static(special)
        other0_static    = self._is_static(others[0])
        other1_static    = self._is_static(others[1])

        special_ng  = ~self.agent.is_grasping(special)
        other0_ng   = ~self.agent.is_grasping(others[0])
        other1_ng   = ~self.agent.is_grasping(others[1])

        is_robot_static = self.agent.is_static(threshold=0.2)

        # Stage 0 -> 1: special_cube -> left_box
        stage_0_done = special_in_left & special_static & special_ng
        self.stage_id = torch.where(
            (self.stage_id == 0) & stage_0_done,
            torch.ones_like(self.stage_id), self.stage_id
        )

        # Stage 1 -> 2: other_0 -> right_box
        stage_1_done = (
            special_in_left & other0_in_right &
            special_static & other0_static &
            special_ng & other0_ng
        )
        self.stage_id = torch.where(
            (self.stage_id == 1) & stage_1_done,
            torch.ones_like(self.stage_id) * 2, self.stage_id
        )

        # Stage 2 -> 3: other_2 -> right_box
        stage_2_done = (
            special_in_left & other0_in_right & other1_in_right &
            special_static & other0_static & other1_static &
            special_ng & other0_ng & other1_ng &
            is_robot_static
        )
        self.stage_id = torch.where(
            (self.stage_id == 2) & stage_2_done,
            torch.ones_like(self.stage_id) * 3, self.stage_id
        )

        success = (self.stage_id == 3)

        return {
            "success": success,
            "stage_id": self.stage_id,
            "special_cube": self.special_cube,
            "special_in_left": special_in_left,
            "other0_in_right": other0_in_right,
            "other1_in_right": other1_in_right,
            "stage_0_done": stage_0_done,
            "stage_1_done": stage_1_done,
            "stage_2_done": stage_2_done,
        }


# ---------------------------------------------------------------------------
# AssemblingKits (original ManiSkill kit-insertion task; kept unchanged so the
# METAFINE letter env below can reuse its asset-loading + evaluate plumbing).
# ---------------------------------------------------------------------------
@register_env(
    "assembling_kits", asset_download_ids=["assembling_kits"], max_episode_steps=200
)
class AssemblingKitsEnv(EvalDREnvMixin, BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    def __init__(
        self,
        asset_root=None,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        from mani_skill import ASSET_DIR
        if asset_root is None:
            asset_root = f"{ASSET_DIR}/tasks/assembling_kits"
        self.asset_root = Path(asset_root)
        self._kit_dir = self.asset_root / "kits"
        self._models_dir = self.asset_root / "models"
        if not (self._kit_dir.exists() and self._models_dir.exists()):
            raise FileNotFoundError(
                "AssemblingKits assets not found. Run "
                "`python -m mani_skill.utils.download_asset assembling_kits` "
                "or pass asset_root."
            )
        self._episode_json = io_utils.load_json(self.asset_root / "episodes.json")
        self._episodes = self._episode_json["episodes"]
        self.symmetry = self._episode_json["config"]["symmetry"]
        self.color = self._episode_json["config"]["color"]
        self.object_scale = self._episode_json["config"]["object_scale"]
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        kwargs.pop("object_name", None)
        kwargs.pop("part_name", None)
        super().__init__(
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2 ** 20))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.2, 0, 0.4], [0, 0, 0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.3, 0.8], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _parse_json(self, path):
        kit_json = io_utils.load_json(path)
        object_goal_pos = {o["object_id"]: common.to_numpy(o["pos"]) for o in kit_json["objects"]}
        objects_goal_rot = {o["object_id"]: o["rot"] for o in kit_json["objects"]}
        return object_goal_pos, objects_goal_rot

    def _get_kit_builder_and_goals(self, kit_id: str):
        object_goal_pos, objects_goal_rot = self._parse_json(self._kit_dir / f"{kit_id}.json")
        builder = self.scene.create_actor_builder()
        kit_path = str(self._kit_dir / f"{kit_id}.obj")
        builder.add_nonconvex_collision_from_file(kit_path)
        builder.add_visual_from_file(
            kit_path,
            material=sapien.render.RenderMaterial(
                base_color=[0.27807487, 0.20855615, 0.16934046, 1.0],
                roughness=0.5, specular=0.0,
            ),
        )
        return builder, object_goal_pos, objects_goal_rot

    def _get_object_builder(self, object_id, static: bool = False, color_id: int = 0):
        collision_path = self._models_dir / "collision" / f"{object_id:02d}.obj"
        visual_path = self._models_dir / "visual" / f"{object_id:02d}.obj"
        builder = self.scene.create_actor_builder()
        if static:
            builder.add_nonconvex_collision_from_file(str(collision_path), scale=self.object_scale)
        else:
            builder.add_multiple_convex_collisions_from_file(str(collision_path), scale=self.object_scale)
        builder.add_visual_from_file(
            str(visual_path), scale=self.object_scale,
            material=sapien.render.RenderMaterial(
                base_color=self.color[color_id], roughness=0.1, specular=0.0,
            ),
        )
        return builder

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()
            self.symmetry = common.to_tensor(self.symmetry)
            eps_idxs = self._batched_episode_rng.randint(0, len(self._episodes))
            pick_color_ids = self._batched_episode_rng.choice(len(self.color))
            other_color_ids = self._batched_episode_rng.choice(len(self.color), size=(10,))
            kits, objs_to_place, all_other_objs = [], [], []
            self.object_ids = []
            self.goal_pos = np.zeros((self.num_envs, 3))
            self.goal_rot = np.zeros((self.num_envs,))
            for i, eps_idx in enumerate(eps_idxs):
                scene_idxs = [i]
                episode = self._episodes[eps_idx]
                kit_builder, object_goal_pos, object_goal_rot = self._get_kit_builder_and_goals(episode["kit"])
                kit = (kit_builder.set_scene_idxs(scene_idxs)
                       .set_initial_pose(sapien.Pose([0, 0, 0.01]))
                       .build_static(f"kit_{i}"))
                kits.append(kit)
                obj_to_place = (
                    self._get_object_builder(episode["obj_to_place"], color_id=pick_color_ids[i])
                    .set_scene_idxs(scene_idxs)
                    .set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
                    .build(f"obj_{i}")
                )
                self.object_ids.append(episode["obj_to_place"])
                objs_to_place.append(obj_to_place)
                other_objs = [
                    self._get_object_builder(obj_id, static=True, color_id=other_color_ids[i, j])
                    .set_scene_idxs(scene_idxs)
                    .set_initial_pose(sapien.Pose(
                        object_goal_pos[obj_id],
                        q=euler2quat(0, 0, object_goal_rot[obj_id]),
                    ))
                    .build_static(f"in_place_obj_{i}_{j}")
                    for j, obj_id in enumerate(episode["obj_in_place"])
                ]
                all_other_objs.append(other_objs)
                self.goal_pos[i] = object_goal_pos[episode["obj_to_place"]]
                self.goal_rot[i] = object_goal_rot[episode["obj_to_place"]]
            self.obj = Actor.merge(objs_to_place)
            self.object_ids = torch.tensor(self.object_ids, dtype=int)
            self.goal_pos = common.to_tensor(self.goal_pos)
            self.goal_rot = common.to_tensor(self.goal_rot)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.2 - 0.1
            xyz[:, 1] = torch.rand((b,)) * 0.364 - 0.364 / 2
            xyz[:, 2] = 0.02
            q = randomization.random_quaternions(b, device=self.device, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _check_pos_diff(self, pos_eps=2e-2):
        pos_diff = self.goal_pos[:, :2] - self.obj.pose.p[:, :2]
        pos_diff_norm = torch.linalg.norm(pos_diff, axis=1)
        return pos_diff, pos_diff_norm, pos_diff_norm < pos_eps

    def _check_rot_diff(self, rot_eps=np.deg2rad(4)):
        rot = rotation_conversions.matrix_to_euler_angles(
            rotation_conversions.quaternion_to_matrix(self.obj.pose.q), "XYZ"
        )[:, -1]
        rot_diff = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        has_symmetries = self.symmetry[self.object_ids] > 0
        rot_diff_sym = torch.abs(rot - self.goal_rot) % self.symmetry[self.object_ids]
        has_half_symmetries = rot_diff_sym > self.symmetry[self.object_ids] / 2
        rot_diff[has_symmetries] = rot_diff_sym[has_symmetries]
        rot_diff[has_half_symmetries] = (
            self.symmetry[self.object_ids][has_half_symmetries]
            - rot_diff_sym[has_half_symmetries]
        )
        return rot_diff, rot_diff < rot_eps

    def _check_in_slot(self, obj, height_eps=3e-3):
        return obj.pose.p[:, 2] < height_eps

    def evaluate(self) -> dict:
        pos_diff, pos_diff_norm, pos_correct = self._check_pos_diff()
        rot_diff, rot_correct = self._check_rot_diff()
        in_slot = self._check_in_slot(self.obj)
        return {
            "pos_diff": pos_diff,
            "pos_diff_norm": pos_diff_norm,
            "pos_correct": pos_correct,
            "rot_diff": rot_diff,
            "rot_correct": rot_correct,
            "in_slot": in_slot,
            "success": pos_correct & rot_correct & in_slot,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                goal_pos=self.goal_pos,
                goal_rot=self.goal_rot,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
            )
        return obs


# ---------------------------------------------------------------------------
# METAFINE single-empty-slot env, parameterized by `target_letter`.
# Kit shows the word METAFINE (linear row); 7 letters are pre-inserted, the
# slot of the chosen letter is empty, and that letter peg starts on the table
# scattered together with 5 distractor letter pegs.
# ---------------------------------------------------------------------------
@register_env("assembling_kits_metafine_letter", max_episode_steps=200)
class AssemblingKitsMetafineLetterEnv(AssemblingKitsEnv):
    """Args:
        target_letter: one of {M, E, T, A, F, I, N}. Defaults to "I".
                       For "E", uses object_id=1 (first E in METAFINE).
    """

    LETTER_TO_ID = {"M": 0, "E": 1, "T": 2, "A": 3, "F": 4, "I": 5, "N": 6}

    SCATTER_X_RANGE = (-0.18, 0.18)
    SCATTER_Y_RANGE = (-0.32, -0.12)
    SCATTER_MIN_DIST = 0.075

    KIT_COLOR = [0.05, 0.05, 0.05, 1.0]
    _ORANGE = [1.00, 0.55, 0.00, 1.0]
    _ICE_BLUE = [0.55, 0.85, 0.95, 1.0]
    _GOLD = [0.95, 0.78, 0.20, 1.0]
    _BLUE = [0.10, 0.30, 0.85, 1.0]
    PEG_COLORS = [
        _ORANGE, _ICE_BLUE, _GOLD, _BLUE,        # 0..3 M E T A
        _ORANGE, _ICE_BLUE, _GOLD, _BLUE,        # 4..7 F I N E
    ]

    def __init__(
        self,
        target_letter: str = "I",
        asset_root: str = None,
        robot_uids: str = "panda_wristcam",
        num_envs: int = 1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = str(Path(__file__).resolve().parents[1] / "assets" / "metafine_kit_linear")
        oid = self.LETTER_TO_ID[str(target_letter).upper()]
        self.target_letter = str(target_letter).upper()
        self.target_episode_idx = oid
        self.distractor_letter_ids = [i for i in range(8) if i != oid][:5]
        kwargs.pop("object_name", None)
        kwargs.pop("part_name", None)
        super().__init__(
            asset_root=asset_root,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0, 0.4], [0, 0, 0])
        return CameraConfig("render_camera", pose, 512, 512, np.pi / 2, 0.01, 100)

    @property
    def _default_sensor_configs(self):
        eye = np.array([0.2, 0, 0.4], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)
        eye, target = self._maybe_jitter_camera(eye, target)
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    # --- relaxed success thresholds ---
    def _check_pos_diff(self, pos_eps=2.0e-2):
        return super()._check_pos_diff(pos_eps=pos_eps)

    def _check_rot_diff(self, rot_eps=np.deg2rad(15)):
        return super()._check_rot_diff(rot_eps=rot_eps)

    def _check_in_slot(self, obj, height_eps=4e-2):
        return obj.pose.p[:, 2] < height_eps

    # --- override colors ---
    def _get_kit_builder_and_goals(self, kit_id: str):
        object_goal_pos, objects_goal_rot = self._parse_json(self._kit_dir / f"{kit_id}.json")
        builder = self.scene.create_actor_builder()
        kit_path = str(self._kit_dir / f"{kit_id}.obj")
        builder.add_nonconvex_collision_from_file(kit_path)
        builder.add_visual_from_file(
            kit_path,
            material=sapien.render.RenderMaterial(
                base_color=self.KIT_COLOR, roughness=0.6, specular=0.0,
            ),
        )
        return builder, object_goal_pos, objects_goal_rot

    def _get_object_builder(self, object_id, static: bool = False, color_id: int = 0):
        collision_path = self._models_dir / "collision" / f"{object_id:02d}.obj"
        visual_path = self._models_dir / "visual" / f"{object_id:02d}.obj"
        builder = self.scene.create_actor_builder()
        if static:
            builder.add_nonconvex_collision_from_file(str(collision_path), scale=self.object_scale)
        else:
            builder.add_multiple_convex_collisions_from_file(str(collision_path), scale=self.object_scale)
        builder.add_visual_from_file(
            str(visual_path), scale=self.object_scale,
            material=sapien.render.RenderMaterial(
                base_color=self.PEG_COLORS[int(object_id)],
                roughness=0.4, specular=0.05,
            ),
        )
        return builder

    # --- scene with 7 pre-inserted letters + 5 distractors + the target peg ---
    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()
            self.symmetry = common.to_tensor(self.symmetry)
            kits, objs_to_place, all_other_objs = [], [], []
            per_env_distractors = [[] for _ in range(self.num_envs)]
            self.object_ids = []
            self.goal_pos = np.zeros((self.num_envs, 3))
            self.goal_rot = np.zeros((self.num_envs,))
            for i in range(self.num_envs):
                scene_idxs = [i]
                episode = self._episodes[self.target_episode_idx]
                kit_builder, object_goal_pos, object_goal_rot = self._get_kit_builder_and_goals(episode["kit"])
                kit = (kit_builder.set_scene_idxs(scene_idxs)
                       .set_initial_pose(sapien.Pose([0, 0, 0.01]))
                       .build_static(f"kit_{i}"))
                kits.append(kit)
                obj_to_place = (
                    self._get_object_builder(episode["obj_to_place"], color_id=0)
                    .set_scene_idxs(scene_idxs)
                    .set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
                    .build(f"obj_{i}")
                )
                self.object_ids.append(episode["obj_to_place"])
                objs_to_place.append(obj_to_place)
                other_objs = [
                    self._get_object_builder(obj_id, static=True, color_id=(jj + 1) % len(self.color))
                    .set_scene_idxs(scene_idxs)
                    .set_initial_pose(sapien.Pose(
                        object_goal_pos[obj_id],
                        q=euler2quat(0, 0, object_goal_rot[obj_id]),
                    ))
                    .build_static(f"in_place_obj_{i}_{jj}")
                    for jj, obj_id in enumerate(episode["obj_in_place"])
                ]
                all_other_objs.append(other_objs)
                for j, did in enumerate(self.distractor_letter_ids):
                    actor = (
                        self._get_object_builder(did, static=False, color_id=(j + 2) % len(self.color))
                        .set_scene_idxs(scene_idxs)
                        .set_initial_pose(sapien.Pose(p=[0, 0, 0.5]))
                        .build(f"distractor_{i}_{j}")
                    )
                    per_env_distractors[i].append(actor)
                self.goal_pos[i] = object_goal_pos[episode["obj_to_place"]]
                self.goal_rot[i] = object_goal_rot[episode["obj_to_place"]]
            self.obj = Actor.merge(objs_to_place)
            self.object_ids = torch.tensor(self.object_ids, dtype=int)
            self.goal_pos = common.to_tensor(self.goal_pos)
            self.goal_rot = common.to_tensor(self.goal_rot)
            self.distractors = []
            for j in range(len(self.distractor_letter_ids)):
                actors = [per_env_distractors[i][j] for i in range(self.num_envs)]
                self.distractors.append(Actor.merge(actors))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            n_d = len(self.distractor_letter_ids)
            n_total = n_d + 1
            x_lo, x_hi = self.SCATTER_X_RANGE
            y_lo, y_hi = self.SCATTER_Y_RANGE
            min_d2 = self.SCATTER_MIN_DIST ** 2
            batch_positions = []
            for _ in range(b):
                positions = []
                for _ in range(2000):
                    if len(positions) >= n_total:
                        break
                    x = float(torch.rand(1).item()) * (x_hi - x_lo) + x_lo
                    y = float(torch.rand(1).item()) * (y_hi - y_lo) + y_lo
                    if all((px - x) ** 2 + (py - y) ** 2 > min_d2 for px, py in positions):
                        positions.append((x, y))
                while len(positions) < n_total:
                    positions.append(positions[-1])
                idx = torch.randperm(n_total).tolist()
                batch_positions.append([positions[k] for k in idx])
            e_xyz = torch.zeros((b, 3))
            for env_i in range(b):
                e_xyz[env_i, 0] = batch_positions[env_i][0][0]
                e_xyz[env_i, 1] = batch_positions[env_i][0][1]
            qE = randomization.random_quaternions(b, device=self.device, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=e_xyz, q=qE))
            for j in range(n_d):
                p = torch.zeros((b, 3))
                for env_i in range(b):
                    p[env_i, 0] = batch_positions[env_i][j + 1][0]
                    p[env_i, 1] = batch_positions[env_i][j + 1][1]
                qd = randomization.random_quaternions(b, device=self.device, lock_x=True, lock_y=True)
                self.distractors[j].set_pose(Pose.create_from_pq(p=p, q=qd))
