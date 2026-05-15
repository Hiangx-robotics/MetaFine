import os.path as osp
from pathlib import Path
import json
import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
from utils.util import rotation_quaternion_z
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils import sapien_utils
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.structs.pose import Pose

class GraspPartSceneBuilder(SceneBuilder):
    """
    Abstract scene builder for grasp part tasks.

    This is a general-purpose scene builder that supports dynamic loading of different objects.
    Config-driven approach allows easy adaptation to different grasping tasks.

    Main features:
    1. Build general table workspace (height set to 0 for easy coordinate calculation)
    2. Set reasonable initial poses for robots
    3. Dynamically load specified URDF object models
    4. Read object parameters from config files (position, scale, etc.)
    5. Build ground and scene objects

    Args:
        object_name (str): Name of the object to load, corresponding to folder name under assets
        robot_init_qpos_noise (float): Noise parameter for robot initial joint angles

    Attributes:
        table_length (float): Table length (X-axis direction)
        table_width (float): Table width (Y-axis direction)
        table_height (float): Table height (Z-axis direction)
        ground (sapien.Entity): Ground object
        table (sapien.Entity): Table object
        target_object (sapien.Articulation): Target object articulation
        scene_objects (list): List of all static objects in the scene
    """

    def __init__(self, env, object_name="cabinet", robot_init_qpos_noise=0.02):
        """
        Initialize the scene builder.

        Args:
            env: Environment instance
            object_name (str): Name of the object to load, default "cabinet"
            robot_init_qpos_noise (float): Noise for robot initial joint angles, default 0.02
        """
        super().__init__(env)
        self.object_name = object_name
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.verbose = False
        # Load object parameters from config file
        self._load_object_config()

    def _load_object_config(self):
        """
        Load object parameters from config file.

        Read configuration from assets/{object_name}/model_data.json,
        maintain compatibility with existing format and convert to internal format.
        """
        config_path = Path(__file__).parent.parent / "assets" / self.object_name / "model_data.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                # Convert config format for compatibility
                self.object_config = self._convert_config_format(raw_config)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in config file {config_path}: {e}")
                self.object_config = self._get_default_object_config()
        else:
            print(f"Warning: Config file not found {config_path}, using defaults")
            self.object_config = self._get_default_object_config()

    # Schema defaults used when an asset's model_data.json omits a field.
    # Per-asset overrides live in the JSON itself, never in this module.
    _DEFAULT_INITIAL_POSE = [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    _DEFAULT_MATERIAL = {
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
    }

    def _convert_config_format(self, raw_config):
        """Translate the on-disk model_data.json schema to the internal one.

        Pure data-driven: every value comes from ``raw_config`` with a
        sensible default; no ``if object_name == ...`` branches. Per-asset
        overrides (initial_pose, load_mode, custom scale) belong in
        ``assets/<name>/model_data.json``.
        """
        return {
            "scale": raw_config.get("scale", 1.0),
            "urdf_file": raw_config.get("urdf_file", "mobility.urdf"),
            "initial_pose": raw_config.get("initial_pose", list(self._DEFAULT_INITIAL_POSE)),
            "material": raw_config.get("material", dict(self._DEFAULT_MATERIAL)),
            # load_mode is ``None`` when the asset does not declare it; the
            # SlideAlongSceneBuilder treats that as "small" (default).
            "load_mode": raw_config.get("load_mode"),
            "init_qpos": raw_config.get("init_qpos", []),
            "grasp_parts": raw_config.get("grasp_parts", {}),
        }

    def _get_default_object_config(self):
        """
        Get default object configuration.

        Returns:
            dict: Default object configuration parameters
        """
        return {
            "scale": 1.0,
            "urdf_file": "mobility.urdf",
            "initial_pose": [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, qw, qx, qy, qz]
            "material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0
            }
        }



    def build(self):
        """
        Build all objects in the scene.

        This method creates and configures all static and dynamic objects in the scene, including:
        - Table workspace (with collision and visual models)
        - Ground
        - Specified target object (loaded from URDF)

        Scene coordinate system: table height is Z=0 for easy position calculation and control.
        """
        # ============ Build table workspace ============
        self._build_table()

        # ============ Dynamically load target object ============
        self._build_target_object()

        # ============ Set scene object list ============
        self.scene_objects: list[sapien.Entity] = [self.table, self.ground]

    def _place_object_with_drop(self, env_idx: torch.Tensor):
        """Place object (currently maintain URDF original pose)"""
        if not hasattr(self, 'target_object'):
            return
    
        # Get current object position
        current_pose = self.target_object.get_pose()
        current_z = current_pose.p.cpu().numpy().flatten()[-1]  # Get z coordinate

        # If object is at high position (above table height 0.1m), run drop simulation
        if current_z > 0.1:  # Table at z=0, object above 0.1m needs to drop
            if self.verbose:
                print(f"Object at high position (z={current_z:.3f}), starting drop simulation...")
            self._simulate_drop(steps=60)  # Run 60-step physics simulation
            final_pose = self.target_object.get_pose()
            final_z = final_pose.p.cpu().numpy().flatten()[-1]
            if self.verbose:
                print(f"✓ Drop completed, final position: z={final_z:.3f}")

    def _simulate_drop(self, steps=30):
        """Run physics simulation to drop object and stabilize"""
        scene = self.env.scene
        for _ in range(steps):
            scene.step()

    def _build_table(self):
        """
        Build table workspace.

        Create table with collision and visual models, table surface height set to Z=0.
        """
        # Build an actor for the table.
        builder = self.scene.create_actor_builder()

        # Locate the table mesh file.
        model_dir = Path(__file__).parent.parent / "assets"
        table_model_file = str(model_dir / "table.glb")

        # Table scale factor.
        scale = 1.75

        # Table orientation: 90° about the world Z axis.
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))

        # Use a simple box collider in place of the full mesh for speed.
        # Collider centre is at half the table height.
        # half_size: (length/2, width/2, height/2).
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )

        # Add the GLB-based visual mesh.
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )

        # Place the table so its top surface sits at Z=0.
        # p=[-0.12, 0, -0.9196429]: shift down by the table thickness.
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build as kinematic so the table is fixed against physics forces.
        table = builder.build_kinematic(name="table-workspace")

        # Pre-computed AABB; avoids re-deriving it from the mesh each run.
        # AABB layout: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        aabb = np.array(
            [
                [-0.7402168, -1.2148621, -0.91964257],  # min corner
                [0.4688596, 1.2030163, 3.5762787e-07],   # max corner
            ]
        )

        # Calculate actual table dimensions from AABB
        self.table_length = aabb[1, 0] - aabb[0, 0]  # X dimension
        self.table_width = aabb[1, 1] - aabb[0, 1]   # Y dimension
        self.table_height = aabb[1, 2] - aabb[0, 2]  # Z dimension

        # Cache the table actor.
        self.table = table

        # ============ Build ground ============
        # Set ground width based on parallel mode
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500  # Larger ground needed for parallel mode

        # Build ground at table bottom level
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )

    def _build_target_object(self):
        """
        Dynamically build target object.

        Load object from corresponding URDF file based on configured object_name.
        Support reading scale, initial position and other parameters from config file.
        """
        # Create a URDF loader for articulated assets.
        object_loader = self.scene.create_urdf_loader()

        # Read parameters from the resolved config.
        scale = self.object_config.get("scale", 1.0)
        urdf_file = self.object_config.get("urdf_file", "mobility.urdf")
        initial_pose_config = self.object_config.get("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        material_config = self.object_config.get("material", {
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0
        })

        # Loader options.
        object_loader.fix_root_link = False  # root may translate freely
        object_loader.scale = scale  # asset scale factor
        object_loader.load_multiple_collisions_from_file = True  # multi-collider URDFs

        # URDF path.
        urdf_path = Path(__file__).parent.parent / "assets" / self.object_name / urdf_file

        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Parse the requested physics material.
        applied_urdf_config = sapien_utils.parse_urdf_config(
            dict(material=material_config)
        )

        # Apply the material to the loader.
        sapien_utils.apply_urdf_config(object_loader, applied_urdf_config)

        # Parse the URDF and collect its articulation builders.
        try:
            object_builders = object_loader.parse(str(urdf_path))["articulation_builders"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse URDF file {urdf_path}: {e}")

        if not object_builders:
            raise RuntimeError(f"No articulation builders found in URDF file {urdf_path}")

        # Use the first articulation builder.
        object_builder = object_builders[0]

        # Pin to scene 0 (extend for parallel-scene support).
        object_builder.set_scene_idxs(scene_idxs=[0])

        # Set the initial pose.
        if len(initial_pose_config) == 7:  # [x, y, z, qw, qx, qy, qz]
            initial_pose = sapien.Pose(
                p=initial_pose_config[:3],
                q=initial_pose_config[3:]
            )
        else:
            # Legacy format / fallback to identity pose.
            initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

        object_builder.initial_pose = initial_pose

        # Build the articulation with a non-fixed root.
        target_object = object_builder.build(
            name=f"{self.object_name}_object",
            fix_root_link=False
        )

        # Cache the target object.
        self.target_object = target_object

        if self.verbose:
            print(f"Successfully loaded target object '{self.object_name}' from {urdf_path}")
            print(f"Scale: {scale}, Initial pose: {initial_pose}")

    def initialize(self, env_idx: torch.Tensor):
        """
        Initialize scene and robot.

        Called at the start of each episode to reset object poses and robot states in the scene.

        Args:
            env_idx (torch.Tensor): Environment indices tensor to initialize
                                   In parallel simulation, multiple environments may run simultaneously

        Functions:
            1. Reset table pose to initial position
            2. Place object at fixed position and let it drop onto the table
            3. Set robot initial joint angles based on robot type
            4. Set robot base pose
        """

        b = len(env_idx)

        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )
        if hasattr(self, 'target_object') and self.target_object is not None:
            # Get URDF original pose from config or current pose
            initial_pose_config = self.object_config.get("initial_pose", [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
            if len(initial_pose_config) == 7:  # [x, y, z, qw, qx, qy, qz]
                quat = initial_pose_config[3:]  # (qw, qx, qy, qz)
            else:
                current_pose = self.target_object.get_pose()
                quat = current_pose.q.cpu().numpy() if hasattr(current_pose.q, 'cpu') else current_pose.q
                # Randomize xy position, keep z=0.2 (let object drop from height)
            random_angle = np.random.uniform(0, 60) 
            quat = rotation_quaternion_z(random_angle)

            xy = randomization.uniform(
            low=torch.tensor([-0.3, -0.3]), 
            high=torch.tensor([0.2, 0.2]), 
            size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.2  # Fixed height 0.2m for dropping
            
                # Use first environment position (single environment), maintain URDF original pose
            if b > 0:
                self.target_object.set_pose(Pose.create_from_pq(pos[0].cpu().numpy(), quat))

        # ============ Place object (maintain URDF original pose) ============
        self._place_object_with_drop(env_idx)

        # ============ Set object initial joint angles ============
        if hasattr(self, 'target_object') and self.target_object is not None:
            if "init_qpos" in self.object_config and len(self.object_config["init_qpos"])!=0:
                init_qpos = self.object_config["init_qpos"]
                try:
                    # Convert init_qpos to tensor
                    qpos_tensor = torch.tensor(
                        init_qpos, 
                        device=self.target_object.device, 
                        dtype=torch.float32
                    )
                    # Expand to batch size
                    b = len(env_idx)
                    if qpos_tensor.ndim == 1:
                        qpos_batch = qpos_tensor.unsqueeze(0).expand(b, -1)
                    else:
                        qpos_batch = qpos_tensor
                    
                    # Set joint angles
                    self.target_object.set_qpos(qpos_batch)
                    
                    # Run steps for physics stabilization
                    for _ in range(10):
                        self.env.scene.step()
                    
                    if self.verbose:
                        print(f"Set object initial joint angles: {init_qpos}")
                except Exception as e:
                    print(f"Error setting object initial joint angles: {e}")

        # Robot-type-specific initial config.
        if hasattr(self.env, 'robot_uids'):
            robot_uids = self.env.robot_uids
        else:
            robot_uids = "panda"  # default

        # Extend this branch to support other robots.
        if robot_uids in ["panda", "panda_wristcam"]:
            self._initialize_panda_robot(env_idx, b)
        else:
            if self.verbose:
                print(f"Warning: Robot type '{robot_uids}' not specifically configured, using default initialization")

    def _initialize_panda_robot(self, env_idx: torch.Tensor, b: int):
        """
        Initialize Panda robot.

        Args:
            env_idx (torch.Tensor): Environment indices
            b (int): Batch size
        """
        # Panda robot initial joint angles (9 joints: 7 arm joints + 2 gripper joints)
        # This configuration puts the robot in a reasonable starting pose
        qpos = np.array(
            [
                0.011,   # Joint 1: shoulder rotation
                0.188,   # Joint 2: shoulder pitch
                -0.047,  # Joint 3: shoulder roll
                -1.519,  # Joint 4: elbow
                -0.031,  # Joint 5: wrist rotation 1
                1.715,   # Joint 6: wrist pitch
                0.788,   # Joint 7: wrist rotation 2
                0.04,    # Gripper left finger
                0.04,    # Gripper right finger (0.04 = open)
            ]
        )

        # Add random noise to enhance training robustness
        if hasattr(self.env, '_enhanced_determinism') and self.env._enhanced_determinism:
            # Enhanced determinism mode: use batch-level random number generator
            # Each environment has independent random seed
            qpos = (
                self.env._batched_episode_rng[env_idx].normal(
                    0, self.robot_init_qpos_noise, len(qpos)
                )
                + qpos
            )
        else:
            # Standard mode: use global random number generator
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )

        # Ensure gripper stays open (override random noise)
        qpos[:, -2:] = 0.04

        # Reset the robot's joints to the prescribed angles.
        self.env.agent.reset(qpos)

        # Set robot base pose
        # p=[-0.615, 0, 0]: place robot in front of table
        self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

class StandUpSceneBuilder(GraspPartSceneBuilder):
    def __init__(self, env, object_name="bottle", robot_init_qpos_noise=0.02):
        super().__init__(env, object_name, robot_init_qpos_noise)

    def initialize(self, env_idx: torch.Tensor):
        """
        Initialize scene and robot.

        Called at the start of each episode to reset object poses and robot states in the scene.

        Args:
            env_idx (torch.Tensor): Environment indices tensor to initialize
                                   In parallel simulation, multiple environments may run simultaneously

        Functions:
            1. Reset table pose to initial position
            2. Place object lying down at fixed position
            3. Set robot initial joint angles based on robot type
            4. Set robot base pose
        """
        # Batch size = number of envs to initialise.
        b = len(env_idx)
            
        # Reset the table pose.
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )

       
        xy = randomization.uniform(
                low=torch.tensor([-0.05, -0.08]), 
                high=torch.tensor([0.15, 0.12]), 
                size=(b, 2)
            )
        pos = self.target_object.get_pose().p.cpu().numpy()        
        pos[:, :2] = xy
        pos[:, 2] = 0.025 
        
        if b > 0:
            self.target_object.set_pose(Pose.create_from_pq(pos, [ 0.707, -0.0,  0.707, 0.0]))
        
        # Robot-type-specific initial config.
        if hasattr(self.env, 'robot_uids'):
            robot_uids = self.env.robot_uids
        else:
            robot_uids = "panda"  # default

        # Extend this branch to support other robots.
        if robot_uids in ["panda", "panda_wristcam"]:
            self._initialize_panda_robot(env_idx, b)
        else:
            print(f"Warning: Robot type '{robot_uids}' not specifically configured, using default initialization")
class ToggleSwitchSceneBuilder(GraspPartSceneBuilder):
    """
    Common builder for vertical toggle switches.
    Scene builder for toggle switch tasks (standing placement/fixed root link).

    Inherits from GraspPartSceneBuilder.
    Features:
    1. Switch placed vertically (standing up/default orientation)
    2. Root link fixed (fix_root_link=True) - "because it's standing"
    3. Support position and rotation randomization
    """

    def __init__(self, env, object_name="100367", robot_init_qpos_noise=0.02):
        super().__init__(env, object_name, robot_init_qpos_noise)

    # Task-specific defaults: switches stand upright at z=0.10 unless
    # the asset's model_data.json overrides initial_pose explicitly.
    _DEFAULT_TOGGLE_POSE = [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0]
    _DEFAULT_TOGGLE_SCALE = 0.2

    def _convert_config_format(self, raw_config):
        """Convert config format — standing placement, override-friendly."""
        return {
            "scale": raw_config.get("scale", self._DEFAULT_TOGGLE_SCALE),
            "urdf_file": raw_config.get("urdf_file", "mobility.urdf"),
            "initial_pose": raw_config.get("initial_pose", list(self._DEFAULT_TOGGLE_POSE)),
            "material": raw_config.get("material", {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
            }),
            "load_mode": raw_config.get("load_mode"),
            "init_qpos": raw_config.get("init_qpos", []),
            "grasp_parts": raw_config.get("grasp_parts", {}),
        }

    def _build_target_object(self):
        """
        Override parent method to support fix_root_link=True
        (Requirement: ToggleSwitchSceneBuilder has fixed root link)
        """
        object_loader = self.scene.create_urdf_loader()
        scale = self.object_config.get("scale", 1.0)
        urdf_file = self.object_config.get("urdf_file", "mobility.urdf")
        initial_pose_config = self.object_config.get("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        material_config = self.object_config.get("material", {})

        object_loader.fix_root_link = False # Loader attribute, but we set it on build
        object_loader.scale = scale
        object_loader.load_multiple_collisions_from_file = True

        urdf_path = Path(__file__).parent.parent / "assets" / self.object_name / urdf_file
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        applied_urdf_config = sapien_utils.parse_urdf_config(dict(material=material_config))
        sapien_utils.apply_urdf_config(object_loader, applied_urdf_config)

        object_builders = object_loader.parse(str(urdf_path))["articulation_builders"]
        if not object_builders:
            raise RuntimeError(f"No articulation builders found in URDF file {urdf_path}")

        object_builder = object_builders[0]
        object_builder.set_scene_idxs(scene_idxs=[0])

        if len(initial_pose_config) == 7:
            initial_pose = sapien.Pose(p=initial_pose_config[:3], q=initial_pose_config[3:])
        else:
            initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

        object_builder.initial_pose = initial_pose

        # Note: fix_root_link=True here.
        target_object = object_builder.build(
            name=f"{self.object_name}_object",
            fix_root_link=True 
        )
        self.target_object = target_object
        print(f"Successfully loaded toggle switch '{self.object_name}' (Fixed/Standing) from {urdf_path}")
        print(f"Scale: {scale}, Initial pose: {initial_pose}")

    def initialize(self, env_idx: torch.Tensor):
        """Initialize with randomization logic"""
        b = len(env_idx)
        super().initialize(env_idx)
        if hasattr(self, 'target_object') and self.target_object is not None:
            # Sample a random xy position.
            xy = randomization.uniform(
                low=torch.tensor([0.12, -0.12]), 
                high=torch.tensor([0.22, 0.12]),
                size=(b, 2)
            )
            z = randomization.uniform(
                low=torch.tensor([0.12]), 
                high=torch.tensor([0.25]),
                size=(b, 1)
            )
            pos = torch.zeros((b, 3))

            pos[:, :2] = xy
            if z.dim() == 1:
                pos[:, 2] = z
            elif z.dim() == 2 and z.shape[1] == 1:
                pos[:, 2] = z.squeeze(1)
            else:
                raise ValueError(f"Invalid shape of z: {z.shape}")

            p = pos.cpu().numpy()

            # Apply pose for each env
            # Standing placement + small Z-axis random rotation (±30 degrees)
            for i in range(b):
                # Random Z-axis rotation angle (±30 degrees)
                z_angle = np.random.uniform(-np.pi/6, np.pi / 6)
                # Quaternion for Z-axis rotation: [cos(angle/2), 0, 0, sin(angle/2)]
                q = [np.cos(z_angle / 2), 0.0, 0.0, np.sin(z_angle / 2)]
                self.target_object.set_pose(sapien.Pose(p[i], q))
class ToggleSwitchTableSceneBuilder(GraspPartSceneBuilder):
    """
    Common builder for table-top toggle switches.
    Scene builder for toggle switch tasks (lying down/unfixed/table environment).

    Inherits from GraspPartSceneBuilder.
    Features:
    1. Switch placed lying down (buttons facing up)
    2. Root link unfixed (fix_root_link=False)
    3. Support position and rotation randomization
    """

    def __init__(self, env, object_name="100367", robot_init_qpos_noise=0.02):
        super().__init__(env, object_name, robot_init_qpos_noise)

    def _convert_config_format(self, raw_config):
        """Convert config format — switch lying on the table, buttons up."""
        # Task-specific default: lying pose (90° about Y) at z=0.05.
        lying_pose = [0.1, 0.0, 0.05, 0.707, 0.0, 0.707, 0.0]

        return {
            "scale": raw_config.get("scale", 0.2),
            "urdf_file": raw_config.get("urdf_file", "mobility.urdf"),
            "initial_pose": raw_config.get("initial_pose", lying_pose),
            "material": raw_config.get("material", {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
            }),
            "load_mode": raw_config.get("load_mode"),
            "init_qpos": raw_config.get("init_qpos", []),
            "grasp_parts": raw_config.get("grasp_parts", {}),
        }
    
    def _build_target_object(self):
        """
        Override parent method to support fix_root_link=True
        (Requirement: ToggleSwitchTableSceneBuilder has fixed root link)
        """
        object_loader = self.scene.create_urdf_loader()
        scale = self.object_config.get("scale", 1.0)
        urdf_file = self.object_config.get("urdf_file", "mobility.urdf")
        initial_pose_config = self.object_config.get("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        material_config = self.object_config.get("material", {})

        object_loader.fix_root_link = False 
        object_loader.scale = scale
        object_loader.load_multiple_collisions_from_file = True

        urdf_path = Path(__file__).parent.parent / "assets" / self.object_name / urdf_file
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        applied_urdf_config = sapien_utils.parse_urdf_config(dict(material=material_config))
        sapien_utils.apply_urdf_config(object_loader, applied_urdf_config)

        object_builders = object_loader.parse(str(urdf_path))["articulation_builders"]
        if not object_builders:
            raise RuntimeError(f"No articulation builders found in URDF file {urdf_path}")

        object_builder = object_builders[0]
        object_builder.set_scene_idxs(scene_idxs=[0])

        if len(initial_pose_config) == 7:
            initial_pose = sapien.Pose(p=initial_pose_config[:3], q=initial_pose_config[3:])
        else:
            initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

        object_builder.initial_pose = initial_pose

        # Note: fix_root_link=True here.
        target_object = object_builder.build(
            name=f"{self.object_name}_object",
            fix_root_link=True 
        )

        self.target_object = target_object
        print(f"Successfully loaded toggle switch '{self.object_name}' (Fixed/Lying) from {urdf_path}")
        print(f"Scale: {scale}, Initial pose: {initial_pose}")

    def initialize(self, env_idx: torch.Tensor):
        """Initialize with randomization logic"""
        b = len(env_idx)
        super().initialize(env_idx)
        
        if hasattr(self, 'target_object') and self.target_object is not None:
            # Sample a random xy position.
            xy = randomization.uniform(
                low=torch.tensor([-0.15, -0.25]), 
                high=torch.tensor([0.15, 0.20]), 
                size=(b, 2)
            )
            
            p = torch.zeros((b, 3))
            p[:, :2] = xy
            p[:, 2] = 0.05  # Slightly elevated

            p_np = p.cpu().numpy()

            # Base orientation (lying down): 90-degree rotation around Y-axis
            # Plus small Z-axis random rotation (±60 degrees)
            for i in range(b):
                # Random Z-axis rotation angle (±60 degrees)
                z_angle = np.random.uniform(-np.pi / 3, np.pi / 3)
                # Base orientation: 90° about Y.
                base_rot = R.from_euler('y', np.pi / 2)
                # Random Z rotation.
                z_rot = R.from_euler('z', z_angle)
                # Compose: Y rotation first, then Z.
                combined_rot = z_rot * base_rot
                q_scipy = combined_rot.as_quat()  # [x, y, z, w]
                # Convert to sapien format [w, x, y, z]
                q = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]
                self.target_object.set_pose(sapien.Pose(p_np[i], q))


class SlideAlongSceneBuilder(GraspPartSceneBuilder):
    """
    Supports two load/placement strategies inside one SceneBuilder:
    - large: big assets (cabinet/window). Use bbox(min.z)*scale to sit them on the table.
    - small: light assets (bottle/mouse/phone). Drop from z=0.2 and let physics settle them.

    Caller controls the choice via the `object_load_mode` kwarg in main.py / gym.make:
    - "auto" (default): cabinet/window/window3 → large, everything else → small.
    - "large": force the bulky-asset strategy.
    - "small": force the drop-and-settle strategy.
    """

    def __init__(
        self,
        env,
        object_name: str = "cabinet",
        robot_init_qpos_noise: float = 0.02,
        object_load_mode: str = "auto",
    ):
        super().__init__(env, object_name, robot_init_qpos_noise)
        self.object_load_mode = (object_load_mode or "auto").lower().strip()

    # _convert_config_format is inherited from GraspPartSceneBuilder; the
    # base class is now fully data-driven (reads scale / initial_pose /
    # material / load_mode straight from model_data.json), so this subclass
    # needs no override. Per-asset placement values that used to live here
    # belong in assets/<obj>/model_data.json.

    def _resolved_mode(self) -> str:
        """Pick the load mode (small/large) for this asset.

        Resolution order:
          1. explicit ``object_load_mode`` ctor argument (override),
          2. ``load_mode`` field in the asset's model_data.json,
          3. fallback to ``"small"``.
        """
        if self.object_load_mode in ("small", "large"):
            return self.object_load_mode
        declared = self.object_config.get("load_mode")
        if declared in ("small", "large"):
            return declared
        return "small"

    def _snap_z_to_table(self, scale: float, fallback_z: float) -> float:
        """Estimate the bottom-face z from bbox.min and drop to the table top."""
        bbox_path = Path(__file__).parent.parent / "assets" / self.object_name / "bounding_box.json"
        if not bbox_path.exists():
            return float(fallback_z)
        try:
            with open(bbox_path, "r", encoding="utf-8") as f:
                bbox = json.load(f)
            local_min_z = float(bbox["min"][2])
            clearance = 0.002
            return float(0.0 + clearance - local_min_z * float(scale))
        except Exception:
            return float(fallback_z)

    def _build_target_object(self):
        """
        Override: object_load_mode picks fix_root_link + whether to snap to table height at build time.
        """
        mode = self._resolved_mode()

        object_loader = self.scene.create_urdf_loader()
        scale = self.object_config.get("scale", 1.0)
        urdf_file = self.object_config.get("urdf_file", "mobility.urdf")
        initial_pose_config = self.object_config.get(
            "initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        material_config = self.object_config.get(
            "material",
            {"static_friction": 1.0, "dynamic_friction": 1.0, "restitution": 0.0},
        )

        # small: free fall onto table; large: fixed root for stability (won't tip).
        fix_root_link = mode == "large"
        object_loader.fix_root_link = fix_root_link
        object_loader.scale = scale
        object_loader.load_multiple_collisions_from_file = True

        urdf_path = Path(__file__).parent.parent / "assets" / self.object_name / urdf_file
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        applied_urdf_config = sapien_utils.parse_urdf_config(dict(material=material_config))
        sapien_utils.apply_urdf_config(object_loader, applied_urdf_config)

        try:
            object_builders = object_loader.parse(str(urdf_path))["articulation_builders"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse URDF file {urdf_path}: {e}")
        if not object_builders:
            raise RuntimeError(f"No articulation builders found in URDF file {urdf_path}")

        object_builder = object_builders[0]
        object_builder.set_scene_idxs(scene_idxs=[0])

        # Build-time initial pose:
        # - large: snap to table surface (robust against scale changes).
        # - small: keep the configured z (usually 0.2 so it drops).
        if isinstance(initial_pose_config, (list, tuple)) and len(initial_pose_config) == 7:
            p = list(initial_pose_config[:3])
            q = list(initial_pose_config[3:])
        else:
            p = [0.0, 0.0, 0.0]
            q = [1.0, 0.0, 0.0, 0.0]

        if mode == "large":
            p[2] = self._snap_z_to_table(scale=float(scale), fallback_z=float(p[2]))

        object_builder.initial_pose = sapien.Pose(p=p, q=q)

        target_object = object_builder.build(
            name=f"{self.object_name}_object",
            fix_root_link=fix_root_link,
        )
        self.target_object = target_object
        print(f"Successfully loaded target object '{self.object_name}' from {urdf_path}")
        print(f"Scale: {scale}, Initial pose: {object_builder.initial_pose}, fix_root_link={fix_root_link}, mode={mode}")

    def initialize(self, env_idx: torch.Tensor):
        """
        Per-episode initialisation:
        - large: configured xy + table-aligned z (matches the bbox calc).
        - small: configured xy + z=0.2 (drop and let physics settle it).
        """
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )

        mode = self._resolved_mode()

        if hasattr(self, "target_object") and self.target_object is not None:
            initial_pose_config = self.object_config.get(
                "initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            )
            if len(initial_pose_config) == 7:
                quat = initial_pose_config[3:]
                init_xy = torch.tensor(initial_pose_config[:2], dtype=torch.float32)
                fallback_z = float(initial_pose_config[2])
            else:
                current_pose = self.target_object.get_pose()
                quat = (
                    current_pose.q.cpu().numpy()
                    if hasattr(current_pose.q, "cpu")
                    else current_pose.q
                )
                init_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
                fallback_z = 0.0

            pos = torch.zeros((b, 3), dtype=torch.float32)
            pos[:, 0] = init_xy[0]
            pos[:, 1] = init_xy[1]

            if mode == "large":
                scale = float(self.object_config.get("scale", 1.0))
                desired_z = self._snap_z_to_table(scale=scale, fallback_z=fallback_z)
                pos[:, 2] = float(desired_z)
            else:
                # small: drop from height by default.
                pos[:, 2] = 0.2

            if b > 0:
                self.target_object.set_pose(Pose.create_from_pq(pos[0].cpu().numpy(), quat))

        # Let the object settle (free-fall only when small + fix_root_link=False).
        self._place_object_with_drop(env_idx)

        # Robot init reuses the base implementation.
        if hasattr(self.env, "robot_uids"):
            robot_uids = self.env.robot_uids
        else:
            robot_uids = "panda"
        if robot_uids in ["panda", "panda_wristcam"]:
            self._initialize_panda_robot(env_idx, b)
        else:
            print(
                f"Warning: Robot type '{robot_uids}' not specifically configured, using default initialization"
            )


class RotateSceneBuilder(GraspPartSceneBuilder):
    """
    Scene builder for rotation tasks (e.g. turning a knob).
    Notes:
    1. Object position: x ≥ 0.3 (clear of the arm), y randomised, z=0.2 drop height.
    2. Orientation: facing the arm (arm sits at x=-0.615, object faces -x).
    """

    def __init__(self, env, object_name="102901", robot_init_qpos_noise=0.02):
        super().__init__(env, object_name, robot_init_qpos_noise)

# core/scene.py 

    def initialize(self, env_idx: torch.Tensor):
        """Reset object position + orientation for the new episode."""
        b = len(env_idx)
        # 1. Skip super().initialize — its drop logic is irrelevant here.
        # super().initialize(env_idx) 
        
        # 2. Reset the table pose manually so its top is at Z=0.
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )

        if hasattr(self, 'target_object') and self.target_object is not None:
            # 3. Pull from the env's RNG so the seed is respected.
            xy = self.env._batched_episode_rng.uniform(
                low=np.array([0.33, -0.2]), 
                high=np.array([0.37, 0.2]), 
                size=(b, 2)
            )
            xy = torch.from_numpy(xy).to(self.env.device).to(torch.float32)

            pos = torch.zeros((b, 3), device=self.env.device)
            pos[:, :2] = xy
            pos[:, 2] = 0.15  # lower drop height (0.15 m settles cleaner than 0.4 m)
            
            # 4. Random rotation (extend if a fully random orientation is desired).
            random_angle = self.env._batched_episode_rng.uniform(0, 5)  # 0–30° range
            quat = rotation_quaternion_z(random_angle)
            
            # print(f"Set initial pos: {pos[0].cpu().numpy()}")
            self.target_object.set_pose(sapien.Pose(pos[0].cpu().numpy(), quat))
            
        # 5. Let the object drop and settle.
        self._place_object_with_drop(env_idx)
        
        # 6. Initialise the robot manually (since we skipped super().initialize).
        if hasattr(self.env, 'robot_uids'):
            self._initialize_panda_robot(env_idx, b)

        # ============ Set object initial joint angles ============
        if hasattr(self, 'target_object') and self.target_object is not None:
            if "init_qpos" in self.object_config and len(self.object_config["init_qpos"])!=0:
                init_qpos = self.object_config["init_qpos"]
                try:
                    # Convert init_qpos to tensor
                    qpos_tensor = torch.tensor(
                        init_qpos, 
                        device=self.target_object.device, 
                        dtype=torch.float32
                    )
                    # Expand to batch size
                    b = len(env_idx)
                    if qpos_tensor.ndim == 1:
                        qpos_batch = qpos_tensor.unsqueeze(0).expand(b, -1)
                    else:
                        qpos_batch = qpos_tensor
                    
                    # Set joint angles
                    self.target_object.set_qpos(qpos_batch)
                    
                    # Run steps for physics stabilization
                    for _ in range(10):
                        self.env.scene.step()
                    
                    if self.verbose:
                        print(f"Set object initial joint angles: {init_qpos}")
                except Exception as e:
                    print(f"Error setting object initial joint angles: {e}")

        # Robot-type-specific initial config.
        if hasattr(self.env, 'robot_uids'):
            robot_uids = self.env.robot_uids
        else:
            robot_uids = "panda"  # default

        # Extend this branch to support other robots.
        if robot_uids in ["panda", "panda_wristcam"]:
            self._initialize_panda_robot(env_idx, b)
        else:
            if self.verbose:
                print(f"Warning: Robot type '{robot_uids}' not specifically configured, using default initialization")


class DoorSceneBuilder(SceneBuilder):

    def __init__(self, env, object_name="cabinet", robot_init_qpos_noise=0.02):
        """
        Initialise the door scene builder.

        Args:
            env: the env instance
            object_name (str): asset folder under assets/, default "cabinet"
            robot_init_qpos_noise (float): noise applied to the initial joint qpos
        """
        super().__init__(env)
        self.object_name = object_name
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Load object parameters from the config file.
        self._load_object_config()

    def _load_object_config(self):
        """
        Load object parameters from the config file.

        Reads assets/{object_name}/model_data.json and
        converts the legacy schema into the internal one we use here.
        """
        config_path = Path(__file__).parent.parent / "assets" / self.object_name / "model_data.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                # Translate to the internal config schema.
                self.object_config = self._convert_config_format(raw_config)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in config file {config_path}: {e}")
                self.object_config = self._get_default_object_config()
        else:
            print(f"Warning: Config file not found {config_path}, using defaults")
            self.object_config = self._get_default_object_config()

    # Reuses the same schema-default constants as GraspPartSceneBuilder so
    # the two builders share the data-driven contract; per-asset values
    # (door's initial pose etc.) live in assets/<obj>/model_data.json.
    _DEFAULT_INITIAL_POSE = [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    _DEFAULT_MATERIAL = {
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
    }

    def _convert_config_format(self, raw_config):
        """Pure data-driven config translation; see :class:`GraspPartSceneBuilder._convert_config_format`."""
        converted_config = {
            "scale": raw_config.get("scale", 1.0),
            "urdf_file": raw_config.get("urdf_file", "mobility.urdf"),
            "initial_pose": raw_config.get("initial_pose", list(self._DEFAULT_INITIAL_POSE)),
            "material": raw_config.get("material", dict(self._DEFAULT_MATERIAL)),
            "load_mode": raw_config.get("load_mode"),
            "init_qpos": raw_config.get("init_qpos", []),
            "grasp_parts": raw_config.get("grasp_parts", {}),
        }

        return converted_config

    def _get_default_object_config(self):
        """
        Return the hardcoded default object config.

        Returns:
            dict: default object-config dictionary
        """
        return {
            "scale": 1.0,
            "urdf_file": "mobility.urdf",
            "initial_pose": [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, qw, qx, qy, qz]
            "material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0
            }
        }



    def build(self):
        """
        Build every object in the scene.

        Creates + configures every static and dynamic object in the scene:
        - the table (collision + visual);
        - the ground plane;
        - the chosen target object (loaded from URDF).

        Scene convention: the table top is at Z=0 to make math easier.
        """
        # ============ Build the table workspace ============
        self._build_table()

        # ============ Load the target object ============
        self._build_target_object()

        # ============ Cache the scene-actor list ============
        self.scene_objects: list[sapien.Entity] = [self.table, self.ground]

    def _place_object_with_drop(self, env_idx: torch.Tensor):
        """Object placement (currently keeps the URDF's original pose)."""
        if not hasattr(self, 'target_object'):
            return
    
        # Read the current object position.
        current_pose = self.target_object.get_pose()
        current_z = current_pose.p.cpu().numpy().flatten()[-1]  # extract z

        # When the object sits above the table by >0.1 m, run a settling sim.
        if current_z > 0.1:  # table at z=0; anything >0.1 m needs to drop
            print(f"Object at z={current_z:.3f} > 0.1 m; running drop sim...")
            self._simulate_drop(steps=60)  # 60 physics steps
            final_pose = self.target_object.get_pose()
            final_z = final_pose.p.cpu().numpy().flatten()[-1]
            print(f"  drop settled at z={final_z:.3f}")

    def _simulate_drop(self, steps=30):
        """Step physics until the object stops bouncing."""
        scene = self.env.scene
        for _ in range(steps):
            scene.step()

    def _build_table(self):
        """
        Build the table workspace.

        Adds collision + visual for the table; top surface ends up at Z=0.
        """
        # Build an actor for the table.
        builder = self.scene.create_actor_builder()

        # Locate the table mesh file.
        model_dir = Path(__file__).parent.parent / "assets"
        table_model_file = str(model_dir / "table.glb")

        # Table scale factor.
        scale = 1.75 # 1.75

        # Table orientation: 90° about the world Z axis.
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        bottom = 0.9196429  # bottom of the table mesh
        # Use a simple box collider in place of the full mesh for speed.
        # Collider centre is at half the table height.
        # half_size: (length/2, width/2, height/2).
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, bottom / 2]),
            half_size=(2.418 / 2, 1.209 / 2, bottom / 2),
        )

        # Add the GLB-based visual mesh.
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )

        # Place the table so its top surface sits at Z=0.
        # p=[-0.12, 0, -0.9196429]: shift down by the table thickness.
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -bottom], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build as kinematic so the table is fixed against physics forces.
        table = builder.build_kinematic(name="table-workspace")

        # Pre-computed AABB; avoids re-deriving it from the mesh each run.
        # AABB layout: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        aabb = np.array(
            [
                [-0.7402168, -1.2148621, -bottom],  # min corner
                [0.4688596, 1.2030163, 3.5762787e-07],   # max corner 3.5762787e-07
            ]
        )

        # Derive table dimensions from the AABB.
        self.table_length = aabb[1, 0] - aabb[0, 0]  # X dimension
        self.table_width = aabb[1, 1] - aabb[0, 1]   # Y dimension
        self.table_height = aabb[1, 2] - aabb[0, 2]  # Z dimension

        # Cache the table actor.
        self.table = table

        # ============ Build the ground plane ============
        # Pick the ground width based on parallel-scene mode.
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500  # parallel mode needs a larger floor

        # Build the ground at z = -table_height so it sits under the table.
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )

    def _build_target_object(self):
        """
        Dynamically build the target object.

        Loads the URDF chosen by object_name.
        Reads scale, initial pose, material from the asset config.
        """
        # Create a URDF loader for articulated assets.
        object_loader = self.scene.create_urdf_loader()

        # Read parameters from the resolved config.
        scale = self.object_config.get("scale", 1.0)
        urdf_file = self.object_config.get("urdf_file", "mobility.urdf")
        initial_pose_config = self.object_config.get("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        material_config = self.object_config.get("material", {
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0
        })

        # Loader options.
        object_loader.fix_root_link = False  # root may translate freely
        object_loader.scale = scale  # asset scale factor
        object_loader.load_multiple_collisions_from_file = True  # multi-collider URDFs

        # URDF path.
        urdf_path = Path(__file__).parent.parent / "assets" / self.object_name / urdf_file

        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Parse the requested physics material.
        applied_urdf_config = sapien_utils.parse_urdf_config(
            dict(material=material_config)
        )

        # Apply the material to the loader.
        sapien_utils.apply_urdf_config(object_loader, applied_urdf_config)

        # Parse the URDF and collect its articulation builders.
        try:
            object_builders = object_loader.parse(str(urdf_path))["articulation_builders"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse URDF file {urdf_path}: {e}")

        if not object_builders:
            raise RuntimeError(f"No articulation builders found in URDF file {urdf_path}")

        # Use the first articulation builder.
        object_builder = object_builders[0]

        # Pin to scene 0 (extend for parallel-scene support).
        object_builder.set_scene_idxs(scene_idxs=[0])

        # Set the initial pose.
        if len(initial_pose_config) == 7:  # [x, y, z, qw, qx, qy, qz]
            initial_pose = sapien.Pose(
                p=initial_pose_config[:3],
                q=initial_pose_config[3:]
            )
        else:
            # Legacy format / fallback to identity pose.
            initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

        object_builder.initial_pose = initial_pose

        # Build with fix_root_link=True so the door doesn't fall over.
        # Note: fix_root_link only fixes the base.
        # link_1 (frame) is fixed via a fixed joint to base,
        # but link_0 (door) and link_2 (handle) are revolute and free to rotate.
        target_object = object_builder.build(
            name=f"{self.object_name}_object",
            fix_root_link=True
        )

        # Cache the target object.
        self.target_object = target_object
        
        # Set the revolute joints to passive mode so the door swings freely.
        # joint_0: door hinge (revolute) — link_0 ↔ link_1
        # joint_2: handle latch (revolute) — link_2 ↔ link_0
        # Pattern matches PartAnnotator.py: zero stiffness + high damping.
        print("Configuring door/handle revolute joints for passive rotation...")
        for joint in target_object.get_joints():
            joint_name = joint.name
            joint_type = joint.type
             
            # Only touch revolute joints.
            if joint_type == "revolute":
                print(f"  configuring joint: {joint_name} (type: {joint_type})")
                try:
                    # Zero stiffness + high damping → free rotation with drag.
                    # stiffness=0 means the joint isn't driven shut.
                    joint.set_drive_properties(stiffness=0.0, damping=1000.0)
                    joint.set_friction(0.0)
                    print(f"    OK: stiffness=0, damping=1000, friction=0")
                except Exception as e:
                    print(f"    FAILED: {e}")
        

        print(f"Successfully loaded target object '{self.object_name}' from {urdf_path}")
        print(f"Scale: {scale}, Initial pose: {initial_pose}")

    def initialize(self, env_idx: torch.Tensor):
        """
        Initialise scene + robot.

        Called at the start of every episode.

        Args:
            env_idx (torch.Tensor): which envs to re-init
                                   (parallel sims may pass multiple)

        Steps:
            1. Reset the table pose.
            2. Place the object and let it drop to the table.
            3. Initialise robot joints based on robot type.
            4. Set the robot base pose.
        """
        # Batch size = number of envs to initialise.
        b = len(env_idx)

        # Reset the table pose.
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )

        # ============ Place the object (keeps URDF base pose) ============
        self._place_object_with_drop(env_idx)

        # Robot-type-specific initial config.
        if hasattr(self.env, 'robot_uids'):
            robot_uids = self.env.robot_uids
        else:
            robot_uids = "panda"  # default

        # Extend this branch to support other robots.
        if robot_uids in ["panda", "panda_wristcam"]:
            self._initialize_panda_robot(env_idx, b)
        else:
            print(f"Warning: Robot type '{robot_uids}' not specifically configured, using default initialization")

    def _initialize_panda_robot(self, env_idx: torch.Tensor, b: int):
        """
        Initialise the Panda robot.

        Args:
            env_idx (torch.Tensor): env indices
            b (int): batch size
        """
        # Panda initial qpos: 7 arm joints + 2 finger joints.
        # This pose is a clean starting configuration.
        qpos = np.array(
            [
                0.011,  # joint 1: shoulder yaw
                0.188,  # joint 2: shoulder pitch
                -0.047,  # joint 3: shoulder roll
                -1.519,  # joint 4: elbow
                -0.031,  # joint 5: wrist roll 1
                1.715,   # joint 6: wrist pitch
                0.788,   # joint 7: wrist roll 2
                0.04,    # left finger
                0.04,    # right finger (0.04 = open)
            ]
        )

        # Add randomisation for training robustness.
        if hasattr(self.env, '_enhanced_determinism') and self.env._enhanced_determinism:
            # Deterministic mode: use the batched RNG.
            # Each env gets an independent seed.
            qpos = (
                self.env._batched_episode_rng[env_idx].normal(
                    0, self.robot_init_qpos_noise, len(qpos)
                )
                + qpos
            )
        else:
            # Default mode: use the global RNG.
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )

        # Force the gripper open regardless of qpos noise.
        qpos[:, -2:] = 0.04

        # Reset the robot's joints to the prescribed angles.
        self.env.agent.reset(qpos)

        # Set the robot base pose.
        # p=[-0.615, 0, 0]: place the arm in front of the table.
        self.env.agent.robot.set_pose(sapien.Pose([0.3973, 0, 0]))
