import logging
import argparse
import readline
import importlib
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Literal
import transforms3d as t3d
from transforms3d import quaternions
from threading import Thread, Lock
import trimesh
import trimesh.bounds
import json
import re
import os
from mani_skill.utils.structs import Actor
import sys
from utils.actor_utils import graspActor, graspArticulationActor
sys.path.append(".")

import sapien
from sapien.render import set_global_config

render_pause = False


class BaseViewer:
    scene: sapien.Scene
    viewer: sapien.utils.Viewer

    actor: graspActor
    modelid: str
    modelname: str
    config_path: Path
    EMPTY_CONFIG: dict
    POINTS: list[tuple[str, str]]

    def __init__(self):
        # create scene and viewer
        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 250)

        # initialize viewer with camera position and orientation
        self.viewer = None
        self.reset()
    
    def open_viewer(self):
        if self.viewer is not None and not self.viewer.closed:
            return 
        self.viewer = self.scene.create_viewer()
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_pose(pose=sapien.Pose(
            [-0.0096987, -0.19846, 0.0955636],
            [0.71241, -0.118063, 0.123576, 0.680634],
        ))

    def reset(self):
        self.scene.clear()
        self.open_viewer()
        # ground
        self.scene.add_ground(0)

        # lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        shadow = True
        # default spotlight angle and intensity
        direction_lights = [[[0, 0.5, -1], [0.5, 0.5, 0.5]]]
        for direction_light in direction_lights:
            self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow)
        # default point lights position and intensity
        point_lights = [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]]
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

        self.update_render()

    @staticmethod
    def trans_mat(to_mat: np.ndarray, from_mat: np.ndarray):
        to_rot = to_mat[:3, :3]
        from_rot = from_mat[:3, :3]
        rot_mat = to_rot @ from_rot.T

        trans_mat = to_mat[:3, 3] - from_mat[:3, 3]

        result = np.eye(4)
        result[:3, :3] = rot_mat
        result[:3, 3] = trans_mat
        result = np.where(np.abs(result) < 1e-5, 0, result)
        return result

    @staticmethod
    def trans_base(
            init_pose_mat: np.ndarray,
            now_base_mat: np.ndarray,
            init_base_mat: np.ndarray = np.eye(4),
    ):
        now_pose_mat = np.eye(4)
        base_trans_mat = BaseViewer.trans_mat(now_base_mat, init_base_mat)
        now_pose_mat[:3, :3] = (base_trans_mat[:3, :3] @ init_pose_mat[:3, :3] @ base_trans_mat[:3, :3].T)
        now_pose_mat[:3, 3] = base_trans_mat[:3, :3] @ init_pose_mat[:3, 3]

        p = now_pose_mat[:3, 3] + now_base_mat[:3, 3]
        q_mat = now_pose_mat[:3, :3] @ now_base_mat[:3, :3]
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))
    
    def build_grasp_pose_visual(self, pose: sapien.Pose, name: str = "grasp_visual"):
        """Build a two-finger gripper visualization for grasp poses"""
        global render_pause
        grasp_pose_visual_width = 0.01
        grasp_width = 0.05
        
        builder = self.scene.create_actor_builder()
        builder.set_name(name)
        self.update_render()
        # Center sphere (blue)
        builder.add_sphere_visual(
            pose=sapien.Pose(p=[0, 0, 0.0]),
            radius=grasp_pose_visual_width,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
        )
        
        # Gripper base (green box at -Z)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.08]),
            half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        
        # Gripper width indicator (green box along Y)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.05]),
            half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        
        # Left finger (blue box)
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0.03 - grasp_pose_visual_width * 3, grasp_width + grasp_pose_visual_width, 0.03 - 0.05],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
        )
        
        # Right finger (red box)
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0.03 - grasp_pose_visual_width * 3, -grasp_width - grasp_pose_visual_width, 0.03 - 0.05],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
        )
        builder.set_initial_pose(pose)
        self.update_render()
        render_pause = True
        builder.build_kinematic(name=name)
        render_pause = False
        self.update_render()
        # return grasp_pose_visual

    def clear_scene(self):
        global render_pause
        render_pause = True
        self.scene.clear()
        render_pause = False
        self.update_render()

    def update_render(self):
        global render_pause
        if not render_pause and not self.viewer.closed:
            self.scene.update_render()
            self.viewer.render()

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.actor.config, f, ensure_ascii=False, indent=4)
        logging.info(f"Config saved to {self.config_path}")
    
    def save_grasp_poses(self):
        """Save only contact grasp poses with id and description to a separate JSON file"""
        grasp_file = self.config_path.parent / "grasp_poses.json"
        
        grasp_parts = self.actor.config.get("grasp_parts", {})
        
        if not grasp_parts:
            logging.info("No grasp parts to save")
            return
        
        grasp_data = {
            "object_name": self.config_path.parent.name,
            "scale": self.actor.config.get("scale", 1.0),
            "grasp_parts": grasp_parts
        }
        
        total_grasps = sum(len(grasps) for grasps in grasp_parts.values())
        
        with open(grasp_file, "w", encoding="utf-8") as f:
            json.dump(grasp_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Grasp poses saved to {grasp_file} ({len(grasp_parts)} parts, {total_grasps} total grasps)")

    def main(self, pose, modelname, modelid, inherit_config: dict = None):
        global render_pause
        self.modelid = modelid
        self.modelname = modelname
        ...

    def load_actor(self, pose, inherit_config):
        ...

    def update_config(self):
        ...

    def visualize(self):
        ...

class URDFViewer(BaseViewer):
    EMPTY_CONFIG = {
        "scale": 1.0,  # Scale
        "transform_matrix": np.eye(4).tolist(),
        # Expected loading position to model actual pose transformation matrix
        "init_qpos": [],  # Initial joint state
        # Grasp parts: key = part name (e.g., "handle", "door"), value = list of grasp poses
        "grasp_parts": {
            # Example:
            # "handle": [
            #     {"id": 0, "matrix": [[...]], "base": "link_1", "description": "top grasp"},
            #     {"id": 1, "matrix": [[...]], "base": "link_1", "description": "side grasp"}
            # ]
        }
    }
    
    POINTS = [
        ("contact_points", "contact"),
    ]

    def __init__(self):
        super().__init__()

    def main(self, pose, modeldir: Path):
        """Launch viewer with the given URDF folder.

        modeldir: absolute/relative path to the folder containing mobility.urdf
        modelname: kept as the last folder name for display/logging
        """
        self.modeldir = Path(modeldir)
        self.modelname = self.modeldir.name  # use last folder name for display
        self.modelid = None
        self.reset()
        self.load_actor(pose,inherit_config=None)
        self.visualize()

        self.active = True

        def render():
            while self.active and not self.viewer.closed:
                self.update_render()
            self.clear_scene()

        self.render = Thread(target=render)
        self.render.start()
        self.console()
        self.active = False
        self.render.join()

    def __del__(self):
        self.active = False
        if hasattr(self, 'render'):
            self.render.join()
        self.scene.clear()
        if self.viewer is not None and not self.viewer.closed:
            self.viewer.close()

    def load_actor(self, pose, inherit_config):
        modeldir = self.modeldir  # full path
        self.config_path = modeldir / "model_data.json"

        if self.config_path.exists():
            try:
                actor_config = json.load(open(self.config_path, "r", encoding="utf-8"))
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in {self.config_path}, using empty config.")
                actor_config = None
        else:
            actor_config = None

        if actor_config is None:
            if inherit_config is None:
                actor_config = deepcopy(self.EMPTY_CONFIG)
            else:
                actor_config = deepcopy(inherit_config)
        else:
            if inherit_config is not None:
                actor_config = deepcopy(inherit_config)
            else:
                actor_config = actor_config

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = actor_config["scale"]
        loader.fix_root_link = False
        loader.load_multiple_collisions_from_file = True
        actor: sapien.physx.PhysxArticulation = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]
        actor.set_name(f"{self.modelname}")
        actor.set_pose(self.get_real_pose(pose, np.array(actor_config.get("transform_matrix", np.eye(4)))))

        self.actor = graspArticulationActor(actor, actor_config)
        for joint in self.actor.actor.get_joints():
            joint.set_drive_properties(
                damping=1000,
                stiffness=0,
            )
        if (self.actor.config.get("init_qpos") is not None and len(self.actor.config["init_qpos"]) > 0):
            self.actor.set_qpos(np.array(self.actor.config["init_qpos"]))
        return True

    def get_real_pose(self, pose: sapien.Pose, trans_matrix):
        pose_matrix = pose.to_transformation_matrix()
        return sapien.Pose(
            p=pose_matrix[:3, 3] + trans_matrix[:3, 3],
            q=t3d.quaternions.mat2quat(trans_matrix[:3, :3] @ pose_matrix[:3, :3]),
        )

    def visualize(self):
        grasp_parts = self.actor.config.get("grasp_parts", {})
        for part_name, grasps in grasp_parts.items():
            for grasp in grasps:
                grasp_id = grasp.get("id", 0)
                base = grasp.get("base", "base")
                self.build_grasp_pose_visual(
                    pose=self.get_grasp_pose(grasp),
                    name=f"contact_{grasp_id}<{part_name}><{base}>"
                )
        self.update_render()
    
    def get_grasp_pose(self, grasp: dict) -> sapien.Pose:
        """Convert grasp matrix to world pose"""
        matrix = np.array(grasp.get("matrix", np.eye(4)))
        base_name = grasp.get("base", "base")
        
        # Get base link pose
        base_link = self.get_link(base_name)
        # base_pose = base_link.get_entity_pose().to_transformation_matrix()
        base_pose = base_link.get_pose().to_transformation_matrix()

        # Transform grasp pose to world coordinates
        world_matrix = base_pose @ (matrix * self.actor.config.get("scale", 1.0))
        world_matrix[:3, 3] = base_pose[:3, :3] @ (matrix[:3, 3] * self.actor.config.get("scale", 1.0)) + base_pose[:3, 3]
        
        return sapien.Pose(
            world_matrix[:3, 3],
            t3d.quaternions.mat2quat(world_matrix[:3, :3])
        )

    def get_link(self, link_name: str):
        for link in self.actor.actor.get_links():
            if link.get_name() == link_name:
                return link
        return self.actor.actor

    def get_link_dict(self):
        link_dict = {}
        for link in self.actor.actor.get_links():
            link_dict[link.get_name()] = link
        return link_dict

    def get_base_name(self, point_name: str):
        res = re.search(r'(.*?)<(.*?)>', point_name)
        return res.group(2) if res else None

    def get_id(self, point_name: str):
        res = re.search(r'_(\d+)', point_name)
        return int(res.group(1)) if res else None

    def update_config(self, save: bool = False):
        config = deepcopy(self.EMPTY_CONFIG)
        config.update(self.actor.config)
        config["grasp_parts"] = {}

        link_dict = self.get_link_dict()

        def get_mat(entity: sapien.Entity, base: str):
            nonlocal config, link_dict
            # mat = entity.get_entity_pose().to_transformation_matrix()
            mat = entity.get_pose().to_transformation_matrix()

            base_link = link_dict.get(base, self.actor)
            # base_mat = base_link.get_entity_pose().to_transformation_matrix()
            base_mat = base_link.get_pose().to_transformation_matrix()

            p = base_mat[:3, :3].T @ (mat[:3, 3] - base_mat[:3, 3])
            mat[:3, 3] = p / config["scale"]
            mat[:3, :3] = base_mat[:3, :3].T @ mat[:3, :3]
            return np.around(mat, 5)

        # Collect all contact points and organize by part name
        for entity in self.scene.get_all_actors():
            e_name = entity.get_name()
            if e_name.startswith("contact_"):
                # Parse: contact_0<part_name><base> (all relative to base)
                match = re.search(r'contact_(\d+)<([^>]+)><([^>]+)>', e_name)
                if match:
                    grasp_id = int(match.group(1))
                    part_name = match.group(2)
                    base_name = match.group(3)
                    
                    # Ensure it's always relative to "base"
                    if base_name != "base":
                        logging.warning(f"Grasp point {e_name} is not relative to base, skipping")
                        continue
                    
                    if part_name not in config["grasp_parts"]:
                        config["grasp_parts"][part_name] = []
                    
                    config["grasp_parts"][part_name].append({
                        "id": grasp_id,
                        "matrix": get_mat(entity, base_name).tolist(),
                        "base": base_name,
                        "description": ""
                    })

        self.actor.config = config
        if save:
            self.save_config()
            self.save_grasp_poses()

    def reset_scale(self, scale):
        if not isinstance(scale, float) \
            and not isinstance(scale, int):
            scale = float(scale[0])
        self.actor.config["scale"] = scale
        self.update_config()
        logging.info("Reloading scene, please wait for about 10 seconds...")
        self.reset()
        self.load_actor(self.actor.get_pose(), self.actor.config)
        self.visualize()

    @staticmethod
    def parse_point(cmd: str, req_id: bool = True):
        """Parse command for point operations (simplified for base-only points)
        
        Args:
            cmd: command string
            req_id: whether ID is required
            
        Returns:
            If req_id: (part_name, grasp_id)
            If not req_id: (part_name,)
        """
        if cmd.strip() == '':
            cmd = input("  >> <part_name> [grasp_id]: ")
        
        parts = cmd.strip().split(" ")
        
        if req_id:
            try:
                if len(parts) >= 2:
                    part_name, grasp_id = parts[0], int(parts[1])
                else:
                    return None, None
            except (IndexError, ValueError):
                return None, None
            return part_name, grasp_id
        else:
            if len(parts) >= 1:
                return parts[0]  # part_name only
            return None

    def get_points(self, part_name: str = None) -> list[sapien.Entity]:
        """Get all contact points, optionally filtered by part name"""
        points = []
        for entity in self.scene.get_all_actors():
            e_name = entity.get_name()
            if e_name.startswith("contact_"):
                if part_name is None:
                    points.append(entity)
                else:
                    # Check if entity belongs to this part
                    match = re.search(r'contact_\d+<([^>]+)>', e_name)
                    if match and match.group(1) == part_name:
                        points.append(entity)
        return points

    def get_next_id(self, part_name: str) -> int:
        """Get next available ID for a part"""
        points = self.get_points(part_name)
        max_id = -1
        for p in points:
            match = re.search(r'contact_(\d+)<', p.get_name())
            if match:
                max_id = max(max_id, int(match.group(1)))
        return max_id + 1

    def edit_grasp_pose(self, part_name: str, grasp_id: int):
        """Interactive editing mode for grasp pose with keyboard control"""
        # Find the entity
        target_entity = None
        for entity in self.scene.get_all_actors():
            e_name = entity.get_name()
            match = re.search(r'contact_(\d+)<([^>]+)><([^>]+)>', e_name)
            if match and int(match.group(1)) == grasp_id and match.group(2) == part_name:
                target_entity = entity
                break
        
        if target_entity is None:
            logging.warning(f"Grasp point {part_name} {grasp_id} not found")
            return
        
        logging.info("=== Edit Mode ===")
        logging.info("Position: W/S (forward/back), A/D (left/right), Q/E (up/down)")
        logging.info("Rotation: I/K (pitch), J/L (yaw), U/O (roll)")
        logging.info("Speed: +/- (increase/decrease step size)")
        logging.info("Press ENTER to save, ESC to cancel")
        
        step_trans = 0.01  # 1cm
        step_rot = np.radians(5)  # 5 degrees
        
        import sys, termios, tty
        
        def get_key():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        
        original_pose = target_entity.get_pose()
        
        while True:
            current_pose = target_entity.get_pose()
            current_mat = current_pose.to_transformation_matrix()
            
            key = get_key()
            
            if key == '\r' or key == '\n':  # Enter
                logging.info(f"Saved new pose for {part_name} grasp {grasp_id}")
                break
            elif key == '\x1b':  # ESC
                target_entity.set_pose(original_pose)
                logging.info("Cancelled, restored original pose")
                return
            
            # Translation
            delta_pos = np.zeros(3)
            if key == 'w': delta_pos[0] = step_trans  # forward
            elif key == 's': delta_pos[0] = -step_trans  # backward
            elif key == 'a': delta_pos[1] = step_trans  # left
            elif key == 'd': delta_pos[1] = -step_trans  # right
            elif key == 'q': delta_pos[2] = step_trans  # up
            elif key == 'e': delta_pos[2] = -step_trans  # down
            
            # Rotation (in local frame)
            delta_rot = np.zeros(3)
            if key == 'i': delta_rot[1] = step_rot  # pitch up
            elif key == 'k': delta_rot[1] = -step_rot  # pitch down
            elif key == 'j': delta_rot[2] = step_rot  # yaw left
            elif key == 'l': delta_rot[2] = -step_rot  # yaw right
            elif key == 'u': delta_rot[0] = step_rot  # roll left
            elif key == 'o': delta_rot[0] = -step_rot  # roll right
            
            # Speed control
            if key == '+' or key == '=':
                step_trans *= 1.5
                step_rot *= 1.5
                logging.info(f"Step size: trans={step_trans:.4f}m, rot={np.degrees(step_rot):.1f}°")
                continue
            elif key == '-':
                step_trans /= 1.5
                step_rot /= 1.5
                logging.info(f"Step size: trans={step_trans:.4f}m, rot={np.degrees(step_rot):.1f}°")
                continue
            
            # Apply transformation
            if np.any(delta_pos):
                new_pos = current_mat[:3, 3] + current_mat[:3, :3] @ delta_pos
                target_entity.set_pose(sapien.Pose(new_pos, current_pose.q))
            
            if np.any(delta_rot):
                rot_delta = t3d.euler.euler2mat(delta_rot[0], delta_rot[1], delta_rot[2])
                new_rot = current_mat[:3, :3] @ rot_delta
                target_entity.set_pose(sapien.Pose(current_mat[:3, 3], t3d.quaternions.mat2quat(new_rot)))

    def console(self):
        global render_pause
        modified = 0
        try:
            while not self.viewer.closed:
                cmd = input("Input command: ")
                if self.viewer.closed:
                    logging.warning("Viewer has been closed manually.")
                    cmd = input("Please choose to reopen, exit with save or exit without save: (r/s/e) ")
                    cmd = cmd.strip().lower()
                    if cmd in ['r', 'reopen']:
                        self.open_viewer()
                    if cmd in ['s', 'save']:
                        self.update_config(True)
                        break
                    if cmd in ['e', 'exit']:
                        break
                
                modified += 1
                if cmd == "save":
                    self.update_config(True)
                    modified = 0
                elif cmd[:6] == "resize":
                    """
                    Usage:
                        resize <size>: Synchronize the scaling of all three axes of the object
                    Example:
                        resize 0.1
                    """
                    args = cmd[7:].strip().split(" ")
                    size = float(args[0])
                    self.reset_scale(size)
                    modified = 0
                elif cmd == "qpos":
                    """
                    Get current joint state
                    """
                    qpos = self.actor.get_qpos()
                    self.actor.config["init_qpos"] = qpos.tolist()
                    print("Current joint state:", qpos)
                    self.update_config(True)

                elif cmd == "info":
                    """Display URDF model information and dimensions"""
                    # 读取边界框文件
                    bbox_file = self.config_path.parent / "bounding_box.json"
                    if bbox_file.exists():
                        try:
                            bbox = json.load(open(bbox_file, "r"))
                            min_bounds = np.array(bbox["min"])
                            max_bounds = np.array(bbox["max"])
                            dimensions = max_bounds - min_bounds
                            
                            current_scale = self.actor.config.get("scale", 1.0)
                            actual_dimensions = dimensions * current_scale
                            
                            logging.info("=== URDF Model Information ===")
                            logging.info(f"Model: {self.modelname}")
                            logging.info(f"Scale: {current_scale}")
                            logging.info("")
                            logging.info("Original dimensions (scale=1.0):")
                            logging.info(".3f")
                            logging.info(".3f")
                            logging.info(".3f")
                            logging.info("")
                            logging.info("Current dimensions (with scale applied):")
                            logging.info(".3f")
                            logging.info(".3f")
                            logging.info(".3f")
                            logging.info("")
                            logging.info(".3f")
                            
                            # 显示关节信息
                            num_joints = len(self.actor.actor.get_joints())
                            num_links = len(self.actor.actor.get_links())
                            logging.info(f"Joints: {num_joints}, Links: {num_links}")
                            
                        except Exception as e:
                            logging.warning(f"Could not read bounding box file: {e}")
                    else:
                        logging.warning("No bounding box file found")
                        # 备选：显示关节信息
                        num_joints = len(self.actor.actor.get_joints())
                        num_links = len(self.actor.actor.get_links())
                        current_scale = self.actor.config.get("scale", 1.0)
                        logging.info(f"Model: {self.modelname} (Scale: {current_scale})")
                        logging.info(f"Joints: {num_joints}, Links: {num_links}")


                elif cmd[:4] == "mass":
                    '''
                    Set joint mass
                    '''
                    mass = cmd[5:].split(' ')
                    links = [(link.get_name(), link) for link in self.actor.actor.get_links()]
                    if len(mass) != len(links):
                        logging.warning(f"Mass list length({len(mass)}) does not match link count({len(links)}).")
                        continue
                    links.sort(key=lambda x: x[0])
                    self.actor.config['mass'] = {}
                    idx = 0
                    for name, link in links:
                        if name == 'base': continue
                        self.actor.config['mass'][name] = float(mass[idx])
                        idx += 1
                    self.save_config()
                elif cmd[:6] == "create":
                    """
                    Usage:
                        create <part_name>: Create a new grasp point for a part (always relative to base)
                    Example:
                        create handle
                        create door
                    """
                    part_name = cmd[7:].strip()
                    if not part_name:
                        logging.warning("Please specify part name")
                        continue

                    # All grasp points are always relative to "base"
                    base_link = self.get_link("base")
                    if base_link is None:
                        logging.warning("Base link 'base' not found in URDF.")
                        continue
                    
                    grasp_id = self.get_next_id(part_name)
                    print(grasp_id)
                    print(part_name)
                    self.build_grasp_pose_visual(
                        self.actor.get_pose(), 
                        name=f"contact_{grasp_id}<{part_name}><base>"
                    )
                    logging.info(f"Successfully created grasp {grasp_id} for part '{part_name}' (relative to base)")
                    self.update_render()
                elif cmd[:4] == "edit":
                    """
                    Usage:
                        edit <part_name> <grasp_id>: Enter interactive editing mode for a grasp point
                    Example:
                        edit handle 0: Edit handle's grasp 0 with keyboard controls
                    """
                    part_name, grasp_id = self.parse_point(cmd[5:], req_id=True)
                    if part_name is None or grasp_id is None:
                        logging.warning("Invalid part name or grasp id.")
                        continue
                    self.edit_grasp_pose(part_name, grasp_id)
                elif cmd[:5] == "clone":
                    """
                    Usage:
                        clone <part_name> <grasp_id>: Clone a grasp point in place
                    Example:
                        clone handle 0: Clones handle's grasp 0 to create a new grasp point
                    """
                    part_name, grasp_id = self.parse_point(cmd[5:], req_id=True)
                    if part_name is None or grasp_id is None:
                        logging.warning("Invalid part name or grasp id.")
                        continue

                    for entity in self.scene.get_all_actors():
                        e_name = entity.get_name()
                        match = re.search(r'contact_(\d+)<([^>]+)><([^>]+)>', e_name)
                        if match and int(match.group(1)) == grasp_id and match.group(2) == part_name:
                            # Verify it's relative to base
                            if match.group(3) != "base":
                                logging.warning(f"Source grasp {grasp_id} is not relative to base, cannot clone")
                                break
                            
                            new_id = self.get_next_id(part_name)
                            self.build_grasp_pose_visual(
                                entity.get_pose(), 
                                name=f"contact_{new_id}<{part_name}><base>"
                            )
                            logging.info(f"Successfully cloned {part_name} grasp {grasp_id} to {new_id}")
                            break
                elif cmd[:6] == "remove":
                    """
                    Usage:
                        remove <part_name> <grasp_id>: Remove a specific grasp point
                        remove <part_name>: Remove all grasp points for a part
                    Example:
                        remove handle 0
                        remove handle
                    """
                    parts = cmd[6:].strip().split(" ")
                    if len(parts) == 0:
                        logging.warning("Please specify part name")
                        continue
                    
                    part_name = parts[0]
                    grasp_id = int(parts[1]) if len(parts) > 1 else None
                    
                    removed = []
                    for entity in list(self.scene.get_all_actors()):
                        e_name = entity.get_name()
                        match = re.search(r'contact_(\d+)<([^>]+)>', e_name)
                        if match and match.group(2) == part_name:
                            if grasp_id is None or int(match.group(1)) == grasp_id:
                                render_pause = True
                                self.scene.remove_actor(entity)
                                render_pause = False
                                removed.append(match.group(1))
                                if grasp_id is not None:
                                    break
                    
                    if removed:
                        logging.info(f"Successfully removed {part_name} grasp(s): {', '.join(removed)}")
                    else:
                        logging.warning(f"No matching grasp found for {part_name}")
                elif cmd[:4] == "list":
                    """
                    Usage:
                        list: List all parts and their grasp counts
                        list <part_name>: List all grasps for a specific part
                    Example:
                        list
                        list handle
                    """
                    part_name = cmd[5:].strip() if len(cmd) > 5 else None
                    
                    if part_name:
                        points = self.get_points(part_name)
                        if not points:
                            logging.info(f"No grasps found for part '{part_name}'")
                        else:
                            logging.info(f"Part '{part_name}' has {len(points)} grasp(s):")
                            for p in points:
                                match = re.search(r'contact_(\d+)<([^>]+)><([^>]+)>', p.get_name())
                                if match:
                                    logging.info(f"  - Grasp {match.group(1)} on {match.group(3)}")
                    else:
                        # List all parts
                        parts_dict = {}
                        for entity in self.scene.get_all_actors():
                            print(entity.get_name())
                            match = re.search(r'contact_\d+<([^>]+)>', entity.get_name())
                            if match:
                                part = match.group(1)
                                parts_dict[part] = parts_dict.get(part, 0) + 1
                        
                        if not parts_dict:
                            logging.info("No grasp parts defined yet")
                        else:
                            logging.info(f"Grasp parts ({len(parts_dict)} total):")
                            for part, count in sorted(parts_dict.items()):
                                logging.info(f"  - {part}: {count} grasp(s)")
                elif cmd == "exit":
                    if modified > 1:
                        cmd = input(
                            f'You have made {modified-1} changes without save, do you want to save them? (y/n/others to abort)'
                        )
                        if cmd.strip().lower() == 'y':
                            self.update_config(True)
                            break
                        elif cmd.strip().lower() == 'n':
                            break
                        else:
                            logging.info("Operation has been aborted.")
                    else:
                        break
                else:
                    modified -= 1
                    if cmd != 'help':
                        logging.info(f"Unknown command: {cmd}")
                    help_info = ""
        except KeyboardInterrupt:
            pass

def auto_loader(folder_path: str):
    model_dir = Path(folder_path)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Check if it's a URDF folder (contains mobility.urdf)
    urdf_file = model_dir / "mobility.urdf"
    if urdf_file.exists():
        logging.info(f"<URDF> Found URDF file: {urdf_file}")
        return URDFViewer(), model_dir, sapien.Pose([0, 0, 0], [1, 0, 0, 0])
    else:
        raise FileNotFoundError(f"No mobility.urdf found in {folder_path}")

def main(folder_path: str):
    try:
        viewer, model_dir, init_pose = auto_loader(folder_path)
    except Exception as e:
        logging.error(f"Failed to load model from {folder_path}: {e}")
        return

    # os.environ["MODEL_NAME"] = str(model_dir)
    # os.environ["MODEL_ID"] = "None"

    try:
        logging.info(f'Annotating {model_dir.name}')
        viewer.main(
            pose=init_pose,
            modeldir=model_dir)
        
    except KeyboardInterrupt:
        logging.info("Annotation interrupted by user")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[{levelname:^8}] {message}', style="{")
    parser = argparse.ArgumentParser(description="URDF Annotation Tool")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing mobility.urdf")
    args = parser.parse_args()
    main(args.folder_path)