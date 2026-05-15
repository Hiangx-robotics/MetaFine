import numpy as np
import sapien
import sapien.core as sapien
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from scipy.spatial.transform import Rotation as R
from utils import get_grasp_pose_from_config, compute_smart_pregrasp_pose, plan_arc_path
import transforms3d as t3d
from typing import Sequence, Optional
import trimesh
from transforms3d.euler import euler2quat
import os
from PIL import Image
from utils.grasp_compute import get_grasp_list, get_grasp_pose_from_config
import time
from utils.evaluator import *
from core.skill_registry import register_skill

@register_skill("align_to_part", affordances=[], group="atomic", part_arg=None, phase="interaction",
                 description="Move TCP to a specified pose, no contact.")
def align_to_part(
    planner,
    *,
    x, y, z,
    rx, ry, rz,
    pregrasp_height_offset=0.2,
    min_pregrasp_height=0.3,
    retreat_distance=0.03,   # key knob: how far to retreat along world Z
    verbose=True,
):
    """
    Drive the gripper to an align-to-part pose in one shot.

    Semantics:
    - close the gripper at the start;
    - retreat (x, y, z) by ``retreat_distance`` along the world Z axis;
    - use that retreated pose as the final align target;
    - never touch the reference pose, never make contact.

    Args:
        planner: PandaArmMotionPlanningSolver.
        x, y, z: reference align position (world frame).
        rx, ry, rz: align orientation (Euler XYZ in radians).
        pregrasp_height_offset: pre-align z lift relative to current TCP.
        min_pregrasp_height: minimum pre-align z.
        retreat_distance: retreat distance along world Z (default 5 cm).
        verbose: emit per-step logs.

    Returns:
        bool: True iff the motion completed.
    """

    agent = planner.env.agent

    # -------------------------
    # 0. Close the gripper up front.
    # -------------------------
    if verbose:
        print("[AlignSkill] init: closing gripper")

    planner.close_gripper()

    # -------------------------
    # 1. Compose the final align target pose (single shot).
    # -------------------------
    quat = R.from_euler("xyz", [rx, ry, rz]).as_quat()  # (x,y,z,w)

    align_target_pose = sapien.Pose(
        p=[x, y, z + retreat_distance],   # already retreated by retreat_distance
        q=[quat[3], quat[0], quat[1], quat[2]],  # sapien: wxyz
    )

    # Snapshot the current TCP pose.
    current_tcp_pose = agent.tcp.pose
    current_pos = current_tcp_pose.p.cpu().numpy()[0]

    if verbose:
        print(f"[AlignSkill] current TCP position: {current_pos}")
        print(f"[AlignSkill] reference pose: {[x, y, z]}")
        print(f"[AlignSkill] retreated align target: {[x, y, z + retreat_distance]}")
        print(f"[AlignSkill] align orientation (rpy): {[rx, ry, rz]}")

    # -------------------------
    # 2. Pre-align pose: approach from above.
    # -------------------------
    safe_z = min(z + retreat_distance, current_pos[2] + pregrasp_height_offset)
    safe_z = max(safe_z, min_pregrasp_height)

    pre_align_pose = sapien.Pose(
        p=[x, y, safe_z],
        q=current_tcp_pose.q.cpu().numpy()[0],  # keep current orientation
    )

    if verbose:
        print(f"[AlignSkill] pre-align z = {safe_z}")

    if not planner.move_to_pose_with_screw(pre_align_pose):
        if verbose:
            print("[AlignSkill] FAIL: could not reach pre-align position")
        return False

    if verbose:
        print("[AlignSkill] OK: reached pre-align position")

    # -------------------------
    # 3. Single-shot move to the final align target.
    # -------------------------
    if not planner.move_to_pose_with_screw(align_target_pose):
        if verbose:
            print("[AlignSkill] FAIL: could not reach final align target")
        return False

    if verbose:
        print("[AlignSkill] OK: align complete (single shot)")

    return True

@register_skill("grasp_part", affordances=["graspable"], group="atomic", part_arg="part_name", phase="interaction",
                 description="Grasp a labeled part using its annotated grasp candidates.")
def grasp_part(planner, part_name='cap', verbose=False, vlm=False):
    FINGER_LENGTH = 0.025
    env = planner.env.unwrapped
    current_tcp_pose = env.agent.tcp.pose
    current_pos = current_tcp_pose.p.cpu().numpy()[0]
    # Resolve target grasp pose.
    config = env.object_config
    grasp_parts = config.get("grasp_parts", {})

    config, grasp_list = get_grasp_list(env, part_name)
    grasp_id = np.random.randint(0, len(grasp_list))
    current_grasp_pose = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)

    target_pos = current_grasp_pose.p
    target_quat = current_grasp_pose.q

    if verbose:
        print(f"current TCP position: {current_pos}")
        print(f"target grasp position: {target_pos}")
    safe_z = min(target_pos[2], current_pos[2] + 0.2)
    safe_z = max(safe_z, 0.2)   


    rotation_matrix = t3d.quaternions.quat2mat(target_quat)
    approach_direction = rotation_matrix[:, 2]  # local Z axis
    
    # 3. Pre-grasp position: retreat along the approach axis.
    pre_grasp_distance = 0.08  # 8 cm retreat
    pre_grasp_pos = target_pos - approach_direction * pre_grasp_distance
    pre_grasp_pose = sapien.Pose(pre_grasp_pos, target_quat)

    res = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
    if res == -1:
        if verbose:
            print("===== failed to reach pre-grasp pose =====")
        return res
    grasp_pose = sapien.Pose(current_grasp_pose.p, pre_grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        if verbose:
            print("===== grasp failed =====")
        return res
    planner.close_gripper()
    res = planner.move_to_pose_with_RRTConnect(sapien.Pose(current_tcp_pose.p.cpu().numpy()[0], current_tcp_pose.q.cpu().numpy()[0]))
    if res == -1:
        if verbose:
            print("===== failed to return to pre-grasp =====")
        return res
    if vlm:
        image_tensor = env.render_rgb_array()
        # Handle batched tensor: [B, H, W, C] -> [H, W, C].
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        # Convert to numpy with the right dtype.
        image_np = image_tensor.cpu().numpy()
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
        # Wrap as PIL Image.
        pil_image = Image.fromarray(image_np)
        # The evaluator accepts torch.Tensor / numpy / PIL / file paths interchangeably.
        success = evaluate_grasp_success(pil_image, env.object_name, part_name)
        if not success:
            return -1
    # Mark the part as engaged so continuation-phase skills (pure_slide,
    # pure_rotate, …) know what to operate on. Composes via the contact-state
    # API on GraspPartEnv (Phase E).
    if hasattr(env, "engage"):
        env.engage(part_name)
    return res

@register_skill("stand_up", affordances=["graspable", "flippable"], group="atomic", part_arg="part_name", phase="bundle",
                 description="Pick up a toppled object and place it upright.")
def stand_up(planner, part_name='cap', verbose=False):

    """
    Pick up a toppled bottle and stand it upright.

    Strategy:
    1. Grasp the lying bottle from the side.
    2. Lift, then rotate to vertical (pitch only; preserve yaw).
    3. Set the bottle down.
    """
    
    FINGER_LENGTH = 0.025
    env = planner.env.unwrapped
    
    current_tcp_pose = env.agent.tcp.pose
    current_pos = current_tcp_pose.p.cpu().numpy()[0]  

    # Resolve target grasp pose.
    config = env.object_config
    grasp_parts = config.get("grasp_parts", {})

    config, grasp_list = get_grasp_list(env, part_name)
    grasp_id = np.random.randint(0, len(grasp_list))
    current_grasp_pose = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)

    
    target_pos = current_grasp_pose.p
    target_quat = current_grasp_pose.q

    if verbose:
        print(f"current TCP position: {current_pos}")
        print(f"target grasp position: {target_pos}")
    safe_z = min(target_pos[2], current_pos[2] + 0.2)
    safe_z = max(safe_z, 0.2)   


    rotation_matrix = t3d.quaternions.quat2mat(target_quat)
    approach_direction = rotation_matrix[:, 2]  # local Z axis
    
    # 3. Pre-grasp position: retreat along the approach axis.
    pre_grasp_distance = 0.08  # 8 cm retreat
    pre_grasp_pos = target_pos - approach_direction * pre_grasp_distance
    pre_grasp_pose = sapien.Pose(pre_grasp_pos, target_quat)
    
    if verbose:
        print(f"adjusted pre-grasp position: {pre_grasp_pos}")

    pre_grasp_pose = sapien.Pose(pre_grasp_pos, current_tcp_pose.q.cpu().numpy()[0])

    res = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
    
    grasp_pose = sapien.Pose(current_grasp_pose.p, pre_grasp_pose.q)
    res = planner.move_to_pose_with_screw(grasp_pose)
    
    
    if res == -1:
        print("===== grasp failed =====")
        return res
    
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # Lift the bottle.
    # -------------------------------------------------------------------------- #

    lift_dist = 0.2
    lift_pose = grasp_pose * sapien.Pose([0, 0, - lift_dist])
    
    res = planner.move_to_pose_with_RRTConnect(lift_pose)
    if res == -1:
        if verbose:
            print("===== failed to lift bottle =====")
        return res
    
    # -------------------------------------------------------------------------- #
    # Rotate the bottle to vertical mid-air.
    # -------------------------------------------------------------------------- #
    
    # Read current bottle + TCP poses.
    current_bottle_pose = env.target_object.pose.sp 
    current_tcp_pose = env.agent.tcp.pose.sp
    
    # Compute grasp offset (TCP relative to bottle).
    grasp_offset = current_bottle_pose.inv() * current_tcp_pose
    
    # Extract current bottle yaw (rotation about world Z).
    current_bottle_quat = current_bottle_pose.q
    current_bottle_euler = R.from_quat([
        current_bottle_quat[1], 
        current_bottle_quat[2], 
        current_bottle_quat[3], 
        current_bottle_quat[0]
    ]).as_euler('xyz', degrees=False)
    
    current_yaw = current_bottle_euler[2]  # preserve yaw
    
    # Build the upright orientation: pitch=0, roll=0, yaw=current.
    upright_euler = np.array([0, 0, current_yaw])
    upright_quat_scipy = R.from_euler('xyz', upright_euler).as_quat()
    upright_quat_sapien = [upright_quat_scipy[3], upright_quat_scipy[0], 
                           upright_quat_scipy[1], upright_quat_scipy[2]]
    
    rotated_pos = current_bottle_pose.p.copy()
    rotated_pos[2] += 0.15  # lift slightly while rotating

    upright_bottle_pose = sapien.Pose(
        p=rotated_pos,
        q=upright_quat_sapien
    )
    
    # Compose the TCP target pose that puts the bottle upright.
    upright_tcp_pose = upright_bottle_pose * grasp_offset
    
    # Drive to the upright orientation.
    res = planner.move_to_pose_with_screw(upright_tcp_pose)
    
    if res == -1:
        if verbose:
            print("Failed to rotate bottle upright")
        return res
    
    # # -------------------------------------------------------------------------- #
    # # Move to target position (commented out).
    # # -------------------------------------------------------------------------- #
    
    target_tcp_pos = upright_tcp_pose.p.copy()
    target_tcp_pos[2] = 0.2

    target_tcp_pose = sapien.Pose(p=target_tcp_pos, q=upright_tcp_pose.q)
    
    # Move to target position.
    res = planner.move_to_pose_with_screw(target_tcp_pose)
    
    if res == -1:
        if verbose:
            print("Failed to move to target position")
        return res
    planner.open_gripper()
    planner.close()
    
    return res

@register_skill("press_switch", affordances=["pressable"], group="atomic", part_arg="part_name", phase="interaction",
                 description="Press a button-style switch from above.")
def press_switch(planner, part_name='button', verbose=False):
    """
    Press a button-style switch.

    Sequence:
    1. Close the gripper.
    2. Move to pre-grasp pose (retreated along approach axis).
    3. Push along the approach axis (press).
    4. Retreat back to pre-grasp.
    """
    import transforms3d as t3d
    
    env = planner.env.unwrapped
    current_tcp_pose = env.agent.tcp.pose
    config, grasp_list = get_grasp_list(env, part_name)
    grasp_id = np.random.randint(0, len(grasp_list))
    # 1. Resolve target pose.
    target_pose = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)
    if target_pose is None:
        if verbose:
            print(f"Skill PressButton: Not found part {part_name}")
        return -1
        
    target_pos = np.array(target_pose.p)
    target_quat = np.array(target_pose.q)  # [w, x, y, z]

    if verbose:
        print(f"target press position: {target_pos}")

    # 2. Extract the approach direction (target Z axis).
    rotation_matrix = t3d.quaternions.quat2mat(target_quat)
    approach_direction = rotation_matrix[:, 2]  # local Z axis
    
    # 3. Pre-grasp position: retreat along the approach axis.
    pre_grasp_distance = 0.08  # 8 cm retreat
    pre_grasp_pos = target_pos - approach_direction * pre_grasp_distance
    pre_grasp_pose = sapien.Pose(pre_grasp_pos, target_quat)

    # 4. Close the gripper.
    planner.close_gripper()

    # 5. Move to pre-grasp pose.
    success = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
    if not success:
        if verbose:
            print("Skill PressButton: could not reach pre-grasp pose")
        return -1

    # 6. Push along approach axis to press the button.
    press_pose = sapien.Pose(target_pos, target_quat)
    success = planner.move_to_pose_with_screw(press_pose)
    if not success:
        if verbose:
            print("Skill PressButton: press failed")
        return -1
    
    # 7. Retreat back along the approach axis.
    result = planner.move_to_pose_with_screw(pre_grasp_pose)
    planner.open_gripper()

    if verbose:
        print("Skill PressButton: done")
    return result

@register_skill("toggle_switch", affordances=["pressable"], group="atomic", part_arg="part_name", phase="interaction",
                 description="Toggle a lever-style switch.")
def toggle_switch(planner, part_name='handle', verbose=False):
    """
    Flip a lever-style toggle switch.

    Sequence:
    1. Move to pre-grasp pose (retreat along pose1's Z axis).
    2. Close the gripper around the lever.
    3. Move to pose1 (grasp pose).
    4. Push along the lever's X axis to toggle.
    """
    import transforms3d as t3d
    
    env = planner.env.unwrapped
    config, grasp_list = get_grasp_list(env, part_name)
    grasp_id = np.random.randint(0, len(grasp_list))
    # 1. Resolve target pose (starting position).
    pose1 = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)
    if pose1 is None:
        if verbose:
            print(f"Skill ToggleSwitch: Not found part {part_name}")
        return -1

    pos1 = np.array(pose1.p)
    quat1 = np.array(pose1.q)

    # 2. Compute approach + lever directions.
    rotation_matrix = t3d.quaternions.quat2mat(quat1)
    approach_dir1 = rotation_matrix[:, 2]  # local Z axis
    y_dir = rotation_matrix[:, 0]         # local X axis = toggle direction
    
    pre_grasp_distance = 0.05
    lever_distance = 0.04

    # 3. Build the waypoints.
    pregrasp1_pos = pos1 - approach_dir1 * pre_grasp_distance
    pos2 = pos1 + y_dir * lever_distance  # toggle endpoint

    # 4. Move to pre-grasp pose.
    pregrasp1_pose = sapien.Pose(pregrasp1_pos, quat1)
    success = planner.move_to_pose_with_RRTConnect(pregrasp1_pose)
    if not success:
        if verbose:
            print("Skill ToggleSwitch: could not reach pre-grasp pose")
        return -1

    # 5. Close the gripper around the lever.
    planner.close_gripper()

    # 6. Move to the grasp pose.
    grasp1_pose = sapien.Pose(pos1, quat1)
    success = planner.move_to_pose_with_screw(grasp1_pose)
    if not success:
        if verbose:
            print("Skill ToggleSwitch: could not reach grasp pose")
        pass 

    # 7. Push the lever to the toggle endpoint.
    grasp2_pose = sapien.Pose(pos2, quat1)
    success = planner.move_to_pose_with_screw(grasp2_pose)
    
    if verbose:
        print("Skill ToggleSwitch: done")
    return success

@register_skill("lid_opening", affordances=["graspable", "openable"], group="atomic", part_arg=None, phase="bundle",
                 description="Open a hinged/articulated lid by following its joint axis.")
class LidOpening:
    """
    Lid-opening action class.

    Bundles all lid-opening operations: axis discovery, path planning, execution.

    Features:
    - read object joint axes from the env;
    - read robot gripper info from the env;
    - execute arc motions around a chosen axis;
    - support reverse motion + gripper-pivot rotation.
    """
    
    def __init__(self, env, planner, 
                 radius_scale: float = 1.0,
                 verbose: bool = True):
        """
        Initialise the lid-opening action.

        Args:
            env: ManiSkill env (LidOpeningEnv recommended but not required).
            planner: motion-planning solver.
            radius_scale: scaling factor for the arc radius.
            verbose: emit progress logs.
        """
        self.env = env
        self.planner = planner
        self.radius_scale = radius_scale
        self.verbose = verbose
        
        # Will hold detected joint + gripper info after init.
        self.object_joints_info = {}
        self.gripper_info = {}
        
        # Run detection at construction time.
        self._initialize_detection()
    
    def _initialize_detection(self):
        """
        Detect object joints + gripper info up front.
        """
        if self.verbose:
            print("\n=== LidOpening init detection ===")
        
        # Discover target-object joints.
        self._detect_object_joints()
        
        # Discover gripper finger positions.
        self._detect_gripper_info()
        
        if self.verbose:
            print(f"=" * 50)
    
    def _detect_object_joints(self):
        """
        Detect every joint on the target object.
        """
        try:
            obj = self.env.unwrapped.target_object
            if obj is None:
                if self.verbose:
                    print("WARN: target object not found")
                return
            
            joints = obj.get_joints()
            active_joints = obj.get_active_joints()
            self.object_joints_info = {}
            
            # Map active joint name → qpos index.
            active_joint_name_to_qpos_idx = {}
            for qpos_idx, active_joint in enumerate(active_joints):
                active_joint_name_to_qpos_idx[active_joint.get_name()] = qpos_idx
            
            if self.verbose:
                print(f"\nObject joint info:")
                print(f"  object: {obj.get_name()}")
                print(f"  joints: {len(joints)}")
            
            for i, joint in enumerate(joints):
                joint_name = joint.get_name()
                joint_type = joint.get_type()
                
                # Look up the qpos index for active joints.
                qpos_index = active_joint_name_to_qpos_idx.get(joint_name, None)
                
                try:
                    joint_pose = joint.get_global_pose()
                    joint_pos = np.asarray(joint_pose.p).ravel()[:3]
                    
                    # Joint axis.
                    full_tf = np.asarray(joint_pose.to_transformation_matrix())
                    if full_tf.ndim == 3 and full_tf.shape[0] == 1:
                        full_tf = full_tf[0]
                    rotation_matrix = full_tf[:3, :3]
                    joint_axis = rotation_matrix[:, 0]  # local X axis
                    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-12)
                    
                    self.object_joints_info[joint_name] = {
                        'index': i,
                        'qpos_index': qpos_index,  # index in qpos vector
                        'type': joint_type,
                        'position': joint_pos,
                        'axis': joint_axis,
                        'joint_object': joint
                    }
                    
                    if self.verbose:
                        print(f"  [{i}] {joint_name}: {joint_type}")
                    
                except Exception as e:
                    self.object_joints_info[joint_name] = {
                        'index': i,
                        'qpos_index': qpos_index,
                        'type': joint_type,
                        'joint_object': joint
                    }
                    if self.verbose:
                        print(f"  [{i}] {joint_name}: {joint_type} (details unavailable)")
            
        except Exception as e:
            if self.verbose:
                print(f"FAILED to detect joints: {e}")
    
    def _detect_gripper_info(self):
        """
        Detect left + right gripper finger positions.
        """
        try:
            robot = self.env.agent.robot
            all_links = robot.get_links()
            
            if self.verbose:
                print(f"\nGripper info:")
                print(f"  robot: {robot.get_name()}")
            
            left_finger_pos = None
            right_finger_pos = None
            left_finger_link = None
            right_finger_link = None
            
            # Exact-name pass.
            exact_patterns = {
                'left': ['panda_leftfinger', 'gripper_left_finger', 'left_finger_tip', 'leftfinger'],
                'right': ['panda_rightfinger', 'gripper_right_finger', 'right_finger_tip', 'rightfinger']
            }
            
            for link in all_links:
                link_name = link.get_name().lower()
                
                for pattern in exact_patterns['left']:
                    if link_name == pattern:
                        try:
                            pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                            left_finger_pos = np.asarray(pose.p).ravel()[:3]
                            left_finger_link = link
                            break
                        except Exception:
                            pass
                
                for pattern in exact_patterns['right']:
                    if link_name == pattern:
                        try:
                            pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                            right_finger_pos = np.asarray(pose.p).ravel()[:3]
                            right_finger_link = link
                            break
                        except Exception:
                            pass
            
            # Keyword-match pass.
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
            
            # Persist detected info.
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
                
                if self.verbose:
                    print(f"  OK left finger: {left_finger_link.get_name()}")
                    print(f"  OK right finger: {right_finger_link.get_name()}")
                    print(f"  finger span: {self.gripper_info['distance']:.4f} m")
            else:
                if self.verbose:
                    print(f"  WARN: could not detect both fingers")
            
        except Exception as e:
            if self.verbose:
                print(f"FAILED to detect gripper: {e}")
    
    def get_joint_axis_by_name(self, joint_name: str) -> tuple:
        """
        Look up the axis info for a joint by name.

        Args:
            joint_name: name of the joint.

        Returns:
            (axis_position, axis_vector) tuple — both ``None`` on miss.
        """
        if joint_name not in self.object_joints_info:
            print(f"FAIL: joint '{joint_name}' not found")
            available = list(self.object_joints_info.keys())
            print(f"  available joints: {available}")
            return None, None
        
        info = self.object_joints_info[joint_name]
        if 'position' in info and 'axis' in info:
            return info['position'], info['axis']
        else:
            print(f"FAIL: joint '{joint_name}' has no position/axis info")
            return None, None
    
    def get_joint_angle(self, joint_name: str) -> float:
        """
        Return the current angle (radians) of the named joint, or None on failure.
        """
        try:
            if joint_name not in self.object_joints_info:
                if self.verbose:
                    print(f"FAIL: joint '{joint_name}' not found")
                return None
            
            info = self.object_joints_info[joint_name]
            qpos_index = info.get('qpos_index')
            
            if qpos_index is None:
                if self.verbose:
                    print(f"FAIL: joint '{joint_name}' is not active (likely fixed)")
                return None
            
            # Read every active joint's qpos from target_object.
            obj = self.env.unwrapped.target_object
            if obj is None:
                if self.verbose:
                    print(f"FAIL: target object is None")
                return None
            
            qpos = obj.get_qpos()  # shape: (num_envs, num_active_joints)
            
            # Handle the batched-env case.
            if hasattr(qpos, 'ndim'):
                if qpos.ndim == 2:
                    # Multi-env: use the first env.
                    angle = float(qpos[0, qpos_index])
                elif qpos.ndim == 1:
                    # Single-env case.
                    angle = float(qpos[qpos_index])
                else:
                    angle = float(qpos)
            else:
                # Scalar fallback.
                angle = float(qpos)
            
            return angle
            
        except Exception as e:
            if self.verbose:
                print(f"FAIL: could not read joint '{joint_name}' angle: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def get_object_axis_info(self) -> dict:
        """
        Return the full joint-info dict for every detected joint.
        """
        return self.object_joints_info.copy()
    
    def get_robot_pose_info(self) -> dict:
        """
        Return current robot pose info (TCP + fingers + joint angles).
        """
        info = {}
        
        try:
            robot = self.env.agent.robot
            
            # TCP position (matches test_arc_skill's access pattern).
            try:
                tcp_pose = self.env.agent.tcp.pose
            except:
                tcp_pose = self.env.agent.robot.get_links()[-1].pose
            
            info['tcp'] = {
                'position': np.asarray(tcp_pose.p).ravel()[:3],
                'quaternion': np.asarray(tcp_pose.q).ravel()
            }
            
            # Gripper info.
            if self.gripper_info:
                info['gripper'] = self.gripper_info.copy()
                
                # Refresh current finger positions.
                if 'left_finger' in self.gripper_info and self.gripper_info['left_finger']['link']:
                    link = self.gripper_info['left_finger']['link']
                    pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                    info['gripper']['left_finger']['current_position'] = np.asarray(pose.p).ravel()[:3]
                
                if 'right_finger' in self.gripper_info and self.gripper_info['right_finger']['link']:
                    link = self.gripper_info['right_finger']['link']
                    pose = link.pose if hasattr(link, 'pose') else link.get_pose()
                    info['gripper']['right_finger']['current_position'] = np.asarray(pose.p).ravel()[:3]
                
                # Recompute centre.
                if 'left_finger' in info['gripper'] and 'right_finger' in info['gripper']:
                    left_pos = info['gripper']['left_finger']['current_position']
                    right_pos = info['gripper']['right_finger']['current_position']
                    info['gripper']['current_center'] = (left_pos + right_pos) / 2.0
                    info['gripper']['current_distance'] = np.linalg.norm(left_pos - right_pos)
            
            # Joint angles.
            info['joint_qpos'] = self.env.agent.robot.get_qpos()
            
        except Exception as e:
            print(f"FAIL: could not read robot pose info: {e}")
        
        return info
    
    def execute_arc_motion(self,
                          axis_pos=None,
                          axis_vec=None,
                          joint_name: str = None,
                          angle_deg: float = -10.0,
                          steps: int = 10,
                          keep_orientation: bool = True,
                          do_reverse: bool = False,
                          reverse_angle_deg: float = None,
                          reverse_steps: int = None,
                          reverse_keep_orientation: bool = None,
                          reverse_pivot_side: str = None,
                          visualize: bool = True,
                          check_joint_movement: bool = True,
                          min_joint_change: float = 0.001):
        """
        Execute an arc motion around the chosen axis.

        Args:
            axis_pos: rotation pivot point (use this OR joint_name).
            axis_vec: rotation axis vector (use this OR joint_name).
            joint_name: joint name to pull axis_pos / axis_vec from.
            angle_deg: rotation angle in degrees.
            steps: number of interpolation steps.
            keep_orientation: keep the end-effector orientation fixed.
            do_reverse: also execute a reverse motion afterwards.
            reverse_angle_deg: angle for the reverse motion.
            reverse_steps: interpolation steps for the reverse motion.
            reverse_keep_orientation: keep orientation during reverse.
            reverse_pivot_side: pivot for the reverse motion ("left", "right", "center", None).
            visualize: draw the axis in the viewer.
            check_joint_movement: verify the joint actually moved.
            min_joint_change: min radians of joint motion to count as moved.

        Returns:
            bool: True on success.
        """
        try:
            # Capture the joint angle at the start.
            initial_joint_angle = None
            target_joint_name = joint_name  # remembered for the post-motion check
            
            # Resolve axis info.
            if joint_name is not None:
                # via joint_name lookup.
                axis_pos, axis_vec = self.get_joint_axis_by_name(joint_name)
                if axis_pos is None or axis_vec is None:
                    return False
                if self.verbose:
                    print(f"OK: using axis of joint '{joint_name}'")
                
                # Capture initial joint angle.
                if check_joint_movement:
                    initial_joint_angle = self.get_joint_angle(joint_name)
                    if initial_joint_angle is not None:
                        if self.verbose:
                            print(f"  initial joint angle: {initial_joint_angle:.6f} rad ({np.degrees(initial_joint_angle):.2f}°)")
                    else:
                        if self.verbose:
                            print(f"  WARN: could not read initial joint angle; skipping movement check")
                        check_joint_movement = False
                        
            elif axis_pos is not None and axis_vec is not None:
                # Use the explicitly-provided axis info.
                axis_pos = np.asarray(axis_pos).ravel()[:3]
                axis_vec = np.asarray(axis_vec).ravel()[:3]
                axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
                if self.verbose:
                    print(f"OK: using custom axis")
                # No joint_name → can't verify joint motion later.
                check_joint_movement = False
            else:
                print(f"FAIL: must provide joint_name or (axis_pos, axis_vec)")
                return False
            
            # Visualise the axis.
            if visualize:
                self._visualize_axis(axis_pos, axis_vec, color=[1.0, 0.0, 0.0, 1.0], prefix="main_axis")
            
            # Snapshot the starting pose (same pattern as test_arc_skill).
            try:
                start_pose = self.env.agent.tcp.pose
            except:
                start_pose = self.env.agent.robot.get_links()[-1].pose
            
            # Prepare reverse-motion parameters.
            reverse_axis_pos = None
            reverse_axis_vec = None
            
            if do_reverse and reverse_pivot_side:
                # Use the pre-motion finger pose as a reference.
                # Note: the actual reverse motion re-reads pose after the main motion.
                if reverse_pivot_side == "left" and 'left_finger' in self.gripper_info:
                    if self.verbose:
                        print(f"OK: reverse motion will pivot on the left finger")
                elif reverse_pivot_side == "right" and 'right_finger' in self.gripper_info:
                    if self.verbose:
                        print(f"OK: reverse motion will pivot on the right finger")
                elif reverse_pivot_side == "center":
                    if self.verbose:
                        print(f"OK: reverse motion will pivot on the gripper centre")
                
                # Reverse motion uses the same axis direction as the main motion.
                reverse_axis_vec = axis_vec.copy()
            
            # Phase 1: main arc.
            if self.verbose:
                print(f"\n=== Phase 1: main arc (angle={angle_deg}°, steps={steps}) ===")
                # Debug: log current TCP + axis positions.
                robot = self.env.agent.robot
                tcp_link = robot.get_links()[-1]
                current_tcp = tcp_link.pose if hasattr(tcp_link, 'pose') else tcp_link.get_pose()
                tcp_pos = np.asarray(current_tcp.p).ravel()[:3]
                print(f"  TCP position: {np.round(tcp_pos, 4)}")
                print(f"  joint axis position: {np.round(axis_pos, 4)}")
                print(f"  TCP-to-axis distance: {np.linalg.norm(tcp_pos - axis_pos):.4f} m")
            
            main_path = plan_arc_path(
                start_pose=start_pose,
                axis_pos=axis_pos,
                axis_vec=axis_vec,
                angle_deg=angle_deg,
                steps=steps,
                keep_orientation=keep_orientation,
                radius_scale=self.radius_scale,
                do_reverse=False
            )
            
            # Execute the main motion.
            success_count = 0
            failed_count = 0
            max_consecutive_failures = 3  # bail after 3 consecutive failures
            consecutive_failures = 0
            
            for i, waypoint in enumerate(main_path, start=1):
                # Preview the current waypoint.
                if visualize:
                    self._visualize_gripper_preview(waypoint, color=[0.0, 1.0, 0.0, 0.5], prefix=f"main_step_{i}")
                
                result = self.planner.move_to_pose_with_screw(waypoint)
                # Same success check as test_arc_skill.
                if result != -1:
                    success_count += 1
                    consecutive_failures = 0  # reset consecutive-failure counter
                    if self.verbose:
                        print(f"  OK waypoint {i}/{len(main_path)}")
                else:
                    failed_count += 1
                    consecutive_failures += 1
                    if self.verbose:
                        print(f"  WARN: waypoint {i}/{len(main_path)} plan failed")
                    
                    # Stop early on too many consecutive failures.
                    if consecutive_failures >= max_consecutive_failures:
                        if self.verbose:
                            print(f"  FAIL: {max_consecutive_failures} consecutive failures, aborting")
                        break
                
                # Clear the waypoint preview.
                if visualize:
                    self._clear_visualization(prefix=f"main_step_{i}")
            
            if self.verbose:
                print(f"  main motion done: {success_count}/{len(main_path)} succeeded, {failed_count} failed")
            
            # Abort when the main-motion success rate is too low.
            success_rate = success_count / len(main_path) if len(main_path) > 0 else 0
            if success_rate < 0.5:  # < 50% counts as failure
                if self.verbose:
                    print(f"  FAIL: main-motion success rate too low ({success_count}/{len(main_path)} = {success_rate:.1%})")
                return False
            
            # Phase 2: reverse motion (optional).
            if do_reverse:
                if self.verbose:
                    print(f"\n=== Phase 2: reverse motion ===")
                
                # Refresh current robot pose.
                current_robot_info = self.get_robot_pose_info()
                
                # Resolve the pivot for the reverse motion.
                if reverse_pivot_side == "left" and 'left_finger' in current_robot_info.get('gripper', {}):
                    reverse_axis_pos = current_robot_info['gripper']['left_finger']['current_position']
                    if self.verbose:
                        print(f"  >>> using current left finger as pivot: {np.round(reverse_axis_pos, 4)}")
                elif reverse_pivot_side == "right" and 'right_finger' in current_robot_info.get('gripper', {}):
                    reverse_axis_pos = current_robot_info['gripper']['right_finger']['current_position']
                    if self.verbose:
                        print(f"  >>> using current right finger as pivot: {np.round(reverse_axis_pos, 4)}")
                elif reverse_pivot_side == "center" and 'current_center' in current_robot_info.get('gripper', {}):
                    reverse_axis_pos = current_robot_info['gripper']['current_center']
                    if self.verbose:
                        print(f"  >>> using current gripper centre as pivot: {np.round(reverse_axis_pos, 4)}")
                else:
                    # Default: reuse the original axis.
                    reverse_axis_pos = axis_pos
                    if self.verbose:
                        print(f"  >>> reusing the original axis")
                
                # Visualise the reverse-motion axis.
                if visualize and reverse_axis_pos is not None:
                    self._visualize_axis(reverse_axis_pos, reverse_axis_vec, 
                                       color=[0.0, 1.0, 0.0, 1.0], prefix="reverse_axis")
                
                # Snapshot the reverse-motion start pose (same pattern as test_arc_skill).
                try:
                    current_tcp_pose = self.env.agent.tcp.pose
                except:
                    current_tcp_pose = self.env.agent.robot.get_links()[-1].pose
                
                # Plan the reverse-motion path.
                reverse_path = plan_arc_path(
                    start_pose=current_tcp_pose,
                    axis_pos=reverse_axis_pos,
                    axis_vec=reverse_axis_vec,
                    angle_deg=reverse_angle_deg if reverse_angle_deg is not None else -angle_deg,
                    steps=reverse_steps if reverse_steps is not None else steps,
                    keep_orientation=reverse_keep_orientation if reverse_keep_orientation is not None else keep_orientation,
                    radius_scale=self.radius_scale,
                    do_reverse=False
                )
                
                # Execute the reverse motion.
                reverse_success_count = 0
                reverse_failed_count = 0
                consecutive_failures = 0
                
                for i, waypoint in enumerate(reverse_path, start=1):
                    # Preview the current reverse waypoint.
                    if visualize:
                        self._visualize_gripper_preview(waypoint, color=[1.0, 0.5, 0.0, 0.5], prefix=f"reverse_step_{i}")
                    
                    result = self.planner.move_to_pose_with_screw(waypoint)
                    # Same success check as test_arc_skill.
                    if result != -1:
                        reverse_success_count += 1
                        consecutive_failures = 0
                        if self.verbose:
                            print(f"  OK reverse waypoint {i}/{len(reverse_path)}")
                    else:
                        reverse_failed_count += 1
                        consecutive_failures += 1
                        if self.verbose:
                            print(f"  WARN: reverse waypoint {i}/{len(reverse_path)} plan failed")
                        
                        # Stop after 3 consecutive failures.
                        if consecutive_failures >= 3:
                            if self.verbose:
                                print(f"  FAIL: reverse motion failed 3 times in a row, aborting")
                            break
                    
                    # Clear the waypoint preview.
                    if visualize:
                        self._clear_visualization(prefix=f"reverse_step_{i}")
                
                if self.verbose:
                    print(f"  reverse motion done: {reverse_success_count}/{len(reverse_path)} succeeded, {reverse_failed_count} failed")
                
                # Check the reverse-motion success rate.
                reverse_success_rate = reverse_success_count / len(reverse_path) if len(reverse_path) > 0 else 0
                if reverse_success_rate < 0.5:
                    if self.verbose:
                        print(f"  FAIL: reverse-motion success rate too low ({reverse_success_count}/{len(reverse_path)} = {reverse_success_rate:.1%})")
                    return False
            
            # ==================== Verify the joint actually moved ====================
            if check_joint_movement and initial_joint_angle is not None and target_joint_name is not None:
                final_joint_angle = self.get_joint_angle(target_joint_name)
                
                if final_joint_angle is not None:
                    joint_change = abs(final_joint_angle - initial_joint_angle)
                    joint_change_deg = np.degrees(joint_change)
                    
                    if self.verbose:
                        print(f"\n=== Joint-movement check ===")
                        print(f"  joint: {target_joint_name}")
                        print(f"  initial angle: {initial_joint_angle:.6f} rad ({np.degrees(initial_joint_angle):.2f}°)")
                        print(f"  final angle: {final_joint_angle:.6f} rad ({np.degrees(final_joint_angle):.2f}°)")
                        print(f"  delta: {joint_change:.6f} rad ({joint_change_deg:.2f}°)")
                        print(f"  threshold: {min_joint_change:.6f} rad ({np.degrees(min_joint_change):.2f}°)")
                    
                    if joint_change < min_joint_change:
                        if self.verbose:
                            print(f"  FAIL: joint moved too little ({joint_change:.6f} < {min_joint_change:.6f})")
                            print(f"  the arm completed the motion but the joint did not actually move")
                        return False
                    else:
                        if self.verbose:
                            print(f"  OK: joint moved as expected")
                else:
                    if self.verbose:
                        print(f"  WARN: could not read final joint angle; skipping check")
            
            if self.verbose:
                print(f"OK: motion completed")
            
            return True
            
        except Exception as e:
            print(f"FAIL: arc motion raised: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _visualize_axis(self, axis_pos, axis_vec, length: float = 0.2, 
                       thickness: float = 0.005, color: list = [1.0, 0.0, 0.0, 1.0],
                       prefix: str = "axis_vis"):
        """
        Render an axis preview in the scene.

        Args:
            axis_pos: pivot point in world coordinates.
            axis_vec: axis direction vector.
            length: rendered axis length.
            thickness: rendered axis thickness.
            color: RGBA colour.
            prefix: object-name prefix used by :meth:`_clear_visualization`.
        """
        try:
            import uuid
            
            scene = self.env.unwrapped.scene
            
            # Remove any previous visuals with the same prefix.
            for actor in list(scene.get_all_actors()):
                try:
                    if actor.get_name().startswith(prefix):
                        scene.remove_actor(actor)
                except Exception:
                    pass
            
            # Normalise the axis vector.
            axis = np.asarray(axis_vec).ravel()[:3].astype(float)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            center = np.asarray(axis_pos).ravel()[:3].astype(float)
            
            # Build the rotation matrix.
            a = np.array([0.0, 0.0, 1.0])
            if np.allclose(np.abs(np.dot(a, axis)), 1.0, atol=1e-6):
                a = np.array([0.0, 1.0, 0.0])
            y = np.cross(a, axis)
            y = y / (np.linalg.norm(y) + 1e-12)
            z = np.cross(axis, y)
            
            rot_mat = np.column_stack([axis, y, z])
            quat = t3d.quaternions.mat2quat(rot_mat)
            
            unique_id = uuid.uuid4().hex[:8]
            
            # Add the axis visual.
            builder = scene.create_actor_builder()
            shaft_name = f"{prefix}_shaft_{unique_id}"
            builder.set_name(shaft_name)
            half_len = length / 2.0
            half_size = [half_len, thickness / 2.0, thickness / 2.0]
            builder.add_box_visual(pose=sapien.Pose(p=[0, 0, 0]), half_size=half_size,
                                  material=sapien.render.RenderMaterial(base_color=color))
            builder.set_initial_pose(sapien.Pose(p=center.tolist(), q=quat))
            builder.build_kinematic(name=shaft_name)
            
            # Add a small sphere at the pivot.
            builder = scene.create_actor_builder()
            center_name = f"{prefix}_center_{unique_id}"
            builder.set_name(center_name)
            builder.add_sphere_visual(pose=sapien.Pose(p=[0, 0, 0]), radius=thickness * 2.0,
                                    material=sapien.render.RenderMaterial(base_color=[0, 1.0, 0.0, 1.0]))
            builder.set_initial_pose(sapien.Pose(p=center.tolist()))
            builder.build_kinematic(name=center_name)
            
            # Force a render update.
            scene.update_render()
            if hasattr(self.env, 'viewer') and self.env.viewer is not None and not self.env.viewer.closed:
                self.env.viewer.render()
            
        except Exception as e:
            if self.verbose:
                print(f"WARN: axis visualisation failed: {e}")
    
    def _visualize_gripper_preview(self, target_pose, color: list = [0.0, 1.0, 0.0, 0.5], 
                                   prefix: str = "gripper_preview"):
        """
        Render a gripper-pose preview at ``target_pose``.

        Args:
            target_pose: target pose (sapien.Pose).
            color: RGBA colour.
            prefix: object-name prefix used by :meth:`_clear_visualization`.
        """
        try:
            import uuid
            
            scene = self.env.unwrapped.scene
            
            # Remove any previous visuals with the same prefix.
            self._clear_visualization(prefix)
            
            # Extract pose components.
            pos = np.asarray(target_pose.p).ravel()[:3].astype(float)
            quat = np.asarray(target_pose.q).ravel().astype(float)
            
            unique_id = uuid.uuid4().hex[:8]
            
            # Spawn a small sphere at the TCP target.
            builder = scene.create_actor_builder()
            tcp_name = f"{prefix}_tcp_{unique_id}"
            builder.set_name(tcp_name)
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0, 0, 0]), 
                radius=0.01,
                material=sapien.render.RenderMaterial(base_color=color)
            )
            builder.set_initial_pose(sapien.Pose(p=pos.tolist(), q=quat))
            builder.build_kinematic(name=tcp_name)
            
            # Spawn a direction indicator (cylinder along the approach axis).
            builder = scene.create_actor_builder()
            arrow_name = f"{prefix}_arrow_{unique_id}"
            builder.set_name(arrow_name)
            # Arrow along the local Z axis.
            arrow_length = 0.03
            arrow_thickness = 0.003
            builder.add_box_visual(
                pose=sapien.Pose(p=[0, 0, arrow_length/2]), 
                half_size=[arrow_thickness, arrow_thickness, arrow_length/2],
                material=sapien.render.RenderMaterial(base_color=[color[0], color[1], color[2], 1.0])
            )
            builder.set_initial_pose(sapien.Pose(p=pos.tolist(), q=quat))
            builder.build_kinematic(name=arrow_name)
            
            # Force a render update.
            scene.update_render()
            if hasattr(self.env, 'viewer') and self.env.viewer is not None and not self.env.viewer.closed:
                self.env.viewer.render()
            
        except Exception as e:
            if self.verbose:
                print(f"WARN: gripper-preview visualisation failed: {e}")
    
    def _clear_visualization(self, prefix: str):
        """
        Remove every preview object whose name starts with ``prefix``.

        Args:
            prefix: visualisation-name prefix to match.
        """
        try:
            scene = self.env.unwrapped.scene
            
            for actor in list(scene.get_all_actors()):
                try:
                    if actor.get_name().startswith(prefix):
                        scene.remove_actor(actor)
                except Exception:
                    pass
            
            # Force a render update.
            scene.update_render()
            
        except Exception:
            pass

@register_skill("peg_in_hole", affordances=["insertable"], group="bespoke", part_arg=None, phase="bundle",
                 description="Bespoke peg-in-hole motion plan for PegInHoleEnv.")
def peg_in_hole(planner, seed=None, part_name=None, debug=False, verbose=False):
    env = planner.env.unwrapped
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose


    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])
    grasp_pose = grasp_pose * (offset)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose
    offset = sapien.Pose([-0.01 - env.peg_half_sizes[0, 0].item(), 0, 0])
    pre_insert_pose = insert_pose * (offset)
    res = planner.move_to_pose_with_screw(pre_insert_pose)
    if res == -1: return res
    # refine the insertion pose
    for i in range(3):
        delta_pose = env.goal_pose * (offset) * env.peg.pose.inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        res = planner.move_to_pose_with_screw(pre_insert_pose)
        if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(insert_pose * (sapien.Pose([0.05, 0, 0])))
    if res == -1: return res
    planner.close()
    return res

@register_skill("plug_charger", affordances=["insertable"], group="bespoke", part_arg=None, phase="bundle",
                 description="Bespoke charger insertion plan for PlugChargerEnv.")
def plug_charger(planner, seed=None, part_name=None, debug=False, verbose=False):
    env = planner.env.unwrapped
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    charger_base_pose = env.charger_base_pose
    charger_base_size = np.array(env.unwrapped._base_size) * 2

    obb = trimesh.primitives.Box(
        extents=charger_base_size,
        transform=charger_base_pose.sp.to_transformation_matrix(),
    )

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # add a angle to grasp
    grasp_angle = np.deg2rad(15)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Align
    # -------------------------------------------------------------------------- #
    pre_insert_pose = (
        env.goal_pose.sp
        * sapien.Pose([-0.05, 0.0, 0.0])
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
    planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
    planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5)
    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(insert_pose)

    planner.close()
    return res

@register_skill("stack_pyramid", affordances=["stackable"], group="bespoke", part_arg=None, phase="bundle",
                 description="Bespoke cube-stacking plan for StackPyramidEnv.")
def stack_pyramid(planner, seed=None, part_name=None, debug=False, verbose=False):
    env = planner.env.unwrapped
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    moving_cube = env.cubeA
    target_cube = env.cubeB

    # -------------------------------------------------------------------------- #
    # Move the specified cube to be next to the other cube
    # -------------------------------------------------------------------------- #
    # Move Gripper to the specified cube
    obb = get_actor_obb(moving_cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    distance = np.linalg.norm(moving_cube.pose.sp.p - target_cube.pose.sp.p)  

    need_move_a_b = (distance > 0.07)
    if need_move_a_b:
        planner.close_gripper()
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, moving_cube.pose.sp.p)

        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)

        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()

        # Move to Goal Pose
        goal_pose = sapien.Pose(target_cube.pose.sp.p * 0.8, grasp_pose.q)
        planner.move_to_pose_with_screw(goal_pose)

    # -------------------------------------------------------------------------- #
    # Stack Cube C onto Cube A and B
    # -------------------------------------------------------------------------- #

    obb = get_actor_obb(env.cubeC)
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    # planner.planner.update_attached_box([0.04, 0.04, 0.04], Pose.create(env.cubeB.pose).raw_pose.numpy().astype(np.float64).reshape(7,1))

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    if need_move_a_b:
         planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pose_A = env.cubeA.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
    goal_pose_B = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
    goal_pose_p = (goal_pose_A.p + goal_pose_B.p)/2
    offset = (goal_pose_p - env.cubeC.pose.p).cpu().numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()
    planner.close()
    return res

@register_skill("move_to_direction", affordances=[], group="atomic", part_arg=None, phase="continuation",
                 description="Translate TCP by `step` metres along a world-frame axis. Calling with no kwargs nudges 5 cm forward, so a chain step that drops in `move_to_direction` without parameters never crashes.")
def move_to_direction(planner, direction: str = "forward", step: float = 0.05, verbose: bool = False, **_):
    """Translate the TCP a small step along a named world-frame axis.

    Defaults to ``direction='forward'`` and ``step=0.05`` m so the skill is
    safe to drop into a task graph with no explicit parameters. Lateral and
    front/back motions are typically more useful than up/down for tabletop
    manipulation, hence the forward default.
    """
    env = planner.env.unwrapped
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    try:
        step = float(step)
    except Exception:
        step = 0.05

    direction_to_axis = {
        "up":       np.array([0.0,  0.0,  1.0]),
        "down":     np.array([0.0,  0.0, -1.0]),
        "left":     np.array([-1.0, 0.0,  0.0]),
        "right":    np.array([1.0,  0.0,  0.0]),
        "forward":  np.array([0.0,  1.0,  0.0]),
        "backward": np.array([0.0, -1.0,  0.0]),
    }
    if direction not in direction_to_axis:
        if verbose:
            print(f"[move_to_direction] unknown direction {direction!r}; "
                  f"valid: {sorted(direction_to_axis)}. Falling back to 'forward'.")
        direction = "forward"

    current_tcp_pose = env.agent.tcp.pose
    current_pos = current_tcp_pose.p.cpu().numpy()[0]
    current_quat = current_tcp_pose.q.cpu().numpy()[0]
    delta = direction_to_axis[direction] * step
    goal_pose = sapien.Pose(p=current_pos + delta, q=current_quat)
    if verbose:
        print(f"[move_to_direction] direction={direction}, step={step:.3f} m")
    return planner.move_to_pose_with_screw(goal_pose)

@register_skill("slide_along", affordances=["slidable", "graspable"], group="atomic", part_arg="part_name", phase="bundle",
                 description="Grasp a handle/drawer and pull along its prismatic joint axis.")
def slide_along(planner, part_name='cap', verbose=False, pull_dist: float = 0.20):
    FINGER_LENGTH = 0.025
    env = planner.env.unwrapped
    current_tcp_pose = env.agent.tcp.pose
    current_pos = current_tcp_pose.p.cpu().numpy()[0]
    # Resolve target grasp pose.
    config = env.object_config
    grasp_parts = config.get("grasp_parts", {})

    config, grasp_list = get_grasp_list(env, part_name)
    grasp_id = np.random.randint(0, len(grasp_list))
    current_grasp_pose = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)

    target_pos = current_grasp_pose.p
    target_quat = current_grasp_pose.q

    if verbose:
        print(f"current TCP position: {current_pos}")
        print(f"target grasp position: {target_pos}")
    safe_z = min(target_pos[2], current_pos[2] + 0.2)
    safe_z = max(safe_z, 0.2)   


    rotation_matrix = t3d.quaternions.quat2mat(target_quat)
    approach_direction = rotation_matrix[:, 2]  # local Z axis
    
    # 3. Pre-grasp position: retreat along the approach axis.
    pre_grasp_distance = 0.08  # 8 cm retreat
    pre_grasp_pos = target_pos - approach_direction * pre_grasp_distance
    pre_grasp_pose = sapien.Pose(pre_grasp_pos, target_quat)

    res = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
    if res == -1:
        if verbose:
            print("===== failed to reach pre-grasp pose =====")
        return res
    grasp_pose = sapien.Pose(current_grasp_pose.p, pre_grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        # Fall back to RRTConnect when the screw planner fails (matches env.grasp_part).
        res = planner.move_to_pose_with_RRTConnect(grasp_pose)
        if res == -1:
            if verbose:
                print("===== grasp failed =====")
            return res
    planner.close_gripper()

    # Second phase: translate along TCP local -Z by pull_dist.
    try:
        pull_dist = float(pull_dist)
    except Exception:
        pull_dist = 0.20
    if pull_dist <= 0:
        return res

    current_tcp_pose = env.agent.tcp.pose
    goal_pose = current_tcp_pose * sapien.Pose([0, 0, -pull_dist])
    if verbose:
        print(f"[slide_along] TCP -Z translation: {pull_dist:.3f} m")
        
    res2 = planner.move_to_pose_with_screw(goal_pose)
    if res2 == -1:
        res2 = planner.move_to_pose_with_RRTConnect(goal_pose)
        if res2 == -1:
            if verbose:
                print("===== TCP -Z translation failed (screw + RRTConnect both failed) =====")
            return res2
    return res2


# --------------------------------------------------------------------------- #
# Pure continuation skills (Phase E)                                          #
# --------------------------------------------------------------------------- #
# These assume an interaction skill ran first and registered an engaged part
# on the env (env.engage(part)). They operate on the engagement state — they
# never do their own grasp — so they can be freely composed in a task graph.

@register_skill("pure_slide", affordances=["slidable"], group="atomic",
                part_arg=None, phase="continuation",
                description="Translate the gripper along its local -Z by `distance` metres. "
                            "Use after an interaction skill engages a slidable part.")
def pure_slide(planner, distance: float = 0.20, verbose: bool = False, **_):
    """Pull (or push, with negative distance) the already-engaged part.

    Returns the underlying planner result; ``-1`` means motion plan failed.
    Validates engagement state so the caller gets a clear error if a
    continuation skill ends up first in the chain.
    """
    env = planner.env.unwrapped
    if getattr(env, "engaged_part", None) is None:
        if verbose:
            print("[pure_slide] no engaged part; an interaction step must run first")
        return -1

    try:
        distance = float(distance)
    except Exception:
        distance = 0.20
    if abs(distance) < 1e-6:
        return 0

    current_tcp_pose = env.agent.tcp.pose
    # Move along TCP local -Z (the gripper's pulling direction at grasp time).
    goal_pose = current_tcp_pose * sapien.Pose([0, 0, -distance])
    if verbose:
        print(f"[pure_slide] distance={distance:.3f}m along TCP -Z")
    res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(goal_pose)
        if res == -1 and verbose:
            print("[pure_slide] screw + RRTConnect both failed")
    return res


@register_skill("release_gripper", affordances=[], group="atomic",
                part_arg=None, phase="continuation",
                description="Open the gripper and clear the engagement state. "
                            "Ends a contact sequence so subsequent continuation skills won't try to act on stale state.")
def release_gripper(planner, verbose: bool = False, **_):
    env = planner.env.unwrapped
    if verbose:
        engaged = getattr(env, "engaged_part", None)
        print(f"[release_gripper] disengaging (was: {engaged!r})")
    planner.open_gripper()
    if hasattr(env, "disengage"):
        env.disengage()
    return 0


@register_skill("pure_rotate", affordances=["rotatable"], group="atomic",
                part_arg=None, phase="continuation",
                description="Rotate the gripper around its local axis by `angle_deg`. "
                            "Use after an interaction skill engages a rotatable part "
                            "(knob, cap, dial, lever).")
def pure_rotate(
    planner,
    angle_deg: float = 60.0,
    axis: str = "z",
    step_angle: float = 15.0,
    verbose: bool = False,
    **_,
):
    """Rotate TCP in its local frame around `axis` ('x'|'y'|'z') by `angle_deg`.

    Splits the rotation into smaller increments of `step_angle` because motion
    planning is more tolerant of small per-step targets than of a single large
    jump. Refuses to run if no part is engaged.

    Returns the underlying planner result for the final step; ``-1`` if motion
    planning fails or no part is engaged.
    """
    env = planner.env.unwrapped
    if getattr(env, "engaged_part", None) is None:
        if verbose:
            print("[pure_rotate] no engaged part; an interaction step must run first")
        return -1

    try:
        angle_deg = float(angle_deg)
        step_angle = float(step_angle)
    except Exception:
        angle_deg = 60.0
        step_angle = 15.0
    if abs(angle_deg) < 1e-6:
        return 0

    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    local_axis = axis_map.get(axis, axis_map["z"])

    step_angle = max(1e-3, abs(step_angle))
    num_steps = max(1, int(abs(angle_deg) // step_angle))
    step = np.sign(angle_deg) * step_angle

    agent = env.agent
    last_res = 0
    if verbose:
        print(f"[pure_rotate] total={angle_deg}°, step={step}°, n={num_steps}, axis={axis}")

    for _i in range(num_steps):
        tcp = agent.tcp.pose
        pos, quat = tcp.p.cpu().numpy()[0], tcp.q.cpu().numpy()[0]
        # SAPIEN quaternion is [w, x, y, z]; scipy wants [x, y, z, w].
        current_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        delta_rot = R.from_rotvec(local_axis * np.deg2rad(step))
        new_rot = current_rot * delta_rot  # right-multiply: TCP-local axis
        new_quat_xyzw = new_rot.as_quat()
        new_quat = np.array(
            [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
        )
        next_pose = sapien.Pose(pos, new_quat)
        last_res = planner.move_to_pose_with_screw(next_pose)
        if last_res == -1 or last_res is False:
            if verbose:
                print(f"[pure_rotate] step {_i + 1}/{num_steps} failed")
            return -1
        time.sleep(0.05)

    return last_res


@register_skill("insert", affordances=["insertable"], group="atomic",
                part_arg=None, phase="continuation",
                description="Move the held item to a target pose and then push along an "
                            "axis to simulate insertion. Two phases: (1) approach a "
                            "standoff above the target along -axis; (2) translate by "
                            "axis*depth into the target. Target may be supplied as a "
                            "world-frame [x,y,z] (`target_pos`) or as a named part on "
                            "the env (`target_part`, looked up via grasp_poses.json).")
def insert(
    planner,
    target_pos=None,
    target_part: str = None,
    axis=(0.0, 0.0, -1.0),
    depth: float = 0.05,
    approach_offset: float = 0.06,
    verbose: bool = False,
    **_,
):
    """Two-phase insertion of an already-engaged item.

    Returns the planner's last result; ``-1`` on motion-plan failure, on
    missing target, or when no part is engaged.

    Args:
        target_pos: optional ``[x, y, z]`` world-frame pose to insert into.
            Takes precedence over ``target_part`` when both are given.
        target_part: optional name of a part on the current env whose
            grasp_poses entry will supply the target position. Resolved via
            :func:`get_grasp_pose_from_config`.
        axis: world-frame insertion direction (defaults to ``[0, 0, -1]``,
            i.e. straight down). Normalised internally.
        depth: distance to push *along* ``axis`` past the target pose to
            simulate full insertion.
        approach_offset: standoff distance *against* ``axis`` at which the
            approach (first-phase) motion targets — leaves room for a clean
            screw motion into the slot.
    """
    env = planner.env.unwrapped
    if getattr(env, "engaged_part", None) is None:
        if verbose:
            print("[insert] no engaged part; an interaction step must run first")
        return -1

    # Resolve target position.
    target_xyz = None
    if target_pos is not None:
        try:
            target_xyz = np.asarray(target_pos, dtype=float).ravel()[:3]
        except Exception as exc:
            if verbose:
                print(f"[insert] bad target_pos {target_pos!r}: {exc}")
            return -1
    elif target_part is not None:
        try:
            pose = get_grasp_pose_from_config(env, target_part, grasp_id=0)
            if pose is None:
                raise ValueError(f"no grasp pose found for part {target_part!r}")
            target_xyz = np.asarray(pose.p, dtype=float).ravel()[:3]
        except Exception as exc:
            if verbose:
                print(f"[insert] couldn't resolve target_part {target_part!r}: {exc}")
            return -1
    else:
        if verbose:
            print("[insert] need target_pos or target_part; aborting")
        return -1

    try:
        axis_v = np.asarray(axis, dtype=float).ravel()[:3]
    except Exception:
        axis_v = np.array([0.0, 0.0, -1.0])
    axis_norm = float(np.linalg.norm(axis_v))
    if axis_norm < 1e-9:
        axis_v = np.array([0.0, 0.0, -1.0])
    else:
        axis_v = axis_v / axis_norm
    try:
        depth = float(depth)
        approach_offset = float(approach_offset)
    except Exception:
        depth = 0.05
        approach_offset = 0.06

    current_tcp_pose = env.agent.tcp.pose
    current_quat = current_tcp_pose.q.cpu().numpy()[0]

    approach_pos = target_xyz - axis_v * approach_offset
    insert_pos = target_xyz + axis_v * depth

    approach_pose = sapien.Pose(approach_pos.astype(np.float32), current_quat)
    if verbose:
        print(f"[insert] phase 1 approach -> {approach_pos.tolist()} (standoff {approach_offset:.3f}m)")
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(approach_pose)
        if res == -1:
            if verbose:
                print("[insert] approach failed (screw + RRTConnect)")
            return -1

    insert_pose = sapien.Pose(insert_pos.astype(np.float32), current_quat)
    if verbose:
        print(f"[insert] phase 2 insert  -> {insert_pos.tolist()} (depth {depth:.3f}m)")
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1 and verbose:
        print("[insert] insertion motion failed")
    return res


@register_skill("draw_triangle", affordances=["drawable"], group="bespoke", part_arg=None, phase="bundle",
                 description="Bespoke triangle drawing plan for DrawTriangleEnv.")
def draw_triangle(planner, seed=None, part_name=None, debug=False, verbose=False):
    env = planner.env.unwrapped
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    rot = list(env.agent.tcp.pose.get_q()[0].cpu().numpy())

    # -------------------------------------------------------------------------- #
    # Move to first vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 0].numpy()), q=rot)
    ipdb.set_trace()
    res = planner.move_to_pose_with_screw(reach_pose)
    # -------------------------------------------------------------------------- #
    # Move to second vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 1]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to third vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 2]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move back to first vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 0]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    planner.close()
    return res

def _rotate_around_axis(pose, axis_point, axis_direction, angle_deg):
    """Rotate ``pose`` about an axis and return the new sapien.Pose."""
    angle_rad = np.deg2rad(angle_deg)
    pos, quat = pose.p, pose.q
    offset = pos - axis_point
    rot = R.from_rotvec(axis_direction * angle_rad)
    new_pos = axis_point + rot.apply(offset)
    current_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    new_quat_xyzw = (rot * current_rot).as_quat()
    new_quat = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])
    return sapien.Pose(new_pos, new_quat)

def open_door(
    planner,
    grasp_pose,
    *,
    pregrasp_height_offset=0.2,
    min_pregrasp_height=0.3,
    final_grasp_offset=-0.02,
    return_to_initial=True,
    verbose=True,
):
    """
    Door-opening skill: approach → close gripper → twist the handle 90° about its axis → pull back 0.2 m → optionally return to start.

    Returns ``(success, info)``.
    """
    agent = planner.env.agent
    tcp = agent.tcp.pose
    p0, q0 = tcp.p.cpu().numpy()[0], tcp.q.cpu().numpy()[0]
    initial_pose = sapien.Pose(p0, q0)
    target_pos = grasp_pose.p

    def log(msg):
        if verbose:
            print(f"[OpenDoor] {msg}")

    try:
        safe_z = max(min_pregrasp_height, min(target_pos[2], p0[2] + pregrasp_height_offset))
        final_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + final_grasp_offset])
        final_grasp_pose = sapien.Pose(final_pos, grasp_pose.q)

        log("moving to grasp pose...")
        if not planner.move_to_pose_with_screw(final_grasp_pose):
            return False, {"error": "could not reach grasp pose"}

        log("closing gripper...")
        _, _, _, _, info = planner.close_gripper()
        grasp_status = info.get("is_grasped", "unknown")
        time.sleep(1)

        axis_y, axis_z = 0.26732, 0.51634
        axis_direction = np.array([1.0, 0.0, 0.0])
        step_angle, num_steps = 30, 3
        log("twisting handle...")
        for _ in range(num_steps):
            tcp = agent.tcp.pose
            pos, quat = tcp.p.cpu().numpy()[0], tcp.q.cpu().numpy()[0]
            axis_point = np.array([pos[0], axis_y, axis_z])
            next_pose = _rotate_around_axis(sapien.Pose(pos, quat), axis_point, axis_direction, step_angle)
            if not planner.move_to_pose_with_screw(next_pose):
                break
            time.sleep(0.1)

        tcp = agent.tcp.pose
        pos, quat = tcp.p.cpu().numpy()[0], tcp.q.cpu().numpy()[0]
        pull_pose = sapien.Pose(np.array([pos[0] - 0.2, pos[1], pos[2]]), quat)
        log("pulling back...")
        planner.move_to_pose_with_screw(pull_pose)

        if return_to_initial:
            log("returning to start pose...")
            planner.move_to_pose_with_screw(initial_pose)

        return True, {"is_grasped": grasp_status, "grasp_pose": final_grasp_pose, "info": info}
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return False, {"error": str(e), "exception": e}

@register_skill("rotate", affordances=["rotatable", "graspable"], group="atomic", part_arg="part_name", phase="bundle",
                 description="Grasp a knob/handle and rotate around its revolute axis.")
def rotate_knob(
    planner,
    *,
    part_name=None,
    total_angle=60,
    step_angle=15,
    pause_after_grasp=0.5,
    axis="z",
    do_grasp=True,
    verbose=True,
):
    """
    Knob-rotation skill. With ``do_grasp=True``, performs the grasp internally before rotating.
    """
    env = planner.env.unwrapped
    agent = env.agent

    def log(msg):
        if verbose:
            print(f"[RotateKnob] {msg}")

    try:
        # -------------------------
        # 1. Grasp phase (when do_grasp=True).
        # -------------------------
        if do_grasp and part_name is not None:
            log(f"grasping part: {part_name}")
            planner.open_gripper()
            
            # Resolve the grasp pose (defaults to candidate 0).
            grasp_pose = get_grasp_pose_from_config(env, part_name, grasp_id=0)
            if grasp_pose is None:
                log(f"error: could not get grasp pose for part {part_name}")
                return -1

            # Compute + move to the pre-grasp pose.
            current_tcp_pose = agent.tcp.pose
            current_pos = current_tcp_pose.p.cpu().numpy()[0]
            target_pos = grasp_pose.p

            # Compute + move to the pre-grasp pose (slight -X offset from the actual grasp pose).
            # Apply the offset along world X.
            pre_grasp_offset = 0.03
            pre_grasp_pos = np.array([target_pos[0] - pre_grasp_offset, target_pos[1], target_pos[2]])
            pre_grasp_pose = sapien.Pose(pre_grasp_pos, grasp_pose.q)

            log("moving to pre-grasp pose...")
            if not planner.move_to_pose_with_RRTConnect(pre_grasp_pose):
                log("could not reach pre-grasp pose")
                return -1

            # Move to the final grasp pose (depth reduced to avoid collisions).
            final_grasp_pos = np.array([target_pos[0], target_pos[1], target_pos[2]])
            final_grasp_pose = sapien.Pose(final_grasp_pos, grasp_pose.q)
            
            log("moving to final grasp pose...")
            if not planner.move_to_pose_with_screw(final_grasp_pose):
                log("could not reach final grasp pose")
                return -1

            # Close the gripper.
            log("closing gripper...")
            planner.close_gripper()

        # -------------------------
        # 2. Rotation phase.
        # -------------------------
        # Brief pause after the grasp so the object settles.
        if pause_after_grasp > 0:
            log(f"settling for {pause_after_grasp:.2f}s after grasp...")
            time.sleep(pause_after_grasp)

        axis_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        local_axis = axis_map.get(axis, axis_map["z"])

        num_steps = max(1, int(abs(total_angle) // abs(step_angle)))
        step = np.sign(total_angle) * abs(step_angle)
        log(f"rotating: total {total_angle}°, step {step}°, {num_steps} steps")

        for _ in range(num_steps):
            tcp = agent.tcp.pose
            pos, quat = tcp.p.cpu().numpy()[0], tcp.q.cpu().numpy()[0]

            # Current rotation (world ← tcp). sapien quat is [w, x, y, z]; scipy wants [x, y, z, w].
            current_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            delta_rot = R.from_rotvec(local_axis * np.deg2rad(step))
            new_rot = current_rot * delta_rot  # rotate in the TCP's own frame
            new_quat_xyzw = new_rot.as_quat()
            new_quat = np.array(
                [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
            )

            next_pose = sapien.Pose(pos, new_quat)
            last_res = planner.move_to_pose_with_screw(next_pose)
            if not last_res:
                log("move failed; aborting rotation early")
                return -1
            time.sleep(0.1)

        log("rotation complete")
        if last_res is not None and isinstance(last_res, tuple) and len(last_res) == 5:
            # Check success using environment evaluation
            eval_result = env.evaluate()
            last_res[4].update(eval_result) # Merge evaluation results into info
        return last_res
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return -1

@register_skill("put_blocks_in_box_and_take_out_memory", phase="bundle",
                 affordances=["graspable", "placeable"], group="composite", part_arg=None,
                 description="Long-horizon: place two blocks in a box, then retrieve the first.")
def put_blocks_in_box_and_take_out_memory(planner, seed=None, part_name=None, debug=False, verbose=False):
    """
    Long-horizon task: put red + blue cubes in the box, then retrieve the one placed first.

    Flow:
    1. Decide which cube goes first based on ``put_order``.
    2. Grasp + drop the first cube into the box.
    3. Grasp + drop the second cube into the box.
    4. Grasp the first cube and put it back outside the box.

    Args:
        planner: motion-planning solver.
        seed: random seed.
        part_name: unused (kept for skill-registry signature compat).
        debug: debug mode toggle.
        verbose: emit progress logs.

    Returns:
        int: 0 on success, -1 on failure.
    """
    env = planner.env.unwrapped
    
    if verbose:
        print("=" * 60)
        print("starting long-horizon task: put cubes in box, retrieve one")
        print("=" * 60)
    
    # Pull the cube + box actors from the env.
    red_cube = env.red_cube
    blue_cube = env.blue_cube
    box = env.box
    
    # Read the put-order tensor.
    put_order = env.put_order[0].item() if hasattr(env.put_order, 'item') else env.put_order[0]
    # Resolve which cube is first vs second based on put_order.
    if put_order == 0:
        first_cube = red_cube
        second_cube = blue_cube
        first_name = "red"
        second_name = "blue"
    else:
        first_cube = blue_cube
        second_cube = red_cube
        first_name = "blue"
        second_name = "red"
    
    if verbose:
        print(f"put order: {first_name} first, then {second_name}")
        print(f"retrieve order: {first_name}")
    
    FINGER_LENGTH = 0.025
    
    # ========================================================================
    # Stage 1: grasp + place the first cube in the box.
    # ========================================================================
    if verbose:
        print(f"\n[Stage 1] grasping {first_name} cube and placing in box")
    
    # 1.1 Compute the grasp pose for the first cube.
    obb = get_actor_obb(first_cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose_1 = env.agent.build_grasp_pose(approaching, closing, center)
    
    # 1.2 Move to pre-grasp pose.
    reach_pose_1 = grasp_pose_1 * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_1)
    if res == -1:
        if verbose:
            print(f"FAIL: could not reach {first_name} cube pre-grasp pose")
        return -1
    
    # 1.3 Grasp the first cube.
    res = planner.move_to_pose_with_screw(grasp_pose_1)
    if res == -1:
        if verbose:
            print(f"FAIL: could not grasp {first_name} cube")
        return -1
    planner.close_gripper()
    
    if verbose:
        print(f"OK: grasped {first_name} cube")
    
    # 1.4 Lift the cube.
    lift_pose_1 = grasp_pose_1 * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(lift_pose_1)
    if res == -1:
        if verbose:
            print(f"FAIL: could not lift {first_name} cube")
        return -1
    
    # 1.5 Move above the box.
    box_pose = box.pose.sp if hasattr(box.pose, 'sp') else box.pose
    box_center = np.array(box_pose.p)
    
    # Compute the place position (above the box centre).
    place_pos_1 = box_center.copy()
    place_pos_1[2] = box_center[2] + 0.15  # 15 cm above the box
    
    # Offset slightly so the two cubes don't perfectly overlap.
    place_pos_1[0] += 0.02  # +2 cm along world X
    
    current_tcp_pose = env.agent.tcp.pose
    current_quat = current_tcp_pose.q.cpu().numpy()[0]
    place_pose_1 = sapien.Pose(place_pos_1, current_quat)
    
    res = planner.move_to_pose_with_screw(place_pose_1)
    if res == -1:
        if verbose:
            print(f"FAIL: could not move above the box")
        return -1
    
    # 1.6 Drop the first cube into the box.
    place_down_pos_1 = place_pos_1.copy()
    place_down_pos_1[2] = box_center[2] + 0.06  # inside the box
    place_down_pose_1 = sapien.Pose(place_down_pos_1, current_quat)
    
    res = planner.move_to_pose_with_screw(place_down_pose_1)
    if res == -1:
        if verbose:
            print(f"FAIL: could not drop {first_name} cube")
        return -1
    
    planner.open_gripper()
    
    if verbose:
        print(f"OK: placed {first_name} cube in the box")
    
    # 1.7 Lift the gripper away.
    retreat_pose_1 = place_down_pose_1 * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(retreat_pose_1)
    
    # ========================================================================
    # Stage 2: grasp + place the second cube in the box.
    # ========================================================================
    if verbose:
        print(f"\n[Stage 2] grasping {second_name} cube and placing in box")
    
    # 2.1 Compute the grasp pose for the second cube.
    obb = get_actor_obb(second_cube)
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose_2 = env.agent.build_grasp_pose(approaching, closing, center)
    
    # 2.2 Move to pre-grasp pose.
    reach_pose_2 = grasp_pose_2 * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_2)
    if res == -1:
        if verbose:
            print(f"FAIL: could not reach {second_name} cube pre-grasp pose")
        return -1
    
    # 2.3 Grasp the second cube.
    res = planner.move_to_pose_with_screw(grasp_pose_2)
    if res == -1:
        if verbose:
            print(f"FAIL: could not grasp {second_name} cube")
        return -1
    planner.close_gripper()
    
    if verbose:
        print(f"OK: grasped {second_name} cube")
    
    # 2.4 Lift the cube.
    lift_pose_2 = grasp_pose_2 * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(lift_pose_2)
    if res == -1:
        if verbose:
            print(f"FAIL: could not lift {second_name} cube")
        return -1
    
    # 2.5 Move above the box (offset to the opposite side).
    place_pos_2 = box_center.copy()
    place_pos_2[2] = box_center[2] + 0.15
    place_pos_2[0] -= 0.02  # -2 cm along world X to clear the first cube
    
    place_pose_2 = sapien.Pose(place_pos_2, current_quat)
    res = planner.move_to_pose_with_screw(place_pose_2)
    if res == -1:
        if verbose:
            print(f"FAIL: could not move above the box")
        return -1
    
    # 2.6 Drop the second cube into the box.
    place_down_pos_2 = place_pos_2.copy()
    place_down_pos_2[2] = box_center[2] + 0.06
    place_down_pose_2 = sapien.Pose(place_down_pos_2, current_quat)
    
    res = planner.move_to_pose_with_screw(place_down_pose_2)
    if res == -1:
        if verbose:
            print(f"FAIL: could not drop {second_name} cube")
        return -1
    
    planner.open_gripper()
    
    if verbose:
        print(f"OK: placed {second_name} cube in the box")
    
    # 2.7 Lift the gripper away.
    retreat_pose_2 = place_down_pose_2 * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(retreat_pose_2)
    
    # ========================================================================
    # Stage 3: retrieve the first cube from the box.
    # ========================================================================
    if verbose:
        print(f"\n[Stage 3] retrieving {first_name} cube from the box")
    
    # 3.1 Move above the first cube.
    first_cube_pose = first_cube.pose.sp if hasattr(first_cube.pose, 'sp') else first_cube.pose
    first_cube_pos = np.array(first_cube_pose.p)
    
    above_first_pos = first_cube_pos.copy()
    above_first_pos[2] += 0.08  # 8 cm above the cube
    above_first_pose = sapien.Pose(above_first_pos, current_quat)
    
    res = planner.move_to_pose_with_screw(above_first_pose)
    if res == -1:
        if verbose:
            print(f"FAIL: could not move above {first_name} cube")
        return -1
    
    # 3.2 Descend and grasp the first cube.
    grasp_first_pos = first_cube_pos.copy()
    grasp_first_pos[2] = first_cube_pos[2] + 0.02  # cube centre
    grasp_first_pose = sapien.Pose(grasp_first_pos, current_quat)
    
    res = planner.move_to_pose_with_screw(grasp_first_pose)
    if res == -1:
        if verbose:
            print(f"FAIL: could not grasp {first_name} cube")
        return -1
    
    planner.close_gripper()
    
    if verbose:
        print(f"OK: grasped {first_name} cube")
    
    lift_out_pose = grasp_first_pose * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(lift_out_pose)
    if res == -1:
        if verbose:
            print(f"FAIL: could not lift {first_name} cube")
        return -1
    
    outside_pos = box_center.copy()
    outside_pos[0] -= 0.25  # 25 cm outside the box
    outside_pos[2] = box_center[2] + 0.15
    outside_pose = sapien.Pose(outside_pos, current_quat)
    
    res = planner.move_to_pose_with_screw(outside_pose)
    if res == -1:
        if verbose:
            print(f"FAIL: could not move outside the box")
        return -1
    
    place_outside_pos = outside_pos.copy()
    place_outside_pos[2] = 0.02 
    place_outside_pose = sapien.Pose(place_outside_pos, current_quat)
    
    res = planner.move_to_pose_with_screw(place_outside_pose)
    if res == -1:
        if verbose:
            print(f"FAIL: could not drop {first_name} cube")
    
    planner.open_gripper()
    
    if verbose:
        print(f"OK: retrieved {first_name} cube from the box")
    
    # 3.6 Lift the gripper away.
    final_retreat_pose = place_outside_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(final_retreat_pose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("OK: long-horizon task complete!")
        print("=" * 60)
    
    planner.close()
    return res

@register_skill("take_out_and_grasp_part_into_box", phase="bundle",
                 affordances=["graspable", "placeable"], group="composite", part_arg=None,
                 description="Long-horizon: take an object out of one container and place it into another.")
def take_out_and_grasp_part_into_box(planner, seed=None, part_name=None, debug=False, verbose=False):
    """
    Long-horizon: take a cube out of the box, then place the target object inside via grasp_part.

    Stage 1: take the cube out of the box (OBB grasp → place on table).
    Stage 2: grasp the target object via grasp_part (config-driven, part-aware).
    Stage 3: place the already-grasped target inside the box.
    """

    env = planner.env.unwrapped

    if verbose:
        print("=" * 60)
        print("starting long-horizon task: remove cube, then place target via grasp_part")
        print("=" * 60)

    FINGER_LENGTH = 0.025

    cube = env.cube
    box = env.box

    if part_name is None:
        part_name = getattr(env, 'part_name', None)

    current_tcp_pose = env.agent.tcp.pose
    current_quat = current_tcp_pose.q.cpu().numpy()[0]

    # ========================================================================
    # Stage 1: take the cube out of the box.
    # ========================================================================
    if verbose:
        print("\n[Stage 1] taking the cube out of the box")

    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose_cube = env.agent.build_grasp_pose(approaching, closing, center)

    reach_pose = grasp_pose_cube * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not move above the cube")
        return -1

    grasp_pose_cube_down = sapien.Pose(
        p=grasp_pose_cube.p + np.array([0, 0, -0.02]),
        q=grasp_pose_cube.q,
    )
    res = planner.move_to_pose_with_screw(grasp_pose_cube_down)
    if res == -1:
        if verbose:
            print("FAIL: could not grasp the cube")
        return -1
    planner.close_gripper()
    if verbose:
        print("OK: grasped cube")

    lift_pose = grasp_pose_cube_down * sapien.Pose([0, 0, -0.18])
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not lift the cube")
        return -1

    box_pose = box.pose.sp if hasattr(box.pose, 'sp') else box.pose
    box_center = np.array(box_pose.p)

    outside_pos = box_center.copy()
    outside_pos[0] -= 0.28   # bias toward the arm side
    outside_pos[2] = box_center[2] + 0.18
    outside_over_pose = sapien.Pose(outside_pos, current_quat)

    res = planner.move_to_pose_with_screw(outside_over_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not move above the area outside the box")
        return -1

    place_pos = outside_pos.copy()
    place_pos[2] = 0.025   # table height
    place_pose = sapien.Pose(place_pos, current_quat)
    res = planner.move_to_pose_with_screw(place_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not place the cube")
        return -1
    planner.open_gripper()
    if verbose:
        print("OK: cube removed from box and placed on the table")

    # 1.7 Lift the gripper away from the cube.
    retreat_cube = place_pose * sapien.Pose([0, 0, -0.12])
    planner.move_to_pose_with_screw(retreat_cube)

    # ========================================================================
    # Stage 2: grasp the target object via grasp_part.
    # ========================================================================
    if verbose:
        print("\n[Stage 2] grasping the target via grasp_part")

    # grasp_part 
    config, grasp_list = get_grasp_list(env, part_name)
    grasp_id = np.random.randint(0, len(grasp_list))
    current_grasp_pose = get_grasp_pose_from_config(env, part_name, grasp_id=grasp_id)

    target_pos = current_grasp_pose.p
    target_quat = current_grasp_pose.q

    rotation_matrix = t3d.quaternions.quat2mat(target_quat)
    approach_direction = rotation_matrix[:, 2]  

    pre_grasp_distance = 0.08
    pre_grasp_pos = target_pos - approach_direction * pre_grasp_distance
    pre_grasp_pose = sapien.Pose(pre_grasp_pos, target_quat)

    res = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not reach target object pre-grasp pose")
        return -1

    grasp_pose_target = sapien.Pose(current_grasp_pose.p, pre_grasp_pose.q)
    res = planner.move_to_pose_with_screw(grasp_pose_target)
    if res == -1:
        if verbose:
            print("FAIL: could not grasp target object")
        return -1
    planner.close_gripper()
    if verbose:
        print("OK: grasped target object")

    # ========================================================================
    # Stage 3: place the target inside the box.
    # ========================================================================
    if verbose:
        print("\n[Stage 3] placing target into the box")

    box_pose = box.pose.sp if hasattr(box.pose, 'sp') else box.pose
    box_center = np.array(box_pose.p)

    current_tcp_p = env.agent.tcp.pose.sp.p
    current_tcp_q = env.agent.tcp.pose.sp.q

    lift_p = current_tcp_p.copy()
    lift_p[2] += 0.25
    lift_pose = sapien.Pose(lift_p, current_tcp_q)
    res = planner.move_to_pose_with_screw(lift_pose)

    over_box_pos = box_center.copy()
    over_box_pos[2] = box_center[2] + 0.35
    over_box_pose = sapien.Pose(over_box_pos, current_tcp_q)

    res = planner.move_to_pose_with_screw(over_box_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not move above the box")
        return -1

    in_box_pos = box_center.copy()
    in_box_pos[2] = box_center[2] + 0.12   # inside the box
    in_box_pose = sapien.Pose(in_box_pos, current_tcp_q)

    res = planner.move_to_pose_with_screw(in_box_pose)
    if res == -1:
        if verbose:
            print("FAIL: could not place target into the box")
        return -1

    planner.open_gripper()
    if verbose:
        print("OK: target object placed in the box")

    retreat_final = in_box_pose * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(retreat_final)

    if verbose:
        print("\n" + "=" * 60)
        print("OK: long-horizon task complete!")
        print("=" * 60)

    planner.close()
    return res

@register_skill("put_blocks_into_boxes", phase="bundle",
                 affordances=["graspable", "placeable"], group="composite", part_arg=None,
                 description="Long-horizon: place each colored block into its matching box.")
def put_blocks_into_boxes(planner, seed=None, part_name=None, debug=False, verbose=False):
    """
    Long-horizon: put special_cube (a chosen colour) into left_box; the others go into right_box.

    Flow:
    1. Grasp special_cube and place it in left_box.
    2. Grasp others[0] and place it in right_box.
    3. Grasp others[1] and place it in right_box.
    """
    env = planner.env.unwrapped

    special      = env._get_special_cube()
    others       = env._get_other_cubes()
    left_box     = env.left_box
    right_box    = env.right_box
    special_name = env.special_cube
    all_names    = ["red", "blue", "green"]
    other_names  = [n for n in all_names if n != special_name]

    FINGER_LENGTH  = 0.025
    approaching    = np.array([0, 0, -1])
    current_tcp    = env.agent.tcp.pose
    current_quat   = current_tcp.q.cpu().numpy()[0]
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    rng            = np.random.default_rng()

    if verbose:
        print("=" * 60)
        print(f"start: {special_name} → left_box, others → right_box")
        print("=" * 60)

    def _place_cube_in_box(cube, box, cube_label, box_label):
        """Helper: grasp + place a cube into the given box. Returns 0 / -1."""
        if verbose:
            print(f"\ngrasping {cube_label} cube → placing in {box_label} box")

        obb = get_actor_obb(cube)
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
        )
        grasp_pose = env.agent.build_grasp_pose(
            approaching, grasp_info["closing"], grasp_info["center"]
        )

        res = planner.move_to_pose_with_RRTConnect(grasp_pose * sapien.Pose([0, 0, -0.05]))
        if res == -1:
            if verbose: print(f"FAIL: could not reach {cube_label} pre-grasp pose")
            return -1

        res = planner.move_to_pose_with_screw(grasp_pose)
        if res == -1:
            if verbose: print(f"FAIL: could not grasp {cube_label} cube")
            return -1
        planner.close_gripper()
        if verbose: print(f"OK: grasped {cube_label} cube")

        res = planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -0.15]))
        if res == -1:
            if verbose: print(f"FAIL: could not lift {cube_label} cube")
            return -1

        bp = box.pose.sp if hasattr(box.pose, 'sp') else box.pose
        bc = np.array(bp.p)

        dx, dy = rng.uniform(-0.03, 0.03, size=2)
        over_pos = np.array([bc[0] + dx, bc[1] + dy, bc[2] + 0.15])
        down_pos = np.array([bc[0] + dx, bc[1] + dy, bc[2] + 0.055])

        q = env.agent.tcp.pose.q.cpu().numpy()[0]
        res = planner.move_to_pose_with_screw(sapien.Pose(over_pos, q))
        if res == -1:
            if verbose: print(f"FAIL: could not move above {box_label} box")
            return -1

        res = planner.move_to_pose_with_screw(sapien.Pose(down_pos, q))
        if res == -1:
            if verbose: print(f"FAIL: could not drop {cube_label} cube")
            return -1
        planner.open_gripper()
        if verbose: print(f"OK: {cube_label} cube placed in {box_label} box")

        res = planner.move_to_pose_with_screw(sapien.Pose(down_pos, q) * sapien.Pose([0, 0, -0.10]))
        return res

    # stage 1: special_cube -> left_box
    res = _place_cube_in_box(special, left_box, special_name, "left")
    if res == -1:
        return -1

    # stage 2: others[0] -> right_box
    res = _place_cube_in_box(others[0], right_box, other_names[0], "right")
    if res == -1:
        return -1

    # stage 3: others[1] -> right_box
    res =  _place_cube_in_box(others[1], right_box, other_names[1], "right")
    if res == -1:
        return -1

    planner.close()
    return res



@register_skill("assembling_kits", affordances=["placeable"], group="bespoke", part_arg=None, phase="bundle",
                 description="Bespoke kit-assembly plan for AssemblingKitsEnv variants.")
def assembling_kits(planner, seed=None, part_name=None, debug=False, verbose=False, **kwargs):
    """Single-peg insert into AssemblingKits-style task.

    Two-phase strategy:
      A) approach via obj_to_tcp snapshot (gross alignment),
      B) one closed-loop correction at pre_insert using observed peg pose,
      C) insert + open gripper, then 50 settle frames before evaluate.
    """
    env = planner.env.unwrapped
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], \
        env.unwrapped.control_mode

    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.obj)
    approaching = np.array([0, 0, -1])
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.06])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        return res
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper()

    goal_p = env.goal_pos[0].cpu().numpy().astype(np.float64)
    goal_yaw = float(env.goal_rot[0].cpu().numpy())
    cur_yaw = t3d.euler.quat2euler(env.obj.pose.sp.q, axes="sxyz")[2]
    sym = float(env.symmetry[env.object_ids[0]].cpu().numpy())
    if sym > 1e-6:
        k = round((cur_yaw - goal_yaw) / sym)
        goal_yaw_eq = goal_yaw + k * sym
    else:
        goal_yaw_eq = goal_yaw
    goal_obj_pose = sapien.Pose(
        p=[float(goal_p[0]), float(goal_p[1]), 0.0],
        q=euler2quat(0.0, 0.0, goal_yaw_eq),
    )

    # --- Phase A: lift + approach above goal ---
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.10])
    res = planner.move_to_pose_with_screw(lift_pose, refine_steps=5)
    if res == -1:
        return res

    obj_to_tcp = env.obj.pose.sp.inv() * env.agent.tcp.pose.sp
    high_over_goal = goal_obj_pose * sapien.Pose([0, 0, 0.10]) * obj_to_tcp
    pre_insert_pose = goal_obj_pose * sapien.Pose([0, 0, 0.05]) * obj_to_tcp

    res = planner.move_to_pose_with_screw(high_over_goal, refine_steps=5)
    if res == -1:
        return res
    res = planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5)
    if res == -1:
        return res

    # --- Phase B: single correction using observed peg pose ---
    pre_insert_obj_target = goal_obj_pose * sapien.Pose([0, 0, 0.05])
    cur_obj = env.obj.pose.sp
    cur_tcp = env.agent.tcp.pose.sp
    corrected_tcp = pre_insert_obj_target * cur_obj.inv() * cur_tcp
    r = planner.move_to_pose_with_screw(corrected_tcp, refine_steps=8)
    if r == -1:
        return r

    # --- Phase C: insert + release + settle ---
    obj_to_tcp = env.obj.pose.sp.inv() * env.agent.tcp.pose.sp
    insert_pose = goal_obj_pose * sapien.Pose([0, 0, 0.0]) * obj_to_tcp
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1:
        return res

    planner.open_gripper()
    retreat_pose = pre_insert_pose
    res2 = planner.move_to_pose_with_screw(retreat_pose)
    if res2 != -1:
        res = res2
    res3 = planner.move_to_pose_with_screw(retreat_pose, refine_steps=50)
    if res3 != -1:
        res = res3

    planner.close()

    if isinstance(res, tuple) and len(res) == 5:
        eval_result = env.evaluate()
        res[4].update(eval_result)
        if verbose:
            try:
                pdn = float(eval_result["pos_diff_norm"][0].cpu())
                rd = float(eval_result["rot_diff"][0].cpu())
                ins = bool(eval_result["in_slot"][0].cpu())
                suc = bool(eval_result["success"][0].cpu())
                obj_z = float(env.obj.pose.p[0, 2].cpu())
                print(
                    f"[assembling_kits] eval pos_diff_norm={pdn:.4f} rot_diff={rd:.4f} "
                    f"in_slot={ins} obj_z={obj_z:.4f} success={suc}"
                )
            except Exception as e:
                print(f"[assembling_kits] eval print err: {e}")
    return res
