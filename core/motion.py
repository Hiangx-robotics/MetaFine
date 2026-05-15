"""Motion-planning utilities.

Thin facade over :class:`mani_skill.examples.motionplanning.panda.PandaArmMotionPlanningSolver`
that exposes a smaller, opinionated API for grasp-style sequences. Every
public method returns a plain dict that bundles ``success`` with whatever
the underlying planner produced, so callers can fan-out into recovery logic
without try/except boilerplate of their own.
"""

import numpy as np
import sapien
from typing import Optional, Dict, Any, Union
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


class MotionPlanner:
    """High-level motion-planning facade for Panda grasping sequences.

    Wraps :class:`PandaArmMotionPlanningSolver` and adds:

    * uniform ``{success, ...}`` return dicts on every operation,
    * a one-call ``execute_grasp_sequence`` that does approach + grasp +
      optional transport,
    * gripper helpers that pull the relevant fields out of the underlying
      step tuple so callers don't have to remember the layout.
    """

    def __init__(self, env, robot_type: str = "panda", debug: bool = False, vis: bool = True):
        """Build a planner bound to ``env``.

        Args:
            env: a ManiSkill env (must expose ``unwrapped.agent.robot.pose``).
            robot_type: only ``"panda"`` is supported today.
            debug: forward to the underlying solver's debug toggle.
            vis: forward to the underlying solver's visualisation toggle.
        """
        self.env = env
        self.robot_type = robot_type

        if robot_type != "panda":
            raise ValueError(f"Robot type '{robot_type}' not supported. Currently only 'panda' is supported.")
        # Create the underlying Panda motion planner.
        self.planner = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=False,
        )

        # Approach/grasp tuning shared across helpers.
        self.finger_length = 0.025  # Panda finger length in metres.

    def move_to_pose_with_screw(self, target_pose: sapien.Pose) -> Dict[str, Any]:
        """Drive TCP to ``target_pose`` via a screw motion.

        Returns:
            ``{"success": True, "result": <planner result>, "target_pose": pose}``
            on success, ``{"success": False, "error": str, "target_pose": pose}``
            on any planner exception.
        """
        try:
            result = self.planner.move_to_pose_with_screw(target_pose)
            return {
                'success': True,
                'result': result,
                'target_pose': target_pose
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'target_pose': target_pose
            }

    def move_to_grasp_pose(self, grasp_pose: sapien.Pose, pre_grasp_offset: float = 0.05) -> Dict[str, Any]:
        """Drive TCP to a standoff above the grasp pose, along the gripper -Z."""
        # Standoff pose computed in the gripper frame so the approach is
        # aligned with the finger direction regardless of world rotation.
        pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -pre_grasp_offset])

        return self.move_to_pose_with_screw(pre_grasp_pose)

    def close_gripper(self) -> Dict[str, Any]:
        """Close the gripper and unpack the resulting step tuple."""
        try:
            obs, reward, terminated, truncated, info = self.planner.close_gripper()
            return {
                'success': True,
                'obs': obs,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def open_gripper(self) -> Dict[str, Any]:
        """Open the gripper and unpack the resulting step tuple."""
        try:
            obs, reward, terminated, truncated, info = self.planner.open_gripper()
            return {
                'success': True,
                'obs': obs,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def move_to_goal_pose(self, goal_pose: sapien.Pose) -> Dict[str, Any]:
        """Translate TCP to ``goal_pose.p`` while keeping its current orientation.

        Useful for transport after a grasp: the grasp pose's rotation is
        carried through so the held object doesn't tilt during transit.
        """
        current_pose = self.env.agent.tcp.pose
        goal_pose_with_orientation = sapien.Pose(goal_pose.p, current_pose.q)

        return self.move_to_pose_with_screw(goal_pose_with_orientation)

    def execute_grasp_sequence(self, grasp_pose: sapien.Pose,
                             goal_pose: Optional[sapien.Pose] = None,
                             pre_grasp_offset: float = 0.05) -> Dict[str, Any]:
        """Run approach → grasp → (optional) transport.

        Args:
            grasp_pose: the final grasp pose to reach.
            goal_pose: when supplied, transport to this pose after grasping.
            pre_grasp_offset: standoff distance along the gripper -Z axis.

        Returns:
            A dict with ``approach_success``, ``grasp_success``,
            ``transport_success`` (or ``True`` if no goal was given), and an
            aggregate ``overall_success`` flag.  ``steps`` is a per-stage
            list of ``(stage_name, result_dict)`` tuples for downstream
            inspection.
        """
        results = {
            'approach_success': False,
            'grasp_success': False,
            'transport_success': False if goal_pose is not None else True,
            'overall_success': False,
            'steps': []
        }

        # Step 1: approach the pre-grasp standoff.
        print("Step 1: Moving to pre-grasp pose...")
        approach_result = self.move_to_grasp_pose(grasp_pose, pre_grasp_offset)
        results['steps'].append(('approach', approach_result))
        results['approach_success'] = approach_result['success']

        if not approach_result['success']:
            print("Failed to approach grasp pose")
            return results

        # Step 2: close the gripper.
        print("Step 2: Closing gripper...")
        grasp_result = self.close_gripper()
        results['steps'].append(('grasp', grasp_result))
        results['grasp_success'] = grasp_result['success']

        if not grasp_result['success']:
            print("Failed to grasp object")
            return results

        # Confirm contact via the planner's `is_grasped` flag.
        is_grasped = grasp_result['info'].get('is_grasped', False)
        if not is_grasped:
            print("Object not grasped successfully")
            results['grasp_success'] = False
            return results

        # Step 3: transport (only when a goal_pose was supplied).
        if goal_pose is not None:
            print("Step 3: Moving to goal pose...")
            transport_result = self.move_to_goal_pose(goal_pose)
            results['steps'].append(('transport', transport_result))
            results['transport_success'] = transport_result['success']

            if not transport_result['success']:
                print("Failed to transport to goal")
                return results

        # Step 4: optional gripper release — left commented because callers
        # often want to retain the grasp for subsequent skills.
        # release_result = self.open_gripper()
        # results['steps'].append(('release', release_result))

        results['overall_success'] = True
        print("Grasp sequence completed successfully!")
        return results

    def update_grasp_visual(self, pose: sapien.Pose) -> bool:
        """Refresh the planner's grasp-pose overlay to ``pose``."""
        try:
            self.planner._update_grasp_visual(pose)
            return True
        except Exception as e:
            print(f"Failed to update grasp visual: {e}")
            return False

    def get_current_tcp_pose(self) -> sapien.Pose:
        """Return the current TCP pose."""
        return self.env.agent.tcp.pose

    def get_current_joint_positions(self) -> np.ndarray:
        """Return the current joint positions (full robot qpos vector)."""
        return self.env.agent.robot.get_qpos()

    def reset_planner(self):
        """Reset planner state. Currently a no-op; reserved for future use."""
        pass
