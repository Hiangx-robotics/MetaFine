import sapien.render
import gymnasium as gym
import numpy as np
import sapien
from typing import Tuple, List, Optional, Sequence, Any
import transforms3d as t3d


def change_table_texture(table_actor, texture_path):
    """Swap a ManiSkill table actor's albedo texture.

    Args:
        table_actor: a ManiSkill Actor wrapping one or more SAPIEN entities.
        texture_path: PNG/JPG file used as the new base_color_texture.
    """
    # ManiSkill Actor wraps one entity per parallel scene, so iterate.
    for obj in table_actor._objs:
        # Grab the render component (skip rigid-body-only entities).
        render_component = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_component is not None:
            new_texture = sapien.render.RenderTexture2D(
                filename=texture_path,
                mipmap_levels=1,
            )
            render_component.material.base_color_texture = new_texture

            # Optional knobs: callers can tweak material parameters by
            # editing render_component.material directly.
            # render_component.material.base_color = [1.0, 1.0, 1.0, 1.0]
            # render_component.material.metallic = 0.0
            # render_component.material.roughness = 0.8


def apply_table_texture(env):
    """Apply a fixed JPG texture to the env's table.

    Convenience wrapper used by interactive demos; the hardcoded
    texture_path is a remnant of the original development setup and
    callers in scripts can pass through to :func:`change_table_texture`
    directly with their own asset.
    """
    table = env.unwrapped.scene_builder.table
    texture_path = "/home/robot/workspace/texture_0.jpg"
    change_table_texture(table, texture_path)

def _as_pose(p_vec, q_vec):
    """Safe constructor for sapien.Pose from array-like or tensor-like inputs."""
    def _to_np(x):
        if isinstance(x, np.ndarray):
            return x
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        if hasattr(x, 'numpy'):
            try:
                return x.numpy()
            except Exception:
                pass
        if hasattr(x, 'cpu'):
            try:
                maybe = x.cpu()
                if hasattr(maybe, 'numpy'):
                    return maybe.numpy()
            except Exception:
                pass
        return np.asarray(x)

    p = _to_np(p_vec).ravel().astype(np.float32)
    q = _to_np(q_vec).ravel().astype(np.float32)
    return sapien.Pose(p, q)

def _quat_slerp(q0, q1, t: float):
    """Slerp between quaternions in (w, x, y, z) format."""
    q0 = np.asarray(q0, dtype=float).ravel()
    q1 = np.asarray(q1, dtype=float).ravel()
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        res = q0 + t * (q1 - q0)
        res = res / np.linalg.norm(res)
        return res.astype(np.float32)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q0) + (s1 * q1)
    return (res / np.linalg.norm(res)).astype(np.float32)

def plan_arc_path(start_pose: sapien.Pose,
                  axis_pos,
                  axis_vec,
                  angle_deg: float,
                  steps: int = 20,
                  keep_orientation: bool = True,
                  radius_scale: float = 1.0,
                  do_reverse: bool = False,
                  reverse_angle_deg: Optional[float] = None,
                  reverse_steps: Optional[int] = None,
                  reverse_keep_orientation: Optional[bool] = None,
                  reverse_axis_pos: Optional[Sequence] = None,
                  reverse_axis_vec: Optional[Sequence] = None) -> Sequence[sapien.Pose]:
    """
    Plan a circular-arc sequence of sapien.Pose waypoints.

    - start_pose: sapien.Pose (start)
    - axis_pos: hinge center in world coords (array-like 3)
    - axis_vec: hinge axis (array-like). If length==4 and normalized, treated as quat -> X axis extracted.
    - angle_deg: angle in degrees for the main arc (signed).
    - steps: number of interpolation steps for the main arc (excluding start_pose).
    - keep_orientation: if True, each waypoint keeps start_pose.q (only positions change).
                        if False, orientations are rotated together with positions.
    - radius_scale: multiply radius (start_p - axis_center) by this factor to tighten/loosen arc.
    - do_reverse: if True, append a reverse arc after main arc.
    - reverse_angle_deg: angle for reverse arc (defaults to -angle_deg).
    - reverse_steps: steps for reverse arc (defaults to steps).
    - reverse_keep_orientation: whether reverse arc keeps orientation (defaults to keep_orientation)
    - reverse_axis_pos: custom rotation center for reverse arc (defaults to axis_pos). 
                       Can be set to gripper end position for pivot-like motion.
    - reverse_axis_vec: custom rotation axis for reverse arc (defaults to axis_vec).

    Returns a list of sapien.Pose with length 1 + steps (+ reverse_steps if appended).
    """
    path = []
    angle_rad = np.radians(angle_deg)

    axis = np.asarray(axis_vec).ravel()
    if axis.size == 0:
        raise ValueError("axis_vec is empty")
    if axis.size == 4:
        if np.isclose(np.linalg.norm(axis), 1.0, atol=1e-3):
            try:
                rot_from_quat = t3d.quaternions.quat2mat(axis)
                axis = np.asarray(rot_from_quat)[:, 0].ravel()
            except Exception:
                axis = axis[:3]
        else:
            axis = axis[:3]
    if axis.size < 3:
        raise ValueError(f"Invalid axis vector shape: {axis.shape}")
    axis = axis[:3]
    axis = axis / np.linalg.norm(axis)

    axis_center = np.asarray(axis_pos).ravel()[:3]
    start_p = np.asarray(start_pose.p).ravel()[:3]

    r_vec = start_p - axis_center
    if not np.isclose(radius_scale, 1.0):
        r_vec = r_vec * float(radius_scale)

    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        if hasattr(x, 'numpy'):
            try:
                return x.numpy()
            except Exception:
                pass
        if hasattr(x, 'cpu'):
            try:
                maybe = x.cpu()
                if hasattr(maybe, 'numpy'):
                    return maybe.numpy()
            except Exception:
                pass
        return np.asarray(x)

    # include start pose
    path.append(start_pose)

    for i in range(1, steps + 1):
        current_angle = (i / float(steps)) * angle_rad
        R = t3d.axangles.axangle2mat(axis, current_angle)
        I = np.eye(3)
        trans = (I - R) @ axis_center

        T_rot = np.eye(4)
        T_rot[:3, :3] = R
        T_rot[:3, 3] = trans

        start_T_raw = _to_numpy(start_pose.to_transformation_matrix())
        if isinstance(start_T_raw, np.ndarray) and start_T_raw.ndim == 3 and start_T_raw.shape[0] == 1:
            start_T_raw = start_T_raw[0]
        start_T = np.asarray(start_T_raw, dtype=float)

        try:
            if not np.isclose(radius_scale, 1.0):
                start_T[:3, 3] = axis_center + r_vec
            else:
                start_T[:3, 3] = start_p
        except Exception:
            pass

        new_T = T_rot @ start_T
        new_p = new_T[:3, 3]
        new_rot = new_T[:3, :3]

        if keep_orientation:
            try:
                start_q = _to_numpy(start_pose.q).ravel().astype(np.float32)
            except Exception:
                # fallback: convert rotation matrix to quat
                start_q = t3d.quaternions.mat2quat(start_T[:3, :3])
            new_q = start_q
        else:
            new_q = t3d.quaternions.mat2quat(new_rot)

        path.append(_as_pose(new_p, new_q))

    # optional reverse arc
    if do_reverse:
        if reverse_angle_deg is None:
            reverse_angle_deg = -float(angle_deg)
        if reverse_steps is None:
            reverse_steps = int(steps)
        if reverse_keep_orientation is None:
            reverse_keep_orientation = bool(keep_orientation)

        rev_start_pose = path[-1]
        rev_angle_rad = np.radians(reverse_angle_deg)
        
        # Use custom axis for reverse arc if provided
        if reverse_axis_vec is not None:
            rev_axis = np.asarray(reverse_axis_vec).ravel()
            if rev_axis.size == 0:
                raise ValueError("reverse_axis_vec is empty")
            if rev_axis.size == 4:
                if np.isclose(np.linalg.norm(rev_axis), 1.0, atol=1e-3):
                    try:
                        rot_from_quat = t3d.quaternions.quat2mat(rev_axis)
                        rev_axis = np.asarray(rot_from_quat)[:, 0].ravel()
                    except Exception:
                        rev_axis = rev_axis[:3]
                else:
                    rev_axis = rev_axis[:3]
            if rev_axis.size < 3:
                raise ValueError(f"Invalid reverse_axis_vec shape: {rev_axis.shape}")
            rev_axis = rev_axis[:3]
            rev_axis = rev_axis / np.linalg.norm(rev_axis)
        else:
            rev_axis = axis
        
        # Use custom center for reverse arc if provided
        if reverse_axis_pos is not None:
            rev_axis_center = np.asarray(reverse_axis_pos).ravel()[:3]
        else:
            rev_axis_center = axis_center

        rev_start_p = np.asarray(rev_start_pose.p).ravel()[:3]
        rev_r_vec = rev_start_p - rev_axis_center
        if not np.isclose(radius_scale, 1.0):
            rev_r_vec = rev_r_vec * float(radius_scale)

        for i in range(1, int(reverse_steps) + 1):
            current_angle = (i / float(reverse_steps)) * rev_angle_rad
            R = t3d.axangles.axangle2mat(rev_axis, current_angle)
            I = np.eye(3)
            trans = (I - R) @ rev_axis_center

            T_rot = np.eye(4)
            T_rot[:3, :3] = R
            T_rot[:3, 3] = trans

            start_T_raw = _to_numpy(rev_start_pose.to_transformation_matrix())
            if isinstance(start_T_raw, np.ndarray) and start_T_raw.ndim == 3 and start_T_raw.shape[0] == 1:
                start_T_raw = start_T_raw[0]
            start_T = np.asarray(start_T_raw, dtype=float)
            try:
                if not np.isclose(radius_scale, 1.0):
                    start_T[:3, 3] = rev_axis_center + rev_r_vec
                else:
                    start_T[:3, 3] = rev_start_p
            except Exception:
                pass

            new_T = T_rot @ start_T
            new_p = new_T[:3, 3]
            new_rot = new_T[:3, :3]

            if reverse_keep_orientation:
                try:
                    start_q = _to_numpy(rev_start_pose.q).ravel().astype(np.float32)
                except Exception:
                    start_q = t3d.quaternions.mat2quat(start_T[:3, :3])
                new_q = start_q
            else:
                new_q = t3d.quaternions.mat2quat(new_rot)

            path.append(_as_pose(new_p, new_q))

    return path

def compute_smart_pregrasp_pose(
    current_tcp_pose: sapien.Pose,
    target_grasp_pose: sapien.Pose,
    env=None,
    min_clearance: float = 0.15,
    max_reach_height: float = 0.6,
    approach_distance: float = 0.1,
    num_candidates: int = 3
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Pick a sensible pregrasp position + candidate orientations.

    Args:
        current_tcp_pose: current TCP pose (sapien.Pose; tensor-backed).
        target_grasp_pose: the planned grasp pose to approach from.
        env: optional env reference (kept for future use, currently unused).
        min_clearance: minimum clearance above the table at the standoff.
        max_reach_height: cap on standoff z to stay inside the workspace.
        approach_distance: how far to retreat along the approach vector
            from the target xy.
        num_candidates: number of orientations to return. The first three
            are deterministic (current orientation, target orientation,
            default top-down); the remainder are small Euler perturbations
            around the target orientation.

    Returns:
        Tuple of ``(pre_grasp_pos, candidate_orientations)``:
        * ``pre_grasp_pos`` — recommended pregrasp xyz (numpy array).
        * ``candidate_orientations`` — list of quaternions in [w,x,y,z]
          order, ordered most→least conservative.
    """

    target_pos = target_grasp_pose.p
    current_pos = current_tcp_pose.p.cpu().numpy()[0]

    # 1. Pick a safe z. Use the lowest of (target+clearance, current+5cm,
    # clearance above table) and never exceed max_reach_height.
    table_height = 0.0  # Tables are at z=0 in our scene builders.
    safe_z_candidates = [
        target_pos[2] + min_clearance,
        current_pos[2] + 0.05,
        min_clearance + table_height,
    ]
    safe_z = min(safe_z_candidates)
    safe_z = min(safe_z, max_reach_height)

    # 2. Pick xy: retreat from the target along the current→target vector
    # so the gripper doesn't slam straight into the part.
    approach_vector = np.array([
        target_pos[0] - current_pos[0],
        target_pos[1] - current_pos[1],
        0,  # planar retreat only
    ])

    if np.linalg.norm(approach_vector) > 0.01:
        approach_vector = approach_vector / np.linalg.norm(approach_vector)
        retreat_distance = min(approach_distance, np.linalg.norm(approach_vector) * 0.5)
        pre_grasp_xy = target_pos[:2] - approach_vector[:2] * retreat_distance
    else:
        # Already on top of the target — offset diagonally so we don't pick
        # a pre-grasp position that overlaps the contact point.
        pre_grasp_xy = target_pos[:2] + np.array([0.02, 0.02])

    pre_grasp_pos = np.array([pre_grasp_xy[0], pre_grasp_xy[1], safe_z])

    # 3. Build the ordered candidate-orientation list.
    candidate_orientations = []

    # Candidate 1: keep the current TCP orientation (most conservative).
    try:
        current_quat = current_tcp_pose.q
        if hasattr(current_quat, 'cpu'):
            current_quat = current_quat.cpu().numpy()
        if current_quat.ndim > 1:
            current_quat = current_quat[0]
        candidate_orientations.append(current_quat)
    except:
        candidate_orientations.append(np.array([1, 0, 0, 0]))

    # Candidate 2: target grasp orientation (most aligned with the actual grasp).
    try:
        target_quat = target_grasp_pose.q
        if hasattr(target_quat, 'cpu'):
            target_quat = target_quat.cpu().numpy()
        if target_quat.ndim > 1:
            target_quat = target_quat[0]
        candidate_orientations.append(target_quat)
    except:
        candidate_orientations.append(np.array([1, 0, 0, 0]))

    # Candidate 3: default top-down ([w, x, y, z] = [1, 0, 0, 0]).
    default_quat = np.array([1, 0, 0, 0])
    candidate_orientations.append(default_quat)

    # Extras: small Euler perturbations of the target orientation, useful
    # when the first three are all IK-infeasible.
    if num_candidates > 3:
        from scipy.spatial.transform import Rotation as R
        for i in range(num_candidates - 3):
            try:
                base_quat = candidate_orientations[1]  # perturb the target quat
                r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])  # wxyz → scipy xyzw
                euler = r.as_euler('xyz', degrees=False)
                perturbed_euler = euler + np.random.normal(0, 0.1, 3)  # ~5.7° one-sigma
                perturbed_r = R.from_euler('xyz', perturbed_euler)
                perturbed_quat_scipy = perturbed_r.as_quat()  # scipy xyzw
                perturbed_quat = np.array([
                    perturbed_quat_scipy[3],  # w
                    perturbed_quat_scipy[0],  # x
                    perturbed_quat_scipy[1],  # y
                    perturbed_quat_scipy[2],  # z
                ])
                candidate_orientations.append(perturbed_quat)
            except:
                candidate_orientations.append(default_quat)

    return pre_grasp_pos, candidate_orientations


def rotation_quaternion_z(theta_degrees: Any):
    """Quaternion for a pure rotation of ``theta_degrees`` about the world Z axis.

    Args:
        theta_degrees: scalar, numpy array, or torch tensor. Tensors are
            detached and copied to host before reading.

    Returns:
        ``np.ndarray`` of shape (4,) in [w, x, y, z] order with only the
        scalar (w) and z components non-zero.
    """
    # Normalise to a plain Python scalar so the rest of the function only
    # deals with one numeric type.
    if hasattr(theta_degrees, 'cpu'):  # torch tensor
        theta_scalar = theta_degrees.detach().cpu().numpy().item()
    elif isinstance(theta_degrees, np.ndarray):
        theta_scalar = theta_degrees.item()
    else:  # plain float / int
        theta_scalar = theta_degrees

    theta_rad = np.radians(theta_scalar)
    w = np.cos(theta_rad / 2)
    z = np.sin(theta_rad / 2)
    return np.array([w, 0, 0, z])

def select_action_subspace(
    action: Any,
    control_mode: str = "pd_ee_delta_pose",
    include_gripper: bool = False,
) -> np.ndarray:
    """
    Select action dimensions used for smoothness metrics.

    - pd_joint_*      : arm dims (first 7) by default
    - pd_ee_delta_pos : xyz (first 3) by default
    - pd_ee_delta_pose: xyz+rpy/rot (first 6) by default
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)

    if control_mode.startswith("pd_joint"):
        if include_gripper:
            return a
        return a[:7] if a.shape[0] >= 7 else a

    if control_mode == "pd_ee_delta_pos":
        if include_gripper:
            return a
        return a[:3] if a.shape[0] >= 3 else a

    if control_mode == "pd_ee_delta_pose":
        if include_gripper:
            return a
        return a[:6] if a.shape[0] >= 6 else a

    return a


def compute_mad(
    actions: Any,
    dt: float = 1.0,
    norm: str = "l2",
    eps: float = 1e-12,
) -> float:
    """
    Compute Mean Action Difference (MAD) as a smoothness metric.

    Args:
        actions: (T, D) action sequence
        dt: control interval in seconds
        norm: 'l2' or 'l1'
    """
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.nan

    da = np.diff(arr, axis=0) / max(float(dt), eps)

    if norm == "l2":
        return float(np.mean(np.linalg.norm(da, ord=2, axis=1)))
    if norm == "l1":
        return float(np.mean(np.linalg.norm(da, ord=1, axis=1)))
    raise ValueError("norm must be 'l1' or 'l2'")


def summarize_metric(values: Any) -> dict:
    """Summarize a metric list with finite-value filtering."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "count": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "count": int(vals.size),
    }