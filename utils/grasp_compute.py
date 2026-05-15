"""Grasp pose computation utilities.

Three concerns live here:

  - :func:`compute_grasp_info_by_point` and :func:`compute_grasp_info_by_obb`
    derive ``(approaching, closing, center, extents)`` from either a target
    point or an oriented bounding box; everything downstream takes one of
    these dicts as input.
  - :func:`get_actor_obb` is the standard way to materialise an OBB for a
    ManiSkill actor so its mesh can be passed into the OBB-based grasp
    planner.
  - :func:`compute_grasp_pose_from_json_matrix` / :func:`get_grasp_list` /
    :func:`get_grasp_pose_from_config` resolve a part-name key in the
    asset's ``grasp_parts`` annotation into a world-frame grasp pose.
"""

import numpy as np
import trimesh
import sapien
import transforms3d.quaternions as t3d_quaternions
from mani_skill.utils.structs import Actor
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.utils import common  # for np_normalize_vector


def compute_grasp_info_by_point(
    grasp_point: np.ndarray,       # target grasp point (x, y, z)
    approaching=(0, 0, -1),        # arm approach direction (default: down)
    closing=(1, 0, 0),             # gripper closing direction (default: +X)
    depth=0.0,                     # offset along the approach axis (finger length)
    ortho=True,                    # whether to orthogonalise closing against approaching
):
    """Build a grasp-info dict from an explicit world-frame point.

    Args:
        grasp_point: target grasp centre in world coordinates, shape (3,).
        approaching: unit-vector along which the gripper approaches the part.
        closing: unit-vector along which the fingers close.
        depth: distance to push the centre along ``approaching`` so the
            gripper's TCP lines up with the contact face once fingers are
            extended.
        ortho: when True, force ``closing`` perpendicular to ``approaching``.

    Returns:
        dict with keys ``approaching``, ``closing``, ``center``, ``extents``.
    """
    # 1. Normalise both direction vectors.
    approaching = np.array(approaching)
    approaching = common.np_normalize_vector(approaching)
    closing = np.array(closing)
    closing = common.np_normalize_vector(closing)

    # 2. Shift the grasp centre to compensate for finger length.
    center = grasp_point + approaching * depth

    # 3. Orthogonalise the closing direction when requested.
    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    # 4. Build the canonical grasp-info dict. ``extents`` is zero here
    # because we don't have an OBB to read it from.
    grasp_info = dict(
        approaching=approaching,
        closing=closing,
        center=center,
        extents=np.array([0.0, 0.0, 0.0]),
    )
    return grasp_info


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
):
    """Same return shape as :func:`compute_grasp_info_by_point` but driven by an OBB.

    Args:
        obb: a trimesh Box representing the part's oriented bounding box.
        approaching: gripper approach direction, normalised internally.
        target_closing: requested closing direction; falls back to +X when None.
        depth: offset along ``approaching`` for finger length.

    Returns:
        dict with keys ``approaching``, ``closing``, ``center``, ``extents``.
    """
    approaching = np.array(approaching)
    approaching = common.np_normalize_vector(approaching)

    center = obb.centroid + approaching * depth

    if target_closing is not None:
        closing = np.array(target_closing)
        closing = common.np_normalize_vector(closing)
        # Project out the component along approaching to ensure orthogonality.
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)
    else:
        # Default: +X projected onto the plane perpendicular to approaching.
        closing = np.array([1, 0, 0])
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching,
        closing=closing,
        center=center,
        extents=obb.extents,
    )

    return grasp_info


def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    """Return the oriented bounding box of a ManiSkill actor.

    Args:
        actor: the ManiSkill Actor whose mesh OBB we need.
        to_world_frame: when True the bbox is expressed in world coordinates,
            otherwise in the actor's local frame.
        vis: when True, open a trimesh viewer to inspect the result.

    Returns:
        ``trimesh.primitives.Box`` covering the actor's mesh.
    """
    mesh = get_component_mesh(actor._objs[0], to_world_frame=to_world_frame)
    assert mesh is not None, f"can not get actor mesh for {actor}"

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def build_grasp_pose_from_info(grasp_info: dict, finger_length: float = 0.025):
    """Turn a grasp-info dict into a fully-specified grasp pose record.

    Args:
        grasp_info: dict with ``approaching``, ``closing``, ``center`` keys
            (as produced by the two compute_grasp_info_* functions above).
        finger_length: Panda finger length used to derive the pre-grasp
            standoff distance.

    Returns:
        dict with the rotation matrix, the pre-grasp offset, and pass-through
        copies of the input fields.
    """
    approaching = grasp_info['approaching']
    closing = grasp_info['closing']
    center = grasp_info['center']

    # Z = approach direction, X = closing direction; rebuild Y, then re-
    # orthogonalise X so the basis is right-handed and exactly orthonormal.
    z_axis = approaching
    x_axis = closing
    y_axis = np.cross(z_axis, x_axis)
    y_axis = common.np_normalize_vector(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = common.np_normalize_vector(x_axis)

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # Note: this function only emits the rotation matrix; callers wrap it
    # into a sapien.Pose at the call site if they need a quaternion.

    return {
        'grasp_center': center,
        'approaching': approaching,
        'closing': closing,
        'rotation_matrix': rotation_matrix,
        'finger_length': finger_length,
        'pre_grasp_offset': finger_length,
    }


def visualize_grasp_pose(env, grasp_pose_info: dict, duration: float = 2.0):
    """Visualise a grasp pose in the env.

    Currently a stub; ``env`` and ``grasp_pose_info`` are accepted for the
    eventual implementation (axis-glyph + contact-point overlay for
    ``duration`` seconds).
    """
    pass


def compute_grasp_pose_from_json_matrix(
    grasp_matrix: np.ndarray,
    target_object: Actor,
    scale: float = 1.0,
) -> sapien.Pose:
    """Promote a part-local 4x4 grasp matrix to a world-frame ``sapien.Pose``.

    Args:
        grasp_matrix: 4x4 affine transform expressed relative to the target
            object's base frame (as stored in ``grasp_poses.json``).
        target_object: the loaded ManiSkill Actor whose world-frame pose
            we'll compose with.
        scale: model scale factor; positions are scaled, rotations are not.

    Returns:
        ``sapien.Pose`` in world coordinates.
    """
    grasp_matrix = np.array(grasp_matrix, dtype=np.float32)

    # Apply scale to the translation column only; rotations are scale-invariant.
    if scale != 1.0:
        grasp_matrix = grasp_matrix.copy()
        grasp_matrix[:3, 3] *= scale

    # Compose: world ← object_base ← grasp (in part local frame).
    object_pose = target_object.get_pose()
    object_matrix = object_pose.to_transformation_matrix()

    world_matrix = object_matrix @ grasp_matrix

    # Slice out the translation + rotation (batch index 0 because target_object
    # pose is batched-by-default).
    position = world_matrix[0, :3, 3].cpu().numpy()
    rotation_matrix = world_matrix[0, :3, :3].cpu().numpy()

    quaternion = t3d_quaternions.mat2quat(rotation_matrix)

    return sapien.Pose(position, quaternion)


def get_grasp_list(env, part_name: str):
    """Return ``(config_dict, grasp_list)`` for the named part on this env.

    Looks up the part's grasp annotations in whichever metadata store the
    env exposes — ``object_config`` for articulated envs, the table scene's
    ``model_data`` for the multi-actor scenes.
    """
    if hasattr(env.unwrapped, 'object_config'):
        config = env.unwrapped.object_config
    elif hasattr(env.unwrapped, 'table_scene') and hasattr(env.unwrapped.table_scene, 'model_data'):
        config = env.unwrapped.table_scene.model_data
    else:
        raise ValueError("could not locate object configuration on env")

    grasp_parts = config.get("grasp_parts", {})
    if part_name not in grasp_parts:
        available_parts = list(grasp_parts.keys())
        raise ValueError(
            f"part '{part_name}' not found; available parts: {available_parts}"
        )

    grasp_list = grasp_parts[part_name]
    return config, grasp_list


def get_grasp_pose_from_config(
    env,
    part_name: str,
    grasp_id: int = 0,
) -> sapien.Pose:
    """Convenience entrypoint: look up a part's grasp candidates and resolve to a world pose.

    Args:
        env: the env whose object_config we'll read.
        part_name: must exist in ``grasp_parts``.
        grasp_id: which candidate within the part's list to use.

    Returns:
        ``sapien.Pose`` in world coordinates.
    """
    config, grasp_list = get_grasp_list(env, part_name)
    if grasp_id >= len(grasp_list):
        raise ValueError(
            f"grasp_id {grasp_id} out of range; part '{part_name}' has {len(grasp_list)} candidates"
        )

    grasp_point_config = grasp_list[grasp_id]
    grasp_matrix = np.array(grasp_point_config["matrix"])
    scale = config.get("scale", 1.0)
    if hasattr(env.unwrapped, 'target_object'):
        target_object = env.unwrapped.target_object
    elif hasattr(env.unwrapped, 'cabinet'):
        target_object = env.unwrapped.cabinet
    else:
        raise ValueError("could not locate target object on env")

    return compute_grasp_pose_from_json_matrix(grasp_matrix, target_object, scale)
