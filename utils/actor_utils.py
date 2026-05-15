import sapien
import numpy as np
from copy import deepcopy
import transforms3d as t3d
from pathlib import Path

# from . import transforms
# from .transforms import *

from sapien import Entity
from sapien.physx import PhysxArticulation, PhysxArticulationLinkComponent

from typing import Literal, Generator


class graspActor:
    """Actor wrapper for managing grasp points only."""

    def __init__(self, actor: Entity, actor_data: dict, mass=0.01):
        self.actor = actor
        self.config = actor_data
        self.set_mass(mass)

    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        return self.actor.get_pose()

    def get_grasp_point(self,
                       part_name: str,
                       idx: int,
                       ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of a grasp point in a part.
        
        Args:
            part_name: name of the grasp part (e.g., "handle", "door")
            idx: index of the grasp point within the part
            ret: return type - "matrix", "list", or "pose"
        
        Returns:
            The grasp point in requested format
        """
        actor_matrix = self.actor.get_pose().to_transformation_matrix()
        
        try:
            grasp_parts = self.config.get("grasp_parts", {})
            if part_name not in grasp_parts:
                return None
            grasp = grasp_parts[part_name][idx]
            local_matrix = np.array(grasp["matrix"])
        except:
            return None
        
        local_matrix[:3, 3] *= np.array(self.config.get("scale", 1.0))
        world_matrix = actor_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3], t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def iter_grasp_parts(self) -> Generator[tuple[str, list], None, None]:
        """Iterate over all grasp parts.
        
        Yields:
            (part_name, grasps_list)
        """
        grasp_parts = self.config.get("grasp_parts", {})
        for part_name, grasps in grasp_parts.items():
            yield part_name, grasps

    def iter_grasp_points(self,
                         part_name: str,
                         ret: Literal["matrix", "list", "pose"] = "list"
    ) -> Generator[tuple[int, np.ndarray | list | sapien.Pose], None, None]:
        """Iterate over all grasp points in a part.
        
        Args:
            part_name: name of the grasp part
            ret: return type for each point
            
        Yields:
            (grasp_id, point_in_requested_format)
        """
        grasp_parts = self.config.get("grasp_parts", {})
        if part_name not in grasp_parts:
            return
        
        for i, grasp in enumerate(grasp_parts[part_name]):
            grasp_id = grasp.get("id", i)
            yield grasp_id, self.get_grasp_point(part_name, i, ret)

    def get_name(self):
        return self.actor.get_name()

    def set_name(self, name):
        self.actor.set_name(name)

    def set_mass(self, mass):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                component.mass = mass


class graspArticulationActor(graspActor):
    """Articulation actor wrapper for managing grasp points only."""

    def __init__(self, actor: PhysxArticulation, actor_data: dict, mass=0.01):
        assert isinstance(actor, PhysxArticulation), "ArticulationActor must be a Articulation"

        self.actor = actor
        self.config = actor_data

        self.link_dict = self.get_link_dict()
        self.set_mass(mass)

    def get_link_dict(self) -> dict[str, PhysxArticulationLinkComponent]:
        link_dict = {}
        for link in self.actor.get_links():
            link_dict[link.get_name()] = link
        return link_dict

    def get_grasp_point(self,
                       part_name: str,
                       idx: int,
                       ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of a grasp point in a part for articulation.
        
        Args:
            part_name: name of the grasp part (e.g., "handle", "door")
            idx: index of the grasp point within the part
            ret: return type - "matrix", "list", or "pose"
        
        Returns:
            The grasp point in requested format
        """
        try:
            grasp_parts = self.config.get("grasp_parts", {})
            if part_name not in grasp_parts:
                return None
            grasp = grasp_parts[part_name][idx]
            local_matrix = np.array(grasp["matrix"])
            base_name = grasp.get("base", "base")
        except:
            return None
        
        local_matrix[:3, 3] *= self.config.get("scale", 1.0)

        link = self.link_dict.get(base_name)
        if link is None:
            return None
        
        link_matrix = link.get_pose().to_transformation_matrix()
        world_matrix = link_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3], t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def set_mass(self, mass, links_name: list[str] = None):
        for link in self.actor.get_links():
            if links_name is None or link.get_name() in links_name:
                link.set_mass(mass)

    def set_properties(self, damping, stiffness, friction=None, force_limit=None):
        for joint in self.actor.get_joints():
            if force_limit is not None:
                joint.set_drive_properties(damping=damping, stiffness=stiffness, force_limit=force_limit)
            else:
                joint.set_drive_properties(
                    damping=damping,
                    stiffness=stiffness,
                )
            if friction is not None:
                joint.set_friction(friction)

    def set_qpos(self, qpos):
        self.actor.set_qpos(qpos)

    def set_qvel(self, qvel):
        self.actor.set_qvel(qvel)

    def get_qlimits(self):
        return self.actor.get_qlimits()

    def get_qpos(self):
        return self.actor.get_qpos()

    def get_qvel(self):
        return self.actor.get_qvel()
