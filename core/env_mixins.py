"""Mixins for MetaFine env classes.

``EvalDREnvMixin`` consolidates the eval-time domain-randomization code that
was previously copy-pasted into every env: per-instance attributes for camera
position / rotation jitter and ambient-light intensity, plus the two helpers
that actually apply the jitter (``_maybe_jitter_camera``) and the override of
``_load_lighting`` that randomizes ambient light when enabled. All defaults
are zero / off, so an env that inherits this mixin without overriding
``_default_sensor_configs`` or ``_load_lighting`` behaves identically to the
pre-refactor baseline.

Usage::

    class MyEnv(EvalDREnvMixin, BaseEnv):
        @property
        def _default_sensor_configs(self):
            eye = np.array([...], dtype=np.float32)
            target = np.array([...], dtype=np.float32)
            eye, target = self._maybe_jitter_camera(eye, target)
            pose = sapien_utils.look_at(eye.tolist(), target.tolist())
            return [CameraConfig("base_camera", pose, ...)]
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R


class EvalDREnvMixin:
    """Adds opt-in eval-time camera + ambient-light jitter to a ManiSkill env.

    Also centralizes the shared MetaFine Franka base pose. The mixin pops six
    ``eval_*`` kwargs from ``__init__`` and stores them as instance
    attributes. Defaults make the mixin a no-op so existing call sites that
    don't pass any ``eval_*`` argument see no behavior change.
    """

    #: Default Panda base offset for MetaFine workspaces. The arm sits 0.615 m
    #: behind the front edge of the table, centered. Override on a subclass
    #: when an env needs a different mount.
    FRANKA_BASE_POSE: Tuple[float, float, float] = (-0.615, 0.0, 0.0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.eval_randomize_camera = bool(kwargs.pop("eval_randomize_camera", False))
        self.eval_camera_pos_jitter = float(kwargs.pop("eval_camera_pos_jitter", 0.0))
        self.eval_camera_rot_jitter_deg = float(kwargs.pop("eval_camera_rot_jitter_deg", 0.0))
        self.eval_randomize_light = bool(kwargs.pop("eval_randomize_light", False))
        self.eval_ambient_low = float(kwargs.pop("eval_ambient_low", 0.5))
        self.eval_ambient_high = float(kwargs.pop("eval_ambient_high", 0.5))
        super().__init__(*args, **kwargs)

    def _maybe_jitter_camera(
        self, eye: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply position + rotation jitter to a camera (eye, target) pair.

        Returns the (possibly jittered) eye and target as float32 arrays. When
        ``eval_randomize_camera`` is disabled this is a no-op.
        """
        if not getattr(self, "eval_randomize_camera", False):
            return eye, target

        pos_jitter = max(0.0, float(getattr(self, "eval_camera_pos_jitter", 0.0)))
        rot_jitter_deg = max(0.0, float(getattr(self, "eval_camera_rot_jitter_deg", 0.0)))

        if pos_jitter > 0:
            eye = eye + np.random.uniform(-pos_jitter, pos_jitter, size=(3,)).astype(np.float32)
            target = target + np.random.uniform(-pos_jitter, pos_jitter, size=(3,)).astype(np.float32)

        if rot_jitter_deg > 0:
            view_vec = target - eye
            view_dist = float(np.linalg.norm(view_vec)) + 1e-8
            view_dir = view_vec / view_dist
            yaw = np.deg2rad(np.random.uniform(-rot_jitter_deg, rot_jitter_deg))
            pitch = np.deg2rad(np.random.uniform(-rot_jitter_deg, rot_jitter_deg))
            jitter_rot = R.from_euler("zy", [yaw, pitch])
            jittered_dir = jitter_rot.apply(view_dir)
            target = eye + jittered_dir.astype(np.float32) * view_dist

        return eye, target

    def _maybe_randomize_ambient(self) -> None:
        """Sample ambient_light in [low, high]^3 for each sub_scene when enabled."""
        if not getattr(self, "eval_randomize_light", False):
            return
        low = float(getattr(self, "eval_ambient_low", 0.5))
        high = float(getattr(self, "eval_ambient_high", 0.5))
        if high < low:
            low, high = high, low
        low = max(0.0, min(1.0, low))
        high = max(0.0, min(1.0, high))
        try:
            sub_scenes = getattr(self.scene, "sub_scenes", [])
            for scene in sub_scenes:
                scene.ambient_light = np.random.uniform(low, high, size=(3,)).tolist()
        except Exception:
            # Light randomization is best-effort and should never break env creation.
            pass

    def _load_lighting(self, options: dict) -> None:
        """Call the base scene's lighting setup, then optionally jitter ambient."""
        super()._load_lighting(options)  # type: ignore[misc]
        self._maybe_randomize_ambient()

    def _load_agent(self, options: dict, base_pose: Optional[sapien.Pose] = None) -> None:
        """Load the robot agent at the MetaFine Franka base pose by default.

        Subclasses that need a different pose can either pass ``base_pose``
        explicitly or override ``FRANKA_BASE_POSE``. Passing ``base_pose=None``
        from a subclass also yields the default — this is what the previous
        per-env duplicated overrides did manually.
        """
        if base_pose is None:
            base_pose = sapien.Pose(p=list(self.FRANKA_BASE_POSE))
        super()._load_agent(options, base_pose)  # type: ignore[misc]
