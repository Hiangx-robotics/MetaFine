"""Core modules for the MetaFine fine-grained manipulation platform.

Exposes the three building blocks used everywhere else:

* :mod:`core.env` — Gym environments (GraspPartEnv and its subclasses).
* :mod:`core.scene` — SceneBuilders that load articulated assets onto the
  table.
* :mod:`core.motion` — Motion-planning utilities shared by the skill solvers.

The skill registry (``core.skill_registry``) and the composable-task layer
(``core.predicates``, ``core.env_mixins``) are deliberately not re-exported
here; import them directly when needed.
"""

from .env import GraspPartEnv
from .scene import GraspPartSceneBuilder
from .motion import MotionPlanner

__all__ = [
    'GraspPartEnv',
    'GraspPartSceneBuilder',
    'MotionPlanner',
]
