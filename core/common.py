"""Backward-compatible exports.

``MP_SOLUTIONS`` was previously a hand-maintained dict mapping skill names to
motion-planning solver callables. It is now derived automatically from
:data:`core.skill_registry.SKILL_REGISTRY`, which is populated when
:mod:`core.skill` is imported (each skill is decorated with
``@register_skill``). New skills only need to add a decorator; this dict
updates without further edits.
"""

import core.skill  # noqa: F401  imported for the @register_skill side effects
from core.skill_registry import SKILL_REGISTRY

MP_SOLUTIONS = {name: spec.callable_ for name, spec in SKILL_REGISTRY.items()}
