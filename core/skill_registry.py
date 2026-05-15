"""Skill registry for MetaFine.

Catalogs every motion-planning solver in :mod:`core.skill` together with the
object affordances it needs, its grouping (atomic / composite / bespoke), the
keyword argument used to pass the target part name, and its *phase* in the
contact lifecycle (interaction / continuation / bundle). It backs the
:data:`core.common.MP_SOLUTIONS` lookup and the task-graph composer that
selects skill chains for a given object's capabilities.

Phase semantics (used by :func:`validate_task_graph`):

* ``interaction`` — establishes physical contact (closes the gripper for the
  first time, or makes the first touch). Example: ``grasp_part``,
  ``press_switch``, ``toggle_switch``, ``align_to_part``.
* ``continuation`` — operates assuming contact already exists (translates,
  rotates, slides while engaged). Example: ``move_to_direction``.
* ``bundle`` — encapsulates engage + continuation internally; not directly
  composable today. Example: ``rotate_knob`` (grasps then rotates),
  ``stand_up``, ``lid_opening``, all bespoke long-horizon solvers.

A human-supplied task graph is a list of step dicts (``{skill, object, part,
**kwargs}``). :func:`validate_task_graph` enforces a small set of rules so
that, when the AI-driven planner replaces the human, the schema and checks
stay identical.

Skills register themselves at import time via :func:`register_skill`. Consumers
query through :func:`get_skill`, :func:`list_skills`,
:func:`applicable_skills`, and :func:`validate_task_graph`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

AFFORDANCES = frozenset(
    {
        "graspable",
        "rotatable",
        "slidable",
        "pressable",
        "openable",
        "liftable",
        "insertable",
        "flippable",
        "placeable",
        "stackable",
        "drawable",
    }
)

_GROUPS = ("atomic", "composite", "bespoke")
_PHASES = ("interaction", "continuation", "bundle")


@dataclass
class SkillSpec:
    name: str
    callable_: Callable[..., Any]
    affordances: List[str] = field(default_factory=list)
    group: str = "atomic"
    part_arg: Optional[str] = None
    phase: str = "interaction"
    description: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.affordances) - AFFORDANCES
        if unknown:
            raise ValueError(
                f"Skill {self.name!r} declares unknown affordances: {sorted(unknown)}. "
                f"Valid set: {sorted(AFFORDANCES)}"
            )
        if self.group not in _GROUPS:
            raise ValueError(
                f"Skill {self.name!r} group must be one of {_GROUPS}, got {self.group!r}"
            )
        if self.phase not in _PHASES:
            raise ValueError(
                f"Skill {self.name!r} phase must be one of {_PHASES}, got {self.phase!r}"
            )


SKILL_REGISTRY: Dict[str, SkillSpec] = {}


def register_skill(
    name: Optional[str] = None,
    *,
    affordances: Optional[Iterable[str]] = None,
    group: str = "atomic",
    part_arg: Optional[str] = None,
    phase: str = "interaction",
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach a :class:`SkillSpec` to a skill callable.

    Returns the original callable unchanged so existing call sites keep
    working. Re-registering an existing name raises ``ValueError``.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec_name = name or fn.__name__
        if spec_name in SKILL_REGISTRY:
            raise ValueError(f"Skill {spec_name!r} is already registered")
        SKILL_REGISTRY[spec_name] = SkillSpec(
            name=spec_name,
            callable_=fn,
            affordances=list(affordances or []),
            group=group,
            part_arg=part_arg,
            phase=phase,
            description=description,
        )
        return fn

    return deco


def get_skill(name: str) -> SkillSpec:
    try:
        return SKILL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown skill {name!r}. Registered: {sorted(SKILL_REGISTRY)}"
        ) from exc


def list_skills(group: Optional[str] = None) -> List[SkillSpec]:
    if group is None:
        return list(SKILL_REGISTRY.values())
    if group not in _GROUPS:
        raise ValueError(f"group must be one of {_GROUPS}, got {group!r}")
    return [s for s in SKILL_REGISTRY.values() if s.group == group]


def validate_task_graph(
    steps: List[Dict[str, Any]],
    capabilities: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run rule-based checks on a human-supplied task graph.

    Each step is ``{"skill": name, "object": obj, "part": part_name,
    **skill_kwargs}``. The validator returns a list of human-readable issue
    strings; an empty list means "passed". This is intentionally warn-only —
    capability metadata is auto-derived and may be conservative, so hard
    raises would block legitimate chains.

    Rules:

    1. Every referenced skill must exist in the registry.
    2. The first step must be ``interaction`` or ``bundle`` (a chain cannot
       start with a pure continuation skill — nothing is engaged yet).
    3. After a non-``bundle`` interaction step, subsequent steps should be
       ``continuation`` (or a fresh ``interaction`` if intentionally
       re-engaging — flagged but allowed).
    4. When ``capabilities`` is supplied and a step names a ``part``, the
       part's declared affordances must cover the skill's
       ``required_affordances``.
    """
    issues: List[str] = []
    if not steps:
        return ["empty task graph"]

    last_phase: Optional[str] = None
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "skill" not in step:
            issues.append(f"step {i}: malformed (must be a dict with a 'skill' key)")
            continue
        skill_name = step["skill"]
        spec = SKILL_REGISTRY.get(skill_name)
        if spec is None:
            issues.append(f"step {i}: unknown skill {skill_name!r}")
            continue

        # Rule 2: first step engages.
        if i == 0 and spec.phase == "continuation":
            issues.append(
                f"step 0: skill {skill_name!r} is a continuation phase but "
                "the chain must start with an interaction or bundle skill"
            )

        # Rule 3: after engaging, prefer continuation; reengaging is allowed but flagged.
        if (
            i > 0
            and last_phase in ("interaction", "continuation")
            and spec.phase == "interaction"
        ):
            issues.append(
                f"step {i}: re-engaging via {skill_name!r} after the previous "
                "step was already engaged (allowed but unusual)"
            )

        # Rule 4: required affordances must be present on the chosen part.
        if capabilities is not None and spec.affordances:
            # Determine which part this step targets.
            part = None
            if spec.part_arg and spec.part_arg in step:
                part = step[spec.part_arg]
            elif "part" in step:
                part = step["part"]
            if part is not None:
                part_aff = set(
                    capabilities.get("parts", {}).get(part, {}).get("affordances", [])
                )
                missing = set(spec.affordances) - part_aff
                if missing:
                    issues.append(
                        f"step {i}: skill {skill_name!r} requires affordances "
                        f"{sorted(missing)} on part {part!r}, but capabilities lists "
                        f"only {sorted(part_aff) or '[]'}"
                    )

        last_phase = spec.phase

    return issues


def applicable_skills(
    capabilities: Dict[str, Any],
    part_name: Optional[str] = None,
    *,
    include_groups: Iterable[str] = ("atomic", "composite", "bespoke"),
) -> List[str]:
    """Return skill names whose required affordances are present.

    If ``part_name`` is supplied only that part's affordances are considered;
    otherwise the union across all parts is used (i.e. the skill is applicable
    to *some* part of the object).
    """
    include_groups = set(include_groups)
    parts = (capabilities or {}).get("parts", {})
    if part_name is None:
        available: set = set()
        for meta in parts.values():
            available.update(meta.get("affordances", []))
    else:
        available = set(parts.get(part_name, {}).get("affordances", []))

    return [
        spec.name
        for spec in SKILL_REGISTRY.values()
        if spec.group in include_groups
        and set(spec.affordances).issubset(available)
    ]
