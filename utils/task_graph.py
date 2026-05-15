"""Human-written task graphs for the ``multi_skill`` env.

A task graph file (YAML or JSON) names an env, an articulated object, an
optional success predicate (predicate-DSL form), and a list of skill steps.
``record.py`` loads it through :func:`load_task_graph`, hands the chain to
:class:`core.env.MultiSkillEnv`, and executes the steps in order via
:func:`run_task_graph`.

The expected schema::

    env: multi_skill           # gym id of the env to create
    object: 3398               # asset folder under assets/
    part: cap                  # part name (passed to the env at construction)
    success:                   # predicate-DSL dict; see core/predicates.py
      and:
        - {grasped: cap}
        - {lifted: [0.05]}
    steps:
      - skill: grasp_part      # name in SKILL_REGISTRY
        part_name: cap
      - skill: rotate
        part_name: cap

When the AI-driven planner replaces hand-written YAML, it emits the same
schema; the loader and runner stay unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TaskStep:
    """One step of a task graph."""
    skill: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskStep":
        if "skill" not in raw:
            raise ValueError(f"task step missing 'skill' key: {raw!r}")
        kwargs = {k: v for k, v in raw.items() if k != "skill"}
        return cls(skill=raw["skill"], kwargs=kwargs)


@dataclass
class TaskGraph:
    """Full parsed task graph.

    ``stages`` (optional) is a list of intermediate checkpoints used by
    policy evaluation to surface per-stage success rates. Each stage is a
    dict ``{name, success}`` where ``success`` is a predicate-DSL spec
    (same grammar as the top-level ``success`` field). A run that reaches
    stage 3 is also counted as reaching stages 1 and 2. The top-level
    ``success`` is still the *final* criterion that drives binary success.
    """
    env: str = "multi_skill"
    object: Optional[str] = None
    part: Optional[str] = None
    steps: List[TaskStep] = field(default_factory=list)
    success: Optional[Any] = None  # raw predicate-DSL spec, or None
    stages: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskGraph":
        if "steps" not in raw or not isinstance(raw["steps"], list):
            raise ValueError("task graph must have a 'steps' list")
        stages_raw = raw.get("stages", []) or []
        if not isinstance(stages_raw, list):
            raise ValueError("task graph 'stages' must be a list when present")
        return cls(
            env=str(raw.get("env", "multi_skill")),
            object=raw.get("object"),
            part=raw.get("part"),
            steps=[TaskStep.from_dict(s) for s in raw["steps"]],
            success=raw.get("success"),
            stages=list(stages_raw),
        )

    def to_chain(self) -> List[Dict[str, Any]]:
        """Render as the list-of-dicts shape MultiSkillEnv.__init__ accepts."""
        chain: List[Dict[str, Any]] = []
        for step in self.steps:
            entry: Dict[str, Any] = {"skill": step.skill, **step.kwargs}
            entry.setdefault("object", self.object)
            chain.append(entry)
        return chain


def load_task_graph(path: str | Path) -> TaskGraph:
    """Read a JSON or YAML task graph from disk."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read YAML task graphs; install with `pip install pyyaml`"
            ) from exc
        raw = yaml.safe_load(text)
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"task graph must be .yaml/.yml/.json, got {suffix!r}")
    if not isinstance(raw, dict):
        raise ValueError(f"task graph root must be a mapping, got {type(raw).__name__}")
    return TaskGraph.from_dict(raw)


def run_task_graph(planner, steps: List[Dict[str, Any]], *, verbose: bool = False) -> bool:
    """Execute a list of skill steps sequentially through ``planner``.

    Returns ``True`` only if every step's solver completes without returning
    a ``-1`` sentinel (the planner-failure marker used by the existing skill
    functions) and without raising. Stops at the first failure.
    """
    import core.skill  # noqa: F401  populate the registry
    from core.skill_registry import get_skill

    for i, step in enumerate(steps):
        skill_name = step["skill"]
        try:
            spec = get_skill(skill_name)
        except KeyError as exc:
            if verbose:
                print(f"[task_graph] step {i}: {exc}")
            return False

        # Strip schema-level keys before forwarding kwargs to the solver.
        kwargs = {k: v for k, v in step.items() if k not in ("skill", "object")}
        if verbose:
            print(f"[task_graph] step {i}: {skill_name}({kwargs})")

        try:
            result = spec.callable_(planner, **kwargs)
        except Exception as exc:
            if verbose:
                print(f"[task_graph] step {i}: {skill_name} raised {type(exc).__name__}: {exc}")
            return False

        if result == -1:
            if verbose:
                print(f"[task_graph] step {i}: {skill_name} returned -1 (planner failure)")
            return False

    return True
