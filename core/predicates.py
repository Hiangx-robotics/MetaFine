"""Predicate DSL for task-graph success specifications.

When a task graph wants its success condition to be serializable (YAML),
encode it as a dict tree instead of a Python callable. ``compile_predicate``
turns the dict into the same ``Callable[[env, obs], bool]`` shape
``MultiSkillEnv`` already supports. This keeps two code paths — Python
callable (most flexible) and YAML dict (replay-friendly) — sharing the same
evaluation contract.

Atomic predicates::

    {"grasped": "handle"}                                        # gripper holding something
    {"joint_value": [">", 1.5, "joint_0"]}                       # parametric op + threshold + joint name
    {"joint_value": [">=", 1.5, "joint_0", "cabinet_01"]}        # 4th element scopes to a named object
    {"lifted": [0.05]}                                           # TCP z above 0.05 m
    {"lifted": [0.05, "object"]}                                 # target object z above 0.05 m
    {"pose_near": [0.03, "tcp", [0.1, 0.0, 0.3]]}                # TCP within 3 cm of (0.1, 0, 0.3)
    {"placed_in": ["red_block", "box_a"]}                        # actor inside named region (TODO)
    {"stacked_on": ["red_block", "blue_block"]}                  # red sits on blue (TODO)

Compound predicates::

    {"and": [..., ...]}
    {"or": [..., ...]}
    {"not": ...}

Operators for ``joint_value``: ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

PredicateFn = Callable[[Any, Any], bool]  # (env, obs) -> bool

_NUM_OPS: Dict[str, Callable[[float, float], bool]] = {
    ">":  lambda a, b: a >  b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a <  b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class PredicateError(ValueError):
    """Raised on a malformed predicate dict."""


# --------------------------------------------------------------------------- #
# Atomic predicate builders                                                   #
# --------------------------------------------------------------------------- #

def _build_grasped(arg: Any) -> PredicateFn:
    if not isinstance(arg, (str, type(None))):
        raise PredicateError(f"grasped expects part-name string or null, got {arg!r}")
    part = arg
    return lambda env, _obs: bool(env.grasped(part))


def _build_joint_value(arg: Any) -> PredicateFn:
    if not isinstance(arg, (list, tuple)) or len(arg) < 3 or len(arg) > 4:
        raise PredicateError(
            "joint_value expects [op, threshold, joint_name, optional_object_name], got "
            f"{arg!r}"
        )
    op_name = arg[0]
    threshold = float(arg[1])
    joint_name = arg[2]
    object_name = arg[3] if len(arg) == 4 else None
    if op_name not in _NUM_OPS:
        raise PredicateError(f"joint_value op must be one of {list(_NUM_OPS)}, got {op_name!r}")
    op = _NUM_OPS[op_name]

    def _eval(env, _obs):
        val = env.joint_value(joint_name) if object_name is None else env.joint_value(joint_name)
        # object_name argument reserved for future multi-object support; ignored today.
        if val is None:
            return False
        return op(float(val), threshold)

    return _eval


def _build_lifted(arg: Any) -> PredicateFn:
    if not isinstance(arg, (list, tuple)) or not (1 <= len(arg) <= 2):
        raise PredicateError(f"lifted expects [min_height] or [min_height, ref], got {arg!r}")
    min_h = float(arg[0])
    ref = arg[1] if len(arg) == 2 else "tcp"
    return lambda env, _obs: bool(env.lifted(min_h, ref=ref))


def _build_pose_near(arg: Any) -> PredicateFn:
    if not isinstance(arg, (list, tuple)) or len(arg) != 3:
        raise PredicateError(
            f"pose_near expects [tolerance, ref, target_xyz], got {arg!r}"
        )
    tol = float(arg[0])
    ref = arg[1]
    target = list(arg[2])
    if ref != "tcp":
        # Targets other than tcp (e.g. named actors) are not wired yet — fail loud.
        raise PredicateError(f"pose_near ref must be 'tcp' today, got {ref!r}")
    if len(target) != 3:
        raise PredicateError(f"pose_near target_xyz must be length-3, got {target!r}")

    def _eval(env, _obs):
        p = env.agent.tcp.pose.p
        # p is shape (B, 3) torch tensor
        import torch
        target_t = torch.tensor(target, dtype=p.dtype, device=p.device)
        dist = float(torch.linalg.norm(p[0] - target_t).item())
        return dist <= tol

    return _eval


def _build_placed_in(arg: Any) -> PredicateFn:
    # Reserved for future multi-actor support. Today: never satisfied.
    return lambda _env, _obs: False


def _build_stacked_on(arg: Any) -> PredicateFn:
    return lambda _env, _obs: False


_ATOMIC_BUILDERS: Dict[str, Callable[[Any], PredicateFn]] = {
    "grasped":    _build_grasped,
    "joint_value": _build_joint_value,
    "lifted":     _build_lifted,
    "pose_near":  _build_pose_near,
    "placed_in":  _build_placed_in,
    "stacked_on": _build_stacked_on,
}


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #

def compile_predicate(spec: Any) -> PredicateFn:
    """Turn a predicate-spec dict into a ``Callable[[env, obs], bool]``.

    ``spec`` may also be ``True`` / ``False`` (constant predicates) — useful
    for stub success conditions during development.
    """
    if spec is True:
        return lambda _env, _obs: True
    if spec is False:
        return lambda _env, _obs: False

    if not isinstance(spec, dict) or len(spec) != 1:
        raise PredicateError(
            f"predicate must be a single-key dict (e.g. {{'grasped': ...}} or "
            f"{{'and': [...]}}), got {spec!r}"
        )
    (op_name, op_arg), = spec.items()

    if op_name == "and":
        if not isinstance(op_arg, list):
            raise PredicateError(f"'and' expects a list of predicates, got {op_arg!r}")
        children = [compile_predicate(c) for c in op_arg]
        return lambda env, obs: all(c(env, obs) for c in children)

    if op_name == "or":
        if not isinstance(op_arg, list):
            raise PredicateError(f"'or' expects a list of predicates, got {op_arg!r}")
        children = [compile_predicate(c) for c in op_arg]
        return lambda env, obs: any(c(env, obs) for c in children)

    if op_name == "not":
        child = compile_predicate(op_arg)
        return lambda env, obs: not child(env, obs)

    builder = _ATOMIC_BUILDERS.get(op_name)
    if builder is None:
        raise PredicateError(
            f"unknown predicate op {op_name!r}; expected one of "
            f"{sorted(set(_ATOMIC_BUILDERS) | {'and', 'or', 'not'})}"
        )
    return builder(op_arg)


def evaluate_predicate(spec: Any, env: Any, obs: Any = None) -> bool:
    """Convenience: compile and evaluate in one step (don't use in hot loops)."""
    return compile_predicate(spec)(env, obs)
