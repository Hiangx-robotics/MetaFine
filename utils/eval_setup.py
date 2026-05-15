"""Env factory for policy evaluation.

A single helper, :func:`make_eval_env`, handles both modes:

1. **Single-skill mode** (default): the caller supplies ``env_id``,
   ``object_name``, ``part_name`` (etc.) directly via the argparse namespace.
   The returned env is whatever ``gym.make(env_id, ...)`` would produce —
   identical to what each policy's evaluate.py builds today, so legacy
   command lines keep working byte-for-byte.

2. **Task-graph mode** (``--task-graph PATH``): the YAML supplies the
   env id (always ``multi_skill``), object, part, skill chain, optional
   ``stages``, and a ``success`` predicate. The helper compiles the
   predicate into ``env.goal_predicate`` and (when present) registers
   ``stages`` so :class:`core.env.MultiSkillEnv` tracks per-stage
   completion across the rollout.

The function returns ``(env, task_graph_or_None)`` so the caller can branch
on multistep behaviour (e.g. collect per-stage success rates) without
re-loading the YAML itself.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import gymnasium as gym

# Importing these for their @register_env side effects so gym.make can
# resolve FGManip env ids regardless of whether the caller imported them.
import mani_skill.envs  # noqa: F401
import core.env  # noqa: F401


# Articulated envs that read ``object_name`` / ``part_name`` from their
# constructor. Bespoke envs (peg_in_hole, plug_charger, stack_pyramid,
# draw_triangle, assembling_kits, put_blocks_*) reject those kwargs, so
# the single-skill path must skip them.
_ARTICULATED_ENVS = frozenset({
    "grasp_part", "align_to_part", "stand_up",
    "toggle_switch", "toggle_switch_table", "lid_opening",
    "slide_along", "multi_skill", "rotate", "door_env",
    "take_out_and_grasp_part_into_box",
})


def _get(args: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an argparse namespace or a dict, defaulting cleanly."""
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def make_eval_env(
    args: Any,
    *,
    extra_env_kwargs: Optional[dict] = None,
) -> Tuple[gym.Env, Optional[Any]]:
    """Build the env a policy should evaluate against.

    Args:
        args: argparse Namespace or dict with at least these optional keys:
            ``task_graph`` (path), ``env_id``, ``object_name``, ``part_name``,
            ``obs_mode``, ``control_mode``, ``sim_backend``, ``render_mode``,
            ``shader``, ``sensor_configs``, ``human_render_camera_configs``,
            ``viewer_camera_configs``. Each policy's CLI typically defines
            most of these; missing ones are skipped silently.
        extra_env_kwargs: Additional kwargs to forward verbatim to
            ``gym.make`` (e.g. eval-DR settings, max_episode_steps).

    Returns:
        ``(env, task_graph_or_None)``. ``task_graph_or_None`` is the parsed
        :class:`utils.task_graph.TaskGraph` when ``--task-graph`` was given,
        else ``None``. The caller uses it to decide whether to enumerate
        stages, etc.
    """
    extra_env_kwargs = dict(extra_env_kwargs or {})

    # ----- common gym.make kwargs (only set keys the caller actually passed) -----
    common: dict = {}
    for k in (
        "obs_mode", "control_mode", "render_mode", "sim_backend",
        "sensor_configs", "human_render_camera_configs",
        "viewer_camera_configs",
    ):
        v = _get(args, k, None)
        if v is not None:
            common[k] = v
    common.update(extra_env_kwargs)

    task_graph_path = _get(args, "task_graph", None)

    if task_graph_path is not None:
        # ---------- multistep mode ----------
        # Lazy imports so utils/eval_setup is cheap to import in single-skill mode.
        from utils.task_graph import load_task_graph
        from core.predicates import compile_predicate

        task_graph = load_task_graph(task_graph_path)
        kwargs = dict(common)
        kwargs.update({
            "object_name": task_graph.object,
            "part_name": task_graph.part,
            "skill_chain": task_graph.to_chain(),
        })
        env = gym.make(task_graph.env or "multi_skill", **kwargs)
        # Attach the compiled goal predicate after gym.make so RecordEpisode
        # (which the caller usually wraps around the env) does not try to
        # JSON-serialize a Python callable for its replay metadata.
        if task_graph.success is not None:
            env.unwrapped.goal_predicate = compile_predicate(task_graph.success)
        # Stages, when present, get compiled here and stored on the env;
        # MultiSkillEnv.evaluate() picks them up.
        stages_raw = getattr(task_graph, "stages", None)
        if stages_raw:
            compiled_stages = []
            for stage in stages_raw:
                spec = stage.get("success") if isinstance(stage, dict) else None
                if spec is None:
                    continue
                compiled_stages.append({
                    "name": stage.get("name") or f"stage_{len(compiled_stages)}",
                    "predicate": compile_predicate(spec),
                })
            env.unwrapped.stage_predicates = compiled_stages
        return env, task_graph

    # ---------- single-skill mode (legacy CLI, untouched) ----------
    env_id = _get(args, "env_id", None)
    if env_id is None:
        raise ValueError(
            "make_eval_env requires either --task-graph or --env-id"
        )

    kwargs = dict(common)
    object_name = _get(args, "object_name", None)
    part_name = _get(args, "part_name", None)
    if env_id in _ARTICULATED_ENVS:
        if object_name is not None:
            kwargs["object_name"] = object_name
        if part_name is not None:
            kwargs["part_name"] = part_name

    env = gym.make(env_id, **kwargs)
    return env, None
