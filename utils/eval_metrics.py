"""Policy-evaluation metrics + result containers.

Two pieces:

1. :class:`EpisodeResult` / :class:`EvalSummary` — JSON-serialisable
   dataclasses for the new ``runs/<exp>/results.json`` format. They are
   intentionally plain dicts under the hood so a downstream
   ``compare_runs.py`` (or a spreadsheet pivot) can ingest them without
   any MetaFine imports.

2. :func:`compute_smoothness` — derives action-trajectory smoothness
   metrics from the per-step actions a rollout recorded. Three numbers
   travel together because no single one captures all "wiggle":

   * ``jerk_rms``     — RMS of the third-order finite difference of
                       per-joint action targets. Small = smooth (no
                       sudden direction flips). Most discriminative on
                       chunk-of-N action heads.
   * ``vel_var``      — variance of the first-order finite difference.
                       Reports how varied the policy's per-step deltas
                       are; bursty policies score high.
   * ``path_length``  — sum of |Δaction| across the episode, normalised
                       per step. Compares to the analogous quantity on
                       the demonstrations for a relative measure.

All metrics are per-episode; aggregate at :class:`EvalSummary` build time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Smoothness                                                                  #
# --------------------------------------------------------------------------- #

def compute_smoothness(actions: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Compute trajectory smoothness metrics from a per-step action sequence.

    Args:
        actions: shape ``(T, action_dim)`` array-like, one row per step.

    Returns:
        Dict with keys ``jerk_rms`` (3rd-order diff RMS), ``vel_var``
        (1st-order diff variance), ``path_length`` (sum of L2 deltas per step
        normalised by T). Empty / single-step inputs return zeros — the
        rollout was too short to be meaningfully (un)smooth.
    """
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return {"jerk_rms": 0.0, "vel_var": 0.0, "path_length": 0.0}

    # 1st derivative — per-step delta.
    d1 = np.diff(arr, n=1, axis=0)
    vel_var = float(np.var(d1))

    # Per-step L2 delta, summed, divided by step count → unit-rate path length.
    step_norms = np.linalg.norm(d1, axis=1)
    path_length = float(step_norms.sum() / max(1, len(step_norms)))

    if arr.shape[0] < 4:
        return {"jerk_rms": 0.0, "vel_var": vel_var, "path_length": path_length}

    # 3rd derivative — jerk in action space.
    d3 = np.diff(arr, n=3, axis=0)
    jerk_rms = float(np.sqrt(np.mean(d3 ** 2)))

    return {
        "jerk_rms": jerk_rms,
        "vel_var": vel_var,
        "path_length": path_length,
    }


# --------------------------------------------------------------------------- #
# Result containers                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeResult:
    """One rollout's worth of evaluation data."""
    seed: int
    success: bool
    episode_length: int
    stage_flags: Dict[str, bool] = field(default_factory=dict)
    smoothness: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def reached_stage(self) -> int:
        """Index (1-based) of the highest reached stage, or 0 if none."""
        if not self.stage_flags:
            return int(self.success)
        ordered = list(self.stage_flags.items())
        last = 0
        for i, (_name, flag) in enumerate(ordered, start=1):
            if flag:
                last = i
        return last


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.std(ddof=0))


@dataclass
class AUSCResult:
    """One DR-axis sweep: per-level success rates + the trapezoidal AUSC."""
    axis: str
    levels: List[float]
    success_rates: List[float]
    ausc: float
    n_episodes_per_level: int


@dataclass
class EvalSummary:
    """JSON-serialisable summary of a full policy-eval run."""
    policy: str
    checkpoint: str
    task_graph: Optional[str] = None
    env_id: Optional[str] = None
    object_name: Optional[str] = None
    part_name: Optional[str] = None
    n_episodes: int = 0
    success_rate: float = 0.0
    stage_rates: Dict[str, float] = field(default_factory=dict)
    smoothness_mean: Dict[str, float] = field(default_factory=dict)
    smoothness_std: Dict[str, float] = field(default_factory=dict)
    ausc: Dict[str, AUSCResult] = field(default_factory=dict)
    episodes: List[EpisodeResult] = field(default_factory=list)

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[EpisodeResult],
        *,
        policy: str,
        checkpoint: str = "",
        task_graph: Optional[str] = None,
        env_id: Optional[str] = None,
        object_name: Optional[str] = None,
        part_name: Optional[str] = None,
        stage_names: Optional[Sequence[str]] = None,
    ) -> "EvalSummary":
        episodes = list(episodes)
        n = len(episodes)
        success_rate = _mean([1.0 if e.success else 0.0 for e in episodes])

        # Per-stage rate: fraction of episodes whose flag fired.
        stage_rates: Dict[str, float] = {}
        if stage_names is None:
            # Discover from the first episode that has any stage flags.
            for ep in episodes:
                if ep.stage_flags:
                    stage_names = list(ep.stage_flags.keys())
                    break
        if stage_names:
            for name in stage_names:
                stage_rates[name] = _mean([1.0 if ep.stage_flags.get(name, False) else 0.0
                                            for ep in episodes])

        # Smoothness aggregation: mean & std of each metric across episodes
        # that actually recorded one (skip empty dicts).
        metric_keys = ["jerk_rms", "vel_var", "path_length"]
        smoothness_mean = {k: _mean([ep.smoothness[k] for ep in episodes if ep.smoothness])
                            for k in metric_keys}
        smoothness_std = {k: _std([ep.smoothness[k] for ep in episodes if ep.smoothness])
                           for k in metric_keys}

        return cls(
            policy=policy,
            checkpoint=checkpoint,
            task_graph=task_graph,
            env_id=env_id,
            object_name=object_name,
            part_name=part_name,
            n_episodes=n,
            success_rate=success_rate,
            stage_rates=stage_rates,
            smoothness_mean=smoothness_mean,
            smoothness_std=smoothness_std,
            episodes=episodes,
        )

    def to_json(self) -> Dict[str, Any]:
        """Plain-dict view suitable for ``json.dump``."""
        return {
            "policy": self.policy,
            "checkpoint": self.checkpoint,
            "task_graph": self.task_graph,
            "env_id": self.env_id,
            "object_name": self.object_name,
            "part_name": self.part_name,
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "stage_rates": self.stage_rates,
            "smoothness_mean": self.smoothness_mean,
            "smoothness_std": self.smoothness_std,
            "ausc": {axis: asdict(res) for axis, res in self.ausc.items()},
            "episodes": [asdict(ep) for ep in self.episodes],
        }

    def write(self, path: Path | str) -> None:
        """Write the summary as ``<path>`` (JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
