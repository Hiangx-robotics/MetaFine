"""Domain-randomization sweep runner for policy evaluation.

Wraps a user-supplied ``run_one_level`` callback that performs ``N`` rollouts
with a given DR kwargs dict, and aggregates the per-level success rates
into an :class:`AUSCResult`. AUSC = trapezoidal area under the success-vs-
perturbation curve, normalised to ``[0, 1]`` so 1.0 == perfectly robust
(success stays high across the whole sweep) and 0.0 == policy fails the
moment any perturbation appears.

Three standard sweeps are predefined to match the EvalDREnvMixin knobs
the env already exposes:

  - ``LIGHT_LEVELS``           — vary ``eval_ambient_low`` / ``high``
  - ``CAMERA_POS_LEVELS``      — vary ``eval_camera_pos_jitter`` (metres)
  - ``CAMERA_ROT_LEVELS_DEG``  — vary ``eval_camera_rot_jitter_deg``

Each function takes the same callback shape so users can plug in any
policy without depending on this module's choice of sweep grid.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from utils.eval_metrics import AUSCResult


# Default sweep grids. The 0.0 entry must be present — it pins the curve's
# left edge to the zero-perturbation success rate (which often matches the
# headline number) so AUSC at small perturbations is interpretable.
LIGHT_LEVELS: List[float] = [0.5, 0.4, 0.3, 0.2, 0.1]
CAMERA_POS_LEVELS: List[float] = [0.0, 0.01, 0.03, 0.06, 0.12]
CAMERA_ROT_LEVELS_DEG: List[float] = [0.0, 2.0, 6.0, 12.0, 20.0]


# Type of the per-level rollout callback. Signature:
#   run_one_level(dr_kwargs: Dict[str, float], n_episodes: int) -> success_rate
RunOneLevelFn = Callable[[Dict[str, float], int], float]


def _trapezoidal_ausc(levels: Sequence[float], rates: Sequence[float]) -> float:
    """Trapezoidal area under (level, rate) normalised to [0, 1].

    Levels may be ascending or descending — orientation only changes the
    sign of the trapezoid spacings, which we take in absolute value. The
    denominator is ``|levels[-1] - levels[0]|`` so AUSC reports the
    *fraction* of perfect-success area regardless of sweep direction.
    """
    levels = list(levels)
    rates = list(rates)
    if len(levels) != len(rates) or len(levels) < 2:
        return 0.0
    width = abs(float(levels[-1] - levels[0]))
    if width <= 0:
        return 0.0
    area = 0.0
    for i in range(len(levels) - 1):
        area += 0.5 * (rates[i] + rates[i + 1]) * abs(levels[i + 1] - levels[i])
    return float(max(0.0, min(1.0, area / width)))


def _kwargs_for_axis(axis: str, level: float) -> Dict[str, float]:
    """Translate a sweep level into the env kwargs that activate it."""
    if axis == "light":
        # Center the [low, high] window on the mean of the baseline and
        # shrink/widen by the level. We sweep ambient_low directly:
        # smaller low means a darker world. Pair with ambient_high so the
        # range stays valid.
        return {
            "eval_randomize_light": True,
            "eval_ambient_low": float(level),
            "eval_ambient_high": min(1.0, float(level) + 0.2),
        }
    if axis == "camera_pos":
        return {
            "eval_randomize_camera": True,
            "eval_camera_pos_jitter": float(level),
            "eval_camera_rot_jitter_deg": 0.0,
        }
    if axis == "camera_rot":
        return {
            "eval_randomize_camera": True,
            "eval_camera_pos_jitter": 0.0,
            "eval_camera_rot_jitter_deg": float(level),
        }
    raise ValueError(
        f"unknown DR axis {axis!r}; valid: light / camera_pos / camera_rot"
    )


def dr_sweep(
    axis: str,
    levels: Sequence[float],
    run_one_level: RunOneLevelFn,
    n_episodes: int,
    *,
    verbose: bool = False,
) -> AUSCResult:
    """Run a one-dimensional DR sweep and return an :class:`AUSCResult`.

    Args:
        axis: one of ``"light"``, ``"camera_pos"``, ``"camera_rot"``.
        levels: perturbation magnitudes to sweep, in ascending order.
        run_one_level: callback that takes ``(dr_kwargs, n_episodes)`` and
            returns a scalar success rate in ``[0, 1]``. Implementations
            typically rebuild the env with the kwargs applied via
            :func:`utils.eval_setup.make_eval_env`'s ``extra_env_kwargs``.
        n_episodes: episodes per level.

    Returns:
        Filled :class:`AUSCResult` with per-level rates and AUSC.
    """
    levels = list(levels)
    rates: List[float] = []
    for lv in levels:
        kw = _kwargs_for_axis(axis, lv)
        try:
            rate = float(run_one_level(kw, n_episodes))
        except Exception as exc:
            if verbose:
                print(f"[dr_sweep] axis={axis} level={lv} raised: {exc}")
            rate = 0.0
        rate = max(0.0, min(1.0, rate))
        rates.append(rate)
        if verbose:
            print(f"[dr_sweep] axis={axis} level={lv}: {rate:.3f}")

    return AUSCResult(
        axis=axis,
        levels=levels,
        success_rates=rates,
        ausc=_trapezoidal_ausc(levels, rates),
        n_episodes_per_level=n_episodes,
    )


def standard_dr_sweeps(
    run_one_level: RunOneLevelFn,
    n_episodes_per_level: int,
    *,
    light: Sequence[float] = LIGHT_LEVELS,
    camera_pos: Sequence[float] = CAMERA_POS_LEVELS,
    camera_rot: Sequence[float] = CAMERA_ROT_LEVELS_DEG,
    verbose: bool = False,
) -> Dict[str, AUSCResult]:
    """Run all three standard DR sweeps and return a dict by axis name."""
    return {
        "light":      dr_sweep("light",      light,      run_one_level, n_episodes_per_level, verbose=verbose),
        "camera_pos": dr_sweep("camera_pos", camera_pos, run_one_level, n_episodes_per_level, verbose=verbose),
        "camera_rot": dr_sweep("camera_rot", camera_rot, run_one_level, n_episodes_per_level, verbose=verbose),
    }
