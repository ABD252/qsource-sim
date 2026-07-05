"""budget.py — Imperfection-budget arithmetic (program layer, no physics).

The paper's deliverable is a table of TOLERANCES: "how much of imperfection
X can the source absorb before the heralded purity drops by 1% / 5% / 10%
of its ideal value?". This module turns sweep curves (values, purities)
into those thresholds by linear interpolation. It never touches the
physics core — it only post-processes numbers the core produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def threshold_crossing(values: List[float], purities: List[float],
                       target: float) -> Optional[float]:
    """First swept value at which purity falls below `target`.

    Assumes the sweep starts at (or near) its best purity and degrades —
    the shape of every imperfection sweep (jitter, spacing, n_modes).
    Linear interpolation between the bracketing points. Returns None when
    purity never drops below target inside the swept range (i.e. the
    tolerance is beyond the studied window), and values[0] when the sweep
    already starts below target.
    """
    v = np.asarray(values, float)
    p = np.asarray(purities, float)
    if len(v) != len(p) or len(v) < 2:
        raise ValueError("need equal-length values/purities with >= 2 points")
    if p[0] < target:
        return float(v[0])
    below = np.nonzero(p < target)[0]
    if len(below) == 0:
        return None
    i = below[0]
    # linear interpolation between (v[i-1], p[i-1]) and (v[i], p[i])
    frac = (p[i - 1] - target) / (p[i - 1] - p[i])
    return float(v[i - 1] + frac * (v[i] - v[i - 1]))


@dataclass
class ThresholdEstimate:
    """Seed-ensemble summary of one budget cell."""
    mean: float
    std: float        # ddof=1 across seeds
    sem: float        # std / sqrt(n_finite) — the honest error bar
    n_finite: int     # seeds whose curve actually crossed the target
    n_total: int


def threshold_distribution(values: Sequence[float],
                           purities_by_seed: Sequence[Sequence[float]],
                           target: float) -> List[Optional[float]]:
    """threshold_crossing applied to each seed's curve independently.

    The budget confidence interval comes from the spread of these
    per-seed crossings — interpolating the seed-mean curve first would
    hide the Monte-Carlo uncertainty the referee asked us to quote.
    """
    return [threshold_crossing(list(values), list(p), target)
            for p in purities_by_seed]


def summarize_thresholds(xs: Sequence[Optional[float]],
                         n_total: int) -> Optional[ThresholdEstimate]:
    """Mean +- SEM of the finite per-seed crossings.

    SEM (not std) is the quoted uncertainty: the budget estimates a
    deterministic quantity, so estimator error shrinks as 1/sqrt(S).
    Returns None when no seed crossed (budget beyond the sweep).
    """
    finite = [x for x in xs if x is not None]
    if not finite:
        return None
    arr = np.asarray(finite, float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return ThresholdEstimate(
        mean=float(arr.mean()), std=std,
        sem=std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0,
        n_finite=len(arr), n_total=n_total,
    )


def tolerance_band(values: List[float], purities: List[float],
                   target: float) -> Optional[Tuple[float, float]]:
    """Contiguous (lo, hi) range around the purity MAXIMUM where P >= target.

    For optimum-shaped curves (purity vs pump bandwidth): the band of the
    swept parameter that keeps the source within `target`. Edges are
    linearly interpolated; an edge clamps to the sweep boundary when the
    curve is still above target there. Returns None if even the maximum
    is below target.
    """
    v = np.asarray(values, float)
    p = np.asarray(purities, float)
    if len(v) != len(p) or len(v) < 2:
        raise ValueError("need equal-length values/purities with >= 2 points")
    k = int(np.argmax(p))
    if p[k] < target:
        return None

    lo = float(v[0])
    for i in range(k, 0, -1):          # walk left from the peak
        if p[i - 1] < target:
            frac = (p[i] - target) / (p[i] - p[i - 1])
            lo = float(v[i] - frac * (v[i] - v[i - 1]))
            break
    hi = float(v[-1])
    for i in range(k, len(v) - 1):     # walk right from the peak
        if p[i + 1] < target:
            frac = (p[i] - target) / (p[i] - p[i + 1])
            hi = float(v[i] + frac * (v[i + 1] - v[i]))
            break
    return lo, hi
