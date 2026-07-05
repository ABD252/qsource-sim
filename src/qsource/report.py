"""report.py — Turn SweepResults into a results/<timestamp>/ folder.

Reproducibility rule: every figure must be regenerable from a config file
alone, so the verbatim YAML of each config is copied into the results
folder next to the figure it produced.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless: the CLI must not require a display
import matplotlib.pyplot as plt

from .config import StudyConfig
from .study import SweepResult, to_csv, to_markdown, to_seeds_csv

_COLORS = ["#d4a24e", "#4ecdc4", "#e06377", "#8fa8d8"]


def make_figure(results: List[SweepResult]):
    """Purity vs swept parameter; adds a second panel for the double-pair
    boost whenever it actually varies (i.e. RIN-type sweeps, where purity
    stays flat and the damage shows up in the pair statistics instead)."""
    boost_varies = any(
        max(r.double_pair_boosts) - min(r.double_pair_boosts) > 1e-6
        for r in results
    )
    ncols = 2 if boost_varies else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.8),
                             constrained_layout=True)
    ax_p = axes[0] if boost_varies else axes

    param = results[0].parameter
    unit = results[0].axis_unit
    for j, r in enumerate(results):
        color = _COLORS[j % len(_COLORS)]
        label = r.label or "sweep"
        ax_p.plot(r.axis_values, r.purities, "o-", color=color, label=label)
        ax_p.axhline(r.ideal_purity, ls=":", color=color, alpha=0.7,
                     label=f"ideal ({label}): {r.ideal_purity:.3f}")
    ax_p.set_xlabel(f"{param} ({unit})")
    ax_p.set_ylabel("Heralded purity P")
    ax_p.set_title(f"Purity vs {param}")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=8)
    # A flat purity curve (e.g. RIN sweeps) makes matplotlib zoom into
    # numerical noise with offset notation — widen to a readable window.
    ax_p.ticklabel_format(useOffset=False, axis="y")
    lo = min(min(r.purities) for r in results)
    hi = max(max(r.purities + [r.ideal_purity]) for r in results)
    if hi - lo < 0.01:
        ax_p.set_ylim(lo - 0.05, hi + 0.05)

    if boost_varies:
        ax_b = axes[1]
        for j, r in enumerate(results):
            ax_b.plot(r.axis_values, r.double_pair_boosts, "s-",
                      color=_COLORS[j % len(_COLORS)],
                      label=r.label or "sweep")
        ax_b.axhline(1.0, ls=":", c="green", label="stable power")
        ax_b.set_xlabel(f"{param} ({unit})")
        ax_b.set_ylabel(r"Double-pair boost  $\langle w^2\rangle/\langle w\rangle^2$")
        ax_b.set_title("Pair statistics")
        ax_b.grid(alpha=0.3)
        ax_b.legend(fontsize=8)
    return fig


def write_results(configs: List[StudyConfig], results: List[SweepResult],
                  root: str = "results",
                  timestamp: Optional[str] = None) -> str:
    """Write figures + CSV + markdown + config copies. Returns the folder."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(root, stamp)
    os.makedirs(folder, exist_ok=True)

    # 1) verbatim config copies — the reproducibility contract
    for i, cfg in enumerate(configs):
        base = os.path.basename(cfg.source_path or f"config_{i}.yaml")
        name = base if len(configs) == 1 else f"{i + 1}_{base}"
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(cfg.raw_yaml)

    # 2) results table
    outputs = configs[0].outputs
    if outputs.csv:
        with open(os.path.join(folder, "results.csv"), "w",
                  encoding="utf-8") as f:
            f.write(to_csv(results))
        # per-seed curves for confidence intervals (multi-seed runs only)
        for r in results:
            if len(r.seeds) > 1:
                name = ("results_seeds.csv" if len(results) == 1
                        else f"results_seeds_{r.label or 'sweep'}.csv")
                with open(os.path.join(folder, name), "w",
                          encoding="utf-8") as f:
                    f.write(to_seeds_csv(r))
    with open(os.path.join(folder, "results.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(results))

    # 3) figure
    if outputs.figures:
        fig = make_figure(results)
        fig.savefig(os.path.join(folder, "sweep.png"), dpi=150)
        plt.close(fig)

    return folder
