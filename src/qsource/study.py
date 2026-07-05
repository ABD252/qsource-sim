"""study.py — Run a configured sweep through the (untouched) physics core.

This is glue, not physics: for each sweep value we build a LaserSpec /
Crystal from the config, hand them to heralded_purity_mc(), and collect
purity, Schmidt number and the double-pair boost <w^2>/<w>^2 (the metric
RIN actually damages, since RIN leaves spectral purity alone).

Reproducibility contract: each (seed, point) pair gets its own rng via
np.random.default_rng([seed, i]) — SeedSequence hashes the pair into an
independent stream, so results are independent of execution order,
identical across runs of the same config, and seed replicas are truly
independent. (The naive seed+i scheme collides: seed 8 at point i would
replay seed 7 at point i+1.)

Monte-Carlo uncertainty: every sweep runs once per seed in
cfg.grid.seeds; the headline curve is the per-point seed MEAN, the
per-point seed std (ddof=1) rides along for error bars, and the full
per-seed curves are kept for budget confidence intervals.
"""

from __future__ import annotations

import csv
import dataclasses
import io
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import StudyConfig, SWEEP_PARAMETERS
from .crystal import Crystal, GaussianCrystal
from .jsa import compute_jsa
from .metrics import schmidt_analysis
from .pump import GaussianPump
from .realistic import LaserSpec, heralded_purity_mc


@dataclass
class SweepResult:
    """Everything needed to plot/tabulate one sweep, in SI units.

    purities/double_pair_boosts are per-point SEED MEANS; the per-seed
    curves (purities_by_seed[seed_index][point_index]) feed the budget
    confidence intervals, and purity_stds are the per-point seed stds.
    """
    parameter: str
    values: List[float]
    purities: List[float]
    schmidt_numbers: List[float]
    double_pair_boosts: List[float]
    ideal_purity: float           # SVD purity of the imperfection-free base
    label: str = ""               # where this sweep came from (config name)
    seeds: List[int] = field(default_factory=lambda: [7])
    purity_stds: List[float] = field(default_factory=list)
    purities_by_seed: List[List[float]] = field(default_factory=list)
    boosts_by_seed: List[List[float]] = field(default_factory=list)

    @property
    def axis_values(self):
        """Sweep values scaled to the user-facing unit (THz, mm, % ...)."""
        s = SWEEP_PARAMETERS[self.parameter]["axis_scale"]
        return [v * s for v in self.values]

    @property
    def axis_unit(self) -> str:
        return SWEEP_PARAMETERS[self.parameter]["axis_unit"]


def _spec_and_crystal(cfg: StudyConfig, value: float):
    """Apply one sweep value on top of the base config -> (LaserSpec, Crystal)."""
    laser_kwargs = dict(
        bandwidth_hz=cfg.pump.bandwidth_hz,
        center_jitter_hz=cfg.pump.center_jitter_hz,
        rin_frac=cfg.pump.rin_frac,
        n_modes=cfg.pump.n_modes,
        mode_spacing_hz=cfg.pump.mode_spacing_hz,
    )
    crystal_kwargs = dict(length_m=cfg.crystal.length_m,
                          kappa_s=cfg.crystal.kappa_s,
                          kappa_i=cfg.crystal.kappa_i)

    param = cfg.sweep.parameter
    if SWEEP_PARAMETERS[param]["target"] == "laser":
        laser_kwargs[param] = int(value) if SWEEP_PARAMETERS[param]["integer"] else value
    else:
        crystal_kwargs[param] = value

    cls = GaussianCrystal if cfg.crystal.phase_matching == "gaussian" else Crystal
    return LaserSpec(**laser_kwargs), cls(**crystal_kwargs)


def _is_spectrally_ideal(spec: LaserSpec) -> bool:
    """True when every realization is the identical pure JSA.

    Then the MC average collapses to rho = A A^H exactly (validated fact #1),
    so a single realization suffices — same code path, K times faster.
    RIN alone still counts as 'ideal' for the SPECTRUM, but we keep the
    full K realizations in that case because the double-pair boost needs
    the weight statistics.
    """
    return (spec.center_jitter_hz == 0.0 and spec.rin_frac == 0.0
            and spec.n_modes <= 1)


def run_study(cfg: StudyConfig, label: str = "") -> SweepResult:
    """Execute the configured sweep once per seed. Pure computation."""
    # Ideal reference: SVD purity of the base pump/crystal, no imperfections.
    base_cls = (GaussianCrystal if cfg.crystal.phase_matching == "gaussian"
                else Crystal)
    base_crystal = base_cls(length_m=cfg.crystal.length_m,
                            kappa_s=cfg.crystal.kappa_s,
                            kappa_i=cfg.crystal.kappa_i)
    ideal = schmidt_analysis(
        compute_jsa(GaussianPump(cfg.pump.bandwidth_hz), base_crystal,
                    span_hz=cfg.grid.span_hz, n=cfg.grid.n).amplitude
    ).purity

    seeds = list(cfg.grid.seeds)
    n_seeds = len(seeds)
    by_seed_p = [[] for _ in seeds]   # [seed_index][point_index]
    by_seed_b = [[] for _ in seeds]
    for i, value in enumerate(cfg.sweep.values):
        spec, crystal = _spec_and_crystal(cfg, value)
        deterministic = _is_spectrally_ideal(spec)
        # macOS BLAS emits spurious divide/overflow/invalid RuntimeWarnings
        # on the complex matmuls inside heralded_purity_mc; the outputs are
        # validated exact by tests/test_core.py, so silence them here only.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if deterministic:
                # every realization is the identical pure JSA (validated
                # fact #1): one K=1 evaluation serves all seeds exactly.
                res = heralded_purity_mc(
                    spec, crystal, span_hz=cfg.grid.span_hz, n=cfg.grid.n,
                    k_realizations=1,
                    rng=np.random.default_rng([seeds[0], i]))
                for s_idx in range(n_seeds):
                    by_seed_p[s_idx].append(res["purity"])
                    by_seed_b[s_idx].append(1.0)
            else:
                for s_idx, seed in enumerate(seeds):
                    res = heralded_purity_mc(
                        spec, crystal, span_hz=cfg.grid.span_hz,
                        n=cfg.grid.n, k_realizations=cfg.grid.realizations,
                        rng=np.random.default_rng([seed, i]))
                    w = res["weights"]
                    by_seed_p[s_idx].append(res["purity"])
                    by_seed_b[s_idx].append(
                        float(np.mean(w ** 2) / np.mean(w) ** 2))

    p_mat = np.array(by_seed_p)   # shape (n_seeds, n_points)
    b_mat = np.array(by_seed_b)
    purities = [float(x) for x in p_mat.mean(axis=0)]
    stds = ([float(x) for x in p_mat.std(axis=0, ddof=1)] if n_seeds > 1
            else [0.0] * p_mat.shape[1])
    boosts = [float(x) for x in b_mat.mean(axis=0)]

    return SweepResult(
        parameter=cfg.sweep.parameter, values=list(cfg.sweep.values),
        purities=purities,
        schmidt_numbers=[1.0 / p for p in purities],
        double_pair_boosts=boosts, ideal_purity=ideal, label=label,
        seeds=seeds, purity_stds=stds,
        purities_by_seed=[list(row) for row in by_seed_p],
        boosts_by_seed=[list(row) for row in by_seed_b],
    )


# ---------------------------------------------------------------------------
# Tabular serialization (CSV + markdown) — shared by run and compare.
# ---------------------------------------------------------------------------

_COLUMNS = ["parameter", "value", "value_display", "unit",
            "purity", "schmidt_number", "double_pair_boost",
            "purity_std", "n_seeds"]


def _rows(result: SweepResult):
    stds = result.purity_stds or [0.0] * len(result.values)
    for v, vd, p, k, b, sd in zip(result.values, result.axis_values,
                                  result.purities, result.schmidt_numbers,
                                  result.double_pair_boosts, stds):
        yield [result.parameter, f"{v:.6g}", f"{vd:.6g}", result.axis_unit,
               f"{p:.6f}", f"{k:.4f}", f"{b:.6f}", f"{sd:.6f}",
               str(len(result.seeds))]


def to_seeds_csv(result: SweepResult) -> str:
    """Long-format per-seed curves — the budget CIs interpolate these."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["parameter", "value", "seed", "purity",
                     "schmidt_number", "double_pair_boost"])
    for s_idx, seed in enumerate(result.seeds):
        for i, v in enumerate(result.values):
            p = result.purities_by_seed[s_idx][i]
            b = result.boosts_by_seed[s_idx][i]
            writer.writerow([result.parameter, f"{v:.6g}", seed,
                             f"{p:.6f}", f"{1.0 / p:.4f}", f"{b:.6f}"])
    return buf.getvalue()


def to_csv(results: List[SweepResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    header = _COLUMNS + (["label"] if len(results) > 1 else [])
    writer.writerow(header)
    for r in results:
        for row in _rows(r):
            writer.writerow(row + ([r.label] if len(results) > 1 else []))
    return buf.getvalue()


def to_markdown(results: List[SweepResult]) -> str:
    lines = []
    for r in results:
        title = f"## Sweep: `{r.parameter}`"
        if r.label:
            title += f" — {r.label}"
        lines += [title, "",
                  f"Ideal (imperfection-free) SVD purity of the base source: "
                  f"**{r.ideal_purity:.4f}**", "",
                  f"| {r.parameter} ({r.axis_unit}) | Purity P | Schmidt K "
                  f"| Double-pair boost |",
                  "|---:|---:|---:|---:|"]
        for vd, p, k, b in zip(r.axis_values, r.purities,
                               r.schmidt_numbers, r.double_pair_boosts):
            lines.append(f"| {vd:.6g} | {p:.4f} | {k:.3f} | {b:.4f} |")
        lines.append("")
    return "\n".join(lines)
