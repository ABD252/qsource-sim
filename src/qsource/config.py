"""config.py — YAML study configuration: the boundary between user and physics.

This module owns the ONLY unit conversion in the program layer:
users write THz / mm / % (datasheet language), the core receives
Hz / m / fractions (SI), and pump.py converts to angular rad/s itself.
No physics lives here — just parsing, validation, and unit boundaries.

Schema (see configs/example.yaml):

    crystal: {length_mm: 10, kappa_s: 0.207e-9, kappa_i: -0.318e-9}
    pump:    {bandwidth_thz: 0.164}          # + optional base imperfections
    grid:    {n: 192, span_thz: 4, realizations: 40, seed: 7}
    sweep:
      parameter: center_jitter_hz  # rin_frac | n_modes | mode_spacing_hz
                                   # | bandwidth_hz | length_m
      values: {start: 0, stop: 0.5e12, points: 9}   # or an explicit list
    outputs: {figures: true, csv: true}

NOTE on YAML numbers: PyYAML (YAML 1.1) reads `0.5e12` as a *string*
because it requires a signed exponent (`0.5e+12`). We therefore coerce
every numeric field with float()/int() instead of trusting YAML's types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

# Sweep parameters the study runner knows how to apply.
# target: "laser" -> field on LaserSpec, "crystal" -> Crystal(length_m=...).
# scale/unit: how to display the value on figure axes (user-facing units).
SWEEP_PARAMETERS = {
    "center_jitter_hz":  {"target": "laser",   "axis_scale": 1e-12, "axis_unit": "THz",  "integer": False},
    "rin_frac":          {"target": "laser",   "axis_scale": 100.0, "axis_unit": "% rms", "integer": False},
    "n_modes":           {"target": "laser",   "axis_scale": 1.0,   "axis_unit": "modes", "integer": True},
    "mode_spacing_hz":   {"target": "laser",   "axis_scale": 1e-12, "axis_unit": "THz",  "integer": False},
    "bandwidth_hz":      {"target": "laser",   "axis_scale": 1e-12, "axis_unit": "THz",  "integer": False},
    "length_m":          {"target": "crystal", "axis_scale": 1e3,   "axis_unit": "mm",   "integer": False},
}


@dataclass
class CrystalConfig:
    length_m: float = 10e-3
    kappa_s: float = +0.207e-9
    kappa_i: float = -0.318e-9
    phase_matching: str = "sinc"   # "sinc" (uniform crystal) | "gaussian"


@dataclass
class PumpConfig:
    """Base laser spec in SI units (already converted from THz)."""
    bandwidth_hz: float = 0.164e12
    center_jitter_hz: float = 0.0
    rin_frac: float = 0.0
    n_modes: int = 1
    mode_spacing_hz: float = 0.0


@dataclass
class GridConfig:
    n: int = 192
    span_hz: float = 4e12
    realizations: int = 40
    seeds: List[int] = field(default_factory=lambda: [7])
    # Monte-Carlo error bars come from replicating every sweep over these
    # seeds; each (seed, point) pair gets an independent stream via
    # np.random.default_rng([seed, i]) (plain seed+i collides: seed 8
    # point i would reuse seed 7 point i+1's draws).


@dataclass
class SweepConfig:
    parameter: str = "center_jitter_hz"
    values: List[float] = field(default_factory=list)


@dataclass
class OutputConfig:
    figures: bool = True
    csv: bool = True


@dataclass
class StudyConfig:
    crystal: CrystalConfig
    pump: PumpConfig
    grid: GridConfig
    sweep: SweepConfig
    outputs: OutputConfig
    source_path: Optional[str] = None  # where the YAML came from
    raw_yaml: str = ""                 # verbatim copy, saved with results


def _num(x, name: str) -> float:
    """Coerce a YAML scalar to float (PyYAML may hand us '0.5e12' as str)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"config field '{name}' is not a number: {x!r}")


def _expand_values(spec, parameter: str) -> List[float]:
    """Turn either {start, stop, points} or an explicit list into values."""
    if isinstance(spec, dict):
        start = _num(spec.get("start", 0), "sweep.values.start")
        stop = _num(spec.get("stop"), "sweep.values.stop")
        points = int(_num(spec.get("points", 9), "sweep.values.points"))
        if points < 2:
            raise ValueError("sweep.values.points must be >= 2")
        vals = np.linspace(start, stop, points)
    elif isinstance(spec, (list, tuple)):
        vals = np.array([_num(v, "sweep.values[]") for v in spec])
    else:
        raise ValueError("sweep.values must be {start, stop, points} or a list")

    if SWEEP_PARAMETERS[parameter]["integer"]:
        vals = np.unique(np.round(vals).astype(int))
        if np.any(vals < 1):
            raise ValueError(f"{parameter} values must be >= 1")
    return [float(v) for v in vals]


def load_config(path: str) -> StudyConfig:
    """Read + validate a study YAML. All THz -> Hz, mm -> m happens HERE."""
    with open(path, "r", encoding="utf-8") as f:
        raw_yaml = f.read()
    data = yaml.safe_load(raw_yaml) or {}

    cry = data.get("crystal", {}) or {}
    pm = str(cry.get("phase_matching", "sinc"))
    if pm not in ("sinc", "gaussian"):
        raise ValueError(f"crystal.phase_matching must be 'sinc' or "
                         f"'gaussian', got {pm!r}")
    crystal = CrystalConfig(
        length_m=_num(cry.get("length_mm", 10), "crystal.length_mm") * 1e-3,
        kappa_s=_num(cry.get("kappa_s", 0.207e-9), "crystal.kappa_s"),
        kappa_i=_num(cry.get("kappa_i", -0.318e-9), "crystal.kappa_i"),
        phase_matching=pm,
    )

    pmp = data.get("pump", {}) or {}
    pump = PumpConfig(
        bandwidth_hz=_num(pmp.get("bandwidth_thz", 0.164), "pump.bandwidth_thz") * 1e12,
        center_jitter_hz=_num(pmp.get("center_jitter_hz", 0.0), "pump.center_jitter_hz"),
        rin_frac=_num(pmp.get("rin_frac", 0.0), "pump.rin_frac"),
        n_modes=int(_num(pmp.get("n_modes", 1), "pump.n_modes")),
        mode_spacing_hz=_num(pmp.get("mode_spacing_hz", 0.0), "pump.mode_spacing_hz"),
    )

    grd = data.get("grid", {}) or {}
    if "seeds" in grd:
        raw_seeds = grd["seeds"]
        if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
            raise ValueError("grid.seeds must be a non-empty list of ints")
        seeds = [int(_num(s, "grid.seeds[]")) for s in raw_seeds]
        if len(set(seeds)) != len(seeds):
            raise ValueError("grid.seeds must be distinct")
    else:  # legacy single-seed configs stay valid
        seeds = [int(_num(grd.get("seed", 7), "grid.seed"))]
    grid = GridConfig(
        n=int(_num(grd.get("n", 192), "grid.n")),
        span_hz=_num(grd.get("span_thz", 4), "grid.span_thz") * 1e12,
        realizations=int(_num(grd.get("realizations", 40), "grid.realizations")),
        seeds=seeds,
    )

    swp = data.get("sweep")
    if not swp:
        raise ValueError("config must define exactly one 'sweep' section")
    parameter = swp.get("parameter")
    if parameter not in SWEEP_PARAMETERS:
        raise ValueError(
            f"unknown sweep.parameter {parameter!r}; choose one of: "
            + ", ".join(SWEEP_PARAMETERS)
        )
    sweep = SweepConfig(parameter=parameter,
                        values=_expand_values(swp.get("values"), parameter))

    out = data.get("outputs", {}) or {}
    outputs = OutputConfig(
        figures=bool(out.get("figures", True)),
        csv=bool(out.get("csv", True)),
    )

    return StudyConfig(crystal=crystal, pump=pump, grid=grid, sweep=sweep,
                       outputs=outputs, source_path=path, raw_yaml=raw_yaml)
