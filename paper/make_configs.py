"""make_configs.py — Generate the paper's study configs (configs/paper/).

Every figure and table in the paper is reproducible from these YAML files
alone (hard rule 5): regenerate them with `python paper/make_configs.py`,
run them with `python paper/run_studies.py`.

Study design notes
------------------
- Monte-Carlo studies run S = 16 seeds (7-22); every (seed, point) pair
  gets an independent stream via default_rng([seed, i]). Budget cells are
  quoted as mean +- SEM over the per-seed threshold crossings.
- The jitter grids are DENSE where the budgets live (5-10 GHz steps
  below 100 GHz) so interpolation error is far below MC error.
- jitter_fine (K = 240, 5 GHz steps over the 1%-crossing region) feeds
  the 1% budget cell; the K30/K60/K240 ladder checks that the finite-K
  estimator bias does not move the budgets (acceptance:
  |b(240) - b(120)| < SEM(b(120)) at every drop level).
- jitter_gauss uses Gaussian phase matching (gamma = 0.193): its MC
  curve is compared against the closed form in qsource/analytic.py —
  the paper's external validation — and doubles as the engineered-
  crystal generalization check.
- n = 256 is grid-converged to < 1e-4 (checked against n = 768); the
  n128/n192 replicas document that. Ideal-pump sweeps use n = 512 via
  the K = 1 fast path.
- The RIN study needs weight statistics, not spectral resolution:
  n = 64, K = 1000, 8 seeds.
"""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "configs", "paper")

CRYSTAL_10 = "crystal: {length_mm: 10, kappa_s: 0.207e-9, kappa_i: -0.318e-9}"
CRYSTAL_10_GAUSS = ("crystal: {length_mm: 10, kappa_s: 0.207e-9, "
                    "kappa_i: -0.318e-9, phase_matching: gaussian}")
CRYSTAL_5 = "crystal: {length_mm: 5, kappa_s: 0.207e-9, kappa_i: -0.318e-9}"
CRYSTAL_20 = "crystal: {length_mm: 20, kappa_s: 0.207e-9, kappa_i: -0.318e-9}"

SEEDS_16 = "[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]"
SEEDS_8 = "[7, 8, 9, 10, 11, 12, 13, 14]"

# Dense where the budgets live (5-10 GHz steps below 100 GHz rms).
JITTER_GHZ = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100,
              125, 150, 200, 250, 300, 350, 400, 450, 500]
FINE_GHZ = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            60, 70, 80, 90, 100, 110, 120]


def _ghz_list(ghz_values, scale=1.0) -> str:
    return "[" + ", ".join(f"{g * scale * 1e9:.6g}" for g in ghz_values) + "]"


def _values_list(vals) -> str:
    return "[" + ", ".join(f"{v:.6e}" for v in vals) + "]"


def _jitter_cfg(crystal_line: str, bw_thz: float, values: str,
                comment: str, n: int = 256, k: int = 120,
                seeds: str = SEEDS_16) -> str:
    return f"""# {comment}
{crystal_line}
pump: {{bandwidth_thz: {bw_thz}}}
grid: {{n: {n}, span_thz: 4, realizations: {k}, seeds: {seeds}}}
sweep:
  parameter: center_jitter_hz
  values: {values}
outputs: {{figures: true, csv: true}}
"""


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    configs = {}

    # --- ideal-pump design curves (deterministic K=1 fast path) ----------
    bw_vals = np.logspace(np.log10(0.05e12), np.log10(1.2e12), 21)
    configs["bandwidth"] = f"""# Purity vs pump bandwidth — the interior optimum (ideal pump).
{CRYSTAL_10}
pump: {{bandwidth_thz: 0.164}}
grid: {{n: 512, span_thz: 6, realizations: 120, seeds: [7]}}
sweep:
  parameter: bandwidth_hz
  values: {_values_list(bw_vals)}
outputs: {{figures: true, csv: true}}
"""
    L_vals = np.linspace(2e-3, 30e-3, 15)
    configs["length"] = f"""# Purity vs crystal length at fixed 0.164 THz pump (ideal pump).
{CRYSTAL_10}
pump: {{bandwidth_thz: 0.164}}
grid: {{n: 512, span_thz: 8, realizations: 120, seeds: [7]}}
sweep:
  parameter: length_m
  values: {_values_list(L_vals)}
outputs: {{figures: true, csv: true}}
"""

    # --- jitter family ----------------------------------------------------
    configs["jitter"] = _jitter_cfg(
        CRYSTAL_10, 0.164, _ghz_list(JITTER_GHZ),
        "Purity vs pump center jitter rms — 10 mm crystal, S = 16 seeds.")
    configs["jitter_L5"] = _jitter_cfg(
        CRYSTAL_5, 0.328, _ghz_list(JITTER_GHZ, scale=2.0),
        "Jitter tolerance of a 5 mm crystal (pump re-matched to 0.328 THz).")
    configs["jitter_L20"] = _jitter_cfg(
        CRYSTAL_20, 0.082, _ghz_list(JITTER_GHZ, scale=0.5),
        "Jitter tolerance of a 20 mm crystal (pump re-matched to 0.082 THz).")
    configs["jitter_gauss"] = _jitter_cfg(
        CRYSTAL_10_GAUSS, 0.164, _ghz_list(JITTER_GHZ),
        "Analytic-validation sweep: Gaussian phase matching (gamma = 0.193,"
        "\n# amplitude-FWHM-matched to sinc); closed form in "
        "qsource/analytic.py.")
    configs["jitter_fine"] = _jitter_cfg(
        CRYSTAL_10, 0.164, _ghz_list(FINE_GHZ),
        "Fine grid + K = 240 for the 1% budget cell (resolves the 0.008 "
        "purity drop).", k=240)

    # --- finite-K convergence ladder (budgets vs K, 8 seeds) --------------
    for k in (30, 60, 240):
        configs[f"jitter_K{k}"] = _jitter_cfg(
            CRYSTAL_10, 0.164, _ghz_list(JITTER_GHZ),
            f"K-convergence ladder rung: K = {k} (baseline study uses 120).",
            k=k, seeds=SEEDS_8)

    # --- grid-convergence replicas (documentation of n-independence) ------
    for n in (128, 192):
        configs[f"jitter_n{n}"] = _jitter_cfg(
            CRYSTAL_10, 0.164, _ghz_list(JITTER_GHZ),
            f"Grid-convergence replica of the jitter study at n = {n}.",
            n=n, seeds="[7]")

    # --- multimode comb ----------------------------------------------------
    configs["modes"] = f"""# Purity vs number of longitudinal modes (comb spacing 0.25 THz).
{CRYSTAL_10}
pump:
  bandwidth_thz: 0.164
  mode_spacing_hz: 0.25e+12
grid: {{n: 256, span_thz: 4, realizations: 120, seeds: {SEEDS_16}}}
sweep:
  parameter: n_modes
  values: [1, 2, 3, 4, 5, 6, 7]
outputs: {{figures: true, csv: true}}
"""
    configs["spacing"] = f"""# Purity vs mode spacing for a 3-mode pump.
{CRYSTAL_10}
pump:
  bandwidth_thz: 0.164
  n_modes: 3
grid: {{n: 256, span_thz: 4, realizations: 120, seeds: {SEEDS_16}}}
sweep:
  parameter: mode_spacing_hz
  values: {{start: 0.02e+12, stop: 0.6e+12, points: 15}}
outputs: {{figures: true, csv: true}}
"""

    # --- RIN ---------------------------------------------------------------
    configs["rin"] = f"""# RIN sweep: heralded purity is exactly invariant; the double-pair
# boost <w^2>/<w>^2 is the metric that degrades (statistical channel).
{CRYSTAL_10}
pump: {{bandwidth_thz: 0.164}}
grid: {{n: 64, span_thz: 4, realizations: 1000, seeds: {SEEDS_8}}}
sweep:
  parameter: rin_frac
  values: {{start: 0, stop: 0.30, points: 13}}
outputs: {{figures: true, csv: true}}
"""

    # remove configs superseded by the seeds schema
    for stale in ("jitter_seed8.yaml", "jitter_seed9.yaml"):
        path = os.path.join(OUT, stale)
        if os.path.exists(path):
            os.remove(path)
            print("removed stale", stale)

    for name, text in configs.items():
        path = os.path.join(OUT, f"{name}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))
    print(f"{len(configs)} configs ready.")


if __name__ == "__main__":
    main()
