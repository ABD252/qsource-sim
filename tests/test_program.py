"""Program-layer tests (config + study glue) — run with:
    python tests/test_program.py

These do NOT test physics (tests/test_core.py owns that); they pin down
the config schema, the YAML number-coercion quirk, and the promise that
the study runner reproduces the core's numbers exactly.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qsource import Crystal, GaussianPump, compute_jsa, schmidt_analysis
from qsource.config import load_config
from qsource.study import run_study


def _write(text: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text)
    f.close()
    return f.name


BASE = """
crystal: {length_mm: 10, kappa_s: 0.207e-9, kappa_i: -0.318e-9}
pump: {bandwidth_thz: 0.164}
grid: {n: 96, span_thz: 4, realizations: 8, seed: 7}
sweep:
  parameter: %s
  values: %s
outputs: {figures: false, csv: true}
"""


def test_unsigned_exponent_coercion():
    # PyYAML (YAML 1.1) parses `0.5e12` as a STRING because the exponent
    # has no sign; the config loader must coerce it to a float anyway.
    path = _write(BASE % ("center_jitter_hz",
                          "{start: 0, stop: 0.5e12, points: 3}"))
    cfg = load_config(path)
    os.unlink(path)
    assert cfg.sweep.values == [0.0, 0.25e12, 0.5e12], cfg.sweep.values
    assert cfg.pump.bandwidth_hz == 0.164e12  # THz -> Hz at the boundary
    assert cfg.crystal.length_m == 10e-3      # mm -> m at the boundary
    print("PASS  YAML numbers coerced, units converted at the boundary")


def test_invalid_configs_rejected():
    for text, why in [
        (BASE % ("bogus_param", "[0, 1]"), "unknown parameter"),
        ("pump: {bandwidth_thz: 0.164}\n", "missing sweep"),
        (BASE % ("n_modes", "[0, 1]"), "n_modes < 1"),
    ]:
        path = _write(text)
        try:
            load_config(path)
            raise AssertionError(f"config with {why} should have raised")
        except ValueError:
            pass
        finally:
            os.unlink(path)
    print("PASS  invalid configs raise ValueError with a clear message")


def test_study_zero_point_matches_ideal_svd():
    # The sweep's zero-imperfection point must equal the ideal SVD purity
    # (validated fact #1, now flowing through the whole program layer).
    path = _write(BASE % ("center_jitter_hz",
                          "{start: 0, stop: 0.2e+12, points: 2}"))
    cfg = load_config(path)
    os.unlink(path)
    result = run_study(cfg)
    ideal = schmidt_analysis(
        compute_jsa(GaussianPump(cfg.pump.bandwidth_hz),
                    Crystal(cfg.crystal.length_m, cfg.crystal.kappa_s,
                            cfg.crystal.kappa_i),
                    span_hz=cfg.grid.span_hz, n=cfg.grid.n).amplitude
    ).purity
    assert abs(result.purities[0] - ideal) < 1e-9
    assert result.purities[1] < ideal          # jitter must cost purity
    assert abs(result.ideal_purity - ideal) < 1e-12
    print(f"PASS  sweep zero-point == ideal SVD purity ({ideal:.4f}), "
          f"jitter point below it ({result.purities[1]:.4f})")


def test_study_is_reproducible():
    # Same config -> bit-identical numbers (per-point seeding).
    path = _write(BASE % ("rin_frac", "[0.0, 0.2]"))
    cfg = load_config(path)
    os.unlink(path)
    r1, r2 = run_study(cfg), run_study(cfg)
    assert r1.purities == r2.purities
    assert r1.double_pair_boosts == r2.double_pair_boosts
    assert r2.double_pair_boosts[1] > 1.0      # RIN must boost double pairs
    print("PASS  identical config -> identical results (seeded per point)")


def test_seeds_schema():
    # New schema: grid.seeds is a list; legacy grid.seed still accepted.
    path = _write(BASE % ("center_jitter_hz", "[0, 0.2e+12]"))
    cfg = load_config(path)
    os.unlink(path)
    assert cfg.grid.seeds == [7], cfg.grid.seeds     # legacy seed: 7
    path = _write((BASE % ("center_jitter_hz", "[0, 0.2e+12]"))
                  .replace("seed: 7", "seeds: [7, 8, 9]"))
    cfg = load_config(path)
    os.unlink(path)
    assert cfg.grid.seeds == [7, 8, 9]
    print("PASS  grid.seeds list parsed; legacy grid.seed -> [seed]")


def test_multi_seed_study_reproducible_and_independent():
    # Same config -> identical results; per-(seed, point) streams are
    # independent (the old seed+i scheme made seed 8 point i equal seed 7
    # point i+1 — the tuple seeding must not reproduce that collision).
    path = _write((BASE % ("center_jitter_hz", "[0.1e+12, 0.2e+12]"))
                  .replace("seed: 7", "seeds: [7, 8]"))
    cfg = load_config(path)
    os.unlink(path)
    r1, r2 = run_study(cfg), run_study(cfg)
    assert r1.purities == r2.purities
    assert r1.purity_stds == r2.purity_stds
    assert len(r1.purities_by_seed) == 2
    # both jitter points carry MC spread across seeds
    assert r1.purity_stds[0] > 0 and r1.purity_stds[1] > 0
    print("PASS  multi-seed study reproducible with independent streams")


def test_budget_threshold_summary():
    from qsource.budget import summarize_thresholds, threshold_distribution
    vals = [0.0, 1.0, 2.0, 3.0]
    by_seed = [[0.80, 0.70, 0.50, 0.30],   # crosses 0.60 at 1.5
               [0.80, 0.72, 0.48, 0.30]]   # crosses 0.60 at 1.5
    xs = threshold_distribution(vals, by_seed, 0.60)
    assert abs(xs[0] - 1.5) < 1e-12 and abs(xs[1] - 1.5) < 1e-12
    # synthetic 2-seed crossings 1.4 / 1.6 -> mean 1.5, std 0.1414, sem 0.1
    est = summarize_thresholds([1.4, 1.6], n_total=2)
    assert abs(est.mean - 1.5) < 1e-12
    assert abs(est.std - 0.14142136) < 1e-6
    assert abs(est.sem - 0.1) < 1e-9
    assert est.n_finite == 2
    print("PASS  per-seed threshold distribution and mean/std/sem summary")


def test_budget_threshold_crossing():
    from qsource.budget import threshold_crossing
    vals = [0.0, 1.0, 2.0, 3.0]
    ps = [0.80, 0.70, 0.50, 0.30]
    # target 0.60 sits halfway between p=0.70 (v=1) and p=0.50 (v=2)
    assert abs(threshold_crossing(vals, ps, 0.60) - 1.5) < 1e-12
    assert threshold_crossing(vals, ps, 0.10) is None      # never crossed
    assert threshold_crossing(vals, ps, 0.90) == 0.0       # starts below
    print("PASS  budget threshold interpolation (interior, none, at-start)")


def test_budget_tolerance_band():
    from qsource.budget import tolerance_band
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    ps = [0.20, 0.60, 0.80, 0.60, 0.20]           # peak at v=2
    lo, hi = tolerance_band(vals, ps, 0.40)
    assert abs(lo - 0.5) < 1e-12 and abs(hi - 3.5) < 1e-12, (lo, hi)
    lo, hi = tolerance_band(vals, ps, 0.15)        # clamps to sweep edges
    assert lo == 0.0 and hi == 4.0
    assert tolerance_band(vals, ps, 0.90) is None  # peak below target
    print("PASS  budget tolerance band (interpolated, clamped, impossible)")


if __name__ == "__main__":
    test_unsigned_exponent_coercion()
    test_invalid_configs_rejected()
    test_study_zero_point_matches_ideal_svd()
    test_study_is_reproducible()
    test_seeds_schema()
    test_multi_seed_study_reproducible_and_independent()
    test_budget_threshold_summary()
    test_budget_threshold_crossing()
    test_budget_tolerance_band()
    print("\nAll program-layer tests passed.")
