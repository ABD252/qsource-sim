"""Sanity tests against known physics — run with: python tests/test_core.py

These encode facts from the literature, so if the code passes, the
core math is trustworthy before we start injecting imperfections.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qsource import GaussianPump, Crystal, compute_jsa, schmidt_analysis
from qsource.crystal import GaussianCrystal, SINC_TO_GAUSSIAN_GAMMA
from qsource.analytic import gaussian_pure_purity, gaussian_jitter_purity
from qsource.realistic import LaserSpec, heralded_purity_mc, _ShiftedPump

# Reference operating point used by the analytic-benchmark tests.
_REF = dict(bandwidth_hz=0.164e12, length_m=10e-3,
            kappa_s=+0.207e-9, kappa_i=-0.318e-9)


def _quadrature_mixed_purity(crystal, jitter_hz, n=256, span_hz=4e12,
                             nodes=31):
    """Deterministic Gauss-Hermite average of rho = E_delta[A A^H].

    E[f(delta)] for delta ~ N(0, Jw^2) equals sum_k (w_k/sqrt(pi)) *
    f(sqrt(2) Jw x_k) with Hermite nodes x_k — no Monte-Carlo noise, so
    the closed form can be checked to numerical precision.
    """
    x, w = np.polynomial.hermite.hermgauss(nodes)
    jw = 2 * np.pi * jitter_hz
    rho = None
    for xk, wk in zip(x, w):
        A = compute_jsa(_ShiftedPump(_REF["bandwidth_hz"],
                                     np.sqrt(2) * jw * xk),
                        crystal, span_hz=span_hz, n=n).amplitude
        contrib = wk * (A @ A.conj().T)
        rho = contrib if rho is None else rho + contrib
    tr = np.trace(rho).real
    return float(np.trace(rho @ rho).real / tr ** 2)


def test_normalization():
    jsa = compute_jsa(GaussianPump(1e12), Crystal())
    assert abs(np.sum(jsa.intensity) - 1.0) < 1e-9, "JSA must be normalized"
    print("PASS  normalization: sum(|JSA|^2) = 1")


def test_narrowband_pump_is_highly_entangled():
    # Known physics: a very narrow (CW-like) pump forces w_s + w_i to a
    # fixed value -> strong spectral anti-correlation -> LOW purity.
    jsa = compute_jsa(GaussianPump(bandwidth_hz=5e9), Crystal(), span_hz=2e12)
    res = schmidt_analysis(jsa.amplitude)
    assert res.purity < 0.15, f"expected low purity, got {res.purity:.3f}"
    print(f"PASS  narrowband pump -> low purity ({res.purity:.3f}, K={res.schmidt_number:.1f})")


def test_purity_bounds():
    for bw in [1e10, 1e11, 1e12]:
        res = schmidt_analysis(compute_jsa(GaussianPump(bw), Crystal()).amplitude)
        assert 0.0 < res.purity <= 1.0 + 1e-9
        assert abs(res.coefficients.sum() - 1.0) < 1e-9
    print("PASS  purity in (0,1], Schmidt coefficients sum to 1")


def test_bandwidth_sweep_has_optimum():
    # Known physics: purity vs pump bandwidth has a single maximum where
    # the pump width "matches" the phase-matching width.
    bws = np.logspace(10.5, 12.5, 15)
    purities = [
        schmidt_analysis(compute_jsa(GaussianPump(b), Crystal(), span_hz=8e12).amplitude).purity
        for b in bws
    ]
    k = int(np.argmax(purities))
    assert 0 < k < len(bws) - 1, "optimum should be interior, not at an edge"
    print(f"PASS  sweep shows interior optimum: P_max={purities[k]:.3f} "
          f"at bw={bws[k]:.2e} Hz")


def test_ideal_mc_matches_ideal_svd():
    # Validated fact #1: with ZERO imperfections the Monte-Carlo mixed-state
    # purity must collapse to the ideal pure-state (SVD) purity exactly —
    # every realization is the same pure JSA, so rho = A A^H and
    # Tr(rho^2)/Tr(rho)^2 = sum(lambda_k^2) by construction.
    n, bw = 128, 0.164e12
    crystal = Crystal()
    ideal = schmidt_analysis(compute_jsa(GaussianPump(bw), crystal, n=n).amplitude).purity
    mc = heralded_purity_mc(LaserSpec(bandwidth_hz=bw), crystal, n=n,
                            k_realizations=8)["purity"]
    assert abs(mc - ideal) < 1e-9, f"MC {mc:.6f} != ideal SVD {ideal:.6f}"
    print(f"PASS  zero-imperfection MC == ideal SVD purity ({mc:.4f})")


def test_rin_does_not_change_purity():
    # Validated fact #4: RIN is a STATISTICAL imperfection — it reweights
    # identical spectral realizations, so the heralded purity is untouched
    # (the double-pair statistics are what suffer, not the spectrum).
    n, bw = 96, 0.164e12
    crystal = Crystal()
    p0 = heralded_purity_mc(LaserSpec(bandwidth_hz=bw, rin_frac=0.0),
                            crystal, n=n, k_realizations=40)["purity"]
    p_rin = heralded_purity_mc(LaserSpec(bandwidth_hz=bw, rin_frac=0.20),
                               crystal, n=n, k_realizations=40)["purity"]
    assert abs(p_rin - p0) < 1e-9, f"RIN changed purity: {p0:.6f} -> {p_rin:.6f}"
    print(f"PASS  RIN leaves heralded purity unchanged (P={p_rin:.4f})")


def test_comb_with_center_weight_only_equals_single_mode():
    # A 3-line comb whose side lines carry ZERO weight is physically the
    # same laser as a single-mode pump — the comb machinery must reproduce
    # the single-mode result exactly (offsets of unweighted lines never fire).
    n, bw = 96, 0.164e12
    crystal = Crystal()
    p_single = heralded_purity_mc(LaserSpec(bandwidth_hz=bw), crystal,
                                  n=n, k_realizations=20)["purity"]
    p_comb = heralded_purity_mc(
        LaserSpec(bandwidth_hz=bw, n_modes=3, mode_spacing_hz=0.25e12,
                  mode_weights=np.array([0.0, 1.0, 0.0])),
        crystal, n=n, k_realizations=20)["purity"]
    assert abs(p_comb - p_single) < 1e-12, f"{p_comb} != {p_single}"
    print(f"PASS  comb with only the center line == single mode ({p_comb:.4f})")


def test_jitter_degrades_purity_monotonically():
    # Known physics: center jitter mixes differently-centred JSAs into the
    # heralded state, so more jitter can only LOWER the purity. Checked on
    # a coarse ladder with margins comfortably above MC noise (~0.03).
    n, bw = 96, 0.164e12
    crystal = Crystal()
    ladder = [0.0, 0.1e12, 0.25e12, 0.5e12]
    ps = [heralded_purity_mc(
        LaserSpec(bandwidth_hz=bw, center_jitter_hz=j), crystal,
        n=n, k_realizations=60,
        rng=np.random.default_rng(3))["purity"] for j in ladder]
    for a, b in zip(ps, ps[1:]):
        assert b < a - 0.02, f"purity did not drop: {ps}"
    print("PASS  jitter ladder degrades purity monotonically "
          + " > ".join(f"{p:.3f}" for p in ps))


def test_grid_convergence_at_reference_point():
    # Discretization error at the reference point must be far below the
    # 1e-2 effects the paper reports on (measured: ~2e-5 between n=128/256).
    crystal = Crystal()
    p = [schmidt_analysis(
            compute_jsa(GaussianPump(0.164e12), crystal, n=n).amplitude
         ).purity for n in (128, 256)]
    assert abs(p[0] - p[1]) < 1e-3, f"grid not converged: {p}"
    print(f"PASS  grid convergence: |P(128)-P(256)| = {abs(p[0]-p[1]):.1e}")


def test_gamma_convention_amplitude_fwhm():
    # gamma = 0.193 is defined by matching the AMPLITUDE half-maximum of
    # sinc(x): both sinc(x) and exp(-gamma x^2) reach 1/2 at x = 1.8955.
    # Getting this convention wrong (gamma vs 2*gamma in the quadratic
    # form) moves the analytic pure purity from 0.982 to 0.873.
    xh = 1.8955
    assert abs(np.sinc(xh / np.pi) - 0.5) < 2e-4
    assert abs(np.exp(-SINC_TO_GAUSSIAN_GAMMA * xh ** 2) - 0.5) < 2e-4
    print("PASS  gamma = 0.193 matches sinc amplitude FWHM at x = 1.8955")


def test_gaussian_pm_pure_purity_matches_closed_form():
    # Double-Gaussian JSA: the SVD purity must equal the closed form
    # P0 = sqrt(1 - C^2/(A B)) exactly (Gaussian quadratures converge
    # super-exponentially on the grid).
    num = schmidt_analysis(
        compute_jsa(GaussianPump(_REF["bandwidth_hz"]),
                    GaussianCrystal(_REF["length_m"], _REF["kappa_s"],
                                    _REF["kappa_i"]),
                    span_hz=4e12, n=256).amplitude).purity
    ana = gaussian_pure_purity(**_REF)
    assert abs(num - ana) < 1e-4, f"SVD {num:.6f} != closed form {ana:.6f}"
    assert abs(ana - 0.981752) < 1e-5   # regression pin at the reference
    print(f"PASS  Gaussian-PM SVD purity == closed form P0 = {ana:.6f}")


def test_gaussian_jitter_quadrature_matches_closed_form():
    # Deterministic Gauss-Hermite mixture vs the closed form
    # P(J) = P0 / sqrt(1 + 4 c (2 pi J)^2) — the sharp validation, and it
    # pins the heralded-idler convention (rho = A A^H keeps the idler;
    # the wrong side is off by 0.10 at 0.25 THz).
    crystal = GaussianCrystal(_REF["length_m"], _REF["kappa_s"],
                              _REF["kappa_i"])
    for j in (0.1e12, 0.25e12):
        num = _quadrature_mixed_purity(crystal, j)
        ana = gaussian_jitter_purity(j, **_REF)
        assert abs(num - ana) < 1e-3, f"J={j:.2e}: {num:.6f} != {ana:.6f}"
    print("PASS  Gauss-Hermite mixed purity == closed form P(J) (<1e-3)")


def test_gaussian_jitter_mc_matches_closed_form():
    # The Monte-Carlo machinery itself against the closed form, including
    # the finite-K estimator bias E[P_MC] = P + (P0 - P)/K.
    crystal = GaussianCrystal(_REF["length_m"], _REF["kappa_s"],
                              _REF["kappa_i"])
    p0 = gaussian_pure_purity(**_REF)
    k = 120
    for j in (0.1e12, 0.25e12, 0.5e12):
        mc = heralded_purity_mc(
            LaserSpec(bandwidth_hz=_REF["bandwidth_hz"],
                      center_jitter_hz=j),
            crystal, n=128, k_realizations=k,
            rng=np.random.default_rng(7))["purity"]
        pred = gaussian_jitter_purity(j, **_REF)
        pred_k = pred + (p0 - pred) / k
        assert abs(mc - pred_k) < 0.05, \
            f"J={j:.2e}: MC {mc:.4f} vs predicted {pred_k:.4f}"
    print("PASS  Monte-Carlo (Gaussian PM) matches closed form within 0.05")


def test_sinc_jitter_tracks_analytic_shape():
    # The sinc model's NORMALIZED decay P(J)/P(0) must track the analytic
    # shape g(J) = (1 + 4 c (2 pi J)^2)^(-1/2) — deterministic quadrature,
    # measured max deviation 0.032 over the full sweep.
    crystal = Crystal(_REF["length_m"], _REF["kappa_s"], _REF["kappa_i"])
    p0_sinc = schmidt_analysis(
        compute_jsa(GaussianPump(_REF["bandwidth_hz"]), crystal,
                    span_hz=4e12, n=128).amplitude).purity
    p0_ana = gaussian_pure_purity(**_REF)
    for j in (0.1e12, 0.25e12, 0.5e12):
        shape_sinc = _quadrature_mixed_purity(crystal, j, n=128) / p0_sinc
        g = gaussian_jitter_purity(j, **_REF) / p0_ana
        assert abs(shape_sinc - g) < 0.05, \
            f"J={j:.2e}: sinc shape {shape_sinc:.4f} vs g {g:.4f}"
    print("PASS  sinc-model jitter decay tracks analytic shape g(J) (<0.05)")


if __name__ == "__main__":
    test_normalization()
    test_narrowband_pump_is_highly_entangled()
    test_purity_bounds()
    test_bandwidth_sweep_has_optimum()
    test_ideal_mc_matches_ideal_svd()
    test_rin_does_not_change_purity()
    test_comb_with_center_weight_only_equals_single_mode()
    test_jitter_degrades_purity_monotonically()
    test_grid_convergence_at_reference_point()
    test_gamma_convention_amplitude_fwhm()
    test_gaussian_pm_pure_purity_matches_closed_form()
    test_gaussian_jitter_quadrature_matches_closed_form()
    test_gaussian_jitter_mc_matches_closed_form()
    test_sinc_jitter_tracks_analytic_shape()
    print("\nAll sanity tests passed — core physics is trustworthy.")
