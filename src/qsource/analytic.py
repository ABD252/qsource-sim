"""analytic.py — Closed-form purities for the double-Gaussian JSA model.

These formulas are the paper's EXTERNAL VALIDATION: for a Gaussian pump
and Gaussian phase matching the heralded purity — pure AND jitter-mixed —
has an exact closed form, so the Monte-Carlo machinery can be checked
against mathematics rather than against itself (tests in test_core.py).

Model
-----
JSA of one realization whose pump center is shifted by delta (rad/s):

    f_delta(vs, vi) = exp(-(vs+vi-delta)^2 / (2 sigma^2))     (pump)
                    * exp(-gamma (a_s vs + a_i vi)^2)          (crystal)

with sigma = 2 pi * bandwidth_hz, a_j = kappa_j L / 2 (seconds) and
gamma = 0.193 matching the AMPLITUDE FWHM of sinc(x) (both = 1/2 at
x = 1.8955). CONVENTION TRAP: because the crystal amplitude is
exp(-gamma x^2), the curvature entering the quadratic form below is
2*gamma; confusing gamma with 2*gamma moves P0 from 0.982 to 0.873.

Quadratic form:  -1/2 [A vs^2 + 2 C vs vi + B vi^2] + (delta/sigma^2)(vs+vi)

    A = 1/sigma^2 + 2 gamma a_s^2
    B = 1/sigma^2 + 2 gamma a_i^2
    C = 1/sigma^2 + 2 gamma a_s a_i
    identity:  A B - C^2 = 2 gamma (a_s - a_i)^2 / sigma^2

Pure purity (delta = 0)
-----------------------
Tracing one photon of the 2D Gaussian gives

    P0 = sqrt( (A B - C^2) / (A B) ) = sqrt(1 - C^2/(A B)).

C = 0 (exact separability) happens at sigma^2 = 1/(2 gamma |a_s a_i|),
possible because kappa_s * kappa_i < 0 for this crystal: at the reference
GVM constants that is a 0.1997 THz pump with P0 = 1.

Jitter-mixed purity
-------------------
An imperfect pump with Gaussian center jitter J (rms, ordinary Hz) is the
classical ensemble rho = E_delta[ A_delta A_delta^dagger ], delta ~
N(0, (2 pi J)^2). Two structural facts make the average tractable:

1. A pump shift is an exact RIGID TRANSLATION of the JSA (true for sinc
   too): f_delta(vs,vi) = f_0(vs - v_s delta, vi - v_i delta) with
   v_s = -a_i/(a_s - a_i), v_i = a_s/(a_s - a_i). Hence ||A_delta|| is
   delta-independent: per-realization normalization is exact and the pair
   rate is jitter-independent in this model.
2. The cross term needed is Tr(A_d A_d^dag A_d' A_d'^dag)
   = ||A_d^dag A_d'||_F^2 — NOT |<f_d|f_d'>|^2, which is the purity of
   the mixed TWO-PHOTON state, a different quantity.

Evaluating the 4D Gaussian integral and collapsing with the identity
above gives Tr(A_d^dag A_d')_F^2 / (same at Delta=0) = exp(-c Delta^2),
Delta = delta - delta', with

    c = 1 / ( 2 sigma^2 + 1/(gamma a^2) )        [seconds^2]

where **a is the GVM constant of the TRACED-OUT photon**. The code's
rho = A @ A^H keeps the IDLER (nu_i is the row index of the JSA grid and
the signal index is contracted), so a = a_s. Mixed reduced states are
signal/idler ASYMMETRIC — using a_i instead of a_s is off by 0.10 in P
at 0.25 THz jitter. Averaging Delta ~ N(0, 2 (2 pi J)^2):

    P(J) = P0 / sqrt( 1 + 4 c (2 pi J)^2 ).

Limits: P(0) = P0; strictly decreasing; P ~ P0/(2 sqrt(c) 2 pi J) for
large J (the number of distinguishable spectral bins grows like J).
Finite-K Monte-Carlo estimator bias: E[P_MC] = P + (P0 - P)/K (the K
diagonal k = k' terms contribute exp(-c*0) = full purity), about +0.003
at K = 120.

Reference values (L = 10 mm, kappa_s = +0.207e-9 s/m,
kappa_i = -0.318e-9 s/m, bandwidth 0.164 THz): P0 = 0.981752,
c = 1.4367e-25 s^2, P(0.1 THz) = 0.88634, P(0.25) = 0.63136,
P(0.5) = 0.38008.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .crystal import SINC_TO_GAUSSIAN_GAMMA


def gaussian_jsa_coefficients(bandwidth_hz: float, length_m: float,
                              kappa_s: float, kappa_i: float,
                              gamma: float = SINC_TO_GAUSSIAN_GAMMA
                              ) -> Tuple[float, float, float]:
    """Quadratic-form coefficients (A, B, C) of the double-Gaussian JSA.

    Note the 2*gamma curvature — see the module docstring's convention
    trap. Units: seconds^2 (inverse angular-frequency squared... i.e. the
    coefficients multiply nu^2 with nu in rad/s).
    """
    sigma = 2.0 * np.pi * bandwidth_hz
    a_s = kappa_s * length_m / 2.0
    a_i = kappa_i * length_m / 2.0
    A = 1.0 / sigma ** 2 + 2.0 * gamma * a_s ** 2
    B = 1.0 / sigma ** 2 + 2.0 * gamma * a_i ** 2
    C = 1.0 / sigma ** 2 + 2.0 * gamma * a_s * a_i
    return A, B, C


def gaussian_pure_purity(bandwidth_hz: float, length_m: float,
                         kappa_s: float, kappa_i: float,
                         gamma: float = SINC_TO_GAUSSIAN_GAMMA) -> float:
    """Closed-form Schmidt purity P0 = sqrt(1 - C^2/(A B)) of the
    double-Gaussian JSA (see module docstring for the derivation)."""
    A, B, C = gaussian_jsa_coefficients(bandwidth_hz, length_m,
                                        kappa_s, kappa_i, gamma)
    return float(np.sqrt(1.0 - C ** 2 / (A * B)))


def gaussian_jitter_purity(jitter_hz_rms, bandwidth_hz: float,
                           length_m: float, kappa_s: float, kappa_i: float,
                           gamma: float = SINC_TO_GAUSSIAN_GAMMA,
                           heralded: str = "idler"):
    """Closed-form mixed purity P(J) = P0 / sqrt(1 + 4 c (2 pi J)^2).

    heralded="idler" matches qsource.realistic.heralded_purity_mc, whose
    rho = A @ A^H keeps the idler (row) index of the JSA grid — then c
    carries the SIGNAL's GVM constant a_s (the traced-out photon).
    Vectorized over jitter_hz_rms.
    """
    if heralded not in ("idler", "signal"):
        raise ValueError("heralded must be 'idler' or 'signal'")
    sigma = 2.0 * np.pi * bandwidth_hz
    kappa_traced = kappa_s if heralded == "idler" else kappa_i
    a = kappa_traced * length_m / 2.0
    c = 1.0 / (2.0 * sigma ** 2 + 1.0 / (gamma * a * a))
    jw = 2.0 * np.pi * np.asarray(jitter_hz_rms, dtype=float)
    p0 = gaussian_pure_purity(bandwidth_hz, length_m, kappa_s, kappa_i,
                              gamma)
    out = p0 / np.sqrt(1.0 + 4.0 * c * jw ** 2)
    return float(out) if np.isscalar(jitter_hz_rms) else out
