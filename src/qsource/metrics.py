"""metrics.py — Entanglement-source quality metrics.

For now: spectral purity via Schmidt decomposition. Later phases add
heralding efficiency and HOM visibility here — one module, one job:
turn a JSA into numbers you can put on a plot.

The Schmidt decomposition of a discretized JSA is *literally* the SVD:

    JSA_matrix = U @ diag(s) @ Vh

The normalized squared singular values lambda_k = s_k^2 / sum(s^2) are
the Schmidt coefficients. Then:

    purity  P = sum(lambda_k^2)          (P = 1 means fully separable)
    Schmidt number K = 1 / P             (effective number of modes)

Physical reading: K ~= "how many distinguishable spectral modes the
photon secretly lives in". K = 1 -> the idler's spectrum carries zero
which-photon information -> polarization entanglement stays clean.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SchmidtResult:
    purity: float
    schmidt_number: float
    coefficients: np.ndarray  # lambda_k, sorted descending, sums to 1


def schmidt_analysis(jsa_amplitude: np.ndarray, tol: float = 1e-12) -> SchmidtResult:
    """Run the Schmidt (SVD) decomposition on a 2D JSA amplitude."""
    s = np.linalg.svd(jsa_amplitude, compute_uv=False)
    lam = s ** 2
    total = lam.sum()
    if total < tol:
        raise ValueError("JSA has ~zero norm; cannot decompose")
    lam = lam / total
    purity = float(np.sum(lam ** 2))
    return SchmidtResult(
        purity=purity,
        schmidt_number=1.0 / purity,
        coefficients=lam,
    )
