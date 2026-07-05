"""realistic.py — Modeling a REAL laser without buying one.

The trick: a real laser's imperfections are STATISTICAL, and manufacturers
publish the statistics for free in datasheets:

  - linewidth        -> spectral width of each realization (Hz)
  - center jitter /  -> the center frequency wanders shot-to-shot and
    drift               slowly over time (Hz rms)
  - RIN (intensity   -> pump power fluctuates -> each realization carries
    noise / spikes)     a random weight (fractional rms)
  - multimode 'gaps' -> the laser emits several discrete lines (a comb),
                        not one line  [Abdullah's own suggestion!]

Physics of WHY this needs more than one JSA:
An imperfect pump is a MIXED state — a classical ensemble of pure pumps.
So we Monte-Carlo K realizations, compute a JSA for each, and average
the *heralded single-photon density matrix*:

    rho_s = sum_k  w_k * (A_k @ A_k^dagger)

Then the heralded-photon purity is  P = Tr(rho_s^2) / Tr(rho_s)^2.
With zero imperfections this collapses to the ideal-pump purity —
that is our sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .jsa import compute_jsa
from .pump import GaussianPump


@dataclass
class LaserSpec:
    """Datasheet numbers -> simulation parameters. All optional;
    zeros = ideal laser (must reproduce the ideal result)."""
    bandwidth_hz: float                 # coherent spectral width (as before)
    center_jitter_hz: float = 0.0       # rms wander of center frequency
    rin_frac: float = 0.0               # relative intensity noise, fractional rms (e.g. 0.02 = 2%)
    n_modes: int = 1                    # >1 = multimode comb ("gaps")
    mode_spacing_hz: float = 0.0        # spacing between comb lines
    mode_weights: np.ndarray | None = None  # optional custom line strengths


class _ShiftedPump(GaussianPump):
    """One pump realization: Gaussian line shifted by delta (rad/s)."""

    def __init__(self, bandwidth_hz: float, delta_rad: float):
        super().__init__(bandwidth_hz)
        self.delta = delta_rad

    def envelope(self, nu_s, nu_i):
        s = nu_s + nu_i - self.delta
        return np.exp(-(s ** 2) / (2.0 * self.sigma ** 2))


def heralded_purity_mc(spec: LaserSpec, crystal, span_hz: float = 4e12,
                       n: int = 256, k_realizations: int = 60,
                       rng: np.random.Generator | None = None) -> dict:
    """Monte-Carlo heralded purity for a realistic laser.

    Returns dict with purity, schmidt_number, and the per-realization
    weights (useful for pair-rate/RIN statistics later).
    """
    rng = rng or np.random.default_rng(7)

    # --- comb structure (multimode): pick line offsets and base weights
    if spec.n_modes > 1:
        idx = np.arange(spec.n_modes) - (spec.n_modes - 1) / 2.0
        line_offsets = 2 * np.pi * spec.mode_spacing_hz * idx
        base_w = (spec.mode_weights if spec.mode_weights is not None
                  else np.ones(spec.n_modes))
        base_w = np.asarray(base_w, float) / np.sum(base_w)
    else:
        line_offsets = np.array([0.0])
        base_w = np.array([1.0])

    rho = None
    weights = []
    for _ in range(k_realizations):
        # sample which comb line this pair came from
        line = rng.choice(len(line_offsets), p=base_w)
        # sample center jitter/drift
        jit = rng.normal(0.0, 2 * np.pi * spec.center_jitter_hz)
        delta = line_offsets[line] + jit
        # sample RIN: pair-generation weight ~ instantaneous power
        w = max(0.0, 1.0 + rng.normal(0.0, spec.rin_frac))
        weights.append(w)

        pump = _ShiftedPump(spec.bandwidth_hz, delta)
        A = compute_jsa(pump, crystal, span_hz=span_hz, n=n).amplitude
        # A @ A^H contracts the COLUMN (signal) index of the JSA grid
        # (nu_i is the row index from meshgrid), so rho is the reduced
        # state of the heralded IDLER. Mixed reduced purities are NOT
        # signal/idler symmetric — see qsource/analytic.py.
        contrib = w * (A @ A.conj().T)
        rho = contrib if rho is None else rho + contrib

    tr = np.trace(rho).real
    purity = float(np.trace(rho @ rho).real / tr ** 2)
    return {
        "purity": purity,
        "schmidt_number": 1.0 / purity,
        "weights": np.array(weights),
    }
