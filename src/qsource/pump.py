"""pump.py — Pump laser spectral envelope.

The pump enters the JSA as alpha(w_s + w_i): energy conservation means
the pair frequencies must sum to (something within) the pump spectrum.

This module is THE entry point for your whole research question:
every pump imperfection you will study later (linewidth, RIN, drift,
jitter) gets injected here, and nowhere else. Keep it clean.

Units convention: all frequencies are ANGULAR frequency detunings in
rad/s relative to the central design frequencies. Working in detunings
(nu = w - w0) keeps numbers small and the math readable.
"""

import numpy as np


class GaussianPump:
    """Ideal transform-limited Gaussian pump — the textbook baseline.

    alpha(nu_s + nu_i) = exp( -(nu_s + nu_i)^2 / (2 * sigma^2) )

    Parameters
    ----------
    bandwidth_hz : float
        Spectral bandwidth as a standard deviation, in ordinary Hz
        (we convert to angular internally). A narrowband CW-like pump
        might be ~1e9 Hz; a femtosecond pulsed pump ~1e12-1e13 Hz.
    """

    def __init__(self, bandwidth_hz: float):
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        self.sigma = 2.0 * np.pi * bandwidth_hz  # rad/s

    def envelope(self, nu_s: np.ndarray, nu_i: np.ndarray) -> np.ndarray:
        """Complex spectral amplitude evaluated on a (nu_s, nu_i) grid."""
        s = nu_s + nu_i
        return np.exp(-(s ** 2) / (2.0 * self.sigma ** 2))

    def __repr__(self):
        return f"GaussianPump(bandwidth={self.sigma / (2 * np.pi):.3e} Hz)"


# ---------------------------------------------------------------------------
# Placeholders for your research phase — implement these one by one.
# Each returns a modified envelope; everything downstream stays unchanged.
# ---------------------------------------------------------------------------

class BroadenedPump(GaussianPump):
    """Week-3 exercise: pump with excess linewidth (incoherent broadening).

    Hint: an incoherently broadened pump is NOT just a wider Gaussian —
    coherent bandwidth and incoherent linewidth affect purity differently.
    Start simple (treat as wider Gaussian), then refine with a mixed-state
    model (spectral density convolution). The difference IS a finding.
    """
    pass
