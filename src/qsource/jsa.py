"""jsa.py — Joint Spectral Amplitude: the heart of the simulation.

    JSA(nu_s, nu_i) = alpha_pump(nu_s + nu_i) * phi_crystal(nu_s, nu_i)

That one multiplication encodes energy conservation (pump term, a -45°
diagonal band) times momentum conservation (crystal term, a tilted sinc
ridge). Their overlap region IS the two-photon state.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class JSAResult:
    """Container for a computed JSA and its frequency grid."""
    nu_s: np.ndarray        # 1D signal detuning axis (rad/s)
    nu_i: np.ndarray        # 1D idler detuning axis (rad/s)
    amplitude: np.ndarray   # 2D complex JSA, shape (len(nu_i), len(nu_s))

    @property
    def intensity(self) -> np.ndarray:
        """|JSA|^2 — the Joint Spectral Intensity (what experiments plot)."""
        return np.abs(self.amplitude) ** 2


def compute_jsa(pump, crystal, span_hz: float = 4e12, n: int = 512) -> JSAResult:
    """Evaluate the JSA on a square grid.

    Parameters
    ----------
    pump : object with .envelope(nu_s, nu_i)
    crystal : object with .phase_matching(nu_s, nu_i)
    span_hz : float
        Half-width of the frequency window in ordinary Hz. Must be wide
        enough to contain the JSA — check visually that the intensity
        decays to ~0 before the edges, or purity numbers will be wrong.
    n : int
        Grid points per axis. 512 is a good accuracy/speed balance.
    """
    axis = np.linspace(-2 * np.pi * span_hz, 2 * np.pi * span_hz, n)
    nu_s, nu_i = np.meshgrid(axis, axis)
    amp = pump.envelope(nu_s, nu_i) * crystal.phase_matching(nu_s, nu_i)

    # Normalize so that sum(|JSA|^2) = 1 — a proper quantum state.
    norm = np.sqrt(np.sum(np.abs(amp) ** 2))
    if norm == 0:
        raise ValueError("JSA is identically zero — check parameters/span")
    return JSAResult(nu_s=axis, nu_i=axis, amplitude=amp / norm)
