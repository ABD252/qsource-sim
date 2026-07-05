"""crystal.py — Nonlinear crystal phase-matching function.

The crystal enters the JSA as phi(nu_s, nu_i): momentum conservation
inside a crystal of finite length L gives the classic sinc shape.

We use the standard first-order (group-velocity) expansion:

    Delta_k ~= kappa_s * nu_s + kappa_i * nu_i

where kappa_j = 1/v_g(pump) - 1/v_g(j) is the group-velocity mismatch
between the pump and photon j, in s/m. The phase-matching function is:

    phi = sinc( Delta_k * L / 2 )

The RATIO kappa_s / kappa_i sets the tilt angle of the sinc ridge in
the (nu_s, nu_i) plane — and that tilt, combined with the pump's -45°
diagonal, decides how separable (pure) the JSA is. This geometric
picture is the single most important intuition of the whole project.
"""

import numpy as np


class Crystal:
    """Quasi-phase-matched crystal, first-order dispersion model.

    Parameters
    ----------
    length_m : float
        Crystal length in meters (typical: 1e-3 to 30e-3).
    kappa_s, kappa_i : float
        Group-velocity mismatch pump-vs-signal and pump-vs-idler, s/m.
        Defaults are representative of PPKTP at 775 -> 1550 nm
        (order of magnitude; refine with Sellmeier data later).
    """

    def __init__(self, length_m: float = 10e-3,
                 kappa_s: float = +0.207e-9,
                 kappa_i: float = -0.318e-9):
        if length_m <= 0:
            raise ValueError("length_m must be positive")
        self.L = length_m
        self.kappa_s = kappa_s
        self.kappa_i = kappa_i

    def phase_matching(self, nu_s: np.ndarray, nu_i: np.ndarray) -> np.ndarray:
        """sinc-shaped phase-matching amplitude on a (nu_s, nu_i) grid."""
        delta_k = self.kappa_s * nu_s + self.kappa_i * nu_i
        # np.sinc(x) = sin(pi x)/(pi x), so divide the argument by pi
        return np.sinc(delta_k * self.L / (2.0 * np.pi))

    def __repr__(self):
        return (f"Crystal(L={self.L * 1e3:.1f} mm, "
                f"kappa_s={self.kappa_s:.3e}, kappa_i={self.kappa_i:.3e})")


# Amplitude-FWHM matching of sinc(x) by exp(-gamma x^2): both reach 1/2
# at x = 1.8955, giving gamma = ln(2)/1.8955^2 = 0.193. NOTE the
# convention: this matches the AMPLITUDE half-maximum; the curvature that
# enters Gaussian quadratic forms is therefore 2*gamma (see analytic.py).
SINC_TO_GAUSSIAN_GAMMA = 0.193


class GaussianCrystal(Crystal):
    """Gaussian phase matching: phi = exp(-gamma * (Delta_k L / 2)^2).

    Physically this is the phase-matching function of an apodized
    (domain-engineered) crystal whose nonlinearity profile is Gaussian —
    the sinc side lobes of a uniform crystal are removed. Two uses here:

    1. The double-Gaussian JSA (Gaussian pump x Gaussian phase matching)
       admits CLOSED-FORM purities, pure and jitter-mixed alike
       (qsource.analytic), so the Monte-Carlo machinery can be validated
       against exact analytics.
    2. It represents the engineered-crystal sources of current practice,
       showing the imperfection budgets are not an artifact of sinc
       side lobes.

    Same interface as Crystal — everything downstream is unchanged.
    """

    def __init__(self, length_m: float = 10e-3,
                 kappa_s: float = +0.207e-9,
                 kappa_i: float = -0.318e-9,
                 gamma: float = SINC_TO_GAUSSIAN_GAMMA):
        super().__init__(length_m, kappa_s, kappa_i)
        self.gamma = gamma

    def phase_matching(self, nu_s: np.ndarray, nu_i: np.ndarray) -> np.ndarray:
        """Gaussian phase-matching amplitude on a (nu_s, nu_i) grid."""
        x = (self.kappa_s * nu_s + self.kappa_i * nu_i) * self.L / 2.0
        return np.exp(-self.gamma * x ** 2)

    def __repr__(self):
        return (f"GaussianCrystal(L={self.L * 1e3:.1f} mm, "
                f"kappa_s={self.kappa_s:.3e}, kappa_i={self.kappa_i:.3e}, "
                f"gamma={self.gamma})")
