"""datasheet.py — Laser datasheet units -> model units. No physics, only algebra.

This module is the bridge the paper's title promises: manufacturers
publish linewidths in MHz, stabilities in pm, RIN in dB/Hz and cavity
lengths in mm — none of which the simulation accepts directly. Each
function here converts ONE datasheet line into the corresponding model
parameter, states its physical assumption, and carries its validity
caveat. Target conventions (verified against pump.py / realistic.py):

- GaussianPump(bandwidth_hz): STANDARD DEVIATION of the amplitude
  envelope, ordinary Hz.
- LaserSpec.center_jitter_hz: Gaussian quasi-static rms of the center
  frequency, ordinary Hz.
- LaserSpec.rin_frac: fractional rms intensity noise.
- LaserSpec.mode_spacing_hz: longitudinal-mode spacing, ordinary Hz.
"""

from __future__ import annotations

import math

C_M_PER_S = 299_792_458.0
_FWHM_INTENSITY = 2.0 * math.sqrt(math.log(2.0))         # 1.6651092...
_FWHM_AMPLITUDE = 2.0 * math.sqrt(2.0 * math.log(2.0))   # 2.3548200...


def fwhm_to_sigma(fwhm_hz: float, of_intensity: bool = True) -> float:
    """Gaussian FWHM (Hz) -> model sigma (amplitude std, Hz).

    Datasheets and OSA traces quote the FWHM of the INTENSITY spectrum;
    the model's |alpha|^2 = exp(-nu^2/sigma^2) has intensity FWHM
    2*sqrt(ln 2)*sigma, so sigma = FWHM/1.6651092. With
    of_intensity=False the FWHM is interpreted as that of the AMPLITUDE
    envelope: sigma = FWHM/2.3548200.

    Caveat: assumes a transform-limited Gaussian line; for sech^2 pulses
    or multimode envelopes this is only an effective width.
    """
    if fwhm_hz <= 0:
        raise ValueError("fwhm_hz must be positive")
    return fwhm_hz / (_FWHM_INTENSITY if of_intensity else _FWHM_AMPLITUDE)


def sigma_to_fwhm(sigma_hz: float, of_intensity: bool = True) -> float:
    """Inverse of fwhm_to_sigma — for writing table entries back in
    datasheet language."""
    if sigma_hz <= 0:
        raise ValueError("sigma_hz must be positive")
    return sigma_hz * (_FWHM_INTENSITY if of_intensity else _FWHM_AMPLITUDE)


def delta_lambda_to_delta_nu(lambda_m: float, dlambda_m: float) -> float:
    """Wavelength excursion (m) -> frequency excursion (Hz): c dl / l^2.

    First-order expansion of nu = c/lambda — exact enough for pm-to-nm
    excursions (dl << l always holds for stability specs).
    """
    return C_M_PER_S * dlambda_m / lambda_m ** 2


def delta_nu_to_delta_lambda(lambda_m: float, dnu_hz: float) -> float:
    """Frequency excursion (Hz) -> wavelength excursion (m): l^2 dnu / c."""
    return lambda_m ** 2 * dnu_hz / C_M_PER_S


def pp_to_rms(pp_hz: float, distribution: str = "uniform") -> float:
    """Peak-to-peak drift spec -> rms jitter (Hz).

    A spec like "+-a over T hours" or "(pm/K coefficient) x (K stability)"
    bounds a quasi-static excursion band of width pp = 2a; it is not a
    Gaussian. distribution="uniform" (default) assumes the center is
    uniformly distributed over the band: rms = pp/(2 sqrt 3).
    distribution="bound" returns the conservative half-range pp/2.

    Caveat: the model then SAMPLES A GAUSSIAN with this rms; the purity
    difference between the two distribution choices is at the few-percent
    level, inside the quoted Monte-Carlo error bars.
    """
    if pp_hz < 0:
        raise ValueError("pp_hz must be non-negative")
    if distribution == "uniform":
        return pp_hz / (2.0 * math.sqrt(3.0))
    if distribution == "bound":
        return pp_hz / 2.0
    raise ValueError("distribution must be 'uniform' or 'bound'")


def rin_db_per_hz_to_frac(rin_db_hz: float, bandwidth_hz: float) -> float:
    """RIN spectral density (dB/Hz) -> fractional rms r over a bandwidth.

    r = sqrt(10^(RIN/10) * B), assuming a WHITE RIN density over the
    integration bandwidth B (the inverse of the pair-collection
    timescale: detection gate or pulse period).

    Caveat: real RIN spectra carry 1/f excess at low frequency and a
    relaxation-oscillation peak; when the manufacturer publishes the
    curve, integrate it numerically instead — this function is the
    flat-spectrum estimate only.
    """
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    return math.sqrt(10.0 ** (rin_db_hz / 10.0) * bandwidth_hz)


def fsr_from_cavity(length_m: float, n_index: float = 1.0,
                    ring: bool = False) -> float:
    """Cavity length -> longitudinal-mode spacing (free spectral range).

    Linear (standing-wave) cavity: FSR = c/(2 n L); ring=True: c/(n L).
    Maps directly onto LaserSpec.mode_spacing_hz.

    Caveat: strictly the GROUP index belongs here; n_index is usually
    quoted as the phase index — the difference is a few percent for
    semiconductor diodes.
    """
    if length_m <= 0 or n_index <= 0:
        raise ValueError("length_m and n_index must be positive")
    round_trip = (1.0 if ring else 2.0) * n_index * length_m
    return C_M_PER_S / round_trip


def lorentzian_fwhm_to_jitter_bound(fwhm_hz: float) -> float:
    """Lorentzian linewidth FWHM -> conservative quasi-static bound (HWHM).

    HONESTY POLICY: a Lorentzian line arises from FAST phase diffusion
    and has no finite second moment, so no rms exists and the
    quasi-static ensemble of realistic.py does not describe it — fast
    phase noise decorrelates within a pulse and does not rigidly displace
    the pump band. This function returns the HWHM as a deliberately
    conservative stand-in bound, valid only when the linewidth is far
    below the phase-matching bandwidth (typical DFB linewidths are MHz —
    three orders below the ~26 GHz jitter budgets, so the bound is
    comfortably negligible). SLOW frequency-noise and thermal-drift specs
    (+-pm over T hours, pm/K x K) ARE quasi-static: convert those via
    delta_lambda_to_delta_nu + pp_to_rms instead.
    """
    if fwhm_hz < 0:
        raise ValueError("fwhm_hz must be non-negative")
    return fwhm_hz / 2.0
