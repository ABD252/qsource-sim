"""Datasheet-translation tests — run with: python tests/test_datasheet.py

Each assertion pins a hand-computed value, so the unit conversions that
feed the paper's "Reading a datasheet" table can never drift silently.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qsource.datasheet import (delta_lambda_to_delta_nu,
                               delta_nu_to_delta_lambda, fsr_from_cavity,
                               fwhm_to_sigma, lorentzian_fwhm_to_jitter_bound,
                               pp_to_rms, rin_db_per_hz_to_frac,
                               sigma_to_fwhm)


def test_fwhm_sigma_roundtrip():
    # Gaussian intensity FWHM = 2*sqrt(ln 2) * sigma = 1.6651092 sigma:
    # 1 THz FWHM -> sigma = 0.6005612 THz.
    assert abs(fwhm_to_sigma(1.0e12) - 0.6005612e12) < 1e6
    # Paper's reference sigma = 0.164 THz -> intensity FWHM 0.2730779 THz.
    assert abs(sigma_to_fwhm(0.164e12) - 0.2730779e12) < 1e6
    # Amplitude convention: FWHM = 2*sqrt(2 ln 2) * sigma = 2.3548200 sigma.
    assert abs(fwhm_to_sigma(1.0e12, of_intensity=False)
               - 1.0e12 / 2.3548200) < 1e6
    print("PASS  Gaussian FWHM <-> sigma (intensity and amplitude)")


def test_wavelength_frequency_conversion():
    # c/lambda^2 at 775 nm = 2.99792458e8 / 6.00625e-13 = 4.99136e20 Hz/m,
    # so 52 pm -> 2.59551e10 Hz — pins the paper's 26 GHz <-> ~52 pm claim.
    assert abs(delta_lambda_to_delta_nu(775e-9, 52e-12) - 2.59551e10) < 1e7
    # Inverse: 26 GHz at 775 nm -> 52.09 pm.
    assert abs(delta_nu_to_delta_lambda(775e-9, 26e9) - 52.09e-12) < 1e-13
    print("PASS  delta_lambda <-> delta_nu at 775 nm (52 pm <-> 26 GHz)")


def test_rin_integration():
    # -110 dB/Hz white RIN over 1 GHz: 10^-11 * 1e9 = 1e-2 -> r = 0.10,
    # exactly the paper's +1% double-pair-boost budget.
    assert abs(rin_db_per_hz_to_frac(-110.0, 1e9) - 0.10) < 1e-12
    print("PASS  RIN -110 dB/Hz over 1 GHz -> r = 10% rms")


def test_fsr_from_cavity():
    # FP diode: L = 0.5 mm, n = 3.5 -> FSR = c/(2 n L) = 85.655 GHz.
    assert abs(fsr_from_cavity(0.5e-3, 3.5) - 8.56550e10) < 1e7
    # Ring cavity doubles the FSR relative formula: c/(n L).
    assert abs(fsr_from_cavity(0.5e-3, 3.5, ring=True) - 1.71310e11) < 2e7
    print("PASS  FSR from cavity length (linear and ring)")


def test_pp_and_lorentzian_bounds():
    # Uniform +-10 GHz band (pp = 20 GHz): rms = pp/(2 sqrt 3) = 5.7735 GHz.
    assert abs(pp_to_rms(20e9) - 5.7735027e9) < 1e3
    # Conservative bound convention: pp/2.
    assert pp_to_rms(20e9, distribution="bound") == 10e9
    # Lorentzian FWHM 2 MHz -> quasi-static bound = HWHM = 1 MHz.
    assert lorentzian_fwhm_to_jitter_bound(2e6) == 1e6
    print("PASS  peak-to-peak -> rms and Lorentzian HWHM bound")


if __name__ == "__main__":
    test_fwhm_sigma_roundtrip()
    test_wavelength_frequency_conversion()
    test_rin_integration()
    test_fsr_from_cavity()
    test_pp_and_lorentzian_bounds()
    print("\nAll datasheet-translation tests passed.")
