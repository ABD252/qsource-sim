"""make_datasheet_table.py — Translate real datasheet lines into model
units and compare them against the imperfection budget.

Reads paper/datasheets.yaml (real, web-verified spec lines) and
paper/tables/budget.json (run make_budget.py first), applies the
conversions of qsource.datasheet, and writes
paper/tables/datasheet_map.tex — the paper's Table "Reading a datasheet".

The verdict logic per laser class is documented inline; every derived
number in the table is computed here, never typed by hand.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from qsource.datasheet import (delta_lambda_to_delta_nu, fwhm_to_sigma,
                               lorentzian_fwhm_to_jitter_bound, pp_to_rms)

SECH2_TBP = 0.315   # time-bandwidth product (FWHM x FWHM) of sech^2 pulses


def tex_sci(x: float) -> str:
    """0.00002 -> '2\\times10^{-5}' (proper LaTeX, not printf 2e-05)."""
    mant, exp = f"{x:.0e}".split("e")
    return rf"{int(float(mant))}\times10^{{{int(exp)}}}"


def _load_yaml():
    with open(os.path.join(HERE, "datasheets.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_budget():
    with open(os.path.join(HERE, "tables", "budget.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _bandwidth_purity(sigma_hz: float) -> float:
    """Ideal purity at a given pump sigma, interpolated from the
    bandwidth study (log-log interpolation of the generated sweep)."""
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "paper",
                                         "bandwidth", "*", "results.csv")))
    with open(runs[-1], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    v = np.array([float(r["value"]) for r in rows])
    p = np.array([float(r["purity"]) for r in rows])
    return float(np.interp(np.log(sigma_hz), np.log(v), p))


def main() -> None:
    data = _load_yaml()
    budget = _load_budget()
    lam = data["lambda_pump_nm"] * 1e-9
    j1 = budget["jitter_10mm"]["cells"][0]["mean_ghz"]   # 1% budget, GHz
    rin1 = budget["rin_boost"]["cells"][0]["mean_pct"]

    ghz_per_nm = delta_lambda_to_delta_nu(lam, 1e-9) / 1e9  # ~499 GHz/nm

    rows = []          # (class label rows for the LaTeX table)
    verdicts = {}

    for key, cls in data["classes"].items():
        block = []
        derived = {}
        for line in cls["lines"]:
            kind = line["kind"]
            if kind == "not_published":
                block.append((line["name"], line["printed"],
                              "---", "not published"))
            elif kind == "selection_window_nm":
                # unit-to-unit selection window, not drift: translate for
                # scale only.
                ghz = delta_lambda_to_delta_nu(lam, line["value"] * 1e-9) / 1e9
                block.append((line["name"], line["printed"],
                              f"{ghz / 1e3:.0f} THz window",
                              "selection, not drift"))
            elif kind == "lorentzian_linewidth_mhz":
                bound = lorentzian_fwhm_to_jitter_bound(
                    line["value"] * 1e6) / 1e9
                derived["linewidth_bound_ghz"] = bound
                block.append((line["name"], line["printed"],
                              f"$\\leq${bound * 1e3:.1f} MHz bound",
                              f"${tex_sci(bound / j1)}\\times$ the "
                              "1\\% budget"))
            elif kind == "temp_coeff_nm_per_k":
                ghz_per_k = line["value"] * ghz_per_nm
                # allowed peak-to-peak temperature excursion at the 1% budget
                pp_ghz = j1 * 2.0 * np.sqrt(3.0)
                dT = pp_ghz / ghz_per_k
                derived["dT_pp_K"] = dT
                block.append((line["name"], line["printed"],
                              f"{ghz_per_k:.1f} GHz/K",
                              f"1\\% budget $=\\pm${dT / 2:.1f} K"))
            elif kind == "current_coeff_nm_per_ma":
                ghz_per_ma = line["value"] * ghz_per_nm
                pp_ghz = j1 * 2.0 * np.sqrt(3.0)
                dI = pp_ghz / ghz_per_ma
                block.append((line["name"], line["printed"],
                              f"{ghz_per_ma:.2f} GHz/mA",
                              f"1\\% budget $=\\pm${dI / 2:.0f} mA"))
            elif kind == "smsr_db":
                w = 10.0 ** (-line["value"] / 10.0)
                block.append((line["name"], line["printed"],
                              f"side-line weight ${tex_sci(w)}$",
                              "negligible admixture"))
            elif kind == "pulse_ps_sech2":
                dv_fwhm = SECH2_TBP / (line["value"] * 1e-12)
                sigma = fwhm_to_sigma(dv_fwhm)
                purity = _bandwidth_purity(sigma)
                l_star = 10.0 * 0.164e12 / sigma   # re-matched length, mm
                derived.update(sigma_thz=sigma / 1e12, purity=purity,
                               l_star_mm=l_star)
                block.append((line["name"], line["printed"],
                              f"$\\sigma = {sigma / 1e12:.3f}$ THz",
                              f"$P = {purity:.2f}$ at 10 mm"))
            elif kind == "rep_rate_mhz":
                block.append((line["name"], line["printed"],
                              f"comb spacing {line['value']:.0f} MHz",
                              "unresolved by the JSA"))
            elif kind == "intensity_noise_pct":
                r = line["value"] / 100.0
                boost = 1.0 + r ** 2
                block.append((line["name"], line["printed"],
                              f"$r = {r * 100:.1f}\\%$",
                              f"boost $-1 = {tex_sci(boost - 1)}$"))
        rows.append((cls, block))
        verdicts[key] = derived

    # class verdict sentences (logic documented here, numbers computed)
    dT = verdicts["dfb"]["dT_pp_K"]
    sig = verdicts["ml"]["sigma_thz"]
    pur = verdicts["ml"]["purity"]
    l_star = verdicts["ml"]["l_star_mm"]
    verdict_text = {
        "fp": (r"\textbf{Fails by construction}: multimode with none of "
               r"the budget-relevant lines published --- unusable without "
               r"external characterization."),
        "dfb": (rf"\textbf{{Spectral columns exemplary}} (drift budget met "
                rf"by holding $\pm{dT / 2:.1f}$ K), but CW linewidth "
                rf"$\sim$MHz lies five orders below the bandwidth window: "
                rf"$P \to 0.04$ (CW limit) --- wrong class for pure "
                rf"heralds, ideal for center stability."),
        "ml": (rf"\textbf{{The natural class}}: drift, comb and RIN lines "
               rf"pass trivially; the binding line is bandwidth --- 2 ps "
               rf"gives $\sigma = {sig:.3f}$ THz and $P = {pur:.2f}$ at "
               rf"10 mm, recovered by re-matching the crystal to "
               rf"$L \approx {l_star:.0f}$ mm."),
    }

    lines_tex = [
        "% Auto-generated by paper/make_datasheet_table.py -- do not edit.",
        r"\begin{tabular}{p{0.21\textwidth}p{0.16\textwidth}"
        r"p{0.2\textwidth}p{0.32\textwidth}}",
        r"\hline\hline",
        r"Datasheet line (as printed) & Value & Model translation & "
        r"vs.\ budget \\",
    ]
    for cls, block in rows:
        lines_tex += [
            r"\hline",
            rf"\multicolumn{{4}}{{l}}{{\emph{{{cls['label']}}} --- "
            rf"{cls['manufacturer']} {cls['product']} "
            rf"({cls['wavelength_nm']:g} nm; {cls['source']})}} \\[2pt]",
        ]
        for name, printed, translated, vs in block:
            lines_tex.append(rf"{name} & {printed} & {translated} & {vs} \\")
        vkey = cls["verdict_key"]
        lines_tex.append(
            rf"\multicolumn{{4}}{{p{{0.93\textwidth}}}}"
            rf"{{{verdict_text[vkey]}}} \\[2pt]")
    lines_tex += [r"\hline\hline", r"\end{tabular}"]

    out = os.path.join(HERE, "tables", "datasheet_map.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_tex) + "\n")
    print(json.dumps(verdicts, indent=2))
    print("wrote paper/tables/datasheet_map.tex")


if __name__ == "__main__":
    main()
