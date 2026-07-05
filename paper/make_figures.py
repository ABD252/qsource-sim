"""make_figures.py — Assemble the paper's figures from results/paper/.

Every curve here comes from a results/paper/<study>/<timestamp>/results.csv
that was produced by `qsource run configs/paper/<study>.yaml` — so each
figure is reproducible from config files alone (hard rule 5). This script
only reads, arranges and styles; it computes no physics, with one
exception: Fig. 1 evaluates the (validated) core functions directly to
draw the JSA anatomy at the reference operating point, using the SAME
parameters as configs/paper/jitter.yaml.

Output: paper/figures/fig*.pdf (for LaTeX) and .png (for quick viewing).
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from qsource import Crystal, GaussianPump, compute_jsa, schmidt_analysis  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "paper")
FIGDIR = os.path.join(HERE, "figures")

# Reference operating point — MUST match configs/paper/jitter.yaml.
BW_HZ = 0.164e12
CRYSTAL = dict(length_m=10e-3, kappa_s=0.207e-9, kappa_i=-0.318e-9)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
})

C = {"gold": "#c99138", "teal": "#2aa198", "red": "#dc4a5e",
     "blue": "#5b7fb9", "green": "#3d8b40"}


def load(study: str):
    """Latest results.csv of a study -> dict of column arrays."""
    runs = sorted(glob.glob(os.path.join(RESULTS, study, "*", "results.csv")))
    if not runs:
        raise FileNotFoundError(f"no results for study '{study}' — "
                                "run paper/run_studies.py")
    with open(runs[-1], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "value": np.array([float(r["value"]) for r in rows]),
        "display": np.array([float(r["value_display"]) for r in rows]),
        "purity": np.array([float(r["purity"]) for r in rows]),
        "schmidt": np.array([float(r["schmidt_number"]) for r in rows]),
        "boost": np.array([float(r["double_pair_boost"]) for r in rows]),
        "purity_std": np.array([float(r.get("purity_std", 0.0))
                                for r in rows]),
        "n_seeds": int(rows[0].get("n_seeds", 1)),
    }


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------- figure 1
def fig1_jsa_anatomy():
    """Pump band x phase-matching ridge = JSA, plus Schmidt spectrum."""
    pump = GaussianPump(BW_HZ)
    crystal = Crystal(**CRYSTAL)
    # Display grid: zoomed to +-1.5 THz so the structure is visible.
    jsa = compute_jsa(pump, crystal, span_hz=1.5e12, n=384)
    ax_thz = jsa.nu_s / (2 * np.pi * 1e12)
    ext = [ax_thz[0], ax_thz[-1], ax_thz[0], ax_thz[-1]]
    ns, ni = np.meshgrid(jsa.nu_s, jsa.nu_i)
    alpha2 = np.abs(pump.envelope(ns, ni)) ** 2
    phi2 = np.abs(crystal.phase_matching(ns, ni)) ** 2
    # Schmidt numbers from the CONVERGED window (the zoomed grid clips the
    # sinc tails and would overstate P: 0.824 instead of 0.810).
    schmidt = schmidt_analysis(
        compute_jsa(pump, crystal, span_hz=4e12, n=384).amplitude)

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), constrained_layout=True)
    for ax, dat, title in [(axes[0], alpha2, r"pump $|\alpha|^2$"),
                           (axes[1], phi2, r"crystal $|\phi|^2$"),
                           (axes[2], jsa.intensity, r"JSA $|f|^2$")]:
        ax.imshow(dat, origin="lower", extent=ext, cmap="magma", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel(r"$\nu_s/2\pi$ (THz)")
        ax.grid(False)
    axes[0].set_ylabel(r"$\nu_i/2\pi$ (THz)")
    lam = schmidt.coefficients[:8]
    axes[3].bar(np.arange(1, len(lam) + 1), lam, color=C["teal"])
    axes[3].set_title(rf"$P={schmidt.purity:.3f}$, "
                      rf"$K={schmidt.schmidt_number:.2f}$", fontsize=8.5)
    axes[3].set_xlabel(r"mode index $k$")
    axes[3].set_ylabel(r"$\lambda_k$")
    save(fig, "fig1_jsa_anatomy")


# --------------------------------------------------------------- figure 2
def fig2_design():
    """Ideal-pump design curves: bandwidth optimum + length matching."""
    bw, ln = load("bandwidth"), load("length")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.semilogx(bw["display"], bw["purity"], "o-", color=C["gold"])
    k = int(np.argmax(bw["purity"]))
    ax.axvline(bw["display"][k], ls=":", c="k", alpha=0.5)
    ax.annotate(rf"$P_{{\max}}={bw['purity'][k]:.3f}$"
                + "\n" + rf"at {bw['display'][k]:.2f} THz",
                (bw["display"][k], bw["purity"][k]),
                textcoords="offset points", xytext=(8, -28), fontsize=7.5)
    ax.set_xlabel("pump bandwidth (THz)")
    ax.set_ylabel("spectral purity $P$")
    ax.set_title("(a) bandwidth matching (ideal pump)")

    ax = axes[1]
    ax.plot(ln["display"], ln["purity"], "s-", color=C["blue"])
    ax.set_xlabel("crystal length (mm)")
    ax.set_ylabel("spectral purity $P$")
    ax.set_title("(b) length matching at 0.164 THz pump")
    save(fig, "fig2_design")


# --------------------------------------------------------------- figure 3
def fig3_jitter():
    """Jitter tolerance for three crystal lengths, +-1 sigma seed bands."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    for study, color, marker, lab in [
            ("jitter_L20", C["red"], "^", "20 mm"),
            ("jitter", C["gold"], "o", "10 mm"),
            ("jitter_L5", C["teal"], "s", "5 mm")]:
        d = load(study)
        ax.fill_between(d["display"], d["purity"] - d["purity_std"],
                        d["purity"] + d["purity_std"], color=color,
                        alpha=0.22)
        ax.plot(d["display"], d["purity"], marker + "-", color=color,
                label=f"{lab} crystal")
        ax.axhline(d["purity"][0], ls=":", c=color, alpha=0.7)
    ax.set_xlabel("center jitter / drift rms (THz)")
    ax.set_ylabel("heralded purity $P$")
    ax.legend(title=f"bands: $\\pm1\\sigma$, {load('jitter')['n_seeds']} seeds",
              fontsize=7, title_fontsize=7)
    save(fig, "fig3_jitter")


# --------------------------------------------------------------- figure 4
def fig4_multimode():
    """Multimode comb: line count and line spacing."""
    md, sp = load("modes"), load("spacing")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(md["display"], md["purity"], "D-", color=C["red"])
    ax.axhline(md["purity"][0], ls=":", c="k", alpha=0.5, label="single mode")
    ax.set_xlabel("number of longitudinal modes")
    ax.set_ylabel("heralded purity $P$")
    ax.set_title("(a) modes at 0.25 THz spacing")
    ax.legend()

    ax = axes[1]
    ax.plot(sp["display"], sp["purity"], "^-", color=C["blue"])
    ax.axhline(md["purity"][0], ls=":", c="k", alpha=0.5, label="single mode")
    ax.set_xlabel("mode spacing (THz)")
    ax.set_ylabel("heralded purity $P$")
    ax.set_title("(b) spacing of a 3-mode comb")
    ax.legend()
    save(fig, "fig4_multimode")


# --------------------------------------------------------------- figure 5
def fig5_rin():
    """RIN: purity exactly invariant, double-pair boost is the casualty."""
    rn = load("rin")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(rn["display"], rn["purity"], "o-", color=C["gold"])
    ax.set_ylim(rn["purity"][0] - 0.05, rn["purity"][0] + 0.05)
    ax.ticklabel_format(useOffset=False, axis="y")
    ax.set_xlabel("RIN (% rms)")
    ax.set_ylabel("heralded purity $P$")
    ax.set_title("(a) purity: exactly invariant")

    ax = axes[1]
    r = rn["display"] / 100.0
    ax.plot(rn["display"], rn["boost"], "s-", color=C["red"],
            label=r"Monte-Carlo ($K=10^3$)")
    ax.plot(rn["display"], 1 + r ** 2, "--", color="k", alpha=0.6,
            label=r"$1+r^2$ (small-$r$ theory)")
    ax.axhline(1.0, ls=":", c=C["green"])
    ax.set_xlabel("RIN (% rms)")
    ax.set_ylabel(r"double-pair boost $\langle w^2\rangle/\langle w\rangle^2$")
    ax.set_title("(b) pair statistics: the real casualty")
    ax.legend()
    save(fig, "fig5_rin")


# --------------------------------------------------------------- figure 6
def fig6_validation():
    """External validation: MC vs the closed form of qsource.analytic.

    (a) Gaussian phase matching, where P(J) = P0/sqrt(1+4c(2 pi J)^2) is
    EXACT — the MC seed band must sit on the dashed curve. (b) The sinc
    model's normalized decay P(J)/P(0) against the same analytic shape
    g(J): an approximation (gamma = 0.193), good to ~0.03.
    """
    from qsource.analytic import gaussian_jitter_purity, gaussian_pure_purity

    ref = dict(bandwidth_hz=BW_HZ, length_m=CRYSTAL["length_m"],
               kappa_s=CRYSTAL["kappa_s"], kappa_i=CRYSTAL["kappa_i"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)

    ax = axes[0]
    g = load("jitter_gauss")
    jgrid = np.linspace(0, g["value"].max(), 300)
    ax.fill_between(g["display"], g["purity"] - g["purity_std"],
                    g["purity"] + g["purity_std"], color=C["gold"],
                    alpha=0.25, label=rf"MC $\pm1\sigma$ ({g['n_seeds']} seeds)")
    ax.plot(g["display"], g["purity"], "o", color=C["gold"], ms=3)
    ax.plot(jgrid / 1e12, gaussian_jitter_purity(jgrid, **ref), "--", c="k",
            label=r"closed form $P_0[1+4c(2\pi J)^2]^{-1/2}$")
    ax.axhline(gaussian_pure_purity(**ref), ls=":", c="k", alpha=0.5)
    ax.set_xlabel("center jitter rms (THz)")
    ax.set_ylabel("heralded purity $P$")
    ax.set_title("(a) Gaussian phase matching: exact benchmark")
    ax.legend(fontsize=7)

    ax = axes[1]
    s = load("jitter")
    shape_mc = s["purity"] / s["purity"][0]
    shape_ana = (gaussian_jitter_purity(jgrid, **ref)
                 / gaussian_pure_purity(**ref))
    ax.plot(s["display"], shape_mc, "o", color=C["blue"], ms=3.5,
            label="sinc model (MC seed mean)")
    ax.plot(jgrid / 1e12, shape_ana, "--", c="k",
            label="analytic shape $g(J)$")
    dev = np.max(np.abs(
        shape_mc - gaussian_jitter_purity(s["value"], **ref)
        / gaussian_pure_purity(**ref)))
    ax.annotate(f"max deviation {dev:.3f}", (0.45, 0.75),
                xycoords="axes fraction", fontsize=7.5)
    ax.set_xlabel("center jitter rms (THz)")
    ax.set_ylabel("normalized purity $P(J)/P(0)$")
    ax.set_title("(b) sinc model: shape transfer")
    ax.legend(fontsize=7)
    save(fig, "fig6_validation")


# --------------------------------------------------------------- figure 7
def fig7_scaling():
    """Jitter budgets vs 1/L: three crystal lengths + fit through origin.

    Reads paper/tables/budget.json — run make_budget.py first.
    """
    import json
    with open(os.path.join(HERE, "tables", "budget.json"),
              encoding="utf-8") as f:
        budget = json.load(f)
    sc = budget["scaling_1_over_L"]
    inv_l_mm = np.array(sc["inv_L_per_m"]) / 1e3   # 1/mm

    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    for drop, color, marker in [("1%", C["gold"], "o"),
                                ("5%", C["teal"], "s"),
                                ("10%", C["blue"], "^")]:
        pts = np.array(sc["budgets_ghz"][drop]["points"], float)
        slope = sc["budgets_ghz"][drop]["fit_slope_ghz_m"]  # GHz per (1/m)
        xfit = np.linspace(0, inv_l_mm.max() * 1.05, 50)
        # x axis is 1/L in 1/mm = 1e3 * (1/m), so budget = slope * 1e3 * x
        ax.plot(xfit, slope * 1e3 * xfit, "--", color=color, alpha=0.6)
        ax.plot(inv_l_mm, pts, marker, color=color,
                label=f"{drop} budget")
    ax.set_xlabel("inverse crystal length $1/L$ (mm$^{-1}$)")
    ax.set_ylabel("jitter budget (GHz rms)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, title="fit: through origin", title_fontsize=7)
    save(fig, "fig7_scaling")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig1_jsa_anatomy()
    fig2_design()
    fig3_jitter()
    fig4_multimode()
    fig5_rin()
    fig6_validation()
    fig7_scaling()
    print("all figures written to paper/figures/")
