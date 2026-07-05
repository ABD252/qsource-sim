"""Week-2 demo — your first original figures.

Run:  python notebooks/demo_week2.py
Produces figures/ with:
  1. jsa_anatomy.png   — pump term x crystal term = JSA (the key intuition)
  2. purity_sweep.png  — purity vs pump bandwidth (your first result curve)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "figures"), exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from qsource import GaussianPump, Crystal, compute_jsa, schmidt_analysis

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
THz = 1e12  # plot axes in THz for readability

# ---------------------------------------------------------------- fig 1
pump = GaussianPump(bandwidth_hz=0.164e12)   # near the optimum found by tests
crystal = Crystal()
span = 2e12
axis = np.linspace(-2*np.pi*span, 2*np.pi*span, 512)
S, I = np.meshgrid(axis, axis)
ext = [-span/THz, span/THz, -span/THz, span/THz]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
for ax, data, title in [
    (axes[0], np.abs(pump.envelope(S, I))**2,            "Pump  |α(ν$_s$+ν$_i$)|²\n(energy conservation)"),
    (axes[1], np.abs(crystal.phase_matching(S, I))**2,   "Crystal  |φ(ν$_s$,ν$_i$)|²\n(momentum conservation)"),
    (axes[2], compute_jsa(pump, crystal, span_hz=span).intensity, "JSA = product\n(the two-photon state)"),
]:
    im = ax.imshow(data, origin="lower", extent=ext, cmap="magma", aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("ν$_s$ (THz)"); ax.set_ylabel("ν$_i$ (THz)")
res = schmidt_analysis(compute_jsa(pump, crystal, span_hz=span).amplitude)
fig.suptitle(f"Anatomy of the JSA — purity P = {res.purity:.3f}, Schmidt K = {res.schmidt_number:.2f}",
             fontsize=13)
fig.savefig(os.path.join(FIG, "jsa_anatomy.png"), dpi=150)
print(f"saved jsa_anatomy.png   (P={res.purity:.3f})")

# ---------------------------------------------------------------- fig 2
bws = np.logspace(10, 13, 40)
purities = [schmidt_analysis(compute_jsa(GaussianPump(b), crystal, span_hz=8e12).amplitude).purity
            for b in bws]
best = int(np.argmax(purities))

plt.figure(figsize=(7.5, 4.6))
plt.semilogx(bws/THz, purities, "o-", ms=4)
plt.axvline(bws[best]/THz, ls="--", c="gray",
            label=f"optimum ≈ {bws[best]/THz:.2f} THz  (P={purities[best]:.2f})")
plt.axhline(1.0, ls=":", c="green", label="ideal separable state (P=1)")
plt.xlabel("Pump bandwidth (THz)"); plt.ylabel("Spectral purity P")
plt.title("Purity vs pump bandwidth — PPKTP-like crystal, L=10 mm")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "purity_sweep.png"), dpi=150)
print(f"saved purity_sweep.png  (optimum at {bws[best]/THz:.2f} THz)")
