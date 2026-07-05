"""demo_imperfections.py — 'Take This' project: 4 real-laser imperfections.

Numbers for each imperfection come from FREE manufacturer datasheets:
  jitter/drift: wavelength stability specs (pm -> Hz)
  RIN:          intensity-noise spec (% rms or dB/Hz)
  multimode:    longitudinal-mode spacing = c / (2 * cavity length)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "figures"), exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from qsource import Crystal
from qsource.realistic import LaserSpec, heralded_purity_mc

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
crystal = Crystal()
BW = 0.164e12          # keep pump at the ideal optimum; perturb around it
N, K = 192, 40
P_IDEAL = 0.81

fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), constrained_layout=True)

# (a) center jitter / drift ------------------------------------------------
jits = np.linspace(0, 0.5e12, 9)
p = [heralded_purity_mc(LaserSpec(BW, center_jitter_hz=j), crystal, n=N,
                        k_realizations=K)["purity"] for j in jits]
ax = axes[0, 0]
ax.plot(jits/1e12, p, "o-", color="#d4a24e")
ax.axhline(P_IDEAL, ls=":", c="green", label="ideal")
ax.set_xlabel("Center jitter / drift rms (THz)"); ax.set_ylabel("Heralded purity P")
ax.set_title("(a) Spectral drift — the silent killer"); ax.legend(); ax.grid(alpha=.3)

# (b) RIN — attacks a DIFFERENT metric --------------------------------------
rins = np.linspace(0, 0.30, 9)
boost = []
for r in rins:
    w = heralded_purity_mc(LaserSpec(BW, rin_frac=r), crystal, n=64,
                           k_realizations=400)["weights"]
    boost.append(np.mean(w**2) / np.mean(w)**2)   # double-pair enhancement
ax = axes[0, 1]
ax.plot(rins*100, boost, "s-", color="#e06377")
ax.axhline(1.0, ls=":", c="green", label="stable power")
ax.set_xlabel("RIN (% rms)"); ax.set_ylabel(r"Double-pair boost  $\langle w^2\rangle/\langle w\rangle^2$")
ax.set_title("(b) RIN — purity untouched, statistics poisoned"); ax.legend(); ax.grid(alpha=.3)

# (c) number of longitudinal modes (Abdullah's 'gaps') ----------------------
modes = [1, 2, 3, 4, 5, 7]
p = [heralded_purity_mc(LaserSpec(BW, n_modes=m, mode_spacing_hz=0.25e12),
                        crystal, n=N, k_realizations=K)["purity"] for m in modes]
ax = axes[1, 0]
ax.plot(modes, p, "D-", color="#4ecdc4")
ax.axhline(P_IDEAL, ls=":", c="green", label="single mode")
ax.set_xlabel("Number of laser modes"); ax.set_ylabel("Heralded purity P")
ax.set_title("(c) Multimode comb (spacing 0.25 THz)"); ax.legend(); ax.grid(alpha=.3)

# (d) mode spacing at fixed 3 modes -----------------------------------------
spacings = np.linspace(0.02e12, 0.6e12, 9)
p = [heralded_purity_mc(LaserSpec(BW, n_modes=3, mode_spacing_hz=s),
                        crystal, n=N, k_realizations=K)["purity"] for s in spacings]
ax = axes[1, 1]
ax.plot(spacings/1e12, p, "^-", color="#8fa8d8")
ax.axhline(P_IDEAL, ls=":", c="green", label="single mode")
ax.set_xlabel("Mode spacing (THz)"); ax.set_ylabel("Heralded purity P")
ax.set_title("(d) 3-mode laser: how far apart the 'gaps' are"); ax.legend(); ax.grid(alpha=.3)

fig.suptitle("Take-This Project — four real-laser imperfections vs entanglement-source quality",
             fontsize=13)
fig.savefig(os.path.join(FIG, "imperfections_grid.png"), dpi=150)
print("saved imperfections_grid.png")
for name, arr in [("jitter", jits), ("rin", rins), ("modes", modes), ("spacing", spacings)]:
    print(f"swept {name}: {len(list(arr))} points")
