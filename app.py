"""app.py — Interactive dashboard for qsource-sim (Streamlit).

Run with:  streamlit run app.py

Live exploration of the same physics the CLI sweeps over: move a slider,
watch the JSA heatmap and the heralded-purity / Schmidt-number / double-
pair-boost readouts respond. ZERO physics lives in this file — every
number on screen comes from the validated core (compute_jsa,
schmidt_analysis, heralded_purity_mc). The UI only converts units at the
boundary (THz/mm/% on screen, SI into the core) and draws.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from qsource import Crystal, GaussianPump, compute_jsa, schmidt_analysis
from qsource.realistic import LaserSpec, heralded_purity_mc

st.set_page_config(page_title="qsource-sim dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cached wrappers around the physics core (inputs already in SI units).
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_ideal_jsa(bw_hz: float, length_m: float, kappa_s: float,
                     kappa_i: float, span_hz: float, n: int):
    """Ideal (imperfection-free) JSA + its SVD purity for the heatmap."""
    jsa = compute_jsa(GaussianPump(bw_hz), Crystal(length_m, kappa_s, kappa_i),
                      span_hz=span_hz, n=n)
    res = schmidt_analysis(jsa.amplitude)
    return jsa.nu_s, jsa.intensity, res.purity, res.schmidt_number


@st.cache_data(show_spinner=False)
def cached_mc(bw_hz: float, jitter_hz: float, rin_frac: float, n_modes: int,
              spacing_hz: float, length_m: float, kappa_s: float,
              kappa_i: float, span_hz: float, n: int, k: int, seed: int):
    """Monte-Carlo heralded purity of the imperfect laser."""
    spec = LaserSpec(bandwidth_hz=bw_hz, center_jitter_hz=jitter_hz,
                     rin_frac=rin_frac, n_modes=n_modes,
                     mode_spacing_hz=spacing_hz)
    # macOS BLAS emits spurious RuntimeWarnings on these complex matmuls;
    # results are validated by tests/test_core.py, so silence them here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        res = heralded_purity_mc(spec, Crystal(length_m, kappa_s, kappa_i),
                                 span_hz=span_hz, n=n, k_realizations=k,
                                 rng=np.random.default_rng(seed))
    w = res["weights"]
    boost = float(np.mean(w ** 2) / np.mean(w) ** 2)
    return res["purity"], res["schmidt_number"], boost


# ---------------------------------------------------------------------------
# Sidebar — user-facing units (THz / mm / %), converted to SI at the call.
# ---------------------------------------------------------------------------

st.sidebar.title("qsource-sim")
st.sidebar.caption("Pump-laser imperfections vs SPDC source quality")

st.sidebar.header("Pump")
bw_thz = st.sidebar.slider("Bandwidth (THz)", 0.01, 1.0, 0.164, 0.001,
                           format="%.3f")
jitter_thz = st.sidebar.slider("Center jitter / drift rms (THz)",
                               0.0, 0.5, 0.0, 0.005, format="%.3f")
rin_pct = st.sidebar.slider("RIN (% rms)", 0.0, 30.0, 0.0, 0.5)
n_modes = st.sidebar.slider("Longitudinal modes", 1, 7, 1)
spacing_thz = st.sidebar.slider("Mode spacing (THz)", 0.0, 0.6, 0.25, 0.01,
                                disabled=(n_modes == 1))

st.sidebar.header("Crystal")
length_mm = st.sidebar.slider("Length (mm)", 1.0, 30.0, 10.0, 0.5)
kappa_s = st.sidebar.number_input("kappa_s (s/m)", value=0.207e-9,
                                  format="%.3e")
kappa_i = st.sidebar.number_input("kappa_i (s/m)", value=-0.318e-9,
                                  format="%.3e")

st.sidebar.header("Numerics")
n_grid = st.sidebar.select_slider("Grid n", options=[96, 128, 192, 256],
                                  value=128)
k_real = st.sidebar.select_slider("MC realizations K",
                                  options=[20, 40, 80, 160], value=40)
span_thz = st.sidebar.select_slider("Span (THz)", options=[2, 4, 8], value=4)
seed = int(st.sidebar.number_input("Seed", value=7, step=1))

# --- boundary: everything below this line is SI ---------------------------
bw_hz = bw_thz * 1e12
jitter_hz = jitter_thz * 1e12
rin_frac = rin_pct / 100.0
spacing_hz = spacing_thz * 1e12
length_m = length_mm * 1e-3
span_hz = span_thz * 1e12

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

col_map, col_metrics = st.columns([3, 2])

nu, jsi, ideal_p, ideal_k = cached_ideal_jsa(
    bw_hz, length_m, kappa_s, kappa_i, span_hz, n_grid)

with col_map:
    st.subheader("Joint Spectral Intensity |JSA|²  (ideal pump)")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    extent = [nu[0] / (2 * np.pi * 1e12), nu[-1] / (2 * np.pi * 1e12)] * 2
    ax.imshow(jsi, origin="lower", extent=extent, cmap="magma",
              aspect="equal")
    ax.set_xlabel("Signal detuning (THz)")
    ax.set_ylabel("Idler detuning (THz)")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "The heatmap shows the imperfection-free JSA for the current pump "
        "bandwidth and crystal — the −45° pump band times the tilted sinc "
        "ridge. Jitter / RIN / multimode act on the ENSEMBLE (mixed state), "
        "so their effect appears in the metrics, not in a single-shot JSA."
    )

with st.spinner("Monte-Carlo mixed-state purity ..."):
    mc_p, mc_k, boost = cached_mc(
        bw_hz, jitter_hz, rin_frac, int(n_modes), spacing_hz,
        length_m, kappa_s, kappa_i, span_hz, n_grid, k_real, seed)

with col_metrics:
    st.subheader("Source quality")
    m1, m2 = st.columns(2)
    m1.metric("Heralded purity P", f"{mc_p:.4f}",
              delta=f"{mc_p - ideal_p:+.4f} vs ideal",
              delta_color="normal" if abs(mc_p - ideal_p) > 1e-6 else "off")
    m2.metric("Schmidt number K", f"{mc_k:.3f}")
    m3, m4 = st.columns(2)
    m3.metric("Ideal (SVD) purity", f"{ideal_p:.4f}",
              help="Pure-state purity of the imperfection-free JSA above")
    m4.metric("Double-pair boost ⟨w²⟩/⟨w⟩²", f"{boost:.4f}",
              delta=f"{(boost - 1) * 100:+.2f}% vs stable power",
              delta_color="inverse" if boost > 1 + 1e-6 else "off",
              help="RIN reweights pair generation: >1 means accidental "
                   "double pairs are enhanced even though purity is not.")

    st.divider()
    st.markdown(
        f"""
**Reading the numbers**

- Ideal Schmidt number: **K = {ideal_k:.3f}** (K = 1 would be a fully
  separable, single-mode source).
- Spectral imperfections (jitter, multimode) push **P down** by mixing
  differently-centred JSAs into the heralded state.
- RIN leaves P untouched but raises the **double-pair boost** — a
  different metric gets damaged (statistical, not spectral).
        """
    )
    if n_modes > 1:
        st.info(f"Multimode comb: {n_modes} lines, "
                f"{spacing_thz:.2f} THz apart — each line adds a "
                "distinguishable spectral mode to the mixture.")
