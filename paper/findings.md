# Findings dossier — single source of truth for the paper's numbers

Auto-compiled from results/paper/ (every dataset reproducible via
`qsource run configs/paper/<study>.yaml`). Verify any claim against
`paper/tables/budget.json`, `kconv.json`, and the CSVs under
`results/paper/<study>/`. Updated for the referee-2 revision round.

## System under study

- SPDC photon-pair source, first-order GVM model:
  JSA(vs, vi) = alpha_pump(vs+vi) x phi(vs, vi);
  phi = sinc(Delta_k L/2) (uniform crystal) or exp(-gamma (Delta_k L/2)^2)
  with gamma = 0.193 (Gaussian phase matching, amplitude-FWHM-matched).
- Reference: L = 10 mm, kappa_s = +0.207e-9 s/m, kappa_i = -0.318e-9 s/m
  (PPKTP-like, 775 -> 1550 nm), PULSED pump, transform-limited Gaussian,
  sigma = 0.164 THz (amplitude std, ordinary Hz) ~ ps-class pulses.
- Pure states: P via SVD Schmidt decomposition. Imperfect pump: MC
  ensemble rho = sum w_k A_k A_k^dagger, P = Tr(rho^2)/Tr(rho)^2.
  NOTE: rho keeps the IDLER (A@A^H traces the signal grid index).
- Monte-Carlo methodology (REVISED): S = 16 seeds (7..22) per sweep,
  independent stream per (seed, point) via default_rng([seed, i]).
  The earlier seed+i scheme made seed replicas CORRELATED (seed 8 point i
  == seed 7 point i+1); it was fixed and all numbers regenerated.
- K = 120 realizations per (seed, point); K = 240 for jitter_fine;
  K = 1000 (8 seeds) for RIN.

## Closed-form external validation (qsource/analytic.py, tests 10-14)

For the double-Gaussian JSA (Gaussian pump x Gaussian phase matching):

    P(J) = P0 / sqrt(1 + 4 c (2 pi J)^2),   J = center jitter rms (Hz)
    P0 = sqrt(1 - C^2/(A B));  A,B,C from sigma, gamma, a_j = kappa_j L/2
    c = 1 / (2 sigma^2 + 1/(gamma a_s^2))   (kept-idler convention)

- Reference values: P0 = 0.981752, c = 1.4367e-25 s^2.
- SVD matches P0 to < 1e-4 (test); deterministic Gauss-Hermite mixture
  matches P(J) to < 1e-3 (test); full MC matches within 0.05 including
  the finite-K bias E[P_MC] = P + (P0-P)/K (test).
- Budgets from the closed form vs the MC sweep (jitter_gauss):
  predicted 29.9 / 69.0 / 101.7 GHz -> measured 30 +- 0.3 / 68.5 +- 0.8
  / 102 +- 1 GHz. EXACT agreement within error bars.
- The sinc model's NORMALIZED decay P(J)/P(0) tracks the analytic shape
  g(J) with max deviation 0.041 over the dense sweep (fig6b).
- Gaussian-PM ideal purity 0.9818 != sinc 0.8100 (sinc side lobes carry
  entanglement): the shape comparison is normalized, never absolute.
- Exact-separability bandwidth of the Gaussian model: 0.1997 THz (P0=1).

## Imperfection budget (Table I; mean +- SEM over per-seed crossings)

Ideal purity of the base source: P = 0.8100 (10 mm), 0.8183 (5 mm),
0.8061 (20 mm at 0.082 THz), 0.9818 (10 mm Gaussian PM). The small
spread between lengths is the fixed-window truncation effect (see
numerics note), not physics — the first-order JSA is invariant under
L -> L/2, sigma -> 2 sigma.

| row | 1% | 5% | 10% |
|---|---|---|---|
| jitter, 20 mm (GHz rms) | 13.5 +- 0.2 | 31.3 +- 0.4 | 45.3 +- 0.6 |
| jitter, 10 mm (GHz rms) | 27.7 +- 0.3 | 62.7 +- 0.7 | 91 +- 1 |
| jitter, 5 mm (GHz rms) | 54.1 +- 0.9 | 125 +- 1 | 181 +- 3 |
| jitter, 10 mm Gaussian PM | 30 +- 0.3 | 68.5 +- 0.8 | 102 +- 1 |
| spacing, 3-mode comb (GHz) | 29.6 +- 0.1 | 75.9 +- 0.5 | 112.1 +- 0.7 |
| additional side modes @0.25 THz | 0 | 0 | 0 |
| bandwidth window (THz) | 0.159-0.202 | 0.131-0.244 | 0.114-0.281 |
| RIN on boost (% rms) | 9.93 +- 0.08 | 22.3 +- 0.2 | beyond sweep |

- The 1% cell of the 10 mm row comes from the dedicated jitter_fine
  study (K = 240, 5 GHz steps around the crossing); all three honest-
  quoting criteria hold for every quoted cell (all seeds cross; CI clear
  of zero; drop resolved at 2 sigma).
- 1/L LAW (fig7): 5% budgets 31.3 / 62.7 / 125 GHz at L = 20/10/5 mm =
  ratios 1 : 2.00 : 4.00 (from the unrounded means) — linear fit
  through the origin to within SEM.
  Same for 1% and 10% columns.
- K-convergence (kconv.json): |b(K240) - b(K120)| < 2 SEM at every drop
  level -> K = 120 accepted; budgets are not moved by finite-K bias.
- Wavelength equivalent: 27.7 GHz at 775 nm = 55.5 pm
  (delta_lambda = lambda^2 delta_nu / c; 1 nm = 499.1 GHz at 775 nm).

## Other validated results (unchanged physics, regenerated data)

- Bandwidth optimum: broad flat top, P_max = 0.8106 at 0.178 THz on the
  log grid; reference point 0.164 THz -> 0.8100 (n = 256, 4 THz window).
  Near log-symmetric fall-off around the peak.
- Length matching at 0.164 THz: optimum 10-12 mm; P = 0.286 at 2 mm.
- Narrowband anchor: 5 GHz pump -> P = 0.044, K = 22.6 (CW limit) on
  grids that resolve the ~5 GHz pump ridge (0.0443/22.59 on the 4 THz
  reference window at n >= 2048; the n = 512, 2 THz test grid with its
  7.8 GHz pixel gives 0.0425/23.51).
- Modes ladder (0.25 THz spacing): 0.810, 0.671, 0.543, 0.445, 0.377,
  0.330, 0.292 for 1..7 lines — a single extra EQUAL-POWER mode costs
  17%, more than the whole 10% budget.
- RIN: purity exactly invariant (< 1e-9); boost follows 1 + r^2
  (1.0221 measured vs 1.0225 theory at 15%; 1.0940 at 30% vs 1.09).
  The w >= 0 clipping shifts the expected boost by only ~1e-4 even at
  30% (exact clipped value 1.0899); the +-0.004 residual scatter about
  1 + r^2 is finite-sample noise of the K = 1e3, 8-seed estimator.

## Regime of validity (quasi-static ensemble)

- Pump coherence time tau_coh ~ 1/(2 pi sigma) ~ 0.97 ps at 0.164 THz.
- The ensemble picture holds for noise SLOWER than tau_coh (each pair
  sees a static center) and FASTER than the integration time (ensemble
  fully sampled): for ~80 MHz rep rate and seconds of integration this
  spans noise frequencies from ~Hz to ~tens of GHz — thermal drift,
  acoustics, mount creep. Sub-ps noise acts coherently on each pure JSA
  and is OUTSIDE the model.
- Comb model asserts MUTUALLY INCOHERENT modes (free-running multimode
  diode: random phases, mode partition). A mode-locked comb is
  phase-coherent — the JSA is a coherent SUM (pure state), not a
  mixture; its ~80 MHz teeth are anyway unresolved by the JSA.
- RIN is pulse-energy noise (pulsed pump).

## Pair-statistics context (CAR worked example)

With mean pair number mu per pulse: true heralds ~ mu, accidentals ~
mu^2, CAR ~ 1/mu. RIN multiplies the mu^2 term by the boost:
CAR -> CAR / boost, equivalent to running at mu_eff = mu x boost.
Example mu = 0.05: 30% RIN (boost 1.0940) costs 8.6% of CAR
(mu_eff = 0.0547); at the +1% budget (9.9% RIN) the cost is 1%.

## Common-mode drift (shared pump) — EXACT model statement

V_HOM = P holds for two INDEPENDENT identical sources. If both sources
share one pump laser, each trial's two JSAs shift by the SAME delta;
since a pump shift is a rigid JSA translation (validated), each
realization's reduced state has the ideal purity, and
V_common = E_delta[Tr(rho_delta^2)] = P0 exactly — jitter cancels
completely in single-laser common-mode architectures. The reported
budgets are therefore CONSERVATIVE for single-laser multiplexing.

## Reading a datasheet (real products, web-verified 2026-07-05)

Translation layer: src/qsource/datasheet.py (7 pinned tests).
Key formulas: sigma = FWHM_intensity/1.6651; dnu = c dlambda/lambda^2
(499.1 GHz/nm at 775 nm); rms = pp/(2 sqrt 3) [uniform];
r = sqrt(10^(RIN_dB/10) B); FSR = c/(2 n L); Lorentzian FWHM -> HWHM
bound only (fast phase noise is NOT quasi-static).

- Thorlabs L785P090 (free-running FP, 785 nm): publishes NONE of the
  budget-relevant lines (no spectral width, mode spacing, dlambda/dT,
  RIN, SMSR); center wavelength 775-795 nm is unit selection. Verdict:
  fails by construction (multimode) and cannot even be budgeted from
  its datasheet.
- eagleyard EYP-DFB-0780 (DFB, 780.24 nm): linewidth max 1 MHz ->
  quasi-static bound 0.5 MHz = 5e-5 of the 1% budget (negligible);
  dlambda/dT = 0.06 nm/K -> 29.9 GHz/K -> 1% budget == holding
  +-1.6 K; dlambda/dI = 0.003 nm/mA -> 1.5 GHz/mA -> +-32 mA;
  SMSR >= 30 dB -> side weight 1e-3, negligible; RIN not published.
  Verdict: spectral-stability columns exemplary, but CW linewidth sits
  five orders below the bandwidth window -> P -> 0.04 (CW limit):
  wrong class for pure heralds, ideal for center stability.
- Spectra-Physics Tsunami ps (775 nm, 80 MHz): 2 ps sech^2 ->
  sigma = 0.095 THz -> P = 0.66 at 10 mm (binding line!), recovered by
  re-matching the crystal to L ~ 17 mm; noise < 0.2% rms -> boost
  excess 4e-6 (negligible); 80 MHz comb unresolved by the JSA.
  Verdict: the natural class; bandwidth is the line to check.

## Numerical robustness

- Grid: n = 256 converged < 1e-4 vs n = 768; n128/n192 replicas of the
  jitter sweep confirm discretization ~ 2e-5, far below MC error.
- IDEAL PURITY IS WINDOW-DEPENDENT at the ~0.5% level (sinc tails):
  P = 0.8100/0.8074/0.8061 at 4/6/8 THz half-window for the same
  operating point. Budgets are RELATIVE to the same-window ideal, so
  this largely cancels. The small ideal-purity spread across the
  length-matched sweeps (0.8061 / 0.8100 / 0.8183 at 20/10/5 mm) is the
  same fixed-window truncation effect, not physics.
- Validation suite: 14 physics tests + 9 program tests + 5 datasheet
  tests = 28 tests, all green.

## Figures

- fig1_jsa_anatomy, fig2_design, fig4_multimode, fig5_rin: as before.
- fig3_jitter: now THREE lengths (20/10/5 mm) with +-1 sigma seed bands.
- fig6_validation (NEW): (a) Gaussian-PM MC seed band ON the closed
  form; (b) sinc normalized decay vs g(J), max deviation 0.041.
- fig7_scaling (NEW): jitter budgets vs 1/L with through-origin fits.

## Repository

Code: https://github.com/ABD252/qsource-sim (MIT). A Zenodo DOI will
accompany submission.
