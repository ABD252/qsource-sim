# qsource-sim

**SPDC entanglement-source simulation — how pump-laser imperfections degrade entanglement quality.**

محاكاة مصدر التشابك الفوتوني (SPDC): كيف تنتقل عيوب ليزر الضخ وتؤثر على جودة التشابك؟

Undergraduate research project — Abdullah Al-Baqami, Physics Dept., King Abdulaziz University.

## The one equation this repo is built on

```
JSA(νs, νi) = α_pump(νs + νi) × φ_crystal(νs, νi)
```

Energy conservation (pump) × momentum conservation (crystal) = the two-photon state.
Every pump imperfection enters through `α` in `pump.py`; everything downstream is untouched.

## Quick start

```bash
pip install -e .                 # installs the `qsource` command (MIT, pip-installable)
python tests/test_core.py        # verify the physics against known results
python notebooks/demo_week2.py   # generate the first figures into figures/
```

## Running studies (no code editing needed)

Everything is driven by a YAML config — change parameters there, never in `src/`:

```bash
qsource run configs/example.yaml                 # one sweep
qsource compare configs/example.yaml configs/example_short_crystal.yaml
```

Each run writes a self-contained folder `results/<timestamp>/` with the
figure (`sweep.png`), the table (`results.csv` + `results.md`) and a
verbatim copy of the config — so every figure is reproducible from the
config alone. Sweepable parameters: `center_jitter_hz`, `rin_frac`,
`n_modes`, `mode_spacing_hz`, `bandwidth_hz`, `length_m`
(see `configs/example.yaml` for the full schema).

## Interactive dashboard

```bash
pip install -e ".[dashboard]"    # adds streamlit
streamlit run app.py
```

Sliders for pump bandwidth, jitter, RIN, longitudinal modes, mode spacing
and crystal length — live JSA heatmap plus heralded purity / Schmidt
number / double-pair boost readouts. All numbers come from the same
validated core; the UI contains zero physics.

## Structure

```
src/qsource/
  pump.py      # pump spectral envelope  <- imperfections get injected HERE
  crystal.py   # sinc phase-matching (first-order GVM model)
  jsa.py       # JSA = pump x crystal, normalized
  metrics.py   # Schmidt/SVD -> purity, Schmidt number
  realistic.py # LaserSpec + Monte-Carlo mixed-state heralded purity
  config.py    # YAML schema + the ONLY user/SI unit boundary
  study.py     # sweep runner (glue, no physics)
  report.py    # figures + CSV/markdown + results/<timestamp>/ folders
  cli.py       # `qsource run` / `qsource compare`
configs/       # example study configs (YAML)
app.py         # Streamlit dashboard (UI only, zero physics)
tests/         # sanity tests encoding known physics
notebooks/     # runnable demos / experiments
```

## The paper

`paper/` holds the arXiv preprint (REVTeX 4-2) and its fully reproducible
pipeline — see [paper/README.md](paper/README.md). Four scripts regenerate
every dataset, figure and budget table from `configs/paper/*.yaml`;
`tectonic main.tex` builds the PDF. An Arabic summary for discussion lives
in `paper/summary_ar.md`.

## Roadmap

- [x] Core: ideal Gaussian pump, sinc phase matching, purity via SVD
- [x] Validation: narrowband→low purity; interior bandwidth optimum
- [x] Imperfection 2: RIN → pair-rate fluctuations → double-pair statistics
- [x] Imperfection 3: spectral drift / jitter (quasi-static mixed states)
- [x] Multimode comb structure (mode count + spacing)
- [x] The paper: the imperfection budget (`paper/`)
- [ ] Imperfection 1: excess linewidth (coherent vs incoherent — mixed states)
- [ ] Metrics: heralding efficiency, HOM visibility
- [ ] Sellmeier-based phase matching for real PPKTP/BBO

## License

MIT — open source from day one.
