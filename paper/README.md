# paper/ — the arXiv preprint and its fully reproducible pipeline

Everything in the paper regenerates from configs with four commands:

```bash
python paper/make_configs.py         # 1. write configs/paper/*.yaml (15 studies)
python paper/run_studies.py          # 2. run them -> results/paper/<study>/<ts>/
python paper/make_budget.py          # 3. budgets + CIs -> paper/tables/budget.*
python paper/make_figures.py         # 4. figures (fig7 reads budget.json)
python paper/make_datasheet_table.py # 5. real-datasheet translation table
tectonic main.tex                    # 6. compile the PDF (from paper/)
```

| file | role |
|---|---|
| `main.tex` + `sections/*.tex` | the manuscript (REVTeX 4-2) |
| `refs.bib` | bibliography (web-verified) |
| `findings.md` | single source of truth for every number in the text |
| `figures/` | paper figures, PDF (LaTeX) + PNG (viewing) |
| `tables/budget.*` | auto-generated imperfection-budget table |
| `summary_ar.md` | ملخص عربي لمناقشة النتائج |

Rules: no number appears in the text unless it exists in `findings.md` /
`tables/budget.json`, and those in turn come only from `results/paper/`
runs driven by the YAML configs (hard rule 5 of CLAUDE.md).
