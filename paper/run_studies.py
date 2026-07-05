"""run_studies.py — Execute every paper study config, 3 at a time.

Each study lands in results/paper/<study>/<timestamp>/ with its config
copy, CSV, markdown table and figure. Re-running overwrites nothing —
the assembly scripts (make_figures.py / make_budget.py) always read the
LATEST timestamp per study.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "paper", "*.yaml")))
MAX_PARALLEL = 3


def main() -> int:
    if not CONFIGS:
        print("no configs found — run paper/make_configs.py first")
        return 1
    pending = list(CONFIGS)
    running: list[tuple[str, subprocess.Popen]] = []
    failed = []
    t0 = time.time()

    while pending or running:
        while pending and len(running) < MAX_PARALLEL:
            cfg = pending.pop(0)
            name = os.path.splitext(os.path.basename(cfg))[0]
            out_dir = os.path.join(ROOT, "results", "paper", name)
            log = open(os.path.join(HERE, f".{name}.log"), "w")
            p = subprocess.Popen(
                [sys.executable, "-m", "qsource.cli",
                 "--results-dir", out_dir, "run", cfg],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ,
                     "PYTHONPATH": os.path.join(ROOT, "src")},
            )
            running.append((name, p))
            print(f"[{time.time()-t0:6.0f}s] started  {name}")
        for name, p in running[:]:
            if p.poll() is not None:
                running.remove((name, p))
                status = "done" if p.returncode == 0 else f"FAILED ({p.returncode})"
                if p.returncode != 0:
                    failed.append(name)
                print(f"[{time.time()-t0:6.0f}s] {status:12s} {name}")
        time.sleep(2)

    print(f"\nall studies finished in {time.time()-t0:.0f}s; "
          f"{len(failed)} failed" + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
