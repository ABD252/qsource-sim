"""cli.py — `qsource` command-line interface.

    qsource run config.yaml            # one sweep -> results/<timestamp>/
    qsource compare cfg1.yaml cfg2.yaml  # overlay two sweeps on one figure

Zero physics here: parse args, load configs, call the study runner,
write the results folder, print where everything went.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_config, SWEEP_PARAMETERS
from .report import write_results
from .study import run_study


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    label = os.path.splitext(os.path.basename(args.config))[0]
    print(f"Running sweep '{cfg.sweep.parameter}' "
          f"({len(cfg.sweep.values)} points, grid n={cfg.grid.n}, "
          f"K={cfg.grid.realizations}) ...")
    result = run_study(cfg, label=label)
    folder = write_results([cfg], [result], root=args.results_dir)

    print(f"Ideal purity (no imperfections): {result.ideal_purity:.4f}")
    for vd, p in zip(result.axis_values, result.purities):
        print(f"  {cfg.sweep.parameter} = {vd:.6g} {result.axis_unit:<6} "
              f"->  P = {p:.4f}")
    print(f"\nResults written to: {folder}")
    return 0


def _cmd_compare(args) -> int:
    cfg1, cfg2 = load_config(args.config1), load_config(args.config2)
    if cfg1.sweep.parameter != cfg2.sweep.parameter:
        print(f"error: cannot overlay different sweep parameters "
              f"({cfg1.sweep.parameter!r} vs {cfg2.sweep.parameter!r})",
              file=sys.stderr)
        return 2

    results = []
    for cfg, path in [(cfg1, args.config1), (cfg2, args.config2)]:
        label = os.path.splitext(os.path.basename(path))[0]
        print(f"Running '{label}' ({len(cfg.sweep.values)} points) ...")
        results.append(run_study(cfg, label=label))

    folder = write_results([cfg1, cfg2], results, root=args.results_dir)
    for r in results:
        print(f"  {r.label}: ideal P = {r.ideal_purity:.4f}, "
              f"sweep P in [{min(r.purities):.4f}, {max(r.purities):.4f}]")
    print(f"\nComparison written to: {folder}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="qsource",
        description="SPDC entanglement-source studies driven by YAML configs "
                    f"(sweepable: {', '.join(SWEEP_PARAMETERS)})",
    )
    parser.add_argument("--results-dir", default="results",
                        help="root folder for outputs (default: results)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute one study config")
    p_run.add_argument("config", help="path to a study YAML")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare",
                           help="overlay two sweeps of the same parameter")
    p_cmp.add_argument("config1")
    p_cmp.add_argument("config2")
    p_cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
