"""Evaluate the dispatching heuristics over seeded episodes and write results/baselines.{md,csv,json}.

    python scripts/run_baselines.py --episodes 30
    python scripts/run_baselines.py --w-tardiness 2.5 --out results/runs/wt2.5_s0
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from flowtwin.config import DEFAULT
from flowtwin.evaluate import evaluate, paired_summary, save_results, summarize
from flowtwin.heuristics import HEURISTICS, TABLE_RULES


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--w-tardiness", type=float, default=DEFAULT.w_tardiness, help="reward weight per minute of lateness")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = replace(DEFAULT, w_tardiness=args.w_tardiness)

    results = {name: evaluate(HEURISTICS[name], cfg, args.episodes, args.seed0) for name in TABLE_RULES + ("random",)}
    print(summarize(results) + "\n\n" + paired_summary(results, "edd"))
    save_results(results, Path(args.out), "baselines", refs=("edd",))
    print(f"\nsaved to {args.out}/baselines.{{md,csv,json}}")


if __name__ == "__main__":
    main()
