"""Evaluate the dispatching heuristics over seeded episodes and write results/baselines.{md,csv,json}.

    python scripts/run_baselines.py --episodes 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowtwin.config import DEFAULT
from flowtwin.evaluate import evaluate, save_results, summarize
from flowtwin.heuristics import HEURISTICS


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    results = {name: evaluate(pol, DEFAULT, args.episodes, args.seed0) for name, pol in HEURISTICS.items()}
    print(summarize(results))
    save_results(results, Path(args.out), "baselines")
    print(f"\nsaved to {args.out}/baselines.{{md,csv,json}}")


if __name__ == "__main__":
    main()
