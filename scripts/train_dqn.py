"""Train the Double-DQN baseline and evaluate it against the heuristics.

    python scripts/train_dqn.py --steps 100000
    python scripts/train_dqn.py --steps 100000 --seed 1 --w-tardiness 2.5 --out results/runs/wt2.5_s1 --models models/runs/wt2.5_s1
"""
from __future__ import annotations

import argparse
import sys
import json
from dataclasses import replace
from pathlib import Path

import torch

from flowtwin.config import DEFAULT
from flowtwin.dqn import DQNConfig, policy_from_qnet, train_dqn
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, save_results
from flowtwin.heuristics import HEURISTICS, TABLE_RULES


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--w-tardiness", type=float, default=DEFAULT.w_tardiness, help="reward weight per minute of lateness")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models", help="directory for the trained checkpoint")
    args = ap.parse_args()
    line_cfg = replace(DEFAULT, w_tardiness=args.w_tardiness)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models = Path(args.models); models.mkdir(parents=True, exist_ok=True)

    q, history = train_dqn(FlowTwinEnv(line_cfg), DQNConfig(total_steps=args.steps, seed=args.seed))
    torch.save(q.state_dict(), models / "dqn.pt")
    (out / "dqn_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    results = {name: evaluate(HEURISTICS[name], line_cfg, args.episodes) for name in TABLE_RULES}
    results["double-dqn"] = evaluate(policy_from_qnet(q), line_cfg, args.episodes)
    save_results(results, out, "dqn_vs_heuristics", refs=("edd", "edd_low"))
    print((out / "dqn_vs_heuristics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
