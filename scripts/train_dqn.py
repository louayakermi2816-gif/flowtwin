"""Train the Double-DQN baseline and evaluate it against the heuristics.

    python scripts/train_dqn.py --steps 200000
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import torch

from flowtwin.config import DEFAULT
from flowtwin.dqn import DQNConfig, policy_from_qnet, train_dqn
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, save_results, summarize
from flowtwin.heuristics import HEURISTICS


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    q, history = train_dqn(FlowTwinEnv(DEFAULT), DQNConfig(total_steps=args.steps, seed=args.seed))
    torch.save(q.state_dict(), "models/dqn.pt")
    (out / "dqn_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    results = {name: evaluate(pol, DEFAULT, args.episodes) for name, pol in HEURISTICS.items() if name != "random"}
    results["double-dqn"] = evaluate(policy_from_qnet(q), DEFAULT, args.episodes)
    print(summarize(results))
    save_results(results, out, "dqn_vs_heuristics")


if __name__ == "__main__":
    main()
