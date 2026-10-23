"""Train the from-scratch PPO agent, optionally cross-check with sb3-contrib MaskablePPO
(same invalid-action masks, same learner-side reward scale), then evaluate both against
the heuristics on the same seeds.

    python scripts/train_ppo.py --steps 100000 --sb3
    python scripts/train_ppo.py --steps 100000 --seed 1 --w-tardiness 2.5 --out results/runs/wt2.5_s1 --models models/runs/wt2.5_s1
"""
from __future__ import annotations

import argparse
import sys
import json
from dataclasses import replace
from pathlib import Path

import torch

from flowtwin.config import DEFAULT
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, save_results
from flowtwin.heuristics import HEURISTICS, TABLE_RULES
from flowtwin.ppo import PPOConfig, policy_from_model, train_ppo
from flowtwin.rule_actions import RuleSelectionEnv, line_policy


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30, help="evaluation episodes")
    ap.add_argument("--sb3", action="store_true", help="also train sb3-contrib MaskablePPO for cross-checking")
    ap.add_argument("--w-tardiness", type=float, default=DEFAULT.w_tardiness, help="reward weight per minute of lateness")
    ap.add_argument("--dense-tardiness", action="store_true",
                    help="train on lateness as it accrues (same episode totals); evaluation keeps the standard reward")
    ap.add_argument("--rule-actions", action="store_true",
                    help="each action picks a dispatching rule (FIFO/SPT/EDD) and a speed instead of a product")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models", help="directory for the trained checkpoints")
    args = ap.parse_args()
    if args.sb3 and args.rule_actions:
        ap.error("--rule-actions is not wired into the --sb3 cross-check")
    if args.sb3:
        from sb3_contrib import MaskablePPO   # fail before training if the optional dependency is missing
    line_cfg = replace(DEFAULT, w_tardiness=args.w_tardiness)
    train_cfg = replace(line_cfg, dense_tardiness=args.dense_tardiness)   # learning signal only
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models = Path(args.models); models.mkdir(parents=True, exist_ok=True)

    env = FlowTwinEnv(train_cfg)
    if args.rule_actions:
        env = RuleSelectionEnv(env)
    cfg = PPOConfig(total_steps=args.steps, seed=args.seed)
    model, history = train_ppo(env, cfg)
    torch.save(model.state_dict(), models / "ppo.pt")
    (out / "ppo_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    results = {name: evaluate(HEURISTICS[name], line_cfg, args.episodes) for name in TABLE_RULES}
    agent = line_policy(policy_from_model(model, masked=False)) if args.rule_actions else policy_from_model(model)
    results["ppo (ours)"] = evaluate(agent, line_cfg, args.episodes)

    if args.sb3:
        from gymnasium.wrappers import TransformReward
        train_env = TransformReward(FlowTwinEnv(train_cfg), lambda r: r * cfg.reward_scale)   # learning only
        sb3 = MaskablePPO("MlpPolicy", train_env, seed=args.seed, n_steps=512, batch_size=64, verbose=0)
        sb3.learn(total_timesteps=args.steps)
        sb3.save(models / "ppo_sb3")
        results["maskable ppo (sb3)"] = evaluate(
            lambda e, o: int(sb3.predict(o, deterministic=True, action_masks=e.action_masks())[0]), line_cfg, args.episodes)

    save_results(results, out, "ppo_vs_heuristics", refs=("edd", "edd_low"))
    print((out / "ppo_vs_heuristics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
