"""Train the from-scratch PPO agent, optionally cross-check with sb3-contrib MaskablePPO
(same invalid-action masks, same learner-side reward scale), then evaluate both against
the heuristics on the same seeds.

    python scripts/train_ppo.py --steps 100000 --sb3
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import torch

from flowtwin.config import DEFAULT
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, save_results
from flowtwin.heuristics import HEURISTICS
from flowtwin.ppo import PPOConfig, policy_from_model, train_ppo


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30, help="evaluation episodes")
    ap.add_argument("--sb3", action="store_true", help="also train sb3-contrib MaskablePPO for cross-checking")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.sb3:
        from sb3_contrib import MaskablePPO   # fail before training if the optional dependency is missing
    out = Path(args.out); out.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    env = FlowTwinEnv(DEFAULT)
    cfg = PPOConfig(total_steps=args.steps, seed=args.seed)
    model, history = train_ppo(env, cfg)
    torch.save(model.state_dict(), "models/ppo.pt")
    (out / "ppo_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    results = {name: evaluate(pol, DEFAULT, args.episodes) for name, pol in HEURISTICS.items() if name != "random"}
    results["ppo (ours)"] = evaluate(policy_from_model(model), DEFAULT, args.episodes)

    if args.sb3:
        from gymnasium.wrappers import TransformReward
        train_env = TransformReward(FlowTwinEnv(DEFAULT), lambda r: r * cfg.reward_scale)   # learning only
        sb3 = MaskablePPO("MlpPolicy", train_env, seed=args.seed, n_steps=512, batch_size=64, verbose=0)
        sb3.learn(total_timesteps=args.steps)
        sb3.save("models/ppo_sb3")
        results["maskable ppo (sb3)"] = evaluate(
            lambda e, o: int(sb3.predict(o, deterministic=True, action_masks=e.action_masks())[0]), DEFAULT, args.episodes)

    save_results(results, out, "ppo_vs_heuristics", refs=("edd", "edd_low"))
    print((out / "ppo_vs_heuristics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
