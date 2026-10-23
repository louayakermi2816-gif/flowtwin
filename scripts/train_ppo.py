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
from flowtwin.heuristics import HEURISTICS, TABLE_RULES, make_edd_slack
from flowtwin.imitation import CloneConfig, warm_start
from flowtwin.order_actions import OrderFirstEnv, order_line_policy
from flowtwin.ppo import PPOConfig, policy_from_model, train_ppo
from flowtwin.rule_actions import RULES, SPEED_ONLY, RuleSelectionEnv, line_policy


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
    actions = ap.add_mutually_exclusive_group()
    actions.add_argument("--rule-actions", action="store_true",
                         help="each action picks a dispatching rule (FIFO/SPT/EDD) and a speed instead of a product")
    actions.add_argument("--speed-only", action="store_true", help="EDD picks the product; each action is only the line speed")
    actions.add_argument("--order-first", action="store_true",
                         help="the first decision of a shift picks the launch order, every later one the line speed")
    ap.add_argument("--warm-start-slack", type=float, metavar="MARGIN",
                    help="before PPO, behaviour-clone edd_slack(MARGIN) (flowtwin.imitation); product actions only")
    ap.add_argument("--clone-episodes", type=int, default=CloneConfig.episodes, help="demonstration shifts for the warm start")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models", help="directory for the trained checkpoints")
    args = ap.parse_args()
    rules = RULES if args.rule_actions else SPEED_ONLY if args.speed_only else None
    if args.sb3 and (rules or args.order_first):
        ap.error("--rule-actions, --speed-only and --order-first are not wired into the --sb3 cross-check")
    if args.warm_start_slack is not None and (rules or args.order_first or args.sb3):
        ap.error("--warm-start-slack works with product actions and without --sb3")
    if args.sb3:
        from sb3_contrib import MaskablePPO   # fail before training if the optional dependency is missing
    line_cfg = replace(DEFAULT, w_tardiness=args.w_tardiness)
    train_cfg = replace(line_cfg, dense_tardiness=args.dense_tardiness)   # learning signal only
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models = Path(args.models); models.mkdir(parents=True, exist_ok=True)

    env = FlowTwinEnv(train_cfg)
    if rules:
        env = RuleSelectionEnv(env, rules)
    elif args.order_first:
        env = OrderFirstEnv(env)
    cfg = PPOConfig(total_steps=args.steps, seed=args.seed)
    start, before_rl = None, None
    if args.warm_start_slack is not None:
        start, clone_stats = warm_start(make_edd_slack(args.warm_start_slack), train_cfg, cfg,
                                        CloneConfig(episodes=args.clone_episodes))
        torch.save(start.state_dict(), models / "ppo_warm_start.pt")
        (out / "ppo_warm_start.json").write_text(json.dumps(clone_stats, indent=1), encoding="utf-8")
        before_rl = evaluate(policy_from_model(start), line_cfg, args.episodes)   # train_ppo updates `start` in place
    model, history = train_ppo(env, cfg, model=start)
    torch.save(model.state_dict(), models / "ppo.pt")
    (out / "ppo_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    results = {name: evaluate(HEURISTICS[name], line_cfg, args.episodes) for name in TABLE_RULES}
    if before_rl is not None:
        results["ppo warm start (before rl)"] = before_rl
    agent = line_policy(policy_from_model(model, masked=False), rules) if rules else policy_from_model(model)
    if args.order_first:
        agent = order_line_policy(lambda o, m: model.act(o, deterministic=True, mask=m))
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
