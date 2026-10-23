"""Show what a trained agent does on the evaluation seeds: its line-speed mix, the rules it picks
(rule-action agents) and the order in which it launches the products, compared with the EDD
order. --trace prints every decision of one shift.

    python scripts/inspect_agent.py --agent ppo --model models/ppo.pt
    python scripts/inspect_agent.py --agent dqn --model models/dqn.pt --trace 1000
    python scripts/inspect_agent.py --agent ppo --model models/runs/rules_wt2.5_s0/ppo.pt --rule-actions --trace 1000
    python scripts/inspect_agent.py --agent dqn --model models/runs/speed_wt2.5_s0/dqn.pt --speed-only --trace 1000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import torch

from flowtwin.config import DEFAULT
from flowtwin.dqn import QNet, policy_from_qnet
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import policy_behaviour
from flowtwin.ppo import ActorCritic, policy_from_model
from flowtwin.order_actions import OrderFirstEnv, order_line_policy
from flowtwin.rule_actions import RULES, SPEED_ONLY, RuleSelectionEnv, to_line_action

SPEEDS = ("low", "nominal", "high")


def trace(policy, picks, seed):
    env = FlowTwinEnv(DEFAULT)
    obs, _ = env.reset(seed=seed)
    ln = env.line
    print(f"\nshift {seed}: " + ", ".join(f"P{p} {q} units due at {d:.0f} min"
                                       for p, (q, d) in enumerate(zip(ln.orders.quantity, ln.orders.due))))
    done = False
    while not done:
        t, before = ln.now, len(picks)
        product, speed = env.decode(policy(env, obs))
        rule = f"{picks[-1]:>4} -> " if len(picks) > before else ""
        print(f"  {t:6.1f} min  {rule}P{product} at {SPEEDS[speed]} speed   remaining before: {ln.remaining}")
        obs, _, term, trunc, info = env.step(env.encode(product, speed))
        done = term or trunc
    k = info["kpis"]
    print(f"  shift ends at {k['time_min']} min: lateness {k['tardiness_min']} min, energy {k['energy']}, "
          f"changeover {k['changeover_min']} min")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=("ppo", "dqn"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    actions = ap.add_mutually_exclusive_group()
    actions.add_argument("--rule-actions", action="store_true", help="the agent was trained with --rule-actions")
    actions.add_argument("--speed-only", action="store_true", help="the agent was trained with --speed-only")
    actions.add_argument("--order-first", action="store_true", help="the agent was trained with --order-first")
    ap.add_argument("--trace", type=int, metavar="SEED", help="also print every decision of this shift")
    args = ap.parse_args()
    rules = RULES if args.rule_actions else SPEED_ONLY if args.speed_only else None

    env = RuleSelectionEnv(FlowTwinEnv(DEFAULT), rules) if rules else FlowTwinEnv(DEFAULT)
    if args.order_first:
        env = OrderFirstEnv(FlowTwinEnv(DEFAULT))
    net = (ActorCritic if args.agent == "ppo" else QNet)(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(args.model, weights_only=True))
    net.eval()
    masked = rules is None               # rule actions are always valid; the line env masks products
    agent = policy_from_model(net, masked=masked) if args.agent == "ppo" else policy_from_qnet(net, masked=masked)

    picks = []                           # the rule chosen at each decision (rule-action agents)

    def policy(env, obs):
        action = agent(env, obs)
        if rules is None:
            return action
        picks.append(rules[action // env.cfg.n_speeds])
        return to_line_action(env, obs, action, rules)

    if args.order_first:                 # one order decision per shift, then speeds
        policy = order_line_policy(lambda o, m: net.act(o, deterministic=True, mask=m))

    b = policy_behaviour(policy, DEFAULT, args.episodes, args.seed0)
    if rules and len(rules) > 1 and picks:
        counts = Counter(picks)
        print(f"rule share ({' / '.join(rules)}): " + " / ".join(f"{counts[r] / len(picks):.1%}" for r in rules))
    print("speed share (low / nominal / high): " + " / ".join(f"{s:.1%}" for s in b["speed_share"]))
    print(f"launch order equals the EDD order on {b['edd_order']}/{args.episodes} episodes")
    print("most common launch sequences (product indices):")
    for seq, count in b["sequences"].most_common(5):
        print(f"  {' -> '.join(map(str, seq))}: {count}")
    if args.trace is not None:
        trace(policy, picks, args.trace)


if __name__ == "__main__":
    main()
