"""Show what a trained agent does on the evaluation seeds: its line-speed mix and the order in
which it launches the products, compared with the EDD order.

    python scripts/inspect_agent.py --agent ppo --model models/ppo.pt
    python scripts/inspect_agent.py --agent dqn --model models/dqn.pt
"""
from __future__ import annotations

import argparse
import sys

import torch

from flowtwin.config import DEFAULT
from flowtwin.dqn import QNet, policy_from_qnet
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import policy_behaviour
from flowtwin.ppo import ActorCritic, policy_from_model


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=("ppo", "dqn"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()

    env = FlowTwinEnv(DEFAULT)
    net = (ActorCritic if args.agent == "ppo" else QNet)(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(args.model, weights_only=True))
    net.eval()
    policy = policy_from_model(net) if args.agent == "ppo" else policy_from_qnet(net)

    b = policy_behaviour(policy, DEFAULT, args.episodes, args.seed0)
    print("speed share (low / nominal / high): " + " / ".join(f"{s:.1%}" for s in b["speed_share"]))
    print(f"launch order equals the EDD order on {b['edd_order']}/{args.episodes} episodes")
    print("most common launch sequences (product indices):")
    for seq, count in b["sequences"].most_common(5):
        print(f"  {' -> '.join(map(str, seq))}: {count}")


if __name__ == "__main__":
    main()
