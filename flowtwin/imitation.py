"""Behaviour cloning: warm-start the PPO actor-critic from a rule's decisions before RL.

The actor learns the rule's action with a cross-entropy loss (invalid actions masked, as in
PPO) and the critic learns the rule's discounted return-to-go at PPO's reward scale, so the
first PPO updates start from the rule's policy with value estimates to match it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import LineConfig
from .env import FlowTwinEnv
from .ppo import ActorCritic, PPOConfig


@dataclass
class CloneConfig:
    episodes: int = 200       # demonstration shifts, seeds seed0 .. seed0 + episodes - 1
    seed0: int = 0            # disjoint from the evaluation seeds (1000 and up)
    epochs: int = 100         # held-out accuracy on shifts 200-249 (seed 0): 0.917 / 0.957 / 0.969 after 20 / 50 / 100
    minibatch: int = 64
    lr: float = 3e-4


def demonstrations(teacher: Callable, cfg: LineConfig, clone: CloneConfig, gamma: float,
                   reward_scale: float) -> Dict[str, np.ndarray]:
    """The teacher's decisions on clone.episodes shifts: observations, action masks, actions and
    the discounted return-to-go of each decision in learner units (rewards x reward_scale)."""
    env = FlowTwinEnv(cfg)
    rows = {k: [] for k in ("obs", "mask", "act", "ret")}
    for i in range(clone.episodes):
        obs, info = env.reset(seed=clone.seed0 + i)
        rewards, done = [], False
        while not done:
            action = teacher(env, obs)
            rows["obs"].append(obs); rows["mask"].append(info["action_mask"]); rows["act"].append(action)
            obs, r, term, trunc, info = env.step(action)
            rewards.append(r * reward_scale); done = term or trunc
        g, returns = 0.0, []
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        rows["ret"].extend(reversed(returns))
    return {k: np.asarray(v) for k, v in rows.items()}


def fit(model: ActorCritic, demos: Dict[str, np.ndarray], clone: CloneConfig, ppo: PPOConfig) -> Dict[str, float]:
    """Fit the actor to the teacher's actions and the critic to its returns. Returns the share
    of demonstrated decisions the greedy actor reproduces and the critic's mean squared error."""
    obs = torch.as_tensor(demos["obs"], dtype=torch.float32)
    mask = torch.as_tensor(demos["mask"], dtype=torch.bool)
    act = torch.as_tensor(demos["act"], dtype=torch.long)
    ret = torch.as_tensor(demos["ret"], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=clone.lr, eps=1e-5)
    for _ in range(clone.epochs):
        idx = torch.randperm(len(obs))
        for start in range(0, len(obs), clone.minibatch):
            mb = idx[start:start + clone.minibatch]
            dist, value = model(obs[mb], mask[mb])
            loss = -dist.log_prob(act[mb]).mean() + ppo.vf_coef * (value - ret[mb]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
            opt.step()
    with torch.no_grad():
        dist, value = model(obs, mask)
        return {"decisions": len(obs), "accuracy": float((dist.probs.argmax(-1) == act).float().mean()),
                "value_mse": float((value - ret).pow(2).mean())}


def warm_start(teacher: Callable, cfg: LineConfig, ppo: PPOConfig,
               clone: CloneConfig = CloneConfig()) -> Tuple[ActorCritic, Dict[str, float]]:
    """A fresh actor-critic (seeded with ppo.seed) fitted to the teacher's decisions."""
    torch.manual_seed(ppo.seed)
    env = FlowTwinEnv(cfg)
    model = ActorCritic(int(np.prod(env.observation_space.shape)), env.action_space.n, ppo.hidden)
    stats = fit(model, demonstrations(teacher, cfg, clone, ppo.gamma, ppo.reward_scale), clone, ppo)
    return model, stats
