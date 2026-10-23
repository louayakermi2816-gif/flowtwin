"""Minimal Proximal Policy Optimization (discrete actions) in PyTorch.

Written from scratch on purpose: actor-critic MLP, Generalised Advantage
Estimation, clipped surrogate objective, value loss and entropy bonus. The
implementation is kept small enough to be read in one sitting and is
cross-checked against sb3-contrib MaskablePPO in `scripts/train_ppo.py`.

Two details matter on FlowTwin:
- invalid-action masking: products with no remaining demand get zero probability
  (env.action_masks()), so no decision is wasted on them;
- reward_scale: rewards are scaled for learning only. Unscaled episode returns are
  ~300, and the critic's gradient then swamps the policy's on the shared trunk under
  the global gradient-norm clip. Logged returns stay in environment units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        for layer in (self.pi, self.v):
            nn.init.orthogonal_(layer.weight, gain=0.01 if layer is self.pi else 1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.trunk(obs)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)          # masked actions get zero probability
        return Categorical(logits=logits), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, mask: Optional[np.ndarray] = None) -> int:
        m = None if mask is None else torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        dist, _ = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), m)
        a = dist.probs.argmax(-1) if deterministic else dist.sample()
        return int(a.item())


@dataclass
class PPOConfig:
    total_steps: int = 50_000
    rollout_len: int = 512
    epochs: int = 4
    minibatch: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    reward_scale: float = 0.01      # learning only: keeps value targets O(1)
    mask_invalid: bool = True       # use info["action_mask"] when the env provides it
    seed: int = 0


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def train_ppo(env: gym.Env, cfg: PPOConfig = PPOConfig(), log: Optional[Callable[[Dict], None]] = print,
              model: Optional[ActorCritic] = None):
    """Train with PPO; `model` starts from given weights (a warm start) instead of a fresh network."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    if model is None:
        model = ActorCritic(obs_dim, env.action_space.n, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    obs, info = env.reset(seed=cfg.seed)
    mask = _mask(info, cfg)
    ep_return, ep_len, history = 0.0, 0, []
    steps = 0
    while steps < cfg.total_steps:
        B = {k: [] for k in ("obs", "mask", "act", "logp", "rew", "done", "val")}
        for _ in range(cfg.rollout_len):
            t_obs = torch.as_tensor(obs, dtype=torch.float32)
            t_mask = None if mask is None else torch.as_tensor(mask).unsqueeze(0)
            with torch.no_grad():
                dist, value = model(t_obs.unsqueeze(0), t_mask)
                action = dist.sample()
                logp = dist.log_prob(action)
            next_obs, r, term, trunc, info = env.step(int(action.item()))
            done = term or trunc
            B["obs"].append(obs); B["mask"].append(mask); B["act"].append(int(action)); B["logp"].append(float(logp))
            B["rew"].append(float(r) * cfg.reward_scale); B["done"].append(float(done)); B["val"].append(float(value))
            ep_return += r; ep_len += 1; steps += 1
            if done:
                history.append({"steps": steps, "return": ep_return, "length": ep_len, "kpis": info.get("kpis")})
                if log:
                    log({"steps": steps, "return": round(ep_return, 1), "length": ep_len})
                ep_return, ep_len = 0.0, 0
                next_obs, info = env.reset()
            obs, mask = next_obs, _mask(info, cfg)
        with torch.no_grad():
            _, last_value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        adv, ret = compute_gae(np.array(B["rew"]), np.array(B["val"]), np.array(B["done"]),
                               float(last_value), cfg.gamma, cfg.gae_lambda)
        t_obs = torch.as_tensor(np.array(B["obs"]), dtype=torch.float32)
        t_mask = None if B["mask"][0] is None else torch.as_tensor(np.array(B["mask"]))
        t_act = torch.as_tensor(B["act"])
        t_logp = torch.as_tensor(B["logp"])
        t_adv = torch.as_tensor(adv)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
        t_ret = torch.as_tensor(ret)

        n = len(t_obs)
        for _ in range(cfg.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, cfg.minibatch):
                mb = idx[start:start + cfg.minibatch]
                dist, value = model(t_obs[mb], None if t_mask is None else t_mask[mb])
                logp = dist.log_prob(t_act[mb])
                ratio = torch.exp(logp - t_logp[mb])
                surr = torch.min(ratio * t_adv[mb], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * t_adv[mb])
                loss = -surr.mean() + cfg.vf_coef * (value - t_ret[mb]).pow(2).mean() - cfg.ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
    return model, history


def _mask(info: Dict, cfg: PPOConfig) -> Optional[np.ndarray]:
    return info.get("action_mask") if cfg.mask_invalid else None


def policy_from_model(model: ActorCritic, deterministic: bool = True, masked: bool = True):
    def _policy(env, obs):
        mask = env.action_masks() if masked and hasattr(env, "action_masks") else None
        return model.act(obs, deterministic=deterministic, mask=mask)
    return _policy
