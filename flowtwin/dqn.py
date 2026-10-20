"""Double DQN baseline (discrete actions) in PyTorch: replay buffer, target
network, epsilon-greedy exploration and the Double-DQN target
    y = r + gamma * Q_target(s', argmax_a Q_online(s', a)).

With invalid-action masking (env.action_masks()), exploration samples only valid
actions and the argmax runs over the valid actions of s'. Rewards are scaled by
reward_scale for learning only, so Q-values are O(1) instead of ~300.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True, mask: Optional[np.ndarray] = None) -> int:
        q = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if mask is not None:
            q = q.masked_fill(~torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0), -float("inf"))
        return int(q.argmax(-1).item())


class Replay:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_mask = np.ones((capacity, n_actions), bool)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def add(self, o, a, r, o2, d, m2):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i], self.next_mask[i] = o, a, r, o2, d, m2
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, size=n)
        f = lambda x: torch.as_tensor(x[idx])
        return f(self.obs), f(self.act), f(self.rew), f(self.next_obs), f(self.done), f(self.next_mask)


@dataclass
class DQNConfig:
    total_steps: int = 50_000
    buffer_size: int = 50_000
    batch_size: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    learning_starts: int = 1_000
    train_every: int = 4
    target_update_every: int = 1_000
    hidden: int = 128
    reward_scale: float = 0.01      # learning only: keeps Q-values O(1)
    mask_invalid: bool = True       # use info["action_mask"] when the env provides it
    seed: int = 0


def train_dqn(env: gym.Env, cfg: DQNConfig = DQNConfig(), log: Optional[Callable[[Dict], None]] = print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    q, q_target = QNet(obs_dim, n_actions, cfg.hidden), QNet(obs_dim, n_actions, cfg.hidden)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = Replay(cfg.buffer_size, obs_dim, n_actions)

    obs, info = env.reset(seed=cfg.seed)
    mask = _mask(info, cfg, n_actions)
    ep_return, ep_len, history = 0.0, 0, []
    for step in range(1, cfg.total_steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_decay_steps)
        action = int(rng.choice(np.flatnonzero(mask))) if rng.random() < eps else q.act(obs, mask=mask)
        next_obs, r, term, trunc, info = env.step(action)
        next_mask = _mask(info, cfg, n_actions)
        buf.add(obs, action, r * cfg.reward_scale, next_obs, float(term), next_mask)
        obs, mask = next_obs, next_mask
        ep_return += r; ep_len += 1
        if term or trunc:
            history.append({"steps": step, "return": ep_return, "length": ep_len, "kpis": info.get("kpis")})
            if log:
                log({"steps": step, "return": round(ep_return, 1), "length": ep_len, "eps": round(eps, 3)})
            ep_return, ep_len = 0.0, 0
            obs, info = env.reset()
            mask = _mask(info, cfg, n_actions)

        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            o, a, rw, o2, d, m2 = buf.sample(cfg.batch_size, rng)
            with torch.no_grad():
                best_next = q(o2).masked_fill(~m2, -float("inf")).argmax(-1, keepdim=True)  # action selection: online net
                target = rw + cfg.gamma * (1 - d) * q_target(o2).gather(1, best_next).squeeze(1)  # evaluation: target net
            pred = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
        if step % cfg.target_update_every == 0:
            q_target.load_state_dict(q.state_dict())
    return q, history


def _mask(info: Dict, cfg: DQNConfig, n_actions: int) -> np.ndarray:
    m = info.get("action_mask") if cfg.mask_invalid else None
    return np.ones(n_actions, bool) if m is None else m


def policy_from_qnet(q: QNet, masked: bool = True):
    def _policy(env, obs):
        mask = env.action_masks() if masked and hasattr(env, "action_masks") else None
        return q.act(obs, mask=mask)
    return _policy
