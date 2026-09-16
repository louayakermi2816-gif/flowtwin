"""Double DQN baseline (discrete actions) in PyTorch: replay buffer, target
network, epsilon-greedy exploration and the Double-DQN target
    y = r + gamma * Q_target(s', argmax_a Q_online(s', a)).
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
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        q = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(-1).item())


class Replay:
    def __init__(self, capacity: int, obs_dim: int):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, size=n)
        f = lambda x: torch.as_tensor(x[idx])
        return f(self.obs), f(self.act), f(self.rew), f(self.next_obs), f(self.done)


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
    seed: int = 0


def train_dqn(env: gym.Env, cfg: DQNConfig = DQNConfig(), log: Optional[Callable[[Dict], None]] = print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    q, q_target = QNet(obs_dim, n_actions, cfg.hidden), QNet(obs_dim, n_actions, cfg.hidden)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = Replay(cfg.buffer_size, obs_dim)

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_len, history = 0.0, 0, []
    for step in range(1, cfg.total_steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_decay_steps)
        action = int(rng.integers(n_actions)) if rng.random() < eps else q.act(obs)
        next_obs, r, term, trunc, info = env.step(action)
        buf.add(obs, action, r, next_obs, float(term))
        obs = next_obs
        ep_return += r; ep_len += 1
        if term or trunc:
            history.append({"steps": step, "return": ep_return, "length": ep_len, "kpis": info.get("kpis")})
            if log:
                log({"steps": step, "return": round(ep_return, 1), "length": ep_len, "eps": round(eps, 3)})
            ep_return, ep_len = 0.0, 0
            obs, _ = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            o, a, rw, o2, d = buf.sample(cfg.batch_size, rng)
            with torch.no_grad():
                best_next = q(o2).argmax(-1, keepdim=True)                 # action selection: online net
                target = rw + cfg.gamma * (1 - d) * q_target(o2).gather(1, best_next).squeeze(1)  # evaluation: target net
            pred = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
        if step % cfg.target_update_every == 0:
            q_target.load_state_dict(q.state_dict())
    return q, history


def policy_from_qnet(q: QNet):
    def _policy(env, obs):
        return q.act(obs)
    return _policy
