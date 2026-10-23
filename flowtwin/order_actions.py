"""Order-first action space (opt-in): the agent decides the launch order once, at the start of
the shift, then only the line speed of every batch. Credit for the order then flows from one
decision to the whole shift instead of being spread over every product choice.

Actions: Discrete(len(ORDERS)). The first decision of a shift picks ORDERS[action], a permutation
of the products, and launches nothing (reward 0); every later decision picks the speed
(actions 0 .. n_speeds - 1, the rest are masked) of a batch of the first product in that order
that still has demand. The observation appends a planning flag (1 at the order decision) and a
one-hot of the chosen order to the line's observation.
"""
from __future__ import annotations

from itertools import permutations
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import FlowTwinEnv

ORDERS: Tuple[Tuple[int, ...], ...] = tuple(permutations(range(3)))


def extend(obs: np.ndarray, order: Optional[Tuple[int, ...]]) -> np.ndarray:
    """The line's observation plus the planning flag and the chosen order (none yet: planning)."""
    tail = np.zeros(1 + len(ORDERS), dtype=np.float32)
    if order is None:
        tail[0] = 1.0
    else:
        tail[1 + ORDERS.index(order)] = 1.0
    return np.concatenate([obs, tail]).astype(np.float32)


def phase_mask(env: FlowTwinEnv, order: Optional[Tuple[int, ...]]) -> np.ndarray:
    """All orders at the planning decision, then the speeds only."""
    mask = np.zeros(len(ORDERS), dtype=bool)
    mask[:len(ORDERS) if order is None else env.cfg.n_speeds] = True
    return mask


def next_product(env: FlowTwinEnv, order: Tuple[int, ...]) -> int:
    """The first product in the chosen order that still has demand to launch."""
    return next((p for p in order if env.line.remaining[p] > 0), order[-1])


class OrderFirstEnv(gym.Wrapper):
    def __init__(self, env: FlowTwinEnv):
        super().__init__(env)
        if env.cfg.n_products != 3 or env.cfg.n_speeds > len(ORDERS):
            raise ValueError("ORDERS covers three products")
        n_obs = env.observation_space.shape[0] + 1 + len(ORDERS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ORDERS))
        self.order: Optional[Tuple[int, ...]] = None
        self._obs, self._info = None, {}

    def reset(self, **kwargs):
        self._obs, self._info = self.env.reset(**kwargs)
        self.order = None
        return extend(self._obs, None), self._with_mask(self._info)

    def step(self, action):
        line = self.env
        if self.order is None:                       # the planning decision: no batch launched
            self.order = ORDERS[int(action)]
            return extend(self._obs, self.order), 0.0, False, False, self._with_mask(self._info)
        product = next_product(line, self.order)
        self._obs, reward, terminated, truncated, self._info = line.step(line.encode(product, int(action)))
        return extend(self._obs, self.order), reward, terminated, truncated, self._with_mask(self._info)

    def action_masks(self) -> np.ndarray:
        return phase_mask(self.env, self.order)

    def _with_mask(self, info: dict) -> dict:
        return {**info, "action_mask": self.action_masks()}


def order_line_policy(choose: Callable[[np.ndarray, np.ndarray], int]) -> Callable:
    """Turn an agent trained in OrderFirstEnv, given as choose(observation, mask) -> action,
    into a policy on FlowTwinEnv for the evaluation helpers: on the first decision of each
    shift it takes the order decision, then it picks the speed of every batch."""
    plan: Dict[str, Any] = {"line": None, "order": None}

    def act(env: FlowTwinEnv, obs: np.ndarray) -> int:
        if plan["line"] is not env.line:             # a new shift
            plan["line"] = env.line
            plan["order"] = ORDERS[choose(extend(obs, None), phase_mask(env, None))]
        order = plan["order"]
        speed = choose(extend(obs, order), phase_mask(env, order))
        return env.encode(next_product(env, order), speed)

    return act
