"""Rule-selection action space (opt-in): each decision picks a dispatching rule, which chooses the
product, and a line speed, instead of picking the product directly. Following EDD order is then
one repeated action, every edd_slack decision is an available action (the EDD rule at its speed),
and no action can pick a product without demand. Observation, reward and line are unchanged.

Action: Discrete(len(rules) * n_speeds) -> (rule, speed) = divmod(action, n_speeds)

With rules=SPEED_ONLY the EDD order is fixed and the action is just the line speed.
"""
from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import FlowTwinEnv
from .heuristics import HEURISTICS

RULES = ("fifo", "spt", "edd")
SPEED_ONLY = ("edd",)


def to_line_action(env: FlowTwinEnv, obs: np.ndarray, action: int, rules: Sequence[str] = RULES) -> int:
    """The line's (product, speed) action for a (rule, speed) action in the current state."""
    rule, speed = divmod(int(action), env.cfg.n_speeds)
    product, _ = env.decode(HEURISTICS[rules[rule]](env, obs))
    return env.encode(product, speed)


def line_policy(policy: Callable, rules: Sequence[str] = RULES) -> Callable:
    """Wrap a policy over (rule, speed) actions so it acts on FlowTwinEnv directly, as the
    evaluation helpers expect. Every rule action is valid, so build `policy` unmasked: the env it
    is handed masks products, not rules."""
    return lambda env, obs: to_line_action(env, obs, policy(env, obs), rules)


class RuleSelectionEnv(gym.Wrapper):
    def __init__(self, env: FlowTwinEnv, rules: Sequence[str] = RULES):
        super().__init__(env)
        self.rules = tuple(rules)
        self.action_space = spaces.Discrete(len(self.rules) * env.cfg.n_speeds)
        self._obs = None

    def reset(self, **kwargs):
        self._obs, info = self.env.reset(**kwargs)
        return self._obs, self._with_mask(info)

    def step(self, action):
        line_action = to_line_action(self.env, self._obs, action, self.rules)
        self._obs, reward, terminated, truncated, info = self.env.step(line_action)
        return self._obs, reward, terminated, truncated, self._with_mask(info)

    def action_masks(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)     # every rule picks a product with demand

    def _with_mask(self, info: dict) -> dict:
        return {**info, "action_mask": self.action_masks()}
