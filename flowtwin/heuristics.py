"""Dispatching heuristics used as baselines. Each policy maps an environment
(with a live ProductionLine) to a discrete action; all run at nominal speed."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .env import FlowTwinEnv

NOMINAL_SPEED = 1

Policy = Callable[[FlowTwinEnv, np.ndarray], int]


def _valid(env: FlowTwinEnv):
    valid = env.line.valid_products()
    return valid if valid else list(range(env.cfg.n_products))


def fifo(env: FlowTwinEnv, obs: np.ndarray) -> int:
    """Orders are released in product index order: serve the lowest index first."""
    p = min(_valid(env))
    return env.encode(p, NOMINAL_SPEED)


def spt(env: FlowTwinEnv, obs: np.ndarray) -> int:
    """Shortest processing time: least remaining work (units x time factor) first."""
    ln, cfg = env.line, env.cfg
    p = min(_valid(env), key=lambda q: ln.remaining[q] * cfg.product_time_factor[q])
    return env.encode(p, NOMINAL_SPEED)


def edd(env: FlowTwinEnv, obs: np.ndarray) -> int:
    """Earliest due date first."""
    ln = env.line
    p = min(_valid(env), key=lambda q: ln.orders.due[q])
    return env.encode(p, NOMINAL_SPEED)


def random_policy(env: FlowTwinEnv, obs: np.ndarray) -> int:
    return int(env.action_space.sample())


HEURISTICS: Dict[str, Policy] = {"fifo": fifo, "spt": spt, "edd": edd, "random": random_policy}
