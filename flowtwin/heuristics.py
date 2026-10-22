"""Dispatching heuristics used as baselines. Each policy maps an environment
(with a live ProductionLine) to a discrete action. All run at nominal speed except
edd_low, the speed-matched baseline: with the default reward weights energy outweighs
tardiness, so EDD at the lowest speed is the strongest fixed rule and any learned
policy must be compared with it, not only with nominal-speed EDD."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .env import FlowTwinEnv

LOW_SPEED, NOMINAL_SPEED, HIGH_SPEED = 0, 1, 2

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


def edd_low(env: FlowTwinEnv, obs: np.ndarray) -> int:
    """Earliest due date first, at the lowest line speed."""
    p, _ = env.decode(edd(env, obs))
    return env.encode(p, LOW_SPEED)


def make_edd_slack(margin: float) -> Policy:
    """EDD order with a slack-aware line speed, the strong baseline once lateness is priced.
    The order about to be launched is projected to finish at now + (units already in the line
    + its remaining units) x packer time per unit at nominal speed. Projected late -> high speed;
    slack above `margin` minutes -> low speed; otherwise nominal."""
    def edd_slack(env: FlowTwinEnv, obs: np.ndarray) -> int:
        ln, cfg = env.line, env.cfg
        p, _ = env.decode(edd(env, obs))
        wip = len(ln.inlet.items) + sum(len(b.items) for b in ln.buffers)
        finish = ln.now + (wip + ln.remaining[p]) * cfg.base_time[-1] * cfg.product_time_factor[p]
        slack = ln.orders.due[p] - finish
        speed = HIGH_SPEED if slack < 0 else (LOW_SPEED if slack > margin else NOMINAL_SPEED)
        return env.encode(p, speed)
    edd_slack.__name__ = f"edd_slack_{margin:g}"
    return edd_slack


def random_policy(env: FlowTwinEnv, obs: np.ndarray) -> int:
    return int(env.action_space.sample())


HEURISTICS: Dict[str, Policy] = {"fifo": fifo, "spt": spt, "edd": edd, "edd_low": edd_low, "random": random_policy}

# Rules shown next to the agents in the training scripts' tables. An explicit list keeps those
# tables unchanged when a rule is added to HEURISTICS for the API.
TABLE_RULES = ("fifo", "spt", "edd", "edd_low")
