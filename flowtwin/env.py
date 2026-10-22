"""Gymnasium environment wrapping the SimPy production line.

Observation (float32 vector):
    buffer fill per inter-stage buffer (n_stages-1)
    inlet queue fill relative to batch size (1)
    stage down flags (n_stages)
    one-hot of the product currently on stage 0 (n_products)
    remaining demand fraction per product (n_products)
    urgency per product: (due - now)/horizon clipped to [-1, 1] (n_products)
    elapsed time fraction (1)

Action: Discrete(n_products * n_speeds) -> (product, speed) = divmod(action, n_speeds)
Action mask: env.action_masks() / info["action_mask"] is True for every action whose product
still has demand to launch (the method name matches sb3-contrib's MaskablePPO).

Reward per decision = w_throughput * new_units
                    - w_changeover * new_changeover_minutes
                    - w_energy     * new_energy
                    - w_tardiness  * new_tardiness_minutes
                    - w_invalid    * invalid_choice
                    - w_unmet      * unmet_units          (terminal step only)

new_tardiness_minutes is, by default, the lateness of the orders that completed during the
step (and of the open ones at the end of the shift). With cfg.dense_tardiness it is the
lateness every overdue order accrued during the step instead: the episode total is the same,
but the signal arrives while an order is late rather than in one lump when it completes.

The end of the shift is part of the MDP (elapsed time is observed and unmet units are
penalised), so an episode always *terminates*; it is never truncated.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DEFAULT, LineConfig
from .line import Orders, ProductionLine


class FlowTwinEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: LineConfig = DEFAULT, orders: Optional[Orders] = None):
        super().__init__()
        self.cfg = cfg
        self._fixed_orders = orders
        self.line: Optional[ProductionLine] = None
        n_obs = (cfg.n_stages - 1) + 1 + cfg.n_stages + 3 * cfg.n_products + 1
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(cfg.n_actions)
        self._snapshot: Dict[str, float] = {}

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        orders = self._fixed_orders or Orders.sample(self.cfg, rng)
        self.line = ProductionLine(self.cfg, orders, rng)
        self.line.run_until_decision()
        self._snapshot = self._metrics()
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.line is not None, "call reset() first"
        product, speed = self.decode(action)
        before = self._snapshot
        invalid = 1.0 if self.line.remaining[product] <= 0 else 0.0

        self.line.submit_action(product, speed)
        self.line.run_until_decision()

        after = self._metrics()
        cfg = self.cfg
        reward = (
            cfg.w_throughput * (after["units"] - before["units"])
            - cfg.w_changeover * (after["changeover"] - before["changeover"])
            - cfg.w_energy * (after["energy"] - before["energy"])
            - cfg.w_tardiness * (after["tardiness"] - before["tardiness"])
            - cfg.w_invalid * invalid
        )
        terminated = bool(self.line.finished)
        truncated = False
        if terminated:
            reward -= cfg.w_unmet * self.line.unmet_units
        self._snapshot = after
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------ helpers
    def decode(self, action: int) -> Tuple[int, int]:
        product, speed = divmod(int(action), self.cfg.n_speeds)
        return product, speed

    def encode(self, product: int, speed: int) -> int:
        return int(product) * self.cfg.n_speeds + int(speed)

    def action_masks(self) -> np.ndarray:
        """Boolean mask over actions: True where the product still has demand to launch
        (all actions once nothing is left, so a masked argmax is always defined)."""
        valid = self.line.valid_products() or range(self.cfg.n_products)
        mask = np.zeros(self.cfg.n_actions, dtype=bool)
        for p in valid:
            mask[self.encode(p, 0):self.encode(p, 0) + self.cfg.n_speeds] = True
        return mask

    def _metrics(self) -> Dict[str, float]:
        ln = self.line
        lateness = self._accrued_lateness() if self.cfg.dense_tardiness else ln.tardiness
        return {"units": ln.units_out, "changeover": ln.changeover_minutes,
                "energy": ln.energy, "tardiness": lateness}

    def _accrued_lateness(self) -> float:
        """Lateness accrued so far: each order counts from its due date until it completes,
        or until now while it is still open. At the end of an episode this equals
        line.tardiness, so the dense reward has the same episode total."""
        ln = self.line
        return sum(max(0.0, (done if done is not None else ln.now) - due)
                   for due, done in zip(ln.orders.due, ln.order_done_at))

    def _obs(self) -> np.ndarray:
        ln, cfg = self.line, self.cfg
        buf = [len(b.items) / cfg.buffer_capacity for b in ln.buffers]
        inlet = [min(1.0, len(ln.inlet.items) / cfg.batch_size)]
        down = [1.0 if d else 0.0 for d in ln.down]
        onehot = [0.0] * cfg.n_products
        if ln.last_product[0] is not None:
            onehot[ln.last_product[0]] = 1.0
        remaining = [ln.remaining[p] / max(1, ln.orders.quantity[p]) for p in range(cfg.n_products)]
        urgency = [float(np.clip((ln.orders.due[p] - ln.now) / cfg.horizon, -1.0, 1.0)) for p in range(cfg.n_products)]
        t = [min(1.0, ln.now / cfg.horizon)]
        return np.asarray(buf + inlet + down + onehot + remaining + urgency + t, dtype=np.float32)

    def _info(self) -> Dict[str, Any]:
        return {"kpis": self.line.kpis(), "valid_products": self.line.valid_products(),
                "action_mask": self.action_masks(),
                "orders": {"quantity": self.line.orders.quantity, "due": self.line.orders.due}}


def make_env(cfg: LineConfig = DEFAULT, orders: Optional[Orders] = None) -> FlowTwinEnv:
    return FlowTwinEnv(cfg, orders)
