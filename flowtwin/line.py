"""Discrete-event digital twin of a multi-product packaging line (SimPy).

The line is a chain of stages separated by finite buffers. A *loader* process
launches batches of one product at one line speed into the first stage; every
launch is a scheduling decision that an external agent (heuristic or RL policy)
has to take. The twin therefore exposes a decision/response handshake:

    line.request_decision   -> triggered by the loader when it needs an action
    line.submit_action(a)   -> called by the environment with the chosen action

Breakdowns are sampled per stage from exponential time-to-failure / time-to-repair
distributions and block the stage before its next unit. Changeovers occur at every
stage when the product variant switches. Energy is accumulated per unit processed
and scales with the chosen line speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import simpy

from .config import LineConfig


@dataclass
class Orders:
    quantity: List[int]          # units demanded per product
    due: List[float]             # due date per product (minutes)

    @staticmethod
    def sample(cfg: LineConfig, rng: np.random.Generator) -> "Orders":
        lo, hi = cfg.order_qty_range
        qty = rng.integers(lo, hi + 1, size=cfg.n_products).tolist()
        due = (rng.uniform(*cfg.due_window, size=cfg.n_products) * cfg.horizon).round(1).tolist()
        return Orders(quantity=[int(q) for q in qty], due=[float(d) for d in due])


class ProductionLine:
    """SimPy model of the line. One instance == one episode."""

    def __init__(self, cfg: LineConfig, orders: Orders, rng: np.random.Generator):
        self.cfg = cfg
        self.orders = orders
        self.rng = rng
        self.sim = simpy.Environment()
        n = cfg.n_stages

        # material flow
        self.inlet = simpy.Store(self.sim)                       # units launched, waiting for stage 0
        self.buffers = [simpy.Store(self.sim, capacity=cfg.buffer_capacity) for _ in range(n - 1)]

        # stage state
        self.last_product: List[Optional[int]] = [None] * n
        self.down = [False] * n
        self.busy = [False] * n
        self._repaired = [self.sim.event() for _ in range(n)]

        # demand / KPI state
        self.remaining = list(orders.quantity)
        self.completed = [0] * cfg.n_products
        self.order_done_at: List[Optional[float]] = [None] * cfg.n_products
        self.units_out = 0
        self.changeover_minutes = 0.0
        self.energy = 0.0
        self.tardiness = 0.0
        self.invalid_actions = 0
        self.decisions = 0
        self.finished = False

        # decision handshake
        self.request_decision = self.sim.event()
        self._action_event = self.sim.event()

        # processes
        self.sim.process(self._loader())
        for i in range(n):
            self.sim.process(self._stage(i))
            if cfg.breakdowns:
                self.sim.process(self._failures(i))

    # ------------------------------------------------------------------ API
    @property
    def now(self) -> float:
        return float(self.sim.now)

    @property
    def total_demand(self) -> int:
        return int(sum(self.orders.quantity))

    def run_until_decision(self) -> None:
        """Advance the simulation until the loader asks for the next decision
        (or the episode finishes / the horizon is reached)."""
        self.request_decision = self.sim.event()
        horizon = self.sim.timeout(max(0.0, self.cfg.horizon - self.sim.now))
        self.sim.run(until=simpy.AnyOf(self.sim, [self.request_decision, horizon]))
        if self.sim.now >= self.cfg.horizon and not self.finished:
            self._close_episode()

    def submit_action(self, product: int, speed: int) -> None:
        self._action_event.succeed((int(product), int(speed)))

    def valid_products(self) -> List[int]:
        return [p for p in range(self.cfg.n_products) if self.remaining[p] > 0]

    # ------------------------------------------------------------ processes
    def _loader(self):
        cfg = self.cfg
        while sum(self.remaining) > 0 and self.sim.now < cfg.horizon:
            # ask the controller what to launch next
            self.request_decision.succeed()
            self._action_event = self.sim.event()
            product, speed = yield self._action_event
            self.decisions += 1

            if self.remaining[product] <= 0:                     # invalid choice -> fall back to EDD
                self.invalid_actions += 1
                product = min(self.valid_products(), key=lambda p: self.orders.due[p])

            qty = min(cfg.batch_size, self.remaining[product])
            self.remaining[product] -= qty
            for _ in range(qty):
                yield self.inlet.put({"product": product, "speed": speed})

            # pace decisions with production: wait until the launched batch entered stage 0
            while len(self.inlet.items) > 0:
                yield self.sim.timeout(1.0)

        # everything launched: let the line drain
        while self.units_out < self.total_demand and self.sim.now < cfg.horizon:
            yield self.sim.timeout(1.0)
        self._close_episode()
        if not self.request_decision.triggered:
            self.request_decision.succeed()

    def _stage(self, i: int):
        cfg = self.cfg
        src = self.inlet if i == 0 else self.buffers[i - 1]
        dst = self.buffers[i] if i < cfg.n_stages - 1 else None
        while True:
            unit = yield src.get()
            while self.down[i]:
                yield self._repaired[i]
            self.busy[i] = True
            p, s = unit["product"], unit["speed"]
            if self.last_product[i] is not None and self.last_product[i] != p:
                yield self.sim.timeout(cfg.changeover_time)
                self.changeover_minutes += cfg.changeover_time
            self.last_product[i] = p
            t = cfg.base_time[i] * cfg.product_time_factor[p] / cfg.speed_factor[s]
            yield self.sim.timeout(t)
            self.energy += cfg.base_energy[i] * cfg.energy_factor[s]
            self.busy[i] = False
            if dst is not None:
                yield dst.put(unit)
            else:
                self._complete(p)

    def _failures(self, i: int):
        cfg = self.cfg
        while True:
            yield self.sim.timeout(self.rng.exponential(cfg.mtbf[i]))
            self.down[i] = True
            self._repaired[i] = self.sim.event()
            yield self.sim.timeout(self.rng.exponential(cfg.mttr[i]))
            self.down[i] = False
            self._repaired[i].succeed()

    # ------------------------------------------------------------- helpers
    def _complete(self, product: int) -> None:
        self.units_out += 1
        self.completed[product] += 1
        if self.completed[product] >= self.orders.quantity[product] and self.order_done_at[product] is None:
            self.order_done_at[product] = self.now
            self.tardiness += max(0.0, self.now - self.orders.due[product])

    def _close_episode(self) -> None:
        if self.finished:
            return
        self.finished = True
        for p in range(self.cfg.n_products):
            if self.order_done_at[p] is None:
                self.tardiness += max(0.0, self.now - self.orders.due[p])

    @property
    def unmet_units(self) -> int:
        return self.total_demand - self.units_out

    def kpis(self) -> dict:
        elapsed = max(self.now, 1e-9)
        return {
            "time_min": round(self.now, 1),
            "units_out": self.units_out,
            "throughput_per_h": round(60.0 * self.units_out / elapsed, 2),
            "changeover_min": round(self.changeover_minutes, 1),
            "energy": round(self.energy, 2),
            "tardiness_min": round(self.tardiness, 1),
            "unmet_units": self.unmet_units,
            "invalid_actions": self.invalid_actions,
            "decisions": self.decisions,
            "finished": self.finished,
        }
