"""Planned-vs-actual layer.

The *plan* is what a planner expects: the EDD schedule executed at nominal speed
on a line without breakdowns. The *actual* is the live (stochastic) twin. The
deviation per product is actual completion time minus planned completion time;
an alert is raised when the gap exceeds a threshold, mirroring the way a sawmill
or packaging planner monitors discrepancies between planned and realised flows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import LineConfig
from .env import FlowTwinEnv
from .line import Orders


@dataclass
class Plan:
    completion: List[float]        # planned completion time per product (minutes)
    makespan: float
    units_by_time: Dict[float, int]  # planned cumulative output at checkpoints


def compute_plan(cfg: LineConfig, orders: Orders, seed: int = 0) -> Plan:
    from .heuristics import edd  # local import to avoid a cycle

    env = FlowTwinEnv(cfg.without_breakdowns(), orders)
    obs, info = env.reset(seed=seed)
    checkpoints: Dict[float, int] = {}
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(edd(env, obs))
        checkpoints[round(env.line.now, 1)] = env.line.units_out
        done = term or trunc
    completion = [c if c is not None else cfg.horizon for c in env.line.order_done_at]
    return Plan(completion=completion, makespan=env.line.now, units_by_time=checkpoints)


def planned_units_at(plan: Plan, t: float) -> int:
    times = sorted(plan.units_by_time)
    val = 0
    for ct in times:
        if ct <= t:
            val = plan.units_by_time[ct]
        else:
            break
    return val


def deviations(env: FlowTwinEnv, plan: Plan, threshold_min: float = 45.0) -> dict:
    ln = env.line
    now = ln.now
    per_product = []
    for p in range(env.cfg.n_products):
        actual = ln.order_done_at[p]
        planned = plan.completion[p]
        if actual is not None:
            gap = actual - planned
        else:
            gap = max(0.0, now - planned)          # already late relative to plan
        per_product.append({"product": p, "planned_completion": round(planned, 1),
                            "actual_completion": None if actual is None else round(actual, 1),
                            "deviation_min": round(gap, 1), "alert": bool(gap > threshold_min)})
    planned_units = planned_units_at(plan, now)
    flow_gap = ln.units_out - planned_units
    return {"time_min": round(now, 1), "planned_units": planned_units, "actual_units": ln.units_out,
            "flow_gap_units": flow_gap, "flow_alert": bool(flow_gap < -env.cfg.batch_size),
            "products": per_product}
