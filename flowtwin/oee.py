"""OEE proxy of the line's bottleneck (the packer), published by the API for supervision.

Time base: elapsed shift time so far, in minutes. There are no planned stops, so every stop is
a loss.

- availability = run time / elapsed, with run time = elapsed - packer downtime - packer changeover
- performance  = ideal packer time for the units produced / run time. The ideal time per unit is
                 the packer's base time x product factor at the fastest line speed, so a line
                 running flat out with no starving or blocking scores 1.
- quality      = 1: scrap is not modelled.
- oee          = availability x performance x quality

A breakdown stops a stage between units, so a repair can overlap the end of a unit in progress;
performance is capped at 1 for that reason.
"""
from __future__ import annotations

from typing import Dict

from .config import LineConfig


def oee_proxy(kpis: dict, cfg: LineConfig) -> Dict[str, float]:
    """kpis: ProductionLine.kpis() (needs time_min, downtime_min, stage_changeover_min, completed)."""
    elapsed = float(kpis["time_min"])
    if elapsed <= 0:
        return {"availability": 1.0, "performance": 0.0, "quality": 1.0, "oee": 0.0}
    run = max(elapsed - kpis["downtime_min"][-1] - kpis["stage_changeover_min"][-1], 1e-9)
    ideal = sum(n * cfg.base_time[-1] * cfg.product_time_factor[p] / max(cfg.speed_factor)
                for p, n in enumerate(kpis["completed"]))
    availability = min(1.0, max(0.0, run / elapsed))
    performance = min(1.0, ideal / run)
    return {"availability": round(availability, 3), "performance": round(performance, 3),
            "quality": 1.0, "oee": round(availability * performance, 3)}
