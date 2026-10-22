"""Configuration of the packaging-line digital twin.

All times are in minutes, energy in arbitrary kWh-like units. Defaults describe a
four-stage bottling/packaging line (filler -> capper -> labeller -> packer) running
three product variants, modelled after the Arena bottling-line coursework.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple


@dataclass(frozen=True)
class LineConfig:
    # --- topology -----------------------------------------------------------
    stages: Tuple[str, ...] = ("filler", "capper", "labeller", "packer")
    n_products: int = 3
    buffer_capacity: int = 30            # units between consecutive stages
    batch_size: int = 20                 # units launched per scheduling decision

    # --- process times ------------------------------------------------------
    base_time: Tuple[float, ...] = (0.50, 0.40, 0.45, 0.60)   # min/unit per stage @ nominal speed
    product_time_factor: Tuple[float, ...] = (1.0, 1.2, 0.9)  # per product variant
    speed_factor: Tuple[float, ...] = (0.8, 1.0, 1.25)        # low / nominal / high line speed
    changeover_time: float = 8.0                              # min lost when a stage switches product

    # --- energy -------------------------------------------------------------
    base_energy: Tuple[float, ...] = (1.0, 0.6, 0.5, 0.8)     # per unit per stage @ nominal speed
    energy_factor: Tuple[float, ...] = (0.7, 1.0, 1.6)        # multiplier per speed level

    # --- reliability (exponential time-to-failure / time-to-repair) ---------
    mtbf: Tuple[float, ...] = (240.0, 400.0, 300.0, 350.0)
    mttr: Tuple[float, ...] = (12.0, 8.0, 10.0, 15.0)
    breakdowns: bool = True

    # --- demand -------------------------------------------------------------
    order_qty_range: Tuple[int, int] = (150, 300)             # units per product order
    due_window: Tuple[float, float] = (0.35, 0.80)            # due dates as fraction of horizon
    horizon: float = 720.0                                    # one 12 h shift

    # --- reward weights -----------------------------------------------------
    w_throughput: float = 1.0       # per finished unit
    w_changeover: float = 0.5       # per minute of changeover
    w_energy: float = 0.15          # per energy unit
    w_tardiness: float = 0.10       # per minute an order finishes late
    w_unmet: float = 2.0            # per unit never produced by the end of the episode
    w_invalid: float = 5.0          # per decision that picked a product with no remaining demand
    dense_tardiness: bool = False   # charge lateness as it accrues instead of when an order completes
                                    # (same episode total, denser learning signal)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def n_speeds(self) -> int:
        return len(self.speed_factor)

    @property
    def n_actions(self) -> int:
        return self.n_products * self.n_speeds

    def without_breakdowns(self) -> "LineConfig":
        return replace(self, breakdowns=False)


DEFAULT = LineConfig()
