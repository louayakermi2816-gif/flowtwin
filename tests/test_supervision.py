"""Supervision KPIs: per-stage downtime and changeover, per-product output, and the OEE proxy."""
from flowtwin.config import DEFAULT
from flowtwin.env import FlowTwinEnv
from flowtwin.heuristics import edd
from flowtwin.oee import oee_proxy


def run_shift(cfg, seed=5):
    env = FlowTwinEnv(cfg)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(edd(env, obs))
        done = term or trunc
    return info["kpis"]


def test_downtime_is_zero_without_breakdowns_and_positive_with_them():
    assert run_shift(DEFAULT.without_breakdowns())["downtime_min"] == [0.0] * DEFAULT.n_stages
    assert sum(run_shift(DEFAULT)["downtime_min"]) > 0


def test_stage_changeovers_and_completed_units_add_up():
    k = run_shift(DEFAULT)
    assert sum(k["completed"]) == k["units_out"]
    assert sum(k["stage_changeover_min"]) == k["changeover_min"]


def test_oee_proxy_is_a_fraction_and_the_product_of_its_factors():
    for cfg in (DEFAULT, DEFAULT.without_breakdowns()):
        o = oee_proxy(run_shift(cfg), cfg)
        assert all(0.0 <= o[key] <= 1.0 for key in ("availability", "performance", "oee"))
        assert abs(o["oee"] - o["availability"] * o["performance"]) < 1e-3


def test_oee_proxy_before_the_shift_starts():
    assert oee_proxy({"time_min": 0.0}, DEFAULT)["oee"] == 0.0
