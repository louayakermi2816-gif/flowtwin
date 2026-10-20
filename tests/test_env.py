import numpy as np
import pytest

from flowtwin.config import DEFAULT, LineConfig
from flowtwin.env import FlowTwinEnv
from flowtwin.heuristics import HEURISTICS, edd
from flowtwin.line import Orders
from flowtwin.plan import compute_plan, deviations


def rollout(env, policy, seed):
    obs, info = env.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        total += r; steps += 1; done = term or trunc
        assert steps < 10_000
    return total, info["kpis"]


def test_spaces_and_reset():
    env = FlowTwinEnv(DEFAULT)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert env.action_space.n == DEFAULT.n_products * DEFAULT.n_speeds
    assert info["kpis"]["units_out"] == 0
    assert len(info["valid_products"]) == DEFAULT.n_products


def test_episode_completes_and_is_deterministic_under_seed():
    a = rollout(FlowTwinEnv(DEFAULT), edd, seed=7)
    b = rollout(FlowTwinEnv(DEFAULT), edd, seed=7)
    assert a == b
    total, k = a
    assert k["finished"] is True
    assert k["units_out"] + k["unmet_units"] == sum(FlowTwinEnv(DEFAULT).reset(seed=7)[1]["orders"]["quantity"])
    assert k["time_min"] <= DEFAULT.horizon


def test_different_seeds_differ():
    assert rollout(FlowTwinEnv(DEFAULT), edd, 1) != rollout(FlowTwinEnv(DEFAULT), edd, 2)


@pytest.mark.parametrize("name", list(HEURISTICS))
def test_heuristics_run(name):
    total, k = rollout(FlowTwinEnv(DEFAULT), HEURISTICS[name], seed=3)
    assert np.isfinite(total)
    assert k["decisions"] > 0


def test_invalid_action_is_penalised_and_recovered():
    cfg = LineConfig(order_qty_range=(20, 20))
    env = FlowTwinEnv(cfg, Orders(quantity=[20, 20, 20], due=[400.0, 500.0, 600.0]))
    obs, _ = env.reset(seed=0)
    env.step(env.encode(0, 1))                         # product 0 fully launched (one batch)
    _, r, _, _, info = env.step(env.encode(0, 1))     # invalid: nothing left of product 0
    assert info["kpis"]["invalid_actions"] == 1
    assert 0 not in info["valid_products"] or info["kpis"]["finished"]


def test_action_masks_follow_remaining_demand():
    cfg = LineConfig(order_qty_range=(20, 20))
    env = FlowTwinEnv(cfg, Orders(quantity=[20, 20, 20], due=[400.0, 500.0, 600.0]))
    obs, info = env.reset(seed=0)
    assert info["action_mask"].dtype == bool and info["action_mask"].all()
    _, _, _, _, info = env.step(env.encode(0, 1))      # product 0 fully launched (one batch)
    mask = info["action_mask"]
    assert np.array_equal(mask, env.action_masks())
    assert not mask[env.encode(0, 0):env.encode(0, 0) + cfg.n_speeds].any()
    assert mask[env.encode(1, 0):].all()


def test_end_of_shift_terminates_with_unmet_penalty():
    env = FlowTwinEnv(LineConfig(horizon=60.0))         # far too short for the demand
    obs, _ = env.reset(seed=2)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(edd(env, obs))
        done = term or trunc
    assert term and not trunc                           # the shift end is a true terminal state
    assert info["kpis"]["unmet_units"] > 0
    assert r < 0                                        # the last step carries the unmet penalty


def test_no_breakdown_plan_and_deviations():
    env = FlowTwinEnv(DEFAULT)
    obs, info = env.reset(seed=11)
    plan = compute_plan(DEFAULT, env.line.orders)
    assert len(plan.completion) == DEFAULT.n_products
    assert all(0 < c <= DEFAULT.horizon for c in plan.completion)
    env.step(edd(env, obs))
    d = deviations(env, plan)
    assert set(d) >= {"planned_units", "actual_units", "flow_gap_units", "products"}
    assert len(d["products"]) == DEFAULT.n_products


def test_changeovers_and_energy_accumulate():
    env = FlowTwinEnv(DEFAULT)
    obs, _ = env.reset(seed=5)
    env.step(env.encode(0, 2)); _, _, _, _, info = env.step(env.encode(1, 0))
    while not env.line.finished:
        _, _, _, _, info = env.step(edd(env, obs))
    k = info["kpis"]
    assert k["changeover_min"] >= DEFAULT.changeover_time
    assert k["energy"] > 0
