"""Order-first action space: one launch-order decision per shift, then the speed of every batch.
Planning the EDD order reproduces the EDD rules exactly, so the best rule is one of the policies
an order-first agent can learn."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flowtwin.config import DEFAULT
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate
from flowtwin.heuristics import HEURISTICS, NOMINAL_SPEED, make_edd_slack
from flowtwin.order_actions import ORDERS, OrderFirstEnv, order_line_policy

ROOT = Path(__file__).resolve().parents[1]
CFG = replace(DEFAULT, w_tardiness=2.5)
URGENCY = slice(14, 17)        # (due - now) / horizon per product in the line's observation


def edd_order(line_env):
    return tuple(int(p) for p in np.argsort(line_env.line.orders.due, kind="stable"))


def run_order_first(seed, speed_of):
    """One shift in the wrapper: plan the EDD order, then take speed_of(line env) for every batch."""
    env = OrderFirstEnv(FlowTwinEnv(CFG))
    obs, info = env.reset(seed=seed)
    line_env = env.unwrapped
    assert obs[-7] == 1.0 and info["action_mask"].all()                 # the planning decision
    obs, r, term, trunc, info = env.step(ORDERS.index(edd_order(line_env)))
    assert r == 0.0 and line_env.line.now == 0.0 and info["kpis"]["decisions"] == 0   # nothing launched
    assert info["action_mask"].sum() == CFG.n_speeds and obs[-7] == 0.0
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(speed_of(line_env))
        total += r; done = term or trunc
    return total, info["kpis"]


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_planning_edd_reproduces_the_edd_rules(seed):
    slack = make_edd_slack(360)
    for speed_of, rule in ((lambda e: NOMINAL_SPEED, HEURISTICS["edd"]), (lambda e: e.decode(slack(e, None))[1], slack)):
        total, kpis = run_order_first(seed, speed_of)
        ref = evaluate(rule, CFG, n_episodes=1, seed0=seed)
        assert total == ref["return"][0] and kpis["tardiness_min"] == ref["tardiness_min"][0]


def test_line_policy_plans_once_per_shift():
    plans = []

    def choose(obs, mask):                 # EDD from the observed urgency, nominal speed afterwards
        if obs[-7] == 1.0:
            plans.append(1)
            return ORDERS.index(tuple(int(p) for p in np.argsort(obs[URGENCY], kind="stable")))
        assert not mask[CFG.n_speeds:].any()
        return NOMINAL_SPEED

    a = evaluate(order_line_policy(choose), CFG, n_episodes=3)
    b = evaluate(HEURISTICS["edd"], CFG, n_episodes=3)
    assert all(np.array_equal(a[c], b[c]) for c in a)
    assert len(plans) == 3


def test_order_first_scripts_run_and_conflicting_flags_are_refused(tmp_path):
    pytest.importorskip("torch")
    subprocess.run([sys.executable, "scripts/train_ppo.py", "--steps", "512", "--episodes", "2", "--w-tardiness", "2.5",
                    "--order-first", "--out", str(tmp_path), "--models", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    saved = json.loads((tmp_path / "ppo_vs_heuristics.json").read_text(encoding="utf-8"))
    assert saved["ppo (ours)"]["invalid_actions"] == [0.0, 0.0]
    shown = subprocess.run([sys.executable, "scripts/inspect_agent.py", "--agent", "ppo", "--model", str(tmp_path / "ppo.pt"),
                            "--order-first", "--episodes", "2"], cwd=ROOT, check=True, capture_output=True,
                           encoding="utf-8").stdout
    assert "launch order equals the EDD order on" in shown
    refused = subprocess.run([sys.executable, "scripts/train_dqn.py", "--order-first", "--speed-only"], cwd=ROOT,
                             capture_output=True)
    assert refused.returncode == 2                 # argparse error before any training
