"""Lateness-weight sensitivity: the weight reaches the reward, the scripts expose it without
changing their default outputs, and the slack-aware rule behaves as specified."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from flowtwin.config import DEFAULT
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate, paired_ci, policy_behaviour, t95
from flowtwin.heuristics import edd, edd_low, make_edd_slack
from flowtwin.line import Orders

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True)


def test_lateness_weight_only_rescales_the_tardiness_term():
    low, high = replace(DEFAULT, w_tardiness=0.10), replace(DEFAULT, w_tardiness=2.5)
    a, b = evaluate(edd, low, n_episodes=10), evaluate(edd, high, n_episodes=10)
    assert a["tardiness_min"].sum() > 0                                  # some shifts finish late
    assert np.array_equal(a["tardiness_min"], b["tardiness_min"])        # same dynamics
    # the KPI reports lateness rounded to 0.1 min, so allow the matching rounding error
    assert np.allclose(a["return"] - b["return"], (2.5 - 0.10) * a["tardiness_min"], rtol=0, atol=(2.5 - 0.10) * 0.05)


def test_baselines_script_takes_the_lateness_weight(tmp_path):
    run_script("scripts/run_baselines.py", "--episodes", "2", "--w-tardiness", "2.5", "--out", str(tmp_path))
    saved = json.loads((tmp_path / "baselines.json").read_text(encoding="utf-8"))
    expected = evaluate(edd, replace(DEFAULT, w_tardiness=2.5), n_episodes=2)["return"]
    assert np.allclose(saved["edd"]["return"], expected)


def _first_decision(policy, due):
    env = FlowTwinEnv(DEFAULT, Orders(quantity=[200, 200, 200], due=due))
    obs, _ = env.reset(seed=0)
    return env.decode(policy(env, obs))


def test_edd_slack_speeds_up_when_late_and_slows_down_with_slack():
    # P0 is due first; 200 units at 0.6 min/unit on the packer -> projected finish at 120 min
    assert _first_decision(make_edd_slack(60), [100.0, 700.0, 710.0]) == (0, 2)    # 20 min late: high
    assert _first_decision(make_edd_slack(60), [500.0, 700.0, 710.0]) == (0, 0)    # 380 min slack: low
    assert _first_decision(make_edd_slack(600), [500.0, 700.0, 710.0]) == (0, 1)   # slack under margin: nominal


def test_paired_ci_is_the_student_t_interval():
    d = np.array([1.0, 2.0, 4.0, 7.0])
    mean, half = paired_ci(d)
    assert mean == 3.5 and np.isclose(half, t95(3) * d.std(ddof=1) / 2.0)


def test_policy_behaviour_reads_speed_mix_and_launch_order():
    b = policy_behaviour(edd_low, DEFAULT, n_episodes=3)
    assert b["speed_share"] == [1.0, 0.0, 0.0]          # every batch at low speed
    assert b["edd_order"] == 3                           # EDD launches in EDD order by construction


def test_default_baselines_reproduce_the_committed_table(tmp_path):
    run_script("scripts/run_baselines.py", "--out", str(tmp_path))
    assert (tmp_path / "baselines.md").read_text(encoding="utf-8") == (ROOT / "results/baselines.md").read_text(encoding="utf-8")
