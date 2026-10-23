"""Rule-selection action space: each action picks a dispatching rule and a speed, the rule picks
the product, so following a rule is one repeated action and no action is ever masked."""
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
from flowtwin.heuristics import HEURISTICS, LOW_SPEED, NOMINAL_SPEED
from flowtwin.rule_actions import RULES, RuleSelectionEnv, line_policy

ROOT = Path(__file__).resolve().parents[1]
CFG = replace(DEFAULT, w_tardiness=2.5)


def rule_action(rule, speed):
    return RULES.index(rule) * CFG.n_speeds + speed


@pytest.mark.parametrize("rule, speed, reference", [("edd", NOMINAL_SPEED, "edd"), ("edd", LOW_SPEED, "edd_low"),
                                                     ("fifo", NOMINAL_SPEED, "fifo"), ("spt", NOMINAL_SPEED, "spt")])
def test_repeating_one_rule_action_reproduces_that_rule(rule, speed, reference):
    agent = line_policy(lambda env, obs: rule_action(rule, speed))
    a, b = evaluate(agent, CFG, n_episodes=3), evaluate(HEURISTICS[reference], CFG, n_episodes=3)
    assert all(np.array_equal(a[c], b[c]) for c in a)


def test_wrapper_follows_the_rule_and_never_masks():
    env = RuleSelectionEnv(FlowTwinEnv(CFG))
    assert env.action_space.n == len(RULES) * CFG.n_speeds
    obs, info = env.reset(seed=1000)
    total, done = 0.0, False
    while not done:
        assert info["action_mask"].all()          # still all valid once a product has no demand left
        obs, r, term, trunc, info = env.step(rule_action("edd", LOW_SPEED))
        total += r; done = term or trunc
    assert info["kpis"]["invalid_actions"] == 0
    assert total == evaluate(HEURISTICS["edd_low"], CFG, n_episodes=1)["return"][0]


def test_training_and_inspect_scripts_take_rule_actions(tmp_path):
    pytest.importorskip("torch")
    subprocess.run([sys.executable, "scripts/train_ppo.py", "--steps", "512", "--episodes", "2", "--w-tardiness", "2.5",
                    "--rule-actions", "--out", str(tmp_path), "--models", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    saved = json.loads((tmp_path / "ppo_vs_heuristics.json").read_text(encoding="utf-8"))
    assert saved["ppo (ours)"]["invalid_actions"] == [0.0, 0.0]
    shown = subprocess.run([sys.executable, "scripts/inspect_agent.py", "--agent", "ppo", "--model", str(tmp_path / "ppo.pt"),
                            "--rule-actions", "--episodes", "2", "--trace", "1000"],
                           cwd=ROOT, check=True, capture_output=True, encoding="utf-8").stdout
    assert "rule share (fifo / spt / edd)" in shown and "shift 1000:" in shown
    refused = subprocess.run([sys.executable, "scripts/train_ppo.py", "--sb3", "--rule-actions"], cwd=ROOT, capture_output=True)
    assert refused.returncode == 2                 # argparse error before any training
