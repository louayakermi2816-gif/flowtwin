"""Rule-selection action space: each action picks a dispatching rule and a speed, the rule picks
the product, so following a rule is one repeated action and no action is ever masked. With
SPEED_ONLY the EDD order is fixed and the action is only the speed."""
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
from flowtwin.heuristics import HEURISTICS, LOW_SPEED, NOMINAL_SPEED, make_edd_slack
from flowtwin.rule_actions import RULES, SPEED_ONLY, RuleSelectionEnv, line_policy

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


@pytest.mark.parametrize("speed, reference", [(LOW_SPEED, "edd_low"), (NOMINAL_SPEED, "edd")])
def test_speed_only_fixes_the_edd_order(speed, reference):
    a = evaluate(line_policy(lambda env, obs: speed, SPEED_ONLY), CFG, n_episodes=3)
    b = evaluate(HEURISTICS[reference], CFG, n_episodes=3)
    assert all(np.array_equal(a[c], b[c]) for c in a)


def test_speed_only_can_express_the_slack_rule():
    assert RuleSelectionEnv(FlowTwinEnv(CFG), SPEED_ONLY).action_space.n == CFG.n_speeds
    rule = make_edd_slack(360)
    agent = line_policy(lambda env, obs: env.decode(rule(env, obs))[1], SPEED_ONLY)   # the rule's own speeds
    a, b = evaluate(agent, CFG, n_episodes=3), evaluate(rule, CFG, n_episodes=3)
    assert all(np.array_equal(a[c], b[c]) for c in a)


def test_speed_only_scripts_run_and_conflicting_action_flags_are_refused(tmp_path):
    pytest.importorskip("torch")
    subprocess.run([sys.executable, "scripts/train_dqn.py", "--steps", "300", "--episodes", "2", "--w-tardiness", "2.5",
                    "--speed-only", "--out", str(tmp_path), "--models", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    saved = json.loads((tmp_path / "dqn_vs_heuristics.json").read_text(encoding="utf-8"))
    assert saved["double-dqn"]["invalid_actions"] == [0.0, 0.0]
    shown = subprocess.run([sys.executable, "scripts/inspect_agent.py", "--agent", "dqn", "--model", str(tmp_path / "dqn.pt"),
                            "--speed-only", "--episodes", "2", "--trace", "1000"],
                           cwd=ROOT, check=True, capture_output=True, encoding="utf-8").stdout
    assert "launch order equals the EDD order on 2/2 episodes" in shown and " edd -> " in shown
    refused = subprocess.run([sys.executable, "scripts/train_dqn.py", "--rule-actions", "--speed-only"], cwd=ROOT, capture_output=True)
    assert refused.returncode == 2                 # argparse error before any training


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
