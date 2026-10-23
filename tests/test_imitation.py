"""Behaviour-cloning warm start: demonstrations carry the learner's discounted returns, the cloned
actor reproduces its teacher, and PPO continues from the cloned weights."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flowtwin.config import DEFAULT, LineConfig
from flowtwin.env import FlowTwinEnv
from flowtwin.heuristics import edd
from flowtwin.imitation import CloneConfig, demonstrations, warm_start
from flowtwin.ppo import PPOConfig, train_ppo

ROOT = Path(__file__).resolve().parents[1]
SMALL = LineConfig(order_qty_range=(40, 60), horizon=600.0)


def test_demonstrations_hold_the_scaled_return_to_go():
    demos = demonstrations(edd, DEFAULT, CloneConfig(episodes=1, seed0=7), gamma=1.0, reward_scale=0.01)
    env = FlowTwinEnv(DEFAULT)
    obs, _ = env.reset(seed=7)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(edd(env, obs))
        total += r; done = term or trunc
    assert len(demos["obs"]) == len(demos["mask"]) == len(demos["act"]) == len(demos["ret"])
    assert np.isclose(demos["ret"][0], 0.01 * total)      # with gamma = 1 the first decision sees the whole shift


def test_warm_start_reproduces_its_teacher():
    _, stats = warm_start(edd, DEFAULT, PPOConfig(seed=0), CloneConfig(episodes=30))   # about 1,100 decisions
    assert stats["accuracy"] > 0.95


def test_ppo_continues_from_the_given_model():
    model, _ = warm_start(edd, SMALL, PPOConfig(seed=0), CloneConfig(episodes=5, epochs=2))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trained, _ = train_ppo(FlowTwinEnv(SMALL), PPOConfig(total_steps=128, rollout_len=64, minibatch=32, epochs=1, seed=0),
                           log=None, model=model)
    assert trained is model
    assert any(not torch.equal(before[k], v) for k, v in trained.state_dict().items())


def test_training_script_warm_starts_from_the_slack_rule(tmp_path):
    subprocess.run([sys.executable, "scripts/train_ppo.py", "--steps", "512", "--episodes", "1", "--w-tardiness", "2.5",
                    "--warm-start-slack", "360", "--clone-episodes", "5", "--out", str(tmp_path), "--models", str(tmp_path)],
                   cwd=ROOT, check=True, capture_output=True)
    saved = json.loads((tmp_path / "ppo_vs_heuristics.json").read_text(encoding="utf-8"))
    assert {"ppo warm start (before rl)", "ppo (ours)"} <= set(saved)
    assert json.loads((tmp_path / "ppo_warm_start.json").read_text(encoding="utf-8"))["decisions"] > 0
    assert (tmp_path / "ppo_warm_start.pt").exists()
    refused = subprocess.run([sys.executable, "scripts/train_ppo.py", "--warm-start-slack", "360", "--speed-only"],
                             cwd=ROOT, capture_output=True)
    assert refused.returncode == 2                 # argparse error before any training
