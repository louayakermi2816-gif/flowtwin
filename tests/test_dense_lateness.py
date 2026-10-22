"""Dense lateness signal (cfg.dense_tardiness): charging lateness while it accrues moves the
penalty earlier in the episode without changing what any episode pays in total."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flowtwin.config import DEFAULT
from flowtwin.env import FlowTwinEnv
from flowtwin.heuristics import HEURISTICS

ROOT = Path(__file__).resolve().parents[1]
SPARSE = replace(DEFAULT, w_tardiness=2.5)
DENSE = replace(SPARSE, dense_tardiness=True)


def step_rewards(cfg, policy, seed):
    env = FlowTwinEnv(cfg)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    rewards, done = [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        rewards.append(r); done = term or trunc
    return np.array(rewards), info["kpis"]


def test_dense_lateness_is_opt_in():
    assert DEFAULT.dense_tardiness is False


@pytest.mark.parametrize("name", ["edd", "fifo", "random"])
def test_dense_lateness_keeps_every_episode_total(name):
    for seed in range(1000, 1005):
        sparse, k_sparse = step_rewards(SPARSE, HEURISTICS[name], seed)
        dense, k_dense = step_rewards(DENSE, HEURISTICS[name], seed)
        assert k_dense == k_sparse                                       # same schedule, same KPIs
        assert np.isclose(dense.sum(), sparse.sum(), rtol=0, atol=1e-6)  # same return up to float noise


def test_dense_lateness_is_charged_while_it_accrues():
    moved = 0
    for seed in range(1000, 1010):
        sparse, _ = step_rewards(SPARSE, HEURISTICS["fifo"], seed)
        dense, _ = step_rewards(DENSE, HEURISTICS["fifo"], seed)
        assert (np.cumsum(dense) <= np.cumsum(sparse) + 1e-6).all()     # never later than on completion
        moved += not np.allclose(dense, sparse)
    assert moved > 0                                                     # and earlier when an order runs late


def test_training_scripts_learn_from_the_dense_signal(tmp_path):
    torch = pytest.importorskip("torch")

    def train(name, *extra):
        out = tmp_path / name
        subprocess.run([sys.executable, "scripts/train_ppo.py", "--steps", "512", "--episodes", "1", "--w-tardiness",
                        "2.5", *extra, "--out", str(out), "--models", str(out)], cwd=ROOT, check=True, capture_output=True)
        weights = torch.load(out / "ppo.pt", weights_only=True)
        rules = json.loads((out / "ppo_vs_heuristics.json").read_text(encoding="utf-8"))
        del rules["ppo (ours)"]
        return weights, rules

    (w_sparse, rules_sparse), (w_dense, rules_dense) = train("sparse"), train("dense", "--dense-tardiness")
    assert any(not torch.equal(w_sparse[k], w_dense[k]) for k in w_sparse)   # one PPO update on a different signal
    assert rules_dense == rules_sparse                                       # evaluated on the standard reward
