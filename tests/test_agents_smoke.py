"""Smoke tests: the PPO and Double-DQN trainers run end to end on a tiny budget."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flowtwin.config import LineConfig
from flowtwin.dqn import DQNConfig, policy_from_qnet, train_dqn
from flowtwin.env import FlowTwinEnv
from flowtwin.evaluate import evaluate
from flowtwin.ppo import PPOConfig, compute_gae, policy_from_model, train_ppo

SMALL = LineConfig(order_qty_range=(40, 60), horizon=600.0)


def test_gae_matches_manual_computation():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, lam=1.0)
    # with gamma=lam=1 and a terminal last step: adv_t = sum of future rewards - v_t
    assert np.allclose(ret, [3.0, 2.0, 1.0])
    assert np.allclose(adv, [2.5, 1.5, 0.5])


def test_ppo_trains_and_acts():
    env = FlowTwinEnv(SMALL)
    model, history = train_ppo(env, PPOConfig(total_steps=256, rollout_len=64, minibatch=32, epochs=2, seed=1), log=None)
    res = evaluate(policy_from_model(model), SMALL, n_episodes=2)
    assert res["return"].shape == (2,) and np.all(np.isfinite(res["return"]))


def test_dqn_trains_and_acts():
    env = FlowTwinEnv(SMALL)
    q, history = train_dqn(env, DQNConfig(total_steps=300, learning_starts=32, batch_size=16, target_update_every=50,
                                          eps_decay_steps=100, seed=1), log=None)
    res = evaluate(policy_from_qnet(q), SMALL, n_episodes=2)
    assert np.all(np.isfinite(res["return"]))


def _invalid_picks(policy, cfg, seeds=(0, 1)):
    env, total = FlowTwinEnv(cfg), 0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            done = term or trunc
        total += info["kpis"]["invalid_actions"]
    return total


def test_maskable_ppo_cross_check_runs(tmp_path):
    pytest.importorskip("sb3_contrib")                  # optional dependency; installed in CI
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/train_ppo.py", "--steps", "256", "--sb3", "--episodes", "1",
                    "--out", str(tmp_path), "--models", str(tmp_path)], cwd=root, check=True, capture_output=True)
    assert "| maskable ppo (sb3) |" in (tmp_path / "ppo_vs_heuristics.md").read_text(encoding="utf-8")


def test_masked_agents_never_pick_exhausted_products():
    model, _ = train_ppo(FlowTwinEnv(SMALL), PPOConfig(total_steps=256, rollout_len=64, minibatch=32, epochs=2, seed=1), log=None)
    q, _ = train_dqn(FlowTwinEnv(SMALL), DQNConfig(total_steps=300, learning_starts=32, batch_size=16, target_update_every=50,
                                                   eps_decay_steps=100, seed=1), log=None)
    assert _invalid_picks(policy_from_model(model), SMALL) == 0
    assert _invalid_picks(policy_from_model(model, deterministic=False), SMALL) == 0
    assert _invalid_picks(policy_from_qnet(q), SMALL) == 0
