"""Smoke tests: the PPO and Double-DQN trainers run end to end on a tiny budget."""
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
