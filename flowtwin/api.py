"""FastAPI service exposing the twin.

    POST /twin/reset?seed=1        start a new episode (computes the plan)
    GET  /twin/state               observation, KPIs, orders, valid products
    POST /twin/step                body: {"action": int} or {"policy": "edd"|"fifo"|"spt"|"ppo"|"dqn"}
    GET  /twin/kpis                KPIs of the running episode
    GET  /twin/deviations          planned-vs-actual deviations and alerts
    POST /policy/act               body: {"policy": "...", "obs": [...] (optional)} -> action
    POST /twin/run?policy=edd      run the episode to completion with a policy

Set MODEL_PPO / MODEL_DQN to checkpoint paths to serve trained agents.
Run:  uvicorn flowtwin.api:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import DEFAULT
from .env import FlowTwinEnv
from .heuristics import HEURISTICS
from .mqtt_bridge import KpiPublisher
from .plan import Plan, compute_plan, deviations

@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        _load_models()
    except Exception as exc:  # torch optional at serve time
        print(f"[api] models not loaded: {exc}")
    reset(seed=0)
    yield
    _publisher.close()


app = FastAPI(title="FlowTwin", version="0.1.0", description="RL production scheduling on a packaging-line digital twin",
              lifespan=_lifespan)

_env = FlowTwinEnv(DEFAULT)
_obs: Optional[np.ndarray] = None
_plan: Optional[Plan] = None
_publisher = KpiPublisher()
_models = {}


def _load_models():
    import torch
    from .ppo import ActorCritic
    from .dqn import QNet
    obs_dim, n_act = _env.observation_space.shape[0], _env.action_space.n
    for name, cls in (("ppo", ActorCritic), ("dqn", QNet)):
        path = os.getenv(f"MODEL_{name.upper()}")
        if path and Path(path).exists():
            m = cls(obs_dim, n_act)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            _models[name] = m


def _policy_action(policy: str, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    if policy in HEURISTICS:
        return HEURISTICS[policy](_env, obs)
    if policy in _models:
        return _models[policy].act(obs, mask=mask)
    raise HTTPException(400, f"unknown policy '{policy}'. Available: {list(HEURISTICS) + list(_models)}")


class StepRequest(BaseModel):
    action: Optional[int] = None
    policy: Optional[str] = None


class ActRequest(BaseModel):
    policy: str = "edd"
    obs: Optional[List[float]] = None


@app.post("/twin/reset")
def reset(seed: int = 0):
    global _obs, _plan
    _obs, info = _env.reset(seed=seed)
    _plan = compute_plan(_env.cfg, _env.line.orders)
    _publisher.publish("kpis", info["kpis"])
    return {"seed": seed, "orders": info["orders"], "plan_completion": _plan.completion, "plan_makespan": _plan.makespan}


@app.get("/twin/state")
def state():
    _require()
    return {"obs": _obs.tolist(), "kpis": _env.line.kpis(), "valid_products": _env.line.valid_products(),
            "orders": {"quantity": _env.line.orders.quantity, "due": _env.line.orders.due},
            "stages": [{"name": s, "down": d, "busy": b, "product": p} for s, d, b, p in
                       zip(_env.cfg.stages, _env.line.down, _env.line.busy, _env.line.last_product)],
            "buffers": [len(b.items) for b in _env.line.buffers], "finished": _env.line.finished}


@app.post("/twin/step")
def step(req: StepRequest):
    global _obs
    _require()
    if _env.line.finished:
        raise HTTPException(409, "episode finished; call /twin/reset")
    action = req.action if req.action is not None else _policy_action(req.policy or "edd", _obs, _env.action_masks())
    _obs, reward, term, trunc, info = _env.step(int(action))
    dev = deviations(_env, _plan)
    _publisher.publish("kpis", info["kpis"])
    _publisher.publish("deviations", dev)
    return {"action": int(action), "decoded": _env.decode(int(action)), "reward": reward,
            "terminated": term, "truncated": trunc, "kpis": info["kpis"], "deviations": dev}


@app.post("/twin/run")
def run(policy: str = "edd", seed: Optional[int] = None):
    if seed is not None or _env.line is None:
        reset(seed=seed or 0)
    total = 0.0
    while not _env.line.finished:
        total += step(StepRequest(policy=policy))["reward"]
    return {"policy": policy, "return": total, "kpis": _env.line.kpis(), "deviations": deviations(_env, _plan)}


@app.get("/twin/kpis")
def kpis():
    _require()
    return _env.line.kpis()


@app.get("/twin/deviations")
def get_deviations(threshold_min: float = 45.0):
    _require()
    return deviations(_env, _plan, threshold_min)


@app.post("/policy/act")
def act(req: ActRequest):
    _require()
    obs = np.asarray(req.obs, dtype=np.float32) if req.obs is not None else _obs
    mask = _env.action_masks() if req.obs is None else None   # a client-supplied obs has no known mask
    a = _policy_action(req.policy, obs, mask)
    return {"policy": req.policy, "action": int(a), "decoded": _env.decode(int(a))}


@app.get("/health")
def health():
    return {"status": "ok", "mqtt": _publisher.enabled, "models": list(_models)}


def _require():
    if _env.line is None or _obs is None:
        raise HTTPException(409, "no episode; call /twin/reset")
