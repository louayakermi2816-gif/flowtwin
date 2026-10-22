import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from flowtwin import api


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_reset_state_step_deviations(client):
    r = client.post("/twin/reset", params={"seed": 3}); assert r.status_code == 200
    assert len(r.json()["plan_completion"]) == 3
    s = client.get("/twin/state").json()
    assert len(s["stages"]) == 4 and s["finished"] is False
    st = client.post("/twin/step", json={"policy": "edd"}).json()
    assert "reward" in st and "deviations" in st
    assert set(st["kpis"]["oee"]) == {"availability", "performance", "quality", "oee"}
    d = client.get("/twin/deviations").json()
    assert d["actual_units"] >= 0
    a = client.post("/policy/act", json={"policy": "spt"}).json()
    assert 0 <= a["action"] < 9


def test_run_to_completion(client):
    out = client.post("/twin/run", params={"policy": "fifo", "seed": 4}).json()
    assert out["kpis"]["finished"] is True
    assert client.post("/twin/step", json={"policy": "edd"}).status_code == 409


def test_served_agents_act_greedily(client):
    torch = pytest.importorskip("torch")
    from flowtwin.ppo import ActorCritic
    torch.manual_seed(0)
    api._models["ppo"] = ActorCritic(api._env.observation_space.shape[0], api._env.action_space.n)
    try:
        client.post("/twin/reset", params={"seed": 3})
        actions = {client.post("/policy/act", json={"policy": "ppo"}).json()["action"] for _ in range(20)}
        assert len(actions) == 1          # same state, same action: greedy, as in the evaluation
    finally:
        api._models.pop("ppo", None)
