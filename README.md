# FlowTwin — RL production scheduling on a digital twin of a packaging line

FlowTwin is a discrete-event **digital twin** of a multi-product bottling/packaging
line (filler → capper → labeller → packer) with finite buffers, product
changeovers, stochastic breakdowns and energy-dependent line speed, exposed as a
**Gymnasium** environment. Scheduling decisions — *which product batch to launch
next, at which speed* — are taken by dispatching heuristics (FIFO, SPT, EDD) or by
reinforcement-learning agents written from scratch in **PyTorch** (PPO and
Double-DQN), and cross-checked against Stable-Baselines3. A **FastAPI** service
exposes the twin, publishes KPIs over **MQTT**, and a **Node-RED** flow computes an
OEE proxy and raises **planned-vs-actual deviation alerts**.

Status: **in development** (started September 2026). Results tables in `results/`
are produced by the scripts below; numbers are only reported once measured.

```
SimPy line model ──▶ Gymnasium env ──▶ heuristics / PPO / Double-DQN (PyTorch)
                          │
                    FastAPI service ──MQTT──▶ Node-RED: OEE proxy, deviation alerts
```

## Why

Production scheduling in real lines is dominated by changeover losses, breakdowns
and energy cost — the same trade-offs studied in simulation-based RL scheduling
research. FlowTwin rebuilds an Arena bottling-line coursework model in code so the
policy that runs it can be *learned* and compared fairly against the heuristics a
planner would use.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pytest -q                                              # environment, plan, agents, API tests

python scripts/run_baselines.py --episodes 30          # FIFO / SPT / EDD / random -> results/baselines.md
python scripts/train_ppo.py --steps 200000 --sb3       # our PPO vs SB3 PPO vs heuristics
python scripts/train_dqn.py --steps 200000             # Double-DQN vs heuristics
uvicorn flowtwin.api:app --reload                      # http://127.0.0.1:8000/docs
```

Full stack (API + Mosquitto + Node-RED): `docker compose up`, then import
`node-red/flows.json` at http://localhost:1880.

## The twin

| Element | Model |
|---|---|
| Stages | 4 in series, each with base time per unit and product-specific factor |
| Buffers | finite `simpy.Store`s (default capacity 30) → blocking when full |
| Changeover | fixed loss at *every* stage when the product variant switches |
| Breakdowns | exponential time-to-failure / time-to-repair per stage |
| Speed | 3 line-speed levels: faster processing, super-linear energy |
| Demand | one order per product with random quantity and due date; 12 h shift horizon |

**Decision** (one per launched batch): `Discrete(n_products × n_speeds)`.
**Observation**: buffer fills, inlet fill, stage down flags, product on stage 0,
remaining demand fraction, due-date urgency, elapsed time.
**Reward**: `+units − 0.5·changeover_min − 0.15·energy − 0.10·tardiness_min − 5·invalid`
with a terminal penalty per unmet unit. Weights live in `flowtwin/config.py`.

## Planned vs actual

`flowtwin/plan.py` runs the EDD schedule on a breakdown-free copy of the line to
obtain the *plan*; the live twin is the *actual*. `GET /twin/deviations` returns the
gap per product and a flow alert when output falls a batch behind plan — the same
signal a planner watches in a sawmill or packaging plant.

## Results

Populated by the scripts (mean ± std over 30 seeded episodes):

- `results/baselines.md` — FIFO / SPT / EDD / random
- `results/ppo_vs_heuristics.md` — from-scratch PPO, SB3 PPO, heuristics
- `results/dqn_vs_heuristics.md` — Double-DQN, heuristics

## Layout

```
flowtwin/config.py      line parameters and reward weights
flowtwin/line.py        SimPy production line (stages, buffers, breakdowns, loader handshake)
flowtwin/env.py         Gymnasium environment
flowtwin/heuristics.py  FIFO, SPT, EDD, random
flowtwin/ppo.py         PPO (actor-critic, GAE, clipped objective)
flowtwin/dqn.py         Double-DQN (replay buffer, target network)
flowtwin/plan.py        planned-vs-actual deviations
flowtwin/evaluate.py    seeded evaluation + Markdown/CSV tables
flowtwin/api.py         FastAPI service; flowtwin/mqtt_bridge.py MQTT publisher
node-red/flows.json     OEE proxy + deviation alerts
scripts/                run_baselines.py, train_ppo.py, train_dqn.py
tests/                  env / plan / agents / API tests
```

## Limitations (known, on purpose)

- Breakdowns block a stage *between* units, not mid-unit.
- One order per product per episode; no order arrivals during the day.
- The OEE in Node-RED is a proxy (availability × performance) — quality is not modelled.
- Agents are single-environment, CPU-only; no vectorised rollouts yet.

## License

MIT — see `LICENSE`.
