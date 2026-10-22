# FlowTwin — RL production scheduling on a digital twin of a packaging line

FlowTwin is a discrete-event **digital twin** of a multi-product bottling/packaging
line (filler → capper → labeller → packer) with finite buffers, product
changeovers, stochastic breakdowns and energy-dependent line speed, exposed as a
**Gymnasium** environment. Scheduling decisions — *which product batch to launch
next, at which speed* — are taken by dispatching heuristics (FIFO, SPT, EDD) or by
reinforcement-learning agents written from scratch in **PyTorch** (PPO and
Double-DQN); `train_ppo.py --sb3` adds a cross-check against Stable-Baselines3
(sb3-contrib MaskablePPO). A **FastAPI** service exposes the twin, publishes KPIs
over **MQTT**, and a **Node-RED** flow computes an OEE proxy and raises
**planned-vs-actual deviation alerts**.

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

python scripts/run_baselines.py --episodes 30          # FIFO / SPT / EDD / EDD-low / random -> results/baselines.md
python scripts/train_ppo.py --steps 100000 --sb3       # our PPO vs sb3-contrib MaskablePPO vs heuristics
python scripts/train_dqn.py --steps 100000             # Double-DQN vs heuristics
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

**Decision** (one per launched batch): `Discrete(n_products × n_speeds)`. Products with no
remaining demand are masked (`env.action_masks()`); the agents never choose them.
**Observation**: buffer fills, inlet fill, stage down flags, product on stage 0,
remaining demand fraction, due-date urgency, elapsed time.
**Reward**: `+units − 0.5·changeover_min − 0.15·energy − 0.10·tardiness_min − 5·invalid`
with a terminal penalty per unmet unit; the invalid term only applies to unmasked
policies. Weights live in `flowtwin/config.py`. The end of the shift terminates the
episode (the clock is observed), it is not a time-limit truncation.

## Planned vs actual

`flowtwin/plan.py` runs the EDD schedule on a breakdown-free copy of the line to
obtain the *plan*; the live twin is the *actual*. `GET /twin/deviations` returns the
gap per product and a flow alert when output falls a batch behind plan — the same
signal a planner watches in a sawmill or packaging plant.

## Results

30 evaluation episodes (seeds 1000–1029). Every policy sees the same orders and the
same breakdown times, so besides mean ± std over episodes the tables give **paired
differences**: the mean per-episode difference ± 95 % CI (Student t), which removes the
demand variance that dominates the std. Agents: 100k training decisions on CPU,
training seed 0, default reward weights. Rows are combined from
`results/ppo_vs_heuristics.md` and `results/dqn_vs_heuristics.md` (their heuristic rows
are identical); `results/baselines.md` adds the random policy.

| policy | return | units_out | throughput_per_h | changeover_min | energy | tardiness_min | unmet_units | invalid_actions |
|---|---|---|---|---|---|---|---|---|
| fifo | 335.2 ± 37.6 | 663.7 ± 74.4 | 86.8 ± 5.1 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 78.4 ± 92.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| spt | 334.7 ± 40.3 | 663.7 ± 74.4 | 86.6 ± 4.9 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 83.4 ± 76.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd | 340.1 ± 40.4 | 663.7 ± 74.4 | 87.2 ± 4.7 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 29.1 ± 67.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_low | 418.8 ± 46.8 | 663.7 ± 74.4 | 70.9 ± 3.5 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 108.0 ± 123.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| ppo (ours) | 411.4 ± 47.8 | 663.7 ± 74.4 | 70.8 ± 3.6 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 182.7 ± 122.1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| double-dqn | 390.7 ± 55.5 | 663.7 ± 74.4 | 69.9 ± 4.0 | 90.7 ± 28.7 | 1353.6 ± 154.5 | 247.0 ± 148.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |

| paired vs edd | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| edd_low | +78.7 ± 2.9 | +78.9 ± 25.2 | -577.4 ± 24.6 | 30/30 |
| ppo (ours) | +71.3 ± 4.2 | +153.5 ± 35.1 | -577.4 ± 24.6 | 30/30 |
| double-dqn | +50.6 ± 9.1 | +217.8 ± 45.9 | -571.2 ± 23.5 | 29/30 |

| paired vs edd_low | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| ppo (ours) | -7.5 ± 3.0 | +74.7 ± 29.9 | +0.0 ± 0.0 | 2/30 (6 tied) |
| double-dqn | -28.2 ± 8.3 | +139.0 ± 43.3 | +6.2 ± 3.1 | 1/30 |

**Reading the results**

- With the default weights energy dominates the objective: for EDD it costs
  0.15 × 1924.8 = 288.7 of the return, tardiness only 0.10 × 29.1 = 2.9. Running the
  same EDD sequence at low speed (`edd_low`) raises the return by +78.7 ± 2.9 on 30/30
  episodes while tardiness grows by +78.9 ± 25.2 min. Any policy that slows the line
  therefore "beats EDD"; `edd_low` is the fair reference.
- **PPO (ours)** beats EDD by +71.3 ± 4.2 on 30/30 episodes and ends 7.5 ± 3.0 below
  `edd_low`. It runs every batch at low speed and launches the products in one fixed
  order (2 → 0 → 1 on all 30 episodes, `scripts/inspect_agent.py`) whatever the due
  dates, so its tardiness is 182.7 min against 29.1 for EDD. It learned the energy
  lever, not scheduling.
- **Double-DQN** beats EDD by +50.6 ± 9.1 (29/30) and ends 28.2 ± 8.3 below `edd_low`:
  98.6 % of batches at low speed, mostly the same 2 → 0 → 1 order, plus extra product
  switches (changeover 90.7 vs 64.0 min).
- **MaskablePPO (sb3-contrib)**: the like-for-like cross-check (same masks, same reward
  scale) is built into `train_ppo.py --sb3` but has not been run yet; it needs `sb3-contrib`
  installed.
- No agent uses the due dates. With `w_tardiness = 0.10` lateness is about 1 % of the
  return, so a better sequence is worth a few points per episode against an
  episode-to-episode std of 40–56.

**Training-seed robustness** (same evaluation episodes, `--seed 1` / `--seed 2`):

| agent | training seed | return | Δ return vs edd | Δ return vs edd_low |
|---|---|---|---|---|
| PPO (ours) | 0 | 411.4 ± 47.8 | +71.3 ± 4.2 | -7.5 ± 3.0 |
| PPO (ours) | 1 | 411.4 ± 47.8 | +71.3 ± 4.2 | -7.5 ± 3.0 |
| PPO (ours) | 2 | 410.9 ± 43.8 | +70.8 ± 3.5 | -8.0 ± 3.2 |
| Double-DQN | 0 | 390.7 ± 55.5 | +50.6 ± 9.1 | -28.2 ± 8.3 |
| Double-DQN | 1 | 373.2 ± 60.2 | +33.1 ± 12.8 | -45.6 ± 11.4 |
| Double-DQN | 2 | 384.6 ± 51.2 | +44.5 ± 7.4 | -34.2 ± 6.6 |

PPO seeds 0 and 1 converge to the same fixed schedule (different weights, identical
actions), hence identical rows; seed 2 learns the order 1 → 0 → 2.

**First runs and what was fixed.** The first 100k-step runs (commit `fd114d2`) lost to
EDD: PPO 297.3 ± 36.5, Double-DQN 335.2 ± 57.1; SB3 PPO reached 382.3 ± 44.7 only by
slowing the line. PPO picked an already finished product 23.4 times per episode and let
the EDD fallback schedule for it (−5 each), and at its first update the critic's
gradient norm was 105.1 against 0.45 for the policy on the shared trunk, so the 0.5
gradient-norm clip cut that update to under 0.5 % of its size. Fixes: invalid-action masking,
learner-side reward scaling (×0.01, reported returns unchanged), the end of the shift as
a true termination, the `edd_low` baseline and paired CIs. The random baseline is now
seeded, which changed its row in `results/baselines.md`.

## Layout

```
flowtwin/config.py      line parameters and reward weights
flowtwin/line.py        SimPy production line (stages, buffers, breakdowns, loader handshake)
flowtwin/env.py         Gymnasium environment
flowtwin/heuristics.py  FIFO, SPT, EDD, EDD at low speed, random
flowtwin/ppo.py         PPO (actor-critic, GAE, clipped objective)
flowtwin/dqn.py         Double-DQN (replay buffer, target network)
flowtwin/plan.py        planned-vs-actual deviations
flowtwin/evaluate.py    seeded evaluation + Markdown/CSV tables
flowtwin/api.py         FastAPI service; flowtwin/mqtt_bridge.py MQTT publisher
node-red/flows.json     OEE proxy + deviation alerts
scripts/                run_baselines.py, train_ppo.py, train_dqn.py, inspect_agent.py
tests/                  env / plan / agents / API tests
```

## Limitations (known, on purpose)

- Breakdowns block a stage *between* units, not mid-unit.
- One order per product per episode; no order arrivals during the day.
- The OEE in Node-RED is a proxy (availability × performance) — quality is not modelled.
- Agents are single-environment, CPU-only; no vectorised rollouts yet.
- The main tables use one training run per agent (100k decisions); three training seeds
  are reported above. PPO's training return plateaus after about 30k decisions.
- The observation gives remaining demand as a fraction of each order, not as absolute
  work, so an agent cannot compute an order's slack exactly.

## License

MIT — see `LICENSE`.
