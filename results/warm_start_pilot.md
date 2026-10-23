# Warm-start pilot (w_tardiness = 2.5)

Protocol, fixed before any of the pilot's runs:

- **Change under test**: a warm start (`flowtwin.imitation`, opt-in). Before RL, PPO's actor-critic is fitted to the
  decisions of edd_slack, with the margin tuned exactly as in the study, on 200 demonstration shifts (seeds
  0–199): the actor to the rule's actions (cross-entropy, invalid actions masked) and the critic
  to the rule's discounted return at PPO's reward scale; 100 epochs, minibatch 64, Adam learning rate 0.0003.
  The 100 epochs were chosen before any pilot run, on held-out demonstration shifts 200–249 (training seed 0:
  held-out accuracy 0.917, 0.957 and 0.969 after 20, 50 and 100 epochs). PPO then trains for 100k decisions from
  those weights with its default hyperparameters; action space, observation, reward and line are unchanged.
- **Agent**: PPO (ours) only, training seeds 0, 1, 2. Double-DQN is left out: its ε-greedy schedule
  starts fully random, so a warm start would also need a schedule change, which is a hyperparameter change.
- **Evaluation**: seeds 1000–1029 for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study; the study's PPO with the same seed (no warm
  start); and the warm-started policy before RL.
- **Decision rule**: the warm start lets PPO beat the rules if the trained agent's paired Δ return against the best
  rule has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ of the trained agent against the policy before RL and
  against the study's PPO; for every policy its speed shares, how often its launch order equals the EDD order, how
  often its action equals the teacher's action in the same state and its most common launch order; the mean
  return of the last 100 training episodes; the clone's accuracy on its demonstrations.
- **Scope**: a pilot. It changes no published result.

## Summary

Best rule, re-selected on training seeds: **edd_slack(360)**, evaluation return 284.7 ± 127.8. Teacher: edd_slack(360).

| training seed | clone accuracy on its demonstrations | eval return before RL | eval return after RL | after − before RL, paired Δ | after RL − best rule, paired Δ | study PPO, eval return |
|---|---|---|---|---|---|---|
| 0 | 97.4% | 271.3 | 274.1 | +2.8 ± 17.9 | -10.5 ± 22.5 | 148.3 |
| 1 | 97.1% | 270.1 | 266.5 | -3.5 ± 20.0 | -18.1 ± 23.0 | 135.5 |
| 2 | 97.4% | 269.9 | 292.0 | +22.1 ± 27.8 | +7.3 ± 17.3 | 159.0 |

Decision rule: the warm start lets PPO beat the rules: **no** (0/3 seeds).

## Details

Evaluation seeds, mean ± std:

| policy | return | units_out | throughput_per_h | changeover_min | energy | tardiness_min | unmet_units | invalid_actions |
|---|---|---|---|---|---|---|---|---|
| fifo | 147.1 ± 210.1 | 663.7 ± 74.4 | 86.8 ± 5.1 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 78.4 ± 92.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| spt | 134.5 ± 182.6 | 663.7 ± 74.4 | 86.6 ± 4.9 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 83.4 ± 76.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd | 270.1 ± 160.1 | 663.7 ± 74.4 | 87.2 ± 4.7 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 29.1 ± 67.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_low | 159.6 ± 286.2 | 663.7 ± 74.4 | 70.9 ± 3.5 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 108.0 ± 123.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_slack(360) | 284.7 ± 127.8 | 663.7 ± 74.4 | 89.0 ± 5.6 | 64.0 ± 0.0 | 2066.1 ± 425.9 | 14.9 ± 39.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s0 | 148.3 ± 207.7 | 663.7 ± 74.4 | 87.9 ± 5.8 | 64.0 ± 0.0 | 2015.1 ± 360.2 | 72.5 ± 81.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s1 | 135.5 ± 208.9 | 663.7 ± 74.4 | 91.9 ± 5.2 | 64.0 ± 0.0 | 2280.7 ± 269.4 | 61.7 ± 89.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s2 | 159.0 ± 165.2 | 663.7 ± 74.4 | 90.3 ± 5.7 | 64.0 ± 0.0 | 2113.6 ± 293.3 | 62.3 ± 59.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO warm start, before RL s0 | 271.3 ± 172.8 | 663.7 ± 74.4 | 88.9 ± 5.4 | 65.1 ± 5.7 | 2060.8 ± 409.3 | 20.3 ± 58.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO warm start + RL s0 | 274.1 ± 170.4 | 663.7 ± 74.4 | 85.7 ± 6.9 | 65.1 ± 5.7 | 2004.4 ± 464.6 | 22.6 ± 54.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO warm start, before RL s1 | 270.1 ± 172.6 | 663.7 ± 74.4 | 88.9 ± 5.3 | 65.1 ± 5.7 | 2063.7 ± 408.0 | 20.6 ± 58.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO warm start + RL s1 | 266.5 ± 177.9 | 663.7 ± 74.4 | 84.9 ± 7.6 | 68.3 ± 10.9 | 2007.7 ± 455.4 | 24.8 ± 57.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO warm start, before RL s2 | 269.9 ± 172.5 | 663.7 ± 74.4 | 89.0 ± 5.4 | 65.1 ± 5.7 | 2071.6 ± 411.3 | 20.2 ± 58.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO warm start + RL s2 | 292.0 ± 123.7 | 663.7 ± 74.4 | 86.3 ± 8.5 | 64.0 ± 0.0 | 1992.8 ± 491.1 | 16.3 ± 35.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Paired against the best rule:

| paired vs edd_slack(360) | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| fifo | -137.6 ± 64.5 | +63.5 ± 28.2 | -141.3 ± 107.6 | 0/30 (8 tied) |
| spt | -150.2 ± 57.1 | +68.6 ± 23.4 | -141.3 ± 107.6 | 0/30 (9 tied) |
| edd | -14.5 ± 14.6 | +14.3 ± 11.7 | -141.3 ± 107.6 | 1/30 (20 tied) |
| edd_low | -125.1 ± 71.0 | +93.2 ± 35.0 | -718.7 ± 120.6 | 9/30 |
| PPO (ours) s0 | -136.4 ± 63.9 | +57.6 ± 24.5 | -51.0 ± 79.0 | 0/30 (8 tied) |
| PPO (ours) s1 | -149.2 ± 62.8 | +46.8 ± 27.1 | +214.6 ± 97.5 | 0/30 |
| PPO (ours) s2 | -125.6 ± 53.9 | +47.4 ± 18.8 | +47.5 ± 101.3 | 1/30 (10 tied) |
| PPO warm start, before RL s0 | -13.3 ± 23.3 | +5.4 ± 8.8 | -5.2 ± 22.8 | 2/30 (22 tied) |
| PPO warm start + RL s0 | -10.5 ± 22.5 | +7.7 ± 6.9 | -61.7 ± 65.1 | 17/30 (1 tied) |
| PPO warm start, before RL s1 | -14.6 ± 23.3 | +5.8 ± 8.8 | -2.3 ± 30.7 | 1/30 (20 tied) |
| PPO warm start + RL s1 | -18.1 ± 23.0 | +9.9 ± 7.8 | -58.3 ± 100.1 | 13/30 |
| PPO warm start, before RL s2 | -14.8 ± 23.3 | +5.4 ± 8.8 | +5.5 ± 26.1 | 1/30 (21 tied) |
| PPO warm start + RL s2 | +7.3 ± 17.3 | +1.5 ± 7.3 | -73.3 ± 96.3 | 18/30 (1 tied) |

After RL against the same seed's policy before RL and against the study's PPO:

| pair | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| PPO warm start + RL s0 vs PPO warm start, before RL s0 | +2.8 ± 17.9 | +2.3 ± 5.3 | -56.4 ± 70.2 | 19/30 (1 tied) |
| PPO warm start + RL s0 vs PPO (ours) s0 | +125.9 ± 63.5 | -49.9 ± 23.3 | -10.7 ± 108.2 | 25/30 |
| PPO warm start + RL s1 vs PPO warm start, before RL s1 | -3.5 ± 20.0 | +4.1 ± 6.9 | -56.0 ± 100.6 | 15/30 (1 tied) |
| PPO warm start + RL s1 vs PPO (ours) s1 | +131.0 ± 62.8 | -36.9 ± 23.9 | -273.0 ± 128.0 | 25/30 |
| PPO warm start + RL s2 vs PPO warm start, before RL s2 | +22.1 ± 27.8 | -3.9 ± 11.0 | -78.8 ± 95.6 | 19/30 (1 tied) |
| PPO warm start + RL s2 vs PPO (ours) s2 | +132.9 ± 49.6 | -45.9 ± 20.5 | -120.8 ± 115.7 | 27/30 |

| policy | speed share low / nominal / high | launch order = EDD order | action = edd_slack(360)'s | most common launch order | training return, last 100 episodes |
|---|---|---|---|---|---|
| PPO (ours) s0 | 0.0% / 91.9% / 8.1% | 6/30 | 45.4% | 0 → 1 → 2 (30/30) | 118.8 |
| PPO warm start, before RL s0 | 0.6% / 87.3% / 12.2% | 28/30 | 94.3% | 1 → 0 → 2 (6/30) | – |
| PPO warm start + RL s0 | 19.7% / 63.5% / 16.9% | 22/30 | 67.2% | 1 → 0 → 2 (7/30) | 280.9 |
| PPO (ours) s1 | 0.0% / 69.1% / 30.9% | 6/30 | 33.6% | 1 → 0 → 2 (30/30) | 144.8 |
| PPO warm start, before RL s1 | 0.6% / 87.2% / 12.2% | 27/30 | 92.5% | 1 → 0 → 2 (5/30) | – |
| PPO warm start + RL s1 | 25.5% / 54.7% / 19.9% | 23/30 | 56.1% | 1 → 0 → 2 (6/30) | 258.2 |
| PPO (ours) s2 | 0.0% / 83.6% / 16.4% | 5/30 | 51.7% | 0 → 2 → 1 (30/30) | 133.7 |
| PPO warm start, before RL s2 | 0.7% / 86.3% / 13.0% | 26/30 | 92.2% | 2 → 1 → 0 (6/30) | – |
| PPO warm start + RL s2 | 23.5% / 58.7% / 17.7% | 24/30 | 60.3% | 2 → 1 → 0 (8/30) | 264.7 |
