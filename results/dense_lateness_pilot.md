# Dense lateness pilot (w_tardiness = 2.5)

Protocol, fixed before any of the pilot's runs:

- **Change under test**: `LineConfig.dense_tardiness`, for training only. Lateness is charged at every
  decision as it accrues instead of in one lump when an order completes. Every episode pays the same
  total, so all policies are evaluated on the unchanged reward.
- **Setting**: as in the sensitivity study at w_tardiness = 2.5: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions, training seeds 0, 1, 2; every other reward weight keeps its default.
- **Evaluation**: seeds 1000–1029 for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  0–29, then the rule with the highest mean training return), and the study's agent trained with
  the same seed on the standard signal ("dense-off").
- **Decision rule**: dense lateness teaches an agent to beat the rules if its paired Δ return against the best
  rule has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the dense-off agent with the same seed; for
  every agent its speed shares, how often its launch order equals the EDD order and its most common launch
  order on the evaluation seeds; the mean return of its last 100 training episodes.
- **Scope**: a pilot. It changes no published result; adopting dense lateness as the default would mean
  re-running the main tables and the study.

## Summary

Best rule, re-selected on training seeds: **edd_slack(360)**, evaluation return 284.7 ± 127.8.

| agent | eval return by seed, dense-off | eval return by seed, dense | dense − dense-off, paired Δ return by seed | launch order = EDD by seed, dense-off → dense | dense beats the rules |
|---|---|---|---|---|---|
| PPO (ours) | 148.3, 135.5, 159.0 | 155.1, 153.6, 154.0 | +6.9 ± 76.9, +18.1 ± 77.6, -5.1 ± 2.2 | 6, 6, 5 → 5, 4, 5 (of 30) | no (0/3 seeds) |
| Double-DQN | -7.4, 101.6, -12.9 | 154.4, 147.0, 147.0 | +161.8 ± 100.3, +45.4 ± 66.7, +159.9 ± 79.1 | 2, 4, 0 → 4, 7, 3 (of 30) | no (0/3 seeds) |

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
| Double-DQN s0 | -7.4 ± 280.2 | 663.7 ± 74.4 | 88.6 ± 7.6 | 106.7 ± 34.4 | 2361.2 ± 491.7 | 105.5 ± 104.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s1 | 101.6 ± 180.8 | 663.7 ± 74.4 | 86.1 ± 8.7 | 102.4 ± 51.9 | 2220.7 ± 448.7 | 71.1 ± 74.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s2 | -12.9 ± 246.0 | 663.7 ± 74.4 | 89.7 ± 7.7 | 120.5 ± 43.5 | 2495.1 ± 397.3 | 96.8 ± 97.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) dense s0 | 155.1 ± 177.0 | 663.7 ± 74.4 | 89.0 ± 6.9 | 64.0 ± 0.0 | 2119.2 ± 340.3 | 63.5 ± 63.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) dense s1 | 153.6 ± 182.2 | 663.7 ± 74.4 | 89.0 ± 6.2 | 64.0 ± 0.0 | 2076.9 ± 420.0 | 66.7 ± 64.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) dense s2 | 154.0 ± 169.0 | 663.7 ± 74.4 | 88.6 ± 5.2 | 64.0 ± 0.0 | 2013.9 ± 269.0 | 70.3 ± 64.9 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN dense s0 | 154.4 ± 179.5 | 663.7 ± 74.4 | 85.1 ± 6.1 | 80.0 ± 27.1 | 2036.3 ± 336.5 | 65.5 ± 63.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN dense s1 | 147.0 ± 193.8 | 663.7 ± 74.4 | 83.1 ± 6.5 | 74.7 ± 19.1 | 2005.5 ± 310.7 | 71.4 ± 74.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN dense s2 | 147.0 ± 176.3 | 663.7 ± 74.4 | 88.4 ± 5.8 | 69.3 ± 11.9 | 2096.6 ± 333.2 | 67.0 ± 67.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |

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
| Double-DQN s0 | -292.1 ± 89.4 | +90.6 ± 34.1 | +295.1 ± 112.0 | 0/30 |
| Double-DQN s1 | -183.1 ± 51.9 | +56.3 ± 22.2 | +154.7 ± 157.5 | 2/30 |
| Double-DQN s2 | -297.6 ± 80.7 | +82.0 ± 32.2 | +429.1 ± 95.3 | 0/30 |
| PPO (ours) dense s0 | -129.6 ± 53.2 | +48.6 ± 18.5 | +53.1 ± 91.2 | 6/30 |
| PPO (ours) dense s1 | -131.1 ± 52.4 | +51.8 ± 20.0 | +10.8 ± 55.7 | 3/30 (6 tied) |
| PPO (ours) dense s2 | -130.7 ± 55.2 | +55.4 ± 20.6 | -52.2 ± 91.0 | 1/30 (10 tied) |
| Double-DQN dense s0 | -130.3 ± 53.4 | +50.7 ± 19.2 | -29.8 ± 87.9 | 3/30 |
| Double-DQN dense s1 | -137.7 ± 56.9 | +56.6 ± 22.9 | -60.6 ± 114.6 | 5/30 |
| Double-DQN dense s2 | -137.7 ± 56.9 | +52.2 ± 21.4 | +30.5 ± 96.8 | 4/30 |

Dense against the dense-off agent with the same training seed:

| dense vs dense-off | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| PPO (ours) dense s0 | +6.9 ± 76.9 | -9.0 ± 27.4 | +104.1 ± 75.9 | 18/30 |
| PPO (ours) dense s1 | +18.1 ± 77.6 | +5.0 ± 30.6 | -203.8 ± 97.8 | 17/30 |
| PPO (ours) dense s2 | -5.1 ± 2.2 | +8.0 ± 3.2 | -99.6 ± 39.7 | 3/30 (11 tied) |
| Double-DQN dense s0 | +161.8 ± 100.3 | -39.9 ± 37.3 | -324.9 ± 116.8 | 24/30 |
| Double-DQN dense s1 | +45.4 ± 66.7 | +0.3 ± 27.2 | -215.3 ± 121.7 | 20/30 |
| Double-DQN dense s2 | +159.9 ± 79.1 | -29.8 ± 30.3 | -398.5 ± 76.2 | 25/30 |

| agent | speed share low / nominal / high | launch order = EDD order | most common launch order | training return, last 100 episodes |
|---|---|---|---|---|
| PPO (ours) s0 | 0.0% / 91.9% / 8.1% | 6/30 | 0 → 1 → 2 (30/30) | 118.8 |
| PPO (ours) dense s0 | 9.1% / 69.1% / 21.8% | 5/30 | 2 → 0 → 1 (30/30) | 128.8 |
| PPO (ours) s1 | 0.0% / 69.1% / 30.9% | 6/30 | 1 → 0 → 2 (30/30) | 144.8 |
| PPO (ours) dense s1 | 0.7% / 85.8% / 13.5% | 4/30 | 2 → 1 → 0 (30/30) | 152.4 |
| PPO (ours) s2 | 0.0% / 83.6% / 16.4% | 5/30 | 0 → 2 → 1 (30/30) | 133.7 |
| PPO (ours) dense s2 | 0.0% / 91.8% / 8.2% | 5/30 | 0 → 2 → 1 (30/30) | 105.8 |
| Double-DQN s0 | 18.2% / 35.1% / 46.7% | 2/30 | 0 → 2 → 1 → 2 (6/30) | -200.9 |
| Double-DQN dense s0 | 27.1% / 49.7% / 23.2% | 4/30 | 2 → 0 → 1 (10/30) | 48.4 |
| Double-DQN s1 | 25.7% / 35.8% / 38.6% | 4/30 | 2 → 1 → 2 → 0 (10/30) | 29.3 |
| Double-DQN dense s1 | 35.8% / 39.3% / 24.9% | 7/30 | 2 → 0 → 1 (15/30) | 5.9 |
| Double-DQN s2 | 19.0% / 21.3% / 59.7% | 0/30 | 0 → 2 → 1 → 2 → 1 (8/30) | -120.1 |
| Double-DQN dense s2 | 9.2% / 70.4% / 20.4% | 3/30 | 2 → 0 → 1 (25/30) | 72.7 |
