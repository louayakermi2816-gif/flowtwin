# Order-first pilot (w_tardiness = 2.5)

Protocol, fixed before any of the pilot's runs:

- **Change under test**: the action space (`flowtwin.order_actions`, opt-in). The first decision of a shift picks one
  of the six launch orders (nothing is launched, reward 0); every later decision picks the line speed of a batch
  of the next product in that order that still has demand. The observation adds a planning flag and the chosen
  order; reward and line are unchanged. Planning the EDD order with the slack rule's speeds reproduces edd_slack
  exactly, so the best rule is one of the policies an agent can learn.
- **Setting**: as in the sensitivity study at w_tardiness = 2.5: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions (planning decisions included), training seeds 0, 1, 2; every other
  reward weight keeps its default.
- **Evaluation**: seeds 1000–1029 for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  0–29, then the rule with the highest mean training return), and the study's agent trained with
  the same seed, which picks products directly ("product actions").
- **Decision rule**: order-first actions let an agent beat the rules if its paired Δ return against the best rule
  has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the product-action agent with the same seed; for
  every agent how often its launch order equals the EDD order and its most common launch order, its speed shares,
  how often its action equals the tuned edd_slack rule's action in the same state, and the mean return of its
  last 100 training episodes.
- **Scope**: a pilot. It changes no published result; the published agents keep product actions.

## Summary

Best rule, re-selected on training seeds: **edd_slack(360)**, evaluation return 284.7 ± 127.8.

| agent | eval return by seed, product actions | eval return by seed, order first | order first − best rule, paired Δ return by seed | launch order = EDD by seed, product actions → order first | order first beats the rules |
|---|---|---|---|---|---|
| PPO (ours) | 148.3, 135.5, 159.0 | 162.0, 194.1, 156.0 | -122.6 ± 50.3, -90.6 ± 43.8, -128.7 ± 53.0 | 6, 6, 5 → 5, 8, 5 (of 30) | no (0/3 seeds) |
| Double-DQN | -7.4, 101.6, -12.9 | 209.4, 244.3, 262.9 | -75.2 ± 51.2, -40.4 ± 33.1, -21.8 ± 24.4 | 2, 4, 0 → 18, 17, 18 (of 30) | no (0/3 seeds) |

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
| PPO (ours) order-first s0 | 162.0 ± 143.8 | 663.7 ± 74.4 | 93.7 ± 8.4 | 64.0 ± 0.0 | 2406.1 ± 408.2 | 43.5 ± 46.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) order-first s1 | 194.1 ± 160.0 | 663.7 ± 74.4 | 93.1 ± 7.9 | 64.0 ± 0.0 | 2329.7 ± 473.4 | 35.3 ± 54.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) order-first s2 | 156.0 ± 161.6 | 663.7 ± 74.4 | 92.0 ± 6.0 | 64.0 ± 0.0 | 2236.8 ± 356.6 | 56.1 ± 55.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN order-first s0 | 209.4 ± 228.7 | 663.7 ± 74.4 | 74.7 ± 4.0 | 64.0 ± 0.0 | 1529.9 ± 229.5 | 77.1 ± 93.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN order-first s1 | 244.3 ± 167.1 | 663.7 ± 74.4 | 77.6 ± 5.8 | 64.0 ± 0.0 | 1661.9 ± 365.1 | 55.3 ± 60.9 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN order-first s2 | 262.9 ± 159.1 | 663.7 ± 74.4 | 84.1 ± 5.8 | 64.0 ± 0.0 | 1862.7 ± 305.0 | 35.8 ± 63.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |

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
| PPO (ours) order-first s0 | -122.6 ± 50.3 | +28.7 ± 16.0 | +340.1 ± 107.0 | 4/30 |
| PPO (ours) order-first s1 | -90.6 ± 43.8 | +20.4 ± 15.2 | +263.6 ± 176.3 | 1/30 (10 tied) |
| PPO (ours) order-first s2 | -128.7 ± 53.0 | +41.2 ± 17.8 | +170.7 ± 83.9 | 1/30 (4 tied) |
| Double-DQN order-first s0 | -75.2 ± 51.2 | +62.3 ± 24.9 | -536.2 ± 95.1 | 9/30 |
| Double-DQN order-first s1 | -40.4 ± 33.1 | +40.4 ± 15.6 | -404.1 ± 77.8 | 11/30 |
| Double-DQN order-first s2 | -21.8 ± 24.4 | +20.9 ± 12.5 | -203.4 ± 99.0 | 15/30 (1 tied) |

Order first against the product-action agent with the same training seed:

| order first vs product actions | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| PPO (ours) order-first s0 | +13.8 ± 75.1 | -29.0 ± 26.5 | +391.1 ± 106.1 | 17/30 |
| PPO (ours) order-first s1 | +58.6 ± 76.7 | -26.4 ± 27.1 | +49.0 ± 167.6 | 20/30 |
| PPO (ours) order-first s2 | -3.0 ± 4.4 | -6.2 ± 2.8 | +123.2 ± 39.1 | 13/30 (5 tied) |
| Double-DQN order-first s0 | +216.9 ± 74.8 | -28.3 ± 27.9 | -831.3 ± 123.3 | 27/30 |
| Double-DQN order-first s1 | +142.7 ± 58.6 | -15.9 ± 23.6 | -558.8 ± 128.2 | 24/30 |
| Double-DQN order-first s2 | +275.8 ± 74.2 | -61.1 ± 28.5 | -632.5 ± 73.1 | 29/30 |

| agent | speed share low / nominal / high | launch order = EDD order | most common launch order | action = edd_slack(360)'s | training return, last 100 episodes |
|---|---|---|---|---|---|
| PPO (ours) s0 | 0.0% / 91.9% / 8.1% | 6/30 | 0 → 1 → 2 (30/30) | – | 118.8 |
| PPO (ours) order-first s0 | 17.3% / 31.9% / 50.8% | 5/30 | 2 → 0 → 1 (30/30) | 34.8% | 142.7 |
| PPO (ours) s1 | 0.0% / 69.1% / 30.9% | 6/30 | 1 → 0 → 2 (30/30) | – | 144.8 |
| PPO (ours) order-first s1 | 1.1% / 63.2% / 35.8% | 8/30 | 2 → 0 → 1 (16/30) | 41.9% | 166.4 |
| PPO (ours) s2 | 0.0% / 83.6% / 16.4% | 5/30 | 0 → 2 → 1 (30/30) | – | 133.7 |
| PPO (ours) order-first s2 | 0.0% / 72.2% / 27.8% | 5/30 | 0 → 2 → 1 (30/30) | 52.7% | 150.9 |
| Double-DQN s0 | 18.2% / 35.1% / 46.7% | 2/30 | 0 → 2 → 1 → 2 (6/30) | – | -200.9 |
| Double-DQN order-first s0 | 74.6% / 21.6% / 3.8% | 18/30 | 2 → 0 → 1 (10/30) | 12.7% | 267.5 |
| Double-DQN s1 | 25.7% / 35.8% / 38.6% | 4/30 | 2 → 1 → 2 → 0 (10/30) | – | 29.3 |
| Double-DQN order-first s1 | 56.7% / 36.8% / 6.5% | 17/30 | 0 → 1 → 2 (10/30) | 25.4% | 236.6 |
| Double-DQN s2 | 19.0% / 21.3% / 59.7% | 0/30 | 0 → 2 → 1 → 2 → 1 (8/30) | – | -120.1 |
| Double-DQN order-first s2 | 19.9% / 74.8% / 5.3% | 18/30 | 1 → 2 → 0 (11/30) | 47.3% | 258.9 |
