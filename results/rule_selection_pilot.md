# Rule-selection pilot (w_tardiness = 2.5)

Protocol, fixed before any of the pilot's runs:

- **Change under test**: the action space (`flowtwin.rule_actions`, opt-in). Each decision picks a dispatching
  rule (fifo, spt, edd), which chooses the product, and a line speed: 9 actions, as before. Observation,
  reward and line are unchanged. Following EDD order is one repeated action, and every edd_slack decision
  is an available action.
- **Setting**: as in the sensitivity study at w_tardiness = 2.5: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions, training seeds 0, 1, 2; every other reward weight keeps its default.
- **Evaluation**: seeds 1000–1029 for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  0–29, then the rule with the highest mean training return), and the study's agent trained with
  the same seed, which picks products directly ("product actions").
- **Decision rule**: rule selection lets an agent beat the rules if its paired Δ return against the best rule
  has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the product-action agent with the same seed;
  for every agent its share of decisions per rule (rule-action agents only), speed shares, how often its
  launch order equals the EDD order, its most common launch order and the mean return of its last
  100 training episodes.
- **Scope**: a pilot. It changes no published result; the published agents keep product actions.

## Summary

Best rule, re-selected on training seeds: **edd_slack(360)**, evaluation return 284.7 ± 127.8.

| agent | eval return by seed, product actions | eval return by seed, rule actions | rule − product actions, paired Δ return by seed | launch order = EDD by seed, product → rule actions | rule actions beat the rules |
|---|---|---|---|---|---|
| PPO (ours) | 148.3, 135.5, 159.0 | 134.6, 153.3, 151.4 | -13.6 ± 79.9, +17.8 ± 20.5, -7.7 ± 39.9 | 6, 6, 5 → 4, 6, 4 (of 30) | no (0/3 seeds) |
| Double-DQN | -7.4, 101.6, -12.9 | 166.7, 236.7, 168.5 | +174.1 ± 81.7, +135.1 ± 53.0, +181.4 ± 91.0 | 2, 4, 0 → 7, 16, 12 (of 30) | no (0/3 seeds) |

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
| PPO (ours) rules s0 | 134.6 ± 182.3 | 663.7 ± 74.4 | 86.6 ± 4.9 | 64.0 ± 0.0 | 1929.6 ± 219.9 | 83.1 ± 76.1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) rules s1 | 153.3 ± 195.5 | 663.7 ± 74.4 | 88.1 ± 7.0 | 64.0 ± 0.0 | 2036.7 ± 440.2 | 69.2 ± 74.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) rules s2 | 151.4 ± 161.6 | 663.7 ± 74.4 | 89.3 ± 5.4 | 64.0 ± 0.0 | 2083.9 ± 385.5 | 67.1 ± 58.1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN rules s0 | 166.7 ± 170.9 | 663.7 ± 74.4 | 87.2 ± 9.2 | 89.6 ± 56.3 | 2276.2 ± 477.4 | 44.3 ± 58.9 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN rules s1 | 236.7 ± 136.3 | 663.7 ± 74.4 | 85.7 ± 7.7 | 76.8 ± 37.5 | 2146.2 ± 475.4 | 26.7 ± 45.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN rules s2 | 168.5 ± 234.1 | 663.7 ± 74.4 | 81.6 ± 7.9 | 98.1 ± 48.8 | 1956.6 ± 393.4 | 61.1 ± 88.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |

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
| PPO (ours) rules s0 | -150.0 ± 57.1 | +68.2 ± 23.3 | -136.4 ± 105.1 | 0/30 (9 tied) |
| PPO (ours) rules s1 | -131.4 ± 61.6 | +54.3 ± 21.9 | -29.4 ± 122.4 | 0/30 (8 tied) |
| PPO (ours) rules s2 | -133.3 ± 59.1 | +52.2 ± 22.1 | +17.8 ± 80.0 | 2/30 (9 tied) |
| Double-DQN rules s0 | -118.0 ± 41.8 | +29.5 ± 14.1 | +210.2 ± 150.4 | 4/30 |
| Double-DQN rules s1 | -48.0 ± 29.9 | +11.8 ± 11.1 | +80.2 ± 109.2 | 8/30 |
| Double-DQN rules s2 | -116.2 ± 68.9 | +46.2 ± 25.1 | -109.4 ± 127.7 | 6/30 |

Rule actions against the product-action agent with the same training seed:

| rule vs product actions | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| PPO (ours) rules s0 | -13.6 ± 79.9 | +10.6 ± 29.4 | -85.4 ± 68.0 | 12/30 (7 tied) |
| PPO (ours) rules s1 | +17.8 ± 20.5 | +7.5 ± 11.9 | -244.0 ± 97.9 | 15/30 |
| PPO (ours) rules s2 | -7.7 ± 39.9 | +4.8 ± 16.7 | -29.6 ± 68.0 | 8/30 (8 tied) |
| Double-DQN rules s0 | +174.1 ± 81.7 | -61.1 ± 28.5 | -84.9 ± 143.7 | 28/30 |
| Double-DQN rules s1 | +135.1 ± 53.0 | -44.5 ± 19.8 | -74.5 ± 93.2 | 27/30 |
| Double-DQN rules s2 | +181.4 ± 91.0 | -35.8 ± 34.4 | -538.5 ± 106.4 | 27/30 |

| agent | rule share fifo / spt / edd | speed share low / nominal / high | launch order = EDD order | most common launch order | training return, last 100 episodes |
|---|---|---|---|---|---|
| PPO (ours) s0 | – | 0.0% / 91.9% / 8.1% | 6/30 | 0 → 1 → 2 (30/30) | 118.8 |
| PPO (ours) rules s0 | 0.0% / 100.0% / 0.0% | 0.0% / 99.5% / 0.5% | 4/30 | 2 → 0 → 1 (13/30) | 130.3 |
| PPO (ours) s1 | – | 0.0% / 69.1% / 30.9% | 6/30 | 1 → 0 → 2 (30/30) | 144.8 |
| PPO (ours) rules s1 | 76.8% / 18.2% / 5.0% | 5.0% / 83.1% / 11.9% | 6/30 | 0 → 1 → 2 (30/30) | 145.8 |
| PPO (ours) s2 | – | 0.0% / 83.6% / 16.4% | 5/30 | 0 → 2 → 1 (30/30) | 133.7 |
| PPO (ours) rules s2 | 0.0% / 100.0% / 0.0% | 0.0% / 86.0% / 14.0% | 4/30 | 2 → 0 → 1 (13/30) | 121.9 |
| Double-DQN s0 | – | 18.2% / 35.1% / 46.7% | 2/30 | 0 → 2 → 1 → 2 (6/30) | -200.9 |
| Double-DQN rules s0 | 49.0% / 34.2% / 16.8% | 26.7% / 28.8% / 44.5% | 7/30 | 0 → 1 → 2 (9/30) | 149.8 |
| Double-DQN s1 | – | 25.7% / 35.8% / 38.6% | 4/30 | 2 → 1 → 2 → 0 (10/30) | 29.3 |
| Double-DQN rules s1 | 30.0% / 17.0% / 53.0% | 34.1% / 28.9% / 36.9% | 16/30 | 0 → 1 → 2 (11/30) | 207.5 |
| Double-DQN s2 | – | 19.0% / 21.3% / 59.7% | 0/30 | 0 → 2 → 1 → 2 → 1 (8/30) | -120.1 |
| Double-DQN rules s2 | 35.8% / 19.3% / 44.9% | 35.5% / 43.9% / 20.6% | 12/30 | 0 → 1 → 2 (10/30) | 178.5 |
