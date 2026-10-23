# Speed-only pilot (w_tardiness = 2.5)

Protocol, fixed before any of the pilot's runs:

- **Change under test**: the action space (`flowtwin.rule_actions.SPEED_ONLY`, opt-in). EDD chooses the product and
  each decision only picks the line speed (low, nominal or high): 3 actions. Observation, reward and line are
  unchanged. Every policy in this space launches in EDD order, and every EDD-order rule, the slack-aware one
  included, is one of them.
- **Setting**: as in the sensitivity study at w_tardiness = 2.5: PPO (ours) and Double-DQN with their default
  hyperparameters, 100k decisions, training seeds 0, 1, 2; every other reward weight keeps its default.
- **Evaluation**: seeds 1000–1029 for every policy; mean ± std and paired differences with 95%
  Student-t confidence intervals.
- **References**: the best rule, re-selected exactly as in the study (edd_slack margin tuned on training seeds
  0–29, then the rule with the highest mean training return), and the study's agent trained with
  the same seed, which picks products directly ("product actions").
- **Decision rule**: speed-only actions let an agent beat the rules if its paired Δ return against the best rule
  has a 95% CI entirely above 0 for at least 2 of its 3 training seeds (the study's rule).
- **Also reported, not part of the decision**: paired Δ against the product-action agent with the same seed; for
  every speed-only agent its speed shares, how often its speed equals the tuned edd_slack rule's choice in the
  same state, and the mean return of its last 100 training episodes.
- **Scope**: a pilot. It changes no published result; the published agents keep product actions.

## Summary

Best rule, re-selected on training seeds: **edd_slack(360)**, evaluation return 284.7 ± 127.8.

| agent | eval return by seed, product actions | eval return by seed, speed only | speed only − product actions, paired Δ return by seed | speed only − best rule, paired Δ return by seed | speed only beats the rules |
|---|---|---|---|---|---|
| PPO (ours) | 148.3, 135.5, 159.0 | 265.8, 269.7, 267.4 | +117.5 ± 64.0, +134.2 ± 60.5, +108.4 ± 62.4 | -18.9 ± 24.2, -14.9 ± 21.3, -17.2 ± 21.0 | no (0/3 seeds) |
| Double-DQN | -7.4, 101.6, -12.9 | 270.2, 250.0, 261.8 | +277.6 ± 81.0, +148.4 ± 55.8, +274.7 ± 59.7 | -14.5 ± 26.4, -34.7 ± 36.8, -22.8 ± 30.9 | no (0/3 seeds) |

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
| PPO (ours) speed s0 | 265.8 ± 182.0 | 663.7 ± 74.4 | 85.0 ± 4.6 | 64.0 ± 0.0 | 1872.2 ± 260.4 | 34.1 ± 72.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) speed s1 | 269.7 ± 174.9 | 663.7 ± 74.4 | 82.8 ± 4.3 | 64.0 ± 0.0 | 1790.1 ± 193.8 | 37.4 ± 74.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) speed s2 | 267.4 ± 175.3 | 663.7 ± 74.4 | 85.6 ± 4.8 | 64.0 ± 0.0 | 1877.8 ± 211.4 | 33.1 ± 73.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN speed s0 | 270.2 ± 172.2 | 663.7 ± 74.4 | 77.7 ± 5.9 | 64.0 ± 0.0 | 1613.6 ± 311.8 | 47.8 ± 67.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN speed s1 | 250.0 ± 177.9 | 663.7 ± 74.4 | 75.9 ± 4.9 | 64.0 ± 0.0 | 1569.5 ± 299.1 | 58.5 ± 70.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN speed s2 | 261.8 ± 157.6 | 663.7 ± 74.4 | 78.2 ± 6.9 | 64.0 ± 0.0 | 1755.3 ± 473.1 | 42.7 ± 54.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |

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
| PPO (ours) speed s0 | -18.9 ± 24.2 | +19.2 ± 14.0 | -193.8 ± 91.5 | 15/30 (4 tied) |
| PPO (ours) speed s1 | -14.9 ± 21.3 | +22.5 ± 14.7 | -276.0 ± 109.4 | 20/30 |
| PPO (ours) speed s2 | -17.2 ± 21.0 | +18.2 ± 14.0 | -188.2 ± 103.2 | 18/30 |
| Double-DQN speed s0 | -14.5 ± 26.4 | +32.9 ± 13.6 | -452.5 ± 75.1 | 14/30 |
| Double-DQN speed s1 | -34.7 ± 36.8 | +43.7 ± 18.0 | -496.6 ± 78.9 | 9/30 |
| Double-DQN speed s2 | -22.8 ± 30.9 | +27.8 ± 13.2 | -310.8 ± 78.0 | 11/30 |

Speed only against the product-action agent with the same training seed:

| speed only vs product actions | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| PPO (ours) speed s0 | +117.5 ± 64.0 | -38.4 ± 23.7 | -142.8 ± 59.0 | 28/30 |
| PPO (ours) speed s1 | +134.2 ± 60.5 | -24.3 ± 23.1 | -490.7 ± 42.3 | 26/30 |
| PPO (ours) speed s2 | +108.4 ± 62.4 | -29.2 ± 23.5 | -235.7 ± 62.8 | 25/30 |
| Double-DQN speed s0 | +277.6 ± 81.0 | -57.7 ± 30.3 | -747.6 ± 92.5 | 30/30 |
| Double-DQN speed s1 | +148.4 ± 55.8 | -12.6 ± 20.8 | -651.3 ± 109.7 | 27/30 |
| Double-DQN speed s2 | +274.7 ± 59.7 | -54.2 ± 23.5 | -739.9 ± 104.4 | 30/30 |

| agent | speed share low / nominal / high | speed = edd_slack(360)'s choice | training return, last 100 episodes |
|---|---|---|---|
| PPO (ours) s0 | 0.0% / 91.9% / 8.1% | – | 118.8 |
| PPO (ours) speed s0 | 12.6% / 85.3% / 2.0% | 71.6% | 261.9 |
| PPO (ours) s1 | 0.0% / 69.1% / 30.9% | – | 144.8 |
| PPO (ours) speed s1 | 23.8% / 76.2% / 0.0% | 66.8% | 262.1 |
| PPO (ours) s2 | 0.0% / 83.6% / 16.4% | – | 133.7 |
| PPO (ours) speed s2 | 7.8% / 92.2% / 0.0% | 76.4% | 258.6 |
| Double-DQN s0 | 18.2% / 35.1% / 46.7% | – | -200.9 |
| Double-DQN speed s0 | 55.2% / 44.2% / 0.7% | 28.2% | 263.9 |
| Double-DQN s1 | 25.7% / 35.8% / 38.6% | – | 29.3 |
| Double-DQN speed s1 | 67.0% / 29.5% / 3.5% | 20.0% | 247.9 |
| Double-DQN s2 | 19.0% / 21.3% / 59.7% | – | -120.1 |
| Double-DQN speed s2 | 65.0% / 15.8% / 19.2% | 18.3% | 284.5 |
