# Lateness-weight sensitivity study

Protocol, fixed before any of the study's runs:

- **Weights**: w_tardiness in {0.1, 1, 2.5}; every other reward weight keeps its default.
- **Rules**: fifo, spt, edd, edd_low and edd_slack. The edd_slack margin is tuned per weight on training
  seeds 0–29 over {0, 30, 60, 120, 180, 240, 360, 480, 720} minutes. The best rule per weight is the
  one with the highest mean return on those training seeds.
- **Agents**: PPO (ours) and Double-DQN with their default hyperparameters, 100k decisions, training
  seeds 0, 1, 2.
- **Evaluation**: seeds 1000–1029 for every policy (same orders, same breakdowns); mean ± std
  and paired differences against the best rule with 95% Student-t confidence intervals.
- **Decision rule**: an agent beats the rules at a weight if its paired Δ return against the best
  rule has a 95% CI entirely above 0 for at least 2 of its 3 training seeds.

## Summary

| w_tardiness | best rule, eval return | PPO (ours), seed means | Double-DQN, seed means | PPO beats the rules | Double-DQN beats the rules |
|---|---|---|---|---|---|
| 0.1 | edd_low: 418.8 ± 46.8 | 410.9 to 411.4 | 373.2 to 390.7 | no (0/3 seeds) | no (0/3 seeds) |
| 1 | edd: 313.9 ± 66.9 | 255.5 to 268.1 | 177.1 to 212.3 | no (0/3 seeds) | no (0/3 seeds) |
| 2.5 | edd_slack(360): 284.7 ± 127.8 | 135.5 to 159.0 | -12.9 to 101.6 | no (0/3 seeds) | no (0/3 seeds) |

## w_tardiness = 0.1

edd_slack margin tuned on training seeds: **0 min** (mean training return by margin: 0: 332.1, 30: 330.2, 60: 327.6, 120: 328.9, 180: 327.1, 240: 323.9, 360: 323.0, 480: 323.0, 720: 323.0).
Best rule on training seeds: **edd_low** (mean training return: fifo: 351.3, spt: 349.4, edd: 360.0, edd_low: 438.3, edd_slack(0): 332.1).

Evaluation seeds, mean ± std:

| policy | return | units_out | throughput_per_h | changeover_min | energy | tardiness_min | unmet_units | invalid_actions |
|---|---|---|---|---|---|---|---|---|
| fifo | 335.2 ± 37.6 | 663.7 ± 74.4 | 86.8 ± 5.1 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 78.4 ± 92.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| spt | 334.7 ± 40.3 | 663.7 ± 74.4 | 86.6 ± 4.9 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 83.4 ± 76.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd | 340.1 ± 40.4 | 663.7 ± 74.4 | 87.2 ± 4.7 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 29.1 ± 67.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_low | 418.8 ± 46.8 | 663.7 ± 74.4 | 70.9 ± 3.5 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 108.0 ± 123.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_slack(0) | 353.0 ± 53.2 | 663.7 ± 74.4 | 78.3 ± 7.1 | 64.0 ± 0.0 | 1831.1 ± 531.5 | 40.3 ± 63.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s0 | 411.4 ± 47.8 | 663.7 ± 74.4 | 70.8 ± 3.6 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 182.7 ± 122.1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s1 | 411.4 ± 47.8 | 663.7 ± 74.4 | 70.8 ± 3.6 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 182.7 ± 122.1 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s2 | 410.9 ± 43.8 | 663.7 ± 74.4 | 70.9 ± 3.6 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 187.6 ± 156.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s0 | 390.7 ± 55.5 | 663.7 ± 74.4 | 69.9 ± 4.0 | 90.7 ± 28.7 | 1353.6 ± 154.5 | 247.0 ± 148.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s1 | 373.2 ± 60.2 | 663.7 ± 74.4 | 70.3 ± 3.7 | 102.4 ± 30.2 | 1394.9 ± 145.6 | 301.1 ± 216.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s2 | 384.6 ± 51.2 | 663.7 ± 74.4 | 73.9 ± 4.2 | 73.6 ± 14.7 | 1488.2 ± 183.8 | 191.0 ± 141.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Paired against the best rule:

| paired vs edd_low | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| fifo | -83.7 ± 4.4 | -29.7 ± 29.1 | +577.4 ± 24.6 | 0/30 |
| spt | -84.2 ± 3.8 | -24.6 ± 34.1 | +577.4 ± 24.6 | 0/30 |
| edd | -78.7 ± 2.9 | -78.9 ± 25.2 | +577.4 ± 24.6 | 0/30 |
| edd_slack(0) | -65.8 ± 21.5 | -67.8 ± 27.1 | +483.7 ± 160.1 | 0/30 (9 tied) |
| PPO (ours) s0 | -7.5 ± 3.0 | +74.7 ± 29.9 | +0.0 ± 0.0 | 2/30 (6 tied) |
| PPO (ours) s1 | -7.5 ± 3.0 | +74.7 ± 29.9 | +0.0 ± 0.0 | 2/30 (6 tied) |
| PPO (ours) s2 | -8.0 ± 3.2 | +79.6 ± 32.0 | +0.0 ± 0.0 | 0/30 (6 tied) |
| Double-DQN s0 | -28.2 ± 8.3 | +139.0 ± 43.3 | +6.2 ± 3.1 | 1/30 |
| Double-DQN s1 | -45.6 ± 11.4 | +193.1 ± 58.5 | +47.5 ± 16.6 | 0/30 |
| Double-DQN s2 | -34.2 ± 6.6 | +83.0 ± 39.1 | +140.9 ± 23.5 | 0/30 |

| agent | speed share low / nominal / high | launch order = EDD order |
|---|---|---|
| PPO (ours) s0 | 100.0% / 0.0% / 0.0% | 5/30 |
| PPO (ours) s1 | 100.0% / 0.0% / 0.0% | 5/30 |
| PPO (ours) s2 | 100.0% / 0.0% / 0.0% | 6/30 |
| Double-DQN s0 | 98.6% / 1.3% / 0.1% | 1/30 |
| Double-DQN s1 | 92.5% / 5.4% / 2.1% | 1/30 |
| Double-DQN s2 | 75.9% / 23.9% / 0.2% | 3/30 |

## w_tardiness = 1

edd_slack margin tuned on training seeds: **180 min** (mean training return by margin: 0: 278.3, 30: 283.0, 60: 287.0, 120: 300.9, 180: 303.5, 240: 301.4, 360: 301.5, 480: 301.5, 720: 301.5).
Best rule on training seeds: **edd** (mean training return: fifo: 233.0, spt: 213.8, edd: 320.0, edd_low: 282.4, edd_slack(180): 303.5).

Evaluation seeds, mean ± std:

| policy | return | units_out | throughput_per_h | changeover_min | energy | tardiness_min | unmet_units | invalid_actions |
|---|---|---|---|---|---|---|---|---|
| fifo | 264.6 ± 77.0 | 663.7 ± 74.4 | 86.8 ± 5.1 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 78.4 ± 92.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| spt | 259.6 ± 74.8 | 663.7 ± 74.4 | 86.6 ± 4.9 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 83.4 ± 76.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd | 313.9 ± 66.9 | 663.7 ± 74.4 | 87.2 ± 4.7 | 64.0 ± 0.0 | 1924.8 ± 215.8 | 29.1 ± 67.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_low | 321.6 ± 107.2 | 663.7 ± 74.4 | 70.9 ± 3.5 | 64.0 ± 0.0 | 1347.4 ± 151.0 | 108.0 ± 123.5 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| edd_slack(180) | 317.0 ± 79.4 | 663.7 ± 74.4 | 85.8 ± 7.3 | 64.0 ± 0.0 | 1989.3 ± 473.2 | 16.3 ± 39.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s0 | 261.6 ± 97.1 | 663.7 ± 74.4 | 76.1 ± 3.7 | 64.0 ± 0.0 | 1526.5 ± 178.1 | 141.2 ± 107.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s1 | 255.5 ± 103.9 | 663.7 ± 74.4 | 71.2 ± 3.7 | 64.0 ± 0.0 | 1360.1 ± 163.8 | 172.2 ± 116.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| PPO (ours) s2 | 268.1 ± 99.4 | 663.7 ± 74.4 | 76.1 ± 3.5 | 64.0 ± 0.0 | 1526.9 ± 178.1 | 134.6 ± 118.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s0 | 177.1 ± 143.4 | 663.7 ± 74.4 | 77.2 ± 5.3 | 105.6 ± 31.1 | 1700.1 ± 254.9 | 178.9 ± 140.9 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s1 | 210.4 ± 114.2 | 663.7 ± 74.4 | 85.8 ± 6.3 | 85.3 ± 26.6 | 2038.7 ± 237.1 | 104.8 ± 125.9 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Double-DQN s2 | 212.3 ± 113.8 | 663.7 ± 74.4 | 78.6 ± 4.2 | 83.2 ± 22.8 | 1702.1 ± 185.3 | 154.5 ± 132.1 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Paired against the best rule:

| paired vs edd | Δ return | Δ tardiness_min | Δ energy | higher return |
|---|---|---|---|---|
| fifo | -49.2 ± 25.4 | +49.2 ± 25.4 | +0.0 ± 0.0 | 0/30 (11 tied) |
| spt | -54.3 ± 24.5 | +54.3 ± 24.5 | +0.0 ± 0.0 | 2/30 (11 tied) |
| edd_low | +7.8 ± 23.2 | +78.9 ± 25.2 | -577.4 ± 24.6 | 18/30 |
| edd_slack(180) | +3.2 ± 10.9 | -12.8 ± 11.8 | +64.5 ± 129.3 | 13/30 (6 tied) |
| PPO (ours) s0 | -52.3 ± 32.6 | +112.0 ± 33.0 | -398.3 ± 18.2 | 11/30 |
| PPO (ours) s1 | -58.3 ± 35.6 | +143.1 ± 36.4 | -564.8 ± 24.5 | 10/30 |
| PPO (ours) s2 | -45.7 ± 32.5 | +105.4 ± 33.2 | -397.9 ± 18.3 | 10/30 |
| Double-DQN s0 | -136.8 ± 45.0 | +149.7 ± 39.9 | -224.8 ± 46.6 | 2/30 |
| Double-DQN s1 | -103.4 ± 37.1 | +75.7 ± 35.3 | +113.9 ± 31.8 | 1/30 |
| Double-DQN s2 | -101.5 ± 34.6 | +125.3 ± 35.3 | -222.7 ± 28.9 | 6/30 |

| agent | speed share low / nominal / high | launch order = EDD order |
|---|---|---|
| PPO (ours) s0 | 68.9% / 31.1% / 0.0% | 5/30 |
| PPO (ours) s1 | 97.8% / 2.2% / 0.0% | 5/30 |
| PPO (ours) s2 | 68.8% / 31.2% / 0.0% | 6/30 |
| Double-DQN s0 | 50.7% / 42.5% / 6.8% | 1/30 |
| Double-DQN s1 | 13.2% / 70.4% / 16.4% | 3/30 |
| Double-DQN s2 | 44.2% / 52.7% / 3.1% | 2/30 |

## w_tardiness = 2.5

edd_slack margin tuned on training seeds: **360 min** (mean training return by margin: 0: 188.6, 30: 204.3, 60: 219.2, 120: 254.4, 180: 264.2, 240: 264.0, 360: 265.6, 480: 265.6, 720: 265.6).
Best rule on training seeds: **edd_slack(360)** (mean training return: fifo: 35.8, spt: -12.1, edd: 253.5, edd_low: 22.7, edd_slack(360): 265.6).

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

| agent | speed share low / nominal / high | launch order = EDD order |
|---|---|---|
| PPO (ours) s0 | 0.0% / 91.9% / 8.1% | 6/30 |
| PPO (ours) s1 | 0.0% / 69.1% / 30.9% | 6/30 |
| PPO (ours) s2 | 0.0% / 83.6% / 16.4% | 5/30 |
| Double-DQN s0 | 18.2% / 35.1% / 46.7% | 2/30 |
| Double-DQN s1 | 25.7% / 35.8% / 38.6% | 4/30 |
| Double-DQN s2 | 19.0% / 21.3% / 59.7% | 0/30 |
