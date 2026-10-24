| Strategy | Exp A: random walk | Exp B: AR(1) drift | Real NSE 2023–26 |
|---|---|---|---|
| AC (closed form), objective | 49.34 ± 1.69 | 49.25 ± 2.38 | 55.23 ± 4.31 |
| AC on the action grid | +0.26 ± 0.01 | +0.27 ± 0.01 | +0.30 ± 0.04 |
| TWAP | +7.60 ± 0.61 | +7.59 ± 0.90 | +8.24 ± 1.18 |
| VWAP | — | +7.23 ± 0.90 | +11.84 ± 1.23 |
| Signal rule (heuristic) | — | -9.85 ± 0.80 | — |
| Immediate | +451.16 ± 1.69 | +451.25 ± 2.38 | +445.27 ± 5.08 |
| DQN, Exp A agent | +0.14 ± 0.04 | — | +0.15 ± 0.10 |
| PPO, Exp A agent | +0.17 ± 0.07 | — | +0.17 ± 0.09 |
| DQN, signal observed | — | -13.94 ± 1.16 | +8.75 ± 5.38 |
| PPO, signal observed | — | -15.01 ± 0.99 | +2.22 ± 0.90 |
| DQN, signal hidden (ablation) | — | +1.07 ± 0.51 | +2.22 ± 0.48 |
| PPO, signal hidden (ablation) | — | +0.70 ± 0.90 | +2.58 ± 0.65 |

All values in bps of arrival value; objective = implementation shortfall + λ·risk (λ = 25), lower is better. First row: AC's objective (± 95% CI over paths; real: over stock-days). Other rows: objective minus AC on identical paths. Baselines: paired 95% CI over 10,000 held-out synthetic paths, or over 803 real test dates (clustered by date, 2,409 stock-days). Agents: mean ± t-interval over 5 training seeds (df 4). Exp A agents were trained without a signal and are not evaluated on Exp B paths. VWAP: synthetic volume profile in Exp B; each stock's training-day profile on real data. Real-data impact is simulated with the AC model.
