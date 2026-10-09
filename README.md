# Optimal Execution of Large NSE Equity Orders using Deep Reinforcement Learning

A fund must sell a large block of shares before the market closes. Selling fast moves the
price against it (market impact); selling slowly leaves it exposed to price risk. We cast this
as an MDP and train RL agents (DQN written from scratch, PPO) to decide how much to sell in
each 15-minute interval, benchmarked against TWAP, VWAP and the closed-form
Almgren–Chriss (2000) optimal schedule.

## RL formulation

| | |
|---|---|
| **State** (4-dim) | inventory left, time left, observed short-term drift, last price move (all normalised) |
| **Action** (12 discrete) | sell `m × (inventory left / intervals left)` shares, `m ∈ {0, 0.25, …, 4}`; 1.0 = TWAP pace |
| **Reward** (bps) | mark-to-market P&L of the position over the interval − λσ²τ·(inventory)² |
| **Episode** | one trading day, 25 decisions; any inventory left at the last step is force-sold |

The episode return equals −(implementation shortfall + λ·risk), the Almgren–Chriss
objective, so in a random-walk market the optimal agent should reproduce the AC schedule
(Experiment A). With a predictable drift, an adaptive agent can beat any static schedule
(Experiment B).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Real data: download an NSE 1-minute OHLCV dataset from Kaggle and place the per-stock CSV
files in `data/raw/`. The loader accepts either a single datetime column or separate
`date` + `time` columns, plus `close` and `volume`.

## Sanity checks (run these first)

```bash
python data.py        # synthetic paths have the configured volatility
python env.py         # Gymnasium API check; episode return == -(IS + risk) exactly
python baselines.py   # simulated costs match the Almgren–Chriss formulas; nothing beats AC on its objective
```

## Files

| File | Purpose |
|---|---|
| `config.yaml` | all market, signal, action and data parameters |
| `data.py` | synthetic price paths (random walk ± AR(1) drift) and the NSE CSV loader |
| `env.py` | Gymnasium `ExecutionEnv` with Almgren–Chriss impact dynamics |
| `baselines.py` | Immediate, TWAP, VWAP, Almgren–Chriss schedules + closed-form cost/risk |
| `dqn.py` | *(day 2)* DQN from scratch |
| `train.py` | *(day 2)* training entry point |
| `evaluate.py` | *(day 6)* held-out evaluation and report figures |

## Status

- [x] Day 1 — environment, baselines, Almgren–Chriss correctness checks
- [ ] Day 2 — DQN + training loop
- [ ] Day 3 — Experiment A (RL recovers AC)
- [ ] Day 4 — Experiment B (signal, real NSE paths)
- [ ] Days 5–7 — seeds, λ sweep, ablation, evaluation, report
