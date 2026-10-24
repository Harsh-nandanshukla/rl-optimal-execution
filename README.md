# Optimal Execution of Large NSE Equity Orders using Deep Reinforcement Learning

A fund must sell a large block of shares before the market closes. Selling fast moves the price against it
(market impact); selling slowly leaves it exposed to price risk. We cast this as a 25-step MDP (one decision
per 15-minute interval) and train two RL agents, a DQN written from scratch in PyTorch and PPO
(stable-baselines3). Their reward is exactly the Almgren–Chriss (AC) mean-variance objective. We test three
hypotheses on held-out data, each evaluated once after a hash-verified freeze:
- **H1.** In a pure random-walk market, RL recovers the closed-form AC schedule. It does, to within 0.14–0.17 bps.
- **H2.** With a predictable AR(1) drift that the agent observes, RL beats AC. It does, by 14–15 bps.
- **H3.** Hiding that drift removes the advantage. It does.

On real NSE price paths from 2023 to 2026, the random-walk agents match AC and beat TWAP and historical VWAP.
The signal-trained agents do not transfer: real paths carry no such signal, and they lose 2–9 bps to AC.

## Headline results (held-out test sets)

Objective = implementation shortfall + λ·risk (λ = 25), in bps of arrival value; **lower is better**. The AC
row gives AC's objective. Every other row gives the objective minus AC on identical paths.

| Strategy | Exp A: random walk | Exp B: AR(1) drift | Real NSE 2023–26 |
|---|---|---|---|
| AC (closed form), objective | 49.34 ± 1.69 | 49.25 ± 2.38 | 55.23 ± 4.31 |
| AC on the action grid | +0.26 ± 0.01 | +0.27 ± 0.01 | +0.30 ± 0.04 |
| TWAP | +7.60 ± 0.61 | +7.59 ± 0.90 | +8.24 ± 1.18 |
| VWAP | — | +7.23 ± 0.90 | +11.84 ± 1.23 |
| Signal rule (heuristic) | — | −9.85 ± 0.80 | — |
| Immediate | +451.16 ± 1.69 | +451.25 ± 2.38 | +445.27 ± 5.08 |
| DQN, Exp A agent | +0.14 ± 0.04 | — | +0.15 ± 0.10 |
| PPO, Exp A agent | +0.17 ± 0.07 | — | +0.17 ± 0.09 |
| DQN, signal observed | — | **−13.94 ± 1.16** | +8.75 ± 5.38 |
| PPO, signal observed | — | **−15.01 ± 0.99** | +2.22 ± 0.90 |
| DQN, signal hidden (ablation) | — | +1.07 ± 0.51 | +2.22 ± 0.48 |
| PPO, signal hidden (ablation) | — | +0.70 ± 0.90 | +2.58 ± 0.65 |

How the intervals are computed:
- **Baselines:** paired 95% CI over 10,000 held-out synthetic paths. On real data the CI is clustered by date
  (803 test dates, 2,409 stock-days).
- **Agents:** mean ± t-interval over 5 training seeds (df 4).
- **B minus ablation** (matched seeds, identical paths): −15.01 ± 1.46 (DQN) and −15.71 ± 1.54 (PPO).

The same table is in `results/main_table.md` / `.csv`. The figures are in `figures/`:
- `fig1_expA_trajectory`: holdings vs AC
- `fig2_expB_signal`: Exp B and the ablation
- `fig3_real_transfer`: real NSE days
- `fig4_learning_curves`: Exp A validation curves

## RL formulation

| | |
|---|---|
| **State** (4-dim) | inventory left, time left, observed drift (0 in the ablation), last price move (all normalised) |
| **Action** (12 discrete) | sell `m × (inventory left / intervals left)`, `m ∈ {0, 0.25, …, 4}`; 1.0 = TWAP pace |
| **Reward** (bps) | mark-to-market P&L of the position over the interval − λσ²τ·(inventory)² |
| **Episode** | one trading day, 25 decisions, γ = 1; any inventory left at the last step is force-sold |

The episode return equals −(implementation shortfall + λ·risk), the AC objective. So in a random-walk market
the optimal policy is the AC schedule, which gives Experiment A a known right answer. `env.py` checks this
identity. Training on synthetic paths uses an unbiased, variance-reduced expected reward. Every evaluation
uses the realised shortfall from cash flows.

## Setup

Python 3.12, CPU only. Package versions are pinned in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Data (real-data parts only).** NSE 1-minute OHLCV bars from Kaggle: [dataset URL — to be added]. Put the
three files in `data/raw/` (gitignored). The freeze manifest records their SHA-256, so a different copy is
detected before any test day is used:
- `ADANIPORTS_minute.csv` (`653097eb01a49fcf…`)
- `BAJAJ-AUTO_minute.csv` (`b5d3f44bcf2671da…`)
- `COALINDIA_minute.csv` (`36c0462cc27caebf…`)

The split is train ≤ 2021, validation 2022, test 2023-01-02 → 2026-04-07. Experiments A and B use synthetic
paths and need no data.

## Reproduce

**1. Correctness checks** (each file has a self-check):

```powershell
python data.py; python env.py; python baselines.py   # last prints "All Day-1 sanity checks passed."
python dqn.py; python train.py --self-check; python evaluate.py --self-check; python report.py --self-check
```

**2. Training** (300k steps per run, about 10 min alone on one CPU core). Set `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1, and run seeds as separate processes (≤ 5 at a time). Each run
saves its best checkpoint, selected on 1,000 validation paths, to `models/<run>.{pt,zip}` with `.json`
metadata, and writes its learning curve to `logs/<run>.csv`.

```powershell
# Experiment A (10 runs): random walk
python train.py --algo dqn --mode synthetic --signal off --observe-signal on --lam 25 --seed 0 --steps 300000 --eval-every 10000
# Experiment B (10 runs): drift generated and observed
python train.py --algo dqn --mode synthetic --signal on  --observe-signal on  --lam 25 --seed 0 --steps 300000 --eval-every 10000
# Ablation (10 runs): drift generated, hidden from the agent
python train.py --algo dqn --mode synthetic --signal on  --observe-signal off --lam 25 --seed 0 --steps 300000 --eval-every 10000
```

Repeat each command for `--algo ppo` and `--seed 0..4`. The 30 trained agents are committed in `models/`, so
you can skip this step.

**3. Development evaluation** (validation data only, optimistic, never reported as results):
`python evaluate.py` (Exp A), `python evaluate.py --exp B` (B + ablation) and `python evaluate.py --exp real`
(2022 days).

**4. Freeze, then test once.**
- `python evaluate.py [--exp B] --freeze` writes `results/exp{A,B}_manifest.json`. It records the commit and the
  SHA-256 of every checkpoint, metadata file, code file and config; for Exp B also the raw CSVs and the test-day
  counts.
- `python evaluate.py [--exp B] --split test` re-verifies all of that before a single test path or test day
  exists. It refuses if the outputs are already published.
- The test sets have been used. To reproduce them independently, check out the freeze commit in a fresh clone,
  where the code matches the manifest and the outputs don't exist yet:

| Experiment | Freeze commit | Command | Published in |
|---|---|---|---|
| A | `c11fcfd` (code of `c9bdc57`) | `python evaluate.py --split test` | `14c4ebc` |
| B + ablation + real NSE | `4814068` (code of `18b0e9b`) | `python evaluate.py --exp B --split test` (~30 min) | `2dc62f2` |

The Exp A manifest no longer verifies at later commits, by design: `data.py`, `evaluate.py` and `config.yaml`
changed for Exp B.

**5. Report figures and table** (from the committed `results/` only, no simulation): `python report.py`.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | all market, signal, action, data-split and evaluation parameters |
| `data.py` | synthetic paths (random walk ± AR(1) drift); clock-time NSE loader, date split, locked test days |
| `env.py` | Gymnasium `ExecutionEnv`: AC impact dynamics, mean-variance reward, path replay, ablation switch |
| `baselines.py` | Immediate, TWAP, VWAP, closed-form AC and AC on the action grid; closed-form cost and risk checks |
| `dqn.py` | DQN from scratch: MLP Q-network, replay buffer, target network, Double-DQN target, Huber loss |
| `train.py` | one CLI for DQN and PPO; validation-based checkpoint selection, metadata, learning curves |
| `evaluate.py` | paired evaluation of all strategies on identical paths, freeze manifests, once-only test runs |
| `report.py` | paper figures and the main table from committed results |
| `models/` | the 30 trained agents (`.pt` DQN, `.zip` PPO) with metadata |
| `results/` | test outputs: summaries, per-seed aggregates, per-path data, trajectories, manifests, main table |
| `figures/` | `fig1`–`fig4` for the report (300-dpi PNG + PDF), plus the figures `evaluate.py` wrote at test time |
| `DECISIONS.md` | every design decision, with the reasoning |

## Limitations

- **Simulated impact.** There is no public NSE order-book data, so on real days the impact is still the AC model
  applied to real prices. The real-data rows test the schedules under real price moves, not real impact. VWAP
  is penalised because this impact model doesn't depend on volume.
- **Fixed σ.** AC, the risk penalty and the observation scaling use the config volatility (2% daily) on every
  real day. They don't use a per-day or per-stock estimate.
- **Noiseless synthetic signal.** In Exp B the drift is a correctly specified AR(1) process, observed without
  noise. The 14–15 bps edge is therefore an upper bound for a real, noisy or misspecified signal. Real paths
  contain no such signal, which is why the B agents fail to transfer.
- **Shared noise draws.** The synthetic sampler draws price noise before drift, so the Exp A and Exp B test paths
  (same seed 200000) share their noise component. The drift differs, and no Exp B decision used Exp A test
  results.
- **Partial ablation.** The ablation hides the drift but keeps the last price move, which is correlated about 0.30
  with the drift. The edge still disappears. A stronger ablation would need an environment change and was not run.
- **Five seeds.** The across-seed intervals use df = 4 and are wide by construction. B-DQN on real data varies
  from +4 to +16 bps by seed.
- **One configuration.** λ = 25, 25 intervals and the D21 hyperparameters were not swept.
