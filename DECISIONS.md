# DECISIONS.md

Key decisions, why we made them, and what was rejected or reversed.
Status: **Done** · **Planned** · **Revoked**.

## Project

**D1 — Optimal trade execution with RL** · Done
Rejected: ambulance repositioning, solar/battery/EV scheduling, insulin dosing, platelet inventory,
UAV delivery. Why this: unique in the batch, real NSE data, a closed-form optimum (Almgren–Chriss)
to validate against, fits one week. UAV rejected: no clear MDP, unstable multi-agent RL, no data.
Insulin rejected: public simulator with many public repos → plagiarism-check risk.

## Data

**D2 — Kaggle NSE minute data, not yfinance** · Planned
yfinance keeps only ~7 days of 1-minute data. Dataset must include volume (for VWAP).

**D3 — Synthetic paths for Experiments A/B, real NSE paths for realism and held-out tests** · Done / Planned
Synthetic world matches the AC assumptions exactly, so Experiment A has a known right answer.
Impact on real paths is still simulated (no public order-book data) — stated as a limitation.

## Formulation

**D4 — Discrete-time Almgren–Chriss dynamics, normalised units (price = 1, order = 1), bps** · Done

**D5 — 25 decisions per day** (15-minute buckets of the 375-minute NSE session) · Done

**D6 — Action = multiplier of TWAP pace, 12 levels {0 … 4}** · Done
Keeps each action's meaning stable over the episode. Rejected: raw "fraction of inventory".

**D7 — State = [inventory left, time left, observed drift, last price move], all O(1)** · Done

**D8 — Reward = mark-to-market P&L − λσ²τx²** · Done
Episode return = −(IS + λ·risk) = the AC objective, so "RL recovers AC" is a real test.
Rejected: reward = −cost only (different objective, could never match AC).

**D9 — Variance-reduced training reward on synthetic paths** · Done
Price noise (~100 bps/episode) drowns the ~7.6 bps AC-vs-TWAP gap. Replacing x·increment by its
conditional mean is unbiased, so the optimal policy is unchanged. Evaluation uses realised IS.

**D10 — Unsold shares force-sold at the deadline; discount γ = 1** · Done

## Parameters

**D11 — λ: 10 → 25** · Done (10 Revoked)
At λ = 10 AC beat TWAP by only 1.5 bps (too small to learn or show); at 25 the gap is 7.6 bps.

**D12 — Signal strength: 0.15 → 0.04** · Done (0.15 Revoked)
0.15 made the problem trivial. At 0.04 a crude reactive rule beats AC by ~9 bps — real but non-trivial edge.

## Algorithms, evaluation, scope

**D13 — DQN from scratch + PPO from stable-baselines3; CPU only** · Planned
**D14 — Paired evaluation on identical paths, 5 seeds, held-out data, CVaR 5%** · Planned
**D15 — Report: 4 figures + 1 table** (trajectory vs AC, cost–risk frontier, learning curves, ablation) · Planned
**D16 — No Streamlit demo** · Revoked — not a deliverable; time goes to reproducibility.
**D17 — 6 Python files; sanity checks inside each file's `__main__`** · Done (4 of 6)
