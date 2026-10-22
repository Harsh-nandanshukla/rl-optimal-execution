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

**D13 — DQN from scratch + PPO from stable-baselines3; CPU only** · Done (DQN) / Done (PPO implemented, smoke-tested)
TA confirmed (Oct 2026) that stable-baselines3 is allowed for PPO. DQN stays our own code; PPO is the library reference.
**D14 — Paired evaluation on identical paths, 5 seeds, held-out data, CVaR 5%** · Planned
**D15 — Report: 4 figures + 1 table** (trajectory vs AC, cost–risk frontier, learning curves, ablation) · Planned
**D16 — No Streamlit demo** · Revoked — not a deliverable; time goes to reproducibility.
**D17 — 6 Python files; sanity checks inside each file's `__main__`** · Done (6 of 6)

**D18 — DQN target network: soft update, τ = 0.005 after every learning update** · Done
Target tracks the online net smoothly instead of jumping every K steps, so TD targets drift gradually.
Rejected: periodic hard copy (an extra interval to tune; target lurches at each copy).

**D19 — DQN scales rewards by 0.01 internally during learning** · Done
Env rewards are in bps (one-step Immediate liquidation ≈ −500), so raw Q-values are O(100); scaled
they are O(1), a convenient range for Huber loss and Adam. With γ = 1, multiplying every reward by a
positive constant multiplies every return by it, so the optimal policy is unchanged. Env and
evaluation stay in bps.

**D20 — Fixed validation/test path sets; checkpoint = best greedy validation objective** · Done
Synthetic val paths from `evaluation.val_seed` (100000), test from `test_seed` (200000), both far from training
seeds 0, 1, … . Score = mean(is_bps + risk_bps) on realised IS, unscaled; 95% CI = 1.96·s/√n over paths.
Real data uses the existing date split. Rejected: selecting on training return (noisy, biased by shaping).
The CI covers path noise only, not training-seed variance, and the best checkpoint is optimistic on the set that
picked it, so reported comparisons come from the test set and multiple seeds, never from validation.

**D21 — PPO: same Discrete(12) actions as DQN, fixed hyperparameters, 0.01 reward wrapper** · Done (implemented, smoke-tested) / Planned (experiments)
γ = 1, 128×128 ReLU pi/vf, n_steps = 2000 (80 episodes), batch 100, 10 epochs, lr 3e-4, clip 0.2, GAE 0.95,
ent 0. Fixed in advance so DQN vs PPO is not a tuning contest. SB3 use confirmed by the TA (D13).

**D22 — PPO is validated between rollouts, not during them** · Done
SB3 calls `on_step` while collecting a rollout, before the update on it, so validating there scored a policy one
rollout behind its step label and never scored the final update. Now validated at rollout start (after `train()`)
and at training end; PPO `--steps` and `--eval-every` must be multiples of n_steps = 2000.

**D23 — Discretised-AC references on the 12-action grid** · Done
RL can only pick the 12 multipliers, so continuous AC is not exactly reachable. *Tracking* (primary): at each
decision pick the action whose actual trade is closest to x_k − x_AC[k+1] (+0.26 bps vs AC, holdings RMSE 0.0028).
*Open-loop* (extra row): closest to x_AC[k] − x_AC[k+1] (+0.09 bps, RMSE 0.0097). Both are quantisation
references, **not lower bounds**: open-loop tracks worse yet scores better, so the grid optimum is unknown.

**D24 — Experiment A statistics: paired per-seed CIs, separate across-seed t-interval** · Done (implemented)
Per seed: paired per-path differences vs AC / both dAC / TWAP, 95% CI = 1.96·sd/√n (path noise only). Across
5 seeds: mean ± sd and t-interval (t₀.₉₇₅,₄ = 2.776) of the per-seed means (training variability). Never pool all
paths across seeds as the headline interval. Trajectory: inventory before each of the 25 decisions,
RMSE² = bias² + dispersion² (shape error vs reacting to noise). Also CVaR 5% of IS, capture (TWAP−RL)/(TWAP−AC).
For adaptive policies the pathwise risk penalty is not Var[IS]; sd of IS is reported alongside.

**D25 — Tracked `results/`, integrity-checked freeze manifest** · Done (implemented) / Planned (freeze)
`results/expA_*` (summary, seed aggregate, trajectories, per-path IS/risk, per-path holdings, run info) and
`figures/expA_trajectories.*` are committed. `--split test` runs only after `results/expA_manifest.json` verifies:
commit, run list, SHA-256 of all checkpoints, metadata and code (JSON canonical, text LF-normalised, so
`core.autocrlf` cannot break it), config and test seed. Order: verify runs → commit models + code → freeze →
commit manifest → test once. Re-running on unchanged frozen files reproduces, it does not re-use the test set.

**D26 — `evaluate.py` moved from Day 6 to Day 3** · Done
Evaluation code (and its test-set guard) is written and self-checked before any expensive training, so the
held-out evaluation is a fixed procedure rather than a script written after seeing results.
