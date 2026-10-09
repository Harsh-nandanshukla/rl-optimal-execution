"""Non-learning execution strategies the RL agents are benchmarked against.

Every strategy is a STATIC schedule: a holdings trajectory x_0 = 1 > x_1 > ... > x_N = 0
decided before the day starts and never revised. That is precisely the limitation the RL
agent can exploit when prices carry a predictable signal.

* Immediate — dump the whole order in the first interval (minimum risk, maximum impact).
* TWAP      — sell equal amounts every interval.
* VWAP      — sell in proportion to the (expected) intraday volume curve.
* Almgren–Chriss — closed-form schedule minimising E[IS] + lambda * Var[IS] for a
  random-walk price with linear impact. TWAP is its lambda -> 0 limit.
"""

from __future__ import annotations

import numpy as np

from data import intraday_volume_profile, load_config
from env import BPS, ExecutionEnv


def _market(cfg: dict):
    m = cfg["market"]
    tau = m["horizon"] / m["n_steps"]
    eta_tilde = m["eta"] - 0.5 * m["gamma"] * tau
    return m, tau, eta_tilde


# ----------------------------------------------------------------------------- schedules
def immediate_holdings(n_steps: int) -> np.ndarray:
    h = np.zeros(n_steps + 1)
    h[0] = 1.0
    return h


def twap_holdings(n_steps: int) -> np.ndarray:
    return 1.0 - np.arange(n_steps + 1) / n_steps


def vwap_holdings(volume_share: np.ndarray) -> np.ndarray:
    return np.concatenate([[1.0], 1.0 - np.cumsum(volume_share / volume_share.sum())]).clip(0, 1)


def ac_kappa(cfg: dict, lam: float) -> float:
    """Urgency parameter of the discrete-time AC solution: cosh(kappa*tau) = 1 + tau^2 * lam*sigma^2 / (2*eta~)."""
    m, tau, eta_tilde = _market(cfg)
    kappa_tilde_sq = lam * m["sigma"] ** 2 / eta_tilde
    return float(np.arccosh(1 + 0.5 * kappa_tilde_sq * tau ** 2) / tau)


def ac_holdings(cfg: dict, lam: float | None = None) -> np.ndarray:
    """Optimal holdings x_j = sinh(kappa (T - t_j)) / sinh(kappa T)."""
    m, tau, _ = _market(cfg)
    lam = m["risk_aversion"] if lam is None else lam
    T, N = m["horizon"], m["n_steps"]
    t = np.arange(N + 1) * tau
    kappa = ac_kappa(cfg, lam)
    if kappa * T < 1e-8:
        return twap_holdings(N)
    return np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


# ----------------------------------------------------------------------------- analytics
def expected_cost_and_variance(holdings: np.ndarray, cfg: dict) -> tuple[float, float]:
    """Closed-form E[IS] and Var[IS] (in bps and bps^2) of a static schedule in the random-walk world."""
    m, tau, eta_tilde = _market(cfg)
    trades = -np.diff(holdings)
    mean = 0.5 * m["gamma"] + m["half_spread"] + (eta_tilde / tau) * np.sum(trades ** 2)
    var = m["sigma"] ** 2 * tau * np.sum(holdings[1:] ** 2)
    return mean * BPS, var * BPS ** 2


def ac_objective(holdings: np.ndarray, cfg: dict, lam: float) -> float:
    """E[IS] + lambda * Var[IS], in bps (variance converted back to price units before weighting)."""
    mean, var = expected_cost_and_variance(holdings, cfg)
    return mean + lam * var / BPS


# ----------------------------------------------------------------------------- simulation
def run_schedule(env: ExecutionEnv, holdings: np.ndarray, path=None, seed=None) -> dict:
    env.reset(seed=seed, options={"path": path} if path is not None else None)
    info = {}
    for k in range(env.N):
        _, _, done, _, info = env.step_shares(holdings[k] - holdings[k + 1])
    return info


def all_baselines(cfg: dict, lam: float | None = None, volume_share=None) -> dict[str, np.ndarray]:
    N = cfg["market"]["n_steps"]
    vol = intraday_volume_profile(N) if volume_share is None else volume_share
    return {
        "Immediate": immediate_holdings(N),
        "TWAP": twap_holdings(N),
        "VWAP": vwap_holdings(vol),
        "Almgren-Chriss": ac_holdings(cfg, lam),
    }


# ----------------------------------------------------------------------------- sanity checks
if __name__ == "__main__":
    cfg = load_config()
    lam = cfg["market"]["risk_aversion"]
    N = cfg["market"]["n_steps"]
    env = ExecutionEnv(cfg)
    rng = np.random.default_rng(0)
    paths = [env.sampler.sample(rng) for _ in range(20000)]   # same paths for every strategy

    print(f"lambda = {lam},  kappa*T = {ac_kappa(cfg, lam) * cfg['market']['horizon']:.3f}\n")
    print(f"{'strategy':<16}{'E[IS] formula':>14}{'E[IS] sim':>11}{'± 95%':>8}"
          f"{'std formula':>13}{'std sim':>9}{'objective':>11}")
    for name, h in all_baselines(cfg, lam).items():
        mean_f, var_f = expected_cost_and_variance(h, cfg)
        res = [run_schedule(env, h, path=p) for p in paths]
        is_bps = np.array([r["is_bps"] for r in res])
        ci = 1.96 * is_bps.std() / np.sqrt(len(is_bps))
        print(f"{name:<16}{mean_f:>14.2f}{is_bps.mean():>11.2f}{ci:>8.2f}"
              f"{np.sqrt(var_f):>13.2f}{is_bps.std():>9.2f}{ac_objective(h, cfg, lam):>11.2f}")
        assert abs(is_bps.mean() - mean_f) < max(3 * ci, 1e-6), f"{name}: simulated mean cost disagrees with formula"
        assert abs(is_bps.std() - np.sqrt(var_f)) <= 0.03 * np.sqrt(var_f) + 1e-6, \
            f"{name}: simulated risk disagrees with formula"

    # Check 2: no feasible schedule beats Almgren–Chriss on its own objective.
    ac = ac_holdings(cfg, lam)
    j_ac = ac_objective(ac, cfg, lam)
    beaten = 0
    for _ in range(5000):
        bump = np.concatenate([[0], rng.normal(0, 0.03, N - 1), [0]])
        h = np.clip(ac + bump, 0, 1)
        h = np.minimum.accumulate(h)                    # sell-only: holdings never increase
        beaten += ac_objective(h, cfg, lam) < j_ac - 1e-9
    print(f"\nrandom perturbed schedules that beat AC on its objective: {beaten} / 5000")
    assert beaten == 0

    # Check 3: TWAP is the zero-risk-aversion limit of AC.
    gap = np.abs(ac_holdings(cfg, 1e-9) - twap_holdings(N)).max()
    print(f"max |AC(lambda->0) - TWAP| holdings gap: {gap:.2e}")
    assert gap < 1e-6
    print("\nAll Day-1 sanity checks passed.")
