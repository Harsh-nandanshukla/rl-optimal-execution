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

from data import PricePath, intraday_volume_profile, load_config
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


# ----------------------------------------------------------------------------- discretised AC
# The RL agents can only pick one of the 12 multipliers, so continuous AC is not exactly reachable.
# These two policies show what following AC costs on that same action grid. They are reference
# baselines, NOT lower bounds: some other 12-action sequence may score better than either.
DISCRETE_AC_RULES = ("tracking", "open-loop")


def discretised_ac_action(env: ExecutionEnv, target: np.ndarray, rule: str = "tracking") -> int:
    """Index of the action whose ACTUAL trade is closest to the AC trade wanted at the current decision.

    tracking  : wanted trade = x_k - target[k+1]   (steer back onto the AC holdings; errors do not pile up)
    open-loop : wanted trade = target[k] - target[k+1]   (copy AC's planned trade, ignore past error)
    Candidates use the env's own mapping (multiplier * x / steps left) and its clip to the inventory left.
    The env still force-sells everything at the last decision, whatever action is returned.
    """
    if rule not in DISCRETE_AC_RULES:
        raise ValueError(rule)
    k, x = env.k, env.x
    candidates = np.minimum(env.multipliers * x / (env.N - k), x)
    wanted = x - target[k + 1] if rule == "tracking" else target[k] - target[k + 1]
    return int(np.argmin(np.abs(candidates - wanted)))      # ties -> lower index (slower), deterministic


def discretised_ac_policy(env: ExecutionEnv, cfg: dict, lam: float, rule: str = "tracking"):
    """act(obs) -> action for evaluation loops. Reads the env's exact float64 state, not the float32 obs."""
    target = ac_holdings(cfg, lam)
    return lambda obs: discretised_ac_action(env, target, rule)


def discretised_ac_holdings(cfg: dict, lam: float, rule: str = "tracking") -> tuple[np.ndarray, list[int]]:
    """Holdings and actions of a discretised-AC policy. They do not depend on prices, so any path will do."""
    env = ExecutionEnv(cfg, risk_aversion=lam)
    N = env.N
    env.reset(seed=0, options={"path": PricePath(np.zeros(N), np.zeros(N), np.full(N, 1.0 / N))})
    act, actions, info = discretised_ac_policy(env, cfg, lam, rule), [], {}
    for _ in range(N):
        actions.append(act(None))
        *_, info = env.step(actions[-1])
    return info["holdings"], actions


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

    # Check 4: discretised AC on the env's 12-action grid (valid, deterministic, close to AC, formulas agree).
    print("\ndiscretised AC on the 12-action grid:")
    M = env.multipliers
    for rule in DISCRETE_AC_RULES:
        h_ref, a_ref = discretised_ac_holdings(cfg, lam, rule)
        assert all(0 <= a < len(M) for a in a_ref), "invalid action index"
        assert np.all(np.diff(h_ref) <= 0) and h_ref.min() >= 0 and h_ref.max() <= 1, "holdings must fall within [0, 1]"
        assert h_ref[0] == 1.0 and h_ref[-1] == 0.0, "must start with the full order and end fully liquidated"

        # Grid approximation: the wanted trade lies inside the grid's range, and the chosen trade is the
        # nearest grid point, i.e. within half the gap between the two grid trades that bracket it.
        x = 1.0
        for k in range(N - 1):
            cand = np.minimum(M * x / (N - k), x)
            wanted = x - ac[k + 1] if rule == "tracking" else ac[k] - ac[k + 1]
            assert cand.min() <= wanted <= cand.max(), f"{rule}: AC trade at k={k} outside the action grid"
            hi = np.searchsorted(np.sort(cand), wanted).clip(1, len(cand) - 1)
            half_gap = 0.5 * (np.sort(cand)[hi] - np.sort(cand)[hi - 1])
            assert abs(cand[a_ref[k]] - wanted) <= half_gap + 1e-12, f"{rule}: not the nearest grid trade at k={k}"
            x -= cand[a_ref[k]]
            assert abs(x - h_ref[k + 1]) < 1e-12

        # Deterministic: the same actions on every price path; and stepping actions in the env gives
        # exactly the cash flows of replaying the holdings schedule, so the closed forms apply.
        sim_is = np.empty(len(paths))
        for i, p in enumerate(paths):
            env.reset(seed=0, options={"path": p})
            act = discretised_ac_policy(env, cfg, lam, rule)
            for k in range(N):
                assert act(None) == a_ref[k], f"{rule}: actions depend on the price path"
                *_, info = env.step(a_ref[k])
            assert np.array_equal(info["holdings"], h_ref)
            sim_is[i] = info["is_bps"]
            if i < 200:
                assert abs(run_schedule(env, h_ref, path=p)["is_bps"] - sim_is[i]) < 1e-9
        mean_f, var_f = expected_cost_and_variance(h_ref, cfg)
        ci = 1.96 * sim_is.std() / np.sqrt(len(sim_is))
        assert abs(sim_is.mean() - mean_f) < 3 * ci and abs(sim_is.std() - np.sqrt(var_f)) <= 0.03 * np.sqrt(var_f)
        j = ac_objective(h_ref, cfg, lam)
        assert np.isfinite(j) and j >= j_ac - 1e-9, "a feasible schedule cannot beat AC on its own objective"
        print(f"  {rule:<10} actions {a_ref[:-1]}  max|x-x_AC| {np.abs(h_ref - ac).max():.4f}  "
              f"RMSE {np.sqrt(np.mean((h_ref[:-1] - ac[:-1]) ** 2)):.4f}  E[IS] {mean_f:.2f} (sim {sim_is.mean():.2f} "
              f"± {ci:.2f})  objective {j:.3f} (AC {j_ac:.3f}, +{j - j_ac:.3f})")
    print("\nAll Day-1 sanity checks passed.")
