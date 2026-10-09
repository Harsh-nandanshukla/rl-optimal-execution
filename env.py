"""Gymnasium environment: sell X shares by the end of the day at minimum cost and risk.

Market dynamics follow the discrete-time Almgren–Chriss (2000) model. At decision k the
agent sells n_k shares during interval k (length tau):

    execution price   S~_k = S_{k-1} - half_spread - eta * n_k / tau     (temporary impact)
    mid price         S_k  = S_{k-1} + increment_k - gamma * n_k          (permanent impact)

where increment_k is the exogenous price move (random walk, signal-driven drift, or real NSE data).

Reward (in basis points of the order's notional) is the mark-to-market P&L of the
position over the interval, minus a variance penalty:

    r_k = n_k (S~_k - S_{k-1}) + x_k (S_k - S_{k-1})  -  lambda * sigma^2 * tau * x_k^2

The P&L terms telescope: summed over an episode they equal minus the implementation
shortfall (IS = X*S0 - cash received). The penalty terms sum to lambda * Var[IS] of a static
schedule. So the episode return is exactly -(IS + lambda * risk), the Almgren–Chriss objective,
which is why an optimal agent in the random-walk world should reproduce the AC schedule.

Variance-reduced training reward (reward_mode="expected", synthetic paths only): the term
x_k * increment_k is replaced by its conditional mean x_k * mu_k * tau. Because x_k is fixed
before the shock xi_k is drawn, E[x_k * xi_k] = 0, so every policy keeps exactly the same
expected return while the reward noise (std ~100 bps per episode) disappears. Evaluation always
uses the realised implementation shortfall from actual cash flows.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data import PricePath, SyntheticPathSampler, load_config

BPS = 1e4
OBS_CLIP = 5.0


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None, sampler=None, observe_signal: bool = True,
                 action_mode: str = "discrete", risk_aversion: float | None = None,
                 reward_mode: str = "realized"):
        super().__init__()
        if reward_mode not in ("realized", "expected"):
            raise ValueError(reward_mode)
        if reward_mode == "expected" and sampler is not None and not isinstance(sampler, SyntheticPathSampler):
            raise ValueError("reward_mode='expected' needs the synthetic model's known drift; use 'realized' on real data")
        self.reward_mode = reward_mode
        self.cfg = cfg or load_config()
        m = self.cfg["market"]
        self.N = m["n_steps"]
        self.tau = m["horizon"] / self.N
        self.sigma, self.eta, self.gamma, self.eps = m["sigma"], m["eta"], m["gamma"], m["half_spread"]
        self.lam = m["risk_aversion"] if risk_aversion is None else risk_aversion

        self.sampler = sampler or SyntheticPathSampler(self.cfg)
        self.observe_signal = observe_signal
        sig = self.cfg["signal"]
        self.signal_scale = sig["strength"] if sig["strength"] > 0 else 1.0
        self.signal_obs_noise = sig["obs_noise"]

        self.multipliers = np.asarray(self.cfg["actions"]["multipliers"], dtype=float)
        self.action_mode = action_mode
        if action_mode == "discrete":
            self.action_space = spaces.Discrete(len(self.multipliers))
        elif action_mode == "continuous":
            self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(action_mode)

        # [inventory left, time left, observed drift, last price move] — all roughly O(1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -OBS_CLIP, -OBS_CLIP], dtype=np.float32),
            high=np.array([1, 1, OBS_CLIP, OBS_CLIP], dtype=np.float32),
        )

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.path: PricePath = options.get("path") or self.sampler.sample(self.np_random)
        self.k = 0
        self.x = 1.0          # inventory left (fraction of X)
        self.S = 1.0          # mid price (arrival price = 1)
        self.cash = 0.0
        self.last_move = 0.0
        self.risk_total = 0.0
        self.holdings = [1.0]
        self.obs_noise = self.np_random.standard_normal(self.N) * self.signal_obs_noise
        return self._obs(), {}

    def step(self, action):
        return self.step_shares(self.action_to_shares(action))

    # ------------------------------------------------------------------ core dynamics
    def action_to_shares(self, action) -> float:
        steps_left = self.N - self.k
        if self.action_mode == "discrete":
            mult = self.multipliers[int(action)]
        else:
            mult = float(np.clip(np.asarray(action).reshape(-1)[0], 0, 1)) * self.multipliers.max()
        return mult * self.x / steps_left

    def step_shares(self, n: float):
        """Sell n shares (fraction of X) this interval. Also used directly by the baselines."""
        n = float(np.clip(n, 0.0, self.x))
        if self.k == self.N - 1:
            n = self.x                                        # deadline: liquidate whatever is left

        S_prev = self.S
        exec_price = S_prev - (self.eps if n > 0 else 0.0) - self.eta * n / self.tau
        self.cash += n * exec_price
        self.x -= n
        if self.x < 1e-12:
            self.x = 0.0
        move = self.path.increments[self.k]
        self.S = S_prev + move - self.gamma * n

        if self.reward_mode == "realized":
            pnl = n * (exec_price - S_prev) + self.x * (self.S - S_prev)
        else:  # conditional expectation of the same quantity given the drift (see module docstring)
            pnl = n * (exec_price - S_prev) + self.x * (self.path.signal[self.k] * self.tau - self.gamma * n)
        risk = self.lam * self.sigma ** 2 * self.tau * self.x ** 2
        self.risk_total += risk
        reward = (pnl - risk) * BPS

        self.last_move = move
        self.k += 1
        self.holdings.append(self.x)
        terminated = self.k == self.N
        info = {"shares": n, "exec_price": exec_price}
        if terminated:
            info.update(self.episode_summary())
        return self._obs(), float(reward), terminated, False, info

    def episode_summary(self) -> dict:
        return {
            "is_bps": (1.0 - self.cash) * BPS,                # implementation shortfall
            "risk_bps": self.risk_total * BPS,                # lambda-weighted variance penalty paid
            "holdings": np.array(self.holdings),
        }

    def _obs(self) -> np.ndarray:
        k = min(self.k, self.N - 1)
        drift = 0.0
        if self.observe_signal:
            drift = self.path.signal[k] / self.signal_scale + self.obs_noise[k]
        move = self.last_move / (self.sigma * np.sqrt(self.tau))
        return np.array([
            self.x,
            (self.N - self.k) / self.N,
            np.clip(drift, -OBS_CLIP, OBS_CLIP),
            np.clip(move, -OBS_CLIP, OBS_CLIP),
        ], dtype=np.float32)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ExecutionEnv()
    check_env(env, skip_render_check=True)
    print("gymnasium check_env: passed")

    # Accounting identity: summed P&L rewards must equal -IS exactly, for any policy.
    rng = np.random.default_rng(1)
    worst = 0.0
    for ep in range(200):
        obs, _ = env.reset(seed=ep)
        pnl_sum, done = 0.0, False
        while not done:
            obs, r, done, _, info = env.step(rng.integers(env.action_space.n))
            pnl_sum += r
        worst = max(worst, abs(pnl_sum + info["is_bps"] + info["risk_bps"]))
        assert info["holdings"][-1] == 0.0, "order must be fully liquidated"
    print(f"return == -(IS + risk) for random policies: max error {worst:.2e} bps")

    # Variance-reduced reward must keep the same expected return (here with the signal ON,
    # and a policy that reacts to it), while cutting the noise.
    import copy
    cfg_sig = copy.deepcopy(env.cfg)
    cfg_sig["signal"]["enabled"] = True
    returns = {}
    for mode in ("realized", "expected"):
        e = ExecutionEnv(cfg_sig, reward_mode=mode)
        rs = []
        for ep in range(4000):
            obs, _ = e.reset(seed=ep)
            total, done = 0.0, False
            while not done:
                a = 2 if obs[2] > 0 else 8          # sell slower when drift is up, faster when down
                obs, r, done, _, _ = e.step(a)
                total += r
            rs.append(total)
        returns[mode] = np.array(rs)
    diff = returns["realized"] - returns["expected"]
    se = diff.std() / np.sqrt(len(diff))
    print(f"mean return realized {returns['realized'].mean():.2f} vs expected {returns['expected'].mean():.2f}"
          f" (diff {diff.mean():.2f} ± {1.96 * se:.2f}); std {returns['realized'].std():.1f} -> "
          f"{returns['expected'].std():.1f} bps")
    assert abs(diff.mean()) < 3 * se
