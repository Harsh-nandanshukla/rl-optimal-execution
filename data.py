"""Price-path generation and loading.

Two sources of price paths, both exposed through the same `sample()` interface so the
environment never needs to know where the prices come from:

* SyntheticPathSampler — arithmetic random walk, optionally with a predictable AR(1)
  drift (the "signal"). With the signal off this is exactly the Almgren–Chriss world.
* CSVPathSampler — real NSE 1-minute bars, bucketed into the environment's decision
  intervals and normalised so that the arrival price is 1.

Every sampler returns a PricePath: per-interval price increments (in units of the arrival
price), the drift the agent may observe at each decision, and the per-interval volume share.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    with open(full) as f:
        return yaml.safe_load(f)


@dataclass
class PricePath:
    increments: np.ndarray  # (N,) exogenous price change over interval k (excludes our own impact)
    signal: np.ndarray      # (N,) drift known at the start of interval k (daily-return units); zeros if none
    volume: np.ndarray      # (N,) share of the day's volume traded in interval k (sums to 1)


def intraday_volume_profile(n_steps: int) -> np.ndarray:
    """Stylised U-shaped intraday volume curve (heavy at open and close), used when no real volume exists."""
    t = (np.arange(n_steps) + 0.5) / n_steps
    profile = 1.0 + 1.5 * (2 * t - 1) ** 2
    return profile / profile.sum()


class SyntheticPathSampler:
    """increment_k = mu_k * tau + sigma * sqrt(tau) * xi_k,   mu_{k+1} = phi * mu_k + innovation.

    mu_k is the drift during interval k and is known (optionally with noise) when the agent
    decides trade k. With signal disabled, mu = 0 and paths are a pure random walk.
    """

    def __init__(self, cfg: dict):
        m, s = cfg["market"], cfg["signal"]
        self.n = m["n_steps"]
        self.tau = m["horizon"] / self.n
        self.sigma = m["sigma"]
        self.signal_on = bool(s["enabled"])
        self.strength, self.phi = s["strength"], s["phi"]
        self.volume = intraday_volume_profile(self.n)

    def sample(self, rng: np.random.Generator) -> PricePath:
        noise = self.sigma * np.sqrt(self.tau) * rng.standard_normal(self.n)
        mu = np.zeros(self.n)
        if self.signal_on:
            innov_sd = self.strength * np.sqrt(1 - self.phi ** 2)
            mu[0] = self.strength * rng.standard_normal()
            for k in range(1, self.n):
                mu[k] = self.phi * mu[k - 1] + innov_sd * rng.standard_normal()
        return PricePath(increments=mu * self.tau + noise, signal=mu, volume=self.volume.copy())


# ----------------------------------------------------------------------------------------
# Real NSE minute data
# ----------------------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "datetime": ["datetime", "date_time", "timestamp", "date"],
    "time": ["time"],
    "close": ["close", "close_price", "ltp"],
    "volume": ["volume", "vol", "qty"],
}


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for alias in _COLUMN_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def load_minute_csv(path: str) -> pd.DataFrame:
    """Read one minute-bar CSV into a frame indexed by timestamp with `close` and `volume` columns.

    Handles the common Kaggle layouts: a single datetime column, or separate date
    (e.g. 20150401 or 2015-04-01) and time (e.g. 09:16) columns.
    """
    df = pd.read_csv(path)
    dt_col, time_col = _find_col(df, "datetime"), _find_col(df, "time")
    close_col, vol_col = _find_col(df, "close"), _find_col(df, "volume")
    if dt_col is None or close_col is None:
        raise ValueError(f"{path}: need a date/datetime and a close column, found {list(df.columns)}")

    stamp = df[dt_col].astype(str)
    if time_col is not None and time_col != dt_col:
        stamp = stamp + " " + df[time_col].astype(str)
    index = pd.to_datetime(stamp, errors="coerce", format="mixed")
    if getattr(index.dt, "tz", None) is not None:
        index = index.dt.tz_localize(None)

    out = pd.DataFrame({
        "close": pd.to_numeric(df[close_col], errors="coerce").to_numpy(),
        "volume": pd.to_numeric(df[vol_col], errors="coerce").to_numpy() if vol_col else np.nan,
    }, index=index.to_numpy())
    out = out[out.index.notna() & out["close"].notna()].sort_index()
    return out[~out.index.duplicated()]


class CSVPathSampler:
    """Samples whole trading days of real minute data, bucketed into the env's decision intervals."""

    def __init__(self, cfg: dict, split: str = "train", csv_dir: str | None = None,
                 min_bars_per_day: int = 300):
        self.n = cfg["market"]["n_steps"]
        csv_dir = csv_dir or cfg["data"]["csv_dir"]
        files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files in {csv_dir}. See README for the dataset download.")

        lo, hi = {
            "train": (None, cfg["data"]["train_until"]),
            "val": (cfg["data"]["train_until"], cfg["data"]["val_until"]),
            "test": (cfg["data"]["val_until"], None),
        }[split]

        self.days: list[PricePath] = []
        self.day_labels: list[str] = []
        for f in files:
            bars = load_minute_csv(f)
            for day, g in bars.groupby(bars.index.date):
                stamp = pd.Timestamp(day)
                if (lo and stamp <= pd.Timestamp(lo)) or (hi and stamp > pd.Timestamp(hi)):
                    continue
                if len(g) < min_bars_per_day:  # skip half-days and gappy days
                    continue
                self.days.append(self._bucket(g))
                self.day_labels.append(f"{os.path.splitext(os.path.basename(f))[0]} {day}")
        if not self.days:
            raise ValueError(f"No usable trading days for split '{split}'.")

    def _bucket(self, g: pd.DataFrame) -> PricePath:
        close = g["close"].to_numpy()
        edges = np.linspace(0, len(close) - 1, self.n + 1).round().astype(int)
        prices = close[edges] / close[0]                    # normalise: arrival price = 1
        vol = g["volume"].to_numpy()
        if np.isfinite(vol).all() and vol.sum() > 0:
            # volume of the bars whose price moves make up interval k
            share = np.array([vol[edges[k] + 1:edges[k + 1] + 1].sum() for k in range(self.n)])
            share = share / share.sum()
        else:
            share = intraday_volume_profile(self.n)
        return PricePath(increments=np.diff(prices), signal=np.zeros(self.n), volume=share)

    def sample(self, rng: np.random.Generator) -> PricePath:
        return self.days[rng.integers(len(self.days))]

    def __len__(self) -> int:
        return len(self.days)


def make_sampler(cfg: dict, source: str = "synthetic", split: str = "train"):
    if source == "synthetic":
        return SyntheticPathSampler(cfg)
    if source == "real":
        return CSVPathSampler(cfg, split=split)
    raise ValueError(f"unknown source {source!r}")


if __name__ == "__main__":
    cfg = load_config()
    rng = np.random.default_rng(0)
    sampler = SyntheticPathSampler(cfg)
    paths = [sampler.sample(rng) for _ in range(20000)]
    inc = np.stack([p.increments for p in paths])
    tau = cfg["market"]["horizon"] / cfg["market"]["n_steps"]
    print(f"increment std: {inc.std():.5f}  (expected sigma*sqrt(tau) = {cfg['market']['sigma'] * np.sqrt(tau):.5f})")
    print(f"end-of-day return std: {inc.sum(1).std():.4f}  (expected sigma = {cfg['market']['sigma']})")
