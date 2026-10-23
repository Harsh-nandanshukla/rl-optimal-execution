"""Price-path generation and loading.

Two sources of price paths, both exposed through the same `sample()` interface so the
environment never needs to know where the prices come from:

* SyntheticPathSampler — arithmetic random walk, optionally with a predictable AR(1)
  drift (the "signal"). With the signal off this is exactly the Almgren–Chriss world.
* CSVPathSampler — real NSE 1-minute bars, bucketed by clock time into the environment's
  decision intervals and normalised so that the arrival price is 1 (conventions below).

Every sampler returns a PricePath: per-interval price increments (in units of the arrival
price), the drift the agent may observe at each decision, and the per-interval volume share a
VWAP schedule may use (real data: a profile from training days, never the day's own volume).
"""

from __future__ import annotations

import glob
import os
from collections import Counter
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
    volume: np.ndarray      # (N,) share of the day's volume expected in interval k (sums to 1)


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
#
# Timestamp convention (data audit, 22 Oct): a bar is labelled by its START. A normal day runs from a
# 09:15 bar to a 15:29 bar (375 bars, session 09:15-15:30), so bar t covers [t, t + 1 min) and its
# close is only known at t + 1 min.
#
# Decision edges e_k = 09:15 + 15k min, k = 0..25 (e_25 = 15:30); interval k is [e_k, e_{k+1}).
# * Price at e_0 (arrival) = OPEN of the 09:15 bar.
# * Price at e_k, k >= 1  = CLOSE of the last bar that has FINISHED by e_k (start + 1 min <= e_k):
#   normally the bar starting at e_k - 1 min; if that minute is missing, an earlier bar (stale, never
#   future). A bar stamped e.g. 09:29:06 ends after 09:30, so it is not used for that edge. At e_25 this
#   is the 15:29 bar, whose close is the 15:30 session close: the last interval sees no later data.
# * Volume of interval k = volume of the bars whose START lies in [e_k, e_{k+1}) (09:15 bar -> k = 0).
# * A day is used only with a valid 09:15 bar (start in [09:15, 09:16), positive open), a bar starting
#   at exactly 15:29:00, and at least one bar in every interval. Bars outside the session (Muhurat
#   evening sessions) are ignored. Nothing is filled in beyond "last finished bar".
# ----------------------------------------------------------------------------------------

SESSION_OPEN, SESSION_MINUTES = pd.Timedelta(hours=9, minutes=15), 375
BAR = pd.Timedelta(minutes=1)
SPLITS = ("train", "val", "test")

_COLUMN_ALIASES = {
    "datetime": ["datetime", "date_time", "timestamp", "date"],
    "time": ["time"],
    "open": ["open", "open_price"],
    "close": ["close", "close_price", "ltp"],
    "volume": ["volume", "vol", "qty"],
}


class TestSplitLocked(RuntimeError):
    """Real test days were requested outside the guarded evaluation path (evaluate.py --split test)."""


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for alias in _COLUMN_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def load_minute_csv(path: str) -> pd.DataFrame:
    """Read one minute-bar CSV into a frame indexed by bar-start timestamp with open/close/volume columns.

    Handles the common Kaggle layouts: a single datetime column, or separate date
    (e.g. 20150401 or 2015-04-01) and time (e.g. 09:16) columns.
    """
    df = pd.read_csv(path)
    dt_col, time_col = _find_col(df, "datetime"), _find_col(df, "time")
    open_col, close_col, vol_col = _find_col(df, "open"), _find_col(df, "close"), _find_col(df, "volume")
    if dt_col is None or open_col is None or close_col is None:
        raise ValueError(f"{path}: need date/datetime, open and close columns, found {list(df.columns)}")

    stamp = df[dt_col].astype(str)
    if time_col is not None and time_col != dt_col:
        stamp = stamp + " " + df[time_col].astype(str)
    index = pd.to_datetime(stamp, errors="coerce", format="mixed")
    if getattr(index.dt, "tz", None) is not None:
        index = index.dt.tz_localize(None)

    numeric = lambda col: pd.to_numeric(df[col], errors="coerce").to_numpy()
    out = pd.DataFrame({
        "open": numeric(open_col),
        "close": numeric(close_col),
        "volume": numeric(vol_col) if vol_col else np.nan,
    }, index=index.to_numpy())
    out = out[out.index.notna() & (out["close"] > 0)].sort_index()   # NaN close compares False -> dropped
    return out[~out.index.duplicated()]


def session_edges(day, n: int) -> pd.DatetimeIndex:
    """The n + 1 decision edges of `day`: 09:15, 09:15 + 375/n min, ..., 15:30."""
    start = pd.Timestamp(day) + SESSION_OPEN
    return pd.DatetimeIndex([start + pd.Timedelta(minutes=SESSION_MINUTES * k / n) for k in range(n + 1)])


def session_bars(day_bars: pd.DataFrame, edges: pd.DatetimeIndex) -> pd.DataFrame:
    """The day's bars that start inside the session [e_0, e_N)."""
    return day_bars[(day_bars.index >= edges[0]) & (day_bars.index < edges[-1])]


def day_problem(bars: pd.DataFrame, edges: pd.DatetimeIndex) -> str | None:
    """Why one day's session bars cannot be used (None = usable). Reads timestamps and the opening price only."""
    if bars.empty:
        return "no bars in the 09:15-15:30 session"
    start = bars.index
    if start[0] >= edges[0] + BAR:
        return f"no 09:15 opening bar (first bar {start[0]:%H:%M:%S})"
    if not bars["open"].iloc[0] > 0:
        return "opening bar has no valid open price"
    if start[-1] != edges[-1] - BAR:
        return f"no 15:29 final bar (last bar {start[-1]:%H:%M:%S})"
    empty = np.flatnonzero(np.diff(start.searchsorted(edges)) == 0)
    if empty.size:
        k = int(empty[0])
        return f"no bars in interval {k} [{edges[k]:%H:%M}, {edges[k + 1]:%H:%M})"
    return None


def bucket_day(bars: pd.DataFrame, edges: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Edge prices (N+1,), the start time of the bar each price came from, and interval volumes (N,).

    Only for days that pass `day_problem`."""
    # last bar with start + 1 min <= edge, i.e. start <= edge - 1 min
    used = bars.index.searchsorted(edges[1:] - BAR, side="right") - 1
    assert (used >= 0).all(), "every edge after the open must have a finished bar"
    prices = np.concatenate([[bars["open"].iloc[0]], bars["close"].to_numpy()[used]])
    first = bars.index.searchsorted(edges)                    # first bar starting at or after each edge
    vol = bars["volume"].to_numpy(dtype=float)
    volume = np.array([vol[first[k]:first[k + 1]].sum() for k in range(len(edges) - 1)])
    return prices, bars.index[np.concatenate([[0], used])], volume


def split_of(day, data_cfg: dict) -> str:
    """train: day <= train_until; val: up to val_until; test: everything after."""
    d = pd.Timestamp(day)
    if d <= pd.Timestamp(data_cfg["train_until"]):
        return "train"
    return "val" if d <= pd.Timestamp(data_cfg["val_until"]) else "test"


def scan_days(path: str, data_cfg: dict, n: int):
    """Yield (day, split, session bars, edges, problem) for every calendar day in one CSV."""
    bars = load_minute_csv(path)
    for day, g in bars.groupby(bars.index.date):
        edges = session_edges(day, n)
        s = session_bars(g, edges)
        yield day, split_of(day, data_cfg), s, edges, day_problem(s, edges)


def _csv_files(cfg: dict, csv_dir: str | None) -> list[str]:
    csv_dir = csv_dir or cfg["data"]["csv_dir"]
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {csv_dir}. See README for the dataset download.")
    return files


def _stock(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def eligible_day_counts(cfg: dict, csv_dir: str | None = None) -> dict:
    """{stock: {split: (eligible days, Counter of rejection reasons)}}. Uses timestamps and the opening-price
    validity check only: no path is built, so no price movement of any split (test included) is computed."""
    out = {}
    for f in _csv_files(cfg, csv_dir):
        counts = {s: [0, Counter()] for s in SPLITS}
        for _, split, _, _, problem in scan_days(f, cfg["data"], cfg["market"]["n_steps"]):
            if problem is None:
                counts[split][0] += 1
            else:
                counts[split][1][problem.split(" (")[0]] += 1
        out[_stock(f)] = {s: tuple(v) for s, v in counts.items()}
    return out


class CSVPathSampler:
    """Every eligible trading day of one date split, bucketed by clock time (conventions above).

    Each path's `volume` is that stock's mean interval-volume share over its TRAINING days, i.e. the profile
    a VWAP trader knows in advance; the evaluated day's own realised volume is never put into a path.
    Test days are locked: only evaluate.py's verified --split test passes allow_test=True.
    """

    def __init__(self, cfg: dict, split: str = "train", csv_dir: str | None = None, allow_test: bool = False):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if split == "test" and not allow_test:
            raise TestSplitLocked("real test days are loaded only by evaluate.py --split test, "
                                  "after the freeze manifest verifies")
        self.n = cfg["market"]["n_steps"]
        self.days: list[PricePath] = []
        self.day_labels: list[str] = []
        self.volume_profiles: dict[str, np.ndarray] = {}
        for f in _csv_files(cfg, csv_dir):
            train_shares, picked = [], []
            for day, day_split, bars, edges, problem in scan_days(f, cfg["data"], self.n):
                if problem is not None or day_split not in ("train", split):
                    continue                                   # other splits are never bucketed
                prices, _, volume = bucket_day(bars, edges)
                if day_split == "train" and np.isfinite(volume).all() and volume.sum() > 0:
                    train_shares.append(volume / volume.sum())
                if day_split == split:
                    picked.append((day, prices))
            if not train_shares:
                raise ValueError(f"{_stock(f)}: no training days with volume to estimate the VWAP profile")
            profile = np.mean(train_shares, axis=0)
            self.volume_profiles[_stock(f)] = profile
            for day, prices in picked:
                self.days.append(PricePath(increments=np.diff(prices / prices[0]), signal=np.zeros(self.n),
                                           volume=profile.copy()))
                self.day_labels.append(f"{_stock(f)} {day}")
        if not self.days:
            raise ValueError(f"No usable trading days for split '{split}'.")

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


# ----------------------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------------------

# Fixture day: bar m (start 09:15 + m min) opens at 100 + 0.01 m, closes at 100 + 0.01 (m + 1) and trades
# 1 + m // 15 shares. Hand calculation: price at edge e_k = 100 + 0.15 k, interval k trades 15 (k + 1).
FIXTURE = {  # day: (what it tests, bar edits, expected rejection reason or None)
    "2020-01-06": ("full session", {}, None),
    "2020-01-07": ("09:29 bar missing, 09:44 bar stamped 09:44:30", dict(drop=[14], late={29: 30}), None),
    "2020-01-08": ("missing opening bar", dict(drop=[0]), "no 09:15 opening bar (first bar 09:16:00)"),
    "2020-01-09": ("missing final bar", dict(drop=[374]), "no 15:29 final bar (last bar 15:28:00)"),
    "2020-01-10": ("10:00-10:14 missing", dict(drop=range(45, 60)), "no bars in interval 3 [10:00, 10:15)"),
    "2020-01-13": ("late open 09:35", dict(drop=range(20)), "no 09:15 opening bar (first bar 09:35:00)"),
    "2020-01-14": ("early close 14:43", dict(drop=range(329, 375)), "no 15:29 final bar (last bar 14:43:00)"),
    "2020-01-15": ("evening session only", dict(evening=True), "no bars in the 09:15-15:30 session"),
    "2022-06-01": ("full session (val split)", {}, None),
    "2023-06-01": ("full session (test split)", {}, None),
}


def _fixture_bars(day: str, drop=(), late=None, evening=False) -> pd.DataFrame:
    m = np.arange(60) if evening else np.setdiff1d(np.arange(SESSION_MINUTES), list(drop))
    t0 = pd.Timestamp(day) + (pd.Timedelta(hours=18, minutes=15) if evening else SESSION_OPEN)
    stamps = [t0 + pd.Timedelta(minutes=int(i), seconds=(late or {}).get(int(i), 0)) for i in m]
    return pd.DataFrame({"date": [s.strftime("%Y-%m-%d %H:%M:%S") for s in stamps], "open": 100 + 0.01 * m,
                         "high": 100 + 0.01 * (m + 1), "low": 100 + 0.01 * m, "close": 100 + 0.01 * (m + 1),
                         "volume": 1 + m // 15})


def fixture_check(cfg: dict) -> None:
    import tempfile
    n = cfg["market"]["n_steps"]
    with tempfile.TemporaryDirectory() as d:
        frame = pd.concat([_fixture_bars(day, **edits) for day, (_, edits, _) in FIXTURE.items()])
        frame.to_csv(os.path.join(d, "FIXTURE_minute.csv"), index=False)
        days = {str(day): (bars, edges, problem)
                for day, _, bars, edges, problem in scan_days(os.path.join(d, "FIXTURE_minute.csv"), cfg["data"], n)}

        # 1. Every day is accepted or rejected for exactly the expected reason.
        for day, (what, _, want) in FIXTURE.items():
            got = days[day][2]
            assert got == want, f"{day} ({what}): got {got!r}, expected {want!r}"
            print(f"  {day}  {what:<46} -> {'accepted' if got is None else 'rejected: ' + got}")

        # 2. Boundary prices and volumes match the hand calculation; show which bar was used at each edge.
        k = np.arange(n + 1)
        for day, changed in (("2020-01-06", {}), ("2020-01-07", {1: (13, "09:28:00"), 2: (28, "09:43:00")})):
            prices, used, volume = bucket_day(*days[day][:2])
            want_p = 100 + 0.15 * k
            want_v = 15.0 * (k[:-1] + 1)
            for j, (m, _) in changed.items():
                want_p[j] = 100 + 0.01 * (m + 1)
            if changed:
                want_v[0] = 14                                     # 09:29 missing; 09:44:30 still in interval 1
            assert np.allclose(prices, want_p) and np.allclose(volume, want_v), day
            assert all(f"{used[j]:%H:%M:%S}" == stamp for j, (_, stamp) in changed.items())
            assert (used[1:] + BAR <= days[day][1][1:]).all(), "a price came from a bar that ends after its edge"
            show = sorted({0, 1, 2, 3, n - 1, n} | set(changed))
            print(f"  {day} edges: " + "; ".join(
                f"e{j} {days[day][1][j]:%H:%M} <- {'open' if j == 0 else 'close'} of {used[j]:%H:%M:%S} = {prices[j]:.2f}"
                for j in show))
        print(f"  interval volumes 2020-01-06: {volume[:3].tolist()} ... {volume[-1]:.0f}  (= 15(k+1)); "
              f"2020-01-07 interval 0 = 14")

        # 3. Sampler: right days per split, normalised increments, VWAP profile from training days only.
        train = CSVPathSampler(cfg, "train", csv_dir=d)
        val = CSVPathSampler(cfg, "val", csv_dir=d)
        assert train.day_labels == ["FIXTURE_minute 2020-01-06", "FIXTURE_minute 2020-01-07"]
        assert val.day_labels == ["FIXTURE_minute 2022-06-01"]
        assert np.allclose(train.days[0].increments, 0.0015), "full day: every interval moves 0.15 / 100"
        shares = [v / v.sum() for v in (15.0 * (k[:-1] + 1), np.r_[14.0, 15.0 * (k[1:-1] + 1)])]
        assert np.allclose(val.days[0].volume, np.mean(shares, axis=0)), "val day must carry the TRAIN profile"
        try:
            CSVPathSampler(cfg, "test", csv_dir=d)
            raise AssertionError("test split loaded without allow_test")
        except TestSplitLocked:
            pass
        assert len(CSVPathSampler(cfg, "test", csv_dir=d, allow_test=True)) == 1      # fixture only
        counts = eligible_day_counts(cfg, csv_dir=d)["FIXTURE_minute"]
        assert {s: c[0] for s, c in counts.items()} == {"train": 2, "val": 1, "test": 1}
        print("  sampler: 2 train / 1 val days, increments 0.0015, val day carries the training-day volume "
              "profile; test split refused without allow_test")


def real_data_check(cfg: dict) -> None:
    """Counts eligible days per split (timestamps only), then builds train and val paths — never test."""
    counts = eligible_day_counts(cfg)
    print(f"  {'stock':<20}" + "".join(f"{s:>8}" for s in SPLITS) + "   rejected (all splits)")
    for stock, c in counts.items():
        rejected = sum((c[s][1] for s in SPLITS), Counter())
        print(f"  {stock:<20}" + "".join(f"{c[s][0]:>8}" for s in SPLITS) + "   " +
              ", ".join(f"{r}: {v}" for r, v in sorted(rejected.items())))
    for split in ("train", "val"):
        sampler = CSVPathSampler(cfg, split)
        assert len(sampler) == sum(c[split][0] for c in counts.values())
        inc = np.stack([p.increments for p in sampler.days])
        assert np.isfinite(inc).all()
        for label, p in zip(sampler.day_labels, sampler.days):
            assert np.array_equal(p.volume, sampler.volume_profiles[label.split(" ")[0]])
        print(f"  {split}: {len(sampler)} paths built, all finite; each carries its stock's training volume profile")


if __name__ == "__main__":
    cfg = load_config()
    rng = np.random.default_rng(0)
    sampler = SyntheticPathSampler(cfg)
    paths = [sampler.sample(rng) for _ in range(20000)]
    inc = np.stack([p.increments for p in paths])
    tau = cfg["market"]["horizon"] / cfg["market"]["n_steps"]
    print(f"increment std: {inc.std():.5f}  (expected sigma*sqrt(tau) = {cfg['market']['sigma'] * np.sqrt(tau):.5f})")
    print(f"end-of-day return std: {inc.sum(1).std():.4f}  (expected sigma = {cfg['market']['sigma']})")

    print("\nclock-time loader on a hand-built fixture:")
    fixture_check(cfg)
    if glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["data"]["csv_dir"], "*.csv")):
        print("\nreal NSE CSVs (eligible days per split; test days counted, never bucketed):")
        real_data_check(cfg)
    else:
        print("\nno real CSVs in data/raw: real-data counts skipped")
    print("\ndata.py self-check passed.")
