"""Report figures and the main results table, built only from committed results (no environment, no agent,
no new evaluation). Every number here was already produced by `evaluate.py --split test`.

    python report.py                # figures/fig1..fig4 (300-dpi PNG + PDF) and results/main_table.{csv,md}
    python report.py --self-check   # rebuild into a temp folder and cross-check the numbers against the CSVs

Fig 4 needs the training learning curves. They live in logs/<run>.csv (gitignored); when present they are
copied into results/expA_learning_curves.csv, which is committed, so Fig 4 can be rebuilt without logs/.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
LOGS = os.path.join(HERE, "logs")
SEEDS = range(5)
T975_DF4 = 2.776                      # t(0.975, 4): across-seed interval with 5 seeds (same as evaluate.py)
COL_IN = 3.4                          # single column of a two-column paper, inches
DPI = 300

# One colour per strategy family, used in every figure.
COLORS = {"dqn": "#2a78d6", "ppo": "#eb6834", "AC": "#0b0b0b", "dAC": "#6b6a66", "TWAP": "#8c8c8c",
          "VWAP": "#a8742a", "rule": "#2e9e6b"}
# Marker per experiment arm: Exp A agents squares, signal-observing agents filled dots, ablation open dots.
MARKERS = {"A": ("s", True), "B": ("o", True), "abl": ("o", False)}
ALGO_LABEL = {"dqn": "DQN", "ppo": "PPO"}


def run_name(algo: str, arm: str, seed: int) -> str:
    """Run names as written by train.py (Exp A: no signal; B: signal observed; abl: signal generated, hidden)."""
    tag = {"A": "sigoff_obson", "B": "sigon_obson", "abl": "sigon_obsoff"}[arm]
    return f"{algo}_synth_{tag}_lam25_300k_s{seed}"


def load() -> dict[str, pd.DataFrame]:
    """The committed test-split tables this script reads. Nothing else is opened (except Fig 4's curves)."""
    read = lambda f: pd.read_csv(os.path.join(RESULTS, f))
    return {"A": read("expA_summary.csv"), "A_agg": read("expA_seed_aggregate.csv"),
            "A_traj": read("expA_trajectories.csv"), "B": read("expB_summary.csv"),
            "B_agg": read("expB_seed_aggregate.csv"), "real": read("expB_real_summary.csv"),
            "real_agg": read("expB_real_seed_aggregate.csv")}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42})
    return plt


def save(fig, out_dir: str, stem: str) -> list[str]:
    """300-dpi PNG and PDF at the figure's exact size (no tight bbox, so the width stays one column)."""
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------------------------- figures

def fig1_trajectories(d: dict, out_dir: str, plt) -> list[str]:
    """Exp A: mean inventory of DQN and PPO vs AC and TWAP (top), and the deviation from AC (bottom).
    RL line = mean of the five per-seed mean trajectories; dark band = min-max over seeds;
    light band (bottom) = seed-averaged 5-95% range over paths."""
    t = d["A_traj"]
    traj = lambda name, col="mean": t[t.strategy == name].sort_values("k")[col].to_numpy()
    ac = traj("AC")
    k = np.arange(len(ac))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(COL_IN, 3.3), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    top.plot(k, ac, color=COLORS["AC"], lw=1.4, label="AC (closed form)")
    top.plot(k, traj("TWAP"), color=COLORS["TWAP"], ls=":", label="TWAP")
    for algo, ls in (("dqn", "-"), ("ppo", "-.")):
        names = [run_name(algo, "A", s) for s in SEEDS]
        seed_means = np.stack([traj(n) for n in names])
        p05 = np.mean([traj(n, "p05") - ac for n in names], axis=0)
        p95 = np.mean([traj(n, "p95") - ac for n in names], axis=0)
        mean, lo, hi = seed_means.mean(axis=0), seed_means.min(axis=0), seed_means.max(axis=0)
        color = COLORS[algo]
        top.plot(k, mean, color=color, ls=ls, label=f"{ALGO_LABEL[algo]} (5 seeds)")
        top.fill_between(k, lo, hi, color=color, alpha=0.25, lw=0)
        bot.plot(k, mean - ac, color=color, ls=ls)
        bot.fill_between(k, lo - ac, hi - ac, color=color, alpha=0.35, lw=0)
        bot.fill_between(k, p05, p95, color=color, alpha=0.10, lw=0)
    top.set(ylabel="inventory left", xlim=(0, k[-1]), ylim=(0, 1.02), title="(a) Mean inventory trajectory")
    top.legend(frameon=False, loc="upper right")
    # TWAP is left out of (b): it peaks ~20x higher and would flatten the RL deviations to a line.
    bot.axhline(0, color=COLORS["AC"], lw=1.4, zorder=1)
    bot.set(xlabel="decision k (15-min interval)", ylabel="inventory − AC", title="(b) Deviation from AC",
            ylim=(-0.04, 0.03))
    bot.text(0.99, 0.96, f"TWAP peaks at +{(traj('TWAP') - ac).max():.2f} (off scale)\n"
             "dark band: range over seeds; light: 5–95% of paths", transform=bot.transAxes,
             ha="right", va="top", fontsize=5.5, color=COLORS["dAC"])
    fig.tight_layout(pad=0.4, h_pad=0.6)
    out = save(fig, out_dir, "fig1_expA_trajectory")
    plt.close(fig)
    return out


def seed_dots(ax, rows: pd.DataFrame, x: float, algo: str, arm: str, ci_col: str) -> None:
    """One dot per seed with its 95% CI, and a bar at the mean over seeds."""
    rows = rows.sort_values("seed")
    marker, filled = MARKERS[arm]
    color = COLORS[algo]
    for j, (_, r) in zip(np.linspace(-0.2, 0.2, len(rows)), rows.iterrows()):
        ax.errorbar(x + j, r.gap_ac_mean, yerr=r[ci_col], fmt=marker, ms=3, color=color,
                    mfc=color if filled else "white", mew=0.8, lw=0.7, capsize=1.2)
    ax.hlines(rows.gap_ac_mean.mean(), x - 0.32, x + 0.32, color=color, lw=1.6)


def legend_above(ax, handles: list, ncol: int = 3) -> None:
    """Legend outside the data area, so it can never cover a point (the Day-4 figure's overlap)."""
    ax.legend(handles=handles, frameon=False, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.0, handlelength=1.8, borderaxespad=0.2)


def seed_handles(arms: list[str]) -> list:
    from matplotlib.lines import Line2D
    labels = {"A": "Exp A agent", "B": "signal observed", "abl": "signal hidden (ablation)"}
    out = [Line2D([], [], marker=MARKERS[a][0], ls="", color="#52514e", ms=3.5,
                  mfc="#52514e" if MARKERS[a][1] else "white", label=labels[a]) for a in arms]
    out.append(Line2D([], [], color="#52514e", lw=1.6, label="mean of 5 seeds"))
    return out


def fig2_signal(d: dict, out_dir: str, plt) -> list[str]:
    """Exp B: per-seed objective gap vs AC (95% paired CI over 10,000 paths) for signal-observing and ablation
    agents; references AC (0), TWAP and the signal rule (a heuristic, not a solution)."""
    from matplotlib.lines import Line2D
    b = d["B"]
    ref = lambda name: float(b.loc[b.strategy == name, "gap_ac_mean"].iloc[0])
    groups = [("dqn", "B"), ("ppo", "B"), ("dqn", "abl"), ("ppo", "abl")]
    fig, ax = plt.subplots(figsize=(COL_IN, 2.55))
    ax.axhline(0, color=COLORS["AC"], lw=0.9)
    ax.axhline(ref("TWAP"), color=COLORS["TWAP"], ls=":", lw=1.2)
    ax.axhline(ref("Signal rule"), color=COLORS["rule"], ls="--", lw=1.0)
    for x, (algo, arm) in enumerate(groups):
        seed_dots(ax, b[(b.algo == algo) & (b.arm == arm)], x, algo, arm, "gap_ac_ci95")
    ax.set_xticks(range(4), ["DQN\nsignal", "PPO\nsignal", "DQN\nablation", "PPO\nablation"])
    ax.set(ylabel="objective − AC (bps)", xlim=(-0.6, 3.6), ylim=(-17.5, 9.5))
    ax.grid(axis="x", visible=False)
    legend_above(ax, [Line2D([], [], color=COLORS["AC"], lw=0.9, label="AC"),
                      Line2D([], [], color=COLORS["TWAP"], ls=":", lw=1.2, label="TWAP"),
                      Line2D([], [], color=COLORS["rule"], ls="--", lw=1.0, label="signal rule (heuristic)")]
                 + seed_handles(["B", "abl"]))
    fig.tight_layout(pad=0.4)
    out = save(fig, out_dir, "fig2_expB_signal")
    plt.close(fig)
    return out


def fig3_real(d: dict, out_dir: str, plt) -> list[str]:
    """Real NSE test days: per-seed gap vs AC (95% CI clustered by date) for all 30 agents; references AC, TWAP
    and historical VWAP. Impact is simulated with the AC model."""
    from matplotlib.lines import Line2D
    r = d["real"]
    ref = lambda name: float(r.loc[r.strategy == name, "gap_ac_mean"].iloc[0])
    groups = [("dqn", "A"), ("ppo", "A"), ("dqn", "B"), ("ppo", "B"), ("dqn", "abl"), ("ppo", "abl")]
    fig, ax = plt.subplots(figsize=(COL_IN, 2.55))
    ax.axhline(0, color=COLORS["AC"], lw=0.9)
    ax.axhline(ref("TWAP"), color=COLORS["TWAP"], ls=":", lw=1.2)
    ax.axhline(ref("VWAP-hist"), color=COLORS["VWAP"], ls="--", lw=1.0)
    for x, (algo, arm) in enumerate(groups):
        seed_dots(ax, r[(r.algo == algo) & (r.arm == arm)], x, algo, arm, "gap_ac_ci95")  # date-clustered
    # Algorithm on the tick, training arm as a group label underneath (three groups of two).
    ax.set_xticks(range(6), ["DQN", "PPO"] * 3)
    for x, group in ((0.5, "Exp A agents"), (2.5, "signal observed"), (4.5, "signal hidden")):
        ax.text(x, -0.13, group, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5)
    for x in (1.5, 3.5):
        ax.axvline(x, color="#c3c2b7", lw=0.6, zorder=0)
    ax.set(ylabel="objective − AC (bps)", xlim=(-0.6, 5.6), ylim=(-1.5, 18.5))
    ax.grid(axis="x", visible=False)
    legend_above(ax, [Line2D([], [], color=COLORS["AC"], lw=0.9, label="AC"),
                      Line2D([], [], color=COLORS["TWAP"], ls=":", lw=1.2, label="TWAP"),
                      Line2D([], [], color=COLORS["VWAP"], ls="--", lw=1.0, label="VWAP (hist. profile)"),
                      seed_handles([])[-1]], ncol=4)
    fig.tight_layout(pad=0.4)
    out = save(fig, out_dir, "fig3_real_transfer")
    plt.close(fig)
    return out


def collect_learning_curves(logs_dir: str = LOGS, results_dir: str = RESULTS) -> pd.DataFrame | None:
    """Exp A validation curves: from logs/ when present (and refresh the committed copy), else the copy."""
    copy = os.path.join(results_dir, "expA_learning_curves.csv")
    logs = {(a, s): os.path.join(logs_dir, run_name(a, "A", s) + ".csv") for a in ALGO_LABEL for s in SEEDS}
    if all(os.path.exists(p) for p in logs.values()):
        cols = ["step", "val_obj_mean", "val_obj_ci95", "val_gap_ac_mean", "val_gap_ac_ci95", "is_best"]
        parts = [pd.read_csv(p)[cols].assign(algo=a, seed=s) for (a, s), p in logs.items()]
        curves = pd.concat(parts, ignore_index=True)[["algo", "seed"] + cols]
        curves.to_csv(copy, index=False)
        return curves
    return pd.read_csv(copy) if os.path.exists(copy) else None


def fig4_learning(curves: pd.DataFrame, out_dir: str, plt) -> list[str]:
    """Exp A learning curves on the 1,000 validation paths (the checkpoint-selection set, so optimistic):
    objective minus AC on the same paths, mean over seeds with the min-max range."""
    ymax = 7.5
    fig, ax = plt.subplots(figsize=(COL_IN, 2.1))
    for algo, ls in (("dqn", "-"), ("ppo", "-.")):
        wide = curves[curves.algo == algo].pivot(index="step", columns="seed", values="val_gap_ac_mean")
        steps = wide.index.to_numpy() / 1e3
        ax.plot(steps, wide.mean(axis=1), color=COLORS[algo], ls=ls, label=f"{ALGO_LABEL[algo]} (mean, 5 seeds)")
        ax.fill_between(steps, wide.min(axis=1), wide.max(axis=1), color=COLORS[algo], alpha=0.22, lw=0)
    ax.axhline(0, color=COLORS["AC"], lw=0.9, label="AC")
    ax.set(xlabel="training steps (thousands)", ylabel="val. objective − AC (bps)", xlim=(0, 300), ylim=(-0.3, ymax))
    ax.legend(frameon=False, loc="upper right")
    # Name every off-scale point instead of stretching the axis (one DQN seed has a transient collapse).
    off = curves[curves.val_gap_ac_mean > ymax]
    notes = [f"{ALGO_LABEL[r.algo]} s{r.seed} at {r.step // 1000}k: +{r.val_gap_ac_mean:.1f}" for r in off.itertuples()]
    ax.text(0.99, 0.55, "\n".join(["band: range over seeds", "off scale: " + "; ".join(notes) if notes else ""]),
            transform=ax.transAxes, ha="right", va="center", fontsize=5.5, color=COLORS["dAC"])
    fig.tight_layout(pad=0.4)
    out = save(fig, out_dir, "fig4_learning_curves")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------- table

def agent_cell(agg: pd.DataFrame, algo: str, arm: str | None = None, metric: str = "gap_ac_mean"):
    rows = agg[(agg.algo == algo) & (agg.metric == metric) & ((agg.arm == arm) if arm else True)]
    return float(rows["mean"].iloc[0]), float(rows["t_ci95"].iloc[0])


def baseline_cell(summary: pd.DataFrame, name: str, col: str = "gap_ac"):
    row = summary[summary.strategy == name]
    if row.empty:
        return np.nan, np.nan
    return float(row[f"{col}_mean"].iloc[0]), float(row[f"{col}_ci95"].iloc[0])


def main_table(d: dict) -> pd.DataFrame:
    """Strategy x {Exp A, Exp B synthetic, real NSE}. AC row: objective (bps); every other row: objective minus
    AC on the same paths (lower is better). Baselines: paired 95% CI over paths (synthetic) or clustered by date
    (real); agents: mean and t-interval over 5 training seeds."""
    A, B, R = d["A"], d["B"], d["real"]
    na = (np.nan, np.nan)
    # Real data has no date-clustered CI for an objective level, only the stock-day one.
    ac = R[R.strategy == "AC"].iloc[0]
    ac_real = (float(ac.obj_mean), float(ac.obj_ci95_days))
    rows = [("AC (closed form), objective", "objective", baseline_cell(A, "AC", "obj"), baseline_cell(B, "AC", "obj"),
             ac_real)]
    for label, a_name, b_name, r_name in (("AC on the action grid", "dAC-tracking", "dAC-tracking", "dAC-tracking"),
                                          ("TWAP", "TWAP", "TWAP", "TWAP"),
                                          ("VWAP", None, "VWAP", "VWAP-hist"),
                                          ("Signal rule (heuristic)", None, "Signal rule", None),
                                          ("Immediate", "Immediate", "Immediate", "Immediate")):
        rows.append((label, "gap_vs_AC", baseline_cell(A, a_name) if a_name else na,
                     baseline_cell(B, b_name) if b_name else na, baseline_cell(R, r_name) if r_name else na))
    for algo in ("dqn", "ppo"):
        rows.append((f"{ALGO_LABEL[algo]}, Exp A agent", "gap_vs_AC", agent_cell(d["A_agg"], algo), na,
                     agent_cell(d["real_agg"], algo, "A")))
    for arm, what in (("B", "signal observed"), ("abl", "signal hidden (ablation)")):
        for algo in ("dqn", "ppo"):
            rows.append((f"{ALGO_LABEL[algo]}, {what}", "gap_vs_AC", na, agent_cell(d["B_agg"], algo, arm),
                         agent_cell(d["real_agg"], algo, arm)))
    return pd.DataFrame([{"strategy": s, "quantity": q, "expA_mean": a[0], "expA_ci95": a[1], "expB_mean": b[0],
                          "expB_ci95": b[1], "real_mean": r[0], "real_ci95": r[1]} for s, q, a, b, r in rows])


def table_markdown(t: pd.DataFrame) -> str:
    def cell(m, c, signed=True):
        if np.isnan(m):
            return "—"
        return f"{m:+.2f} ± {c:.2f}" if signed else f"{m:.2f} ± {c:.2f}"
    lines = ["| Strategy | Exp A: random walk | Exp B: AR(1) drift | Real NSE 2023–26 |", "|---|---|---|---|"]
    for _, r in t.iterrows():
        signed = r.quantity != "objective"
        lines.append(f"| {r.strategy} | {cell(r.expA_mean, r.expA_ci95, signed)} | "
                     f"{cell(r.expB_mean, r.expB_ci95, signed)} | {cell(r.real_mean, r.real_ci95, signed)} |")
    notes = ("All values in bps of arrival value; objective = implementation shortfall + λ·risk (λ = 25), lower is "
             "better. First row: AC's objective (± 95% CI over paths; real: over stock-days). Other rows: objective "
             "minus AC on identical paths. Baselines: paired 95% CI over 10,000 held-out synthetic paths, or over "
             "803 real test dates (clustered by date, 2,409 stock-days). Agents: mean ± t-interval over 5 training "
             "seeds (df 4). Exp A agents were trained without a signal and are not evaluated on Exp B paths. VWAP: "
             "synthetic volume profile in Exp B; each stock's training-day profile on real data. Real-data impact "
             "is simulated with the AC model.")
    return "\n".join(lines) + "\n\n" + notes + "\n"


def write_table(d: dict, results_dir: str) -> list[str]:
    t = main_table(d)
    csv_path, md_path = os.path.join(results_dir, "main_table.csv"), os.path.join(results_dir, "main_table.md")
    t.to_csv(csv_path, index=False, float_format="%.6g")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(table_markdown(t))
    return [csv_path, md_path]


def build(fig_dir: str = FIGURES, results_dir: str = RESULTS, logs_dir: str = LOGS) -> list[str]:
    plt = setup_matplotlib()
    d = load()
    out = fig1_trajectories(d, fig_dir, plt) + fig2_signal(d, fig_dir, plt) + fig3_real(d, fig_dir, plt)
    curves = collect_learning_curves(logs_dir, results_dir)
    if curves is None:
        print("Fig 4 skipped: no learning curves in logs/ or results/expA_learning_curves.csv")
    else:
        out += fig4_learning(curves, fig_dir, plt)
        out.append(os.path.join(results_dir, "expA_learning_curves.csv"))
    return out + write_table(d, results_dir)


# ---------------------------------------------------------------------------------------------- self-check

def self_check() -> None:
    """Rebuild everything into a temp folder and check it against the committed CSVs."""
    import shutil
    import matplotlib.image as mpimg
    d = load()
    with tempfile.TemporaryDirectory() as tmp:
        res_tmp = os.path.join(tmp, "results")
        os.makedirs(res_tmp)
        if os.path.exists(os.path.join(RESULTS, "expA_learning_curves.csv")):
            shutil.copy(os.path.join(RESULTS, "expA_learning_curves.csv"), res_tmp)
        files = build(tmp, res_tmp, LOGS)
        # 1. Figures: one column wide at 300 dpi, PDFs non-empty.
        figs = [f for f in files if f.endswith((".png", ".pdf"))]
        assert len(figs) == 8, figs
        for f in figs:
            assert os.path.getsize(f) > 10_000, f
            if f.endswith(".png"):
                w = mpimg.imread(f).shape[1]
                assert w == round(COL_IN * DPI), (f, w)
        # 2. Agent cells equal mean and t-interval recomputed from the per-seed rows.
        t = pd.read_csv(os.path.join(res_tmp, "main_table.csv"))
        checks = [("DQN, Exp A agent", "expA", d["A"], "dqn", None), ("PPO, signal observed", "expB", d["B"], "ppo", "B"),
                  ("DQN, signal hidden (ablation)", "real", d["real"], "dqn", "abl")]
        for label, col, summary, algo, arm in checks:
            seeds = summary[(summary.algo == algo) & ((summary.arm == arm) if arm else True)].gap_ac_mean.to_numpy()
            assert len(seeds) == 5
            row = t[t.strategy == label].iloc[0]
            assert abs(row[f"{col}_mean"] - seeds.mean()) < 1e-4      # CSV keeps 6 significant figures
            assert abs(row[f"{col}_ci95"] - T975_DF4 * seeds.std(ddof=1) / np.sqrt(5)) < 1e-3
        # 3. The real-data baseline CI is the date-clustered one, recomputed from the per-path file.
        rp = pd.read_csv(os.path.join(RESULTS, "expB_real_per_path.csv.gz"))
        diff = (rp["TWAP:is_bps"] + rp["TWAP:risk_bps"]) - (rp["AC:is_bps"] + rp["AC:risk_bps"])
        per_date = diff.groupby(rp.day.str.rsplit(" ", n=1).str[1]).mean()
        ci = 1.96 * per_date.std(ddof=1) / np.sqrt(len(per_date))
        assert len(per_date) == 803 and abs(t.loc[t.strategy == "TWAP", "real_ci95"].iloc[0] - ci) < 1e-4
        # 4. Exp A trajectories: AC starts at the full order and ends flat; AC row holds the objective.
        ac = d["A_traj"][d["A_traj"].strategy == "AC"].sort_values("k")["mean"].to_numpy()
        assert ac[0] == 1.0 and ac[-1] == 0.0 and np.all(np.diff(ac) <= 0)
        assert t.loc[0, "quantity"] == "objective" and abs(t.loc[0, "expA_mean"] - 49.339) < 1e-2
        # 5. Learning curves: the committed copy equals logs/ (when logs/ is present).
        if os.path.exists(os.path.join(LOGS, run_name("dqn", "A", 0) + ".csv")):
            a = pd.read_csv(os.path.join(res_tmp, "expA_learning_curves.csv"))
            assert len(a) == 300 and a.step.min() == 10000 and a.step.max() == 300000
    # 6. Reporting only: no simulator, agent or evaluation code was imported.
    for mod in ("env", "evaluate", "dqn", "train", "torch", "stable_baselines3"):
        assert mod not in sys.modules, mod
    print("report.py self-check passed: figure sizes, table vs per-seed rows, date-clustered CI, no env/agents.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    if a.self_check:
        return self_check()
    for f in build():
        print("wrote", os.path.relpath(f, HERE))


if __name__ == "__main__":
    main()
