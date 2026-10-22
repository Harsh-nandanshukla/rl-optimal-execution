"""Experiment A evaluation: do DQN and PPO recover the Almgren–Chriss schedule in a random-walk market?

    python evaluate.py --self-check                           # bounded check of this file (validation paths only)
    python evaluate.py                                        # 10 final agents on validation paths -> logs/expA_val/
    python evaluate.py --models-dir D --steps 20000 --seeds 0 --n-paths 500 --out-dir O   # pilot smoke test
    python evaluate.py --freeze                               # LATER, when authorised: write results/expA_manifest.json
    python evaluate.py --split test                           # once-only held-out evaluation -> results/, figures/

Every strategy is scored on the SAME price paths: the ten RL checkpoints (DQN and PPO, seeds 0-4),
continuous AC, the two discretised-AC policies on the 12-action grid, TWAP and Immediate.

The held-out test paths (config evaluation.test_seed) are generated only by `--split test`, and only after
the freeze manifest has been verified: commit, run list and SHA-256 of every checkpoint, metadata and code
file. The validation set picked the checkpoints, so validation numbers here are optimistic and are for
development only; reported numbers come from the test split.

Definitions (also written to <out>/expA_run_info.json):
* objective      = is_bps + risk_bps per path (realised IS + realised lambda-weighted risk penalty) = -return.
                   For a static schedule its mean estimates E[IS] + lambda Var[IS]. For an adaptive policy the
                   pathwise penalty lambda sigma^2 tau sum x_k^2 is NOT the variance of realised IS (sd is reported too).
* paired gap     = mean over paths of (objective_RL - objective_ref) on identical paths; 95% CI = 1.96 sd / sqrt(n).
                   Answers: is THIS trained agent better or worse than the reference, beyond price-path noise?
* across seeds   = mean, sd (ddof 1) and t-interval mean +- t(0.975, n-1) sd / sqrt(n) of the per-seed means.
                   Answers: what should we expect from re-running training? With n = 5 seeds it is wide, honestly so.
* trajectory     = normalised inventory x_k before each of the N = 25 decisions (k = 0..24), error e_ik = x_ik - ref_k.
                   RMSE^2 = mean_ik e^2 = bias^2 + dispersion^2, with bias^2 = mean_k (mean_i e_ik)^2 (systematic shape
                   error) and dispersion^2 = mean_k var_i(e_ik) (path-to-path reaction, ddof 0).
* CVaR 5% of IS  = mean of the ceil(0.05 n) largest is_bps values.
* capture        = (obj_TWAP - obj_RL) / (obj_TWAP - obj_AC), means over the same paths; 1 = all of AC's gain over TWAP.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from importlib import metadata as pkg_metadata

import numpy as np

from baselines import (DISCRETE_AC_RULES, ac_holdings, discretised_ac_holdings, discretised_ac_policy,
                       immediate_holdings, twap_holdings)
from data import SyntheticPathSampler, load_config
from train import (HERE, RunSpec, evaluate_policy, evaluate_schedule, make_env, resolve_config, summarize,
                   validation_paths)

LAM, SEEDS, ALGOS, FINAL_STEPS, EVAL_EVERY = 25.0, (0, 1, 2, 3, 4), ("dqn", "ppo"), 300_000, 10_000
REFS = {"ac": "AC", "dac_track": "dAC-tracking", "dac_open": "dAC-open-loop", "twap": "TWAP"}
BASELINES = ("AC", "dAC-tracking", "dAC-open-loop", "TWAP", "Immediate")
MANIFEST = os.path.join(HERE, "results", "expA_manifest.json")
CODE_FILES = ("data.py", "env.py", "baselines.py", "dqn.py", "train.py", "evaluate.py", "config.yaml",
              "requirements.txt")
PACKAGES = ("numpy", "pandas", "PyYAML", "gymnasium", "matplotlib", "torch", "stable_baselines3")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


class RunError(RuntimeError):
    """A checkpoint or its metadata is missing, inconsistent, or fails its reload check."""


class ManifestError(RuntimeError):
    """The freeze manifest is missing, unreadable, stale or does not match the files."""


class TestSetAccess(RuntimeError):
    """Something tried to generate held-out test paths outside a verified --split test run."""


# ----------------------------------------------------------------------------- the ten runs
def expected_specs(steps: int = FINAL_STEPS, seeds=SEEDS, n_val: int | None = None) -> list[RunSpec]:
    """Experiment A runs, enumerated explicitly (no globbing): synthetic, signal off, observed, lambda 25."""
    n_val = load_config()["evaluation"]["n_val_paths"] if n_val is None else n_val
    return [RunSpec(algo, "synthetic", False, True, LAM, s, steps, EVAL_EVERY, n_val) for algo in ALGOS for s in seeds]


def run_files(spec: RunSpec, models_dir: str) -> tuple[str, str]:
    ext = ".pt" if spec.algo == "dqn" else ".zip"
    return os.path.join(models_dir, spec.run_name + ext), os.path.join(models_dir, spec.run_name + ".json")


def check_metadata(meta: dict, spec: RunSpec, cfg: dict) -> list[str]:
    """Every way this run's metadata can disagree with Experiment A and config.yaml (empty list = OK)."""
    want = dict(run_name=spec.run_name, algo=spec.algo, mode="synthetic", signal=False, observe_signal=True,
                lam=spec.lam, seed=spec.seed, steps=spec.steps, train_reward_mode="expected",
                eval_reward_mode="realized")
    problems = [f"{k} = {meta.get(k)!r}, expected {v!r}" for k, v in want.items() if meta.get(k) != v]
    if (meta.get("final") or {}).get("step") != spec.steps:
        problems.append(f"final validated step {(meta.get('final') or {}).get('step')} != {spec.steps}")
    if not (meta.get("best") or {}).get("val_obj_mean"):
        problems.append("no best validation row")
    diff = meta.get("reload_check_abs_diff_bps")
    if diff is None or diff >= 1e-6:
        problems.append(f"training-time reload check failed or missing ({diff})")
    rc = meta.get("resolved_config") or {}
    expected = {"market": dict(cfg["market"], risk_aversion=spec.lam), "actions": cfg["actions"],
                "signal": dict(cfg["signal"], enabled=False), "evaluation": cfg["evaluation"]}
    problems += [f"resolved_config.{k} differs from config.yaml" for k, v in expected.items() if rc.get(k) != v]
    return problems


def load_policy(spec: RunSpec, ckpt: str, meta: dict):
    """Greedy act(obs) -> action, rebuilt from the checkpoint and its saved hyperparameters (CPU only)."""
    if spec.algo == "dqn":
        from dqn import DQNAgent
        hp = dict(meta["hyperparameters"])
        agent = DQNAgent(hp.pop("obs_dim"), hp.pop("n_actions"), **hp)
        agent.load(ckpt)
        return lambda o: agent.act(o, greedy=True)
    from stable_baselines3 import PPO
    model = PPO.load(ckpt, device="cpu")
    return lambda o: int(model.predict(o, deterministic=True)[0])


def load_runs(specs: list[RunSpec], cfg: dict, models_dir: str) -> dict[str, tuple[RunSpec, dict, object]]:
    """Validate every run BEFORE evaluating any of them, then re-score each checkpoint on its own validation
    set: it must reproduce the logged best score exactly, or the file is not the checkpoint that was selected."""
    problems, loaded = [], {}
    for spec in specs:
        ckpt, meta_path = run_files(spec, models_dir)
        missing = [p for p in (ckpt, meta_path) if not os.path.exists(p)]
        if missing:
            problems.append(f"{spec.run_name}: missing {', '.join(missing)}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        problems += [f"{spec.run_name}: {p}" for p in check_metadata(meta, spec, cfg)]
        loaded[spec.run_name] = (spec, meta, ckpt)
    if problems:
        raise RunError("refusing to evaluate:\n  " + "\n  ".join(problems))

    out = {}
    for name, (spec, meta, ckpt) in loaded.items():
        act = load_policy(spec, ckpt, meta)
        rcfg = resolve_config(cfg, spec)
        val_spec = RunSpec(**{**spec.__dict__, "n_val": meta["n_val_actual"]})
        env = make_env(rcfg, spec, SyntheticPathSampler(rcfg), "realized")
        score = summarize(evaluate_policy(env, act, validation_paths(rcfg, val_spec, "val"),
                                          cfg["evaluation"]["val_seed"])["objective"])[0]
        if abs(score - meta["best"]["val_obj_mean"]) >= 1e-6:
            raise RunError(f"{name}: reloaded checkpoint scores {score} on validation, metadata says "
                           f"{meta['best']['val_obj_mean']}")
        out[name] = (spec, meta, act)
    return out


# ----------------------------------------------------------------------------- test-set protection
@contextlib.contextmanager
def forbid_test_paths(cfg: dict):
    """Inside this block, seeding a generator with the test seed raises. Path generation goes through
    np.random.default_rng(seed) (train.validation_paths), so held-out paths cannot be drawn by accident."""
    real, test_seed = np.random.default_rng, cfg["evaluation"]["test_seed"]

    def guarded(seed=None, *args, **kwargs):
        if isinstance(seed, (int, np.integer)) and int(seed) == test_seed:
            raise TestSetAccess("held-out test paths requested outside a verified --split test run")
        return real(seed, *args, **kwargs)

    np.random.default_rng = guarded
    try:
        yield
    finally:
        np.random.default_rng = real


def file_digest(path: str) -> str:
    """SHA-256 that survives Git line-ending conversion (core.autocrlf): JSON is hashed in canonical form,
    other text with LF line endings, binaries (.pt, .zip) byte for byte."""
    with open(path, "rb") as f:
        raw = f.read()
    if path.endswith(".json"):
        raw = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode()
    elif not path.endswith((".pt", ".zip")):
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True)


def manifest_files(models_dir: str, specs: list[RunSpec]) -> dict[str, str]:
    """{repo-relative name: absolute path} of everything the freeze covers: checkpoints, metadata, code."""
    files = [p for spec in specs for p in run_files(spec, models_dir)] + [os.path.join(HERE, f) for f in CODE_FILES]
    return {os.path.relpath(p, HERE).replace(os.sep, "/"): p for p in files}


def build_manifest(cfg: dict, models_dir: str, commit: str, specs: list[RunSpec] | None = None) -> dict:
    specs = specs or expected_specs()
    return {
        "experiment": "A", "commit": commit, "runs": [s.run_name for s in specs],
        "files": {rel: file_digest(p) for rel, p in manifest_files(models_dir, specs).items()},
        "hash_rule": file_digest.__doc__.split("\n")[0].strip() + " " + file_digest.__doc__.split("\n")[1].strip(),
        "test_seed": cfg["evaluation"]["test_seed"], "n_test_paths": cfg["evaluation"]["n_test_paths"],
        "config": {k: cfg[k] for k in ("market", "actions", "signal", "evaluation")},
        "versions": {p: pkg_metadata.version(p) for p in PACKAGES} | {"python": sys.version.split()[0]},
    }


def freeze(cfg: dict, models_dir: str, manifest_path: str = MANIFEST) -> dict:
    """Record the frozen state. Run only after the checkpoints AND the experiment code are committed."""
    if os.path.exists(manifest_path):
        raise ManifestError(f"{manifest_path} already exists; a frozen experiment is never re-frozen")
    with forbid_test_paths(cfg):
        load_runs(expected_specs(), cfg, models_dir)                  # metadata, settings, reload checks
    rel = [os.path.relpath(p, HERE) for s in expected_specs() for p in run_files(s, models_dir)] + list(CODE_FILES)
    dirty = git("status", "--porcelain", "--", *rel)
    if dirty.returncode or dirty.stdout.strip():
        raise ManifestError("checkpoints, metadata and code must be committed before freezing:\n" + dirty.stdout)
    manifest = build_manifest(cfg, models_dir, git("rev-parse", "HEAD").stdout.strip())
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path} for commit {manifest['commit'][:10]} ({len(manifest['files'])} files hashed)")
    return manifest


def verify_manifest(cfg: dict, models_dir: str, manifest_path: str = MANIFEST, check_git: bool = True,
                    specs: list[RunSpec] | None = None) -> dict:
    """Every check passes, or ManifestError. Runs before a single test path exists.
    (`specs` and check_git=False exist only so the self-check can exercise it on tiny runs.)"""
    specs = specs or expected_specs()
    if not os.path.exists(manifest_path):
        raise ManifestError(f"no manifest at {manifest_path}: freeze the checkpoints first (see README)")
    try:
        with open(manifest_path) as f:
            m = json.load(f)
        runs, files, commit = m["runs"], dict(m["files"]), str(m["commit"])
    except (ValueError, KeyError, TypeError) as e:
        raise ManifestError(f"unreadable manifest: {e!r}") from e
    if runs != [s.run_name for s in specs]:
        raise ManifestError("manifest run list differs from the ten expected Experiment A runs")
    wanted = manifest_files(models_dir, specs)
    if set(wanted) != set(files):
        raise ManifestError("manifest file list does not match the expected checkpoints/metadata/code files")
    missing = [rel for rel, p in wanted.items() if not os.path.exists(p)]
    if missing:
        raise ManifestError("frozen files are missing: " + ", ".join(missing))
    changed = [rel for rel, p in wanted.items() if file_digest(p) != files[rel]]
    if changed:
        raise ManifestError("files changed since the freeze: " + ", ".join(changed))
    if m.get("config") != {k: cfg[k] for k in ("market", "actions", "signal", "evaluation")}:
        raise ManifestError("config.yaml differs from the frozen config")
    if (m.get("test_seed"), m.get("n_test_paths")) != (cfg["evaluation"]["test_seed"], cfg["evaluation"]["n_test_paths"]):
        raise ManifestError("test seed / size differ from the frozen values")
    if check_git:
        if git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
            raise ManifestError(f"frozen commit {commit} is not in this repository")
        if git("diff", "--quiet", commit, "--", *files).returncode:
            raise ManifestError(f"checkpoint/code files differ from frozen commit {commit[:10]}")
    return m


def test_paths(cfg: dict, models_dir: str, manifest_path: str = MANIFEST) -> list:
    verify_manifest(cfg, models_dir, manifest_path)                   # raises before anything is generated
    spec = expected_specs()[0]
    return validation_paths(resolve_config(cfg, spec), spec, "test")


# ----------------------------------------------------------------------------- metrics
def cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    k = math.ceil(alpha * len(x))
    return float(np.sort(x)[-k:].mean())


def trajectory_error(holdings: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """RMSE, bias and dispersion of inventory before each decision (columns 0..N-1); see module docstring."""
    e = holdings[:, :-1] - ref[:-1]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.sqrt(np.mean(e.mean(axis=0) ** 2))),
            "disp": float(np.sqrt(np.mean(e.var(axis=0))))}


def evaluate_strategies(cfg: dict, runs: dict, paths: list, seed: int) -> dict[str, dict]:
    """Per-path results for every baseline and agent, all on the identical `paths`."""
    spec = expected_specs()[0]
    rcfg = resolve_config(cfg, spec)
    env = make_env(rcfg, spec, SyntheticPathSampler(rcfg), "realized")
    N = env.N
    res = {"AC": evaluate_schedule(env, ac_holdings(rcfg, LAM), paths)}
    for rule, name in zip(DISCRETE_AC_RULES, ("dAC-tracking", "dAC-open-loop")):
        res[name] = evaluate_policy(env, discretised_ac_policy(env, rcfg, LAM, rule), paths, seed)
        assert (res[name]["holdings"] == discretised_ac_holdings(rcfg, LAM, rule)[0]).all(), "dAC must be static"
    res["TWAP"] = evaluate_schedule(env, twap_holdings(N), paths)
    res["Immediate"] = evaluate_schedule(env, immediate_holdings(N), paths)
    for name, (_, _, act) in runs.items():
        print(f"  evaluating {name} on {len(paths)} paths")
        res[name] = evaluate_policy(env, act, paths, seed)
    return res


def summary_rows(res: dict, runs: dict) -> list[dict]:
    ac_ref, dac_ref = res["AC"]["holdings"][0], res["dAC-tracking"]["holdings"][0]
    twap_obj, ac_obj = res["TWAP"]["objective"].mean(), res["AC"]["objective"].mean()
    rows = []
    for name, r in res.items():
        spec = runs[name][0] if name in runs else None
        row = {"strategy": name, "algo": spec.algo if spec else "baseline", "seed": spec.seed if spec else "",
               "n_paths": len(r["objective"])}
        row["is_mean"], row["is_ci95"] = summarize(r["is_bps"])
        row["is_std"], row["risk_mean"] = float(r["is_bps"].std(ddof=1)), float(r["risk_bps"].mean())
        row["obj_mean"], row["obj_ci95"] = summarize(r["objective"])
        row["cvar5_is"] = cvar(r["is_bps"])
        for key, ref in REFS.items():                                       # paired: same paths, per-path difference
            row[f"gap_{key}_mean"], row[f"gap_{key}_ci95"] = summarize(r["objective"] - res[ref]["objective"])
        for key, ref in (("ac", ac_ref), ("dac_track", dac_ref)):
            row.update({f"{m}_{key}": v for m, v in trajectory_error(r["holdings"], ref).items()})
        row["capture"] = float((twap_obj - r["objective"].mean()) / (twap_obj - ac_obj))
        rows.append(row)
    return rows


SEED_METRICS = ("obj_mean", "is_mean", "is_std", "risk_mean", "cvar5_is", "gap_ac_mean", "gap_dac_track_mean",
                "gap_dac_open_mean", "rmse_ac", "bias_ac", "disp_ac", "rmse_dac_track", "capture")


def seed_aggregate(rows: list[dict]) -> list[dict]:
    """Across-seed spread of per-seed means: training variability, kept separate from path-level CIs."""
    out = []
    for algo in ALGOS:
        seeds = [r for r in rows if r["algo"] == algo]
        for metric in SEED_METRICS if seeds else ():
            v = np.array([r[metric] for r in seeds])
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            t_ci = T975[len(v) - 1] * sd / math.sqrt(len(v)) if len(v) > 1 else float("nan")
            out.append({"algo": algo, "metric": metric, "n_seeds": len(v), "mean": float(v.mean()), "sd": sd,
                        "t_ci95": t_ci})
    return out


def trajectory_rows(res: dict) -> list[dict]:
    rows = []
    for name, r in res.items():
        h = r["holdings"]
        for k in range(h.shape[1]):
            rows.append({"strategy": name, "k": k, "mean": float(h[:, k].mean()),
                         "p05": float(np.quantile(h[:, k], 0.05)), "p95": float(np.quantile(h[:, k], 0.95))})
    return rows


# ----------------------------------------------------------------------------- outputs
def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(res: dict, runs: dict, out_dir: str, fig_dir: str, info: dict) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    rows = summary_rows(res, runs)
    files = {k: os.path.join(out_dir, f"expA_{k}") for k in
             ("summary.csv", "seed_aggregate.csv", "trajectories.csv", "per_path.csv.gz", "holdings.npz",
              "run_info.json")}
    write_csv(files["summary.csv"], rows)
    agg = seed_aggregate(rows)
    if agg:
        write_csv(files["seed_aggregate.csv"], agg)
    else:
        files.pop("seed_aggregate.csv")
    write_csv(files["trajectories.csv"], trajectory_rows(res))
    n = len(res["AC"]["objective"])
    with gzip.open(files["per_path.csv.gz"], "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"] + [f"{s}:{m}" for s in res for m in ("is_bps", "risk_bps")])
        for i in range(n):
            w.writerow([i] + [repr(float(res[s][m][i])) for s in res for m in ("is_bps", "risk_bps")])
    np.savez_compressed(files["holdings.npz"], **{s: r["holdings"] for s, r in res.items()})
    with open(files["run_info.json"], "w") as f:
        json.dump(info, f, indent=2)
    files.update(plot_trajectories(res, runs, fig_dir))
    return files


def plot_trajectories(res: dict, runs: dict, fig_dir: str) -> dict[str, str]:
    """(a) inventory vs decision; (b) deviation from AC. RL: mean of the per-seed mean trajectories, band =
    min-max over seeds; (b) also shades the seed-averaged 5-95% path range, so no single path is shown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ac = res["AC"]["holdings"][0]
    k = np.arange(len(ac))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    refs = [("AC", "AC (continuous)", "#0b0b0b", "-", 1.6), ("dAC-tracking", "AC on action grid", "#52514e", "--", 1.2),
            ("TWAP", "TWAP", "#9b9a95", ":", 1.2)]
    for name, label, color, ls, lw in refs:
        h = res[name]["holdings"][0]
        ax1.plot(k, h, color=color, ls=ls, lw=lw, label=label)
        ax2.plot(k, h - ac, color=color, ls=ls, lw=lw)
    for algo, color, ls in (("dqn", "#2a78d6", "-"), ("ppo", "#eb6834", "-.")):
        names = [n for n in runs if runs[n][0].algo == algo]
        if not names:
            continue
        seed_means = np.stack([res[n]["holdings"].mean(axis=0) for n in names])
        lo, hi = seed_means.min(axis=0), seed_means.max(axis=0)
        p05 = np.mean([np.quantile(res[n]["holdings"] - ac, 0.05, axis=0) for n in names], axis=0)
        p95 = np.mean([np.quantile(res[n]["holdings"] - ac, 0.95, axis=0) for n in names], axis=0)
        label = f"{algo.upper()} ({len(names)} seed{'s' if len(names) > 1 else ''})"
        ax1.plot(k, seed_means.mean(axis=0), color=color, ls=ls, lw=1.6, label=label)
        ax1.fill_between(k, lo, hi, color=color, alpha=0.25, lw=0)
        ax2.plot(k, seed_means.mean(axis=0) - ac, color=color, ls=ls, lw=1.6)
        ax2.fill_between(k, lo - ac, hi - ac, color=color, alpha=0.30, lw=0)
        ax2.fill_between(k, p05, p95, color=color, alpha=0.10, lw=0)
    ax1.set(xlabel="decision k (15-min interval)", ylabel="inventory left (fraction of order)", xlim=(0, k[-1]),
            ylim=(0, 1.02), title="(a) Inventory trajectory")
    ax2.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
    ax2.set(xlabel="decision k", ylabel="inventory minus AC", xlim=(0, k[-1]), title="(b) Deviation from AC")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontsize(7)
    ax1.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax2.text(0.98, 0.03, "dark band: range over seeds\nlight band: 5-95% of paths", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=6, color="#52514e")
    fig.tight_layout()
    out = {}
    for ext in ("pdf", "png"):
        out[f"figure.{ext}"] = os.path.join(fig_dir, f"expA_trajectories.{ext}")
        fig.savefig(out[f"figure.{ext}"], dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    return out


def run_info(split: str, n_paths: int, seed: int, models_dir: str, specs: list[RunSpec]) -> dict:
    doc = __doc__.split("Definitions (also written to <out>/expA_run_info.json):")[1].strip()
    return {"experiment": "A", "split": split, "path_seed": seed, "n_paths": n_paths,
            "models_dir": os.path.relpath(models_dir, HERE).replace(os.sep, "/") if split == "test" else "(dev)",
            "runs": [s.run_name for s in specs], "baselines": list(BASELINES), "definitions": doc,
            "caution": None if split == "test" else "validation paths selected the checkpoints: optimistic, dev only",
            "versions": {p: pkg_metadata.version(p) for p in PACKAGES}}


# ----------------------------------------------------------------------------- entry points
def run_eval(cfg: dict, split: str, models_dir: str, specs: list[RunSpec], n_paths: int | None, out_dir: str,
             fig_dir: str, manifest_path: str = MANIFEST) -> dict:
    if split == "test":
        paths = test_paths(cfg, models_dir, manifest_path)           # verifies the manifest first
        seed = cfg["evaluation"]["test_seed"]
        runs = load_runs(specs, cfg, models_dir)
    else:
        with forbid_test_paths(cfg):
            runs = load_runs(specs, cfg, models_dir)
            n = n_paths or cfg["evaluation"]["n_val_paths"]
            rcfg = resolve_config(cfg, specs[0])
            paths = validation_paths(rcfg, RunSpec(**{**specs[0].__dict__, "n_val": n}), "val")
            seed = cfg["evaluation"]["val_seed"]
    with forbid_test_paths(cfg) if split != "test" else contextlib.nullcontext():
        res = evaluate_strategies(cfg, runs, paths, seed)
    files = write_outputs(res, runs, out_dir, fig_dir, run_info(split, len(paths), seed, models_dir, specs))
    print_table(summary_rows(res, runs))
    print("wrote " + ", ".join(sorted(os.path.basename(p) for p in files.values())) + f" to {out_dir}")
    return {"res": res, "runs": runs, "files": files}


def print_table(rows: list[dict]) -> None:
    print(f"\n{'strategy':<40}{'obj':>8}{'±':>6}{'vs AC':>8}{'±':>6}{'vs dAC':>8}{'±':>6}{'RMSE AC':>9}{'capture':>9}")
    for r in rows:
        print(f"{r['strategy']:<40}{r['obj_mean']:>8.2f}{r['obj_ci95']:>6.2f}{r['gap_ac_mean']:>+8.2f}"
              f"{r['gap_ac_ci95']:>6.2f}{r['gap_dac_track_mean']:>+8.2f}{r['gap_dac_track_ci95']:>6.2f}"
              f"{r['rmse_ac']:>9.4f}{r['capture']:>9.3f}")


def self_check() -> None:
    """Bounded: two tiny runs per algorithm, 200 validation paths. Never generates a held-out test path."""
    import statistics

    from train import run as train_run
    cfg = load_config()
    with tempfile.TemporaryDirectory() as d, forbid_test_paths(cfg):
        models = os.path.join(d, "models")
        specs = expected_specs(steps=4_000, seeds=(0, 1), n_val=100)
        for s in specs:
            train_run(RunSpec(**{**s.__dict__, "eval_every": 2_000}), cfg, out_dir=d)
        a = run_eval(cfg, "val", models, specs, 200, os.path.join(d, "out"), os.path.join(d, "fig"))
        b = run_eval(cfg, "val", models, specs, 200, os.path.join(d, "out2"), os.path.join(d, "fig2"))
        res, rows = a["res"], {r["strategy"]: r for r in summary_rows(a["res"], a["runs"])}

        # 1. Deterministic: a second evaluation reproduces every per-path number and holding exactly.
        assert all(np.array_equal(res[s][m], b["res"][s][m]) for s in res for m in res[s])
        print("repeat evaluation: identical per-path results")

        # 2. Baselines agree with the existing implementations on the same paths.
        rcfg = resolve_config(cfg, specs[0])
        paths = validation_paths(rcfg, RunSpec(**{**specs[0].__dict__, "n_val": 200}), "val")
        env = make_env(rcfg, specs[0], SyntheticPathSampler(rcfg), "realized")
        for name, h in (("AC", ac_holdings(rcfg, LAM)), ("TWAP", twap_holdings(25)), ("Immediate", immediate_holdings(25))):
            assert np.array_equal(evaluate_schedule(env, h, paths)["objective"], res[name]["objective"])
            assert np.array_equal(res[name]["holdings"][0], h)
        for rule, name in zip(DISCRETE_AC_RULES, ("dAC-tracking", "dAC-open-loop")):
            assert np.array_equal(res[name]["holdings"][0], discretised_ac_holdings(rcfg, LAM, rule)[0])
        assert np.allclose(res["Immediate"]["objective"], res["Immediate"]["objective"][0])
        print("baselines: objectives and holdings match baselines.py / train.py on identical paths")

        # 3. Paired CI and CVaR against an independent hand calculation (plain Python, no NumPy).
        name = specs[0].run_name
        d_i = [float(x) - float(y) for x, y in zip(res[name]["objective"], res["AC"]["objective"])]
        mean, half = statistics.fmean(d_i), 1.96 * statistics.stdev(d_i) / math.sqrt(len(d_i))
        assert math.isclose(rows[name]["gap_ac_mean"], mean, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(rows[name]["gap_ac_ci95"], half, rel_tol=1e-9)
        worst = sorted(float(x) for x in res[name]["is_bps"])[-math.ceil(0.05 * 200):]
        assert math.isclose(rows[name]["cvar5_is"], sum(worst) / len(worst), rel_tol=1e-12)
        print(f"paired CI vs AC ({name}): {mean:+.4f} ± {half:.4f} bps matches a hand calculation; CVaR matches")

        # 4. Trajectory metrics: AC vs itself is zero; RMSE^2 = bias^2 + dispersion^2 for every strategy.
        assert rows["AC"]["rmse_ac"] == 0 and rows["AC"]["bias_ac"] == 0 and rows["AC"]["disp_ac"] == 0
        for r in rows.values():
            for key in ("ac", "dac_track"):
                assert math.isclose(r[f"rmse_{key}"] ** 2, r[f"bias_{key}"] ** 2 + r[f"disp_{key}"] ** 2,
                                    rel_tol=1e-9, abs_tol=1e-15)
        assert math.isclose(rows["dAC-tracking"]["rmse_ac"], 0.0028, abs_tol=5e-5)
        assert rows["dAC-tracking"]["disp_ac"] < 1e-15, "a static policy has no path-to-path dispersion"
        print(f"trajectory: AC RMSE vs itself 0; bias^2 + disp^2 = RMSE^2; dAC-tracking RMSE "
              f"{rows['dAC-tracking']['rmse_ac']:.4f} (closed form 0.0028)")

        # 5. Seed aggregate uses the per-seed means and the t quantile for n = 2.
        agg = {(r["algo"], r["metric"]): r for r in seed_aggregate(list(rows.values()))}
        g = [rows[s.run_name]["gap_ac_mean"] for s in specs if s.algo == "dqn"]
        assert math.isclose(agg["dqn", "gap_ac_mean"]["t_ci95"], 12.706 * statistics.stdev(g) / math.sqrt(2))
        assert all(np.isfinite(v) for r in rows.values() for v in r.values() if isinstance(v, float))
        assert all(os.path.exists(p) for p in a["files"].values())
        print("seed aggregate: t-interval matches; all metrics finite; all output files written")

        # 6. Bad metadata or a missing file aborts before anything is evaluated.
        ckpt, meta_path = run_files(specs[0], models)
        meta = json.load(open(meta_path))
        for bad in ({"signal": True}, {"lam": 10.0}, {"steps": 300_000}, {"reload_check_abs_diff_bps": 0.1},
                    {"resolved_config": {**meta["resolved_config"], "actions": {"multipliers": [0.0, 1.0]}}}):
            assert check_metadata({**meta, **bad}, specs[0], cfg), f"metadata change {list(bad)} not caught"
        os.rename(ckpt, ckpt + ".bak")
        try:
            load_runs(specs, cfg, models)
            raise AssertionError("missing checkpoint not caught")
        except RunError:
            pass
        os.rename(ckpt + ".bak", ckpt)
        print("metadata validation: wrong signal/lambda/steps/reload/actions and missing files are refused")

        # 7. --split test refuses without a valid manifest, before any test path is generated (the guard is
        #    still active, so an attempt to draw test paths would raise TestSetAccess, not ManifestError).
        mpath = os.path.join(d, "manifest.json")

        def refused(label, check) -> str:
            try:
                check()
            except ManifestError:
                return label
            raise AssertionError(f"a {label} manifest was accepted")

        def write(m):
            with open(mpath, "w") as f:
                f.write(m if isinstance(m, str) else json.dumps(m))

        cli = lambda: run_eval(cfg, "test", models, expected_specs(), None, d, d, manifest_path=mpath)
        tiny = lambda git_check=False: verify_manifest(cfg, models, mpath, check_git=git_check, specs=specs)
        good = build_manifest(cfg, models, "0" * 40, specs)
        labels = [refused("missing", cli)]
        write("{not json")
        labels.append(refused("corrupt JSON", cli))
        write(good)                                                   # tiny-run manifest: wrong run list for the CLI
        labels.append(refused("wrong run list", cli))
        assert tiny()["commit"] == "0" * 40                           # a correct manifest passes the file checks
        write({**good, "files": {**good["files"], next(iter(good["files"])): "f" * 64}})
        labels.append(refused("tampered hash", tiny))
        write({**good, "config": {**good["config"], "market": {**cfg["market"], "sigma": 0.03}}})
        labels.append(refused("changed config", tiny))
        write(good)
        labels.append(refused("unknown commit", lambda: tiny(True)))
        with open(meta_path, "a") as f:                               # modify a frozen metadata file
            f.write("\n ")
        assert tiny()                                                 # whitespace only: canonical JSON hash unchanged
        meta["episodes"] += 1
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        labels.append(refused("modified metadata", tiny))
        print(f"--split test / manifest refused: {', '.join(labels)}; never generated a test path")

        # 8. The guard itself works.
        try:
            validation_paths(rcfg, specs[0], "test")
            raise AssertionError("test paths were generated inside the guard")
        except TestSetAccess:
            pass
        print("test-path guard: active for the whole self-check; direct test-path generation raises")
    print("\nevaluate.py self-check passed.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", choices=["A"], default="A")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--models-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--steps", type=int, default=FINAL_STEPS, help="training steps in the run names (val only)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="(val only)")
    p.add_argument("--n-paths", type=int, default=None, help="validation paths (val only; default config)")
    p.add_argument("--out-dir", default=None, help="val only; default logs/expA_val (test: results/)")
    p.add_argument("--freeze", action="store_true", help="write results/expA_manifest.json (once, when authorised)")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config()
    try:
        if a.self_check:
            return self_check()
        if a.freeze:
            freeze(cfg, a.models_dir)
            return
        if a.split == "test":
            if (a.steps, a.seeds, a.n_paths, a.out_dir) != (FINAL_STEPS, list(SEEDS), None, None):
                p.error("--split test always uses the ten final runs, all test paths and results/")
            run_eval(cfg, "test", a.models_dir, expected_specs(), None, os.path.join(HERE, "results"),
                     os.path.join(HERE, "figures"))
        else:
            out = a.out_dir or os.path.join(HERE, "logs", "expA_val")
            run_eval(cfg, "val", a.models_dir, expected_specs(a.steps, tuple(a.seeds)), a.n_paths, out, out)
    except (RunError, ManifestError, TestSetAccess) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
