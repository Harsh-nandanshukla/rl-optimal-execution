"""Evaluation: Experiment A (does RL recover Almgren–Chriss?), Experiment B + ablation (does RL exploit a
predictable drift, and does hiding it remove the edge?) and real NSE days (do the agents transfer?).

    python evaluate.py --self-check                    # bounded check of this file (validation paths only)
    python evaluate.py                                 # Exp A: 10 agents on synthetic validation paths -> logs/expA_val/
    python evaluate.py --exp B                         # Exp B + ablation: 20 agents, signal-on validation -> logs/expB_val/
    python evaluate.py --exp real                      # 30 agents on the real validation days (2022) -> logs/expB_real_val/
    python evaluate.py --models-dir D --steps 20000 --seeds 0 --n-paths 500 --out-dir O   # pilot smoke test
    python evaluate.py --exp B --freeze                # LATER, when authorised: write results/expB_manifest.json
    python evaluate.py --exp B --split test            # once only: synthetic B test set AND real test days

Every strategy of an experiment is scored on the SAME paths. Exp A: ten RL checkpoints (DQN and PPO, seeds
0-4), continuous AC, the two discretised-AC policies on the 12-action grid, TWAP and Immediate. Exp B: ten
signal-observing agents, ten ablation agents (signal generated but hidden), the same baselines plus VWAP and
the signal rule. Real: all thirty agents vs AC, dAC-tracking, TWAP, historical VWAP and Immediate.

Held-out data is reached only through `--split test` after the experiment's freeze manifest verifies: commit,
run list and SHA-256 of every checkpoint, metadata and code file (Exp B also: the raw CSVs and the number of
eligible real test days). Synthetic test paths come from config evaluation.test_seed; real test days
(after data.val_until) are locked in data.py unless allow_test=True. `--exp B --split test` builds every
output in a staging folder and publishes only if both parts succeed. The validation sets picked the
checkpoints, so validation numbers are optimistic and for development only.

Definitions (also written to every *_run_info.json):
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
* signal rule    = reference heuristic (Exp B only, D12): TWAP pace x0.5 while the observed drift is > 0, x2 otherwise.
* signal slope   = least-squares slope of trade_k (% of the order) on the standardised drift z_k = mu_k / strength,
                   k = 0..N-2, after subtracting each decision's mean trade over paths (the schedule's shape is not
                   signal use). Negative = sells less when the price is expected to rise. Static schedules: exactly 0.
* B - ablation   = per matched (algo, seed): paired per-path objective difference, 95% CI over paths. Across seeds:
                   the per-seed differences, mean +- t(0.975, n-1) sd / sqrt(n) (matched: same training seed); also
                   the unmatched two-sample interval, pooled sd, t(0.975, 2n-2).
* real gaps      = paired per stock-day. Headline 95% CI clusters by DATE: the per-date mean difference over stocks,
                   1.96 sd / sqrt(n_dates) (stocks on one date share market moves). Stock-day CI reported alongside.
* VWAP-hist      = VWAP on each stock's mean interval-volume profile over TRAINING days (never the day's own volume).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata as pkg_metadata

import numpy as np

from baselines import (DISCRETE_AC_RULES, ac_holdings, discretised_ac_holdings, discretised_ac_policy,
                       expected_cost_and_variance, immediate_holdings, twap_holdings, vwap_holdings)
from data import CSVPathSampler, SyntheticPathSampler, TestSplitLocked, bucket_day, eligible_day_counts, load_config, scan_days
from train import (HERE, RunSpec, evaluate_policy, evaluate_schedule, make_env, resolve_config, summarize,
                   validation_paths)

LAM, SEEDS, ALGOS, FINAL_STEPS, EVAL_EVERY = 25.0, (0, 1, 2, 3, 4), ("dqn", "ppo"), 300_000, 10_000
REFS = {"ac": "AC", "dac_track": "dAC-tracking", "dac_open": "dAC-open-loop", "twap": "TWAP"}
REFS_B = {**REFS, "rule": "Signal rule"}
REFS_REAL = {"ac": "AC", "dac_track": "dAC-tracking", "twap": "TWAP", "vwap": "VWAP-hist"}
BASELINES = ("AC", "dAC-tracking", "dAC-open-loop", "TWAP", "Immediate")
BASELINES_B = ("AC", "dAC-tracking", "dAC-open-loop", "TWAP", "VWAP", "Immediate", "Signal rule")
BASELINES_REAL = ("AC", "dAC-tracking", "TWAP", "VWAP-hist", "Immediate")
MANIFESTS = {"A": os.path.join(HERE, "results", "expA_manifest.json"),
             "B": os.path.join(HERE, "results", "expB_manifest.json")}
MANIFEST = MANIFESTS["A"]
FROZEN_CONFIG_KEYS = {"A": ("market", "actions", "signal", "evaluation"),
                      "B": ("market", "actions", "signal", "evaluation", "data")}
EXPB_RESULTS = ("expB_summary.csv", "expB_seed_aggregate.csv", "expB_ablation.csv", "expB_trajectories.csv",
                "expB_per_path.csv.gz", "expB_run_info.json", "expB_real_summary.csv", "expB_real_by_stock.csv",
                "expB_real_seed_aggregate.csv", "expB_real_per_path.csv.gz", "expB_real_run_info.json")
EXPB_FIGURES = ("expB_signal.pdf", "expB_signal.png")
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


# ----------------------------------------------------------------------------- the runs
def expected_specs(steps: int = FINAL_STEPS, seeds=SEEDS, n_val: int | None = None, exp: str = "A") -> list[RunSpec]:
    """An experiment's runs, enumerated explicitly (no globbing); all trained on synthetic paths at lambda 25.
    A: signal off. B: signal on and observed (Exp B), then signal on but hidden (ablation). real: A then B."""
    n_val = load_config()["evaluation"]["n_val_paths"] if n_val is None else n_val
    settings = {"A": [(False, True)], "B": [(True, True), (True, False)]}
    settings["real"] = settings["A"] + settings["B"]
    return [RunSpec(algo, "synthetic", signal, observe, LAM, s, steps, EVAL_EVERY, n_val)
            for signal, observe in settings[exp] for algo in ALGOS for s in seeds]


def arm(spec: RunSpec) -> str:
    """A = trained on a random walk; B = drift generated and observed; abl = drift generated but hidden."""
    return "A" if not spec.signal else ("B" if spec.observe_signal else "abl")


def run_files(spec: RunSpec, models_dir: str) -> tuple[str, str]:
    ext = ".pt" if spec.algo == "dqn" else ".zip"
    return os.path.join(models_dir, spec.run_name + ext), os.path.join(models_dir, spec.run_name + ".json")


def check_metadata(meta: dict, spec: RunSpec, cfg: dict) -> list[str]:
    """Every way this run's metadata can disagree with its experiment settings and config.yaml (empty = OK)."""
    want = dict(run_name=spec.run_name, algo=spec.algo, mode="synthetic", signal=spec.signal,
                observe_signal=spec.observe_signal, lam=spec.lam, seed=spec.seed, steps=spec.steps,
                train_reward_mode="expected", eval_reward_mode="realized")
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
                "signal": dict(cfg["signal"], enabled=spec.signal), "evaluation": cfg["evaluation"]}
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
    np.random.default_rng(seed) (train.validation_paths), so held-out paths cannot be drawn by accident.
    Real test days have their own lock (data.CSVPathSampler, allow_test)."""
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


def csv_dir(cfg: dict) -> str:
    d = cfg["data"]["csv_dir"]
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def raw_csv_files(cfg: dict) -> list[str]:
    return sorted(glob.glob(os.path.join(csv_dir(cfg), "*.csv")))


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, HERE).replace(os.sep, "/")
    except ValueError:                                # Windows: another drive has no relative path
        return path.replace(os.sep, "/")


def frozen_specs(exp: str) -> list[RunSpec]:
    """Exp B's freeze also covers the Exp A agents, because the real-data test evaluates all thirty."""
    return expected_specs(exp="real" if exp == "B" else "A")


def frozen_config(cfg: dict, exp: str) -> dict:
    return {k: cfg[k] for k in FROZEN_CONFIG_KEYS[exp]}


def real_test_days(cfg: dict) -> dict[str, int]:
    """Eligible real test days per stock, from timestamps only (no test path is built)."""
    return {stock: c["test"][0] for stock, c in eligible_day_counts(cfg, csv_dir(cfg)).items()}


def manifest_files(models_dir: str, specs: list[RunSpec], cfg: dict | None = None, exp: str = "A") -> dict[str, str]:
    """{repo-relative name: absolute path} of everything the freeze covers: checkpoints, metadata, code
    and, for Exp B, the raw CSVs (gitignored, so their hashes are the only record of the real test data)."""
    files = [p for spec in specs for p in run_files(spec, models_dir)] + [os.path.join(HERE, f) for f in CODE_FILES]
    files += raw_csv_files(cfg) if exp == "B" else []
    return {_rel(p): p for p in files}


def build_manifest(cfg: dict, models_dir: str, commit: str, specs: list[RunSpec] | None = None,
                   exp: str = "A") -> dict:
    specs = specs or frozen_specs(exp)
    m = {
        "experiment": exp, "commit": commit, "runs": [s.run_name for s in specs],
        "files": {rel: file_digest(p) for rel, p in manifest_files(models_dir, specs, cfg, exp).items()},
        "hash_rule": file_digest.__doc__.split("\n")[0].strip() + " " + file_digest.__doc__.split("\n")[1].strip(),
        "test_seed": cfg["evaluation"]["test_seed"], "n_test_paths": cfg["evaluation"]["n_test_paths"],
        "config": frozen_config(cfg, exp),
        "versions": {p: pkg_metadata.version(p) for p in PACKAGES} | {"python": sys.version.split()[0]},
    }
    if exp == "B":
        m["real_test_days"] = real_test_days(cfg)
    return m


def freeze(cfg: dict, models_dir: str, manifest_path: str | None = None, exp: str = "A") -> dict:
    """Record the frozen state. Run only after the checkpoints AND the experiment code are committed."""
    manifest_path = manifest_path or MANIFESTS[exp]
    if os.path.exists(manifest_path):
        raise ManifestError(f"{manifest_path} already exists; a frozen experiment is never re-frozen")
    specs = frozen_specs(exp)
    with forbid_test_paths(cfg):
        load_runs(specs, cfg, models_dir)                             # metadata, settings, reload checks
    rel = [os.path.relpath(p, HERE) for s in specs for p in run_files(s, models_dir)] + list(CODE_FILES)
    dirty = git("status", "--porcelain", "--", *rel)
    if dirty.returncode or dirty.stdout.strip():
        raise ManifestError("checkpoints, metadata and code must be committed before freezing:\n" + dirty.stdout)
    manifest = build_manifest(cfg, models_dir, git("rev-parse", "HEAD").stdout.strip(), specs, exp)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path} for commit {manifest['commit'][:10]} ({len(manifest['files'])} files hashed)")
    return manifest


def verify_manifest(cfg: dict, models_dir: str, manifest_path: str | None = None, check_git: bool = True,
                    specs: list[RunSpec] | None = None, exp: str = "A") -> dict:
    """Every check passes, or ManifestError. Runs before a single test path exists or a test day is bucketed.
    (`specs` and check_git=False exist only so the self-check can exercise it on tiny runs.)"""
    manifest_path = manifest_path or MANIFESTS[exp]
    specs = specs or frozen_specs(exp)
    if not os.path.exists(manifest_path):
        raise ManifestError(f"no manifest at {manifest_path}: freeze the checkpoints first (see README)")
    try:
        with open(manifest_path) as f:
            m = json.load(f)
        runs, files, commit = m["runs"], dict(m["files"]), str(m["commit"])
    except (ValueError, KeyError, TypeError) as e:
        raise ManifestError(f"unreadable manifest: {e!r}") from e
    if m.get("experiment", "A") != exp or runs != [s.run_name for s in specs]:
        raise ManifestError(f"manifest experiment/run list differs from the expected Experiment {exp} runs")
    wanted = manifest_files(models_dir, specs, cfg, exp)
    if set(wanted) != set(files):
        raise ManifestError("manifest file list does not match the expected checkpoints/metadata/code/data files")
    missing = [rel for rel, p in wanted.items() if not os.path.exists(p)]
    if missing:
        raise ManifestError("frozen files are missing: " + ", ".join(missing))
    changed = [rel for rel, p in wanted.items() if file_digest(p) != files[rel]]
    if changed:
        raise ManifestError("files changed since the freeze: " + ", ".join(changed))
    if m.get("config") != frozen_config(cfg, exp):
        raise ManifestError("config.yaml differs from the frozen config")
    if (m.get("test_seed"), m.get("n_test_paths")) != (cfg["evaluation"]["test_seed"], cfg["evaluation"]["n_test_paths"]):
        raise ManifestError("test seed / size differ from the frozen values")
    if exp == "B" and m.get("real_test_days") != real_test_days(cfg):
        raise ManifestError("number of eligible real test days differs from the frozen count")
    if check_git:
        tracked = [rel for rel in files if rel not in {_rel(p) for p in raw_csv_files(cfg)}]
        if git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
            raise ManifestError(f"frozen commit {commit} is not in this repository")
        if git("diff", "--quiet", commit, "--", *tracked).returncode:
            raise ManifestError(f"checkpoint/code files differ from frozen commit {commit[:10]}")
    return m


def test_paths(cfg: dict, models_dir: str, manifest_path: str = MANIFEST) -> list:
    verify_manifest(cfg, models_dir, manifest_path)                   # raises before anything is generated
    spec = expected_specs()[0]
    return validation_paths(resolve_config(cfg, spec), spec, "test")


@contextlib.contextmanager
def staged_publish(targets: list[str]):
    """Yield a staging folder (next to the first target). Only if the block finishes AND every target's file
    exists there are they moved into place; on any error the staging folder is deleted and nothing is
    published, so a failed test run leaves no partial results."""
    existing = [t for t in targets if os.path.exists(t)]
    if existing:
        raise ManifestError("already published (the test set is used once): " + ", ".join(map(_rel, existing)))
    os.makedirs(os.path.dirname(targets[0]), exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".staging_", dir=os.path.dirname(targets[0]))
    try:
        yield stage
        missing = [os.path.basename(t) for t in targets if not os.path.exists(os.path.join(stage, os.path.basename(t)))]
        if missing:
            raise RunError("incomplete outputs, nothing published: " + ", ".join(missing))
        for t in targets:                                             # same volume: each move is a rename
            os.makedirs(os.path.dirname(t), exist_ok=True)
            os.replace(os.path.join(stage, os.path.basename(t)), t)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ----------------------------------------------------------------------------- metrics
def cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    k = math.ceil(alpha * len(x))
    return float(np.sort(x)[-k:].mean())


def trajectory_error(holdings: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """RMSE, bias and dispersion of inventory before each decision (columns 0..N-1); see module docstring."""
    e = holdings[:, :-1] - ref[:-1]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.sqrt(np.mean(e.mean(axis=0) ** 2))),
            "disp": float(np.sqrt(np.mean(e.var(axis=0))))}


def signal_slope(holdings: np.ndarray, signals: np.ndarray, strength: float) -> float:
    """Signal-use diagnostic (module docstring), in % of the order per one standard deviation of drift."""
    trades = -np.diff(holdings, axis=1)[:, :-1]          # decisions 0..N-2; the last one is a forced sale
    z = signals[:, :-1] / strength
    t, zc = trades - trades.mean(axis=0), z - z.mean(axis=0)
    return 100.0 * float((t * zc).sum() / (zc ** 2).sum())


def date_clustered(d: np.ndarray, dates: list[str]) -> tuple[float, float, int]:
    """(mean, 95% CI, n_dates) of the per-date mean of `d` over stocks; dates are the independent units."""
    uniq, inv = np.unique(np.asarray(dates), return_inverse=True)
    per_date = np.bincount(inv, weights=d) / np.bincount(inv)
    return (*summarize(per_date), len(uniq))


def signal_rule_policy(cfg: dict):
    """Exp B reference heuristic (D12; same rule as env.py's check): wait at half TWAP pace while the observed
    drift says the price will rise, sell at double pace otherwise. A hand rule, never the solution."""
    m = list(cfg["actions"]["multipliers"])
    slow, fast = m.index(0.5), m.index(2.0)
    return lambda obs: slow if obs[2] > 0 else fast


def reference_env(cfg: dict, signal: bool, observe: bool = True):
    """(resolved config, env) for these signal settings; the same env is used for every baseline."""
    spec = RunSpec("dqn", "synthetic", signal, observe, LAM, 0, 1, 1, 2)
    rcfg = resolve_config(cfg, spec)
    return rcfg, make_env(rcfg, spec, SyntheticPathSampler(rcfg), "realized")


def agent_envs(cfg: dict):
    """spec -> env with that agent's own observation setting (an ablation agent never sees the drift)."""
    cache = {}

    def get(spec: RunSpec):
        key = (spec.signal, spec.observe_signal)
        if key not in cache:
            cache[key] = reference_env(cfg, *key)[1]
        return cache[key]
    return get


def evaluate_strategies(cfg: dict, runs: dict, paths: list, seed: int, exp: str = "A") -> dict[str, dict]:
    """Per-path results for every baseline and agent on the identical synthetic `paths` (exp "A" or "B")."""
    signal = exp == "B"
    rcfg, env = reference_env(cfg, signal)
    N = env.N
    res = {"AC": evaluate_schedule(env, ac_holdings(rcfg, LAM), paths)}
    for rule, name in zip(DISCRETE_AC_RULES, ("dAC-tracking", "dAC-open-loop")):
        res[name] = evaluate_policy(env, discretised_ac_policy(env, rcfg, LAM, rule), paths, seed)
        assert (res[name]["holdings"] == discretised_ac_holdings(rcfg, LAM, rule)[0]).all(), "dAC must be static"
    res["TWAP"] = evaluate_schedule(env, twap_holdings(N), paths)
    if signal:                                                        # synthetic paths share one volume curve
        res["VWAP"] = evaluate_schedule(env, vwap_holdings(paths[0].volume), paths)
    res["Immediate"] = evaluate_schedule(env, immediate_holdings(N), paths)
    if signal:
        res["Signal rule"] = evaluate_policy(env, signal_rule_policy(rcfg), paths, seed)
    env_of = agent_envs(cfg)
    for name, (spec, _, act) in runs.items():
        print(f"  evaluating {name} on {len(paths)} paths")
        res[name] = evaluate_policy(env_of(spec), act, paths, seed)
    return res


def evaluate_real(cfg: dict, runs: dict, sampler: CSVPathSampler, seed: int) -> dict[str, dict]:
    """Per-day results on real NSE days (simulated impact) for the baselines and every agent."""
    paths, stocks = sampler.days, [label.rsplit(" ", 1)[0] for label in sampler.day_labels]
    rcfg, env = reference_env(cfg, signal=False)
    N = env.N
    res = {"AC": evaluate_schedule(env, ac_holdings(rcfg, LAM), paths),
           "dAC-tracking": evaluate_policy(env, discretised_ac_policy(env, rcfg, LAM, "tracking"), paths, seed),
           "TWAP": evaluate_schedule(env, twap_holdings(N), paths)}
    vwap = {k: np.empty(len(paths)) for k in ("is_bps", "risk_bps", "objective")}
    vwap["holdings"] = np.empty((len(paths), N + 1))
    for stock, profile in sampler.volume_profiles.items():          # one schedule per stock, from training days
        idx = [i for i, s in enumerate(stocks) if s == stock]
        part = evaluate_schedule(env, vwap_holdings(profile), [paths[i] for i in idx])
        for k in vwap:
            vwap[k][idx] = part[k]
    res["VWAP-hist"] = vwap
    res["Immediate"] = evaluate_schedule(env, immediate_holdings(N), paths)
    env_of = agent_envs(cfg)
    for name, (spec, _, act) in runs.items():
        print(f"  evaluating {name} on {len(paths)} real days")
        res[name] = evaluate_policy(env_of(spec), act, paths, seed)
    return res


def _identity(name: str, runs: dict) -> dict:
    spec = runs[name][0] if name in runs else None
    return {"strategy": name, "arm": arm(spec) if spec else "baseline", "algo": spec.algo if spec else "baseline",
            "seed": spec.seed if spec else ""}


def summary_rows(res: dict, runs: dict, refs: dict = REFS, signals: np.ndarray | None = None,
                 strength: float = 1.0) -> list[dict]:
    ac_ref, dac_ref = res["AC"]["holdings"][0], res["dAC-tracking"]["holdings"][0]
    twap_obj, ac_obj = res["TWAP"]["objective"].mean(), res["AC"]["objective"].mean()
    rows = []
    for name, r in res.items():
        row = {**_identity(name, runs), "n_paths": len(r["objective"])}
        row["is_mean"], row["is_ci95"] = summarize(r["is_bps"])
        row["is_std"], row["risk_mean"] = float(r["is_bps"].std(ddof=1)), float(r["risk_bps"].mean())
        row["obj_mean"], row["obj_ci95"] = summarize(r["objective"])
        row["cvar5_is"] = cvar(r["is_bps"])
        for key, ref in refs.items():                                 # paired: same paths, per-path difference
            row[f"gap_{key}_mean"], row[f"gap_{key}_ci95"] = summarize(r["objective"] - res[ref]["objective"])
        for key, ref in (("ac", ac_ref), ("dac_track", dac_ref)):
            row.update({f"{m}_{key}": v for m, v in trajectory_error(r["holdings"], ref).items()})
        row["capture"] = float((twap_obj - r["objective"].mean()) / (twap_obj - ac_obj))
        if signals is not None:
            row["slope_pct"] = signal_slope(r["holdings"], signals, strength)
        rows.append(row)
    return rows


SEED_METRICS = ("obj_mean", "is_mean", "is_std", "risk_mean", "cvar5_is", "gap_ac_mean", "gap_dac_track_mean",
                "gap_dac_open_mean", "gap_rule_mean", "rmse_ac", "bias_ac", "disp_ac", "rmse_dac_track", "capture",
                "slope_pct")
REAL_SEED_METRICS = ("obj_mean", "is_mean", "is_std", "cvar5_is", "gap_ac_mean", "gap_dac_track_mean",
                     "gap_twap_mean", "gap_vwap_mean")


def seed_aggregate(rows: list[dict], metrics=SEED_METRICS) -> list[dict]:
    """Across-seed spread of per-seed means, per (arm, algo): training variability, kept apart from path CIs."""
    out = []
    groups = list(dict.fromkeys((r["arm"], r["algo"]) for r in rows if r["algo"] != "baseline"))
    for group_arm, algo in groups:
        seeds = [r for r in rows if (r["arm"], r["algo"]) == (group_arm, algo)]
        for metric in (m for m in metrics if m in seeds[0]):
            v = np.array([r[metric] for r in seeds])
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            t_ci = T975[len(v) - 1] * sd / math.sqrt(len(v)) if len(v) > 1 else float("nan")
            out.append({"arm": group_arm, "algo": algo, "metric": metric, "n_seeds": len(v), "mean": float(v.mean()),
                        "sd": sd, "t_ci95": t_ci})
    return out


ABLATION_FIELDS = ("algo", "level", "seed", "n", "diff_mean", "ci95", "sd", "two_sample_ci95", "B_gap_ac", "abl_gap_ac")


def ablation_rows(res: dict, runs: dict) -> list[dict]:
    """B minus ablation, per matched (algo, seed) on identical paths, then across seeds (module docstring)."""
    by = {(arm(s), s.algo, s.seed): name for name, (s, _, _) in runs.items()}
    ac = res["AC"]["objective"]
    rows = []
    for algo in ALGOS:
        diffs, b_means, a_means = [], [], []
        for seed in sorted({k[2] for k in by if k[1] == algo}):
            if ("B", algo, seed) not in by or ("abl", algo, seed) not in by:
                continue
            b, a = res[by["B", algo, seed]]["objective"], res[by["abl", algo, seed]]["objective"]
            m, ci = summarize(b - a)
            rows.append(dict.fromkeys(ABLATION_FIELDS, "") | {
                "algo": algo, "level": "seed", "seed": seed, "n": len(b), "diff_mean": m, "ci95": ci,
                "B_gap_ac": float((b - ac).mean()), "abl_gap_ac": float((a - ac).mean())})
            diffs.append(m), b_means.append(b.mean()), a_means.append(a.mean())
        if len(diffs) > 1:
            n, sd = len(diffs), float(np.std(diffs, ddof=1))
            pooled = math.sqrt((np.var(b_means, ddof=1) + np.var(a_means, ddof=1)) / 2)
            rows.append(dict.fromkeys(ABLATION_FIELDS, "") | {
                "algo": algo, "level": "across seeds", "n": n, "diff_mean": float(np.mean(diffs)),
                "ci95": T975[n - 1] * sd / math.sqrt(n), "sd": sd,
                "two_sample_ci95": T975[2 * n - 2] * pooled * math.sqrt(2 / n)})
    return rows


def real_rows(res: dict, runs: dict, labels: list[str]) -> tuple[list[dict], list[dict]]:
    """(all-stock rows with date-clustered and stock-day CIs, per-stock rows)."""
    stocks = [label.rsplit(" ", 1)[0] for label in labels]
    dates = [label.rsplit(" ", 1)[1] for label in labels]
    rows, by_stock = [], []
    for name, r in res.items():
        row = {**_identity(name, runs), "n_days": len(r["objective"]), "n_dates": len(set(dates))}
        row["obj_mean"], row["obj_ci95_days"] = summarize(r["objective"])
        row["is_mean"], row["is_std"] = float(r["is_bps"].mean()), float(r["is_bps"].std(ddof=1))
        row["risk_mean"], row["cvar5_is"] = float(r["risk_bps"].mean()), cvar(r["is_bps"])
        for key, ref in REFS_REAL.items():
            d = r["objective"] - res[ref]["objective"]
            row[f"gap_{key}_mean"], row[f"gap_{key}_ci95"], _ = date_clustered(d, dates)    # headline
            row[f"gap_{key}_mean_days"], row[f"gap_{key}_ci95_days"] = summarize(d)          # secondary
        rows.append(row)
        for stock in dict.fromkeys(stocks):
            idx = np.array([s == stock for s in stocks])
            srow = {**_identity(name, runs), "stock": stock, "n_days": int(idx.sum()),
                    "obj_mean": float(r["objective"][idx].mean()), "is_std": float(r["is_bps"][idx].std(ddof=1))}
            for key, ref in REFS_REAL.items():
                srow[f"gap_{key}_mean"], srow[f"gap_{key}_ci95"] = summarize((r["objective"] - res[ref]["objective"])[idx])
            by_stock.append(srow)
    return rows, by_stock


def trajectory_rows(res: dict) -> list[dict]:
    rows = []
    for name, r in res.items():
        h = r["holdings"]
        for k in range(h.shape[1]):
            rows.append({"strategy": name, "k": k, "mean": float(h[:, k].mean()),
                         "p05": float(np.quantile(h[:, k], 0.05)), "p95": float(np.quantile(h[:, k], 0.95))})
    return rows


def ac_formula_check(res: dict, cfg: dict) -> dict:
    """AC's simulated mean IS vs its closed form. The drift has mean zero, so E[IS] is unchanged by the signal."""
    formula = expected_cost_and_variance(res["AC"]["holdings"][0], cfg)[0]
    mean, ci = summarize(res["AC"]["is_bps"])
    return {"ac_is_formula": float(formula), "ac_is_sim": mean, "ac_is_ci95": ci,
            "within_3ci": bool(abs(mean - formula) < 3 * ci)}


# ----------------------------------------------------------------------------- outputs
def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_per_path(path: str, res: dict, labels: list[str] | None = None) -> None:
    n = len(res["AC"]["objective"])
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"] + (["day"] if labels else []) + [f"{s}:{m}" for s in res for m in ("is_bps", "risk_bps")])
        for i in range(n):
            w.writerow([i] + ([labels[i]] if labels else []) +
                       [repr(float(res[s][m][i])) for s in res for m in ("is_bps", "risk_bps")])


def write_outputs(res: dict, runs: dict, out_dir: str, fig_dir: str, info: dict) -> dict[str, str]:
    """Experiment A files (expA_*)."""
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
    write_per_path(files["per_path.csv.gz"], res)
    np.savez_compressed(files["holdings.npz"], **{s: r["holdings"] for s, r in res.items()})
    with open(files["run_info.json"], "w") as f:
        json.dump(info, f, indent=2)
    files.update(plot_trajectories(res, runs, fig_dir))
    return files


def write_expB(rows: list[dict], res: dict, runs: dict, out_dir: str, fig_dir: str, info: dict) -> dict[str, str]:
    """Experiment B + ablation files (expB_*). No per-path holdings file: adaptive holdings barely compress."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    files = {k: os.path.join(out_dir, f"expB_{k}") for k in
             ("summary.csv", "seed_aggregate.csv", "ablation.csv", "trajectories.csv", "per_path.csv.gz",
              "run_info.json")}
    write_csv(files["summary.csv"], rows)
    for key, content in (("seed_aggregate.csv", seed_aggregate(rows)), ("ablation.csv", ablation_rows(res, runs))):
        if content:
            write_csv(files[key], content)
        else:
            files.pop(key)
    write_csv(files["trajectories.csv"], trajectory_rows(res))
    write_per_path(files["per_path.csv.gz"], res)
    with open(files["run_info.json"], "w") as f:
        json.dump(info, f, indent=2)
    files.update(plot_signal(rows, fig_dir))
    return files


def write_real(rows: list[dict], by_stock: list[dict], res: dict, labels: list[str], out_dir: str,
               info: dict) -> dict[str, str]:
    """Real-data files (expB_real_*)."""
    os.makedirs(out_dir, exist_ok=True)
    files = {k: os.path.join(out_dir, f"expB_real_{k}") for k in
             ("summary.csv", "by_stock.csv", "seed_aggregate.csv", "per_path.csv.gz", "run_info.json")}
    write_csv(files["summary.csv"], rows)
    write_csv(files["by_stock.csv"], by_stock)
    agg = seed_aggregate(rows, REAL_SEED_METRICS)
    if agg:
        write_csv(files["seed_aggregate.csv"], agg)
    else:
        files.pop("seed_aggregate.csv")
    write_per_path(files["per_path.csv.gz"], res, labels)
    with open(files["run_info.json"], "w") as f:
        json.dump(info, f, indent=2)
    return files


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    ax.title.set_fontsize(8)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontsize(7)


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
        _style(ax)
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


def plot_signal(rows: list[dict], fig_dir: str) -> dict[str, str]:
    """(a) per-seed objective gap vs AC (95% path CI) for each arm; (b) per-seed signal-use slope.
    Reference lines: AC (0), AC on the action grid, and the signal rule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {r["strategy"]: r for r in rows}
    arms = [("dqn", "B", "DQN\nsignal"), ("dqn", "abl", "DQN\nablation"), ("ppo", "B", "PPO\nsignal"),
            ("ppo", "abl", "PPO\nablation")]
    colors = {"dqn": "#2a78d6", "ppo": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, metric, ci_key, title, ylabel in (
            (axes[0], "gap_ac_mean", "gap_ac_ci95", "(a) Objective vs AC", "objective minus AC (bps)"),
            (axes[1], "slope_pct", None, "(b) Signal use", "trade slope on drift (% of order / sd)")):
        for x, (algo, group, _) in enumerate(arms):
            seeds = [r for r in rows if (r["algo"], r["arm"]) == (algo, group)]
            jitter = np.linspace(-0.18, 0.18, max(len(seeds), 1))
            for j, r in zip(jitter, seeds):
                ax.errorbar(x + j, r[metric], yerr=r[ci_key] if ci_key else None, fmt="o", ms=3,
                            color=colors[algo], mfc=colors[algo] if group == "B" else "white", lw=0.8, capsize=1.5)
            if seeds:
                ax.hlines(np.mean([r[metric] for r in seeds]), x - 0.3, x + 0.3, color=colors[algo], lw=1.6)
        ax.axhline(0, color="#0b0b0b", lw=0.9, label="AC" if metric == "gap_ac_mean" else "static schedules")
        if metric == "gap_ac_mean":
            ax.axhline(by["dAC-tracking"][metric], color="#52514e", ls="--", lw=0.9, label="AC on action grid")
        ax.axhline(by["Signal rule"][metric], color="#9b9a95", ls=":", lw=1.2, label="signal rule (heuristic)")
        ax.set_xticks(range(len(arms)), [a[2] for a in arms])
        ax.set(title=title, ylabel=ylabel)
        _style(ax)
        ax.legend(fontsize=6, frameon=False, loc="best")
    axes[0].text(0.98, 0.03, "dots: seeds (95% path CI)\nbar: mean over seeds", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=6, color="#52514e")
    fig.tight_layout()
    out = {}
    for ext in ("pdf", "png"):
        out[f"figure.{ext}"] = os.path.join(fig_dir, f"expB_signal.{ext}")
        fig.savefig(out[f"figure.{ext}"], dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    return out


def run_info(split: str, n_paths: int, seed: int, models_dir: str, specs: list[RunSpec], exp: str = "A",
             baselines=BASELINES, extra: dict | None = None) -> dict:
    doc = __doc__.split("Definitions (also written to every *_run_info.json):")[1].strip()
    return {"experiment": exp, "split": split, "path_seed": seed, "n_paths": n_paths,
            "models_dir": _rel(models_dir) if split == "test" else "(dev)",
            "runs": [s.run_name for s in specs], "baselines": list(baselines), "definitions": doc,
            "caution": None if split == "test" else "validation data selected the checkpoints: optimistic, dev only",
            **(extra or {}), "versions": {p: pkg_metadata.version(p) for p in PACKAGES}}


# ----------------------------------------------------------------------------- experiment B and real
def finish_expB(cfg: dict, runs: dict, paths: list, seed: int, split: str, models_dir: str, specs: list[RunSpec],
                out_dir: str, fig_dir: str) -> dict:
    """Evaluate Exp B + ablation on `paths` and write expB_* files."""
    with forbid_test_paths(cfg) if split != "test" else contextlib.nullcontext():
        res = evaluate_strategies(cfg, runs, paths, seed, exp="B")
    rcfg = reference_env(cfg, True)[0]
    signals = np.stack([p.signal for p in paths])
    rows = summary_rows(res, runs, REFS_B, signals, rcfg["signal"]["strength"])
    check = ac_formula_check(res, rcfg)
    info = run_info(split, len(paths), seed, models_dir, specs, "B", BASELINES_B, {
        "signal": rcfg["signal"], "ac_formula_check": check,
        "notes": "ablation agents are evaluated with the drift hidden, as trained; obs[3] (last price move) "
                 "stays visible to every agent and is correlated ~0.30 with the current drift"})
    files = write_expB(rows, res, runs, out_dir, fig_dir, info)
    print_table_b(rows, check)
    return {"res": res, "runs": runs, "rows": rows, "files": files, "signals": signals, "check": check}


def finish_real(cfg: dict, runs: dict, sampler: CSVPathSampler, seed: int, split: str, models_dir: str,
                specs: list[RunSpec], out_dir: str) -> dict:
    """Evaluate the agents on real days and write expB_real_* files."""
    res = evaluate_real(cfg, runs, sampler, seed)
    rows, by_stock = real_rows(res, runs, sampler.day_labels)
    days = {s: sum(label.startswith(s + " ") for label in sampler.day_labels) for s in sampler.volume_profiles}
    info = run_info(split, len(sampler), seed, models_dir, specs, "real", BASELINES_REAL, {
        "data": {**cfg["data"], "csv_dir": _rel(csv_dir(cfg))}, "days_per_stock": days,
        "notes": "impact is simulated with the Almgren-Chriss model (no order-book data); sigma fixed at the "
                 "config value for AC, the risk penalty and observation scaling; real paths carry no drift, so "
                 "Exp B agents see drift = 0: this tests transfer, not Hypothesis 2"})
    files = write_real(rows, by_stock, res, sampler.day_labels, out_dir, info)
    print_table_real(rows)
    return {"res": res, "runs": runs, "rows": rows, "by_stock": by_stock, "files": files}


def run_real_val(cfg: dict, models_dir: str, specs: list[RunSpec], out_dir: str) -> dict:
    """Development only: the validation days (2022). The test days stay locked in data.py."""
    with forbid_test_paths(cfg):
        runs = load_runs(specs, cfg, models_dir)
    sampler = CSVPathSampler(cfg, "val", csv_dir=csv_dir(cfg))
    return finish_real(cfg, runs, sampler, cfg["evaluation"]["val_seed"], "val", models_dir, specs, out_dir)


def run_test_B(cfg: dict, models_dir: str, results_dir: str, fig_dir: str, manifest_path: str | None = None) -> dict:
    """The once-only held-out run: verify everything, then the synthetic Exp B test set AND the real test days,
    staged and published together."""
    targets = [os.path.join(results_dir, f) for f in EXPB_RESULTS] + [os.path.join(fig_dir, f) for f in EXPB_FIGURES]
    existing = [t for t in targets if os.path.exists(t)]
    if existing:
        raise ManifestError("already published (the test set is used once): " + ", ".join(map(_rel, existing)))
    verify_manifest(cfg, models_dir, manifest_path, exp="B")         # before any test path or test day exists
    specs = frozen_specs("B")
    with forbid_test_paths(cfg):
        runs = load_runs(specs, cfg, models_dir)
    seed = cfg["evaluation"]["test_seed"]
    with staged_publish(targets) as stage:
        b_runs = {n: r for n, r in runs.items() if r[0].signal}
        spec = next(iter(b_runs.values()))[0]
        paths = validation_paths(resolve_config(cfg, spec), spec, "test")
        out_b = finish_expB(cfg, b_runs, paths, seed, "test", models_dir, [r[0] for r in b_runs.values()], stage, stage)
        sampler = CSVPathSampler(cfg, "test", csv_dir=csv_dir(cfg), allow_test=True)   # manifest verified above
        out_r = finish_real(cfg, runs, sampler, seed, "test", models_dir, specs, stage)
    print(f"published {len(targets)} files to {_rel(results_dir)}/ and {_rel(fig_dir)}/")
    return {"B": out_b, "real": out_r}


# ----------------------------------------------------------------------------- entry points
def run_eval(cfg: dict, split: str, models_dir: str, specs: list[RunSpec], n_paths: int | None, out_dir: str,
             fig_dir: str, manifest_path: str = MANIFEST, exp: str = "A") -> dict:
    """Synthetic evaluation. Exp A: val or (verified) test. Exp B: val only here; its test is run_test_B."""
    if split == "test":
        assert exp == "A", "Experiment B's test set is run by run_test_B together with the real test days"
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
    if exp == "B":
        return finish_expB(cfg, runs, paths, seed, split, models_dir, specs, out_dir, fig_dir)
    with forbid_test_paths(cfg) if split != "test" else contextlib.nullcontext():
        res = evaluate_strategies(cfg, runs, paths, seed)
    files = write_outputs(res, runs, out_dir, fig_dir, run_info(split, len(paths), seed, models_dir, specs))
    rows = summary_rows(res, runs)
    print_table(rows)
    print("wrote " + ", ".join(sorted(os.path.basename(p) for p in files.values())) + f" to {out_dir}")
    return {"res": res, "runs": runs, "rows": rows, "files": files}


def print_table(rows: list[dict]) -> None:
    print(f"\n{'strategy':<40}{'obj':>8}{'±':>6}{'vs AC':>8}{'±':>6}{'vs dAC':>8}{'±':>6}{'RMSE AC':>9}{'capture':>9}")
    for r in rows:
        print(f"{r['strategy']:<40}{r['obj_mean']:>8.2f}{r['obj_ci95']:>6.2f}{r['gap_ac_mean']:>+8.2f}"
              f"{r['gap_ac_ci95']:>6.2f}{r['gap_dac_track_mean']:>+8.2f}{r['gap_dac_track_ci95']:>6.2f}"
              f"{r['rmse_ac']:>9.4f}{r['capture']:>9.3f}")


def print_table_b(rows: list[dict], check: dict) -> None:
    print(f"\n{'strategy':<40}{'obj':>8}{'±':>6}{'vs AC':>8}{'±':>6}{'vs rule':>9}{'±':>6}{'IS sd':>7}{'CVaR5':>8}{'slope':>7}")
    for r in rows:
        print(f"{r['strategy']:<40}{r['obj_mean']:>8.2f}{r['obj_ci95']:>6.2f}{r['gap_ac_mean']:>+8.2f}"
              f"{r['gap_ac_ci95']:>6.2f}{r['gap_rule_mean']:>+9.2f}{r['gap_rule_ci95']:>6.2f}{r['is_std']:>7.1f}"
              f"{r['cvar5_is']:>8.1f}{r['slope_pct']:>+7.2f}")
    print(f"AC mean IS {check['ac_is_sim']:.2f} ± {check['ac_is_ci95']:.2f} vs closed form {check['ac_is_formula']:.2f}")


def print_table_real(rows: list[dict]) -> None:
    print(f"\n{'strategy':<40}{'obj':>8}{'vs AC':>8}{'±date':>7}{'vs TWAP':>9}{'±date':>7}{'vs VWAP':>9}{'±date':>7}{'IS sd':>7}")
    for r in rows:
        print(f"{r['strategy']:<40}{r['obj_mean']:>8.2f}{r['gap_ac_mean']:>+8.2f}{r['gap_ac_ci95']:>7.2f}"
              f"{r['gap_twap_mean']:>+9.2f}{r['gap_twap_ci95']:>7.2f}{r['gap_vwap_mean']:>+9.2f}"
              f"{r['gap_vwap_ci95']:>7.2f}{r['is_std']:>7.1f}")


def _fixture_csvs(folder: str, rng: np.random.Generator) -> None:
    """Two stocks of random-walk minute bars on 6 train, 4 val and 2 test days (full 09:15-15:29 sessions)."""
    import pandas as pd
    days = pd.bdate_range("2021-12-21", periods=6).append(pd.bdate_range("2022-03-01", periods=4)) \
        .append(pd.bdate_range("2023-02-01", periods=2))
    os.makedirs(folder, exist_ok=True)
    for stock in ("AAA_minute", "BBB_minute"):
        frames = []
        for day in days:
            stamps = day + pd.Timedelta(hours=9, minutes=15) + pd.to_timedelta(np.arange(375), unit="min")
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02 / math.sqrt(375), 375)))
            opens = np.r_[100.0, close[:-1]]
            vol = rng.integers(50, 150, 375) * (1 + 2 * (np.abs(np.arange(375) - 187) / 187) ** 2)
            frames.append(pd.DataFrame({"date": stamps.strftime("%Y-%m-%d %H:%M:%S"), "open": opens,
                                        "high": np.maximum(opens, close), "low": np.minimum(opens, close),
                                        "close": close, "volume": vol.astype(int)}))
        pd.concat(frames).to_csv(os.path.join(folder, f"{stock}.csv"), index=False)


def self_check() -> None:
    """Bounded: tiny runs per algorithm and arm, 200 validation paths, a synthetic real-data fixture.
    Never generates a held-out test path and never loads a real CSV."""
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
        res, rows = a["res"], {r["strategy"]: r for r in a["rows"]}

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
        agg = {(r["arm"], r["algo"], r["metric"]): r for r in seed_aggregate(list(rows.values()))}
        g = [rows[s.run_name]["gap_ac_mean"] for s in specs if s.algo == "dqn"]
        assert math.isclose(agg["A", "dqn", "gap_ac_mean"]["t_ci95"], 12.706 * statistics.stdev(g) / math.sqrt(2))
        assert all(np.isfinite(v) for r in rows.values() for v in r.values() if isinstance(v, float))
        assert all(os.path.exists(p) for p in a["files"].values())
        print("seed aggregate: t-interval matches; all metrics finite; all output files written")

        # 6. Bad metadata or a missing file aborts before anything is evaluated.
        ckpt, meta_path = run_files(specs[0], models)
        meta = json.load(open(meta_path))
        for bad in ({"signal": True}, {"observe_signal": False}, {"lam": 10.0}, {"steps": 300_000},
                    {"reload_check_abs_diff_bps": 0.1},
                    {"resolved_config": {**meta["resolved_config"], "actions": {"multipliers": [0.0, 1.0]}}}):
            assert check_metadata({**meta, **bad}, specs[0], cfg), f"metadata change {list(bad)} not caught"
        os.rename(ckpt, ckpt + ".bak")
        try:
            load_runs(specs, cfg, models)
            raise AssertionError("missing checkpoint not caught")
        except RunError:
            pass
        os.rename(ckpt + ".bak", ckpt)
        print("metadata validation: wrong signal/observe/lambda/steps/reload/actions and missing files are refused")

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
        meta["episodes"] -= 1
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"--split test / manifest refused: {', '.join(labels)}; never generated a test path")

        # 8. The guard itself works.
        try:
            validation_paths(rcfg, specs[0], "test")
            raise AssertionError("test paths were generated inside the guard")
        except TestSetAccess:
            pass
        print("test-path guard: active for the whole self-check; direct test-path generation raises")

        # 9. Experiment B + ablation on signal-on validation paths.
        specs_b = expected_specs(steps=4_000, seeds=(0, 1), n_val=100, exp="B")
        for s in specs_b:
            train_run(RunSpec(**{**s.__dict__, "eval_every": 2_000}), cfg, out_dir=d)
        rb = run_eval(cfg, "val", models, specs_b, 200, os.path.join(d, "outB"), os.path.join(d, "figB"), exp="B")
        res_b, rows_b = rb["res"], {r["strategy"]: r for r in rb["rows"]}
        assert all(np.isfinite(v) for r in rows_b.values() for v in r.values() if isinstance(v, float))
        assert all(os.path.exists(p) for p in rb["files"].values()) and rb["check"]["within_3ci"]
        abl = next(s for s in specs_b if not s.observe_signal)
        hidden = make_env(resolve_config(cfg, abl), abl, SyntheticPathSampler(resolve_config(cfg, abl)), "realized")
        obs, _ = hidden.reset(seed=0, options={"path": validation_paths(resolve_config(cfg, abl), abl)[0]})
        assert obs[2] == 0.0, "ablation env must hide the drift"
        act_abl = rb["runs"][abl.run_name][2]
        b_paths = validation_paths(resolve_config(cfg, abl), RunSpec(**{**abl.__dict__, "n_val": 200}), "val")
        assert np.array_equal(evaluate_policy(hidden, act_abl, b_paths, cfg["evaluation"]["val_seed"])["objective"],
                              res_b[abl.run_name]["objective"]), "ablation agent must be scored with the drift hidden"
        rule = rows_b["Signal rule"]
        assert rule["gap_ac_mean"] + rule["gap_ac_ci95"] < 0 and rule["slope_pct"] < 0, "signal rule should beat AC"
        for static in ("AC", "dAC-tracking", "dAC-open-loop", "TWAP", "VWAP", "Immediate"):
            assert abs(rows_b[static]["slope_pct"]) < 1e-9, f"{static} is static: slope must be 0 (up to rounding)"
        print(f"Exp B: ablation agents scored with drift hidden; signal rule {rule['gap_ac_mean']:+.2f} ± "
              f"{rule['gap_ac_ci95']:.2f} bps vs AC, slope {rule['slope_pct']:+.2f}; static slopes 0; "
              f"AC IS within 3 CI of its closed form")

        # 10. Signal slope and B - ablation against hand calculations (plain Python).
        bname = next(s for s in specs_b if s.observe_signal).run_name
        h, sig, strength = res_b[bname]["holdings"], rb["signals"], cfg["signal"]["strength"]
        n_p, N = h.shape[0], h.shape[1] - 1
        tr = [[float(h[i][k] - h[i][k + 1]) for k in range(N - 1)] for i in range(n_p)]
        zz = [[float(sig[i][k]) / strength for k in range(N - 1)] for i in range(n_p)]
        tm = [statistics.fmean(tr[i][k] for i in range(n_p)) for k in range(N - 1)]
        zm = [statistics.fmean(zz[i][k] for i in range(n_p)) for k in range(N - 1)]
        num = sum((tr[i][k] - tm[k]) * (zz[i][k] - zm[k]) for i in range(n_p) for k in range(N - 1))
        den = sum((zz[i][k] - zm[k]) ** 2 for i in range(n_p) for k in range(N - 1))
        assert math.isclose(rows_b[bname]["slope_pct"], 100 * num / den, rel_tol=1e-9, abs_tol=1e-12)
        abl_rows = ablation_rows(res_b, rb["runs"])
        seed_row = next(r for r in abl_rows if (r["algo"], r["level"], r["seed"]) == ("dqn", "seed", 0))
        b0 = next(s for s in specs_b if (s.algo, s.seed, s.observe_signal) == ("dqn", 0, True)).run_name
        a0 = next(s for s in specs_b if (s.algo, s.seed, s.observe_signal) == ("dqn", 0, False)).run_name
        dd = [float(x) - float(y) for x, y in zip(res_b[b0]["objective"], res_b[a0]["objective"])]
        assert math.isclose(seed_row["diff_mean"], statistics.fmean(dd), rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(seed_row["ci95"], 1.96 * statistics.stdev(dd) / math.sqrt(len(dd)), rel_tol=1e-9)
        across = next(r for r in abl_rows if (r["algo"], r["level"]) == ("dqn", "across seeds"))
        per_seed = [r["diff_mean"] for r in abl_rows if (r["algo"], r["level"]) == ("dqn", "seed")]
        assert math.isclose(across["ci95"], 12.706 * statistics.stdev(per_seed) / math.sqrt(2))
        print("signal slope and B - ablation (per seed and across seeds) match hand calculations")

        # 11. Real data on a synthetic minute-bar fixture (never the real CSVs).
        fx = os.path.join(d, "raw")
        _fixture_csvs(fx, np.random.default_rng(7))
        cfg_fx = {**cfg, "data": {**cfg["data"], "csv_dir": fx}}
        specs_r = specs + specs_b
        rr = run_real_val(cfg_fx, models, specs_r, os.path.join(d, "outR"))
        rows_r = {r["strategy"]: r for r in rr["rows"]}
        assert all(np.isfinite(v) for r in rr["rows"] + rr["by_stock"] for v in r.values() if isinstance(v, float))
        sampler = CSVPathSampler(cfg_fx, "val", csv_dir=fx)
        for stock, profile in sampler.volume_profiles.items():
            idx = [i for i, label in enumerate(sampler.day_labels) if label.startswith(stock + " ")]
            gap = max(np.abs(rr["res"]["VWAP-hist"]["holdings"][i] - vwap_holdings(profile)).max() for i in idx)
            assert gap < 1e-12, f"VWAP-hist holdings differ from the {stock} training profile by {gap}"   # env: 1 - sum(trades)
        own = next(bucket_day(bars, edges)[2] for day, split, bars, edges, prob in
                   scan_days(os.path.join(fx, "AAA_minute.csv"), cfg_fx["data"], 25) if split == "val")
        assert not np.allclose(own / own.sum(), sampler.volume_profiles["AAA_minute"]), "VWAP must not use the day's own volume"
        dates = [label.rsplit(" ", 1)[1] for label in sampler.day_labels]
        dd = [float(x) - float(y) for x, y in zip(rr["res"][bname]["objective"], rr["res"]["AC"]["objective"])]
        per_date = [statistics.fmean(v for v, t in zip(dd, dates) if t == u) for u in sorted(set(dates))]
        assert math.isclose(rows_r[bname]["gap_ac_mean"], statistics.fmean(per_date), rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(rows_r[bname]["gap_ac_ci95"], 1.96 * statistics.stdev(per_date) / math.sqrt(len(per_date)),
                            rel_tol=1e-9)
        real_env = agent_envs(cfg)(next(s for s in specs_b if s.observe_signal))
        obs, _ = real_env.reset(seed=0, options={"path": sampler.days[0]})
        assert obs[2] == 0.0, "real days carry no drift"
        try:
            CSVPathSampler(cfg_fx, "test", csv_dir=fx)
            raise AssertionError("real test days loaded without allow_test")
        except TestSplitLocked:
            pass
        print(f"real (fixture): {len(sampler)} val days; VWAP-hist uses each stock's training profile, not the "
              f"day's volume; date-clustered CI matches a hand calculation; drift obs 0; test days locked")

        # 12. Exp B manifest: CSV hashes and test-day counts are part of the freeze.
        bpath = os.path.join(d, "manifestB.json")
        tinyB = lambda c=cfg_fx: verify_manifest(c, models, bpath, check_git=False, specs=specs_r, exp="B")
        goodB = build_manifest(cfg_fx, models, "0" * 40, specs_r, exp="B")
        with open(bpath, "w") as f:
            json.dump(goodB, f)
        assert tinyB()["real_test_days"] == {"AAA_minute": 2, "BBB_minute": 2}
        labelsB = [refused("missing (run_test_B)", lambda: run_test_B(cfg_fx, models, os.path.join(d, "res"),
                                                                      os.path.join(d, "figs"), os.path.join(d, "none.json")))]
        csv_a = os.path.join(fx, "AAA_minute.csv")
        original = open(csv_a, "rb").read()
        with open(csv_a, "ab") as f:
            f.write(b"2023-02-03 09:15:00,1,1,1,1,1\n")
        labelsB.append(refused("changed raw CSV", tinyB))
        with open(csv_a, "wb") as f:
            f.write(original)
        assert tinyB()
        with open(bpath, "w") as f:
            json.dump({**goodB, "real_test_days": {"AAA_minute": 3, "BBB_minute": 2}}, f)
        labelsB.append(refused("changed test-day count", tinyB))
        with open(bpath, "w") as f:
            json.dump(goodB, f)
        labelsB.append(refused("changed data split", lambda: tinyB({**cfg_fx, "data": {**cfg_fx["data"],
                                                                                       "val_until": "2022-06-30"}})))
        print(f"Exp B manifest refused: {', '.join(labelsB)}")

        # 13. Staged publishing: all files or none.
        pub = [os.path.join(d, "pub", "r", "x.csv"), os.path.join(d, "pub", "r", "y.csv"), os.path.join(d, "pub", "f", "z.png")]
        try:
            with staged_publish(pub) as stage:
                open(os.path.join(stage, "x.csv"), "w").close()
                raise RuntimeError("simulated failure in the real-data part")
        except RuntimeError:
            pass
        assert not any(os.path.exists(p) for p in pub) and os.listdir(os.path.dirname(pub[0])) == []
        try:
            with staged_publish(pub) as stage:
                open(os.path.join(stage, "x.csv"), "w").close()
            raise AssertionError("incomplete outputs were published")
        except RunError:
            pass
        assert not any(os.path.exists(p) for p in pub)
        with staged_publish(pub) as stage:
            for p in pub:
                open(os.path.join(stage, os.path.basename(p)), "w").close()
        assert all(os.path.exists(p) for p in pub) and sorted(os.listdir(os.path.dirname(pub[0]))) == ["x.csv", "y.csv"]
        refused("already published", lambda: staged_publish(pub).__enter__())
        print("staged publish: failure or incomplete output publishes nothing; success publishes all; no overwrite")
    print("\nevaluate.py self-check passed.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", choices=["A", "B", "real"], default="A")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--models-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--steps", type=int, default=FINAL_STEPS, help="training steps in the run names (val only)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="(val only)")
    p.add_argument("--n-paths", type=int, default=None, help="synthetic validation paths (val only; default config)")
    p.add_argument("--out-dir", default=None, help="val only; default logs/exp<A|B|B_real>_val (test: results/)")
    p.add_argument("--freeze", action="store_true", help="write results/exp<A|B>_manifest.json (once, when authorised)")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config()
    try:
        if a.self_check:
            return self_check()
        if a.freeze:
            if a.exp == "real":
                p.error("the real-data agents are frozen by --exp B --freeze (one manifest)")
            freeze(cfg, a.models_dir, exp=a.exp)
            return
        if a.split == "test":
            if (a.steps, a.seeds, a.n_paths, a.out_dir) != (FINAL_STEPS, list(SEEDS), None, None):
                p.error("--split test always uses the final runs, all test data and results/")
            if a.exp == "real":
                p.error("the real test days are evaluated by --exp B --split test, with the synthetic test set")
            if a.exp == "A":
                run_eval(cfg, "test", a.models_dir, expected_specs(), None, os.path.join(HERE, "results"),
                         os.path.join(HERE, "figures"))
            else:
                run_test_B(cfg, a.models_dir, os.path.join(HERE, "results"), os.path.join(HERE, "figures"))
            return
        specs = expected_specs(a.steps, tuple(a.seeds), exp=a.exp)
        if a.exp == "real":
            run_real_val(cfg, a.models_dir, specs, a.out_dir or os.path.join(HERE, "logs", "expB_real_val"))
        else:
            out = a.out_dir or os.path.join(HERE, "logs", f"exp{a.exp}_val")
            run_eval(cfg, "val", a.models_dir, specs, a.n_paths, out, out, exp=a.exp)
    except (RunError, ManifestError, TestSetAccess, TestSplitLocked) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
