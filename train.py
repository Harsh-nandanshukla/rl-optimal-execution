"""Training entry point: one CLI for DQN (ours, from scratch) and PPO (stable-baselines3).

    python train.py --algo dqn --mode synthetic --signal off --observe-signal on --lam 25 --seed 0 --steps 300000
    python train.py --self-check          # bounded end-to-end check of this file

Every run periodically scores its GREEDY policy on a fixed validation set and keeps the best
checkpoint. Validation always uses realised implementation shortfall from cash flows and the
original, unscaled objective  is_bps + risk_bps  (= minus the episode return, see env.py), no
matter how the training reward was shaped (variance-reduced, or scaled by 0.01 for learning).

Outputs (run name encodes every setting, e.g. dqn_synth_sigoff_obson_lam25_300k_s0):
    models/<run>.pt|.zip   best checkpoint (deliverable, committed)
    models/<run>.json      run metadata: settings, resolved config, hyperparameters, best result
    logs/<run>.csv         learning curve, one row per validation (gitignored)

The evaluation helpers below (validation_paths, evaluate_policy, summarize, ...) are written to be
reused by evaluate.py for the held-out test set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from baselines import all_baselines, run_schedule
from data import CSVPathSampler, SyntheticPathSampler, load_config
from dqn import DQNAgent, TrainMetrics, set_seed, train_dqn
from env import ExecutionEnv

HERE = os.path.dirname(os.path.abspath(__file__))

# PPO settings are fixed in advance (not tuned per experiment) and written to every run's metadata.
# n_steps = 80 whole episodes of 25 steps; batch_size divides n_steps exactly.
PPO_REWARD_SCALE = 0.01          # same reason as DQN's internal scale (DECISIONS.md D19)
PPO_HP = dict(learning_rate=3e-4, n_steps=2000, batch_size=100, n_epochs=10, gamma=1.0,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              normalize_advantage=True)
PPO_NET = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn="ReLU")

CSV_COLUMNS = ["step", "episodes", "epsilon", "train_return_mean", "train_return_std",
               "val_is_mean", "val_obj_mean", "val_obj_ci95", "val_gap_ac_mean", "val_gap_ac_ci95",
               "best_val_obj", "is_best", "wall_s"]
OBJECTIVE_DEF = "mean over validation paths of is_bps + risk_bps (realised IS + lambda-weighted risk, bps)"
CI_DEF = "95% CI half-width of the mean over paths: 1.96 * sample std (ddof=1) / sqrt(n_paths)"


@dataclass(frozen=True)
class RunSpec:
    algo: str             # "dqn" | "ppo"
    mode: str             # "synthetic" | "real"
    signal: bool          # generate the AR(1) drift in the price paths
    observe_signal: bool  # let the agent see it (False = ablation)
    lam: float
    seed: int
    steps: int
    eval_every: int
    n_val: int

    @property
    def run_name(self) -> str:
        onoff = lambda b: "on" if b else "off"
        lam = f"{self.lam:g}".replace(".", "p")
        steps = f"{self.steps // 1000}k" if self.steps % 1000 == 0 else str(self.steps)
        mode = "synth" if self.mode == "synthetic" else "real"
        return (f"{self.algo}_{mode}_sig{onoff(self.signal)}_obs{onoff(self.observe_signal)}"
                f"_lam{lam}_{steps}_s{self.seed}")


# ----------------------------------------------------------------------------- config and envs
def resolve_config(cfg: dict, spec: RunSpec, data_dir: str | None = None) -> dict:
    """Deep copy of config.yaml with this run's overrides; the loaded dict and the file are untouched."""
    out = copy.deepcopy(cfg)
    out["signal"]["enabled"] = spec.signal       # the sampler reads this flag; there is no constructor arg
    out["market"]["risk_aversion"] = spec.lam    # recorded for clarity; lam is ALSO passed to the env
    csv_dir = data_dir or out["data"]["csv_dir"]
    out["data"]["csv_dir"] = csv_dir if os.path.isabs(csv_dir) else os.path.join(HERE, csv_dir)
    return out


def portable_config(cfg: dict) -> dict:
    """Copy for committed metadata: data.csv_dir relative to the repo, so no local absolute path is recorded.
    Loading still uses the absolute path from resolve_config; this only changes what is written down."""
    out = copy.deepcopy(cfg)
    try:
        out["data"]["csv_dir"] = os.path.relpath(out["data"]["csv_dir"], HERE).replace(os.sep, "/")
    except ValueError:                               # Windows: a --data-dir on another drive has no relative path
        pass
    return out


class DataError(RuntimeError):
    """Real-data input is missing or unusable. Never fall back to fabricated data."""


def load_real(cfg: dict, split: str) -> CSVPathSampler:
    try:
        return CSVPathSampler(cfg, split=split)
    except (OSError, ValueError, KeyError) as e:      # missing dir/files, bad columns, no usable days
        raise DataError(f"real-data split '{split}' unusable ({cfg['data']['csv_dir']}): {e}") from e


def make_sampler(cfg: dict, spec: RunSpec, split: str):
    return SyntheticPathSampler(cfg) if spec.mode == "synthetic" else load_real(cfg, split)


def make_env(cfg: dict, spec: RunSpec, sampler, reward_mode: str) -> ExecutionEnv:
    return ExecutionEnv(cfg, sampler=sampler, observe_signal=spec.observe_signal, action_mode="discrete",
                        risk_aversion=spec.lam, reward_mode=reward_mode)


def validation_paths(cfg: dict, spec: RunSpec, split: str = "val") -> list:
    """Synthetic: a fixed path set drawn from the dedicated seed. Real: every day of the date split."""
    if spec.mode == "real":
        return list(load_real(cfg, split).days)
    ev = cfg["evaluation"]
    seed, n = (ev["val_seed"], spec.n_val) if split == "val" else (ev["test_seed"], ev["n_test_paths"])
    rng = np.random.default_rng(seed)               # own generator: never touches the training stream
    sampler = SyntheticPathSampler(cfg)
    return [sampler.sample(rng) for _ in range(n)]


# ----------------------------------------------------------------------------- evaluation (reusable)
def evaluate_policy(env: ExecutionEnv, act, paths: list, seed: int) -> dict[str, np.ndarray]:
    """Run `act(obs) -> action` greedily on every path; return per-path is_bps, risk_bps, objective
    and holdings (n_paths x N+1: inventory left before each decision, then 0 after the last).

    `env` must use reward_mode="realized". Each reset is seeded (seed + i) so evaluation is a pure
    function of (policy, paths), whatever ran before. Also checks the identity return == -objective.
    """
    assert env.reward_mode == "realized", "evaluation must use realised rewards"
    is_bps, risk_bps = np.empty(len(paths)), np.empty(len(paths))
    holdings = np.empty((len(paths), env.N + 1))
    for i, path in enumerate(paths):
        obs, _ = env.reset(seed=seed + i, options={"path": path})
        done, ret, info = False, 0.0, {}
        while not done:
            obs, r, done, _, info = env.step(act(obs))
            ret += r
        is_bps[i], risk_bps[i], holdings[i] = info["is_bps"], info["risk_bps"], info["holdings"]
        assert abs(ret + is_bps[i] + risk_bps[i]) < 1e-6, "episode return != -(IS + risk)"
    return {"is_bps": is_bps, "risk_bps": risk_bps, "objective": is_bps + risk_bps, "holdings": holdings}


def evaluate_schedule(env: ExecutionEnv, holdings: np.ndarray, paths: list) -> dict[str, np.ndarray]:
    res = [run_schedule(env, holdings, path=p) for p in paths]
    is_bps, risk_bps = np.array([r["is_bps"] for r in res]), np.array([r["risk_bps"] for r in res])
    return {"is_bps": is_bps, "risk_bps": risk_bps, "objective": is_bps + risk_bps,
            "holdings": np.stack([r["holdings"] for r in res])}


def summarize(x: np.ndarray) -> tuple[float, float]:
    """(mean, 95% CI half-width); see CI_DEF."""
    ci = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(ci)


class ValidationTracker:
    """Scores the policy on the validation set, logs a CSV row, and saves a checkpoint on improvement."""

    def __init__(self, env: ExecutionEnv, paths: list, seed: int, ac_objective: np.ndarray,
                 csv_path: str, ckpt_path: str):
        self.env, self.paths, self.seed, self.ac_obj = env, paths, seed, ac_objective
        self.ckpt_path = ckpt_path
        self.best, self.best_row = float("inf"), None
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()
        self._file = open(csv_path, "w", newline="")
        self._csv = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._csv.writeheader()

    def record(self, step, episodes, epsilon, ret_mean, ret_std, act, save) -> dict:
        res = evaluate_policy(self.env, act, self.paths, self.seed)
        obj_mean, obj_ci = summarize(res["objective"])
        gap_mean, gap_ci = summarize(res["objective"] - self.ac_obj)     # paired: same paths as AC
        improved = obj_mean < self.best
        if improved:
            self.best = obj_mean
            save(self.ckpt_path)
        row = dict(step=step, episodes=episodes, epsilon=epsilon, train_return_mean=ret_mean,
                   train_return_std=ret_std, val_is_mean=float(res["is_bps"].mean()), val_obj_mean=obj_mean,
                   val_obj_ci95=obj_ci, val_gap_ac_mean=gap_mean, val_gap_ac_ci95=gap_ci,
                   best_val_obj=self.best, is_best=int(improved), wall_s=round(time.perf_counter() - self.t0, 2))
        if improved:
            self.best_row = row
        self.rows.append(row)
        self._csv.writerow(row)
        self._file.flush()                                                # keep partial curves on a crash
        eps = "" if epsilon == "" else f" eps {epsilon:.3f}"
        print(f"step {step:>7}  ep {episodes:>6}{eps}  val obj {obj_mean:7.2f} ± {obj_ci:4.2f}  "
              f"vs AC {gap_mean:+6.2f} ± {gap_ci:4.2f}  best {self.best:7.2f}{'  *' if improved else ''}")
        return row

    def close(self) -> None:
        self._file.close()


# ----------------------------------------------------------------------------- algorithms
def train_dqn_run(spec: RunSpec, train_env: ExecutionEnv, tracker: ValidationTracker):
    """Our DQN with its default hyperparameters (incl. 20k-step epsilon decay and internal 0.01 scale)."""
    agent = DQNAgent(train_env.observation_space.shape[0], train_env.action_space.n, seed=spec.seed)

    def on_progress(m: TrainMetrics) -> None:
        tracker.record(m.step, m.episodes, m.epsilon, m.recent_return_mean, m.recent_return_std,
                       act=lambda o: m.agent.act(o, greedy=True), save=m.agent.save)

    returns = train_dqn(agent, train_env, spec.steps, seed=spec.seed, callback=on_progress,
                        callback_every=spec.eval_every)

    def load_best():
        best = DQNAgent(agent.hp["obs_dim"], agent.hp["n_actions"], **{k: v for k, v in agent.hp.items()
                                                                      if k not in ("obs_dim", "n_actions")})
        best.load(tracker.ckpt_path)
        return lambda o: best.act(o, greedy=True)

    return dict(agent.hp), len(returns), load_best


class ScaledReward(gym.Wrapper):
    """PPO training only: multiplies rewards by `scale`, and records UNSCALED episode returns for logs."""

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self.scale, self.episode_returns, self._ret = scale, [], 0.0

    def reset(self, **kwargs):
        self._ret = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self._ret += r
        if terminated or truncated:
            self.episode_returns.append(self._ret)
        return obs, r * self.scale, terminated, truncated, info


def train_ppo_run(spec: RunSpec, train_env: ExecutionEnv, tracker: ValidationTracker):
    """Stable-baselines3 PPO on the SAME discrete 12-action env as DQN. Imported here so DQN never needs SB3."""
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    wrapped = ScaledReward(train_env, PPO_REWARD_SCALE)
    policy_kwargs = dict(net_arch=PPO_NET["net_arch"], activation_fn=getattr(torch.nn, PPO_NET["activation_fn"]))
    model = PPO("MlpPolicy", wrapped, policy_kwargs=policy_kwargs, seed=spec.seed, device="cpu", verbose=0,
                **PPO_HP)
    act = lambda o: int(model.predict(o, deterministic=True)[0])

    # SB3 calls on_step DURING rollout collection, before the gradient update on that rollout, so
    # validating there would score a policy one rollout behind its step label and never score the
    # final update. The policy only changes between rollouts: validate at the start of the next
    # rollout (after train()) and once more at the end of training.
    class Validate(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            if self.num_timesteps > 0 and self.num_timesteps % spec.eval_every == 0:
                self._record()

        def _on_training_end(self) -> None:
            if not tracker.rows or tracker.rows[-1]["step"] != self.num_timesteps:
                self._record()

        def _record(self) -> None:
            recent = wrapped.episode_returns[-100:]
            mean, std = (float(np.mean(recent)), float(np.std(recent))) if recent else (float("nan"),) * 2
            tracker.record(self.num_timesteps, len(wrapped.episode_returns), "", mean, std, act=act,
                           save=model.save)

    model.learn(total_timesteps=spec.steps, callback=Validate())
    hp = dict(PPO_HP, policy="MlpPolicy", policy_kwargs=PPO_NET, reward_scale=PPO_REWARD_SCALE, seed=spec.seed,
              n_envs=model.n_envs, device=str(model.device), sb3_version=stable_baselines3.__version__)

    def load_best():
        best = PPO.load(tracker.ckpt_path, device="cpu")
        return lambda o: int(best.predict(o, deterministic=True)[0])

    return hp, len(wrapped.episode_returns), load_best


# ----------------------------------------------------------------------------- run orchestration
def _clean(x):
    """JSON-safe: NumPy scalars to Python, NaN to null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def artifact_paths(spec: RunSpec, out_dir: str) -> dict[str, str]:
    name, ext = spec.run_name, ".pt" if spec.algo == "dqn" else ".zip"
    return {"ckpt": os.path.join(out_dir, "models", name + ext),
            "meta": os.path.join(out_dir, "models", name + ".json"),
            "csv": os.path.join(out_dir, "logs", name + ".csv")}


def run(spec: RunSpec, base_cfg: dict, out_dir: str = HERE, data_dir: str | None = None,
        overwrite: bool = False) -> dict:
    """Train one agent, keep the best validated checkpoint, write metadata. Returns the metadata dict."""
    torch.set_num_threads(1)                         # parallel seed runs are separate processes
    cfg = resolve_config(base_cfg, spec, data_dir)
    paths = artifact_paths(spec, out_dir)
    if not overwrite and any(os.path.exists(p) for p in paths.values()):
        raise FileExistsError(f"artifacts for {spec.run_name} already exist in {out_dir}; pass --overwrite")

    set_seed(spec.seed)                              # Python, NumPy, PyTorch
    train_mode = "expected" if spec.mode == "synthetic" else "realized"   # D9: expected only on synthetic
    train_env = make_env(cfg, spec, make_sampler(cfg, spec, "train"), train_mode)
    train_env.action_space.seed(spec.seed)
    train_env.observation_space.seed(spec.seed)

    # Validation env is a separate instance; it is always given an explicit path, so its sampler is never used.
    val = validation_paths(cfg, spec, "val")
    val_env = make_env(cfg, spec, SyntheticPathSampler(cfg), "realized")
    val_seed = cfg["evaluation"]["val_seed"]
    baselines = {k: evaluate_schedule(val_env, h, val) for k, h in all_baselines(cfg, spec.lam).items()}
    for p in paths.values():                         # only once all inputs loaded, so failures leave no files
        os.makedirs(os.path.dirname(p), exist_ok=True)

    tracker = ValidationTracker(val_env, val, val_seed, baselines["Almgren-Chriss"]["objective"],
                                paths["csv"], paths["ckpt"])
    if spec.mode == "real" and spec.n_val != len(val):
        print(f"note: real mode validates on all {len(val)} days of the 'val' date split (--n-val ignored)")
    print(f"run {spec.run_name}: {spec.steps} steps, train reward '{train_mode}', {len(val)} validation paths")
    t0 = time.perf_counter()
    try:
        trainer = train_dqn_run if spec.algo == "dqn" else train_ppo_run
        hp, episodes, load_best = trainer(spec, train_env, tracker)
    finally:
        tracker.close()
    wall = time.perf_counter() - t0

    # The saved checkpoint must reproduce its logged validation score on exactly the same paths.
    reloaded = summarize(evaluate_policy(val_env, load_best(), val, val_seed)["objective"])[0]
    reload_diff = abs(reloaded - tracker.best)
    assert reload_diff < 1e-6, f"reloaded checkpoint scores {reloaded} but logged best was {tracker.best}"

    meta = {
        "run_name": spec.run_name, **asdict(spec), "n_val_actual": len(val),
        "train_reward_mode": train_mode, "eval_reward_mode": "realized",
        "val_seed": val_seed if spec.mode == "synthetic" else None,
        "val_source": "synthetic val_seed" if spec.mode == "synthetic" else "real date split 'val'",
        "objective": OBJECTIVE_DEF, "ci95": CI_DEF,
        "checkpoint": os.path.basename(paths["ckpt"]), "learning_curve": os.path.basename(paths["csv"]),
        "best": tracker.best_row, "final": tracker.rows[-1], "episodes": episodes,
        "reload_check_abs_diff_bps": reload_diff,
        "baselines_val": {k: dict(zip(("obj_mean", "obj_ci95"), summarize(v["objective"])),
                                  is_mean=float(v["is_bps"].mean())) for k, v in baselines.items()},
        "hyperparameters": hp, "resolved_config": portable_config(cfg), "command": sys.argv,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "gymnasium": gym.__version__},
        "timing": {"train_wall_s": round(wall, 2)},          # the only non-deterministic field
    }
    with open(paths["meta"], "w") as f:
        json.dump(_clean(meta), f, indent=2)
    best = tracker.best_row
    print(f"best val objective {best['val_obj_mean']:.2f} ± {best['val_obj_ci95']:.2f} bps at step {best['step']} "
          f"(AC {meta['baselines_val']['Almgren-Chriss']['obj_mean']:.2f}); {wall:.1f} s; saved {paths['ckpt']}")
    return meta


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> tuple[RunSpec, argparse.Namespace]:
    onoff = {"on": True, "off": False}
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--signal", choices=onoff, default="off", help="generate the AR(1) drift in prices")
    p.add_argument("--observe-signal", choices=onoff, default="on", help="off = ablation: drift hidden from agent")
    p.add_argument("--lam", type=float, default=None, help="risk aversion (default: config.yaml)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--n-val", type=int, default=None, help="synthetic validation paths (default: config.yaml)")
    p.add_argument("--out-dir", default=HERE, help="root for models/ and logs/ (default: repo)")
    p.add_argument("--data-dir", default=None, help="real-data CSV directory (default: config.yaml data.csv_dir)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing run's artifacts")
    p.add_argument("--self-check", action="store_true", help="run this file's bounded self-check and exit")
    a = p.parse_args(argv)
    if a.self_check:
        return None, a

    cfg = load_config()
    ev = cfg["evaluation"]
    lam = cfg["market"]["risk_aversion"] if a.lam is None else a.lam
    n_val = ev["n_val_paths"] if a.n_val is None else a.n_val
    errors = []
    if a.mode == "real" and onoff[a.signal]:
        errors.append("--mode real --signal on: real NSE paths carry no observable drift signal "
                      "(the loader sets signal = 0); use --signal off with real data")
    if a.steps <= 0 or a.eval_every <= 0 or n_val <= 1:
        errors.append("--steps and --eval-every must be positive and --n-val at least 2")
    if not math.isfinite(lam) or lam < 0:
        errors.append("--lam must be a finite number >= 0")
    if a.seed < 0 or a.seed in (ev["val_seed"], ev["test_seed"]):
        errors.append(f"--seed must be >= 0 and differ from the validation/test seeds {ev['val_seed']}, "
                      f"{ev['test_seed']}")
    if a.algo == "ppo" and (a.steps % PPO_HP["n_steps"] or a.eval_every % PPO_HP["n_steps"]):
        errors.append(f"--algo ppo: --steps and --eval-every must be multiples of PPO n_steps = "
                      f"{PPO_HP['n_steps']} (SB3 collects whole rollouts; the policy only changes between them)")
    if errors:
        p.error("\n  ".join(errors))
    spec = RunSpec(a.algo, a.mode, onoff[a.signal], onoff[a.observe_signal], float(lam), a.seed, a.steps,
                   a.eval_every, n_val)
    return spec, a


def self_check() -> None:
    """Ablation switch, artifacts, checkpoint reload and determinism, on a tiny bounded DQN run."""
    cfg = load_config()

    # 1. Ablation: with the signal generated but not observed, the drift feature is identically 0.
    spec = RunSpec("dqn", "synthetic", True, False, 25.0, 0, 1, 1, 2)
    rcfg = resolve_config(cfg, spec)
    rng = np.random.default_rng(0)
    for observe in (False, True):
        env = make_env(rcfg, RunSpec(**{**asdict(spec), "observe_signal": observe}), SyntheticPathSampler(rcfg),
                       "realized")
        drift_seen, drift_true = [], []
        for _ in range(50):
            path = env.sampler.sample(rng)
            obs, _ = env.reset(seed=0, options={"path": path})
            for k in range(env.N):
                drift_seen.append(obs[2])
                drift_true.append(path.signal[k])
                obs, *_ = env.step(4)
        assert np.abs(drift_true).max() > 0, "signal should be generated"
        assert (np.abs(drift_seen).max() == 0) == (not observe)
    assert cfg["signal"]["enabled"] is False, "resolve_config must not mutate the loaded config"
    print("ablation: signal generated in prices; drift feature identically 0 when not observed, non-zero when observed")

    # 2. Tiny run twice: artifacts, CSV columns, metadata, reload consistency (asserted in run), determinism.
    spec = RunSpec("dqn", "synthetic", False, True, 25.0, 0, 2_000, 500, 100)
    metas, curves = [], []
    with tempfile.TemporaryDirectory() as d:
        for rep in range(2):
            out = os.path.join(d, f"rep{rep}")
            metas.append(run(spec, cfg, out_dir=out))
            p = artifact_paths(spec, out)
            assert all(os.path.exists(f) for f in p.values())
            with open(p["csv"]) as f:
                rows = list(csv.DictReader(f))
            assert list(rows[0]) == CSV_COLUMNS and [int(r["step"]) for r in rows] == [500, 1000, 1500, 2000]
            curves.append([{k: v for k, v in r.items() if k != "wall_s"} for r in rows])
    strip = lambda m: {k: strip(v) for k, v in m.items() if k not in ("timing", "wall_s")} \
        if isinstance(m, dict) else m                                     # drop wall-clock fields only
    metas = [_clean(m) for m in metas]
    assert strip(metas[0]) == strip(metas[1]) and curves[0] == curves[1], "seeded runs differ"
    assert metas[0]["resolved_config"]["data"]["csv_dir"] == "data/raw", "metadata must not record absolute paths"

    # 3. Evaluation helpers return per-path holdings consistent with the env (start at 1, end at 0, never rise).
    rcfg = resolve_config(cfg, spec)
    env, paths = make_env(rcfg, spec, SyntheticPathSampler(rcfg), "realized"), validation_paths(rcfg, spec)[:20]
    for res in (evaluate_policy(env, lambda o: 4, paths, 0), evaluate_schedule(env, all_baselines(rcfg)["TWAP"], paths)):
        h = res["holdings"]
        assert h.shape == (20, env.N + 1) and (h[:, 0] == 1).all() and (h[:, -1] == 0).all()
        assert (np.diff(h, axis=1) <= 0).all()
    print("metadata stores csv_dir relative to the repo; evaluation helpers return valid per-path holdings")
    print("tiny run: artifacts and CSV columns OK, checkpoint reload reproduces best score, repeat run identical")
    print("\ntrain.py self-check passed.")


if __name__ == "__main__":
    spec, args = parse_args()
    if args.self_check:
        self_check()
        sys.exit(0)
    try:
        run(spec, load_config(), out_dir=args.out_dir, data_dir=args.data_dir, overwrite=args.overwrite)
    except (DataError, FileExistsError) as e:
        sys.exit(f"error: {e}")
