"""Deep Q-Network written from scratch in PyTorch (Double DQN variant).

Components:
* QNetwork      — MLP: obs (4) -> 128 -> 128 -> one Q-value per discrete action (12), ReLU.
* ReplayBuffer  — fixed-capacity circular buffer in NumPy arrays, uniform mini-batch sampling.
* DQNAgent      — epsilon-greedy acting, Double-DQN TD targets, Huber loss, Adam, gradient
                  clipping, soft (Polyak) target-network updates, save/load.
* train_dqn     — plain interaction loop: act, store, update once per environment step; optional
                  periodic callback receiving a TrainMetrics snapshot.

Design choices (and why):
* gamma = 1: the episode is a finite 25-step horizon and the episode return is exactly the
  Almgren–Chriss objective (DECISIONS.md D8, D10). Discounting would change that objective.
* Terminal transitions never bootstrap: their TD target is the immediate reward alone.
* Double DQN: the online net picks the next action, the target net scores it. This reduces the
  max-operator's upward bias, which matters here because Q-value gaps between actions are small.
* reward_scale = 0.01 inside the agent: env rewards are in bps (one step of Immediate liquidation
  is about -500), so scaled Q-values are O(1), which suits Huber loss and Adam's default step size.
  With gamma = 1, multiplying every reward by a positive constant leaves the optimal policy unchanged.
* Epsilon decays linearly from eps_start to eps_end over the first `eps_decay_steps` agent steps,
  then stays at eps_end.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    """Circular buffer: once full, the oldest transition is overwritten."""

    def __init__(self, obs_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, *, hidden: int = 128, lr: float = 5e-4,
                 gamma: float = 1.0, batch_size: int = 64, buffer_size: int = 100_000,
                 learning_starts: int = 1_000, tau: float = 0.005, grad_clip: float = 10.0,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 20_000,
                 reward_scale: float = 0.01, seed: int = 0):
        obs_dim, n_actions = int(obs_dim), int(n_actions)   # gym returns NumPy ints; keep checkpoints plain
        self.hp = dict(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden, lr=lr, gamma=gamma,
                       batch_size=batch_size, buffer_size=buffer_size, learning_starts=learning_starts,
                       tau=tau, grad_clip=grad_clip, eps_start=eps_start, eps_end=eps_end,
                       eps_decay_steps=eps_decay_steps, reward_scale=reward_scale, seed=seed)
        self.n_actions = n_actions
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(obs_dim, buffer_size)
        self.steps = 0                      # environment steps seen (drives the epsilon schedule)

    # ------------------------------------------------------------------ acting
    @property
    def epsilon(self) -> float:
        hp = self.hp
        frac = min(1.0, self.steps / max(1, hp["eps_decay_steps"]))
        return hp["eps_start"] + frac * (hp["eps_end"] - hp["eps_start"])

    def act(self, obs, greedy: bool = True) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------ learning
    def td_targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Double-DQN target y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))."""
        with torch.no_grad():
            next_a = self.q(batch["next_obs"]).argmax(dim=1, keepdim=True)          # (B, 1)
            next_q = self.q_target(batch["next_obs"]).gather(1, next_a).squeeze(1)  # (B,)
            r = batch["rewards"] * self.hp["reward_scale"]
            return r + self.hp["gamma"] * (1.0 - batch["dones"]) * next_q

    def update(self) -> float | None:
        """One gradient step on a replay mini-batch, then a soft target update. Returns the loss."""
        if len(self.buffer) < max(self.hp["learning_starts"], self.hp["batch_size"]):
            return None
        batch = self.buffer.sample(self.hp["batch_size"], self.rng)
        target = self.td_targets(batch)
        q_taken = self.q(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_taken, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.hp["grad_clip"])
        self.opt.step()
        self.soft_update()
        return float(loss.item())

    def soft_update(self) -> None:
        tau = self.hp["tau"]
        with torch.no_grad():
            for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_t.mul_(1.0 - tau).add_(p, alpha=tau)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        torch.save({"hp": self.hp, "steps": self.steps, "q": self.q.state_dict(),
                    "q_target": self.q_target.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("obs_dim", "n_actions", "hidden"):
            if ckpt["hp"][key] != self.hp[key]:
                raise ValueError(f"checkpoint {key}={ckpt['hp'][key]} but agent has {self.hp[key]}")
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.opt.load_state_dict(ckpt["opt"])
        self.steps = ckpt["steps"]


@dataclass
class TrainMetrics:
    """Snapshot of training progress handed to the `train_dqn` callback.

    Episode returns are in the env's reward units (bps), NOT the agent's internally scaled units.
    The recent-return statistics are NaN until the first episode finishes.
    """
    step: int                   # environment steps completed in this call to train_dqn (1-based)
    agent: DQNAgent
    episodes: int               # episodes completed so far
    epsilon: float              # exploration rate that will be used for the next action
    recent_return_mean: float   # mean return of the last `recent_window` completed episodes
    recent_return_std: float    # their (population) standard deviation
    recent_window: int          # number of episodes the two statistics above were computed over


def train_dqn(agent: DQNAgent, env, total_steps: int, seed: int = 0,
              callback: Callable[[TrainMetrics], None] | None = None,
              callback_every: int = 10_000, recent_window: int = 100) -> list[float]:
    """Interact for `total_steps` environment steps, updating once per step. Returns episode returns.

    If `callback` is given it is called with a TrainMetrics after every `callback_every` steps and
    after the final step (used by train.py for periodic validation and checkpointing). The callback
    must not touch `env` or the agent's exploration RNG, so training is identical with or without it.
    """
    obs, _ = env.reset(seed=seed)
    returns, ep_return = [], 0.0
    for t in range(1, total_steps + 1):
        action = agent.act(obs, greedy=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.buffer.add(obs, action, reward, next_obs, terminated)   # truncation would still bootstrap
        agent.steps += 1
        agent.update()
        ep_return += reward
        obs = next_obs
        if terminated or truncated:
            returns.append(ep_return)
            obs, ep_return = env.reset()[0], 0.0
        if callback is not None and (t % callback_every == 0 or t == total_steps):
            callback(_metrics(t, agent, returns, recent_window))
    return returns


def _metrics(step: int, agent: DQNAgent, returns: list[float], window: int) -> TrainMetrics:
    recent = returns[-window:]
    mean, std = (float(np.mean(recent)), float(np.std(recent))) if recent else (float("nan"),) * 2
    return TrainMetrics(step=step, agent=agent, episodes=len(returns), epsilon=agent.epsilon,
                        recent_return_mean=mean, recent_return_std=std, recent_window=len(recent))


# ----------------------------------------------------------------------------- self-check
if __name__ == "__main__":
    import os
    import tempfile
    import time

    from baselines import ac_holdings, immediate_holdings, run_schedule, twap_holdings
    from data import load_config
    from env import ExecutionEnv

    torch.set_num_threads(1)
    SEED, TRAIN_STEPS, N_EVAL = 0, 50_000, 2_000
    set_seed(SEED)
    cfg = load_config()

    # 1. Dimensions: the network must match the environment, not the other way round.
    env = ExecutionEnv(cfg, reward_mode="expected")          # D9: variance-reduced reward, synthetic only
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    assert (obs_dim, n_actions) == (4, 12), f"spec says 4 obs / 12 actions, env has {obs_dim} / {n_actions}"
    agent = DQNAgent(obs_dim, n_actions, seed=SEED)
    assert agent.q(torch.zeros(7, obs_dim)).shape == (7, n_actions)
    print(f"dimensions: obs {obs_dim}, actions {n_actions}, Q-net output (7, {n_actions}) OK")

    # 2. Replay buffer: shapes, dtypes, and circular overwrite.
    small = ReplayBuffer(obs_dim, capacity=5)
    for i in range(8):
        small.add(np.full(obs_dim, i), i % n_actions, float(i), np.full(obs_dim, i + 1), i == 7)
    assert len(small) == 5 and sorted(small.rewards.tolist()) == [3, 4, 5, 6, 7]
    b = small.sample(32, np.random.default_rng(0))
    assert b["obs"].shape == (32, obs_dim) and b["actions"].dtype == torch.int64
    assert b["rewards"].shape == b["dones"].shape == (32,)
    assert torch.all(b["next_obs"][:, 0] == b["obs"][:, 0] + 1)          # rows stay aligned
    print("replay buffer: shapes, dtypes, alignment, circular overwrite OK")

    # 3. Action selection: greedy is the argmax and deterministic; epsilon = 1 explores.
    o = env.reset(seed=SEED)[0]
    expected = int(agent.q(torch.as_tensor(o).unsqueeze(0)).argmax())
    assert all(agent.act(o, greedy=True) == expected for _ in range(20))
    explored = {agent.act(o, greedy=False) for _ in range(500)}           # steps = 0 -> epsilon = 1
    assert agent.epsilon == 1.0 and len(explored) == n_actions
    print(f"action selection: greedy deterministic, epsilon-greedy covered {len(explored)}/{n_actions} actions")

    # 4. TD targets: terminal rows equal the scaled reward; others match a hand-computed Double-DQN target.
    batch = {"obs": torch.rand(6, obs_dim), "actions": torch.randint(0, n_actions, (6,)),
             "rewards": torch.tensor([-10., -20., -30., -40., -50., -60.]),
             "next_obs": torch.rand(6, obs_dim), "dones": torch.tensor([1., 0., 1., 0., 0., 1.])}
    y = agent.td_targets(batch)
    scale = agent.hp["reward_scale"]
    term = batch["dones"] == 1
    assert torch.equal(y[term], batch["rewards"][term] * scale), "terminal targets must not bootstrap"
    with torch.no_grad():
        a_star = agent.q(batch["next_obs"]).argmax(1)
        manual = batch["rewards"] * scale + agent.q_target(batch["next_obs"])[torch.arange(6), a_star]
    assert torch.allclose(y[~term], manual[~term]), "non-terminal targets must be Double-DQN"
    print("TD targets: terminal = reward only, non-terminal = r + Q_target(s', argmax Q_online) OK")

    # 5. One training update changes the online net and moves the target net by exactly tau.
    env.observation_space.seed(SEED)                     # gym spaces have their own RNG; seed it for reproducibility
    env.action_space.seed(SEED)
    for _ in range(agent.hp["learning_starts"]):
        agent.buffer.add(env.observation_space.sample(), env.action_space.sample(), -1.0,
                         env.observation_space.sample(), False)
    before = [p.clone() for p in agent.q.parameters()]
    target_before = [p.clone() for p in agent.q_target.parameters()]
    loss = agent.update()
    assert loss is not None and np.isfinite(loss)
    assert any(not torch.equal(a, p) for a, p in zip(before, agent.q.parameters()))
    tau = agent.hp["tau"]
    for t0, p, t1 in zip(target_before, agent.q.parameters(), agent.q_target.parameters()):
        assert torch.allclose(t1, (1 - tau) * t0 + tau * p.detach())
    print(f"training update: loss {loss:.4f} finite, online net changed, soft target update OK")

    # 5b. Callback: fires on schedule with correct metrics, and leaves training bit-for-bit unchanged.
    short = 2_510                                          # not a multiple of the interval -> final call too
    set_seed(SEED)
    plain = DQNAgent(obs_dim, n_actions, seed=SEED)
    plain_returns = train_dqn(plain, env, short, seed=SEED)
    set_seed(SEED)
    hooked = DQNAgent(obs_dim, n_actions, seed=SEED)
    seen: list[TrainMetrics] = []
    hooked_returns = train_dqn(hooked, env, short, seed=SEED, callback=seen.append, callback_every=1_000,
                               recent_window=50)
    assert [m.step for m in seen] == [1_000, 2_000, short], [m.step for m in seen]
    n_ep = env.N                                           # every episode lasts exactly N steps
    assert [m.episodes for m in seen] == [s // n_ep for s in (1_000, 2_000, short)]
    assert all(m.agent is hooked for m in seen)
    eps = hooked.hp["eps_start"] + np.array([1_000, 2_000, short]) / hooked.hp["eps_decay_steps"] \
        * (hooked.hp["eps_end"] - hooked.hp["eps_start"])
    assert np.allclose([m.epsilon for m in seen], eps)
    last = seen[-1]
    assert last.recent_window == 50 and np.isclose(last.recent_return_mean, np.mean(hooked_returns[-50:]))
    assert np.isclose(last.recent_return_std, np.std(hooked_returns[-50:]))
    assert plain_returns == hooked_returns, "callback changed the training trajectory"
    assert all(torch.equal(a, b) for a, b in zip(plain.q.parameters(), hooked.q.parameters()))
    print(f"callback: fired at steps {[m.step for m in seen]}, metrics correct, training unchanged by callback")

    # 6. Real training run with a fixed seed and a bounded CPU budget.
    set_seed(SEED)
    agent = DQNAgent(obs_dim, n_actions, seed=SEED)
    t0 = time.perf_counter()
    ep_returns = train_dqn(agent, env, TRAIN_STEPS, seed=SEED)
    train_secs = time.perf_counter() - t0
    print(f"\ntraining: {TRAIN_STEPS} env steps ({len(ep_returns)} episodes), seed {SEED}, "
          f"{train_secs:.1f} s on CPU (1 thread), final epsilon {agent.epsilon:.2f}")
    print(f"  mean training return (expected reward), first 100 eps {np.mean(ep_returns[:100]):.1f}, "
          f"last 100 eps {np.mean(ep_returns[-100:]):.1f}")

    # 7. Save / load: a fresh agent restored from disk must produce identical Q-values.
    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "dqn_selfcheck.pt")
        agent.save(ckpt_path)
        restored = DQNAgent(obs_dim, n_actions, seed=123)
        restored.load(ckpt_path)
    probe = torch.rand(256, obs_dim)
    with torch.no_grad():
        assert torch.equal(agent.q(probe), restored.q(probe))
        assert torch.equal(agent.q_target(probe), restored.q_target(probe))
    assert restored.steps == agent.steps
    print("save/load: restored online and target Q-values identical, step count restored")

    # 8. Paired evaluation on held-out synthetic paths, scored on REALISED IS from cash flows.
    eval_env = ExecutionEnv(cfg, reward_mode="realized")
    eval_rng = np.random.default_rng(10_000 + SEED)                       # separate from the training stream
    paths = [eval_env.sampler.sample(eval_rng) for _ in range(N_EVAL)]

    def run_dqn(path) -> dict:
        obs, _ = eval_env.reset(options={"path": path})
        done, info = False, {}
        while not done:
            obs, _, done, _, info = eval_env.step(restored.act(obs, greedy=True))
        return info

    N = cfg["market"]["n_steps"]
    schedules = {"Immediate": immediate_holdings(N), "TWAP": twap_holdings(N),
                 "Almgren-Chriss": ac_holdings(cfg)}
    results = {name: [run_schedule(eval_env, h, path=p) for p in paths] for name, h in schedules.items()}
    results["DQN"] = [run_dqn(p) for p in paths]

    print(f"\nevaluation: {N_EVAL} paired synthetic paths (same paths for every strategy), realised IS")
    print(f"{'strategy':<16}{'mean IS':>9}{'std IS':>9}{'mean obj':>10}{'std obj':>9}{'± 95% obj':>11}")
    objective = {}
    for name in ["Immediate", "TWAP", "Almgren-Chriss", "DQN"]:
        is_bps = np.array([r["is_bps"] for r in results[name]])
        obj = is_bps + np.array([r["risk_bps"] for r in results[name]])
        objective[name] = obj
        print(f"{name:<16}{is_bps.mean():>9.2f}{is_bps.std():>9.2f}{obj.mean():>10.2f}{obj.std():>9.2f}"
              f"{1.96 * obj.std() / np.sqrt(N_EVAL):>11.2f}")

    diff = objective["DQN"] - objective["Immediate"]
    ci = 1.96 * diff.std() / np.sqrt(N_EVAL)
    print(f"\nDQN - Immediate objective (paired): {diff.mean():.2f} ± {ci:.2f} bps")
    holdings = results["DQN"][0]["holdings"]
    print("DQN holdings on eval path 0:", np.array2string(holdings[::4], precision=2))
    print("AC  holdings (static)      :", np.array2string(schedules["Almgren-Chriss"][::4], precision=2))
    assert diff.mean() + ci < 0, "DQN did not beat Immediate liquidation on the objective"
    print("\nDQN smoke test passed: DQN beats Immediate liquidation on mean objective (paired 95% CI < 0).")
