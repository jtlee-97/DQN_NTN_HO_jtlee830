#!/usr/bin/env python3
"""Train a small presentation-oriented DQN handover policy."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leohosim_py import config, simulator
from leohosim_py.agents import DQNAgent
from leohosim_py.envs.simple_dqn_ho_env import BASE_STEP_REWARD, SIMPLE_DQN_HO_ACTIONS, SimpleDQNHandoverEnv
from leohosim_py.training.dqn_trainer import compute_episode_kpi_reward, reward_config_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple two-action DQN handover policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--rl-reward-scale", type=float, default=5.0)
    parser.add_argument("--reward-transform", choices=["scale", "tanh", "clip"], default="scale")
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--scenario-mode", choices=["fixed", "curriculum", "random"], default="curriculum")
    parser.add_argument("--train-scenario-count", type=int, default=64)
    parser.add_argument("--val-scenario-count", type=int, default=None)
    parser.add_argument("--train-seed-offset", type=int, default=10_000)
    parser.add_argument("--val-seed-offset", type=int, default=100_000)
    parser.add_argument("--test-seed-offset", type=int, default=200_000)
    parser.add_argument("--scenario-random-mix-start", type=int, default=10_000)
    parser.add_argument("--scenario-random-mix-decay", type=int, default=4000)
    parser.add_argument("--initial-eval", action="store_true")
    parser.add_argument("--disable-paired-eval", action="store_true")
    parser.add_argument("--ddqn", action="store_true", help="Optional; default is plain DQN for presentation.")
    args = parser.parse_args()

    cfg = config.LeohosimConfig.from_yaml(Path(args.config))
    cfg.baseline_policy = "a3_rsrp"
    cfg.validate()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out / "reward_summary.json").write_text(json.dumps(reward_config_summary(cfg), indent=2), encoding="utf-8")
    write_action_grid(out)

    env = SimpleDQNHandoverEnv(cfg, np.random.default_rng(cfg.seed))
    dqn_cfg = cfg.ddqn if args.ddqn else cfg.dqn
    agent = DQNAgent(env.state_dim, env.action_dim, dqn_cfg, seed=cfg.seed, double_dqn=args.ddqn)
    scenario_rng = np.random.default_rng(cfg.seed + 777)
    train_baselines = build_training_scenario_baselines(cfg, args, out)
    val_episodes = max(1, int(args.val_scenario_count if args.val_scenario_count is not None else args.eval_episodes))
    paired_eval_enabled = not args.disable_paired_eval
    rows: list[dict] = []
    best_score = -float("inf")

    if args.initial_eval and paired_eval_enabled:
        summary = paired_summary(evaluate_paired(cfg, agent, val_episodes, args.val_seed_offset))
        row = {"method": "eval_checkpoint", "episode": 0}
        add_eval_summary(row, summary, best_score)
        best_score = row["eval_safety_score"]
        row["eval_best_safety_score_so_far"] = best_score
        agent.save(str(out / "best_dqn_model.pt"))
        rows.append(row)

    started = time.time()
    for ep in range(args.episodes):
        scenario = training_scenario_seed(cfg.seed, ep, args, scenario_rng)
        rng = np.random.default_rng(cfg.seed if scenario["seed"] is None else int(scenario["seed"]))
        env = SimpleDQNHandoverEnv(cfg, rng)
        state = env.reset()
        epsilon = agent.epsilon(ep)
        dense_total = 0.0
        train_total = 0.0
        counts = np.zeros(env.action_dim, dtype=int)
        last_loss = None
        while not env.sim.is_done():
            valid = env.valid_action_indices()
            action = agent.select_action(state, epsilon, valid_actions=valid)
            counts[action] += 1
            result = env.step(action)
            shaped = transform_reward(result.reward, args.rl_reward_scale, args.reward_transform, args.reward_clip)
            next_mask = valid_mask(env.action_dim, env.valid_action_indices()) if not result.done else np.ones(env.action_dim, dtype=np.bool_)
            agent.remember((state, action, shaped, result.state, result.done, next_mask))
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss
            dense_total += result.reward
            train_total += shaped
            state = result.state
            if result.done:
                break

        k = env.get_kpi()
        row = kpi_row(cfg, "Simple-DQN-HO", ep, k, dense_total)
        row.update(
            {
                "epsilon": float(epsilon),
                "rl_training_reward": float(dense_total),
                "kpi_score": float(row["reward"]),
                "rl_base_step_reward": float(BASE_STEP_REWARD),
                "mean_train_reward": train_total / max(float(np.sum(counts)), 1.0),
                "loss": -1.0 if last_loss is None else float(last_loss),
                "scenario_seed": -1 if scenario["seed"] is None else int(scenario["seed"]),
                "scenario_idx": int(scenario["idx"]),
                "scenario_random_mix": int(scenario["random_mix"]),
            }
        )
        add_training_reference_metrics(row, train_baselines)
        for idx, count in enumerate(counts):
            row[f"action_count_{idx}"] = int(count)

        if paired_eval_enabled and args.eval_interval > 0 and ((ep + 1) % args.eval_interval == 0 or ep + 1 == args.episodes):
            summary = paired_summary(evaluate_paired(cfg, agent, val_episodes, args.val_seed_offset))
            add_eval_summary(row, summary, best_score)
            if row["eval_safety_score"] > best_score:
                best_score = row["eval_safety_score"]
                row["eval_best_safety_score_so_far"] = best_score
                agent.save(str(out / "best_dqn_model.pt"))
        elif not paired_eval_enabled:
            score = training_checkpoint_metric(row)
            row["train_checkpoint_metric"] = score
            row["train_best_checkpoint_metric_so_far"] = max(best_score, score)
            if score > best_score:
                best_score = score
                agent.save(str(out / "best_dqn_model.pt"))

        rows.append(row)

        if ep == 0 or (ep + 1) % args.log_interval == 0 or ep + 1 == args.episodes:
            elapsed = time.time() - started
            progress = (ep + 1) / max(args.episodes, 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            print(
                f"[simple-dqn-ho] {ep+1:5d}/{args.episodes} eps={epsilon:.3f} "
                f"denseR={dense_total:7.2f} KPI={row['reward']:7.2f} "
                f"SINR={k.avg_sinr_db:6.2f} outage={k.outage_fraction:.3f} "
                f"RLF={k.rlf_count} UHO={k.uho_count} HO={k.ho_count} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                flush=True,
            )
            write_training_artifacts(out, rows)

    write_training_artifacts(out, rows)
    agent.save(str(out / "dqn_model.pt"))
    best_path = out / "best_dqn_model.pt"
    if best_path.exists():
        agent.load(str(best_path))
    eval_rows = evaluate_paired(cfg, agent, args.eval_episodes, args.test_seed_offset)
    write_csv(out / "simple_dqn_ho_eval.csv", eval_rows)
    write_eval_summary(out / "simple_dqn_ho_summary.md", eval_rows)
    write_eval_plots(out, eval_rows)
    print(f"Saved simple DQN handover outputs to {out}")


def transform_reward(raw: float, scale: float, mode: str, clip: float) -> float:
    x = float(raw) / max(float(scale), 1e-9)
    if mode == "tanh":
        return float(np.tanh(x))
    if mode == "clip":
        return float(np.clip(x, -abs(clip), abs(clip)))
    return x


def valid_mask(action_dim: int, valid_actions: np.ndarray) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=np.bool_)
    mask[np.asarray(valid_actions, dtype=np.int64)] = True
    if not mask.any():
        mask[0] = True
    return mask


def training_scenario_seed(cfg_seed: int, ep: int, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    count = max(1, int(args.train_scenario_count))
    fixed_idx = int(ep % count)
    fixed_seed = int(cfg_seed + args.train_seed_offset + fixed_idx)
    if args.scenario_mode == "random":
        return {"seed": None, "idx": -1, "random_mix": 1}
    if args.scenario_mode == "fixed":
        return {"seed": fixed_seed, "idx": fixed_idx, "random_mix": 0}
    mix_start = int(args.scenario_random_mix_start)
    mix_decay = max(0, int(args.scenario_random_mix_decay))
    if ep < mix_start:
        random_prob = 0.0
    elif mix_decay <= 0:
        random_prob = 1.0
    else:
        random_prob = float(np.clip((ep - mix_start) / mix_decay, 0.0, 1.0))
    if random_prob > 0.0 and float(rng.random()) < random_prob:
        seed = int(cfg_seed + args.train_seed_offset + 1_000_000 + ep * 37 + int(rng.integers(0, 10_000)))
        return {"seed": seed, "idx": -1, "random_mix": 1}
    return {"seed": fixed_seed, "idx": fixed_idx, "random_mix": 0}


def build_training_scenario_baselines(
    cfg: config.LeohosimConfig,
    args: argparse.Namespace,
    out: Path,
) -> dict[int, dict[str, dict]]:
    if args.scenario_mode == "random":
        return {}
    rows = []
    baselines: dict[int, dict[str, dict]] = {}
    for idx in range(max(1, int(args.train_scenario_count))):
        seed = int(cfg.seed + args.train_seed_offset + idx)
        unsafe = run_a3(cfg, seed, idx, "Unsafe-A3-0/0/0", 0.0, 0.0, 0.0)
        standard = run_a3(cfg, seed, idx, "Standard-A3", 1.0, 0.5, 0.4)
        baselines[idx] = {"unsafe": unsafe, "standard": standard}
        for name, row in [("unsafe", unsafe), ("standard", standard)]:
            out_row = {"scenario_idx": idx, "scenario_seed": seed, "baseline": name}
            for key in [
                "reward",
                "avg_sinr_db",
                "outage_time_s",
                "rlf_count",
                "uho_count",
                "uho_per_ho",
                "ho_count",
                "rb_per_s_ue",
                "avg_tos_s",
                "short_tos_count",
            ]:
                out_row[key] = row.get(key, 0.0)
            rows.append(out_row)
    write_csv(out / "train_scenario_baselines.csv", rows)
    return baselines


def add_training_reference_metrics(row: dict, baselines: dict[int, dict[str, dict]]) -> None:
    idx = int(row.get("scenario_idx", -1))
    if int(row.get("scenario_random_mix", 0)) != 0 or idx not in baselines:
        return
    for name, baseline in baselines[idx].items():
        prefix = f"train_{name}"
        row[f"{prefix}_baseline_reward"] = float(baseline["reward"])
        row[f"{prefix}_reward_gain"] = float(row["reward"]) - float(baseline["reward"])
        row[f"{prefix}_sinr_gain_db"] = float(row["avg_sinr_db"]) - float(baseline["avg_sinr_db"])
        row[f"{prefix}_outage_time_saved_s"] = float(baseline["outage_time_s"]) - float(row["outage_time_s"])
        row[f"{prefix}_rlf_delta"] = float(row["rlf_count"]) - float(baseline["rlf_count"])
        row[f"{prefix}_uho_delta"] = float(row["uho_count"]) - float(baseline["uho_count"])
        row[f"{prefix}_extra_ho"] = float(row["ho_count"]) - float(baseline["ho_count"])
        row[f"{prefix}_extra_rb_per_s_ue"] = float(row["rb_per_s_ue"]) - float(baseline["rb_per_s_ue"])
        row[f"{prefix}_safety_score"] = safety_score(row, prefix)


def training_checkpoint_metric(row: dict) -> float:
    if "train_unsafe_safety_score" in row:
        return float(row["train_unsafe_safety_score"])
    return float(row["reward"])


def safety_score(row: dict, prefix: str) -> float:
    rlf_delta = float(row.get(f"{prefix}_rlf_delta", 0.0))
    uho_delta = float(row.get(f"{prefix}_uho_delta", 0.0))
    return float(
        float(row.get(f"{prefix}_reward_gain", 0.0))
        + 55.0 * float(row.get(f"{prefix}_outage_time_saved_s", 0.0))
        + 95.0 * max(-rlf_delta, 0.0)
        - 170.0 * max(rlf_delta, 0.0)
        + 3.0 * max(-uho_delta, 0.0)
        - 6.0 * max(uho_delta, 0.0)
        + 2.0 * float(row.get(f"{prefix}_sinr_gain_db", 0.0))
        - 0.5 * max(float(row.get(f"{prefix}_extra_ho", 0.0)), 0.0)
        - 1.5 * max(float(row.get(f"{prefix}_extra_rb_per_s_ue", 0.0)), 0.0)
    )


def run_a3(cfg: config.LeohosimConfig, seed: int, episode: int, method: str, offset_db: float, hysteresis_db: float, ttt_s: float) -> dict:
    local_cfg = copy.deepcopy(cfg)
    local_cfg.baseline_policy = "a3_rsrp"
    local_cfg.system.a3_offset_db = float(offset_db)
    local_cfg.system.a3_hysteresis_db = float(hysteresis_db)
    local_cfg.system.a3_ttt_s = float(ttt_s)
    sim = simulator.LEOSimulator(local_cfg, np.random.default_rng(seed))
    sim.reset()
    while not sim.is_done():
        sim.step(local_cfg.system.cell_radius_m, local_cfg.system.cell_radius_m, action_idx=-1)
    return kpi_row(cfg, method, episode, sim.get_kpi(), sum(h["reward"] for h in sim.get_history_dicts()))


def run_policy(cfg: config.LeohosimConfig, agent: DQNAgent, seed: int, episode: int) -> dict:
    env = SimpleDQNHandoverEnv(cfg, np.random.default_rng(seed))
    state = env.reset()
    dense_total = 0.0
    counts = np.zeros(env.action_dim, dtype=int)
    while not env.sim.is_done():
        valid = env.valid_action_indices()
        action = agent.select_action(state, epsilon=0.0, valid_actions=valid)
        counts[action] += 1
        result = env.step(action)
        dense_total += result.reward
        state = result.state
        if result.done:
            break
    row = kpi_row(cfg, "Simple-DQN-HO", episode, env.get_kpi(), dense_total)
    for idx, count in enumerate(counts):
        row[f"action_count_{idx}"] = int(count)
    return row


def evaluate_paired(cfg: config.LeohosimConfig, agent: DQNAgent, episodes: int, seed_offset: int) -> list[dict]:
    rows = []
    for ep in range(episodes):
        seed = cfg.seed + seed_offset + ep
        unsafe = run_a3(cfg, seed, ep, "Unsafe-A3-0/0/0", 0.0, 0.0, 0.0)
        standard = run_a3(cfg, seed, ep, "Standard-A3", 1.0, 0.5, 0.4)
        dqn = run_policy(cfg, agent, seed, ep)
        rows.extend([unsafe, standard, dqn])
        rows.append(delta_row(ep, unsafe, dqn, "paired_delta_vs_unsafe"))
    return rows


def paired_summary(rows: list[dict]) -> dict[str, float]:
    deltas = [r for r in rows if r.get("method") == "paired_delta_vs_unsafe"]

    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in deltas])) if deltas else 0.0

    summary = {f"unsafe_{key}_mean": mean(key) for key in [
        "reward_gain",
        "sinr_gain_db",
        "outage_time_saved_s",
        "rlf_delta",
        "uho_delta",
        "uho_per_ho_delta",
        "extra_ho",
        "extra_rb_per_s_ue",
    ]}
    rlf_delta = summary["unsafe_rlf_delta_mean"]
    uho_delta = summary["unsafe_uho_delta_mean"]
    summary["safety_score"] = (
        summary["unsafe_reward_gain_mean"]
        + 55.0 * summary["unsafe_outage_time_saved_s_mean"]
        + 95.0 * max(-rlf_delta, 0.0)
        - 170.0 * max(rlf_delta, 0.0)
        + 3.0 * max(-uho_delta, 0.0)
        - 6.0 * max(uho_delta, 0.0)
        + 2.0 * summary["unsafe_sinr_gain_db_mean"]
        - 0.5 * max(summary["unsafe_extra_ho_mean"], 0.0)
        - 1.5 * max(summary["unsafe_extra_rb_per_s_ue_mean"], 0.0)
    )
    return summary


def add_eval_summary(row: dict, summary: dict[str, float], best_so_far: float) -> None:
    for key, value in summary.items():
        row[f"eval_{key}"] = value
    row["eval_safety_score"] = float(summary["safety_score"])
    row["eval_best_safety_score_so_far"] = max(float(best_so_far), float(summary["safety_score"]))


def delta_row(ep: int, baseline: dict, dqn: dict, method: str) -> dict:
    row = {"method": method, "episode": int(ep)}
    metrics = ["reward", "avg_sinr_db", "outage_time_s", "rlf_count", "uho_count", "uho_per_ho", "ho_count", "rb_per_s_ue", "avg_tos_s"]
    for metric in metrics:
        row[f"baseline_{metric}"] = baseline[metric]
        row[f"dqn_{metric}"] = dqn[metric]
    row["reward_gain"] = dqn["reward"] - baseline["reward"]
    row["sinr_gain_db"] = dqn["avg_sinr_db"] - baseline["avg_sinr_db"]
    row["outage_time_saved_s"] = baseline["outage_time_s"] - dqn["outage_time_s"]
    row["rlf_delta"] = dqn["rlf_count"] - baseline["rlf_count"]
    row["uho_delta"] = dqn["uho_count"] - baseline["uho_count"]
    row["uho_per_ho_delta"] = dqn["uho_per_ho"] - baseline["uho_per_ho"]
    row["extra_ho"] = dqn["ho_count"] - baseline["ho_count"]
    row["extra_rb_per_s_ue"] = dqn["rb_per_s_ue"] - baseline["rb_per_s_ue"]
    return row


def kpi_row(cfg: config.LeohosimConfig, method: str, episode: int, k, dense_reward: float) -> dict:
    total_time = float(cfg.simulation.total_time_s)
    kpi_score = compute_episode_kpi_reward(cfg, k)
    return {
        "method": method,
        "episode": int(episode),
        "dense_reward": float(dense_reward),
        "reward": kpi_score,
        "kpi_score": kpi_score,
        "avg_sinr_db": float(k.avg_sinr_db),
        "outage_fraction": float(k.outage_fraction),
        "outage_time_s": float(k.outage_fraction) * total_time,
        "rlf_count": int(k.rlf_count),
        "uho_count": int(k.uho_count),
        "uho_per_ho": float(k.uho_count) / max(float(k.ho_count), 1.0),
        "ho_count": int(k.ho_count),
        "short_tos_count": int(k.short_tos_count),
        "rb_count": int(k.rb_count),
        "rb_per_s_ue": float(k.rb_count) / max(total_time, 1e-9),
        "avg_tos_s": float(k.avg_tos_s),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_action_grid(out: Path) -> None:
    write_csv(out / "action_grid.csv", [{"action_idx": i, "action_name": name} for i, name in enumerate(SIMPLE_DQN_HO_ACTIONS)])


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def column_array(rows: list[dict], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def write_training_artifacts(out: Path, rows: list[dict]) -> None:
    write_csv(out / "train_log.csv", rows)
    train = [r for r in rows if r.get("method") == "Simple-DQN-HO"]
    if not train:
        return
    x = np.asarray([float(r["episode"]) for r in train], dtype=float)
    window = max(10, min(150, len(train) // 10))
    dense = column_array(train, "dense_reward")
    reward = column_array(train, "reward")
    gain = column_array(train, "train_unsafe_reward_gain")
    score = column_array(train, "train_unsafe_safety_score")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, y, title in [
        (axes[0, 0], dense, "Dense step reward"),
        (axes[0, 1], reward, "Episode KPI reward"),
        (axes[1, 0], gain, "Reward gain vs unsafe A3"),
        (axes[1, 1], score, "Safety score vs unsafe A3"),
    ]:
        finite = np.isfinite(y)
        ax.plot(x[finite], y[finite], alpha=0.18, linewidth=0.7)
        if finite.any():
            ax.plot(x, rolling_nanmean(y, window), linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(out / "simple_dqn_training_reward.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, key, title in [
        (axes[0, 0], "avg_sinr_db", "Avg SINR [dB]"),
        (axes[0, 1], "outage_time_s", "Outage [s]"),
        (axes[0, 2], "rlf_count", "RLF"),
        (axes[1, 0], "uho_count", "UHO"),
        (axes[1, 1], "ho_count", "HO"),
        (axes[1, 2], "rb_per_s_ue", "RB/s/UE"),
    ]:
        y = column_array(train, key)
        ax.plot(x, y, alpha=0.18, linewidth=0.7)
        ax.plot(x, rolling_nanmean(y, window), linewidth=2.0)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "simple_dqn_training_kpis.png", dpi=180)
    plt.close(fig)

    rl_reward = column_array(train, "rl_training_reward")
    if not np.isfinite(rl_reward).any():
        rl_reward = dense
    kpi_score = column_array(train, "kpi_score")
    if not np.isfinite(kpi_score).any():
        kpi_score = reward
    for filename, y, title, ylabel, color in [
        ("simple_dqn_rl_training_reward_only.png", rl_reward, "RL training reward", "Positive dense reward", "tab:blue"),
        ("simple_dqn_kpi_score_only.png", kpi_score, "KPI score", "KPI score", "tab:orange"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        finite = np.isfinite(y)
        ax.plot(x[finite], y[finite], linewidth=0.8, alpha=0.55, color=color, label="raw episode")
        if finite.any():
            ax.plot(x, rolling_nanmean(y, window), linewidth=2.2, color="black", label=f"rolling mean ({window})")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    uho = column_array(train, "uho_count")
    finite = np.isfinite(uho)
    ax.plot(x[finite], uho[finite], linewidth=0.8, alpha=0.55, color="tab:red", label="raw episode")
    if finite.any():
        ax.plot(x, rolling_nanmean(uho, window), linewidth=2.2, color="black", label=f"rolling mean ({window})")
    ax.set_title("UHO learning curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("UHO count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "simple_dqn_uho_learning_curve.png", dpi=180)
    plt.close(fig)

    eval_rows = [r for r in rows if np.isfinite(safe_float(r.get("eval_safety_score")))]
    if eval_rows:
        eval_x = np.asarray([float(r["episode"]) for r in eval_rows], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, key, title in [
            (axes[0, 0], "eval_safety_score", "Evaluation safety score"),
            (axes[0, 1], "eval_unsafe_reward_gain_mean", "Reward gain vs unsafe A3"),
            (axes[1, 0], "eval_unsafe_rlf_delta_mean", "RLF delta vs unsafe A3"),
            (axes[1, 1], "eval_unsafe_uho_delta_mean", "UHO delta vs unsafe A3"),
        ]:
            y = np.asarray([safe_float(r.get(key)) for r in eval_rows], dtype=float)
            finite = np.isfinite(y)
            ax.plot(eval_x[finite], y[finite], marker="o", linewidth=2.0)
            if key == "eval_safety_score" and finite.any():
                ax.plot(eval_x[finite], np.maximum.accumulate(y[finite]), linestyle="--", linewidth=1.5)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(title)
            ax.grid(alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("Episode")
        fig.tight_layout()
        fig.savefig(out / "simple_dqn_eval_convergence.png", dpi=180)
        plt.close(fig)


def safe_float(value) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    half = max(int(window) // 2, 0)
    for i in range(len(values)):
        chunk = values[max(0, i - half) : min(len(values), i + half + 1)]
        out[i] = float(np.nanmean(chunk)) if np.isfinite(chunk).any() else np.nan
    return out


def write_eval_summary(path: Path, rows: list[dict]) -> None:
    methods = ["Unsafe-A3-0/0/0", "Standard-A3", "Simple-DQN-HO"]
    metrics = [
        ("Reward", "reward"),
        ("Avg SINR [dB]", "avg_sinr_db"),
        ("Outage time [s]", "outage_time_s"),
        ("RLF count", "rlf_count"),
        ("UHO count", "uho_count"),
        ("HO count", "ho_count"),
        ("RB/s/UE", "rb_per_s_ue"),
        ("Avg ToS [s]", "avg_tos_s"),
    ]
    lines = [
        "# Proposed DQN Handover Evaluation",
        "",
        "| Metric | Unsafe A3 | Standard A3 | Proposed DQN |",
        "|---|---:|---:|---:|",
    ]
    for label, key in metrics:
        cells = []
        for method in methods:
            vals = [float(r[key]) for r in rows if r.get("method") == method]
            cells.append(f"{np.mean(vals):.6f}" if vals else "n/a")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    summary = paired_summary(rows)
    lines += ["", "## Paired DQN - Unsafe A3", ""]
    for key in [
        "unsafe_reward_gain_mean",
        "unsafe_sinr_gain_db_mean",
        "unsafe_outage_time_saved_s_mean",
        "unsafe_rlf_delta_mean",
        "unsafe_uho_delta_mean",
        "unsafe_extra_ho_mean",
        "safety_score",
    ]:
        lines.append(f"- {key}: {summary.get(key, 0.0):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_plots(out: Path, rows: list[dict]) -> None:
    methods = ["Unsafe-A3-0/0/0", "Standard-A3", "Simple-DQN-HO"]
    labels = ["Unsafe A3", "Standard A3", "Proposed DQN"]
    metrics = [
        ("Reward", "reward"),
        ("Avg SINR [dB]", "avg_sinr_db"),
        ("Outage [s]", "outage_time_s"),
        ("RLF", "rlf_count"),
        ("UHO", "uho_count"),
        ("HO", "ho_count"),
        ("RB/s/UE", "rb_per_s_ue"),
        ("Avg ToS [s]", "avg_tos_s"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (title, key) in zip(axes.ravel(), metrics):
        means = []
        for method in methods:
            vals = [float(r[key]) for r in rows if r.get("method") == method]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax.bar(labels, means)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "simple_dqn_eval_bars.png", dpi=180)
    plt.close(fig)

    paired = [r for r in rows if r.get("method") == "paired_delta_vs_unsafe"]
    if paired:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
        for ax, (title, key) in zip(axes.ravel(), [
            ("Reward gain", "reward_gain"),
            ("Outage saved [s]", "outage_time_saved_s"),
            ("RLF delta", "rlf_delta"),
            ("UHO delta", "uho_delta"),
        ]):
            vals = [float(r[key]) for r in paired]
            ax.bar([title], [float(np.mean(vals))])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "simple_dqn_delta_vs_unsafe.png", dpi=180)
        plt.close(fig)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


if __name__ == "__main__":
    main()
