"""Training helpers for DQN-based Event D2 threshold optimization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List
import shutil
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from leohosim_py import config
from leohosim_py.agents import DQNAgent
from leohosim_py.envs import D2ThresholdEnv


def train_dqn(
    cfg: config.LeohosimConfig,
    output_dir: Path,
    episodes: int | None = None,
    double_dqn: bool = False,
    eval_interval: int = 50,
    eval_episodes: int | None = None,
    episode_action: bool = False,
    train_repeats: int = 1,
    kpi_reward: bool = False,
    plot_window: int = 100,
    clean_plots: bool = True,
    hybrid_window_action: bool = False,
    event_window_action: bool = False,
    event_decision_action: bool = False,
    event_decision_wide_gate: bool = False,
    event_decision_discriminative_gate: bool = False,
    contextual_bandit_action: bool = False,
    handover_mode_action: bool = False,
    mode_teacher_warmup_episodes: int = 0,
    mode_safety_shield: bool = True,
    counterfactual_samples: int = 0,
    bandit_counterfactual_samples: int = 0,
    counterfactual_suffix_global: bool = False,
    local_window_s: float = 1.0,
    local_reward_weight: float = 0.5,
    global_reward_weight: float = 0.5,
    rl_reward_scale: float = 30.0,
    fixed_eval_seed: bool = True,
    log_interval: int | None = None,
    diagnostic_sensitivity_interval: int = 0,
    guard_short_tos: bool = False,
    guard_tos_s: float | None = None,
    guard_min_thresh1_m: float | None = None,
    guard_max_thresh2_m: float | None = None,
) -> List[Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    env = D2ThresholdEnv(cfg, rng)
    action_pairs = env.threshold_mode_pairs() if handover_mode_action else env.actions
    action_names = env.mode_action_names() if handover_mode_action else ["" for _ in action_pairs]
    agent_action_dim = len(action_pairs)
    dqn_cfg = cfg.ddqn if double_dqn else cfg.dqn
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=agent_action_dim,
        cfg=dqn_cfg,
        seed=cfg.seed,
        double_dqn=double_dqn,
    )
    label = "ddqn" if double_dqn else "dqn"
    n_episodes = episodes or cfg.simulation.episodes_train
    rows: List[Dict[str, float]] = []
    best_eval_reward = -float("inf")
    all_action_counts = np.zeros(agent_action_dim, dtype=np.int64)
    started_at = time.time()
    if log_interval is None:
        log_interval = eval_interval if eval_interval > 0 else max(n_episodes // 100, 1)
    log_interval = max(int(log_interval), 1)

    with open(output_dir / "reward_summary.json", "w", encoding="utf-8") as f:
        json.dump(reward_config_summary(cfg), f, indent=2)

    with open(output_dir / "action_grid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["action_idx", "thresh1_m", "thresh2_m", "mode"])
        writer.writeheader()
        for idx, (th1, th2) in enumerate(action_pairs):
            writer.writerow({"action_idx": idx, "thresh1_m": th1, "thresh2_m": th2, "mode": action_names[idx]})

    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        last_loss = None
        epsilon = agent.epsilon(ep)
        actions_taken: List[int] = []
        opportunity_count = 0
        discriminative_opportunity_count = 0
        pass_action_counts: List[float] = []
        action_reward_stds: List[float] = []
        best_tie_counts: List[float] = []

        if handover_mode_action:
            mode_pairs = env.threshold_mode_pairs()
            while True:
                valid_mode_actions = valid_handover_mode_actions(env) if mode_safety_shield else None
                if ep < max(mode_teacher_warmup_episodes, 0):
                    action = select_balanced_mode_action(env)
                    if valid_mode_actions is not None and action not in set(valid_mode_actions.tolist()):
                        action = int(valid_mode_actions[-1])
                    imitation_loss = agent.supervised_action_step(state, action)
                    last_loss = imitation_loss
                else:
                    action = agent.select_action(state, epsilon, valid_actions=valid_mode_actions)
                actions_taken.append(action)
                thresh1_m, thresh2_m = mode_pairs[int(action)]
                result = env.step_thresholds(thresh1_m, thresh2_m, action_idx=int(action))
                mode_reward = compute_handover_mode_step_reward(cfg, result.info, action_idx=int(action))
                agent.remember((state, action, mode_reward / max(rl_reward_scale, 1e-9), result.state, result.done))
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss
                episode_reward += result.reward
                state = result.state
                if result.done:
                    break

        elif event_decision_action:
            pending_transitions = []
            baseline_action = env.baseline_action_index()
            max_window_s = max(local_window_s, cfg.simulation.sample_time_s)
            initial_snapshot = env.snapshot()
            _, _, baseline_fixed_rewards = env.rollout_fixed_to_done(baseline_action)
            baseline_episode_reward = (
                compute_episode_kpi_reward(cfg, env.get_kpi()) if kpi_reward else float(np.sum(baseline_fixed_rewards))
            )
            state = env.restore(initial_snapshot)
            while True:
                has_opportunity = (
                    env.has_ho_opportunity()
                    if event_decision_wide_gate
                    else env.has_baseline_ho_opportunity(baseline_action)
                )
                if has_opportunity:
                    opportunity_count += 1
                    pass_mask = env.d2_action_pass_mask()
                    pass_action_counts.append(float(np.sum(pass_mask)))
                    is_discriminative = bool(np.any(pass_mask) and np.any(~pass_mask))
                    if is_discriminative:
                        discriminative_opportunity_count += 1
                    if event_decision_discriminative_gate and not is_discriminative:
                        has_opportunity = False
                if not has_opportunity:
                    result = env.step(baseline_action)
                    actions_taken.append(baseline_action)
                    episode_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                    continue

                start_state = state
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon, valid_actions=valid_actions)
                actions_taken.append(action)
                event_snapshot = env.snapshot()
                if diagnostic_sensitivity_interval > 0 and (ep + 1) % diagnostic_sensitivity_interval == 0:
                    sensitivity = env.evaluate_action_sensitivity(max_window_s, baseline_action)
                    action_reward_stds.append(float(sensitivity["reward_std"]))
                    best_tie_counts.append(float(sensitivity["best_tie_count"]))
                    env.restore(event_snapshot)
                _, _, baseline_infos, baseline_rewards = env.rollout_event_decision(baseline_action, max_window_s)
                env.restore(event_snapshot)
                next_state, done, local_infos, local_rewards = env.rollout_event_decision(action, max_window_s)
                actual_after_event = env.snapshot()
                episode_reward += float(np.sum(local_rewards))
                event_reward = compute_event_decision_reward(
                    cfg,
                    local_infos,
                    local_rewards,
                    window_s=len(local_infos) * cfg.simulation.sample_time_s,
                    baseline_infos=baseline_infos,
                    baseline_step_rewards=baseline_rewards,
                )
                pending_transitions.append((start_state, action, event_reward, next_state, done, False, None))
                extra_count = min(max(counterfactual_samples, 0), max(len(valid_actions) - 1, 0))
                if extra_count > 0:
                    alternatives = [idx for idx in valid_actions if idx != action]
                    sampled = rng.choice(alternatives, size=extra_count, replace=False)
                    for cf_action in sampled:
                        env.restore(event_snapshot)
                        cf_next_state, cf_done, cf_infos, cf_rewards = env.rollout_event_decision(
                            int(cf_action),
                            max_window_s,
                        )
                        cf_event_reward = compute_event_decision_reward(
                            cfg,
                            cf_infos,
                            cf_rewards,
                            window_s=len(cf_infos) * cfg.simulation.sample_time_s,
                            baseline_infos=baseline_infos,
                            baseline_step_rewards=baseline_rewards,
                        )
                        cf_global_reward = None
                        if counterfactual_suffix_global:
                            if not cf_done:
                                env.rollout_fixed_to_done(baseline_action)
                            cf_score = compute_episode_kpi_reward(cfg, env.get_kpi()) if kpi_reward else float(
                                sum(item.get("reward", 0.0) for item in env.get_history_dicts())
                            )
                            cf_global_reward = cf_score - baseline_episode_reward
                        pending_transitions.append(
                            (start_state, int(cf_action), cf_event_reward, cf_next_state, cf_done, True, cf_global_reward)
                        )
                    env.restore(actual_after_event)
                state = next_state
                if done:
                    break

            kpi_after_episode = env.get_kpi()
            global_score = compute_episode_kpi_reward(cfg, kpi_after_episode) if kpi_reward else episode_reward
            global_reward = global_score - baseline_episode_reward
            for transition in pending_transitions:
                start_state, action, event_reward, next_state, done, is_counterfactual, cf_global_reward = transition
                train_reward = local_reward_weight * event_reward
                if not is_counterfactual:
                    train_reward += global_reward_weight * global_reward
                elif counterfactual_suffix_global:
                    train_reward += global_reward_weight * float(cf_global_reward or 0.0)
                agent.remember((start_state, action, train_reward / max(rl_reward_scale, 1e-9), next_state, done))
            for _ in range(max(train_repeats, 1) * max(1, len(pending_transitions))):
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss

        elif event_window_action:
            pending_transitions = []
            window_steps = max(1, int(round(local_window_s / cfg.simulation.sample_time_s)))
            fallback_action = env.conservative_action_index()
            while True:
                if not env.has_ho_opportunity():
                    result = env.step(fallback_action)
                    actions_taken.append(fallback_action)
                    episode_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                    continue

                start_state = state
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon, valid_actions=valid_actions)
                actions_taken.append(action)
                local_infos = []
                local_rewards = []
                for _ in range(window_steps):
                    result = env.step(action)
                    local_infos.append(result.info)
                    local_rewards.append(result.reward)
                    episode_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                    if not env.has_ho_opportunity():
                        break
                local_reward = compute_local_window_reward(
                    cfg,
                    local_infos,
                    local_rewards,
                    window_s=len(local_infos) * cfg.simulation.sample_time_s,
                )
                pending_transitions.append((start_state, action, local_reward, state, env.sim.is_done()))
                if env.sim.is_done():
                    break

            kpi_after_episode = env.get_kpi()
            global_reward = compute_episode_kpi_reward(cfg, kpi_after_episode) if kpi_reward else episode_reward
            for start_state, action, local_reward, next_state, done in pending_transitions:
                train_reward = local_reward_weight * local_reward + global_reward_weight * global_reward
                agent.remember((start_state, action, train_reward / max(rl_reward_scale, 1e-9), next_state, done))
            for _ in range(max(train_repeats, 1) * max(1, len(pending_transitions))):
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss

        elif hybrid_window_action:
            pending_transitions = []
            window_steps = max(1, int(round(local_window_s / cfg.simulation.sample_time_s)))
            while True:
                start_state = state
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon, valid_actions=valid_actions)
                actions_taken.append(action)
                local_infos = []
                local_rewards = []
                for _ in range(window_steps):
                    result = env.step(action)
                    local_infos.append(result.info)
                    local_rewards.append(result.reward)
                    episode_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                local_reward = compute_local_window_reward(
                    cfg,
                    local_infos,
                    local_rewards,
                    window_s=len(local_infos) * cfg.simulation.sample_time_s,
                )
                pending_transitions.append((start_state, action, local_reward, state, env.sim.is_done()))
                if env.sim.is_done():
                    break

            kpi_after_episode = env.get_kpi()
            global_reward = compute_episode_kpi_reward(cfg, kpi_after_episode) if kpi_reward else episode_reward
            for start_state, action, local_reward, next_state, done in pending_transitions:
                train_reward = local_reward_weight * local_reward + global_reward_weight * global_reward
                agent.remember((start_state, action, train_reward / max(rl_reward_scale, 1e-9), next_state, done))
            for _ in range(max(train_repeats, 1) * max(1, len(pending_transitions))):
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss

        elif contextual_bandit_action:
            initial_state = state
            baseline_action = env.baseline_action_index()
            initial_snapshot = env.snapshot()

            _, _, baseline_fixed_rewards = env.rollout_fixed_to_done(baseline_action)
            baseline_episode_reward = (
                compute_episode_kpi_reward(cfg, env.get_kpi()) if kpi_reward else float(np.sum(baseline_fixed_rewards))
            )

            env.restore(initial_snapshot)
            valid_actions = env.valid_action_indices(
                guard_short_tos,
                guard_tos_s,
                guard_min_thresh1_m,
                guard_max_thresh2_m,
            )
            action = agent.select_action(state, epsilon, valid_actions=valid_actions)
            actions_taken.append(action)
            terminal_state, _, fixed_rewards = env.rollout_fixed_to_done(action)
            episode_reward = float(np.sum(fixed_rewards))
            action_episode_reward = compute_episode_kpi_reward(cfg, env.get_kpi()) if kpi_reward else episode_reward
            relative_reward = action_episode_reward - baseline_episode_reward
            agent.remember((initial_state, action, relative_reward / max(rl_reward_scale, 1e-9), terminal_state, True))

            extra_count = min(max(bandit_counterfactual_samples, 0), max(len(valid_actions) - 1, 0))
            if extra_count > 0:
                alternatives = [idx for idx in valid_actions if idx != action]
                sampled = rng.choice(alternatives, size=extra_count, replace=False)
                actual_snapshot = env.snapshot()
                for cf_action in sampled:
                    env.restore(initial_snapshot)
                    cf_terminal_state, _, cf_rewards = env.rollout_fixed_to_done(int(cf_action))
                    cf_episode_reward = (
                        compute_episode_kpi_reward(cfg, env.get_kpi()) if kpi_reward else float(np.sum(cf_rewards))
                    )
                    cf_relative_reward = cf_episode_reward - baseline_episode_reward
                    agent.remember(
                        (
                            initial_state,
                            int(cf_action),
                            cf_relative_reward / max(rl_reward_scale, 1e-9),
                            cf_terminal_state,
                            True,
                        )
                    )
                env.restore(actual_snapshot)

            for _ in range(max(train_repeats, 1) * max(1, 1 + extra_count)):
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss

        elif episode_action:
            initial_state = state
            valid_actions = env.valid_action_indices(
                guard_short_tos,
                guard_tos_s,
                guard_min_thresh1_m,
                guard_max_thresh2_m,
            )
            action = agent.select_action(state, epsilon, valid_actions=valid_actions)
            actions_taken.append(action)
            while True:
                result = env.step(action)
                episode_reward += result.reward
                state = result.state
                if result.done:
                    break
            if kpi_reward:
                train_reward = compute_episode_kpi_reward(cfg, env.get_kpi())
            else:
                train_reward = episode_reward
            scaled_reward = train_reward / max(rl_reward_scale, 1e-9)
            agent.remember((initial_state, action, scaled_reward, state, True))
            for _ in range(max(train_repeats, 1)):
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss
        else:
            while True:
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon, valid_actions=valid_actions)
                actions_taken.append(action)
                result = env.step(action)
                agent.remember((state, action, result.reward, result.state, result.done))
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss
                episode_reward += result.reward
                state = result.state
                if result.done:
                    break

        kpi = env.get_kpi()
        logged_reward = compute_episode_kpi_reward(cfg, kpi) if kpi_reward else episode_reward
        reward_breakdown = compute_episode_kpi_reward_breakdown(cfg, kpi)
        if abs(float(reward_breakdown["total"]) - compute_episode_kpi_reward(cfg, kpi)) > 1e-6:
            warnings.warn("episode reward breakdown total does not match compute_episode_kpi_reward", RuntimeWarning)
        eval_summary = {}
        if eval_interval > 0 and ((ep + 1) % eval_interval == 0 or ep == n_episodes - 1):
            eval_summary = _evaluate_current_policy(
                cfg,
                agent,
                episodes=eval_episodes or cfg.simulation.episodes_eval,
                seed=cfg.seed + 100_000 if fixed_eval_seed else cfg.seed + 100_000 + ep,
                episode_action=episode_action,
                kpi_reward=kpi_reward,
                hybrid_window_action=hybrid_window_action,
                event_window_action=event_window_action,
                event_decision_action=event_decision_action,
                event_decision_wide_gate=event_decision_wide_gate,
                event_decision_discriminative_gate=event_decision_discriminative_gate,
                contextual_bandit_action=contextual_bandit_action,
                handover_mode_action=handover_mode_action,
                mode_safety_shield=mode_safety_shield,
                local_window_s=local_window_s,
                guard_short_tos=guard_short_tos,
                guard_tos_s=guard_tos_s,
                guard_min_thresh1_m=guard_min_thresh1_m,
                guard_max_thresh2_m=guard_max_thresh2_m,
            )
            if eval_summary["reward"] > best_eval_reward:
                best_eval_reward = eval_summary["reward"]
                suffix = "ddqn" if double_dqn else "dqn"
                agent.save(str(output_dir / f"best_eval_{suffix}_model.pt"))
        if not actions_taken:
            actions_taken = [2 if handover_mode_action else env.baseline_action_index()]
        action_counts = np.bincount(actions_taken, minlength=agent_action_dim)
        all_action_counts += action_counts
        dominant_action = int(np.argmax(action_counts))
        mean_thresh1 = float(np.mean([action_pairs[a][0] for a in actions_taken]))
        mean_thresh2 = float(np.mean([action_pairs[a][1] for a in actions_taken]))
        row = {
            "episode": ep,
            "epsilon": epsilon,
            "reward": logged_reward,
            "step_reward_sum": episode_reward,
            "loss": -1.0 if last_loss is None else last_loss,
            "dominant_action": dominant_action,
            "dominant_action_fraction": float(action_counts[dominant_action] / max(len(actions_taken), 1)),
            "decision_count": int(len(actions_taken)),
            "mean_thresh1_m": mean_thresh1,
            "mean_thresh2_m": mean_thresh2,
            "avg_sinr_db": kpi.avg_sinr_db,
            "ho_count": kpi.ho_count,
            "uho_count": kpi.uho_count,
            "rlf_count": kpi.rlf_count,
            "hopp_count": kpi.hopp_count,
            "avg_tos_s": kpi.avg_tos_s,
            "rb_count": kpi.rb_count,
            "reward_sinr": reward_breakdown["sinr_term"],
            "reward_outage": reward_breakdown["outage_term"],
            "reward_rlf": reward_breakdown["rlf_term"],
            "reward_uho": reward_breakdown["uho_term"],
            "reward_short_tos": reward_breakdown["short_tos_term"],
            "reward_ho": reward_breakdown["ho_term"],
            "reward_rb": reward_breakdown["rb_term"],
            "reward_avg_tos": reward_breakdown["avg_tos_term"],
            "opportunity_count": opportunity_count,
            "discriminative_opportunity_count": discriminative_opportunity_count,
            "mean_pass_action_count": float(np.mean(pass_action_counts)) if pass_action_counts else 0.0,
            "mean_action_reward_std": float(np.mean(action_reward_stds)) if action_reward_stds else 0.0,
            "mean_best_tie_count": float(np.mean(best_tie_counts)) if best_tie_counts else 0.0,
            "eval_reward": eval_summary.get("reward", np.nan),
            "eval_avg_sinr_db": eval_summary.get("avg_sinr_db", np.nan),
            "eval_outage_fraction": eval_summary.get("outage_fraction", np.nan),
            "eval_ho_count": eval_summary.get("ho_count", np.nan),
            "eval_uho_count": eval_summary.get("uho_count", np.nan),
            "eval_rlf_count": eval_summary.get("rlf_count", np.nan),
            "eval_hopp_count": eval_summary.get("hopp_count", np.nan),
            "eval_short_tos_count": eval_summary.get("short_tos_count", np.nan),
            "eval_avg_tos_s": eval_summary.get("avg_tos_s", np.nan),
            "eval_rb_count": eval_summary.get("rb_count", np.nan),
            "eval_reward_sinr": eval_summary.get("reward_sinr", np.nan),
            "eval_reward_outage": eval_summary.get("reward_outage", np.nan),
            "eval_reward_rlf": eval_summary.get("reward_rlf", np.nan),
            "eval_reward_uho": eval_summary.get("reward_uho", np.nan),
            "eval_reward_short_tos": eval_summary.get("reward_short_tos", np.nan),
            "eval_reward_ho": eval_summary.get("reward_ho", np.nan),
            "eval_reward_rb": eval_summary.get("reward_rb", np.nan),
            "eval_reward_avg_tos": eval_summary.get("reward_avg_tos", np.nan),
        }
        for idx, count in enumerate(action_counts):
            row[f"action_count_{idx}"] = int(count)
        rows.append(row)

        if (ep + 1) % log_interval == 0 or ep == 0 or ep == n_episodes - 1:
            elapsed_s = max(time.time() - started_at, 1e-9)
            progress = (ep + 1) / max(n_episodes, 1)
            eta_s = elapsed_s * (1.0 - progress) / max(progress, 1e-9)
            eval_text = ""
            if eval_summary:
                eval_text = (
                    f", eval_reward={eval_summary['reward']:8.2f}, "
                    f"eval_SINR={eval_summary['avg_sinr_db']:6.2f} dB"
                )
            progress_line = (
                f"[{label}] episode={ep + 1:4d}/{n_episodes}, "
                f"{_progress_bar(progress)} "
                f"progress={100.0 * progress:5.1f}%, elapsed={_format_duration(elapsed_s)}, "
                f"eta={_format_duration(eta_s)}, "
                f"eps={epsilon:.3f}, reward={episode_reward:8.2f}, "
                f"SINR={kpi.avg_sinr_db:6.2f} dB, HO={kpi.ho_count}, RLF={kpi.rlf_count}"
                f"{eval_text}"
            )
            print("\r" + progress_line.ljust(180), end="" if ep != n_episodes - 1 else "\n", flush=True)

    if rows:
        with open(output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with open(output_dir / "action_counts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["action_idx", "thresh1_m", "thresh2_m", "mode", "count"])
            writer.writeheader()
            for idx, (th1, th2) in enumerate(action_pairs):
                writer.writerow(
                    {
                        "action_idx": idx,
                        "thresh1_m": th1,
                        "thresh2_m": th2,
                        "mode": action_names[idx],
                        "count": int(all_action_counts[idx]),
                    }
                )
        if clean_plots:
            _clean_plot_outputs(output_dir)
        _plot_training_outputs(rows, output_dir, action_pairs, double_dqn, plot_window)

    agent.save(str(output_dir / ("ddqn_model.pt" if double_dqn else "dqn_model.pt")))
    return rows


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_bar(progress: float, width: int = 28) -> str:
    progress = float(np.clip(progress, 0.0, 1.0))
    filled = int(round(progress * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _clean_plot_outputs(output_dir: Path) -> None:
    for pattern in ("fig_*.png", "paper_*.png"):
        for path in output_dir.glob(pattern):
            path.unlink()
    paper_dir = output_dir / "paper_figures"
    if paper_dir.exists():
        shutil.rmtree(paper_dir)


def _plot_training_outputs(
    rows: List[Dict[str, float]],
    output_dir: Path,
    actions: List[tuple[float, float]],
    double_dqn: bool,
    plot_window: int = 100,
) -> None:
    label = "DDQN" if double_dqn else "DQN"
    episodes = np.array([r["episode"] for r in rows], dtype=float)
    rewards = np.array([r["reward"] for r in rows], dtype=float)
    plot_window = max(1, min(plot_window, len(rewards)))
    ma = _moving_average(rewards, window=plot_window)
    ma_x = episodes[len(episodes) - len(ma):]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.18, label="episode reward")
    plt.plot(ma_x, ma, linewidth=2.2, label=f"moving average ({plot_window})")
    plt.xlabel("Episode")
    plt.ylabel("Normalized KPI reward")
    plt.title(f"{label} Episode Reward Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_01_reward_convergence.png", dpi=180)
    plt.close()

    eval_rows = [r for r in rows if np.isfinite(r.get("eval_reward", np.nan))]
    if eval_rows:
        eval_ep = np.array([r["episode"] for r in eval_rows], dtype=float)
        plt.figure(figsize=(9, 5))
        plt.plot(eval_ep, [r["eval_reward"] for r in eval_rows], marker="o", markersize=3, label="greedy eval reward")
        plt.xlabel("Training episode")
        plt.ylabel("Normalized KPI reward")
        plt.title(f"{label} Greedy Policy Evaluation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "fig_02_eval_reward_convergence.png", dpi=180)
        plt.close()

        fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        axs[0, 0].plot(eval_ep, [r["eval_avg_sinr_db"] for r in eval_rows], marker="o", markersize=3, color="tab:green")
        axs[0, 0].set_ylabel("Avg DL SINR (dB)")
        axs[0, 1].plot(eval_ep, [r.get("eval_outage_fraction", np.nan) for r in eval_rows], marker="o", markersize=3, color="tab:gray")
        axs[0, 1].set_ylabel("Outage fraction")
        axs[1, 0].plot(eval_ep, [r["eval_rlf_count"] for r in eval_rows], marker="o", markersize=3, label="RLF", color="tab:red")
        axs[1, 0].plot(eval_ep, [r.get("eval_uho_count", np.nan) for r in eval_rows], marker="o", markersize=3, label="UHO", color="tab:orange")
        axs[1, 0].plot(eval_ep, [r.get("eval_hopp_count", np.nan) for r in eval_rows], marker="o", markersize=3, label="HOPP", color="tab:pink")
        axs[1, 0].set_ylabel("Avg count")
        axs[1, 0].legend()
        axs[1, 1].plot(eval_ep, [r["eval_ho_count"] for r in eval_rows], marker="o", markersize=3, label="HO", color="tab:purple")
        axs[1, 1].plot(eval_ep, [r.get("eval_short_tos_count", np.nan) for r in eval_rows], marker="o", markersize=3, label="Short ToS", color="tab:brown")
        axs[1, 1].plot(eval_ep, [r.get("eval_avg_tos_s", np.nan) for r in eval_rows], marker="o", markersize=3, label="Avg ToS", color="tab:blue")
        axs[1, 1].set_ylabel("Count / seconds")
        axs[1, 1].legend()
        for ax in axs.flat:
            ax.set_xlabel("Training episode")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{label} Evaluation KPI Convergence")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_03_eval_kpi_convergence.png", dpi=180)
        plt.close(fig)

    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(3, 1, 1)
    ma_ep = episodes[len(episodes) - len(ma):]
    ax1.plot(ma_ep, _moving_average(np.array([r["avg_sinr_db"] for r in rows], dtype=float), plot_window), color="tab:blue")
    ax1.set_ylabel("Avg SINR (dB)")
    ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(ma_ep, _moving_average(np.array([r["rlf_count"] for r in rows], dtype=float), plot_window), label="RLF", color="tab:red")
    ax2.plot(ma_ep, _moving_average(np.array([r["uho_count"] for r in rows], dtype=float), plot_window), label="UHO", color="tab:orange")
    ax2.plot(ma_ep, _moving_average(np.array([r["hopp_count"] for r in rows], dtype=float), plot_window), label="HOPP", color="tab:pink")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(ma_ep, _moving_average(np.array([r["ho_count"] for r in rows], dtype=float), plot_window), label="HO", color="tab:green")
    ax3.plot(ma_ep, _moving_average(np.array([r["avg_tos_s"] for r in rows], dtype=float), plot_window), label="Avg ToS", color="tab:blue")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Count / seconds")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_04_training_kpi_moving_average.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(ma_ep, _moving_average(np.array([r["mean_thresh1_m"] for r in rows], dtype=float), plot_window), label="Thresh1")
    plt.plot(ma_ep, _moving_average(np.array([r["mean_thresh2_m"] for r in rows], dtype=float), plot_window), label="Thresh2")
    plt.xlabel("Episode")
    plt.ylabel("Threshold (m)")
    plt.title(f"{label} Learned Threshold Trace")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_05_threshold_trace.png", dpi=180)
    plt.close()

    recent_n = min(1000, len(rows))
    count_cols = [f"action_count_{idx}" for idx in range(len(actions))]
    if all(col in rows[0] for col in count_cols):
        usage = np.array([sum(int(r.get(col, 0)) for r in rows[-recent_n:]) for col in count_cols], dtype=int)
    else:
        dominant = np.array([int(r["dominant_action"]) for r in rows])
        usage = np.bincount(dominant[-recent_n:], minlength=len(actions))
    labels = [f"{i}\n({th1:.0f},{th2:.0f})" for i, (th1, th2) in enumerate(actions)]
    plt.figure(figsize=(max(9, len(actions) * 0.55), 5))
    plt.bar(np.arange(len(actions)), usage, color="tab:cyan")
    plt.xticks(np.arange(len(actions)), labels, rotation=45, ha="right")
    plt.xlabel("Action idx (Thresh1, Thresh2)")
    plt.ylabel(f"Selections in last {recent_n} episodes")
    plt.title(f"{label} Action Usage")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_06_action_usage_recent.png", dpi=180)
    plt.close()

    sorted_rewards = np.sort(rewards)
    cdf = np.linspace(0.0, 1.0, len(sorted_rewards), endpoint=True)
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_rewards, cdf, linewidth=2.0, color="tab:blue")
    plt.xlabel("Episode normalized KPI reward")
    plt.ylabel("CDF")
    plt.title(f"{label} Reward Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_07_reward_cdf.png", dpi=180)
    plt.close()


def evaluate_greedy(
    cfg: config.LeohosimConfig,
    agent: DQNAgent,
    episodes: int | None = None,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed + 10_000)
    env = D2ThresholdEnv(cfg, rng)
    rows: List[Dict[str, float]] = []
    for ep in range(episodes or cfg.simulation.episodes_eval):
        state = env.reset()
        total_reward = 0.0
        while True:
            action = agent.select_action(state, epsilon=0.0)
            result = env.step(action)
            total_reward += result.reward
            state = result.state
            if result.done:
                break
        kpi = env.get_kpi()
        rows.append(
            {
                "episode": ep,
                "reward": total_reward,
                "avg_sinr_db": kpi.avg_sinr_db,
                "ho_count": kpi.ho_count,
                "uho_count": kpi.uho_count,
                "rlf_count": kpi.rlf_count,
                "hopp_count": kpi.hopp_count,
                "avg_tos_s": kpi.avg_tos_s,
                "rb_count": kpi.rb_count,
            }
        )
    return rows


def _evaluate_current_policy(
    cfg: config.LeohosimConfig,
    agent: DQNAgent,
    episodes: int,
    seed: int,
    episode_action: bool = False,
    kpi_reward: bool = False,
    hybrid_window_action: bool = False,
    event_window_action: bool = False,
    event_decision_action: bool = False,
    event_decision_wide_gate: bool = False,
    event_decision_discriminative_gate: bool = False,
    contextual_bandit_action: bool = False,
    handover_mode_action: bool = False,
    mode_safety_shield: bool = True,
    local_window_s: float = 1.0,
    guard_short_tos: bool = False,
    guard_tos_s: float | None = None,
    guard_min_thresh1_m: float | None = None,
    guard_max_thresh2_m: float | None = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    env = D2ThresholdEnv(cfg, rng)
    rewards = []
    sinrs = []
    hos = []
    rlfs = []
    uhos = []
    hopps = []
    short_tos = []
    outages = []
    avg_tos = []
    rbs = []
    reward_parts: dict[str, list[float]] = {
        "sinr_term": [],
        "outage_term": [],
        "rlf_term": [],
        "uho_term": [],
        "short_tos_term": [],
        "ho_term": [],
        "rb_term": [],
        "avg_tos_term": [],
    }
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        if handover_mode_action:
            mode_pairs = env.threshold_mode_pairs()
            while True:
                valid_mode_actions = valid_handover_mode_actions(env) if mode_safety_shield else None
                action = agent.select_action(state, epsilon=0.0, valid_actions=valid_mode_actions)
                thresh1_m, thresh2_m = mode_pairs[int(action)]
                result = env.step_thresholds(thresh1_m, thresh2_m, action_idx=int(action))
                total_reward += result.reward
                state = result.state
                if result.done:
                    break
        elif contextual_bandit_action:
            valid_actions = env.valid_action_indices(
                guard_short_tos,
                guard_tos_s,
                guard_min_thresh1_m,
                guard_max_thresh2_m,
            )
            action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
            _, _, fixed_rewards = env.rollout_fixed_to_done(action)
            total_reward += float(np.sum(fixed_rewards))
        elif event_decision_action:
            baseline_action = env.baseline_action_index()
            max_window_s = max(local_window_s, cfg.simulation.sample_time_s)
            while True:
                has_opportunity = (
                    env.has_ho_opportunity()
                    if event_decision_wide_gate
                    else env.has_baseline_ho_opportunity(baseline_action)
                )
                if has_opportunity and event_decision_discriminative_gate and not env.has_discriminative_ho_opportunity():
                    has_opportunity = False
                if not has_opportunity:
                    result = env.step(baseline_action)
                    total_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                    continue
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
                state, done, local_infos, local_rewards = env.rollout_event_decision(action, max_window_s)
                total_reward += float(np.sum(local_rewards))
                if done:
                    break
        elif event_window_action:
            window_steps = max(1, int(round(local_window_s / cfg.simulation.sample_time_s)))
            fallback_action = env.conservative_action_index()
            while True:
                if not env.has_ho_opportunity():
                    result = env.step(fallback_action)
                    total_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                    continue
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
                for _ in range(window_steps):
                    result = env.step(action)
                    total_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                    if not env.has_ho_opportunity():
                        break
                if env.sim.is_done():
                    break
        elif hybrid_window_action:
            window_steps = max(1, int(round(local_window_s / cfg.simulation.sample_time_s)))
            while True:
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
                for _ in range(window_steps):
                    result = env.step(action)
                    total_reward += result.reward
                    state = result.state
                    if result.done:
                        break
                if env.sim.is_done():
                    break
        elif episode_action:
            valid_actions = env.valid_action_indices(
                guard_short_tos,
                guard_tos_s,
                guard_min_thresh1_m,
                guard_max_thresh2_m,
            )
            action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
            while True:
                result = env.step(action)
                total_reward += result.reward
                state = result.state
                if result.done:
                    break
        else:
            while True:
                valid_actions = env.valid_action_indices(
                    guard_short_tos,
                    guard_tos_s,
                    guard_min_thresh1_m,
                    guard_max_thresh2_m,
                )
                action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
                result = env.step(action)
                total_reward += result.reward
                state = result.state
                if result.done:
                    break
        k = env.get_kpi()
        breakdown = compute_episode_kpi_reward_breakdown(cfg, k)
        reward_value = float(breakdown["total"]) if kpi_reward else total_reward
        if abs(float(breakdown["total"]) - compute_episode_kpi_reward(cfg, k)) > 1e-6:
            warnings.warn("eval reward breakdown total does not match compute_episode_kpi_reward", RuntimeWarning)
        rewards.append(reward_value)
        for key in reward_parts:
            reward_parts[key].append(float(breakdown[key]))
        sinrs.append(k.avg_sinr_db)
        hos.append(k.ho_count)
        rlfs.append(k.rlf_count)
        uhos.append(k.uho_count)
        hopps.append(k.hopp_count)
        short_tos.append(k.short_tos_count)
        outages.append(k.outage_fraction)
        avg_tos.append(k.avg_tos_s)
        rbs.append(k.rb_count)
    summary = {
        "reward": float(np.mean(rewards)),
        "avg_sinr_db": float(np.mean(sinrs)),
        "outage_fraction": float(np.mean(outages)),
        "ho_count": float(np.mean(hos)),
        "uho_count": float(np.mean(uhos)),
        "rlf_count": float(np.mean(rlfs)),
        "hopp_count": float(np.mean(hopps)),
        "short_tos_count": float(np.mean(short_tos)),
        "avg_tos_s": float(np.mean(avg_tos)),
        "rb_count": float(np.mean(rbs)),
    }
    for key, values in reward_parts.items():
        out_key = {
            "sinr_term": "reward_sinr",
            "outage_term": "reward_outage",
            "rlf_term": "reward_rlf",
            "uho_term": "reward_uho",
            "short_tos_term": "reward_short_tos",
            "ho_term": "reward_ho",
            "rb_term": "reward_rb",
            "avg_tos_term": "reward_avg_tos",
        }[key]
        summary[out_key] = float(np.mean(values))
    return summary


def compute_episode_kpi_reward(cfg: config.LeohosimConfig, k) -> float:
    return float(compute_episode_kpi_reward_breakdown(cfg, k)["total"])


def select_balanced_mode_action(env: D2ThresholdEnv, guard_tos_s: float = 1.0, sinr_margin_db: float = 1.0, target_gap_db: float = 2.0) -> int:
    """Teacher policy for mode-DQN warmup: guard short ToS, pull HO only under clear radio risk."""
    ue = env.sim.ue
    if ue.ml_m is None or ue.sinr_db is None:
        env._refresh_measurements()
    current_tos_s = max(0.0, env.sim.time_s - ue.serving_start_time_s)
    if current_tos_s < guard_tos_s:
        return 1  # conservative
    serving_sinr = float(ue.serving_sinr_db)
    target_sinrs = np.asarray(ue.sinr_db, dtype=float).copy()
    target_sinrs[int(ue.serving_idx)] = -np.inf
    best_target_sinr = float(np.max(target_sinrs))
    if serving_sinr < env.cfg.system.q_in_db + sinr_margin_db and best_target_sinr - serving_sinr > target_gap_db:
        return 3  # aggressive
    return 2  # nominal


def valid_handover_mode_actions(env: D2ThresholdEnv) -> np.ndarray:
    """Safety shield for semantic handover-mode actions.

    The DQN still chooses among modes, but physically unsafe actions are masked:
    do not hold a degrading radio link, and do not trigger aggressive HO while
    ToS is still short unless the link is already near outage.
    """
    ue = env.sim.ue
    if ue.ml_m is None or ue.sinr_db is None:
        env._refresh_measurements()
    current_tos_s = max(0.0, env.sim.time_s - ue.serving_start_time_s)
    serving_sinr = float(ue.serving_sinr_db)
    target_sinrs = np.asarray(ue.sinr_db, dtype=float).copy()
    target_sinrs[int(ue.serving_idx)] = -np.inf
    best_target_sinr = float(np.max(target_sinrs))
    target_gap = best_target_sinr - serving_sinr

    if serving_sinr < env.cfg.system.q_out_db + 1.0:
        return np.asarray([3] if target_gap > 0.5 else [2, 3], dtype=np.int64)
    if serving_sinr < env.cfg.system.q_in_db + 1.0 and target_gap > 1.0:
        return np.asarray([2, 3], dtype=np.int64)
    if current_tos_s < env.cfg.system.min_tos_s:
        return np.asarray([0, 1, 2], dtype=np.int64)
    return np.asarray([1, 2, 3], dtype=np.int64)


def compute_handover_mode_step_reward(
    cfg: config.LeohosimConfig,
    info: Dict[str, float],
    action_idx: int | None = None,
) -> float:
    """Dense step reward for mode-DQN.

    It emphasizes the paper objective directly: keep SINR above outage, avoid
    RLF/UHO/HOPP, and avoid unnecessary short-ToS handovers.
    """
    sinr = float(info.get("serving_sinr_db", cfg.system.q_out_db))
    best_sinr = float(info.get("best_sinr_db", sinr))
    sinr_span = max(cfg.reward.episode_norm_sinr_max_db - cfg.reward.episode_norm_sinr_min_db, 1e-9)
    sinr_term = 2.5 * float(np.clip((sinr - cfg.reward.episode_norm_sinr_min_db) / sinr_span, 0.0, 1.0))
    recovery_term = 0.2 * float(np.clip((best_sinr - sinr + 5.0) / 10.0, 0.0, 1.0))
    outage_penalty = 5.0 * float(sinr < cfg.system.q_out_db)
    weak_link_penalty = 1.0 * float(sinr < cfg.system.q_in_db)
    rlf_penalty = 20.0 * float(bool(info.get("rlf_event", False)))
    uho_penalty = 8.0 * float(bool(info.get("uho_event", False)))
    hopp_penalty = 4.0 * float(bool(info.get("hopp_event", False)))
    ho_penalty = 0.05 * float(bool(info.get("ho_event", False)))
    prev_tos = float(info.get("prev_tos_s", -1.0))
    short_tos_penalty = 2.0 * float(bool(info.get("ho_event", False)) and 0.0 <= prev_tos < cfg.system.min_tos_s)
    unsafe_hold_penalty = 0.0
    if action_idx in (0, 1) and sinr < cfg.system.q_in_db and best_sinr - sinr > 1.0:
        unsafe_hold_penalty = 3.0
    return float(
        sinr_term
        + recovery_term
        - outage_penalty
        - weak_link_penalty
        - rlf_penalty
        - uho_penalty
        - hopp_penalty
        - ho_penalty
        - short_tos_penalty
        - unsafe_hold_penalty
    )


def compute_episode_kpi_reward_breakdown(cfg: config.LeohosimConfig, k) -> Dict[str, float]:
    """KPI-level terminal reward for simulation-based threshold optimization."""
    w = cfg.reward
    sinr_span = max(w.episode_norm_sinr_max_db - w.episode_norm_sinr_min_db, 1e-9)
    sinr_norm = float(np.clip((k.avg_sinr_db - w.episode_norm_sinr_min_db) / sinr_span, 0.0, 1.0))
    tos_norm = float(np.clip(k.avg_tos_s / max(w.episode_norm_tos_s, 1e-9), 0.0, 1.0))
    rlf_norm = float(np.clip(k.rlf_count / max(w.episode_norm_rlf_count, 1e-9), 0.0, 1.0))
    uho_norm = float(np.clip(k.uho_count / max(w.episode_norm_uho_count, 1e-9), 0.0, 1.0))
    hopp_norm = float(np.clip(k.hopp_count / max(w.episode_norm_hopp_count, 1e-9), 0.0, 1.0))
    short_tos_norm = float(np.clip(k.short_tos_count / max(w.episode_norm_short_tos_count, 1e-9), 0.0, 1.0))
    ho_norm = float(np.clip(k.ho_count / max(w.episode_norm_ho_count, 1e-9), 0.0, 1.0))
    rb_norm = float(np.clip(k.rb_count / max(w.episode_norm_rb_count, 1e-9), 0.0, 1.0))
    parts = {
        "sinr_term": w.episode_w_sinr * sinr_norm,
        "outage_term": -w.episode_w_outage * float(np.clip(k.outage_fraction, 0.0, 1.0)),
        "rlf_term": -w.episode_w_rlf * rlf_norm,
        "uho_term": -w.episode_w_uho * uho_norm,
        "hopp_term": -w.episode_w_hopp * hopp_norm,
        "short_tos_term": -w.episode_w_short_tos * short_tos_norm,
        "ho_term": -w.episode_w_ho * ho_norm,
        "rb_term": -w.episode_w_rb * rb_norm,
        "avg_tos_term": w.episode_w_avg_tos * tos_norm,
    }
    parts["total"] = float(sum(float(v) for v in parts.values()))
    return {k: float(v) for k, v in parts.items()}


def reward_config_summary(cfg: config.LeohosimConfig) -> Dict[str, object]:
    w = cfg.reward
    return {
        "weights": {
            "sinr": w.episode_w_sinr,
            "outage": w.episode_w_outage,
            "rlf": w.episode_w_rlf,
            "uho": w.episode_w_uho,
            "hopp": w.episode_w_hopp,
            "short_tos": w.episode_w_short_tos,
            "ho": w.episode_w_ho,
            "rb": w.episode_w_rb,
            "avg_tos": w.episode_w_avg_tos,
        },
        "normalizers": {
            "sinr_min_db": w.episode_norm_sinr_min_db,
            "sinr_max_db": w.episode_norm_sinr_max_db,
            "rlf_count": w.episode_norm_rlf_count,
            "uho_count": w.episode_norm_uho_count,
            "hopp_count": w.episode_norm_hopp_count,
            "short_tos_count": w.episode_norm_short_tos_count,
            "ho_count": w.episode_norm_ho_count,
            "rb_count": w.episode_norm_rb_count,
            "avg_tos_s": w.episode_norm_tos_s,
            "outage_fraction": 1.0,
        },
    }


def compute_event_decision_reward(
    cfg: config.LeohosimConfig,
    infos: List[Dict[str, float]],
    step_rewards: List[float],
    window_s: float,
    baseline_infos: List[Dict[str, float]] | None = None,
    baseline_step_rewards: List[float] | None = None,
) -> float:
    """Outcome reward for one event-level threshold decision.

    Unlike the one-second window reward, this is deliberately dominated by HO
    stability terms. A threshold choice only gets a strong positive score when
    it preserves radio quality without causing UHO/HOPP/RLF around that specific
    candidate event.
    """
    if not infos:
        return 0.0

    action_score = _event_outcome_score(cfg, infos, step_rewards, window_s)
    if baseline_infos is None:
        return action_score
    baseline_score = _event_outcome_score(
        cfg,
        baseline_infos,
        baseline_step_rewards or [],
        max(len(baseline_infos) * cfg.simulation.sample_time_s, cfg.simulation.sample_time_s),
    )
    return float(action_score - baseline_score)


def _event_outcome_score(
    cfg: config.LeohosimConfig,
    infos: List[Dict[str, float]],
    step_rewards: List[float],
    window_s: float,
) -> float:
    """Absolute local outcome score used before baseline-relative differencing."""
    if not infos:
        return 0.0

    w = cfg.reward
    sinrs = np.array([float(i.get("serving_sinr_db", -20.0)) for i in infos], dtype=float)
    best_sinrs = np.array([float(i.get("best_sinr_db", i.get("serving_sinr_db", -20.0))) for i in infos], dtype=float)
    mean_sinr = float(np.mean(sinrs))
    final_sinr = float(sinrs[-1])
    initial_sinr = float(sinrs[0])
    sinr_span = max(w.episode_norm_sinr_max_db - w.episode_norm_sinr_min_db, 1e-9)
    sinr_norm = float(np.clip((mean_sinr - w.episode_norm_sinr_min_db) / sinr_span, 0.0, 1.0))
    sinr_delta_norm = float(np.clip((final_sinr - initial_sinr + 10.0) / 20.0, 0.0, 1.0))
    best_gap_norm = float(np.clip((float(np.mean(best_sinrs - sinrs)) + 10.0) / 20.0, 0.0, 1.0))

    ho_count = float(sum(bool(i.get("ho_event", False)) for i in infos))
    uho_count = float(sum(bool(i.get("uho_event", False)) for i in infos))
    rlf_count = float(sum(bool(i.get("rlf_event", False)) for i in infos))
    hopp_count = float(sum(bool(i.get("hopp_event", False)) for i in infos))
    prep_fail_count = float(sum(bool(i.get("prep_failed", False)) for i in infos))
    rb_count = float(sum(float(i.get("rb_delta", 0.0)) for i in infos))
    outage = float(np.mean(sinrs < cfg.system.q_out_db))
    near_outage = float(np.mean(sinrs < cfg.system.q_in_db))

    # No-op decisions are acceptable only when the local link remains healthy.
    no_ho_quality_penalty = 0.0
    if ho_count == 0.0 and final_sinr < cfg.system.q_in_db:
        no_ho_quality_penalty = 1.0

    return float(
        0.35 * w.episode_w_sinr * sinr_norm
        + 0.10 * w.episode_w_sinr * sinr_delta_norm
        - 0.10 * w.episode_w_sinr * best_gap_norm
        - 2.00 * w.episode_w_outage * outage
        - 0.60 * w.episode_w_outage * near_outage
        - 2.00 * w.episode_w_rlf * rlf_count
        - 1.75 * w.episode_w_uho * uho_count
        - 1.75 * w.episode_w_hopp * hopp_count
        - 1.25 * w.episode_w_short_tos * uho_count
        - 0.50 * w.episode_w_ho * ho_count
        - 0.50 * w.episode_w_rb * rb_count
        - 0.75 * w.episode_w_outage * prep_fail_count
        - 0.75 * w.episode_w_outage * no_ho_quality_penalty
    )


def compute_local_window_reward(
    cfg: config.LeohosimConfig,
    infos: List[Dict[str, float]],
    step_rewards: List[float],
    window_s: float,
) -> float:
    """Short-horizon reward for an action over the next local window.

    This gives immediate credit/blame to a threshold pair for what happens soon
    after it is selected: local SINR/outage, RLF, UHO, HOPP, HO, and RB usage.
    Counts are normalized to the local window length, using the episode-level
    normalizers scaled by window_s / total_time_s with a minimum of one event.
    """
    if not infos:
        return 0.0

    w = cfg.reward
    sinrs = np.array([float(i.get("serving_sinr_db", -20.0)) for i in infos], dtype=float)
    ho_count = float(sum(bool(i.get("ho_event", False)) for i in infos))
    uho_count = float(sum(bool(i.get("uho_event", False)) for i in infos))
    rlf_count = float(sum(bool(i.get("rlf_event", False)) for i in infos))
    hopp_count = float(sum(bool(i.get("hopp_event", False)) for i in infos))
    rb_count = float(sum(float(i.get("rb_delta", 0.0)) for i in infos))
    outage = float(np.mean(sinrs < cfg.system.q_out_db))

    sinr_span = max(w.episode_norm_sinr_max_db - w.episode_norm_sinr_min_db, 1e-9)
    sinr_norm = float(np.clip((float(np.mean(sinrs)) - w.episode_norm_sinr_min_db) / sinr_span, 0.0, 1.0))

    scale = max(window_s / max(cfg.simulation.total_time_s, 1e-9), 1e-9)
    ho_norm = float(np.clip(ho_count / max(1.0, w.episode_norm_ho_count * scale), 0.0, 1.0))
    uho_norm = float(np.clip(uho_count / max(1.0, w.episode_norm_uho_count * scale), 0.0, 1.0))
    rlf_norm = float(np.clip(rlf_count / max(1.0, w.episode_norm_rlf_count * scale), 0.0, 1.0))
    hopp_norm = float(np.clip(hopp_count / max(1.0, w.episode_norm_hopp_count * scale), 0.0, 1.0))
    rb_norm = float(np.clip(rb_count / max(1.0, w.episode_norm_rb_count * scale), 0.0, 1.0))

    return float(
        w.episode_w_sinr * sinr_norm
        - w.episode_w_outage * outage
        - w.episode_w_rlf * rlf_norm
        - w.episode_w_uho * uho_norm
        - w.episode_w_hopp * hopp_norm
        - w.episode_w_ho * ho_norm
        - w.episode_w_rb * rb_norm
    )
