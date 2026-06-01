"""Minimal RL environment for adaptive Event D2 threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Dict, Tuple

import numpy as np

from leohosim_py import config, simulator


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class D2ThresholdEnv:
    """Small Gym-like wrapper around :class:`LEOSimulator`.

    Actions are discrete `(Thresh1, Thresh2)` pairs from `cfg.action_grid`.
    The observation is a compact numeric state derived from current serving
    quality, candidate quality, geometry, ToS, counters, and simulation progress.
    """

    state_dim = 40

    def __init__(self, cfg: config.LeohosimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.actions = cfg.action_grid.get_actions()
        if not self.actions:
            raise ValueError("action_grid must contain at least one threshold pair")
        self.sim = simulator.LEOSimulator(cfg, self.rng)

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def reset(self) -> np.ndarray:
        self.sim = simulator.LEOSimulator(self.cfg, self.rng)
        self.sim.reset()
        self._refresh_measurements()
        return self._observe()

    def snapshot(self):
        """Return a restorable simulator snapshot for counterfactual rollouts."""
        return copy.deepcopy(self.sim)

    def restore(self, snapshot) -> np.ndarray:
        """Restore a simulator snapshot and return the corresponding state."""
        self.sim = copy.deepcopy(snapshot)
        self._refresh_measurements()
        return self._observe()

    def step(self, action_idx: int) -> StepResult:
        thresh1_m, thresh2_m = self.actions[int(action_idx)]
        return self.step_thresholds(thresh1_m, thresh2_m, action_idx=int(action_idx))

    def step_thresholds(self, thresh1_m: float, thresh2_m: float, action_idx: int = -1) -> StepResult:
        info, reward = self.sim.step(thresh1_m, thresh2_m, action_idx=int(action_idx))
        return StepResult(
            state=self._observe(),
            reward=float(reward),
            done=self.sim.is_done(),
            info=info,
        )

    def threshold_mode_pairs(self, hold_margin_m: float = 1000.0) -> list[tuple[float, float]]:
        """Compact handover-control modes for DQN.

        The dense threshold grid is hard to identify locally. These modes make
        the action semantic and robust: suppress HO, make HO hard, use nominal
        3GPP, or make HO easy.
        """
        t1_values = [float(th1) for th1, _ in self.actions]
        t2_values = [float(th2) for _, th2 in self.actions]
        min_t1, max_t1 = min(t1_values), max(t1_values)
        min_t2, max_t2 = min(t2_values), max(t2_values)
        nominal = (float(self.cfg.system.cell_radius_m), float(self.cfg.system.cell_radius_m))
        return [
            (max_t1 + float(hold_margin_m), min_t2 - float(hold_margin_m)),
            (max_t1, min_t2),
            nominal,
            (min_t1, max_t2),
        ]

    def mode_action_names(self) -> list[str]:
        return ["hold", "conservative", "nominal", "aggressive"]

    def valid_action_indices(
        self,
        guard_short_tos: bool = False,
        guard_tos_s: float | None = None,
        min_thresh1_m: float | None = None,
        max_thresh2_m: float | None = None,
    ) -> np.ndarray:
        """Return actions allowed by optional HO-stability guardrails.

        When current ToS is short, aggressive D2 thresholds can cause UHO or
        ping-pong before the episode reward can assign clean credit. The guard
        keeps only conservative actions in that state: higher Thresh1 and lower
        Thresh2, which makes Event D2 harder to trigger.
        """
        if not guard_short_tos:
            return np.arange(len(self.actions), dtype=np.int64)
        ue = self.sim.ue
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        guard_tos_s = self.cfg.system.min_tos_s if guard_tos_s is None else guard_tos_s
        if current_tos_s >= guard_tos_s:
            return np.arange(len(self.actions), dtype=np.int64)
        min_thresh1_m = self.cfg.system.cell_radius_m if min_thresh1_m is None else min_thresh1_m
        max_thresh2_m = self.cfg.system.cell_radius_m if max_thresh2_m is None else max_thresh2_m
        valid = [
            idx
            for idx, (thresh1_m, thresh2_m) in enumerate(self.actions)
            if thresh1_m >= min_thresh1_m and thresh2_m <= max_thresh2_m
        ]
        return np.asarray(valid if valid else np.arange(len(self.actions)), dtype=np.int64)

    def conservative_action_index(self) -> int:
        """Action that makes Event D2 hardest to trigger in the current grid."""
        scores = [(float(thresh1_m), -float(thresh2_m), idx) for idx, (thresh1_m, thresh2_m) in enumerate(self.actions)]
        return int(max(scores)[2])

    def baseline_action_index(self, thresh1_m: float | None = None, thresh2_m: float | None = None) -> int:
        """Closest action to the nominal fixed 3GPP D2 threshold pair."""
        target1 = self.cfg.system.cell_radius_m if thresh1_m is None else float(thresh1_m)
        target2 = self.cfg.system.cell_radius_m if thresh2_m is None else float(thresh2_m)
        scores = [
            ((float(th1) - target1) ** 2 + (float(th2) - target2) ** 2, idx)
            for idx, (th1, th2) in enumerate(self.actions)
        ]
        return int(min(scores)[1])

    def has_ho_opportunity(self, min_serving_m: float | None = None, max_target_m: float | None = None) -> bool:
        """Broad, threshold-independent HO candidate gate.

        The actual D2 decision still uses the selected action thresholds. This
        gate only decides whether the agent should be queried at this time
        instant. It fires when the serving link is near the action grid's
        possible Thresh1 region and at least one non-serving cell is inside the
        grid's possible Thresh2 region.
        """
        ue = self.sim.ue
        if ue.ml_m is None:
            self._refresh_measurements()
        ml_m = ue.ml_m
        min_serving_m = min(thresh1_m for thresh1_m, _ in self.actions) if min_serving_m is None else min_serving_m
        max_target_m = max(thresh2_m for _, thresh2_m in self.actions) if max_target_m is None else max_target_m
        serving_ready = float(ml_m[ue.serving_idx]) >= float(min_serving_m)
        candidates = np.asarray(ml_m, dtype=float) <= float(max_target_m)
        candidates = candidates.copy()
        candidates[ue.serving_idx] = False
        return bool(serving_ready and np.any(candidates))

    def has_baseline_ho_opportunity(self, baseline_action_idx: int | None = None) -> bool:
        """Gate agent decisions at states where the nominal D2 rule is close to firing.

        This is intentionally tighter than :meth:`has_ho_opportunity`: the agent
        is queried around a real fixed-threshold HO candidate, not at every state
        in the broad action-grid range. That keeps the RL step aligned with a
        handover decision opportunity.
        """
        ue = self.sim.ue
        if ue.ml_m is None:
            self._refresh_measurements()
        idx = self.baseline_action_index() if baseline_action_idx is None else int(baseline_action_idx)
        thresh1_m, thresh2_m = self.actions[idx]
        ml_m = np.asarray(ue.ml_m, dtype=float)
        cond1 = float(ml_m[ue.serving_idx]) - self.cfg.system.hys_m > float(thresh1_m)
        candidates = ml_m + self.cfg.system.hys_m < float(thresh2_m)
        candidates = candidates.copy()
        candidates[ue.serving_idx] = False
        return bool(cond1 and np.any(candidates))

    def d2_action_pass_mask(self) -> np.ndarray:
        """Return whether each action would satisfy both D2 trigger conditions now."""
        ue = self.sim.ue
        if ue.ml_m is None:
            self._refresh_measurements()
        ml_m = np.asarray(ue.ml_m, dtype=float)
        pass_mask = []
        for thresh1_m, thresh2_m in self.actions:
            cond1 = float(ml_m[ue.serving_idx]) - self.cfg.system.hys_m > float(thresh1_m)
            candidates = ml_m + self.cfg.system.hys_m < float(thresh2_m)
            candidates = candidates.copy()
            candidates[ue.serving_idx] = False
            pass_mask.append(bool(cond1 and np.any(candidates)))
        return np.asarray(pass_mask, dtype=bool)

    def has_discriminative_ho_opportunity(self) -> bool:
        """True only when some threshold actions trigger D2 and some do not."""
        pass_mask = self.d2_action_pass_mask()
        return bool(np.any(pass_mask) and np.any(~pass_mask))

    def evaluate_action_sensitivity(
        self,
        max_window_s: float,
        baseline_action_idx: int | None = None,
    ) -> Dict[str, Any]:
        """Roll out all actions from the current snapshot and summarize local outcome spread."""
        from leohosim_py.training.dqn_trainer import compute_event_decision_reward

        snapshot = self.snapshot()
        baseline_idx = self.baseline_action_index() if baseline_action_idx is None else int(baseline_action_idx)
        max_window_s = max(float(max_window_s), self.cfg.simulation.sample_time_s)
        _, _, baseline_infos, baseline_rewards = self.rollout_event_decision(baseline_idx, max_window_s)
        self.restore(snapshot)

        rewards: list[float] = []
        ho_counts: list[int] = []
        uho_counts: list[int] = []
        rlf_counts: list[int] = []
        outage_counts: list[int] = []
        for action_idx in range(len(self.actions)):
            self.restore(snapshot)
            _, _, infos, step_rewards = self.rollout_event_decision(action_idx, max_window_s)
            reward = compute_event_decision_reward(
                self.cfg,
                infos,
                step_rewards,
                window_s=len(infos) * self.cfg.simulation.sample_time_s,
                baseline_infos=baseline_infos,
                baseline_step_rewards=baseline_rewards,
            )
            rewards.append(float(reward))
            ho_counts.append(int(sum(bool(i.get("ho_event", False)) for i in infos)))
            uho_counts.append(int(sum(bool(i.get("uho_event", False)) for i in infos)))
            rlf_counts.append(int(sum(bool(i.get("rlf_event", False)) for i in infos)))
            outage_counts.append(int(sum(float(i.get("serving_sinr_db", 0.0)) < self.cfg.system.q_out_db for i in infos)))

        self.restore(snapshot)
        reward_arr = np.asarray(rewards, dtype=float)
        pass_mask = self.d2_action_pass_mask()
        best = float(np.max(reward_arr)) if len(reward_arr) else 0.0
        best_tie_count = int(np.sum(np.isclose(reward_arr, best, rtol=0.0, atol=1e-9)))
        return {
            "local_rewards": rewards,
            "unique_reward_count": int(len(np.unique(np.round(reward_arr, 9)))),
            "reward_std": float(np.std(reward_arr)) if len(reward_arr) else 0.0,
            "best_tie_count": best_tie_count,
            "pass_action_count": int(np.sum(pass_mask)),
            "action_count": int(len(self.actions)),
            "has_discriminative_state": bool(np.any(pass_mask) and np.any(~pass_mask)),
            "ho_counts": ho_counts,
            "uho_counts": uho_counts,
            "rlf_counts": rlf_counts,
            "outage_counts": outage_counts,
        }

    def rollout_event_decision(self, action_idx: int, max_window_s: float) -> tuple[np.ndarray, bool, list[Dict[str, Any]], list[float]]:
        """Apply one selected action until a HO outcome or decision timeout.

        The returned transition is one RL decision: "given this HO-candidate
        state, choosing this threshold pair led to these near-term HO outcomes".
        """
        max_steps = max(1, int(round(max_window_s / self.cfg.simulation.sample_time_s)))
        infos: list[Dict[str, Any]] = []
        rewards: list[float] = []
        for _ in range(max_steps):
            result = self.step(action_idx)
            infos.append(result.info)
            rewards.append(result.reward)
            if result.done:
                return result.state, True, infos, rewards
            if (
                result.info.get("ho_event", False)
                or result.info.get("prep_failed", False)
                or result.info.get("rlf_event", False)
            ):
                return result.state, result.done, infos, rewards
        return self._observe(), self.sim.is_done(), infos, rewards

    def rollout_fixed_to_done(self, action_idx: int) -> tuple[np.ndarray, list[Dict[str, Any]], list[float]]:
        """Run one fixed action until episode end.

        This is used for same-scenario 3GPP counterfactual rewards. The caller
        should snapshot/restore around this method when the main episode must
        continue from the original state.
        """
        infos: list[Dict[str, Any]] = []
        rewards: list[float] = []
        while not self.sim.is_done():
            result = self.step(action_idx)
            infos.append(result.info)
            rewards.append(result.reward)
        return self._observe(), infos, rewards

    def _refresh_measurements(self) -> None:
        ue = self.sim.ue
        rsrp_dbm, sinr_db, ml_m = self.sim.channel_calc.update_channel(
            self.sim.beam_centers[:, 0],
            self.sim.beam_centers[:, 1],
            ue.x_m,
            ue.y_m,
            self.cfg.system.altitude_m,
        )
        ue.rsrp_dbm = rsrp_dbm
        ue.sinr_db = sinr_db
        ue.ml_m = ml_m
        ue.update_serving_metrics()

    def _observe(self) -> np.ndarray:
        ue = self.sim.ue
        if ue.ml_m is None or ue.sinr_db is None:
            self._refresh_measurements()
        ml_m = ue.ml_m
        sinr_db = ue.sinr_db
        rsrp_dbm = ue.rsrp_dbm
        serving_idx = int(ue.serving_idx)
        target_order = [int(idx) for idx in np.argsort(ml_m) if int(idx) != serving_idx]
        best_idx = target_order[0] if target_order else serving_idx
        second_idx = target_order[1] if len(target_order) > 1 else best_idx
        serving_ml = float(ml_m[ue.serving_idx])
        best_ml = float(ml_m[best_idx])
        second_ml = float(ml_m[second_idx])
        best_sinr = float(sinr_db[best_idx])
        best_rsrp = float(rsrp_dbm[best_idx])
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        span = max(radius * 6.0, 1.0)
        serving_center = self.sim.beam_centers[ue.serving_idx]
        best_center = self.sim.beam_centers[best_idx]
        serving_dx = (serving_center[0] - ue.x_m) / radius
        serving_dy = (serving_center[1] - ue.y_m) / radius
        best_dx = (best_center[0] - ue.x_m) / radius
        best_dy = (best_center[1] - ue.y_m) / radius
        total_steps = max(self.sim.total_steps, 1)
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        sinr_gap = best_sinr - ue.serving_sinr_db
        rsrp_gap = best_rsrp - ue.serving_rsrp_dbm
        speed_norm = np.clip(ue.speed_mps / max(self.cfg.simulation.ue_speed_max_mps, 83.333, 1e-9), 0.0, 1.0)
        state = np.array(
            [
                np.clip((ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
                np.clip((best_sinr + 20.0) / 40.0, 0.0, 1.0),
                np.clip((sinr_gap + 20.0) / 40.0, 0.0, 1.0),
                np.clip((ue.serving_rsrp_dbm + 160.0) / 100.0, 0.0, 1.0),
                np.clip((best_rsrp + 160.0) / 100.0, 0.0, 1.0),
                np.clip((rsrp_gap + 30.0) / 60.0, 0.0, 1.0),
                np.clip(serving_ml / radius, 0.0, 5.0) / 5.0,
                np.clip(best_ml / radius, 0.0, 5.0) / 5.0,
                np.clip(second_ml / radius, 0.0, 5.0) / 5.0,
                np.clip((serving_ml - best_ml) / radius, -2.0, 2.0) / 4.0 + 0.5,
                np.clip((second_ml - best_ml) / radius, 0.0, 2.0) / 2.0,
                np.clip(ue.x_m / span, -1.0, 1.0),
                np.clip(ue.y_m / span, -1.0, 1.0),
                np.clip(serving_dx, -3.0, 3.0) / 6.0 + 0.5,
                np.clip(serving_dy, -3.0, 3.0) / 6.0 + 0.5,
                np.clip(best_dx, -3.0, 3.0) / 6.0 + 0.5,
                np.clip(best_dy, -3.0, 3.0) / 6.0 + 0.5,
                self.sim.step_idx / total_steps,
                np.clip(current_tos_s / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
                np.clip(ue.ho_count / 20.0, 0.0, 1.0),
                np.clip(ue.rlf_event_count / 20.0, 0.0, 1.0),
                np.clip(ue.rb_count / 200.0, 0.0, 1.0),
                speed_norm,
                0.5 + 0.5 * float(np.cos(ue.heading_rad)),
                0.5 + 0.5 * float(np.sin(ue.heading_rad)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([state, self._lookahead_features(serving_idx)]).astype(np.float32)

    def _lookahead_features(self, serving_idx: int) -> np.ndarray:
        ue = self.sim.ue
        horizons = list(self.cfg.simulation.lookahead_horizons_s)
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        features: list[float] = []
        for horizon_s in horizons[:3]:
            horizon_s = float(horizon_s)
            beam_centers = np.array(self.sim.beam_centers, copy=True)
            beam_centers[:, 1] += self.cfg.system.sat_speed_mps * horizon_s
            if self.cfg.simulation.ue_mobility_mode == "linear":
                ux = ue.x_m + ue.speed_mps * np.cos(ue.heading_rad) * horizon_s
                uy = ue.y_m + ue.speed_mps * np.sin(ue.heading_rad) * horizon_s
            else:
                ux, uy = ue.x_m, ue.y_m
            _, sinr_db, ml_m = self.sim.channel_calc.update_channel(
                beam_centers[:, 0],
                beam_centers[:, 1],
                ux,
                uy,
                self.cfg.system.altitude_m,
            )
            target_order = [int(idx) for idx in np.argsort(ml_m) if int(idx) != serving_idx]
            best_idx = target_order[0] if target_order else serving_idx
            serving_sinr = float(sinr_db[serving_idx])
            best_sinr = float(sinr_db[best_idx])
            serving_ml = float(ml_m[serving_idx])
            best_ml = float(ml_m[best_idx])
            features.extend(
                [
                    float(np.clip((serving_sinr + 20.0) / 40.0, 0.0, 1.0)),
                    float(np.clip((best_sinr + 20.0) / 40.0, 0.0, 1.0)),
                    float(np.clip((best_sinr - serving_sinr + 20.0) / 40.0, 0.0, 1.0)),
                    float(np.clip(serving_ml / radius, 0.0, 5.0) / 5.0),
                    float(np.clip(best_ml / radius, 0.0, 5.0) / 5.0),
                ]
            )
        while len(features) < 15:
            features.append(0.0)
        return np.asarray(features[:15], dtype=np.float32)

    def get_kpi(self):
        return self.sim.get_kpi()

    def get_history_dicts(self):
        return self.sim.get_history_dicts()
