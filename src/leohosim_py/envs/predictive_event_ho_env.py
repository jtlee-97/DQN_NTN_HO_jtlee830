"""Event-level predictive handover environment.

The agent is called only at identifiable handover-risk states. Its reward is a
paired local counterfactual: short-horizon KPI after the selected action minus
the KPI obtained by the A3-RSRP baseline from the same snapshot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from leohosim_py import config, history, simulator


@dataclass
class EventHOStep:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class LocalRolloutKPI:
    score: float
    avg_sinr_db: float
    outage_fraction: float
    weak_fraction: float
    rlf_count: int
    uho_count: int
    ho_count: int
    rb_count: int
    short_tos_count: int
    tos_sum_delta: float
    components: Dict[str, float]

    @property
    def avg_tos_s(self) -> float:
        """Compatibility alias for older evaluation code."""
        return self.tos_sum_delta


class PredictiveEventHandoverEnv:
    """Decision-event DQN for hold/current/future target handover.

    Actions:
    0. hold: no DQN handover during the decision window
    1. choose best stable target in the 0.0-0.5 s HO window
    2. choose best stable target in the 0.5-1.0 s HO window
    3. choose best stable target in the 1.0-1.5 s HO window
    4. choose best stable target in the 1.5-2.0 s HO window
    5. choose best stable target in the 2.0-3.0 s HO window
    6. choose best next-stable target in 0.5-3.0 s while skipping transient nearest cells
    """

    state_dim = 44
    action_dim = 7

    def __init__(
        self,
        cfg: config.LeohosimConfig,
        rng: np.random.Generator | None = None,
        decision_window_s: float = 3.0,
        max_skip_s: float = 8.0,
    ):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.decision_window_s = float(decision_window_s)
        self.max_skip_s = float(max_skip_s)
        self.sim = simulator.LEOSimulator(cfg, self.rng)
        self.decision_count = 0
        self.last_info: Dict[str, Any] = {}
        self.last_guard_info: Dict[str, Any] = {}

    def reset(self) -> np.ndarray:
        self.sim = simulator.LEOSimulator(self.cfg, self.rng)
        self.sim.reset()
        self.decision_count = 0
        self.last_info = {}
        self.last_guard_info = {}
        self._advance_to_decision()
        return self._observe()

    def step(self, action_idx: int) -> EventHOStep:
        action_idx = int(action_idx)
        valid = self.valid_action_indices()
        if action_idx not in set(valid.tolist()):
            action_idx = int(valid[0])

        snapshot = copy.deepcopy(self.sim)
        baseline_kpi = self._rollout_baseline(copy.deepcopy(snapshot), self.decision_window_s)
        action_kpi = self._rollout_action(copy.deepcopy(snapshot), action_idx, self.decision_window_s)
        reward = float(action_kpi.score - baseline_kpi.score)

        before = self._counter_snapshot(self.sim)
        self._execute_action_window(self.sim, action_idx, self.decision_window_s)
        after = self._counter_snapshot(self.sim)
        guard_info = dict(self.last_guard_info)
        self.decision_count += 1
        self._advance_to_decision()

        info = {
            "action_idx": action_idx,
            "baseline_score": baseline_kpi.score,
            "action_score": action_kpi.score,
            "relative_reward": reward,
            "decision_count": self.decision_count,
            "event_ho_count": after["ho_count"] - before["ho_count"],
            "event_rlf_count": after["rlf_count"] - before["rlf_count"],
            "event_uho_count": after["uho_count"] - before["uho_count"],
            "event_rb_count": after["rb_count"] - before["rb_count"],
            "action_avg_sinr_db": action_kpi.avg_sinr_db,
            "baseline_avg_sinr_db": baseline_kpi.avg_sinr_db,
            "action_outage_fraction": action_kpi.outage_fraction,
            "baseline_outage_fraction": baseline_kpi.outage_fraction,
        }
        info.update(guard_info)
        for key, value in action_kpi.components.items():
            info[f"reward_component_delta_{key}"] = float(value - baseline_kpi.components.get(key, 0.0))
        self.last_info = info
        return EventHOStep(self._observe(), reward, self.sim.is_done(), info)

    def valid_action_indices(self) -> np.ndarray:
        ue = self.sim.ue
        assert ue is not None
        self._refresh_measurements(self.sim)
        tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        serving_idx = int(ue.serving_idx)
        required_wait_s = max(0.0, self.cfg.system.min_tos_s - tos_s)
        valid = [0]
        for action_idx in range(1, self.action_dim):
            window_start, window_end = self._action_window_s(action_idx)
            if window_end + 1e-9 < required_wait_s:
                continue
            target_idx = self._action_target(self.sim, action_idx)
            if target_idx is None or int(target_idx) == serving_idx:
                continue
            valid.append(action_idx)
        return np.asarray(valid if len(valid) > 1 else [0], dtype=np.int64)

    def teacher_action(self) -> int:
        valid = self.valid_action_indices()
        snapshot = copy.deepcopy(self.sim)
        high_risk = self._serving_risk(self.sim) or self._future_serving_risk(self.sim, self.decision_window_s)
        scored: list[tuple[int, LocalRolloutKPI]] = []
        for action in valid:
            k = self._rollout_action(copy.deepcopy(snapshot), int(action), self.decision_window_s)
            scored.append((int(action), k))
        if not scored:
            return 0

        hold_kpi = next((k for action, k in scored if action == 0), None)
        candidates = scored
        if high_risk and hold_kpi is not None:
            # When the serving link is currently or predictively risky, the
            # teacher should not demonstrate passive hold if a scheduled HO can
            # avoid equal-or-worse RLF/outage. This prevents early imitation
            # from collapsing into a low-HO, high-RLF policy.
            safer_actions = [
                (action, k)
                for action, k in scored
                if action != 0
                and k.rlf_count <= hold_kpi.rlf_count
                and k.outage_fraction <= hold_kpi.outage_fraction + 1e-9
            ]
            if safer_actions:
                candidates = safer_actions

        def teacher_key(item: tuple[int, LocalRolloutKPI]) -> tuple[float, ...]:
            action, k = item
            outage_time_s = k.outage_fraction * self.decision_window_s
            future_action_bonus = 0.05 if action >= 2 else 0.0
            return (
                -float(k.rlf_count),
                -float(k.uho_count),
                -float(outage_time_s),
                float(k.avg_sinr_db),
                float(k.score),
                future_action_bonus,
                -float(k.ho_count),
                -float(k.rb_count),
            )

        return int(max(candidates, key=teacher_key)[0])

    def is_decision_state(self) -> bool:
        if self.sim.is_done():
            return True
        ue = self.sim.ue
        assert ue is not None
        self._refresh_measurements(self.sim)
        serving_sinr = float(ue.serving_sinr_db)
        rlf_risk = float(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9))
        future_risk = False
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            f_serving = self._future_distance_to_cell(self.sim, int(ue.serving_idx), float(horizon_s))
            f_nearest = self._future_distance_to_cell(
                self.sim,
                self._future_nearest_nonserving_by_distance(self.sim, float(horizon_s)),
                float(horizon_s),
            )
            if f_serving > f_nearest + self.cfg.future_distance_margin_m:
                future_risk = True
                break
        return (
            serving_sinr < self.cfg.system.q_in_db + 1.0
            or self._distance_to_cell(self.sim, int(ue.serving_idx)) > self._distance_to_cell(
                self.sim, self._nearest_nonserving_by_distance(self.sim)
            )
            or rlf_risk > 0.0
            or future_risk
        )

    def _advance_to_decision(self) -> None:
        max_steps = max(1, int(round(self.max_skip_s / self.cfg.simulation.sample_time_s)))
        steps = 0
        while not self.sim.is_done() and not self.is_decision_state() and steps < max_steps:
            self._direct_step(self.sim, None, log_action_idx=-1)
            steps += 1

    def _rollout_baseline(self, sim: simulator.LEOSimulator, horizon_s: float) -> LocalRolloutKPI:
        before = self._counter_snapshot(sim)
        rewards = []
        sinrs = []
        outages = []
        weak = []
        steps = max(1, int(round(horizon_s / self.cfg.simulation.sample_time_s)))
        for _ in range(steps):
            if sim.is_done():
                break
            _, reward = sim.step(self.cfg.system.cell_radius_m, self.cfg.system.cell_radius_m, action_idx=0)
            ue = sim.ue
            assert ue is not None
            rewards.append(float(reward))
            sinrs.append(float(ue.serving_sinr_db))
            outages.append(float(ue.serving_sinr_db < self.cfg.system.q_out_db))
            weak.append(float(ue.serving_sinr_db < self.cfg.system.q_in_db))
        after = self._counter_snapshot(sim)
        return self._local_kpi(before, after, sinrs, outages, weak)

    def _rollout_action(self, sim: simulator.LEOSimulator, action_idx: int, horizon_s: float) -> LocalRolloutKPI:
        before = self._counter_snapshot(sim)
        rewards = []
        sinrs = []
        outages = []
        weak = []
        steps = max(1, int(round(horizon_s / self.cfg.simulation.sample_time_s)))
        plan_delay_s, planned_target = self._strategy_plan(sim, action_idx)
        execute_step = self._delay_to_step(plan_delay_s, steps)
        executed = False
        for step_idx in range(steps):
            if sim.is_done():
                break
            if planned_target is not None and action_idx > 0 and not executed and (step_idx >= execute_step or self._serving_risk(sim)):
                executed = self._apply_direct_action(sim, action_idx, log_action_idx=action_idx, target_idx_override=planned_target)
            reward = self._direct_step(sim, None, log_action_idx=action_idx)
            ue = sim.ue
            assert ue is not None
            rewards.append(float(reward))
            sinrs.append(float(ue.serving_sinr_db))
            outages.append(float(ue.serving_sinr_db < self.cfg.system.q_out_db))
            weak.append(float(ue.serving_sinr_db < self.cfg.system.q_in_db))
        after = self._counter_snapshot(sim)
        return self._local_kpi(before, after, sinrs, outages, weak)

    def _execute_action_window(self, sim: simulator.LEOSimulator, action_idx: int, horizon_s: float) -> None:
        self.last_guard_info = self._empty_guard_info(action_idx, None)
        steps = max(1, int(round(horizon_s / self.cfg.simulation.sample_time_s)))
        plan_delay_s, planned_target = self._strategy_plan(sim, action_idx)
        execute_step = self._delay_to_step(plan_delay_s, steps)
        executed = False
        for step_idx in range(steps):
            if sim.is_done():
                break
            if planned_target is not None and action_idx > 0 and not executed and (step_idx >= execute_step or self._serving_risk(sim)):
                executed = self._apply_direct_action(sim, action_idx, log_action_idx=action_idx, target_idx_override=planned_target)
            self._direct_step(sim, None, log_action_idx=action_idx)

    def _apply_direct_action(
        self,
        sim: simulator.LEOSimulator,
        action_idx: int,
        log_action_idx: int,
        target_idx_override: int | None = None,
    ) -> bool:
        target_idx = self._action_target(sim, action_idx) if target_idx_override is None else int(target_idx_override)
        self.last_guard_info = self._empty_guard_info(action_idx, target_idx)
        if target_idx is None:
            return False
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        if int(target_idx) == int(ue.serving_idx):
            return False
        allowed, reason, details = self._guard_decision(sim, action_idx, int(target_idx))
        self.last_guard_info.update(details)
        self.last_guard_info["dqn_guard_executed"] = int(allowed)
        self.last_guard_info["dqn_guard_reason"] = reason
        if self.cfg.future_action_mode == "guarded" and not allowed:
            return False
        prev_tos_s = sim.time_s - ue.serving_start_time_s
        ue.change_serving_cell(int(target_idx), sim.time_s)
        ue.ho_count += 1
        if 0.0 <= prev_tos_s < self.cfg.system.min_tos_s:
            ue.uho_count += 1
        direct_ho_prep_rb = 3
        direct_ho_exec_rb = 7
        direct_ho_rb = direct_ho_prep_rb + direct_ho_exec_rb
        # For fair comparison, proactive direct HO is charged the same RB cost
        # as a full 3GPP preparation+execution HO.
        ue.rb_count += direct_ho_rb
        is_uho = 0.0 <= prev_tos_s < self.cfg.system.min_tos_s
        hopp_event = sim.hopp_detector.record_handover(ue.serving_idx, sim.time_s)
        setattr(
            sim,
            "_pending_direct_ho",
            {
                "ho_event": True,
                "prev_tos_s": float(prev_tos_s),
                "rb_delta": int(direct_ho_rb),
                "is_uho": bool(is_uho),
                "hopp_event": bool(hopp_event),
                "log_action_idx": int(log_action_idx),
            },
        )
        return True

    def _direct_step(self, sim: simulator.LEOSimulator, target_idx: int | None, log_action_idx: int) -> float:
        if target_idx is not None:
            self._apply_direct_action(sim, 1, log_action_idx, target_idx_override=int(target_idx))
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        rlf_event = sim.rlf_detector.update_rlf(ue, ue.serving_sinr_db, self.cfg.simulation.sample_time_s, sim.time_s)
        if rlf_event:
            ue.rlf_event_count += 1
        pending = getattr(sim, "_pending_direct_ho", None) or {}
        ho_event = bool(pending.get("ho_event", False))
        prev_tos_s = float(pending.get("prev_tos_s", -1.0))
        rb_delta = int(pending.get("rb_delta", 0))
        is_uho = bool(pending.get("is_uho", False))
        reward = self._dense_score(float(ue.serving_sinr_db), bool(rlf_event), is_uho, ho_event, rb_delta)
        self._log_current(
            sim,
            int(pending.get("log_action_idx", log_action_idx)),
            ho_event,
            prev_tos_s,
            rb_delta,
            rlf_event=bool(rlf_event),
            reward=reward,
        )
        if hasattr(sim, "_pending_direct_ho"):
            delattr(sim, "_pending_direct_ho")
        sim.beam_centers[:, 1] += self.cfg.system.sat_speed_mps * self.cfg.simulation.sample_time_s
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ue.move_linear(self.cfg.simulation.sample_time_s)
        sim.time_s += self.cfg.simulation.sample_time_s
        sim.step_idx += 1
        return reward

    def _serving_risk(self, sim: simulator.LEOSimulator) -> bool:
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        return bool(ue.serving_sinr_db < self.cfg.system.q_in_db + 0.5 or ue.rlf.timer_s > 0.0)

    def _future_serving_risk(self, sim: simulator.LEOSimulator, horizon_s: float) -> bool:
        ue = sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        step_s = max(self.cfg.simulation.sample_time_s, 0.2)
        times = np.arange(step_s, max(float(horizon_s), step_s) + 1e-9, step_s)
        for h in times:
            _, f_sinr, f_ml = self._future_measurements(sim, float(h))
            serving_sinr = float(f_sinr[serving_idx])
            values = np.asarray(f_ml, dtype=float).copy()
            values[serving_idx] = np.inf
            nearest_distance = float(np.min(values)) if np.isfinite(values).any() else float("inf")
            serving_distance = float(f_ml[serving_idx])
            if serving_sinr < self.cfg.system.q_in_db + 0.5:
                return True
            if serving_distance > nearest_distance + self.cfg.future_distance_margin_m:
                return True
        return False

    def _dense_score(self, sinr_db: float, rlf: bool, uho: bool, ho: bool, rb_delta: int) -> float:
        sinr_norm = float(np.clip((sinr_db + 20.0) / 30.0, -2.0, 2.0))
        outage = sinr_db < self.cfg.system.q_out_db
        weak = sinr_db < self.cfg.system.q_in_db
        return float(
            5.0 * sinr_norm
            - 8.0 * float(outage)
            - 2.0 * float(weak)
            - 60.0 * float(rlf)
            - 50.0 * float(uho)
            - 0.8 * float(ho)
            - 0.03 * float(rb_delta)
        )

    def _local_kpi(
        self,
        before: Dict[str, int],
        after: Dict[str, int],
        sinrs: list[float],
        outages: list[float],
        weak: list[float],
    ) -> LocalRolloutKPI:
        ho_count = after["ho_count"] - before["ho_count"]
        rlf_count = after["rlf_count"] - before["rlf_count"]
        uho_count = after["uho_count"] - before["uho_count"]
        rb_count = after["rb_count"] - before["rb_count"]
        short_tos_count = after["short_tos_count"] - before["short_tos_count"]
        tos_sum_delta = after["tos_sum"] - before["tos_sum"]
        avg_sinr = float(np.mean(sinrs)) if sinrs else -200.0
        outage_fraction = float(np.mean(outages)) if outages else 0.0
        weak_fraction = float(np.mean(weak)) if weak else 0.0
        sinr_min = float(self.cfg.reward.episode_norm_sinr_min_db)
        sinr_max = float(self.cfg.reward.episode_norm_sinr_max_db)
        sinr_norm = float(np.clip((avg_sinr - sinr_min) / max(sinr_max - sinr_min, 1e-9), 0.0, 1.0))
        outage_time_s = outage_fraction * self.decision_window_s
        components = {
            # Primary objective: keep average DL SINR high during the local
            # decision window, while treating RLF/UHO as hard failures.
            "sinr": 24.0 * sinr_norm,
            "sinr_db": 1.50 * avg_sinr,
            "weak_time": -5.0 * weak_fraction * self.decision_window_s,
            "outage_time": -12.0 * outage_time_s,
            "rlf": -150.0 * rlf_count,
            "uho": -120.0 * uho_count,
            "short_tos": -8.0 * short_tos_count,
            # HO/RB remain costs, but they should not dominate reliability.
            "ho": -0.25 * ho_count,
            "rb": -0.02 * rb_count,
            "tos": 0.10 * tos_sum_delta,
        }
        score = float(sum(components.values()))
        return LocalRolloutKPI(
            score,
            avg_sinr,
            outage_fraction,
            weak_fraction,
            rlf_count,
            uho_count,
            ho_count,
            rb_count,
            short_tos_count,
            tos_sum_delta,
            components,
        )

    def _counter_snapshot(self, sim: simulator.LEOSimulator) -> Dict[str, int]:
        ue = sim.ue
        assert ue is not None
        return {
            "ho_count": int(ue.ho_count),
            "rlf_count": int(ue.rlf_event_count),
            "uho_count": int(ue.uho_count),
            "rb_count": int(ue.rb_count),
            "short_tos_count": int(sum(1 for tos in ue.tos_list if 0.0 <= tos < self.cfg.system.min_tos_s)),
            "tos_sum": float(sum(ue.tos_list)),
        }

    def _action_target(self, sim: simulator.LEOSimulator, action_idx: int) -> int | None:
        if action_idx == 0:
            return None
        self._refresh_measurements(sim)
        _, target = self._strategy_plan(sim, action_idx)
        return target

    def _scheduled_action_target(self, sim: simulator.LEOSimulator, action_idx: int) -> int | None:
        if action_idx == 0:
            return None
        # Keep the originally planned target semantics: the decision event
        # chooses both when to hand over and which stable future cell to use.
        return self._action_target(sim, action_idx)

    def _best_nonserving_index(self, sim: simulator.LEOSimulator, ml_m: np.ndarray) -> int:
        serving_idx = int(sim.ue.serving_idx)
        order = [int(idx) for idx in np.argsort(ml_m) if int(idx) != serving_idx]
        return order[0] if order else serving_idx

    def _best_nonserving_by_sinr(self, serving_idx: int, sinr_db: np.ndarray) -> int:
        values = np.asarray(sinr_db, dtype=float).copy()
        if values.size == 0:
            return int(serving_idx)
        values[int(serving_idx)] = -np.inf
        if not np.isfinite(values).any():
            return int(serving_idx)
        return int(np.argmax(values))

    def _distance_to_cell(self, sim: simulator.LEOSimulator, cell_idx: int) -> float:
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        ml_m = np.asarray(ue.ml_m, dtype=float)
        if cell_idx < 0 or cell_idx >= len(ml_m):
            return float("inf")
        return float(ml_m[int(cell_idx)])

    def _future_distance_to_cell(self, sim: simulator.LEOSimulator, cell_idx: int, horizon_s: float) -> float:
        _, _, f_ml = self._future_measurements(sim, float(horizon_s))
        if cell_idx < 0 or cell_idx >= len(f_ml):
            return float("inf")
        return float(f_ml[int(cell_idx)])

    def _nearest_nonserving_by_distance(self, sim: simulator.LEOSimulator) -> int:
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        return self._best_nonserving_index(sim, np.asarray(ue.ml_m, dtype=float))

    def _future_nearest_nonserving_by_distance(self, sim: simulator.LEOSimulator, horizon_s: float) -> int:
        _, _, f_ml = self._future_measurements(sim, float(horizon_s))
        return self._best_nonserving_index(sim, np.asarray(f_ml, dtype=float))

    def _approaching_score(self, sim: simulator.LEOSimulator, cell_idx: int, horizon_s: float) -> float:
        return self._distance_to_cell(sim, cell_idx) - self._future_distance_to_cell(sim, cell_idx, horizon_s)

    def _stable_target_by_future_geometry(
        self,
        sim: simulator.LEOSimulator,
        delay_s: float,
        skip_transient: bool = False,
    ) -> int:
        """Select a future-stable handover target from distance trajectories.

        The selector is deliberately geometry-based: it ranks non-serving cells
        by distance after the planned delay, approach trend, predicted dwell
        near the nearest-cell frontier, and short-ToS risk. It does not rank by
        target RSRP/SINR.
        """
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        sample_offsets = [0.0, 0.5, 1.0, 1.5, 2.0]
        horizons = [max(0.0, float(delay_s) + offset) for offset in sample_offsets]
        future_ml_by_horizon: dict[float, np.ndarray] = {}
        for h in horizons:
            _, _, f_ml = self._future_measurements(sim, h)
            future_ml_by_horizon[h] = np.asarray(f_ml, dtype=float)

        delay_ml = future_ml_by_horizon[horizons[0]].copy()
        delay_ml[serving_idx] = np.inf
        order = [int(idx) for idx in np.argsort(delay_ml) if np.isfinite(delay_ml[int(idx)])]
        # Strategy decisions only need a compact candidate set: current nearest,
        # nearest-at-delay, and a few near-frontier cells that could remain
        # stable after the planned wait.
        current_nearest = self._nearest_nonserving_by_distance(sim)
        future_nearest = [self._best_nonserving_index(sim, future_ml_by_horizon[h]) for h in horizons]
        candidates = list(dict.fromkeys([current_nearest] + future_nearest + order[:8]))
        candidates = [idx for idx in candidates if idx != serving_idx]
        if not candidates:
            return serving_idx

        radius = max(self.cfg.system.cell_radius_m, 1.0)
        stable_margin_m = max(self.cfg.distance_margin_m, 0.05 * radius)
        scores: list[tuple[float, int]] = []

        for idx in candidates:
            distances = np.asarray([future_ml_by_horizon[h][idx] for h in horizons], dtype=float)
            if not np.isfinite(distances).all():
                continue
            nearest_distances = []
            nearest_ids = []
            for h in horizons:
                values = future_ml_by_horizon[h].copy()
                values[serving_idx] = np.inf
                nearest = int(np.argmin(values))
                nearest_ids.append(nearest)
                nearest_distances.append(float(values[nearest]))
            nearest_distances_arr = np.asarray(nearest_distances, dtype=float)
            dwell_slots = float(np.sum(distances <= nearest_distances_arr + stable_margin_m))
            dwell_s = dwell_slots * 0.5
            distance_at_delay = float(distances[0])
            mean_distance = float(np.mean(distances[:4]))
            approach = self._distance_to_cell(sim, idx) - distance_at_delay
            # Penalize cells that are only briefly attractive; this is the
            # geometry-level proxy for UHO risk.
            short_dwell_penalty = max(0.0, self.cfg.system.min_tos_s - dwell_s)
            transient_penalty = 1.0 if idx != nearest_ids[0] and not skip_transient else 0.0
            if skip_transient and idx == nearest_ids[0]:
                transient_penalty += 3.0
            stability_bonus = dwell_s / max(self.cfg.system.min_tos_s, 1e-9)
            score = (
                -1.0 * distance_at_delay / radius
                -0.35 * mean_distance / radius
                +0.60 * np.clip(approach / radius, -2.0, 2.0)
                +0.80 * stability_bonus
                -2.50 * short_dwell_penalty
                -0.50 * transient_penalty
            )
            scores.append((float(score), int(idx)))

        if not scores:
            return self._future_nearest_nonserving_by_distance(sim, max(float(delay_s), 0.0))
        scores.sort(reverse=True)
        return int(scores[0][1])

    def _action_horizon(self, action_idx: int) -> float:
        if action_idx <= 1:
            return 1.0
        horizons = list(self.cfg.simulation.lookahead_horizons_s)
        return float(horizons[min(max(action_idx - 2, 0), len(horizons) - 1)] if horizons else 1.0)

    def _action_delay_s(self, action_idx: int) -> float:
        window = self._action_window_s(action_idx)
        return float(window[0])

    def _action_window_s(self, action_idx: int) -> tuple[float, float]:
        windows = {
            0: (0.0, 0.0),
            1: (0.0, 0.5),
            2: (0.5, 1.0),
            3: (1.0, 1.5),
            4: (1.5, 2.0),
            5: (2.0, 3.0),
            6: (0.5, 3.0),
        }
        return windows.get(int(action_idx), (0.0, 0.0))

    def _delay_to_step(self, delay_s: float, total_steps: int) -> int:
        delay_steps = int(round(float(delay_s) / self.cfg.simulation.sample_time_s))
        return int(np.clip(delay_steps, 0, max(total_steps - 1, 0)))

    def _strategy_plan(self, sim: simulator.LEOSimulator, action_idx: int) -> tuple[float, int | None]:
        if action_idx == 0:
            return 0.0, None
        start_s, end_s = self._action_window_s(action_idx)
        step_s = max(self.cfg.simulation.sample_time_s, 0.5)
        times = np.arange(start_s, end_s + 1e-9, step_s)
        if len(times) == 0:
            times = np.asarray([start_s], dtype=float)
        best_score = -float("inf")
        best_delay = float(times[0])
        best_target: int | None = None
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        for delay_s in times:
            target = self._stable_target_by_future_geometry(sim, float(delay_s), skip_transient=(action_idx == 6))
            if target is None:
                continue
            target_distance = self._future_distance_to_cell(sim, int(target), float(delay_s))
            serving_distance = self._future_distance_to_cell(sim, int(sim.ue.serving_idx), float(delay_s))
            wait_risk = self._serving_wait_risk(sim, float(delay_s))
            approach = self._approaching_score(sim, int(target), max(float(delay_s), 1.0))
            distance_advantage = (serving_distance - target_distance) / radius
            current_target_distance = self._distance_to_cell(sim, int(target))
            current_serving_distance = self._distance_to_cell(sim, int(sim.ue.serving_idx))
            current_distance_advantage = (current_serving_distance - current_target_distance) / radius
            tolerance_m = max(self.cfg.future_distance_margin_m, 0.10 * radius)
            currently_plausible = current_target_distance <= current_serving_distance + tolerance_m
            future_plausible = target_distance <= serving_distance + tolerance_m
            if not (future_plausible or (currently_plausible and approach >= 0.0) or wait_risk >= 0.4):
                continue
            score = (
                1.4 * distance_advantage
                + 0.4 * current_distance_advantage
                + 0.5 * np.clip(approach / radius, -2.0, 2.0)
                - 5.0 * wait_risk
                - 0.28 * float(delay_s)
            )
            if score > best_score:
                best_score = float(score)
                best_delay = float(delay_s)
                best_target = int(target)
        return best_delay, best_target

    def _serving_wait_risk(self, sim: simulator.LEOSimulator, delay_s: float) -> float:
        if delay_s <= 0.0:
            return 0.0
        ue = sim.ue
        assert ue is not None
        times = np.arange(0.0, delay_s + 1e-9, max(self.cfg.simulation.sample_time_s, 0.2))
        risk = 0.0
        for h in times:
            _, f_sinr, _ = self._future_measurements(sim, float(h))
            serving_sinr = float(f_sinr[int(ue.serving_idx)])
            if serving_sinr < self.cfg.system.q_out_db:
                risk += 1.0
            elif serving_sinr < self.cfg.system.q_in_db + 0.5:
                risk += 0.4
        return risk / max(len(times), 1)

    def _guard_decision(self, sim: simulator.LEOSimulator, action_idx: int, target_idx: int) -> tuple[bool, str, Dict[str, Any]]:
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        serving_idx = int(ue.serving_idx)
        horizon_s = max(self._action_delay_s(action_idx), 1.0)
        target_distance = self._distance_to_cell(sim, target_idx)
        serving_distance = self._distance_to_cell(sim, serving_idx)
        future_target_distance = self._future_distance_to_cell(sim, target_idx, horizon_s)
        future_serving_distance = self._future_distance_to_cell(sim, serving_idx, horizon_s)
        approaching = target_distance - future_target_distance
        details = {
            "dqn_target_idx": int(target_idx),
            "dqn_target_current_distance": float(target_distance),
            "dqn_target_future_distance": float(future_target_distance),
            "dqn_serving_current_distance": float(serving_distance),
            "dqn_serving_future_distance": float(future_serving_distance),
            "dqn_approaching_score": float(approaching),
        }
        prev_tos_s = sim.time_s - ue.serving_start_time_s
        if prev_tos_s < self.cfg.system.min_tos_s:
            return False, "min_tos", details
        if int(target_idx) == serving_idx or not np.isfinite(target_distance):
            return False, "invalid_target", details
        if self.cfg.future_action_mode == "guarded":
            # Independent strategy guard: allow planned geometry handovers, but
            # block handovers to clearly worse cells. This is not an A3 trigger;
            # it only enforces that the planned target is current/future
            # plausible or needed under serving-link risk.
            radius = max(self.cfg.system.cell_radius_m, 1.0)
            current_tolerance_m = max(self.cfg.distance_margin_m, 0.10 * radius)
            future_tolerance_m = max(self.cfg.future_distance_margin_m, 0.10 * radius)
            current_better = target_distance <= serving_distance - self.cfg.distance_margin_m
            future_better = future_target_distance <= future_serving_distance - self.cfg.future_distance_margin_m
            current_not_worse = target_distance <= serving_distance + current_tolerance_m
            future_not_worse = future_target_distance <= future_serving_distance + future_tolerance_m
            approaching_ok = approaching >= self.cfg.approaching_margin_m and future_not_worse
            serving_risk = ue.serving_sinr_db < self.cfg.system.q_in_db + 0.5 or ue.rlf.timer_s > 0.0
            reasons = []
            if current_better:
                reasons.append("current_distance")
            if future_better:
                reasons.append("future_distance")
            if approaching_ok:
                reasons.append("approaching")
            if serving_risk and (current_not_worse or future_not_worse or approaching >= 0.0):
                reasons.append("serving_risk")
            if reasons:
                return True, "+".join(reasons), details
            return False, "planned_target_worse", details
        geometry_ok = target_distance <= serving_distance - self.cfg.distance_margin_m
        future_geometry_ok = future_target_distance <= future_serving_distance - self.cfg.future_distance_margin_m
        approaching_ok = (
            approaching >= self.cfg.approaching_margin_m
            and future_target_distance <= future_serving_distance
        )
        serving_risk = ue.serving_sinr_db < self.cfg.system.q_in_db + 0.5 or ue.rlf.timer_s > 0.0

        reasons = []
        if geometry_ok:
            reasons.append("geometry")
        if future_geometry_ok:
            reasons.append("future_geometry")

        # Approaching alone is too permissive against a strong A3-RSRP
        # baseline. Require either a clear current/future distance advantage,
        # or serving-cell risk plus a non-worse geometric candidate.
        risk_geometry_ok = serving_risk and (
            target_distance <= serving_distance
            or future_target_distance <= future_serving_distance
            or approaching_ok
        )
        if risk_geometry_ok:
            reasons.append("serving_risk")
        if action_idx >= 2 and approaching_ok and future_target_distance <= future_serving_distance - 0.5 * self.cfg.future_distance_margin_m:
            reasons.append("future_approaching")
        if reasons:
            return True, "+".join(reasons), details
        return False, "guard_blocked", details

    def _empty_guard_info(self, action_idx: int, target_idx: int | None) -> Dict[str, Any]:
        return {
            "dqn_guard_executed": 0,
            "dqn_guard_reason": "hold" if action_idx == 0 else "not_evaluated",
            "dqn_target_idx": -1 if target_idx is None else int(target_idx),
            "dqn_target_current_distance": float("nan"),
            "dqn_target_future_distance": float("nan"),
            "dqn_serving_current_distance": float("nan"),
            "dqn_serving_future_distance": float("nan"),
            "dqn_approaching_score": float("nan"),
        }

    def score_valid_actions(self) -> Dict[int, LocalRolloutKPI]:
        snapshot = copy.deepcopy(self.sim)
        scores: Dict[int, LocalRolloutKPI] = {}
        for action_idx in self.valid_action_indices():
            scores[int(action_idx)] = self._rollout_action(copy.deepcopy(snapshot), int(action_idx), self.decision_window_s)
        return scores

    def _future_measurements(self, sim: simulator.LEOSimulator, horizon_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ue = sim.ue
        assert ue is not None
        beam_centers = np.array(sim.beam_centers, copy=True)
        beam_centers[:, 1] += self.cfg.system.sat_speed_mps * float(horizon_s)
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ux = ue.x_m + ue.speed_mps * np.cos(ue.heading_rad) * horizon_s
            uy = ue.y_m + ue.speed_mps * np.sin(ue.heading_rad) * horizon_s
        else:
            ux, uy = ue.x_m, ue.y_m
        return sim.channel_calc.update_channel(beam_centers[:, 0], beam_centers[:, 1], ux, uy, self.cfg.system.altitude_m)

    def _refresh_measurements(self, sim: simulator.LEOSimulator) -> None:
        ue = sim.ue
        assert ue is not None
        rsrp_dbm, sinr_db, ml_m = sim.channel_calc.update_channel(
            sim.beam_centers[:, 0], sim.beam_centers[:, 1], ue.x_m, ue.y_m, self.cfg.system.altitude_m
        )
        ue.rsrp_dbm = rsrp_dbm
        ue.sinr_db = sinr_db
        ue.ml_m = ml_m
        ue.update_serving_metrics()

    def _log_current(
        self,
        sim: simulator.LEOSimulator,
        action_idx: int,
        ho_event: bool,
        prev_tos_s: float,
        rb_delta: int,
        rlf_event: bool = False,
        reward: float = 0.0,
    ) -> None:
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        sinr_db = np.asarray(ue.sinr_db, dtype=float)
        best_idx = self._best_nonserving_by_sinr(int(ue.serving_idx), sinr_db)
        sim.sim_history.log_step(
            history.HistoryEntry(
                sim.time_s,
                sim.step_idx,
                action_idx,
                -1.0,
                -1.0,
                int(ue.serving_idx),
                float(ml_m[int(ue.serving_idx)]),
                float(ue.serving_rsrp_dbm),
                float(ue.serving_sinr_db),
                int(best_idx),
                float(ml_m[best_idx]),
                float(sinr_db[best_idx]),
                int(ho_event),
                0,
                int(ho_event),
                int(ho_event and 0.0 <= prev_tos_s < self.cfg.system.min_tos_s),
                int(rlf_event),
                int(rb_delta),
                int(ue.rb_count),
                float(reward),
            )
        )

    def _observe(self) -> np.ndarray:
        if self.cfg.dqn_feature_mode == "signal_predictive":
            return self._observe_signal_predictive()
        return self._observe_distance_predictive()

    def _observe_distance_predictive(self) -> np.ndarray:
        ue = self.sim.ue
        assert ue is not None
        self._refresh_measurements(self.sim)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        sinr_db = np.asarray(ue.sinr_db, dtype=float)
        serving_idx = int(ue.serving_idx)
        dist_order = [int(idx) for idx in np.argsort(ml_m) if int(idx) != serving_idx and np.isfinite(ml_m[int(idx)])]
        nearest_idx = dist_order[0] if dist_order else serving_idx
        second_idx = dist_order[1] if len(dist_order) > 1 else nearest_idx
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        serving_distance = float(ml_m[serving_idx])
        nearest_distance = float(ml_m[nearest_idx])
        second_distance = float(ml_m[second_idx])
        future_nearest_ids = []
        base = [
            np.clip(serving_distance / radius, 0.0, 5.0) / 5.0,
            np.clip(nearest_distance / radius, 0.0, 5.0) / 5.0,
            np.clip((nearest_distance - serving_distance) / radius, -5.0, 5.0) / 10.0 + 0.5,
            np.clip(second_distance / radius, 0.0, 5.0) / 5.0,
            np.clip(self._approaching_score(self.sim, nearest_idx, 1.0) / radius, -2.0, 2.0) / 4.0 + 0.5,
            np.clip(self._approaching_score(self.sim, nearest_idx, 2.0) / radius, -2.0, 2.0) / 4.0 + 0.5,
            np.clip(self._approaching_score(self.sim, nearest_idx, 3.0) / radius, -2.0, 2.0) / 4.0 + 0.5,
            np.clip(ue.speed_mps / max(self.cfg.simulation.ue_speed_max_mps, 1.0), 0.0, 1.0),
            0.5 + 0.5 * float(np.cos(ue.heading_rad)),
            0.5 + 0.5 * float(np.sin(ue.heading_rad)),
            np.clip(self.sim.time_s / max(self.cfg.simulation.total_time_s, 1e-9), 0.0, 1.0),
            np.clip(current_tos_s / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            np.clip(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9), 0.0, 1.0),
            np.clip(ue.ho_count / 20.0, 0.0, 1.0),
            np.clip(ue.rlf_event_count / 20.0, 0.0, 1.0),
            np.clip(ue.rb_count / 250.0, 0.0, 1.0),
            np.clip((ue.serving_sinr_db + 30.0) / 50.0, 0.0, 1.0),
            float(ue.serving_sinr_db < self.cfg.system.q_in_db + 0.5),
            float(self.is_decision_state()),
        ]
        future: list[float] = []
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, _, f_ml = self._future_measurements(self.sim, float(horizon_s))
            f_best = self._best_nonserving_index(self.sim, f_ml)
            future_nearest_ids.append(f_best)
            f_serving_distance = float(f_ml[serving_idx])
            f_nearest_distance = float(f_ml[f_best])
            future.extend(
                [
                    np.clip(f_serving_distance / radius, 0.0, 5.0) / 5.0,
                    np.clip(f_nearest_distance / radius, 0.0, 5.0) / 5.0,
                    np.clip((f_nearest_distance - f_serving_distance) / radius, -5.0, 5.0) / 10.0 + 0.5,
                    float(f_best == nearest_idx),
                ]
            )
        if future_nearest_ids:
            future.extend(
                [
                    float(len(set(future_nearest_ids)) == 1),
                    float(len(future_nearest_ids) >= 2 and future_nearest_ids[0] == future_nearest_ids[1]),
                    float(len(future_nearest_ids) >= 3 and future_nearest_ids[1] == future_nearest_ids[2]),
                ]
            )
        state = base + future
        while len(state) < self.state_dim:
            state.append(0.0)
        return np.asarray(state[: self.state_dim], dtype=np.float32)

    def _observe_signal_predictive(self) -> np.ndarray:
        feature_mode = self.cfg.dqn_feature_mode
        self.cfg.dqn_feature_mode = "distance_predictive"
        try:
            state = self._observe_distance_predictive()
        finally:
            self.cfg.dqn_feature_mode = feature_mode
        return state

    def get_kpi(self):
        return self.sim.get_kpi()

    def get_history_dicts(self):
        return self.sim.get_history_dicts()
