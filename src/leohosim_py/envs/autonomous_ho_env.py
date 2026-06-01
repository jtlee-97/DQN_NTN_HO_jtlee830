"""Autonomous direct-handover DQN environment.

This environment does not wrap A3. The agent directly decides whether to keep
the current serving cell or hand over to a current/future predicted target at
every simulator step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from leohosim_py import config, history, simulator


ACTION_NAMES = [
    "hold_or_keep_plan",
    "plan_best_sinr_now",
    "plan_nearest_now",
    "plan_best_sinr_1s",
    "plan_best_sinr_2s",
    "plan_best_sinr_3s",
    "plan_stable_dwell",
]


@dataclass
class AutonomousHOStep:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class AutonomousHandoverEnv:
    """Gym-like autonomous handover environment.

    Actions:
    0. stay on current serving cell or keep an active handover plan
    1. plan/execute handover to current best non-serving SINR cell
    2. plan/execute handover to current nearest non-serving cell
    3. plan/execute handover to best predicted SINR cell at horizon[0]
    4. plan/execute handover to best predicted SINR cell at horizon[1]
    5. plan/execute handover to best predicted SINR cell at horizon[2]
    6. plan/execute handover to a stable dwell target across current/future horizons

    A non-zero action schedules a target while the serving ToS is below
    min_tos_s, then executes it automatically when the ToS guard is satisfied.
    RLF-risk rescue is allowed to break the ToS guard when the target is safer
    than staying on a collapsing serving cell.
    """

    action_names = ACTION_NAMES
    state_dim = 54
    action_dim = len(ACTION_NAMES)

    def __init__(self, cfg: config.LeohosimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.sim = simulator.LEOSimulator(cfg, self.rng)
        self.last_info: Dict[str, Any] = {}
        self.planned_target_idx: int | None = None
        self.planned_action_idx: int | None = None
        self.planned_start_time_s: float = 0.0
        self._measurement_cache_step: int | None = None

    def reset(self) -> np.ndarray:
        self.sim = simulator.LEOSimulator(self.cfg, self.rng)
        self.sim.reset()
        self._clear_plan()
        self._measurement_cache_step = None
        self._refresh_measurements()
        self.last_info = {}
        return self._observe()

    def step(self, action_idx: int) -> AutonomousHOStep:
        action_idx = int(np.clip(action_idx, 0, self.action_dim - 1))
        valid = self.valid_action_indices()
        if action_idx not in set(valid.tolist()):
            action_idx = int(valid[0])

        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_idx_before = int(ue.serving_idx)
        prev_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        pre_sinr = float(ue.serving_sinr_db)
        pre_best_idx = self._best_nonserving_by_sinr(np.asarray(ue.sinr_db, dtype=float))
        pre_best_sinr = float(ue.sinr_db[pre_best_idx])
        pre_best_gap = float(pre_best_sinr - pre_sinr)
        pre_rlf_risk = self._rlf_risk()
        future_serving_min, future_best_gap = self._future_risk_summary(serving_idx_before)

        target_idx = self._action_target(action_idx)
        selected_target_idx = -1 if target_idx is None else int(target_idx)
        selected_target_gap = 0.0
        if target_idx is not None and int(target_idx) != serving_idx_before:
            selected_target_gap = float(ue.sinr_db[int(target_idx)] - pre_sinr)
        had_plan_before = self.planned_target_idx is not None
        plan_event = False
        plan_replaced = False
        if target_idx is not None and int(target_idx) != serving_idx_before:
            plan_event, plan_replaced = self._set_plan(int(target_idx), action_idx)

        ho_event = False
        uho_event = False
        plan_failed = False
        emergency_exec = False
        rb_delta = 0
        exec_target, plan_failed, emergency_exec = self._planned_execution_target(prev_tos_s)
        if exec_target is not None and int(exec_target) != serving_idx_before:
            ho_event = True
            target_idx = int(exec_target)
            uho_event = prev_tos_s < self.cfg.system.min_tos_s - 1e-6
            ue.change_serving_cell(int(exec_target), self.sim.time_s)
            ue.ho_count += 1
            if uho_event:
                ue.uho_count += 1
            rb_delta += 10
            self._clear_plan()
        ue.rb_count += rb_delta

        hopp_event = self.sim.hopp_detector.record_handover(ue.serving_idx, self.sim.time_s) if ho_event else False
        self._refresh_measurements()
        rlf_event = self.sim.rlf_detector.update_rlf(
            ue,
            ue.serving_sinr_db,
            self.cfg.simulation.sample_time_s,
            self.sim.time_s,
        )
        if rlf_event:
            ue.rlf_event_count += 1

        best_idx = self._best_nonserving_by_sinr(np.asarray(ue.sinr_db, dtype=float))
        reward = self._dense_reward(
            action_idx=action_idx,
            ho_event=ho_event,
            uho_event=uho_event,
            rlf_event=bool(rlf_event),
            hopp_event=bool(hopp_event),
            rb_delta=rb_delta,
            prev_tos_s=prev_tos_s,
            pre_sinr=pre_sinr,
            pre_best_gap=pre_best_gap,
            selected_target_gap=selected_target_gap,
            pre_rlf_risk=pre_rlf_risk,
            future_serving_min=future_serving_min,
            future_best_gap=future_best_gap,
            had_plan_before=had_plan_before,
            has_plan_after=self.planned_target_idx is not None,
            plan_event=plan_event,
            plan_replaced=plan_replaced,
            plan_failed=plan_failed,
            emergency_exec=emergency_exec,
        )
        self._log_step(
            action_idx=action_idx,
            target_idx=-1 if target_idx is None else int(target_idx),
            plan_event=plan_event,
            plan_failed=plan_failed,
            ho_event=ho_event,
            uho_event=uho_event,
            rlf_event=bool(rlf_event),
            hopp_event=bool(hopp_event),
            rb_delta=rb_delta,
            reward=reward,
            best_idx=best_idx,
        )
        self._advance_time()

        info = {
            "action_idx": action_idx,
            "action_name": self.action_names[action_idx],
            "selected_target_idx": int(selected_target_idx),
            "target_idx": -1 if target_idx is None else int(target_idx),
            "planned_target_idx": -1 if self.planned_target_idx is None else int(self.planned_target_idx),
            "planned_action_idx": -1 if self.planned_action_idx is None else int(self.planned_action_idx),
            "serving_idx_before": serving_idx_before,
            "serving_idx_after": int(ue.serving_idx),
            "plan_event": int(plan_event),
            "plan_replaced": int(plan_replaced),
            "plan_failed": int(plan_failed),
            "emergency_exec": int(emergency_exec),
            "ho_event": int(ho_event),
            "uho_event": int(uho_event),
            "rlf_event": int(bool(rlf_event)),
            "hopp_event": int(bool(hopp_event)),
            "rb_delta": int(rb_delta),
            "prev_tos_s": float(prev_tos_s),
            "pre_sinr_db": float(pre_sinr),
            "serving_sinr_db": float(ue.serving_sinr_db),
            "best_sinr_db": float(ue.sinr_db[best_idx]),
            "pre_best_gap_db": float(pre_best_gap),
            "selected_target_gap_db": float(selected_target_gap),
            "rlf_risk": float(pre_rlf_risk),
            "future_serving_min_sinr_db": float(future_serving_min),
            "future_best_gap_db": float(future_best_gap),
        }
        self.last_info = info
        return AutonomousHOStep(self._observe(), float(reward), self.sim.is_done(), info)

    def valid_action_indices(self) -> np.ndarray:
        if self.sim.is_done():
            return np.asarray([0], dtype=np.int64)
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        serving_sinr = float(ue.serving_sinr_db)
        sinrs = np.asarray(ue.sinr_db, dtype=float)
        best_idx = self._best_nonserving_by_sinr(sinrs)
        best_gap = float(sinrs[best_idx] - serving_sinr)
        future_serving_min, future_best_gap = self._future_risk_summary(int(ue.serving_idx))
        radio_risk = (
            serving_sinr < self.cfg.system.q_in_db
            or self._rlf_risk() > 0.2
            or (future_serving_min < self.cfg.system.q_out_db and future_best_gap > 0.75)
        )
        rescue_required = self._serving_rescue_required(
            serving_idx=int(ue.serving_idx),
            serving_sinr=serving_sinr,
            future_serving_min=future_serving_min,
            future_best_gap=future_best_gap,
        )
        valid = [0]
        plan_actions = []
        for action in range(1, self.action_dim):
            target = self._action_target(action)
            if target is not None and int(target) != int(ue.serving_idx):
                if self._target_is_schedulable(int(target), serving_sinr, radio_risk):
                    plan_actions.append(action)
        valid.extend(plan_actions)
        has_useful_plan = self._has_useful_plan(int(ue.serving_idx))
        should_force_decision = (
            radio_risk
            and len(plan_actions) > 0
            and (rescue_required or not has_useful_plan or tos_s >= self.cfg.system.min_tos_s)
        )
        if should_force_decision:
            valid = plan_actions
        return np.asarray(sorted(set(valid)), dtype=np.int64)

    def _dense_reward(
        self,
        action_idx: int,
        ho_event: bool,
        uho_event: bool,
        rlf_event: bool,
        hopp_event: bool,
        rb_delta: int,
        prev_tos_s: float,
        pre_sinr: float,
        pre_best_gap: float,
        selected_target_gap: float,
        pre_rlf_risk: float,
        future_serving_min: float,
        future_best_gap: float,
        had_plan_before: bool,
        has_plan_after: bool,
        plan_event: bool,
        plan_replaced: bool,
        plan_failed: bool,
        emergency_exec: bool,
    ) -> float:
        ue = self.sim.ue
        assert ue is not None
        sinr = float(ue.serving_sinr_db)
        q_out = self.cfg.system.q_out_db
        q_in = self.cfg.system.q_in_db
        sinr_span = max(self.cfg.reward.episode_norm_sinr_max_db - self.cfg.reward.episode_norm_sinr_min_db, 1e-9)
        sinr_quality = float(np.clip((sinr - self.cfg.reward.episode_norm_sinr_min_db) / sinr_span, 0.0, 1.25))
        outage = sinr < q_out
        weak = sinr < q_in
        future_outage_risk = future_serving_min < q_out and future_best_gap > 0.75
        unnecessary_ho = ho_event and pre_sinr > q_in + 2.0 and pre_best_gap < 1.0
        risk_state = pre_sinr < q_in or pre_rlf_risk > 0.25 or future_outage_risk
        useful_plan_or_new_plan = had_plan_before or has_plan_after or plan_event
        unsafe_hold = action_idx == 0 and risk_state and pre_best_gap > 0.75 and not useful_plan_or_new_plan
        missed_plan = (not useful_plan_or_new_plan) and prev_tos_s < self.cfg.system.min_tos_s and risk_state and pre_best_gap > 0.75
        timely_plan = plan_event and prev_tos_s < self.cfg.system.min_tos_s and risk_state and selected_target_gap > -0.5
        good_replan = plan_replaced and risk_state and selected_target_gap > 0.75
        bad_plan = plan_event and (not risk_state) and selected_target_gap < 0.25
        rescue = ho_event and (pre_sinr < q_in or pre_rlf_risk > 0.25 or future_outage_risk)
        sinr_improvement = max(0.0, sinr - pre_sinr)
        short_tos_penalty = 6.0 if rescue else 16.0

        reward = (
            6.0 * sinr_quality
            - 90.0 * float(outage)
            - 10.0 * float(weak)
            - 500.0 * float(rlf_event)
            - 50.0 * float(pre_rlf_risk)
            - 25.0 * float(uho_event)
            - 18.0 * float(hopp_event)
            - 0.5 * float(ho_event)
            - 0.035 * float(rb_delta)
            - short_tos_penalty * float(ho_event and prev_tos_s < self.cfg.system.min_tos_s)
            - 12.0 * float(unnecessary_ho)
            - 48.0 * float(unsafe_hold)
            - 45.0 * float(missed_plan)
            - 25.0 * float(plan_failed)
            - 4.0 * float(bad_plan)
            - 28.0 * float(action_idx == 0 and future_outage_risk and not useful_plan_or_new_plan)
            + 16.0 * float(timely_plan)
            + 6.0 * float(good_replan)
            + 70.0 * float(rescue)
            + 8.0 * min(sinr_improvement, 7.0) * float(rescue)
            + 90.0 * float(emergency_exec and not rlf_event)
            + 0.08 * min(max(self.sim.time_s - ue.serving_start_time_s, 0.0), 8.0)
        )
        return float(reward)

    def _clear_plan(self) -> None:
        self.planned_target_idx = None
        self.planned_action_idx = None
        self.planned_start_time_s = 0.0

    def _set_plan(self, target_idx: int, action_idx: int) -> tuple[bool, bool]:
        ue = self.sim.ue
        assert ue is not None
        if int(target_idx) == int(ue.serving_idx):
            return False, False
        previous = self.planned_target_idx
        if previous == int(target_idx):
            self.planned_action_idx = int(action_idx)
            return False, False
        if previous is not None and int(previous) != int(target_idx):
            plan_age_s = max(0.0, self.sim.time_s - self.planned_start_time_s)
            old_score = self._target_plan_score(int(previous))
            new_score = self._target_plan_score(int(target_idx))
            if self._serving_rescue_required():
                old_rescue = self._target_rescue_score(int(previous))
                new_rescue = self._target_rescue_score(int(target_idx))
                if new_rescue < old_rescue + 0.25:
                    return False, False
            else:
                if plan_age_s < 0.6 and new_score < old_score + 1.0:
                    return False, False
                if self._has_useful_plan(int(ue.serving_idx)) and new_score < old_score + 0.5:
                    return False, False
        self.planned_target_idx = int(target_idx)
        self.planned_action_idx = int(action_idx)
        self.planned_start_time_s = float(self.sim.time_s)
        return True, previous is not None and int(previous) != int(target_idx)

    def _target_plan_score(self, target_idx: int) -> float:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size:
            return -float("inf")
        serving_idx = int(ue.serving_idx)
        values = [float(sinr[target_idx])]
        gaps = [float(sinr[target_idx] - sinr[serving_idx])]
        distances = [float(ml_m[target_idx])]
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, f_ml = self._future_measurements(float(horizon_s))
            values.append(float(f_sinr[target_idx]))
            gaps.append(float(f_sinr[target_idx] - f_sinr[serving_idx]))
            distances.append(float(f_ml[target_idx]))
        vals = np.asarray(values, dtype=float)
        gap_arr = np.asarray(gaps, dtype=float)
        dist_arr = np.asarray(distances, dtype=float)
        dwell_votes = float(np.sum(vals >= self.cfg.system.q_in_db))
        return float(
            0.45 * np.mean(vals)
            + 0.35 * np.min(vals)
            + 0.65 * np.max(gap_arr)
            + 0.5 * dwell_votes
            - 0.00001 * np.mean(dist_arr)
        )

    def _planned_execution_target(self, tos_s: float) -> tuple[int | None, bool, bool]:
        ue = self.sim.ue
        assert ue is not None
        if self.planned_target_idx is None:
            return None, False, False
        target = int(self.planned_target_idx)
        if target == int(ue.serving_idx):
            self._clear_plan()
            return None, False, False
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target < 0 or target >= sinr.size or not np.isfinite(sinr[target]):
            self._clear_plan()
            return None, True, False

        sinr = np.asarray(ue.sinr_db, dtype=float)
        stable_target = self._best_execution_target(target)
        current_best_target = self._best_nonserving_by_sinr(sinr)
        rescue_target = self._best_rescue_target(target, stable_target, current_best_target)
        due = tos_s >= self.cfg.system.min_tos_s + 1e-6
        emergency = self._emergency_execution_allowed(rescue_target, tos_s)
        if not due and not emergency:
            return None, False, False
        if emergency:
            target = int(rescue_target)
        elif ue.serving_sinr_db < self.cfg.system.q_in_db and sinr[current_best_target] > sinr[stable_target] + 0.25:
            target = int(current_best_target)
        else:
            target = int(stable_target)
        if emergency:
            return target, False, bool(not due)

        serving_risky = (
            ue.serving_sinr_db < self.cfg.system.q_in_db
            or self._rlf_risk() > 0.25
            or self._future_risk_summary(int(ue.serving_idx))[0] < self.cfg.system.q_out_db
        )
        gap = float(sinr[target] - ue.serving_sinr_db)
        if gap < -0.75 and not (serving_risky and sinr[target] > ue.serving_sinr_db):
            self._clear_plan()
            return None, True, False
        if gap < 0.25 and not serving_risky:
            self._clear_plan()
            return None, True, False
        return target, False, bool(emergency and not due)

    def _emergency_execution_allowed(self, target_idx: int, tos_s: float) -> bool:
        ue = self.sim.ue
        assert ue is not None
        if tos_s >= self.cfg.system.min_tos_s:
            return False
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size:
            return False
        serving_sinr = float(ue.serving_sinr_db)
        future_serving_min, future_best_gap = self._future_risk_summary(int(ue.serving_idx))
        q_out = self.cfg.system.q_out_db
        risk = self._rlf_risk()
        hard_risk = risk >= 0.55 or serving_sinr < q_out
        min_tos_elapsed = tos_s >= max(0.25, 0.12 * self.cfg.system.min_tos_s) or hard_risk
        profile = self._target_rescue_profile(target_idx)
        if not profile:
            return False
        strong_correction = (
            tos_s >= max(0.2, 0.08 * self.cfg.system.min_tos_s)
            and serving_sinr < self.cfg.system.q_in_db + 6.0
            and profile["gap_now"] >= 1.4
            and profile["target_now"] >= q_out - 1.0
        )
        early_risk = (
            risk >= 0.30
            or strong_correction
            or serving_sinr < q_out + 0.75
            or (future_serving_min < q_out and future_best_gap > 0.75)
        )
        if not early_risk:
            return False
        anticipatory_rescue = (
            future_serving_min < q_out
            and profile["target_min"] >= q_out - 1.5
            and profile["target_min"] > profile["serving_min"] + 2.0
            and profile["target_mean"] > profile["serving_mean"] + 1.0
        )
        target_viable = (
            profile["target_now"] >= q_out - 1.5
            and (profile["target_min"] >= q_out - 2.5 or profile["target_mean"] >= q_out - 0.25)
        )
        target_better = (
            strong_correction
            or anticipatory_rescue
            or (
                profile["gap_now"] >= 1.0
                and (
                    profile["gap_mean"] >= 0.5
                    or profile["target_min"] > profile["serving_min"] + 0.75
                    or profile["target_now"] > serving_sinr + 2.0
                )
            )
        )
        return bool(min_tos_elapsed and (target_viable or anticipatory_rescue or strong_correction) and target_better)

    def _serving_rescue_required(
        self,
        serving_idx: int | None = None,
        serving_sinr: float | None = None,
        future_serving_min: float | None = None,
        future_best_gap: float | None = None,
    ) -> bool:
        ue = self.sim.ue
        assert ue is not None
        if serving_idx is None:
            serving_idx = int(ue.serving_idx)
        if serving_sinr is None:
            serving_sinr = float(ue.serving_sinr_db)
        if future_serving_min is None or future_best_gap is None:
            future_serving_min, future_best_gap = self._future_risk_summary(int(serving_idx))
        return bool(
            self._rlf_risk() >= 0.30
            or serving_sinr < self.cfg.system.q_out_db
            or serving_sinr < self.cfg.system.q_out_db + 0.75
            or (future_serving_min < self.cfg.system.q_out_db and future_best_gap > 0.75)
        )

    def _target_rescue_profile(self, target_idx: int) -> dict[str, float]:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size or int(target_idx) == int(ue.serving_idx):
            return {}
        serving_idx = int(ue.serving_idx)
        target_values = [float(sinr[target_idx])]
        serving_values = [float(sinr[serving_idx])]
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, _ = self._future_measurements(float(horizon_s))
            target_values.append(float(f_sinr[target_idx]))
            serving_values.append(float(f_sinr[serving_idx]))
        target_arr = np.asarray(target_values, dtype=float)
        serving_arr = np.asarray(serving_values, dtype=float)
        gap_arr = target_arr - serving_arr
        return {
            "target_now": float(target_arr[0]),
            "target_min": float(np.min(target_arr)),
            "target_mean": float(np.mean(target_arr)),
            "serving_min": float(np.min(serving_arr)),
            "serving_mean": float(np.mean(serving_arr)),
            "gap_now": float(gap_arr[0]),
            "gap_min": float(np.min(gap_arr)),
            "gap_mean": float(np.mean(gap_arr)),
            "gap_max": float(np.max(gap_arr)),
        }

    def _target_rescue_score(self, target_idx: int) -> float:
        profile = self._target_rescue_profile(target_idx)
        if not profile:
            return -float("inf")
        q_out = self.cfg.system.q_out_db
        viability_margin = min(profile["target_now"] - (q_out - 1.5), profile["target_min"] - (q_out - 2.5))
        if viability_margin < -2.0:
            return -float("inf")
        return float(
            1.00 * profile["gap_now"]
            + 0.75 * profile["gap_mean"]
            + 0.55 * profile["gap_max"]
            + 0.45 * profile["target_min"]
            + 0.25 * profile["target_mean"]
            + 0.80 * min(max(viability_margin, -2.0), 4.0)
        )

    def _best_rescue_target(self, *seed_targets: int) -> int:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        candidates = {
            int(idx)
            for idx in seed_targets
            if int(idx) >= 0 and int(idx) < sinr.size and int(idx) != int(ue.serving_idx)
        }
        candidates.add(self._best_nonserving_by_sinr(sinr))
        for action in range(1, self.action_dim):
            target = self._action_target(action)
            if target is not None and int(target) != int(ue.serving_idx):
                candidates.add(int(target))
        scored = [(self._target_rescue_score(idx), idx) for idx in candidates]
        scored = [(score, idx) for score, idx in scored if np.isfinite(score)]
        if not scored:
            return self._best_nonserving_by_sinr(sinr)
        return int(max(scored, key=lambda item: item[0])[1])

    def _best_execution_target(self, planned_target_idx: int) -> int:
        ue = self.sim.ue
        assert ue is not None
        candidates = {int(planned_target_idx)}
        for action in range(1, self.action_dim):
            target = self._action_target(action)
            if target is not None and int(target) != int(ue.serving_idx):
                candidates.add(int(target))
        scored = [(self._target_plan_score(idx), idx) for idx in candidates]
        scored = [(score, idx) for score, idx in scored if np.isfinite(score)]
        if not scored:
            return int(planned_target_idx)
        return int(max(scored, key=lambda item: item[0])[1])

    def _target_is_schedulable(self, target_idx: int, serving_sinr: float, radio_risk: bool) -> bool:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size or int(target_idx) == int(ue.serving_idx):
            return False
        current_gap = float(sinr[target_idx] - serving_sinr)
        future_gaps = [current_gap]
        future_values = [float(sinr[target_idx])]
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, _ = self._future_measurements(float(horizon_s))
            future_gaps.append(float(f_sinr[target_idx] - f_sinr[int(ue.serving_idx)]))
            future_values.append(float(f_sinr[target_idx]))
        best_future_gap = float(np.max(future_gaps))
        best_future_sinr = float(np.max(future_values))
        if current_gap >= 0.25 or best_future_gap >= 0.75:
            return True
        return bool(radio_risk and best_future_sinr >= self.cfg.system.q_out_db and current_gap >= -2.0)

    def _has_useful_plan(self, serving_idx: int) -> bool:
        if self.planned_target_idx is None:
            return False
        target = int(self.planned_target_idx)
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target < 0 or target >= sinr.size or target == int(serving_idx):
            return False
        gap = float(sinr[target] - ue.serving_sinr_db)
        return bool(gap >= -0.5 or self._target_is_schedulable(target, ue.serving_sinr_db, True))

    def _action_target(self, action_idx: int) -> int | None:
        if action_idx == 0:
            return None
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        if action_idx == 1:
            return self._best_nonserving_by_sinr(np.asarray(ue.sinr_db, dtype=float))
        if action_idx == 2:
            return self._nearest_nonserving_by_distance(np.asarray(ue.ml_m, dtype=float))
        if action_idx in (3, 4, 5):
            horizons = list(self.cfg.simulation.lookahead_horizons_s)
            h_idx = min(action_idx - 3, max(len(horizons) - 1, 0))
            horizon_s = float(horizons[h_idx]) if horizons else float(action_idx - 2)
            _, f_sinr, _ = self._future_measurements(horizon_s)
            return self._best_nonserving_by_sinr(f_sinr)
        if action_idx == 6:
            return self._stable_dwell_target()
        return None

    def _stable_dwell_target(self) -> int | None:
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        sinr = np.asarray(ue.sinr_db, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        order = [int(i) for i in np.argsort(-sinr) if int(i) != serving_idx]
        for i in np.argsort(ml_m):
            idx = int(i)
            if idx != serving_idx and idx not in order[:8]:
                order.append(idx)
            if len(order) >= 10:
                break
        candidates = order[:10]
        horizons = [0.0] + [float(h) for h in list(self.cfg.simulation.lookahead_horizons_s)[:3]]
        best: tuple[float, int] | None = None
        for idx in candidates:
            vals = []
            distances = []
            for horizon_s in horizons:
                if horizon_s == 0.0:
                    vals.append(float(sinr[idx]))
                    distances.append(float(ml_m[idx]))
                else:
                    _, f_sinr, f_ml = self._future_measurements(horizon_s)
                    vals.append(float(f_sinr[idx]))
                    distances.append(float(f_ml[idx]))
            vals_arr = np.asarray(vals, dtype=float)
            dist_arr = np.asarray(distances, dtype=float)
            dwell_votes = float(np.sum(vals_arr >= self.cfg.system.q_in_db))
            score = float(np.mean(vals_arr) + 0.35 * np.min(vals_arr) + 0.6 * dwell_votes - 0.00001 * np.mean(dist_arr))
            if best is None or score > best[0]:
                best = (score, int(idx))
        return None if best is None else int(best[1])

    def _future_risk_summary(self, serving_idx: int) -> tuple[float, float]:
        horizons = list(self.cfg.simulation.lookahead_horizons_s)[:3]
        if not horizons:
            return 0.0, 0.0
        serving_vals = []
        best_gaps = []
        for horizon_s in horizons:
            _, f_sinr, _ = self._future_measurements(float(horizon_s))
            f_best = self._best_nonserving_by_sinr(f_sinr)
            serving_vals.append(float(f_sinr[int(serving_idx)]))
            best_gaps.append(float(f_sinr[f_best] - f_sinr[int(serving_idx)]))
        return float(np.min(serving_vals)), float(np.max(best_gaps))

    def _best_nonserving_by_sinr(self, sinr_db: np.ndarray) -> int:
        ue = self.sim.ue
        assert ue is not None
        values = np.asarray(sinr_db, dtype=float).copy()
        if values.size == 0:
            return int(ue.serving_idx)
        values[int(ue.serving_idx)] = -np.inf
        if not np.isfinite(values).any():
            return int(ue.serving_idx)
        return int(np.argmax(values))

    def _nearest_nonserving_by_distance(self, ml_m: np.ndarray) -> int:
        ue = self.sim.ue
        assert ue is not None
        values = np.asarray(ml_m, dtype=float).copy()
        if values.size == 0:
            return int(ue.serving_idx)
        values[int(ue.serving_idx)] = np.inf
        if not np.isfinite(values).any():
            return int(ue.serving_idx)
        return int(np.argmin(values))

    def _future_measurements(self, horizon_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ue = self.sim.ue
        assert ue is not None
        beam_centers = np.array(self.sim.beam_centers, copy=True)
        beam_centers[:, 1] += self.cfg.system.sat_speed_mps * float(horizon_s)
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ux = ue.x_m + ue.speed_mps * np.cos(ue.heading_rad) * horizon_s
            uy = ue.y_m + ue.speed_mps * np.sin(ue.heading_rad) * horizon_s
        else:
            ux, uy = ue.x_m, ue.y_m
        return self.sim.channel_calc.update_channel(
            beam_centers[:, 0],
            beam_centers[:, 1],
            ux,
            uy,
            self.cfg.system.altitude_m,
        )

    def _refresh_measurements(self) -> None:
        ue = self.sim.ue
        assert ue is not None
        if self._measurement_cache_step == int(self.sim.step_idx) and ue.rsrp_dbm is not None and ue.sinr_db is not None and ue.ml_m is not None:
            ue.update_serving_metrics()
            return
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
        self._measurement_cache_step = int(self.sim.step_idx)

    def _rlf_risk(self) -> float:
        ue = self.sim.ue
        assert ue is not None
        return float(np.clip(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9), 0.0, 1.0))

    def _advance_time(self) -> None:
        ue = self.sim.ue
        assert ue is not None
        self.sim.beam_centers[:, 1] += self.cfg.system.sat_speed_mps * self.cfg.simulation.sample_time_s
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ue.move_linear(self.cfg.simulation.sample_time_s)
        self.sim.time_s += self.cfg.simulation.sample_time_s
        self.sim.step_idx += 1

    def _log_step(
        self,
        action_idx: int,
        target_idx: int,
        plan_event: bool,
        plan_failed: bool,
        ho_event: bool,
        uho_event: bool,
        rlf_event: bool,
        hopp_event: bool,
        rb_delta: int,
        reward: float,
        best_idx: int,
    ) -> None:
        ue = self.sim.ue
        assert ue is not None
        self.sim.sim_history.log_step(
            history.HistoryEntry(
                self.sim.time_s,
                self.sim.step_idx,
                int(action_idx),
                -1.0,
                -1.0,
                int(ue.serving_idx),
                float(ue.ml_m[int(ue.serving_idx)]),
                float(ue.serving_rsrp_dbm),
                float(ue.serving_sinr_db),
                int(best_idx),
                float(ue.ml_m[int(best_idx)]),
                float(ue.sinr_db[int(best_idx)]),
                int(plan_event),
                int(plan_failed),
                int(ho_event),
                int(uho_event),
                int(rlf_event),
                int(rb_delta),
                int(ue.rb_count),
                float(reward),
            )
        )

    def _observe(self) -> np.ndarray:
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        sinr_db = np.asarray(ue.sinr_db, dtype=float)
        rsrp_dbm = np.asarray(ue.rsrp_dbm, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        serving_idx = int(ue.serving_idx)
        best_sinr_idx = self._best_nonserving_by_sinr(sinr_db)
        nearest_idx = self._nearest_nonserving_by_distance(ml_m)
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        time_norm = np.clip(self.sim.time_s / max(self.cfg.simulation.total_time_s, 1e-9), 0.0, 1.0)
        base = [
            np.clip((ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
            np.clip((sinr_db[best_sinr_idx] + 20.0) / 40.0, 0.0, 1.0),
            np.clip((sinr_db[best_sinr_idx] - ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
            np.clip((rsrp_dbm[serving_idx] + 160.0) / 100.0, 0.0, 1.0),
            np.clip((rsrp_dbm[best_sinr_idx] + 160.0) / 100.0, 0.0, 1.0),
            np.clip(ml_m[serving_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(ml_m[best_sinr_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(ml_m[nearest_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(current_tos_s / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            self._rlf_risk(),
            np.clip(ue.ho_count / 20.0, 0.0, 1.0),
            np.clip(ue.uho_count / 10.0, 0.0, 1.0),
            np.clip(ue.rlf_event_count / 20.0, 0.0, 1.0),
            np.clip(ue.rb_count / 300.0, 0.0, 1.0),
            np.clip(ue.speed_mps / max(self.cfg.simulation.ue_speed_max_mps, 1.0), 0.0, 1.0),
            0.5 + 0.5 * float(np.cos(ue.heading_rad)),
            0.5 + 0.5 * float(np.sin(ue.heading_rad)),
            time_norm,
        ]
        future: list[float] = []
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, f_ml = self._future_measurements(float(horizon_s))
            f_best = self._best_nonserving_by_sinr(f_sinr)
            future.extend(
                [
                    np.clip((f_sinr[serving_idx] + 20.0) / 40.0, 0.0, 1.0),
                    np.clip((f_sinr[f_best] + 20.0) / 40.0, 0.0, 1.0),
                    np.clip((f_sinr[f_best] - f_sinr[serving_idx] + 20.0) / 40.0, 0.0, 1.0),
                    np.clip(f_ml[serving_idx] / radius, 0.0, 5.0) / 5.0,
                    np.clip(f_ml[f_best] / radius, 0.0, 5.0) / 5.0,
                    float(f_best == best_sinr_idx),
                    float(f_best == nearest_idx),
                    np.clip((f_sinr[f_best] - ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
                ]
            )
        stable = self._stable_dwell_target()
        stable_features = []
        if stable is None:
            stable_features = [0.0, 0.0, 0.0, 0.0]
        else:
            stable_features = [
                np.clip((sinr_db[stable] + 20.0) / 40.0, 0.0, 1.0),
                np.clip((sinr_db[stable] - ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
                np.clip(ml_m[stable] / radius, 0.0, 5.0) / 5.0,
                float(stable == best_sinr_idx),
            ]
        plan_features = self._plan_features(
            sinr_db=sinr_db,
            ml_m=ml_m,
            serving_idx=serving_idx,
            best_sinr_idx=best_sinr_idx,
            current_tos_s=current_tos_s,
            radius=radius,
        )
        state = base + future + stable_features + plan_features
        while len(state) < self.state_dim:
            state.append(0.0)
        return np.asarray(state[: self.state_dim], dtype=np.float32)

    def _plan_features(
        self,
        sinr_db: np.ndarray,
        ml_m: np.ndarray,
        serving_idx: int,
        best_sinr_idx: int,
        current_tos_s: float,
        radius: float,
    ) -> list[float]:
        if self.planned_target_idx is None:
            return [0.0] * 8
        target = int(self.planned_target_idx)
        if target < 0 or target >= len(sinr_db) or target == int(serving_idx):
            return [0.0] * 8
        time_to_ready = max(0.0, self.cfg.system.min_tos_s - current_tos_s)
        plan_age = max(0.0, self.sim.time_s - self.planned_start_time_s)
        current_gap = float(sinr_db[target] - sinr_db[int(serving_idx)])
        future_gaps = [current_gap]
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, _ = self._future_measurements(float(horizon_s))
            future_gaps.append(float(f_sinr[target] - f_sinr[int(serving_idx)]))
        best_future_gap = float(np.max(future_gaps))
        return [
            1.0,
            np.clip(time_to_ready / max(self.cfg.system.min_tos_s, 1e-9), 0.0, 1.0),
            np.clip(plan_age / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            np.clip((current_gap + 20.0) / 40.0, 0.0, 1.0),
            np.clip((best_future_gap + 20.0) / 40.0, 0.0, 1.0),
            np.clip((sinr_db[target] + 20.0) / 40.0, 0.0, 1.0),
            np.clip(ml_m[target] / radius, 0.0, 5.0) / 5.0,
            float(target == int(best_sinr_idx)),
        ]

    def get_kpi(self):
        return self.sim.get_kpi()

    def get_history_dicts(self):
        return self.sim.get_history_dicts()


class CandidateAutonomousHandoverEnv(AutonomousHandoverEnv):
    """Direct-HO environment where actions choose among ranked candidate cells.

    The base autonomous environment lets the DQN choose a target-selection
    heuristic. This variant exposes a fixed top-K candidate set at every step:
    action 0 holds/keeps a plan and actions 1..K schedule the corresponding
    ranked cell. The target rank is recomputed from current and lookahead radio
    measurements, so the policy learns when candidate #3 is safer than the
    instantaneous best, when to hold, and when to replan.
    """

    def __init__(
        self,
        cfg: config.LeohosimConfig,
        rng: np.random.Generator | None = None,
        candidate_count: int = 8,
    ):
        self.candidate_count = max(1, int(candidate_count))
        self.action_names = ["hold_or_keep_plan"] + [f"candidate_rank_{idx + 1}" for idx in range(self.candidate_count)]
        self.action_dim = len(self.action_names)
        self.state_dim = 34 + 10 * self.candidate_count
        super().__init__(cfg, rng)
        self.action_names = ["hold_or_keep_plan"] + [f"candidate_rank_{idx + 1}" for idx in range(self.candidate_count)]
        self.action_dim = len(self.action_names)
        self.state_dim = 34 + 10 * self.candidate_count
        self._candidate_cache_key: tuple[int, int] | None = None
        self._candidate_cache_indices: list[int] = []
        self._future_cache_key: int | None = None
        self._future_cache: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])

    def reset(self) -> np.ndarray:
        self._candidate_cache_key = None
        self._candidate_cache_indices = []
        self._future_cache_key = None
        self._future_cache = ([], [])
        return super().reset()

    def valid_action_indices(self) -> np.ndarray:
        if self.sim.is_done():
            return np.asarray([0], dtype=np.int64)
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        serving_sinr = float(ue.serving_sinr_db)
        candidates = self._candidate_indices()
        future_serving_min, future_best_gap = self._future_risk_summary(int(ue.serving_idx))
        radio_risk = (
            serving_sinr < self.cfg.system.q_in_db
            or self._rlf_risk() > 0.2
            or (future_serving_min < self.cfg.system.q_out_db and future_best_gap > 0.75)
        )
        rescue_required = self._serving_rescue_required(
            serving_idx=int(ue.serving_idx),
            serving_sinr=serving_sinr,
            future_serving_min=future_serving_min,
            future_best_gap=future_best_gap,
        )
        plan_actions = []
        for rank, target in enumerate(candidates[: self.candidate_count], start=1):
            if self._target_is_schedulable(int(target), serving_sinr, radio_risk):
                plan_actions.append(rank)
        valid = [0] + plan_actions
        has_useful_plan = self._has_useful_plan(int(ue.serving_idx))
        if radio_risk and plan_actions and (rescue_required or not has_useful_plan or tos_s >= self.cfg.system.min_tos_s):
            valid = plan_actions
        return np.asarray(sorted(set(valid)), dtype=np.int64)

    def _action_target(self, action_idx: int) -> int | None:
        if action_idx == 0:
            return None
        candidates = self._candidate_indices()
        rank = int(action_idx) - 1
        if rank < 0 or rank >= len(candidates):
            return None
        return int(candidates[rank])

    def _candidate_indices(self) -> list[int]:
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        cache_key = (int(self.sim.step_idx), serving_idx)
        if self._candidate_cache_key == cache_key:
            return list(self._candidate_cache_indices)
        rsrp = np.asarray(ue.rsrp_dbm, dtype=float)
        sinr = np.asarray(ue.sinr_db, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        if sinr.size <= 1:
            return []

        future_sinrs, future_mls = self._future_arrays()

        pool: list[int] = []

        def add_many(order: np.ndarray, limit: int) -> None:
            for raw_idx in order:
                idx = int(raw_idx)
                if idx == serving_idx or idx in pool:
                    continue
                pool.append(idx)
                if len(pool) >= limit:
                    break

        add_many(np.argsort(-rsrp), self.candidate_count * 4)
        add_many(np.argsort(-sinr), self.candidate_count * 3)
        add_many(np.argsort(ml_m), self.candidate_count * 3)
        for f_sinr in future_sinrs:
            add_many(np.argsort(-f_sinr), self.candidate_count * 3)

        all_nonserving = [int(i) for i in range(sinr.size) if int(i) != serving_idx]
        for idx in all_nonserving:
            if idx not in pool:
                pool.append(idx)
            if len(pool) >= self.candidate_count * 4:
                break

        scored = [(self._candidate_score(idx, sinr, rsrp, ml_m, future_sinrs, future_mls, serving_idx), idx) for idx in pool]
        scored = [(score, idx) for score, idx in scored if np.isfinite(score)]
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [idx for _, idx in scored[: self.candidate_count]]
        self._candidate_cache_key = cache_key
        self._candidate_cache_indices = list(candidates)
        return candidates

    def _future_arrays(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        cache_key = int(self.sim.step_idx)
        if self._future_cache_key == cache_key:
            return self._future_cache
        future_sinrs: list[np.ndarray] = []
        future_mls: list[np.ndarray] = []
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, f_ml = self._future_measurements(float(horizon_s))
            future_sinrs.append(np.asarray(f_sinr, dtype=float))
            future_mls.append(np.asarray(f_ml, dtype=float))
        self._future_cache_key = cache_key
        self._future_cache = (future_sinrs, future_mls)
        return self._future_cache

    def _candidate_score(
        self,
        idx: int,
        sinr: np.ndarray,
        rsrp: np.ndarray,
        ml_m: np.ndarray,
        future_sinrs: list[np.ndarray],
        future_mls: list[np.ndarray],
        serving_idx: int,
    ) -> float:
        values = [float(sinr[idx])] + [float(f_sinr[idx]) for f_sinr in future_sinrs]
        serving_values = [float(sinr[serving_idx])] + [float(f_sinr[serving_idx]) for f_sinr in future_sinrs]
        gaps = [v - s for v, s in zip(values, serving_values)]
        distances = [float(ml_m[idx])] + [float(f_ml[idx]) for f_ml in future_mls]
        vals = np.asarray(values, dtype=float)
        gap_arr = np.asarray(gaps, dtype=float)
        dist_arr = np.asarray(distances, dtype=float)
        dwell_votes = float(np.sum(vals >= self.cfg.system.q_in_db))
        near_outage_margin = float(np.min(vals) - self.cfg.system.q_out_db)
        rsrp_gap = 0.0
        if idx < rsrp.size and serving_idx < rsrp.size and np.isfinite(rsrp[idx]) and np.isfinite(rsrp[serving_idx]):
            rsrp_gap = float(rsrp[idx] - rsrp[serving_idx])
        rescue_pressure = float(
            np.clip((self.cfg.system.q_in_db - serving_values[0]) / 4.0, 0.0, 1.0)
            + np.clip((self.cfg.system.q_out_db + 0.75 - np.min(serving_values)) / 4.0, 0.0, 1.0)
            + np.clip(self._rlf_risk() / 0.55, 0.0, 1.0)
        )
        rescue_pressure = min(rescue_pressure, 1.5)
        return float(
            0.65 * np.max(gap_arr)
            + 0.35 * np.mean(gap_arr)
            + 0.35 * np.mean(vals)
            + 0.45 * np.min(vals)
            + 0.50 * min(dwell_votes, 4.0)
            + 0.18 * near_outage_margin
            + 0.40 * rsrp_gap
            + 0.20 * max(rsrp_gap, 0.0)
            + rescue_pressure
            * (
                0.90 * gap_arr[0]
                + 0.55 * np.max(gap_arr)
                + 0.35 * np.mean(gap_arr)
                + 0.35 * max(rsrp_gap, -3.0)
                + 0.45 * max(vals[0] - self.cfg.system.q_out_db, -2.0)
                + 0.30 * max(np.min(vals) - (self.cfg.system.q_out_db - 2.5), -2.0)
            )
            - 0.000012 * np.mean(dist_arr)
            - 0.12 * np.std(vals)
        )

    def _target_plan_score(self, target_idx: int) -> float:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size:
            return -float("inf")
        future_sinrs, future_mls = self._future_arrays()
        rsrp = np.asarray(ue.rsrp_dbm, dtype=float)
        return self._candidate_score(int(target_idx), sinr, rsrp, ml_m, future_sinrs, future_mls, int(ue.serving_idx))

    def _target_is_schedulable(self, target_idx: int, serving_sinr: float, radio_risk: bool) -> bool:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size or int(target_idx) == int(ue.serving_idx):
            return False
        future_sinrs, _ = self._future_arrays()
        current_gap = float(sinr[target_idx] - serving_sinr)
        future_gaps = [current_gap]
        future_values = [float(sinr[target_idx])]
        for f_sinr in future_sinrs:
            future_gaps.append(float(f_sinr[target_idx] - f_sinr[int(ue.serving_idx)]))
            future_values.append(float(f_sinr[target_idx]))
        best_future_gap = float(np.max(future_gaps))
        best_future_sinr = float(np.max(future_values))
        if current_gap >= 0.15 or best_future_gap >= 0.65:
            return True
        return bool(radio_risk and best_future_sinr >= self.cfg.system.q_out_db - 0.5 and current_gap >= -2.5)

    def _best_execution_target(self, planned_target_idx: int) -> int:
        ue = self.sim.ue
        assert ue is not None
        candidates = {int(planned_target_idx)}
        candidates.update(int(idx) for idx in self._candidate_indices())
        scored = [(self._target_plan_score(idx), idx) for idx in candidates if idx != int(ue.serving_idx)]
        scored = [(score, idx) for score, idx in scored if np.isfinite(score)]
        if not scored:
            return int(planned_target_idx)
        return int(max(scored, key=lambda item: item[0])[1])

    def _future_risk_summary(self, serving_idx: int) -> tuple[float, float]:
        future_sinrs, _ = self._future_arrays()
        if not future_sinrs:
            return 0.0, 0.0
        serving_vals = []
        best_gaps = []
        for f_sinr in future_sinrs:
            f_best = self._best_nonserving_by_sinr(f_sinr)
            serving_vals.append(float(f_sinr[int(serving_idx)]))
            best_gaps.append(float(f_sinr[f_best] - f_sinr[int(serving_idx)]))
        return float(np.min(serving_vals)), float(np.max(best_gaps))

    def _observe(self) -> np.ndarray:
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        sinr_db = np.asarray(ue.sinr_db, dtype=float)
        rsrp_dbm = np.asarray(ue.rsrp_dbm, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        serving_idx = int(ue.serving_idx)
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        time_norm = np.clip(self.sim.time_s / max(self.cfg.simulation.total_time_s, 1e-9), 0.0, 1.0)
        candidates = self._candidate_indices()
        best_sinr_idx = candidates[0] if candidates else self._best_nonserving_by_sinr(sinr_db)
        nearest_idx = self._nearest_nonserving_by_distance(ml_m)
        future_serving_min, future_best_gap = self._future_risk_summary(serving_idx)

        base = [
            self._norm_sinr(float(ue.serving_sinr_db)),
            self._norm_sinr(float(sinr_db[best_sinr_idx])),
            self._norm_gap(float(sinr_db[best_sinr_idx] - ue.serving_sinr_db)),
            self._norm_sinr(float(future_serving_min)),
            self._norm_gap(float(future_best_gap)),
            np.clip((rsrp_dbm[serving_idx] + 160.0) / 100.0, 0.0, 1.0),
            np.clip(ml_m[serving_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(ml_m[best_sinr_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(ml_m[nearest_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(current_tos_s / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            np.clip(max(0.0, self.cfg.system.min_tos_s - current_tos_s) / max(self.cfg.system.min_tos_s, 1e-9), 0.0, 1.0),
            self._rlf_risk(),
            float(ue.serving_sinr_db < self.cfg.system.q_out_db),
            float(ue.serving_sinr_db < self.cfg.system.q_in_db),
            np.clip(ue.ho_count / 24.0, 0.0, 1.0),
            np.clip(ue.uho_count / 16.0, 0.0, 1.0),
            np.clip(ue.rlf_event_count / 24.0, 0.0, 1.0),
            np.clip(ue.rb_count / 360.0, 0.0, 1.0),
            np.clip(ue.speed_mps / max(self.cfg.simulation.ue_speed_max_mps, 1.0), 0.0, 1.0),
            0.5 + 0.5 * float(np.cos(ue.heading_rad)),
            0.5 + 0.5 * float(np.sin(ue.heading_rad)),
            time_norm,
        ]

        future_sinrs, future_mls = self._future_arrays()

        candidate_features: list[float] = []
        scores = []
        for idx in candidates[: self.candidate_count]:
            score = self._candidate_score(idx, sinr_db, rsrp_dbm, ml_m, future_sinrs, future_mls, serving_idx)
            scores.append(score)
        best_score = max(scores) if scores else 0.0
        for rank in range(self.candidate_count):
            if rank >= len(candidates):
                candidate_features.extend([0.0] * 10)
                continue
            idx = int(candidates[rank])
            values = [float(sinr_db[idx])] + [float(f_sinr[idx]) for f_sinr in future_sinrs]
            serving_values = [float(sinr_db[serving_idx])] + [float(f_sinr[serving_idx]) for f_sinr in future_sinrs]
            gaps = [v - s for v, s in zip(values, serving_values)]
            distances = [float(ml_m[idx])] + [float(f_ml[idx]) for f_ml in future_mls]
            vals = np.asarray(values, dtype=float)
            gap_arr = np.asarray(gaps, dtype=float)
            dist_arr = np.asarray(distances, dtype=float)
            dwell_votes = float(np.sum(vals >= self.cfg.system.q_in_db))
            candidate_features.extend(
                [
                    1.0,
                    self._norm_sinr(float(vals[0])),
                    self._norm_gap(float(gap_arr[0])),
                    self._norm_sinr(float(np.min(vals))),
                    self._norm_sinr(float(np.mean(vals))),
                    self._norm_gap(float(np.max(gap_arr))),
                    np.clip(np.mean(dist_arr) / radius, 0.0, 5.0) / 5.0,
                    np.clip(dwell_votes / max(len(vals), 1), 0.0, 1.0),
                    float(idx == nearest_idx),
                    np.clip((scores[rank] - best_score + 10.0) / 20.0, 0.0, 1.0),
                ]
            )

        plan_features = self._plan_features(
            sinr_db=sinr_db,
            ml_m=ml_m,
            serving_idx=serving_idx,
            best_sinr_idx=best_sinr_idx,
            current_tos_s=current_tos_s,
            radius=radius,
        )
        state = base + candidate_features + plan_features
        while len(state) < self.state_dim:
            state.append(0.0)
        return np.asarray(state[: self.state_dim], dtype=np.float32)

    def _norm_sinr(self, value: float) -> float:
        lo = float(self.cfg.reward.episode_norm_sinr_min_db)
        hi = float(self.cfg.reward.episode_norm_sinr_max_db)
        return float(np.clip((float(value) - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    def _norm_gap(self, value: float) -> float:
        return float(np.clip((float(value) + 15.0) / 30.0, 0.0, 1.0))


class ImmediateCandidateHandoverEnv(CandidateAutonomousHandoverEnv):
    """Candidate-cell autonomous HO where non-hold actions execute immediately.

    This is the fully direct control variant: the DQN decides every sample
    whether to stay or move to a ranked candidate cell. UHO is still counted
    when the previous serving time is shorter than min_tos_s, but min_tos_s no
    longer blocks execution. That lets training learn the RLF-vs-UHO trade-off
    from reward instead of inheriting a hard A3-style dwell guard.
    """

    def step(self, action_idx: int) -> AutonomousHOStep:
        action_idx = int(np.clip(action_idx, 0, self.action_dim - 1))
        valid = self.valid_action_indices()
        if action_idx not in set(valid.tolist()):
            action_idx = int(valid[0])

        self._clear_plan()
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_idx_before = int(ue.serving_idx)
        prev_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        pre_sinr = float(ue.serving_sinr_db)
        pre_best_idx = self._best_nonserving_by_sinr(np.asarray(ue.sinr_db, dtype=float))
        pre_best_sinr = float(ue.sinr_db[pre_best_idx])
        pre_best_gap = float(pre_best_sinr - pre_sinr)
        pre_rlf_risk = self._rlf_risk()
        future_serving_min, future_best_gap = self._future_risk_summary(serving_idx_before)

        target_idx = self._action_target(action_idx)
        selected_target_idx = -1 if target_idx is None else int(target_idx)
        selected_target_gap = 0.0
        if target_idx is not None and int(target_idx) != serving_idx_before:
            selected_target_gap = float(ue.sinr_db[int(target_idx)] - pre_sinr)

        rescue_required = self._serving_rescue_required(
            serving_idx=serving_idx_before,
            serving_sinr=pre_sinr,
            future_serving_min=future_serving_min,
            future_best_gap=future_best_gap,
        )
        plan_event = target_idx is not None and int(target_idx) != serving_idx_before
        plan_failed = False
        emergency_exec = False
        ho_event = False
        uho_event = False
        rb_delta = 0

        if plan_event:
            if self._direct_execution_allowed(int(target_idx), rescue_required):
                ho_event = True
                uho_event = prev_tos_s < self.cfg.system.min_tos_s - 1e-6
                emergency_exec = bool(rescue_required and uho_event)
                ue.change_serving_cell(int(target_idx), self.sim.time_s)
                ue.ho_count += 1
                if uho_event:
                    ue.uho_count += 1
                rb_delta += 10
            else:
                plan_failed = True
        ue.rb_count += rb_delta

        hopp_event = self.sim.hopp_detector.record_handover(ue.serving_idx, self.sim.time_s) if ho_event else False
        self._refresh_measurements()
        rlf_event = self.sim.rlf_detector.update_rlf(
            ue,
            ue.serving_sinr_db,
            self.cfg.simulation.sample_time_s,
            self.sim.time_s,
        )
        if rlf_event:
            ue.rlf_event_count += 1

        best_idx = self._best_nonserving_by_sinr(np.asarray(ue.sinr_db, dtype=float))
        reward = self._dense_reward(
            action_idx=action_idx,
            ho_event=ho_event,
            uho_event=uho_event,
            rlf_event=bool(rlf_event),
            hopp_event=bool(hopp_event),
            rb_delta=rb_delta,
            prev_tos_s=prev_tos_s,
            pre_sinr=pre_sinr,
            pre_best_gap=pre_best_gap,
            selected_target_gap=selected_target_gap,
            pre_rlf_risk=pre_rlf_risk,
            future_serving_min=future_serving_min,
            future_best_gap=future_best_gap,
            had_plan_before=False,
            has_plan_after=False,
            plan_event=plan_event,
            plan_replaced=False,
            plan_failed=plan_failed,
            emergency_exec=emergency_exec,
        )
        self._log_step(
            action_idx=action_idx,
            target_idx=-1 if target_idx is None else int(target_idx),
            plan_event=plan_event,
            plan_failed=plan_failed,
            ho_event=ho_event,
            uho_event=uho_event,
            rlf_event=bool(rlf_event),
            hopp_event=bool(hopp_event),
            rb_delta=rb_delta,
            reward=reward,
            best_idx=best_idx,
        )
        self._advance_time()

        info = {
            "action_idx": action_idx,
            "action_name": self.action_names[action_idx],
            "selected_target_idx": int(selected_target_idx),
            "target_idx": -1 if target_idx is None else int(target_idx),
            "planned_target_idx": -1,
            "planned_action_idx": -1,
            "serving_idx_before": serving_idx_before,
            "serving_idx_after": int(ue.serving_idx),
            "plan_event": int(plan_event),
            "plan_replaced": 0,
            "plan_failed": int(plan_failed),
            "emergency_exec": int(emergency_exec),
            "ho_event": int(ho_event),
            "uho_event": int(uho_event),
            "rlf_event": int(bool(rlf_event)),
            "hopp_event": int(bool(hopp_event)),
            "rb_delta": int(rb_delta),
            "prev_tos_s": float(prev_tos_s),
            "pre_sinr_db": float(pre_sinr),
            "serving_sinr_db": float(ue.serving_sinr_db),
            "best_sinr_db": float(ue.sinr_db[best_idx]),
            "pre_best_gap_db": float(pre_best_gap),
            "selected_target_gap_db": float(selected_target_gap),
            "rlf_risk": float(pre_rlf_risk),
            "future_serving_min_sinr_db": float(future_serving_min),
            "future_best_gap_db": float(future_best_gap),
        }
        self.last_info = info
        return AutonomousHOStep(self._observe(), float(reward), self.sim.is_done(), info)

    def _direct_execution_allowed(self, target_idx: int, rescue_required: bool) -> bool:
        ue = self.sim.ue
        assert ue is not None
        sinr = np.asarray(ue.sinr_db, dtype=float)
        if target_idx < 0 or target_idx >= sinr.size or int(target_idx) == int(ue.serving_idx):
            return False
        profile = self._target_rescue_profile(target_idx)
        if not profile:
            return False
        q_out = self.cfg.system.q_out_db
        q_in = self.cfg.system.q_in_db
        current_good = profile["gap_now"] >= 0.25 and profile["target_now"] >= q_out - 1.5
        future_good = (
            profile["target_min"] >= q_out - 2.0
            and profile["gap_mean"] >= 0.25
            and profile["target_mean"] >= q_in - 1.0
        )
        rescue_good = (
            rescue_required
            and profile["target_now"] >= q_out - 2.0
            and (
                profile["gap_now"] >= 0.5
                or profile["target_min"] > profile["serving_min"] + 1.5
                or profile["gap_mean"] >= 0.75
            )
        )
        critical_rescue = (
            rescue_required
            and (self._rlf_risk() >= 0.55 or float(ue.serving_sinr_db) < q_out)
            and (
                profile["gap_now"] >= 0.25
                or profile["gap_mean"] >= 0.5
                or profile["target_min"] > profile["serving_min"] + 1.0
            )
        )
        return bool(current_good or future_good or rescue_good or critical_rescue)
