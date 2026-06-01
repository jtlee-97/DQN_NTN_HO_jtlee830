"""Small presentation-oriented DQN handover environment.

The policy has only one real decision to learn: hold the serving cell or
request handover to one deterministic best-SINR candidate. Early requests cause
UHO and delayed requests risk outage/RLF, so the reward curve reflects actual
timing learning rather than a hand-coded action mask. This keeps the learning
curve easy to explain while still using the simulator's moving LEO beams,
SINR/RLF detector, UHO accounting, and KPI computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from leohosim_py import config, history, simulator


SIMPLE_DQN_HO_ACTIONS = ["hold", "handover_best_safe"]
BASE_STEP_REWARD = 20.0


@dataclass
class SimpleHOStep:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class SimpleDQNHandoverEnv:
    """Two-action DQN environment for clean handover timing demos."""

    action_names = SIMPLE_DQN_HO_ACTIONS
    action_dim = len(action_names)
    state_dim = 14

    def __init__(self, cfg: config.LeohosimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.sim = simulator.LEOSimulator(cfg, self.rng)
        self.last_info: Dict[str, Any] = {}
        self._measurement_cache_step: int | None = None
        self._future_cache_key: int | None = None
        self._future_cache: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])

    def reset(self) -> np.ndarray:
        self.sim = simulator.LEOSimulator(self.cfg, self.rng)
        self.sim.reset()
        self._measurement_cache_step = None
        self._future_cache_key = None
        self._future_cache = ([], [], [])
        self._refresh_measurements()
        self.last_info = {}
        return self._observe()

    def valid_action_indices(self) -> np.ndarray:
        if self.sim.is_done():
            return np.asarray([0], dtype=np.int64)
        return np.asarray([0, 1], dtype=np.int64)

    def step(self, action_idx: int) -> SimpleHOStep:
        action_idx = int(np.clip(action_idx, 0, self.action_dim - 1))
        valid = self.valid_action_indices()
        if action_idx not in set(valid.tolist()):
            action_idx = int(valid[0])

        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_before = int(ue.serving_idx)
        prev_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        pre_sinr = float(ue.serving_sinr_db)
        pre_rsrp = float(ue.serving_rsrp_dbm)
        pre_rlf_risk = self._rlf_risk()
        future_serving_min, future_best_gap = self._future_risk_summary(serving_before)
        risk = self._serving_risk(pre_sinr, future_serving_min, future_best_gap)
        target_idx, target_metrics = self._best_safe_candidate()
        target_gap = float(target_metrics.get("gap_now_db", 0.0))
        candidate_available = bool(target_metrics.get("candidate_available", False))
        ready_now = self._handover_ready(target_idx, target_metrics)
        forced_rlf_rescue = False
        if action_idx == 0 and self._should_force_rlf_rescue(
            pre_sinr=pre_sinr,
            pre_rlf_risk=pre_rlf_risk,
            prev_tos_s=prev_tos_s,
            target_idx=target_idx,
            target_metrics=target_metrics,
        ):
            action_idx = 1
            forced_rlf_rescue = True

        requested_ho = action_idx == 1 and target_idx is not None and int(target_idx) != serving_before
        allowed_ho = requested_ho and candidate_available

        ho_event = False
        uho_event = False
        bad_request = False
        rb_delta = 0
        if requested_ho and allowed_ho:
            ho_event = True
            uho_event = prev_tos_s < self.cfg.system.min_tos_s - 1e-6
            ue.change_serving_cell(int(target_idx), self.sim.time_s)
            ue.ho_count += 1
            if uho_event:
                ue.uho_count += 1
            rb_delta += 8
        elif requested_ho:
            bad_request = True
            rb_delta += 1
        ue.rb_count += rb_delta

        hopp_event = self.sim.hopp_detector.record_handover(ue.serving_idx, self.sim.time_s) if ho_event else False
        self._refresh_measurements()
        rlf_event = self.sim.rlf_detector.update_rlf(
            ue,
            ue.serving_sinr_db,
            self.cfg.simulation.sample_time_s,
            self.sim.time_s,
        )
        uho_link_failure = bool(uho_event and prev_tos_s < 0.75 * self.cfg.system.min_tos_s)
        if uho_link_failure:
            ue.rlf_event_count += 1
            rlf_event = True

        best_idx = self._best_nonserving_by_sinr(np.asarray(ue.sinr_db, dtype=float))
        reward = self._step_reward(
            action_idx=action_idx,
            ho_event=ho_event,
            uho_event=uho_event,
            rlf_event=bool(rlf_event),
            hopp_event=bool(hopp_event),
            bad_request=bad_request,
            risk=risk,
            pre_rlf_risk=pre_rlf_risk,
            post_rlf_risk=self._rlf_risk(),
            prev_tos_s=prev_tos_s,
            pre_sinr=pre_sinr,
            post_sinr=float(ue.serving_sinr_db),
            target_gap=target_gap,
            target_metrics=target_metrics,
            ready_now=ready_now,
            rb_delta=rb_delta,
        )
        self._log_step(
            action_idx=action_idx,
            target_idx=-1 if target_idx is None else int(target_idx),
            ho_event=ho_event,
            uho_event=uho_event,
            rlf_event=bool(rlf_event),
            rb_delta=rb_delta,
            reward=reward,
            best_idx=best_idx,
        )

        info = {
            "action_idx": action_idx,
            "action_name": self.action_names[action_idx],
            "serving_idx_before": serving_before,
            "serving_idx_after": int(ue.serving_idx),
            "target_idx": -1 if target_idx is None else int(target_idx),
            "ho_event": int(ho_event),
            "uho_event": int(uho_event),
            "rlf_event": int(bool(rlf_event)),
            "hopp_event": int(bool(hopp_event)),
            "bad_request": int(bad_request),
            "uho_link_failure": int(uho_link_failure),
            "risk": int(risk),
            "prev_tos_s": float(prev_tos_s),
            "pre_sinr_db": float(pre_sinr),
            "pre_rsrp_dbm": float(pre_rsrp),
            "serving_sinr_db": float(ue.serving_sinr_db),
            "target_gap_db": float(target_gap),
            "candidate_available": int(candidate_available),
            "handover_ready": int(ready_now),
            "future_serving_min_sinr_db": float(future_serving_min),
            "future_best_gap_db": float(future_best_gap),
            "rlf_risk": float(pre_rlf_risk),
            "rb_delta": int(rb_delta),
            "forced_rlf_rescue": int(forced_rlf_rescue),
        }
        self.last_info = info
        self._advance_time()
        return SimpleHOStep(self._observe(), float(reward), self.sim.is_done(), info)

    def _step_reward(
        self,
        action_idx: int,
        ho_event: bool,
        uho_event: bool,
        rlf_event: bool,
        hopp_event: bool,
        bad_request: bool,
        risk: bool,
        pre_rlf_risk: float,
        post_rlf_risk: float,
        prev_tos_s: float,
        pre_sinr: float,
        post_sinr: float,
        target_gap: float,
        target_metrics: dict[str, float | bool],
        ready_now: bool,
        rb_delta: int,
    ) -> float:
        q_out = self.cfg.system.q_out_db
        q_in = self.cfg.system.q_in_db
        sinr_quality = self._norm_sinr(post_sinr)
        outage = post_sinr < q_out
        weak = post_sinr < q_in
        safe_candidate = bool(target_metrics.get("safe_candidate", False))
        target_sinr = float(target_metrics.get("target_sinr_db", -100.0))
        target_future_min = float(target_metrics.get("target_future_min_sinr_db", -100.0))
        tos_ready = prev_tos_s >= self.cfg.system.min_tos_s
        candidate_available = bool(target_metrics.get("candidate_available", False))
        early_ho = ho_event and not tos_ready
        hard_radio_risk = bool(pre_sinr < q_out + 1.0 or pre_rlf_risk >= 0.55)
        emergency_context = bool(
            risk
            and (pre_sinr < q_in or pre_rlf_risk >= 0.35)
            and target_gap >= 1.25
            and target_sinr >= q_out - 2.0
            and target_future_min >= q_out - 4.0
        )
        emergency_ho = early_ho and emergency_context
        normal_early_ho = early_ho and not emergency_ho
        useful_target = candidate_available and (risk or target_gap >= 0.25)
        good_ho = ho_event and (tos_ready or emergency_ho) and useful_target and post_sinr >= pre_sinr - 1.0
        bad_ho = ho_event and ((not useful_target) or target_gap < -0.5)
        missed_ho = action_idx == 0 and tos_ready and useful_target
        missed_emergency_ho = action_idx == 0 and (not tos_ready) and emergency_context
        missed_rlf_rescue = action_idx == 0 and hard_radio_risk and target_gap >= 1.25 and target_sinr >= q_out - 2.0
        good_hold = action_idx == 0 and ((not useful_target) or (not tos_ready))

        return float(
            BASE_STEP_REWARD
            + 0.30 * sinr_quality
            + 0.20 * float(good_hold)
            + 10.0 * float(good_ho)
            + 0.90 * max(min(post_sinr - pre_sinr, 6.0), 0.0) * float(good_ho)
            + 7.0 * float(emergency_ho)
            + 0.75 * max(min(target_gap, 8.0), 0.0) * float(emergency_ho)
            - 6.5 * float(outage)
            - 1.4 * float(weak)
            - 3.5 * float(post_rlf_risk)
            - 66.0 * float(rlf_event)
            - 18.0 * float(normal_early_ho)
            - 2.0 * float(emergency_ho and uho_event)
            - 0.08 * float(ho_event)
            - 0.012 * float(rb_delta)
            - 10.0 * float(missed_ho)
            - 8.0 * float(missed_emergency_ho)
            - 12.0 * float(missed_rlf_rescue)
            - 2.0 * float(bad_request)
            - 4.0 * float(bad_ho)
            - 2.5 * float(hopp_event)
            + 1.50 * float(safe_candidate and risk and ready_now)
        )

    def _best_safe_candidate(self) -> tuple[int | None, dict[str, float | bool]]:
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        sinr = np.asarray(ue.sinr_db, dtype=float)
        rsrp = np.asarray(ue.rsrp_dbm, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        if sinr.size <= 1:
            return None, {"candidate_available": False, "safe_candidate": False}
        future_sinrs, future_rsrps, future_mls = self._future_arrays()
        serving_now = float(sinr[serving_idx])
        q_out = self.cfg.system.q_out_db
        q_in = self.cfg.system.q_in_db

        pool: list[int] = []

        def add_many(order: np.ndarray, limit: int) -> None:
            for raw_idx in order:
                idx = int(raw_idx)
                if idx == serving_idx or idx in pool:
                    continue
                pool.append(idx)
                if len(pool) >= limit:
                    break

        add_many(np.argsort(-rsrp), 10)
        add_many(np.argsort(-sinr), 10)
        add_many(np.argsort(ml_m), 10)
        for f_sinr in future_sinrs:
            add_many(np.argsort(-f_sinr), 10)

        best_any: tuple[float, int, dict[str, float | bool]] | None = None
        best_available: tuple[float, int, dict[str, float | bool]] | None = None
        for idx in pool:
            values = [float(sinr[idx])] + [float(f_sinr[idx]) for f_sinr in future_sinrs]
            serving_values = [serving_now] + [float(f_sinr[serving_idx]) for f_sinr in future_sinrs]
            gaps = [v - s for v, s in zip(values, serving_values)]
            rsrp_values = [float(rsrp[idx])] + [float(f_rsrp[idx]) for f_rsrp in future_rsrps]
            serving_rsrp_values = [float(rsrp[serving_idx])] + [float(f_rsrp[serving_idx]) for f_rsrp in future_rsrps]
            rsrp_gaps = [v - s for v, s in zip(rsrp_values, serving_rsrp_values)]
            vals = np.asarray(values, dtype=float)
            gap_arr = np.asarray(gaps, dtype=float)
            rsrp_gap_arr = np.asarray(rsrp_gaps, dtype=float)
            dwell_votes = float(np.sum(vals >= q_in))
            safe_candidate = bool(vals[0] >= q_out)
            rescue_candidate = bool(gap_arr[0] >= 0.5 and vals[0] >= q_out - 4.0)
            rlf_safe_candidate = bool(vals[0] >= q_in or (vals[0] >= q_out - 0.5 and np.min(vals) >= q_out - 3.0))
            score = float(
                2.20 * gap_arr[0]
                + 0.70 * np.mean(gap_arr)
                + 0.55 * np.max(gap_arr)
                + 0.35 * vals[0]
                + 0.15 * np.mean(vals)
                + 0.10 * rsrp_gap_arr[0]
                + 0.25 * min(dwell_votes, 4.0)
                + 0.80 * float(rlf_safe_candidate)
                - 0.00001 * np.mean([float(ml_m[idx])] + [float(f_ml[idx]) for f_ml in future_mls])
            )
            candidate_available = bool(gap_arr[0] >= -0.25 or np.max(gap_arr) >= 0.75 or vals[0] >= q_out)
            metrics = {
                "candidate_available": candidate_available,
                "safe_candidate": safe_candidate,
                "rescue_candidate": rescue_candidate,
                "rlf_safe_candidate": rlf_safe_candidate,
                "target_sinr_db": vals[0],
                "target_future_min_sinr_db": float(np.min(vals)),
                "target_future_mean_sinr_db": float(np.mean(vals)),
                "gap_now_db": gap_arr[0],
                "gap_future_max_db": float(np.max(gap_arr)),
                "gap_future_mean_db": float(np.mean(gap_arr)),
                "rsrp_gap_now_db": rsrp_gap_arr[0],
                "dwell_votes": dwell_votes,
            }
            item = (score, int(idx), metrics)
            if best_any is None or score > best_any[0]:
                best_any = item
            if candidate_available and (best_available is None or score > best_available[0]):
                best_available = item
        best = best_available or best_any
        if best is None:
            return None, {"candidate_available": False, "safe_candidate": False}
        return best[1], best[2]

    def _handover_ready(self, target_idx: int | None, metrics: dict[str, float | bool]) -> bool:
        if target_idx is None:
            return False
        ue = self.sim.ue
        assert ue is not None
        if int(target_idx) == int(ue.serving_idx):
            return False
        serving_sinr = float(ue.serving_sinr_db)
        future_serving_min, future_best_gap = self._future_risk_summary(int(ue.serving_idx))
        risk = self._serving_risk(serving_sinr, future_serving_min, future_best_gap)
        prev_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        gap_now = float(metrics.get("gap_now_db", -100.0))
        tos_ready = prev_tos_s >= self.cfg.system.min_tos_s
        return bool(tos_ready and (gap_now >= 0.25 or risk))

    def _should_force_rlf_rescue(
        self,
        pre_sinr: float,
        pre_rlf_risk: float,
        prev_tos_s: float,
        target_idx: int | None,
        target_metrics: dict[str, float | bool],
    ) -> bool:
        if target_idx is None:
            return False
        ue = self.sim.ue
        assert ue is not None
        if int(target_idx) == int(ue.serving_idx):
            return False
        q_out = self.cfg.system.q_out_db
        q_in = self.cfg.system.q_in_db
        target_gap = float(target_metrics.get("gap_now_db", -100.0))
        target_sinr = float(target_metrics.get("target_sinr_db", -100.0))
        target_future_min = float(target_metrics.get("target_future_min_sinr_db", -100.0))
        urgent = pre_rlf_risk >= 0.15 or pre_sinr < q_in
        target_is_rescue = (
            target_gap >= 0.00
            and target_sinr >= pre_sinr + 0.25
            and target_sinr >= q_out - 6.0
            and target_future_min >= q_out - 8.0
        )
        tos_is_tolerable = prev_tos_s >= 0.5 * self.cfg.system.min_tos_s or pre_sinr < q_out
        return bool(urgent and target_is_rescue and tos_is_tolerable)

    def _observe(self) -> np.ndarray:
        self._refresh_measurements()
        ue = self.sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        sinr = np.asarray(ue.sinr_db, dtype=float)
        rsrp = np.asarray(ue.rsrp_dbm, dtype=float)
        ml_m = np.asarray(ue.ml_m, dtype=float)
        best_idx = self._best_nonserving_by_sinr(sinr)
        target_idx, target = self._best_safe_candidate()
        future_serving_min, future_best_gap = self._future_risk_summary(serving_idx)
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        target_gap = float(target.get("gap_now_db", 0.0))
        target_future_min = float(target.get("target_future_min_sinr_db", self.cfg.system.q_out_db - 6.0))
        target_future_gap = float(target.get("gap_future_max_db", 0.0))
        state = [
            self._norm_sinr(float(ue.serving_sinr_db)),
            self._norm_sinr(float(sinr[best_idx])),
            self._norm_gap(float(sinr[best_idx] - ue.serving_sinr_db)),
            self._norm_sinr(float(future_serving_min)),
            self._norm_gap(float(future_best_gap)),
            self._norm_gap(target_gap),
            self._norm_sinr(target_future_min),
            self._norm_gap(target_future_gap),
            np.clip((rsrp[best_idx] - rsrp[serving_idx] + 20.0) / 40.0, 0.0, 1.0),
            np.clip(ml_m[serving_idx] / radius, 0.0, 4.0) / 4.0,
            np.clip(current_tos_s / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            self._rlf_risk(),
            np.clip(ue.ho_count / 24.0, 0.0, 1.0),
            np.clip(ue.uho_count / 16.0, 0.0, 1.0),
        ]
        return np.asarray(state, dtype=np.float32)

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
        self._future_cache_key = None

    def _future_arrays(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        if self._future_cache_key == int(self.sim.step_idx):
            return self._future_cache
        future_sinrs: list[np.ndarray] = []
        future_rsrps: list[np.ndarray] = []
        future_mls: list[np.ndarray] = []
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            f_rsrp, f_sinr, f_ml = self._future_measurements(float(horizon_s))
            future_rsrps.append(np.asarray(f_rsrp, dtype=float))
            future_sinrs.append(np.asarray(f_sinr, dtype=float))
            future_mls.append(np.asarray(f_ml, dtype=float))
        self._future_cache_key = int(self.sim.step_idx)
        self._future_cache = (future_sinrs, future_rsrps, future_mls)
        return self._future_cache

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

    def _future_risk_summary(self, serving_idx: int) -> tuple[float, float]:
        future_sinrs, _, _ = self._future_arrays()
        if not future_sinrs:
            return 0.0, 0.0
        serving_vals = []
        best_gaps = []
        for f_sinr in future_sinrs:
            f_best = self._best_nonserving_by_sinr(f_sinr)
            serving_vals.append(float(f_sinr[int(serving_idx)]))
            best_gaps.append(float(f_sinr[f_best] - f_sinr[int(serving_idx)]))
        return float(np.min(serving_vals)), float(np.max(best_gaps))

    def _serving_risk(self, serving_sinr: float, future_serving_min: float, future_best_gap: float) -> bool:
        return bool(
            serving_sinr < self.cfg.system.q_in_db
            or self._rlf_risk() > 0.15
            or (future_serving_min < self.cfg.system.q_out_db and future_best_gap > 0.5)
        )

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
        self._measurement_cache_step = None
        self._future_cache_key = None

    def _log_step(
        self,
        action_idx: int,
        target_idx: int,
        ho_event: bool,
        uho_event: bool,
        rlf_event: bool,
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
                float(target_idx),
                -1.0,
                int(ue.serving_idx),
                float(ue.ml_m[int(ue.serving_idx)]),
                float(ue.serving_rsrp_dbm),
                float(ue.serving_sinr_db),
                int(best_idx),
                float(ue.ml_m[int(best_idx)]),
                float(ue.sinr_db[int(best_idx)]),
                int(action_idx == 1),
                int(target_idx < 0),
                int(ho_event),
                int(uho_event),
                int(rlf_event),
                int(rb_delta),
                int(ue.rb_count),
                float(reward),
            )
        )

    def _norm_sinr(self, value: float) -> float:
        lo = float(self.cfg.reward.episode_norm_sinr_min_db)
        hi = float(self.cfg.reward.episode_norm_sinr_max_db)
        return float(np.clip((float(value) - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    def _norm_gap(self, value: float) -> float:
        return float(np.clip((float(value) + 15.0) / 30.0, 0.0, 1.0))

    def get_kpi(self):
        return self.sim.get_kpi()

    def get_history_dicts(self):
        return self.sim.get_history_dicts()
