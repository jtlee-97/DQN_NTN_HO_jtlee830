"""Predictive direct-handover RL environment.

This environment gives the agent direct handover authority instead of asking it
to pick indirect Event D2 thresholds. The action therefore has identifiable
causal impact on the next serving cell and local radio outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from leohosim_py import config, history, simulator


@dataclass
class DirectHOStep:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class PredictiveHandoverEnv:
    """Gym-like environment for predictive stay/HO-target selection.

    Actions:
    0. stay on serving cell
    1. hand over to current best non-serving cell
    2. hand over to best predicted cell at horizon[0]
    3. hand over to best predicted cell at horizon[1]
    4. hand over to best predicted cell at horizon[2]
    """

    state_dim = 40
    action_dim = 5

    def __init__(self, cfg: config.LeohosimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.sim = simulator.LEOSimulator(cfg, self.rng)

    def reset(self) -> np.ndarray:
        self.sim = simulator.LEOSimulator(self.cfg, self.rng)
        self.sim.reset()
        self._refresh_measurements()
        return self._observe()

    def step(self, action_idx: int) -> DirectHOStep:
        action_idx = int(action_idx)
        ue = self.sim.ue
        assert ue is not None
        self._refresh_measurements()
        ml_m = np.asarray(ue.ml_m, dtype=float)
        sinr_db = np.asarray(ue.sinr_db, dtype=float)
        target_idx = self._action_target(action_idx)

        prev_serving = int(ue.serving_idx)
        prev_tos_s = self.sim.time_s - ue.serving_start_time_s
        ho_event = False
        uho_event = False
        rb_delta = 0
        if target_idx is not None and target_idx != prev_serving:
            ho_event = True
            uho_event = prev_tos_s < self.cfg.system.min_tos_s
            ue.change_serving_cell(int(target_idx), self.sim.time_s)
            ue.ho_count += 1
            if uho_event:
                ue.uho_count += 1
            rb_delta = 7
            ue.rb_count += rb_delta

        hopp_event = self.sim.hopp_detector.record_handover(ue.serving_idx, self.sim.time_s) if ho_event else False
        ue.update_serving_metrics()
        rlf_event = self.sim.rlf_detector.update_rlf(
            ue,
            ue.serving_sinr_db,
            self.cfg.simulation.sample_time_s,
            self.sim.time_s,
        )

        best_idx = self._best_nonserving_index(ml_m)
        reward = self._reward(action_idx, ho_event, uho_event, rlf_event, hopp_event, rb_delta, prev_tos_s)
        self.sim.sim_history.log_step(
            history.HistoryEntry(
                self.sim.time_s,
                self.sim.step_idx,
                action_idx,
                -1.0,
                -1.0,
                ue.serving_idx,
                float(ml_m[ue.serving_idx]),
                ue.serving_rsrp_dbm,
                ue.serving_sinr_db,
                best_idx,
                float(ml_m[best_idx]),
                float(sinr_db[best_idx]),
                int(ho_event),
                0,
                int(ho_event),
                int(uho_event),
                int(rlf_event),
                rb_delta,
                ue.rb_count,
                reward,
            )
        )

        self.sim.beam_centers[:, 1] += self.cfg.system.sat_speed_mps * self.cfg.simulation.sample_time_s
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ue.move_linear(self.cfg.simulation.sample_time_s)
        self.sim.time_s += self.cfg.simulation.sample_time_s
        self.sim.step_idx += 1
        info = {
            "action_idx": action_idx,
            "target_idx": -1 if target_idx is None else int(target_idx),
            "ho_event": ho_event,
            "uho_event": uho_event,
            "rlf_event": rlf_event,
            "hopp_event": hopp_event,
            "serving_sinr_db": ue.serving_sinr_db,
            "best_sinr_db": float(sinr_db[best_idx]),
            "prev_tos_s": float(prev_tos_s),
            "rb_delta": rb_delta,
        }
        return DirectHOStep(self._observe(), float(reward), self.sim.is_done(), info)

    def valid_action_indices(self) -> np.ndarray:
        ue = self.sim.ue
        assert ue is not None
        self._refresh_measurements()
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        serving_sinr = float(ue.serving_sinr_db)
        target_sinrs = np.asarray(ue.sinr_db, dtype=float).copy()
        target_sinrs[int(ue.serving_idx)] = -np.inf
        best_gap = float(np.max(target_sinrs) - serving_sinr)
        if current_tos_s < max(self.cfg.system.min_tos_s, 1.5):
            return np.asarray([0], dtype=np.int64)
        if serving_sinr >= self.cfg.system.q_in_db + 1.0:
            return np.asarray([0], dtype=np.int64)
        rlf_risk = float(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9))
        if rlf_risk > 0.45 and best_gap > 0.25:
            return np.asarray([1, 2, 3, 4], dtype=np.int64)
        if best_gap < 2.0 and serving_sinr >= self.cfg.system.q_out_db:
            return np.asarray([0], dtype=np.int64)
        if serving_sinr < self.cfg.system.q_out_db + 1.0 and best_gap > 0.5:
            return np.asarray([1, 2, 3, 4], dtype=np.int64)
        return np.arange(self.action_dim, dtype=np.int64)

    def _reward(
        self,
        action_idx: int,
        ho_event: bool,
        uho_event: bool,
        rlf_event: bool,
        hopp_event: bool,
        rb_delta: int,
        prev_tos_s: float,
    ) -> float:
        ue = self.sim.ue
        assert ue is not None
        sinr = float(ue.serving_sinr_db)
        target_sinrs = np.asarray(ue.sinr_db, dtype=float).copy()
        target_sinrs[int(ue.serving_idx)] = -np.inf
        best_gap = float(np.max(target_sinrs) - sinr)
        sinr_norm = float(np.clip((sinr + 20.0) / 25.0, 0.0, 1.5))
        outage = sinr < self.cfg.system.q_out_db
        weak = sinr < self.cfg.system.q_in_db
        rlf_risk = float(np.clip(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9), 0.0, 1.0))
        unnecessary_ho = ho_event and sinr >= self.cfg.system.q_in_db + 2.0 and best_gap < 1.0
        missed_ho = action_idx == 0 and sinr < self.cfg.system.q_in_db and best_gap > 1.5
        return float(
            3.0 * sinr_norm
            - 6.0 * float(outage)
            - 1.5 * float(weak)
            - 45.0 * float(rlf_event)
            - 4.0 * rlf_risk
            - 50.0 * float(uho_event)
            - 6.0 * float(hopp_event)
            - 0.8 * float(ho_event)
            - 0.03 * float(rb_delta)
            - 6.0 * float(unnecessary_ho)
            - 3.0 * float(missed_ho)
            + 0.08 * min(max(self.sim.time_s - ue.serving_start_time_s, 0.0), 5.0)
            - 20.0 * float(ho_event and 0.0 <= prev_tos_s < self.cfg.system.min_tos_s)
        )

    def _action_target(self, action_idx: int) -> int | None:
        if action_idx == 0:
            return None
        if action_idx == 1:
            return self._best_nonserving_index(np.asarray(self.sim.ue.ml_m, dtype=float))
        horizons = list(self.cfg.simulation.lookahead_horizons_s)
        h = horizons[min(max(action_idx - 2, 0), len(horizons) - 1)] if horizons else 1.0
        _, _, ml_m = self._future_measurements(float(h))
        return self._best_nonserving_index(ml_m)

    def _best_nonserving_index(self, ml_m: np.ndarray) -> int:
        serving_idx = int(self.sim.ue.serving_idx)
        order = [int(idx) for idx in np.argsort(ml_m) if int(idx) != serving_idx]
        return order[0] if order else serving_idx

    def _future_measurements(self, horizon_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ue = self.sim.ue
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
        assert ue is not None
        self._refresh_measurements()
        ml_m = np.asarray(ue.ml_m, dtype=float)
        sinr_db = np.asarray(ue.sinr_db, dtype=float)
        rsrp_dbm = np.asarray(ue.rsrp_dbm, dtype=float)
        serving_idx = int(ue.serving_idx)
        order = [int(idx) for idx in np.argsort(ml_m) if int(idx) != serving_idx]
        best_idx = order[0] if order else serving_idx
        second_idx = order[1] if len(order) > 1 else best_idx
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        current_tos_s = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        base = [
            np.clip((ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
            np.clip((sinr_db[best_idx] + 20.0) / 40.0, 0.0, 1.0),
            np.clip((sinr_db[best_idx] - ue.serving_sinr_db + 20.0) / 40.0, 0.0, 1.0),
            np.clip((ue.serving_rsrp_dbm + 160.0) / 100.0, 0.0, 1.0),
            np.clip((rsrp_dbm[best_idx] + 160.0) / 100.0, 0.0, 1.0),
            np.clip(ml_m[serving_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(ml_m[best_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(ml_m[second_idx] / radius, 0.0, 5.0) / 5.0,
            np.clip(current_tos_s / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            np.clip(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9), 0.0, 1.0),
            np.clip(ue.ho_count / 20.0, 0.0, 1.0),
            np.clip(ue.rlf_event_count / 20.0, 0.0, 1.0),
            np.clip(ue.rb_count / 250.0, 0.0, 1.0),
            np.clip(ue.speed_mps / max(self.cfg.simulation.ue_speed_max_mps, 1.0), 0.0, 1.0),
            0.5 + 0.5 * float(np.cos(ue.heading_rad)),
            0.5 + 0.5 * float(np.sin(ue.heading_rad)),
        ]
        future: list[float] = []
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            _, f_sinr, f_ml = self._future_measurements(float(horizon_s))
            f_best = self._best_nonserving_index(f_ml)
            future.extend(
                [
                    np.clip((f_sinr[serving_idx] + 20.0) / 40.0, 0.0, 1.0),
                    np.clip((f_sinr[f_best] + 20.0) / 40.0, 0.0, 1.0),
                    np.clip((f_sinr[f_best] - f_sinr[serving_idx] + 20.0) / 40.0, 0.0, 1.0),
                    np.clip(f_ml[serving_idx] / radius, 0.0, 5.0) / 5.0,
                    np.clip(f_ml[f_best] / radius, 0.0, 5.0) / 5.0,
                    float(f_best == best_idx),
                ]
            )
        state = base + future
        while len(state) < self.state_dim:
            state.append(0.0)
        return np.asarray(state[: self.state_dim], dtype=np.float32)

    def get_kpi(self):
        return self.sim.get_kpi()

    def get_history_dicts(self):
        return self.sim.get_history_dicts()
