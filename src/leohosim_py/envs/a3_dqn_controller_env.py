"""Event-level DQN controller wrapped around an A3-RSRP handover baseline.

The baseline is conventional A3-RSRP. The proposed controller is called only at
A3/risk events and decides whether to suppress, allow, accelerate, or defer a
handover using mobility and distance-prediction features.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from leohosim_py import config, handover, history, simulator


ACTION_NAMES = [
    "conservative_a3",
    "normal_a3",
    "aggressive_a3",
    "defer_a3",
    "dwell_guarded_a3",
]


@dataclass
class A3DQNEventStep:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class LocalKPI:
    score: float
    avg_sinr_db: float
    outage_time_s: float
    rlf_count: int
    uho_count: int
    ho_count: int
    rb_count: int
    avg_tos_delta_s: float
    components: Dict[str, float]


class MobilityAwareA3DQNEnv:
    """A3 admission/timing controller.

    Actions:
    0. conservative_a3: raise A3 offset/TTT for UHO/RB stability.
    1. normal_a3: use baseline A3-RSRP parameters.
    2. aggressive_a3: lower A3 offset/TTT for serving-link risk.
    3. defer_a3: suppress briefly, then use normal A3.
    4. dwell_guarded_a3: use normal A3 only when predicted dwell is stable.
    """

    state_dim = 34
    action_dim = len(ACTION_NAMES)

    def __init__(
        self,
        cfg: config.LeohosimConfig,
        rng: np.random.Generator | None = None,
        decision_window_s: float = 2.0,
        defer_s: float = 0.4,
        max_skip_s: float = 8.0,
        min_predicted_dwell_s: float = 1.6,
        dwell_margin_m: float = 1200.0,
        aggressive_guard_mode: str = "mintos",
    ):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.decision_window_s = float(decision_window_s)
        self.defer_s = float(defer_s)
        self.max_skip_s = float(max_skip_s)
        self.min_predicted_dwell_s = float(min_predicted_dwell_s)
        self.dwell_margin_m = float(dwell_margin_m)
        self.aggressive_guard_mode = str(aggressive_guard_mode)
        self.sim = simulator.LEOSimulator(cfg, self.rng)
        self.last_info: Dict[str, Any] = {}
        self.last_action_details: Dict[str, Any] = {}

    def reset(self) -> np.ndarray:
        self.sim = simulator.LEOSimulator(self.cfg, self.rng)
        self.sim.reset()
        self.last_info = {}
        self.last_action_details = {}
        self._advance_to_decision()
        return self._observe()

    def step(self, action_idx: int) -> A3DQNEventStep:
        action_idx = int(np.clip(action_idx, 0, self.action_dim - 1))
        snapshot = copy.deepcopy(self.sim)
        baseline = self._rollout(copy.deepcopy(snapshot), action_idx=1, horizon_s=self.decision_window_s)
        controlled = self._rollout(copy.deepcopy(snapshot), action_idx=action_idx, horizon_s=self.decision_window_s)
        reward = float(controlled.score - baseline.score)

        before = self._counter_snapshot(self.sim)
        self._execute_action_window(self.sim, action_idx, self.decision_window_s)
        after = self._counter_snapshot(self.sim)
        details = dict(self.last_action_details)
        self._advance_to_decision()

        info = {
            "action_idx": action_idx,
            "action_name": ACTION_NAMES[action_idx],
            "baseline_score": baseline.score,
            "controlled_score": controlled.score,
            "relative_reward": reward,
            "event_ho_count": after["ho_count"] - before["ho_count"],
            "event_rlf_count": after["rlf_count"] - before["rlf_count"],
            "event_uho_count": after["uho_count"] - before["uho_count"],
            "event_rb_count": after["rb_count"] - before["rb_count"],
            "baseline_avg_sinr_db": baseline.avg_sinr_db,
            "controlled_avg_sinr_db": controlled.avg_sinr_db,
            "baseline_outage_time_s": baseline.outage_time_s,
            "controlled_outage_time_s": controlled.outage_time_s,
        }
        for key, value in controlled.components.items():
            info[f"reward_component_delta_{key}"] = float(value - baseline.components.get(key, 0.0))
        info.update(details)
        self.last_info = info
        return A3DQNEventStep(self._observe(), reward, self.sim.is_done(), info)

    def valid_action_indices(self) -> np.ndarray:
        if self.sim.is_done():
            return np.asarray([0], dtype=np.int64)
        valid = [0, 1, 3, 4]
        if self._aggressive_a3_guard(self.sim):
            valid.append(2)
        return np.asarray(sorted(valid), dtype=np.int64)

    def teacher_action(self) -> int:
        """Fast rule teacher used only for behavior-cloning warmup.

        The teacher avoids expensive all-action rollouts. It demonstrates the
        intended controller shape: protect risky serving links, suppress
        unstable A3 handovers, and otherwise allow the conventional baseline.
        """
        serving_risk = self._serving_risk(self.sim)
        a3_target, a3_margin = self._a3_candidate(self.sim)
        a3_dwell = self._predicted_dwell_s(self.sim, int(a3_target)) if a3_target is not None else 0.0
        if serving_risk:
            return 2 if self._aggressive_a3_guard(self.sim) else 1
        if a3_margin >= 0.0 and a3_dwell < self.min_predicted_dwell_s:
            return 0
        if a3_margin >= 0.0:
            return 1
        return 4

    def score_valid_actions(self) -> Dict[int, LocalKPI]:
        snapshot = copy.deepcopy(self.sim)
        return {
            int(action): self._rollout(copy.deepcopy(snapshot), int(action), self.decision_window_s)
            for action in self.valid_action_indices()
        }

    def _advance_to_decision(self) -> None:
        max_steps = max(1, int(round(self.max_skip_s / self.cfg.simulation.sample_time_s)))
        steps = 0
        while not self.sim.is_done() and not self._is_decision_state(self.sim) and steps < max_steps:
            self._controlled_step(self.sim, "a3", action_idx=-1)
            steps += 1

    def _is_decision_state(self, sim: simulator.LEOSimulator) -> bool:
        if sim.is_done():
            return True
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        a3_target, a3_margin = self._a3_candidate(sim)
        active_a3 = a3_target is not None and a3_margin >= 0.0
        serving_risk = ue.serving_sinr_db < self.cfg.system.q_in_db + 0.5 or ue.rlf.timer_s > 0.0
        future_risk = False
        serving_idx = int(ue.serving_idx)
        for horizon_s in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            f_serv = self._future_distance_to_cell(sim, serving_idx, float(horizon_s))
            f_best = self._future_nearest_nonserving(sim, float(horizon_s))
            f_best_dist = self._future_distance_to_cell(sim, f_best, float(horizon_s))
            if f_serv > f_best_dist + self.cfg.future_distance_margin_m:
                future_risk = True
                break
        return bool(active_a3 or ue.handover.is_active() or serving_risk or future_risk)

    def _rollout(self, sim: simulator.LEOSimulator, action_idx: int, horizon_s: float) -> LocalKPI:
        before = self._counter_snapshot(sim)
        sinrs: list[float] = []
        outage_flags: list[float] = []
        steps = max(1, int(round(horizon_s / self.cfg.simulation.sample_time_s)))
        defer_steps = int(round(self.defer_s / self.cfg.simulation.sample_time_s))
        action_executed = False
        for step_idx in range(steps):
            if sim.is_done():
                break
            mode = "a3"
            direct_target = None
            serving_risk = self._serving_risk(sim)
            a3_target, _ = self._a3_candidate(sim)
            a3_dwell = self._predicted_dwell_s(sim, int(a3_target)) if a3_target is not None else 0.0
            if action_idx == 0:
                mode = "a3_conservative"
            elif action_idx == 2:
                mode = "a3_aggressive" if self._aggressive_a3_guard(sim) else "a3"
            elif action_idx == 3:
                mode = "a3" if serving_risk else ("suppress" if step_idx < defer_steps else "a3")
            elif action_idx == 4:
                mode = "a3" if (serving_risk or a3_dwell >= self.min_predicted_dwell_s) else "a3_conservative"
            self._controlled_step(sim, mode, action_idx=action_idx, direct_target=direct_target)
            ue = sim.ue
            assert ue is not None
            sinrs.append(float(ue.serving_sinr_db))
            outage_flags.append(float(ue.serving_sinr_db < self.cfg.system.q_out_db))
        after = self._counter_snapshot(sim)
        return self._local_kpi(before, after, sinrs, outage_flags)

    def _execute_action_window(self, sim: simulator.LEOSimulator, action_idx: int, horizon_s: float) -> None:
        self.last_action_details = {"controller_executed_direct": 0, "controller_target_idx": -1, "controller_reason": ACTION_NAMES[action_idx]}
        before_ho = self._counter_snapshot(sim)["ho_count"]
        self._rollout(sim, action_idx, horizon_s)
        after_ho = self._counter_snapshot(sim)["ho_count"]
        self.last_action_details["controller_ho_delta"] = int(after_ho - before_ho)

    def _controlled_step(
        self,
        sim: simulator.LEOSimulator,
        mode: str,
        action_idx: int,
        direct_target: int | None = None,
    ) -> float:
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        ho = handover.A3HandoverResult()
        if mode in {"a3", "a3_conservative", "a3_aggressive"}:
            old_offset = sim.a3_engine.offset_db
            old_ttt = sim.a3_engine.ttt_s
            if mode == "a3_conservative":
                sim.a3_engine.offset_db = self.cfg.system.a3_offset_db + 1.0
                sim.a3_engine.ttt_s = self.cfg.system.a3_ttt_s + 0.4
            elif mode == "a3_aggressive":
                sim.a3_engine.offset_db = max(0.0, self.cfg.system.a3_offset_db - 0.75)
                sim.a3_engine.ttt_s = 0.0
            ho = sim.a3_engine.apply_a3(ue, ue.rsrp_dbm, sim.time_s, self.cfg.simulation.sample_time_s)
            sim.a3_engine.offset_db = old_offset
            sim.a3_engine.ttt_s = old_ttt
        elif mode == "suppress":
            if ue.handover.is_active():
                ue.handover.reset()
                ho.prep_failed = True
        elif mode == "direct" and direct_target is not None and int(direct_target) != int(ue.serving_idx):
            ho = self._execute_direct_ho(sim, int(direct_target))
        hopp_event = sim.hopp_detector.record_handover(ue.serving_idx, sim.time_s) if ho.ho_event else False
        self._refresh_measurements(sim)
        rlf_event = sim.rlf_detector.update_rlf(ue, ue.serving_sinr_db, self.cfg.simulation.sample_time_s, sim.time_s)
        if rlf_event:
            ue.rlf_event_count += 1
        best_idx = self._best_sinr_nonserving(sim)
        reward = sim._compute_reward(
            ue.serving_sinr_db,
            bool(rlf_event),
            bool(ho.is_uho),
            bool(ho.ho_event),
            int(ho.rb_delta),
            bool(ue.serving_sinr_db < self.cfg.system.q_out_db),
            bool(ho.prep_failed),
            bool(hopp_event),
            float(ue.ml_m[int(ue.serving_idx)]),
            float(np.min(ue.ml_m)),
        )
        sim.sim_history.log_step(
            history.HistoryEntry(
                sim.time_s,
                sim.step_idx,
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
                int(ho.prep_initiated),
                int(ho.prep_failed),
                int(ho.ho_event),
                int(ho.is_uho),
                int(rlf_event),
                int(ho.rb_delta),
                int(ue.rb_count),
                float(reward),
            )
        )
        sim.beam_centers[:, 1] += self.cfg.system.sat_speed_mps * self.cfg.simulation.sample_time_s
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ue.move_linear(self.cfg.simulation.sample_time_s)
        sim.time_s += self.cfg.simulation.sample_time_s
        sim.step_idx += 1
        return float(reward)

    def _execute_direct_ho(self, sim: simulator.LEOSimulator, target_idx: int) -> handover.A3HandoverResult:
        ue = sim.ue
        assert ue is not None
        result = handover.A3HandoverResult()
        prev_tos_s = sim.time_s - ue.serving_start_time_s
        result.ho_event = True
        result.target_idx = int(target_idx)
        result.prev_tos_s = float(prev_tos_s)
        result.is_uho = bool(prev_tos_s < self.cfg.system.min_tos_s)
        result.rb_delta = 10
        ue.change_serving_cell(int(target_idx), sim.time_s)
        ue.ho_count += 1
        ue.rb_count += result.rb_delta
        if result.is_uho:
            ue.uho_count += 1
        self.last_action_details = {
            "controller_executed_direct": 1,
            "controller_target_idx": int(target_idx),
            "controller_reason": "direct_geometry",
        }
        return result

    def _local_kpi(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        sinrs: list[float],
        outage_flags: list[float],
    ) -> LocalKPI:
        avg_sinr = float(np.mean(sinrs)) if sinrs else -200.0
        outage_time_s = float(np.sum(outage_flags) * self.cfg.simulation.sample_time_s)
        rlf_count = int(after["rlf_count"] - before["rlf_count"])
        uho_count = int(after["uho_count"] - before["uho_count"])
        ho_count = int(after["ho_count"] - before["ho_count"])
        rb_count = int(after["rb_count"] - before["rb_count"])
        tos_delta = float(after["tos_sum"] - before["tos_sum"])
        sinr_norm = float(np.clip((avg_sinr - self.cfg.reward.episode_norm_sinr_min_db) / 25.0, 0.0, 1.0))
        components = {
            "sinr": 4.0 * sinr_norm,
            "outage_time": -28.0 * outage_time_s,
            "rlf": -90.0 * rlf_count,
            "uho": -70.0 * uho_count,
            "ho": -0.6 * ho_count,
            "rb": -0.035 * rb_count,
            "tos": 0.4 * tos_delta,
        }
        return LocalKPI(
            score=float(sum(components.values())),
            avg_sinr_db=avg_sinr,
            outage_time_s=outage_time_s,
            rlf_count=rlf_count,
            uho_count=uho_count,
            ho_count=ho_count,
            rb_count=rb_count,
            avg_tos_delta_s=tos_delta,
            components=components,
        )

    def _counter_snapshot(self, sim: simulator.LEOSimulator) -> Dict[str, float]:
        ue = sim.ue
        assert ue is not None
        return {
            "ho_count": float(ue.ho_count),
            "uho_count": float(ue.uho_count),
            "rlf_count": float(ue.rlf_event_count),
            "rb_count": float(ue.rb_count),
            "tos_sum": float(sum(ue.tos_list)),
        }

    def _a3_candidate(self, sim: simulator.LEOSimulator) -> tuple[int | None, float]:
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        rsrp = np.asarray(ue.rsrp_dbm, dtype=float).copy()
        serving_idx = int(ue.serving_idx)
        if rsrp.size == 0:
            return None, -float("inf")
        serving_rsrp = float(rsrp[serving_idx])
        rsrp[serving_idx] = -np.inf
        if not np.isfinite(rsrp).any():
            return None, -float("inf")
        target = int(np.argmax(rsrp))
        margin = float(rsrp[target] - serving_rsrp - self.cfg.system.a3_offset_db - self.cfg.system.a3_hysteresis_db)
        return target, margin

    def _geometry_stable_target(self, sim: simulator.LEOSimulator) -> int | None:
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        horizons = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        current_ml = np.asarray(ue.ml_m, dtype=float)
        order = [int(i) for i in np.argsort(current_ml) if int(i) != serving_idx]
        candidates = order[:8]
        a3_target, _ = self._a3_candidate(sim)
        if a3_target is not None and a3_target not in candidates and a3_target != serving_idx:
            candidates.append(int(a3_target))
        best: tuple[float, int] | None = None
        for idx in candidates:
            distances = np.asarray([self._future_distance_to_cell(sim, idx, h) for h in horizons], dtype=float)
            if not np.isfinite(distances).all():
                continue
            nearest_distances = []
            for h in horizons:
                _, _, f_ml = self._future_measurements(sim, h)
                f = np.asarray(f_ml, dtype=float).copy()
                f[serving_idx] = np.inf
                nearest_distances.append(float(np.min(f)))
            dwell_s = 0.5 * float(np.sum(distances <= np.asarray(nearest_distances) + self.dwell_margin_m))
            approach = float(distances[0] - distances[-1])
            future_adv = self._future_distance_to_cell(sim, serving_idx, 1.0) - self._future_distance_to_cell(sim, idx, 1.0)
            score = -distances[2] / radius + 0.7 * np.clip(approach / radius, -2.0, 2.0) + 0.8 * dwell_s + 0.4 * future_adv / radius
            if best is None or score > best[0]:
                best = (float(score), int(idx))
        return None if best is None else int(best[1])

    def _predicted_dwell_s(self, sim: simulator.LEOSimulator, target_idx: int) -> float:
        horizons = np.arange(0.0, 3.0 + 1e-9, 0.5)
        serving_idx = int(sim.ue.serving_idx)
        dwell_slots = 0
        for h in horizons:
            _, _, f_ml = self._future_measurements(sim, float(h))
            f = np.asarray(f_ml, dtype=float).copy()
            f[serving_idx] = np.inf
            if float(f_ml[int(target_idx)]) <= float(np.min(f)) + self.dwell_margin_m:
                dwell_slots += 1
        return float(dwell_slots) * 0.5

    def _direct_guard(self, sim: simulator.LEOSimulator, target_idx: int) -> bool:
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        if int(target_idx) == int(ue.serving_idx):
            return False
        if sim.time_s - ue.serving_start_time_s < self.cfg.system.min_tos_s:
            return False
        serving_d = self._distance_to_cell(sim, int(ue.serving_idx))
        target_d = self._distance_to_cell(sim, int(target_idx))
        f_serv = self._future_distance_to_cell(sim, int(ue.serving_idx), 1.0)
        f_tgt = self._future_distance_to_cell(sim, int(target_idx), 1.0)
        dwell_ok = self._predicted_dwell_s(sim, int(target_idx)) >= self.min_predicted_dwell_s
        geom_ok = target_d <= serving_d - self.cfg.distance_margin_m or f_tgt <= f_serv - self.cfg.future_distance_margin_m
        risk_ok = self._serving_risk(sim) and f_tgt <= f_serv + self.cfg.future_distance_margin_m
        return bool(dwell_ok and (geom_ok or risk_ok))

    def _stable_guard(self, sim: simulator.LEOSimulator, target_idx: int) -> bool:
        return bool(self._predicted_dwell_s(sim, int(target_idx)) >= self.min_predicted_dwell_s and self._direct_guard(sim, int(target_idx)))

    def _urgent_a3_guard(self, sim: simulator.LEOSimulator, target_idx: int) -> bool:
        ue = sim.ue
        assert ue is not None
        self._refresh_measurements(sim)
        a3_target, a3_margin = self._a3_candidate(sim)
        if a3_target is None or int(a3_target) != int(target_idx) or a3_margin < 0.0:
            return False
        if sim.time_s - ue.serving_start_time_s < self.cfg.system.min_tos_s:
            return False
        if not self._serving_risk(sim):
            return False
        if self._predicted_dwell_s(sim, int(target_idx)) < self.min_predicted_dwell_s:
            return False
        f_serv = self._future_distance_to_cell(sim, int(ue.serving_idx), 1.0)
        f_tgt = self._future_distance_to_cell(sim, int(target_idx), 1.0)
        return bool(f_tgt <= f_serv + self.cfg.future_distance_margin_m)

    def _aggressive_a3_guard(self, sim: simulator.LEOSimulator) -> bool:
        ue = sim.ue
        assert ue is not None
        if not self._serving_risk(sim):
            return False
        mode = self.aggressive_guard_mode
        if mode == "risk":
            # Reliability-first, but do not create avoidable UHO by forcing an
            # aggressive A3 execution before the minimum time-of-stay.
            return bool(sim.time_s - ue.serving_start_time_s >= self.cfg.system.min_tos_s)
        a3_target, a3_margin = self._a3_candidate(sim)
        if a3_target is None or a3_margin < -0.5:
            return False
        dwell = self._predicted_dwell_s(sim, int(a3_target))
        if mode == "strict":
            if sim.time_s - ue.serving_start_time_s < 2.0 * self.cfg.system.min_tos_s:
                return False
            return bool(dwell >= self.min_predicted_dwell_s + 0.4)
        if mode == "off":
            return False
        # Default: min-ToS + dwell guard, tuned to reduce UHO risk.
        if sim.time_s - ue.serving_start_time_s < self.cfg.system.min_tos_s:
            return False
        return bool(dwell >= self.min_predicted_dwell_s)

    def _serving_risk(self, sim: simulator.LEOSimulator) -> bool:
        self._refresh_measurements(sim)
        ue = sim.ue
        assert ue is not None
        return bool(ue.serving_sinr_db < self.cfg.system.q_in_db + 0.5 or ue.rlf.timer_s > 0.0)

    def _distance_to_cell(self, sim: simulator.LEOSimulator, cell_idx: int) -> float:
        self._refresh_measurements(sim)
        ml = np.asarray(sim.ue.ml_m, dtype=float)
        if cell_idx < 0 or cell_idx >= len(ml):
            return float("inf")
        return float(ml[int(cell_idx)])

    def _future_distance_to_cell(self, sim: simulator.LEOSimulator, cell_idx: int, horizon_s: float) -> float:
        _, _, f_ml = self._future_measurements(sim, float(horizon_s))
        if cell_idx < 0 or cell_idx >= len(f_ml):
            return float("inf")
        return float(f_ml[int(cell_idx)])

    def _future_nearest_nonserving(self, sim: simulator.LEOSimulator, horizon_s: float) -> int:
        _, _, f_ml = self._future_measurements(sim, float(horizon_s))
        values = np.asarray(f_ml, dtype=float).copy()
        values[int(sim.ue.serving_idx)] = np.inf
        return int(np.argmin(values))

    def _future_measurements(self, sim: simulator.LEOSimulator, horizon_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ue = sim.ue
        assert ue is not None
        beam_centers = np.array(sim.beam_centers, copy=True)
        beam_centers[:, 1] += self.cfg.system.sat_speed_mps * float(horizon_s)
        ux, uy = ue.x_m, ue.y_m
        if self.cfg.simulation.ue_mobility_mode == "linear":
            ux += ue.speed_mps * np.cos(ue.heading_rad) * float(horizon_s)
            uy += ue.speed_mps * np.sin(ue.heading_rad) * float(horizon_s)
        return sim.channel_calc.update_channel(beam_centers[:, 0], beam_centers[:, 1], ux, uy, self.cfg.system.altitude_m)

    def _refresh_measurements(self, sim: simulator.LEOSimulator) -> None:
        ue = sim.ue
        assert ue is not None
        rsrp, sinr, ml = sim.channel_calc.update_channel(
            sim.beam_centers[:, 0], sim.beam_centers[:, 1], ue.x_m, ue.y_m, self.cfg.system.altitude_m
        )
        ue.rsrp_dbm = rsrp
        ue.sinr_db = sinr
        ue.ml_m = ml
        ue.update_serving_metrics()

    def _best_sinr_nonserving(self, sim: simulator.LEOSimulator) -> int:
        ue = sim.ue
        assert ue is not None
        values = np.asarray(ue.sinr_db, dtype=float).copy()
        values[int(ue.serving_idx)] = -np.inf
        return int(np.argmax(values)) if np.isfinite(values).any() else int(ue.serving_idx)

    def _observe(self) -> np.ndarray:
        self._refresh_measurements(self.sim)
        ue = self.sim.ue
        assert ue is not None
        serving_idx = int(ue.serving_idx)
        a3_target, a3_margin = self._a3_candidate(self.sim)
        if a3_target is None:
            a3_target = self._future_nearest_nonserving(self.sim, 1.0)
            a3_margin = -10.0
        geom_target = self._geometry_stable_target(self.sim)
        if geom_target is None:
            geom_target = a3_target
        radius = max(self.cfg.system.cell_radius_m, 1.0)
        serving_d = self._distance_to_cell(self.sim, serving_idx)
        a3_d = self._distance_to_cell(self.sim, int(a3_target))
        geom_d = self._distance_to_cell(self.sim, int(geom_target))
        dwell = self._predicted_dwell_s(self.sim, int(geom_target))
        tos = max(0.0, self.sim.time_s - ue.serving_start_time_s)
        state = [
            np.clip((ue.serving_sinr_db + 30.0) / 50.0, 0.0, 1.0),
            np.clip((ue.serving_rsrp_dbm + 140.0) / 70.0, 0.0, 1.0),
            np.clip(a3_margin / 10.0, -1.0, 1.0) / 2.0 + 0.5,
            np.clip(serving_d / radius, 0.0, 5.0) / 5.0,
            np.clip(a3_d / radius, 0.0, 5.0) / 5.0,
            np.clip(geom_d / radius, 0.0, 5.0) / 5.0,
            np.clip((a3_d - serving_d) / radius, -3.0, 3.0) / 6.0 + 0.5,
            np.clip((geom_d - serving_d) / radius, -3.0, 3.0) / 6.0 + 0.5,
            np.clip(dwell / 3.0, 0.0, 1.0),
            np.clip(tos / max(self.cfg.reward.episode_norm_tos_s, 1e-9), 0.0, 1.0),
            np.clip(ue.rlf.timer_s / max(self.cfg.system.t310_s, 1e-9), 0.0, 1.0),
            np.clip(ue.ho_count / 20.0, 0.0, 1.0),
            np.clip(ue.uho_count / 10.0, 0.0, 1.0),
            np.clip(ue.rlf_event_count / 10.0, 0.0, 1.0),
            np.clip(ue.rb_count / 250.0, 0.0, 1.0),
            np.clip(ue.speed_mps / max(self.cfg.simulation.ue_speed_max_mps, 1.0), 0.0, 1.0),
            0.5 + 0.5 * float(np.cos(ue.heading_rad)),
            0.5 + 0.5 * float(np.sin(ue.heading_rad)),
            np.clip(self.sim.time_s / max(self.cfg.simulation.total_time_s, 1e-9), 0.0, 1.0),
            float(ue.handover.is_active()),
            float(self._serving_risk(self.sim)),
            float(a3_target == geom_target),
        ]
        for h in list(self.cfg.simulation.lookahead_horizons_s)[:3]:
            f_serv = self._future_distance_to_cell(self.sim, serving_idx, float(h))
            f_a3 = self._future_distance_to_cell(self.sim, int(a3_target), float(h))
            f_geom = self._future_distance_to_cell(self.sim, int(geom_target), float(h))
            _, f_sinr, _ = self._future_measurements(self.sim, float(h))
            state.extend(
                [
                    np.clip((f_a3 - f_serv) / radius, -3.0, 3.0) / 6.0 + 0.5,
                    np.clip((f_geom - f_serv) / radius, -3.0, 3.0) / 6.0 + 0.5,
                    np.clip((f_sinr[serving_idx] + 30.0) / 50.0, 0.0, 1.0),
                    float(f_geom <= f_serv + self.cfg.future_distance_margin_m),
                ]
            )
        while len(state) < self.state_dim:
            state.append(0.0)
        return np.asarray(state[: self.state_dim], dtype=np.float32)

    def get_kpi(self):
        return self.sim.get_kpi()

    def get_history_dicts(self):
        return self.sim.get_history_dicts()
