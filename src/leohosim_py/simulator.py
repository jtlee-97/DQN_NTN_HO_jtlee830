"""Core LEO NTN Event D2 simulator."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from . import channel, config, entities, geometry, handover, history, hopp, kpi, rlf


class LEOSimulator:
    def __init__(self, cfg: config.LeohosimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.sim_cfg = cfg.simulation
        self.sys_cfg = cfg.system
        self.beam_centers = self._initial_beam_centers()
        self.n_sites = len(self.beam_centers)
        self.channel_calc = channel.ChannelCalculator(
            self.sys_cfg.eirp_dbm,
            self.sys_cfg.carrier_freq_hz,
            self.sys_cfg.bandwidth_hz,
            self.sys_cfg.noise_figure_db,
            self.sys_cfg.cell_radius_m,
            self.sys_cfg.aperture_m,
            self.sim_cfg.channel_mode,
            rng,
            self.sys_cfg.interference_top_k,
            self.sys_cfg.interference_neighbor_count,
        )
        self.d2_engine = handover.D2HandoverEngine(
            self.sys_cfg.hys_m,
            self.sys_cfg.ttt_s,
            self.sys_cfg.min_tos_s,
            self.sim_cfg.exclude_serving_from_targets,
        )
        self.a3_engine = handover.A3RSRPHandoverEngine(
            self.sys_cfg.a3_offset_db,
            self.sys_cfg.a3_hysteresis_db,
            self.sys_cfg.a3_ttt_s,
            self.sys_cfg.min_tos_s if self.sys_cfg.a3_min_tos_s is None else self.sys_cfg.a3_min_tos_s,
            self.sim_cfg.exclude_serving_from_targets,
        )
        self.rlf_detector = rlf.RLFDetector(self.sys_cfg.q_out_db, self.sys_cfg.q_in_db, self.sys_cfg.t310_s)
        self.hopp_detector = hopp.HOPPDetector()
        self.sim_history = history.History()
        self.ue: entities.UE | None = None
        self.time_s = 0.0
        self.step_idx = 0
        self.total_steps = int(round(self.sim_cfg.total_time_s / self.sim_cfg.sample_time_s))

    def reset(self) -> None:
        self.beam_centers = self._initial_beam_centers()
        x, y = self._initial_ue_position()
        speed_mps, heading_rad = self._initial_ue_velocity()
        self.ue = entities.UE(x, y, speed_mps, heading_rad=heading_rad)
        ml = geometry.calculate_ml_distance(self.beam_centers[:, 0], self.beam_centers[:, 1], x, y)
        self.ue.serving_idx = int(np.argmin(ml))
        self.ue.serving_start_time_s = 0.0
        self.time_s = 0.0
        self.step_idx = 0
        self.sim_history.clear()
        self.hopp_detector.reset()

    def step(self, thresh1_m: float, thresh2_m: float, action_idx: int = -1) -> Tuple[Dict[str, Any], float]:
        ue = self.ue
        assert ue is not None
        rsrp, sinr, ml = self.channel_calc.update_channel(
            self.beam_centers[:, 0], self.beam_centers[:, 1], ue.x_m, ue.y_m, self.sys_cfg.altitude_m
        )
        ue.rsrp_dbm = rsrp
        ue.sinr_db = sinr
        ue.ml_m = ml
        ue.update_serving_metrics()

        if self.cfg.baseline_policy == "a3_rsrp":
            ho = self.a3_engine.apply_a3(ue, rsrp, self.time_s, self.sim_cfg.sample_time_s)
        else:
            ho = self.d2_engine.apply_d2(
                ue, ml, ue.serving_sinr_db, thresh1_m, thresh2_m, self.time_s, self.sim_cfg.sample_time_s
            )
        hopp_event = self.hopp_detector.record_handover(ue.serving_idx, self.time_s) if ho.ho_event else False
        rlf_event = self.rlf_detector.update_rlf(ue, ue.serving_sinr_db, self.sim_cfg.sample_time_s, self.time_s)
        uho_link_failure = bool(
            ho.is_uho
            and ho.prev_tos_s is not None
            and float(ho.prev_tos_s) < 0.75 * self.sys_cfg.min_tos_s
        )
        if uho_link_failure:
            ue.rlf_event_count += 1
            rlf_event = True
        best_idx = int(np.argmin(ml))
        reward = self._compute_reward(
            ue.serving_sinr_db,
            rlf_event,
            ho.is_uho,
            ho.ho_event,
            ho.rb_delta,
            ue.serving_sinr_db < self.sys_cfg.q_out_db,
            ho.prep_failed,
            hopp_event,
            float(ml[ue.serving_idx]),
            float(ml[best_idx]),
        )
        self.sim_history.log_step(
            history.HistoryEntry(
                self.time_s,
                self.step_idx,
                action_idx,
                thresh1_m,
                thresh2_m,
                ue.serving_idx,
                float(ml[ue.serving_idx]),
                ue.serving_rsrp_dbm,
                ue.serving_sinr_db,
                best_idx,
                float(ml[best_idx]),
                float(sinr[best_idx]),
                int(ho.prep_initiated),
                int(ho.prep_failed),
                int(ho.ho_event),
                int(ho.is_uho),
                int(rlf_event),
                ho.rb_delta,
                ue.rb_count,
                reward,
            )
        )
        self.beam_centers[:, 1] += self.sys_cfg.sat_speed_mps * self.sim_cfg.sample_time_s
        if self.sim_cfg.ue_mobility_mode == "linear":
            ue.move_linear(self.sim_cfg.sample_time_s)
        self.time_s += self.sim_cfg.sample_time_s
        self.step_idx += 1
        return {
            "time_s": self.time_s,
            "step_idx": self.step_idx,
            "serving_idx": ue.serving_idx,
            "serving_sinr_db": ue.serving_sinr_db,
            "best_sinr_db": float(sinr[best_idx]),
            "serving_ml_m": float(ml[ue.serving_idx]),
            "best_ml_m": float(ml[best_idx]),
            "prep_initiated": ho.prep_initiated,
            "prep_failed": ho.prep_failed,
            "ho_event": ho.ho_event,
            "uho_event": ho.is_uho,
            "rlf_event": rlf_event,
            "hopp_event": hopp_event,
            "prev_tos_s": -1.0 if ho.prev_tos_s is None else float(ho.prev_tos_s),
            "rb_delta": ho.rb_delta,
            "thresh1_m": thresh1_m,
            "thresh2_m": thresh2_m,
        }, reward

    def _initial_beam_centers(self) -> np.ndarray:
        layout_radius_m = self.sys_cfg.cell_radius_m
        if self.sys_cfg.cell_isd_m > 0:
            layout_radius_m = self.sys_cfg.cell_isd_m / np.sqrt(3.0)
        if self.sim_cfg.beam_layout_mode == "overlap_19beam":
            return geometry.get_overlapping_19beam_layout(
                layout_radius_m,
                self.sim_cfg.overlay_satellite_count,
                self.sim_cfg.overlay_beams_per_satellite,
                self.sim_cfg.overlay_satellite_spread_m,
                self.sim_cfg.overlay_beam_jitter_m,
                self.sim_cfg.overlay_ring_jitter_m,
                self.sim_cfg.overlay_track_group_count,
                self.sim_cfg.overlay_track_spacing_m,
                self.rng,
            )
        return geometry.get_61_site_layout(layout_radius_m, self.sys_cfg.num_sites)

    def _compute_reward(
        self,
        serving_sinr_db: float,
        rlf_event: bool,
        uho_event: bool,
        ho_event: bool,
        rb_delta: int,
        outage: bool,
        prep_fail: bool,
        hopp_event: bool,
        serving_ml_m: float,
        best_ml: float,
    ) -> float:
        w = self.cfg.reward
        sinr_norm = float(np.clip((serving_sinr_db + 20.0) / 40.0, 0.0, 1.0))
        serving_norm = float(np.clip(serving_ml_m / self.sys_cfg.cell_radius_m, 0.0, 3.0))
        best_gain = float(np.clip((serving_ml_m - best_ml) / self.sys_cfg.cell_radius_m, 0.0, 3.0))
        return float(
            w.w_sinr * sinr_norm
            - w.w_rlf * rlf_event
            - w.w_uho * uho_event
            - w.w_ho * ho_event
            - w.w_rbs * rb_delta
            - w.w_outage * outage
            - w.w_prep_fail * prep_fail
            - w.w_hopp * hopp_event
            - w.w_serving_distance * serving_norm
            + w.w_best_distance_gain * best_gain
        )

    def _initial_ue_position(self) -> Tuple[float, float]:
        mode = self.sim_cfg.ue_mode
        if mode == "fixed":
            return float(self.sim_cfg.ue_x_m), float(self.sim_cfg.ue_y_m)
        cell_idx = int(np.clip(self.sim_cfg.ue_cell_index, 0, len(self.beam_centers) - 1))
        if mode == "random_cell":
            cell_idx = self._sample_random_cell_index()
        center = self.beam_centers[cell_idx]
        if mode == "cell_center":
            rho = 0.0
        elif mode == "cell_mid":
            rho = 0.5
        elif mode == "cell_edge":
            rho = 0.95
        elif mode == "random_cell":
            r0 = float(np.clip(self.sim_cfg.ue_random_radius_min_ratio, 0.0, 1.0))
            r1 = float(np.clip(self.sim_cfg.ue_random_radius_max_ratio, r0, 1.0))
            rho = float(np.sqrt(self.rng.uniform(r0 * r0, r1 * r1)))
        else:
            raise ValueError(f"Unsupported ue_mode: {mode}")
        theta = (
            float(self.rng.uniform(0.0, 2.0 * np.pi))
            if self.sim_cfg.ue_random_angle_deg is None or mode == "random_cell"
            else float(np.radians(self.sim_cfg.ue_random_angle_deg))
        )
        radius = self.sys_cfg.cell_radius_m
        return float(center[0] + rho * radius * np.cos(theta)), float(center[1] + rho * radius * np.sin(theta))

    def _initial_ue_velocity(self) -> Tuple[float, float]:
        if self.sim_cfg.ue_mobility_mode == "static":
            return float(self.sim_cfg.ue_speed_mps), 0.0
        if self.sim_cfg.ue_mobility_mode != "linear":
            raise ValueError(f"Unsupported ue_mobility_mode: {self.sim_cfg.ue_mobility_mode}")
        speed = float(self.rng.uniform(self.sim_cfg.ue_speed_min_mps, self.sim_cfg.ue_speed_max_mps))
        if self.sim_cfg.ue_heading_deg is not None and not self.sim_cfg.ue_random_heading:
            heading = float(np.radians(self.sim_cfg.ue_heading_deg))
        else:
            heading = float(self.rng.uniform(0.0, 2.0 * np.pi))
        return speed, heading

    def _sample_random_cell_index(self) -> int:
        max_tier = self.sim_cfg.ue_random_cell_max_tier
        if max_tier is None:
            return int(self.rng.integers(0, len(self.beam_centers)))
        min_tier = max(int(self.sim_cfg.ue_random_cell_min_tier), 0)
        max_tier = max(int(max_tier), min_tier)
        tier_unit = np.sqrt(3.0) * self.sys_cfg.cell_radius_m
        dist = np.linalg.norm(self.beam_centers, axis=1)
        mask = (dist >= (min_tier - 0.5) * tier_unit) & (dist <= (max_tier + 0.5) * tier_unit)
        candidates = np.where(mask)[0]
        return int(self.rng.choice(candidates if len(candidates) else np.arange(len(self.beam_centers))))

    def get_kpi(self) -> kpi.KPI:
        assert self.ue is not None
        return kpi.compute_kpi(
            self.sim_history.to_dicts(),
            self.ue.tos_list,
            self.sys_cfg.q_out_db,
            self.time_s,
            sum(e.reward for e in self.sim_history.entries),
            self.hopp_detector.get_hopp_count(),
            self.sys_cfg.min_tos_s,
        )

    def get_history_dicts(self) -> List[Dict[str, Any]]:
        return self.sim_history.to_dicts()

    def is_done(self) -> bool:
        return self.step_idx >= self.total_steps
