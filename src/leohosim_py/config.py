"""Configuration objects for the LEO D2 simulator and RL training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class RewardConfig:
    w_sinr: float = 1.0
    w_rlf: float = 10.0
    w_uho: float = 5.0
    w_ho: float = 0.1
    w_rbs: float = 0.02
    w_outage: float = 1.0
    w_prep_fail: float = 0.2
    w_hopp: float = 2.0
    w_serving_distance: float = 0.05
    w_best_distance_gain: float = 0.05
    episode_w_sinr: float = 40.0
    episode_w_outage: float = 20.0
    episode_w_rlf: float = 12.0
    episode_w_uho: float = 6.0
    episode_w_hopp: float = 4.0
    episode_w_short_tos: float = 4.0
    episode_w_ho: float = 0.3
    episode_w_rb: float = 0.02
    episode_w_avg_tos: float = 1.5
    episode_norm_sinr_min_db: float = -20.0
    episode_norm_sinr_max_db: float = 5.0
    episode_norm_rlf_count: float = 10.0
    episode_norm_ho_count: float = 20.0
    episode_norm_uho_count: float = 20.0
    episode_norm_hopp_count: float = 10.0
    episode_norm_short_tos_count: float = 20.0
    episode_norm_rb_count: float = 250.0
    episode_norm_tos_s: float = 6.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ActionGridConfig:
    thresh1_m: List[float] = field(
        default_factory=lambda: [11560.0, 16825.0147, 20022.50735, 23120.0, 27744.0]
    )
    thresh2_m: List[float] = field(default_factory=lambda: [17340.0, 23120.0, 28900.0])
    thresh1_range_m: Optional[List[float]] = None
    thresh2_range_m: Optional[List[float]] = None
    range_step_m: Optional[float] = None

    def get_actions(self) -> List[Tuple[float, float]]:
        t1_values = self._values_from_range_or_list(self.thresh1_range_m, self.thresh1_m)
        t2_values = self._values_from_range_or_list(self.thresh2_range_m, self.thresh2_m)
        return [(float(t1), float(t2)) for t1 in t1_values for t2 in t2_values]

    def _values_from_range_or_list(self, value_range: Optional[List[float]], fallback: List[float]) -> List[float]:
        if value_range is None:
            return [float(v) for v in fallback]
        if self.range_step_m is None:
            raise ValueError("range_step_m is required when threshold ranges are used")
        if len(value_range) != 2:
            raise ValueError("threshold range must be [min_m, max_m]")
        start, stop = float(value_range[0]), float(value_range[1])
        step = float(self.range_step_m)
        if step <= 0:
            raise ValueError("range_step_m must be positive")
        n = int(round((stop - start) / step))
        return [start + i * step for i in range(n + 1)]


@dataclass
class SimulationConfig:
    seed: int = 7
    total_time_s: float = 24.0
    sample_time_s: float = 0.2
    ue_mode: str = "fixed"
    ue_x_m: float = 17_340.0
    ue_y_m: float = 80_090.0293
    ue_speed_mps: float = 0.0
    ue_mobility_mode: str = "static"
    ue_speed_min_mps: float = 0.0
    ue_speed_max_mps: float = 0.0
    ue_heading_deg: Optional[float] = None
    ue_random_heading: bool = True
    lookahead_horizons_s: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    ue_cell_index: int = 0
    ue_random_radius_min_ratio: float = 0.0
    ue_random_radius_max_ratio: float = 1.0
    ue_random_angle_deg: Optional[float] = None
    ue_random_cell_min_tier: int = 0
    ue_random_cell_max_tier: Optional[int] = None
    channel_mode: str = "matlab_like_deterministic"
    beam_layout_mode: str = "hex"
    overlay_satellite_count: int = 1
    overlay_beams_per_satellite: int = 19
    overlay_satellite_spread_m: float = 8000.0
    overlay_beam_jitter_m: float = 0.0
    overlay_ring_jitter_m: float = 0.0
    overlay_track_group_count: int = 1
    overlay_track_spacing_m: float = 0.0
    episodes_eval: int = 20
    episodes_train: int = 500
    exclude_serving_from_targets: bool = True
    rlf_mode: str = "timer_qout_qin"


@dataclass
class SystemConfig:
    cell_radius_m: float = 23_120.0
    cell_isd_m: float = 40_045.0147
    altitude_m: float = 600_000.0
    sat_speed_mps: float = 7_560.0
    carrier_freq_hz: float = 2e9
    bandwidth_hz: float = 20e6
    rb_bandwidth_hz: float = 180_000.0
    tx_gain_db: float = 30.0
    aperture_m: float = 2.0
    eirp_dbm: float = 34.0
    noise_figure_db: float = 7.0
    interference_top_k: int = 0
    interference_neighbor_count: int = 0
    hys_m: float = 0.0
    ttt_s: float = 0.0
    t310_s: float = 1.0
    q_out_db: float = -8.0
    q_in_db: float = -6.0
    min_tos_s: float = 1.0
    num_sites: int = 127
    a3_offset_db: float = 1.0
    a3_hysteresis_db: float = 0.5
    a3_ttt_s: float = 0.4
    a3_min_tos_s: Optional[float] = None


@dataclass
class QLearningConfig:
    alpha: float = 0.15
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    discretization_bins: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class DQNConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 800
    hidden_dim: int = 128
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_freq: int = 200
    warmup_steps: int = 50
    train_freq: int = 4
    gradient_clip: float = 1.0
    loss_type: str = "huber"


@dataclass
class DDQNConfig(DQNConfig):
    pass


@dataclass
class LeohosimConfig:
    seed: int = 7
    baseline_policy: str = "a3_rsrp"
    dqn_feature_mode: str = "distance_predictive"
    future_action_mode: str = "guarded"
    distance_margin_m: float = 1000.0
    future_distance_margin_m: float = 1000.0
    approaching_margin_m: float = 500.0
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ddqn: DDQNConfig = field(default_factory=DDQNConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "LeohosimConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        if "seed" in data:
            cfg.seed = data["seed"]
        for key in (
            "baseline_policy",
            "dqn_feature_mode",
            "future_action_mode",
            "distance_margin_m",
            "future_distance_margin_m",
            "approaching_margin_m",
        ):
            if key in data:
                setattr(cfg, key, data[key])
        if "simulation" in data:
            sim = data["simulation"]
            if "simulation_time_s" in sim and "total_time_s" not in sim:
                sim["total_time_s"] = sim.pop("simulation_time_s")
            cfg.simulation = SimulationConfig(**sim)
        if "system" in data:
            cfg.system = SystemConfig(**data["system"])
        if "action_grid" in data:
            cfg.action_grid = ActionGridConfig(**data["action_grid"])
        if "reward" in data:
            cfg.reward = RewardConfig(**data["reward"])
        if "q_learning" in data:
            cfg.q_learning = QLearningConfig(**data["q_learning"])
        if "dqn" in data:
            cfg.dqn = DQNConfig(**data["dqn"])
        if "ddqn" in data:
            cfg.ddqn = DDQNConfig(**data["ddqn"])
        return cfg

    @classmethod
    def default(cls) -> "LeohosimConfig":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        assert self.seed >= 0
        assert self.simulation.total_time_s > 0
        assert self.simulation.sample_time_s > 0
        assert self.simulation.ue_speed_min_mps >= 0
        assert self.simulation.ue_speed_max_mps >= self.simulation.ue_speed_min_mps
        assert all(float(h) > 0 for h in self.simulation.lookahead_horizons_s)
        assert self.system.altitude_m > 0
        assert self.system.num_sites > 0
        assert self.system.q_out_db < self.system.q_in_db
        assert self.dqn.learning_rate > 0
        assert self.baseline_policy in {"a3_rsrp", "d2_fixed"}
        assert self.dqn_feature_mode in {"distance_predictive", "signal_predictive"}
        assert self.future_action_mode in {"guarded", "unguarded"}
        assert self.simulation.beam_layout_mode in {"hex", "overlap_19beam"}
        assert self.simulation.overlay_satellite_count > 0
        assert self.simulation.overlay_beams_per_satellite > 0
        assert self.simulation.overlay_satellite_spread_m >= 0.0
        assert self.simulation.overlay_beam_jitter_m >= 0.0
        assert self.simulation.overlay_ring_jitter_m >= 0.0
        assert self.simulation.overlay_track_group_count > 0
        assert self.simulation.overlay_track_spacing_m >= 0.0
        assert self.system.interference_top_k >= 0
        assert self.system.interference_neighbor_count >= 0
        assert self.system.a3_ttt_s >= 0.0


def get_small_debug_config() -> LeohosimConfig:
    cfg = LeohosimConfig()
    cfg.simulation.total_time_s = 8.0
    cfg.simulation.episodes_eval = 2
    cfg.simulation.episodes_train = 5
    cfg.simulation.ue_mode = "random_cell"
    cfg.simulation.ue_random_cell_max_tier = 3
    cfg.action_grid = ActionGridConfig(thresh1_m=[16825.0147, 20022.50735, 23120.0], thresh2_m=[23120.0])
    cfg.dqn.batch_size = 16
    cfg.dqn.warmup_steps = 20
    cfg.ddqn.batch_size = 16
    cfg.ddqn.warmup_steps = 20
    return cfg


def get_matlab_default_config() -> LeohosimConfig:
    return LeohosimConfig()
