"""Simulator entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class HandoverState:
    preparation_state: bool = False
    execution_state: bool = False
    prep_target_idx: Optional[int] = None
    ttt_timer_s: float = 0.0
    exec_ttt_timer_s: float = 0.0
    prep_start_time_s: float = 0.0
    exec_start_time_s: float = 0.0

    def initiate_preparation(self, target_idx: int, current_time_s: float) -> None:
        self.preparation_state = True
        self.prep_target_idx = target_idx
        self.prep_start_time_s = current_time_s
        self.exec_ttt_timer_s = 0.0

    def reset(self) -> None:
        self.preparation_state = False
        self.execution_state = False
        self.prep_target_idx = None
        self.ttt_timer_s = 0.0
        self.exec_ttt_timer_s = 0.0

    def is_active(self) -> bool:
        return self.preparation_state or self.execution_state


@dataclass
class RLFState:
    timer_s: float = 0.0
    event_count: int = 0
    last_event_time_s: float = -1.0

    def reset_timer(self) -> None:
        self.timer_s = 0.0

    def increment_timer(self, dt_s: float) -> None:
        self.timer_s += dt_s

    def trigger_event(self, current_time_s: float) -> None:
        self.event_count += 1
        self.last_event_time_s = current_time_s
        self.timer_s = 0.0


@dataclass
class UE:
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float = 0.0
    serving_idx: int = 0
    rsrp_dbm: Optional[np.ndarray] = None
    sinr_db: Optional[np.ndarray] = None
    ml_m: Optional[np.ndarray] = None
    serving_rsrp_dbm: float = -200.0
    serving_sinr_db: float = -200.0
    handover: HandoverState = field(default_factory=HandoverState)
    rlf: RLFState = field(default_factory=RLFState)
    ho_count: int = 0
    uho_count: int = 0
    rlf_event_count: int = 0
    rb_count: int = 0
    tos_list: List[float] = field(default_factory=list)
    serving_start_time_s: float = 0.0

    def update_serving_metrics(self) -> None:
        if self.rsrp_dbm is not None:
            self.serving_rsrp_dbm = float(self.rsrp_dbm[self.serving_idx])
        if self.sinr_db is not None:
            self.serving_sinr_db = float(self.sinr_db[self.serving_idx])

    def change_serving_cell(self, new_serving_idx: int, current_time_s: float) -> None:
        tos = current_time_s - self.serving_start_time_s
        if tos >= 0:
            self.tos_list.append(tos)
        self.serving_idx = new_serving_idx
        self.serving_start_time_s = current_time_s
        self.handover.reset()
        self.update_serving_metrics()

    def move_linear(self, dt_s: float) -> None:
        self.x_m += self.speed_mps * float(np.cos(self.heading_rad)) * dt_s
        self.y_m += self.speed_mps * float(np.sin(self.heading_rad)) * dt_s
