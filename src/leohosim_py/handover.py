"""Event D2 handover logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import entities


@dataclass
class D2HandoverResult:
    d2_cond1: bool = False
    d2_cond2_any: bool = False
    prep_initiated: bool = False
    exec_initiated: bool = False
    target_idx: Optional[int] = None
    prep_failed: bool = False
    ho_event: bool = False
    prev_tos_s: Optional[float] = None
    rb_delta: int = 0
    is_uho: bool = False


class D2HandoverEngine:
    def __init__(self, hys_m: float, ttt_s: float, min_tos_s: float, exclude_serving_from_targets: bool = True):
        self.hys_m = hys_m
        self.ttt_s = ttt_s
        self.min_tos_s = min_tos_s
        self.exclude_serving_from_targets = exclude_serving_from_targets

    def apply_d2(
        self,
        ue: entities.UE,
        ml_m: np.ndarray,
        serving_sinr_db: float,
        thresh1_m: float,
        thresh2_m: float,
        current_time_s: float,
        dt_s: float,
    ) -> D2HandoverResult:
        result = D2HandoverResult()
        ml_serving = ml_m[ue.serving_idx]
        result.d2_cond1 = ml_serving - self.hys_m > thresh1_m
        candidates = ml_m + self.hys_m < thresh2_m
        if self.exclude_serving_from_targets:
            candidates = candidates.copy()
            candidates[ue.serving_idx] = False
        result.d2_cond2_any = bool(np.any(candidates))

        if not ue.handover.is_active():
            if result.d2_cond1 and result.d2_cond2_any:
                idx = np.where(candidates)[0]
                target = int(idx[np.argmin(ml_m[idx])])
                ue.handover.initiate_preparation(target, current_time_s)
                result.prep_initiated = True
                result.target_idx = target
                result.rb_delta += 3
        else:
            target = ue.handover.prep_target_idx
            if target is not None and result.d2_cond1 and ml_m[target] + self.hys_m < thresh2_m:
                ue.handover.exec_ttt_timer_s += dt_s
                if ue.handover.exec_ttt_timer_s >= self.ttt_s:
                    result.exec_initiated = True
                    result.ho_event = True
                    result.target_idx = target
                    prev_tos = current_time_s - ue.serving_start_time_s
                    result.prev_tos_s = prev_tos
                    result.is_uho = prev_tos < self.min_tos_s
                    ue.change_serving_cell(target, current_time_s)
                    ue.ho_count += 1
                    if result.is_uho:
                        ue.uho_count += 1
                    result.rb_delta += 7
                    ue.handover.reset()
            else:
                result.prep_failed = True
                ue.handover.reset()

        ue.rb_count += result.rb_delta
        return result


@dataclass
class A3HandoverResult:
    a3_condition: bool = False
    prep_initiated: bool = False
    prep_failed: bool = False
    ho_event: bool = False
    target_idx: Optional[int] = None
    prev_tos_s: Optional[float] = None
    rb_delta: int = 0
    is_uho: bool = False


class A3RSRPHandoverEngine:
    """Conventional Event A3 handover using neighbor and serving RSRP."""

    def __init__(
        self,
        offset_db: float,
        hysteresis_db: float,
        ttt_s: float,
        min_tos_s: float,
        exclude_serving_from_targets: bool = True,
        guard_min_tos: bool = False,
    ):
        self.offset_db = float(offset_db)
        self.hysteresis_db = float(hysteresis_db)
        self.ttt_s = float(ttt_s)
        self.min_tos_s = float(min_tos_s)
        self.exclude_serving_from_targets = bool(exclude_serving_from_targets)
        self.guard_min_tos = bool(guard_min_tos)

    def apply_a3(
        self,
        ue: entities.UE,
        rsrp_dbm: np.ndarray,
        current_time_s: float,
        dt_s: float,
    ) -> A3HandoverResult:
        result = A3HandoverResult()
        values = np.asarray(rsrp_dbm, dtype=float)
        serving_idx = int(ue.serving_idx)
        serving_rsrp = float(values[serving_idx])
        target_values = values.copy()
        if self.exclude_serving_from_targets and target_values.size:
            target_values[serving_idx] = -np.inf

        if not np.isfinite(target_values).any():
            if ue.handover.is_active():
                result.prep_failed = True
                ue.handover.reset()
            return result

        target = int(np.argmax(target_values))
        target_rsrp = float(target_values[target])
        result.a3_condition = bool(target_rsrp >= serving_rsrp + self.offset_db + self.hysteresis_db)

        if not ue.handover.is_active():
            if result.a3_condition:
                ue.handover.initiate_preparation(target, current_time_s)
                result.prep_initiated = True
                result.target_idx = target
                result.rb_delta += 3
        else:
            prep_target = ue.handover.prep_target_idx
            same_target = prep_target is not None and int(prep_target) == target
            if same_target and result.a3_condition:
                ue.handover.exec_ttt_timer_s += dt_s
                if ue.handover.exec_ttt_timer_s >= self.ttt_s:
                    prev_tos = current_time_s - ue.serving_start_time_s
                    if self.guard_min_tos and prev_tos < self.min_tos_s:
                        result.target_idx = target
                        ue.handover.exec_ttt_timer_s = self.ttt_s
                        ue.rb_count += result.rb_delta
                        return result
                    result.ho_event = True
                    result.target_idx = target
                    result.prev_tos_s = prev_tos
                    result.is_uho = prev_tos < self.min_tos_s
                    ue.change_serving_cell(target, current_time_s)
                    ue.ho_count += 1
                    if result.is_uho:
                        ue.uho_count += 1
                    result.rb_delta += 7
                    ue.handover.reset()
            else:
                result.prep_failed = True
                ue.handover.reset()

        ue.rb_count += result.rb_delta
        return result
