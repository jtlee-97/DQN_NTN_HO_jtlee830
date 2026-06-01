"""Episode KPI calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class KPI:
    avg_sinr_db: float
    min_sinr_db: float
    max_sinr_db: float
    avg_rsrp_dbm: float
    outage_fraction: float
    ho_count: int
    ho_rate_per_min: float
    uho_count: int
    uho_rate: float
    avg_tos_s: float
    min_tos_s: float
    short_tos_count: int
    rlf_count: int
    rlf_rate: float
    hopp_count: int
    hopp_rate: float
    rb_count: int
    rb_rate_per_min: float
    episode_reward: float


def compute_kpi(
    history_entries: List[dict],
    ue_tos_list: List[float],
    q_out_db: float,
    total_time_s: float,
    episode_reward: float,
    hopp_count: int = 0,
    short_tos_threshold_s: float = 1.0,
) -> KPI:
    if not history_entries:
        return KPI(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    sinr = np.array([e["serving_sinr_db"] for e in history_entries], dtype=float)
    rsrp = np.array([e["serving_rsrp_dbm"] for e in history_entries], dtype=float)
    ho = np.array([e["ho_event"] for e in history_entries], dtype=float)
    uho = np.array([e["uho_event"] for e in history_entries], dtype=float)
    rlf = np.array([e["rlf_event"] for e in history_entries], dtype=float)
    rb = np.array([e["rb_delta"] for e in history_entries], dtype=float)
    tos = np.array(list(ue_tos_list), dtype=float)

    ho_count = int(np.sum(ho))
    uho_count = int(np.sum(uho))
    rlf_count = int(np.sum(rlf))
    rb_count = int(np.sum(rb))
    avg_tos = float(np.mean(tos)) if len(tos) else 0.0
    min_tos = float(np.min(tos)) if len(tos) else 0.0
    short_tos = int(np.sum(tos < short_tos_threshold_s)) if len(tos) else 0

    return KPI(
        avg_sinr_db=float(np.mean(sinr)),
        min_sinr_db=float(np.min(sinr)),
        max_sinr_db=float(np.max(sinr)),
        avg_rsrp_dbm=float(np.mean(rsrp)),
        outage_fraction=float(np.mean(sinr < q_out_db)),
        ho_count=ho_count,
        ho_rate_per_min=60.0 * ho_count / max(total_time_s, 1.0),
        uho_count=uho_count,
        uho_rate=uho_count / max(ho_count, 1),
        avg_tos_s=avg_tos,
        min_tos_s=min_tos,
        short_tos_count=short_tos,
        rlf_count=rlf_count,
        rlf_rate=rlf_count / max(total_time_s, 1.0),
        hopp_count=hopp_count,
        hopp_rate=hopp_count / max(total_time_s, 1.0),
        rb_count=rb_count,
        rb_rate_per_min=60.0 * rb_count / max(total_time_s, 1.0),
        episode_reward=episode_reward,
    )
