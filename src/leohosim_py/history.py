"""Step history logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class HistoryEntry:
    time_s: float
    step_idx: int
    action_idx: int
    action_thresh1_m: float
    action_thresh2_m: float
    serving_idx: int
    serving_ml_m: float
    serving_rsrp_dbm: float
    serving_sinr_db: float
    best_target_idx: int
    best_target_ml_m: float
    best_target_sinr_db: float
    prep_event: int = 0
    prep_fail_event: int = 0
    ho_event: int = 0
    uho_event: int = 0
    rlf_event: int = 0
    rb_delta: int = 0
    rb_cumulative: int = 0
    reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class History:
    def __init__(self):
        self.entries: List[HistoryEntry] = []

    def log_step(self, entry: HistoryEntry) -> None:
        self.entries.append(entry)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    def clear(self) -> None:
        self.entries = []
