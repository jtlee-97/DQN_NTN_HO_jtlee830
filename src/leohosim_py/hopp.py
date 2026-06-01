"""Handover ping-pong detector."""


class HOPPDetector:
    def __init__(self, hopp_window_s: float = 2.0):
        self.hopp_window_s = hopp_window_s
        self.recent_cells: list[tuple[int, float]] = []
        self.hopp_count = 0

    def record_handover(self, serving_idx: int, current_time_s: float) -> bool:
        cutoff = current_time_s - self.hopp_window_s
        self.recent_cells = [(idx, t) for idx, t in self.recent_cells if t > cutoff]
        detected = any(idx == serving_idx for idx, _ in self.recent_cells)
        self.recent_cells.append((serving_idx, current_time_s))
        if detected:
            self.hopp_count += 1
        return detected

    def get_hopp_count(self) -> int:
        return self.hopp_count

    def reset(self) -> None:
        self.recent_cells = []
        self.hopp_count = 0
