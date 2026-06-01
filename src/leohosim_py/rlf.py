"""Radio Link Failure detector."""

from . import entities


class RLFDetector:
    def __init__(self, q_out_db: float = -8.0, q_in_db: float = -6.0, t310_s: float = 1.0):
        self.q_out_db = q_out_db
        self.q_in_db = q_in_db
        self.t310_s = t310_s

    def update_rlf(self, ue: entities.UE, serving_sinr_db: float, dt_s: float, current_time_s: float) -> bool:
        if serving_sinr_db < self.q_out_db:
            ue.rlf.increment_timer(dt_s)
            if ue.rlf.timer_s >= self.t310_s:
                ue.rlf.trigger_event(current_time_s)
                ue.rlf_event_count += 1
                return True
        elif serving_sinr_db >= self.q_in_db:
            ue.rlf.reset_timer()
        return False
