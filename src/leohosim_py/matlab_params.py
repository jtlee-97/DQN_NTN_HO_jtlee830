"""MATLAB reference parameter defaults."""

from dataclasses import dataclass


@dataclass
class MATLABSystemParameters:
    cell_radius_m: float = 23_120.0
    cell_isd_m: float = 40_045.0147
    altitude_m: float = 600_000.0
    sat_speed_mps: float = 7_560.0
    carrier_freq_hz: float = 2.0e9
    bandwidth_hz: float = 20.0e6
    rb_bandwidth_hz: float = 180_000.0
    tx_gain_db: float = 30.0
    aperture_m: float = 2.0
    eirp_dbm: float = 34.0
    noise_figure_db: float = 7.0
    sample_time_s: float = 0.2
    total_time_s: float = 24.0
    ue_x_m: float = 17_340.0
    ue_y_m: float = 80_090.0293
    ue_speed_mps: float = 0.0
    hys_m: float = 0.0
    ttt_s: float = 0.0
    q_out_db: float = -8.0
    q_in_db: float = -6.0
    t310_s: float = 1.0
    min_tos_s: float = 1.0
    num_sites: int = 127


def get_matlab_default_params() -> MATLABSystemParameters:
    return MATLABSystemParameters()
