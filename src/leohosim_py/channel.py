"""Channel model for RSRP/SINR calculations."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.special import j1

C_LIGHT = 299_792_458.0


def dbm_to_mw(x_dbm):
    return np.power(10.0, np.asarray(x_dbm) / 10.0)


def path_loss_free_space(distance_m: np.ndarray, freq_hz: float) -> np.ndarray:
    wavelength = C_LIGHT / freq_hz
    return 20.0 * np.log10(4.0 * math.pi * np.maximum(distance_m, 1.0) / wavelength)


def antenna_gain_circular_aperture(elevation_deg: np.ndarray, aperture_m: float, freq_hz: float) -> np.ndarray:
    theta = 90.0 - np.asarray(elevation_deg, dtype=np.float64)
    maxgain_db = 30.0
    ka = 2.0 * math.pi * freq_hz / C_LIGHT * aperture_m / 2.0
    z = ka * np.sin(np.radians(theta))
    y = np.ones_like(z)
    nz = np.abs(z) > 1e-12
    y[nz] = 4.0 * np.abs(j1(z[nz]) / z[nz]) ** 2
    return 10.0 * np.log10(np.maximum(y, 1e-300)) + maxgain_db


def los_probability_rural(elevation_deg: np.ndarray) -> np.ndarray:
    vals = np.array([78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8])
    idx = np.clip(np.rint(elevation_deg / 10.0).astype(int), 1, 9) - 1
    return vals[idx] / 100.0


def path_loss_matlab_rural(distance_m: np.ndarray, freq_hz: float, elevation_deg: np.ndarray) -> np.ndarray:
    fs = path_loss_free_space(distance_m, freq_hz)
    p_los = los_probability_rural(elevation_deg)
    nlos_cl = np.array([19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30])
    idx = np.clip(np.rint(elevation_deg / 10.0).astype(int), 1, 9) - 1
    return p_los * fs + (1.0 - p_los) * (fs + nlos_cl[idx])


def beam_loss_model(ml: np.ndarray, cell_radius_m: float, factor_db: float = 6.0, clip_db: float = 35.0) -> np.ndarray:
    return np.minimum(factor_db * (ml / cell_radius_m) ** 2, clip_db)


def calculate_thermal_noise(bandwidth_hz: float, noise_figure_db: float) -> float:
    return -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db


def calculate_interference_and_sinr(
    rsrp_mw: np.ndarray,
    noise_mw: float,
    interference_top_k: int = 0,
    ml_m: np.ndarray | None = None,
    interference_neighbor_count: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rsrp_mw, dtype=np.float64)
    top_k = int(interference_top_k)
    neighbor_count = int(interference_neighbor_count)
    ml_values = None if ml_m is None else np.asarray(ml_m, dtype=np.float64)

    interf = np.zeros_like(values)
    all_indices = np.arange(values.size)
    for idx in range(values.size):
        eligible = all_indices[all_indices != idx]
        if eligible.size == 0:
            continue
        if ml_values is not None and 0 < neighbor_count < eligible.size:
            order_by_distance = eligible[np.argsort(ml_values[eligible])]
            eligible = order_by_distance[:neighbor_count]
        if top_k <= 0 or top_k >= eligible.size:
            interferers = eligible
        else:
            order = eligible[np.argsort(values[eligible])[::-1]]
            interferers = order[:top_k]
        interf[idx] = float(np.sum(values[interferers]))
    sinr = rsrp_mw / np.maximum(interf + noise_mw, 1e-300)
    return interf, 10.0 * np.log10(np.maximum(sinr, 1e-300))


class ChannelCalculator:
    def __init__(
        self,
        tx_eirp_dbm: float,
        carrier_freq_hz: float,
        bandwidth_hz: float,
        noise_figure_db: float,
        cell_radius_m: float,
        aperture_m: float,
        channel_mode: str = "matlab_like_deterministic",
        rng: np.random.Generator | None = None,
        interference_top_k: int = 0,
        interference_neighbor_count: int = 0,
    ):
        self.tx_eirp_dbm = tx_eirp_dbm
        self.carrier_freq_hz = carrier_freq_hz
        self.cell_radius_m = cell_radius_m
        self.aperture_m = aperture_m
        self.channel_mode = channel_mode
        self.rng = rng
        self.interference_top_k = int(interference_top_k)
        self.interference_neighbor_count = int(interference_neighbor_count)
        self.noise_mw = dbm_to_mw(calculate_thermal_noise(bandwidth_hz, noise_figure_db))

    def update_channel(self, bore_x, bore_y, ue_x: float, ue_y: float, altitude_m: float):
        from . import geometry

        ml = geometry.calculate_ml_distance(bore_x, bore_y, ue_x, ue_y)
        slant = geometry.calculate_slant_distance(bore_x, bore_y, ue_x, ue_y, altitude_m)
        elev = geometry.calculate_elevation_angle(ml, altitude_m)
        if self.channel_mode.startswith("matlab_like"):
            path_loss = path_loss_matlab_rural(slant, self.carrier_freq_hz, elev)
        else:
            path_loss = path_loss_free_space(slant, self.carrier_freq_hz)
        gain = antenna_gain_circular_aperture(elev, self.aperture_m, self.carrier_freq_hz)
        beam_loss = beam_loss_model(ml, self.cell_radius_m)
        rsrp_dbm = self.tx_eirp_dbm - path_loss - beam_loss + gain
        _, sinr_db = calculate_interference_and_sinr(
            dbm_to_mw(rsrp_dbm),
            self.noise_mw,
            self.interference_top_k,
            ml,
            self.interference_neighbor_count,
        )
        return rsrp_dbm, sinr_db, ml
