"""Geometry helpers for MATLAB-compatible hexagonal LEO beam layouts."""

from __future__ import annotations

import math
import numpy as np


def generate_hexagonal_cells(cell_radius_m: float = 23_120.0, tiers: int = 6) -> np.ndarray:
    centers = [[0.0, 0.0]]
    for tier in range(1, tiers + 1):
        for side in range(6):
            for step in range(tier):
                angle = math.radians(side * 60.0 + 30.0)
                dx = cell_radius_m * math.sqrt(3.0) * (
                    tier * math.cos(angle) - step * math.sin(angle + math.pi / 6.0)
                )
                dy = cell_radius_m * math.sqrt(3.0) * (
                    tier * math.sin(angle) + step * math.cos(angle + math.pi / 6.0)
                )
                centers.append([dx, dy])
    return np.asarray(centers, dtype=np.float64)


def tiers_for_site_count(num_sites: int) -> int:
    tiers = 0
    while 1 + 3 * tiers * (tiers + 1) < num_sites:
        tiers += 1
    return tiers


def get_61_site_layout(cell_radius_m: float = 23_120.0, num_sites: int = 127, tiers: int | None = None) -> np.ndarray:
    if tiers is None:
        tiers = max(6, tiers_for_site_count(num_sites))
    return generate_hexagonal_cells(cell_radius_m, tiers)[:num_sites].copy()


def get_overlapping_19beam_layout(
    beam_spacing_m: float,
    satellite_count: int,
    beams_per_satellite: int = 19,
    satellite_spread_m: float = 8000.0,
    beam_jitter_m: float = 0.0,
    ring_jitter_m: float = 0.0,
    track_group_count: int = 1,
    track_spacing_m: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate overlapping 19-beam satellite footprints.

    Each satellite contributes a compact 19-beam hexagonal footprint. The
    satellite footprint centers are placed near the same service region with
    small deterministic/rng-driven offsets so several candidate cells overlap
    for the same UE position.
    """
    count = max(int(satellite_count), 1)
    beams = max(int(beams_per_satellite), 1)
    base_tiers = tiers_for_site_count(beams)
    base = generate_hexagonal_cells(float(beam_spacing_m), max(base_tiers, 2))[:beams].copy()

    layouts = []
    group_count = max(int(track_group_count), 1)
    group_center = 0.5 * float(group_count - 1)
    for group_idx in range(group_count):
        group_offset = np.asarray([0.0, (float(group_idx) - group_center) * float(track_spacing_m)], dtype=np.float64)
        offsets = []
        for sat_idx in range(count):
            if sat_idx == 0:
                offset = np.asarray([0.0, 0.0], dtype=np.float64)
            else:
                ring = 1 + (sat_idx - 1) // 6
                pos = (sat_idx - 1) % 6
                angle = np.radians(60.0 * pos + 30.0 * (ring - 1) + 11.0 * group_idx)
                radius = float(satellite_spread_m) * ring
                offset = np.asarray([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64)
            if rng is not None and ring_jitter_m > 0.0:
                offset = offset + rng.normal(0.0, float(ring_jitter_m), size=2)
            offsets.append(group_offset + offset)

        for sat_idx, offset in enumerate(offsets):
            angle = np.radians((sat_idx % 6) * 10.0 + 4.0 * group_idx)
            rot = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=np.float64,
            )
            footprint = base @ rot.T + offset
            if rng is not None and beam_jitter_m > 0.0:
                footprint = footprint + rng.normal(0.0, float(beam_jitter_m), size=footprint.shape)
            layouts.append(footprint)
    return np.vstack(layouts).astype(np.float64)


def calculate_ml_distance(bore_x: np.ndarray, bore_y: np.ndarray, ue_x: float, ue_y: float) -> np.ndarray:
    return np.sqrt((bore_x - ue_x) ** 2 + (bore_y - ue_y) ** 2)


def calculate_slant_distance(
    bore_x: np.ndarray, bore_y: np.ndarray, ue_x: float, ue_y: float, altitude_m: float
) -> np.ndarray:
    ml = calculate_ml_distance(bore_x, bore_y, ue_x, ue_y)
    return np.sqrt(ml**2 + altitude_m**2)


def calculate_elevation_angle(ml: np.ndarray, altitude_m: float) -> np.ndarray:
    return np.degrees(np.arctan2(altitude_m, np.maximum(ml, 1.0)))


def calculate_azimuth_angle(bore_x: np.ndarray, bore_y: np.ndarray, ue_x: float, ue_y: float) -> np.ndarray:
    az = np.degrees(np.arctan2(bore_y - ue_y, bore_x - ue_x))
    return np.where(az < 0, az + 360.0, az)
