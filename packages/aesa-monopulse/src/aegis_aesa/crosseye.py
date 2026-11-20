"""Retrodirective Cross-Eye DRFM Jamming & Quadrature Monopulse ECCM Detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .monopulse import AESAMonopulseArray, MonopulseMeasurement


@dataclass
class CrossEyeConfig:
    """Two-element retrodirective Cross-Eye DRFM jammer parameters."""

    baseline_m: float = 15.0        # Separation between wingtip DRFM antennas (m)
    amplitude_ratio: float = 0.90   # Amplitude ratio a in (0, 1)
    phase_deg: float = 178.0        # Relative phase phi (near 180 deg for max pull-off)
    target_range_m: float = 3000.0  # Range to monopulse radar (m)
    target_angle_deg: float = 0.0   # True target angle relative to boresight (deg)
    platform_aspect_deg: float = 0.0  # 0 deg = broadside (full baseline visible)
    jsr_db: float = 20.0            # Jammer-to-skin-echo ratio (dB)


def compute_crosseye_gain(amplitude_ratio: float, phase_deg: float) -> Tuple[float, float]:
    """Analytical Cross-Eye Gain G_C = (1 - a*exp(j*phi)) / (1 + a*exp(j*phi)).

    Returns:
        (real_gain, imag_quadrature_factor)
        - real_gain > 1 pulls apparent angle outside the physical baseline!
        - imag_quadrature_factor != 0 creates a quadrature monopulse ECCM signature.
    """
    a_exp = amplitude_ratio * np.exp(1j * np.deg2rad(phase_deg))
    gc = (1.0 - a_exp) / (1.0 + a_exp + 1e-18)
    return float(np.real(gc)), float(np.imag(gc))


def simulate_crosseye_vs_monopulse(
    array: AESAMonopulseArray,
    cfg: CrossEyeConfig,
    include_skin_echo: bool = True,
    snr_db: Optional[float] = 30.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Simulate a 2-antenna retrodirective Cross-Eye DRFM illuminating an AESA monopulse radar."""
    gen = rng if rng is not None else np.random.default_rng(42)

    # Half angular separation theta_e subtended by jammer baseline
    eff_baseline = cfg.baseline_m * np.cos(np.deg2rad(cfg.platform_aspect_deg))
    theta_e_deg = float(np.rad2deg(np.arctan2(0.5 * eff_baseline, cfg.target_range_m)))

    th1 = cfg.target_angle_deg - theta_e_deg
    th2 = cfg.target_angle_deg + theta_e_deg

    a1 = array.steering_vector(th1)
    a2 = array.steering_vector(th2)

    # Retrodirective Cross-Eye: Antenna 1 -> DRFM(1) -> Antenna 2, Antenna 2 -> DRFM(a*e^jphi) -> Antenna 1
    # Both legs share identical round-trip propagation phase; only the controlled DRFM weight differs!
    jam_amp = float(10.0 ** (cfg.jsr_db / 20.0))
    w_ce = cfg.amplitude_ratio * np.exp(1j * np.deg2rad(cfg.phase_deg))
    snap_jam = jam_amp * (a1 + w_ce * a2)

    snap_skin = array.steering_vector(cfg.target_angle_deg) if include_skin_echo else np.zeros(array.n_elem, dtype=np.complex128)
    snapshot = snap_skin + snap_jam

    if snr_db is not None:
        p_noise = 1.0 / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(p_noise / 2.0)
        snapshot = snapshot + sigma * (gen.standard_normal(array.n_elem) + 1j * gen.standard_normal(array.n_elem))

    meas = array.measure_snapshot(snapshot)
    gc_real, gc_imag = compute_crosseye_gain(cfg.amplitude_ratio, cfg.phase_deg)
    predicted_pull_deg = cfg.target_angle_deg - gc_real * theta_e_deg

    return {
        "measurement": meas,
        "theta_e_deg": theta_e_deg,
        "crosseye_gain_real": gc_real,
        "crosseye_gain_imag": gc_imag,
        "predicted_angle_deg": predicted_pull_deg,
        "angle_error_deg": meas.est_angle_deg - cfg.target_angle_deg,
    }


def detect_crosseye_eccm(
    meas: MonopulseMeasurement,
    quadrature_thresh: float = 0.15,
) -> Dict[str, object]:
    """ECCM Counter-Cross-Eye detector using Monopulse Quadrature Channel |Im(Delta / j*Sigma)|.

    For any single real point target, Im(Delta / j*Sigma) == 0 (within thermal noise).
    Under Cross-Eye jamming with realistic phase/amplitude imbalance, the quadrature
    monopulse ratio spikes sharply, flagging angle deception.
    """
    quad_mag = abs(meas.ratio_imag)
    detected = quad_mag > quadrature_thresh
    return {
        "crosseye_detected": detected,
        "quadrature_ratio": quad_mag,
        "threshold": quadrature_thresh,
    }
