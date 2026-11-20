"""Uniform Linear Array (ULA) AESA Beamforming and Phase-Comparison Monopulse Discriminator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from aegis_core.types import C_LIGHT


@dataclass
class MonopulseMeasurement:
    """Monopulse Sum/Difference channel output and angle estimation."""

    sum_val: complex
    diff_val: complex
    ratio_real: float       # In-phase monopulse ratio (proportional to off-boresight angle)
    ratio_imag: float       # Quadrature monopulse ratio (~0 for clean target, large under Cross-Eye!)
    est_angle_deg: float    # Estimated off-boresight angle (deg)
    sum_power_db: float     # Sum channel power (dB)


class AESAMonopulseArray:
    """N-element ULA AESA Radar with Sum (Sigma) and Difference (Delta) Monopulse Channels."""

    def __init__(
        self,
        num_elements: int = 16,
        fc_hz: float = 10.0e9,
        spacing_wavelengths: float = 0.5,
        boresight_deg: float = 0.0,
    ) -> None:
        if num_elements < 2 or num_elements % 2 != 0:
            raise ValueError("num_elements must be an even integer >= 2 for symmetric monopulse.")
        self.n_elem = num_elements
        self.fc_hz = fc_hz
        self.wavelength_m = C_LIGHT / fc_hz
        self.d_m = spacing_wavelengths * self.wavelength_m
        self.boresight_deg = boresight_deg

        # Symmetric element coordinates centered at 0
        self.elem_pos = (np.arange(num_elements, dtype=np.float64) - 0.5 * (num_elements - 1)) * self.d_m

        # Precompute Sum and Difference weights steered to boresight
        a0 = self.steering_vector(boresight_deg)
        half = num_elements // 2
        diff_sign = np.concatenate((-np.ones(half), np.ones(half)))

        self.w_sum = a0 / float(num_elements)
        self.w_diff = (a0 * diff_sign) / float(num_elements)

        # Calibrate linear monopulse slope k_m (ratio_real / deg) near boresight
        eps_deg = 0.05
        m_plus = self._raw_ratio(boresight_deg + eps_deg)
        m_minus = self._raw_ratio(boresight_deg - eps_deg)
        self.k_monopulse = float(np.real(m_plus - m_minus) / (2.0 * eps_deg))

    def steering_vector(self, angle_deg: float) -> np.ndarray:
        """Array manifold vector a(theta) for plane wave arriving at angle_deg."""
        sin_th = np.sin(np.deg2rad(angle_deg))
        phase = 2.0 * np.pi * (self.elem_pos / self.wavelength_m) * sin_th
        return np.exp(1j * phase)

    def _raw_ratio(self, angle_deg: float) -> complex:
        a = self.steering_vector(angle_deg)
        s = complex(np.vdot(self.w_sum, a))
        d = complex(np.vdot(self.w_diff, a))
        return d / (1j * s + 1e-18)

    def beam_patterns(self, angles_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Sum (dB), Difference (dB), and Monopulse S-curve over angles_deg."""
        s_vals = np.zeros(len(angles_deg), dtype=np.complex128)
        d_vals = np.zeros(len(angles_deg), dtype=np.complex128)
        for i, th in enumerate(angles_deg):
            a = self.steering_vector(float(th))
            s_vals[i] = np.vdot(self.w_sum, a)
            d_vals[i] = np.vdot(self.w_diff, a)
        sum_db = 20.0 * np.log10(np.abs(s_vals) + 1e-12)
        diff_db = 20.0 * np.log10(np.abs(d_vals) + 1e-12)
        s_curve = np.real(d_vals / (1j * s_vals + 1e-12))
        return sum_db, diff_db, s_curve

    def measure_snapshot(self, array_snapshot: np.ndarray) -> MonopulseMeasurement:
        """Apply Sum/Diff beamformers to a complex N-element array snapshot."""
        s = complex(np.vdot(self.w_sum, array_snapshot))
        d = complex(np.vdot(self.w_diff, array_snapshot))
        ratio = d / (1j * s + 1e-18)
        r_real = float(np.real(ratio))
        r_imag = float(np.imag(ratio))
        est_deg = self.boresight_deg + (r_real / self.k_monopulse)
        sum_pow_db = float(20.0 * np.log10(abs(s) + 1e-12))
        return MonopulseMeasurement(
            sum_val=s,
            diff_val=d,
            ratio_real=r_real,
            ratio_imag=r_imag,
            est_angle_deg=est_deg,
            sum_power_db=sum_pow_db,
        )
