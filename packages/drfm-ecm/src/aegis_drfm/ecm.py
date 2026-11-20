"""DRFM Core Engine and Electronic Countermeasures (ECM) Techniques.

Supports:
  - RGPO (Range Gate Pull-Off: linear & parabolic)
  - VGPO (Velocity Gate Pull-Off)
  - RGPO_VGPO (Kinematically coupled Range + Velocity pull-off)
  - FALSE_TARGETS (Coherent multiple ghost targets)
  - ISRJ (Interrupted Sampling Repeater Jamming)
  - NOISE (Barrage / Cover Pulse jamming)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from aegis_core.rf_channel import apply_doppler_shift, apply_fractional_delay, apply_lo_phase_noise
from aegis_core.types import C_LIGHT, RadarConfig, TargetState
from .quantizer import quantize_complex_drfm


@dataclass
class DRFMConfig:
    """Hardware-realistic DRFM configuration."""

    adc_bits: int = 8
    loop_delay_ns: float = 120.0     # Minimum capture-to-replay hardware latency
    jsr_db: float = 10.0             # Jammer-to-Signal ratio relative to platform echo
    phase_noise_std_rad: float = 0.002
    full_scale: float = 1.0


@dataclass
class FalseTargetSpec:
    """Specification for a single DRFM-synthesized coherent false target."""

    delta_range_m: float             # Positive = farther than true platform (causal)
    delta_velocity_mps: float = 0.0  # Velocity offset relative to platform
    rel_power_db: float = 0.0        # Power offset relative to base jammer JSR


class DRFMCore:
    """Phase-coherent Digital Radio Frequency Memory (DRFM) repeater & ECM generator."""

    def __init__(self, config: Optional[DRFMConfig] = None, rng: Optional[np.random.Generator] = None) -> None:
        self.cfg = config if config is not None else DRFMConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.memory_buffer: Optional[np.ndarray] = None

    @property
    def jammer_amp(self) -> float:
        """Amplitude scaling relative to unit-amplitude true target echo."""
        return float(10.0 ** (self.cfg.jsr_db / 20.0))

    def capture(self, incident_pulse: np.ndarray) -> np.ndarray:
        """Digitize and store an intercepted radar pulse in DRFM memory."""
        peak = float(np.max(np.abs(incident_pulse)))
        norm_in = incident_pulse / peak if peak > 0 else incident_pulse
        # Simulate AGC keeping input at ~0.95 full-scale before ADC quantization
        quantized = quantize_complex_drfm(0.95 * norm_in, bits=self.cfg.adc_bits, full_scale=self.cfg.full_scale)
        # Restore unit RMS/peak scale after DAC
        q_peak = float(np.max(np.abs(quantized)))
        stored = quantized / q_peak if q_peak > 0 else quantized
        if self.cfg.phase_noise_std_rad > 0.0:
            stored = apply_lo_phase_noise(stored, fs_hz=1.0, phase_noise_std_rad=self.cfg.phase_noise_std_rad, rng=self.rng)
        self.memory_buffer = stored
        return stored

    def _synthesize_return(
        self,
        pulse_base: np.ndarray,
        radar_cfg: RadarConfig,
        true_range_m: float,
        true_vel_mps: float,
        delta_range_m: float,
        delta_vel_mps: float,
        t_pulse_s: float,
        n_fast: int,
        amp_scale: float = 1.0,
        kinematic_coupled: bool = False,
    ) -> np.ndarray:
        """Synthesize a coherent jammer return at (true_range + delta_range, true_vel + delta_vel)."""
        # Add minimum hardware loop latency to causal range offset
        min_hw_range_m = 0.5 * C_LIGHT * (self.cfg.loop_delay_ns * 1e-9)
        eff_delta_r = max(delta_range_m, 0.0) + min_hw_range_m
        r_false = true_range_m + eff_delta_r
        v_false = true_vel_mps + delta_vel_mps

        tau_s = 2.0 * r_false / C_LIGHT
        delay_samp = tau_s * radar_cfg.fs_hz

        # Coherent slow-time + fast-time Doppler phase for the spoofed target
        fd_true = -2.0 * true_vel_mps / radar_cfg.wavelength_m
        fd_false = -2.0 * v_false / radar_cfg.wavelength_m
        # Slow-time phase progression across CPI
        slow_phase = np.exp(-1j * 4.0 * np.pi * (true_range_m + eff_delta_r) / radar_cfg.wavelength_m)
        # When delta_range_m(t) already evolves at delta_vel_mps (kinematically coupled),
        # slow_phase already contains the exact Doppler shift; avoid double-counting.
        extra_doppler_slow = (
            1.0
            if kinematic_coupled
            else np.exp(1j * 2.0 * np.pi * (fd_false - fd_true) * t_pulse_s)
        )

        tx_mod = (
            self.jammer_amp
            * amp_scale
            * slow_phase
            * extra_doppler_slow
            * apply_doppler_shift(pulse_base, fd_false, radar_cfg.fs_hz, t0_s=0.0)
        )
        return apply_fractional_delay(tx_mod, delay_samp, out_len=n_fast)

    def replay_isrj(
        self,
        incident_pulse: np.ndarray,
        radar_cfg: RadarConfig,
        true_range_m: float,
        true_vel_mps: float,
        n_fast: int,
        slice_period_us: float = 2.0,
        duty_cycle: float = 0.5,
        repeat_count: int = 1,
    ) -> np.ndarray:
        """Interrupted Sampling Repeater Jamming (ISRJ): slice & coherently retransmit intra-pulse."""
        stored = self.capture(incident_pulse)
        n_p = len(stored)
        period_samp = max(4, int(round(slice_period_us * 1e-6 * radar_cfg.fs_hz)))
        on_samp = max(2, int(round(period_samp * duty_cycle)))

        # Build rectangular sampling train
        sample_mask = np.zeros(n_p, dtype=np.float64)
        for start in range(0, n_p, period_samp):
            sample_mask[start : min(start + on_samp, n_p)] = 1.0

        sliced = stored * sample_mask

        # Retransmit each sampled slice right after sampling (delayed by on_samp * k)
        isrj_pulse = np.zeros(n_p + on_samp * repeat_count, dtype=np.complex128)
        for k in range(1, repeat_count + 1):
            shift = k * on_samp
            isrj_pulse[shift : shift + n_p] += sliced

        # Place at true platform range + hardware loop delay
        min_hw_range_m = 0.5 * C_LIGHT * (self.cfg.loop_delay_ns * 1e-9)
        r_jam = true_range_m + min_hw_range_m
        delay_samp = (2.0 * r_jam / C_LIGHT) * radar_cfg.fs_hz
        fd_true = -2.0 * true_vel_mps / radar_cfg.wavelength_m
        slow_phase = np.exp(-1j * 4.0 * np.pi * r_jam / radar_cfg.wavelength_m)

        tx_mod = self.jammer_amp * slow_phase * apply_doppler_shift(isrj_pulse, fd_true, radar_cfg.fs_hz, t0_s=0.0)
        return apply_fractional_delay(tx_mod, delay_samp, out_len=n_fast)

    def make_cpi_callback(
        self,
        radar_cfg: RadarConfig,
        mode: str,
        n_fast: int,
        **kwargs,
    ) -> Callable[[np.ndarray, int, float, TargetState], np.ndarray]:
        """Create a per-pulse DRFM callback compatible with simulate_radar_cpi().

        Supported modes:
          - 'RGPO': kwargs = {range_offset_m, pull_rate_mps, accel_mps2}
          - 'VGPO': kwargs = {vel_offset_mps, vel_rate_mps2}
          - 'RGPO_VGPO': kwargs = {range_offset_m, vel_offset_mps, accel_mps2}
          - 'FALSE_TARGETS': kwargs = {specs: Sequence[FalseTargetSpec]}
          - 'ISRJ': kwargs = {slice_period_us, duty_cycle, repeat_count}
          - 'NOISE': kwargs = {cover_only: bool}
        """
        mode_u = mode.upper()

        def callback(incident_pulse: np.ndarray, pulse_idx: int, t_pulse_s: float, tgt: TargetState) -> np.ndarray:
            r_inst = tgt.range_m + tgt.velocity_mps * t_pulse_s

            if mode_u == "OFF":
                return np.zeros(n_fast, dtype=np.complex128)

            if mode_u == "NOISE":
                sigma = self.jammer_amp / np.sqrt(2.0)
                noise = sigma * (self.rng.standard_normal(n_fast) + 1j * self.rng.standard_normal(n_fast))
                return noise

            if mode_u == "ISRJ":
                return self.replay_isrj(
                    incident_pulse=incident_pulse,
                    radar_cfg=radar_cfg,
                    true_range_m=r_inst,
                    true_vel_mps=tgt.velocity_mps,
                    n_fast=n_fast,
                    slice_period_us=float(kwargs.get("slice_period_us", 2.0)),
                    duty_cycle=float(kwargs.get("duty_cycle", 0.5)),
                    repeat_count=int(kwargs.get("repeat_count", 1)),
                )

            stored = self.capture(incident_pulse)

            if mode_u == "RGPO":
                r0 = float(kwargs.get("range_offset_m", 600.0))
                vr = float(kwargs.get("pull_rate_mps", 0.0))
                ar = float(kwargs.get("accel_mps2", 0.0))
                dr = r0 + vr * t_pulse_s + 0.5 * ar * (t_pulse_s ** 2)
                return self._synthesize_return(
                    stored, radar_cfg, r_inst, tgt.velocity_mps, dr, 0.0, t_pulse_s, n_fast
                )

            if mode_u == "VGPO":
                dv0 = float(kwargs.get("vel_offset_mps", 35.0))
                dv_rate = float(kwargs.get("vel_rate_mps2", 0.0))
                dv = dv0 + dv_rate * t_pulse_s
                return self._synthesize_return(
                    stored, radar_cfg, r_inst, tgt.velocity_mps, 0.0, dv, t_pulse_s, n_fast
                )

            if mode_u == "RGPO_VGPO":
                r0 = float(kwargs.get("range_offset_m", 450.0))
                dv0 = float(kwargs.get("vel_offset_mps", 30.0))
                ar = float(kwargs.get("accel_mps2", 0.0))
                dr = r0 + dv0 * t_pulse_s + 0.5 * ar * (t_pulse_s ** 2)
                dv = dv0 + ar * t_pulse_s
                return self._synthesize_return(
                    stored, radar_cfg, r_inst, tgt.velocity_mps, dr, dv, t_pulse_s, n_fast,
                    kinematic_coupled=True,
                )

            if mode_u == "FALSE_TARGETS":
                specs: Sequence[FalseTargetSpec] = kwargs.get(
                    "specs",
                    [
                        FalseTargetSpec(delta_range_m=300.0, delta_velocity_mps=15.0, rel_power_db=0.0),
                        FalseTargetSpec(delta_range_m=750.0, delta_velocity_mps=-25.0, rel_power_db=-2.0),
                        FalseTargetSpec(delta_range_m=1200.0, delta_velocity_mps=40.0, rel_power_db=-3.0),
                    ],
                )
                total = np.zeros(n_fast, dtype=np.complex128)
                for spec in specs:
                    scale = float(10.0 ** (spec.rel_power_db / 20.0))
                    total += self._synthesize_return(
                        stored,
                        radar_cfg,
                        r_inst,
                        tgt.velocity_mps,
                        spec.delta_range_m,
                        spec.delta_velocity_mps,
                        t_pulse_s,
                        n_fast,
                        amp_scale=scale,
                    )
                return total

            raise ValueError(f"Unknown DRFM ECM mode: '{mode}'")

        return callback
