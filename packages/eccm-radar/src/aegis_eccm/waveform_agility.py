"""Signal-Level ECCM: Pulse-to-Pulse Waveform Agility against Repeat-Back DRFM."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from aegis_core import (
    C_LIGHT,
    RadarConfig,
    TargetState,
    add_awgn,
    apply_doppler_shift,
    apply_fractional_delay,
    generate_lfm_chirp,
    matched_filter,
)
from aegis_drfm import DRFMConfig, DRFMCore


def generate_agile_pulse_bank(
    radar_cfg: RadarConfig,
    mode: str = "up_down_lfm",
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate a pulse-to-pulse agile waveform bank for a CPI.

    Modes:
      - 'fixed': Identical LFM up-chirp on every pulse (baseline, vulnerable to DRFM).
      - 'up_down_lfm': Alternates orthogonal Up-Chirp / Down-Chirp + random initial phase.
      - 'agile_subband': Hopped baseband LFM sub-bands + alternating chirp slope.
    """
    gen = rng if rng is not None else np.random.default_rng(42)
    pulses: List[np.ndarray] = []

    p_up = generate_lfm_chirp(radar_cfg.fs_hz, radar_cfg.pulse_width_s, radar_cfg.bandwidth_hz, up=True)
    p_dn = generate_lfm_chirp(radar_cfg.fs_hz, radar_cfg.pulse_width_s, radar_cfg.bandwidth_hz, up=False)
    n_p = len(p_up)
    t_loc = np.arange(n_p, dtype=np.float64) / radar_cfg.fs_hz

    for m in range(radar_cfg.num_pulses):
        if mode == "fixed":
            pulses.append(p_up.copy())
        elif mode == "up_down_lfm":
            base = p_up if (m % 2 == 0) else p_dn
            pulses.append(base.copy())
        elif mode == "agile_subband":
            base = p_up if (m % 2 == 0) else p_dn
            f_hop = float(gen.choice([-8e6, -4e6, 0.0, 4e6, 8e6]))
            pulses.append(base * np.exp(1j * 2.0 * np.pi * f_hop * t_loc))
        else:
            raise ValueError(f"Unknown agility mode '{mode}'.")
    return pulses


def evaluate_agility_suppression(
    radar_cfg: RadarConfig,
    target: TargetState,
    false_range_offset_m: float = 500.0,
    jsr_db: float = 12.0,
    snr_db: float = 8.0,
    mode: str = "up_down_lfm",
    seed: int = 42,
) -> Dict[str, object]:
    """Simulate a radar with pulse-to-pulse agility against a 1-PRI DRFM repeat-back jammer.

    A 1-PRI repeater stores pulse (m-1) and replays it during pulse m to create a
    range-leading or delayed false target. When the radar changes waveform on pulse m,
    the matched filter compresses the true echo while suppressing the mismatched DRFM replay.
    """
    rng = np.random.default_rng(seed)
    pulses = generate_agile_pulse_bank(radar_cfg, mode=mode, rng=rng)
    drfm = DRFMCore(DRFMConfig(adc_bits=8, loop_delay_ns=120.0, jsr_db=jsr_db), rng=rng)

    max_r = target.range_m + abs(false_range_offset_m) + 2000.0
    n_fast = int(np.ceil((2.0 * max_r / C_LIGHT) * radar_cfg.fs_hz)) + len(pulses[0]) + 64
    fd_true = -2.0 * target.velocity_mps / radar_cfg.wavelength_m

    comp_cube = np.zeros((radar_cfg.num_pulses, n_fast), dtype=np.complex128)
    # Causal 1-PRI DRFM: before pulse 0 is intercepted, memory holds prior CPI's last waveform (or silence)
    prev_captured = np.zeros_like(pulses[0])

    for m in range(radar_cfg.num_pulses):
        t_pulse = m * radar_cfg.pri_s
        tx_m = pulses[m]
        r_inst = target.range_m + target.velocity_mps * t_pulse
        tau_true = 2.0 * r_inst / C_LIGHT
        phase_true = np.exp(-1j * 4.0 * np.pi * r_inst / radar_cfg.wavelength_m)

        # True target echo uses current pulse tx_m
        echo = phase_true * apply_doppler_shift(tx_m, fd_true, radar_cfg.fs_hz)
        rx_row = apply_fractional_delay(echo, tau_true * radar_cfg.fs_hz, out_len=n_fast)

        # DRFM replays previously captured pulse (m-1) at false target range
        r_false = r_inst + false_range_offset_m
        tau_false = 2.0 * r_false / C_LIGHT
        phase_false = np.exp(-1j * 4.0 * np.pi * r_false / radar_cfg.wavelength_m)
        jam = drfm.jammer_amp * phase_false * apply_doppler_shift(prev_captured, fd_true, radar_cfg.fs_hz)
        rx_row += apply_fractional_delay(jam, tau_false * radar_cfg.fs_hz, out_len=n_fast)

        # DRFM captures current pulse for next PRI
        prev_captured = drfm.capture(tx_m)

        rx_row = add_awgn(rx_row, snr_db=snr_db, signal_power=1.0, rng=rng)
        # Pulse-specific matched filter
        comp_cube[m, :] = matched_filter(rx_row, tx_m, normalize=True)

    # Non-coherent range profile (average power across CPI)
    range_profile_pow = np.mean(np.abs(comp_cube) ** 2, axis=0)
    range_profile_db = 10.0 * np.log10(range_profile_pow + 1e-18)
    range_axis_m = np.arange(n_fast, dtype=np.float64) * (C_LIGHT / (2.0 * radar_cfg.fs_hz))

    # Measure peak power at true target gate vs false target gate (+/- 40 m window)
    true_mask = np.abs(range_axis_m - target.range_m) <= 40.0
    false_mask = np.abs(range_axis_m - (target.range_m + false_range_offset_m)) <= 40.0
    true_peak_db = float(np.max(range_profile_db[true_mask]))
    false_peak_db = float(np.max(range_profile_db[false_mask]))
    sjr_out_db = true_peak_db - false_peak_db

    return {
        "mode": mode,
        "range_axis_m": range_axis_m,
        "range_profile_db": range_profile_db,
        "true_peak_db": true_peak_db,
        "false_peak_db": false_peak_db,
        "sjr_out_db": sjr_out_db,
    }
