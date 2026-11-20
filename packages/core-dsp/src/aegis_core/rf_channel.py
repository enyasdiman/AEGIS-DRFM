"""RF channel propagation, hardware impairments, and coherent CPI radar echo simulator."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .types import C_LIGHT, RadarConfig, RFImpairments, TargetState


def apply_fractional_delay(signal: np.ndarray, delay_samples: float, out_len: Optional[int] = None) -> np.ndarray:
    """Shift a complex baseband signal by an arbitrary fractional delay (samples >= 0).

    Uses exact integer placement + FFT linear phase ramp for the fractional part [0, 1).
    """
    n_in = len(signal)
    n_out = out_len if out_len is not None else n_in
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative.")

    int_delay = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay

    if frac_delay > 1e-12:
        n_fft = 1 << int(np.ceil(np.log2(n_in + 64)))
        freqs = np.fft.fftfreq(n_fft)
        spec = np.fft.fft(signal, n=n_fft)
        shifted = np.fft.ifft(spec * np.exp(-1j * 2.0 * np.pi * freqs * frac_delay))
        seg = shifted[: min(n_in + 32, n_fft)]
    else:
        seg = signal

    out = np.zeros(n_out, dtype=np.complex128)
    if int_delay < n_out:
        copy_len = min(len(seg), n_out - int_delay)
        out[int_delay : int_delay + copy_len] = seg[:copy_len]
    return out


def apply_doppler_shift(
    signal: np.ndarray,
    fd_hz: float,
    fs_hz: float,
    t0_s: float = 0.0,
) -> np.ndarray:
    """Apply continuous-phase Doppler shift exp(j * 2 * pi * fd * (t0 + n/fs))."""
    n = np.arange(len(signal), dtype=np.float64)
    phase = 2.0 * np.pi * fd_hz * (t0_s + n / fs_hz)
    return signal * np.exp(1j * phase)


def apply_lo_phase_noise(
    signal: np.ndarray,
    fs_hz: float,
    phase_noise_std_rad: float = 0.005,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply simplified Leeson-like LO phase noise (white floor + slow Wiener wander)."""
    if phase_noise_std_rad <= 0.0:
        return signal.copy()
    gen = rng if rng is not None else np.random.default_rng()
    n = len(signal)
    white_phase = gen.normal(0.0, phase_noise_std_rad, size=n)
    wander_step = phase_noise_std_rad * 0.05
    wiener_phase = np.cumsum(gen.normal(0.0, wander_step, size=n))
    return signal * np.exp(1j * (white_phase + wiener_phase))


def apply_iq_imbalance(
    signal: np.ndarray,
    amp_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
) -> np.ndarray:
    """Apply receiver/transmitter I/Q gain and quadrature phase imbalance."""
    if amp_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
        return signal.copy()
    g = 10.0 ** (amp_imbalance_db / 20.0)
    phi = np.deg2rad(phase_imbalance_deg)
    alpha = np.cos(phi / 2.0) + 1j * g * np.sin(phi / 2.0)
    beta = g * np.cos(phi / 2.0) - 1j * np.sin(phi / 2.0)
    # Standard mirror-image mixing model: y = k1 * x + k2 * conj(x)
    k1 = 0.5 * (1.0 + g * np.exp(-1j * phi))
    k2 = 0.5 * (1.0 - g * np.exp(1j * phi))
    _ = (alpha, beta)
    return k1 * signal + k2 * np.conj(signal)


def apply_multipath(
    signal: np.ndarray,
    taps: Sequence[Tuple[float, complex]],
) -> np.ndarray:
    """Add multipath delayed copies: taps = [(extra_delay_samples, complex_gain), ...]."""
    if not taps:
        return signal.copy()
    out = signal.copy()
    n = len(signal)
    for delay_samp, gain in taps:
        out += gain * apply_fractional_delay(signal, delay_samp, out_len=n)
    return out


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    signal_power: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add complex circular AWGN to achieve target SNR (dB) relative to signal_power."""
    gen = rng if rng is not None else np.random.default_rng()
    p_sig = float(np.mean(np.abs(signal) ** 2)) if signal_power is None else float(signal_power)
    if p_sig <= 0.0:
        p_sig = 1.0
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(p_noise / 2.0)
    noise = sigma * (gen.standard_normal(signal.shape) + 1j * gen.standard_normal(signal.shape))
    return signal + noise


def simulate_radar_cpi(
    tx_pulse: np.ndarray,
    radar_cfg: RadarConfig,
    targets: Sequence[TargetState],
    impairments: Optional[RFImpairments] = None,
    drfm_callback: Optional[Callable[[np.ndarray, int, float, TargetState], np.ndarray]] = None,
    max_range_m: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Simulate a Coherent Processing Interval (CPI) of radar echoes + optional DRFM replay.

    Args:
        tx_pulse: Baseband complex TX pulse (unit amplitude).
        radar_cfg: RadarConfig instance.
        targets: Sequence of point targets.
        impairments: Optional RFImpairments (AWGN, phase noise, I/Q imbalance, multipath).
        drfm_callback: Optional callable(incident_pulse, pulse_idx, t_pulse_s, target) -> jammer_rx_wave.
            If provided, it is invoked for targets[0] (self-screening jammer platform) and its
            returned waveform (length num_fast_samples) is summed into the radar RX buffer.
        max_range_m: Fast-time receive window max range. Defaults to unambiguous range (capped for memory).
        rng: Optional NumPy random Generator for reproducibility.

    Returns:
        (rx_cube, meta) where rx_cube has shape (num_pulses, num_fast_samples).
    """
    gen = rng if rng is not None else np.random.default_rng()
    imp = impairments if impairments is not None else RFImpairments()

    # Determine fast-time window length
    if max_range_m is None:
        max_target_r = max((t.range_m for t in targets), default=12000.0)
        max_range_m = min(radar_cfg.unambiguous_range_m, max(max_target_r + 4000.0, 18000.0))

    max_delay_s = 2.0 * max_range_m / C_LIGHT
    n_fast = int(np.ceil(max_delay_s * radar_cfg.fs_hz)) + len(tx_pulse) + 64
    n_pulses = radar_cfg.num_pulses
    rx_cube = np.zeros((n_pulses, n_fast), dtype=np.complex128)

    # Reference amplitude normalization: targets[0] has unit echo amplitude = 1.0
    ref_target = targets[0] if targets else TargetState(12000.0, 0.0, 5.0)
    ref_factor = np.sqrt(ref_target.rcs_m2) / (ref_target.range_m ** 2)

    for m in range(n_pulses):
        t_pulse = m * radar_cfg.pri_s
        rx_row = np.zeros(n_fast, dtype=np.complex128)

        for idx, tgt in enumerate(targets):
            # Instantaneous range at pulse m (negative velocity = closing)
            r_inst = tgt.range_m + tgt.velocity_mps * t_pulse
            tau_s = 2.0 * r_inst / C_LIGHT
            delay_samp = tau_s * radar_cfg.fs_hz

            # Two-way Doppler frequency: fd = -2 * v / lambda (closing v < 0 -> fd > 0)
            fd_hz = -2.0 * tgt.velocity_mps / radar_cfg.wavelength_m
            amp = (np.sqrt(tgt.rcs_m2) / (r_inst ** 2)) / ref_factor

            # Coherent carrier phase from two-way propagation: exp(-j * 4*pi*R / lambda)
            carrier_phase = np.exp(-1j * 4.0 * np.pi * r_inst / radar_cfg.wavelength_m)
            echo = amp * carrier_phase * apply_doppler_shift(tx_pulse, fd_hz, radar_cfg.fs_hz, t0_s=0.0)
            echo_delayed = apply_fractional_delay(echo, delay_samp, out_len=n_fast)

            if imp.multipath_taps:
                echo_delayed = apply_multipath(echo_delayed, imp.multipath_taps)

            rx_row += echo_delayed

            # Self-screening DRFM mounted on targets[0]
            if idx == 0 and drfm_callback is not None:
                jam_rx = drfm_callback(tx_pulse, m, t_pulse, tgt)
                if len(jam_rx) != n_fast:
                    # Pad or trim safely
                    buf = np.zeros(n_fast, dtype=np.complex128)
                    L = min(len(jam_rx), n_fast)
                    buf[:L] = jam_rx[:L]
                    jam_rx = buf
                rx_row += jam_rx

        if imp.phase_noise_std_rad > 0.0:
            rx_row = apply_lo_phase_noise(rx_row, radar_cfg.fs_hz, imp.phase_noise_std_rad, rng=gen)
        if imp.iq_amp_imbalance_db != 0.0 or imp.iq_phase_imbalance_deg != 0.0:
            rx_row = apply_iq_imbalance(rx_row, imp.iq_amp_imbalance_db, imp.iq_phase_imbalance_deg)
        if imp.snr_single_pulse_db is not None:
            # Reference single-pulse signal power = 1.0 (unit amplitude inside pulse)
            rx_row = add_awgn(rx_row, imp.snr_single_pulse_db, signal_power=1.0, rng=gen)

        rx_cube[m, :] = rx_row

    meta = {
        "n_pulses": n_pulses,
        "n_fast": n_fast,
        "max_range_m": max_range_m,
        "fs_hz": radar_cfg.fs_hz,
        "prf_hz": radar_cfg.prf_hz,
        "wavelength_m": radar_cfg.wavelength_m,
    }
    return rx_cube, meta
