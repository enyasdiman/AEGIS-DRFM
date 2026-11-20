"""Radar receiver DSP: Matched Filtering (Pulse Compression), Range-Doppler Map, and CFAR."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import get_window

from .types import C_LIGHT, RadarConfig


def matched_filter(
    rx_cube: np.ndarray,
    ref_pulse: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Fast FFT-based matched filter (pulse compression) along fast-time axis.

    Args:
        rx_cube: Complex array of shape (n_pulses, n_fast) or (n_fast,).
        ref_pulse: Reference TX pulse waveform.
        normalize: If True, scales by 1/sqrt(E_ref) so noise variance is preserved.

    Returns:
        Pulse-compressed complex array with same shape as rx_cube.
    """
    is_1d = rx_cube.ndim == 1
    data = rx_cube[np.newaxis, :] if is_1d else rx_cube
    n_pulses, n_fast = data.shape
    n_ref = len(ref_pulse)

    n_fft = 1 << int(np.ceil(np.log2(n_fast + n_ref - 1)))
    h = np.conj(ref_pulse[::-1])
    if normalize:
        energy = float(np.sum(np.abs(ref_pulse) ** 2))
        if energy > 0:
            h = h / np.sqrt(energy)

    H = np.fft.fft(h, n=n_fft)
    X = np.fft.fft(data, n=n_fft, axis=1)
    y = np.fft.ifft(X * H[np.newaxis, :], axis=1)

    # Align peak at exact delay index (compensate for n_ref - 1 convolution shift)
    start = n_ref - 1
    out = y[:, start : start + n_fast]
    return out[0] if is_1d else out


def range_doppler_map(
    compressed_cube: np.ndarray,
    radar_cfg: RadarConfig,
    doppler_window: str = "hann",
    n_doppler_fft: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Range-Doppler power map (in dB) with physical range (m) and velocity (m/s) axes.

    Returns:
        (rd_db, range_axis_m, vel_axis_mps)
        rd_db has shape (n_doppler_fft, n_fast).
    """
    n_pulses, n_fast = compressed_cube.shape
    n_d = n_doppler_fft if n_doppler_fft is not None else max(n_pulses, 128)

    if doppler_window:
        win = get_window(doppler_window, n_pulses, fftbins=True)
        win = win / np.sqrt(np.mean(win ** 2))
        windowed = compressed_cube * win[:, np.newaxis]
    else:
        windowed = compressed_cube

    rd_complex = np.fft.fftshift(np.fft.fft(windowed, n=n_d, axis=0), axes=0)
    rd_power = np.abs(rd_complex) ** 2
    rd_db = 10.0 * np.log10(rd_power + 1e-18)

    range_axis_m = np.arange(n_fast, dtype=np.float64) * (C_LIGHT / (2.0 * radar_cfg.fs_hz))
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_d, d=radar_cfg.pri_s))
    # fd = -2 * v / lambda => v = -fd * lambda / 2
    vel_axis_mps = -doppler_freqs * radar_cfg.wavelength_m / 2.0

    return rd_db, range_axis_m, vel_axis_mps


def ca_cfar_2d(
    rd_power: np.ndarray,
    guard: Tuple[int, int] = (2, 4),
    train: Tuple[int, int] = (4, 8),
    pfa: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D Cell-Averaging CFAR detector on linear power Range-Doppler map."""
    gd, gr = guard
    td, tr = train
    hd, hr = gd + td, gr + tr

    # Build 2D training-cell kernel (1 in training ring, 0 in guard + CUT)
    k_d = 2 * hd + 1
    k_r = 2 * hr + 1
    kernel = np.ones((k_d, k_r), dtype=np.float64)
    kernel[td : td + 2 * gd + 1, tr : tr + 2 * gr + 1] = 0.0
    n_train = float(np.sum(kernel))

    # Exact CA-CFAR threshold factor for exponential (square-law) noise
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    # Fast 2D convolution via FFT for speed
    from scipy.signal import fftconvolve

    noise_sum = fftconvolve(rd_power, kernel, mode="same")
    noise_est = np.maximum(noise_sum / n_train, 1e-18)
    threshold = alpha * noise_est
    detections = rd_power > threshold

    # Suppress border artifacts
    detections[:hd, :] = False
    detections[-hd:, :] = False
    detections[:, :hr] = False
    detections[:, -hr:] = False

    return detections, threshold


def os_cfar_1d(
    power_profile: np.ndarray,
    guard: int = 4,
    train: int = 16,
    rank_ratio: float = 0.75,
    alpha: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """1D Ordered-Statistic (OS) CFAR — robust against closely spaced false targets."""
    n = len(power_profile)
    half = guard + train
    n_cells = 2 * train
    k_rank = min(max(int(round(rank_ratio * n_cells)) - 1, 0), n_cells - 1)

    threshold = np.full(n, np.inf, dtype=np.float64)
    detections = np.zeros(n, dtype=bool)

    for i in range(half, n - half):
        left = power_profile[i - half : i - guard]
        right = power_profile[i + guard + 1 : i + half + 1]
        window = np.concatenate((left, right))
        stat = np.partition(window, k_rank)[k_rank]
        threshold[i] = alpha * stat
        detections[i] = power_profile[i] > threshold[i]

    return detections, threshold
