"""Radar waveform generators: LFM Chirp, Barker BPSK, Frank Polyphase, and PRI schedules."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_BARKER_CODES = {
    2: np.array([1, -1], dtype=np.float64),
    3: np.array([1, 1, -1], dtype=np.float64),
    4: np.array([1, 1, -1, 1], dtype=np.float64),
    5: np.array([1, 1, 1, -1, 1], dtype=np.float64),
    7: np.array([1, 1, 1, -1, -1, 1, -1], dtype=np.float64),
    11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=np.float64),
    13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64),
}


def generate_lfm_chirp(
    fs_hz: float,
    pulse_width_s: float,
    bandwidth_hz: float,
    up: bool = True,
) -> np.ndarray:
    """Generate unit-amplitude complex baseband Linear Frequency Modulated (LFM) pulse."""
    n_samples = max(2, int(round(pulse_width_s * fs_hz)))
    t = np.arange(n_samples, dtype=np.float64) / fs_hz - (pulse_width_s / 2.0)
    k = (bandwidth_hz / pulse_width_s) * (1.0 if up else -1.0)
    phase = np.pi * k * (t ** 2)
    return np.exp(1j * phase).astype(np.complex128)


def generate_barker_code(
    fs_hz: float,
    chip_width_s: float,
    length: int = 13,
) -> np.ndarray:
    """Generate baseband BPSK Barker phase-coded pulse."""
    if length not in _BARKER_CODES:
        raise ValueError(f"Unsupported Barker length {length}. Valid: {sorted(_BARKER_CODES)}")
    code = _BARKER_CODES[length]
    samples_per_chip = max(1, int(round(chip_width_s * fs_hz)))
    pulse = np.repeat(code, samples_per_chip).astype(np.complex128)
    return pulse


def generate_frank_code(
    fs_hz: float,
    chip_width_s: float,
    m: int = 4,
) -> np.ndarray:
    """Generate baseband Frank polyphase code of length m^2."""
    if m < 2:
        raise ValueError("Frank code order m must be >= 2.")
    idx = np.arange(m)
    phases = (2.0 * np.pi / m) * np.outer(idx, idx).ravel()
    code = np.exp(1j * phases)
    samples_per_chip = max(1, int(round(chip_width_s * fs_hz)))
    return np.repeat(code, samples_per_chip).astype(np.complex128)


def generate_pri_schedule(
    num_pulses: int,
    prf_hz: float,
    mode: str = "constant",
    stagger_ratios: Optional[Sequence[float]] = None,
    jitter_std_s: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return pulse transmission start times (TOA array, length num_pulses) in seconds."""
    base_pri = 1.0 / prf_hz
    if mode == "constant":
        pris = np.full(num_pulses, base_pri, dtype=np.float64)
    elif mode == "staggered":
        ratios = np.asarray(stagger_ratios if stagger_ratios is not None else [0.9, 1.0, 1.1], dtype=np.float64)
        ratios = ratios / np.mean(ratios)
        pris = np.array([base_pri * ratios[i % len(ratios)] for i in range(num_pulses)], dtype=np.float64)
    elif mode == "jittered":
        gen = rng if rng is not None else np.random.default_rng()
        pris = np.clip(base_pri + gen.normal(0.0, jitter_std_s, size=num_pulses), 0.5 * base_pri, 1.5 * base_pri)
    else:
        raise ValueError(f"Unknown PRI mode '{mode}'. Use 'constant', 'staggered', or 'jittered'.")

    toas = np.zeros(num_pulses, dtype=np.float64)
    if num_pulses > 1:
        toas[1:] = np.cumsum(pris[:-1])
    return toas
