"""DRFM ADC/DAC quantization models and spectral purity (SQNR / ENOB / SFDR) analyzers."""

from __future__ import annotations

from typing import Dict

import numpy as np


def quantize_complex_drfm(
    signal: np.ndarray,
    bits: int = 8,
    full_scale: float = 1.0,
) -> np.ndarray:
    """Quantize a complex baseband IQ waveform using an N-bit DRFM ADC/DAC model.

    Bit-matched to matlab/drfm_models/drfm_quantization_sfdr.m:
      - bits == 1: Hard-limiter / phase-only 1-bit DRFM (quadrature sign quantization).
      - bits >= 2: Uniform mid-tread I/Q quantizer with clipping at [-full_scale, full_scale - step/2].
    """
    if bits < 1:
        raise ValueError("bits must be >= 1.")
    if full_scale <= 0.0:
        raise ValueError("full_scale must be > 0.")

    x = np.asarray(signal, dtype=np.complex128) / full_scale

    if bits == 1:
        qi = np.where(np.real(x) >= 0.0, 1.0, -1.0)
        qq = np.where(np.imag(x) >= 0.0, 1.0, -1.0)
        return ((qi + 1j * qq) / np.sqrt(2.0)) * full_scale

    levels = 2 ** bits
    step = 2.0 / levels
    ri = np.clip(np.real(x), -1.0, 1.0 - step / 2.0)
    rq = np.clip(np.imag(x), -1.0, 1.0 - step / 2.0)
    qi = np.round(ri / step) * step
    qq = np.round(rq / step) * step
    return (qi + 1j * qq) * full_scale


def measure_sqnr_sfdr(
    signal_clean: np.ndarray,
    signal_quantized: np.ndarray,
    mask_half_width_bins: int = 4,
) -> Dict[str, float]:
    """Measure SQNR (dB), ENOB (bits), and SFDR (dBc) on a complex test tone."""
    x = np.asarray(signal_clean, dtype=np.complex128)
    xq = np.asarray(signal_quantized, dtype=np.complex128)
    n = len(x)

    # Optimal linear amplitude match for fair quantization error calculation
    denom = np.vdot(xq, xq)
    scale = np.vdot(xq, x) / denom if np.abs(denom) > 0 else 1.0
    xq_norm = xq * scale

    p_sig = float(np.mean(np.abs(x) ** 2))
    p_err = float(np.mean(np.abs(xq_norm - x) ** 2))
    sqnr_db = 10.0 * np.log10(p_sig / max(p_err, 1e-18))
    enob = (sqnr_db - 1.76) / 6.02

    # Hann window matches MATLAB hann(N) (symmetric)
    w = np.hanning(n)
    w = w / np.sum(w)
    spec = np.fft.fftshift(np.fft.fft(xq * w))
    mag_db = 20.0 * np.log10(np.abs(spec) + 1e-15)
    mag_db = mag_db - float(np.max(mag_db))

    pk_idx = int(np.argmax(mag_db))
    spur_mask = np.ones(n, dtype=bool)
    lo = max(0, pk_idx - mask_half_width_bins)
    hi = min(n, pk_idx + mask_half_width_bins + 1)
    spur_mask[lo:hi] = False
    sfdr_dbc = float(-np.max(mag_db[spur_mask]))

    return {
        "sqnr_db": float(sqnr_db),
        "sfdr_dbc": sfdr_dbc,
        "enob": float(enob),
    }
