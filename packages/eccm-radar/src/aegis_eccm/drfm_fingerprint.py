"""DRFM Micro-Signature Fingerprinting (Quantization Spurs, Phase Jitter) & ROC/AUC Evaluation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def extract_pulse_fingerprint(
    rx_pulse: np.ndarray,
    ref_pulse: np.ndarray,
) -> np.ndarray:
    """Extract a 4D physical micro-signature vector from a de-chirped/demodulated pulse.

    De-chirping (rx_pulse * conj(ref_pulse)) collapses a clean echo into a pure DC/narrowband
    tone, exposing DRFM quantization harmonics, elevated spur floor, amplitude distortion,
    and residual LO phase jitter.

    Features:
      0. inverse_sfdr_db: Strongest out-of-mainlobe spur relative to peak (dBc, higher = more spurious)
      1. spectral_flatness_db: Ratio of geometric to arithmetic mean of out-of-band spectrum (dB)
      2. env_cv: Envelope coefficient of variation std(|y|)/mean(|y|)
      3. phase_diff_std: Std of sample-to-sample phase difference (rad)
    """
    n = min(len(rx_pulse), len(ref_pulse))
    y = rx_pulse[:n] * np.conj(ref_pulse[:n])
    # Normalize RMS power to 1.0
    rms = float(np.sqrt(np.mean(np.abs(y) ** 2))) + 1e-12
    y = y / rms

    # Spectral spur analysis
    w = np.hanning(n)
    w = w / np.sum(w)
    spec = np.fft.fftshift(np.fft.fft(y * w))
    pow_spec = np.abs(spec) ** 2 + 1e-18
    mag_db = 10.0 * np.log10(pow_spec)
    pk_idx = int(np.argmax(mag_db))
    pk_db = float(mag_db[pk_idx])

    mask = np.ones(n, dtype=bool)
    lo = max(0, pk_idx - 4)
    hi = min(n, pk_idx + 5)
    mask[lo:hi] = False

    oob_db = mag_db[mask] - pk_db
    inv_sfdr_db = float(np.max(oob_db))  # e.g. -9.5 dBc for 1-bit DRFM, -45 dBc for clean

    oob_pow = pow_spec[mask]
    geo_mean = float(np.exp(np.mean(np.log(oob_pow))))
    arith_mean = float(np.mean(oob_pow))
    spectral_flatness_db = 10.0 * np.log10(geo_mean / max(arith_mean, 1e-18))

    # Envelope and phase statistics
    amp = np.abs(y)
    env_cv = float(np.std(amp) / (np.mean(amp) + 1e-12))

    dphi = np.angle(y[1:] * np.conj(y[:-1]))
    phase_diff_std = float(np.std(dphi))

    return np.array([inv_sfdr_db, spectral_flatness_db, env_cv, phase_diff_std], dtype=np.float64)


class DRFMFingerprintDetector:
    """Standardized Fisher Linear Discriminant classifier for Real Echo (0) vs. DRFM Replay (1)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DRFMFingerprintDetector":
        """Fit standardized Fisher LDA on feature matrix X (N, D) and binary labels y in {0, 1}."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-9
        Xn = (X - self.mean_) / self.std_

        X0 = Xn[y == 0]
        X1 = Xn[y == 1]
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)

        # Regularized within-class scatter
        cov0 = np.cov(X0, rowvar=False) if len(X0) > 1 else np.eye(X.shape[1])
        cov1 = np.cov(X1, rowvar=False) if len(X1) > 1 else np.eye(X.shape[1])
        sw = cov0 + cov1 + 1e-3 * np.eye(X.shape[1])

        w = np.linalg.solve(sw, mu1 - mu0)
        w = w / (np.linalg.norm(w) + 1e-12)
        self.w_ = w
        self.b_ = -0.5 * float(np.dot(w, mu0 + mu1))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return continuous log-odds-like DRFM score (higher = more likely DRFM)."""
        assert self.mean_ is not None and self.std_ is not None and self.w_ is not None
        Xn = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        return Xn @ self.w_ + self.b_

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.decision_function(X) >= threshold).astype(np.int32)


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute exact ROC curve (Pfa, Pd) and Area Under Curve (AUC) without external ML libs."""
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    tps = np.cumsum(y_sorted == 1) / n_pos
    fps = np.cumsum(y_sorted == 0) / n_neg

    pfa = np.concatenate(([0.0], fps, [1.0]))
    pd = np.concatenate(([0.0], tps, [1.0]))
    auc = float(np.trapezoid(pd, pfa)) if hasattr(np, "trapezoid") else float(np.trapz(pd, pfa))
    return pfa, pd, auc
