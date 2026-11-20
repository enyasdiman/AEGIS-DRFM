"""Phase 3 Verification Suite: ECCM Waveform Agility, DRFM Fingerprinting ROC, and Kalman Anti-RGPO."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "eccm-radar" / "src"))

from aegis_core import RadarConfig, TargetState, add_awgn, generate_lfm_chirp
from aegis_drfm import DRFMConfig, DRFMCore
from aegis_eccm import (
    DRFMFingerprintDetector,
    KinematicECCMTracker,
    MeasurementRD,
    compute_roc_auc,
    evaluate_agility_suppression,
    extract_pulse_fingerprint,
)


class TestPhase3ECCMDefense(unittest.TestCase):
    def test_waveform_agility_suppresses_repeat_drfm(self) -> None:
        cfg = RadarConfig(num_pulses=16, prf_hz=5000.0, fs_hz=50e6, bandwidth_hz=20e6, pulse_width_s=10e-6)
        tgt = TargetState(range_m=8000.0, velocity_mps=-20.0)

        fixed = evaluate_agility_suppression(cfg, tgt, false_range_offset_m=400.0, jsr_db=10.0, mode="fixed", seed=1)
        agile = evaluate_agility_suppression(cfg, tgt, false_range_offset_m=400.0, jsr_db=10.0, mode="up_down_lfm", seed=1)

        suppression_gain_db = float(agile["sjr_out_db"]) - float(fixed["sjr_out_db"])
        self.assertGreater(suppression_gain_db, 12.0)

    def test_drfm_fingerprint_detector_high_auc_low_bits(self) -> None:
        rng = np.random.default_rng(7)
        ref = generate_lfm_chirp(50e6, 10e-6, 20e6)

        for bits in (1, 2, 4):
            drfm = DRFMCore(DRFMConfig(adc_bits=bits, phase_noise_std_rad=0.015), rng=rng)
            X, y = [], []
            for _ in range(80):
                X.append(extract_pulse_fingerprint(add_awgn(ref, 18.0, 1.0, rng=rng), ref))
                y.append(0)
                X.append(extract_pulse_fingerprint(add_awgn(drfm.capture(ref), 18.0, 1.0, rng=rng), ref))
                y.append(1)
            X_arr, y_arr = np.vstack(X), np.array(y)
            clf = DRFMFingerprintDetector().fit(X_arr[:80], y_arr[:80])
            _, _, auc = compute_roc_auc(y_arr[80:], clf.decision_function(X_arr[80:]))
            self.assertGreater(auc, 0.90, f"Expected AUC > 0.90 for {bits}-bit DRFM, got {auc:.3f}")

    def test_kinematic_kalman_rejects_rgpo_walkoff(self) -> None:
        rng = np.random.default_rng(13)
        dt = 0.05
        r0, v_true = 10000.0, -150.0
        tracker = KinematicECCMTracker(r0, v_true, dt_s=dt, sigma_r_m=6.0, sigma_v_mps=1.5)

        errors_est = []
        rgpo_alerts = 0
        for k in range(60):
            t = k * dt
            r_true = r0 + v_true * t
            pull = 300.0 * ((t - 1.0) ** 2) if 1.0 <= t <= 2.5 else 0.0
            res = tracker.step(
                MeasurementRD(
                    t_s=t,
                    range_m=r_true + pull + float(rng.normal(0.0, 4.0)),
                    velocity_mps=v_true + float(rng.normal(0.0, 1.0)),
                )
            )
            errors_est.append(res.est_range_m - r_true)
            if res.ecm_alert in ("RGPO_REJECTED", "GATE_REJECTED"):
                rgpo_alerts += 1

        rms_err = float(np.sqrt(np.mean(np.square(errors_est))))
        self.assertGreaterEqual(rgpo_alerts, 10)
        self.assertLess(rms_err, 25.0)


if __name__ == "__main__":
    unittest.main()
