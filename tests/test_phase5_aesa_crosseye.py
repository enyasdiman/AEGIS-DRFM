"""Phase 5 Verification Suite: AESA Monopulse Angle Accuracy, Cross-Eye Pull-Off & Quadrature ECCM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "aesa-monopulse" / "src"))

from aegis_aesa import (
    AESAMonopulseArray,
    CrossEyeConfig,
    compute_crosseye_gain,
    detect_crosseye_eccm,
    simulate_crosseye_vs_monopulse,
)


class TestPhase5AESAMonopulseAndCrossEye(unittest.TestCase):
    def test_clean_monopulse_angle_estimation_and_zero_quadrature(self) -> None:
        arr = AESAMonopulseArray(num_elements=16, fc_hz=10e9, boresight_deg=0.0)
        for true_ang in (-1.5, -0.5, 0.0, 0.75, 1.5):
            snap = arr.steering_vector(true_ang)
            meas = arr.measure_snapshot(snap)
            self.assertAlmostEqual(meas.est_angle_deg, true_ang, delta=0.08)
            self.assertAlmostEqual(abs(meas.ratio_imag), 0.0, places=6)

    def test_crosseye_pulls_angle_outside_wingspan_and_eccm_detects(self) -> None:
        arr = AESAMonopulseArray(num_elements=16, fc_hz=10e9, boresight_deg=0.0)
        # Analytical Cross-Eye gain at a=0.9, phi=180 deg == (1 - (-0.9)) / (1 + (-0.9)) = 19.0
        gc_r, gc_i = compute_crosseye_gain(0.90, 180.0)
        self.assertAlmostEqual(gc_r, 19.0, places=5)
        self.assertAlmostEqual(gc_i, 0.0, places=5)

        # Realistic Cross-Eye at phi=178 deg: pulls angle > 10x wingspan & triggers quadrature ECCM
        cfg = CrossEyeConfig(baseline_m=15.0, amplitude_ratio=0.90, phase_deg=178.0, target_range_m=3000.0, jsr_db=25.0)
        res = simulate_crosseye_vs_monopulse(arr, cfg, include_skin_echo=True, snr_db=None)
        pull_ratio = abs(res["angle_error_deg"]) / res["theta_e_deg"]
        self.assertGreater(pull_ratio, 10.0)

        eccm = detect_crosseye_eccm(res["measurement"], quadrature_thresh=0.05)
        self.assertTrue(eccm["crosseye_detected"])

    def test_matlab_aesa_crosseye_cross_validation(self) -> None:
        import json

        ref_path = REPO_ROOT / "matlab" / "outputs" / "aesa_crosseye_ref.json"
        if not ref_path.exists():
            self.skipTest("Run matlab/aesa_crosseye/aesa_monopulse_crosseye_design.m first.")

        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        arr = AESAMonopulseArray(num_elements=int(ref["num_elements"]), fc_hz=float(ref["fc_hz"]), boresight_deg=0.0)
        cfg = CrossEyeConfig(baseline_m=15.0, amplitude_ratio=0.90, phase_deg=178.0, target_range_m=3000.0, jsr_db=25.0)
        res = simulate_crosseye_vs_monopulse(arr, cfg, include_skin_echo=True, snr_db=None)

        self.assertAlmostEqual(arr.k_monopulse, ref["k_monopulse"], places=5)
        self.assertAlmostEqual(res["theta_e_deg"], ref["theta_e_deg"], places=5)
        self.assertAlmostEqual(res["measurement"].est_angle_deg, ref["est_angle_deg"], places=4)
        self.assertAlmostEqual(abs(res["measurement"].ratio_imag), ref["quad_ratio"], places=4)


if __name__ == "__main__":
    unittest.main()
