"""Phase 2 Verification Suite: SIGINT/ELINT Pulse Detection, MOP, SDIF Deinterleaving & Cognitive EW."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "sigint-esm" / "src"))

from aegis_core import add_awgn, generate_barker_code, generate_lfm_chirp
from aegis_sigint import (
    CognitiveEWController,
    classify_intra_pulse_mop,
    deinterleave_sdif,
    detect_pulses_iq,
)


class TestPhase2SigintAndCognitiveEW(unittest.TestCase):
    def test_intra_pulse_mop_classification(self) -> None:
        fs = 50e6
        rng = np.random.default_rng(1)

        p_up = add_awgn(generate_lfm_chirp(fs, 10e-6, 12e6, up=True), snr_db=20.0, rng=rng)
        p_dn = add_awgn(generate_lfm_chirp(fs, 10e-6, 12e6, up=False), snr_db=20.0, rng=rng)
        p_bk = add_awgn(generate_barker_code(fs, 0.5e-6, length=13), snr_db=20.0, rng=rng)
        p_cw = add_awgn(np.ones(200, dtype=np.complex128), snr_db=20.0, rng=rng)

        self.assertEqual(classify_intra_pulse_mop(p_up, fs)[0], "LFM_UP")
        self.assertEqual(classify_intra_pulse_mop(p_dn, fs)[0], "LFM_DOWN")
        self.assertEqual(classify_intra_pulse_mop(p_bk, fs)[0], "BPSK_BARKER")
        self.assertEqual(classify_intra_pulse_mop(p_cw, fs)[0], "CW")

    def test_multi_emitter_detection_deinterleaving_and_cognitive_ew(self) -> None:
        fs = 50e6
        duration_s = 2.0e-3
        rf_center = 10.0e9
        n_total = int(round(duration_s * fs))
        iq = np.zeros(n_total, dtype=np.complex128)
        rng = np.random.default_rng(99)

        def inject(pulse: np.ndarray, f_off: float, pri_s: float, t0: float, amp: float) -> None:
            t_loc = np.arange(len(pulse)) / fs
            sig = amp * pulse * np.exp(1j * 2.0 * np.pi * f_off * t_loc)
            t_c = t0
            while t_c + len(pulse) / fs < duration_s:
                idx = int(round(t_c * fs))
                iq[idx : idx + len(pulse)] += sig
                t_c += pri_s

        # 3 distinct emitters matching default ThreatLibrary
        inject(generate_lfm_chirp(fs, 10.0e-6, 12.0e6, up=True), 0.0, 200e-6, 20e-6, 1.0)
        inject(generate_barker_code(fs, 0.5e-6, length=13), +14.0e6, 130e-6, 55e-6, 0.9)
        inject(np.ones(int(4.0e-6 * fs), dtype=np.complex128), -15.0e6, 340e-6, 100e-6, 0.8)

        iq_noisy = add_awgn(iq, snr_db=16.0, signal_power=1.0, rng=rng)
        pdws = detect_pulses_iq(iq_noisy, fs_hz=fs, rf_center_hz=rf_center, threshold_db_above_noise=10.0)
        self.assertGreaterEqual(len(pdws), 25)

        tracks, _ = deinterleave_sdif(pdws)
        self.assertEqual(len(tracks), 3)

        # Verify PRI accuracy of all 3 recovered emitters
        recovered_pris_us = sorted([t.pri_s * 1e6 for t in tracks])
        self.assertAlmostEqual(recovered_pris_us[0], 130.0, delta=3.0)
        self.assertAlmostEqual(recovered_pris_us[1], 200.0, delta=3.0)
        self.assertAlmostEqual(recovered_pris_us[2], 340.0, delta=4.0)

        # Verify Cognitive EW ranks Fire-Control Tracker as Priority 1 and assigns RGPO_VGPO
        ctrl = CognitiveEWController()
        primary = ctrl.select_primary_response(tracks)
        self.assertIsNotNone(primary)
        self.assertIsNotNone(primary.signature)
        self.assertEqual(primary.signature.role, "FIRE_CONTROL")
        self.assertEqual(primary.ecm_command.mode, "RGPO_VGPO")
        self.assertGreaterEqual(primary.confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
