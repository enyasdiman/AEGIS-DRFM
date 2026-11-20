"""Phase 9 Verification Suite: Bit-Exact C++20 DLL <-> Python Bridge & SAR DRFM False-Scene Deception."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "rt-cpp-engine" / "python"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "sar-isar" / "src"))

from aegis_core import generate_lfm_chirp
from aegis_rt_bridge import DRFMCoreQ15Golden, DRFMCoreQ15Native, float_to_q15
from aegis_sar import PointScatterer, SARConfig, sar_backprojection_image, simulate_sar_raw


class TestPhase9NativeCppBridgeAndSAR(unittest.TestCase):
    def test_cpp20_dll_bit_exact_parity_with_python_golden(self) -> None:
        fs = 50e6
        n = 16384
        t = np.arange(n, dtype=np.float64) / fs
        sig = 0.85 * np.exp(1j * 2.0 * np.pi * 1.37e6 * t)
        in_i, in_q = float_to_q15(sig)

        py_core = DRFMCoreQ15Golden(max_delay_pow2=4096, adc_bits=8)
        py_core.set_delay_samples(137)
        py_core.set_doppler_hz(3250.0, fs)

        cpp_core = DRFMCoreQ15Native(max_delay_pow2=4096, adc_bits=8)
        cpp_core.set_delay_samples(137)
        cpp_core.set_doppler_hz(3250.0, fs)

        t0 = time.perf_counter()
        py_i, py_q = py_core.process_block_q15(in_i, in_q)
        t_py = time.perf_counter() - t0

        t1 = time.perf_counter()
        cpp_i, cpp_q = cpp_core.process_block_q15(in_i, in_q)
        t_cpp = time.perf_counter() - t1
        cpp_core.close()

        # Bit-exact 0-LSB match across all 16,384 complex IQ16 samples!
        self.assertTrue(np.array_equal(py_i, cpp_i), "I channel mismatch between C++20 DLL and Python Golden!")
        self.assertTrue(np.array_equal(py_q, cpp_q), "Q channel mismatch between C++20 DLL and Python Golden!")
        self.assertLess(t_cpp, t_py, "Expected C++20 DLL to be faster than Python loop!")

    def test_sar_backprojection_and_drfm_false_scene_injection(self) -> None:
        cfg = SARConfig(num_pulses=48, aperture_m=90.0)
        real_pt = [PointScatterer(az_m=-10.0, rng_m=-6.0, reflectivity=1.0)]
        decoy_pt = [PointScatterer(az_m=+12.0, rng_m=+8.0, reflectivity=1.0)]

        az_grid = np.linspace(-20.0, 20.0, 81)
        rng_grid = np.linspace(-16.0, 16.0, 65)
        rng = np.random.default_rng(11)

        raw, tx, r0_s = simulate_sar_raw(
            real_pt, cfg, drfm_decoy_scatterers=decoy_pt, drfm_jsr_db=6.0, drfm_adc_bits=8, snr_db=20.0, rng=rng
        )
        img = sar_backprojection_image(raw, tx, cfg, r0_s, az_grid, rng_grid)

        # Peak of SAR image should land right on the stronger (+6 dB JSR) DRFM decoy at (+12 m, +8 m)!
        pk_r, pk_a = np.unravel_index(int(np.argmax(img)), img.shape)
        self.assertAlmostEqual(float(az_grid[pk_a]), +12.0, delta=1.5)
        self.assertAlmostEqual(float(rng_grid[pk_r]), +8.0, delta=1.5)

        # Real target at (-10 m, -6 m) should also remain visible above background (-15 dB)
        idx_a = int(np.argmin(np.abs(az_grid - (-10.0))))
        idx_r = int(np.argmin(np.abs(rng_grid - (-6.0))))
        self.assertGreater(float(img[idx_r, idx_a]), -12.0)


if __name__ == "__main__":
    unittest.main()
