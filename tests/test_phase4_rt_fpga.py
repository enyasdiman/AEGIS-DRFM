"""Phase 4 Verification Suite: Bit-True Q1.15 Fixed-Point DRFM Golden Model & FPGA Vector Export."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "rt-cpp-engine" / "python"))

from aegis_core import generate_lfm_chirp, matched_filter
from aegis_drfm import measure_sqnr_sfdr
from aegis_rt_bridge import DRFMCoreQ15Golden, export_verilog_hex_vectors, float_to_q15, q15_to_float


class TestPhase4FixedPointAndFPGAEngine(unittest.TestCase):
    def test_q15_delay_line_and_nco_doppler_accuracy(self) -> None:
        fs = 50e6
        n = 4096
        delay_samp = 120
        fd_hz = 1.25e6

        # 1) Test BRAM circular delay line with LFM chirp
        chirp = 0.9 * generate_lfm_chirp(fs, 4e-6, 10e6)
        padded = np.zeros(1024, dtype=np.complex128)
        padded[: len(chirp)] = chirp
        ii, qq = float_to_q15(padded)

        core = DRFMCoreQ15Golden(max_delay_pow2=4096, adc_bits=12)
        core.set_delay_samples(delay_samp)
        core.set_doppler_hz(0.0, fs)
        oi, oq = core.process_block_q15(ii, qq)
        y = q15_to_float(oi, oq)

        comp = matched_filter(y, chirp)
        pk_idx = int(np.argmax(np.abs(comp)))
        self.assertEqual(pk_idx, delay_samp)

        # 2) Test 32-bit NCO Doppler frequency shift & SFDR on DC input
        dc_i, dc_q = float_to_q15(np.full(n, 0.95 + 0j))
        core_nco = DRFMCoreQ15Golden(max_delay_pow2=4096, adc_bits=12)
        core_nco.set_delay_samples(0)
        core_nco.set_doppler_hz(fd_hz, fs)
        ni, nq = core_nco.process_block_q15(dc_i, dc_q)
        y_nco = q15_to_float(ni, nq)

        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        spec = np.abs(np.fft.fft(y_nco))
        est_fd = float(freqs[int(np.argmax(spec))])
        self.assertAlmostEqual(est_fd, fd_hz, delta=fs / n)

        # Verify 10-bit non-interpolated NCO LUT achieves > 48 dBc worst-case SFDR
        t = np.arange(n) / fs
        ideal_tone = 0.95 * np.exp(1j * 2.0 * np.pi * fd_hz * t)
        metrics = measure_sqnr_sfdr(ideal_tone, y_nco)
        self.assertGreater(metrics["sfdr_dbc"], 48.0)

    def test_verilog_hex_vector_export(self) -> None:
        fs = 50e6
        sig = 0.9 * generate_lfm_chirp(fs, 2e-6, 10e6)
        ii, qq = float_to_q15(sig)
        core = DRFMCoreQ15Golden(max_delay_pow2=1024, adc_bits=8)
        core.set_delay_samples(16)
        core.set_doppler_hz(500e3, fs)
        oi, oq = core.process_block_q15(ii, qq)

        vec_dir = REPO_ROOT / "packages" / "rt-cpp-engine" / "fpga" / "vectors"
        stim, gold = export_verilog_hex_vectors(vec_dir, ii, qq, oi, oq)
        self.assertTrue(stim.exists())
        self.assertTrue(gold.exists())
        self.assertEqual(len(stim.read_text(encoding="utf-8").splitlines()), len(sig))


if __name__ == "__main__":
    unittest.main()
