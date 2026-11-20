"""Phase 1 Verification & MATLAB <-> Python Cross-Validation Test Suite."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))

from aegis_core import (
    C_LIGHT,
    RadarConfig,
    RFImpairments,
    TargetState,
    apply_fractional_delay,
    ca_cfar_2d,
    generate_barker_code,
    generate_frank_code,
    generate_lfm_chirp,
    generate_pri_schedule,
    matched_filter,
    os_cfar_1d,
    range_doppler_map,
    simulate_radar_cpi,
)
from aegis_drfm import DRFMConfig, DRFMCore, FalseTargetSpec, measure_sqnr_sfdr, quantize_complex_drfm


class TestPhase1CoreAndMATLABCrossValidation(unittest.TestCase):
    def test_matlab_system_config_cross_validation(self) -> None:
        cfg_path = REPO_ROOT / "matlab" / "outputs" / "system_config.json"
        self.assertTrue(cfg_path.exists(), "Run MATLAB rf_ew_system_design.m first.")

        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = RadarConfig.from_matlab_json(cfg_path)

        self.assertAlmostEqual(cfg.range_resolution_m, raw["derived"]["range_res_m"], places=5)
        self.assertAlmostEqual(cfg.velocity_resolution_mps, raw["derived"]["vel_res_mps"], places=5)
        self.assertAlmostEqual(cfg.unambiguous_range_m, raw["derived"]["unambiguous_rng_m"], places=3)

    def test_matlab_drfm_quantization_cross_validation(self) -> None:
        ref_path = REPO_ROOT / "matlab" / "outputs" / "drfm_quantization_ref.json"
        self.assertTrue(ref_path.exists(), "Run MATLAB drfm_quantization_sfdr.m first.")

        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        fs = float(ref["fs_hz"])
        n = int(ref["n_fft"])
        f0 = float(ref["tone_hz"])
        t = np.arange(n, dtype=np.float64) / fs
        x = 0.98 * np.exp(1j * 2.0 * np.pi * f0 * t)

        for idx, bits in enumerate(ref["bit_depths"]):
            xq = quantize_complex_drfm(x, bits=int(bits))
            metrics = measure_sqnr_sfdr(x, xq)
            self.assertAlmostEqual(metrics["sqnr_db"], ref["sqnr_meas_db"][idx], delta=0.15)
            self.assertAlmostEqual(metrics["sfdr_dbc"], ref["sfdr_meas_db"][idx], delta=0.25)
            self.assertAlmostEqual(metrics["enob"], ref["enob_meas"][idx], delta=0.05)

    def test_waveforms_and_matched_filter_gain(self) -> None:
        # LFM Pulse Compression SNR gain ~= BT = 20 MHz * 10 us = 200 (23.01 dB)
        fs = 50e6
        pw = 10e-6
        bw = 20e6
        chirp = generate_lfm_chirp(fs, pw, bw, up=True)
        out = matched_filter(chirp, chirp, normalize=True)
        # Peak power should equal energy of unit-amplitude chirp = len(chirp) = 500
        self.assertAlmostEqual(float(np.abs(out[0]) ** 2), float(len(chirp)), delta=1.0)

        # Barker-13 peak-to-sidelobe ratio == 13:1 (22.28 dB)
        b13 = generate_barker_code(fs_hz=1e6, chip_width_s=1e-6, length=13)
        acf = np.abs(np.correlate(b13, b13, mode="full"))
        peak = float(np.max(acf))
        sidelobes = np.delete(acf, int(np.argmax(acf)))
        self.assertAlmostEqual(peak / float(np.max(sidelobes)), 13.0, places=5)

        # Frank code length = m^2
        frank = generate_frank_code(fs_hz=1e6, chip_width_s=1e-6, m=4)
        self.assertEqual(len(frank), 16)

        # PRI schedules
        toas_stagger = generate_pri_schedule(6, 5000.0, mode="staggered", stagger_ratios=[0.8, 1.0, 1.2])
        self.assertEqual(len(toas_stagger), 6)
        self.assertTrue(np.all(np.diff(toas_stagger) > 0))

    def test_fractional_delay_precision(self) -> None:
        fs = 50e6
        chirp = generate_lfm_chirp(fs, 4e-6, 10e6)
        delay_samp = 123.35
        delayed = apply_fractional_delay(chirp, delay_samp, out_len=1024)
        comp = matched_filter(delayed, chirp)
        pk_idx = int(np.argmax(np.abs(comp)))
        self.assertAlmostEqual(pk_idx, round(delay_samp), delta=1)

    def test_drfm_rgpo_and_vgpo_physical_accuracy(self) -> None:
        cfg = RadarConfig(num_pulses=32, prf_hz=5000.0, fs_hz=50e6, bandwidth_hz=20e6, pulse_width_s=5e-6)
        tx = generate_lfm_chirp(cfg.fs_hz, cfg.pulse_width_s, cfg.bandwidth_hz)
        tgt = TargetState(range_m=6000.0, velocity_mps=-20.0, rcs_m2=5.0)
        max_r = 8500.0
        n_fast = int(np.ceil((2.0 * max_r / C_LIGHT) * cfg.fs_hz)) + len(tx) + 64
        rng = np.random.default_rng(7)
        imp = RFImpairments(snr_single_pulse_db=10.0)

        # 1) RGPO +500 m
        drfm_rgpo = DRFMCore(DRFMConfig(adc_bits=8, loop_delay_ns=0.0, jsr_db=12.0, phase_noise_std_rad=0.0), rng=rng)
        rx_rgpo, _ = simulate_radar_cpi(
            tx, cfg, [tgt], imp, drfm_callback=drfm_rgpo.make_cpi_callback(cfg, "RGPO", n_fast, range_offset_m=500.0),
            max_range_m=max_r, rng=rng,
        )
        rd_db, r_ax, v_ax = range_doppler_map(matched_filter(rx_rgpo, tx), cfg, n_doppler_fft=128)
        pk_d, pk_r = np.unravel_index(int(np.argmax(rd_db)), rd_db.shape)
        self.assertAlmostEqual(r_ax[pk_r], 6500.0, delta=15.0)
        self.assertAlmostEqual(v_ax[pk_d], -20.0, delta=3.0)

        # 2) VGPO +25 m/s (from -20 m/s -> +5 m/s)
        drfm_vgpo = DRFMCore(DRFMConfig(adc_bits=8, loop_delay_ns=0.0, jsr_db=12.0, phase_noise_std_rad=0.0), rng=rng)
        rx_vgpo, _ = simulate_radar_cpi(
            tx, cfg, [tgt], imp, drfm_callback=drfm_vgpo.make_cpi_callback(cfg, "VGPO", n_fast, vel_offset_mps=25.0),
            max_range_m=max_r, rng=rng,
        )
        rd_db_v, r_ax_v, v_ax_v = range_doppler_map(matched_filter(rx_vgpo, tx), cfg, n_doppler_fft=128)
        pk_dv, pk_rv = np.unravel_index(int(np.argmax(rd_db_v)), rd_db_v.shape)
        self.assertAlmostEqual(r_ax_v[pk_rv], 6000.0, delta=15.0)
        self.assertAlmostEqual(v_ax_v[pk_dv], 5.0, delta=3.0)

        # 3) CA-CFAR & OS-CFAR detect the peak
        dets_2d, _ = ca_cfar_2d(10.0 ** (rd_db / 10.0), guard=(2, 4), train=(4, 8), pfa=1e-4)
        self.assertTrue(bool(dets_2d[pk_d, pk_r]))
        dets_1d, _ = os_cfar_1d(10.0 ** (rd_db[pk_d, :] / 10.0), guard=4, train=16, alpha=10.0)
        self.assertTrue(bool(dets_1d[pk_r]))

    def test_drfm_false_targets_and_isrj(self) -> None:
        cfg = RadarConfig(num_pulses=16, prf_hz=5000.0, fs_hz=50e6, bandwidth_hz=20e6, pulse_width_s=10e-6)
        tx = generate_lfm_chirp(cfg.fs_hz, cfg.pulse_width_s, cfg.bandwidth_hz)
        tgt = TargetState(range_m=5000.0, velocity_mps=0.0)
        max_r = 7500.0
        n_fast = int(np.ceil((2.0 * max_r / C_LIGHT) * cfg.fs_hz)) + len(tx) + 64
        rng = np.random.default_rng(11)
        imp = RFImpairments(snr_single_pulse_db=12.0)

        drfm = DRFMCore(DRFMConfig(adc_bits=8, loop_delay_ns=100.0, jsr_db=10.0, phase_noise_std_rad=0.0), rng=rng)
        specs = [
            FalseTargetSpec(delta_range_m=400.0, delta_velocity_mps=15.0),
            FalseTargetSpec(delta_range_m=900.0, delta_velocity_mps=-15.0),
        ]
        rx_ft, _ = simulate_radar_cpi(
            tx, cfg, [tgt], imp, drfm_callback=drfm.make_cpi_callback(cfg, "FALSE_TARGETS", n_fast, specs=specs),
            max_range_m=max_r, rng=rng,
        )
        rd_ft, _, _ = range_doppler_map(matched_filter(rx_ft, tx), cfg, n_doppler_fft=64)
        dets, _ = ca_cfar_2d(10.0 ** (rd_ft / 10.0), guard=(2, 4), train=(4, 8), pfa=1e-4)
        self.assertGreaterEqual(int(np.sum(dets)), 3)

        # ISRJ produces significant energy above noise floor
        rx_isrj, _ = simulate_radar_cpi(
            tx, cfg, [tgt], imp,
            drfm_callback=drfm.make_cpi_callback(cfg, "ISRJ", n_fast, slice_period_us=2.0, duty_cycle=0.5, repeat_count=2),
            max_range_m=max_r, rng=rng,
        )
        comp_isrj = matched_filter(rx_isrj, tx)
        self.assertGreater(float(np.max(np.abs(comp_isrj))), 10.0)


if __name__ == "__main__":
    unittest.main()
