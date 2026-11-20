"""AEGIS-DRFM Phase 7: Interactive Web EW Studio Backend (Zero-Dependency Python Server).

Run:
    python apps/web-ew-studio/server.py
Then open http://127.0.0.1:8765 in your browser.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "aesa-monopulse" / "src"))

from aegis_aesa import AESAMonopulseArray, CrossEyeConfig, detect_crosseye_eccm, simulate_crosseye_vs_monopulse
from aegis_core import (
    C_LIGHT,
    RadarConfig,
    RFImpairments,
    TargetState,
    ca_cfar_2d,
    generate_lfm_chirp,
    matched_filter,
    range_doppler_map,
    simulate_radar_cpi,
)
from aegis_drfm import DRFMConfig, DRFMCore, FalseTargetSpec, measure_sqnr_sfdr, quantize_complex_drfm

STATIC_DIR = Path(__file__).resolve().parent


def run_rd_sim(params: dict) -> dict:
    mode = str(params.get("mode", "RGPO")).upper()
    jsr_db = float(params.get("jsr_db", 8.0))
    snr_db = float(params.get("snr_db", 6.0))
    adc_bits = int(params.get("adc_bits", 8))
    rgpo_m = float(params.get("rgpo_m", 600.0))
    vgpo_mps = float(params.get("vgpo_mps", 25.0))

    cfg = RadarConfig(num_pulses=32, prf_hz=5000.0, fs_hz=50e6, bandwidth_hz=20e6, pulse_width_s=10e-6)
    tgt = TargetState(range_m=12000.0, velocity_mps=-25.0, rcs_m2=5.0)
    tx = generate_lfm_chirp(cfg.fs_hz, cfg.pulse_width_s, cfg.bandwidth_hz)

    max_r = 14200.0
    n_fast = int(np.ceil((2.0 * max_r / C_LIGHT) * cfg.fs_hz)) + len(tx) + 64
    rng = np.random.default_rng(42)
    imp = RFImpairments(snr_single_pulse_db=snr_db, phase_noise_std_rad=0.002)

    drfm = DRFMCore(DRFMConfig(adc_bits=adc_bits, loop_delay_ns=120.0, jsr_db=jsr_db), rng=rng)
    kwargs = {}
    if mode == "RGPO":
        kwargs = {"range_offset_m": rgpo_m}
    elif mode == "VGPO":
        kwargs = {"vel_offset_mps": vgpo_mps}
    elif mode == "RGPO_VGPO":
        kwargs = {"range_offset_m": rgpo_m, "vel_offset_mps": vgpo_mps}
    elif mode == "FALSE_TARGETS":
        kwargs = {
            "specs": [
                FalseTargetSpec(delta_range_m=300.0, delta_velocity_mps=15.0),
                FalseTargetSpec(delta_range_m=750.0, delta_velocity_mps=-18.0),
                FalseTargetSpec(delta_range_m=1200.0, delta_velocity_mps=25.0),
            ]
        }
    elif mode == "ISRJ":
        kwargs = {"slice_period_us": 2.0, "duty_cycle": 0.5, "repeat_count": 2}

    cb = None if mode == "OFF" else drfm.make_cpi_callback(cfg, mode, n_fast=n_fast, **kwargs)
    rx_cube, _ = simulate_radar_cpi(tx, cfg, [tgt], imp, drfm_callback=cb, max_range_m=max_r, rng=rng)
    rd_db, r_ax, v_ax = range_doppler_map(matched_filter(rx_cube, tx), cfg, n_doppler_fft=64)

    r_mask = (r_ax >= 11400.0) & (r_ax <= 13800.0)
    rd_zoom = rd_db[:, r_mask]
    r_sub = r_ax[r_mask]
    # Downsample fast-time by 4x for fast JSON transfer to browser canvas
    rd_ds = rd_zoom[:, ::4]
    r_ds = r_sub[::4]

    dets, _ = ca_cfar_2d(10.0 ** (rd_zoom / 10.0), guard=(2, 4), train=(4, 8), pfa=1e-4)
    pk_d, pk_r = np.unravel_index(int(np.argmax(rd_zoom)), rd_zoom.shape)

    # Also compute quantization purity for current adc_bits
    n_tone = 1024
    t = np.arange(n_tone) / cfg.fs_hz
    tone = 0.98 * np.exp(1j * 2.0 * np.pi * (83 * cfg.fs_hz / n_tone) * t)
    q_metrics = measure_sqnr_sfdr(tone, quantize_complex_drfm(tone, bits=adc_bits))

    return {
        "rd_db": np.round(rd_ds, 2).tolist(),
        "range_km": np.round(r_ds / 1e3, 3).tolist(),
        "vel_mps": np.round(v_ax, 2).tolist(),
        "vmax": float(np.max(rd_ds)),
        "peak_range_km": round(float(r_sub[pk_r] / 1e3), 3),
        "peak_vel_mps": round(float(v_ax[pk_d]), 2),
        "true_range_km": 12.0,
        "true_vel_mps": -25.0,
        "cfar_cells": int(np.sum(dets)),
        "sqnr_db": round(q_metrics["sqnr_db"], 2),
        "sfdr_dbc": round(q_metrics["sfdr_dbc"], 2),
        "enob": round(q_metrics["enob"], 2),
    }


def run_crosseye_sim(params: dict) -> dict:
    a_ratio = float(params.get("amplitude_ratio", 0.90))
    phi_deg = float(params.get("phase_deg", 178.0))
    baseline_m = float(params.get("baseline_m", 15.0))
    range_m = float(params.get("range_km", 3.0)) * 1e3
    jsr_db = float(params.get("jsr_db", 25.0))

    arr = AESAMonopulseArray(num_elements=16, fc_hz=10e9, boresight_deg=0.0)
    cfg = CrossEyeConfig(
        baseline_m=baseline_m,
        amplitude_ratio=a_ratio,
        phase_deg=phi_deg,
        target_range_m=range_m,
        jsr_db=jsr_db,
    )
    res = simulate_crosseye_vs_monopulse(arr, cfg, include_skin_echo=True, snr_db=35.0)
    eccm = detect_crosseye_eccm(res["measurement"], quadrature_thresh=0.05)

    # Phase sweep curve for plotting
    phis = np.linspace(168.0, 192.0, 61)
    angles_out = []
    quads_out = []
    for ph in phis:
        c2 = CrossEyeConfig(baseline_m=baseline_m, amplitude_ratio=a_ratio, phase_deg=float(ph), target_range_m=range_m, jsr_db=jsr_db)
        r2 = simulate_crosseye_vs_monopulse(arr, c2, include_skin_echo=True, snr_db=None)
        angles_out.append(round(float(r2["measurement"].est_angle_deg), 3))
        quads_out.append(round(float(abs(r2["measurement"].ratio_imag)), 4))

    return {
        "est_angle_deg": round(float(res["measurement"].est_angle_deg), 3),
        "theta_e_deg": round(float(res["theta_e_deg"]), 3),
        "pull_factor": round(float(abs(res["angle_error_deg"]) / max(res["theta_e_deg"], 1e-6)), 2),
        "crosseye_gain": round(float(res["crosseye_gain_real"]), 2),
        "quadrature_ratio": round(float(eccm["quadrature_ratio"]), 4),
        "crosseye_detected": bool(eccm["crosseye_detected"]),
        "sweep_phi": phis.tolist(),
        "sweep_angle": angles_out,
        "sweep_quad": quads_out,
    }


class EWStudioHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if parsed.path == "/api/health":
            self._send_json({"status": "ok", "suite": "AEGIS-DRFM Web EW Studio"})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        params = json.loads(raw.decode("utf-8"))
        parsed = urlparse(self.path)

        if parsed.path == "/api/simulate_rd":
            self._send_json(run_rd_sim(params))
            return
        if parsed.path == "/api/simulate_crosseye":
            self._send_json(run_crosseye_sim(params))
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        return  # Keep terminal output clean


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), EWStudioHandler)
    print(f"[AEGIS-DRFM Web EW Studio] Running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
