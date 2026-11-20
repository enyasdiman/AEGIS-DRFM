"""AEGIS-DRFM Phase 1 End-to-End Demo & Visual Analytics.

Loads unified radar/EW parameters exported by MATLAB R2025b (system_config.json),
simulates an X-band LFM radar CPI under multiple DRFM ECM modes, and generates
publication-grade Range-Doppler and STFT Spectrogram comparison figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

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
from aegis_drfm import DRFMConfig, DRFMCore, FalseTargetSpec


def main() -> None:
    cfg_path = REPO_ROOT / "matlab" / "outputs" / "system_config.json"
    out_dir = REPO_ROOT / "apps" / "ew-studio" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    radar_cfg = RadarConfig.from_matlab_json(cfg_path) if cfg_path.exists() else RadarConfig()
    # Focus on a clean unambiguous Doppler velocity (-45 m/s) for clear visual RD inspection
    target = TargetState(range_m=12000.0, velocity_mps=-25.0, rcs_m2=5.0, name="Hostile-Fighter")
    tx_pulse = generate_lfm_chirp(radar_cfg.fs_hz, radar_cfg.pulse_width_s, radar_cfg.bandwidth_hz, up=True)

    max_range_m = 15000.0
    max_delay_s = 2.0 * max_range_m / C_LIGHT
    n_fast = int(np.ceil(max_delay_s * radar_cfg.fs_hz)) + len(tx_pulse) + 64

    impairments = RFImpairments(
        snr_single_pulse_db=5.0,
        phase_noise_std_rad=0.003,
        iq_amp_imbalance_db=0.1,
        iq_phase_imbalance_deg=0.5,
    )

    drfm_cfg = DRFMConfig(adc_bits=8, loop_delay_ns=120.0, jsr_db=8.0, phase_noise_std_rad=0.002)
    rng = np.random.default_rng(42)

    scenarios = [
        ("1. Clean Target (No ECM)", "OFF", {}),
        ("2. DRFM RGPO (+700 m)", "RGPO", {"range_offset_m": 700.0}),
        ("3. DRFM VGPO (+30 m/s)", "VGPO", {"vel_offset_mps": 30.0}),
        (
            "4. Multiple False Targets (3 Ghosts)",
            "FALSE_TARGETS",
            {
                "specs": [
                    FalseTargetSpec(delta_range_m=350.0, delta_velocity_mps=15.0, rel_power_db=0.0),
                    FalseTargetSpec(delta_range_m=800.0, delta_velocity_mps=-18.0, rel_power_db=-1.0),
                    FalseTargetSpec(delta_range_m=1300.0, delta_velocity_mps=25.0, rel_power_db=-2.0),
                ]
            },
        ),
        (
            "5. DRFM ISRJ (2 us Slice, 2x Repeat)",
            "ISRJ",
            {"slice_period_us": 2.0, "duty_cycle": 0.5, "repeat_count": 2},
        ),
        ("6. Coordinated RGPO+VGPO", "RGPO_VGPO", {"range_offset_m": 550.0, "vel_offset_mps": 22.0}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.ravel()

    isrj_sample_pulse = None

    print(f"[AEGIS-DRFM] Running Phase 1 Demo with {radar_cfg.name} (fc={radar_cfg.fc_hz/1e9:.1f} GHz, B={radar_cfg.bandwidth_hz/1e6:.0f} MHz)")
    for ax, (title, mode, kwargs) in zip(axes_flat, scenarios):
        drfm = DRFMCore(config=drfm_cfg, rng=rng)
        cb = None if mode == "OFF" else drfm.make_cpi_callback(radar_cfg, mode, n_fast=n_fast, **kwargs)
        rx_cube, _ = simulate_radar_cpi(
            tx_pulse=tx_pulse,
            radar_cfg=radar_cfg,
            targets=[target],
            impairments=impairments,
            drfm_callback=cb,
            max_range_m=max_range_m,
            rng=rng,
        )
        if mode == "ISRJ":
            isrj_sample_pulse = rx_cube[0, :]

        comp = matched_filter(rx_cube, tx_pulse)
        rd_db, r_axis, v_axis = range_doppler_map(comp, radar_cfg, doppler_window="hann", n_doppler_fft=128)
        rd_pow = 10.0 ** (rd_db / 10.0)
        dets, _ = ca_cfar_2d(rd_pow, guard=(2, 4), train=(4, 8), pfa=1e-5)

        # Zoom around target range [11.2 km, 14.0 km]
        r_mask = (r_axis >= 11200.0) & (r_axis <= 14000.0)
        r_km = r_axis[r_mask] / 1e3
        rd_zoom = rd_db[:, r_mask]
        vmax = float(np.max(rd_zoom))

        im = ax.imshow(
            rd_zoom,
            aspect="auto",
            origin="lower",
            extent=[r_km[0], r_km[-1], v_axis[0], v_axis[-1]],
            vmin=vmax - 45.0,
            vmax=vmax,
            cmap="inferno",
        )
        # Mark true target position
        ax.plot(target.range_m / 1e3, target.velocity_mps, "c+", markersize=10, markeredgewidth=1.8, label="True Target")

        # Report peak location in zoomed window
        pk_d, pk_r = np.unravel_index(int(np.argmax(rd_zoom)), rd_zoom.shape)
        print(
            f"  {title:36s} | Peak: R={r_km[pk_r]:6.3f} km, V={v_axis[pk_d]:+6.1f} m/s | CFAR cells={int(np.sum(dets[:, r_mask]))}"
        )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Range (km)")
        ax.set_ylabel("Radial Velocity (m/s)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dB")

    rd_fig_path = out_dir / "phase1_range_doppler_ecm_gallery.png"
    fig.suptitle("AEGIS-DRFM Phase 1: Range-Doppler Map Under DRFM ECM Techniques", fontsize=13, fontweight="bold")
    fig.savefig(rd_fig_path, dpi=150)
    plt.close(fig)

    # Generate STFT Spectrogram of Clean LFM vs ISRJ pulse
    if isrj_sample_pulse is not None:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

        # Clean LFM pulse padded for STFT
        clean_pad = np.zeros(len(tx_pulse) * 3, dtype=np.complex128)
        clean_pad[len(tx_pulse) : 2 * len(tx_pulse)] = tx_pulse
        f1, t1, s1 = spectrogram(clean_pad, fs=radar_cfg.fs_hz, nperseg=64, noverlap=56, return_onesided=False)
        s1_shift = np.fft.fftshift(s1, axes=0)
        f1_shift = np.fft.fftshift(f1) / 1e6
        ax1.pcolormesh(t1 * 1e6, f1_shift, 10 * np.log10(s1_shift + 1e-15), shading="gouraud", cmap="viridis")
        ax1.set_title("Transmitted LFM Chirp Spectrogram (20 MHz, 10 us)")
        ax1.set_xlabel("Time (us)")
        ax1.set_ylabel("Baseband Frequency (MHz)")

        # Slice around target echo in ISRJ received row
        start_idx = int((2.0 * target.range_m / C_LIGHT) * radar_cfg.fs_hz) - 100
        seg = isrj_sample_pulse[max(0, start_idx) : start_idx + len(tx_pulse) * 3]
        f2, t2, s2 = spectrogram(seg, fs=radar_cfg.fs_hz, nperseg=64, noverlap=56, return_onesided=False)
        s2_shift = np.fft.fftshift(s2, axes=0)
        f2_shift = np.fft.fftshift(f2) / 1e6
        ax2.pcolormesh(t2 * 1e6, f2_shift, 10 * np.log10(s2_shift + 1e-15), shading="gouraud", cmap="viridis")
        ax2.set_title("Received Echo + DRFM ISRJ Sliced Replay Spectrogram")
        ax2.set_xlabel("Time (us)")
        ax2.set_ylabel("Baseband Frequency (MHz)")

        stft_path = out_dir / "phase1_stft_lfm_vs_isrj.png"
        fig2.savefig(stft_path, dpi=150)
        plt.close(fig2)
        print(f"[AEGIS-DRFM] Saved STFT plot: {stft_path}")

    print(f"[AEGIS-DRFM] Saved RD gallery: {rd_fig_path}")


if __name__ == "__main__":
    main()
