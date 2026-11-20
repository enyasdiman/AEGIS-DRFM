"""AEGIS-DRFM Phase 5 Demo: AESA Monopulse Beamforming & Cross-Eye DRFM Angle Deception."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "aesa-monopulse" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aegis_aesa import (
    AESAMonopulseArray,
    CrossEyeConfig,
    compute_crosseye_gain,
    detect_crosseye_eccm,
    simulate_crosseye_vs_monopulse,
)


def main() -> None:
    out_dir = REPO_ROOT / "apps" / "ew-studio" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    array = AESAMonopulseArray(num_elements=16, fc_hz=10.0e9, spacing_wavelengths=0.5, boresight_deg=0.0)

    # 1. Sum/Diff Beam Patterns & S-curve
    angles = np.linspace(-25.0, 25.0, 501)
    sum_db, diff_db, s_curve = array.beam_patterns(angles)

    # 2. Cross-Eye Gain Sensitivity Map over Amplitude Ratio a and Phase Error delta_phi
    a_vals = np.linspace(0.75, 0.99, 100)
    phi_vals = np.linspace(170.0, 190.0, 101)
    A_grid, P_grid = np.meshgrid(a_vals, phi_vals)
    GC_real = np.zeros_like(A_grid)
    GC_imag = np.zeros_like(A_grid)
    for i in range(A_grid.shape[0]):
        for j in range(A_grid.shape[1]):
            gr, gi = compute_crosseye_gain(float(A_grid[i, j]), float(P_grid[i, j]))
            GC_real[i, j] = np.clip(gr, -5.0, 25.0)
            GC_imag[i, j] = abs(gi)

    # 3. Sweep Cross-Eye Phase from 165 deg to 195 deg at R = 3 km, d_j = 15 m, a = 0.90
    sweep_phi = np.linspace(165.0, 195.0, 121)
    est_angles = []
    quad_ratios = []
    sum_pows = []
    for phi in sweep_phi:
        cfg = CrossEyeConfig(baseline_m=15.0, amplitude_ratio=0.90, phase_deg=float(phi), target_range_m=3000.0, jsr_db=25.0)
        res = simulate_crosseye_vs_monopulse(array, cfg, include_skin_echo=True, snr_db=None)
        est_angles.append(res["measurement"].est_angle_deg)
        quad_ratios.append(abs(res["measurement"].ratio_imag))
        sum_pows.append(res["measurement"].sum_power_db)

    # Highlight operating point a=0.90, phi=178 deg
    op_cfg = CrossEyeConfig(baseline_m=15.0, amplitude_ratio=0.90, phase_deg=178.0, target_range_m=3000.0, jsr_db=25.0)
    op_res = simulate_crosseye_vs_monopulse(array, op_cfg, include_skin_echo=True, snr_db=35.0)
    eccm_res = detect_crosseye_eccm(op_res["measurement"], quadrature_thresh=0.10)

    print(f"[AESA-MONOPULSE] 16-Element X-Band ULA | Monopulse slope k_m = {array.k_monopulse:.4f} / deg")
    print(
        f"[CROSS-EYE] Wingtip baseline=15 m @ 3 km (half-angle={op_res['theta_e_deg']:.3f} deg) | "
        f"a=0.90, phi=178 deg -> Apparent Angle={op_res['measurement'].est_angle_deg:+.2f} deg "
        f"(Pull-Off = {abs(op_res['angle_error_deg'])/op_res['theta_e_deg']:.1f}x wingspan!)"
    )
    print(
        f"[ECCM-QUAD] Quadrature |Im(Delta/j*Sigma)| = {eccm_res['quadrature_ratio']:.3f} "
        f"-> Cross-Eye Detected: {eccm_res['crosseye_detected']}"
    )

    # 2x2 Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    # Panel 1: Sum & Difference Beam Patterns
    ax = axes[0, 0]
    ax.plot(angles, sum_db, lw=2.0, color="#2563eb", label="Sum Channel (Sigma)")
    ax.plot(angles, diff_db, lw=2.0, color="#ef4444", ls="--", label="Difference Channel (Delta)")
    ax.set_ylim(-45, 2)
    ax.set_title("1. 16-Element AESA Monopulse Sum & Difference Patterns", fontweight="bold")
    ax.set_xlabel("Angle Off-Boresight (deg)")
    ax.set_ylabel("Normalized Gain (dB)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cross-Eye Gain G_C Heatmap
    ax = axes[0, 1]
    im = ax.pcolormesh(a_vals, phi_vals, GC_real, shading="auto", cmap="magma")
    ax.plot(0.90, 178.0, "c*", markersize=12, label="Operating Point (a=0.90, 178 deg)")
    ax.set_title("2. Cross-Eye Gain G_C Sensitivity (Amplitude Ratio vs. Phase)", fontweight="bold")
    ax.set_xlabel("DRFM Amplitude Ratio (a)")
    ax.set_ylabel("DRFM Relative Phase (deg)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cross-Eye Gain G_C")

    # Panel 3: Apparent Monopulse Angle Pull-Off vs Phase
    ax = axes[1, 0]
    ax.plot(sweep_phi, est_angles, lw=2.2, color="#dc2626", label="Apparent Monopulse Angle")
    ax.axhline(+op_res["theta_e_deg"], color="#10b981", ls="--", lw=1.2, label="Physical Wingtip Bounds (+/- 0.14 deg)")
    ax.axhline(-op_res["theta_e_deg"], color="#10b981", ls="--", lw=1.2)
    ax.set_title("3. Monopulse Angle Pull-Off Outside Aircraft Wingspan", fontweight="bold")
    ax.set_xlabel("Cross-Eye Relative Phase (deg)")
    ax.set_ylabel("Indicated Angle (deg)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: ECCM Quadrature Monopulse Detector
    ax = axes[1, 1]
    ax.plot(sweep_phi, quad_ratios, lw=2.2, color="#7c3aed", label="|Im(Delta / j*Sigma)| Quadrature Channel")
    ax.axhline(0.10, color="#ef4444", ls="--", lw=1.4, label="ECCM Cross-Eye Alarm Threshold (0.10)")
    ax.set_title("4. ECCM Counter-Cross-Eye Quadrature Anomaly Monitor", fontweight="bold")
    ax.set_xlabel("Cross-Eye Relative Phase (deg)")
    ax.set_ylabel("Quadrature Monopulse Ratio")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("AEGIS-DRFM Phase 5: AESA Monopulse Radar vs. Retrodirective Cross-Eye DRFM", fontsize=13, fontweight="bold")
    out_path = out_dir / "phase5_aesa_crosseye_dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[AESA-MONOPULSE] Saved Phase 5 dashboard: {out_path}")


if __name__ == "__main__":
    main()
