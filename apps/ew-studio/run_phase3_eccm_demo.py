"""AEGIS-DRFM Phase 3 Demo: ECCM & Smart Anti-DRFM Radar Defense Dashboard.

Demonstrates 3 orthogonal ECCM defense layers against our own DRFM ECM:
  1. Signal-Level ECCM: Pulse-to-Pulse Waveform Agility (Up/Down LFM) suppressing 1-PRI DRFM replay.
  2. Micro-Signature ECCM: DRFM Quantization/Spur Fingerprinting + ROC/AUC across 1, 2, 4, 8 bits.
  3. Kinematic ECCM: Range-Doppler Consistency Gated Kalman Tracker rejecting an active RGPO attack.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "eccm-radar" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def run_fingerprint_roc_experiment(bit_depths=(1, 2, 4, 8), n_per_class: int = 120, snr_db: float = 18.0):
    rng = np.random.default_rng(2026)
    fs = 50e6
    ref = generate_lfm_chirp(fs, pulse_width_s=10e-6, bandwidth_hz=20e6, up=True)

    results = {}
    for bits in bit_depths:
        drfm = DRFMCore(DRFMConfig(adc_bits=bits, phase_noise_std_rad=0.015), rng=rng)
        feats = []
        labels = []
        for _ in range(n_per_class):
            # Real passive echo
            rx_real = add_awgn(ref, snr_db=snr_db, signal_power=1.0, rng=rng)
            feats.append(extract_pulse_fingerprint(rx_real, ref))
            labels.append(0)
            # DRFM replay echo
            rx_drfm = add_awgn(drfm.capture(ref), snr_db=snr_db, signal_power=1.0, rng=rng)
            feats.append(extract_pulse_fingerprint(rx_drfm, ref))
            labels.append(1)

        X = np.vstack(feats)
        y = np.array(labels, dtype=np.int32)
        # 50/50 train-test split
        X_tr, y_tr = X[:n_per_class], y[:n_per_class]
        X_te, y_te = X[n_per_class:], y[n_per_class:]

        clf = DRFMFingerprintDetector().fit(X_tr, y_tr)
        scores = clf.decision_function(X_te)
        pfa, pd, auc = compute_roc_auc(y_te, scores)
        results[bits] = {"pfa": pfa, "pd": pd, "auc": auc}
    return results


def run_rgpo_kinematic_defense_sim():
    rng = np.random.default_rng(55)
    dt = 0.05
    n_steps = 80
    r0 = 12000.0
    v_true = -200.0  # Closing at 200 m/s

    tracker = KinematicECCMTracker(
        init_range_m=r0,
        init_vel_mps=v_true,
        dt_s=dt,
        sigma_r_m=6.0,
        sigma_v_mps=1.5,
        process_accel_std=2.0,
        kinematic_thresh_mps=15.0,
    )

    t_arr, r_true_arr, r_meas_arr, r_est_arr, kin_res_arr, alerts = [], [], [], [], [], []
    for k in range(n_steps):
        t = k * dt
        r_true = r0 + v_true * t
        # Active uncoordinated RGPO walk-off between t=1.2s and t=3.0s (parabolic pull-off up to +800 m)
        if 1.2 <= t <= 3.0:
            tau = t - 1.2
            rgpo_pull = 250.0 * (tau ** 2)
        else:
            rgpo_pull = 0.0

        r_meas = r_true + rgpo_pull + float(rng.normal(0.0, 5.0))
        v_meas = v_true + float(rng.normal(0.0, 1.2))
        res = tracker.step(MeasurementRD(t_s=t, range_m=r_meas, velocity_mps=v_meas))

        t_arr.append(t)
        r_true_arr.append(r_true)
        r_meas_arr.append(r_meas)
        r_est_arr.append(res.est_range_m)
        kin_res_arr.append(res.kinematic_residual_mps)
        alerts.append(res.ecm_alert)

    return (
        np.array(t_arr),
        np.array(r_true_arr),
        np.array(r_meas_arr),
        np.array(r_est_arr),
        np.array(kin_res_arr),
        alerts,
    )


def main() -> None:
    out_dir = REPO_ROOT / "apps" / "ew-studio" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RadarConfig(num_pulses=32, prf_hz=5000.0, fs_hz=50e6, bandwidth_hz=20e6, pulse_width_s=10e-6)
    tgt = TargetState(range_m=10000.0, velocity_mps=-25.0, rcs_m2=5.0)

    # 1. Waveform Agility ECCM
    res_fixed = evaluate_agility_suppression(cfg, tgt, false_range_offset_m=500.0, jsr_db=12.0, mode="fixed")
    res_agile = evaluate_agility_suppression(cfg, tgt, false_range_offset_m=500.0, jsr_db=12.0, mode="up_down_lfm")
    eccm_gain_db = float(res_agile["sjr_out_db"]) - float(res_fixed["sjr_out_db"])
    print(
        f"[ECCM-1] Waveform Agility: Fixed SJR={res_fixed['sjr_out_db']:+.1f} dB -> "
        f"Agile SJR={res_agile['sjr_out_db']:+.1f} dB (ECCM Suppression Gain = +{eccm_gain_db:.1f} dB)"
    )

    # 2. DRFM Fingerprinting ROC/AUC
    roc_res = run_fingerprint_roc_experiment(bit_depths=(1, 2, 4, 8), n_per_class=120, snr_db=18.0)
    print("[ECCM-2] DRFM Micro-Signature Detector AUC:")
    for b, d in roc_res.items():
        print(f"    {b}-bit DRFM -> AUC = {d['auc']:.3f}")

    # 3. Kinematic Kalman ECCM vs RGPO
    t_arr, r_true, r_meas, r_est, kin_res, alerts = run_rgpo_kinematic_defense_sim()
    rms_unprotected = float(np.sqrt(np.mean((r_meas - r_true) ** 2)))
    rms_eccm = float(np.sqrt(np.mean((r_est - r_true) ** 2)))
    n_rejected = sum(1 for a in alerts if a != "CLEAN")
    print(
        f"[ECCM-3] Kinematic Kalman vs RGPO: Unprotected RMS={rms_unprotected:.1f} m -> "
        f"ECCM Kalman RMS={rms_eccm:.1f} m ({n_rejected} spoofed updates rejected)"
    )

    # Plot 2x2 ECCM Defense Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    # Panel 1: Waveform Agility Range Profile
    ax = axes[0, 0]
    r_km = res_fixed["range_axis_m"] / 1e3
    mask = (r_km >= 9.6) & (r_km <= 11.0)
    ax.plot(r_km[mask], res_fixed["range_profile_db"][mask], "--", color="#ef4444", lw=1.4, label="Fixed LFM (Vulnerable)")
    ax.plot(r_km[mask], res_agile["range_profile_db"][mask], "-", color="#2563eb", lw=1.8, label="Pulse-Agile Up/Down LFM (ECCM)")
    ax.axvline(10.0, color="#10b981", ls=":", lw=1.4, label="True Target (10.0 km)")
    ax.axvline(10.5, color="#f59e0b", ls=":", lw=1.4, label="DRFM False Target (10.5 km)")
    ax.set_title(f"1. Waveform Agility ECCM (+{eccm_gain_db:.1f} dB Jammer Suppression)", fontweight="bold")
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Pulse-Compressed Power (dB)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: DRFM Fingerprint ROC Curves
    ax = axes[0, 1]
    colors = {1: "#ef4444", 2: "#f59e0b", 4: "#2563eb", 8: "#10b981"}
    for b, d in roc_res.items():
        ax.plot(d["pfa"], d["pd"], lw=2.0, color=colors[b], label=f"{b}-bit DRFM (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random Chance")
    ax.set_title("2. DRFM Quantization/Spur Fingerprint ROC (SNR=18 dB)", fontweight="bold")
    ax.set_xlabel("False Alarm Rate (Pfa)")
    ax.set_ylabel("Detection Probability (Pd)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Kalman Tracker vs RGPO
    ax = axes[1, 0]
    ax.plot(t_arr, r_meas / 1e3, "o", markersize=3.5, color="#ef4444", alpha=0.75, label="Radar Meas (Under RGPO Walk-Off)")
    ax.plot(t_arr, r_true / 1e3, "k--", lw=1.8, label="True Target Trajectory")
    ax.plot(t_arr, r_est / 1e3, "-", color="#2563eb", lw=2.0, label=f"ECCM Kalman Track (RMS={rms_eccm:.1f} m)")
    ax.axvspan(1.2, 3.0, color="#ef4444", alpha=0.08, label="Active RGPO Attack Window")
    ax.set_title("3. Kinematic ECCM Kalman Tracker Rejecting RGPO Walk-Off", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Range (km)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Kinematic Consistency Residual
    ax = axes[1, 1]
    ax.plot(t_arr, kin_res, lw=1.8, color="#7c3aed", label="|dR/dt - v_Doppler| Residual")
    ax.axhline(15.0, color="#ef4444", ls="--", lw=1.4, label="ECCM Consistency Alarm (15 m/s)")
    ax.axvspan(1.2, 3.0, color="#ef4444", alpha=0.08)
    ax.set_title("4. Range-Doppler Kinematic Consistency Monitor", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Discrepancy (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("AEGIS-DRFM Phase 3: Multi-Layer ECCM & Anti-DRFM Smart Radar Defense", fontsize=13, fontweight="bold")
    out_path = out_dir / "phase3_eccm_defense_dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ECCM] Saved Phase 3 dashboard: {out_path}")


if __name__ == "__main__":
    main()
