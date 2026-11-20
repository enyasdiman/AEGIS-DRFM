"""AEGIS-DRFM Phase 2 Demo: Multi-Emitter SIGINT/ELINT & Closed-Loop Cognitive EW.

Synthesizes a dense 3-radar interleaved wideband IQ stream:
  1. AEGIS-XBand-FireControl   (LFM_UP,      RF=10.000 GHz, PRI=200 us, PW=10.0 us)
  2. Barker-Airborne-Interceptor (BPSK_BARKER, RF=10.014 GHz, PRI=130 us, PW= 6.5 us)
  3. CW-Maritime-Surveillance  (CW,          RF= 9.985 GHz, PRI=340 us, PW= 4.0 us)

Pipeline:
  Wideband IQ -> Blind Pulse Detector -> PDW + Intra-Pulse MOP Classifier
              -> SDIF Deinterleaver -> Threat Library Match -> Cognitive DRFM ECM!
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "sigint-esm" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aegis_core import (
    C_LIGHT,
    RadarConfig,
    RFImpairments,
    TargetState,
    add_awgn,
    generate_barker_code,
    generate_lfm_chirp,
    matched_filter,
    range_doppler_map,
    simulate_radar_cpi,
)
from aegis_drfm import DRFMConfig, DRFMCore
from aegis_sigint import CognitiveEWController, deinterleave_sdif, detect_pulses_iq


def synthesize_multi_emitter_iq(
    fs_hz: float = 50e6,
    duration_s: float = 2.2e-3,
    rf_center_hz: float = 10.0e9,
    snr_db: float = 16.0,
    seed: int = 123,
) -> np.ndarray:
    """Create a noisy wideband IQ capture containing 3 interleaved radar emitters."""
    rng = np.random.default_rng(seed)
    n_total = int(round(duration_s * fs_hz))
    iq = np.zeros(n_total, dtype=np.complex128)

    def inject_train(pulse: np.ndarray, f_offset_hz: float, pri_s: float, t_start_s: float, amp: float) -> None:
        n_p = len(pulse)
        t_local = np.arange(n_p, dtype=np.float64) / fs_hz
        shifted = amp * pulse * np.exp(1j * 2.0 * np.pi * f_offset_hz * t_local)
        t_cur = t_start_s
        while t_cur + (n_p / fs_hz) < duration_s:
            idx = int(round(t_cur * fs_hz))
            if idx + n_p <= n_total:
                iq[idx : idx + n_p] += shifted
            t_cur += pri_s

    # Emitter A: Fire-Control Tracker (LFM Up-Chirp, 12 MHz BW, 10 us PW, 200 us PRI, +0 MHz)
    p_fc = generate_lfm_chirp(fs_hz, pulse_width_s=10.0e-6, bandwidth_hz=12.0e6, up=True)
    inject_train(p_fc, f_offset_hz=0.0, pri_s=200e-6, t_start_s=25e-6, amp=1.0)

    # Emitter B: Airborne Interceptor (Barker-13 BPSK, 0.5 us chip -> 6.5 us PW, 130 us PRI, +14 MHz)
    p_int = generate_barker_code(fs_hz, chip_width_s=0.5e-6, length=13)
    inject_train(p_int, f_offset_hz=+14.0e6, pri_s=130e-6, t_start_s=62e-6, amp=0.85)

    # Emitter C: Maritime Surveillance (CW, 4.0 us PW, 340 us PRI, -15 MHz)
    p_surv = np.ones(int(round(4.0e-6 * fs_hz)), dtype=np.complex128)
    inject_train(p_surv, f_offset_hz=-15.0e6, pri_s=340e-6, t_start_s=115e-6, amp=0.75)

    # Add receiver thermal noise relative to unit pulse power
    iq_noisy = add_awgn(iq, snr_db=snr_db, signal_power=1.0, rng=rng)
    return iq_noisy


def main() -> None:
    out_dir = REPO_ROOT / "apps" / "ew-studio" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_hz = 50e6
    rf_center_hz = 10.0e9
    iq_stream = synthesize_multi_emitter_iq(fs_hz=fs_hz, duration_s=2.2e-3, rf_center_hz=rf_center_hz, snr_db=16.0)

    # Step 1: Blind Pulse Detection & PDW Extraction
    pdws = detect_pulses_iq(iq_stream, fs_hz=fs_hz, rf_center_hz=rf_center_hz, threshold_db_above_noise=10.0)
    print(f"[SIGINT] Detected {len(pdws)} raw pulses from wideband IQ stream.")

    # Step 2: SDIF Deinterleaving
    tracks, unassigned = deinterleave_sdif(pdws)
    print(f"[SIGINT] Deinterleaved {len(tracks)} distinct emitters ({len(unassigned)} unassigned pulses).")

    # Step 3: Threat Library Identification & Cognitive EW Prioritization
    controller = CognitiveEWController()
    ranked_matches = controller.assess_environment(tracks)

    print("\n--- ELINT / ESM Emitter Table (Prioritized) ---")
    for m in ranked_matches:
        sig_name = m.signature.name if m.signature else "UNKNOWN"
        prio = m.signature.priority if m.signature else 99
        t = m.track
        print(
            f"  [P{prio}] {t.emitter_id} | {sig_name:28s} | Conf={m.confidence:.2f} | "
            f"RF={t.rf_hz/1e9:7.4f} GHz | PRI={t.pri_s*1e6:6.1f} us | PW={t.pw_s*1e6:4.1f} us | "
            f"MOP={t.modulation:11s} | N={t.pulse_count:2d} -> ECM: {m.ecm_command.mode}"
        )

    # Step 4: Closed-Loop Cognitive EW Engagement on Highest-Priority Threat
    primary = controller.select_primary_response(tracks)
    assert primary is not None, "No primary threat detected!"
    print(f"\n[COGNITIVE EW] Engaging {primary.signature.name} with {primary.ecm_command.mode} (J/S={primary.ecm_command.jsr_db} dB)...")

    radar_cfg = RadarConfig(
        fc_hz=primary.track.rf_hz,
        bandwidth_hz=12.0e6,
        pulse_width_s=primary.track.pw_s,
        prf_hz=1.0 / primary.track.pri_s,
        num_pulses=48,
        fs_hz=fs_hz,
    )
    tx_pulse = generate_lfm_chirp(radar_cfg.fs_hz, radar_cfg.pulse_width_s, radar_cfg.bandwidth_hz, up=True)
    target = TargetState(range_m=10000.0, velocity_mps=-25.0, rcs_m2=5.0)
    max_r = 12500.0
    n_fast = int(np.ceil((2.0 * max_r / C_LIGHT) * radar_cfg.fs_hz)) + len(tx_pulse) + 64

    rng = np.random.default_rng(77)
    drfm = DRFMCore(DRFMConfig(adc_bits=8, loop_delay_ns=120.0, jsr_db=primary.ecm_command.jsr_db), rng=rng)
    cb = drfm.make_cpi_callback(radar_cfg, primary.ecm_command.mode, n_fast=n_fast, **primary.ecm_command.params)
    rx_cube, _ = simulate_radar_cpi(
        tx_pulse, radar_cfg, [target], RFImpairments(snr_single_pulse_db=6.0),
        drfm_callback=cb, max_range_m=max_r, rng=rng,
    )
    rd_db, r_ax, v_ax = range_doppler_map(matched_filter(rx_cube, tx_pulse), radar_cfg, n_doppler_fft=128)

    # Visualization Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    # Panel 1: Interleaved Wideband IQ Power Envelope (first 0.8 ms)
    ax = axes[0, 0]
    n_show = int(0.8e-3 * fs_hz)
    t_ms = np.arange(n_show) / fs_hz * 1e3
    env_db = 10.0 * np.log10(np.abs(iq_stream[:n_show]) ** 2 + 1e-12)
    ax.plot(t_ms, env_db, lw=0.6, color="#334155")
    ax.set_title("1. Wideband Interleaved IQ Capture (First 0.8 ms)", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Instantaneous Power (dB)")
    ax.set_ylim(-25, 10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Deinterleaved PDWs (TOA vs RF colored by Emitter)
    ax = axes[0, 1]
    colors = ["#ef4444", "#2563eb", "#10b981", "#f59e0b"]
    for idx, m in enumerate(ranked_matches):
        t = m.track
        toas_ms = [p.toa_s * 1e3 for p in t.pdws]
        rfs_mhz = [(p.rf_hz - rf_center_hz) / 1e6 for p in t.pdws]
        lbl = f"{t.emitter_id}: {m.signature.name if m.signature else 'UNK'} ({t.modulation}, PRI={t.pri_s*1e6:.0f}us)"
        ax.scatter(toas_ms, rfs_mhz, s=45, color=colors[idx % len(colors)], label=lbl, edgecolors="k", lw=0.4)
    ax.set_title("2. SDIF Deinterleaved Emitters (TOA vs. Baseband RF)", fontweight="bold")
    ax.set_xlabel("Time of Arrival (ms)")
    ax.set_ylabel("RF Offset from 10 GHz (MHz)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Intra-Pulse Instantaneous Frequency / Phase Signatures
    ax = axes[1, 0]
    for idx, m in enumerate(ranked_matches):
        p0 = m.track.pdws[0]
        s0 = int(round(p0.toa_s * fs_hz))
        e0 = s0 + int(round(p0.pw_s * fs_hz))
        seg = iq_stream[s0:e0]
        f_off = p0.rf_hz - rf_center_hz
        t_loc = np.arange(len(seg)) / fs_hz
        base = seg * np.exp(-1j * 2.0 * np.pi * f_off * t_loc)
        inst_f_mhz = np.angle(base[1:] * np.conj(base[:-1])) * fs_hz / (2.0 * np.pi) / 1e6
        # Light smoothing for visual clarity
        if len(inst_f_mhz) > 5:
            inst_f_mhz = np.convolve(inst_f_mhz, np.ones(5) / 5.0, mode="valid")
        ax.plot(np.arange(len(inst_f_mhz)) / fs_hz * 1e6, inst_f_mhz, lw=1.5, color=colors[idx % len(colors)],
                label=f"{m.track.emitter_id} ({m.track.modulation})")
    ax.set_title("3. Extracted Intra-Pulse Modulation (MOP) Signatures", fontweight="bold")
    ax.set_xlabel("Intra-Pulse Time (us)")
    ax.set_ylabel("Baseband Inst. Frequency (MHz)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Closed-Loop Cognitive DRFM Engagement Result (RD Map)
    ax = axes[1, 1]
    r_mask = (r_ax >= 9400.0) & (r_ax <= 11600.0)
    r_km = r_ax[r_mask] / 1e3
    rd_zoom = rd_db[:, r_mask]
    vmax = float(np.max(rd_zoom))
    im = ax.imshow(
        rd_zoom, aspect="auto", origin="lower",
        extent=[r_km[0], r_km[-1], v_ax[0], v_ax[-1]],
        vmin=vmax - 45.0, vmax=vmax, cmap="inferno",
    )
    ax.plot(target.range_m / 1e3, target.velocity_mps, "c+", markersize=11, markeredgewidth=2.0, label="True Target")
    ax.set_title(
        f"4. Closed-Loop Cognitive DRFM: {primary.ecm_command.mode} on {primary.signature.name}",
        fontweight="bold",
    )
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Radial Velocity (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dB")

    fig.suptitle("AEGIS-DRFM Phase 2: SIGINT / ELINT Deinterleaving & Closed-Loop Cognitive EW", fontsize=13, fontweight="bold")
    out_path = out_dir / "phase2_sigint_elint_dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SIGINT] Saved Phase 2 dashboard: {out_path}")


if __name__ == "__main__":
    main()
