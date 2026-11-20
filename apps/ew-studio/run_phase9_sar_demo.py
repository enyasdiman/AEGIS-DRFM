"""AEGIS-DRFM Phase 9 Demo: SAR Imaging Radar vs. Coherent DRFM False-Scene Deception."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "drfm-ecm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "sar-isar" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aegis_sar import SARConfig, make_aircraft_target, sar_backprojection_image, simulate_sar_raw


def main() -> None:
    out_dir = REPO_ROOT / "apps" / "ew-studio" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SARConfig(num_pulses=72, aperture_m=110.0)
    real_aircraft = make_aircraft_target(center_az_m=-12.0, center_rng_m=-8.0, scale=1.0)
    decoy_aircraft = make_aircraft_target(center_az_m=+14.0, center_rng_m=+10.0, scale=1.0)

    az_grid = np.linspace(-30.0, 30.0, 121)
    rng_grid = np.linspace(-26.0, 26.0, 105)
    rng = np.random.default_rng(42)

    print("[SAR-DRFM] Synthesizing 3 X-Band SAR scenarios (Time-Domain Backprojection)...")

    # 1. Clean SAR
    raw_clean, tx, r0_s = simulate_sar_raw(real_aircraft, cfg, snr_db=15.0, rng=rng)
    img_clean = sar_backprojection_image(raw_clean, tx, cfg, r0_s, az_grid, rng_grid)

    # 2. DRFM Coherent False-Aircraft Injection
    raw_decoy, tx, r0_s = simulate_sar_raw(
        real_aircraft, cfg, drfm_decoy_scatterers=decoy_aircraft, drfm_jsr_db=3.0, drfm_adc_bits=8, snr_db=15.0, rng=rng
    )
    img_decoy = sar_backprojection_image(raw_decoy, tx, cfg, r0_s, az_grid, rng_grid)

    # 3. DRFM High-Power Barrage Blanket
    raw_blanket, tx, r0_s = simulate_sar_raw(
        real_aircraft, cfg, drfm_blanket_jsr_db=22.0, snr_db=15.0, rng=rng
    )
    img_blanket = sar_backprojection_image(raw_blanket, tx, cfg, r0_s, az_grid, rng_grid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    panels = [
        ("1. Clean X-Band SAR Image\n(Real Aircraft at Az=-12m, Rng=-8m)", img_clean),
        ("2. DRFM False-Scene Deception\n(Coherent Ghost Aircraft at Az=+14m, Rng=+10m)", img_decoy),
        ("3. DRFM Barrage Blanket Jamming\n(J/S = 22 dB Noise Obscuration)", img_blanket),
    ]

    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(
            img,
            aspect="equal",
            origin="lower",
            extent=[az_grid[0], az_grid[-1], rng_grid[0], rng_grid[-1]],
            vmin=-30.0,
            vmax=0.0,
            cmap="gray",
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Cross-Range / Azimuth (m)")
        ax.set_ylabel("Slant Range Offset (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dB")

    fig.suptitle("AEGIS-DRFM Extension 5: X-Band SAR Imaging Radar vs. DRFM False-Scene Injection", fontsize=13, fontweight="bold")
    out_path = out_dir / "phase9_sar_drfm_deception.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAR-DRFM] Saved Phase 9 SAR dashboard: {out_path}")


if __name__ == "__main__":
    main()
