"""Shared data structures and physical constants for the AEGIS-DRFM ecosystem."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

C_LIGHT: float = 299_792_458.0
K_BOLTZMANN: float = 1.380649e-23


@dataclass
class RadarConfig:
    """Radar waveform and RF front-end parameters."""

    name: str = "AEGIS-XBand-Tracker"
    fc_hz: float = 10.0e9
    bandwidth_hz: float = 20.0e6
    pulse_width_s: float = 10.0e-6
    prf_hz: float = 5000.0
    num_pulses: int = 64
    fs_hz: float = 50.0e6
    pt_w: float = 25e3
    gt_db: float = 32.0
    gr_db: float = 32.0
    nf_db: float = 4.0
    loss_db: float = 5.0

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.fc_hz

    @property
    def pri_s(self) -> float:
        return 1.0 / self.prf_hz

    @property
    def unambiguous_range_m(self) -> float:
        return C_LIGHT / (2.0 * self.prf_hz)

    @property
    def range_resolution_m(self) -> float:
        return C_LIGHT / (2.0 * self.bandwidth_hz)

    @property
    def doppler_resolution_hz(self) -> float:
        return self.prf_hz / self.num_pulses

    @property
    def velocity_resolution_mps(self) -> float:
        return self.doppler_resolution_hz * self.wavelength_m / 2.0

    @classmethod
    def from_matlab_json(cls, path: str | Path) -> "RadarConfig":
        """Load unified radar parameters exported by MATLAB rf_ew_system_design.m."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        r = data["radar"]
        return cls(
            name=r.get("name", "AEGIS-XBand-Tracker"),
            fc_hz=float(r["fc_hz"]),
            bandwidth_hz=float(r["bandwidth_hz"]),
            pulse_width_s=float(r["pulse_width_s"]),
            prf_hz=float(r["prf_hz"]),
            num_pulses=int(r["num_pulses"]),
            fs_hz=float(r["fs_hz"]),
            pt_w=float(r["pt_w"]),
            gt_db=float(r["gt_db"]),
            gr_db=float(r["gr_db"]),
            nf_db=float(r["nf_db"]),
            loss_db=float(r["loss_db"]),
        )


@dataclass
class TargetState:
    """Point scatterer target kinematics and RCS."""

    range_m: float
    velocity_mps: float  # Negative = closing (positive Doppler), Positive = receding
    rcs_m2: float = 5.0
    name: str = "Target-1"


@dataclass
class RFImpairments:
    """Realistic RF channel and analog front-end impairments."""

    snr_single_pulse_db: Optional[float] = 0.0
    phase_noise_std_rad: float = 0.0
    iq_amp_imbalance_db: float = 0.0
    iq_phase_imbalance_deg: float = 0.0
    # List of (extra_delay_samples, complex_gain) multipath taps
    multipath_taps: List[Tuple[float, complex]] = field(default_factory=list)


@dataclass
class PDW:
    """Pulse Descriptor Word — shared contract between SIGINT/ESM and Cognitive DRFM."""

    toa_s: float
    pw_s: float
    rf_hz: float
    pa_db: float
    pri_s: Optional[float] = None
    modulation: str = "UNKNOWN"  # CW, LFM_UP, LFM_DOWN, BPSK_BARKER, POLYPHASE
    emitter_id: Optional[str] = None


@dataclass
class ECMCommand:
    """Cognitive EW command passed from ESM/Threat Library to DRFM ECM engine."""

    mode: str  # RGPO, VGPO, RGPO_VGPO, FALSE_TARGETS, ISRJ, NOISE, OFF
    jsr_db: float = 10.0
    params: dict = field(default_factory=dict)
