"""AEGIS-DRFM Extension 4: AESA / Monopulse Beamforming & Cross-Eye DRFM Angle Deception."""

from .monopulse import AESAMonopulseArray, MonopulseMeasurement
from .crosseye import (
    CrossEyeConfig,
    compute_crosseye_gain,
    detect_crosseye_eccm,
    simulate_crosseye_vs_monopulse,
)

__all__ = [
    "AESAMonopulseArray",
    "MonopulseMeasurement",
    "CrossEyeConfig",
    "compute_crosseye_gain",
    "simulate_crosseye_vs_monopulse",
    "detect_crosseye_eccm",
]
