"""AEGIS-DRFM Extension 2: ECCM & Smart Anti-DRFM Radar Defense."""

from .waveform_agility import (
    evaluate_agility_suppression,
    generate_agile_pulse_bank,
)
from .drfm_fingerprint import (
    DRFMFingerprintDetector,
    compute_roc_auc,
    extract_pulse_fingerprint,
)
from .kinematic_tracker import (
    KinematicECCMTracker,
    MeasurementRD,
    TrackStepResult,
)

__all__ = [
    "generate_agile_pulse_bank",
    "evaluate_agility_suppression",
    "extract_pulse_fingerprint",
    "DRFMFingerprintDetector",
    "compute_roc_auc",
    "MeasurementRD",
    "TrackStepResult",
    "KinematicECCMTracker",
]
