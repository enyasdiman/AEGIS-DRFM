"""AEGIS-DRFM Core: Digital Radio Frequency Memory & Electronic Countermeasures (ECM)."""

from .quantizer import measure_sqnr_sfdr, quantize_complex_drfm
from .ecm import DRFMConfig, DRFMCore, FalseTargetSpec

__all__ = [
    "quantize_complex_drfm",
    "measure_sqnr_sfdr",
    "DRFMConfig",
    "DRFMCore",
    "FalseTargetSpec",
]
