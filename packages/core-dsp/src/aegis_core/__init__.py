"""AEGIS-DRFM Core DSP, Radar, and RF Channel Library."""

from .types import (
    C_LIGHT,
    K_BOLTZMANN,
    ECMCommand,
    PDW,
    RadarConfig,
    RFImpairments,
    TargetState,
)
from .waveforms import (
    generate_barker_code,
    generate_frank_code,
    generate_lfm_chirp,
    generate_pri_schedule,
)
from .rf_channel import (
    add_awgn,
    apply_doppler_shift,
    apply_fractional_delay,
    apply_iq_imbalance,
    apply_lo_phase_noise,
    apply_multipath,
    simulate_radar_cpi,
)
from .radar_rx import (
    ca_cfar_2d,
    matched_filter,
    os_cfar_1d,
    range_doppler_map,
)

__all__ = [
    "C_LIGHT",
    "K_BOLTZMANN",
    "RadarConfig",
    "TargetState",
    "RFImpairments",
    "PDW",
    "ECMCommand",
    "generate_lfm_chirp",
    "generate_barker_code",
    "generate_frank_code",
    "generate_pri_schedule",
    "apply_fractional_delay",
    "apply_doppler_shift",
    "apply_lo_phase_noise",
    "apply_iq_imbalance",
    "apply_multipath",
    "add_awgn",
    "simulate_radar_cpi",
    "matched_filter",
    "range_doppler_map",
    "ca_cfar_2d",
    "os_cfar_1d",
]
