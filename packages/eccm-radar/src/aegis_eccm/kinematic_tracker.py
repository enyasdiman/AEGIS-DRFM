"""Kinematic ECCM: Range-Doppler Consistency & Gated Kalman Tracker (Anti-RGPO / Anti-VGPO)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MeasurementRD:
    """Radar range + Doppler radial velocity measurement at time t_s."""

    t_s: float
    range_m: float
    velocity_mps: float


@dataclass
class TrackStepResult:
    """Output of a single ECCM Kalman tracker update step."""

    t_s: float
    est_range_m: float
    est_vel_mps: float
    meas_range_m: float
    meas_vel_mps: float
    kinematic_residual_mps: float
    mahalanobis_dist: float
    ecm_alert: str  # 'CLEAN', 'RGPO_REJECTED', 'VGPO_REJECTED', 'GATE_REJECTED'


class KinematicECCMTracker:
    """2-State Constant-Velocity Kalman Tracker with Range-Doppler Kinematic Consistency ECCM.

    Detects and rejects:
      - Uncoordinated RGPO: measured range rate (dR/dt) disagrees with measured Doppler velocity.
      - Uncoordinated VGPO: measured Doppler velocity jumps inconsistently with range progression.
      - Sudden false-target capture: Mahalanobis innovation distance exceeds chi-squared gate.
    """

    def __init__(
        self,
        init_range_m: float,
        init_vel_mps: float,
        dt_s: float = 0.05,
        sigma_r_m: float = 8.0,
        sigma_v_mps: float = 1.5,
        process_accel_std: float = 3.0,
        kinematic_thresh_mps: float = 12.0,
        gate_chi2_thresh: float = 13.8,  # 2-DOF p < 0.001 gate
    ) -> None:
        self.dt = dt_s
        self.x = np.array([init_range_m, init_vel_mps], dtype=np.float64)
        self.P = np.diag([sigma_r_m ** 2, sigma_v_mps ** 2])
        self.R_full = np.diag([sigma_r_m ** 2, sigma_v_mps ** 2])
        self.sigma_r = sigma_r_m
        self.sigma_v = sigma_v_mps
        self.q_acc = process_accel_std
        self.kinematic_thresh_mps = kinematic_thresh_mps
        self.gate_chi2_thresh = gate_chi2_thresh

        self._prev_meas_r: float = init_range_m
        self._prev_t: Optional[float] = None
        self._meas_window: List[MeasurementRD] = []
        self.history: List[TrackStepResult] = []

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = self.q_acc ** 2
        Q = q * np.array(
            [[0.25 * (dt ** 4), 0.5 * (dt ** 3)], [0.5 * (dt ** 3), dt ** 2]],
            dtype=np.float64,
        )
        return F, Q

    def step(self, meas: MeasurementRD) -> TrackStepResult:
        """Predict, test kinematic consistency + Mahalanobis gate, and selectively update."""
        dt = self.dt if self._prev_t is None else max(meas.t_s - self._prev_t, 1e-4)
        F, Q = self._transition(dt)

        # 1. Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # 2. Multi-step smoothed Range-rate vs Doppler kinematic consistency check
        self._meas_window.append(meas)
        if len(self._meas_window) > 6:
            self._meas_window.pop(0)
        if len(self._meas_window) >= 2:
            dt_win = max(self._meas_window[-1].t_s - self._meas_window[0].t_s, dt)
            dr_dt = (self._meas_window[-1].range_m - self._meas_window[0].range_m) / dt_win
            v_avg = float(np.mean([m.velocity_mps for m in self._meas_window]))
            kinematic_res = abs(dr_dt - v_avg)
        else:
            kinematic_res = 0.0

        # Innovation & Mahalanobis distance (2D)
        z = np.array([meas.range_m, meas.velocity_mps], dtype=np.float64)
        y_innov = z - x_pred
        S = P_pred + self.R_full
        mahal_sq = float(y_innov.T @ np.linalg.solve(S, y_innov))
        mahal = float(np.sqrt(max(mahal_sq, 0.0)))

        r_err = abs(meas.range_m - x_pred[0])
        v_err = abs(meas.velocity_mps - x_pred[1])

        alert = "CLEAN"
        is_anomalous = (mahal_sq > self.gate_chi2_thresh) or (
            len(self._meas_window) >= 4 and kinematic_res > self.kinematic_thresh_mps
        )
        if is_anomalous:
            # Classify which channel was spoofed and perform selective partial update or coast
            if r_err > 3.5 * np.sqrt(S[0, 0]) and v_err <= 3.5 * np.sqrt(S[1, 1]):
                # RGPO: Range gate pulled off, Doppler still consistent -> update ONLY velocity!
                alert = "RGPO_REJECTED"
                H = np.array([[0.0, 1.0]], dtype=np.float64)
                S_v = float((H @ P_pred @ H.T)[0, 0] + self.sigma_v ** 2)
                K = (P_pred @ H.T) / S_v
                self.x = x_pred + (K.ravel() * (meas.velocity_mps - x_pred[1]))
                self.P = (np.eye(2) - K @ H) @ P_pred
            elif v_err > 3.5 * np.sqrt(S[1, 1]) and r_err <= 3.5 * np.sqrt(S[0, 0]):
                # VGPO: Doppler pulled off, Range still consistent -> update ONLY range!
                alert = "VGPO_REJECTED"
                H = np.array([[1.0, 0.0]], dtype=np.float64)
                S_r = float((H @ P_pred @ H.T)[0, 0] + self.sigma_r ** 2)
                K = (P_pred @ H.T) / S_r
                self.x = x_pred + (K.ravel() * (meas.range_m - x_pred[0]))
                self.P = (np.eye(2) - K @ H) @ P_pred
            else:
                # Both channels anomalous -> coast on kinematic prediction!
                alert = "GATE_REJECTED"
                self.x = x_pred
                self.P = P_pred
        else:
            # Clean full 2D Kalman update
            K = np.linalg.solve(S, P_pred).T
            self.x = x_pred + K @ y_innov
            self.P = (np.eye(2) - K) @ P_pred

        self._prev_meas_r = meas.range_m
        self._prev_t = meas.t_s

        res = TrackStepResult(
            t_s=meas.t_s,
            est_range_m=float(self.x[0]),
            est_vel_mps=float(self.x[1]),
            meas_range_m=meas.range_m,
            meas_vel_mps=meas.velocity_mps,
            kinematic_residual_mps=float(kinematic_res),
            mahalanobis_dist=mahal,
            ecm_alert=alert,
        )
        self.history.append(res)
        return res
