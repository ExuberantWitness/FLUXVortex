"""Rojratsirikul 2011 case definition for the unified framework."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RojratsirikulCaseConfig:
    case_id: str
    angle_deg: float
    # Paper digitized targets (from the frozen adapter)
    target_zmax_over_c: float
    target_cn_band: tuple[float, float]
    # Material branch label
    material_branch: str = "primary_E2.2"
    # E=1.4 gets post_hoc_calibrated_sensitivity, NOT independent verification
    is_calibration_sensitivity: bool = False

ROJ_A16_PRIMARY = RojratsirikulCaseConfig(
    case_id="ROJ11-A16",
    angle_deg=16.0,
    target_zmax_over_c=0.043,
    target_cn_band=(0.92, 0.95),
    material_branch="primary_E2.2",
    is_calibration_sensitivity=False,
)

ROJ_A16_E14 = RojratsirikulCaseConfig(
    case_id="ROJ11-A16-E14",
    angle_deg=16.0,
    target_zmax_over_c=0.043,
    target_cn_band=(0.92, 0.95),
    material_branch="post_hoc_calibrated_sensitivity_E1.4",
    is_calibration_sensitivity=True,
)
