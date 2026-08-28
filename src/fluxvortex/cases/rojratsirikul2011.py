"""Rojratsirikul 2011 unified case schema (HANDOFF_UNIFIED_FRAMEWORK §9 P0).

Four formal members live here: A10, A16 (primary accuracy case), A17-MODE
(dynamic-mode case at the paper's ~17° condition) and A23.  A16 and A17 are
separate cases with separate oracles — the A17 mode/St targets are never
applied to A16 (handoff §4.1).

Every frozen value carries one evidence role (handoff §9 P0 item 2):

* ``paper_printed``              — printed in the paper text/figures;
* ``derived_from_printed_pairs`` — cross-fit from printed (U, Re)/(U, Pi1);
* ``digitized_approx``           — project figure reading, not an author table;
* ``model_assumption``           — parameter the paper does not report;
* ``numerical_protocol``         — solver protocol, not an experimental input.

The frozen platform adapter
(``platform/forward_flight_benchmarks/rojratsirikul2011_q16.py``) stays the
numerical-input oracle for mesh building and statistics; this module holds
its own copy of the shared constants and ``to_platform_case()`` projects onto
the adapter's frozen dataclass so both cannot drift (enforced by tests).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import hashlib

from ._roj_platform_bridge import make_platform_case

ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS_CSV = (
    ROOT
    / "artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/observations/"
    "rojratsirikul2011_digitized_observations.csv"
)
OBSERVATIONS_CSV_SHA256 = (
    "4864749f8d3718f3571a71a33a97365feec1d2f3762a410751bfb1056585f12e"
)

# Evidence roles for the shared physical constants (single source in this
# module; the platform adapter mirrors them and tests pin the equality).
FIELD_ROLES: dict[str, str] = {
    "chord_m": "paper_printed",
    "span_m": "paper_printed",
    "thickness_m": "paper_printed",
    "young_modulus_pa": "paper_printed",
    "membrane_density_kg_m3": "paper_printed",
    "freestream_m_s": "paper_printed",
    "kinematic_viscosity_m2_s": "derived_from_printed_pairs",
    "fluid_density_kg_m3": "derived_from_printed_pairs",
    "poisson_ratio_assumed": "model_assumption",
    "prestress_n_m_assumed": "model_assumption",
    "structural_damping_loss_factor": "model_assumption",
    "aerodynamic_dt_star": "numerical_protocol",
    "structural_substeps_per_aerodynamic_step": "numerical_protocol",
    "startup_time_star": "numerical_protocol",
    "statistics_min_time_star": "numerical_protocol",
    "statistics_start_time_star": "numerical_protocol",
    "lesp_crit": "model_assumption",
    "target_zmax_over_c": "digitized_approx",
    "target_cn_band": "digitized_approx",
    "target_strouhal": "digitized_approx",
    "target_chordwise_peak_count": "digitized_approx",
    "target_spanwise_peak_count": "digitized_approx",
}

# Shared frozen physics (identical to the platform adapter defaults; the
# equality is pinned by tests on every field).
_SHARED = dict(
    chord_m=0.0688,
    span_m=0.1375,
    thickness_m=2.0e-4,
    young_modulus_pa=2.2e6,
    membrane_density_kg_m3=1000.0,
    freestream_m_s=5.0,
    kinematic_viscosity_m2_s=1.4142e-5,
    fluid_density_kg_m3=1.208,
    poisson_ratio_assumed=0.49,
    prestress_n_m_assumed=0.0,
    structural_damping_loss_factor=0.1,
    aerodynamic_dt_star=0.01,
    structural_substeps_per_aerodynamic_step=10,
    startup_time_star=1.0,
    statistics_min_time_star=20.0,
    statistics_start_time_star=4.0,
    lesp_crit=0.11,
)


@dataclass(frozen=True)
class RojratsirikulCaseConfig:
    case_id: str
    angle_deg: float
    # Paper digitized targets (from the frozen adapter / handoff §4.2)
    target_zmax_over_c: float
    target_cn_band: tuple[float, float]
    # Material branch label
    material_branch: str = "primary_E2.2"
    # E=1.4 gets post_hoc_calibrated_sensitivity, NOT independent verification
    is_calibration_sensitivity: bool = False
    # Dynamic-mode oracles (None = not gated for this case; A16 deliberately
    # carries no St/peak oracle so the A17 mode targets cannot leak into it).
    target_strouhal: float | None = None
    target_chordwise_peak_count: int | None = None
    target_spanwise_peak_count: int | None = None
    # Project digitized tolerance on St (handoff §9 P8: A17/A23 use ±0.08,
    # A10 uses ±0.10); zmax keeps the ±0.005 and Cn the ±10% project gates.
    strouhal_tolerance: float = 0.10
    zmax_tolerance: float = 0.005
    cn_band_relative_tolerance: float = 0.10
    purpose: str = ""
    shared_field_roles: tuple[tuple[str, str], ...] = (
        tuple(sorted(FIELD_ROLES.items()))
    )
    _shared: tuple[tuple[str, float], ...] = tuple(sorted(_SHARED.items()))

    # -- shared frozen physics (read-only properties over _shared) ----------
    @property
    def chord_m(self) -> float:
        return dict(self._shared)["chord_m"]

    @property
    def span_m(self) -> float:
        return dict(self._shared)["span_m"]

    @property
    def thickness_m(self) -> float:
        return dict(self._shared)["thickness_m"]

    @property
    def young_modulus_pa(self) -> float:
        return dict(self._shared)["young_modulus_pa"]

    @property
    def membrane_density_kg_m3(self) -> float:
        return dict(self._shared)["membrane_density_kg_m3"]

    @property
    def freestream_m_s(self) -> float:
        return dict(self._shared)["freestream_m_s"]

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        return dict(self._shared)["kinematic_viscosity_m2_s"]

    @property
    def fluid_density_kg_m3(self) -> float:
        return dict(self._shared)["fluid_density_kg_m3"]

    @property
    def poisson_ratio_assumed(self) -> float:
        return dict(self._shared)["poisson_ratio_assumed"]

    @property
    def prestress_n_m_assumed(self) -> float:
        return dict(self._shared)["prestress_n_m_assumed"]

    @property
    def structural_damping_loss_factor(self) -> float:
        return dict(self._shared)["structural_damping_loss_factor"]

    @property
    def aerodynamic_dt_star(self) -> float:
        return dict(self._shared)["aerodynamic_dt_star"]

    @property
    def structural_substeps_per_aerodynamic_step(self) -> float:
        return dict(self._shared)["structural_substeps_per_aerodynamic_step"]

    @property
    def startup_time_star(self) -> float:
        return dict(self._shared)["startup_time_star"]

    @property
    def statistics_min_time_star(self) -> float:
        return dict(self._shared)["statistics_min_time_star"]

    @property
    def statistics_start_time_star(self) -> float:
        return dict(self._shared)["statistics_start_time_star"]

    @property
    def lesp_crit(self) -> float:
        return dict(self._shared)["lesp_crit"]

    @property
    def damping_evidence_role(self) -> str:
        """eta=0.1 is a literature assumption, never a paper input."""

        return "assumed_literature_sensitivity"

    def with_shared(self, **overrides: float) -> "RojratsirikulCaseConfig":
        merged = dict(self._shared)
        unknown = set(overrides) - set(merged)
        if unknown:
            raise ValueError(f"unknown shared fields: {sorted(unknown)}")
        merged.update(overrides)
        return replace(self, _shared=tuple(sorted(merged.items())))

    def with_material_branch(
        self, *, young_modulus_pa: float, branch: str, case_id: str | None = None
    ) -> "RojratsirikulCaseConfig":
        """Labeled material-uncertainty branch (handoff §3.3 discipline)."""

        return replace(
            self.with_shared(young_modulus_pa=young_modulus_pa),
            case_id=case_id or self.case_id,
            material_branch=branch,
            is_calibration_sensitivity=True,
        )

    def to_platform_case(self) -> Any:
        """Project onto the frozen adapter dataclass (mesh/statistics input)."""

        shared = dict(self._shared)
        return make_platform_case(
            self.case_id,
            self.angle_deg,
            zmax_over_c=self.target_zmax_over_c,
            cn_low=self.target_cn_band[0],
            cn_high=self.target_cn_band[1],
            strouhal=self.target_strouhal,
            chordwise_peaks=self.target_chordwise_peak_count,
            spanwise_peaks=self.target_spanwise_peak_count,
            **shared,
        )


ROJ_A10 = RojratsirikulCaseConfig(
    case_id="ROJ11-A10",
    angle_deg=10.0,
    target_zmax_over_c=0.032,
    target_cn_band=(0.50, 0.52),
    target_strouhal=1.10,
    target_chordwise_peak_count=3,
    target_spanwise_peak_count=3,
    strouhal_tolerance=0.10,
    purpose="low/mid-angle generalization: 3 chordwise + 3 spanwise peaks, St~1.10",
)
ROJ_A16_PRIMARY = RojratsirikulCaseConfig(
    case_id="ROJ11-A16",
    angle_deg=16.0,
    target_zmax_over_c=0.043,
    target_cn_band=(0.92, 0.95),
    # A16 gates ONLY the mean quantities; the dynamic-mode oracles at ~17 deg
    # belong to ROJ11-A17-MODE and must never be applied here.
    target_strouhal=None,
    target_chordwise_peak_count=None,
    target_spanwise_peak_count=None,
    purpose="primary accuracy case: max(mean z)/c and same-window mean Cn",
)
ROJ_A17_MODE = RojratsirikulCaseConfig(
    case_id="ROJ11-A17-MODE",
    angle_deg=17.0,
    target_zmax_over_c=0.0445,
    target_cn_band=(0.97, 0.97),
    target_strouhal=0.85,
    target_chordwise_peak_count=2,
    # "spanwise peaks no longer visible" — zero peaks above the digitization
    # threshold, which is itself the oracle (handoff §4.2).
    target_spanwise_peak_count=0,
    strouhal_tolerance=0.08,
    purpose="dynamic-mode case: 2 chordwise peaks, spanwise peaks gone, St~0.85",
)
ROJ_A23 = RojratsirikulCaseConfig(
    case_id="ROJ11-A23",
    angle_deg=23.0,
    target_zmax_over_c=0.0475,
    target_cn_band=(0.98, 1.02),
    target_strouhal=0.83,
    target_chordwise_peak_count=2,
    target_spanwise_peak_count=None,
    strouhal_tolerance=0.08,
    purpose="deep separation: chordwise two-peak dominance, St~0.83, "
    "amplitude below A17",
)

ROJ_A16_E14 = ROJ_A16_PRIMARY.with_material_branch(
    young_modulus_pa=1.4e6,
    branch="post_hoc_calibrated_sensitivity_E1.4",
    case_id="ROJ11-A16-E14",
)

ROJRATSIRIKUL2011_UNIFIED_CASES: dict[str, RojratsirikulCaseConfig] = {
    config.case_id: config
    for config in (ROJ_A10, ROJ_A16_PRIMARY, ROJ_A17_MODE, ROJ_A23)
}
ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES: dict[str, RojratsirikulCaseConfig] = {
    ROJ_A16_E14.case_id: ROJ_A16_E14
}


def validate_rojratsirikul2011_unified_sources() -> None:
    """Verify the frozen observation CSV (digitized GT) hash."""

    if not OBSERVATIONS_CSV.is_file():
        raise FileNotFoundError(OBSERVATIONS_CSV)
    observed = hashlib.sha256(OBSERVATIONS_CSV.read_bytes()).hexdigest()
    if observed != OBSERVATIONS_CSV_SHA256:
        raise RuntimeError(f"Rojratsirikul 2011 observation CSV drift: {observed}")


def cross_check_against_platform_adapter() -> dict[str, bool]:
    """The unified constants and the frozen adapter must agree everywhere."""

    from forward_flight_benchmarks.rojratsirikul2011_q16 import (
        ROJRATSIRIKUL2011_CASES as PLATFORM_CASES,
    )

    report: dict[str, bool] = {}
    shared = dict(ROJ_A16_PRIMARY._shared)
    shared_fields = (
        "chord_m",
        "span_m",
        "thickness_m",
        "young_modulus_pa",
        "membrane_density_kg_m3",
        "freestream_m_s",
        "kinematic_viscosity_m2_s",
        "fluid_density_kg_m3",
        "poisson_ratio_assumed",
        "prestress_n_m_assumed",
        "structural_damping_loss_factor",
        "aerodynamic_dt_star",
        "structural_substeps_per_aerodynamic_step",
        "startup_time_star",
        "statistics_min_time_star",
        "statistics_start_time_star",
        "lesp_crit",
    )
    for name in shared_fields:
        report[f"shared.{name}"] = shared[name] == getattr(
            PLATFORM_CASES["ROJ11-A16"], name
        )
    for case_id in ("ROJ11-A10", "ROJ11-A16", "ROJ11-A23"):
        unified = ROJRATSIRIKUL2011_UNIFIED_CASES[case_id]
        platform = PLATFORM_CASES[case_id]
        report[f"{case_id}.angle_deg"] = unified.angle_deg == platform.angle_deg
        report[f"{case_id}.zmax"] = (
            unified.target_zmax_over_c == platform.digitized_approx_zmax_over_c
        )
        report[f"{case_id}.cn_band"] = unified.target_cn_band == (
            platform.digitized_approx_cn_low,
            platform.digitized_approx_cn_high,
        )
        report[f"{case_id}.strouhal"] = (
            unified.target_strouhal == platform.digitized_approx_strouhal
        )
        report[f"{case_id}.chordwise_peaks"] = (
            unified.target_chordwise_peak_count
            == platform.digitized_approx_chordwise_peak_count
        )
        report[f"{case_id}.spanwise_peaks"] = (
            unified.target_spanwise_peak_count
            == platform.digitized_approx_spanwise_peak_count
        )
    return report


__all__ = [
    "FIELD_ROLES",
    "OBSERVATIONS_CSV",
    "OBSERVATIONS_CSV_SHA256",
    "ROJ_A10",
    "ROJ_A16_E14",
    "ROJ_A16_PRIMARY",
    "ROJ_A17_MODE",
    "ROJ_A23",
    "ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES",
    "ROJRATSIRIKUL2011_UNIFIED_CASES",
    "RojratsirikulCaseConfig",
    "cross_check_against_platform_adapter",
    "validate_rojratsirikul2011_unified_sources",
]
