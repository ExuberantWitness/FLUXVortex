"""Source-derived LDVM separated-flow correction for a retained UVLM load.

This module does not replace the finite-wing UVLM.  It runs two otherwise
identical clean-room Ramesh-style two-dimensional section histories:

* an attached reference with the LESP cap disabled; and
* a separated history with an explicit, provenance-carrying LESP threshold.

Only their load difference is eligible to correct the UVLM result.  Therefore
``Lcrit -> infinity`` and a history that never reaches the threshold reduce
exactly to the original UVLM.  This is the first causal diagnostic bridge; the
next production step is to replace the independent strips by mutually coupled
three-dimensional LEV wake rows in the UVLM AIC.

The implementation intentionally imports :class:`ldvm_fourier.LDVM2D` rather
than copying the GPL-v3 Fortran source.  The Python model is a clean-room paper
translation already present in the project and is independently regression
checked against the author's bundled v2.5 output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LESPThreshold:
    """A required section/Re-specific LDVM threshold with provenance."""

    value: float
    section_family: str
    reynolds: float
    source: str
    source_role: str = "published_model_parameter"

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value <= 0.0:
            raise ValueError("LESP threshold must be finite and positive")
        if not np.isfinite(self.reynolds) or self.reynolds <= 0.0:
            raise ValueError("Reynolds number must be finite and positive")
        if not self.section_family.strip() or not self.source.strip():
            raise ValueError("LESP threshold provenance must be explicit")

    def manifest(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class LDVMSectionSettings:
    ndiv: int = 50
    naterm: int = 24
    core_radius_chord: float | None = 0.02
    core_radius_time_step_ratio: float = 1.3
    max_wake_steps: int = 512
    attached_lesp_critical: float = 1.0e6

    def __post_init__(self) -> None:
        if self.ndiv < 12 or self.naterm < 3:
            raise ValueError("LDVM Fourier resolution is too small")
        if self.naterm >= self.ndiv:
            raise ValueError("naterm must be smaller than ndiv")
        if self.core_radius_chord is not None and self.core_radius_chord <= 0.0:
            raise ValueError("fixed core radius must be positive")
        if self.core_radius_time_step_ratio <= 0.0:
            raise ValueError("core-radius/time-step ratio must be positive")
        if self.max_wake_steps < 8:
            raise ValueError("max_wake_steps must be at least eight")


def _validate_history(
    alpha_rad: np.ndarray,
    alpha_rate_per_convective_time: np.ndarray,
    heave_rate_over_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.asarray(alpha_rad, dtype=float)
    alpha_rate = np.asarray(alpha_rate_per_convective_time, dtype=float)
    heave_rate = np.asarray(heave_rate_over_u, dtype=float)
    if alpha.ndim != 1 or alpha.size < 4:
        raise ValueError("section history must be a one-dimensional time series")
    if alpha_rate.shape != alpha.shape or heave_rate.shape != alpha.shape:
        raise ValueError("alpha, alpha-rate and heave-rate histories must align")
    if not np.all(np.isfinite(alpha + alpha_rate + heave_rate)):
        raise FloatingPointError("section history contains non-finite values")
    return alpha, alpha_rate, heave_rate


def run_ldvm_separation_pair(
    *,
    alpha_rad: np.ndarray,
    alpha_rate_per_convective_time: np.ndarray,
    heave_rate_over_u: np.ndarray,
    delta_time_convective: float,
    pivot_fraction_chord: float,
    threshold: LESPThreshold,
    settings: LDVMSectionSettings = LDVMSectionSettings(),
) -> dict[str, Any]:
    """Return separated-minus-attached LDVM histories for one section.

    Time is nondimensionalized by ``c/U``.  ``heave_rate_over_u`` is positive
    in the clean-room LDVM's upward direction.  The output contains complete
    separated and attached histories so the correction and its force owner can
    be audited independently.
    """

    if not np.isfinite(delta_time_convective) or delta_time_convective <= 0.0:
        raise ValueError("delta_time_convective must be finite and positive")
    if not 0.0 <= pivot_fraction_chord <= 1.0:
        raise ValueError("pivot must lie on the chord")
    alpha, alpha_rate, heave_rate = _validate_history(
        alpha_rad, alpha_rate_per_convective_time, heave_rate_over_u
    )

    from ldvm_fourier import LDVM2D

    common = dict(
        U=1.0,
        c=1.0,
        ndiv=settings.ndiv,
        naterm=settings.naterm,
        dt=float(delta_time_convective),
        rho=1.0,
        camber_m=0.0,
        pivot_xc=float(pivot_fraction_chord),
        core_rc=(
            settings.core_radius_chord
            if settings.core_radius_chord is not None
            else settings.core_radius_time_step_ratio * float(delta_time_convective)
        ),
        max_wake=settings.max_wake_steps,
    )
    separated = LDVM2D(lesp_crit=threshold.value, **common)
    attached = LDVM2D(lesp_crit=settings.attached_lesp_critical, **common)

    fields = (
        "CLf",
        "CDf",
        "CNf",
        "CNc",
        "CNnc",
        "CNnonl",
        "CSf",
        "A0",
        "lesp",
    )
    separated_history = {field: np.empty(alpha.size) for field in fields}
    attached_history = {field: np.empty(alpha.size) for field in fields}
    lev_count = np.empty(alpha.size, dtype=int)

    for index, (angle, rate, plunge_rate) in enumerate(
        zip(alpha, alpha_rate, heave_rate)
    ):
        result_separated = separated.step(float(angle), float(rate), float(plunge_rate))
        result_attached = attached.step(float(angle), float(rate), float(plunge_rate))
        for field in fields:
            separated_history[field][index] = result_separated[field]
            attached_history[field][index] = result_attached[field]
        lev_count[index] = int(result_separated["n_lev"])

    delta = {
        field: separated_history[field] - attached_history[field]
        for field in (
            "CLf",
            "CDf",
            "CNf",
            "CNc",
            "CNnc",
            "CNnonl",
            "CSf",
        )
    }
    if not all(np.all(np.isfinite(value)) for value in delta.values()):
        raise FloatingPointError("LDVM pair produced a non-finite correction")

    return {
        "separated": separated_history,
        "attached": attached_history,
        "delta": delta,
        "lev_count": lev_count,
        "shed_lev": np.diff(np.concatenate(([0], lev_count))) > 0,
        "threshold": threshold.manifest(),
        "settings": asdict(settings),
        "delta_time_convective": float(delta_time_convective),
        "pivot_fraction_chord": float(pivot_fraction_chord),
        "semantics": (
            "causal clean-room Ramesh LDVM separated-minus-attached section "
            "correction; finite-wing UVLM remains the baseline load owner"
        ),
    }


def apply_section_delta_to_coefficients(
    baseline_cl: np.ndarray,
    baseline_cd: np.ndarray,
    section_pair: dict[str, Any],
    *,
    finite_wing_gain: float,
) -> dict[str, np.ndarray]:
    """Apply a section-pair correction without replacing the UVLM baseline."""

    cl = np.asarray(baseline_cl, dtype=float)
    cd = np.asarray(baseline_cd, dtype=float)
    delta_cl = np.asarray(section_pair["delta"]["CLf"], dtype=float)
    delta_cd = np.asarray(section_pair["delta"]["CDf"], dtype=float)
    if cl.shape != cd.shape or cl.shape != delta_cl.shape or cl.shape != delta_cd.shape:
        raise ValueError("baseline and LDVM histories must have identical shapes")
    if not np.isfinite(finite_wing_gain) or not 0.0 <= finite_wing_gain <= 1.0:
        raise ValueError("finite_wing_gain must lie in [0, 1]")
    return {
        "CL": cl + finite_wing_gain * delta_cl,
        "CD": cd + finite_wing_gain * delta_cd,
        "delta_CL": finite_wing_gain * delta_cl,
        "delta_CD": finite_wing_gain * delta_cd,
    }


def project_ldvm_delta_to_finite_wing(
    delta_cnc: np.ndarray,
    delta_cnnc: np.ndarray,
    delta_cn_nonl: np.ndarray,
    delta_cs: np.ndarray,
    alpha_rad: np.ndarray,
    *,
    aspect_ratio: float,
) -> dict[str, np.ndarray | float]:
    """Project a component-resolved 2-D LDVM increment to a finite wing.

    The finite-wing lift-slope ratio is ``g=1/(1+2/AR)``.  Normal circulation
    scales with ``g``.  The non-circulatory term uses the Izraelevitz finite-
    span added-mass factor interpolated from their AR=3 and AR=6 values.  The
    wake--bound nonlinear term and Ramesh's axial suction are quadratic in
    circulation and therefore scale with ``g**2``.  Keeping these four ledgers
    separate prevents the earlier, inconsistent scaling of the complete
    normal-force increment with one gain.
    """

    cnc = np.asarray(delta_cnc, dtype=float)
    cnnc = np.asarray(delta_cnnc, dtype=float)
    cn_nonl = np.asarray(delta_cn_nonl, dtype=float)
    cs = np.asarray(delta_cs, dtype=float)
    alpha = np.asarray(alpha_rad, dtype=float)
    if not (cnc.shape == cnnc.shape == cn_nonl.shape == cs.shape == alpha.shape):
        raise ValueError("all LDVM component histories must have identical shapes")
    if np.any(~np.isfinite(cnc + cnnc + cn_nonl + cs + alpha)):
        raise FloatingPointError("finite-wing projection input is non-finite")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be finite and positive")
    gain = 1.0 / (1.0 + 2.0 / aspect_ratio)
    added_mass_gain = float(np.clip(0.75 + aspect_ratio / 30.0, 0.0, 1.0))
    projected_cnc = gain * cnc
    projected_cnnc = added_mass_gain * cnnc
    projected_cn_nonl = gain**2 * cn_nonl
    projected_cs = gain**2 * cs
    projected_cn = projected_cnc + projected_cnnc + projected_cn_nonl
    delta_cl = projected_cn * np.cos(alpha) + projected_cs * np.sin(alpha)
    delta_cd = projected_cn * np.sin(alpha) - projected_cs * np.cos(alpha)
    return {
        "delta_CL": delta_cl,
        "delta_CD": delta_cd,
        "projected_delta_CNc": projected_cnc,
        "projected_delta_CNnc": projected_cnnc,
        "projected_delta_CNnonl": projected_cn_nonl,
        "projected_delta_CS": projected_cs,
        "normal_gain": float(gain),
        "added_mass_gain": added_mass_gain,
        "nonlinear_normal_gain": float(gain**2),
        "axial_suction_gain": float(gain**2),
    }
