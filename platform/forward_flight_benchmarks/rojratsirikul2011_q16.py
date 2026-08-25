"""Frozen Rojratsirikul et al. (2011) membrane-wing inputs for the Q16 FSI
reproduction.

Source of truth is the authors' public PDF (hash frozen below).  Every value
here is either (a) printed in the paper, (b) a ``digitized_approx`` figure
reading, or (c) an explicitly labelled assumption for a parameter the paper
does not report.  Nothing in this adapter is tuned against observed responses.

This is a *fixed membrane wing* case: constant freestream, constant geometric
angle of attack, four-edge clamped latex membrane, no prescribed structural
motion of any kind.  The mean camber, vibration amplitude and frequencies are
FSI outputs, never inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fluxvortex.q16_ancf_mesh import (
    Q16MITC16EASMesh,
    Q16Mesh,
    make_rectangular_q16_mesh,
)
from fluxvortex.q16_boundary_constraints import (
    Q16BoundaryConstraints,
    make_clamped_q16_nodes,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PDF = (
    ROOT
    / "artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/references/"
    "Rojratsirikul2011_JFS.pdf"
)
REFERENCE_PDF_URL = "https://purehost.bath.ac.uk/ws/files/227159/Gursul_JFS_2011.pdf"
REFERENCE_PDF_SHA256 = (
    "c9d8f59b4fefafd846fae77fdda6376424b70032db6ae6c40f1f28d51aa9a6a4"
)

# Numerical-reproduction protocol (not experimental inputs from the paper).
FORMAL_Q16_GRID = (5, 10)
FORMAL_AERO_GRID = (15, 30)


def validate_rojratsirikul2011_sources() -> None:
    if not REFERENCE_PDF.is_file():
        raise FileNotFoundError(REFERENCE_PDF)
    observed = hashlib.sha256(REFERENCE_PDF.read_bytes()).hexdigest()
    if observed != REFERENCE_PDF_SHA256:
        raise RuntimeError(f"Rojratsirikul 2011 source drift: {observed}")


@dataclass(frozen=True, slots=True)
class Rojratsirikul2011MembraneCase:
    """One fixed-alpha membrane-wing case of the 2011 paper (shared physics)."""

    case_id: str
    angle_deg: float
    purpose: str
    # Figure-read response oracles (digitized, NOT author-reported tables).
    digitized_approx_zmax_over_c: float
    digitized_approx_cn_low: float
    digitized_approx_cn_high: float
    digitized_approx_strouhal: float | None = None
    digitized_approx_chordwise_peak_count: int | None = None
    digitized_approx_spanwise_peak_count: int | None = None
    # Paper-printed geometry / material truth (Rojratsirikul et al. 2011).
    paper_title: str = (
        "Flow-induced vibrations of low aspect ratio rectangular membrane wings"
    )
    doi: str = "10.1016/j.jfluidstructs.2011.06.007"
    chord_m: float = 0.0688
    span_m: float = 0.1375
    thickness_m: float = 2.0e-4
    young_modulus_pa: float = 2.2e6
    membrane_density_kg_m3: float = 1000.0
    freestream_m_s: float = 5.0
    # Cross-fit of the paper's three (U, Re) printed pairs; the paper itself
    # never prints nu.  Gives Re = 24,324 vs the paper's displayed 24,300.
    kinematic_viscosity_m2_s: float = 1.4142e-5
    # Cross-fit of the paper's three (U, Pi1) printed pairs; reproduces the
    # printed Pi1 = 7.51 at U = 5 m/s exactly to the displayed digits.
    fluid_density_kg_m3: float = 1.208
    # Declared assumptions for parameters the paper does not report.
    poisson_ratio_assumed: float = 0.49
    prestress_n_m_assumed: float = 0.0
    # Latex viscoelastic loss factor from materials literature (DMA:
    # eta ~ 0.05-0.15 for natural rubber above Tg at 1-100 Hz; see
    # research/CLAIM_TREE.md node N4).  Kelvin-Voigt theta = eta/omega with
    # omega taken at the physical lock-in frequency St~1 (f = U/c ~ 72.7 Hz),
    # so the model dissipates eta=0.1 at the frequency that matters.  This
    # replaces the earlier frozen-zero assumption with a literature-derived
    # physical value; the zero-damping run is preserved as the primary
    # artifact and eta in {0.05, 0.1, 0.15} are sensitivity branches.
    structural_damping_loss_factor: float = 0.1
    # Frozen numerical clock protocol (t* = t * U_inf / c).
    aerodynamic_dt_star: float = 0.01
    # Time-convergence frozen at 10 substeps (dt_s = 1.376e-5 s, ~725
    # samples per cycle at 100 Hz): matched-window comparison against the
    # 50-substep run agrees to <=5e-4 relative on Cn (six decimals on
    # camber) over the shared 20-step window, at 2.7x lower cost.
    structural_substeps_per_aerodynamic_step: int = 10
    startup_time_star: float = 1.0
    statistics_min_time_star: float = 20.0
    statistics_start_time_star: float = 2.0
    lesp_crit: float = 0.11

    @property
    def aspect_ratio(self) -> float:
        """b/c from the figure-2 dimensions (1.9985; the paper text rounds
        this to AR = 2 — the printed lengths are the frozen truth)."""

        return self.span_m / self.chord_m

    @property
    def reference_area_m2(self) -> float:
        return self.chord_m * self.span_m

    @property
    def thickness_ratio(self) -> float:
        return self.thickness_m / self.chord_m

    @property
    def reynolds(self) -> float:
        return self.freestream_m_s * self.chord_m / self.kinematic_viscosity_m2_s

    @property
    def dynamic_pressure_pa(self) -> float:
        return 0.5 * self.fluid_density_kg_m3 * self.freestream_m_s**2

    @property
    def pi1(self) -> float:
        """Paper aeroelastic parameter Pi1 = (E t / (q c))^(1/3)."""

        return (
            self.young_modulus_pa
            * self.thickness_m
            / (self.dynamic_pressure_pa * self.chord_m)
        ) ** (1.0 / 3.0)

    @property
    def mass_ratio(self) -> float:
        """rho_m t / (rho_air c): structural-to-added-mass areal ratio."""

        return (
            self.membrane_density_kg_m3
            * self.thickness_m
            / (self.fluid_density_kg_m3 * self.chord_m)
        )

    @property
    def aerodynamic_dt_s(self) -> float:
        return self.aerodynamic_dt_star * self.chord_m / self.freestream_m_s

    @property
    def structural_dt_s(self) -> float:
        return self.aerodynamic_dt_s / self.structural_substeps_per_aerodynamic_step

    @property
    def convective_time_s(self) -> float:
        return self.chord_m / self.freestream_m_s

    def freestream_factor(self, time_star: float) -> float:
        """Frozen half-cosine startup ramp 0 -> 1 over t* in [0, 1]."""

        time_star = float(time_star)
        if not math.isfinite(time_star) or time_star < 0.0:
            raise ValueError("time_star must be finite and non-negative")
        if time_star >= self.startup_time_star:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * time_star / self.startup_time_star))

    def config_digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ROJ11_A10 = Rojratsirikul2011MembraneCase(
    case_id="ROJ11-A10",
    angle_deg=10.0,
    purpose="low/mid-angle 3-D LEV / tip-vortex / membrane coupling",
    digitized_approx_zmax_over_c=0.032,
    digitized_approx_cn_low=0.50,
    digitized_approx_cn_high=0.52,
    digitized_approx_strouhal=1.10,
    digitized_approx_chordwise_peak_count=3,
    digitized_approx_spanwise_peak_count=3,
)
ROJ11_A16 = Rojratsirikul2011MembraneCase(
    case_id="ROJ11-A16",
    angle_deg=16.0,
    purpose="primary accuracy case: clear separation, large mean camber",
    digitized_approx_zmax_over_c=0.043,
    digitized_approx_cn_low=0.92,
    digitized_approx_cn_high=0.95,
)
ROJ11_A23 = Rojratsirikul2011MembraneCase(
    case_id="ROJ11-A23",
    angle_deg=23.0,
    purpose="deep separation, chordwise second-mode dominance",
    digitized_approx_zmax_over_c=0.0475,
    digitized_approx_cn_low=0.98,
    digitized_approx_cn_high=1.02,
    digitized_approx_strouhal=0.83,
    digitized_approx_chordwise_peak_count=2,
)

ROJRATSIRIKUL2011_CASES: dict[str, Rojratsirikul2011MembraneCase] = {
    case.case_id: case for case in (ROJ11_A10, ROJ11_A16, ROJ11_A23)
}


def static_motion_contract(case: Rojratsirikul2011MembraneCase) -> dict[str, Any]:
    """The external-motion contract: everything rigid and constant."""

    if type(case) is not Rojratsirikul2011MembraneCase:
        raise TypeError("case must be an exact Rojratsirikul2011MembraneCase")
    return {
        "U_inf_history": f"U0={case.freestream_m_s:g} m/s constant along +x (frozen "
        "half-cosine ramp only over t*<="
        f"{case.startup_time_star:g})",
        "alpha_history": (
            f"alpha0={case.angle_deg:g} deg constant geometric angle, realized "
            "by rigidly pitching the reference membrane nose-up about its "
            "leading-edge line (the oracle-verified native-path incidence "
            "mechanism); the freestream vector is never tilted"
        ),
        "frame_pose": "frame_pose(0) rigid and fixed for all t",
        "frame_velocity_m_s": 0.0,
        "prescribed_structural_forces": None,
        "boundary": "four-edge six-DOF clamped at rotated reference values",
        "flapping_heave_pitch_laws": "forbidden for this case",
    }


def perimeter_node_ids(
    mesh: Q16Mesh, case: Rojratsirikul2011MembraneCase
) -> np.ndarray:
    """Sorted unique node ids on the four membrane edges (x=0, x=c, y=0, y=b)."""

    if type(mesh) is not Q16Mesh:
        raise TypeError("mesh must be an exact Q16Mesh")
    if type(case) is not Rojratsirikul2011MembraneCase:
        raise TypeError("case must be an exact Rojratsirikul2011MembraneCase")
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(case.chord_m, case.span_m)
    )
    x = mesh.reference_rows[:, 0]
    y = mesh.reference_rows[:, 1]
    on_edge = (
        (np.abs(x) <= tolerance)
        | (np.abs(x - case.chord_m) <= tolerance)
        | (np.abs(y) <= tolerance)
        | (np.abs(y - case.span_m) <= tolerance)
    )
    return np.ascontiguousarray(np.flatnonzero(on_edge), dtype=np.int64)


def validate_perimeter(
    mesh: Q16Mesh,
    case: Rojratsirikul2011MembraneCase,
    chordwise_element_count: int,
    spanwise_element_count: int,
) -> dict[str, Any]:
    """Audit the four-edge clamping set: covers all edges, no interior nodes."""

    nodes = perimeter_node_ids(mesh, case)
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(case.chord_m, case.span_m)
    )
    x = mesh.reference_rows[:, 0]
    y = mesh.reference_rows[:, 1]
    # Structured-lattice perimeter count: 2(3nc+1) + 2(3ns+1) - 4 corners.
    expected = (
        2 * (3 * chordwise_element_count + 1)
        + 2 * (3 * spanwise_element_count + 1)
        - 4
    )
    if nodes.size != expected:
        raise RuntimeError(
            f"perimeter node count {nodes.size} != lattice expectation {expected}"
        )
    edges = {
        "leading": np.count_nonzero(np.abs(x) <= tolerance),
        "trailing": np.count_nonzero(np.abs(x - case.chord_m) <= tolerance),
        "side": np.count_nonzero(np.abs(y) <= tolerance),
        "side_opposite": np.count_nonzero(np.abs(y - case.span_m) <= tolerance),
    }
    chord_nodes = 3 * chordwise_element_count + 1
    span_nodes = 3 * spanwise_element_count + 1
    # Leading/trailing edges (x = 0 / x = c) run spanwise; side edges run
    # chordwise.
    if edges["leading"] != span_nodes or edges["trailing"] != span_nodes:
        raise RuntimeError(f"chordwise edge coverage drift: {edges}")
    if edges["side"] != chord_nodes or edges["side_opposite"] != chord_nodes:
        raise RuntimeError(f"spanwise edge coverage drift: {edges}")
    interior = np.setdiff1d(np.arange(mesh.node_count, dtype=np.int64), nodes)
    if bool(
        (
            (x[interior] <= tolerance)
            | (np.abs(x[interior] - case.chord_m) <= tolerance)
            | (y[interior] <= tolerance)
            | (np.abs(y[interior] - case.span_m) <= tolerance)
        ).any()
    ):
        raise RuntimeError("an interior node was classified as perimeter")
    return {
        "perimeter_node_count": int(nodes.size),
        "interior_node_count": int(interior.size),
        "edge_node_counts": {name: int(value) for name, value in edges.items()},
        "clamped_dof_count": int(6 * nodes.size),
        "free_dof_count": int(6 * interior.size),
    }


def plate_normal(case: Rojratsirikul2011MembraneCase) -> np.ndarray:
    """Unit normal of the tilted reference membrane (+z rotated nose-up)."""

    if type(case) is not Rojratsirikul2011MembraneCase:
        raise TypeError("case must be an exact Rojratsirikul2011MembraneCase")
    alpha = math.radians(case.angle_deg)
    return np.array([math.sin(alpha), 0.0, math.cos(alpha)], dtype=np.float64)


def rotate_q16_mesh_about_leading_edge(
    mesh: Q16Mesh, case: Rojratsirikul2011MembraneCase
) -> Q16Mesh:
    """Rigidly pitch the planar membrane nose-up about its leading-edge line.

    This is the incidence mechanism of every oracle-verified native-path case
    (Goland: ``z = -x sin(alpha)``, flow stays +x).  The freestream vector is
    never tilted; positive alpha means the trailing edge goes DOWN and the
    flow approaches the pressure side from below.
    """

    if type(mesh) is not Q16Mesh:
        raise TypeError("mesh must be an exact Q16Mesh")
    if type(case) is not Rojratsirikul2011MembraneCase:
        raise TypeError("case must be an exact Rojratsirikul2011MembraneCase")
    alpha = math.radians(case.angle_deg)
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    rows = np.array(mesh.reference_rows, dtype=np.float64, copy=True)
    x, z = rows[:, 0].copy(), rows[:, 2].copy()
    gx, gz = rows[:, 3].copy(), rows[:, 5].copy()
    rows[:, 0] = x * cos_a + z * sin_a
    rows[:, 2] = -x * sin_a + z * cos_a
    rows[:, 3] = gx * cos_a + gz * sin_a
    rows[:, 5] = -gx * sin_a + gz * cos_a
    return Q16Mesh(
        reference_rows=np.ascontiguousarray(rows, dtype=np.float64),
        connectivity=mesh.connectivity,
    )


def make_rojratsirikul2011_q16_model(
    *,
    chordwise_element_count: int = FORMAL_Q16_GRID[0],
    spanwise_element_count: int = FORMAL_Q16_GRID[1],
    case: Rojratsirikul2011MembraneCase,
) -> tuple[Q16Mesh, Q16MITC16EASMesh, Q16BoundaryConstraints, dict[str, Any]]:
    """Build the four-edge clamped membrane pitched nose-up by the case angle.

    Returns (rotated mesh, model, constraints, perimeter audit).  The
    perimeter set is selected on the unrotated lattice (rotation-invariant
    node ids) and clamps all six DOFs at the rotated reference values.
    """

    if type(case) is not Rojratsirikul2011MembraneCase:
        raise TypeError("case must be an exact Rojratsirikul2011MembraneCase")
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=chordwise_element_count,
        spanwise_element_count=spanwise_element_count,
        chord=case.chord_m,
        span=case.span_m,
        thickness=case.thickness_m,
    )
    audit = validate_perimeter(
        mesh, case, chordwise_element_count, spanwise_element_count
    )
    perimeter = perimeter_node_ids(mesh, case)
    mesh = rotate_q16_mesh_about_leading_edge(mesh, case)
    model = Q16MITC16EASMesh(
        mesh,
        young_modulus=case.young_modulus_pa,
        poisson_ratio=case.poisson_ratio_assumed,
        density=case.membrane_density_kg_m3,
    )
    constraints = make_clamped_q16_nodes(mesh, perimeter)
    return mesh, model, constraints, audit


def normal_force_coefficient(
    total_force_n: np.ndarray | list[float],
    case: Rojratsirikul2011MembraneCase,
    normal: np.ndarray | None = None,
) -> float:
    """Cn = F . n_hat / (q_inf S) with n_hat the chord-plane normal (+z side).

    For a pitched membrane pass ``normal=plate_normal(case)``; the default
    keeps the historical flat-membrane +z convention.
    """

    values = np.asarray(total_force_n, dtype=np.float64)
    if values.shape != (3,) or not bool(np.isfinite(values).all()):
        raise ValueError("total_force_n must be a finite 3-vector")
    direction = plate_normal(case) if normal is None else np.asarray(normal, dtype=np.float64)
    if direction.shape != (3,) or not math.isfinite(float(np.linalg.norm(direction))):
        raise ValueError("normal must be a finite 3-vector")
    return float(
        values @ direction / (case.dynamic_pressure_pa * case.reference_area_m2)
    )


def count_interior_peaks(profile: np.ndarray, relative_threshold: float = 0.15) -> int:
    """Count interior local maxima above a fraction of the profile maximum."""

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("profile must be a 1-D vector of length >= 3")
    if not bool(np.isfinite(values).all()):
        raise ValueError("profile must be finite")
    threshold = relative_threshold * float(values.max())
    interior = (
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] > threshold)
    )
    return int(np.count_nonzero(interior))


def dominant_strouhal(
    series: np.ndarray,
    aerodynamic_dt_s: float,
    case: Rojratsirikul2011MembraneCase,
) -> dict[str, Any] | None:
    """Hann-windowed rFFT peak -> Strouhal St = f c / U_inf (None if short)."""

    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 64:
        return None
    if not bool(np.isfinite(values).all()):
        raise ValueError("series must be finite")
    if not math.isfinite(aerodynamic_dt_s) or aerodynamic_dt_s <= 0.0:
        raise ValueError("aerodynamic_dt_s must be finite and positive")
    window = np.hanning(values.size)
    spectrum = np.abs(np.fft.rfft((values - values.mean()) * window))
    frequencies = np.fft.rfftfreq(values.size, d=aerodynamic_dt_s)
    peak = 1 + int(np.argmax(spectrum[1:]))
    frequency_hz = float(frequencies[peak])
    return {
        "strouhal": frequency_hz * case.chord_m / case.freestream_m_s,
        "frequency_hz": frequency_hz,
        "sample_count": int(values.size),
        "window": "hann",
    }


def membrane_statistics(
    z_history: np.ndarray,
    aerodynamic_dt_s: float,
    case: Rojratsirikul2011MembraneCase,
) -> dict[str, Any]:
    """Mean camber, fluctuation (zsd) map, peak counts and dominant St.

    ``z_history`` is the (T, chordwise, spanwise) quarter-grid membrane height
    sampled once per committed aerodynamic step, already restricted to the
    post-startup statistics window.
    """

    history = np.asarray(z_history, dtype=np.float64)
    if history.ndim != 3 or history.shape[0] < 8:
        raise ValueError("z_history must be (T, nc, ns) with T >= 8")
    if not bool(np.isfinite(history).all()):
        raise ValueError("z_history must be finite")
    mean_map = history.mean(axis=0)
    zsd_map = history.std(axis=0)
    chord_index, span_index = np.unravel_index(
        int(np.argmax(zsd_map)), zsd_map.shape
    )
    chordwise_profile = zsd_map[:, span_index]
    spanwise_profile = zsd_map[chord_index, :]
    fluctuation = dominant_strouhal(
        history[:, chord_index, span_index], aerodynamic_dt_s, case
    )
    return {
        "statistics_sample_count": int(history.shape[0]),
        "mean_zmax_over_c": float(mean_map.max() / case.chord_m),
        "mean_map_over_c": (mean_map / case.chord_m).tolist(),
        "zsd_map_over_c": (zsd_map / case.chord_m).tolist(),
        "zsd_max_over_c": float(zsd_map.max() / case.chord_m),
        "zsd_max_station": {
            "chordwise_index": int(chord_index),
            "spanwise_index": int(span_index),
        },
        "chordwise_peak_count": count_interior_peaks(chordwise_profile),
        "spanwise_peak_count": count_interior_peaks(spanwise_profile),
        "fluctuation_spectrum": fluctuation,
    }


def assumption_ledger(case: Rojratsirikul2011MembraneCase) -> dict[str, Any]:
    """Ledger of parameters the paper does not report, with frozen baselines."""

    if type(case) is not Rojratsirikul2011MembraneCase:
        raise TypeError("case must be an exact Rojratsirikul2011MembraneCase")
    return {
        "poisson_ratio": {
            "paper_status": "not reported",
            "frozen_value": case.poisson_ratio_assumed,
            "basis": "declared near-incompressible latex assumption",
            "discipline": "must not be tuned against responses",
        },
        "initial_prestress_n_m": {
            "paper_status": "not reported",
            "frozen_value": case.prestress_n_m_assumed,
            "basis": "zero",
            "discipline": "no adding prestress after seeing under-deformation",
        },
        "initial_slack_or_excess": {
            "paper_status": "not reported",
            "frozen_value": 0.0,
            "basis": "perfectly taut planar membrane",
        },
        "structural_damping": {
            "paper_status": "not reported",
            "frozen_value": case.structural_damping_loss_factor,
            "basis": "literature latex viscoelastic loss factor "
            "eta~0.05-0.15 (DMA literature; experiment membrane is real "
            "latex), applied as Kelvin-Voigt theta*K at the St~1 lock-in "
            "frequency; the earlier zero assumption is rewritten per "
            "research/CLAIM_TREE.md N4 with independent-source evidence, "
            "not fitted against responses",
            "discipline": "sensitivity branches eta in {0.05, 0.1, 0.15}; "
            "zero-damping run preserved as the primary artifact",
        },
        "initial_geometric_imperfection": {
            "paper_status": "not reported",
            "frozen_value": "perfectly flat",
            "basis": "symmetric bifurcation issues get machine-scale "
            "reproducible perturbation, reported separately",
        },
        "latex_constitutive_curve": {
            "paper_status": "only E = 2.2 MPa reported",
            "frozen_value": "isotropic geometrically nonlinear Q16 baseline",
            "discipline": "no hiding hyperelasticity inside a fitted E",
        },
        "boundary_director_condition": {
            "paper_status": "not reported",
            "frozen_value": "four-edge six-DOF clamped (positions + directors)",
            "discipline": "translation-only relaxation requires independent "
            "oracle evidence first",
        },
        "frame_aerodynamic_thickness": {
            "paper_status": "only figure-2 cross-section given",
            "frozen_value": "fixed boundary, full c x b outer mould line",
            "discipline": "explicit frame is a separate later branch",
        },
        "inlet_turbulence_transition": {
            "paper_status": "not reported",
            "frozen_value": "none; no empirical turbulence tuning",
        },
    }


__all__ = [
    "FORMAL_AERO_GRID",
    "FORMAL_Q16_GRID",
    "REFERENCE_PDF",
    "REFERENCE_PDF_SHA256",
    "REFERENCE_PDF_URL",
    "ROJ11_A10",
    "ROJ11_A16",
    "ROJ11_A23",
    "ROJRATSIRIKUL2011_CASES",
    "Rojratsirikul2011MembraneCase",
    "assumption_ledger",
    "count_interior_peaks",
    "dominant_strouhal",
    "make_rojratsirikul2011_q16_model",
    "membrane_statistics",
    "normal_force_coefficient",
    "perimeter_node_ids",
    "plate_normal",
    "rotate_q16_mesh_about_leading_edge",
    "static_motion_contract",
    "validate_perimeter",
    "validate_rojratsirikul2011_sources",
]
