"""Fail-closed S3ai-v2.2 reachable-pressure obstruction runner.

The executable definition is the hash-verified v2 -> v2.1 -> v2.2 addendum
chain.  Nothing executes at import time and this module never writes a result
artifact.  The v2.2 contract explicitly sets ``formal_execution_allowed`` to
false, so the public runner fails before constructing a mesh.

Every history constructs a fresh body mesh and classified topology, advances
the unmodified S3e material wake from ``[-dt, 1]``, and observes only stored
states.  The primary residual uses the face-mu material trace and the stored
body potential.  The independent direct solve changes only the pressure
observer and never changes the material derivative or the propagated march.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actual_boundary_3d_cut_wake_junction_guard import (  # noqa: E402
    build_canonical_diamond_wing,
)
from claim_runtime.actual_boundary_body_wake import (  # noqa: E402
    MaterialWakeCutAttachment,
)
from claim_runtime.actual_wake_reachable_pressure import (  # noqa: E402
    DirectIndependentStageObservation,
    centered_tangent,
    observe_direct_independent_stage,
    weak_pressure_step_residual,
)
from claim_runtime.actual_wake_kutta_closure_roles import (  # noqa: E402
    cut_role_operators,
)
from claim_runtime.classified_p2_cut_topology import (  # noqa: E402
    classified_p2_cut_topology,
)
from claim_runtime.material_attachment_inventory import (  # noqa: E402
    MaterialAttachmentInventory,
    material_inventory_increment,
    observe_material_attachment_inventory,
)
from claim_runtime.material_wake_time_march import (  # noqa: E402
    ExplicitMidpointWakeMarch,
    march_actual_boundary_material_wake_explicit_midpoint,
)
from claim_runtime.reachable_pressure_cases import (  # noqa: E402
    HistoryCase,
    frozen_execution_accounting,
    frozen_history_cases,
    frozen_tangent_configurations,
    mixed_cube_configuration_names,
)
from claim_runtime.reachable_pressure_symmetry import (  # noqa: E402
    parity_quadrature_decision,
    project_active_parity,
    span_parity_operators,
    stagewise_odd_noncancellation,
    typed_mass_norm,
)
from claim_runtime.reachable_pressure_uncertainty import (  # noqa: E402
    adjacent_mixed_cube,
    contracting_quadrature_tail,
    dual_mass_norm,
    floating_plateau_floor,
    second_order_richardson,
)


BASE_CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_cases_20260728_131922.yaml"
)
V21_CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_cases_20260728_133218.yaml"
)
CASES = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_reachable_pressure_obstruction_cases_20260728_134229.yaml"
)
PRESTEP_AUDIT = (
    HERE
    / "docs"
    / "diag"
    / "actual_wake_reachable_prestep_symmetry_diagnostic_20260728_133553.json"
)
_FROZEN_HASHES = {
    BASE_CASES.name: (
        "8345035356d300d4154ac276fc54b8ab27b6e83704cb4444436dbc0c2c59c75b"
    ),
    V21_CASES.name: (
        "e751381942ec7c0cac8ea055c6aad1a5756643b85992bc04f4bee88339b563b4"
    ),
    CASES.name: (
        "3d662a69c1da80a1452b6b05c67107b188070871bb1b594977467ab7384e0b27"
    ),
    PRESTEP_AUDIT.name: (
        "f10e6ce2e31aea570b6b07d1607b7cdebe5336dd7e9700845596e2338318b4f2"
    ),
}

_EXPECTED_ACCOUNTING = {
    "histories": 31,
    "measurement_steps": 380,
    "compatible_presteps": 31,
    "marcher_steps": 411,
    "half_full_solves": 822,
    "observed_stages": 791,
}


class ReachablePressureV2GuardError(RuntimeError):
    """The frozen S3ai-v2.2 definition or one observation is invalid."""


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_active_mass(value: Any) -> np.ndarray:
    """Return the frozen 7x7 bit-exact symmetric SPD/persymmetric mass."""
    try:
        return span_parity_operators(value).mass
    except ValueError as error:
        raise ReachablePressureV2GuardError(
            f"invalid frozen active mass: {error}"
        ) from error


@dataclass(frozen=True)
class TypedStageArrays:
    """Parity-ready primal/dual arrays in their declared trace spaces."""

    stage_times: np.ndarray
    stage_roles: tuple[str, ...]
    canonical_material_current_trace: np.ndarray
    body_cut_trace: np.ndarray
    canonical_material_release: np.ndarray
    representation_inventory: np.ndarray
    stored_weak_pressure: np.ndarray
    direct_weak_pressure: np.ndarray
    measurement_times: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.stage_times, dtype=float)
        roles = tuple(self.stage_roles)
        primal = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.canonical_material_current_trace,
                self.body_cut_trace,
                self.canonical_material_release,
                self.representation_inventory,
            )
        )
        dual = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.stored_weak_pressure,
                self.direct_weak_pressure,
            )
        )
        measurement = np.asarray(self.measurement_times, dtype=float)
        stage_count = len(times)
        if (
            times.shape != (stage_count,)
            or len(roles) != stage_count
            or any(value.shape != (stage_count, 9) for value in primal)
            or any(value.shape != (stage_count, 7) for value in dual)
            or measurement.ndim != 2
            or measurement.shape[1:] != (3,)
            or not all(roles)
            or not all(
                np.all(np.isfinite(value))
                for value in (times, measurement, *primal, *dual)
            )
        ):
            raise ReachablePressureV2GuardError(
                "typed stage arrays have inconsistent primal/dual spaces"
            )
        if stage_count != 1 + 2 * len(measurement):
            raise ReachablePressureV2GuardError(
                "typed stage arrays must contain prestep full plus each "
                "measured half/full stage"
            )
        if roles[0] != "entrance_prestep_full":
            raise ReachablePressureV2GuardError(
                "typed stage arrays must start with entrance_prestep_full"
            )
        object.__setattr__(self, "stage_times", times.copy())
        object.__setattr__(self, "stage_roles", roles)
        for name, value in zip(
            (
                "canonical_material_current_trace",
                "body_cut_trace",
                "canonical_material_release",
                "representation_inventory",
            ),
            primal,
            strict=True,
        ):
            object.__setattr__(self, name, value.copy())
        for name, value in zip(
            ("stored_weak_pressure", "direct_weak_pressure"),
            dual,
            strict=True,
        ):
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "measurement_times", measurement.copy())


@dataclass(frozen=True)
class CompatibleHistoryObservation:
    """One fresh compatible-prestep history in the fixed active-P2 space."""

    case: HistoryCase
    stored_window_residual: np.ndarray
    direct_window_residual: np.ndarray
    stored_step_residuals: np.ndarray
    direct_step_residuals: np.ndarray
    mass_active: np.ndarray
    stage_arrays: TypedStageArrays
    diagnostics: Mapping[str, float | int]
    checks: Mapping[str, bool]
    observed_stage_count: int

    def __post_init__(self) -> None:
        stored = np.asarray(self.stored_window_residual, dtype=float)
        direct = np.asarray(self.direct_window_residual, dtype=float)
        stored_steps = np.asarray(self.stored_step_residuals, dtype=float)
        direct_steps = np.asarray(self.direct_step_residuals, dtype=float)
        mass = _strict_active_mass(self.mass_active)
        size = stored.shape[0] if stored.ndim == 1 else -1
        if (
            size != 7
            or direct.shape != (size,)
            or stored_steps.shape != (self.case.measurement_steps, size)
            or direct_steps.shape != stored_steps.shape
            or not all(
                np.all(np.isfinite(value))
                for value in (
                    stored,
                    direct,
                    stored_steps,
                    direct_steps,
                    mass,
                )
            )
        ):
            raise ReachablePressureV2GuardError(
                f"history {self.case.name} has inconsistent residual spaces"
            )
        if (
            not np.array_equal(stored, np.sum(stored_steps, axis=0))
            or not np.array_equal(direct, np.sum(direct_steps, axis=0))
        ):
            raise ReachablePressureV2GuardError(
                f"history {self.case.name} window residual is not its exact "
                "step sum"
            )
        if int(self.observed_stage_count) != self.case.observed_stages:
            raise ReachablePressureV2GuardError(
                f"history {self.case.name} observed-stage accounting drifted"
            )
        if not self.diagnostics or not self.checks:
            raise ReachablePressureV2GuardError(
                f"history {self.case.name} lacks guard diagnostics"
            )
        if (
            not isinstance(self.stage_arrays, TypedStageArrays)
            or len(self.stage_arrays.measurement_times)
            != self.case.measurement_steps
            or len(self.stage_arrays.stage_times)
            != self.observed_stage_count
        ):
            raise ReachablePressureV2GuardError(
                f"history {self.case.name} typed stage accounting drifted"
            )
        object.__setattr__(self, "stored_window_residual", stored.copy())
        object.__setattr__(self, "direct_window_residual", direct.copy())
        object.__setattr__(self, "stored_step_residuals", stored_steps.copy())
        object.__setattr__(self, "direct_step_residuals", direct_steps.copy())
        object.__setattr__(self, "mass_active", mass.copy())
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "checks", dict(self.checks))

    @property
    def stored_noncancellation(self) -> float:
        return float(
            sum(
                dual_mass_norm(step, self.mass_active)
                for step in self.stored_step_residuals
            )
        )

    @property
    def direct_noncancellation(self) -> float:
        return float(
            sum(
                dual_mass_norm(step, self.mass_active)
                for step in self.direct_step_residuals
            )
        )


def _load_frozen_contract() -> dict[str, Any]:
    """Load and hash-check the immutable v2 -> v2.1 -> v2.2 chain."""

    for path, expected in (
        (BASE_CASES, _FROZEN_HASHES[BASE_CASES.name]),
        (V21_CASES, _FROZEN_HASHES[V21_CASES.name]),
        (CASES, _FROZEN_HASHES[CASES.name]),
        (PRESTEP_AUDIT, _FROZEN_HASHES[PRESTEP_AUDIT.name]),
    ):
        actual = _sha256_path(path)
        if actual != expected:
            raise ReachablePressureV2GuardError(
                f"frozen definition hash drifted for {path.name}: {actual}"
            )
    base = yaml.safe_load(BASE_CASES.read_text(encoding="utf-8"))
    v21 = yaml.safe_load(V21_CASES.read_text(encoding="utf-8"))
    v22 = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    for contract, version in (
        (base, "S3ai-v2"),
        (v21, "S3ai-v2.1"),
        (v22, "S3ai-v2.2"),
    ):
        if not isinstance(contract, dict):
            raise ReachablePressureV2GuardError(
                f"{version} preregistration must be a mapping"
            )
        required = {
            "artifact": (
                "actual_wake_reachable_pressure_obstruction_preregistration"
            ),
            "version": version,
            "stage": (
                "S3ai_reachable_material_history_pressure_obstruction"
            ),
            "status": "preregistered_before_any_formal_execution",
        }
        for key, expected in required.items():
            if contract.get(key) != expected:
                raise ReachablePressureV2GuardError(
                    f"{version} contract field {key!r} drifted"
                )
    v21_link = v21.get("base_protocol", {})
    v22_chain = v22.get("definition_chain", {})
    links = (
        (
            v21_link,
            BASE_CASES.name,
            _FROZEN_HASHES[BASE_CASES.name],
            "v2.1 base",
        ),
        (
            v22_chain.get("base_protocol", {}),
            BASE_CASES.name,
            _FROZEN_HASHES[BASE_CASES.name],
            "v2.2 base",
        ),
        (
            v22_chain.get("executable_addendum_v21", {}),
            V21_CASES.name,
            _FROZEN_HASHES[V21_CASES.name],
            "v2.2 v2.1 addendum",
        ),
        (
            v22_chain.get("implementation_audit", {}),
            PRESTEP_AUDIT.name,
            _FROZEN_HASHES[PRESTEP_AUDIT.name],
            "v2.2 implementation audit",
        ),
    )
    for link, filename, digest, role in links:
        if link.get("file") != filename or link.get("sha256") != digest:
            raise ReachablePressureV2GuardError(
                f"{role} definition-chain link drifted"
            )
    if v22.get("supersedes_for_execution") != [V21_CASES.name]:
        raise ReachablePressureV2GuardError(
            "v2.2 execution supersession link drifted"
        )
    decision = v22.get("decision", {})
    if (
        decision.get("production_activation_allowed") is not False
        or decision.get("formal_execution_allowed") is not False
        or decision.get("no_result_exists_at_preregistration") is not True
    ):
        raise ReachablePressureV2GuardError(
            "v2.2 must remain no-production and formal-execution closed"
        )

    # Base physical/numerical families remain authoritative; the addenda are
    # retained verbatim and replace only their explicitly named definitions.
    effective = copy.deepcopy(base)
    effective["version"] = "S3ai-v2.2"
    effective["role"] = v22["role"]
    effective["decision"] = copy.deepcopy(decision)
    effective["nonclaims"] = copy.deepcopy(v22["nonclaims"])
    effective["fixed_space"] = copy.deepcopy(v21["fixed_space"])
    effective["execution_accounting"] = copy.deepcopy(
        v21["execution_accounting"]
    )
    effective["case_identity"] = copy.deepcopy(v21["case_identity"])
    effective["quadrature_provenance"] = copy.deepcopy(
        v21["quadrature_provenance"]
    )
    effective["uncertainty_correction"] = copy.deepcopy(
        v21["uncertainty_correction"]
    )
    effective["negative_controls_v21"] = copy.deepcopy(
        v21["negative_controls"]
    )
    effective["continuum_zero_initializer"] = copy.deepcopy(
        v22["continuum_zero_initializer"]
    )
    effective["span_reflection"] = copy.deepcopy(
        v22["span_reflection"]
    )
    effective["projected_uncertainty"] = copy.deepcopy(
        v22["projected_uncertainty"]
    )
    effective["manufactured_parity_controls"] = copy.deepcopy(
        v22["manufactured_parity_controls"]
    )
    effective["_definition_chain"] = {
        "S3ai-v2": {
            "file": BASE_CASES.name,
            "sha256": _FROZEN_HASHES[BASE_CASES.name],
        },
        "S3ai-v2.1": {
            "file": V21_CASES.name,
            "sha256": _FROZEN_HASHES[V21_CASES.name],
        },
        "S3ai-v2.2": {
            "file": CASES.name,
            "sha256": _FROZEN_HASHES[CASES.name],
        },
        "implementation_audit": {
            "file": PRESTEP_AUDIT.name,
            "sha256": _FROZEN_HASHES[PRESTEP_AUDIT.name],
        },
    }
    _validated_registry(effective)
    return effective


def _validated_registry(
    contract: Mapping[str, Any],
) -> tuple[HistoryCase, ...]:
    cases = frozen_history_cases()
    accounting = frozen_execution_accounting()
    for key, expected in _EXPECTED_ACCOUNTING.items():
        if accounting.get(key) != expected:
            raise ReachablePressureV2GuardError(
                f"S3ai-v2 {key} accounting drifted: "
                f"{accounting.get(key)!r} != {expected}"
            )

    families = contract.get("families", {})
    expected_axes = {
        "epsilon": [0.01, 0.005, 0.0025],
        "timestep": [0.25, 0.125, 0.0625],
        "body_and_direct_w_quadrature": [8, 10, 12],
    }
    actual_axes = {
        "epsilon": families.get("epsilon", {}).get("values_rad"),
        "timestep": families.get("timestep", {}).get("values"),
        "body_and_direct_w_quadrature": families.get(
            "body_and_direct_w_quadrature", {}
        ).get("values"),
    }
    if actual_axes != expected_axes:
        raise ReachablePressureV2GuardError(
            "S3ai-v2 axis families drifted from the 31-case registry"
        )
    cube = families.get("adjacent_mixed_cube", {})
    if (
        cube.get("epsilon_rad") != [0.0025, 0.005]
        or cube.get("timestep") != [0.0625, 0.125]
        or cube.get("quadrature_order") != [12, 10]
    ):
        raise ReachablePressureV2GuardError(
            "S3ai-v2 mixed cube drifted from the frozen registry"
        )
    return cases


def frozen_registry_report() -> dict[str, Any]:
    """Return the solver-free 31-history accounting for definition audits."""

    contract = _load_frozen_contract()
    cases = _validated_registry(contract)
    return {
        "contract_file": CASES.name,
        "version": contract["version"],
        "definition_chain": contract["_definition_chain"],
        "case_names": [case.name for case in cases],
        "accounting": frozen_execution_accounting(),
    }


def _maximum_abs(value: Any) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=float)), initial=0.0))


def span_project(value: Any, parity: str) -> np.ndarray:
    """Apply the frozen exact-index reversal projector on the last axis."""

    array = np.asarray(value, dtype=float)
    if (
        array.ndim < 1
        or array.shape[-1] not in (7, 9)
        or not np.all(np.isfinite(array))
        or parity not in {"even", "odd"}
    ):
        raise ReachablePressureV2GuardError(
            "span projection requires finite 7/9-vectors and even/odd parity"
        )
    if array.shape[-1] == 7:
        return np.stack(
            [
                project_active_parity(
                    vector,
                    np.eye(7),
                    parity=parity,
                )
                for vector in array.reshape(-1, 7)
            ],
            axis=0,
        ).reshape(array.shape)
    reflected = array[..., ::-1]
    sign = 1.0 if parity == "even" else -1.0
    return 0.5 * (array + sign * reflected)


def primal_mass_norm(value: Any, mass_active: Any) -> float:
    """Return the P2 coefficient-field norm sqrt(g.T M g)."""

    mass = _strict_active_mass(mass_active)
    vector = np.asarray(value, dtype=float)
    if vector.shape == (9,):
        if vector[0] != 0.0 or vector[-1] != 0.0:
            raise ReachablePressureV2GuardError(
                "complete primal trace must have exact zero tip nodes"
            )
        active = vector[1:-1]
    elif vector.shape == (7,):
        active = vector
    else:
        raise ReachablePressureV2GuardError(
            "primal trace must have exact shape (7,) or (9,)"
        )
    if not np.all(np.isfinite(active)):
        raise ReachablePressureV2GuardError(
            "primal trace contains non-finite values"
        )
    square = float(active @ mass @ active)
    if square < -64.0 * np.finfo(float).eps:
        raise ReachablePressureV2GuardError(
            "primal mass norm received a non-positive metric"
        )
    return float(np.sqrt(max(square, 0.0)))


def _typed_norm(
    value: Any,
    mass: np.ndarray,
    space: str,
) -> float:
    if space == "primal":
        return primal_mass_norm(value, mass)
    if space == "dual":
        return typed_mass_norm(value, mass, metric_role="dual")
    raise ReachablePressureV2GuardError(
        "typed norm space must be primal or dual"
    )


def manufactured_parity_control_report(
    mass_active: Any,
) -> dict[str, bool | float]:
    """Evaluate the frozen v2.2 projector controls without any S3e march."""

    mass = _strict_active_mass(mass_active)
    even = np.array((1.0, -2.0, 3.0, 4.0, 3.0, -2.0, 1.0))
    odd = np.array((1.0, -2.0, 3.0, 0.0, -3.0, 2.0, -1.0))
    amplitude = 2.0**-10
    primal = np.concatenate(
        ((0.0,), even + amplitude * odd, (0.0,))
    )
    dual_even = mass @ even
    dual_odd = mass @ odd
    dual = dual_even + amplitude * dual_odd
    primal_even = span_project(primal, "even")
    primal_odd = span_project(primal, "odd")
    dual_even_observed = span_project(dual, "even")
    dual_odd_observed = span_project(dual, "odd")
    primal_total_square = primal_mass_norm(primal, mass) ** 2
    primal_parts_square = (
        primal_mass_norm(primal_even, mass) ** 2
        + primal_mass_norm(primal_odd, mass) ** 2
    )
    dual_total_square = dual_mass_norm(dual, mass) ** 2
    dual_parts_square = (
        dual_mass_norm(dual_even_observed, mass) ** 2
        + dual_mass_norm(dual_odd_observed, mass) ** 2
    )
    tiny_even = (2.0**-40) * even
    tiny_odd = (2.0**-40) * odd
    wrong_permutation = np.arange(7)
    wrong_permutation[[1, 2]] = wrong_permutation[[2, 1]]
    return {
        "primal_even_exact": np.array_equal(
            primal_even, np.concatenate(((0.0,), even, (0.0,)))
        ),
        "primal_odd_exact": np.array_equal(
            primal_odd,
            np.concatenate(((0.0,), amplitude * odd, (0.0,))),
        ),
        "dual_even_exact": np.array_equal(
            dual_even_observed, dual_even
        ),
        "dual_odd_exact": np.array_equal(
            dual_odd_observed, amplitude * dual_odd
        ),
        "exact_zero_preserved": np.array_equal(
            span_project(np.zeros(7), "odd"), np.zeros(7)
        ),
        "tiny_even_has_zero_odd": np.array_equal(
            span_project(tiny_even, "odd"), np.zeros(7)
        ),
        "tiny_odd_is_retained": np.array_equal(
            span_project(tiny_odd, "odd"), tiny_odd
        ),
        "wrong_permutation_rejected": not np.array_equal(
            even[wrong_permutation], even[::-1]
        ),
        "primal_pythagorean": bool(
            np.isclose(
                primal_total_square,
                primal_parts_square,
                rtol=64.0 * np.finfo(float).eps,
                atol=0.0,
            )
        ),
        "dual_pythagorean": bool(
            np.isclose(
                dual_total_square,
                dual_parts_square,
                rtol=64.0 * np.finfo(float).eps,
                atol=0.0,
            )
        ),
        "primal_pythagorean_abs": abs(
            primal_total_square - primal_parts_square
        ),
        "dual_pythagorean_abs": abs(
            dual_total_square - dual_parts_square
        ),
    }


def _fresh_input_snapshot(
    mesh: Any,
    upper: np.ndarray,
    lower: np.ndarray,
    topology: Any,
) -> dict[str, np.ndarray]:
    return {
        "mesh_vertices": mesh.vertices.copy(),
        "mesh_faces": mesh.faces.copy(),
        "upper": np.asarray(upper).copy(),
        "lower": np.asarray(lower).copy(),
        "cut_coordinates": topology.cut_node_coordinates.copy(),
        "cut_vertices": topology.ordered_cut_vertex_indices.copy(),
        "local_to_global": topology.local_to_global.copy(),
    }


def _fresh_input_mutation(
    snapshot: Mapping[str, np.ndarray],
    mesh: Any,
    upper: np.ndarray,
    lower: np.ndarray,
    topology: Any,
) -> float:
    current = {
        "mesh_vertices": mesh.vertices,
        "mesh_faces": mesh.faces,
        "upper": upper,
        "lower": lower,
        "cut_coordinates": topology.cut_node_coordinates,
        "cut_vertices": topology.ordered_cut_vertex_indices,
        "local_to_global": topology.local_to_global,
    }
    return max(
        _maximum_abs(np.asarray(current[name]) - original)
        for name, original in snapshot.items()
    )


def _incident_history(
    time: float,
    *,
    epsilon_signed: float,
    face_count: int,
) -> np.ndarray:
    alpha = (
        0.0
        if time <= 0.0
        else epsilon_signed * np.sin(np.pi * time)
    )
    velocity = np.array((np.cos(alpha), 0.0, np.sin(alpha)))
    return np.repeat(velocity[None, :], face_count, axis=0)


def _new_diagnostics() -> dict[str, float | int]:
    return {
        "stored_bie_backward_error_max": 0.0,
        "direct_bie_backward_error_max": 0.0,
        "condition_times_backward_error_max": 0.0,
        "direct_w_factorization_abs_max": 0.0,
        "direct_w_rank_deficiency_max": 0,
        "stored_material_body_trace_abs_max": 0.0,
        "stored_body_compatibility_abs_max": 0.0,
        "direct_body_compatibility_abs_max": 0.0,
        "zero_tip_abs_max": 0.0,
        "surface_internal_trace_abs_max": 0.0,
        "surface_history_seam_abs_max": 0.0,
        "surface_row_cache_abs_max": 0.0,
        "surface_boundary_duplicate_abs_max": 0.0,
        "material_inventory_increment_abs_max": 0.0,
        "material_inventory_abs_max": 0.0,
        "wrong_birth_inventory_increment_abs_max": 0.0,
        "wrong_attachment_inventory_increment_abs_max": 0.0,
        "wrong_attachment_inventory_abs_max": 0.0,
        "old_state_mutation_abs_max": 0.0,
        "history_time_gap_abs_max": 0.0,
        "history_geometry_gap_abs_max": 0.0,
        "midpoint_identity_abs_max": 0.0,
        "current_attachment_abs_max": 0.0,
        "entrance_prestep_finite_q_trace_abs_max": 0.0,
        "entrance_prestep_inventory_abs_max": 0.0,
        "march_body_quadrature_mismatch_count": 0,
        "direct_w_quadrature_mismatch_count": 0,
        "input_mutation_abs_max": 0.0,
    }


def _accumulate_stage_diagnostics(
    diagnostics: dict[str, float | int],
    observation: DirectIndependentStageObservation,
    *,
    expected_quadrature_order: int,
) -> None:
    diagnostics["stored_bie_backward_error_max"] = max(
        float(diagnostics["stored_bie_backward_error_max"]),
        observation.stored_bie_backward_error,
    )
    diagnostics["direct_bie_backward_error_max"] = max(
        float(diagnostics["direct_bie_backward_error_max"]),
        observation.direct_bie_backward_error,
    )
    diagnostics["condition_times_backward_error_max"] = max(
        float(diagnostics["condition_times_backward_error_max"]),
        observation.stored_condition_times_backward_error,
        observation.direct_condition_times_backward_error,
    )
    diagnostics["direct_w_factorization_abs_max"] = max(
        float(diagnostics["direct_w_factorization_abs_max"]),
        observation.direct_w_factorization_abs_residual,
    )
    diagnostics["direct_w_rank_deficiency_max"] = max(
        int(diagnostics["direct_w_rank_deficiency_max"]),
        observation.direct_w_rank_deficiency,
    )
    diagnostics["stored_material_body_trace_abs_max"] = max(
        float(diagnostics["stored_material_body_trace_abs_max"]),
        observation.stored_material_body_trace_abs_residual,
    )
    diagnostics["stored_body_compatibility_abs_max"] = max(
        float(diagnostics["stored_body_compatibility_abs_max"]),
        observation.stored_compatibility_abs_residual,
    )
    diagnostics["direct_body_compatibility_abs_max"] = max(
        float(diagnostics["direct_body_compatibility_abs_max"]),
        observation.compatibility_abs_residual,
    )
    diagnostics["zero_tip_abs_max"] = max(
        float(diagnostics["zero_tip_abs_max"]),
        observation.zero_tip_abs_residual,
        _maximum_abs(
            observation.canonical_material_trace[[0, -1]]
        ),
    )
    if (
        observation.body_and_direct_w_quadrature_order
        != expected_quadrature_order
    ):
        diagnostics["direct_w_quadrature_mismatch_count"] = (
            int(diagnostics["direct_w_quadrature_mismatch_count"]) + 1
        )
    direct = observation.direct_assembly
    if (
        direct.target_quadrature_order != expected_quadrature_order
        or direct.source_quadrature_order != expected_quadrature_order
    ):
        diagnostics["direct_w_quadrature_mismatch_count"] = (
            int(diagnostics["direct_w_quadrature_mismatch_count"]) + 1
        )


def _accumulate_inventory_diagnostics(
    diagnostics: dict[str, float | int],
    inventory: MaterialAttachmentInventory,
) -> None:
    history = inventory.history
    diagnostics["zero_tip_abs_max"] = max(
        float(diagnostics["zero_tip_abs_max"]),
        _maximum_abs(inventory.body_cut_trace[[0, -1]]),
        _maximum_abs(inventory.canonical_release[[0, -1]]),
        _maximum_abs(inventory.inventory[[0, -1]]),
    )
    diagnostics["surface_internal_trace_abs_max"] = max(
        float(diagnostics["surface_internal_trace_abs_max"]),
        history.maximum_surface_internal_trace_error,
    )
    diagnostics["surface_history_seam_abs_max"] = max(
        float(diagnostics["surface_history_seam_abs_max"]),
        history.maximum_history_seam_error,
    )
    diagnostics["surface_row_cache_abs_max"] = max(
        float(diagnostics["surface_row_cache_abs_max"]),
        history.maximum_row_surface_cache_error,
    )
    diagnostics["history_time_gap_abs_max"] = max(
        float(diagnostics["history_time_gap_abs_max"]),
        history.maximum_time_gap,
    )
    diagnostics["history_geometry_gap_abs_max"] = max(
        float(diagnostics["history_geometry_gap_abs_max"]),
        history.maximum_geometry_gap,
    )
    diagnostics["surface_boundary_duplicate_abs_max"] = max(
        float(diagnostics["surface_boundary_duplicate_abs_max"]),
        max(
            band.boundary_duplicate_abs_error
            for band in history.bands
        ),
    )


def _inventories(
    solution: Any,
    *,
    topology: Any,
    attachment: MaterialWakeCutAttachment,
    wrong_attachment: MaterialWakeCutAttachment,
) -> tuple[
    MaterialAttachmentInventory,
    MaterialAttachmentInventory,
    MaterialAttachmentInventory,
]:
    arguments = {
        "topology": topology,
        "history": solution.wake_history,
        "global_body_potential": solution.global_body_potential,
    }
    correct = observe_material_attachment_inventory(
        **arguments,
        attachment=attachment,
        birth_sign=1,
    )
    wrong_birth = observe_material_attachment_inventory(
        **arguments,
        attachment=attachment,
        birth_sign=-1,
    )
    wrong_orientation = observe_material_attachment_inventory(
        **arguments,
        attachment=wrong_attachment,
        birth_sign=1,
    )
    return correct, wrong_birth, wrong_orientation


def _history_checks(
    diagnostics: Mapping[str, float | int],
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "stored_bie": float(diagnostics["stored_bie_backward_error_max"])
        <= float(thresholds["stored_bie_backward_error_max"]),
        "direct_bie": float(diagnostics["direct_bie_backward_error_max"])
        <= float(thresholds["direct_bie_backward_error_max"]),
        "conditioned_backward_error": float(
            diagnostics["condition_times_backward_error_max"]
        )
        <= float(thresholds["condition_times_backward_error_max"]),
        "direct_w_factorization": float(
            diagnostics["direct_w_factorization_abs_max"]
        )
        <= float(thresholds["direct_w_factorization_abs_max"])
        and int(diagnostics["direct_w_rank_deficiency_max"])
        <= int(thresholds["direct_w_rank_deficiency_max"]),
        "stored_material_body_trace": float(
            diagnostics["stored_material_body_trace_abs_max"]
        )
        <= float(thresholds["stored_material_body_trace_abs_max"]),
        "direct_body_compatibility": float(
            diagnostics["direct_body_compatibility_abs_max"]
        )
        <= float(thresholds["direct_body_compatibility_abs_max"]),
        "zero_tip": float(diagnostics["zero_tip_abs_max"])
        <= float(thresholds["zero_tip_abs_max"]),
        "surface_trace": float(
            diagnostics["surface_internal_trace_abs_max"]
        )
        <= float(thresholds["surface_internal_trace_abs_max"]),
        "surface_history": float(
            diagnostics["surface_history_seam_abs_max"]
        )
        <= float(thresholds["surface_history_seam_abs_max"]),
        "surface_row_cache": float(
            diagnostics["surface_row_cache_abs_max"]
        )
        <= float(thresholds["surface_row_cache_abs_max"]),
        "surface_boundary_duplicate": float(
            diagnostics["surface_boundary_duplicate_abs_max"]
        )
        <= float(thresholds["surface_internal_trace_abs_max"]),
        "material_inventory": float(
            diagnostics["material_inventory_increment_abs_max"]
        )
        <= float(thresholds["material_inventory_increment_abs_max"]),
        "old_state": float(diagnostics["old_state_mutation_abs_max"])
        <= float(thresholds["old_state_mutation_abs_max"]),
        "history_time": float(diagnostics["history_time_gap_abs_max"])
        <= float(thresholds["history_time_gap_abs_max"]),
        "history_geometry": float(
            diagnostics["history_geometry_gap_abs_max"]
        )
        <= float(thresholds["history_geometry_gap_abs_max"]),
        "midpoint_identity": float(
            diagnostics["midpoint_identity_abs_max"]
        )
        <= float(thresholds["midpoint_identity_abs_max"]),
        "current_attachment": float(
            diagnostics["current_attachment_abs_max"]
        )
        <= float(thresholds["current_attachment_abs_max"]),
        "entrance_prestep_inventory": float(
            diagnostics["entrance_prestep_inventory_abs_max"]
        )
        <= float(thresholds["material_inventory_increment_abs_max"]),
        "typed_body_direct_w_quadrature": int(
            diagnostics["march_body_quadrature_mismatch_count"]
        )
        == 0
        and int(diagnostics["direct_w_quadrature_mismatch_count"]) == 0,
        "input_immutable": float(diagnostics["input_mutation_abs_max"])
        <= float(thresholds["input_mutation_abs_max"]),
    }


def observe_compatible_history(
    march: ExplicitMidpointWakeMarch,
    *,
    case: HistoryCase,
    mesh: Any,
    topology: Any,
    upper_face_indices: np.ndarray,
    lower_face_indices: np.ndarray,
    attachment: MaterialWakeCutAttachment,
    body_and_direct_w_quadrature_order: int,
    pressure_line_quadrature_order: int,
    thresholds: Mapping[str, Any],
    input_snapshot: Mapping[str, np.ndarray],
) -> CompatibleHistoryObservation:
    """Observe one prestepped S3e history without modifying or re-solving it."""

    if not isinstance(march, ExplicitMidpointWakeMarch):
        raise ReachablePressureV2GuardError(
            "S3ai-v2 requires an ExplicitMidpointWakeMarch"
        )
    if (
        len(march.steps) != case.marcher_steps
        or not np.isclose(march.time_start, -case.timestep)
        or not np.isclose(march.time_end, 1.0)
        or not np.isclose(march.timestep, case.timestep)
    ):
        raise ReachablePressureV2GuardError(
            f"history {case.name} does not contain exactly one compatible "
            "pre-step"
        )
    prestep = march.steps[0]
    if (
        not np.isclose(prestep.time_previous, -case.timestep)
        or not np.isclose(prestep.time_current, 0.0)
    ):
        raise ReachablePressureV2GuardError(
            f"history {case.name} pre-step does not end at t=0"
        )
    measured = march.steps[1:]
    if any(step.time_previous < -2.0e-15 for step in measured):
        raise ReachablePressureV2GuardError(
            f"history {case.name} includes prehistory in the pressure window"
        )

    diagnostics = _new_diagnostics()
    roles = cut_role_operators(topology)
    if (
        not np.array_equal(
            roles.active_row_indices, np.arange(1, 8, dtype=np.int64)
        )
        or not np.array_equal(
            roles.zero_row_indices,
            np.array((0, 8), dtype=np.int64),
        )
    ):
        raise ReachablePressureV2GuardError(
            "v2.2 canonical active rows must be complete rows 1..7 "
            "with zero-tip rows 0 and 8"
        )
    wrong_attachment = MaterialWakeCutAttachment(
        topology.ordered_cut_vertex_indices,
        -attachment.wake_jump_from_body_cut_sign,
    )
    previous_stage = observe_direct_independent_stage(
        prestep.full_stage,
        attachment=attachment,
        upper_face_indices=upper_face_indices,
        lower_face_indices=lower_face_indices,
        body_and_direct_w_quadrature_order=(
            body_and_direct_w_quadrature_order
        ),
        pressure_line_quadrature_order=pressure_line_quadrature_order,
    )
    (
        previous_inventory,
        previous_wrong_birth,
        previous_wrong_attachment,
    ) = _inventories(
        prestep.full_stage,
        topology=topology,
        attachment=attachment,
        wrong_attachment=wrong_attachment,
    )
    _accumulate_stage_diagnostics(
        diagnostics,
        previous_stage,
        expected_quadrature_order=body_and_direct_w_quadrature_order,
    )
    _accumulate_inventory_diagnostics(diagnostics, previous_inventory)
    # v2.2 records the finite-q entrance trace but never rejects it against a
    # pointwise or fixed tolerance.  Its continuum-zero status is decided
    # only by the matched q-space projected interval.
    diagnostics["entrance_prestep_finite_q_trace_abs_max"] = _maximum_abs(
        previous_stage.material_current_trace
    )
    diagnostics["entrance_prestep_inventory_abs_max"] = _maximum_abs(
        previous_inventory.inventory
    )
    diagnostics["material_inventory_abs_max"] = _maximum_abs(
        previous_inventory.inventory
    )
    diagnostics["wrong_attachment_inventory_abs_max"] = _maximum_abs(
        previous_wrong_attachment.inventory
    )

    stored_steps: list[np.ndarray] = []
    direct_steps: list[np.ndarray] = []
    stage_times = [float(prestep.time_current)]
    stage_roles = ["entrance_prestep_full"]
    canonical_traces = [
        previous_stage.canonical_material_trace.copy()
    ]
    body_traces = [previous_inventory.body_cut_trace.copy()]
    canonical_releases = [
        previous_inventory.canonical_release.copy()
    ]
    inventories = [previous_inventory.inventory.copy()]
    stored_pressures = [previous_stage.weak_pressure.copy()]
    direct_pressures = [previous_stage.direct_weak_pressure.copy()]
    measurement_times: list[np.ndarray] = []
    mass = _strict_active_mass(previous_stage.active_mass)
    observed_stage_count = 1

    for step in march.steps:
        diagnostics["old_state_mutation_abs_max"] = max(
            float(diagnostics["old_state_mutation_abs_max"]),
            float(step.old_strength_mutation),
        )
        diagnostics["history_geometry_gap_abs_max"] = max(
            float(diagnostics["history_geometry_gap_abs_max"]),
            float(step.old_geometry_convection_error),
        )
        diagnostics["midpoint_identity_abs_max"] = max(
            float(diagnostics["midpoint_identity_abs_max"]),
            float(step.midpoint_row_identity_error),
        )
        diagnostics["current_attachment_abs_max"] = max(
            float(diagnostics["current_attachment_abs_max"]),
            float(step.current_attachment_error),
        )

    for step in measured:
        midpoint = observe_direct_independent_stage(
            step.half_stage,
            attachment=attachment,
            upper_face_indices=upper_face_indices,
            lower_face_indices=lower_face_indices,
            body_and_direct_w_quadrature_order=(
                body_and_direct_w_quadrature_order
            ),
            pressure_line_quadrature_order=pressure_line_quadrature_order,
        )
        current = observe_direct_independent_stage(
            step.full_stage,
            attachment=attachment,
            upper_face_indices=upper_face_indices,
            lower_face_indices=lower_face_indices,
            body_and_direct_w_quadrature_order=(
                body_and_direct_w_quadrature_order
            ),
            pressure_line_quadrature_order=pressure_line_quadrature_order,
        )
        (
            midpoint_inventory,
            midpoint_wrong_birth,
            midpoint_wrong_attachment,
        ) = _inventories(
            step.half_stage,
            topology=topology,
            attachment=attachment,
            wrong_attachment=wrong_attachment,
        )
        (
            current_inventory,
            current_wrong_birth,
            current_wrong_attachment,
        ) = _inventories(
            step.full_stage,
            topology=topology,
            attachment=attachment,
            wrong_attachment=wrong_attachment,
        )

        for stage in (midpoint, current):
            if not np.array_equal(
                _strict_active_mass(stage.active_mass), mass
            ):
                raise ReachablePressureV2GuardError(
                    f"history {case.name} changed its active P2 mass"
                )
            _accumulate_stage_diagnostics(
                diagnostics,
                stage,
                expected_quadrature_order=(
                    body_and_direct_w_quadrature_order
                ),
            )
        for inventory in (midpoint_inventory, current_inventory):
            _accumulate_inventory_diagnostics(diagnostics, inventory)

        correct_increments = (
            material_inventory_increment(
                previous_inventory, midpoint_inventory
            ),
            material_inventory_increment(
                midpoint_inventory, current_inventory
            ),
        )
        wrong_birth_increments = (
            material_inventory_increment(
                previous_wrong_birth, midpoint_wrong_birth
            ),
            material_inventory_increment(
                midpoint_wrong_birth, current_wrong_birth
            ),
        )
        wrong_attachment_increments = (
            material_inventory_increment(
                previous_wrong_attachment, midpoint_wrong_attachment
            ),
            material_inventory_increment(
                midpoint_wrong_attachment, current_wrong_attachment
            ),
        )
        diagnostics["material_inventory_increment_abs_max"] = max(
            float(diagnostics["material_inventory_increment_abs_max"]),
            *(_maximum_abs(value) for value in correct_increments),
        )
        diagnostics["material_inventory_abs_max"] = max(
            float(diagnostics["material_inventory_abs_max"]),
            _maximum_abs(midpoint_inventory.inventory),
            _maximum_abs(current_inventory.inventory),
        )
        diagnostics["wrong_birth_inventory_increment_abs_max"] = max(
            float(
                diagnostics[
                    "wrong_birth_inventory_increment_abs_max"
                ]
            ),
            *(_maximum_abs(value) for value in wrong_birth_increments),
        )
        diagnostics[
            "wrong_attachment_inventory_increment_abs_max"
        ] = max(
            float(
                diagnostics[
                    "wrong_attachment_inventory_increment_abs_max"
                ]
            ),
            *(
                _maximum_abs(value)
                for value in wrong_attachment_increments
            ),
        )
        diagnostics["wrong_attachment_inventory_abs_max"] = max(
            float(diagnostics["wrong_attachment_inventory_abs_max"]),
            _maximum_abs(midpoint_wrong_attachment.inventory),
            _maximum_abs(current_wrong_attachment.inventory),
        )

        dt = float(step.time_current - step.time_previous)
        stored_residual = weak_pressure_step_residual(
            mass,
            previous_stage.active_trace,
            current.active_trace,
            dt,
            midpoint.weak_pressure,
        )
        # The direct cross-observer changes only P(phi); it retains the same
        # face_mu material derivative and never propagates direct_phi.
        direct_residual = weak_pressure_step_residual(
            mass,
            previous_stage.active_trace,
            current.active_trace,
            dt,
            midpoint.direct_weak_pressure,
        )
        stored_steps.append(stored_residual)
        direct_steps.append(direct_residual)
        stage_times.extend(
            (float(step.time_midpoint), float(step.time_current))
        )
        stage_roles.extend(("measured_midpoint", "measured_full"))
        canonical_traces.extend(
            (
                midpoint.canonical_material_trace.copy(),
                current.canonical_material_trace.copy(),
            )
        )
        body_traces.extend(
            (
                midpoint_inventory.body_cut_trace.copy(),
                current_inventory.body_cut_trace.copy(),
            )
        )
        canonical_releases.extend(
            (
                midpoint_inventory.canonical_release.copy(),
                current_inventory.canonical_release.copy(),
            )
        )
        inventories.extend(
            (
                midpoint_inventory.inventory.copy(),
                current_inventory.inventory.copy(),
            )
        )
        stored_pressures.extend(
            (midpoint.weak_pressure.copy(), current.weak_pressure.copy())
        )
        direct_pressures.extend(
            (
                midpoint.direct_weak_pressure.copy(),
                current.direct_weak_pressure.copy(),
            )
        )
        measurement_times.append(
            np.array(
                (
                    step.time_previous,
                    step.time_midpoint,
                    step.time_current,
                ),
                dtype=float,
            )
        )
        observed_stage_count += 2

        previous_stage = current
        previous_inventory = current_inventory
        previous_wrong_birth = current_wrong_birth
        previous_wrong_attachment = current_wrong_attachment

    stored_array = np.stack(stored_steps, axis=0)
    direct_array = np.stack(direct_steps, axis=0)
    diagnostics["input_mutation_abs_max"] = _fresh_input_mutation(
        input_snapshot,
        mesh,
        upper_face_indices,
        lower_face_indices,
        topology,
    )
    checks = _history_checks(diagnostics, thresholds)
    return CompatibleHistoryObservation(
        case=case,
        stored_window_residual=np.sum(stored_array, axis=0),
        direct_window_residual=np.sum(direct_array, axis=0),
        stored_step_residuals=stored_array,
        direct_step_residuals=direct_array,
        mass_active=mass,
        stage_arrays=TypedStageArrays(
            stage_times=np.asarray(stage_times, dtype=float),
            stage_roles=tuple(stage_roles),
            canonical_material_current_trace=np.stack(
                canonical_traces, axis=0
            ),
            body_cut_trace=np.stack(body_traces, axis=0),
            canonical_material_release=np.stack(
                canonical_releases, axis=0
            ),
            representation_inventory=np.stack(inventories, axis=0),
            stored_weak_pressure=np.stack(
                stored_pressures, axis=0
            ),
            direct_weak_pressure=np.stack(
                direct_pressures, axis=0
            ),
            measurement_times=np.stack(measurement_times, axis=0),
        ),
        diagnostics=diagnostics,
        checks=checks,
        observed_stage_count=observed_stage_count,
    )


def _execute_history_case(
    case: HistoryCase,
    contract: Mapping[str, Any],
    *,
    _marcher: Callable[..., Any] | None = None,
    _observer: Callable[..., Any] | None = None,
) -> CompatibleHistoryObservation:
    """Construct and observe one fresh path; injection points are test-only."""

    if not isinstance(case, HistoryCase):
        raise ReachablePressureV2GuardError(
            "history execution requires one frozen HistoryCase"
        )
    marcher = (
        march_actual_boundary_material_wake_explicit_midpoint
        if _marcher is None
        else _marcher
    )
    observer = (
        observe_compatible_history if _observer is None else _observer
    )
    mesh, upper, lower, cut_edges, endpoints = (
        build_canonical_diamond_wing()
    )
    topology = classified_p2_cut_topology(
        mesh,
        upper_face_indices=upper,
        lower_face_indices=lower,
        cut_edges=cut_edges,
        zero_jump_end_vertices=endpoints,
    )
    attachment = MaterialWakeCutAttachment(
        topology.ordered_cut_vertex_indices,
        int(
            contract["typed_attachment"][
                "wake_jump_from_body_cut_sign"
            ]
        ),
    )
    snapshot = _fresh_input_snapshot(
        mesh, upper, lower, topology
    )
    q = int(case.quadrature_order)
    canonical = contract["canonical"]
    march = marcher(
        mesh,
        topology,
        incident_velocity_at_time=lambda time: _incident_history(
            time,
            epsilon_signed=case.epsilon_signed,
            face_count=len(mesh.faces),
        ),
        initial_body_cut_jump=np.zeros(
            len(topology.cut_node_coordinates),
            dtype=float,
        ),
        time_start=-case.timestep,
        time_end=1.0,
        timestep=case.timestep,
        trailing_edge_x=float(canonical["trailing_edge_x"]),
        convection_speed=float(
            canonical["prescribed_material_convection_speed"]
        ),
        target_quadrature_order=q,
        source_quadrature_order=q,
    )
    return observer(
        march,
        case=case,
        mesh=mesh,
        topology=topology,
        upper_face_indices=upper,
        lower_face_indices=lower,
        attachment=attachment,
        body_and_direct_w_quadrature_order=q,
        pressure_line_quadrature_order=int(
            canonical["common_anchor"][
                "pressure_line_quadrature_order"
            ]
        ),
        thresholds=contract["hard_guards"]["thresholds"],
        input_snapshot=snapshot,
    )


def _collect_frozen_histories(
    contract: Mapping[str, Any],
    *,
    case_executor: Callable[
        [HistoryCase, Mapping[str, Any]], CompatibleHistoryObservation
    ]
    | None = None,
) -> dict[str, CompatibleHistoryObservation]:
    cases = _validated_registry(contract)
    executor = _execute_history_case if case_executor is None else case_executor
    observations = {
        case.name: executor(case, contract) for case in cases
    }
    if tuple(observations) != tuple(case.name for case in cases):
        raise ReachablePressureV2GuardError(
            "31-history registry order or identity drifted"
        )
    return observations


def _richardson_allowance_without_round(
    report: Any,
    fine: np.ndarray,
    mass: np.ndarray,
) -> float:
    return float(
        dual_mass_norm(report.fine_extrapolated - fine, mass)
        + dual_mass_norm(
            report.fine_extrapolated
            - report.medium_extrapolated,
            mass,
        )
    )


def _quadrature_allowance(report: Any) -> float:
    if report.plateau:
        return float(report.medium_fine_change)
    if not report.passed:
        return float("inf")
    return float(
        report.medium_fine_change
        / (1.0 - report.contraction_ratio)
    )


def _scalar_second_order(
    *,
    fine: float,
    medium: float,
    coarse: float,
    contraction_ratio_min: float,
    floating_floor_override: float | None = None,
) -> dict[str, Any]:
    values = np.asarray((fine, medium, coarse), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ReachablePressureV2GuardError(
            "scalar zero-reference family is non-finite"
        )
    fine_r = (4.0 * fine - medium) / 3.0
    medium_r = (4.0 * medium - coarse) / 3.0
    coarse_medium = abs(coarse - medium)
    medium_fine = abs(medium - fine)
    default_floor = (
        4096.0
        * np.finfo(float).eps
        * float(np.max(np.abs(values), initial=0.0))
    )
    floor = (
        default_floor
        if floating_floor_override is None
        else float(floating_floor_override)
    )
    if not np.isfinite(floor) or floor < 0.0:
        raise ReachablePressureV2GuardError(
            "scalar Richardson floating floor must be finite and non-negative"
        )
    plateau = max(coarse_medium, medium_fine) <= floor
    ratio = (
        (np.inf if coarse_medium > 0.0 else np.nan)
        if medium_fine == 0.0
        else coarse_medium / medium_fine
    )
    passed = bool(
        plateau
        or (
            coarse_medium > medium_fine
            and ratio >= contraction_ratio_min
        )
    )
    return {
        "fine_extrapolated": float(fine_r),
        "medium_extrapolated": float(medium_r),
        "coarse_medium_change": float(coarse_medium),
        "medium_fine_change": float(medium_fine),
        "contraction_ratio": float(ratio),
        "floating_floor": float(floor),
        "plateau": plateau,
        "allowance_without_round": float(
            abs(fine_r - fine) + abs(fine_r - medium_r)
        ),
        "passed": passed,
    }


def _scalar_quadrature_tail(
    *,
    coarse: float,
    medium: float,
    fine: float,
    floating_floor_override: float | None = None,
) -> dict[str, Any]:
    values = np.asarray((coarse, medium, fine), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ReachablePressureV2GuardError(
            "scalar zero-reference quadrature family is non-finite"
        )
    delta1 = abs(medium - coarse)
    delta2 = abs(fine - medium)
    default_floor = (
        4096.0
        * np.finfo(float).eps
        * float(np.max(np.abs(values), initial=0.0))
    )
    floor = (
        default_floor
        if floating_floor_override is None
        else float(floating_floor_override)
    )
    if not np.isfinite(floor) or floor < 0.0:
        raise ReachablePressureV2GuardError(
            "scalar quadrature floating floor must be finite and non-negative"
        )
    plateau = max(delta1, delta2) <= floor
    if plateau:
        ratio = np.nan if delta1 == 0.0 else delta2 / delta1
        allowance = delta2
        passed = True
    elif delta1 <= floor:
        ratio = np.inf
        allowance = np.inf
        passed = False
    else:
        ratio = delta2 / delta1
        passed = bool(ratio < 1.0)
        allowance = (
            delta2 / (1.0 - ratio) if passed else np.inf
        )
    return {
        "coarse_medium_change": float(delta1),
        "medium_fine_change": float(delta2),
        "contraction_ratio": float(ratio),
        "floating_floor": float(floor),
        "plateau": plateau,
        "allowance": float(allowance),
        "passed": passed,
    }


def _typed_floating_floor(
    values: list[np.ndarray] | tuple[np.ndarray, ...],
    mass: np.ndarray,
    space: str,
) -> float:
    if not values:
        raise ReachablePressureV2GuardError(
            "floating floor needs at least one unprojected operand"
        )
    return float(
        4096.0
        * np.finfo(float).eps
        * max(_typed_norm(value, mass, space) for value in values)
    )


def _typed_richardson(
    *,
    fine: np.ndarray,
    medium: np.ndarray,
    coarse: np.ndarray,
    unprojected_operands: tuple[np.ndarray, np.ndarray, np.ndarray],
    mass: np.ndarray,
    space: str,
    contraction_ratio_min: float,
) -> dict[str, Any]:
    fine_r = (4.0 * fine - medium) / 3.0
    medium_r = (4.0 * medium - coarse) / 3.0
    coarse_medium = _typed_norm(coarse - medium, mass, space)
    medium_fine = _typed_norm(medium - fine, mass, space)
    floor = _typed_floating_floor(
        list(unprojected_operands), mass, space
    )
    plateau = max(coarse_medium, medium_fine) <= floor
    ratio = (
        (np.inf if coarse_medium > 0.0 else np.nan)
        if medium_fine == 0.0
        else coarse_medium / medium_fine
    )
    passed = bool(
        plateau
        or (
            coarse_medium > medium_fine
            and ratio >= contraction_ratio_min
        )
    )
    allowance = (
        _typed_norm(fine_r - fine, mass, space)
        + _typed_norm(fine_r - medium_r, mass, space)
    )
    return {
        "fine_extrapolated": fine_r,
        "medium_extrapolated": medium_r,
        "coarse_medium_change": coarse_medium,
        "medium_fine_change": medium_fine,
        "contraction_ratio": ratio,
        "floating_floor_from_unprojected_operands": floor,
        "plateau": plateau,
        "allowance_without_round": allowance,
        "passed": passed,
    }


def _typed_quadrature_tail(
    *,
    coarse: np.ndarray,
    medium: np.ndarray,
    fine: np.ndarray,
    unprojected_operands: tuple[np.ndarray, np.ndarray, np.ndarray],
    mass: np.ndarray,
    space: str,
) -> dict[str, Any]:
    delta1 = _typed_norm(medium - coarse, mass, space)
    delta2 = _typed_norm(fine - medium, mass, space)
    floor = _typed_floating_floor(
        list(unprojected_operands), mass, space
    )
    plateau = max(delta1, delta2) <= floor
    if plateau:
        ratio = np.nan if delta1 == 0.0 else delta2 / delta1
        allowance = delta2
        passed = True
    elif delta1 <= floor:
        ratio = np.inf
        allowance = np.inf
        passed = False
    else:
        ratio = delta2 / delta1
        passed = bool(ratio < 1.0)
        allowance = (
            delta2 / (1.0 - ratio) if passed else np.inf
        )
    return {
        "coarse_medium_change": delta1,
        "medium_fine_change": delta2,
        "contraction_ratio": ratio,
        "floating_floor_from_unprojected_operands": floor,
        "plateau": plateau,
        "allowance_without_round": allowance,
        "passed": passed,
    }


def _typed_mixed_cube(
    projected: Mapping[str, np.ndarray],
    unprojected: Mapping[str, np.ndarray],
    mass: np.ndarray,
    space: str,
) -> dict[str, Any]:
    expected = {f"{index:03b}" for index in range(8)}
    if set(projected) != expected or set(unprojected) != expected:
        raise ReachablePressureV2GuardError(
            "typed mixed cube requires keys 000 through 111"
        )
    cube = projected
    components = {
        "epsilon_timestep": (
            cube["110"] - cube["100"] - cube["010"] + cube["000"]
        ),
        "epsilon_quadrature": (
            cube["101"] - cube["100"] - cube["001"] + cube["000"]
        ),
        "timestep_quadrature": (
            cube["011"] - cube["010"] - cube["001"] + cube["000"]
        ),
        "epsilon_timestep_quadrature": (
            cube["111"]
            - cube["110"]
            - cube["101"]
            - cube["011"]
            + cube["100"]
            + cube["010"]
            + cube["001"]
            - cube["000"]
        ),
    }
    norms = {
        name: _typed_norm(value, mass, space)
        for name, value in components.items()
    }
    return {
        "components": components,
        "component_norms": norms,
        "allowance_without_round": float(sum(norms.values())),
        "floating_floor_from_unprojected_operands": (
            _typed_floating_floor(
                list(unprojected.values()), mass, space
            )
        ),
    }


def _projected_omega_summary(
    tangents: Mapping[str, np.ndarray],
    direct_tangents: Mapping[str, np.ndarray],
    repeat_tangent: np.ndarray,
    mass: np.ndarray,
    parity: str,
    contraction_ratio_min: float,
) -> dict[str, Any]:
    projected = {
        name: span_project(value, parity)
        for name, value in tangents.items()
    }
    projected_repeat = span_project(repeat_tangent, parity)
    projected_direct = span_project(direct_tangents["A"], parity)
    epsilon = _typed_richardson(
        fine=projected["A"],
        medium=projected["E2"],
        coarse=projected["E4"],
        unprojected_operands=(
            tangents["A"], tangents["E2"], tangents["E4"]
        ),
        mass=mass,
        space="dual",
        contraction_ratio_min=contraction_ratio_min,
    )
    timestep = _typed_richardson(
        fine=projected["A"],
        medium=projected["DT2"],
        coarse=projected["DT4"],
        unprojected_operands=(
            tangents["A"], tangents["DT2"], tangents["DT4"]
        ),
        mass=mass,
        space="dual",
        contraction_ratio_min=contraction_ratio_min,
    )
    quadrature = _typed_quadrature_tail(
        coarse=projected["Q8"],
        medium=projected["Q10"],
        fine=projected["A"],
        unprojected_operands=(
            tangents["Q8"], tangents["Q10"], tangents["A"]
        ),
        mass=mass,
        space="dual",
    )
    cube_names = mixed_cube_configuration_names()
    cube = _typed_mixed_cube(
        {
            bits: projected[name]
            for bits, name in cube_names.items()
        },
        {
            bits: tangents[name]
            for bits, name in cube_names.items()
        },
        mass,
        "dual",
    )
    repeat_allowance = dual_mass_norm(
        projected_repeat - projected["A"], mass
    )
    algebraic_allowance = dual_mass_norm(
        projected_direct - projected["A"], mass
    )
    repeat_floor = _typed_floating_floor(
        [repeat_tangent, tangents["A"]], mass, "dual"
    )
    algebraic_floor = _typed_floating_floor(
        [direct_tangents["A"], tangents["A"]], mass, "dual"
    )
    round_once = max(
        epsilon["floating_floor_from_unprojected_operands"],
        timestep["floating_floor_from_unprojected_operands"],
        quadrature["floating_floor_from_unprojected_operands"],
        cube["floating_floor_from_unprojected_operands"],
        repeat_floor,
        algebraic_floor,
    )
    uncertainty = sum(
        (
            epsilon["allowance_without_round"],
            timestep["allowance_without_round"],
            quadrature["allowance_without_round"],
            cube["allowance_without_round"],
            repeat_allowance,
            algebraic_allowance,
            round_once,
        )
    )
    return {
        "parity": parity,
        "anchor_vector": projected["A"],
        "interval": _finite_interval(
            dual_mass_norm(projected["A"], mass), uncertainty
        ),
        "uncertainty_components": {
            "epsilon": epsilon["allowance_without_round"],
            "timestep": timestep["allowance_without_round"],
            "quadrature": quadrature["allowance_without_round"],
            "mixed_cube": cube["allowance_without_round"],
            "fresh_repeat": repeat_allowance,
            "algebraic_cross_observer": algebraic_allowance,
            "floating_round_once": round_once,
        },
        "reports": {
            "epsilon": epsilon,
            "timestep": timestep,
            "quadrature": quadrature,
            "mixed_cube": cube,
        },
        "convergence_passed": bool(
            epsilon["passed"]
            and timestep["passed"]
            and quadrature["passed"]
        ),
    }


def _projected_zero_sum_summary(
    zero_vectors: Mapping[str, np.ndarray],
    zero_direct: Mapping[str, np.ndarray],
    repeat: np.ndarray,
    mass: np.ndarray,
    parity: str,
    contraction_ratio_min: float,
) -> dict[str, Any]:
    projected = {
        name: span_project(value, parity)
        for name, value in zero_vectors.items()
    }
    projected_repeat = span_project(repeat, parity)
    projected_direct = span_project(zero_direct["Z_A"], parity)
    timestep = _typed_richardson(
        fine=projected["Z_A"],
        medium=projected["Z_DT2"],
        coarse=projected["Z_DT4"],
        unprojected_operands=(
            zero_vectors["Z_A"],
            zero_vectors["Z_DT2"],
            zero_vectors["Z_DT4"],
        ),
        mass=mass,
        space="dual",
        contraction_ratio_min=contraction_ratio_min,
    )
    quadrature = _typed_quadrature_tail(
        coarse=projected["Z_Q8"],
        medium=projected["Z_Q10"],
        fine=projected["Z_A"],
        unprojected_operands=(
            zero_vectors["Z_Q8"],
            zero_vectors["Z_Q10"],
            zero_vectors["Z_A"],
        ),
        mass=mass,
        space="dual",
    )
    mixed = (
        projected["Z_DT2_Q10"]
        - projected["Z_DT2"]
        - projected["Z_Q10"]
        + projected["Z_A"]
    )
    mixed_allowance = dual_mass_norm(mixed, mass)
    repeat_allowance = dual_mass_norm(
        projected_repeat - projected["Z_A"], mass
    )
    algebraic_allowance = dual_mass_norm(
        projected_direct - projected["Z_A"], mass
    )
    round_once = max(
        timestep["floating_floor_from_unprojected_operands"],
        quadrature["floating_floor_from_unprojected_operands"],
        _typed_floating_floor(
            list(zero_vectors.values())
            + [repeat, zero_direct["Z_A"]],
            mass,
            "dual",
        ),
    )
    uncertainty = sum(
        (
            timestep["allowance_without_round"],
            quadrature["allowance_without_round"],
            mixed_allowance,
            repeat_allowance,
            algebraic_allowance,
            round_once,
        )
    )
    return {
        "parity": parity,
        "anchor_vector": projected["Z_A"],
        "interval": _finite_interval(
            dual_mass_norm(projected["Z_A"], mass), uncertainty
        ),
        "uncertainty_components": {
            "timestep": timestep["allowance_without_round"],
            "quadrature": quadrature["allowance_without_round"],
            "mixed_corner": mixed_allowance,
            "fresh_repeat": repeat_allowance,
            "algebraic_cross_observer": algebraic_allowance,
            "floating_round_once": round_once,
        },
        "reports": {
            "timestep": timestep,
            "quadrature": quadrature,
        },
        "convergence_passed": bool(
            timestep["passed"] and quadrature["passed"]
        ),
    }


def _projected_noncancellation(
    observation: CompatibleHistoryObservation,
    parity: str,
    *,
    direct: bool,
) -> float:
    steps = (
        observation.direct_step_residuals
        if direct
        else observation.stored_step_residuals
    )
    return float(
        sum(
            dual_mass_norm(span_project(step, parity), observation.mass_active)
            for step in steps
        )
    )


def _projected_zero_noncancellation_summary(
    observations: Mapping[str, CompatibleHistoryObservation],
    parity: str,
    contraction_ratio_min: float,
) -> dict[str, Any]:
    names = (
        "Z_A",
        "Z_DT2",
        "Z_DT4",
        "Z_Q10",
        "Z_Q8",
        "Z_DT2_Q10",
    )
    values = {
        name: _projected_noncancellation(
            observations[name], parity, direct=False
        )
        for name in names
    }
    direct = {
        name: _projected_noncancellation(
            observations[name], parity, direct=True
        )
        for name in names
    }
    unprojected = {
        name: observations[name].stored_noncancellation
        for name in names
    }
    repeat = _projected_noncancellation(
        observations["REPEAT_A_zero"], parity, direct=False
    )
    repeat_unprojected = observations[
        "REPEAT_A_zero"
    ].stored_noncancellation
    timestep_unprojected_floor = float(
        4096.0
        * np.finfo(float).eps
        * max(
            unprojected["Z_A"],
            unprojected["Z_DT2"],
            unprojected["Z_DT4"],
        )
    )
    quadrature_unprojected_floor = float(
        4096.0
        * np.finfo(float).eps
        * max(
            unprojected["Z_Q8"],
            unprojected["Z_Q10"],
            unprojected["Z_A"],
        )
    )
    timestep = _scalar_second_order(
        fine=values["Z_A"],
        medium=values["Z_DT2"],
        coarse=values["Z_DT4"],
        contraction_ratio_min=contraction_ratio_min,
        floating_floor_override=timestep_unprojected_floor,
    )
    quadrature = _scalar_quadrature_tail(
        coarse=values["Z_Q8"],
        medium=values["Z_Q10"],
        fine=values["Z_A"],
        floating_floor_override=quadrature_unprojected_floor,
    )
    mixed = abs(
        values["Z_DT2_Q10"]
        - values["Z_DT2"]
        - values["Z_Q10"]
        + values["Z_A"]
    )
    repeat_allowance = abs(repeat - values["Z_A"])
    algebraic_allowance = abs(direct["Z_A"] - values["Z_A"])
    round_once = float(
        max(
            timestep_unprojected_floor,
            quadrature_unprojected_floor,
            4096.0
            * np.finfo(float).eps
            * max(
                *unprojected.values(),
                repeat_unprojected,
                observations["Z_A"].direct_noncancellation,
            ),
        )
    )
    uncertainty = sum(
        (
            timestep["allowance_without_round"],
            quadrature["allowance"],
            mixed,
            repeat_allowance,
            algebraic_allowance,
            round_once,
        )
    )
    return {
        "parity": parity,
        "anchor": values["Z_A"],
        "interval": _finite_interval(values["Z_A"], uncertainty),
        "uncertainty_components": {
            "timestep": timestep["allowance_without_round"],
            "quadrature": quadrature["allowance"],
            "mixed_corner": mixed,
            "fresh_repeat": repeat_allowance,
            "algebraic_cross_observer": algebraic_allowance,
            "floating_round_once": round_once,
        },
        "reports": {
            "timestep": timestep,
            "quadrature": quadrature,
        },
        "convergence_passed": bool(
            timestep["passed"] and quadrature["passed"]
        ),
    }


def _finite_interval(
    observed_norm: float,
    uncertainty: float,
) -> dict[str, float | None]:
    if not np.isfinite(uncertainty):
        return {
            "observed_norm": float(observed_norm),
            "uncertainty": None,
            "lower": None,
            "upper": None,
        }
    return {
        "observed_norm": float(observed_norm),
        "uncertainty": float(uncertainty),
        "lower": max(0.0, float(observed_norm) - float(uncertainty)),
        "upper": float(observed_norm) + float(uncertainty),
    }


def _report_dict(report: Any) -> dict[str, Any]:
    return _json_safe(asdict(report))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return value


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _code_fingerprints() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        HERE / "actual_boundary_3d_cut_wake_junction_guard.py",
        HERE / "claim_runtime" / "actual_boundary_body_wake.py",
        HERE / "claim_runtime" / "actual_wake_direct_independent.py",
        HERE / "claim_runtime" / "actual_wake_kutta_compatibility.py",
        HERE / "claim_runtime" / "actual_wake_kutta_closure_roles.py",
        HERE / "claim_runtime" / "actual_wake_reachable_pressure.py",
        HERE / "claim_runtime" / "classified_p2_cut_topology.py",
        HERE / "claim_runtime" / "distributed_doublet.py",
        HERE / "claim_runtime" / "material_attachment_inventory.py",
        HERE / "claim_runtime" / "material_birth_flux.py",
        HERE / "claim_runtime" / "reachable_pressure_uncertainty.py",
        HERE / "claim_runtime" / "reachable_pressure_symmetry.py",
        HERE / "claim_runtime" / "reachable_pressure_cases.py",
        HERE / "claim_runtime" / "material_wake_time_march.py",
    )
    return {
        str(path.relative_to(HERE.parent)): _sha256_path(path)
        for path in paths
    }


def _case_identity_payload(case: HistoryCase) -> dict[str, Any]:
    replicate = 1 if case.role == "fresh_repeat" else 0
    identity = {
        "epsilon_signed_hex": float(case.epsilon_signed).hex(),
        "timestep_hex": float(case.timestep).hex(),
        "quadrature_order": int(case.quadrature_order),
        "replicate_id": replicate,
    }
    encoded = repr(tuple(identity.values())).encode("utf-8")
    identity["sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def _matched_stage_parity_report(
    values: Mapping[str, CompatibleHistoryObservation],
    mass: np.ndarray,
) -> dict[str, Any]:
    """Apply v2.2 q8/q10/q12 parity gates at every matched stage."""

    groups = {
        "signed_positive": ("Q8_plus", "Q10_plus", "A_plus"),
        "signed_negative": ("Q8_minus", "Q10_minus", "A_minus"),
        "zero": ("Z_Q8", "Z_Q10", "Z_A"),
    }
    entries: list[dict[str, Any]] = []
    for group, names in groups.items():
        q8, q10, q12 = (values[name] for name in names)
        for candidate in (q10, q12):
            if (
                not np.array_equal(
                    candidate.stage_arrays.stage_times,
                    q8.stage_arrays.stage_times,
                )
                or candidate.stage_arrays.stage_roles
                != q8.stage_arrays.stage_roles
                or not np.array_equal(
                    candidate.stage_arrays.measurement_times,
                    q8.stage_arrays.measurement_times,
                )
            ):
                raise ReachablePressureV2GuardError(
                    f"matched q stages drifted in {group}"
                )
        observables = (
            (
                "canonical_material_current_trace",
                q8.stage_arrays.canonical_material_current_trace[:, 1:-1],
                q10.stage_arrays.canonical_material_current_trace[:, 1:-1],
                q12.stage_arrays.canonical_material_current_trace[:, 1:-1],
                q8.stage_arrays.stage_times,
                "primal",
            ),
            (
                "stored_weak_pressure",
                q8.stage_arrays.stored_weak_pressure,
                q10.stage_arrays.stored_weak_pressure,
                q12.stage_arrays.stored_weak_pressure,
                q8.stage_arrays.stage_times,
                "dual",
            ),
            (
                "direct_weak_pressure",
                q8.stage_arrays.direct_weak_pressure,
                q10.stage_arrays.direct_weak_pressure,
                q12.stage_arrays.direct_weak_pressure,
                q8.stage_arrays.stage_times,
                "dual",
            ),
            (
                "stored_step_residual",
                q8.stored_step_residuals,
                q10.stored_step_residuals,
                q12.stored_step_residuals,
                q8.stage_arrays.measurement_times[:, 1],
                "dual",
            ),
            (
                "direct_step_residual",
                q8.direct_step_residuals,
                q10.direct_step_residuals,
                q12.direct_step_residuals,
                q8.stage_arrays.measurement_times[:, 1],
                "dual",
            ),
        )
        for observable, coarse, medium, fine, times, metric_role in observables:
            if coarse.shape != medium.shape or coarse.shape != fine.shape:
                raise ReachablePressureV2GuardError(
                    f"matched q observable shape drifted for {group}/{observable}"
                )
            for index, (v8, v10, v12) in enumerate(
                zip(coarse, medium, fine, strict=True)
            ):
                decision = parity_quadrature_decision(
                    q8=v8,
                    q10=v10,
                    q12=v12,
                    mass=mass,
                    metric_role=metric_role,
                    unprojected_round_operands=(v8, v10, v12),
                )
                initializer_zero_target = bool(
                    observable
                    == "canonical_material_current_trace"
                    and index == 0
                )
                initializer_zero_target_passed = bool(
                    not initializer_zero_target
                    or (
                        not decision.protocol_no_go
                        and decision.even.lower == 0.0
                        and decision.odd.lower == 0.0
                    )
                )
                entries.append(
                    {
                        "group": group,
                        "observable": observable,
                        "index": index,
                        "time": float(times[index]),
                        "continuum_zero_initializer": (
                            initializer_zero_target
                        ),
                        "initializer_zero_target_passed": (
                            initializer_zero_target_passed
                        ),
                        "decision": _json_safe(asdict(decision)),
                    }
                )
    failed = [
        {
            "group": entry["group"],
            "observable": entry["observable"],
            "index": entry["index"],
            "time": entry["time"],
            "reasons": entry["decision"]["reasons"],
            "initializer_zero_target_passed": (
                entry["initializer_zero_target_passed"]
            ),
        }
        for entry in entries
        if (
            entry["decision"]["protocol_no_go"]
            or not entry["initializer_zero_target_passed"]
        )
    ]
    return {
        "passed": not failed,
        "entry_count": len(entries),
        "failed": failed,
        "entries": entries,
    }


def aggregate_frozen_histories(
    observations: Mapping[str, CompatibleHistoryObservation],
    contract: Mapping[str, Any],
    *,
    execution_code_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Construct the preregistered same-space error balls and decision."""

    cases = _validated_registry(contract)
    expected_names = tuple(case.name for case in cases)
    if tuple(observations) != expected_names:
        raise ReachablePressureV2GuardError(
            "aggregate requires all 31 frozen histories in registry order"
        )
    values = dict(observations)
    mass = values["A_plus"].mass_active
    if any(
        not np.array_equal(observation.mass_active, mass)
        for observation in values.values()
    ):
        raise ReachablePressureV2GuardError(
            "all histories must share the exact same active-P2 mass"
        )

    configurations = frozen_tangent_configurations()
    tangents: dict[str, np.ndarray] = {}
    direct_tangents: dict[str, np.ndarray] = {}
    for name, configuration in configurations.items():
        plus = values[f"{name}_plus"]
        minus = values[f"{name}_minus"]
        tangents[name] = centered_tangent(
            plus.stored_window_residual,
            minus.stored_window_residual,
            configuration.epsilon,
        )
        direct_tangents[name] = centered_tangent(
            plus.direct_window_residual,
            minus.direct_window_residual,
            configuration.epsilon,
        )

    repeat_tangent = centered_tangent(
        values["REPEAT_A_plus"].stored_window_residual,
        values["REPEAT_A_minus"].stored_window_residual,
        configurations["A"].epsilon,
    )
    epsilon_report = second_order_richardson(
        fine=tangents["A"],
        medium=tangents["E2"],
        coarse=tangents["E4"],
        mass=mass,
        contraction_ratio_min=float(
            contract["uncertainty"]["epsilon_and_timestep"][
                "contraction_ratio_min"
            ]
        ),
    )
    timestep_report = second_order_richardson(
        fine=tangents["A"],
        medium=tangents["DT2"],
        coarse=tangents["DT4"],
        mass=mass,
        contraction_ratio_min=float(
            contract["uncertainty"]["epsilon_and_timestep"][
                "contraction_ratio_min"
            ]
        ),
    )
    quadrature_report = contracting_quadrature_tail(
        coarse=tangents["Q8"],
        medium=tangents["Q10"],
        fine=tangents["A"],
        mass=mass,
    )
    cube_report = adjacent_mixed_cube(
        {
            bits: tangents[name]
            for bits, name in mixed_cube_configuration_names().items()
        },
        mass,
    )
    u_epsilon = _richardson_allowance_without_round(
        epsilon_report, tangents["A"], mass
    )
    u_timestep = _richardson_allowance_without_round(
        timestep_report, tangents["A"], mass
    )
    u_quadrature = _quadrature_allowance(quadrature_report)
    u_mixed = cube_report.allowance
    u_repeat = dual_mass_norm(
        repeat_tangent - tangents["A"], mass
    )
    u_algebraic = dual_mass_norm(
        direct_tangents["A"] - tangents["A"], mass
    )
    u_round = floating_plateau_floor(
        list(tangents.values())
        + [repeat_tangent, direct_tangents["A"]],
        mass,
    )
    u_omega = sum(
        (
            u_epsilon,
            u_timestep,
            u_quadrature,
            u_mixed,
            u_repeat,
            u_algebraic,
            u_round,
        )
    )
    omega_norm = dual_mass_norm(tangents["A"], mass)
    omega_interval = _finite_interval(omega_norm, u_omega)

    zero_vectors = {
        name: values[name].stored_window_residual
        for name in (
            "Z_A",
            "Z_DT2",
            "Z_DT4",
            "Z_Q10",
            "Z_Q8",
            "Z_DT2_Q10",
        )
    }
    zero_direct = {
        name: values[name].direct_window_residual
        for name in zero_vectors
    }
    z_dt_report = second_order_richardson(
        fine=zero_vectors["Z_A"],
        medium=zero_vectors["Z_DT2"],
        coarse=zero_vectors["Z_DT4"],
        mass=mass,
        contraction_ratio_min=float(
            contract["uncertainty"]["epsilon_and_timestep"][
                "contraction_ratio_min"
            ]
        ),
    )
    z_q_report = contracting_quadrature_tail(
        coarse=zero_vectors["Z_Q8"],
        medium=zero_vectors["Z_Q10"],
        fine=zero_vectors["Z_A"],
        mass=mass,
    )
    z_mixed = (
        zero_vectors["Z_DT2_Q10"]
        - zero_vectors["Z_DT2"]
        - zero_vectors["Z_Q10"]
        + zero_vectors["Z_A"]
    )
    z_repeat = values["REPEAT_A_zero"].stored_window_residual
    u_z_dt = _richardson_allowance_without_round(
        z_dt_report, zero_vectors["Z_A"], mass
    )
    u_z_q = _quadrature_allowance(z_q_report)
    u_z_mixed = dual_mass_norm(z_mixed, mass)
    u_z_repeat = dual_mass_norm(
        z_repeat - zero_vectors["Z_A"], mass
    )
    u_z_algebraic = dual_mass_norm(
        zero_direct["Z_A"] - zero_vectors["Z_A"], mass
    )
    u_z_round = floating_plateau_floor(
        list(zero_vectors.values())
        + [z_repeat, zero_direct["Z_A"]],
        mass,
    )
    u_z = sum(
        (
            u_z_dt,
            u_z_q,
            u_z_mixed,
            u_z_repeat,
            u_z_algebraic,
            u_z_round,
        )
    )
    z_norm = dual_mass_norm(zero_vectors["Z_A"], mass)
    z_interval = _finite_interval(z_norm, u_z)

    zero_v = {
        name: values[name].stored_noncancellation
        for name in zero_vectors
    }
    zero_v_direct = {
        name: values[name].direct_noncancellation
        for name in zero_vectors
    }
    repeat_v = values["REPEAT_A_zero"].stored_noncancellation
    v_dt_report = _scalar_second_order(
        fine=zero_v["Z_A"],
        medium=zero_v["Z_DT2"],
        coarse=zero_v["Z_DT4"],
        contraction_ratio_min=float(
            contract["uncertainty"]["epsilon_and_timestep"][
                "contraction_ratio_min"
            ]
        ),
    )
    v_q_report = _scalar_quadrature_tail(
        coarse=zero_v["Z_Q8"],
        medium=zero_v["Z_Q10"],
        fine=zero_v["Z_A"],
    )
    v_mixed = abs(
        zero_v["Z_DT2_Q10"]
        - zero_v["Z_DT2"]
        - zero_v["Z_Q10"]
        + zero_v["Z_A"]
    )
    v_round = (
        4096.0
        * np.finfo(float).eps
        * max(
            *zero_v.values(),
            repeat_v,
            zero_v_direct["Z_A"],
        )
    )
    u_v = sum(
        (
            float(v_dt_report["allowance_without_round"]),
            float(v_q_report["allowance"]),
            float(v_mixed),
            abs(repeat_v - zero_v["Z_A"]),
            abs(zero_v_direct["Z_A"] - zero_v["Z_A"]),
            float(v_round),
        )
    )
    v_interval = _finite_interval(zero_v["Z_A"], u_v)

    ratio_min = float(
        contract["uncertainty"]["epsilon_and_timestep"][
            "contraction_ratio_min"
        ]
    )
    omega_projected = {
        parity: _projected_omega_summary(
            tangents,
            direct_tangents,
            repeat_tangent,
            mass,
            parity,
            ratio_min,
        )
        for parity in ("even", "odd")
    }
    zero_sum_projected = {
        parity: _projected_zero_sum_summary(
            zero_vectors,
            zero_direct,
            z_repeat,
            mass,
            parity,
            ratio_min,
        )
        for parity in ("even", "odd")
    }
    zero_noncancellation_projected = {
        parity: _projected_zero_noncancellation_summary(
            values,
            parity,
            ratio_min,
        )
        for parity in ("even", "odd")
    }
    matched_stage_parity = _matched_stage_parity_report(values, mass)
    parity_controls = manufactured_parity_control_report(mass)
    stagewise_zero_odd = stagewise_odd_noncancellation(
        values["Z_A"].stored_step_residuals,
        mass,
    )

    accounting = frozen_execution_accounting()
    actual_accounting = {
        "histories": len(values),
        "measurement_steps": sum(
            observation.case.measurement_steps
            for observation in values.values()
        ),
        "compatible_presteps": len(values),
        "marcher_steps": sum(
            observation.case.marcher_steps
            for observation in values.values()
        ),
        "half_full_solves": sum(
            observation.case.half_full_solves
            for observation in values.values()
        ),
        "observed_stages": sum(
            observation.observed_stage_count
            for observation in values.values()
        ),
    }
    # v2.1 freezes both sign controls on the nominal positive anchor.  Their
    # denominator is the larger of the corresponding correct-sign residual
    # and a same-trace float64 floor; using float.tiny would manufacture an
    # arbitrarily large separation ratio when the correct residual is zero.
    anchor_positive = values["A_plus"]
    anchor_diagnostics = anchor_positive.diagnostics
    inventory_scale = float(
        anchor_diagnostics["material_inventory_increment_abs_max"]
    )
    correct_attachment_scale = float(
        anchor_diagnostics["material_inventory_abs_max"]
    )
    wrong_birth = float(
        anchor_diagnostics["wrong_birth_inventory_increment_abs_max"]
    )
    wrong_attachment = float(
        anchor_diagnostics["wrong_attachment_inventory_abs_max"]
    )
    same_trace_round_floor = float(
        4096.0
        * np.finfo(float).eps
        * max(
            _maximum_abs(
                anchor_positive.stage_arrays.body_cut_trace
            ),
            _maximum_abs(
                anchor_positive.stage_arrays.canonical_material_release
            ),
        )
    )
    wrong_birth_denominator = max(
        inventory_scale, same_trace_round_floor
    )
    wrong_attachment_denominator = max(
        correct_attachment_scale, same_trace_round_floor
    )
    wrong_birth_ratio = (
        wrong_birth / wrong_birth_denominator
        if wrong_birth_denominator > 0.0
        else (np.inf if wrong_birth > 0.0 else 0.0)
    )
    wrong_attachment_ratio = (
        wrong_attachment / wrong_attachment_denominator
        if wrong_attachment_denominator > 0.0
        else (np.inf if wrong_attachment > 0.0 else 0.0)
    )
    separation_min = float(
        contract["hard_guards"]["thresholds"][
            "negative_control_separation_ratio_min"
        ]
    )

    checks = {
        "registry_accounting": all(
            actual_accounting[key] == expected
            for key, expected in _EXPECTED_ACCOUNTING.items()
        )
        and all(
            accounting[key] == expected
            for key, expected in _EXPECTED_ACCOUNTING.items()
        ),
        "all_history_hard_guards": all(
            all(observation.checks.values())
            for observation in values.values()
        ),
        "wrong_birth_negative_control": (
            wrong_birth > 0.0
            and wrong_birth_ratio >= separation_min
        ),
        "wrong_attachment_negative_control": (
            wrong_attachment > 0.0
            and wrong_attachment_ratio >= separation_min
        ),
        "projected_omega_even_convergence": bool(
            omega_projected["even"]["convergence_passed"]
        ),
        "projected_omega_odd_convergence": bool(
            omega_projected["odd"]["convergence_passed"]
        ),
        "projected_zero_sum_even_convergence": bool(
            zero_sum_projected["even"]["convergence_passed"]
        ),
        "projected_zero_sum_odd_convergence": bool(
            zero_sum_projected["odd"]["convergence_passed"]
        ),
        "projected_zero_noncancellation_even_convergence": bool(
            zero_noncancellation_projected["even"][
                "convergence_passed"
            ]
        ),
        "projected_zero_noncancellation_odd_convergence": bool(
            zero_noncancellation_projected["odd"][
                "convergence_passed"
            ]
        ),
        "matched_stage_projected_q_families": bool(
            matched_stage_parity["passed"]
        ),
        "manufactured_parity_controls": all(
            value
            for value in parity_controls.values()
            if isinstance(value, (bool, np.bool_))
        ),
        "omega_odd_interval_includes_zero": (
            omega_projected["odd"]["interval"]["lower"] == 0.0
        ),
        "zero_sum_odd_interval_includes_zero": (
            zero_sum_projected["odd"]["interval"]["lower"] == 0.0
        ),
        "zero_stagewise_odd_interval_includes_zero": (
            zero_noncancellation_projected["odd"]["interval"]["lower"]
            == 0.0
        ),
    }
    protocol_passed = all(checks.values())
    if not protocol_passed:
        decision = "PROTOCOL-NO-GO"
    elif (
        float(
            zero_sum_projected["even"]["interval"]["lower"] or 0.0
        )
        > 0.0
        or float(
            zero_noncancellation_projected["even"]["interval"][
                "lower"
            ]
            or 0.0
        )
        > 0.0
    ):
        decision = "ZEROTH-ORDER NAMED-LAW OBSTRUCTION"
    elif float(
        omega_projected["even"]["interval"]["lower"] or 0.0
    ) > 0.0:
        decision = (
            "FIXED-SPACE REACHABLE FIRST-ORDER OBSTRUCTION WITNESS"
        )
    else:
        decision = "NO RESOLVED WITNESS"

    case_payload = {
        name: {
            "configuration": observation.case.configuration,
            "role": observation.case.role,
            "epsilon_signed": observation.case.epsilon_signed,
            "timestep": observation.case.timestep,
            "quadrature_order": observation.case.quadrature_order,
            "case_identity": _case_identity_payload(observation.case),
            "measurement_steps": observation.case.measurement_steps,
            "observed_stages": observation.observed_stage_count,
            "stage_times": observation.stage_arrays.stage_times,
            "stage_roles": observation.stage_arrays.stage_roles,
            "measurement_times": (
                observation.stage_arrays.measurement_times
            ),
            "stored_window_residual": observation.stored_window_residual,
            "direct_window_residual": observation.direct_window_residual,
            "stored_step_residuals": (
                observation.stored_step_residuals
            ),
            "direct_step_residuals": (
                observation.direct_step_residuals
            ),
            "typed_stage_arrays": {
                "canonical_material_current_trace": (
                    observation.stage_arrays
                    .canonical_material_current_trace
                ),
                "body_cut_trace": (
                    observation.stage_arrays.body_cut_trace
                ),
                "canonical_material_release": (
                    observation.stage_arrays.canonical_material_release
                ),
                "representation_inventory": (
                    observation.stage_arrays.representation_inventory
                ),
                "stored_weak_pressure": (
                    observation.stage_arrays.stored_weak_pressure
                ),
                "direct_weak_pressure": (
                    observation.stage_arrays.direct_weak_pressure
                ),
            },
            "value_hashes": {
                "mass_active": _array_sha256(
                    observation.mass_active
                ),
                "stored_step_residuals": _array_sha256(
                    observation.stored_step_residuals
                ),
                "direct_step_residuals": _array_sha256(
                    observation.direct_step_residuals
                ),
                "stage_times": _array_sha256(
                    observation.stage_arrays.stage_times
                ),
                "canonical_material_current_trace": _array_sha256(
                    observation.stage_arrays
                    .canonical_material_current_trace
                ),
            },
            "stored_noncancellation": (
                observation.stored_noncancellation
            ),
            "direct_noncancellation": (
                observation.direct_noncancellation
            ),
            "diagnostics": observation.diagnostics,
            "checks": observation.checks,
        }
        for name, observation in values.items()
    }
    even_lower = omega_projected["even"]["interval"]["lower"]
    odd_upper = omega_projected["odd"]["interval"]["upper"]
    odd_upper_over_even_lower = (
        float(odd_upper / even_lower)
        if (
            even_lower is not None
            and even_lower > 0.0
            and odd_upper is not None
        )
        else None
    )
    result_code_fingerprints = (
        _code_fingerprints()
        if execution_code_fingerprints is None
        else dict(execution_code_fingerprints)
    )
    if (
        not result_code_fingerprints
        or not all(
            isinstance(path, str)
            and isinstance(digest, str)
            and len(digest) == 64
            for path, digest in result_code_fingerprints.items()
        )
    ):
        raise ReachablePressureV2GuardError(
            "execution code fingerprints are incomplete"
        )
    result = {
        "stage": contract["stage"],
        "protocol_version": contract["version"],
        "contract_file": CASES.name,
        "definition_chain": contract["_definition_chain"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_fingerprints": result_code_fingerprints,
        "stage_decision": decision,
        "production_activation_allowed": False,
        "fixed_space_only": True,
        "checks": checks,
        "execution_accounting": actual_accounting,
        "negative_controls": {
            "correct_inventory_increment_abs_max": inventory_scale,
            "correct_attachment_inventory_abs_max": (
                correct_attachment_scale
            ),
            "same_trace_floating_round_floor": same_trace_round_floor,
            "wrong_birth_increment_abs_max": wrong_birth,
            "wrong_birth_denominator": wrong_birth_denominator,
            "wrong_birth_separation_ratio": wrong_birth_ratio,
            "wrong_attachment_inventory_abs_max": wrong_attachment,
            "wrong_attachment_denominator": (
                wrong_attachment_denominator
            ),
            "wrong_attachment_separation_ratio": (
                wrong_attachment_ratio
            ),
        },
        "v22_span_parity": {
            "omega": omega_projected,
            "zero_sum": zero_sum_projected,
            "zero_noncancellation": (
                zero_noncancellation_projected
            ),
            "matched_stage_q_families": matched_stage_parity,
            "manufactured_controls": parity_controls,
            "zero_anchor_stagewise_odd": asdict(stagewise_zero_odd),
            "odd_upper_over_even_lower": (
                odd_upper_over_even_lower
            ),
            "symmetry_statement": (
                "NO RESOLVED SYMMETRY VIOLATION"
                if (
                    checks["omega_odd_interval_includes_zero"]
                    and checks["zero_sum_odd_interval_includes_zero"]
                    and checks[
                        "zero_stagewise_odd_interval_includes_zero"
                    ]
                    and checks["matched_stage_projected_q_families"]
                )
                else "RESOLVED OR UNCONVERGED SYMMETRY FAILURE"
            ),
        },
        "omega": {
            "role": "unprojected diagnostic only; decision uses v22_span_parity",
            "anchor_vector": tangents["A"],
            "interval": omega_interval,
            "uncertainty_components": {
                "epsilon": u_epsilon,
                "timestep": u_timestep,
                "quadrature": u_quadrature,
                "mixed_cube": u_mixed,
                "fresh_repeat": u_repeat,
                "algebraic_cross_observer": u_algebraic,
                "floating_round": u_round,
            },
            "epsilon_report": _report_dict(epsilon_report),
            "timestep_report": _report_dict(timestep_report),
            "quadrature_report": _report_dict(quadrature_report),
            "mixed_cube_report": _report_dict(cube_report),
        },
        "zero_sum": {
            "role": "unprojected diagnostic only; decision uses v22_span_parity",
            "anchor_vector": zero_vectors["Z_A"],
            "interval": z_interval,
            "uncertainty_components": {
                "timestep": u_z_dt,
                "quadrature": u_z_q,
                "mixed_corner": u_z_mixed,
                "fresh_repeat": u_z_repeat,
                "algebraic_cross_observer": u_z_algebraic,
                "floating_round": u_z_round,
            },
            "timestep_report": _report_dict(z_dt_report),
            "quadrature_report": _report_dict(z_q_report),
        },
        "zero_noncancellation": {
            "role": "unprojected diagnostic only; decision uses v22_span_parity",
            "anchor": zero_v["Z_A"],
            "interval": v_interval,
            "uncertainty_components": {
                "timestep": v_dt_report[
                    "allowance_without_round"
                ],
                "quadrature": v_q_report["allowance"],
                "mixed_corner": v_mixed,
                "fresh_repeat": abs(repeat_v - zero_v["Z_A"]),
                "algebraic_cross_observer": abs(
                    zero_v_direct["Z_A"] - zero_v["Z_A"]
                ),
                "floating_round": v_round,
            },
            "timestep_report": v_dt_report,
            "quadrature_report": v_q_report,
        },
        "cases": case_payload,
        "nonclaims": contract["nonclaims"],
    }
    return _json_safe(result)


def run_preregistered_observation() -> dict[str, Any]:
    """Fail before mesh construction while v2.2 execution is prohibited."""

    contract = _load_frozen_contract()
    if contract.get("decision", {}).get("formal_execution_allowed") is not True:
        raise ReachablePressureV2GuardError(
            "S3ai-v2.2 formal execution is fail-closed by the frozen "
            "decision.formal_execution_allowed=false contract"
        )
    start_fingerprints = _code_fingerprints()
    observations = _collect_frozen_histories(contract)
    result = aggregate_frozen_histories(
        observations,
        contract,
        execution_code_fingerprints=start_fingerprints,
    )
    end_fingerprints = _code_fingerprints()
    if end_fingerprints != start_fingerprints:
        raise ReachablePressureV2GuardError(
            "implementation source drifted during formal execution"
        )
    return result
