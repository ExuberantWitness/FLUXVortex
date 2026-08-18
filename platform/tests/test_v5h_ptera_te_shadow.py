from __future__ import annotations

import os
from typing import Any

# Ptera's Numba decorators initialize their cache during import.  Keep all
# smoke-test caches in /tmp rather than a user configuration directory.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fluxv-v5h-g4-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluxv-v5h-g4-mpl-cache")

import numpy as np
import pterasoftware as ps
import pytest

from fluxvortex.rvpm_edge_bridge import (
    DIAGNOSTIC_SHADOW_OWNER,
    RING_PHYSICAL_OWNER,
    canonical_edge_key,
)
from fluxvortex.solver import UVPMHybridSolver
from forward_flight_benchmarks.v5h_ptera_te_shadow import (
    build_ptera_te_diagnostic_shadow,
)


class _ExplodingInput:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"disabled adapter read input attribute {name}")

    def __iter__(self) -> object:
        raise AssertionError("disabled adapter iterated an input")

    def __array__(self) -> object:
        raise AssertionError("disabled adapter converted an input")

    def __int__(self) -> int:
        raise AssertionError("disabled adapter converted an input")


def _generic_non_target_problem() -> ps.problems.UnsteadyProblem:
    """Return a tiny rectangular geometry used only for Ptera API mechanics."""

    # A 1%-thick symmetric numerical surface keeps this API smoke close to a
    # plate without importing any target-paper geometry or measurements.
    airfoil = ps.geometry.airfoil.Airfoil(name="naca0001")
    sections = [
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=0.2,
            Lp_Wcsp_Lpp=(0.0, 0.0, 0.0),
            num_spanwise_panels=2,
            spanwise_spacing="uniform",
        ),
        ps.geometry.wing_cross_section.WingCrossSection(
            airfoil=airfoil,
            chord=0.2,
            Lp_Wcsp_Lpp=(0.0, 0.4, 0.0),
            num_spanwise_panels=None,
            spanwise_spacing=None,
        ),
    ]
    wing = ps.geometry.wing.Wing(
        name="v5h G4 generic API rectangle",
        wing_cross_sections=sections,
        symmetric=False,
        num_chordwise_panels=1,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name="v5h G4 generic API airplane",
        s_ref=0.08,
        c_ref=0.2,
        b_ref=0.4,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in sections
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[wing_movement],
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=1.2,
        vCg__E=2.0,
        alpha=4.0,
        beta=0.0,
        nu=1.5e-5,
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=(
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point
            )
        ),
        delta_time=0.02,
        num_steps=3,
        max_wake_rows=2,
    )
    return ps.problems.UnsteadyProblem(
        movement=movement,
        only_final_results=False,
    )


class _ShadowRecordingSolver(UVPMHybridSolver):
    def __init__(self, unsteady_problem: Any) -> None:
        self.current_shadows: list[Any] = []
        self.next_shadows: list[Any] = []
        self.next_precollapse_flat_strengths: list[np.ndarray] = []
        super().__init__(
            unsteady_problem,
            max_particles=100,
            stretch=False,
            free_wake=False,
        )

    def _populate_next_airplanes_wake(self) -> None:
        if self._current_step > 0:
            self.current_shadows.append(
                build_ptera_te_diagnostic_shadow(
                    self,
                    time_layer="current",
                    subdivisions=2,
                )
            )

        super()._populate_next_airplanes_wake()

        if self._current_step < self.num_steps - 1:
            # This Ptera allocation is intentionally captured, not passed to
            # the adapter.  It is stale until the next _collapse_geometry().
            self.next_precollapse_flat_strengths.append(
                self._list_wake_vortex_strengths[self._current_step + 1].copy()
            )
            self.next_shadows.append(
                build_ptera_te_diagnostic_shadow(
                    self,
                    time_layer="next",
                    subdivisions=2,
                )
            )


def _run_parent() -> UVPMHybridSolver:
    solver = UVPMHybridSolver(
        _generic_non_target_problem(),
        max_particles=100,
        stretch=False,
        free_wake=False,
    )
    # FluxV's current solver forces its Ptera wake back to prescribed motion.
    solver.run(
        prescribed_wake=False,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _run_recording() -> _ShadowRecordingSolver:
    solver = _ShadowRecordingSolver(_generic_non_target_problem())
    solver.run(
        prescribed_wake=False,
        calculate_streamlines=False,
        show_progress=False,
    )
    return solver


def _array_record(name: str, value: object) -> tuple[str, str, tuple[int, ...], bytes]:
    array = np.asarray(value)
    return name, array.dtype.str, array.shape, array.tobytes(order="C")


def _native_state_and_load_bits(
    solver: UVPMHybridSolver,
) -> tuple[tuple[str, str, tuple[int, ...], bytes], ...]:
    records: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for attribute in (
        "_current_bound_vortex_strengths",
        "_current_wake_vortex_strengths",
        "_current_wake_vortex_ages",
        "_currentStackBrwrvp_GP1_CgP1",
        "_currentStackFrwrvp_GP1_CgP1",
        "_currentStackFlwrvp_GP1_CgP1",
        "_currentStackBlwrvp_GP1_CgP1",
    ):
        records.append(_array_record(f"solver.{attribute}", getattr(solver, attribute)))

    for history_name in (
        "_list_wake_vortex_strengths",
        "_list_wake_vortex_ages",
        "listStackBrwrvp_GP1_CgP1",
        "listStackFrwrvp_GP1_CgP1",
        "listStackFlwrvp_GP1_CgP1",
        "listStackBlwrvp_GP1_CgP1",
    ):
        for step, value in enumerate(getattr(solver, history_name)):
            records.append(_array_record(f"{history_name}[{step}]", value))

    for step, problem in enumerate(solver.steady_problems):
        for airplane_index, airplane in enumerate(problem.airplanes):
            for attribute in (
                "forces_W",
                "forceCoefficients_W",
                "moments_W_CgP1",
                "momentCoefficients_W_CgP1",
            ):
                records.append(
                    _array_record(
                        f"step{step}.airplane{airplane_index}.{attribute}",
                        getattr(airplane, attribute),
                    )
                )
            for wing_index, wing in enumerate(airplane.wings):
                records.append(
                    _array_record(
                        f"step{step}.wing{wing_index}.grid",
                        wing.gridWrvp_GP1_CgP1,
                    )
                )
                wake_strengths = np.asarray(
                    [ring.strength for ring in wing.wake_ring_vortices.ravel()]
                )
                wake_ages = np.asarray(
                    [ring.age for ring in wing.wake_ring_vortices.ravel()]
                )
                records.append(
                    _array_record(
                        f"step{step}.wing{wing_index}.wake_strengths",
                        wake_strengths,
                    )
                )
                records.append(
                    _array_record(f"step{step}.wing{wing_index}.wake_ages", wake_ages)
                )
                for panel_index, panel in enumerate(np.ravel(wing.panels)):
                    records.append(
                        _array_record(
                            f"step{step}.wing{wing_index}.panel{panel_index}.gamma",
                            np.asarray([panel.ring_vortex.strength]),
                        )
                    )
                    for attribute in (
                        "forces_GP1",
                        "moments_GP1_CgP1",
                        "forces_W",
                        "moments_W_CgP1",
                    ):
                        records.append(
                            _array_record(
                                (
                                    f"step{step}.wing{wing_index}.panel"
                                    f"{panel_index}.{attribute}"
                                ),
                                getattr(panel, attribute),
                            )
                        )

    count = solver._vpm_field.np
    for attribute in ("_pos", "_gamma", "_sigma", "_age"):
        records.append(
            _array_record(
                f"vpm.{attribute}", getattr(solver._vpm_field, attribute)[:count]
            )
        )
    return tuple(records)


def test_g4_disabled_adapter_is_input_blind_and_has_no_owner_or_feedback() -> None:
    exploding = _ExplodingInput()
    result = build_ptera_te_diagnostic_shadow(
        exploding,
        time_layer=exploding,
        airplane_index=exploding,
        wing_index=exploding,
        subdivisions=exploding,
        enabled=False,
    )

    assert result.provenance is None
    assert result.source_ring_ids == ()
    assert result.source_ring_node_ids == ()
    assert result.source_strengths_m2_s == ()
    assert result.bridge.feedback_velocity is None
    assert not result.bridge.diagnostics.enabled
    assert result.diagnostics.parent_write_count == 0
    assert result.diagnostics.load_write_count == 0
    assert result.diagnostics.feedback_call_count == 0


def test_g4_actual_ptera_current_next_roundtrip_is_read_only_and_bitwise_parent() -> (
    None
):
    parent = _run_parent()
    candidate = _run_recording()

    assert candidate._prescribed_wake is True
    assert _native_state_and_load_bits(candidate) == _native_state_and_load_bits(parent)
    assert len(candidate.next_shadows) == 2
    assert len(candidate.current_shadows) == 2

    next_step_one = candidate.next_shadows[0]
    current_step_one = candidate.current_shadows[0]
    assert next_step_one.diagnostics.source_step == 1
    assert current_step_one.diagnostics.source_step == 1
    assert "pre-collapse object fact source" in (
        next_step_one.diagnostics.strength_fact_source
    )
    assert "post-collapse current arrays" in (
        current_step_one.diagnostics.strength_fact_source
    )

    # The just-populated next object row is nonzero while its preallocated flat
    # array is still zero.  The exact next/current roundtrip demonstrates that
    # the adapter used the object fact source and that collapse later agrees.
    stale_next = candidate.next_precollapse_flat_strengths[0]
    assert np.all(stale_next == 0.0)
    assert np.any(np.asarray(next_step_one.source_strengths_m2_s) != 0.0)
    assert next_step_one.source_ring_ids == current_step_one.source_ring_ids
    assert next_step_one.source_ring_node_ids == current_step_one.source_ring_node_ids
    assert next_step_one.source_strengths_m2_s == (
        current_step_one.source_strengths_m2_s
    )
    np.testing.assert_array_equal(
        next_step_one.bridge.positions, current_step_one.bridge.positions
    )
    np.testing.assert_array_equal(
        next_step_one.bridge.gamma, current_step_one.bridge.gamma
    )
    np.testing.assert_array_equal(
        next_step_one.bridge.sigma, current_step_one.bridge.sigma
    )
    assert next_step_one.bridge.particle_ids == current_step_one.bridge.particle_ids

    assert next_step_one.diagnostics.prescribed_wake is True
    assert next_step_one.diagnostics.physical_owner == RING_PHYSICAL_OWNER
    assert next_step_one.diagnostics.shadow_owner == DIAGNOSTIC_SHADOW_OWNER
    assert next_step_one.diagnostics.strength_layer_verified
    assert next_step_one.diagnostics.ring_object_geometry_verified
    assert next_step_one.diagnostics.source_kernel_channel_count == 1
    assert next_step_one.diagnostics.heterogeneous_kernel_merge_count == 0
    assert next_step_one.diagnostics.source_shadow_merge_count == 0
    assert next_step_one.bridge.feedback_velocity is None
    assert next_step_one.bridge.diagnostics.feedback_call_count == 0
    assert all(
        lineage.physical_owner == RING_PHYSICAL_OWNER
        and lineage.owner_state == DIAGNOSTIC_SHADOW_OWNER
        for lineage in next_step_one.bridge.lineage
    )

    provenance = next_step_one.provenance
    assert provenance is not None
    assert provenance.source_kernel_family == (
        "ptera_modified_lamb_oseen_finite_segment"
    )
    assert provenance.shadow_kernel_family == "flowvpm_gaussian_erf_direct"
    assert provenance.source_age_s == 0.0
    assert not provenance.source_shadow_merge_allowed
    assert "abs(gamma)" in provenance.source_core_law

    # Two spanwise-adjacent rings share one explicit physical edge.  Its two
    # opposite incidences occupy one ledger, while the supplied ring traversal
    # remains BR -> FR -> FL -> BL.
    assert len(next_step_one.source_ring_node_ids) == 2
    first_nodes = next_step_one.source_ring_node_ids[0]
    second_nodes = next_step_one.source_ring_node_ids[1]
    assert first_nodes[0] == second_nodes[3]
    assert first_nodes[1] == second_nodes[2]
    shared_key = canonical_edge_key(first_nodes[0], first_nodes[1])
    shared_ledgers = [
        edge for edge in next_step_one.bridge.edge_graph.edges if edge.key == shared_key
    ]
    assert len(shared_ledgers) == 1
    assert len(shared_ledgers[0].incidences) == 2
    assert {
        (item.ring_id, item.traversal_index) for item in shared_ledgers[0].incidences
    } == {
        (next_step_one.source_ring_ids[0], 0),
        (next_step_one.source_ring_ids[1], 2),
    }


def test_g4_mismatched_time_layer_and_heterogeneous_kernel_fail_closed() -> None:
    solver = _run_recording()

    original_strength = solver._current_wake_vortex_strengths[0]
    solver._current_wake_vortex_strengths[0] = original_strength + 1.0
    try:
        with pytest.raises(RuntimeError, match="strengths disagree"):
            build_ptera_te_diagnostic_shadow(
                solver,
                time_layer="current",
                subdivisions=2,
            )
    finally:
        solver._current_wake_vortex_strengths[0] = original_strength

    current_wing = solver.current_airplanes[0].wings[0]
    first_ring = current_wing.wake_ring_vortices[0, 0]
    original_age = first_ring.age
    original_flat_age = solver._current_wake_vortex_ages[0]
    first_ring.age = 0.01
    solver._current_wake_vortex_ages[0] = 0.01
    try:
        with pytest.raises(RuntimeError, match="heterogeneous Ptera kernel channels"):
            build_ptera_te_diagnostic_shadow(
                solver,
                time_layer="current",
                subdivisions=2,
            )
    finally:
        first_ring.age = original_age
        solver._current_wake_vortex_ages[0] = original_flat_age


def test_g4_rejects_free_wake_claim_and_missing_next_layer() -> None:
    solver = _run_recording()
    solver._prescribed_wake = False
    with pytest.raises(RuntimeError, match="prescribed Ptera wake"):
        build_ptera_te_diagnostic_shadow(
            solver,
            time_layer="current",
            subdivisions=2,
        )

    solver._prescribed_wake = True
    with pytest.raises(ValueError, match="next time layer does not exist"):
        build_ptera_te_diagnostic_shadow(
            solver,
            time_layer="next",
            subdivisions=2,
        )
