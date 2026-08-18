"""Tests for the FluxV v5h8 causal incremental connected-sheet oracle.

The tests deliberately distinguish three representations:

* a direct connected graph, which is the zero-transport reference;
* the append-only live-material basis, whose cancelling boundary copies keep
  the old particle prefix immutable; and
* a freshly redeposited geometry, which is a negative control after material
  transport and must not be advertised as the live-material comparator.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from math import ceil
from pathlib import Path
import sys

import numpy as np
import pytest

from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DirectedRing,
    assemble_ring_edge_graph,
    canonical_edge_key,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "forward_flight_benchmarks"
    / "fluxv_v5h8_incremental_sheet.py"
)
_MODULES_BEFORE_ISOLATED_IMPORT = frozenset(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_fluxv_v5h8_incremental_sheet_under_test", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load the isolated v5h8 module")
sheet = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sheet
_SPEC.loader.exec_module(sheet)
_MODULES_ADDED_BY_ISOLATED_IMPORT = frozenset(sys.modules).difference(
    _MODULES_BEFORE_ISOLATED_IMPORT
)


SPAN_M = 0.60
PANEL_STEP_M = 0.04
DELTA_GAMMA_M2_PER_S = 0.016
SMOOTHING_RADIUS_M = 0.085
TARGET_SPACING_M = 0.02
PARTICLE_CAP = 1_000
FIELD_IMPULSE_TOLERANCE = 2.0e-14
CLONE_TOLERANCE = 1.0e-14
FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE = 1.0e-8
PROBES_M = np.asarray(
    ((0.05, 0.30, 0.30), (0.10, 0.15, 0.40), (0.20, 0.45, 0.50)),
    dtype=np.float64,
)
AFFINE_MATRIX = np.asarray(
    ((1.0, 0.015, 0.0), (0.0, 1.0, 0.01), (0.02, 0.0, 0.99)),
    dtype=np.float64,
)
AFFINE_TRANSLATION_M = np.asarray((0.04, -0.002, 0.003), dtype=np.float64)
AFFINE_SIGMA_SCALE = 1.01


def _first_panel() -> sheet.SheetPanel:
    return sheet.make_panel(
        (0.0, 0.0, 0.0),
        (0.0, SPAN_M, 0.0),
        (PANEL_STEP_M, 0.0, 0.0),
        (PANEL_STEP_M, SPAN_M, 0.0),
        DELTA_GAMMA_M2_PER_S,
        release_index=1,
    )


def test_isolated_module_import_does_not_load_ptera_or_benchmark_package() -> None:
    assert not any(
        name == "pterasoftware" or name.startswith("pterasoftware.")
        for name in _MODULES_ADDED_BY_ISOLATED_IMPORT
    )
    assert "forward_flight_benchmarks" not in _MODULES_ADDED_BY_ISOLATED_IMPORT


def _start() -> sheet.AppendResult:
    return sheet.start_incremental_sheet(
        _first_panel(),
        SMOOTHING_RADIUS_M,
        TARGET_SPACING_M,
        particle_cap=PARTICLE_CAP,
    )


def _zero_transport_state(release_count: int) -> sheet.IncrementalSheetState:
    if release_count < 1:
        raise ValueError("release_count must be positive")
    result = _start()
    for release_index in range(2, release_count + 1):
        old = result.state
        old_positions = old.positions.copy()
        old_gamma = old.gamma.copy()
        old_sigma = old.sigma.copy()
        old_particle_ids = old.particle_ids
        old_lineage = old.lineage
        result = sheet.append_live_basis_panel(
            old,
            (release_index * PANEL_STEP_M, 0.0, 0.0),
            (release_index * PANEL_STEP_M, SPAN_M, 0.0),
            release_index * DELTA_GAMMA_M2_PER_S,
            particle_cap=PARTICLE_CAP,
        )
        assert result.diagnostics.prefix_bitwise_unchanged
        assert np.array_equal(
            result.state.positions[: len(old_positions)], old_positions
        )
        assert np.array_equal(result.state.gamma[: len(old_gamma)], old_gamma)
        assert np.array_equal(result.state.sigma[: len(old_sigma)], old_sigma)
        assert (
            result.state.positions[: len(old_positions)].tobytes()
            == old_positions.tobytes()
        )
        assert result.state.gamma[: len(old_gamma)].tobytes() == old_gamma.tobytes()
        assert result.state.sigma[: len(old_sigma)].tobytes() == old_sigma.tobytes()
        assert result.state.particle_ids[: len(old_particle_ids)] == old_particle_ids
        assert result.state.lineage[: len(old_lineage)] == old_lineage
    return result.state


def _affine_point(point: np.ndarray) -> np.ndarray:
    return AFFINE_MATRIX @ point + AFFINE_TRANSLATION_M


def _pair_indices(pair: object) -> tuple[int, int]:
    """Accept the frozen two-index tuple and provide a useful assertion error."""

    try:
        old_index, clone_index = pair  # type: ignore[misc]
    except (TypeError, ValueError) as error:
        raise AssertionError(
            "clone_pairs entries must be (old_index, clone_index)"
        ) from error
    assert isinstance(old_index, (int, np.integer))
    assert isinstance(clone_index, (int, np.integer))
    return int(old_index), int(clone_index)


def _affine_transport_state(release_count: int) -> sheet.IncrementalSheetState:
    if release_count < 1:
        raise ValueError("release_count must be positive")
    state = _start().state
    live_left = np.asarray((PANEL_STEP_M, 0.0, 0.0), dtype=np.float64)
    live_right = np.asarray((PANEL_STEP_M, SPAN_M, 0.0), dtype=np.float64)

    for release_index in range(2, release_count + 1):
        birth_particle_ids = state.particle_ids
        birth_lineage = state.lineage
        state = sheet.affine_transport_state(
            state,
            AFFINE_MATRIX,
            AFFINE_TRANSLATION_M,
            AFFINE_SIGMA_SCALE,
        )
        assert state.particle_ids == birth_particle_ids
        assert state.lineage == birth_lineage
        live_left = _affine_point(live_left)
        live_right = _affine_point(live_right)
        downstream_left = live_left + np.asarray((PANEL_STEP_M, 0.0, 0.0))
        downstream_right = live_right + np.asarray((PANEL_STEP_M, 0.0, 0.0))

        prefix_positions = state.positions.copy()
        prefix_gamma = state.gamma.copy()
        prefix_sigma = state.sigma.copy()
        prefix_particle_ids = state.particle_ids
        prefix_lineage = state.lineage
        prior_pair_count = len(state.clone_pairs)
        result = sheet.append_live_basis_panel(
            state,
            downstream_left,
            downstream_right,
            release_index * DELTA_GAMMA_M2_PER_S,
            particle_cap=PARTICLE_CAP,
        )
        state = result.state

        assert result.diagnostics.prefix_bitwise_unchanged
        assert np.array_equal(
            state.positions[: len(prefix_positions)], prefix_positions
        )
        assert np.array_equal(state.gamma[: len(prefix_gamma)], prefix_gamma)
        assert np.array_equal(state.sigma[: len(prefix_sigma)], prefix_sigma)
        assert (
            state.positions[: len(prefix_positions)].tobytes()
            == prefix_positions.tobytes()
        )
        assert state.gamma[: len(prefix_gamma)].tobytes() == prefix_gamma.tobytes()
        assert state.sigma[: len(prefix_sigma)].tobytes() == prefix_sigma.tobytes()
        assert state.particle_ids[: len(prefix_particle_ids)] == prefix_particle_ids
        assert state.lineage[: len(prefix_lineage)] == prefix_lineage
        assert result.diagnostics.max_clone_position_abs <= CLONE_TOLERANCE
        assert result.diagnostics.max_clone_sigma_abs <= CLONE_TOLERANCE
        assert result.diagnostics.max_clone_gamma_relation_abs <= CLONE_TOLERANCE
        assert result.diagnostics.excluded_fresh_upstream_count > 0

        new_pairs = state.clone_pairs[prior_pair_count:]
        assert new_pairs
        expected_scale = -release_index / (release_index - 1)
        for pair in new_pairs:
            old_index, clone_index = _pair_indices(pair)
            assert np.array_equal(
                state.positions[clone_index], state.positions[old_index]
            )
            assert state.sigma[clone_index] == state.sigma[old_index]
            assert (
                np.max(
                    np.abs(
                        state.gamma[clone_index]
                        - expected_scale * state.gamma[old_index]
                    )
                )
                <= CLONE_TOLERANCE
            )

        live_left = downstream_left
        live_right = downstream_right
    return state


def _assert_particle_contract(value: object) -> None:
    positions = value.positions  # type: ignore[attr-defined]
    gamma = value.gamma  # type: ignore[attr-defined]
    sigma = value.sigma  # type: ignore[attr-defined]
    assert positions.dtype == np.float64
    assert gamma.dtype == np.float64
    assert sigma.dtype == np.float64
    assert positions.ndim == 2 and positions.shape[1:] == (3,)
    assert gamma.shape == positions.shape
    assert sigma.shape == (positions.shape[0],)
    assert positions.flags.c_contiguous
    assert gamma.flags.c_contiguous
    assert sigma.flags.c_contiguous
    assert not positions.flags.writeable
    assert not gamma.flags.writeable
    assert not sigma.flags.writeable
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


@pytest.mark.parametrize("release_count", (1, 2, 3, 4))
def test_zero_transport_append_matches_direct_connected_graph(
    release_count: int,
) -> None:
    state = _zero_transport_state(release_count)
    direct = sheet.direct_connected_redeposit(
        state.panels,
        SMOOTHING_RADIUS_M,
        TARGET_SPACING_M,
    )
    incremental_field = sheet.particle_velocity(state, PROBES_M)
    direct_field = sheet.particle_velocity(direct, PROBES_M)
    incremental_impulse = sheet.particle_impulse(state)
    direct_impulse = sheet.particle_impulse(direct)

    assert np.max(np.abs(incremental_field - direct_field)) <= FIELD_IMPULSE_TOLERANCE
    assert (
        np.max(np.abs(incremental_impulse - direct_impulse)) <= FIELD_IMPULSE_TOLERANCE
    )
    expected_impulse = np.asarray(
        (
            0.0,
            0.0,
            -SPAN_M
            * PANEL_STEP_M
            * DELTA_GAMMA_M2_PER_S
            * release_count
            * (release_count + 1)
            / 2.0,
        )
    )
    assert direct_impulse == pytest.approx(expected_impulse, abs=1.0e-14)
    assert state.positions.shape[0] <= PARTICLE_CAP
    _assert_particle_contract(state)
    _assert_particle_contract(direct)


@pytest.mark.parametrize("release_count", (2, 3, 4))
def test_affine_live_basis_collapse_is_exact_but_fresh_geometry_is_not(
    release_count: int,
) -> None:
    state = _affine_transport_state(release_count)
    consolidated = sheet.collapse_live_basis_pairs(state)
    fresh = sheet.fresh_geometry_redeposit(state)

    live_field = sheet.particle_velocity(state, PROBES_M)
    consolidated_field = sheet.particle_velocity(consolidated, PROBES_M)
    live_impulse = sheet.particle_impulse(state)
    consolidated_impulse = sheet.particle_impulse(consolidated)
    fresh_field = sheet.particle_velocity(fresh, PROBES_M)

    assert np.max(np.abs(live_field - consolidated_field)) <= FIELD_IMPULSE_TOLERANCE
    assert (
        np.max(np.abs(live_impulse - consolidated_impulse)) <= FIELD_IMPULSE_TOLERANCE
    )
    relative_fresh_difference = float(
        np.linalg.norm(live_field - fresh_field) / np.linalg.norm(live_field)
    )
    assert relative_fresh_difference > FRESH_REDEPOSITION_MIN_RELATIVE_DIFFERENCE
    assert consolidated.positions.shape[0] == (
        state.positions.shape[0] - len(state.clone_pairs)
    )
    assert state.positions.shape[0] <= PARTICLE_CAP
    _assert_particle_contract(state)
    _assert_particle_contract(consolidated)
    _assert_particle_contract(fresh)


def test_state_is_readonly_finite_unique_and_index_closed() -> None:
    state = _affine_transport_state(4)
    _assert_particle_contract(state)
    assert len(state.particle_ids) == state.positions.shape[0]
    assert len(state.lineage) == state.positions.shape[0]
    assert len(set(state.particle_ids)) == len(state.particle_ids)
    assert len(state.panels) == 4
    assert len(state.downstream_particle_indices) > 0
    assert len(set(state.downstream_particle_indices)) == len(
        state.downstream_particle_indices
    )
    assert all(
        0 <= int(index) < state.positions.shape[0]
        for index in state.downstream_particle_indices
    )
    for pair in state.clone_pairs:
        old_index, clone_index = _pair_indices(pair)
        assert 0 <= old_index < clone_index < state.positions.shape[0]
    with pytest.raises(ValueError):
        state.positions[0, 0] = 0.0
    with pytest.raises(ValueError):
        state.gamma[0, 0] = 0.0
    with pytest.raises(ValueError):
        state.sigma[0] = SMOOTHING_RADIUS_M
    assert np.all(np.isfinite(sheet.particle_velocity(state, PROBES_M)))
    assert np.all(np.isfinite(sheet.particle_impulse(state)))


def test_independent_pair_reduction_and_perturbation_sensitivity() -> None:
    state = _affine_transport_state(4)
    clone_to_old = {
        _pair_indices(pair)[1]: _pair_indices(pair)[0] for pair in state.clone_pairs
    }
    removed = set(clone_to_old)
    expected_gamma = state.gamma.copy()
    for clone_index, old_index in clone_to_old.items():
        expected_gamma[old_index] = state.gamma[old_index] + state.gamma[clone_index]
    kept = tuple(index for index in range(len(state.positions)) if index not in removed)
    expected_field = direct_gaussian_erf_velocity_jacobian(
        state.positions[list(kept)],
        expected_gamma[list(kept)],
        state.sigma[list(kept)],
        target_positions=PROBES_M,
    ).velocity
    consolidated = sheet.collapse_live_basis_pairs(state)
    assert (
        np.max(np.abs(sheet.particle_velocity(consolidated, PROBES_M) - expected_field))
        <= FIELD_IMPULSE_TOLERANCE
    )

    tampered = state.gamma.copy()
    first_clone = next(iter(clone_to_old))
    tampered[first_clone, 0] += 1.0e-9
    immutable_tampered = np.frombuffer(tampered.tobytes(), dtype=np.float64).reshape(
        tampered.shape
    )
    forged = replace(state, gamma=immutable_tampered)
    with pytest.raises(ValueError, match="clone gamma"):
        sheet.collapse_live_basis_pairs(forged)


def test_nonaffine_map_breaks_material_midpoint_and_tangent_redeposition() -> None:
    start = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    end = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)

    def quadratic_warp(point: np.ndarray) -> np.ndarray:
        warped = point.copy()
        warped[0] += 0.2 * point[1] ** 2
        return warped

    material_midpoint = quadratic_warp(0.5 * (start + end))
    fresh_geometry_midpoint = 0.5 * (quadratic_warp(start) + quadratic_warp(end))
    assert np.linalg.norm(material_midpoint - fresh_geometry_midpoint) == pytest.approx(
        0.05
    )

    def cubic_warp(point: np.ndarray) -> np.ndarray:
        warped = point.copy()
        warped[0] += 0.2 * point[1] ** 3
        return warped

    midpoint = 0.5 * (start + end)
    midpoint_jacobian = np.asarray(
        ((1.0, 0.6 * midpoint[1] ** 2, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    transported_tangent = midpoint_jacobian @ (end - start)
    fresh_geometry_tangent = cubic_warp(end) - cubic_warp(start)
    assert np.linalg.norm(
        transported_tangent - fresh_geometry_tangent
    ) == pytest.approx(0.05)

    gamma = np.asarray(((0.0, DELTA_GAMMA_M2_PER_S, 0.0),), dtype=np.float64)
    sigma = np.asarray((SMOOTHING_RADIUS_M,), dtype=np.float64)
    material_field = direct_gaussian_erf_velocity_jacobian(
        material_midpoint[None, :], gamma, sigma, target_positions=PROBES_M
    ).velocity
    redeposited_field = direct_gaussian_erf_velocity_jacobian(
        fresh_geometry_midpoint[None, :], gamma, sigma, target_positions=PROBES_M
    ).velocity
    assert np.linalg.norm(material_field - redeposited_field) > 1.0e-8


def _threshold_edge_particle_count(length: float) -> int:
    nodes = (
        BridgeNode("a", (0.0, 0.0, 0.0)),
        BridgeNode("b", (length, 0.0, 0.0)),
        BridgeNode("c", (length, 0.1, 0.0)),
        BridgeNode("d", (0.0, 0.1, 0.0)),
    )
    graph = assemble_ring_edge_graph(
        nodes,
        (DirectedRing("threshold", ("a", "b", "c", "d"), 0.016),),
    )
    result = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=SMOOTHING_RADIUS_M,
        target_spacing=TARGET_SPACING_M,
        step=0,
    )
    selected = canonical_edge_key("a", "b")
    return sum(item.source_edge == selected for item in result.lineage)


def test_ceil_count_changes_across_an_arbitrarily_small_geometry_threshold() -> None:
    below = 2.0 * TARGET_SPACING_M - 1.0e-12
    above = 2.0 * TARGET_SPACING_M + 1.0e-12
    assert ceil(below / TARGET_SPACING_M) == 2
    assert ceil(above / TARGET_SPACING_M) == 3
    assert _threshold_edge_particle_count(below) == 2
    assert _threshold_edge_particle_count(above) == 3


def test_particle_cap_failure_rolls_back_without_touching_parent() -> None:
    state = _start().state
    positions = state.positions.copy()
    gamma = state.gamma.copy()
    sigma = state.sigma.copy()
    particle_ids = state.particle_ids
    lineage = state.lineage
    panels = state.panels
    clone_pairs = state.clone_pairs
    with pytest.raises(RuntimeError, match="cap"):
        sheet.append_live_basis_panel(
            state,
            (2.0 * PANEL_STEP_M, 0.0, 0.0),
            (2.0 * PANEL_STEP_M, SPAN_M, 0.0),
            2.0 * DELTA_GAMMA_M2_PER_S,
            particle_cap=state.positions.shape[0],
        )
    assert np.array_equal(state.positions, positions)
    assert np.array_equal(state.gamma, gamma)
    assert np.array_equal(state.sigma, sigma)
    assert state.particle_ids == particle_ids
    assert state.lineage == lineage
    assert state.panels is panels
    assert state.clone_pairs is clone_pairs


@pytest.mark.parametrize(
    "factory",
    (
        lambda: sheet.make_panel(
            (np.nan, 0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            (PANEL_STEP_M, 0.0, 0.0),
            (PANEL_STEP_M, SPAN_M, 0.0),
            DELTA_GAMMA_M2_PER_S,
            release_index=1,
        ),
        lambda: sheet.make_panel(
            (0.0, 0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            (PANEL_STEP_M, 0.0, 0.0),
            (PANEL_STEP_M, SPAN_M, 0.0),
            np.inf,
            release_index=1,
        ),
        lambda: sheet.make_panel(
            (0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            (PANEL_STEP_M, 0.0, 0.0),
            (PANEL_STEP_M, SPAN_M, 0.0),
            DELTA_GAMMA_M2_PER_S,
            release_index=1,
        ),
        lambda: sheet.make_panel(
            (0.0, 0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            (PANEL_STEP_M, 0.0, 0.0),
            (PANEL_STEP_M, SPAN_M, 0.0),
            True,
            release_index=1,
        ),
    ),
)
def test_bad_panel_inputs_fail_closed(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_bow_tie_and_collapsed_side_panels_fail_closed() -> None:
    with pytest.raises(ValueError):
        sheet.make_panel(
            (0.0, 0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            (PANEL_STEP_M, SPAN_M, 0.0),
            (PANEL_STEP_M, 0.0, 0.0),
            DELTA_GAMMA_M2_PER_S,
            release_index=1,
        )
    forged_bow_tie = sheet.SheetPanel(
        upstream_left=(0.0, 0.0, 0.0),
        upstream_right=(0.0, SPAN_M, 0.0),
        downstream_left=(PANEL_STEP_M, SPAN_M, 0.0),
        downstream_right=(PANEL_STEP_M, 0.0, 0.0),
        circulation_m2_s=DELTA_GAMMA_M2_PER_S,
        release_index=1,
    )
    with pytest.raises(ValueError):
        sheet.direct_connected_redeposit(
            (forged_bow_tie,), SMOOTHING_RADIUS_M, TARGET_SPACING_M
        )
    with pytest.raises(ValueError):
        sheet.make_panel(
            (0.0, 0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            (PANEL_STEP_M, 0.0, 0.0),
            (0.0, SPAN_M, 0.0),
            DELTA_GAMMA_M2_PER_S,
            release_index=1,
        )


def test_start_requires_release_one_and_public_state_validator_rejects_forgery() -> (
    None
):
    wrong_release = sheet.make_panel(
        (0.0, 0.0, 0.0),
        (0.0, SPAN_M, 0.0),
        (PANEL_STEP_M, 0.0, 0.0),
        (PANEL_STEP_M, SPAN_M, 0.0),
        DELTA_GAMMA_M2_PER_S,
        release_index=2,
    )
    with pytest.raises(ValueError, match="release_index"):
        sheet.start_incremental_sheet(
            wrong_release,
            SMOOTHING_RADIUS_M,
            TARGET_SPACING_M,
        )

    state = _start().state
    forged_writeable = replace(state, positions=state.positions.copy())
    with pytest.raises(ValueError, match="immutable"):
        sheet.append_live_basis_panel(
            forged_writeable,
            (0.08, 0.0, 0.0),
            (0.08, SPAN_M, 0.0),
            0.032,
        )
    duplicate_ids = replace(
        state,
        particle_ids=(state.particle_ids[0],) * len(state.particle_ids),
    )
    with pytest.raises(ValueError, match="unique"):
        sheet.particle_impulse(duplicate_ids)
    non_downstream_index = next(
        index
        for index, record in enumerate(state.lineage)
        if record.role != "fresh_downstream"
    )
    wrong_boundary = replace(state, downstream_particle_indices=(non_downstream_index,))
    with pytest.raises(ValueError, match="downstream"):
        sheet.append_live_basis_panel(
            wrong_boundary,
            (0.08, 0.0, 0.0),
            (0.08, SPAN_M, 0.0),
            0.032,
        )
    truncated_boundary = replace(
        state, downstream_particle_indices=state.downstream_particle_indices[:1]
    )
    with pytest.raises(ValueError, match="latest boundary"):
        sheet.append_live_basis_panel(
            truncated_boundary,
            (0.08, 0.0, 0.0),
            (0.08, SPAN_M, 0.0),
            0.032,
        )
    forged_record = replace(state.lineage[0], physical_owner="ptera")
    forged_owner = replace(
        state,
        lineage=(forged_record,) + state.lineage[1:],
    )
    with pytest.raises(ValueError, match="physical owner"):
        sheet.particle_velocity(forged_owner, PROBES_M)

    state_with_clones = _affine_transport_state(2)
    missing_pair = replace(state_with_clones, clone_pairs=())
    with pytest.raises(ValueError, match="cover every inherited"):
        sheet.particle_impulse(missing_pair)
    clone_index = _pair_indices(state_with_clones.clone_pairs[0])[1]
    forged_clone_edge_record = replace(
        state_with_clones.lineage[clone_index], source_edge=("forged", "edge")
    )
    forged_clone_edge = replace(
        state_with_clones,
        lineage=(
            state_with_clones.lineage[:clone_index]
            + (forged_clone_edge_record,)
            + state_with_clones.lineage[clone_index + 1 :]
        ),
    )
    with pytest.raises(ValueError, match="upstream edge"):
        sheet.particle_impulse(forged_clone_edge)
    fresh_index = next(
        index
        for index, record in enumerate(state.lineage)
        if record.role == "fresh_upstream"
    )
    forged_role_record = replace(state.lineage[fresh_index], role="fresh_left_tip")
    forged_role = replace(
        state,
        lineage=(
            state.lineage[:fresh_index]
            + (forged_role_record,)
            + state.lineage[fresh_index + 1 :]
        ),
    )
    with pytest.raises(ValueError, match="role disagrees"):
        sheet.particle_velocity(forged_role, PROBES_M)


@pytest.mark.parametrize(
    ("sigma", "spacing", "cap"),
    (
        (0.0, TARGET_SPACING_M, PARTICLE_CAP),
        (np.nan, TARGET_SPACING_M, PARTICLE_CAP),
        (SMOOTHING_RADIUS_M, 0.0, PARTICLE_CAP),
        (SMOOTHING_RADIUS_M, np.inf, PARTICLE_CAP),
        (SMOOTHING_RADIUS_M, TARGET_SPACING_M, 0),
        (SMOOTHING_RADIUS_M, TARGET_SPACING_M, True),
    ),
)
def test_bad_start_contract_fails_closed(
    sigma: object, spacing: object, cap: object
) -> None:
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        sheet.start_incremental_sheet(
            _first_panel(),
            sigma,  # type: ignore[arg-type]
            spacing,  # type: ignore[arg-type]
            particle_cap=cap,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("matrix", "translation", "sigma_scale"),
    (
        (np.eye(2), AFFINE_TRANSLATION_M, AFFINE_SIGMA_SCALE),
        (np.full((3, 3), np.nan), AFFINE_TRANSLATION_M, AFFINE_SIGMA_SCALE),
        (AFFINE_MATRIX, np.zeros(2), AFFINE_SIGMA_SCALE),
        (AFFINE_MATRIX, np.asarray((0.0, np.inf, 0.0)), AFFINE_SIGMA_SCALE),
        (AFFINE_MATRIX, AFFINE_TRANSLATION_M, 0.0),
        (AFFINE_MATRIX, AFFINE_TRANSLATION_M, True),
    ),
)
def test_bad_affine_transport_fails_without_mutating_state(
    matrix: object, translation: object, sigma_scale: object
) -> None:
    state = _start().state
    before = (state.positions.copy(), state.gamma.copy(), state.sigma.copy())
    with pytest.raises((TypeError, ValueError)):
        sheet.affine_transport_state(
            state,
            matrix,  # type: ignore[arg-type]
            translation,  # type: ignore[arg-type]
            sigma_scale,  # type: ignore[arg-type]
        )
    assert np.array_equal(state.positions, before[0])
    assert np.array_equal(state.gamma, before[1])
    assert np.array_equal(state.sigma, before[2])


@pytest.mark.parametrize(
    ("downstream_left", "downstream_right", "circulation", "cap"),
    (
        ((np.nan, 0.0, 0.0), (0.08, SPAN_M, 0.0), 0.032, PARTICLE_CAP),
        ((0.08, 0.0), (0.08, SPAN_M, 0.0), 0.032, PARTICLE_CAP),
        ((0.08, 0.0, 0.0), (0.08, SPAN_M, 0.0), np.inf, PARTICLE_CAP),
        (
            (0.08, 0.0, 0.0),
            (0.08, SPAN_M, 0.0),
            DELTA_GAMMA_M2_PER_S,
            PARTICLE_CAP,
        ),
        ((0.08, 0.0, 0.0), (0.08, SPAN_M, 0.0), 0.032, True),
    ),
)
def test_bad_append_fails_without_mutating_parent(
    downstream_left: object,
    downstream_right: object,
    circulation: object,
    cap: object,
) -> None:
    state = _start().state
    before = (state.positions.copy(), state.gamma.copy(), state.sigma.copy())
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        sheet.append_live_basis_panel(
            state,
            downstream_left,  # type: ignore[arg-type]
            downstream_right,  # type: ignore[arg-type]
            circulation,  # type: ignore[arg-type]
            particle_cap=cap,  # type: ignore[arg-type]
        )
    assert np.array_equal(state.positions, before[0])
    assert np.array_equal(state.gamma, before[1])
    assert np.array_equal(state.sigma, before[2])


def test_probe_validation_and_direct_redeposit_validation_fail_closed() -> None:
    state = _start().state
    with pytest.raises((TypeError, ValueError)):
        sheet.particle_velocity(state, np.zeros((3, 2)))
    with pytest.raises((TypeError, ValueError)):
        sheet.particle_velocity(state, np.asarray(((np.nan, 0.0, 0.0),)))
    with pytest.raises((TypeError, ValueError)):
        sheet.direct_connected_redeposit((), SMOOTHING_RADIUS_M, TARGET_SPACING_M)
    with pytest.raises((TypeError, ValueError)):
        sheet.direct_connected_redeposit(state.panels, -1.0, TARGET_SPACING_M)


def test_zero_and_affine_constructions_are_content_deterministic() -> None:
    zero_first = _zero_transport_state(4)
    zero_second = _zero_transport_state(4)
    affine_first = _affine_transport_state(4)
    affine_second = _affine_transport_state(4)
    for first, second in (
        (zero_first, zero_second),
        (affine_first, affine_second),
    ):
        assert np.array_equal(first.positions, second.positions)
        assert np.array_equal(first.gamma, second.gamma)
        assert np.array_equal(first.sigma, second.sigma)
        assert first.particle_ids == second.particle_ids
        assert first.lineage == second.lineage
        assert first.downstream_particle_indices == second.downstream_particle_indices
        assert first.clone_pairs == second.clone_pairs
        assert np.array_equal(
            sheet.particle_velocity(first, PROBES_M),
            sheet.particle_velocity(second, PROBES_M),
        )
        assert np.array_equal(
            sheet.particle_impulse(first), sheet.particle_impulse(second)
        )
