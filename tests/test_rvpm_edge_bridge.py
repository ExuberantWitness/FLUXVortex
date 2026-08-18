from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DIAGNOSTIC_SHADOW_OWNER,
    DirectedRing,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
    assemble_ring_edge_graph,
    build_diagnostic_shadow_bridge,
    canonical_edge_key,
    deposit_edge_graph,
    deposit_edge_graph_fixed_sigma,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian


def _rectangle_inputs(
    circulation: float = 1.0,
) -> tuple[list[BridgeNode], list[DirectedRing]]:
    nodes = [
        BridgeNode("a", (-1.0, -0.5, 0.0)),
        BridgeNode("b", (1.0, -0.5, 0.0)),
        BridgeNode("c", (1.0, 0.5, 0.0)),
        BridgeNode("d", (-1.0, 0.5, 0.0)),
    ]
    rings = [DirectedRing("panel", ("a", "b", "c", "d"), circulation)]
    return nodes, rings


def _adjacent_inputs(
    left_circulation: float,
    right_circulation: float,
) -> tuple[list[BridgeNode], list[DirectedRing]]:
    nodes = [
        BridgeNode("a", (0.0, 0.0, 0.0)),
        BridgeNode("b", (1.0, 0.0, 0.0)),
        BridgeNode("c", (1.0, 1.0, 0.0)),
        BridgeNode("d", (0.0, 1.0, 0.0)),
        BridgeNode("e", (2.0, 0.0, 0.0)),
        BridgeNode("f", (2.0, 1.0, 0.0)),
    ]
    # Both rings are counter-clockwise.  Their shared-edge traversals are
    # b->c and c->b, respectively.
    rings = [
        DirectedRing("left", ("a", "b", "c", "d"), left_circulation),
        DirectedRing("right", ("b", "e", "f", "c"), right_circulation),
    ]
    return nodes, rings


def _edge_map(graph: object) -> dict[tuple[object, object], object]:
    return {edge.key: edge for edge in graph.edges}  # type: ignore[attr-defined]


class _ExplodingInput:
    def __iter__(self) -> object:
        raise AssertionError("disabled bridge evaluated an input")

    def __array__(self) -> object:
        raise AssertionError("disabled bridge converted an input")


def test_g0_disabled_dispatcher_is_input_blind_and_has_no_feedback() -> None:
    exploding = _ExplodingInput()
    result = build_diagnostic_shadow_bridge(
        exploding,
        exploding,
        subdivisions=exploding,
        step=exploding,
        enabled=False,
        physical_owner=exploding,
        owner_state=exploding,
        overlap_lambda=exploding,
    )

    assert result.edge_graph is None
    assert result.positions.shape == (0, 3)
    assert result.gamma.shape == (0, 3)
    assert result.sigma.shape == (0,)
    assert result.particle_ids == ()
    assert result.lineage == ()
    assert result.feedback_velocity is None
    assert not result.diagnostics.enabled
    assert result.diagnostics.feedback_call_count == 0


@pytest.mark.parametrize(
    ("nodes", "rings", "kwargs", "message"),
    [
        (
            [
                BridgeNode("a", (0.0, 0.0, 0.0)),
                BridgeNode("a", (1.0, 0.0, 0.0)),
            ],
            [DirectedRing("p", ("a", "b", "c", "d"), 1.0)],
            {},
            "inconsistent coordinates",
        ),
        (
            [
                BridgeNode("a", (0.0, 0.0, 0.0)),
                BridgeNode("b", (1.0, 0.0, 0.0)),
                BridgeNode("c", (1.0, 1.0, 0.0)),
            ],
            [DirectedRing("p", ("a", "b", "c"), 1.0)],
            {},
            "exactly four",
        ),
        (
            [
                BridgeNode("a", (0.0, 0.0, 0.0)),
                BridgeNode("b", (0.0, 0.0, 0.0)),
                BridgeNode("c", (1.0, 1.0, 0.0)),
                BridgeNode("d", (0.0, 1.0, 0.0)),
            ],
            [DirectedRing("p", ("a", "b", "c", "d"), 1.0)],
            {},
            "nonpositive length",
        ),
        (
            [
                BridgeNode("a", (0.0, 0.0, 0.0)),
                BridgeNode("b", (1.0, 0.0, 0.0)),
                BridgeNode("c", (1.0, np.nan, 0.0)),
                BridgeNode("d", (0.0, 1.0, 0.0)),
            ],
            [DirectedRing("p", ("a", "b", "c", "d"), 1.0)],
            {},
            "finite",
        ),
        (
            *_rectangle_inputs(),
            {"overlap_lambda": 0.0},
            "positive sigma",
        ),
        (
            *_rectangle_inputs(),
            {"overlap_lambda": 2.0},
            "frozen",
        ),
        (
            *_rectangle_inputs(),
            {"physical_owner": "particle"},
            "unsupported physical owner",
        ),
        (
            *_rectangle_inputs(),
            {"owner_state": "active"},
            "unsupported particle owner",
        ),
    ],
)
def test_g0_malformed_inputs_and_owner_transitions_fail_closed(
    nodes: list[BridgeNode],
    rings: list[DirectedRing],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=message):
        build_diagnostic_shadow_bridge(
            nodes,
            rings,
            subdivisions=4,
            step=0,
            **kwargs,
        )


def test_g0_topology_uses_explicit_ids_not_coordinate_rounding() -> None:
    first_nodes, first_rings = _rectangle_inputs()
    second_nodes = [
        BridgeNode(f"{node.node_id}2", np.asarray(node.position) + (1.0e-14, 0.0, 0.0))
        for node in first_nodes
    ]
    second_rings = [DirectedRing("panel2", ("a2", "b2", "c2", "d2"), 1.0)]
    graph = assemble_ring_edge_graph(
        first_nodes + second_nodes, first_rings + second_rings
    )
    assert len(graph.edges) == 8
    assert len(graph.retained_edges) == 8


def test_g1_single_ring_preserves_all_four_directed_incidences() -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    graph = assemble_ring_edge_graph(nodes, rings)
    expected_traversal = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))

    assert len(graph.edges) == 4
    assert len(graph.retained_edges) == 4
    observed = {
        (edge.incidences[0].source_start_id, edge.incidences[0].source_end_id)
        for edge in graph.edges
    }
    assert observed == set(expected_traversal)
    for traversal_index, directed_edge in enumerate(expected_traversal):
        edge = _edge_map(graph)[canonical_edge_key(*directed_edge)]
        incidence = edge.incidences[0]
        assert incidence.traversal_index == traversal_index
        assert incidence.ring_circulation == 0.7
        assert incidence.canonical_sign in (-1, 1)
    assert graph.incidence_residual <= 1.0e-12
    assert graph.edge_reconstruction_residual <= 1.0e-12


def test_g1_equal_adjacent_rings_record_then_cancel_shared_edge_exactly() -> None:
    nodes, rings = _adjacent_inputs(1.25, 1.25)
    graph = assemble_ring_edge_graph(nodes, rings)
    shared = _edge_map(graph)[canonical_edge_key("b", "c")]

    assert len(graph.edges) == 7
    assert len(shared.incidences) == 2
    assert {item.canonical_sign for item in shared.incidences} == {-1, 1}
    assert shared.circulation == 0.0
    assert not shared.retained
    assert len(graph.retained_edges) == 6


def test_g1_unequal_adjacent_rings_retain_signed_circulation_difference() -> None:
    nodes, rings = _adjacent_inputs(1.5, 0.4)
    graph = assemble_ring_edge_graph(nodes, rings)
    shared = _edge_map(graph)[canonical_edge_key("b", "c")]
    assert shared.circulation == pytest.approx(1.1, rel=0.0, abs=2.0e-16)
    assert shared.retained


def test_g1_reversed_traversal_with_reversed_gamma_preserves_physical_edges() -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    forward = assemble_ring_edge_graph(nodes, rings)
    reversed_graph = assemble_ring_edge_graph(
        nodes,
        [DirectedRing("panel", ("a", "d", "c", "b"), -0.7)],
    )

    forward_edges = _edge_map(forward)
    reversed_edges = _edge_map(reversed_graph)
    assert forward_edges.keys() == reversed_edges.keys()
    for key in forward_edges:
        assert forward_edges[key].circulation == reversed_edges[key].circulation
        assert forward_edges[key].vector_moment == reversed_edges[key].vector_moment
        original_incidence = forward_edges[key].incidences[0]
        reversed_incidence = reversed_edges[key].incidences[0]
        assert (
            original_incidence.source_start_id,
            original_incidence.source_end_id,
        ) == (
            reversed_incidence.source_end_id,
            reversed_incidence.source_start_id,
        )
        assert original_incidence.canonical_sign == -reversed_incidence.canonical_sign
        assert (
            original_incidence.ring_circulation == -reversed_incidence.ring_circulation
        )


def test_g1_graph_output_is_stable_under_generic_ring_input_order() -> None:
    nodes, rings = _adjacent_inputs(0.8, 0.3)
    forward = assemble_ring_edge_graph(nodes, rings)
    reordered = assemble_ring_edge_graph(list(reversed(nodes)), list(reversed(rings)))
    assert forward == reordered


def test_g1_nonmanifold_and_cooriented_shared_edges_fail_closed() -> None:
    nodes, rings = _adjacent_inputs(1.0, 1.0)
    cooriented = [
        rings[0],
        DirectedRing("right", ("b", "c", "f", "e"), 1.0),
    ]
    with pytest.raises(ValueError, match="co-oriented"):
        assemble_ring_edge_graph(nodes, cooriented)

    nonmanifold_nodes = nodes + [
        BridgeNode("g", (0.5, 0.0, 1.0)),
        BridgeNode("h", (0.5, 1.0, 1.0)),
    ]
    nonmanifold_rings = rings + [DirectedRing("third", ("b", "c", "h", "g"), 0.2)]
    with pytest.raises(ValueError, match="non-manifold"):
        assemble_ring_edge_graph(nonmanifold_nodes, nonmanifold_rings)


def test_g2_deposition_is_conservative_unique_stable_and_shadow_only() -> None:
    nodes, rings = _adjacent_inputs(1.5, 0.4)
    graph = assemble_ring_edge_graph(nodes, rings)
    first = deposit_edge_graph(graph, subdivisions=7, step=13)
    second = deposit_edge_graph(graph, subdivisions=7, step=13)

    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.gamma, second.gamma)
    assert np.array_equal(first.sigma, second.sigma)
    assert first.particle_ids == second.particle_ids
    assert first.lineage == second.lineage
    assert len(set(first.particle_ids)) == len(first.particle_ids)
    assert len({item.particle_id for item in first.lineage}) == len(first.lineage)
    assert first.feedback_velocity is None

    for edge in graph.retained_edges:
        indices = [
            index
            for index, item in enumerate(first.lineage)
            if item.source_edge == edge.key
        ]
        deposited = np.sum(first.gamma[indices], axis=0)
        expected = np.asarray(edge.vector_moment)
        absolute = np.linalg.norm(deposited - expected)
        relative = absolute / np.linalg.norm(expected)
        assert absolute <= 1.0e-14
        assert relative <= 1.0e-12
        assert all(first.lineage[index].step == 13 for index in indices)
        assert all(
            first.lineage[index].physical_owner == RING_PHYSICAL_OWNER
            for index in indices
        )
        assert all(
            first.lineage[index].owner_state == DIAGNOSTIC_SHADOW_OWNER
            for index in indices
        )

    diagnostics = first.diagnostics
    assert diagnostics.max_edge_conservation_abs <= 1.0e-14
    assert diagnostics.max_edge_conservation_rel <= 1.0e-12
    assert diagnostics.global_conservation_abs <= 1.0e-14
    assert diagnostics.global_conservation_rel <= 1.0e-12
    assert diagnostics.clip_count == 0
    assert diagnostics.nonfinite_count == 0
    assert diagnostics.owner_conflict_count == 0
    assert diagnostics.feedback_call_count == 0


def test_g2_fixed_sigma_uses_flowunsteady_overlap_as_a_spacing_bound() -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    graph = assemble_ring_edge_graph(nodes, rings)
    smoothing_radius = 0.5
    result = deposit_edge_graph_fixed_sigma(
        graph,
        smoothing_radius=smoothing_radius,
        step=13,
    )

    assert np.all(result.sigma == smoothing_radius)
    for edge in graph.retained_edges:
        indices = [
            index
            for index, item in enumerate(result.lineage)
            if item.source_edge == edge.key
        ]
        edge_length = np.linalg.norm(
            np.asarray(edge.end_position) - np.asarray(edge.start_position)
        )
        expected_count = int(
            np.ceil(edge_length / (smoothing_radius / FROZEN_OVERLAP_LAMBDA))
        )
        assert len(indices) == expected_count
        realized_spacing = edge_length / expected_count
        assert smoothing_radius / realized_spacing >= FROZEN_OVERLAP_LAMBDA
        assert np.allclose(
            np.sum(result.gamma[indices], axis=0),
            np.asarray(edge.vector_moment),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_g2_fixed_sigma_does_not_collapse_on_a_short_source_edge() -> None:
    nodes = [
        BridgeNode("a", (0.0, 0.0, 0.0)),
        BridgeNode("b", (0.01, 0.0, 0.0)),
        BridgeNode("c", (0.01, 1.0, 0.0)),
        BridgeNode("d", (0.0, 1.0, 0.0)),
    ]
    graph = assemble_ring_edge_graph(
        nodes,
        [DirectedRing("short-edge-panel", ("a", "b", "c", "d"), 1.0)],
    )
    result = deposit_edge_graph_fixed_sigma(
        graph,
        smoothing_radius=0.5,
        step=0,
    )
    short_key = canonical_edge_key("a", "b")
    short_indices = [
        index
        for index, item in enumerate(result.lineage)
        if item.source_edge == short_key
    ]
    assert len(short_indices) == 1
    assert result.sigma[short_indices[0]] == 0.5


def test_g2_prescribed_sigma_and_spacing_are_independent_and_conservative() -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    graph = assemble_ring_edge_graph(nodes, rings)
    coarse = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=0.5,
        target_spacing=0.1,
        step=13,
    )
    fine = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=0.5,
        target_spacing=0.05,
        step=13,
    )

    assert np.all(coarse.sigma == 0.5)
    assert np.all(fine.sigma == 0.5)
    assert fine.positions.shape[0] == 2 * coarse.positions.shape[0]
    assert coarse.particle_ids != fine.particle_ids
    for result, requested_spacing in ((coarse, 0.1), (fine, 0.05)):
        for edge in graph.retained_edges:
            indices = [
                index
                for index, item in enumerate(result.lineage)
                if item.source_edge == edge.key
            ]
            edge_length = np.linalg.norm(
                np.asarray(edge.end_position) - np.asarray(edge.start_position)
            )
            realized_spacing = edge_length / len(indices)
            assert realized_spacing <= requested_spacing * (1.0 + 1.0e-15)
            assert 0.5 / realized_spacing >= FROZEN_OVERLAP_LAMBDA
            assert np.allclose(
                np.sum(result.gamma[indices], axis=0),
                np.asarray(edge.vector_moment),
                rtol=0.0,
                atol=1.0e-14,
            )


def test_g3_fixed_core_quadrature_refinement_converges_without_changing_sigma() -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    graph = assemble_ring_edge_graph(nodes, rings)
    probes = np.asarray(
        ((0.0, 0.0, 0.2), (0.3, -0.1, 0.4), (-0.4, 0.2, 0.7)),
        dtype=float,
    )
    sigma = 0.3
    base_spacing = sigma / FROZEN_OVERLAP_LAMBDA

    reference = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=sigma,
        target_spacing=base_spacing / 32,
        step=0,
    )
    reference_field = direct_gaussian_erf_velocity_jacobian(
        reference.positions,
        reference.gamma,
        reference.sigma,
        target_positions=probes,
    ).velocity

    errors: list[float] = []
    counts: list[int] = []
    for refinement in (1, 2, 4, 8):
        result = deposit_edge_graph_prescribed_sigma_and_spacing(
            graph,
            smoothing_radius=sigma,
            target_spacing=base_spacing / refinement,
            step=0,
        )
        field = direct_gaussian_erf_velocity_jacobian(
            result.positions,
            result.gamma,
            result.sigma,
            target_positions=probes,
        ).velocity
        errors.append(float(np.linalg.norm(field - reference_field)))
        counts.append(result.positions.shape[0])
        assert np.all(result.sigma == sigma)

    assert counts == sorted(counts)
    assert all(left > right for left, right in zip(errors[:-1], errors[1:]))
    assert errors[-3] / errors[-2] >= 1.5
    assert errors[-2] / errors[-1] >= 1.5


@pytest.mark.parametrize("refinement", (48, 96))
def test_g2_high_count_prescribed_spacing_uses_stable_conservation_sum(
    refinement: int,
) -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    graph = assemble_ring_edge_graph(nodes, rings)
    sigma = 0.3
    result = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=sigma,
        target_spacing=sigma / FROZEN_OVERLAP_LAMBDA / refinement,
        step=0,
    )

    assert result.diagnostics.max_edge_conservation_abs <= 1.0e-14
    assert result.diagnostics.global_conservation_abs <= 1.0e-14
    assert np.all(result.sigma == sigma)


@pytest.mark.parametrize("smoothing_radius", [0.0, -1.0, np.inf, np.nan, True])
def test_g2_fixed_sigma_rejects_invalid_spatial_scales(
    smoothing_radius: object,
) -> None:
    nodes, rings = _rectangle_inputs()
    graph = assemble_ring_edge_graph(nodes, rings)
    with pytest.raises(ValueError, match="smoothing_radius"):
        deposit_edge_graph_fixed_sigma(
            graph,
            smoothing_radius=smoothing_radius,  # type: ignore[arg-type]
            step=0,
        )


@pytest.mark.parametrize(
    ("smoothing_radius", "target_spacing", "minimum_overlap", "message"),
    [
        (0.0, 0.1, FROZEN_OVERLAP_LAMBDA, "smoothing_radius"),
        (0.5, 0.0, FROZEN_OVERLAP_LAMBDA, "target_spacing"),
        (0.5, np.inf, FROZEN_OVERLAP_LAMBDA, "target_spacing"),
        (0.5, 0.1, 2.0, "frozen"),
        (0.5, 0.3, FROZEN_OVERLAP_LAMBDA, "too large"),
    ],
)
def test_g2_prescribed_sigma_and_spacing_reject_invalid_or_underlapped_inputs(
    smoothing_radius: object,
    target_spacing: object,
    minimum_overlap: object,
    message: str,
) -> None:
    nodes, rings = _rectangle_inputs()
    graph = assemble_ring_edge_graph(nodes, rings)
    with pytest.raises(ValueError, match=message):
        deposit_edge_graph_prescribed_sigma_and_spacing(
            graph,
            smoothing_radius=smoothing_radius,  # type: ignore[arg-type]
            target_spacing=target_spacing,  # type: ignore[arg-type]
            minimum_overlap=minimum_overlap,  # type: ignore[arg-type]
            step=0,
        )


def test_g2_canceled_shared_edge_has_ledger_but_emits_no_particles() -> None:
    nodes, rings = _adjacent_inputs(1.0, 1.0)
    result = build_diagnostic_shadow_bridge(
        nodes,
        rings,
        subdivisions=8,
        step=0,
    )
    shared_key = canonical_edge_key("b", "c")
    assert result.edge_graph is not None
    assert shared_key in _edge_map(result.edge_graph)
    assert _edge_map(result.edge_graph)[shared_key].circulation == 0.0
    assert all(item.source_edge != shared_key for item in result.lineage)
    assert result.diagnostics.source_edge_count == 7
    assert result.diagnostics.retained_edge_count == 6
    assert result.diagnostics.particle_count == 48


def test_g2_forged_graph_node_and_incidence_ledgers_fail_closed() -> None:
    nodes, rings = _rectangle_inputs(circulation=0.7)
    graph = assemble_ring_edge_graph(nodes, rings)

    first_node = graph.nodes[0]
    forged_nodes = (
        replace(
            first_node,
            position=np.asarray(first_node.position) + np.array([0.1, 0.0, 0.0]),
        ),
        *graph.nodes[1:],
    )
    with pytest.raises(ValueError, match="disagrees with its graph node"):
        deposit_edge_graph(replace(graph, nodes=forged_nodes), subdivisions=4, step=0)

    first_edge = graph.edges[0]
    first_incidence = first_edge.incidences[0]
    forged_incidence = replace(
        first_incidence,
        canonical_sign=-first_incidence.canonical_sign,
        signed_circulation=-first_incidence.signed_circulation,
    )
    forged_edge = replace(first_edge, incidences=(forged_incidence,))
    with pytest.raises(ValueError, match="sign disagrees with traversal"):
        deposit_edge_graph(
            replace(graph, edges=(forged_edge, *graph.edges[1:])),
            subdivisions=4,
            step=0,
        )


def _finite_segment_velocity(
    start: np.ndarray,
    end: np.ndarray,
    circulation: float,
    probes: np.ndarray,
) -> np.ndarray:
    """Independent analytic Biot-Savart field of one finite straight segment."""

    r1 = probes - start
    r2 = probes - end
    segment = end - start
    cross = np.cross(r1, r2)
    cross_norm_squared = np.einsum("ij,ij->i", cross, cross)
    if np.any(cross_norm_squared == 0.0):
        raise ValueError("analytic probe is collinear with a source segment")
    endpoint_factor = np.einsum(
        "j,ij->i",
        segment,
        r1 / np.linalg.norm(r1, axis=1)[:, None]
        - r2 / np.linalg.norm(r2, axis=1)[:, None],
    )
    return (
        circulation
        / (4.0 * pi)
        * cross
        * (endpoint_factor / cross_norm_squared)[:, None]
    )


def _point_segment_distance(
    probes: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    delta = end - start
    parameter = np.clip(
        np.einsum("ij,j->i", probes - start, delta) / np.dot(delta, delta),
        0.0,
        1.0,
    )
    closest = start + parameter[:, None] * delta
    return np.linalg.norm(probes - closest, axis=1)


@pytest.mark.parametrize("case", ["single_segment", "rectangular_ring"])
def test_g3_fixed_overlap_converges_to_analytic_finite_segment_fields(
    case: str,
) -> None:
    nodes, rings = _rectangle_inputs(circulation=1.0)
    vertices = {
        node.node_id: np.asarray(node.position, dtype=np.float64) for node in nodes
    }
    probes = np.array(
        [
            [0.0, 0.0, 0.8],
            [0.25, 0.1, 1.0],
            [-0.4, 0.15, 1.2],
            [0.6, -0.2, 0.9],
            [0.0, 0.0, 1.5],
        ],
        dtype=np.float64,
    )
    directed_edges = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
    selected_edges = directed_edges[:1] if case == "single_segment" else directed_edges
    analytic = sum(
        (
            _finite_segment_velocity(vertices[start], vertices[end], 1.0, probes)
            for start, end in selected_edges
        ),
        np.zeros((probes.shape[0], 3)),
    )

    finest_subdivisions = 32
    for start, end in selected_edges:
        finest_sigma = (
            FROZEN_OVERLAP_LAMBDA
            * np.linalg.norm(vertices[end] - vertices[start])
            / finest_subdivisions
        )
        assert np.all(
            _point_segment_distance(probes, vertices[start], vertices[end])
            >= 4.0 * finest_sigma
        )

    errors: list[float] = []
    for subdivisions in (4, 8, 16, 32):
        result = build_diagnostic_shadow_bridge(
            nodes,
            rings,
            subdivisions=subdivisions,
            step=0,
        )
        selected_keys = {
            canonical_edge_key(start, end) for start, end in selected_edges
        }
        indices = [
            index
            for index, item in enumerate(result.lineage)
            if item.source_edge in selected_keys
        ]
        field = direct_gaussian_erf_velocity_jacobian(
            result.positions[indices],
            result.gamma[indices],
            result.sigma[indices],
            target_positions=probes,
        ).velocity
        errors.append(
            float(np.linalg.norm(field - analytic) / np.linalg.norm(analytic))
        )

    assert errors[-1] <= 0.01
    if errors[-1] > 1.0e-10:
        assert errors[-3] / errors[-2] >= 1.5
        assert errors[-2] / errors[-1] >= 1.5
