from __future__ import annotations

import ast
from dataclasses import fields, replace
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

import fluxvortex.rvpm_dyadic_edge_bridge as dyadic_module
from fluxvortex.rvpm_dyadic_edge_bridge import (
    DYADIC_BRIDGE_INTERFACE_ID,
    DYADIC_PARTICLE_SCHEMA,
    DYADIC_PLAN_INTERFACE_ID,
    MAX_PARTICLES_PER_EDGE,
    MAX_REFINEMENT_LEVEL,
    MAX_TOTAL_PARTICLES,
    deposit_edge_graph_prescribed_sigma_dyadic_panels,
    edge_graph_parent_digest_sha256,
    plan_edge_graph_prescribed_sigma_dyadic_panels,
    recompute_dyadic_bridge_sidecar,
    validate_dyadic_bridge_result,
    validate_dyadic_deposition_plan,
)
from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DIAGNOSTIC_SHADOW_OWNER,
    DirectedRing,
    FROZEN_OVERLAP_LAMBDA,
    RING_PHYSICAL_OWNER,
    assemble_ring_edge_graph,
    canonical_edge_key,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)


SIGMA = 0.85
BASE_SPACING = 0.4


def _rectangle_graph() -> object:
    nodes = (
        BridgeNode("a", (-1.0, -0.5, 0.0)),
        BridgeNode("b", (1.0, -0.5, 0.0)),
        BridgeNode("c", (1.0, 0.5, 0.0)),
        BridgeNode("d", (-1.0, 0.5, 0.0)),
    )
    return assemble_ring_edge_graph(
        nodes,
        (DirectedRing("panel", ("a", "b", "c", "d"), 0.7),),
    )


def _adjacent_graph(*, equal: bool = False) -> object:
    nodes = (
        BridgeNode("a", (0.0, 0.0, 0.0)),
        BridgeNode("b", (1.0, 0.0, 0.0)),
        BridgeNode("c", (1.0, 1.0, 0.0)),
        BridgeNode("d", (0.0, 1.0, 0.0)),
        BridgeNode("e", (2.0, 0.0, 0.0)),
        BridgeNode("f", (2.0, 1.0, 0.0)),
    )
    right = 1.5 if equal else 0.4
    rings = (
        DirectedRing("left", ("a", "b", "c", "d"), 1.5),
        DirectedRing("right", ("b", "e", "f", "c"), right),
    )
    return assemble_ring_edge_graph(nodes, rings)


def _deposit(graph: object, level: int, *, spacing: float = BASE_SPACING) -> object:
    return deposit_edge_graph_prescribed_sigma_dyadic_panels(
        graph,  # type: ignore[arg-type]
        smoothing_radius=SIGMA,
        base_target_spacing=spacing,
        refinement_level=level,
        step=13,
    )


def _counts_by_edge(result: object) -> dict[tuple[object, object], int]:
    counts: dict[tuple[object, object], int] = {}
    for item in result.lineage:  # type: ignore[attr-defined]
        counts[item.source_edge] = counts.get(item.source_edge, 0) + 1
        assert item.subdivision_count >= counts[item.source_edge]
    return counts


def _assert_result_exact(left: object, right: object) -> None:
    assert np.array_equal(left.positions, right.positions)  # type: ignore[attr-defined]
    assert np.array_equal(left.gamma, right.gamma)  # type: ignore[attr-defined]
    assert np.array_equal(left.sigma, right.sigma)  # type: ignore[attr-defined]
    assert left.positions.dtype == right.positions.dtype == np.float64  # type: ignore[attr-defined]
    assert left.gamma.dtype == right.gamma.dtype == np.float64  # type: ignore[attr-defined]
    assert left.sigma.dtype == right.sigma.dtype == np.float64  # type: ignore[attr-defined]
    assert left.particle_ids == right.particle_ids  # type: ignore[attr-defined]
    assert left.lineage == right.lineage  # type: ignore[attr-defined]
    assert left.edge_graph == right.edge_graph  # type: ignore[attr-defined]
    assert left.feedback_velocity is right.feedback_velocity is None  # type: ignore[attr-defined]
    assert left.diagnostics == right.diagnostics  # type: ignore[attr-defined]


def test_public_api_and_caps_are_frozen_and_source_only() -> None:
    assert tuple(
        signature(deposit_edge_graph_prescribed_sigma_dyadic_panels).parameters
    ) == (
        "edge_graph",
        "smoothing_radius",
        "base_target_spacing",
        "refinement_level",
        "step",
    )
    assert DYADIC_PARTICLE_SCHEMA == ("rvpm-edge-shadow-prescribed-sigma-spacing-v1")
    assert MAX_REFINEMENT_LEVEL == 30
    assert MAX_PARTICLES_PER_EDGE == 250_000
    assert MAX_TOTAL_PARTICLES == 1_000_000

    source_path = Path(dyadic_module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        token in module.casefold()
        for module in imported_modules
        for token in ("baik", "yang", "izraelevitz", "ptera")
    )


def test_plan_and_result_sidecars_are_independently_recomputable() -> None:
    graph = _adjacent_graph()
    plan = plan_edge_graph_prescribed_sigma_dyadic_panels(
        graph,
        smoothing_radius=SIGMA,
        base_target_spacing=BASE_SPACING,
        refinement_level=3,
        step=13,
    )
    assert plan.interface_id == DYADIC_PLAN_INTERFACE_ID
    assert validate_dyadic_deposition_plan(plan, graph) is plan
    assert plan.parent_edge_graph_sha256 == edge_graph_parent_digest_sha256(graph)
    assert len(plan.producer_artifact_sha256) == 64
    assert len(plan.edge_bridge_artifact_sha256) == 64
    assert plan.particle_schema == DYADIC_PARTICLE_SCHEMA
    assert plan.physical_owner == RING_PHYSICAL_OWNER
    assert plan.owner_state == DIAGNOSTIC_SHADOW_OWNER
    assert len(plan.parent_edge_graph_sha256) == 64
    assert len(plan.plan_sha256) == 64
    assert plan.predicted_total_particle_count == sum(
        panel.panel_count for panel in plan.edge_panels
    )
    for panel, edge in zip(plan.edge_panels, graph.retained_edges, strict=True):
        length = float(
            np.linalg.norm(
                np.asarray(edge.end_position) - np.asarray(edge.start_position)
            )
        )
        expected_n0 = max(1, int(np.ceil(length / BASE_SPACING)))
        assert panel.edge_key == edge.key
        assert panel.edge_length_m == length
        assert panel.base_panel_count == expected_n0
        assert panel.refinement_level == 3
        assert panel.panel_count == expected_n0 * 8
        assert panel.realized_spacing_m == length / panel.panel_count
        assert panel.smoothing_radius_m == SIGMA
        assert panel.parent_edge_graph_sha256 == plan.parent_edge_graph_sha256

    result = _deposit(graph, 3)
    assert result.interface_id == DYADIC_BRIDGE_INTERFACE_ID
    assert validate_dyadic_bridge_result(result) is result
    recomputed = recompute_dyadic_bridge_sidecar(result)
    assert recomputed["plan"] == result.plan == plan
    assert recomputed["particle_ledger"] == result.particle_ledger
    assert recomputed["shadow_state_sha256"] == result.shadow_state_sha256
    assert recomputed["manifest_sha256"] == result.manifest_sha256
    assert len(result.particle_ledger) == len(result.edge_panels)
    assert result.particle_ledger[0].particle_slice_start == 0
    assert result.particle_ledger[-1].particle_slice_stop == len(result.particle_ids)
    for panel, particle in zip(result.edge_panels, result.particle_ledger, strict=True):
        assert particle.edge_key == panel.edge_key
        assert particle.particle_slice_stop - particle.particle_slice_start == (
            panel.panel_count
        )
        assert particle.parent_edge_graph_sha256 == result.parent_edge_graph_sha256
        assert len(particle.particle_ids_sha256) == 64
        assert len(particle.lineage_sha256) == 64


def test_parent_digest_changes_for_one_ulp_geometry_and_rejects_wrong_parent() -> None:
    graph = _rectangle_graph()
    plan = plan_edge_graph_prescribed_sigma_dyadic_panels(
        graph,
        smoothing_radius=SIGMA,
        base_target_spacing=BASE_SPACING,
        refinement_level=0,
        step=13,
    )
    moved_nodes = tuple(
        BridgeNode(
            node.node_id,
            (
                float(np.nextafter(node.position[0], np.inf)),
                node.position[1],
                node.position[2],
            ),
        )
        if node.node_id == "b"
        else node
        for node in graph.nodes
    )
    moved_graph = assemble_ring_edge_graph(
        moved_nodes,
        (DirectedRing("panel", ("a", "b", "c", "d"), 0.7),),
    )
    assert edge_graph_parent_digest_sha256(moved_graph) != (
        plan.parent_edge_graph_sha256
    )
    with pytest.raises(ValueError, match="another graph"):
        validate_dyadic_deposition_plan(plan, moved_graph)


@pytest.mark.parametrize(
    ("graph_factory", "spacing"),
    [
        (_rectangle_graph, BASE_SPACING),
        (_rectangle_graph, 0.3),
        (_adjacent_graph, BASE_SPACING),
    ],
)
def test_level_zero_is_an_exact_reduction_of_frozen_prescribed_spacing(
    graph_factory: object,
    spacing: float,
) -> None:
    graph = graph_factory()  # type: ignore[operator]
    baseline = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=SIGMA,
        target_spacing=spacing,
        step=13,
    )
    observed = _deposit(graph, 0, spacing=spacing)
    _assert_result_exact(observed, baseline)


def test_level_zero_rejects_runtime_public_callable_replacement_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _rectangle_graph()
    original = dyadic_module.deposit_edge_graph_prescribed_sigma_and_spacing
    calls = 0

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        dyadic_module,
        "deposit_edge_graph_prescribed_sigma_and_spacing",
        spy,
    )
    with pytest.raises(ValueError, match="public_deposition binding"):
        _deposit(graph, 0)
    assert calls == 0
    monkeypatch.setattr(
        dyadic_module,
        "deposit_edge_graph_prescribed_sigma_and_spacing",
        original,
    )
    level_zero = _deposit(graph, 0)
    assert validate_dyadic_bridge_result(level_zero) is level_zero


def test_each_retained_edge_count_doubles_exactly_and_sigma_stays_fixed() -> None:
    graph = _adjacent_graph()
    results = [_deposit(graph, level) for level in range(5)]
    counts = [_counts_by_edge(result) for result in results]

    assert set(counts[0]) == {edge.key for edge in graph.retained_edges}
    for coarse, fine in zip(counts[:-1], counts[1:]):
        assert coarse.keys() == fine.keys()
        assert all(fine[key] == 2 * coarse[key] for key in coarse)
    for level, result in enumerate(results):
        assert result.positions.shape[0] == sum(counts[level].values())
        assert np.all(result.sigma == SIGMA)
        assert len(set(result.particle_ids)) == len(result.particle_ids)
        assert tuple(item.particle_id for item in result.lineage) == result.particle_ids
        for item in result.lineage:
            assert item.subdivision_count == counts[level][item.source_edge]
            assert item.step == 13
            assert item.physical_owner == RING_PHYSICAL_OWNER
            assert item.owner_state == DIAGNOSTIC_SHADOW_OWNER


def test_dyadic_results_are_deterministic_conservative_and_c_contiguous() -> None:
    graph = _adjacent_graph()
    first = _deposit(graph, 3)
    second = _deposit(graph, 3)
    _assert_result_exact(first, second)

    assert first.positions.flags.c_contiguous
    assert first.gamma.flags.c_contiguous
    assert first.sigma.flags.c_contiguous
    assert first.diagnostics.incidence_residual <= 1.0e-12
    assert first.diagnostics.edge_reconstruction_residual <= 1.0e-12
    assert first.diagnostics.max_edge_conservation_abs <= 1.0e-14
    assert first.diagnostics.max_edge_conservation_rel <= 1.0e-12
    assert first.diagnostics.global_conservation_abs <= 1.0e-14
    assert first.diagnostics.global_conservation_rel <= 1.0e-12
    assert first.diagnostics.nonfinite_count == 0
    assert first.diagnostics.owner_conflict_count == 0
    assert first.diagnostics.feedback_call_count == 0

    for edge in graph.retained_edges:
        indices = [
            index
            for index, item in enumerate(first.lineage)
            if item.source_edge == edge.key
        ]
        np.testing.assert_allclose(
            np.sum(first.gamma[indices], axis=0),
            np.asarray(edge.vector_moment),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_canceled_edge_remains_in_graph_but_never_receives_particles() -> None:
    graph = _adjacent_graph(equal=True)
    shared = canonical_edge_key("b", "c")
    result = _deposit(graph, 2)
    graph_edges = {edge.key: edge for edge in graph.edges}

    assert shared in graph_edges
    assert not graph_edges[shared].retained
    assert all(item.source_edge != shared for item in result.lineage)
    assert result.diagnostics.source_edge_count == 7
    assert result.diagnostics.retained_edge_count == 6


def test_one_ulp_around_an_integer_base_count_is_not_rounded_or_welded() -> None:
    graph = _rectangle_graph()
    exact = _deposit(graph, 0, spacing=0.25)
    spacing_up = float(np.nextafter(0.25, np.inf))
    spacing_down = float(np.nextafter(0.25, 0.0))
    upward = _deposit(graph, 0, spacing=spacing_up)
    downward = _deposit(graph, 0, spacing=spacing_down)
    long_edge = canonical_edge_key("a", "b")

    assert _counts_by_edge(exact)[long_edge] == 8
    assert _counts_by_edge(upward)[long_edge] == 8
    assert _counts_by_edge(downward)[long_edge] == 9
    assert not np.array_equal(exact.positions, downward.positions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("smoothing_radius", 0.0, "smoothing_radius"),
        ("smoothing_radius", -1.0, "smoothing_radius"),
        ("smoothing_radius", np.nan, "smoothing_radius"),
        ("smoothing_radius", np.inf, "smoothing_radius"),
        ("smoothing_radius", True, "smoothing_radius"),
        ("base_target_spacing", 0.0, "base_target_spacing"),
        ("base_target_spacing", -1.0, "base_target_spacing"),
        ("base_target_spacing", np.nan, "base_target_spacing"),
        ("base_target_spacing", np.inf, "base_target_spacing"),
        ("base_target_spacing", True, "base_target_spacing"),
        ("refinement_level", -1, "refinement_level"),
        ("refinement_level", 0.5, "refinement_level"),
        ("refinement_level", True, "refinement_level"),
        ("step", -1, "step"),
        ("step", 0.5, "step"),
        ("step", True, "step"),
    ],
)
def test_invalid_scalar_inputs_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "smoothing_radius": SIGMA,
        "base_target_spacing": BASE_SPACING,
        "refinement_level": 0,
        "step": 0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        deposit_edge_graph_prescribed_sigma_dyadic_panels(
            _rectangle_graph(),  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )


def test_underlap_and_foreign_or_forged_graphs_fail_closed() -> None:
    graph = _rectangle_graph()
    with pytest.raises(ValueError, match="too large"):
        deposit_edge_graph_prescribed_sigma_dyadic_panels(
            graph,  # type: ignore[arg-type]
            smoothing_radius=0.5,
            base_target_spacing=(0.5 / FROZEN_OVERLAP_LAMBDA) * (1.0 + 1.0e-12),
            refinement_level=0,
            step=0,
        )
    with pytest.raises(ValueError, match="edge_graph"):
        _deposit(object(), 0)

    edge = graph.edges[0]
    incidence = edge.incidences[0]
    forged_incidence = replace(
        incidence,
        signed_circulation=float(np.nextafter(incidence.signed_circulation, np.inf)),
    )
    forged_edge = replace(edge, incidences=(forged_incidence,))
    with pytest.raises(ValueError, match="inconsistent signed incidence"):
        _deposit(replace(graph, edges=(forged_edge, *graph.edges[1:])), 0)


def test_level_and_particle_caps_fail_before_deposition_allocation() -> None:
    graph = _rectangle_graph()
    with pytest.raises(ValueError, match="level cap"):
        _deposit(graph, MAX_REFINEMENT_LEVEL + 1)
    with pytest.raises(ValueError, match="particle-per-edge"):
        _deposit(graph, MAX_REFINEMENT_LEVEL)

    tiny = float(np.nextafter(0.0, 1.0))
    with pytest.raises(ValueError, match="non-finite base panel count"):
        deposit_edge_graph_prescribed_sigma_dyadic_panels(
            graph,  # type: ignore[arg-type]
            smoothing_radius=3.0 * tiny,
            base_target_spacing=tiny,
            refinement_level=0,
            step=0,
        )


def test_per_edge_and_total_caps_are_independent_and_transactional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _rectangle_graph()
    monkeypatch.setattr(dyadic_module, "MAX_PARTICLES_PER_EDGE", 4)
    with pytest.raises(ValueError, match="particle-per-edge"):
        _deposit(graph, 0)

    monkeypatch.setattr(dyadic_module, "MAX_PARTICLES_PER_EDGE", 100)
    monkeypatch.setattr(dyadic_module, "MAX_TOTAL_PARTICLES", 12)
    with pytest.raises(ValueError, match="total-particle"):
        _deposit(graph, 0)

    monkeypatch.setattr(dyadic_module, "MAX_TOTAL_PARTICLES", MAX_TOTAL_PARTICLES)
    clean = _deposit(graph, 0)
    assert clean.diagnostics.particle_count == 16


def test_tampered_internal_deposition_fails_before_registration_and_retries_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _rectangle_graph()
    original = dyadic_module._deposit_validated_graph

    calls = 0

    def tampered_backend(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        bridge = original(*args, **kwargs)  # type: ignore[arg-type]
        positions = bridge.positions.copy()
        positions[0, 0] = np.nextafter(positions[0, 0], np.inf)
        return replace(bridge, positions=positions)

    monkeypatch.setattr(
        dyadic_module,
        "_deposit_validated_graph",
        tampered_backend,
    )
    with pytest.raises(ValueError, match="private_deposition binding"):
        _deposit(graph, 1)
    assert calls == 0

    monkeypatch.setattr(dyadic_module, "_deposit_validated_graph", original)
    clean = _deposit(graph, 1)
    assert validate_dyadic_bridge_result(clean) is clean


@pytest.mark.parametrize(
    ("target", "attribute", "level", "message"),
    (
        (
            dyadic_module.edge_bridge_module,
            "deposit_edge_graph_prescribed_sigma_and_spacing",
            0,
            "public_deposition binding",
        ),
        (
            dyadic_module.edge_bridge_module,
            "_deposit_validated_graph",
            1,
            "private_deposition binding",
        ),
        (
            dyadic_module.edge_bridge_module,
            "_validate_graph_for_deposition",
            0,
            "graph_validator binding",
        ),
    ),
)
def test_module_callable_forwarders_fail_before_call_and_retry_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    attribute: str,
    level: int,
    message: str,
) -> None:
    graph = _rectangle_graph()
    original = getattr(target, attribute)
    calls = 0

    def forwarding(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(target, attribute, forwarding)
    with pytest.raises(ValueError, match=message):
        _deposit(graph, level)
    assert calls == 0
    monkeypatch.setattr(target, attribute, original)
    clean = _deposit(graph, level)
    assert validate_dyadic_bridge_result(clean) is clean


def test_transitive_private_helper_replacement_is_rejected_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _rectangle_graph()
    original = dyadic_module.edge_bridge_module._stable_vector_sum
    calls = 0

    def forwarding(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dyadic_module.edge_bridge_module, "_stable_vector_sum", forwarding
    )
    with pytest.raises(ValueError, match="dependency"):
        _deposit(graph, 1)
    assert calls == 0
    monkeypatch.setattr(
        dyadic_module.edge_bridge_module, "_stable_vector_sum", original
    )
    assert validate_dyadic_bridge_result(_deposit(graph, 1))


@pytest.mark.parametrize(
    "mutation",
    (
        "bool_sign",
        "bool_traversal",
        "float_traversal",
        "bool_ring_id",
        "reverse_nodes",
        "reverse_edges",
    ),
)
def test_parent_graph_requires_strict_types_and_canonical_assembler_order(
    mutation: str,
) -> None:
    graph = _rectangle_graph()
    edge = graph.edges[0]
    incidence = edge.incidences[0]
    if mutation == "bool_sign":
        forged_incidence = replace(incidence, canonical_sign=True)
        forged = replace(
            graph,
            edges=(
                replace(edge, incidences=(forged_incidence, *edge.incidences[1:])),
                *graph.edges[1:],
            ),
        )
    elif mutation == "bool_traversal":
        forged_incidence = replace(incidence, traversal_index=True)
        forged = replace(
            graph,
            edges=(
                replace(edge, incidences=(forged_incidence, *edge.incidences[1:])),
                *graph.edges[1:],
            ),
        )
    elif mutation == "float_traversal":
        forged_incidence = replace(incidence, traversal_index=0.5)
        forged = replace(
            graph,
            edges=(
                replace(edge, incidences=(forged_incidence, *edge.incidences[1:])),
                *graph.edges[1:],
            ),
        )
    elif mutation == "bool_ring_id":
        forged_incidence = replace(incidence, ring_id=True)
        forged = replace(
            graph,
            edges=(
                replace(edge, incidences=(forged_incidence, *edge.incidences[1:])),
                *graph.edges[1:],
            ),
        )
    elif mutation == "reverse_nodes":
        forged = replace(graph, nodes=tuple(reversed(graph.nodes)))
    else:
        forged = replace(graph, edges=tuple(reversed(graph.edges)))
    with pytest.raises(ValueError):
        _deposit(forged, 0)
    assert validate_dyadic_bridge_result(_deposit(graph, 0))


@pytest.mark.parametrize("target_name", ("graph", "node", "edge", "incidence"))
def test_legacy_parent_objects_reject_undeclared_fields_and_method_shadow(
    target_name: str,
) -> None:
    graph = _rectangle_graph()
    targets = {
        "graph": graph,
        "node": graph.nodes[0],
        "edge": graph.edges[0],
        "incidence": graph.edges[0].incidences[0],
    }
    target = targets[target_name]
    injected_name = "retained" if target_name == "edge" else "load"
    target.__dict__[injected_name] = False if target_name == "edge" else 1
    try:
        with pytest.raises(ValueError, match="undeclared"):
            _deposit(graph, 0)
    finally:
        target.__dict__.pop(injected_name)
    assert validate_dyadic_bridge_result(_deposit(graph, 0))


def test_live_plan_bool_and_shadow_dtype_or_layout_tamper_fail_closed() -> None:
    graph = _rectangle_graph()
    result = _deposit(graph, 1)
    panel = result.plan.edge_panels[0]
    original_level = panel.refinement_level
    object.__setattr__(panel, "refinement_level", True)
    with pytest.raises(ValueError, match="exact integer"):
        validate_dyadic_bridge_result(result)
    object.__setattr__(panel, "refinement_level", original_level)
    assert validate_dyadic_bridge_result(result) is result

    original_positions = result.bridge.positions
    object.__setattr__(
        result.bridge, "positions", original_positions.astype(np.float32)
    )
    with pytest.raises(ValueError, match="float64"):
        validate_dyadic_bridge_result(result)
    object.__setattr__(result.bridge, "positions", original_positions)

    fortran_positions = np.asfortranarray(original_positions)
    assert np.array_equal(fortran_positions, original_positions)
    assert not fortran_positions.flags.c_contiguous
    object.__setattr__(result.bridge, "positions", fortran_positions)
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_dyadic_bridge_result(result)
    object.__setattr__(result.bridge, "positions", original_positions)
    assert validate_dyadic_bridge_result(result) is result


@pytest.mark.parametrize("target_name", ("bridge", "diagnostics", "lineage"))
def test_live_legacy_shadow_objects_reject_undeclared_fields(
    target_name: str,
) -> None:
    result = _deposit(_rectangle_graph(), 1)
    targets = {
        "bridge": result.bridge,
        "diagnostics": result.bridge.diagnostics,
        "lineage": result.bridge.lineage[0],
    }
    target = targets[target_name]
    object.__setattr__(target, "load", 1)
    try:
        with pytest.raises(ValueError, match="undeclared"):
            validate_dyadic_bridge_result(result)
    finally:
        object.__delattr__(target, "load")
    assert validate_dyadic_bridge_result(result) is result


def test_sidecar_copy_tamper_and_mutable_shadow_attacks_fail_closed() -> None:
    graph = _rectangle_graph()
    result = _deposit(graph, 1)
    with pytest.raises(ValueError, match="directly produced live"):
        validate_dyadic_bridge_result(replace(result))

    first_panel = result.edge_panels[0]
    forged_panel = replace(first_panel, panel_count=first_panel.panel_count + 1)
    forged_plan = replace(
        result.plan,
        edge_panels=(forged_panel, *result.edge_panels[1:]),
    )
    forged_result = replace(result, plan=forged_plan)
    recomputed = recompute_dyadic_bridge_sidecar(forged_result)
    assert recomputed["plan"] != forged_plan
    with pytest.raises(ValueError, match="plan"):
        validate_dyadic_bridge_result(forged_result)

    first_particle = result.particle_ledger[0]
    forged_particle = replace(
        first_particle,
        particle_slice_stop=first_particle.particle_slice_stop + 1,
    )
    with pytest.raises(ValueError, match="particle ledger"):
        validate_dyadic_bridge_result(
            replace(
                result,
                particle_ledger=(forged_particle, *result.particle_ledger[1:]),
            )
        )

    old_value = float(result.positions[0, 0])
    result.positions[0, 0] = np.nextafter(old_value, np.inf)
    with pytest.raises(ValueError, match="physical state|shadow-state"):
        validate_dyadic_bridge_result(result)
    result.positions[0, 0] = old_value
    assert validate_dyadic_bridge_result(result) is result
    clean_retry = _deposit(graph, 1)
    assert validate_dyadic_bridge_result(clean_retry) is clean_retry


def test_result_wrapper_preserves_the_frozen_shadow_bridge_surface() -> None:
    result = _deposit(_rectangle_graph(), 1)
    assert tuple(item.name for item in fields(type(result))) == (
        "interface_id",
        "producer_id",
        "producer_artifact_sha256",
        "plan",
        "particle_ledger",
        "shadow_state_sha256",
        "manifest_sha256",
        "bridge",
    )
    assert tuple(item.name for item in fields(type(result.bridge))) == (
        "edge_graph",
        "positions",
        "gamma",
        "sigma",
        "particle_ids",
        "lineage",
        "feedback_velocity",
        "diagnostics",
    )
    assert result.shadow is result.bridge
    assert result.edge_graph is result.bridge.edge_graph
    assert result.positions is result.bridge.positions
    assert result.gamma is result.bridge.gamma
    assert result.sigma is result.bridge.sigma
    assert result.particle_ids is result.bridge.particle_ids
    assert result.lineage is result.bridge.lineage
    assert result.diagnostics is result.bridge.diagnostics
    assert result.feedback_velocity is None
