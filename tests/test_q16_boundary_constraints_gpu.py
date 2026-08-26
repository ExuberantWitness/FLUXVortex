"""Essential boundary ownership for the projected shared-node Q16 operator."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_mesh import (
    Q16MITC16EASMesh,
    Q16Mesh,
    make_rectangular_q16_mesh,
)
from fluxvortex.q16_boundary_constraints import (
    Q16BoundaryConstraints,
    make_clamped_q16_nodes,
    make_clamped_span_root,
)
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_constraints import (
    Q16CudaBoundaryConstraints,
)


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _model() -> Q16MITC16EASMesh:
    return Q16MITC16EASMesh(
        make_rectangular_q16_mesh(
            chordwise_element_count=2,
            spanwise_element_count=1,
            chord=2.4,
            span=1.1,
            thickness=0.12,
        ),
        young_modulus=6.8e7,
        poisson_ratio=0.29,
        density=1050.0,
    )


def test_clamped_span_root_has_one_exact_global_owner() -> None:
    model = _model()
    boundary = make_clamped_span_root(model.mesh)
    expected_nodes = np.flatnonzero(model.mesh.reference_rows[:, 1] == 0.0)
    np.testing.assert_array_equal(boundary.constrained_nodes, expected_nodes)
    assert expected_nodes.size == 7
    assert boundary.constrained_dof_count == 42
    assert boundary.free_dof_count == model.mesh.dof_count - 42
    np.testing.assert_array_equal(
        boundary.prescribed_values,
        model.mesh.reference_state[boundary.constrained_dofs],
    )
    assert boundary.constrained_dofs.flags.writeable is False
    assert boundary.prescribed_values.flags.writeable is False


def test_explicit_root_node_owner_survives_rigid_rotation() -> None:
    mesh = _model().mesh
    root_nodes = np.ascontiguousarray(
        np.flatnonzero(mesh.reference_rows[:, 1] == 0.0), dtype=np.int64
    )
    theta = np.deg2rad(120.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rows = mesh.reference_rows.copy()
    rows[:, :3] = rows[:, :3] @ rotation.T
    rows[:, 3:] = rows[:, 3:] @ rotation.T
    rotated_mesh = Q16Mesh(
        reference_rows=np.ascontiguousarray(rows),
        connectivity=np.ascontiguousarray(mesh.connectivity),
    )

    boundary = make_clamped_q16_nodes(rotated_mesh, root_nodes)
    np.testing.assert_array_equal(boundary.constrained_nodes, root_nodes)
    np.testing.assert_array_equal(
        boundary.prescribed_values,
        rotated_mesh.reference_state[boundary.constrained_dofs],
    )
    heuristic = make_clamped_span_root(rotated_mesh, span_axis=1)
    assert not np.array_equal(heuristic.constrained_nodes, root_nodes)


def test_state_projection_and_reaction_are_exact_complements() -> None:
    model = _model()
    boundary = make_clamped_span_root(model.mesh)
    vector = np.ascontiguousarray(
        np.random.default_rng(521).normal(size=model.mesh.dof_count), dtype=np.float64
    )
    projected = boundary.project_free(vector)
    reaction = boundary.extract_reaction(vector)
    np.testing.assert_array_equal(projected + reaction, vector)
    np.testing.assert_array_equal(projected[boundary.constrained_dofs], 0.0)
    np.testing.assert_array_equal(
        reaction[boundary.free_dofs], np.zeros(boundary.free_dof_count)
    )
    np.testing.assert_array_equal(boundary.project_free(projected), projected)

    trial_state = np.ascontiguousarray(model.mesh.reference_state + vector)
    enforced = boundary.enforce_state(trial_state)
    np.testing.assert_array_equal(
        enforced[boundary.constrained_dofs], boundary.prescribed_values
    )
    np.testing.assert_array_equal(
        enforced[boundary.free_dofs], trial_state[boundary.free_dofs]
    )


def test_projected_force_virtual_work_uses_only_admissible_direction() -> None:
    model = _model()
    boundary = make_clamped_span_root(model.mesh)
    trial = model.mesh.reference_state.copy()
    rows = trial.reshape(model.mesh.node_count, 6)
    x = model.mesh.reference_rows[:, 0]
    y = model.mesh.reference_rows[:, 1]
    rows[:, 2] += 0.015 * x * y + 0.004 * x**2
    rows[:, 3] += 0.0007 * y
    state = boundary.enforce_state(np.ascontiguousarray(trial))
    projected_direction = boundary.project_free(
        np.ascontiguousarray(
            np.random.default_rng(701).normal(size=model.mesh.dof_count),
            dtype=np.float64,
        )
    )
    direction = np.ascontiguousarray(
        projected_direction / np.linalg.norm(projected_direction), dtype=np.float64
    )
    force = model.internal_force(state)
    projected_force = boundary.project_free(force)
    assert float(force @ direction) == pytest.approx(
        float(projected_force @ direction), rel=0.0, abs=1.0e-12
    )
    assert bool(np.isfinite(model.tangent_action(state, direction)).all())


def test_cuda_boundary_operations_match_cpu_and_repeat_bitwise() -> None:
    model = _model()
    boundary = make_clamped_span_root(model.mesh)
    cuda = Q16CudaBoundaryConstraints(boundary, device=config.DEVICE)
    values = np.ascontiguousarray(
        np.random.default_rng(2408).normal(size=(2, model.mesh.dof_count)),
        dtype=np.float64,
    )
    values_cuda = wp.array(values, dtype=config.DTYPE, device=config.DEVICE)
    projected = cuda.project_free(values_cuda)
    projected_repeat = cuda.project_free(values_cuda)
    reaction = cuda.extract_reaction(values_cuda)
    states = np.ascontiguousarray(model.mesh.reference_state[None, :] + values)
    states_cuda = wp.array(states, dtype=config.DTYPE, device=config.DEVICE)
    enforced = cuda.enforce_state(states_cuda)
    wp.synchronize_device(config.DEVICE)

    np.testing.assert_array_equal(
        projected.numpy(),
        np.stack([boundary.project_free(value) for value in values]),
    )
    np.testing.assert_array_equal(projected_repeat.numpy(), projected.numpy())
    np.testing.assert_array_equal(
        reaction.numpy(),
        np.stack([boundary.extract_reaction(value) for value in values]),
    )
    np.testing.assert_array_equal(
        enforced.numpy(),
        np.stack([boundary.enforce_state(state) for state in states]),
    )


def test_boundary_owner_and_cuda_inputs_fail_closed() -> None:
    model = _model()
    dofs = np.asarray([0, 1, 1], dtype=np.int64)
    values = np.zeros(3, dtype=np.float64)
    with pytest.raises(ValueError, match="unique"):
        Q16BoundaryConstraints(model.mesh, dofs, values)
    with pytest.raises(TypeError, match="int64"):
        Q16BoundaryConstraints(
            model.mesh, dofs.astype(np.int32), np.zeros(3, dtype=np.float64)
        )
    with pytest.raises(ValueError, match="global range"):
        Q16BoundaryConstraints(
            model.mesh,
            np.asarray([model.mesh.dof_count], dtype=np.int64),
            np.zeros(1, dtype=np.float64),
        )
    bad_values = np.asarray([np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        Q16BoundaryConstraints(model.mesh, np.asarray([0], dtype=np.int64), bad_values)

    boundary = make_clamped_span_root(model.mesh)
    cuda = Q16CudaBoundaryConstraints(boundary, device=config.DEVICE)
    host = wp.zeros((1, model.mesh.dof_count), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        cuda.project_free(host)
    wrong_dtype = wp.zeros(
        (1, model.mesh.dof_count), dtype=wp.float32, device=config.DEVICE
    )
    with pytest.raises(TypeError, match="float64"):
        cuda.enforce_state(wrong_dtype)
    wrong_shape = wp.zeros(
        (1, model.mesh.dof_count - 1), dtype=config.DTYPE, device=config.DEVICE
    )
    with pytest.raises(ValueError, match="shape"):
        cuda.extract_reaction(wrong_shape)
    bad = np.zeros((1, model.mesh.dof_count), dtype=np.float64)
    bad[0, 3] = np.inf
    bad_cuda = wp.array(bad, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="non-finite"):
        cuda.project_free(bad_cuda)
