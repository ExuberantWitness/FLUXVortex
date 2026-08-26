"""Work-conjugate Q16 aerodynamic/structure transfer on CPU oracle and CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16Mesh, make_rectangular_q16_mesh
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_transfer import Q16CudaSurfaceTransfer


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _mesh(*, chordwise_elements: int = 2, spanwise_elements: int = 1) -> Q16Mesh:
    return make_rectangular_q16_mesh(
        chordwise_element_count=chordwise_elements,
        spanwise_element_count=spanwise_elements,
        chord=2.0,
        span=1.4,
        thickness=0.18,
    )


def _map() -> Q16SurfaceTransferMap:
    mesh = _mesh()
    coordinates = np.array(
        [
            [-0.83, -0.72, 1.0],
            [-0.19, -0.31, 1.0],
            [0.41, 0.15, -1.0],
            [0.79, 0.68, 0.55],
            [-0.63, 0.81, -0.35],
        ],
        dtype=np.float64,
    )
    return Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.array([0, 1, 1, 0, 1], dtype=np.int64),
        parametric_coordinates=coordinates,
    )


def _state(mesh: Q16Mesh) -> np.ndarray:
    rows = mesh.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    rows[:, :3] += np.column_stack([0.01 * x + 0.002 * x * y, -0.004 * y, 0.03 * x * y])
    rows[:, 3:] += np.column_stack(
        [0.001 * y, -0.0015 * x, 0.0008 * np.ones(rows.shape[0])]
    )
    return np.ascontiguousarray(rows.ravel())


def _forces() -> np.ndarray:
    return np.array(
        [
            [0.3, -0.7, 1.1],
            [-0.2, 0.4, 0.8],
            [0.6, 0.1, -0.5],
            [-0.9, 0.2, 0.7],
            [0.15, -0.35, 0.45],
        ],
        dtype=np.float64,
    )


def test_q16_transfer_is_exactly_work_conjugate_and_preserves_force_moment() -> None:
    transfer = _map()
    state = _state(transfer.mesh)
    forces = _forces()
    rng = np.random.default_rng(20260821)
    increment = np.ascontiguousarray(
        rng.normal(scale=0.03, size=transfer.structural_dof_count), dtype=np.float64
    )

    point_increment = transfer.interpolate(increment)
    generalized = transfer.transpose(forces)
    assert float(increment @ generalized) == pytest.approx(
        float(np.sum(point_increment * forces)), rel=0.0, abs=2.0e-14
    )

    balance = transfer.force_moment_balance(state, forces)
    np.testing.assert_allclose(
        balance.generalized_force, balance.aerodynamic_force, rtol=0.0, atol=3.0e-14
    )
    np.testing.assert_allclose(
        balance.generalized_moment,
        balance.aerodynamic_moment,
        rtol=0.0,
        atol=4.0e-14,
    )


def test_surface_offset_loads_directors_and_rejects_equal_four_node_idea() -> None:
    mesh = _mesh(chordwise_elements=1)
    transfer = Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.array([0], dtype=np.int64),
        parametric_coordinates=np.array([[0.23, -0.47, 1.0]], dtype=np.float64),
    )
    force = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
    generalized = transfer.transpose(force).reshape(16, 6)
    np.testing.assert_array_equal(generalized[:, 3:], generalized[:, :3])
    assert int(np.count_nonzero(np.linalg.norm(generalized[:, :3], axis=1) > 0.0)) == 16

    old_equal_four = np.zeros((16, 6), dtype=np.float64)
    old_equal_four[[0, 3, 12, 15], :3] = force[0] / 4.0
    assert not np.array_equal(old_equal_four, generalized)
    assert float(np.linalg.norm(old_equal_four[:, 3:])) == 0.0
    assert float(np.linalg.norm(generalized[:, 3:])) > 0.0


def test_shared_mesh_transfer_uses_unique_global_dofs_and_edge_owner_is_equivalent() -> (
    None
):
    mesh = _mesh()
    assert mesh.dof_count == 168
    left = Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.array([0], dtype=np.int64),
        parametric_coordinates=np.array([[1.0, -0.23, 1.0]], dtype=np.float64),
    )
    right = Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.array([1], dtype=np.int64),
        parametric_coordinates=np.array([[-1.0, -0.23, 1.0]], dtype=np.float64),
    )
    assert left.structural_dof_count == right.structural_dof_count == mesh.dof_count
    state = _state(mesh)
    force = np.array([[0.3, -0.2, 0.7]], dtype=np.float64)
    np.testing.assert_allclose(
        left.interpolate(state), right.interpolate(state), atol=1e-15
    )
    np.testing.assert_allclose(
        left.transpose(force), right.transpose(force), atol=1e-15
    )
    with pytest.raises(ValueError, match="multiple element owners"):
        Q16SurfaceTransferMap(
            mesh=mesh,
            element_indices=np.array([0, 1], dtype=np.int64),
            parametric_coordinates=np.array(
                [[1.0, -0.23, 1.0], [-1.0, -0.23, 1.0]], dtype=np.float64
            ),
        )


def test_cuda_q16_interpolation_and_transpose_match_independent_oracle() -> None:
    assert config.dtype_name() == "float64"
    transfer = _map()
    cuda = Q16CudaSurfaceTransfer(transfer, device=config.DEVICE)
    state_0 = _state(transfer.mesh)
    state_1 = np.ascontiguousarray(state_0 + 0.01)
    states = np.ascontiguousarray(np.stack([state_0, state_1]))
    force_0 = _forces()
    force_1 = np.ascontiguousarray(-0.7 * force_0)
    forces = np.ascontiguousarray(np.stack([force_0, force_1]))

    q_wp = wp.array(states, dtype=config.DTYPE, device=config.DEVICE)
    force_wp = wp.array(forces, dtype=config.VEC3, device=config.DEVICE)
    points_wp = cuda.interpolate(q_wp)
    generalized_wp = cuda.transpose(force_wp)
    generalized_repeat_wp = cuda.transpose(force_wp)
    wp.synchronize_device(config.DEVICE)

    expected_points = np.stack(
        [transfer.interpolate(state_0), transfer.interpolate(state_1)]
    )
    expected_generalized = np.stack(
        [transfer.transpose(force_0), transfer.transpose(force_1)]
    )
    np.testing.assert_allclose(
        points_wp.numpy(), expected_points, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        generalized_wp.numpy(), expected_generalized, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_array_equal(generalized_repeat_wp.numpy(), generalized_wp.numpy())


def test_two_by_two_shared_mesh_retains_virtual_work_force_and_moment() -> None:
    mesh = _mesh(chordwise_elements=2, spanwise_elements=2)
    transfer = Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.arange(4, dtype=np.int64),
        parametric_coordinates=np.array(
            [
                [-0.61, -0.37, 1.0],
                [0.42, -0.18, -1.0],
                [-0.31, 0.53, 0.4],
                [0.72, 0.29, -0.6],
            ],
            dtype=np.float64,
        ),
    )
    state = _state(mesh)
    direction = np.ascontiguousarray(
        np.random.default_rng(8821).normal(size=mesh.dof_count), dtype=np.float64
    )
    force = np.ascontiguousarray(
        np.random.default_rng(8822).normal(size=(4, 3)), dtype=np.float64
    )
    assert float(transfer.interpolate(direction).ravel() @ force.ravel()) == (
        pytest.approx(
            float(direction @ transfer.transpose(force)), rel=0.0, abs=2.0e-14
        )
    )
    balance = transfer.force_moment_balance(state, force)
    np.testing.assert_allclose(
        balance.generalized_force, balance.aerodynamic_force, atol=2.0e-15
    )
    np.testing.assert_allclose(
        balance.generalized_moment, balance.aerodynamic_moment, atol=2.0e-15
    )


def test_cuda_q16_transfer_rejects_host_arrays_before_launch() -> None:
    transfer = _map()
    cuda = Q16CudaSurfaceTransfer(transfer, device=config.DEVICE)
    host_state = wp.array(
        _state(transfer.mesh)[None, :], dtype=config.DTYPE, device="cpu"
    )
    with pytest.raises(ValueError, match="must reside on CUDA device"):
        cuda.interpolate(host_state)


def test_cuda_q16_transfer_rejects_wrong_dtype_and_nonfinite_inputs() -> None:
    transfer = _map()
    cuda = Q16CudaSurfaceTransfer(transfer, device=config.DEVICE)
    wrong_dtype = wp.zeros(
        (1, transfer.structural_dof_count),
        dtype=wp.float32,
        device=config.DEVICE,
    )
    with pytest.raises(TypeError, match="frozen float64"):
        cuda.interpolate(wrong_dtype)

    bad_state = _state(transfer.mesh)[None, :].copy()
    bad_state[0, 7] = np.nan
    bad_state_wp = wp.array(bad_state, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="state contains non-finite"):
        cuda.interpolate(bad_state_wp)

    bad_force = _forces()[None, :].copy()
    bad_force[0, 2, 1] = np.inf
    bad_force_wp = wp.array(bad_force, dtype=config.VEC3, device=config.DEVICE)
    with pytest.raises(
        FloatingPointError, match="aerodynamic_force contains non-finite"
    ):
        cuda.transpose(bad_force_wp)


@pytest.mark.parametrize(
    ("indices", "coordinates"),
    [
        (
            np.array([0], dtype=np.int32),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        ),
        (
            np.array([1], dtype=np.int64),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([[1.01, 0.0, 0.0]], dtype=np.float64),
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([[0.0, np.nan, 0.0]], dtype=np.float64),
        ),
    ],
)
def test_q16_transfer_map_rejects_wrong_dtype_range_and_nonfinite(
    indices: np.ndarray, coordinates: np.ndarray
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Q16SurfaceTransferMap(
            mesh=_mesh(chordwise_elements=1),
            element_indices=indices,
            parametric_coordinates=coordinates,
        )
