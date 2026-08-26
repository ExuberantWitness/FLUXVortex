"""Shared-node Q16 mesh ownership and deterministic CUDA assembly."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_mesh import (
    Q16ContinuumMesh,
    Q16MacroPropertyField,
    Q16MITC16EASMesh,
    Q16Mesh,
    make_rectangular_q16_mesh,
)
from fluxvortex.q16_ancf_shell import Q16_DOF_PER_ELEMENT
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_mesh import (
    Q16CudaMITC16EASMeshOperator,
    Q16CudaMeshOperator,
)


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _mesh() -> Q16Mesh:
    return make_rectangular_q16_mesh(
        chordwise_element_count=2,
        spanwise_element_count=1,
        chord=2.4,
        span=1.1,
        thickness=0.12,
    )


def _continuum() -> Q16ContinuumMesh:
    return Q16ContinuumMesh(
        _mesh(),
        young_modulus=6.8e7,
        poisson_ratio=0.29,
        density=1050.0,
    )


def _projected_continuum() -> Q16MITC16EASMesh:
    return Q16MITC16EASMesh(
        _mesh(),
        young_modulus=6.8e7,
        poisson_ratio=0.29,
        density=1050.0,
    )


def _deformed(model: Q16ContinuumMesh, scale: float = 1.0) -> np.ndarray:
    rows = model.mesh.reference_rows.copy()
    x = rows[:, 0].copy()
    y = rows[:, 1].copy()
    rows[:, 0] += scale * (0.008 * x + 0.004 * x * y)
    rows[:, 1] += scale * (-0.005 * y + 0.002 * x**2)
    rows[:, 2] += scale * (0.015 * x * y + 0.006 * x**2)
    rows[:, 3:] += scale * np.column_stack(
        [0.0008 * y, -0.0012 * x, 0.002 * np.ones(rows.shape[0])]
    )
    return np.ascontiguousarray(rows.ravel())


def test_two_q16_elements_share_exactly_one_four_node_edge() -> None:
    mesh = _mesh()
    assert mesh.element_count == 2
    assert mesh.node_count == 28
    assert mesh.dof_count == 28 * 6
    left = mesh.connectivity[0]
    right = mesh.connectivity[1]
    left_edge = left[[3, 7, 11, 15]]
    right_edge = right[[0, 4, 8, 12]]
    np.testing.assert_array_equal(left_edge, right_edge)
    assert len(set(left.tolist()) & set(right.tolist())) == 4
    np.testing.assert_allclose(mesh.reference_rows[:, 0].min(), 0.0, atol=0.0)
    np.testing.assert_allclose(mesh.reference_rows[:, 0].max(), 2.4, atol=0.0)
    assert mesh.reference_rows.flags.writeable is False
    assert mesh.connectivity.flags.writeable is False


def test_rectangular_two_by_two_q16_mesh_has_unique_global_node_ownership() -> None:
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=2,
        spanwise_element_count=2,
        chord=2.0,
        span=1.5,
        thickness=0.1,
    )
    assert mesh.element_count == 4
    assert mesh.node_count == 49
    assert np.unique(mesh.connectivity).size == mesh.node_count
    assert mesh.connectivity.shape == (4, 16)


def test_shared_mesh_energy_force_jv_and_mass_are_consistent() -> None:
    model = _continuum()
    state = _deformed(model)
    direction = np.random.default_rng(2401).normal(size=model.mesh.dof_count)
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 2.0e-7
    energy_difference = (
        model.strain_energy(np.ascontiguousarray(state + step * direction))
        - model.strain_energy(np.ascontiguousarray(state - step * direction))
    ) / (2.0 * step)
    force = model.internal_force(state)
    assert float(force @ direction) == pytest.approx(
        energy_difference, rel=3.0e-7, abs=3.0e-5
    )

    jv_step = 1.0e-6
    force_difference = (
        model.internal_force(np.ascontiguousarray(state + jv_step * direction))
        - model.internal_force(np.ascontiguousarray(state - jv_step * direction))
    ) / (2.0 * jv_step)
    jv = model.tangent_action(state, direction)
    assert (
        np.linalg.norm(jv - force_difference)
        / max(np.linalg.norm(force_difference), 1.0)
        <= 3.0e-7
    )

    rigid_x = np.zeros(model.mesh.dof_count, dtype=np.float64)
    rigid_x.reshape(model.mesh.node_count, 6)[:, 0] = 1.0
    recovered_mass = float(rigid_x @ model.mass_action(rigid_x))
    assert recovered_mass == pytest.approx(
        model.total_reference_volume * model.density, rel=4.0e-13
    )


def test_shared_interface_contributions_are_summed_not_overwritten() -> None:
    model = _continuum()
    local = np.zeros((model.mesh.element_count, Q16_DOF_PER_ELEMENT))
    local[0].reshape(16, 6)[[3, 7, 11, 15], 2] = 1.25
    local[1].reshape(16, 6)[[0, 4, 8, 12], 2] = -0.5
    assembled = model.mesh.assemble(np.ascontiguousarray(local)).reshape(
        model.mesh.node_count, 6
    )
    shared = model.mesh.connectivity[0, [3, 7, 11, 15]]
    np.testing.assert_array_equal(assembled[shared, 2], np.full(4, 0.75))


def test_cuda_shared_mesh_force_mass_and_jv_match_cpu_oracle() -> None:
    model = _continuum()
    cuda = Q16CudaMeshOperator(model, device=config.DEVICE)
    states_np = np.ascontiguousarray(
        np.stack([_deformed(model, 0.75), _deformed(model, 1.05)])
    )
    directions_np = np.ascontiguousarray(
        np.random.default_rng(991).normal(size=states_np.shape), dtype=np.float64
    )
    states = wp.array(states_np, dtype=config.DTYPE, device=config.DEVICE)
    directions = wp.array(directions_np, dtype=config.DTYPE, device=config.DEVICE)
    force = cuda.internal_force(states)
    force_repeat = cuda.internal_force(states)
    tangent = cuda.tangent_action(states, directions)
    mass = cuda.mass_action(directions)
    wp.synchronize_device(config.DEVICE)

    expected_force = np.stack([model.internal_force(state) for state in states_np])
    expected_tangent = np.stack(
        [
            model.tangent_action(state, direction)
            for state, direction in zip(states_np, directions_np, strict=True)
        ]
    )
    expected_mass = np.stack(
        [model.mass_action(direction) for direction in directions_np]
    )
    np.testing.assert_allclose(force.numpy(), expected_force, rtol=1e-10, atol=3e-7)
    np.testing.assert_allclose(tangent.numpy(), expected_tangent, rtol=1e-10, atol=8e-7)
    np.testing.assert_allclose(mass.numpy(), expected_mass, rtol=1e-11, atol=3e-9)
    np.testing.assert_array_equal(force_repeat.numpy(), force.numpy())


def test_projected_ans_eas_shared_mesh_cpu_and_cuda_are_consistent() -> None:
    model = _projected_continuum()
    state = _deformed(model)
    direction = np.random.default_rng(4041).normal(size=model.mesh.dof_count)
    direction = np.ascontiguousarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 3.0e-8
    energy_difference = (
        model.strain_energy(np.ascontiguousarray(state + step * direction))
        - model.strain_energy(np.ascontiguousarray(state - step * direction))
    ) / (2.0 * step)
    force_cpu = model.internal_force(state)
    assert float(force_cpu @ direction) == pytest.approx(
        energy_difference, rel=8.0e-7, abs=8.0e-5
    )

    cuda = Q16CudaMITC16EASMeshOperator(model, device=config.DEVICE)
    state_cuda = wp.array(state[None, :], dtype=config.DTYPE, device=config.DEVICE)
    direction_cuda = wp.array(
        direction[None, :], dtype=config.DTYPE, device=config.DEVICE
    )
    force = cuda.internal_force(state_cuda)
    force_repeat = cuda.internal_force(state_cuda)
    tangent = cuda.tangent_action(state_cuda, direction_cuda)
    mass = cuda.mass_action(direction_cuda)
    wp.synchronize_device(config.DEVICE)
    expected_tangent = model.tangent_action(state, direction)
    expected_mass = model.mass_action(direction)
    np.testing.assert_allclose(force.numpy()[0], force_cpu, rtol=2e-10, atol=3e-6)
    np.testing.assert_allclose(
        tangent.numpy()[0], expected_tangent, rtol=3e-10, atol=1e-5
    )
    np.testing.assert_allclose(mass.numpy()[0], expected_mass, rtol=1e-11, atol=3e-9)
    np.testing.assert_array_equal(force_repeat.numpy(), force.numpy())


def test_macro_property_field_drives_elementwise_stiffness_and_mass_on_gpu() -> None:
    mesh = _mesh()
    properties = Q16MacroPropertyField(
        young_modulus=np.array([5.0e7, 9.0e7], dtype=np.float64),
        poisson_ratio=np.array([0.27, 0.31], dtype=np.float64),
        density=np.array([700.0, 1400.0], dtype=np.float64),
    )
    model = Q16MITC16EASMesh(mesh, macro_properties=properties)
    assert model.macro_properties is properties
    np.testing.assert_array_equal(
        [element.young_modulus for element in model.elements],
        properties.young_modulus,
    )
    np.testing.assert_array_equal(
        [element.density for element in model.mass_elements], properties.density
    )
    assert properties.young_modulus.flags.writeable is False
    assert model.total_reference_mass == pytest.approx(
        sum(
            element.reference_volume * density
            for element, density in zip(
                model.mass_elements, properties.density, strict=True
            )
        ),
        rel=2.0e-15,
    )

    state = _deformed(model)
    direction = np.ascontiguousarray(
        np.random.default_rng(884).normal(size=mesh.dof_count), dtype=np.float64
    )
    cuda = Q16CudaMITC16EASMeshOperator(model, device=config.DEVICE)
    force = cuda.internal_force(
        wp.array(state[None, :], dtype=config.DTYPE, device=config.DEVICE)
    )
    mass = cuda.mass_action(
        wp.array(direction[None, :], dtype=config.DTYPE, device=config.DEVICE)
    )
    wp.synchronize_device(config.DEVICE)
    np.testing.assert_allclose(
        force.numpy()[0], model.internal_force(state), rtol=3.0e-10, atol=4.0e-6
    )
    np.testing.assert_allclose(
        mass.numpy()[0], model.mass_action(direction), rtol=1.0e-11, atol=4.0e-9
    )


def test_macro_property_field_rejects_wrong_shape_nonfinite_and_mixed_ownership() -> (
    None
):
    with pytest.raises(ValueError, match="same non-empty shape"):
        Q16MacroPropertyField(
            young_modulus=np.ones(2, dtype=np.float64),
            poisson_ratio=np.ones(1, dtype=np.float64) * 0.3,
            density=np.ones(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        Q16MacroPropertyField(
            young_modulus=np.array([1.0, np.nan], dtype=np.float64),
            poisson_ratio=np.array([0.3, 0.3], dtype=np.float64),
            density=np.ones(2, dtype=np.float64),
        )
    properties = Q16MacroPropertyField(
        young_modulus=np.ones(2, dtype=np.float64),
        poisson_ratio=np.full(2, 0.3, dtype=np.float64),
        density=np.ones(2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="either uniform scalars or macro_properties"):
        Q16MITC16EASMesh(
            _mesh(),
            young_modulus=1.0,
            poisson_ratio=0.3,
            density=1.0,
            macro_properties=properties,
        )


def test_cuda_shared_mesh_rejects_host_and_nonfinite_global_states() -> None:
    model = _continuum()
    cuda = Q16CudaMeshOperator(model, device=config.DEVICE)
    host = wp.zeros((1, model.mesh.dof_count), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="must reside on CUDA device"):
        cuda.internal_force(host)

    bad = _deformed(model)[None, :].copy()
    bad[0, 3] = np.nan
    bad_cuda = wp.array(bad, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="global state contains non-finite"):
        cuda.internal_force(bad_cuda)


@pytest.mark.parametrize(
    "connectivity",
    [
        np.zeros((1, 16), dtype=np.int32),
        np.array([list(range(15)) + [99]], dtype=np.int64),
        np.array([list(range(15)) + [14]], dtype=np.int64),
    ],
)
def test_q16_mesh_rejects_wrong_dtype_range_and_duplicate_local_nodes(
    connectivity: np.ndarray,
) -> None:
    rows = np.zeros((16, 6), dtype=np.float64)
    rows[:, 5] = 0.05
    with pytest.raises((TypeError, ValueError)):
        Q16Mesh(reference_rows=rows, connectivity=connectivity)
