"""GPU PCG regression for the nonlinear projected Q16 structural step."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh, make_rectangular_q16_mesh
from fluxvortex.q16_boundary_constraints import make_clamped_span_root
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_structural_solver import (
    Q16CudaNewmarkStepper,
    _q16_invert_jacobi_kernel,
)


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _case() -> tuple[Q16MITC16EASMesh, tuple[wp.array, ...]]:
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=1,
        spanwise_element_count=1,
        chord=1.0,
        span=0.8,
        thickness=0.08,
    )
    model = Q16MITC16EASMesh(
        mesh,
        young_modulus=1.0e8,
        poisson_ratio=0.28,
        density=950.0,
    )
    state = np.ascontiguousarray(mesh.reference_state[None, :])
    velocity = np.zeros_like(state)
    acceleration = np.zeros_like(state)
    force = np.zeros_like(state)
    tip_nodes = np.flatnonzero(mesh.reference_rows[:, 1] == 0.8)
    force[0, tip_nodes * 6 + 2] = 0.25
    arrays = tuple(
        wp.array(value, dtype=config.DTYPE, device=config.DEVICE)
        for value in (state, velocity, acceleration, force)
    )
    return model, arrays


def _stepper(model: Q16MITC16EASMesh, preconditioner: str) -> Q16CudaNewmarkStepper:
    return Q16CudaNewmarkStepper(
        model,
        make_clamped_span_root(model.mesh),
        device=config.DEVICE,
        newton_tolerance=2.0e-8,
        max_newton_iterations=15,
        cg_tolerance=2.0e-10,
        max_cg_iterations=1024,
        cg_check_every=16,
        preconditioner=preconditioner,
    )


def test_material_jacobi_pcg_reduces_krylov_work_without_changing_solution() -> None:
    model, inputs = _case()
    baseline = _stepper(model, "none").step(
        *inputs,
        delta_time=4.0e-2,
    )
    accelerated = _stepper(model, "material_jacobi").step(
        *inputs,
        delta_time=4.0e-2,
    )

    assert baseline.cg_iteration_count >= 100
    assert accelerated.cg_iteration_count < baseline.cg_iteration_count
    assert accelerated.newton_iteration_count == baseline.newton_iteration_count
    assert accelerated.relative_residual_max <= 2.0e-8
    np.testing.assert_allclose(
        accelerated.state.numpy(), baseline.state.numpy(), rtol=2.0e-8, atol=2.0e-10
    )
    np.testing.assert_allclose(
        accelerated.velocity.numpy(),
        baseline.velocity.numpy(),
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        accelerated.acceleration.numpy(),
        baseline.acceleration.numpy(),
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_cuda_material_and_mass_diagonals_match_independent_reference_oracle() -> None:
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=1,
        spanwise_element_count=1,
        chord=1.0,
        span=0.8,
        thickness=0.08,
    )
    model = Q16MITC16EASMesh(
        mesh,
        young_modulus=2.0e5,
        poisson_ratio=0.28,
        density=950.0,
    )
    stepper = _stepper(model, "material_jacobi")
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    linearization = stepper._operator._linearization_prechecked(state)
    cuda_material = stepper._operator._material_diagonal_linearized_prechecked(
        linearization
    ).numpy()[0]
    cuda_mass = stepper._operator._mass_diagonal_prechecked(1).numpy()[0]

    reference_material = np.empty(mesh.dof_count, dtype=np.float64)
    reference_mass = np.empty_like(reference_material)
    for dof in range(mesh.dof_count):
        basis = np.zeros(mesh.dof_count, dtype=np.float64)
        basis[dof] = 1.0
        reference_material[dof] = model.tangent_action(mesh.reference_state, basis)[dof]
        reference_mass[dof] = model.mass_action(basis)[dof]

    assert bool((cuda_material > 0.0).all())
    assert bool((cuda_mass > 0.0).all())
    np.testing.assert_allclose(
        cuda_material, reference_material, rtol=2.0e-14, atol=5.0e-10
    )
    np.testing.assert_allclose(cuda_mass, reference_mass, rtol=2.0e-14, atol=2.0e-15)


def test_material_jacobi_is_default_and_unknown_preconditioner_is_rejected() -> None:
    model, _ = _case()
    default = Q16CudaNewmarkStepper(
        model,
        make_clamped_span_root(model.mesh),
        device=config.DEVICE,
        newton_tolerance=2.0e-8,
        max_newton_iterations=15,
        cg_tolerance=2.0e-10,
        max_cg_iterations=1024,
        cg_check_every=16,
    )
    assert default.preconditioner == "material_jacobi"
    with pytest.raises(ValueError, match="preconditioner"):
        _stepper(model, "diagonal-ish")


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_jacobi_inverse_kernel_rejects_nonpositive_or_nonfinite_diagonal(
    bad_value: float,
) -> None:
    diagonal = wp.array(
        np.asarray([[1.0, bad_value]], dtype=np.float64),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    inverse = wp.zeros_like(diagonal)
    failure = wp.zeros(1, dtype=wp.int32, device=config.DEVICE)
    wp.launch(
        _q16_invert_jacobi_kernel,
        dim=diagonal.shape,
        inputs=[diagonal],
        outputs=[inverse, failure],
        device=config.DEVICE,
    )
    wp.synchronize_device(config.DEVICE)
    assert int(failure.numpy()[0]) == 1
    assert inverse.numpy()[0, 1] == 0.0
