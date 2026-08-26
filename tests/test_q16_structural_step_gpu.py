"""GPU-only nonlinear Newmark trial for the projected shared-node Q16 shell."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh, make_rectangular_q16_mesh
from fluxvortex.q16_boundary_constraints import make_clamped_span_root
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_structural_solver import (
    Q16CudaNewmarkStepper,
    Q16StructuralStepResult,
    Q16StructuralStepStopped,
)


pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def _model() -> Q16MITC16EASMesh:
    return Q16MITC16EASMesh(
        make_rectangular_q16_mesh(
            chordwise_element_count=1,
            spanwise_element_count=1,
            chord=1.0,
            span=0.8,
            thickness=0.08,
        ),
        young_modulus=2.0e5,
        poisson_ratio=0.28,
        density=950.0,
    )


def _inputs(
    model: Q16MITC16EASMesh, *, batch: int = 2, load_scale: float = 0.0
) -> tuple[wp.array, wp.array, wp.array, wp.array]:
    state = np.ascontiguousarray(
        np.repeat(model.mesh.reference_state[None, :], batch, axis=0),
        dtype=np.float64,
    )
    velocity = np.zeros_like(state)
    acceleration = np.zeros_like(state)
    force = np.zeros_like(state)
    tip_nodes = np.flatnonzero(model.mesh.reference_rows[:, 1] == 0.8)
    for sample in range(batch):
        force[sample, tip_nodes * 6 + 2] = load_scale * float(sample + 1)
    return tuple(
        wp.array(value, dtype=config.DTYPE, device=config.DEVICE)
        for value in (state, velocity, acceleration, force)
    )


def _stepper(
    model: Q16MITC16EASMesh,
    *,
    newton_tolerance: float = 2.0e-9,
    max_newton_iterations: int = 8,
    mass_damping_coefficient: float = 0.0,
) -> Q16CudaNewmarkStepper:
    return Q16CudaNewmarkStepper(
        model,
        make_clamped_span_root(model.mesh),
        device=config.DEVICE,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=2.0e-10,
        max_cg_iterations=256,
        cg_check_every=4,
        mass_damping_coefficient=mass_damping_coefficient,
    )


def test_zero_load_reference_batch_is_bitwise_stationary() -> None:
    model = _model()
    stepper = _stepper(model)
    state, velocity, acceleration, force = _inputs(model)
    before = tuple(value.numpy().copy() for value in (state, velocity, acceleration))
    result = stepper.step(
        state,
        velocity,
        acceleration,
        force,
        delta_time=2.0e-3,
    )
    assert type(result) is Q16StructuralStepResult
    np.testing.assert_array_equal(result.state.numpy(), before[0])
    np.testing.assert_array_equal(result.velocity.numpy(), before[1])
    np.testing.assert_array_equal(result.acceleration.numpy(), before[2])
    assert result.newton_iteration_count == 0
    assert result.cg_iteration_count == 0
    assert result.gmres_iteration_count == 0
    assert result.indefinite_fallback_count == 0
    assert result.relative_residual_max <= stepper.newton_tolerance


def test_high_modulus_reference_remains_discrete_stress_free() -> None:
    base = _model()
    model = Q16MITC16EASMesh(
        base.mesh,
        young_modulus=1.0e9,
        poisson_ratio=0.28,
        density=950.0,
    )
    stepper = _stepper(model)
    state, velocity, acceleration, force = _inputs(model, batch=1)
    result = stepper.step(
        state,
        velocity,
        acceleration,
        force,
        delta_time=4.0e-2,
    )

    np.testing.assert_array_equal(result.state.numpy(), state.numpy())
    np.testing.assert_array_equal(result.velocity.numpy(), velocity.numpy())
    np.testing.assert_array_equal(result.acceleration.numpy(), acceleration.numpy())
    assert result.newton_iteration_count == 0
    assert result.cg_iteration_count == 0
    assert result.gmres_iteration_count == 0
    assert result.indefinite_fallback_count == 0
    assert result.relative_residual_max == 0.0


def test_public_newmark_predictor_matches_zero_state_kinematics_and_is_detached() -> (
    None
):
    model = _model()
    stepper = _stepper(model)
    state, velocity, acceleration, _ = _inputs(model, batch=1)
    state_before = state.numpy().copy()
    velocity_before = velocity.numpy().copy()
    predicted_state, predicted_velocity = stepper.predict_kinematics(
        state,
        velocity,
        acceleration,
        delta_time=2.0e-3,
    )
    np.testing.assert_array_equal(predicted_state.numpy(), state_before)
    np.testing.assert_array_equal(predicted_velocity.numpy(), velocity_before)
    assert predicted_state is not state
    assert predicted_velocity is not velocity


def test_loaded_batched_step_converges_on_gpu_and_preserves_boundary() -> None:
    model = _model()
    boundary = make_clamped_span_root(model.mesh)
    stepper = _stepper(model)
    state, velocity, acceleration, force = _inputs(model, load_scale=2.0e-3)
    inputs_before = tuple(
        value.numpy().copy() for value in (state, velocity, acceleration, force)
    )
    delta_time = 2.0e-3
    result = stepper.step(
        state,
        velocity,
        acceleration,
        force,
        delta_time=delta_time,
    )
    for value in (result.state, result.velocity, result.acceleration, result.reaction):
        assert value.device.is_cuda
        assert value.dtype == config.DTYPE
        assert bool(np.isfinite(value.numpy()).all())
    np.testing.assert_array_equal(
        result.state.numpy()[:, boundary.constrained_dofs],
        np.repeat(boundary.prescribed_values[None, :], 2, axis=0),
    )
    np.testing.assert_array_equal(
        result.velocity.numpy()[:, boundary.constrained_dofs], 0.0
    )
    np.testing.assert_array_equal(
        result.acceleration.numpy()[:, boundary.constrained_dofs], 0.0
    )
    assert 1 <= result.newton_iteration_count <= 8
    assert result.cg_iteration_count >= 1
    assert result.gmres_iteration_count == 0
    assert result.indefinite_fallback_count == 0
    assert result.relative_residual_max <= stepper.newton_tolerance
    expected_state = (
        inputs_before[0]
        + delta_time * inputs_before[1]
        + delta_time
        * delta_time
        * (
            (0.5 - stepper.beta) * inputs_before[2]
            + stepper.beta * result.acceleration.numpy()
        )
    )
    expected_velocity = inputs_before[1] + delta_time * (
        (1.0 - stepper.gamma) * inputs_before[2]
        + stepper.gamma * result.acceleration.numpy()
    )
    np.testing.assert_allclose(
        result.state.numpy(), expected_state, rtol=0.0, atol=4.0e-15
    )
    np.testing.assert_allclose(
        result.velocity.numpy(), expected_velocity, rtol=0.0, atol=4.0e-15
    )
    for value, expected in zip(
        (state, velocity, acceleration, force), inputs_before, strict=True
    ):
        np.testing.assert_array_equal(value.numpy(), expected)


def test_mass_damping_is_in_effective_tangent_and_endpoint_work_ledger() -> None:
    model = _model()
    stepper = _stepper(model, mass_damping_coefficient=20.0)
    state, velocity, acceleration, force = _inputs(model, batch=1, load_scale=2.0e-3)
    tip_nodes = np.flatnonzero(model.mesh.reference_rows[:, 1] == 0.8)
    velocity_values = velocity.numpy()
    velocity_values[0, tip_nodes * 6 + 2] = 1.0e-2
    velocity = wp.array(
        np.ascontiguousarray(velocity_values),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    result = stepper.step(
        state,
        velocity,
        acceleration,
        force,
        delta_time=2.0e-3,
    )
    work = stepper.audit_step_work(
        state,
        velocity,
        acceleration,
        result.state,
        result.velocity,
        result.acceleration,
        force,
    )

    assert result.relative_residual_max <= stepper.newton_tolerance
    assert work.damping_trapezoidal_work > 0.0
    assert work.relative_balance_residual <= 1.0e-8


def test_nonconvergence_is_transactional_and_clean_retry_matches_fresh() -> None:
    model = _model()
    state, velocity, acceleration, force = _inputs(model, batch=1, load_scale=0.1)
    inputs_before = tuple(
        value.numpy().copy() for value in (state, velocity, acceleration, force)
    )
    failing = _stepper(model, newton_tolerance=1.0e-30, max_newton_iterations=1)
    with pytest.raises(Q16StructuralStepStopped, match="did not converge") as caught:
        failing.step(
            state,
            velocity,
            acceleration,
            force,
            delta_time=2.0e-3,
        )
    assert caught.value.phase in {"linear_solve", "newton_convergence"}
    for value, expected in zip(
        (state, velocity, acceleration, force), inputs_before, strict=True
    ):
        np.testing.assert_array_equal(value.numpy(), expected)

    retry = _stepper(model).step(
        state, velocity, acceleration, force, delta_time=2.0e-3
    )
    fresh_state, fresh_velocity, fresh_acceleration, fresh_force = _inputs(
        model, batch=1, load_scale=0.1
    )
    fresh = _stepper(model).step(
        fresh_state,
        fresh_velocity,
        fresh_acceleration,
        fresh_force,
        delta_time=2.0e-3,
    )
    np.testing.assert_array_equal(retry.state.numpy(), fresh.state.numpy())
    np.testing.assert_array_equal(retry.velocity.numpy(), fresh.velocity.numpy())
    np.testing.assert_array_equal(
        retry.acceleration.numpy(), fresh.acceleration.numpy()
    )


def test_stepper_rejects_host_dtype_shape_nonfinite_and_boundary_drift() -> None:
    model = _model()
    stepper = _stepper(model)
    state, velocity, acceleration, force = _inputs(model, batch=1)
    host = wp.array(state.numpy(), dtype=config.DTYPE, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        stepper.step(host, velocity, acceleration, force, delta_time=2.0e-3)
    wrong_dtype = wp.array(
        state.numpy().astype(np.float32), dtype=wp.float32, device=config.DEVICE
    )
    with pytest.raises(TypeError, match="float64"):
        stepper.step(wrong_dtype, velocity, acceleration, force, delta_time=2.0e-3)
    wrong_shape = wp.zeros(
        (1, model.mesh.dof_count - 1), dtype=config.DTYPE, device=config.DEVICE
    )
    with pytest.raises(ValueError, match="shape"):
        stepper.step(wrong_shape, velocity, acceleration, force, delta_time=2.0e-3)
    bad = state.numpy()
    bad[0, 10] = np.nan
    bad_state = wp.array(bad, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="non-finite"):
        stepper.step(bad_state, velocity, acceleration, force, delta_time=2.0e-3)

    boundary = make_clamped_span_root(model.mesh)
    drifted = state.numpy()
    drifted[0, boundary.constrained_dofs[0]] += 1.0e-12
    drifted_state = wp.array(drifted, dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(ValueError, match="boundary"):
        stepper.step(
            drifted_state,
            velocity,
            acceleration,
            force,
            delta_time=2.0e-3,
        )

    with pytest.raises(ValueError, match="positive"):
        stepper.step(state, velocity, acceleration, force, delta_time=0.0)
    with pytest.raises(ValueError, match="Newmark denominator"):
        stepper.step(state, velocity, acceleration, force, delta_time=5.0e-324)
    with pytest.raises(ValueError, match="Newmark denominator"):
        stepper.step(state, velocity, acceleration, force, delta_time=1.0e308)

    with pytest.raises(ValueError, match="less than one"):
        Q16CudaNewmarkStepper(
            model,
            make_clamped_span_root(model.mesh),
            device=config.DEVICE,
            newton_tolerance=1.0,
            max_newton_iterations=8,
            cg_tolerance=2.0e-10,
            max_cg_iterations=256,
            cg_check_every=4,
        )
