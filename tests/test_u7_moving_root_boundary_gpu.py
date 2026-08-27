"""U7-2: moving-root boundary updates over the production Q16 owner (GPU).

The A16 production model clamps the full membrane perimeter.  The moving
boundary transforms only the root edge (the x = 0 leading-edge nodes, the
same selection the fixed Yamano root uses) and re-uploads the device-side
prescribed state each step, leaving the frozen host owner untouched.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluxvortex.dynamics.boundary import MovingRootBoundary, MovingRootTransform
from fluxvortex.q16_boundary_constraints import Q16BoundaryConstraints
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_q16_constraints import Q16CudaBoundaryConstraints
from fluxvortex.warp_fsi.q16_structural_solver import (
    Q16CudaNewmarkStepper,
    Q16StructuralStepResult,
)
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_Q16_GRID,
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")

_ROOT_COUNT = 3 * FORMAL_Q16_GRID[1] + 1


def _identity() -> MovingRootTransform:
    return MovingRootTransform(
        position_I=np.zeros(3, dtype=np.float64),
        rotation_IB=np.eye(3, dtype=np.float64),
        velocity_I=np.zeros(3, dtype=np.float64),
        angular_velocity_I=np.zeros(3, dtype=np.float64),
    )


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _root_edge_node_ids(boundary: Q16BoundaryConstraints) -> np.ndarray:
    """The x = 0 leading-edge nodes of the pitched production mesh."""

    mesh = boundary.mesh
    scale = max(
        float(np.max(np.abs(mesh.reference_rows[:, :2]))), 1.0
    )
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    return np.ascontiguousarray(
        np.flatnonzero(np.abs(mesh.reference_rows[:, 0]) <= tolerance),
        dtype=np.int64,
    )


def _expected_prescribed_state(
    boundary: Q16BoundaryConstraints,
    mesh_rows: np.ndarray,
    root_ids: np.ndarray,
    transform: MovingRootTransform,
) -> np.ndarray:
    """Host-side reference for the full scattered prescribed-state vector."""

    full = np.zeros(boundary.mesh.dof_count, dtype=np.float64)
    full[boundary.constrained_dofs] = boundary.prescribed_values
    ref = mesh_rows[root_ids]
    positions = (transform.rotation_IB @ ref[:, :3].T).T + transform.position_I
    directors = (transform.rotation_IB @ ref[:, 3:].T).T
    root_dofs = (root_ids[:, None] * 6 + np.arange(6, dtype=np.int64)[None, :])
    full[root_dofs.reshape(-1)] = np.concatenate(
        [positions, directors], axis=1
    ).reshape(-1)
    return full


@pytest.fixture(scope="module")
def stack() -> SimpleNamespace:
    mesh, model, boundary, _ = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=ROJ11_A16,
    )
    root_ids = _root_edge_node_ids(boundary)
    operator = Q16CudaBoundaryConstraints(boundary, device=config.DEVICE)
    moving = MovingRootBoundary(boundary, root_ids, mesh.reference_rows)
    original = operator._prescribed_state.numpy().copy()
    return SimpleNamespace(
        mesh=mesh,
        model=model,
        boundary=boundary,
        root_ids=root_ids,
        operator=operator,
        moving=moving,
        original=original,
    )


def test_01_root_edge_is_the_leading_edge_subset() -> None:
    """The moving root owns exactly the x = 0 edge inside the perimeter."""

    mesh, _, boundary, _ = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=ROJ11_A16,
    )
    root_ids = _root_edge_node_ids(boundary)
    assert root_ids.size == _ROOT_COUNT
    assert bool(np.isin(root_ids, boundary.constrained_nodes).all())
    assert root_ids.size < boundary.constrained_nodes.size
    np.testing.assert_array_equal(
        mesh.reference_rows[root_ids, 0], np.zeros(root_ids.size)
    )


def test_02_identity_transform_preserves_frozen_reference_bitwise(stack) -> None:
    """Identity update leaves every prescribed value bit-identical."""

    stack.moving.update(_identity(), stack.operator)
    np.testing.assert_array_equal(
        stack.operator._prescribed_state.numpy(), stack.original
    )
    # The frozen host owner itself is untouched.
    np.testing.assert_array_equal(
        stack.boundary.prescribed_values,
        stack.boundary.mesh.reference_state[stack.boundary.constrained_dofs],
    )


def test_03_pure_translation_shifts_root_positions_exactly(stack) -> None:
    """A (0.01, 0, 0) translation moves only root x, directors unchanged."""

    shift = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    transform = MovingRootTransform(
        position_I=shift,
        rotation_IB=np.eye(3, dtype=np.float64),
        velocity_I=np.zeros(3, dtype=np.float64),
        angular_velocity_I=np.zeros(3, dtype=np.float64),
    )
    stack.moving.update(transform, stack.operator)
    device = stack.operator._prescribed_state.numpy()
    expected = _expected_prescribed_state(
        stack.boundary, stack.mesh.reference_rows, stack.root_ids, transform
    )
    np.testing.assert_allclose(device, expected, rtol=0.0, atol=1e-15)
    root_dofs = (stack.root_ids[:, None] * 6 + np.arange(6)[None, :]).reshape(-1)
    device_root = device[root_dofs].reshape(-1, 6)
    original_root = stack.original[root_dofs].reshape(-1, 6)
    np.testing.assert_allclose(
        device_root[:, 0] - original_root[:, 0],
        np.full(_ROOT_COUNT, 0.01),
        rtol=0.0,
        atol=1e-15,
    )
    # y/z positions, all directors, and every non-root constrained DOF are
    # bitwise unchanged.
    np.testing.assert_array_equal(device_root[:, 1:], original_root[:, 1:])
    non_root = np.setdiff1d(stack.boundary.constrained_dofs, root_dofs)
    np.testing.assert_array_equal(device[non_root], stack.original[non_root])


def test_04_pure_z_rotation_rotates_positions_and_directors(stack) -> None:
    """A 90-degree z-rotation rigidly rotates root rows about the origin."""

    transform = MovingRootTransform(
        position_I=np.zeros(3, dtype=np.float64),
        rotation_IB=_rot_z(math.pi / 2.0),
        velocity_I=np.zeros(3, dtype=np.float64),
        angular_velocity_I=np.zeros(3, dtype=np.float64),
    )
    stack.moving.update(transform, stack.operator)
    device = stack.operator._prescribed_state.numpy()
    expected = _expected_prescribed_state(
        stack.boundary, stack.mesh.reference_rows, stack.root_ids, transform
    )
    np.testing.assert_allclose(device, expected, rtol=0.0, atol=1e-15)
    root_dofs = (stack.root_ids[:, None] * 6 + np.arange(6)[None, :]).reshape(-1)
    device_root = device[root_dofs].reshape(-1, 6)
    original_root = stack.mesh.reference_rows[stack.root_ids]
    np.testing.assert_allclose(
        np.linalg.norm(device_root[:, :3], axis=1),
        np.linalg.norm(original_root[:, :3], axis=1),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.linalg.norm(device_root[:, 3:], axis=1),
        np.linalg.norm(original_root[:, 3:], axis=1),
        rtol=0.0,
        atol=1e-15,
    )
    # The pitched reference directors (h sin(alpha), 0, h cos(alpha)) are
    # rigidly rotated: compare against R @ d_ref and confirm they moved.
    np.testing.assert_allclose(
        device_root[:, 3:],
        original_root[:, 3:] @ transform.rotation_IB.T,
        rtol=0.0,
        atol=1e-15,
    )
    assert not np.array_equal(device_root[:, 3:], original_root[:, 3:])
    non_root = np.setdiff1d(stack.boundary.constrained_dofs, root_dofs)
    np.testing.assert_array_equal(device[non_root], stack.original[non_root])


def test_05_round_trip_transform_recovers_original_values(stack) -> None:
    """Transform then its SE(3) inverse recovers the frozen reference.

    Updates command absolute poses from the reference rows, so the inverse
    pose itself is the identity; the inverse map is checked on the moved
    values, and the identity pose restores the frozen bits exactly.
    """

    forward = MovingRootTransform(
        position_I=np.array([0.002, -0.001, 0.003], dtype=np.float64),
        rotation_IB=_rot_z(math.radians(30.0)),
        velocity_I=np.zeros(3, dtype=np.float64),
        angular_velocity_I=np.zeros(3, dtype=np.float64),
    )
    stack.moving.update(forward, stack.operator)
    device = stack.operator._prescribed_state.numpy()
    assert not np.array_equal(device, stack.original)
    root_dofs = (stack.root_ids[:, None] * 6 + np.arange(6)[None, :]).reshape(-1)
    moved = device[root_dofs].reshape(-1, 6)
    reference = stack.mesh.reference_rows[stack.root_ids]
    inverse_rotation = forward.rotation_IB.T
    # The SE(3) inverse (R^T, -R^T t) undoes the forward map on the moved
    # values: p_ref = R^T (v - t), d_ref = R^T d.
    np.testing.assert_allclose(
        (inverse_rotation @ (moved[:, :3] - forward.position_I).T).T,
        reference[:, :3],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        (inverse_rotation @ moved[:, 3:].T).T,
        reference[:, 3:],
        rtol=0.0,
        atol=1e-15,
    )
    # The inverse pose of the reference is the identity update.
    stack.moving.update(_identity(), stack.operator)
    np.testing.assert_array_equal(
        stack.operator._prescribed_state.numpy(), stack.original
    )


@pytest.fixture(scope="module")
def production_stepper():
    mesh, model, boundary, _ = make_rojratsirikul2011_q16_model(
        chordwise_element_count=FORMAL_Q16_GRID[0],
        spanwise_element_count=FORMAL_Q16_GRID[1],
        case=ROJ11_A16,
    )
    structural = Q16CudaNewmarkStepper(
        model,
        boundary,
        device=config.DEVICE,
        newton_tolerance=3.0e-7,
        max_newton_iterations=128,
        cg_tolerance=2.0e-10,
        max_cg_iterations=2048,
        cg_check_every=16,
        nonsymmetric_solver="reference_dense",
        reference_dense_refresh_after=48,
        mass_damping_coefficient=0.0,
        stiffness_damping_coefficient=(
            ROJ11_A16.structural_damping_loss_factor
            / (2.0 * math.pi * ROJ11_A16.freestream_m_s / ROJ11_A16.chord_m)
        ),
    )
    return SimpleNamespace(
        structural=structural, mesh=mesh, model=model, boundary=boundary
    )


def test_06_structural_solver_accepts_moving_root_boundary(
    production_stepper,
) -> None:
    """One Newmark step with a moved root edge returns finite kinematics."""

    structural = production_stepper.structural
    model = production_stepper.model
    boundary = production_stepper.boundary
    operator = structural._boundary_operator
    root_ids = _root_edge_node_ids(boundary)
    moving = MovingRootBoundary(boundary, root_ids, model.mesh.reference_rows)
    # A 1e-4 m transverse root shift: small enough to keep the first trial
    # near the stress-free reference, large enough to exercise the upload.
    transform = MovingRootTransform(
        position_I=np.array([0.0, 0.0, 1.0e-4], dtype=np.float64),
        rotation_IB=np.eye(3, dtype=np.float64),
        velocity_I=np.zeros(3, dtype=np.float64),
        angular_velocity_I=np.zeros(3, dtype=np.float64),
    )
    moving.update(transform, operator)
    prescribed = operator._prescribed_state.numpy()
    state = wp.array(
        np.ascontiguousarray(model.mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    enforced = operator.enforce_state(state)
    velocity = wp.zeros_like(enforced)
    acceleration = wp.zeros_like(enforced)
    force = wp.zeros_like(enforced)
    result = structural.step(
        enforced,
        velocity,
        acceleration,
        force,
        delta_time=ROJ11_A16.aerodynamic_dt_s,
    )
    assert type(result) is Q16StructuralStepResult
    for name, value in (
        ("state", result.state),
        ("velocity", result.velocity),
        ("acceleration", result.acceleration),
        ("reaction", result.reaction),
    ):
        host = value.numpy()
        assert bool(np.isfinite(host).all()), f"{name} contains non-finite values"
    state_end = result.state.numpy()[0]
    np.testing.assert_array_equal(
        state_end[boundary.constrained_dofs],
        prescribed[boundary.constrained_dofs],
    )
    assert result.relative_residual_max <= structural.newton_tolerance


def test_07_flapping_cycle_updates_repeatedly_and_returns_to_reference(stack) -> None:
    """A full flap cycle of updates tracks the analytic pose each substep."""

    period = 0.2
    substeps = 24
    amplitude = math.radians(5.0)
    heave = 2.0e-3
    rate = 2.0 * math.pi / period

    def transform_at(time: float) -> MovingRootTransform:
        angle = amplitude * math.sin(rate * time)
        lift = heave * math.sin(rate * time)
        angle_rate = amplitude * rate * math.cos(rate * time)
        lift_rate = heave * rate * math.cos(rate * time)
        return MovingRootTransform(
            position_I=np.array([0.0, 0.0, lift], dtype=np.float64),
            rotation_IB=_rot_y(angle),
            velocity_I=np.array([0.0, 0.0, lift_rate], dtype=np.float64),
            angular_velocity_I=np.array([0.0, angle_rate, 0.0], dtype=np.float64),
        )

    max_deviation = 0.0
    for index in range(substeps + 1):
        time = index * period / substeps
        transform = transform_at(time)
        stack.moving.update(transform, stack.operator)
        device = stack.operator._prescribed_state.numpy()
        expected = _expected_prescribed_state(
            stack.boundary, stack.mesh.reference_rows, stack.root_ids, transform
        )
        np.testing.assert_allclose(device, expected, rtol=0.0, atol=1e-15)
        max_deviation = max(
            max_deviation, float(np.max(np.abs(device - stack.original)))
        )
    assert max_deviation > 1.0e-6  # the boundary really moved mid-cycle
    # Closing the cycle with the exact identity transform restores the
    # frozen reference bitwise (sin(2*pi) is not exactly zero, so the last
    # analytic substep alone would leave ~1e-21 director residue).
    stack.moving.update(_identity(), stack.operator)
    np.testing.assert_array_equal(
        stack.operator._prescribed_state.numpy(), stack.original
    )

    # Velocity layer (§7.5): v = v_I + omega x r matches the analytic
    # time derivative of the transformed root positions.
    probe = transform_at(period / 8.0)
    velocity = stack.moving.root_velocities(probe)
    h = 1.0e-6
    forward_rows = stack.moving.prescribed_rows(transform_at(period / 8.0 + h))
    backward_rows = stack.moving.prescribed_rows(transform_at(period / 8.0 - h))
    numeric = (forward_rows[:, :3] - backward_rows[:, :3]) / (2.0 * h)
    np.testing.assert_allclose(velocity, numeric, rtol=0.0, atol=1.0e-6)


def test_08_update_prescribed_values_rejects_malformed_candidates(stack) -> None:
    """The CUDA update validates length, free-DOF zeros and finiteness."""

    dof_count = stack.boundary.mesh.dof_count
    base = np.zeros(dof_count, dtype=np.float64)
    with pytest.raises(ValueError):
        stack.operator.update_prescribed_values(
            wp.array(base[:-1], dtype=config.DTYPE, device=config.DEVICE)
        )
    with pytest.raises(ValueError):
        # The (1, n) packed layout is rejected: the operator stores the
        # full scattered prescribed state.
        stack.operator.update_prescribed_values(
            wp.array(base[None, :], dtype=config.DTYPE, device=config.DEVICE)
        )
    mask = np.zeros(dof_count, dtype=np.int32)
    mask[stack.boundary.constrained_dofs] = 1
    free_index = int(np.flatnonzero(mask == 0)[0])
    leak = base.copy()
    leak[free_index] = 1.0
    with pytest.raises(ValueError):
        stack.operator.update_prescribed_values(
            wp.array(leak, dtype=config.DTYPE, device=config.DEVICE)
        )
    nan = base.copy()
    nan[int(stack.boundary.constrained_dofs[0])] = np.nan
    with pytest.raises(FloatingPointError):
        stack.operator.update_prescribed_values(
            wp.array(nan, dtype=config.DTYPE, device=config.DEVICE)
        )
    with pytest.raises(TypeError):
        stack.operator.update_prescribed_values(
            wp.array(base, dtype=wp.float32, device=config.DEVICE)
        )
    with pytest.raises(TypeError):
        stack.operator.update_prescribed_values(base)
    # No malformed candidate changed the committed prescribed state.
    np.testing.assert_array_equal(
        stack.operator._prescribed_state.numpy(), stack.original
    )


def test_09_moving_root_inputs_are_validated(stack) -> None:
    """Transforms and ownership errors fail closed before any upload."""

    with pytest.raises(ValueError):
        MovingRootTransform(
            position_I=np.zeros(3),
            rotation_IB=2.0 * np.eye(3),
            velocity_I=np.zeros(3),
            angular_velocity_I=np.zeros(3),
        )
    with pytest.raises(ValueError):
        MovingRootTransform(
            position_I=np.zeros(3),
            rotation_IB=np.diag([1.0, 1.0, -1.0]),
            velocity_I=np.zeros(3),
            angular_velocity_I=np.zeros(3),
        )
    with pytest.raises(ValueError):
        MovingRootTransform(
            position_I=np.zeros(2),
            rotation_IB=np.eye(3),
            velocity_I=np.zeros(3),
            angular_velocity_I=np.zeros(3),
        )
    mask = np.zeros(stack.boundary.mesh.node_count, dtype=np.int32)
    mask[stack.boundary.constrained_nodes] = 1
    interior = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(ValueError):
        MovingRootBoundary(
            stack.boundary,
            np.ascontiguousarray(np.array([interior], dtype=np.int64)),
            stack.mesh.reference_rows,
        )
    with pytest.raises(ValueError):
        MovingRootBoundary(
            stack.boundary,
            np.ascontiguousarray(stack.root_ids[::-1].copy()),
            stack.mesh.reference_rows,
        )
    with pytest.raises(TypeError):
        MovingRootBoundary(stack.boundary, stack.root_ids, object())
    with pytest.raises(NotImplementedError):
        stack.moving.constraint_reaction()
