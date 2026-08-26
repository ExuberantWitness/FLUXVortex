"""Real one-step Q16 / separated-LEV / free-wake strong-coupling gates."""

from __future__ import annotations

import hashlib
import math
import pickle

import numpy as np
import pytest
import torch
import warp as wp

from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh, Q16Mesh
from fluxvortex.q16_boundary_constraints import make_clamped_span_root
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import Q16CudaLEVImpulseTransfer
from fluxvortex.warp_fsi.q16_ptera_resolved_transfer import (
    Q16CudaCompleteAeroLoadTransfer,
    Q16CudaPteraResolvedLoadTransfer,
)
from fluxvortex.warp_fsi.q16_prescribed_endpoint_load import (
    Q16CudaPrescribedEndpointLoad,
)
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from q16_incremental_ptera_owner import Q16CudaIncrementalAeroSession
from q16_ptera_trial_kinematics import (
    Q16CudaPteraIncrementalGeometry,
    Q16PteraPanelVertexTopology,
)
from q16_real_aero_branch_transaction import Q16CudaAeroSolverOwner
from q16_real_fsi_coupling import (
    Q16CudaRealFSIOwner,
    Q16CudaRealFSIStepper,
    Q16RealFSIStepResult,
    Q16RealFSIStepStopped,
)
from test_q16_incremental_trial_geometry import _solver
from test_q16_ptera_trial_kinematics import _mesh_and_map


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)


def _solver_hash(value: object) -> str:
    torch.cuda.synchronize()
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _warp_reference(mesh: object) -> wp.array:
    return wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )


def _zero_like(reference: wp.array) -> wp.array:
    return wp.zeros(reference.shape, dtype=config.DTYPE, device=config.DEVICE)


def _pitched_mesh_and_map(
    pitch_angle_degrees: float = 5.0,
) -> tuple[Q16Mesh, Q16SurfaceTransferMap]:
    base, transfer = _mesh_and_map()
    angle = math.radians(pitch_angle_degrees)
    rotation = np.array(
        [
            [math.cos(angle), 0.0, -math.sin(angle)],
            [0.0, 1.0, 0.0],
            [math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )
    rows = np.ascontiguousarray(base.reference_rows.copy())
    rows[:, :3] = rows[:, :3] @ rotation.T
    rows[:, 3:] = rows[:, 3:] @ rotation.T
    mesh = Q16Mesh(
        reference_rows=rows,
        connectivity=np.ascontiguousarray(base.connectivity.copy()),
    )
    return mesh, Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.ascontiguousarray(transfer.element_indices.copy()),
        parametric_coordinates=np.ascontiguousarray(
            transfer.parametric_coordinates.copy()
        ),
    )


def _build(
    *,
    coupling_tolerance: float = 2.0e-7,
    max_coupling_iterations: int = 10,
    newton_tolerance: float = 2.0e-9,
    max_newton_iterations: int = 15,
    relaxation: float = 0.7,
    aerodynamic_step_count: int = 3,
    aerodynamic_solver: object | None = None,
    young_modulus: float = 1.0e8,
    mass_damping_coefficient: float = 0.0,
    pitch_angle_degrees: float = 5.0,
    required_separated_source: str | None = None,
) -> tuple[Q16CudaRealFSIOwner, Q16CudaRealFSIStepper]:
    mesh, vertex_map = _pitched_mesh_and_map(pitch_angle_degrees)
    model = Q16MITC16EASMesh(
        mesh,
        young_modulus=young_modulus,
        poisson_ratio=0.28,
        density=950.0,
    )
    structural = Q16CudaNewmarkStepper(
        model,
        make_clamped_span_root(mesh),
        device=config.DEVICE,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=2.0e-10,
        max_cg_iterations=1024,
        cg_check_every=16,
        mass_damping_coefficient=mass_damping_coefficient,
    )
    binder = Q16CudaPteraIncrementalGeometry(
        vertex_map,
        Q16PteraPanelVertexTopology(2, 3),
        device=config.DEVICE,
    )
    state = _warp_reference(mesh)
    velocity = _zero_like(state)
    acceleration = _zero_like(state)
    session = Q16CudaIncrementalAeroSession.begin(
        _solver(steps=aerodynamic_step_count)
        if aerodynamic_solver is None
        else aerodynamic_solver
    )
    binder.bind_next_state(session, state, velocity)
    session.advance_one_step()
    aero_owner = Q16CudaAeroSolverOwner(session.solver)
    owner = Q16CudaRealFSIOwner(
        aero_owner=aero_owner,
        state=state,
        velocity=velocity,
        acceleration=acceleration,
    )
    complete = Q16CudaCompleteAeroLoadTransfer(
        Q16CudaPteraResolvedLoadTransfer(
            vertex_map,
            chordwise_panel_count=2,
            spanwise_panel_count=3,
            device=config.DEVICE,
        ),
        Q16CudaLEVImpulseTransfer(
            vertex_map,
            leading_edge_point_indices=np.arange(4, dtype=np.int64),
            device=config.DEVICE,
        ),
    )
    return owner, Q16CudaRealFSIStepper(
        structural_stepper=structural,
        binder=binder,
        complete_transfer=complete,
        coupling_tolerance=coupling_tolerance,
        max_coupling_iterations=max_coupling_iterations,
        relaxation=relaxation,
        required_separated_source=required_separated_source,
    )


def test_real_complete_load_converges_and_commits_both_owners_once() -> None:
    owner, stepper = _build()
    old_solver = owner.aero_owner.current_solver
    old_solver_hash = _solver_hash(old_solver)
    old_structural_hash = owner.state_sha256
    result = stepper.advance(owner, delta_time=0.04)

    assert type(result) is Q16RealFSIStepResult
    assert owner.generation == result.owner_generation == 1
    assert owner.aero_owner.generation == 1
    assert owner.aero_owner.current_solver is result.committed_solver
    assert owner.aero_owner.current_solver is not old_solver
    assert _solver_hash(old_solver) == old_solver_hash
    assert owner.state_sha256 != old_structural_hash
    np.testing.assert_array_equal(owner.state.numpy(), result.structural.state.numpy())
    assert 1 <= result.coupling_iteration_count <= 10
    assert result.aerodynamic_evaluation_count >= 2
    assert result.relative_residual <= stepper.coupling_tolerance
    assert float(np.linalg.norm(result.complete_load.generalized_force.numpy())) > 0.0
    committed = result.committed_solver
    assert committed._steps_done == 2
    assert committed.lev_pf.n > 0
    assert committed._tev_solved is not None
    assert committed._prescribed_wake is False
    assert committed.cuda_counters["wake_convection"] == 1


def test_prescribed_load_stays_distinct_from_aero_and_enters_structural_work() -> None:
    owner, stepper = _build()
    prescribed_tensor = torch.zeros_like(wp.to_torch(owner.state))
    prescribed_tensor[:, 2::6] = 2.0e-3
    prescribed = Q16CudaPrescribedEndpointLoad.from_force(
        wp.from_torch(prescribed_tensor, dtype=config.DTYPE),
        source_id="regression-uniform-z",
        endpoint_time_s=0.04,
    )

    result = stepper.advance(
        owner,
        delta_time=0.04,
        prescribed_load=prescribed,
    )
    aerodynamic = wp.to_torch(result.complete_load.generalized_force)
    independent = wp.to_torch(prescribed.generalized_force)
    total = wp.to_torch(result.total_external_force)

    assert result.prescribed_load is prescribed
    torch.testing.assert_close(total, aerodynamic + independent, rtol=0.0, atol=0.0)
    assert result.total_external_force_sha256
    assert result.work_balance.relative_balance_residual <= 1.0e-6
    assert result.committed_solver.diag[-1]["load_owner"] == (
        "ptera_plus_legacy_vortex_impulse"
    )


def test_two_structural_substeps_share_one_aero_transaction() -> None:
    # The interpolated block-start load exposes the known float64 Newton
    # residual floor of this tiny regression model; keep the gate above that
    # floor while remaining two orders tighter than the outer FSI tolerance.
    owner, stepper = _build(newton_tolerance=2.0e-8)
    force = torch.zeros_like(wp.to_torch(owner.state))
    force[:, 2::6] = 1.0e-4
    schedule = tuple(
        Q16CudaPrescribedEndpointLoad.from_force(
            wp.from_torch(force * scale, dtype=config.DTYPE),
            source_id="regression-two-rate-z",
            endpoint_time_s=endpoint,
        )
        for endpoint, scale in ((0.02, 0.5), (0.04, 1.0))
    )

    result = stepper.advance(
        owner,
        delta_time=0.04,
        structural_substep_count=2,
        prescribed_substep_loads=schedule,
        aerodynamic_substep_scheme="block_linear",
    )

    assert result.structural_substep_count == 2
    assert result.aerodynamic_substep_scheme == "block_linear"
    assert result.structural.direct_solve_count > 0
    assert result.structural.cg_iteration_count == 0
    assert result.structural.gmres_iteration_count == 0
    assert result.prescribed_substep_loads == schedule
    assert result.structural.delta_time == 0.04
    assert result.work_balance.relative_balance_residual <= 1.0e-6
    assert owner.generation == owner.aero_owner.generation == 1
    assert result.committed_solver.cuda_counters["wake_convection"] == 1


def test_nonconvergence_commits_neither_owner_and_clean_retry_matches_fresh() -> None:
    owner, failing = _build(
        coupling_tolerance=1.0e-30,
        max_coupling_iterations=1,
        relaxation=1.0,
    )
    parent = owner.aero_owner.current_solver
    parent_hash = _solver_hash(parent)
    structural_hash = owner.state_sha256
    with pytest.raises(Q16RealFSIStepStopped, match="did not converge") as caught:
        failing.advance(owner, delta_time=0.04)
    assert caught.value.phase == "coupling_convergence"
    assert owner.generation == 0
    assert owner.aero_owner.generation == 0
    assert owner.aero_owner.current_solver is parent
    assert _solver_hash(parent) == parent_hash
    assert owner.state_sha256 == structural_hash

    _, retry = _build()
    retry_result = retry.advance(owner, delta_time=0.04)
    fresh_owner, fresh = _build()
    fresh_result = fresh.advance(fresh_owner, delta_time=0.04)
    np.testing.assert_allclose(
        retry_result.structural.state.numpy(),
        fresh_result.structural.state.numpy(),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        retry_result.structural.velocity.numpy(),
        fresh_result.structural.velocity.numpy(),
        rtol=0.0,
        atol=2.0e-13,
    )
    assert (
        retry_result.committed_solver.cuda_counters
        == fresh_result.committed_solver.cuda_counters
    )
    for name, expected in fresh_result.committed_solver.lev_pf.snapshot_numpy().items():
        np.testing.assert_allclose(
            retry_result.committed_solver.lev_pf.snapshot_numpy()[name],
            expected,
            rtol=0.0,
            atol=2.0e-13,
        )


def test_second_aero_evaluation_failure_commits_neither_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, stepper = _build()
    parent = owner.aero_owner.current_solver
    parent_hash = _solver_hash(parent)
    structural_hash = owner.state_sha256
    original = Q16CudaCompleteAeroLoadTransfer.map
    calls = 0

    def fail_second(
        self: Q16CudaCompleteAeroLoadTransfer,
        packet: object,
        lev_impulse_load: object,
        structural_state: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second aerodynamic evaluation failure")
        return original(self, packet, lev_impulse_load, structural_state)

    monkeypatch.setattr(Q16CudaCompleteAeroLoadTransfer, "map", fail_second)
    with pytest.raises(RuntimeError, match="injected second"):
        stepper.advance(owner, delta_time=0.04)

    assert calls == 2
    assert owner.generation == 0
    assert owner.aero_owner.generation == 0
    assert owner.aero_owner.current_solver is parent
    assert _solver_hash(parent) == parent_hash
    assert owner.state_sha256 == structural_hash


def test_structural_owner_drift_fails_before_aero_branch_evaluation() -> None:
    owner, stepper = _build()
    parent = owner.aero_owner.current_solver
    owner.state.zero_()
    with pytest.raises(RuntimeError, match="structural owner drift"):
        stepper.advance(owner, delta_time=0.04)
    assert owner.generation == 0
    assert owner.aero_owner.generation == 0
    assert owner.aero_owner.current_solver is parent


def test_fsi_step_rejects_time_step_drift_before_aero_branch_evaluation() -> None:
    owner, stepper = _build()
    parent = owner.aero_owner.current_solver
    with pytest.raises(ValueError, match="aerodynamic delta_time"):
        stepper.advance(owner, delta_time=0.02)
    assert owner.generation == 0
    assert owner.aero_owner.generation == 0
    assert owner.aero_owner.current_solver is parent
