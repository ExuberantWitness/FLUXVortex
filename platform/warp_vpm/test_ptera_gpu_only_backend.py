"""Regression gates for the strict CUDA attached-flow Ptera backend."""
# ruff: noqa: E402
from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pterasoftware = pytest.importorskip("pterasoftware")

from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement

from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaAttachedJointLEVTEVSolver


_FROZEN_CPU_ORACLE_FORCES = np.array(
    [
        [-1.0287597567513486e-32, -4.889798628094506e-18, -8.400460912220662e-17],
        [0.04230924282875022, 0.22182513176341734, 0.15841579164153896],
        [0.027822416248351155, 0.2772396807773241, 0.32138165516712003],
        [-0.02361153341385339, 0.07077290788536697, 0.07437330786865376],
        [0.05279818361335325, 0.039619504688416976, -0.3017580585693031],
        [0.12248475610879489, 0.27190726139483656, -0.49367190797668087],
        [0.04788995701239867, 0.298948551075234, -0.43033940574987506],
        [-0.038003116308250456, 0.08218777056223159, -0.10966280444541643],
    ],
    dtype=np.float64,
)


def _small_solver() -> CudaAttachedJointLEVTEVSolver:
    movement = build_yang2025_movement(0.0, "full", settings=(2, 3, 8, 1, 1))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False
    )
    return CudaAttachedJointLEVTEVSolver(
        problem, JointConfig(enable_lev=False), device="cuda:0"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is mandatory")
def test_cuda_ptera_matches_frozen_oracle_without_cpu_hot_path(monkeypatch) -> None:
    solver = _small_solver()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU aerodynamic numerical path was called")

    from pterasoftware import _aerodynamics_functions as aero
    from pterasoftware import _functions
    from pterasoftware.unsteady_ring_vortex_lattice_method import (
        UnsteadyRingVortexLatticeMethodSolver,
    )

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(aero, "expanded_velocities_from_ring_vortices", forbidden)
    monkeypatch.setattr(aero, "collapsed_velocities_from_ring_vortices", forbidden)
    monkeypatch.setattr(_functions, "numba_1d_explicit_cross", forbidden)
    monkeypatch.setattr(_functions, "process_solver_loads", forbidden)
    monkeypatch.setattr(
        UnsteadyRingVortexLatticeMethodSolver,
        "_populate_next_airplanes_wake_vortex_points",
        forbidden,
    )
    monkeypatch.setattr(
        UnsteadyRingVortexLatticeMethodSolver,
        "_populate_next_airplanes_wake_vortices",
        forbidden,
    )
    monkeypatch.setattr(
        UnsteadyRingVortexLatticeMethodSolver,
        "_finalize_loads",
        forbidden,
    )

    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    forces = np.array(
        [problem.airplanes[0].forces_W for problem in solver.steady_problems]
    )
    np.testing.assert_allclose(
        forces, _FROZEN_CPU_ORACLE_FORCES, rtol=3.0e-13, atol=3.0e-13
    )
    assert solver.cuda_numerical_contract == "torch-cuda-float64-no-cpu-fallback-v1"
    assert solver.cuda_counters == {
        "aic": 8,
        "wake": 8,
        "solve": 8,
        "velocity": 0,
        "loads": 8,
        "ledger": 8,
        "wake_convection": 7,
    }


def test_cuda_ptera_rejects_non_cuda_device() -> None:
    movement = build_yang2025_movement(0.0, "full", settings=(2, 3, 8, 1, 1))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        CudaAttachedJointLEVTEVSolver(
            problem, JointConfig(enable_lev=False), device="cpu"
        )


def test_cuda_ptera_rejects_lev_configuration() -> None:
    movement = build_yang2025_movement(0.0, "full", settings=(2, 3, 8, 1, 1))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False
    )
    with pytest.raises(ValueError, match="LEV and joint_tev disabled"):
        CudaAttachedJointLEVTEVSolver(problem, JointConfig(enable_lev=True))


def test_cuda_ptera_rejects_host_final_result_mode() -> None:
    movement = build_yang2025_movement(0.0, "full", settings=(2, 3, 8, 1, 1))
    if isinstance(movement, tuple):
        movement = movement[0]
    problem = pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=True
    )
    with pytest.raises(ValueError, match="only_final_results=False"):
        CudaAttachedJointLEVTEVSolver(problem, JointConfig(enable_lev=False))


def test_cuda_wake_age_is_derived_on_device_not_read_from_host_objects() -> None:
    source = inspect.getsource(CudaAttachedJointLEVTEVSolver._wake_velocity)
    assert "_current_wake_vortex_ages" not in source
    assert "torch.arange" in source
    source = inspect.getsource(
        CudaAttachedJointLEVTEVSolver._populate_next_airplanes_wake_vortices
    )
    assert ".age" not in source
