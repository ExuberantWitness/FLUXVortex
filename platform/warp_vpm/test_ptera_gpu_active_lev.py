"""Non-degenerate active-LEV regression for the strict CUDA Ptera backend."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pterasoftware as ps
from bing_joint_ptera import JointConfig, JointLEVTEVSolver
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _problem(
    num_steps: int = 10,
    *,
    delta_time: float = 0.04,
    velocity_amplitude: float = 0.0,
    velocity_period: float = 0.0,
) -> ps.problems.UnsteadyProblem:
    chord = 1.0
    span = 4.0
    outline = np.asarray(
        [[1.0, 1.0e-4], [0.5, 1.0e-4], [0.0, 0.0], [0.5, -1.0e-4], [1.0, -1.0e-4]]
    )
    airfoil = ps.geometry.airfoil.Airfoil(
        name="flat", outline_A_lp=outline, resample=False
    )
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=chord,
        num_spanwise_panels=3,
        spanwise_spacing="cosine",
    )
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=chord,
        Lp_Wcsp_Lpp=(0.0, span, 0.0),
        num_spanwise_panels=None,
        spanwise_spacing=None,
    )
    wing = ps.geometry.wing.Wing(
        name="flat",
        wing_cross_sections=[root, tip],
        angles_Gs_to_Wn_ixyz=(0.0, 20.0, 0.0),
        symmetric=False,
        num_chordwise_panels=2,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing], name="flat", s_ref=chord * span, c_ref=chord, b_ref=span
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=1.225, vCg__E=4.0, alpha=0.0, beta=0.0, nu=1.5e-5
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in (root, tip)
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=section_movements
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement]
    )
    operating_point_movement = (
        ps.movements.operating_point_movement.OperatingPointMovement(
            base_operating_point=operating_point,
            ampVCg__E=velocity_amplitude,
            periodVCg__E=velocity_period,
        )
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=operating_point_movement,
        delta_time=delta_time,
        num_steps=num_steps,
    )
    return ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)


def _forces(solver: object) -> np.ndarray:
    return np.asarray(
        [problem.airplanes[0].forces_W for problem in solver.steady_problems],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    "joint_tev, num_steps, tolerance",
    ((False, 10, 2.0e-10), (True, 6, 2.0e-9)),
)
def test_active_lev_cuda_contract_and_legacy_reference_avoid_host_hot_paths(
    monkeypatch: pytest.MonkeyPatch,
    joint_tev: bool,
    num_steps: int,
    tolerance: float,
) -> None:
    config = JointConfig(
        enable_lev=True,
        joint_tev=joint_tev,
        lev_start_step=1,
        particle_capacity=512,
        load_mode="bing",
    )
    expected = None
    if not joint_tev:
        reference = JointLEVTEVSolver(_problem(num_steps), config)
        reference.run(
            prescribed_wake=True, calculate_streamlines=False, show_progress=False
        )
        expected = _forces(reference)
        assert reference.lev_pf.n > 0

    solver = CudaJointLEVTEVSolver(_problem(num_steps), config, device="cuda:0")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("host aerodynamic numerical path was called")

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
        UnsteadyRingVortexLatticeMethodSolver, "_finalize_loads", forbidden
    )

    solver.run(
        prescribed_wake=not joint_tev,
        calculate_streamlines=False,
        show_progress=False,
    )
    actual = _forces(solver)
    assert solver.lev_pf.n > 0
    assert solver.lev_pf.positions_cuda.is_cuda
    assert solver.cuda_counters["particle_shed"] > 0
    assert solver.cuda_counters["particle_advance"] > 0
    assert solver.cuda_counters["particle_velocity"] > 0
    assert solver.cuda_counters["impulse"] > 0
    if joint_tev:
        assert solver._tev_solved is not None
        assert solver._tev_solved.is_cuda
        assert solver._prescribed_wake is False
        assert solver.cuda_counters["wake_convection"] == num_steps - 1
        assert max(row["kelvin_eq9_max_abs"] for row in solver.diag) <= 1.0e-12
        assert all(np.isfinite(row["lesp_max"]) for row in solver.diag)
        assert np.all(np.isfinite(actual))
    else:
        assert expected is not None
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
