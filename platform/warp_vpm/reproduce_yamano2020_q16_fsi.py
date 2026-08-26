"""GPU-only Q16+DVM entry run for Yamano et al. (2020) single sheet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import warp as wp

import pterasoftware as ps
from bing_joint_ptera import JointConfig
from bing_joint_ptera_gpu import CudaJointLEVTEVSolver
from fluxvortex.q16_ancf_mesh import Q16MITC16EASMesh, Q16Mesh
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import (
    Q16CudaAerodynamicLoadPacket,
)
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
    Q16CudaLEVImpulseTransfer,
)
from fluxvortex.warp_fsi.q16_ptera_resolved_transfer import (
    Q16CudaCompleteAeroLoadTransfer,
    Q16CudaPteraResolvedLoadTransfer,
)
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    load_tip_displacement_reference,
    make_yamano2020_q16_model,
    make_yamano2020_surface_transfer_map,
)
from q16_incremental_ptera_owner import Q16CudaIncrementalAeroSession
from q16_ptera_trial_kinematics import (
    Q16CudaPteraIncrementalGeometry,
    Q16PteraPanelVertexTopology,
)
from q16_real_aero_branch_transaction import Q16CudaAeroSolverOwner
from q16_real_fsi_coupling import (
    Q16CudaRealFSIOwner,
    Q16CudaRealFSIStepper,
    _Q16CudaFrozenAuthorCirculatoryLoad,
    _Q16CudaFrozenAddedMassLoad,
    _Q16CudaFrozenKJVelocityLoad,
    _Q16CudaFrozenLift2VelocityLoad,
    _Q16CudaFrozenMf21VelocityLoad,
    _ptera_unsteady_generalized_force,
)
from yamano2020_q16_pulse import Yamano2020Q16CudaPulse


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, output)


def _problem(
    *,
    outer_step_count: int,
    chordwise_panel_count: int,
    spanwise_panel_count: int,
) -> ps.problems.UnsteadyProblem:
    case = YAMANO_2020_SINGLE_SHEET
    outline = np.asarray(
        [
            [1.0, 1.0e-4],
            [0.5, 1.0e-4],
            [0.0, 0.0],
            [0.5, -1.0e-4],
            [1.0, -1.0e-4],
        ],
        dtype=np.float64,
    )
    airfoil = ps.geometry.airfoil.Airfoil(
        name="yamano-flat-sheet",
        outline_A_lp=outline,
        resample=False,
    )
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.length_m,
        num_spanwise_panels=spanwise_panel_count,
        spanwise_spacing="uniform",
    )
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil,
        chord=case.length_m,
        Lp_Wcsp_Lpp=(0.0, case.width_m, 0.0),
        num_spanwise_panels=None,
        spanwise_spacing=None,
    )
    wing = ps.geometry.wing.Wing(
        name="yamano-single-sheet",
        wing_cross_sections=[root, tip],
        symmetric=False,
        num_chordwise_panels=chordwise_panel_count,
        chordwise_spacing="uniform",
    )
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing],
        name="yamano-single-sheet",
        s_ref=case.length_m * case.width_m,
        c_ref=case.length_m,
        b_ref=case.width_m,
    )
    operating_point = ps.operating_point.OperatingPoint(
        rho=case.fluid_density_kg_m3,
        vCg__E=case.freestream_m_s,
        alpha=0.0,
        beta=0.0,
        nu=1.5e-5,
    )
    section_movements = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=section
        )
        for section in (root, tip)
    ]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=section_movements,
    )
    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[wing_movement],
    )
    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=(
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point
            )
        ),
        delta_time=case.aerodynamic_dt_s,
        num_steps=outer_step_count + 1,
    )
    return ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)


def _q16_model(
    chordwise_element_count: int,
    spanwise_element_count: int,
    *,
    coordinate_frame: str,
) -> tuple[Q16Mesh, Q16MITC16EASMesh, object]:
    base, base_model, base_boundary = make_yamano2020_q16_model(
        chordwise_element_count=chordwise_element_count,
        spanwise_element_count=spanwise_element_count,
    )
    if coordinate_frame != "author":
        raise ValueError(
            "Yamano production FSI owns only the FLUX-V5M author frame"
        )
    return base, base_model, base_boundary


def _build(
    *,
    outer_step_count: int,
    q16_chordwise_element_count: int,
    q16_spanwise_element_count: int,
    aerodynamic_chordwise_panel_count: int,
    aerodynamic_spanwise_panel_count: int,
    coordinate_frame: str,
) -> tuple[
    Q16CudaRealFSIOwner,
    Q16CudaRealFSIStepper,
    Yamano2020Q16CudaPulse,
]:
    mesh, model, boundary = _q16_model(
        q16_chordwise_element_count,
        q16_spanwise_element_count,
        coordinate_frame=coordinate_frame,
    )
    transfer_map = make_yamano2020_surface_transfer_map(
        mesh,
        q16_chordwise_element_count=q16_chordwise_element_count,
        q16_spanwise_element_count=q16_spanwise_element_count,
        aerodynamic_chordwise_panel_count=aerodynamic_chordwise_panel_count,
        aerodynamic_spanwise_panel_count=aerodynamic_spanwise_panel_count,
    )
    topology = Q16PteraPanelVertexTopology(
        aerodynamic_chordwise_panel_count,
        aerodynamic_spanwise_panel_count,
    )
    structural = Q16CudaNewmarkStepper(
        model,
        boundary,
        device=config.DEVICE,
        # The 1x1 development mesh reaches a repeatable CUDA float64 residual
        # floor of 2.41--2.48e-7 under block-linear aero interpolation.  Keep
        # this inner gate below the frozen 5e-7 outer FSI tolerance without
        # changing any physical case parameter.
        newton_tolerance=3.0e-7,
        # The cached reference-tangent quasi-Newton path remains gated by the
        # same live 3e-7 residual.  Longer paper horizons can need 33--35
        # monotone corrections, so budget 48 without relaxing the gate.
        max_newton_iterations=48,
        cg_tolerance=2.0e-10,
        max_cg_iterations=2048,
        cg_check_every=16,
        nonsymmetric_solver=(
            "reference_dense"
            if q16_chordwise_element_count * q16_spanwise_element_count >= 15
            else "direct"
        ),
        reference_dense_refresh_after=24,
        mass_damping_coefficient=0.0,
    )
    binder = Q16CudaPteraIncrementalGeometry(
        transfer_map,
        topology,
        device=config.DEVICE,
    )
    solver = CudaJointLEVTEVSolver(
        _problem(
            outer_step_count=outer_step_count,
            chordwise_panel_count=aerodynamic_chordwise_panel_count,
            spanwise_panel_count=aerodynamic_spanwise_panel_count,
        ),
        JointConfig(
            enable_lev=True,
            joint_tev=True,
            lesp_crit=0.11,
            lev_start_step=0,
            separated_source="dvm_node_ribbon",
            particle_capacity=8192,
            dvm_ndiv=20,
            dvm_naterm=8,
            dvm_max_wake=64,
            # dump_traj_long.m uses max(panel_size, 1/Nx)*r_eps.fine =
            # 0.1*1e-6 = 1e-7 for this 15x10 unit sheet.  Ptera's generic
            # 0.03c core is far too diffusive for this paper's bound AIC.
            bound_core_radius_chord=1.0e-7,
            full_trailing_edge_bound_ring=True,
            q16_added_mass_column_scope=(
                "author_aerodynamic_element_projection"
            ),
        ),
        device=config.DEVICE,
    )
    solver._prescribed_wake = False
    state = wp.array(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    velocity = wp.zeros_like(state)
    acceleration = wp.zeros_like(state)
    session = Q16CudaIncrementalAeroSession.begin(solver)
    binder.bind_next_state(session, state, velocity)
    session.advance_one_step()
    owner = Q16CudaRealFSIOwner(
        aero_owner=Q16CudaAeroSolverOwner(session.solver),
        state=state,
        velocity=velocity,
        acceleration=acceleration,
    )
    complete = Q16CudaCompleteAeroLoadTransfer(
        Q16CudaPteraResolvedLoadTransfer(
            transfer_map,
            chordwise_panel_count=aerodynamic_chordwise_panel_count,
            spanwise_panel_count=aerodynamic_spanwise_panel_count,
            device=config.DEVICE,
        ),
        Q16CudaLEVImpulseTransfer(
            transfer_map,
            leading_edge_point_indices=np.arange(
                aerodynamic_spanwise_panel_count + 1, dtype=np.int64
            ),
            device=config.DEVICE,
        ),
    )
    stepper = Q16CudaRealFSIStepper(
        structural_stepper=structural,
        binder=binder,
        complete_transfer=complete,
        coupling_tolerance=5.0e-7,
        max_coupling_iterations=20,
        relaxation=0.7,
        relaxation_method="aitken",
        required_separated_source="dvm_node_ribbon",
    )
    return owner, stepper, Yamano2020Q16CudaPulse(model)


def run_case(
    *,
    outer_step_count: int = 1,
    q16_chordwise_element_count: int = 1,
    q16_spanwise_element_count: int = 1,
    aerodynamic_chordwise_panel_count: int = 5,
    aerodynamic_spanwise_panel_count: int = 4,
    coordinate_frame: str = "author",
    checkpoint_output: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("Yamano Q16 FSI reproduction requires CUDA")
    if outer_step_count <= 0:
        raise ValueError("outer_step_count must be positive")
    case = YAMANO_2020_SINGLE_SHEET
    owner, stepper, pulse = _build(
        outer_step_count=outer_step_count,
        q16_chordwise_element_count=q16_chordwise_element_count,
        q16_spanwise_element_count=q16_spanwise_element_count,
        aerodynamic_chordwise_panel_count=aerodynamic_chordwise_panel_count,
        aerodynamic_spanwise_panel_count=aerodynamic_spanwise_panel_count,
        coordinate_frame=coordinate_frame,
    )
    reference = torch.as_tensor(
        np.array(load_tip_displacement_reference(), copy=True),
        device="cuda:0",
        dtype=torch.float64,
    )
    reference_times = reference[:, 0].contiguous()
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for outer_index in range(outer_step_count):
        start_time_s = outer_index * case.aerodynamic_dt_s
        schedule = tuple(
            pulse.endpoint_load(start_time_s + (substep + 1) * case.structural_dt_s)
            for substep in range(case.structural_substeps_per_aerodynamic_step)
        )
        result = stepper.advance(
            owner,
            delta_time=case.aerodynamic_dt_s,
            structural_substep_count=(case.structural_substeps_per_aerodynamic_step),
            prescribed_substep_loads=schedule,
            aerodynamic_substep_scheme="author_corrector",
        )
        vertices = wp.to_torch(stepper.binder.surface_transfer.interpolate(owner.state))
        tip_index = (
            aerodynamic_chordwise_panel_count * (aerodynamic_spanwise_panel_count + 1)
            + aerodynamic_spanwise_panel_count // 2
        )
        tip_z = vertices[0, tip_index, 2]
        time_star = (outer_index + 1) * case.aerodynamic_dt_star
        reference_time = torch.as_tensor(
            time_star, device=reference.device, dtype=torch.float64
        )
        reference_right = int(
            torch.searchsorted(reference_times, reference_time).item()
        )
        if reference_right <= 0 or reference_right >= reference.shape[0]:
            raise RuntimeError(
                "Yamano FSI endpoint left the correct-index MATLAB oracle"
            )
        reference_left = reference_right - 1
        reference_beta = (reference_time - reference[reference_left, 0]) / (
            reference[reference_right, 0] - reference[reference_left, 0]
        )
        reference_tip = reference[reference_left, 1] + reference_beta * (
            reference[reference_right, 1] - reference[reference_left, 1]
        )
        solver = result.committed_solver
        endpoint_packet = Q16CudaAerodynamicLoadPacket.from_solver(solver)
        endpoint_lev = Q16CudaLEVImpulseStripLoad.from_solver(solver)
        endpoint_complete = stepper.complete_transfer.map(
            endpoint_packet, endpoint_lev, owner.state
        )
        endpoint_dgamma = _ptera_unsteady_generalized_force(
            endpoint_packet,
            stepper.complete_transfer.resolved_transfer,
            owner.state,
        )
        endpoint_kj = _Q16CudaFrozenKJVelocityLoad.from_solver(
            solver,
            endpoint_complete,
            stepper.complete_transfer.resolved_transfer,
        )
        endpoint_lift2 = _Q16CudaFrozenLift2VelocityLoad.from_solver(
            solver,
            endpoint_complete,
            stepper.complete_transfer.resolved_transfer,
            stepper.binder.surface_transfer,
            owner.state,
        )
        endpoint_added = _Q16CudaFrozenAddedMassLoad.from_solver(
            solver,
            endpoint_complete,
            stepper.complete_transfer.resolved_transfer,
        )
        endpoint_mf21 = _Q16CudaFrozenMf21VelocityLoad.from_solver(
            solver,
            endpoint_complete,
            stepper.complete_transfer.resolved_transfer,
            stepper.binder.surface_transfer,
            owner.state,
        )
        endpoint_author_circulatory = (
            _Q16CudaFrozenAuthorCirculatoryLoad.from_solver(
                solver,
                endpoint_complete,
                endpoint_lift2,
                endpoint_added,
            )
        )
        endpoint_kj_movement = endpoint_kj.map(owner.velocity)
        endpoint_circulatory_constant = endpoint_author_circulatory.map()
        endpoint_components = {
            "circulatory_constant": endpoint_circulatory_constant,
            "mf2_wake_motion": (
                endpoint_author_circulatory.wake_motion_generalized_force
            ),
            "ptera_dgamma_removed": endpoint_dgamma,
            "ptera_kj_movement_removed": endpoint_kj_movement,
            "lift2": endpoint_lift2.map(owner.velocity),
            "mf21": endpoint_mf21.map(owner.velocity),
            "mf1_acceleration": endpoint_added.map(owner.acceleration),
            "prescribed_pulse": schedule[-1].generalized_force,
        }
        endpoint_velocity = wp.to_torch(owner.velocity)
        endpoint_component_metrics: dict[str, dict[str, float]] = {}
        for component_name, component_force in endpoint_components.items():
            component = wp.to_torch(component_force)
            endpoint_component_metrics[component_name] = {
                "l2_n": float(torch.linalg.vector_norm(component).item()),
                "position_z_sum_n": float(torch.sum(component[:, 2::6]).item()),
                "power_w": float(torch.sum(component * endpoint_velocity).item()),
            }
        records.append(
            {
                "outer_step": outer_index + 1,
                "time_star": time_star,
                "tip_z_star": float(tip_z.item() / case.length_m),
                "reference_tip_z_star": float(reference_tip.item()),
                "reference_oracle": "matlab-h_X_vec-correct-9dof-index",
                "reference_dt_star": 2.0e-3,
                "reference_interpolated": True,
                "signed_tip_error_star": float(
                    (tip_z / case.length_m - reference_tip).item()
                ),
                "relative_tip_error": float(
                    (
                        torch.abs(tip_z / case.length_m - reference_tip)
                        / torch.abs(reference_tip)
                    ).item()
                ),
                "coupling_iterations": result.coupling_iteration_count,
                "aerodynamic_evaluations": result.aerodynamic_evaluation_count,
                "structural_substeps": result.structural_substep_count,
                "aerodynamic_substep_scheme": (result.aerodynamic_substep_scheme),
                "q16_ptera_kj_endpoint_term_removed": True,
                "q16_ptera_dgamma_removed_for_explicit_mf1": True,
                "q16_uvlm_lift2_substep_coupling": True,
                "q16_uvlm_lift2_distributed_pressure": True,
                "q16_uvlm_added_mass_substep_coupling": True,
                "q16_uvlm_distributed_pressure_projection": True,
                "q16_uvlm_author_p_interp": True,
                "q16_uvlm_author_circulatory_pressure_projection": True,
                "q16_uvlm_author_wake_motion_pressure": True,
                "q16_uvlm_author_wake_motion_cuda": True,
                "q16_uvlm_added_mass_column_scope": endpoint_added.column_scope,
                "q16_initial_added_mass_history_zero_in_committed_replay": False,
                "q16_predictor_zero_history_discarded": outer_index == 0,
                "q16_author_predictor_anchor": True,
                "q16_committed_corrector_anchor_to_endpoint": True,
                "q16_author_corrector_beta_start": 0.0,
                "q16_author_corrector_beta_end": (
                    (case.structural_substeps_per_aerodynamic_step - 1)
                    / case.structural_substeps_per_aerodynamic_step
                ),
                "q16_uvlm_mf21_substep_coupling": True,
                "q16_uvlm_mf21_distributed_pressure": True,
                "structural_newton_iterations": (
                    result.structural.newton_iteration_count
                ),
                "structural_cg_iterations": (result.structural.cg_iteration_count),
                "structural_gmres_iterations": (
                    result.structural.gmres_iteration_count
                ),
                "structural_gpu_direct_solves": (result.structural.direct_solve_count),
                "structural_live_tangent_refreshes": (
                    result.structural.live_tangent_refresh_count
                ),
                "structural_reference_dense_refresh_after": (
                    stepper.structural_stepper.reference_dense_refresh_after
                ),
                "structural_outer_tangent_cache_refresh_count": (
                    stepper.structural_stepper.reference_tangent_cache_refresh_count
                ),
                "structural_nonsymmetric_solver": (
                    stepper.structural_stepper.nonsymmetric_solver
                ),
                "structural_relative_residual": (
                    result.structural.relative_residual_max
                ),
                "coupling_relative_residual": result.relative_residual,
                "work_balance_relative_residual": (
                    result.work_balance.relative_balance_residual
                ),
                "lev_particle_count": solver.lev_pf.n,
                "lev_release_strip_count": solver.diag[-1]["lev_strips"],
                "lesp_pre_max_abs": solver.diag[-1]["lesp_pre_max_abs"],
                "wake_convection_count": solver.cuda_counters["wake_convection"],
                "load_owner": solver.diag[-1]["load_owner"],
                "production_impulse_count": solver.cuda_counters["impulse"],
                "endpoint_force_components": endpoint_component_metrics,
            }
        )
        if checkpoint_output is not None:
            _write_json_atomic(
                checkpoint_output,
                {
                    "schema": "yamano2020-q16-dvm-fsi-progress-v1",
                    "status": "RUNNING",
                    "completed_outer_steps": len(records),
                    "requested_outer_steps": outer_step_count,
                    "q16_mesh": [
                        q16_chordwise_element_count,
                        q16_spanwise_element_count,
                    ],
                    "aerodynamic_panels": [
                        aerodynamic_chordwise_panel_count,
                        aerodynamic_spanwise_panel_count,
                    ],
                    "lcrit": 0.11,
                    "separated_source": "dvm_node_ribbon",
                    "joint_tev": True,
                    "free_wake": True,
                    "cpu_numerical_fallback": False,
                    "records": records,
                },
            )
    torch.cuda.synchronize()
    maximum_relative_tip_error = max(
        float(record["relative_tip_error"]) for record in records
    )
    if maximum_relative_tip_error <= 0.05:
        reproduction_status = (
            "FIRST_ENDPOINT_ACCURACY_PASS"
            if outer_step_count == 1
            else "SHORT_TRAJECTORY_ACCURACY_PASS"
        )
    else:
        reproduction_status = "PARTIAL_ACCURACY_NOT_YET_MATCHED"
    return {
        "schema": "yamano2020-q16-dvm-fsi-case-entry-v3",
        "status": "CASE_ENTRY_PASS",
        "paper_reproduction_status": reproduction_status,
        "maximum_relative_tip_error": maximum_relative_tip_error,
        "paper_doi": case.doi,
        "case_id": case.case_id,
        "coordinate_frame": coordinate_frame,
        "q16_mesh": [
            q16_chordwise_element_count,
            q16_spanwise_element_count,
        ],
        "aerodynamic_panels": [
            aerodynamic_chordwise_panel_count,
            aerodynamic_spanwise_panel_count,
        ],
        "structural_dt_star": case.structural_dt_star,
        "aerodynamic_dt_star": case.aerodynamic_dt_star,
        "structural_substeps_per_aerodynamic_step": (
            case.structural_substeps_per_aerodynamic_step
        ),
        "lcrit": 0.11,
        "separated_source": "dvm_node_ribbon",
        "joint_tev": True,
        "free_wake": True,
        "cpu_numerical_fallback": False,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def _pair(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("value must be AxB")
    try:
        pair = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair values must be integers") from error
    if len(pair) != 2 or any(item <= 0 for item in pair):
        raise argparse.ArgumentTypeError("pair values must be positive")
    return pair[0], pair[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--q16-mesh", type=_pair, default=(1, 1))
    parser.add_argument("--aero-panels", type=_pair, default=(5, 4))
    parser.add_argument(
        "--coordinate-frame",
        choices=("author",),
        default="author",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically persist the JSON result instead of relying on stdout",
    )
    args = parser.parse_args()
    checkpoint_output = None
    if args.output is not None:
        checkpoint_output = args.output.with_name(
            f"{args.output.stem}.progress{args.output.suffix}"
        )
    payload = run_case(
        outer_step_count=args.outer_steps,
        q16_chordwise_element_count=args.q16_mesh[0],
        q16_spanwise_element_count=args.q16_mesh[1],
        aerodynamic_chordwise_panel_count=args.aero_panels[0],
        aerodynamic_spanwise_panel_count=args.aero_panels[1],
        coordinate_frame=args.coordinate_frame,
        checkpoint_output=checkpoint_output,
    )
    if args.output is not None:
        _write_json_atomic(args.output, payload)
        if checkpoint_output is not None:
            _write_json_atomic(checkpoint_output, payload)
        print(str(args.output.resolve()))
    else:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
