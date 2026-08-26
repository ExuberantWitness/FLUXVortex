"""Yamano 2020 single-sheet provenance and CUDA Q16 modal gates."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
import warp as wp

from forward_flight_benchmarks.yamano2020_q16 import (
    LEGACY_MISINDEXED_TIP_EXPORT,
    YAMANO_2020_SINGLE_SHEET,
    load_nondimensional_natural_frequencies,
    load_mf2_history_oracle,
    load_tip_displacement_reference,
    make_yamano2020_q16_model,
    make_yamano2020_surface_transfer_map,
    validate_yamano2020_sources,
)
from reproduce_yamano2020_q16 import run_modal_case
from reproduce_yamano2020_q16_fsi import _build, run_case
from q16_real_fsi_coupling import (
    _Q16CudaFrozenAddedMassLoad,
    _Q16CudaInterpolatedAddedMassAction,
)
from bing_joint_ptera_gpu import _material_ring_velocity_derivative_expanded
from yamano2020_author_aero_projection import (
    build_yamano2020_author_added_mass_projection,
)
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_aero_load_packet import (
    Q16CudaAerodynamicLoadPacket,
)
from fluxvortex.warp_fsi.q16_lev_impulse_transfer import (
    Q16CudaLEVImpulseStripLoad,
)
from yamano2020_q16_pulse import Yamano2020Q16CudaPulse


def test_yamano2020_sources_and_dimensionalization_are_frozen() -> None:
    validate_yamano2020_sources()
    case = YAMANO_2020_SINGLE_SHEET
    np.testing.assert_array_equal(
        load_nondimensional_natural_frequencies(),
        np.asarray(
            [
                0.145542670589961,
                0.356709675448751,
                0.894175518802112,
                1.13932229697982,
                1.29800443934213,
            ],
            dtype=np.float64,
        ),
    )
    assert case.solid_density_kg_m3 == pytest.approx(1225.0, rel=0.0, abs=0.0)
    assert case.young_modulus_pa == pytest.approx(2.352e9, rel=1.0e-15)
    assert case.structural_dt_s == pytest.approx(2.0e-4, rel=1.0e-15)
    assert case.structural_substeps_per_aerodynamic_step == 34
    assert case.aerodynamic_dt_star == pytest.approx(0.068, rel=0.0, abs=0.0)
    assert case.aerodynamic_dt_s == pytest.approx(0.0068, rel=1.0e-15)
    assert case.pulse_duration_s == pytest.approx(0.02, rel=1.0e-15)
    assert case.pulse_nondimensional_at_time_s(0.0) == 0.0
    assert case.pulse_nondimensional_at_time_s(0.01) == pytest.approx(0.5)
    assert case.pulse_nondimensional_at_time_s(0.02) == 0.0
    assert case.pulse_body_force_density_n_m3(0.01) == pytest.approx(61250.0)
    tip = load_tip_displacement_reference()
    assert tip.shape == (501, 2)
    np.testing.assert_array_equal(tip[[0, -1], 0], [0.0, 1.0])
    assert tip[34, 0] == pytest.approx(0.068, rel=0.0, abs=1.0e-15)
    assert tip[34, 1] == pytest.approx(3.904553922604729e-4, rel=0.0, abs=1.0e-18)
    assert tip[34, 1] > 0.0


@pytest.mark.parametrize("step", (1, 2, 3, 4))
@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_cuda_wake_material_derivative_matches_matlab_per_panel(
    step: int,
) -> None:
    oracle = load_mf2_history_oracle()
    device = torch.device("cuda:0")

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(oracle[f"step_{step}_{name}"]),
            device=device,
            dtype=torch.float64,
        )

    derivative = _material_ring_velocity_derivative_expanded(
        tensor("rc_vec"),
        tensor("dt_rc_vec"),
        tensor("r_wake_1"),
        tensor("r_wake_2"),
        tensor("r_wake_3"),
        tensor("r_wake_4"),
        tensor("dt_r_wake_1"),
        tensor("dt_r_wake_2"),
        tensor("dt_r_wake_3"),
        tensor("dt_r_wake_4"),
    )
    wake_velocity_rate = torch.sum(
        derivative * tensor("Gamma_wake").unsqueeze(0).unsqueeze(2), dim=1
    )
    normal_rate = torch.sum(wake_velocity_rate * tensor("n_vec_i"), dim=1)
    expected_normal_rate = tensor("Gamma_wake_dt_q1234_n")
    mf2 = torch.linalg.solve(tensor("A_mat"), -normal_rate)
    expected_mf2 = tensor("Mf2_vec1")
    torch.testing.assert_close(
        normal_rate,
        expected_normal_rate,
        rtol=2.0e-13,
        atol=3.0e-16,
    )
    torch.testing.assert_close(mf2, expected_mf2, rtol=2.0e-13, atol=1.0e-16)
    if step == 1:
        assert torch.count_nonzero(mf2).item() == 0
    else:
        assert torch.linalg.vector_norm(mf2).item() > 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_q4_mf2_pressure_projection_matches_matlab_every_dof() -> None:
    step = 4
    oracle = load_mf2_history_oracle()
    device = torch.device("cuda:0")

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(oracle[f"step_{step}_{name}"]),
            device=device,
            dtype=torch.float64,
        )

    mesh, _, _ = make_yamano2020_q16_model(
        chordwise_element_count=5,
        spanwise_element_count=3,
    )
    projection = build_yamano2020_author_added_mass_projection(
        mesh,
        q16_chord_count=5,
        q16_span_count=3,
        aerodynamic_chord_count=15,
        aerodynamic_span_count=10,
        chord_length=1.0,
        span_length=1.0,
        aic=tensor("A_mat"),
        panel_normals=tensor("n_vec_i"),
        density=1.0,
    )
    combined_pressure = tensor("dp_lift1") + tensor("Mf2_vec1")
    actual = projection.q4_pressure_to_generalized @ combined_pressure
    torch.testing.assert_close(
        actual,
        tensor("Qf_p_global"),
        rtol=2.0e-13,
        atol=5.0e-19,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_v5m_scientific_frame_ignores_ptera_presentation() -> None:
    oracle = load_mf2_history_oracle()
    owner, stepper, _ = _build(
        outer_step_count=1,
        q16_chordwise_element_count=5,
        q16_spanwise_element_count=3,
        aerodynamic_chordwise_panel_count=15,
        aerodynamic_spanwise_panel_count=10,
        coordinate_frame="author",
    )
    solver = owner.aero_owner.current_solver
    assert solver.v5m_scientific_frame_schema == (
        "flux-v5m-author-x-chord-y-span-z-normal-v1"
    )
    presentation = np.asarray(
        solver.current_operating_point.T_pas_GP1_CgP1_to_W_CgP1,
        dtype=np.float64,
    )
    assert not np.allclose(presentation[:3, :3], np.eye(3))
    panels = solver.current_airplanes[0].wings[0].panels
    assert panels is not None
    assert panels[0, 0].Flpp_GP1_CgP1[0] == pytest.approx(0.0, abs=0.0)
    assert panels[-1, 0].Blpp_GP1_CgP1[0] == pytest.approx(1.0, abs=1.0e-15)
    np.testing.assert_allclose(panels[0, 0].unitNormal_GP1, [0.0, 0.0, 1.0])
    vertices = wp.to_torch(
        stepper.binder.surface_transfer.interpolate(owner.state)
    )[0]
    assert torch.min(vertices[:, 0]).item() == pytest.approx(0.0, abs=0.0)
    assert torch.max(vertices[:, 0]).item() == pytest.approx(1.0, abs=1.0e-15)
    assert solver._v5m_scientific_surface_contract_step == 0

    author_aic = solver._v5m_author_aic_cuda()
    matlab_aic = torch.as_tensor(
        np.array(oracle["step_1_A_mat"], dtype=np.float64, copy=True),
        device=solver.cuda_device,
        dtype=torch.float64,
    )
    cosine = torch.sum(author_aic * matlab_aic) / (
        torch.linalg.vector_norm(author_aic) * torch.linalg.vector_norm(matlab_aic)
    )
    norm_ratio = torch.linalg.vector_norm(author_aic) / torch.linalg.vector_norm(
        matlab_aic
    )
    assert cosine.item() > 0.998
    assert 0.98 < norm_ratio.item() < 1.02

    original_v_inf = solver.current_operating_point.vInf_GP1__E.copy()
    object.__setattr__(
        solver.current_operating_point,
        "_vInf_GP1__E",
        np.ascontiguousarray(-original_v_inf),
    )
    with pytest.raises(RuntimeError, match="chord.*downstream"):
        solver._require_v5m_scientific_surface_contract()
    object.__setattr__(
        solver.current_operating_point,
        "_vInf_GP1__E",
        np.ascontiguousarray(original_v_inf),
    )


def test_correct_tip_oracle_rejects_the_historical_three_dof_export() -> None:
    correct = load_tip_displacement_reference()
    left = correct[33]
    right = correct[34]
    beta = (0.0675 - left[0]) / (right[0] - left[0])
    correct_tip = left[1] + beta * (right[1] - left[1])
    with LEGACY_MISINDEXED_TIP_EXPORT.open(newline="") as stream:
        legacy = {
            float(row["Time"]): float(row["Z_tip"]) for row in csv.DictReader(stream)
        }
    assert correct_tip == pytest.approx(3.8250830776521516e-4, rel=0.0, abs=1.0e-18)
    assert legacy[0.0675] == pytest.approx(-1.18976242262246e-4, rel=0.0, abs=1.0e-18)
    assert correct_tip > 0.0 > legacy[0.0675]


def test_yamano2020_q16_geometry_clamps_only_the_leading_edge() -> None:
    mesh, model, constraints = make_yamano2020_q16_model(
        chordwise_element_count=2,
        spanwise_element_count=2,
    )
    rows = mesh.reference_rows
    expected_nodes = np.flatnonzero(rows[:, 0] == 0.0)
    np.testing.assert_array_equal(constraints.constrained_nodes, expected_nodes)
    assert model.total_reference_mass == pytest.approx(1.225, rel=5.0e-13)
    assert constraints.constrained_dof_count == 6 * expected_nodes.size
    assert np.all(rows[expected_nodes, 0] == 0.0)
    assert np.ptp(rows[expected_nodes, 1]) == pytest.approx(1.0)


def test_yamano2020_surface_map_owns_every_structural_macro_region() -> None:
    mesh, _, _ = make_yamano2020_q16_model(
        chordwise_element_count=2,
        spanwise_element_count=2,
    )
    transfer = make_yamano2020_surface_transfer_map(
        mesh,
        q16_chordwise_element_count=2,
        q16_spanwise_element_count=2,
        aerodynamic_chordwise_panel_count=4,
        aerodynamic_spanwise_panel_count=4,
    )
    assert transfer.point_count == 25
    np.testing.assert_array_equal(np.unique(transfer.element_indices), np.arange(4))
    assert np.max(np.abs(transfer.parametric_coordinates[:, :2])) <= 1.0
    assert np.all(transfer.parametric_coordinates[:, 2] == 0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_one_by_one_q16_modal_path_is_cuda_only() -> None:
    result = run_modal_case(1, 1)
    assert result["cuda_scientific_path"] is True
    assert result["cpu_numerical_fallback"] is False
    assert len(result["q16_omega_star"]) == 5
    assert result["stiffness_symmetry_relative"] <= 2.0e-11
    assert result["mass_symmetry_relative"] <= 2.0e-13
    assert result["minimum_mass_cholesky_diagonal"] > 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_two_by_two_q16_keeps_physical_low_modes() -> None:
    result = run_modal_case(2, 2)
    assert result["generalized_eigenvalue_minimum"] > 0.0
    assert result["clamped_mode_selection_threshold"] == 0.0
    assert result["relative_errors"][0] <= 0.02
    assert result["relative_errors"][1] <= 0.02
    assert result["max_relative_error"] <= 0.06


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_q16_pulse_is_consistent_and_cuda_resident() -> None:
    _, model, _ = make_yamano2020_q16_model(
        chordwise_element_count=2,
        spanwise_element_count=2,
    )
    pulse = Yamano2020Q16CudaPulse(model)
    peak = pulse.endpoint_load(0.01)
    finished = pulse.endpoint_load(0.02)
    peak_force = wp.to_torch(peak.generalized_force)
    finished_force = wp.to_torch(finished.generalized_force)

    assert peak_force.device.type == "cuda"
    assert peak_force.dtype is torch.float64
    assert pulse.total_force_z_n(0.01) == pytest.approx(61.25, rel=2.0e-13)
    assert torch.linalg.vector_norm(peak_force.reshape(1, -1, 6)[:, :, 3:]).item() <= (
        1.0e-12
    )
    assert torch.count_nonzero(finished_force).item() == 0
    assert peak.load_sha256 != finished.load_sha256


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_five_by_three_cached_tangent_matches_live_tangent() -> None:
    owner, fsi, pulse = _build(
        outer_step_count=1,
        q16_chordwise_element_count=5,
        q16_spanwise_element_count=3,
        aerodynamic_chordwise_panel_count=15,
        aerodynamic_spanwise_panel_count=10,
        coordinate_frame="author",
    )
    solver = owner.aero_owner.current_solver
    packet = Q16CudaAerodynamicLoadPacket.from_solver(solver)
    lev = Q16CudaLEVImpulseStripLoad.from_solver(solver)
    complete = fsi.complete_transfer.map(packet, lev, owner.state)
    frozen = _Q16CudaFrozenAddedMassLoad.from_solver(
        solver, complete, fsi.complete_transfer.resolved_transfer
    )
    added_mass = _Q16CudaInterpolatedAddedMassAction.between(
        frozen, frozen, 1.0
    )
    external_force = pulse.endpoint_load(0.0002).generalized_force
    structural = fsi.structural_stepper

    structural.nonsymmetric_solver = "direct"
    live = structural.step(
        owner.state,
        owner.velocity,
        owner.acceleration,
        external_force,
        delta_time=0.0002,
        acceleration_load_action=added_mass,
    )
    structural.nonsymmetric_solver = "reference_dense"
    cached = structural.step(
        owner.state,
        owner.velocity,
        owner.acceleration,
        external_force,
        delta_time=0.0002,
        acceleration_load_action=added_mass,
    )
    structural.reference_dense_refresh_after = 1
    hybrid = structural.step(
        owner.state,
        owner.velocity,
        owner.acceleration,
        external_force,
        delta_time=0.0002,
        acceleration_load_action=added_mass,
    )

    torch.testing.assert_close(
        wp.to_torch(cached.state),
        wp.to_torch(live.state),
        rtol=3.0e-11,
        atol=5.0e-19,
    )
    torch.testing.assert_close(
        wp.to_torch(hybrid.state),
        wp.to_torch(live.state),
        rtol=3.0e-11,
        atol=5.0e-19,
    )
    assert cached.direct_solve_count == live.direct_solve_count == 2
    assert cached.live_tangent_refresh_count == 0
    assert hybrid.live_tangent_refresh_count == 1
    assert cached.relative_residual_max <= structural.newton_tolerance


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_q16_distributed_added_mass_preserves_positive_initial_tip_acceleration() -> (
    None
):
    owner, fsi, pulse = _build(
        outer_step_count=1,
        q16_chordwise_element_count=2,
        q16_spanwise_element_count=2,
        aerodynamic_chordwise_panel_count=15,
        aerodynamic_spanwise_panel_count=10,
        coordinate_frame="author",
    )
    solver = owner.aero_owner.current_solver
    packet = Q16CudaAerodynamicLoadPacket.from_solver(solver)
    lev = Q16CudaLEVImpulseStripLoad.from_solver(solver)
    complete = fsi.complete_transfer.map(packet, lev, owner.state)
    added = _Q16CudaFrozenAddedMassLoad.from_solver(
        solver, complete, fsi.complete_transfer.resolved_transfer
    )
    matrix = added.generalized_matrix
    dof_count = matrix.shape[0]
    identity = torch.eye(dof_count, device=matrix.device, dtype=torch.float64)
    mass = (
        wp.to_torch(
            fsi.structural_stepper._operator._mass_action_prechecked(
                wp.from_torch(identity, dtype=config.DTYPE, requires_grad=False)
            )
        )
        .transpose(0, 1)
        .contiguous()
    )
    constrained = wp.to_torch(
        fsi.structural_stepper._boundary_operator._constrained_mask
    ).bool()
    free = torch.nonzero(~constrained, as_tuple=False).flatten()
    force = wp.to_torch(pulse.endpoint_load(0.01).generalized_force)[0]
    acceleration = torch.zeros_like(force)
    acceleration[free] = torch.linalg.solve((mass - matrix)[free][:, free], force[free])
    surface = wp.to_torch(
        fsi.binder.surface_transfer.interpolate(
            wp.from_torch(
                acceleration.unsqueeze(0),
                dtype=config.DTYPE,
                requires_grad=False,
            )
        )
    )[0]
    trailing_midspan = 15 * (10 + 1) + 5
    tip_acceleration = float(surface[trailing_midspan, 2].item())
    assert torch.linalg.vector_norm(matrix).item() > 0.0
    assert torch.min(surface[:, 2]).item() >= -1.0e-12
    assert added.column_scope == "author_aerodynamic_element_projection"
    # Frozen on the paper's 15x10 aerodynamic grid after its per-panel Q4 Mf1
    # blocks were projected to the Q16 field.  A 5x4 developer grid does not
    # represent the paper's local column topology and is intentionally not an
    # accuracy oracle.
    assert tip_acceleration == pytest.approx(
        42.533567319347796, rel=2.0e-11, abs=1.0e-11
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not wp.is_cuda_available(), reason="CUDA required"
)
def test_yamano2020_formal_q16_first_endpoint_passes_five_percent_gate() -> None:
    payload = run_case(
        outer_step_count=1,
        q16_chordwise_element_count=5,
        q16_spanwise_element_count=3,
        aerodynamic_chordwise_panel_count=15,
        aerodynamic_spanwise_panel_count=10,
        coordinate_frame="author",
    )
    record = payload["records"][0]

    assert payload["paper_reproduction_status"] == "FIRST_ENDPOINT_ACCURACY_PASS"
    assert payload["maximum_relative_tip_error"] <= 0.05
    assert record["q16_uvlm_author_circulatory_pressure_projection"] is True
    assert record["q16_author_corrector_beta_start"] == 0.0
    assert record["q16_author_corrector_beta_end"] == pytest.approx(33.0 / 34.0)
    assert record["structural_nonsymmetric_solver"] == "reference_dense"
    assert record["coupling_relative_residual"] <= 5.0e-7
    assert record["work_balance_relative_residual"] <= 1.0e-6
    assert record["lesp_pre_max_abs"] < payload["lcrit"]
    assert record["lev_particle_count"] == 0
    assert record["wake_convection_count"] == 1
