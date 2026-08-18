from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import forward_flight_benchmarks.fluxv_v5b_sequence as sequence_module
from forward_flight_benchmarks.fluxv_v5b_sequence import (
    FORCE_OWNER,
    FluxVV5BSequenceConfig,
    FluxVV5BSequenceError,
    run_fluxv_v5b_force_sequence,
)
from forward_flight_benchmarks.fluxv_v5b_shared_wake import (
    FluxVV5BSharedWakeConfig,
)


def _rectangular_wing(nc: int, ns: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nc + 1)
    y = np.linspace(0.0, 1.5, ns + 1)
    corners = np.zeros((nc + 1, ns + 1, 3))
    corners[..., 0] = x[:, None]
    corners[..., 1] = y[None, :]
    return corners, np.zeros_like(corners)


def _pitched_wing(
    nc: int,
    ns: int,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    corners, velocity = _rectangular_wing(nc, ns)
    angle = np.deg2rad(alpha_deg)
    relative = corners[..., 0] - 0.25
    corners[..., 0] = 0.25 + relative * np.cos(angle)
    corners[..., 2] = relative * np.sin(angle)
    return corners, velocity


def _config(*, lesp_crit: float) -> FluxVV5BSequenceConfig:
    return FluxVV5BSequenceConfig(
        shared_wake=FluxVV5BSharedWakeConfig(
            nc=2,
            ns=3,
            u_infinity=(2.0, 0.0, 0.0),
            dt=0.01,
            lesp_crit=lesp_crit,
            core_radius=0.01,
        ),
        density=1.225,
        moment_origin=(0.25, 0.0, 0.0),
    )


def test_static_sequence_is_exact_no_lev_baseline_with_one_force_call_per_step() -> (
    None
):
    corners, velocity = _rectangular_wing(2, 3)
    corners_history = np.repeat(corners[None], 3, axis=0)
    velocity_history = np.repeat(velocity[None], 3, axis=0)
    real_force = sequence_module.fluxv_v5b_surface_force
    with patch.object(
        sequence_module,
        "fluxv_v5b_surface_force",
        wraps=real_force,
    ) as force_call:
        result = run_fluxv_v5b_force_sequence(
            _config(lesp_crit=10.0),
            corners_history,
            velocity_history,
        )

    assert force_call.call_count == 3
    np.testing.assert_array_equal(result.phase, [0.0, 1.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_array_equal(result.force_history_n, 0.0)
    np.testing.assert_array_equal(result.moment_history_nm, 0.0)
    assert result.guards.passed
    assert result.guards.force_ledger_count == 3
    assert result.guards.no_lev_steps == 3
    assert result.guards.no_lev_exact_reduction_passed
    assert result.guards.unique_force_owner
    assert result.manifest["force_owner"] == FORCE_OWNER
    assert result.manifest["force_evaluations_per_step"] == 1
    assert result.manifest["second_force_source"] == "forbidden"
    assert result.paper_scoring_status == "blocked_not_scored"
    assert not result.phase.flags.writeable
    assert not result.force_history_n.flags.writeable
    for witness in result.steps:
        assert witness.pristine_no_lev
        assert witness.no_lev_exact_reduction_passed
        assert witness.force_ledger.guards.no_lev_exact_reduction_required
        assert witness.force_ledger.guards.no_lev_exact_reduction_passed


def test_pitched_sequence_uses_saved_previous_bound_and_lev_chronologically() -> None:
    corners, velocity = _pitched_wing(2, 3, 15.0)
    corners_history = np.repeat(corners[None], 3, axis=0)
    velocity_history = np.repeat(velocity[None], 3, axis=0)
    result = run_fluxv_v5b_force_sequence(
        _config(lesp_crit=0.05),
        corners_history,
        velocity_history,
        phase=np.array([0.0, 0.25, 0.5]),
    )

    assert result.guards.passed
    assert result.guards.force_ledger_count == len(result.steps) == 3
    np.testing.assert_array_equal(result.steps[0].previous_bound_gamma, 0.0)
    np.testing.assert_array_equal(result.steps[0].previous_gamma_lev, 0.0)
    for previous, current in zip(result.steps[:-1], result.steps[1:], strict=True):
        np.testing.assert_array_equal(
            current.previous_bound_gamma,
            previous.pre_convection_report.bound_gamma,
        )
        np.testing.assert_array_equal(
            current.previous_gamma_lev,
            previous.pre_convection_report.gamma_lev,
        )
    assert np.all(np.isfinite(result.force_history_n))
    assert np.max(np.abs(result.force_history_n)) > 0.0
    assert result.guards.max_eq9_residual == 0.0
    assert result.guards.max_lesp_residual < 1.0e-12
    assert result.guards.max_material_gamma_change == 0.0
    for witness in result.steps:
        report = witness.pre_convection_report
        assert len(report.tev_pre_convection.rings) > 0
        assert len(report.lev_pre_convection.rings) > 0
        assert witness.force_ledger.guards.passed
        assert set(witness.force_ledger.pressure.pressure_channels) == {
            "surface_advection",
            "bound_unsteady",
            "lev_sheet_unsteady",
        }


def test_force_failure_returns_no_partial_sequence() -> None:
    corners, velocity = _pitched_wing(2, 3, 15.0)
    histories = np.repeat(corners[None], 2, axis=0)
    rates = np.repeat(velocity[None], 2, axis=0)
    bad_ledger = SimpleNamespace(guards=SimpleNamespace(passed=False))
    with patch.object(
        sequence_module,
        "fluxv_v5b_surface_force",
        return_value=bad_ledger,
    ) as force_call:
        with pytest.raises(FluxVV5BSequenceError, match="single-owner force ledger"):
            run_fluxv_v5b_force_sequence(
                _config(lesp_crit=0.05),
                histories,
                rates,
            )
    assert force_call.call_count == 1


def test_structured_history_and_phase_contract_fail_before_live_state() -> None:
    corners, velocity = _rectangular_wing(2, 3)
    config = _config(lesp_crit=10.0)
    with patch.object(sequence_module, "HiratoLiveShadow") as constructor:
        with pytest.raises(ValueError, match="corner_velocity_history"):
            run_fluxv_v5b_force_sequence(
                config,
                corners[None],
                np.repeat(velocity[None], 2, axis=0),
            )
        constructor.assert_not_called()

    with pytest.raises(ValueError, match="strictly increasing"):
        run_fluxv_v5b_force_sequence(
            config,
            np.repeat(corners[None], 2, axis=0),
            np.repeat(velocity[None], 2, axis=0),
            phase=[0.25, 0.25],
        )
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        run_fluxv_v5b_force_sequence(
            config,
            corners[None],
            velocity[None],
            phase=[1.0],
        )


def test_configuration_keeps_synthetic_scope_and_single_owner_explicit() -> None:
    manifest = _config(lesp_crit=0.05).manifest()
    assert manifest["status"] == "synthetic_sequence_only"
    assert manifest["paper_scoring_status"] == "blocked_not_scored"
    assert manifest["observation_fit"] == "none"
    assert manifest["force_owner"] == FORCE_OWNER
    assert manifest["shared_wake"]["equation_core"].endswith("hirato_live_shadow")
