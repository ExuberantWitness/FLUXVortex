from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from claim_runtime.hirato_equations import HiratoEquationError
from forward_flight_benchmarks.fluxv_v5b_shared_wake import (
    CORE_STATUS,
    FORCE_COUPLING,
    SCORING_STATUS,
    FluxVV5BSharedWakeConfig,
    FluxVV5BSharedWakeCore,
    birth_limit_diagnostic,
    dispatch_v5b_or_parent,
)


def _rectangular_wing(nc: int, ns: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nc + 1)
    y = np.linspace(0.0, 1.0, ns + 1)
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
    relative_x = corners[..., 0] - 0.25
    corners[..., 0] = 0.25 + relative_x * np.cos(angle)
    corners[..., 2] = relative_x * np.sin(angle)
    return corners, velocity


def _config(*, lesp_crit: float) -> FluxVV5BSharedWakeConfig:
    return FluxVV5BSharedWakeConfig(
        nc=2,
        ns=2,
        u_infinity=(2.0, 0.0, 0.0),
        dt=0.01,
        lesp_crit=lesp_crit,
        core_radius=0.01,
    )


def test_no_lev_dispatch_returns_pristine_parent_without_constructing_state() -> None:
    sentinel = object()
    calls = 0

    def parent() -> object:
        nonlocal calls
        calls += 1
        return sentinel

    with patch(
        "forward_flight_benchmarks.fluxv_v5b_shared_wake." "FluxVV5BSharedWakeCore"
    ) as constructor:
        result = dispatch_v5b_or_parent(
            False,
            parent,
            config=None,
            corners_history=None,
            corner_velocity_history=None,
        )
    assert result is sentinel
    assert calls == 1
    constructor.assert_not_called()


def test_manifest_fails_closed_on_force_and_pressure_scoring() -> None:
    manifest = _config(lesp_crit=0.05).manifest()
    assert manifest["force_coupling"] == "not_implemented"
    assert manifest["pressure_coupling"] == "not_implemented"
    assert manifest["scoring_status"] == "blocked_not_scored"
    assert manifest["status"] == "no_force_diagnostic_only"
    assert manifest["observation_fit"] == "none"


def test_high_threshold_flat_sequence_has_no_lev_and_closes_no_force_ledgers() -> None:
    corners, velocity = _rectangular_wing(2, 2)
    core = FluxVV5BSharedWakeCore(_config(lesp_crit=10.0))
    result = core.run_sequence(
        [corners, corners, corners],
        [velocity, velocity, velocity],
    )
    assert result["step_count"] == 3
    assert result["force_coupling"] == FORCE_COUPLING
    assert result["status"] == CORE_STATUS
    assert result["scoring_status"] == SCORING_STATUS
    assert result["max_eq9_residual"] == 0.0
    assert result["max_kelvin_residual"] == 0.0
    assert result["max_lesp_residual"] == 0.0
    assert result["max_material_gamma_change"] == 0.0
    assert result["material_gamma_immutable"]
    for step in result["steps"]:
        assert not np.any(step["active"])
        assert step["new_lev_count"] == 0
        assert step["eq9_max_abs_residual"] == 0.0
        assert step["kelvin_max_abs_residual"] == 0.0
        assert step["material_gamma_immutable"]
        assert step["tev_birth_track_ratio_max_abs_error_from_0p3"] < 2.0e-15
        assert "force" not in step
        assert "pressure" not in step


def test_active_shared_wake_closes_eq9_lesp_and_material_gamma() -> None:
    corners, velocity = _pitched_wing(2, 2, 15.0)
    core = FluxVV5BSharedWakeCore(_config(lesp_crit=0.05))
    first = core.step(corners, velocity)
    second = core.step(corners, velocity)
    for report in (first, second):
        assert np.all(report["active"])
        assert report["new_lev_count"] == 2
        assert report["eq9_max_abs_residual"] == 0.0
        assert report["kelvin_max_abs_residual"] == 0.0
        assert report["lesp_max_abs_residual"] < 1.0e-13
        assert report["material_gamma_max_abs_change"] == 0.0
        assert report["material_gamma_immutable"]
        assert report["convection_ledger_max_abs_residual"] < 1.0e-13
        assert report["birth_gamma_max_abs"] > 0.0
        assert report["force_coupling"] == "not_implemented"
        assert report["scoring_status"] == "blocked_not_scored"
    np.testing.assert_allclose(
        second["gamma_tev_eq9"],
        first["bound_gamma"].reshape(2, 2)[-1] + first["gamma_lev_lesp"],
        atol=1.0e-14,
        rtol=0.0,
    )


def test_material_gamma_audit_detects_between_step_mutation() -> None:
    corners, velocity = _pitched_wing(2, 2, 15.0)
    core = FluxVV5BSharedWakeCore(_config(lesp_crit=0.05))
    core.step(corners, velocity)
    core._shadow.tev.gamma[0] += 1.0e-6
    report = core.step(corners, velocity)
    assert not report["material_gamma_immutable"]
    assert report["material_gamma_max_abs_change"] == pytest.approx(1.0e-6)
    assert not report["material_gamma_checks"]["between_steps"]["passed"]


def test_enabled_dispatch_returns_only_blocked_no_force_sequence() -> None:
    corners, velocity = _pitched_wing(2, 2, 15.0)
    parent_calls = 0

    def parent() -> object:
        nonlocal parent_calls
        parent_calls += 1
        return object()

    result = dispatch_v5b_or_parent(
        True,
        parent,
        config=_config(lesp_crit=0.05),
        corners_history=[corners, corners],
        corner_velocity_history=[velocity, velocity],
    )
    assert parent_calls == 0
    assert result["step_count"] == 2
    assert result["force_coupling"] == "not_implemented"
    assert result["scoring_status"] == "blocked_not_scored"


def test_birth_limit_diagnostic_recovers_positive_power_law() -> None:
    time_steps = np.array([0.02, 0.01, 0.005, 0.0025])
    exponent = 1.25
    births = 0.7 * time_steps**exponent
    report = birth_limit_diagnostic(time_steps, births)
    assert report["slope_p"] == pytest.approx(exponent, abs=2.0e-14)
    np.testing.assert_allclose(report["local_orders"], exponent)
    assert report["tends_to_zero"]
    assert report["force_coupling"] == "not_implemented"
    assert report["scoring_status"] == "blocked_not_scored"

    with pytest.raises(ValueError):
        birth_limit_diagnostic(time_steps[:2], births[:2])
    with pytest.raises(ValueError):
        birth_limit_diagnostic(time_steps, np.array([*births[:3], 0.0]))


def test_step_and_sequence_refuse_nonchronological_or_misaligned_inputs() -> None:
    corners, velocity = _rectangular_wing(2, 2)
    core = FluxVV5BSharedWakeCore(_config(lesp_crit=10.0))
    with pytest.raises(HiratoEquationError):
        core.step(corners, velocity, step=1)
    with pytest.raises(ValueError):
        core.run_sequence([corners], [])
