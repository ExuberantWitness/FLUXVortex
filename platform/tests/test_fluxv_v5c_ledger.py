from __future__ import annotations

import numpy as np
import pytest

from forward_flight_benchmarks.cases import IZRAELEVITZ_2017_FIG14_SCHERER
from forward_flight_benchmarks.fluxv_v5c_ledger import (
    ProfileDragReference,
    local_velocity_reference_delta,
    profile_drag_reference_delta,
    rebase_constant_profile_drag_reference,
    rebase_local_velocity_reference,
)
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_scherer_movement,
)
from forward_flight_benchmarks.uvlm_polar_correction import (
    FullAnglePolarParameters,
    movement_polar_residual,
)


def _kinematics(fraction: float, lift: np.ndarray, drag: np.ndarray) -> dict:
    phase = np.arange(lift.size, dtype=float) / lift.size
    return {
        "phase": phase,
        "unit_profile_drag_lift_n": np.asarray(lift, dtype=float),
        "unit_profile_drag_drag_n": np.asarray(drag, dtype=float),
        "parameters": {
            "section_velocity_reference_fraction_chord": fraction,
        },
    }


def _history(size: int) -> dict:
    phase = np.arange(size, dtype=float) / size
    lift = np.linspace(-2.0, 3.0, size)
    drag = np.linspace(0.2, 0.8, size)
    return {
        "phase": phase,
        "lift_n": lift,
        "drag_n": drag,
        "mean_lift_n": float(np.mean(lift)),
        "mean_drag_n": float(np.mean(drag)),
        "profile_drag_reference_fraction_chord": 0.25,
        "profile_drag_coefficient": 0.057,
    }


def test_profile_reference_delta_has_one_owner_and_correct_signs() -> None:
    owned = _kinematics(0.25, np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]))
    target = _kinematics(0.75, np.zeros(4), np.array([1.5, 2.5, 3.5, 4.5]))
    result = profile_drag_reference_delta(
        owned,
        target,
        ownership=ProfileDragReference(0.057, 0.25),
        target_fraction_chord=0.75,
        rho_kg_m3=2.0,
        freestream_m_s=2.0,
        area_m2=0.5,
    )
    np.testing.assert_allclose(result["delta_drag_n"], 0.057 * 0.5)
    np.testing.assert_allclose(result["delta_thrust_n"], -result["delta_drag_n"])
    np.testing.assert_allclose(result["delta_CT"], -result["delta_CD"])
    assert result["mean_delta_drag_n"] > 0.0


def test_disabled_rebase_is_exact_history_identity() -> None:
    history = _history(8)
    history["profile_drag_coefficient"] = 0.0
    owned = _kinematics(0.25, np.arange(8.0), np.arange(8.0) + 1.0)
    target = _kinematics(0.75, -np.arange(8.0), np.arange(8.0) + 2.0)
    result = rebase_constant_profile_drag_reference(
        history,
        owned,
        target,
        ownership=ProfileDragReference(0.0, 0.25),
        target_fraction_chord=0.75,
        rho_kg_m3=1.0,
        freestream_m_s=3.0,
        area_m2=2.0,
    )
    np.testing.assert_array_equal(result["lift_n"], history["lift_n"])
    np.testing.assert_array_equal(result["drag_n"], history["drag_n"])
    assert result["mean_lift_n"] == history["mean_lift_n"]
    assert result["mean_drag_n"] == history["mean_drag_n"]
    assert result["profile_drag_rebase_applied"] is False


def test_same_reference_preserves_nonzero_profile_owner_metadata() -> None:
    history = _history(8)
    owned = _kinematics(0.25, np.arange(8.0), np.arange(8.0) + 1.0)
    result = rebase_constant_profile_drag_reference(
        history,
        owned,
        owned,
        ownership=ProfileDragReference(0.057, 0.25),
        target_fraction_chord=0.25,
        rho_kg_m3=1.0,
        freestream_m_s=3.0,
        area_m2=2.0,
    )
    np.testing.assert_array_equal(result["lift_n"], history["lift_n"])
    np.testing.assert_array_equal(result["drag_n"], history["drag_n"])
    assert result["profile_drag_coefficient"] == 0.057
    assert result["profile_drag_reference_fraction_chord"] == 0.25
    assert result["profile_drag_rebase_applied"] is False


def test_rebase_updates_load_once_and_blocks_double_application() -> None:
    history = _history(8)
    owned = _kinematics(0.25, np.zeros(8), np.ones(8))
    target = _kinematics(0.75, np.zeros(8), 2.0 * np.ones(8))
    result = rebase_constant_profile_drag_reference(
        history,
        owned,
        target,
        ownership=ProfileDragReference(0.057, 0.25),
        target_fraction_chord=0.75,
        rho_kg_m3=1.0,
        freestream_m_s=2.0,
        area_m2=1.0,
    )
    np.testing.assert_allclose(result["drag_n"] - history["drag_n"], 0.057)
    np.testing.assert_allclose(result["thrust_n"], -result["drag_n"])
    assert result["profile_drag_reference_fraction_chord"] == 0.75
    assert result["profile_drag_rebase_applied"] is True

    with pytest.raises(ValueError, match="different profile reference"):
        rebase_constant_profile_drag_reference(
            result,
            owned,
            target,
            ownership=ProfileDragReference(0.057, 0.25),
            target_fraction_chord=0.75,
            rho_kg_m3=1.0,
            freestream_m_s=2.0,
            area_m2=1.0,
        )


def test_reference_and_phase_contracts_fail_closed() -> None:
    owned = _kinematics(0.25, np.zeros(4), np.ones(4))
    target = _kinematics(0.75, np.zeros(4), np.ones(4))
    target["phase"] = np.asarray(target["phase"]) + 0.01
    with pytest.raises(ValueError, match="phases are not aligned"):
        profile_drag_reference_delta(
            owned,
            target,
            ownership=ProfileDragReference(0.057, 0.25),
            target_fraction_chord=0.75,
            rho_kg_m3=1.0,
            freestream_m_s=1.0,
            area_m2=1.0,
        )

    aligned = _kinematics(0.75, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match="declared ledger ownership"):
        profile_drag_reference_delta(
            owned,
            aligned,
            ownership=ProfileDragReference(0.057, 0.5),
            target_fraction_chord=0.75,
            rho_kg_m3=1.0,
            freestream_m_s=1.0,
            area_m2=1.0,
        )


def test_polar_plus_profile_reference_is_replaced_not_stacked() -> None:
    history = _history(4)
    owned = _kinematics(0.25, np.zeros(4), np.ones(4))
    target = _kinematics(0.75, np.zeros(4), 2.0 * np.ones(4))
    owned["delta_lift_n"] = np.array([1.0, 2.0, 3.0, 4.0])
    owned["delta_drag_n"] = np.array([0.1, 0.2, 0.3, 0.4])
    target["delta_lift_n"] = np.array([2.0, 4.0, 6.0, 8.0])
    target["delta_drag_n"] = np.array([0.4, 0.6, 0.8, 1.0])
    ownership = ProfileDragReference(0.057, 0.25)
    delta = local_velocity_reference_delta(
        owned,
        target,
        ownership=ownership,
        target_fraction_chord=0.75,
        replace_polar_residual=True,
        rho_kg_m3=1.0,
        freestream_m_s=2.0,
        area_m2=1.0,
    )
    expected_drag = (
        target["delta_drag_n"]
        - owned["delta_drag_n"]
        + 0.057
        * (target["unit_profile_drag_drag_n"] - owned["unit_profile_drag_drag_n"])
    )
    np.testing.assert_allclose(delta["delta_drag_n"], expected_drag)

    rebased = rebase_local_velocity_reference(
        history,
        owned,
        target,
        ownership=ownership,
        target_fraction_chord=0.75,
        replace_polar_residual=True,
        rho_kg_m3=1.0,
        freestream_m_s=2.0,
        area_m2=1.0,
    )
    np.testing.assert_allclose(rebased["drag_n"], history["drag_n"] + expected_drag)
    assert rebased["polar_reference_replaced"] is True


def test_scherer_source_reference_executes_at_three_quarter_chord() -> None:
    case = IZRAELEVITZ_2017_FIG14_SCHERER
    movement, metadata = build_izraelevitz_scherer_movement(15.0, 60.0, "smoke")
    common = {
        "movement": movement,
        "source_cycle_step_range": (128, 191),
        "period_s": case.period_s,
        "freestream_m_s": case.freestream_m_s,
        "rho_kg_m3": case.rho_kg_m3,
        "aspect_ratio": case.aspect_ratio,
        "output_samples": 64,
    }
    quarter = movement_polar_residual(
        **common,
        parameters=FullAnglePolarParameters(
            section_velocity_reference_fraction_chord=0.25
        ),
    )
    source = movement_polar_residual(
        **common,
        parameters=FullAnglePolarParameters(
            section_velocity_reference_fraction_chord=case.pivot_fraction_chord
        ),
    )
    assert metadata["pivot_fraction_chord"] == case.pivot_fraction_chord == 0.75
    assert source["parameters"]["section_velocity_reference_fraction_chord"] == 0.75
    assert not np.allclose(
        quarter["unit_profile_drag_drag_n"],
        source["unit_profile_drag_drag_n"],
    )
