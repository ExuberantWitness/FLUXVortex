"""P0 unified case schema tests (CPU): registry, roles, GT CSV, window search."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fluxvortex.cases.rojratsirikul2011 import (
    FIELD_ROLES,
    OBSERVATIONS_CSV,
    OBSERVATIONS_CSV_SHA256,
    ROJ_A10,
    ROJ_A16_E14,
    ROJ_A16_PRIMARY,
    ROJ_A17_MODE,
    ROJ_A23,
    ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES,
    ROJRATSIRIKUL2011_UNIFIED_CASES,
    cross_check_against_platform_adapter,
    validate_rojratsirikul2011_unified_sources,
)
from fluxvortex.runtime.case_runner import select_statistics_window


class TestUnifiedRegistry:
    def test_four_formal_members_registered(self) -> None:
        assert sorted(ROJRATSIRIKUL2011_UNIFIED_CASES) == [
            "ROJ11-A10",
            "ROJ11-A16",
            "ROJ11-A17-MODE",
            "ROJ11-A23",
        ]

    def test_a16_and_a17_are_separate_cases(self) -> None:
        # A16 gates only mean quantities; the 17-degree mode oracles must
        # never be applied to A16 (handoff §4.1).
        assert ROJ_A16_PRIMARY.target_strouhal is None
        assert ROJ_A16_PRIMARY.target_chordwise_peak_count is None
        assert ROJ_A16_PRIMARY.target_spanwise_peak_count is None
        assert ROJ_A17_MODE.angle_deg == 17.0
        assert ROJ_A17_MODE.target_strouhal == pytest.approx(0.85)
        assert ROJ_A17_MODE.target_spanwise_peak_count == 0

    def test_mode_case_st_tolerances(self) -> None:
        assert ROJ_A17_MODE.strouhal_tolerance == pytest.approx(0.08)
        assert ROJ_A23.strouhal_tolerance == pytest.approx(0.08)
        assert ROJ_A10.strouhal_tolerance == pytest.approx(0.10)

    def test_e14_is_calibration_sensitivity_only(self) -> None:
        assert ROJ_A16_E14.is_calibration_sensitivity is True
        assert ROJ_A16_E14.young_modulus_pa == pytest.approx(1.4e6)
        assert ROJ_A16_PRIMARY.young_modulus_pa == pytest.approx(2.2e6)
        assert (
            "ROJ11-A16-E14"
            in ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES
        )

    def test_field_roles_cover_every_target_and_assumption(self) -> None:
        required = {
            "chord_m",
            "span_m",
            "thickness_m",
            "young_modulus_pa",
            "kinematic_viscosity_m2_s",
            "fluid_density_kg_m3",
            "poisson_ratio_assumed",
            "structural_damping_loss_factor",
            "lesp_crit",
            "target_zmax_over_c",
            "target_cn_band",
        }
        assert required <= set(FIELD_ROLES)
        for role in FIELD_ROLES.values():
            assert role in {
                "paper_printed",
                "derived_from_printed_pairs",
                "digitized_approx",
                "model_assumption",
                "numerical_protocol",
            }

    def test_gt_csv_is_hash_frozen(self) -> None:
        import hashlib

        validate_rojratsirikul2011_unified_sources()
        observed = hashlib.sha256(OBSERVATIONS_CSV.read_bytes()).hexdigest()
        assert observed == OBSERVATIONS_CSV_SHA256

    def test_platform_cross_check_all_equal(self) -> None:
        report = cross_check_against_platform_adapter()
        assert report, "cross-check report must not be empty"
        assert all(report.values()), sorted(
            key for key, value in report.items() if not value
        )

    def test_platform_projection_carries_shared_physics(self) -> None:
        case = ROJ_A17_MODE.to_platform_case()
        assert case.angle_deg == 17.0
        assert case.chord_m == pytest.approx(0.0688)
        assert case.young_modulus_pa == pytest.approx(2.2e6)
        assert case.aerodynamic_dt_star == pytest.approx(0.01)


class TestWindowSelection:
    def _series(self, n: int, drift: float, noise: float, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        t = torch.arange(1, n + 1, dtype=torch.float64) * 0.01
        base = 1.0 + drift * t
        fast = 0.02 * torch.sin(2 * np.pi * 70.0 * t)
        wobble = noise * torch.randn(n, generator=generator)
        return t, base + fast + wobble

    def test_stationary_series_finds_window(self) -> None:
        n = 2100
        t, series = self._series(n, drift=0.0, noise=0.001)
        report = select_statistics_window(
            t, series, series, series, dt_star=0.01, startup_time_star=1.0
        )
        assert report["stationary_window_found"]
        chosen = report["window"]
        assert float(t[chosen.start].item()) > 1.0

    def test_drifting_series_is_rejected(self) -> None:
        n = 2100
        t, series = self._series(n, drift=0.02, noise=0.001)
        report = select_statistics_window(
            t, series, series, series, dt_star=0.01, startup_time_star=1.0
        )
        assert not report["stationary_window_found"]

    def test_window_excludes_startup(self) -> None:
        n = 2100
        t, series = self._series(n, drift=0.0, noise=0.001)
        series[:150] += 5.0  # violent startup transient
        report = select_statistics_window(
            t, series, series, series, dt_star=0.01, startup_time_star=1.0
        )
        assert report["window"].start >= 100
        assert all(
            candidate["start_time_star"] > 1.0
            for candidate in report["candidates"]
        )

    def test_candidates_cover_at_least_three_slow_periods(self) -> None:
        n = 2100
        t, series = self._series(n, drift=0.0, noise=0.001)
        report = select_statistics_window(
            t, series, series, series, dt_star=0.01, startup_time_star=1.0
        )
        chosen = report["candidates"][0] if report["candidates"] else None
        if chosen is not None:
            assert chosen["slow_mode_periods_covered"] >= 3.0 - 1e-9
