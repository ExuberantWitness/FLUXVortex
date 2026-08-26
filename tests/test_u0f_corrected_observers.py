"""U0-F corrected observers: paper metrics, sign-crossing, window serialization,
gates, stationarity, and case sensitivity labels.

Each test contrasts the corrected observer against the defective metric it
replaces (mean-of-maxes vs max-of-mean, end-point vs window mean,
instantaneous max(z) vs station sign series, intended vs actual window).
CPU-only.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from fluxvortex.cases.rojratsirikul2011 import (
    ROJ_A16_E14,
    ROJ_A16_PRIMARY,
    RojratsirikulCaseConfig,
)
from fluxvortex.runtime.result_schema import ResultStatus
from fluxvortex.validation.gates import (
    camber_gate,
    cn_gate,
    no_sign_crossing_gate,
)
from fluxvortex.validation.observers import (
    SignCrossingResult,
    StatisticsWindowRecord,
    mean_cn_over_window,
    mean_map_then_max,
    serialize_statistics_window,
    sign_crossing_at_station,
)
from fluxvortex.validation.stationarity import block_stationarity


# ---------------------------------------------------------------------------
# 1. Paper displacement metric: max(mean z), NOT mean(max z)
# ---------------------------------------------------------------------------

def _alternating_bump_history() -> torch.Tensor:
    """(T=8, nc=3, ns+1=4) field where a 0.10 bump alternates between two
    stations every step. Mean-map max is 0.05; mean of per-step maxes is 0.10."""
    T, nc, ns1 = 8, 3, 4
    z = torch.zeros(T, nc, ns1, dtype=torch.float64)
    station_a = (1, 1)
    station_b = (2, 3)
    for t in range(T):
        target = station_a if t % 2 == 0 else station_b
        z[t, target[0], target[1]] = 0.10
    return z


class TestMeanMapThenMax:
    def test_returns_max_of_mean_not_mean_of_max(self):
        z = _alternating_bump_history()
        wrong = float(z.max(dim=-1).values.max(dim=-1).values.mean().item())  # mean_t(max_xy(z))
        right = mean_map_then_max(z, slice(None))
        assert wrong == pytest.approx(0.10)
        assert right == pytest.approx(0.05)
        assert right != pytest.approx(wrong)

    def test_window_selects_rows_before_reducing(self):
        z = _alternating_bump_history()
        # Even steps only: station A carries the bump in every selected row,
        # so the mean map max is the full 0.10 — proving the mean is taken
        # over the window FIRST and the max afterwards.
        assert mean_map_then_max(z, slice(0, 8, 2)) == pytest.approx(0.10)
        assert mean_map_then_max(z, slice(1, 8, 2)) == pytest.approx(0.10)
        # Truncated window over the first two rows: each station active once.
        assert mean_map_then_max(z, slice(0, 2)) == pytest.approx(0.05)

    def test_uniform_field(self):
        z = torch.full((16, 3, 4), 0.043, dtype=torch.float64)
        assert mean_map_then_max(z, slice(None)) == pytest.approx(0.043)


# ---------------------------------------------------------------------------
# 2. Same-window time-averaged Cn (not an end-point)
# ---------------------------------------------------------------------------

class TestMeanCnOverWindow:
    def test_averages_over_window_not_end_point(self):
        cn = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        window = slice(1, 4)
        end_point = float(cn[window][-1].item())
        result = mean_cn_over_window(cn, window)
        assert end_point == pytest.approx(4.0)
        assert result == pytest.approx(3.0)  # (2+3+4)/3
        assert result != end_point

    def test_window_excludes_ramp_up(self):
        # Early transient ramp (0 -> 0.9) followed by a settled 0.93 plateau.
        cn = torch.cat([
            torch.linspace(0.0, 0.9, 20, dtype=torch.float64),
            torch.full((40,), 0.93, dtype=torch.float64),
        ])
        val = mean_cn_over_window(cn, slice(20, 60))
        assert val == pytest.approx(0.93, abs=1e-12)
        # Including the ramp would bias the mean low — the corrected window
        # must not silently pick it up.
        assert mean_cn_over_window(cn, slice(None)) < 0.93


# ---------------------------------------------------------------------------
# 3. Sign-crossing at a specified station (not instantaneous max(z))
# ---------------------------------------------------------------------------

def _oscillating_station_history() -> torch.Tensor:
    """(T=6, nc=3, ns+1=4) field with a +0.05 pedestal everywhere except one
    station whose series oscillates through zero, while the instantaneous
    field max stays positive at every step (so max(z) would MISS it)."""
    T, nc, ns1 = 6, 3, 4
    z = torch.full((T, nc, ns1), 0.05, dtype=torch.float64)
    station = (1, 2)
    oscillation = [0.02, -0.03, 0.03, -0.02, 0.01, -0.04]
    for t, v in enumerate(oscillation):
        z[t, station[0], station[1]] = float(v)
    return z


class TestSignCrossingAtStation:
    def test_detects_crossing_that_instantaneous_max_hides(self):
        z = _oscillating_station_history()
        instant_max = z.amax(dim=(1, 2))  # old defective metric
        assert float(instant_max.min().item()) > 0.0  # never crosses -> old gate passes
        result = sign_crossing_at_station(z, slice(None), chord_index=1, span_index=2)
        assert isinstance(result, SignCrossingResult)
        assert result.station == (1, 2)
        assert result.crossings == 5
        assert result.min_value == pytest.approx(-0.04)
        assert result.max_value == pytest.approx(0.03)
        assert result.crossings > 0  # corrected observer catches it

    def test_window_restriction_changes_count(self):
        z = _oscillating_station_history()
        first_two = sign_crossing_at_station(z, slice(0, 2), 1, 2)
        assert first_two.crossings == 1  # +0.02 -> -0.03
        flat = sign_crossing_at_station(
            torch.full((6, 3, 4), 0.05, dtype=torch.float64), slice(None), 0, 0
        )
        assert flat.crossings == 0
        assert flat.min_value > 0.0


# ---------------------------------------------------------------------------
# 4. Actual statistics window serialization
# ---------------------------------------------------------------------------

class TestSerializeStatisticsWindow:
    def test_records_actual_start_end_count(self):
        time_star = torch.arange(0, 11, dtype=torch.float64) * 0.1
        rec = serialize_statistics_window(time_star, slice(2, 9))
        assert isinstance(rec, StatisticsWindowRecord)
        assert rec.start_time_star == pytest.approx(0.2)
        assert rec.end_time_star == pytest.approx(0.8)
        assert rec.sample_count == 7
        assert rec.full_slow_mode_periods_covered == pytest.approx(-1.0)

    def test_slow_mode_periods_when_known(self):
        time_star = torch.arange(0, 11, dtype=torch.float64) * 0.1
        rec = serialize_statistics_window(time_star, slice(2, 9), slow_mode_period=0.25)
        assert rec.full_slow_mode_periods_covered == pytest.approx(0.6 / 0.25)

    def test_intended_vs_actual_can_diverge(self):
        # A run that ended early must serialize the ACTUAL window it used
        # (6 samples), not the intended 9-sample plan.
        time_star = torch.arange(0, 6, dtype=torch.float64) * 0.05
        rec = serialize_statistics_window(time_star, slice(None))
        assert rec.sample_count == 6
        assert rec.end_time_star == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 5. Acceptance gates
# ---------------------------------------------------------------------------

class TestGates:
    def test_camber_gate_within_tolerance(self):
        g = camber_gate(0.05, target=0.043, tolerance=0.01)
        assert g.gate_name == "mean_camber_max"
        assert g.passed is True
        assert g.value == pytest.approx(0.05)
        assert g.target == pytest.approx(0.043)
        assert g.tolerance == pytest.approx(0.01)

    def test_camber_gate_outside_tolerance(self):
        assert camber_gate(0.06, target=0.043, tolerance=0.01).passed is False
        assert camber_gate(0.02, target=0.043, tolerance=0.01).passed is False
        # Just inside the lower edge passes (<=).
        assert camber_gate(0.033, target=0.043, tolerance=0.01).passed is True

    def test_cn_gate_band(self):
        band = (0.92, 0.95)
        assert cn_gate(0.93, band, rel_tol=0.1).passed is True
        assert cn_gate(0.92, band, rel_tol=0.1).passed is True
        assert cn_gate(0.95, band, rel_tol=0.1).passed is True
        g_hi = cn_gate(1.05, band, rel_tol=0.1)
        assert g_hi.passed is False  # 1.05 > 1.1 * 0.95
        assert g_hi.target == pytest.approx(0.935)
        assert g_hi.detail.startswith("band [0.92,0.95]")

    def test_cn_gate_band_edges(self):
        band = (0.92, 0.95)
        assert cn_gate(0.829, band, rel_tol=0.1).passed is True    # just above 0.9 * lo
        assert cn_gate(0.82, band, rel_tol=0.1).passed is False    # below
        assert cn_gate(1.045, band, rel_tol=0.1).passed is True    # ~1.1 * hi
        assert cn_gate(1.06, band, rel_tol=0.1).passed is False    # above

    def test_no_sign_crossing_gate(self):
        ok = no_sign_crossing_gate(0)
        assert ok.passed is True
        assert ok.value == 0.0
        assert ok.target == 0.0
        bad = no_sign_crossing_gate(2)
        assert bad.passed is False
        assert bad.value == 2.0

    def test_full_chain_gates_on_corrected_observables(self):
        """Observers feed gates; a failing accuracy gate must give non-zero exit."""
        z = _alternating_bump_history()
        camber = camber_gate(mean_map_then_max(z, slice(None)), 0.043, 0.01)
        cn = cn_gate(
            mean_cn_over_window(
                torch.tensor([0.90, 0.92, 0.94, 0.95], dtype=torch.float64),
                slice(None),
            ),
            (0.92, 0.95),
            0.1,
        )
        crossing = no_sign_crossing_gate(
            sign_crossing_at_station(z, slice(None), 1, 2).crossings
        )
        status = ResultStatus(
            execution_status="completed",
            numerical_status="converged",
            physics_gate_status="passed",
            accuracy_gate_status="passed" if (camber.passed and cn.passed and crossing.passed) else "failed",
            reproduction_status="passed",
        )
        assert camber.passed is True   # 0.05 within 0.043 +/- 0.01
        assert cn.passed is True
        assert status.exit_code == 0

        failing = ResultStatus(
            execution_status="completed",
            numerical_status="converged",
            physics_gate_status="passed",
            accuracy_gate_status="failed",
            reproduction_status="pending",
        )
        assert failing.exit_code == 2  # accuracy failure -> non-zero exit


# ---------------------------------------------------------------------------
# 6. Block stationarity
# ---------------------------------------------------------------------------

class TestBlockStationarity:
    def test_trending_series_is_non_stationary(self):
        trend = torch.linspace(0.0, 10.0, 400, dtype=torch.float64)
        out = block_stationarity(trend, n_blocks=4)
        assert out["stationary"] is False
        # 4 blocks of 100 samples cover 99 linspace intervals each:
        # first/last block means ~1.244/8.756 -> spread ~7.519.
        assert out["block_spread"] == pytest.approx(7.5188, abs=1e-3)
        assert out["spread_over_std"] > 0.5

    def test_constant_series_is_stationary(self):
        const = torch.full((100,), 3.0, dtype=torch.float64)
        out = block_stationarity(const, n_blocks=4)
        assert out["stationary"] is True
        assert out["block_spread"] == 0.0
        assert out["series_std"] == 0.0

    def test_iid_noise_is_stationary(self):
        gen = torch.Generator().manual_seed(20260826)
        noise = torch.randn(400, generator=gen, dtype=torch.float64)
        out = block_stationarity(noise, n_blocks=4)
        assert out["stationary"] is True
        assert len(out["block_means"]) == 4
        assert out["spread_over_std"] < 0.5

    def test_block_mean_count_matches_n_blocks(self):
        series = torch.arange(80, dtype=torch.float64)
        assert len(block_stationarity(series, n_blocks=8)["block_means"]) == 8


# ---------------------------------------------------------------------------
# 7. Case config: E1.4 is a calibration sensitivity branch
# ---------------------------------------------------------------------------

class TestRojratsirikulCaseConfig:
    def test_primary_branch_is_not_calibration_sensitivity(self):
        assert ROJ_A16_PRIMARY.case_id == "ROJ11-A16"
        assert ROJ_A16_PRIMARY.angle_deg == pytest.approx(16.0)
        assert ROJ_A16_PRIMARY.material_branch == "primary_E2.2"
        assert ROJ_A16_PRIMARY.is_calibration_sensitivity is False

    def test_e14_flagged_as_calibration_sensitivity(self):
        assert ROJ_A16_E14.case_id == "ROJ11-A16-E14"
        assert ROJ_A16_E14.material_branch.startswith("post_hoc_calibrated_sensitivity")
        assert ROJ_A16_E14.is_calibration_sensitivity is True
        # It must never be mistaken for an independent verification branch.
        assert ROJ_A16_E14.is_calibration_sensitivity != ROJ_A16_PRIMARY.is_calibration_sensitivity

    def test_digitized_targets_shared(self):
        for cfg in (ROJ_A16_PRIMARY, ROJ_A16_E14):
            assert cfg.target_zmax_over_c == pytest.approx(0.043)
            assert cfg.target_cn_band == (0.92, 0.95)

    def test_config_is_frozen_and_hashable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ROJ_A16_PRIMARY.angle_deg = 12.0  # type: ignore[misc]
        # Frozen dataclass with a tuple band is hashable (usable in sets/keys).
        assert len({ROJ_A16_PRIMARY, ROJ_A16_E14}) == 2
        custom = RojratsirikulCaseConfig(
            case_id="ROJ11-A16-E22",
            angle_deg=16.0,
            target_zmax_over_c=0.043,
            target_cn_band=(0.92, 0.95),
        )
        assert custom.material_branch == "primary_E2.2"
        assert custom.is_calibration_sensitivity is False
