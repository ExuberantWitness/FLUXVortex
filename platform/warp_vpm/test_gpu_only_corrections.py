"""Frozen-oracle gates for CUDA-only V5M correction ledgers."""
# ruff: noqa: E402
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bing_drag_ledger import LedgerConfig
from bing_gpu_corrections import (
    ledger_step_cuda,
    movement_polar_residual_cuda,
    project_ldvm_delta_to_finite_wing_cuda,
    run_ldvm_separation_pair_cuda,
)
from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_cuda_ldvm_pair_and_projection_match_frozen_oracle() -> None:
    time = torch.arange(32, device="cuda:0", dtype=torch.float64) * 0.04
    alpha = 0.3 * torch.sin(time)
    pair = run_ldvm_separation_pair_cuda(
        alpha_rad=alpha,
        alpha_rate_per_convective_time=0.3 * torch.cos(time),
        heave_rate_over_u=0.1 * torch.sin(0.7 * time),
        delta_time_convective=0.04,
        pivot_fraction_chord=0.25,
        threshold=LESPThreshold(0.11, "flat", 1.0e4, "frozen unit oracle"),
        settings=LDVMSectionSettings(ndiv=20, naterm=8, max_wake_steps=64),
    )
    actual = torch.stack(
        [
            pair["delta"][field][-1]
            for field in ("CLf", "CDf", "CNc", "CNnc", "CNnonl", "CSf")
        ]
    )
    expected = torch.tensor(
        [
            0.4162459093011015,
            0.20152386829171764,
            -1.2484481493345951,
            1.5127276108437038,
            0.19173876279133734,
            -0.07694092283711232,
        ],
        device="cuda:0",
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=3.0e-13, atol=3.0e-13)
    projected = project_ldvm_delta_to_finite_wing_cuda(
        pair["delta"]["CNc"],
        pair["delta"]["CNnc"],
        pair["delta"]["CNnonl"],
        pair["delta"]["CSf"],
        alpha,
        aspect_ratio=3.0,
    )
    sums = torch.stack(
        (torch.sum(projected["delta_CL"]), torch.sum(projected["delta_CD"]))
    )
    torch.testing.assert_close(
        sums,
        torch.tensor(
            [0.7354020770334744, 0.43302928923412304],
            device="cuda:0",
            dtype=torch.float64,
        ),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert pair["numerical_contract"] == "torch-cuda-float64-no-cpu-fallback-v1"


def test_cuda_polar_residual_matches_frozen_oracle() -> None:
    movement = build_yang2025_movement(10.0, "full", settings=(2, 3, 8, 1, 1))
    if isinstance(movement, tuple):
        movement = movement[0]
    out = movement_polar_residual_cuda(
        movement,
        source_cycle_step_range=[0, 7],
        period_s=8 * movement.delta_time,
        freestream_m_s=1.0,
        rho_kg_m3=1.2,
        aspect_ratio=4.0,
        output_samples=8,
    )
    torch.testing.assert_close(
        out["mean_delta_force_g_n"],
        torch.tensor(
            [0.1181554509834248, 0.0012625283657037118, -0.01637286534893556],
            device="cuda:0",
            dtype=torch.float64,
        ),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert out["numerical_contract"] == "torch-cuda-float64-no-cpu-fallback-v1"


def test_cuda_drag_ledger_closes_after_projection() -> None:
    record = {
        "le_now": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "te_now": np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "v_rel_st": np.array([[1.0, 0.0, 0.3], [1.0, 0.0, 0.3]]),
        "areas": np.array([2.0]),
        "lesp": np.array([0.2]),
        "cn_strip": np.array([0.8]),
    }
    out = ledger_step_cuda(
        record,
        LedgerConfig(
            lesp_crit=0.1,
            aspect_ratio=4.0,
            rho=1.2,
            cd0=0.02,
            enable_t2=True,
        ),
    )
    torch.testing.assert_close(
        out["total"], out["t1"] + out["t2"] + out["t3"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        out["total"],
        torch.tensor([0.1300141443561664], device="cuda:0", dtype=torch.float64),
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_finite_wing_projection_rejects_cpu_tensor() -> None:
    values = torch.ones(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be CUDA"):
        project_ldvm_delta_to_finite_wing_cuda(
            values,
            values,
            values,
            values,
            values,
            aspect_ratio=3.0,
        )
