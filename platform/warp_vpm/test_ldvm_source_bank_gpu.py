"""Parity and transaction gates for the span-parallel CUDA DVM source bank."""
from __future__ import annotations

import pickle

import pytest
import torch

from ldvm_source_bank_gpu import CudaLDVMSourceBank
from ldvm_torch_gpu import LDVM2DCuda


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _kinematics(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0")
    dtype = torch.float64
    alpha = torch.tensor(
        [0.20 + 0.025 * step, 0.45 - 0.01 * step, -0.32 + 0.02 * step],
        device=device,
        dtype=dtype,
    )
    alpha_rate = torch.tensor(
        [0.08, -0.035, 0.055], device=device, dtype=dtype
    )
    heave = torch.tensor([0.0, 0.03, -0.02], device=device, dtype=dtype)
    return alpha, alpha_rate, heave


def _bank() -> CudaLDVMSourceBank:
    device = torch.device("cuda:0")
    return CudaLDVMSourceBank(
        batch_size=3,
        ndiv=20,
        naterm=8,
        delta_time_convective=torch.tensor(
            [0.01, 0.012, 0.009], device=device, dtype=torch.float64
        ),
        lesp_crit=torch.tensor(
            [0.05, 0.07, 0.06], device=device, dtype=torch.float64
        ),
        pivot_fraction_chord=torch.tensor(
            [0.25, 0.30, 0.20], device=device, dtype=torch.float64
        ),
        core_radius_chord=torch.tensor(
            [0.02, 0.025, 0.018], device=device, dtype=torch.float64
        ),
        max_wake=32,
        source_parity=True,
        device="cuda:0",
    )


def _lanes() -> list[LDVM2DCuda]:
    return [
        LDVM2DCuda(
            U=1.0,
            c=1.0,
            ndiv=20,
            naterm=8,
            dt=dt,
            rho=1.0,
            lesp_crit=critical,
            pivot_xc=pivot,
            core_rc=core,
            max_wake=32,
            source_parity=True,
            device="cuda:0",
        )
        for dt, critical, pivot, core in zip(
            (0.01, 0.012, 0.009),
            (0.05, 0.07, 0.06),
            (0.25, 0.30, 0.20),
            (0.02, 0.025, 0.018),
            strict=True,
        )
    ]


def test_source_bank_matches_independent_cuda_lanes() -> None:
    bank = _bank()
    lanes = _lanes()
    compared = (
        "A0",
        "lesp_pre",
        "lesp_constraint_residual",
        "shed_lev",
        "gamma_lev_new",
        "gamma_tev_new_solved",
        "gamma_tev_new_persisted",
        "lev_birth_position",
        "tev_birth_position",
        "lev_edge_position",
        "tev_edge_position",
        "first_tev_zeroed",
        "n_lev",
        "n_tev",
    )
    for step in range(6):
        alpha, alpha_rate, heave = _kinematics(step)
        result = bank.step(alpha, alpha_rate, heave)
        lane_results = [
            lane.step(alpha[index], alpha_rate[index], heave[index])
            for index, lane in enumerate(lanes)
        ]
        for name in compared:
            expected = torch.stack([row[name] for row in lane_results])
            actual = result[name]
            assert actual.device.type == "cuda"
            if actual.dtype is torch.float64:
                torch.testing.assert_close(actual, expected, rtol=2.0e-13, atol=2.0e-13)
            else:
                assert torch.equal(actual, expected)
    assert bank.it == 6
    assert bank.nt == 6
    assert bank.wake_convection_count == 6


def test_source_bank_pickle_branch_is_bitwise_and_isolated() -> None:
    parent = _bank()
    for step in range(3):
        parent.step(*_kinematics(step))
    branch = pickle.loads(pickle.dumps(parent, protocol=5))
    assert type(branch) is CudaLDVMSourceBank
    assert branch.device.type == "cuda"
    for name, value in parent.__dict__.items():
        if type(value) is torch.Tensor:
            other = getattr(branch, name)
            assert other.device.type == "cuda"
            assert other.data_ptr() != value.data_ptr()
            assert torch.equal(value, other)

    expected = parent.step(*_kinematics(3))
    actual = branch.step(*_kinematics(3))
    for name in (
        "A0",
        "gamma_lev_new",
        "gamma_tev_new_solved",
        "lev_birth_position",
        "lev_edge_position",
    ):
        assert torch.equal(actual[name], expected[name])
    frozen = parent.lg.clone()
    branch.lg.add_(1.0)
    assert torch.equal(parent.lg, frozen)
    assert not torch.equal(parent.lg, branch.lg)


@pytest.mark.parametrize(
    ("critical", "expected"),
    (
        ((10.0, 1.0e-6, 1.0e-6), (False, False, False)),
        ((1.0e-6, 10.0, 10.0), (True, True, True)),
    ),
)
def test_node_material_events_are_projected_from_adjacent_cell_condition(
    critical: tuple[float, float, float],
    expected: tuple[bool, bool, bool],
) -> None:
    device = torch.device("cuda:0")
    bank = CudaLDVMSourceBank(
        batch_size=3,
        ndiv=20,
        naterm=8,
        delta_time_convective=0.01,
        lesp_crit=torch.tensor(critical, device=device, dtype=torch.float64),
        max_wake=8,
        source_parity=True,
        device="cuda:0",
    )
    result = bank.step(
        torch.full((3,), 0.35, device=device, dtype=torch.float64),
        torch.zeros(3, device=device, dtype=torch.float64),
        node_topology_from_cell_count=1,
    )

    assert tuple(result["shed_lev"].tolist()) == expected
    assert bool(result["shed_lev"][1] == result["shed_lev"][0])
    assert bool(result["shed_lev"][2] == result["shed_lev"][0])
