"""CUDA-only gates for the source packet exported by ``LDVM2DCuda``."""
from __future__ import annotations

import pickle

import torch

from ldvm_torch_gpu import LDVM2DCuda


def _source() -> LDVM2DCuda:
    return LDVM2DCuda(
        U=1.0,
        c=1.0,
        ndiv=20,
        naterm=8,
        dt=0.01,
        rho=1.0,
        lesp_crit=0.05,
        pivot_xc=0.25,
        core_rc=0.02,
        max_wake=32,
        source_parity=True,
        device="cuda:0",
    )


def _step(source: LDVM2DCuda) -> dict[str, torch.Tensor | bool]:
    device = torch.device("cuda:0")
    return source.step(
        torch.tensor(0.5, device=device, dtype=torch.float64),
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(0.0, device=device, dtype=torch.float64),
    )


def test_source_packet_stays_cuda_and_pins_lesp() -> None:
    source = _source()
    result = _step(source)
    tensor_fields = (
        "lesp_pre",
        "A0",
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
    assert result["source_parity"] is True
    for name in tensor_fields:
        value = result[name]
        assert type(value) is torch.Tensor
        assert value.device.type == "cuda"
        assert bool(torch.isfinite(value).all().item())
    assert bool(result["shed_lev"].item())
    assert abs(float(result["A0"].item()) - 0.05) <= 2.0e-15
    assert abs(float(result["lesp_constraint_residual"].item())) <= 2.0e-15


def test_source_parity_zeroes_only_first_tev_persistence() -> None:
    source = _source()
    first = _step(source)
    assert bool(first["first_tev_zeroed"].item())
    assert float(first["gamma_tev_new_solved"].item()) != 0.0
    assert float(first["gamma_tev_new_persisted"].item()) == 0.0
    assert float(source.tg[0].item()) == 0.0

    second = _step(source)
    assert not bool(second["first_tev_zeroed"].item())
    assert float(second["gamma_tev_new_persisted"].item()) == float(
        second["gamma_tev_new_solved"].item()
    )
    assert source.wake_convection_count == 2


def _assert_cuda_state_equal(left: LDVM2DCuda, right: LDVM2DCuda) -> None:
    """Compare every serialized scientific tensor without a CPU oracle."""
    assert left is not right
    assert left.device == right.device == torch.device("cuda:0")
    assert left.cuda_stream == torch.cuda.current_stream(left.device)
    assert right.cuda_stream == torch.cuda.current_stream(right.device)
    assert left.it == right.it
    assert left.nt == right.nt
    assert left.wake_convection_count == right.wake_convection_count
    assert left.source_parity is right.source_parity
    for name, value in left.__dict__.items():
        if type(value) is torch.Tensor:
            restored = getattr(right, name)
            assert restored.device.type == "cuda"
            assert restored.data_ptr() != value.data_ptr()
            assert torch.equal(value, restored)


def test_pickle_roundtrip_preserves_and_isolates_cuda_source_branch() -> None:
    source = _source()
    _step(source)
    _step(source)

    restored = pickle.loads(pickle.dumps(source, protocol=5))
    assert type(restored) is LDVM2DCuda
    _assert_cuda_state_equal(source, restored)

    # The fork must advance bitwise-identically from the same state, while its
    # in-place wake update remains isolated from the parent branch.
    parent_it = source.it
    parent = _step(source)
    branch = _step(restored)
    for name in (
        "A0",
        "gamma_lev_new",
        "gamma_tev_new_solved",
        "gamma_tev_new_persisted",
        "lev_birth_position",
        "tev_birth_position",
        "lev_edge_position",
        "tev_edge_position",
    ):
        assert torch.equal(parent[name], branch[name])
    assert source.it == restored.it == parent_it + 1

    parent_tg = source.tg.clone()
    restored.tg.add_(1.0)
    assert torch.equal(source.tg, parent_tg)
    assert not torch.equal(source.tg, restored.tg)
