"""Contract tests for the only FLUX-V5M production entry point."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from bing_joint_ptera import JointConfig
from flux_v5m_gpu import (
    FAIL_CLOSED_STATUS,
    OUT_OF_SCOPE_STATUS,
    PRODUCTION_STATUS,
    capability_matrix,
    capability_status,
    make_flux_v5m_ptera_solver,
    require_cuda_device,
    require_cuda_float64_tensors,
    run_flux_v5m_ptera,
)


def test_capability_matrix_is_exact_immutable_and_honest() -> None:
    matrix = capability_matrix()
    assert type(matrix) is tuple
    assert len(matrix) == 7
    status = capability_status()
    assert (
        status["ptera_attached_single_wing_prescribed_wake"]
        == FAIL_CLOSED_STATUS
    )
    assert status["ldvm_and_finite_wing_corrections"] == PRODUCTION_STATUS
    assert status["ptera_active_lev"] == PRODUCTION_STATUS
    assert status["ptera_joint_lev_tev"] == PRODUCTION_STATUS
    assert status["ptera_free_wake"] == PRODUCTION_STATUS
    assert status["warp_fsi_end_to_end"] == PRODUCTION_STATUS
    assert status["ptera_multi_airplane_multi_wing_or_image"] == OUT_OF_SCOPE_STATUS
    with pytest.raises(TypeError):
        status["ptera_active_lev"] = PRODUCTION_STATUS  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        matrix[0].status = FAIL_CLOSED_STATUS  # type: ignore[misc]


def test_device_contract_rejects_cpu_without_fallback() -> None:
    with pytest.raises(ValueError, match="must be CUDA"):
        require_cuda_device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tensor_contract_rejects_cpu_mixed_dtype_and_nonfinite() -> None:
    good = torch.ones(4, device="cuda:0", dtype=torch.float64)
    assert require_cuda_float64_tensors({"good": good}) == good.device
    with pytest.raises(ValueError, match="must be CUDA"):
        require_cuda_float64_tensors(
            (("good", good), ("host", torch.ones(4, dtype=torch.float64)))
        )
    with pytest.raises(TypeError, match="float64"):
        require_cuda_float64_tensors(
            {"wrong_dtype": torch.ones(4, device="cuda:0", dtype=torch.float32)}
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_cuda_float64_tensors(
            {"bad": torch.tensor([float("nan")], device="cuda:0", dtype=torch.float64)}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_factory_does_not_expose_a_cpu_solver_fallback() -> None:
    source = make_flux_v5m_ptera_solver.__code__.co_names
    assert "JointLEVTEVSolver" not in source
    assert "CudaAttachedJointLEVTEVSolver" in source


def test_production_entry_rejects_reduced_aerodynamic_modes_before_run() -> None:
    with pytest.raises(ValueError, match="separated LEV"):
        make_flux_v5m_ptera_solver(
            object(), JointConfig(enable_lev=False, joint_tev=True)
        )
    with pytest.raises(ValueError, match="joint TEV"):
        make_flux_v5m_ptera_solver(
            object(), JointConfig(enable_lev=True, joint_tev=False)
        )
    with pytest.raises(ValueError, match="free wake"):
        run_flux_v5m_ptera(
            object(),
            JointConfig(enable_lev=True, joint_tev=True),
            prescribed_wake=True,
        )
