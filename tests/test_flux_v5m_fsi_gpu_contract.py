"""Strict GPU contract tests for the FLUX-V5M Warp-FSI facade."""
from __future__ import annotations

import inspect
import sys
import types

import numpy as np
import pytest
import warp as wp

from fluxvortex.warp_fsi import batched_solver, config
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve, structural_cg
from fluxvortex.warp_fsi.flux_v5m_fsi_gpu import (
    assert_v5m_fsi_runtime,
    run_v5m_gpu_coupled_trajectory,
)

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")


def test_v5m_fsi_defaults_to_mandatory_cuda() -> None:
    assert config.GPU_ONLY is True
    assert wp.get_device(config.DEVICE).is_cuda
    assert assert_v5m_fsi_runtime().startswith("cuda")


def test_v5m_fsi_rejects_loaded_cpu_reference_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "fluxvortex.warp_fsi.ml_fluid"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(RuntimeError, match="CPU MATLAB-parity reference"):
        assert_v5m_fsi_runtime()


def test_v5m_fsi_rejects_float32_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "_DTYPE_NAME", "float32")
    monkeypatch.setattr(config, "DTYPE", wp.float32)
    with pytest.raises(RuntimeError, match="requires the frozen float64 precision"):
        assert_v5m_fsi_runtime()


def test_v5m_fsi_rejects_host_initial_state_before_solver_call() -> None:
    q = wp.zeros((1, 2), dtype=config.DTYPE, device="cpu")
    fake_fluid = types.SimpleNamespace(device=config.DEVICE)
    with pytest.raises(ValueError, match="q0 must reside"):
        run_v5m_gpu_coupled_trajectory(
            object(),
            fake_fluid,
            q,
            q,
            np.ones(2),
            lambda _time: 0.0,
            0.01,
            1,
        )


def test_dense_solve_stays_on_cuda_when_numpy_lapack_is_hostile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = wp.array(
        np.array([[[3.0, 1.0], [1.0, 2.0]]]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    rhs = wp.array(np.array([[9.0, 8.0]]), dtype=config.DTYPE, device=config.DEVICE)
    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("host LAPACK called")
        ),
    )
    solution = batched_dense_solve(matrix, rhs, device=config.DEVICE)
    np.testing.assert_allclose(solution.numpy(), [[2.0, 3.0]], atol=1.0e-12)


def test_structural_cg_reduction_is_a_gpu_kernel() -> None:
    source = inspect.getsource(structural_cg)
    assert "np.max" not in source
    assert "rr.numpy" not in source
    assert "_gpu_max_scalar" in source


def test_structural_cg_batches_reductions_and_fuses_vector_updates() -> None:
    signature = inspect.signature(structural_cg)
    assert signature.parameters["check_every"].default == 8
    source = inspect.getsource(batched_solver)
    assert "_dot_parallel_kernel" in source
    assert "_pcg_update_solution_residual" in source
    assert "_pcg_precondition_dot" in source
    assert "_pcg_update_direction" in source


def test_structural_cg_fails_closed_when_iteration_budget_is_exhausted() -> None:
    device = config.DEVICE
    b_wp = wp.array(
        np.array([[1.0, 2.0]], dtype=np.float64),
        dtype=config.DTYPE,
        device=device,
    )
    mass = wp.array(
        np.array([[[4.0, 1.0], [1.0, 3.0]]], dtype=np.float64),
        dtype=config.DTYPE,
        device=device,
    )
    stiffness = wp.zeros((1, 1, 2, 2), dtype=config.DTYPE, device=device)
    edofs = wp.array(np.array([[0, 1]], dtype=np.int32), device=device)
    free = wp.ones(2, dtype=config.DTYPE, device=device)
    with pytest.raises(RuntimeError, match="PCG did not converge after 1 iterations"):
        structural_cg(
            b_wp,
            mass,
            stiffness,
            edofs,
            free,
            0.0,
            2,
            max_iter=1,
            tol=1.0e-30,
            device=device,
        )


def test_dense_solve_rejects_empty_and_nonfinite_batches() -> None:
    empty_matrix = wp.zeros((0, 2, 2), dtype=config.DTYPE, device=config.DEVICE)
    empty_rhs = wp.zeros((0, 2), dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(ValueError, match="non-empty"):
        batched_dense_solve(empty_matrix, empty_rhs, device=config.DEVICE)

    bad_matrix = wp.array(
        np.array([[[1.0, 0.0], [0.0, np.nan]]]),
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    rhs = wp.array(np.ones((1, 2)), dtype=config.DTYPE, device=config.DEVICE)
    with pytest.raises(FloatingPointError, match="dense matrix"):
        batched_dense_solve(bad_matrix, rhs, device=config.DEVICE)
