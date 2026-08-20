"""Regression gates for the user-required CUDA-only numerical path.

CPU remains permitted for Python orchestration and serialization, but every
aerodynamic numerical kernel exercised here must remain on a CUDA device.
"""
from __future__ import annotations

import os
import inspect
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _python(code: str, **env_updates: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "platform"), str(HERE))
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pfield_rejects_cpu_and_invalid_device() -> None:
    for value in ("cpu", "bogus"):
        result = _python("import pfield", PFIELD_DEVICE=value)
        assert result.returncode != 0, (value, result.stdout, result.stderr)


def test_warp_config_rejects_cpu_in_gpu_only_mode() -> None:
    result = _python(
        "from fluxvortex.warp_fsi import config; print(config.summary())",
        FLUXV_GPU_ONLY="1",
        FLUXV_DEVICE="cpu",
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_single_system_dense_solve_never_calls_numpy_lapack() -> None:
    code = r"""
import numpy as np
import warp as wp
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve

assert str(config.DEVICE).startswith("cuda"), config.DEVICE
A = wp.array(np.array([[[3.0, 1.0], [1.0, 2.0]]]),
             dtype=config.DTYPE, device=config.DEVICE)
b = wp.array(np.array([[9.0, 8.0]]), dtype=config.DTYPE, device=config.DEVICE)
np.linalg.solve = lambda *_a, **_k: (_ for _ in ()).throw(
    AssertionError("CPU LAPACK fallback used"))
x = batched_dense_solve(A, b, device=config.DEVICE).numpy()[0]
np.testing.assert_allclose(x, np.array([2.0, 3.0]), rtol=1e-12, atol=1e-12)
"""
    result = _python(
        code,
        FLUXV_GPU_ONLY="1",
        FLUXV_DEVICE="cuda:0",
        FLUXV_DTYPE="float64",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_particle_velocity_and_jacobian_are_cuda_and_match_oracle() -> None:
    code = r"""
import numpy as np
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
from pfield import ParticleField, _DEVICE

assert str(_DEVICE).startswith("cuda"), _DEVICE
p = ParticleField(capacity=4)
pos = np.array([[0.0, 0.0, 0.0], [0.7, -0.2, 0.1], [-0.4, 0.3, 0.8]])
gamma = np.array([[0.2, -0.5, 0.7], [0.6, 0.1, -0.3], [-0.4, 0.9, 0.2]])
sigma = np.array([0.2, 0.35, 0.4])
p.add_particles(pos, gamma, sigma)
oracle = direct_gaussian_erf_velocity_jacobian(pos, gamma, sigma)

# The production method must no longer import/call the CPU reference.
import fluxvortex.rvpm_reference as reference
reference.direct_gaussian_erf_velocity_jacobian = lambda *_a, **_k: (_ for _ in ()).throw(
    AssertionError("CPU Jacobian fallback used"))
velocity, jacobian = p.velocity_and_jacobian_self()
np.testing.assert_allclose(velocity, oracle.velocity, rtol=2e-13, atol=2e-13)
np.testing.assert_allclose(jacobian, oracle.jacobian, rtol=5e-13, atol=5e-13)
"""
    result = _python(code, PFIELD_DEVICE="cuda:0")
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_cuda_ldvm_short_trajectory_matches_frozen_oracle() -> None:
    code = r"""
import numpy as np
import torch
from ldvm_fourier import LDVM2D
from ldvm_torch_gpu import LDVM2DCuda

cpu = LDVM2D(U=1.0, c=1.0, ndiv=20, naterm=8, dt=0.01, rho=1.0,
             lesp_crit=0.11, pivot_xc=0.25, core_rc=0.02, max_wake=32)
gpu = LDVM2DCuda(U=1.0, c=1.0, ndiv=20, naterm=8, dt=0.01, rho=1.0,
                 lesp_crit=0.11, pivot_xc=0.25, core_rc=0.02,
                 max_wake=32, device="cuda:0")
cpu_hist, gpu_hist = [], []
for k in range(24):
    phase = 2.0 * np.pi * k / 24.0
    alpha = np.deg2rad(8.0 + 18.0 * np.sin(phase))
    dalpha = np.deg2rad(18.0) * 2.0 * np.pi / 0.24 * np.cos(phase)
    hdot = -0.16 * np.cos(phase)
    rc = cpu.step(alpha, dalpha, hdot)
    rg = gpu.step(alpha, dalpha, hdot)
    assert rg["CLf"].is_cuda and rg["CDf"].is_cuda
    cpu_hist.append([rc["CLf"], rc["CDf"], rc["A0"]])
    gpu_hist.append(torch.stack((rg["CLf"], rg["CDf"], rg["A0"])))
torch.cuda.synchronize()
gpu_values = torch.stack(gpu_hist).cpu().numpy()
np.testing.assert_allclose(gpu_values, np.asarray(cpu_hist), rtol=3e-12, atol=3e-12)
"""
    result = _python(code, CUDA_VISIBLE_DEVICES="0")
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_cuda_ldvm_step_has_no_host_scalar_branch() -> None:
    from ldvm_torch_gpu import LDVM2DCuda

    source = inspect.getsource(LDVM2DCuda.step)
    assert ".item(" not in source
    assert "bool(" not in source
    assert "shed_lev = torch.abs(a0) > self.lesp_crit" in source
