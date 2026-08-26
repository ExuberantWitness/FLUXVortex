"""Q16 production aerodynamics must include LEV, joint TEV and free wake."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fluxvortex.warp_fsi.q16_mandatory_aero_mode import (
    Q16MandatoryAeroModeGuard,
    require_q16_mandatory_aero_mode,
)


class _Device:
    type = "cuda"


class _ParticleOwner:
    def __init__(self) -> None:
        self.device = _Device()


def _solver(
    *,
    enable_lev: object = True,
    joint_tev: object = True,
    prescribed_wake: object = False,
    separated_source: object = "hirato_ring",
) -> SimpleNamespace:
    return SimpleNamespace(
        jcfg=SimpleNamespace(
            enable_lev=enable_lev,
            joint_tev=joint_tev,
            lev_start_step=0,
            separated_source=separated_source,
        ),
        _prescribed_wake=prescribed_wake,
        lev_pf=_ParticleOwner(),
        cuda_device=_Device(),
        cuda_numerical_contract="torch-cuda-float64-no-cpu-fallback-v1",
    )


def test_q16_mandatory_mode_accepts_only_full_lev_joint_tev_free_wake() -> None:
    solver = _solver()
    guard = require_q16_mandatory_aero_mode(solver)
    assert type(guard) is Q16MandatoryAeroModeGuard
    assert guard.verify(solver) is guard
    assert guard.separated_lev is True
    assert guard.joint_tev is True
    assert guard.free_wake is True
    assert guard.separated_source == "hirato_ring"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"enable_lev": False}, "separated LEV"),
        ({"joint_tev": False}, "joint TEV"),
        ({"prescribed_wake": True}, "free wake"),
        ({"enable_lev": np.bool_(True)}, "exact bool"),
        ({"joint_tev": np.bool_(True)}, "exact bool"),
        ({"separated_source": "disabled"}, "unsupported"),
        ({"separated_source": 1}, "exact str"),
    ],
)
def test_q16_mandatory_mode_rejects_every_reduced_aero_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, RuntimeError), match=match):
        require_q16_mandatory_aero_mode(_solver(**kwargs))


def test_q16_mandatory_guard_rejects_runtime_mode_and_owner_drift() -> None:
    solver = _solver()
    guard = require_q16_mandatory_aero_mode(solver)
    solver.jcfg.separated_source = "disabled"
    with pytest.raises(RuntimeError, match="unsupported"):
        guard.verify(solver)

    solver = _solver()
    guard = require_q16_mandatory_aero_mode(solver)

    solver.jcfg.enable_lev = False
    with pytest.raises(RuntimeError, match="separated LEV"):
        guard.verify(solver)
    solver.jcfg.enable_lev = True
    solver.jcfg = SimpleNamespace(
        enable_lev=True,
        joint_tev=True,
        lev_start_step=0,
    )
    with pytest.raises(RuntimeError, match="configuration owner"):
        guard.verify(solver)

    solver = _solver()
    guard = require_q16_mandatory_aero_mode(solver)
    solver.lev_pf = _ParticleOwner()
    with pytest.raises(RuntimeError, match="particle owner"):
        guard.verify(solver)


def test_q16_mandatory_mode_rejects_non_cuda_and_unarmed_solver() -> None:
    solver = _solver()
    del solver._prescribed_wake
    with pytest.raises(RuntimeError, match="free-wake state"):
        require_q16_mandatory_aero_mode(solver)

    solver = _solver()
    solver.cuda_device = SimpleNamespace(type="cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        require_q16_mandatory_aero_mode(solver)
