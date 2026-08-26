"""Fail-closed aerodynamic mode ownership for Q16 FSI.

This module deliberately has no Ptera or Torch import.  It validates the live
production solver by its already-constructed owners, so importing Q16
structural code cannot silently select an attached-flow or prescribed-wake
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Q16_MANDATORY_AERO_MODE_SCHEMA = "flux-v5m-q16-lev-joint-tev-free-wake-v1"
_CUDA_CONTRACT = "torch-cuda-float64-no-cpu-fallback-v1"
_MISSING = object()


def _exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _validate_live_mode(solver: Any) -> tuple[Any, Any, int, str]:
    config = getattr(solver, "jcfg", _MISSING)
    if config is _MISSING:
        raise RuntimeError("Q16 aerodynamic solver has no configuration owner")
    enable_lev = _exact_bool(
        "separated LEV enable flag", getattr(config, "enable_lev", _MISSING)
    )
    joint_tev = _exact_bool(
        "joint TEV enable flag", getattr(config, "joint_tev", _MISSING)
    )
    prescribed_wake = getattr(solver, "_prescribed_wake", _MISSING)
    if prescribed_wake is _MISSING:
        raise RuntimeError("Q16 aerodynamic solver has no armed free-wake state")
    prescribed_wake = _exact_bool("prescribed wake flag", prescribed_wake)
    if not enable_lev:
        raise RuntimeError("Q16 FSI requires separated LEV")
    if not joint_tev:
        raise RuntimeError("Q16 FSI requires joint TEV")
    if prescribed_wake:
        raise RuntimeError("Q16 FSI requires a free wake")

    lev_start_step = getattr(config, "lev_start_step", _MISSING)
    if type(lev_start_step) is not int or lev_start_step < 0:
        raise RuntimeError("Q16 separated-LEV start step must be a non-negative int")
    if getattr(solver, "cuda_numerical_contract", None) != _CUDA_CONTRACT:
        raise RuntimeError("Q16 aerodynamic solver lacks the CUDA float64 contract")
    cuda_device = getattr(solver, "cuda_device", _MISSING)
    if getattr(cuda_device, "type", None) != "cuda":
        raise RuntimeError("Q16 aerodynamic solver must use CUDA")
    particle_owner = getattr(solver, "lev_pf", _MISSING)
    if particle_owner is _MISSING:
        raise RuntimeError("Q16 aerodynamic solver has no separated-LEV particle owner")
    if getattr(getattr(particle_owner, "device", None), "type", None) != "cuda":
        raise RuntimeError("Q16 separated-LEV particle owner must use CUDA")
    separated_source = getattr(config, "separated_source", "hirato_ring")
    if type(separated_source) is not str:
        raise TypeError("Q16 separated-LEV source model must be an exact str")
    if separated_source not in {"hirato_ring", "dvm_node_ribbon"}:
        raise RuntimeError("Q16 separated-LEV source model is unsupported")
    if separated_source == "dvm_node_ribbon":
        if lev_start_step != 0:
            raise RuntimeError("Q16 DVM node ribbon must start at step zero")
        steps_done = getattr(solver, "_steps_done", 0)
        source_bank = getattr(solver, "dvm_source_bank", None)
        if steps_done:
            if source_bank is None:
                raise RuntimeError("Q16 DVM node ribbon has no source-bank owner")
            if (
                getattr(getattr(source_bank, "device", None), "type", None)
                != "cuda"
                or getattr(source_bank, "source_parity", None) is not True
                or getattr(source_bank, "it", None) != steps_done
            ):
                raise RuntimeError("Q16 DVM source-bank state drifted")
            frontier = getattr(solver, "_cuda_dvm_frontier_nodes", None)
            if getattr(getattr(frontier, "device", None), "type", None) != "cuda":
                raise RuntimeError("Q16 DVM node frontier left CUDA")
    return config, particle_owner, lev_start_step, separated_source


@dataclass(frozen=True, slots=True, eq=False)
class Q16MandatoryAeroModeGuard:
    """Identity-bound guard rechecked before and after every future trial."""

    schema_id: str
    solver_owner: object
    configuration_owner: object
    particle_owner: object
    lev_start_step: int
    separated_source: str
    separated_lev: bool = True
    joint_tev: bool = True
    free_wake: bool = True

    def verify(self, solver: Any) -> Q16MandatoryAeroModeGuard:
        if solver is not self.solver_owner:
            raise RuntimeError("Q16 aerodynamic solver owner drifted")
        if getattr(solver, "jcfg", _MISSING) is not self.configuration_owner:
            raise RuntimeError("Q16 aerodynamic configuration owner drifted")
        if getattr(solver, "lev_pf", _MISSING) is not self.particle_owner:
            raise RuntimeError("Q16 separated-LEV particle owner drifted")
        config, particle_owner, lev_start_step, separated_source = (
            _validate_live_mode(solver)
        )
        if (
            config is not self.configuration_owner
            or particle_owner is not self.particle_owner
        ):
            raise RuntimeError("Q16 aerodynamic owner identity drifted")
        if lev_start_step != self.lev_start_step:
            raise RuntimeError("Q16 separated-LEV start step drifted")
        if separated_source != self.separated_source:
            raise RuntimeError("Q16 separated-LEV source model drifted")
        return self


def require_q16_mandatory_aero_mode(solver: Any) -> Q16MandatoryAeroModeGuard:
    """Freeze the mandatory separated-LEV/joint-TEV/free-wake live mode."""

    config, particle_owner, lev_start_step, separated_source = _validate_live_mode(
        solver
    )
    return Q16MandatoryAeroModeGuard(
        schema_id=Q16_MANDATORY_AERO_MODE_SCHEMA,
        solver_owner=solver,
        configuration_owner=config,
        particle_owner=particle_owner,
        lev_start_step=lev_start_step,
        separated_source=separated_source,
    )


__all__ = [
    "Q16_MANDATORY_AERO_MODE_SCHEMA",
    "Q16MandatoryAeroModeGuard",
    "require_q16_mandatory_aero_mode",
]
