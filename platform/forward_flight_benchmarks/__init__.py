"""Published forward-flight benchmark definitions.

The package namespace is deliberately lazy.  Importing a data-only paper
adapter must not initialize an unrelated solver backend as a side effect.
Public names remain source compatible through :func:`__getattr__`.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "BAIK_2012_CASES",
    "Baik2012Case",
    "apply_declared_v4b_transfer",
    "baik_kinematics",
    "build_baik_movement",
    "run_baik_old_fluxv",
    "sharp_fourier_lowpass",
    "IZRAELEVITZ_2017",
    "IZRAELEVITZ_2017_FIG11",
    "IZRAELEVITZ_2017_FIG14_SCHERER",
    "YANG_2023",
    "YANG_2025",
    "IzraelevitzCase",
    "IzraelevitzHeavePitchCase",
    "IzraelevitzSchererCase",
    "YangCase",
    "Yang2025RigidCase",
    "DEFAULT_CAUSAL_OWNER",
    "CausalIncidenceOwnerParameters",
    "causal_incidence_persistence",
    "LDVMSectionSettings",
    "LESPThreshold",
    "project_ldvm_delta_to_finite_wing",
    "run_ldvm_separation_pair",
    "IMPLEMENTATION_CONVENTIONS",
    "MODEL_SCOPE_LIMITATIONS",
    "YANG_2025_PARAMETERS",
    "Yang2025Parameters",
    "YangPLEVSolver",
]


_PUBLIC_MODULE = {
    **{
        name: "baik2012"
        for name in (
            "BAIK_2012_CASES",
            "Baik2012Case",
            "apply_declared_v4b_transfer",
            "baik_kinematics",
            "build_baik_movement",
            "run_baik_old_fluxv",
            "sharp_fourier_lowpass",
        )
    },
    **{
        name: "cases"
        for name in (
            "IZRAELEVITZ_2017",
            "IZRAELEVITZ_2017_FIG11",
            "IZRAELEVITZ_2017_FIG14_SCHERER",
            "YANG_2023",
            "YANG_2025",
            "IzraelevitzCase",
            "IzraelevitzHeavePitchCase",
            "IzraelevitzSchererCase",
            "YangCase",
            "Yang2025RigidCase",
        )
    },
    **{
        name: "causal_incidence_owner"
        for name in (
            "DEFAULT_CAUSAL_OWNER",
            "CausalIncidenceOwnerParameters",
            "causal_incidence_persistence",
        )
    },
    **{
        name: "ldvm_uvlm_correction"
        for name in (
            "LDVMSectionSettings",
            "LESPThreshold",
            "project_ldvm_delta_to_finite_wing",
            "run_ldvm_separation_pair",
        )
    },
    **{
        name: "yang_plev"
        for name in (
            "IMPLEMENTATION_CONVENTIONS",
            "MODEL_SCOPE_LIMITATIONS",
            "YANG_2025_PARAMETERS",
            "Yang2025Parameters",
            "YangPLEVSolver",
        )
    },
}


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
