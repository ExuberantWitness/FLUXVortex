"""Single switchable precision/device entry point for the Warp FSI port.

EVERYTHING (array allocation, kernel signatures, solver tolerances) references the
aliases defined here, so flipping fp64<->fp32 or 4090<->H100 is a one-line change
(or one env var) — never hard-coded per file.

Switching
---------
- Default is float64 (required for the MATLAB 1% accuracy target).
- Override before importing any warp_fsi kernel module:
    * env var:   FLUXV_DTYPE=float32  FLUXV_DEVICE=cuda:0
    * or in code: import fluxvortex.warp_fsi.config as cfg; cfg.set_dtype("float32")
      (must be called before kernel modules are imported, since Warp captures
       types at kernel-definition time).

Because Warp resolves kernel argument types at decoration (import) time, the
robust pattern is: set the dtype, THEN import kernels. `validate.py` exercises
both fp64 (accuracy baseline) and fp32 (throughput) by importing in subprocesses.
"""
from __future__ import annotations
import os
import numpy as np
import warp as wp

# ── Active selection (mutable until kernel modules import) ──────────────────
_DTYPE_NAME = os.environ.get("FLUXV_DTYPE", "float64").lower()
_DEVICE = os.environ.get("FLUXV_DEVICE", "")  # "" -> auto (cuda if available)

_TABLE = {
    "float64": dict(DTYPE=wp.float64, VEC3=wp.vec3d, MAT33=wp.mat33d,
                    NP_DTYPE=np.float64, CR_TOL=1e-10, NEWTON_TOL=1e-8,
                    GEOM_ATOL=1e-12, PORT_ATOL=1e-11),
    "float32": dict(DTYPE=wp.float32, VEC3=wp.vec3f, MAT33=wp.mat33f,
                    NP_DTYPE=np.float32, CR_TOL=1e-5, NEWTON_TOL=1e-5,
                    GEOM_ATOL=1e-4, PORT_ATOL=1e-4),
}

# Module-level aliases — referenced everywhere. Re-pointed by set_dtype().
DTYPE = None
VEC3 = None
MAT33 = None
NP_DTYPE = None
CR_TOL = None
NEWTON_TOL = None
GEOM_ATOL = None
PORT_ATOL = None   # GPU-vs-CPU(fp64) porting tolerance (dtype-aware)
DEVICE = None


def _resolve_device(spec: str) -> str:
    if spec:
        return spec
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def set_dtype(name: str) -> None:
    """Repoint all aliases to the given precision ('float64' | 'float32').

    Call BEFORE importing warp_fsi kernel modules. After kernels are compiled,
    their argument types are fixed; switch by re-importing in a fresh process.
    """
    global _DTYPE_NAME, DTYPE, VEC3, MAT33, NP_DTYPE, CR_TOL, NEWTON_TOL
    global GEOM_ATOL, PORT_ATOL
    name = name.lower()
    if name not in _TABLE:
        raise ValueError(f"dtype must be one of {list(_TABLE)}, got {name!r}")
    _DTYPE_NAME = name
    cfg = _TABLE[name]
    DTYPE = cfg["DTYPE"]; VEC3 = cfg["VEC3"]; MAT33 = cfg["MAT33"]
    NP_DTYPE = cfg["NP_DTYPE"]; CR_TOL = cfg["CR_TOL"]
    NEWTON_TOL = cfg["NEWTON_TOL"]; GEOM_ATOL = cfg["GEOM_ATOL"]
    PORT_ATOL = cfg["PORT_ATOL"]


def set_device(spec: str) -> None:
    global _DEVICE, DEVICE
    _DEVICE = spec
    DEVICE = _resolve_device(spec)


def dtype_name() -> str:
    return _DTYPE_NAME


def summary() -> str:
    return (f"warp_fsi config: dtype={_DTYPE_NAME} device={DEVICE} "
            f"CR_TOL={CR_TOL} NEWTON_TOL={NEWTON_TOL}")


# Initialize on import from env (default float64 + auto device).
wp.init()
set_dtype(_DTYPE_NAME)
set_device(_DEVICE)
