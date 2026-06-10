"""warp_fsi — batched, GPU (NVIDIA Warp) port of the FLUXVortex FSI solver.

Multi-environment ANCF shell + UVLM strong-coupling on Warp, validated against
the CPU `standalone_*` reference (MATLAB-aligned). See the plan at
~/.claude/plans/plan-matlab-log-python-expressive-toast.md.

Precision/device is switchable via `config` (default float64). Import `config`
and call `set_dtype(...)` BEFORE importing kernel modules to flip fp64<->fp32.
"""
from . import config  # noqa: F401  (initializes warp + precision on import)
