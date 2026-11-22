"""Physically weighted metrics for comparing two generalized aero-load vectors.

Why this module exists
----------------------
The Q16 shell DOFs mix units: position rows carry force (N), first-derivative rows
carry moment (N·m) and mixed-torsion rows carry N·m².  Comparing two generalized load
vectors with a plain Euclidean ``R^dof`` norm therefore mixes those units, and the
number it returns is dominated by whichever block happens to have the largest raw
magnitude.  Measured consequence: on the Roj membrane state the two load tracks
differed by ``||A − B||/||B|| = 0.338`` while their MAGNITUDES differed by only 8% --
i.e. most of that number was direction/units, not load error.  An external audit
flagged this and it was right.

Two metrics that are well defined for this purpose:

``flexibility_energy_norm``
    ``sqrt(r^T K_ref^-1 r)`` -- the strain energy a load difference ``r`` would put
    into the structure, divided appropriately.  It has units of energy^1/2, it is
    invariant to the DOF scaling that plagues the raw L2, and it answers the question
    that matters for FSI: *does this disagreement move the structure?*  Solved via a
    linear solve, never by forming an inverse.

``wrench_components``
    the 6-DOF rigid resultant (force, moment about an origin) -- already produced by
    ``q16_rigid_body_wrench.aero_generalized_wrench``; re-exported here so a caller
    does not have to know both modules.

Both are pure functions over torch tensors, so they are unit-testable on CPU.  Neither
one is a claim about which load track is correct; they only make the comparison
meaningful.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .q16_rigid_body_wrench import aero_generalized_wrench


def wrench_components(
    generalized_load: torch.Tensor,
    structural_state: torch.Tensor,
    reference_rows: torch.Tensor,
    *,
    origin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """6-DOF rigid resultant (force, moment) of a generalized aerodynamic load.

    Re-exported here because a load-track comparison needs BOTH a
    stiffness-weighted scalar (how much the disagreement would move the structure) and
    the rigid resultant (whether the two tracks even push the wing the same way).  The
    module docstring promised this name before it existed; A3 makes it real instead of
    leaving a documented-but-missing API.
    """

    return aero_generalized_wrench(
        generalized_load, structural_state, reference_rows=reference_rows, origin=origin
    )

__all__ = [
    "flexibility_energy_norm",
    "relative_flexibility_gap",
    "wrench_components",
]


def flexibility_energy_norm(
    residual: torch.Tensor,
    stiffness_reference: torch.Tensor,
    *,
    regularisation: float = 0.0,
) -> float:
    """``sqrt(r^T K^-1 r)`` for a symmetric positive-definite reference stiffness.

    ``residual`` is a ``(dof,)`` or ``(1, dof)`` generalized-load difference;
    ``stiffness_reference`` is ``(dof, dof)``.  The quadratic form is evaluated as
    ``r . solve(K, r)`` so no inverse is formed, and a negative or non-finite result
    (which a non-SPD K can produce) is rejected rather than silently rooted.

    ``regularisation`` optionally adds ``regularisation * max(diag(K)) * I`` so a
    singular or indefinite reference (e.g. a free-free assembly before constraints)
    still yields a finite, monotone metric.  It defaults to 0, which requires K to be
    genuinely SPD -- the tests pin both branches.
    """

    if residual.ndim == 2:
        if residual.shape[0] != 1:
            raise ValueError("residual must be (dof,) or (1, dof)")
        residual = residual[0]
    if residual.ndim != 1:
        raise ValueError("residual must be (dof,) or (1, dof)")
    if stiffness_reference.ndim != 2 or stiffness_reference.shape[0] != stiffness_reference.shape[1]:
        raise ValueError("stiffness_reference must be square")
    if stiffness_reference.shape[0] != residual.shape[0]:
        raise ValueError("residual and stiffness_reference sizes differ")
    if not math.isfinite(regularisation) or regularisation < 0.0:
        raise ValueError("regularisation must be finite and non-negative")

    matrix = stiffness_reference
    if regularisation > 0.0:
        diagonal_scale = torch.diagonal(matrix).abs().max()
        if not bool(torch.isfinite(diagonal_scale).item()):
            raise FloatingPointError("stiffness reference has a non-finite diagonal")
        matrix = matrix + (regularisation * diagonal_scale) * torch.eye(
            matrix.shape[0], device=matrix.device, dtype=matrix.dtype
        )
    # Symmetric quadratic form: use the symmetric part so a slightly non-symmetric
    # reference (a tangent assembled column-wise) still gives a real value.
    symmetric = 0.5 * (matrix + matrix.transpose(0, 1))
    solution = torch.linalg.solve(symmetric, residual.unsqueeze(-1)).squeeze(-1)
    quadratic = torch.dot(residual, solution)
    if not bool(torch.isfinite(quadratic).item()):
        raise FloatingPointError("flexibility form produced a non-finite value")
    if bool((quadratic < 0.0).item()):
        raise ValueError(
            "flexibility form is negative: stiffness_reference is not positive definite"
        )
    return float(torch.sqrt(quadratic).item())


def relative_flexibility_gap(
    first: torch.Tensor,
    second: torch.Tensor,
    stiffness_reference: torch.Tensor,
    *,
    regularisation: float = 0.0,
) -> float:
    """``||first − second||_K-1 / ||second||_K-1`` -- a units-consistent relative gap.

    This is the number that should replace an unweighted ``R^dof`` L2 ratio when
    comparing two load definitions.
    """

    residual = first - second
    numerator = flexibility_energy_norm(
        residual, stiffness_reference, regularisation=regularisation
    )
    denominator = flexibility_energy_norm(
        second, stiffness_reference, regularisation=regularisation
    )
    if denominator == 0.0:
        raise ValueError("reference load has zero flexibility norm")
    return numerator / denominator
