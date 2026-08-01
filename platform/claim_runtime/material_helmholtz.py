"""Per-element Kelvin--Helmholtz identity for a material P2 doublet sheet.

For an oriented material triangle with coordinates ``(xi, eta)``, let
``chi`` be the N1 pressure-potential jump and ``mu=-chi`` the aligned DDE
scalar convention.  If ``D chi / Dt = 0``, then

``J*gamma = chi_xi*g_eta - chi_eta*g_xi``

where ``g_xi`` and ``g_eta`` are the current covariant edge vectors,
``J=|g_xi x g_eta|``, and ``gamma=n x grad_s(chi)=grad_s(mu) x n``.
This is the surface Cauchy transport identity: material potential jump
conservation automatically stretches the physical sheet-vorticity density.

The identity is local to one nondegenerate material element.  It does not
establish curved inter-element compatibility, remeshing, reconnection,
viscous diffusion, or production wake accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletElement,
)


@dataclass(frozen=True)
class MaterialHelmholtzReport:
    points: int
    max_material_mu_residual: float
    max_material_derivative_residual: float
    max_cauchy_vector_density_residual: float
    passed: bool


def material_p2_helmholtz_report(
    reference: QuadraticDoubletElement,
    current: QuadraticDoubletElement,
    barycentric,
    *,
    tolerance: float = 1.0e-12,
) -> MaterialHelmholtzReport:
    """Check material-scalar and Cauchy stretching identities after deformation."""
    if not isinstance(reference, QuadraticDoubletElement) or not isinstance(
        current,
        QuadraticDoubletElement,
    ):
        raise DistributedDoubletError(
            "reference and current must be QuadraticDoubletElement"
        )
    bary = np.asarray(barycentric, dtype=float)
    if (
        bary.ndim != 2
        or bary.shape[1] != 3
        or not np.all(np.isfinite(bary))
        or np.any(bary < 0.0)
        or np.max(np.abs(bary.sum(axis=1) - 1.0), initial=0.0)
        > 1.0e-12
    ):
        raise DistributedDoubletError(
            "barycentric must be finite convex coordinates with shape (n,3)"
        )
    if len(bary) == 0:
        raise DistributedDoubletError(
            "at least one material evaluation point is required"
        )
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise DistributedDoubletError(
            "tolerance must be finite and non-negative"
        )

    mu_reference = reference.evaluate_barycentric(bary)
    mu_current = current.evaluate_barycentric(bary)
    material_residual = float(
        np.max(np.abs(mu_current - mu_reference), initial=0.0)
    )

    reference_gradient = reference.surface_gradient_barycentric(bary)
    current_gradient = current.surface_gradient_barycentric(bary)
    reference_xi = reference.vertices[1] - reference.vertices[0]
    reference_eta = reference.vertices[2] - reference.vertices[0]
    current_xi = current.vertices[1] - current.vertices[0]
    current_eta = current.vertices[2] - current.vertices[0]

    # chi=-mu for an aligned N1/DDE normal convention.
    chi_xi_reference = -reference_gradient @ reference_xi
    chi_eta_reference = -reference_gradient @ reference_eta
    chi_xi_current = -current_gradient @ current_xi
    chi_eta_current = -current_gradient @ current_eta
    derivative_residual = float(
        max(
            np.max(
                np.abs(chi_xi_current - chi_xi_reference),
                initial=0.0,
            ),
            np.max(
                np.abs(chi_eta_current - chi_eta_reference),
                initial=0.0,
            ),
        )
    )

    expected_density = (
        chi_xi_reference[:, None] * current_eta[None, :]
        - chi_eta_reference[:, None] * current_xi[None, :]
    )
    current_density = (
        np.linalg.norm(np.cross(current_xi, current_eta))
        * current.sheet_vorticity_barycentric(bary)
    )
    cauchy_residual = float(
        np.max(
            np.linalg.norm(current_density - expected_density, axis=1),
            initial=0.0,
        )
    )
    return MaterialHelmholtzReport(
        points=len(bary),
        max_material_mu_residual=material_residual,
        max_material_derivative_residual=derivative_residual,
        max_cauchy_vector_density_residual=cauchy_residual,
        passed=bool(
            max(
                material_residual,
                derivative_residual,
                cauchy_residual,
            )
            <= tolerance
        ),
    )

