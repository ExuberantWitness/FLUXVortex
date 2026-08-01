"""Closure-free conservation skeleton for a three-dimensional surface IBL.

The exact normal-integrated boundary-layer equations store the tangential
mass-flux-defect vector ``M`` and the scalar ``trace(T)``.  The tensor ``T``,
kinetic-energy-defect flux ``E``, wall shear and dissipation are closure
quantities used by the fluxes and sources; this module keeps them named and
independent instead of inventing a closure.

No pressure, force, LESP, separation threshold or target load is accepted.
The finite-volume budget reports missing physics as a residual and never
infers a source from that residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SurfaceIBLError(ValueError):
    """Invalid surface integral-boundary-layer field or budget."""


def _array(
    name: str,
    value,
    *,
    shape_tail: tuple[int, ...],
    count: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    expected_ndim = 1 + len(shape_tail)
    if array.ndim != expected_ndim or array.shape[1:] != shape_tail:
        raise SurfaceIBLError(
            f"{name} must have shape (n,{','.join(map(str, shape_tail))}), "
            f"got {array.shape}"
        )
    if count is not None and len(array) != count:
        raise SurfaceIBLError(
            f"{name} must contain {count} cells, got {len(array)}"
        )
    if not np.all(np.isfinite(array)):
        raise SurfaceIBLError(f"{name} contains non-finite values")
    return array.copy()


def _scalars(name: str, value, count: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise SurfaceIBLError(f"{name} must have shape (n,), got {array.shape}")
    if count is not None and len(array) != count:
        raise SurfaceIBLError(
            f"{name} must contain {count} cells, got {len(array)}"
        )
    if not np.all(np.isfinite(array)):
        raise SurfaceIBLError(f"{name} contains non-finite values")
    return array.copy()


@dataclass(frozen=True)
class SurfaceIBLFields:
    """Cell-centred exact fields and still-unclosed profile moments."""

    mass_flux_defect: np.ndarray
    momentum_flux_defect: np.ndarray
    kinetic_energy_defect_flux: np.ndarray
    external_tangential_velocity: np.ndarray
    external_velocity_surface_gradient: np.ndarray
    wall_shear_over_density: np.ndarray
    dissipation_integral: np.ndarray
    surface_normal: np.ndarray

    def __post_init__(self) -> None:
        mass = _array(
            "mass_flux_defect",
            self.mass_flux_defect,
            shape_tail=(3,),
        )
        count = len(mass)
        tensor = _array(
            "momentum_flux_defect",
            self.momentum_flux_defect,
            shape_tail=(3, 3),
            count=count,
        )
        energy_flux = _array(
            "kinetic_energy_defect_flux",
            self.kinetic_energy_defect_flux,
            shape_tail=(3,),
            count=count,
        )
        velocity = _array(
            "external_tangential_velocity",
            self.external_tangential_velocity,
            shape_tail=(3,),
            count=count,
        )
        gradient = _array(
            "external_velocity_surface_gradient",
            self.external_velocity_surface_gradient,
            shape_tail=(3, 3),
            count=count,
        )
        shear = _array(
            "wall_shear_over_density",
            self.wall_shear_over_density,
            shape_tail=(3,),
            count=count,
        )
        dissipation = _scalars(
            "dissipation_integral",
            self.dissipation_integral,
            count,
        )
        normal = _array(
            "surface_normal",
            self.surface_normal,
            shape_tail=(3,),
            count=count,
        )
        normal_error = np.max(
            np.abs(np.linalg.norm(normal, axis=1)-1.0),
            initial=0.0,
        )
        if normal_error > 1.0e-12:
            raise SurfaceIBLError("surface_normal must contain unit vectors")

        tangent_errors = [
            np.max(np.abs(np.einsum("ni,ni->n", value, normal)), initial=0.0)
            for value in (mass, energy_flux, velocity, shear)
        ]
        tangent_errors.extend([
            np.max(
                np.abs(np.einsum("ni,nij->nj", normal, tensor)),
                initial=0.0,
            ),
            np.max(
                np.abs(np.einsum("nij,nj->ni", tensor, normal)),
                initial=0.0,
            ),
            np.max(
                np.abs(np.einsum("ni,nij->nj", normal, gradient)),
                initial=0.0,
            ),
            np.max(
                np.abs(np.einsum("nij,nj->ni", gradient, normal)),
                initial=0.0,
            ),
        ])
        if max(tangent_errors, default=0.0) > 1.0e-11:
            raise SurfaceIBLError(
                "IBL vectors and both tensor slots must lie in the tangent plane"
            )

        for name, value in (
            ("mass_flux_defect", mass),
            ("momentum_flux_defect", tensor),
            ("kinetic_energy_defect_flux", energy_flux),
            ("external_tangential_velocity", velocity),
            ("external_velocity_surface_gradient", gradient),
            ("wall_shear_over_density", shear),
            ("dissipation_integral", dissipation),
            ("surface_normal", normal),
        ):
            object.__setattr__(self, name, value)

    @property
    def count(self) -> int:
        return len(self.mass_flux_defect)

    @property
    def momentum_flux_trace(self) -> np.ndarray:
        """The scalar storage variable in the integrated energy equation."""
        return np.trace(
            self.momentum_flux_defect,
            axis1=1,
            axis2=2,
        )


@dataclass(frozen=True)
class SurfaceIBLSourceTerms:
    momentum: np.ndarray
    energy: np.ndarray


@dataclass(frozen=True)
class SurfaceIBLPhysicalFlux:
    momentum_out: np.ndarray
    energy_out: np.ndarray


@dataclass(frozen=True)
class SurfaceIBLBudgetReport:
    momentum_storage_rate: np.ndarray
    energy_storage_rate: np.ndarray
    internal_momentum_net_in_rate: np.ndarray
    internal_energy_net_in_rate: np.ndarray
    boundary_momentum_net_in_rate: np.ndarray
    boundary_energy_net_in_rate: np.ndarray
    momentum_source_integral_rate: np.ndarray
    energy_source_integral_rate: np.ndarray
    momentum_residual: np.ndarray
    energy_residual: np.ndarray
    max_momentum_residual: float
    max_energy_residual: float
    global_internal_momentum_flux_residual: float
    global_internal_energy_flux_residual: float
    passed: bool


def surface_ibl_source_terms(fields: SurfaceIBLFields) -> SurfaceIBLSourceTerms:
    """Evaluate the exact named sources without providing their closure."""
    if not isinstance(fields, SurfaceIBLFields):
        raise SurfaceIBLError("fields must be SurfaceIBLFields")
    gradient_times_mass = np.einsum(
        "nij,nj->ni",
        fields.external_velocity_surface_gradient,
        fields.mass_flux_defect,
    )
    momentum = -gradient_times_mass+fields.wall_shear_over_density
    tensor_gradient = np.einsum(
        "nij,nij->n",
        fields.momentum_flux_defect,
        fields.external_velocity_surface_gradient,
    )
    energy = (
        2.0*fields.dissipation_integral
        - tensor_gradient
        + np.einsum(
            "ni,ni->n",
            fields.external_tangential_velocity,
            gradient_times_mass-fields.wall_shear_over_density,
        )
    )
    return SurfaceIBLSourceTerms(momentum=momentum, energy=energy)


def surface_ibl_physical_flux(
    fields: SurfaceIBLFields,
    *,
    outward_surface_conormal,
    edge_measure=1.0,
) -> SurfaceIBLPhysicalFlux:
    """Contract the exact IBL flux tensors with an outward surface conormal."""
    if not isinstance(fields, SurfaceIBLFields):
        raise SurfaceIBLError("fields must be SurfaceIBLFields")
    conormal = _array(
        "outward_surface_conormal",
        outward_surface_conormal,
        shape_tail=(3,),
        count=fields.count,
    )
    measure = np.asarray(edge_measure, dtype=float)
    if measure.ndim == 0:
        measure = np.full(fields.count, float(measure))
    measure = _scalars("edge_measure", measure, fields.count)
    if np.any(measure <= 0.0):
        raise SurfaceIBLError("edge_measure must be positive")
    unit_error = np.max(
        np.abs(np.linalg.norm(conormal, axis=1)-1.0),
        initial=0.0,
    )
    normal_dot = np.max(
        np.abs(np.einsum("ni,ni->n", conormal, fields.surface_normal)),
        initial=0.0,
    )
    if unit_error > 1.0e-12 or normal_dot > 1.0e-12:
        raise SurfaceIBLError(
            "outward_surface_conormal must be a unit tangent vector"
        )
    momentum = np.einsum(
        "nij,nj->ni",
        fields.momentum_flux_defect,
        conormal,
    )
    energy_transport = (
        fields.kinetic_energy_defect_flux
        - np.einsum(
            "nij,nj->ni",
            fields.momentum_flux_defect,
            fields.external_tangential_velocity,
        )
    )
    energy = np.einsum("ni,ni->n", energy_transport, conormal)
    return SurfaceIBLPhysicalFlux(
        momentum_out=momentum*measure[:, None],
        energy_out=energy*measure,
    )


def rotate_surface_ibl_fields(
    fields: SurfaceIBLFields,
    rotation,
) -> SurfaceIBLFields:
    """Express the same physical fields in a properly rotated frame."""
    if not isinstance(fields, SurfaceIBLFields):
        raise SurfaceIBLError("fields must be SurfaceIBLFields")
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise SurfaceIBLError("rotation must have shape (3,3)")
    if (
        np.max(np.abs(matrix.T@matrix-np.eye(3)), initial=0.0) > 1.0e-12
        or abs(float(np.linalg.det(matrix))-1.0) > 1.0e-12
    ):
        raise SurfaceIBLError("rotation must be proper orthogonal")

    def vector(value):
        return np.einsum("ij,nj->ni", matrix, value)

    def tensor(value):
        return np.einsum("ia,nab,jb->nij", matrix, value, matrix)

    return SurfaceIBLFields(
        mass_flux_defect=vector(fields.mass_flux_defect),
        momentum_flux_defect=tensor(fields.momentum_flux_defect),
        kinetic_energy_defect_flux=vector(
            fields.kinetic_energy_defect_flux
        ),
        external_tangential_velocity=vector(
            fields.external_tangential_velocity
        ),
        external_velocity_surface_gradient=tensor(
            fields.external_velocity_surface_gradient
        ),
        wall_shear_over_density=vector(fields.wall_shear_over_density),
        dissipation_integral=fields.dissipation_integral,
        surface_normal=vector(fields.surface_normal),
    )


def surface_ibl_budget_report(
    *,
    previous_fields: SurfaceIBLFields,
    current_fields: SurfaceIBLFields,
    previous_cell_area,
    current_cell_area,
    dt: float,
    internal_edges,
    internal_momentum_flux_out_of_first_rate,
    internal_energy_flux_out_of_first_rate,
    boundary_momentum_net_in_rate,
    boundary_energy_net_in_rate,
    momentum_source_integral_rate,
    energy_source_integral_rate,
    tolerance: float = 1.0e-12,
) -> SurfaceIBLBudgetReport:
    """Check extensive finite-volume storage against named fluxes and sources."""
    if (
        not isinstance(previous_fields, SurfaceIBLFields)
        or not isinstance(current_fields, SurfaceIBLFields)
    ):
        raise SurfaceIBLError(
            "previous_fields and current_fields must be SurfaceIBLFields"
        )
    count = previous_fields.count
    if current_fields.count != count:
        raise SurfaceIBLError("field snapshots must have the same cell count")
    area_previous = _scalars(
        "previous_cell_area",
        previous_cell_area,
        count,
    )
    area_current = _scalars(
        "current_cell_area",
        current_cell_area,
        count,
    )
    if np.any(area_previous <= 0.0) or np.any(area_current <= 0.0):
        raise SurfaceIBLError("cell areas must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise SurfaceIBLError("dt must be positive and finite")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise SurfaceIBLError("tolerance must be finite and non-negative")

    edges = np.asarray(internal_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise SurfaceIBLError("internal_edges must have shape (m,2)")
    if (
        np.any(edges < 0)
        or np.any(edges >= count)
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise SurfaceIBLError(
            "internal edges must connect two distinct valid cells"
        )
    momentum_flux = _array(
        "internal_momentum_flux_out_of_first_rate",
        internal_momentum_flux_out_of_first_rate,
        shape_tail=(3,),
    )
    energy_flux = _scalars(
        "internal_energy_flux_out_of_first_rate",
        internal_energy_flux_out_of_first_rate,
    )
    if len(momentum_flux) != len(edges) or len(energy_flux) != len(edges):
        raise SurfaceIBLError("one momentum and energy flux is needed per edge")

    boundary_momentum = _array(
        "boundary_momentum_net_in_rate",
        boundary_momentum_net_in_rate,
        shape_tail=(3,),
        count=count,
    )
    boundary_energy = _scalars(
        "boundary_energy_net_in_rate",
        boundary_energy_net_in_rate,
        count,
    )
    momentum_source = _array(
        "momentum_source_integral_rate",
        momentum_source_integral_rate,
        shape_tail=(3,),
        count=count,
    )
    energy_source = _scalars(
        "energy_source_integral_rate",
        energy_source_integral_rate,
        count,
    )

    internal_momentum = np.zeros((count, 3), dtype=float)
    internal_energy = np.zeros(count, dtype=float)
    for (first, second), momentum_out, energy_out in zip(
        edges,
        momentum_flux,
        energy_flux,
    ):
        internal_momentum[int(first)] -= momentum_out
        internal_momentum[int(second)] += momentum_out
        internal_energy[int(first)] -= energy_out
        internal_energy[int(second)] += energy_out

    momentum_storage = (
        area_current[:, None]*current_fields.mass_flux_defect
        - area_previous[:, None]*previous_fields.mass_flux_defect
    )/dt
    energy_storage = (
        area_current*current_fields.momentum_flux_trace
        - area_previous*previous_fields.momentum_flux_trace
    )/dt
    momentum_residual = (
        momentum_storage
        - internal_momentum
        - boundary_momentum
        - momentum_source
    )
    energy_residual = (
        energy_storage
        - internal_energy
        - boundary_energy
        - energy_source
    )
    max_momentum = float(
        np.max(np.linalg.norm(momentum_residual, axis=1), initial=0.0)
    )
    max_energy = float(np.max(np.abs(energy_residual), initial=0.0))
    global_internal_momentum = float(
        np.linalg.norm(np.sum(internal_momentum, axis=0))
    )
    global_internal_energy = float(abs(np.sum(internal_energy)))
    return SurfaceIBLBudgetReport(
        momentum_storage_rate=momentum_storage,
        energy_storage_rate=energy_storage,
        internal_momentum_net_in_rate=internal_momentum,
        internal_energy_net_in_rate=internal_energy,
        boundary_momentum_net_in_rate=boundary_momentum,
        boundary_energy_net_in_rate=boundary_energy,
        momentum_source_integral_rate=momentum_source,
        energy_source_integral_rate=energy_source,
        momentum_residual=momentum_residual,
        energy_residual=energy_residual,
        max_momentum_residual=max_momentum,
        max_energy_residual=max_energy,
        global_internal_momentum_flux_residual=global_internal_momentum,
        global_internal_energy_flux_residual=global_internal_energy,
        passed=max(
            max_momentum,
            max_energy,
            global_internal_momentum,
            global_internal_energy,
        ) <= tolerance,
    )

