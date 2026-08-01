"""No-force material advection for a continuous doublet sheet.

The spatial field is sampled at strict-interior DDE points, projected to the
shared P1 geometry space, and advanced with explicit Heun (RK2).  Doublet
strengths remain attached to material faces.  This module does not generate a
new shedding row, solve bound circulation, compute pressure, or book force.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .distributed_doublet import (
    DistributedDoubletError,
    DoubletAssemblyReport,
    DoubletBoundaryReport,
    DoubletContinuityReport,
    MaterialKelvinReport,
    QuadraticDoubletAssembly,
    QuadraticDoubletPatch,
    QuadraticDoubletSurface,
)
from .sheet_velocity_projection import (
    ProjectedAssemblyNormalGeometryVelocity,
    ProjectedNormalGeometryVelocity,
    ProjectedSheetVelocity,
    SheetVelocityProjectionReport,
    project_assembly_vertex_star_normal_geometry_velocity,
    project_sheet_normal_geometry_velocity,
    project_sheet_average_velocity,
    project_vertex_star_normal_geometry_velocity,
)


VelocityProvider = Callable[
    [QuadraticDoubletSurface],
    ProjectedSheetVelocity,
]


@dataclass(frozen=True)
class MaterialAdvectionReport:
    dt: float
    stage0_projection: SheetVelocityProjectionReport
    stage1_projection: SheetVelocityProjectionReport
    kelvin: MaterialKelvinReport
    continuity: DoubletContinuityReport
    boundary: DoubletBoundaryReport
    passed: bool


@dataclass(frozen=True)
class MaterialAdvectionStep:
    surface: QuadraticDoubletSurface
    report: MaterialAdvectionReport


@dataclass(frozen=True)
class NormalGeometryAdvectionStep:
    surface: QuadraticDoubletSurface
    report: MaterialAdvectionReport


@dataclass(frozen=True)
class AssemblyMaterialAdvectionReport:
    dt: float
    stage0_projection: SheetVelocityProjectionReport
    stage1_projection: SheetVelocityProjectionReport
    patch_kelvin: tuple[MaterialKelvinReport, ...]
    topology: DoubletAssemblyReport
    passed: bool


@dataclass(frozen=True)
class AssemblyNormalGeometryAdvectionStep:
    assembly: QuadraticDoubletAssembly
    report: AssemblyMaterialAdvectionReport


def self_induced_geometry_velocity(
    surface: QuadraticDoubletSurface,
    *,
    quadrature_order: int = 48,
) -> ProjectedSheetVelocity:
    """Return the no-core sheet-average velocity in the P1 geometry space."""
    _, owners, barycentric = surface.interior_collocation_points()
    velocity = surface.induced_velocity_sheet_average(
        owners,
        barycentric,
        quadrature_order=quadrature_order,
    )
    return project_sheet_average_velocity(surface, velocity)


def self_induced_normal_geometry_velocity(
    surface: QuadraticDoubletSurface,
    *,
    quadrature_order: int = 48,
) -> ProjectedNormalGeometryVelocity:
    """Return only the sheet-average normal velocity in P1 geometry space."""
    _, owners, barycentric = surface.interior_collocation_points()
    velocity = surface.induced_velocity_sheet_average(
        owners,
        barycentric,
        quadrature_order=quadrature_order,
    )
    return project_sheet_normal_geometry_velocity(surface, velocity)


def self_induced_vertex_star_normal_velocity(
    surface: QuadraticDoubletSurface,
    *,
    quadrature_order: int = 48,
) -> ProjectedNormalGeometryVelocity:
    """Return the local vertex-star limit of the sheet-normal velocity."""
    _, owners, barycentric = surface.interior_collocation_points()
    velocity = surface.induced_velocity_sheet_average(
        owners,
        barycentric,
        quadrature_order=quadrature_order,
    )
    return project_vertex_star_normal_geometry_velocity(
        surface,
        velocity,
    )


def self_induced_assembly_vertex_star_normal_velocity(
    assembly: QuadraticDoubletAssembly,
    *,
    quadrature_order: int = 48,
    geometry_tolerance: float = 1.0e-12,
) -> ProjectedAssemblyNormalGeometryVelocity:
    """Return the seam-welded vertex-star normal velocity of an assembly."""
    if not isinstance(assembly, QuadraticDoubletAssembly):
        raise DistributedDoubletError(
            "assembly must be QuadraticDoubletAssembly"
        )
    _, patch_owner, face_owner, barycentric = (
        assembly.interior_collocation_points()
    )
    velocity = assembly.induced_velocity_sheet_average(
        patch_owner,
        face_owner,
        barycentric,
        quadrature_order=quadrature_order,
    )
    return project_assembly_vertex_star_normal_geometry_velocity(
        assembly,
        velocity,
        geometry_tolerance=geometry_tolerance,
    )


def advance_material_surface_heun(
    surface: QuadraticDoubletSurface,
    *,
    dt: float,
    velocity_provider: VelocityProvider,
    continuity_tolerance: float = 1.0e-12,
    boundary_tolerance: float = 1.0e-12,
    kelvin_tolerance: float = 1.0e-12,
) -> MaterialAdvectionStep:
    """Advance one material step without changing the potential jump."""
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be QuadraticDoubletSurface"
        )
    if dt <= 0.0 or not np.isfinite(dt):
        raise DistributedDoubletError("dt must be finite and positive")
    if not callable(velocity_provider):
        raise DistributedDoubletError("velocity_provider must be callable")
    tolerances = (
        continuity_tolerance,
        boundary_tolerance,
        kelvin_tolerance,
    )
    if any(
        tolerance < 0.0 or not np.isfinite(tolerance)
        for tolerance in tolerances
    ):
        raise DistributedDoubletError(
            "advection tolerances must be finite and non-negative"
        )

    stage0 = velocity_provider(surface)
    if not isinstance(stage0, ProjectedSheetVelocity):
        raise DistributedDoubletError(
            "velocity_provider must return ProjectedSheetVelocity"
        )
    predicted = surface.material_update(
        surface.vertices + dt * stage0.vertex_velocity
    )
    stage1 = velocity_provider(predicted)
    if not isinstance(stage1, ProjectedSheetVelocity):
        raise DistributedDoubletError(
            "velocity_provider must return ProjectedSheetVelocity"
        )
    advanced = surface.material_update(
        surface.vertices
        + 0.5
        * dt
        * (stage0.vertex_velocity + stage1.vertex_velocity)
    )
    kelvin = surface.kelvin_report(
        advanced,
        tolerance=kelvin_tolerance,
    )
    continuity = advanced.continuity_report(
        tolerance=continuity_tolerance
    )
    boundary = advanced.boundary_report(
        tolerance=boundary_tolerance
    )
    passed = (
        stage0.report.full_rank
        and stage1.report.full_rank
        and kelvin.passed
        and continuity.compatible
        and boundary.compatible
    )
    return MaterialAdvectionStep(
        surface=advanced,
        report=MaterialAdvectionReport(
            dt=float(dt),
            stage0_projection=stage0.report,
            stage1_projection=stage1.report,
            kelvin=kelvin,
            continuity=continuity,
            boundary=boundary,
            passed=passed,
        ),
    )


def advance_surface_normal_geometry_heun(
    surface: QuadraticDoubletSurface,
    *,
    dt: float,
    velocity_provider: Callable[
        [QuadraticDoubletSurface],
        ProjectedNormalGeometryVelocity,
    ],
    continuity_tolerance: float = 1.0e-12,
    boundary_tolerance: float = 1.0e-12,
    kelvin_tolerance: float = 1.0e-12,
) -> NormalGeometryAdvectionStep:
    """Advance sheet shape in the zero-tangential reparameterization gauge."""
    if not isinstance(surface, QuadraticDoubletSurface):
        raise DistributedDoubletError(
            "surface must be QuadraticDoubletSurface"
        )
    if dt <= 0.0 or not np.isfinite(dt):
        raise DistributedDoubletError("dt must be finite and positive")
    if not callable(velocity_provider):
        raise DistributedDoubletError("velocity_provider must be callable")
    stage0 = velocity_provider(surface)
    if not isinstance(stage0, ProjectedNormalGeometryVelocity):
        raise DistributedDoubletError(
            "normal velocity provider returned the wrong type"
        )
    predicted = surface.material_update(
        surface.vertices + dt * stage0.vertex_velocity
    )
    stage1 = velocity_provider(predicted)
    if not isinstance(stage1, ProjectedNormalGeometryVelocity):
        raise DistributedDoubletError(
            "normal velocity provider returned the wrong type"
        )
    advanced = surface.material_update(
        surface.vertices
        + 0.5
        * dt
        * (stage0.vertex_velocity + stage1.vertex_velocity)
    )
    kelvin = surface.kelvin_report(
        advanced,
        tolerance=kelvin_tolerance,
    )
    continuity = advanced.continuity_report(
        tolerance=continuity_tolerance
    )
    boundary = advanced.boundary_report(
        tolerance=boundary_tolerance
    )
    passed = (
        stage0.report.full_rank
        and stage1.report.full_rank
        and kelvin.passed
        and continuity.compatible
        and boundary.compatible
    )
    return NormalGeometryAdvectionStep(
        surface=advanced,
        report=MaterialAdvectionReport(
            dt=float(dt),
            stage0_projection=stage0.report,
            stage1_projection=stage1.report,
            kelvin=kelvin,
            continuity=continuity,
            boundary=boundary,
            passed=passed,
        ),
    )


def _update_assembly_geometry(
    assembly: QuadraticDoubletAssembly,
    patch_vertex_velocity: tuple[np.ndarray, ...],
    *,
    dt: float,
) -> QuadraticDoubletAssembly:
    if len(patch_vertex_velocity) != len(assembly.patches):
        raise DistributedDoubletError(
            "patch_vertex_velocity count does not match assembly"
        )
    updated = []
    for patch, velocity in zip(
        assembly.patches,
        patch_vertex_velocity,
    ):
        value = np.asarray(velocity, dtype=float)
        if value.shape != patch.surface.vertices.shape:
            raise DistributedDoubletError(
                f"patch {patch.name!r} velocity has shape {value.shape}; "
                f"expected {patch.surface.vertices.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise DistributedDoubletError(
                f"patch {patch.name!r} velocity is non-finite"
            )
        updated.append(
            QuadraticDoubletPatch(
                patch.name,
                patch.surface.material_update(
                    patch.surface.vertices + dt * value
                ),
                patch.boundary_roles,
            )
        )
    return QuadraticDoubletAssembly(updated)


def advance_assembly_normal_geometry_heun(
    assembly: QuadraticDoubletAssembly,
    *,
    dt: float,
    velocity_provider: Callable[
        [QuadraticDoubletAssembly],
        ProjectedAssemblyNormalGeometryVelocity,
    ],
    topology_strength_tolerance: float = 1.0e-12,
    topology_geometry_tolerance: float = 1.0e-12,
    kelvin_tolerance: float = 1.0e-12,
) -> AssemblyNormalGeometryAdvectionStep:
    """Advance an explicit multi-patch topology with one velocity per seam DOF.

    The provider must return a projection whose patch-local vertex arrays are
    views of welded global degrees of freedom.  The same stage velocity is
    therefore applied to both copies of every declared interface vertex.
    Potential-jump values and boundary roles remain material identities.
    """
    if not isinstance(assembly, QuadraticDoubletAssembly):
        raise DistributedDoubletError(
            "assembly must be QuadraticDoubletAssembly"
        )
    if dt <= 0.0 or not np.isfinite(dt):
        raise DistributedDoubletError("dt must be finite and positive")
    if not callable(velocity_provider):
        raise DistributedDoubletError("velocity_provider must be callable")
    tolerances = (
        topology_strength_tolerance,
        topology_geometry_tolerance,
        kelvin_tolerance,
    )
    if any(
        tolerance < 0.0 or not np.isfinite(tolerance)
        for tolerance in tolerances
    ):
        raise DistributedDoubletError(
            "advection tolerances must be finite and non-negative"
        )

    stage0 = velocity_provider(assembly)
    if not isinstance(
        stage0,
        ProjectedAssemblyNormalGeometryVelocity,
    ):
        raise DistributedDoubletError(
            "assembly velocity provider returned the wrong type"
        )
    stage0_patch_velocity = tuple(
        stage0.vertex_velocity(index)
        for index in range(len(assembly.patches))
    )
    predicted = _update_assembly_geometry(
        assembly,
        stage0_patch_velocity,
        dt=dt,
    )
    stage1 = velocity_provider(predicted)
    if not isinstance(
        stage1,
        ProjectedAssemblyNormalGeometryVelocity,
    ):
        raise DistributedDoubletError(
            "assembly velocity provider returned the wrong type"
        )
    stage1_patch_velocity = tuple(
        stage1.vertex_velocity(index)
        for index in range(len(assembly.patches))
    )
    mean_velocity = tuple(
        0.5 * (first + second)
        for first, second in zip(
            stage0_patch_velocity,
            stage1_patch_velocity,
        )
    )
    advanced = _update_assembly_geometry(
        assembly,
        mean_velocity,
        dt=dt,
    )
    patch_kelvin = tuple(
        original.surface.kelvin_report(
            current.surface,
            tolerance=kelvin_tolerance,
        )
        for original, current in zip(
            assembly.patches,
            advanced.patches,
        )
    )
    topology = advanced.topology_report(
        strength_tolerance=topology_strength_tolerance,
        geometry_tolerance=topology_geometry_tolerance,
    )
    passed = (
        stage0.report.full_rank
        and stage1.report.full_rank
        and all(report.passed for report in patch_kelvin)
        and topology.compatible
    )
    return AssemblyNormalGeometryAdvectionStep(
        assembly=advanced,
        report=AssemblyMaterialAdvectionReport(
            dt=float(dt),
            stage0_projection=stage0.report,
            stage1_projection=stage1.report,
            patch_kelvin=patch_kelvin,
            topology=topology,
            passed=passed,
        ),
    )
