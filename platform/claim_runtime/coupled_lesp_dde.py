"""No-force algebraic coupling between the frozen N1 solve and a DDE source.

At one physical time level, a caller supplies the normal-velocity influence
of one unit nascent DDE sheet basis per spanwise strip.  This module solves
the bound system and the LESP equality together, retaining the source/bound
reaction in one ledger.  It does not choose LEV geometry, infer a half-time
state, convect a wake, compute pressure, or book force.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hirato_equations import (
    HiratoEquationError,
    lesp_eq6,
    lesp_sensitivity_eq6,
    preconstraint_shed_mask,
    solve_lesp_constraint,
)
from .continuous_shedding import newborn_halfwing_shedding_band
from .distributed_doublet import (
    DistributedDoubletError,
    QuadraticDoubletSurface,
)


@dataclass(frozen=True)
class CoupledLespStage:
    bound_pre: np.ndarray
    bound_post: np.ndarray
    gamma_lev: np.ndarray
    a0_pre: np.ndarray
    a0_post: np.ndarray
    active: np.ndarray
    aic_condition_number: float
    lesp_condition_number: float
    bound_equation_max_abs_residual: float
    lesp_active_max_abs_residual: float
    passed: bool


@dataclass(frozen=True)
class DDEUnitInfluence:
    normal_influence: np.ndarray
    basis_surfaces: tuple[QuadraticDoubletSurface, ...]
    quadrature_order: int
    finite: bool


def mirror_halfwing_surface(
    surface: QuadraticDoubletSurface,
) -> QuadraticDoubletSurface:
    """Reflect an oriented DDE surface through the half-wing root plane.

    Reflection reverses orientation.  Reversing each face winding and its P2
    local degrees of freedom restores the physical image convention used by
    the frozen N1 half-wing AIC.
    """
    vertices = surface.vertices.copy()
    vertices[:, 1] *= -1.0
    faces = surface.faces[:, [0, 2, 1]]
    # Reversing local vertices 1/2 maps
    # (v0,v1,v2,e01,e12,e20) -> (v0,v2,v1,e20,e12,e01).
    face_mu = surface.face_mu[:, [0, 2, 1, 5, 4, 3]]
    return QuadraticDoubletSurface(vertices, faces, face_mu)


_mirror_halfwing_surface = mirror_halfwing_surface


def build_dde_unit_strip_normal_influence(
    *,
    collocation,
    normals,
    previous_edge,
    current_edge,
    span_edges,
    time_nodes,
    quadrature_order: int = 96,
    mirror_symmetry: bool = True,
) -> DDEUnitInfluence:
    """Evaluate one continuous P2 nascent-sheet basis per N1 strip."""
    points = np.asarray(collocation, dtype=float)
    normal = np.asarray(normals, dtype=float)
    edges = np.asarray(span_edges, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or normal.shape != points.shape
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(normal))
    ):
        raise HiratoEquationError(
            "collocation and normals must be finite arrays with shape (n,3)"
        )
    if (
        not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 2
    ):
        raise HiratoEquationError(
            "quadrature_order must be an integer >=2"
        )
    strip_count = len(edges) - 1
    if strip_count < 1:
        raise HiratoEquationError("span_edges must define at least one strip")
    columns = []
    surfaces = []
    for source_strip in range(strip_count):
        strip_basis = np.zeros((3, strip_count))
        strip_basis[:, source_strip] = 1.0
        birth = newborn_halfwing_shedding_band(
            sheet_id=f"unit-lev-strip-{source_strip}",
            vortex_family="LEV_SUCTION",
            previous_edge=previous_edge,
            current_edge=current_edge,
            span_edges=edges,
            time_nodes=time_nodes,
            strip_strength_rows=strip_basis,
        )
        surface = birth.band.surface
        try:
            velocity = surface.induced_velocity_line_reduced(
                points,
                quadrature_order=int(quadrature_order),
            )
            if mirror_symmetry:
                velocity += mirror_halfwing_surface(
                    surface
                ).induced_velocity_line_reduced(
                    points,
                    quadrature_order=int(quadrature_order),
                )
        except DistributedDoubletError as error:
            raise HiratoEquationError(
                "DDE unit source field evaluation failed"
            ) from error
        columns.append(np.einsum("ij,ij->i", velocity, normal))
        surfaces.append(surface)
    influence = np.column_stack(columns)
    return DDEUnitInfluence(
        normal_influence=influence,
        basis_surfaces=tuple(surfaces),
        quadrature_order=int(quadrature_order),
        finite=bool(np.all(np.isfinite(influence))),
    )


def solve_coupled_lesp_dde_stage(
    *,
    aic,
    rhs_without_nascent_lev,
    unit_lev_normal_influence,
    u_infinity: float,
    chord,
    delta_x_front,
    lesp_crit: float,
    equation_tolerance: float = 1.0e-10,
) -> CoupledLespStage:
    """Solve one N1/DDE LESP stage with explicit action/reaction."""
    matrix = np.asarray(aic, dtype=float)
    rhs = np.asarray(rhs_without_nascent_lev, dtype=float)
    influence = np.asarray(unit_lev_normal_influence, dtype=float)
    local_chord = np.asarray(chord, dtype=float)
    delta_x = np.asarray(delta_x_front, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or rhs.shape != (matrix.shape[0],)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(rhs))
    ):
        raise HiratoEquationError(
            "aic must be finite square and rhs must match it"
        )
    ns = len(local_chord)
    if (
        local_chord.shape != (ns,)
        or delta_x.shape != (ns,)
        or influence.shape != (len(rhs), ns)
        or not np.all(np.isfinite(influence))
    ):
        raise HiratoEquationError(
            "DDE influence/chord arrays have incompatible shapes"
        )
    if equation_tolerance < 0.0 or not np.isfinite(equation_tolerance):
        raise HiratoEquationError(
            "equation_tolerance must be finite and non-negative"
        )
    bound_pre = np.linalg.solve(matrix, rhs)
    a0_pre = lesp_eq6(
        bound_pre[:ns],
        u_infinity,
        local_chord,
        delta_x,
    )
    active = preconstraint_shed_mask(a0_pre, lesp_crit)
    gamma = np.zeros(ns)
    bound_post = bound_pre.copy()
    a0_post = a0_pre.copy()
    lesp_condition = 1.0
    if np.any(active):
        response = np.linalg.solve(matrix, -influence)
        sensitivity = lesp_sensitivity_eq6(
            response.T,
            u_infinity,
            local_chord,
            delta_x,
        )
        constraint = solve_lesp_constraint(
            a0_pre,
            sensitivity,
            active,
            lesp_crit,
        )
        gamma = constraint.gamma_lev
        bound_post = np.linalg.solve(
            matrix,
            rhs - influence @ gamma,
        )
        a0_post = lesp_eq6(
            bound_post[:ns],
            u_infinity,
            local_chord,
            delta_x,
        )
        lesp_condition = constraint.condition_number
    bound_residual = matrix @ bound_post - (
        rhs - influence @ gamma
    )
    target = lesp_crit * np.sign(a0_pre[active])
    active_residual = a0_post[active] - target
    max_bound = float(
        np.max(np.abs(bound_residual), initial=0.0)
    )
    max_lesp = float(
        np.max(np.abs(active_residual), initial=0.0)
    )
    return CoupledLespStage(
        bound_pre=bound_pre,
        bound_post=bound_post,
        gamma_lev=gamma,
        a0_pre=a0_pre,
        a0_post=a0_post,
        active=active,
        aic_condition_number=float(np.linalg.cond(matrix)),
        lesp_condition_number=float(lesp_condition),
        bound_equation_max_abs_residual=max_bound,
        lesp_active_max_abs_residual=max_lesp,
        passed=bool(
            max_bound <= equation_tolerance
            and max_lesp <= equation_tolerance
        ),
    )
