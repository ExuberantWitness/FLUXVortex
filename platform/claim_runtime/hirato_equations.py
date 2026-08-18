"""Equation-level invariants for the N3.1i Hirato LEV-sheet candidate.

These functions implement only identities stated in Hirato et al. (2019),
DOI 10.2514/1.C035124.  They are deliberately not connected to the V4.1 force
path.  The historical ``lev_shed_mode='hirato'`` branch is incomplete and must
not use this module's existence as evidence of a faithful implementation.

All circulation inputs use the paper's sign convention.  A future runtime
adapter must make any solver-sign conversion explicit at its boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class HiratoEquationError(ValueError):
    """An equation input has an invalid shape or value."""


RAMESH_2014_LDVM_V25_BIRTH_SOURCE_TAG = (
    "ramesh-jfm-2014-sec2.3__ldvm-v2.5-ldvm.f95-L300-L327"
)


def _finite_array(name: str, value, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise HiratoEquationError(f"{name} must have ndim={ndim}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise HiratoEquationError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class LespConstraintSolution:
    """Affine LESP constraint result for active spanwise strips."""

    gamma_lev: np.ndarray
    a0_post: np.ndarray
    active_residual: np.ndarray
    condition_number: float


def first_vortex_displacement_ansari(
    local_edge_velocity,
    dt: float,
) -> np.ndarray:
    """Ansari et al. first-vortex half-step placement.

    In the two-dimensional complex-plane formulation of Ansari, Zbikowski &
    Knowles (2006), the first shed vortex is placed using one half of the
    local shedding-edge velocity over one time step.  This function preserves
    only that kinematic identity in Cartesian-vector form:

    ``delta_r = 0.5 * q_edge * dt``.

    Treating a physical three-dimensional strip-edge velocity as ``q_edge``
    is a separate finite-wing adapter hypothesis; this identity is not
    Hirato Eq.7 and is not connected to the production force path.
    """
    velocity = _finite_array("local_edge_velocity", local_edge_velocity)
    if velocity.ndim < 1 or velocity.shape[-1] not in (2, 3):
        raise HiratoEquationError(
            "local_edge_velocity must end in a 2D or 3D vector, "
            f"got {velocity.shape}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise HiratoEquationError(f"dt must be positive and finite, got {dt}")
    return 0.5 * velocity * dt


def first_lev_displacement_ramesh_2014_ldvm_v25(
    local_edge_velocity,
    dt: float,
) -> np.ndarray:
    """Ramesh mature-LDVM first/restart LEV half-step placement.

    Ramesh et al. (JFM 751, 2014), section 2.3, defines a newly initiated
    shed vortex from the velocity at its shedding edge.  The distributed
    LDVM-v2.5 source makes the coefficient explicit in ``ldvm.f95`` lines
    300--327: a first LEV, including one after ``levflag`` is reset, is placed
    at

    ``delta_r = 0.5 * q_LE * dt``.

    The mathematical half-step construction is inherited from Ansari,
    Zbikowski & Knowles (2006).  This source-tagged wrapper distinguishes the
    mature Ramesh first/restart law from the earlier Ramesh et al. (2012)
    ``U*A0*dt/sqrt(2)`` law implemented by
    :func:`first_lev_displacement_ramesh_2d`.  Which velocity components make
    up ``q_LE`` and how the 2-D law is mapped to a finite wing remain explicit
    solver-adapter responsibilities.
    """

    return first_vortex_displacement_ansari(local_edge_velocity, dt)


def first_lev_displacement_ramesh_2d(
    u_infinity: float,
    a0,
    alpha_rad,
    dt: float,
) -> np.ndarray:
    """Ramesh et al. first-LEV placement in its published 2D basis.

    The LESP-modulated discrete-vortex model places the first LEV at

    ``delta_(x,z) = U_inf*A0*dt/sqrt(2) * (sin(alpha), cos(alpha))``.

    ``A0`` retains its sign.  The returned last dimension is ``[x, z]`` in
    the paper's two-dimensional coordinate convention.  Mapping these two
    components to a curved finite-wing strip is deliberately handled by
    :func:`embed_chord_normal_displacement`, so that the 3D adapter cannot be
    mistaken for a published identity.
    """
    lesp = _finite_array("a0", a0)
    alpha = _finite_array("alpha_rad", alpha_rad)
    if lesp.shape != alpha.shape:
        raise HiratoEquationError(
            f"a0 and alpha_rad must have matching shapes, got {lesp.shape} and {alpha.shape}"
        )
    if not np.isfinite(u_infinity) or u_infinity <= 0.0:
        raise HiratoEquationError(
            f"u_infinity must be positive and finite, got {u_infinity}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise HiratoEquationError(f"dt must be positive and finite, got {dt}")
    scale = u_infinity * lesp * dt / np.sqrt(2.0)
    return np.stack(
        [scale * np.sin(alpha), scale * np.cos(alpha)],
        axis=-1,
    )


def embed_chord_normal_displacement(
    displacement_xz,
    chord_tangent,
    suction_normal,
    *,
    orthogonality_tolerance: float = 1e-10,
) -> np.ndarray:
    """Embed a published 2D displacement in an explicit finite-wing basis.

    This is an adapter hypothesis, not an equation from Ansari, Ramesh or
    Hirato.  The caller must provide unit chord-tangent and suction-normal
    vectors with the same leading shape as ``displacement_xz``.  Refusing to
    normalize or orthogonalize them here keeps geometry errors visible.
    """
    displacement = _finite_array("displacement_xz", displacement_xz)
    tangent = _finite_array("chord_tangent", chord_tangent)
    normal = _finite_array("suction_normal", suction_normal)
    if displacement.ndim < 1 or displacement.shape[-1] != 2:
        raise HiratoEquationError(
            f"displacement_xz must end in [x,z], got {displacement.shape}"
        )
    expected_basis_shape = displacement.shape[:-1] + (3,)
    if tangent.shape != expected_basis_shape or normal.shape != expected_basis_shape:
        raise HiratoEquationError(
            "chord_tangent and suction_normal must both have shape "
            f"{expected_basis_shape}, got {tangent.shape} and {normal.shape}"
        )
    if not np.isfinite(orthogonality_tolerance) or orthogonality_tolerance <= 0.0:
        raise HiratoEquationError("orthogonality_tolerance must be positive and finite")
    tangent_norm = np.linalg.norm(tangent, axis=-1)
    normal_norm = np.linalg.norm(normal, axis=-1)
    dot = np.einsum("...i,...i->...", tangent, normal)
    if (
        np.any(np.abs(tangent_norm - 1.0) > orthogonality_tolerance)
        or np.any(np.abs(normal_norm - 1.0) > orthogonality_tolerance)
        or np.any(np.abs(dot) > orthogonality_tolerance)
    ):
        raise HiratoEquationError(
            "finite-wing chord_tangent and suction_normal must be orthonormal"
        )
    return displacement[..., 0, None] * tangent + displacement[..., 1, None] * normal


def tev_first_displacement_hirato(
    u_infinity,
    dt: float,
) -> np.ndarray:
    """Hirato dissertation TEV placement: ``delta_x_W = 0.3 U_inf dt``.

    Section 4.5.3 of Hirato's dissertation states that the new trailing-edge
    ring is placed along the swept trailing-edge track at 0.3 of one
    freestream translation step.  The journal article abbreviates the
    conventional TEV implementation, so this helper records the dissertation
    implementation identity explicitly instead of inheriting the historical
    FLUXV full-step placement.
    """
    velocity = _finite_array("u_infinity", u_infinity)
    if velocity.ndim < 1 or velocity.shape[-1] != 3:
        raise HiratoEquationError(
            f"u_infinity must end in a 3D vector, got {velocity.shape}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise HiratoEquationError(f"dt must be positive and finite, got {dt}")
    return 0.3 * velocity * dt


def rollup_displacement_eq24(
    local_velocity_now,
    local_velocity_previous,
    dt: float,
) -> np.ndarray:
    """Hirato Eq.24 two-level rollup displacement.

    The published finite-wing model advances a free vertex with the average
    of its current and previous local velocities.  This is not the
    current-velocity Euler update used by the first shadow topology test.
    """
    current = _finite_array("local_velocity_now", local_velocity_now)
    previous = _finite_array("local_velocity_previous", local_velocity_previous)
    if current.shape != previous.shape:
        raise HiratoEquationError(
            "local_velocity_now and local_velocity_previous must match, "
            f"got {current.shape} and {previous.shape}"
        )
    if current.ndim < 1 or current.shape[-1] != 3:
        raise HiratoEquationError(
            f"local velocities must end in a 3D vector, got {current.shape}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise HiratoEquationError(f"dt must be positive and finite, got {dt}")
    return 0.5 * (current + previous) * dt


def cutoff_radius_family(
    anticipated_smallest_ring_dimension: float,
    ratios=(0.10, 0.25, 0.49),
) -> np.ndarray:
    """Return an explicit Hirato Eq.25 sensitivity family with no default pick.

    Hirato specifies a *fixed singular radius* as a small fraction, typically
    less than 0.5, of the anticipated smallest vortex-ring dimension.  The
    ratios here are sensitivity coordinates, not fitted physical constants.
    """
    scale = float(anticipated_smallest_ring_dimension)
    ratio = _finite_array("ratios", ratios, ndim=1)
    if not np.isfinite(scale) or scale <= 0.0:
        raise HiratoEquationError(
            "anticipated_smallest_ring_dimension must be positive and finite"
        )
    if ratio.size == 0 or np.any((ratio <= 0.0) | (ratio >= 0.5)):
        raise HiratoEquationError("every cutoff ratio must lie strictly in (0, 0.5)")
    return scale * ratio


def lesp_eq6(
    gamma_front,
    u_infinity: float,
    chord,
    delta_x_front,
) -> np.ndarray:
    """Hirato Eq.6 using the actual forwardmost lattice extent.

    ``gamma_front`` is the circulation of the forwardmost bound-vortex ring
    for each spanwise strip.  ``delta_x_front`` must describe that same
    lattice: pairing ``Gamma_1`` with an unrelated fixed chord fraction
    changes the LESP observation operator.  The published equation uses the
    freestream speed ``U_infinity`` rather than a local relative-speed
    normalization.
    """
    gamma = _finite_array("gamma_front", gamma_front, ndim=1)
    local_chord = _finite_array("chord", chord, ndim=1)
    delta_x = _finite_array("delta_x_front", delta_x_front, ndim=1)
    if gamma.shape != local_chord.shape or gamma.shape != delta_x.shape:
        raise HiratoEquationError(
            "gamma_front, chord and delta_x_front must have matching shapes"
        )
    if not np.isfinite(u_infinity) or u_infinity <= 0.0:
        raise HiratoEquationError(
            f"u_infinity must be positive and finite, got {u_infinity}"
        )
    if np.any(local_chord <= 0.0):
        raise HiratoEquationError("chord must be positive")
    fraction = delta_x / local_chord
    if np.any((fraction <= 0.0) | (fraction > 1.0)):
        raise HiratoEquationError("delta_x_front/chord must lie in (0, 1]")
    theta = np.arccos(np.clip(1.0 - 2.0 * fraction, -1.0, 1.0))
    denominator = u_infinity * local_chord * (theta + np.sin(theta))
    return 1.13 * gamma / denominator


def lesp_sensitivity_eq6(
    bound_response,
    u_infinity: float,
    chord,
    delta_x_front,
) -> np.ndarray:
    """Map unit LEV strengths to Eq.6 LESP changes.

    ``bound_response[j, p]`` is the bound-circulation response at panel ``p``
    to a unit nascent-LEV strength on source strip ``j``.  Panels are flattened
    chord-major, so the first ``ns`` entries are the forwardmost rings used by
    Eq.6.  The result is shaped ``(target_strip, source_strip)``.
    """
    response = _finite_array("bound_response", bound_response, ndim=2)
    local_chord = _finite_array("chord", chord, ndim=1)
    delta_x = _finite_array("delta_x_front", delta_x_front, ndim=1)
    ns = local_chord.shape[0]
    if delta_x.shape != (ns,):
        raise HiratoEquationError(
            f"delta_x_front must have shape {(ns,)}, got {delta_x.shape}"
        )
    if response.shape[0] != ns or response.shape[1] < ns:
        raise HiratoEquationError(
            f"bound_response must have shape ({ns}, npan>=ns), got {response.shape}"
        )
    scale = lesp_eq6(
        np.ones(ns),
        u_infinity,
        local_chord,
        delta_x,
    )
    return scale[:, None] * response[:, :ns].T


def solve_lesp_constraint(
    a0_pre,
    sensitivity,
    active,
    lesp_crit: float,
) -> LespConstraintSolution:
    """Solve the affine Hirato LESP=LESPcrit condition without fitted damping.

    Only active source strips receive a new ``Gamma_L``.  Spanwise coupling to
    inactive target strips remains visible in ``a0_post``.  A singular active
    system fails explicitly; no diagonal fallback or unregistered
    regularization is applied.
    """
    a0 = _finite_array("a0_pre", a0_pre, ndim=1)
    matrix = _finite_array("sensitivity", sensitivity, ndim=2)
    mask = np.asarray(active, dtype=bool)
    ns = a0.shape[0]
    if matrix.shape != (ns, ns) or mask.shape != (ns,):
        raise HiratoEquationError(
            f"sensitivity and active must have shapes {(ns, ns)} and {(ns,)}"
        )
    if not np.isfinite(lesp_crit) or lesp_crit <= 0.0:
        raise HiratoEquationError(
            f"lesp_crit must be positive and finite, got {lesp_crit}"
        )
    gamma = np.zeros(ns)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return LespConstraintSolution(
            gamma_lev=gamma,
            a0_post=a0.copy(),
            active_residual=np.empty(0),
            condition_number=1.0,
        )
    submatrix = matrix[np.ix_(indices, indices)]
    condition = float(np.linalg.cond(submatrix))
    target = lesp_crit * np.sign(a0[indices])
    try:
        gamma[indices] = np.linalg.solve(
            submatrix,
            target - a0[indices],
        )
    except np.linalg.LinAlgError as error:
        raise HiratoEquationError(
            "active LESP sensitivity matrix is singular"
        ) from error
    post = a0 + matrix @ gamma
    residual = post[indices] - target
    return LespConstraintSolution(
        gamma_lev=gamma,
        a0_post=post,
        active_residual=residual,
        condition_number=condition,
    )


def rearmost_bound_aft_edge(
    bound_rings, nc: int, ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the aft-right/aft-left edge used by the Fig.5 pseudovortex.

    FLUXV rings are flattened chord-major and store corners as
    ``[front-left, front-right, aft-right, aft-left]``.  Hirato's current
    formulation aligns the pseudovortex aft edge with the *rearmost* bound
    ring, not the frontmost ring.
    """
    rings = _finite_array("bound_rings", bound_rings, ndim=3)
    if nc <= 0 or ns <= 0:
        raise HiratoEquationError("nc and ns must be positive")
    if rings.shape != (nc * ns, 4, 3):
        raise HiratoEquationError(
            f"bound_rings must have shape {(nc * ns, 4, 3)}, got {rings.shape}"
        )
    rear = rings[(nc - 1) * ns : nc * ns]
    return rear[:, 2].copy(), rear[:, 3].copy()


def tev_strength_eq9(prev_bound_rear, prev_lev) -> np.ndarray:
    """Hirato Eq.9: Gamma_W^n = Gamma_b,max^(n-1) + Gamma_L^(n-1)."""
    bound = _finite_array("prev_bound_rear", prev_bound_rear, ndim=1)
    lev = _finite_array("prev_lev", prev_lev, ndim=1)
    if bound.shape != lev.shape:
        raise HiratoEquationError(
            f"prev_bound_rear and prev_lev must match, got {bound.shape} and {lev.shape}"
        )
    return bound + lev


def kelvin_residual_eq9(tev_now, prev_bound_rear, prev_lev) -> np.ndarray:
    """Per-strip residual of Hirato Eq.9 in the paper's sign convention."""
    tev = _finite_array("tev_now", tev_now, ndim=1)
    target = tev_strength_eq9(prev_bound_rear, prev_lev)
    if tev.shape != target.shape:
        raise HiratoEquationError(
            f"tev_now and Eq.9 target must match, got {tev.shape} and {target.shape}"
        )
    return tev - target


@dataclass(frozen=True)
class PotentialRate:
    """Separated terms in Hirato Eq.17, shaped ``(nc, ns)``."""

    bound: np.ndarray
    lev: np.ndarray
    total: np.ndarray


def potential_rate_eq17(
    bound_now,
    bound_prev,
    lev_now,
    lev_prev,
    active,
    dt: float,
) -> PotentialRate:
    """Evaluate the bound and LEV parts of the Eq.17 potential-jump rate.

    During active LEV shedding, Eq.16 gives
    ``phi_u-phi_l = Gamma_L + Gamma_b,x``.  Hence Eq.17 adds the same
    stripwise ``dGamma_L/dt`` to each chordwise evaluation location.  When a
    strip is inactive, the LEV term is zero as stated immediately after Eq.17.
    """
    current = _finite_array("bound_now", bound_now, ndim=2)
    previous = _finite_array("bound_prev", bound_prev, ndim=2)
    lev_n = _finite_array("lev_now", lev_now, ndim=1)
    lev_p = _finite_array("lev_prev", lev_prev, ndim=1)
    mask = np.asarray(active, dtype=bool)
    if current.shape != previous.shape:
        raise HiratoEquationError(
            f"bound_now and bound_prev must match, got {current.shape} and {previous.shape}"
        )
    _, ns = current.shape
    if lev_n.shape != (ns,) or lev_p.shape != (ns,) or mask.shape != (ns,):
        raise HiratoEquationError(
            f"lev_now, lev_prev and active must have shape {(ns,)}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise HiratoEquationError(f"dt must be positive and finite, got {dt}")

    bound_rate = (current - previous) / dt
    strip_lev_rate = np.where(mask, (lev_n - lev_p) / dt, 0.0)
    lev_rate = np.broadcast_to(strip_lev_rate, current.shape).copy()
    return PotentialRate(
        bound=bound_rate,
        lev=lev_rate,
        total=bound_rate + lev_rate,
    )


def preconstraint_shed_mask(a0_pre, lesp_crit: float) -> np.ndarray:
    """Return the Hirato shedding event from the pre-constraint LESP.

    The event cannot be reconstructed from the post-solve LESP because the
    implicit constraint intentionally drives active strips back to the
    threshold.
    """
    a0 = _finite_array("a0_pre", a0_pre, ndim=1)
    if not np.isfinite(lesp_crit) or lesp_crit <= 0.0:
        raise HiratoEquationError(
            f"lesp_crit must be positive and finite, got {lesp_crit}"
        )
    return np.abs(a0) > lesp_crit
