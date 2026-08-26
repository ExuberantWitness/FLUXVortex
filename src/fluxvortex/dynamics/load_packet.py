"""GeneralizedLoadPacket: body wrench + joint torques + elastic forces from surface loads.

Implements plan §5.8 and §7.3: the SAME surface forces project via the
kinematic Jacobian transpose into body/joint/elastic generalized loads,
ensuring work conjugacy (surface power = sum of generalized powers).

The packet is the single load boundary between the aerodynamic solver and
the dynamic subsystems.  Producing the body wrench, the joint torques and
the Q16 elastic generalized forces from three independent force
evaluations would double-count pressure; here ONE projection feeds all
consumers, and the work-conjugacy audit

    |f_s . v_s - Q_g . qd_g| / max(|f_s . v_s|, |Q_g . qd_g|, eps) <= eps_work

is evaluated at construction time and carried on the packet as
``relative_work_error``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

WORK_AUDIT_EPS = 1.0e-30


@dataclass(frozen=True)
class SurfaceLoad:
    """Per-surface aerodynamic load, expressed in the inertial frame I.

    ``total_force_I``/``total_moment_I`` are the resultant about the
    world ORIGIN.  ``panel_points_I``/``panel_velocities_I`` (the force
    application points and their velocities) are optional bookkeeping
    kept from the surface frame; they make the work-conjugacy audit
    exact because the surface power is summed panel by panel at the very
    points where the forces act.
    """

    surface_id: str
    panel_forces_I: torch.Tensor                     # (n_panels, 3)
    total_force_I: torch.Tensor                      # (3,)
    total_moment_I: torch.Tensor                     # (3,) about the origin
    panel_points_I: torch.Tensor | None = None       # (n_panels, 3)
    panel_velocities_I: torch.Tensor | None = None   # (n_panels, 3)

    @classmethod
    def from_panels(
        cls,
        surface_id: str,
        panel_forces_I: torch.Tensor,
        panel_points_I: torch.Tensor,
        panel_velocities_I: torch.Tensor | None = None,
    ) -> "SurfaceLoad":
        """Build from raw panel data; totals are reduced on the same tensors.

        The moment is taken about the world origin: M = sum r_p x f_p.
        """
        if panel_forces_I.ndim != 2 or panel_forces_I.shape[1] != 3:
            raise ValueError("panel_forces_I must have shape (n_panels, 3)")
        if panel_points_I.shape != panel_forces_I.shape:
            raise ValueError("panel_points_I must match panel_forces_I shape")
        total_force = panel_forces_I.sum(dim=0)
        total_moment = torch.cross(panel_points_I, panel_forces_I, dim=-1).sum(dim=0)
        return cls(
            surface_id=surface_id,
            panel_forces_I=panel_forces_I,
            total_force_I=total_force,
            total_moment_I=total_moment,
            panel_points_I=panel_points_I,
            panel_velocities_I=panel_velocities_I,
        )

    def validate(self) -> None:
        if self.panel_forces_I.ndim != 2 or self.panel_forces_I.shape[1] != 3:
            raise ValueError("panel_forces_I must have shape (n_panels, 3)")
        for name, tensor in (
            ("total_force_I", self.total_force_I),
            ("total_moment_I", self.total_moment_I),
        ):
            if tensor.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
        if bool((~torch.isfinite(self.panel_forces_I)).any()):
            raise ValueError("panel forces non-finite")


def project_surface_loads_to_body(
    surface_loads: tuple[SurfaceLoad, ...],
    body_position_I: torch.Tensor,
    body_id: str = "body_0",
) -> dict[str, dict[str, torch.Tensor]]:
    """Project surface loads into a body wrench about the body COM.

    F_body = sum F_s
    M_body = M_origin - r_body x F_total    (moment shifted from the
    world origin to the body center of mass, still expressed in frame I)

    This is the rigid block of the J^T f projection (plan §7.3): the
    same surface forces that generate the elastic/joint loads produce
    exactly this resultant, never a second pressure evaluation.
    """
    device = body_position_I.device
    dtype = body_position_I.dtype if body_position_I.dtype.is_floating_point else torch.float64
    total_force = torch.zeros(3, device=device, dtype=dtype)
    total_moment = torch.zeros(3, device=device, dtype=dtype)
    for load in surface_loads:
        total_force = total_force + load.total_force_I.to(device=device, dtype=dtype)
        # Accumulate the origin moment; the shift to the body COM happens
        # once below so it applies to the summed force exactly.
        total_moment = total_moment + load.total_moment_I.to(device=device, dtype=dtype)
    # Shift moment from origin to body COM: M_body = M_origin - r_body x F_total
    total_moment = total_moment - torch.cross(
        body_position_I.to(dtype=dtype), total_force, dim=0)
    return {body_id: {"force_I": total_force, "moment_I": total_moment}}


def _tensor(value: Any) -> torch.Tensor | None:
    return value if isinstance(value, torch.Tensor) else None


def _extract_body_state(dynamic_state: Any):
    """Pull (body_id, RigidBodyState-like) pairs out of a dynamic state.

    Accepts a ``WorldDynamicState``-like container (``body_states``) or a
    bare RigidBodyState-like object.
    """
    body_states = getattr(dynamic_state, "body_states", None)
    if body_states:
        return dict(body_states)
    if hasattr(dynamic_state, "position_I"):
        return {"body_0": dynamic_state}
    return {}


def _dot(a: Any, b: Any) -> float:
    """Scalar product of two generalized-force/rate vectors (torch or wp.array)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return float((a.reshape(-1) * b.reshape(-1)).sum().item())
    # Warp arrays (production Q16 states): reduce through numpy.
    import numpy as np

    def flat(x):
        return np.asarray(x.numpy() if hasattr(x, "numpy") else x).reshape(-1)

    return float(np.dot(flat(a), flat(b)))


def build_generalized_load_packet(
    surface_loads: tuple[SurfaceLoad, ...],
    dynamic_state: Any,
    *,
    joint_torques: dict[str, torch.Tensor] | None = None,
    elastic_forces: dict[str, Any] | None = None,
    audit_velocities: bool = True,
) -> "GeneralizedLoadPacket":
    """One J^T f projection shared by body/joint/elastic (plan §7.3).

    Parameters
    ----------
    surface_loads:
        Per-surface resultants (and panel data for the audit).
    dynamic_state:
        The CURRENT (guess) dynamic state carrying body twists, joint
        rates and elastic rates used for the power audit.
    joint_torques / elastic_forces:
        The generalized blocks of the projection.  Joint torques come
        from the joint Jacobian transpose; elastic forces from the
        production panel-load transfer (e.g. Q16NativePanelLoadTransfer).

    The work-conjugacy audit sums panel power f_p . v_p at the exact
    force application points and compares against the generalized power
    F.v_com + M_com.omega + tau_J.qd_J + Q_e.qd_e.  Loads without panel
    velocities make the audit impossible; they are rejected fail-closed
    unless ``audit_velocities`` is explicitly disabled.
    """
    joint_torques = dict(joint_torques or {})
    elastic_forces = dict(elastic_forces or {})

    bodies = _extract_body_state(dynamic_state)
    body_wrenches_I: dict[str, dict[str, torch.Tensor]] = {}
    for body_id, body_state in bodies.items():
        body_wrenches_I.update(
            project_surface_loads_to_body(
                surface_loads, body_state.position_I, body_id=body_id))

    # ------------------------------------------------------------------
    # Work-conjugacy audit (plan §5.8 acceptance).
    # ------------------------------------------------------------------
    surface_power = 0.0
    if audit_velocities and surface_loads:
        for load in surface_loads:
            if load.panel_velocities_I is None:
                raise ValueError(
                    f"surface load '{load.surface_id}' carries no panel "
                    "velocities: the work-conjugacy audit would silently "
                    "under-count the surface power (plan 5.8)")
            forces = load.panel_forces_I.reshape(-1, 3)
            velocities = load.panel_velocities_I.reshape(-1, 3).to(
                device=forces.device, dtype=forces.dtype)
            surface_power += float(
                (forces * velocities).sum(dim=-1).sum().item())

    generalized_power = 0.0
    for body_id, body_state in bodies.items():
        wrench = body_wrenches_I.get(body_id)
        if wrench is None:
            continue
        from .rigid_body_se3 import RigidBodySE3Dynamics

        R_IB = RigidBodySE3Dynamics().rotation_matrix(body_state.quaternion_IB)
        omega_I = R_IB @ body_state.angular_velocity_B
        generalized_power += _dot(wrench["force_I"], body_state.linear_velocity_I)
        generalized_power += _dot(wrench["moment_I"], omega_I)

    joint_states = getattr(dynamic_state, "joint_states", None) or {}
    for joint_id, torque in joint_torques.items():
        joint_state = joint_states.get(joint_id)
        if joint_state is None or getattr(joint_state, "rates", None) is None:
            continue
        generalized_power += _dot(torque, joint_state.rates)

    elastic_states = getattr(dynamic_state, "elastic_states", None) or {}
    for elastic_id, force in elastic_forces.items():
        elastic_state = elastic_states.get(elastic_id)
        if elastic_state is None:
            continue
        rate = getattr(elastic_state, "qd", None)
        if rate is None:
            continue
        generalized_power += _dot(force, rate)

    scale = max(abs(surface_power), abs(generalized_power))
    relative_error = (
        abs(surface_power - generalized_power) / max(scale, WORK_AUDIT_EPS)
        if scale > WORK_AUDIT_EPS else 0.0)

    return GeneralizedLoadPacket(
        body_wrenches_I=body_wrenches_I,
        joint_torques=joint_torques,
        elastic_forces=elastic_forces,
        surface_power=surface_power,
        generalized_power=generalized_power,
        relative_work_error=relative_error,
    )


# Backwards/forwards-friendly alias: the coupling layer calls this name.
project_surface_loads = build_generalized_load_packet


@dataclass(frozen=True)
class GeneralizedLoadPacket:
    """Unified load boundary between aero and dynamics (plan §5.8).

    ``body_wrenches_I``: body_id -> {"force_I": (3,), "moment_I": (3,)}
    about the body COM, expressed in the inertial frame.
    ``joint_torques``: joint_id -> generalized torque (J_J^T f).
    ``elastic_forces``: elastic_id -> generalized force (wp.array or
    torch.Tensor) from the SAME surface forces.

    The audit fields enforce work conjugacy: surface power (f_s . v_s
    summed at the application points) must equal the generalized power
    (Q_g . qd_g summed over all generalized coordinates).
    """

    body_wrenches_I: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    joint_torques: dict[str, torch.Tensor] = field(default_factory=dict)
    elastic_forces: dict[str, Any] = field(default_factory=dict)
    surface_power: float = 0.0            # sum(f_s . v_s)
    generalized_power: float = 0.0        # sum(Q_g . qd_g)
    relative_work_error: float = 0.0      # |surface - generalized| / max(...)

    def audit_passes(self, tolerance: float = 1.0e-10) -> bool:
        return self.relative_work_error <= tolerance

    @staticmethod
    def zero_load(device: str = "cpu") -> "GeneralizedLoadPacket":
        """A packet with no surface loads: zero wrench, zero power, zero error."""
        zero = torch.zeros(3, device=device, dtype=torch.float64)
        return GeneralizedLoadPacket(
            body_wrenches_I={"body_0": {"force_I": zero.clone(), "moment_I": zero.clone()}},
        )
