"""Analytic rigid kinematics for the Izraelevitz/Scherer rectangular wing.

H2 (HANDOFF_IZRAELEVITZ2017_FIG14_SCHERER1968_20260827 §2.1/§2.3/§8): the
wing is a flat rectangular AR=3 finite wing (zero-camber NACA 63A015
mid-surface), full span from -b/2 to +b/2 with two free tips, heaving and
pitching about x/c = 0.75 in a forward water flow.  This module generates
the complete SurfaceFrame (positions + velocities + normals) at each time
step from the exact analytic motion law

    z(t)     = h         * cos(omega t)
    theta(t) = theta_max * cos(omega t + psi)   (pitch about x/c = 0.75)

with every velocity the analytic time derivative — no sine-phase adapters,
no stale velocities, and no prediction/experiment phase fitting.

Frame conventions
-----------------
x: chordwise (LE -> TE), y: spanwise (-b/2 -> +b/2), z: heave normal.
The reference geometry is expressed with its origin AT THE PITCH AXIS, so
``PrescribedRigidSurfaceKinematics`` (which rotates about the body origin)
implements the 0.75c pivot exactly: the reference LE sits at x = -0.75c and
the TE at x = +0.25c, the motion-law position is the pivot's world position
(0, 0, z(t)), and pitch is a rotation about the body y axis.  Panel rings
follow the native V5M layout (``assemble_native_ring_geometry``): quarter
lines at (i + 0.25) dx with the rear row one panel behind the last quarter
line, rings ordered reshape(nc, ns, 4, 3) with panel index i * ns + j.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from ..kinematics.frames import SurfaceFrame
from ..kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics


def build_izra_reference_grid(
    *,
    chordwise_panels: int = 8,
    spanwise_panels: int = 24,
    chord_m: float = 0.1016,
    span_m: float = 0.3048,
    pivot_fraction_chord: float = 0.75,
    device: str = "cuda:0",
) -> SurfaceFrame:
    """Build the reference (undeformed) SurfaceFrame for the flat wing.

    The returned frame is the full wing at zero heave and zero pitch, with
    its origin on the pitch axis (x/c = ``pivot_fraction_chord``), y spanning
    the complete [-b/2, +b/2] with two free tips, and zero velocity everywhere.
    Rings/collocation/normals/areas come from the shared native V5M ring
    assembly, so the rigid reference layout is by construction the layout
    ``Q16NativeV5MSurface.evaluate`` would produce for a flat mesh.
    """
    nc = int(chordwise_panels)
    ns = int(spanwise_panels)
    if nc < 1 or ns < 1:
        raise ValueError("chordwise/spanwise panel counts must be positive")
    chord_m = float(chord_m)
    span_m = float(span_m)
    pivot_x = float(pivot_fraction_chord) * chord_m
    dx = chord_m / nc
    dy = span_m / ns
    dtype = torch.float64

    span_nodes = torch.tensor(
        [-0.5 * span_m + j * dy for j in range(ns + 1)],
        device=device,
        dtype=dtype,
    )

    def line(x_ref: float) -> torch.Tensor:
        zeros = torch.zeros(ns + 1, device=device, dtype=dtype)
        return torch.stack(
            (torch.full((ns + 1,), x_ref, device=device, dtype=dtype), span_nodes, zeros),
            dim=-1,
        )

    # Quarter-chord lines (nc, ns+1, 3) at x_ref = (i + 0.25) dx - pivot_x.
    quarter = torch.stack(
        [line((i + 0.25) * dx - pivot_x) for i in range(nc)], dim=0
    )
    quarter_velocity = torch.zeros_like(quarter)
    leading = line(-pivot_x)
    trailing = line(chord_m - pivot_x)
    leading_velocity = torch.zeros_like(leading)
    trailing_velocity = torch.zeros_like(trailing)

    # Deferred import: the helper lives in the (GPU-stack) native V5M module.
    from ..warp_fsi.q16_flux_v5m_native import assemble_native_ring_geometry

    rings, ring_velocity, collocation, collocation_velocity, normals, areas = (
        assemble_native_ring_geometry(
            quarter,
            quarter_velocity,
            trailing,
            trailing_velocity,
            panel_count=nc * ns,
        )
    )

    return SurfaceFrame(
        surface_id="izra_wing",
        body_id="body_0",
        panel_rings_I=rings,
        panel_ring_velocity_I=ring_velocity,
        collocation_I=collocation,
        collocation_velocity_I=collocation_velocity,
        normals_I=normals,
        areas=areas,
        leading_edge_I=leading,
        trailing_edge_I=trailing,
        leading_velocity_I=leading_velocity,
        trailing_velocity_I=trailing_velocity,
        chordwise_panels=nc,
        spanwise_panels=ns,
        topology_digest=f"izra-flat:{nc}x{ns}:pivot{pivot_fraction_chord:g}",
    )


def izra_motion_law(
    config, *, device: str = "cuda:0"
) -> Callable[[float], tuple]:
    """Return the analytic motion law for one Izraelevitz condition.

    The callable maps ``time`` -> ``(position (3,), quaternion (4,),
    linear_velocity (3,), angular_velocity (3,))`` for the WING REFERENCE
    FRAME whose origin is the pitch-axis mid-span point.  Heave translates
    that point in z; pitch rotates about the body y axis (positive theta =
    LE up = positive angle of attack against the +x freestream).
    """
    h = float(config.heave_amplitude_m)
    omega = float(config.omega_rad_s)
    theta_max = math.radians(float(config.theta_max_deg))
    psi = math.radians(float(config.phase_offset_deg))
    dtype = torch.float64

    def law(t: float):
        phase = omega * float(t)
        z = h * math.cos(phase)
        zdot = -h * omega * math.sin(phase)
        theta = theta_max * math.cos(phase + psi)
        thetadot = -theta_max * omega * math.sin(phase + psi)
        half = 0.5 * theta
        pos = torch.tensor([0.0, 0.0, z], device=device, dtype=dtype)
        # Quaternion [w, x, y, z] for rotation about +y by theta.
        quat = torch.tensor(
            [math.cos(half), 0.0, math.sin(half), 0.0], device=device, dtype=dtype
        )
        lin_vel = torch.tensor([0.0, 0.0, zdot], device=device, dtype=dtype)
        ang_vel = torch.tensor([0.0, thetadot, 0.0], device=device, dtype=dtype)
        return pos, quat, lin_vel, ang_vel

    return law


class IzraRigidSurfaceKinematics:
    """Evaluates the complete Izraelevitz wing SurfaceFrame at each time step.

    Combines the pivot-centered flat-wing reference grid with the exact
    analytic heave+pitch law through ``PrescribedRigidSurfaceKinematics``,
    so positions, panel-node velocities, LE/TE velocities and rotated normals
    all come from one rigid transform — the arm is always the rotated
    reference point, and areas are preserved exactly.
    """

    def __init__(
        self,
        config,
        *,
        chordwise_panels: int = 8,
        spanwise_panels: int = 24,
        device: str = "cuda:0",
    ):
        self.config = config
        self.device = device
        self.reference_frame = build_izra_reference_grid(
            chordwise_panels=chordwise_panels,
            spanwise_panels=spanwise_panels,
            chord_m=float(config.chord_m),
            span_m=float(config.span_m),
            pivot_fraction_chord=float(config.pivot_fraction_chord),
            device=device,
        )
        self._kin = PrescribedRigidSurfaceKinematics(
            self.reference_frame,
            izra_motion_law(config, device=device),
            surface_id="izra_wing",
            body_id="body_0",
        )

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return self._kin.surface_ids

    def evaluate(self, time: float) -> SurfaceFrame:
        return self._kin.evaluate(time)


__all__ = [
    "IzraRigidSurfaceKinematics",
    "build_izra_reference_grid",
    "izra_motion_law",
]
