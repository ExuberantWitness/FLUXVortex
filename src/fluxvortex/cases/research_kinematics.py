"""Kinematics of the published cases reproduced in note A24.

Two kinds of motion, both handed to the solver as a ``SurfaceFrame``:

* a single rigid rectangular wing on the Izraelevitz reference grid
  (``build_izra_reference_grid``, at any chord, span and pivot): harmonic heave and pitch
  (``heave_pitch_law``) or a pitch ramp that is piecewise linear in convective time
  (``pitch_ramp_law``), through ``PrescribedRigidSurfaceKinematics``;
* a wing of two rigid halves hinged on its root chord line (``HingedFlapKinematics``): the
  root flapping of Kumar et al. (arXiv 2309.14832; Sadhana 51:103, 2026), delta(t) = delta0 +
  delta1 sin(omega t) at a fixed geometric angle of attack alpha.

The conventions are the solver's: x along the freestream, z up, and a positive pitch about +y
puts the leading edge up.  A motion law maps the time to ``(position (3,), quaternion (4,)
[w, x, y, z], linear velocity (3,), angular velocity (3,))`` in world axes, which is what
``PrescribedRigidSurfaceKinematics`` consumes.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

from ..kinematics.frames import SurfaceFrame
from ..kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from .izraelevitz_kinematics import build_izra_reference_grid

MotionLaw = Callable[[float], tuple]


def heave_pitch_law(
    *,
    omega_rad_s: float,
    heave_amplitude_m: float = 0.0,
    pitch_amplitude_rad: float = 0.0,
    pitch_offset_rad: float = 0.0,
    phase_pitch_rad: float = 0.0,
    phase_heave_rad: float = 0.0,
    device: str = "cuda:0",
) -> MotionLaw:
    """z = h cos(omega t + phase_heave), theta = theta0 + theta1 cos(omega t + phase_pitch).

    With both phases zero this is the Izraelevitz law.  The phases give the other published
    waveforms: Li et al. (JFM 972, A30) put the plate at -h cos(omega t) with
    alpha_geo = alpha_max sin(omega t), which is phase_heave = pi and phase_pitch = -pi / 2.
    """

    h = float(heave_amplitude_m)
    omega = float(omega_rad_s)
    theta1 = float(pitch_amplitude_rad)
    theta0 = float(pitch_offset_rad)
    phase_pitch = float(phase_pitch_rad)
    phase_heave = float(phase_heave_rad)
    dtype = torch.float64

    def law(t: float):
        time = float(t)
        heave_phase = omega * time + phase_heave
        z = h * math.cos(heave_phase)
        zdot = -h * omega * math.sin(heave_phase)
        theta = theta0 + theta1 * math.cos(omega * time + phase_pitch)
        thetadot = -theta1 * omega * math.sin(omega * time + phase_pitch)
        half = 0.5 * theta
        pos = torch.tensor([0.0, 0.0, z], device=device, dtype=dtype)
        # Quaternion [w, x, y, z] for a rotation about +y by theta.
        quat = torch.tensor(
            [math.cos(half), 0.0, math.sin(half), 0.0], device=device, dtype=dtype
        )
        lin_vel = torch.tensor([0.0, 0.0, zdot], device=device, dtype=dtype)
        ang_vel = torch.tensor([0.0, thetadot, 0.0], device=device, dtype=dtype)
        return pos, quat, lin_vel, ang_vel

    return law


def pitch_profile_law(
    *, profile: Callable[[float], tuple[float, float]], speed: float, chord: float,
    device: str = "cuda:0",
) -> MotionLaw:
    """Pure pitch about +y from a profile in convective time.

    ``profile(t_star)`` returns the angle in radians and its derivative d(alpha)/d(t*) for
    ``t_star = U t / c``.  The position does not move.
    """

    rate_over_speed = float(speed) / float(chord)
    dtype = torch.float64

    def law(t: float):
        theta, rate = profile(float(t) * rate_over_speed)
        thetadot = rate * rate_over_speed
        half = 0.5 * theta
        pos = torch.zeros(3, device=device, dtype=dtype)
        quat = torch.tensor(
            [math.cos(half), 0.0, math.sin(half), 0.0], device=device, dtype=dtype
        )
        return pos, quat, torch.zeros(3, device=device, dtype=dtype), torch.tensor(
            [0.0, thetadot, 0.0], device=device, dtype=dtype
        )

    return law


def piecewise_linear_profile(knots: Sequence[tuple[float, float]]) -> Callable[[float], tuple[float, float]]:
    """(t_star, angle_deg) knots, increasing: the angle and its slope, held outside the knots.

    The slope between two knots is the pitch rate: K = d(alpha)/d(t*) / 2 in the convention of
    the papers.
    """

    times = [float(pair[0]) for pair in knots]
    angles = [math.radians(float(pair[1])) for pair in knots]
    if len(times) < 2 or any(b <= a for a, b in zip(times[:-1], times[1:])):
        raise ValueError("knots need at least two entries with increasing convective time")

    def profile(star: float) -> tuple[float, float]:
        if star <= times[0]:
            return angles[0], 0.0
        if star >= times[-1]:
            return angles[-1], 0.0
        index = max(i for i in range(len(times) - 1) if times[i] <= star)
        slope = (angles[index + 1] - angles[index]) / (times[index + 1] - times[index])
        return angles[index] + slope * (star - times[index]), slope

    return profile


def smoothed_ramp_profile(
    *, rate: float, smoothing: float, start: float, maximum_deg: float, hold: float
) -> Callable[[float], tuple[float, float]]:
    """Ramp, hold and return, smoothed at the corners, in convective time.

    alpha(t*) = (K / a) ln[ cosh(a(t* - t1)) cosh(a(t* - t4)) / (cosh(a(t* - t2)) cosh(a(t* - t3))) ]
    with t2 = t1 + alpha_max / (2K), t3 = t2 + hold and t4 = t3 + alpha_max / (2K): the
    reduced pitch rate K = d(alpha)/d(t*) / 2 far from the corners, the angle alpha_max
    through the hold, zero before and after.  Martinez-Carmena et al. print the same motion
    with (t1, t3) over (t2, t4); with their own t3 = t2 + t_hold that is a second ramp where
    the hold should be, and their figures show the hold, so the form here is the one with the
    hold (note A24).
    """

    k, a = float(rate), float(smoothing)
    peak = math.radians(float(maximum_deg))
    t1 = float(start)
    t2 = t1 + peak / (2.0 * k)
    t3 = t2 + float(hold)
    t4 = t3 + peak / (2.0 * k)

    def profile(star: float) -> tuple[float, float]:
        def term(offset: float) -> float:
            argument = a * (star - offset)
            # log(cosh) without overflow
            return abs(argument) + math.log1p(math.exp(-2.0 * abs(argument))) - math.log(2.0)

        tanh_sum = (
            math.tanh(a * (star - t1)) + math.tanh(a * (star - t4))
            - math.tanh(a * (star - t2)) - math.tanh(a * (star - t3))
        )
        return (k / a) * (term(t1) + term(t4) - term(t2) - term(t3)), k * tanh_sum

    return profile


def pitch_ramp_law(
    *,
    knots: Sequence[tuple[float, float]],
    speed: float,
    chord: float,
    device: str = "cuda:0",
) -> MotionLaw:
    """Pitch about +y, piecewise linear in convective time, held outside the knots."""

    return pitch_profile_law(
        profile=piecewise_linear_profile(knots), speed=speed, chord=chord, device=device
    )


def rigid_case_kinematics(
    *,
    law: MotionLaw,
    chord_m: float,
    span_m: float,
    pivot_fraction_chord: float,
    chordwise_panels: int = 8,
    spanwise_panels: int = 24,
    device: str = "cuda:0",
    surface_id: str = "paper_wing",
    body_id: str = "body_0",
) -> PrescribedRigidSurfaceKinematics:
    """One rigid flat rectangular wing of any chord, span and pivot, moved by ``law``."""

    reference = build_izra_reference_grid(
        chordwise_panels=chordwise_panels,
        spanwise_panels=spanwise_panels,
        chord_m=chord_m,
        span_m=span_m,
        pivot_fraction_chord=pivot_fraction_chord,
        device=device,
    )
    return PrescribedRigidSurfaceKinematics(
        reference, law, surface_id=surface_id, body_id=body_id
    )


class HingedFlapKinematics:
    """A flat wing of two rigid halves, each turning about the root chord line.

    The wing spans ``[-half_span, +half_span]`` with its leading edge at x = 0 and its root
    leading edge for origin.  Every point (X, y, 0) of the right half (y > 0) goes to
    R_y(alpha) R_x(+delta) (X, y, 0) and every point of the left half to
    R_y(alpha) R_x(-delta) (X, y, 0): both tips rise together, the centreline node line stays
    on the pitched root chord, and the two halves are mirror images.  The velocity is the
    derivative of that, s delta_dot (h x r) with h the pitched chord direction and s the sign
    of y.

    Why the flap comes before the pitch.  The paper's eqs. (17)-(18) pitch the wing about y
    and then turn it about the x axis.  Taken in that order on two mirrored halves, each root
    chord line leaves the centre plane by x sin(alpha) sin(delta) in opposite directions, the
    halves cross at the root, and the overlapping wedge is 0.17 c wide at the trailing edge for
    alpha = 20 deg and delta = 30 deg (0.017 c at alpha = 2 deg).  With the hinge on the
    pitched chord line there is no such wedge.  The two orders differ in the static part of the
    load, which carries cos(delta) here and does not there; that is small against the part the
    flap rate produces (note A24).

    The lattice is the native one (``assemble_native_ring_geometry``): quarter lines at
    (i + 0.25) dx, the rear row a quarter panel behind the geometric trailing edge.
    """

    def __init__(
        self,
        *,
        chord_m: float,
        half_span_m: float,
        alpha_deg: float,
        delta1_deg: float,
        omega_rad_s: float,
        delta0_deg: float = 0.0,
        start_ramp_s: float = 0.0,
        chordwise_panels: int = 8,
        spanwise_panels: int = 34,
        device: str = "cuda:0",
        surface_id: str = "hinged_wing",
        body_id: str = "body_0",
    ) -> None:
        if int(spanwise_panels) % 2:
            raise ValueError(
                "the centreline has to be a node line: an even number of spanwise panels"
            )
        nc, ns = int(chordwise_panels), int(spanwise_panels)
        chord, half_span = float(chord_m), float(half_span_m)
        dx, dy = chord / nc, 2.0 * half_span / ns
        dtype = torch.float64
        # node lines: the geometric leading edge, the quarter lines, the geometric trailing edge
        x_lines = [0.0] + [(i + 0.25) * dx for i in range(nc)] + [chord]
        y_nodes = [-half_span + j * dy for j in range(ns + 1)]
        y = torch.tensor(y_nodes, device=device, dtype=dtype)
        self._x = torch.tensor(x_lines, device=device, dtype=dtype).reshape(-1, 1, 1)
        self._y = y.reshape(1, -1, 1)
        self._abs_y = y.abs().reshape(1, -1, 1)
        self._side = torch.sign(y).reshape(1, -1, 1)
        self._alpha = math.radians(float(alpha_deg))
        self._delta0 = math.radians(float(delta0_deg))
        self._delta1 = math.radians(float(delta1_deg))
        self._omega = float(omega_rad_s)
        # delta(t) = r(t) (delta0 + delta1 sin(omega t)) with r rising from 0 to 1 over this
        # time by a half cosine.  The paper's law starts at its full flap rate: the first step
        # is then an impulsive start, and the added-mass force of that one step is -34.8 N on
        # this wing (CL -16 against a cycle peak near 1), which the wake then carries for a few
        # chords.  With the ramp the motion reaches the paper's law after one period (note A24).
        self._ramp = float(start_ramp_s)
        self._nc, self._ns = nc, ns
        self._device = torch.device(device)
        self._surface_id, self._body_id = surface_id, body_id
        self.topology_digest = f"hinged-flat:{nc}x{ns}:alpha{float(alpha_deg):g}"

    def delta(self, time: float) -> tuple[float, float]:
        """The flap angle and its rate at ``time``, in radians."""

        time = float(time)
        angle = self._delta0 + self._delta1 * math.sin(self._omega * time)
        rate = self._delta1 * self._omega * math.cos(self._omega * time)
        if self._ramp > 0.0 and time < self._ramp:
            phase = math.pi * time / self._ramp
            factor, factor_rate = 0.5 * (1.0 - math.cos(phase)), 0.5 * math.pi / self._ramp * math.sin(phase)
            return factor * angle, factor_rate * angle + factor * rate
        return angle, rate

    def evaluate(self, time: float) -> SurfaceFrame:
        from ..warp_fsi.q16_flux_v5m_native import assemble_native_ring_geometry

        delta, rate = self.delta(time)
        cos_d, sin_d = math.cos(delta), math.sin(delta)
        cos_a, sin_a = math.cos(self._alpha), math.sin(self._alpha)
        flapped_y = self._y * cos_d                               # (1, ns+1, 1)
        flapped_z = self._abs_y * sin_d
        span_shape = (self._x.shape[0], flapped_y.shape[1], 1)
        x = self._x * cos_a + flapped_z * sin_a                   # (nc+2, ns+1, 1)
        y = flapped_y.expand(span_shape)
        z = -self._x * sin_a + flapped_z * cos_a
        position = torch.cat((x, y, z), dim=2)                    # (nc+2, ns+1, 3)
        velocity = self._side * rate * torch.cat(
            (sin_a * y, -sin_a * x - cos_a * z, cos_a * y), dim=2
        )
        leading, leading_velocity = position[0], velocity[0]
        rings, ring_velocity, collocation, collocation_velocity, normals, areas = (
            assemble_native_ring_geometry(
                position[1:-1],
                velocity[1:-1],
                position[-1],
                velocity[-1],
                panel_count=self._nc * self._ns,
            )
        )
        return SurfaceFrame(
            surface_id=self._surface_id,
            body_id=self._body_id,
            panel_rings_I=rings,
            panel_ring_velocity_I=ring_velocity,
            collocation_I=collocation,
            collocation_velocity_I=collocation_velocity,
            normals_I=normals,
            areas=areas,
            leading_edge_I=leading,
            trailing_edge_I=position[-1],
            leading_velocity_I=leading_velocity,
            trailing_velocity_I=velocity[-1],
            chordwise_panels=self._nc,
            spanwise_panels=self._ns,
            topology_digest=self.topology_digest,
        )


__all__ = [
    "HingedFlapKinematics",
    "MotionLaw",
    "heave_pitch_law",
    "piecewise_linear_profile",
    "pitch_profile_law",
    "pitch_ramp_law",
    "rigid_case_kinematics",
    "smoothed_ramp_profile",
]
