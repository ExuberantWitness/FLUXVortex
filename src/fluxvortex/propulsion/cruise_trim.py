"""3D GPU UVLM periodic cruise-trim and morphing-wing benchmark solvers.

Provides:
- ``MorphingFlappingWingKinematics``: tapered, swept, cambered, aeroelastically
  twisted 3D flapping-wing (and morphing fixed-wing) kinematics producing exact
  CUDA float64 ``SurfaceFrame`` states with analytical node velocities.
- ``evaluate_flapping_operating_point``: runs FLUXVortex's 3D GPU UVLM solver
  with live Hirato v2 LEV + free TE wake + Squire viscous core, extracts cycle-
  averaged lift, thrust, viscous profile drag, and exact aerodynamic virtual
  work ``P_aero(t) = -sum_p F_p . v_p``, and evaluates 1st-order BLDC motor
  electrical power.
- ``solve_flapping_cruise_trim``: 2D damped Newton/Broyden trim solver over
  ``(alpha_deg, frequency_hz)`` satisfying steady-level flight ``L = m g`` and
  ``T_wing = D_body_tail``.
- ``solve_fixed_wing_cruise_trim``: trims ``alpha_deg`` for steady-level flight
  on morphing fixed-wing propeller drones (RAVEN, LisEagle) and evaluates
  1st-order propeller-motor electrical power.
- ``compute_neutral_point_sweep``: evaluates 3D aerodynamic center & whole-
  aircraft neutral point across swept morphing wing configurations (LisEagle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.propulsion.actuator import (
    ActuatorPowerBreakdown,
    FirstOrderMotorSpec,
    evaluate_flapping_motor_cycle,
    evaluate_propeller_motor_cruise,
)

if TYPE_CHECKING:
    from fluxvortex.cases.avian_aircraft import AvianAircraftSpec


class MorphingFlappingWingKinematics:
    """Tapered, swept, aeroelastically twisted 3D hinged flapping or fixed wing.

    Coordinate conventions match FLUXVortex native solvers:
    - +x: freestream direction (LE -> TE at zero AoA)
    - +y: starboard spanwise axis
    - +z: upward lift axis
    - positive ``alpha_deg``: nose-up pitch about +y (TE drops in -z)
    - flap law: ``phi(t) = r(t) * (phi0 + phi1 * sin(omega * t))``
      so ``cos(omega * t) < 0`` is downstroke (``phidot < 0``) and
      ``cos(omega * t) > 0`` is upstroke (``phidot > 0``).
    - aeroelastic twist about the leading-edge spar:
      ``beta(eta, t) = r(t) * |eta| * theta_tip * cos(omega * t)``
      (nose-down on downstroke to tilt resultant force into thrust, nose-up on
      upstroke to feather and avoid negative-lift stall).
    - optional FSF upstroke span folding (RoboFalcon 2.0):
      ``k_span(t) = 1 - r(t) * f_fold * 0.5 * (1 + cos(omega * t))``
      (full span at mid-downstroke, retracted by ``f_fold`` at mid-upstroke).
    """

    def __init__(
        self,
        *,
        span_m: float,
        area_m2: float,
        alpha_deg: float,
        taper_ratio: float = 0.65,
        quarter_chord_sweep_deg: float = 0.0,
        sweep_start_eta: float = 0.0,
        camber_zero_lift_deg: float = 0.0,
        flap_amplitude_half_deg: float = 0.0,
        flap_dihedral_deg: float = 0.0,
        twist_tip_deg: float = 0.0,
        upstroke_span_fold: float = 0.0,
        omega_rad_s: float = 0.0,
        start_ramp_s: float = 0.0,
        chordwise_panels: int = 6,
        spanwise_panels: int = 20,
        device: str = "cuda:0",
        surface_id: str = "avian_wing",
        body_id: str = "body_0",
    ) -> None:
        if int(spanwise_panels) % 2 != 0:
            raise ValueError("spanwise_panels must be even so y=0 is a centerline node line")
        nc, ns = int(chordwise_panels), int(spanwise_panels)
        self._nc, self._ns = nc, ns
        self._device = torch.device(device)
        dtype = torch.float64

        half_span = 0.5 * float(span_m)
        lam = float(taper_ratio)
        c_root = 2.0 * float(area_m2) / (float(span_m) * (1.0 + lam))
        tan_sweep = math.tan(math.radians(float(quarter_chord_sweep_deg)))

        # Normalized chord lines matching assemble_native_ring_geometry:
        # [0 (LE), 0.25/nc, 1.25/nc, ..., (nc-0.75)/nc, 1.0 (TE)]
        xi = [0.0] + [(i + 0.25) / nc for i in range(nc)] + [1.0]
        xi_t = torch.tensor(xi, device=self._device, dtype=dtype).reshape(nc + 2, 1)

        y0 = torch.linspace(-half_span, half_span, ns + 1, device=self._device, dtype=dtype)
        eta = y0 / max(half_span, 1.0e-12)
        abs_eta = torch.abs(eta)
        chord_y = c_root * (1.0 - (1.0 - lam) * abs_eta)
        y_wrist = float(sweep_start_eta) * half_span
        y_swept = torch.clamp(torch.abs(y0) - y_wrist, min=0.0)
        x_le_y = y_swept * tan_sweep + 0.25 * (c_root - chord_y)

        self._y0 = y0.reshape(1, ns + 1)
        self._abs_y0 = torch.abs(y0).reshape(1, ns + 1)
        self._abs_eta = abs_eta.reshape(1, ns + 1)
        self._side = torch.sign(y0).reshape(1, ns + 1)
        self._x_le = x_le_y.reshape(1, ns + 1)
        self._sx = xi_t * chord_y.reshape(1, ns + 1)  # (nc+2, ns+1)

        self._alpha = math.radians(float(alpha_deg) + float(camber_zero_lift_deg))
        self._phi0 = math.radians(float(flap_dihedral_deg))
        self._phi1 = math.radians(float(flap_amplitude_half_deg))
        self._twist_tip = math.radians(float(twist_tip_deg))
        self._fold = float(upstroke_span_fold)
        self._omega = float(omega_rad_s)
        self._ramp = float(start_ramp_s)
        self._surface_id = surface_id
        self._body_id = body_id
        self.planform_area_m2 = float(area_m2)
        self.span_m = float(span_m)
        self.c_root_m = c_root
        self.topology_digest = (
            f"avian-wing:{nc}x{ns}:b{span_m:.3f}:S{area_m2:.4f}:a{alpha_deg:.2f}"
        )

    def _ramp_and_deriv(self, t: float) -> tuple[float, float]:
        if self._ramp <= 0.0 or t >= self._ramp:
            return 1.0, 0.0
        if t <= 0.0:
            return 0.0, 0.0
        phase = math.pi * t / self._ramp
        r = 0.5 * (1.0 - math.cos(phase))
        rdot = 0.5 * (math.pi / self._ramp) * math.sin(phase)
        return r, rdot

    def flap_state(self, t: float) -> tuple[float, float, float]:
        """Return (phi, phi_dot, phi_ddot) in radians, rad/s, rad/s^2."""

        t = float(t)
        w = self._omega
        s = math.sin(w * t)
        c = math.cos(w * t)
        base = self._phi0 + self._phi1 * s
        bdot = self._phi1 * w * c
        bddot = -self._phi1 * w * w * s
        if self._ramp <= 0.0 or t >= self._ramp:
            return base, bdot, bddot
        r, rdot = self._ramp_and_deriv(t)
        rddot = 0.5 * ((math.pi / self._ramp) ** 2) * math.cos(math.pi * t / self._ramp) if t > 0.0 else 0.0
        phi = r * base
        phidot = rdot * base + r * bdot
        phiddot = rddot * base + 2.0 * rdot * bdot + r * bddot
        return phi, phidot, phiddot

    def positions_and_velocities(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact analytical node positions and velocities of shape (nc+2, ns+1, 3)."""

        t = float(t)
        w = self._omega
        r, rdot = self._ramp_and_deriv(t)
        sin_wt = math.sin(w * t)
        cos_wt = math.cos(w * t)

        # Flap angle phi(t) and phidot(t)
        phi, phidot, _ = self.flap_state(t)
        cos_p, sin_p = math.cos(phi), math.sin(phi)

        # Spanwise aeroelastic twist beta(eta, t) = r(t) * |eta| * twist_tip * cos(w*t)
        twist_amp = r * self._twist_tip * cos_wt
        twist_amp_dot = (rdot * cos_wt - r * w * sin_wt) * self._twist_tip
        beta = self._abs_eta * twist_amp
        beta_dot = self._abs_eta * twist_amp_dot
        cos_b = torch.cos(beta)
        sin_b = torch.sin(beta)

        # Local twisted chord coordinates
        x_tw = self._x_le + self._sx * cos_b
        z_tw = -self._sx * sin_b
        x_tw_dot = -self._sx * sin_b * beta_dot
        z_tw_dot = -self._sx * cos_b * beta_dot

        # Optional FSF upstroke span folding
        k_span = 1.0 - r * self._fold * 0.5 * (1.0 + cos_wt)
        k_span_dot = -self._fold * 0.5 * (rdot * (1.0 + cos_wt) - r * w * sin_wt)
        y_eff = k_span * self._y0
        abs_y_eff = k_span * self._abs_y0
        y_eff_dot = k_span_dot * self._y0
        abs_y_eff_dot = k_span_dot * self._abs_y0

        # Symmetric root flapping R_x(side * phi)
        y_fl = y_eff * cos_p - self._side * z_tw * sin_p
        z_fl = abs_y_eff * sin_p + z_tw * cos_p
        y_fl_dot = (
            y_eff_dot * cos_p
            - y_eff * sin_p * phidot
            - self._side * (z_tw_dot * sin_p + z_tw * cos_p * phidot)
        )
        z_fl_dot = (
            abs_y_eff_dot * sin_p
            + abs_y_eff * cos_p * phidot
            + (z_tw_dot * cos_p - z_tw * sin_p * phidot)
        )

        # Root geometric incidence R_y(alpha)
        cos_a, sin_a = math.cos(self._alpha), math.sin(self._alpha)
        x = x_tw * cos_a + z_fl * sin_a
        y = y_fl
        z = -x_tw * sin_a + z_fl * cos_a

        vx = x_tw_dot * cos_a + z_fl_dot * sin_a
        vy = y_fl_dot
        vz = -x_tw_dot * sin_a + z_fl_dot * cos_a

        pos = torch.stack((x, y, z), dim=-1)
        vel = torch.stack((vx, vy, vz), dim=-1)
        return pos, vel

    def evaluate(self, time: float) -> SurfaceFrame:
        from fluxvortex.warp_fsi.q16_flux_v5m_native import assemble_native_ring_geometry

        position, velocity = self.positions_and_velocities(time)
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


@dataclass(frozen=True)
class OperatingPointResult:
    """Single converged operating point from 3D GPU UVLM + 1st-order motor."""

    airspeed_m_s: float
    alpha_deg: float
    frequency_hz: float
    mean_lift_n: float
    weight_n: float
    mean_wing_thrust_n: float
    mean_viscous_wing_drag_n: float
    body_tail_drag_n: float
    net_thrust_n: float
    mean_cl: float
    mean_ct_net: float
    mean_cd_induced: float
    mean_my_nm: float
    max_lesp_pre: float
    max_neumann_residual: float
    separated_step_fraction: float
    cycle_lift_drift_rel: float
    wall_seconds: float
    power: ActuatorPowerBreakdown


@dataclass(frozen=True)
class CruiseTrimResult:
    """Converged steady-level cruise trim result for an aircraft."""

    aircraft_id: str
    name: str
    propulsion_kind: str
    converged: bool
    iterations: int
    lift_error_rel: float
    thrust_error_rel: float
    operating_point: OperatingPointResult


def _make_rigid_stepper(
    *,
    nc: int,
    ns: int,
    density: float,
    speed_m_s: float,
    dt: float,
    wake_max_rows: int,
    lesp_crit: float,
    device: str,
):
    from fluxvortex.aero.v5m.stepper import RigidV5MStepper
    from fluxvortex.warp_fsi import q16_flux_v5m_native as native
    from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
        RigidNativeV5MSolver,
        RigidV5MSurface,
    )

    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device=device)
    settings = native.NativeV5MConfig(
        chordwise_panels=nc,
        spanwise_panels=ns,
        density=density,
        freestream=speed_m_s,
        aerodynamic_dt=dt,
        wake_max_rows=wake_max_rows,
        particle_capacity=65_536,
        wake_free_rows=0,
        particle_max_age_steps=0,
        dvm_target_spacing_chord=0.018,
        lesp_crit_override=lesp_crit,
        wake_history_mode="bound_rate",
        device=device,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=device)
    return solver, stepper


def evaluate_flapping_operating_point(
    spec: AvianAircraftSpec,
    *,
    alpha_deg: float,
    frequency_hz: float,
    airspeed_m_s: float | None = None,
    cd0_override: float | None = None,
    area_override: float | None = None,
    cycles: int | None = None,
    steps_per_cycle: int | None = None,
    cfl_target: float = 0.85,
    wing_profile_cd0: float = 0.045,
    use_flexible_fsi: bool | None = None,
    device: str = "cuda:0",
) -> OperatingPointResult:
    """Run multi-cycle 3D GPU UVLM + Hirato v2 LEV (with optional 2-way FSI & FSF wrist) and motor."""

    import time as _time
    from fluxvortex.propulsion.flexible_avian_fsi import (
        FlexibleArticulatedWingKinematics,
        TwoWayNewmarkAitkenFSI,
        build_tail_geometry_and_aero,
        build_wing_structural_model,
    )

    u_inf = float(spec.cruise_speed_m_s if airspeed_m_s is None else airspeed_m_s)
    area_eff = float(spec.area_m2 if area_override is None else area_override)
    freq = float(frequency_hz)
    omega = 2.0 * math.pi * freq
    n_cyc = int(spec.trim_cycles if cycles is None else cycles)
    if steps_per_cycle is not None:
        spc = int(steps_per_cycle)
    else:
        # Enforce local tip convective CFL = U_rel_tip * dt / dx_tip <= cfl_target so
        # nascent Hirato v2 LEV rings never jump past the leading-edge collocation panel.
        c_root = 2.0 * area_eff / (spec.span_m * (1.0 + spec.taper_ratio))
        c_tip = c_root * spec.taper_ratio
        dx_tip = c_tip / max(spec.chordwise_panels, 1)
        v_tip = omega * math.radians(spec.flap_amplitude_half_deg) * (0.5 * spec.span_m)
        u_rel_tip = math.hypot(u_inf, v_tip)
        spc_cfl = int(math.ceil(u_rel_tip / max(cfl_target * freq * dx_tip, 1.0e-9)))
        spc = max(int(spec.steps_per_cycle), min(spc_cfl, 192))
        # Round up to multiple of 8 for clean quarter-cycle alignment
        spc = int(math.ceil(spc / 8.0) * 8)
    dt = (1.0 / freq) / spc
    total_steps = n_cyc * spc
    wake_rows = int(math.ceil(1.25 * spc))

    cd0_total = float(spec.cd0_body_and_profile if cd0_override is None else cd0_override)
    cd0_wing = min(float(wing_profile_cd0), 0.55 * cd0_total)
    cd0_body = max(cd0_total - cd0_wing, 0.015)
    q_dyn = 0.5 * spec.rho_kg_m3 * u_inf * u_inf
    d_body_tail = q_dyn * spec.area_m2 * cd0_body

    fsi_active = bool(spec.flexibility.enabled) if use_flexible_fsi is None else bool(use_flexible_fsi)
    use_articulated = fsi_active or bool(spec.fsf.enabled) or (spec.planform_kind != "tapered")

    if use_articulated:
        kin: Any = FlexibleArticulatedWingKinematics(
            span_m=spec.span_m,
            area_m2=area_eff,
            alpha_deg=alpha_deg,
            taper_ratio=spec.taper_ratio,
            quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
            camber_zero_lift_deg=spec.camber_zero_lift_deg,
            planform_kind=spec.planform_kind,
            planform_id=spec.planform_id,
            airfoil_camber_id=spec.airfoil_camber_id,
            airfoil_camber_scale=spec.airfoil_camber_scale,
            flap_amplitude_half_deg=spec.flap_amplitude_half_deg,
            flap_dihedral_deg=spec.flap_dihedral_deg,
            omega_rad_s=omega,
            start_ramp_s=0.20 / freq,
            fsf=spec.fsf,
            prescribed_twist_tip_deg=0.0 if fsi_active else spec.twist_tip_deg,
            chordwise_panels=spec.chordwise_panels,
            spanwise_panels=spec.spanwise_panels,
            device=device,
        )
    else:
        kin = MorphingFlappingWingKinematics(
            span_m=spec.span_m,
            area_m2=area_eff,
            alpha_deg=alpha_deg,
            taper_ratio=spec.taper_ratio,
            quarter_chord_sweep_deg=spec.quarter_chord_sweep_deg,
            camber_zero_lift_deg=spec.camber_zero_lift_deg,
            flap_amplitude_half_deg=spec.flap_amplitude_half_deg,
            flap_dihedral_deg=spec.flap_dihedral_deg,
            twist_tip_deg=spec.twist_tip_deg,
            upstroke_span_fold=spec.upstroke_span_fold,
            omega_rad_s=omega,
            start_ramp_s=0.20 / freq,
            chordwise_panels=spec.chordwise_panels,
            spanwise_panels=spec.spanwise_panels,
            device=device,
        )

    fsi_integrator = None
    if fsi_active:
        struct_model = build_wing_structural_model(kin, spec.flexibility)
        fsi_integrator = TwoWayNewmarkAitkenFSI(kin, struct_model, dt=dt)

    _, stepper = _make_rigid_stepper(
        nc=spec.chordwise_panels,
        ns=spec.spanwise_panels,
        density=spec.rho_kg_m3,
        speed_m_s=u_inf,
        dt=dt,
        wake_max_rows=wake_rows,
        lesp_crit=spec.lesp_crit,
        device=device,
    )

    v_inf_t = torch.tensor([u_inf, 0.0, 0.0], device=device, dtype=torch.float64)
    # Pitched root flapping hinge axis e_h = (cos(alpha_eff), 0, -sin(alpha_eff))
    hinge_axis = torch.tensor(
        [math.cos(kin._alpha), 0.0, -math.sin(kin._alpha)],
        device=device,
        dtype=torch.float64,
    )
    torch.cuda.synchronize(torch.device(device))
    t0 = _time.perf_counter()
    owner = stepper.initialize((kin.evaluate(0.0),))

    t_hist = np.empty(total_steps, dtype=np.float64)
    lift_hist = np.empty(total_steps, dtype=np.float64)
    wing_thrust_invisc_hist = np.empty(total_steps, dtype=np.float64)
    visc_drag_hist = np.empty(total_steps, dtype=np.float64)
    my_hist = np.empty(total_steps, dtype=np.float64)
    hinge_moment_hist = np.empty(total_steps, dtype=np.float64)
    p_aero_hist = np.empty(total_steps, dtype=np.float64)
    phidot_hist = np.empty(total_steps, dtype=np.float64)
    phiddot_hist = np.empty(total_steps, dtype=np.float64)
    max_lesp = 0.0
    max_neu = 0.0
    sep_steps = 0

    for step in range(1, total_steps + 1):
        t_now = step * dt
        if fsi_integrator is not None:
            proposal, frame, _, f_visc, v_rigid_panel, _ = fsi_integrator.step_coupled(
                t_now=t_now,
                stepper=stepper,
                owner=owner,
                v_inf_t=v_inf_t,
                rho=spec.rho_kg_m3,
                cd0_wing=cd0_wing,
                max_subiters=3,
                tol=1.0e-3,
            )
            panel_f = proposal.load.panel_forces
            panel_r = frame.collocation_I
            f_total_panel = panel_f + f_visc
        else:
            frame = kin.evaluate(t_now)
            proposal = stepper.propose(owner, (frame,), dt)
            stepper.commit(owner, proposal)
            panel_f = proposal.load.panel_forces
            panel_v = frame.collocation_velocity_I
            panel_r = frame.collocation_I
            v_rel = v_inf_t[None, :] - panel_v
            v_rel_mag = torch.linalg.vector_norm(v_rel, dim=1, keepdim=True)
            f_visc = (
                0.5 * spec.rho_kg_m3 * cd0_wing * frame.areas[:, None] * v_rel_mag * v_rel
            )
            f_total_panel = panel_f + f_visc
            v_rigid_panel = None

        diag = owner.state.diagnostics[-1]

        _, phidot, phiddot = kin.flap_state(t_now)
        if v_rigid_panel is not None and bool(spec.fsf.enabled):
            # Exact virtual work of single-motor conical-rocker FSF actuator (root + wrist joints),
            # excluding passive elastic modal velocities qdot.
            p_inst = -float(torch.sum(f_total_panel * v_rigid_panel).item())
            m_hinge = p_inst / phidot if abs(phidot) > 1.0e-6 else 0.0
        else:
            # Exact root-hinge aerodynamic torque opposing actuator flapping rate phidot:
            # M_hinge(t) = -sum_p sign(y_p) * ((r_p x F_p) . e_hinge).
            panel_side = torch.sign(panel_r[:, 1])
            r_cross_f = torch.linalg.cross(panel_r, f_total_panel, dim=1)
            m_hinge = -float(torch.sum(panel_side * torch.sum(r_cross_f * hinge_axis[None, :], dim=1)).item())
            p_inst = m_hinge * phidot

        tot_invisc = proposal.load.total_force
        tot_visc = torch.sum(f_visc, dim=0)
        tot_my = float(proposal.load.total_moment[1].item())

        idx = step - 1
        t_hist[idx] = t_now
        lift_hist[idx] = float(tot_invisc[2].item() + tot_visc[2].item())
        wing_thrust_invisc_hist[idx] = -float(tot_invisc[0].item())
        visc_drag_hist[idx] = float(tot_visc[0].item())
        my_hist[idx] = tot_my
        hinge_moment_hist[idx] = m_hinge
        p_aero_hist[idx] = p_inst
        phidot_hist[idx] = phidot
        phiddot_hist[idx] = phiddot

        max_lesp = max(max_lesp, float(diag.get("lesp_pre_max_abs", 0.0)))
        neu = float(diag.get("full_surface_neumann_max_abs", 0.0))
        if math.isfinite(neu):
            max_neu = max(max_neu, neu)
        if int(diag.get("separated_strip_count", 0)) > 0:
            sep_steps += 1

    torch.cuda.synchronize(torch.device(device))
    wall_s = _time.perf_counter() - t0

    # Last complete cycle statistics + physical tail lift contribution
    sl = slice(total_steps - spc, total_steps)
    mean_wing_lift = float(np.mean(lift_hist[sl]))
    q_s = max(q_dyn * spec.area_m2, 1.0e-9)
    tail_aero = build_tail_geometry_and_aero(
        spec.tail,
        alpha_deg=alpha_deg,
        airspeed_m_s=u_inf,
        rho=spec.rho_kg_m3,
        wing_cl_mean=mean_wing_lift / q_s,
        wing_ar=spec.aspect_ratio,
    )
    mean_lift = mean_wing_lift + float(tail_aero["lift_n"])
    mean_invisc_t = float(np.mean(wing_thrust_invisc_hist[sl]))
    mean_visc_d = float(np.mean(visc_drag_hist[sl]))
    mean_wing_t = mean_invisc_t - mean_visc_d
    net_thrust = mean_wing_t - d_body_tail
    mean_my = float(np.mean(my_hist[sl])) + float(tail_aero["moment_y_nm"])

    if n_cyc >= 2:
        prev_lift = float(np.mean(lift_hist[total_steps - 2 * spc : total_steps - spc])) + float(tail_aero["lift_n"])
        drift_rel = abs(mean_lift - prev_lift) / max(abs(mean_lift), spec.weight_n, 1.0e-6)
    else:
        drift_rel = 0.0

    power = evaluate_flapping_motor_cycle(
        motor=spec.motor,
        t_s=t_hist[sl],
        flap_rate_rad_s=phidot_hist[sl],
        aero_power_w=p_aero_hist[sl],
        frequency_hz=freq,
        mass_kg=spec.mass_kg,
        cruise_speed_m_s=u_inf,
        wing_inertia_kg_m2=spec.wing_inertia_kg_m2,
        flap_accel_rad_s2=phiddot_hist[sl],
        inertial_elastic_recovery=spec.inertial_elastic_recovery,
    )
    return OperatingPointResult(
        airspeed_m_s=u_inf,
        alpha_deg=float(alpha_deg),
        frequency_hz=freq,
        mean_lift_n=mean_lift,
        weight_n=spec.weight_n,
        mean_wing_thrust_n=mean_wing_t,
        mean_viscous_wing_drag_n=mean_visc_d,
        body_tail_drag_n=d_body_tail,
        net_thrust_n=net_thrust,
        mean_cl=mean_lift / q_s,
        mean_ct_net=net_thrust / q_s,
        mean_cd_induced=max(-mean_invisc_t, 0.0) / q_s,
        mean_my_nm=mean_my,
        max_lesp_pre=max_lesp,
        max_neumann_residual=max_neu,
        separated_step_fraction=sep_steps / max(total_steps, 1),
        cycle_lift_drift_rel=drift_rel,
        wall_seconds=wall_s,
        power=power,
    )


def solve_flapping_cruise_trim(
    spec: AvianAircraftSpec,
    *,
    airspeed_m_s: float | None = None,
    cd0_override: float | None = None,
    area_override: float | None = None,
    alpha_init_deg: float | None = None,
    freq_init_hz: float | None = None,
    max_iters: int = 7,
    tol_rel: float = 0.025,
    verify_cycles: int = 3,
    device: str = "cuda:0",
) -> CruiseTrimResult:
    """Solve 2D steady-level cruise trim (alpha_deg, frequency_hz) for L = m*g, T = D.

    Runs consistent grid topology across search (2 cycles) and verification (3 cycles)
    so the converged root never drifts across a mesh switch, refreshing the 2x2
    finite-difference Jacobian if Broyden progress slows.
    """

    u_inf = float(spec.cruise_speed_m_s if airspeed_m_s is None else airspeed_m_s)
    area_eff = float(spec.area_m2 if area_override is None else area_override)
    q_s = 0.5 * spec.rho_kg_m3 * u_inf * u_inf * area_eff
    cl_req = spec.weight_n / max(q_s, 1.0e-9)
    ar = (spec.span_m ** 2) / area_eff
    cl_alpha = 2.0 * math.pi * ar / (ar + 2.0)
    a_lo, a_hi = spec.alpha_bounds_deg
    f_lo, f_hi = spec.frequency_bounds_hz

    alpha = (
        float(alpha_init_deg)
        if alpha_init_deg is not None
        else float(np.clip(math.degrees(cl_req / cl_alpha) - spec.camber_zero_lift_deg, a_lo + 0.5, a_hi - 0.5))
    )
    freq = (
        float(freq_init_hz)
        if freq_init_hz is not None
        else float(np.clip(spec.nominal_frequency_hz * (u_inf / spec.cruise_speed_m_s) ** 0.70, f_lo, f_hi))
    )

    def eval_res(
        a_val: float, f_val: float, *, n_cyc: int = 2
    ) -> tuple[OperatingPointResult, np.ndarray]:
        op = evaluate_flapping_operating_point(
            spec,
            alpha_deg=a_val,
            frequency_hz=f_val,
            airspeed_m_s=u_inf,
            cd0_override=cd0_override,
            area_override=area_override,
            cycles=n_cyc,
            wing_profile_cd0=0.045,
            device=device,
        )
        t_scale = max(op.body_tail_drag_n + op.mean_viscous_wing_drag_n, 0.08 * op.weight_n)
        res = np.array(
            [(op.mean_lift_n - op.weight_n) / op.weight_n, op.net_thrust_n / t_scale],
            dtype=np.float64,
        )
        return op, res

    def compute_fd_jacobian(a_c: float, f_c: float, r_c: np.ndarray) -> np.ndarray:
        da = 2.0 if a_c + 2.0 <= a_hi else -2.0
        df = max(0.55, 0.12 * f_c)
        if f_c + df > f_hi:
            df = -df
        a_p = float(np.clip(a_c + da, a_lo, a_hi))
        f_p = float(np.clip(f_c + df, f_lo, f_hi))
        _, r_a = eval_res(a_p, f_c, n_cyc=2)
        _, r_f = eval_res(a_c, f_p, n_cyc=2)
        da_eff = a_p - a_c
        df_eff = f_p - f_c
        # Enforce physically sane sign floor on diagonal sensitivities so noisy
        # strip-separation ripples never invert the Newton direction.
        j00 = (r_a[0] - r_c[0]) / da_eff if abs(da_eff) > 1e-6 else 0.10
        j10 = (r_a[1] - r_c[1]) / da_eff if abs(da_eff) > 1e-6 else 0.0
        j01 = (r_f[0] - r_c[0]) / df_eff if abs(df_eff) > 1e-6 else 0.0
        j11 = (r_f[1] - r_c[1]) / df_eff if abs(df_eff) > 1e-6 else 0.25
        j00 = max(j00, 0.03)
        j11 = max(j11, 0.05)
        return np.array([[j00, j01], [j10, j11]], dtype=np.float64)

    op, r = eval_res(alpha, freq, n_cyc=2)
    best_op, best_r = op, r
    converged = bool(max(abs(r[0]), abs(r[1])) <= tol_rel)
    J = None
    iters = 0
    if not converged:
        J = compute_fd_jacobian(alpha, freq, r)
        for it in range(1, min(max_iters, 5) + 1):
            iters = it
            if it == 3 and max(abs(r[0]), abs(r[1])) > 0.06:
                J = compute_fd_jacobian(alpha, freq, r)
            jtj = J.T @ J + 2.0e-3 * np.eye(2)
            step = -np.linalg.solve(jtj, J.T @ r)
            max_da, max_df = 4.5, max(2.2, 0.30 * freq)
            scale = max(1.0, abs(step[0]) / max_da, abs(step[1]) / max_df)
            step = step / scale

            norm_cur = float(np.linalg.norm(r))
            accepted = False
            for damp in (1.0, 0.5):
                a_try = float(np.clip(alpha + damp * step[0], a_lo, a_hi))
                f_try = float(np.clip(freq + damp * step[1], f_lo, f_hi))
                s_vec = np.array([a_try - alpha, f_try - freq], dtype=np.float64)
                if np.linalg.norm(s_vec) < 1.0e-4:
                    continue
                try:
                    op_try, r_try = eval_res(a_try, f_try, n_cyc=2)
                except (FloatingPointError, RuntimeError):
                    continue
                if not np.all(np.isfinite(r_try)):
                    continue
                if float(np.linalg.norm(r_try)) < float(np.linalg.norm(best_r)):
                    best_op, best_r = op_try, r_try
                if float(np.linalg.norm(r_try)) < norm_cur or damp == 0.5:
                    alpha, freq, op, r = a_try, f_try, op_try, r_try
                    accepted = True
                    break
            if max(abs(best_r[0]), abs(best_r[1])) <= tol_rel:
                converged = True
                break
            if not accepted:
                break

    final_op, final_r = eval_res(
        best_op.alpha_deg,
        best_op.frequency_hz,
        n_cyc=max(int(verify_cycles), spec.trim_cycles),
    )
    # Up to two full-cycle polish corrections so lift and thrust converge cleanly on cycle 3
    for _ in range(2):
        if max(abs(final_r[0]), abs(final_r[1])) <= 0.04 or J is None:
            break
        try:
            step = -np.linalg.solve(J.T @ J + 2e-3 * np.eye(2), J.T @ final_r)
            a_pol = float(np.clip(final_op.alpha_deg + np.clip(step[0], -3.5, 3.5), a_lo, a_hi))
            f_pol = float(np.clip(final_op.frequency_hz + np.clip(step[1], -1.8, 1.8), f_lo, f_hi))
            pol_op, pol_r = eval_res(
                a_pol, f_pol, n_cyc=max(int(verify_cycles), spec.trim_cycles)
            )
            if float(np.linalg.norm(pol_r)) < float(np.linalg.norm(final_r)):
                final_op, final_r = pol_op, pol_r
            else:
                break
        except Exception:
            break
    converged = bool(max(abs(final_r[0]), abs(final_r[1])) <= max(tol_rel * 2.0, 0.05))
    return CruiseTrimResult(
        aircraft_id=spec.aircraft_id,
        name=spec.name,
        propulsion_kind=spec.propulsion_kind,
        converged=converged,
        iterations=iters,
        lift_error_rel=float(final_r[0]),
        thrust_error_rel=float(final_r[1]),
        operating_point=final_op,
    )


def evaluate_fixed_wing_operating_point(
    spec: AvianAircraftSpec,
    *,
    alpha_deg: float,
    airspeed_m_s: float | None = None,
    span_override: float | None = None,
    area_override: float | None = None,
    sweep_override_deg: float | None = None,
    sweep_start_eta: float = 0.0,
    camber_override_deg: float | None = None,
    cd0_area_override_m2: float | None = None,
    steps: int = 36,
    device: str = "cuda:0",
) -> OperatingPointResult:
    """Run steady 3D GPU UVLM convergence on a fixed/morphing wing + propeller motor."""

    import time as _time

    u_inf = float(spec.cruise_speed_m_s if airspeed_m_s is None else airspeed_m_s)
    span = float(spec.span_m if span_override is None else span_override)
    area = float(spec.area_m2 if area_override is None else area_override)
    sweep = float(spec.quarter_chord_sweep_deg if sweep_override_deg is None else sweep_override_deg)
    camber = float(spec.camber_zero_lift_deg if camber_override_deg is None else camber_override_deg)

    c_mean = area / span
    dt = 0.25 * c_mean / u_inf  # convective dt* = 0.25 -> 36 steps = 9 chords of wake
    wake_rows = steps

    q_dyn = 0.5 * spec.rho_kg_m3 * u_inf * u_inf
    if cd0_area_override_m2 is not None:
        d_parasitic = q_dyn * float(cd0_area_override_m2)
    else:
        d_parasitic = q_dyn * area * spec.cd0_body_and_profile

    kin = MorphingFlappingWingKinematics(
        span_m=span,
        area_m2=area,
        alpha_deg=alpha_deg,
        taper_ratio=spec.taper_ratio,
        quarter_chord_sweep_deg=sweep,
        sweep_start_eta=sweep_start_eta,
        camber_zero_lift_deg=camber,
        flap_amplitude_half_deg=0.0,
        flap_dihedral_deg=0.0,
        twist_tip_deg=0.0,
        omega_rad_s=0.0,
        chordwise_panels=spec.chordwise_panels,
        spanwise_panels=spec.spanwise_panels,
        device=device,
    )
    _, stepper = _make_rigid_stepper(
        nc=spec.chordwise_panels,
        ns=spec.spanwise_panels,
        density=spec.rho_kg_m3,
        speed_m_s=u_inf,
        dt=dt,
        wake_max_rows=wake_rows,
        lesp_crit=10.0,  # steady cruise fixed wing: attached flow
        device=device,
    )

    torch.cuda.synchronize(torch.device(device))
    t0 = _time.perf_counter()
    frame = kin.evaluate(0.0)
    owner = stepper.initialize((frame,))
    lift_vals = []
    drag_i_vals = []
    my_vals = []
    max_neu = 0.0
    max_lesp = 0.0

    for step in range(1, steps + 1):
        proposal = stepper.propose(owner, (frame,), dt)
        stepper.commit(owner, proposal)
        diag = owner.state.diagnostics[-1]
        f = proposal.load.total_force
        m = proposal.load.total_moment
        lift_vals.append(float(f[2].item()))
        drag_i_vals.append(float(f[0].item()))
        my_vals.append(float(m[1].item()))
        max_lesp = max(max_lesp, float(diag.get("lesp_pre_max_abs", 0.0)))
        neu = float(diag.get("full_surface_neumann_max_abs", 0.0))
        if math.isfinite(neu):
            max_neu = max(max_neu, neu)

    torch.cuda.synchronize(torch.device(device))
    wall_s = _time.perf_counter() - t0

    # Average last 4 steady steps
    lift_n = float(np.mean(lift_vals[-4:]))
    drag_ind_n = max(float(np.mean(drag_i_vals[-4:])), 0.0)
    my_nm = float(np.mean(my_vals[-4:]))
    drift = abs(lift_vals[-1] - lift_vals[-5]) / max(abs(lift_n), 1.0e-6)
    thrust_req_n = drag_ind_n + d_parasitic

    power = evaluate_propeller_motor_cruise(
        motor=spec.motor,
        thrust_n=thrust_req_n,
        cruise_speed_m_s=u_inf,
        mass_kg=spec.mass_kg,
        rho_kg_m3=spec.rho_kg_m3,
    )
    q_s = max(q_dyn * area, 1.0e-9)
    return OperatingPointResult(
        airspeed_m_s=u_inf,
        alpha_deg=float(alpha_deg),
        frequency_hz=0.0,
        mean_lift_n=lift_n,
        weight_n=spec.weight_n,
        mean_wing_thrust_n=thrust_req_n,
        mean_viscous_wing_drag_n=0.0,
        body_tail_drag_n=d_parasitic,
        net_thrust_n=0.0,
        mean_cl=lift_n / q_s,
        mean_ct_net=0.0,
        mean_cd_induced=drag_ind_n / q_s,
        mean_my_nm=my_nm,
        max_lesp_pre=max_lesp,
        max_neumann_residual=max_neu,
        separated_step_fraction=0.0,
        cycle_lift_drift_rel=drift,
        wall_seconds=wall_s,
        power=power,
    )


def solve_fixed_wing_cruise_trim(
    spec: AvianAircraftSpec,
    *,
    airspeed_m_s: float | None = None,
    span_override: float | None = None,
    area_override: float | None = None,
    sweep_override_deg: float | None = None,
    camber_override_deg: float | None = None,
    cd0_area_override_m2: float | None = None,
    device: str = "cuda:0",
) -> CruiseTrimResult:
    """Secant trim of AoA for L = m*g on fixed/morphing propeller drones."""

    u_inf = float(spec.cruise_speed_m_s if airspeed_m_s is None else airspeed_m_s)
    span = float(spec.span_m if span_override is None else span_override)
    area = float(spec.area_m2 if area_override is None else area_override)
    camber = float(spec.camber_zero_lift_deg if camber_override_deg is None else camber_override_deg)

    q_s = 0.5 * spec.rho_kg_m3 * u_inf * u_inf * area
    ar = span * span / area
    cl_alpha = 2.0 * math.pi * ar / (ar + 2.0)
    a0 = math.degrees((spec.weight_n / q_s) / cl_alpha) - camber
    a1 = a0 + 1.5

    def eval_a(a_deg: float) -> OperatingPointResult:
        return evaluate_fixed_wing_operating_point(
            spec,
            alpha_deg=a_deg,
            airspeed_m_s=u_inf,
            span_override=span,
            area_override=area,
            sweep_override_deg=sweep_override_deg,
            camber_override_deg=camber,
            cd0_area_override_m2=cd0_area_override_m2,
            device=device,
        )

    op0 = eval_a(a0)
    r0 = (op0.mean_lift_n - op0.weight_n) / op0.weight_n
    if abs(r0) <= 0.005:
        return CruiseTrimResult(
            aircraft_id=spec.aircraft_id,
            name=spec.name,
            propulsion_kind=spec.propulsion_kind,
            converged=True,
            iterations=1,
            lift_error_rel=r0,
            thrust_error_rel=0.0,
            operating_point=op0,
        )

    op1 = eval_a(a1)
    r1 = (op1.mean_lift_n - op1.weight_n) / op1.weight_n
    best_op, best_r = (op1, r1) if abs(r1) < abs(r0) else (op0, r0)
    iters = 2
    for it in range(3, 7):
        if abs(r1 - r0) < 1.0e-7:
            break
        a2 = float(np.clip(a1 - r1 * (a1 - a0) / (r1 - r0), -8.0, 20.0))
        op2 = eval_a(a2)
        r2 = (op2.mean_lift_n - op2.weight_n) / op2.weight_n
        iters = it
        if abs(r2) < abs(best_r):
            best_op, best_r = op2, r2
        a0, r0 = a1, r1
        a1, r1 = a2, r2
        if abs(best_r) <= 0.005:
            break

    return CruiseTrimResult(
        aircraft_id=spec.aircraft_id,
        name=spec.name,
        propulsion_kind=spec.propulsion_kind,
        converged=bool(abs(best_r) <= 0.01),
        iterations=iters,
        lift_error_rel=float(best_r),
        thrust_error_rel=0.0,
        operating_point=best_op,
    )


def compute_neutral_point_sweep(
    spec: AvianAircraftSpec,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Compute 3D wing aerodynamic center & whole-aircraft neutral point vs sweep."""

    cfg = spec.extra_benchmarks["neutral_point_sweep"]
    u_inf = float(cfg["airspeed_m_s"])
    s_tail = float(cfg["tail_area_m2"])
    x_tail = float(cfg["tail_arm_m"])
    cl_a_tail = float(cfg["tail_cl_alpha_eff"])
    wrist_eta = float(cfg.get("wrist_eta", 0.38))
    q_dyn = 0.5 * spec.rho_kg_m3 * u_inf * u_inf

    rows = []
    for item in cfg["configs"]:
        span = float(item["span_m"])
        area = float(item["area_m2"])
        le_sweep_deg = float(item["le_sweep_deg"])
        # Convert outer hand-wing LE sweep to quarter-chord sweep
        c_root = 2.0 * area / (span * (1.0 + spec.taper_ratio))
        c_tip = c_root * spec.taper_ratio
        tan_c4 = math.tan(math.radians(le_sweep_deg)) - 0.25 * (c_root - c_tip) / (0.5 * span)
        c4_sweep_deg = math.degrees(math.atan(tan_c4))

        op_lo = evaluate_fixed_wing_operating_point(
            spec,
            alpha_deg=2.0,
            airspeed_m_s=u_inf,
            span_override=span,
            area_override=area,
            sweep_override_deg=c4_sweep_deg,
            sweep_start_eta=wrist_eta,
            camber_override_deg=0.0,
            device=device,
        )
        op_hi = evaluate_fixed_wing_operating_point(
            spec,
            alpha_deg=6.0,
            airspeed_m_s=u_inf,
            span_override=span,
            area_override=area,
            sweep_override_deg=c4_sweep_deg,
            sweep_start_eta=wrist_eta,
            camber_override_deg=0.0,
            device=device,
        )
        d_alpha = math.radians(4.0)
        d_lift = op_hi.mean_lift_n - op_lo.mean_lift_n
        d_my = op_hi.mean_my_nm - op_lo.mean_my_nm
        # Positive My = nose-up -> M_y = -x_ac * L -> x_ac = -dMy / dL (aft of root LE)
        x_ac_wing_m = -d_my / max(d_lift, 1.0e-9)
        cl_a_wing = (d_lift / (q_dyn * area)) / d_alpha
        # Whole-aircraft neutral point (wing + tail) aft of root LE
        x_np_m = (cl_a_wing * area * x_ac_wing_m + cl_a_tail * s_tail * x_tail) / (
            cl_a_wing * area + cl_a_tail * s_tail
        )
        rows.append(
            {
                "name": item["name"],
                "wing_sweep_joint_deg": item["wing_sweep_joint_deg"],
                "span_m": span,
                "area_m2": area,
                "cl_alpha_wing": cl_a_wing,
                "x_ac_wing_cm": 100.0 * x_ac_wing_m,
                "x_np_aft_le_cm": 100.0 * x_np_m,
                "published_np_cm": item["published_np_cm"],
            }
        )

    # Forward shift from collapsed (45 deg) to extended (130 deg)
    wing_ac_shift_cm = rows[0]["x_ac_wing_cm"] - rows[-1]["x_ac_wing_cm"]
    np_shift_cm = rows[0]["x_np_aft_le_cm"] - rows[-1]["x_np_aft_le_cm"]
    # Express simulated NP relative to central config datum (offset by central published +1.6 cm)
    datum_offset = rows[1]["published_np_cm"] + rows[1]["x_np_aft_le_cm"]
    for row in rows:
        row["sim_np_forward_datum_cm"] = datum_offset - row["x_np_aft_le_cm"]

    return {
        "airspeed_m_s": u_inf,
        "configs": rows,
        "sim_wing_ac_shift_cm": wing_ac_shift_cm,
        "sim_np_shift_cm": np_shift_cm,
        "published_np_shift_cm": float(cfg["published_np_shift_cm"]),
        "np_shift_rel_error_pct": 100.0
        * (np_shift_cm - float(cfg["published_np_shift_cm"]))
        / float(cfg["published_np_shift_cm"]),
    }


__all__ = [
    "CruiseTrimResult",
    "MorphingFlappingWingKinematics",
    "OperatingPointResult",
    "compute_neutral_point_sweep",
    "evaluate_fixed_wing_operating_point",
    "evaluate_flapping_operating_point",
    "solve_fixed_wing_cruise_trim",
    "solve_flapping_cruise_trim",
]
