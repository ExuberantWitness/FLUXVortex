#!/usr/bin/env python3
"""Generate 6-panel 3D + Front/Top Geometry + 2D Slice + FSI + Motor animations for 5 flapping aircraft.

Stage-4 full-fidelity ground-truth visualization pipeline featuring:
  1. Stage-2 RK2 TE/LE closure bridge + matched-horizon Hirato v2 LEV gauge rebasing
     (zero TE-trapped LEV spikes, zero cross-cycle pseudovortex ratcheting).
  2. Ground-truth digitized wing planforms & UIUC airfoil cambers (E378 / GOE225) +
     exact tails (Cloud Owl inverted-T 500/350/250 mm -4 deg, RoboFalcon 2.0 diamond
     kite tail, NPU-Sparrow morphing fan tail -10 deg, Dove pigeon tail + vfin).
  3. Exact RoboFalcon 2.0 Science Advances Eq. S1 in-plane 3-segment Humerus-Radius-Manus
     FSF folding (600 mm -> 327 mm semispan) + whole-wing forward sweep (+2.9..+23.6 deg) +
     asymmetric flapping (-46.5..+38.3 deg), rendered with color-coded Humerus (red),
     Radius (green), Manus (blue), and Fingers (purple) skeleton bars in 3D, Top & Front views.
  4. Added-mass-preconditioned 2-way GPU FSI coupling (spar bending + passive aeroelastic
     torsional twist + nonlinear Foppl-von Karman chordwise membrane camber billowing).
  5. Unobstructed Front View (y-z) & Top View (x-y) panels + 4-row x 4-column high-res
     verification contact sheet (3D + Top + Front + 2D Slice across all 4 stroke phases).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# Configure unified v4 solver + Stage-2 TE/LE closure + matched-horizon LEV gauge rebasing
_UNIFIED_V4_ENV = {
    "FLUXV_GPU_ONLY": "1",
    "FLUXV_LEV_MODEL": "hirato_v2",
    "FLUXV_TE_WAKE": "v4",
    "FLUXV_CONVECTION_V4": "1",
    "FLUXV_CONVECTION_V4_STAGE2_BRIDGE": "1",
    "FLUXV_LEV_GAUGE_REBASE": "1",
    "FLUXV_RING_KERNEL": "v3",
    "FLUXV_WAKE_CORE_PANELS": "0.35",
    "FLUXV_WAKE_CORE_AT_COLLOCATION": "1",
    "FLUXV_WING_SLIP_WALL": "0.50",
    "FLUXV_SQUIRE_CORE": "0.02",
    "FLUXV_LESP_UREL": "1",
    "FLUXV_LEV_SEP_SUCTION": "1",
    "FLUXV_RING_MIXED_PRECISION": "1",
    "FLUXV_FAST_RIGID_LOADS": "1",
    "FLUXV_LEV_MAX_ROWS": "128",
    "FLUXV_PACKET_CLOSURE_SCALE": "l1",
}
for _k, _v in _UNIFIED_V4_ENV.items():
    os.environ.setdefault(_k, _v)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from fluxvortex.cases.avian_aircraft import AVIAN_AIRCRAFT_REGISTRY, AvianAircraftSpec
from fluxvortex.propulsion.actuator import evaluate_flapping_motor_cycle
from fluxvortex.propulsion.cruise_trim import (
    _make_rigid_stepper,
    solve_flapping_cruise_trim,
)
from fluxvortex.propulsion.flexible_avian_fsi import (
    FlexibleArticulatedWingKinematics,
    TwoWayNewmarkAitkenFSI,
    build_tail_geometry_and_aero,
    build_wing_structural_model,
)
from fluxvortex.warp_fsi import lev_hirato

TARGET_AIRCRAFT = (
    "cloud_owl",
    "robofalcon_2",
    "npu_sparrow",
    "dove_2018",
    "dove_highspeed_2024",
)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().cpu().numpy()


def simulate_and_capture_cycle(
    spec: AvianAircraftSpec,
    *,
    alpha_deg: float,
    frequency_hz: float,
    airspeed_m_s: float,
    cycles: int = 3,
    n_frames: int = 48,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run 3D GPU UVLM + 2-way FSI + exact digitized planforms/FSF + rebased LEV and capture final cycle."""

    u_inf = float(airspeed_m_s)
    freq = float(frequency_hz)
    omega = 2.0 * math.pi * freq

    c_root = 2.0 * spec.area_m2 / (spec.span_m * (1.0 + spec.taper_ratio))
    c_tip = c_root * spec.taper_ratio
    dx_tip = c_tip / max(spec.chordwise_panels, 1)
    v_tip = omega * math.radians(spec.flap_amplitude_half_deg) * (0.5 * spec.span_m)
    u_rel_tip = math.hypot(u_inf, v_tip)
    spc_cfl = int(math.ceil(u_rel_tip / max(0.85 * freq * dx_tip, 1.0e-9)))
    spc = max(int(spec.steps_per_cycle), min(spc_cfl, 192))
    spc = int(math.ceil(spc / 8.0) * 8)

    dt = (1.0 / freq) / spc
    total_steps = cycles * spc
    wake_rows = int(math.ceil(1.25 * spc))

    cd0_total = float(spec.cd0_body_and_profile)
    cd0_wing = min(0.045, 0.55 * cd0_total)
    cd0_body = max(cd0_total - cd0_wing, 0.015)
    q_dyn = 0.5 * spec.rho_kg_m3 * u_inf * u_inf
    d_body_tail = q_dyn * spec.area_m2 * cd0_body

    kin = FlexibleArticulatedWingKinematics(
        span_m=spec.span_m,
        area_m2=spec.area_m2,
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
        prescribed_twist_tip_deg=0.0 if spec.flexibility.enabled else spec.twist_tip_deg,
        chordwise_panels=spec.chordwise_panels,
        spanwise_panels=spec.spanwise_panels,
        device=device,
    )
    fsi = None
    if spec.flexibility.enabled:
        struct = build_wing_structural_model(kin, spec.flexibility, fluid_density=spec.rho_kg_m3)
        fsi = TwoWayNewmarkAitkenFSI(kin, struct, dt=dt)

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
    hinge_axis = torch.tensor(
        [math.cos(kin._alpha), 0.0, -math.sin(kin._alpha)],
        device=device,
        dtype=torch.float64,
    )

    owner = stepper.initialize((kin.evaluate(0.0),))
    last_start = total_steps - spc + 1
    sample_offsets = np.linspace(0, spc - 1, min(n_frames, spc), dtype=int)
    sample_steps = {int(last_start + off) for off in sample_offsets}

    t_cyc: list[float] = []
    lift_cyc: list[float] = []
    thrust_cyc: list[float] = []
    p_hinge_cyc: list[float] = []
    phidot_cyc: list[float] = []
    phiddot_cyc: list[float] = []
    phi_deg_cyc: list[float] = []
    q_bend_cyc: list[float] = []
    q_twist_deg_cyc: list[float] = []
    q_camber_cyc: list[float] = []
    wrist_fold_deg_cyc: list[float] = []
    wrist_sweep_deg_cyc: list[float] = []
    semispan_wp_pct_cyc: list[float] = []
    frames: list[dict[str, Any]] = []

    ns = spec.spanwise_panels
    nc = spec.chordwise_panels
    j_slice = int(ns // 2 + round(0.70 * (ns // 2)))
    j_slice = min(max(j_slice, ns // 2 + 1), ns - 1)
    j_wrist_node = (ns // 2) + kin._j_wrist_half if kin._fsf_enabled else -1
    fsf_mode = kin._fsf_mode

    for step in range(1, total_steps + 1):
        t_now = step * dt
        if fsi is not None:
            proposal, frame, pos_grid, f_visc, v_rigid_panel, _ = fsi.step_coupled(
                t_now=t_now,
                stepper=stepper,
                owner=owner,
                v_inf_t=v_inf_t,
                rho=spec.rho_kg_m3,
                cd0_wing=cd0_wing,
                max_subiters=4,
                tol=1.0e-3,
            )
            q_now = fsi.q.copy()
        else:
            pos_grid = kin.positions_at(t_now, (0.0, 0.0, 0.0))
            frame = kin.evaluate(t_now)
            proposal = stepper.propose(owner, (frame,), dt)
            stepper.commit(owner, proposal)
            v_rel = v_inf_t[None, :] - frame.collocation_velocity_I
            v_rel_mag = torch.linalg.vector_norm(v_rel, dim=1, keepdim=True)
            f_visc = 0.5 * spec.rho_kg_m3 * cd0_wing * frame.areas[:, None] * v_rel_mag * v_rel
            q_now = np.zeros(3, dtype=np.float64)
            v_rigid_panel = None

        diag = owner.state.diagnostics[-1]

        if step >= last_start:
            panel_f = proposal.load.panel_forces
            panel_r = frame.collocation_I
            f_total = panel_f + f_visc

            phi, phidot, phiddot = kin.flap_state(t_now)
            gw_rad, sw_rad = kin.fsf_wrist_angles(t_now)
            if v_rigid_panel is not None and bool(spec.fsf.enabled):
                p_inst_val = -float(torch.sum(f_total * v_rigid_panel).item())
            else:
                panel_side = torch.sign(panel_r[:, 1])
                r_cross_f = torch.linalg.cross(panel_r, f_total, dim=1)
                m_hinge = -float(torch.sum(panel_side * torch.sum(r_cross_f * hinge_axis[None, :], dim=1)).item())
                p_inst_val = m_hinge * phidot
            tot_invisc = proposal.load.total_force
            tot_visc = torch.sum(f_visc, dim=0)
            lift_val = float(tot_invisc[2].item() + tot_visc[2].item())
            wing_t_val = -float(tot_invisc[0].item()) - float(tot_visc[0].item())

            if fsf_mode == "robofalcon2_eqs1":
                rf_st = kin.robofalcon2_eqs1_state(t_now)
                y_ss_m = rf_st["semispan_wing_plane_m"]
                semispan_pct = 100.0 * y_ss_m / max(0.5 * spec.span_m, 1.0e-9)
            else:
                y_ss_m = 0.5 * spec.span_m
                semispan_pct = 100.0

            t_cyc.append(t_now)
            lift_cyc.append(lift_val)
            thrust_cyc.append(wing_t_val)
            p_hinge_cyc.append(p_inst_val)
            phidot_cyc.append(phidot)
            phiddot_cyc.append(phiddot)
            phi_deg_cyc.append(math.degrees(phi))
            q_bend_cyc.append(float(q_now[0]) * 100.0)
            q_twist_deg_cyc.append(math.degrees(float(q_now[1])))
            q_camber_cyc.append(float(q_now[2]) * 100.0)
            wrist_fold_deg_cyc.append(math.degrees(gw_rad))
            wrist_sweep_deg_cyc.append(math.degrees(sw_rad))
            semispan_wp_pct_cyc.append(semispan_pct)

            if step in sample_steps:
                st = owner.state
                pos_np = _to_np(pos_grid)
                strip_chord_xz = 0.5 * (pos_np[:, j_slice, :] + pos_np[:, j_slice + 1, :])
                le_np = pos_np[0]
                te_np = pos_np[-1]
                span_eff = float(np.max(le_np[:, 1]) - np.min(le_np[:, 1]))
                skel = kin.robofalcon_skeleton_3d(t_now, q_now)

                wake_r = _to_np(st.wake_rings)
                wake_g = _to_np(st.wake_gamma)
                n_wrows = wake_r.shape[0] // ns
                if n_wrows > 0:
                    wr_grid = wake_r.reshape(n_wrows, ns, 4, 3)
                    wg_grid = wake_g.reshape(n_wrows, ns)
                    tev_slice_pts = wr_grid[:, j_slice].mean(axis=1)
                    tev_slice_g = wg_grid[:, j_slice]
                else:
                    tev_slice_pts = np.zeros((0, 3))
                    tev_slice_g = np.zeros((0,))

                if st.lev_chain is not None and bool((st.lev_chain.count > 0).any().item()):
                    ch = st.lev_chain
                    lr_t, lg_t = lev_hirato.flat_live(ch)
                    lev_r = _to_np(lr_t)
                    lev_g = _to_np(lg_t)
                    g_grid = _to_np(ch.gamma)
                    cnt_np = _to_np(ch.count)
                    dg_grid = np.zeros_like(g_grid)
                    for j_col in range(ns):
                        c_j = int(cnt_np[j_col])
                        if c_j > 1:
                            dg_grid[: c_j - 1, j_col] = g_grid[: c_j - 1, j_col] - g_grid[1:c_j, j_col]
                            dg_grid[c_j - 1, j_col] = g_grid[c_j - 1, j_col]
                        elif c_j == 1:
                            dg_grid[0, j_col] = g_grid[0, j_col]
                    live_mask_np = _to_np(lev_hirato.active_rows(ch))
                    lev_dg = dg_grid.reshape(-1)[live_mask_np]
                    pv_mean = float(ch.prev.mean().item())
                    cnt_j = int(cnt_np[j_slice])
                    if cnt_j > 0:
                        lev_slice_r = _to_np(ch.rings[:cnt_j, j_slice])
                        lev_slice_pts = lev_slice_r.mean(axis=1)
                        lev_slice_g = g_grid[:cnt_j, j_slice].copy()
                        lev_slice_dg = dg_grid[:cnt_j, j_slice].copy()
                    else:
                        lev_slice_pts = np.zeros((0, 3))
                        lev_slice_g = np.zeros((0,))
                        lev_slice_dg = np.zeros((0,))
                else:
                    lev_r = np.zeros((0, 4, 3))
                    lev_g = np.zeros((0,))
                    lev_dg = np.zeros((0,))
                    pv_mean = 0.0
                    lev_slice_pts = np.zeros((0, 3))
                    lev_slice_g = np.zeros((0,))
                    lev_slice_dg = np.zeros((0,))

                frames.append(
                    {
                        "cyc_idx": step - last_start,
                        "phase": (step - last_start) / float(spc),
                        "phi_deg": math.degrees(phi),
                        "phidot_rad_s": phidot,
                        "wrist_fold_deg": math.degrees(gw_rad),
                        "wrist_sweep_deg": math.degrees(sw_rad),
                        "semispan_wp_m": y_ss_m,
                        "q_bend_pct": float(q_now[0]) * 100.0,
                        "q_twist_deg": math.degrees(float(q_now[1])),
                        "q_camber_pct": float(q_now[2]) * 100.0,
                        "span_eff_m": span_eff,
                        "skeleton_3d": skel,
                        "bound_rings": _to_np(frame.panel_rings_I),
                        "gamma_bound": _to_np(st.gamma_bound),
                        "le_line": le_np,
                        "te_line": te_np,
                        "pos_grid": pos_np,
                        "wake_rings": wake_r,
                        "wake_gamma": wake_g,
                        "lev_rings": lev_r,
                        "lev_gamma": lev_g,
                        "lev_dgamma": lev_dg,
                        "pv_mean": pv_mean,
                        "strip_chord_xyz": strip_chord_xz,
                        "tev_slice_pts": tev_slice_pts,
                        "tev_slice_g": tev_slice_g,
                        "lev_slice_pts": lev_slice_pts,
                        "lev_slice_g": lev_slice_g,
                        "lev_slice_dg": lev_slice_dg,
                        "sep_strips": int(diag.get("separated_strip_count", 0)),
                        "max_lesp": float(diag.get("lesp_pre_max_abs", 0.0)),
                        "neumann_res": float(diag.get("full_surface_neumann_max_abs", 0.0)),
                    }
                )

    q_s = max(q_dyn * spec.area_m2, 1.0e-9)
    mean_wing_lift = float(np.mean(lift_cyc))
    tail_info = build_tail_geometry_and_aero(
        spec.tail,
        alpha_deg=alpha_deg,
        airspeed_m_s=u_inf,
        rho=spec.rho_kg_m3,
        wing_cl_mean=mean_wing_lift / q_s,
        wing_ar=spec.aspect_ratio,
    )
    lift_arr = np.asarray(lift_cyc, dtype=np.float64) + float(tail_info["lift_n"])

    t_arr = np.asarray(t_cyc, dtype=np.float64)
    p_arr = np.asarray(p_hinge_cyc, dtype=np.float64)
    phidot_arr = np.asarray(phidot_cyc, dtype=np.float64)
    phiddot_arr = np.asarray(phiddot_cyc, dtype=np.float64)
    power = evaluate_flapping_motor_cycle(
        motor=spec.motor,
        t_s=t_arr,
        flap_rate_rad_s=phidot_arr,
        aero_power_w=p_arr,
        frequency_hz=freq,
        mass_kg=spec.mass_kg,
        cruise_speed_m_s=u_inf,
        wing_inertia_kg_m2=spec.wing_inertia_kg_m2,
        flap_accel_rad_s2=phiddot_arr,
        inertial_elastic_recovery=spec.inertial_elastic_recovery,
    )
    p_load = p_arr + (1.0 - spec.inertial_elastic_recovery) * spec.wing_inertia_kg_m2 * phiddot_arr * phidot_arr
    p_shaft = np.maximum(p_load, 0.0) / max(spec.motor.gear_efficiency, 1e-6)
    omega_m = 2.0 * math.pi * spec.motor.gear_ratio * freq
    tau_m = p_shaft / max(omega_m, 1e-9)
    i_inst = spec.motor.no_load_current_a + tau_m / spec.motor.kt_nm_per_a
    v_inst = spec.motor.ke_vs_per_rad * omega_m + spec.motor.resistance_ohm * i_inst
    p_elec_trace = (v_inst * i_inst) / max(spec.motor.esc_efficiency, 1e-6) + spec.motor.avionics_power_w

    return {
        "aircraft_id": spec.aircraft_id,
        "name": spec.name,
        "span_m": spec.span_m,
        "area_m2": spec.area_m2,
        "mean_chord_m": spec.mean_chord_m,
        "mass_kg": spec.mass_kg,
        "weight_n": spec.weight_n,
        "body_drag_n": d_body_tail,
        "airspeed_m_s": u_inf,
        "alpha_deg": alpha_deg,
        "frequency_hz": freq,
        "fsf_enabled": bool(spec.fsf.enabled),
        "fsf_mode": fsf_mode,
        "flex_enabled": bool(spec.flexibility.enabled),
        "tail_info": tail_info,
        "j_wrist_node": j_wrist_node,
        "nc": nc,
        "ns": ns,
        "spc": spc,
        "j_slice": j_slice,
        "slice_eta": (j_slice + 0.5 - ns / 2.0) / (ns / 2.0),
        "phase_trace": np.arange(spc, dtype=np.float64) / float(spc),
        "lift_trace": lift_arr,
        "thrust_trace": np.asarray(thrust_cyc, dtype=np.float64),
        "p_elec_trace": p_elec_trace,
        "phi_deg_trace": np.asarray(phi_deg_cyc, dtype=np.float64),
        "q_bend_trace": np.asarray(q_bend_cyc, dtype=np.float64),
        "q_twist_deg_trace": np.asarray(q_twist_deg_cyc, dtype=np.float64),
        "q_camber_trace": np.asarray(q_camber_cyc, dtype=np.float64),
        "wrist_fold_deg_trace": np.asarray(wrist_fold_deg_cyc, dtype=np.float64),
        "wrist_sweep_deg_trace": np.asarray(wrist_sweep_deg_cyc, dtype=np.float64),
        "semispan_wp_pct_trace": np.asarray(semispan_wp_pct_cyc, dtype=np.float64),
        "mean_elec_power_w": power.total_elec_power_w,
        "frames": frames,
    }


def _compute_global_bounds(data: dict[str, Any]) -> dict[str, Any]:
    """Compute fixed physical axis bounds and color limits across the entire flapping cycle."""

    all_x, all_y, all_z = [], [], []
    wing_x, wing_y, wing_z = [], [], []
    wg_abs, lg_abs, ldg_abs = [], [], []
    slice_x, slice_z = [], []
    for fr in data["frames"]:
        pg = fr["pos_grid"].reshape(-1, 3)
        wing_x.append(pg[:, 0])
        wing_y.append(pg[:, 1])
        wing_z.append(pg[:, 2])
        for arr in (fr["bound_rings"], fr["wake_rings"], fr["lev_rings"]):
            if arr.shape[0] > 0:
                pts = arr.reshape(-1, 3)
                all_x.append(pts[:, 0])
                all_y.append(pts[:, 1])
                all_z.append(pts[:, 2])
        if fr["wake_gamma"].size > 0:
            wg_abs.append(np.abs(fr["wake_gamma"]))
        if fr["lev_gamma"].size > 0:
            lg_abs.append(np.abs(fr["lev_gamma"]))
        if fr["lev_dgamma"].size > 0:
            nz = np.abs(fr["lev_dgamma"])
            nz = nz[nz > 1e-8]
            if nz.size > 0:
                ldg_abs.append(nz)
        slice_x.append(fr["strip_chord_xyz"][:, 0])
        slice_z.append(fr["strip_chord_xyz"][:, 2])
        if fr["tev_slice_pts"].shape[0] > 0:
            slice_x.append(fr["tev_slice_pts"][:, 0])
            slice_z.append(fr["tev_slice_pts"][:, 2])
        if fr["lev_slice_pts"].shape[0] > 0:
            slice_x.append(fr["lev_slice_pts"][:, 0])
            slice_z.append(fr["lev_slice_pts"][:, 2])

    ti = data["tail_info"]
    if ti["h_quads"].shape[0] > 0:
        tpts = ti["h_quads"].reshape(-1, 3)
        wing_x.append(tpts[:, 0])
        wing_y.append(tpts[:, 1])
        wing_z.append(tpts[:, 2])

    b = data["span_m"]
    c = data["mean_chord_m"]
    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)
    zs = np.concatenate(all_z)

    x_min = min(float(np.percentile(xs, 0.5)), -0.35 * c)
    x_max = min(float(np.percentile(xs, 99.5)) + 0.15 * c, max(2.1 * b, 5.2 * c))
    y_lim = max(0.60 * b, float(np.max(np.abs(ys))) * 1.06)
    z_lo = float(np.percentile(zs, 0.5)) - 0.12 * b
    z_hi = float(np.percentile(zs, 99.5)) + 0.12 * b
    z_span = max(z_hi - z_lo, 0.72 * b)
    z_mid = 0.5 * (z_lo + z_hi)

    wx = np.concatenate(wing_x)
    wy = np.concatenate(wing_y)
    wz = np.concatenate(wing_z)
    wy_lim = max(0.58 * b, float(np.max(np.abs(wy))) * 1.12)
    wz_lim = max(0.45 * b, float(np.max(np.abs(wz))) * 1.18)

    w_vmax = float(np.percentile(np.concatenate(wg_abs), 97)) if wg_abs else 1.0
    l_vmax = float(np.percentile(np.concatenate(lg_abs), 97)) if lg_abs else 1.0
    l_dg_vmax = float(np.percentile(np.concatenate(ldg_abs), 95)) if ldg_abs else max(0.1 * l_vmax, 1.0e-4)

    sx = np.concatenate(slice_x)
    sz = np.concatenate(slice_z)
    sx_min = min(float(np.min(sx)), -0.35 * c) - 0.15 * c
    sx_max = min(float(np.percentile(sx, 98.5)) + 0.25 * c, x_max)
    sz_min = float(np.min(sz)) - 0.45 * c
    sz_max = float(np.max(sz)) + 0.45 * c

    return {
        "xlim": (x_min, x_max),
        "ylim": (-y_lim, y_lim),
        "zlim": (z_mid - 0.5 * z_span, z_mid + 0.5 * z_span),
        "wing_xlim": (float(np.min(wx)) - 0.22 * c, float(np.max(wx)) + 0.22 * c),
        "wing_ylim": (-wy_lim, wy_lim),
        "wing_zlim": (-wz_lim, wz_lim),
        "w_vmax": max(w_vmax, 1.0e-4),
        "l_vmax": max(l_vmax, 1.0e-4),
        "l_dg_vmax": max(l_dg_vmax, 1.0e-5),
        "sxlim": (sx_min, sx_max),
        "szlim": (sz_min, sz_max),
    }


def _wing_quads_from_grid(pos_grid: np.ndarray) -> np.ndarray:
    rows = [0] + list(range(2, pos_grid.shape[0] - 1)) + [pos_grid.shape[0] - 1]
    g = pos_grid[rows]
    return np.stack([g[:-1, :-1], g[:-1, 1:], g[1:, 1:], g[1:, :-1]], axis=2).reshape(-1, 4, 3)


def _draw_tail_3d(ax, tail_info: dict[str, Any]) -> None:
    if tail_info["boom_xyz"].shape[0] >= 2:
        bxyz = tail_info["boom_xyz"]
        ax.plot(bxyz[:, 0], bxyz[:, 1], bxyz[:, 2], color="#1e293b", lw=2.8, zorder=9)
    if tail_info["h_quads"].shape[0] > 0:
        ax.add_collection3d(
            Poly3DCollection(
                tail_info["h_quads"],
                facecolors=(0.24, 0.34, 0.48, 0.88),
                edgecolors=(0.82, 0.88, 0.96, 0.80),
                linewidths=0.6,
            )
        )
    if tail_info["v_quads"].shape[0] > 0:
        ax.add_collection3d(
            Poly3DCollection(
                tail_info["v_quads"],
                facecolors=(0.28, 0.38, 0.52, 0.88),
                edgecolors=(0.82, 0.88, 0.96, 0.80),
                linewidths=0.6,
            )
        )


def _draw_rf2_skeleton_2d(ax, skel: dict[str, np.ndarray], c0: int, c1: int, show_labels: bool = True) -> None:
    """Draw exact color-coded RoboFalcon 2.0 Humerus/Radius/Manus/Fingers skeleton on a 2D projection."""

    for s_idx in (0, 1):
        hum = skel["humerus"][s_idx]
        rad = skel["radius"][s_idx]
        man = skel["manus"][s_idx]
        ax.plot(hum[:, c0], hum[:, c1], color="#dc2626", lw=3.0, zorder=10, label="Humerus (110mm)" if (show_labels and s_idx == 1) else None)
        ax.plot(rad[:, c0], rad[:, c1], color="#16a34a", lw=3.0, zorder=10, label="Radius (180mm)" if (show_labels and s_idx == 1) else None)
        ax.plot(man[:, c0], man[:, c1], color="#2563eb", lw=3.0, zorder=10, label="Manus (325mm)" if (show_labels and s_idx == 1) else None)
        for f_idx, fg in enumerate(skel["fingers"][s_idx]):
            ax.plot(fg[:, c0], fg[:, c1], color="#9333ea", lw=1.8, zorder=9, label="Fingers" if (show_labels and s_idx == 1 and f_idx == 0) else None)
        jts = skel["joints"][s_idx]
        # Highlight Elbow (green/red) & Wrist (bright red)
        ax.scatter([jts[1, c0]], [jts[1, c1]], color="#16a34a", edgecolor="#ffffff", s=38, zorder=12)
        ax.scatter([jts[2, c0]], [jts[2, c1]], color="#ef4444", edgecolor="#ffffff", s=58, zorder=12)


def _draw_rf2_skeleton_3d(ax, skel: dict[str, np.ndarray]) -> None:
    for s_idx in (0, 1):
        hum = skel["humerus"][s_idx]
        rad = skel["radius"][s_idx]
        man = skel["manus"][s_idx]
        ax.plot(hum[:, 0], hum[:, 1], hum[:, 2], color="#dc2626", lw=3.0, zorder=12)
        ax.plot(rad[:, 0], rad[:, 1], rad[:, 2], color="#16a34a", lw=3.0, zorder=12)
        ax.plot(man[:, 0], man[:, 1], man[:, 2], color="#2563eb", lw=3.0, zorder=12)
        for fg in skel["fingers"][s_idx]:
            ax.plot(fg[:, 0], fg[:, 1], fg[:, 2], color="#9333ea", lw=1.7, zorder=11)
        jts = skel["joints"][s_idx]
        ax.scatter([jts[2, 0]], [jts[2, 1]], [jts[2, 2]], color="#ef4444", edgecolor="#ffffff", s=48, zorder=13)


def _draw_3d_scene(
    ax,
    fr: dict[str, Any],
    data: dict[str, Any],
    bounds: dict[str, Any],
    *,
    elev: float,
    azim: float,
    title: str,
    show_legend: bool = False,
) -> None:
    """Draw 3D wing + fuselage + exact tail + connected TEV + clean Hirato v2 LEV."""

    xlim, ylim, zlim = bounds["xlim"], bounds["ylim"], bounds["zlim"]
    x_max = xlim[1]
    cmap_tev = plt.get_cmap("coolwarm")
    cmap_lev = plt.get_cmap("autumn_r")
    norm_tev = mcolors.TwoSlopeNorm(vmin=-bounds["w_vmax"], vcenter=0.0, vmax=bounds["w_vmax"])
    norm_ldg = mcolors.Normalize(vmin=0.0, vmax=bounds["l_dg_vmax"])

    # 1. TEV connected wake sheet
    wr, wg = fr["wake_rings"], fr["wake_gamma"]
    if wr.shape[0] > 0:
        keep = wr[:, :, 0].mean(axis=1) <= x_max + 0.05 * (x_max - xlim[0])
        if np.any(keep):
            fc = cmap_tev(norm_tev(wg[keep]))
            fc[:, 3] = 0.46
            ax.add_collection3d(
                Poly3DCollection(wr[keep], facecolors=fc, edgecolors=(0.18, 0.26, 0.42, 0.24), linewidths=0.30)
            )

    # 2. Clean Hirato v2 LEV sheet (honestly showing active cores + translucent held connectors)
    lr, lg, ldg = fr["lev_rings"], fr["lev_gamma"], fr["lev_dgamma"]
    if lr.shape[0] > 0:
        in_box = (lr[:, :, 0].mean(axis=1) <= x_max + 0.05 * (x_max - xlim[0])) & (np.abs(lg) > 1e-6 * bounds["l_vmax"])
        dg_abs = np.abs(ldg)
        thr = 0.05 * bounds["l_dg_vmax"]
        held_mask = in_box & (dg_abs < thr)
        core_mask = in_box & (dg_abs >= thr)
        if np.any(held_mask):
            ax.add_collection3d(
                Poly3DCollection(
                    lr[held_mask],
                    facecolors=(0.98, 0.58, 0.22, 0.08),
                    edgecolors=(0.82, 0.30, 0.05, 0.22),
                    linewidths=0.28,
                )
            )
        if np.any(core_mask):
            lr_c, dg_c = lr[core_mask], dg_abs[core_mask]
            lev_c = cmap_lev(norm_ldg(dg_c))
            lev_c[:, 3] = 0.34 + 0.52 * np.clip(dg_c / bounds["l_dg_vmax"], 0.0, 1.0)
            ax.add_collection3d(
                Poly3DCollection(lr_c, facecolors=lev_c, edgecolors=(0.62, 0.0, 0.45, 0.55), linewidths=0.48)
            )

    # 3. Wing surface + LE spar + skeleton + tail
    w_quads = _wing_quads_from_grid(fr["pos_grid"])
    wing_alpha = 0.72 if fr["skeleton_3d"] is not None else 0.90
    ax.add_collection3d(
        Poly3DCollection(
            w_quads,
            facecolors=(0.14, 0.20, 0.30, wing_alpha),
            edgecolors=(0.82, 0.88, 0.96, 0.78),
            linewidths=0.58,
        )
    )
    le, te = fr["le_line"], fr["te_line"]
    ax.plot(le[:, 0], le[:, 1], le[:, 2], color="#f59e0b", linewidth=2.2, zorder=10)
    ax.plot(te[:, 0], te[:, 1], te[:, 2], color="#06b6d4", linewidth=1.3, zorder=9)
    sc = fr["strip_chord_xyz"]
    ax.plot(sc[:, 0], sc[:, 1], sc[:, 2], color="#22c55e", linewidth=2.4, zorder=11)

    if fr["skeleton_3d"] is not None:
        _draw_rf2_skeleton_3d(ax, fr["skeleton_3d"])
    elif data["fsf_enabled"] and data["j_wrist_node"] > 0:
        jw_r = data["j_wrist_node"]
        jw_l = data["ns"] - jw_r
        ax.scatter(
            [le[jw_r, 0], le[jw_l, 0]],
            [le[jw_r, 1], le[jw_l, 1]],
            [le[jw_r, 2], le[jw_l, 2]],
            color="#ef4444", edgecolor="#ffffff", s=46, zorder=12,
        )

    _draw_tail_3d(ax, data["tail_info"])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect((max(xlim[1] - xlim[0], 1e-4), max(ylim[1] - ylim[0], 1e-4), max(zlim[1] - zlim[0], 1e-4)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [m] (flow →)", fontsize=8, labelpad=2)
    ax.set_ylabel("y [m] (span)", fontsize=8, labelpad=2)
    ax.set_zlabel("z [m] (lift ↑)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9.0, fontweight="bold", pad=4)
    ax.xaxis.pane.set_facecolor((0.95, 0.97, 0.99, 0.9))
    ax.yaxis.pane.set_facecolor((0.95, 0.97, 0.99, 0.9))
    ax.zaxis.pane.set_facecolor((0.92, 0.95, 0.98, 0.9))

    if show_legend:
        handles = [
            Patch(facecolor=(0.14, 0.20, 0.30, 0.85), edgecolor="#cbd5e1", label="Exact Wing + Tail"),
            Patch(facecolor="#f97316", edgecolor="#a21caf", alpha=0.80, label="LEV Cores (|ΔΓ|>0)"),
            Patch(facecolor=(0.98, 0.58, 0.22, 0.16), edgecolor=(0.82, 0.30, 0.05, 0.45), label="LEV Held Sheet"),
            Patch(facecolor="#3b82f6", edgecolor="#dc2626", alpha=0.55, label="TEV Wake (±Γ)"),
            Line2D([0], [0], color="#f59e0b", lw=2.2, label="LE Spar"),
            Line2D([0], [0], color="#22c55e", lw=2.2, label="2D Slice"),
        ]
        if fr["skeleton_3d"] is not None:
            handles.extend([
                Line2D([0], [0], color="#dc2626", lw=2.5, label="Humerus (Red)"),
                Line2D([0], [0], color="#16a34a", lw=2.5, label="Radius (Green)"),
                Line2D([0], [0], color="#2563eb", lw=2.5, label="Manus (Blue)"),
            ])
        ax.legend(handles=handles, loc="upper right", fontsize=6.2, framealpha=0.92)


def _draw_front_geometry_view(ax, fr: dict[str, Any], data: dict[str, Any], bounds: dict[str, Any], show_legend: bool = True) -> None:
    """Pure unobstructed Front View (y-z plane): flapping stroke, span contraction & spar bending."""

    ax.set_facecolor("#ffffff")
    pos = fr["pos_grid"]
    le, te = fr["le_line"], fr["te_line"]
    w_quads = _wing_quads_from_grid(pos)
    for q in w_quads:
        ax.add_patch(Polygon(q[:, [1, 2]], closed=True, facecolor="#334155", edgecolor="#94a3b8", lw=0.45, alpha=0.48))
    ti = data["tail_info"]
    for q in ti["h_quads"]:
        ax.add_patch(Polygon(q[:, [1, 2]], closed=True, facecolor="#64748b", edgecolor="#cbd5e1", lw=0.45, alpha=0.50))
    for q in ti["v_quads"]:
        ax.add_patch(Polygon(q[:, [1, 2]], closed=True, facecolor="#475569", edgecolor="#cbd5e1", lw=0.50, alpha=0.65))

    half_b = 0.5 * data["span_m"]
    ax.plot([-half_b, half_b], [0.0, 0.0], color="#cbd5e1", ls="--", lw=1.1, label="Full span ref")
    ax.plot(le[:, 1], le[:, 2], color="#f59e0b", lw=2.6, zorder=6, label="LE Carbon Spar")
    ax.plot(te[:, 1], te[:, 2], color="#06b6d4", lw=1.5, zorder=5, label="Trailing Edge")

    if fr["skeleton_3d"] is not None:
        _draw_rf2_skeleton_2d(ax, fr["skeleton_3d"], 1, 2, show_labels=show_legend)

    ax.set_xlim(bounds["wing_ylim"])
    ax.set_ylim(bounds["wing_zlim"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_xlabel("y [m] (span)", fontsize=8)
    ax.set_ylabel("z [m] (vertical ↑)", fontsize=8)
    ax.tick_params(labelsize=7)
    if data["fsf_mode"] == "robofalcon2_eqs1":
        ax.set_title(
            f"Front (y-z): Θ={fr['phi_deg']:+.1f}° | y_ss={fr['semispan_wp_m']*1000:.0f} mm | b_y={fr['span_eff_m']:.2f} m",
            fontsize=8.5, fontweight="bold",
        )
    else:
        ax.set_title(
            f"Front (y-z): φ={fr['phi_deg']:+.1f}° | Spar Bend={fr['q_bend_pct']:+.1f}% b/2",
            fontsize=8.5, fontweight="bold",
        )
    if show_legend:
        ax.legend(loc="upper right", fontsize=6.0, framealpha=0.9)


def _draw_top_geometry_view(ax, fr: dict[str, Any], data: dict[str, Any], bounds: dict[str, Any], show_legend: bool = True) -> None:
    """Pure unobstructed Top View (x-y plane): exact digitized planform, tail, and in-plane 3-seg FSF folding!"""

    ax.set_facecolor("#ffffff")
    pos = fr["pos_grid"]
    le, te = fr["le_line"], fr["te_line"]
    w_quads = _wing_quads_from_grid(pos)
    wing_alpha = 0.42 if fr["skeleton_3d"] is not None else 0.78
    for q in w_quads:
        ax.add_patch(Polygon(q[:, [0, 1]], closed=True, facecolor="#1e3a5f", edgecolor="#94a3b8", lw=0.45, alpha=wing_alpha))
    ti = data["tail_info"]
    if ti["boom_xyz"].shape[0] >= 2:
        ax.plot(ti["boom_xyz"][:, 0], ti["boom_xyz"][:, 1], color="#0f172a", lw=2.6, zorder=5)
    for q in ti["h_quads"]:
        ax.add_patch(Polygon(q[:, [0, 1]], closed=True, facecolor="#475569", edgecolor="#e2e8f0", lw=0.50, alpha=0.82))

    ax.plot(le[:, 0], le[:, 1], color="#f59e0b", lw=2.3, zorder=7)
    ax.plot(te[:, 0], te[:, 1], color="#06b6d4", lw=1.4, zorder=6)
    sc = fr["strip_chord_xyz"]
    ax.plot(sc[:, 0], sc[:, 1], color="#22c55e", lw=2.2, zorder=8)

    if fr["skeleton_3d"] is not None:
        _draw_rf2_skeleton_2d(ax, fr["skeleton_3d"], 0, 1, show_labels=show_legend)

    ax.set_xlim(bounds["wing_xlim"])
    ax.set_ylim(bounds["wing_ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_xlabel("x [m] (nose → tail)", fontsize=8)
    ax.set_ylabel("y [m] (span)", fontsize=8)
    ax.tick_params(labelsize=7)
    if data["fsf_mode"] == "robofalcon2_eqs1":
        ax.set_title(
            f"Top (x-y) In-Plane FSF: γ={fr['wrist_fold_deg']:.1f}° | Ψ={fr['wrist_sweep_deg']:.1f}° | y_ss={fr['semispan_wp_m']*1000:.0f} mm",
            fontsize=8.3, fontweight="bold",
        )
        if show_legend:
            ax.legend(loc="lower right", fontsize=6.0, framealpha=0.92)
    else:
        ax.set_title(
            f"Top (x-y): b={data['span_m']:.2f} m, S={data['area_m2']:.4f} m² + Tail",
            fontsize=8.5, fontweight="bold",
        )


def _render_single_frame(args: tuple[int, dict[str, Any], dict[str, Any], str]) -> str:
    """Render one complete 6-panel diagnostic PNG frame (multiprocessing worker)."""

    frame_idx, data, bounds, out_path = args
    fr = data["frames"][frame_idx]

    fig = plt.figure(figsize=(17.2, 9.2), dpi=100, facecolor="#f8fafc")
    gs = GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.25, 1.0],
        width_ratios=[1.25, 1.0, 1.0],
        hspace=0.28, wspace=0.22, left=0.04, right=0.97, top=0.91, bottom=0.06,
    )

    # Row 0, Col 0: 3D Isometric Perspective View
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_3d_scene(
        ax3d, fr, data, bounds, elev=24, azim=-128,
        title="3D Flow: Exact Wing + Tail + Rebased LEV + TEV",
        show_legend=True,
    )

    # Row 0, Col 1: Pure Front View (y-z)
    ax_front = fig.add_subplot(gs[0, 1])
    _draw_front_geometry_view(ax_front, fr, data, bounds, show_legend=True)

    # Row 0, Col 2: Pure Top View (x-y) — unmistakable in-plane 3-segment FSF & exact planform!
    ax_top = fig.add_subplot(gs[0, 2])
    _draw_top_geometry_view(ax_top, fr, data, bounds, show_legend=True)

    # Row 1, Col 0: 2D Side-View Cross-Section
    ax2d = fig.add_subplot(gs[1, 0])
    ax2d.set_facecolor("#ffffff")
    tev_pts, tev_g = fr["tev_slice_pts"], fr["tev_slice_g"]
    if tev_pts.shape[0] > 0:
        ax2d.plot(tev_pts[:, 0], tev_pts[:, 2], color="#64748b", lw=1.1, alpha=0.75, zorder=2, label="TEV wake")
        ax2d.scatter(
            tev_pts[:, 0], tev_pts[:, 2],
            c=tev_g, cmap="coolwarm", vmin=-bounds["w_vmax"], vmax=bounds["w_vmax"],
            s=16, edgecolors="none", alpha=0.88, zorder=3,
        )
    lev_pts, lev_dg = fr["lev_slice_pts"], fr["lev_slice_dg"]
    if lev_pts.shape[0] > 0:
        ax2d.plot(lev_pts[:, 0], lev_pts[:, 2], color="#c026d3", lw=1.0, alpha=0.60, zorder=4, label="LEV chain")
        dg_abs = np.abs(lev_dg)
        dot_sz = 9.0 + 40.0 * np.clip(dg_abs / bounds["l_dg_vmax"], 0.0, 1.0)
        ax2d.scatter(
            lev_pts[:, 0], lev_pts[:, 2],
            c=dg_abs, cmap="autumn_r", vmin=0.0, vmax=bounds["l_dg_vmax"],
            s=dot_sz, edgecolors="#86198f", linewidths=0.45, alpha=0.92, zorder=5,
        )
    chord_xyz = fr["strip_chord_xyz"]
    ax2d.plot(chord_xyz[:, 0], chord_xyz[:, 2], color="#0f172a", lw=3.0, zorder=6, label="Flexible airfoil")
    ax2d.scatter([chord_xyz[0, 0]], [chord_xyz[0, 2]], color="#f59e0b", edgecolors="#0f172a", s=50, zorder=7)
    ax2d.scatter([chord_xyz[-1, 0]], [chord_xyz[-1, 2]], color="#06b6d4", edgecolors="#0f172a", marker="s", s=36, zorder=7)
    ax2d.set_xlim(bounds["sxlim"])
    ax2d.set_ylim(bounds["szlim"])
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, ls="--", alpha=0.35)
    ax2d.set_xlabel("x [m] (downstream →)", fontsize=8)
    ax2d.set_ylabel("z [m] (vertical ↑)", fontsize=8)
    ax2d.tick_params(labelsize=7.5)
    ax2d.set_title(
        f"2D Slice (η={data['slice_eta']:.2f}) | Γ_pv={fr['pv_mean']:+.2f} | camb={fr['q_camber_pct']:+.1f}%c",
        fontsize=8.8, fontweight="bold",
    )
    ax2d.legend(loc="upper right", fontsize=6.8, framealpha=0.9)

    # Row 1, Col 1: 2-Way FSI Flexible Deformation & Eq. S1 FSF Kinematics Trace
    ax_fsi = fig.add_subplot(gs[1, 1])
    ax_fsi.set_facecolor("#ffffff")
    ph = data["phase_trace"]
    ax_fsi.plot(ph, data["q_twist_deg_trace"], color="#7c3aed", lw=1.8, label="FSI Tip Twist θ [deg]")
    if data["fsf_mode"] == "robofalcon2_eqs1":
        ax_fsi.plot(ph, data["wrist_fold_deg_trace"], color="#2563eb", lw=2.0, ls="-", label="Manus Fold γ [deg]")
        ax_fsi.plot(ph, data["wrist_sweep_deg_trace"], color="#f59e0b", lw=1.6, ls="--", label="Wing Sweep Ψ [deg]")
    elif data["fsf_enabled"]:
        ax_fsi.plot(ph, data["wrist_fold_deg_trace"], color="#ef4444", lw=2.0, ls="-", label="FSF Fold [deg]")
        ax_fsi.plot(ph, data["wrist_sweep_deg_trace"], color="#f59e0b", lw=1.4, ls="--", label="FSF Sweep [deg]")
    ax_fsi.axvline(fr["phase"], color="#111827", lw=1.5, ls="-", alpha=0.8)
    ax_fsi.axhline(0.0, color="#cbd5e1", lw=0.8)
    ax_fsi.set_xlim(0.0, 1.0)
    ax_fsi.set_xlabel("Cycle Phase t / T", fontsize=8)
    ax_fsi.set_ylabel("Angle [deg]", fontsize=8, color="#4c1d95")
    ax_fsi.tick_params(labelsize=7.5)
    ax_fsi.grid(True, ls="--", alpha=0.35)

    ax_fsi2 = ax_fsi.twinx()
    ax_fsi2.plot(ph, data["q_bend_trace"], color="#0284c7", lw=1.6, ls="-.", label="Spar Bend [% b/2]")
    ax_fsi2.plot(ph, data["q_camber_trace"], color="#059669", lw=1.6, ls=":", label="Camber [% c]")
    if data["fsf_mode"] == "robofalcon2_eqs1":
        ax_fsi2.plot(ph, data["semispan_wp_pct_trace"], color="#16a34a", lw=2.0, ls="-", label="Semispan y_ss [%]")
        ax_fsi2.set_ylabel("Semispan [%] / Bend / Camber", fontsize=7.8, color="#15803d")
    else:
        ax_fsi2.set_ylabel("Bend [% b/2] / Camber [% c]", fontsize=8, color="#0369a1")
    ax_fsi2.tick_params(labelsize=7.5, labelcolor="#0369a1")
    h1, l1 = ax_fsi.get_legend_handles_labels()
    h2, l2 = ax_fsi2.get_legend_handles_labels()
    ax_fsi.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.0, framealpha=0.9)
    ax_fsi.set_title("2-Way GPU FSI Wing Flexibility & FSF Kinematics", fontsize=8.8, fontweight="bold")

    # Row 1, Col 2: Instantaneous Lift, Thrust & 1st-Order Motor Power
    ax_ts = fig.add_subplot(gs[1, 2])
    ax_ts.set_facecolor("#ffffff")
    ax_ts.plot(ph, data["lift_trace"], color="#2563eb", lw=1.7, label=f"Lift [{np.mean(data['lift_trace']):.2f} N]")
    ax_ts.axhline(data["weight_n"], color="#2563eb", lw=0.9, ls="--", alpha=0.65, label=f"mg={data['weight_n']:.2f} N")
    ax_ts.plot(ph, data["thrust_trace"], color="#16a34a", lw=1.5, label=f"Thrust [{np.mean(data['thrust_trace']):.2f} N]")
    ax_ts.axhline(data["body_drag_n"], color="#16a34a", lw=0.9, ls="--", alpha=0.65, label=f"D_body={data['body_drag_n']:.2f} N")
    ax_ts.axvline(fr["phase"], color="#ef4444", lw=1.6, alpha=0.85)
    ax_ts.set_xlim(0.0, 1.0)
    ax_ts.set_xlabel("Cycle Phase t / T", fontsize=8)
    ax_ts.set_ylabel("Force [N]", fontsize=8)
    ax_ts.tick_params(labelsize=7.5)
    ax_ts.grid(True, ls="--", alpha=0.35)

    ax_pw = ax_ts.twinx()
    ax_pw.plot(ph, data["p_elec_trace"], color="#ea580c", lw=1.5, ls="-.", label=f"P_elec [{data['mean_elec_power_w']:.1f} W]")
    ax_pw.set_ylabel("Motor Power [W]", fontsize=8, color="#ea580c")
    ax_pw.tick_params(labelsize=7.5, labelcolor="#ea580c")
    h1, l1 = ax_ts.get_legend_handles_labels()
    h2, l2 = ax_pw.get_legend_handles_labels()
    ax_ts.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.4, framealpha=0.9)
    ax_ts.set_title("Forces & 1st-Order Motor Power", fontsize=8.8, fontweight="bold")

    stroke_str = "DOWNSTROKE" if fr["phidot_rad_s"] < 0.0 else "UPSTROKE"
    fig.suptitle(
        f"FLUXVortex Stage-4 Exact Geometry + 2-Way FSI + Rebased LEV — {data['name']}  |  "
        f"U={data['airspeed_m_s']:.1f} m/s, α={data['alpha_deg']:.2f}°, f={data['frequency_hz']:.2f} Hz, "
        f"t/T={fr['phase']:.2f} ({stroke_str})  |  "
        f"Twist={fr['q_twist_deg']:+.1f}°, Bend={fr['q_bend_pct']:+.1f}%, LESP={fr['max_lesp']:.3f}, Neu={fr['neumann_res']:.1e}",
        fontsize=10.2, fontweight="bold", color="#0f172a", y=0.975,
    )

    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def _render_4phase_contact_sheet(data: dict[str, Any], bounds: dict[str, Any], out_png: Path) -> None:
    """Create a 4-row x 4-column high-res inspection sheet: 3D Flow + Top (x-y) + Front (y-z) + 2D Slice."""

    if data["fsf_mode"] == "robofalcon2_eqs1":
        target_phases = [
            (0.04, "Mid-Downstroke (Unfolded 600mm)"),
            (0.26, "Bottom Reversal (Max Sweep Ψ=24°)"),
            (0.51, "Mid-Upstroke (In-Plane Fold γ=63°)"),
            (0.78, "Top Reversal (Unfolding Θ=+38°)"),
        ]
    else:
        target_phases = [
            (0.50, "Mid-Downstroke (Max Lift)"),
            (0.75, "Bottom Reversal (φ_min)"),
            (0.00, "Mid-Upstroke (Feathered)"),
            (0.25, "Top Reversal (φ_max)"),
        ]

    frames = data["frames"]
    phases = np.array([fr["phase"] for fr in frames])

    fig = plt.figure(figsize=(19.2, 14.8), dpi=120, facecolor="#f8fafc")
    gs = GridSpec(
        4, 4, figure=fig,
        height_ratios=[1.25, 0.95, 0.90, 0.88],
        hspace=0.28, wspace=0.16, left=0.035, right=0.985, top=0.925, bottom=0.04,
    )

    for col, (t_ph, lbl) in enumerate(target_phases):
        idx = int(np.argmin(np.abs(phases - t_ph)))
        fr = frames[idx]

        # Row 0: 3D Flow
        ax3 = fig.add_subplot(gs[0, col], projection="3d")
        _draw_3d_scene(
            ax3, fr, data, bounds, elev=22, azim=-130,
            title=f"{lbl}\nφ={fr['phi_deg']:+.1f}°, b_y={fr['span_eff_m']:.2f}m, sep={fr['sep_strips']}/{data['ns']}",
            show_legend=(col == 0),
        )

        # Row 1: Pure Top View (x-y) — crystal-clear planform, tail, AND RoboFalcon 2.0 in-plane Z-folding!
        axt = fig.add_subplot(gs[1, col])
        _draw_top_geometry_view(axt, fr, data, bounds, show_legend=(col == 0))
        if col > 0:
            axt.set_ylabel("")

        # Row 2: Pure Front View (y-z) — flapping angle, spar bending & span contraction!
        axf = fig.add_subplot(gs[2, col])
        _draw_front_geometry_view(axf, fr, data, bounds, show_legend=(col == 0))
        if col > 0:
            axf.set_ylabel("")

        # Row 3: 2D Side-View Slice (curved E378/GOE225 flexible airfoil + clean LEV/TEV!)
        ax2 = fig.add_subplot(gs[3, col])
        ax2.set_facecolor("#ffffff")
        if fr["tev_slice_pts"].shape[0] > 0:
            ax2.plot(fr["tev_slice_pts"][:, 0], fr["tev_slice_pts"][:, 2], color="#64748b", lw=1.0, alpha=0.75)
            ax2.scatter(
                fr["tev_slice_pts"][:, 0], fr["tev_slice_pts"][:, 2],
                c=fr["tev_slice_g"], cmap="coolwarm", vmin=-bounds["w_vmax"], vmax=bounds["w_vmax"], s=14,
            )
        if fr["lev_slice_pts"].shape[0] > 0:
            ax2.plot(fr["lev_slice_pts"][:, 0], fr["lev_slice_pts"][:, 2], color="#c026d3", lw=0.9, alpha=0.55)
            dg_abs = np.abs(fr["lev_slice_dg"])
            dot_sz = 8.0 + 34.0 * np.clip(dg_abs / bounds["l_dg_vmax"], 0.0, 1.0)
            ax2.scatter(
                fr["lev_slice_pts"][:, 0], fr["lev_slice_pts"][:, 2],
                c=dg_abs, cmap="autumn_r", vmin=0.0, vmax=bounds["l_dg_vmax"],
                s=dot_sz, edgecolors="#86198f", linewidths=0.35, alpha=0.92,
            )
        sc = fr["strip_chord_xyz"]
        ax2.plot(sc[:, 0], sc[:, 2], color="#0f172a", lw=2.8)
        ax2.scatter([sc[0, 0]], [sc[0, 2]], color="#f59e0b", edgecolors="#0f172a", s=42, zorder=7)
        ax2.set_xlim(bounds["sxlim"])
        ax2.set_ylim(bounds["szlim"])
        ax2.set_aspect("equal", adjustable="box")
        ax2.grid(True, ls="--", alpha=0.35)
        ax2.set_xlabel("x [m]", fontsize=8)
        if col == 0:
            ax2.set_ylabel("z [m] (2D slice)", fontsize=8)
        ax2.tick_params(labelsize=7.5)
        ax2.set_title(
            f"2D Slice | Γ_pv={fr['pv_mean']:+.2f} | θ={fr['q_twist_deg']:+.1f}°, camb={fr['q_camber_pct']:+.1f}%c",
            fontsize=8.0,
        )

    fig.suptitle(
        f"{data['name']} — Stage-4 Exact Ground-Truth Geometry + 2-Way FSI + FSF Verification Sheet  "
        f"(U={data['airspeed_m_s']:.1f} m/s, α={data['alpha_deg']:.2f}°, f={data['frequency_hz']:.2f} Hz, "
        f"P_elec={data['mean_elec_power_w']:.1f} W)",
        fontsize=12.5, fontweight="bold", color="#0f172a", y=0.972,
    )
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def encode_mp4_and_gif(frame_paths: list[str], mp4_path: Path, gif_path: Path, fps: int = 20) -> None:
    """Encode 2 looped periodic cycles into universal H.264 MP4 and crisp animated GIF."""

    concat_file = mp4_path.with_suffix(".concat.txt")
    lines = []
    dur = 1.0 / float(fps)
    for _loop in range(2):
        for p in frame_paths:
            lines.append(f"file '{Path(p).resolve()}'")
            lines.append(f"duration {dur:.6f}")
    lines.append(f"file '{Path(frame_paths[-1]).resolve()}'")
    concat_file.write_text("\n".join(lines) + "\n")

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
            str(mp4_path),
        ],
        check=True,
    )

    gif_concat = gif_path.with_suffix(".concat.txt")
    gif_lines = []
    for p in frame_paths:
        gif_lines.append(f"file '{Path(p).resolve()}'")
        gif_lines.append(f"duration {dur:.6f}")
    gif_lines.append(f"file '{Path(frame_paths[-1]).resolve()}'")
    gif_concat.write_text("\n".join(gif_lines) + "\n")

    palette_png = gif_path.with_suffix(".palette.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(gif_concat),
            "-vf", f"fps={fps},scale=1200:-1:flags=lanczos,palettegen=stats_mode=diff",
            str(palette_png),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(gif_concat),
            "-i", str(palette_png),
            "-lavfi", f"fps={fps},scale=1200:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
            str(gif_path),
        ],
        check=True,
    )
    concat_file.unlink(missing_ok=True)
    gif_concat.unlink(missing_ok=True)
    palette_png.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/experiment/20261127_stage4_exact_geom/avian_fsi_cruise_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment/20261127_stage4_exact_geom/animations"),
    )
    parser.add_argument("--retrim", action="store_true", help="Solve fresh 2-way FSI cruise trim before rendering")
    parser.add_argument("--aircraft", default="all", help="comma-separated IDs or 'all'")
    parser.add_argument("--frames", type=int, default=48, help="frames per flapping cycle")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    selected = list(TARGET_AIRCRAFT) if args.aircraft == "all" else [x.strip() for x in args.aircraft.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.retrim or not args.summary_json.exists():
        summary_data: dict[str, Any] = {"aircraft": {}}
        if args.summary_json.exists():
            summary_data = json.loads(args.summary_json.read_text())
        for aid in selected:
            spec = AVIAN_AIRCRAFT_REGISTRY[aid]
            print(f"[FSI Cruise Trim] {aid} ({spec.name}) @ U={spec.cruise_speed_m_s:.1f} m/s ...", flush=True)
            trim = solve_flapping_cruise_trim(
                spec,
                airspeed_m_s=spec.cruise_speed_m_s,
                max_iters=6,
                tol_rel=0.03,
                verify_cycles=3,
                device=args.device,
            )
            op = trim.operating_point
            summary_data["aircraft"][aid] = {
                "name": spec.name,
                "converged": trim.converged,
                "iterations": trim.iterations,
                "lift_error_rel": trim.lift_error_rel,
                "thrust_error_rel": trim.thrust_error_rel,
                "published_power_w": spec.published_cruise.power_w,
                "published_power_range_w": list(spec.published_cruise.power_range_w) if spec.published_cruise.power_range_w else None,
                "trim": {
                    "operating_point": {
                        "airspeed_m_s": op.airspeed_m_s,
                        "alpha_deg": op.alpha_deg,
                        "frequency_hz": op.frequency_hz,
                        "mean_lift_n": op.mean_lift_n,
                        "weight_n": op.weight_n,
                        "net_thrust_n": op.net_thrust_n,
                        "total_elec_power_w": op.power.total_elec_power_w,
                        "cycle_lift_drift_rel": op.cycle_lift_drift_rel,
                        "max_neumann_residual": op.max_neumann_residual,
                    }
                },
            }
            print(
                f"  trimmed: alpha={op.alpha_deg:.2f} deg, f={op.frequency_hz:.2f} Hz | "
                f"L={op.mean_lift_n:.2f}/{op.weight_n:.2f} N ({trim.lift_error_rel*100:+.1f}%), "
                f"Tnet={op.net_thrust_n:+.3f} N | P_elec={op.power.total_elec_power_w:.2f} W "
                f"(pub: {spec.published_cruise.power_w or spec.published_cruise.power_range_w}) | "
                f"drift={op.cycle_lift_drift_rel*100:.1f}%, neu={op.max_neumann_residual:.1e}",
                flush=True,
            )
        args.summary_json.write_text(json.dumps(summary_data, indent=2))
    else:
        summary_data = json.loads(args.summary_json.read_text())

    # Stage 1: Sequential GPU simulation capture for all selected aircraft
    captured: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for aid in selected:
        spec = AVIAN_AIRCRAFT_REGISTRY[aid]
        op = summary_data["aircraft"][aid]["trim"]["operating_point"]
        print(
            f"[GPU Capture] {aid} ({spec.name}): U={op['airspeed_m_s']:.2f} m/s, "
            f"alpha={op['alpha_deg']:.2f} deg, f={op['frequency_hz']:.2f} Hz ...",
            flush=True,
        )
        t_sim = time.perf_counter()
        data = simulate_and_capture_cycle(
            spec,
            alpha_deg=float(op["alpha_deg"]),
            frequency_hz=float(op["frequency_hz"]),
            airspeed_m_s=float(op["airspeed_m_s"]),
            cycles=3,
            n_frames=args.frames,
            device=args.device,
        )
        bounds = _compute_global_bounds(data)
        captured[aid] = (data, bounds)
        bend_mm = 0.01 * np.max(np.abs(data["q_bend_trace"])) * 0.5 * spec.span_m * 1000.0
        print(
            f"  captured {len(data['frames'])} frames (spc={data['spc']}) in {time.perf_counter() - t_sim:.1f}s | "
            f"mean_L={np.mean(data['lift_trace']):.2f} N, mean_P={data['mean_elec_power_w']:.2f} W | "
            f"bend={np.max(np.abs(data['q_bend_trace'])):.1f}% ({bend_mm:.1f} mm), "
            f"twist={np.max(np.abs(data['q_twist_deg_trace'])):.1f} deg, "
            f"camb={np.max(np.abs(data['q_camber_trace'])):.1f}%c",
            flush=True,
        )

    torch.cuda.empty_cache()

    # Stage 2: Parallel CPU 6-panel rendering + ffmpeg MP4/GIF encoding
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, min(args.workers, os.cpu_count() or 4))) as pool:
        for aid in selected:
            data, bounds = captured[aid]
            t_ren = time.perf_counter()
            frame_dir = args.output_dir / f"_frames_{aid}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            tasks = [
                (i, data, bounds, str(frame_dir / f"frame_{i:04d}.png"))
                for i in range(len(data["frames"]))
            ]
            frame_paths = pool.map(_render_single_frame, tasks)

            mp4_path = args.output_dir / f"{aid}_flight_lev_tev.mp4"
            gif_path = args.output_dir / f"{aid}_flight_lev_tev.gif"
            sheet_path = args.output_dir / f"{aid}_keyframes_4phase.png"

            encode_mp4_and_gif(frame_paths, mp4_path, gif_path, fps=args.fps)
            _render_4phase_contact_sheet(data, bounds, sheet_path)

            for fp in frame_paths:
                Path(fp).unlink(missing_ok=True)
            try:
                frame_dir.rmdir()
            except OSError:
                pass
            print(
                f"[Rendered] {aid} in {time.perf_counter() - t_ren:.1f}s -> "
                f"{mp4_path.name} ({mp4_path.stat().st_size / 1e6:.2f} MB), "
                f"{gif_path.name} ({gif_path.stat().st_size / 1e6:.2f} MB), "
                f"{sheet_path.name}",
                flush=True,
            )

    print(f"All {len(selected)} Stage-4 exact-geometry FSI/FSF animations completed in {time.perf_counter() - t0:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
