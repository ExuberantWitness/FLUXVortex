"""Goland flutter — verified GPU V5M aero (propose() unchanged) + GPU Beam.

Architecture: beam w(y),θ(y) mapped into Q16 reference_state →
Q16NativeV5MSolver.propose() (zero modification, oracle-verified) →
panel_forces integrated to beam station forces → GPU beam Newmark step.

This uses the EXACT SAME aerodynamic code path as Yamano (verified:
AIC 3e-7, Mf2 4e-17, Qf 0.3%). No new UVLM is written.

Angle of attack baked into the mesh reference (z = -x·sinα), not v_inf.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))

from fluxvortex.q16_ancf_mesh import make_rectangular_q16_mesh
from fluxvortex.warp_fsi import config as wsi
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    Q16NativeV5MSurface, Q16NativeV5MSolver, NativeV5MConfig,
)

DEVICE = "cuda:0"
DTYPE = torch.float64

# Goland frozen
CHORD = 1.8288
SEMI_SPAN = 6.096
EI = 9.773e6
GJ = 0.988e6
M_PER_LEN = 35.72
IP = M_PER_LEN * CHORD**2 / 24
X_EA_CG = 0.10 * CHORD
ZETA = 0.005
RHO = 1.225
ALPHA_RAD = math.radians(2.0)
X_EA = 0.33 * CHORD  # elastic axis for aerodynamic moment

# Q16 mesh (2×8 elements = 7×25 nodes, exact for beam cubic Hermite w)
QX = 2
QY = 8
# Aero panels (matching Goland reference resolution)
NC = 8
NS = 16
# Structural time step (matches reference)
DT = 0.003
# Aero step = 4 structural steps (matches reference dt ratio)
STEPS_PER_AERO = 4
AERO_DT = DT * STEPS_PER_AERO
# Total aero steps (0.9s of simulation)
N_AERO_STEPS = 250


def build_goland_q16_mesh(thickness=0.02):
    """Build a Q16 mesh with α=2° baked into the reference z."""
    mesh = make_rectangular_q16_mesh(
        chordwise_element_count=QX,
        spanwise_element_count=QY,
        chord=CHORD,
        span=SEMI_SPAN,
        thickness=thickness,
    )
    # Bake α into reference rows: z = -(x)·sin(α), pivot at x=0 (LE)
    # Reference convention: x from 0 to CHORD, z=0 for flat plate
    # For α>0 (nose up), LE stays, TE goes down: z = -(x)*sin(α)
    # But we want the beam pivot at x_ea, so use z = -(x-x_ea)*sin(α)
    rows = mesh.reference_rows.copy()
    rows[:, 2] = -(rows[:, 0]) * math.sin(ALPHA_RAD)
    # Keep directors unchanged ([0, 0, 0.5*t])
    return mesh, rows


def beam_to_q16_state(beam, mesh, rows_ref):
    """Map beam w(y),θ(y) to Q16 reference_state format.

    The Q16 state is ABSOLUTE geometry: [rx,ry,rz, gx,gy,gz] per node.
    Beam deflection: z = -(x)*sin(α) + w(y) + (x-x_ea)·sin(θ(y))
    (matching the reference AeroelasticSolver panel deformation pattern).
    """
    w_np, th_np = beam.w_theta()
    th_np = th_np.cpu().numpy()
    w_np = w_np.cpu().numpy()
    y_nodes = beam.y.cpu().numpy()

    rows = rows_ref.copy()
    y_all = rows[:, 1]
    x_all = rows[:, 0]

    # Interpolate beam w and θ at Q16 node y-positions
    w_at = np.interp(y_all, y_nodes, w_np)
    th_at = np.interp(y_all, y_nodes, th_np)

    # z = -(x)*sin(α) + w(y) + (x-x_ea)·sin(θ(y))
    # (linearized twist about elastic axis)
    rows[:, 2] = (-x_all * math.sin(ALPHA_RAD)
                  + w_at
                  + (x_all - X_EA) * np.sin(th_at))
    return rows


def beam_to_q16_velocity(beam, mesh):
    """Map beam ẇ(y),θ̇(y) to Q16 velocity format."""
    w_dot, th_dot = beam.wdot_thetadot()
    w_dot = w_dot.cpu().numpy()
    th_dot = th_dot.cpu().numpy()
    y_nodes = beam.y.cpu().numpy()

    rows = np.zeros((mesh.node_count, 6))
    y_all = mesh.reference_rows[:, 1]
    x_all = mesh.reference_rows[:, 0]

    w_at = np.interp(y_all, y_nodes, w_dot)
    th_at = np.interp(y_all, y_nodes, th_dot)

    rows[:, 2] = w_at + (x_all - X_EA) * th_at
    return rows


def extract_beam_forces(proposal, beam, nc, ns):
    """Extract station lift/moment from proposal.load.panel_forces.

    Panel ordering: p = i*ns + j (chord-major).
    Per strip j: lift = Σ_i F[i*ns+j, 2], moment = Σ_i (x_i - x_ea)·F_z.
    """
    pf = proposal.load.panel_forces      # (nc*ns, 3) CUDA
    pos = proposal.load.panel_positions  # (nc*ns, 3) CUDA
    pf_2d = pf.reshape(nc, ns, 3)
    pos_2d = pos.reshape(nc, ns, 3)

    lift_st = torch.sum(pf_2d[:, :, 2], dim=0)  # (ns,)
    moment_st = torch.sum(pf_2d[:, :, 2] * (pos_2d[:, :, 0] - X_EA), dim=0)
    station_y = pos_2d[0, :, 1]
    return station_y, lift_st, moment_st


def map_to_beam(beam, station_y, lift_st, moment_st):
    """Map station forces to beam nodal forces (GPU)."""
    F = torch.zeros(beam.ndof, device=DEVICE, dtype=DTYPE)
    y_np = beam.y.cpu().numpy()
    for k in range(len(station_y)):
        yk = float(station_y[k])
        idx = np.searchsorted(y_np, yk, side='right') - 1
        idx = max(0, min(idx, beam.n_nodes - 2))
        Le = y_np[idx+1] - y_np[idx]
        xi = min(max((yk - y_np[idx]) / Le, 0.0), 1.0)
        xi_t = torch.tensor(xi, device=DEVICE, dtype=DTYPE)
        F[3*idx] += lift_st[k] * (1 - xi_t)
        F[3*(idx+1)] += lift_st[k] * xi_t
        F[3*idx+2] += moment_st[k] * (1 - xi_t)
        F[3*(idx+1)+2] += moment_st[k] * xi_t
    return F


# ── GPU Beam (same as before, CUDA float64) ──

class GpuBeam:
    def __init__(self, n_elem=8):
        self.n_elem = n_elem
        self.n_nodes = n_elem + 1
        self.ndof = 3 * self.n_nodes
        Le = SEMI_SPAN / n_elem
        self.y_nodes_np = np.linspace(0, SEMI_SPAN, self.n_nodes)
        self.y = torch.tensor(self.y_nodes_np, device=DEVICE, dtype=DTYPE)

        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))
        for e in range(n_elem):
            Kb = EI / Le**3 * np.array([
                [12, 6*Le, -12, 6*Le], [6*Le, 4*Le**2, -6*Le, 2*Le**2],
                [-12, -6*Le, 12, -6*Le], [6*Le, 2*Le**2, -6*Le, 4*Le**2]])
            Mb = M_PER_LEN * Le / 420 * np.array([
                [156, 22*Le, 54, -13*Le], [22*Le, 4*Le**2, 13*Le, -3*Le**2],
                [54, 13*Le, 156, -22*Le], [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
            Kt = GJ / Le * np.array([[1, -1], [-1, 1]])
            Mt = IP * Le / 6 * np.array([[2, 1], [1, 2]])
            Ke6 = np.zeros((6, 6)); Me6 = np.zeros((6, 6))
            bi, ti = [0, 1, 3, 4], [2, 5]
            for ii, b in enumerate(bi):
                for jj, bb in enumerate(bi):
                    Ke6[b, bb] = Kb[ii, jj]; Me6[b, bb] = Mb[ii, jj]
            for ii, t in enumerate(ti):
                for jj, tt in enumerate(ti):
                    Ke6[t, tt] = Kt[ii, jj]; Me6[t, tt] = Mt[ii, jj]
            dofs = [e*3+i for i in range(6)]
            for ii in range(6):
                for jj in range(6):
                    K[dofs[ii], dofs[jj]] += Ke6[ii, jj]
                    M[dofs[ii], dofs[jj]] += Me6[ii, jj]

        omega1 = 3.516 * math.sqrt(EI / (M_PER_LEN * SEMI_SPAN**4))
        C = 2 * ZETA / omega1 * K

        fixed = [0, 1, 2]
        free = [i for i in range(self.ndof) if i not in fixed]
        self.free = torch.tensor(free, device=DEVICE, dtype=torch.long)
        self.K = torch.tensor(K, device=DEVICE, dtype=DTYPE)
        self.M = torch.tensor(M, device=DEVICE, dtype=DTYPE)
        self.C = torch.tensor(C, device=DEVICE, dtype=DTYPE)
        self.Kr = self.K[self.free][:, self.free]
        self.Mr = self.M[self.free][:, self.free]
        self.Cr = self.C[self.free][:, self.free]

        self.d = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)
        self.v = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)
        self.a = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)

    def set_added_mass(self, m_add, i_add):
        Le = SEMI_SPAN / self.n_elem
        M_np = self.M.cpu().numpy().copy()
        for e in range(self.n_elem):
            dofs = [e*3+i for i in range(6)]
            Mb = m_add * Le / 420 * np.array([
                [156, 22*Le, 54, -13*Le], [22*Le, 4*Le**2, 13*Le, -3*Le**2],
                [54, 13*Le, 156, -22*Le], [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
            Mt = i_add * Le / 6 * np.array([[2, 1], [1, 2]])
            bi, ti = [0, 1, 3, 4], [2, 5]
            for ii, b in enumerate(bi):
                for jj, bb in enumerate(bi):
                    M_np[dofs[b], dofs[bb]] += Mb[ii, jj]
            for ii, t in enumerate(ti):
                for jj, tt in enumerate(ti):
                    M_np[dofs[t], dofs[tt]] += Mt[ii, jj]
        self.M = torch.tensor(M_np, device=DEVICE, dtype=DTYPE)
        self.Mr = self.M[self.free][:, self.free]

    def step(self, F, dt):
        beta, gamma = 0.25, 0.5
        F_r = F[self.free]
        d_r, v_r, a_r = self.d[self.free], self.v[self.free], self.a[self.free]
        K_eff = self.Kr + gamma/(beta*dt)*self.Cr + 1/(beta*dt**2)*self.Mr
        F_eff = (F_r + self.Mr @ (1/(beta*dt**2)*d_r + 1/(beta*dt)*v_r
                                  + (1/(2*beta)-1)*a_r)
                 + self.Cr @ (gamma/(beta*dt)*d_r + (gamma/beta-1)*v_r
                              + dt*(gamma/(2*beta)-1)*a_r))
        d_new = torch.linalg.solve(K_eff, F_eff)
        a_new = (1/(beta*dt**2)*(d_new-d_r) - 1/(beta*dt)*v_r
                 - (1/(2*beta)-1)*a_r)
        v_new = v_r + dt*((1-gamma)*a_r + gamma*a_new)
        self.d.zero_(); self.v.zero_(); self.a.zero_()
        self.d[self.free] = d_new; self.v[self.free] = v_new; self.a[self.free] = a_new

    def w_theta(self):
        return self.d[0::3], self.d[2::3]

    def wdot_thetadot(self):
        return self.v[0::3], self.v[2::3]


# ── Coupled runner ──

def run_flutter(V, use_mf1=True, n_aero=N_AERO_STEPS):
    beam = GpuBeam(n_elem=8)
    if use_mf1:
        m_add = RHO * math.pi * (CHORD / 2)**2
        i_add = m_add * CHORD**2 / 24
        beam.set_added_mass(m_add, i_add)

    mesh, rows_ref = build_goland_q16_mesh()
    surface = Q16NativeV5MSurface(
        mesh, q16_chordwise_elements=QX, q16_spanwise_elements=QY,
        aerodynamic_chordwise_panels=NC, aerodynamic_spanwise_panels=NS,
        device=DEVICE)
    # Goland NACA0012: physics-based LESPcrit from thickness and Re
    # (compute_lesp_crit in q16_flux_v5m_native.py, calibrated to Narsipur 2022)
    re_goland = RHO * V * CHORD / 15.06e-6  # nu=15.06e-6 for air
    settings = NativeV5MConfig(
        chordwise_panels=NC, spanwise_panels=NS,
        density=RHO, freestream=V,
        aerodynamic_dt=AERO_DT,
        wake_max_rows=96,
        dvm_smoothing_radius_chord=0.04,
        dvm_target_spacing_chord=0.04/2.5,
        particle_capacity=65536,
        lesp_thickness_ratio=0.12,   # NACA0012
        lesp_reynolds=re_goland,
    )
    solver = Q16NativeV5MSolver(surface, settings)

    # Initial perturbation
    tip = beam.n_nodes - 1
    beam.d[3*tip] = 0.05
    beam.d[3*tip+2] = math.radians(2.0)
    a0 = torch.linalg.solve(beam.Mr, -beam.Kr @ beam.d[beam.free])
    beam.a[beam.free] = a0

    # Build initial Q16 state from beam
    rows0 = beam_to_q16_state(beam, mesh, rows_ref)
    state0 = wp.array(np.ascontiguousarray(rows0.reshape(1, -1)),
                      dtype=wsi.DTYPE, device=DEVICE)
    vel0 = wp.zeros_like(state0)

    committed = solver.initialize(state0, vel0)

    tip_w_hist = []
    t0 = time.perf_counter()

    for aero_step in range(n_aero):
        # Map beam state to Q16 format
        rows = beam_to_q16_state(beam, mesh, rows_ref)
        vrows = beam_to_q16_velocity(beam, mesh)
        state = wp.array(np.ascontiguousarray(rows.reshape(1, -1)),
                         dtype=wsi.DTYPE, device=DEVICE)
        vel = wp.array(np.ascontiguousarray(vrows.reshape(1, -1)),
                       dtype=wsi.DTYPE, device=DEVICE)

        # Verified GPU V5M aero step (zero modification)
        proposal = solver.propose(committed, state, vel)
        committed = proposal.trial_state  # advance aero state

        # Extract panel forces → beam forces
        y_st, lift_st, moment_st = extract_beam_forces(proposal, beam, NC, NS)
        F = map_to_beam(beam, y_st, lift_st, moment_st)

        # Structural substeps (4 per aero step)
        for sub in range(STEPS_PER_AERO):
            beam.step(F, DT)

        w, th = beam.w_theta()
        tip_w_hist.append(float(w[-1]))

    return np.array(tip_w_hist), time.perf_counter() - t0


def envelope_growth(sig, dt_aero):
    if len(sig) < 10:
        return 0.0
    s = np.abs(sig)
    pk = [(i*dt_aero, s[i]) for i in range(1, len(s)-1)
          if s[i] > s[i-1] and s[i] > s[i+1]]
    if len(pk) < 3:
        return 0.0
    tp = np.array([p[0] for p in pk])
    ap = np.maximum(np.array([p[1] for p in pk]), 1e-15)
    la, tf = (np.log(ap[1:]), tp[1:]) if len(tp) > 4 else (np.log(ap), tp)
    return float(np.polyfit(tf, la, 1)[0]) if len(tf) >= 2 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--velocities", type=float, nargs="+",
                   default=[120, 130, 140, 145, 150, 160])
    p.add_argument("--no-mf1", action="store_true")
    p.add_argument("--output", type=Path,
                   default=Path("/tmp/goland_v5m_direct.json"))
    args = p.parse_args()

    print("=" * 60)
    print("Goland — verified GPU V5M (propose() unchanged) + GPU Beam")
    print("=" * 60)
    print(f"  Q16 mesh: {QX}x{QY} elements (aero-only surface)")
    print(f"  Aero: {NC}x{NS} panels, α={math.degrees(ALPHA_RAD):.0f}° baked in mesh")
    print(f"  Beam: 8 elem CUDA, Mf1={'ON' if not args.no_mf1 else 'OFF'}")
    print(f"  Refs: 137 (Goland-Luke), 140.2 (lagged Pterra)\n")

    results = []
    for V in args.velocities:
        print(f"V={V:.0f} ...", end="", flush=True)
        try:
            tw, el = run_flutter(V, use_mf1=not args.no_mf1)
            sw = envelope_growth(tw, AERO_DT)
            status = "FLUTTER" if sw > 0 else "stable"
            print(f" {status} (σ={sw:+.4f}, {el:.1f}s)", flush=True)
            results.append({"V": V, "sigma_w": sw, "elapsed": el,
                           "tip_w": tw.tolist()})
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"V": V, "error": str(e)})

    valid = [(r["V"], r["sigma_w"]) for r in results if "sigma_w" in r]
    for i in range(len(valid) - 1):
        v1, s1 = valid[i]; v2, s2 = valid[i+1]
        if s1 < 0 and s2 > 0:
            vf = v1 + (0-s1)/(s2-s1)*(v2-v1)
            print(f"\nFlutter speed: V_f = {vf:.1f} m/s")
            break
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
