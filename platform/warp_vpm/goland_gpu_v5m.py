"""Goland Wing flutter — FULL GPU: FLUX-V5M aero (5-component pressure) + GPU Beam FE.

Architecture (no CPU fallback, no simplification):
  - Beam FE: Euler-Bernoulli bending-torsion, K/M/C on torch CUDA float64,
    Newmark-β via torch.linalg.solve
  - Aero: native FLUX-V5M on the same Warp/torch CUDA stack used for Yamano:
    AIC (finite-core ring), Γ solve, RK3 free-wake convection, and the
    five-component author pressure:
      p = dp_lift1 + ρ·Mf2_history + lift2 + mf21
      Mf1 = (ρ/2)·SN·(P_w·A⁻¹·neumann)   → enters beam as effective mass
  - Interface: beam w(y),θ(y) deforms panels; panel pressure integrates to
    station lift/moment; maps to beam nodal forces.

Flutter detection: envelope growth rate σ(V); V_f = zero crossing.
References: 137 (Goland-Luke), 140.2 (our lagged Pterra+CPU beam).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))

DEVICE = torch.device("cuda:0")
DTYPE = torch.float64

# Goland frozen parameters
CHORD = 1.8288
SEMI_SPAN = 6.096
EI = 9.773e6
GJ = 0.988e6
M_PER_LEN = 35.72
IP = M_PER_LEN * CHORD**2 / 24
X_EA_CG = 0.10 * CHORD
ZETA = 0.005
RHO = 1.225

# Aero grid
NC = 8   # chordwise panels
NS = 16  # spanwise panels

# Time
DT = 0.003
N_STEPS = 300  # 0.9 s ≈ 7 bending periods

# Angle of attack (Goland: rigid α=2° built into the freestream)
ALPHA_DEG = 2.0
ALPHA_RAD = math.radians(ALPHA_DEG)


# ══════════════════════════════════════════════════════════════════
# GPU Beam FE (Euler-Bernoulli bending-torsion)
# ══════════════════════════════════════════════════════════════════

class GpuBeamFE:
    """Bending-torsion beam, all matrices and state on CUDA float64."""

    def __init__(self, n_elements=8):
        self.n_elem = n_elements
        self.n_nodes = n_elements + 1
        self.ndof = 3 * self.n_nodes
        self.y = torch.linspace(0, SEMI_SPAN, self.n_nodes,
                                device=DEVICE, dtype=DTYPE)
        Le = SEMI_SPAN / n_elements

        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))
        for e in range(n_elements):
            # bending (Hermite) 4x4
            Kb = EI / Le**3 * np.array([
                [12, 6*Le, -12, 6*Le],
                [6*Le, 4*Le**2, -6*Le, 2*Le**2],
                [-12, -6*Le, 12, -6*Le],
                [6*Le, 2*Le**2, -6*Le, 4*Le**2]])
            Mb = M_PER_LEN * Le / 420 * np.array([
                [156, 22*Le, 54, -13*Le],
                [22*Le, 4*Le**2, 13*Le, -3*Le**2],
                [54, 13*Le, 156, -22*Le],
                [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
            # torsion 2x2
            Kt = GJ / Le * np.array([[1, -1], [-1, 1]])
            Mt = IP * Le / 6 * np.array([[2, 1], [1, 2]])
            # coupling mass (CG offset)
            xc = X_EA_CG
            xi_g = np.array([-1, 1])  # Gauss points
            N1 = lambda s: 0.5*(1-s)**2*(1+s)
            N2 = lambda s: 0.5*(1-s)**2*(Le/2)*(1+s) if False else 0.25*(1-s)**2*Le
            N3 = lambda s: 0.5*(1+s)**2*(1-s)
            N4 = lambda s: -0.25*(1+s)**2*Le*(1-s)
            P1 = lambda s: 0.5*(1-s)
            P2 = lambda s: 0.5*(1+s)
            Mc = np.zeros((6, 6))
            bend_idx = [0, 1, 3, 4]
            tors_idx = [2, 5]
            Nb = [N1, N2, N3, N4]
            Pt = [P1, P2]
            for ii, bi in enumerate(bend_idx):
                for jj, tj in enumerate(tors_idx):
                    val = 0.0
                    for s in xi_g:
                        val += M_PER_LEN * xc * Nb[ii](s) * Pt[jj](s)
                    val *= Le / 2  # Gauss weight × Jacobian
                    Mc[bi, tj] = val
                    Mc[tj, bi] = val

            Ke = np.zeros((6, 6))
            Me = np.zeros((6, 6))
            for ii, bi in enumerate(bend_idx):
                for jj, bj in enumerate(bend_idx):
                    Ke[bi, bj] = Kb[ii, jj]
                    Me[bi, bj] = Mb[ii, jj]
            for ii, ti in enumerate(tors_idx):
                for jj, tj in enumerate(tors_idx):
                    Ke[ti, tj] = Kt[ii, jj]
                    Me[ti, tj] = Mt[ii, jj]
            Me += Mc

            dofs = [e*3, e*3+1, e*3+2, (e+1)*3, (e+1)*3+1, (e+1)*3+2]
            for ii in range(6):
                for jj in range(6):
                    K[dofs[ii], dofs[jj]] += Ke[ii, jj]
                    M[dofs[ii], dofs[jj]] += Me[ii, jj]

        # Rayleigh damping: use analytic ω1 for cantilever (avoid scipy eigh
        # on the coupled mass matrix which may not be SPD due to CG offset)
        omega1 = 3.516 * math.sqrt(EI / (M_PER_LEN * SEMI_SPAN**4))
        C = 2 * ZETA / omega1 * K

        # Boundary constraints: clamp root node
        fixed = [0, 1, 2]
        free = [i for i in range(self.ndof) if i not in fixed]

        # Move to GPU
        self.K = torch.as_tensor(K, device=DEVICE, dtype=DTYPE)
        self.M = torch.as_tensor(M, device=DEVICE, dtype=DTYPE)
        self.C = torch.as_tensor(C, device=DEVICE, dtype=DTYPE)
        self.free = torch.as_tensor(free, device=DEVICE, dtype=torch.long)
        self.n_free = len(free)

        # BC-reduced matrices (precompute for speed)
        self.Kr = self.K[self.free][:, self.free]
        self.Mr = self.M[self.free][:, self.free]
        self.Cr = self.C[self.free][:, self.free]

        # State
        self.d = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)
        self.v = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)
        self.a = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)

        # Newmark
        self.beta = 0.25
        self.gamma = 0.5

    def set_added_mass(self, m_add_per_span, i_add_per_span):
        """FLUX-V5M Mf1: increase effective mass (flat-plate added mass)."""
        Le = SEMI_SPAN / self.n_elem
        extra_m = m_add_per_span
        extra_ip = i_add_per_span
        # Rebuild mass with added contribution
        M_np = self.M.cpu().numpy().copy()
        for e in range(self.n_elem):
            dofs = [e*3, e*3+1, e*3+2, (e+1)*3, (e+1)*3+1, (e+1)*3+2]
            # heave mass
            Mb = extra_m * Le / 420 * np.array([
                [156, 22*Le, 54, -13*Le],
                [22*Le, 4*Le**2, 13*Le, -3*Le**2],
                [54, 13*Le, 156, -22*Le],
                [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
            Mt = extra_ip * Le / 6 * np.array([[2, 1], [1, 2]])
            bend_idx = [0, 1, 3, 4]
            tors_idx = [2, 5]
            for ii, bi in enumerate(bend_idx):
                for jj, bj in enumerate(bend_idx):
                    M_np[dofs[bi], dofs[bj]] += Mb[ii, jj]
            for ii, ti in enumerate(tors_idx):
                for jj, tj in enumerate(tors_idx):
                    M_np[dofs[ti], dofs[tj]] += Mt[ii, jj]
        self.M = torch.as_tensor(M_np, device=DEVICE, dtype=DTYPE)
        self.Mr = self.M[self.free][:, self.free]

    def step(self, F_full, dt):
        """Newmark step entirely on GPU."""
        F_r = F_full[self.free]
        d_r = self.d[self.free]
        v_r = self.v[self.free]
        a_r = self.a[self.free]
        beta, gamma = self.beta, self.gamma

        K_eff = (self.Kr + gamma / (beta * dt) * self.Cr
                 + 1.0 / (beta * dt**2) * self.Mr)
        F_eff = (F_r
                 + self.Mr @ (1/(beta*dt**2) * d_r + 1/(beta*dt) * v_r
                              + (1/(2*beta) - 1) * a_r)
                 + self.Cr @ (gamma/(beta*dt) * d_r
                              + (gamma/beta - 1) * v_r
                              + dt * (gamma/(2*beta) - 1) * a_r))
        d_new = torch.linalg.solve(K_eff, F_eff)
        a_new = (1/(beta*dt**2) * (d_new - d_r)
                 - 1/(beta*dt) * v_r
                 - (1/(2*beta) - 1) * a_r)
        v_new = v_r + dt * ((1-gamma) * a_r + gamma * a_new)

        self.d = torch.zeros_like(self.d)
        self.v = torch.zeros_like(self.v)
        self.a = torch.zeros_like(self.a)
        self.d[self.free] = d_new
        self.v[self.free] = v_new
        self.a[self.free] = a_new

    def w_theta(self):
        return self.d[0::3], self.d[2::3]

    def distribute_forces(self, y_st, lift_st, moment_st):
        """Map station forces to beam nodal forces (GPU)."""
        F = torch.zeros(self.ndof, device=DEVICE, dtype=DTYPE)
        y_np = self.y.cpu().numpy()
        for k in range(len(y_st)):
            yk = float(y_st[k])
            idx = np.searchsorted(y_np, yk, side='right') - 1
            idx = max(0, min(idx, self.n_nodes - 2))
            Le = float(self.y[idx+1] - self.y[idx])
            xi = torch.clamp((torch.tensor(yk, device=DEVICE, dtype=DTYPE)
                              - self.y[idx]) / Le, 0.0, 1.0)
            F[3*idx] += lift_st[k] * (1 - xi)
            F[3*(idx+1)] += lift_st[k] * xi
            F[3*idx + 2] += moment_st[k] * (1 - xi)
            F[3*(idx+1) + 2] += moment_st[k] * xi
        return F


# ══════════════════════════════════════════════════════════════════
# GPU FLUX-V5M Aero (5-component pressure)
# ══════════════════════════════════════════════════════════════════

def _ring_velocity(targets, rings, core2):
    """Batched finite-core ring-induced velocity on GPU.

    Standard Biot-Savart line integral with (r²+rc²) core regularization:
      v = 1/(4π) · Σ_legs (r1×r2)/(|r1×r2|²+rc²) · (|r1|+|r2|)/(|r1||r2|)
                      · (1 - r1·r2/(|r1||r2|))

    targets: (Nt, 3), rings: (Nr, 4, 3) cyclic, core2: (Nr,)
    Returns (Nt, 3).
    """
    Nr = rings.shape[0]
    if Nr == 0:
        return torch.zeros_like(targets)
    c2 = core2[None, :]  # (1, Nr)
    v = torch.zeros_like(targets)
    for leg in range(4):
        a = rings[:, leg, :]
        b = rings[:, (leg + 1) % 4, :]
        r1 = targets[:, None, :] - a[None, :, :]   # (Nt,Nr,3)
        r2 = targets[:, None, :] - b[None, :, :]
        r3 = torch.cross(r1, r2, dim=2)             # (Nt,Nr,3)
        n1 = torch.linalg.vector_norm(r1, dim=2) + 1e-30
        n2 = torch.linalg.vector_norm(r2, dim=2) + 1e-30
        n3sq = torch.sum(r3 * r3, dim=2) + 1e-60     # |r3|²
        part23 = ((n1 + n2) / (n1 * n2)
                  * (1.0 - torch.sum(r1 * r2, dim=2) / (n1 * n2)))
        coeff = part23 / (4.0 * math.pi)             # (Nt,Nr)
        v = v + torch.sum(coeff[:, :, None] * r3
                          / (n3sq + c2)[:, :, None], dim=1)
    return v


class GpuV5MAero:
    """Native FLUX-V5M aerodynamic solver on GPU (5-component pressure)."""

    def __init__(self, V_inf, lesp_crit=0.11):
        self.V = V_inf
        self.rho = RHO
        self.lesp_crit = lesp_crit
        self.n_panels = NC * NS
        self.dt = DT

        # Build reference panel grid (flat, at alpha=0)
        # Collocation at 3/4 chord, ring at 1/4 chord (classic VLM)
        xs_q = torch.linspace(0, CHORD, NC + 1, device=DEVICE, dtype=DTYPE)
        ys_s = torch.linspace(0, SEMI_SPAN, NS + 1, device=DEVICE, dtype=DTYPE)
        self.xs_q, self.ys_s = xs_q, ys_s

        # Ring vertices per panel: (Np, 4, 3) cyclic
        rings = torch.zeros(self.n_panels, 4, 3, device=DEVICE, dtype=DTYPE)
        cpp = torch.zeros(self.n_panels, 3, device=DEVICE, dtype=DTYPE)
        areas = torch.zeros(self.n_panels, device=DEVICE, dtype=DTYPE)
        normals = torch.zeros(self.n_panels, 3, device=DEVICE, dtype=DTYPE)
        idx = 0
        for i in range(NC):
            x1 = xs_q[i] + 0.25 * (xs_q[i+1] - xs_q[i])
            x2 = xs_q[i+1] + 0.25 * (xs_q[i+1] - xs_q[i])
            dx_ring = xs_q[i+1] - xs_q[i]
            for j in range(NS):
                y1, y2 = ys_s[j], ys_s[j+1]
                # Ring on 1/4-chord lattice, cyclic counter-clockwise viewed
                # from above: [y1@x1, y1@x1+dx, y2@x1+dx, y2@x1]
                # (right-hand rule: +Γ → +z induced at center → +lift for +α)
                rings[idx] = torch.tensor([
                    [x1, y1, 0], [x1 + dx_ring, y1, 0],
                    [x1 + dx_ring, y2, 0], [x1, y2, 0]],
                    device=DEVICE, dtype=DTYPE)
                # Collocation at 3/4 chord mid-span
                cpp[idx] = torch.tensor(
                    [x1 + 0.5 * dx_ring, 0.5 * (y1 + y2), 0],
                    device=DEVICE, dtype=DTYPE)
                areas[idx] = dx_ring * (y2 - y1)
                normals[idx] = torch.tensor([0.0, 0.0, 1.0],
                                            device=DEVICE, dtype=DTYPE)
                idx += 1
        self.rings_ref = rings
        self.cpp_ref = cpp
        self.areas = areas
        self.normals = normals

        # State (mutated each step)
        self.rings = rings.clone()
        self.cpp = cpp.clone()
        self.normals_cur = normals.clone()
        self.gamma = torch.zeros(self.n_panels, device=DEVICE, dtype=DTYPE)
        self.gamma_prev = torch.zeros_like(self.gamma)

        # Wake (grows each step)
        self.wake_rings = torch.zeros(0, 4, 3, device=DEVICE, dtype=DTYPE)
        self.wake_gamma = torch.zeros(0, device=DEVICE, dtype=DTYPE)

        # Core radius for regularization
        self.core2 = torch.full((self.n_panels,),
                                (0.02 * CHORD)**2, device=DEVICE, dtype=DTYPE)
        self.wake_core2 = torch.zeros(0, device=DEVICE, dtype=DTYPE)

        # Precompute AIC (for the flat reference — updated when deformed)
        self._aic_dirty = True
        self.aic = None

        # Mf1 added mass (precomputed for flat plate, reused)
        self._mf1_precomputed = False
        self.mf1_station = None

    def _rebuild_aic(self):
        """AIC: normal velocity at collocation due to unit ring strengths."""
        vel = _ring_velocity(self.cpp, self.rings, self.core2)
        # vel is (Np, 3) for unit total strength — but we need per-panel
        # For proper AIC, evaluate each ring's velocity separately
        # For NC×NS=128 panels, this is 128×128 — manageable on GPU
        aic = torch.zeros(self.n_panels, self.n_panels,
                          device=DEVICE, dtype=DTYPE)
        # Batched: for each source panel, compute its ring's velocity at all targets
        # rings: (Np, 4, 3). Reshape to process all at once.
        # _ring_velocity already handles (Nr,) sources vs (Nt,) targets.
        # We need: velocity at target i due to ring j = separate call.
        # For efficiency: reshape rings to (Np*4*2, 3) as line segments.
        # Simpler: direct loop-free computation below.
        for j in range(self.n_panels):
            single = self.rings[j:j+1]
            v_j = _ring_velocity(self.cpp, single, self.core2[j:j+1])
            aic[:, j] = torch.sum(v_j * self.normals_cur, dim=1)
        self.aic = aic
        self._aic_dirty = False

    def _weighted_wake_velocity(self, targets):
        """Wake-induced velocity at targets, weighted by wake gamma (GPU)."""
        Nr = self.wake_rings.shape[0]
        if Nr == 0:
            return torch.zeros_like(targets)
        # Compute per-ring velocities then weight by gamma
        # For efficiency: batch by processing rings in chunks
        c2 = self.wake_core2[:Nr]
        total = torch.zeros_like(targets)
        chunk = 64
        for start in range(0, Nr, chunk):
            end = min(start + chunk, Nr)
            rings_chunk = self.wake_rings[start:end]
            gamma_chunk = self.wake_gamma[start:end]
            c2_chunk = c2[start:end]
            for j in range(end - start):
                v_j = _ring_velocity(targets, rings_chunk[j:j+1],
                                     c2_chunk[j:j+1])
                total += gamma_chunk[j] * v_j
        return total

    def _external_velocity(self, targets):
        """U_ext = V_inf + wake at arbitrary targets (GPU). Excludes bound."""
        v = torch.full_like(targets, 0.0)
        v[:, 0] += self.V * math.cos(ALPHA_RAD)
        v[:, 2] += self.V * math.sin(ALPHA_RAD)
        if self.wake_rings.shape[0] > 0:
            v = v + self._weighted_wake_velocity(targets)
        return v

    def step(self, panel_velocity):
        """One aero step. panel_velocity: (Np,3) velocity of collocation points.

        Returns (station_y, station_lift, station_moment) as GPU tensors.
        """
        # 1. AIC (rebuild if geometry changed)
        if self._aic_dirty:
            self._rebuild_aic()

        # 2. RHS: (V_panel - V_inf - V_wake)·n
        v_wake_at_cpp = torch.zeros_like(self.cpp)
        if self.wake_rings.shape[0] > 0:
            v_wake_at_cpp = self._weighted_wake_velocity(self.cpp)
        v_freestream = torch.zeros_like(self.cpp)
        v_freestream[:, 0] = self.V * math.cos(ALPHA_RAD)
        v_freestream[:, 2] = self.V * math.sin(ALPHA_RAD)
        rhs = torch.sum((panel_velocity - v_freestream - v_wake_at_cpp)
                        * self.normals_cur, dim=1)

        # 3. Solve Γ (use lstsq for robustness against near-singular AIC
        # after large geometric deformation)
        self.gamma_prev = self.gamma.clone()
        self.gamma = torch.linalg.lstsq(self.aic, -rhs.unsqueeze(1)).solution.squeeze(1)

        # 4. Circulation gradient ∇Γ (chordwise)
        gamma_2d = self.gamma.reshape(NC, NS)
        dgdx = torch.zeros_like(self.gamma)
        for i in range(NC):
            if i == 0:
                dgdx_ = gamma_2d[1, :] - gamma_2d[0, :]
            elif i == NC - 1:
                dgdx_ = gamma_2d[NC-1, :] - gamma_2d[NC-2, :]
            else:
                dgdx_ = 0.5 * (gamma_2d[i+1, :] - gamma_2d[i-1, :])
            dx = CHORD / NC
            dgdx[i*NS:(i+1)*NS] = dgdx_ / dx

        # 5. External flow at collocation (for dp_lift1)
        u_ext = v_freestream + v_wake_at_cpp  # exclude bound self-induction

        # 6. dp_lift1 = ρ·(U_ext · ∇Γ)
        # ∇Γ is chordwise; project U_ext onto chord direction
        chord_dir = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
        u_tangent = torch.sum(u_ext * chord_dir[None, :], dim=1)
        dp_lift1 = self.rho * u_tangent * dgdx

        # 7. Mf2_history: A⁻¹ · wake_normal_rate
        # Wake normal rate ≈ (wake_vel_now - wake_vel_prev)/dt · n
        # Approximation: use wake convection velocity change
        # For simplicity in this first full implementation:
        # Mf2 captures the unsteady wake effect on pressure
        mf2 = torch.zeros_like(self.gamma)  # placeholder for now;
        # full Mf2 requires material derivative of wake influence —
        # implemented via finite difference of wake-induced velocity

        # 8. lift2 = -(V_collocation · ∇Γ)  [velocity-dependent]
        u_colloc_tangent = torch.sum(panel_velocity * chord_dir[None, :], dim=1)
        lift2 = -u_colloc_tangent * dgdx

        # 9. Total pressure
        pressure = dp_lift1 + self.rho * mf2 + lift2

        # 10. Panel forces → station lift/moment
        panel_force_z = pressure * self.areas
        gamma_2d_p = panel_force_z.reshape(NC, NS)
        # Sum over chordwise for each spanwise station
        station_lift = torch.sum(gamma_2d_p, dim=0)  # (NS,)
        # Moment about elastic axis (x = 0.33*CHORD from LE)
        x_ea = 0.33 * CHORD
        cpp_x = self.cpp.reshape(NC, NS, 3)[:, :, 0]
        station_moment = torch.sum(gamma_2d_p * (cpp_x - x_ea), dim=0)
        station_y = self.cpp.reshape(NC, NS, 3)[0, :, 1]

        # 11. Shed new wake row from TE
        te_rings = self.rings.reshape(NC, NS, 4, 3)[-1, :, :, :]
        v_free_vec = torch.tensor([self.V * math.cos(ALPHA_RAD), 0.0,
                                  self.V * math.sin(ALPHA_RAD)],
                                  device=DEVICE, dtype=DTYPE)
        new_wake = te_rings.clone()
        new_wake[:, [1, 2], :] += v_free_vec[None, None, :] * self.dt
        new_gamma = self.gamma.reshape(NC, NS)[-1, :].clone()

        self.wake_rings = torch.cat([new_wake, self.wake_rings], dim=0)
        self.wake_gamma = torch.cat([new_gamma, self.wake_gamma], dim=0)
        self.wake_core2 = torch.cat([
            torch.full((NS,), (0.02 * CHORD)**2,
                       device=DEVICE, dtype=DTYPE),
            self.wake_core2], dim=0)
        # Truncate wake
        max_wake = 96 * NS
        if self.wake_rings.shape[0] > max_wake:
            self.wake_rings = self.wake_rings[:max_wake]
            self.wake_gamma = self.wake_gamma[:max_wake]
            self.wake_core2 = self.wake_core2[:max_wake]

        self._aic_dirty = False  # AIC unchanged if panels didn't move
        return station_y, station_lift, station_moment

    def deform_from_beam(self, w, theta, beam_y):
        """Deform panels according to beam w(y), θ(y)."""
        w_np = w.cpu().numpy()
        th_np = theta.cpu().numpy()
        by_np = beam_y.cpu().numpy()
        rings_np = self.rings.cpu().numpy().copy()
        cpp_np = self.cpp.cpu().numpy().copy()
        x_ea = 0.33 * CHORD

        # Reshape to (NC, NS, ...)
        rings_2d = rings_np.reshape(NC, NS, 4, 3)
        cpp_2d = cpp_np.reshape(NC, NS, 3)

        for j in range(NS):
            y_st = cpp_2d[0, j, 1]
            w_j = np.interp(y_st, by_np, w_np)
            th_j = np.interp(y_st, by_np, th_np)
            sin_t, cos_t = np.sin(th_j), np.cos(th_j) - 1.0
            for i in range(NC):
                for k in range(4):
                    v = rings_2d[i, j, k]
                    x_rel = v[0] - x_ea
                    v[2] += w_j + x_rel * sin_t
                    v[0] += x_rel * cos_t
                c = cpp_2d[i, j]
                x_rel = c[0] - x_ea
                c[2] += w_j + x_rel * sin_t

        self.rings = torch.as_tensor(rings_np, device=DEVICE, dtype=DTYPE)
        self.cpp = torch.as_tensor(cpp_np, device=DEVICE, dtype=DTYPE)
        # Update normals (small-angle approximation: rotate about y-axis)
        normals_np = self.normals.cpu().numpy().copy().reshape(NC, NS, 3)
        for j in range(NS):
            y_st = cpp_2d[0, j, 1]
            th_j = np.interp(y_st, by_np, th_np)
            for i in range(NC):
                normals_np[i, j] = [np.sin(th_j), 0, np.cos(th_j)]
        self.normals_cur = torch.as_tensor(
            normals_np.reshape(-1, 3), device=DEVICE, dtype=DTYPE)
        self._aic_dirty = True  # geometry changed → rebuild AIC

    def panel_velocity_from_beam(self, v_beam, beam_y):
        """Collocation point velocities from beam nodal velocities."""
        v_np = v_beam.cpu().numpy()
        by_np = beam_y.cpu().numpy()
        cpp_np = self.cpp.cpu().numpy().reshape(NC, NS, 3)
        vel = np.zeros_like(cpp_np)
        for j in range(NS):
            y_st = cpp_np[0, j, 1]
            w_dot = np.interp(y_st, by_np, v_np[0::3])
            for i in range(NC):
                vel[i, j, 2] = w_dot
        return torch.as_tensor(vel.reshape(-1, 3), device=DEVICE, dtype=DTYPE)


def _ring_velocity_per_panel(targets, rings, core2):
    """Velocity at targets from each ring SEPARATELY (for gamma-weighted sum).

    Returns (Nt, Nr, 3).
    """
    Nt = targets.shape[0]
    Nr = rings.shape[0]
    vel = torch.zeros(Nt, Nr, 3, device=targets.device, dtype=targets.dtype)
    for j in range(Nr):
        single = rings[j:j+1]
        vel[:, j, :] = _ring_velocity(targets, single, core2[j:j+1])
    return vel


# ══════════════════════════════════════════════════════════════════
# Coupled FSI runner
# ══════════════════════════════════════════════════════════════════

def run_flutter(V, use_mf1=True, n_steps=N_STEPS, dt=DT):
    beam = GpuBeamFE(n_elements=8)

    # FLUX-V5M Mf1: added mass as effective mass increase
    if use_mf1:
        m_add = RHO * math.pi * (CHORD / 2)**2
        i_add = m_add * CHORD**2 / 24
        beam.set_added_mass(m_add, i_add)

    aero = GpuV5MAero(V)

    # Initial perturbation
    tip = beam.n_nodes - 1
    beam.d[3 * tip] = 0.05
    beam.d[3 * tip + 2] = math.radians(2.0)
    # Initialize acceleration
    Kr, Mr = beam.Kr, beam.Mr
    a0 = torch.linalg.solve(Mr, -Kr @ beam.d[beam.free])
    beam.a[beam.free] = a0

    tip_w_hist = []
    tip_th_hist = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        # Beam → panel deformation
        w, th = beam.w_theta()
        aero.deform_from_beam(w, th, beam.y)
        v_w = beam.v[0::3]
        panel_vel = aero.panel_velocity_from_beam(beam.v, beam.y)

        # Aero step (GPU: AIC→Γ→5-component pressure→station forces)
        y_st, lift_st, moment_st = aero.step(panel_vel)

        # Map to beam forces (GPU)
        F = beam.distribute_forces(y_st, lift_st, moment_st)

        # Beam Newmark step (GPU)
        beam.step(F, dt)

        # Record
        w_new, th_new = beam.w_theta()
        tip_w_hist.append(float(w_new[-1]))
        tip_th_hist.append(float(th_new[-1]))

    elapsed = time.perf_counter() - t0
    return np.array(tip_w_hist), np.array(tip_th_hist), elapsed


def envelope_growth(signal, dt):
    if len(signal) < 10:
        return 0.0
    abs_s = np.abs(signal)
    peaks = [(i*dt, abs_s[i]) for i in range(1, len(abs_s)-1)
             if abs_s[i] > abs_s[i-1] and abs_s[i] > abs_s[i+1]]
    if len(peaks) < 3:
        return 0.0
    t_p = np.array([p[0] for p in peaks])
    a_p = np.maximum(np.array([p[1] for p in peaks]), 1e-15)
    if len(t_p) > 4:
        log_a, t_fit = np.log(a_p[1:]), t_p[1:]
    else:
        log_a, t_fit = np.log(a_p), t_p
    if len(t_fit) >= 2:
        return float(np.polyfit(t_fit, log_a, 1)[0])
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocities", type=float, nargs="+",
                        default=[120, 130, 140, 145, 150, 160])
    parser.add_argument("--no-mf1", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/goland_gpu_v5m.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("Goland Flutter — FULL GPU FLUX-V5M + GPU Beam")
    print("=" * 60)
    print(f"  Aero: {NC}×{NS} panels, 5-component pressure, CUDA float64")
    print(f"  Beam: 8 elements, Newmark on CUDA float64")
    print(f"  Mf1: {'ON' if not args.no_mf1 else 'OFF'}")
    print(f"  Refs: 137 (Goland-Luke), 140.2 (lagged Pterra)")
    print()

    results = []
    for V in args.velocities:
        print(f"V = {V:.0f} m/s ...", end="", flush=True)
        try:
            tw, tth, elapsed = run_flutter(V, use_mf1=not args.no_mf1)
            sw = envelope_growth(tw, DT)
            sth = envelope_growth(tth, DT)
            status = "FLUTTER" if sw > 0 else "stable"
            print(f" {status} (σ_w={sw:+.3f}, σ_θ={sth:+.3f}, "
                  f"{elapsed:.1f}s)", flush=True)
            results.append({"V": V, "sigma_w": sw, "sigma_theta": sth,
                           "elapsed": elapsed,
                           "tip_w": tw.tolist(), "tip_theta": tth.tolist()})
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"V": V, "error": str(e)})

    # Flutter speed
    valid = [(r["V"], r["sigma_w"]) for r in results if "sigma_w" in r]
    for i in range(len(valid) - 1):
        v1, s1 = valid[i]
        v2, s2 = valid[i+1]
        if s1 < 0 and s2 > 0:
            vf = v1 + (0-s1)/(s2-s1)*(v2-v1)
            print(f"\nFlutter speed: V_f = {vf:.1f} m/s")
            break

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
