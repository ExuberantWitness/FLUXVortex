"""Standalone UVLM (Unsteady Vortex Lattice Method) — no PteraSoftware dependency.

Replaces PteraSoftware's ring-vortex panel solver with a minimal NumPy implementation.

Features:
  - Vortex ring panel discretization (collocation + 4 corner vertices)
  - AIC matrix assembly (Biot-Savart on vortex filaments)
  - Vatistas Ncore=2 vortex core regularization (matching Yamano)
  - Wake shedding from trailing edge
  - Wake rollup (RK4 advection of wake ring vortices)
  - Unsteady Bernoulli force computation
  - VPM particle shedding for far-field wake

Reference: Katz & Plotkin, "Low-Speed Aerodynamics" (2001), Ch. 13
"""
import numpy as np
from math import pi as _pi


# ══════════════════════════════════════════════════════════════════════════
# Vortex filament Biot-Savart
# ══════════════════════════════════════════════════════════════════════════

def _vortex_segment_velocity(points, A, B, gamma, core_radius=1e-6):
    """Induced velocity at M points from a straight vortex segment A→B.

    Parameters
    ----------
    points : (M, 3) ndarray — field points
    A, B : (3,) ndarray — segment endpoints
    gamma : float — circulation strength
    core_radius : float — desingularization radius (matching Yamano eps_v)

    Returns
    -------
    V : (M, 3) ndarray — induced velocity
    """
    r1 = points - A  # (M, 3)
    r2 = points - B  # (M, 3)
    r0 = B - A       # (3,)

    # Cross products
    cr = np.cross(r1, r2)           # (M, 3)
    cr_norm2 = np.sum(cr**2, axis=1)  # (M,)

    # Desingularization (Yamano: eps_v = 1e-9, in squared denominator)
    eps2 = core_radius**2
    cr_norm2 = np.maximum(cr_norm2, eps2)

    # Tangent dot products
    r1_norm = np.sqrt(np.maximum(np.sum(r1**2, axis=1), eps2))
    r2_norm = np.sqrt(np.maximum(np.sum(r2**2, axis=1), eps2))

    dot = np.sum(r1 * r0, axis=1) / r1_norm - np.sum(r2 * r0, axis=1) / r2_norm

    # Velocity: V = gamma/(4*pi) * (r1×r2)/|r1×r2|² * (r1·r0/|r1| - r2·r0/|r2|)
    coeff = (gamma / (4 * _pi)) * dot / cr_norm2
    return cr * coeff[:, None]


def ring_vortex_velocity(points, corners, gamma, core_radius=1e-6):
    """Induced velocity from a closed vortex ring (4 segments).

    Parameters
    ----------
    points : (M, 3) — field points
    corners : (4, 3) — [Fr, Fl, Bl, Br] corner vertices (ordered sequentially)
    gamma : float — circulation
    core_radius : float — desingularization radius

    Returns
    -------
    V : (M, 3)

    SIGN CONVENTION (verified against MATLAB fixture at t*=0.1995):
      This returns -V to align with MATLAB's `q1234_mat = -(q1+q2+q3+q4)`.
      Consequence: Python AIC, V_bound, V_wake all have OPPOSITE sign from
      MATLAB intermediates, but the production force/AoA come out physically
      right because the RHS of the AIC solve uses the matching convention and
      the Bernoulli formula (linear in V·∇Γ) cancels the sign.

      ⚠ Downstream code that *constructs* matrices from this output (e.g.,
      `_build_added_mass_matrix`) must compensate. See the sign-flip note in
      `standalone_hybrid_solver.py:_build_added_mass_matrix`.
    """
    V = np.zeros((len(points), 3))
    for k in range(4):
        A = corners[k]
        B = corners[(k + 1) % 4]
        V += _vortex_segment_velocity(points, A, B, gamma, core_radius)
    return -V


def _dt_vortex_segment_velocity(points, A, B, gamma,
                                dt_points, dt_A, dt_B,
                                core_radius=1e-9):
    """Time derivative of `_vortex_segment_velocity`.

    Ports the per-segment computation from
    `cores/functions/fluid/dt_generate_q1234_mat.m` lines 80-93 (segment 1).
    Uses product rule on (r1×r2)/|r1×r2|² · (r1·r0/|r1| − r2·r0/|r2|).

    Parameters mirror `_vortex_segment_velocity` plus dt_points, dt_A, dt_B
    giving the time-derivative of each endpoint position.
    """
    r1 = points - A
    r2 = points - B
    r0 = B - A

    dt_r1 = dt_points - dt_A
    dt_r2 = dt_points - dt_B
    dt_r0 = dt_B - dt_A

    cr = np.cross(r1, r2)                              # (M, 3)
    dt_cr = np.cross(dt_r1, r2) + np.cross(r1, dt_r2)  # (M, 3)

    cr_norm2 = np.sum(cr**2, axis=1)                   # (M,)
    eps2 = core_radius**2
    cr_norm2 = np.maximum(cr_norm2, eps2)
    cr_norm4 = cr_norm2**2

    r1_norm = np.sqrt(np.maximum(np.sum(r1**2, axis=1), eps2))
    r2_norm = np.sqrt(np.maximum(np.sum(r2**2, axis=1), eps2))
    r1_norm3 = r1_norm**3
    r2_norm3 = r2_norm**3

    # d/dt of cr / |cr|²
    inner_dtcr_cr = np.sum(dt_cr * cr, axis=1)         # (M,)
    dt_cr_over_norm2 = (dt_cr / cr_norm2[:, None]
                       - 2 * cr * (inner_dtcr_cr / cr_norm4)[:, None])

    # d/dt of (r1/|r1| − r2/|r2|)
    inner_dtr1_r1 = np.sum(dt_r1 * r1, axis=1)
    inner_dtr2_r2 = np.sum(dt_r2 * r2, axis=1)
    d_r_per_norm = (dt_r1 / r1_norm[:, None]
                    - r1 * (inner_dtr1_r1 / r1_norm3)[:, None]
                    - dt_r2 / r2_norm[:, None]
                    + r2 * (inner_dtr2_r2 / r2_norm3)[:, None])

    # Static parts (mirror static formula)
    r_per_norm_diff = r1 / r1_norm[:, None] - r2 / r2_norm[:, None]
    cr_over_norm2 = cr / cr_norm2[:, None]

    # Product rule:
    #   d/dt[ (cr/|cr|²) · ⟨r0, r1/|r1| − r2/|r2|⟩ ]
    #   = (dt_cr/|cr|² − 2·cr·⟨dt_cr,cr⟩/|cr|⁴) · ⟨r0, r̂⟩
    #     + (cr/|cr|²) · ⟨dt_r0, r̂⟩
    #     + (cr/|cr|²) · ⟨r0, d/dt(r̂)⟩
    inner_r0_rhat = np.sum(r0 * r_per_norm_diff, axis=1)
    inner_dtr0_rhat = np.sum(dt_r0 * r_per_norm_diff, axis=1)
    inner_r0_drhat = np.sum(r0 * d_r_per_norm, axis=1)

    dt_V = (dt_cr_over_norm2 * inner_r0_rhat[:, None]
            + cr_over_norm2 * inner_dtr0_rhat[:, None]
            + cr_over_norm2 * inner_r0_drhat[:, None])

    return (gamma / (4 * _pi)) * dt_V


def dt_ring_vortex_velocity(points, corners, gamma,
                            dt_points, dt_corners,
                            core_radius=1e-9):
    """Time derivative of `ring_vortex_velocity` (4-segment ring).

    Same sign convention as the static version: returns −Σ_segs to mirror
    MATLAB q1234_mat = −(q1+q2+q3+q4). Used by `compute_mf2_vec1`.
    """
    dt_V = np.zeros((len(points), 3))
    for k in range(4):
        A = corners[k]
        B = corners[(k + 1) % 4]
        dt_A = dt_corners[k]
        dt_B = dt_corners[(k + 1) % 4]
        dt_V += _dt_vortex_segment_velocity(
            points, A, B, gamma,
            dt_points, dt_A, dt_B, core_radius)
    return -dt_V


# ══════════════════════════════════════════════════════════════════════════
# UVLM Solver
# ══════════════════════════════════════════════════════════════════════════

class StandaloneUVLM:
    """Minimal UVLM solver for rectangular wings.

    Wing geometry is defined by a grid of panel corner vertices:
      vertices[i, j] = (x, y, z) for vertex at chord index i, span index j
      where i = 0...nc (nc+1 vertices chordwise) and j = 0...ns (ns+1 spanwise)

    Panels are indexed (i, j) with i=0...nc-1, j=0...ns-1.
    Panel (i,j) corners: vertices[i,j], vertices[i+1,j], vertices[i+1,j+1], vertices[i,j+1]
    Trailing edge panels: i = nc-1 for all j
    """

    def __init__(self, vertices, V_inf, rho=1.225, core_radius=1e-6):
        """
        Parameters
        ----------
        vertices : (nc+1, ns+1, 3) ndarray — panel corner vertices
        V_inf : (3,) ndarray — freestream velocity vector
        rho : float — fluid density
        core_radius : float — vortex core desingularization radius
        """
        self._verts = vertices.copy()
        self._V_inf = np.asarray(V_inf, dtype=float)
        self._rho = rho
        self._core_radius = core_radius

        nc, ns = vertices.shape[0] - 1, vertices.shape[1] - 1
        self._nc = nc
        self._ns = ns
        self._n_panels = nc * ns

        # Pre-compute panel geometry (flat panels assumed)
        self._colloc = np.zeros((nc, ns, 3))     # collocation points
        self._normals = np.zeros((nc, ns, 3))    # unit normals
        self._areas = np.zeros((nc, ns))          # panel areas
        self._corners = np.zeros((nc, ns, 4, 3))  # ring corners [B_out, Bnext_out, Bnext_in, B_in] (MATLAB)
        self._compute_geometry()

        # State variables
        self.gamma = np.zeros((nc, ns))           # ring circulation [current]
        self.gamma_prev = np.zeros((nc, ns))      # ring circulation [previous step]
        self.gamma_bound_prev = np.zeros((nc, ns))  # bound circulation (Σγ) [previous step]
        self.forces = np.zeros((nc, ns, 3))       # aerodynamic forces per panel
        # Forces split for MATLAB-like velocity coupling:
        self.forces_no_vstruct = np.zeros((nc, ns, 3))  # forces excluding V_struct
        self.dp_lift2 = np.zeros((nc, ns, 3))     # ρ*(τ_x*dG_dx + τ_y*dG_dy) vector
        # Exposed gradient + Mf2_vec1 for MATLAB layered comparison:
        self.dG_dx = np.zeros((nc, ns))           # MATLAB dx_Gamma (per-panel diff)
        self.dG_dy = np.zeros((nc, ns))           # MATLAB dy_Gamma (per-panel central diff)
        self.Mf2_vec1 = np.zeros((nc, ns))        # wake-time-deriv compensation

        # Wake
        self.wake_vertices = []    # list of (ns, 4, 3) arrays — each row = one chordwise wake station
        self.wake_gamma = []       # list of (ns,) arrays
        self.wake_ages = []        # list of (ns,) arrays

        # AIC
        self._AIC = None           # influence matrix
        self._AIC_wake = None      # wake influence accumulator

    def _compute_geometry(self):
        """Compute collocation points, normals, areas, ring vortex corners.

        Matches MATLAB generate_panel.m: ring corners at 1/4-chord, shared edges.
        Corner order: v1→v2→v3→v4 = B_out→Bnext_out→Bnext_in→B_in (MATLAB).
        """
        v = self._verts
        nc, ns = self._nc, self._ns

        for i in range(nc):
            for j in range(ns):
                LE_in  = v[i, j]
                LE_out = v[i, j+1]
                TE_in  = v[i+1, j]
                TE_out = v[i+1, j+1]

                # Front: panel i's 1/4-chord
                B_out = 0.75 * LE_out + 0.25 * TE_out
                B_in  = 0.75 * LE_in  + 0.25 * TE_in

                # Back: panel i+1's 1/4-chord (shared edges), or TE for last panel
                if i < nc - 1:
                    LE_in_next  = v[i+1, j]
                    LE_out_next = v[i+1, j+1]
                    TE_in_next  = v[i+2, j]
                    TE_out_next = v[i+2, j+1]
                    Bnext_out = 0.75 * LE_out_next + 0.25 * TE_out_next
                    Bnext_in  = 0.75 * LE_in_next  + 0.25 * TE_in_next
                else:
                    Bnext_out = TE_out
                    Bnext_in  = TE_in

                # MATLAB order: v1=B_out, v2=Bnext_out, v3=Bnext_in, v4=B_in
                self._corners[i, j] = [B_out, Bnext_out, Bnext_in, B_in]

                # Collocation point: 3/4 chord, mid-span
                xi_c = 0.75
                eta_c = 0.5
                self._colloc[i, j] = (
                    (1 - xi_c) * (1 - eta_c) * LE_in +
                    xi_c * (1 - eta_c) * TE_in +
                    xi_c * eta_c * TE_out +
                    (1 - xi_c) * eta_c * LE_out
                )

                # Normal vector
                diag1 = TE_out - LE_in
                diag2 = LE_out - TE_in
                normal = np.cross(diag1, diag2)
                area = 0.5 * np.linalg.norm(normal)
                self._areas[i, j] = area
                if area > 1e-15:
                    self._normals[i, j] = normal / (2 * area)

    def _build_aic_ring(self):
        """AIC from ring vortices (4 segments per source panel)."""
        n_total = self._n_panels
        AIC = np.zeros((n_total, n_total))
        for i in range(self._nc):
            for j in range(self._ns):
                idx_i = i * self._ns + j
                colloc = self._colloc[i, j]
                normal = self._normals[i, j]
                for ip in range(self._nc):
                    for jp in range(self._ns):
                        idx_j = ip * self._ns + jp
                        corners = self._corners[ip, jp]
                        V_ind = ring_vortex_velocity(
                            colloc[None, :], corners, 1.0,
                            self._core_radius)
                        AIC[idx_i, idx_j] = np.dot(V_ind[0], normal)
        return AIC

    def build_aic(self):
        """Assemble aerodynamic influence coefficient matrix from ring vortices."""
        self._AIC = self._build_aic_ring()
        return self._AIC

    def compute_wake_influence(self):
        """Compute wake contribution to the AIC linear-system RHS.

        SIGN: `ring_vortex_velocity` returns -V (MATLAB q1234 convention), so
        V_wake here = -V_wake_matlab. MATLAB's rhs adds -V_wake_ml·n. To match
        MATLAB's effective rhs, we need +V_wake_py·n = +(-V_wake_ml)·n
        = -V_wake_ml·n  ✓

        Returns
        -------
        rhs_wake : (n_panels,) ndarray — +V_wake_py · n at each collocation,
                   equivalent to -V_wake_matlab · n in MATLAB convention.
        """
        if not self.wake_vertices:
            return np.zeros(self._n_panels)

        rhs_wake = np.zeros(self._n_panels)
        n_wake_rows = len(self.wake_vertices)

        for i in range(self._nc):
            for j in range(self._ns):
                idx = i * self._ns + j
                colloc = self._colloc[i, j]
                normal = self._normals[i, j]

                V_wake = np.zeros(3)
                for w in range(n_wake_rows):
                    for js in range(self._ns):
                        gamma_w = self.wake_gamma[w][js]
                        if abs(gamma_w) < 1e-15:
                            continue
                        corners = self.wake_vertices[w][js]
                        V_wake += ring_vortex_velocity(
                            colloc[None, :], corners, gamma_w,
                            self._core_radius)[0]

                rhs_wake[idx] = +np.dot(V_wake, normal)

        return rhs_wake

    def compute_wake_velocity_at_colloc(self):
        """Compute full wake-induced velocity vector at all collocation points.

        Returns
        -------
        V_wake : (nc, ns, 3) ndarray — wake-induced velocity at each collocation
        """
        if not self.wake_vertices:
            return np.zeros((self._nc, self._ns, 3))

        V_wake = np.zeros((self._nc, self._ns, 3))
        n_wake_rows = len(self.wake_vertices)

        for i in range(self._nc):
            for j in range(self._ns):
                colloc = self._colloc[i, j]
                for w in range(n_wake_rows):
                    for js in range(self._ns):
                        gamma_w = self.wake_gamma[w][js]
                        if abs(gamma_w) < 1e-15:
                            continue
                        corners = self.wake_vertices[w][js]
                        V_wake[i, j] += ring_vortex_velocity(
                            colloc[None, :], corners, gamma_w,
                            self._core_radius)[0]

        return V_wake

    def solve(self, V_ext_colloc=None, V_struct_colloc=None):
        """Solve for bound circulation: AIC * gamma = RHS.

        RHS = -(V_inf + V_wake + V_ext - V_struct) · n
        (no-penetration boundary condition on moving surface)

        Parameters
        ----------
        V_ext_colloc : (nc, ns, 3) ndarray or None
            External velocity at each collocation point (VPM + wake panels).
        V_struct_colloc : (nc, ns, 3) ndarray or None
            Structural velocity at each collocation point (dt_r).
        """
        if self._AIC is None:
            self.build_aic()

        # RHS: freestream contribution: -(V_inf - V_struct) · n
        rhs = np.zeros(self._n_panels)
        for i in range(self._nc):
            for j in range(self._ns):
                idx = i * self._ns + j
                v_eff = self._V_inf.copy()
                if V_struct_colloc is not None:
                    v_eff = v_eff - V_struct_colloc[i, j]
                rhs[idx] = -np.dot(v_eff, self._normals[i, j])

        # Add V_ext contribution (VPM particles, bound induction etc.)
        if V_ext_colloc is not None:
            for i in range(self._nc):
                for j in range(self._ns):
                    idx = i * self._ns + j
                    rhs[idx] -= np.dot(V_ext_colloc[i, j], self._normals[i, j])

        # Add wake contribution
        rhs += self.compute_wake_influence()

        # Solve linear system
        try:
            gamma_flat = np.linalg.solve(self._AIC, rhs)
        except np.linalg.LinAlgError:
            gamma_flat = np.linalg.lstsq(self._AIC, rhs, rcond=None)[0]

        # Store bound circulation (cumulative sum) from previous step before updating gamma
        self.gamma_bound_prev = np.cumsum(self.gamma_prev, axis=0)
        self.gamma_prev = self.gamma.copy()
        self.gamma = gamma_flat.reshape(self._nc, self._ns)

        return self.gamma

    def compute_bound_induction_at_colloc(self):
        """Compute velocity induced by ALL bound ring vortices at each collocation point.

        Returns (nc, ns, 3) ndarray — V_gamma at each panel's collocation point.
        """
        nc, ns = self._nc, self._ns
        pts_flat = self._colloc.reshape(-1, 3)  # (nc*ns, 3)
        V_bound = np.zeros((nc * ns, 3))
        for bi in range(nc):
            for bj in range(ns):
                g = self.gamma[bi, bj]
                if abs(g) < 1e-15:
                    continue
                V_bound += ring_vortex_velocity(pts_flat, self._corners[bi, bj],
                                                g, self._core_radius)
        return V_bound.reshape(nc, ns, 3)

    def compute_mf2(self):
        """Compute Mf2 = inv(AIC) — maps BC changes to circulation changes.

        MATLAB: Mf2_mat = inv(A_mat). Used in strong coupling to account
        for structural velocity changes between UVLM solves.
        """
        if self._AIC is None:
            self.build_aic()
        return np.linalg.inv(self._AIC)

    def compute_mf1(self, nSc):
        """Compute Mf1 = inv(AIC) @ nSc — maps structural acceleration to dΓ/dt.

        MATLAB: Mf1_mat = A_mat \\ nvec_Sc_global

        nSc: (n_panels, ndof) — maps structural DOF velocities to panel
             normal velocities at collocation points.

        Returns Mf1: (n_panels, ndof) — maps structural DOF accelerations
                to circulation time derivatives.
        """
        if self._AIC is None:
            self.build_aic()
        return np.linalg.solve(self._AIC, nSc)

    def compute_mf2_vec1(self, dt_rc_colloc,
                         wake_corner_list, dt_wake_corner_list,
                         wake_gamma_list):
        """Compute Mf2_vec1 = A^{-1} · (−∑ Γ_w ∂q_w/∂t · n) — wake time-derivative
        compensation for strong coupling (MATLAB solve_fluid.m Mf2_vec1).

        Adds a scalar pressure correction to dp_lift1 in the Bernoulli force,
        accounting for how the convecting wake's induced normal flow at the
        plate's collocation points evolves between fluid time steps.

        Parameters
        ----------
        dt_rc_colloc : (nc, ns, 3) — velocity of each plate collocation point.
            Zero for rigid plate; V_struct for FSI-coupled plate.
        wake_corner_list : list of (n_w, 4, 3) per wake row — wake panel corners.
            Each entry is one chordwise wake-station row (n_w = ns wake panels).
            Order of 4 corners matches MATLAB: [B_out, Bnext_out, Bnext_in, B_in].
        dt_wake_corner_list : list of (n_w, 4, 3) — wake corner velocities.
            For free-stream convection, each entry ≈ V_inf.
        wake_gamma_list : list of (n_w,) — wake panel circulations.

        Returns
        -------
        Mf2_vec1 : (nc, ns) — per-panel scalar pressure correction.
        """
        if self._AIC is None:
            self.build_aic()

        nc, ns = self._nc, self._ns
        n_panels = nc * ns
        colloc_flat = self._colloc.reshape(-1, 3)               # (n_panels, 3)
        dt_rc_flat = np.asarray(dt_rc_colloc).reshape(-1, 3)
        normals_flat = self._normals.reshape(-1, 3)

        # Accumulate ∑_w Γ_w · ∂q_w(rc)/∂t  — a (n_panels, 3) vector field
        Gamma_w_dt_q1234 = np.zeros((n_panels, 3))

        for wake_corners, dt_wake_corners, gamma_w in zip(
                wake_corner_list, dt_wake_corner_list, wake_gamma_list):
            n_w = len(gamma_w)
            for js in range(n_w):
                g = float(gamma_w[js])
                if abs(g) < 1e-15:
                    continue
                corners_j = wake_corners[js]               # (4, 3)
                dt_corners_j = dt_wake_corners[js]         # (4, 3)
                dt_q = dt_ring_vortex_velocity(
                    colloc_flat, corners_j, g,
                    dt_rc_flat, dt_corners_j,
                    self._core_radius)
                Gamma_w_dt_q1234 += dt_q

        # Dot with collocation normals
        Gamma_w_dt_q1234_n = np.sum(Gamma_w_dt_q1234 * normals_flat, axis=1)

        # Mf2_vec1 = A^{-1} · (−Gamma_w_dt_q1234_n)
        Mf2_flat = np.linalg.solve(self._AIC, -Gamma_w_dt_q1234_n)
        self.Mf2_vec1 = Mf2_flat.reshape(nc, ns)
        return self.Mf2_vec1

    def compute_dt_normals(self, dt_verts):
        """Compute time derivative of each panel normal from vertex velocities.

        Ports MATLAB generate_dt_n_vec.m:
          n = (r13 × r42) / |r13 × r42|
          dt_n = dt(cross)/|cross| − n·(n·dt(cross)/|cross|)
        where r13, r42 are the panel diagonals (corner3-corner1, corner2-corner4).

        Parameters
        ----------
        dt_verts : (nc+1, ns+1, 3) — velocity of each ANCF vertex

        Returns
        -------
        dt_n : (nc, ns, 3) — time derivative of each panel's unit normal
        """
        nc, ns = self._nc, self._ns
        dt_n = np.zeros((nc, ns, 3))
        for i in range(nc):
            for j in range(ns):
                # Same construction as _compute_geometry: diag1 = TE_out - LE_in,
                #                                          diag2 = LE_out - TE_in
                LE_in_dt  = dt_verts[i,   j]
                LE_out_dt = dt_verts[i,   j+1]
                TE_in_dt  = dt_verts[i+1, j]
                TE_out_dt = dt_verts[i+1, j+1]
                LE_in  = self._verts[i,   j]
                LE_out = self._verts[i,   j+1]
                TE_in  = self._verts[i+1, j]
                TE_out = self._verts[i+1, j+1]

                diag1 = TE_out - LE_in
                diag2 = LE_out - TE_in
                dt_diag1 = TE_out_dt - LE_in_dt
                dt_diag2 = LE_out_dt - TE_in_dt

                cross = np.cross(diag1, diag2)
                dt_cross = np.cross(dt_diag1, diag2) + np.cross(diag1, dt_diag2)
                norm_cross = np.linalg.norm(cross) + 1e-30
                n_unit = cross / norm_cross
                dt_cross_over_norm = dt_cross / norm_cross
                # Project to perpendicular of n
                dt_n[i, j] = dt_cross_over_norm - n_unit * np.dot(n_unit, dt_cross_over_norm)
        return dt_n

    def compute_mf2_1_force(self, V_struct_colloc, V_wake_colloc, dt_n_colloc):
        """Compute Mf2_1 damping force per panel — MATLAB's Qf_p_mat0 mechanism.

        Per-panel scalar pressure: dp = Mf2_mat · (slip · dt_n)
          where slip = V_struct - V_in - V_wake_physical
                Mf2_mat = AIC^{-1}
                dt_n = time derivative of panel normal

        SIGN: Python's AIC = -AIC_matlab, and Python's V_wake_colloc returned by
        compute_wake_velocity_at_colloc has -V_wake_matlab convention. Net effect
        cancels and the resulting force is physical.

        Returns
        -------
        forces_mf2_1 : (nc, ns, 3) per-panel force in dimensional N.
        """
        if self._AIC is None:
            self.build_aic()
        nc, ns = self._nc, self._ns
        # slip_physical = V_struct - V_in - V_wake_physical
        # V_wake_colloc here is Python sign convention (-V_wake_matlab), so
        # V_wake_physical = -V_wake_colloc. Therefore:
        # slip = V_struct - V_in - (-V_wake_colloc) = V_struct - V_in + V_wake_colloc
        slip = V_struct_colloc - self._V_inf + V_wake_colloc  # (nc, ns, 3)
        scalar_panel = np.sum(slip * dt_n_colloc, axis=-1)     # (nc, ns)

        # Apply AIC^{-1} to scalar. Python AIC = -AIC_ml, so the MATLAB-physical
        # pressure p_ml = AIC_ml^{-1} · scalar = (-AIC_py)^{-1} · scalar
        #                = -AIC_py^{-1} · scalar.
        pressure_phys = -np.linalg.solve(self._AIC, scalar_panel.ravel()).reshape(nc, ns)

        forces = np.zeros((nc, ns, 3))
        for i in range(nc):
            for j in range(ns):
                forces[i, j] = (self._rho * pressure_phys[i, j]
                                * self._areas[i, j] * self._normals[i, j])
        return forces

    def compute_mf2_vec1_from_internal_wake(self, V_struct_colloc=None,
                                             dt_wake_corner_list=None):
        """Convenience wrapper: build wake state from self.wake_vertices and
        call compute_mf2_vec1.

        Parameters
        ----------
        V_struct_colloc : (nc, ns, 3) or None
            Velocity at plate collocation points. Zero if None.
        dt_wake_corner_list : list of (ns, 4, 3) or None
            Velocity of each wake corner. If None, assumes free-stream
            convection: all wake corners move at V_inf.
        """
        nc, ns = self._nc, self._ns
        if getattr(self, 'disable_mf2_vec1', False) or not self.wake_vertices:
            self.Mf2_vec1 = np.zeros((nc, ns))
            return self.Mf2_vec1

        if V_struct_colloc is None:
            V_struct_colloc = np.zeros((nc, ns, 3))

        if dt_wake_corner_list is None:
            dt_wake_corner_list = []
            for w in range(len(self.wake_vertices)):
                shape = self.wake_vertices[w].shape  # (ns, 4, 3)
                dt_wake_corner_list.append(
                    np.broadcast_to(self._V_inf, shape).copy())

        return self.compute_mf2_vec1(
            V_struct_colloc,
            list(self.wake_vertices),
            dt_wake_corner_list,
            list(self.wake_gamma))

    def compute_forces(self, dt, V_ext_colloc=None, V_struct_colloc=None):
        """Compute aerodynamic forces via unsteady Bernoulli (Katz & Plotkin).

        Also stores forces_no_vstruct and dp_lift2 for MATLAB-like velocity
        coupling update at each structural substep.

        Parameters
        ----------
        dt : float — time since previous solve
        V_ext_colloc : (nc, ns, 3) ndarray or None
        V_struct_colloc : (nc, ns, 3) ndarray or None
        """
        nc, ns = self._nc, self._ns

        # Bound circulation Γ_bound[i,j] = Σ_{k=0}^{i} γ[k,j]  (chordwise cumulative)
        gamma_bound = np.cumsum(self.gamma, axis=0)  # (nc, ns)

        for i in range(nc):
            for j in range(ns):
                c = self._corners[i, j]
                r21 = c[1] - c[0]; r34 = c[2] - c[3]
                r14 = c[0] - c[3]; r23 = c[1] - c[2]
                tau_x = (r21 + r34) / 2
                tau_y = (r14 + r23) / 2
                tau_x_norm = np.linalg.norm(tau_x) + 1e-15
                tau_y_norm = np.linalg.norm(tau_y) + 1e-15
                tau_x = tau_x / tau_x_norm
                tau_y = tau_y / tau_y_norm

                # Velocity excluding structural (for baseline force)
                V_ext_only = self._V_inf.copy()
                if V_ext_colloc is not None:
                    V_ext_only = V_ext_only + V_ext_colloc[i, j]

                # Full velocity including structural (for total force)
                V_colloc = V_ext_only.copy()
                if V_struct_colloc is not None:
                    V_colloc = V_colloc - V_struct_colloc[i, j]

                # MATLAB convention (calc_fluid_force.m:36-38, Ghommem 2011 p.138):
                # First chordwise panel: dx_Γ_1 = Γ_1/Δx
                # Other panels: dx_Γ_i = (Γ_i − Γ_{i-1})/Δx (backward difference of per-panel ring Γ)
                # This is the non-circulatory part; the KJ contribution is supplied by
                # Mf2_vec1 (wake-time-derivative compensation, computed elsewhere).
                if i == 0:
                    dG_dx = self.gamma[0, j] / tau_x_norm
                else:
                    dG_dx = (self.gamma[i, j] - self.gamma[i-1, j]) / tau_x_norm

                # MATLAB convention (calc_fluid_force.m:40-43):
                # Zero-padded central difference on per-panel ring Γ (not gamma_bound)
                # End j=0: Γ_0/Δy ; End j=ns-1: -Γ_{ns-1}/Δy
                if ns == 1:
                    dG_dy = 0.0
                elif j == 0:
                    dG_dy = self.gamma[i, j] / tau_y_norm
                elif j == ns - 1:
                    dG_dy = -self.gamma[i, j] / tau_y_norm
                else:
                    dG_dy = (self.gamma[i, j+1] - self.gamma[i, j-1]) / (2 * tau_y_norm)

                # Unsteady term: MATLAB dp_add = (Gamma - old_Gamma)/d_t_wake
                # Gamma = per-panel ring circulation, NOT cumulative bound circulation
                dG_dt = (self.gamma[i, j] - self.gamma_prev[i, j]) / max(dt, 1e-15)

                # dp_lift2 magnitude: ρ*(τ_x*dΓ/dx + τ_y*dΓ/dy). The MATLAB sign convention
                # is dp_lift2 = -(τ_x*dx_Γ + τ_y*dy_Γ); here the negative is applied at
                # consumption time in _compute_lift2_force (f_p = -p_lift2 * area * n).
                self.dp_lift2[i, j] = self._rho * (tau_x * dG_dx + tau_y * dG_dy)
                # Expose gradients for MATLAB layered comparison
                self.dG_dx[i, j] = dG_dx
                self.dG_dy[i, j] = dG_dy

                # Total pressure: steady + velocity-coupling + unsteady
                dp_total = (np.dot(V_colloc, tau_x) * dG_dx * self._rho +
                           np.dot(V_colloc, tau_y) * dG_dy * self._rho +
                           dG_dt * self._rho)

                # forces_no_vstruct is consumed by strong-coupling corrector as the
                # MATLAB-equivalent Qf_p_vec source pressure. MATLAB
                # calc_fluid_force_strong.m:6 uses (dp_lift1 + Mf2_vec1) — NO
                # dG_dt (=dp_add) term because dp_add is absorbed into the
                # structural Newmark via the added-mass matrix (Mf1 path).
                # Adding dG_dt here would double-count and over-predict force by
                # ~50% (verified against MATLAB fixture at t*=0.1995).
                dp_no_vstruct = (np.dot(V_ext_only, tau_x) * dG_dx * self._rho +
                                np.dot(V_ext_only, tau_y) * dG_dy * self._rho)
                dp_no_vstruct = dp_no_vstruct + self._rho * self.Mf2_vec1[i, j]

                area_n = self._areas[i, j] * self._normals[i, j]
                self.forces[i, j] = dp_total * area_n
                self.forces_no_vstruct[i, j] = dp_no_vstruct * area_n

        return self.forces

    def shed_wake(self, dt, gamma_source=None):
        """Shed a row of wake ring vortices from the trailing edge.

        Matches MATLAB generate_wake.m: the new panel's FRONT edge is pinned at
        the TE (r_wake_1 = r_panel_vec_2_end) and its BACK edge sits one full
        convection step downstream (r_wake_2 = TE + V_inf*d_t_wake). The shed
        circulation is the delayed-Kutta value (previous bound TE).

        Parameters
        ----------
        dt : float — wake convection timestep (d_t_wake).
        gamma_source : (nc, ns) ndarray or None — bound circulation field whose
            TE row provides the shed circulation. Defaults to self.gamma_prev
            (delayed Kutta: bound TE from two solves ago when shed BEFORE the
            current solve).
        """
        if getattr(self, 'disable_wake', False):
            return
        if gamma_source is None:
            gamma_source = self.gamma_prev
        if np.all(np.abs(gamma_source) < 1e-15):
            return

        nc = self._nc
        ns = self._ns
        te_offset = self._V_inf * dt   # full convection step (MATLAB d_t_wake)

        new_vertices = np.zeros((ns, 4, 3))
        new_gamma = np.zeros(ns)

        for j in range(ns):
            # TE panel (nc-1, j)
            corners_te = self._corners[nc-1, j]
            Fr_te = corners_te[0]  # Fr
            Fl_te = corners_te[1]  # Fl

            # New wake panel: front pinned at TE, back one full step downstream
            Bl_wake = Fl_te + te_offset
            Br_wake = Fr_te + te_offset
            Fl_wake = Fl_te
            Fr_wake = Fr_te

            new_vertices[j] = [Fr_wake, Fl_wake, Bl_wake, Br_wake]
            new_gamma[j] = gamma_source[nc-1, j]

        self.wake_vertices.append(new_vertices)
        self.wake_gamma.append(new_gamma)
        self.wake_ages.append(np.zeros(ns))

    def advect_wake(self, dt, V_ext_func=None):
        """Advect wake ring vortex vertices with local velocity (RK2).

        Parameters
        ----------
        dt : float
        V_ext_func : callable(points) -> (M,3) or None
            External velocity function (e.g. VPM particle induction).
        """
        if not self.wake_vertices:
            return

        n_wake_rows = len(self.wake_vertices)

        # Frozen-snapshot mode (matches MATLAB generate_wake.m frozen-base RK
        # stages and the GPU port): both the evaluation points AND the wake
        # induction sources come from the pre-step snapshot, then all corners
        # update together. Default False = legacy in-place sequential update.
        frozen = getattr(self, 'advect_frozen', False)
        wake_src = ([v.copy() for v in self.wake_vertices] if frozen
                    else self.wake_vertices)

        for w in range(n_wake_rows):
            for js in range(self._ns):
                gamma_w = self.wake_gamma[w][js]
                if abs(gamma_w) < 1e-15:
                    continue

                corners = self.wake_vertices[w][js]
                for kv in range(4):
                    pt = wake_src[w][js][kv] if frozen else corners[kv]

                    # Velocity at vertex: freestream + bound + wake + VPM
                    V_pt = self._V_inf.copy()

                    # Add bound vortex induction
                    for ib in range(self._nc):
                        for jb in range(self._ns):
                            if abs(self.gamma[ib, jb]) > 1e-15:
                                V_pt += ring_vortex_velocity(
                                    pt[None, :], self._corners[ib, jb],
                                    self.gamma[ib, jb], self._core_radius)[0]

                    # Add wake self-induction (all wake rings except own)
                    for w2 in range(n_wake_rows):
                        for js2 in range(self._ns):
                            if w2 == w and js2 == js:
                                continue
                            gamma_w2 = self.wake_gamma[w2][js2]
                            if abs(gamma_w2) < 1e-15:
                                continue
                            V_pt += ring_vortex_velocity(
                                pt[None, :], wake_src[w2][js2],
                                gamma_w2, self._core_radius)[0]

                    # Add VPM induction
                    if V_ext_func is not None:
                        V_pt += V_ext_func(pt[None, :])[0]

                    corners[kv] += V_pt * dt

        # Update ages
        for w in range(len(self.wake_ages)):
            self.wake_ages[w] += dt

    def truncate_wake(self, max_x=5.5):
        """Remove wake panels beyond max_x downstream."""
        if not self.wake_vertices:
            return

        keep = []
        for w in range(len(self.wake_vertices)):
            x_centroid = np.mean(self.wake_vertices[w][:, :, 0])
            if x_centroid < max_x:
                keep.append(w)

        if len(keep) < len(self.wake_vertices):
            self.wake_vertices = [self.wake_vertices[k] for k in keep]
            self.wake_gamma = [self.wake_gamma[k] for k in keep]
            self.wake_ages = [self.wake_ages[k] for k in keep]

    # ─── VPM interface (for hybrid solver) ─────────────────────────────

    def get_wake_particle_sources(self, dt):
        """Convert oldest wake row to VPM particles — 4 particles per ring vortex.

        Decomposes each ring vortex into 4 line-segment particles:
          - Front leg: Fr → Fl
          - Left trailing leg: Fl → Bl
          - Back leg: Bl → Br
          - Right trailing leg: Br → Fr

        Parameters
        ----------
        dt : float — UVLM time step (for sigma = |V_inf| * dt)

        Returns (positions, gammas, sigmas) for new VPM particles.
        """
        if not self.wake_vertices:
            return None, None, None

        V_inf_mag = np.linalg.norm(self._V_inf)
        if V_inf_mag < 1e-10:
            return None, None, None
        sigma = V_inf_mag * dt

        oldest = self.wake_vertices[0]
        oldest_gamma = self.wake_gamma[0]

        positions, gammas, sigmas = [], [], []

        for js in range(self._ns):
            strength = oldest_gamma[js]
            if abs(strength) < 1e-15:
                continue

            fr = oldest[js, 0]  # Fr — TE, outboard
            fl = oldest[js, 1]  # Fl — TE, inboard
            bl = oldest[js, 2]  # Bl — LE, inboard (downstream)
            br = oldest[js, 3]  # Br — LE, outboard (downstream)

            # 4 particles per ring: one per leg
            legs = [
                (0.5 * (fr + fl), strength * (fl - fr)),   # front leg (spanwise)
                (0.5 * (fl + bl), strength * (bl - fl)),   # left trailing (streamwise)
                (0.5 * (bl + br), strength * (br - bl)),   # back leg (spanwise)
                (0.5 * (br + fr), strength * (fr - br)),   # right trailing (streamwise)
            ]

            for pos, gam in legs:
                if np.dot(gam, gam) > 1e-30:
                    positions.append(pos)
                    gammas.append(gam)
                    sigmas.append(sigma)

        if positions:
            return (np.array(positions), np.array(gammas), np.array(sigmas))
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════
# Convenience: build wing grid from ANCF mesh
# ══════════════════════════════════════════════════════════════════════════

def build_uvlm_vertices_from_ancf(shell, V_inf, alpha_deg=0.0):
    """Build UVLM vertex grid matching ANCF shell mesh.

    Parameters
    ----------
    shell : ANCFShell
    V_inf : float — freestream speed
    alpha_deg : float — angle of attack in degrees

    Returns
    -------
    vertices : (nx+1, ny+1, 3) ndarray
    V_inf_vec : (3,) ndarray
    """
    nodes = shell.positions()
    nx = 0
    ny = 0
    # Count unique x and y coordinates to determine panel counts
    x_unique = np.sort(np.unique(np.round(nodes[:, 0], 10)))
    y_unique = np.sort(np.unique(np.round(nodes[:, 1], 10)))
    nx = len(x_unique) - 1
    ny = len(y_unique) - 1

    alpha = np.radians(alpha_deg)
    V_inf_vec = np.array([
        V_inf * np.cos(alpha),
        0.0,
        -V_inf * np.sin(alpha)
    ])

    # Build vertex grid from node positions (flat plate, z=0 for UVLM)
    vertices = np.zeros((nx + 1, ny + 1, 3))
    for i in range(nx + 1):
        for j in range(ny + 1):
            # Find the ANCF node closest to this grid position
            x_tgt = x_unique[i]
            y_tgt = y_unique[j]
            # For initial UVLM, use undeformed positions
            vertices[i, j, 0] = x_tgt
            vertices[i, j, 1] = y_tgt
            vertices[i, j, 2] = 0.0

    # Update from actual node positions
    for idx in range(len(nodes)):
        x_n = nodes[idx, 0]
        y_n = nodes[idx, 1]
        z_n = nodes[idx, 2]
        # Find closest grid vertex
        i = np.argmin(np.abs(x_unique - x_n))
        j = np.argmin(np.abs(y_unique - y_n))
        if i <= nx and j <= ny:
            vertices[i, j, 2] = z_n

    return vertices, V_inf_vec
