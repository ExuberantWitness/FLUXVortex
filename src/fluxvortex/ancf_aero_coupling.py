"""ANCFShell + UVLM/VPM hybrid aeroelastic coupling.

Architecture:
  - Inherits PteraSoftware UnsteadyRingVortexLatticeMethodSolver
  - Collocated mesh: each ANCF element ↔ one UVLM panel
  - Consistent load transfer via shape functions (virtual work)
  - Accurate displacement transfer via ANCF shape function interpolation
  - Supports both explicit (Velocity-Verlet) and implicit (Newmark-β) structural steps

References:
  - Yamano et al., J. Sound and Vibration (2020) — ANCF shell FSI formulation
  - https://github.com/KRproject-tech/FSI_by_FEM_and_UVLM
  - https://github.com/KRproject-tech/MATLAB_ANCF_shell
"""

import numpy as np

import pterasoftware as ps
from .ancf_shell import ANCFShell, NDOF_NODE, NDOF_ELEM


class ANCFAeroelasticSolver(
    ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver
):
    """Coupled ANCF shell + hybrid UVLM/VPM aeroelastic solver.

    Parameters
    ----------
    unsteady_problem : UnsteadyProblem
        PteraSoftware unsteady problem with wing mesh matching ANCF element grid.
    shell : ANCFShell
        ANCF shell structural model. Must have matching nx/ny element counts.
    integrator : str
        'implicit' (Newmark-β) or 'explicit' (Velocity-Verlet).
    relaxation : float
        Structural relaxation factor (0 < r ≤ 1). Helps stability for large dt.
    structural_dt_ratio : float
        Sub-cycling ratio: structural dt = uvlm_dt / structural_dt_ratio.
        >1 when structure needs finer dt than fluid.
    newton_tol : float
        Newton-Raphson convergence tolerance (implicit only).
    max_newton : int
        Max Newton iterations per step (implicit only).
    """

    def __init__(self, unsteady_problem, shell, integrator='implicit',
                 relaxation=1.0, structural_dt_ratio=1,
                 newton_tol=1e-8, max_newton=20):
        super().__init__(unsteady_problem)
        self.shell = shell
        self._integrator = integrator
        self._relaxation = relaxation
        self._struct_ratio = structural_dt_ratio
        self._newton_tol = newton_tol
        self._max_newton = max_newton

        # ── Mesh mapping: UVLM panels ↔ ANCF elements ──
        self._n_aero_chord = None
        self._n_aero_span = None
        self._panel_to_elem = None   # (nc, ns) → element index
        self._panel_xi_eta = None    # (nc, ns, 2) → (xi, eta) of panel center in element
        self._corner_xi_eta = {}     # corner_name → (nc, ns, 2) xi_eta lookup
        self._build_panel_mapping()

        # Cache reference ANCF DOF vector for displacement transfer
        self._q_ref = np.zeros(self.shell.ndof)
        for n in range(self.shell.nn):
            base = n * NDOF_NODE
            self._q_ref[base:base + 3] = self.shell.nodes[n]
            self._q_ref[base + 3:base + 6] = [1.0, 0.0, 0.0]
            self._q_ref[base + 6:base + 9] = [0.0, 1.0, 0.0]

        # ── Results tracking ──
        self.tip_w_history = []
        self.tip_x_history = []
        self.force_history = []
        self.strain_energy_history = []

    # ─── Mesh mapping ────────────────────────────────────────────────────

    def _build_panel_mapping(self):
        """Map each UVLM panel to its corresponding ANCF element and xi/eta."""
        # Determine UVLM panel layout from the first steady problem
        if len(self.steady_problems) == 0:
            raise RuntimeError("No steady problems available for panel mapping.")

        prob0 = self.steady_problems[0]
        for airplane in prob0.airplanes:
            for wing in airplane.wings:
                self._n_aero_chord = wing.num_chordwise_panels
                self._n_aero_span = wing.num_spanwise_panels
                break
            break

        nc = self._n_aero_chord
        ns = self._n_aero_span

        if nc is None or ns is None:
            raise RuntimeError("Could not determine UVLM panel layout.")

        self._panel_to_elem = np.full((nc, ns), -1, dtype=np.int32)
        self._panel_xi_eta = np.zeros((nc, ns, 2))

        # Build ANCF element bounding boxes for fast lookup
        elem_bbox = []
        for e in range(self.shell.ne):
            nd = self.shell.quads[e]
            x_vals = self.shell.nodes[nd, 0]
            y_vals = self.shell.nodes[nd, 1]
            elem_bbox.append((x_vals.min(), x_vals.max(),
                              y_vals.min(), y_vals.max()))

        # For each UVLM panel, find which ANCF element contains its center
        # Use steady_problems (not current_airplanes) since we're in init
        p0_airplanes = prob0.airplanes
        for i in range(nc):
            for j in range(ns):
                p = None
                for airplane in p0_airplanes:
                    for wing in airplane.wings:
                        try:
                            p = wing.panels[i, j]
                        except (IndexError, TypeError):
                            pass
                if p is None:
                    continue
                cx, cy = p.Cpp_GP1_CgP1[0], abs(p.Cpp_GP1_CgP1[1])

                # Find containing ANCF element
                for e, (xmin, xmax, ymin, ymax) in enumerate(elem_bbox):
                    if xmin <= cx <= xmax and ymin <= cy <= ymax:
                        self._panel_to_elem[i, j] = e
                        dL = self.shell._dL[e]
                        dW = self.shell._dW[e]
                        xi = (cx - xmin) / dL if dL > 1e-15 else 0.5
                        eta = (cy - ymin) / dW if dW > 1e-15 else 0.5
                        self._panel_xi_eta[i, j] = [xi, eta]
                        break

        # Build corner xi/eta lookup
        # Frpp (TE-tip), Flpp (TE-root), Blpp (LE-root), Brpp (LE-tip)
        # In ANCF element coords: xi=0→LE, xi=1→TE; eta=0→root, eta=1→tip
        corner_names = ['Frpp', 'Flpp', 'Blpp', 'Brpp',
                        'Frrvp', 'Flrvp', 'Blrvp', 'Brrvp']
        for name in corner_names:
            self._corner_xi_eta[name] = np.zeros((nc, ns, 2))

        # PteraSoftware panel corner positions in ANCF element (xi, eta):
        # Fr = front-right (TE, tip)     → xi=1.0, eta=1.0
        # Fl = front-left  (TE, root)    → xi=1.0, eta=0.0
        # Bl = back-left   (LE, root)    → xi=0.0, eta=0.0
        # Br = back-right  (LE, tip)     → xi=0.0, eta=1.0
        self._corner_xi_eta['Frpp'][:, :] = [1.0, 1.0]
        self._corner_xi_eta['Flpp'][:, :] = [1.0, 0.0]
        self._corner_xi_eta['Blpp'][:, :] = [0.0, 0.0]
        self._corner_xi_eta['Brpp'][:, :] = [0.0, 1.0]
        self._corner_xi_eta['Frrvp'][:, :] = [1.0, 1.0]
        self._corner_xi_eta['Flrvp'][:, :] = [1.0, 0.0]
        self._corner_xi_eta['Blrvp'][:, :] = [0.0, 0.0]
        self._corner_xi_eta['Brrvp'][:, :] = [0.0, 1.0]

        n_mapped = np.sum(self._panel_to_elem >= 0)
        print(f"[ancf_aero] Panel mapping: {n_mapped}/{nc*ns} UVLM panels → ANCF elements")

    def _get_panel(self, i, j):
        """Get UVLM panel (i_chord, j_span)."""
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                try:
                    return wing.panels[i, j]
                except (IndexError, TypeError):
                    pass
        return None

    # ─── Override _calculate_loads for structural coupling ───────────────

    def _calculate_loads(self):
        super()._calculate_loads()
        if self._current_step >= 1:
            self._structural_coupling()

    def run(self, prescribed_wake=True, calculate_streamlines=False,
            show_progress=True):
        super().run(
            prescribed_wake=prescribed_wake,
            calculate_streamlines=calculate_streamlines,
            show_progress=show_progress,
        )

    # ─── Main coupling logic ─────────────────────────────────────────────

    def _structural_coupling(self):
        step = self._current_step
        if step >= self.num_steps - 1:
            return

        dt_uvlm = self.delta_time
        dt_struct = dt_uvlm / self._struct_ratio

        prev_q = self.shell.q.copy()

        # Sub-cycle structure within one UVLM step
        for sub in range(self._struct_ratio):
            # 1. Extract per-panel aerodynamic forces
            panel_forces = self._extract_panel_forces()

            # 2. Consistent load transfer to ANCF DOFs
            F_struct = self._load_transfer(panel_forces)

            # 3. Apply initial perturbation (first step only)
            if step == 1 and sub == 0:
                F_struct = self._apply_initial_perturbation(F_struct)

            # 4. ANCF structural step
            if self._integrator == 'implicit':
                self.shell.step_newmark(
                    F_struct, dt_struct,
                    newton_tol=self._newton_tol,
                    max_newton=self._max_newton)
            else:
                self.shell.step(F_struct, dt_struct)

        # Relaxation
        if self._relaxation < 1.0:
            self.shell.q = prev_q + self._relaxation * (self.shell.q - prev_q)
            self.shell.dq *= self._relaxation

        # 5. Displacement transfer to UVLM panels for next step
        if step + 1 < self.num_steps:
            self._displacement_transfer(step + 1)

        # Record diagnostics
        self._record_history()
        self.force_history.append(
            np.sum(np.abs(panel_forces)) if len(panel_forces) > 0 else 0.0)

    # ─── Force extraction ────────────────────────────────────────────────

    def _extract_panel_forces(self):
        """Extract per-panel aerodynamic force vectors from UVLM.

        Returns (nc, ns, 3) array of force vectors.
        """
        nc, ns = self._n_aero_chord, self._n_aero_span
        forces = np.zeros((nc, ns, 3))

        for i in range(nc):
            for j in range(ns):
                p = self._get_panel(i, j)
                if p is not None:
                    f = getattr(p, 'forces_GP1', None)
                    if f is not None:
                        forces[i, j] = f

        return forces

    # ─── Consistent load transfer ────────────────────────────────────────

    def _load_transfer(self, panel_forces):
        """Consistent load transfer: UVLM panel forces → ANCF nodal DOFs.

        Uses virtual work principle:
          δW = δr(xi_c, eta_c)^T · F_panel = δq_e^T · S^T · F_panel
        Therefore: Q_e = S^T(xi_c, eta_c) @ F_panel

        For each UVLM panel, the force at the collocation point is distributed
        to the 4 nodes of the containing ANCF element via the shape function
        matrix evaluated at the panel center.
        """
        F_struct = np.zeros(self.shell.ndof)
        nc, ns = self._n_aero_chord, self._n_aero_span

        for i in range(nc):
            for j in range(ns):
                f_panel = panel_forces[i, j]
                if np.all(f_panel == 0.0):
                    continue

                e = self._panel_to_elem[i, j]
                if e < 0:
                    continue

                xi, eta = self._panel_xi_eta[i, j]
                dL = self.shell._dL[e]
                dW = self.shell._dW[e]

                # Shape function matrix at panel center: (3, 36)
                from .ancf_shell import _shape_funcs
                S_scalar = _shape_funcs(xi, eta, dL, dW)  # (12,)
                S = np.kron(S_scalar, np.eye(3))           # (3, 36)

                # Consistent force: Q_e = S^T @ F_panel
                Q_e = S.T @ f_panel  # (36,)

                # Scatter to global DOFs
                dofs = self.shell._elem_dofs(e)
                F_struct[dofs] += Q_e

        return F_struct

    # ─── Displacement transfer ───────────────────────────────────────────

    def _displacement_transfer(self, next_step):
        """Map ANCF nodal displacements to UVLM panel vertices for next_step."""
        problem = self.steady_problems[next_step]
        q_ref = self._q_ref

        from .ancf_shell import _shape_funcs

        for airplane in problem.airplanes:
            for wing in airplane.wings:
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels

                for i in range(nc):
                    for j in range(ns):
                        p = wing.panels[i, j]
                        e = self._panel_to_elem[i, j]
                        if e < 0:
                            continue

                        dL = self.shell._dL[e]
                        dW = self.shell._dW[e]
                        dofs = self.shell._elem_dofs(e)
                        q_e = self.shell.q[dofs]
                        q_ref_e = q_ref[dofs]

                        for corner_attr, rv_attr in [
                            ('_Frpp_GP1_CgP1', '_Frrvp_GP1_CgP1'),
                            ('_Flpp_GP1_CgP1', '_Flrvp_GP1_CgP1'),
                            ('_Blpp_GP1_CgP1', '_Blrvp_GP1_CgP1'),
                            ('_Brpp_GP1_CgP1', '_Brrvp_GP1_CgP1'),
                        ]:
                            name = corner_attr.lstrip('_').split('_')[0]
                            xi_c, eta_c = self._corner_xi_eta[name][i, j]
                            S_scalar = _shape_funcs(xi_c, eta_c, dL, dW)
                            S = np.kron(S_scalar, np.eye(3))
                            r_curr = S @ q_e
                            r_ref = S @ q_ref_e
                            disp = r_curr - r_ref

                            # Panel vertex — force writable (PteraSoftware returns read-only views)
                            try:
                                v = getattr(p, corner_attr)
                                if v is not None:
                                    v.flags.writeable = True
                                    v[:] += disp
                                    v.flags.writeable = False
                            except (AttributeError, ValueError, TypeError):
                                pass

                            # Ring vortex vertex
                            try:
                                rv_obj = p.ring_vortex
                                if rv_obj is not None:
                                    v_rv = getattr(rv_obj, rv_attr)
                                    if v_rv is not None:
                                        v_rv.flags.writeable = True
                                        v_rv[:] += disp
                                        v_rv.flags.writeable = False
                            except (AttributeError, ValueError, TypeError):
                                pass

                        # Update collocation point
                        try:
                            cpp = p._Cpp_GP1_CgP1
                            if cpp is not None:
                                xi_c, eta_c = self._panel_xi_eta[i, j]
                                S_scalar = _shape_funcs(xi_c, eta_c, dL, dW)
                                S = np.kron(S_scalar, np.eye(3))
                                r_curr = S @ q_e
                                r_ref = S @ q_ref_e
                                cpp.flags.writeable = True
                                cpp[:] += (r_curr - r_ref)
                                cpp.flags.writeable = False
                        except (AttributeError, ValueError, TypeError):
                            pass

    # ─── Initial perturbation ────────────────────────────────────────────

    def _apply_initial_perturbation(self, F_struct):
        """Override to apply initial perturbation to the structure."""
        return F_struct

    def apply_tip_perturbation(self, tip_force_z=1.0, tip_moment_y=1.0):
        """Apply initial tip perturbation for flutter detection.

        tip_force_z: heave perturbation (N)
        tip_moment_y: pitch/twist perturbation about y-axis (N·m)
        """
        y_max = self.shell.nodes[:, 1].max()
        tip_nodes = np.where(np.abs(self.shell.nodes[:, 1] - y_max) < 1e-6)[0]

        # Determine trailing edge (x_max) among tip nodes
        x_max_tip = self.shell.nodes[tip_nodes, 0].max()

        F_pert = np.zeros(self.shell.ndof)
        for n in tip_nodes:
            base = n * NDOF_NODE
            if abs(self.shell.nodes[n, 0] - x_max_tip) < 1e-6:
                # TE tip: apply both heave force and pitch moment
                F_pert[base + 2] = tip_force_z / 2.0   # heave (z)
                F_pert[base + 5] = tip_moment_y / 2.0  # pitch via dx_r_z
            else:
                F_pert[base + 2] = tip_force_z / 2.0

        # Apply perturbation as initial condition on velocity
        dt = self.delta_time
        M_free = self.shell.M
        # Set velocity via impulse: v = M^{-1} * F * dt
        a_init = np.zeros(self.shell.ndof)
        bc = np.array(sorted(self.shell._bc_dofs), dtype=np.int32)
        free = np.setdiff1d(np.arange(self.shell.ndof), bc)
        if len(free) > 0:
            from scipy.sparse.linalg import spsolve
            M_ff = M_free[np.ix_(free, free)].tocsc()
            a_init[free] = spsolve(M_ff, F_pert[free])
            self.shell.dq[free] = a_init[free] * dt

    # ─── Diagnostics ─────────────────────────────────────────────────────

    def _record_history(self):
        """Record tip displacement and energies for flutter detection."""
        y_max = self.shell.nodes[:, 1].max()
        x_max = self.shell.nodes[:, 0].max()
        tip_mask = (np.abs(self.shell.nodes[:, 1] - y_max) < 1e-6) & \
                   (np.abs(self.shell.nodes[:, 0] - x_max) < 1e-6)
        if np.any(tip_mask):
            tip_idx = np.where(tip_mask)[0][0]
            base = tip_idx * NDOF_NODE
            self.tip_w_history.append(
                self.shell.q[base + 2] - self.shell.nodes[tip_idx, 2])
            self.tip_x_history.append(
                self.shell.q[base + 0] - self.shell.nodes[tip_idx, 0])
        else:
            self.tip_w_history.append(0.0)
            self.tip_x_history.append(0.0)

        # Strain energy for energy budget
        Qe = self.shell._internal_forces()
        self.strain_energy_history.append(0.5 * np.dot(self.shell.q, Qe))

    def get_tip_displacement_history(self):
        """Return (time, tip_w, tip_x) for post-processing."""
        dt = self.delta_time
        n = len(self.tip_w_history)
        t = np.arange(n) * dt
        return t, np.array(self.tip_w_history), np.array(self.tip_x_history)


# ─── Envelope growth rate (flutter detection) ────────────────────────────

def compute_envelope_growth(signal, dt):
    """Compute exponential growth rate σ from signal envelope.

    Positive σ = growing (flutter), negative = decaying (stable).

    Fits an exponential envelope to peak amplitudes using linear regression
    on log(amplitude) vs time.
    """
    if len(signal) < 10:
        return 0.0

    abs_signal = np.abs(signal)
    peaks = []
    for i in range(1, len(abs_signal) - 1):
        if abs_signal[i] > abs_signal[i-1] and abs_signal[i] > abs_signal[i+1]:
            peaks.append((i * dt, abs_signal[i]))

    if len(peaks) < 3:
        return 0.0

    t_peaks = np.array([p[0] for p in peaks])
    a_peaks = np.maximum(np.array([p[1] for p in peaks]), 1e-15)

    # Skip first peak (initial transient)
    if len(t_peaks) > 4:
        log_a = np.log(a_peaks[1:])
        t_fit = t_peaks[1:]
    else:
        log_a = np.log(a_peaks)
        t_fit = t_peaks

    if len(t_fit) >= 2:
        coeffs = np.polyfit(t_fit, log_a, 1)
        return coeffs[0]
    return 0.0


# ─── Convenience: build matching ANCF mesh + UVLM problem ────────────────

def build_ancf_wing(Length=1.0, Width=1.0, thickness=1e-3,
                    nx=6, ny=8, rho=1000.0, E=1e7, nu=0.3,
                    bc_type='clamped'):
    """Build ANCFShell and matching nodal grid for a rectangular cantilever/pinned wing.

    Parameters
    ----------
    Length, Width : float
        Chord length and half-span (nondimensional or meters).
    thickness : float
        Plate thickness.
    nx, ny : int
        Number of elements chordwise × spanwise.
    rho : float
        Material density.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    bc_type : str
        'clamped' (fix all DOFs at LE) or 'pinned' (fix position + y-gradient at LE).

    Returns
    -------
    shell : ANCFShell
    bc_nodes : ndarray
        Node indices of boundary condition nodes at LE.
    """
    nn_x = nx + 1
    nn_y = ny + 1
    nn = nn_x * nn_y

    nodes = np.zeros((nn, 3))
    for j in range(nn_y):
        for i in range(nn_x):
            idx = j * nn_x + i
            nodes[idx, 0] = i * Length / nx
            nodes[idx, 1] = j * Width / ny

    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * nn_x + i
            n1 = n0 + 1
            n2 = n1 + nn_x
            n3 = n0 + nn_x
            quads.append([n0, n1, n2, n3])
    quads = np.array(quads, dtype=np.int32)

    shell = ANCFShell(nodes, quads, h=thickness, rho=rho,
                      Ex=E, Ey=E, nu_xy=nu, mode='full',
                      n_gauss=5)

    # Set boundary conditions
    le_nodes = np.where(nodes[:, 0] < 1e-10)[0]

    if bc_type == 'clamped':
        shell.set_bc(le_nodes)
    elif bc_type == 'pinned':
        # Fix position (0,1,2) and y-gradient (6,7,8), leave x-gradient free
        for n in le_nodes:
            base = n * NDOF_NODE
            for k in list(range(3)) + list(range(6, 9)):
                shell._bc_dofs.add(base + k)

    return shell, le_nodes


def build_uvlm_problem(shell, V_inf, rho=1.225, alpha=2.0,
                       dt=0.001, num_chords=100, mirror_uvlm=False):
    """Build PteraSoftware unsteady UVLM problem matching ANCF mesh.

    The UVLM wing mesh uses the same chordwise/spanwise panel counts
    as the ANCF element grid for collocated coupling.

    Parameters
    ----------
    shell : ANCFShell
        Structural model (used to extract mesh dimensions).
    V_inf : float
        Freestream velocity.
    rho : float
        Fluid density.
    alpha : float
        Angle of attack in degrees.
    dt : float
        UVLM time step.
    num_chords : int
        Number of wake chord lengths to simulate.
    mirror_uvlm : bool
        If True, use symmetric mirror wing.

    Returns
    -------
    movement, operating_point
    """
    # Extract mesh dimensions from ANCF nodes
    nodes = shell.nodes
    chord = nodes[:, 0].max() - nodes[:, 0].min()
    semi_span = nodes[:, 1].max() - nodes[:, 1].min()

    # Count elements per direction
    nx = 0
    ny = 0
    unique_x = np.unique(np.round(nodes[:, 0], 10))
    unique_y = np.unique(np.round(nodes[:, 1], 10))
    # More robust: use the fact that element grid is nx × ny
    for e in range(shell.ne):
        nd = shell.quads[e]
        dL = shell._dL[e]
        dW = shell._dW[e]
    # Count by checking element lengths
    nx = len(np.unique(np.round(shell._dL, 10))) if shell.ne > 0 else 1
    # Actually, let me just use the element topology
    nn_x = int(np.sum(shell.nodes[:, 0] < shell.nodes[:, 0].max() * 0.99) /
               max(1, np.sum(shell.nodes[:, 0] < 1e-6))) + 1
    # Simpler: extract from the node grid directly
    x_vals = np.sort(np.unique(np.round(nodes[:, 0], 10)))
    y_vals = np.sort(np.unique(np.round(nodes[:, 1], 10)))
    nx = len(x_vals) - 1
    ny = len(y_vals) - 1

    # Build PteraSoftware wing matching ANCF grid
    from pterasoftware.geometry.airplane import Airplane
    from pterasoftware.geometry.wing import Wing
    from pterasoftware.geometry.wing_cross_section import WingCrossSection
    from pterasoftware.geometry.airfoil import Airfoil

    wing_cross_sections = [
        WingCrossSection(
            num_spanwise_panels=ny, chord=chord,
            airfoil=Airfoil(name='naca0012', n_points_per_side=200),
            spanwise_spacing='uniform'),
        WingCrossSection(
            num_spanwise_panels=None, chord=chord,
            Lp_Wcsp_Lpp=(0.0, semi_span, 0.0),
            airfoil=Airfoil(name='naca0012', n_points_per_side=200),
            spanwise_spacing=None),
    ]

    wing = Wing(
        wing_cross_sections=wing_cross_sections,
        name='ANCF Wing',
        Ler_Gs_Cgs=(0.0, 0.0, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        symmetric=mirror_uvlm, mirror_only=False,
        num_chordwise_panels=nx, chordwise_spacing='uniform',
    )

    airplane = Airplane(wings=[wing], name='ANCF Wing Model')

    op = ps.operating_point.OperatingPoint(
        rho=rho, vCg__E=V_inf, alpha=alpha, beta=0.0)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wcs)
            for wcs in wing.wing_cross_sections
        ],
    )
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    mv = ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_chords=num_chords, delta_time=dt)

    return mv, op
