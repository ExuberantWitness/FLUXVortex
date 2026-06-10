"""Numerical solver: Newmark-β implicit time integrator.

Domain-agnostic — knows nothing about structure or aero. Inputs:
  - M (mass matrix, possibly with M_added)
  - Kt (tangent stiffness at q_n)
  - q_n, dq_n (current state)
  - F_constant: forces evaluated at end-of-step (pulse, Bernoulli — not averaged)
  - F_velocity_callback(q, dq) → F_velocity: velocity-coupled forces averaged
    between q_n,dq_n and predicted q_{n+1},dq_{n+1}
  - Q_internal_callback(q) → (Q_mem, Q_bend): structural internal forces.
    Q_bend is averaged between q_n and q_p1 (matching MATLAB stage 1 logic).
    Q_mem uses q_n only.

Outputs: q_{n+1}, dq_{n+1}.

Matches MATLAB new_X_func_FAST.m exactly with C_damp=2, alpha_v=0.5 (trapezoidal).
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import eye as speye, bmat as spbmat, csc_matrix as spcsc
from scipy.sparse.linalg import spsolve, splu


class NewmarkSolver:
    """Implicit Newmark-β trapezoidal integrator (α=0.5, C_damp=2).

    Mirrors MATLAB new_X_func_FAST.m line-by-line semantics. Two-stage
    corrector handles nonlinear structures and velocity-coupled forces.
    """

    def __init__(self, alpha_v: float = 0.5, c_damp: float = 2.0):
        self.alpha_v = alpha_v
        self.c_damp = c_damp

    def step(self, M_ff, Kt_ff, q_n, dq_n,
             free_dofs, dt,
             F_constant=None,
             F_velocity_callback=None,
             Q_internal_callback=None,
             newton_tol=1e-8, max_newton=20):
        """Single Newmark step using MATLAB stage-0/stage-1 logic.

        Parameters
        ----------
        M_ff : scipy sparse (nf, nf) — effective mass (includes M_added if any)
        Kt_ff : scipy sparse (nf, nf) — tangent stiffness at q_n on free DOFs
        q_n, dq_n : ndarray (ndof,) — current full-DOF state
        free_dofs : ndarray (nf,) — indices of free (non-BC) DOFs
        dt : float — time step
        F_constant : ndarray (ndof,) or None — forces NOT averaged (pulse + Bernoulli).
            Evaluated at end-of-step time per MATLAB Qf_time*q_in_norm(time) convention.
        F_velocity_callback : callable (q, dq) → ndarray (ndof,) or None.
            Velocity-coupled forces (F_lift2, F_mf2_1). Stage 1 uses average of
            F(q_n, dq_n) and F(q_p1, dq_p1).
        Q_internal_callback : callable (q) → (Q_mem, Q_bend) or None.
            If provided, stage 1 uses Qe_corr = Q_mem_n + (Q_bend_n + Q_bend_p1)/2.
            Otherwise Q_internal = 0 throughout.

        Returns
        -------
        q_new, dq_new : ndarray (ndof,) — updated state
        """
        nf = len(free_dofs)
        ndof = len(q_n)

        # Stage 0 forces
        if Q_internal_callback is not None:
            Q_mem_n, Q_bend_n = Q_internal_callback(q_n)
            Qe_n = Q_mem_n + Q_bend_n
        else:
            Q_mem_n = np.zeros(ndof)
            Q_bend_n = np.zeros(ndof)
            Qe_n = np.zeros(ndof)

        F_vel_n = F_velocity_callback(q_n, dq_n) if F_velocity_callback else np.zeros(ndof)
        F_const = F_constant if F_constant is not None else np.zeros(ndof)

        # ── Block-reduced Newmark solve (mathematically identical to the 2×2
        #    block system A1·X = b, but ~8× cheaper) ──
        # A1 = [[I, -α·dt·I], [D_bl, M]],  D_bl = c_damp·dt/2·Kt.
        # A1·[x1;x2]=[b1;b2]  ⇒  x1 = b1 + α·dt·x2,
        #   (M + α·dt·D_bl)·x2 = b2 - D_bl·b1.
        # So factor S = M + α·dt·D_bl = M + (α·c_damp·dt²/2)·Kt ONCE (nf×nf),
        # reuse for all 3 RHS. A2 = [[I, (1-α)·dt·I], [D_bl, M]].
        alpha = self.alpha_v
        D_bl = (self.c_damp * dt / 2.0) * Kt_ff           # sparse (nf,nf)
        S = (M_ff + (alpha * dt) * D_bl).tocsc()
        lu = splu(S)

        def solve_A1(b1, b2):
            x2 = lu.solve(b2 - D_bl.dot(b1))
            x1 = b1 + (alpha * dt) * x2
            return x1, x2

        q_free = q_n[free_dofs]
        dq_free = dq_n[free_dofs]
        # A2·X_n = [q + (1-α)dt·dq ; D_bl·q + M·dq]
        b1 = q_free + (1.0 - alpha) * dt * dq_free
        b2 = D_bl.dot(q_free) + M_ff.dot(dq_free)
        a1, a2 = solve_A1(b1, b2)              # A1^{-1} A2 X_n  (the homogeneous part)

        # Stage 0: predictor — rhs = [0 ; Q_global]
        Q_global = F_const + F_vel_n - Qe_n
        s0_1, s0_2 = solve_A1(np.zeros(nf), Q_global[free_dofs])
        X_p1_1 = a1 + dt * s0_1
        X_p1_2 = a2 + dt * s0_2

        q_p1 = q_n.copy(); q_p1[free_dofs] = X_p1_1
        dq_p1 = dq_n.copy(); dq_p1[free_dofs] = X_p1_2

        if Q_internal_callback is not None:
            _, Q_bend_p1 = Q_internal_callback(q_p1)
            Qe_corr = Q_mem_n + (Q_bend_n + Q_bend_p1) / 2.0
        else:
            Qe_corr = Qe_n

        if F_velocity_callback is not None:
            F_vel_p1 = F_velocity_callback(q_p1, dq_p1)
            F_vel_avg = (F_vel_n + F_vel_p1) / 2.0
        else:
            F_vel_avg = F_vel_n

        # Stage 1: corrector — rhs = [0 ; Q_global2]
        Q_global2 = F_const + F_vel_avg - Qe_corr
        s1_1, s1_2 = solve_A1(np.zeros(nf), Q_global2[free_dofs])
        X_new_1 = a1 + dt * s1_1
        X_new_2 = a2 + dt * s1_2

        q_new = q_n.copy(); dq_new = dq_n.copy()
        q_new[free_dofs] = X_new_1
        dq_new[free_dofs] = X_new_2
        return q_new, dq_new
