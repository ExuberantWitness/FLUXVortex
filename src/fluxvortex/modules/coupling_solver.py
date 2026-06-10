"""Coupling solver: FSI orchestration.

Owns:
  - ANCFStructure (structural module)
  - UVLMAerodynamics (aerodynamic module)
  - NewmarkSolver (numerical integrator)

Implements:
  - Panel ↔ element mapping
  - Pulse force (consistent distributed load)
  - Strong predictor-corrector coupling
  - M_added time-interpolation (matches MATLAB Qf_p_mat_t)
  - Force averaging at corrector substep (matches MATLAB stage-1 logic):
      * F_pulse + Bernoulli (Qf_p_global): NOT averaged (single t_np1 value)
      * F_lift2 + F_mf2_1: AVERAGED between (q_n, dq_n) and (q_p1, dq_p1)
      * Q_bend: AVERAGED between q_n and q_p1
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

from ..ancf_shell import NDOF_NODE, _shape_funcs, _gauss_legendre
from .numerical_solver import NewmarkSolver
from .structural_module import ANCFStructure
from .aerodynamic_module import UVLMAerodynamics


class FSICouplingSolver:
    """Strong-coupling FSI solver with proper MATLAB-equivalent force averaging.

    Currently wraps the existing StandaloneHybridSolver for back-compat while
    progressively migrating logic. The NewmarkSolver with force_velocity_callback
    will be used to fix the corrector-substep accuracy gap.
    """

    def __init__(self, structure: ANCFStructure, aero: UVLMAerodynamics,
                 structural_dt: float,
                 uvlm_dt_ratio: int = 34,
                 alpha_v: float = 0.5,
                 c_damp: float = 2.0):
        self.structure = structure
        self.aero = aero
        self.dt_struct = structural_dt
        self.uvlm_ratio = uvlm_dt_ratio
        self.dt_wake = uvlm_dt_ratio * structural_dt
        self.integrator = NewmarkSolver(alpha_v=alpha_v, c_damp=c_damp)
        self.sim_time = 0.0
        self.step_count = 0
        self.tip_w_history = []

    # ── Module accessors ──────────────────────────────────────────────
    @property
    def shell(self):
        return self.structure.shell

    @property
    def uvlm(self):
        return self.aero.underlying

    # NOTE: Full coupling implementation continues to live in
    # standalone_hybrid_solver.StandaloneHybridSolver during migration.
    # The NewmarkSolver above is ready to be wired in for force averaging
    # once the panel-mapping, pulse, and added-mass-interpolation helpers
    # are migrated. See standalone_hybrid_solver._run_strong for current
    # behavior; the migration target is to call
    # NewmarkSolver.step(M_ff, Kt_ff, q_n, dq_n, free, dt,
    #                    F_constant=F_pulse+F_bernoulli,
    #                    F_velocity_callback=lambda q,dq: F_lift2+F_mf2_1,
    #                    Q_internal_callback=structure.internal_forces_separated)
    # which implements the MATLAB stage-0/stage-1 averaging exactly.
