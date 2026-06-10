"""Aerodynamic module: UVLM unsteady vortex lattice with wake.

Thin wrapper over `standalone_uvlm.StandaloneUVLM` exposing only the
aerodynamic interface (no coupling). Used by CouplingSolver.

Provides:
  - solve(V_struct, V_ext) → gamma : bound circulation from BC
  - compute_forces(...) → per-panel forces (Bernoulli + Mf2_vec1)
  - compute_mf1(nSc) → AIC^{-1} @ nSc (added-mass kernel)
  - compute_dt_normals(dt_verts) → time derivative of panel normals
  - compute_mf2_1_pressure(V_struct, V_wake, dt_n) → AIC^{-1} damping pressure
  - compute_bound_induction_at_colloc() : V_bound
  - compute_wake_velocity_at_colloc() : V_wake
  - shed_wake(dt), advect_wake(dt)
"""
from __future__ import annotations
import numpy as np
from ..standalone_uvlm import StandaloneUVLM


class UVLMAerodynamics:
    """UVLM unsteady aerodynamics — pure fluid solver."""

    def __init__(self, vertices, V_inf, rho=1.225, core_radius=1e-6):
        self._uvlm = StandaloneUVLM(vertices=vertices, V_inf=V_inf,
                                    rho=rho, core_radius=core_radius)

    # ── Direct passthrough for current code ──
    @property
    def nc(self): return self._uvlm._nc
    @property
    def ns(self): return self._uvlm._ns
    @property
    def n_panels(self): return self._uvlm._n_panels
    @property
    def gamma(self): return self._uvlm.gamma
    @property
    def gamma_prev(self): return self._uvlm.gamma_prev
    @property
    def colloc(self): return self._uvlm._colloc
    @property
    def normals(self): return self._uvlm._normals
    @property
    def corners(self): return self._uvlm._corners
    @property
    def areas(self): return self._uvlm._areas
    @property
    def V_inf(self): return self._uvlm._V_inf
    @property
    def rho(self): return self._uvlm._rho
    @property
    def AIC(self): return self._uvlm._AIC
    @property
    def wake_vertices(self): return self._uvlm.wake_vertices
    @property
    def wake_gamma(self): return self._uvlm.wake_gamma
    @property
    def forces(self): return self._uvlm.forces
    @property
    def forces_no_vstruct(self): return self._uvlm.forces_no_vstruct
    @property
    def dp_lift2(self): return self._uvlm.dp_lift2
    @property
    def Mf2_vec1(self): return self._uvlm.Mf2_vec1

    @property
    def underlying(self):
        """Direct StandaloneUVLM access for legacy code."""
        return self._uvlm

    # ── UVLM solve and force chain ───────────────────────────────────
    def build_aic(self):
        return self._uvlm.build_aic()

    def solve(self, V_ext_colloc=None, V_struct_colloc=None):
        return self._uvlm.solve(V_ext_colloc=V_ext_colloc,
                                V_struct_colloc=V_struct_colloc)

    def compute_forces(self, dt, V_ext_colloc=None, V_struct_colloc=None):
        return self._uvlm.compute_forces(dt,
                                         V_ext_colloc=V_ext_colloc,
                                         V_struct_colloc=V_struct_colloc)

    def compute_wake_influence(self):
        return self._uvlm.compute_wake_influence()

    def compute_wake_velocity_at_colloc(self):
        return self._uvlm.compute_wake_velocity_at_colloc()

    def compute_bound_induction_at_colloc(self):
        return self._uvlm.compute_bound_induction_at_colloc()

    def shed_wake(self, dt):
        return self._uvlm.shed_wake(dt)

    def advect_wake(self, dt):
        return self._uvlm.advect_wake(dt)

    def truncate_wake_far(self, threshold):
        return self._uvlm.truncate_wake_far(threshold)

    # ── Added-mass / damping kernels ─────────────────────────────────
    def compute_mf1(self, nSc):
        return self._uvlm.compute_mf1(nSc)

    def compute_mf2(self):
        return self._uvlm.compute_mf2()

    def compute_dt_normals(self, dt_verts):
        return self._uvlm.compute_dt_normals(dt_verts)

    def compute_mf2_1_force(self, V_struct_colloc, V_wake_colloc, dt_n_colloc):
        return self._uvlm.compute_mf2_1_force(V_struct_colloc, V_wake_colloc, dt_n_colloc)

    def compute_mf2_vec1_from_internal_wake(self, V_struct_colloc=None):
        return self._uvlm.compute_mf2_vec1_from_internal_wake(V_struct_colloc=V_struct_colloc)
