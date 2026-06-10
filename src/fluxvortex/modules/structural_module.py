"""Structural module: ANCF shell finite element.

Thin wrapper over `ancf_shell.ANCFShell` exposing only the structural mechanics
interface (no integration). Used by CouplingSolver and NewmarkSolver.

Provides:
  - mass_matrix : assembled M (sparse, constant)
  - internal_forces(q) → Qe : nonlinear membrane + bending forces
  - internal_forces_separated(q) → (Q_mem, Q_bend) : for stage-1 averaging
  - tangent_stiffness(q) → Kt : analytical tangent stiffness
  - distributed_load(F_body) → consistent nodal force
  - bc_dofs : frozen-DOF indices
  - free_dofs : complement of bc_dofs
"""
from __future__ import annotations
import numpy as np
from ..ancf_shell import ANCFShell


class ANCFStructure:
    """ANCF shell mechanics — Bogner-Fox-Schmit Hermitian 36-DOF element."""

    def __init__(self, nodes, quads, h, rho, Ex, Ey, nu_xy, **kw):
        self._shell = ANCFShell(nodes=nodes, quads=quads, h=h, rho=rho,
                                Ex=Ex, Ey=Ey, nu_xy=nu_xy, **kw)

    # ── State accessors ───────────────────────────────────────────────
    @property
    def ndof(self): return self._shell.ndof
    @property
    def nn(self): return self._shell.nn
    @property
    def ne(self): return self._shell.ne
    @property
    def nodes(self): return self._shell.nodes
    @property
    def quads(self): return self._shell.quads
    @property
    def q(self): return self._shell.q
    @q.setter
    def q(self, value): self._shell.q = value
    @property
    def dq(self): return self._shell.dq
    @dq.setter
    def dq(self, value): self._shell.dq = value
    @property
    def h(self): return self._shell.h
    @property
    def rho(self): return self._shell.rho

    # ── Mass and force assembly ──────────────────────────────────────
    @property
    def mass_matrix(self):
        return self._shell.M

    def internal_forces(self, q=None):
        return self._shell._internal_forces(q)

    def internal_forces_separated(self, q=None):
        return self._shell._internal_forces_separated(q)

    def tangent_stiffness(self, q=None):
        _, Kt = self._shell._internal_forces_and_tangent(q if q is not None else self._shell.q)
        return Kt

    def distributed_load(self, F_body):
        return self._shell.distributed_load(F_body)

    # ── BC handling ───────────────────────────────────────────────────
    def set_bc(self, nodes_bc, fix_slopes=True):
        return self._shell.set_bc(nodes_bc, fix_slopes=fix_slopes)

    @property
    def bc_dofs(self):
        return np.array(sorted(self._shell._bc_dofs), dtype=np.int32)

    @property
    def free_dofs(self):
        return np.setdiff1d(np.arange(self.ndof), self.bc_dofs)

    # ── Geometry ──────────────────────────────────────────────────────
    def positions(self):
        return self._shell.positions()

    def elem_dofs(self, e):
        return self._shell._elem_dofs(e)

    # ── Element geometry accessors (needed by CouplingSolver) ────────
    @property
    def dL(self):
        return self._shell._dL

    @property
    def dW(self):
        return self._shell._dW

    # ── Pass-through for legacy compatibility ────────────────────────
    @property
    def shell(self):
        """Direct access to underlying ANCFShell for back-compat."""
        return self._shell
