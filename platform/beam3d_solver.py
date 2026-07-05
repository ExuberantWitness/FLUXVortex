"""Warp geometrically-exact 3D beam solver atom (P2-S1).

Wraps the gate-verified warp_fsi beam kernels (`Beam3DConstants`,
`beam_internal_force`, `assemble_beam_kblocks`, `beam_newmark_step`) as a
`newton_pc.StructuralEntry`-conforming atom:

  - state (q, dq) shape (B, ndof=6*nn), DOF layout [u(3), psi(3)] per node,
    psi = global total rotation vector (additive Newmark update; |psi|<0.9*pi
    chart guard every substep);
  - prescribed root motion (rigid flapping frame) via the three-hook recipe
    mirrored from ancf_shell.set_prescribed_motion/step_newmark:
      (1) inertial coupling  F -= M[:,presc] @ ddq_b(t_end)
      (2) stage-1 recompute scatters q_b(t_end) into q_p1 before Q_int
      (3) end-of-step writeback q[presc]=q_b, dq[presc]=dq_b
  - inner linear solve is the batched dense LU (beam tangent kappa ~ 1e6
    defeats Jacobi-PCG; dense step cross-verified against gpu_newmark_step
    in tests/test_beam3d_warp.py gate 5).

Run me for the entry-level checks:  cd FLUXV && python platform/beam3d_solver.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

_FSI = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.abspath(_FSI) not in sys.path:
    sys.path.insert(0, os.path.abspath(_FSI))

from fluxvortex.warp_fsi import config as cfg               # noqa: E402
from fluxvortex.warp_fsi import kernels_beam3d as kb        # noqa: E402


class NodalForceSet:
    """Generalized nodal force vector ``gen`` (B, ndof), interpolable."""

    __slots__ = ("gen",)

    def __init__(self, gen):
        self.gen = np.asarray(gen, dtype=cfg.NP_DTYPE)

    def affine(self, other, beta):
        return NodalForceSet(self.gen + (other.gen - self.gen) * beta)


class WarpBeam3DEntry:
    """Geometrically-exact beam as a window-PC StructuralEntry (B environments).

    presc_nodes/presc_cb: prescribed rigid-frame motion. presc_cb(t) returns
    (q_b, dq_b, ddq_b), each (npresc*6,) in sorted-node-major [u(3), psi(3)]
    layout, evaluated at the END of the substep (t_end semantics, as in
    ancf_shell.set_prescribed_motion).
    """

    PSI_CHART_MAX = 0.9 * np.pi

    def __init__(self, C: kb.Beam3DConstants, B=1, alpha_v=0.5, c_damp=2.0,
                 presc_nodes=None, presc_cb=None, device=None):
        self.C = C
        self.B = B
        self.alpha_v = alpha_v
        self.c_damp = c_damp
        self.device = device or C.device
        self._pd = None
        self._cb = None
        q0 = np.zeros((B, C.ndof), dtype=cfg.NP_DTYPE)
        dq0 = np.zeros((B, C.ndof), dtype=cfg.NP_DTYPE)
        if presc_nodes is not None:
            assert presc_cb is not None
            C.set_bc(presc_nodes, fix_rot=True)      # held by the solver mask
            self._pd = np.concatenate([C.dof_map_np[n] for n in sorted(presc_nodes)])
            Mg = kb.scatter_beam_global(C.Me_np, C.edofs_np, C.ndof)
            self._Mfp = Mg[:, self._pd]              # (ndof, npresc*6)
            self._cb = presc_cb
            qb, dqb, _ = presc_cb(0.0)
            q0[:, self._pd] = qb
            dq0[:, self._pd] = dqb
        self.q = wp.array(q0, dtype=cfg.DTYPE, device=self.device)
        self.dq = wp.array(dq0, dtype=cfg.DTYPE, device=self.device)
        self.t = 0.0
        self._rot_idx = C.dof_map_np[:, 3:6]         # (nn,3) for the chart guard

    # ── StructuralEntry protocol ──────────────────────────────────────────────
    def snapshot(self):
        return (self.t, self.q.numpy().copy(), self.dq.numpy().copy())

    def restore(self, snap):
        t, qn, dqn = snap
        self.t = t
        self.q = wp.array(qn, dtype=cfg.DTYPE, device=self.device)
        self.dq = wp.array(dqn, dtype=cfg.DTYPE, device=self.device)

    def state(self):
        qn = self.q.numpy()[0]
        dqn = self.dq.numpy()[0]
        tr = self.C.dof_map_np[:, :3]
        return dict(verts=self.C.nodes_np + qn[tr], vels=dqn[tr])

    def substep(self, t, dt, forces=None):
        C, B = self.C, self.B
        f = np.zeros((B, C.ndof), dtype=cfg.NP_DTYPE)
        if forces is not None:
            f = f + np.asarray(forces.gen, dtype=cfg.NP_DTYPE).reshape(B, C.ndof)
        qb = dqb = None
        if self._pd is not None:
            qb, dqb, ddqb = self._cb(t)              # t = end of this substep
            f = f - (self._Mfp @ ddqb)[None, :]      # inertial coupling (hook 1)
        F = wp.array(f, dtype=cfg.DTYPE, device=self.device)
        Qn = kb.beam_internal_force(self.q, C)
        Kblk = kb.assemble_beam_kblocks(self.q, C)

        def recompute(q_p1):
            if self._pd is not None:                 # hook 2: scatter q_b(t_end)
                qn = q_p1.numpy()
                qn[:, self._pd] = qb
                q_p1 = wp.array(qn, dtype=cfg.DTYPE, device=self.device)
            return kb.beam_internal_force(q_p1, C)

        q_new, dq_new = kb.beam_newmark_step(self.q, self.dq, Kblk, C, F, Qn,
                                             recompute, self.alpha_v, self.c_damp,
                                             dt, device=self.device)
        qn = q_new.numpy(); dqn = dq_new.numpy()
        if self._pd is not None:                     # hook 3: end-of-step writeback
            qn[:, self._pd] = qb
            dqn[:, self._pd] = dqb
        psi_max = np.linalg.norm(qn[:, self._rot_idx], axis=-1).max()
        if psi_max > self.PSI_CHART_MAX:
            raise RuntimeError(f"rotation-vector chart exceeded: max|psi|={psi_max:.3f}")
        self.q = wp.array(qn, dtype=cfg.DTYPE, device=self.device)
        self.dq = wp.array(dqn, dtype=cfg.DTYPE, device=self.device)
        self.t = t
        return self


# ── entry-level checks ────────────────────────────────────────────────────────

def _build_spar(nel=16, L=0.8, axis=1):
    nodes = np.zeros((nel + 1, 3))
    nodes[:, axis] = np.linspace(0.0, L, nel + 1)
    elems = np.array([[i, i + 1] for i in range(nel)])
    return kb.Beam3DConstants(nodes, elems, kb.MAIN_SPAR)


def verify_entry():
    # 1) substep == direct beam_newmark_step (no prescribed motion)
    # IC: strain-free rigid rotation-rate field (a random nodal q is a violently
    # stressed state at EA~4e6 and explodes in one step — physics, not a bug)
    C = _build_spar()
    C.set_bc([0], fix_rot=True)
    entry = WarpBeam3DEntry(C)
    om = np.array([0.5, 0.0, 0.2])
    q0 = np.zeros((1, C.ndof))
    dq0 = np.zeros((1, C.ndof))
    for n in range(C.nn):
        dq0[0, C.dof_map_np[n][:3]] = np.cross(om, C.nodes_np[n])
        dq0[0, C.dof_map_np[n][3:]] = om
    q0[:, sorted(C.bc_dofs)] = 0.0; dq0[:, sorted(C.bc_dofs)] = 0.0
    entry.q = wp.array(q0.astype(cfg.NP_DTYPE), dtype=cfg.DTYPE, device=entry.device)
    entry.dq = wp.array(dq0.astype(cfg.NP_DTYPE), dtype=cfg.DTYPE, device=entry.device)
    snap = entry.snapshot()
    entry.substep(1e-4, 1e-4, None)
    qA = entry.q.numpy()
    qw = wp.array(q0.astype(cfg.NP_DTYPE), dtype=cfg.DTYPE, device=entry.device)
    dqw = wp.array(dq0.astype(cfg.NP_DTYPE), dtype=cfg.DTYPE, device=entry.device)
    F = wp.zeros((1, C.ndof), dtype=cfg.DTYPE, device=entry.device)
    Qn = kb.beam_internal_force(qw, C)
    Kblk = kb.assemble_beam_kblocks(qw, C)
    qB, _ = kb.beam_newmark_step(qw, dqw, Kblk, C, F, Qn,
                                 lambda qp: kb.beam_internal_force(qp, C), 0.5, 2.0, 1e-4)
    d1 = np.abs(qA - qB.numpy()).max()
    assert d1 < 1e-14, f"entry substep != direct step: {d1:.3e}"
    print(f"entry substep == direct beam_newmark_step: max diff {d1:.2e}")

    # 2) snapshot/restore replay (GPU atomics caveat -> 1e-12 rel tolerance)
    entry.restore(snap)
    entry.substep(1e-4, 1e-4, None)
    d2 = np.abs(entry.q.numpy() - qA).max() / max(np.abs(qA).max(), 1e-30)
    assert d2 < 1e-12, f"replay mismatch: {d2:.3e}"
    print(f"snapshot->substep->restore->substep replay: rel diff {d2:.2e}")

    # 3) prescribed quasi-static rigid follow: slow root rotation, stiff beam
    C2 = _build_spar()
    om = 0.05                                       # rad/s — quasi-static
    def cb(t):
        th, thd, thdd = om * t, om, 0.0
        return (np.array([0, 0, 0, th, 0, 0.0]),
                np.array([0, 0, 0, thd, 0, 0.0]),
                np.array([0, 0, 0, thdd, 0, 0.0]))
    e2 = WarpBeam3DEntry(C2, presc_nodes=[0], presc_cb=cb)
    dt = 5e-4
    for k in range(2000):                            # t=1.0 s, theta=0.05 rad
        e2.substep((k + 1) * dt, dt, None)
    th = om * 2000 * dt
    tip = e2.state()["verts"][-1]
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    tip_rigid = R @ C2.nodes_np[-1]
    d3 = np.linalg.norm(tip - tip_rigid) / 0.8
    assert d3 < 1e-3, f"prescribed rigid follow err {d3:.3e}"
    print(f"prescribed slow rotation: tip follows rigid frame to {d3:.2e} of L")
    print("WarpBeam3DEntry checks PASS")


if __name__ == "__main__":
    print(cfg.summary())
    verify_entry()
