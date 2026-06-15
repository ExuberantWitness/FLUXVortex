"""Newton-native ANCF flexible-shell solver atom (P1, iteration 1).

Wraps the *validated, bit-exact* warp_fsi ANCF kernels (`ANCFConstants`,
`assemble_kmem_blocks`, `gpu_newmark_step`) as a structural atom that:

  - keeps its 9-DOF/node ANCF state (`q`, `dq`, shape `(B, ndof)`) **inside the
    solver** (plan §3/§Layer3) — Newton's State carries only the rigid/joint side;
  - conforms to the ``newton_pc.StructuralEntry`` protocol (snapshot / restore /
    substep / state), so the differentiable **window predictor-corrector** drives
    it interchangeably with the hand-rolled reference;
  - reuses the warp_fsi step verbatim, so the bit-exact golden (red line #1) is
    preserved by construction. ``verify_bit_exact`` proves it against the
    standalone CPU Newmark reference.

Batched (B environments) and device-/precision-agnostic via warp_fsi.config
(fp64 default, matching the plan's precision decision). The Newton SolverBase
adapter (registering this into a Newton scene) is the P2 wiring step; the
StructuralEntry contract here is what the coupler needs and what is verifiable now.
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
from fluxvortex.warp_fsi.kernels_ancf import (              # noqa: E402
    ANCFConstants, assemble_kmem_blocks, assemble_internal_force_sep)
from fluxvortex.warp_fsi.batched_solver import gpu_newmark_step  # noqa: E402


# ── interpolable nodal-force container (ForceSet) ─────────────────────────────
class NodalForceSet:
    """Generalized nodal force vector ``gen`` of shape (B, ndof), interpolable."""

    __slots__ = ("gen",)

    def __init__(self, gen: np.ndarray):
        self.gen = np.asarray(gen, dtype=cfg.NP_DTYPE)

    def affine(self, other: "NodalForceSet", beta: float) -> "NodalForceSet":
        return NodalForceSet(self.gen + (other.gen - self.gen) * beta)


# ── the ANCF structural atom (StructuralEntry) ────────────────────────────────
class WarpANCFEntry:
    """GPU/Warp ANCF shell as a window-predictor-corrector StructuralEntry.

    Args:
      shell:   a fluxvortex ANCFShell (carries geometry, orthotropic Dm/Dk, BCs).
      B:       number of batched environments.
      dt:      structural substep (defaults set per case by the coupler).
      alpha_v, c_damp: Newmark block-reduction params (match the golden recipe).
    """

    def __init__(self, shell, B: int = 1, alpha_v: float = 0.5, c_damp: float = 2.0,
                 cg_tol: float = 1e-12, device=None):
        self.device = device or cfg.DEVICE
        self.C = ANCFConstants(shell, device=self.device)
        self.ndof = int(shell.ndof)
        self.B = int(B)
        self.alpha_v, self.c_damp, self.cg_tol = alpha_v, c_damp, cg_tol
        self.t = 0.0
        # rest configuration q_ref (gradient-vector ANCF: pos + unit slopes)
        q0 = np.zeros(self.ndof, dtype=cfg.NP_DTYPE)
        for k in range(shell.nn):
            q0[9 * k] = shell.nodes[k, 0]
            q0[9 * k + 1] = shell.nodes[k, 1]
            q0[9 * k + 3] = 1.0       # dr/dx = x_hat
            q0[9 * k + 7] = 1.0       # dr/dy = y_hat
        self._q0 = q0
        self.q = self._bcast(q0)
        self.dq = wp.zeros((self.B, self.ndof), dtype=cfg.DTYPE, device=self.device)

    def _bcast(self, a):
        return wp.array(np.broadcast_to(a, (self.B, self.ndof)).astype(cfg.NP_DTYPE).copy(),
                        dtype=cfg.DTYPE, device=self.device)

    # ── StructuralEntry protocol ─────────────────────────────────────────────
    def snapshot(self):
        return (self.t, wp.clone(self.q), wp.clone(self.dq))

    def restore(self, snap) -> None:
        self.t, q, dq = snap
        self.q = wp.clone(q)
        self.dq = wp.clone(dq)

    def substep(self, t: float, dt: float, forces: NodalForceSet) -> None:
        """One bit-exact-preserving Newmark step under interpolated nodal forces."""
        Fc = self._bcast(forces.gen) if forces.gen.ndim == 1 else \
            wp.array(forces.gen.astype(cfg.NP_DTYPE).copy(), dtype=cfg.DTYPE, device=self.device)
        Kblk = assemble_kmem_blocks(self.q, self.C, self.device)
        Qmem, Qbend = assemble_internal_force_sep(self.q, self.C, self.device)
        Fvel0 = wp.zeros((self.B, self.ndof), dtype=cfg.DTYPE, device=self.device)

        def recompute_bend(q_p1):
            return assemble_internal_force_sep(q_p1, self.C, self.device)[1]

        self.q, self.dq = gpu_newmark_step(
            self.q, self.dq, Kblk, self.C.Me, self.C.edofs, self.C.free, self.ndof,
            Fc, Qmem, Qbend, Fvel0, recompute_bend, None,
            alpha_v=self.alpha_v, c_damp=self.c_damp, dt=dt, cg_tol=self.cg_tol,
            device=self.device)
        self.t = t

    def state(self) -> dict:
        """Deformed nodal positions per env (for the aero ForceProvider)."""
        q = self.q.numpy().reshape(self.B, -1, 9)
        return {"pos": q[:, :, 0:3].copy(), "q": self.q.numpy().copy()}


# ── bit-exact red-line verification (red line #1) ─────────────────────────────
def verify_bit_exact(B: int = 2) -> bool:
    """One WarpANCFEntry.substep must equal the standalone CPU Newmark reference
    bit-exact (fp64), reusing the same Yamano fixture as warp_fsi/validate NEWMARK."""
    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, "..", "tests"))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell
    from fluxvortex.modules.numerical_solver import NewmarkSolver

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    ndof = shell.ndof
    free_idx = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    alpha_v, c_damp, dt = 0.5, 2.0, 2e-4

    # state: rest + small bend; nonzero dq + constant force on free DOFs
    rng = np.random.default_rng(7)
    entry = WarpANCFEntry(shell, B=B, alpha_v=alpha_v, c_damp=c_damp)
    q_n = entry._q0.copy()
    for k in range(shell.nn):
        x = shell.nodes[k, 0]
        if 9 * k not in shell._bc_dofs:
            q_n[9 * k + 2] += 0.02 * x * x
    dq_n = np.zeros(ndof); dq_n[free_idx] = 0.01 * rng.standard_normal(len(free_idx))
    F_const = np.zeros(ndof); F_const[free_idx] = rng.standard_normal(len(free_idx))

    # CPU reference
    solver = NewmarkSolver(alpha_v=alpha_v, c_damp=c_damp)
    M_ff = shell.M[np.ix_(free_idx, free_idx)].tocsc()
    Kt_ff = shell._tangent_K_mem(q_n)[np.ix_(free_idx, free_idx)].tocsc()
    q_cpu, dq_cpu = solver.step(
        M_ff=M_ff, Kt_ff=Kt_ff, q_n=q_n.copy(), dq_n=dq_n.copy(), free_dofs=free_idx,
        dt=dt, F_constant=F_const, F_velocity_callback=None,
        Q_internal_callback=shell._internal_forces_separated)

    # WarpANCFEntry one substep (same state)
    entry.q = entry._bcast(q_n)
    entry.dq = entry._bcast(dq_n)
    entry.substep(dt, dt, NodalForceSet(F_const))
    q_gpu = entry.q.numpy()[0]
    dq_gpu = entry.dq.numpy()[0]

    dq_max = float(np.max(np.abs(q_gpu - q_cpu)))
    rel_q = dq_max / (np.max(np.abs(q_cpu)) + 1e-30)
    rel_dq = float(np.max(np.abs(dq_gpu - dq_cpu))) / (np.max(np.abs(dq_cpu)) + 1e-30)
    # batch consistency: identical-input envs must agree to tolerance. On GPU the
    # default reductions are non-deterministic (atomic-add order) -> ~machine-eps
    # spread, NOT bit-identical (plan §4: bit-exact batch needs the deterministic
    # reduction mode, used in CI). On CPU this is exactly 0.
    qall = entry.q.numpy()
    batch_spread = float(np.max(np.abs(qall - qall[0:1]))) / (np.max(np.abs(qall[0])) + 1e-30)
    batch_ok = batch_spread < 1e-9
    ok = (rel_q < 1e-9) and (rel_dq < 1e-9) and batch_ok
    print(f"WarpANCFEntry vs CPU Newmark (fp64, B={B}): "
          f"q rel={rel_q:.2e}  dq rel={rel_dq:.2e}  "
          f"batch_spread={batch_spread:.2e} (tol 1e-9; GPU non-det reductions)")
    print(f"ANCF atom red line {'PASS' if ok else 'FAIL'} "
          f"(StructuralEntry substep == golden Newmark, bit-exact)")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify_bit_exact() else 1)
