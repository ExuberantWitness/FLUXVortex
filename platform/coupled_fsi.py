"""P1 atom #4 — differentiable predictor-corrector coupled FSI loop (iteration 1).

Wires the validated atoms into one coupled fluid-structure loop driven by the
newton_pc **window predictor-corrector**:

  WarpANCFEntry (structural, §ancf_solver)         <-- two-pass PC -->   UVLMForceProvider (aero)

The aero ForceProvider reuses warp_fsi's `GpuFluidSolve` — its `.solve(q, dq)`
returns the assembled nodal pressure force `Fbern` (via the bit-exact `_P_load`
shape-function transfer, COUPLING red line). First increment runs the **bound
UVLM Surface without free wake** (`wake=False`): the two-pass PC calls `solve`
twice per window, and a free wake would double-advect; the bound solve is
re-entrant and is exactly the "Surface" half of the plan's UVLM atom. Free-wake
coupling (commit-time advance) is the next increment.

Verification (`verify_coupled`): run a short coupled trajectory on GPU and check
it is stable (no NaN, bounded tip), the aero coupling is engaged (nonzero force),
and the two-pass vs lagged schemes differ as expected (PC actually corrects).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_SRC, _TESTS, _ROOT, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                            # noqa: E402
from fluxvortex.warp_fsi import config as cfg               # noqa: E402
from fluxvortex.warp_fsi.coupled import GpuFluidSolve       # noqa: E402
from newton_pc import WindowPredictorCorrector              # noqa: E402
from ancf_solver import WarpANCFEntry, NodalForceSet        # noqa: E402


# ── aero ForceProvider (bound UVLM Surface) ───────────────────────────────────
class UVLMForceProvider:
    """ForceProvider: solve bound UVLM at the entry's deformed state -> nodal force.

    Reuses GpuFluidSolve (validated L1-L6 + _P_load transfer). solve(state) reads
    the bound structural state (q, dq) the predictor-corrector just marched the
    entry to, solves the deformed-geometry UVLM, and returns the assembled nodal
    pressure force Fbern as an interpolable NodalForceSet.
    """

    def __init__(self, solver, entry: WarpANCFEntry, wake: bool = False, device=None):
        self.device = device or cfg.DEVICE
        self.entry = entry
        self.fluid = GpuFluidSolve(solver, wake=wake, device=self.device)
        self.n_solves = 0
        self.last_force_norm = 0.0

    def solve(self, state) -> NodalForceSet:
        out = self.fluid.solve(self.entry.q, self.entry.dq)   # (dp, dp2, gamma, Vb, Fbern)
        Fbern = out[4]
        wp.synchronize()
        gen = Fbern.numpy()                                   # (B, ndof)
        self.n_solves += 1
        self.last_force_norm = float(np.linalg.norm(gen[0]))
        return NodalForceSet(gen)

    def commit(self, forces: NodalForceSet) -> None:
        # bound (no free wake) -> nothing to advance; free-wake commit is next increment
        pass


# ── short coupled-trajectory verification ─────────────────────────────────────
def verify_coupled(B: int = 2, n_windows: int = 12, substeps: int = 34,
                   struct_dt: float = 2e-4) -> bool:
    from run_standalone_yamano import yamano_params, build_yamano_shell
    from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    solver = StandaloneHybridSolver(
        shell, np.array([params["V_inf"], 0.0, 0.0]), rho_fluid=params["rho_fluid"],
        structural_dt=struct_dt, uvlm_dt_ratio=substeps, integrator="implicit",
        relaxation=1.0, newton_tol=1e-4, max_newton=20, max_particles=5000,
        wake_truncation=5.5, core_radius=1e-6, coupling="strong")

    def run(mode):
        entry = WarpANCFEntry(shell, B=B, alpha_v=0.5, c_damp=2.0)
        # seed a small tip-ward heave perturbation on free DOFs (z-position)
        q0 = entry._q0.copy()
        for k in range(shell.nn):
            x = shell.nodes[k, 0]
            if 9 * k not in shell._bc_dofs:
                q0[9 * k + 2] += 0.01 * x * x
        entry.q = entry._bcast(q0)
        provider = UVLMForceProvider(solver, entry, wake=False)
        pc = WindowPredictorCorrector(entry=entry, provider=provider,
                                      substeps=substeps, dt=struct_dt, mode=mode)
        pc.initialize(NodalForceSet(np.zeros((B, entry.ndof), dtype=cfg.NP_DTYPE)))
        pc.advance(n_substeps=1)
        tip = shell.nn - 1
        tip_z, fnorms = [], []
        for w in range(n_windows):
            pc.advance()
            tip_z.append(float(entry.q.numpy()[0, 9 * tip + 2]))
            fnorms.append(provider.last_force_norm)
        return np.array(tip_z), np.array(fnorms), provider.n_solves

    tip_pc, f_pc, ns_pc = run("two-pass")
    tip_lag, f_lag, _ = run("lagged")

    finite = bool(np.all(np.isfinite(tip_pc)) and np.all(np.isfinite(f_pc)))
    bounded = bool(np.max(np.abs(tip_pc)) < 1.0)               # no blow-up (L=1m plate)
    aero_engaged = bool(np.mean(f_pc) > 1e-6)                  # nonzero aero force
    pc_corrects = bool(np.max(np.abs(tip_pc - tip_lag)) > 0)   # PC differs from lagged
    ok = finite and bounded and aero_engaged and pc_corrects

    print(f"[coupled FSI] two-pass: {n_windows} windows x {substeps} substeps, B={B}, "
          f"{ns_pc} fluid solves")
    print(f"  tip_z range=[{tip_pc.min():+.4e},{tip_pc.max():+.4e}]  "
          f"finite={finite} bounded={bounded}")
    print(f"  mean aero |F|={np.mean(f_pc):.3e}  aero_engaged={aero_engaged}")
    print(f"  max|tip_PC - tip_lagged|={np.max(np.abs(tip_pc-tip_lag)):.3e}  "
          f"pc_corrects={pc_corrects}")
    print(f"coupled FSI loop (GPU two-pass predictor-corrector) "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify_coupled() else 1)
