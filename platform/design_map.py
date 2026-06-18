"""DesignMap — design vector -> ANCF material -> structural response (co-design core).

The heart of co-design (plan §5): a low-dim design vector maps to per-shell ANCF
material (here a stiffness scale and chordwise/spanwise orthotropy ratio), the
platform evaluates a structural response, and the optimizer (MAP-Elites + DQD)
searches the design space. This minimal map demonstrates the forward
design->response signal and its sensitivity (the DQD gradient direction) on the
real GPU structural atom.

  design d = [s_stiff, r_ortho]  ->  Ex = E0*s_stiff,  Ey = Ex*r_ortho
  response(d) = tip deflection after N differentiable structural steps under a
                reference load (uses DiffStructStep; design enters via K_t).

verify: response varies smoothly with the design, and the finite-difference design
sensitivity d(response)/d(d) is finite and sensible (stiffer -> smaller deflection).
This is iteration-1 co-design's forward evaluation; PPO/MOME plug in on top.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
for p in (_SRC, _TESTS, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from fluxvortex.warp_fsi.config import DTYPE                    # noqa: E402
from fluxvortex.ancf_shell import ANCFShell                    # noqa: E402
from ancf_solver import WarpANCFEntry, NodalForceSet            # noqa: E402


class DesignMap:
    """Low-dim design vector -> ANCF shell material. Owns the fixed mesh/geometry."""

    def __init__(self, nodes, quads, h, rho, nu, E0, bc_dofs):
        self.nodes, self.quads = nodes, quads
        self.h, self.rho, self.nu, self.E0 = h, rho, nu, E0
        self._bc = bc_dofs

    def to_shell(self, design):
        s_stiff, r_ortho = float(design[0]), float(design[1])
        Ex = self.E0 * s_stiff
        Ey = Ex * r_ortho
        G = Ex / (2.0 * (1.0 + self.nu))            # explicit -> orthotropic path
        shell = ANCFShell(self.nodes, self.quads, self.h, self.rho,
                          Ex=Ex, Ey=Ey, nu_xy=self.nu, G_xy=G, mode="full")
        return shell


def evaluate(dmap: DesignMap, design, N=120, dt=2e-4, tip=None, device=None):
    """Forward response: settled tip |z| deflection under a reference tip load,
    via the *unconditionally stable* implicit Newmark (WarpANCFEntry)."""
    dev = device or cfg.DEVICE
    shell = dmap.to_shell(design)
    ndof = shell.nn * 9
    free = set(range(ndof)) - set(shell._bc_dofs)
    tip = shell.nn - 1 if tip is None else tip
    F = np.zeros(ndof)
    if (9 * tip + 2) in free:
        F[9 * tip + 2] = 5.0                            # reference z-load at the tip
    entry = WarpANCFEntry(shell, B=1, alpha_v=0.5, c_damp=2.0, device=dev)
    z0 = float(entry.q.numpy()[0][9 * tip + 2])
    forces = NodalForceSet(F)
    for k in range(N):
        entry.substep((k + 1) * dt, dt, forces)
    wp.synchronize()
    return abs(float(entry.q.numpy()[0][9 * tip + 2]) - z0)


def verify() -> bool:
    dev = cfg.DEVICE
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=6, ny=4)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)

    # response vs stiffness scale: should DECREASE as the wing stiffens
    scales = [0.6, 1.0, 1.6]
    resp = [evaluate(dmap, [s, 1.0], device=dev) for s in scales]
    monotone = resp[0] > resp[1] > resp[2]

    # FD design sensitivity d(response)/d(s_stiff) at s=1 (the DQD gradient direction)
    eps = 1e-3
    rp = evaluate(dmap, [1.0 + eps, 1.0], device=dev)
    rm = evaluate(dmap, [1.0 - eps, 1.0], device=dev)
    dresp_ds = (rp - rm) / (2 * eps)
    # orthotropy axis responds too
    r_ortho_hi = evaluate(dmap, [1.0, 1.6], device=dev)
    ortho_sens = abs(r_ortho_hi - resp[1]) > 1e-9

    ok = monotone and np.isfinite(dresp_ds) and dresp_ds < 0 and ortho_sens
    print(f"DesignMap forward + sensitivity (co-design core):")
    print(f"  response(stiffness scale 0.6,1.0,1.6) = "
          f"[{resp[0]:.4e}, {resp[1]:.4e}, {resp[2]:.4e}]  stiffer->smaller={monotone}")
    print(f"  d(response)/d(stiffness) @1.0 = {dresp_ds:+.3e} (FD, DQD direction; <0 expected)")
    print(f"  orthotropy axis active: {ortho_sens}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: design->material->response map works "
          f"(iteration-1 co-design forward signal)")
    return ok


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    raise SystemExit(0 if verify() else 1)
