"""P1 red-line battery for the ANCF atom (run from FLUXV/src).

  1. **Orthotropic reduction** (red line #2): an *orthotropic* shell built with
     Ex=Ey and G_xy=Ex/(2(1+nu)) must reduce to the *isotropic* legacy shell to
     machine precision — Dm, Dk, and the assembled tangent stiffness.
  2. **True orthotropic** sanity: Ex!=Ey builds a symmetric, sensible Dm.
  3. **Bit-exact golden** (red line #1): the ANCF atom substep still equals the
     CPU Newmark reference bit-exact (proves the isotropic path is untouched).

Run: cd FLUXV/src && python ../platform/verify_redlines.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
for p in (_SRC, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                            # noqa: E402
from fluxvortex.ancf_shell import ANCFShell                 # noqa: E402
from run_standalone_yamano import yamano_params, build_yamano_shell  # noqa: E402


def _shell(nodes, quads, h, rho, Ex, Ey, nu, G=None, mode="full"):
    return ANCFShell(nodes, quads, h, rho, Ex=Ex, Ey=Ey, nu_xy=nu, G_xy=G, mode=mode)


def reduction_redline():
    params = yamano_params()
    base, _, _, _ = build_yamano_shell(params, nx=8, ny=6)
    nodes, quads, h, rho = base.nodes, base.quads, base.h, base.rho
    E, nu = base.Ex, base.nu_xy

    iso = _shell(nodes, quads, h, rho, E, E, nu)               # legacy isotropic path
    G_iso = E / (2.0 * (1.0 + nu))
    orth = _shell(nodes, quads, h, rho, E, E, nu, G=G_iso)     # orthotropic branch

    dDm = float(np.max(np.abs(orth.Dm - iso.Dm))) / (np.max(np.abs(iso.Dm)) + 1e-30)
    dDk = float(np.max(np.abs(orth.Dk - iso.Dk))) / (np.max(np.abs(iso.Dk)) + 1e-30)
    # assembled tangent stiffness on a perturbed state
    rng = np.random.default_rng(3)
    q = np.zeros(iso.ndof)
    for k in range(iso.nn):
        q[9 * k] = iso.nodes[k, 0]; q[9 * k + 1] = iso.nodes[k, 1]
        q[9 * k + 3] = 1.0; q[9 * k + 7] = 1.0
    free = np.array(sorted(set(range(iso.ndof)) - set(iso._bc_dofs)))
    q[free] += 1e-3 * rng.standard_normal(len(free))
    Ki = iso._tangent_K_mem(q).toarray()
    Ko = orth._tangent_K_mem(q).toarray()
    dK = float(np.max(np.abs(Ko - Ki))) / (np.max(np.abs(Ki)) + 1e-30)

    ok = (dDm < 1e-12) and (dDk < 1e-12) and (dK < 1e-12)
    print(f"[reduction] Dm rel={dDm:.2e}  Dk rel={dDk:.2e}  K_tangent rel={dK:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def true_orthotropic_sanity():
    params = yamano_params()
    base, _, _, _ = build_yamano_shell(params, nx=8, ny=6)
    E = base.Ex
    s = _shell(base.nodes, base.quads, base.h, base.rho, Ex=E, Ey=0.4 * E, nu=0.3,
               G=0.35 * E)
    Dm = s.Dm
    sym = float(np.max(np.abs(Dm - Dm.T)))      # reciprocity -> symmetric Dm
    distinct = abs(Dm[0, 0] - Dm[1, 1]) / abs(Dm[0, 0])   # Ex!=Ey -> distinct diag
    ok = (sym < 1e-9 * np.max(np.abs(Dm))) and (distinct > 0.1) and (Dm[2, 2] > 0)
    print(f"[orthotropic] Dm symmetric (|Dm-Dmᵀ|={sym:.2e}), "
          f"Ex/Ey anisotropy diag-ratio gap={distinct:.2f}, G={Dm[2,2]:.3e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def golden_bit_exact():
    from ancf_solver import verify_bit_exact   # noqa: PLC0415
    return verify_bit_exact(B=2)


if __name__ == "__main__":
    wp.init()
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # for ancf_solver
    r1 = reduction_redline()
    r2 = true_orthotropic_sanity()
    r3 = golden_bit_exact()
    print(f"\nP1 ANCF red lines: reduction={r1}  orthotropic={r2}  golden-bit-exact={r3}")
    raise SystemExit(0 if (r1 and r2 and r3) else 1)
