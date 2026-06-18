"""FIX 1 — real aerodynamic power / efficiency from the coupled FSI rollout.

Replaces codesign_eval's `efficiency = mean(|F|)` proxy (coupled_fsi.py:64) with
quantities computed from the *real* predictor-corrector rollout:

  lift   L      = sum_nodes Fz            (vertical aero force, N)
  drag   D_ind  = sum_nodes Fx            (streamwise aero force = induced drag, N)
  side   Y      = sum_nodes Fy
  aero power P  = sum_dof  F . v_struct   (rate of work aero does on the structure, W)
  efficiency    = L / D_total  with  D_total = D_ind + D_pro (profile strip term)

The induced part (L, D_ind, P) is computed bit-for-bit from the validated UVLM
nodal force `Fbern` and the ANCF structural velocity `dq` — no analytical model.
The profile-drag term D_pro = 0.5*rho*V^2*S*CD_pro is the documented strip
viscous correction (plan §2; cot.CD_pro), needed because inviscid UVLM has no
skin friction. Both are reported separately so the inviscid measurement is honest.

A flat plate at zero AoA makes ~zero lift, so the probe runs at a cruise AoA
(freestream tilted) to put the wing in a real lift-producing cruise — the regime
where "flight efficiency" is defined. The gust is then injected on top, so one
rollout yields BOTH co-design objectives: (gust_rejection, L/D).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.join(_FLUXV, "src"), os.path.join(_FLUXV, "tests"),
          os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
from fluxvortex.warp_fsi import config as cfg                   # noqa: E402
from newton_pc import WindowPredictorCorrector                  # noqa: E402
from ancf_solver import WarpANCFEntry, NodalForceSet            # noqa: E402
from coupled_fsi import UVLMForceProvider                       # noqa: E402
from design_map import DesignMap                                # noqa: E402

CD_PRO = 0.04        # profile drag coeff (Zhong&Xu/cot.py); strip viscous correction


def aero_breakdown(Fbern_b: np.ndarray, dq_b: np.ndarray, nn: int) -> dict:
    """Decompose one env's generalized nodal force (ndof,) into physical aero loads.

    Fbern/dq are 9-DOF/node ANCF generalized vectors; the first 3 of each node are
    the translational force/velocity. L=sum Fz, D_ind=sum Fx, and the *generalized*
    power F.v uses the full vector (includes slope-DOF moments — the true work rate).
    """
    F9 = Fbern_b.reshape(nn, 9)
    Fx = float(F9[:, 0].sum())
    Fy = float(F9[:, 1].sum())
    Fz = float(F9[:, 2].sum())
    P = float(np.dot(Fbern_b, dq_b))            # generalized aero power (W)
    return dict(Fx=Fx, Fy=Fy, Fz=Fz, P=P)


def evaluate_real(dmap: DesignMap, design, *, aoa_deg=6.0, n_base=3, n_gust=3,
                  n_recover=3, substeps=34, struct_dt=2e-4, gust_w=2.0,
                  c_damp=2.0, device=None):
    """Coupled FSI at a cruise AoA + gust -> REAL (gust_rejection, L/D, power).

    Returns dict with gust_rejection (peak tip excursion, smaller better), and the
    cruise-window means lift/drag/L_over_D/P_aero (computed from the rollout).
    """
    dev = device or cfg.DEVICE
    shell = dmap.to_shell(design)
    from run_standalone_yamano import yamano_params
    from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
    params = yamano_params()
    V0 = float(params["V_inf"])
    rho = float(params["rho_fluid"])
    a = np.deg2rad(aoa_deg)
    # tilt the freestream down so the flat plate sees a cruise AoA (-> steady lift)
    Vx, Vz = V0 * np.cos(a), -V0 * np.sin(a)
    S = float(np.ptp(shell.nodes[:, 0]) * np.ptp(shell.nodes[:, 1]))   # planform area
    qdyn = 0.5 * rho * V0 * V0

    solver = StandaloneHybridSolver(
        shell, np.array([Vx, 0.0, Vz]), rho_fluid=rho,
        structural_dt=struct_dt, uvlm_dt_ratio=substeps, integrator="implicit",
        relaxation=1.0, newton_tol=1e-4, max_newton=20, max_particles=5000,
        wake_truncation=5.5, core_radius=1e-6, coupling="strong")
    entry = WarpANCFEntry(shell, B=1, alpha_v=0.5, c_damp=c_damp, device=dev)
    provider = UVLMForceProvider(solver, entry, wake=False)
    pc = WindowPredictorCorrector(entry=entry, provider=provider, substeps=substeps,
                                  dt=struct_dt, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros((1, entry.ndof), dtype=cfg.NP_DTYPE)))
    pc.advance(n_substeps=1)
    tip = shell.nn - 1
    z0 = float(entry.q.numpy()[0, 9 * tip + 2])

    tip_z = []
    cruise = []          # aero breakdown during the steady cruise (pre-gust) windows
    total = n_base + n_gust + n_recover
    for w in range(total):
        gz = 0.0
        if n_base <= w < n_base + n_gust:
            frac = (w - n_base + 0.5) / n_gust
            gz = 0.5 * gust_w * (1.0 - np.cos(2.0 * np.pi * frac))
        provider.fluid.V_inf = np.array([Vx, 0.0, Vz + gz])
        pc.advance()
        tip_z.append(float(entry.q.numpy()[0, 9 * tip + 2]) - z0)
        # re-solve at the committed state to pair F with the marched velocity
        out = provider.fluid.solve(entry.q, entry.dq)
        wp.synchronize()
        Fbern = out[4].numpy()[0]
        bd = aero_breakdown(Fbern, entry.dq.numpy()[0], shell.nn)
        if w < n_base:                       # steady cruise window (no gust yet)
            cruise.append(bd)

    tip_z = np.array(tip_z)
    L = float(np.mean([c["Fz"] for c in cruise]))
    D_ind = float(np.mean([c["Fx"] for c in cruise]))
    P_aero = float(np.mean([abs(c["P"]) for c in cruise]))
    D_pro = qdyn * S * CD_PRO                 # profile strip term (documented)
    D_tot = abs(D_ind) + D_pro
    LD = L / (D_tot + 1e-30)
    LD_ind = L / (abs(D_ind) + 1e-30)         # design-discriminating (pure UVLM, no model)
    return dict(gust_rejection=float(np.max(np.abs(tip_z))),
                lift=L, drag_induced=D_ind, drag_profile=D_pro, drag_total=D_tot,
                L_over_D=LD, L_over_D_induced=LD_ind, P_aero=P_aero, tip_z=tip_z, area=S)


def _probe():
    """Quick look at REAL aero numbers for a flexible vs stiff wing at cruise AoA."""
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    print("FIX 1 power probe — real aero from coupled FSI at cruise AoA")
    for name, d in {"flexible": [0.6, 1.0], "stiff": [1.6, 1.0]}.items():
        r = evaluate_real(dmap, d, aoa_deg=6.0)
        print(f"  {name:9s} {d}: gust={r['gust_rejection']:.3e}  "
              f"L={r['lift']:+.3e}N  D_ind={r['drag_induced']:+.3e}N  "
              f"D_pro={r['drag_profile']:.3e}N  L/D={r['L_over_D']:+.2f}  "
              f"P_aero={r['P_aero']:.3e}W", flush=True)


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    _probe()
