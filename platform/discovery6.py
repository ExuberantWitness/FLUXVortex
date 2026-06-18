"""Discovery run #6 — the ranking-inversion SYNERGY on the real (gust × L/D) front.

Discovery #5 (F8) established a clean, real, monotone (gust × induced-L/D) trade-off
from the coupled FSI: gust favors flexible, efficiency favors stiff. The synergy
question (plan §0, FIX 4) is now well-posed on real physics:

  Does closing the gust-rejection control loop move the optimal design along this
  real Pareto front — and in which direction?

For each stiffness we run the SAME cruise-AoA coupled FSI, once passive and once with
the Takens-embedding PD policy inside the predictor-corrector loop, measuring:

  gust_rejection = peak tip excursion relative to the SETTLED cruise deflection
                   (isolates the gust response from the steady cruise sag)
  L/D            = induced aerodynamic efficiency (UVLM, design-discriminating)

Then we compare the passive vs controlled Pareto fronts and the scalarized optimum.
Honest either way: the plan's hypothesis is that control lets the wing be STIFFENED
to its efficient point (optimum shifts toward stiff); the F4/F5 mechanism (control
authority ∝ 1/stiffness) pulls the other way. The data decides.
"""
from __future__ import annotations

import os
import sys
import time

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
from control_eval import ControlledProvider, TakensPolicy       # noqa: E402
from design_map import DesignMap                                # noqa: E402
from power_probe import aero_breakdown, CD_PRO                  # noqa: E402

STIFF = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
AOA_DEG = 6.0


def evaluate_cc(dmap, design, policy, *, aoa_deg=AOA_DEG, n_base=4, n_gust=3,
                n_recover=4, substeps=34, struct_dt=2e-4, gust_w=2.5, device=None):
    """Cruise-AoA coupled FSI, optional control -> (gust_rejection, L/D_induced).

    gust is measured relative to the settled cruise tip (after n_base windows), so it
    is the pure gust response, not the steady cruise sag.
    """
    dev = device or cfg.DEVICE
    shell = dmap.to_shell(design)
    from run_standalone_yamano import yamano_params
    from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
    params = yamano_params()
    V0 = float(params["V_inf"]); rho = float(params["rho_fluid"])
    a = np.deg2rad(aoa_deg)
    Vx, Vz = V0 * np.cos(a), -V0 * np.sin(a)
    solver = StandaloneHybridSolver(
        shell, np.array([Vx, 0.0, Vz]), rho_fluid=rho,
        structural_dt=struct_dt, uvlm_dt_ratio=substeps, integrator="implicit",
        relaxation=1.0, newton_tol=1e-4, max_newton=20, max_particles=5000,
        wake_truncation=5.5, core_radius=1e-6, coupling="strong")
    entry = WarpANCFEntry(shell, B=1, alpha_v=0.5, c_damp=2.0, device=dev)
    tip = shell.nn - 1
    provider = ControlledProvider(solver, entry, wake=False)
    provider.policy = None
    pc = WindowPredictorCorrector(entry=entry, provider=provider, substeps=substeps,
                                  dt=struct_dt, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros((1, entry.ndof), dtype=cfg.NP_DTYPE)))
    pc.advance(n_substeps=1)

    # --- settle the cruise, collect L/D, capture the steady reference tip ---
    cruise = []
    for w in range(n_base):
        provider.fluid.V_inf = np.array([Vx, 0.0, Vz])
        pc.advance()
        out = provider.fluid.solve(entry.q, entry.dq); wp.synchronize()
        cruise.append(aero_breakdown(out[4].numpy()[0], entry.dq.numpy()[0], shell.nn))
    z_ref = float(entry.q.numpy()[0, 9 * tip + 2])
    L = float(np.mean([c["Fz"] for c in cruise]))
    D_ind = float(np.mean([c["Fx"] for c in cruise]))
    LD_ind = L / (abs(D_ind) + 1e-30)

    # --- arm the controller at the cruise reference, inject the gust ---
    if policy is not None:
        policy.reset(); provider.bind_control(policy, tip, z_ref)
    peak = 0.0
    for w in range(n_gust + n_recover):
        gz = 0.0
        if w < n_gust:
            frac = (w + 0.5) / n_gust
            gz = 0.5 * gust_w * (1.0 - np.cos(2.0 * np.pi * frac))
        provider.fluid.V_inf = np.array([Vx, 0.0, Vz + gz])
        pc.advance()
        peak = max(peak, abs(float(entry.q.numpy()[0, 9 * tip + 2]) - z_ref))
    return dict(gust=peak, L_over_D=LD_ind, lift=L, drag_induced=D_ind)


def run():
    from run_standalone_yamano import yamano_params, build_yamano_shell
    params = yamano_params()
    shell0, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    dmap = DesignMap(shell0.nodes, shell0.quads, shell0.h, shell0.rho,
                     shell0.nu_xy, shell0.Ex, shell0._bc_dofs)
    gp, gc, ld = [], [], []
    t0 = time.time()
    for s in STIFF:
        rp = evaluate_cc(dmap, [s, 1.0], None)
        rc = evaluate_cc(dmap, [s, 1.0], TakensPolicy(n_embed=20, kp=50.0, kd=0.5))
        gp.append(rp["gust"]); gc.append(rc["gust"]); ld.append(rp["L_over_D"])
        print(f"  s={s:.2f}: gust_passive={rp['gust']:.4e}  gust_ctrl={rc['gust']:.4e}  "
              f"reduction={100*(1-rc['gust']/rp['gust']):.0f}%  L/D={rp['L_over_D']:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return (np.array(STIFF), np.array(gp), np.array(gc), np.array(ld))


def _norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-30)


def analyze(stiff, gp, gc, ld):
    # scalarize: cost = norm(gust) + norm(efficiency_cost); efficiency_cost = -L/D
    eff_cost = _norm(-ld)
    Jp = _norm(gp) + eff_cost
    Jc = _norm(gc) + eff_cost
    bp, bc = int(np.argmin(Jp)), int(np.argmin(Jc))
    print("\n=== synergy on the real (gust × L/D) front ===")
    print("  stiffness    : " + "  ".join(f"{s:.2f}" for s in stiff))
    print("  gust passive : " + "  ".join(f"{v:.2e}" for v in gp))
    print("  gust control : " + "  ".join(f"{v:.2e}" for v in gc))
    print("  L/D          : " + "  ".join(f"{v:.2f}" for v in ld))
    print("  reduction    : " + "  ".join(f"{100*(1-c/p):.0f}%" for p, c in zip(gp, gc)))
    print("  J decoupled  : " + "  ".join(f"{v:.2f}" for v in Jp))
    print("  J co-design  : " + "  ".join(f"{v:.2f}" for v in Jc))
    print("\n=== FINDING ===")
    print(f"  optimal DESIGN-ALONE  (passive gust + L/D): stiffness={stiff[bp]:.2f}")
    print(f"  optimal DESIGN+CONTROL (ctrl gust + L/D):   stiffness={stiff[bc]:.2f}")
    red = 1 - gc / gp
    auth = float(np.corrcoef(stiff, red)[0, 1])
    if bc > bp + 1e-9:
        print(f"  -> SYNERGY (plan direction): control SHIFTS the optimum "
              f"{stiff[bp]:.2f} -> {stiff[bc]:.2f} (stiffer/efficient). Closing the loop "
              f"lets the wing bank efficiency — co-design beats decoupled.")
    elif bc < bp - 1e-9:
        print(f"  -> ANTI-SYNERGY: control shifts the optimum {stiff[bp]:.2f} -> "
              f"{stiff[bc]:.2f} (more flexible). Control authority ∝ 1/stiffness (F4) "
              f"reinforces flexibility; the single wing cannot host the plan's inversion.")
    else:
        print(f"  -> NO SHIFT: optimum {stiff[bp]:.2f} under both. Control compresses the "
              f"gust axis but does not flip the discrete optimum on the single wing.")
    print(f"  control authority vs stiffness corr = {auth:+.2f} "
          f"({'flexible controls better' if auth < -0.2 else 'stiffer controls better' if auth > 0.2 else 'stiffness-independent'})")
    return bc, bp


if __name__ == "__main__":
    wp.init()
    print(cfg.summary())
    print(f"discovery #6: synergy on real (gust × L/D) front, AoA={AOA_DEG} deg, "
          f"{len(STIFF)} stiffnesses × (passive, controlled)")
    stiff, gp, gc, ld = run()
    np.savez(os.path.join(_FLUXV, "docs", "discovery6.npz"),
             stiff=stiff, gust_passive=gp, gust_control=gc, ld=ld)
    analyze(stiff, gp, gc, ld)
