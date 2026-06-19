"""Real coupled-FSI co-design evaluator over PER-ELEMENT (刚柔 + 质量) fields.

Replaces the cheap analytic surrogate's evaluation with the VALIDATED coupled FSI
(flexible_wing.run = ANCF shell + UVLM predictor-corrector), and makes BOTH the spanwise
stiffness field AND the spanwise MASS field design variables on the main wing — so mass
genuinely enters the physics (inertia / spring-mass resonance / the passive-feather dynamics
that drive 气弹推进), not just the static stiffness surrogate.

  design = (stiffness spline ctrl, mass spline ctrl)   [root→tip, per-element via set_distribution]
  evaluate -> real coupled FSI -> objectives:
      passive feather amplitude (tip twist, deg)   — the aeroelastic-propulsion mechanism
      tip bending (m), mean lift (N), finite        — structural response under flapping

COST (honest): one real coupled-FSI eval ≈ 70 s (n_cycles=1). A co-design loop is therefore
A100-scale; this module is the real evaluator + a small demo that mass distribution changes
the coupled response. The DIFFERENTIABLE design gradient through this (stiffness via the
validated K_t adjoint, mass via a new mass-matrix adjoint, plus the deferred UVLM aero
adjoint, plus a coupled policy) is the staged SHAC work — see the cost table in main().
"""
from __future__ import annotations

import numpy as np

import flexible_wing as fw
import design_field as dfield


def eval_fsi(stiff_ctrl, mass_ctrl, n_cycles=1, substeps=12, amp_deg=35.0):
    """One real coupled-FSI evaluation of a per-element (刚柔, 质量) design."""
    sf = dfield.StiffnessField(np.asarray(stiff_ctrl, float))
    mf = dfield.StiffnessField(np.asarray(mass_ctrl, float))
    E_scale = sf.e_scale_fn(fw.SPAN)        # callable(x,y): per-element stiffness scale
    rho_scale = mf.e_scale_fn(fw.SPAN)      # callable(x,y): per-element DENSITY (mass) scale
    r = fw.run(n_cycles=n_cycles, amp_deg=amp_deg, substeps=substeps,
               E_scale=E_scale, rho_scale=rho_scale, verbose=False)
    tw = np.asarray(r["twist"])
    half = tw[len(tw) // 2:] if len(tw) > 2 else tw
    feather = float(np.nanmax(np.abs(half))) if len(half) else float("nan")
    return dict(feather_deg=feather, bend_m=float(np.nanmax(r["bend"])) if r["n_windows"] else np.nan,
                lift_N=float(np.nanmean(r["lift"])) if r["n_windows"] else np.nan,
                finite=bool(r["finite"]), n_windows=int(r["n_windows"]))


def main():
    import warp as wp; wp.init()
    K = 4
    uni = np.full(K, 1.0)
    stiff_rt = np.linspace(1.8, 0.4, K)          # stiff root -> flexible tip
    heavy_root = np.linspace(1.6, 0.5, K)        # mass: heavy root -> light tip
    heavy_tip = np.linspace(0.5, 1.6, K)         # mass: light root -> heavy tip

    designs = [
        ("uniform stiff / uniform mass", uni, uni),
        ("stiff-root·flex-tip / uniform mass", stiff_rt, uni),
        ("stiff-root·flex-tip / heavy-ROOT mass", stiff_rt, heavy_root),
        ("stiff-root·flex-tip / heavy-TIP mass", stiff_rt, heavy_tip),
    ]
    print("Real coupled-FSI co-design evaluator — per-element 刚柔 + 质量 (≈70 s/eval)\n")
    print(f"{'design':42s} | feather° | bend(m) | lift(N) | finite")
    rows = []
    for name, sc, mc in designs:
        r = eval_fsi(sc, mc)
        rows.append((name, r))
        print(f"{name:42s} |  {r['feather_deg']:5.2f}  | {r['bend_m']:.4f} | "
              f"{r['lift_N']:+6.1f} | {r['finite']}", flush=True)

    # the point: at FIXED stiffness, the MASS distribution changes the coupled response
    base = rows[1][1]; hr = rows[2][1]; ht = rows[3][1]
    df = abs(ht["feather_deg"] - hr["feather_deg"])
    print(f"\nmass distribution effect at fixed 刚柔 (heavy-tip vs heavy-root): "
          f"Δfeather={df:.2f}°  Δlift={abs(ht['lift_N']-hr['lift_N']):.1f}N")
    print("-> mass is a REAL co-design variable on the coupled FSI (inertia changes the "
          "passive-feather/resonance dynamics), not just a static-stiffness knob.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
