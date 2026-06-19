"""End-to-end real-coupled-FSI co-design pipeline (4090 run; A100 for detailed sweeps).

Runs the WHOLE co-design flow on the real physics, not the cheap surrogate:
  1. seed the stiffness field from the validated cheap-surrogate optimum (stiff-root/flex-tip
     — what the differentiable DQD/SHAC gradients already found);
  2. pattern-search the per-element (刚柔, 质量) fields, evaluating EACH candidate on the real
     coupled FSI (flexible_wing.run = ANCF shell + UVLM predictor-corrector, ≈70 s/eval);
  3. objective from the real FSI: healthy passive feathering (气弹推进) WITHOUT the over-flex
     lift blow-up — exactly the failure the mass distribution tames (Stage-1 finding), so the
     co-design has to balance 刚柔 against 质量 on the true aeroelastic dynamics.

This is the DEV-scale proof of the full pipeline on a 4090 (slow but real). Detailed
parameter / hyper-parameter sweeps (design ranges, objective weights, gust models, longer
rollouts) go to A100. The control policy stays on the validated cheap surrogate here; the
joint policy+structure SHAC on the coupled FSI is the A100 Stage-3/4 capability.
"""
from __future__ import annotations

import os
import time

import numpy as np

import flexible_wing as fw
import design_field as dfield

LIFT_CAP, BEND_CAP = 60.0, 0.60          # physical caps (over-flex blow-up / excessive bending)
K = 4


def real_eval(stiff_rt, mass_rt, n_cycles=1, substeps=12):
    """One real coupled-FSI evaluation of a (刚柔, 质量) design given as (root, tip) of each
    spanwise spline. Returns the FSI metrics + a co-design score."""
    sf = dfield.StiffnessField(np.linspace(stiff_rt[0], stiff_rt[1], K))
    mf = dfield.StiffnessField(np.linspace(mass_rt[0], mass_rt[1], K))
    r = fw.run(n_cycles=n_cycles, amp_deg=35.0, substeps=substeps,
               E_scale=sf.e_scale_fn(fw.SPAN), rho_scale=mf.e_scale_fn(fw.SPAN), verbose=False)
    tw = np.asarray(r["twist"]); half = tw[len(tw) // 2:] if len(tw) > 2 else tw
    feather = float(np.nanmax(np.abs(half))) if len(half) else 0.0
    bend = float(np.nanmax(r["bend"])) if r["n_windows"] else 1e9
    lift = float(np.nanmean(r["lift"])) if r["n_windows"] else 0.0
    finite = bool(r["finite"]) and np.isfinite(feather) and np.isfinite(bend)
    # score: reward passive feathering (propulsion), penalize the over-flex lift spike that
    # mass distribution must tame, and excessive bending; hard-fail on divergence.
    score = (feather - 0.02 * max(0.0, abs(lift) - LIFT_CAP) - 8.0 * max(0.0, bend - BEND_CAP)
             if finite else -1e6)
    return dict(feather=feather, bend=bend, lift=lift, finite=finite, score=float(score))


def pipeline(log=print):
    t_start = time.time()
    # 1) stiffness seed = the cheap-surrogate optimum (stiff root, flexible tip)
    stiff_rt = [1.8, 0.4]
    # 2) pattern search over the MASS field (root, tip) on the REAL coupled FSI
    mass_rt = np.array([1.0, 1.0]); step = 0.6
    evals = []

    def ev(srt, mrt, tag):
        r = real_eval(srt, list(mrt))
        evals.append(dict(stiff=list(srt), mass=list(mrt), tag=tag, **r))
        log(f"  [{len(evals):2d}] {tag:18s} 刚柔={srt} 质量=[{mrt[0]:.2f},{mrt[1]:.2f}] "
            f"-> feather={r['feather']:.2f}° lift={r['lift']:+7.1f}N bend={r['bend']:.3f} "
            f"score={r['score']:+.2f}  ({time.time()-t_start:.0f}s)", flush=True)
        return r["score"]

    best_s = ev(stiff_rt, mass_rt, "seed (uniform mass)")
    best_m = mass_rt.copy()
    for it in range(3):                              # 3 pattern-search refinements
        improved = False
        for axis in (0, 1):                          # mass root, mass tip
            for d in (+step, -step):
                cand = best_m.copy(); cand[axis] = float(np.clip(cand[axis] + d, 0.4, 2.0))
                if np.allclose(cand, best_m):
                    continue
                s = ev(stiff_rt, cand, f"mass move ax{axis}{'+' if d>0 else '-'}")
                if s > best_s:
                    best_s, best_m, improved = s, cand.copy(), True
        # one stiffness-tip refinement per round (couple 刚柔 with the found 质量)
        for d in (+0.3, -0.3):
            cand = [stiff_rt[0], float(np.clip(stiff_rt[1] + d, 0.3, 1.5))]
            s = ev(cand, best_m, f"stiff-tip move {'+' if d>0 else '-'}")
            if s > best_s:
                best_s, stiff_rt = s, cand
        if not improved:
            step *= 0.5
        log(f"  -- round {it}: best score={best_s:+.2f} 刚柔={stiff_rt} 质量="
            f"[{best_m[0]:.2f},{best_m[1]:.2f}]  step={step:.2f}", flush=True)

    best = max(evals, key=lambda e: e["score"])
    log(f"\nBEST real-FSI design: 刚柔(root,tip)={best['stiff']} 质量(root,tip)={best['mass']}")
    log(f"  feather={best['feather']:.2f}°  lift={best['lift']:+.1f}N  bend={best['bend']:.3f}m  "
        f"score={best['score']:+.2f}")
    log(f"  ({len(evals)} real coupled-FSI evals, {time.time()-t_start:.0f}s total on this GPU)")
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "codesign_fsi_pipeline.npz")
    np.savez(out, evals=np.array([(e["stiff"][0], e["stiff"][1], e["mass"][0], e["mass"][1],
                                   e["feather"], e["lift"], e["bend"], e["score"])
                                  for e in evals]),
             best=np.array([best["stiff"][0], best["stiff"][1], best["mass"][0], best["mass"][1],
                            best["feather"], best["lift"], best["bend"], best["score"]]))
    log(f"  saved -> {out}")
    return evals, best


if __name__ == "__main__":
    import warp as wp; wp.init()
    print("End-to-end real-coupled-FSI co-design (刚柔 + 质量) on this GPU (≈70 s/eval)\n")
    pipeline()
