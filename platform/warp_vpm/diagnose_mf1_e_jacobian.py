"""D1 analysis: 2x2 degeneracy Jacobian of (Cn, |zmax|/c) wrt (ln mf1_scale, ln E).

Reads /tmp/d1_{base,mf1,e,both}.json (200-step A16 runs) and reports, for three windows:

  W1 = steps 101..150   (t* in [1.01, 1.50])
  W2 = steps 151..200   (t* in [1.51, 2.00])
  Wp = steps 101..200   (pooled)

  f0 = window mean of Cn,  f1 = window mean of |instantaneous_zmax_over_c|

  J[:,0] = (f(mf1 x1.05) - f(base)) / ln(1.05)   change per unit ln(Mf1 scale)
  J[:,1] = (f(E   x1.05) - f(base)) / ln(1.05)   change per unit ln(E)

Reported: the two columns, their norms, the angle between them (0 deg = perfectly
degenerate), det, |det|/||J||_F^2 (scale-free rank-deficiency measure), cond(J), and a
first-order validity check (predicted vs actual response to perturbing BOTH).

Scope limit that must travel with these numbers: the ramp ends at step 100 and the
membrane is still in its startup transient at t* in [1,2], so this is the local
Jacobian of the EARLY-TRANSIENT map.  It is not the Jacobian of the converged window
(t* >= 20) where the experimental residual is defined, and therefore it is NOT used to
solve J dp = residual.
"""

from __future__ import annotations

import json
import math

import numpy as np

LN = math.log(1.05)
WINDOWS = {
    "W1_101_150": (101, 150),
    "W2_151_200": (151, 200),
    "Wp_101_200": (101, 200),
}


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    records = {rec["aero_step"]: rec for rec in payload["records"]}
    return payload, records


def window_mean(records, lo, hi):
    cn = [records[s]["cn"] for s in range(lo, hi + 1)]
    zm = [abs(records[s]["instantaneous_zmax_over_c"]) for s in range(lo, hi + 1)]
    return np.array([float(np.mean(cn)), float(np.mean(zm))])


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(
            "usage: diagnose_mf1_e_jacobian.py BASE.json MF1.json E.json BOTH.json\n"
            "  BASE = FLUXV_MF1_SCALE unset, E = nominal\n"
            "  MF1  = FLUXV_MF1_SCALE=1.05, E = nominal\n"
            "  E    = FLUXV_MF1_SCALE=1.0,   E = 1.05x nominal\n"
            "  BOTH = both perturbed (first-order validity check)"
        )
    payloads = dict(zip(("base", "mf1", "e", "both"), (load(p) for p in argv)))
    for tag, (payload, records) in payloads.items():
        print(
            f"{tag:5s} E={payload['E']:.6g} steps={payload['aero_steps']} "
            f"elapsed={payload['elapsed_seconds']:.0f}s "
            f"wall={payload['step_wall_seconds_mean']:.3f}s/step records={len(records)}"
        )
    print()

    summary = {}
    for name, (lo, hi) in WINDOWS.items():
        f = {tag: window_mean(records, lo, hi) for tag, (_, records) in payloads.items()}
        jacobian = np.column_stack(
            ((f["mf1"] - f["base"]) / LN, (f["e"] - f["base"]) / LN)
        )
        det = float(np.linalg.det(jacobian))
        frob2 = float(np.linalg.norm(jacobian) ** 2)
        col_norms = np.linalg.norm(jacobian, axis=0)
        cosine = float(
            jacobian[:, 0] @ jacobian[:, 1] / max(col_norms[0] * col_norms[1], 1e-300)
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        condition = float(np.linalg.cond(jacobian))
        predicted_both = f["base"] + jacobian @ np.array([LN, LN])
        actual_both = f["both"]
        moved = np.abs(actual_both - f["base"])
        second_order = (actual_both - predicted_both) / np.maximum(moved, 1e-300)
        summary[name] = dict(
            jacobian=jacobian, det=det, frob2=frob2, angle=angle,
            condition=condition, second_order=second_order,
        )

        print(f"--- {name}: steps {lo}..{hi} ---")
        print(f"  base   Cn={f['base'][0]:+.9f}   |zmax|/c={f['base'][1]:.6e}")
        for tag in ("mf1", "e", "both"):
            print(
                f"  {tag:5s}  Cn={f[tag][0]:+.9f}   |zmax|/c={f[tag][1]:.6e}"
                f"   dCn={f[tag][0]-f['base'][0]:+.3e}"
                f"  d|zmax|/c={f[tag][1]-f['base'][1]:+.3e}"
            )
        print(
            f"  J(:,lnMf1) = [{jacobian[0,0]:+.6e}, {jacobian[1,0]:+.6e}]  "
            f"norm={col_norms[0]:.4e}"
        )
        print(
            f"  J(:,lnE)   = [{jacobian[0,1]:+.6e}, {jacobian[1,1]:+.6e}]  "
            f"norm={col_norms[1]:.4e}"
        )
        print(
            f"  angle={angle:.3f} deg   det={det:+.4e}   "
            f"|det|/||J||^2={abs(det)/max(frob2,1e-300):.4f}   cond={condition:.3f}"
        )
        print(
            f"  first-order: predicted both=[{predicted_both[0]:+.9f},"
            f"{predicted_both[1]:.6e}] actual=[{actual_both[0]:+.9f},"
            f"{actual_both[1]:.6e}] 2nd-order/moved="
            f"[{second_order[0]:+.3f}, {second_order[1]:+.3f}]"
        )
        # Per-observable leverage ratio: how many units of ln(E) one unit of lnMf1
        # is worth, for each observable separately.  Equal ratios for both
        # observables is the classic signature of collinearity.
        for index, label in enumerate(("Cn", "zmax")):
            ratio = jacobian[index, 0] / jacobian[index, 1] if jacobian[index, 1] else float("nan")
            print(f"  leverage ratio d{label}/dlnMf1 : d{label}/dlnE = {ratio:+.4f}")
        print()

    print("--- window stability (same runs, disjoint windows) ---")
    w1, w2 = summary["W1_101_150"], summary["W2_151_200"]
    print(f"  angle: W1={w1['angle']:.3f}  W2={w2['angle']:.3f} deg")
    print(f"  cond : W1={w1['condition']:.3f}  W2={w2['condition']:.3f}")
    print(
        f"  |det|/||J||^2: W1={abs(w1['det'])/max(w1['frob2'],1e-300):.4f}"
        f"  W2={abs(w2['det'])/max(w2['frob2'],1e-300):.4f}"
    )
    print(
        "  Mf1 column dominates? |J_Mf1| vs |J_E|: "
        f"W1={np.linalg.norm(w1['jacobian'][:,0]):.3e} vs "
        f"{np.linalg.norm(w1['jacobian'][:,1]):.3e}; "
        f"W2={np.linalg.norm(w2['jacobian'][:,0]):.3e} vs "
        f"{np.linalg.norm(w2['jacobian'][:,1]):.3e}"
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
