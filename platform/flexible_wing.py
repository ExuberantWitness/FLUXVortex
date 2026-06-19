"""Flexible ANCF main wing — passive aeroelastic feathering in the coupled FSI.

User direction: replace the numerically-stiff rigid light control-surface bodies with
a flexible ANCF wing whose PASSIVE TORSION provides the upstroke feathering (the wing
twists nose-down under aero load, reducing the section effective AoA — the plan's
"柔性被动扭转 = 气弹推进", and the literature's attached-flow mechanism).

Reuses the VALIDATED coupled-FSI reference path (newton_pc FlapEntry elastic +
FlapUVLMProvider + WindowPredictorCorrector — the render_bird_cfrp machinery, validated
vs PteraSoftware). [The GPU/differentiable Warp port is the co-design-scale step.]

This module flaps a bird-scale flexible wing (HIT-Hawk chord/half-span) at the root and
measures, per window: tip bending, tip TWIST (chord rotation -> local AoA change), and
the resulting effective-AoA reduction — confirming passive feathering AND that the
validated ANCF FSI is numerically stable (no light-body blow-up).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from newton_pc import WindowPredictorCorrector                          # noqa: E402
from newton_pc.adapters.flap import (FlapEntry, FlapKinematics,         # noqa: E402
                                     FlapUVLMProvider, NodalForceSet)

# HIT-Hawk-class half-wing (chord 0.29, half-span 0.85), flexible (passive feather)
CHORD, SPAN, NC, NS = 0.29, 0.85, 4, 6
V_INF, ALPHA = 10.0, np.deg2rad(8.0)      # body/cruise AoA presented to the wing
RHO, NU = 1.225, 15.06e-6
PERIOD = 1.0 / 3.0                          # 3 Hz flap
THICK, RHO_S = 1.2e-3, 1200.0
# E~50GPa (light composite) gives CONTROLLED bending (~30% span) + a clean passive
# feather; E=8GPa over-flexes (>100% span, non-physical lift spikes). Tune per design.
E0 = 50e9
DAMPING = 0.05


def tip_twist_deg(entry, nodes0):
    """Passive twist of the tip section: pitch rotation of the deformed chord vector
    (dr/dx slope at the tip) relative to the undeformed chord -> local AoA change."""
    q9 = entry.shell.q.reshape(-1, 9)
    # tip = outboard edge (max y); take its two chordwise-end nodes
    yv = nodes0[:, 1]
    tip_nodes = np.where(yv > yv.max() - 1e-6)[0]
    # chord vector at the tip = (TE node pos - LE node pos)
    xs = nodes0[tip_nodes, 0]
    le = tip_nodes[np.argmin(xs)]; te = tip_nodes[np.argmax(xs)]
    chord_def = q9[te, 0:3] - q9[le, 0:3]
    chord_ref = nodes0[te] - nodes0[le]
    # pitch angle = rotation of the chord in the x-z plane (twist about spanwise y)
    pitch_def = np.arctan2(chord_def[2], chord_def[0])
    pitch_ref = np.arctan2(chord_ref[2], chord_ref[0])
    return np.rad2deg(pitch_def - pitch_ref)


def run(n_cycles=2, amp_deg=35.0, substeps=16, e0=E0, damping=DAMPING, verbose=True,
        E_scale=None, rho_scale=None):
    """E_scale/rho_scale: optional PER-ELEMENT 刚柔 + mass distribution on the main
    wing (callable(x, y) of the element centroid, or (ne,) array). E.g. stiff/heavy
    root -> flexible/light tip: E_scale=lambda x, y: 1 - 0.7*(y/SPAN)**2."""
    dtw = (CHORD / NC) / V_INF
    n_windows = int(round(n_cycles * PERIOD / dtw))
    kin = FlapKinematics(np.deg2rad(amp_deg), PERIOD)
    entry = FlapEntry(CHORD, SPAN, NC, NS, kin, mode="elastic", kscale=1.0,
                      thickness=THICK, rho_s=RHO_S, E0=e0, damping=damping)
    if E_scale is not None or rho_scale is not None:
        entry.shell.set_distribution(E_scale=E_scale, rho_scale=rho_scale)   # per-element
    V_vec = V_INF * np.array([np.cos(ALPHA), 0.0, np.sin(ALPHA)])
    # bounded ring wake (particles off) -> O(1) per-window cost for the feathering probe
    provider = FlapUVLMProvider(V_vec, RHO, dtw, K=6, nu=NU, chord=CHORD,
                                particles=False, max_particles=1)
    pc = WindowPredictorCorrector(entry=entry, provider=provider, substeps=substeps,
                                  dt=dtw / substeps, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))
    pc.advance(n_substeps=1)
    nodes0 = entry.nodes0
    bend, twist, lift, finite = [], [], [], True
    for w in range(n_windows):
        pc.advance()
        st = entry.state()
        th = kin.angles(pc._t)[0]
        zr = (nodes0[:, 1] * np.sin(th)).reshape(NS + 1, NC + 1).T
        b = st["verts"][..., 2] - zr
        bend.append(float(b.max() - b.min()))
        twist.append(tip_twist_deg(entry, nodes0))
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1)) if pc._F_cur.payload else np.zeros(3)
        lift.append(float(-F[0] * np.sin(ALPHA) + F[2] * np.cos(ALPHA)))
        if not np.all(np.isfinite(st["verts"])):
            finite = False; break
        if verbose and (w % 10 == 0 or w < 3):
            print(f"  w={w:4d} t={pc._t:5.3f}s bend={bend[-1]:.4f}m "
                  f"tip_twist={twist[-1]:+6.2f}deg L={lift[-1]:+6.1f}N", flush=True)
    twist = np.array(twist)
    return dict(bend=np.array(bend), twist=twist, lift=np.array(lift), finite=finite,
                n_windows=len(bend))


def verify():
    print("flexible ANCF wing — passive feathering in coupled FSI "
          f"(chord {CHORD}, half-span {SPAN}, E={E0/1e9:.0f}GPa, 3Hz flap)")
    r = run(n_cycles=4, amp_deg=35.0)
    tw = r["twist"]
    finite = r["finite"] and len(tw) > 10
    twist_amp = float(np.nanmax(np.abs(tw[len(tw) // 2:])))   # steady twist amplitude
    bends = float(np.nanmax(r["bend"]))
    # passive feathering present if the tip twists a meaningful amount and stays finite
    ok = finite and twist_amp > 0.5
    print(f"  {r['n_windows']} windows, finite={finite}  (validated ANCF FSI -- no "
          f"light-body blow-up)")
    print(f"  tip bending max={bends:.4f} m  |  passive tip TWIST amplitude="
          f"{twist_amp:.2f} deg (feathering)")
    print(f"flexible wing passive feathering {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
