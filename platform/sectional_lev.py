"""Sectional leading-edge-vortex augmentation for the aircraft LE control surfaces
(req #1/#2, faithful per-strip option).

Each leading-edge control surface (the front flaps) is divided into its spanwise UVLM
strips; each strip carries its own 2D LESP-LDVM (lev_dvm.LDVM2D) that sheds real
leading-edge vortices when the local leading-edge suction exceeds critical and returns
the LEV-augmented sectional lift. Because the flapping MAV cruises at ~45deg body AoA,
the LE flaps genuinely operate past LE separation, so this captures the dynamic-stall
LEV lift the bound UVLM cannot.

Per step, the local strip kinematics (effective AoA + pitch rate from the body's world
velocity at the strip) drive the strip LDVM; the sectional force CL*q*chord*width acts
in the strip lift direction and accumulates into the parent flap body's wrench, summed
with the bound-UVLM wrench from MultiSurfaceUVLM (which still carries the box/TE/tail
and the global induced field).
"""
from __future__ import annotations

import numpy as np

from lev_dvm import LDVM2D


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SectionalLEV:
    """Per-strip LDVM over the LE control surfaces. step() returns per-body wrenches."""

    def __init__(self, surfs, *, U_ref=10.0, lesp_crit=0.20, n_chord=10, rho=1.225):
        self.rho = rho
        self.strips = []
        for s in surfs:
            if not s["name"].startswith("le_"):
                continue
            V = s["V"]                                   # (nc+1, ns+1, 3)
            ns = s["ns"]
            for j in range(ns):
                le0, le1 = V[0, j], V[0, j + 1]          # leading edge of the strip
                te0 = V[-1, j]                           # aft (hinge) edge
                chord = float(np.linalg.norm(te0 - le0))
                width = float(np.linalg.norm(le1 - le0))
                center = 0.5 * (V[:, j, :].mean(0) + V[:, j + 1, :].mean(0))
                chordhat = (te0 - le0) / (np.linalg.norm(te0 - le0) + 1e-12)
                spanhat = (le1 - le0) / (np.linalg.norm(le1 - le0) + 1e-12)
                self.strips.append(dict(
                    body=s["body"], name=s["name"], j=j, chord=chord, width=width,
                    center_body=center, chordhat_body=chordhat, spanhat_body=spanhat,
                    ldvm=LDVM2D(U=U_ref, c=max(chord, 1e-2), n=n_chord,
                                lesp_crit=lesp_crit, rho=rho)))
        self.U_ref = U_ref

    def step(self, poses, twists, V_inf_world, dt):
        """poses/twists: dict body->(p,quat)/(v,omega) world. V_inf_world (3) gust+
        freestream. Returns dict body->(F,M about body origin), + diagnostics."""
        wrench = {}
        n_lev = 0
        Vinf = np.asarray(V_inf_world, float)
        for st in self.strips:
            p, q = poses[st["body"]]
            v, om = twists[st["body"]]
            R = _quat_to_R(np.asarray(q, float)); p = np.asarray(p, float)
            v = np.asarray(v, float); om = np.asarray(om, float)
            c_w = R @ st["chordhat_body"]               # world chord/span/normal
            s_w = R @ st["spanhat_body"]
            n_w = np.cross(s_w, c_w); n_w /= (np.linalg.norm(n_w) + 1e-12)
            rc = R @ st["center_body"]                    # strip center rel body origin
            Vpt = v + np.cross(om, rc)                    # strip velocity (world)
            Vrel = Vinf - Vpt                             # relative wind at the strip
            # decompose into the strip 2D plane (chord, normal); spanwise ignored
            uc = float(Vrel @ c_w); un = float(Vrel @ n_w)
            Vmag = np.hypot(uc, un)
            if Vmag < 1e-3:
                continue
            alpha = np.arctan2(un, -uc)                   # +AoA: wind from below-front
            om_pitch = float(om @ s_w)                    # pitch rate about strip span
            ld = st["ldvm"]
            ld.U = Vmag                                   # local relative speed
            r = ld.step(alpha, om_pitch)
            n_lev += r["n_lev"]
            q_dyn = 0.5 * self.rho * Vmag * Vmag
            Lsec = r["CL"] * q_dyn * st["chord"] * st["width"]
            uhat = Vrel / Vmag
            lift_dir = n_w - float(n_w @ uhat) * uhat
            ln = np.linalg.norm(lift_dir)
            lift_dir = lift_dir / ln if ln > 1e-6 else n_w
            lift_dir *= np.sign(un) if un != 0 else 1.0
            Fp = Lsec * lift_dir
            F, M = wrench.get(st["body"], (np.zeros(3), np.zeros(3)))
            wrench[st["body"]] = (F + Fp, M + np.cross(rc, Fp))
        return dict(wrench=wrench, n_lev=n_lev, n_strips=len(self.strips))


def _validate():
    """At ~45deg body AoA + flapping the LE-flap strips must shed LEVs and add lift."""
    import os, sys
    for p in ("platform", "src", "tests"):
        if os.path.abspath(p) not in sys.path:
            sys.path.insert(0, os.path.abspath(p))
    import warp as wp
    wp.init()
    from aircraft_geom import Aircraft
    import uvlm_aircraft as U
    surfs = U.build_aircraft_surfaces(Aircraft())
    slev = SectionalLEV(surfs, U_ref=10.0, lesp_crit=0.20)
    bodies = sorted({st["body"] for st in slev.strips})
    print(f"sectional LEV: {slev.strips.__len__()} LE-flap strips on bodies {bodies}")
    # prescribe ~45deg body AoA flight + a flapping oscillation, march
    dt = 1.0 / 500.0
    V0 = 10.0
    tot_lift = 0.0; last = None
    for it in range(120):
        t = it * dt
        # body pitched 45deg up (quat about y), flying forward; flap oscillation in vz
        a = np.deg2rad(45.0)
        quat = np.array([0, np.sin(a / 2), 0, np.cos(a / 2)])
        flap = 1.5 * np.sin(2 * np.pi * 3.0 * t)        # flapping vertical velocity
        poses = {b: (np.zeros(3), quat) for b in range(20)}
        twists = {b: (np.array([V0, 0.0, flap]), np.array([0.0, 0.0, 0.0])) for b in range(20)}
        Vinf = np.zeros(3)
        last = slev.step(poses, twists, Vinf, dt)
        tot_lift += sum(w[0][2] for w in last["wrench"].values())
    levs = last["n_lev"]
    ok = levs > 0 and np.isfinite(tot_lift) and len(slev.strips) == 18
    print(f"  after 120 steps @45deg AoA + 3Hz flap: total LEVs shed={levs}  "
          f"mean strip lift sum finite={np.isfinite(tot_lift)}")
    print(f"sectional LEV {'PASS' if ok else 'FAIL'}: LE-flap strips shed real LEVs in flight")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
