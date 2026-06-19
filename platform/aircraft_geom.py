"""Parametric aircraft geometry (plan §1/§5) — the DESIGNABLE planform.

Defines the real flapping-aircraft planform that the multibody + UVLM are built on,
with the design variables the co-design optimizes:

  main wing : area, aspect ratio, taper ratio (根稍比), LE sweep, dihedral, incidence
              + a contiguous leading-edge flap line (3 segments) hinged at the LE and a
              contiguous trailing-edge flap line (3 segments) hinged at the TE — no gaps.
  V-tail    : FIXED stabilizer (designable AR, taper, dihedral 上下反, incidence 安装角)
              + a movable ruddervator (rear flap fraction) on each panel.

14 control surfaces = 6 per wing (3 LE + 3 TE) + 2 ruddervators. Each surface is a real
trapezoidal strip at the wing/tail edge (not a floating panel). Coordinates +X forward,
+Z up. Geometry is returned as polygons (top-view x,y and a chordwise/section helper)
reused by the schematic, the multibody assembly, and the UVLM panel lattices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WingDesign:
    area: float = 0.49           # planform area of ONE wing (m^2)  (both -> ~0.49*? see span)
    aspect_ratio: float = 5.9    # AR = b^2 / S_total
    taper: float = 0.5           # tip chord / root chord (根稍比 = root/tip = 1/taper)
    sweep_le_deg: float = 8.0    # leading-edge sweep
    dihedral_deg: float = 4.0    # 上反角
    incidence_deg: float = 3.0   # 安装角 (root)
    root_offset: float = 0.05    # half fuselage width (wing root y)
    le_flap_frac: float = 0.20   # LE flap = front 20% chord
    te_flap_frac: float = 0.28   # TE flap = rear 28% chord
    n_le: int = 3                # contiguous LE flap segments
    n_te: int = 3                # contiguous TE flap segments


@dataclass
class TailDesign:
    area: float = 0.10           # total V-tail area (both panels)
    aspect_ratio: float = 3.2
    taper: float = 0.6
    dihedral_deg: float = 38.0   # 上下反角 (V angle from horizontal)
    incidence_deg: float = 45.0  # 安装角 — large UP incidence: the flapping MAV cruises
    #                              at ~45deg body AoA, so the tail is mounted ~45deg up
    #                              to meet the flow at an effective angle (designable).
    boom: float = 0.60           # tail arm aft of CG
    ruddervator_frac: float = 0.40   # rear 40% chord is the movable ruddervator


@dataclass
class Aircraft:
    wing: WingDesign = field(default_factory=WingDesign)
    tail: TailDesign = field(default_factory=TailDesign)

    # ── main-wing trapezoid (one side, sgn=+1 left / -1 right) ────────────────
    def wing_dims(self):
        w = self.wing
        S_total = w.area
        b = np.sqrt(w.aspect_ratio * S_total)        # full span
        semi = b / 2.0
        # trapezoid: S_total = b * c_root*(1+taper)/2  -> c_root
        c_root = 2.0 * S_total / (b * (1.0 + w.taper))
        c_tip = w.taper * c_root
        return dict(b=b, semi=semi, c_root=c_root, c_tip=c_tip)

    def _wing_chord(self, y_abs):
        """Local chord & LE-x at absolute spanwise |y| (root_offset..semi)."""
        w = self.wing; d = self.wing_dims()
        frac = (abs(y_abs) - w.root_offset) / (d["semi"] - w.root_offset + 1e-9)
        frac = np.clip(frac, 0.0, 1.0)
        c = d["c_root"] + (d["c_tip"] - d["c_root"]) * frac
        x_le = -np.tan(np.deg2rad(w.sweep_le_deg)) * (abs(y_abs) - w.root_offset) + 0.5 * d["c_root"]
        return c, x_le

    def _wing_strip(self, sgn, y0f, y1f, fa, fb):
        """4 corners (x,y) of a chordwise strip frac∈[fa,fb] over span frac∈[y0f,y1f].
        frac-from-LE: x = x_le(y) - frac*c(y). Spanwise frac over [root_offset, semi]."""
        w = self.wing; d = self.wing_dims()
        ys = w.root_offset + np.array([y0f, y1f]) * (d["semi"] - w.root_offset)
        pts = []
        for yy in ys:
            c, x_le = self._wing_chord(yy)
            pts.append((x_le - fa * c, sgn * yy))
            pts.append((x_le - fb * c, sgn * yy))
        # order CCW: (y0,fa),(y0,fb),(y1,fb),(y1,fa)
        return np.array([pts[0], pts[1], pts[3], pts[2]])

    def wing_components(self, sgn):
        """Return dict: box, le_flaps(list), te_flaps(list) polygons (top view)."""
        w = self.wing
        lef, tef = w.le_flap_frac, w.te_flap_frac
        box = self._wing_strip(sgn, 0.0, 1.0, lef, 1.0 - tef)
        le_flaps, te_flaps = [], []
        for k in range(w.n_le):
            y0, y1 = k / w.n_le, (k + 1) / w.n_le
            le_flaps.append(self._wing_strip(sgn, y0, y1, 0.0, lef))
        for k in range(w.n_te):
            y0, y1 = k / w.n_te, (k + 1) / w.n_te
            te_flaps.append(self._wing_strip(sgn, y0, y1, 1.0 - tef, 1.0))
        return dict(box=box, le_flaps=le_flaps, te_flaps=te_flaps)

    # ── V-tail (fixed stabilizer + ruddervator), one side ─────────────────────
    def tail_dims(self):
        t = self.tail
        S = t.area
        bproj = np.sqrt(t.aspect_ratio * S)          # total tip-to-tip (projected)
        semi = bproj / 2.0
        c_root = 2.0 * S / (bproj * (1.0 + t.taper))
        c_tip = t.taper * c_root
        return dict(semi=semi, c_root=c_root, c_tip=c_tip)

    def tail_components(self, sgn):
        """Fixed stabilizer + movable ruddervator polygons (top-view projection)."""
        t = self.tail; d = self.tail_dims()
        x0 = -t.boom
        rf = t.ruddervator_frac
        ys = np.array([0.04, d["semi"]])             # small root offset on the boom
        stab, rud = [], []
        for frac_a, frac_b, store in ((0.0, 1.0 - rf, stab), (1.0 - rf, 1.0, rud)):
            pts = []
            for yy in ys:
                fr = (yy - 0.04) / (d["semi"] - 0.04 + 1e-9)
                c = d["c_root"] + (d["c_tip"] - d["c_root"]) * fr
                x_le = x0 + 0.5 * d["c_root"]
                pts.append((x_le - frac_a * c, sgn * yy))
                pts.append((x_le - frac_b * c, sgn * yy))
            store.append(np.array([pts[0], pts[1], pts[3], pts[2]]))
        return dict(stab=stab[0], ruddervator=rud[0])

    def summary(self):
        d = self.wing_dims(); td = self.tail_dims()
        return (f"wing: span={d['b']:.2f}m AR={self.wing.aspect_ratio} taper={self.wing.taper} "
                f"c_root={d['c_root']:.3f} c_tip={d['c_tip']:.3f} dih={self.wing.dihedral_deg} "
                f"inc={self.wing.incidence_deg}deg | Vtail: AR={self.tail.aspect_ratio} "
                f"taper={self.tail.taper} dih={self.tail.dihedral_deg} inc={self.tail.incidence_deg} "
                f"rudder={self.tail.ruddervator_frac}")
