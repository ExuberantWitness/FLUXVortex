"""Spanwise 刚柔 (stiffness) DESIGN FIELD — co-design's structural design variable.

Upgrades co-design's design space from a single scalar wing-stiffness `s` to a
PER-ELEMENT / spline stiffness field s(ξ) over the span (ξ=0 root → ξ=1 tip), the
plan's §5 "刚柔(+质量)分布低维压缩(样条控制点)". A few control points define a smooth
spanwise field; the field maps BOTH to:

  (a) the per-element E-scale on the real ANCF wing  -> ANCFShell.set_distribution
  (b) reduced physical AGGREGATES that drive the fast meta-RL flight dynamics:

        s_gust  = load-weighted (tip-biased) effective stiffness  -> gust response
        s_root  = bending-moment-weighted (root-biased) stiffness -> cruise efficiency
        C_feath = load-weighted tip COMPLIANCE (passive washout)  -> over-flex penalty

The physics of the reduction (standard flapping-wing aeroelasticity, anchored to the
validated ANCF FSI in calibrate.py):
  - Unsteady load concentrates OUTBOARD (plunge velocity ḣ(ξ)=ξ·a·ω grows with span,
    q_dyn∝U²+ḣ²) -> a flexible TIP sheds gust load by passive washout (nose-down twist).
    => gust alleviation & control authority are governed by the TIP-biased compliance.
  - Bending moment is largest at the ROOT (M(ξ)=∫_ξ^1 load) -> holding the designed
    incidence/planform (hence cruise L/D) is governed by the ROOT-biased stiffness.
  => a stiff-root + flexible-tip FIELD can get BOTH high efficiency AND gust rejection,
     which a single uniform scalar cannot — this is the distributional payoff co-design
     can now discover (a Pareto front that dominates the uniform-stiffness front).

Backward-compatible: a UNIFORM field s reduces EXACTLY to the old scalar surrogate
(s_gust=s_root=s, C_feath=1/s, no penalty for s∈[0.5,2]) — the scalar co-design is the
diagonal slice of the field design space.
"""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy>=2 renamed trapz

# span-load model: chord taper used for the aerodynamic load weight w(ξ)=ξ·c(ξ)
_TAPER = 0.5            # tip/root chord (matches WingDesign.taper)
_NG = 96               # spanwise quadrature resolution for the aggregates
_C0 = 2.0             # over-flex compliance threshold (= 1/s at s=0.5 -> uniform exact)
_PEN = 3.0           # L/D penalty slope beyond C0


def _chord(xi):
    return 1.0 - (1.0 - _TAPER) * xi


class StiffnessField:
    """Spanwise stiffness-scale field s(ξ), ξ∈[0,1] (0=root,1=tip), from K control pts.

    Linear interpolation between control points (smooth, monotone-free, differentiable
    surrogate; no scipy). `ctrl` are the per-control-point stiffness scales root->tip.
    """

    def __init__(self, ctrl, xi=None):
        self.ctrl = np.asarray(ctrl, float)
        K = len(self.ctrl)
        self.xi_ctrl = np.linspace(0.0, 1.0, K) if xi is None else np.asarray(xi, float)
        # precompute the spanwise quadrature
        self._xg = np.linspace(0.0, 1.0, _NG)
        self._sg = self.value(self._xg)
        self._wg = self._xg * _chord(self._xg)                      # aero load weight w(ξ)
        # bending-moment arm m(ξ)=∫_ξ^1 w dξ' (CLOSED FORM, matches the Warp kernel):
        #   W(u)=∫_0^u xi(1-(1-taper)xi) dxi = u²/2 - (1-taper)u³/3 ;  m(ξ)=W(1)-W(ξ)
        W1 = 0.5 - (1.0 - _TAPER) / 3.0
        self._mg = W1 - (0.5 * self._xg ** 2 - (1.0 - _TAPER) * self._xg ** 3 / 3.0)

    # ── representation ────────────────────────────────────────────────────────
    @classmethod
    def uniform(cls, s, K=4):
        return cls(np.full(K, float(s)))

    @classmethod
    def from_root_tip(cls, root, tip, K=4):
        """Linear root->tip field (the 2-D structured design slice)."""
        return cls(np.linspace(float(root), float(tip), K))

    @classmethod
    def sample(cls, rng, K=4, lo=0.3, hi=2.5):
        """Random smooth field: log-uniform control points (physical spread of EI)."""
        c = np.exp(rng.uniform(np.log(lo), np.log(hi), size=K))
        return cls(c)

    def value(self, xi):
        """Interpolated stiffness scale at span fraction(s) ξ."""
        return np.interp(np.clip(xi, 0.0, 1.0), self.xi_ctrl, self.ctrl)

    # ── (a) maps onto the real ANCF wing (per-element) ─────────────────────────
    def e_scale_fn(self, span, root_y=0.0):
        """callable(x, y) for ANCFShell.set_distribution — per-element E-scale field."""
        denom = (span - root_y + 1e-9)
        return lambda x, y: float(self.value((abs(y) - root_y) / denom))

    def per_element(self, centers_y, span, root_y=0.0):
        """(ne,) per-element E-scale from element centroid y-coordinates."""
        xi = (np.abs(np.asarray(centers_y, float)) - root_y) / (span - root_y + 1e-9)
        return self.value(xi)

    # ── (b) reduced physical aggregates that drive the fast flight dynamics ─────
    def feather_compliance(self):
        """Load-weighted (tip-biased) compliance C=∫w/s / ∫w -> passive washout."""
        return float(_trapz(self._wg / self._sg, self._xg) / _trapz(self._wg, self._xg))

    def s_gust(self):
        """Effective gust stiffness = 1/C_feather (tip-biased; uniform -> s)."""
        return 1.0 / self.feather_compliance()

    def s_root(self):
        """Bending-moment-weighted (root-biased) stiffness (uniform -> s)."""
        return float(_trapz(self._mg * self._sg, self._xg) / _trapz(self._mg, self._xg))

    def aggregates(self):
        C = self.feather_compliance()
        return dict(s_gust=1.0 / C, s_root=self.s_root(), C_feather=C,
                    s_mean=float(self._sg.mean()))


def gust_factor(field):
    """Gust transmissibility (plan F1: passive gust ALLEVIATION). A flexible tip (low
    s_gust) washes out under a gust, shedding load -> LESS effective gust. Monotone
    increasing & bounded in (0.5,1): flexible-tip attenuates, stiff transmits."""
    sg = field.s_gust()
    return 0.5 + 0.5 * sg / (sg + 1.0)


def ctrl_factor(field):
    """Design-dependent control authority (uniform field s -> old 1.6-0.5(s-0.5))."""
    return 1.6 - 0.5 * (field.s_gust() - 0.5)


def cruise_efficiency(field):
    """Cruise L/D: root-stiffness driven (shape retention) minus over-flex penalty.

    Uniform field s∈[0.5,2]: s_root=s, C=1/s≤2 -> penalty 0 -> EXACTLY the old
    22.0+2.2*(s-0.5). A stiff-root/flexible-tip field keeps high s_root (efficient)
    while its tip compliance stays below C0 -> efficiency AND gust rejection."""
    if np.isscalar(field):
        field = StiffnessField.uniform(float(field))
    a = field.aggregates()
    pen = _PEN * max(0.0, a["C_feather"] - _C0)
    return 22.0 + 2.2 * (a["s_root"] - 0.5) - pen


def as_field(design, K=4):
    """Coerce a design (scalar / (K,) ctrl array / StiffnessField) to a StiffnessField."""
    if isinstance(design, StiffnessField):
        return design
    arr = np.asarray(design, float)
    if arr.ndim == 0:
        return StiffnessField.uniform(float(arr), K=K)
    return StiffnessField(arr)


def _selfcheck():
    """Verify the field surrogate reduces EXACTLY to the old scalar for uniform fields,
    and that stiff-root/flexible-tip decouples efficiency from gust rejection."""
    print("design_field self-check: uniform-field == old scalar surrogate")
    ok = True
    for s in [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]:
        f = StiffnessField.uniform(s)
        a = f.aggregates()
        gf, cf, ld = gust_factor(f), ctrl_factor(f), cruise_efficiency(f)
        gf0 = 1.0 / (0.6 + 0.4 * s); cf0 = 1.6 - 0.5 * (s - 0.5); ld0 = 22.0 + 2.2 * (s - 0.5)
        d = max(abs(gf - gf0), abs(cf - cf0), abs(ld - ld0),
                abs(a["s_gust"] - s), abs(a["s_root"] - s))
        ok &= d < 1e-6
        print(f"  s={s:.2f}: s_gust={a['s_gust']:.4f} s_root={a['s_root']:.4f} "
              f"L/D={ld:.3f} (old {ld0:.3f})  max|Δ|={d:.2e}")
    print(f"  uniform-recovery {'PASS' if ok else 'FAIL'} (field generalizes the scalar)\n")

    print("distributional payoff: equal mean stiffness, different distribution")
    base = StiffnessField.uniform(1.2)
    sr_ft = StiffnessField.from_root_tip(2.0, 0.4)   # stiff root, flexible tip
    fr_st = StiffnessField.from_root_tip(0.4, 2.0)   # flexible root, stiff tip
    for name, f in [("uniform   s=1.2", base), ("stiff-root/flex-tip", sr_ft),
                    ("flex-root/stiff-tip", fr_st)]:
        a = f.aggregates()
        print(f"  {name:22s} s_mean={a['s_mean']:.2f} | s_gust={a['s_gust']:.2f} "
              f"s_root={a['s_root']:.2f} -> gust×eff: gf={gust_factor(f):.3f} "
              f"L/D={cruise_efficiency(f):.2f}")
    a_sr = sr_ft.aggregates()
    payoff = (a_sr["s_root"] > sr_ft.aggregates()["s_gust"])    # root stiff, gust soft
    print(f"  stiff-root/flex-tip decouples (s_root>{a_sr['s_gust']:.2f}=s_gust): "
          f"{'YES — efficient AND gust-tolerant' if payoff else 'no'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selfcheck() else 1)
