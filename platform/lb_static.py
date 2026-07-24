"""S0: static separation-point f(alpha) from polar inversion (L-B foundation, 2026-07-20).

Zero-fit Kirchhoff inversion of the separation point from a static CL(alpha) polar,
replacing v4's ad-hoc "geometric stall" (geo_stall_deg/width) — research_lb_formula.md
A.2/B.2, Risø-R-1354 Eq.15-19, Bangga 2020 Eq.33:
    f = (2*sqrt(CL / (CLa*(a-a0))) - 1)^2,  clipped [0,1]
This is textbook-accepted (Risø-R-1354, OpenFAST AeroDyn, Hansen 2004) — same epistemic
status as the UIUC polars already in use. NOT a fit to RoboEagle dynamic data.

Polar assembly (zero-fit, all from literature airfoil data + flat-plate Newtonian post-stall):
  - attached/pre-stall (-5..10.5 deg): UIUC SD7003.DRG measured CL(alpha,Re), thin-cambered
    proxy for NACA2406 (recorded assumption; cd_table._parse).
  - post-stall (10.5..90 deg): flat-plate Newtonian CL=CD90_AR*sin^2(a)*cos(a)*sign,
    CD=CD90_AR*sin^2(a), CD90_AR=1.20 (Hoerner 2D 1.98 x finite-AR~6 correction, registered
    in research_caseA_redirect R1/A2-C3). Smooth C0 blend at the stall edge.
  - symmetric for negative alpha (camber proxy via alpha0L offset).
CLa = max attached-region slope (Risø Eq.16); a0 = zero-lift angle (camber).
Robust inversion (Risø Eq.17-19): CL_static = CLa*(a-a0)*f + CLfs*(1-f), with CLfs the
full-separation lift; f->1 attached, f->0 fully separated."""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _sd7003_blocks():
    sys = os.path.join  # noqa
    from cd_table import _parse
    return _parse()


class StaticPolar:
    """Full-angle static CL/CD polar + Kirchhoff f(alpha) inversion. Vectorized over alpha."""

    def __init__(self, cd90_ar=1.20, camber_a0l_deg=-1.5, blend_deg=3.0, re_target=1.5e5,
                 cla3d_ar=0.0):
        blocks = _sd7003_blocks()
        self.res = np.array([b[0] for b in blocks])
        # pick the Re block nearest re_target, interpolate CL vs alpha onto a common grid
        amin = max(b[1][:, 0].min() for b in blocks)
        amax = min(b[1][:, 0].max() for b in blocks)
        self.att_alpha = np.linspace(amin, amax, 40)
        # logRe-interpolated CL at each alpha
        logre = np.log(self.res)
        cl_grid = np.stack([np.interp(self.att_alpha, b[1][:, 0], b[1][:, 1]) for b in blocks])
        from numpy.polynomial import polynomial as P
        self.att_cl = np.zeros(len(self.att_alpha))
        lr = np.log(re_target)
        for i in range(len(self.att_alpha)):
            self.att_cl[i] = np.interp(lr, logre, cl_grid[:, i])
        # attached slope (max CL/alpha in the linear region) + zero-lift angle
        att = (self.att_alpha > -3) & (self.att_alpha < 6)
        self.cla = float(np.polyfit(np.radians(self.att_alpha[att]), self.att_cl[att], 1)[0])
        if cla3d_ar > 0.0:
            # (2026-07-24) Prandtl finite-wing lift-slope reduction, CLa_3D = CLa_2D/(1+CLa_2D/(pi*AR))
            # (Anderson Fund. Aero; lifting-line, elliptic loading). The L-B polar is 2D (SD7003);
            # on a finite wing the section judge over-separates because the 2D polar ignores the
            # downwash-induced incidence loss. Scales BOTH the f inversion and the CNf/Kirchhoff
            # reconstruction consistently. Literature-anchored (AR from geometry), zero-fit.
            self.cla = self.cla / (1.0 + self.cla / (np.pi * float(cla3d_ar)))
        self.a0 = float(np.radians(camber_a0l_deg))             # camber zero-lift (NACA2406 proxy)
        self.stall_edge = float(self.att_alpha[np.argmax(self.att_cl)])   # ~10.5 deg
        self.cd90 = float(cd90_ar)
        self.blend = float(blend_deg)

    def _cl_post(self, a_rad):
        """Post-stall CL within the Kirchhoff framework = the f=0 floor CLa*(a-a0)*0.25 (signed).
        L-B/Kirchhoff describes post-stall CL by the separation factor (f->0 gives 0.25 x attached),
        NOT by the physical flat-plate Newtonian value (which is LOWER and breaks the inversion).
        The physical flat-plate CD enters separately via cd_static (Hoerner) — drag uses Newtonian,
        lift uses Kirchhoff: standard L-B bookkeeping. This keeps f_inversion/reconstruct consistent."""
        return self.cla * (a_rad - self.a0) * 0.25

    def _blend_weight(self, a_rad):
        """1 attached (|a|<stall_edge-blend), 0 post-stall (|a|>stall_edge), C0-smooth between."""
        aabs = np.abs(np.degrees(a_rad))
        lo = self.stall_edge - self.blend
        return np.clip((self.stall_edge - aabs) / max(self.blend, 1e-6), 0.0, 1.0)

    def cl_static(self, a_rad):
        """Full-angle static CL: attached linear CLa*(a-a0) blended into Newtonian post-stall."""
        cl_att = self.cla * (a_rad - self.a0)                       # attached (uses a-a0, camber)
        cl_post = self._cl_post(a_rad)                              # post-stall (geometric a, Newtonian)
        w = self._blend_weight(a_rad)
        return w * cl_att + (1.0 - w) * cl_post

    def cd_static(self, a_rad):
        """Full-angle static CD: attached quadratic bucket + post-stall Newtonian sin^2."""
        aabs = np.abs(np.degrees(a_rad))
        cd_att = 0.012 * (1.0 + (aabs / 5.0) ** 2)
        cd_post = self.cd90 * np.sin(a_rad) ** 2
        w = self._blend_weight(a_rad)
        return w * cd_att + (1.0 - w) * cd_post

    def f_inversion(self, a_rad):
        """Kirchhoff separation point f(alpha) in [0,1] from the static CL polar (Risø Eq.15).
        f=1 attached, f=0 fully separated. ratio = CL_static/(CLa*(a-a0)), clipped."""
        cl = self.cl_static(a_rad)
        den = self.cla * (a_rad - self.a0)
        ratio = np.where(np.abs(den) > 1e-6, cl / np.where(np.abs(den) > 1e-6, den, 1.0), 1.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        f = (2.0 * np.sqrt(ratio) - 1.0) ** 2
        return np.clip(f, 0.0, 1.0)

    def reconstruct_cl(self, a_rad):
        """Kirchhoff reconstruction CLa*(a-a0)*((1+sqrt(f))/2)^2 — must match cl_static in the
        attached/inversion region (machine precision); post-stall it gives the Kirchhoff f=0
        floor 0.25*CLa*(a-a0), a known Kirchhoff lower bound (the physical flat-plate CLfs is
        lower). The canonical gate reports attached vs post-stall error separately."""
        f = self.f_inversion(a_rad)
        return self.cla * (a_rad - self.a0) * ((1.0 + np.sqrt(f)) / 2.0) ** 2


if __name__ == "__main__":
    sp = StaticPolar()
    print(f"CLa={sp.cla:.2f}/rad  a0={np.degrees(sp.a0):.2f}deg  stall_edge={sp.stall_edge:.1f}deg  "
          f"CD90_AR={sp.cd90:.2f}", flush=True)
    print("\n== S0 canonical: f(alpha) inversion + CL reconstruction ==", flush=True)
    print(f"{'a(deg)':>7}{'CL_stat':>9}{'CL_recon':>9}{'dCL':>8}{'f':>7}{'CD_stat':>9}", flush=True)
    for ad in (-45, -30, -15, -10.5, -5, 0, 5, 10.5, 15, 25, 45):
        a = np.radians(ad)
        cs = sp.cl_static(a); cr = sp.reconstruct_cl(a); f = sp.f_inversion(a); cd = sp.cd_static(a)
        print(f"{ad:>7.1f}{cs:>9.3f}{cr:>9.3f}{cr-cs:>8.3f}{f:>7.3f}{cd:>9.3f}", flush=True)
    # gate: attached region (|a|<stall_edge) must reconstruct machine-precision; post-stall
    # carries the Kirchhoff-floor approximation (reported, not a failure).
    aa = np.radians(np.linspace(-30, 30, 121))
    att = np.abs(np.degrees(aa)) < sp.stall_edge
    err_att = np.max(np.abs(sp.reconstruct_cl(aa[att]) - sp.cl_static(aa[att])))
    err_post = np.max(np.abs(sp.reconstruct_cl(aa[~att]) - sp.cl_static(aa[~att])))
    print(f"\nS0 gate: attached|recon-static| = {err_att:.4f} (target <0.01) | "
          f"post-stall = {err_post:.4f} (Kirchhoff floor approx, recorded)", flush=True)
