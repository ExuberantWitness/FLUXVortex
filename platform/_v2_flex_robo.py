"""Phase 2: RoboEagle FLEXIBLE flapping wing FSI — real geometry + measured stiffness + physical mass.

Foundation: build an orthotropic ANCF wing on the REAL planform (_v2_robogeom.robowing_real), then
calibrate the chordwise modulus Ex to the paper's MEASURED chordwise EI via a simulated cantilever
(clamp LE/spar, point-load the TE at the measurement station, match the measured k = F/delta), so the
stiffness is NOT guessed. The spanwise modulus Ey (carbon spar, 1-2 orders stiffer) is set separately.
Mass uses the physical areal density (~150 g/m^2 -> rho_eff ~135 kg/m^3), not solid 1200.

Measured (eiDATA, L = local chord 0.2735m at the measurement point, EI=k L^3/3):
  11a (original, flexible): k=40.9 N/m -> EI=0.279 N*m^2 ;  11b (optimized/stiff): k=166.9 -> EI=1.138.
"""
import os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from fluxvortex.ancf_shell import ANCFShell
import _v2_robogeom as rg

THICK = 1.2e-3
RHO_S = 135.0                       # physical areal density 150 g/m^2 / 1.2mm
K_MEAS = {"11a": 40.9, "11b": 166.9}   # measured chordwise cantilever stiffness (N/m)
MEAS_SPAN = 0.5886                  # measurement point spanwise (m)


def build_robo_shell(nc, ns, Ex, Ey, h=THICK, rho_s=RHO_S, nu=0.3, damping=0.0):
    """Orthotropic ANCF shell on the REAL RoboEagle planform (raked TE + NACA2406 camber).
    Ex = chordwise modulus (soft, membrane), Ey = spanwise modulus (stiff, carbon spar)."""
    C0 = rg.robowing_real(nc, ns)                       # (nc+1, ns+1, 3) [x_chord, y_span, z_camber]
    nn = (nc + 1) * (ns + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ns + 1):
        for i in range(nc + 1):
            nodes[j * (nc + 1) + i] = C0[i, j]
    quads = np.zeros((nc * ns, 4), dtype=np.int32)
    for j in range(ns):
        for i in range(nc):
            n1 = j * (nc + 1) + i
            quads[j * nc + i] = (n1, n1 + 1, n1 + nc + 2, n1 + nc + 1)
    return ANCFShell(nodes, quads, h=h, rho=rho_s, Ex=Ex, Ey=Ey, nu_xy=nu,
                     structural_damping=damping), nodes


def cantilever_k(shell, nodes, nc, ns, F=0.5):
    """Simulated chordwise cantilever stiffness: clamp the LE edge (spar, i=0), apply a vertical (z)
    point load F at the TE node nearest the measurement spanwise station, LINEAR solve -> k = F/delta."""
    le_nodes = [j * (nc + 1) + 0 for j in range(ns + 1)]   # LE edge (chordwise i=0) = spar
    # TE node (i=nc) nearest the measurement spanwise station
    te_js = [(abs(nodes[j * (nc + 1) + nc, 1] - MEAS_SPAN), j) for j in range(ns + 1)]
    j_load = min(te_js)[1]
    load_node = j_load * (nc + 1) + nc
    load_dof = 9 * load_node + 2                            # z-DOF of the TE load node
    shell.set_bc(le_nodes, fix_slopes=True)
    _, Kt = shell._internal_forces_and_tangent(shell.q)
    Kt = np.asarray(Kt.todense()) if hasattr(Kt, "todense") else np.asarray(Kt)
    free = np.array([d for d in range(shell.ndof) if d not in shell._bc_dofs])
    Fv = np.zeros(shell.ndof); Fv[load_dof] = F
    d = np.zeros(shell.ndof)
    Ktff = Kt[np.ix_(free, free)]
    d[free] = np.linalg.solve(Ktff, Fv[free])
    delta = abs(d[load_dof])
    return F / delta if delta > 0 else float('inf'), load_node


def calibrate_Ex(nc, ns, Ey, wing="11b", Ex0=4e6, n_it=6):
    """Iterate Ex so the simulated cantilever k matches the MEASURED k (k(Ex) is monotone but not
    exactly linear because the spanwise Ey path couples in -> fixed-point iterate)."""
    Ex = Ex0; k = ln = None
    for _ in range(n_it):
        sh, nodes = build_robo_shell(nc, ns, Ex, Ey)
        k, ln = cantilever_k(sh, nodes, nc, ns)
        Ex = Ex * (K_MEAS[wing] / k) ** 0.8                # damped fixed point
    return Ex, k, ln


from newton_pc import WindowPredictorCorrector                          # noqa: E402
from newton_pc.adapters.flap import (FlapEntry, FlapKinematics,         # noqa: E402
                                     FlapUVLMProvider, NodalForceSet)


class FlapEntryRobo(FlapEntry):
    """FlapEntry on the REAL RoboEagle geometry + orthotropic calibrated stiffness (root y=0 edge
    prescribed-flap about the x-axis; reuses FlapEntry.substep/state/snapshot/restore)."""

    def __init__(self, nc, ns, kin, Ex, Ey, thickness=THICK, rho_s=RHO_S, damping=0.05):
        self.kin = kin; self.mode = "elastic"; self.nc, self.ns = nc, ns
        self.extra_force_fn = None
        self.shell, self.nodes0 = build_robo_shell(nc, ns, Ex, Ey, h=thickness, rho_s=rho_s, damping=damping)
        self.t = 0.0
        root = [n for n in range(self.shell.nn) if abs(self.nodes0[n, 1]) < 1e-9]   # y=0 root edge
        pd = np.array([9 * n + d for n in sorted(root) for d in range(9)])
        q0 = self.shell.q[pd].reshape(-1, 3, 3)

        def cb(t):
            th, thd, thdd = kin.angles(t)
            R, Rp = kin.rot(th), kin.rot_p(th)
            Rpp = -kin.rot(th) @ np.diag([0.0, 1.0, 1.0])
            dR = thd * Rp; ddR = thdd * Rp + thd ** 2 * Rpp
            return ((q0 @ R.T).reshape(-1), (q0 @ dR.T).reshape(-1), (q0 @ ddR.T).reshape(-1))
        if abs(kin.A) < 1e-15:
            self.shell.set_bc(root, fix_slopes=True)
        else:
            self.shell.set_prescribed_motion(root, cb)


def run_fsi(wing="11b", freq=2.3, amp_deg=45.0, aoa_deg=5.0, Ey=10e9, nc=6, ns=12,
            n_cycles=4, substeps=10, damping=0.05, verbose=False, E_override=None):
    """Flexible RoboEagle FSI: real geometry + measured-EI-calibrated Ex + physical mass, flap +-amp.
    E_override: skip the (fragile) point-load calibration, use this E for both Ex,Ey (diagnostic)."""
    if E_override is not None:
        Ex, Ey, kfit = E_override, E_override, 0.0
    else:
        Ex, kfit, _ = calibrate_Ex(nc, ns, Ey, wing)
    alpha = np.deg2rad(aoa_deg); period = 1.0 / freq
    CHORD = 0.287
    dtw = (CHORD / nc) / 8.0
    wpc = int(round(period / dtw)); n_windows = int(round(n_cycles * period / dtw))
    kin = FlapKinematics(np.deg2rad(amp_deg), period)
    entry = FlapEntryRobo(nc, ns, kin, Ex, Ey, damping=damping)
    V_vec = 8.0 * np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    provider = FlapUVLMProvider(V_vec, 1.225, dtw, K=6, nu=15.06e-6, chord=CHORD,
                                particles=False, max_particles=1)
    pc = WindowPredictorCorrector(entry=entry, provider=provider, substeps=substeps,
                                  dt=dtw / substeps, mode="two-pass")
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))
    pc.advance(n_substeps=1)
    lift = []; bend = []; finite = True; nodes0 = entry.nodes0
    for w in range(n_windows):
        pc.advance()
        st = entry.state()
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1)) if pc._F_cur.payload else np.zeros(3)
        lift.append(float(-F[0] * np.sin(alpha) + F[2] * np.cos(alpha)))
        th = kin.angles(pc._t)[0]
        zr = (nodes0[:, 1] * np.sin(th)).reshape(ns + 1, nc + 1).T
        b = st["verts"][..., 2] - zr
        bend.append(float(b.max() - b.min()))
        if not np.all(np.isfinite(st["verts"])):
            finite = False; break
        if verbose and w % 15 == 0:
            print(f"    w={w} L={lift[-1]:+.1f}N bend={bend[-1]:.3f}m", flush=True)
    lift = np.array(lift)
    Lcyc = 2.0 * float(np.mean(lift[-wpc:])) if len(lift) >= wpc else float('nan')
    return dict(L=Lcyc, Ex=Ex, kfit=kfit, finite=finite, n=len(lift),
                bend_max=float(np.max(bend)) if bend else 0.0)


if __name__ == "__main__":
    import warp as wp; wp.init()
    print("RoboEagle FLEXIBLE FSI (real geom + measured-EI stiffness + physical mass 135):", flush=True)
    print("  rigid 3D UVLM ~4.2N; measured flapping lift ~7.79N (8m/s, 5deg, +-45deg, 2.3Hz)\n", flush=True)
    for wing in ("11b", "11a"):                          # 11b stiff (aero-exp wing), 11a flexible
        for Ey in (10e9, 50e9):
            r = run_fsi(wing=wing, Ey=Ey, n_cycles=4)
            print(f"  {wing} (Ey={Ey/1e9:.0f}GPa, Ex={r['Ex']/1e6:.1f}MPa, k={r['kfit']:.0f}): "
                  f"L={r['L']:+6.2f}N  bend_max={r['bend_max']:.3f}m  finite={r['finite']} ({r['n']}w)", flush=True)
    print("DONE", flush=True)
