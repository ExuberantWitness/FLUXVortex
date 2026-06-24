import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flex_aircraft as F
from newton_pc.adapters.flap import NodalForceSet


def add_aitken(ac, omega0=0.5, omin=0.05, omax=1.0):
    """Wrap provider.solve with Aitken Δ² under-relaxation on the interface force vector
    (the cure newton_pc's plain-Picard strong coupling lacks). Reset each window via commit."""
    prov = ac.provider
    orig_solve = prov.solve
    orig_commit = prov.commit
    st = {"x": None, "r_prev": None, "omega": omega0, "first": True}

    def solve(state):
        out = orig_solve(state)
        g = out.f.copy()
        if st["first"] or st["x"] is None:
            st["x"] = g; st["r_prev"] = None; st["omega"] = omega0; st["first"] = False
            return out
        r = g - st["x"]
        if st["r_prev"] is not None:
            dr = r - st["r_prev"]
            denom = float(np.dot(dr, dr)) + 1e-30
            st["omega"] = float(np.clip(-st["omega"] * float(np.dot(st["r_prev"], dr)) / denom,
                                        omin, omax))
        x_new = st["x"] + st["omega"] * r
        st["x"] = x_new; st["r_prev"] = r
        return NodalForceSet(x_new, payload=out.payload, madd=out.madd)

    def commit(F_new):
        st["first"] = True; st["x"] = None; st["r_prev"] = None
        return orig_commit(F_new)

    prov.solve = solve
    prov.commit = commit


def run(tag, *, rho_s, m_wing, E0, aitken, added_mass, iters=8, N=60):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=E0, thick=1.2e-3, rho_s=rho_s,
                        m_wing=m_wing, body_aoa_deg=45.0, V0=6.0, trim_aoa_deg=6.0, substeps=16)
    ac.provider.added_mass_operator = bool(added_mass)
    ac.pc.iterations = iters; ac.pc.adaptive_tol = 1e-4
    ac.pc.residual_norm = lambda a, b: float(
        np.linalg.norm(np.asarray(b['verts']) - np.asarray(a['verts'])))
    if aitken:
        add_aitken(ac)
    zmax = 0.0; lift = []
    for i in range(N):
        try:
            o = ac.step_window()
        except Exception as e:
            print(f"  {tag:36s}: CRASH@{i} ({type(e).__name__})", flush=True); return
        zmax = max(zmax, np.abs(np.asarray(ac.entry.state()['verts'])[..., 2]).max())
        lift.append(o['F_lift'])
        if not np.isfinite(o['z']) or abs(o['z']) > 1e4:
            print(f"  {tag:36s}: DIVERGE@{i} (max|z_w|={zmax:.2e})", flush=True); return
    ls = np.std(lift[5:]); lm = np.mean(lift[5:])
    print(f"  {tag:36s}: OK {N}w  max|z_w|={zmax:.4f}  lift_mean={lm:+.2f}N "
          f"lift_std={ls:.2f}N  z={o['z']:.1f}", flush=True)


print("AITKEN under-relaxation test (physical light wing):", flush=True)
run("Picard strong (no aitken)",  rho_s=135, m_wing=0.04, E0=50e9, aitken=False, added_mass=True)
run("AITKEN relaxed",             rho_s=135, m_wing=0.04, E0=50e9, aitken=True,  added_mass=True)
run("AITKEN relaxed E=30G",       rho_s=135, m_wing=0.04, E0=30e9, aitken=True,  added_mass=True)
run("AITKEN no-addedmass",        rho_s=135, m_wing=0.04, E0=50e9, aitken=True,  added_mass=False)
print("DONE", flush=True)
