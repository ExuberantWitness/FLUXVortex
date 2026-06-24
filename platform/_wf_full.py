import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flex_aircraft as F
from newton_pc.adapters.flap import NodalForceSet


def add_aitken(ac, omega0=0.5, omin=0.05, omax=1.0):
    prov = ac.provider; orig_solve = prov.solve; orig_commit = prov.commit
    st = {"x": None, "r_prev": None, "omega": omega0, "first": True}

    def solve(state):
        out = orig_solve(state); g = out.f.copy()
        if st["first"] or st["x"] is None:
            st["x"] = g; st["r_prev"] = None; st["omega"] = omega0; st["first"] = False
            return out
        r = g - st["x"]
        if st["r_prev"] is not None:
            dr = r - st["r_prev"]; denom = float(np.dot(dr, dr)) + 1e-30
            st["omega"] = float(np.clip(-st["omega"] * float(np.dot(st["r_prev"], dr)) / denom, omin, omax))
        x_new = st["x"] + st["omega"] * r; st["x"] = x_new; st["r_prev"] = r
        return NodalForceSet(x_new, payload=out.payload, madd=out.madd)

    def commit(F_new):
        st["first"] = True; st["x"] = None; st["r_prev"] = None; return orig_commit(F_new)
    prov.solve = solve; prov.commit = commit


def run(tag, *, m_fus, m_wing, V0, flap_hz, amp_deg, E0=40e9, rho_s=142.0, N=80):
    ac = F.FlexAircraft(amp_deg=amp_deg, flap_hz=flap_hz, E0=E0, thick=1.2e-3, rho_s=rho_s,
                        m_fus=m_fus, m_wing=m_wing, body_aoa_deg=45.0, V0=V0,
                        trim_aoa_deg=6.0, substeps=16)
    # physical body inertia for a ~1.7 kg bird (was 3e-3/5e-3, ~10x too small)
    mt = ac.m_total
    ac.I_fus = np.diag([0.03 * mt / 1.7, 0.05 * mt / 1.7, 0.05 * mt / 1.7])
    ac.provider.added_mass_operator = True
    ac.pc.iterations = 8; ac.pc.adaptive_tol = 1e-4
    ac.pc.residual_norm = lambda a, b: float(np.linalg.norm(np.asarray(b['verts']) - np.asarray(a['verts'])))
    add_aitken(ac)
    W = mt * 9.81
    zmax = 0.0; lift = []; zs = []
    for i in range(N):
        try:
            o = ac.step_window()
        except Exception as e:
            print(f"  {tag:30s}: CRASH@{i} ({type(e).__name__})", flush=True); return
        zmax = max(zmax, np.abs(np.asarray(ac.entry.state()['verts'])[..., 2]).max())
        lift.append(o['F_lift']); zs.append(o['z'])
        if not np.isfinite(o['z']) or abs(o['z']) > 1e4:
            print(f"  {tag:30s}: DIVERGE@{i} W={W:.1f}N max|z_w|={zmax:.2e}", flush=True); return
    ls = np.std(lift[10:]); lm = np.mean(lift[10:])
    print(f"  {tag:30s}: OK {N}w W={W:.1f}N  max|z_w|={zmax:.3f}  lift={lm:+.1f}±{ls:.1f}N  "
          f"z:{zs[0]:.0f}->{zs[-1]:.0f}m", flush=True)


print("FULL physical config (anchored: 1.7-2.0kg, V~10, wing 42g/4.7%, Aitken+addedmass):", flush=True)
run("1.7kg V10 2.5Hz 25deg", m_fus=1.616, m_wing=0.042, V0=10.0, flap_hz=2.5, amp_deg=25.0)
run("2.0kg V10 2.5Hz 25deg", m_fus=1.916, m_wing=0.042, V0=10.0, flap_hz=2.5, amp_deg=25.0)
run("1.7kg V11 3.0Hz 22deg", m_fus=1.616, m_wing=0.042, V0=11.0, flap_hz=3.0, amp_deg=22.0)
run("1.7kg V10 2.5Hz 25 E30", m_fus=1.616, m_wing=0.042, V0=10.0, flap_hz=2.5, amp_deg=25.0, E0=30e9)
print("DONE", flush=True)
