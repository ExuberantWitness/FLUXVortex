import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flex_aircraft as F


def run(tag, *, rho_s, m_wing, E0, thick=1.2e-3, strong=False, added_mass=False,
        substeps=16, N=50):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=E0, thick=thick, rho_s=rho_s,
                        m_wing=m_wing, body_aoa_deg=45.0, V0=6.0, trim_aoa_deg=6.0,
                        substeps=substeps)
    ac.provider.added_mass_operator = bool(added_mass)
    if strong:
        ac.pc.iterations = 8; ac.pc.adaptive_tol = 1e-4
        ac.pc.residual_norm = lambda a, b: float(
            np.linalg.norm(np.asarray(b['verts']) - np.asarray(a['verts'])))
    # structural wing mass implied by rho_s*thick*area
    area = ac.chord * ac.span
    m_struct = rho_s * thick * area
    W = ac.m_total * 9.81
    zmax = 0.0; lift = []
    for i in range(N):
        try:
            o = ac.step_window()
        except Exception as e:
            print(f"  {tag:34s}: CRASH@{i} ({type(e).__name__})", flush=True); return
        zmax = max(zmax, np.abs(np.asarray(ac.entry.state()['verts'])[..., 2]).max())
        lift.append(o['F_lift'])
        if not np.isfinite(o['z']) or abs(o['z']) > 1e4:
            print(f"  {tag:34s}: DIVERGE@{i} (m_struct={m_struct*1e3:.0f}g W={W:.1f}N "
                  f"max|z_w|={zmax:.2e})", flush=True); return
    ls = np.std(lift[5:]); lm = np.mean(lift[5:])
    print(f"  {tag:34s}: OK {N}w  m_struct={m_struct*1e3:.0f}g W={W:.1f}N  "
          f"max|z_w|={zmax:.4f}  lift_mean={lm:+.2f}N lift_std={ls:.2f}N  z={o['z']:.1f}",
          flush=True)


print("PHYSICAL params (light wing rho~135, m_wing~40g, resonance E):", flush=True)
# old unphysical baseline for reference
run("OLD rho=1200 m=80g E=50G (loose)", rho_s=1200, m_wing=0.08, E0=50e9)
# physical light wing, loose coupling (does physical mass alone fix it?)
run("PHYS rho=135 m=40g E=50G loose",   rho_s=135,  m_wing=0.04, E0=50e9)
run("PHYS rho=135 m=40g E=30G loose",   rho_s=135,  m_wing=0.04, E0=30e9)
# physical + strong coupling
run("PHYS rho=135 m=40g E=50G strong",  rho_s=135,  m_wing=0.04, E0=50e9, strong=True, added_mass=True)
print("DONE", flush=True)
