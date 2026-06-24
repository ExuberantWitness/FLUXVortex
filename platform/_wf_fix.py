import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flex_aircraft as F


def run(tag, *, strong, added_mass, iters=10, atol=1e-4, substeps=16, E0=50e9, N=40):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=E0, body_aoa_deg=45.0, V0=6.0,
                        trim_aoa_deg=6.0, substeps=substeps)
    ac.provider.added_mass_operator = bool(added_mass)
    if strong:
        ac.pc.iterations = iters
        ac.pc.adaptive_tol = atol
        ac.pc.residual_norm = lambda a, b: float(
            np.linalg.norm(np.asarray(b['verts']) - np.asarray(a['verts'])))
    zmax = 0.0; lift = []
    for i in range(N):
        try:
            o = ac.step_window()
        except Exception as e:
            print(f"  {tag:38s}: CRASH@{i} ({type(e).__name__}: {e})", flush=True); return
        zmax = max(zmax, np.abs(np.asarray(ac.entry.state()['verts'])[..., 2]).max())
        lift.append(o['F_lift'])
        if not np.isfinite(o['z']) or abs(o['z']) > 1e4:
            print(f"  {tag:38s}: DIVERGE@{i} (max|z_wing|={zmax:.2e})", flush=True); return
    ls = np.std(lift[5:]); lm = np.mean(lift[5:])
    print(f"  {tag:38s}: OK {N}w  max|z_wing|={zmax:.4f}  lift_mean={lm:+.1f}N "
          f"lift_std={ls:.1f}N  z={o['z']:.1f}", flush=True)


print("FIX test (strong coupling + implicit added mass):", flush=True)
run("baseline (loose, no addedmass)",  strong=False, added_mass=False)
run("FIX strong+addedmass iters=10",   strong=True,  added_mass=True, iters=10)
run("FIX strong+addedmass iters=6",    strong=True,  added_mass=True, iters=6)
print("DONE", flush=True)
