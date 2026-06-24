import numpy as np, sys
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import flex_aircraft as F

def run(tag, *, added_mass, substeps, E0=50e9, N=30):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=E0, body_aoa_deg=45.0, V0=6.0,
                        trim_aoa_deg=6.0, substeps=substeps)
    ac.provider.added_mass_operator = bool(added_mass)
    zmax_wing = 0.0; lift_hist = []
    for i in range(N):
        try:
            o = ac.step_window()
        except Exception as e:
            print(f"  {tag:42s}: CRASH@{i} ({type(e).__name__})", flush=True); return
        v = np.asarray(ac.entry.state()["verts"]); zmax_wing = max(zmax_wing, np.abs(v[...,2]).max())
        lift_hist.append(o["F_lift"])
        if not np.isfinite(o["z"]) or abs(o["z"]) > 1e4:
            print(f"  {tag:42s}: DIVERGE@{i} (max|z_wing|={zmax_wing:.2e})", flush=True); return
    lstd = np.std(lift_hist[5:]) if len(lift_hist) > 5 else 0
    print(f"  {tag:42s}: OK {N}w  max|z_wing|={zmax_wing:.4f}  lift_std={lstd:.1f}N  z={o['z']:.1f}", flush=True)

print("baseline vs fixes (divergence shows by ~win 18):", flush=True)
run("baseline (added_mass=OFF, ss=16)",  added_mass=False, substeps=16)
run("FIX added_mass=ON, ss=16",          added_mass=True,  substeps=16)
run("finer dt added_mass=OFF, ss=64",    added_mass=False, substeps=64)
run("FIX added_mass=ON, ss=32",          added_mass=True,  substeps=32)
print("DONE", flush=True)
