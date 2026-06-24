import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')
import numpy as np
import flex_aircraft as F

def make():
    return F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0, V0=6.0, trim_aoa_deg=6.0, substeps=16)

def run(label, ac, N=40):
    lifts=[]; defls=[]; div=None
    for i in range(N):
        try:
            o = ac.step_window()
        except np.linalg.LinAlgError as e:
            div=i; break
        d = np.abs(ac.entry.state()['verts'][...,2]).max()
        lifts.append(o['F_lift']); defls.append(d)
        if not np.isfinite(o['z']) or d>10: div=i; break
    L=np.array(lifts) if lifts else np.array([np.nan])
    print(f"[{label}] N={len(lifts)} div@{div} lift_mean={np.nanmean(L):+.2f} "
          f"std={np.nanstd(L):.2f} max_defl={max(defls) if defls else 0:.4f}m")

# Fix A: implicit added mass only
ac = make(); ac.provider.added_mass_operator = True
run("A:addedmass_only", ac)
