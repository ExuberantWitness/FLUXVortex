import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')
import numpy as np
import flex_aircraft as F

def run(label, cfg):
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0, V0=6.0, trim_aoa_deg=6.0,
                        substeps=cfg.get("substeps",16))
    cfg.get("setup", lambda a: None)(ac)
    lifts=[]; defls=[]; diverged_at=None
    for i in range(35):
        try:
            o = ac.step_window()
        except np.linalg.LinAlgError as e:
            diverged_at=i; print(f"[{label}] LinAlgError@{i}: {e}"); break
        d = np.abs(ac.entry.state()['verts'][...,2]).max()
        lifts.append(o['F_lift']); defls.append(d)
        if not np.isfinite(o['z']) or d>10:
            diverged_at=i; break
    L=np.array(lifts)
    print(f"[{label}] windows={len(lifts)} diverged_at={diverged_at} "
          f"lift0={L[0]:+.1f} lift_mean={np.mean(L):+.1f} lift_std={np.std(L):.1f} "
          f"max_defl={max(defls):.4f}m" if defls else f"[{label}] no data")
    return L, defls

# baseline
run("baseline", {})
