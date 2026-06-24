import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
import flex_aircraft as F

np.set_printoptions(suppress=True, precision=4)


def run(ac, N=30, label=""):
    """Run N windows, return divergence window (or None), lift stats, max deflection."""
    lifts, defls, zs = [], [], []
    div_at = None
    try:
        for i in range(N):
            o = ac.step_window()
            d = float(np.abs(ac.entry.state()['verts'][..., 2]).max())
            lifts.append(o['F_lift']); defls.append(d); zs.append(o['z'])
            if (not np.isfinite(o['F_lift'])) or (not np.isfinite(d)) or d > 1e3:
                div_at = i
                break
    except Exception as e:
        div_at = i if 'i' in dir() else 0
        lifts.append(float('nan'))
        print(f"    [{label}] EXC@{div_at}: {type(e).__name__}: {str(e)[:60]}")
    lifts = np.array(lifts); defls = np.array(defls)
    fin = lifts[np.isfinite(lifts)]
    # lift std over windows 5+ (skip transient)
    seg = lifts[5:] if len(lifts) > 5 else lifts
    seg = seg[np.isfinite(seg)]
    lift_std = float(np.std(seg)) if len(seg) else float('nan')
    lift_mean = float(np.mean(seg)) if len(seg) else float('nan')
    maxd = float(np.nanmax(defls[np.isfinite(defls)])) if np.isfinite(defls).any() else float('nan')
    survived = (div_at is None) and len(lifts) >= N
    return dict(div_at=div_at, lift_std=lift_std, lift_mean=lift_mean,
                max_defl=maxd, survived=survived, n=len(lifts))


def measure_madd(ac):
    """Force a trial with added_mass_operator on; report madd magnitude vs physical."""
    prov = ac.provider
    saved = prov.added_mass_operator
    prov.added_mass_operator = True
    st = ac.entry.state()
    out = prov._trial(st)
    madd = out['madd']
    prov.added_mass_operator = saved
    nn = ac.entry.shell.nn
    zidx = 9 * np.arange(nn) + 2
    Mzz = madd[np.ix_(zidx, zidx)]
    total_madd = float(Mzz.sum())          # row-summed lumped added mass (z)
    diag_madd = float(np.trace(Mzz))
    return dict(total_madd=total_madd, diag_madd=diag_madd,
                max_abs=float(np.abs(Mzz).max()),
                Mzz_shape=Mzz.shape)


print("=" * 78)
print("STRUCTURAL MASS vs ADDED MASS (the real ratio)")
print("=" * 78)
ac0 = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                     V0=6.0, trim_aoa_deg=6.0, substeps=16)
sh = ac0.entry.shell
Md = sh.M.toarray()
xdofs = np.arange(sh.ndof)[np.arange(sh.ndof) % 9 == 0]
m_struct = Md[np.ix_(xdofs, xdofs)].sum()
rho_air = 1.225; area = 0.29 * 0.85; chord = 0.29; span = 0.85
m_add_phys_chord = rho_air * area * chord
m_add_phys_plate = rho_air * np.pi / 4 * chord**2 * span
print(f"  wing STRUCTURAL mass (rho_s*h*A) = {m_struct:.4f} kg")
print(f"  physical added mass rho*A*c       = {m_add_phys_chord:.4f} kg")
print(f"  physical added mass rho*pi/4*c^2*s = {m_add_phys_plate:.4f} kg")
print(f"  -> structural/added ratio ~ {m_struct/m_add_phys_plate:.1f}  (>>1 = HEAVY wing, NOT light)")

mm = measure_madd(ac0)
print(f"\n  madd operator (z-z block): total(rowsum)={mm['total_madd']:.4f} kg  "
      f"diag={mm['diag_madd']:.4f} kg  max|elt|={mm['max_abs']:.4f}")
print(f"  -> madd total vs physical plate added mass {m_add_phys_plate:.4f}: "
      f"ratio={mm['total_madd']/m_add_phys_plate:.2f}")

print("\n" + "=" * 78)
print("(a) m_wing SWEEP (FlexAircraft arg) @ substeps=16, baseline coupling")
print("    NOTE: m_wing arg only changes fuselage m_total, NOT wing structure!")
print("=" * 78)
for mw in [0.08, 0.2, 0.5, 1.0]:
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                        V0=6.0, trim_aoa_deg=6.0, substeps=16, m_wing=mw)
    r = run(ac, 30, f"m_wing={mw}")
    div = r['div_at'] if r['div_at'] is not None else '-'
    print(f"  m_wing={mw:4.2f}: div@{div!s:>3}  survived={r['survived']!s:5}  "
          f"lift_std={r['lift_std']:8.1f}N  lift_mean={r['lift_mean']:7.1f}N  "
          f"max_defl={r['max_defl']:.3g}m")

print("\n" + "=" * 78)
print("(a') TRUE wing-mass sweep via rho_scale (changes STRUCTURAL inertia)")
print("=" * 78)
for rs in [1.0, 2.0, 4.0, 8.0]:
    ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                        V0=6.0, trim_aoa_deg=6.0, substeps=16)
    ac.entry.shell.set_distribution(rho_scale=rs)  # scale wing density
    r = run(ac, 30, f"rho_scale={rs}")
    div = r['div_at'] if r['div_at'] is not None else '-'
    eff_m = m_struct * rs
    print(f"  rho_scale={rs:4.1f} (m_struct={eff_m:.3f}kg, ratio={eff_m/m_add_phys_plate:5.1f}): "
          f"div@{div!s:>3}  survived={r['survived']!s:5}  "
          f"lift_std={r['lift_std']:8.1f}N  max_defl={r['max_defl']:.3g}m")
