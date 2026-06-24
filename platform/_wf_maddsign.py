import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np
import flex_aircraft as F

ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0,
                    V0=6.0, trim_aoa_deg=6.0, substeps=16)
prov = ac.provider
prov.added_mass_operator = True
st = ac.entry.state()
out = prov._trial(st)
madd = out['madd']
sh = ac.entry.shell
nn = sh.nn
zidx = 9 * np.arange(nn) + 2
Mzz = madd[np.ix_(zidx, zidx)]

print("madd z-z block stats:")
print(f"  shape={Mzz.shape}")
print(f"  trace(diag sum) = {np.trace(Mzz):+.5f} kg")
print(f"  rowsum total    = {Mzz.sum():+.5f} kg")
print(f"  min diag        = {np.diag(Mzz).min():+.5f}")
print(f"  max diag        = {np.diag(Mzz).max():+.5f}")
print(f"  symmetric?      = {np.allclose(Mzz, Mzz.T, atol=1e-9)}")
# eigenvalues sign
ev = np.linalg.eigvalsh(0.5 * (Mzz + Mzz.T))
print(f"  sym eig range   = [{ev.min():+.4f}, {ev.max():+.4f}]   "
      f"#neg={int((ev < -1e-9).sum())} #pos={int((ev > 1e-9).sum())}")

# How it enters: M_eff = M_struct - M_add.  Compare to free-free struct mass.
M = sh.M.toarray()
bc = sorted(sh._bc_dofs)
free = np.setdiff1d(np.arange(sh.ndof), bc)
Mff = M[np.ix_(free, free)]
Maddff = madd[np.ix_(free, free)]
Meff = Mff - Maddff
ev_M = np.linalg.eigvalsh(0.5 * (Mff + Mff.T))
ev_Me = np.linalg.eigvalsh(0.5 * (Meff + Meff.T))
print(f"\n  Mff (struct free) sym eig min = {ev_M.min():+.5g} (PD => all>0)")
print(f"  M_eff = Mff - Madd  sym eig min = {ev_Me.min():+.5g}")
print(f"  -> M_eff still positive-definite? {ev_Me.min() > 0}")
print(f"  fraction of struct z-inertia removed by Madd: "
      f"{Maddff[ (free%9==2), :][:, (free%9==2)].trace() / Mff[(free%9==2),:][:,(free%9==2)].trace():.3f}")

# physical added mass per wing (flat plate) for reference
rho_air = 1.225; chord = 0.29; span = 0.85
m_add_phys = rho_air * np.pi / 4 * chord**2 * span
print(f"\n  physical plate added mass (rho*pi/4*c^2*s) = {m_add_phys:.4f} kg")
print(f"  madd total rowsum                          = {Mzz.sum():.4f} kg")
print(f"  ratio madd_total / physical                = {Mzz.sum()/m_add_phys:.2f}")
