import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')
import numpy as np
import flex_aircraft as F
ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0, V0=6.0, trim_aoa_deg=6.0, substeps=16)
print("coupler mode=", ac.pc.mode, "iters=", ac.pc.iterations, "adaptive_tol=", ac.pc.adaptive_tol,
      "residual_norm=", ac.pc.residual_norm, "interp=", ac.pc.interp)
print("provider.added_mass_operator=", ac.provider.added_mass_operator)
print("shell has _M_added:", hasattr(ac.entry.shell, "_M_added"))
print("m_total=", ac.m_total, "W=", ac.m_total*F.G, "dtw_ms=", ac.dtw*1e3, "substep dt_us=", ac.dtw/16*1e6)
print("ndof=", ac.entry.shell.ndof, "nn=", ac.entry.shell.nn)
# wing mass
M = ac.entry.shell.M
import scipy.sparse as sp
print("wing total mass (trace of pos-dofs /3):", M.diagonal()[0::9].sum())
