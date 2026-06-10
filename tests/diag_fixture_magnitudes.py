"""Quick diagnostic: inspect MATLAB fixture to understand Mf2_vec1's role.

Questions we answer:
  Q1. Is Mf2_vec1 the main force term or a small correction?
      → compare |Mf2_vec1| vs |dp_lift1| vs |dp_lift|
  Q2. Does Python's forces_no_vstruct (which = dp_lift1) miss most of the force?
  Q3. What's the wake shape (N_wake rows)?
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from load_matlab_fixture import MatlabFixture

FIX = "/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/FSI_by_FEM_and_UVLM/single_sheet/fixture_t0.1995.mat"
fx = MatlabFixture(FIX)

def stats(name, x):
    x = np.asarray(x).ravel()
    print(f"  {name:18s}: shape={np.asarray(fx._raw[name]).shape}  "
          f"|max|={np.max(np.abs(x)):.4e}  mean|·|={np.mean(np.abs(x)):.4e}  "
          f"rms={np.sqrt(np.mean(x**2)):.4e}")

print(f"\n=== Time/geometry ===")
print(f"  t* = {fx.time}, Nx={fx.Nx}, Ny={fx.Ny}, N_element={fx.Nx*fx.Ny}")
print(f"  Gamma_wake shape = {np.asarray(fx._raw['Gamma_wake']).shape}")
print(f"  dt_q1234_wake_mat shape = {np.asarray(fx._raw['dt_q1234_wake_mat']).shape}")

print(f"\n=== Pressure components (per-panel scalars) ===")
for n in ['dp_lift', 'dp_lift1', 'dp_lift2', 'dp_add', 'Mf2_vec1']:
    stats(n, fx._raw[n])

print(f"\n=== Q1: is Mf2_vec1 dominant or corrective? ===")
dp_lift1 = np.asarray(fx._raw['dp_lift1']).ravel()
mf2v1 = np.asarray(fx._raw['Mf2_vec1']).ravel()
ratio = np.abs(mf2v1) / (np.abs(dp_lift1) + 1e-15)
print(f"  |Mf2_vec1| / |dp_lift1| per-panel:")
print(f"    median = {np.median(ratio):.3f}")
print(f"    p95    = {np.percentile(ratio, 95):.3f}")
print(f"    max    = {np.max(ratio):.3f}")
combined = dp_lift1 + mf2v1
print(f"  |dp_lift1+Mf2_vec1| vs |dp_lift1|: ratio max = "
      f"{np.max(np.abs(combined)/(np.abs(dp_lift1)+1e-15)):.3f}")

print(f"\n=== Wake state ===")
gw = np.asarray(fx._raw['Gamma_wake']).ravel()
print(f"  Gamma_wake: {len(gw)} entries, n_wake_rows = {len(gw)//fx.Ny}")
print(f"  Gamma_wake max = {np.max(np.abs(gw)):.4e}")
gwdtn = np.asarray(fx._raw['Gamma_wake_dt_q1234_n']).ravel()
print(f"  Gamma_wake_dt_q1234_n (RHS of Mf2_vec1 solve): "
      f"|max|={np.max(np.abs(gwdtn)):.4e}, rms={np.sqrt(np.mean(gwdtn**2)):.4e}")

print(f"\n=== Total force composition ===")
dp_vec = np.asarray(fx._raw['dp_vec']).ravel()
dp_lift = np.asarray(fx._raw['dp_lift']).ravel()
dp_add = np.asarray(fx._raw['dp_add']).ravel()
print(f"  dp_vec      (dp_lift+dp_add): rms = {np.sqrt(np.mean(dp_vec**2)):.4e}")
print(f"  dp_lift     (used in weak coupling): rms = {np.sqrt(np.mean(dp_lift**2)):.4e}")
print(f"  dp_lift1+Mf2_vec1 (used in strong coupling): rms = "
      f"{np.sqrt(np.mean((dp_lift1+mf2v1)**2)):.4e}")
