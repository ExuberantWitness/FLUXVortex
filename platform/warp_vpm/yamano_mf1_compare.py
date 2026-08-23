"""Mf1 comparison: author frozen oracle vs our solver's output."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))
sys.path.insert(0, str(REPO / "platform/warp_vpm"))

from forward_flight_benchmarks.yamano2020_q16 import (
    load_mf1_step1_oracle, make_yamano2020_q16_model,
    YAMANO_2020_SINGLE_SHEET as CASE,
)
from fluxvortex.warp_fsi import config as wsi
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    Q16NativeV5MSurface, Q16NativeV5MSolver, NativeV5MConfig, native_aic)

device = "cuda:0"
oracle = load_mf1_step1_oracle()
aic_author = oracle["aic"]
neumann_author = oracle["neumann"]
gamma_rate_author = oracle["gamma_rate"]
phi = oracle["phi_dq"]
n = int(oracle["matrix_shape"][0])
mf1_author = sp.csr_matrix(
    (oracle["mf1_data"], (oracle["mf1_row"], oracle["mf1_col"])),
    shape=(n, n)).toarray()

print("AUTHOR Mf1 (oracle)")
for i in range(5):
    w = phi[:, i] @ mf1_author @ phi[:, i]
    print(f"  mode {i}: {w:.10f}")
work_author = phi[:, 0] @ mf1_author @ phi[:, 0]

print("\nBuilding solver...")
mesh, model, boundary = make_yamano2020_q16_model(chordwise_element_count=5, spanwise_element_count=3)
settings = NativeV5MConfig(
    chordwise_panels=15, spanwise_panels=10,
    density=CASE.fluid_density_kg_m3,
    freestream=CASE.freestream_m_s,
    aerodynamic_dt=CASE.aerodynamic_dt_s,
    lesp_crit=0.11, wake_max_rows=96)
surface = Q16NativeV5MSurface(
    mesh, q16_chordwise_elements=5, q16_spanwise_elements=3,
    aerodynamic_chordwise_panels=15, aerodynamic_spanwise_panels=10,
    device=device)
aero = Q16NativeV5MSolver(surface, settings)

state0 = wp.from_torch(
    torch.as_tensor(
        np.ascontiguousarray(mesh.reference_state[None, :]),
        dtype=torch.float64, device=device),
    dtype=wp.float64)
vel0 = wp.zeros_like(state0)

committed0 = aero.initialize(state0, vel0)
prop0 = aero.propose(committed0, state0, vel0)
mf1_ours = (prop0.author_load.added_mass.generalized_matrix
            .detach().cpu().numpy())

print(f"\nOUR Mf1: shape={mf1_ours.shape}")
for i in range(5):
    w = phi[:, i] @ mf1_ours @ phi[:, i]
    wa = phi[:, i] @ mf1_author @ phi[:, i]
    r = w / wa if abs(wa) > 1e-15 else float("nan")
    print(f"  mode {i}: ours={w:.10f}  author={wa:.10f}  ratio={r:.6f}")
work_ours = phi[:, 0] @ mf1_ours @ phi[:, 0]

print(f"\n  trace: ours={np.trace(mf1_ours):.8f}  author={np.trace(mf1_author):.8f}")
print(f"  frobenius: ours={np.linalg.norm(mf1_ours):.8f}  author={np.linalg.norm(mf1_author):.8f}")

# AIC comparison
geometry = surface.evaluate(state0, vel0)
aic_ours = native_aic(geometry, chordwise_panels=15)
aic_ours_np = (aic_ours.cpu().numpy() if isinstance(aic_ours, torch.Tensor)
               else wp.to_torch(aic_ours).cpu().numpy())
aic_diff = np.max(np.abs(aic_ours_np - aic_author))
aic_rel = aic_diff / np.max(np.abs(aic_author))
print(f"\nAIC rel err: {aic_rel:.6e}  {'PASS' if aic_rel < 1e-6 else 'MISMATCH'}")

# Gamma rate cross-check
gamma_ours = np.linalg.solve(aic_ours_np, neumann_author)
gamma_err = np.max(np.abs(gamma_ours - gamma_rate_author))
gamma_rel = gamma_err / np.max(np.abs(gamma_rate_author))
print(f"Gamma rate (our AIC, author neumann) rel: {gamma_rel:.6e}")

# Neumann map structure
neu_norms = np.linalg.norm(neumann_author, axis=0)
nz = np.flatnonzero(neu_norms > 1e-15)
print(f"\nAuthor neumann: {nz.sum()} nonzero DOFs, range [{nz.min()},{nz.max()}]")
print(f"  first 10 DOF gaps: {np.diff(nz[:11])}")
print(f"  max col norm: {neu_norms.max():.6f}")

# Summary
print(f"\n{'='*60}")
print(f"DIAGNOSIS")
print(f"{'='*60}")
print(f"Author work: {work_author:.10f}")
print(f"Our work:    {work_ours:.10f}")
print(f"Ratio:       {work_ours/work_author:.6f}")
if aic_rel < 1e-6:
    print(f"\n→ AIC correct. Problem is in the Mf1 assembly AFTER AIC.")
    print(f"  Author: ∫ S^T · p_interp · (A⁻¹·neumann) · dA")
    print(f"  Ours:  (quad_xfer(A⁻¹·n·area))^T · neumann")
