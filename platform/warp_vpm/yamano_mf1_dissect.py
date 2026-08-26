"""Mf1 dissection: compare our added-mass matrix against the author's formula
on a flat plate, using the frozen Mf1 step-1 oracle as ground truth.

Author's formula (calc_fluid_force.m:140-154):
  nvec_Sc_global[ii, dof] = n_vec_i[ii,:] @ Sc_mat_col_global[:, dof]
  Mf1_mat = A_mat^{-1} @ nvec_Sc_global        (n_panels x n_dof)
  Mf1_gen = Σ_elements ∫ S^T @ p_interp @ Mf1_mat @ dA   (via Gauss quadrature)

Our formula (q16_flux_v5m_author_loads.py:448-490):
  source_vectors = A^{-1} * area_ratio * n
  quadrature contraction → Q16 transfer → .T @ neumann

This script:
1. Loads the frozen Mf1 step-1 oracle (A_mat, Neumann maps, etc.)
2. Reconstructs Mf1 using the author's exact formula
3. Reconstructs Mf1 using our formula
4. Compares the common-field work and identifies the discrepancy
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))
sys.path.insert(0, str(REPO / "platform/warp_vpm"))

from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET as CASE,
    load_mf1_step1_oracle,
    make_yamano2020_q16_model,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native import Q16NativeV5MSurface
from fluxvortex.warp_fsi import config as wsi_config

device = "cuda:0"
print("=" * 60)
print("Mf1 DISSECTION: author formula vs our implementation")
print("=" * 60)

# --- Load frozen oracle ---
oracle = load_mf1_step1_oracle()
print(f"\nOracle keys: {sorted(oracle.keys())}")
A_mat = oracle["A_mat"]  # (150, 150) AIC matrix
print(f"A_mat shape: {A_mat.shape}")

# Check what Neumann-related data exists
for k in sorted(oracle.keys()):
    v = oracle[k]
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} {v.dtype}")
    else:
        print(f"  {k}: {type(v)}")

# --- Build the Q16 model and surface ---
mesh, model, boundary = make_yamano2020_q16_model(
    chordwise_element_count=5, spanwise_element_count=3)
surface = Q16NativeV5MSurface(
    mesh,
    q16_chordwise_elements=5,
    q16_spanwise_elements=3,
    aerodynamic_chordwise_panels=15,
    aerodynamic_spanwise_panels=10,
    device=device,
)

# --- Get the initial flat geometry ---
from fluxvortex.warp_fsi.q16_flux_v5m_native import Q16NativeV5MGeometry
geometry = surface.evaluate_geometry(
    wp.from_torch(
        torch.as_tensor(
            np.ascontiguousarray(mesh.reference_state[None, :]),
            dtype=wsi_config.DTYPE, device=device),
        dtype=wsi_config.DTYPE))

print(f"\nGeometry: {geometry.areas.shape} areas, "
      f"{geometry.normals.shape} normals, "
      f"{geometry.collocation.shape} collocation")

# --- Compute AIC from our solver ---
from fluxvortex.warp_fsi.q16_flux_v5m_native import native_aic
aic = native_aic(geometry, device=device)
aic_np = wp.to_torch(aic).cpu().numpy()
print(f"Our AIC shape: {aic_np.shape}")

# Compare our AIC with author's
aic_diff = np.max(np.abs(aic_np - A_mat))
aic_rel = aic_diff / np.max(np.abs(A_mat))
print(f"AIC max|diff|: {aic_diff:.6e}  rel: {aic_rel:.6e}")

# --- Author's Mf1 formula ---
# Step 1: Build nvec_Sc_global (n_panels × n_dof)
# For each panel ii, evaluate Q16 shape functions at panel center and
# contract with the normal
nc, ns = 15, 10
n_panels = nc * ns
n_dof = mesh.reference_state.size
print(f"\nn_panels={n_panels}, n_dof={n_dof}")

# For the flat plate, all normals are (0,0,1)
normals_np = wp.to_torch(geometry.normals).cpu().numpy()
print(f"Normal at panel 0: {normals_np[0]}")
areas_np = wp.to_torch(geometry.areas).cpu().numpy()
print(f"Panel area[0]: {areas_np[0]:.6f}  total: {areas_np.sum():.6f}")

# --- Compare Mf1 oracle fields with our data ---
if "Neumann" in oracle or "neumann" in oracle:
    print("\nOracle has Neumann map")
if "Gamma_rate_map" in oracle or "gamma_rate" in oracle:
    print("Oracle has Gamma rate map")

# Check what the Mf1 oracle's common-field work computation looks like
if "common_field_work" in oracle:
    print(f"\nOracle common_field_work: {oracle['common_field_work']}")

# --- Direct comparison: apply a unit DOF displacement and compare ---
# The author's Mf1_mat maps structural DOF -> panel pressure
# Our Mf1 is assembled through a different path
# We can compare them by computing the common-field work both ways

# First, let's extract the author's nvec_Sc_global equivalent
# The oracle should have a Neumann or similar map
for key in ["Neumann_map", "neumann_map", "N_map", "nvec_Sc_global",
            "Sc_global", "shape_global"]:
    if key in oracle:
        print(f"\nFound oracle key '{key}': {oracle[key].shape}")
        break

# Let me check what modal data exists
if "modal_basis" in oracle:
    modal = oracle["modal_basis"]
    print(f"\nModal basis shape: {modal.shape}")
    # The common-field work is phi^T Mf1 phi for the first mode
    # Let's compute this using the author's A_mat and our AIC

# --- Summary so far ---
print("\n" + "=" * 60)
print("NEXT: assemble Mf1 both ways and compare common-field work")
print("=" * 60)

# Method 1: Author's formula using oracle A_mat
# Mf1_author = A_mat^{-1} @ nvec_Sc_global
# We need nvec_Sc_global which depends on the structural discretization

# Method 2: Our implementation
# Already have the assembly code; run it on the flat state

# For now, let's just verify the AIC matches and report
if aic_rel < 1e-6:
    print(f"\n✓ AIC matches oracle (rel={aic_rel:.2e})")
else:
    print(f"\n✗ AIC mismatch (rel={aic_rel:.2e})")

# Let's also compare the author's Mf1 common-field work with a direct
# computation using the oracle's A_mat
print("\nDone — next step is to implement the author's nvec_Sc_global")
print("and compare the full Mf1 assembly.")
