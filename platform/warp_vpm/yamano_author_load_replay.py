"""Author-load structural replay: drive Q16 with author's frozen Qf oracle.

DIAGNOSTIC: no aero solver. Applies pulse + lerped author Qf + Mf1 at each
of 34 substeps per outer step. If Q16 reproduces author displacement, the
structural model is correct and the error is in the aero force path.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform"))
sys.path.insert(0, str(REPO / "platform/warp_vpm"))

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from fluxvortex.warp_fsi.q16_flux_v5m_author_loads import (
    Q16NativeAddedMassAction,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET, make_yamano2020_q16_model,
    load_tip_displacement_reference,
)
from yamano2020_q16_pulse import Yamano2020Q16CudaPulse

case = YAMANO_2020_SINGLE_SHEET
SUB = case.structural_substeps_per_aerodynamic_step  # 34
dt_s = case.structural_dt_s
print(f"case: dt*={dt_s}, substeps/aero={SUB}, "
      f"aero_dt*={case.aerodynamic_dt_star}")

# --- load author Qf oracle (4 steps) ---
oracle = np.load(
    REPO / "platform/forward_flight_benchmarks/data/"
    "yamano2020_matlab_mf2_history_oracle.npz")
author_qf = {}
for s in range(1, 5):
    author_qf[s] = torch.as_tensor(
        oracle[f"step_{s}_Qf_p_global"], device="cuda:0", dtype=torch.float64)
print(f"author Qf loaded: 4 steps, dim={author_qf[1].shape[0]}")

# --- build Q16 structural model ---
mesh, model, boundary = make_yamano2020_q16_model(
    chordwise_element_count=5, spanwise_element_count=3)
structural = Q16CudaNewmarkStepper(
    model, boundary, device=config.DEVICE,
    newton_tolerance=3.0e-7, max_newton_iterations=128,
    cg_tolerance=2.0e-10, max_cg_iterations=2048, cg_check_every=16,
    nonsymmetric_solver="reference_dense", reference_dense_refresh_after=48)

# --- pulse ---
pulse = Yamano2020Q16CudaPulse(model, case=case, device=config.DEVICE)

# --- Mf1 from a dummy aero state (we use the same Mf1 as the FSI solver) ---
# For the structural replay we use the Mf1 from step 1 (uniform flow, flat
# plate) — this is the same Mf1 the FSI uses at initialization.
# Build it from the author loads module's added_mass action.
# We'll extract it by running one propose() call on the flat initial state.
print("Building Mf1 from initial flat state...")
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    Q16NativeV5MSolver, Q16NativeV5MSurface,
)

surface = Q16NativeV5MSurface(
    mesh, model, n_chord=15, n_span=10, device=config.DEVICE)
aero = Q16NativeV5MSolver(
    surface, density=case.fluid_density_kg_m3,
    freestream_m_s=case.freestream_m_s,
    delta_time_s=case.aerodynamic_dt_s,
    lesp_critical=0.11, wake_max_rows=96,
    device=config.DEVICE)

state0 = wp.from_torch(
    torch.as_tensor(np.ascontiguousarray(mesh.reference_state[None, :]),
                    dtype=config.DTYPE, device=config.DEVICE),
    dtype=config.DTYPE)
vel0 = wp.zeros_like(state0)
prop0 = aero.propose(state0, vel0, parent_step=0)
mf1_matrix = prop0.aerodynamic_load.added_mass.generalized_matrix
mf1_action = Q16NativeAddedMassAction(mf1_matrix.detach().clone())
print(f"Mf1 matrix shape: {mf1_matrix.shape}, "
      f"norm: {mf1_matrix.norm():.6f}")

# --- reference displacement ---
ref = load_tip_displacement_reference()
ref_t = np.array([r[0] for r in ref])
ref_z = np.array([r[1] for r in ref])

# --- structural replay ---
ndof = mesh.reference_state.size
q = wp.from_torch(
    torch.as_tensor(np.ascontiguousarray(mesh.reference_state[None, :]),
                    dtype=config.DTYPE, device=config.DEVICE),
    dtype=config.DTYPE)
v = wp.zeros_like(q)
a = wp.zeros_like(q)

# author Qf at step 0 = zero (initial state)
prev_qf = torch.zeros(ndof, device="cuda:0", dtype=torch.float64)
tip_idx = model.surface_transfer_map.trailing_edge_span_indices[10 // 2]

results = []
t_star = 0.0
# startup step (t*=0.002)
startup_pulse = pulse.endpoint_load(dt_s)
q, v, a = structural.step(
    q, v, a, dt_s,
    constant_load=startup_pulse.generalized_force,
    acceleration_load_action=mf1_action,
)
t_star = dt_s / (case.chord_m / case.freestream_m_s)  # t*=0.002

for outer in range(4):  # only 4 steps have author Qf
    curr_qf = author_qf[outer + 1]
    # 34 substeps with lerped load
    for sub in range(SUB):
        beta = (sub + 1) / SUB
        lerped_qf = torch.lerp(prev_qf, curr_qf, beta)

        # pulse at this substep's time
        t_s = t_star * case.chord_m / case.freestream_m_s + sub * dt_s
        # actually: total structural time
        total_sub = outer * SUB + sub + 1
        t_s = (total_sub + 1) * dt_s  # approximate

        pulse_load = pulse.endpoint_load(t_s)
        total_load = pulse_load.generalized_force + lerped_qf

        q, v, a = structural.step(
            q, v, a, dt_s,
            constant_load=total_load,
            acceleration_load_action=mf1_action,
        )

        if sub == SUB - 2:  # substep 33 = paper sample
            q_t = wp.to_torch(q).cpu().numpy().flatten()
            tip_z = q_t[tip_idx * 3 + 2] if tip_idx * 3 + 2 < len(q_t) else 0
            paper_t = (outer + 1) * case.aerodynamic_dt_star
            ref_z_val = np.interp(paper_t, ref_t, ref_z)
            err = abs(tip_z - ref_z_val) / max(abs(ref_z_val), 1e-30) * 100
            results.append(dict(
                step=outer + 1, t_star=paper_t,
                tip_z=tip_z, author_z=ref_z_val, error_pct=err))
            print(f"  step {outer+1}: t*={paper_t:.3f}  tip_z={tip_z:.8f}  "
                  f"author={ref_z_val:.8f}  err={err:.3f}%")

    prev_qf = curr_qf

print("\n=== Author-load structural replay ===")
for r in results:
    print(f"  step {r['step']}: err={r['error_pct']:.3f}%")

# compare with full FSI errors at same steps
fsi_errs = [1.191, 2.040, 4.002, 7.968]
print(f"\nFull FSI errors at same steps: {fsi_errs}")
print(f"Structural replay errors: {[r['error_pct'] for r in results]}")

Path("/tmp/yamano_structural_replay.json").write_text(
    json.dumps(results, indent=2, default=float))
print("\nDONE — results saved to /tmp/yamano_structural_replay.json")
