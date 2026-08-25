"""DIAGNOSTIC: does the rigid-plate steady normal force drift upward as the
high-circulation wake accumulates?  Extends the 40-step finding (+0.01/step
drift) to 200 steps with the full 300-row wake and particle culling, i.e.
the exact production aero configuration minus the structural response."""
import sys, math
sys.path.insert(0, "src"); sys.path.insert(0, "platform"); sys.path.insert(0, "platform/warp_vpm")
import numpy as np, torch, warp as wp
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig, Q16NativeV5MSolver, Q16NativeV5MSurface, Q16NativeV5MOwner
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID, FORMAL_Q16_GRID, ROJ11_A16, make_rojratsirikul2011_q16_model, plate_normal)

case = ROJ11_A16
mesh, _, _, _ = make_rojratsirikul2011_q16_model(
    chordwise_element_count=FORMAL_Q16_GRID[0], spanwise_element_count=FORMAL_Q16_GRID[1], case=case)
surface = Q16NativeV5MSurface(mesh, q16_chordwise_elements=FORMAL_Q16_GRID[0],
    q16_spanwise_elements=FORMAL_Q16_GRID[1], aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
    aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1], device=config.DEVICE)
solver = Q16NativeV5MSolver(surface, NativeV5MConfig(
    chordwise_panels=FORMAL_AERO_GRID[0], spanwise_panels=FORMAL_AERO_GRID[1],
    density=case.fluid_density_kg_m3, freestream=case.freestream_m_s,
    aerodynamic_dt=case.aerodynamic_dt_s, lesp_crit=case.lesp_crit,
    wake_max_rows=300, particle_capacity=32768, particle_max_age_steps=100,
    wake_history_mode="bound_rate", wake_free_rows=100,
    dvm_target_spacing_chord=0.018, device=config.DEVICE))
state = wp.array(np.ascontiguousarray(mesh.reference_state[None, :]), dtype=config.DTYPE, device=config.DEVICE)
velocity = wp.zeros_like(state)
owner = Q16NativeV5MOwner(solver.initialize(state, velocity))
qs = case.dynamic_pressure_pa * case.reference_area_m2
normal_t = torch.tensor(plate_normal(case), device="cuda:0", dtype=torch.float64)
print("rigid plate, full speed; paper steady Cn ~ 0.92-0.95", flush=True)
for step in range(1, 201):
    proposal = owner.propose(solver, state, velocity)
    owner.commit(proposal)
    if step % 10 == 0:
        fz = float((proposal.load.total_force @ normal_t).item())
        d = proposal.trial_state.diagnostics[-1]
        print(f"step {step:3d}: Cn={fz/qs:+8.4f} parts={d['particle_count']:6d} wake={d['wake_ring_count']:5d} lmax={d['lesp_pre_max_abs']:.3f}", flush=True)
