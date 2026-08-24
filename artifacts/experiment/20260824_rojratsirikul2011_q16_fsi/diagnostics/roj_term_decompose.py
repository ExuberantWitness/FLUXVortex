"""DIAGNOSTIC ONLY: decompose the native pressure terms under steady
full-speed flow at alpha=16 deg (no ramp) to isolate which term flips Cn
negative.  Runs 30 pure-aero proposes on the tilted rigid membrane."""
import sys, math
sys.path.insert(0, "src"); sys.path.insert(0, "platform"); sys.path.insert(0, "platform/warp_vpm")
import numpy as np, torch, warp as wp
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi import q16_flux_v5m_author_loads as AL
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig, Q16NativeV5MSolver, Q16NativeV5MSurface, Q16NativeV5MOwner
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID, FORMAL_Q16_GRID, ROJ11_A16, make_rojratsirikul2011_q16_model, plate_normal,
    normal_force_coefficient)

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
    wake_max_rows=60, particle_capacity=32768, particle_max_age_steps=100,
    dvm_target_spacing_chord=0.018, device=config.DEVICE))
state = wp.array(np.ascontiguousarray(mesh.reference_state[None, :]), dtype=config.DTYPE, device=config.DEVICE)
velocity = wp.zeros_like(state)
owner = Q16NativeV5MOwner(solver.initialize(state, velocity))

# Wrap assemble to record the two pressure components.
records = []
orig_assemble = AL.Q16NativeAuthorLoadAssembler.assemble
def spy_assemble(self, *, structural_state, geometry, aic, gamma, external_flow,
                 gamma_gradient, mf2_history):
    result = orig_assemble(self, structural_state=structural_state, geometry=geometry,
        aic=aic, gamma=gamma, external_flow=external_flow, gamma_gradient=gamma_gradient,
        mf2_history=mf2_history)
    pressure = result.constant_pressure
    dp1 = self.density * torch.sum(external_flow * gamma_gradient, dim=1)
    mf2p = self.density * mf2_history
    records.append({
        "dp_lift1_mean": float(dp1.mean().item()),
        "mf2_mean": float(mf2p.mean().item()),
        "dp_lift1_integral": float((dp1 * geometry.areas @ geometry.normals.T).sum().item()) if False else float(torch.sum(dp1[:, None] * geometry.areas[:, None] * geometry.normals, dim=0)[2].item()),
        "mf2_integral": float(torch.sum(mf2p[:, None] * geometry.areas[:, None] * geometry.normals, dim=0)[2].item()),
        "gamma_mean": float(gamma.mean().item()),
        "gamma_min": float(gamma.min().item()),
        "gamma_max": float(gamma.max().item()),
    })
    return result
AL.Q16NativeAuthorLoadAssembler.assemble = spy_assemble

qs = case.dynamic_pressure_pa * case.reference_area_m2
normal = plate_normal(case)
print(f"alpha={case.angle_deg} deg; thin-airfoil CL(2D)={2*math.pi*math.sin(math.radians(case.angle_deg)):.3f}; paper Cn~0.93")
for step in range(1, 41):
    proposal = owner.propose(solver, state, velocity)
    owner.commit(proposal)
    fz = float(proposal.load.total_force @ torch.tensor(normal, device='cuda:0', dtype=torch.float64))
    rec = records[-1]
    if step in (1, 2, 5, 10, 20, 30, 40):
        print(f"step {step:3d}: Cn={fz/qs:+8.4f}  F_dp1_z={rec['dp_lift1_integral']/qs:+8.4f}  F_mf2_z={rec['mf2_integral']/qs:+8.4f}  "
              f"gamma[min,max]=[{rec['gamma_min']:+.4e},{rec['gamma_max']:+.4e}]")
