"""DIAGNOSTIC ONLY: per-row-group decomposition of the mf2 wake-history term
and wake geometry statistics at alpha=16 deg full speed."""
import sys, math
sys.path.insert(0, "src"); sys.path.insert(0, "platform"); sys.path.insert(0, "platform/warp_vpm")
import numpy as np, torch, warp as wp
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi import q16_flux_v5m_native as NV
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig, Q16NativeV5MSolver, Q16NativeV5MSurface, Q16NativeV5MOwner
from fluxvortex.warp_fsi.q16_flux_v5m_author_loads import material_ring_velocity_derivative_expanded
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
    wake_max_rows=60, particle_capacity=32768, particle_max_age_steps=100,
    dvm_target_spacing_chord=0.018, device=config.DEVICE))
state = wp.array(np.ascontiguousarray(mesh.reference_state[None, :]), dtype=config.DTYPE, device=config.DEVICE)
velocity = wp.zeros_like(state)
owner = Q16NativeV5MOwner(solver.initialize(state, velocity))

# Spy on propose: after mf2 is computed we cannot easily intercept internals;
# instead recompute the decomposition from the committed state each step.
qs = case.dynamic_pressure_pa * case.reference_area_m2
normal_t = torch.tensor(plate_normal(case), device="cuda:0", dtype=torch.float64)
chord = case.chord_m
for step in range(1, 31):
    proposal = owner.propose(solver, state, velocity)
    owner.commit(proposal)
    trial = owner.state
    if step in (10, 20, 30) and trial.wake_rings.shape[0]:
        geometry = surface.evaluate(state, velocity)
        wake_vertex_velocity = solver._external_velocity(
            trial.wake_rings.reshape(-1, 3), geometry, trial
        ).reshape_as(trial.wake_rings)
        influence_rate = material_ring_velocity_derivative_expanded(
            geometry.collocation, geometry.collocation_velocity,
            trial.wake_rings, wake_vertex_velocity)
        wake_velocity_rate = torch.sum(
            influence_rate * trial.wake_gamma[None, :, None], dim=1)
        wake_normal_rate = torch.sum(wake_velocity_rate * geometry.normals, dim=1)
        ns = solver.settings.spanwise_panels
        rows = trial.wake_gamma.numel() // ns
        per_row = wake_normal_rate.reshape(-1, rows) if False else None
        # groups of rows: rate contribution per ring already summed over lanes;
        # reshape rings -> (rows, ns)
        rate_rings = torch.sum(influence_rate * trial.wake_gamma[None, :, None], dim=2)  # (colloc, ring)
        gamma_rate_n = torch.sum(wake_velocity_rate * geometry.normals, dim=1)
        # contribution of ring j to the Fz via A^-1 is global; approximate the
        # driver distribution: normal-rate induced by each row group
        ring_rate_n = rate_rings @ geometry.normals.T if False else None
        row_groups = {}
        wr = influence_rate * trial.wake_gamma[None, :, None]  # (colloc, ring, 3)
        ring_normal_rate = torch.einsum('ijk,ik->j', wr, geometry.normals)  # (ring,)
        contrib = torch.linalg.vector_norm(wr, dim=(0, 2)).reshape(rows, ns)
        nr = ring_normal_rate.reshape(rows, ns)
        g = trial.wake_gamma.reshape(rows, ns)
        x_front = trial.wake_rings.reshape(rows, ns, 4, 3)[:, :, 0, 0]
        x_back = trial.wake_rings.reshape(rows, ns, 4, 3)[:, :, 2, 0]
        spacing = (x_front[1:] - x_back[:-1]).abs().max(dim=1).values / chord
        print(f"step {step}: rows={rows} gamma_wake[row0] mean={g[0].mean().item():+.3e} "
              f"row_mid={g[rows//2].mean().item():+.3e} row_last={g[-1].mean().item():+.3e}")
        print(f"   contrib(norm) rows[0:5]={contrib[:5].mean().item():.3e} [5:20]={contrib[5:20].mean().item():.3e} [20:]={contrib[20:].mean().item():.3e}")
        print(f"   normal-rate sum rows[0:5]={nr[:5].sum().item():+.3e} [5:20]={nr[5:20].sum().item():+.3e} [20:]={nr[20:].sum().item():+.3e} total={nr.sum().item():+.3e}")
        print(f"   row spacing/c: first5={spacing[:5].mean().item():.4f} last5={spacing[-5:].mean().item():.4f}")
        print(f"   wake vertex speed|v|/U: front={torch.linalg.vector_norm(wake_vertex_velocity.reshape(rows,ns,4,3)[:1,:,0],dim=-1).mean().item()/5:.3f} "
              f"back={torch.linalg.vector_norm(wake_vertex_velocity.reshape(rows,ns,4,3)[-1,:,2],dim=-1).mean().item()/5:.3f}")
