"""Synchronous RK3 of native wake-ring and particle positions.

Circulation, particle strength and the bound solution are held fixed during
this material substep. Every stage updates sources AND targets of both free
fields before evaluating either velocity. Birth and retention remain owned
by the native solver.
"""
from __future__ import annotations

import math
import torch


def advance_material_positions_rk3(solver, geometry, trial) -> None:
    dt = solver.settings.aerodynamic_dt
    if not math.isfinite(dt) or dt <= 0.:
        raise ValueError("material delta_time must be finite and positive")
    field = trial.particle_field
    initial_w = trial.wake_rings.clone()
    initial_p = field.pos[:field.n].clone()
    ns = solver.settings.spanwise_panels
    free_rows = solver.settings.wake_free_rows
    active_count = min(len(initial_w), free_rows*ns) if free_rows else len(initial_w)

    def velocity(wake, particles):
        trial.wake_rings = wake
        field.pos[:field.n].copy_(particles)
        uw = solver.v_inf.expand_as(wake).clone()
        if active_count:
            points = wake[:active_count].reshape(-1,3)
            live = solver._external_velocity(points,geometry,trial)
            if field.n:
                live = live + field.velocity_at_cuda(points)
            uw[:active_count] = live.reshape(-1,4,3)
        up = solver._external_velocity(particles,geometry,trial)
        if field.n:
            up = up + field.velocity_self_cuda()
        return uw, up

    w1,p1 = velocity(initial_w, initial_p)
    w2,p2 = velocity(initial_w + .5*dt*w1, initial_p + .5*dt*p1)
    w3,p3 = velocity(initial_w + dt*(-w1+2*w2), initial_p + dt*(-p1+2*p2))
    final_w = initial_w + dt/6*(w1+4*w2+w3)
    final_p = initial_p + dt/6*(p1+4*p2+p3)
    if not torch.isfinite(final_w).all().item() or not torch.isfinite(final_p).all().item():
        trial.wake_rings = initial_w
        field.pos[:field.n].copy_(initial_p)
        raise FloatingPointError("synchronous material RK3 produced non-finite positions")
    trial.wake_rings = final_w
    field.pos[:field.n].copy_(final_p)
    field.promote_fresh()
