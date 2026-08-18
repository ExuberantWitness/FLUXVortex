"""RK3 time stepping and particle shedding for the Warp VPM.

RK3 coefficients match the WRK3 stream: (0,1/3), (-5/9,15/16), (-153/128,8/15).
LEV shedding from LESP events; TEV from wake panel conversion.
"""
from __future__ import annotations

import numpy as np
from warp_vpm.pfield import ParticleField, TYPE_FRESH_SHED, TYPE_FREE

# Low-storage RK3 coefficients (same as WRK3 stream)
RK3_A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
RK3_B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)


def rk3_advance(pf: ParticleField, dt: float, vinf: np.ndarray, n_stages: int = 3):
    """Advance particle positions by dt using low-storage RK3.

    Velocity = self-induced + freestream (parent contribution added externally).
    Only updates positions; strength evolution (stretching) in Session 3.
    """
    if pf.n == 0:
        return

    # Memory register (position accumulator)
    mem = np.zeros_like(pf.positions)

    for stage in range(n_stages):
        a, b = RK3_A[stage], RK3_B[stage]

        # Total velocity at current positions
        v_self = pf.velocity_self()  # (N, 3)
        v_total = v_self + vinf[None, :]

        # Update memory and position
        mem = a * mem + v_total
        pf.pos[:pf.n] += b * dt * mem


def rk3_advance_with_parent(pf: ParticleField, dt: float,
                            vinf: np.ndarray,
                            parent_vel_fn, n_stages: int = 3):
    """RK3 with external parent velocity function parent_vel_fn(positions) -> (N,3)."""
    if pf.n == 0:
        return

    mem = np.zeros((pf.n, 3), dtype=np.float64)

    for stage in range(n_stages):
        a, b = RK3_A[stage], RK3_B[stage]
        pos = pf.positions
        v_self = pf.velocity_self()
        v_parent = parent_vel_fn(pos)
        v_total = v_self + v_parent + vinf[None, :]
        mem = a * mem + v_total
        pf.pos[:pf.n] += b * dt * mem


def shed_lev_particles(pf: ParticleField,
                       lesp_events: list,
                       step: int,
                       dt: float,
                       sigma_scale: float = 1.1):
    """Release LEV particles from LESP-critical events.

    lesp_events: list of dicts with keys 'step', 'spanwise_positions' (M,3),
    'gamma_lev' (M,3 vector circulation), 'sigma' (M,)
    """
    for event in lesp_events:
        if event["step"] != step:
            continue
        pos = event["spanwise_positions"]
        gam = event["gamma_lev"]
        sig = event.get("sigma", np.full(len(pos), 0.005))
        pf.add_particles(pos, gam, sig, ptype=TYPE_FRESH_SHED, birth_step=step)


def merge_aged_particles(pf: ParticleField, current_step: int,
                         age_threshold: int = 20,
                         cell_size: float = 0.15):
    """Circulation-conserving merge of old particles (grid-cell method)."""
    if pf.n < 100:
        return

    ages = current_step - pf.birth_step[:pf.n]
    old_mask = ages > age_threshold
    n_old = old_mask.sum()
    if n_old < 50:
        return

    idx = np.flatnonzero(old_mask)
    # Bin old particles into grid cells
    keys = np.floor(pf.pos[idx] / cell_size).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = len(uniq)
    if n_cells > n_old * 0.7:  # not enough merging opportunity
        return

    g = pf.gamma[idx]
    x = pf.pos[idx]
    s = pf.sigma[idx]
    w = np.linalg.norm(g, axis=1) + 1e-30

    # Merge: sum gamma (exact), strength-weighted centroid (impulse conserving)
    ng = np.zeros((n_cells, 3))
    nx = np.zeros((n_cells, 3))
    ns = np.zeros(n_cells)
    wsum = np.zeros(n_cells)
    np.add.at(ng, inv, g)
    np.add.at(nx, inv, x * w[:, None])
    np.add.at(ns, inv, s ** 3)
    np.add.at(wsum, inv, w)
    nx /= wsum[:, None]
    ns = np.maximum((ns / np.bincount(inv, minlength=n_cells)) ** (1.0 / 3.0),
                    cell_size * 0.5)

    # Replace old particles with merged ones
    pf.remove_indices(idx)
    pf.add_particles(nx, ng, ns, ptype=TYPE_FREE, birth_step=current_step)
