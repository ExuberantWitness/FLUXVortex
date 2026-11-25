"""Minimal time-marching model of the leading-edge vortex after Hirato: where does the coupled solve run away?

The same ring kernel and the same chain functions as the solver (kernels_ring_velocity,
lev_hirato), on a small lattice on the CPU: a rigid flat plate of `--strips` spanwise strips with
the Rojratsirikul chord and panel size, a start-up ramp, a trailing-edge wake of connected rows
solved implicitly (the P0-B row), and per strip a chain of leading-edge rings with the strength
solved together with the bound circulation (pin at the critical suction parameter, no penetration
everywhere else and at that leading-edge row).  No loads: the coefficient here is the
Kutta-Joukowski one on what the trailing edge sheds.

Switches take coupling channels out to find the one that destabilises:

  --no-chain-induction   the leading-edge rings induce nothing on the wing (not in the solve);
  --no-chain-convection  the leading-edge rings do not move (they keep their birth positions);
  --no-wake              no trailing-edge wake at all (the lattice sees only the sheet);
  --fixed-lambda X       the leading-edge strength is prescribed, the solve is linear in Gamma.

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 /tmp/zcode_lev_minimal.py --alpha 16 --steps 400 --every 25 [--strips 3] [...]
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

sys.path.insert(0, "src")
from fluxvortex.warp_fsi import kernels_ring_velocity as kernel  # noqa: E402
from fluxvortex.warp_fsi import lev_hirato as h  # noqa: E402

CHORD, NC = 0.0688, 15
DY = 0.1375 / 30                      # the production spanwise panel
SPEED, DT_STAR, LESP_FACTOR = 5.0, 0.01, 1.13
CORE = 0.1                            # FLUXV_FREE_WAKE_CORE_FRACTION


def velocity(points, rings, strengths, cored=False):
    if rings.shape[0] == 0:
        return torch.zeros_like(points)
    fraction = CORE if cored else 1.0e-6
    reference = max(kernel.mean_chordwise_panel(rings), CHORD / NC)
    # the expanded path materialises points x rings x 4 x 3: fuse the larger sums, as the solver does
    if points.shape[0] * rings.shape[0] * 3 * 8 > 64 * 1024 * 1024:
        return kernel.native_ring_velocity_fused(points, rings, strengths, core_fraction=fraction, reference_length=reference)
    expanded = kernel.native_ring_velocity_expanded(points, rings, core_fraction=fraction, reference_length=reference)
    return torch.sum(expanded * strengths[None, :, None], dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--strips", type=int, default=3)
    parser.add_argument("--crit", type=float, default=0.11)
    parser.add_argument("--no-chain-induction", action="store_true")
    parser.add_argument("--no-chain-convection", action="store_true")
    parser.add_argument("--no-wake", action="store_true")
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument("--wake-rows", type=int, default=100)
    parser.add_argument("--rows", type=int, default=h.LEV_MAX_ROWS, help="ring rows kept per strip")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.device != "cpu":
        torch.set_default_device(arguments.device)

    ns, nc = arguments.strips, NC
    dx, dt = CHORD / nc, DT_STAR * CHORD / SPEED
    span = ns * DY
    crit = arguments.crit
    alpha = math.radians(arguments.alpha)
    dtype = torch.float64

    # lattice as the solver builds it: rings from 0.25 dx to c + 0.25 dx, collocation at the centres
    front = 0.25 * dx + torch.arange(nc, dtype=dtype) * dx
    y = torch.arange(ns + 1, dtype=dtype) * DY
    def cols(v):
        return v.reshape(nc, 1, 1).expand(nc, ns, 1)
    left = y[:-1].reshape(1, ns, 1).expand(nc, ns, 1)
    right = y[1:].reshape(1, ns, 1).expand(nc, ns, 1)
    zero = torch.zeros(nc, ns, 1, dtype=dtype)
    rings = torch.cat((
        torch.cat((cols(front), left, zero), 2), torch.cat((cols(front), right, zero), 2),
        torch.cat((cols(front + dx), right, zero), 2), torch.cat((cols(front + dx), left, zero), 2),
    ), 2).reshape(-1, 4, 3)
    collocation = torch.cat(((cols(front) + cols(front + dx)) / 2, (left + right) / 2, zero), 2).reshape(-1, 3)
    normals = torch.zeros(nc * ns, 3, dtype=dtype)
    normals[:, 2] = 1.0
    leading = torch.stack((torch.zeros(ns + 1, dtype=dtype), y, torch.zeros(ns + 1, dtype=dtype)), dim=1)
    rear = h.bound_rear(rings, nc)
    pseudovortex = h.pseudovortex_ring(leading, rear)
    edge = h.strip_pairs(leading)
    reference_bound = kernel.mean_chordwise_panel(rings)
    panels = nc * ns
    first_dx = 0.5 * (
        torch.linalg.vector_norm(rings[:ns, 3] - rings[:ns, 0], dim=1) + torch.linalg.vector_norm(rings[:ns, 2] - rings[:ns, 1], dim=1)
    )
    theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / CHORD, -1.0, 1.0))
    scale = CHORD * SPEED * (theta + torch.sin(theta)) / LESP_FACTOR
    chord_tangent = 0.5 * (rear[:, 0] + rear[:, 1]) - 0.5 * (edge[:, 0] + edge[:, 1])
    chord_tangent = chord_tangent / torch.linalg.vector_norm(chord_tangent, dim=1, keepdim=True)
    sin_a = torch.full((ns,), math.sin(alpha), dtype=dtype)
    cos_a = torch.full((ns,), math.cos(alpha), dtype=dtype)
    aic = torch.sum(
        kernel.native_ring_velocity_expanded(
            collocation, rings, core_fraction=1.0e-6, reference_length=reference_bound
        ) * normals[:, None, :], dim=2,
    )

    gamma = torch.zeros(panels, dtype=dtype)
    wake_rings = torch.zeros((0, 4, 3), dtype=dtype)
    wake_gamma = torch.zeros(0, dtype=dtype)
    chain = h.LevChain.empty(ns, torch.device(arguments.device), rows=arguments.rows)
    history = []
    print(f"alpha {arguments.alpha:g}, {ns} strips, crit {crit}; "
          + " ".join(name for name in ("no_chain_induction", "no_chain_convection", "no_wake") if getattr(arguments, name))
          + (f" fixed lambda {arguments.fixed_lambda}" if arguments.fixed_lambda is not None else ""), flush=True)
    print(f"{'step':>5} {'act':>4} {'lam_max':>8} {'lam_min':>8} {'G_TE':>7} {'shed':>7} {'Cn_KJ':>7} {'|Gb|max':>8} {'rows':>4}", flush=True)

    def state_velocity(points, gamma_now, core_wake=True):
        value = velocity(points, rings, gamma_now)
        if wake_rings.shape[0] and not arguments.no_wake:
            value = value + velocity(points, wake_rings.reshape(-1, 4, 3), wake_gamma, cored=core_wake)
        if chain.count.max() > 0 and not arguments.no_chain_induction:
            cr, cg = h.flat(chain)
            value = value + velocity(points, cr, cg, cored=True)
        return value

    def rk3(points, gamma_now):
        free = torch.tensor([SPEED * math.cos(alpha), 0.0, SPEED * math.sin(alpha)], dtype=dtype)
        base = state_velocity(points, gamma_now) + free[None, :]
        mid = state_velocity(points + 0.5 * dt * base, gamma_now) + free[None, :]
        last = state_velocity(points + dt * (-base + 2.0 * mid), gamma_now) + free[None, :]
        return points + (dt / 6.0) * (base + 4.0 * mid + last)

    for step in range(1, arguments.steps + 1):
        factor = 1.0 if step >= 100 else 0.5 * (1.0 - math.cos(math.pi * step / 100))
        v_inf = torch.tensor([SPEED * math.cos(alpha), 0.0, SPEED * math.sin(alpha)], dtype=dtype) * factor
        scale_now = scale * factor
        # 1. convection (the ramp scales the freestream; the induction is what it is)
        if wake_rings.shape[0] and not arguments.no_wake:
            flat = wake_rings.reshape(-1, 3)
            moved = rk3(flat, gamma) if factor == 1.0 else flat + dt * v_inf[None, :]
            wake_rings = moved.reshape(-1, 4, 3)
            rows = wake_rings.reshape(-1, ns, 4, 3)
            if rows.shape[0] > 1:
                rows[1:, :, 0] = rows[:-1, :, 3]
                rows[1:, :, 1] = rows[:-1, :, 2]
        if chain.count.max() > 0 and not arguments.no_chain_convection:
            cr, cg = h.flat(chain)
            corners = chain.rings.reshape(-1, 3)
            moved = rk3(corners, gamma) if factor == 1.0 else corners + dt * v_inf[None, :]
            chain = h._mask_invalid(h.reconnect(chain.with_rings(moved.reshape(chain.rings.shape))), chain.count)
        # 2. the solve
        wake_v = velocity(collocation, wake_rings.reshape(-1, 4, 3), wake_gamma, cored=True) if (wake_rings.shape[0] and not arguments.no_wake) else torch.zeros(panels, 3, dtype=dtype)
        chain_v = torch.zeros(panels, 3, dtype=dtype)
        if chain.count.max() > 0 and not arguments.no_chain_induction:
            cr, cg = h.flat(chain)
            chain_v = velocity(collocation, cr, cg, cored=True)
        rhs = -torch.sum((v_inf[None, :] + wake_v + chain_v) * normals, dim=1)
        system = aic.clone()
        nascent_normal = torch.zeros(panels, ns, dtype=dtype)
        nascent_rings = None
        if not arguments.no_wake:
            if wake_rings.shape[0]:
                front0 = torch.stack((wake_rings.reshape(-1, ns, 4, 3)[0, :, 0], wake_rings.reshape(-1, ns, 4, 3)[0, :, 1]), dim=1)
            else:
                front0 = rear + v_inf[None, :] * dt
            nascent_rings = h.nascent_row(front0, rear)
            nascent_normal = torch.sum(
                kernel.native_ring_velocity_expanded(collocation, nascent_rings, core_fraction=1.0e-6, reference_length=reference_bound)
                * normals[:, None, :], dim=2,
            )
            system[:, -ns:] += nascent_normal
        gamma_pre = torch.linalg.solve(system, rhs)
        lesp = -gamma_pre[:ns] / scale_now
        active = torch.abs(lesp) > crit
        lam = torch.zeros(ns, dtype=dtype)
        if bool(active.any()) or h.continuing(chain).any():
            active = active | h.continuing(chain)
            target = -torch.sign(lesp) * crit * scale_now
            aft = h.row_aft_pairs(chain.rings[0])
            first = h.first_free_points(edge, chord_tangent, normals[:ns], SPEED * factor, dt, -lesp, sin_a, cos_a)
            free = torch.where(h.continuing(chain)[:, None, None], h.split_points(edge, aft), first)
            nascent = h.nascent_row(free, edge)
            def unit_normal(rset):
                return torch.sum(
                    kernel.native_ring_velocity_expanded(collocation, rset, core_fraction=1.0e-6, reference_length=reference_bound)
                    * normals[:, None, :], dim=2,
                )
            basis = unit_normal(nascent) + unit_normal(pseudovortex)
            if arguments.fixed_lambda is not None:
                index = torch.nonzero(active, as_tuple=False).reshape(-1)
                gamma = torch.linalg.solve(system, rhs - basis[:, index] @ torch.full((index.numel(),), arguments.fixed_lambda, dtype=dtype))
                lam[index] = arguments.fixed_lambda
            else:
                # the solver's iteration: strips that cross join; strips that would need a reversed
                # leading-edge vortex leave
                was = h.continuing(chain)
                while True:
                    index = torch.nonzero(active, as_tuple=False).reshape(-1)
                    gamma, strength, residual = h.solve_coupled(system, nascent_normal, basis[:, index], rhs, active, target)
                    lam.zero_()
                    lam[index] = strength
                    solved = -gamma[:ns] / scale_now
                    crossed = (torch.abs(solved) > crit) & ~active
                    reversed_sign = was & active & (lam * torch.sign(target) < 0.0)
                    if not (bool(crossed.any()) or bool(reversed_sign.any())):
                        break
                    active = (active | crossed) & ~reversed_sign
            chain, leaving, leaving_gamma, released = h.push_nascent(chain, active, nascent, lam)
        else:
            gamma = gamma_pre
        chain = h.LevChain(rings=chain.rings, gamma=chain.gamma, count=chain.count,
                           prev=torch.where(active, lam, 0.0), joined=chain.joined)
        # 3. the trailing-edge row
        if not arguments.no_wake:
            new_rings = nascent_rings
            new_gamma = gamma.reshape(nc, ns)[-1] + lam
            wake_rings = torch.cat((new_rings, wake_rings), dim=0)
            wake_gamma = torch.cat((new_gamma, wake_gamma), dim=0)
            if wake_rings.shape[0] > arguments.wake_rows * ns:
                wake_rings, wake_gamma = wake_rings[: arguments.wake_rows * ns], wake_gamma[: arguments.wake_rows * ns]
        shed = float((gamma.reshape(nc, ns)[-1] + lam).sum().item())
        if step >= 1 and (step % arguments.every == 0 or step == arguments.steps):
            print(f"{step:5d} {int(active.sum()):4d} {float(lam.max()):8.4f} {float(lam.min()):8.4f} "
                  f"{'L' + format(float(lesp.abs().max()), '.3f'):>6s}"
                  f"{float(gamma.reshape(nc, ns)[-1].sum()):+7.4f} {shed:+7.4f} "
                  f"{2.0 * shed * DY / (SPEED * CHORD * span) * factor:+7.4f} {float(gamma.abs().max()):8.4f} "
                  f"{int(chain.count.max()):4d}", flush=True)
        history.append((float(lam.max()), float(gamma.abs().max()), shed))
    final = history[-20:]
    early = history[len(history) // 3 : len(history) // 3 + 20]
    print(f"mean max lambda over steps {len(history) // 3}-{len(history) // 3 + 20}: "
          f"{sum(item[0] for item in early) / len(early):.4f}; over the last 20: {sum(item[0] for item in final) / len(final):.4f}; "
          f"largest |Gamma_b| in the last 20: {max(item[1] for item in final):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
