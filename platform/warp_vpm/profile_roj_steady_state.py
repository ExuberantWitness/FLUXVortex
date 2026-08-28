"""Profile the unified Roj runner at a synthetic steady-state field.

Injects 300 wake rows (9000 rings) + 24k LEV particles into the aero state
right after build, then times every major phase across 3 committed steps:
aero propose total, wake/particle convection, ring-induced velocities,
particle velocity queries, surface transfers, author-load assembly, AIC, and
each Newmark structural solve.  The synthetic field is not physical; only the
COST STRUCTURE matches the real step-300+ equilibrium (ring/particle counts).
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
import warp as wp

from fluxvortex.cases.rojratsirikul2011 import ROJ_A16_PRIMARY
from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

TIMER = defaultdict(float)
COUNTS = defaultdict(int)


def timed(name):
    def wrap(fn):
        def inner(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            TIMER[name] += time.perf_counter() - started
            COUNTS[name] += 1
            return result

        return inner

    return wrap


def inject_steady_state(runner) -> None:
    state = runner.owner.aerodynamic.state
    ns = runner.surface.ns
    rows = 300
    lanes = ns
    device = runner.device
    y = np.linspace(0.0, 0.1375, lanes + 1)
    rings = np.zeros((rows * lanes, 4, 3), dtype=np.float64)
    for r in range(rows):
        x0 = 0.07 + 0.002 * r
        z0 = 0.0
        for l in range(lanes):
            dx, dz = 0.0015, 0.0002 * ((r % 7) - 3)
            corners = np.array(
                [
                    [x0, y[l], z0],
                    [x0, y[l + 1], z0],
                    [x0 + dx, y[l + 1], z0 + dz],
                    [x0 + dx, y[l], z0 + dz],
                ]
            )
            rings[r * lanes + l] = corners
    state.wake_rings = torch.tensor(rings, device=device, dtype=torch.float64)
    gamma = np.zeros(rows * lanes, dtype=np.float64)
    for r in range(rows):
        gamma[r * lanes : (r + 1) * lanes] = 1.0e-6 * np.exp(-r / 60.0)
    state.wake_gamma = torch.tensor(gamma, device=device, dtype=torch.float64)

    field = state.particle_field
    n = 24000
    field.pos[:n] = torch.rand((n, 3), device=device, dtype=torch.float64) * torch.tensor(
        [0.1, 0.13, 0.02], device=device, dtype=torch.float64
    )
    field.gamma[:n] = torch.rand((n, 3), device=device, dtype=torch.float64) * 1e-9
    field.sigma[:n] = 0.01
    field.circul[:n] = 1e-9
    field.vol[:n] = 1e-9
    field.birth_step[:n] = 1
    field.n = n


def main() -> None:
    runner = RojratsirikulCaseRunner(ROJ_A16_PRIMARY)
    runner.build()
    inject_steady_state(runner)

    solver = runner.aerodynamic
    structural = runner.structural
    # Coarse, NON-overlapping instrumentation only: nested timers with their
    # two syncs per call serialize the launch pipeline and inflate the
    # unattributed bucket (measured: 29% -> 56% artifact when nesting).
    structural_cls = type(structural)
    structural_cls.step = timed("g.Newmark structural solve")(structural_cls.step)
    solver_cls = type(solver)
    solver_cls.propose = timed("TOTAL aero propose")(solver_cls.propose)
    import fluxvortex.warp_fsi.q16_flux_v5m_native_fsi as fsi_mod

    fsi_mod.Q16NativeV5MFSIStepper._integrate_structure = timed(
        "h.structural schedule (loads+solve glue)"
    )(fsi_mod.Q16NativeV5MFSIStepper._integrate_structure)

    steps = 5
    payload = runner.run(max_aero_steps=steps, output=None)
    total = TIMER["TOTAL aero propose"]
    schedule_total = TIMER["h.structural schedule (loads+solve glue)"]
    structural_total = TIMER["g.Newmark structural solve"]
    wall = payload["elapsed_seconds"]
    print(f"\n=== {steps} steps, wall {wall:.1f}s ({wall/steps:.1f} s/step) ===")
    for key in sorted(TIMER, key=TIMER.get, reverse=True):
        print(
            f"{TIMER[key]:8.2f}s  {COUNTS[key]:5d}x  {key}  "
            f"({100.0*TIMER[key]/wall:.1f}% wall, {TIMER[key]/max(COUNTS[key],1)*1000:.0f} ms/call)"
        )
    glue = wall - total - schedule_total
    print(
        f"coupling/runner glue (wall - aero - schedule): {glue:.1f}s "
        f"({100.0*glue/wall:.1f}%, {glue/steps:.2f} s/step)"
    )
    print(
        f"structural schedule split: solves {structural_total:.1f}s + "
        f"load-interpolation glue {schedule_total - structural_total:.1f}s"
    )


if __name__ == "__main__":
    main()
