"""Where does the flapping wing blow up under the new defaults?  Per-step probe of one condition.

Steps one Izraelevitz condition exactly as reproduce_izraelevitz2017_fig14_v5m_mandatory.py does
(same kinematics, grid, settings), and after every step records

  max|LESP|        the leading-edge suction parameter the solver reported,
  d(colloc, wake)  the smallest distance from any wake-ring corner or leg midpoint to any
                   collocation point, in chordwise panels,
  d(LE row, wake)  the same for the first (leading-edge) row only,
  d(bound, wake)   the smallest distance from any wake corner to any bound leg midpoint,
  gamma            the largest bound circulation.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \\
        python3 /tmp/zcode_flap_probe.py --case IZRA-15-015 --steps 400 [--from 300 --every 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd()
for sub in ("src", "platform", "platform/warp_vpm"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import torch  # noqa: E402

import reproduce_izraelevitz2017_fig14_v5m_mandatory as flap  # noqa: E402


def closest(first: torch.Tensor, second: torch.Tensor) -> float:
    """Smallest distance between two point clouds, in chunks over the first one."""
    best = float("inf")
    for start in range(0, first.shape[0], 4096):
        block = first[start : start + 4096]
        best = min(best, float(torch.cdist(block, second).min().item()))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="IZRA-15-015")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--from", dest="first", type=int, default=1)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--cycles-report", type=int, default=0)
    arguments = parser.parse_args()
    from fluxvortex.warp_fsi import kernels_ring_velocity as kernel
    from fluxvortex.warp_fsi import q16_flux_v5m_native as native
    from fluxvortex.aero.v5m.stepper import RigidV5MStepper
    from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface

    case = flap.IZRA_CASES[arguments.case]
    nc, ns = 8, 24
    steps_per_cycle = flap.DEFAULT_STEPS_PER_CYCLE
    aero_dt = 1.0 / case.frequency_hz / steps_per_cycle
    kin = flap.IzraRigidSurfaceKinematics(case, chordwise_panels=nc, spanwise_panels=ns, device="cuda:0")
    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device="cuda:0")
    settings = native.NativeV5MConfig(
        chordwise_panels=nc, spanwise_panels=ns, density=case.rho_kg_m3, freestream=case.freestream_m_s,
        aerodynamic_dt=aero_dt, wake_max_rows=128, particle_capacity=131_072, particle_max_age_steps=64,
        dvm_target_spacing_chord=0.018, lesp_crit_override=flap.LESP_CRIT_STATIC_POLAR,
        wake_history_mode=flap.DEFAULT_WAKE_HISTORY_MODE, device="cuda:0",
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device="cuda:0")
    owner = stepper.initialize((kin.evaluate(0.0),))
    panel = float(kernel.mean_chordwise_panel(solver.surface.evaluate(None, None).rings))
    print(f"{arguments.case}: kernel constant {kernel._EPS_V:g}, trailing-edge wake "
          f"{'implicit' if native._TE_WAKE_V3 else 'legacy'}; panel {panel * 1e3:.3f} mm; "
          f"U dt {case.freestream_m_s * aero_dt * 1e3:.3f} mm", flush=True)
    print(f"{'step':>5} {'t*':>6} {'max|LESP|':>9} {'d(colloc)':>9} {'d(LE row)':>9} {'d(bound)':>9} "
          f"{'max|gamma|':>10} {'rows':>5} {'part':>6} {'Fx/N':>12}", flush=True)
    started = time.time()
    for step in range(1, arguments.steps + 1):
        frame = kin.evaluate(step * aero_dt)
        proposal = stepper.propose(owner, (frame,), aero_dt)
        state = proposal.trial_state
        geometry = solver.surface.evaluate(None, None)
        wake = state.wake_rings.reshape(-1, 4, 3)
        corners = wake.reshape(-1, 3)
        mids = 0.5 * (wake + torch.roll(wake, shifts=-1, dims=1)).reshape(-1, 3)
        wake_points = torch.cat((corners, mids), dim=0)
        bound_mids = 0.5 * (geometry.rings + torch.roll(geometry.rings, shifts=-1, dims=1)).reshape(-1, 3)
        record = (
            float(state.diagnostics[-1]["lesp_pre_max_abs"]) if state.diagnostics else float("nan"),
            closest(wake_points, geometry.collocation) / panel if corners.shape[0] else float("nan"),
            closest(wake_points, geometry.collocation[:ns]) / panel if corners.shape[0] else float("nan"),
            closest(corners, bound_mids) / panel if corners.shape[0] else float("nan"),
            float(state.gamma_bound.abs().max()),
            int(wake.shape[0]),
            int(state.particle_field.n),
            float(proposal.load.total_force[0]),
        )
        stepper.commit(owner, proposal)
        if step >= arguments.first and (step - arguments.first) % arguments.every == 0:
            print(f"{step:5d} {step * aero_dt * case.freestream_m_s / case.chord_m:6.2f} {record[0]:9.4f} "
                  f"{record[1]:9.4f} {record[2]:9.4f} {record[3]:9.4f} {record[4]:10.5f} {record[5]:5d} "
                  f"{record[6]:6d} {record[7]:12.2f}", flush=True)
        if not torch.isfinite(torch.tensor(record[0])):
            print("non-finite"); break
    print(f"done in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
