"""Per-step probe of the rigid plate with leading-edge shedding: what does the Hirato sheet do?

Same solver, grid and start-up ramp as diagnose_rigid_plate_coefficients.py (its `build`), no
flow sampling.  After every step it prints

  active   strips shedding at this step, and how many times the active set was iterated,
  lambda   largest and summed leading-edge strength of the nascent rings (m^2 / s),
  Gamma_TE summed circulation of the bound trailing-edge row, and of the newest wake row
           (the bound one plus lambda, what is shed),
  Cn       force coefficient of the step, along the wing normal, and the same from the newest
           wake row by Kutta-Joukowski,
  chain    rows in use summed over the strips, rows audited away at this step,
  lesp     largest attached suction parameter before shedding.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \\
        python3 /tmp/zcode_rigid_lev_probe.py --alpha 16 --steps 400 --every 5 [--from 200]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd()
for sub in ("src", "platform", "platform/warp_vpm"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--from", dest="first", type=int, default=1)
    arguments = parser.parse_args()
    from diagnose_roj_rigid_loads import build
    from fluxvortex.warp_fsi import kernels_ring_velocity as kernel
    from fluxvortex.warp_fsi import q16_flux_v5m_native as native

    speed, chord, span = 5.0, 0.0688, 0.1375
    nc, ns = 15, 30
    ds = span / ns
    q_s = 0.5 * 1.208 * speed * speed * chord * span
    solver, stepper, owner, kin = build(arguments.alpha, observed=False)
    solver.v_inf = solver.v_inf.to("cuda:0")
    dt = float(solver.settings.aerodynamic_dt)
    base_v_inf = solver.v_inf.clone()
    panel = float(kernel.mean_chordwise_panel(solver.surface.evaluate(None, None).rings))
    print(f"alpha {arguments.alpha:g} deg: constant {kernel._EPS_V:g}, leading-edge vortex "
          f"{native._LEV_CHOICE}, panel {panel * 1e3:.2f} mm, U dt {speed * dt * 1e3:.2f} mm, "
          f"crit {solver.settings.effective_lesp_crit:g}", flush=True)
    print(f"{'step':>5} {'t*':>6} {'act':>4} {'it':>3} {'lam_max':>8} {'lam_sum':>8} {'G_TE':>7} "
          f"{'shed':>7} {'Cn':>7} {'Cn_KJ':>7} {'rows':>5} {'cut':>4} {'lesp':>6}", flush=True)
    started = time.time()
    for step in range(1, arguments.steps + 1):
        factor = 1.0 if step >= 100 else 0.5 * (1.0 - math.cos(math.pi * step / 100))
        solver.v_inf = base_v_inf * factor
        object.__setattr__(solver.settings, "freestream", speed * factor)
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        state = proposal.trial_state
        diag = state.diagnostics[-1]
        record = diag.get("lev_hirato") or {}
        grid = state.gamma_bound.detach().reshape(nc, ns)
        wake = state.wake_gamma.detach().reshape(-1, ns)
        cn = float((proposal.load.total_force @ torch.tensor([0.0, 0.0, 1.0], device="cuda:0", dtype=torch.float64) / q_s).item())
        shed = float(wake[0].sum().item()) if wake.shape[0] else 0.0
        stepper.commit(owner, proposal)
        if step >= arguments.first and (step - arguments.first) % arguments.every == 0:
            g_te = float(grid[-1].sum().item())
            print(f"{step:5d} {step * speed * dt / chord:6.2f} {record.get('active_strips', 0):4d} "
                  f"{record.get('iterations', 0):3d} {record.get('strength_max_abs', 0.0):8.4f} "
                  f"{record.get('strength_sum', 0.0):+8.4f} {g_te:+7.4f} {shed:+7.4f} {cn:+7.4f} "
                  f"{2.0 * shed * ds / (speed * chord * span):+7.4f} {record.get('chain_rows', 0):5d} "
                  f"{diag.get('lev_truncate_ring_count', 0):4d} {diag['lesp_pre_max_abs']:6.3f}", flush=True)
    print(f"done in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
